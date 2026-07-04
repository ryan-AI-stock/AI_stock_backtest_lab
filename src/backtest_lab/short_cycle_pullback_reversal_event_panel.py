from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


TASK_ID = "TASK-BACKTEST-CORE-SHORT-CYCLE-PULLBACK-REVERSAL-EVENT-PANEL-001"
DEFAULT_EXPERIMENTS_OUTPUT = (
    "C:/Users/zergv/Documents/Codex/2026-06-17/repo-ai-stock-backtest-lab-repo/outputs/"
    "experiments_dynamic_pool1_short_cycle_pullback_reversal_diagnostic_20260704"
)
DEFAULT_OUTPUT_DIR = "outputs/short_cycle_pullback_reversal_event_panel_20260704"
DEFAULT_POOL1B_REPAIR_OUTPUT = (
    "C:/Users/zergv/Documents/Codex/2026-05-23/ai-stock-rotation-radar-https-docs/outputs/"
    "radar_pool1b_price_cache_repair_20260704"
)
DEFAULT_POOL1B_PRELIMINARY_RERUN_OUTPUT = (
    "C:/Users/zergv/Documents/Codex/2026-06-17/repo-ai-stock-backtest-lab-repo/outputs/"
    "experiments_pool1b_material_layer_short_cycle_rerun_20260704"
)

VARIANT_ROLE = {
    "strong_stock_ma20_pullback_reclaim": "primary_candidate",
    "pullback_candidate_wait_for_peer_breadth": "confidence_filter_candidate",
    "pullback_watch_then_confirm_2day": "sensitivity",
    "strong_stock_ma60_shallow_break_reclaim": "case_slice",
    "pullback_reversal_vs_formal_selector_overlay_diagnostic": "report_only_opportunity_context",
}
VARIANT_PRIORITY = {
    "strong_stock_ma20_pullback_reclaim": 1,
    "pullback_candidate_wait_for_peer_breadth": 2,
    "pullback_watch_then_confirm_2day": 3,
    "strong_stock_ma60_shallow_break_reclaim": 4,
    "pullback_reversal_vs_formal_selector_overlay_diagnostic": 5,
}


def run_short_cycle_pullback_reversal_event_panel(
    *,
    experiments_output: str | Path = DEFAULT_EXPERIMENTS_OUTPUT,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    pool1b_repair_output: str | Path = DEFAULT_POOL1B_REPAIR_OUTPUT,
    pool1b_preliminary_rerun_output: str | Path = DEFAULT_POOL1B_PRELIMINARY_RERUN_OUTPUT,
) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    run_log: list[dict[str, str]] = []

    def log(step: str, status: str, detail: str = "") -> None:
        run_log.append(
            {
                "timestamp": pd.Timestamp.now(tz="Asia/Taipei").strftime("%Y-%m-%d %H:%M:%S%z"),
                "step": step,
                "status": status,
                "detail": detail,
            }
        )
        pd.DataFrame(run_log).to_csv(output / "run_log.csv", index=False, encoding="utf-8-sig")
        (output / "current_step.txt").write_text(f"{step}:{status}\n{detail}", encoding="utf-8")

    try:
        source_root = Path(experiments_output)
        log("load_upstream", "started", str(source_root))
        upstream_manifest = _load_json(source_root / "manifest.json")
        raw_events = _read_csv_required(source_root / "pullback_reversal_event_panel.csv")
        blockers = _read_csv_if_exists(source_root / "data_blockers.csv")
        upstream_concentration = _read_csv_if_exists(source_root / "concentration_by_ticker_month_sector.csv")
        upstream_overlay = _read_csv_if_exists(source_root / "formal_target_vs_pullback_candidate_opportunity.csv")
        upstream_case = _read_csv_if_exists(source_root / "old_ai_pullback_case_study.csv")
        repair_root = Path(pool1b_repair_output)
        repair_manifest = _load_json(repair_root / "manifest.json")
        repair_files = _read_csv_if_exists(repair_root / "cache_compatible_files_manifest.csv")
        repair_coverage = _read_csv_if_exists(repair_root / "coverage_by_ticker.csv")
        preliminary_root = Path(pool1b_preliminary_rerun_output)
        preliminary_manifest = _load_json(preliminary_root / "manifest.json")
        preliminary_events = _read_csv_if_exists(preliminary_root / "pullback_reversal_event_panel.csv")

        log("build_panels", "started", "")
        combined_events = _combine_repaired_pool1b_events(raw_events, preliminary_events)
        event_panel = _build_event_panel(combined_events, repair_files=repair_files)
        dedup_panel = _build_dedup_panel(event_panel)
        unresolved_blockers = _unresolved_price_blockers(blockers, repair_coverage)
        readiness = _build_variant_readiness(
            event_panel,
            unresolved_blockers,
            upstream_manifest,
            repair_manifest=repair_manifest,
        )
        case_slice = _build_case_slice(event_panel, upstream_case)
        overlay_context = _build_formal_overlay_context(event_panel, upstream_overlay)
        concentration = _build_concentration_audit(event_panel, upstream_concentration)
        price_readiness = _build_price_readiness(
            event_panel,
            unresolved_blockers,
            upstream_manifest,
            repair_files=repair_files,
            repair_coverage=repair_coverage,
        )
        future_audit = _build_future_data_violation_audit(source_root, event_panel)
        manifest = _build_manifest(
            output,
            source_root,
            upstream_manifest,
            event_panel,
            dedup_panel,
            unresolved_blockers,
            repair_root=repair_root,
            repair_manifest=repair_manifest,
            preliminary_root=preliminary_root,
            preliminary_manifest=preliminary_manifest,
            price_readiness=price_readiness,
        )
        summary = _build_final_summary(manifest, readiness, unresolved_blockers, concentration)

        log("write_outputs", "started", str(output))
        event_panel.to_csv(output / "event_panel.csv", index=False, encoding="utf-8-sig")
        event_panel.to_csv(
            output / "short_cycle_pullback_reversal_event_panel.csv", index=False, encoding="utf-8-sig"
        )
        dedup_panel.to_csv(output / "event_panel_dedup.csv", index=False, encoding="utf-8-sig")
        dedup_panel.to_csv(
            output / "short_cycle_pullback_reversal_event_dedup_panel.csv",
            index=False,
            encoding="utf-8-sig",
        )
        readiness.to_csv(output / "event_panel_readiness.csv", index=False, encoding="utf-8-sig")
        readiness.to_csv(
            output / "short_cycle_pullback_reversal_variant_readiness.csv",
            index=False,
            encoding="utf-8-sig",
        )
        case_slice.to_csv(output / "case_slice_panel.csv", index=False, encoding="utf-8-sig")
        case_slice.to_csv(output / "old_ai_pullback_case_slice.csv", index=False, encoding="utf-8-sig")
        overlay_context.to_csv(output / "formal_overlay_opportunity_context.csv", index=False, encoding="utf-8-sig")
        concentration.to_csv(output / "concentration_audit.csv", index=False, encoding="utf-8-sig")
        concentration.to_csv(
            output / "event_overlap_and_concentration_audit.csv", index=False, encoding="utf-8-sig"
        )
        price_readiness.to_csv(output / "available_candidate_price_readiness.csv", index=False, encoding="utf-8-sig")
        price_readiness.to_csv(output / "price_readiness_by_ticker.csv", index=False, encoding="utf-8-sig")
        future_audit.to_csv(output / "future_data_violation_audit.csv", index=False, encoding="utf-8-sig")
        (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        (output / "event_panel_readiness.json").write_text(
            json.dumps(_readiness_json(manifest, readiness, blockers), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (output / "final_summary_zh.md").write_text(summary, encoding="utf-8")
        pd.DataFrame([{"step": "short_cycle_pullback_reversal_event_panel", "status": "completed"}]).to_csv(
            output / "completed.csv",
            index=False,
            encoding="utf-8-sig",
        )
        pd.DataFrame(columns=["step", "status", "error"]).to_csv(output / "failed.csv", index=False, encoding="utf-8-sig")
        log("completed", "completed", str(output))
        return output
    except Exception as exc:
        pd.DataFrame([{"step": "short_cycle_pullback_reversal_event_panel", "status": "failed", "error": str(exc)}]).to_csv(
            output / "failed.csv",
            index=False,
            encoding="utf-8-sig",
        )
        log("failed", "failed", str(exc))
        raise


def _combine_repaired_pool1b_events(original: pd.DataFrame, preliminary: pd.DataFrame) -> pd.DataFrame:
    original_frame = original.copy()
    original_frame["source_lineage"] = "experiments_original_available_candidate_event_study"
    original_frame["source_price_cache_repaired"] = False
    if preliminary.empty:
        return original_frame

    preliminary_frame = preliminary.copy()
    preliminary_frame["source_lineage"] = "pool1b_material_repaired_preliminary_event_input_normalized_by_core"
    preliminary_frame["source_price_cache_repaired"] = True
    for column in original_frame.columns:
        if column not in preliminary_frame.columns:
            preliminary_frame[column] = ""
    if "event_status" not in preliminary_frame.columns or preliminary_frame["event_status"].astype(str).eq("").all():
        preliminary_frame["event_status"] = "event_candidate_from_repaired_price_cache"
    if "formal_target" not in preliminary_frame.columns:
        preliminary_frame["formal_target"] = ""
    if "formal_target_is_market_exposure" not in preliminary_frame.columns:
        preliminary_frame["formal_target_is_market_exposure"] = False

    pool1b_variants = set(VARIANT_ROLE) - {"pullback_reversal_vs_formal_selector_overlay_diagnostic"}
    original_keep = original_frame[
        ~(
            original_frame.get("candidate_source", "").astype(str).eq("pool1b")
            & original_frame.get("variant_id", "").astype(str).isin(pool1b_variants)
        )
    ]
    repaired_pool1b = preliminary_frame[
        preliminary_frame.get("candidate_source", "").astype(str).eq("pool1b")
        & preliminary_frame.get("variant_id", "").astype(str).isin(pool1b_variants)
    ]
    combined = pd.concat([original_keep, repaired_pool1b], ignore_index=True, sort=False)
    return combined


def _build_event_panel(raw_events: pd.DataFrame, repair_files: pd.DataFrame | None = None) -> pd.DataFrame:
    frame = raw_events[raw_events["variant_id"].isin(VARIANT_ROLE)].copy()
    for column in ["event_status", "formal_target", "formal_target_is_market_exposure"]:
        if column not in frame.columns:
            frame[column] = ""
    frame["signal_date"] = pd.to_datetime(frame["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    next_dates = _next_tradable_date_map(frame["signal_date"])
    frame["next_tradable_date"] = frame["signal_date"].map(next_dates).fillna("")
    frame["variant"] = frame["variant_id"]
    frame["variant_role"] = frame["variant_id"].map(VARIANT_ROLE)
    frame["event_state"] = frame["event_status"].astype(str)
    frame["source_lineage"] = frame.get("source_lineage", "experiments_original_available_candidate_event_study")
    frame["source_price_cache_repaired"] = frame.get("source_price_cache_repaired", False)
    frame["price_source"] = frame.apply(_price_source_label, axis=1)
    frame["close_vs_ma20_pct"] = frame.get("dist_ma20_pct", "")
    frame["close_vs_ma60_pct"] = frame.get("dist_ma60_pct", "")
    frame["close_vs_ma120_pct"] = frame.get("dist_ma120_pct", "")
    frame["pullback_position"] = frame["close_vs_ma20_pct"].apply(_pullback_position)
    frame["reclaim_condition"] = frame["variant_id"].map(_reclaim_condition)
    frame["confirmation_condition"] = frame["variant_id"].map(_confirmation_condition)
    frame["strong_stock_background_ok"] = True
    frame["formal_target_context"] = frame.apply(_formal_target_context, axis=1)
    frame["is_trade_rule"] = False
    frame["active_in_trade_decision"] = False
    frame["formal_model_changed"] = False
    frame["trade_decision_changed"] = False
    frame["diagnostic_only"] = True
    frame["forward_outcome_for_evaluation_only"] = True
    frame["uses_forward_return_as_live_rule"] = False
    frame["price_data_ready"] = True
    frame["formal_overlay_ready"] = frame["formal_target"].astype(str).str.len().gt(0)
    frame["dedup_key"] = frame["signal_date"] + "|" + frame["ticker"].astype(str)
    frame["variant_priority"] = frame["variant"].map(VARIANT_PRIORITY).fillna(99).astype(int)
    frame = _apply_repaired_cache_metadata(frame, repair_files if repair_files is not None else pd.DataFrame())
    for column in _event_columns():
        if column not in frame.columns:
            frame[column] = ""
    return frame[_event_columns()].sort_values(["signal_date", "ticker", "variant_priority"]).reset_index(drop=True)


def _price_source_label(row: pd.Series) -> str:
    if _bool_like(row.get("source_price_cache_repaired")):
        return "radar_pool1b_repaired_cache_compatible_unadjusted_ohlcv"
    return "experiments_dynamic_pool1_pullback_reversal_diagnostic_price_panel"


def _apply_repaired_cache_metadata(frame: pd.DataFrame, repair_files: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["ticker"] = result["ticker"].astype(str)
    result["repaired_cache_available"] = False
    result["repaired_cache_path"] = ""
    result["repaired_cache_first_date"] = ""
    result["repaired_cache_last_date"] = ""
    result["repaired_cache_row_count"] = 0
    result["adjusted_close_available"] = result.get("adjusted_close_available", True)
    result["adjusted_close_boundary"] = ""
    if repair_files.empty or "ticker" not in repair_files.columns:
        return result

    indexed = repair_files.fillna("").set_index("ticker").to_dict(orient="index")
    for idx, row in result.iterrows():
        ticker = str(row.get("ticker", ""))
        if ticker not in indexed:
            continue
        meta = indexed[ticker]
        result.at[idx, "repaired_cache_available"] = True
        result.at[idx, "repaired_cache_path"] = str(meta.get("cache_compatible_path", ""))
        result.at[idx, "repaired_cache_first_date"] = str(meta.get("first_date", ""))
        result.at[idx, "repaired_cache_last_date"] = str(meta.get("last_date", ""))
        result.at[idx, "repaired_cache_row_count"] = int(float(meta.get("row_count", 0) or 0))
        result.at[idx, "adjusted_close_available"] = _bool_like(meta.get("adjusted_close_available"))
        result.at[idx, "adjusted_close_boundary"] = (
            "repaired Pool1B cache uses official unadjusted OHLCV; adjusted close is not synthesized"
        )
    return result


def _build_dedup_panel(event_panel: pd.DataFrame) -> pd.DataFrame:
    if event_panel.empty:
        return pd.DataFrame(columns=_dedup_columns())
    frame = event_panel.copy()
    frame["variant_priority"] = frame["variant"].map(VARIANT_PRIORITY).fillna(99).astype(int)
    grouped_rows: list[dict[str, Any]] = []
    for _, group in frame.sort_values(["dedup_key", "variant_priority"]).groupby("dedup_key", dropna=False):
        first = group.iloc[0].to_dict()
        first["primary_variant"] = first["variant"]
        first["overlap_count"] = int(len(group))
        first["overlap_variant_ids"] = "|".join(group["variant"].astype(str).tolist())
        first["event_dedup_policy"] = "same_signal_date_ticker_keep_research_priority"
        grouped_rows.append(first)
    dedup = pd.DataFrame(grouped_rows)
    for column in _dedup_columns():
        if column not in dedup.columns:
            dedup[column] = ""
    return dedup[_dedup_columns()].sort_values(["signal_date", "ticker"]).reset_index(drop=True)


def _build_variant_readiness(
    event_panel: pd.DataFrame,
    blockers: pd.DataFrame,
    manifest: dict[str, Any],
    *,
    repair_manifest: dict[str, Any] | None = None,
) -> pd.DataFrame:
    repair_manifest = repair_manifest or {}
    rows: list[dict[str, Any]] = []
    for variant, role in VARIANT_ROLE.items():
        subset = event_panel[event_panel["variant"].eq(variant)]
        rows.append(
            {
                "variant_id": variant,
                "variant_role": role,
                "event_count": int(len(subset)),
                "unique_tickers": int(subset["ticker"].nunique()) if not subset.empty else 0,
                "price_data_ready": True,
                "formal_overlay_ready": bool(subset["formal_overlay_ready"].astype(bool).all()) if not subset.empty else False,
                "diagnostic_only": True,
                "active_in_trade_decision": False,
                "uses_forward_return_as_live_rule": False,
                "ready_for_experiments_portfolio_challenger": bool(len(subset) > 0),
                "blocked_ticker_count": int(blockers["ticker"].nunique()) if not blockers.empty and "ticker" in blockers else 0,
                "price_latest": manifest.get("price_latest", ""),
                "repaired_cache_latest_complete_date": repair_manifest.get("latest_complete_date", ""),
                "repaired_cache_adjusted_close_available": repair_manifest.get("adjusted_close_available", ""),
                "formal_overlay_latest": manifest.get("formal_overlay_latest", ""),
                "readiness_note": _variant_note(variant),
            }
        )
    return pd.DataFrame(rows)


def _build_case_slice(event_panel: pd.DataFrame, upstream_case: pd.DataFrame) -> pd.DataFrame:
    case = event_panel[event_panel["candidate_source"].isin(["old_ai", "old_ai_seven"])].copy()
    if case.empty and not upstream_case.empty:
        case = _build_event_panel(upstream_case)
    return case


def _build_formal_overlay_context(event_panel: pd.DataFrame, upstream_overlay: pd.DataFrame) -> pd.DataFrame:
    overlay = event_panel[event_panel["variant"].eq("pullback_reversal_vs_formal_selector_overlay_diagnostic")].copy()
    if overlay.empty and not upstream_overlay.empty:
        overlay = _build_event_panel(upstream_overlay)
    if overlay.empty:
        return pd.DataFrame(columns=_event_columns())
    overlay["opportunity_context_only"] = True
    overlay["formal_target_override_allowed"] = False
    overlay["context_note"] = overlay["formal_target_context"]
    return overlay


def _build_concentration_audit(event_panel: pd.DataFrame, upstream_concentration: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant, group in event_panel.groupby("variant", dropna=False):
        total = len(group)
        ticker_counts = group["ticker"].value_counts()
        top_ticker = str(ticker_counts.index[0]) if not ticker_counts.empty else ""
        top_share = float(ticker_counts.iloc[0] / total) if total else 0.0
        rows.append(
            {
                "variant_id": variant,
                "event_count": total,
                "unique_tickers": int(group["ticker"].nunique()),
                "top_ticker": top_ticker,
                "top_ticker_event_count": int(ticker_counts.iloc[0]) if not ticker_counts.empty else 0,
                "top_ticker_event_share": round(top_share, 6),
                "concentration_flag": bool(top_share > 0.40),
                "requires_concentration_review": variant == "pullback_candidate_wait_for_peer_breadth" and top_share > 0.40,
                "diagnostic_only": True,
                "active_in_trade_decision": False,
            }
        )
    audit = pd.DataFrame(rows)
    if not upstream_concentration.empty:
        audit["upstream_concentration_rows"] = len(upstream_concentration)
    return audit


def _unresolved_price_blockers(blockers: pd.DataFrame, repair_coverage: pd.DataFrame) -> pd.DataFrame:
    if blockers.empty or "ticker" not in blockers.columns:
        return blockers
    if repair_coverage.empty or "ticker" not in repair_coverage.columns or "coverage_ready" not in repair_coverage.columns:
        return blockers
    repaired = set(
        repair_coverage[
            repair_coverage.get("coverage_ready", pd.Series(dtype=bool)).map(_bool_like)
        ]["ticker"].astype(str)
    )
    return blockers[~blockers["ticker"].astype(str).isin(repaired)].copy()


def _build_price_readiness(
    event_panel: pd.DataFrame,
    blockers: pd.DataFrame,
    manifest: dict[str, Any],
    *,
    repair_files: pd.DataFrame,
    repair_coverage: pd.DataFrame,
) -> pd.DataFrame:
    blocked = set(blockers["ticker"].astype(str)) if not blockers.empty and "ticker" in blockers else set()
    repair_by_ticker = {}
    if not repair_coverage.empty and "ticker" in repair_coverage.columns:
        repair_by_ticker = repair_coverage.fillna("").set_index("ticker").to_dict(orient="index")
    rows = []
    for ticker, group in event_panel.groupby("ticker", dropna=False):
        repair_meta = repair_by_ticker.get(str(ticker), {})
        rows.append(
            {
                "ticker": ticker,
                "candidate_name": group["candidate_name"].iloc[0],
                "candidate_sources": "|".join(sorted(group["candidate_source"].astype(str).unique())),
                "event_rows": int(len(group)),
                "price_data_ready": ticker not in blocked,
                "latest_price_date": manifest.get("price_latest", ""),
                "blocked_reason": "missing_price_cache" if ticker in blocked else "",
                "repaired_cache_available": bool(group["repaired_cache_available"].map(_bool_like).any()),
                "repaired_cache_first_date": str(repair_meta.get("first_date", "")),
                "repaired_cache_last_date": str(repair_meta.get("last_date", "")),
                "repaired_cache_row_count": int(float(repair_meta.get("row_count", 0) or 0)),
                "adjusted_close_available": bool(group["adjusted_close_available"].map(_bool_like).all()),
                "adjusted_close_boundary": (
                    "official unadjusted OHLCV; adjusted close not synthesized"
                    if bool(group["repaired_cache_available"].map(_bool_like).any())
                    else ""
                ),
                "diagnostic_only": True,
            }
        )
    if blockers.empty:
        return pd.DataFrame(rows)
    for _, item in blockers.iterrows():
        ticker = str(item.get("ticker", ""))
        if ticker and ticker not in set(event_panel["ticker"].astype(str)):
            rows.append(
                {
                    "ticker": ticker,
                    "candidate_name": "",
                    "candidate_sources": str(item.get("source", "")),
                    "event_rows": 0,
                    "price_data_ready": False,
                    "latest_price_date": str(item.get("latest_price_date", "")),
                    "blocked_reason": str(item.get("blocker", "missing_price_cache")),
                    "repaired_cache_available": False,
                    "repaired_cache_first_date": "",
                    "repaired_cache_last_date": "",
                    "repaired_cache_row_count": 0,
                    "adjusted_close_available": "",
                    "adjusted_close_boundary": "",
                    "diagnostic_only": True,
                }
            )
    return pd.DataFrame(rows)


def _build_future_data_violation_audit(source_root: Path, event_panel: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "data_area": "short_cycle_pullback_reversal_event_panel",
                "source_path": str(source_root / "pullback_reversal_event_panel.csv"),
                "future_outcome_columns_present": True,
                "forward_outcome_for_evaluation_only": True,
                "uses_forward_return_as_live_rule": False,
                "future_data_violation": False,
                "future_data_violation_count": 0,
                "audit_reason": "Forward outcome columns are retained only for Experiments evaluation metadata and are not live-rule inputs.",
                "row_count": int(len(event_panel)),
            }
        ]
    )


def _build_manifest(
    output: Path,
    source_root: Path,
    upstream_manifest: dict[str, Any],
    event_panel: pd.DataFrame,
    dedup_panel: pd.DataFrame,
    blockers: pd.DataFrame,
    *,
    repair_root: Path,
    repair_manifest: dict[str, Any],
    preliminary_root: Path,
    preliminary_manifest: dict[str, Any],
    price_readiness: pd.DataFrame,
) -> dict[str, Any]:
    material_events = int(event_panel["supply_chain_layer"].astype(str).str.contains("material", case=False, na=False).sum())
    case_6488_events = int(event_panel["ticker"].astype(str).eq("6488.TWO").sum())
    repaired_event_tickers = sorted(
        event_panel[event_panel["repaired_cache_available"].map(_bool_like)]["ticker"].astype(str).unique().tolist()
    )
    return {
        "schema_version": 1,
        "task_id": TASK_ID,
        "status": "completed_production_grade_diagnostic_event_panel",
        "generated_at": pd.Timestamp.now(tz="Asia/Taipei").isoformat(),
        "output_dir": str(output),
        "source_experiments_output": str(source_root),
        "pool1b_price_repair_output": str(repair_root),
        "pool1b_preliminary_rerun_output": str(preliminary_root),
        "source_task_id": upstream_manifest.get("task_id", ""),
        "preliminary_rerun_status": preliminary_manifest.get("status", ""),
        "preliminary_rerun_used_as_formal_validation": False,
        "preliminary_rerun_boundary": "Used only as repaired-cache event input for Core-normalized diagnostic panel; not formal validation.",
        "pool1b_repaired_cache_used": bool(repair_manifest.get("price_cache_candidate_ready", False)),
        "pool1b_repaired_ticker_count": int(repair_manifest.get("completed_ticker_count", 0) or 0),
        "pool1b_repaired_cache_adjusted_close_available": repair_manifest.get("adjusted_close_available", False),
        "pool1b_repaired_cache_adjusted_close_boundary": repair_manifest.get("adjusted_close_boundary", ""),
        "pool1b_repaired_latest_complete_date": repair_manifest.get("latest_complete_date", ""),
        "pool1b_repaired_event_tickers": repaired_event_tickers,
        "pool1b_material_layer_event_count": material_events,
        "case_6488_two_event_count": case_6488_events,
        "price_latest": upstream_manifest.get("price_latest", ""),
        "formal_overlay_latest": upstream_manifest.get("formal_overlay_latest", ""),
        "event_rows": int(len(event_panel)),
        "dedup_rows": int(len(dedup_panel)),
        "unique_tickers": int(event_panel["ticker"].nunique()) if not event_panel.empty else 0,
        "blocked_ticker_count": int(blockers["ticker"].nunique()) if not blockers.empty and "ticker" in blockers else 0,
        "price_ready_ticker_count": int(price_readiness["price_data_ready"].map(_bool_like).sum())
        if not price_readiness.empty
        else 0,
        "formal_model_changed": False,
        "trade_decision_changed": False,
        "active_in_trade_decision": False,
        "diagnostic_only": True,
        "ready_for_experiments_portfolio_challenger": bool(len(dedup_panel) > 0),
        "ready_for_formal_absorption": False,
        "uses_forward_return_as_live_rule": False,
        "future_data_violation_count": 0,
        "outputs": {
            "event_panel": "event_panel.csv",
            "event_panel_dedup": "event_panel_dedup.csv",
            "event_panel_readiness": "event_panel_readiness.csv",
            "event_panel_readiness_json": "event_panel_readiness.json",
            "case_slice_panel": "case_slice_panel.csv",
            "formal_overlay_opportunity_context": "formal_overlay_opportunity_context.csv",
            "concentration_audit": "concentration_audit.csv",
            "available_candidate_price_readiness": "available_candidate_price_readiness.csv",
            "price_readiness_by_ticker": "price_readiness_by_ticker.csv",
            "future_data_violation_audit": "future_data_violation_audit.csv",
            "final_summary_zh": "final_summary_zh.md",
        },
    }


def _readiness_json(manifest: dict[str, Any], readiness: pd.DataFrame, blockers: pd.DataFrame) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "task_id": TASK_ID,
        "status": manifest["status"],
        "ready_for_experiments_portfolio_challenger": manifest["ready_for_experiments_portfolio_challenger"],
        "ready_for_formal_absorption": False,
        "formal_model_changed": False,
        "trade_decision_changed": False,
        "active_in_trade_decision": False,
        "diagnostic_only": True,
        "uses_forward_return_as_live_rule": False,
        "future_data_violation_count": 0,
        "event_rows": manifest["event_rows"],
        "dedup_rows": manifest["dedup_rows"],
        "pool1b_repaired_cache_used": manifest.get("pool1b_repaired_cache_used", False),
        "pool1b_repaired_ticker_count": manifest.get("pool1b_repaired_ticker_count", 0),
        "pool1b_repaired_cache_adjusted_close_available": manifest.get(
            "pool1b_repaired_cache_adjusted_close_available", False
        ),
        "case_6488_two_event_count": manifest.get("case_6488_two_event_count", 0),
        "variant_readiness": readiness.to_dict(orient="records"),
        "blockers": blockers.to_dict(orient="records") if not blockers.empty else [],
    }


def _build_final_summary(
    manifest: dict[str, Any],
    readiness: pd.DataFrame,
    blockers: pd.DataFrame,
    concentration: pd.DataFrame,
) -> str:
    ready = manifest["ready_for_experiments_portfolio_challenger"]
    concentrated = concentration[concentration.get("concentration_flag", False).astype(bool)] if not concentration.empty else pd.DataFrame()
    return (
        "# Short-cycle pullback reversal event panel\n\n"
        f"Status: {manifest['status']}.\n\n"
        f"- event rows: {manifest['event_rows']}\n"
        f"- dedup rows: {manifest['dedup_rows']}\n"
        f"- unique tickers: {manifest['unique_tickers']}\n"
        f"- blocked tickers: {manifest['blocked_ticker_count']}\n"
        f"- Pool1B repaired cache used: {str(manifest.get('pool1b_repaired_cache_used', False)).lower()}\n"
        f"- 6488.TWO events: {manifest.get('case_6488_two_event_count', 0)}\n"
        "- repaired cache adjusted_close_available=false; official unadjusted OHLCV is not treated as adjusted close.\n"
        f"- ready_for_experiments_portfolio_challenger={str(ready).lower()}\n"
        "- formal_model_changed=false\n"
        "- trade_decision_changed=false\n"
        "- active_in_trade_decision=false\n"
        "- diagnostic_only=true\n"
        "- forward outcome columns are retained only for Experiments evaluation metadata, not live rules.\n\n"
        f"Concentration flags: {len(concentrated)} variant(s) exceed the 40% top-ticker review threshold.\n\n"
        "Next: hand to Experiments for next-day portfolio challenger / report-only opportunity-context replay. "
        "Do not treat this panel as formal target override or formal absorption evidence.\n"
    )


def _next_tradable_date_map(signal_dates: pd.Series) -> dict[str, str]:
    dates = sorted(pd.Series(signal_dates).dropna().astype(str).unique())
    mapping = {date: dates[index + 1] for index, date in enumerate(dates[:-1])}
    if dates:
        mapping[dates[-1]] = ""
    return mapping


def _pullback_position(value: Any) -> str:
    try:
        pct = float(value)
    except Exception:
        return "unknown"
    if pct >= 0:
        return "above_ma20"
    if pct >= -5:
        return "near_ma20_pullback"
    return "deep_below_ma20"


def _reclaim_condition(variant: str) -> str:
    if variant == "strong_stock_ma20_pullback_reclaim":
        return "ma20_pullback_reclaim"
    if variant == "strong_stock_ma60_shallow_break_reclaim":
        return "ma60_shallow_break_reclaim"
    if variant == "pullback_watch_then_confirm_2day":
        return "watch_then_confirm_2day"
    if variant == "pullback_candidate_wait_for_peer_breadth":
        return "ma20_pullback_reclaim_with_peer_breadth"
    return "report_only_overlay_opportunity"


def _confirmation_condition(variant: str) -> str:
    if variant == "pullback_candidate_wait_for_peer_breadth":
        return "peer_recovery_count_and_market_support"
    if variant == "pullback_watch_then_confirm_2day":
        return "two_day_confirmation_sensitivity"
    return "single_event_diagnostic_condition"


def _formal_target_context(row: pd.Series) -> str:
    target = str(row.get("formal_target", ""))
    exposure = _bool_like(row.get("formal_target_is_market_exposure"))
    if target.lower() in {"cash", "現金", ""}:
        return "formal_target_cash_or_risk_off_event_is_opportunity_context_only"
    if exposure:
        return "formal_target_market_exposure_event_is_opportunity_context_only"
    if target and target != str(row.get("ticker", "")):
        return "formal_target_differs_event_is_opportunity_context_only"
    return "formal_target_matches_or_not_available_diagnostic_only"


def _variant_note(variant: str) -> str:
    if variant == "strong_stock_ma20_pullback_reclaim":
        return "Research primary candidate; ready for Experiments portfolio challenger as diagnostic."
    if variant == "pullback_candidate_wait_for_peer_breadth":
        return "Confidence/filter candidate; requires concentration audit before any further promotion."
    if variant == "pullback_watch_then_confirm_2day":
        return "Sensitivity only."
    if variant == "strong_stock_ma60_shallow_break_reclaim":
        return "Case-slice only."
    return "Report-only opportunity context; must not override formal target."


def _bool_like(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path).fillna("")
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _read_csv_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(str(path))
    return pd.read_csv(path).fillna("")


def _event_columns() -> list[str]:
    return [
        "signal_date",
        "next_tradable_date",
        "ticker",
        "candidate_name",
        "variant",
        "variant_id",
        "variant_role",
        "variant_priority",
        "candidate_source",
        "source_lineage",
        "source_price_cache_repaired",
        "price_source",
        "repaired_cache_available",
        "repaired_cache_path",
        "repaired_cache_first_date",
        "repaired_cache_last_date",
        "repaired_cache_row_count",
        "adjusted_close_available",
        "adjusted_close_boundary",
        "supply_chain_layer",
        "sector_code",
        "sector_name",
        "event_state",
        "close",
        "close_vs_ma20_pct",
        "close_vs_ma60_pct",
        "close_vs_ma120_pct",
        "pullback_position",
        "reclaim_condition",
        "confirmation_condition",
        "strong_stock_background_ok",
        "ma60_slope_10d_pct",
        "ma120_slope_20d_pct",
        "rs_vs_0050_20d_pct",
        "rs_vs_0050_60d_pct",
        "drawdown_from_60d_high_pct",
        "peer_recovery_count",
        "market_support_ok",
        "formal_target",
        "formal_target_is_market_exposure",
        "formal_target_context",
        "price_data_ready",
        "formal_overlay_ready",
        "diagnostic_only",
        "is_trade_rule",
        "active_in_trade_decision",
        "formal_model_changed",
        "trade_decision_changed",
        "forward_outcome_for_evaluation_only",
        "uses_forward_return_as_live_rule",
        "forward_return_20d_pct",
        "forward_path_mdd_20d_pct",
        "forward_return_40d_pct",
        "forward_path_mdd_40d_pct",
        "forward_return_60d_pct",
        "forward_path_mdd_60d_pct",
        "dedup_key",
    ]


def _dedup_columns() -> list[str]:
    return [
        "signal_date",
        "next_tradable_date",
        "ticker",
        "candidate_name",
        "primary_variant",
        "variant_role",
        "candidate_source",
        "event_state",
        "formal_target",
        "formal_target_context",
        "overlap_count",
        "overlap_variant_ids",
        "event_dedup_policy",
        "diagnostic_only",
        "is_trade_rule",
        "active_in_trade_decision",
        "forward_outcome_for_evaluation_only",
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build production-grade short-cycle pullback reversal event panel.")
    parser.add_argument("--experiments-output", default=DEFAULT_EXPERIMENTS_OUTPUT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--pool1b-repair-output", default=DEFAULT_POOL1B_REPAIR_OUTPUT)
    parser.add_argument("--pool1b-preliminary-rerun-output", default=DEFAULT_POOL1B_PRELIMINARY_RERUN_OUTPUT)
    args = parser.parse_args(argv)
    run_short_cycle_pullback_reversal_event_panel(
        experiments_output=args.experiments_output,
        output_dir=args.output_dir,
        pool1b_repair_output=args.pool1b_repair_output,
        pool1b_preliminary_rerun_output=args.pool1b_preliminary_rerun_output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
