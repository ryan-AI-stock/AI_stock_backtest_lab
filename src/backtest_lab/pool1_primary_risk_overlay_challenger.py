from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from backtest_lab.three_pool_vs_pool1_comparison_panels import (
    DEFAULT_FORMAL_REPLAY_DIR,
    DEFAULT_PRICE_CACHE_DIR,
    INITIAL_CASH,
    VARIANT_BASELINE,
    VARIANT_POOL1,
    _copy_variant_daily,
    _copy_variant_trades,
    _load_prices,
    _needed_tickers,
    _normalize_baseline_daily,
    _period_performance,
    _pool_dominance_summary,
    _same_date_range,
    _simulate_variant,
    _target_stability,
    _text,
    _trade_ledger_from_daily,
    _validate_baseline_daily,
    _validate_decision,
)


DEFAULT_OUTPUT_DIR = "outputs/pool1_primary_risk_overlay_challenger_panels_20260626"
DEFAULT_TARGET_DROP_SOURCE = "outputs/three_pool_vs_pool1_comparison_panels_20260626/target_drop_from_top3_diagnostics.csv"
VARIANT_POOL1_PRIMARY = "pool1_primary_no_overlay"
VARIANT_POOL2_VETO = "pool1_primary_pool2_risk_veto"
VARIANT_POOL2_WARNING = "pool1_primary_pool2_disagreement_warning_only"
VARIANT_POOL23_VETO = "pool1_primary_pool2_pool3_risk_veto"
VARIANT_POOL23_REPORT = "pool1_primary_pool2_pool3_report_only_overlay"
VARIANTS = (
    VARIANT_BASELINE,
    VARIANT_POOL1,
    VARIANT_POOL1_PRIMARY,
    VARIANT_POOL2_VETO,
    VARIANT_POOL2_WARNING,
    VARIANT_POOL23_VETO,
    VARIANT_POOL23_REPORT,
)


def run_pool1_primary_risk_overlay_challenger(
    *,
    formal_replay_dir: str | Path = DEFAULT_FORMAL_REPLAY_DIR,
    price_cache_dir: str | Path = DEFAULT_PRICE_CACHE_DIR,
    target_drop_source_path: str | Path = DEFAULT_TARGET_DROP_SOURCE,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    initial_cash: float = INITIAL_CASH,
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
        (output / "current_step.txt").write_text(step, encoding="utf-8")

    try:
        formal_dir = Path(formal_replay_dir)
        log("load_inputs", "started", str(formal_dir))
        decision = pd.read_csv(formal_dir / "formal_three_pool_decision_panel.csv").fillna("")
        baseline_daily_source = pd.read_csv(formal_dir / "baseline_three_pool_formal_daily_equity.csv").fillna("")
        target_drop_source = _read_optional_csv(target_drop_source_path)
        _validate_decision(decision)
        _validate_baseline_daily(baseline_daily_source)
        prices = _load_prices(_needed_tickers(decision), Path(price_cache_dir))

        log("build_variant_targets", "started", "")
        variant_decisions = _build_variant_decisions(decision)
        veto_events = _build_veto_events(variant_decisions)
        warning_events = _build_warning_events(variant_decisions)

        log("simulate_variants", "started", "")
        baseline_daily = _normalize_baseline_daily(baseline_daily_source, VARIANT_BASELINE)
        baseline_trades = _trade_ledger_from_daily(baseline_daily, VARIANT_BASELINE)
        daily_frames = [baseline_daily]
        trade_frames = [baseline_trades]
        for variant in (
            VARIANT_POOL1,
            VARIANT_POOL1_PRIMARY,
            VARIANT_POOL2_VETO,
            VARIANT_POOL2_WARNING,
            VARIANT_POOL23_VETO,
            VARIANT_POOL23_REPORT,
        ):
            if variant in {VARIANT_POOL2_WARNING, VARIANT_POOL23_REPORT}:
                source_variant = VARIANT_POOL1_PRIMARY
                source_daily = next(frame for frame in daily_frames if str(frame["variant"].iloc[0]) == source_variant)
                source_trades = next(frame for frame in trade_frames if not frame.empty and str(frame["variant"].iloc[0]) == source_variant)
                daily_frames.append(_copy_variant_daily(source_daily, variant))
                trade_frames.append(_copy_variant_trades(source_trades, variant))
                continue
            adjusted = variant_decisions[variant]
            daily, trades = _simulate_variant(
                adjusted,
                prices,
                winner_column="variant_target",
                variant=variant,
                initial_cash=initial_cash,
            )
            daily_frames.append(daily)
            trade_frames.append(trades)

        daily_equity = pd.concat(daily_frames, ignore_index=True)
        trade_ledger = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()
        daily_target = _build_daily_target_panel(variant_decisions, daily_equity)
        target_stability = _target_stability(daily_target)
        target_drop = _target_drop_from_top3_overlay(daily_target, target_drop_source)
        period_performance = _period_performance(daily_equity)
        cost_turnover = _cost_turnover_summary(daily_equity)
        concentration = _concentration_summary(daily_equity)
        scorecard = _risk_adjusted_scorecard(period_performance, cost_turnover, concentration)
        risk_overlay = _risk_overlay_event_panel(variant_decisions)
        entry_without_exit = _entry_without_exit_panel(daily_target)

        log("write_outputs", "started", "")
        period_performance.to_csv(output / "period_performance_by_variant.csv", index=False, encoding="utf-8-sig")
        daily_equity.to_csv(output / "daily_equity_by_variant.csv", index=False, encoding="utf-8-sig")
        daily_target.to_csv(output / "daily_target_by_variant.csv", index=False, encoding="utf-8-sig")
        trade_ledger.to_csv(output / "trade_ledger_by_variant.csv", index=False, encoding="utf-8-sig")
        risk_overlay.to_csv(output / "risk_overlay_event_panel.csv", index=False, encoding="utf-8-sig")
        veto_events.to_csv(output / "veto_event_panel.csv", index=False, encoding="utf-8-sig")
        warning_events.to_csv(output / "warning_only_event_panel.csv", index=False, encoding="utf-8-sig")
        entry_without_exit.to_csv(output / "entry_without_exit_confirmation_panel.csv", index=False, encoding="utf-8-sig")
        target_stability.to_csv(output / "target_stability_panel.csv", index=False, encoding="utf-8-sig")
        target_drop.to_csv(output / "target_drop_from_top3_diagnostics.csv", index=False, encoding="utf-8-sig")
        cost_turnover.to_csv(output / "cost_turnover_summary.csv", index=False, encoding="utf-8-sig")
        concentration.to_csv(output / "concentration_summary.csv", index=False, encoding="utf-8-sig")
        scorecard.to_csv(output / "risk_adjusted_scorecard.csv", index=False, encoding="utf-8-sig")
        _pool_dominance_summary(decision).to_csv(output / "pool_dominance_summary.csv", index=False, encoding="utf-8-sig")
        (output / "pool1_primary_challenger_summary_zh.md").write_text(
            _summary_markdown(period_performance),
            encoding="utf-8",
        )
        manifest = {
            "schema_version": 1,
            "task_id": "TASK-BACKTEST-CORE-POOL1-PRIMARY-RISK-OVERLAY-CHALLENGER-PANELS-001",
            "model": "pool1_primary_risk_overlay_challenger",
            "status": "completed",
            "formal_replay_dir": str(formal_dir),
            "price_cache_dir": str(price_cache_dir),
            "target_drop_source_path": str(target_drop_source_path),
            "initial_cash": initial_cash,
            "start_date": str(baseline_daily_source["date"].iloc[0]),
            "latest_complete_common_date": str(baseline_daily_source["date"].iloc[-1]),
            "formal_model_changed": False,
            "trade_decision_changed": False,
            "pool3_shadow_used_as_formal": False,
            "report_only_labels_used_in_performance": False,
            "rr_partial_switch_used_in_performance": False,
            "valuation_used": False,
            "h3_used": False,
            "same_date_range_for_variants": _same_date_range(daily_equity),
            "same_cost_model_for_variants": True,
            "variants": list(VARIANTS),
            "outputs": {
                "period_performance_by_variant": "period_performance_by_variant.csv",
                "daily_equity_by_variant": "daily_equity_by_variant.csv",
                "daily_target_by_variant": "daily_target_by_variant.csv",
                "trade_ledger_by_variant": "trade_ledger_by_variant.csv",
                "risk_overlay_event_panel": "risk_overlay_event_panel.csv",
                "veto_event_panel": "veto_event_panel.csv",
                "warning_only_event_panel": "warning_only_event_panel.csv",
                "entry_without_exit_confirmation_panel": "entry_without_exit_confirmation_panel.csv",
                "target_stability_panel": "target_stability_panel.csv",
                "target_drop_from_top3_diagnostics": "target_drop_from_top3_diagnostics.csv",
                "cost_turnover_summary": "cost_turnover_summary.csv",
                "concentration_summary": "concentration_summary.csv",
                "risk_adjusted_scorecard": "risk_adjusted_scorecard.csv",
                "summary": "pool1_primary_challenger_summary_zh.md",
            },
        }
        (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        pd.DataFrame([{"status": "completed", "output_dir": str(output.resolve())}]).to_csv(
            output / "completed.csv",
            index=False,
            encoding="utf-8-sig",
        )
        pd.DataFrame(columns=["step", "error"]).to_csv(output / "failed.csv", index=False, encoding="utf-8-sig")
        log("completed", "completed", str(output.resolve()))
        (output / "current_step.txt").write_text("completed\n", encoding="utf-8")
        return output
    except Exception as exc:
        pd.DataFrame([{"step": "run_pool1_primary_risk_overlay_challenger", "error": str(exc)}]).to_csv(
            output / "failed.csv",
            index=False,
            encoding="utf-8-sig",
        )
        log("failed", "failed", str(exc))
        raise


def _build_variant_decisions(decision: pd.DataFrame) -> dict[str, pd.DataFrame]:
    variants: dict[str, pd.DataFrame] = {}
    for variant in (
        VARIANT_POOL1,
        VARIANT_POOL1_PRIMARY,
        VARIANT_POOL2_VETO,
        VARIANT_POOL2_WARNING,
        VARIANT_POOL23_VETO,
        VARIANT_POOL23_REPORT,
    ):
        frame = decision.copy()
        targets = []
        veto_reasons = []
        for row in frame.to_dict(orient="records"):
            target = _text(row.get("pool1_vote"))
            reason = _veto_reason(row, variant)
            targets.append("" if reason and variant in {VARIANT_POOL2_VETO, VARIANT_POOL23_VETO} else target)
            veto_reasons.append(reason)
        frame["variant"] = variant
        frame["variant_target"] = targets
        frame["risk_veto_reason"] = veto_reasons
        frame["pool3_shadow_used_as_formal"] = False
        variants[variant] = frame
    return variants


def _veto_reason(row: dict[str, Any], variant: str) -> str:
    pool1 = _text(row.get("pool1_vote"))
    pool2 = _text(row.get("pool2_vote"))
    pool3 = _text(row.get("pool3_vote"))
    if not pool1:
        return "pool1_no_target"
    if variant in {VARIANT_POOL2_VETO, VARIANT_POOL2_WARNING} and pool2 and pool2 != pool1:
        return "pool2_disagrees_with_pool1"
    if variant in {VARIANT_POOL23_VETO, VARIANT_POOL23_REPORT}:
        reasons = []
        if pool2 and pool2 != pool1:
            reasons.append("pool2_disagrees_with_pool1")
        if pool3 and pool3 != pool1:
            reasons.append("pool3_disagrees_with_pool1")
        return ";".join(reasons)
    return ""


def _build_daily_target_panel(variant_decisions: dict[str, pd.DataFrame], daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    lookup = {
        (variant, pd.Timestamp(row["date"]).strftime("%Y-%m-%d")): row
        for variant, frame in variant_decisions.items()
        for row in frame.to_dict(orient="records")
    }
    for item in daily.to_dict(orient="records"):
        variant = _text(item.get("variant"))
        date = _text(item.get("date"))
        source = lookup.get((variant, date), lookup.get((VARIANT_POOL1_PRIMARY, date), {}))
        target = _text(item.get("winner_ticker"))
        consensus = _text(source.get("consensus_state"))
        rows.append(
            {
                "variant": variant,
                "date": date,
                "period": item.get("period", ""),
                "formal_target": target,
                "position_ticker": item.get("position_ticker", ""),
                "action": item.get("action", ""),
                "pool1_vote": source.get("pool1_vote", item.get("pool1_vote", "")),
                "pool2_vote": source.get("pool2_vote", item.get("pool2_vote", "")),
                "pool3_vote": source.get("pool3_vote", item.get("pool3_vote", "")),
                "consensus_state": consensus,
                "risk_veto_reason": source.get("risk_veto_reason", ""),
                "entry_signal_without_exit_confirmation": bool(target and consensus != "consensus"),
                "entry_without_exit_confirmation_outcome": "needs_experiments_forward_outcome" if target and consensus != "consensus" else "",
                "possible_execution_layer_issue": False,
            }
        )
    output = pd.DataFrame(rows)
    return _add_target_stability_flags(output)


def _add_target_stability_flags(panel: pd.DataFrame) -> pd.DataFrame:
    output = panel.sort_values(["variant", "date"]).copy()
    for column in (
        "formal_target_changed_within_1d",
        "formal_target_changed_within_3d",
        "pool1_target_changed_within_1d",
        "pool1_target_changed_within_3d",
        "rapid_flip_same_target_window_1_3d",
    ):
        output[column] = False
    for _, idx in output.groupby("variant").groups.items():
        target = output.loc[idx, "formal_target"].astype(str).tolist()
        pool1 = output.loc[idx, "pool1_vote"].astype(str).tolist()
        changed = _changed_within(target)
        pool1_changed = _changed_within(pool1)
        output.loc[idx, "formal_target_changed_within_1d"] = changed[1]
        output.loc[idx, "formal_target_changed_within_3d"] = changed[3]
        output.loc[idx, "pool1_target_changed_within_1d"] = pool1_changed[1]
        output.loc[idx, "pool1_target_changed_within_3d"] = pool1_changed[3]
        output.loc[idx, "rapid_flip_same_target_window_1_3d"] = _rapid_flip_same_target(target)
        output.loc[idx, "possible_execution_layer_issue"] = output.loc[idx, "formal_target_changed_within_3d"].map(_truthy)
    return output


def _changed_within(values: list[str]) -> dict[int, list[bool]]:
    result = {1: [], 3: []}
    for index, value in enumerate(values):
        for window in result:
            future = values[index + 1 : index + window + 1]
            result[window].append(bool(value and any(item and item != value for item in future)))
    return result


def _rapid_flip_same_target(values: list[str]) -> list[bool]:
    flags = []
    for index, value in enumerate(values):
        future = values[index + 1 : index + 4]
        flags.append(bool(value and value in future and any(item and item != value for item in future)))
    return flags


def _risk_overlay_event_panel(variant_decisions: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames = []
    for variant, frame in variant_decisions.items():
        output = frame[["period", "date", "pool1_vote", "pool2_vote", "pool3_vote", "variant_target", "risk_veto_reason"]].copy()
        output.insert(0, "variant", variant)
        output["risk_veto_applied"] = output["risk_veto_reason"].astype(str).str.strip().ne("") & output["variant"].isin([VARIANT_POOL2_VETO, VARIANT_POOL23_VETO])
        output["report_only_overlay"] = output["variant"].isin([VARIANT_POOL2_WARNING, VARIANT_POOL23_REPORT])
        frames.append(output)
    return pd.concat(frames, ignore_index=True)


def _build_veto_events(variant_decisions: dict[str, pd.DataFrame]) -> pd.DataFrame:
    risk = _risk_overlay_event_panel(variant_decisions)
    veto = risk[risk["risk_veto_applied"]].copy()
    veto = veto.rename(columns={"pool1_vote": "vetoed_target"})
    return veto


def _build_warning_events(variant_decisions: dict[str, pd.DataFrame]) -> pd.DataFrame:
    risk = _risk_overlay_event_panel(variant_decisions)
    return risk[risk["report_only_overlay"] & risk["risk_veto_reason"].astype(str).str.strip().ne("")].copy()


def _entry_without_exit_panel(daily_target: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "variant",
        "date",
        "period",
        "formal_target",
        "pool1_vote",
        "pool2_vote",
        "pool3_vote",
        "consensus_state",
        "entry_signal_without_exit_confirmation",
        "entry_without_exit_confirmation_outcome",
        "risk_veto_reason",
    ]
    return daily_target[columns].copy()


def _target_drop_from_top3_overlay(daily_target: pd.DataFrame, source: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "target_in_top3_today",
        "target_drop_from_top3_next_1d",
        "target_drop_from_top3_next_2d",
        "target_drop_from_top3_next_3d",
        "target_reappears_in_top3_within_5d",
    ]
    base = daily_target[["variant", "date", "period", "formal_target"]].copy()
    if source.empty or "date" not in source.columns:
        for column in columns:
            base[column] = False
        base["target_drop_source_status"] = "missing_source"
        return base
    available = source[["date"] + [column for column in columns if column in source.columns]].copy()
    merged = base.merge(available, on="date", how="left")
    for column in columns:
        if column not in merged.columns:
            merged[column] = False
        merged[column] = merged[column].map(_truthy)
    merged["target_drop_source_status"] = "reused_three_pool_vs_pool1_comparison_panel"
    return merged


def _cost_turnover_summary(daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant, group in daily.groupby("variant", dropna=False):
        rows.append(
            {
                "variant": variant,
                "total_transaction_cost": round(float(pd.to_numeric(group["transaction_cost"], errors="coerce").sum()), 2),
                "total_turnover": round(float(pd.to_numeric(group["turnover"], errors="coerce").sum()), 2),
                "trade_days": int(group["action"].astype(str).ne("hold").sum()),
                "cost_drag_vs_initial_cash_pct": round(float(pd.to_numeric(group["transaction_cost"], errors="coerce").sum()) / INITIAL_CASH * 100, 4),
            }
        )
    return pd.DataFrame(rows)


def _concentration_summary(daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant, group in daily.groupby("variant", dropna=False):
        positions = group["position_ticker"].astype(str)
        active = positions[positions.ne("cash") & positions.ne("")]
        top = active.value_counts().index[0] if not active.empty else ""
        top_count = int(active.value_counts().iloc[0]) if not active.empty else 0
        rows.append(
            {
                "variant": variant,
                "active_position_days": int(len(active)),
                "top_ticker": top,
                "top_ticker_share": round(top_count / len(active), 6) if len(active) else 0.0,
                "00631L_share": round(float(active.eq("00631L.TW").mean()), 6) if len(active) else 0.0,
                "contribution_concentration_note": "position-day concentration only; Experiments should validate contribution concentration",
            }
        )
    return pd.DataFrame(rows)


def _risk_adjusted_scorecard(performance: pd.DataFrame, cost: pd.DataFrame, concentration: pd.DataFrame) -> pd.DataFrame:
    full = performance[performance["period_label"].eq("full")].copy()
    merged = full.merge(cost, on="variant", how="left").merge(concentration, on="variant", how="left")
    merged["diagnostic_pass_candidate"] = (
        pd.to_numeric(merged["return_pct"], errors="coerce").fillna(-999) > 0
    ) & (pd.to_numeric(merged["max_drawdown_pct"], errors="coerce").fillna(-999) > -70)
    merged["formal_absorption_ready"] = False
    merged["scorecard_note"] = "diagnostic only; formal absorption requires Experiments validation"
    return merged


def _summary_markdown(performance: pd.DataFrame) -> str:
    full = performance[performance["period_label"].eq("full")]
    lines = [
        "# Pool1 Primary Risk Overlay Challenger Panels",
        "",
        "本輸出只建立 challenger comparison panels，不改正式模型、不直接切 Pool1-only 上線。",
        "warning/report-only variants 不改績效；risk-veto variants 逐日列出 veto reason。",
        "",
        "## Full Period",
        "",
    ]
    for row in full.to_dict(orient="records"):
        lines.append(
            f"- {row.get('variant')}: return {row.get('return_pct')}%, "
            f"MDD {row.get('max_drawdown_pct')}%, trades {row.get('trade_days')}"
        )
    return "\n".join(lines) + "\n"


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _read_optional_csv(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    if not source.exists():
        return pd.DataFrame()
    return pd.read_csv(source).fillna("")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Pool1 primary + risk/report overlay challenger panels.")
    parser.add_argument("--formal-replay-dir", default=DEFAULT_FORMAL_REPLAY_DIR)
    parser.add_argument("--price-cache-dir", default=DEFAULT_PRICE_CACHE_DIR)
    parser.add_argument("--target-drop-source", default=DEFAULT_TARGET_DROP_SOURCE)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--initial-cash", type=float, default=INITIAL_CASH)
    args = parser.parse_args()
    output = run_pool1_primary_risk_overlay_challenger(
        formal_replay_dir=args.formal_replay_dir,
        price_cache_dir=args.price_cache_dir,
        target_drop_source_path=args.target_drop_source,
        output_dir=args.output_dir,
        initial_cash=args.initial_cash,
    )
    print(f"OUTPUT_DIR={output.resolve()}")


if __name__ == "__main__":
    main()
