from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


POOL3_ID = "large_core_bluechip_v0"
GATE_RULE_ID = "pool3_independent_stock_expert_v1"
ETF_TICKERS = {"0050.TW", "00631L.TW"}
LEVERAGED_ETF_TICKERS = {"00631L.TW"}
AI_MAINLINE_TICKERS = {
    "2330.TW",
    "2454.TW",
    "2308.TW",
    "2317.TW",
    "2382.TW",
    "3231.TW",
    "6669.TW",
}
VARIANTS = (
    "pool3_independent_stock_ranker_base",
    "pool3_independent_stock_ranker_core_veto",
    "pool3_early_cycle_stock_rotation",
    "pool3_style_breadth_to_leader",
    "pool3_current_pure_stock_style_base_control",
    "pool3_strict_confirmed_attack_ablation",
)


def run_pool3_independent_stock_expert(
    *,
    replay_panel_path: str | Path,
    top_candidates_path: str | Path,
    output_dir: str | Path,
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

    log("load_inputs", "started")
    replay = pd.read_csv(replay_panel_path).fillna("")
    top_candidates = pd.read_csv(top_candidates_path).fillna("")
    _validate_inputs(replay, top_candidates)

    log("build_candidate_coverage_audit", "started")
    audit = _candidate_coverage_audit(replay, top_candidates)
    slate = _candidate_slate(top_candidates)
    audit.to_csv(output / "pool3_candidate_coverage_audit_daily.csv", index=False, encoding="utf-8-sig")
    slate.to_csv(output / "pool3_independent_stock_candidate_slate.csv", index=False, encoding="utf-8-sig")
    _coverage_summary(audit).to_csv(output / "pool3_candidate_coverage_summary.csv", index=False, encoding="utf-8-sig")

    log("build_variant_panels", "started")
    diff_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    variant_outputs: dict[str, str] = {}
    for variant in VARIANTS:
        panel, diffs = _build_variant_panel(replay, top_candidates, audit, variant=variant)
        path = output / f"{variant}_replay_panel.csv"
        panel.to_csv(path, index=False, encoding="utf-8-sig")
        variant_outputs[variant] = str(path)
        diff_rows.extend(diffs)
        summary_rows.append(_variant_summary(panel, diffs, variant=variant))

    pd.DataFrame(diff_rows).to_csv(output / "pool3_independent_stock_expert_decision_diff.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(summary_rows).to_csv(output / "pool3_independent_stock_expert_variant_summary.csv", index=False, encoding="utf-8-sig")
    metadata = {
        "schema_version": 1,
        "task_id": "TASK-BACKTEST-CORE-POOL3-INDEPENDENT-STOCK-EXPERT-001",
        "status": "completed",
        "model": "pool3_independent_stock_expert",
        "formal_model_changed": False,
        "trade_decision_changed": False,
        "active_in_trade_decision": False,
        "gate_rule_id": GATE_RULE_ID,
        "replay_panel_path": str(replay_panel_path),
        "top_candidates_path": str(top_candidates_path),
        "variants": list(VARIANTS),
        "phase0_outputs": {
            "daily_audit": "pool3_candidate_coverage_audit_daily.csv",
            "summary": "pool3_candidate_coverage_summary.csv",
            "candidate_slate": "pool3_independent_stock_candidate_slate.csv",
        },
        "variant_outputs": variant_outputs,
        "hard_boundaries": [
            "pool3_formal_vote_target_stock_only",
            "pool1_pool2_are_risk_veto_not_confirmation",
            "etf_not_eligible_for_pool3_stock_vote",
            "direction_support_only_not_used_as_hole_filler",
            "pool3_radar_not_used",
            "valuation_not_used",
            "h3_day_trading_margin_overheat_not_used",
        ],
    }
    (output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame([{"status": "completed", "output_dir": str(output.resolve())}]).to_csv(
        output / "completed.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(columns=["step", "error"]).to_csv(output / "failed.csv", index=False, encoding="utf-8-sig")
    log("completed", "completed", str(output.resolve()))
    (output / "current_step.txt").write_text("completed\n", encoding="utf-8")
    return output


def _validate_inputs(replay: pd.DataFrame, top_candidates: pd.DataFrame) -> None:
    required_replay = {"period", "requested_signal_date", "pool_id", "top_ticker", "eligible_for_pool_selection"}
    required_candidates = {"period", "requested_signal_date", "pool_id", "ticker", "asset_type", "selection_layer", "eligible_for_pool_selection"}
    missing_replay = required_replay - set(replay.columns)
    missing_candidates = required_candidates - set(top_candidates.columns)
    if missing_replay:
        raise ValueError("missing replay panel columns: " + ",".join(sorted(missing_replay)))
    if missing_candidates:
        raise ValueError("missing top candidate columns: " + ",".join(sorted(missing_candidates)))


def _candidate_coverage_audit(replay: pd.DataFrame, top_candidates: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    keys = replay[replay["pool_id"].astype(str) == POOL3_ID][["period", "requested_signal_date"]].drop_duplicates()
    for _, key in keys.iterrows():
        period = str(key["period"])
        date = str(key["requested_signal_date"])
        candidates = _pool3_candidates(top_candidates, period=period, date=date)
        stock = _stock_candidates(candidates)
        formal_stock = _formal_stock_candidates(candidates)
        observation_stock = stock[~stock.index.isin(formal_stock.index)]
        max_stock_score = _max_score(stock)
        reason = _coverage_state(candidates, stock, formal_stock, max_stock_score)
        rows.append(
            {
                "period": period,
                "requested_signal_date": date,
                "pool3_candidate_rows": int(len(candidates)),
                "stock_candidate_rows": int(len(stock)),
                "etf_candidate_rows": int(len(candidates) - len(stock)),
                "formal_stock_candidate_rows": int(len(formal_stock)),
                "observation_stock_candidate_rows": int(len(observation_stock)),
                "max_stock_score": round(max_stock_score, 6),
                "true_data_blocked": len(candidates) == 0,
                "candidate_empty": len(candidates) > 0 and len(stock) == 0,
                "filtered_out_by_gate": len(stock) > 0 and len(formal_stock) == 0 and max_stock_score >= 0.25,
                "no_edge_after_scoring": len(stock) > 0 and len(formal_stock) == 0 and max_stock_score < 0.25,
                "coverage_state": reason,
            }
        )
    return pd.DataFrame(rows)


def _coverage_state(candidates: pd.DataFrame, stock: pd.DataFrame, formal_stock: pd.DataFrame, max_score: float) -> str:
    if candidates.empty:
        return "true_data_blocked"
    if stock.empty:
        return "candidate_empty"
    if not formal_stock.empty:
        return "formal_stock_available"
    if max_score >= 0.25:
        return "filtered_out_by_gate"
    return "no_edge_after_scoring"


def _candidate_slate(top_candidates: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in top_candidates[top_candidates["pool_id"].astype(str) == POOL3_ID].iterrows():
        ticker = str(row.get("ticker") or "").strip()
        is_etf = _is_etf(ticker, row.get("asset_type", ""))
        rows.append(
            {
                "period": row.get("period", ""),
                "requested_signal_date": row.get("requested_signal_date", ""),
                "ticker": ticker,
                "display": row.get("display", ""),
                "rank": row.get("rank", ""),
                "score": _score(row.to_dict()),
                "asset_type": row.get("asset_type", ""),
                "is_etf": is_etf,
                "is_leveraged_etf": ticker in LEVERAGED_ETF_TICKERS,
                "selection_layer": row.get("selection_layer", ""),
                "eligible_for_pool_selection": _truthy(row.get("eligible_for_pool_selection")),
                "attack_gate_open": _truthy(row.get("attack_gate_open")),
                "base_pool_passed": _truthy(row.get("base_pool_passed")),
                "pool3_independent_candidate_class": _candidate_class(row.to_dict()),
                "gate_rule_id": row.get("gate_rule_id", ""),
            }
        )
    return pd.DataFrame(rows)


def _coverage_summary(audit: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if audit.empty:
        return pd.DataFrame(rows)
    for period, frame in audit.groupby("period", dropna=False):
        total = len(frame)
        rows.append(
            {
                "period": period,
                "dates": total,
                "formal_stock_available_dates": int((frame["coverage_state"] == "formal_stock_available").sum()),
                "true_data_blocked_dates": int(frame["true_data_blocked"].sum()),
                "candidate_empty_dates": int(frame["candidate_empty"].sum()),
                "filtered_out_by_gate_dates": int(frame["filtered_out_by_gate"].sum()),
                "no_edge_after_scoring_dates": int(frame["no_edge_after_scoring"].sum()),
                "formal_stock_available_rate": round(float((frame["coverage_state"] == "formal_stock_available").sum() / total), 6) if total else 0.0,
                "candidate_problem_rate": round(float((frame["coverage_state"] != "formal_stock_available").sum() / total), 6) if total else 0.0,
            }
        )
    return pd.DataFrame(rows)


def _build_variant_panel(
    replay: pd.DataFrame,
    top_candidates: pd.DataFrame,
    audit: pd.DataFrame,
    *,
    variant: str,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    panel = replay.copy().astype(object)
    _ensure_contract_columns(panel)
    diffs: list[dict[str, Any]] = []
    audit_index = {
        (str(row["period"]), str(row["requested_signal_date"])): row.to_dict()
        for _, row in audit.iterrows()
    }
    for index, row in panel[panel["pool_id"].astype(str) == POOL3_ID].iterrows():
        original = row.to_dict()
        period = str(row.get("period") or "")
        date = str(row.get("requested_signal_date") or row.get("signal_date") or "")
        candidates = _pool3_candidates(top_candidates, period=period, date=date)
        coverage = audit_index.get((period, date), {})
        selected = _select_candidate(panel, candidates, row, variant=variant)
        adjusted = _apply_selection(row.to_dict(), selected, coverage=coverage, variant=variant)
        for key, value in adjusted.items():
            panel.at[index, key] = value
        if _changed(original, adjusted):
            diffs.append(
                {
                    "variant": variant,
                    "period": period,
                    "requested_signal_date": date,
                    "original_ticker": original.get("top_ticker", ""),
                    "challenger_ticker": adjusted.get("top_ticker", ""),
                    "original_selection_layer": original.get("selection_layer", ""),
                    "challenger_selection_layer": adjusted.get("selection_layer", ""),
                    "coverage_state": adjusted.get("pool3_candidate_coverage_state", ""),
                    "blocked_reason": adjusted.get("blocked_reason", ""),
                }
            )
    return panel, diffs


def _ensure_contract_columns(panel: pd.DataFrame) -> None:
    for column in (
        "asset_class",
        "is_etf",
        "is_leveraged_etf",
        "eligible_for_pool3_stock_vote",
        "eligible_for_market_exposure",
        "eligible_for_exact_ticker_consensus",
        "pool3_independent_variant",
        "pool3_candidate_coverage_state",
        "pool3_candidate_source_state",
        "pool3_filtered_out_by_gate",
        "pool3_true_data_blocked",
        "pool3_candidate_empty",
        "pool3_no_edge_after_scoring",
        "pool3_full_stock_vote_rate_target",
        "blocked_reason",
    ):
        if column not in panel.columns:
            panel[column] = ""


def _select_candidate(panel: pd.DataFrame, candidates: pd.DataFrame, row: pd.Series, *, variant: str) -> dict[str, Any] | None:
    date = str(row.get("requested_signal_date") or row.get("signal_date") or "")
    period = str(row.get("period") or "")
    peer = _peer_votes(panel, period=period, date=date)
    stock = _stock_candidates(candidates)
    formal = _formal_stock_candidates(candidates)
    if variant == "pool3_current_pure_stock_style_base_control":
        return _first_row(formal)
    if variant == "pool3_strict_confirmed_attack_ablation":
        confirmed = formal[formal["attack_gate_open"].map(_truthy)] if "attack_gate_open" in formal.columns else formal.iloc[0:0]
        return _first_row(confirmed)
    if variant == "pool3_independent_stock_ranker_core_veto" and _core_risk_veto(peer):
        return None
    if variant == "pool3_early_cycle_stock_rotation":
        selected = _first_row(formal)
        if selected is not None:
            return selected
        early = stock[(stock.apply(lambda r: _score(r.to_dict()), axis=1) >= 0.25) & (stock.get("rank", 999).apply(_rank_value) <= 5)]
        return _first_row(early)
    if variant == "pool3_style_breadth_to_leader":
        breadth = stock[stock.apply(lambda r: _score(r.to_dict()), axis=1) >= 0.2]
        if len(breadth) >= 2:
            return _first_row(breadth)
        return _first_row(formal)
    return _first_row(formal)


def _apply_selection(row: dict[str, Any], selected: dict[str, Any] | None, *, coverage: dict[str, Any], variant: str) -> dict[str, Any]:
    row.update(
        {
            "pool3_independent_variant": variant,
            "pool3_candidate_coverage_state": coverage.get("coverage_state", ""),
            "pool3_filtered_out_by_gate": _bool_text(bool(coverage.get("filtered_out_by_gate", False))),
            "pool3_true_data_blocked": _bool_text(bool(coverage.get("true_data_blocked", False))),
            "pool3_candidate_empty": _bool_text(bool(coverage.get("candidate_empty", False))),
            "pool3_no_edge_after_scoring": _bool_text(bool(coverage.get("no_edge_after_scoring", False))),
            "pool3_full_stock_vote_rate_target": ">=0.15",
        }
    )
    if selected is None:
        reason = f"Pool3 independent stock expert：{variant} 無可投純個股；coverage={coverage.get('coverage_state', 'unknown')}。"
        row.update(
            {
                "selection_layer": "observation_only",
                "eligible_for_pool_selection": "false",
                "eligible_for_pool3_stock_vote": "false",
                "eligible_for_market_exposure": "false",
                "eligible_for_exact_ticker_consensus": "false",
                "attack_gate_open": "false",
                "gate_rule_id": GATE_RULE_ID,
                "gate_reason": reason,
                "selection_reason": reason,
                "blocked_reason": reason,
            }
        )
        return row
    ticker = str(selected.get("ticker") or "").strip()
    display = str(selected.get("display") or ticker).strip()
    row.update(
        {
            "top_ticker": ticker,
            "top_display": display,
            "top_asset_type": "stock",
            "asset_class": "stock",
            "is_etf": "false",
            "is_leveraged_etf": "false",
            "selection_layer": "formal_candidate",
            "eligible_for_pool_selection": "true",
            "eligible_for_pool3_stock_vote": "true",
            "eligible_for_market_exposure": "false",
            "eligible_for_exact_ticker_consensus": "true",
            "attack_gate_open": "true",
            "gate_rule_id": GATE_RULE_ID,
            "gate_reason": f"Pool3 independent stock expert：{variant} 選出純個股；Pool1/Pool2 不作確認條件。",
            "selection_reason": f"Pool3 independent stock expert：{variant} 選出純個股；Pool1/Pool2 不作確認條件。",
            "blocked_reason": "",
        }
    )
    return row


def _pool3_candidates(top_candidates: pd.DataFrame, *, period: str, date: str) -> pd.DataFrame:
    subset = top_candidates[
        (top_candidates["period"].astype(str) == period)
        & (top_candidates["requested_signal_date"].astype(str) == date)
        & (top_candidates["pool_id"].astype(str) == POOL3_ID)
    ].copy()
    if "rank" in subset.columns:
        subset["_rank_number"] = pd.to_numeric(subset["rank"], errors="coerce").fillna(999)
        subset = subset.sort_values("_rank_number")
    return subset


def _stock_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty:
        return candidates.copy()
    return candidates[~candidates.apply(lambda row: _is_etf(str(row.get("ticker") or ""), row.get("asset_type", "")), axis=1)].copy()


def _formal_stock_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    stock = _stock_candidates(candidates)
    if stock.empty:
        return stock
    return stock[
        stock["eligible_for_pool_selection"].map(_truthy)
        & (stock["selection_layer"].astype(str) == "formal_candidate")
    ].copy()


def _first_row(frame: pd.DataFrame) -> dict[str, Any] | None:
    if frame.empty:
        return None
    return frame.iloc[0].to_dict()


def _peer_votes(panel: pd.DataFrame, *, period: str, date: str) -> dict[str, str]:
    subset = panel[
        (panel["period"].astype(str) == period)
        & (panel["requested_signal_date"].astype(str) == date)
        & panel["eligible_for_pool_selection"].map(_truthy)
    ]
    return {
        "pool1": _vote_for_fragment(subset, "ai_theme_large_cap"),
        "pool2": _vote_for_fragment(subset, "tw50_dynamic_constituents"),
    }


def _vote_for_fragment(subset: pd.DataFrame, fragment: str) -> str:
    rows = subset[subset["pool_id"].astype(str).str.contains(fragment, na=False)]
    if rows.empty:
        return ""
    return str(rows.iloc[0].get("top_ticker") or "").strip()


def _core_risk_veto(peer: dict[str, str]) -> bool:
    return not any(str(value or "").strip() for value in peer.values())


def _candidate_class(candidate: dict[str, Any]) -> str:
    ticker = str(candidate.get("ticker") or "").strip()
    if _is_etf(ticker, candidate.get("asset_type", "")):
        return "market_exposure_excluded"
    if _truthy(candidate.get("eligible_for_pool_selection")) and str(candidate.get("selection_layer") or "") == "formal_candidate":
        return "formal_stock_candidate"
    if _score(candidate) >= 0.25:
        return "filtered_or_watch_stock"
    return "low_edge_stock"


def _variant_summary(panel: pd.DataFrame, diff_rows: list[dict[str, Any]], *, variant: str) -> dict[str, Any]:
    pool3 = panel[panel["pool_id"].astype(str) == POOL3_ID]
    stock_votes = pool3["eligible_for_pool3_stock_vote"].map(_truthy)
    exact_votes = pool3["eligible_for_exact_ticker_consensus"].map(_truthy)
    etf_votes = pool3["is_etf"].map(_truthy) & stock_votes
    tickers = pool3.loc[stock_votes, "top_ticker"].astype(str).str.strip()
    if tickers.empty:
        top_ticker = ""
        top_share = 0.0
    else:
        counts = tickers.value_counts()
        top_ticker = str(counts.index[0])
        top_share = round(float(counts.iloc[0] / len(tickers)), 6)
    full_vote_rows = int(stock_votes.sum())
    pool3_rows = int(len(pool3))
    return {
        "variant": variant,
        "pool3_rows": pool3_rows,
        "full_stock_vote_rows": full_vote_rows,
        "full_stock_vote_rate": round(float(full_vote_rows / pool3_rows), 6) if pool3_rows else 0.0,
        "direction_support_only_rows": 0,
        "direction_support_only_rate": 0.0,
        "pool3_etf_stock_vote_rows": int(etf_votes.sum()),
        "pool3_etf_exact_consensus_rows": int((pool3["is_etf"].map(_truthy) & exact_votes).sum()),
        "top_ticker": top_ticker,
        "top_ticker_day_share_among_stock_votes": top_share,
        "insufficient_vote_count_fail": full_vote_rows < max(1, int(pool3_rows * 0.15)),
        "changed_rows": len(diff_rows),
        "formal_model_changed": False,
        "trade_decision_changed": False,
        "active_in_trade_decision": False,
    }


def _changed(original: dict[str, Any], adjusted: dict[str, Any]) -> bool:
    keys = ("top_ticker", "selection_layer", "eligible_for_pool_selection", "eligible_for_pool3_stock_vote")
    return any(str(original.get(key, "")) != str(adjusted.get(key, "")) for key in keys)


def _max_score(frame: pd.DataFrame) -> float:
    if frame.empty:
        return 0.0
    return max(_score(row.to_dict()) for _, row in frame.iterrows())


def _score(candidate: dict[str, Any]) -> float:
    for key in ("rank_score", "score"):
        value = pd.to_numeric(pd.Series([candidate.get(key, "")]), errors="coerce").iloc[0]
        if pd.notna(value):
            return float(value)
    rank = _rank_value(candidate.get("rank", ""))
    return max(0.0, 1.0 - rank / 10.0)


def _rank_value(value: object) -> float:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(number) if pd.notna(number) else 999.0


def _is_etf(ticker: str, asset_type: object = "") -> bool:
    return ticker in ETF_TICKERS or str(asset_type).strip().lower() == "etf"


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _bool_text(value: bool) -> str:
    return "true" if value else "false"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Pool3 independent pure-stock expert audit and challenger panels.")
    parser.add_argument("--replay-panel", required=True)
    parser.add_argument("--top-candidates", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = run_pool3_independent_stock_expert(
        replay_panel_path=args.replay_panel,
        top_candidates_path=args.top_candidates,
        output_dir=args.output_dir,
    )
    print(f"OUTPUT_DIR={output.resolve()}")


if __name__ == "__main__":
    main()
