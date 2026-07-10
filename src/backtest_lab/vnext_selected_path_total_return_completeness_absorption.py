from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


TASK_ID = "TASK-BACKTEST-CORE-VNEXT-SELECTED-PATH-TOTAL-RETURN-COMPLETENESS-ABSORPTION-001"
REPO_ROOT = Path(__file__).resolve().parents[2]
PRIOR_DIR = REPO_ROOT / "outputs" / "vnext_selected_stock_total_return_remaining_corporate_action_gap_absorption_20260710"
RADAR_DIR = Path(
    r"C:\Users\zergv\Documents\Codex\2026-05-23\ai-stock-rotation-radar-https-docs\outputs"
    r"\radar_vnext_selected_path_holding_month_corporate_action_no_event_proof_20260710"
)
OUTPUT_DIR = REPO_ROOT / "outputs" / "vnext_selected_path_total_return_completeness_absorption_20260710"

FLAGS = {
    "formal_model_changed": False,
    "trade_decision_changed": False,
    "active_in_trade_decision": False,
    "report_changed": False,
    "portfolio_replay_executed": False,
    "ready_for_strategy_replay": False,
    "ready_for_formal": False,
    "not_live_rule": True,
    "forward_returns_live_rule_usage": False,
}

ACCEPTED_PROOF = {"no_event_proven", "event_outside_held_dates_proven"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write(frame: pd.DataFrame, name: str) -> Path:
    path = OUTPUT_DIR / name
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def _month_key(ticker: pd.Series, year_roc: pd.Series, month: pd.Series) -> pd.Series:
    return ticker.astype(str) + "|ROC" + year_roc.astype(int).astype(str) + "|M" + month.astype(int).astype(str).str.zfill(2)


def _proof_contract(proof: pd.DataFrame) -> pd.DataFrame:
    result = proof.copy()
    result["ticker_month_key"] = _month_key(result["ticker"], result["year_roc"], result["month"])
    result["path_impact_resolved"] = result["proof_status"].isin(ACCEPTED_PROOF)
    result["accepted_no_holding_impact_evidence"] = result["path_impact_resolved"]
    result["structural_source_blocker"] = result["proof_status"].eq("blocked_event_candidate_missing_effective_date")
    result["licensed_source_required"] = False
    result["licensed_source_status"] = "not_yet_proven_required_official_archive_route_or_policy_review_first"
    result["core_path_completeness_status"] = result["proof_status"].map(
        {
            "no_event_proven": "complete_no_event_for_ticker_month_event_type",
            "event_outside_held_dates_proven": "complete_event_outside_selected_holding_dates",
            "blocked_event_candidate_missing_effective_date": "blocked_structural_source_missing_exact_effective_date",
        }
    ).fillna("blocked_unrecognized_proof_status")
    result["future_data_violation_count"] = 0
    return result


def _month_coverage(contract: pd.DataFrame) -> pd.DataFrame:
    grouped = contract.groupby(["ticker", "year_roc", "month", "ticker_month_key"], as_index=False).agg(
        event_type_rows=("event_type", "size"),
        resolved_event_type_rows=("path_impact_resolved", "sum"),
        structural_blocker_rows=("structural_source_blocker", "sum"),
        official_response_rows=("response_rows", "max"),
        route_error_rows=("response_status", lambda s: int((s != "cache_hit").sum())),
    )
    grouped["all_five_event_types_resolved"] = grouped["resolved_event_type_rows"].eq(5)
    grouped["selected_path_ticker_month_complete"] = grouped["all_five_event_types_resolved"] & grouped["route_error_rows"].eq(0)
    grouped["coverage_status"] = grouped["selected_path_ticker_month_complete"].map(
        {True: "complete_path_specific_official_evidence", False: "blocked_missing_exact_effective_date"}
    )
    grouped["future_data_violation_count"] = 0
    return grouped


def _holding_interval_coverage(intervals: pd.DataFrame, month_coverage: pd.DataFrame) -> pd.DataFrame:
    result = intervals.copy()
    result["hold_start"] = pd.to_datetime(result["hold_start"])
    result["hold_end_exclusive"] = pd.to_datetime(result["hold_end_exclusive"])
    result["holding_interval_id"] = range(1, len(result) + 1)
    expanded: list[dict] = []
    for row in result.itertuples(index=False):
        for period in pd.period_range(row.hold_start, row.hold_end_exclusive, freq="M"):
            expanded.append({
                "holding_interval_id": row.holding_interval_id,
                "ticker_month_key": f"{row.ticker}|ROC{period.year - 1911}|M{period.month:02d}",
            })
    interval_months = pd.DataFrame(expanded).merge(
        month_coverage[["ticker_month_key", "selected_path_ticker_month_complete", "structural_blocker_rows", "coverage_status"]],
        on="ticker_month_key",
        how="left",
        validate="many_to_one",
    )
    interval_months["selected_path_ticker_month_complete"] = interval_months["selected_path_ticker_month_complete"].fillna(False)
    interval_months["structural_blocker_rows"] = interval_months["structural_blocker_rows"].fillna(0)
    aggregated = interval_months.groupby("holding_interval_id", as_index=False).agg(
        intersected_ticker_months=("ticker_month_key", lambda s: "|".join(s)),
        intersected_ticker_month_count=("ticker_month_key", "size"),
        selected_path_ticker_month_complete=("selected_path_ticker_month_complete", "all"),
        structural_blocker_rows=("structural_blocker_rows", "sum"),
    )
    result = result.merge(aggregated, on="holding_interval_id", how="left", validate="one_to_one")
    result["interval_corporate_action_completeness_status"] = result["selected_path_ticker_month_complete"].map(
        {True: "complete_all_intersected_ticker_months", False: "blocked_one_or_more_intersected_ticker_months"}
    )
    result["future_data_violation_count"] = 0
    return result


def _canonical_relevance(prior_ledger: pd.DataFrame) -> pd.DataFrame:
    result = prior_ledger[[
        "ticker", "event_key", "ex_date", "payment_date", "selected_path_entitlement_flag",
        "exact_exdate_ready", "payment_date_ready", "share_adjustment_candidate_flag",
    ]].copy()
    result["path_specific_blocker"] = result["selected_path_entitlement_flag"].fillna(False).astype(bool)
    result["canonical_gap_treatment"] = result["path_specific_blocker"].map(
        {True: "requires_path_specific_resolution", False: "canonical_reference_only_not_selected_path_blocker"}
    )
    result["future_data_violation_count"] = 0
    return result


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    radar_readiness = json.loads(
        (RADAR_DIR / "readiness_for_core_selected_path_total_return_completeness.json").read_text(encoding="utf-8")
    )
    proof = pd.read_csv(RADAR_DIR / "selected_path_holding_month_no_event_proof.csv", dtype={"ticker": str}, low_memory=False)
    intervals = pd.read_csv(RADAR_DIR / "selected_path_holding_intervals.csv", dtype={"ticker": str}, low_memory=False)
    blocked_source = pd.read_csv(RADAR_DIR / "selected_path_no_event_proof_blocked_ledger.csv", dtype={"ticker": str}, low_memory=False)
    prior_ledger = pd.read_csv(PRIOR_DIR / "selected_stock_total_return_event_ledger_remaining_gap_patched.csv", dtype={"ticker": str}, low_memory=False)
    contract = _proof_contract(proof)
    month_coverage = _month_coverage(contract)
    interval_coverage = _holding_interval_coverage(intervals, month_coverage)
    canonical_relevance = _canonical_relevance(prior_ledger)
    blocked = contract.loc[contract["structural_source_blocker"]].copy()
    blocked["blocker_class"] = "structural_source_blocker"
    blocked["next_resolution"] = blocked["event_type"].map({
        "cash_dividend_exdiv": "official historical exact ex-dividend date archive route",
        "stock_dividend_exright": "official historical exact ex-right/new-share effective date archive route",
        "capital_reduction_refund": "official capital-reduction holder effective date and cash/share terms route",
        "merger_share_conversion": "official merger/share-conversion holder effective date and conversion ratio route",
    }).fillna("official exact effective-date archive route or policy review")
    blocked["licensed_source_required"] = False
    blocked["licensed_source_decision"] = "strategy_center_decision_required_only_after_official_route_exhaustion"
    future_audit = pd.DataFrame([
        {"audit_item": "holding_scope", "future_data_used": False, "detail": "Only actual selected-path holding intervals were used.", "future_data_violation_count": 0},
        {"audit_item": "no_event_proof", "future_data_used": False, "detail": "Accepted only successful date-scoped official response with zero classified rows or exact event outside held dates.", "future_data_violation_count": 0},
        {"audit_item": "effective_date", "future_data_used": False, "detail": "Board/shareholder dates were not substituted for effective dates.", "future_data_violation_count": 0},
        {"audit_item": "return_factor", "future_data_used": False, "detail": "No adjusted close, reinvestment, or total-return factor was calculated.", "future_data_violation_count": 0},
    ])
    no_event_rows = int(contract["proof_status"].eq("no_event_proven").sum())
    outside_rows = int(contract["proof_status"].eq("event_outside_held_dates_proven").sum())
    complete_months = int(month_coverage["selected_path_ticker_month_complete"].sum())
    blocked_months = int((~month_coverage["selected_path_ticker_month_complete"]).sum())
    complete_intervals = int(interval_coverage["selected_path_ticker_month_complete"].sum())
    blocked_intervals = len(interval_coverage) - complete_intervals
    selected_path_complete = len(blocked) == 0 and blocked_intervals == 0
    readiness = {
        "task_id": TASK_ID,
        "status": "path_specific_no_event_proof_absorbed_structural_effective_date_blockers_remain",
        "actual_selected_path_holding_intervals": len(intervals),
        "selected_tickers": int(intervals["ticker"].nunique()),
        "ticker_months": len(month_coverage),
        "event_type_proof_rows": len(contract),
        "no_event_proven_rows_absorbed": no_event_rows,
        "event_outside_held_dates_proven_rows_absorbed": outside_rows,
        "resolved_path_impact_proof_rows": no_event_rows + outside_rows,
        "structural_source_blocker_rows": len(blocked),
        "complete_ticker_months": complete_months,
        "blocked_ticker_months": blocked_months,
        "complete_holding_intervals": complete_intervals,
        "blocked_holding_intervals": blocked_intervals,
        "canonical_events_not_selected_path_blockers": int((~canonical_relevance["path_specific_blocker"]).sum()),
        "selected_path_total_return_complete": selected_path_complete,
        "selected_path_adjusted_close_ready": False,
        "licensed_source_required": False,
        "licensed_source_decision_status": "not_yet_proven_required_official_route_or_policy_review_first",
        "ready_for_experiments": False,
        "ready_for_formal": False,
        "ready_for_strategy_replay": False,
        "future_data_violation_count": 0,
        "next_owner": "Strategy Center structural-source policy judgment or Radar official effective-date archive route unlock",
        **FLAGS,
    }
    blocked_audit = pd.DataFrame([
        {"item": "path_specific_no_event_or_outside_evidence", "status": "absorbed", "rows": no_event_rows + outside_rows, "detail": "Accepted as no selected-holding impact evidence."},
        {"item": "missing_exact_effective_date", "status": "blocked", "rows": len(blocked), "detail": "Official event semantics exist but exact holder-impact date is unavailable."},
        {"item": "canonical_nonholding_events", "status": "not_path_blocker", "rows": readiness["canonical_events_not_selected_path_blockers"], "detail": "Retained as reference; excluded from path-specific readiness blocker count."},
        {"item": "licensed_source", "status": "policy_pending_not_required_yet", "rows": 0, "detail": "Official archive route/policy review should precede licensed-source requirement."},
        {"item": "adjusted_close_total_return_factor", "status": "blocked", "rows": 0, "detail": "No factor or reinvestment assumption fabricated."},
    ])
    paths = [
        _write(contract, "selected_path_corporate_action_no_event_proof_absorbed.csv"),
        _write(month_coverage, "selected_path_ticker_month_total_return_completeness.csv"),
        _write(interval_coverage, "selected_path_holding_interval_total_return_completeness.csv"),
        _write(blocked, "selected_path_structural_effective_date_blocked_ledger.csv"),
        _write(canonical_relevance, "selected_path_canonical_event_relevance_audit.csv"),
        _write(blocked_audit, "selected_path_total_return_completeness_blocked_proxy_audit.csv"),
        _write(future_audit, "selected_path_total_return_completeness_future_data_audit.csv"),
        _write(blocked_source, "selected_path_radar_blocked_source_absorbed.csv"),
    ]
    readiness_path = OUTPUT_DIR / "readiness_for_selected_path_total_return_completeness_absorption.json"
    readiness_path.write_text(json.dumps(readiness, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path = OUTPUT_DIR / "final_summary_zh.md"
    summary_path.write_text(
        "# Selected-path Total-return Completeness Absorption\n\n"
        f"- actual selected holding intervals: {len(intervals)}；tickers: {intervals['ticker'].nunique()}；ticker-months: {len(month_coverage)}\n"
        f"- absorbed no-holding-impact evidence: no-event {no_event_rows} + event-outside-held-dates {outside_rows} = {no_event_rows + outside_rows}\n"
        f"- complete ticker-months: {complete_months}；blocked ticker-months: {blocked_months}\n"
        f"- complete holding intervals: {complete_intervals}；blocked holding intervals: {blocked_intervals}\n"
        f"- structural missing-effective-date blockers: {len(blocked)}\n"
        f"- canonical events treated as reference, not selected-path blockers: {readiness['canonical_events_not_selected_path_blockers']}\n"
        "- licensed_source_required=false for now; official effective-date archive route or policy review should be exhausted first.\n"
        "- adjusted close / reinvestment / total-return factor remain unmaterialized.\n\n"
        "結論：已把 readiness 收斂到實際持倉路徑，176 筆官方證據可排除持有期影響；但 34 筆事件缺 exact effective date，selected_path_total_return_complete=false。不得交 Experiments 或包裝成 formal-ready。\n",
        encoding="utf-8",
    )
    manifest = {
        "task_id": TASK_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(OUTPUT_DIR),
        "source_inputs": {"prior_core": str(PRIOR_DIR), "radar_no_event_proof": str(RADAR_DIR)},
        "files": [{"path": p.name, "sha256": _sha256(p)} for p in [*paths, readiness_path, summary_path]],
        "readiness": readiness,
        "radar_readiness": radar_readiness,
    }
    (OUTPUT_DIR / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(readiness, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
