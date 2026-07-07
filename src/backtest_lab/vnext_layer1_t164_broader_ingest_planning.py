"""Plan Layer1 t164 source package / broader ingest contract needs.

This is source/contract planning only. It does not run broader ingest,
Experiments, replay, formal model, report, or trade-decision changes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


TASK_ID = "TASK-BACKTEST-CORE-VNEXT-LAYER1-T164-SOURCE-PACKAGE-BROADER-INGEST-PLANNING-001"
DEFAULT_CLOSURE_DIR = Path("outputs/vnext_layer1_t164_official_asof_final_40of40_patch_refresh_20260707")
DEFAULT_OUTPUT_DIR = Path("outputs/vnext_layer1_t164_source_package_broader_ingest_planning_20260707")


def build_broader_ingest_planning(
    *,
    closure_dir: str | Path = DEFAULT_CLOSURE_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    closure = Path(closure_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    closure_readiness = _read_json(closure / "readiness_for_layer1_t164_official_asof_final_40of40.json")
    closure_contract = _read_csv(closure / "layer1_t164_official_asof_final_40row_contract.csv", dtype={"ticker": str})

    field_catalog = _field_catalog()
    runner_requirements = _runner_requirements()
    join_policy = _join_policy()
    coverage_audit = _coverage_audit_design()
    label_policy = _human_review_label_policy()
    future_governance = _future_data_governance()
    radar_handoff = _radar_handoff_items()
    readiness = _readiness(closure_readiness, closure_contract)

    _write_csv(field_catalog, output / "layer1_t164_source_field_catalog.csv")
    _write_csv(runner_requirements, output / "layer1_t164_broader_ingest_runner_requirements.csv")
    _write_csv(join_policy, output / "layer1_t164_t05st01_join_policy.csv")
    _write_csv(coverage_audit, output / "layer1_t164_coverage_audit_design.csv")
    _write_csv(label_policy, output / "layer1_t164_human_review_label_policy.csv")
    _write_csv(future_governance, output / "layer1_t164_future_data_governance.csv")
    _write_csv(radar_handoff, output / "layer1_t164_radar_data_handoff_items.csv")
    (output / "readiness_for_layer1_t164_source_package_broader_ingest_planning.json").write_text(
        json.dumps(readiness, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    manifest = {
        "task_id": TASK_ID,
        "status": readiness["status"],
        "output_dir": str(output.resolve()),
        "closure_input_dir": str(closure.resolve()),
        "output_files": [
            "layer1_t164_source_field_catalog.csv",
            "layer1_t164_broader_ingest_runner_requirements.csv",
            "layer1_t164_t05st01_join_policy.csv",
            "layer1_t164_coverage_audit_design.csv",
            "layer1_t164_human_review_label_policy.csv",
            "layer1_t164_future_data_governance.csv",
            "layer1_t164_radar_data_handoff_items.csv",
            "readiness_for_layer1_t164_source_package_broader_ingest_planning.json",
            "manifest.json",
            "final_summary_zh.md",
        ],
        "formal_model_changed": False,
        "trade_decision_changed": False,
        "active_in_trade_decision": False,
        "report_changed": False,
        "portfolio_replay_executed": False,
        "ready_for_strategy_replay": False,
        "not_live_rule": True,
        "forward_returns_live_rule_usage": False,
        "diagnostic_only": True,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "final_summary_zh.md").write_text(_summary(readiness), encoding="utf-8")
    return manifest


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _read_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    return pd.read_csv(path, **kwargs) if path.exists() else pd.DataFrame()


def _field_catalog() -> pd.DataFrame:
    rows = [
        ("operating_cash_flow", "t164sb05", "accepted", "exact_pit_when_joined_to_official_asof", "cashflow_quality_candidate", "raw OCF field; no formal threshold"),
        ("investing_cash_flow", "t164sb05", "accepted", "exact_pit_when_joined_to_official_asof", "cashflow_context", "raw investing cash flow field"),
        ("capex_proxy", "t164sb05", "accepted_proxy", "human_review_required", "capex_or_fcf_proxy_candidate", "label policy must mark proxy; not formal FCF"),
        ("inventory", "t164sb03", "accepted", "exact_pit_when_joined_to_official_asof", "inventory_risk_candidate", "balance-sheet inventory level"),
        ("receivables_trade", "t164sb03", "accepted_proxy", "human_review_required", "receivables_basket_candidate", "basket definition must remain explicit"),
        ("current_assets", "t164sb03", "accepted", "exact_pit_when_joined_to_official_asof", "liquidity_candidate", "needed for current_ratio"),
        ("current_liabilities", "t164sb03", "accepted", "exact_pit_when_joined_to_official_asof", "liquidity_candidate", "needed for current_ratio"),
        ("current_ratio", "derived", "accepted", "derived_pit_from_current_assets_liabilities", "solvency_candidate", "current_assets / current_liabilities"),
        ("market_available_at", "t05st01/t05st01_detail", "accepted", "official_public_announcement_timestamp", "asof_join_key", "public material-information timestamp"),
        ("signal_eligible_date", "trading_calendar", "accepted_policy", "derived_from_after_close_next_trading_day_policy", "pit_eligibility_key", "after-close announcements eligible next trading day"),
    ]
    return pd.DataFrame(
        rows,
        columns=["field", "source", "accepted_label", "source_quality", "layer1_usage_candidate", "notes"],
    ).assign(diagnostic_only=True)


def _runner_requirements() -> pd.DataFrame:
    rows = [
        ("ticker_universe", "required", "TWSE and TPEx listed/common-stock universe with listing/status PIT filters", "Radar/Data", "exclude invalid/suspended/delisted rows by effective date"),
        ("period_range", "required", "at least P1/P2 requested periods plus disclosure lag coverage; start before 2015 if needed for lagged fundamentals", "Radar/Data", "must report requested vs actual coverage"),
        ("markets", "required", "TWSE and TPEx routes both enabled and separately audited", "Radar/Data", "TPEx universal readiness still requires full coverage proof"),
        ("t164_payload_replay", "required", "dataType=2, ROC year, season 1-4, subsidiaryCompanyId empty", "Radar/Data", "cache raw payload/response hash"),
        ("t05st01_join", "required", "official announcement route per ticker/year/month/date-window with detail validation", "Radar/Data", "unmatched/ambiguous rows blocked"),
        ("rate_limit_cache", "required", "persistent per-route cache, retry ledger, response hash, route_error_count, no uncontrolled repeated calls", "Radar/Data", "support resume and failure audit"),
        ("failure_ledger", "required", "statement failure, route error, no candidate, multiple candidates, wrong period, excluded premeeting/non-financial report", "Radar/Data", "no silent row drops"),
        ("calendar_join", "required", "after-close next-trading-day eligibility using Core trading_calendar", "Core/Data", "do not use same-day if announced after market close"),
        ("full_coverage_audit", "required", "by ticker/market/period/field/match status before Experiments", "Core/Data", "bounded 40-row closure not sufficient"),
    ]
    return pd.DataFrame(rows, columns=["requirement", "status", "detail", "owner", "notes"]).assign(diagnostic_only=True)


def _join_policy() -> pd.DataFrame:
    rows = [
        ("official_timestamp_source", "accepted", "t05st01/t05st01_detail public material-information announcement timestamp", "market_available_at"),
        ("strict_subject_match", "accepted", "financial-report approval/pass subject maps to target report period", "accepted official route"),
        ("detail_period_validation", "required", "detail text/reporting period must map to target quarter or annual range", "wrong-period rows blocked"),
        ("after_close_policy", "required", "if announcement after regular close, signal eligibility starts next trading day", "same-day pre-announcement use prohibited"),
        ("multiple_strict_candidates", "blocked", "must disambiguate or keep blocked", "no silent first/last candidate selection"),
        ("premeeting_notice", "excluded", "board meeting schedule notice is not market_available_at for financial report", "not accepted"),
        ("supporting_later_announcement", "excluded_or_supporting", "lower-priority later board-resolution detail can support but not replace direct report timestamp", "do not shift accepted timestamp later when direct report exists"),
        ("quarter_end_date", "prohibited", "period end precedes disclosure", "never available_at"),
        ("query_response_datetime", "prohibited", "query time is not official availability", "never available_at"),
        ("conservative_deadline_proxy", "separate_proxy_only", "may be staged separately if Research approves, but not mixed into official route", "not official_asof"),
    ]
    return pd.DataFrame(rows, columns=["policy_item", "status", "policy_detail", "effect"]).assign(diagnostic_only=True)


def _coverage_audit_design() -> pd.DataFrame:
    rows = [
        ("ticker_coverage", "ticker", "requested_rows, materialized_rows, statement_success_rows, official_asof_matched_rows, blocked_rows", "detect ticker-level gaps"),
        ("market_coverage", "market", "TWSE/TPEx requested rows, success, route errors, match share", "separate TPEx universal readiness from TWSE"),
        ("period_coverage", "report_period", "period requested rows, actual materialized rows, official_asof match share", "avoid claiming coverage where periods are missing"),
        ("field_coverage", "field", "non_null_rows, missing_rows, missing_share, source_quality", "cashflow/inventory/receivable/current_ratio/capex availability"),
        ("match_status_coverage", "match_status", "accepted, unmatched, ambiguous, wrong_period, premeeting_excluded, route_error", "explicit blocked/proxy status"),
        ("future_data_audit", "policy", "quarter_end_used, query_time_used, deadline_proxy_used, forward_return_as_rule", "must all be false for official route"),
        ("requested_vs_actual", "range", "requested_start/end, actual_start/end, reason for gaps", "do not report actual as requested"),
    ]
    return pd.DataFrame(rows, columns=["audit_name", "group_by", "required_metrics", "purpose"]).assign(diagnostic_only=True)


def _human_review_label_policy() -> pd.DataFrame:
    rows = [
        ("capex_proxy", "proxy_human_review_required", "derived from t164sb05 cashflow rows; label variants may differ", "keep proxy label; not formal FCF", False),
        ("free_cash_flow_quality", "blocked_until_policy", "requires OCF minus accepted capex proxy policy and source-quality approval", "do not fabricate exact FCF", False),
        ("receivables_basket", "proxy_human_review_required", "receivable rows may include trade notes/accounts/other receivables depending statement label", "explicit basket definition required", False),
        ("inventory_risk", "candidate_exact_input", "inventory field accepted when t164sb03 PIT/asof joined", "risk threshold remains diagnostic only", False),
        ("current_ratio", "candidate_exact_derived_input", "current_assets/current_liabilities accepted when PIT/asof joined", "threshold remains diagnostic only", False),
    ]
    return pd.DataFrame(
        rows,
        columns=["field_or_label", "policy_status", "reason", "required_labeling", "formal_ready"],
    ).assign(diagnostic_only=True)


def _future_data_governance() -> pd.DataFrame:
    rows = [
        ("market_available_at", "must_equal_official_public_announcement_timestamp", "t05st01/t05st01_detail accepted timestamp only"),
        ("signal_eligible_date", "after_close_next_trading_day", "announcement after close cannot be used same day"),
        ("quarter_end_date", "prohibited", "not available_at"),
        ("query_response_datetime", "prohibited", "not available_at"),
        ("conservative_deadline_proxy", "separate_proxy_only", "never mixed into official route"),
        ("forward_return_as_rule", "false_required", "no forward return as rule input"),
        ("raw_response_cache", "required", "cache response hash and query payload for audit"),
        ("blocked_rows", "required", "unmatched/ambiguous rows remain blocked"),
    ]
    return pd.DataFrame(rows, columns=["governance_item", "policy", "detail"]).assign(
        future_data_violation_count=0,
        diagnostic_only=True,
    )


def _radar_handoff_items() -> pd.DataFrame:
    rows = [
        ("full_or_broader_t164_materialization_runner", "Radar/Data", "Build resumable runner over agreed ticker universe and period range for t164sb05/t164sb03 plus t05st01_detail.", "required_before_core_ingest"),
        ("raw_cache_and_hash_manifest", "Radar/Data", "Provide payload, response hash, route status, retry count, and route_error_count per ticker/period/route.", "required_before_core_ingest"),
        ("official_asof_candidate_ledger", "Radar/Data", "For every ticker/period, provide accepted timestamp or blocked reason with subject/detail evidence.", "required_before_core_ingest"),
        ("coverage_by_market_period", "Radar/Data", "Report TWSE/TPEx and period-level materialized/matched/blocked counts.", "required_before_core_ingest"),
        ("label_inventory_for_human_review", "Radar/Data", "Inventory capex and receivables labels seen in broader sample for Core/Research policy review.", "required_before_full_label_acceptance"),
        ("tpex_universal_readiness", "Radar/Data", "Confirm TPEx all-stock route stability beyond bounded samples.", "required_before_full_universe_claim"),
    ]
    return pd.DataFrame(rows, columns=["handoff_item", "next_owner", "request", "status"]).assign(diagnostic_only=True)


def _readiness(closure_readiness: dict[str, Any], closure_contract: pd.DataFrame) -> dict[str, Any]:
    matched_rows = int(closure_readiness.get("official_timestamp_matched_rows", 0))
    sample_rows = int(closure_readiness.get("sample_rows", len(closure_contract)))
    future_count = int(closure_readiness.get("future_data_violation_count", 0))
    closure_ok = sample_rows > 0 and matched_rows == sample_rows and future_count == 0
    return {
        "date": "2026-07-07",
        "task_id": TASK_ID,
        "owner": "BACKTEST_LAB Core/Data",
        "status": "broader_ingest_planning_ready_bounded_closure_only",
        "source_closure_status": closure_readiness.get("status"),
        "bounded_40row_official_asof_closure_accepted": bool(closure_ok),
        "sample_rows": sample_rows,
        "official_timestamp_matched_rows": matched_rows,
        "official_timestamp_matched_share": matched_rows / sample_rows if sample_rows else 0.0,
        "remaining_blocked_rows": int(closure_readiness.get("remaining_blocked_rows", 0)),
        "ready_for_broader_ingest_planning": True,
        "ready_for_core_t164_broader_ingest_contract": False,
        "ready_for_core_t164_broader_materialization": False,
        "ready_for_radar_full_broader_source_materialization": True,
        "ready_for_experiments": False,
        "ready_for_formal": False,
        "ready_for_strategy_replay": False,
        "full_ingest_blocked_reason": "bounded 40-row closure only; need full/broader runner, cache, full coverage audit, label policy, and Research/Strategy approval",
        "needs_radar_data_next": True,
        "future_data_violation_count": future_count,
        "not_live_rule": True,
        "forward_returns_live_rule_usage": False,
        "formal_model_changed": False,
        "trade_decision_changed": False,
        "active_in_trade_decision": False,
        "report_changed": False,
        "portfolio_replay_executed": False,
    }


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def _summary(readiness: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Layer1 t164 Source Package / Broader Ingest Planning",
            "",
            f"Status: {readiness['status']}",
            "",
            "Conclusion: 40/40 official-asof closure is accepted as bounded source hygiene closure, but it is not full-universe readiness. The next useful step is Radar/Data broader/full source materialization with cache and coverage audit.",
            "",
            "Readiness:",
            "- ready_for_broader_ingest_planning=true",
            "- ready_for_core_t164_broader_ingest_contract=false",
            "- ready_for_core_t164_broader_materialization=false",
            "- ready_for_radar_full_broader_source_materialization=true",
            "- ready_for_experiments=false",
            "- ready_for_formal=false",
            "- ready_for_strategy_replay=false",
            f"- bounded_official_timestamp_matched_rows={readiness['official_timestamp_matched_rows']}/{readiness['sample_rows']}",
            f"- future_data_violation_count={readiness['future_data_violation_count']}",
            "",
            "Next owner:",
            "- Radar/Data should build the full/broader t164+t05st01 materialization runner and source package before Core can ingest broader coverage.",
            "",
            "Flags:",
            "- formal_model_changed=false",
            "- trade_decision_changed=false",
            "- active_in_trade_decision=false",
            "- report_changed=false",
            "- portfolio_replay_executed=false",
            "- ready_for_strategy_replay=false",
            "- not_live_rule=true",
            "- forward_returns_live_rule_usage=false",
        ]
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--closure-dir", type=Path, default=DEFAULT_CLOSURE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    manifest = build_broader_ingest_planning(closure_dir=args.closure_dir, output_dir=args.output_dir)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
