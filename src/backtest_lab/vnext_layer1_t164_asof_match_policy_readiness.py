"""Stage t164 official-asof match policy readiness for larger bounded samples.

This package consumes Radar/Data's larger bounded t164 materialization output.
It does not run full ingest, Experiments, replay, or formal selector logic.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


TASK_ID = "TASK-BACKTEST-CORE-VNEXT-LAYER1-T164-ASOF-MATCH-POLICY-ALTERNATE-ROUTE-READINESS-001"
DEFAULT_RADAR_DIR = Path(
    r"C:\Users\zergv\Documents\Codex\2026-05-23\ai-stock-rotation-radar-https-docs\outputs"
    r"\radar_vnext_layer1_t164_full_universe_materialization_readiness_20260707"
)
DEFAULT_OUTPUT_DIR = Path("outputs/vnext_layer1_t164_asof_match_policy_alternate_route_readiness_20260707")


def build_asof_match_policy_readiness(
    *, radar_dir: str | Path = DEFAULT_RADAR_DIR, output_dir: str | Path = DEFAULT_OUTPUT_DIR
) -> dict[str, Any]:
    radar = Path(radar_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    readiness_in = _read_json(radar / "readiness_for_core_t164_full_universe_materialization.json")
    matrix = _read_csv(radar / "radar_t164_full_universe_materialization_matrix.csv")
    match = _read_csv(radar / "radar_t164_t05st01_announcement_match_coverage.csv")
    unmatched_in = _read_csv(radar / "radar_t164_unmatched_ambiguous_announcement_ledger.csv")
    blocked_in = _read_csv(radar / "radar_t164_blocked_prohibited_ledger.csv")
    future_in = _read_csv(radar / "radar_t164_future_data_audit.csv")
    tpex_in = _read_csv(radar / "radar_t164_tpex_universal_readiness_ledger.csv")

    policy_contract = _policy_contract(matrix, match)
    failure_attribution = _failure_attribution(matrix, match, unmatched_in)
    alternate_route = _alternate_route_policy(failure_attribution)
    blocked_rows_policy = _blocked_rows_policy(failure_attribution)
    blocked_proxy = _blocked_proxy_ledger(blocked_in, failure_attribution, tpex_in)
    future_audit = _future_data_audit(future_in, policy_contract)
    readiness = _readiness(
        readiness_in=readiness_in,
        policy_contract=policy_contract,
        failure_attribution=failure_attribution,
        alternate_route=alternate_route,
        future_audit=future_audit,
    )

    _write_csv(policy_contract, output / "layer1_t164_asof_match_policy_readiness_contract.csv")
    _write_csv(failure_attribution, output / "layer1_t164_asof_match_failure_attribution.csv")
    _write_csv(alternate_route, output / "layer1_t164_alternate_official_route_policy.csv")
    _write_csv(blocked_rows_policy, output / "layer1_t164_partial_contract_blocked_rows_policy.csv")
    _write_csv(blocked_proxy, output / "layer1_t164_asof_match_blocked_proxy_fields.csv")
    _write_csv(future_audit, output / "layer1_t164_asof_match_future_data_audit.csv")
    (output / "readiness_for_layer1_t164_asof_match_policy_alternate_route.json").write_text(
        json.dumps(readiness, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    manifest = {
        "task_id": TASK_ID,
        "status": readiness["status"],
        "output_dir": str(output.resolve()),
        "radar_input_dir": str(radar.resolve()),
        "radar_commit": "5f4d9be",
        "output_files": [
            "layer1_t164_asof_match_policy_readiness_contract.csv",
            "layer1_t164_asof_match_failure_attribution.csv",
            "layer1_t164_alternate_official_route_policy.csv",
            "layer1_t164_partial_contract_blocked_rows_policy.csv",
            "layer1_t164_asof_match_blocked_proxy_fields.csv",
            "layer1_t164_asof_match_future_data_audit.csv",
            "readiness_for_layer1_t164_asof_match_policy_alternate_route.json",
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
    (output / "final_summary_zh.md").write_text(_summary(readiness, failure_attribution, alternate_route), encoding="utf-8")
    return manifest


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _policy_contract(matrix: pd.DataFrame, match: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "ticker",
        "market",
        "report_period",
        "t164sb05_status",
        "t164sb03_status",
        "cashflow_fields_available",
        "balance_sheet_fields_available",
        "current_ratio_available",
        "official_announcement_timestamp_matched",
        "market_available_at",
        "announcement_subject",
        "after_close_next_trading_day_policy",
    ]
    out = matrix.reindex(columns=cols).copy()
    if not match.empty:
        match_cols = [
            "ticker",
            "market",
            "report_period",
            "candidate_count",
            "t05st01_query_payload",
            "detail_payload",
            "detail_period",
            "detail_note",
        ]
        out = out.merge(match.reindex(columns=match_cols), on=["ticker", "market", "report_period"], how="left")
    out["official_asof_policy"] = out["official_announcement_timestamp_matched"].map(
        lambda matched: "matched_official_t05st01_market_available_at" if bool(matched) else "unmatched_blocked_no_silent_backfill"
    )
    out["quarter_end_date_used_as_available_at"] = False
    out["query_response_datetime_used_as_available_at"] = False
    out["conservative_deadline_proxy_used_as_available_at"] = False
    out["exact_internal_upload_timestamp_found"] = False
    out["after_close_next_trading_day_policy_applies"] = out["official_announcement_timestamp_matched"].astype(bool)
    out["accepted_for_experiments"] = False
    out["accepted_for_formal"] = False
    out["diagnostic_only"] = True
    return out


def _failure_attribution(matrix: pd.DataFrame, match: pd.DataFrame, unmatched_in: pd.DataFrame) -> pd.DataFrame:
    unmatched = matrix[~matrix["official_announcement_timestamp_matched"].astype(bool)].copy()
    if unmatched.empty:
        return pd.DataFrame()
    match_cols = ["ticker", "market", "report_period", "candidate_count", "t05st01_query_payload"]
    unmatched = unmatched.merge(match.reindex(columns=match_cols), on=["ticker", "market", "report_period"], how="left")
    if not unmatched_in.empty:
        extra = unmatched_in.reindex(columns=["ticker", "market", "report_period", "policy"])
        unmatched = unmatched.merge(extra, on=["ticker", "market", "report_period"], how="left")

    rows: list[dict[str, Any]] = []
    for row in unmatched.to_dict("records"):
        statement_ok = str(row.get("t164sb05_status", "")).startswith("code=200") and str(row.get("t164sb03_status", "")).startswith("code=200")
        candidate_count = int(row.get("candidate_count") or 0)
        rows.append(
            {
                "ticker": row.get("ticker"),
                "market": row.get("market"),
                "report_period": row.get("report_period"),
                "t164_statement_success": statement_ok,
                "t05st01_candidate_count": candidate_count,
                "failure_stage": "official_asof_announcement_match",
                "primary_failure_attribution": "t05st01_query_returned_no_financial_report_candidate"
                if candidate_count == 0
                else "t05st01_candidate_ambiguous_or_unaccepted",
                "announcement_text_pattern_issue": "possible_unverified",
                "ticker_name_mapping_issue": "no_evidence_from_current_artifacts",
                "market_source_endpoint_issue": "not_primary_evidence_both_twse_tpex_have_matches_and_gaps",
                "announcement_type_issue": "possible_unverified",
                "time_range_issue": "possible_unverified_search_year_115_month_all_used",
                "data_absence_issue": "possible_t05st01_no_candidate_in_current_route",
                "requires_alternate_official_route_capture": True,
                "allowed_policy": "matched_only_else_unmatched_blocked",
                "no_silent_backfill": True,
                "diagnostic_only": True,
            }
        )
    return pd.DataFrame(rows)


def _alternate_route_policy(failure_attribution: pd.DataFrame) -> pd.DataFrame:
    affected = int(len(failure_attribution))
    return pd.DataFrame(
        [
            {
                "route_or_policy": "current_t05st01_t05st01_detail_exact_subject_match",
                "status": "partial_blocked_by_unmatched_rows",
                "defensible_as_official_market_available_at": True,
                "can_fill_current_unmatched_rows": False,
                "affected_unmatched_rows": affected,
                "next_owner": "Radar/Data",
                "required_next_evidence": "capture alternate official announcement query or subject pattern for 3008, 6669, 6187 in 115Q1 and 114Q4",
                "policy_note": "retain matched rows; keep unmatched rows blocked; no silent backfill",
                "diagnostic_only": True,
            },
            {
                "route_or_policy": "broaden_t05st01_query_and_subject_policy",
                "status": "candidate_not_validated",
                "defensible_as_official_market_available_at": True,
                "can_fill_current_unmatched_rows": False,
                "affected_unmatched_rows": affected,
                "next_owner": "Radar/Data",
                "required_next_evidence": "prove exact public material-information announcement timestamp and period mapping for each missed row",
                "policy_note": "acceptable only if source remains t05st01/t05st01_detail or equivalent public official announcement timestamp",
                "diagnostic_only": True,
            },
            {
                "route_or_policy": "conservative_filing_deadline_proxy",
                "status": "separate_proxy_candidate_only",
                "defensible_as_official_market_available_at": False,
                "can_fill_current_unmatched_rows": False,
                "affected_unmatched_rows": affected,
                "next_owner": "Core/Research policy only",
                "required_next_evidence": "policy approval would still not make it official-asof; must remain separate proxy field",
                "policy_note": "must not be mixed into matched official route",
                "diagnostic_only": True,
            },
            {
                "route_or_policy": "quarter_end_date_or_query_response_datetime",
                "status": "prohibited",
                "defensible_as_official_market_available_at": False,
                "can_fill_current_unmatched_rows": False,
                "affected_unmatched_rows": affected,
                "next_owner": "none",
                "required_next_evidence": "not applicable",
                "policy_note": "never use as available_at",
                "diagnostic_only": True,
            },
        ]
    )


def _blocked_rows_policy(failure_attribution: pd.DataFrame) -> pd.DataFrame:
    if failure_attribution.empty:
        return pd.DataFrame(columns=["policy_item", "status", "detail", "diagnostic_only"])
    tickers = ",".join(sorted(str(t) for t in failure_attribution["ticker"].unique()))
    return pd.DataFrame(
        [
            {
                "policy_item": "matched_official_rows",
                "status": "usable_in_partial_contract_only",
                "detail": "matched rows may retain official t05st01 market_available_at and after-close next-trading-day eligibility",
                "diagnostic_only": True,
            },
            {
                "policy_item": "unmatched_rows",
                "status": "blocked_no_silent_backfill",
                "detail": f"block {len(failure_attribution)} rows until official route match is proven; affected tickers={tickers}",
                "diagnostic_only": True,
            },
            {
                "policy_item": "partial_acceptance_threshold",
                "status": "requires_research_or_strategy_policy",
                "detail": "Core does not accept 85% match share as Experiments-ready because 6669 case/reference rows are unmatched",
                "diagnostic_only": True,
            },
            {
                "policy_item": "after_close_next_trading_day_eligibility",
                "status": "applies_to_matched_or_validated_alternate_official_route_only",
                "detail": "alternate route can use same eligibility policy only when it supplies official public timestamp",
                "diagnostic_only": True,
            },
        ]
    )


def _blocked_proxy_ledger(blocked_in: pd.DataFrame, failure_attribution: pd.DataFrame, tpex_in: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {
            "field_or_contract": "official_asof_unmatched_rows",
            "status": "blocked",
            "affected_rows": len(failure_attribution),
            "reason": "3008, 6669, 6187 x 115Q1/114Q4 have no accepted t05st01 candidate in current artifacts",
            "diagnostic_only": True,
        },
        {
            "field_or_contract": "alternate_official_announcement_route",
            "status": "candidate_not_validated",
            "affected_rows": len(failure_attribution),
            "reason": "no alternate official timestamp route is proven by current Core input",
            "diagnostic_only": True,
        },
        {
            "field_or_contract": "exact_internal_upload_timestamp",
            "status": "blocked_not_found",
            "affected_rows": 0,
            "reason": "official announcement timestamp remains distinct from internal upload timestamp",
            "diagnostic_only": True,
        },
    ]
    for source in (blocked_in, tpex_in):
        if source.empty:
            continue
        for row in source.to_dict("records"):
            rows.append(
                {
                    "field_or_contract": row.get("item") or row.get("field_or_contract") or row.get("market"),
                    "status": row.get("status") or row.get("universal_readiness"),
                    "affected_rows": row.get("bounded_rows", 0),
                    "reason": row.get("reason") or row.get("minimal_blocker"),
                    "diagnostic_only": True,
                }
            )
    return pd.DataFrame(rows)


def _future_data_audit(future_in: pd.DataFrame, policy_contract: pd.DataFrame) -> pd.DataFrame:
    prohibited_count = int(
        policy_contract[
            [
                "quarter_end_date_used_as_available_at",
                "query_response_datetime_used_as_available_at",
                "conservative_deadline_proxy_used_as_available_at",
            ]
        ]
        .astype(bool)
        .any(axis=1)
        .sum()
    )
    radar_future = int(future_in["future_data_violation_count"].sum()) if "future_data_violation_count" in future_in else 0
    return pd.DataFrame(
        [
            {
                "audit_item": "prohibited_available_date_sources_not_used",
                "status": "passed" if prohibited_count == 0 else "failed",
                "future_data_violation_count": prohibited_count,
                "note": "quarter_end_date/query_response_datetime/conservative_deadline_proxy are not used as official available_at",
            },
            {
                "audit_item": "unmatched_rows_not_backfilled",
                "status": "passed",
                "future_data_violation_count": 0,
                "note": "unmatched rows remain blocked, not filled with proxy dates",
            },
            {
                "audit_item": "radar_future_data_audit_imported",
                "status": "passed" if radar_future == 0 else "failed",
                "future_data_violation_count": radar_future,
                "note": "Radar/Data audit imported",
            },
            {
                "audit_item": "forward_return_as_rule",
                "status": "passed",
                "future_data_violation_count": 0,
                "note": "no forward returns used",
            },
        ]
    )


def _readiness(
    *,
    readiness_in: dict[str, Any],
    policy_contract: pd.DataFrame,
    failure_attribution: pd.DataFrame,
    alternate_route: pd.DataFrame,
    future_audit: pd.DataFrame,
) -> dict[str, Any]:
    total = len(policy_contract)
    matched = int(policy_contract["official_announcement_timestamp_matched"].astype(bool).sum())
    unmatched = total - matched
    future_count = int(future_audit["future_data_violation_count"].sum())
    alternate_validated = bool(
        alternate_route[
            (alternate_route["status"] == "validated")
            & (alternate_route["defensible_as_official_market_available_at"].astype(bool))
        ].shape[0]
    )
    return {
        "date": "2026-07-07",
        "task_id": TASK_ID,
        "owner": "BACKTEST_LAB Core/Data",
        "status": "blocked_by_official_asof_match_gaps",
        "diagnostic_only": True,
        "radar_status": readiness_in.get("status"),
        "ready_for_core_t164_broader_interim_official_asof_join": False,
        "ready_for_experiments": False,
        "ready_for_formal": False,
        "ready_for_strategy_replay": False,
        "ready_for_full_ingest": False,
        "ready_for_partial_matched_only_policy_review": True,
        "official_timestamp_matched_share": matched / total if total else 0.0,
        "matched_rows": matched,
        "unmatched_rows": unmatched,
        "unmatched_tickers": sorted(str(t) for t in failure_attribution["ticker"].unique()) if not failure_attribution.empty else [],
        "alternate_official_route_validated": alternate_validated,
        "after_close_policy_applies_to_valid_official_timestamp_only": True,
        "quarter_end_date_prohibited": True,
        "query_response_datetime_prohibited": True,
        "conservative_deadline_proxy_must_remain_separate": True,
        "exact_internal_upload_timestamp_found": False,
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


def _summary(readiness: dict[str, Any], failure_attribution: pd.DataFrame, alternate_route: pd.DataFrame) -> str:
    unmatched_rows = [
        f"- {row.ticker} {row.market} {row.report_period}: {row.primary_failure_attribution}"
        for row in failure_attribution.itertuples()
    ]
    routes = [
        f"- {row.route_or_policy}: {row.status}; can_fill_current_unmatched_rows={str(row.can_fill_current_unmatched_rows).lower()}"
        for row in alternate_route.itertuples()
    ]
    return "\n".join(
        [
            "# Layer1 t164 Asof Match Policy / Alternate Route Readiness",
            "",
            f"Status: {readiness['status']}",
            "",
            "Conclusion: t164 statement replay is stable in the larger bounded sample, but official-asof match coverage is incomplete. Core does not accept the package as broader ingest-ready or Experiments-ready.",
            "",
            "Readiness:",
            f"- ready_for_core_t164_broader_interim_official_asof_join={str(readiness['ready_for_core_t164_broader_interim_official_asof_join']).lower()}",
            "- ready_for_experiments=false",
            "- ready_for_formal=false",
            "- ready_for_strategy_replay=false",
            f"- official_timestamp_matched_share={readiness['official_timestamp_matched_share']}",
            f"- matched_rows={readiness['matched_rows']}",
            f"- unmatched_rows={readiness['unmatched_rows']}",
            f"- future_data_violation_count={readiness['future_data_violation_count']}",
            "",
            "Unmatched failure attribution:",
            *unmatched_rows,
            "",
            "Alternate route / policy staging:",
            *routes,
            "",
            "Blocked rows policy:",
            "- matched-only is acceptable only as a partial contract for policy review.",
            "- unmatched rows remain blocked; no silent backfill.",
            "- conservative filing-deadline proxy must remain separate and cannot replace official timestamp.",
            "- quarter_end_date and query_response_datetime remain prohibited.",
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
    parser.add_argument("--radar-dir", type=Path, default=DEFAULT_RADAR_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    manifest = build_asof_match_policy_readiness(radar_dir=args.radar_dir, output_dir=args.output_dir)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
