"""Build bounded broader Layer1 t164 ingest contract from pruning v2 seed.

This is a source/contract build only. It does not run broader materialization,
Experiments, replay, formal model, report, or trade-decision changes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


TASK_ID = "TASK-BACKTEST-CORE-VNEXT-LAYER1-T164-BOUNDED-BROADER-INGEST-CONTRACT-BUILD-001"
DEFAULT_RADAR_DIR = Path(
    r"C:\Users\zergv\Documents\Codex\2026-05-23\ai-stock-rotation-radar-https-docs\outputs"
    r"\radar_vnext_layer1_t164_candidate_detail_pruning_runner_v2_20260707"
)
DEFAULT_CALENDAR_DIR = Path("outputs/vnext_dynamic_candidate_pool_data_materialization_20260706")
DEFAULT_OUTPUT_DIR = Path("outputs/vnext_layer1_t164_bounded_broader_ingest_contract_20260707")


def build_contract(
    *,
    radar_dir: str | Path = DEFAULT_RADAR_DIR,
    calendar_dir: str | Path = DEFAULT_CALENDAR_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    radar = Path(radar_dir)
    calendar_path = Path(calendar_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    readiness_in = _read_json(radar / "readiness_for_core_t164_candidate_detail_pruning_runner_v2.json")
    matrix = _read_csv(radar / "t164_materialized_field_matrix.csv", dtype={"ticker": str})
    asof = _read_csv(radar / "official_asof_candidate_ledger.csv", dtype={"ticker": str})
    coverage_market_period = _read_csv(radar / "coverage_by_market_period.csv")
    coverage_field = _read_csv(radar / "coverage_by_field.csv")
    route_cost = _read_csv(radar / "projected_route_cost_report.csv")
    future_in = _read_csv(radar / "future_data_governance_audit.csv")
    calendar = _read_csv(calendar_path / "trading_calendar.csv")

    contract = _bounded_contract(matrix, asof, calendar)
    schema = _schema()
    field_policy = _field_policy(coverage_field)
    join_policy = _join_policy()
    runner_input = _runner_input_contract(contract, route_cost)
    coverage_audit = _coverage_audit_design(coverage_market_period, coverage_field)
    future_governance = _future_governance(future_in, contract)
    blockers = _blockers()
    readiness = _readiness(readiness_in, contract, route_cost, future_governance)

    _write_csv(contract, output / "layer1_t164_bounded_broader_ingest_contract.csv")
    _write_csv(schema, output / "layer1_t164_bounded_broader_ingest_schema.csv")
    _write_csv(field_policy, output / "layer1_t164_field_policy.csv")
    _write_csv(join_policy, output / "layer1_t164_t05st01_official_asof_join_policy.csv")
    _write_csv(runner_input, output / "layer1_t164_runner_input_contract.csv")
    _write_csv(coverage_audit, output / "layer1_t164_coverage_audit_design.csv")
    _write_csv(future_governance, output / "layer1_t164_future_data_governance.csv")
    _write_csv(blockers, output / "layer1_t164_bounded_contract_blockers.csv")
    (output / "readiness_for_layer1_t164_bounded_broader_ingest_contract.json").write_text(
        json.dumps(readiness, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    manifest = {
        "task_id": TASK_ID,
        "status": readiness["status"],
        "output_dir": str(output.resolve()),
        "radar_input_dir": str(radar.resolve()),
        "radar_commit": "542d529",
        "output_files": [
            "layer1_t164_bounded_broader_ingest_contract.csv",
            "layer1_t164_bounded_broader_ingest_schema.csv",
            "layer1_t164_field_policy.csv",
            "layer1_t164_t05st01_official_asof_join_policy.csv",
            "layer1_t164_runner_input_contract.csv",
            "layer1_t164_coverage_audit_design.csv",
            "layer1_t164_future_data_governance.csv",
            "layer1_t164_bounded_contract_blockers.csv",
            "readiness_for_layer1_t164_bounded_broader_ingest_contract.json",
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
    if not path.exists() or path.read_text(encoding="utf-8").strip() == "empty":
        return pd.DataFrame()
    return pd.read_csv(path, **kwargs)


def _bounded_contract(matrix: pd.DataFrame, asof: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    out = matrix.merge(
        asof[
            [
                "ticker",
                "market",
                "report_period",
                "match_status",
                "market_available_at",
                "accepted_subject",
                "accepted_match_status",
                "blocked_reason",
                "after_close_next_trading_day_policy",
                "quarter_end_date_used",
                "query_response_datetime_used",
                "conservative_deadline_proxy_used",
            ]
        ],
        on=["ticker", "market", "report_period"],
        how="left",
    )
    out["official_market_available_at"] = out["market_available_at"]
    out["official_market_available_at_iso"] = out["official_market_available_at"].map(_roc_datetime_to_iso)
    out["official_market_available_date"] = pd.to_datetime(out["official_market_available_at_iso"]).dt.date.astype(str)
    out["official_market_available_time"] = pd.to_datetime(out["official_market_available_at_iso"]).dt.time.astype(str)
    out["after_close_policy_applies"] = out["official_market_available_time"] >= "13:30:00"
    out = _join_signal_eligible_date(out, calendar)

    out["current_ratio"] = out["current_assets"] / out["current_liabilities"]
    for field in [
        "operating_cash_flow",
        "investing_cash_flow",
        "capex_proxy",
        "inventory",
        "receivables_trade",
        "current_assets",
        "current_liabilities",
        "current_ratio",
    ]:
        out[f"{field}_available"] = out[field].notna()
    out["cashflow_source_quality"] = "exact_pit_after_official_asof_join"
    out["inventory_source_quality"] = "exact_pit_after_official_asof_join"
    out["current_ratio_source_quality"] = "derived_pit_after_official_asof_join"
    out["capex_proxy_source_quality"] = "human_review_proxy_label_required"
    out["receivables_trade_source_quality"] = "human_review_proxy_label_required"
    out["source_scope"] = "bounded_pruning_v2_seed_20_tickers_2_periods"
    out["full_universe"] = False
    out["full_period_range"] = False
    out["accepted_for_experiments"] = False
    out["accepted_for_formal"] = False
    out["diagnostic_only"] = True
    out["not_live_rule"] = True
    out["forward_returns_live_rule_usage"] = False
    keep = [
        "ticker",
        "market",
        "report_period",
        "source_scope",
        "t164sb05_status",
        "t164sb03_status",
        "official_asof_match_status",
        "match_status",
        "official_market_available_at",
        "official_market_available_at_iso",
        "signal_eligible_date",
        "signal_eligible_date_policy",
        "accepted_subject",
        "accepted_match_status",
        "blocked_reason",
        "operating_cash_flow",
        "investing_cash_flow",
        "capex_proxy",
        "inventory",
        "receivables_trade",
        "current_assets",
        "current_liabilities",
        "current_ratio",
        "operating_cash_flow_available",
        "investing_cash_flow_available",
        "capex_proxy_available",
        "inventory_available",
        "receivables_trade_available",
        "current_assets_available",
        "current_liabilities_available",
        "current_ratio_available",
        "cashflow_source_quality",
        "inventory_source_quality",
        "current_ratio_source_quality",
        "capex_proxy_source_quality",
        "receivables_trade_source_quality",
        "quarter_end_date_used",
        "query_response_datetime_used",
        "conservative_deadline_proxy_used",
        "after_close_policy_applies",
        "full_universe",
        "full_period_range",
        "accepted_for_experiments",
        "accepted_for_formal",
        "diagnostic_only",
        "not_live_rule",
        "forward_returns_live_rule_usage",
    ]
    return out.reindex(columns=keep)


def _join_signal_eligible_date(out: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    cal = calendar[["trade_date", "next_trade_date"]].copy()
    cal["trade_date"] = cal["trade_date"].astype(str)
    out = out.merge(cal, left_on="official_market_available_date", right_on="trade_date", how="left")
    out["signal_eligible_date"] = out["official_market_available_date"]
    out.loc[out["after_close_policy_applies"], "signal_eligible_date"] = out.loc[out["after_close_policy_applies"], "next_trade_date"]
    out["signal_eligible_date_policy"] = out["after_close_policy_applies"].map(
        {True: "after_close_next_trading_day", False: "same_trading_day_if_market_open_and_timestamp_public"}
    )
    return out.drop(columns=["trade_date", "next_trade_date"], errors="ignore")


def _roc_datetime_to_iso(value: str) -> str:
    date_part, time_part = str(value).split(" ")
    roc_year, month, day = date_part.split("/")
    return f"{int(roc_year) + 1911:04d}-{int(month):02d}-{int(day):02d}T{time_part}"


def _schema() -> pd.DataFrame:
    rows = [
        ("ticker", "string", "join key", "exact"),
        ("market", "TWSE/TPEx", "market route key", "exact"),
        ("report_period", "ROC quarter/annual period label", "financial report period", "exact"),
        ("official_market_available_at", "ROC datetime", "official t05st01/t05st01_detail public timestamp", "exact_official_asof"),
        ("signal_eligible_date", "YYYY-MM-DD", "after-close adjusted trading eligibility date", "derived_policy"),
        ("operating_cash_flow", "numeric", "t164sb05 cashflow field", "exact_pit"),
        ("investing_cash_flow", "numeric", "t164sb05 cashflow field", "exact_pit"),
        ("capex_proxy", "numeric", "cashflow capex-like proxy", "human_review_proxy"),
        ("inventory", "numeric", "t164sb03 balance-sheet field", "exact_pit"),
        ("receivables_trade", "numeric", "receivables basket/proxy field", "human_review_proxy"),
        ("current_assets", "numeric", "t164sb03 balance-sheet field", "exact_pit"),
        ("current_liabilities", "numeric", "t164sb03 balance-sheet field", "exact_pit"),
        ("current_ratio", "numeric", "current_assets/current_liabilities", "derived_pit"),
        ("blocked_reason", "string", "blocked/unmatched/ambiguous reason", "contract_hygiene"),
    ]
    return pd.DataFrame(rows, columns=["column", "type", "meaning", "source_quality"]).assign(diagnostic_only=True)


def _field_policy(coverage_field: pd.DataFrame) -> pd.DataFrame:
    policy = {
        "operating_cash_flow": ("accepted", "exact_pit_after_official_asof_join"),
        "investing_cash_flow": ("accepted", "exact_pit_after_official_asof_join"),
        "capex_proxy": ("accepted_proxy_human_review_required", "proxy_not_formal_fcf"),
        "inventory": ("accepted", "exact_pit_after_official_asof_join"),
        "receivables_trade": ("accepted_proxy_human_review_required", "receivables_basket_proxy"),
        "current_assets": ("accepted", "exact_pit_after_official_asof_join"),
        "current_liabilities": ("accepted", "exact_pit_after_official_asof_join"),
        "current_ratio": ("accepted_derived", "derived_pit_current_assets_div_current_liabilities"),
    }
    rows = []
    coverage = {row["field"]: row for row in coverage_field.to_dict("records")} if not coverage_field.empty else {}
    for field, (status, quality) in policy.items():
        cov = coverage.get(field, {})
        rows.append(
            {
                "field": field,
                "policy_status": status,
                "source_quality": quality,
                "missing_rows": cov.get("missing_rows"),
                "missing_share": cov.get("missing_share"),
                "formal_ready": False,
                "diagnostic_only": True,
            }
        )
    return pd.DataFrame(rows)


def _join_policy() -> pd.DataFrame:
    rows = [
        ("subject_detail_matching", "required", "accepted subject must be financial-report approval/pass and detail period must map to target report period"),
        ("detail_period_mapping", "required", "detail text/reporting period validates quarter or annual date range"),
        ("after_close_next_trading_day", "required", "timestamp >= 13:30 uses next trading day for signal eligibility"),
        ("unmatched_rows", "blocked", "no silent fill; write blocked reason"),
        ("ambiguous_rows", "blocked", "no first/last candidate default; require disambiguation"),
        ("quarter_end_date", "prohibited", "not available_at"),
        ("query_response_datetime", "prohibited", "not available_at"),
        ("conservative_deadline_proxy", "separate_proxy_only", "not official route"),
    ]
    return pd.DataFrame(rows, columns=["policy_item", "status", "detail"]).assign(diagnostic_only=True)


def _runner_input_contract(contract: pd.DataFrame, route_cost: pd.DataFrame) -> pd.DataFrame:
    row = route_cost.iloc[0].to_dict() if not route_cost.empty else {}
    return pd.DataFrame(
        [
            {
                "runner_scope": "bounded_pruning_v2_seed",
                "ticker_count": int(contract["ticker"].nunique()),
                "period_count": int(contract["report_period"].nunique()),
                "markets": ",".join(sorted(contract["market"].dropna().unique())),
                "sample_rows": len(contract),
                "projected_routes_per_row": row.get("projected_routes_per_row"),
                "budget_routes_per_row": row.get("budget_routes_per_row"),
                "budget_status": row.get("budget_status"),
                "full_universe": False,
                "full_period_range": False,
                "diagnostic_only": True,
            }
        ]
    )


def _coverage_audit_design(coverage_market_period: pd.DataFrame, coverage_field: pd.DataFrame) -> pd.DataFrame:
    rows = [
        ("ticker", "ticker", "requested_rows, materialized_rows, statement_success, official_asof_match, blocked_reason"),
        ("market", "market", "requested_rows, materialized_rows, statement_success_rows, official_asof_matched_rows, blocked_rows"),
        ("period", "report_period", "requested_rows, materialized_rows, statement_success_rows, official_asof_matched_rows, blocked_rows"),
        ("field", "field", "requested_rows, non_null_rows, missing_rows, missing_share, source_quality"),
        ("future_data", "policy", "quarter_end_used, query_response_used, deadline_proxy_used, forward_return_as_rule"),
    ]
    return pd.DataFrame(rows, columns=["audit_axis", "group_by", "required_metrics"]).assign(diagnostic_only=True)


def _future_governance(future_in: pd.DataFrame, contract: pd.DataFrame) -> pd.DataFrame:
    prohibited = int(
        contract[["quarter_end_date_used", "query_response_datetime_used", "conservative_deadline_proxy_used"]]
        .astype(bool)
        .any(axis=1)
        .sum()
    )
    rows = [
        ("quarter_end_date", "prohibited", prohibited, "not used as official available_at"),
        ("query_response_datetime", "prohibited", prohibited, "not used as official available_at"),
        ("conservative_deadline_proxy", "separate_proxy_only", prohibited, "not mixed into official route"),
        ("forward_return_as_rule", "false_required", 0, "no forward returns included"),
        ("official_market_available_at", "required", 0, "must equal t05st01/t05st01_detail public timestamp"),
    ]
    return pd.DataFrame(rows, columns=["governance_item", "policy", "future_data_violation_count", "detail"]).assign(diagnostic_only=True)


def _blockers() -> pd.DataFrame:
    rows = [
        ("TPEx all-stock proof not complete", "blocked", "full universe false"),
        ("full period range not complete", "blocked", "bounded seed has only 115Q1 and 114Q4"),
        ("full universe false", "blocked", "20 ticker seed only"),
        ("capex_proxy human-review proxy policy required", "human_review_required", "not formal FCF"),
        ("receivables_trade human-review proxy policy required", "human_review_required", "receivables basket/proxy"),
    ]
    return pd.DataFrame(rows, columns=["blocker", "status", "detail"]).assign(diagnostic_only=True)


def _readiness(readiness_in: dict[str, Any], contract: pd.DataFrame, route_cost: pd.DataFrame, future_governance: pd.DataFrame) -> dict[str, Any]:
    future_count = int(future_governance["future_data_violation_count"].sum())
    all_matched = contract["official_asof_match_status"].eq("accepted").all() and contract["match_status"].eq("accepted").all()
    budget_pass = True
    if not route_cost.empty:
        budget_pass = route_cost.iloc[0].get("budget_status") == "pass"
    ready_bounded = bool(all_matched and budget_pass and future_count == 0 and len(contract) == int(readiness_in.get("sample_rows", len(contract))))
    return {
        "date": "2026-07-07",
        "task_id": TASK_ID,
        "owner": "BACKTEST_LAB Core/Data",
        "status": "bounded_broader_ingest_contract_built_not_materialized",
        "radar_status": readiness_in.get("status"),
        "diagnostic_only": True,
        "sample_rows": len(contract),
        "ticker_count": int(contract["ticker"].nunique()),
        "period_count": int(contract["report_period"].nunique()),
        "markets": sorted(contract["market"].dropna().unique().tolist()),
        "official_asof_matched_rows": int(contract["match_status"].eq("accepted").sum()),
        "statement_success_rows": int(contract["t164sb05_status"].astype(str).str.startswith("code=200").sum()),
        "ready_for_core_t164_bounded_broader_materialization": ready_bounded,
        "ready_for_radar_tpex_all_stock_proof_if_needed": True,
        "ready_for_experiments": False,
        "ready_for_formal": False,
        "ready_for_strategy_replay": False,
        "ready_for_full_universe": False,
        "blocked_fields": [
            "TPEx all-stock proof not complete",
            "full period range not complete",
            "full universe false",
            "capex_proxy / receivables_trade human-review proxy policy required",
        ],
        "future_data_violation_count": future_count,
        "not_live_rule": True,
        "forward_returns_live_rule_usage": False,
        "formal_model_changed": False,
        "trade_decision_changed": False,
        "active_in_trade_decision": False,
        "report_changed": False,
        "portfolio_replay_executed": False,
    }


def _summary(readiness: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Layer1 t164 Bounded Broader Ingest Contract",
            "",
            f"Status: {readiness['status']}",
            "",
            "Conclusion: Core built a bounded broader ingest contract from the pruning v2 seed. This is contract readiness only, not broader materialization, not Experiments-ready, and not formal-ready.",
            "",
            "Readiness:",
            f"- ready_for_core_t164_bounded_broader_materialization={str(readiness['ready_for_core_t164_bounded_broader_materialization']).lower()}",
            "- ready_for_experiments=false",
            "- ready_for_formal=false",
            "- ready_for_strategy_replay=false",
            "- ready_for_full_universe=false",
            f"- sample_rows={readiness['sample_rows']}",
            f"- ticker_count={readiness['ticker_count']}",
            f"- period_count={readiness['period_count']}",
            f"- official_asof_matched_rows={readiness['official_asof_matched_rows']}",
            f"- future_data_violation_count={readiness['future_data_violation_count']}",
            "",
            "Retained blockers:",
            *[f"- {item}" for item in readiness["blocked_fields"]],
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


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--radar-dir", type=Path, default=DEFAULT_RADAR_DIR)
    parser.add_argument("--calendar-dir", type=Path, default=DEFAULT_CALENDAR_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    manifest = build_contract(radar_dir=args.radar_dir, calendar_dir=args.calendar_dir, output_dir=args.output_dir)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
