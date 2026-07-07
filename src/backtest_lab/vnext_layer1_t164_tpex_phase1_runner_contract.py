"""Build Layer1 t164 TPEx phase-1 bounded proof runner contract.

This is a source/contract readiness package only. It does not execute the
runner, materialize t164 data, run Experiments, replay portfolios, or change
formal model/report/trade-decision paths.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


TASK_ID = "TASK-BACKTEST-CORE-VNEXT-LAYER1-T164-TPEX-PHASE1-BOUNDED-PROOF-RUNNER-CONTRACT-001"
DEFAULT_RADAR_DIR = Path(
    r"C:\Users\zergv\Documents\Codex\2026-05-23\ai-stock-rotation-radar-https-docs\outputs"
    r"\radar_vnext_layer1_t164_tpex_all_stock_proof_full_period_bounded_expansion_plan_20260707"
)
DEFAULT_OUTPUT_DIR = Path("outputs/vnext_layer1_t164_tpex_phase1_bounded_proof_runner_contract_20260707")


def build_contract(
    *,
    radar_dir: str | Path = DEFAULT_RADAR_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    radar = Path(radar_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    readiness_in = _read_json(radar / "readiness_for_core_t164_tpex_all_stock_full_period_expansion_plan.json")
    sample_policy = _read_csv(radar / "tpex_all_stock_proof_sample_policy.csv", dtype={"ticker": str})
    batch_plan = _read_csv(radar / "full_period_bounded_expansion_batch_plan.csv")
    cost_estimate = _read_csv(radar / "full_period_bounded_expansion_cost_estimate.csv")
    coverage_summary = _read_csv(radar / "source_route_proof_coverage_summary.csv")
    expansion_items = _read_csv(radar / "core_contract_expansion_items.csv")
    blocked_in = _read_csv(radar / "blocked_proxy_human_review_ledger.csv")
    future_in = _read_csv(radar / "future_data_governance_audit.csv")

    runner_contract = _phase1_runner_contract(sample_policy)
    runner_input = _runner_input_contract(readiness_in, runner_contract)
    route_budget = _route_budget_guard(batch_plan, cost_estimate)
    asof_policy = _official_asof_policy()
    field_policy = _field_policy()
    coverage_audit = _coverage_audit_design(coverage_summary)
    full_period_guard = _full_period_guard(batch_plan)
    blocked_ledger = _blocked_proxy_ledger(blocked_in, expansion_items)
    future_audit = _future_data_audit(future_in)
    readiness = _readiness(readiness_in, runner_contract, route_budget, future_audit)

    _write_csv(runner_contract, output / "layer1_t164_tpex_phase1_50x2_runner_contract.csv")
    _write_csv(runner_input, output / "layer1_t164_tpex_phase1_runner_input_contract.csv")
    _write_csv(route_budget, output / "layer1_t164_tpex_phase1_route_budget_guard.csv")
    _write_csv(asof_policy, output / "layer1_t164_tpex_official_asof_join_policy.csv")
    _write_csv(field_policy, output / "layer1_t164_tpex_field_policy.csv")
    _write_csv(coverage_audit, output / "layer1_t164_tpex_phase1_coverage_audit_design.csv")
    _write_csv(full_period_guard, output / "layer1_t164_tpex_full_period_expansion_guard.csv")
    _write_csv(blocked_ledger, output / "layer1_t164_tpex_blocked_proxy_human_review_ledger.csv")
    _write_csv(future_audit, output / "layer1_t164_tpex_future_data_governance_audit.csv")
    (output / "readiness_for_layer1_t164_tpex_phase1_runner_contract.json").write_text(
        json.dumps(readiness, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    manifest = {
        "task_id": TASK_ID,
        "status": readiness["status"],
        "output_dir": str(output.resolve()),
        "radar_input_dir": str(radar.resolve()),
        "radar_commit": "3217ea2",
        "output_files": [
            "layer1_t164_tpex_phase1_50x2_runner_contract.csv",
            "layer1_t164_tpex_phase1_runner_input_contract.csv",
            "layer1_t164_tpex_phase1_route_budget_guard.csv",
            "layer1_t164_tpex_official_asof_join_policy.csv",
            "layer1_t164_tpex_field_policy.csv",
            "layer1_t164_tpex_phase1_coverage_audit_design.csv",
            "layer1_t164_tpex_full_period_expansion_guard.csv",
            "layer1_t164_tpex_blocked_proxy_human_review_ledger.csv",
            "layer1_t164_tpex_future_data_governance_audit.csv",
            "readiness_for_layer1_t164_tpex_phase1_runner_contract.json",
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


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def _phase1_runner_contract(sample_policy: pd.DataFrame) -> pd.DataFrame:
    phase1 = sample_policy.head(50).copy()
    rows: list[dict[str, Any]] = []
    for item in phase1.to_dict("records"):
        periods = [p.strip() for p in str(item["target_periods"]).split(";") if p.strip()]
        for period in periods:
            rows.append(
                {
                    "phase": "phase_1_tpex_stratified_50x2",
                    "ticker": item["ticker"],
                    "name": item.get("name"),
                    "market": "TPEx",
                    "report_period": period,
                    "sample_policy": item.get("sample_policy"),
                    "source_universe_quality": item.get("source_universe_quality"),
                    "historical_pit_universe_ready": False,
                    "prior_t164_seed_status": item.get("prior_t164_seed_status"),
                    "planned_routes": item.get("planned_routes"),
                    "expected_route_budget_per_ticker_period": item.get("expected_route_budget_per_ticker_period"),
                    "official_asof_required": True,
                    "after_close_next_trading_day_policy_required": True,
                    "unmatched_or_ambiguous_policy": "blocked_no_silent_backfill",
                    "quarter_end_date_available_at_allowed": False,
                    "query_response_datetime_available_at_allowed": False,
                    "conservative_deadline_proxy_allowed_for_official_route": False,
                    "accepted_for_runner_execution": True,
                    "accepted_for_materialization": False,
                    "accepted_for_experiments": False,
                    "accepted_for_formal": False,
                    "diagnostic_only": True,
                    "not_live_rule": True,
                    "forward_returns_live_rule_usage": False,
                    "formal_model_changed": False,
                    "trade_decision_changed": False,
                    "active_in_trade_decision": False,
                    "report_changed": False,
                    "portfolio_replay_executed": False,
                    "ready_for_strategy_replay": False,
                }
            )
    return pd.DataFrame(rows)


def _runner_input_contract(readiness: dict[str, Any], runner_contract: pd.DataFrame) -> pd.DataFrame:
    coverage = readiness.get("coverage", {})
    return pd.DataFrame(
        [
            {
                "input_item": "phase_1_ticker_period_rows",
                "value": len(runner_contract),
                "source": "tpex_all_stock_proof_sample_policy.csv first 50 tickers x target periods",
                "source_quality": "bounded_sampling_contract",
                "diagnostic_only": True,
            },
            {
                "input_item": "tpex_current_or_carried_universe_candidate_count",
                "value": coverage.get("tpex_current_or_carried_universe_candidate_count"),
                "source": "Radar current-or-carried TPEx universe candidate",
                "source_quality": "sampling_universe_only_not_historical_pit",
                "diagnostic_only": True,
            },
            {
                "input_item": "planned_phase_1_sample_ticker_count",
                "value": coverage.get("planned_phase_1_sample_ticker_count"),
                "source": "Radar readiness",
                "source_quality": "bounded_plan",
                "diagnostic_only": True,
            },
            {
                "input_item": "planned_full_period_range",
                "value": coverage.get("planned_full_period_range"),
                "source": "Radar readiness",
                "source_quality": "planning_only_requires_checkpoint_for_execution",
                "diagnostic_only": True,
            },
        ]
    )


def _route_budget_guard(batch_plan: pd.DataFrame, cost_estimate: pd.DataFrame) -> pd.DataFrame:
    source = batch_plan if not batch_plan.empty else cost_estimate
    rows: list[dict[str, Any]] = []
    for item in source.to_dict("records"):
        phase = item.get("phase")
        rows.append(
            {
                "phase": phase,
                "ticker_count": item.get("ticker_count"),
                "period_count": item.get("period_count"),
                "ticker_period_rows": item.get("ticker_period_rows"),
                "projected_routes_per_row": item.get("projected_routes_per_row"),
                "projected_total_routes": item.get("projected_total_routes"),
                "route_budget_guard_per_row": item.get("route_budget_guard_per_row"),
                "budget_status": item.get("budget_status"),
                "checkpoint_required": item.get("checkpoint_required"),
                "allowed_by_core_contract_now": phase == "phase_1_tpex_stratified_50x2",
                "blocked_reason": ""
                if phase == "phase_1_tpex_stratified_50x2"
                else "not authorized in this Core contract; requires separate Radar/Core expansion decision",
                "diagnostic_only": True,
                "not_live_rule": True,
            }
        )
    return pd.DataFrame(rows)


def _official_asof_policy() -> pd.DataFrame:
    rows = [
        (
            "market_available_at",
            "must_equal_t05st01_or_t05st01_detail_public_financial_report_announcement_timestamp",
            "required",
        ),
        ("after_close_eligibility", "if public timestamp is after regular close, eligible signal date is next trading day", "required"),
        ("unmatched_or_ambiguous", "blocked_no_silent_backfill", "required"),
        ("quarter_end_date", "prohibited_as_available_at", "required"),
        ("query_response_datetime", "prohibited_as_available_at", "required"),
        ("conservative_deadline_proxy", "separate_diagnostic_candidate_only_not_official_route_backfill", "required"),
        ("exact_internal_upload_timestamp", "not_found_not_required_for_market_available_at_but_must_remain_explicit", "required"),
    ]
    return pd.DataFrame(rows, columns=["policy_item", "policy", "status"]).assign(diagnostic_only=True)


def _field_policy() -> pd.DataFrame:
    rows = [
        ("operating_cash_flow", "accepted_if_t164sb05_statement_success_and_official_asof_matched", "exact_pit_after_official_asof_join"),
        ("investing_cash_flow", "accepted_if_t164sb05_statement_success_and_official_asof_matched", "exact_pit_after_official_asof_join"),
        ("capex_proxy", "accepted_proxy_human_review_required_not_formal", "diagnostic_proxy_human_review_required"),
        ("inventory", "accepted_if_t164sb03_statement_success_and_official_asof_matched", "exact_pit_after_official_asof_join"),
        ("receivables_trade", "accepted_proxy_human_review_required_not_formal", "diagnostic_proxy_human_review_required"),
        ("current_assets", "accepted_if_t164sb03_statement_success_and_official_asof_matched", "exact_pit_after_official_asof_join"),
        ("current_liabilities", "accepted_if_t164sb03_statement_success_and_official_asof_matched", "exact_pit_after_official_asof_join"),
        ("current_ratio", "derive_current_assets_div_current_liabilities_after_official_asof_join", "derived_pit_after_official_asof_join"),
    ]
    return pd.DataFrame(rows, columns=["field", "policy_status", "source_quality"]).assign(
        accepted_for_formal=False,
        diagnostic_only=True,
    )


def _coverage_audit_design(coverage_summary: pd.DataFrame) -> pd.DataFrame:
    rows = coverage_summary.to_dict("records") if not coverage_summary.empty else []
    rows.append(
        {
            "scope": "phase_1_tpex_stratified_50x2_runner_contract",
            "ticker_universe_count": "891_source_candidate",
            "sample_ticker_count": 50,
            "period_count": 2,
            "materialized_rows": "pending_runner",
            "statement_success_rows": "pending_runner",
            "official_asof_matched_rows": "pending_runner",
            "blocked_rows": "pending_runner",
            "success_share": "pending_runner",
            "all_stock_universal_ready": False,
            "readiness_label": "contract_ready_not_executed",
            "blocked_reason": "bounded runner execution required before TPEx all-stock proof can be claimed",
            "formal_model_changed": False,
            "trade_decision_changed": False,
            "active_in_trade_decision": False,
            "report_changed": False,
            "portfolio_replay_executed": False,
            "ready_for_strategy_replay": False,
            "not_live_rule": True,
            "forward_returns_live_rule_usage": False,
        }
    )
    return pd.DataFrame(rows)


def _full_period_guard(batch_plan: pd.DataFrame) -> pd.DataFrame:
    if batch_plan.empty:
        return pd.DataFrame()
    out = batch_plan.copy()
    out["core_contract_decision"] = out["phase"].map(
        lambda phase: "allowed_phase1_only"
        if phase == "phase_1_tpex_stratified_50x2"
        else "blocked_requires_separate_checkpoint_resume_runner_contract"
    )
    out["full_universe_ready"] = False
    out["full_period_ready"] = False
    out["ready_for_experiments"] = False
    out["diagnostic_only"] = True
    return out


def _blocked_proxy_ledger(blocked_in: pd.DataFrame, expansion_items: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if not blocked_in.empty:
        for item in blocked_in.to_dict("records"):
            rows.append(
                {
                    "item": item.get("item") or item.get("field") or item.get("contract_item"),
                    "status": item.get("status") or item.get("source_quality") or "proxy_or_blocked",
                    "blocked_reason": item.get("blocked_reason") or item.get("reason") or item.get("policy"),
                    "source": "Radar blocked_proxy_human_review_ledger.csv",
                    "diagnostic_only": True,
                }
            )
    if not expansion_items.empty:
        for item in expansion_items.to_dict("records"):
            if str(item.get("pit_ready")) == "False" or item.get("blocked_reason"):
                rows.append(
                    {
                        "item": item.get("contract_item"),
                        "status": "blocked_or_proxy",
                        "blocked_reason": item.get("blocked_reason"),
                        "source": "Radar core_contract_expansion_items.csv",
                        "diagnostic_only": True,
                    }
                )
    rows.extend(
        [
            {
                "item": "tpex_historical_all_stock_universe",
                "status": "blocked",
                "blocked_reason": "current-or-carried TPEx universe candidate cannot be backfilled as historical PIT membership",
                "source": "Core policy",
                "diagnostic_only": True,
            },
            {
                "item": "full_current_snapshot_tpex_x_full_period",
                "status": "blocked_for_this_contract",
                "blocked_reason": "891 x 46 plan needs checkpoint/resume/budget guard and separate authorization",
                "source": "Core policy",
                "diagnostic_only": True,
            },
        ]
    )
    return pd.DataFrame(rows)


def _future_data_audit(future_in: pd.DataFrame) -> pd.DataFrame:
    rows = future_in.to_dict("records") if not future_in.empty else []
    rows.extend(
        [
            {
                "audit_item": "phase1_contract_forward_return_as_rule",
                "status": "passed",
                "future_data_violation_count": 0,
                "note": "runner contract contains no forward-return rule input",
            },
            {
                "audit_item": "available_at_source_policy",
                "status": "passed",
                "future_data_violation_count": 0,
                "note": "quarter_end_date/query_response_datetime/conservative deadline proxy are prohibited as official route available_at",
            },
        ]
    )
    return pd.DataFrame(rows)


def _readiness(
    readiness_in: dict[str, Any],
    runner_contract: pd.DataFrame,
    route_budget: pd.DataFrame,
    future_audit: pd.DataFrame,
) -> dict[str, Any]:
    phase1_budget = route_budget[route_budget["phase"].eq("phase_1_tpex_stratified_50x2")]
    budget_pass = bool(phase1_budget["budget_status"].eq("pass_planning").all()) if not phase1_budget.empty else False
    future_violations = int(pd.to_numeric(future_audit.get("future_data_violation_count", 0), errors="coerce").fillna(0).sum())
    return {
        "task_id": TASK_ID,
        "status": "phase1_tpex_50x2_runner_contract_ready_not_executed",
        "diagnostic_only": True,
        "runner_contract_rows": int(len(runner_contract)),
        "phase1_ticker_count": int(runner_contract["ticker"].nunique()),
        "phase1_period_count": int(runner_contract["report_period"].nunique()),
        "market": "TPEx",
        "planned_routes_per_row": 8.0,
        "route_budget_guard_per_row": 10.0,
        "phase1_budget_status": "pass_planning" if budget_pass else "blocked",
        "ready_for_radar_phase1_tpex_50x2_runner_execution": bool(budget_pass and future_violations == 0),
        "ready_for_core_t164_tpex_all_stock_proof_runner_contract": True,
        "ready_for_core_t164_full_period_bounded_expansion_contract": False,
        "ready_for_core_t164_broader_or_full_ingest_contract": False,
        "ready_for_core_t164_broader_or_full_materialization": False,
        "ready_for_experiments": False,
        "ready_for_formal": False,
        "ready_for_strategy_replay": False,
        "ready_for_full_universe": False,
        "tpex_all_stock_universal_ready": False,
        "future_data_violation_count": future_violations,
        "blocked_fields": [
            "tpex_historical_all_stock_universe",
            "full_period_materialization",
            "full_universe_materialization",
            "capex_proxy_formal_label",
            "receivables_trade_formal_label",
        ],
        "proxy_fields": ["capex_proxy", "receivables_trade", "current_or_carried_tpex_universe_candidate"],
        "source_package_status": readiness_in.get("status"),
        "formal_model_changed": False,
        "trade_decision_changed": False,
        "active_in_trade_decision": False,
        "report_changed": False,
        "portfolio_replay_executed": False,
        "not_live_rule": True,
        "forward_returns_live_rule_usage": False,
    }


def _summary(readiness: dict[str, Any]) -> str:
    return f"""# Layer1 t164 TPEx phase-1 bounded proof runner contract

## Verdict
- status={readiness["status"]}
- runner_contract_rows={readiness["runner_contract_rows"]}
- phase1_ticker_count={readiness["phase1_ticker_count"]}
- phase1_period_count={readiness["phase1_period_count"]}
- ready_for_radar_phase1_tpex_50x2_runner_execution={str(readiness["ready_for_radar_phase1_tpex_50x2_runner_execution"]).lower()}
- ready_for_experiments=false
- ready_for_formal=false
- ready_for_strategy_replay=false

## Boundary
This is a diagnostic/source runner contract only. It does not execute t164 materialization, Experiments, portfolio replay, strategy replay, formal model changes, report changes, or trade decisions.

## Retained blockers
- TPEx historical all-stock universe remains blocked; current-or-carried universe is sampling-only.
- Full-period 891 x 46 expansion requires a separate checkpoint/resume runner contract.
- capex_proxy and receivables_trade remain diagnostic proxy / human-review required.

## Flags
- formal_model_changed=false
- trade_decision_changed=false
- active_in_trade_decision=false
- report_changed=false
- portfolio_replay_executed=false
- ready_for_strategy_replay=false
- not_live_rule=true
- forward_returns_live_rule_usage=false
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--radar-dir", default=str(DEFAULT_RADAR_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()
    manifest = build_contract(radar_dir=args.radar_dir, output_dir=args.output_dir)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
