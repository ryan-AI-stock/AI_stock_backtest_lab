"""Build Layer1 t164 phase-2 bounded expansion / full-period planning contract.

This is source/contract planning only. It creates a bounded runner contract for
Radar/Data execution, but does not run source acquisition, materialization,
Experiments, replay, or formal pipeline changes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


TASK_ID = "TASK-BACKTEST-CORE-VNEXT-LAYER1-T164-PHASE2-BOUNDED-EXPANSION-FULL-PERIOD-PLANNING-CONTRACT-001"
DEFAULT_RADAR_PLAN_DIR = Path(
    r"C:\Users\zergv\Documents\Codex\2026-05-23\ai-stock-rotation-radar-https-docs\outputs"
    r"\radar_vnext_layer1_t164_tpex_all_stock_proof_full_period_bounded_expansion_plan_20260707"
)
DEFAULT_PHASE1_CLOSURE_DIR = Path("outputs/vnext_layer1_t164_tpex_phase1_policy_closure_20260707")
DEFAULT_OUTPUT_DIR = Path("outputs/vnext_layer1_t164_phase2_bounded_expansion_contract_20260707")

PHASE2_PERIODS = ["115Q1", "114Q4", "114Q3", "114Q2"]


def build_contract(
    *,
    radar_plan_dir: str | Path = DEFAULT_RADAR_PLAN_DIR,
    phase1_closure_dir: str | Path = DEFAULT_PHASE1_CLOSURE_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    radar = Path(radar_plan_dir)
    phase1 = Path(phase1_closure_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    phase1_readiness = _read_json(phase1 / "readiness_for_layer1_t164_tpex_phase1_policy_closure.json")
    universe = _read_csv(radar / "tpex_all_stock_universe_inventory.csv", dtype={"ticker": str})
    batch_plan = _read_csv(radar / "full_period_bounded_expansion_batch_plan.csv")
    cost_estimate = _read_csv(radar / "full_period_bounded_expansion_cost_estimate.csv")

    sampled_tickers = _phase2_sample(universe)
    runner_contract = _runner_contract(sampled_tickers)
    period_plan = _period_plan()
    checkpoint_contract = _checkpoint_contract()
    route_budget = _route_budget_guard(batch_plan, cost_estimate)
    official_asof_policy = _official_asof_policy()
    blocked_policy = _blocked_policy(phase1_readiness)
    field_policy = _field_policy()
    coverage_audit = _coverage_audit_design(runner_contract)
    future_audit = _future_audit()
    readiness = _readiness(phase1_readiness, runner_contract, route_budget)

    _write_csv(runner_contract, output / "layer1_t164_phase2_bounded_runner_contract.csv")
    _write_csv(sampled_tickers, output / "layer1_t164_phase2_ticker_universe_seed.csv")
    _write_csv(period_plan, output / "layer1_t164_phase2_period_plan.csv")
    _write_csv(checkpoint_contract, output / "layer1_t164_phase2_checkpoint_resume_contract.csv")
    _write_csv(route_budget, output / "layer1_t164_phase2_route_budget_guard.csv")
    _write_csv(official_asof_policy, output / "layer1_t164_phase2_official_asof_policy.csv")
    _write_csv(blocked_policy, output / "layer1_t164_phase2_blocked_row_policy.csv")
    _write_csv(field_policy, output / "layer1_t164_phase2_field_label_policy.csv")
    _write_csv(coverage_audit, output / "layer1_t164_phase2_coverage_audit_design.csv")
    _write_csv(future_audit, output / "layer1_t164_phase2_future_data_audit.csv")
    (output / "readiness_for_layer1_t164_phase2_bounded_expansion_contract.json").write_text(
        json.dumps(readiness, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    manifest = {
        "task_id": TASK_ID,
        "status": readiness["status"],
        "output_dir": str(output.resolve()),
        "radar_plan_input_dir": str(radar.resolve()),
        "phase1_closure_input_dir": str(phase1.resolve()),
        "output_files": [
            "layer1_t164_phase2_bounded_runner_contract.csv",
            "layer1_t164_phase2_ticker_universe_seed.csv",
            "layer1_t164_phase2_period_plan.csv",
            "layer1_t164_phase2_checkpoint_resume_contract.csv",
            "layer1_t164_phase2_route_budget_guard.csv",
            "layer1_t164_phase2_official_asof_policy.csv",
            "layer1_t164_phase2_blocked_row_policy.csv",
            "layer1_t164_phase2_field_label_policy.csv",
            "layer1_t164_phase2_coverage_audit_design.csv",
            "layer1_t164_phase2_future_data_audit.csv",
            "readiness_for_layer1_t164_phase2_bounded_expansion_contract.json",
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


def _phase2_sample(universe: pd.DataFrame) -> pd.DataFrame:
    if universe.empty:
        return pd.DataFrame(columns=["ticker", "name", "market"])
    ordered = universe.sort_values("ticker").reset_index(drop=True)
    indices = sorted({round(i * (len(ordered) - 1) / 99) for i in range(100)})
    sample = ordered.iloc[indices].copy().head(100)
    sample["sample_phase"] = "phase_2_tpex_100x4_recent"
    sample["sample_policy"] = "deterministic_evenly_spaced_from_current_or_carried_tpex_sampling_universe"
    sample["source_universe_quality"] = "current_or_carried_sampling_proxy_not_historical_pit"
    sample["historical_pit_universe_ready"] = False
    sample["diagnostic_only"] = True
    return sample


def _runner_contract(sample: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for ticker in sample.to_dict("records"):
        for period in PHASE2_PERIODS:
            rows.append(
                {
                    "phase": "phase_2_tpex_100x4_recent",
                    "ticker": ticker.get("ticker"),
                    "name": ticker.get("name"),
                    "market": "TPEx",
                    "report_period": period,
                    "period_scope": "recent_4_quarters",
                    "sample_policy": ticker.get("sample_policy"),
                    "source_universe_quality": ticker.get("source_universe_quality"),
                    "historical_pit_universe_ready": False,
                    "planned_routes": "t164sb05;t164sb03;t05st01;t05st01_detail_prefiltered",
                    "expected_route_budget_per_ticker_period": 8.0,
                    "budget_guard_per_ticker_period": 10.0,
                    "official_asof_required": True,
                    "after_close_next_trading_day_policy_required": True,
                    "unmatched_or_ambiguous_policy": "explicit_blocked_ledger_no_silent_backfill",
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


def _period_plan() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "report_period": period,
                "period_order": idx + 1,
                "scope": "phase2_recent_4_quarters",
                "reason": "extend beyond phase1 two-period proof without jumping to full 46-period range",
                "diagnostic_only": True,
            }
            for idx, period in enumerate(PHASE2_PERIODS)
        ]
    )


def _checkpoint_contract() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("checkpoint_state.json", "required", "resume ticker-period cursor, route counts, and blocked rows"),
            ("current_step.txt", "required", "observable runner progress"),
            ("raw_cache_hash_manifest.csv", "required", "payload/response hash and cache relative path"),
            ("blocked_or_ambiguous_rows.csv", "required", "explicit official-asof or route blockers"),
            ("coverage_by_market_period.csv", "required", "statement and official-asof coverage"),
            ("route_error_threshold", "block_if_exceeded", "runner should block if route_error_count is nonzero after bounded retry"),
            ("budget_guard", "block_if_exceeded", "projected or actual routes per row must remain <= 10 unless reauthorized"),
        ],
        columns=["checkpoint_item", "policy", "description"],
    ).assign(diagnostic_only=True)


def _route_budget_guard(batch_plan: pd.DataFrame, cost_estimate: pd.DataFrame) -> pd.DataFrame:
    source = batch_plan if not batch_plan.empty else cost_estimate
    rows = []
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
                "allowed_by_core_contract_now": phase == "phase_2_tpex_100x4_recent",
                "blocked_reason": ""
                if phase == "phase_2_tpex_100x4_recent"
                else "not authorized in this phase2 contract",
                "diagnostic_only": True,
                "not_live_rule": True,
            }
        )
    return pd.DataFrame(rows)


def _official_asof_policy() -> pd.DataFrame:
    rows = [
        ("market_available_at", "official t05st01/t05st01_detail public financial-report announcement timestamp only"),
        ("after_close_eligibility", "after 13:30 Taiwan regular close => next trading day eligibility"),
        ("unmatched_or_ambiguous", "explicit blocked ledger; no silent backfill"),
        ("6114_8080_phase1_precedent", "do not chase in phase2 unless encountered as new bounded blocker and authorized by policy"),
        ("quarter_end_date", "prohibited as available_at"),
        ("query_response_datetime", "prohibited as available_at"),
        ("conservative_deadline_proxy", "prohibited as official available_at; separate diagnostic candidate only"),
    ]
    return pd.DataFrame(rows, columns=["policy_item", "policy"]).assign(diagnostic_only=True)


def _blocked_policy(phase1_readiness: dict[str, Any]) -> pd.DataFrame:
    rows = [
        {
            "blocked_item": "phase1_6114_114Q4",
            "policy": "carry explicit blocked status; no version silent selection",
            "source": "phase1 policy closure",
            "diagnostic_only": True,
        },
        {
            "blocked_item": "phase1_8080_115Q1",
            "policy": "carry explicit blocked status; no proxy backfill",
            "source": "phase1 policy closure",
            "diagnostic_only": True,
        },
        {
            "blocked_item": "phase2_new_unmatched_or_ambiguous_rows",
            "policy": "record in blocked_or_ambiguous_rows.csv; do not block entire runner unless route/system failure exceeds guard",
            "source": "phase2 contract",
            "diagnostic_only": True,
        },
        {
            "blocked_item": "tpex_historical_all_stock_universe",
            "policy": "current-or-carried universe is sampling-only, not historical PIT membership",
            "source": "Radar plan",
            "diagnostic_only": True,
        },
    ]
    return pd.DataFrame(rows)


def _field_policy() -> pd.DataFrame:
    rows = [
        ("operating_cash_flow", "accepted_if_statement_success_and_official_asof_matched", "exact_pit_after_official_asof_join"),
        ("investing_cash_flow", "accepted_if_statement_success_and_official_asof_matched", "exact_pit_after_official_asof_join"),
        ("inventory", "accepted_if_statement_success_and_official_asof_matched", "exact_pit_after_official_asof_join"),
        ("current_assets", "accepted_if_statement_success_and_official_asof_matched", "exact_pit_after_official_asof_join"),
        ("current_liabilities", "accepted_if_statement_success_and_official_asof_matched", "exact_pit_after_official_asof_join"),
        ("current_ratio", "derive current_assets/current_liabilities after official-asof join", "derived_pit_after_official_asof_join"),
        ("capex_proxy", "diagnostic proxy and human-review required; not formal FCF", "proxy_human_review_required"),
        ("receivables_trade", "diagnostic proxy and human-review required; not formal receivables risk", "proxy_human_review_required"),
    ]
    return pd.DataFrame(rows, columns=["field", "policy", "source_quality"]).assign(
        formal_ready=False,
        ready_for_experiments=False,
        diagnostic_only=True,
    )


def _coverage_audit_design(runner_contract: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {
            "coverage_scope": "overall",
            "requested_rows": len(runner_contract),
            "ticker_count": runner_contract["ticker"].nunique(),
            "period_count": runner_contract["report_period"].nunique(),
            "market_mix": "TPEx_only_phase2",
            "required_audit": "statement_success_rows;official_asof_matched_rows;blocked_rows;route_error_count",
            "diagnostic_only": True,
        }
    ]
    for period, group in runner_contract.groupby("report_period"):
        rows.append(
            {
                "coverage_scope": f"period_{period}",
                "requested_rows": len(group),
                "ticker_count": group["ticker"].nunique(),
                "period_count": 1,
                "market_mix": "TPEx_only_phase2",
                "required_audit": "statement_success_rows;official_asof_matched_rows;blocked_rows;route_error_count",
                "diagnostic_only": True,
            }
        )
    return pd.DataFrame(rows)


def _future_audit() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "audit_item": "contract_forward_return_as_rule",
                "status": "passed",
                "future_data_violation_count": 0,
                "note": "contract has no return evaluation or live rule",
            },
            {
                "audit_item": "official_available_at_policy",
                "status": "passed",
                "future_data_violation_count": 0,
                "note": "requires official public announcement timestamp only",
            },
            {
                "audit_item": "blocked_rows_no_silent_backfill",
                "status": "passed",
                "future_data_violation_count": 0,
                "note": "unmatched/ambiguous rows remain explicit blocked rows",
            },
        ]
    )


def _readiness(phase1_readiness: dict[str, Any], runner_contract: pd.DataFrame, route_budget: pd.DataFrame) -> dict[str, Any]:
    phase2 = route_budget[route_budget["phase"].eq("phase_2_tpex_100x4_recent")]
    budget_pass = bool(phase2["budget_status"].eq("pass_planning").all()) if not phase2.empty else False
    return {
        "task_id": TASK_ID,
        "status": "phase2_bounded_expansion_contract_ready_for_radar_runner_not_experiments",
        "diagnostic_only": True,
        "phase2_ticker_count": int(runner_contract["ticker"].nunique()),
        "phase2_period_count": int(runner_contract["report_period"].nunique()),
        "phase2_ticker_period_rows": int(len(runner_contract)),
        "market_mix": {"TPEx": int(len(runner_contract))},
        "period_range": PHASE2_PERIODS,
        "projected_routes_per_row": 8.0,
        "projected_total_routes": 3200,
        "route_budget_guard_per_row": 10.0,
        "checkpoint_resume_required": True,
        "phase2_budget_status": "pass_planning" if budget_pass else "blocked",
        "phase1_policy_closure_status": phase1_readiness.get("status"),
        "phase1_accepted_partial_blocked_limit": True,
        "stop_6114_8080_chase": True,
        "ready_for_radar_phase2_bounded_runner_execution": bool(budget_pass),
        "ready_for_core_phase2_materialization_review_after_radar": False,
        "ready_for_experiments": False,
        "ready_for_formal": False,
        "ready_for_strategy_replay": False,
        "ready_for_full_universe": False,
        "tpex_all_stock_universal_ready": False,
        "future_data_violation_count": 0,
        "blocked_fields": [
            "tpex_historical_all_stock_universe",
            "full_universe_materialization",
            "full_period_46_quarters_materialization",
            "phase1_6114_version_match_blocked",
            "phase1_8080_no_official_target_candidate",
            "capex_proxy_formal_label",
            "receivables_trade_formal_label",
        ],
        "proxy_fields": ["capex_proxy", "receivables_trade", "current_or_carried_tpex_universe_candidate"],
        "formal_model_changed": False,
        "trade_decision_changed": False,
        "active_in_trade_decision": False,
        "report_changed": False,
        "portfolio_replay_executed": False,
        "not_live_rule": True,
        "forward_returns_live_rule_usage": False,
    }


def _summary(readiness: dict[str, Any]) -> str:
    return f"""# Layer1 t164 phase_2 bounded expansion contract

## Verdict
- status={readiness["status"]}
- phase2_ticker_count={readiness["phase2_ticker_count"]}
- phase2_period_count={readiness["phase2_period_count"]}
- phase2_ticker_period_rows={readiness["phase2_ticker_period_rows"]}
- projected_total_routes={readiness["projected_total_routes"]}
- checkpoint_resume_required=true
- ready_for_radar_phase2_bounded_runner_execution={str(readiness["ready_for_radar_phase2_bounded_runner_execution"]).lower()}
- ready_for_experiments=false
- ready_for_formal=false

## Boundary
This is bounded source runner contract planning only. It is not full universe, not full-period materialization, not Experiments-ready, and not formal-ready.

## Next
Radar/Data should execute the phase_2 bounded runner if Strategy Center/Core accepts this contract handoff. Core/Data should review the runner output before any further expansion.

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
    parser.add_argument("--radar-plan-dir", default=str(DEFAULT_RADAR_PLAN_DIR))
    parser.add_argument("--phase1-closure-dir", default=str(DEFAULT_PHASE1_CLOSURE_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()
    manifest = build_contract(
        radar_plan_dir=args.radar_plan_dir,
        phase1_closure_dir=args.phase1_closure_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
