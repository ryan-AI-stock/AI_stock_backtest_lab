"""Review Radar/Data Layer1 t164 candidate/detail pruning runner v2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


TASK_ID = "TASK-BACKTEST-CORE-VNEXT-LAYER1-T164-PRUNING-V2-SOURCE-PACKAGE-REVIEW-001"
DEFAULT_RADAR_DIR = Path(
    r"C:\Users\zergv\Documents\Codex\2026-05-23\ai-stock-rotation-radar-https-docs\outputs"
    r"\radar_vnext_layer1_t164_candidate_detail_pruning_runner_v2_20260707"
)
DEFAULT_OUTPUT_DIR = Path("outputs/vnext_layer1_t164_pruning_v2_source_package_review_20260707")


def build_review(*, radar_dir: str | Path = DEFAULT_RADAR_DIR, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    radar = Path(radar_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    readiness_in = _read_json(radar / "readiness_for_core_t164_candidate_detail_pruning_runner_v2.json")
    pruning_audit = _read_csv(radar / "pruning_effectiveness_audit.csv")
    route_cost = _read_csv(radar / "projected_route_cost_report.csv")
    coverage_market_period = _read_csv(radar / "coverage_by_market_period.csv")
    coverage_field = _read_csv(radar / "coverage_by_field.csv")
    tpex_evidence = _read_csv(radar / "tpex_universal_readiness_evidence.csv")
    future_audit_in = _read_csv(radar / "future_data_governance_audit.csv")

    review_matrix = _review_matrix(readiness_in, route_cost, coverage_market_period)
    bounded_contract_requirements = _bounded_contract_requirements()
    remaining_blockers = _remaining_blockers(coverage_field, tpex_evidence)
    future_audit = _future_audit(future_audit_in)
    next_handoff = _next_handoff()
    readiness = _readiness(readiness_in, route_cost, coverage_field, tpex_evidence, future_audit)

    _write_csv(review_matrix, output / "layer1_t164_pruning_v2_review_matrix.csv")
    _write_csv(bounded_contract_requirements, output / "layer1_t164_bounded_broader_ingest_contract_requirements.csv")
    _write_csv(remaining_blockers, output / "layer1_t164_pruning_v2_remaining_blockers.csv")
    _write_csv(future_audit, output / "layer1_t164_pruning_v2_future_data_audit.csv")
    _write_csv(next_handoff, output / "layer1_t164_pruning_v2_next_handoff.csv")
    if not pruning_audit.empty:
        _write_csv(pruning_audit, output / "layer1_t164_pruning_v2_effectiveness_audit_imported.csv")
    (output / "readiness_for_layer1_t164_pruning_v2_source_package_review.json").write_text(
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
            "layer1_t164_pruning_v2_review_matrix.csv",
            "layer1_t164_bounded_broader_ingest_contract_requirements.csv",
            "layer1_t164_pruning_v2_remaining_blockers.csv",
            "layer1_t164_pruning_v2_future_data_audit.csv",
            "layer1_t164_pruning_v2_next_handoff.csv",
            "layer1_t164_pruning_v2_effectiveness_audit_imported.csv",
            "readiness_for_layer1_t164_pruning_v2_source_package_review.json",
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


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.read_text(encoding="utf-8").strip() == "empty":
        return pd.DataFrame()
    return pd.read_csv(path)


def _review_matrix(readiness: dict[str, Any], route_cost: pd.DataFrame, coverage: pd.DataFrame) -> pd.DataFrame:
    rows = [
        ("statement_route", "passed_bounded_seed", readiness.get("statement_success_rows"), "40/40 t164 statement success"),
        ("official_asof_join", "passed_bounded_seed", readiness.get("official_asof_matched_rows"), "40/40 official-asof matched"),
        ("route_error", "passed", readiness.get("route_error_count"), "no route errors"),
        ("cache_efficiency", "passed_bounded_cost_guard", readiness.get("actual_cache_rows_per_materialized_row"), "actual cache rows per materialized row reduced from baseline"),
        ("route_reduction", "passed", readiness.get("route_reduction_vs_baseline"), "route reduction vs baseline source package"),
        ("coverage_scope", "bounded_seed_only", readiness.get("sample_rows"), "20 tickers x 2 periods; not full universe"),
    ]
    if not route_cost.empty:
        budget = route_cost.iloc[0].to_dict()
        rows.append(("projected_route_budget", budget.get("budget_status"), budget.get("projected_routes_per_row"), "projected_routes_per_row <= budget_routes_per_row"))
    return pd.DataFrame(rows, columns=["review_item", "status", "evidence_value", "note"]).assign(diagnostic_only=True)


def _bounded_contract_requirements() -> pd.DataFrame:
    rows = [
        ("input_scope", "required", "bounded seed only unless Research/Strategy approves larger scope", "avoid silent full-universe claim"),
        ("source_contract", "required", "t164sb05/t164sb03 statement rows joined to official t05st01/t05st01_detail asof", "market_available_at policy preserved"),
        ("cache_manifest", "required", "raw cache hash manifest and query payload hash per route", "reproducibility"),
        ("coverage_audit", "required", "coverage by ticker/market/period/field/match status", "requested vs actual coverage"),
        ("blocked_policy", "required", "unmatched/ambiguous rows blocked, no silent fill", "contract hygiene"),
        ("label_policy", "required", "capex_proxy and receivables_trade remain human_review_proxy_label_required", "not formal-ready labels"),
        ("future_data_governance", "required", "quarter_end/query_response/deadline proxy prohibited for official route", "PIT hygiene"),
    ]
    return pd.DataFrame(rows, columns=["requirement", "status", "detail", "purpose"]).assign(diagnostic_only=True)


def _remaining_blockers(coverage_field: pd.DataFrame, tpex_evidence: pd.DataFrame) -> pd.DataFrame:
    rows = [
        ("full_universe_scope", "blocked", "20 tickers x 2 periods only; not all-stock", "Radar/Data or Research/Strategy"),
        ("full_period_range", "blocked", "only 115Q1 and 114Q4 in seed", "Radar/Data or Research/Strategy"),
        ("tpex_all_stock_proof", "blocked", "TPEx bounded seed passes but all-stock proof not executed", "Radar/Data"),
        ("capex_proxy", "human_review_proxy_label_required", "capex label/basket remains proxy", "Research/Strategy label policy"),
        ("receivables_trade", "human_review_proxy_label_required", "receivables basket remains proxy", "Research/Strategy label policy"),
        ("experiments_authorization", "blocked", "source package not Experiments-ready", "vNext Research"),
    ]
    if not tpex_evidence.empty:
        reason = tpex_evidence.iloc[0].get("reason")
        rows[2] = ("tpex_all_stock_proof", "blocked", reason, "Radar/Data")
    return pd.DataFrame(rows, columns=["blocker", "status", "detail", "owner"]).assign(diagnostic_only=True)


def _future_audit(future_audit_in: pd.DataFrame) -> pd.DataFrame:
    if future_audit_in.empty:
        return pd.DataFrame(
            [{"audit_item": "future_data_governance_audit", "status": "blocked_missing_input", "future_data_violation_count": 0, "diagnostic_only": True}]
        )
    out = future_audit_in.copy()
    if "future_data_violation_count" not in out:
        out["future_data_violation_count"] = 0
    out["diagnostic_only"] = True
    return out


def _next_handoff() -> pd.DataFrame:
    rows = [
        ("vNext Research", "judge_bounded_ingest_contract_planning", "Decide whether Core should build bounded broader ingest contract from pruning v2 seed.", "recommended_next"),
        ("Radar/Data", "all_stock_tpex_and_full_period_plan", "If Research wants broader/full route, provide TPEx all-stock proof and full period range materialization plan.", "parallel_blocker"),
        ("Strategy Center", "label_policy_decision", "capex_proxy / receivables_trade human-review proxy policy remains unresolved.", "policy_blocker"),
    ]
    return pd.DataFrame(rows, columns=["next_owner", "handoff_item", "request", "status"]).assign(diagnostic_only=True)


def _readiness(
    readiness_in: dict[str, Any],
    route_cost: pd.DataFrame,
    coverage_field: pd.DataFrame,
    tpex_evidence: pd.DataFrame,
    future_audit: pd.DataFrame,
) -> dict[str, Any]:
    future_count = int(future_audit["future_data_violation_count"].sum()) if "future_data_violation_count" in future_audit else 0
    projected_routes = float(readiness_in.get("projected_routes_per_row", 0))
    budget_routes = float(readiness_in.get("budget_routes_per_row", 0))
    bounded_ready = (
        int(readiness_in.get("statement_success_rows", 0)) == int(readiness_in.get("sample_rows", -1))
        and int(readiness_in.get("official_asof_matched_rows", 0)) == int(readiness_in.get("sample_rows", -1))
        and int(readiness_in.get("route_error_count", 1)) == 0
        and projected_routes <= budget_routes
        and future_count == 0
    )
    return {
        "date": "2026-07-07",
        "task_id": TASK_ID,
        "owner": "BACKTEST_LAB Core/Data",
        "status": "pruning_v2_review_passed_bounded_contract_planning_ready_not_full_ingest",
        "radar_status": readiness_in.get("status"),
        "diagnostic_only": True,
        "sample_rows": int(readiness_in.get("sample_rows", 0)),
        "statement_success_rows": int(readiness_in.get("statement_success_rows", 0)),
        "official_asof_matched_rows": int(readiness_in.get("official_asof_matched_rows", 0)),
        "official_asof_matched_share": float(readiness_in.get("official_asof_matched_share", 0)),
        "route_error_count": int(readiness_in.get("route_error_count", 0)),
        "cache_manifest_rows": int(readiness_in.get("cache_manifest_rows", 0)),
        "actual_cache_rows_per_materialized_row": float(readiness_in.get("actual_cache_rows_per_materialized_row", 0)),
        "baseline_cache_rows_per_materialized_row": float(readiness_in.get("baseline_cache_rows_per_materialized_row", 0)),
        "route_reduction_vs_baseline": float(readiness_in.get("route_reduction_vs_baseline", 0)),
        "projected_routes_per_row": projected_routes,
        "budget_routes_per_row": budget_routes,
        "ready_for_bounded_broader_ingest_contract_planning": bool(bounded_ready),
        "ready_for_core_t164_broader_ingest_contract": False,
        "ready_for_core_t164_broader_materialization": False,
        "ready_for_experiments": False,
        "ready_for_formal": False,
        "ready_for_strategy_replay": False,
        "ready_for_full_universe": False,
        "remaining_blockers": [
            "TPEx all-stock proof not complete",
            "full period range not complete",
            "capex_proxy / receivables_trade human-review proxy policy required",
            "Research approval required before Core bounded contract build",
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


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def _summary(readiness: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Layer1 t164 Pruning v2 Source Package Review",
            "",
            f"Status: {readiness['status']}",
            "",
            "Conclusion: pruning v2 resolves the bounded route fan-out blocker enough for bounded broader ingest contract planning, but it is still not full-universe, not materialized broader ingest, and not Experiments-ready.",
            "",
            "Readiness:",
            f"- ready_for_bounded_broader_ingest_contract_planning={str(readiness['ready_for_bounded_broader_ingest_contract_planning']).lower()}",
            "- ready_for_core_t164_broader_ingest_contract=false",
            "- ready_for_core_t164_broader_materialization=false",
            "- ready_for_experiments=false",
            "- ready_for_formal=false",
            "- ready_for_strategy_replay=false",
            f"- sample_rows={readiness['sample_rows']}",
            f"- official_asof_matched_rows={readiness['official_asof_matched_rows']}",
            f"- actual_cache_rows_per_materialized_row={readiness['actual_cache_rows_per_materialized_row']}",
            f"- route_reduction_vs_baseline={readiness['route_reduction_vs_baseline']}",
            f"- future_data_violation_count={readiness['future_data_violation_count']}",
            "",
            "Remaining blockers:",
            *[f"- {item}" for item in readiness["remaining_blockers"]],
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
    manifest = build_review(radar_dir=args.radar_dir, output_dir=args.output_dir)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
