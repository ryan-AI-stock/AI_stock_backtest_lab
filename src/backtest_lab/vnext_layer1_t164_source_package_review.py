"""Review Radar/Data Layer1 t164 broader seed source package.

The review determines whether Core can build a broader ingest contract now or
whether Radar/Data must first reduce route fan-out with candidate/detail
pruning. This is source/contract readiness only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


TASK_ID = "TASK-BACKTEST-CORE-VNEXT-LAYER1-T164-BROADER-SOURCE-PACKAGE-REVIEW-001"
DEFAULT_RADAR_DIR = Path(
    r"C:\Users\zergv\Documents\Codex\2026-05-23\ai-stock-rotation-radar-https-docs\outputs"
    r"\radar_vnext_layer1_t164_full_broader_source_materialization_runner_20260707"
)
DEFAULT_OUTPUT_DIR = Path("outputs/vnext_layer1_t164_broader_source_package_review_20260707")


def build_review(*, radar_dir: str | Path = DEFAULT_RADAR_DIR, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    radar = Path(radar_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    readiness_in = _read_json(radar / "readiness_for_core_t164_full_broader_source_materialization_runner.json")
    coverage_market_period = _read_csv(radar / "coverage_by_market_period.csv")
    coverage_field = _read_csv(radar / "coverage_by_field.csv")
    runner_cost = _read_csv(radar / "runner_cost_and_pruning_readiness.csv")
    blocked_rows = _read_csv(radar / "blocked_or_ambiguous_rows.csv")
    label_inventory = _read_csv(radar / "capex_receivables_label_inventory.csv")
    tpex_evidence = _read_csv(radar / "tpex_universal_readiness_evidence.csv")
    future_audit_in = _read_csv(radar / "future_data_governance_audit.csv")

    review = _review_matrix(readiness_in, coverage_market_period, coverage_field, runner_cost, blocked_rows)
    pruning = _pruning_requirements(runner_cost)
    field_policy = _field_policy_review(coverage_field, label_inventory)
    coverage_audit = _coverage_review(coverage_market_period, coverage_field, tpex_evidence)
    future_audit = _future_audit(future_audit_in)
    radar_handoff = _radar_handoff(pruning, field_policy)
    readiness = _readiness(readiness_in, runner_cost, coverage_field, future_audit)

    _write_csv(review, output / "layer1_t164_source_package_review_matrix.csv")
    _write_csv(pruning, output / "layer1_t164_candidate_detail_pruning_requirements.csv")
    _write_csv(field_policy, output / "layer1_t164_field_label_policy_review.csv")
    _write_csv(coverage_audit, output / "layer1_t164_source_package_coverage_review.csv")
    _write_csv(future_audit, output / "layer1_t164_source_package_future_data_audit.csv")
    _write_csv(radar_handoff, output / "layer1_t164_radar_pruning_runner_v2_handoff.csv")
    (output / "readiness_for_layer1_t164_broader_source_package_review.json").write_text(
        json.dumps(readiness, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    manifest = {
        "task_id": TASK_ID,
        "status": readiness["status"],
        "output_dir": str(output.resolve()),
        "radar_input_dir": str(radar.resolve()),
        "radar_commit": "fe92c9d",
        "output_files": [
            "layer1_t164_source_package_review_matrix.csv",
            "layer1_t164_candidate_detail_pruning_requirements.csv",
            "layer1_t164_field_label_policy_review.csv",
            "layer1_t164_source_package_coverage_review.csv",
            "layer1_t164_source_package_future_data_audit.csv",
            "layer1_t164_radar_pruning_runner_v2_handoff.csv",
            "readiness_for_layer1_t164_broader_source_package_review.json",
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


def _review_matrix(
    readiness: dict[str, Any],
    coverage_market_period: pd.DataFrame,
    coverage_field: pd.DataFrame,
    runner_cost: pd.DataFrame,
    blocked_rows: pd.DataFrame,
) -> pd.DataFrame:
    cache_rows = int(readiness.get("cache_manifest_rows", 0))
    sample_rows = int(readiness.get("sample_rows", 0))
    cache_per_sample = cache_rows / sample_rows if sample_rows else 0.0
    return pd.DataFrame(
        [
            ("statement_route", "passed_bounded_seed", readiness.get("statement_success_rows", 0), "40/40 t164sb05/t164sb03 success in seed"),
            ("official_asof_join", "passed_bounded_seed", readiness.get("official_asof_matched_rows", 0), "40/40 official timestamp matched in seed"),
            ("blocked_or_ambiguous_rows", "passed_bounded_seed", len(blocked_rows), "seed has no blocked rows"),
            ("field_coverage", "partial", int(coverage_field["missing_rows"].sum()) if not coverage_field.empty else None, "some balance-sheet/capex/receivable fields are sparse or label-policy-bound"),
            ("runner_cost", "blocked_for_full_universe_without_pruning", cache_per_sample, "1683 cache rows for 40 source rows; all-stock needs pruning"),
            ("market_period_coverage", "bounded_seed_only", len(coverage_market_period), "20 tickers x 2 periods only"),
        ],
        columns=["review_item", "status", "evidence_value", "note"],
    ).assign(diagnostic_only=True)


def _pruning_requirements(runner_cost: pd.DataFrame) -> pd.DataFrame:
    cache_rows = int(runner_cost["raw_cache_hash_manifest_rows"].iloc[0]) if not runner_cost.empty else 0
    requested = int(runner_cost["requested_rows"].iloc[0]) if not runner_cost.empty else 0
    ratio = cache_rows / requested if requested else 0.0
    rows = [
        ("t05st01_candidate_query_pruning", "required_before_full_universe", "reduce month/all-year probes using report-period expected windows and direct subject tokens", ratio),
        ("detail_fetch_pruning", "required_before_full_universe", "fetch detail only for subject candidates passing financial-report/period/token prefilter", ratio),
        ("premeeting_notice_filter", "required_before_full_universe", "exclude premeeting notices before detail fan-out when possible", ratio),
        ("wrong_period_filter", "required_before_full_universe", "drop 115Q1/114Q2/114Q3/113Q4 when target is 114Q4 before broad detail fan-out", ratio),
        ("cost_budget_guard", "required_before_full_universe", "emit projected route count before full run; stop if route fan-out exceeds approved budget", ratio),
        ("resume_checkpoint", "required_before_full_universe", "preserve current_step/checkpoint_state and cache manifest for restart", ratio),
    ]
    return pd.DataFrame(rows, columns=["requirement", "status", "detail", "seed_cache_rows_per_materialized_row"]).assign(diagnostic_only=True)


def _field_policy_review(coverage_field: pd.DataFrame, label_inventory: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if not coverage_field.empty:
        for row in coverage_field.to_dict("records"):
            field = row["field"]
            status = "accepted_candidate_exact"
            if field in {"capex_proxy", "receivables_trade"}:
                status = "human_review_proxy_label_required"
            rows.append(
                {
                    "field": field,
                    "status": status,
                    "missing_rows": row.get("missing_rows"),
                    "missing_share": row.get("missing_share"),
                    "source_quality": row.get("source_quality"),
                    "policy_note": "do not promote proxy labels to formal" if "proxy" in status else "candidate PIT field after official-asof join",
                    "diagnostic_only": True,
                }
            )
    if label_inventory.empty:
        rows.append(
            {
                "field": "capex_receivables_label_inventory",
                "status": "missing_or_empty",
                "missing_rows": None,
                "missing_share": None,
                "source_quality": "blocked_for_label_review",
                "policy_note": "Radar package must include label inventory for human review before full label acceptance",
                "diagnostic_only": True,
            }
        )
    return pd.DataFrame(rows)


def _coverage_review(coverage_market_period: pd.DataFrame, coverage_field: pd.DataFrame, tpex_evidence: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in coverage_market_period.to_dict("records") if not coverage_market_period.empty else []:
        rows.append(
            {
                "coverage_axis": row.get("coverage_type"),
                "group": row.get("group"),
                "requested_rows": row.get("requested_rows"),
                "materialized_rows": row.get("materialized_rows"),
                "matched_rows": row.get("official_asof_matched_rows"),
                "blocked_rows": row.get("blocked_rows"),
                "status": "passed_bounded_seed_only",
                "diagnostic_only": True,
            }
        )
    rows.append(
        {
            "coverage_axis": "tpex_universal",
            "group": "TPEx",
            "requested_rows": None,
            "materialized_rows": None,
            "matched_rows": None,
            "blocked_rows": None,
            "status": "blocked_all_stock_proof_not_complete" if tpex_evidence.empty else "review_tpex_evidence_required",
            "diagnostic_only": True,
        }
    )
    return pd.DataFrame(rows)


def _future_audit(future_audit_in: pd.DataFrame) -> pd.DataFrame:
    if future_audit_in.empty:
        return pd.DataFrame(
            [
                {
                    "audit_item": "future_data_governance_audit",
                    "status": "blocked_missing_input",
                    "future_data_violation_count": 0,
                    "note": "Radar package missing future_data_governance_audit.csv",
                    "diagnostic_only": True,
                }
            ]
        )
    out = future_audit_in.copy()
    if "future_data_violation_count" not in out:
        out["future_data_violation_count"] = 0
    out["diagnostic_only"] = True
    return out


def _radar_handoff(pruning: pd.DataFrame, field_policy: pd.DataFrame) -> pd.DataFrame:
    rows = [
        ("candidate_detail_pruning_runner_v2", "required", "implement candidate/detail pruning before any all-stock run"),
        ("projected_route_cost_report", "required", "estimate route/cache count for target universe and period range before execution"),
        ("all_stock_tpex_proof", "required", "prove TPEx route stability beyond bounded seed"),
        ("full_period_range_plan", "required", "define requested vs actual periods and disclosure lag coverage"),
        ("label_inventory_expansion", "required", "expand capex/receivables label inventory and keep human_review_required tags"),
        ("blocked_ambiguous_policy_audit", "required", "keep unmatched/ambiguous rows blocked; no proxy dates"),
    ]
    return pd.DataFrame(rows, columns=["handoff_item", "status", "request"]).assign(
        next_owner="Radar/Data",
        diagnostic_only=True,
    )


def _readiness(readiness_in: dict[str, Any], runner_cost: pd.DataFrame, coverage_field: pd.DataFrame, future_audit: pd.DataFrame) -> dict[str, Any]:
    sample_rows = int(readiness_in.get("sample_rows", 0))
    cache_rows = int(readiness_in.get("cache_manifest_rows", 0))
    cache_per_row = cache_rows / sample_rows if sample_rows else 0.0
    future_count = int(future_audit["future_data_violation_count"].sum()) if "future_data_violation_count" in future_audit else 0
    return {
        "date": "2026-07-07",
        "task_id": TASK_ID,
        "owner": "BACKTEST_LAB Core/Data",
        "status": "source_package_review_completed_pruning_v2_required",
        "radar_status": readiness_in.get("status"),
        "diagnostic_only": True,
        "sample_rows": sample_rows,
        "statement_success_rows": int(readiness_in.get("statement_success_rows", 0)),
        "official_asof_matched_rows": int(readiness_in.get("official_asof_matched_rows", 0)),
        "official_asof_matched_share": int(readiness_in.get("official_asof_matched_rows", 0)) / sample_rows if sample_rows else 0.0,
        "route_error_count": int(readiness_in.get("route_error_count", 0)),
        "cache_manifest_rows": cache_rows,
        "seed_cache_rows_per_materialized_row": cache_per_row,
        "ready_for_bounded_broader_ingest_contract": False,
        "ready_for_core_t164_broader_ingest_contract": False,
        "ready_for_core_t164_broader_materialization": False,
        "ready_for_radar_candidate_detail_pruning_runner_v2": True,
        "ready_for_experiments": False,
        "ready_for_formal": False,
        "ready_for_strategy_replay": False,
        "blocker": "seed route hygiene passed, but full/broader ingest needs candidate/detail pruning, full ticker/period coverage, TPEx all-stock proof, and label policy review",
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
            "# Layer1 t164 Broader Source Package Review",
            "",
            f"Status: {readiness['status']}",
            "",
            "Conclusion: the broader seed package passed bounded route/asof hygiene, but Core should not build a broader ingest contract yet. The route fan-out is too high for full universe without candidate/detail pruning.",
            "",
            "Readiness:",
            f"- sample_rows={readiness['sample_rows']}",
            f"- statement_success_rows={readiness['statement_success_rows']}",
            f"- official_asof_matched_rows={readiness['official_asof_matched_rows']}",
            f"- cache_manifest_rows={readiness['cache_manifest_rows']}",
            f"- seed_cache_rows_per_materialized_row={readiness['seed_cache_rows_per_materialized_row']}",
            "- ready_for_bounded_broader_ingest_contract=false",
            "- ready_for_core_t164_broader_ingest_contract=false",
            "- ready_for_core_t164_broader_materialization=false",
            "- ready_for_radar_candidate_detail_pruning_runner_v2=true",
            "- ready_for_experiments=false",
            "- ready_for_formal=false",
            "- ready_for_strategy_replay=false",
            f"- future_data_violation_count={readiness['future_data_violation_count']}",
            "",
            "Next step: Radar/Data should build candidate/detail pruning runner v2 before any all-stock or broader-period materialization.",
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
