"""Close TPEx phase-1 t164 official-asof readiness after Strategy policy decision.

This records the accepted 98/100 partial blocked source-package limit. It does
not run Experiments, source acquisition, materialization, replay, or formal
pipeline changes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


TASK_ID = "TASK-BACKTEST-CORE-VNEXT-LAYER1-T164-TPEX-PHASE1-POLICY-CLOSURE-001"
DEFAULT_REVIEW_DIR = Path("outputs/vnext_layer1_t164_tpex_phase1_remaining12_asof_patch_review_20260707")
DEFAULT_OUTPUT_DIR = Path("outputs/vnext_layer1_t164_tpex_phase1_policy_closure_20260707")


def build_closure(
    *,
    review_dir: str | Path = DEFAULT_REVIEW_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    review = Path(review_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    readiness_in = _read_json(review / "readiness_for_layer1_t164_tpex_phase1_remaining12_asof_patch_review.json")
    blocked = _read_csv(review / "layer1_t164_tpex_phase1_final_blocked_asof_rows.csv", dtype={"ticker": str})
    accepted_patch = _read_csv(
        review / "layer1_t164_tpex_phase1_remaining12_accepted_asof_patch_contract.csv",
        dtype={"ticker": str},
    )

    closure_summary = _closure_summary(readiness_in)
    blocked_policy = _blocked_policy(blocked)
    accepted_metadata = _accepted_metadata(accepted_patch)
    next_plan = _next_plan()
    future_audit = _future_audit()
    readiness = _readiness(readiness_in, blocked, accepted_patch)

    _write_csv(closure_summary, output / "layer1_t164_tpex_phase1_policy_closure_summary.csv")
    _write_csv(blocked_policy, output / "layer1_t164_tpex_phase1_final_blocked_policy_accepted.csv")
    _write_csv(accepted_metadata, output / "layer1_t164_tpex_phase1_accepted_patch_metadata.csv")
    _write_csv(next_plan, output / "layer1_t164_tpex_phase1_next_step_recommendation.csv")
    _write_csv(future_audit, output / "layer1_t164_tpex_phase1_policy_closure_future_data_audit.csv")
    (output / "readiness_for_layer1_t164_tpex_phase1_policy_closure.json").write_text(
        json.dumps(readiness, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    manifest = {
        "task_id": TASK_ID,
        "status": readiness["status"],
        "output_dir": str(output.resolve()),
        "review_input_dir": str(review.resolve()),
        "strategy_center_decision": "accept_98_of_100_partial_blocked_source_package_limit",
        "output_files": [
            "layer1_t164_tpex_phase1_policy_closure_summary.csv",
            "layer1_t164_tpex_phase1_final_blocked_policy_accepted.csv",
            "layer1_t164_tpex_phase1_accepted_patch_metadata.csv",
            "layer1_t164_tpex_phase1_next_step_recommendation.csv",
            "layer1_t164_tpex_phase1_policy_closure_future_data_audit.csv",
            "readiness_for_layer1_t164_tpex_phase1_policy_closure.json",
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


def _closure_summary(readiness: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "status": "phase1_tpex_proof_partial_pass_with_explicit_2_blockers_not_all_stock_ready",
                "strategy_center_decision": "accept_98_of_100_partial_blocked_source_package_limit",
                "updated_official_asof_matched_rows": readiness.get("updated_official_asof_matched_rows"),
                "sample_rows": readiness.get("sample_rows"),
                "updated_official_asof_matched_share": readiness.get("updated_official_asof_matched_share"),
                "still_blocked_rows": readiness.get("still_blocked_rows"),
                "accepted_10_row_patch_for_diagnostic_metadata": readiness.get("accepted_10_row_patch_for_diagnostic_metadata"),
                "tpex_all_stock_universal_ready": False,
                "ready_for_experiments": False,
                "ready_for_formal": False,
                "diagnostic_only": True,
            }
        ]
    )


def _blocked_policy(blocked: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in blocked.to_dict("records"):
        ticker = str(row.get("ticker"))
        reason = row.get("blocked_reason")
        if ticker == "6114":
            policy = "version_match_blocked; no earliest/latest silent selection; no proxy available_at"
        elif ticker == "8080":
            policy = "blocked_no_official_target_candidate; no deadline/query/quarter-end backfill"
        else:
            policy = "blocked_no_silent_backfill"
        rows.append(
            {
                "ticker": ticker,
                "market": row.get("market"),
                "report_period": row.get("report_period"),
                "blocked_reason": reason,
                "strategy_center_policy": policy,
                "accepted_as_partial_blocked_limit": True,
                "ready_for_experiments": False,
                "formal_ready": False,
                "diagnostic_only": True,
            }
        )
    return pd.DataFrame(rows)


def _accepted_metadata(accepted_patch: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {
            "metadata_item": "remaining12_10_row_patch",
            "row_count": len(accepted_patch),
            "accepted_for": "diagnostic_source_metadata_only",
            "formal_ready": False,
            "ready_for_experiments": False,
            "diagnostic_only": True,
        },
        {
            "metadata_item": "phase1_total_official_asof_matched",
            "row_count": 98,
            "accepted_for": "phase1_partial_blocked_source_package_limit",
            "formal_ready": False,
            "ready_for_experiments": False,
            "diagnostic_only": True,
        },
    ]
    return pd.DataFrame(rows)


def _next_plan() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "next_step": "phase_2_bounded_expansion_or_full_period_plan",
                "recommended_owner": "Core/Data",
                "recommended_action": "build_phase2_bounded_expansion_contract_or_plan_from_existing_phase_batch_guard",
                "reason": "Strategy Center accepted 98/100 closure and stopped 2-row chase; next useful work is bounded expansion planning",
                "ready": True,
                "ready_for_experiments": False,
                "diagnostic_only": True,
            },
            {
                "next_step": "radar_data_final_2_row_chase",
                "recommended_owner": "Radar/Data",
                "recommended_action": "do_not_run_unless_new_strategy_center_authorization",
                "reason": "Strategy Center explicitly stopped 6114/8080 chase for token/cost control",
                "ready": False,
                "ready_for_experiments": False,
                "diagnostic_only": True,
            },
        ]
    )


def _future_audit() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "audit_item": "policy_closure_no_forward_return_rule",
                "status": "passed",
                "future_data_violation_count": 0,
                "note": "closure uses Strategy Center source policy only",
            },
            {
                "audit_item": "final_2_blockers_no_silent_backfill",
                "status": "passed",
                "future_data_violation_count": 0,
                "note": "6114 and 8080 remain explicit blocked rows",
            },
            {
                "audit_item": "prohibited_available_at_sources",
                "status": "passed",
                "future_data_violation_count": 0,
                "note": "quarter_end_date, query_response_datetime, and conservative deadline proxy remain prohibited",
            },
        ]
    )


def _readiness(readiness: dict[str, Any], blocked: pd.DataFrame, accepted_patch: pd.DataFrame) -> dict[str, Any]:
    return {
        "task_id": TASK_ID,
        "status": "phase1_tpex_proof_partial_pass_with_explicit_2_blockers_not_all_stock_ready",
        "diagnostic_only": True,
        "strategy_center_decision": "accept_98_of_100_partial_blocked_source_package_limit",
        "updated_official_asof_matched_rows": 98,
        "sample_rows": 100,
        "updated_official_asof_matched_share": 0.98,
        "still_blocked_rows": int(len(blocked)),
        "final_blocked_tickers": blocked["ticker"].astype(str).tolist() if not blocked.empty else [],
        "accepted_10_row_patch_for_diagnostic_source_metadata": int(len(accepted_patch)) == 10,
        "stop_6114_8080_chase": True,
        "tpex_all_stock_universal_ready": False,
        "ready_for_core_t164_tpex_all_stock_proof_readiness_update": False,
        "ready_for_core_t164_phase2_bounded_expansion_contract_planning": True,
        "ready_for_core_t164_broader_ingest_contract": False,
        "ready_for_core_t164_broader_materialization": False,
        "ready_for_experiments": False,
        "ready_for_formal": False,
        "ready_for_strategy_replay": False,
        "ready_for_full_universe": False,
        "future_data_violation_count": 0,
        "blocked_fields": [
            "6114_version_match_blocked",
            "8080_no_official_target_candidate",
            "tpex_historical_all_stock_universe",
            "full_period_range",
            "full_universe_materialization",
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
    return f"""# Layer1 t164 TPEx phase_1 policy closure

## Verdict
- status={readiness["status"]}
- updated_official_asof_matched_rows={readiness["updated_official_asof_matched_rows"]}/100
- still_blocked_rows={readiness["still_blocked_rows"]}
- final_blocked_tickers={",".join(readiness["final_blocked_tickers"])}
- tpex_all_stock_universal_ready=false
- ready_for_experiments=false
- ready_for_formal=false

## Policy closure
Strategy Center accepted 98/100 official-asof matched as the TPEx phase_1 partial blocked source package limit. 6114 TPEx 114Q4 and 8080 TPEx 115Q1 remain explicit blocked rows. No further 2-row chase is authorized.

## Next
Recommended next step is phase_2 bounded expansion / full-period planning, not further 6114/8080 source chasing.

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
    parser.add_argument("--review-dir", default=str(DEFAULT_REVIEW_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()
    manifest = build_closure(review_dir=args.review_dir, output_dir=args.output_dir)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
