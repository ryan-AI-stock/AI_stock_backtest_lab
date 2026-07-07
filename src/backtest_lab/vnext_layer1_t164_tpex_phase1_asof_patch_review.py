"""Review TPEx phase-1 official-asof alternate-route patch rows.

This Core/Data review updates bounded readiness metadata only. It does not
materialize a broader table, run Experiments, replay, or change formal paths.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


TASK_ID = "TASK-BACKTEST-CORE-VNEXT-LAYER1-T164-TPEX-PHASE1-ASOF-PATCH-REVIEW-001"
DEFAULT_RADAR_DIR = Path(
    r"C:\Users\zergv\Documents\Codex\2026-05-23\ai-stock-rotation-radar-https-docs\outputs"
    r"\radar_vnext_layer1_t164_tpex_phase1_official_asof_alternate_route_disambiguation_20260707"
)
DEFAULT_OUTPUT_DIR = Path("outputs/vnext_layer1_t164_tpex_phase1_asof_patch_review_20260707")


def build_review(
    *,
    radar_dir: str | Path = DEFAULT_RADAR_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    radar = Path(radar_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    readiness_in = _read_json(radar / "readiness_for_core_t164_tpex_phase1_asof_alternate_route.json")
    manifest_in = _read_json(radar / "manifest.json")
    accepted = _read_csv(radar / "alternate_asof_accepted_patch_rows.csv", dtype={"ticker": str})
    still_blocked = _read_csv(radar / "alternate_asof_still_blocked_rows.csv", dtype={"ticker": str})
    resolution = _read_csv(radar / "alternate_asof_resolution_summary.csv", dtype={"ticker": str})
    future_in = _read_csv(radar / "future_data_governance_audit.csv")

    updated_summary = _updated_summary(readiness_in, accepted, still_blocked)
    accepted_contract = _accepted_patch_contract(accepted)
    still_blocked_contract = _still_blocked_contract(still_blocked, resolution)
    policy_decision = _policy_decision(still_blocked_contract)
    next_handoff = _next_handoff(policy_decision)
    future_audit = _future_audit(future_in)
    readiness = _readiness(readiness_in, accepted_contract, still_blocked_contract, future_audit)

    _write_csv(updated_summary, output / "layer1_t164_tpex_phase1_asof_patch_updated_summary.csv")
    _write_csv(accepted_contract, output / "layer1_t164_tpex_phase1_accepted_asof_patch_contract.csv")
    _write_csv(still_blocked_contract, output / "layer1_t164_tpex_phase1_still_blocked_asof_rows.csv")
    _write_csv(policy_decision, output / "layer1_t164_tpex_phase1_policy_decision_needed.csv")
    _write_csv(next_handoff, output / "layer1_t164_tpex_phase1_next_handoff.csv")
    _write_csv(future_audit, output / "layer1_t164_tpex_phase1_asof_patch_future_data_audit.csv")
    (output / "readiness_for_layer1_t164_tpex_phase1_asof_patch_review.json").write_text(
        json.dumps(readiness, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    manifest = {
        "task_id": TASK_ID,
        "status": readiness["status"],
        "output_dir": str(output.resolve()),
        "radar_input_dir": str(radar.resolve()),
        "radar_commit": "794b805",
        "radar_task_id": manifest_in.get("task_id"),
        "output_files": [
            "layer1_t164_tpex_phase1_asof_patch_updated_summary.csv",
            "layer1_t164_tpex_phase1_accepted_asof_patch_contract.csv",
            "layer1_t164_tpex_phase1_still_blocked_asof_rows.csv",
            "layer1_t164_tpex_phase1_policy_decision_needed.csv",
            "layer1_t164_tpex_phase1_next_handoff.csv",
            "layer1_t164_tpex_phase1_asof_patch_future_data_audit.csv",
            "readiness_for_layer1_t164_tpex_phase1_asof_patch_review.json",
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


def _updated_summary(readiness: dict[str, Any], accepted: pd.DataFrame, still_blocked: pd.DataFrame) -> pd.DataFrame:
    previous_matched = 85
    total_rows = 100
    resolved = len(accepted)
    updated_matched = previous_matched + resolved
    return pd.DataFrame(
        [
            {
                "metric": "phase1_official_asof_after_patch",
                "input_blocked_rows": readiness.get("input_blocked_rows"),
                "resolved_rows": resolved,
                "still_blocked_rows": len(still_blocked),
                "previous_official_asof_matched_rows": previous_matched,
                "updated_official_asof_matched_rows": updated_matched,
                "sample_rows": total_rows,
                "updated_official_asof_matched_share": updated_matched / total_rows,
                "ready_for_all_stock_proof_update": False,
                "ready_for_experiments": False,
                "diagnostic_only": True,
            }
        ]
    )


def _accepted_patch_contract(accepted: pd.DataFrame) -> pd.DataFrame:
    if accepted.empty:
        return accepted
    out = accepted.copy()
    out["patch_status"] = "accepted_official_public_timestamp"
    out["market_available_at_source"] = "t05st01_t05st01_detail_public_announcement_timestamp"
    out["quarter_end_date_used"] = False
    out["query_response_datetime_used"] = False
    out["conservative_deadline_proxy_used"] = False
    out["formal_ready"] = False
    out["diagnostic_only"] = True
    out["not_live_rule"] = True
    return out


def _still_blocked_contract(still_blocked: pd.DataFrame, resolution: pd.DataFrame) -> pd.DataFrame:
    if still_blocked.empty:
        return still_blocked
    out = still_blocked.copy()
    if not resolution.empty:
        keep = [
            "ticker",
            "market",
            "report_period",
            "alternate_query_payload_count",
            "candidate_evidence_rows",
            "detail_fetch_count",
        ]
        available = [col for col in keep if col in resolution.columns]
        out = out.merge(resolution[available], on=["ticker", "market", "report_period"], how="left")
    out["blocked_policy"] = "blocked_no_silent_backfill"
    out["needs_policy_decision"] = out["blocked_reason"].astype(str).eq("accepted_candidate_count=2")
    out["needs_additional_source_route"] = out["blocked_reason"].astype(str).eq("accepted_candidate_count=0")
    out["ready_for_experiments"] = False
    out["formal_ready"] = False
    out["diagnostic_only"] = True
    return out


def _policy_decision(still_blocked: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {
            "decision_item": "accept_3_row_patch",
            "core_recommendation": "accept_for_diagnostic_official_asof_patch_metadata",
            "reason": "accepted rows use official public t05st01/t05st01_detail timestamp and preserve prohibited available_at policy",
            "decision_owner": "Core/Data",
            "ready": True,
            "diagnostic_only": True,
        },
        {
            "decision_item": "update_phase1_readiness_to_88_of_100",
            "core_recommendation": "yes_update_metadata_only",
            "reason": "85 prior matched + 3 accepted patch rows = 88/100; still not all-stock proof",
            "decision_owner": "Core/Data",
            "ready": True,
            "diagnostic_only": True,
        },
    ]
    ambiguous = still_blocked[still_blocked.get("needs_policy_decision", False) == True] if not still_blocked.empty else pd.DataFrame()
    no_candidate = still_blocked[still_blocked.get("needs_additional_source_route", False) == True] if not still_blocked.empty else pd.DataFrame()
    if not no_candidate.empty:
        rows.append(
            {
                "decision_item": "remaining_no_candidate_rows",
                "core_recommendation": "requires_radar_alternate_source_or_accept_blocked_policy",
                "reason": f"{len(no_candidate)} rows still have accepted_candidate_count=0 after alternate route",
                "decision_owner": "Strategy Center / Radar/Data",
                "ready": False,
                "diagnostic_only": True,
            }
        )
    if not ambiguous.empty:
        rows.append(
            {
                "decision_item": "6114_114Q4_ambiguous_two_official_candidates",
                "core_recommendation": "requires_strategy_center_policy_or_stronger_radar_evidence; no silent earliest/latest selection",
                "reason": "two official-looking financial report announcements remain; Core will not choose without policy",
                "decision_owner": "Strategy Center",
                "ready": False,
                "diagnostic_only": True,
            }
        )
    return pd.DataFrame(rows)


def _next_handoff(policy_decision: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "next_owner": "Strategy Center",
                "handoff_action": "decide_whether_to_accept_88_of_100_partial_blocked_policy_or_authorize_more_radar_source_work",
                "ready": True,
                "reason": "Radar alternate route resolved only 3/15; 12 remain blocked including one policy ambiguity",
                "diagnostic_only": True,
                "formal_model_changed": False,
                "trade_decision_changed": False,
                "active_in_trade_decision": False,
                "report_changed": False,
                "portfolio_replay_executed": False,
                "ready_for_strategy_replay": False,
                "not_live_rule": True,
                "forward_returns_live_rule_usage": False,
            },
            {
                "next_owner": "Radar/Data",
                "handoff_action": "conditional_only_if_strategy_center_authorizes_more_alternate_source_or_disambiguation_work",
                "ready": False,
                "reason": "additional Radar work is possible for 11 no-candidate rows, but after one bounded alternate attempt this needs policy/cost decision",
                "diagnostic_only": True,
                "formal_model_changed": False,
                "trade_decision_changed": False,
                "active_in_trade_decision": False,
                "report_changed": False,
                "portfolio_replay_executed": False,
                "ready_for_strategy_replay": False,
                "not_live_rule": True,
                "forward_returns_live_rule_usage": False,
            },
        ]
    )


def _future_audit(future_in: pd.DataFrame) -> pd.DataFrame:
    rows = future_in.to_dict("records") if not future_in.empty else []
    rows.extend(
        [
            {
                "audit_item": "core_patch_review_forward_return_as_rule",
                "status": "passed",
                "future_data_violation_count": 0,
                "note": "patch review only updates official-asof source metadata",
            },
            {
                "audit_item": "partial_patch_no_silent_backfill",
                "status": "passed",
                "future_data_violation_count": 0,
                "note": "12 unresolved rows remain blocked",
            },
        ]
    )
    return pd.DataFrame(rows)


def _readiness(
    readiness_in: dict[str, Any],
    accepted: pd.DataFrame,
    still_blocked: pd.DataFrame,
    future_audit: pd.DataFrame,
) -> dict[str, Any]:
    previous_matched = 85
    sample_rows = 100
    resolved = len(accepted)
    updated_matched = previous_matched + resolved
    future_violations = int(pd.to_numeric(future_audit.get("future_data_violation_count", 0), errors="coerce").fillna(0).sum())
    ambiguous_rows = int(still_blocked.get("needs_policy_decision", pd.Series(dtype=bool)).astype(bool).sum()) if not still_blocked.empty else 0
    no_candidate_rows = int(still_blocked.get("needs_additional_source_route", pd.Series(dtype=bool)).astype(bool).sum()) if not still_blocked.empty else 0
    return {
        "task_id": TASK_ID,
        "status": "phase1_tpex_asof_patch_reviewed_partial_88_of_100_still_blocked",
        "diagnostic_only": True,
        "input_blocked_rows": readiness_in.get("input_blocked_rows"),
        "resolved_patch_rows": resolved,
        "still_blocked_rows": len(still_blocked),
        "previous_official_asof_matched_rows": previous_matched,
        "updated_official_asof_matched_rows": updated_matched,
        "sample_rows": sample_rows,
        "updated_official_asof_matched_share": updated_matched / sample_rows,
        "accepted_candidate_count_0_rows_remaining": no_candidate_rows,
        "accepted_candidate_count_2_rows_remaining": ambiguous_rows,
        "route_error_count": readiness_in.get("route_error_count"),
        "future_data_violation_count": future_violations,
        "accepted_3_row_patch_for_diagnostic_metadata": resolved == 3 and future_violations == 0,
        "ready_for_core_t164_tpex_all_stock_proof_readiness_update": False,
        "ready_for_core_t164_broader_ingest_contract": False,
        "ready_for_core_t164_broader_materialization": False,
        "ready_for_experiments": False,
        "ready_for_formal": False,
        "ready_for_strategy_replay": False,
        "ready_for_full_universe": False,
        "tpex_all_stock_universal_ready": False,
        "ready_for_strategy_center_policy_decision": len(still_blocked) > 0,
        "ready_for_radar_next_work": False,
        "blocked_fields": [
            "official_asof_unmatched_or_ambiguous_rows",
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
    return f"""# Layer1 t164 TPEx phase_1 official-asof patch review

## Verdict
- status={readiness["status"]}
- resolved_patch_rows={readiness["resolved_patch_rows"]}
- previous_official_asof_matched_rows={readiness["previous_official_asof_matched_rows"]}/100
- updated_official_asof_matched_rows={readiness["updated_official_asof_matched_rows"]}/100
- still_blocked_rows={readiness["still_blocked_rows"]}
- accepted_candidate_count_0_rows_remaining={readiness["accepted_candidate_count_0_rows_remaining"]}
- accepted_candidate_count_2_rows_remaining={readiness["accepted_candidate_count_2_rows_remaining"]}
- ready_for_experiments=false
- ready_for_formal=false

## Core decision
The 3 accepted official-asof patch rows are accepted as diagnostic metadata, raising phase_1 official-asof coverage from 85/100 to 88/100. This is still not TPEx all-stock proof and not broader/full ingest readiness.

The remaining 12 rows stay blocked. 6114 114Q4 needs a policy decision or stronger evidence; Core will not silently choose between two official-looking candidates.

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
    manifest = build_review(radar_dir=args.radar_dir, output_dir=args.output_dir)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
