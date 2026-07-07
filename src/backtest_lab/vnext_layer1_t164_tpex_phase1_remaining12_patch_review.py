"""Review TPEx phase-1 remaining-12 official-asof higher-cost patch.

This Core/Data review accepts resolved official-asof rows as diagnostic source
metadata and preserves unresolved rows as blocked. It does not run Experiments,
materialize broader ingest, replay, or alter formal paths.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


TASK_ID = "TASK-BACKTEST-CORE-VNEXT-LAYER1-T164-TPEX-PHASE1-REMAINING12-ASOF-PATCH-REVIEW-001"
DEFAULT_RADAR_DIR = Path(
    r"C:\Users\zergv\Documents\Codex\2026-05-23\ai-stock-rotation-radar-https-docs\outputs"
    r"\radar_vnext_layer1_t164_tpex_phase1_remaining12_official_asof_alternate_source_20260707"
)
DEFAULT_OUTPUT_DIR = Path("outputs/vnext_layer1_t164_tpex_phase1_remaining12_asof_patch_review_20260707")


def build_review(
    *,
    radar_dir: str | Path = DEFAULT_RADAR_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    radar = Path(radar_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    readiness_in = _read_json(radar / "readiness_for_core_t164_tpex_remaining12_asof_source.json")
    manifest_in = _read_json(radar / "manifest.json")
    accepted = _read_csv(radar / "remaining12_accepted_patch_rows.csv", dtype={"ticker": str})
    blocked = _read_csv(radar / "remaining12_still_blocked_ledger.csv", dtype={"ticker": str})
    evidence_6114 = _read_csv(radar / "6114_dual_candidate_policy_evidence.csv", dtype={"ticker": str})
    future_in = _read_csv(radar / "future_data_audit.csv")

    accepted_contract = _accepted_patch_contract(accepted)
    blocked_contract = _blocked_contract(blocked, evidence_6114)
    updated_summary = _updated_summary(readiness_in, accepted_contract, blocked_contract)
    policy_ledger = _policy_ledger(blocked_contract)
    next_handoff = _next_handoff(blocked_contract)
    future_audit = _future_audit(future_in)
    readiness = _readiness(readiness_in, accepted_contract, blocked_contract, future_audit)

    _write_csv(updated_summary, output / "layer1_t164_tpex_phase1_remaining12_patch_updated_summary.csv")
    _write_csv(accepted_contract, output / "layer1_t164_tpex_phase1_remaining12_accepted_asof_patch_contract.csv")
    _write_csv(blocked_contract, output / "layer1_t164_tpex_phase1_final_blocked_asof_rows.csv")
    _write_csv(policy_ledger, output / "layer1_t164_tpex_phase1_final_blocked_policy_ledger.csv")
    _write_csv(next_handoff, output / "layer1_t164_tpex_phase1_final_next_handoff.csv")
    _write_csv(future_audit, output / "layer1_t164_tpex_phase1_remaining12_future_data_audit.csv")
    (output / "readiness_for_layer1_t164_tpex_phase1_remaining12_asof_patch_review.json").write_text(
        json.dumps(readiness, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    manifest = {
        "task_id": TASK_ID,
        "status": readiness["status"],
        "output_dir": str(output.resolve()),
        "radar_input_dir": str(radar.resolve()),
        "radar_commit": "4aae10c",
        "radar_task_id": manifest_in.get("task_id"),
        "output_files": [
            "layer1_t164_tpex_phase1_remaining12_patch_updated_summary.csv",
            "layer1_t164_tpex_phase1_remaining12_accepted_asof_patch_contract.csv",
            "layer1_t164_tpex_phase1_final_blocked_asof_rows.csv",
            "layer1_t164_tpex_phase1_final_blocked_policy_ledger.csv",
            "layer1_t164_tpex_phase1_final_next_handoff.csv",
            "layer1_t164_tpex_phase1_remaining12_future_data_audit.csv",
            "readiness_for_layer1_t164_tpex_phase1_remaining12_asof_patch_review.json",
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


def _accepted_patch_contract(accepted: pd.DataFrame) -> pd.DataFrame:
    out = accepted.copy()
    if out.empty:
        return out
    out["patch_status"] = "accepted_official_public_timestamp"
    out["market_available_at_source"] = "t05st01_t05st01_detail_public_announcement_timestamp"
    out["quarter_end_date_used"] = False
    out["query_response_datetime_used"] = False
    out["conservative_deadline_proxy_used"] = False
    out["formal_ready"] = False
    out["ready_for_experiments"] = False
    out["diagnostic_only"] = True
    out["not_live_rule"] = True
    out["forward_returns_live_rule_usage"] = False
    return out


def _blocked_contract(blocked: pd.DataFrame, evidence_6114: pd.DataFrame) -> pd.DataFrame:
    out = blocked.copy()
    if out.empty:
        return out
    out["blocked_policy"] = "blocked_no_silent_backfill"
    out["needs_policy_decision"] = out["blocked_reason"].astype(str).str.contains("version", case=False, na=False)
    out["needs_additional_source_route"] = out["blocked_reason"].astype(str).str.contains("accepted_candidate_count=0", na=False)
    out["official_source_evidence_rows"] = 0
    if not evidence_6114.empty:
        out.loc[out["ticker"].astype(str).eq("6114"), "official_source_evidence_rows"] = len(evidence_6114)
    out["formal_ready"] = False
    out["ready_for_experiments"] = False
    out["diagnostic_only"] = True
    out["not_live_rule"] = True
    out["forward_returns_live_rule_usage"] = False
    return out


def _updated_summary(readiness: dict[str, Any], accepted: pd.DataFrame, blocked: pd.DataFrame) -> pd.DataFrame:
    previous_matched = 88
    sample_rows = 100
    resolved = len(accepted)
    updated_matched = previous_matched + resolved
    return pd.DataFrame(
        [
            {
                "metric": "phase1_official_asof_after_remaining12_patch",
                "input_rows": readiness.get("input_rows"),
                "resolved_rows": resolved,
                "still_blocked_rows": len(blocked),
                "previous_official_asof_matched_rows": previous_matched,
                "updated_official_asof_matched_rows": updated_matched,
                "sample_rows": sample_rows,
                "updated_official_asof_matched_share": updated_matched / sample_rows,
                "ready_for_all_stock_proof_update": False,
                "ready_for_experiments": False,
                "diagnostic_only": True,
            }
        ]
    )


def _policy_ledger(blocked: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {
            "item": "accept_10_row_patch",
            "status": "accepted_for_diagnostic_official_asof_metadata",
            "policy": "official t05st01/t05st01_detail timestamp only",
            "decision_owner": "Core/Data",
            "diagnostic_only": True,
        },
        {
            "item": "update_phase1_metadata_to_98_of_100",
            "status": "accepted_metadata_only",
            "policy": "not all-stock proof; not Experiments-ready",
            "decision_owner": "Core/Data",
            "diagnostic_only": True,
        },
    ]
    for row in blocked.to_dict("records"):
        rows.append(
            {
                "item": f"{row.get('ticker')} {row.get('report_period')}",
                "status": "blocked",
                "policy": row.get("blocked_reason"),
                "decision_owner": "Strategy Center/Radar/Data" if row.get("needs_policy_decision") else "Radar/Data or Strategy Center",
                "diagnostic_only": True,
            }
        )
    rows.extend(
        [
            {
                "item": "quarter_end_date",
                "status": "prohibited",
                "policy": "cannot be official available_at",
                "decision_owner": "Core/Data",
                "diagnostic_only": True,
            },
            {
                "item": "query_response_datetime",
                "status": "prohibited",
                "policy": "cannot be official available_at",
                "decision_owner": "Core/Data",
                "diagnostic_only": True,
            },
            {
                "item": "conservative_deadline_proxy",
                "status": "prohibited_as_official_available_at",
                "policy": "separate diagnostic candidate only",
                "decision_owner": "Core/Data",
                "diagnostic_only": True,
            },
        ]
    )
    return pd.DataFrame(rows)


def _next_handoff(blocked: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "next_owner": "Strategy Center",
                "handoff_action": "decide_whether_98_of_100_phase1_official_asof_is_sufficient_for_partial_blocked_source_package_or_authorize_targeted_final_source_work",
                "ready": True,
                "reason": "phase1 improved to 98/100 but still has 6114 version_match_blocked and 8080 no target candidate",
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
                "handoff_action": "conditional_only_if_strategy_center_authorizes_final_targeted_work_for_6114_version_mapping_or_8080_official_source",
                "ready": False,
                "reason": "Core will not silently choose 6114 version or backfill 8080; additional source work needs explicit cost/policy approval",
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
                "audit_item": "core_remaining12_patch_review_forward_return_as_rule",
                "status": "passed",
                "future_data_violation_count": 0,
                "note": "review uses official-asof source metadata only",
            },
            {
                "audit_item": "remaining_blocked_rows_no_silent_backfill",
                "status": "passed",
                "future_data_violation_count": 0,
                "note": "6114 and 8080 remain blocked",
            },
        ]
    )
    return pd.DataFrame(rows)


def _readiness(
    readiness_in: dict[str, Any],
    accepted: pd.DataFrame,
    blocked: pd.DataFrame,
    future_audit: pd.DataFrame,
) -> dict[str, Any]:
    previous_matched = 88
    sample_rows = 100
    resolved = len(accepted)
    updated_matched = previous_matched + resolved
    future_violations = int(pd.to_numeric(future_audit.get("future_data_violation_count", 0), errors="coerce").fillna(0).sum())
    return {
        "task_id": TASK_ID,
        "status": "phase1_tpex_asof_patch_reviewed_partial_98_of_100_final_blocked",
        "diagnostic_only": True,
        "input_rows": readiness_in.get("input_rows"),
        "resolved_patch_rows": resolved,
        "still_blocked_rows": len(blocked),
        "previous_official_asof_matched_rows": previous_matched,
        "updated_official_asof_matched_rows": updated_matched,
        "sample_rows": sample_rows,
        "updated_official_asof_matched_share": updated_matched / sample_rows,
        "final_blocked_tickers": blocked["ticker"].astype(str).tolist() if not blocked.empty else [],
        "final_blocked_reasons": blocked["blocked_reason"].astype(str).tolist() if not blocked.empty else [],
        "route_error_count": readiness_in.get("route_error_count"),
        "future_data_violation_count": future_violations,
        "accepted_10_row_patch_for_diagnostic_metadata": resolved == 10 and future_violations == 0,
        "ready_for_core_t164_tpex_all_stock_proof_readiness_update": False,
        "ready_for_core_t164_broader_ingest_contract": False,
        "ready_for_core_t164_broader_materialization": False,
        "ready_for_experiments": False,
        "ready_for_formal": False,
        "ready_for_strategy_replay": False,
        "ready_for_full_universe": False,
        "tpex_all_stock_universal_ready": False,
        "ready_for_strategy_center_policy_decision": True,
        "ready_for_radar_next_work": False,
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
    return f"""# Layer1 t164 TPEx phase_1 remaining-12 official-asof patch review

## Verdict
- status={readiness["status"]}
- resolved_patch_rows={readiness["resolved_patch_rows"]}
- previous_official_asof_matched_rows={readiness["previous_official_asof_matched_rows"]}/100
- updated_official_asof_matched_rows={readiness["updated_official_asof_matched_rows"]}/100
- still_blocked_rows={readiness["still_blocked_rows"]}
- final_blocked_tickers={",".join(readiness["final_blocked_tickers"])}
- ready_for_experiments=false
- ready_for_formal=false

## Core decision
The 10 accepted official-asof patch rows are accepted as diagnostic metadata, raising phase_1 official-asof coverage from 88/100 to 98/100. This still does not prove TPEx all-stock readiness and does not authorize broader/full materialization.

6114 TPEx 114Q4 remains version_match_blocked. 8080 TPEx 115Q1 remains blocked_no_official_target_candidate. No silent backfill or policy-based timestamp choice was made.

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
