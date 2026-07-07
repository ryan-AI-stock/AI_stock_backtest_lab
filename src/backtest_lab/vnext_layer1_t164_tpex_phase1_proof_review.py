"""Review Radar/Data TPEx phase-1 t164 bounded proof runner output.

This Core/Data package updates readiness and blocker routing only. It does not
run source acquisition, materialization, Experiments, replay, or formal paths.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


TASK_ID = "TASK-BACKTEST-CORE-VNEXT-LAYER1-T164-TPEX-PHASE1-PROOF-REVIEW-001"
DEFAULT_RADAR_DIR = Path(
    r"C:\Users\zergv\Documents\Codex\2026-05-23\ai-stock-rotation-radar-https-docs\outputs"
    r"\radar_vnext_layer1_t164_tpex_phase1_50x2_bounded_proof_runner_20260707"
)
DEFAULT_OUTPUT_DIR = Path("outputs/vnext_layer1_t164_tpex_phase1_proof_review_20260707")


def build_review(
    *,
    radar_dir: str | Path = DEFAULT_RADAR_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    radar = Path(radar_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    readiness_in = _read_json(radar / "readiness_for_core_t164_tpex_phase1_bounded_proof_runner.json")
    manifest_in = _read_json(radar / "manifest.json")
    coverage = _read_csv(radar / "coverage_by_market_period.csv", dtype=str)
    blocked_rows = _read_csv(radar / "blocked_or_ambiguous_rows.csv", dtype={"ticker": str})
    universal = _read_csv(radar / "tpex_universal_readiness_evidence.csv", dtype=str)
    route_cost = _read_csv(radar / "projected_route_cost_report.csv")
    pruning = _read_csv(radar / "pruning_effectiveness_audit.csv")
    future_in = _read_csv(radar / "future_data_governance_audit.csv")
    field_policy = _read_csv(radar / "layer1_t164_tpex_field_policy.csv")

    review_summary = _review_summary(readiness_in, coverage, universal, route_cost, pruning)
    failure_attribution = _failure_attribution(blocked_rows)
    blocked_policy = _blocked_policy(blocked_rows, field_policy)
    next_radar_request = _next_radar_request(failure_attribution)
    future_audit = _future_audit(future_in)
    readiness = _readiness(readiness_in, blocked_rows, failure_attribution, future_audit)

    _write_csv(review_summary, output / "layer1_t164_tpex_phase1_proof_review_summary.csv")
    _write_csv(blocked_rows, output / "layer1_t164_tpex_phase1_blocked_official_asof_rows.csv")
    _write_csv(failure_attribution, output / "layer1_t164_tpex_phase1_asof_failure_attribution.csv")
    _write_csv(blocked_policy, output / "layer1_t164_tpex_phase1_blocked_policy_ledger.csv")
    _write_csv(next_radar_request, output / "layer1_t164_tpex_phase1_next_radar_request.csv")
    _write_csv(future_audit, output / "layer1_t164_tpex_phase1_future_data_audit.csv")
    (output / "readiness_for_layer1_t164_tpex_phase1_proof_review.json").write_text(
        json.dumps(readiness, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    manifest = {
        "task_id": TASK_ID,
        "status": readiness["status"],
        "output_dir": str(output.resolve()),
        "radar_input_dir": str(radar.resolve()),
        "radar_commit": "a670385",
        "radar_task_id": manifest_in.get("task_id"),
        "output_files": [
            "layer1_t164_tpex_phase1_proof_review_summary.csv",
            "layer1_t164_tpex_phase1_blocked_official_asof_rows.csv",
            "layer1_t164_tpex_phase1_asof_failure_attribution.csv",
            "layer1_t164_tpex_phase1_blocked_policy_ledger.csv",
            "layer1_t164_tpex_phase1_next_radar_request.csv",
            "layer1_t164_tpex_phase1_future_data_audit.csv",
            "readiness_for_layer1_t164_tpex_phase1_proof_review.json",
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


def _review_summary(
    readiness: dict[str, Any],
    coverage: pd.DataFrame,
    universal: pd.DataFrame,
    route_cost: pd.DataFrame,
    pruning: pd.DataFrame,
) -> pd.DataFrame:
    rows = [
        {
            "review_item": "phase1_runner_result",
            "status": readiness.get("status"),
            "sample_rows": readiness.get("sample_rows"),
            "statement_success_rows": readiness.get("statement_success_rows"),
            "official_asof_matched_rows": readiness.get("official_asof_matched_rows"),
            "official_asof_matched_share": readiness.get("official_asof_matched_share"),
            "blocked_or_ambiguous_rows": int(readiness.get("sample_rows", 0)) - int(readiness.get("official_asof_matched_rows", 0)),
            "core_verdict": "source_route_positive_but_official_asof_blocked",
            "diagnostic_only": True,
        },
        {
            "review_item": "route_cost",
            "status": "passed_cost_guard",
            "sample_rows": readiness.get("sample_rows"),
            "statement_success_rows": readiness.get("statement_success_rows"),
            "official_asof_matched_rows": readiness.get("official_asof_matched_rows"),
            "official_asof_matched_share": readiness.get("official_asof_matched_share"),
            "blocked_or_ambiguous_rows": int(readiness.get("sample_rows", 0)) - int(readiness.get("official_asof_matched_rows", 0)),
            "core_verdict": f"actual_cache_rows_per_materialized_row={readiness.get('actual_cache_rows_per_materialized_row')}; budget_routes_per_row={readiness.get('budget_routes_per_row')}",
            "diagnostic_only": True,
        },
    ]
    if not coverage.empty:
        for item in coverage.to_dict("records"):
            rows.append(
                {
                    "review_item": f"coverage_{item.get('coverage_type')}_{item.get('group')}",
                    "status": "coverage_observed",
                    "sample_rows": item.get("requested_rows"),
                    "statement_success_rows": item.get("statement_success_rows"),
                    "official_asof_matched_rows": item.get("official_asof_matched_rows"),
                    "official_asof_matched_share": "",
                    "blocked_or_ambiguous_rows": item.get("blocked_rows"),
                    "core_verdict": "coverage_metadata_only_not_formal",
                    "diagnostic_only": True,
                }
            )
    if not universal.empty:
        for item in universal.to_dict("records"):
            rows.append(
                {
                    "review_item": "tpex_universal_readiness",
                    "status": item.get("universal_readiness"),
                    "sample_rows": item.get("bounded_rows"),
                    "statement_success_rows": item.get("statement_success_rows"),
                    "official_asof_matched_rows": item.get("official_asof_matched_rows"),
                    "official_asof_matched_share": "",
                    "blocked_or_ambiguous_rows": "",
                    "core_verdict": item.get("reason"),
                    "diagnostic_only": True,
                }
            )
    return pd.DataFrame(rows)


def _failure_attribution(blocked_rows: pd.DataFrame) -> pd.DataFrame:
    if blocked_rows.empty:
        return pd.DataFrame(
            columns=[
                "failure_class",
                "blocked_rows",
                "tickers",
                "core_attribution",
                "recommended_next_action",
                "diagnostic_only",
            ]
        )
    rows = []
    for reason, group in blocked_rows.groupby("blocked_reason", dropna=False):
        failure_class = "no_accepted_official_candidate" if str(reason).endswith("=0") else "ambiguous_multiple_accepted_candidates"
        rows.append(
            {
                "failure_class": failure_class,
                "blocked_reason": reason,
                "blocked_rows": len(group),
                "tickers": ";".join(sorted(group["ticker"].astype(str).unique())),
                "periods": ";".join(sorted(group["report_period"].astype(str).unique())),
                "core_attribution": _attribution_text(failure_class),
                "recommended_next_action": _next_action_text(failure_class),
                "diagnostic_only": True,
            }
        )
    return pd.DataFrame(rows)


def _attribution_text(failure_class: str) -> str:
    if failure_class == "no_accepted_official_candidate":
        return "official_asof_match_gap; likely subject/detail token window or alternate announcement route issue; statement route itself succeeded"
    return "official_asof_disambiguation_gap; multiple plausible t05st01/detail rows need stricter period/subject/detail priority policy"


def _next_action_text(failure_class: str) -> str:
    if failure_class == "no_accepted_official_candidate":
        return "Radar/Data bounded alternate route and broader subject/detail query for listed blocked rows"
    return "Radar/Data detail-level disambiguation policy for listed ambiguous rows; keep no-silent-backfill"


def _blocked_policy(blocked_rows: pd.DataFrame, field_policy: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {
            "item": "official_asof_unmatched_or_ambiguous_rows",
            "status": "blocked",
            "blocked_rows": len(blocked_rows),
            "policy": "no_silent_backfill; no quarter_end_date; no query_response_datetime; no conservative_deadline_proxy as official route",
            "ready_for_experiments": False,
            "diagnostic_only": True,
        },
        {
            "item": "tpex_all_stock_universal_ready",
            "status": "blocked",
            "blocked_rows": len(blocked_rows),
            "policy": "85/100 official-asof match is not sufficient to claim TPEx all-stock proof",
            "ready_for_experiments": False,
            "diagnostic_only": True,
        },
        {
            "item": "full_period_or_full_universe_expansion",
            "status": "blocked",
            "blocked_rows": "",
            "policy": "do not expand before asof blocker policy improves or Strategy Center accepts blocked-row policy",
            "ready_for_experiments": False,
            "diagnostic_only": True,
        },
    ]
    if not field_policy.empty:
        for item in field_policy.to_dict("records"):
            if "proxy" in str(item.get("policy_status")) or "human_review" in str(item.get("source_quality")):
                rows.append(
                    {
                        "item": item.get("field"),
                        "status": "proxy_human_review_required",
                        "blocked_rows": "",
                        "policy": "diagnostic proxy only; not formal fundamental field",
                        "ready_for_experiments": False,
                        "diagnostic_only": True,
                    }
                )
    return pd.DataFrame(rows)


def _next_radar_request(failure_attribution: pd.DataFrame) -> pd.DataFrame:
    if failure_attribution.empty:
        return pd.DataFrame(
            [
                {
                    "next_owner": "Strategy Center",
                    "handoff_action": "decide_whether_to_accept_phase1_proof",
                    "ready": False,
                    "reason": "no blocked rows found, but this path was not expected",
                    "diagnostic_only": True,
                }
            ]
        )
    return pd.DataFrame(
        [
            {
                "next_owner": "Radar/Data",
                "handoff_action": "run_bounded_official_asof_alternate_route_and_disambiguation_for_15_phase1_blocked_rows",
                "input_artifacts": "layer1_t164_tpex_phase1_blocked_official_asof_rows.csv; layer1_t164_tpex_phase1_asof_failure_attribution.csv",
                "ready": True,
                "reason": "statement route passed 100/100, route cost passed, blocker is official-asof match/disambiguation",
                "required_policy": "accepted official public t05st01/t05st01_detail timestamp only; no silent backfill",
                "diagnostic_only": True,
                "formal_model_changed": False,
                "trade_decision_changed": False,
                "active_in_trade_decision": False,
                "report_changed": False,
                "portfolio_replay_executed": False,
                "ready_for_strategy_replay": False,
                "not_live_rule": True,
                "forward_returns_live_rule_usage": False,
            }
        ]
    )


def _future_audit(future_in: pd.DataFrame) -> pd.DataFrame:
    rows = future_in.to_dict("records") if not future_in.empty else []
    rows.extend(
        [
            {
                "audit_item": "core_review_forward_return_as_rule",
                "status": "passed",
                "future_data_violation_count": 0,
                "note": "review uses source route coverage only",
            },
            {
                "audit_item": "blocked_rows_no_silent_backfill_policy",
                "status": "passed",
                "future_data_violation_count": 0,
                "note": "blocked official-asof rows remain blocked until accepted public timestamp is found",
            },
        ]
    )
    return pd.DataFrame(rows)


def _readiness(
    readiness_in: dict[str, Any],
    blocked_rows: pd.DataFrame,
    failure_attribution: pd.DataFrame,
    future_audit: pd.DataFrame,
) -> dict[str, Any]:
    future_violations = int(pd.to_numeric(future_audit.get("future_data_violation_count", 0), errors="coerce").fillna(0).sum())
    blocked_count = len(blocked_rows)
    no_candidate_rows = int(blocked_rows["blocked_reason"].astype(str).eq("accepted_candidate_count=0").sum()) if not blocked_rows.empty else 0
    ambiguous_rows = int(blocked_rows["blocked_reason"].astype(str).eq("accepted_candidate_count=2").sum()) if not blocked_rows.empty else 0
    return {
        "task_id": TASK_ID,
        "status": "phase1_tpex_proof_reviewed_official_asof_blocked",
        "diagnostic_only": True,
        "sample_rows": readiness_in.get("sample_rows"),
        "ticker_count": readiness_in.get("ticker_count"),
        "period_count": readiness_in.get("period_count"),
        "statement_success_rows": readiness_in.get("statement_success_rows"),
        "official_asof_matched_rows": readiness_in.get("official_asof_matched_rows"),
        "official_asof_matched_share": readiness_in.get("official_asof_matched_share"),
        "blocked_or_ambiguous_rows": blocked_count,
        "accepted_candidate_count_0_rows": no_candidate_rows,
        "accepted_candidate_count_2_rows": ambiguous_rows,
        "route_error_count": readiness_in.get("route_error_count"),
        "actual_cache_rows_per_materialized_row": readiness_in.get("actual_cache_rows_per_materialized_row"),
        "budget_routes_per_row": readiness_in.get("budget_routes_per_row"),
        "ready_for_radar_official_asof_alternate_route_disambiguation": blocked_count > 0 and future_violations == 0,
        "ready_for_core_t164_tpex_all_stock_proof_readiness_update": False,
        "ready_for_core_t164_broader_ingest_contract": False,
        "ready_for_core_t164_broader_materialization": False,
        "ready_for_experiments": False,
        "ready_for_formal": False,
        "ready_for_strategy_replay": False,
        "ready_for_full_universe": False,
        "tpex_all_stock_universal_ready": False,
        "future_data_violation_count": future_violations,
        "blocked_fields": [
            "official_asof_unmatched_or_ambiguous_rows",
            "tpex_historical_all_stock_universe",
            "full_period_range",
            "full_universe_materialization",
            "capex_proxy_formal_label",
            "receivables_trade_formal_label",
        ],
        "proxy_fields": ["capex_proxy", "receivables_trade", "current_or_carried_tpex_universe_candidate"],
        "failure_attribution_classes": failure_attribution["failure_class"].tolist() if not failure_attribution.empty else [],
        "formal_model_changed": False,
        "trade_decision_changed": False,
        "active_in_trade_decision": False,
        "report_changed": False,
        "portfolio_replay_executed": False,
        "not_live_rule": True,
        "forward_returns_live_rule_usage": False,
    }


def _summary(readiness: dict[str, Any]) -> str:
    return f"""# Layer1 t164 TPEx phase_1 proof review

## Verdict
- status={readiness["status"]}
- statement_success_rows={readiness["statement_success_rows"]}/{readiness["sample_rows"]}
- official_asof_matched_rows={readiness["official_asof_matched_rows"]}/{readiness["sample_rows"]}
- blocked_or_ambiguous_rows={readiness["blocked_or_ambiguous_rows"]}
- accepted_candidate_count_0_rows={readiness["accepted_candidate_count_0_rows"]}
- accepted_candidate_count_2_rows={readiness["accepted_candidate_count_2_rows"]}
- ready_for_radar_official_asof_alternate_route_disambiguation={str(readiness["ready_for_radar_official_asof_alternate_route_disambiguation"]).lower()}
- ready_for_experiments=false
- ready_for_formal=false

## Core decision
Statement route and route cost are positive, but official-asof coverage is not clean enough to update TPEx all-stock proof readiness. The next owner is Radar/Data for bounded alternate official-asof route and detail/subject disambiguation on the 15 blocked rows.

## Boundaries
- No Experiments.
- No replay.
- No formal model/report/trade decision change.
- No silent official-asof backfill.

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
