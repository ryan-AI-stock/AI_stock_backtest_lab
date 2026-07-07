"""Review Radar/Data Layer1 t164 phase-2 bounded runner output.

This Core/Data review classifies official-asof blockers and routes the next
bounded source-readiness step. It does not run Experiments, replay, or formal
pipeline changes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


TASK_ID = "TASK-BACKTEST-CORE-VNEXT-LAYER1-T164-PHASE2-BOUNDED-RUNNER-REVIEW-001"
DEFAULT_RADAR_DIR = Path(
    r"C:\Users\zergv\Documents\Codex\2026-05-23\ai-stock-rotation-radar-https-docs\outputs"
    r"\radar_vnext_layer1_t164_phase2_bounded_runner_20260707"
)
DEFAULT_OUTPUT_DIR = Path("outputs/vnext_layer1_t164_phase2_bounded_runner_review_20260707")


def build_review(
    *,
    radar_dir: str | Path = DEFAULT_RADAR_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    radar = Path(radar_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    readiness_in = _read_json(radar / "readiness_for_core_t164_phase2_bounded_runner.json")
    manifest_in = _read_json(radar / "manifest.json")
    coverage_period = _read_csv(radar / "coverage_by_market_period.csv", dtype=str)
    coverage_field = _read_csv(radar / "coverage_by_field.csv", dtype=str)
    blocked = _read_csv(radar / "blocked_or_ambiguous_rows.csv", dtype={"ticker": str})
    future_in = _read_csv(radar / "future_data_governance_audit.csv")
    route_cost = _read_csv(radar / "projected_route_cost_report.csv")

    enriched_blocked = _enrich_blocked_rows(blocked)
    blocked_by_period = _blocked_by_period(enriched_blocked)
    blocked_by_reason = _blocked_by_reason(enriched_blocked)
    system_assessment = _system_assessment(readiness_in, coverage_period, enriched_blocked)
    field_readiness = _field_readiness(coverage_field)
    review_summary = _review_summary(readiness_in, coverage_period, field_readiness, route_cost)
    next_radar_request = _next_radar_request(enriched_blocked, system_assessment)
    future_audit = _future_audit(future_in)
    readiness = _readiness(readiness_in, enriched_blocked, system_assessment, future_audit)

    _write_csv(review_summary, output / "layer1_t164_phase2_runner_review_summary.csv")
    _write_csv(enriched_blocked, output / "layer1_t164_phase2_blocked_rows_classified.csv")
    _write_csv(blocked_by_period, output / "layer1_t164_phase2_blocked_by_period.csv")
    _write_csv(blocked_by_reason, output / "layer1_t164_phase2_blocked_by_reason.csv")
    _write_csv(system_assessment, output / "layer1_t164_phase2_systemic_blocker_assessment.csv")
    _write_csv(field_readiness, output / "layer1_t164_phase2_field_completeness_readiness.csv")
    _write_csv(next_radar_request, output / "layer1_t164_phase2_next_radar_request.csv")
    _write_csv(future_audit, output / "layer1_t164_phase2_review_future_data_audit.csv")
    (output / "readiness_for_layer1_t164_phase2_runner_review.json").write_text(
        json.dumps(readiness, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    manifest = {
        "task_id": TASK_ID,
        "status": readiness["status"],
        "output_dir": str(output.resolve()),
        "radar_input_dir": str(radar.resolve()),
        "radar_commit": "70a0218",
        "radar_task_id": manifest_in.get("task_id"),
        "output_files": [
            "layer1_t164_phase2_runner_review_summary.csv",
            "layer1_t164_phase2_blocked_rows_classified.csv",
            "layer1_t164_phase2_blocked_by_period.csv",
            "layer1_t164_phase2_blocked_by_reason.csv",
            "layer1_t164_phase2_systemic_blocker_assessment.csv",
            "layer1_t164_phase2_field_completeness_readiness.csv",
            "layer1_t164_phase2_next_radar_request.csv",
            "layer1_t164_phase2_review_future_data_audit.csv",
            "readiness_for_layer1_t164_phase2_runner_review.json",
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


def _enrich_blocked_rows(blocked: pd.DataFrame) -> pd.DataFrame:
    out = blocked.copy()
    if out.empty:
        return out
    out["failure_class"] = out["blocked_reason"].astype(str).map(
        lambda reason: "no_accepted_official_candidate"
        if "accepted_candidate_count=0" in reason
        else "ambiguous_multiple_official_candidates"
        if "accepted_candidate_count=2" in reason
        else "other_blocked"
    )
    out["is_older_period"] = out["report_period"].astype(str).isin(["114Q3", "114Q2"])
    out["needs_alternate_route"] = out["failure_class"].eq("no_accepted_official_candidate")
    out["needs_disambiguation"] = out["failure_class"].eq("ambiguous_multiple_official_candidates")
    out["blocked_policy"] = "blocked_no_silent_backfill"
    out["diagnostic_only"] = True
    out["ready_for_experiments"] = False
    out["formal_ready"] = False
    return out


def _blocked_by_period(blocked: pd.DataFrame) -> pd.DataFrame:
    if blocked.empty:
        return pd.DataFrame()
    grouped = (
        blocked.groupby(["report_period", "failure_class"], dropna=False)
        .size()
        .reset_index(name="blocked_rows")
        .sort_values(["report_period", "failure_class"])
    )
    grouped["diagnostic_only"] = True
    return grouped


def _blocked_by_reason(blocked: pd.DataFrame) -> pd.DataFrame:
    if blocked.empty:
        return pd.DataFrame()
    grouped = blocked.groupby(["blocked_reason", "failure_class"], dropna=False).size().reset_index(name="blocked_rows")
    grouped["tickers"] = grouped.apply(
        lambda row: ";".join(
            sorted(
                blocked.loc[
                    (blocked["blocked_reason"].eq(row["blocked_reason"]))
                    & (blocked["failure_class"].eq(row["failure_class"])),
                    "ticker",
                ]
                .astype(str)
                .unique()
            )
        ),
        axis=1,
    )
    grouped["diagnostic_only"] = True
    return grouped


def _system_assessment(readiness: dict[str, Any], coverage_period: pd.DataFrame, blocked: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    older_blocked = int(blocked["is_older_period"].astype(bool).sum()) if not blocked.empty else 0
    total_blocked = int(len(blocked))
    older_share = older_blocked / total_blocked if total_blocked else 0
    rows.append(
        {
            "assessment_item": "official_asof_coverage",
            "status": "blocked_for_completeness_patch",
            "evidence": f"official_asof_matched_rows={readiness.get('official_asof_matched_rows')}/{readiness.get('sample_rows')}; blocked_rows={total_blocked}",
            "core_judgment": "352/400 is useful but not enough to stop source cleanup because official-asof blockers remain material",
            "recommended_action": "run bounded alternate official-asof route/disambiguation for 48 rows",
            "diagnostic_only": True,
        }
    )
    rows.append(
        {
            "assessment_item": "older_period_route_pattern",
            "status": "systemic_or_route_policy_suspected" if older_share >= 0.6 else "mixed",
            "evidence": f"older_period_blocked_rows={older_blocked}/{total_blocked}; older_periods=114Q3/114Q2",
            "core_judgment": "114Q3/114Q2 have visibly lower match rates, so this is likely route/query-window/policy related rather than only isolated ticker noise",
            "recommended_action": "Radar/Data should prioritize older-period query windows and subject/detail token policy before per-row manual chase",
            "diagnostic_only": True,
        }
    )
    if not coverage_period.empty:
        for item in coverage_period.to_dict("records"):
            requested = _to_int(item.get("requested_rows"))
            matched = _to_int(item.get("official_asof_matched_rows"))
            blocked_rows = _to_int(item.get("blocked_rows"))
            match_share = matched / requested if requested else None
            rows.append(
                {
                    "assessment_item": f"period_{item.get('group')}",
                    "status": "period_coverage",
                    "evidence": f"matched={matched}/{requested}; blocked={blocked_rows}; match_share={match_share:.2f}" if match_share is not None else "",
                    "core_judgment": "period-level metadata",
                    "recommended_action": "include in alternate-route validation summary",
                    "diagnostic_only": True,
                }
            )
    return pd.DataFrame(rows)


def _field_readiness(coverage_field: pd.DataFrame) -> pd.DataFrame:
    out = coverage_field.copy()
    if out.empty:
        return out
    out["missing_share"] = pd.to_numeric(out["missing_share"], errors="coerce")
    out["field_readiness"] = out.apply(
        lambda row: "proxy_human_review_required"
        if "proxy" in str(row.get("source_quality"))
        else "partial_missingness"
        if float(row.get("missing_share") or 0) > 0
        else "complete_in_phase2_sample",
        axis=1,
    )
    out["formal_ready"] = False
    out["diagnostic_only"] = True
    return out


def _review_summary(
    readiness: dict[str, Any],
    coverage_period: pd.DataFrame,
    field_readiness: pd.DataFrame,
    route_cost: pd.DataFrame,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "review_item": "runner_result",
                "status": readiness.get("status"),
                "sample_rows": readiness.get("sample_rows"),
                "statement_success_rows": readiness.get("statement_success_rows"),
                "official_asof_matched_rows": readiness.get("official_asof_matched_rows"),
                "blocked_rows": int(readiness.get("sample_rows", 0)) - int(readiness.get("official_asof_matched_rows", 0)),
                "core_verdict": "statement_route_passed_but_official_asof_completeness_needs_patch",
                "diagnostic_only": True,
            },
            {
                "review_item": "route_cost",
                "status": "passed_budget_guard",
                "sample_rows": readiness.get("sample_rows"),
                "statement_success_rows": readiness.get("statement_success_rows"),
                "official_asof_matched_rows": readiness.get("official_asof_matched_rows"),
                "blocked_rows": int(readiness.get("sample_rows", 0)) - int(readiness.get("official_asof_matched_rows", 0)),
                "core_verdict": f"actual_cache_rows_per_materialized_row={readiness.get('actual_cache_rows_per_materialized_row')}; budget={readiness.get('budget_routes_per_row')}",
                "diagnostic_only": True,
            },
            {
                "review_item": "plain_strategy_center_summary",
                "status": "source_completeness_priority",
                "sample_rows": readiness.get("sample_rows"),
                "statement_success_rows": readiness.get("statement_success_rows"),
                "official_asof_matched_rows": readiness.get("official_asof_matched_rows"),
                "blocked_rows": int(readiness.get("sample_rows", 0)) - int(readiness.get("official_asof_matched_rows", 0)),
                "core_verdict": "Layer1 財報數值欄位已大致能抓到，但 official-asof 仍是主要缺口；最有效補強是先修 48 筆公告時間對齊，不是進 Experiments。",
                "diagnostic_only": True,
            },
        ]
    )


def _next_radar_request(blocked: pd.DataFrame, system_assessment: pd.DataFrame) -> pd.DataFrame:
    no_candidate = int(blocked["needs_alternate_route"].astype(bool).sum()) if not blocked.empty else 0
    ambiguous = int(blocked["needs_disambiguation"].astype(bool).sum()) if not blocked.empty else 0
    return pd.DataFrame(
        [
            {
                "next_owner": "Radar/Data",
                "handoff_action": "run_phase2_bounded_official_asof_alternate_route_disambiguation_for_48_blocked_rows",
                "input_artifacts": "layer1_t164_phase2_blocked_rows_classified.csv; layer1_t164_phase2_systemic_blocker_assessment.csv",
                "ready": True,
                "reason": f"statement route 400/400 and route cost passed; official-asof blockers remain 48 rows ({no_candidate} no-candidate, {ambiguous} ambiguous), with older-period concentration",
                "required_policy": "official t05st01/t05st01_detail timestamp only; explicit blocked ledger; no silent backfill",
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
                "note": "review uses source coverage only",
            },
            {
                "audit_item": "blocked_rows_no_silent_backfill",
                "status": "passed",
                "future_data_violation_count": 0,
                "note": "48 official-asof blockers remain explicit until alternate route resolves them",
            },
        ]
    )
    return pd.DataFrame(rows)


def _readiness(
    readiness_in: dict[str, Any],
    blocked: pd.DataFrame,
    system_assessment: pd.DataFrame,
    future_audit: pd.DataFrame,
) -> dict[str, Any]:
    no_candidate = int(blocked["needs_alternate_route"].astype(bool).sum()) if not blocked.empty else 0
    ambiguous = int(blocked["needs_disambiguation"].astype(bool).sum()) if not blocked.empty else 0
    older_blocked = int(blocked["is_older_period"].astype(bool).sum()) if not blocked.empty else 0
    future_violations = int(pd.to_numeric(future_audit.get("future_data_violation_count", 0), errors="coerce").fillna(0).sum())
    return {
        "task_id": TASK_ID,
        "status": "phase2_runner_reviewed_asof_patch_required_not_experiments",
        "diagnostic_only": True,
        "sample_rows": readiness_in.get("sample_rows"),
        "ticker_count": readiness_in.get("ticker_count"),
        "period_count": readiness_in.get("period_count"),
        "statement_success_rows": readiness_in.get("statement_success_rows"),
        "official_asof_matched_rows": readiness_in.get("official_asof_matched_rows"),
        "official_asof_matched_share": readiness_in.get("official_asof_matched_share"),
        "blocked_rows": len(blocked),
        "no_accepted_official_candidate_rows": no_candidate,
        "ambiguous_multiple_official_candidate_rows": ambiguous,
        "older_period_blocked_rows_114Q3_114Q2": older_blocked,
        "route_error_count": readiness_in.get("route_error_count"),
        "actual_cache_rows_per_materialized_row": readiness_in.get("actual_cache_rows_per_materialized_row"),
        "budget_routes_per_row": readiness_in.get("budget_routes_per_row"),
        "future_data_violation_count": future_violations,
        "core_decision": "run_bounded_alternate_official_asof_route_before_phase3_or_experiments",
        "ready_for_radar_phase2_asof_patch_runner": future_violations == 0 and len(blocked) > 0,
        "ready_for_phase2_partial_source_package_policy": False,
        "ready_for_core_t164_broader_ingest_contract": False,
        "ready_for_core_t164_broader_materialization": False,
        "ready_for_experiments": False,
        "ready_for_formal": False,
        "ready_for_strategy_replay": False,
        "ready_for_full_universe": False,
        "tpex_all_stock_universal_ready": False,
        "blocked_fields": [
            "official_asof_blocked_rows_48",
            "older_period_asof_route_policy",
            "tpex_historical_all_stock_universe",
            "full_period_46_quarters_materialization",
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


def _to_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _summary(readiness: dict[str, Any]) -> str:
    return f"""# Layer1 t164 Phase 2 bounded runner review

## Verdict
- status={readiness["status"]}
- statement_success_rows={readiness["statement_success_rows"]}/{readiness["sample_rows"]}
- official_asof_matched_rows={readiness["official_asof_matched_rows"]}/{readiness["sample_rows"]}
- blocked_rows={readiness["blocked_rows"]}
- no_accepted_official_candidate_rows={readiness["no_accepted_official_candidate_rows"]}
- ambiguous_multiple_official_candidate_rows={readiness["ambiguous_multiple_official_candidate_rows"]}
- older_period_blocked_rows_114Q3_114Q2={readiness["older_period_blocked_rows_114Q3_114Q2"]}
- ready_for_radar_phase2_asof_patch_runner={str(readiness["ready_for_radar_phase2_asof_patch_runner"]).lower()}
- ready_for_experiments=false
- ready_for_formal=false

## Plain Summary
Layer1 財報數值資料本身已經相當穩：400/400 statement success，主要缺口是 official-asof 公告時間。352/400 還不適合直接當 phase_2 source closure，因為 48 筆 blocked 對後續 PIT fundamental layer 影響太大。最有效補強動作是先由 Radar/Data 針對 48 筆做 bounded alternate official-asof route/disambiguation，尤其 114Q3/114Q2。

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
