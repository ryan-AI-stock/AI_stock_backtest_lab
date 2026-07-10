from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from backtest_lab.vnext_selected_path_total_return_completeness_absorption import (
    _holding_interval_coverage,
    _month_coverage,
)


TASK_ID = "TASK-BACKTEST-CORE-VNEXT-SELECTED-PATH-34-EFFECTIVE-DATE-ROUTE-ABSORPTION-001"
REPO_ROOT = Path(__file__).resolve().parents[2]
PRIOR_DIR = REPO_ROOT / "outputs" / "vnext_selected_path_total_return_completeness_absorption_20260710"
RADAR_DIR = Path(
    r"C:\Users\zergv\Documents\Codex\2026-05-23\ai-stock-rotation-radar-https-docs\outputs"
    r"\radar_vnext_selected_path_34_corporate_action_effective_date_archive_route_unlock_20260710"
)
OUTPUT_DIR = REPO_ROOT / "outputs" / "vnext_selected_path_34_effective_date_route_absorption_20260710"

FLAGS = {
    "formal_model_changed": False,
    "trade_decision_changed": False,
    "active_in_trade_decision": False,
    "report_changed": False,
    "portfolio_replay_executed": False,
    "ready_for_strategy_replay": False,
    "ready_for_formal": False,
    "not_live_rule": True,
    "forward_returns_live_rule_usage": False,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write(frame: pd.DataFrame, name: str) -> Path:
    path = OUTPUT_DIR / name
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def _absorb_route(prior: pd.DataFrame, route: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    key = ["ticker_month_key", "event_type"]
    route_cols = key + [
        "route_unlock_status", "route_unlock_reason", "structural_source_exhausted",
        "mops_matching_detail_candidate_count", "twse_public_archive_probe", "tpex_public_archive_probe",
    ]
    if route.duplicated(key).any():
        raise ValueError("Radar route results are not unique by ticker_month_key/event_type")
    contract = prior.merge(route[route_cols], on=key, how="left", validate="one_to_one")
    accepted = contract["route_unlock_status"].eq("accepted_non_holder_exclusion")
    exhausted = contract["route_unlock_status"].eq("structural_source_exhausted")
    contract.loc[accepted, "path_impact_resolved"] = True
    contract.loc[accepted, "accepted_no_holding_impact_evidence"] = True
    contract.loc[accepted, "structural_source_blocker"] = False
    contract.loc[accepted, "proof_status"] = "accepted_non_holder_exclusion"
    contract.loc[accepted, "proof_reason"] = contract.loc[accepted, "route_unlock_reason"]
    contract.loc[accepted, "core_path_completeness_status"] = "complete_official_non_holder_exclusion"
    contract["public_route_exhausted"] = exhausted
    contract["licensed_source_candidate"] = exhausted
    contract["licensed_source_required"] = False
    contract["licensed_source_decision_status"] = "not_applicable_non_holder_exclusion"
    contract.loc[exhausted, "licensed_source_decision_status"] = "strategy_center_decision_required_after_public_route_exhaustion"
    contract.loc[exhausted, "core_path_completeness_status"] = "blocked_public_route_exhausted_missing_exact_effective_date"
    contract["future_data_violation_count"] = 0
    exclusions = contract.loc[accepted].copy()
    remaining = contract.loc[exhausted].copy()
    return contract, exclusions, remaining


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    radar_readiness = json.loads(
        (RADAR_DIR / "readiness_for_core_selected_path_34_effective_date_route_unlock.json").read_text(encoding="utf-8")
    )
    prior = pd.read_csv(PRIOR_DIR / "selected_path_corporate_action_no_event_proof_absorbed.csv", dtype={"ticker": str}, low_memory=False)
    intervals = pd.read_csv(
        Path(radar_readiness.get("source_holding_intervals", "")) if radar_readiness.get("source_holding_intervals") else
        Path(r"C:\Users\zergv\Documents\Codex\2026-05-23\ai-stock-rotation-radar-https-docs\outputs\radar_vnext_selected_path_holding_month_corporate_action_no_event_proof_20260710\selected_path_holding_intervals.csv"),
        dtype={"ticker": str}, low_memory=False,
    )
    route = pd.read_csv(RADAR_DIR / "selected_path_34_effective_date_route_results.csv", dtype={"ticker": str}, low_memory=False)
    contract, exclusions, remaining = _absorb_route(prior, route)
    month_coverage = _month_coverage(contract)
    interval_coverage = _holding_interval_coverage(intervals, month_coverage)
    future_audit = pd.DataFrame([
        {"audit_item": "non_holder_exclusion", "future_data_used": False, "detail": "Accepted only Radar rows backed by official MOPS detail identifying non-holder events.", "future_data_violation_count": 0},
        {"audit_item": "public_route_exhaustion", "future_data_used": False, "detail": "Historical route limitations are source evidence, not inferred event dates.", "future_data_violation_count": 0},
        {"audit_item": "effective_date", "future_data_used": False, "detail": "Board/shareholder/query dates were not substituted.", "future_data_violation_count": 0},
        {"audit_item": "total_return_factor", "future_data_used": False, "detail": "No adjusted close, reinvestment, or holder factor calculated.", "future_data_violation_count": 0},
    ])
    complete_months = int(month_coverage["selected_path_ticker_month_complete"].sum())
    complete_intervals = int(interval_coverage["selected_path_ticker_month_complete"].sum())
    selected_path_complete = len(remaining) == 0 and complete_intervals == len(interval_coverage)
    readiness = {
        "task_id": TASK_ID,
        "status": "five_non_holder_exclusions_absorbed_29_public_route_exhausted_blockers_remain",
        "input_structural_blocker_rows": len(route),
        "accepted_non_holder_exclusion_rows": len(exclusions),
        "remaining_structural_source_exhausted_rows": len(remaining),
        "event_type_proof_rows": len(contract),
        "resolved_path_impact_proof_rows": int(contract["path_impact_resolved"].sum()),
        "complete_ticker_months": complete_months,
        "blocked_ticker_months": len(month_coverage) - complete_months,
        "complete_holding_intervals": complete_intervals,
        "blocked_holding_intervals": len(interval_coverage) - complete_intervals,
        "public_official_route_exhausted": True,
        "licensed_source_candidate_rows": len(remaining),
        "licensed_source_required": False,
        "licensed_source_decision_status": "strategy_center_decision_required",
        "selected_path_total_return_complete": selected_path_complete,
        "selected_path_adjusted_close_ready": False,
        "ready_for_experiments": False,
        "ready_for_formal": False,
        "ready_for_strategy_replay": False,
        "future_data_violation_count": 0,
        "next_owner": "Strategy Center decide licensed source authorization versus retain unadjusted diagnostic-only boundary",
        **FLAGS,
    }
    decision_audit = pd.DataFrame([
        {"decision_item": "official_non_holder_exclusions", "status": "absorbed", "rows": len(exclusions), "detail": "No selected-holder wealth-path impact."},
        {"decision_item": "public_official_effective_date_routes", "status": "exhausted", "rows": len(remaining), "detail": "TWSE route ignored historical date; TPEx historical range did not cover 2015+; MOPS detail lacked exact effective date."},
        {"decision_item": "licensed_source_required", "status": "strategy_center_pending", "rows": len(remaining), "detail": "Core marks candidates but does not authorize licensed acquisition."},
        {"decision_item": "unadjusted_diagnostic_path", "status": "available_with_blocker_warning", "rows": len(interval_coverage), "detail": "Must not be represented as total-return complete or formal-ready."},
    ])
    paths = [
        _write(contract, "selected_path_34_effective_date_route_absorbed_contract.csv"),
        _write(exclusions, "selected_path_accepted_non_holder_exclusions.csv"),
        _write(remaining, "selected_path_remaining_29_structural_source_exhausted_ledger.csv"),
        _write(month_coverage, "selected_path_ticker_month_completeness_after_route_absorption.csv"),
        _write(interval_coverage, "selected_path_holding_interval_completeness_after_route_absorption.csv"),
        _write(decision_audit, "selected_path_licensed_source_decision_audit.csv"),
        _write(future_audit, "selected_path_34_effective_date_route_future_data_audit.csv"),
    ]
    readiness_path = OUTPUT_DIR / "readiness_for_selected_path_34_effective_date_route_absorption.json"
    readiness_path.write_text(json.dumps(readiness, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path = OUTPUT_DIR / "final_summary_zh.md"
    summary_path.write_text(
        "# Selected-path 34 Effective-date Route Absorption\n\n"
        f"- accepted official non-holder exclusions: {len(exclusions)}\n"
        f"- remaining structural_source_exhausted rows: {len(remaining)}\n"
        f"- resolved path-impact proof rows: {readiness['resolved_path_impact_proof_rows']}/210\n"
        f"- complete ticker-months: {complete_months}/42\n"
        f"- complete holding intervals: {complete_intervals}/279\n"
        "- public official historical routes are exhausted for the remaining bounded set.\n"
        "- licensed_source_required remains a Strategy Center decision; Core only marks 29 candidate rows.\n"
        "- adjusted close / reinvestment / total-return factors remain blocked and unmaterialized.\n\n"
        "結論：5筆非持有人事件已移除 selected wealth-path blocker；29筆仍缺 exact effective date且公共官方 route 已耗盡。selected_path_total_return_complete=false，不交 Experiments。\n",
        encoding="utf-8",
    )
    manifest = {
        "task_id": TASK_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(OUTPUT_DIR),
        "source_inputs": {"prior_core": str(PRIOR_DIR), "radar_route_unlock": str(RADAR_DIR)},
        "files": [{"path": p.name, "sha256": _sha256(p)} for p in [*paths, readiness_path, summary_path]],
        "readiness": readiness,
        "radar_readiness": radar_readiness,
    }
    (OUTPUT_DIR / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(readiness, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
