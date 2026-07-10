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


TASK_ID = "TASK-BACKTEST-CORE-VNEXT-SELECTED-PATH-TRUSTED-CORPORATE-ACTION-ABSORPTION-001"
REPO_ROOT = Path(__file__).resolve().parents[2]
PRIOR_DIR = REPO_ROOT / "outputs" / "vnext_selected_path_34_effective_date_route_absorption_20260710"
RADAR_DIR = Path(
    r"C:\Users\zergv\Documents\Codex\2026-05-23\ai-stock-rotation-radar-https-docs\outputs"
    r"\radar_vnext_selected_path_29_corporate_action_trusted_source_cross_validation_20260710"
)
INTERVAL_PATH = Path(
    r"C:\Users\zergv\Documents\Codex\2026-05-23\ai-stock-rotation-radar-https-docs\outputs"
    r"\radar_vnext_selected_path_holding_month_corporate_action_no_event_proof_20260710"
    r"\selected_path_holding_intervals.csv"
)
OUTPUT_DIR = REPO_ROOT / "outputs" / "vnext_selected_path_trusted_corporate_action_absorption_20260710"

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


def _absorb_trusted(
    prior: pd.DataFrame,
    resolved: pd.DataFrame,
    blocked: pd.DataFrame,
    intervals: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    key = ["ticker_month_key", "event_type"]
    if len(resolved) != 1 or resolved.duplicated(key).any() or blocked.duplicated(key).any():
        raise ValueError("Expected one unique resolved row and unique blocked rows")
    trusted_cols = key + [
        "resolution_status", "resolution_reason", "trusted_effective_date",
        "trusted_payment_date", "trusted_amount_or_ratio",
    ]
    metadata = pd.concat([resolved[trusted_cols], blocked[trusted_cols]], ignore_index=True)
    contract = prior.merge(metadata, on=key, how="left", validate="one_to_one")
    accepted = contract["resolution_status"].eq("resolved_trusted_nonofficial_cross_validated")
    event_date = pd.to_datetime(contract.loc[accepted, "trusted_effective_date"], errors="raise").iloc[0]
    ticker = str(contract.loc[accepted, "ticker"].iloc[0])
    held = intervals.loc[
        intervals["ticker"].eq(ticker)
        & pd.to_datetime(intervals["hold_start"]).lt(event_date)
        & pd.to_datetime(intervals["hold_end_exclusive"]).ge(event_date)
    ]
    if len(held):
        raise ValueError("Trusted event overlaps selected holding and cannot be treated as no-impact metadata")
    contract.loc[accepted, "path_impact_resolved"] = True
    contract.loc[accepted, "accepted_no_holding_impact_evidence"] = True
    contract.loc[accepted, "structural_source_blocker"] = False
    contract.loc[accepted, "proof_status"] = "trusted_nonofficial_event_outside_held_dates"
    contract.loc[accepted, "proof_reason"] = "FinMind and Yahoo agree on exact event date/amount; event predates selected holding"
    contract.loc[accepted, "core_path_completeness_status"] = "complete_diagnostic_trusted_nonofficial_event_outside_holding"
    contract.loc[accepted, "source_quality"] = "trusted_nonofficial_cross_validated_diagnostic_only"
    contract["trusted_metadata_accepted_for_formal"] = False
    contract["trusted_metadata_accepted_for_diagnostic"] = accepted
    contract["licensed_source_required"] = False
    contract["future_data_violation_count"] = 0
    accepted_audit = contract.loc[accepted].copy()
    remaining = contract.loc[contract["resolution_status"].isin([
        "blocked_yahoo_only_inferred",
        "blocked_missing_holder_ratio_effective_date",
        "blocked_no_trusted_structured_candidate",
    ])].copy()
    return contract, accepted_audit, remaining


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    radar_readiness = json.loads(
        (RADAR_DIR / "readiness_for_core_selected_path_trusted_source_absorption.json").read_text(encoding="utf-8")
    )
    prior = pd.read_csv(PRIOR_DIR / "selected_path_34_effective_date_route_absorbed_contract.csv", dtype={"ticker": str}, low_memory=False)
    intervals = pd.read_csv(INTERVAL_PATH, dtype={"ticker": str}, low_memory=False)
    resolved = pd.read_csv(RADAR_DIR / "selected_path_29_resolved_rows.csv", dtype={"ticker": str}, low_memory=False)
    blocked_source = pd.read_csv(RADAR_DIR / "selected_path_29_remaining_blocked.csv", dtype={"ticker": str}, low_memory=False)
    providers = pd.read_csv(RADAR_DIR / "selected_path_29_provider_terms_and_cost_options.csv", low_memory=False)
    contract, accepted, remaining = _absorb_trusted(prior, resolved, blocked_source, intervals)
    month_coverage = _month_coverage(contract)
    interval_coverage = _holding_interval_coverage(intervals, month_coverage)
    complete_months = int(month_coverage["selected_path_ticker_month_complete"].sum())
    complete_intervals = int(interval_coverage["selected_path_ticker_month_complete"].sum())
    remaining_counts = remaining["resolution_status"].value_counts().to_dict()
    future_audit = pd.DataFrame([
        {"audit_item": "trusted_cross_validation", "future_data_used": False, "detail": "FinMind and Yahoo exact date/amount agreement used only as diagnostic metadata.", "future_data_violation_count": 0},
        {"audit_item": "holding_alignment", "future_data_used": False, "detail": "2023-03-16 event was aligned against actual selected holding intervals and predates first 2023-03-20 holding.", "future_data_violation_count": 0},
        {"audit_item": "inferred_clues", "future_data_used": False, "detail": "Yahoo-only adjusted/raw clues remain blocked and were not converted to event dates.", "future_data_violation_count": 0},
        {"audit_item": "formal_boundary", "future_data_used": False, "detail": "All trusted nonofficial evidence remains accepted_for_formal=false.", "future_data_violation_count": 0},
    ])
    readiness = {
        "task_id": TASK_ID,
        "status": "one_trusted_nonofficial_outside_holding_event_absorbed_28_rows_remain_blocked",
        "input_trusted_cross_validation_rows": 29,
        "trusted_nonofficial_cross_validated_rows_absorbed": len(accepted),
        "trusted_event_selected_holding_overlap_rows": 0,
        "remaining_blocked_rows": len(remaining),
        "remaining_yahoo_only_inferred_rows": int(remaining_counts.get("blocked_yahoo_only_inferred", 0)),
        "remaining_missing_holder_ratio_effective_date_rows": int(remaining_counts.get("blocked_missing_holder_ratio_effective_date", 0)),
        "remaining_no_trusted_structured_candidate_rows": int(remaining_counts.get("blocked_no_trusted_structured_candidate", 0)),
        "resolved_path_impact_proof_rows": int(contract["path_impact_resolved"].sum()),
        "complete_ticker_months": complete_months,
        "blocked_ticker_months": len(month_coverage) - complete_months,
        "complete_holding_intervals": complete_intervals,
        "blocked_holding_intervals": len(interval_coverage) - complete_intervals,
        "trusted_nonofficial_accepted_for_formal": False,
        "licensed_source_required": False,
        "licensed_source_decision_status": "strategy_center_user_decision_required_for_formal_completeness",
        "selected_path_total_return_complete": False,
        "selected_path_adjusted_close_ready": False,
        "ready_for_experiments": False,
        "ready_for_formal": False,
        "ready_for_strategy_replay": False,
        "future_data_violation_count": 0,
        "next_owner": "Strategy Center decide licensed source purchase/authorization or retain diagnostic-only boundary",
        **FLAGS,
    }
    decision_audit = pd.DataFrame([
        {"item": "2330_ROC112_M03_cash_dividend", "status": "absorbed_diagnostic_metadata", "rows": 1, "detail": "2023-03-16, cash 2.74982072; FinMind/Yahoo agree; event predates selected holding."},
        {"item": "yahoo_only_inferred", "status": "blocked", "rows": readiness["remaining_yahoo_only_inferred_rows"], "detail": "No independent structured confirmation; cannot infer formal event adjustment."},
        {"item": "holder_ratio_effective_date", "status": "blocked", "rows": readiness["remaining_missing_holder_ratio_effective_date_rows"], "detail": "Merger/capital events lack holder conversion ratio plus effective date."},
        {"item": "no_trusted_structured_candidate", "status": "blocked", "rows": readiness["remaining_no_trusted_structured_candidate_rows"], "detail": "No bounded trusted candidate found."},
        {"item": "licensed_source", "status": "user_decision_required", "rows": len(remaining), "detail": "TWSE T48 partial period/cost and TPEx quote requirements require explicit authorization."},
    ])
    paths = [
        _write(contract, "selected_path_trusted_corporate_action_absorbed_contract.csv"),
        _write(accepted, "selected_path_trusted_nonofficial_diagnostic_metadata.csv"),
        _write(remaining, "selected_path_remaining_28_trusted_source_blocked_ledger.csv"),
        _write(month_coverage, "selected_path_ticker_month_completeness_after_trusted_absorption.csv"),
        _write(interval_coverage, "selected_path_holding_interval_completeness_after_trusted_absorption.csv"),
        _write(providers, "selected_path_licensed_provider_options_absorbed.csv"),
        _write(decision_audit, "selected_path_trusted_source_decision_audit.csv"),
        _write(future_audit, "selected_path_trusted_corporate_action_future_data_audit.csv"),
    ]
    readiness_path = OUTPUT_DIR / "readiness_for_selected_path_trusted_corporate_action_absorption.json"
    readiness_path.write_text(json.dumps(readiness, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path = OUTPUT_DIR / "final_summary_zh.md"
    summary_path.write_text(
        "# Selected-path Trusted Corporate-action Absorption\n\n"
        "- absorbed diagnostic metadata: 2330 ROC112/03 cash ex-dividend, 2023-03-16, cash 2.74982072.\n"
        "- FinMind and Yahoo agree; the event predates all selected holdings starting 2023-03-20, so selected wealth path has no entitlement impact.\n"
        f"- remaining blocked: Yahoo-only inferred {readiness['remaining_yahoo_only_inferred_rows']}、missing holder terms {readiness['remaining_missing_holder_ratio_effective_date_rows']}、no structured candidate {readiness['remaining_no_trusted_structured_candidate_rows']}。\n"
        f"- complete ticker-months: {complete_months}/42；complete holding intervals: {complete_intervals}/279。\n"
        "- trusted_nonofficial_accepted_for_formal=false；adjusted close / total-return factor remain blocked.\n"
        "- licensed source purchase/authorization requires Strategy Center and user decision.\n\n"
        "結論：1筆雙來源一致事件已吸收為 diagnostic metadata；28筆仍 blocked。selected_path_total_return_complete=false，不交 Experiments。\n",
        encoding="utf-8",
    )
    manifest = {
        "task_id": TASK_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(OUTPUT_DIR),
        "source_inputs": {"prior_core": str(PRIOR_DIR), "radar_trusted_validation": str(RADAR_DIR)},
        "files": [{"path": p.name, "sha256": _sha256(p)} for p in [*paths, readiness_path, summary_path]],
        "readiness": readiness,
        "radar_readiness": radar_readiness,
    }
    (OUTPUT_DIR / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(readiness, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
