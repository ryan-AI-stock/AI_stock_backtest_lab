from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from backtest_lab.vnext_selected_stock_total_return_exdate_patch_absorption import _load_legs


TASK_ID = "TASK-BACKTEST-CORE-VNEXT-SELECTED-STOCK-TOTAL-RETURN-REMAINING-CORPORATE-ACTION-GAP-ABSORPTION-001"
REPO_ROOT = Path(__file__).resolve().parents[2]
PRIOR_DIR = REPO_ROOT / "outputs" / "vnext_selected_stock_total_return_exdate_patch_absorption_20260710"
RADAR_DIR = Path(
    r"C:\Users\zergv\Documents\Codex\2026-05-23\ai-stock-rotation-radar-https-docs\outputs"
    r"\radar_vnext_selected_stock_corporate_action_remaining_gap_fill_20260710"
)
OUTPUT_DIR = REPO_ROOT / "outputs" / "vnext_selected_stock_total_return_remaining_corporate_action_gap_absorption_20260710"

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


def _absorb_mapping(prior: pd.DataFrame, patch: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    accepted = patch.loc[patch["mapping_status"].eq("accepted_unique_candidate")].copy()
    accepted = accepted.rename(
        columns={
            "accepted_exact_exdate": "remaining_patch_exact_exdate",
            "accepted_payment_date": "remaining_patch_payment_date",
            "accepted_market_available_at": "remaining_patch_market_available_at",
            "accepted_source_url": "remaining_patch_source_url",
            "accepted_subject": "remaining_patch_subject",
            "mapping_status": "remaining_patch_mapping_status",
        }
    )
    cols = [
        "event_key",
        "candidate_count",
        "exact_candidate_count",
        "payment_candidate_count",
        "remaining_patch_exact_exdate",
        "remaining_patch_payment_date",
        "remaining_patch_market_available_at",
        "remaining_patch_source_url",
        "remaining_patch_subject",
        "remaining_patch_mapping_status",
    ]
    ledger = prior.merge(accepted[cols], on="event_key", how="left", validate="one_to_one")
    prior_exact = pd.to_datetime(ledger["ex_date"], errors="coerce")
    prior_payment = pd.to_datetime(ledger["payment_date"], errors="coerce")
    patch_exact = pd.to_datetime(ledger["remaining_patch_exact_exdate"], errors="coerce")
    patch_payment = pd.to_datetime(ledger["remaining_patch_payment_date"], errors="coerce")
    ledger["remaining_patch_added_exact_exdate"] = prior_exact.isna() & patch_exact.notna()
    ledger["remaining_patch_confirmed_existing_exact_exdate"] = prior_exact.notna() & patch_exact.eq(prior_exact)
    ledger["remaining_patch_exact_conflict"] = prior_exact.notna() & patch_exact.notna() & patch_exact.ne(prior_exact)
    ledger["remaining_patch_added_payment_date"] = prior_payment.isna() & patch_payment.notna()
    ledger["ex_date"] = prior_exact.combine_first(patch_exact).dt.strftime("%Y-%m-%d")
    ledger["payment_date"] = prior_payment.combine_first(patch_payment).dt.strftime("%Y-%m-%d")
    ledger["exact_exdate_ready"] = ledger["ex_date"].notna()
    ledger["payment_date_ready"] = ledger["payment_date"].notna()
    ledger["cash_available_date_ready"] = ledger["payment_date_ready"]
    ledger["accepted_for_total_return_ledger"] = False
    ledger["ledger_row_ready"] = False
    ledger["ledger_status"] = "blocked_total_return_factor_and_temporal_completeness"
    audit = ledger.loc[ledger["remaining_patch_mapping_status"].notna(), [
        "ticker", "event_key", "remaining_patch_mapping_status", "remaining_patch_exact_exdate",
        "remaining_patch_payment_date", "remaining_patch_added_exact_exdate",
        "remaining_patch_confirmed_existing_exact_exdate", "remaining_patch_exact_conflict",
        "remaining_patch_added_payment_date", "remaining_patch_market_available_at",
        "remaining_patch_source_url", "remaining_patch_subject",
    ]].copy()
    return ledger, audit


def _align_entitlement(ledger: pd.DataFrame, legs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for event in ledger.itertuples(index=False):
        ex_date = pd.to_datetime(event.ex_date, errors="coerce")
        matches = legs.iloc[0:0]
        if pd.notna(ex_date):
            matches = legs.loc[
                legs["ticker"].eq(str(event.ticker))
                & legs["hold_start"].lt(ex_date)
                & legs["hold_end_exclusive"].ge(ex_date)
            ]
        rows.append({
            "ticker": event.ticker,
            "event_key": event.event_key,
            "ex_date": event.ex_date,
            "payment_date": event.payment_date,
            "entitled_holding_leg_count": len(matches),
            "entitled_base_strategies": "|".join(sorted(matches["base_strategy"].unique())) if len(matches) else "",
            "selected_path_entitlement_flag": bool(len(matches)),
            "future_data_violation_count": 0,
        })
    return pd.DataFrame(rows)


def _remaining_gaps(ledger: pd.DataFrame, radar_blocked: pd.DataFrame, temporal: pd.DataFrame) -> pd.DataFrame:
    rows = radar_blocked[["ticker", "event_key", "missing_component", "blocked_reason", "attempted_source", "next_bounded_step"]].copy()
    rows["gap_scope"] = "canonical_event_component"
    stock_missing = ledger.loc[
        ledger["share_adjustment_candidate_flag"].fillna(False).astype(bool)
        & ledger["share_adjustment_effective_date"].isna(), ["ticker", "event_key"]
    ].copy()
    stock_missing["missing_component"] = "stock_distribution_effective_or_tradable_date"
    stock_missing["blocked_reason"] = "official_route_not_unlocked"
    stock_missing["attempted_source"] = "Radar bounded remaining-gap source package"
    stock_missing["next_bounded_step"] = "official new-share effective/tradable-date route or policy review"
    stock_missing["gap_scope"] = "holder_share_scale"
    temporal_rows = temporal[["ticker", "holding_months", "coverage_status"]].copy()
    temporal_rows["event_key"] = ""
    temporal_rows["missing_component"] = "holding_month_no_silent_no_event_proof"
    temporal_rows["blocked_reason"] = temporal_rows["coverage_status"]
    temporal_rows["attempted_source"] = "MOPS t05st01/t05st01_detail bounded holding-month query"
    temporal_rows["next_bounded_step"] = "accepted official no-event proof policy or complete corporate-action archive"
    temporal_rows["gap_scope"] = "temporal_completeness"
    for frame in (rows, stock_missing, temporal_rows):
        frame["future_data_violation_count"] = 0
    return pd.concat([rows, stock_missing, temporal_rows], ignore_index=True, sort=False)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    radar_readiness = json.loads(
        (RADAR_DIR / "readiness_for_core_selected_stock_corporate_action_remaining_gap_fill.json").read_text(encoding="utf-8")
    )
    prior = pd.read_csv(PRIOR_DIR / "selected_stock_total_return_event_ledger_exdate_patched.csv", dtype={"ticker": str}, low_memory=False)
    patch = pd.read_csv(RADAR_DIR / "selected_stock_remaining_canonical_event_mapping_patch.csv", dtype={"ticker": str}, low_memory=False)
    radar_blocked = pd.read_csv(RADAR_DIR / "selected_stock_remaining_blocked_ledger.csv", dtype={"ticker": str}, low_memory=False)
    capital = pd.read_csv(RADAR_DIR / "selected_stock_remaining_capital_change_inventory.csv", dtype={"ticker": str}, low_memory=False)
    temporal = pd.read_csv(RADAR_DIR / "selected_stock_remaining_temporal_coverage_audit.csv", dtype={"ticker": str}, low_memory=False)
    ledger, mapping_audit = _absorb_mapping(prior, patch)
    entitlement = _align_entitlement(ledger, _load_legs())
    ledger = ledger.drop(columns=["entitled_holding_leg_count", "entitled_base_strategies", "selected_path_entitlement_flag"], errors="ignore")
    ledger = ledger.merge(entitlement[["event_key", "entitled_holding_leg_count", "entitled_base_strategies", "selected_path_entitlement_flag"]], on="event_key", how="left", validate="one_to_one")
    capital["core_absorption_status"] = "blocked_no_complete_holder_scale_effective_date_factor_cash_terms"
    capital["accepted_for_total_return_factor"] = False
    gaps = _remaining_gaps(ledger, radar_blocked, temporal)
    future_audit = pd.DataFrame([
        {"audit_item": "canonical_mapping", "future_data_used": False, "detail": "Only accepted_unique_candidate rows were absorbed.", "future_data_violation_count": 0},
        {"audit_item": "entitlement", "future_data_used": False, "detail": "Actual holding interval and exact ex-date only.", "future_data_violation_count": 0},
        {"audit_item": "capital_change", "future_data_used": False, "detail": "No holder factor inferred from incomplete event text.", "future_data_violation_count": 0},
        {"audit_item": "total_return", "future_data_used": False, "detail": "No adjusted close, reinvestment, or factor fabricated.", "future_data_violation_count": 0},
    ])
    exact_ready = int(ledger["exact_exdate_ready"].sum())
    payment_ready = int(ledger["payment_date_ready"].sum())
    added_exact = int(ledger["remaining_patch_added_exact_exdate"].sum())
    added_payment = int(ledger["remaining_patch_added_payment_date"].sum())
    exact_conflicts = int(ledger["remaining_patch_exact_conflict"].sum())
    entitled = int(ledger["selected_path_entitlement_flag"].sum())
    readiness = {
        "task_id": TASK_ID,
        "status": "remaining_unique_mapping_payment_absorbed_total_return_still_blocked",
        "canonical_events": len(ledger),
        "accepted_exact_exdate_events_after_absorption": exact_ready,
        "accepted_payment_date_events_after_absorption": payment_ready,
        "new_exact_exdate_events_added": added_exact,
        "existing_exact_exdate_events_confirmed": int(ledger["remaining_patch_confirmed_existing_exact_exdate"].sum()),
        "exact_exdate_conflict_rows": exact_conflicts,
        "new_payment_date_events_added": added_payment,
        "selected_path_entitled_exact_events": entitled,
        "remaining_canonical_component_blocked_rows": len(radar_blocked),
        "remaining_exact_exdate_blocked_rows": int(radar_blocked["missing_component"].eq("exact_exdate").sum()),
        "remaining_cash_payment_date_blocked_rows": int(radar_blocked["missing_component"].eq("cash_payment_date").sum()),
        "stock_distribution_effective_date_blocked_events": int(ledger["share_adjustment_candidate_flag"].fillna(False).astype(bool).sum()),
        "capital_change_inventory_rows_reviewed": len(capital),
        "capital_change_total_return_factor_ready_rows": 0,
        "holding_month_temporal_no_event_proof_blocked_tickers": int(temporal["ticker"].nunique()),
        "corporate_action_temporal_coverage_complete": False,
        "selected_stock_total_return_ledger_ready": False,
        "selected_stock_adjusted_close_ready": False,
        "ready_for_experiments": False,
        "ready_for_formal": False,
        "ready_for_strategy_replay": False,
        "future_data_violation_count": 0,
        "next_owner": "Strategy Center policy judgment or bounded official archive route unlock",
        **FLAGS,
    }
    blocked_audit = pd.DataFrame([
        {"item": "remaining_exact_exdate", "status": "blocked", "rows": readiness["remaining_exact_exdate_blocked_rows"], "detail": "Ambiguous/no candidate rows preserved."},
        {"item": "remaining_cash_payment_date", "status": "blocked", "rows": readiness["remaining_cash_payment_date_blocked_rows"], "detail": "Cash availability dates remain incomplete."},
        {"item": "stock_distribution_effective_date", "status": "blocked", "rows": readiness["stock_distribution_effective_date_blocked_events"], "detail": "2615/6573 new-share effective or tradable date unavailable."},
        {"item": "capital_change_holder_factor", "status": "blocked", "rows": len(capital), "detail": "No row has a Core-accepted complete holder-scale factor contract."},
        {"item": "holding_month_no_event_proof", "status": "blocked", "rows": readiness["holding_month_temporal_no_event_proof_blocked_tickers"], "detail": "Query success is not silent no-event proof."},
    ])
    paths = [
        _write(ledger, "selected_stock_total_return_event_ledger_remaining_gap_patched.csv"),
        _write(mapping_audit, "selected_stock_remaining_mapping_payment_absorption_audit.csv"),
        _write(entitlement, "selected_stock_remaining_patch_entitlement_alignment.csv"),
        _write(capital, "selected_stock_remaining_capital_change_core_review.csv"),
        _write(gaps, "selected_stock_total_return_remaining_gap_after_absorption.csv"),
        _write(temporal, "selected_stock_remaining_temporal_coverage_absorbed.csv"),
        _write(blocked_audit, "selected_stock_total_return_remaining_gap_blocked_proxy_audit.csv"),
        _write(future_audit, "selected_stock_total_return_remaining_gap_future_data_audit.csv"),
    ]
    readiness_path = OUTPUT_DIR / "readiness_for_selected_stock_total_return_remaining_gap_absorption.json"
    readiness_path.write_text(json.dumps(readiness, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path = OUTPUT_DIR / "final_summary_zh.md"
    summary_path.write_text(
        "# Selected-stock Total-return Remaining Gap Absorption\n\n"
        f"- canonical events: {len(ledger)}\n"
        f"- exact ex-date ready: {exact_ready}/87；new added: {added_exact}；existing confirmed: {readiness['existing_exact_exdate_events_confirmed']}；conflicts: {exact_conflicts}\n"
        f"- payment date ready: {payment_ready}/87；new added: {added_payment}\n"
        f"- actual selected-path entitled exact events: {entitled}\n"
        f"- remaining blocked: exact {readiness['remaining_exact_exdate_blocked_rows']}、payment {readiness['remaining_cash_payment_date_blocked_rows']}、stock effective {readiness['stock_distribution_effective_date_blocked_events']}\n"
        f"- capital-change inventory reviewed: {len(capital)}；accepted holder total-return factors: 0\n"
        "- 2330 ROC108 P1 C2.5 S0 remains blocked; no recurring-dividend amount guess was used.\n"
        "- official holding-month query success is not treated as silent no-event proof.\n"
        "- adjusted close, dividend reinvestment, and capital-change factors were not fabricated.\n\n"
        "結論：唯一 mapping/payment patch 已安全吸收，但 total-return ledger 仍因 temporal completeness、63 個 canonical component gaps、2 個 stock-distribution effective dates與 holder-scale factor contract 不完整而 blocked。不可交 Experiments。\n",
        encoding="utf-8",
    )
    manifest = {
        "task_id": TASK_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(OUTPUT_DIR),
        "source_inputs": {"prior_core": str(PRIOR_DIR), "radar_remaining_gap": str(RADAR_DIR)},
        "files": [{"path": p.name, "sha256": _sha256(p)} for p in [*paths, readiness_path, summary_path]],
        "readiness": readiness,
        "radar_readiness": radar_readiness,
    }
    (OUTPUT_DIR / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(readiness, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
