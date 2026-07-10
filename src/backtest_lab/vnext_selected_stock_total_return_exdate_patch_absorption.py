from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
CORE_DIR = REPO_ROOT / "outputs" / "vnext_selected_stock_total_return_corporate_action_ledger_20260710"
RADAR_DIR = Path(
    "C:/Users/zergv/Documents/Codex/2026-05-23/ai-stock-rotation-radar-https-docs/outputs/"
    "radar_vnext_selected_stock_exact_exdate_capital_change_route_unlock_20260710"
)
OUTPUT_DIR = REPO_ROOT / "outputs" / "vnext_selected_stock_total_return_exdate_patch_absorption_20260710"

TASK_ID = "TASK-BACKTEST-CORE-VNEXT-SELECTED-STOCK-TOTAL-RETURN-AND-CORPORATE-ACTION-LEDGER-EXDATE-PATCH-ABSORPTION-001"
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


def _ticker(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    return text[:-2] if text.endswith(".0") else text


def _bool(value: Any) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, str):
        return value.lower() in {"true", "1", "yes"}
    return bool(value)


def _join_unique(values: pd.Series) -> str:
    return "|".join(sorted(set(str(value) for value in values.dropna() if str(value))))


def _load_legs() -> pd.DataFrame:
    legs = pd.read_csv(CORE_DIR / "selected_stock_actual_holding_legs.csv", low_memory=False, dtype={"ticker": str})
    legs["ticker"] = legs["ticker"].map(_ticker)
    legs["hold_start"] = pd.to_datetime(legs["hold_start"], errors="coerce")
    legs["hold_end_exclusive"] = pd.to_datetime(legs["hold_end_exclusive"], errors="coerce")
    return legs


def _entitlement(legs: pd.DataFrame, ticker: str, ex_date: pd.Timestamp) -> pd.DataFrame:
    if pd.isna(ex_date):
        return legs.iloc[0:0]
    return legs[
        legs["ticker"].eq(ticker)
        & legs["hold_start"].lt(ex_date)
        & legs["hold_end_exclusive"].ge(ex_date)
    ]


def _absorb_patch(legs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ledger = pd.read_csv(CORE_DIR / "selected_stock_total_return_event_ledger_template.csv", low_memory=False, dtype={"ticker": str})
    patch = pd.read_csv(RADAR_DIR / "selected_stock_exact_exdate_accepted_patch_rows.csv", low_memory=False, dtype={"ticker": str})
    patch["ticker"] = patch["ticker"].map(_ticker)
    patch["accepted_exact_exdate"] = pd.to_datetime(patch["accepted_exact_exdate"], errors="coerce")
    patch["accepted_payment_date"] = pd.to_datetime(patch["accepted_payment_date"], errors="coerce")
    patch["accepted_market_available_at"] = pd.to_datetime(patch["accepted_market_available_at"], errors="coerce")
    exact_accept = patch["accepted_status"].eq("accepted_unique_exact_exdate_candidate") & patch["accepted_exact_exdate"].notna()
    patch["exact_exdate_patch_accepted"] = exact_accept
    patch["payment_date_patch_accepted"] = exact_accept & patch["accepted_payment_date"].notna()
    patch_cols = [
        "event_key", "accepted_exact_exdate", "accepted_payment_date", "accepted_market_available_at",
        "accepted_source_url", "accepted_subject", "accepted_detail_excerpt", "accepted_status",
        "exact_candidate_count", "payment_candidate_count", "capital_change_candidate_count",
        "exact_exdate_patch_accepted", "payment_date_patch_accepted",
    ]
    ledger = ledger.merge(patch[patch_cols], on="event_key", how="left")
    ledger["exact_exdate_patch_accepted"] = ledger["exact_exdate_patch_accepted"].fillna(False).astype(bool)
    ledger["payment_date_patch_accepted"] = ledger["payment_date_patch_accepted"].fillna(False).astype(bool)
    ledger["ex_date"] = pd.to_datetime(ledger["ex_date"], errors="coerce")
    ledger["payment_date"] = pd.to_datetime(ledger["payment_date"], errors="coerce")
    ledger.loc[ledger["exact_exdate_patch_accepted"], "ex_date"] = ledger.loc[ledger["exact_exdate_patch_accepted"], "accepted_exact_exdate"]
    ledger.loc[ledger["payment_date_patch_accepted"], "payment_date"] = ledger.loc[ledger["payment_date_patch_accepted"], "accepted_payment_date"]
    ledger["exact_exdate_ready"] = ledger["exact_exdate_patch_accepted"]
    ledger["payment_date_ready"] = ledger["payment_date_patch_accepted"]
    entitlement_rows = []
    for row in ledger.itertuples(index=False):
        overlaps = _entitlement(legs, _ticker(row.ticker), row.ex_date) if row.exact_exdate_ready else legs.iloc[0:0]
        entitlement_rows.append({
            "event_key": row.event_key,
            "ticker": _ticker(row.ticker),
            "company_name": row.company_name,
            "exact_exdate": row.ex_date,
            "exact_exdate_ready": row.exact_exdate_ready,
            "payment_date": row.payment_date,
            "payment_date_ready": row.payment_date_ready,
            "entitled_holding_leg_count": len(overlaps),
            "entitled_base_strategies": _join_unique(overlaps["base_strategy"]) if len(overlaps) else "",
            "selected_path_entitlement_flag": bool(len(overlaps)),
            "entitlement_status": "exact_exdate_entitled_selected_path" if len(overlaps) else "accepted_exact_date_no_selected_holding" if row.exact_exdate_ready else "blocked_exact_exdate_unresolved",
            "cash_receivable_amount_ready": bool(len(overlaps) and row.cash_dividend_total_per_share_candidate > 0),
            "cash_available_date_ready": bool(len(overlaps) and row.payment_date_ready),
            "future_data_violation_count": 0,
            **FLAGS,
        })
    entitlement = pd.DataFrame(entitlement_rows)
    ledger = ledger.merge(entitlement[[
        "event_key", "entitled_holding_leg_count", "entitled_base_strategies", "selected_path_entitlement_flag",
        "entitlement_status", "cash_receivable_amount_ready", "cash_available_date_ready",
    ]], on="event_key", how="left")
    ledger["ledger_row_ready"] = (
        ledger["exact_exdate_ready"]
        & (~ledger["selected_path_entitlement_flag"] | ledger["payment_date_ready"])
        & ~ledger["share_adjustment_candidate_flag"]
    )
    ledger["ledger_status"] = np.select(
        [
            ledger["selected_path_entitlement_flag"] & ~ledger["payment_date_ready"],
            ledger["share_adjustment_candidate_flag"],
            ledger["exact_exdate_ready"],
        ],
        [
            "entitled_cash_event_payment_date_blocked",
            "stock_distribution_effective_date_blocked",
            "exact_exdate_absorbed_no_selected_path_entitlement",
        ],
        default="exact_exdate_unresolved",
    )
    ledger["accepted_for_total_return_ledger"] = False
    return ledger, entitlement


def _candidate_date_overlap(legs: pd.DataFrame) -> pd.DataFrame:
    evidence = pd.read_csv(RADAR_DIR / "selected_stock_exact_exdate_candidate_evidence.csv", low_memory=False, dtype={"ticker": str})
    evidence["ticker"] = evidence["ticker"].map(_ticker)
    evidence["ex_date"] = pd.to_datetime(evidence["ex_date"], errors="coerce")
    candidates = evidence[evidence["ex_date"].notna() & evidence["accepted_exact_exdate_candidate"].map(_bool)].copy()
    candidates = candidates.drop_duplicates(["event_key", "ticker", "ex_date", "market_available_at", "subject"])
    rows = []
    for item in candidates.itertuples(index=False):
        overlaps = _entitlement(legs, item.ticker, item.ex_date)
        rows.append({
            "event_key": item.event_key,
            "ticker": item.ticker,
            "candidate_ex_date": item.ex_date,
            "candidate_subject": item.subject,
            "candidate_acceptance_status": item.acceptance_status,
            "candidate_holding_leg_overlap_count": len(overlaps),
            "candidate_overlap_base_strategies": _join_unique(overlaps["base_strategy"]) if len(overlaps) else "",
            "candidate_selected_path_overlap_flag": bool(len(overlaps)),
            "candidate_date_role": "source_candidate_only_until_canonical_event_mapping_unique",
            "future_data_violation_count": 0,
        })
    return pd.DataFrame(rows)


def _capital_class(text: str) -> str:
    compact = text.replace("\r", "").replace("\n", "")
    if any(term in compact for term in ["代子公司", "大陸投資事業", "轉投資大陸", "重要子公司"]):
        return "subsidiary_event_not_selected_listed_security"
    if ("庫藏股" in compact or "限制員工權利新股" in compact) and "註銷" in compact:
        return "issuer_share_count_change_no_holder_share_conversion"
    if "現金減資" in compact and any(term in compact for term in ["每仟股", "每股可退還", "減資換發新股票"]):
        return "listed_holder_share_scale_and_cash_return_candidate"
    if "分割" in compact:
        return "listed_holder_split_candidate_requires_review"
    if "合併" in compact or "股份轉換" in compact:
        return "listed_holder_identity_or_conversion_candidate_requires_review"
    return "other_capital_change_candidate_requires_review"


def _capital_review(legs: pd.DataFrame) -> pd.DataFrame:
    capital = pd.read_csv(RADAR_DIR / "selected_stock_non_dividend_capital_change_inventory.csv", low_memory=False, dtype={"ticker": str})
    capital["ticker"] = capital["ticker"].map(_ticker)
    for column in ["share_adjustment_effective_date", "trading_resumption_date", "record_date", "fact_date", "market_available_at"]:
        capital[column] = pd.to_datetime(capital[column], errors="coerce")
    capital["review_text"] = capital["subject"].fillna("") + " " + capital["detail_text_excerpt"].fillna("")
    capital["core_capital_event_class"] = capital["review_text"].map(_capital_class)
    capital["listed_holder_scale_candidate"] = capital["core_capital_event_class"].str.contains("listed_holder")
    capital["effective_date_candidate"] = capital["trading_resumption_date"].combine_first(capital["share_adjustment_effective_date"])
    overlaps = []
    for row in capital.itertuples(index=False):
        match = _entitlement(legs, row.ticker, row.effective_date_candidate) if pd.notna(row.effective_date_candidate) else legs.iloc[0:0]
        overlaps.append({
            "capital_row_index": row.Index if hasattr(row, "Index") else len(overlaps),
            "selected_path_effective_date_overlap_count": len(match),
            "selected_path_overlap_base_strategies": _join_unique(match["base_strategy"]) if len(match) else "",
            "selected_path_capital_impact_candidate": bool(len(match) and row.listed_holder_scale_candidate),
        })
    capital = pd.concat([capital.reset_index(drop=True), pd.DataFrame(overlaps).drop(columns=["capital_row_index"])], axis=1)
    capital["share_factor_ready"] = False
    capital["cash_return_ready"] = pd.to_numeric(capital["cash_return"], errors="coerce").notna()
    capital["core_review_status"] = np.select(
        [
            capital["core_capital_event_class"].eq("subsidiary_event_not_selected_listed_security"),
            capital["core_capital_event_class"].eq("issuer_share_count_change_no_holder_share_conversion"),
            capital["selected_path_capital_impact_candidate"],
            capital["listed_holder_scale_candidate"],
        ],
        [
            "not_applicable_to_selected_listed_security",
            "no_holder_share_factor_required_information_only",
            "selected_path_impact_candidate_factor_and_cash_terms_blocked",
            "holder_scale_candidate_no_selected_path_date_overlap",
        ],
        default="manual_review_required",
    )
    capital["future_data_violation_count"] = 0
    return capital.drop(columns=["review_text"])


def _remaining_gaps(ledger: pd.DataFrame, capital: pd.DataFrame, universe: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in ledger.itertuples(index=False):
        if not row.exact_exdate_ready:
            rows.append({"ticker": row.ticker, "event_key": row.event_key, "missing_component": "canonical_event_exact_exdate_mapping", "priority": "high" if row.candidate_window_overlap_flag else "medium", "blocks_actual_selected_path": bool(row.candidate_window_overlap_flag)})
        if row.cash_dividend_total_per_share_candidate > 0 and not row.payment_date_ready:
            rows.append({"ticker": row.ticker, "event_key": row.event_key, "missing_component": "cash_payment_date", "priority": "high_if_entitled_else_medium", "blocks_actual_selected_path": bool(row.selected_path_entitlement_flag)})
        if row.share_adjustment_candidate_flag:
            rows.append({"ticker": row.ticker, "event_key": row.event_key, "missing_component": "stock_distribution_effective_tradable_date", "priority": "high_if_entitled_else_medium", "blocks_actual_selected_path": bool(row.selected_path_entitlement_flag)})
    capital_blocked = capital[capital["core_review_status"].isin([
        "selected_path_impact_candidate_factor_and_cash_terms_blocked",
        "manual_review_required",
    ])]
    for row in capital_blocked.itertuples(index=False):
        rows.append({
            "ticker": row.ticker,
            "event_key": row.event_key,
            "missing_component": "capital_change_exact_factor_effective_date_or_cash_terms",
            "priority": "high" if row.selected_path_capital_impact_candidate else "medium",
            "blocks_actual_selected_path": bool(row.selected_path_capital_impact_candidate),
        })
    for row in universe[universe["instrument_type"].eq("ordinary_stock")].itertuples(index=False):
        rows.append({
            "ticker": _ticker(row.ticker), "event_key": "full_selected_coverage_inventory",
            "missing_component": "corporate_action_temporal_coverage_outside_ROC107_110",
            "priority": "high", "blocks_actual_selected_path": True,
        })
    gaps = pd.DataFrame(rows)
    gaps["next_owner"] = "Radar/Data bounded selected-ticker remaining corporate-action gap fill"
    gaps["no_full_market_download"] = True
    gaps["future_data_violation_count"] = 0
    return gaps


def _future_audit() -> pd.DataFrame:
    return pd.DataFrame([
        {"audit_item": "exact_exdate_absorption", "future_data_used": False, "detail": "Only Radar unique accepted official ex-date rows are absorbed.", "future_data_violation_count": 0},
        {"audit_item": "entitlement_alignment", "future_data_used": False, "detail": "Entitlement candidate requires holding before ex-date; later price/return is not used.", "future_data_violation_count": 0},
        {"audit_item": "capital_change_review", "future_data_used": False, "detail": "Official event text is classified for source readiness only; no adjustment factor is computed.", "future_data_violation_count": 0},
        {"audit_item": "total_return_factor", "future_data_used": False, "detail": "Adjusted close and total-return factor remain unmaterialized.", "future_data_violation_count": 0},
    ])


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    readiness_in = json.loads((RADAR_DIR / "readiness_for_core_selected_stock_exact_exdate_capital_change_route_unlock.json").read_text(encoding="utf-8"))
    universe = pd.read_csv(CORE_DIR / "selected_stock_total_return_universe.csv", low_memory=False, dtype={"ticker": str})
    legs = _load_legs()
    ledger, entitlement = _absorb_patch(legs)
    candidate_overlap = _candidate_date_overlap(legs)
    capital = _capital_review(legs)
    gaps = _remaining_gaps(ledger, capital, universe)
    accepted_exact = int(ledger["exact_exdate_ready"].sum())
    accepted_payment = int(ledger["payment_date_ready"].sum())
    payment_source_candidate_rows = len(pd.read_csv(RADAR_DIR / "selected_stock_cash_payment_date_candidate_rows.csv", low_memory=False))
    radar_accepted_payment_events = int(readiness_in["coverage"]["accepted_payment_date_events"])
    entitled = int(ledger["selected_path_entitlement_flag"].sum())
    candidate_date_overlap = int(candidate_overlap["candidate_selected_path_overlap_flag"].sum()) if len(candidate_overlap) else 0
    capital_path_impact = int(capital["selected_path_capital_impact_candidate"].sum())
    actual_path_blockers = int(gaps["blocks_actual_selected_path"].sum())
    readiness = {
        "task_id": TASK_ID,
        "status": "exdate_patch_absorbed_actual_known_events_no_overlap_total_return_still_blocked_temporal_coverage",
        "canonical_events": len(ledger),
        "accepted_exact_exdate_events": accepted_exact,
        "accepted_payment_date_events": accepted_payment,
        "payment_date_source_candidate_rows": payment_source_candidate_rows,
        "radar_accepted_payment_date_events": radar_accepted_payment_events,
        "selected_path_entitled_accepted_events": entitled,
        "candidate_exact_date_selected_path_overlap_rows": candidate_date_overlap,
        "capital_change_inventory_rows": len(capital),
        "capital_change_selected_path_impact_candidates": capital_path_impact,
        "actual_selected_path_remaining_blocker_rows": actual_path_blockers,
        "remaining_component_gap_rows": len(gaps),
        "corporate_action_temporal_coverage_complete": False,
        "selected_stock_total_return_ledger_ready": False,
        "selected_stock_adjusted_close_ready": False,
        "ready_for_experiments": False,
        "ready_for_formal": False,
        "ready_for_strategy_replay": False,
        "future_data_violation_count": 0,
        "next_owner": "Radar/Data remaining exact mapping/payment/stock-effective/full selected-period coverage gap fill",
        **FLAGS,
    }
    blocked = pd.DataFrame([
        {"item": "remaining_exact_event_mapping", "status": "blocked", "rows": int((~ledger["exact_exdate_ready"]).sum()), "detail": "Ambiguous/no unique canonical event mapping preserved."},
        {"item": "remaining_payment_dates", "status": "blocked", "rows": int((ledger["cash_dividend_total_per_share_candidate"].gt(0) & ~ledger["payment_date_ready"]).sum()), "detail": "Cash availability timing remains sparse."},
        {"item": "stock_distribution_effective_dates", "status": "blocked", "rows": int(ledger["share_adjustment_candidate_flag"].sum()), "detail": "New-share tradable/effective date unavailable."},
        {"item": "temporal_source_coverage", "status": "blocked", "rows": int((universe["instrument_type"] == "ordinary_stock").sum()), "detail": "Current distribution source inventory covers ROC107-110 only, not all selected holding years."},
        {"item": "adjusted_close_total_return_factor", "status": "blocked", "rows": 0, "detail": "No factor or reinvestment assumption fabricated."},
    ])
    output_paths = [
        _write(ledger, "selected_stock_total_return_event_ledger_exdate_patched.csv"),
        _write(entitlement, "selected_stock_exact_exdate_entitlement_alignment.csv"),
        _write(candidate_overlap, "selected_stock_ambiguous_event_candidate_date_overlap_audit.csv"),
        _write(capital, "selected_stock_capital_change_core_review.csv"),
        _write(gaps, "selected_stock_total_return_remaining_gap_ledger.csv"),
        _write(pd.read_csv(RADAR_DIR / "selected_stock_exact_exdate_accepted_patch_rows.csv", low_memory=False), "selected_stock_exact_exdate_patch_source_absorbed.csv"),
        _write(pd.read_csv(RADAR_DIR / "selected_stock_cash_payment_date_candidate_rows.csv", low_memory=False), "selected_stock_payment_date_candidates_absorbed.csv"),
        _write(pd.read_csv(RADAR_DIR / "selected_stock_stock_distribution_effective_date_candidates.csv", low_memory=False), "selected_stock_stock_distribution_effective_candidates_absorbed.csv"),
        _write(blocked, "selected_stock_total_return_exdate_patch_blocked_audit.csv"),
        _write(_future_audit(), "selected_stock_total_return_exdate_patch_future_data_audit.csv"),
    ]
    readiness_path = OUTPUT_DIR / "readiness_for_selected_stock_total_return_exdate_patch_absorption.json"
    readiness_path.write_text(json.dumps(readiness, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path = OUTPUT_DIR / "final_summary_zh.md"
    summary_path.write_text(
        "# Selected-stock Total-return Ex-date Patch Absorption\n\n"
        f"- exact ex-date: {accepted_exact}/{len(ledger)}；payment date: {accepted_payment}/{len(ledger)}\n"
        f"- Radar payment candidate rows: {payment_source_candidate_rows}；Radar accepted events: {radar_accepted_payment_events}；canonical-safe absorbed events: {accepted_payment}；unmapped/ambiguous candidates remain blocked.\n"
        f"- accepted ex-date events entitled by actual R6/F holdings: {entitled}\n"
        f"- all candidate exact-date overlaps with actual holdings: {candidate_date_overlap}\n"
        f"- capital-change rows: {len(capital)}；selected-path impact candidates: {capital_path_impact}\n"
        f"- remaining component gap rows: {len(gaps)}；actual selected-path blocker rows: {actual_path_blockers}\n"
        "- current known event dates do not overlap actual selected-stock holdings, but source temporal coverage is only ROC107-110.\n"
        "- adjusted close / total-return factor / dividend reinvestment remain blocked and unmaterialized.\n\n"
        "結論：patch absorption 有效降低已知事件不確定性，但 selected_stock_total_return_ledger_ready=false；需補剩餘事件 mapping、payment、stock effective date 與全 selected-period corporate-action inventory。\n",
        encoding="utf-8",
    )
    manifest = {
        "task_id": TASK_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(OUTPUT_DIR),
        "files": [{"path": path.name, "sha256": _sha256(path)} for path in [*output_paths, readiness_path, summary_path]],
        "readiness": readiness,
        "source_inputs": {"core_prior_ledger": str(CORE_DIR), "radar_patch": str(RADAR_DIR)},
        "upstream_readiness": readiness_in,
    }
    (OUTPUT_DIR / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(readiness, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
