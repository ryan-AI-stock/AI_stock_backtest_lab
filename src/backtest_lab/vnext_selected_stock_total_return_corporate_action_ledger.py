from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
RADAR_DIR = Path(
    "C:/Users/zergv/Documents/Codex/2026-05-23/ai-stock-rotation-radar-https-docs/outputs/"
    "radar_vnext_selected_stock_corporate_action_distribution_source_package_20260710"
)
R6_STATE = REPO_ROOT / "outputs" / "vnext_weekly_r6_single_position_state_boundary_reconstruction_contract_20260710" / "reconstructed_weekly_r6_single_position_daily_state_rows.csv"
DAILY_F_STATE = REPO_ROOT / "outputs" / "vnext_daily_incumbent_challenger_state_machine_contract_ohlc_absorbed_20260710" / "daily_incumbent_challenger_state_machine_contract_ohlc_absorbed.csv"
OUTPUT_DIR = REPO_ROOT / "outputs" / "vnext_selected_stock_total_return_corporate_action_ledger_20260710"

TASK_ID = "TASK-BACKTEST-CORE-VNEXT-SELECTED-STOCK-TOTAL-RETURN-AND-CORPORATE-ACTION-LEDGER-001"
RAW_F_VARIANT = "F_two_day_confirmation_and_risk_adjusted_edge"
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


def _join_unique(values: pd.Series) -> str:
    return "|".join(sorted(set(str(value) for value in values.dropna() if str(value))))


def _load_holding_legs() -> pd.DataFrame:
    r6 = pd.read_csv(R6_STATE, low_memory=False, dtype={"selected_ticker_after": str})
    r6["base_strategy"] = "reconstructed_single_position_R6"
    r6["ticker"] = r6["selected_ticker_after"].map(_ticker)
    r6["asset_type"] = r6["selected_asset_type_after"]
    r6["hold_start"] = pd.to_datetime(r6["next_trading_day_execution_date"], errors="coerce")
    r6["hold_end_exclusive"] = pd.to_datetime(r6["next_trading_day_after_execution_date"], errors="coerce")
    r6["signal_date"] = pd.to_datetime(r6["signal_date"], errors="coerce")

    raw_f = pd.read_csv(DAILY_F_STATE, low_memory=False, dtype={"selected_ticker_after": str})
    raw_f = raw_f[raw_f["state_machine_variant"].eq(RAW_F_VARIANT)].copy()
    raw_f["base_strategy"] = "raw_Daily_F_challenger"
    raw_f["ticker"] = raw_f["selected_ticker_after"].map(_ticker)
    raw_f["asset_type"] = raw_f["selected_asset_type_after"]
    raw_f["hold_start"] = pd.to_datetime(raw_f["next_trading_day_execution_date"], errors="coerce")
    raw_f["hold_end_exclusive"] = pd.to_datetime(raw_f["next_trading_day_after_execution_date"], errors="coerce")
    raw_f["signal_date"] = pd.to_datetime(raw_f["signal_date"], errors="coerce")

    keep = ["base_strategy", "signal_date", "ticker", "asset_type", "hold_start", "hold_end_exclusive"]
    legs = pd.concat([r6[keep], raw_f[keep]], ignore_index=True)
    legs = legs[legs["asset_type"].eq("stock") & legs["ticker"].ne("")].dropna(subset=["hold_start"]).copy()
    legs["hold_end_exclusive"] = legs["hold_end_exclusive"].fillna(legs["hold_start"] + pd.Timedelta(days=1))
    legs["holding_leg_source_quality"] = "official_unadjusted_selected_stock_daily_state_path_diagnostic"
    return legs.sort_values(["ticker", "base_strategy", "hold_start"])


def _canonical_events() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(RADAR_DIR / "selected_stock_cash_distribution_events.csv", low_memory=False, dtype={"ticker": str})
    raw["ticker"] = raw["ticker"].map(_ticker)
    for column in ["board_resolution_date", "shareholder_meeting_date", "ex_date", "payment_date"]:
        raw[column] = pd.to_datetime(raw[column], errors="coerce")
    for column in ["cash_dividend_total_per_share_candidate", "stock_dividend_total_per_share_candidate"]:
        raw[column] = pd.to_numeric(raw[column], errors="coerce").fillna(0.0)
    raw["dividend_year_roc"] = pd.to_numeric(raw["dividend_year_roc"], errors="coerce").astype("Int64")
    raw["period"] = raw["period"].astype(str)
    keys = [
        "ticker", "dividend_year_roc", "period", "cash_dividend_total_per_share_candidate",
        "stock_dividend_total_per_share_candidate",
    ]
    canonical = raw.groupby(keys, dropna=False, as_index=False).agg(
        company_name=("company_name", "first"),
        market=("market", "first"),
        board_resolution_date=("board_resolution_date", "min"),
        shareholder_meeting_date=("shareholder_meeting_date", "min"),
        ex_date=("ex_date", "min"),
        payment_date=("payment_date", "min"),
        source_ids=("source_id", _join_unique),
        source_urls=("source_url", _join_unique),
        source_quality=("source_quality", _join_unique),
        source_candidate_rows=("source_id", "size"),
    )
    canonical["event_key"] = canonical.apply(
        lambda row: f"{row.ticker}|ROC{row.dividend_year_roc}|P{row.period}|C{row.cash_dividend_total_per_share_candidate:g}|S{row.stock_dividend_total_per_share_candidate:g}",
        axis=1,
    )
    canonical["candidate_window_start"] = canonical["shareholder_meeting_date"].combine_first(canonical["board_resolution_date"])
    canonical["candidate_window_end"] = canonical["candidate_window_start"] + pd.Timedelta(days=180)
    canonical["candidate_window_policy"] = "source_acquisition_priority_only_shareholder_or_board_date_plus_180d_not_exdate"
    canonical["exact_exdate_ready"] = canonical["ex_date"].notna()
    canonical["payment_date_ready"] = canonical["payment_date"].notna()
    canonical["cash_distribution_amount_candidate_ready"] = canonical["cash_dividend_total_per_share_candidate"].gt(0)
    canonical["share_adjustment_candidate_flag"] = canonical["stock_dividend_total_per_share_candidate"].gt(0)
    canonical["accepted_for_total_return_ledger"] = False
    canonical["acceptance_blocker"] = "exact_exdate_and_payment_date_and_complete_capital_change_inventory_required"
    canonical["future_data_violation_count"] = 0
    return canonical, raw


def _align_events(canonical: pd.DataFrame, legs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for event in canonical.itertuples(index=False):
        candidates = legs[legs["ticker"].eq(event.ticker)]
        if pd.isna(event.candidate_window_start):
            overlaps = candidates.iloc[0:0]
        else:
            overlaps = candidates[
                candidates["hold_start"].le(event.candidate_window_end)
                & candidates["hold_end_exclusive"].gt(event.candidate_window_start)
            ]
        strategies = _join_unique(overlaps["base_strategy"]) if len(overlaps) else ""
        rows.append({
            "event_key": event.event_key,
            "ticker": event.ticker,
            "company_name": event.company_name,
            "dividend_year_roc": event.dividend_year_roc,
            "cash_dividend_per_share_candidate": event.cash_dividend_total_per_share_candidate,
            "stock_dividend_per_share_candidate": event.stock_dividend_total_per_share_candidate,
            "candidate_window_start": event.candidate_window_start,
            "candidate_window_end": event.candidate_window_end,
            "candidate_window_overlap_holding_leg_count": len(overlaps),
            "candidate_window_overlap_base_strategies": strategies,
            "candidate_window_overlap_flag": bool(len(overlaps)),
            "alignment_status": "high_priority_exact_event_date_required" if len(overlaps) else "selected_ticker_event_candidate_no_proxy_window_overlap",
            "entitlement_determined": False,
            "entitlement_blocker": "exact_exdate_missing_candidate_window_is_not_entitlement_date",
            "diagnostic_only": True,
            **FLAGS,
        })
    alignment = pd.DataFrame(rows)
    detail = canonical.merge(alignment[[
        "event_key", "candidate_window_overlap_holding_leg_count", "candidate_window_overlap_base_strategies",
        "candidate_window_overlap_flag", "alignment_status", "entitlement_determined", "entitlement_blocker",
    ]], on="event_key", how="left")
    return alignment, detail


def _wealth_requirements() -> pd.DataFrame:
    rows = [
        ("official_price_path", "daily executable/mark close by ticker", "ready_unadjusted_diagnostic", "price change mark-to-market", True),
        ("single_position_state", "ticker, shares, cash, receivable by execution date", "schema_ready_not_materialized_total_return", "one asset plus distribution receivable", True),
        ("cash_dividend_amount", "cash dividend per share", "partial_source_candidate_ROC107_110", "cash receivable = entitled shares * cash/share", True),
        ("exact_exdate", "historical ex-dividend/ex-right trading date", "blocked", "determine entitlement and price discontinuity date", True),
        ("record_date", "historical record date", "blocked", "audit entitlement semantics", False),
        ("payment_date", "historical cash payment date", "blocked", "move receivable to cash without lookahead", True),
        ("stock_distribution_ratio", "new shares per held share", "partial_candidate_3_rows", "adjust share count", True),
        ("stock_distribution_effective_date", "tradable effective/listing date for new shares", "blocked", "avoid crediting shares before tradable", True),
        ("capital_reduction", "ratio, cash return, effective/trading-resumption date", "blocked", "adjust shares, price basis and cash", True),
        ("split_reverse_split", "share ratio and effective date", "blocked", "adjust shares without creating wealth", True),
        ("merger_share_conversion", "old/new ticker ratio and effective date", "blocked", "preserve wealth across security identity change", True),
        ("par_value_change", "share ratio and effective date", "blocked", "adjust shares and price scale", True),
        ("market_available_timestamp", "PIT publication timestamp for event terms", "blocked_exact", "prevent using terms before public", True),
        ("transition_cost", "EP05 stock/ETF fee and tax", "ready", "charge only actual state transitions", True),
        ("cash_dividend_reinvestment", "policy choice", "blocked_policy_not_assumed", "default should hold cash until strategy transition unless explicitly approved", False),
    ]
    return pd.DataFrame(rows, columns=["requirement", "required_fields", "status", "wealth_path_use", "required_before_total_return_ready"]).assign(
        no_silent_fill=True,
        no_adjusted_close_fabrication=True,
        diagnostic_only=True,
        **FLAGS,
    )


def _event_ledger_template(detail: pd.DataFrame) -> pd.DataFrame:
    out = detail.copy()
    out["event_type"] = np.where(
        out["share_adjustment_candidate_flag"], "cash_and_or_stock_distribution_candidate", "cash_distribution_candidate"
    )
    out["announcement_available_date"] = out["board_resolution_date"].combine_first(out["shareholder_meeting_date"])
    out["record_date"] = pd.NaT
    out["share_adjustment_effective_date"] = pd.NaT
    out["capital_change_type"] = "blocked_no_complete_inventory"
    out["shares_before"] = np.nan
    out["shares_after"] = np.nan
    out["cash_receivable"] = np.nan
    out["cash_paid"] = np.nan
    out["event_wealth_delta"] = np.nan
    out["total_return_factor"] = np.nan
    out["ledger_row_ready"] = False
    out["ledger_status"] = "source_candidate_only_not_wealth_path_ready"
    return out


def _gap_ledger(detail: pd.DataFrame, universe: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for event in detail.itertuples(index=False):
        priority = "high" if event.candidate_window_overlap_flag else "medium"
        common = {
            "ticker": event.ticker,
            "company_name": event.company_name,
            "event_key": event.event_key,
            "candidate_window_start": event.candidate_window_start,
            "candidate_window_end": event.candidate_window_end,
            "acquisition_priority": priority,
        }
        rows.append({**common, "missing_component": "exact_historical_exdate", "required_source": "official TWSE/TPEx historical ex-right/dividend date route", "blocks_total_return": True})
        if event.cash_dividend_total_per_share_candidate > 0:
            rows.append({**common, "missing_component": "cash_payment_date", "required_source": "official issuer/MOPS/TWSE/TPEx historical payment-date route", "blocks_total_return": True})
        if event.stock_dividend_total_per_share_candidate > 0:
            rows.append({**common, "missing_component": "stock_distribution_effective_tradable_date", "required_source": "official historical new-share listing/effective-date route", "blocks_total_return": True})
    for item in universe.itertuples(index=False):
        if str(item.instrument_type) != "ordinary_stock":
            continue
        rows.append({
            "ticker": _ticker(item.ticker), "company_name": item.name, "event_key": "ticker_level_inventory",
            "candidate_window_start": item.coverage_start, "candidate_window_end": item.coverage_end,
            "acquisition_priority": "high_if_selected_period_intersects_event",
            "missing_component": "non_dividend_capital_change_inventory",
            "required_source": "official capital reduction/split/reverse split/merger/share conversion/par-value change historical routes",
            "blocks_total_return": True,
        })
    gap = pd.DataFrame(rows)
    gap["next_owner"] = "Radar/Data bounded selected-ticker official route unlock"
    gap["no_full_market_download"] = True
    gap["future_data_violation_count"] = 0
    return gap


def _source_quality(raw: pd.DataFrame, canonical: pd.DataFrame, detail: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([
        {"source_component": "t187ap39_40_raw_candidates", "rows": len(raw), "coverage": "ROC107-110 route snapshot", "status": "partial_source_candidate", "usable_for_total_return": False},
        {"source_component": "canonical_distribution_candidates", "rows": len(canonical), "coverage": f"tickers={canonical['ticker'].nunique()}", "status": "deduplicated_candidate_inventory", "usable_for_total_return": False},
        {"source_component": "exact_exdate", "rows": int(detail["exact_exdate_ready"].sum()), "coverage": f"of {len(detail)} canonical events", "status": "blocked", "usable_for_total_return": False},
        {"source_component": "payment_date", "rows": int(detail["payment_date_ready"].sum()), "coverage": f"of {len(detail)} canonical events", "status": "blocked", "usable_for_total_return": False},
        {"source_component": "stock_distribution_candidates", "rows": int(detail["share_adjustment_candidate_flag"].sum()), "coverage": "candidate only", "status": "partial_missing_exact_effective_date", "usable_for_total_return": False},
        {"source_component": "capital_change_inventory", "rows": 0, "coverage": "selected tickers", "status": "blocked", "usable_for_total_return": False},
    ]).assign(future_data_violation_count=0, diagnostic_only=True, **FLAGS)


def _future_audit() -> pd.DataFrame:
    return pd.DataFrame([
        {"audit_item": "candidate_distribution_absorption", "future_data_used": False, "detail": "Official source candidates copied without calculating returns or factors.", "future_data_violation_count": 0},
        {"audit_item": "candidate_window_alignment", "future_data_used": False, "detail": "Board/shareholder date +180d is acquisition-priority proxy only, never treated as ex-date or entitlement.", "future_data_violation_count": 0},
        {"audit_item": "adjusted_close_total_return", "future_data_used": False, "detail": "No adjusted close, reinvestment, share factor or wealth path was fabricated.", "future_data_violation_count": 0},
    ])


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    readiness_in = json.loads((RADAR_DIR / "readiness_for_core_selected_stock_total_return_ledger.json").read_text(encoding="utf-8"))
    universe = pd.read_csv(RADAR_DIR / "selected_stock_universe.csv", low_memory=False, dtype={"ticker": str})
    manifest = pd.read_csv(RADAR_DIR / "selected_stock_event_source_manifest.csv", low_memory=False)
    benchmark_status = pd.read_csv(RADAR_DIR / "selected_stock_benchmark_etf_source_status.csv", low_memory=False)
    factor_candidates = pd.read_csv(RADAR_DIR / "selected_stock_adjustment_factor_source_candidates.csv", low_memory=False, dtype={"ticker": str})
    holding_legs = _load_holding_legs()
    canonical, raw = _canonical_events()
    alignment, detail = _align_events(canonical, holding_legs)
    ledger_template = _event_ledger_template(detail)
    gaps = _gap_ledger(detail, universe)
    high_priority_events = int(detail["candidate_window_overlap_flag"].sum())
    exact_exdate_ready = int(detail["exact_exdate_ready"].sum())
    payment_ready = int(detail["payment_date_ready"].sum())
    total_return_ready = bool(
        len(detail) > 0
        and exact_exdate_ready == len(detail)
        and payment_ready == int(detail["cash_distribution_amount_candidate_ready"].sum())
        and not gaps["missing_component"].eq("non_dividend_capital_change_inventory").any()
    )
    readiness = {
        "task_id": TASK_ID,
        "status": "partial_source_absorption_useful_total_return_ledger_blocked_exact_dates_and_capital_changes",
        "radar_source_absorption_review_ready": True,
        "selected_ordinary_stock_tickers": int((universe["instrument_type"] == "ordinary_stock").sum()),
        "raw_distribution_candidate_rows": len(raw),
        "canonical_distribution_candidate_events": len(canonical),
        "candidate_event_ticker_count": int(canonical["ticker"].nunique()),
        "holding_window_high_priority_event_candidates": high_priority_events,
        "exact_exdate_ready_events": exact_exdate_ready,
        "payment_date_ready_events": payment_ready,
        "stock_distribution_candidate_events": int(detail["share_adjustment_candidate_flag"].sum()),
        "capital_change_inventory_ready": False,
        "selected_stock_total_return_ledger_ready": total_return_ready,
        "selected_stock_adjusted_close_ready": False,
        "cash_dividend_reinvestment_policy_decided": False,
        "ready_for_experiments": False,
        "ready_for_formal": False,
        "ready_for_strategy_replay": False,
        "future_data_violation_count": 0,
        "next_owner": "Radar/Data exact exdate/payment-date/capital-change official route unlock",
        **FLAGS,
    }
    blocked = pd.DataFrame([
        {"item": "exact_historical_exdate", "status": "blocked", "detail": "Required to determine entitlement and price discontinuity date.", "can_compute_total_return": False},
        {"item": "cash_payment_date", "status": "blocked", "detail": "Required to recognize receivable-to-cash timing without lookahead.", "can_compute_total_return": False},
        {"item": "capital_change_inventory", "status": "blocked", "detail": "Capital reduction/split/merger/share conversion/par-value changes incomplete.", "can_compute_total_return": False},
        {"item": "cash_dividend_reinvestment", "status": "policy_blocked", "detail": "No automatic reinvestment assumption; default wealth schema keeps cash receivable/cash separate.", "can_compute_total_return": False},
        {"item": "benchmark_ETFs", "status": "separate", "detail": "0050/00631L status remains outside ordinary-stock corporate-action ledger.", "can_compute_total_return": False},
    ])
    output_paths = [
        _write(universe, "selected_stock_total_return_universe.csv"),
        _write(manifest, "selected_stock_corporate_action_source_manifest.csv"),
        _write(benchmark_status, "selected_stock_benchmark_etf_separate_status.csv"),
        _write(raw, "selected_stock_distribution_source_candidates_absorbed.csv"),
        _write(canonical, "selected_stock_canonical_distribution_event_candidates.csv"),
        _write(factor_candidates, "selected_stock_adjustment_factor_source_candidates_absorbed.csv"),
        _write(holding_legs, "selected_stock_actual_holding_legs.csv"),
        _write(alignment, "selected_stock_holding_event_candidate_alignment.csv"),
        _write(ledger_template, "selected_stock_total_return_event_ledger_template.csv"),
        _write(_wealth_requirements(), "selected_stock_total_return_wealth_path_requirements.csv"),
        _write(gaps, "selected_stock_total_return_corporate_action_gap_ledger.csv"),
        _write(_source_quality(raw, canonical, detail), "selected_stock_corporate_action_source_quality_audit.csv"),
        _write(blocked, "selected_stock_total_return_blocked_proxy_audit.csv"),
        _write(_future_audit(), "selected_stock_total_return_future_data_audit.csv"),
    ]
    readiness_path = OUTPUT_DIR / "readiness_for_selected_stock_total_return_corporate_action_ledger.json"
    readiness_path.write_text(json.dumps(readiness, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path = OUTPUT_DIR / "final_summary_zh.md"
    summary_path.write_text(
        "# Selected-stock Total-return / Corporate-action Ledger\n\n"
        "Radar source candidates 可用於建立 selected-stock event inventory 與補源優先順序，但不能直接計算 adjusted close 或 total-return wealth path。\n\n"
        f"- raw distribution candidates: {len(raw)}；canonical events: {len(canonical)}；tickers: {canonical['ticker'].nunique()}\n"
        f"- candidate-window overlaps with actual holding legs: {high_priority_events}（只作補源優先，不是 entitlement）\n"
        f"- exact ex-date ready: {exact_exdate_ready}/{len(canonical)}；payment date ready: {payment_ready}/{len(canonical)}\n"
        "- non-dividend capital-change inventory: blocked\n"
        "- no adjusted-close calculation；no dividend reinvestment assumption；no strategy replay\n\n"
        "結論：partial source absorption useful，但 selected_stock_total_return_ledger_ready=false。下一棒需 Radar/Data bounded exact ex-date/payment-date/capital-change official route unlock。\n",
        encoding="utf-8",
    )
    manifest_out = {
        "task_id": TASK_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(OUTPUT_DIR),
        "files": [{"path": path.name, "sha256": _sha256(path)} for path in [*output_paths, readiness_path, summary_path]],
        "readiness": readiness,
        "source_inputs": {"radar_package": str(RADAR_DIR), "R6_state": str(R6_STATE), "Daily_F_state": str(DAILY_F_STATE)},
        "upstream_readiness": readiness_in,
    }
    (OUTPUT_DIR / "manifest.json").write_text(json.dumps(manifest_out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(readiness, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
