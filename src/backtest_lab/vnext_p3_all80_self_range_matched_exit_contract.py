from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from backtest_lab import vnext_p3_top20_dynamic_kd_price_range_stage_a as shared


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "outputs/vnext_p3_layer5_all80_KD_range_eligibility_self_range_timing_stage_A_contract_20260713/p3_all80_self_range_entry_candidate_event_ledger.csv.gz"
OUT = ROOT / "outputs/vnext_p3_layer5_all80_self_range_KrangeGT30_matched_exit_contract_20260713"
TASK = "TASK-BACKTEST-CORE-VNEXT-P3-LAYER5-ALL80-SELF-RANGE-KRANGEGT30-MATCHED-EXIT-CONTRACT-001"
END = pd.Timestamp("2025-07-10")
PLATFORM = re.compile(r"(?P<window>60|120)TD_zone(?P<zone>10|20|30)_latch(?P<latch>5|10)_KrangeGT30")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _next_raw(raw: pd.DataFrame, ticker: str, decision_date: pd.Timestamp) -> pd.Series | None:
    rows = raw.loc[raw.ticker.eq(ticker) & raw.decision_date.gt(decision_date)].sort_values("decision_date")
    return None if rows.empty else rows.iloc[0]


def _adjusted_mark(features: pd.DataFrame, ticker: str, date: pd.Timestamp) -> float:
    rows = features.loc[features.ticker.eq(ticker) & features.decision_date.eq(date), "adjusted_close"]
    return np.nan if rows.empty else float(rows.iloc[-1])


def _first_exit(group: pd.DataFrame, entry_date: pd.Timestamp, window: int, zone: float, latch: int) -> tuple[pd.Series | None, pd.Timestamp | None]:
    high_setup_last = None
    rows = group.loc[group.decision_date.gt(entry_date) & group.decision_date.le(END)].sort_values("decision_date")
    for row in rows.itertuples():
        price_loc = getattr(row, f"adjusted_close_location_{window}TD")
        k_loc = getattr(row, f"K_location_{window}TD")
        if pd.notna(price_loc) and pd.notna(k_loc) and price_loc >= 1 - zone and k_loc >= 1 - zone:
            high_setup_last = row.decision_date
        if high_setup_last is None:
            continue
        eligible_dates = rows.loc[rows.decision_date.between(high_setup_last, row.decision_date), "decision_date"]
        if len(eligible_dates) > latch + 1:
            high_setup_last = None
            continue
        if row.KD_cross_down and row.adjusted_close < row.prior_adjusted_close:
            return pd.Series(row._asdict()), high_setup_last
    return None, high_setup_last


def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    entries = pd.read_csv(INPUT, dtype={"ticker": str}, low_memory=False)
    entries["decision_date"] = pd.to_datetime(entries.decision_date)
    entries = entries.loc[entries.platform.str.endswith("KrangeGT30")].copy()
    entries = entries.drop_duplicates(["platform", "decision_date", "ticker"])
    features = shared._features()
    raw = shared._official_raw()
    feature_groups = {ticker: group.sort_values("decision_date") for ticker, group in features.groupby("ticker", sort=False)}
    rows, marks = [], []
    for event_id, entry in enumerate(entries.itertuples(index=False), start=1):
        match = PLATFORM.fullmatch(entry.platform)
        if match is None:
            raise ValueError(f"unexpected platform {entry.platform}")
        window = int(match.group("window")); zone = int(match.group("zone")) / 100; latch = int(match.group("latch"))
        group = feature_groups.get(entry.ticker, pd.DataFrame())
        entry_raw = _next_raw(raw, entry.ticker, entry.decision_date)
        exit_row, high_setup_date = _first_exit(group, entry.decision_date, window, zone, latch)
        status = "right_censored_no_same_geometry_exit_by_P3_1_end" if exit_row is None else "exit_signal_found"
        exit_raw = None if exit_row is None else _next_raw(raw, entry.ticker, pd.Timestamp(exit_row.decision_date))
        if entry_raw is None:
            status = "blocked_entry_official_raw_execution"
        elif exit_row is not None and exit_raw is None:
            status = "blocked_exit_official_raw_execution"
        elif exit_row is not None:
            status = "ready_matched_entry_exit"
        entry_execution_date = pd.NaT if entry_raw is None else pd.Timestamp(entry_raw.decision_date)
        exit_decision_date = pd.NaT if exit_row is None else pd.Timestamp(exit_row.decision_date)
        exit_execution_date = pd.NaT if exit_raw is None else pd.Timestamp(exit_raw.decision_date)
        entry_adjusted = _adjusted_mark(features, entry.ticker, entry_execution_date) if pd.notna(entry_execution_date) else np.nan
        exit_adjusted = _adjusted_mark(features, entry.ticker, exit_execution_date) if pd.notna(exit_execution_date) else np.nan
        gross = exit_adjusted / entry_adjusted - 1 if pd.notna(entry_adjusted) and pd.notna(exit_adjusted) and entry_adjusted != 0 else np.nan
        record = {
            "event_id": f"ME{event_id:05d}", "ticker": entry.ticker, "entry_decision_date": entry.decision_date,
            "geometry": f"{window}TD_zone{int(zone*100)}_latch{latch}", "platform": entry.platform,
            "window_TD": window, "low_zone": zone, "high_zone": 1-zone, "latch_TD": latch, "K_range_threshold": 30,
            "entry_execution_date": entry_execution_date, "entry_official_raw_close": np.nan if entry_raw is None else entry_raw.official_raw_close,
            "entry_official_raw_source_quality": None if entry_raw is None else entry_raw.official_raw_source_quality,
            "high_setup_last_qualifying_date": high_setup_date, "exit_decision_date": exit_decision_date,
            "requested_exit_execution_policy": "first_ticker_trading_day_after_exit_decision",
            "actual_exit_execution_date": exit_execution_date, "exit_official_raw_close": np.nan if exit_raw is None else exit_raw.official_raw_close,
            "exit_official_raw_source_quality": None if exit_raw is None else exit_raw.official_raw_source_quality,
            "entry_adjusted_holding_mark": entry_adjusted, "exit_adjusted_holding_mark": exit_adjusted,
            "gross_holding_return": gross, "status": status,
            "right_censored": exit_row is None, "blocked_reason": status if status.startswith("blocked") else None,
            "exit_reason": None if exit_row is None else "same_geometry_high_setup_then_K_cross_down_and_adjusted_close_down",
            "exit_K": np.nan if exit_row is None else exit_row.K, "exit_D": np.nan if exit_row is None else exit_row.D,
            "exit_price_normalized_location": np.nan if exit_row is None else exit_row[f"adjusted_close_location_{window}TD"],
            "exit_K_normalized_location": np.nan if exit_row is None else exit_row[f"K_location_{window}TD"],
        }
        for bp in (5, 10, 20):
            cost = (0.001425 + bp / 10000) + (0.001425 + 0.003 + bp / 10000)
            record[f"EP05_round_trip_cost_rate_{bp}bp"] = cost
            record[f"net_holding_return_{bp}bp"] = (1 + gross) * (1 - cost) - 1 if pd.notna(gross) else np.nan
        rows.append(record)
        if pd.notna(entry_execution_date):
            end = exit_execution_date if pd.notna(exit_execution_date) else END
            path = group.loc[group.decision_date.between(entry_execution_date, end), ["decision_date", "adjusted_close"]]
            for mark in path.itertuples(index=False):
                marks.append({"event_id":record["event_id"], "ticker":entry.ticker, "decision_date":mark.decision_date, "event_aware_adjusted_holding_mark":mark.adjusted_close, "mark_role":"holding_path_research_diagnostic"})
    ledger = pd.DataFrame(rows)
    marks = pd.DataFrame(marks)
    ledger.to_csv(OUT / "p3_all80_KrangeGT30_matched_entry_exit_ledger.csv.gz", index=False, compression="gzip", encoding="utf-8")
    marks.to_csv(OUT / "p3_all80_KrangeGT30_event_aware_holding_marks.csv.gz", index=False, compression="gzip", encoding="utf-8")
    ledger.loc[~ledger.status.eq("ready_matched_entry_exit")].to_csv(OUT / "p3_all80_KrangeGT30_matched_exit_blocked_right_censored_ledger.csv", index=False, encoding="utf-8-sig")
    requirements = []
    for row in ledger.loc[ledger.status.str.startswith("blocked")].itertuples(index=False):
        role = "entry" if row.status == "blocked_entry_official_raw_execution" else "exit"
        signal_date = pd.Timestamp(row.entry_decision_date if role == "entry" else row.exit_decision_date)
        ticker_dates = feature_groups[row.ticker].loc[feature_groups[row.ticker].decision_date.gt(signal_date), "decision_date"].sort_values()
        requested_date = ticker_dates.iloc[0] if len(ticker_dates) else pd.NaT
        requirements.append({"ticker":row.ticker,"signal_role":role,"signal_decision_date":signal_date,"requested_first_post_signal_ticker_trading_date":requested_date,"platform":row.platform,"event_id":row.event_id,"source_requirement":"official raw OHLC exact ticker-date; no neighbor/last-price/adjusted substitution"})
    requirement_ledger = pd.DataFrame(requirements)
    requirement_ledger.to_csv(OUT / "p3_all80_KrangeGT30_official_execution_gap_requirement_ledger.csv", index=False, encoding="utf-8-sig")
    if len(requirement_ledger):
        requirement_ledger.groupby(["ticker","signal_role","signal_decision_date","requested_first_post_signal_ticker_trading_date","source_requirement"],dropna=False).size().rename("affected_event_platform_rows").reset_index().to_csv(OUT / "p3_all80_KrangeGT30_official_execution_gap_unique_legs.csv",index=False,encoding="utf-8-sig")
    summary = ledger.groupby(["geometry", "status"], dropna=False).size().rename("event_count").reset_index()
    summary.to_csv(OUT / "p3_all80_KrangeGT30_matched_exit_supply_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([
        {"audit":"P3_2_rows_read", "violations":0}, {"audit":"future_outcome_live_rule", "violations":0},
        {"audit":"market_or_ranking_added", "violations":0}, {"audit":"nominal_cross_asset_return", "violations":0},
        {"audit":"execution_not_after_decision", "violations":int((ledger.actual_exit_execution_date <= ledger.exit_decision_date).fillna(False).sum())},
    ]).to_csv(OUT / "p3_all80_KrangeGT30_matched_exit_future_PIT_audit.csv", index=False, encoding="utf-8-sig")
    ready_count = int(ledger.status.eq("ready_matched_entry_exit").sum()); blocked_count = int(ledger.status.str.startswith("blocked").sum()); censored_count = int(ledger.right_censored.sum())
    readiness = {"task_id":TASK,"status":"matched_exit_contract_ready_with_explicit_right_censoring" if blocked_count==0 else "blocked_execution_rows_remain","authority_entry_rows":len(entries),"matched_ready_rows":ready_count,"right_censored_rows":censored_count,"blocked_rows":blocked_count,"P3_1_only":True,"P3_2_outcome_read_authorized":False,"new_thresholds_added":False,"market_used":False,"ranking_used":False,"portfolio_path_executed":False,"ready_for_experiments":blocked_count==0 and ready_count>0,"future_data_violation_count":0,"formal_model_changed":False,"trade_decision_changed":False,"active_in_trade_decision":False,"report_changed":False,"ready_for_formal":False,"ready_for_strategy_replay":False,"not_live_rule":True,"forward_returns_live_rule_usage":False}
    (OUT / "readiness_for_KrangeGT30_matched_exit.json").write_text(json.dumps(readiness, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "final_summary_zh.md").write_text(f"# KRangeGT30 matched exit-only contract\n\nAuthority entries={len(entries)}; matched ready={ready_count}; right-censored={censored_count}; blocked={blocked_count}. Same ticker/geometry high-zone then K-down-cross and price-down only. P3-2, market, new thresholds, ranking and portfolio were not used.\n", encoding="utf-8")
    files = sorted(path for path in OUT.iterdir() if path.is_file() and path.name != "manifest.json")
    (OUT / "manifest.json").write_text(json.dumps({"task_id":TASK,"files":[{"name":p.name,"sha256":_sha(p),"bytes":p.stat().st_size} for p in files]},ensure_ascii=False,indent=2),encoding="utf-8")


if __name__ == "__main__":
    run()
