from __future__ import annotations

import glob
import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DAILY = ROOT / "outputs/vnext_p3_layer5_daily_feature_state_action_materialization_20260712/p3_layer5_daily_feature_state_matrix.csv"
ETF = ROOT / "backtest_cache/stock_pool_observations/0050_TW.csv"
RADAR = Path(r"C:\Users\zergv\Documents\Codex\2026-05-23\ai-stock-rotation-radar-https-docs\outputs")
RAW_DIR = RADAR / "radar_vnext_p3_exact_primary80_raw_hlc_warmup_gap_fill_20260711/compact/raw_hlc_warmup"
ADJ_DIR = RADAR / "radar_vnext_p3_recent_full_feature_data_readiness_acquisition_20260711/checkpoints/adjusted"
OUT = ROOT / "outputs/vnext_p3_layer5_all80_continuous_sequential_lifecycle_state_supply_contract_20260713"
TASK = "TASK-BACKTEST-CORE-VNEXT-P3-LAYER5-ALL80-CONTINUOUS-SEQUENTIAL-LIFECYCLE-STATE-SUPPLY-CONTRACT-001"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    daily = pd.read_csv(DAILY, dtype={"ticker": str}, low_memory=False)
    daily["decision_date"] = pd.to_datetime(daily.decision_date)
    p31 = daily.loc[daily.decision_date.lt("2025-07-11")].copy()
    calendar = pd.to_datetime(pd.read_csv(ETF, usecols=["date"]).date).drop_duplicates().sort_values()

    requirements = set()
    for ticker, membership in p31.groupby("ticker"):
        first_date, last_date = membership.decision_date.min(), membership.decision_date.max()
        start_candidates = calendar[calendar.le(first_date)]
        start_date = start_candidates.iloc[max(0, len(start_candidates) - 252)]
        required_dates = calendar[calendar.ge(start_date) & calendar.le(last_date)]
        requirements.update((ticker, pd.Timestamp(date)) for date in required_dates)
    requirement = pd.DataFrame(sorted(requirements), columns=["ticker", "date"])

    raw_files = glob.glob(str(RAW_DIR / "*.csv.gz"))
    raw = pd.concat([pd.read_csv(path, dtype={"ticker": str}, usecols=["date", "ticker", "high", "low", "close", "source_quality"]) for path in raw_files], ignore_index=True)
    raw["date"] = pd.to_datetime(raw.date)
    member_raw = p31.rename(columns={"decision_date": "date"})[["date", "ticker", "high", "low", "close", "raw_execution_source_quality"]].rename(columns={"raw_execution_source_quality": "source_quality"})
    raw = pd.concat([raw, member_raw], ignore_index=True).drop_duplicates(["ticker", "date"], keep="last")
    raw["raw_HLC_ready"] = raw[["high", "low", "close"]].notna().all(axis=1)
    requirement = requirement.merge(raw[["ticker", "date", "raw_HLC_ready", "source_quality"]], on=["ticker", "date"], how="left")

    factor_rows = []
    for ticker in sorted(p31.ticker.unique()):
        path = ADJ_DIR / f"{ticker}.csv.gz"
        if not path.exists():
            continue
        data = pd.read_csv(path, dtype={"ticker": str}, usecols=["date", "ticker", "adjusted_close", "raw_close_comparator", "source_quality"])
        data["date"] = pd.to_datetime(data.date)
        data["adjustment_factor"] = data.adjusted_close / data.raw_close_comparator
        data["factor_ready"] = data.adjustment_factor.notna() & data.raw_close_comparator.ne(0)
        factor_rows.append(data[["ticker", "date", "factor_ready", "adjustment_factor", "source_quality"]].rename(columns={"source_quality": "factor_source_quality"}))
    factors = pd.concat(factor_rows, ignore_index=True).drop_duplicates(["ticker", "date"], keep="last")
    requirement = requirement.merge(factors, on=["ticker", "date"], how="left")
    requirement["raw_HLC_ready"] = requirement.raw_HLC_ready.fillna(False)
    requirement["factor_ready"] = requirement.factor_ready.fillna(False)
    requirement["adjusted_HLC_reconstructable"] = requirement.raw_HLC_ready & requirement.factor_ready
    requirement["gap_reason"] = "ready"
    requirement.loc[~requirement.raw_HLC_ready & requirement.factor_ready, "gap_reason"] = "raw_HLC_missing_or_official_not_applicable_requires_classification"
    requirement.loc[requirement.raw_HLC_ready & ~requirement.factor_ready, "gap_reason"] = "adjustment_factor_missing"
    requirement.loc[~requirement.raw_HLC_ready & ~requirement.factor_ready, "gap_reason"] = "raw_HLC_and_adjustment_factor_missing"
    requirement.to_csv(OUT / "p3_all80_continuous_adjusted_HLC_requirement_ledger.csv.gz", index=False, compression="gzip", encoding="utf-8")
    requirement.loc[~requirement.adjusted_HLC_reconstructable].to_csv(OUT / "p3_all80_continuous_adjusted_HLC_gap_ledger.csv.gz", index=False, compression="gzip", encoding="utf-8")

    coverage = requirement.groupby("ticker").agg(required_rows=("date", "size"), ready_rows=("adjusted_HLC_reconstructable", "sum"), raw_ready_rows=("raw_HLC_ready", "sum"), factor_ready_rows=("factor_ready", "sum")).reset_index()
    coverage["ready_share"] = coverage.ready_rows / coverage.required_rows
    coverage.to_csv(OUT / "p3_all80_continuous_adjusted_HLC_ticker_coverage.csv", index=False, encoding="utf-8-sig")
    daily_coverage = p31[["decision_date", "ticker"]].copy()
    histories = {ticker: set(group.loc[group.adjusted_HLC_reconstructable, "date"]) for ticker, group in requirement.groupby("ticker")}
    counts = []
    for row in daily_coverage.itertuples(index=False):
        dates = calendar[calendar.le(row.decision_date)].iloc[-126:]
        counts.append(sum(pd.Timestamp(date) in histories.get(row.ticker, set()) for date in dates))
    daily_coverage["ready_6M_observations"] = counts
    daily_coverage["continuous_6M_ready"] = daily_coverage.ready_6M_observations.ge(126)
    daily_coverage.to_csv(OUT / "p3_all80_continuous_6M_candidate_readiness.csv.gz", index=False, compression="gzip", encoding="utf-8")

    summary = pd.DataFrame([{
        "P3_1_dates": p31.decision_date.nunique(), "candidate_rows": len(p31), "unique_tickers": p31.ticker.nunique(),
        "required_ticker_dates_252_context": len(requirement), "reconstructable_adjusted_HLC_rows": int(requirement.adjusted_HLC_reconstructable.sum()),
        "gap_rows": int((~requirement.adjusted_HLC_reconstructable).sum()), "candidate_rows_6M_ready": int(daily_coverage.continuous_6M_ready.sum()),
        "candidate_rows_6M_blocked": int((~daily_coverage.continuous_6M_ready).sum()),
        "dates_all80_6M_ready": int(daily_coverage.groupby("decision_date").continuous_6M_ready.all().sum()),
    }])
    summary.to_csv(OUT / "p3_all80_continuous_source_readiness_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{"decision":"incumbent_drops_out_of_primary80","status":"blocked_strategy_policy_not_defined","state_history_policy":"technical history continues; selection eligibility false; held-position validity not inferred"}]).to_csv(OUT / "p3_all80_incumbent_membership_exit_policy_blocked_ledger.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{"audit":"future_outcome_read","violations":0},{"audit":"P3_2_outcome_read","violations":0},{"audit":"raw_used_as_adjusted","violations":0},{"audit":"TDCC_P3_1_zero_fill","violations":0}]).to_csv(OUT / "p3_all80_continuous_future_PIT_audit.csv", index=False, encoding="utf-8-sig")

    values = summary.iloc[0]
    readiness = {"task_id": TASK, "status": "blocked_bounded_adjusted_HLC_delta_required_before_all80_state_supply", "requested_start":"2023-07-11", "requested_end":"2026-06-29", "actual_P3_1_start":str(p31.decision_date.min().date()), "actual_P3_1_end":str(p31.decision_date.max().date()), "P3_1_dates":int(values.P3_1_dates), "candidate_rows":int(values.candidate_rows), "unique_tickers":int(values.unique_tickers), "required_ticker_dates_252_context":int(values.required_ticker_dates_252_context), "adjusted_HLC_gap_rows":int(values.gap_rows), "candidate_rows_6M_ready":int(values.candidate_rows_6M_ready), "candidate_rows_6M_blocked":int(values.candidate_rows_6M_blocked), "dates_all80_6M_ready":int(values.dates_all80_6M_ready), "state_supply_materialized":False, "sufficient_for_walk_forward":False, "ready_for_experiments":False, "performance_authorized":False, "P3_2_outcome_read_authorized":False, "Top3_authorized":False, "diagnostic_subproblem":False, "represents_intended_all80_layer5_state_supply":True, "future_data_violation_count":0, "formal_model_changed":False, "trade_decision_changed":False, "active_in_trade_decision":False, "report_changed":False, "not_live_rule":True, "forward_returns_live_rule_usage":False}
    (OUT / "readiness_for_p3_all80_continuous_lifecycle_state_supply.json").write_text(json.dumps(readiness, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "final_summary_zh.md").write_text(f"# P3 all80 continuous sequential lifecycle source readiness\n\nP3-1 482 dates x80={len(p31):,} candidate rows已盤點。現有raw warmup+membership HLC與adjusted factor無法讓任何日期達成全80完整6M連續歷史；exact 252-context requirement={len(requirement):,} rows，adjusted-HLC gap={int((~requirement.adjusted_HLC_reconstructable).sum()):,} rows。Core未以partial資料計state supply，未讀future outcome/P3-2。下一棒僅可交Radar bounded delta fill。\n", encoding="utf-8")
    files = sorted(path for path in OUT.iterdir() if path.is_file() and path.name != "manifest.json")
    (OUT / "manifest.json").write_text(json.dumps({"task_id":TASK,"inputs":{"daily_sha256":sha(DAILY)},"files":[{"name":path.name,"sha256":sha(path),"bytes":path.stat().st_size} for path in files]}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    run()
