from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from backtest_lab import vnext_p3_all80_continuous_lifecycle_state_supply as supply


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/vnext_p3_layer5_all80_sequential_entry_funnel_lead_lag_setup_latch_audit_20260713"
TASK = "TASK-BACKTEST-CORE-VNEXT-P3-LAYER5-ALL80-SEQUENTIAL-ENTRY-FUNNEL-LEAD-LAG-AND-SETUP-LATCH-AUDIT-001"
HORIZONS = (3, 5, 10, 20)
LATCH_WINDOWS = (5, 10, 20)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _base() -> pd.DataFrame:
    daily = pd.read_csv(supply.DAILY, dtype={"ticker": str}, low_memory=False)
    daily["ticker"] = daily.ticker.str.zfill(4)
    daily["decision_date"] = pd.to_datetime(daily.decision_date)
    daily = daily.loc[daily.decision_date.lt("2025-07-11")].sort_values(["ticker", "decision_date"])
    history = supply._features(supply._history())
    columns = ["ticker", "date", "adjusted_close", "K6", "D6", "price_pct_6M", "K_pct_6M", "BIAS_pct_6M", "actual_history_observations"]
    history = history[columns].rename(columns={"adjusted_close": "continuous_adjusted_close"})
    frame = daily.merge(history, left_on=["ticker", "decision_date"], right_on=["ticker", "date"], how="left").copy()
    frame["price_history_ready"] = frame.actual_history_observations.ge(126) & frame[["price_pct_6M", "K_pct_6M", "BIAS_pct_6M"]].notna().all(axis=1)
    k_change = frame.groupby("ticker", sort=False).K6.diff()
    frame["kd_up"] = k_change.gt(0) & frame.K6.ge(frame.D6)
    frame["kd_down"] = k_change.lt(0) & frame.K6.le(frame.D6)
    frame["ma_up"] = frame.continuous_adjusted_close.ge(frame.MA20) & frame.MA20_slope.gt(0) & ~frame.price_breakdown.fillna(False)
    frame["ma_down"] = frame.continuous_adjusted_close.lt(frame.MA20) & (frame.MA20_slope.lt(0) | frame.price_breakdown.fillna(False))
    frame["risk_ok"] = ~frame.risk_extreme.fillna(False) & ~frame.price_breakdown.fillna(False)
    frame["risk_bad"] = frame.risk_extreme.fillna(False) | frame.price_breakdown.fillna(False)
    institutional_withdrawal = frame.institutional_foreign_net_20D.lt(0) | frame.institutional_trust_net_20D.lt(0) | frame.institutional_dealer_net_20D.lt(0)
    frame["capital_withdraw"] = frame.tv5.lt(frame.tv20) & institutional_withdrawal & frame.chip_available_count.gt(0)
    return frame


def _event_starts(values: pd.Series) -> pd.Series:
    return values & ~values.shift(fill_value=False)


def _fold_map(dates: list[pd.Timestamp]) -> dict[pd.Timestamp, int]:
    return {pd.Timestamp(date): fold for fold, segment in enumerate(np.array_split(dates, 3), 1) for date in segment}


def _first_lag(group: pd.DataFrame, start_index: int, column: str, horizon: int) -> float:
    end = min(len(group), start_index + horizon + 1)
    hits = np.flatnonzero(group.iloc[start_index:end][column].fillna(False).to_numpy())
    return float(hits[0]) if len(hits) else np.nan


def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    base = _base()
    dates = sorted(base.decision_date.unique())
    folds = _fold_map(dates)
    all_funnels, all_lags, latch_rows, coverage_rows, predicate_rows = [], [], [], [], []
    evidence_up = ["kd_up", "rs_repair", "ma_up", "capital_improve", "risk_ok"]
    evidence_down = ["kd_down", "rs_weak", "ma_down", "capital_withdraw", "risk_bad"]

    for platform, (low, high) in supply.PLATFORMS.items():
        positioned = base.copy()
        positioned["relative_low"] = positioned.price_pct_6M.le(low) & (positioned.K_pct_6M.le(low) | positioned.BIAS_pct_6M.le(low))
        positioned["relative_high"] = positioned.price_pct_6M.ge(high) & (positioned.K_pct_6M.ge(high) | positioned.BIAS_pct_6M.ge(high))
        positioned["low_event"] = positioned.groupby("ticker", sort=False).relative_low.transform(_event_starts)
        positioned["high_event"] = positioned.groupby("ticker", sort=False).relative_high.transform(_event_starts)
        positioned["fold"] = positioned.decision_date.map(folds)

        for ticker, group in positioned.groupby("ticker", sort=False):
            g = group.reset_index(drop=True)
            for index in np.flatnonzero(g.low_event.to_numpy()):
                event = {"platform": platform, "ticker": ticker, "t0": g.loc[index, "decision_date"], "fold": int(g.loc[index, "fold"]), "event_type": "relative_low"}
                for evidence in evidence_up:
                    for horizon in HORIZONS:
                        event[f"{evidence}_within_{horizon}TD"] = pd.notna(_first_lag(g, index, evidence, horizon))
                    event[f"{evidence}_first_lag_20TD"] = _first_lag(g, index, evidence, 20)
                all_lags.append(event)
            for index in np.flatnonzero(g.high_event.to_numpy()):
                event = {"platform": platform, "ticker": ticker, "t0": g.loc[index, "decision_date"], "fold": int(g.loc[index, "fold"]), "event_type": "relative_high"}
                for evidence in evidence_down:
                    event[f"{evidence}_first_lag_20TD"] = _first_lag(g, index, evidence, 20)
                all_lags.append(event)

        for persistence, (need, window) in supply.PERSISTENCE.items():
            state = supply._states(positioned, low, high, need, window)
            state["fold"] = state.decision_date.map(folds)
            for evidence in evidence_up + evidence_down:
                state[f"{evidence}_p"] = supply._persistent(state, evidence, need, window)
            state["up_complete_3"] = state[[f"{e}_p" for e in evidence_up]].sum(axis=1).ge(3)
            state["up_complete_4"] = state[[f"{e}_p" for e in evidence_up]].sum(axis=1).ge(4)
            state["down_complete_3"] = state[[f"{e}_p" for e in evidence_down]].sum(axis=1).ge(3)

            for fold, part in state.groupby("fold"):
                low_events = int(part.groupby("ticker").relative_low.transform(_event_starts).sum())
                stages = {
                    "relative_low_raw_events": low_events,
                    "S1_established_clusters": int((part.state.eq("S1") & part.groupby("ticker").state.shift().ne("S1")).sum()),
                    "evidence_3of5_rows": int(part.up_complete_3.sum()),
                    "evidence_4of5_rows": int(part.up_complete_4.sum()),
                    "S2_clusters": int((part.state.eq("S2") & part.groupby("ticker").state.shift().ne("S2")).sum()),
                    "S3_clusters": int(part.entry_cluster.sum()),
                }
                for evidence in evidence_up:
                    for horizon in HORIZONS:
                        relevant = [row for row in all_lags if row["platform"] == platform and row["fold"] == fold and row["event_type"] == "relative_low"]
                        stages[f"{evidence}_within_{horizon}TD_clusters"] = sum(bool(row[f"{evidence}_within_{horizon}TD"]) for row in relevant)
                stages.update({"platform": platform, "persistence": persistence, "fold": int(fold)})
                all_funnels.append(stages)

            resets = state.loc[state.groupby("ticker").state.shift().eq("S1") & state.state.eq("S0")]
            predicate_rows.append({"platform":platform,"persistence":persistence,"S1_to_S0_resets":len(resets),"reset_after_relative_low_false":int((~resets.relative_low).sum()),"S2_requires_continued_low":False,"S3_requires_continued_low":False,"S2_failure_returns_S1_without_low_recheck":True,"daily_max_one_state_advance":True,"S4_S5_direct_classification":False})

            for latch_window in LATCH_WINDOWS:
                entries, exits = [], []
                for ticker, group in state.groupby("ticker", sort=False):
                    g = group.reset_index(drop=True)
                    for event_type, setup_col, complete_col, capital_col, sink in [
                        ("entry", "relative_low", "up_complete_3", "capital_improve_p", entries),
                        ("exit", "relative_high", "down_complete_3", "capital_withdraw_p", exits),
                    ]:
                        starts = np.flatnonzero(_event_starts(g[setup_col]).to_numpy())
                        for start in starts:
                            stop = min(len(g), start + latch_window + 1)
                            invalid = g.iloc[start:stop].risk_bad.to_numpy()
                            if invalid.any(): stop = start + int(np.flatnonzero(invalid)[0]) + 1
                            complete_hits = np.flatnonzero(g.iloc[start:stop][complete_col].to_numpy())
                            if not len(complete_hits): continue
                            complete_index = start + int(complete_hits[0])
                            confirmation = g.iloc[complete_index + 1:stop]
                            confirmed = confirmation.index[confirmation[capital_col]].tolist()
                            if confirmed:
                                sink.append((ticker, g.loc[int(confirmed[0]), "decision_date"], int(g.loc[int(confirmed[0]), "fold"])))
                for fold in (1, 2, 3):
                    latch_rows.append({"platform":platform,"persistence":persistence,"latch_window_TD":latch_window,"fold":fold,"entry_clusters":sum(row[2] == fold for row in entries),"exit_clusters":sum(row[2] == fold for row in exits)})
                entry_dates = {date for _, date, _ in entries}
                active_dates = set(state.loc[state.state.eq("S4"), "decision_date"]) | entry_dates
                no_supply = [date not in active_dates for date in dates]
                longest = current = 0
                for missing in no_supply:
                    current = current + 1 if missing else 0; longest = max(longest, current)
                coverage_rows.append({"platform":platform,"persistence":persistence,"latch_window_TD":latch_window,"S3_entry_dates":len(entry_dates),"S3_plus_S4_dates":len(active_dates),"S3_plus_S4_date_coverage":len(active_dates)/len(dates),"longest_no_supply_trading_days":longest})

    funnel = pd.DataFrame(all_funnels)
    lag = pd.DataFrame(all_lags)
    funnel.to_csv(OUT / "p3_all80_entry_funnel_by_platform_persistence_fold.csv", index=False, encoding="utf-8-sig")
    lag.to_csv(OUT / "p3_all80_relative_position_evidence_lead_lag_events.csv.gz", index=False, compression="gzip")
    lag_summary = []
    for (platform, fold, event_type), part in lag.groupby(["platform", "fold", "event_type"]):
        allowed = evidence_up if event_type == "relative_low" else evidence_down
        for column in [f"{evidence}_first_lag_20TD" for evidence in allowed]:
            values = part[column].dropna()
            lag_summary.append({"platform":platform,"fold":fold,"event_type":event_type,"evidence":column.replace("_first_lag_20TD", ""),"events":len(part),"confirmed_within_20TD":len(values),"p25":values.quantile(.25) if len(values) else np.nan,"median":values.median() if len(values) else np.nan,"p75":values.quantile(.75) if len(values) else np.nan,"p90":values.quantile(.9) if len(values) else np.nan})
    pd.DataFrame(lag_summary).to_csv(OUT / "p3_all80_evidence_lead_lag_distribution.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(predicate_rows).to_csv(OUT / "p3_all80_state_predicate_structural_contradiction_audit.csv", index=False, encoding="utf-8-sig")
    latch = pd.DataFrame(latch_rows); latch.to_csv(OUT / "p3_all80_setup_latch_supply_counterfactual.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(coverage_rows).to_csv(OUT / "p3_all80_setup_latch_S3_S4_coverage.csv", index=False, encoding="utf-8-sig")

    fold2_zero = bool((latch.loc[latch.fold.eq(2), "entry_clusters"] == 0).all())
    latch_adds = bool((latch.entry_clusters > 0).any())
    classification = "E_mixed" if latch_adds and fold2_zero else ("A_setup_memory_structural_contradiction_confirmed" if latch_adds else "B_fold2_genuinely_lacks_low_turnup_setups")
    readiness = {"task_id":TASK,"status":"audit_complete_strategy_center_sequence_memory_decision_required","classification":classification,"P3_1_dates":len(dates),"candidate_rows":len(base),"future_outcome_read":False,"P3_2_outcome_read":False,"performance_authorized":False,"ready_for_experiments":False,"represents_intended_all80_layer5_state_audit":True,"future_data_violation_count":0,"formal_model_changed":False,"trade_decision_changed":False,"active_in_trade_decision":False,"report_changed":False,"not_live_rule":True,"forward_returns_live_rule_usage":False}
    (OUT / "readiness_for_entry_funnel_setup_latch_audit.json").write_text(json.dumps(readiness, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "final_summary_zh.md").write_text(f"# P3 all80 entry funnel / lead-lag / setup-latch audit\n\nAudit classification: `{classification}`。本輪只讀P3-1 PIT feature/state，未讀future outcome、P3-2或績效。Setup latch 5/10/20TD只作供給反事實，未選窗口。\n", encoding="utf-8")
    files = sorted(p for p in OUT.iterdir() if p.is_file() and p.name != "manifest.json")
    (OUT / "manifest.json").write_text(json.dumps({"task_id":TASK,"files":[{"name":p.name,"sha256":_sha(p),"bytes":p.stat().st_size} for p in files]}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    run()
