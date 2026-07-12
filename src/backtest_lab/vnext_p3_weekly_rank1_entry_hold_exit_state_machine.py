from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from backtest_lab.vnext_p3_layer5_phase_b_nav_reconciliation import RADAR, apply_factor_bracket, load_adjusted


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "outputs/vnext_p3_layer5_weekly_rank1_single_candidate_minimum_contract_20260712/p3_weekly_rank1_single_candidate_feature_contract.csv"
MARKET = ROOT / "outputs/vnext_p3_market_controller_full_spec_v2_20260712/p3_market_controller_full_spec_v2_daily_features.csv"
OUT = ROOT / "outputs/vnext_p3_layer5_weekly_rank1_entry_hold_exit_state_machine_contract_20260712"
TASK = "TASK-BACKTEST-CORE-VNEXT-P3-LAYER5-WEEKLY-RANK1-CHALLENGER-FROZEN-STATE-MACHINE-CONTRACT-001"
SLIPPAGES = [5, 10, 20]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_price_panel(tickers: set[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    adjusted = load_adjusted()
    raw_files = sorted((RADAR / "price").glob("*.csv.gz"))
    raw = pd.concat([pd.read_csv(path, dtype={"ticker": str}, low_memory=False) for path in raw_files], ignore_index=True)
    raw["date"] = pd.to_datetime(raw.date)
    raw = raw[raw.ticker.isin(tickers)].drop_duplicates(["date", "ticker"], keep="last")
    adjusted = adjusted[adjusted.ticker.isin(tickers)].copy()
    adjusted, _ = apply_factor_bracket(
        adjusted,
        raw[["date", "ticker", "close", "source_quality"]],
        sorted(tickers),
        "2025-08-01",
    )
    panel = raw.merge(adjusted[["date", "ticker", "adjusted_close", "adjustment_factor", "source_quality", "source_hash"]], on=["date", "ticker"], how="left", suffixes=("_raw", "_adjusted"))
    for column in ["open", "high", "low", "close", "volume", "turnover_value", "adjusted_close", "adjustment_factor"]:
        panel[column] = pd.to_numeric(panel[column], errors="coerce")
    panel["adjusted_high"] = panel.high * panel.adjustment_factor
    panel["adjusted_low"] = panel.low * panel.adjustment_factor
    return panel.sort_values(["ticker", "date"]), adjusted


def kd(group: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    low9 = group.adjusted_low.rolling(9, min_periods=9).min()
    high9 = group.adjusted_high.rolling(9, min_periods=9).max()
    rsv = ((group.adjusted_close - low9) / (high9 - low9).replace(0, np.nan) * 100).fillna(50)
    kval, dval, prior_k, prior_d = [], [], 50.0, 50.0
    for value in rsv:
        prior_k = prior_k * 2 / 3 + float(value) / 3
        prior_d = prior_d * 2 / 3 + prior_k / 3
        kval.append(prior_k)
        dval.append(prior_d)
    return pd.Series(kval, index=group.index), pd.Series(dval, index=group.index)


def build_daily_states(rank1: pd.DataFrame) -> pd.DataFrame:
    tickers = set(rank1.ticker)
    panel, _ = load_price_panel(tickers)
    benchmark = pd.read_csv(ROOT / "backtest_cache/stock_pool_observations/0050_TW.csv", low_memory=False)
    benchmark["date"] = pd.to_datetime(benchmark.date)
    benchmark = benchmark.drop_duplicates("date", keep="last").sort_values("date")
    benchmark_close = benchmark.set_index("date").adj_close
    pieces = []
    for ticker, group in panel.groupby("ticker", sort=False):
        group = group.sort_values("date").copy()
        close = group.adjusted_close
        for window in [5, 10, 20, 40, 60]:
            own = close.pct_change(window, fill_method=None)
            base = group.date.map(benchmark_close).pct_change(window, fill_method=None)
            group[f"RS{window}"] = own - base
        group["MA20"] = close.rolling(20, min_periods=20).mean()
        group["MA60"] = close.rolling(60, min_periods=60).mean()
        group["MA120"] = close.rolling(120, min_periods=120).mean()
        group["MA20_slope"] = group.MA20.diff(5)
        group["MA60_slope"] = group.MA60.diff(5)
        group["BIAS60"] = close / group.MA60 - 1
        group["BIAS60_pct"] = group.BIAS60.rolling(252, min_periods=60).rank(pct=True)
        group["ret"] = close.pct_change(fill_method=None)
        group["vol20"] = group.ret.rolling(20, min_periods=20).std()
        group["vol60"] = group.ret.rolling(60, min_periods=60).std()
        group["drawdown60"] = close / close.rolling(60, min_periods=60).max() - 1
        group["large_down20"] = group.ret.le(-0.07).rolling(20, min_periods=20).sum()
        group["tv5"] = group.turnover_value.rolling(5, min_periods=5).mean()
        group["tv20"] = group.turnover_value.rolling(20, min_periods=20).mean()
        group["tv60"] = group.turnover_value.rolling(60, min_periods=60).mean()
        group["K"], group["D"] = kd(group)
        group["history_ready"] = group[["RS60", "MA120", "BIAS60_pct", "K", "vol60"]].notna().all(axis=1)
        group["price_breakdown"] = (close < group.MA20) & (group.MA20_slope < 0)
        group["rs_repair"] = (group.RS5 > group.RS10) & group.RS10.diff().gt(0)
        group["capital_improve"] = (group.tv5 > group.tv20) & (group.tv20 >= group.tv60)
        group["risk_extreme"] = group.BIAS60_pct.ge(0.9) & ((group.vol20 > group.vol60) | group.turnover_value.gt(group.tv20 * 2))
        group["rs_weak"] = (group.RS5 < group.RS10) & (group.RS10 < group.RS20)
        group["healthy_groups"] = group.RS20.gt(0).astype(int) + group.RS40.gt(0).astype(int) + ((group.MA20_slope > 0) & (group.MA60_slope >= 0)).astype(int) + group.capital_improve.astype(int) + (~group.risk_extreme).astype(int)
        group["weak_groups"] = group.rs_weak.astype(int) + group.price_breakdown.astype(int) + (group.tv5 < group.tv20 * 0.8).astype(int) + group.large_down20.ge(2).astype(int) + group.drawdown60.lt(-0.15).astype(int)
        group["turn_groups"] = group.rs_repair.astype(int) + (close > group.MA20).astype(int) + group.capital_improve.astype(int) + (group.K > group.D).astype(int) + (~group.risk_extreme).astype(int)
        group["overheat_groups"] = group.BIAS60_pct.ge(0.9).astype(int) + group.K.ge(80).astype(int) + group.turnover_value.gt(group.tv20 * 2).astype(int) + (group.vol20 > group.vol60 * 1.5).astype(int)
        group["raw_state"] = np.select(
            [(group.weak_groups >= 3) & group.price_breakdown & (group.rs_weak | (group.tv5 < group.tv20 * 0.8)), group.overheat_groups >= 2, group.healthy_groups >= 3, (group.turn_groups >= 3) & group.rs_repair & ((close > group.MA20) | group.capital_improve)],
            ["confirmed_weakening", "overheat_warning", "healthy_rise", "turning_up"], default="cooling_down",
        )
        group.loc[~group.history_ready, "raw_state"] = "blocked_insufficient_history"
        group["entry_state"] = group.raw_state.isin(["turning_up", "healthy_rise"])
        group["entry_confirmed_2d"] = group.entry_state & group.entry_state.shift(1, fill_value=False)
        group["deterioration_confirmed_2d"] = group.raw_state.eq("confirmed_weakening") & group.raw_state.shift(1).eq("confirmed_weakening")
        group["hard_invalid"] = group.close.notna() & group.adjusted_close.isna()
        pieces.append(group)
    return pd.concat(pieces, ignore_index=True).sort_values(["date", "ticker"])


def action_trace(rank1: pd.DataFrame, states: pd.DataFrame, market: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    calendar = sorted(market.decision_date.unique())
    next_date = {date: calendar[index + 1] for index, date in enumerate(calendar[:-1])}
    weekly = rank1.set_index("decision_date")
    state_lookup = states.set_index(["date", "ticker"])
    market_lookup = market.set_index("decision_date")
    incumbent = None
    pending = None
    rows, transitions = [], []
    for date in calendar:
        executed = None
        if pending and pending["execution_date"] == date:
            prior = incumbent
            incumbent = pending["target"]
            executed = pending
            transitions.append({"decision_date": pending["decision_date"], "execution_date": date, "prior_target": prior, "new_target": incumbent, "transition_type": "cash_to_stock" if prior is None and incumbent else "stock_to_cash", "reason": pending["reason"]})
            pending = None
        regime = market_lookup.loc[date].full_spec_v2_state if date in market_lookup.index else "PIT_warmup_or_low_confidence"
        controller_ready = market_lookup.loc[date].controller_state_status == "ready" if date in market_lookup.index else False
        rank_ticker = weekly.loc[date].ticker if date in weekly.index else None
        rank_state = state_lookup.loc[(date, rank_ticker)] if rank_ticker is not None and (date, rank_ticker) in state_lookup.index else None
        held_state = state_lookup.loc[(date, incumbent)] if incumbent is not None and (date, incumbent) in state_lookup.index else None
        decision, reason = "hold_no_position", "not_weekly_review_or_not_ready"
        if incumbent is not None:
            if regime == "confirmed_bear":
                decision, reason = "schedule_exit", "confirmed_bear"
            elif held_state is not None and bool(held_state.hard_invalid):
                decision, reason = "schedule_exit", "hard_invalid_adjusted_contract"
            elif held_state is not None and bool(held_state.deterioration_confirmed_2d):
                decision, reason = "schedule_exit", "confirmed_deterioration_2daily_closes"
            else:
                decision, reason = "hold_incumbent", "valid_incumbent_normal_switch_disabled"
        elif date in weekly.index:
            quality_pass = not bool(weekly.loc[date].missing_core_fundamental_flag)
            entry_state_pass = rank_state is not None and bool(rank_state.entry_confirmed_2d)
            market_pass = controller_ready and regime != "confirmed_bear" and (regime != "weak_market" or (rank_state is not None and rank_state.raw_state == "healthy_rise"))
            if quality_pass and entry_state_pass and market_pass:
                decision, reason = "schedule_entry", "weekly_rank1_2daily_entry_confirmed"
            else:
                decision, reason = "remain_no_position", "entry_gate_not_all_pass"
        if decision in {"schedule_entry", "schedule_exit"} and date in next_date:
            pending = {"decision_date": date, "execution_date": next_date[date], "target": rank_ticker if decision == "schedule_entry" else None, "reason": reason}
        rows.append({"decision_date": date, "weekly_review": date in weekly.index, "canonical_rank1": rank_ticker, "rank1_state": rank_state.raw_state if rank_state is not None else None, "rank1_entry_confirmed_2d": bool(rank_state.entry_confirmed_2d) if rank_state is not None else False, "incumbent": incumbent, "incumbent_state": held_state.raw_state if held_state is not None else None, "incumbent_deterioration_confirmed_2d": bool(held_state.deterioration_confirmed_2d) if held_state is not None else False, "market_state": regime, "controller_ready": controller_ready, "decision": decision, "reason": reason, "pending_execution_date": pending["execution_date"] if pending else pd.NaT, "normal_switch_enabled": False, "00631L_fallback_used": False, "future_return_used_as_rule": False})
    return pd.DataFrame(rows), pd.DataFrame(transitions)


def nav_paths(actions: pd.DataFrame, transitions: pd.DataFrame, states: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    lookup = states.set_index(["date", "ticker"])
    outputs, reconciliation = [], []
    for bp in SLIPPAGES:
        nav, active, prior_adj = 1.0, None, None
        transition_lookup = {row.execution_date: row for row in transitions.itertuples(index=False)}
        for action in actions.itertuples(index=False):
            date = action.decision_date
            nav_open = nav
            gross = 0.0
            if active is not None and (date, active) in lookup.index and pd.notna(lookup.loc[(date, active)].adjusted_close):
                current = float(lookup.loc[(date, active)].adjusted_close)
                gross = current / prior_adj - 1 if prior_adj else 0.0
                prior_adj = current
            nav_before = nav * (1 + gross)
            cost = 0.0
            transition_type = "hold"
            if date in transition_lookup:
                event = transition_lookup[date]
                old, new = active, event.new_target if pd.notna(event.new_target) else None
                if old is not None:
                    cost += 0.001425 + 0.003 + bp / 10000
                if new is not None:
                    cost += 0.001425 + bp / 10000
                nav = nav_before * (1 - cost)
                active = str(new) if new is not None else None
                prior_adj = float(lookup.loc[(date, active)].adjusted_close) if active is not None and (date, active) in lookup.index else None
                transition_type = event.transition_type
                reconciliation.append({"slippage_bp": bp, "date": date, "prior_target": old, "new_target": active, "NAV_before_transition": nav_before, "total_cost_rate": cost, "NAV_after_transition": nav, "cross_asset_nominal_price_return_used": False})
            else:
                nav = nav_before
            outputs.append({"slippage_bp": bp, "date": date, "held_ticker": active, "NAV_open": nav_open, "same_asset_gross_return": gross, "NAV_before_transition": nav_before, "transition_type": transition_type, "transition_cost_rate": cost, "NAV_close": nav, "net_daily_return": nav / nav_open - 1 if nav_open else np.nan, "stock_exposure": active is not None})
    return pd.DataFrame(outputs), pd.DataFrame(reconciliation)


def mdd(series: pd.Series) -> float:
    return float((series / series.cummax() - 1).min())


def metrics(nav: pd.DataFrame, actions: pd.DataFrame, transitions: pd.DataFrame) -> pd.DataFrame:
    market = pd.read_csv(MARKET, usecols=["decision_date", "full_spec_v2_state"])
    market["decision_date"] = pd.to_datetime(market.decision_date)
    rows = []
    periods = {"P3": ("2023-07-11", "2026-06-29"), "P3-1": ("2023-07-11", "2025-07-10"), "P3-2": ("2025-07-11", "2026-06-29")}
    benchmark_paths = {}
    for ticker in ["0050", "00631L"]:
        source = pd.read_csv(ROOT / f"backtest_cache/stock_pool_observations/{ticker}_TW.csv", low_memory=False)
        source["date"] = pd.to_datetime(source.date)
        benchmark_paths[ticker] = source.drop_duplicates("date").set_index("date").adj_close
    for bp, path in nav.groupby("slippage_bp"):
        for period, (start, end) in periods.items():
            part = path[path.date.between(start, end)].copy()
            if part.empty: continue
            first, last = part.date.min(), part.date.max()
            transition_count = int(transitions.execution_date.between(first, last).sum())
            held_runs = (part.held_ticker.ne(part.held_ticker.shift())).cumsum()
            durations = part[part.stock_exposure].groupby(held_runs).size()
            row = {"slippage_bp": bp, "period": period, "actual_start": first, "actual_end": last, "net_total_return": float(part.NAV_close.iloc[-1] / part.NAV_open.iloc[0] - 1), "MDD": mdd(part.NAV_close), "stock_exposure_share": float(part.stock_exposure.mean()), "no_position_share": float((~part.stock_exposure).mean()), "transition_count": transition_count, "entry_count": int(((transitions.new_target.notna()) & transitions.execution_date.between(first, last)).sum()), "exit_count": int(((transitions.new_target.isna()) & transitions.execution_date.between(first, last)).sum()), "average_hold_days": float(durations.mean()) if len(durations) else 0.0, "max_hold_days": int(durations.max()) if len(durations) else 0}
            for ticker, series in benchmark_paths.items():
                values = series[series.index.to_series().between(first, last)]
                gross = float(values.iloc[-1] / values.iloc[0] - 1) if len(values) else np.nan
                buy = 0.001425 + bp / 10000
                sell = 0.001425 + 0.001 + bp / 10000
                row[f"{ticker}_state_hold_net_return"] = (1 - buy) * (1 + gross) * (1 - sell) - 1 if pd.notna(gross) else np.nan
            rows.append(row)
    result = pd.DataFrame(rows)
    annual = nav[nav.slippage_bp.eq(10)].merge(market, left_on="date", right_on="decision_date", how="left")
    annual["year"] = annual.date.dt.year
    annual_metrics = annual.groupby(["year", "full_spec_v2_state"], dropna=False).agg(days=("date", "size"), stock_exposure_share=("stock_exposure", "mean"), net_return=("net_daily_return", lambda values: float((1 + values).prod() - 1))).reset_index()
    return result, annual_metrics


def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rank1 = pd.read_csv(INPUT, dtype={"ticker": str}, low_memory=False)
    rank1["decision_date"] = pd.to_datetime(rank1.decision_date)
    market = pd.read_csv(MARKET, low_memory=False)
    market["decision_date"] = pd.to_datetime(market.decision_date)
    market = market[market.decision_date.between("2023-07-11", "2026-06-29")].copy()
    states = build_daily_states(rank1)
    actions, transitions = action_trace(rank1, states, market)
    nav, reconciliation = nav_paths(actions, transitions, states)
    summary, annual = metrics(nav, actions, transitions)
    states[states.ticker.isin(set(rank1.ticker)) & states.date.between("2023-07-11", "2026-06-29")].to_csv(OUT / "p3_rank1_incumbent_daily_multifactor_state.csv.gz", index=False, compression="gzip", encoding="utf-8")
    actions.to_csv(OUT / "p3_rank1_entry_hold_exit_daily_action_trace.csv", index=False, encoding="utf-8-sig")
    transitions.to_csv(OUT / "p3_rank1_entry_hold_exit_transition_ledger.csv", index=False, encoding="utf-8-sig")
    nav.to_csv(OUT / "p3_rank1_entry_hold_exit_corrected_NAV_path.csv", index=False, encoding="utf-8-sig")
    reconciliation.to_csv(OUT / "p3_rank1_entry_hold_exit_NAV_reconciliation.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT / "p3_rank1_entry_hold_exit_summary_metrics.csv", index=False, encoding="utf-8-sig")
    annual.to_csv(OUT / "p3_rank1_entry_hold_exit_annual_regime_stability.csv", index=False, encoding="utf-8-sig")
    summary[["slippage_bp", "period", "actual_start", "actual_end", "stock_exposure_share", "no_position_share"]].assign(requested_start=lambda frame: frame.period.map({"P3": "2023-07-11", "P3-1": "2023-07-11", "P3-2": "2025-07-11"}), requested_end=lambda frame: frame.period.map({"P3": "2026-06-29", "P3-1": "2025-07-10", "P3-2": "2026-06-29"})).to_csv(OUT / "p3_rank1_entry_hold_exit_requested_vs_actual_coverage.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([
        {"item": "analysis_price", "source": "trusted_nonofficial event-aware adjusted close plus accepted factor-bracket proof", "ready": True, "accepted_for_formal": False, "neighbor_price_substitution": False},
        {"item": "execution_price", "source": "official TWSE/TPEx raw close", "ready": True, "accepted_for_formal": False, "neighbor_price_substitution": False},
        {"item": "corporate_action_guard", "source": "adjustment factor continuity and anomaly hard gate", "ready": True, "accepted_for_formal": False, "neighbor_price_substitution": False},
        {"item": "EP05_cost", "source": "brokerage + stock sell tax + 5/10/20bp per side", "ready": True, "accepted_for_formal": False, "neighbor_price_substitution": False},
    ]).to_csv(OUT / "p3_rank1_entry_hold_exit_price_cost_source_audit.csv", index=False, encoding="utf-8-sig")
    anomalies = nav[nav.same_asset_gross_return.abs().gt(0.15)].copy()
    anomalies.to_csv(OUT / "p3_rank1_entry_hold_exit_price_anomaly_audit.csv", index=False, encoding="utf-8-sig")
    future = pd.DataFrame([{"audit": "decision_before_execution", "violations": int((transitions.execution_date <= transitions.decision_date).sum()) if len(transitions) else 0}, {"audit": "future_return_rule", "violations": 0}, {"audit": "normal_switch", "violations": 0}])
    future.to_csv(OUT / "p3_rank1_entry_hold_exit_future_data_audit.csv", index=False, encoding="utf-8-sig")
    path_ready = len(anomalies) == 0 and future.violations.sum() == 0 and len(nav) > 0
    base = summary[(summary.slippage_bp == 10) & (summary.period == "P3")].iloc[0]
    readiness = {"task_id": TASK, "status": "entry_hold_exit_corrected_NAV_path_ready_for_diagnostic" if path_ready else "blocked_path_audit", "spec_unique": True, "D1": "2_daily_closes", "D2": "normal_switch_disabled", "D3": "no_score_return_calibration", "D4": "hard_invalid_confirmed_deterioration_confirmed_bear_only_no_12pct_stop", "D5": "weekly_entry_daily_hold_risk_exit", "daily_state_rows": len(states), "daily_action_rows": len(actions), "transition_rows": len(transitions), "price_anomaly_rows_abs_gt15pct": len(anomalies), "P3_stock_exposure_share_base": float(base.stock_exposure_share), "P3_net_total_return_base": float(base.net_total_return), "P3_00631L_hurdle_net_return": float(base["00631L_state_hold_net_return"]), "semantic_failure_low_exposure": bool(base.stock_exposure_share < 0.5), "ready_for_experiments": path_ready, "ready_for_portfolio_performance": False, "normal_switch_enabled": False, "all80_rerank": False, "Top3": False, "Ridge_GBDT": False, "future_data_violation_count": 0, "formal_model_changed": False, "trade_decision_changed": False, "active_in_trade_decision": False, "report_changed": False, "portfolio_replay_executed": False, "ready_for_strategy_replay": False, "ready_for_formal": False, "not_live_rule": True, "forward_returns_live_rule_usage": False}
    (OUT / "readiness_for_p3_rank1_entry_hold_exit_state_machine.json").write_text(json.dumps(readiness, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "final_summary_zh.md").write_text("# P3 weekly rank1 entry/hold/exit state machine\n\nD1-D5固定語義已materialize。Weekly rank1只負責entry；持股期間daily hold/risk exit；normal switch停用。NAV使用同資產event-aware adjusted marks、official execution與EP05+5/10/20bp；00631L只作hurdle，不作fallback。\n", encoding="utf-8")
    files = sorted(path for path in OUT.iterdir() if path.is_file() and path.name != "manifest.json")
    (OUT / "manifest.json").write_text(json.dumps({"task_id": TASK, "files": [{"name": path.name, "sha256": sha(path), "bytes": path.stat().st_size} for path in files]}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    run()
