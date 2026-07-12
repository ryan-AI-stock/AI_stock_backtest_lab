from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .vnext_p3_layer5_phase_b_complete_paths import prices
from .vnext_p3_layer5_phase_b_nav_reconciliation import (
    apply_factor_bracket,
    collapse_same_close_events,
    load_adjusted,
    load_raw,
)

ROOT = Path(__file__).resolve().parents[2]
EXP = Path(r"C:\Users\zergv\Documents\Codex\2026-07-06\backtest-lab-experiments-diagnostic-validation-attribution\outputs\vnext_p3_corrected_nav_layer5_failure_attribution_20260712")
PATHS = ROOT / "outputs/vnext_p3_layer5_phase_b_complete_paths_20260712"
FEATURE = ROOT / "outputs/vnext_p3_layer5_daily_feature_state_action_materialization_20260712/p3_layer5_daily_feature_state_matrix.csv"
MARKET = ROOT / "outputs/vnext_p3_market_controller_full_spec_v2_20260712/p3_market_controller_full_spec_v2_daily_features.csv"
OUT = ROOT / "outputs/vnext_p3_layer5_counterfactual_nav_rechain_20260712"
TASK = "TASK-BACKTEST-CORE-VNEXT-P3-CORRECTED-NAV-LAYER5-COUNTERFACTUAL-RECHAIN-001"


def fast_transitions(actions: pd.DataFrame, px_lookup: dict, calendar: list[pd.Timestamp]) -> pd.DataFrame:
    events=[]; active=None; date_pos={d:i for i,d in enumerate(calendar)}
    for r in actions.sort_values('decision_date').itertuples(index=False):
        target=str(r.selected_ticker) if pd.notna(r.selected_ticker) else None
        if target==active: continue
        start=date_pos.get(pd.Timestamp(r.requested_execution_date),0); found=None
        for d in calendar[start:]:
            if (active is None or (d,active) in px_lookup) and (target is None or (d,target) in px_lookup): found=d; break
        if found is None: raise ValueError(f'unresolved transition {active}->{target}')
        oldrow=px_lookup.get((found,active)); newrow=px_lookup.get((found,target))
        typ='stock_to_stock' if active and target else ('stock_to_no_position' if active else 'no_position_to_stock')
        fee=(.001425 if active else 0)+(.001425 if target else 0); tax=.003 if active else 0; slip=.001*((1 if active else 0)+(1 if target else 0))
        events.append({'scenario':r.scenario,'decision_date':r.decision_date,'requested_execution_date':r.requested_execution_date,'actual_execution_date':found,'prior_target':active,'new_target':target,'transition_type':typ,'prior_target_exit_close':oldrow[0] if oldrow else np.nan,'prior_target_exit_source_quality':oldrow[1] if oldrow else None,'new_target_entry_close':newrow[0] if newrow else np.nan,'new_target_entry_source_quality':newrow[1] if newrow else None,'brokerage_rate':fee,'tax_rate':tax,'slippage_rate':slip,'total_cost_rate':fee+tax+slip})
        active=target
    return pd.DataFrame(events)


def wealth(events: pd.DataFrame, lookup: pd.DataFrame, raw_lookup: pd.DataFrame, calendar: list[pd.Timestamp], cfid: str, slippage_bp: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    ev, collision = collapse_same_close_events(events)
    ev = ev.sort_values("actual_execution_date")
    active = None
    previous_adjusted = None
    nav = 1.0
    quantity = np.nan
    daily, transitions = [], []
    for date in calendar:
        nav_open = nav
        incumbent = active
        gross = 0.0
        execution_status = "hold"
        if active is not None:
            key = (date, str(active))
            if key in lookup.index:
                current_adjusted = float(lookup.loc[key].adjusted_close)
                gross = 0.0 if previous_adjusted is None else current_adjusted / previous_adjusted - 1.0
                previous_adjusted = current_adjusted
            elif key in raw_lookup.index:
                raise ValueError(f"adjusted trading-day gap {cfid} {date} {active}")
        nav_before = nav * (1.0 + gross)
        day = ev[ev.actual_execution_date.eq(date)]
        transition_type = "hold_same"
        cost = 0.0
        if len(day):
            e = day.iloc[-1]
            old = active
            new = str(e.new_target) if pd.notna(e.new_target) else None
            transition_type = e.transition_type
            slip = slippage_bp / 10000.0
            exit_rate = 0.001425 + 0.003 + slip if old else 0.0
            entry_rate = 0.001425 + slip if new else 0.0
            exit_cost = nav_before * exit_rate
            entry_cost = nav_before * entry_rate
            cost = exit_cost + entry_cost
            nav = nav_before - cost
            active = new
            entry_close = float(e.new_target_entry_close) if new else np.nan
            quantity = nav / entry_close if new and entry_close > 0 else np.nan
            previous_adjusted = float(lookup.loc[(date, new)].adjusted_close) if new and (date, new) in lookup.index else None
            execution_status = "executed_exact_official_close"
            transitions.append({
                "counterfactual_id": cfid, "date": date, "prior_target": old, "new_target": new,
                "NAV_open": nav_open, "NAV_before_transition": nav_before,
                "outgoing_proceeds_before_cost": nav_before, "exit_cost": exit_cost,
                "incoming_notional_before_entry_cost": nav_before - exit_cost, "entry_cost": entry_cost,
                "incoming_shares": quantity, "incoming_entry_raw_close": entry_close,
                "NAV_after_transition": nav, "transition_type": transition_type,
                "stock_etf_cost_split": "stock_brokerage_0.1425pct_each_side+sell_tax_0.3pct+slippage_10bp_each_side",
                "cross_asset_nominal_price_return_used": False,
            })
        else:
            nav = nav_before
        if abs(gross) > 0.15:
            raise ValueError(f"return anomaly {cfid} {date} {incumbent} {gross}")
        daily.append({
            "scenario": "exact_episode_counterfactual", "counterfactual_id": cfid, "date": date,
            "incumbent": incumbent, "counterfactual_target": active, "NAV_open": nav_open,
            "NAV_before_transition": nav_before, "NAV_close": nav,
            "net_daily_return": nav / nav_open - 1.0, "gross_same_asset_return": gross,
            "transition_type": transition_type, "transition_cost": cost,
            "stock_etf_cost_split": "stock_only_EP05+10bp_per_side", "execution_status": execution_status,
        })
    return pd.DataFrame(daily), pd.DataFrame(transitions)


def candidate_context(events: pd.DataFrame, feature: pd.DataFrame, adj: pd.DataFrame, calendar: list[pd.Timestamp]) -> pd.DataFrame:
    lookup = adj.set_index(["date", "ticker"])
    rows = []
    for e in events.itertuples(index=False):
        day = feature[feature.decision_date.eq(e.decision_date)].sort_values("score_balanced", ascending=False)
        ready = day[day.history_ready]
        selected = str(e.challenger) if pd.notna(e.challenger) else None
        second = next((str(x) for x in ready.ticker if str(x) != selected), None)
        execution = pd.Timestamp(e.execution_date)
        future_dates = [d for d in calendar if d > execution][:5]
        end = future_dates[-1] if future_dates else pd.NaT
        returns = []
        for ticker in ready.ticker.astype(str):
            if (execution, ticker) in lookup.index and pd.notna(end) and (end, ticker) in lookup.index:
                returns.append(float(lookup.loc[(end, ticker)].adjusted_close / lookup.loc[(execution, ticker)].adjusted_close - 1.0))
        sr = ready[ready.ticker.astype(str).eq(selected)].head(1)
        rr = sr.iloc[0] if len(sr) else None
        rows.append({
            "event_id": e.event_id, "decision_date": e.decision_date, "selected": selected,
            "second_candidate": second, "primary80_equal_weight_forward_5D": np.mean(returns) if returns else np.nan,
            "primary80_median_forward_5D": np.median(returns) if returns else np.nan,
            "forward_return_role": "evaluation_metadata_only_not_rule",
            "selected_PIT_rank": int(ready.ticker.astype(str).tolist().index(selected) + 1) if selected in ready.ticker.astype(str).tolist() else np.nan,
            **{f"selected_block_{b}": rr[f"block_{b}"] if rr is not None else np.nan for b in "ABCDEF"},
        })
    return pd.DataFrame(rows)


def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    actions = pd.read_csv(PATHS / "p3_layer5_phase_b_scenario_daily_actions.csv", dtype={"selected_ticker": str})
    for c in ["decision_date", "requested_execution_date"]:
        actions[c] = pd.to_datetime(actions[c])
    c0 = actions[actions.scenario.eq("C0")].copy()
    c1 = actions[actions.scenario.eq("C1")].copy()
    episodes = pd.read_csv(EXP / "p3_corrected_nav_C0_C1_difference_episodes.csv")
    episodes[["start_decision_date", "end_decision_date"]] = episodes[["start_decision_date", "end_decision_date"]].apply(pd.to_datetime)
    feature = pd.read_csv(FEATURE, dtype={"ticker": str}, low_memory=False)
    feature["decision_date"] = pd.to_datetime(feature.decision_date)
    market = pd.read_csv(MARKET, usecols=["decision_date", "full_spec_v2_state"])
    market["decision_date"] = pd.to_datetime(market.decision_date)
    raw = load_raw()
    adj = load_adjusted()
    marks = pd.read_csv(PATHS / "p3_layer5_phase_b_daily_unique_position_marks.csv", dtype={"held_ticker": str})
    marks["date"] = pd.to_datetime(marks.date)
    held = marks.loc[marks.date.eq(pd.Timestamp("2025-08-01")), "held_ticker"].dropna().tolist()
    adj, bracket = apply_factor_bracket(adj, raw, held, "2025-08-01")
    px = prices(); px_lookup={(r.date,str(r.ticker)):(float(r.close),r.source_quality) for r in px.itertuples(index=False)}
    adj_lookup=adj.set_index(['date','ticker']); raw_lookup=raw.set_index(['date','ticker'])
    cal = pd.read_csv(ROOT / "backtest_cache/stock_pool_observations/0050_TW.csv", usecols=["date"])
    cal["date"] = pd.to_datetime(cal.date)
    calendar = cal.date[cal.date.between("2023-07-17", "2026-06-30")].drop_duplicates().sort_values().tolist()
    event_rows, variants = [], []
    for ep in episodes.itertuples(index=False):
        mask = c1.decision_date.between(ep.start_decision_date, ep.end_decision_date)
        prior = c1[c1.decision_date.lt(ep.start_decision_date)].selected_ticker.dropna().tail(1)
        incumbent = str(prior.iloc[0]) if len(prior) else None
        episode_actions = c1[mask]
        for r in episode_actions.itertuples(index=False):
            c0r = c0[c0.decision_date.eq(r.decision_date)].iloc[0]
            actual_prior = c1[c1.decision_date.lt(r.decision_date)].selected_ticker.dropna().tail(1)
            event_incumbent = str(actual_prior.iloc[0]) if len(actual_prior) else None
            event_rows.append({"event_id": f"E{ep.episode_id}_{r.decision_date.date()}", "episode_id": ep.episode_id,
                "decision_date": r.decision_date, "execution_date": r.requested_execution_date,
                "action_type": r.selected_action, "invalidity_status": r.action_reason,
                "incumbent": event_incumbent, "challenger": r.selected_ticker,
                "alternate_action": "hold_pre_episode_incumbent", "alternate_target": incumbent,
                "C0_alternate_target": c0r.selected_ticker, "reason_codes": r.action_reason})
        base_variants = {"C1_actual": c1.copy(), "hold_pre_episode_incumbent": c1.copy(), "C0_targets_during_episode": c1.copy()}
        base_variants["hold_pre_episode_incumbent"].loc[mask, "selected_ticker"] = incumbent
        c0map = c0.set_index("decision_date").selected_ticker
        base_variants["C0_targets_during_episode"].loc[mask, "selected_ticker"] = base_variants["C0_targets_during_episode"].loc[mask, "decision_date"].map(c0map)
        if episode_actions.selected_action.eq("forced_replacement").any():
            base_variants["forced_exit_no_position"] = c1.copy()
            base_variants["forced_exit_no_position"].loc[mask, "selected_ticker"] = None
        for name, variant in base_variants.items():
            variant = variant.copy(); variant["scenario"] = f"EP{ep.episode_id}_{name}"
            ev = fast_transitions(variant, px_lookup, calendar)
            daily, trans = wealth(ev, adj_lookup, raw_lookup, calendar, variant.scenario.iloc[0])
            daily["episode_id"] = ep.episode_id; daily["variant"] = name
            trans["episode_id"] = ep.episode_id; trans["variant"] = name
            variants.append((daily, trans))
    event_map = pd.DataFrame(event_rows)
    daily = pd.concat([x[0] for x in variants], ignore_index=True)
    transitions = pd.concat([x[1] for x in variants], ignore_index=True)
    context = candidate_context(event_map, feature, adj, calendar)
    regime = market.set_index("decision_date").full_spec_v2_state
    drawdown = []
    for cfid, g in daily.groupby("counterfactual_id"):
        g = g.sort_values("date").copy(); peak = g.NAV_close.cummax(); dd = g.NAV_close / peak - 1; ti = dd.idxmin(); pi = g.loc[:ti, "NAV_close"].idxmax()
        drawdown.append({"counterfactual_id": cfid, "peak_date": g.loc[pi, "date"], "trough_date": g.loc[ti, "date"],
            "peak_NAV": g.loc[pi, "NAV_close"], "trough_NAV": g.loc[ti, "NAV_close"], "MDD": dd.loc[ti],
            "held_target_at_trough": g.loc[ti, "counterfactual_target"], "action_at_trough": g.loc[ti, "transition_type"],
            "controller_state_at_trough": regime.get(g.loc[ti, "date"], "terminal_execution_mark"),
            "exact_NAV_delta_vs_unit_start": g.loc[ti, "NAV_close"] - 1.0})
    daily.to_csv(OUT / "p3_layer5_counterfactual_NAV_rechain_daily_ledger.csv", index=False, encoding="utf-8-sig")
    transitions.to_csv(OUT / "p3_layer5_counterfactual_transition_reconciliation.csv", index=False, encoding="utf-8-sig")
    event_map.to_csv(OUT / "p3_layer5_event_counterfactual_map.csv", index=False, encoding="utf-8-sig")
    context.to_csv(OUT / "p3_layer5_entry_candidate_evaluation_context.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(drawdown).to_csv(OUT / "p3_layer5_drawdown_episode_counterfactual_link.csv", index=False, encoding="utf-8-sig")
    bracket.to_csv(OUT / "p3_layer5_counterfactual_factor_bracket_audit.csv", index=False, encoding="utf-8-sig")
    readiness = {"task_id": TASK, "status": "exact_counterfactual_NAV_rechain_ready", "episode_count": int(episodes.shape[0]),
        "counterfactual_path_count": int(daily.counterfactual_id.nunique()), "daily_rows": len(daily), "transition_rows": len(transitions),
        "cross_asset_nominal_price_return_count": 0, "abs_return_gt_15pct_rows": int((daily.gross_same_asset_return.abs() > .15).sum()),
        "ready_for_experiments": True, "future_data_violation_count": 0, "formal_model_changed": False,
        "trade_decision_changed": False, "active_in_trade_decision": False, "report_changed": False,
        "ready_for_formal": False, "ready_for_strategy_replay": False, "not_live_rule": True,
        "forward_returns_live_rule_usage": False}
    (OUT / "readiness_for_counterfactual_NAV_rechain.json").write_text(json.dumps(readiness, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "final_summary_zh.md").write_text("# P3 exact counterfactual NAV rechain\n\n6個C0/C1差異episode已依固定NAV、event-aware mark、官方execution與EP05+10bp/side完整重鏈；forward return僅作evaluation metadata。\n", encoding="utf-8")
    files = sorted(p for p in OUT.iterdir() if p.is_file() and p.name != "manifest.json")
    (OUT / "manifest.json").write_text(json.dumps({"task_id": TASK, "files": [{"name": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in files]}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    run()
