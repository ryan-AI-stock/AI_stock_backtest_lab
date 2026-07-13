from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from backtest_lab import vnext_p3_all80_continuous_lifecycle_state_supply as all80
from backtest_lab import vnext_p3_rank1_dynamic_kd_price_range_stage_a as rank1
from backtest_lab import vnext_p3_rank1_sequential_lifecycle_contract as rank1_source


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/vnext_p3_layer04_top20_ticker_specific_KD_price_range_timing_stage_A_contract_20260713"
TASK = "TASK-BACKTEST-CORE-VNEXT-P3-LAYER04-TOP20-TICKER-SPECIFIC-KD-PRICE-RANGE-TIMING-STAGE-A-CONTRACT-001"
P3_1_END = pd.Timestamp("2025-07-10")
STOPPED_GOVERNANCE = {
    "superseded_by_all80_K_range_eligibility_comparison": True,
    "non_representative_of_current_scope": True,
    "follow_up_stopped": True,
    "ready_for_experiments": False,
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _features() -> pd.DataFrame:
    history = all80._history().rename(columns={"date": "decision_date"})
    pieces = []
    for ticker, g in history.groupby("ticker", sort=False):
        g = g.sort_values("decision_date").copy()
        low9 = g.adjusted_low.rolling(9, min_periods=9).min()
        high9 = g.adjusted_high.rolling(9, min_periods=9).max()
        g["RSV9"] = 100 * (g.adjusted_close - low9) / (high9 - low9).replace(0, np.nan)
        ks, ds, pk, pd_ = [], [], 50.0, 50.0
        for value in g.RSV9:
            if pd.notna(value):
                pk = (2 * pk + value) / 3
                pd_ = (2 * pd_ + pk) / 3
            ks.append(pk); ds.append(pd_)
        g["K"] = ks; g["D"] = ds
        g["prior_adjusted_close"] = g.adjusted_close.shift()
        g["KD_cross_up"] = g.K.gt(g.D) & g.K.shift().le(g.D.shift())
        g["KD_cross_down"] = g.K.lt(g.D) & g.K.shift().ge(g.D.shift())
        for window in (60, 120):
            for field in ("K", "D", "adjusted_close"):
                g[f"{field}_min_{window}TD"] = g[field].rolling(window, min_periods=window).min()
                g[f"{field}_max_{window}TD"] = g[field].rolling(window, min_periods=window).max()
                width = g[f"{field}_max_{window}TD"] - g[f"{field}_min_{window}TD"]
                g[f"{field}_location_{window}TD"] = ((g[field] - g[f"{field}_min_{window}TD"]) / width).where(width.gt(0))
                g[f"{field}_empirical_pct_{window}TD"] = g[field].rolling(window, min_periods=window).apply(rank1._last_percentile, raw=True)
            g[f"K_range_width_{window}TD"] = g[f"K_max_{window}TD"] - g[f"K_min_{window}TD"]
            g[f"D_range_width_{window}TD"] = g[f"D_max_{window}TD"] - g[f"D_min_{window}TD"]
            g[f"adjusted_price_range_pct_{window}TD"] = g.adjusted_high.rolling(window, min_periods=window).max() / g.adjusted_low.rolling(window, min_periods=window).min() - 1
        pieces.append(g)
    return pd.concat(pieces, ignore_index=True)


def _official_raw() -> pd.DataFrame:
    daily = pd.read_csv(rank1_source.DAILY, dtype={"ticker": str}, usecols=["decision_date", "ticker", "close", "raw_execution_source_quality"])
    daily["decision_date"] = pd.to_datetime(daily.decision_date)
    daily = daily.rename(columns={"close": "official_raw_close", "raw_execution_source_quality": "official_raw_source_quality"})
    return daily.dropna(subset=["official_raw_close"]).drop_duplicates(["decision_date", "ticker"], keep="last")


def _candidate_events(membership: pd.DataFrame, features: pd.DataFrame, window: int, zone: float, latch: int, gate: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    eligible_keys = set(zip(membership.decision_date, membership.ticker))
    membership_lookup = membership.set_index(["decision_date", "ticker"])
    setups: dict[str, dict] = {}
    events, excluded = [], []
    for ticker, g in features.groupby("ticker", sort=False):
        g = g.loc[g.decision_date.le(P3_1_END)].sort_values("decision_date")
        prior_qualified = False
        for row in g.itertuples():
            key = (row.decision_date, ticker)
            in_top20 = key in eligible_keys
            price_loc = getattr(row, f"adjusted_close_location_{window}TD")
            k_loc = getattr(row, f"K_location_{window}TD")
            k_range = getattr(row, f"K_range_width_{window}TD")
            low = in_top20 and pd.notna(price_loc) and pd.notna(k_loc) and price_loc <= zone and k_loc <= zone
            if low:
                if pd.notna(k_range) and k_range > gate:
                    setups[ticker] = {"remaining": latch, "setup_date": row.decision_date}
                else:
                    excluded.append({"platform":f"{window}TD_zone{int(zone*100)}_latch{latch}_KrangeGT{gate}","decision_date":row.decision_date,"ticker":ticker,"pool_rank":membership_lookup.loc[key].pool_rank,"K_range_width":k_range,"minimum_K_range_threshold":gate,"adjusted_price_range_pct":getattr(row,f"adjusted_price_range_pct_{window}TD"),"outcome_5_10_20_40_role":"evaluation_metadata_only"})
            elif ticker in setups:
                setups[ticker]["remaining"] -= 1
                if setups[ticker]["remaining"] < 0:
                    setups.pop(ticker)
            qualified = bool(in_top20 and ticker in setups and row.KD_cross_up and row.adjusted_close > row.prior_adjusted_close)
            if qualified and not prior_qualified:
                rank_row = membership_lookup.loc[key]
                events.append({"platform":f"{window}TD_zone{int(zone*100)}_latch{latch}_KrangeGT{gate}","decision_date":row.decision_date,"ticker":ticker,"pool_rank":int(rank_row.pool_rank),"combined_self_location":float(np.mean([price_loc,k_loc])),"price_normalized_location":price_loc,"K_normalized_location":k_loc,"K":row.K,"D":row.D,"K_min":getattr(row,f"K_min_{window}TD"),"K_max":getattr(row,f"K_max_{window}TD"),"K_range_width":k_range,"D_range_width":getattr(row,f"D_range_width_{window}TD"),"price_min":getattr(row,f"adjusted_close_min_{window}TD"),"price_max":getattr(row,f"adjusted_close_max_{window}TD"),"adjusted_price_range_pct":getattr(row,f"adjusted_price_range_pct_{window}TD"),"reason":"top20_low_setup_then_K_cross_up_and_price_up","event_cluster_start":True})
            prior_qualified = qualified
    return pd.DataFrame(events), pd.DataFrame(excluded)


def _path(events: pd.DataFrame, features: pd.DataFrame, raw: pd.DataFrame, window: int, zone: float, latch: int, gate: int, selection_order: str = "self_location") -> tuple[pd.DataFrame, pd.DataFrame]:
    platform = f"{window}TD_zone{int(zone*100)}_latch{latch}_KrangeGT{gate}"
    ev = events.loc[events.platform.eq(platform)].copy()
    order = ["decision_date","pool_rank","ticker"] if selection_order == "pool_rank" else ["decision_date","combined_self_location","pool_rank","ticker"]
    winners = ev.sort_values(order).drop_duplicates("decision_date")
    by_date = {d:g for d,g in winners.groupby("decision_date")}
    lookup = features.set_index(["decision_date","ticker"])
    raw_lookup = raw.set_index(["decision_date","ticker"])
    calendar = sorted(features.loc[features.decision_date.between("2023-07-11",P3_1_END),"decision_date"].unique())
    incumbent = None; high_setup = None; rows=[]; transitions=[]
    for date in calendar:
        action="hold_or_wait"; target=None; reason="no_entry_candidate" if incumbent is None else "hold_incumbent"
        if incumbent is None and date in by_date:
            target = by_date[date].iloc[0].ticker; action="entry_signal"; reason="highest_Layer4_rank_among_entry_qualified" if selection_order == "pool_rank" else "deterministic_lowest_self_location_top20"
        elif incumbent is not None and (date,incumbent) in lookup.index:
            row=lookup.loc[(date,incumbent)]; price_loc=row[f"adjusted_close_location_{window}TD"]; k_loc=row[f"K_location_{window}TD"]
            if pd.notna(price_loc) and pd.notna(k_loc) and price_loc>=1-zone and k_loc>=1-zone:
                high_setup={"remaining":latch}
            elif high_setup:
                high_setup["remaining"]-=1
                if high_setup["remaining"]<0: high_setup=None
            if high_setup and row.KD_cross_down and row.adjusted_close<row.prior_adjusted_close:
                action="exit_signal"; target=incumbent; reason="high_setup_then_K_cross_down_and_price_down"
        execution_date=pd.NaT; execution_close=np.nan; execution_status="not_requested"
        if action in {"entry_signal","exit_signal"}:
            future=raw.loc[raw.ticker.eq(target)&raw.decision_date.gt(date)].sort_values("decision_date")
            if len(future):
                execution_date=future.iloc[0].decision_date; execution_close=future.iloc[0].official_raw_close
                execution_status="ready_exact_next_ticker_trading_day"
                transitions.append({"platform":platform,"decision_date":date,"execution_date":execution_date,"ticker":target,"transition_type":"cash_to_stock" if action=="entry_signal" else "stock_to_cash","official_raw_close":execution_close,"official_raw_source_quality":future.iloc[0].official_raw_source_quality,"EP05_cost":True,"slippage_bp_primary":10})
                incumbent=target if action=="entry_signal" else None; high_setup=None
            else: execution_status="blocked_official_raw_execution_close"
        rows.append({"platform":platform,"decision_date":date,"signal_target":target,"incumbent":incumbent,"action":action,"reason":reason,"execution_date":execution_date,"execution_status":execution_status,"normal_switch":False,"market_controller_used":False})
    return pd.DataFrame(rows),pd.DataFrame(transitions)


def run() -> None:
    OUT.mkdir(parents=True,exist_ok=True)
    readiness = {
        "task_id": TASK,
        "status": "stopped_checkpoint_before_materialization",
        **STOPPED_GOVERNANCE,
        "materialization_executed": False,
        "performance_executed": False,
        "allowed_role": "checkpoint_reference_only",
        "formal_model_changed": False,
        "trade_decision_changed": False,
        "active_in_trade_decision": False,
        "report_changed": False,
        "not_live_rule": True,
    }
    (OUT/"readiness_for_top20_dynamic_self_range_stage_A.json").write_text(json.dumps(readiness,ensure_ascii=False,indent=2),encoding="utf-8")
    (OUT/"final_summary_zh.md").write_text("# Top20 dynamic self-range Stage A\n\nSTOPPED CHECKPOINT before materialization. Superseded by all80 K-range eligibility comparison; no supply, outcome, NAV, P3-2, or Experiments handoff was executed.\n",encoding="utf-8")
    files=sorted(p for p in OUT.iterdir() if p.is_file() and p.name!="manifest.json")
    (OUT/"manifest.json").write_text(json.dumps({"task_id":TASK,**STOPPED_GOVERNANCE,"files":[{"name":p.name,"sha256":_sha(p),"bytes":p.stat().st_size} for p in files]},ensure_ascii=False,indent=2),encoding="utf-8")
    return

    # Preserved implementation checkpoint. It must not run unless Strategy Center
    # explicitly re-authorizes this superseded Top20 scope.
    daily=pd.read_csv(rank1_source.DAILY,dtype={"ticker":str},usecols=["decision_date","ticker","pool_rank","membership_snapshot_date","membership_effective_date"])
    daily["decision_date"]=pd.to_datetime(daily.decision_date)
    membership=daily.loc[daily.pool_rank.between(1,20)&daily.decision_date.le(P3_1_END)].drop_duplicates(["decision_date","ticker"])
    features=_features(); raw=_official_raw()
    all_events=[];all_exclusions=[];all_actions=[];all_transitions=[];summary=[]
    for window in (60,120):
      for zone in (.1,.2,.3):
       for latch in (5,10):
        for gate in (0,20,25,30):
         events,exclusions=_candidate_events(membership,features,window,zone,latch,gate)
         all_events.append(events);all_exclusions.append(exclusions)
         events_all=pd.concat(all_events,ignore_index=True) if all_events else pd.DataFrame()
         actions,transitions=_path(events_all,features,raw,window,zone,latch,gate)
         all_actions.append(actions);all_transitions.append(transitions)
         summary.append({"platform":f"{window}TD_zone{int(zone*100)}_latch{latch}_KrangeGT{gate}","candidate_entry_events":len(events),"range_gate_exclusions":len(exclusions),"multi_candidate_dates":int(events.groupby("decision_date").size().gt(1).sum()) if len(events) else 0,"no_candidate_dates":membership.decision_date.nunique()-events.decision_date.nunique() if len(events) else membership.decision_date.nunique(),"path_entries":int(transitions.transition_type.eq("cash_to_stock").sum()) if len(transitions) else 0,"path_exits":int(transitions.transition_type.eq("stock_to_cash").sum()) if len(transitions) else 0,"holding_share":float(actions.incumbent.notna().mean()),"execution_blockers":int(actions.execution_status.eq("blocked_official_raw_execution_close").sum())})
    events=pd.concat(all_events,ignore_index=True); exclusions=pd.concat(all_exclusions,ignore_index=True) if any(len(x) for x in all_exclusions) else pd.DataFrame(); actions=pd.concat(all_actions,ignore_index=True); transitions=pd.concat(all_transitions,ignore_index=True) if any(len(x) for x in all_transitions) else pd.DataFrame(); summary=pd.DataFrame(summary)
    folds=pd.read_csv(rank1_source.FOLDS); fold_rows=[]
    for platform,part in events.groupby("platform"):
      for fold in folds.itertuples():
       sample=part.loc[part.decision_date.between(pd.Timestamp(fold.validation_start),pd.Timestamp(fold.validation_end))]
       fold_rows.append({"platform":platform,"fold_id":fold.fold_id,"entry_unique_events":len(sample),"entry_eligible_dates":sample.decision_date.nunique(),"multi_candidate_dates":int(sample.groupby("decision_date").size().gt(1).sum()) if len(sample) else 0,"embargo_decision_dates":40})
    fold_supply=pd.DataFrame(fold_rows)
    cross_fold=fold_supply.groupby("platform").entry_unique_events.apply(lambda s:(s>0).all())
    evaluable_platforms=set(cross_fold[cross_fold].index)
    exact_path=summary.execution_blockers.sum()==0 and len(transitions)>0
    ready=bool(evaluable_platforms) and exact_path
    membership.to_csv(OUT/"p3_top20_exact_membership_PIT.csv.gz",index=False,compression="gzip",encoding="utf-8")
    events.to_csv(OUT/"p3_top20_dynamic_self_range_candidate_event_ledger.csv.gz",index=False,compression="gzip",encoding="utf-8")
    exclusions.to_csv(OUT/"p3_top20_dynamic_self_range_K_range_exclusion_ledger.csv",index=False,encoding="utf-8-sig")
    actions.to_csv(OUT/"p3_top20_dynamic_self_range_single_position_action_trace.csv.gz",index=False,compression="gzip",encoding="utf-8")
    transitions.to_csv(OUT/"p3_top20_dynamic_self_range_execution_requirement_ledger.csv",index=False,encoding="utf-8-sig")
    summary.to_csv(OUT/"p3_top20_dynamic_self_range_48_platform_supply.csv",index=False,encoding="utf-8-sig")
    fold_supply.to_csv(OUT/"p3_top20_dynamic_self_range_fold_supply.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame([{"requirement":"candidate_outcome_5_10_20_40TD","authority":"event-aware adjusted same-ticker path; evaluation only","ready":True},{"requirement":"corrected_NAV","authority":"official raw execution plus event-aware adjusted holding and EP05 5/10/20bp","ready":exact_path},{"requirement":"00631L_same_coverage","authority":"EP05 ETF basis; no fallback","ready":True},{"requirement":"P3_2","authority":"locked until P3-1 pass","ready":False}]).to_csv(OUT/"p3_top20_dynamic_self_range_outcome_path_contract.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame([{"audit":"future_outcome_live_rule","violations":0},{"audit":"P3_2_outcome_read","violations":0},{"audit":"market_controller","violations":0},{"audit":"normal_switch","violations":0},{"audit":"same_day_execution","violations":int((transitions.execution_date<=transitions.decision_date).sum()) if len(transitions) else 0}]).to_csv(OUT/"p3_top20_dynamic_self_range_future_PIT_audit.csv",index=False,encoding="utf-8-sig")
    readiness={"task_id":TASK,"status":"ready_for_experiments_P3_1" if ready else "insufficient_cross_fold_supply_or_path_blocked","diagnostic_subproblem":True,"stage_scope":"Layer4_top20_stock_only_dynamic_self_range_timing","representative_of_full_all80_layer5":False,"may_be_used_to_reject_full_layer5":False,"market_controller_used":False,"normal_switch":False,"Top3":False,"threshold_grid_expansion":False,"frozen_platforms":48,"P3_1_decision_dates":membership.decision_date.nunique(),"top20_membership_rows":len(membership),"candidate_event_rows":len(events),"evaluable_cross_3fold_platform_count":len(evaluable_platforms),"execution_blockers":int(summary.execution_blockers.sum()),"ready_for_experiments":ready,"P3_2_outcome_read_authorized":False,"future_data_violation_count":0,"formal_model_changed":False,"trade_decision_changed":False,"active_in_trade_decision":False,"report_changed":False,"portfolio_replay_executed":False,"ready_for_formal":False,"ready_for_strategy_replay":False,"not_live_rule":True,"forward_returns_live_rule_usage":False}
    (OUT/"readiness_for_top20_dynamic_self_range_stage_A.json").write_text(json.dumps(readiness,ensure_ascii=False,indent=2),encoding="utf-8")
    (OUT/"final_summary_zh.md").write_text(f"# Top20 stock-only dynamic self-range Stage A\n\nP3-1 candidate events={len(events)}; cross-3fold evaluable platforms={len(evaluable_platforms)}; execution blockers={int(summary.execution_blockers.sum())}; ready_for_experiments={str(ready).lower()}. No market, all80 rerank, Top3, normal switch, performance, or P3-2 outcome read.\n",encoding="utf-8")
    files=sorted(p for p in OUT.iterdir() if p.is_file() and p.name!="manifest.json")
    (OUT/"manifest.json").write_text(json.dumps({"task_id":TASK,"files":[{"name":p.name,"sha256":_sha(p),"bytes":p.stat().st_size} for p in files]},ensure_ascii=False,indent=2),encoding="utf-8")


if __name__=="__main__": run()
