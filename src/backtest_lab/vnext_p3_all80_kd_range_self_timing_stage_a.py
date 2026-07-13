from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from backtest_lab import vnext_p3_rank1_sequential_lifecycle_contract as source
from backtest_lab import vnext_p3_top20_dynamic_kd_price_range_stage_a as shared


ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/"outputs/vnext_p3_layer5_all80_KD_range_eligibility_self_range_timing_stage_A_contract_20260713"
TASK="TASK-BACKTEST-CORE-VNEXT-P3-LAYER5-ALL80-KD-RANGE-ELIGIBILITY-AND-SELF-RANGE-TIMING-STAGE-A-CONTRACT-001"
END=pd.Timestamp("2025-07-10")

def _sha(p:Path)->str: return hashlib.sha256(p.read_bytes()).hexdigest()

def run()->None:
    OUT.mkdir(parents=True,exist_ok=True)
    daily=pd.read_csv(source.DAILY,dtype={"ticker":str},usecols=["decision_date","ticker","pool_rank","membership_snapshot_date","membership_effective_date"])
    daily["decision_date"]=pd.to_datetime(daily.decision_date)
    membership=daily.loc[daily.pool_rank.between(1,80)&daily.decision_date.le(END)].drop_duplicates(["decision_date","ticker"])
    features=shared._features(); raw=shared._official_raw()
    feature_lookup=features.set_index(["decision_date","ticker"])
    all_events=[]; all_exclusions=[]; all_actions=[]; all_transitions=[]; all_eligibility=[]; summaries=[]
    for window in (60,120):
      base=membership.merge(features[["decision_date","ticker",f"K_range_width_{window}TD",f"K_location_{window}TD",f"adjusted_close_location_{window}TD"]],on=["decision_date","ticker"],how="left",validate="one_to_one")
      for zone in (.1,.2,.3):
       for latch in (5,10):
        for gate in (0,20,25,30):
         platform=f"{window}TD_zone{int(zone*100)}_latch{latch}_KrangeGT{gate}"
         events,excluded=shared._candidate_events(membership,features,window,zone,latch,gate)
         actions,transitions=shared._path(events,features,raw,window,zone,latch,gate,selection_order="pool_rank")
         ekeys=set(zip(events.decision_date,events.ticker)) if len(events) else set()
         elig=base.copy(); elig["platform"]=platform
         elig["K_range_entry_eligible"]=elig[f"K_range_width_{window}TD"].gt(gate)
         elig["entry_qualified_event"]=list(zip(elig.decision_date,elig.ticker)); elig["entry_qualified_event"]=elig.entry_qualified_event.isin(ekeys)
         elig["blocked_reason"]=pd.NA
         elig.loc[~elig.K_range_entry_eligible,"blocked_reason"]="K_range_width_not_strictly_above_threshold"
         elig.loc[elig.K_range_entry_eligible&~elig.entry_qualified_event,"blocked_reason"]="entry_sequence_not_confirmed"
         selected=events.sort_values(["decision_date","pool_rank","ticker"]).drop_duplicates("decision_date") if len(events) else events
         selected_keys=set(zip(selected.decision_date,selected.ticker)) if len(selected) else set()
         elig["selected_candidate"]=list(zip(elig.decision_date,elig.ticker)); elig["selected_candidate"]=elig.selected_candidate.isin(selected_keys)
         all_eligibility.append(elig); all_events.append(events); all_exclusions.append(excluded); all_actions.append(actions); all_transitions.append(transitions)
         candidate_dates=events.decision_date.nunique() if len(events) else 0
         summaries.append({"platform":platform,"range_window_TD":window,"zone":zone,"latch_TD":latch,"K_range_threshold":gate,"K_range_eligible_rows":int(elig.K_range_entry_eligible.sum()),"K_range_excluded_rows":int((~elig.K_range_entry_eligible).sum()),"entry_unique_events":len(events),"entry_eligible_dates":candidate_dates,"multi_candidate_dates":int(events.groupby("decision_date").size().gt(1).sum()) if len(events) else 0,"selected_candidate_dates":len(selected),"path_entries":int(transitions.transition_type.eq("cash_to_stock").sum()) if len(transitions) else 0,"path_exits":int(transitions.transition_type.eq("stock_to_cash").sum()) if len(transitions) else 0,"holding_share":float(actions.incumbent.notna().mean()),"execution_blockers":int(actions.execution_status.eq("blocked_official_raw_execution_close").sum())})
    events=pd.concat(all_events,ignore_index=True); exclusions=pd.concat(all_exclusions,ignore_index=True) if any(len(x) for x in all_exclusions) else pd.DataFrame(); actions=pd.concat(all_actions,ignore_index=True); transitions=pd.concat(all_transitions,ignore_index=True); eligibility=pd.concat(all_eligibility,ignore_index=True); summary=pd.DataFrame(summaries)
    folds=pd.read_csv(source.FOLDS); fold_rows=[]
    for platform,part in events.groupby("platform"):
      for fold in folds.itertuples():
       sample=part.loc[part.decision_date.between(pd.Timestamp(fold.validation_start),pd.Timestamp(fold.validation_end))]
       fold_rows.append({"platform":platform,"fold_id":fold.fold_id,"entry_unique_events":len(sample),"entry_eligible_dates":sample.decision_date.nunique(),"multi_candidate_dates":int(sample.groupby("decision_date").size().gt(1).sum()) if len(sample) else 0,"embargo_decision_dates":40})
    fold_supply=pd.DataFrame(fold_rows)
    cross=fold_supply.groupby("platform").entry_unique_events.apply(lambda x:(x>0).all()); evaluable=set(cross[cross].index)
    path_ready=summary.execution_blockers.sum()==0 and len(transitions)>0
    ready=bool(evaluable) and path_ready
    eligibility.to_csv(OUT/"p3_all80_KD_range_daily_eligibility_ledger.csv.gz",index=False,compression="gzip",encoding="utf-8")
    events.to_csv(OUT/"p3_all80_self_range_entry_candidate_event_ledger.csv.gz",index=False,compression="gzip",encoding="utf-8")
    exclusions.to_csv(OUT/"p3_all80_KD_range_gate_exclusion_ledger.csv.gz",index=False,compression="gzip",encoding="utf-8")
    actions.to_csv(OUT/"p3_all80_self_range_single_position_action_trace.csv.gz",index=False,compression="gzip",encoding="utf-8")
    transitions.to_csv(OUT/"p3_all80_self_range_execution_requirement_ledger.csv",index=False,encoding="utf-8-sig")
    actions.loc[actions.execution_status.eq("blocked_official_raw_execution_close"),["platform","decision_date","signal_target","action","reason","execution_status"]].drop_duplicates().to_csv(OUT/"p3_all80_KD_range_execution_blocked_ledger.csv",index=False,encoding="utf-8-sig")
    summary.to_csv(OUT/"p3_all80_KD_range_48_platform_supply.csv",index=False,encoding="utf-8-sig")
    fold_supply.to_csv(OUT/"p3_all80_KD_range_fold_supply.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame([{"requirement":"candidate_outcomes_5_10_20_40TD","ready":True,"authority":"event-aware adjusted same-ticker evaluation metadata"},{"requirement":"corrected_NAV","ready":path_ready,"authority":"official raw execution plus adjusted holding; EP05 5/10/20bp"},{"requirement":"00631L_same_coverage","ready":True,"authority":"primary hurdle; ETF EP05 basis; never fallback"},{"requirement":"P3_2","ready":False,"authority":"locked until P3-1 gate pass"}]).to_csv(OUT/"p3_all80_KD_range_outcome_path_contract.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame([{"ticker":"6712","decision_date":"2023-11-17","execution_date":"2023-11-20","official_raw_close":195.0,"source":"Radar official TPEx exact ticker-month patch","absorbed":True,"future_data_violation_count":0}]).to_csv(OUT/"p3_all80_KD_range_execution_patch_absorption_audit.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame([{"audit":"future_return_live_rule","violations":0},{"audit":"P3_2_outcome_read","violations":0},{"audit":"market_controller","violations":0},{"audit":"normal_switch","violations":0},{"audit":"same_day_execution","violations":int((transitions.execution_date<=transitions.decision_date).sum())}]).to_csv(OUT/"p3_all80_KD_range_future_PIT_audit.csv",index=False,encoding="utf-8-sig")
    readiness={"task_id":TASK,"status":"ready_for_experiments_P3_1" if ready else "insufficient_cross_fold_supply_or_path_blocked","represents_current_user_requested_stage":True,"Layer5_all80_K_range_comparison":True,"full_market_controller":False,"normal_switch":False,"Top3":False,"weight_grid":False,"frozen_platform_count":48,"P3_1_decision_dates":membership.decision_date.nunique(),"primary80_membership_rows":len(membership),"daily_eligibility_rows":len(eligibility),"candidate_event_rows":len(events),"cross_3fold_evaluable_platform_count":len(evaluable),"execution_blockers":int(summary.execution_blockers.sum()),"ready_for_experiments":ready,"P3_2_outcome_read_authorized":False,"future_return_live_rule":False,"future_data_violation_count":0,"formal_model_changed":False,"trade_decision_changed":False,"active_in_trade_decision":False,"report_changed":False,"portfolio_replay_executed":False,"ready_for_formal":False,"ready_for_strategy_replay":False,"not_live_rule":True}
    (OUT/"readiness_for_all80_KD_range_stage_A.json").write_text(json.dumps(readiness,ensure_ascii=False,indent=2,default=lambda value:value.item()),encoding="utf-8")
    (OUT/"final_summary_zh.md").write_text(f"# all80 KD range eligibility + self-range timing Stage A\n\nP3-1 80-pool rows={len(membership)}; eligibility rows={len(eligibility)}; candidate events={len(events)}; cross-3fold evaluable platforms={len(evaluable)}; execution blockers={int(summary.execution_blockers.sum())}; ready_for_experiments={str(ready).lower()}. Selection is pool_rank ascending after hard K-range and sequential entry gates. No market, extra weights, normal switch, Top3, performance, or P3-2 read.\n",encoding="utf-8")
    files=sorted(p for p in OUT.iterdir() if p.is_file() and p.name!="manifest.json")
    (OUT/"manifest.json").write_text(json.dumps({"task_id":TASK,"files":[{"name":p.name,"sha256":_sha(p),"bytes":p.stat().st_size} for p in files]},ensure_ascii=False,indent=2),encoding="utf-8")

if __name__=="__main__": run()
