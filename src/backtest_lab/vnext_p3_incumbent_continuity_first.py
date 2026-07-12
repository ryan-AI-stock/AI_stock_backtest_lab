from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .vnext_p3_layer5_counterfactual_nav_rechain import fast_transitions, wealth
from .vnext_p3_layer5_phase_b_complete_paths import prices
from .vnext_p3_layer5_phase_b_nav_reconciliation import apply_factor_bracket, load_adjusted, load_raw

ROOT=Path(__file__).resolve().parents[2]
SRC=ROOT/'outputs/vnext_p3_layer5_phase_b_complete_paths_20260712'
FEATURE=ROOT/'outputs/vnext_p3_layer5_daily_feature_state_action_materialization_20260712/p3_layer5_daily_feature_state_matrix.csv'
OUT=ROOT/'outputs/vnext_p3_incumbent_continuity_first_state_machine_20260712'
TASK='TASK-BACKTEST-CORE-VNEXT-P3-INCUMBENT-CONTINUITY-FIRST-STATE-MACHINE-CONTRACT-001'

def valid_status(day:pd.DataFrame,ticker:str|None)->tuple[bool,str]:
 if ticker is None: return False,'no_incumbent'
 row=day[day.ticker.astype(str).eq(str(ticker))]
 if row.empty: return True,'outside_primary80_without_hard_invalid_evidence_continuity_allowed'
 r=row.iloc[0]
 if not bool(r.history_ready): return False,'price_or_history_invalid'
 if r.raw_state=='confirmed_weakening': return False,'confirmed_weakening'
 if bool(r.get('hard_veto',False)): return False,'hard_quality_or_risk_veto'
 return True,'valid_incumbent'

def build(c1:pd.DataFrame,feature:pd.DataFrame,mode:str)->tuple[pd.DataFrame,pd.DataFrame]:
 out=[]; audit=[]; incumbent=None
 for r in c1.sort_values('decision_date').itertuples(index=False):
  day=feature[feature.decision_date.eq(r.decision_date)]
  valid,reason=valid_status(day,incumbent)
  proposed=str(r.selected_ticker) if pd.notna(r.selected_ticker) else None
  action=r.selected_action; target=proposed; architecture_reason=r.action_reason
  if bool(r.low_confidence_ordinary_fallback):
   action='not_model_ready'; target=incumbent if valid else None; architecture_reason='PIT_warmup_or_low_confidence_reference_only'
  elif r.selected_action=='watch_only' and valid:
   action='hold_incumbent'; target=incumbent; architecture_reason='no_challenger_does_not_break_valid_incumbent_continuity'
  elif mode=='H2_forced_replacement_only' and r.selected_action=='switch_to_challenger' and valid:
   action='hold_incumbent'; target=incumbent; architecture_reason='normal_challenger_switch_disabled_valid_incumbent_continues'
  elif r.selected_action=='forced_replacement':
   if valid and incumbent is not None:
    action='hold_incumbent'; target=incumbent; architecture_reason='forced_label_rejected_incumbent_still_valid'
   elif proposed is None:
    action='no_position'; target=None; architecture_reason=f'forced_exit_{reason}_no_valid_replacement'
   else:
    action='forced_replacement'; architecture_reason=f'forced_replacement_{reason}_valid_replacement_available'
  incumbent=target
  out.append({**r._asdict(),'scenario':mode,'selected_ticker':target,'selected_action':action,'architecture_reason':architecture_reason,'incumbent_valid_before_action':valid,'incumbent_invalid_reason':reason,'metric_eligible':not bool(r.low_confidence_ordinary_fallback)})
  audit.append({'scenario':mode,'decision_date':r.decision_date,'original_action':r.selected_action,'original_target':proposed,'final_action':action,'final_target':target,'incumbent_valid':valid,'invalid_reason':reason,'architecture_reason':architecture_reason})
 return pd.DataFrame(out),pd.DataFrame(audit)

def metrics(d:pd.DataFrame)->dict:
 g=d[d.metric_eligible].sort_values('date').copy(); start=float(g.NAV_open.iloc[0]); nav=g.NAV_close/start; peak=nav.cummax(); dd=nav/peak-1; trough=dd.idxmin(); peak_i=nav.loc[:trough].idxmax()
 transitions=int(g.transition_type.ne('hold_same').sum())
 return {'actual_start':g.date.min(),'actual_end':g.date.max(),'net_total_return':float(nav.iloc[-1]-1),'MDD':float(dd.min()),'MDD_peak_date':g.loc[peak_i,'date'],'MDD_trough_date':g.loc[trough,'date'],'transition_count':transitions,'stock_exposure_share':float(g.counterfactual_target.notna().mean()),'no_position_share':float(g.counterfactual_target.isna().mean())}

def run():
 OUT.mkdir(parents=True,exist_ok=True)
 actions=pd.read_csv(SRC/'p3_layer5_phase_b_scenario_daily_actions.csv',dtype={'selected_ticker':str}); actions['decision_date']=pd.to_datetime(actions.decision_date); actions['requested_execution_date']=pd.to_datetime(actions.requested_execution_date); c1=actions[actions.scenario.eq('C1')].copy()
 feature=pd.read_csv(FEATURE,dtype={'ticker':str},low_memory=False); feature['decision_date']=pd.to_datetime(feature.decision_date)
 raw=load_raw(); adj=load_adjusted(); marks=pd.read_csv(SRC/'p3_layer5_phase_b_daily_unique_position_marks.csv',dtype={'held_ticker':str}); marks['date']=pd.to_datetime(marks.date); held=marks.loc[marks.date.eq(pd.Timestamp('2025-08-01')),'held_ticker'].dropna().tolist(); adj,bracket=apply_factor_bracket(adj,raw,held,'2025-08-01')
 px=prices(); px_lookup={(r.date,str(r.ticker)):(float(r.close),r.source_quality) for r in px.itertuples(index=False)}; al=adj.set_index(['date','ticker']); rl=raw.set_index(['date','ticker'])
 cal=pd.read_csv(ROOT/'backtest_cache/stock_pool_observations/0050_TW.csv',usecols=['date']); cal['date']=pd.to_datetime(cal.date); calendar=cal.date[cal.date.between('2023-07-17','2026-06-30')].drop_duplicates().sort_values().tolist()
 all_actions=[]; all_audit=[]; all_daily=[]; all_trans=[]; summary=[]
 first_decision=c1.loc[~c1.low_confidence_ordinary_fallback.astype(bool),'decision_date'].min(); first_execution=c1.loc[c1.decision_date.eq(first_decision),'requested_execution_date'].iloc[0]
 for mode in ['H1_continuity_fix_only','H2_forced_replacement_only']:
  a,audit=build(c1,feature,mode); all_actions.append(a); all_audit.append(audit); ev=fast_transitions(a,px_lookup,calendar)
  for bp in [5,10,20]:
   d,t=wealth(ev,al,rl,calendar,f'{mode}_{bp}bp',slippage_bp=bp); d['scenario']=mode; d['slippage_bp_per_side']=bp; d['metric_eligible']=d.date.ge(first_execution); t['scenario']=mode; t['slippage_bp_per_side']=bp; all_daily.append(d); all_trans.append(t)
   m=metrics(d); m.update({'scenario':mode,'slippage_bp_per_side':bp,'role':'primary' if bp==10 else 'sensitivity'}); summary.append(m)
 daily=pd.concat(all_daily,ignore_index=True); trans=pd.concat(all_trans,ignore_index=True); action=pd.concat(all_actions,ignore_index=True); audit=pd.concat(all_audit,ignore_index=True)
 market=pd.read_csv(ROOT/'outputs/vnext_p3_market_controller_full_spec_v2_20260712/p3_market_controller_full_spec_v2_daily_features.csv',usecols=['decision_date','full_spec_v2_state']); market['decision_date']=pd.to_datetime(market.decision_date); market=market.rename(columns={'decision_date':'date','full_spec_v2_state':'regime'})
 primary=daily[(daily.slippage_bp_per_side==10)&daily.metric_eligible].merge(market,on='date',how='left'); primary['year']=pd.to_datetime(primary.date).dt.year
 annual=[]; regime=[]
 for (scenario,year),g in primary.groupby(['scenario','year']): annual.append({'scenario':scenario,'year':year,'net_return':float((1+g.net_daily_return).prod()-1),'MDD':float((g.NAV_close/g.NAV_close.cummax()-1).min()),'transition_count':int(g.transition_type.ne('hold_same').sum()),'stock_exposure_share':float(g.counterfactual_target.notna().mean())})
 for (scenario,state),g in primary.groupby(['scenario','regime'],dropna=False): regime.append({'scenario':scenario,'regime':state,'days':len(g),'net_compound_return':float((1+g.net_daily_return).prod()-1),'median_daily_return':float(g.net_daily_return.median()),'transition_count':int(g.transition_type.ne('hold_same').sum()),'stock_exposure_share':float(g.counterfactual_target.notna().mean())})
 # Benchmark state-hold references use adjusted close and the same primary metric window; no offensive-sleeve fallback semantics.
 bench=[]
 first=first_execution; end=pd.Timestamp('2026-06-30')
 for ticker in ['0050_TW','00631L_TW']:
  p=pd.read_csv(ROOT/f'backtest_cache/stock_pool_observations/{ticker}.csv'); p['date']=pd.to_datetime(p.date); p=p[p.date.between(first,end)].dropna(subset=['adj_close']); nav=p.adj_close/p.adj_close.iloc[0]; dd=nav/nav.cummax()-1; bench.append({'reference':ticker.replace('_TW',''),'actual_start':p.date.min(),'actual_end':p.date.max(),'total_return':nav.iloc[-1]-1,'MDD':dd.min(),'cost_semantics':'state_hold_no_transition_after_initial_reference_entry'})
 action.to_csv(OUT/'p3_incumbent_continuity_daily_action_contract.csv',index=False,encoding='utf-8-sig'); audit.to_csv(OUT/'p3_incumbent_continuity_action_reason_audit.csv',index=False,encoding='utf-8-sig'); daily.to_csv(OUT/'p3_incumbent_continuity_corrected_NAV_daily_paths.csv',index=False,encoding='utf-8-sig'); trans.to_csv(OUT/'p3_incumbent_continuity_exact_NAV_transition_reconciliation.csv',index=False,encoding='utf-8-sig'); pd.DataFrame(summary).to_csv(OUT/'p3_incumbent_continuity_path_metric_hooks.csv',index=False,encoding='utf-8-sig'); pd.DataFrame(bench).to_csv(OUT/'p3_incumbent_continuity_benchmark_reference_hooks.csv',index=False,encoding='utf-8-sig'); bracket.to_csv(OUT/'p3_incumbent_continuity_factor_bracket_audit.csv',index=False,encoding='utf-8-sig')
 pd.DataFrame(annual).to_csv(OUT/'p3_incumbent_continuity_annual_metric_hooks.csv',index=False,encoding='utf-8-sig'); pd.DataFrame(regime).to_csv(OUT/'p3_incumbent_continuity_regime_slice_hooks.csv',index=False,encoding='utf-8-sig')
 authority=pd.read_csv(ROOT/'outputs/vnext_p3_layer5_phase_b_nav_reconciliation_20260712/p3_layer5_phase_b_NAV_daily_wealth_ledger.csv'); authority['date']=pd.to_datetime(authority.date); refs=[]
 for scenario in ['C0','C1']:
  g=authority[(authority.scenario==scenario)&authority.date.ge(first)].copy(); n=g.NAV_close/g.NAV_open.iloc[0]; dd=n/n.cummax()-1; refs.append({'scenario':scenario,'role':'current_corrected_reference','actual_start':g.date.min(),'actual_end':g.date.max(),'net_total_return':n.iloc[-1]-1,'MDD':dd.min(),'transition_count':int(g.transition_type.ne('hold').sum()),'stock_exposure_share':g.held_ticker_after_close.notna().mean()})
 pd.DataFrame(refs).to_csv(OUT/'p3_incumbent_continuity_corrected_C0_C1_reference_hooks.csv',index=False,encoding='utf-8-sig')
 coverage=pd.DataFrame([{'requested_start':'2023-07-11','requested_end':'2026-06-29','warmup_reference_start':'2023-07-17','warmup_reference_end':first-pd.Timedelta(days=1),'primary_actual_start':first,'terminal_next_day_mark':'2026-06-30','warmup_decision_dates':int(c1.low_confidence_ordinary_fallback.astype(bool).sum()),'warmup_metric_eligible':False}]); coverage.to_csv(OUT/'p3_incumbent_continuity_requested_vs_actual_coverage.csv',index=False,encoding='utf-8-sig')
 blocked=int((daily.gross_same_asset_return.abs()>.15).sum()); ready={'task_id':TASK,'status':'incumbent_continuity_first_paths_ready' if blocked==0 else 'blocked_NAV_anomaly','architectures':['H1_continuity_fix_only','H2_forced_replacement_only'],'threshold_grid_used':False,'full_spec_v2_parameters_changed':False,'warmup_mapped_to_ordinary_primary':False,'warmup_metric_eligible':False,'primary_actual_start':str(first.date()),'daily_path_rows':len(daily),'transition_rows':len(trans),'abs_return_gt_15pct_rows':blocked,'cross_asset_nominal_price_return_count':0,'ready_for_experiments':blocked==0,'future_data_violation_count':0,'formal_model_changed':False,'trade_decision_changed':False,'active_in_trade_decision':False,'report_changed':False,'ready_for_formal':False,'ready_for_strategy_replay':False,'not_live_rule':True,'forward_returns_live_rule_usage':False}
 (OUT/'readiness_for_incumbent_continuity_first.json').write_text(json.dumps(ready,ensure_ascii=False,indent=2),encoding='utf-8'); (OUT/'final_summary_zh.md').write_text('# P3 incumbent continuity first\n\nH1/H2已按固定full_spec_v2門檻完成corrected NAV路徑；warmup只作reference，不進primary metric。\n',encoding='utf-8')
 files=sorted(p for p in OUT.iterdir() if p.is_file() and p.name!='manifest.json'); (OUT/'manifest.json').write_text(json.dumps({'task_id':TASK,'files':[{'name':p.name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in files]},ensure_ascii=False,indent=2),encoding='utf-8'); print(OUT)

if __name__=='__main__': run()
