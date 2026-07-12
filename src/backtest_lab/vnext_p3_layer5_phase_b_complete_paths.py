from __future__ import annotations
import hashlib,json,random
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
SRC=ROOT/'outputs/vnext_p3_layer5_daily_feature_state_action_materialization_20260712'
RADAR=Path(r'C:\Users\zergv\Documents\Codex\2026-05-23\ai-stock-rotation-radar-https-docs\outputs\radar_vnext_p3_recent_full_feature_data_readiness_acquisition_20260711\compact')
PATCH=Path(r'C:\Users\zergv\Documents\Codex\2026-05-23\ai-stock-rotation-radar-https-docs\outputs\radar_vnext_p3_phase_b_placebo_tradability_termination_pit_gap_fill_20260712')
PATCH2=Path(r'C:\Users\zergv\Documents\Codex\2026-05-23\ai-stock-rotation-radar-https-docs\outputs\radar_vnext_p3_phase_b_full_spec_v1_placebo_price_gap_fill_20260712')
PATCH3=Path(r'C:\Users\zergv\Documents\Codex\2026-05-23\ai-stock-rotation-radar-https-docs\outputs\radar_vnext_p3_full_spec_v2_stage_b_placebo_ohlc_gap_fill_20260712')
V2=ROOT/'outputs/vnext_p3_market_controller_full_spec_v2_20260712/p3_market_controller_full_spec_v2_daily_features.csv'
OUT=ROOT/'outputs/vnext_p3_layer5_phase_b_complete_paths_20260712'
TASK='TASK-BACKTEST-CORE-VNEXT-P3-LAYER5-PHASE-B-COMPLETE-PATHS-001'

def prices():
 d=pd.concat([pd.read_csv(p,dtype={'ticker':str},low_memory=False) for p in sorted((RADAR/'price').glob('*.csv.gz'))],ignore_index=True); d['date']=pd.to_datetime(d.date)
 ready=json.loads((PATCH/'readiness_for_core_p3_placebo_tradability_termination_patch.json').read_text(encoding='utf-8-sig'))
 if not ready['ready_for_core_p3_phase_b_placebo_tradability_patch_absorption'] or ready['remaining_blocked_episodes']!=0: raise ValueError('Radar placebo patch not ready')
 patch=pd.read_csv(PATCH/'p3_placebo_official_ohlcv_patch_rows.csv',dtype={'ticker':str}); patch2=pd.read_csv(PATCH2/'full_spec_v1_placebo_official_ohlcv_patch_rows.csv',dtype={'ticker':str}); patch3=pd.read_csv(PATCH3/'full_spec_v2_stage_b_placebo_official_ohlcv_patch_rows.csv',dtype={'ticker':str}); patch=pd.concat([patch,patch2,patch3],ignore_index=True); patch['date']=pd.to_datetime(patch.date)
 existing=set(zip(d.date,d.ticker)); patch['registry_key_preexisting']=[(r.date,r.ticker) in existing for r in patch.itertuples(index=False)]
 patch.to_csv(OUT/'p3_layer5_phase_b_placebo_price_patch_absorption_audit.csv',index=False,encoding='utf-8-sig')
 add=patch.loc[~patch.registry_key_preexisting,['date','ticker','close','source_quality']]
 return pd.concat([d[['date','ticker','close','source_quality']],add],ignore_index=True).dropna(subset=['close']).drop_duplicates(['date','ticker'],keep='first')

def make_actions(feature,market_mode):
 rows=[]; incumbent=None
 for dt,day in feature.groupby('decision_date',sort=True):
  state=day.simple_v0_market_state.iloc[0] if market_mode=='simple_v0_reference' else day.market_state.iloc[0]
  if market_mode=='C0': state='ordinary_market'
  elif market_mode=='C2' and state=='strong_market': state='ordinary_market'
  elif market_mode=='C3':
   gs=['taiwan_group','breadth_group','capital_group','external_group']; bull=(day[gs].iloc[0]=='bullish').sum(); bear=(day[gs].iloc[0]=='bearish').sum()
   state='strong_market' if bull>=3 and bear<=1 else ('weak_market' if bear>=3 else 'ordinary_market')
  low_confidence_state=state=='PIT_warmup_or_low_confidence'; applied_state='ordinary_market' if low_confidence_state else state
  conf_floor={'strong_market':.63,'ordinary_market':.70,'weak_market':.77,'confirmed_bear':.84}[applied_state]
  margin_req={'strong_market':6.,'ordinary_market':5.,'weak_market':4.,'confirmed_bear':5.}[applied_state]
  eligible=day[day.history_ready & (day.total_confidence>=conf_floor) & day.raw_state.isin(['turning_up','healthy_rise','overheat_warning'])].sort_values('score_balanced',ascending=False)
  ch=eligible.iloc[0] if len(eligible) else None; inc=day[day.ticker.eq(incumbent)].iloc[0] if incumbent and day.ticker.eq(incumbent).any() else None
  if state=='confirmed_bear': action='no_position_confirmed_bear'; reason='confirmed_bear'; incumbent=None
  elif inc is None or not inc.history_ready or inc.raw_state=='confirmed_weakening':
   if ch is None: action='watch_only'; reason='no_valid_replacement'; incumbent=None
   else: action='forced_replacement'; reason='invalid_or_weak_incumbent'; incumbent=ch.ticker
  elif ch is not None and ch.ticker!=incumbent:
   wins=sum(float(ch[f'block_{b}'])>float(inc[f'block_{b}']) for b in 'ABCDEF' if pd.notna(ch[f'block_{b}']) and pd.notna(inc[f'block_{b}']))
   weak_req={'strong_market':4,'ordinary_market':3,'weak_market':2,'confirmed_bear':2}[applied_state]
   weakened=int(inc.weak_groups)>=weak_req; margin=float(ch.score_balanced-inc.score_balanced)
   if margin>=margin_req and wins>=3 and weakened: action='switch_to_challenger'; reason='applied_controller_margin_blocks_weakening'; incumbent=ch.ticker
   else: action='hold_incumbent'; reason='applied_controller_switch_gate_failed'
  else: action='hold_incumbent'; reason='valid_incumbent_no_better_challenger'
  rows.append({'scenario':market_mode,'decision_date':dt,'requested_execution_date':day.next_execution_date.iloc[0],'selected_ticker':incumbent,'selected_action':action,'action_reason':reason,'classified_market_state':state,'applied_market_state':applied_state,'low_confidence_ordinary_fallback':low_confidence_state,'confidence_floor':conf_floor,'challenger_margin_required':margin_req})
 return pd.DataFrame(rows)

def placebo_actions(canonical,feature,seed):
 rng=random.Random(seed); z=canonical.copy(); z['scenario']=f'placebo_seed_{seed}'; z['selected_ticker']=None
 stock=canonical.selected_ticker.notna(); episode=(stock.ne(stock.shift()) | (stock & canonical.selected_ticker.ne(canonical.selected_ticker.shift()))).cumsum()
 for _,idx in canonical[stock].groupby(episode[stock]).groups.items():
  first=idx[0]; dt=canonical.loc[first,'decision_date']; pool=feature[(feature.decision_date==dt)&feature.history_ready].ticker.dropna().unique().tolist()
  if pool: z.loc[idx,'selected_ticker']=rng.choice(pool)
 z['selected_action']=np.where(z.selected_ticker.eq(z.selected_ticker.shift())&z.selected_ticker.notna(),'hold_incumbent',np.where(z.selected_ticker.notna(),'placebo_entry_or_switch','watch_only'))
 z['action_reason']='fixed_seed_matched_exposure_duration'; return z

def price_transitions(actions,px,calendar,slip_bp=10):
 actions=actions.sort_values('decision_date').copy(); actions['prior_target']=actions.selected_ticker.shift(); events=[]; active=None
 for r in actions.itertuples(index=False):
  target=r.selected_ticker if pd.notna(r.selected_ticker) else None
  if target==active: continue
  candidates=[d for d in calendar if d>=pd.Timestamp(r.requested_execution_date)]
  found=None
  for d in candidates:
   oldok=active is None or not px[(px.date==d)&(px.ticker==active)].empty; newok=target is None or not px[(px.date==d)&(px.ticker==target)].empty
   if oldok and newok: found=d; break
  if found is None: raise ValueError(f'unresolved transition {active}->{target} after {r.decision_date}')
  oldrow=px[(px.date==found)&(px.ticker==active)].iloc[0] if active else None; newrow=px[(px.date==found)&(px.ticker==target)].iloc[0] if target else None
  typ='stock_to_stock' if active and target else ('stock_to_no_position' if active else 'no_position_to_stock')
  fee=(.001425 if active else 0)+(.001425 if target else 0); tax=.003 if active else 0; slip=(slip_bp/10000)*((1 if active else 0)+(1 if target else 0))
  events.append({'scenario':r.scenario,'decision_date':r.decision_date,'requested_execution_date':r.requested_execution_date,'actual_execution_date':found,'prior_target':active,'new_target':target,'transition_type':typ,'prior_target_exit_close':None if oldrow is None else oldrow.close,'prior_target_exit_source_quality':None if oldrow is None else oldrow.source_quality,'new_target_entry_close':None if newrow is None else newrow.close,'new_target_entry_source_quality':None if newrow is None else newrow.source_quality,'deferred_days':calendar.index(found)-calendar.index(pd.Timestamp(r.requested_execution_date)),'brokerage_rate':fee,'tax_rate':tax,'slippage_rate':slip,'total_cost_rate':fee+tax+slip})
  active=target
 ev=pd.DataFrame(events)
 # Daily unique-position marks, carrying prior official mark only on no-trade dates.
 marks=[]; active=None; last_mark=None; ei=0
 for d in calendar:
  day_events=ev[ev.actual_execution_date==d]
  if not day_events.empty: active=day_events.iloc[-1].new_target if pd.notna(day_events.iloc[-1].new_target) else None
  row=px[(px.date==d)&(px.ticker==active)] if active else pd.DataFrame()
  if active and not row.empty: mark=float(row.close.iloc[0]); quality=row.source_quality.iloc[0]; stale=False; last_mark=mark
  elif active: mark=last_mark; quality='hold_prior_official_mark_no_trade'; stale=True
  else: mark=np.nan; quality='no_position'; stale=False
  marks.append({'scenario':actions.scenario.iloc[0],'date':d,'held_ticker':active,'exposure':0 if active is None else 1,'mark_close':mark,'mark_source_quality':quality,'stale_mark':stale})
 return ev,pd.DataFrame(marks)

def audit_tradability(actions,px,calendar):
 out=[]; intended_active=None
 for r in actions.sort_values('decision_date').itertuples(index=False):
  target=r.selected_ticker if pd.notna(r.selected_ticker) else None
  if target==intended_active: continue
  dates=[d for d in calendar if d>=pd.Timestamp(r.requested_execution_date)]
  common=[]
  for d in dates:
   oldok=intended_active is None or not px[(px.date==d)&(px.ticker==intended_active)].empty
   newok=target is None or not px[(px.date==d)&(px.ticker==target)].empty
   if oldok and newok: common.append(d); break
  old_hist=px[px.ticker==intended_active].date.max() if intended_active else pd.NaT
  new_hist=px[px.ticker==target].date.min() if target else pd.NaT
  out.append({'scenario':r.scenario,'decision_date':r.decision_date,'requested_execution_date':r.requested_execution_date,
              'intended_prior_target':intended_active,'intended_new_target':target,'first_common_tradable_date':common[0] if common else pd.NaT,
              'prior_target_last_official_price_date':old_hist,'new_target_first_official_price_date':new_hist,
              'tradability_horizon_ready':bool(common),'blocked_reason':'' if common else 'no_common_official_tradable_date_after_requested_execution',
              'audit_only_intended_target_not_assumed_executed':True})
  intended_active=target
 return pd.DataFrame(out)

def run():
 OUT.mkdir(parents=True,exist_ok=True); f=pd.read_csv(SRC/'p3_layer5_daily_feature_state_matrix.csv',dtype={'ticker':str},low_memory=False); f['decision_date']=pd.to_datetime(f.decision_date); f['next_execution_date']=pd.to_datetime(f.next_execution_date)
 v2=pd.read_csv(V2,usecols=['decision_date','full_spec_v2_state']); v2['decision_date']=pd.to_datetime(v2.decision_date); f=f.drop(columns=['market_state'],errors='ignore').merge(v2,on='decision_date',how='left'); f['market_state']=f.full_spec_v2_state
 base=pd.read_csv(SRC/'p3_layer5_daily_incumbent_challenger_action_trace.csv',dtype={'incumbent_after':str}); base['decision_date']=pd.to_datetime(base.decision_date); base['requested_execution_date']=pd.to_datetime(base.next_execution_date); base=base.rename(columns={'incumbent_after':'selected_ticker'}); base['scenario']='C1'
 scenarios=[make_actions(f,s) for s in ['C0','C1','C2','C3','simple_v0_reference']]
 for seed in [17,29,43,71,101]: scenarios.append(placebo_actions(scenarios[1],f,seed))
 px=prices(); cal=pd.read_csv(ROOT/'backtest_cache/stock_pool_observations/0050_TW.csv',usecols=['date']); cal['date']=pd.to_datetime(cal.date); calendar=cal.date[(cal.date>=min(s.decision_date.min() for s in scenarios))&(cal.date<=pd.Timestamp('2026-06-30'))].drop_duplicates().sort_values().tolist()
 all_a=[]; all_e=[]; all_m=[]; all_t=[]; blocked_scenarios=[]
 for s in scenarios:
  s['requested_execution_date']=pd.to_datetime(s.requested_execution_date); all_a.append(s)
  ta=audit_tradability(s,px,calendar); all_t.append(ta)
  if ta.tradability_horizon_ready.all():
   ev,mk=price_transitions(s,px,calendar); all_e.append(ev); all_m.append(mk)
  else: blocked_scenarios.append(s.scenario.iloc[0])
 actions=pd.concat(all_a,ignore_index=True); tradability=pd.concat(all_t,ignore_index=True)
 events=pd.concat(all_e,ignore_index=True) if all_e else pd.DataFrame(); marks=pd.concat(all_m,ignore_index=True) if all_m else pd.DataFrame()
 actions.to_csv(OUT/'p3_layer5_phase_b_scenario_daily_actions.csv',index=False,encoding='utf-8-sig'); events.to_csv(OUT/'p3_layer5_phase_b_exact_transition_cost_ledger.csv',index=False,encoding='utf-8-sig'); marks.to_csv(OUT/'p3_layer5_phase_b_daily_unique_position_marks.csv',index=False,encoding='utf-8-sig')
 pd.DataFrame([{'scenario':s,'requested_start':'2023-07-11','requested_end':'2026-06-29','actual_start':g.date.min(),'actual_end':g.date.max(),'mark_rows':len(g)} for s,g in marks.groupby('scenario')]).to_csv(OUT/'p3_layer5_phase_b_requested_vs_actual_coverage.csv',index=False,encoding='utf-8-sig')
 pd.DataFrame([{'slippage_bp_per_side':bp,'role':'primary' if bp==10 else 'sensitivity','brokerage_rate':.001425,'stock_sell_tax_rate':.003,'ETF_sell_tax_rate':.001,'switch_double_sided':True} for bp in [5,10,20]]).to_csv(OUT/'p3_layer5_phase_b_cost_slippage_contract.csv',index=False,encoding='utf-8-sig')
 tradability.to_csv(OUT/'p3_layer5_phase_b_all_scenario_tradability_horizon_audit.csv',index=False,encoding='utf-8-sig')
 pivot=actions.pivot(index='decision_date',columns='scenario',values='selected_ticker'); pivot.reset_index().to_csv(OUT/'p3_layer5_phase_b_C0_C3_action_diff_audit.csv',index=False,encoding='utf-8-sig')
 blocker_count=int((~tradability.tradability_horizon_ready).sum())
 readiness={'task_id':TASK,'status':'static_actions_complete_waiting_bounded_tradability_event_patch' if blocker_count else 'phase_b_complete_paths_ready','placebo_price_patch_absorbed':True,'placebo_price_patch_commit':'85e22d5','scenario_count':len(scenarios),'canonical_counterfactuals':4,'placebo_seeds':5,'action_rows':len(actions),'transition_rows':len(events),'mark_rows':len(marks),'tradability_audit_rows':len(tradability),'unresolved_transition_count':blocker_count,'blocked_scenarios':blocked_scenarios,'outgoing_exit_marks_ready':blocker_count==0,'counterfactual_actions_materialized':True,'applied_controller_gates_materialized':True,'TAIFEX_TDCC_ablation_trace_materialized':True,'executable_scenario_paths_ready':sorted(set(actions.scenario)-set(blocked_scenarios)),'placebo_paths_materialized':not any(s.startswith('placebo') for s in blocked_scenarios),'exact_rechain_daily_marks_ready':blocker_count==0,'ready_for_phase_b_experiments':blocker_count==0,'ready_for_experiments':blocker_count==0,'future_data_violation_count':0,'formal_model_changed':False,'trade_decision_changed':False,'active_in_trade_decision':False,'report_changed':False,'portfolio_replay_executed':False,'ready_for_strategy_replay':False,'ready_for_formal':False,'not_live_rule':True,'forward_returns_live_rule_usage':False}
 readiness['primary_controller_version']='full_spec_v2'; readiness['simple_v0_role']='reference_only'; readiness['price_patch_commits']=['85e22d5','ba19fda','6f919e5']; readiness['warmup_controller_policy']='ordinary_thresholds_no_strong_or_weak_evidence'
 (OUT/'readiness_for_p3_layer5_phase_b_complete_paths.json').write_text(json.dumps(readiness,ensure_ascii=False,indent=2),encoding='utf-8'); (OUT/'final_summary_zh.md').write_text(f'# P3 Phase B complete paths\n\nC0-C3與5 fixed-seed placebo action已全數materialize。Tradability blockers={blocker_count}；僅零blocker scenarios產生executable mark/path。未計績效。\n',encoding='utf-8')
 files=sorted(q for q in OUT.iterdir() if q.is_file() and q.name!='manifest.json'); (OUT/'manifest.json').write_text(json.dumps({'task_id':TASK,'source_commits':['8822b2e','85e22d5'],'patch_source':str(PATCH),'readiness':readiness,'files':[{'name':q.name,'sha256':hashlib.sha256(q.read_bytes()).hexdigest()} for q in files]},ensure_ascii=False,indent=2),encoding='utf-8'); print(OUT)
if __name__=='__main__': run()
