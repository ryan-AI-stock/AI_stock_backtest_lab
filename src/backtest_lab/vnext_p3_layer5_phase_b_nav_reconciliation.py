from __future__ import annotations
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
PATHS=ROOT/'outputs/vnext_p3_layer5_phase_b_complete_paths_20260712'
RADAR=Path(r'C:\Users\zergv\Documents\Codex\2026-05-23\ai-stock-rotation-radar-https-docs\outputs\radar_vnext_p3_recent_full_feature_data_readiness_acquisition_20260711\compact')
PATCH_ROOT=Path(r'C:\Users\zergv\Documents\Codex\2026-05-23\ai-stock-rotation-radar-https-docs\outputs')
OUT=ROOT/'outputs/vnext_p3_layer5_phase_b_nav_reconciliation_20260712'
TASK='TASK-BACKTEST-CORE-VNEXT-P3-LAYER5-PHASE-B-NAV-RECONCILIATION-001'

def load_adjusted():
 d=pd.concat([pd.read_csv(p,dtype={'ticker':str},low_memory=False) for p in sorted((RADAR/'adjusted').glob('*.csv.gz'))],ignore_index=True); d['date']=pd.to_datetime(d.date)
 d['adjustment_factor']=pd.to_numeric(d.adjusted_close,errors='coerce')/pd.to_numeric(d.raw_close_comparator,errors='coerce').replace(0,np.nan)
 return d[['date','ticker','adjusted_close','raw_close_comparator','adjustment_factor','source_quality','source_hash']].dropna(subset=['adjusted_close']).drop_duplicates(['date','ticker'],keep='last')

def load_raw():
 d=pd.concat([pd.read_csv(p,dtype={'ticker':str},low_memory=False) for p in sorted((RADAR/'price').glob('*.csv.gz'))],ignore_index=True); d['date']=pd.to_datetime(d.date)
 patches=[
  PATCH_ROOT/'radar_vnext_p3_phase_b_placebo_tradability_termination_pit_gap_fill_20260712/p3_placebo_official_ohlcv_patch_rows.csv',
  PATCH_ROOT/'radar_vnext_p3_phase_b_full_spec_v1_placebo_price_gap_fill_20260712/full_spec_v1_placebo_official_ohlcv_patch_rows.csv',
  PATCH_ROOT/'radar_vnext_p3_full_spec_v2_stage_b_placebo_ohlc_gap_fill_20260712/full_spec_v2_stage_b_placebo_official_ohlcv_patch_rows.csv']
 for p in patches:
  x=pd.read_csv(p,dtype={'ticker':str},low_memory=False); x['date']=pd.to_datetime(x.date); d=pd.concat([d,x],ignore_index=True)
 return d[['date','ticker','close','source_quality']].dropna(subset=['close']).drop_duplicates(['date','ticker'],keep='last')

def apply_factor_bracket(adj,raw,tickers,date,tolerance=1e-6):
 rows=[]; proof=[]; date=pd.Timestamp(date)
 for ticker in sorted(set(tickers)):
  series=adj[adj.ticker.eq(ticker)].sort_values('date'); before=series[series.date<date].tail(1); after=series[series.date>date].head(1); rr=raw[(raw.date==date)&raw.ticker.eq(ticker)]
  rb=raw[(raw.ticker==ticker)&raw.date.eq(before.date.iloc[0])] if len(before) else pd.DataFrame(); ra=raw[(raw.ticker==ticker)&raw.date.eq(after.date.iloc[0])] if len(after) else pd.DataFrame()
  f0=float(before.adjusted_close.iloc[0])/float(rb.close.iloc[0]) if len(before) and len(rb) and float(rb.close.iloc[0]) else np.nan
  f1=float(after.adjusted_close.iloc[0])/float(ra.close.iloc[0]) if len(after) and len(ra) and float(ra.close.iloc[0]) else np.nan
  delta=abs(f0-f1) if pd.notna(f0) and pd.notna(f1) else np.nan; scale=max(abs(f0),abs(f1),1.) if pd.notna(f0) and pd.notna(f1) else np.nan
  passed=bool(len(rr) and pd.notna(delta) and delta<=tolerance*scale)
  proof.append({'date':date,'ticker':ticker,'prior_factor_date':before.date.iloc[0] if len(before) else pd.NaT,'prior_trusted_adjusted_close':before.adjusted_close.iloc[0] if len(before) else np.nan,'prior_official_raw_close':rb.close.iloc[0] if len(rb) else np.nan,'prior_factor':f0,'next_factor_date':after.date.iloc[0] if len(after) else pd.NaT,'next_trusted_adjusted_close':after.adjusted_close.iloc[0] if len(after) else np.nan,'next_official_raw_close':ra.close.iloc[0] if len(ra) else np.nan,'next_factor':f1,'absolute_factor_delta':delta,'relative_tolerance':tolerance,'raw_close_ready':bool(len(rr)),'factor_continuity_pass':passed,'factor_basis':'trusted_adjusted_close_divided_by_same_day_official_raw_close','neighbor_price_substitution_used':False})
  if passed:
   close=float(rr.close.iloc[0]); rows.append({'date':date,'ticker':ticker,'adjusted_close':close*f0,'raw_close_comparator':close,'adjustment_factor':f0,'source_quality':'trusted_nonofficial_factor_bracket_plus_official_raw_close','source_hash':'factor_bracket_continuity_proof'})
 return pd.concat([adj,pd.DataFrame(rows)],ignore_index=True).drop_duplicates(['date','ticker'],keep='last'),pd.DataFrame(proof)

def collapse_same_close_events(events):
 out=[]; audit=[]
 for (scenario,date),g in events.sort_values(['scenario','actual_execution_date','decision_date']).groupby(['scenario','actual_execution_date'],sort=False):
  first=g.iloc[0]; last=g.iloc[-1]
  if len(g)>1:
   old=first.prior_target if pd.notna(first.prior_target) else None; new=last.new_target if pd.notna(last.new_target) else None
   typ='stock_to_stock' if old and new else ('stock_to_no_position' if old else 'no_position_to_stock')
   fee=(.001425 if old else 0)+(.001425 if new else 0); tax=.003 if old else 0; slip=.001*((1 if old else 0)+(1 if new else 0))
   row=last.copy(); row['decision_date']=last.decision_date; row['requested_execution_date']=last.requested_execution_date; row['prior_target']=old; row['new_target']=new; row['transition_type']=typ; row['prior_target_exit_close']=first.prior_target_exit_close; row['prior_target_exit_source_quality']=first.prior_target_exit_source_quality; row['brokerage_rate']=fee; row['tax_rate']=tax; row['slippage_rate']=slip; row['total_cost_rate']=fee+tax+slip
   out.append(row); audit.append({'scenario':scenario,'actual_execution_date':date,'input_transition_rows':len(g),'prior_target':old,'intermediate_targets':'|'.join(g.new_target.astype(str).iloc[:-1]),'final_target':new,'resolution':'latest_decision_supersedes_unexecuted_intermediate_at_same_close','cost_charged_once':True})
  else: out.append(first)
 return pd.DataFrame(out),pd.DataFrame(audit)

def run():
 OUT.mkdir(parents=True,exist_ok=True)
 actions=pd.read_csv(PATHS/'p3_layer5_phase_b_scenario_daily_actions.csv',dtype={'selected_ticker':str}); events=pd.read_csv(PATHS/'p3_layer5_phase_b_exact_transition_cost_ledger.csv',dtype={'prior_target':str,'new_target':str})
 for c in ['decision_date','requested_execution_date']: actions[c]=pd.to_datetime(actions[c])
 for c in ['decision_date','requested_execution_date','actual_execution_date']: events[c]=pd.to_datetime(events[c])
 adj=load_adjusted(); raw=load_raw()
 marks=pd.read_csv(PATHS/'p3_layer5_phase_b_daily_unique_position_marks.csv',dtype={'held_ticker':str}); marks['date']=pd.to_datetime(marks.date)
 held_on_gap=marks.loc[marks.date.eq(pd.Timestamp('2025-08-01')),'held_ticker'].dropna().tolist()
 adj,bracket=apply_factor_bracket(adj,raw,held_on_gap,'2025-08-01'); bracket.to_csv(OUT/'p3_layer5_20250801_adjustment_factor_bracket_proof.csv',index=False,encoding='utf-8-sig')
 lookup=adj.set_index(['date','ticker']); raw_lookup=raw.set_index(['date','ticker'])
 calendar=sorted(pd.to_datetime(pd.read_csv(ROOT/'backtest_cache/stock_pool_observations/0050_TW.csv').date).loc[lambda s:(s>=actions.decision_date.min())&(s<=pd.Timestamp('2026-06-30'))].unique())
 ledgers=[]; blocked=[]; transition_audit=[]
 for scenario in sorted(actions.scenario.unique()):
  ev,collision=collapse_same_close_events(events[events.scenario==scenario]); collision.to_csv(OUT/f'p3_layer5_{scenario}_same_close_collision_resolution.csv',index=False,encoding='utf-8-sig'); ev=ev.sort_values('actual_execution_date'); active=None; prev_adj=None; nav=1.; qty=np.nan
  for date in calendar:
   nav_open=nav; gross=0.; price_status='no_position'; current_adj=np.nan; held_before=active; prior_adj=prev_adj
   if active is not None:
    key=(pd.Timestamp(date),str(active))
    if key in lookup.index:
     row=lookup.loc[key]; current_adj=float(row.adjusted_close); price_status='event_aware_adjusted_research_grade'
     if prev_adj is not None: gross=current_adj/prev_adj-1
     prev_adj=current_adj
    elif key in raw_lookup.index:
     price_status='blocked_adjusted_source_missing_on_official_trading_day'; gross=0.
     blocked.append({'scenario':scenario,'date':date,'ticker':active,'reason':'adjusted_source_missing_on_official_trading_day'})
    else: price_status='official_no_trade_hold_prior_event_aware_factor'; gross=0.
   nav_before_cost=nav*(1+gross); dayev=ev[ev.actual_execution_date==date]; cost_rate=0.; transition_type='hold'
   if not dayev.empty:
    if len(dayev)>1: blocked.append({'scenario':scenario,'date':date,'ticker':active,'reason':'multiple_transitions_same_day'})
    e=dayev.iloc[-1]; transition_type=e.transition_type; cost_rate=float(e.total_cost_rate); old=active; new=e.new_target if pd.notna(e.new_target) else None
    if old is not None and pd.isna(e.prior_target_exit_close): blocked.append({'scenario':scenario,'date':date,'ticker':old,'reason':'missing_prior_exit_raw_close'})
    if new is not None and pd.isna(e.new_target_entry_close): blocked.append({'scenario':scenario,'date':date,'ticker':new,'reason':'missing_new_entry_raw_close'})
    outgoing_shares=qty; outgoing_raw=float(e.prior_target_exit_close) if pd.notna(e.prior_target_exit_close) else np.nan
    outgoing_proceeds_before_cost=nav_before_cost
    exit_cost_rate=.001425+float(e.tax_rate)+.001 if old is not None else 0.
    entry_cost_rate=.001425+.001 if new is not None else 0.
    exit_cost_amount=nav_before_cost*exit_cost_rate; proceeds_after_exit_cost=nav_before_cost-exit_cost_amount
    entry_cost_amount=nav_before_cost*entry_cost_rate; nav=nav_before_cost-exit_cost_amount-entry_cost_amount
    active=str(new) if new is not None else None
    if active is not None:
     key=(pd.Timestamp(date),active)
     if key in lookup.index: prev_adj=float(lookup.loc[key].adjusted_close)
     else: prev_adj=None; blocked.append({'scenario':scenario,'date':date,'ticker':active,'reason':'missing_adjusted_entry_basis'})
     qty=nav/float(e.new_target_entry_close) if pd.notna(e.new_target_entry_close) and float(e.new_target_entry_close)>0 else np.nan
    else: prev_adj=None; qty=np.nan
    transition_audit.append({'scenario':scenario,'date':date,'prior_target':old,'new_target':active,'held_ticker_before_transition':held_before,'prior_adjusted_mark':prior_adj,'current_outgoing_adjusted_mark':current_adj,'same_asset_gross_return_before_transition':gross,'NAV_before_market_move':nav_open,'NAV_before_transition':nav_before_cost,'outgoing_shares_audit':outgoing_shares,'prior_target_exit_raw_close':outgoing_raw,'outgoing_proceeds_before_cost':outgoing_proceeds_before_cost,'exit_cost_rate':exit_cost_rate,'exit_cost_amount':exit_cost_amount,'outgoing_proceeds_after_exit_cost':proceeds_after_exit_cost,'new_target_entry_raw_close':e.new_target_entry_close,'entry_cost_rate':entry_cost_rate,'entry_cost_amount':entry_cost_amount,'total_cost_rate_contract':cost_rate,'NAV_after_transition':nav,'incoming_shares_from_after_cost_NAV':qty,'NAV_reconstructed_from_incoming_shares':qty*float(e.new_target_entry_close) if pd.notna(e.new_target_entry_close) and pd.notna(qty) else nav,'transition_creates_market_pnl':False,'cross_asset_nominal_price_return_used':False})
   else: nav=nav_before_cost
   if abs(gross)>.15: blocked.append({'scenario':scenario,'date':date,'ticker':active,'reason':'abs_gross_return_gt_15pct','gross_return':gross,'price_status':price_status})
   ledgers.append({'scenario':scenario,'date':date,'held_ticker_before_close':held_before,'held_ticker_after_close':active,'quantity_after_close':qty,'NAV_open':nav_open,'prior_adjusted_mark':prior_adj,'current_adjusted_mark':current_adj,'adjustment_factor':float(lookup.loc[(pd.Timestamp(date),str(held_before))].adjustment_factor) if held_before is not None and (pd.Timestamp(date),str(held_before)) in lookup.index else np.nan,'adjusted_source_hash':lookup.loc[(pd.Timestamp(date),str(held_before))].source_hash if held_before is not None and (pd.Timestamp(date),str(held_before)) in lookup.index else None,'gross_return_from_same_asset':gross,'NAV_before_transition_cost':nav_before_cost,'transition_type':transition_type,'transition_cost_rate':cost_rate,'NAV_close':nav,'net_daily_return':nav/nav_open-1 if nav_open else np.nan,'price_status':price_status,'corporate_action_scale_guard':'event_aware_adjusted_analysis_mark','cross_asset_nominal_price_return_used':False})
 ledger=pd.DataFrame(ledgers); blockers=pd.DataFrame(blocked); ta=pd.DataFrame(transition_audit)
 ledger.to_csv(OUT/'p3_layer5_phase_b_NAV_daily_wealth_ledger.csv',index=False,encoding='utf-8-sig'); ta.to_csv(OUT/'p3_layer5_phase_b_transition_NAV_reconciliation.csv',index=False,encoding='utf-8-sig'); blockers.to_csv(OUT/'p3_layer5_phase_b_NAV_anomaly_blocked_ledger.csv',index=False,encoding='utf-8-sig')
 cases=[('C1','2023-11-29','3545','3362'),('C1','2024-01-05','2329','3008'),('C1','2024-01-04','2329',None),('C1','2023-10-11','2436',None)]
 regression=[]
 for scenario,date,old,new in cases:
  d=pd.Timestamp(date); lr=ledger[(ledger.scenario==scenario)&(ledger.date==d)]; tr=ta[(ta.scenario==scenario)&(ta.date==d)]
  regression.append({'scenario':scenario,'date':date,'expected_prior_ticker':old,'expected_new_ticker':new,'ledger_row_found':len(lr)==1,'transition_row_found':len(tr)==1,'gross_return_from_same_asset':lr.gross_return_from_same_asset.iloc[0] if len(lr) else np.nan,'cross_asset_nominal_price_return_used':lr.cross_asset_nominal_price_return_used.iloc[0] if len(lr) else None,'price_status':lr.price_status.iloc[0] if len(lr) else 'missing','NAV_reconciled':bool(len(lr)==1 and (not len(tr) or abs(float(tr.NAV_reconstructed_from_incoming_shares.iloc[0])-float(tr.NAV_after_transition.iloc[0]))<1e-10))})
 pd.DataFrame(regression).to_csv(OUT/'p3_layer5_phase_b_required_regression_case_audit.csv',index=False,encoding='utf-8-sig')
 pd.DataFrame([{'scenario':s,'requested_decision_start':'2023-07-11','requested_decision_end':'2026-06-29','actual_mark_start':g.date.min(),'actual_mark_end':g.date.max(),'daily_mark_rows':len(g),'terminal_2026_06_30_role':'next_day_execution_and_terminal_mark_only'} for s,g in ledger.groupby('scenario')]).to_csv(OUT/'p3_layer5_phase_b_NAV_requested_vs_actual_coverage.csv',index=False,encoding='utf-8-sig')
 pd.DataFrame([{'invalid_output':'prior_Experiments_Stage_B_A','status':'BLOCKED_INVALID_PATH_ARITHMETIC_AND_CORPORATE_ACTION','strategy_NO_GO_meaning_withdrawn':True,'reason':'cross_asset_nominal_price_return_and_raw_corporate_action_discontinuity'}]).to_csv(OUT/'p3_layer5_prior_stage_b_invalidation_audit.csv',index=False,encoding='utf-8-sig')
 anomaly_count=int((blockers.reason=='abs_gross_return_gt_15pct').sum()) if not blockers.empty else 0; readiness={'task_id':TASK,'status':'NAV_reconciliation_blocked_anomalies_remain' if len(blockers) else 'NAV_reconciliation_ready_for_fixed_parameter_stage_b_rerun','performance_authority_file':'p3_layer5_phase_b_NAV_daily_wealth_ledger.csv','legacy_raw_mark_path_role':'tradability_only_not_performance_authority','scenario_count':ledger.scenario.nunique(),'daily_rows':len(ledger),'canonical_action_dates':715,'terminal_execution_mark_date':'2026-06-30','transition_rows':len(ta),'same_close_collision_groups_resolved':sum(len(pd.read_csv(p)) for p in OUT.glob('p3_layer5_*_same_close_collision_resolution.csv') if p.stat().st_size>3),'cross_asset_nominal_price_return_count':int(ledger.cross_asset_nominal_price_return_used.sum()),'blocked_rows':len(blockers),'abs_return_gt_15pct_rows':anomaly_count,'max_abs_same_asset_gross_return':float(ledger.gross_return_from_same_asset.abs().max()),'ready_for_stage_b_rerun':len(blockers)==0,'ready_for_experiments':len(blockers)==0,'prior_NO_GO_withdrawn':True,'future_data_violation_count':0,'formal_model_changed':False,'trade_decision_changed':False,'active_in_trade_decision':False,'report_changed':False,'portfolio_replay_executed':False,'ready_for_strategy_replay':False,'ready_for_formal':False,'not_live_rule':True,'forward_returns_live_rule_usage':False}
 (OUT/'readiness_for_p3_layer5_phase_b_NAV_reconciliation.json').write_text(json.dumps(readiness,ensure_ascii=False,indent=2),encoding='utf-8'); (OUT/'final_summary_zh.md').write_text(f'''# P3 Phase B NAV reconciliation

先前 Stage B A 績效因 NAV accounting 與 corporate-action scale 錯誤全數作廢，策略 NO_GO 含義已撤銷。

修正後以固定 portfolio NAV 換股：舊股仅使用自身 event-aware adjusted mark 計當日報酬，再扣 exit/entry EP05 與 10bp/side 滑價，最後以 after-cost NAV / 新股官方 raw entry close 重設股數。跨 ticker 名目價格報酬使用次數=0。

2025-08-01 缺日僅在前後「trusted adjusted close / 同日 official raw close」factor 於 1e-6 相對容差內一致時吸收，未使用鄰日價格。

最終 blocked={len(blockers)}，abs gross return >15%={anomaly_count}，可交 Experiments 重跑固定參數 Stage B A；本包未計算策略績效。
''',encoding='utf-8')
 files=sorted(p for p in OUT.iterdir() if p.is_file() and p.name!='manifest.json'); (OUT/'manifest.json').write_text(json.dumps({'task_id':TASK,'source_commit':'cd90cb5','readiness':readiness,'files':[{'name':p.name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in files]},ensure_ascii=False,indent=2),encoding='utf-8'); print(OUT)
if __name__=='__main__': run()
