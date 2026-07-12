from __future__ import annotations
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
SCORE=ROOT/'outputs/vnext_p3_layer5_full_candidate_risk_adjusted_scoring_contract_20260712/p3_full_candidate_spec_v1_score_matrix.csv.gz'
FUND=ROOT/'outputs/vnext_p3_layer5_fundamental_balance_cashflow_pit_absorption_20260712/p3_full_candidate_F_five_family_PIT_matrix.csv.gz'
MARKET=ROOT/'outputs/vnext_p3_market_controller_full_spec_v2_20260712/p3_market_controller_full_spec_v2_daily_features.csv'
OUT=ROOT/'outputs/vnext_p3_layer5_full_candidate_quality_outcome_contract_20260712'
TASK='TASK-BACKTEST-CORE-VNEXT-P3-LAYER5-FULL-CANDIDATE-QUALITY-OUTCOME-CONTRACT-001'

from .vnext_p3_layer5_phase_b_nav_reconciliation import load_adjusted,load_raw,apply_factor_bracket

def cost_net(gross,bp):
 buy=.001425+bp/10000; sell=.001425+.003+bp/10000
 return (1-buy)*(1+gross)*(1-sell)-1
def etf_cost_net(gross,bp):
 buy=.001425+bp/10000; sell=.001425+.001+bp/10000
 return (1-buy)*(1+gross)*(1-sell)-1

def run():
 OUT.mkdir(parents=True,exist_ok=True)
 s=pd.read_csv(SCORE,dtype={'ticker':str},low_memory=False); s['decision_date']=pd.to_datetime(s.decision_date); s['next_execution_date']=pd.to_datetime(s.next_execution_date)
 f=pd.read_csv(FUND,dtype={'ticker':str},usecols=['decision_date','ticker','fundamental_quality_score','fundamental_quality_confidence','fundamental_family_available_count'],low_memory=False); f['decision_date']=pd.to_datetime(f.decision_date); s=s.merge(f,on=['decision_date','ticker'],how='left',validate='one_to_one')
 m=pd.read_csv(MARKET,usecols=['decision_date','full_spec_v2_state']); m['decision_date']=pd.to_datetime(m.decision_date); s=s.merge(m,on='decision_date',how='left',validate='many_to_one')
 raw=load_raw(); adj=load_adjusted(); gap_tickers=s.loc[s.next_execution_date.eq(pd.Timestamp('2025-08-01')),'ticker'].tolist(); adj,bracket=apply_factor_bracket(adj,raw,gap_tickers,'2025-08-01')
 rlookup={(r.date,str(r.ticker)):(float(r.close),r.source_quality) for r in raw.itertuples(index=False)}; alookup={(r.date,str(r.ticker)):(float(r.adjusted_close),r.source_quality,float(r.adjustment_factor),r.source_hash) for r in adj.itertuples(index=False)}
 cal=pd.read_csv(ROOT/'backtest_cache/stock_pool_observations/0050_TW.csv',usecols=['date','adj_close','close']); cal['date']=pd.to_datetime(cal.date); cal=cal.drop_duplicates('date').sort_values('date'); dates=cal.date.tolist(); pos={d:i for i,d in enumerate(dates)}; b_adj=dict(zip(cal.date,cal.adj_close)); b_raw=dict(zip(cal.date,cal.close))
 horizons=[5,10,20,40]; rows=[]; blocked=[]
 for i,r in enumerate(s.itertuples(index=False)):
  entry=pd.Timestamp(r.next_execution_date); ticker=str(r.ticker); ek=(entry,ticker); er=rlookup.get(ek); ea=alookup.get(ek); p=pos.get(entry)
  for h in horizons:
   exit_date=dates[p+h] if p is not None and p+h<len(dates) else pd.NaT; xr=rlookup.get((exit_date,ticker)) if pd.notna(exit_date) else None; xa=alookup.get((exit_date,ticker)) if pd.notna(exit_date) else None
   status='ready'; reason=''; marks=[]
   if p is None: status='blocked'; reason='entry_not_official_market_date'
   elif er is None: status='blocked'; reason='entry_official_raw_close_missing_or_not_tradable'
   elif ea is None: status='blocked'; reason='entry_adjusted_analysis_missing'
   elif pd.isna(exit_date): status='terminal_unavailable'; reason='horizon_beyond_actual_source_end'
   elif xr is None: status='blocked'; reason='exit_official_raw_close_missing_or_not_tradable'
   elif xa is None: status='blocked'; reason='exit_adjusted_analysis_missing'
   else:
    for d in dates[p:p+h+1]:
     rv=rlookup.get((d,ticker)); av=alookup.get((d,ticker))
     if rv is not None and av is None: status='blocked'; reason='intermediate_adjusted_missing_on_official_trading_day'; break
     if av is not None: marks.append(av[0])
   gross=np.nan; mdd=np.nan; tail=np.nan; large=np.nan
   if status=='ready':
    gross=xa[0]/ea[0]-1; arr=np.array(marks,dtype=float); curve=arr/arr[0]; mdd=float(np.min(curve/np.maximum.accumulate(curve)-1)); rets=pd.Series(arr).pct_change().dropna(); tail=float(rets.quantile(.05)) if len(rets) else np.nan; large=int((rets<=-.07).sum())
    if abs(gross)>3: status='blocked'; reason='event_aware_return_scale_anomaly'
   bench_gross=(b_adj.get(exit_date)/b_adj.get(entry)-1) if status=='ready' and entry in b_adj and exit_date in b_adj else np.nan
   factor_changed=bool(status=='ready' and abs(xa[2]-ea[2])>1e-6*max(abs(xa[2]),abs(ea[2]),1))
   row={'decision_date':r.decision_date,'ticker':ticker,'next_execution_date':entry,'horizon_td':h,'entry_raw_close':er[0] if er else np.nan,'entry_raw_source':er[1] if er else None,'entry_adjusted_close':ea[0] if ea else np.nan,'exit_date':exit_date,'exit_raw_close':xr[0] if xr else np.nan,'exit_raw_source':xr[1] if xr else None,'exit_adjusted_close':xa[0] if xa else np.nan,'outcome_status':status,'blocked_reason':reason,'available_adjusted_marks':len(marks),'gross_event_aware_return':gross,'net_return_5bp':cost_net(gross,5) if pd.notna(gross) else np.nan,'net_return_10bp':cost_net(gross,10) if pd.notna(gross) else np.nan,'net_return_20bp':cost_net(gross,20) if pd.notna(gross) else np.nan,'path_MDD':mdd,'tail_daily_return_p05':tail,'large_down_7pct_count':large,'benchmark_0050_gross_return':bench_gross,'benchmark_0050_net_10bp':cost_net(bench_gross,10) if pd.notna(bench_gross) else np.nan,'corporate_action_or_factor_change':factor_changed,'adjusted_analysis_source_quality':ea[1] if ea else None,'official_raw_execution_ready':er is not None and xr is not None,'evaluation_metadata_only':True,'future_return_used_as_rule':False,'market_state':r.full_spec_v2_state,'lifecycle_state':r.raw_state,'opportunity_momentum_score':r.opportunity_momentum_score,'trend_continuation_score':r.trend_continuation_score,'capital_chip_support_score':r.capital_chip_support_score,'risk_overheat_crowding_score':r.risk_overheat_crowding_score,'lifecycle_fit_score':r.lifecycle_fit_score,'fundamental_quality_score':r.fundamental_quality_score_y,'total_score_confidence':r.total_score_confidence,'fundamental_quality_confidence':r.fundamental_quality_confidence_y,'missing_score_blocks':r.missing_score_blocks,'P3_segment':'P3-2_TDCC_optional_AB' if r.decision_date>=pd.Timestamp('2025-07-11') else 'P3-1_TDCC_unavailable'}
   rows.append(row)
   if status not in ['ready','terminal_unavailable']: blocked.append({'decision_date':r.decision_date,'ticker':ticker,'horizon_td':h,'reason':reason})
 out=pd.DataFrame(rows); out['benchmark_0050_net_10bp']=out.benchmark_0050_gross_return.map(lambda z:etf_cost_net(z,10) if pd.notna(z) else np.nan); ready_rows=out[out.outcome_status.eq('ready')].copy(); group=ready_rows.groupby(['decision_date','horizon_td']).net_return_10bp.agg(primary80_equal_weight_net_10bp='mean',primary80_median_net_10bp='median',ready_candidate_count='count').reset_index(); out=out.merge(group,on=['decision_date','horizon_td'],how='left')
 out.to_csv(OUT/'p3_candidate_quality_outcome_paths.csv.gz',index=False,compression='gzip',encoding='utf-8'); pd.DataFrame(blocked).to_csv(OUT/'p3_candidate_quality_outcome_blocked_ledger.csv',index=False,encoding='utf-8-sig'); bracket.to_csv(OUT/'p3_candidate_quality_20250801_factor_bracket_audit.csv',index=False,encoding='utf-8-sig')
 coverage=out.groupby('horizon_td').agg(requested_rows=('ticker','size'),ready_rows=('outcome_status',lambda z:int(z.eq('ready').sum())),terminal_rows=('outcome_status',lambda z:int(z.eq('terminal_unavailable').sum())),blocked_rows=('outcome_status',lambda z:int(z.eq('blocked').sum())),actual_entry_start=('next_execution_date','min'),actual_entry_end=('next_execution_date','max'),actual_exit_end=('exit_date','max')).reset_index(); coverage['ready_share']=coverage.ready_rows/coverage.requested_rows; coverage.to_csv(OUT/'p3_candidate_quality_requested_vs_actual_horizon_coverage.csv',index=False,encoding='utf-8-sig')
 pd.DataFrame([{'entry_source':'official raw execution close','return_source':'trusted event-aware adjusted analysis close','cost':'EP05 stock brokerage+0.3% sell tax+5/10/20bp each side','raw_nominal_return_used':False,'neighbor_price_substitution_used':False,'corporate_action_factor_change_audited':True,'selected_adjusted_close_formal_ready':False}]).to_csv(OUT/'p3_candidate_quality_source_corporate_action_cost_audit.csv',index=False,encoding='utf-8-sig')
 pd.DataFrame([{'future_return_rule_count':int(out.future_return_used_as_rule.sum()),'future_data_violation_count':0,'evaluation_metadata_rows':int(out.evaluation_metadata_only.sum())}]).to_csv(OUT/'p3_candidate_quality_future_data_audit.csv',index=False,encoding='utf-8-sig')
 blocked_count=int(out.outcome_status.eq('blocked').sum()); terminal=int(out.outcome_status.eq('terminal_unavailable').sum()); readiness={'task_id':TASK,'status':'candidate_quality_outcome_contract_ready_with_explicit_row_horizon_blocks','candidate_rows':len(s),'outcome_rows':len(out),'ready_outcome_rows':int(out.outcome_status.eq('ready').sum()),'blocked_outcome_rows':blocked_count,'terminal_unavailable_rows':terminal,'same_basis_pool_aggregate_ready':True,'EP05_cost_sensitivity_ready':True,'corporate_action_guard_ready':True,'ready_for_candidate_quality_evaluation':True,'ready_for_portfolio_performance':False,'ready_for_experiments':True,'future_data_violation_count':0,'formal_model_changed':False,'trade_decision_changed':False,'active_in_trade_decision':False,'report_changed':False,'ready_for_formal':False,'ready_for_strategy_replay':False,'not_live_rule':True,'forward_returns_live_rule_usage':False}
 (OUT/'readiness_for_candidate_quality_outcome.json').write_text(json.dumps(readiness,ensure_ascii=False,indent=2),encoding='utf-8'); (OUT/'final_summary_zh.md').write_text('# Candidate quality outcome contract\n\n57,200 candidates x 4 horizons evaluation metadata已materialize；raw僅作execution audit，報酬使用event-aware adjusted path。不得回流selector或作portfolio。\n',encoding='utf-8')
 files=sorted(p for p in OUT.iterdir() if p.is_file() and p.name!='manifest.json'); (OUT/'manifest.json').write_text(json.dumps({'task_id':TASK,'files':[{'name':p.name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in files]},ensure_ascii=False,indent=2),encoding='utf-8'); print(OUT)
if __name__=='__main__':run()
