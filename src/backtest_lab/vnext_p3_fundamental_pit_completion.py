from __future__ import annotations
import glob,hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'outputs/vnext_p3_layer5_full_candidate_risk_adjusted_scoring_contract_20260712'
RADAR=Path(r'C:\Users\zergv\Documents\Codex\2026-05-23\ai-stock-rotation-radar-https-docs\outputs')
MONTH=RADAR/'radar_dynamic_pool1_mops_monthly_revenue_full_universe_pit_20260703/accepted_monthly_revenue_rows_shards'
QUARTER=RADAR/'radar_dynamic_pool1_quarterly_fundamentals_full_sweep_20260703/shards'
OUT=ROOT/'outputs/vnext_p3_layer5_full_candidate_scoring_fundamental_pit_completion_20260712'
TASK='TASK-BACKTEST-CORE-VNEXT-P3-LAYER5-FULL-CANDIDATE-SCORING-FUNDAMENTAL-PIT-COMPLETION-001'

def rolling_self(s,window=252): return s.rolling(window,min_periods=20).rank(pct=True)*100
def context(candidates,col):
 candidates[f'{col}_cross_pct']=candidates.groupby('decision_date')[col].rank(pct=True)*100
 candidates[f'{col}_self_pct']=candidates.groupby('ticker',group_keys=False)[col].apply(rolling_self)
 return candidates[[f'{col}_cross_pct',f'{col}_self_pct']].mean(axis=1)
def asof_join(candidates,events,cols,prefix):
 out=[]
 for ticker,g in candidates.groupby('ticker',sort=False):
  e=events[events.ticker.eq(ticker)].sort_values('available_date')
  z=g.sort_values('decision_date')
  if e.empty:
   for c in cols: z[f'{prefix}{c}']=np.nan
  else:
   z=pd.merge_asof(z,e[['available_date',*cols]].sort_values('available_date'),left_on='decision_date',right_on='available_date',direction='backward')
   z=z.rename(columns={c:f'{prefix}{c}' for c in cols}); z=z.rename(columns={'available_date':f'{prefix}available_date'})
  out.append(z)
 return pd.concat(out,ignore_index=True)

def run():
 OUT.mkdir(parents=True,exist_ok=True)
 base=pd.read_csv(BASE/'p3_full_candidate_spec_v1_score_matrix.csv.gz',dtype={'ticker':str},low_memory=False); base['decision_date']=pd.to_datetime(base.decision_date)
 ms=pd.concat([pd.read_csv(p,dtype={'ticker':str},low_memory=False) for p in glob.glob(str(MONTH/'*.csv'))],ignore_index=True); ms['available_date']=pd.to_datetime(ms.available_date); ms['period']=pd.PeriodIndex(ms.revenue_year_month,freq='M'); ms=ms.sort_values(['ticker','period'])
 gr=ms.groupby('ticker'); ms['revenue_yoy']=gr.revenue_value.pct_change(12,fill_method=None); ms['revenue_3m']=gr.revenue_value.rolling(3,min_periods=3).sum().reset_index(level=0,drop=True); ms['revenue_3m_yoy']=ms.groupby('ticker').revenue_3m.pct_change(12,fill_method=None); ms['revenue_ttm']=gr.revenue_value.rolling(12,min_periods=12).sum().reset_index(level=0,drop=True); ms['revenue_ttm_yoy']=ms.groupby('ticker').revenue_ttm.pct_change(12,fill_method=None); ms['revenue_yoy_vol36']=ms.groupby('ticker').revenue_yoy.rolling(36,min_periods=12).std().reset_index(level=0,drop=True); ms['revenue_growth_persistence']=ms.groupby('ticker').revenue_yoy.rolling(6,min_periods=3).apply(lambda s:(s>0).mean(),raw=False).reset_index(level=0,drop=True); ms['revenue_spike_anomaly']=ms.groupby('ticker').revenue_yoy.transform(lambda s:s.gt(s.expanding(min_periods=12).quantile(.9))) & ms.revenue_ttm_yoy.le(ms.revenue_yoy)
 ms['revenue_period']=ms.period.astype(str); ms['revenue_source']='official_MOPS_monthly_conservative_next_month_day10'; ms['revenue_formal_exact']=False
 mcols=['revenue_period','revenue_yoy','revenue_3m_yoy','revenue_ttm','revenue_ttm_yoy','revenue_yoy_vol36','revenue_growth_persistence','revenue_spike_anomaly','revenue_source','revenue_formal_exact']
 x=asof_join(base,ms,mcols,'m_')
 qs=pd.concat([pd.read_csv(p,dtype={'ticker':str},low_memory=False) for p in glob.glob(str(QUARTER/'*.csv'))],ignore_index=True); qs['available_date']=pd.to_datetime(qs.available_date); qs=qs.sort_values(['ticker','fiscal_year','quarter']); qg=qs.groupby('ticker')
 qs['quarter_revenue_yoy']=qg.operating_revenue.pct_change(4,fill_method=None); qs['eps_yoy']=qg.eps.pct_change(4,fill_method=None); qs['gross_margin_change']=qg.gross_margin.diff(); qs['operating_margin_change']=qg.operating_margin.diff(); qs['net_margin']=qs.net_income/qs.operating_revenue.replace(0,np.nan); qs['leverage_ratio']=qs.total_liabilities/qs.total_assets.replace(0,np.nan); qs['quarter_period']=qs.fiscal_year.astype(str)+'Q'+qs.quarter.astype(str); qs['quarter_source']='official_MOPS_t163sb04_conservative_statutory_deadline'; qs['quarter_formal_exact']=False
 qcols=['quarter_period','quarter_revenue_yoy','eps','eps_yoy','roe','net_margin','gross_margin','gross_margin_change','operating_margin','operating_margin_change','leverage_ratio','quarter_source','quarter_formal_exact']
 x=asof_join(x,qs,qcols,'q_')
 # PIT audit before scoring.
 x['monthly_staleness_days']=(x.decision_date-x.m_available_date).dt.days; x['quarterly_staleness_days']=(x.decision_date-x.q_available_date).dt.days
 x['monthly_future_violation']=x.m_available_date.gt(x.decision_date); x['quarterly_future_violation']=x.q_available_date.gt(x.decision_date)
 # Family contexts. Growth is capped when anomaly warning is active; anomaly is not a broad rerank penalty.
 rev_growth=pd.concat([context(x,'m_revenue_yoy'),context(x,'m_revenue_3m_yoy'),context(x,'m_revenue_ttm_yoy')],axis=1).mean(axis=1); rev_stability=100-context(x,'m_revenue_yoy_vol36'); persistence=x.m_revenue_growth_persistence*100; x['F_revenue_score']=pd.concat([rev_growth,rev_stability,persistence],axis=1).mean(axis=1); x.loc[x.m_revenue_spike_anomaly.eq(True),'F_revenue_score']=x.loc[x.m_revenue_spike_anomaly.eq(True),'F_revenue_score'].clip(upper=70)
 x['revenue_anomaly_role']='warning_context_cap_positive_growth_component_not_rerank_penalty'
 x['F_profitability_score']=pd.concat([context(x,'q_eps'),context(x,'q_eps_yoy'),context(x,'q_roe'),context(x,'q_net_margin')],axis=1).mean(axis=1)
 x['F_margins_score']=pd.concat([context(x,'q_gross_margin'),context(x,'q_gross_margin_change'),context(x,'q_operating_margin'),context(x,'q_operating_margin_change')],axis=1).mean(axis=1)
 x['F_cashflow_score']=np.nan; x['F_cashflow_status']='blocked_no_accepted_PIT_OCF_or_FCF_source'
 x['F_leverage_liquidity_score']=100-context(x,'q_leverage_ratio'); x['current_ratio_status']='blocked_no_accepted_PIT_balance_sheet_ratio_source'
 families=['F_revenue_score','F_profitability_score','F_margins_score','F_cashflow_score','F_leverage_liquidity_score']; x['fundamental_family_available_count']=x[families].notna().sum(axis=1); x['fundamental_quality_confidence']=x.fundamental_family_available_count/len(families); x['fundamental_quality_score']=x[families].mean(axis=1,skipna=True); x['fundamental_quality_status']=np.select([x.fundamental_family_available_count.eq(5),x.fundamental_family_available_count.eq(4),x.fundamental_family_available_count.eq(3)],['ready_all_five_families','numeric_four_family_ready_one_blocked','numeric_three_family_ready_cashflow_and_leverage_blocked'],'partial_less_than_three_families')
 # Recompute the two-axis hook only; no selected outcome or performance.
 x['opportunity_axis_without_F']=x[['opportunity_momentum_score','trend_continuation_score','capital_chip_support_score','lifecycle_fit_score']].mean(axis=1); x['opportunity_axis_with_F']=x[['opportunity_momentum_score','trend_continuation_score','capital_chip_support_score','lifecycle_fit_score','fundamental_quality_score']].mean(axis=1); x['risk_adjusted_opportunity_axis_with_F']=x.opportunity_axis_with_F*(1-x.risk_axis/200)
 x['PIT_available_at_status']='asof_join_available_date_lte_decision_date'; x['future_return_used_as_rule']=False
 keep=['decision_date','membership_snapshot_date','next_execution_date','ticker','name','market','raw_state','selected_eligibility','m_revenue_period','m_available_date','monthly_staleness_days','m_revenue_yoy','m_revenue_3m_yoy','m_revenue_ttm','m_revenue_ttm_yoy','m_revenue_growth_persistence','m_revenue_spike_anomaly','q_quarter_period','q_available_date','quarterly_staleness_days','q_quarter_revenue_yoy','q_eps','q_eps_yoy','q_roe','q_net_margin','q_gross_margin','q_gross_margin_change','q_operating_margin','q_operating_margin_change','q_leverage_ratio',*families,'fundamental_family_available_count','fundamental_quality_confidence','fundamental_quality_score','fundamental_quality_status','revenue_anomaly_role','F_cashflow_status','current_ratio_status','opportunity_axis_without_F','opportunity_axis_with_F','risk_axis','risk_adjusted_opportunity_axis_with_F','PIT_available_at_status','future_return_used_as_rule']
 out=x[keep].copy(); out.to_csv(OUT/'p3_full_candidate_spec_v1_fundamental_PIT_completed_matrix.csv.gz',index=False,compression='gzip',encoding='utf-8')
 coverage=[]
 for family,col in [('revenue','F_revenue_score'),('profitability','F_profitability_score'),('margins','F_margins_score'),('cashflow','F_cashflow_score'),('leverage_liquidity','F_leverage_liquidity_score')]:
  share=float(x[col].notna().mean()); coverage.append({'family':family,'available_rows':int(x[col].notna().sum()),'coverage_share':share,'median_staleness_days':float(x.monthly_staleness_days.median()) if family=='revenue' else (float(x.quarterly_staleness_days.median()) if family!='cashflow' else np.nan),'status':'blocked' if share==0 else ('partial' if share<.95 else 'ready')})
 pd.DataFrame(coverage).to_csv(OUT/'p3_fundamental_family_coverage_confidence.csv',index=False,encoding='utf-8-sig')
 pd.DataFrame([{'requested_start':'2023-07-11','requested_end':'2026-06-29','actual_candidate_start':x.decision_date.min(),'actual_candidate_end':x.decision_date.max(),'rows':len(x),'P3_1_TDCC':'unavailable_not_F','P3_2_TDCC':'optional_chip_AB_not_F','monthly_source_start':ms.available_date.min(),'monthly_source_end':ms.available_date.max(),'quarter_source_start':qs.available_date.min(),'quarter_source_end':qs.available_date.max()}]).to_csv(OUT/'p3_fundamental_requested_vs_actual_coverage.csv',index=False,encoding='utf-8-sig')
 pd.DataFrame([{'field':'monthly available_date','policy':'conservative next-month day10 weekday-adjusted','formal_exact':False},{'field':'quarter available_date','policy':'conservative statutory filing deadline by quarter','formal_exact':False},{'field':'cashflow/FCF','policy':'blocked no accepted PIT source','formal_exact':False},{'field':'PE/PB/PS/current ratio','policy':'blocked independently; does not erase ready families','formal_exact':False}]).to_csv(OUT/'p3_fundamental_PIT_policy_and_blockers.csv',index=False,encoding='utf-8-sig')
 pd.DataFrame([{'monthly_future_violation_count':int(x.monthly_future_violation.sum()),'quarterly_future_violation_count':int(x.quarterly_future_violation.sum()),'future_return_rule_count':int(x.future_return_used_as_rule.sum())}]).to_csv(OUT/'p3_fundamental_future_data_audit.csv',index=False,encoding='utf-8-sig')
 # Dimension-level correlation audit after precombination, not raw-field vote inflation.
 dims=['opportunity_momentum_score','trend_continuation_score','capital_chip_support_score','risk_axis','lifecycle_fit_score','F_revenue_score','F_profitability_score','F_margins_score','F_leverage_liquidity_score']; corr=x[dims].corr(method='spearman').stack().reset_index(); corr.columns=['dimension_a','dimension_b','spearman']; corr=corr[corr.dimension_a<corr.dimension_b]; corr['high_correlation_flag']=corr.spearman.abs().ge(.8); corr['policy']='dimension_precombined_once_no_raw_duplicate_vote'; corr.to_csv(OUT/'p3_fundamental_cross_block_dimension_correlation_audit.csv',index=False,encoding='utf-8-sig')
 dist=x[['F_revenue_score','F_profitability_score','F_margins_score','F_leverage_liquidity_score','fundamental_quality_score','fundamental_quality_confidence']].describe().T.reset_index().rename(columns={'index':'score'}); dist.to_csv(OUT/'p3_fundamental_score_distribution_sanity.csv',index=False,encoding='utf-8-sig')
 ready={'task_id':TASK,'status':'fundamental_PIT_numeric_three_family_ready_cashflow_and_leverage_blocked','candidate_rows':len(x),'F_all_rows_partial':True,'numeric_family_ready_count':3,'mandatory_family_count':5,'cashflow_family_ready':False,'leverage_liquidity_family_ready':False,'current_ratio_ready':False,'PE_PB_PS_ready':False,'constant_or_zero_fill_used':False,'current_snapshot_history_backfill_used':False,'future_data_violation_count':int(x.monthly_future_violation.sum()+x.quarterly_future_violation.sum()),'ready_for_stage_a_revalidation':False,'ready_for_candidate_quality_evaluation':False,'ready_for_portfolio_performance':False,'ready_for_experiments':False,'ready_for_radar_bounded_balance_cashflow_source_fill':True,'formal_model_changed':False,'trade_decision_changed':False,'active_in_trade_decision':False,'report_changed':False,'ready_for_formal':False,'ready_for_strategy_replay':False,'not_live_rule':True,'forward_returns_live_rule_usage':False}
 (OUT/'readiness_for_fundamental_PIT_completion.json').write_text(json.dumps(ready,ensure_ascii=False,indent=2),encoding='utf-8'); (OUT/'final_summary_zh.md').write_text('# Fundamental PIT completion\n\nRevenue/profitability/margins三family已按available_date as-of join；cashflow與leverage/liquidity零numeric coverage，F仍全體partial。不可交Stage A revalidation或績效，需Radar bounded source fill。\n',encoding='utf-8')
 files=sorted(p for p in OUT.iterdir() if p.is_file() and p.name!='manifest.json'); (OUT/'manifest.json').write_text(json.dumps({'task_id':TASK,'files':[{'name':p.name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in files]},ensure_ascii=False,indent=2),encoding='utf-8'); print(OUT)
if __name__=='__main__':run()
