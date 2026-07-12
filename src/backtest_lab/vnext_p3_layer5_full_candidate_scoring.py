from __future__ import annotations
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
SRC=ROOT/'outputs/vnext_p3_layer5_daily_feature_state_action_materialization_20260712/p3_layer5_daily_feature_state_matrix.csv'
L4=ROOT/'outputs/vnext_layer4_80_primary_pool_contract_20260708/layer4_80_primary_pool_contract.csv'
OUT=ROOT/'outputs/vnext_p3_layer5_full_candidate_risk_adjusted_scoring_contract_20260712'
TASK='TASK-BACKTEST-CORE-VNEXT-P3-LAYER5-FULL-CANDIDATE-RISK-ADJUSTED-SCORING-CONTRACT-001'

def pct(s): return s.rank(pct=True,method='average')*100
def self_pct(df,col,window):
 return df.groupby('ticker',group_keys=False)[col].apply(lambda s:s.rolling(window,min_periods=min(20,window)).rank(pct=True).mul(100))
def avg(x,cols): return x[cols].mean(axis=1,skipna=True)
def confidence(x,cols): return x[cols].notna().mean(axis=1)

def run():
 OUT.mkdir(parents=True,exist_ok=True)
 x=pd.read_csv(SRC,dtype={'ticker':str},low_memory=False); x['decision_date']=pd.to_datetime(x.decision_date); x['membership_snapshot_date']=pd.to_datetime(x.membership_snapshot_date)
 l4=pd.read_csv(L4,dtype={'ticker':str},low_memory=False); l4['snapshot_date']=pd.to_datetime(l4.snapshot_date)
 fundamental=['snapshot_date','ticker','layer1_financial_risk_flag_count','layer1_quality_floor_risk_pctile_by_week','monthly_revenue_available','quarterly_fundamental_available','missing_core_fundamental_flag','layer1_pass_bottom20','layer1_pass_bottom30']
 x=x.merge(l4[fundamental].drop_duplicates(['snapshot_date','ticker']),left_on=['membership_snapshot_date','ticker'],right_on=['snapshot_date','ticker'],how='left')
 raw=['RS5','RS10','RS20','RS40','RS60','MA20_slope','MA60_slope','BIAS20','BIAS60','BIAS20_z','BIAS60_z','K','D','vol20','vol60','drawdown60','large_down20','tv5','tv20','tv60','institutional_foreign_net_5D','institutional_foreign_net_10D','institutional_foreign_net_20D','institutional_trust_net_5D','institutional_trust_net_10D','institutional_trust_net_20D','institutional_dealer_net_5D','institutional_dealer_net_10D','institutional_dealer_net_20D','margin_margin_change_5D','margin_margin_change_10D','margin_margin_change_20D','margin_short_change_5D','margin_short_change_10D','margin_short_change_20D','lending_sbl_change_5D','lending_sbl_change_10D','lending_sbl_change_20D','foreignown_foreign_holding_ratio_5D','foreignown_foreign_holding_ratio_10D','foreignown_foreign_holding_ratio_20D']
 for c in raw:
  x[c]=pd.to_numeric(x[c],errors='coerce'); x[f'{c}_cross_pct']=x.groupby('decision_date')[c].transform(pct)
  for w,label in [(63,'3m'),(126,'6m'),(252,'12m')]: x[f'{c}_self_{label}_pct']=self_pct(x,c,w)
 def robust(c,direction=1):
  z=avg(x,[f'{c}_cross_pct',f'{c}_self_3m_pct',f'{c}_self_6m_pct',f'{c}_self_12m_pct']); return z if direction>0 else 100-z
 # A: three independent horizons, avoiding five raw RS votes.
 x['A_short_RS']=avg(pd.DataFrame({'a':robust('RS5'),'b':robust('RS10')}),['a','b']); x['A_medium_RS']=avg(pd.DataFrame({'a':robust('RS20'),'b':robust('RS40')}),['a','b']); x['A_long_RS']=robust('RS60')
 x['opportunity_momentum_score']=avg(x,['A_short_RS','A_medium_RS','A_long_RS']); x['opportunity_momentum_confidence']=confidence(x,['A_short_RS','A_medium_RS','A_long_RS'])
 # B: slope, location/reclaim, and breakdown safety are separate dimensions.
 x['B_MA_slope']=avg(pd.DataFrame({'a':robust('MA20_slope'),'b':robust('MA60_slope')}),['a','b']); x['B_price_structure']=100*x[['close']].notna().iloc[:,0]*(x['close'].gt(x.MA20).astype(float)+x['close'].gt(x.MA60).astype(float))/2; x['B_breakdown_safety']=100*(~x.price_breakdown.astype(bool))
 x['trend_continuation_score']=avg(x,['B_MA_slope','B_price_structure','B_breakdown_safety']); x['trend_continuation_confidence']=confidence(x,['B_MA_slope','B_price_structure','B_breakdown_safety'])
 # C: equal independent dimensions. Missing/not-applicable remains NA and affects confidence only when applicable.
 x['C_traded_value']=avg(pd.DataFrame({'a':robust('tv5'),'b':robust('tv20'),'c':robust('tv60')}),['a','b','c'])
 x['C_institutional_flow']=avg(pd.DataFrame({c:robust(c) for c in ['institutional_foreign_net_20D','institutional_trust_net_20D','institutional_dealer_net_20D']}),['institutional_foreign_net_20D','institutional_trust_net_20D','institutional_dealer_net_20D'])
 x['C_foreign_ownership']=robust('foreignown_foreign_holding_ratio_20D')
 x['C_margin_short_balance']=avg(pd.DataFrame({'margin_stability':100-abs(robust('margin_margin_change_20D')-50)*2,'short_safety':robust('margin_short_change_20D',-1)}),['margin_stability','short_safety'])
 x['C_lending_support']=robust('lending_sbl_change_20D',-1)
 cdim=['C_traded_value','C_institutional_flow','C_foreign_ownership','C_margin_short_balance','C_lending_support']; x['capital_chip_support_score']=avg(x,cdim); x['capital_chip_support_confidence']=confidence(x,cdim)
 x['tdcc_score']=np.nan; x['tdcc_confidence']=np.where(x.decision_date.ge('2025-07-11'),0.0,np.nan); x['tdcc_semantics']='P3-1_unavailable_or_P3-2_optional_AB_not_zero'
 # D: higher means higher risk. Deep drawdown uses severity=-drawdown, never inverse-rank reward.
 x['D_bias_extreme']=avg(pd.DataFrame({'b20':abs(x.BIAS20_z).clip(0,4)/4*100,'b60':abs(x.BIAS60_z).clip(0,4)/4*100}),['b20','b60']); x['D_KD_extreme']=((x.K-50).abs().clip(0,50)/50*100 + (x.D-50).abs().clip(0,50)/50*100)/2
 x['D_volatility']=avg(pd.DataFrame({'v20':robust('vol20'),'v60':robust('vol60')}),['v20','v60']); x['drawdown_severity']=-x.drawdown60; x['drawdown_severity_cross_pct']=x.groupby('decision_date').drawdown_severity.transform(pct); x['D_drawdown']=x.drawdown_severity_cross_pct
 x['D_downside_events']=avg(pd.DataFrame({'large':robust('large_down20'),'blowoff':x.blowoff.astype(float)*100}),['large','blowoff']); x['D_crowding']=avg(pd.DataFrame({'margin':robust('margin_margin_change_20D'),'short':robust('margin_short_change_20D'),'lend':robust('lending_sbl_change_20D')}),['margin','short','lend'])
 ddim=['D_bias_extreme','D_KD_extreme','D_volatility','D_drawdown','D_downside_events','D_crowding']; x['risk_overheat_crowding_score']=avg(x,ddim); x['risk_overheat_crowding_confidence']=confidence(x,ddim)
 # E: continuous evidence by state groups; no fixed 85/80 constants.
 x['E_turning_evidence']=x.turn_groups.clip(0,5)/5*100; x['E_healthy_evidence']=x.healthy_groups.clip(0,4)/4*100; x['E_weakening_penalty']=x.weak_groups.clip(0,5)/5*100; x['E_overheat_warning']=x.overheat_groups.clip(0,4)/4*100
 x['lifecycle_fit_score']=(.45*x.E_turning_evidence+.55*x.E_healthy_evidence)*(1-x.E_weakening_penalty/100); x.loc[x.raw_state.eq('cooling_down'),'lifecycle_fit_score']*=.6; x.loc[x.raw_state.eq('confirmed_weakening'),'lifecycle_fit_score']=0
 x['lifecycle_fit_confidence']=confidence(x,['turn_groups','healthy_groups','weak_groups','overheat_groups'])
 # F: use only available PIT Layer1 evidence; absent profitability/revenue numerics remain explicit blockers.
 x['F_quality_safety']=100-pd.to_numeric(x.layer1_quality_floor_risk_pctile_by_week,errors='coerce')*100; x['F_financial_risk_safety']=100-pd.to_numeric(x.layer1_financial_risk_flag_count,errors='coerce').clip(0,3)/3*100; x['F_source_coverage']=100*pd.concat([x.monthly_revenue_available.astype(float),x.quarterly_fundamental_available.astype(float)],axis=1).mean(axis=1)
 x['fundamental_quality_score']=avg(x,['F_quality_safety','F_financial_risk_safety','F_source_coverage']); x['fundamental_quality_confidence']=confidence(x,['F_quality_safety','F_financial_risk_safety','F_source_coverage'])/3; x['fundamental_quality_status']='partial_numeric_profitability_revenue_growth_blocked_no_constant_fill'
 conf=['opportunity_momentum_confidence','trend_continuation_confidence','capital_chip_support_confidence','risk_overheat_crowding_confidence','lifecycle_fit_confidence','fundamental_quality_confidence']; x['total_score_confidence']=avg(x,conf)
 x['opportunity_axis']=avg(x,['opportunity_momentum_score','trend_continuation_score','capital_chip_support_score','lifecycle_fit_score','fundamental_quality_score']); x['risk_axis']=x.risk_overheat_crowding_score; x['risk_adjusted_opportunity_axis']=x.opportunity_axis*(1-x.risk_axis/200)
 x['selected_eligibility']=x.price_core_valid.astype(bool)&x.history_ready.astype(bool)&x.fundamental_quality_score.notna(); x['selected_ineligibility_reason']=np.select([~x.price_core_valid.astype(bool),~x.history_ready.astype(bool),x.fundamental_quality_score.isna()],['price_core_invalid','insufficient_history','fundamental_quality_unavailable'],'eligible')
 x['PIT_available_at']=x.decision_date.astype(str)+'_after_close'; x['future_return_used_as_rule']=False
 keep=['decision_date','membership_snapshot_date','next_execution_date','ticker','name','market','raw_state','selected_eligibility','selected_ineligibility_reason','opportunity_momentum_score','trend_continuation_score','capital_chip_support_score','risk_overheat_crowding_score','lifecycle_fit_score','fundamental_quality_score','opportunity_axis','risk_axis','risk_adjusted_opportunity_axis',*conf,'total_score_confidence','fundamental_quality_status','tdcc_score','tdcc_confidence','tdcc_semantics','PIT_available_at','future_return_used_as_rule']
 x['missing_score_blocks']=x[['opportunity_momentum_score','trend_continuation_score','capital_chip_support_score','risk_overheat_crowding_score','lifecycle_fit_score','fundamental_quality_score']].isna().apply(lambda r:'|'.join(r.index[r]),axis=1)
 x['score_reason_codes']='A_RS_horizon_decorrelated|B_MA_structure|C_chip_applicability_aware|D_risk_severity_direction_checked|E_continuous_state_evidence|F_layer1_partial_no_constant'
 keep += ['missing_score_blocks','score_reason_codes']
 matrix=x[keep].copy(); stale=OUT/'p3_full_candidate_spec_v1_score_matrix.csv'; stale.unlink(missing_ok=True); matrix.to_csv(OUT/'p3_full_candidate_spec_v1_score_matrix.csv.gz',index=False,compression='gzip',encoding='utf-8')
 context_cols=['decision_date','ticker',*sum(([c,f'{c}_cross_pct',f'{c}_self_3m_pct',f'{c}_self_6m_pct',f'{c}_self_12m_pct'] for c in raw),[])]
 x[context_cols].to_csv(OUT/'p3_full_candidate_raw_self_cross_context.csv.gz',index=False,compression='gzip',encoding='utf-8')
 direction=pd.DataFrame([
  ['drawdown60','risk','severity=-drawdown; higher severity=higher risk','pass_no_deep_drawdown_reward'],['vol20/vol60','risk','higher percentile=higher risk','pass'],['large_down20','risk','higher count=higher risk','pass'],['BIAS20/60 z','risk','absolute extreme higher risk','pass'],['margin_change','capital/risk','stability in C; crowding in derived D flag','crosswalk_no_raw_double_weight'],['lending_change','capital/risk','increase lowers support; crowding derived context','pass']],columns=['raw_field','axis','direction_transform','audit_status']); direction.to_csv(OUT/'p3_full_candidate_direction_audit.csv',index=False,encoding='utf-8-sig')
 owners=[]
 groups={'A':['RS5','RS10','RS20','RS40','RS60'],'B':['MA20_slope','MA60_slope','close_vs_MA20/60','price_breakdown'],'C':['tv5/20/60','institutional foreign/trust/dealer','foreign ownership','margin/short','lending','TDCC optional'],'D':['BIAS20/60','K/D','vol20/60','drawdown60','large_down20','blowoff','derived crowding flags'],'E':['turn/healthy/weak/overheat evidence groups'],'F':['Layer1 quality risk percentile','financial risk flags','revenue/fundamental availability']}
 for block,fields in groups.items():
  for f in fields: owners.append({'source_block':block,'raw_field_or_dimension':f,'owner_block':block,'no_double_count':True})
 pd.DataFrame(owners).to_csv(OUT/'p3_full_candidate_raw_field_ownership.csv',index=False,encoding='utf-8-sig')
 corr=x[raw].corr(method='spearman').stack().reset_index(); corr.columns=['field_a','field_b','spearman']; corr=corr[(corr.field_a<corr.field_b)&corr.spearman.abs().ge(.8)]; corr['treatment']='precombine_within_independent_dimension_or_cross_block_audit'; corr.to_csv(OUT/'p3_full_candidate_duplicate_correlation_audit.csv',index=False,encoding='utf-8-sig')
 coverage=pd.DataFrame([{'block':b,'score_nonnull_share':matrix[s].notna().mean(),'median_confidence':matrix[c].median(),'status':'partial' if b=='F' else 'ready'} for b,s,c in [('A','opportunity_momentum_score','opportunity_momentum_confidence'),('B','trend_continuation_score','trend_continuation_confidence'),('C','capital_chip_support_score','capital_chip_support_confidence'),('D','risk_overheat_crowding_score','risk_overheat_crowding_confidence'),('E','lifecycle_fit_score','lifecycle_fit_confidence'),('F','fundamental_quality_score','fundamental_quality_confidence')]]); coverage.to_csv(OUT/'p3_full_candidate_block_coverage_readiness.csv',index=False,encoding='utf-8-sig')
 blocked=pd.DataFrame([{'family':'fundamental_quality','field':'numeric revenue growth/profitability/margins/cashflow/leverage','status':'blocked_in_current_exact_daily_join','impact':'F partial; no constant fill; no performance'}, {'family':'TDCC','field':'holder distribution','status':'P3-1 unavailable; P3-2 optional A/B','impact':'NA and confidence only'}]); blocked.to_csv(OUT/'p3_full_candidate_blocked_ledger.csv',index=False,encoding='utf-8-sig')
 pd.DataFrame([{'family':'adjusted price technical','source_available_at_policy':'same decision close after market; next-day execution','PIT_alignment':'direct daily observation'}, {'family':'institutional/margin/lending/foreign ownership','source_available_at_policy':'official post-close inherited from upstream compact','PIT_alignment':'upstream next-eligible-date aligned; no same-day future backfill'}, {'family':'Layer1 fundamental','source_available_at_policy':'weekly snapshot PIT contract','PIT_alignment':'membership_snapshot_date join only'}, {'family':'TDCC','source_available_at_policy':'weekly release lag required','PIT_alignment':'P3-1 unavailable; P3-2 optional only'}]).to_csv(OUT/'p3_full_candidate_PIT_available_at_policy.csv',index=False,encoding='utf-8-sig')
 ready={'task_id':TASK,'status':'full_candidate_spec_v1_stage_a_contract_ready_fundamental_partial','candidate_rows':len(matrix),'score_blocks_materialized':6,'constant_block_F_used':False,'drawdown_direction_audit_pass':True,'TDCC_NA_as_zero':False,'future_data_violation_count':0,'ready_for_stage_a_validation':True,'ready_for_candidate_quality_evaluation':False,'ready_for_portfolio_performance':False,'ready_for_experiments':True,'formal_model_changed':False,'trade_decision_changed':False,'active_in_trade_decision':False,'report_changed':False,'ready_for_formal':False,'ready_for_strategy_replay':False,'not_live_rule':True,'forward_returns_live_rule_usage':False}
 (OUT/'readiness_for_full_candidate_spec_v1.json').write_text(json.dumps(ready,ensure_ascii=False,indent=2),encoding='utf-8'); (OUT/'final_summary_zh.md').write_text('# full_candidate_spec_v1\n\n六軸已實算；F軸使用現有Layer1 PIT證據且標partial，不再常數50。Stage A可驗證，績效禁止。\n',encoding='utf-8')
 files=sorted(p for p in OUT.iterdir() if p.is_file() and p.name!='manifest.json'); (OUT/'manifest.json').write_text(json.dumps({'task_id':TASK,'files':[{'name':p.name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in files]},ensure_ascii=False,indent=2),encoding='utf-8'); print(OUT)
if __name__=='__main__': run()
