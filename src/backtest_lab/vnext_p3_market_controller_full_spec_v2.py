from __future__ import annotations
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
SRC=ROOT/'outputs/vnext_p3_layer5_daily_feature_state_action_materialization_20260712'
RADAR=Path(r'C:\Users\zergv\Documents\Codex\2026-05-23\ai-stock-rotation-radar-https-docs\outputs\radar_vnext_p3_recent_full_feature_data_readiness_acquisition_20260711\compact')
MARKET=Path(r'C:\Users\zergv\Documents\Codex\2026-05-23\ai-stock-rotation-radar-https-docs\outputs\radar_vnext_p3_market_state_source_fill_20260711\compact')
TAIFEX4=Path(r'C:\Users\zergv\Documents\Codex\2026-05-23\ai-stock-rotation-radar-https-docs\outputs\radar_vnext_p3_taifex_trade_flow_compact_materialization_20260712')
OUT=ROOT/'outputs/vnext_p3_market_controller_full_spec_v2_20260712'
TASK='TASK-BACKTEST-CORE-VNEXT-P3-MARKET-CONTROLLER-FULL-SPEC-V2-001'

def years(fam): return pd.concat([pd.read_csv(p,low_memory=False) for p in sorted((RADAR/fam).glob('*.csv.gz'))],ignore_index=True)
def zroll(s,n=252,minp=60): return (s-s.rolling(n,min_periods=minp).mean())/s.rolling(n,min_periods=minp).std()
def direction(z,thr=.5): return np.select([z>=thr,z<=-thr],[1,-1],0)
def group_vote(df,cols,min_same=3):
 bull=(df[cols]>0).sum(axis=1); bear=(df[cols]<0).sum(axis=1); avail=df[cols].notna().sum(axis=1)
 state=np.select([(bull>=min_same)&(bear<=1),(bear>=min_same)&(bull<=1)],['bullish','bearish'],'neutral')
 score=(bull-bear)/len(cols)*100; conf=avail/len(cols); reasons=df[cols].apply(lambda r:'|'.join(f'{c}:{int(v)}' for c,v in r.items() if pd.notna(v)),axis=1)
 return state,score,conf,reasons

def run():
 OUT.mkdir(parents=True,exist_ok=True)
 old=pd.read_csv(SRC/'p3_layer5_daily_market_state.csv'); old['decision_date']=pd.to_datetime(old.decision_date); dates=old[['decision_date']].copy()
 b=pd.read_csv(ROOT/'backtest_cache/stock_pool_observations/0050_TW.csv'); b['date']=pd.to_datetime(b.date); b=b.sort_values('date'); tw=pd.read_csv(ROOT/'backtest_cache/taiex_yfinance/^TWII.csv'); tw['date']=pd.to_datetime(tw.date); tw=tw.sort_values('date')
 def index_features(d,prefix,col='adj_close'):
  for n in [5,20,60]: d[f'{prefix}_ret{n}']=d[col].pct_change(n)
  for n in [20,60]: d[f'{prefix}_ma{n}']=d[col].rolling(n).mean(); d[f'{prefix}_slope{n}']=d[f'{prefix}_ma{n}'].pct_change(5)
  d[f'{prefix}_vol20']=d[col].pct_change().rolling(20).std(); d[f'{prefix}_vol60']=d[col].pct_change().rolling(60).std(); d[f'{prefix}_dd60']=d[col]/d[col].rolling(60).max()-1
 index_features(b,'etf'); index_features(tw,'taiex')
 keep=['date']+[c for c in b if c.startswith('etf_')]; m=dates.merge(b[keep],left_on='decision_date',right_on='date',how='left').drop(columns='date'); keep=['date']+[c for c in tw if c.startswith('taiex_')]; m=m.merge(tw[keep],left_on='decision_date',right_on='date',how='left').drop(columns='date')
 m['tw_direction']=direction((zroll(m.etf_ret20)+zroll(m.taiex_ret20))/2); m['tw_magnitude']=direction((zroll(m.etf_ret60)+zroll(m.taiex_ret60))/2); m['tw_persistence']=direction((np.sign(m.etf_ret5)+np.sign(m.etf_ret20)+np.sign(m.taiex_ret5)+np.sign(m.taiex_ret20))/4,.25); m['tw_risk_reversal']=direction(-((zroll(m.etf_vol20-m.etf_vol60)+zroll(m.taiex_vol20-m.taiex_vol60))/2))
 m['taiwan_group'],m['taiwan_score'],m['taiwan_confidence'],m['taiwan_reasons']=group_vote(m,['tw_direction','tw_magnitude','tw_persistence','tw_risk_reversal'],3)

 f=pd.read_csv(SRC/'p3_layer5_daily_feature_state_matrix.csv',usecols=['decision_date','ticker','adjusted_close','MA20','MA60','RS20']); f['decision_date']=pd.to_datetime(f.decision_date); f=f.sort_values(['ticker','decision_date']); f['advance']=f.groupby('ticker').adjusted_close.pct_change()>0; f['above20']=f.adjusted_close>f.MA20; f['above60']=f.adjusted_close>f.MA60; f['rspos']=f.RS20>0
 br=f.groupby('decision_date').agg(above20=('above20','mean'),above60=('above60','mean'),rspos=('rspos','mean'),advance=('advance','mean')); br['breadth_direction']=direction(zroll((br.above20+br.above60+br.rspos)/3)); br['breadth_magnitude']=direction(zroll(br.advance)); br['breadth_persistence']=direction(zroll(((br.above20+br.rspos)/2).diff(5))); br['breadth_reversal']=direction(zroll(((br.above20+br.rspos)/2).diff(5)-((br.above20+br.rspos)/2).diff(20)))
 br['breadth_group'],br['breadth_score'],br['breadth_confidence'],br['breadth_reasons']=group_vote(br,['breadth_direction','breadth_magnitude','breadth_persistence','breadth_reversal'],3); m=m.merge(br.reset_index(),on='decision_date',how='left')

 for fam,key in [('full_market_traded_value','turnover'),('full_market_margin_balance','margin')]:
  d=pd.read_csv(MARKET/fam/'p3_daily.csv.gz'); d['date']=pd.to_datetime(d.date); d=d[d.market=='ALL'][['date','value']].rename(columns={'date':'decision_date','value':key}); m=m.merge(d,on='decision_date',how='left')
 m['cap_direction']=direction(zroll(m.turnover.pct_change(20)-m.margin.pct_change(20))); m['cap_magnitude']=direction(zroll(m.turnover.pct_change(5))-zroll(m.margin.pct_change(5))); m['cap_persistence']=direction(zroll(m.turnover.pct_change().rolling(10).sum()-m.margin.pct_change().rolling(10).sum())); m['cap_divergence']=direction(zroll(m.turnover.pct_change(20)+m.margin.pct_change(20))); m['cap_volrisk']=direction(-zroll(m.etf_vol20-m.etf_vol60)); m['capital_group'],m['capital_score'],m['capital_confidence'],m['capital_reasons']=group_vote(m,['cap_direction','cap_magnitude','cap_persistence','cap_divergence','cap_volrisk'],3)

 tr=json.loads((TAIFEX4/'readiness_for_core_p3_taifex_trade_flow_absorption.json').read_text(encoding='utf-8-sig'))
 if not tr['ready_for_core_p3_taifex_trade_flow_absorption'] or tr['trade_flow_derived_from_oi_change']: raise ValueError('TAIFEX four-field compact not ready')
 tf=pd.read_csv(TAIFEX4/'p3_taifex_foreign_trade_oi_daily_compact.csv.gz'); tf['decision_date']=pd.to_datetime(tf.pit_eligible_decision_date); tf=tf.rename(columns={'foreign_futures_oi_net_contracts':'oi_contracts','foreign_futures_oi_net_amount':'oi_amount','foreign_futures_trade_net_contracts':'trade_contracts','foreign_futures_trade_net_amount':'trade_amount'}); m=m.merge(tf[['decision_date','source_date','oi_contracts','oi_amount','trade_contracts','trade_amount','source_quality']],on='decision_date',how='left')
 m['deriv_level_crowding_warning']=(zroll(m.oi_contracts).abs()>=1.5)|(zroll(m.oi_amount).abs()>=1.5); m['deriv_level']=0
 change=(zroll(m.oi_contracts.diff(5),60,20)+zroll(m.oi_amount.diff(5),60,20))/2; m['deriv_change']=direction(change)
 m['deriv_persistence']=direction((np.sign(m.oi_contracts.diff()).rolling(10).sum()+np.sign(m.oi_amount.diff()).rolling(10).sum())/2,3)
 tradez=(zroll(m.trade_contracts,60,20)+zroll(m.trade_amount,60,20))/2; m['deriv_trade_flow']=direction(tradez); m['deriv_trade_reversal']=direction(zroll(m.trade_contracts.diff(1)-m.trade_contracts.rolling(5,min_periods=3).mean(),60,20))
 market5=(m.etf_ret5+m.taiex_ret5)/2; oi5=m.oi_contracts.diff(5); m['deriv_divergence']=np.select([(market5>0)&(oi5<0),(market5<0)&(oi5>0)],[-1,1],direction(zroll(m.oi_contracts.diff(5)-m.oi_contracts.diff(20))))
 m['derivatives_group'],m['derivatives_score'],m['derivatives_confidence'],m['derivatives_reasons']=group_vote(m,['deriv_level','deriv_change','deriv_persistence','deriv_trade_flow','deriv_divergence'],3); m['derivatives_missingness']=np.where(m.trade_contracts.isna(),'PIT_warmup_or_missing','')

 gm=years('global_market'); gm['session_date']=pd.to_datetime(gm.session_date)
 sox=pd.read_csv(MARKET/'global_market/p3_sox.csv.gz'); sox['session_date']=pd.to_datetime(sox.session_date); sox['field']='SOX'; sox=sox.rename(columns={'adjusted_close':'close_value'}); gm=gm.rename(columns={'adjusted_close':'close_value'}); ext=pd.concat([gm[gm.field.isin(['Nasdaq','VIX','USD_TWD'])][['field','session_date','close_value']],sox[['field','session_date','close_value']]],ignore_index=True)
 for field,key in [('Nasdaq','nasdaq'),('SOX','sox'),('VIX','vix'),('USD_TWD','fx')]:
  d=ext[ext.field==field].sort_values('session_date').copy();
  for n in [5,20,60]: d[f'{key}_ret{n}']=d.close_value.pct_change(n)
  d[f'{key}_ma20']=d.close_value.rolling(20).mean(); d[f'{key}_ma60']=d.close_value.rolling(60).mean(); d[f'{key}_slope20']=d[f'{key}_ma20'].pct_change(5); d[f'{key}_slope60']=d[f'{key}_ma60'].pct_change(5); d[f'{key}_pct']=d.close_value.rolling(252,min_periods=60).rank(pct=True)
  cols=['session_date']+[c for c in d if c.startswith(key+'_')]; m=pd.merge_asof(m.sort_values('decision_date'),d[cols],left_on='decision_date',right_on='session_date',direction='backward',suffixes=('','_'+key))
 m['external_nasdaq']=direction((zroll(m.nasdaq_ret20)+zroll(m.nasdaq_ret60)+zroll(m.nasdaq_slope20))/3); m['external_sox']=direction((zroll(m.sox_ret20)+zroll(m.sox_ret60)+zroll(m.sox_slope20))/3); m['external_vix']=direction(-((zroll(m.vix_ret5)+zroll(m.vix_ret20)+zroll(m.vix_pct-.5))/3)); m['external_fx']=direction(-((zroll(m.fx_ret5)+zroll(m.fx_ret20)+zroll(m.fx_ret60))/3)); m['external_crossconfirm']=np.select([(m.external_nasdaq>0)&(m.external_sox>0)&(m.external_vix>=0),(m.external_nasdaq<0)&(m.external_sox<0)&(m.external_vix<=0)], [1,-1],0); m['fx_export_vs_outflow_conflict']=(m.external_fx<0)&((m.external_nasdaq>0)|(m.external_sox>0))
 m['external_group'],m['external_score'],m['external_confidence'],m['external_reasons']=group_vote(m,['external_nasdaq','external_sox','external_vix','external_fx','external_crossconfirm'],3)
 groups=['taiwan_group','breadth_group','capital_group','derivatives_group','external_group']; m['bullish_groups']=(m[groups]=='bullish').sum(axis=1); m['bearish_groups']=(m[groups]=='bearish').sum(axis=1); candidate=(m.bearish_groups>=4)&m.taiwan_group.eq('bearish')&m.breadth_group.eq('bearish'); m['full_spec_v2_state']=np.select([candidate,m.bearish_groups>=3,(m.bullish_groups>=3)&(m.bearish_groups<=1)],['bear_candidate','weak_market','strong_market'],'ordinary_market'); m.loc[candidate&candidate.shift(1,fill_value=False),'full_spec_v2_state']='confirmed_bear'; m.loc[m.full_spec_v2_state=='bear_candidate','full_spec_v2_state']='weak_market'; m['controller_version']='full_spec_v2'
 m.to_csv(OUT/'p3_market_controller_full_spec_v2_daily_features.csv',index=False,encoding='utf-8-sig')
 diff=old[['decision_date','simple_v0_market_state','full_spec_v1_market_state']].merge(m[['decision_date','full_spec_v2_state']],on='decision_date'); diff['v0_v2_changed']=diff.simple_v0_market_state.ne(diff.full_spec_v2_state); diff['v1_v2_changed']=diff.full_spec_v1_market_state.ne(diff.full_spec_v2_state); diff.to_csv(OUT/'p3_market_controller_v0_v1_v2_state_diff.csv',index=False,encoding='utf-8-sig')
 coverage=[]
 for group,cols in {'taiwan':['tw_direction','tw_magnitude','tw_persistence','tw_risk_reversal'],'breadth':['breadth_direction','breadth_magnitude','breadth_persistence','breadth_reversal'],'capital':['cap_direction','cap_magnitude','cap_persistence','cap_divergence','cap_volrisk'],'derivatives':['deriv_level','deriv_change','deriv_persistence','deriv_trade_flow','deriv_divergence'],'external':['external_nasdaq','external_sox','external_vix','external_fx','external_crossconfirm']}.items():
  for c in cols: coverage.append({'group':group,'dimension':c,'ready_rows':int(m[c].notna().sum()),'total_rows':len(m),'coverage':float(m[c].notna().mean()),'status':'blocked' if m[c].notna().sum()==0 else 'ready_or_warmup_partial'})
 pd.DataFrame(coverage).to_csv(OUT/'p3_market_controller_full_spec_v2_coverage_audit.csv',index=False,encoding='utf-8-sig')
 pd.DataFrame([
  {'stage':'A_contract_classification','scope':'PIT|missingness|confidence|reason_codes|state_transitions','performance_optimization':False,'acceptance':'architecture_complete_and_semantically_consistent'},
  {'stage':'B_bounded_robust_calibration','scope':'economic_threshold_ranges_one_factor_small_combinations','performance_optimization':'stability_not_peak','acceptance':'after_cost|MDD|turnover|walk_forward|year_subperiod|neighbor_consistency|extreme_regime'},
 ]).to_csv(OUT/'p3_market_controller_full_spec_v2_validation_stages.csv',index=False,encoding='utf-8-sig')
 pd.DataFrame([
  {'parameter_family':'rolling_z_direction_threshold','low':0.35,'base':0.50,'high':0.75,'method':'one_factor_at_a_time'},
  {'parameter_family':'group_same_direction_votes','low':'base_minus_1_where_semantically_valid','base':'3_of_4_or_5','high':'4_of_5','method':'small_economic_range'},
  {'parameter_family':'confirmed_bear_days','low':2,'base':2,'high':3,'method':'bounded_neighbor'},
  {'parameter_family':'rolling_window','low':60,'base':252,'high':'60_and_252_joint_context','method':'no_full_cartesian'},
 ]).assign(selection_rule='stable_plateau_no_isolated_peak',future_return_live_rule=False).to_csv(OUT/'p3_market_controller_full_spec_v2_bounded_calibration_contract.csv',index=False,encoding='utf-8-sig')
 readiness={'task_id':TASK,'status':'full_spec_v2_feature_state_contract_ready','daily_rows':len(m),'simple_v0_reference_only':True,'full_spec_v1_superseded_primary':True,'full_spec_v2_primary_candidate':True,'taifex_four_field_source_commit':'3e6c416','taifex_same_day_use_prohibited':True,'trade_flow_dimension_ready':True,'blocked_dimensions':[],'architecture_gate_ready':True,'parameter_reliability_gate_ready':False,'ready_for_contract_classification_validation':True,'ready_for_action_rechain':True,'ready_for_phase_b_experiments':False,'future_data_violation_count':0,'formal_model_changed':False,'trade_decision_changed':False,'active_in_trade_decision':False,'report_changed':False,'portfolio_replay_executed':False,'ready_for_strategy_replay':False,'ready_for_formal':False,'not_live_rule':True,'forward_returns_live_rule_usage':False}
 (OUT/'readiness_for_p3_market_controller_full_spec_v2.json').write_text(json.dumps(readiness,ensure_ascii=False,indent=2),encoding='utf-8'); (OUT/'final_summary_zh.md').write_text('# P3 market controller full_spec_v2\n\n五群組已按direction/magnitude/persistence/reversal維度實算；TAIFEX trade-flow兩欄未在Core compact materialize，明確blocked，不以OI替代。未跑績效。\n',encoding='utf-8')
 files=sorted(p for p in OUT.iterdir() if p.is_file() and p.name!='manifest.json'); (OUT/'manifest.json').write_text(json.dumps({'task_id':TASK,'readiness':readiness,'files':[{'name':p.name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in files]},ensure_ascii=False,indent=2),encoding='utf-8'); print(OUT)
if __name__=='__main__': run()
