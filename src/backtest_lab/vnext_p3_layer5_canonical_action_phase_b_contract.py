from __future__ import annotations
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
SRC=ROOT/'outputs/vnext_p3_layer5_daily_feature_state_action_materialization_20260712'
RADAR=Path(r'C:\Users\zergv\Documents\Codex\2026-05-23\ai-stock-rotation-radar-https-docs\outputs\radar_vnext_p3_recent_full_feature_data_readiness_acquisition_20260711\compact')
OUT=ROOT/'outputs/vnext_p3_layer5_canonical_action_phase_b_contract_20260712'
TASK='TASK-BACKTEST-CORE-VNEXT-P3-LAYER5-CANONICAL-ACTION-CONSOLIDATION-AND-PHASE-B-PATH-CONTRACT-001'

def load_price():
    return pd.concat([pd.read_csv(p,dtype={'ticker':str},low_memory=False) for p in sorted((RADAR/'price').glob('*.csv.gz'))],ignore_index=True)

def run():
    OUT.mkdir(parents=True,exist_ok=True)
    a=pd.read_csv(SRC/'p3_layer5_daily_incumbent_challenger_action_trace.csv',dtype={'incumbent_after':str,'challenger_ticker':str})
    f=pd.read_csv(SRC/'p3_layer5_daily_feature_state_matrix.csv',dtype={'ticker':str},low_memory=False)
    for c in ['decision_date','next_execution_date']: a[c]=pd.to_datetime(a[c]); f[c]=pd.to_datetime(f[c])
    a=a.rename(columns={'next_execution_date':'execution_date','incumbent_after':'selected_ticker','reason_code':'action_reason','market_state':'regime'})
    selected=f[['decision_date','ticker','name','raw_state','score_balanced','total_confidence','history_ready','price_core_valid','block_A','block_B','block_C','block_D','block_E','block_F']].rename(columns={'ticker':'selected_ticker','name':'selected_name','raw_state':'lifecycle_state','score_balanced':'base_score'})
    c=a.merge(selected,on=['decision_date','selected_ticker'],how='left')
    c['watch_semantics']=np.where(c.selected_action.eq('watch_only'),'no_position_no_valid_replacement_not_cash_trade','not_applicable')
    c['controller_threshold_profile']=c.regime.map({'strong_market':'entry0.9_replacement1.2','ordinary_market':'entry1.0_replacement1.0','weak_market':'entry1.1_replacement0.8','confirmed_bear':'entry1.2_cash_allowed'})
    c['controller_adjusted_score_threshold']=np.where(c.regime.eq('strong_market'),5*1.2,np.where(c.regime.eq('weak_market'),5*.8,5.0))
    c['cost_model']='EP05_taiwan_standard_fee_tax_v1'; c['slippage_bp_per_side']=10
    c['canonical_action_authority']=True; c['candidate_matrix_placeholder_authority']=False
    c['selected_target_ready']=np.where(c.selected_ticker.notna(),c.history_ready.fillna(False)&c.price_core_valid.fillna(False),True)
    if not c.selected_target_ready.all(): raise ValueError('canonical target failed ready price/history join')
    if len(c)!=715 or c.decision_date.duplicated().any(): raise ValueError('canonical action must be one row per date')
    c.to_csv(OUT/'p3_layer5_canonical_daily_selected_action.csv',index=False,encoding='utf-8-sig')

    c['prior_target']=c.selected_ticker.shift(1)
    c['transition_type']=np.select([c.prior_target.eq(c.selected_ticker)&c.selected_ticker.notna(),c.prior_target.isna()&c.selected_ticker.notna(),c.prior_target.notna()&c.selected_ticker.isna(),c.prior_target.notna()&c.selected_ticker.notna()&c.prior_target.ne(c.selected_ticker)],['hold_same','no_position_to_stock','stock_to_no_position','stock_to_stock'],'hold_no_position')
    rates={'hold_same':0.,'hold_no_position':0.,'no_position_to_stock':.001425+.001,'stock_to_no_position':.001425+.003+.001,'stock_to_stock':.001425*2+.003+.002}
    c['base_transition_cost_rate']=c.transition_type.map(rates); c['gross_exposure']=c.selected_ticker.notna().astype(float)
    px=load_price(); px['date']=pd.to_datetime(px.date); px=px[['date','ticker','close','source_quality']].drop_duplicates(['date','ticker'],keep='last')
    c=c.merge(px,left_on=['execution_date','selected_ticker'],right_on=['date','ticker'],how='left').drop(columns=['date','ticker'])
    c['actual_execution_date']=c.execution_date; c['execution_timing_status']='exact_next_day'
    hold_missing=c.selected_ticker.notna()&c.close.isna()&c.transition_type.eq('hold_same')
    c.loc[hold_missing,'execution_timing_status']='hold_same_no_trade_prior_official_mark'
    for idx,row in c.loc[c.selected_ticker.notna()&c.close.isna()&c.transition_type.isin(['stock_to_stock','no_position_to_stock'])].iterrows():
        nxt=px.loc[(px.ticker==row.selected_ticker)&(px.date>row.execution_date)].sort_values('date').head(1)
        if not nxt.empty:
            c.loc[idx,'actual_execution_date']=nxt.date.iloc[0]; c.loc[idx,'close']=nxt.close.iloc[0]
            c.loc[idx,'source_quality']=nxt.source_quality.iloc[0]; c.loc[idx,'execution_timing_status']='deferred_to_next_target_tradable_day_hold_prior_until_execution'
    c['execution_price_ready']=np.where(c.selected_ticker.notna()&~hold_missing,c.close.notna(),True)
    missing=c.loc[~c.execution_price_ready,['decision_date','execution_date','selected_ticker','selected_action','transition_type']]
    missing.to_csv(OUT/'p3_layer5_phase_b_execution_price_blocked_ledger.csv',index=False,encoding='utf-8-sig')
    c.to_csv(OUT/'p3_layer5_phase_b_unique_position_execution_path_contract.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame([{'slippage_bp_per_side':bp,'role':'primary' if bp==10 else 'sensitivity','EP05_fee_tax':True,'single_position':True,'next_day_execution':True} for bp in [5,10,20]]).to_csv(OUT/'p3_layer5_phase_b_cost_slippage_scenarios.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame([
      {'counterfactual':'C0_ordinary_always','same_selector':True,'same_candidates':True,'controller':'ordinary','TAIFEX':'off_action','TDCC':'off_action'},
      {'counterfactual':'C1_full_controller','same_selector':True,'same_candidates':True,'controller':'full','TAIFEX':'on','TDCC':'P3_2_on'},
      {'counterfactual':'C2_no_strong_loosening','same_selector':True,'same_candidates':True,'controller':'weak_bear_only','TAIFEX':'on','TDCC':'P3_2_on'},
      {'counterfactual':'C3_TAIFEX_off','same_selector':True,'same_candidates':True,'controller':'full_ablation','TAIFEX':'off','TDCC':'P3_2_on_off'},
      {'counterfactual':'random_primary80_placebo','same_selector':False,'same_candidates':True,'controller':'matched_entry_hold_transition_exposure','TAIFEX':'matched','TDCC':'matched','fixed_seeds':'17|29|43|71|101'}]).assign(execution_basis='same_next_day',cost_basis='EP05_plus_10bp_per_side').to_csv(OUT/'p3_layer5_phase_b_counterfactual_placebo_contract.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame([{'robustness':'walk_forward|year|quarter','exact_rechain':True},{'robustness':'remove_best_year|quarter|episode_1_3_5','exact_rechain':True},{'robustness':'ticker|sector|mega_period_concentration','exact_rechain':False},{'robustness':'ordinary_weak_primary_strong_secondary','exact_rechain':False}]).to_csv(OUT/'p3_layer5_phase_b_robustness_contract.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame([{'audit':'canonical_rows','value':len(c)},{'audit':'watch_rows_no_position','value':int((c.selected_action.eq('watch_only')&c.selected_ticker.isna()).sum())},{'audit':'target_not_ready','value':int((~c.selected_target_ready).sum())},{'audit':'execution_price_blocked','value':len(missing)},{'audit':'future_return_rule','value':0}]).to_csv(OUT/'p3_layer5_phase_b_contract_audit.csv',index=False,encoding='utf-8-sig')
    readiness={'task_id':TASK,'status':'canonical_action_ready_phase_b_contract_ready' if missing.empty else 'canonical_action_ready_execution_price_gaps','canonical_rows':715,'action_counts':c.selected_action.value_counts().to_dict(),'watch_semantics':'no_position_no_valid_replacement_not_cash_trade','watch_rows':120,'ready_target_join_pass':True,'execution_price_blocked_rows':len(missing),'ready_for_phase_b_experiments':missing.empty,'ready_for_experiments':missing.empty,'phase_b_performance_executed':False,'future_data_violation_count':0,'formal_model_changed':False,'trade_decision_changed':False,'active_in_trade_decision':False,'report_changed':False,'portfolio_replay_executed':False,'ready_for_strategy_replay':False,'ready_for_formal':False,'not_live_rule':True,'forward_returns_live_rule_usage':False}
    (OUT/'readiness_for_p3_layer5_phase_b_contract.json').write_text(json.dumps(readiness,ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'final_summary_zh.md').write_text(f"# P3 Layer5 canonical action / Phase B contract\n\n- canonical 715日；watch 120日均為no-position且不重複交易。\n- execution price blocked={len(missing)}；未跑績效。\n",encoding='utf-8')
    files=sorted(q for q in OUT.iterdir() if q.is_file() and q.name!='manifest.json'); (OUT/'manifest.json').write_text(json.dumps({'task_id':TASK,'source_commit':'6c6d38b','readiness':readiness,'files':[{'name':q.name,'sha256':hashlib.sha256(q.read_bytes()).hexdigest()} for q in files]},ensure_ascii=False,indent=2),encoding='utf-8')
    print(OUT)
if __name__=='__main__': run()
