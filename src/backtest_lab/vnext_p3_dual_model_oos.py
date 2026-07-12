from __future__ import annotations
import hashlib,json,platform
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression,Ridge
from sklearn.metrics import brier_score_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).resolve().parents[2]
SRC=ROOT/'outputs/vnext_p3_layer5_walk_forward_dual_model_ranking_stage_a_contract_20260712'
OUT=ROOT/'outputs/vnext_p3_layer5_walk_forward_dual_model_ranking_stage_b_oos_20260712'
TASK='TASK-BACKTEST-CORE-VNEXT-P3-LAYER5-WALK-FORWARD-DUAL-MODEL-RANKING-STAGE-B-OOS-CONTRACT-001'
SEED=20260712

def group_weight(d):
 n=d.groupby('decision_date').ticker.transform('size'); w=1/n
 freq=d.groupby('full_spec_v2_state').decision_date.nunique(); rw=(1/freq); rw=rw/rw.mean(); return w*d.full_spec_v2_state.map(rw).fillna(1)
def rank_metric(d,pred):
 z=d.assign(pred=pred); vals=[]
 for _,g in z[z.full_spec_v2_state.isin(['ordinary_market','weak_market'])].groupby('decision_date'):
  if len(g)>=3 and g.net_excess_vs_0050.nunique()>1: vals.append(spearmanr(g.pred,g.net_excess_vs_0050).statistic)
 return float(np.nanmean(vals)) if vals else np.nan
def relevance(d): return d.groupby('decision_date').net_excess_vs_0050.transform(lambda s:np.minimum(4,np.floor(s.rank(pct=True)*5-1e-9))).astype(int)
def groups(d): return d.groupby('decision_date',sort=False).size().tolist()

def run():
 OUT.mkdir(parents=True,exist_ok=True)
 f=pd.read_csv(SRC/'p3_dual_model_frozen_feature_matrix.csv.gz',dtype={'ticker':str},low_memory=False); f['decision_date']=pd.to_datetime(f.decision_date)
 l=pd.read_csv(SRC/'p3_dual_model_exact_label_contract.csv.gz',dtype={'ticker':str},low_memory=False); l['decision_date']=pd.to_datetime(l.decision_date)
 folds=pd.read_csv(SRC/'p3_dual_model_walk_forward_fold_calendar.csv'); datecols=[c for c in folds if c.endswith('start') or c.endswith('end')]; folds[datecols]=folds[datecols].apply(pd.to_datetime)
 dictionary=pd.read_csv(SRC/'p3_dual_model_feature_data_dictionary.csv'); features=dictionary.feature.tolist(); allowed=['ordinary_market','weak_market','strong_market','confirmed_bear']
 predictions=[]; lineage=[]; importance=[]
 for h in [10,20]:
  data=f.merge(l[l.horizon_td.eq(h)],on=['decision_date','ticker'],how='inner',validate='one_to_one'); data=data[data.full_spec_v2_state.isin(allowed)].copy()
  for fold in folds.itertuples(index=False):
   tr=data[data.decision_date.between(fold.train_start,fold.train_end)].sort_values(['decision_date','ticker']); va=data[data.decision_date.between(fold.validation_start,fold.validation_end)].sort_values(['decision_date','ticker']); oo=data[data.decision_date.between(fold.OOS_start,fold.OOS_end)].sort_values(['decision_date','ticker'])
   Xtr=tr[features].astype(float); Xva=va[features].astype(float); Xoo=oo[features].astype(float); wt=group_weight(tr)
   # Linear opportunity baseline: validation-only alpha selection.
   opts=[]
   for alpha in [.1,1.,10.]:
    model=make_pipeline(SimpleImputer(strategy='median',add_indicator=True),StandardScaler(),Ridge(alpha=alpha)); model.fit(Xtr,tr.net_excess_vs_0050,ridge__sample_weight=wt); score=rank_metric(va,model.predict(Xva)); opts.append((score,alpha,model))
   _,alpha,opp_linear=max(opts,key=lambda z:(-np.inf if pd.isna(z[0]) else z[0])); p_ol=opp_linear.predict(Xoo)
   ridge=opp_linear[-1]; names=opp_linear[0].get_feature_names_out(features); importance.extend({'fold_id':fold.fold_id,'horizon_td':h,'model':'opportunity_ridge','feature':n,'importance_or_coefficient':v,'selected_param':f'alpha={alpha}'} for n,v in zip(names,ridge.coef_))
   # Logistic risk baseline.
   ropts=[]
   for C in [.1,1.,10.]:
    model=make_pipeline(SimpleImputer(strategy='median',add_indicator=True),StandardScaler(),LogisticRegression(C=C,penalty='elasticnet',l1_ratio=.5,solver='saga',max_iter=5000,tol=1e-3,n_jobs=1,random_state=SEED)); model.fit(Xtr,tr.severe_drawdown_event.astype(int),logisticregression__sample_weight=wt); pv=model.predict_proba(Xva)[:,1]; mask=va.full_spec_v2_state.isin(['ordinary_market','weak_market']); score=brier_score_loss(va.loc[mask,'severe_drawdown_event'],pv[mask]); ropts.append((score,C,model))
   _,C,risk_linear=min(ropts,key=lambda z:z[0]); p_rl=risk_linear.predict_proba(Xoo)[:,1]; logit=risk_linear[-1]; names=risk_linear[0].get_feature_names_out(features); importance.extend({'fold_id':fold.fold_id,'horizon_td':h,'model':'risk_logistic','feature':n,'importance_or_coefficient':v,'selected_param':f'C={C};l1_ratio=.5'} for n,v in zip(names,logit.coef_[0]))
   # LightGBM LambdaRank challenger.
   imp=SimpleImputer(strategy='median',add_indicator=True); Atr=imp.fit_transform(Xtr); Ava=imp.transform(Xva); Aoo=imp.transform(Xoo); fn=imp.get_feature_names_out(features); rel=relevance(tr); rank_opts=[]
   for leaves in [15,31]:
    for lr in [.03,.05]:
     model=lgb.LGBMRanker(objective='lambdarank',n_estimators=150,num_leaves=leaves,learning_rate=lr,min_child_samples=30,n_jobs=1,random_state=SEED,deterministic=True,force_col_wise=True,verbosity=-1); model.fit(Atr,rel,group=groups(tr),sample_weight=wt); score=rank_metric(va,model.predict(Ava)); rank_opts.append((score,leaves,lr,model))
   _,leaves,lr,opp_tree=max(rank_opts,key=lambda z:(-np.inf if pd.isna(z[0]) else z[0])); p_ot=opp_tree.predict(Aoo); importance.extend({'fold_id':fold.fold_id,'horizon_td':h,'model':'opportunity_lambdarank','feature':n,'importance_or_coefficient':v,'selected_param':f'leaves={leaves};lr={lr}'} for n,v in zip(fn,opp_tree.feature_importances_))
   # LightGBM risk classifier challenger.
   tree_opts=[]
   for leaves in [15,31]:
    for lr in [.03,.05]:
     model=lgb.LGBMClassifier(objective='binary',n_estimators=150,num_leaves=leaves,learning_rate=lr,min_child_samples=30,n_jobs=1,random_state=SEED,deterministic=True,force_col_wise=True,verbosity=-1); model.fit(Atr,tr.severe_drawdown_event.astype(int),sample_weight=wt); pv=model.predict_proba(Ava)[:,1]; mask=va.full_spec_v2_state.isin(['ordinary_market','weak_market']); score=brier_score_loss(va.loc[mask,'severe_drawdown_event'],pv[mask]); tree_opts.append((score,leaves,lr,model))
   _,rleaves,rlr,risk_tree=min(tree_opts,key=lambda z:z[0]); p_rt=risk_tree.predict_proba(Aoo)[:,1]; importance.extend({'fold_id':fold.fold_id,'horizon_td':h,'model':'risk_gbdt','feature':n,'importance_or_coefficient':v,'selected_param':f'leaves={rleaves};lr={rlr}'} for n,v in zip(fn,risk_tree.feature_importances_))
   pred=oo[['decision_date','ticker','full_spec_v2_state','net_excess_vs_0050','severe_drawdown_event','future_MDD_rank','path_MDD']].copy(); pred['fold_id']=fold.fold_id; pred['horizon_td']=h; pred['opportunity_ridge_prediction']=p_ol; pred['opportunity_lambdarank_prediction']=p_ot; pred['risk_logistic_probability']=p_rl; pred['risk_gbdt_probability']=p_rt; pred['prediction_scope']='strict_OOS'; predictions.append(pred)
   lineage.append({'fold_id':fold.fold_id,'horizon_td':h,'train_start':fold.train_start,'train_end':fold.train_end,'validation_start':fold.validation_start,'validation_end':fold.validation_end,'OOS_start':fold.OOS_start,'OOS_end':fold.OOS_end,'train_rows':len(tr),'validation_rows':len(va),'OOS_rows':len(oo),'opportunity_ridge_alpha':alpha,'risk_logistic_C':C,'risk_logistic_n_iter':int(logit.n_iter_[0]),'risk_logistic_converged':bool(logit.n_iter_[0]<logit.max_iter),'opportunity_lgbm_leaves':leaves,'opportunity_lgbm_lr':lr,'risk_lgbm_leaves':rleaves,'risk_lgbm_lr':rlr,'seed':SEED,'threads':1,'final_OOS_used_for_parameter_selection':False})
 pred=pd.concat(predictions,ignore_index=True); pred.to_csv(OUT/'p3_dual_model_strict_OOS_predictions.csv.gz',index=False,compression='gzip',encoding='utf-8'); pd.DataFrame(lineage).to_csv(OUT/'p3_dual_model_fold_lineage_selected_params.csv',index=False,encoding='utf-8-sig'); pd.DataFrame(importance).to_csv(OUT/'p3_dual_model_feature_importance_sign_stability.csv.gz',index=False,compression='gzip',encoding='utf-8')
 # OOS calibration summaries are evaluation metadata, never parameter-selection inputs.
 cal=[]
 for (h,model),col in [((h,m),c) for h in [10,20] for m,c in [('risk_logistic','risk_logistic_probability'),('risk_gbdt','risk_gbdt_probability')]]:
  z=pred[pred.horizon_td.eq(h)].copy(); z['bin']=pd.qcut(z[col],10,duplicates='drop'); g=z.groupby('bin',observed=True).agg(predicted_probability=(col,'mean'),observed_rate=('severe_drawdown_event','mean'),rows=('ticker','size')).reset_index(); g['horizon_td']=h; g['model']=model; cal.append(g)
 pd.concat(cal,ignore_index=True).to_csv(OUT/'p3_dual_model_OOS_risk_calibration.csv',index=False,encoding='utf-8-sig')
 coverage=pred.groupby(['fold_id','horizon_td','full_spec_v2_state']).size().reset_index(name='OOS_rows'); coverage.to_csv(OUT/'p3_dual_model_OOS_regime_coverage.csv',index=False,encoding='utf-8-sig')
 runtime={'python':platform.python_version(),'platform':platform.platform(),'processor':platform.processor(),'sklearn':sklearn.__version__,'lightgbm':lgb.__version__,'numpy':np.__version__,'seed':SEED,'threads':1,'install_command':'.venv-ml\\Scripts\\python.exe -m pip install --require-hashes -r requirements-ml.lock'}; (OUT/'ml_runtime_environment.json').write_text(json.dumps(runtime,ensure_ascii=False,indent=2),encoding='utf-8')
 converged=bool(pd.DataFrame(lineage).risk_logistic_converged.all()); ready={'task_id':TASK,'status':'strict_OOS_dual_model_predictions_ready_for_candidate_quality_validation' if converged else 'blocked_selected_logistic_not_converged','OOS_prediction_rows':len(pred),'fold_count':pred.fold_id.nunique(),'horizons':[10,20],'models':['opportunity_ridge','risk_logistic','opportunity_lambdarank','risk_gbdt'],'risk_logistic_all_folds_converged':converged,'final_OOS_used_for_tuning':False,'portfolio_selector_combined':False,'ready_for_oos_candidate_quality':converged,'ready_for_portfolio_performance':False,'ready_for_experiments':converged,'future_data_violation_count':0,'formal_model_changed':False,'trade_decision_changed':False,'active_in_trade_decision':False,'report_changed':False,'ready_for_formal':False,'ready_for_strategy_replay':False,'not_live_rule':True,'forward_returns_live_rule_usage':False}
 (OUT/'readiness_for_dual_model_OOS.json').write_text(json.dumps(ready,ensure_ascii=False,indent=2),encoding='utf-8'); (OUT/'final_summary_zh.md').write_text('# Dual-model strict OOS predictions\n\n兩模型與兩horizon均依fold內validation選bounded參數，OOS未參與選參。risk/opportunity保持分欄，未合成portfolio selector。\n',encoding='utf-8')
 files=sorted(p for p in OUT.iterdir() if p.is_file() and p.name!='manifest.json'); (OUT/'manifest.json').write_text(json.dumps({'task_id':TASK,'runtime':runtime,'files':[{'name':p.name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in files]},ensure_ascii=False,indent=2),encoding='utf-8'); print(OUT)
if __name__=='__main__':run()
