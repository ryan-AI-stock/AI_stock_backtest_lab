"""PIT-aligned one-year official net chip-family panel for old AI7 research."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
RADAR=Path(r"C:\Users\zergv\Documents\Codex\2026-05-23\ai-stock-rotation-radar-https-docs\outputs\radar_old_ai7_one_year_three_chip_families_official_net_source_package_20260721")
DELTA=Path(r"C:\Users\zergv\Documents\Codex\2026-05-23\ai-stock-rotation-radar-https-docs\outputs\radar_vnext_p3_layer5_all80_continuous_lifecycle_adjusted_hlc_bounded_delta_acquisition_20260713\all80_bounded_delta_adjusted_hlc_exact_key_compact.csv.gz")
OUT=ROOT/"outputs/vnext_old_ai7_one_year_three_chip_families_pit_panel_contract_20260721"
TASK="TASK-BACKTEST-CORE-VNEXT-OLD-AI7-ONE-YEAR-THREE-CHIP-FAMILIES-PIT-PANEL-CONTRACT-001"

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(name):
    x=pd.read_csv(RADAR/name,dtype={"ticker":str}); x["date"]=pd.to_datetime(x["date"]).dt.tz_localize(None); return x
def zprior(x,col):
    x=x.sort_values(["ticker","decision_date"]); g=x.groupby("ticker",sort=False)[col]
    m=g.transform(lambda v:v.shift(1).rolling(60,min_periods=20).mean()); s=g.transform(lambda v:v.shift(1).rolling(60,min_periods=20).std(ddof=0))
    return (x[col]-m)/s.where(s.ne(0))
def weekly_aggregate(daily,tdcc):
    out=[]
    for ticker,releases in tdcc.groupby("ticker",sort=False):
        flow=daily.loc[daily.ticker.eq(ticker)].sort_values("decision_date"); previous=pd.NaT
        for rel in releases.sort_values("publication_date").itertuples(index=False):
            # Post-close daily flows are only eligible after their source date.
            part=flow.loc[(flow.decision_date>previous)&(flow.decision_date<rel.publication_date)]
            item=rel._asdict(); item.update({"prior_publication_date":previous,"flow_interval_start_exclusive":previous,"flow_interval_end_exclusive":rel.publication_date,"flow_trading_days":len(part)})
            for col in ("foreign_net","trust_net","margin_balance_change_from_prior_trade","short_balance_change_from_prior_trade"):
                v=pd.to_numeric(part[col],errors="coerce"); item[f"{col}_interval_sum"]=v.sum(min_count=1); item[f"{col}_interval_mean"]=v.mean(); item[f"{col}_observed_days"]=int(v.notna().sum())
            out.append(item); previous=rel.publication_date
    return pd.DataFrame(out)
def adjusted_audit(req):
    candidate_rows=0
    if DELTA.exists():
        d=pd.read_csv(DELTA,compression="gzip",dtype={"\ufeffticker":str}); t="\ufeffticker" if "\ufeffticker" in d else "ticker"; candidate_rows=int(d[t].isin(req.ticker).sum())
    audit=pd.DataFrame([
        {"candidate_source":"Core local adjusted cache","accepted_exact_adjusted_authority":False,"reason":"no accepted event-aware exact source lineage for this panel","candidate_rows_seen":0},
        {"candidate_source":str(DELTA),"accepted_exact_adjusted_authority":False,"reason":"partial P3 delta marked accepted_for_formal=false and human_review_required=true","candidate_rows_seen":candidate_rows},
    ])
    blocked=req.copy(); blocked["family"]="adjusted_analysis_close"; blocked["classification"]="accepted_adjusted_analysis_authority_missing"; blocked["reason"]="price returns/percentiles require exact event-aware adjusted close; raw close is not substituted"
    return audit,blocked
def main():
    OUT.mkdir(parents=True,exist_ok=True); (OUT/"current_step.txt").write_text("running_pit_panel_materialization\n",encoding="utf-8")
    inst,margin,raw=read("institutional_net_daily.csv.gz"),read("margin_short_daily.csv.gz"),read("raw_close_daily.csv.gz")
    tdcc=pd.read_csv(RADAR/"tdcc_400plus_weekly.csv",dtype={"ticker":str}); tdcc["publication_date"]=pd.to_datetime(tdcc["publication_date"]).dt.tz_localize(None)
    blocked_source=pd.read_csv(RADAR/"blocked_ledger.csv",dtype={"ticker":str}); blocked_source["date"]=pd.to_datetime(blocked_source["date"]).dt.tz_localize(None)
    dates=pd.DatetimeIndex(sorted(margin.date.drop_duplicates())); universe=margin[["ticker","name","market"]].drop_duplicates().sort_values("ticker")
    daily=pd.MultiIndex.from_product([dates,universe.ticker],names=["decision_date","ticker"]).to_frame(index=False).merge(universe,on="ticker",how="left")
    daily=daily.merge(raw.rename(columns={"date":"decision_date","close":"official_raw_close"})[["decision_date","ticker","official_raw_close","source_quality","source_url","source_hash"]],on=["decision_date","ticker"],how="left").rename(columns={"source_quality":"raw_close_source_quality","source_url":"raw_close_source_url","source_hash":"raw_close_source_hash"})
    daily=daily.merge(inst.rename(columns={"date":"decision_date"})[["decision_date","ticker","foreign_net","trust_net","source_quality","source_url","source_hash","available_at_policy"]],on=["decision_date","ticker"],how="left").rename(columns={"source_quality":"institutional_source_quality","source_url":"institutional_source_url","source_hash":"institutional_source_hash","available_at_policy":"institutional_available_at_policy"})
    daily=daily.merge(margin.rename(columns={"date":"decision_date"})[["decision_date","ticker","margin_balance","short_balance","margin_balance_change_from_prior_trade","short_balance_change_from_prior_trade","source_quality","source_url","source_hash","available_at_policy"]],on=["decision_date","ticker"],how="left").rename(columns={"source_quality":"margin_short_source_quality","source_url":"margin_short_source_url","source_hash":"margin_short_source_hash","available_at_policy":"margin_short_available_at_policy"})
    for col in ("foreign_net","trust_net","margin_balance_change_from_prior_trade","short_balance_change_from_prior_trade"): daily[f"{col}_ticker_prior_z60"]=zprior(daily,col).sort_index().to_numpy()
    raw_block=set(map(tuple,blocked_source.loc[blocked_source.family.eq("raw_close"),["ticker","date"]].to_records(index=False))); inst_block=set(map(tuple,blocked_source.loc[blocked_source.family.eq("institutional_net"),["ticker","date"]].to_records(index=False)))
    daily["raw_close_ready"]=daily.official_raw_close.notna(); daily["institutional_net_ready"]=daily.foreign_net.notna()|daily.trust_net.notna(); daily["margin_short_ready"]=daily.margin_balance.notna()|daily.short_balance.notna()
    daily["raw_close_explicit_source_blocked"]=daily.apply(lambda r:(r.ticker,r.decision_date) in raw_block,axis=1); daily["institutional_net_explicit_source_blocked"]=daily.apply(lambda r:(r.ticker,r.decision_date) in inst_block,axis=1)
    daily["raw_close_status"]=np.where(daily.raw_close_explicit_source_blocked,"explicit_source_blocked",np.where(daily.raw_close_ready,"ready_same_close","not_observed_no_substitution")); daily["institutional_net_status"]=np.where(daily.institutional_net_explicit_source_blocked,"explicit_source_blocked",np.where(daily.institutional_net_ready,"ready_post_close_next_trading_day","not_observed_no_substitution")); daily["margin_short_status"]=np.where(daily.margin_short_ready,"ready_post_close_next_trading_day","not_observed_no_substitution")
    req=daily[["ticker","decision_date","market"]].rename(columns={"decision_date":"date"}); audit,adjusted_blocked=adjusted_audit(req)
    daily["adjusted_analysis_close"]=np.nan; daily["adjusted_analysis_status"]="blocked_missing_accepted_exact_adjusted_authority"; daily["raw_as_adjusted_used"]=False; daily["future_data_violation"]=False
    weekly=weekly_aggregate(daily,tdcc); weekly["tdcc_exact_400_lots_available"]=False; weekly["tdcc_definition"]="strictly_more_than_400_lots; exactly_400_lots_not_isolable_and_excluded"; weekly["adjusted_analysis_close"]=np.nan
    for h in (1,2,4): weekly[f"past_return_{h}w"]=np.nan; weekly[f"future_return_{h}w_evaluation_only"]=np.nan
    for h in (13,26,52): weekly[f"trailing_price_percentile_{h}w"]=np.nan
    weekly["price_feature_status"]="blocked_missing_accepted_exact_adjusted_authority"; weekly["future_return_metadata_only"]=True
    coverage=pd.DataFrame([
        {"family":"raw_close","ready_rows":int(daily.raw_close_ready.sum()),"total_daily_rows":len(daily),"explicit_blocked_rows":int(daily.raw_close_explicit_source_blocked.sum()),"PIT_policy":"same decision close; never adjusted"},
        {"family":"institutional_net","ready_rows":int(daily.institutional_net_ready.sum()),"total_daily_rows":len(daily),"explicit_blocked_rows":int(daily.institutional_net_explicit_source_blocked.sum()),"PIT_policy":"post-close; next trading day eligible"},
        {"family":"margin_short","ready_rows":int(daily.margin_short_ready.sum()),"total_daily_rows":len(daily),"explicit_blocked_rows":0,"PIT_policy":"post-close; next trading day eligible"},
        {"family":"TDCC_strict_gt_400_lots","ready_rows":len(weekly),"total_daily_rows":len(daily),"explicit_blocked_rows":0,"PIT_policy":"publication date only; no forward fill"},
        {"family":"adjusted_analysis_close","ready_rows":0,"total_daily_rows":len(daily),"explicit_blocked_rows":len(adjusted_blocked),"PIT_policy":"accepted event-aware exact authority required; raw substitution forbidden"},])
    ticker_weeks=weekly.groupby("ticker",as_index=False).agg(weekly_authority_rows=("publication_date","size"),first_publication_date=("publication_date","min"),last_publication_date=("publication_date","max"))
    policy={"task_id":TASK,"scope":"daily_weekly_PIT_aligned_panel_only","institutional_requested_measure":"official foreign/trust net only","gross_buy_sell_used":False,"weekly_authority":"TDCC publication_date only; no forward fill","weekly_flow_aggregation":"previous publication exclusive through current publication exclusive because daily flows are next-trading-day eligible","cross_ticker_share_normalization":"not calculated; no exact issued-share denominator","ticker_standardization":"prior-only 60-trading-day zscore","raw_as_adjusted_used":False,"performance_authorized":False,"future_return_live_rule_usage":False,"future_data_violation_count":0,"formal_model_changed":False,"trade_decision_changed":False,"active_in_trade_decision":False,"report_changed":False,"not_live_rule":True}
    readiness={"ready_for_core_panel":True,"ready_for_experiments":False,"price_lead_lag_ready":False,"price_lead_lag_blocker":"accepted_adjusted_analysis_close_exact_authority_missing","adjusted_analysis_exact_requirement_rows":len(adjusted_blocked),"explicit_source_blocked_rows":int((daily.raw_close_explicit_source_blocked|daily.institutional_net_explicit_source_blocked).sum()),"tdcc_exact_400_lots_available":False,"future_data_violation_count":0,"performance_authorized":False}
    files={"daily_pit_aligned_three_chip_family_panel.csv.gz":daily.sort_values(["decision_date","ticker"]),"weekly_tdcc_publication_aligned_chip_panel.csv.gz":weekly.sort_values(["publication_date","ticker"]),"adjusted_analysis_exact_requirement_ledger.csv":adjusted_blocked,"adjusted_analysis_authority_audit.csv":audit,"coverage.csv":coverage,"ticker_week_coverage.csv":ticker_weeks,"source_blocked_ledger.csv":blocked_source}
    for name,frame in files.items(): frame.to_csv(OUT/name,index=False,compression="gzip" if name.endswith(".gz") else None,encoding=None if name.endswith(".gz") else "utf-8-sig")
    (OUT/"policy.json").write_text(json.dumps(policy,ensure_ascii=False,indent=2),encoding="utf-8"); (OUT/"readiness_for_core.json").write_text(json.dumps(readiness,ensure_ascii=False,indent=2),encoding="utf-8")
    manifest=[*files,"policy.json","readiness_for_core.json"]; (OUT/"manifest.json").write_text(json.dumps({"files":[{"path":n,"sha256":sha(OUT/n)} for n in manifest]},ensure_ascii=False,indent=2),encoding="utf-8")
    (OUT/"final_summary_zh.md").write_text(f"# old AI7 three-chip PIT panel\n\n- daily rows: {len(daily)}\n- TDCC publication-week rows: {len(weekly)}\n- adjusted authority blockers: {len(adjusted_blocked)}\n- performance/correlation/strategy work: not authorized\n",encoding="utf-8"); (OUT/"current_step.txt").write_text("completed_pit_panel_adjusted_authority_blocked\n",encoding="utf-8")
if __name__=="__main__": main()
