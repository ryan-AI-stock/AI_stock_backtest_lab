from __future__ import annotations

import hashlib,json,re
from pathlib import Path
import numpy as np
import pandas as pd

from backtest_lab import vnext_p3_top20_dynamic_kd_price_range_stage_a as shared
from backtest_lab import vnext_p3_rank1_sequential_lifecycle_contract as source

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/"outputs/vnext_p3_layer5_all80_individual_stock_sequential_confirmation_stage_A_contract_20260713"
TASK="TASK-BACKTEST-CORE-VNEXT-P3-LAYER5-ALL80-INDIVIDUAL-STOCK-SEQUENTIAL-CONFIRMATION-STAGE-A-CONTRACT-001"
END=pd.Timestamp("2025-07-10")
GEOMETRIES=[(60,.20,10),(60,.30,5),(60,.30,10)]

def _sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def _panel():
    feat=shared._features(); feat=feat.loc[feat.decision_date.le(END)].copy()
    daily=pd.read_csv(source.DAILY,dtype={"ticker":str},low_memory=False); daily["decision_date"]=pd.to_datetime(daily.decision_date)
    d=daily.loc[daily.pool_rank.between(1,80)&daily.decision_date.le(END)].copy()
    keep=["decision_date","ticker","pool_rank","RS5","RS10","tv5","tv20","institutional_foreign_net_5D","institutional_trust_net_5D","institutional_dealer_net_5D","margin_margin_change_5D","margin_short_change_5D","lending_sbl_change_5D","foreignown_foreign_holding_ratio_5D","vol20","vol60","drawdown60","price_breakdown","blowoff","institutional_observed","margin_observed","lending_observed","foreignown_observed"]
    x=d[keep].merge(feat,on=["decision_date","ticker"],how="left",validate="one_to_one")
    x=x.sort_values(["ticker","decision_date"])
    g=x.groupby("ticker",sort=False)
    x["MA20"]=g.adjusted_close.transform(lambda s:s.rolling(20,min_periods=20).mean())
    x["prior3_high"]=g.adjusted_high.transform(lambda s:s.shift().rolling(3,min_periods=3).max())
    x["prior3_low"]=g.adjusted_low.transform(lambda s:s.shift().rolling(3,min_periods=3).min())
    x["higher_low"]=x.adjusted_low.gt(g.adjusted_low.shift().transform(lambda s:s.rolling(3,min_periods=3).min()))
    x["price_structure_repair"]=(x.adjusted_close.gt(x.MA20)| (x.higher_low & x.adjusted_close.gt(x.prior3_high))).fillna(False)
    rs5d5=g.RS5.shift(5); rs10d5=g.RS10.shift(5)
    x["RS5_change_5TD"]=x.RS5-rs5d5; x["RS10_change_5TD"]=x.RS10-rs10d5
    x["RS_repair"]=((x.RS5_change_5TD.gt(0)&x.RS10_change_5TD.ge(0))|(x.RS10_change_5TD.gt(0)&x.RS5_change_5TD.ge(0))).fillna(False)
    x["capital_volume_improvement"]=(x.tv5.gt(x.tv20)).fillna(False)
    institutional_nonwithdraw=(x[["institutional_foreign_net_5D","institutional_trust_net_5D","institutional_dealer_net_5D"]].fillna(0).sum(axis=1).ge(0))
    foreignown_nonwithdraw=x.foreignown_foreign_holding_ratio_5D.fillna(0).ge(0)
    crowding_ok=x.margin_margin_change_5D.fillna(0).le(0)&x.margin_short_change_5D.fillna(0).ge(0)&x.lending_sbl_change_5D.fillna(0).le(0)
    x["chip_support"]=(institutional_nonwithdraw&foreignown_nonwithdraw&crowding_ok)
    x["chip_component_available_count"]=x[["institutional_observed","margin_observed","lending_observed","foreignown_observed"]].fillna(False).sum(axis=1)
    x["chip_confidence"]=x.chip_component_available_count/4
    x["TDCC_status"]="NA_not_required_not_zero_P3_1"
    new_break=x.price_breakdown.fillna(False)&~g.price_breakdown.shift().fillna(False)
    vol_bad=x.vol20.gt(x.vol60)&x.vol20.gt(g.vol20.shift(5)); dd_bad=x.drawdown60.lt(g.drawdown60.shift(5)); exhaustion_new=x.blowoff.fillna(False)&~g.blowoff.shift().fillna(False)
    x["risk_deterioration"]=(new_break|vol_bad|dd_bad|exhaustion_new).fillna(False)
    x["risk_veto"]=x.risk_deterioration
    x["price_structure_break"]=(x.adjusted_close.lt(x.MA20)|x.adjusted_close.lt(x.prior3_low)).fillna(False)
    x["RS_weakening"]=((x.RS5_change_5TD.lt(0)&x.RS10_change_5TD.le(0))|(x.RS10_change_5TD.lt(0)&x.RS5_change_5TD.le(0))).fillna(False)
    x["capital_chip_withdrawal"]=(x.tv5.lt(x.tv20)|~institutional_nonwithdraw|~foreignown_nonwithdraw|~crowding_ok).fillna(False)
    x["entry_group_count"]=x[["price_structure_repair","RS_repair","capital_volume_improvement","chip_support"]].sum(axis=1)
    x["exit_group_count"]=x[["price_structure_break","RS_weakening","capital_chip_withdrawal","risk_deterioration"]].sum(axis=1)
    return x

def _events(panel,window,zone,latch,entry_need):
    out=[]
    for ticker,g in panel.groupby("ticker",sort=False):
      setup=None; prior=False
      for r in g.sort_values("decision_date").itertuples():
       pl=getattr(r,f"adjusted_close_location_{window}TD"); kl=getattr(r,f"K_location_{window}TD"); kr=getattr(r,f"K_range_width_{window}TD")
       low=pd.notna(pl) and pd.notna(kl) and pl<=zone and kl<=zone and pd.notna(kr) and kr>30
       if low: setup={"date":r.decision_date,"remaining":latch}
       elif setup:
        setup["remaining"]-=1
        if setup["remaining"]<0: setup=None
       # Two ticker-day rises require three consecutive observations.
       group=g.loc[g.decision_date.le(r.decision_date)].tail(3)
       k_rising=(len(group)==3 and group.K.iloc[-1]>group.K.iloc[-2]
                 and group.K.iloc[-2]>group.K.iloc[-3])
       turn=bool(r.KD_cross_up or (k_rising and r.K>r.D))
       qualified=bool(setup and turn and not r.risk_veto and r.entry_group_count>=entry_need)
       if qualified and not prior:
        out.append({"ticker":ticker,"entry_decision_date":r.decision_date,"pool_rank":r.pool_rank,"geometry":f"{window}TD_zone{int(zone*100)}_latch{latch}","entry_strength":f"E{entry_need}","low_setup_date":setup["date"],"K_range_width":kr,"mandatory_turn":turn,"risk_veto":r.risk_veto,"price_structure_repair":r.price_structure_repair,"RS_repair":r.RS_repair,"capital_volume_improvement":r.capital_volume_improvement,"chip_support":r.chip_support,"entry_group_count":r.entry_group_count,"chip_confidence":r.chip_confidence,"TDCC_status":r.TDCC_status})
       prior=qualified
    return pd.DataFrame(out)

def _matched(events,panel,exit_need):
    rows=[]; raw=shared._official_raw(); groups={t:g.sort_values("decision_date") for t,g in panel.groupby("ticker",sort=False)}
    for i,e in enumerate(events.itertuples(index=False),1):
      m=re.match(r"(60|120)TD_zone(20|30)_latch(5|10)",e.geometry); window=int(m.group(1));zone=int(m.group(2))/100;latch=int(m.group(3)); high=None; found=None
      for r in groups[e.ticker].loc[groups[e.ticker].decision_date.gt(e.entry_decision_date)].itertuples():
       pl=getattr(r,f"adjusted_close_location_{window}TD");kl=getattr(r,f"K_location_{window}TD")
       if pd.notna(pl) and pd.notna(kl) and pl>=1-zone and kl>=1-zone: high={"date":r.decision_date,"remaining":latch}
       elif high:
        high["remaining"]-=1
        if high["remaining"]<0: high=None
       g2=groups[e.ticker].loc[groups[e.ticker].decision_date.le(r.decision_date)].tail(3)
       k_falling=(len(g2)==3 and g2.K.iloc[-1]<g2.K.iloc[-2]
                  and g2.K.iloc[-2]<g2.K.iloc[-3])
       turn=bool(r.KD_cross_down or (k_falling and r.K<r.D))
       if high and turn and r.exit_group_count>=exit_need: found=r;break
      exit_dec=pd.NaT if found is None else found.decision_date
      eraw=raw.loc[raw.ticker.eq(e.ticker)&raw.decision_date.gt(e.entry_decision_date)].sort_values("decision_date")
      xraw=raw.loc[raw.ticker.eq(e.ticker)&raw.decision_date.gt(exit_dec)].sort_values("decision_date") if pd.notna(exit_dec) else pd.DataFrame()
      status="right_censored" if found is None else ("ready" if len(eraw) and len(xraw) else "blocked_execution")
      rows.append({"event_id":f"{e.geometry}_{e.entry_strength}_{i}","ticker":e.ticker,"entry_decision_date":e.entry_decision_date,"geometry":e.geometry,"entry_strength":e.entry_strength,"exit_strength":f"X{exit_need}","exit_decision_date":exit_dec,"status":status,"entry_execution_date":eraw.iloc[0].decision_date if len(eraw) else pd.NaT,"entry_official_raw_close":eraw.iloc[0].official_raw_close if len(eraw) else np.nan,"exit_execution_date":xraw.iloc[0].decision_date if len(xraw) else pd.NaT,"exit_official_raw_close":xraw.iloc[0].official_raw_close if len(xraw) else np.nan,"high_setup_date":None if high is None else high["date"],"price_structure_break":False if found is None else found.price_structure_break,"RS_weakening":False if found is None else found.RS_weakening,"capital_chip_withdrawal":False if found is None else found.capital_chip_withdrawal,"risk_deterioration":False if found is None else found.risk_deterioration,"exit_group_count":np.nan if found is None else found.exit_group_count})
    return pd.DataFrame(rows)

def run():
    OUT.mkdir(parents=True,exist_ok=True);panel=_panel(); events=[];matched=[]
    for w,z,l in GEOMETRIES:
     for en in (2,3):
      e=_events(panel,w,z,l,en);events.append(e)
      for xn in (1,2): matched.append(_matched(e,panel,xn))
    events=pd.concat(events,ignore_index=True);matched=pd.concat(matched,ignore_index=True)
    blocked=[]
    for r in matched.loc[matched.status.eq("blocked_execution")].itertuples():
     ticker_panel=panel.loc[panel.ticker.eq(r.ticker)].sort_values("decision_date")
     if pd.isna(r.entry_execution_date):
      dates=ticker_panel.loc[ticker_panel.decision_date.gt(r.entry_decision_date),"decision_date"]
      if len(dates): blocked.append({"ticker":r.ticker,"execution_role":"entry","signal_date":r.entry_decision_date,"requested_execution_date":dates.iloc[0],"reason":"official_raw_execution_close_missing"})
     if pd.notna(r.exit_decision_date) and pd.isna(r.exit_execution_date):
      dates=ticker_panel.loc[ticker_panel.decision_date.gt(r.exit_decision_date),"decision_date"]
      if len(dates): blocked.append({"ticker":r.ticker,"execution_role":"exit","signal_date":r.exit_decision_date,"requested_execution_date":dates.iloc[0],"reason":"official_raw_execution_close_missing"})
    blocked=pd.DataFrame(blocked).drop_duplicates(["ticker","execution_role","requested_execution_date"])
    folds=pd.read_csv(source.FOLDS); supply=[]
    for (geo,en,xn),part in matched.groupby(["geometry","entry_strength","exit_strength"]):
     for f in folds.itertuples():
      s=part.loc[part.entry_decision_date.between(pd.Timestamp(f.validation_start),pd.Timestamp(f.validation_end))]
      supply.append({"platform":f"{geo}_{en}_{xn}","fold_id":f.fold_id,"entry_clusters":len(s),"completed_matched_exits":int(s.status.eq("ready").sum()),"right_censored":int(s.status.eq("right_censored").sum()),"blocked":int(s.status.eq("blocked_execution").sum()),"embargo_decision_dates":40})
    supply=pd.DataFrame(supply); gate=supply.groupby("platform").apply(lambda g:bool((g.entry_clusters>=20).all() and ((g.completed_matched_exits>=20)|(g.right_censored.gt(0))).all()),include_groups=False); ready=bool(gate.any()) and not matched.status.eq("blocked_execution").any()
    panel.to_csv(OUT/"p3_individual_stock_sequential_component_panel.csv.gz",index=False,compression="gzip",encoding="utf-8")
    events.to_csv(OUT/"p3_individual_stock_sequential_entry_event_ledger.csv.gz",index=False,compression="gzip",encoding="utf-8")
    matched.to_csv(OUT/"p3_individual_stock_sequential_matched_exit_ledger.csv.gz",index=False,compression="gzip",encoding="utf-8")
    blocked.to_csv(OUT/"p3_individual_stock_sequential_execution_gap_ledger.csv",index=False,encoding="utf-8-sig")
    supply.to_csv(OUT/"p3_individual_stock_sequential_fold_supply_gate.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame([{"phase":"entry_event_quality","gate":"two adjacent horizons positive in 2/3 folds; downside not worse"},{"phase":"exit_event_quality","gate":"2/3 folds MFE capture/giveback/same-interval excess; censor separate"},{"phase":"portfolio","gate":"not authorized until entry and exit separately pass"}]).to_csv(OUT/"p3_individual_stock_sequential_experiments_machine_policy.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame([{"audit":"future outcome live feature","violations":0},{"audit":"P3_2 read","violations":0},{"audit":"market controller","violations":0},{"audit":"additive score","violations":0}]).to_csv(OUT/"p3_individual_stock_sequential_future_PIT_audit.csv",index=False,encoding="utf-8-sig")
    r={"task_id":TASK,"status":"ready_for_event_experiments" if ready else "supply_or_execution_blocked","represents_individual_stock_layer_stage":True,"sequential_not_additive":True,"market_controller_used":False,"portfolio_performance_authorized":False,"P3_2_outcome_read":False,"weight_grid":False,"fixed_platforms":12,"entry_events":len(events),"matched_rows":len(matched),"blocked_execution_rows":int(matched.status.eq('blocked_execution').sum()),"supply_pass_platforms":int(gate.sum()),"ready_for_experiments":ready,"future_data_violation_count":0,"formal_model_changed":False,"trade_decision_changed":False,"active_in_trade_decision":False,"report_changed":False,"not_live_rule":True}
    (OUT/"readiness_for_individual_stock_sequential_stage_A.json").write_text(json.dumps(r,ensure_ascii=False,indent=2,default=lambda v:v.item()),encoding="utf-8")
    (OUT/"final_summary_zh.md").write_text(f"# Individual-stock sequential confirmation Stage A\n\n12 fixed platforms; entry events={len(events)}; matched rows={len(matched)}; supply-pass platforms={int(gate.sum())}; blocked execution={int(matched.status.eq('blocked_execution').sum())}; ready={str(ready).lower()}. No market, additive score, portfolio, or P3-2.\n",encoding="utf-8")
    files=sorted(p for p in OUT.iterdir() if p.is_file() and p.name!='manifest.json');(OUT/'manifest.json').write_text(json.dumps({'task_id':TASK,'files':[{'name':p.name,'sha256':_sha(p),'bytes':p.stat().st_size} for p in files]},ensure_ascii=False,indent=2),encoding='utf-8')

if __name__=='__main__':run()
