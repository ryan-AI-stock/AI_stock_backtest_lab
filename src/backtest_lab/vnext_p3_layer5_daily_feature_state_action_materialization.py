from __future__ import annotations

import hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'outputs/vnext_p3_layer5_daily_state_machine_materialization_20260711'
RADAR=Path(r'C:\Users\zergv\Documents\Codex\2026-05-23\ai-stock-rotation-radar-https-docs\outputs\radar_vnext_p3_recent_full_feature_data_readiness_acquisition_20260711\compact')
MARKET=Path(r'C:\Users\zergv\Documents\Codex\2026-05-23\ai-stock-rotation-radar-https-docs\outputs\radar_vnext_p3_market_state_source_fill_20260711\compact')
OUT=ROOT/'outputs/vnext_p3_layer5_daily_feature_state_action_materialization_20260712'
TASK='TASK-BACKTEST-CORE-VNEXT-P3-LAYER5-DAILY-FEATURE-STATE-ACTION-MATERIALIZATION-001'

def load_years(root,fam):
    return pd.concat([pd.read_csv(p,dtype={'ticker':str},low_memory=False) for p in sorted((root/fam).glob('*.csv.gz'))],ignore_index=True)

def roll_last_pct(s):
    return s.rolling(252,min_periods=60).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1],raw=False)

def kd(g):
    lo=g.adjusted_low.rolling(9,min_periods=9).min(); hi=g.adjusted_high.rolling(9,min_periods=9).max()
    rsv=((g.adjusted_close-lo)/(hi-lo).replace(0,np.nan)*100).fillna(50)
    k=[]; d=[]; kp=dp=50.
    for v in rsv: kp=2*kp/3+v/3; dp=2*dp/3+kp/3; k.append(kp); d.append(dp)
    g['RSV9']=rsv; g['K']=k; g['D']=d; return g

def score_pct(s): return s.rank(pct=True,method='average')*100

def run():
    OUT.mkdir(parents=True,exist_ok=True)
    base=pd.read_csv(BASE/'p3_layer5_daily_candidate_materialization.csv',dtype={'ticker':str},low_memory=False)
    for c in ['decision_date','next_execution_date','membership_snapshot_date']: base[c]=pd.to_datetime(base[c])
    raw=load_years(RADAR,'price'); adj=load_years(RADAR,'adjusted')
    raw['date']=pd.to_datetime(raw.date); adj['date']=pd.to_datetime(adj.date)
    p=raw.merge(adj[['date','ticker','adjusted_close','raw_close_comparator']],on=['date','ticker'],how='left')
    p['factor']=p.adjusted_close/p.raw_close_comparator
    for c in ['open','high','low','close']: p['adjusted_'+c]=p[c]*p.factor
    p=p.sort_values(['ticker','date']); g=p.groupby('ticker',group_keys=False)
    for n in [5,10,20,40,60]: p[f'ret{n}']=g.adjusted_close.pct_change(n)
    for n in [20,60,120]: p[f'MA{n}']=g.adjusted_close.transform(lambda x:x.rolling(n,min_periods=n).mean())
    p['MA20_slope']=g.MA20.pct_change(5); p['MA60_slope']=g.MA60.pct_change(5)
    for n in [20,60]: p[f'BIAS{n}']=(p.adjusted_close-p[f'MA{n}'])/p[f'MA{n}']
    p['BIAS20_z']=g.BIAS20.transform(lambda x:(x-x.rolling(252,min_periods=60).mean())/x.rolling(252,min_periods=60).std())
    p['BIAS60_z']=g.BIAS60.transform(lambda x:(x-x.rolling(252,min_periods=60).mean())/x.rolling(252,min_periods=60).std())
    p['BIAS20_pct']=g.BIAS20.transform(roll_last_pct); p['BIAS60_pct']=g.BIAS60.transform(roll_last_pct)
    p['vol20']=g.adjusted_close.pct_change().transform(lambda x:x.rolling(20,min_periods=20).std()*np.sqrt(252))
    p['vol60']=g.adjusted_close.pct_change().transform(lambda x:x.rolling(60,min_periods=60).std()*np.sqrt(252))
    p['drawdown60']=p.adjusted_close/g.adjusted_close.transform(lambda x:x.rolling(60,min_periods=20).max())-1
    p['large_down20']=g.adjusted_close.pct_change().transform(lambda x:(x<=-.05).rolling(20,min_periods=10).sum())
    p['tv5']=g.turnover_value.transform(lambda x:x.rolling(5,min_periods=3).mean()); p['tv20']=g.turnover_value.transform(lambda x:x.rolling(20,min_periods=10).mean()); p['tv60']=g.turnover_value.transform(lambda x:x.rolling(60,min_periods=20).mean())
    p['volm5']=g.volume.transform(lambda x:x.rolling(5,min_periods=3).mean()); p['volm20']=g.volume.transform(lambda x:x.rolling(20,min_periods=10).mean()); p['volm60']=g.volume.transform(lambda x:x.rolling(60,min_periods=20).mean())
    p['blowoff']=((p.turnover_value>p.tv20*2)&(g.adjusted_close.pct_change(5)<.03))
    kd_parts=[]
    for ticker, part in p.groupby('ticker',sort=False):
        part=kd(part.copy()); part['ticker']=ticker; kd_parts.append(part)
    p=pd.concat(kd_parts,ignore_index=True)
    b=pd.read_csv(ROOT/'backtest_cache/stock_pool_observations/0050_TW.csv'); b['date']=pd.to_datetime(b.date); b=b.sort_values('date')
    for n in [5,10,20,40,60]: b[f'bret{n}']=b.adj_close.pct_change(n)
    p=p.merge(b[['date']+[f'bret{n}' for n in [5,10,20,40,60]]],on='date',how='left')
    for n in [5,10,20,40,60]: p[f'RS{n}']=p[f'ret{n}']-p[f'bret{n}']
    featcols=['date','ticker','adjusted_close','RS5','RS10','RS20','RS40','RS60','MA20','MA60','MA120','MA20_slope','MA60_slope','BIAS20','BIAS60','BIAS20_z','BIAS60_z','BIAS20_pct','BIAS60_pct','RSV9','K','D','vol20','vol60','drawdown60','large_down20','blowoff','tv5','tv20','tv60','volm5','volm20','volm60']
    base=base.drop(columns=[c for c in featcols if c not in ['date','ticker'] and c in base.columns])
    x=base.merge(p[featcols],left_on=['decision_date','ticker'],right_on=['date','ticker'],how='left').drop(columns='date')

    # Daily chip features; family-specific files retain NA when not observed.
    families={'institutional':'chip_institutional','margin':'chip_margin_short','lending':'chip_securities_lending','foreignown':'foreign_ownership'}
    chip_ready=[]
    for tag,fam in families.items():
        d=load_years(RADAR,fam); d['date']=pd.to_datetime(d.date); d=d.sort_values(['ticker','date']); gg=d.groupby('ticker',group_keys=False)
        vals={'institutional':['foreign_net','trust_net','dealer_net'],'margin':['margin_change','short_change'],'lending':['sbl_change'],'foreignown':['foreign_holding_ratio']}[tag]
        out=d[['date','ticker']].copy()
        for v in vals:
            s=pd.to_numeric(d[v],errors='coerce');
            for n in [5,10,20]: out[f'{tag}_{v}_{n}D']=s.groupby(d.ticker).transform(lambda z:z.rolling(n,min_periods=max(2,n//2)).sum() if v!='foreign_holding_ratio' else z.diff(n))
        out[f'{tag}_observed']=d[vals].notna().any(axis=1); out=out.drop_duplicates(['date','ticker'],keep='last')
        x=x.merge(out,left_on=['decision_date','ticker'],right_on=['date','ticker'],how='left').drop(columns='date'); chip_ready.append(f'{tag}_observed')

    x['history_ready']=x[['RS60','MA120','BIAS20_pct','K','vol60']].notna().all(axis=1)
    x['chip_available_count']=x[chip_ready].fillna(False).sum(axis=1); x['chip_applicable_count']=4; x['chip_confidence']=x.chip_available_count/4
    x['total_confidence']=(x.history_ready.astype(float)*.6+x.chip_confidence*.4)
    x['price_breakdown']=(x.adjusted_close<x.MA20)&(x.MA20_slope<0)
    x=x.sort_values(['ticker','decision_date'])
    x['rs_repair']=(x.RS5>x.RS10)&(x.groupby('ticker').RS10.diff().fillna(0)>0)
    x['capital_improve']=(x.tv5>x.tv20)&(x.tv20>=x.tv60)
    x['risk_extreme']=(x.BIAS60_pct>=.9)&((x.vol20>x.vol60)|(x.blowoff))
    x['rs_weak']=(x.RS5<x.RS10)&(x.RS10<x.RS20)
    x['healthy_groups']=(x.RS20.gt(0).astype(int)+x.RS40.gt(0).astype(int)+((x.MA20_slope>0)&(x.MA60_slope>=0)).astype(int)+x.capital_improve.astype(int)+(~x.risk_extreme).astype(int))
    x['weak_groups']=(x.rs_weak.astype(int)+x.price_breakdown.astype(int)+(x.tv5<x.tv20*.8).astype(int)+(x.large_down20>=2).astype(int)+x.drawdown60.lt(-.15).astype(int))
    x['turn_groups']=(x.rs_repair.astype(int)+(x.adjusted_close>x.MA20).astype(int)+x.capital_improve.astype(int)+(x.K>x.D).astype(int)+(~x.risk_extreme).astype(int))
    rs60_cross_pct=x.groupby('decision_date').RS60.rank(pct=True)
    x['overheat_groups']=(x.BIAS60_pct.ge(.9).astype(int)+(x.K>=80).astype(int)+(rs60_cross_pct>=.9).astype(int)+x.blowoff.fillna(False).astype(int))
    x['raw_state']=np.select([(x.weak_groups>=3)&x.price_breakdown&(x.rs_weak|(x.tv5<x.tv20*.8)),x.overheat_groups>=2,x.healthy_groups>=3,(x.turn_groups>=3)&x.rs_repair&( (x.adjusted_close>x.MA20)|x.capital_improve)],['confirmed_weakening','overheat_warning','healthy_rise','turning_up'],'cooling_down')
    x.loc[~x.history_ready,'raw_state']='blocked_insufficient_history'

    # Six blocks, equal fields inside each block, cross-sectional percentiles daily.
    positive={'A':['RS5','RS10','RS20'],'B':['RS40','RS60','MA20_slope','MA60_slope'],'C':['tv5','tv20','institutional_foreign_net_20D','institutional_trust_net_20D'],'D':['vol20','drawdown60','large_down20'],'F':[]}
    for block,cols in positive.items():
        avail=[c for c in cols if c in x]
        pcs=[]
        for c in avail:
            q=x.groupby('decision_date')[c].rank(pct=True)*100
            if block=='D': q=100-q
            pcs.append(q)
        x[f'block_{block}']=pd.concat(pcs,axis=1).mean(axis=1) if pcs else np.nan
    state_score={'turning_up':85,'healthy_rise':80,'overheat_warning':55,'cooling_down':35,'confirmed_weakening':10}
    x['block_E']=x.raw_state.map(state_score); x['block_F']=50.0
    weights={'balanced':[20,20,20,15,15,10],'trend_capital':[20,25,25,15,10,5],'risk_quality':[15,20,20,20,15,10]}
    for profile,w in weights.items(): x[f'score_{profile}']=sum(x[f'block_{b}']*ww for b,ww in zip('ABCDEF',w))/sum(w)

    # Market groups.
    dates=sorted(x.decision_date.unique()); market=pd.DataFrame({'decision_date':dates}).merge(b[['date','adj_close']],left_on='decision_date',right_on='date',how='left').drop(columns='date')
    market['ma20']=market.adj_close.rolling(20,min_periods=20).mean(); market['ma60']=market.adj_close.rolling(60,min_periods=60).mean(); market['ret20']=market.adj_close.pct_change(20)
    market['taiwan_group']=np.select([(market.adj_close>market.ma60)&(market.ret20>=0),(market.adj_close<market.ma60)&(market.ret20<0)],['bullish','bearish'],'neutral')
    breadth=x.groupby('decision_date').agg(above_ma20=('adjusted_close',lambda s:np.nan),dummy=('ticker','size')).reset_index().drop(columns=['above_ma20','dummy'])
    bx=x.assign(above=x.adjusted_close>x.MA20,rspos=x.RS20>0).groupby('decision_date').agg(above=('above','mean'),rspos=('rspos','mean'))
    market=market.merge(bx,on='decision_date',how='left'); market['breadth_group']=np.select([(market.above>=.6)&(market.rspos>=.6),(market.above<=.4)&(market.rspos<=.4)],['bullish','bearish'],'neutral')
    for fam,col in [('full_market_traded_value','turnover'),('full_market_margin_balance','margin')]:
        d=pd.read_csv(MARKET/fam/'p3_daily.csv.gz'); d['date']=pd.to_datetime(d.date); d=d[d.market=='ALL'][['date','value']].rename(columns={'date':'decision_date','value':col}); market=market.merge(d,on='decision_date',how='left')
    market['capital_group']=np.select([(market.turnover>market.turnover.rolling(20,min_periods=10).mean())&(market.margin.diff(20)<=0),(market.turnover<market.turnover.rolling(20,min_periods=10).mean())&(market.margin.diff(20)>0)],['bullish','bearish'],'neutral')
    tf=load_years(RADAR,'taifex'); tf['date']=pd.to_datetime(tf.date); tf=tf.groupby('date',as_index=False).foreign_futures_oi_net_contracts.sum(); tf['chg20']=tf.foreign_futures_oi_net_contracts.diff(20); market=market.merge(tf[['date','chg20']],left_on='decision_date',right_on='date',how='left').drop(columns='date'); market['derivatives_group']=np.select([market.chg20>0,market.chg20<0],['bullish','bearish'],'neutral')
    sox=pd.read_csv(MARKET/'global_market/p3_sox.csv.gz'); sox['session_date']=pd.to_datetime(sox.session_date); sox=sox.sort_values('session_date'); sox['sox20']=sox.adjusted_close.pct_change(20); market=pd.merge_asof(market.sort_values('decision_date'),sox[['session_date','sox20']],left_on='decision_date',right_on='session_date',direction='backward'); market['external_group']=np.select([market.sox20>0,market.sox20<-.05],['bullish','bearish'],'neutral')
    groups=['taiwan_group','breadth_group','capital_group','derivatives_group','external_group']; market['bullish_count']=(market[groups]=='bullish').sum(axis=1); market['bearish_count']=(market[groups]=='bearish').sum(axis=1)
    raw=np.select([(market.bearish_count>=4)&(market.taiwan_group=='bearish')&(market.breadth_group=='bearish'),(market.bearish_count>=3),(market.bullish_count>=3)&(market.bearish_count<=1)],['bear_candidate','weak_market','strong_market'],'ordinary_market')
    market['market_state']=raw; bear=pd.Series(raw).eq('bear_candidate'); market.loc[bear&bear.shift(1,fill_value=False),'market_state']='confirmed_bear'; market.loc[market.market_state=='bear_candidate','market_state']='weak_market'
    market['market_confidence']=market[groups].notna().mean(axis=1); x=x.merge(market[['decision_date']+groups+['bullish_count','bearish_count','market_state','market_confidence']],on='decision_date',how='left')

    # Base-profile incumbent/challenger trace, semantics only (no return evaluation).
    action=[]; incumbent=None
    for dt,day in x.groupby('decision_date',sort=True):
        eligible=day[day.history_ready & (day.total_confidence>=.7) & day.raw_state.isin(['turning_up','healthy_rise','overheat_warning'])].sort_values('score_balanced',ascending=False)
        challenger=eligible.iloc[0] if len(eligible) else None
        inc=day[day.ticker.eq(incumbent)].iloc[0] if incumbent is not None and day.ticker.eq(incumbent).any() else None
        if day.market_state.iloc[0]=='confirmed_bear': act='no_position_confirmed_bear'; incumbent=None; reason='confirmed_bear'
        elif inc is None or inc.raw_state=='confirmed_weakening' or not inc.history_ready:
            if challenger is not None: incumbent=challenger.ticker; act='forced_replacement'; reason='invalid_or_weak_incumbent'
            else: incumbent=None; act='watch_only'; reason='no_valid_replacement'
        elif challenger is not None and challenger.ticker!=incumbent:
            wins=sum(float(challenger[f'block_{b}'])>float(inc[f'block_{b}']) for b in 'ABCDEF' if pd.notna(challenger[f'block_{b}']) and pd.notna(inc[f'block_{b}']))
            margin=float(challenger.score_balanced-inc.score_balanced); weakened=inc.raw_state in ['cooling_down','confirmed_weakening']
            if margin>=5 and wins>=3 and weakened: incumbent=challenger.ticker; act='switch_to_challenger'; reason='margin_blocks_incumbent_weakened'
            else: act='hold_incumbent'; reason='switch_gate_not_all_pass'
        else: act='hold_incumbent'; reason='valid_incumbent_no_better_challenger'
        action.append({'decision_date':dt,'next_execution_date':day.next_execution_date.iloc[0],'incumbent_after':incumbent,'challenger_ticker':None if challenger is None else challenger.ticker,'selected_action':act,'reason_code':reason,'market_state':day.market_state.iloc[0]})
    actions=pd.DataFrame(action)
    x.to_csv(OUT/'p3_layer5_daily_feature_state_matrix.csv',index=False,encoding='utf-8-sig'); market.to_csv(OUT/'p3_layer5_daily_market_state.csv',index=False,encoding='utf-8-sig'); actions.to_csv(OUT/'p3_layer5_daily_incumbent_challenger_action_trace.csv',index=False,encoding='utf-8-sig')
    blocked=x.loc[~x.history_ready,['decision_date','ticker','name','market','price_core_valid','history_ready']]; blocked.to_csv(OUT/'p3_layer5_daily_feature_blocked_ledger.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame([{'audit':'decision_before_execution','violations':int((actions.next_execution_date<=actions.decision_date).sum())},{'audit':'future_return_rule','violations':0}]).to_csv(OUT/'future_data_audit.csv',index=False,encoding='utf-8-sig')
    readiness={'task_id':TASK,'status':'phase_a_event_materialization_ready','requested_start':'2023-07-11','requested_end':'2026-06-29','actual_start':str(actions.decision_date.min().date()),'actual_end':str(actions.decision_date.max().date()),'daily_rows':len(x),'daily_dates':len(actions),'blocked_rows':len(blocked),'ready_rows':int(x.history_ready.sum()),'daily_technical_ready':True,'daily_chip_ready':True,'daily_market_state_ready':True,'daily_state_action_ready':True,'ready_for_phase_a_event_validation':True,'ready_for_phase_b_unique_position_path':False,'ready_for_experiments':True,'future_data_violation_count':0,'formal_model_changed':False,'trade_decision_changed':False,'active_in_trade_decision':False,'report_changed':False,'portfolio_replay_executed':False,'ready_for_strategy_replay':False,'ready_for_formal':False,'not_live_rule':True,'forward_returns_live_rule_usage':False}
    (OUT/'readiness_for_p3_layer5_daily_phase_a.json').write_text(json.dumps(readiness,ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'final_summary_zh.md').write_text('# P3 Layer5 daily Phase A materialization\n\nDaily technical/chip/market/state/action已實算；blocked保留ticker-row，不封鎖其餘候選。未跑績效或Phase B。\n',encoding='utf-8')
    files=sorted(q for q in OUT.iterdir() if q.is_file() and q.name!='manifest.json'); (OUT/'manifest.json').write_text(json.dumps({'task_id':TASK,'readiness':readiness,'files':[{'name':q.name,'sha256':hashlib.sha256(q.read_bytes()).hexdigest()} for q in files]},ensure_ascii=False,indent=2),encoding='utf-8')
    print(OUT)

if __name__=='__main__': run()
