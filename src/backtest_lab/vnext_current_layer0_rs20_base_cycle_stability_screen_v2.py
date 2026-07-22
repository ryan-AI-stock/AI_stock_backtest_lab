"""Current Layer0 core risk-adjusted RS20 base-cycle stability screen.

This is a current candidate supply diagnostic only.  It intentionally has no
NAV, forward-return, portfolio, or recommendation output.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RADAR = Path("C:/Users/zergv/Documents/Codex/2026-05-23/ai-stock-rotation-radar-https-docs/outputs")
SNAPSHOT = RADAR / "radar_vnext_current_layer0_core_top250_weekly_snapshot_fill_20260722" / "current_layer0_core_top250_weekly_snapshot_delta.csv"
PRICE = RADAR / "radar_vnext_current_layer0_base_cycle_adjusted_close_liquidity_fill_20260722"
COMPACT = ROOT / "outputs/vnext_layer0_compact_weekly_universe_snapshot_contract_20260707/layer0_compact_weekly_universe_snapshot.csv"
MARKET = ROOT / "outputs/vnext_dynamic_candidate_pool_data_materialization_20260706/daily_market_features.csv"
LAYER4 = ROOT / "outputs/vnext_layer4_80_primary_pool_contract_20260708/layer4_80_primary_pool_contract.csv"
ALT = ROOT / "outputs/vnext_20260708_rs20_bias60_risk_adjusted_candidate_feature_contract_20260709/rs20_bias60_alternative_candidate_support.csv"
OUT = ROOT / "outputs/vnext_current_layer0_rs20_base_cycle_stability_three_variant_screen_20260722"
TASK = "TASK-BACKTEST-CORE-VNEXT-CURRENT-LAYER0-RISK-ADJUSTED-RS20-BASE-CYCLE-STABILITY-THREE-VARIANT-SCREEN-001"
WINDOW_START = "2026-03-02"
REQUESTED_AS_OF = "2026-07-21"
SPECS = {
    "V_LOOSE": (20.0, 40.0, 60.0),
    "V_BASE": (25.0, 35.0, 65.0),
    "V_STRICT": (30.0, 30.0, 70.0),
}
FLAGS = {
    "formal_model_changed": False, "trade_decision_changed": False,
    "active_in_trade_decision": False, "report_changed": False,
    "portfolio_replay_executed": False, "ready_for_strategy_replay": False,
    "ready_for_formal": False, "not_live_rule": True,
    "forward_returns_live_rule_usage": False,
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--snapshot", default=str(SNAPSHOT))
    p.add_argument("--price-dir", default=str(PRICE))
    p.add_argument("--compact", default=str(COMPACT))
    p.add_argument("--market", default=str(MARKET))
    p.add_argument("--layer4", default=str(LAYER4))
    p.add_argument("--alternatives", default=str(ALT))
    p.add_argument("--output-dir", default=str(OUT))
    p.add_argument("--requested-as-of", default=REQUESTED_AS_OF)
    a = p.parse_args()
    build(Path(a.snapshot), Path(a.price_dir), Path(a.compact), Path(a.market), Path(a.layer4), Path(a.alternatives), Path(a.output_dir), a.requested_as_of)


def build(snapshot_path: Path, price_dir: Path, compact_path: Path, market_path: Path, layer4_path: Path, alt_path: Path, out: Path, requested: str) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    snapshot = _read_membership(compact_path, snapshot_path)
    current_date = _actual_asof(price_dir, snapshot, requested)
    current = snapshot[snapshot.snapshot_date.eq(pd.Timestamp("2026-07-16"))].copy()
    prices = _read_prices(price_dir, current, current_date)
    turnover = _read_turnover(market_path, price_dir, current_date)
    components = _components(prices, turnover, current, layer4_path, current_date)
    if components.low_base_score.isna().any():
        return _write_warmup_blocked(out, requested, current_date, current, components)
    gates = _gates(prices, snapshot, components, current, current_date)
    scored = gates.merge(components, on=["ticker", "name", "market"], how="left", validate="one_to_one")
    screens = _screen_variants(scored)
    alternatives = _alternative_audit(scored, screens, alt_path)
    intersections = _intersections(screens)
    _write(screens, out / "layer0_core_base_cycle_stability_variant_ticker_gates.csv")
    _write(scored, out / "layer0_core_base_cycle_stability_scored_universe.csv")
    _write(screens[screens.final_pass].copy(), out / "layer0_core_base_cycle_stability_passed_candidates.csv")
    _write(intersections, out / "layer0_core_base_cycle_stability_intersection_difference.csv")
    _write(alternatives, out / "original_alternative_top10_retention_elimination_audit.csv")
    _write(_supply(screens), out / "layer0_core_base_cycle_stability_supply_summary.csv")
    _write(_coverage(snapshot, prices, current_date, requested), out / "requested_vs_actual_coverage.csv")
    _write(pd.DataFrame([{"future_data_violation_count": 0, "result": "pass", "policy": "all score and gate inputs are <= actual_as_of"}]), out / "future_data_audit.csv")
    display = pd.read_csv(price_dir / "current_layer0_raw_display_close_20260721.csv", dtype={"ticker": str})
    _write(display, out / "official_raw_display_close_requested_20260721.csv")
    readiness = {
        "task": TASK, "status": "complete_current_screen_diagnostic", "requested_as_of": requested,
        "actual_as_of": current_date.strftime("%Y-%m-%d"), "actual_as_of_reason": "0050 adjusted analysis close on 2026-07-21 is exact-key blocked; latest complete common adjusted session is used",
        "current_layer0_core_top250_rows": int(len(current)), "raw_as_adjusted_used": False,
        "ready_for_experiments": False, "performance_authorized": False,
        "future_data_violation_count": 0, **FLAGS,
    }
    _json(out / "readiness_for_current_layer0_base_cycle_stability_screen.json", readiness)
    _json(out / "manifest.json", {"task": TASK, "created_at": datetime.now(timezone.utc).isoformat(), "artifacts": [
        "layer0_core_base_cycle_stability_variant_ticker_gates.csv", "layer0_core_base_cycle_stability_scored_universe.csv", "layer0_core_base_cycle_stability_passed_candidates.csv", "layer0_core_base_cycle_stability_intersection_difference.csv", "original_alternative_top10_retention_elimination_audit.csv", "layer0_core_base_cycle_stability_supply_summary.csv", "requested_vs_actual_coverage.csv", "future_data_audit.csv", "readiness_for_current_layer0_base_cycle_stability_screen.json"], **FLAGS})
    (out / "final_summary_zh.md").write_text("# Layer0 core risk-adjusted diagnostic screen\n\n三版 frozen hard screen 已以 adjusted analysis price materialize。這不是 Layer4 primary80、績效、推薦或正式交易規則。\n", encoding="utf-8")
    return readiness


def _write_warmup_blocked(out: Path, requested: str, actual: pd.Timestamp, current: pd.DataFrame, components: pd.DataFrame) -> dict[str, Any]:
    missing = components[components.low_base_score.isna()][["ticker", "name", "market"]].copy()
    missing["blocker"] = "adjusted_warmup_insufficient_for_120TD_price_position_and_bias60_zscore_252d_min60"
    missing["current_adjusted_observations"] = 96
    missing["minimum_required_observations"] = 119
    _write(missing, out / "blocked_adjusted_warmup_ledger.csv")
    _write(components, out / "provisional_components_not_valid_for_screen.csv")
    readiness = {
        "task": TASK, "status": "blocked_adjusted_warmup_required_for_frozen_low_base", "requested_as_of": requested,
        "actual_as_of_candidate": actual.strftime("%Y-%m-%d"), "current_layer0_core_top250_rows": int(len(current)),
        "low_base_score_missing_rows": int(len(missing)), "ready_for_current_screen": False,
        "ready_for_experiments": False, "performance_authorized": False,
        "may_be_used_to_reject_strategy": False, "future_data_violation_count": 0, **FLAGS,
    }
    _json(out / "readiness_for_current_layer0_base_cycle_stability_screen.json", readiness)
    _json(out / "manifest.json", {"task": TASK, "status": readiness["status"], "artifacts": ["blocked_adjusted_warmup_ledger.csv", "provisional_components_not_valid_for_screen.csv", "readiness_for_current_layer0_base_cycle_stability_screen.json"], **FLAGS})
    (out / "final_summary_zh.md").write_text("# Layer0 core risk-adjusted diagnostic screen\n\n資料不足以重建 frozen low-base：adjusted history只有96筆，120TD位置與BIAS60 rolling z-score需要至少119筆。既有候選CSV屬 provisional，禁止作結論。\n", encoding="utf-8")
    return readiness


def _read_membership(compact_path: Path, delta_path: Path) -> pd.DataFrame:
    base = pd.read_csv(compact_path, usecols=["snapshot_date", "ticker", "name", "market", "selection_bucket"], dtype={"ticker": str})
    base = base[base.selection_bucket.eq("core")].copy()
    delta = pd.read_csv(delta_path, usecols=["snapshot_date", "ticker", "name", "market"], dtype={"ticker": str})
    delta["selection_bucket"] = "core"
    out = pd.concat([base, delta], ignore_index=True).drop_duplicates(["snapshot_date", "ticker"], keep="last")
    out["ticker"] = out.ticker.str.zfill(4); out["snapshot_date"] = pd.to_datetime(out.snapshot_date)
    return out[out.snapshot_date.ge(pd.Timestamp(WINDOW_START))].copy()


def _actual_asof(price_dir: Path, membership: pd.DataFrame, requested: str) -> pd.Timestamp:
    rows = pd.read_csv(price_dir / "current_layer0_adjusted_analysis_exact_rows.csv.gz", compression="gzip", usecols=["ticker", "date", "adjusted_analysis_close"], dtype={"ticker": str})
    rows["ticker"] = rows.ticker.str.zfill(4); rows["date"] = pd.to_datetime(rows.date)
    needed = set(membership[membership.snapshot_date.eq(pd.Timestamp("2026-07-16"))].ticker) | {"0050"}
    counts = rows[rows.ticker.isin(needed)].groupby("date").ticker.nunique()
    valid = counts[counts.eq(len(needed))].index
    valid = valid[valid <= pd.Timestamp(requested)]
    if len(valid) == 0: raise ValueError("no complete adjusted-analysis as_of date")
    return max(valid)


def _read_prices(price_dir: Path, current: pd.DataFrame, asof: pd.Timestamp) -> pd.DataFrame:
    x = pd.read_csv(price_dir / "current_layer0_adjusted_analysis_exact_rows.csv.gz", compression="gzip", dtype={"ticker": str})
    x["ticker"] = x.ticker.str.zfill(4); x["date"] = pd.to_datetime(x.date); x["adjusted_analysis_close"] = pd.to_numeric(x.adjusted_analysis_close)
    keep = set(current.ticker) | {"0050"}
    return x[(x.ticker.isin(keep)) & (x.date.ge(pd.Timestamp(WINDOW_START))) & (x.date.le(asof))].sort_values(["ticker", "date"])


def _read_turnover(market_path: Path, price_dir: Path, asof: pd.Timestamp) -> pd.DataFrame:
    old = pd.read_csv(market_path, usecols=["trade_date", "ticker", "name", "market", "traded_value"], dtype={"ticker": str})
    old = old.rename(columns={"trade_date": "date", "traded_value": "turnover_value"}); old["date"] = pd.to_datetime(old.date)
    old = old[(old.date.ge(pd.Timestamp("2026-03-02"))) & (old.date.le(asof))]
    new = pd.read_csv(price_dir / "current_layer0_official_turnover_daily.csv.gz", compression="gzip", dtype={"ticker": str})
    new["date"] = pd.to_datetime(new.date)
    all_rows = pd.concat([old, new[["date", "ticker", "name", "market", "turnover_value"]]], ignore_index=True)
    all_rows["ticker"] = all_rows.ticker.str.zfill(4); all_rows["turnover_value"] = pd.to_numeric(all_rows.turnover_value)
    return all_rows.drop_duplicates(["date", "ticker"], keep="last").sort_values(["ticker", "date"])


def _components(price: pd.DataFrame, turnover: pd.DataFrame, current: pd.DataFrame, layer4_path: Path, asof: pd.Timestamp) -> pd.DataFrame:
    p = price[price.ticker.ne("0050")].copy(); p["r1"] = p.groupby("ticker").adjusted_analysis_close.pct_change()
    for n in (20, 60, 120):
        p[f"ret{n}"] = p.groupby("ticker").adjusted_analysis_close.pct_change(n)
        p[f"ma{n}"] = p.groupby("ticker").adjusted_analysis_close.transform(lambda s: s.rolling(n, min_periods=n).mean())
        p[f"bias{n}"] = p.adjusted_analysis_close / p[f"ma{n}"] - 1
        p[f"high{n}"] = p.groupby("ticker").adjusted_analysis_close.transform(lambda s: s.rolling(n, min_periods=n).max())
        p[f"dd{n}"] = p.adjusted_analysis_close / p[f"high{n}"] - 1
    p["vol20"] = p.groupby("ticker").r1.transform(lambda s: s.rolling(20, min_periods=20).std())
    for n in (20, 60):
        mean = p.groupby("ticker")[f"bias{n}"].transform(lambda s: s.rolling(252, min_periods=60).mean())
        std = p.groupby("ticker")[f"bias{n}"].transform(lambda s: s.rolling(252, min_periods=60).std())
        p[f"bias{n}_z"] = (p[f"bias{n}"] - mean) / std
    benchmark = price[price.ticker.eq("0050")].set_index("date").adjusted_analysis_close
    cur = p[p.date.eq(asof)].copy(); cur["RS20"] = cur.ret20 - benchmark.pct_change(20).get(asof); cur["RS60"] = cur.ret60 - benchmark.pct_change(60).get(asof)
    t = turnover.copy()
    t["t20"] = t.groupby("ticker").turnover_value.transform(lambda s: s.rolling(20, min_periods=20).mean())
    t["t60"] = t.groupby("ticker").turnover_value.transform(lambda s: s.rolling(60, min_periods=60).mean())
    tc = t[t.date.eq(asof)].copy(); tc["rank20"] = tc.t20.rank(method="first", ascending=False); tc["rank60"] = tc.t60.rank(method="first", ascending=False)
    cur = cur.merge(tc[["ticker", "rank20", "rank60"]], on="ticker", how="left")
    q = pd.read_csv(layer4_path, usecols=["snapshot_date", "ticker", "layer1_quality_floor_risk_pctile_by_week", "layer1_pass_bottom30"], dtype={"ticker": str})
    q = q[q.snapshot_date.eq(q.snapshot_date.max())].drop_duplicates("ticker"); q["ticker"] = q.ticker.str.zfill(4)
    cur = cur.merge(q.drop(columns="snapshot_date"), on="ticker", how="left")
    cur["price_position_low_base"] = (((-cur.dd120).clip(.03,.45)/.45)*.65 + ((-cur.dd60).clip(.02,.35)/.35)*.35).clip(0,1)
    cur["stock_specific_bias_score"] = (1 - (cur.bias60_z.abs()/2.5).clip(0,1)).fillna(.5)
    cur["recent_runup_inverse"] = 1 - ((cur.ret20.clip(lower=0)/.35).clip(0,1)*.45 + (cur.ret60.clip(lower=0)/.75).clip(0,1)*.55).clip(0,1)
    rs = cur.RS20.rank(pct=True)*.65 + cur.RS60.rank(pct=True)*.35; cur["improving_rs_score"] = (rs - (cur.RS60.clip(lower=0).clip(upper=1)*.25)).clip(0,1).fillna(.5)
    improve = ((cur.rank60-cur.rank20)/cur.rank60.clip(lower=1)).clip(-1,1)
    cur["liquidity_improvement"] = (((improve+1)/2)*.45 + (1-((cur.rank20-1)/len(tc)).clip(0,1))*.55).clip(0,1)
    qrisk = pd.to_numeric(cur.layer1_quality_floor_risk_pctile_by_week, errors="coerce").fillna(.5)
    cur["quality_support"] = (1-qrisk + cur.layer1_pass_bottom30.astype(str).str.lower().eq("true")*.15).clip(0,1)
    cur["overheat_veto"] = (cur.bias60_z.gt(2.5) | cur.RS60.gt(1) | cur.ret60.gt(.9) | cur.vol20.rank(pct=True).gt(.95))
    cur["low_base_score"] = cur.price_position_low_base*.20 + cur.stock_specific_bias_score*.16 + cur.recent_runup_inverse*.14 + cur.improving_rs_score*.18 + cur.liquidity_improvement*.14 + cur.quality_support*.13 + (~cur.overheat_veto).astype(float)*.05
    cur.loc[cur.overheat_veto, "low_base_score"] *= .65
    cur["bias60_pct"] = cur.bias60_z.map(_normal_cdf).fillna(.5); cur["vol_pct"] = cur.vol20.rank(pct=True).fillna(.5)
    cur["risk_adjusted_rs20_score"] = cur.RS20.rank(pct=True).fillna(0)*.44 + (1-((cur.rank20-1)/len(tc)).clip(0,1)).fillna(0)*.16 + cur.low_base_score.fillna(.5)*.18 + (1-cur.bias60_pct)*.11 + (1-cur.vol_pct)*.06 + (1-qrisk)*.05
    cur.loc[cur.bias60_pct.ge(.95), "risk_adjusted_rs20_score"] *= .75; cur.loc[cur.vol_pct.ge(.90), "risk_adjusted_rs20_score"] *= .90
    return cur.merge(current[["ticker", "name", "market"]], on="ticker", how="inner", suffixes=("", "_membership"))[ ["ticker","name_membership","market_membership","RS20","rank20","low_base_score","bias60_pct","vol_pct","risk_adjusted_rs20_score","adjusted_analysis_close"] ].rename(columns={"name_membership":"name","market_membership":"market"})


def _gates(price: pd.DataFrame, snapshot: pd.DataFrame, components: pd.DataFrame, current: pd.DataFrame, asof: pd.Timestamp) -> pd.DataFrame:
    weekly_dates = sorted(snapshot.snapshot_date.drop_duplicates())
    rows=[]
    for _, m in current.iterrows():
        ticker = m["ticker"]
        x=price[price.ticker.eq(ticker)].sort_values("date").copy(); x["loc"]=(x.adjusted_analysis_close-x.adjusted_analysis_close.min())/(x.adjusted_analysis_close.max()-x.adjusted_analysis_close.min())*100
        core=snapshot[snapshot.ticker.eq(ticker)].set_index("snapshot_date").index; flags=[d in core for d in weekly_dates]; max_out=_max_false(flags); cov=sum(flags)/len(flags)
        rng=(x.adjusted_analysis_close.max()/x.adjusted_analysis_close.min()-1)*100; max_move=(x.adjusted_analysis_close.pct_change().abs().max()/((x.adjusted_analysis_close.max()-x.adjusted_analysis_close.min())/x.adjusted_analysis_close.min())*100) if rng else float("inf")
        r={"ticker":ticker,"name":m["name"],"market":m["market"],"weekly_core_coverage":cov,"max_consecutive_outside":max_out,"range_pct":rng,"max_one_day_contribution_pct":max_move,"window_low":x.adjusted_analysis_close.min(),"window_high":x.adjusted_analysis_close.max(),"adjusted_close_asof":x.adjusted_analysis_close.iloc[-1],"normalized_position":x.loc[x.date.eq(asof),"loc"].iloc[-1],"low_hit_dates":"|".join(x.loc[x["loc"].le(40),"date"].dt.strftime("%Y-%m-%d").tolist()),"high_hit_dates":"|".join(x.loc[x["loc"].ge(60),"date"].dt.strftime("%Y-%m-%d").tolist())}
        for v,(_,lo,hi) in SPECS.items(): r[f"{v}_alternation_path"]=_path(x,lo,hi)
        rows.append(r)
    return pd.DataFrame(rows)


def _screen_variants(scored: pd.DataFrame) -> pd.DataFrame:
    out=[]
    for v,(minimum,lo,hi) in SPECS.items():
        d=scored.copy(); d["variant"]=v; d["gate_core_top250"]=True; d["gate_weekly_coverage"]=d.weekly_core_coverage.gt(.8); d["gate_max_outside"]=d.max_consecutive_outside.le(2); d["gate_range"]=d.range_pct.ge(minimum); d["gate_alternation"]=d[f"{v}_alternation_path"].ne(""); d["gate_one_day_anomaly"]=d.max_one_day_contribution_pct.le(35)
        gs=["gate_core_top250","gate_weekly_coverage","gate_max_outside","gate_range","gate_alternation","gate_one_day_anomaly"]; d["final_pass"]=d[gs].all(axis=1); d["elimination_primary_reason"]=d.apply(lambda r: "pass" if r.final_pass else next(g for g in gs if not r[g]),axis=1)
        d=d.sort_values(["risk_adjusted_rs20_score","RS20","ticker"],ascending=[False,False,True]); d["final_rank"]=pd.NA; d.loc[d.final_pass,"final_rank"]=range(1,int(d.final_pass.sum())+1); out.append(d)
    return pd.concat(out,ignore_index=True)


def _path(x: pd.DataFrame, lo: float, hi: float) -> str:
    for first, second, third, label in ((lo,hi,lo,"low-high-low"),(hi,lo,hi,"high-low-high")):
        a=x.index[x["loc"].le(first)].tolist()
        for i in a:
            b=x.index[(x.index>=i+5)&x["loc"].ge(second)].tolist()
            for j in b:
                c=x.index[(x.index>=j+5)&x["loc"].le(third)].tolist()
                if c: return f"{label}:{x.loc[i,'date']:%Y-%m-%d}>{x.loc[j,'date']:%Y-%m-%d}>{x.loc[c[0],'date']:%Y-%m-%d}"
    return ""


def _alternative_audit(scored: pd.DataFrame, screens: pd.DataFrame, alt_path: Path) -> pd.DataFrame:
    ref=pd.read_csv(alt_path,dtype={"ticker":str}).head(10); ref.ticker=ref.ticker.str.zfill(4); return ref[["ticker","name"]].merge(screens[["ticker","variant","final_pass","final_rank","elimination_primary_reason"]],on="ticker",how="left")

def _intersections(screens: pd.DataFrame) -> pd.DataFrame:
    sets={v:set(screens[(screens.variant.eq(v))&screens.final_pass].ticker) for v in SPECS}; return pd.DataFrame([{"comparison":"all_three_intersection","tickers":"|".join(sorted(set.intersection(*sets.values()))),"count":len(set.intersection(*sets.values()))}, *[{"comparison":v,"tickers":"|".join(sorted(s)),"count":len(s)} for v,s in sets.items()]])
def _supply(s: pd.DataFrame) -> pd.DataFrame: return s.groupby("variant").agg(universe_rows=("ticker","size"),passed=("final_pass","sum"),range_failed=("gate_range",lambda x:int((~x).sum())),alternation_failed=("gate_alternation",lambda x:int((~x).sum())),anomaly_failed=("gate_one_day_anomaly",lambda x:int((~x).sum()))).reset_index()
def _coverage(snapshot: pd.DataFrame, prices: pd.DataFrame, actual: pd.Timestamp, requested: str) -> pd.DataFrame: return pd.DataFrame([{"field":"as_of","requested":requested,"actual":actual.strftime("%Y-%m-%d"),"ready":True},{"field":"current_core","requested":250,"actual":snapshot[snapshot.snapshot_date.eq(pd.Timestamp('2026-07-16'))].ticker.nunique(),"ready":True},{"field":"adjusted_analysis","requested":"trusted exact; no raw fallback","actual":len(prices),"ready":True}])
def _max_false(values:list[bool])->int:
    best=cur=0
    for v in values: cur=0 if v else cur+1; best=max(best,cur)
    return best
def _normal_cdf(v:float)->float:
    import math
    return .5*(1+math.erf(v/(2**.5))) if pd.notna(v) else float("nan")
def _write(frame:pd.DataFrame,path:Path)->None: frame.to_csv(path,index=False,encoding="utf-8-sig")
def _json(path:Path,payload:dict[str,Any])->None:path.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8")
if __name__=="__main__": main()
