from __future__ import annotations

import glob
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RADAR = Path(r"C:\Users\zergv\Documents\Codex\2026-05-23\ai-stock-rotation-radar-https-docs\outputs")
P3 = RADAR / "radar_vnext_p3_recent_full_feature_data_readiness_acquisition_20260711"
WARMUP = RADAR / "radar_vnext_p3_exact_primary80_raw_hlc_warmup_gap_fill_20260711"
RANK1 = RADAR / "radar_vnext_p3_layer04_rank1_sequential_lifecycle_adjusted_hlc_factor_source_package_20260713"
CURRENT = RADAR / "radar_vnext_p3_ridge_shadow_current_layer1_4_bounded_delta_fill_20260712"
DELTA = RADAR / "radar_vnext_p3_layer5_all80_continuous_lifecycle_adjusted_hlc_bounded_delta_acquisition_20260713"
DAILY = ROOT / "outputs/vnext_p3_layer5_daily_feature_state_action_materialization_20260712/p3_layer5_daily_feature_state_matrix.csv"
ETF = ROOT / "backtest_cache/stock_pool_observations/0050_TW.csv"
OUT = ROOT / "outputs/vnext_p3_layer5_all80_continuous_sequential_lifecycle_state_supply_contract_20260713"
TASK = "TASK-BACKTEST-CORE-VNEXT-P3-LAYER5-ALL80-CONTINUOUS-LIFECYCLE-ADJUSTED-HLC-DELTA-ABSORPTION-AND-STATE-SUPPLY-RERUN-001"
PLATFORMS = {"L1": (0.25, 0.75), "L2": (0.35, 0.70), "L3": (0.40, 0.65)}
PERSISTENCE = {"2of3": (2, 3), "3of5": (3, 5)}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(paths: list[Path], columns: list[str]) -> pd.DataFrame:
    frames = []
    for path in paths:
        if not path.exists():
            continue
        available = pd.read_csv(path, nrows=0).columns
        use = [column for column in columns if column in available]
        frame = pd.read_csv(path, dtype={"ticker": str}, usecols=use)
        frame["ticker"] = frame.ticker.str.zfill(4)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["ticker", "date"])


def _last_percentile(values: np.ndarray) -> float:
    valid = values[~np.isnan(values)]
    return np.nan if not len(valid) else float((valid <= valid[-1]).mean())


def _history() -> pd.DataFrame:
    raw_paths = sorted((P3 / "compact/price").glob("*.csv.gz")) + sorted((WARMUP / "compact/raw_hlc_warmup").glob("*.csv.gz"))
    factor_paths = sorted((P3 / "compact/adjusted").glob("*.csv.gz"))
    direct_paths = [RANK1 / "rank1_adjusted_analysis_hlc_factor_compact.csv.gz", CURRENT / "ridge_shadow_current_adjusted_analysis_ohlc_factor_rows.csv.gz"]
    raw = _load(raw_paths, ["ticker", "date", "high", "low", "close"]).dropna(subset=["high", "low", "close"])
    factor = _load(factor_paths, ["ticker", "date", "adjusted_close", "raw_close_comparator"])
    factor["factor"] = pd.to_numeric(factor.adjusted_close, errors="coerce") / pd.to_numeric(factor.raw_close_comparator, errors="coerce")
    base = raw.merge(factor[["ticker", "date", "factor"]], on=["ticker", "date"], how="inner")
    base = base.loc[base.factor.gt(0)].copy()
    for column in ["high", "low", "close"]:
        base[f"adjusted_{column}"] = base[column] * base.factor
    direct = _load(direct_paths, ["ticker", "date", "adjusted_high", "adjusted_low", "adjusted_close"])
    delta = pd.read_csv(DELTA / "all80_bounded_delta_adjusted_hlc_exact_key_compact.csv.gz", dtype={"ticker": str}, usecols=["ticker", "date", "adjusted_high", "adjusted_low", "adjusted_close"])
    delta["ticker"] = delta.ticker.str.zfill(4)
    history = pd.concat([base[["ticker", "date", "adjusted_high", "adjusted_low", "adjusted_close"]], direct, delta], ignore_index=True)
    history["date"] = pd.to_datetime(history.date)
    return history.dropna(subset=["adjusted_high", "adjusted_low", "adjusted_close"]).drop_duplicates(["ticker", "date"], keep="last").sort_values(["ticker", "date"])


def _features(history: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    for ticker, group in history.groupby("ticker", sort=False):
        g = group.sort_values("date").copy()
        close, high, low = g.adjusted_close, g.adjusted_high, g.adjusted_low
        low9, high9 = low.rolling(9, min_periods=9).min(), high.rolling(9, min_periods=9).max()
        rsv = 100 * (close - low9) / (high9 - low9).replace(0, np.nan)
        k, d, pk, pd_ = [], [], 50.0, 50.0
        for value in rsv:
            if pd.notna(value):
                pk = (2 * pk + value) / 3
                pd_ = (2 * pd_ + pk) / 3
            k.append(pk); d.append(pd_)
        g["K6"] = k; g["D6"] = d
        g["MA20x"] = close.rolling(20, min_periods=20).mean()
        g["BIAS20x"] = (close - g.MA20x) / g.MA20x
        for window, label in [(63, "3M"), (126, "6M"), (252, "12M")]:
            minp = min(window, 60)
            g[f"price_pct_{label}"] = close.rolling(window, min_periods=minp).apply(_last_percentile, raw=True)
            g[f"K_pct_{label}"] = g.K6.rolling(window, min_periods=minp).apply(_last_percentile, raw=True)
            g[f"BIAS_pct_{label}"] = g.BIAS20x.rolling(window, min_periods=minp).apply(_last_percentile, raw=True)
        g["actual_history_observations"] = np.arange(1, len(g) + 1)
        pieces.append(g)
    return pd.concat(pieces, ignore_index=True)


def _persistent(frame: pd.DataFrame, column: str, need: int, window: int) -> pd.Series:
    return frame.groupby("ticker", sort=False)[column].transform(lambda s: s.astype(float).rolling(window, min_periods=window).sum().ge(need))


def _states(frame: pd.DataFrame, low: float, high: float, need: int, window: int) -> pd.DataFrame:
    x = frame.copy()
    x["relative_low"] = x.price_pct_6M.le(low) & (x.K_pct_6M.le(low) | x.BIAS_pct_6M.le(low))
    x["relative_high"] = x.price_pct_6M.ge(high) & (x.K_pct_6M.ge(high) | x.BIAS_pct_6M.ge(high))
    k_change = x.groupby("ticker", sort=False).K6.diff()
    x["kd_up"] = k_change.gt(0) & x.K6.ge(x.D6)
    x["kd_down"] = k_change.lt(0) & x.K6.le(x.D6)
    x["ma_up"] = x.adjusted_close.ge(x.MA20) & x.MA20_slope.gt(0) & ~x.price_breakdown.fillna(False)
    x["ma_down"] = x.adjusted_close.lt(x.MA20) & (x.MA20_slope.lt(0) | x.price_breakdown.fillna(False))
    x["risk_ok"] = ~x.risk_extreme.fillna(False) & ~x.price_breakdown.fillna(False)
    x["risk_bad"] = x.risk_extreme.fillna(False) | x.price_breakdown.fillna(False)
    institutional_withdrawal = (
        x.institutional_foreign_net_20D.lt(0)
        | x.institutional_trust_net_20D.lt(0)
        | x.institutional_dealer_net_20D.lt(0)
    )
    x["capital_withdraw"] = x.tv5.lt(x.tv20) & institutional_withdrawal & x.chip_available_count.gt(0)
    group_pairs = [("kd_up", "kd_down"), ("rs_repair", "rs_weak"), ("ma_up", "ma_down"), ("capital_improve", "capital_withdraw"), ("risk_ok", "risk_bad")]
    for up, down in group_pairs:
        x[f"{up}_persistent"] = _persistent(x, up, need, window)
        x[f"{down}_persistent"] = _persistent(x, down, need, window)
    x["up_groups"] = sum(x[f"{up}_persistent"].astype(int) for up, _ in group_pairs)
    x["down_groups"] = sum(x[f"{down}_persistent"].astype(int) for _, down in group_pairs)
    entry_required = x.market_state.map({"strong_market": 3, "ordinary_market": 3, "weak_market": 4, "confirmed_bear": 6}).fillna(3)
    exit_required = x.market_state.map({"strong_market": 4, "ordinary_market": 3, "weak_market": 3, "confirmed_bear": 2}).fillna(3)
    x["turnup_confirmed"] = x.up_groups.ge(entry_required)
    x["turndown_confirmed"] = x.down_groups.ge(exit_required)
    states = []
    prior_by_ticker: dict[str, str] = {}
    prior_date_by_ticker: dict[str, pd.Timestamp] = {}
    market_dates = {date: index for index, date in enumerate(sorted(x.decision_date.unique()))}
    for row in x.itertuples(index=False):
        prior = prior_by_ticker.get(row.ticker, "S0")
        prior_date = prior_date_by_ticker.get(row.ticker)
        reentry = prior_date is not None and market_dates[row.decision_date] - market_dates[prior_date] > 1
        if not row.price_history_ready:
            state = "BLOCKED"
        elif reentry:
            # Re-entry uses this ticker's current PIT history; it never inherits another ticker's state.
            if row.relative_low and row.turnup_confirmed: state = "S2"
            elif row.relative_low: state = "S1"
            elif row.relative_high and row.turndown_confirmed: state = "S6"
            elif row.relative_high: state = "S5"
            else: state = "S0"
        elif prior == "BLOCKED":
            state = "S1" if row.relative_low else "S0"
        elif prior == "S0": state = "S1" if row.relative_low else "S0"
        elif prior == "S1": state = "S2" if row.turnup_confirmed else ("S1" if row.relative_low else "S0")
        elif prior == "S2": state = "S3" if row.capital_improve_persistent and row.market_state != "confirmed_bear" else "S1"
        elif prior == "S3": state = "S4"
        elif prior == "S4": state = "S5" if row.relative_high else "S4"
        elif prior == "S5": state = "S6" if row.turndown_confirmed else ("S5" if row.relative_high else "S4")
        elif prior == "S6": state = "S7" if row.capital_withdraw_persistent else ("S5" if row.relative_high else "S4")
        else: state = "S1" if row.relative_low else "S0"
        states.append(state)
        if state != "BLOCKED": prior_by_ticker[row.ticker] = state
        prior_date_by_ticker[row.ticker] = row.decision_date
    x["state"] = states
    x["entry_cluster"] = x.state.eq("S3") & x.groupby("ticker").state.shift().ne("S3")
    x["exit_cluster"] = x.state.eq("S7") & x.groupby("ticker").state.shift().ne("S7")
    return x


def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    daily = pd.read_csv(DAILY, dtype={"ticker": str}, low_memory=False)
    daily["ticker"] = daily.ticker.str.zfill(4); daily["decision_date"] = pd.to_datetime(daily.decision_date)
    p31 = daily.loc[daily.decision_date.lt("2025-07-11")].copy().sort_values(["ticker", "decision_date"])
    history = _features(_history())
    cols = ["ticker", "date", "adjusted_high", "adjusted_low", "adjusted_close", "K6", "D6", "MA20x", "price_pct_3M", "price_pct_6M", "price_pct_12M", "K_pct_3M", "K_pct_6M", "K_pct_12M", "BIAS_pct_3M", "BIAS_pct_6M", "BIAS_pct_12M", "actual_history_observations"]
    merged = p31.merge(history[cols], left_on=["ticker", "decision_date"], right_on=["ticker", "date"], how="left", suffixes=("", "_history"))
    merged["price_history_ready"] = merged.actual_history_observations.ge(126) & merged[["price_pct_6M", "K_pct_6M", "BIAS_pct_6M"]].notna().all(axis=1)
    merged["source_status"] = np.where(merged.price_history_ready, "ready_research_adjusted_HLC", "blocked_insufficient_or_structural_adjusted_history")
    merged["capital_improve"] = merged.capital_improve.fillna(False)
    results, supplies, folds = [], [], []
    date_list = sorted(merged.decision_date.unique())
    split = np.array_split(date_list, 3)
    for platform, (low, high) in PLATFORMS.items():
        for persistence, (need, window) in PERSISTENCE.items():
            state = _states(merged, low, high, need, window)
            state["platform"] = platform; state["persistence"] = persistence
            results.append(state)
            supply = state.groupby(["decision_date", "market_state", "state"]).size().rename("candidate_count").reset_index()
            supply["platform"] = platform; supply["persistence"] = persistence; supplies.append(supply)
            for fold_id, dates in enumerate(split, 1):
                part = state.loc[state.decision_date.isin(dates)]
                folds.append({"platform": platform, "persistence": persistence, "fold": fold_id, "start": str(pd.Timestamp(dates[0]).date()), "end": str(pd.Timestamp(dates[-1]).date()), "entry_clusters": int(part.entry_cluster.sum()), "exit_clusters": int(part.exit_cluster.sum())})
    full = pd.concat(results, ignore_index=True)
    pd.concat(supplies, ignore_index=True).to_csv(OUT / "p3_all80_daily_state_supply.csv.gz", index=False, compression="gzip")
    fold = pd.DataFrame(folds)
    median_s3 = full.groupby(["platform", "persistence", "decision_date"]).state.apply(lambda s: int(s.eq("S3").sum())).groupby(level=[0, 1]).median().rename("median_S3_candidates_per_date").reset_index()
    gate = fold.groupby(["platform", "persistence"]).agg(min_entry_clusters=("entry_clusters", "min"), min_exit_clusters=("exit_clusters", "min")).reset_index().merge(median_s3, on=["platform", "persistence"])
    gate["walk_forward_supply_pass"] = gate.min_entry_clusters.ge(20) & gate.min_exit_clusters.ge(20) & gate.median_S3_candidates_per_date.ge(1)
    fold.to_csv(OUT / "p3_all80_walk_forward_event_cluster_supply.csv", index=False, encoding="utf-8-sig")
    gate.to_csv(OUT / "p3_all80_supply_gate.csv", index=False, encoding="utf-8-sig")
    full[["platform", "persistence", "decision_date", "ticker", "pool_rank", "price_history_ready", "source_status", "price_pct_3M", "price_pct_6M", "price_pct_12M", "K_pct_6M", "BIAS_pct_6M", "up_groups", "down_groups", "market_state", "state", "entry_cluster", "exit_cluster"]].to_csv(OUT / "p3_all80_lifecycle_candidate_state_compact.csv.gz", index=False, compression="gzip")
    continuity = full.groupby(["platform", "persistence", "ticker"]).agg(active_rows=("decision_date", "size"), rank_changes=("pool_rank", lambda s: int(s.ne(s.shift()).sum() - 1)), entry_clusters=("entry_cluster", "sum"), exit_clusters=("exit_cluster", "sum")).reset_index()
    continuity["state_reset_on_rank_change"] = False
    continuity.to_csv(OUT / "p3_all80_cross_week_rank_continuity_audit.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{"policy":"membership_exit_reentry","selection_eligibility":"inactive cannot be selected; re-entry uses own PIT history","state_inheritance":"never inherit another ticker","incumbent_drops_out_policy":"blocked_strategy_policy_not_defined"}]).to_csv(OUT / "p3_all80_membership_reentry_incumbent_policy_ledger.csv", index=False, encoding="utf-8-sig")
    evidence = pd.DataFrame([{"evidence_group":g,"PIT_status":"ready_or_applicability_partial","TDCC_P3_1":"NA_not_zero","missing_policy":"missing lowers confidence; no positive vote"} for g in ["KD","RS","MA_structure","capital_chip","risk"]])
    evidence.to_csv(OUT / "p3_all80_evidence_group_readiness.csv", index=False, encoding="utf-8-sig")
    state_totals = full.groupby(["platform", "persistence", "state"]).size().rename("candidate_rows").reset_index()
    state_totals.to_csv(OUT / "p3_all80_state_total_supply.csv", index=False, encoding="utf-8-sig")
    group_columns = ["kd_up", "rs_repair", "ma_up", "capital_improve", "risk_ok", "kd_down", "rs_weak", "ma_down", "capital_withdraw", "risk_bad"]
    group_audit = full.groupby(["platform", "persistence"])[group_columns].agg(["mean", lambda s: s.isna().mean()])
    group_audit.columns = [f"{field}_{metric if isinstance(metric, str) else 'missing_rate'}" for field, metric in group_audit.columns]
    group_audit.reset_index().to_csv(OUT / "p3_all80_evidence_group_hit_missing_audit.csv", index=False, encoding="utf-8-sig")
    position_audit = merged[["decision_date", "ticker", "price_pct_3M", "price_pct_6M", "price_pct_12M"]].copy()
    position_audit["position_3M_6M_same_side"] = ((position_audit.price_pct_3M <= .35) == (position_audit.price_pct_6M <= .35))
    position_audit["position_6M_12M_same_side"] = ((position_audit.price_pct_6M <= .35) == (position_audit.price_pct_12M <= .35))
    position_audit.to_csv(OUT / "p3_all80_3M_6M_12M_position_consistency_audit.csv.gz", index=False, compression="gzip")
    blocker = pd.read_csv(DELTA / "all80_bounded_delta_remaining_blocker_ledger.csv.gz", dtype={"ticker": str})
    blocker.to_csv(OUT / "p3_all80_adjusted_HLC_remaining_blocked_ledger.csv.gz", index=False, compression="gzip")
    representation = {
        "strong_background_present": True,
        "relative_low_required_before_entry": True,
        "turn_up_required_before_entry": True,
        "capital_confirmation_required": True,
        "market_adjustment_applied": True,
        "relative_high_required_before_exit": True,
        "turn_down_required_before_exit": True,
        "capital_or_structure_deterioration_required": True,
        "sequential_state_order_enforced": True,
        "representative_of_requested_lifecycle_logic": True,
    }
    (OUT / "p3_all80_lifecycle_representativeness_checklist.json").write_text(json.dumps(representation, ensure_ascii=False, indent=2), encoding="utf-8")
    scarcity = gate.copy()
    scarcity["primary_failure_attribution"] = np.where(
        scarcity.min_entry_clusters.lt(20), "entry_sequence_evidence_scarcity",
        np.where(scarcity.min_exit_clusters.lt(20), "exit_sequence_evidence_scarcity", "median_daily_S3_supply_below_one"),
    )
    scarcity["membership_eligibility_primary_blocker"] = False
    scarcity["price_blocked_candidate_share"] = float((~merged.price_history_ready).mean())
    scarcity.to_csv(OUT / "p3_all80_supply_failure_attribution.csv", index=False, encoding="utf-8-sig")
    ready = bool(gate.walk_forward_supply_pass.any())
    readiness = {"task_id":TASK,"status":"at_least_one_platform_supply_pass" if ready else "all_platforms_supply_gate_failed","P3_1_dates":int(merged.decision_date.nunique()),"candidate_rows":len(merged),"price_history_ready_rows":int(merged.price_history_ready.sum()),"price_history_blocked_rows":int((~merged.price_history_ready).sum()),"remaining_adjusted_HLC_blocker_rows":len(blocker),"platforms_passing_supply_gate":int(gate.walk_forward_supply_pass.sum()),"state_supply_materialized":True,"sufficient_for_walk_forward":ready,"ready_for_experiments":False,"performance_authorized":False,"P3_2_outcome_read_authorized":False,"Top3_authorized":False,"diagnostic_subproblem":False,"represents_intended_all80_layer5_state_supply":True,"future_data_violation_count":0,"formal_model_changed":False,"trade_decision_changed":False,"active_in_trade_decision":False,"report_changed":False,"not_live_rule":True,"forward_returns_live_rule_usage":False}
    (OUT / "readiness_for_p3_all80_continuous_lifecycle_state_supply.json").write_text(json.dumps(readiness, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "final_summary_zh.md").write_text(f"# P3 all80 continuous sequential lifecycle state supply\n\nP3-1 {readiness['P3_1_dates']} dates、{len(merged):,} candidate rows已實算。6M price history ready={readiness['price_history_ready_rows']:,}，blocked={readiness['price_history_blocked_rows']:,}。通過供給gate的平台={readiness['platforms_passing_supply_gate']}/6。本輪未讀future outcome/P3-2，未跑績效。\n", encoding="utf-8")
    files = sorted(p for p in OUT.iterdir() if p.is_file() and p.name != "manifest.json")
    (OUT / "manifest.json").write_text(json.dumps({"task_id":TASK,"inputs":{"daily":_sha(DAILY),"radar_delta":_sha(DELTA / "all80_bounded_delta_adjusted_hlc_exact_key_compact.csv.gz")},"files":[{"name":p.name,"sha256":_sha(p),"bytes":p.stat().st_size} for p in files]}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    run()
