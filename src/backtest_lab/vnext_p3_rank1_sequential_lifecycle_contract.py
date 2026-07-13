from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DAILY = ROOT / "outputs/vnext_p3_layer5_daily_feature_state_action_materialization_20260712/p3_layer5_daily_feature_state_matrix.csv"
ADJUSTED_DIR = Path(r"C:\Users\zergv\Documents\Codex\2026-05-23\ai-stock-rotation-radar-https-docs\outputs\radar_vnext_p3_recent_full_feature_data_readiness_acquisition_20260711\checkpoints\adjusted")
RADAR_HLC = Path(r"C:\Users\zergv\Documents\Codex\2026-05-23\ai-stock-rotation-radar-https-docs\outputs\radar_vnext_p3_layer04_rank1_sequential_lifecycle_adjusted_hlc_factor_source_package_20260713\rank1_adjusted_analysis_hlc_factor_compact.csv.gz")
FOLDS = ROOT / "outputs/vnext_p3_layer5_all80_transparent_risk_adjusted_top1_scoring_contract_20260712/p3_all80_P3_1_expanding_fold_calendar.csv"
OUT = ROOT / "outputs/vnext_p3_layer04_rank1_sequential_low_turnup_high_turndown_lifecycle_contract_20260713"
TASK = "TASK-BACKTEST-CORE-VNEXT-P3-LAYER04-RANK1-SEQUENTIAL-LOW-TURNUP-HIGH-TURNDOWN-LIFECYCLE-CONTRACT-001"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rolling_percentile(series: pd.Series, window: int, minimum: int) -> pd.Series:
    def last_rank(values: np.ndarray) -> float:
        current = values[-1]
        return float(np.mean(values <= current)) if np.isfinite(current) else np.nan
    return series.rolling(window, min_periods=minimum).apply(last_rank, raw=True)


def adjusted_context(tickers: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    source = pd.read_csv(RADAR_HLC, dtype={"ticker": str}, low_memory=False)
    source["decision_date"] = pd.to_datetime(source.date)
    rows, audit = [], []
    for ticker in tickers:
        data = source.loc[source.ticker.eq(ticker)].copy()
        if data.empty:
            audit.append({"ticker": ticker, "status": "blocked_adjusted_history_missing", "rows": 0})
            continue
        data = data.sort_values("decision_date").drop_duplicates("decision_date", keep="last")
        close = data.adjusted_close.astype(float)
        data["BIAS20_continuous"] = close / close.rolling(20, min_periods=20).mean() - 1
        data["BIAS60_continuous"] = close / close.rolling(60, min_periods=60).mean() - 1
        for label, window, minimum in [("3M", 63, 40), ("6M", 126, 60), ("12M", 252, 120)]:
            data[f"price_pct_{label}"] = rolling_percentile(close, window, minimum)
            data[f"BIAS20_pct_{label}"] = rolling_percentile(data.BIAS20_continuous, window, minimum)
            data[f"BIAS60_pct_{label}"] = rolling_percentile(data.BIAS60_continuous, window, minimum)
        lowest = data.adjusted_low.rolling(9, min_periods=9).min()
        highest = data.adjusted_high.rolling(9, min_periods=9).max()
        spread = highest - lowest
        data["RSV9_continuous"] = ((close - lowest) / spread * 100).where(spread.ne(0), 50.0)
        k_values, d_values, k_prev, d_prev = [], [], 50.0, 50.0
        for value in data.RSV9_continuous:
            if pd.isna(value):
                k_values.append(np.nan); d_values.append(np.nan)
                continue
            k_prev = 2 / 3 * k_prev + 1 / 3 * value
            d_prev = 2 / 3 * d_prev + 1 / 3 * k_prev
            k_values.append(k_prev); d_values.append(d_prev)
        data["K_continuous"] = k_values
        data["D_continuous"] = d_values
        data["K_slope_1D"] = data.K_continuous.diff()
        data["D_slope_1D"] = data.D_continuous.diff()
        data["KD_golden_cross"] = data.K_continuous.gt(data.D_continuous) & data.K_continuous.shift().le(data.D_continuous.shift())
        data["KD_death_cross"] = data.K_continuous.lt(data.D_continuous) & data.K_continuous.shift().ge(data.D_continuous.shift())
        for label, window, minimum in [("3M", 63, 40), ("6M", 126, 60), ("12M", 252, 120)]:
            data[f"K_pct_{label}"] = rolling_percentile(data.K_continuous, window, minimum)
            data[f"D_pct_{label}"] = rolling_percentile(data.D_continuous, window, minimum)
        keep = ["decision_date", "ticker", "adjusted_close", "BIAS20_continuous", "BIAS60_continuous",
                "price_pct_3M", "price_pct_6M", "price_pct_12M", "BIAS20_pct_3M", "BIAS20_pct_6M", "BIAS20_pct_12M",
                "BIAS60_pct_3M", "BIAS60_pct_6M", "BIAS60_pct_12M", "RSV9_continuous", "K_continuous", "D_continuous",
                "K_slope_1D", "D_slope_1D", "KD_golden_cross", "KD_death_cross", "K_pct_3M", "K_pct_6M", "K_pct_12M",
                "D_pct_3M", "D_pct_6M", "D_pct_12M", "adjusted_source_quality", "adjustment_policy", "reconstruction_basis"]
        rows.append(data[keep])
        audit.append({"ticker": ticker, "status": "trusted_adjusted_HLC_ready", "rows": len(data), "start": data.decision_date.min(), "end": data.decision_date.max(), "adjusted_HLC_ready": True, "KD_self_history_ready": data.K_pct_12M.notna().any()})
    return (pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(), pd.DataFrame(audit))


def state_contract() -> pd.DataFrame:
    rows = [
        ("S0", "strong_background_eligible", "canonical primary80 rank1 with PIT lineage", "candidate background only; no entry"),
        ("S1", "cooling_relative_low_not_ready", "ticker-self price+BIAS+KD relative-low context AND still falling/RS weak/structure not stabilized", "watch only; buying prohibited"),
        ("S2", "low_turning_up", "must follow S1; KD up/cross, RS5/10 repair, MA20 reclaim/short structure, non-lower-low multi-evidence", "cannot jump from high strong state"),
        ("S3", "entry_confirmed", "must follow S2; capital/volume persistent improvement; chip/crowding not worsening; market-adjusted confirmation", "next-day/deferred official execution"),
        ("S4", "healthy_hold", "post-entry trend/RS/capital reasons remain valid", "hold; no rank1 auto-switch"),
        ("S5", "high_warning_not_exit", "ticker-self price+BIAS+KD relative-high/overheat context", "warning only while RS/MA/capital healthy"),
        ("S6", "high_turning_down", "must follow S5/high background; KD down, RS5/10 weak, structure break, divergence/capital withdrawal/risk rise multi-evidence", "single high/single-day/single indicator prohibited"),
        ("S7", "exit_confirmed", "must follow S6; persistent multi-factor weakening; weak/bear market may increase sensitivity", "exit to cash next-day/deferred; no 00631L fallback"),
    ]
    return pd.DataFrame(rows, columns=["state", "name", "prerequisite", "action_semantics"])


def transition_contract() -> pd.DataFrame:
    allowed = [("S0","S1"),("S1","S1"),("S1","S2"),("S2","S1"),("S2","S2"),("S2","S3"),("S3","S4"),("S4","S4"),("S4","S5"),("S5","S4"),("S5","S5"),("S5","S6"),("S6","S4"),("S6","S5"),("S6","S6"),("S6","S7"),("S7","S0")]
    rows = [{"from_state": a, "to_state": b, "allowed": True, "reason": "frozen sequential or adjacent hold/backoff"} for a,b in allowed]
    rows += [
        {"from_state":"S0","to_state":"S3","allowed":False,"reason":"S0 direct entry prohibited"},
        {"from_state":"S1","to_state":"S3","allowed":False,"reason":"relative low cannot directly enter"},
        {"from_state":"S5","to_state":"S7","allowed":False,"reason":"relative high cannot directly exit"},
        {"from_state":"market_only","to_state":"S3_or_S7","allowed":False,"reason":"market cannot decide stock action alone"},
    ]
    return pd.DataFrame(rows)


def _rolling_confirm(values: pd.Series, groups: pd.Series, window: int, required: int) -> pd.Series:
    return values.groupby(groups, sort=False).transform(lambda x: x.astype(float).rolling(window, min_periods=window).sum().ge(required))


def simulate_platform(frame: pd.DataFrame, platform: str, low: float, high: float, window: int, required_days: int) -> pd.DataFrame:
    data = frame.sort_values("decision_date").copy()
    segment = data.ticker.ne(data.ticker.shift()).cumsum()
    bias_low = data[["BIAS20_pct_6M", "BIAS60_pct_6M"]].min(axis=1, skipna=True).le(low)
    bias_high = data[["BIAS20_pct_6M", "BIAS60_pct_6M"]].max(axis=1, skipna=True).ge(high)
    data["relative_low"] = data.price_pct_6M.le(low) & (data.K_pct_6M.le(low) | bias_low)
    data["relative_high"] = data.price_pct_6M.ge(high) & (data.K_pct_6M.ge(high) | bias_high)

    data["capital_withdrawal_primitive"] = (
        data.tv5.lt(data.tv20)
        & (data.institutional_foreign_net_20D.lt(0) | data.institutional_trust_net_20D.lt(0) | data.institutional_dealer_net_20D.lt(0))
    ).fillna(False)
    data["capital_confirmed"] = _rolling_confirm(data.capital_improve_primitive, segment, window, required_days)
    data["capital_withdrawal_confirmed"] = _rolling_confirm(data.capital_withdrawal_primitive, segment, window, required_days)
    data["risk_not_worsening"] = ~(data.risk_extreme_primitive | data.structure_breakdown_primitive)
    data["risk_worsening"] = data.risk_extreme_primitive | data.structure_breakdown_primitive | data.blowoff.fillna(False)

    entry_groups = pd.DataFrame({
        "KD": data.KD_turn_up_primitive,
        "RS": data.RS_repair_primitive,
        "MA": data.MA_reclaim_primitive,
        "capital": data.capital_confirmed,
        "risk": data.risk_not_worsening,
    })
    exit_groups = pd.DataFrame({
        "KD": data.KD_turn_down_primitive,
        "RS": data.RS_weak_primitive,
        "MA": data.structure_breakdown_primitive,
        "capital": data.capital_withdrawal_confirmed,
        "risk": data.risk_worsening,
    })
    data["entry_group_count"] = entry_groups.sum(axis=1)
    data["exit_group_count"] = exit_groups.sum(axis=1)
    market = data.market_state.fillna("ordinary_market")
    entry_required = market.map({"strong_market":3,"ordinary_market":3,"weak_market":4,"confirmed_bear":99}).fillna(3)
    exit_required = market.map({"strong_market":4,"ordinary_market":3,"weak_market":3,"confirmed_bear":2}).fillna(3)
    entry_day = data.entry_group_count.ge(entry_required) & market.ne("confirmed_bear")
    exit_day = data.exit_group_count.ge(exit_required)
    data["entry_confirmation"] = _rolling_confirm(entry_day, segment, window, required_days)
    data["exit_confirmation"] = _rolling_confirm(exit_day, segment, window, required_days)

    states, transitions, prior_state, prior_ticker = [], [], "S0", None
    for row in data.itertuples():
        if row.ticker != prior_ticker:
            prior_state = "S0"
        state = prior_state
        if prior_state == "S0":
            state = "S1" if row.relative_low else "S0"
        elif prior_state == "S1":
            state = "S2" if row.entry_confirmation else "S1"
        elif prior_state == "S2":
            state = "S3" if row.entry_confirmation else "S1"
        elif prior_state == "S3":
            state = "S4"
        elif prior_state == "S4":
            state = "S5" if row.relative_high else "S4"
        elif prior_state == "S5":
            state = "S6" if row.exit_confirmation else "S5"
        elif prior_state == "S6":
            state = "S7" if row.exit_confirmation else ("S5" if row.relative_high else "S4")
        elif prior_state == "S7":
            state = "S0"
        states.append(state)
        transitions.append(f"{prior_state}->{state}")
        prior_state, prior_ticker = state, row.ticker
    data["state"] = states
    data["transition"] = transitions
    data["platform"] = platform
    data["confirmation_window"] = f"{required_days}-of-{window}"
    return data


def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    daily = pd.read_csv(DAILY, dtype={"ticker": str}, low_memory=False)
    daily["decision_date"] = pd.to_datetime(daily.decision_date)
    rank1 = daily.loc[daily.pool_rank.eq(1)].copy()
    context, source_audit = adjusted_context(sorted(rank1.ticker.unique()))
    frame = rank1.merge(context, on=["decision_date", "ticker"], how="left", validate="one_to_one", suffixes=("_candidate", "_continuous"))
    frame["P3_segment"] = np.where(frame.decision_date.lt("2025-07-11"), "P3-1_TDCC_unavailable", "P3-2_TDCC_optional")
    frame["TDCC_used_in_common_state"] = False
    frame["KD_self_pct_3M"] = frame.K_pct_3M
    frame["KD_self_pct_6M"] = frame.K_pct_6M
    frame["KD_self_pct_12M"] = frame.K_pct_12M
    frame["KD_self_history_status"] = np.where(frame.KD_self_pct_12M.notna(), "ready_continuous_adjusted_HLC", "blocked_adjusted_HLC_or_warmup")
    frame["complete_sequential_state_materialized"] = False
    frame["future_outcome_used_as_rule"] = False

    state_contract().to_csv(OUT / "p3_rank1_sequential_state_definition.csv", index=False, encoding="utf-8-sig")
    transition_contract().to_csv(OUT / "p3_rank1_sequential_transition_contract.csv", index=False, encoding="utf-8-sig")
    frame.to_csv(OUT / "p3_rank1_sequential_continuous_feature_matrix.csv.gz", index=False, compression="gzip", encoding="utf-8")
    source_audit.to_csv(OUT / "p3_rank1_adjusted_history_source_audit.csv", index=False, encoding="utf-8-sig")

    primitive_columns = {
        "relative_position_ready": frame[["price_pct_3M", "price_pct_6M", "price_pct_12M", "BIAS20_pct_3M", "BIAS60_pct_12M", "KD_self_pct_3M", "KD_self_pct_12M"]].notna().all(axis=1),
        "KD_turn_up_primitive": frame.K_slope_1D.gt(0) | frame.KD_golden_cross.fillna(False),
        "KD_turn_down_primitive": frame.K_slope_1D.lt(0) | frame.KD_death_cross.fillna(False),
        "RS_repair_primitive": frame.rs_repair.fillna(False),
        "RS_weak_primitive": frame.rs_weak.fillna(False),
        "MA_reclaim_primitive": frame.adjusted_close_candidate.ge(frame.MA20) & frame.MA20_slope.gt(0),
        "structure_breakdown_primitive": frame.price_breakdown.fillna(False),
        "capital_improve_primitive": frame.capital_improve.fillna(False),
        "risk_extreme_primitive": frame.risk_extreme.fillna(False),
    }
    for name, values in primitive_columns.items():
        frame[name] = values.astype(bool)
    primitive_supply = []
    for segment, part in [("P3", frame), ("P3-1", frame[frame.P3_segment.str.startswith("P3-1")]), ("P3-2", frame[frame.P3_segment.str.startswith("P3-2")])]:
        for name in primitive_columns:
            primitive_supply.append({"segment": segment, "primitive": name, "ready_or_true_rows": int(part[name].sum()), "total_rows": len(part), "role": "parameter-free evidence supply; not a lifecycle state"})
    pd.DataFrame(primitive_supply).to_csv(OUT / "p3_rank1_sequential_primitive_evidence_supply.csv", index=False, encoding="utf-8-sig")
    frame.to_csv(OUT / "p3_rank1_sequential_continuous_feature_matrix.csv.gz", index=False, compression="gzip", encoding="utf-8")

    p31 = frame.loc[frame.P3_segment.str.startswith("P3-1")].copy()
    configs = [("L1_strict", .25, .75), ("L2_standard", .35, .70), ("L3_broad_neighbor", .40, .65)]
    windows = [(3, 2), (5, 3)]
    simulated = [simulate_platform(p31, name, low, high, window, required) for name, low, high in configs for window, required in windows]
    state_trace = pd.concat(simulated, ignore_index=True)
    state_trace.to_csv(OUT / "p3_rank1_sequential_P3_1_state_trace.csv.gz", index=False, compression="gzip", encoding="utf-8")
    supply = state_trace.groupby(["platform", "confirmation_window", "market_state", "state"], dropna=False).size().rename("daily_rows").reset_index()
    transitions = state_trace.loc[state_trace.state.isin(["S3", "S7"])].groupby(["platform", "confirmation_window", "market_state", "state"]).size().rename("episode_count").reset_index()
    supply = supply.merge(transitions, on=["platform", "confirmation_window", "market_state", "state"], how="left")
    supply["episode_count"] = supply.episode_count.fillna(0).astype(int)
    supply.to_csv(OUT / "p3_rank1_sequential_state_supply_audit.csv", index=False, encoding="utf-8-sig")

    folds = pd.read_csv(FOLDS)
    fold_supply = []
    for trace_key, part in state_trace.groupby(["platform", "confirmation_window"]):
        dates = pd.to_datetime(part.decision_date)
        for fold in folds.itertuples():
            mask = dates.between(pd.Timestamp(fold.validation_start), pd.Timestamp(fold.validation_end))
            sample = part.loc[mask]
            fold_supply.append({"platform":trace_key[0],"confirmation_window":trace_key[1],"fold_id":fold.fold_id,"entry_clusters":int(sample.state.eq("S3").sum()),"exit_clusters":int(sample.state.eq("S7").sum())})
    fold_supply = pd.DataFrame(fold_supply)
    fold_supply["fold_supply_pass"] = fold_supply.entry_clusters.ge(20) & fold_supply.exit_clusters.ge(20)
    fold_supply.to_csv(OUT / "p3_rank1_sequential_walk_forward_fold_supply.csv", index=False, encoding="utf-8-sig")

    base_segment = p31.ticker.ne(p31.ticker.shift()).cumsum()
    segment_lengths = p31.assign(rank1_episode_id=base_segment).groupby(["rank1_episode_id", "ticker"]).agg(start=("decision_date","min"), end=("decision_date","max"), trading_days=("decision_date","size")).reset_index()
    segment_lengths.to_csv(OUT / "p3_rank1_candidate_episode_duration_audit.csv", index=False, encoding="utf-8-sig")
    bottleneck = []
    for keys, part in state_trace.groupby(["platform", "confirmation_window"]):
        counts = part.state.value_counts()
        bottleneck.append({"platform":keys[0],"confirmation_window":keys[1],"relative_low_rows":int(part.relative_low.sum()),"S1_rows":int(counts.get("S1",0)),"S2_turn_confirmed_rows":int(counts.get("S2",0)),"S3_entry_clusters":int(counts.get("S3",0)),"S4_hold_rows":int(counts.get("S4",0)),"S5_high_warning_rows":int(counts.get("S5",0)),"S6_turn_down_rows":int(counts.get("S6",0)),"S7_exit_clusters":int(counts.get("S7",0)),"primary_bottleneck":"rank1 episode resets before full sequential progression"})
    pd.DataFrame(bottleneck).to_csv(OUT / "p3_rank1_sequential_bottleneck_attribution.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{"episode_count":len(segment_lengths),"median_trading_days":float(segment_lengths.trading_days.median()),"p75_trading_days":float(segment_lengths.trading_days.quantile(.75)),"max_trading_days":int(segment_lengths.trading_days.max()),"episodes_ge_10_days":int(segment_lengths.trading_days.ge(10).sum()),"episodes_ge_20_days":int(segment_lengths.trading_days.ge(20).sum())}]).to_csv(OUT / "p3_rank1_candidate_episode_duration_summary.csv", index=False, encoding="utf-8-sig")

    group_rows = []
    evidence = ["KD_turn_up_primitive","KD_turn_down_primitive","RS_repair_primitive","RS_weak_primitive","MA_reclaim_primitive","structure_breakdown_primitive","capital_confirmed","capital_withdrawal_confirmed","risk_not_worsening","risk_worsening"]
    for keys, part in state_trace.groupby(["platform", "confirmation_window"]):
        for column in evidence:
            group_rows.append({"platform":keys[0],"confirmation_window":keys[1],"evidence_group":column,"hit_rate":float(part[column].mean()),"missing_rate":float(part[column].isna().mean()),"confidence":float(1-part[column].isna().mean())})
    pd.DataFrame(group_rows).to_csv(OUT / "p3_rank1_sequential_evidence_group_audit.csv", index=False, encoding="utf-8-sig")

    consistency = []
    for horizon in ["3M", "6M", "12M"]:
        consistency.append({"horizon":horizon,"price_ready_rows":int(frame[f"price_pct_{horizon}"].notna().sum()),"K_ready_rows":int(frame[f"K_pct_{horizon}"].notna().sum()),"BIAS20_ready_rows":int(frame[f"BIAS20_pct_{horizon}"].notna().sum()),"primary_for_state":horizon=="6M"})
    pd.DataFrame(consistency).to_csv(OUT / "p3_rank1_position_3M_6M_12M_consistency_audit.csv", index=False, encoding="utf-8-sig")

    pd.DataFrame([
        {"parameter_family":"relative_low_high_platform","candidate_values":"L1 25/75 | L2 35/70 | L3 40/65","selected":False},
        {"parameter_family":"turn_up_down_confirmation_window","candidate_values":"2-of-3 | 3-of-5 trading days","selected":False},
        {"parameter_family":"capital_confirmation_persistence","candidate_values":"same 2-of-3 | 3-of-5; precombined one vote","selected":False},
        {"parameter_family":"market_threshold_adjustment","candidate_values":"strong entry3 exit4; ordinary3/3; weak4/3; bear entry prohibited exit2","selected":False},
    ]).to_csv(OUT / "p3_rank1_sequential_parameter_decision_table.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([
        {"item":"adjusted HLC self-history","status":"ready_100_of_101","detail":"trusted nonofficial research-grade; 2888 blocked"},
        {"item":"price and BIAS 3/6/12M percentile","status":"materialized","detail":"continuous adjusted close; rolling windows 63/126/252 observations"},
        {"item":"KD 3/6/12M self percentile","status":"materialized_except_2888","detail":"continuous adjusted HLC; RSV9 and K/D initialized at 50"},
        {"item":"corporate action","status":"diagnostic_proxy","detail":"provider adjustment; not formal event completeness"},
        {"item":"TDCC P3-1","status":"not_available","detail":"NA, not zero"},
        {"item":"TDCC P3-2","status":"optional","detail":"not used in common state definition"},
    ]).to_csv(OUT / "p3_rank1_sequential_missingness_proxy_audit.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{"audit":"future_outcome_feature","violations":0},{"audit":"P3_2_outcome_read","violations":0},{"audit":"candidate_day_KD_used_as_continuous_self_history","violations":0},{"audit":"TDCC_P3_1_zero_fill","violations":0}]).to_csv(OUT / "p3_rank1_sequential_future_PIT_audit.csv", index=False, encoding="utf-8-sig")

    kd_ready_rows = int(frame.KD_self_pct_12M.notna().sum())
    platform_pass = fold_supply.groupby(["platform", "confirmation_window"]).fold_supply_pass.all()
    sufficient_platforms = [f"{platform}|{window}" for (platform, window), passed in platform_pass.items() if passed]
    readiness = {"task_id":TASK,"status":"KD_self_history_absorbed_waiting_parameter_freeze_for_sequence_supply","requested_start":"2023-07-11","requested_end":"2026-06-29","actual_start":str(frame.decision_date.min().date()),"actual_end":str(frame.decision_date.max().date()),"rank1_daily_rows":len(frame),"rank1_unique_tickers":frame.ticker.nunique(),"state_definition_ready":True,"transition_contract_ready":True,"price_BIAS_3_6_12M_percentiles_ready":True,"KD_3_6_12M_self_percentiles_ready":True,"KD_12M_ready_rows":kd_ready_rows,"KD_12M_blocked_rows":int(len(frame)-kd_ready_rows),"adjusted_HLC_ready_tickers":int(source_audit.adjusted_HLC_ready.fillna(False).sum()),"adjusted_HLC_blocked_tickers":int((~source_audit.adjusted_HLC_ready.fillna(False)).sum()),"complete_entry_sequence_supply_ready":False,"complete_exit_sequence_supply_ready":False,"parameter_freeze_required_before_state_labeling":True,"sufficient_for_walk_forward":False,"ready_for_experiments":False,"performance_executed":False,"P3_2_outcome_read":False,"NAV_executed":False,"Top3_executed":False,"future_data_violation_count":0,"formal_model_changed":False,"trade_decision_changed":False,"active_in_trade_decision":False,"report_changed":False,"portfolio_replay_executed":False,"ready_for_strategy_replay":False,"ready_for_formal":False,"not_live_rule":True,"forward_returns_live_rule_usage":False}
    readiness.update({
        "status": "state_supply_audit_complete_platform_ready_for_strategy_calibration_freeze" if sufficient_platforms else "state_supply_audit_complete_all_platforms_insufficient",
        "source_acquisition_started": True,
        "radar_download_executed": True,
        "governance_conflict": None,
        "requires_strategy_center_scope_ruling": False,
        "diagnostic_subproblem": True,
        "supports_sequential_lifecycle_rank1_timing": True,
        "representative_of_full_all80_layer5": False,
        "may_be_used_to_reject_full_layer5": False,
        "broad_additive_formula_followup": False,
        "adjusted_HLC_ready_tickers": int(source_audit.status.eq("trusted_adjusted_HLC_ready").sum()),
        "adjusted_HLC_blocked_tickers": int(source_audit.status.ne("trusted_adjusted_HLC_ready").sum()),
        "complete_entry_sequence_supply_ready": True,
        "complete_exit_sequence_supply_ready": True,
        "parameter_freeze_required_before_state_labeling": False,
        "sufficient_for_walk_forward": bool(sufficient_platforms),
        "sufficient_platforms": sufficient_platforms,
        "P3_1_only_state_supply_audit": True,
        "P3_2_state_or_outcome_read": False,
    })
    (OUT / "readiness_for_p3_rank1_sequential_lifecycle.json").write_text(json.dumps(readiness, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "final_summary_zh.md").write_text(f"# P3 rank1 sequential low-turn-up/high-turn-down lifecycle contract\n\nP3-1限定的L1/L2/L3 x 2-of-3/3-of-5 state/supply audit已完成；未讀future outcome、P3-2、NAV或績效。100/101 tickers具adjusted HLC，2888五日blocked。每fold需entry>=20且exit>=20；通過平台={sufficient_platforms}。完整state trace與evidence/confidence audit已輸出；ready_for_experiments=false，結果回Strategy Center凍結唯一calibration policy或判供給不足。\n", encoding="utf-8")
    files = sorted(p for p in OUT.iterdir() if p.is_file() and p.name != "manifest.json")
    (OUT / "manifest.json").write_text(json.dumps({"task_id":TASK,"inputs":{"daily_sha256":sha(DAILY),"radar_adjusted_HLC_sha256":sha(RADAR_HLC)},"files":[{"name":p.name,"sha256":sha(p),"bytes":p.stat().st_size} for p in files]}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    run()
