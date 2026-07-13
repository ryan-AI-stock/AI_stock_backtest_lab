from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from backtest_lab import vnext_p3_rank1_sequential_lifecycle_contract as source


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/vnext_p3_layer04_rank1_ticker_specific_KD_price_range_timing_stage_A_contract_20260713"
TASK = "TASK-BACKTEST-CORE-VNEXT-P3-LAYER04-RANK1-TICKER-SPECIFIC-KD-PRICE-RANGE-TIMING-STAGE-A-CONTRACT-001"
P3_1_END = pd.Timestamp("2025-07-10")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _last_percentile(values: np.ndarray) -> float:
    current = values[-1]
    return float(np.mean(values <= current)) if np.isfinite(current) else np.nan


def _build_features() -> tuple[pd.DataFrame, pd.DataFrame]:
    daily = pd.read_csv(source.DAILY, dtype={"ticker": str}, low_memory=False)
    daily["decision_date"] = pd.to_datetime(daily.decision_date)
    rank1 = daily.loc[daily.pool_rank.eq(1), ["decision_date", "ticker", "membership_snapshot_date", "membership_effective_date"]].copy()
    rank1 = rank1.loc[rank1.decision_date.le(P3_1_END)].drop_duplicates("decision_date")

    raw = pd.read_csv(source.RADAR_HLC, dtype={"ticker": str}, low_memory=False)
    raw["decision_date"] = pd.to_datetime(raw.date)
    raw = raw.loc[raw.ticker.isin(rank1.ticker.unique())].sort_values(["ticker", "decision_date"])
    pieces = []
    for ticker, data in raw.groupby("ticker", sort=False):
        data = data.drop_duplicates("decision_date", keep="last").copy()
        low9 = data.adjusted_low.rolling(9, min_periods=9).min()
        high9 = data.adjusted_high.rolling(9, min_periods=9).max()
        spread9 = high9 - low9
        data["RSV9"] = ((data.adjusted_close - low9) / spread9 * 100).where(spread9.ne(0), 50.0)
        k_values, d_values, k_prev, d_prev = [], [], 50.0, 50.0
        for value in data.RSV9:
            if pd.isna(value):
                k_values.append(np.nan); d_values.append(np.nan)
                continue
            k_prev = 2 / 3 * k_prev + value / 3
            d_prev = 2 / 3 * d_prev + k_prev / 3
            k_values.append(k_prev); d_values.append(d_prev)
        data["K"] = k_values
        data["D"] = d_values
        data["prior_adjusted_close"] = data.adjusted_close.shift()
        data["KD_cross_up"] = data.K.gt(data.D) & data.K.shift().le(data.D.shift())
        data["KD_cross_down"] = data.K.lt(data.D) & data.K.shift().ge(data.D.shift())
        for window in (60, 120):
            minimum = window
            for field in ("K", "D", "adjusted_close"):
                data[f"{field}_min_{window}TD"] = data[field].rolling(window, min_periods=minimum).min()
                data[f"{field}_max_{window}TD"] = data[field].rolling(window, min_periods=minimum).max()
                width = data[f"{field}_max_{window}TD"] - data[f"{field}_min_{window}TD"]
                data[f"{field}_location_{window}TD"] = ((data[field] - data[f"{field}_min_{window}TD"]) / width).where(width.gt(0))
                data[f"{field}_empirical_pct_{window}TD"] = data[field].rolling(window, min_periods=minimum).apply(_last_percentile, raw=True)
            data[f"K_range_width_{window}TD"] = data[f"K_max_{window}TD"] - data[f"K_min_{window}TD"]
            data[f"D_range_width_{window}TD"] = data[f"D_max_{window}TD"] - data[f"D_min_{window}TD"]
            rolling_high = data.adjusted_high.rolling(window, min_periods=minimum).max()
            rolling_low = data.adjusted_low.rolling(window, min_periods=minimum).min()
            data[f"adjusted_price_range_pct_{window}TD"] = rolling_high / rolling_low - 1
        pieces.append(data)
    features = pd.concat(pieces, ignore_index=True)
    features["rank1_on_date"] = features.set_index(["decision_date", "ticker"]).index.isin(rank1.set_index(["decision_date", "ticker"]).index)
    lineage = rank1.merge(features[["decision_date", "ticker", "adjusted_source_quality", "official_raw_source_quality"]], on=["decision_date", "ticker"], how="left", validate="one_to_one")
    return features, lineage


def _next_ticker_row(features: pd.DataFrame, ticker: str, date: pd.Timestamp) -> pd.Series | None:
    rows = features.loc[features.ticker.eq(ticker) & features.decision_date.gt(date)].sort_values("decision_date")
    return None if rows.empty else rows.iloc[0]


def _simulate(features: pd.DataFrame, rank1_map: dict[pd.Timestamp, str], window: int, zone: float, latch: int, minimum_k_range: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    calendar = sorted(date for date in rank1_map if date <= P3_1_END)
    lookup = features.set_index(["decision_date", "ticker"])
    incumbent = None
    low_setup = None
    high_setup = None
    prior_candidate = None
    actions, events, exclusions = [], [], []
    for date in calendar:
        candidate = rank1_map[date]
        monitored = incumbent or candidate
        if incumbent is None and candidate != prior_candidate:
            low_setup = None
        row = lookup.loc[(date, monitored)] if (date, monitored) in lookup.index else None
        status, reason, execution_status = "hold_or_wait", "no_signal", "not_requested"
        entry_signal = exit_signal = False
        if row is None:
            reason = "blocked_missing_analysis_row"
        else:
            price_loc = row[f"adjusted_close_location_{window}TD"]
            k_loc = row[f"K_location_{window}TD"]
            ready = pd.notna(price_loc) and pd.notna(k_loc) and pd.notna(row.D)
            if not ready:
                reason = "blocked_self_range_warmup_or_missing"
            elif incumbent is None:
                if price_loc <= zone and k_loc <= zone:
                    k_range = row[f"K_range_width_{window}TD"]
                    if pd.notna(k_range) and k_range > minimum_k_range:
                        low_setup = {"ticker": candidate, "date": date, "remaining": latch}
                    else:
                        reason = "entry_setup_blocked_K_range_not_strictly_above_threshold"
                        exclusions.append({"platform": f"{window}TD_zone{int(zone*100)}_latch{latch}_KrangeGT{minimum_k_range}", "decision_date": date, "ticker": candidate, "minimum_K_range_threshold": minimum_k_range, "K_min": row[f"K_min_{window}TD"], "K_max": row[f"K_max_{window}TD"], "K_range_width": k_range, "D_range_width": row[f"D_range_width_{window}TD"], "adjusted_price_range_pct": row[f"adjusted_price_range_pct_{window}TD"], "future_5_10_20_40TD_outcome_role": "evaluation_metadata_only_not_materialized_by_state_rule"})
                elif low_setup and low_setup["ticker"] == candidate:
                    low_setup["remaining"] -= 1
                    if low_setup["remaining"] < 0:
                        low_setup = None
                entry_signal = bool(low_setup and low_setup["ticker"] == candidate and row.KD_cross_up and row.adjusted_close > row.prior_adjusted_close)
                if entry_signal:
                    status, reason = "entry_signal", "low_setup_then_K_cross_up_and_price_up"
            else:
                if price_loc >= 1 - zone and k_loc >= 1 - zone:
                    high_setup = {"ticker": incumbent, "date": date, "remaining": latch}
                elif high_setup and high_setup["ticker"] == incumbent:
                    high_setup["remaining"] -= 1
                    if high_setup["remaining"] < 0:
                        high_setup = None
                exit_signal = bool(high_setup and high_setup["ticker"] == incumbent and row.KD_cross_down and row.adjusted_close < row.prior_adjusted_close)
                if exit_signal:
                    status, reason = "exit_signal", "high_setup_then_K_cross_down_and_price_down"

        if entry_signal or exit_signal:
            target = candidate if entry_signal else incumbent
            execution = _next_ticker_row(features, target, date)
            execution_ready = execution is not None and pd.notna(execution.official_raw_close)
            execution_status = "ready_exact_next_ticker_trading_day" if execution_ready else "blocked_official_raw_execution_close"
            event = {
                "platform": f"{window}TD_zone{int(zone*100)}_latch{latch}_KrangeGT{minimum_k_range}", "decision_date": date, "ticker": target,
                "event_type": status, "reason": reason, "analysis_adjusted_close": row.adjusted_close,
                "K": row.K, "D": row.D, "K_min": row[f"K_min_{window}TD"], "K_max": row[f"K_max_{window}TD"],
                "price_min": row[f"adjusted_close_min_{window}TD"], "price_max": row[f"adjusted_close_max_{window}TD"],
                "K_normalized_location": row[f"K_location_{window}TD"], "price_normalized_location": row[f"adjusted_close_location_{window}TD"],
                "K_range_width": row[f"K_range_width_{window}TD"], "D_range_width": row[f"D_range_width_{window}TD"], "adjusted_price_range_pct": row[f"adjusted_price_range_pct_{window}TD"], "minimum_K_range_threshold": minimum_k_range,
                "K_empirical_percentile_context": row[f"K_empirical_pct_{window}TD"], "price_empirical_percentile_context": row[f"adjusted_close_empirical_pct_{window}TD"],
                "cross_up": bool(row.KD_cross_up), "cross_down": bool(row.KD_cross_down),
                "execution_date": execution.decision_date if execution is not None else pd.NaT,
                "official_raw_execution_close": execution.official_raw_close if execution is not None else np.nan,
                "official_raw_source_quality": execution.official_raw_source_quality if execution is not None else None,
                "execution_status": execution_status, "EP05_cost_hook": True, "slippage_bp_primary": 10,
            }
            events.append(event)
            if execution_ready:
                incumbent = target if entry_signal else None
                low_setup = None if entry_signal else low_setup
                high_setup = None if exit_signal else high_setup
        actions.append({"platform": f"{window}TD_zone{int(zone*100)}_latch{latch}_KrangeGT{minimum_k_range}", "decision_date": date, "canonical_rank1": candidate, "monitored_ticker": monitored, "incumbent": incumbent, "action": status, "reason": reason, "execution_status": execution_status, "market_controller_used": False, "future_outcome_used_as_rule": False})
        prior_candidate = candidate
    return pd.DataFrame(actions), pd.DataFrame(events), pd.DataFrame(exclusions)


def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    obsolete = OUT / "p3_rank1_dynamic_self_range_12_platform_supply.csv"
    if obsolete.exists():
        obsolete.unlink()
    features, lineage = _build_features()
    p31_features = features.loc[features.decision_date.le(P3_1_END)].copy()
    daily = pd.read_csv(source.DAILY, dtype={"ticker": str}, usecols=["decision_date", "ticker", "pool_rank"])
    daily["decision_date"] = pd.to_datetime(daily.decision_date)
    rank1 = daily.loc[daily.pool_rank.eq(1) & daily.decision_date.le(P3_1_END)].drop_duplicates("decision_date")
    rank1_map = dict(zip(rank1.decision_date, rank1.ticker))
    platform_rows = []
    all_actions, all_events, all_exclusions = [], [], []
    for window in (60, 120):
        for zone in (.10, .20, .30):
            for latch in (5, 10):
                for minimum_k_range in (0, 20, 25, 30):
                    actions, events, exclusions = _simulate(p31_features, rank1_map, window, zone, latch, minimum_k_range)
                    all_actions.append(actions); all_events.append(events); all_exclusions.append(exclusions)
                    platform_rows.append({"platform": f"{window}TD_zone{int(zone*100)}_latch{latch}_KrangeGT{minimum_k_range}", "range_window_TD": window, "low_zone": zone, "high_zone": 1-zone, "latch_TD": latch, "minimum_K_range_threshold": minimum_k_range, "entry_events": int(events.event_type.eq("entry_signal").sum()) if len(events) else 0, "exit_events": int(events.event_type.eq("exit_signal").sum()) if len(events) else 0, "range_gate_excluded_low_setup_rows": len(exclusions), "execution_ready_events": int(events.execution_status.eq("ready_exact_next_ticker_trading_day").sum()) if len(events) else 0, "blocked_execution_events": int(events.execution_status.ne("ready_exact_next_ticker_trading_day").sum()) if len(events) else 0, "holding_date_share": float(actions.incumbent.notna().mean())})
    actions = pd.concat(all_actions, ignore_index=True)
    events = pd.concat(all_events, ignore_index=True) if any(len(x) for x in all_events) else pd.DataFrame()
    exclusions = pd.concat(all_exclusions, ignore_index=True) if any(len(x) for x in all_exclusions) else pd.DataFrame()
    platforms = pd.DataFrame(platform_rows)

    folds = pd.read_csv(source.FOLDS)
    fold_rows = []
    for platform, part in events.groupby("platform"):
        for fold in folds.itertuples():
            sample = part.loc[part.decision_date.between(pd.Timestamp(fold.validation_start), pd.Timestamp(fold.validation_end))]
            fold_rows.append({"platform": platform, "fold_id": fold.fold_id, "validation_start": fold.validation_start, "validation_end": fold.validation_end, "embargo_decision_dates": 40, "entry_events": int(sample.event_type.eq("entry_signal").sum()), "exit_events": int(sample.event_type.eq("exit_signal").sum()), "performance_not_computed_by_core": True})
    fold_supply = pd.DataFrame(fold_rows)

    feature_cols = ["decision_date", "ticker", "adjusted_close", "K", "D", "KD_cross_up", "KD_cross_down", "prior_adjusted_close", "adjusted_source_quality", "adjustment_policy", "reconstruction_basis", "official_raw_close", "official_raw_source_quality"] + [f"{field}_{kind}_{window}TD" for window in (60,120) for field in ("K","D","adjusted_close") for kind in ("min","max","location","empirical_pct")] + [f"{field}_{window}TD" for window in (60,120) for field in ("K_range_width","D_range_width","adjusted_price_range_pct")]
    p31_features[feature_cols].to_csv(OUT / "p3_rank1_dynamic_self_range_continuous_features.csv.gz", index=False, compression="gzip", encoding="utf-8")
    actions.to_csv(OUT / "p3_rank1_dynamic_self_range_daily_action_trace.csv.gz", index=False, compression="gzip", encoding="utf-8")
    events.to_csv(OUT / "p3_rank1_dynamic_self_range_entry_exit_event_ledger.csv", index=False, encoding="utf-8-sig")
    exclusions.to_csv(OUT / "p3_rank1_dynamic_self_range_K_range_gate_excluded_event_ledger.csv", index=False, encoding="utf-8-sig")
    platforms.to_csv(OUT / "p3_rank1_dynamic_self_range_12x4_platform_supply.csv", index=False, encoding="utf-8-sig")
    fold_supply.to_csv(OUT / "p3_rank1_dynamic_self_range_fold_supply.csv", index=False, encoding="utf-8-sig")
    lineage.to_csv(OUT / "p3_rank1_dynamic_self_range_PIT_lineage.csv", index=False, encoding="utf-8-sig")
    blocked = events.loc[events.execution_status.ne("ready_exact_next_ticker_trading_day")].copy() if len(events) else pd.DataFrame()
    blocked.to_csv(OUT / "p3_rank1_dynamic_self_range_blocked_ledger.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([
        {"audit": "future_outcome_used_as_rule", "violations": 0},
        {"audit": "P3_2_outcome_read", "violations": 0},
        {"audit": "market_controller_used", "violations": 0},
        {"audit": "same_day_execution", "violations": int((events.execution_date <= events.decision_date).sum()) if len(events) else 0},
    ]).to_csv(OUT / "p3_rank1_dynamic_self_range_future_PIT_audit.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{"requested_start":"2023-07-11","requested_end":"2025-07-10","actual_start":actions.decision_date.min(),"actual_end":actions.decision_date.max(),"decision_dates":actions.decision_date.nunique(),"rank1_tickers":rank1.ticker.nunique(),"self_range_geometry_platforms":12,"K_range_threshold_candidates":4,"total_bounded_combinations":48}]).to_csv(OUT / "p3_rank1_dynamic_self_range_requested_vs_actual_coverage.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([
        {"item":"stock_buy","brokerage_rate":0.001425,"transaction_tax_rate":0.0,"slippage_bp_per_side":10,"role":"primary"},
        {"item":"stock_sell","brokerage_rate":0.001425,"transaction_tax_rate":0.003,"slippage_bp_per_side":10,"role":"primary"},
        {"item":"slippage_sensitivity","brokerage_rate":np.nan,"transaction_tax_rate":np.nan,"slippage_bp_per_side":5,"role":"secondary"},
        {"item":"slippage_sensitivity","brokerage_rate":np.nan,"transaction_tax_rate":np.nan,"slippage_bp_per_side":20,"role":"secondary"},
        {"item":"00631L_same_coverage_hurdle","brokerage_rate":0.001425,"transaction_tax_rate":0.001,"slippage_bp_per_side":10,"role":"primary_benchmark_EP05_ETF_basis"},
    ]).to_csv(OUT / "p3_rank1_dynamic_self_range_EP05_benchmark_cost_contract.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([
        {"requirement":"corrected_NAV","authority":"same-ticker event-aware adjusted holding returns; official raw execution close","ready":True,"experiments_action":"rechain each frozen platform"},
        {"requirement":"00631L_same_coverage","authority":"same actual dates and EP05 ETF cost basis","ready":True,"experiments_action":"materialize benchmark alongside each platform"},
        {"requirement":"K_range_exclusion_outcomes","authority":"5/10/20/40TD evaluation metadata only","ready":len(exclusions)>0,"experiments_action":"empty-by-construction when exclusion rows=0; do not invent rows"},
        {"requirement":"fold_stability","authority":"P3-1 3-fold validation with 40 decision-date embargo","ready":True,"experiments_action":"at least 2/3 folds positive and median fold excess positive"},
        {"requirement":"P3_2","authority":"mechanically locked until P3-1 pass","ready":False,"experiments_action":"must not read"},
    ]).to_csv(OUT / "p3_rank1_dynamic_self_range_experiments_path_requirement_contract.csv", index=False, encoding="utf-8-sig")

    ready = len(platforms) == 48 and int(platforms.blocked_execution_events.sum()) == 0 and len(events) > 0
    readiness = {"task_id":TASK,"status":"ready_for_experiments_stage_A_event_quality" if ready else "partial_exact_execution_blockers","diagnostic_subproblem":True,"stage_scope":"Layer4_rank1_stock_only_dynamic_self_range_timing","representative_of_full_all80_layer5":False,"may_be_used_to_reject_full_layer5":False,"market_controller_used":False,"all80_rerank":False,"Top3":False,"frozen_self_range_geometry_platform_count":12,"frozen_K_range_thresholds":[0,20,25,30],"total_bounded_combination_count":48,"threshold_expansion_authorized":False,"K_range_gate_entry_only":True,"price_range_pct_audit_only":True,"P3_1_only":True,"P3_2_outcome_read_authorized":False,"candidate_feature_rows":len(p31_features),"decision_dates":actions.decision_date.nunique(),"entry_exit_event_rows":len(events),"K_range_gate_excluded_rows":len(exclusions),"blocked_execution_event_rows":int(platforms.blocked_execution_events.sum()),"ready_for_experiments":ready,"ready_for_formal":False,"ready_for_strategy_replay":False,"formal_model_changed":False,"trade_decision_changed":False,"active_in_trade_decision":False,"report_changed":False,"portfolio_replay_executed":False,"not_live_rule":True,"forward_returns_live_rule_usage":False,"future_data_violation_count":0}
    (OUT / "readiness_for_rank1_dynamic_self_range_stage_A.json").write_text(json.dumps(readiness, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (OUT / "final_summary_zh.md").write_text(f"# Rank1 ticker-specific KD/price range timing Stage A\n\n本包只 materialize canonical Layer4 rank1 的純個股 self-range timing，沒有 market/all80/Top3。固定 12 個 range/zone/latch 幾何平台，另只允許 K range strict-greater-than 0/20/25/30 四個 entry-only gate，共48個 bounded combinations；events={len(events)}，gate exclusions={len(exclusions)}，execution blockers={int(platforms.blocked_execution_events.sum())}，ready_for_experiments={str(ready).lower()}。Price range僅audit；P3-2 outcome未讀。\n", encoding="utf-8")
    files = sorted(p for p in OUT.iterdir() if p.is_file() and p.name != "manifest.json")
    (OUT / "manifest.json").write_text(json.dumps({"task_id":TASK,"files":[{"name":p.name,"sha256":_sha(p),"bytes":p.stat().st_size} for p in files]}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    run()
