from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "outputs" / "vnext_daily_incumbent_challenger_state_machine_contract_20260710"
LAYER4_POOL = REPO_ROOT / "outputs" / "vnext_layer4_80_primary_pool_contract_20260708" / "layer4_80_primary_pool_contract.csv"
MARKET_FIELDS = REPO_ROOT / "outputs" / "vnext_regime_switch_hybrid_route_market_fields_path_materialization_20260708" / "regime_switch_market_regime_fields.csv"
POOL_FIELDS = REPO_ROOT / "outputs" / "vnext_regime_switch_hybrid_route_market_fields_path_materialization_20260708" / "regime_switch_hybrid_route_signal_table.csv"
EXACT_TRIGGER = REPO_ROOT / "outputs" / "vnext_full_period_exact_consensus_trigger_contract_20260708" / "full_period_exact_consensus_trigger_contract.csv"
P1_WEIGHTED = REPO_ROOT / "outputs" / "vnext_p1_c2_weighted_pool80_top5_ohlc_absorption_20260708" / "p1_c2_weighted_pool80_top5_contract_refreshed.csv"
R6_UNIFIED = REPO_ROOT / "outputs" / "vnext_r6_guard_first_market_bias_override_unified_contract_20260709" / "r6_guard_first_market_bias_override_unified_contract.csv"
REVENUE_INTEGRATED = REPO_ROOT / "outputs" / "vnext_revenue_anomaly_integrated_route_support_r6_contract_20260710" / "revenue_anomaly_integrated_route_support_r6_contract.csv"
FULL_STOCK_PATH = REPO_ROOT / "outputs" / "vnext_full_period_regime_switch_benchmark_exception_path_20260708" / "full_period_regime_switch_stock_route_path.csv"
P1_BENCHMARK = REPO_ROOT / "outputs" / "vnext_p1_state_hold_base_exception_path_contract_20260708" / "p1_state_hold_benchmark_path_00631L.csv"
CACHE_00631L = REPO_ROOT / "backtest_cache" / "00631L_TW.csv"

TASK_ID = "TASK-BACKTEST-CORE-VNEXT-DAILY-INCUMBENT-CHALLENGER-STATE-MACHINE-CONTRACT-001"
PRIMARY_TIMING = "signal_day_close_next_trading_day_execution"
# Radar/Data official TWSE absence evidence confirms 2016-07-08 has no 00631L
# or MI_INDEX row. It is not eligible as a daily execution/mark calendar date.
MARKET_CALENDAR_EXCLUDED_DATES = {"2016-07-08"}
PERIODS = {
    "P1": ("2015-01-02", "2022-12-29"),
    "P2": ("2023-01-02", "2026-06-30"),
    "2024_latest": ("2024-01-02", "2026-06-30"),
    "2026YTD": ("2026-01-02", "2026-06-30"),
    "full_integrated": ("2015-01-02", "2026-06-30"),
}
TRANSITION_COSTS = {
    "00631L_to_stock": {"transition_cost_rate": 0.00385, "sell_fee_twd": 1425, "buy_fee_twd": 1425, "securities_transaction_tax_twd": 1000, "total_transition_cost_twd": 3850},
    "stock_to_00631L": {"transition_cost_rate": 0.00585, "sell_fee_twd": 1425, "buy_fee_twd": 1425, "securities_transaction_tax_twd": 3000, "total_transition_cost_twd": 5850},
    "stock_to_stock": {"transition_cost_rate": 0.00585, "sell_fee_twd": 1425, "buy_fee_twd": 1425, "securities_transaction_tax_twd": 3000, "total_transition_cost_twd": 5850},
    "hold": {"transition_cost_rate": 0.0, "sell_fee_twd": 0, "buy_fee_twd": 0, "securities_transaction_tax_twd": 0, "total_transition_cost_twd": 0},
}
VARIANTS = {
    "A_any_positive_edge": {"minimum_edge_pct_points": 0.0, "requires_deterioration": False, "requires_two_day_confirmation": False},
    "B_score_edge_ge_5_pct_points": {"minimum_edge_pct_points": 5.0, "requires_deterioration": False, "requires_two_day_confirmation": False},
    "C_score_edge_ge_10_pct_points": {"minimum_edge_pct_points": 10.0, "requires_deterioration": False, "requires_two_day_confirmation": False},
    "D_score_edge_ge_15_pct_points": {"minimum_edge_pct_points": 15.0, "requires_deterioration": False, "requires_two_day_confirmation": False},
    "E_risk_adjusted_edge_and_incumbent_deterioration": {"minimum_edge_pct_points": 5.0, "requires_deterioration": True, "requires_two_day_confirmation": False},
    "F_two_day_confirmation_and_risk_adjusted_edge": {"minimum_edge_pct_points": 5.0, "requires_deterioration": False, "requires_two_day_confirmation": True},
}
FLAGS = {
    "formal_model_changed": False,
    "trade_decision_changed": False,
    "active_in_trade_decision": False,
    "report_changed": False,
    "portfolio_replay_executed": False,
    "ready_for_strategy_replay": False,
    "ready_for_formal": False,
    "not_live_rule": True,
    "forward_returns_live_rule_usage": False,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ticker(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    return text[:-2] if text.endswith(".0") else text


def _bool(value: Any) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, str):
        return value.lower() in {"true", "1", "yes"}
    return bool(value)


def _num(value: Any, default: float = 0.0) -> float:
    return default if pd.isna(value) else float(value)


def _period_flags(date_text: str) -> dict[str, bool]:
    date = pd.Timestamp(date_text)
    return {f"in_{name}": pd.Timestamp(start) <= date <= pd.Timestamp(end) for name, (start, end) in PERIODS.items() if name != "full_integrated"}


def _period_label(date_text: str) -> str:
    flags = _period_flags(date_text)
    return "|".join(name[3:] for name, active in flags.items() if active) or "outside_requested_periods"


def _rank_pct(group: pd.DataFrame, col: str, higher_better: bool = True) -> pd.Series:
    values = pd.to_numeric(group.get(col), errors="coerce")
    if values.notna().sum() == 0:
        return pd.Series(0.5, index=group.index)
    return values.rank(pct=True, ascending=not higher_better).fillna(0.5)


def _load_market_daily() -> pd.DataFrame:
    market = pd.read_csv(MARKET_FIELDS, low_memory=False)
    market["signal_date"] = pd.to_datetime(market["snapshot_date"], errors="coerce")
    market = market.dropna(subset=["signal_date"]).sort_values("signal_date")
    market = market[(market["signal_date"] >= pd.Timestamp(PERIODS["P1"][0])) & (market["signal_date"] <= pd.Timestamp("2026-06-29"))].copy()
    market = market[~market["signal_date"].dt.strftime("%Y-%m-%d").isin(MARKET_CALENDAR_EXCLUDED_DATES)].copy()
    market["next_trading_day_execution_date"] = market["signal_date"].shift(-1)
    market = market.dropna(subset=["next_trading_day_execution_date"]).copy()
    market["c2_pass_daily"] = (
        (pd.to_numeric(market["0050_price_vs_ma60"], errors="coerce") >= 0)
        & (pd.to_numeric(market["0050_return_20d"], errors="coerce") >= 0)
        & (pd.to_numeric(market["0050_return_40d"], errors="coerce") >= 0)
    )
    return market


def _route_support_counts() -> pd.DataFrame:
    path = pd.read_csv(FULL_STOCK_PATH, low_memory=False, dtype={"ticker": str})
    path = path[path["timing_variant"].eq("next_day_close_entry_fixed_5td_exit")].copy()
    path["snapshot_date"] = pd.to_datetime(path["signal_date"], errors="coerce")
    path["ticker"] = path["ticker"].map(_ticker)
    selected = path[path["route_variant"].isin([
        "hybrid_pullback_base_mega_override", "conservative_hurdle_route", "pool_breadth_route", "market_bias_pool_trend_route", "dispersion_route",
    ])]
    return selected.groupby(["snapshot_date", "ticker"], as_index=False).agg(
        route_support_variant_count=("route_variant", "nunique"),
        route_support_variant_flags=("route_variant", lambda x: "|".join(sorted(set(map(str, x.dropna()))))),
    )


def _weekly_candidate_matrix() -> pd.DataFrame:
    pool = pd.read_csv(LAYER4_POOL, low_memory=False, dtype={"ticker": str})
    pool = pool[pool["is_layer4_primary_pool"].astype(str).str.lower().eq("true")].copy()
    pool["snapshot_date"] = pd.to_datetime(pool["snapshot_date"], errors="coerce")
    pool["ticker"] = pool["ticker"].map(_ticker)
    pool = pool.merge(_route_support_counts(), on=["snapshot_date", "ticker"], how="left")
    pool["route_support_variant_count"] = pd.to_numeric(pool["route_support_variant_count"], errors="coerce").fillna(0)
    pool["route_support_variant_flags"] = pool["route_support_variant_flags"].fillna("")
    pool["quality_component"] = (
        1.0 - pd.to_numeric(pool["layer1_quality_floor_risk_pctile_by_week"], errors="coerce").fillna(0.5)
        + pool["layer1_pass_bottom30"].astype(str).str.lower().eq("true").astype(float) * 0.15
    ).clip(0, 1)
    pool["rs_component"] = pool.groupby("snapshot_date", group_keys=False).apply(
        lambda group: pd.concat([_rank_pct(group, col) for col in ["RS20", "RS40", "RS60", "RS30_proxy"]], axis=1).mean(axis=1),
        include_groups=False,
    ).reindex(pool.index).fillna(0.5)
    liquidity_cols = ["traded_value_rank_5d", "traded_value_rank_20d", "traded_value_rank_60d"]
    pool["liquidity_component"] = pool.apply(
        lambda row: np.mean([1 - np.clip((_num(row.get(col), 40.5) - 1) / 80, 0, 1) for col in liquidity_cols]), axis=1
    )
    bias_cols = ["BIAS20_percentile", "BIAS60_percentile", "BIAS120_percentile"]
    pool["bias_health_component"] = pool.apply(
        lambda row: np.mean([1 - abs(np.clip(_num(row.get(col), 0.5), 0, 1) - 0.5) * 2 for col in bias_cols]), axis=1
    )
    pool["risk_inverse_component"] = (
        1
        - (
            pd.to_numeric(pool["exhaustion_risk_score"], errors="coerce").fillna(0) * 0.25
            + pd.to_numeric(pool["breakdown_risk_score"], errors="coerce").fillna(0) * 0.25
        ).clip(0, 1)
    )
    pool["route_support_component"] = (pool["route_support_variant_count"] / 5.0).clip(0, 1)
    pool["route_support_score_raw"] = (
        pool["quality_component"] * 0.10
        + pool["rs_component"] * 0.20
        + pool["liquidity_component"] * 0.10
        + pool["bias_health_component"] * 0.10
        + pool["route_support_component"] * 0.38
        + pool["risk_inverse_component"] * 0.12
    )
    p1 = pd.read_csv(P1_WEIGHTED, low_memory=False, dtype={"ticker": str})
    p1 = p1[p1["score_variant"].eq("route_support")].copy()
    p1["snapshot_date"] = pd.to_datetime(p1["signal_date"], errors="coerce")
    p1["ticker"] = p1["ticker"].map(_ticker)
    exact = p1[["snapshot_date", "ticker", "weighted_score"]].rename(columns={"weighted_score": "p1_exact_weighted_score"})
    pool = pool.merge(exact, on=["snapshot_date", "ticker"], how="left")
    pool["route_support_score"] = pool["p1_exact_weighted_score"].combine_first(pool["route_support_score_raw"])
    pool["route_support_score_source_quality"] = np.where(
        pool["p1_exact_weighted_score"].notna(), "p1_exact_weighted_pool80_route_support", "reconstructed_weekly_layer4_route_support_component"
    )
    pool["route_support_score_percentile"] = pool.groupby("snapshot_date")["route_support_score"].rank(pct=True) * 100
    pool["incumbent_deterioration_signal_count"] = (
        pool["rs_short_deterioration_flag"].astype(str).str.lower().eq("true").astype(int)
        + pool["risk_overheat_penalty_context"].astype(str).str.lower().eq("true").astype(int)
        + pool["volatility_high_context"].astype(str).str.lower().eq("true").astype(int)
        + pool["rs60_high_short_rs_weakening_exhaustion_context"].astype(str).str.lower().eq("true").astype(int)
        + pool["high_exhaustion_or_breakdown_context"].astype(str).str.lower().eq("true").astype(int)
    )
    pool["incumbent_deterioration_confirmed"] = pool["incumbent_deterioration_signal_count"] >= 2
    return pool


def _weekly_context(matrix: pd.DataFrame) -> pd.DataFrame:
    weekly = matrix[["snapshot_date"]].drop_duplicates().sort_values("snapshot_date").rename(columns={"snapshot_date": "pool_snapshot_date"}).copy()
    exact = pd.read_csv(EXACT_TRIGGER, low_memory=False, dtype={"candidate_ticker": str})
    exact["snapshot_date"] = pd.to_datetime(exact["signal_date"], errors="coerce")
    passes = exact[exact["exact_trigger_pass"].astype(str).str.lower().eq("true")].groupby("snapshot_date", as_index=False).agg(
        consensus_trigger_weekly=("exact_trigger_pass", "size"),
        consensus_candidate_count=("candidate_ticker", "nunique"),
        consensus_max_count=("consensus_count", "max"),
    )
    passes["consensus_trigger_weekly"] = True
    passes = passes.rename(columns={"snapshot_date": "pool_snapshot_date"})
    weekly = weekly.merge(passes, on="pool_snapshot_date", how="left")
    weekly["consensus_trigger_weekly"] = weekly["consensus_trigger_weekly"].fillna(False).astype(bool)
    weekly["consensus_candidate_count"] = weekly["consensus_candidate_count"].fillna(0).astype(int)
    weekly["consensus_max_count"] = weekly["consensus_max_count"].fillna(0).astype(int)
    tops = matrix.sort_values(["snapshot_date", "route_support_score", "ticker"], ascending=[True, False, True]).drop_duplicates("snapshot_date")
    weekly = weekly.merge(
        tops[["snapshot_date", "ticker", "name", "route_support_score", "route_support_score_percentile", "route_support_score_source_quality"]]
        .rename(columns={"snapshot_date": "pool_snapshot_date"})
        .rename(columns={"ticker": "route_support_challenger_ticker", "name": "route_support_challenger_name", "route_support_score": "route_support_challenger_score", "route_support_score_percentile": "route_support_challenger_score_percentile"}),
        on="pool_snapshot_date", how="left",
    )
    pool_fields = pd.read_csv(POOL_FIELDS, low_memory=False)
    pool_fields["snapshot_date"] = pd.to_datetime(pool_fields["snapshot_date"], errors="coerce")
    pool_cols = ["dynamic80_rs20_positive_share", "dynamic80_rs60_positive_share", "pool_high_exhaustion_breakdown_share"]
    weekly_pool = pool_fields.groupby("snapshot_date", as_index=False)[pool_cols].first().rename(columns={"snapshot_date": "pool_snapshot_date"})
    weekly = weekly.merge(weekly_pool, on="pool_snapshot_date", how="left")
    r6 = pd.read_csv(R6_UNIFIED, low_memory=False, dtype={"selected_ticker": str})
    r6["snapshot_date"] = pd.to_datetime(r6["signal_date"], errors="coerce")
    r6["selected_ticker"] = r6["selected_ticker"].map(_ticker)
    r6 = r6[
        r6["r6_override_flag"].astype(str).str.lower().eq("true")
        & r6["selected_asset_type"].eq("stock")
    ].copy()
    weekly = weekly.merge(
        r6[["snapshot_date", "selected_ticker", "selected_ticker_name"]].rename(columns={"snapshot_date": "pool_snapshot_date", "selected_ticker": "r6_market_bias_candidate_ticker", "selected_ticker_name": "r6_market_bias_candidate_name"}),
        on="pool_snapshot_date", how="left",
    )
    return weekly.sort_values("pool_snapshot_date")


def _revenue_map() -> pd.DataFrame:
    revenue = pd.read_csv(REVENUE_INTEGRATED, low_memory=False, dtype={"selected_ticker": str})
    revenue["snapshot_date"] = pd.to_datetime(revenue["signal_date"], errors="coerce")
    revenue["ticker"] = revenue["selected_ticker"].map(_ticker)
    cols = ["snapshot_date", "ticker", "abnormal_revenue_review_flag", "revenue_anomaly_penalty_score", "revenue_hygiene_confidence_level", "report_revenue_anomaly_warning", "report_revenue_anomaly_reason"]
    return revenue[revenue["selected_primary_asset_type_for_anomaly"].eq("stock")][cols].drop_duplicates(["snapshot_date", "ticker"])


def _transition(previous_ticker: str, previous_type: str, target_ticker: str, target_type: str) -> tuple[str, str]:
    if previous_ticker == target_ticker and previous_type == target_type:
        return "hold_same", "hold"
    if previous_type == "etf" and target_type == "stock":
        return "base_to_stock", "00631L_to_stock"
    if previous_type == "stock" and target_type == "etf":
        return "stock_to_base", "stock_to_00631L"
    return "stock_to_stock", "stock_to_stock"


def _date_text(value: Any) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%d") if pd.notna(value) else ""


def _daily_state_rows() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    matrix = _weekly_candidate_matrix()
    weekly = _weekly_context(matrix)
    revenue = _revenue_map()
    market = _load_market_daily()
    aligned = pd.merge_asof(
        market.sort_values("signal_date"), weekly.sort_values("pool_snapshot_date"), left_on="signal_date", right_on="pool_snapshot_date", direction="backward"
    )
    matrix_key = {(r.snapshot_date, r.ticker): r._asdict() for r in matrix.itertuples(index=False)}
    revenue_key = {(r.snapshot_date, r.ticker): r._asdict() for r in revenue.itertuples(index=False)}
    rows: list[dict[str, Any]] = []
    transitions: list[dict[str, Any]] = []
    for variant, policy in VARIANTS.items():
        incumbent_ticker, incumbent_type = "00631L", "etf"
        previous_challenger = ""
        candidate_streak = 0
        for day in aligned.itertuples(index=False):
            snapshot = getattr(day, "pool_snapshot_date", pd.NaT)
            c2 = _bool(getattr(day, "c2_pass_daily", False))
            consensus = _bool(getattr(day, "consensus_trigger_weekly", False))
            breakout_breadth = (
                _num(getattr(day, "rolling_high_breakout_count", 0)) >= 1
                and _num(getattr(day, "0050_return_20d", 0)) >= 0.03
                and _num(getattr(day, "dynamic80_rs20_positive_share", 0)) >= 0.70
            )
            p1_risk_veto = (
                _num(getattr(day, "0050_return_60d", 0)) < 0
                or _num(getattr(day, "dynamic80_rs60_positive_share", 0)) < 0.45
                or _num(getattr(day, "00631L_vs_0050_return_20d", 0)) < -0.05
                or _num(getattr(day, "pool_high_exhaustion_breakdown_share", 0)) > 0.55
            )
            r6_daily = breakout_breadth and not p1_risk_veto
            route_ticker = _ticker(getattr(day, "route_support_challenger_ticker", ""))
            r6_ticker = _ticker(getattr(day, "r6_market_bias_candidate_ticker", ""))
            use_r6_candidate = c2 and r6_daily and bool(r6_ticker)
            candidate_ticker = r6_ticker if use_r6_candidate else route_ticker
            candidate_branch = "market_bias_override" if use_r6_candidate else "route_support"
            stock_allowed = c2 and (use_r6_candidate or consensus) and bool(candidate_ticker)
            if stock_allowed and candidate_ticker == previous_challenger:
                candidate_streak += 1
            elif stock_allowed:
                candidate_streak = 1
            else:
                candidate_streak = 0
            challenger = matrix_key.get((snapshot, candidate_ticker), {}) if pd.notna(snapshot) else {}
            incumbent = matrix_key.get((snapshot, incumbent_ticker), {}) if incumbent_type == "stock" and pd.notna(snapshot) else {}
            challenger_score = _num(challenger.get("route_support_score_percentile"), 0.0)
            incumbent_score = _num(incumbent.get("route_support_score_percentile"), 0.0) if incumbent_type == "stock" else 0.0
            score_edge = challenger_score - incumbent_score
            incumbent_in_80 = incumbent_type == "etf" or bool(incumbent)
            incumbent_deterioration = bool(incumbent and incumbent.get("incumbent_deterioration_confirmed", False)) or (incumbent_type == "stock" and not incumbent_in_80)
            eligible_switch = score_edge >= policy["minimum_edge_pct_points"]
            if policy["requires_deterioration"]:
                eligible_switch = eligible_switch and incumbent_deterioration
            if policy["requires_two_day_confirmation"]:
                eligible_switch = eligible_switch and candidate_streak >= 2
            if not stock_allowed:
                target_ticker, target_type = "00631L", "etf"
                decision_reason = "daily_c2_failed_or_weekly_consensus_not_active_return_to_00631L_base"
            elif incumbent_type == "etf":
                target_ticker, target_type = candidate_ticker, "stock"
                decision_reason = "base_to_stock_c2_and_candidate_context_active"
            elif incumbent_ticker == candidate_ticker:
                target_ticker, target_type = incumbent_ticker, "stock"
                decision_reason = "hold_same_incumbent_challenger_unchanged"
            elif eligible_switch:
                target_ticker, target_type = candidate_ticker, "stock"
                decision_reason = "challenger_edge_threshold_passed"
            else:
                target_ticker, target_type = incumbent_ticker, "stock"
                decision_reason = "keep_incumbent_challenger_threshold_not_passed"
            action, cost_key = _transition(incumbent_ticker, incumbent_type, target_ticker, target_type)
            cost = TRANSITION_COSTS[cost_key]
            rev = revenue_key.get((snapshot, target_ticker), {}) if target_type == "stock" and pd.notna(snapshot) else {}
            row = {
                "task": TASK_ID,
                "state_machine_variant": variant,
                "signal_date": _date_text(day.signal_date),
                "next_trading_day_execution_date": _date_text(day.next_trading_day_execution_date),
                "pool_snapshot_date": _date_text(snapshot),
                "pool_context_update_frequency": "weekly_last_trading_day_forward_filled_to_daily_signal",
                "market_context_update_frequency": "daily_close_recomputed_from_0050_market_fields",
                "period_label": _period_label(_date_text(day.signal_date)),
                **_period_flags(_date_text(day.signal_date)),
                "c2_pass_flag": c2,
                "c2_definition": "0050_price_vs_ma60>=0 AND 0050_return_20d>=0 AND 0050_return_40d>=0",
                "consensus_trigger_flag": consensus,
                "consensus_trigger_source_quality": "exact_consensus_trigger_weekly_asof_pool_snapshot_forward_filled",
                "r6_override_flag": r6_daily,
                "r6_override_candidate_available": bool(r6_ticker),
                "p1_risk_veto_flag": p1_risk_veto,
                "selected_branch_candidate": candidate_branch,
                "incumbent_ticker_before": incumbent_ticker,
                "incumbent_asset_type_before": incumbent_type,
                "incumbent_still_in_80_flag": incumbent_in_80,
                "incumbent_score_percentile": incumbent_score,
                "incumbent_deterioration_confirmed": incumbent_deterioration,
                "challenger_ticker": candidate_ticker,
                "challenger_name": challenger.get("name", getattr(day, "route_support_challenger_name", "")),
                "challenger_score_percentile": challenger_score,
                "challenger_score_source_quality": challenger.get("route_support_score_source_quality", "weekly_candidate_score_missing"),
                "challenger_score_edge_pct_points": score_edge,
                "challenger_confirmation_days": candidate_streak,
                "stock_candidate_allowed": stock_allowed,
                "switch_threshold_pct_points": policy["minimum_edge_pct_points"],
                "requires_incumbent_deterioration": policy["requires_deterioration"],
                "requires_two_day_confirmation": policy["requires_two_day_confirmation"],
                "selected_ticker_after": target_ticker,
                "selected_asset_type_after": target_type,
                "decision_reason": decision_reason,
                "transition_type": action,
                "transition_cost_rate_hook": cost["transition_cost_rate"],
                "transition_cost_model_status": "EP05_TaiwanCostModel_unit_notional_hook_stock_etf_separated",
                "stock_bias20_percentile_weekly_asof": challenger.get("BIAS20_percentile", np.nan) if target_type == "stock" else np.nan,
                "stock_bias60_percentile_weekly_asof": challenger.get("BIAS60_percentile", np.nan) if target_type == "stock" else np.nan,
                "stock_bias120_percentile_weekly_asof": challenger.get("BIAS120_percentile", np.nan) if target_type == "stock" else np.nan,
                "stock_volatility_percentile_weekly_asof": challenger.get("volatility_pctile_by_week", np.nan) if target_type == "stock" else np.nan,
                "stock_layer1_quality_risk_percentile_weekly_asof": challenger.get("layer1_quality_floor_risk_pctile_by_week", np.nan) if target_type == "stock" else np.nan,
                "stock_route_support_variant_count_weekly_asof": challenger.get("route_support_variant_count", np.nan) if target_type == "stock" else np.nan,
                "stock_risk_penalty_weekly_asof": challenger.get("layer4_risk_penalty_score", np.nan) if target_type == "stock" else np.nan,
                "revenue_anomaly_warning": rev.get("report_revenue_anomaly_warning", "historical_candidate_revenue_anomaly_not_materialized") if target_type == "stock" else "not_applicable_00631L_base",
                "revenue_hygiene_confidence_level": rev.get("revenue_hygiene_confidence_level", "historical_candidate_revenue_anomaly_not_materialized") if target_type == "stock" else "not_applicable_00631L_base",
                "revenue_anomaly_used_for_selection": False,
                "rs20_top3_reference_only": True,
                "cash_bear_classifier_status": "blocked_no_accepted_cash_bear_classifier",
                "selected_stock_adjusted_close_ready": False if target_type == "stock" else True,
                "official_unadjusted_daily_ohlc_ready": False if target_type == "stock" else True,
                "daily_stock_bias_recalculated": False,
                "data_readiness": "partial_weekly_candidate_context_daily_stock_path_not_materialized" if target_type == "stock" else "benchmark_base_daily_path_hook",
                "diagnostic_only": True,
                **FLAGS,
            }
            rows.append(row)
            if action != "hold_same":
                transitions.append({
                    "state_machine_variant": variant,
                    "signal_date": row["signal_date"],
                    "execution_date": row["next_trading_day_execution_date"],
                    "from_ticker": incumbent_ticker,
                    "from_asset_type": incumbent_type,
                    "to_ticker": target_ticker,
                    "to_asset_type": target_type,
                    "transition_type": action,
                    **cost,
                    "cost_model_status": row["transition_cost_model_status"],
                    "diagnostic_only": True,
                    **FLAGS,
                })
            incumbent_ticker, incumbent_type = target_ticker, target_type
            previous_challenger = candidate_ticker if stock_allowed else ""
    state = pd.DataFrame(rows)
    transition = pd.DataFrame(transitions)
    state = state.sort_values(["state_machine_variant", "signal_date"]).reset_index(drop=True)
    state["next_trading_day_after_execution_date"] = state.groupby("state_machine_variant")["next_trading_day_execution_date"].shift(-1)
    price_need = state[state["selected_asset_type_after"].eq("stock")][[
        "state_machine_variant", "signal_date", "next_trading_day_execution_date", "next_trading_day_after_execution_date", "selected_ticker_after", "transition_type", "pool_snapshot_date"
    ]].rename(columns={"selected_ticker_after": "ticker", "next_trading_day_execution_date": "entry_date", "next_trading_day_after_execution_date": "exit_date"})
    price_need = (
        price_need.groupby(["ticker", "signal_date", "entry_date", "exit_date", "transition_type", "pool_snapshot_date"], as_index=False)
        .agg(impacted_state_machine_variants=("state_machine_variant", lambda values: "|".join(sorted(set(values)))))
    )
    price_need["required_fields"] = "entry_close;next_daily_close_for_state_return"
    price_need["source_requirement"] = "official_selected_ticker_daily_unadjusted_ohlc_only"
    price_need["adjusted_close_ready"] = False
    price_need["path_status"] = "blocked_not_materialized_in_current_weekly_5td_selected_path_packages"
    price_need["next_owner"] = "Radar/Data bounded selected-ticker daily OHLC gap fill"
    return state, transition, price_need


def _coverage(state: pd.DataFrame, price_need: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, (start, end) in PERIODS.items():
        sub = state[(pd.to_datetime(state["signal_date"]) >= pd.Timestamp(start)) & (pd.to_datetime(state["signal_date"]) <= pd.Timestamp(end))]
        needs = price_need[(pd.to_datetime(price_need["signal_date"]) >= pd.Timestamp(start)) & (pd.to_datetime(price_need["signal_date"]) <= pd.Timestamp(end))]
        rows.append({
            "period": name,
            "requested_start": start,
            "requested_end": end,
            "actual_start": sub["signal_date"].min() if len(sub) else "",
            "actual_end": sub["signal_date"].max() if len(sub) else "",
            "daily_state_rows": int(len(sub)),
            "unique_daily_signal_dates": int(sub["signal_date"].nunique()),
            "stock_daily_price_requirement_rows": int(len(needs)),
            "stock_daily_price_requirement_tickers": int(needs["ticker"].nunique()),
            "weekly_pool_context_source": "Layer4 primary80 weekly snapshot forward-filled",
            "market_context_source": "0050 daily market regime fields",
            **FLAGS,
        })
    return pd.DataFrame(rows)


def _policy() -> pd.DataFrame:
    rows = []
    for variant, policy in VARIANTS.items():
        rows.append({
            "state_machine_variant": variant,
            **policy,
            "base_to_stock_policy": "requires daily C2 and either weekly exact consensus trigger or R6 override candidate; score edge thresholds govern stock-to-stock switches",
            "stock_to_base_policy": "daily C2 false or weekly exact consensus trigger inactive when no R6 stock override; return to 00631L, never cash",
            "same_ticker_policy": "hold same ticker; no weekly sell/rebuy",
            "future_return_used_as_rule": False,
            "revenue_anomaly_used_for_selection": False,
            "rs20_top3_role": "reference_only",
            **FLAGS,
        })
    return pd.DataFrame(rows)


def _blocked(state: pd.DataFrame, price_need: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([
        {"item": "selected_stock_daily_official_unadjusted_ohlc", "status": "blocked_pending_bounded_gap_fill", "count": int(len(price_need)), "detail": "daily state return path requires selected-ticker entry/next-close rows; no weekly 5TD proxy substitution", "next_owner": "Radar/Data"},
        {"item": "selected_stock_adjusted_close", "status": "blocked", "count": int((state["selected_asset_type_after"] == "stock").sum()), "detail": "unadjusted OHLC diagnostic path only; adjusted close not fabricated", "next_owner": "Strategy Center/Radar Data only if trusted source route authorized"},
        {"item": "stock_candidate_context_frequency", "status": "weekly_asof_proxy", "count": int((state["selected_asset_type_after"] == "stock").sum()), "detail": "Layer4/quality/BIAS/risk/route_support use latest weekly pool snapshot; only 0050 market C2/R6 fields are daily recomputed", "next_owner": "Core/Data if daily Layer0-4 source materialization is later authorized"},
        {"item": "historical_candidate_revenue_anomaly", "status": "partial", "count": int((state["revenue_hygiene_confidence_level"] == "historical_candidate_revenue_anomaly_not_materialized").sum()), "detail": "warning/confidence only; no rerank and no hard exclude", "next_owner": "Core/Data optional hygiene expansion"},
        {"item": "cash_bear_classifier", "status": "blocked", "count": 0, "detail": "no cash rule created; gate failure returns to 00631L only", "next_owner": "Strategy Center/Core Data later"},
    ])


def _future_audit() -> pd.DataFrame:
    return pd.DataFrame([
        {"audit_item": "daily_C2", "future_return_used_as_rule": False, "source": "same-day 0050 PIT close and trailing fields", "future_data_violation_count": 0},
        {"audit_item": "weekly_candidate_context", "future_return_used_as_rule": False, "source": "latest Layer4 weekly snapshot forward-filled only", "future_data_violation_count": 0},
        {"audit_item": "execution_path", "future_return_used_as_rule": False, "source": "next trading day prices requested solely for later diagnostic evaluation", "future_data_violation_count": 0},
    ])


def _readiness(state: pd.DataFrame, price_need: pd.DataFrame, coverage: pd.DataFrame) -> dict[str, Any]:
    stock_rows = int((state["selected_asset_type_after"] == "stock").sum())
    return {
        "task_id": TASK_ID,
        "status": "daily_candidate_contract_ready_price_path_gap_fill_required",
        "daily_state_rows": int(len(state)),
        "daily_signal_dates": int(state["signal_date"].nunique()),
        "weekly_pool_snapshot_count": int(state["pool_snapshot_date"].nunique()),
        "state_machine_variants": list(VARIANTS.keys()),
        "stock_selected_daily_rows": stock_rows,
        "stock_daily_ohlc_gap_rows": int(len(price_need)),
        "stock_daily_ohlc_gap_tickers": int(price_need["ticker"].nunique()),
        "market_daily_c2_ready": True,
        "weekly_layer0_to_layer4_context_ready": True,
        "official_selected_stock_daily_ohlc_ready_share": 0.0,
        "selected_stock_adjusted_close_ready": False,
        "EP05_transaction_cost_hooks_ready": True,
        "revenue_anomaly_rerank_applied": False,
        "rs20_top3_reference_only": True,
        "cash_bear_classifier_ready": False,
        "ready_for_experiments": False,
        "ready_for_daily_incumbent_challenger_state_machine_diagnostic": False,
        "ready_for_radar_data_gap_fill": bool(len(price_need)),
        "ready_for_formal": False,
        "ready_for_strategy_replay": False,
        "future_data_violation_count": 0,
        "coverage_by_period": coverage.to_dict(orient="records"),
        **FLAGS,
    }


def _write_summary(path: Path, readiness: dict[str, Any]) -> None:
    path.write_text("\n".join([
        "# Daily incumbent/challenger state-machine candidate contract",
        "",
        "## 結論",
        "",
        "- 已建立 daily close signal / next-trading-day execution 的 incumbent-challenger candidate contract。",
        "- 0050 market C2/R6 欄位每日重算；Layer0-4、route_support、品質與個股風險欄位僅使用最近週池快照並明確 forward-fill。",
        "- 已 materialize A-F 六種換倉門檻；同 ticker 續抱，不會每週賣出再買回。",
        "- gate 失效只切回 00631L base；未建立 cash rule。Revenue anomaly 只保留 warning/confidence，不改選股。",
        f"- daily state rows={readiness['daily_state_rows']}；selected-stock daily OHLC gap rows={readiness['stock_daily_ohlc_gap_rows']}。",
        "- 因 selected-stock 日頻 official OHLC 尚未 materialize，未交 Experiments；需先做 bounded selected-ticker-only gap fill。",
        "- adjusted close blocked；日頻 candidate context 對個股仍為 weekly-asof proxy；不升 formal/replay/report/trade decision。",
        "",
        "下一棒：交 Radar/Data 執行 TASK-RADAR-DATA-VNEXT-DAILY-INCUMBENT-CHALLENGER-SELECTED-STOCK-DAILY-OHLC-GAP-FILL-001。",
        "",
        "完成後如果下一棒明確，請直接指派下一個 thread；如果下一棒不明確，請回報 Strategy Center 判斷。不要完成後停住不回報。",
        "",
        "Flags: formal_model_changed=false; trade_decision_changed=false; active_in_trade_decision=false; report_changed=false; portfolio_replay_executed=false; ready_for_strategy_replay=false; ready_for_formal=false; not_live_rule=true; forward_returns_live_rule_usage=false.",
    ]), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    state, transitions, price_need = _daily_state_rows()
    coverage = _coverage(state, price_need)
    policy = _policy()
    blocked = _blocked(state, price_need)
    future = _future_audit()
    readiness = _readiness(state, price_need, coverage)
    paths = {
        "state": OUTPUT_DIR / "daily_incumbent_challenger_state_rows.csv",
        "comparison": OUTPUT_DIR / "daily_incumbent_challenger_comparison.csv",
        "transitions": OUTPUT_DIR / "daily_incumbent_challenger_transition_trace.csv",
        "policy": OUTPUT_DIR / "daily_incumbent_challenger_threshold_variant_design.csv",
        "price_gap": OUTPUT_DIR / "daily_incumbent_challenger_selected_stock_daily_ohlc_gap_ledger.csv",
        "cost": OUTPUT_DIR / "daily_incumbent_challenger_ep05_cost_hooks.csv",
        "coverage": OUTPUT_DIR / "daily_incumbent_challenger_requested_vs_actual_coverage.csv",
        "blocked": OUTPUT_DIR / "daily_incumbent_challenger_blocked_proxy_audit.csv",
        "future": OUTPUT_DIR / "daily_incumbent_challenger_future_data_audit.csv",
        "readiness": OUTPUT_DIR / "readiness_for_daily_incumbent_challenger_state_machine_diagnostic.json",
        "summary": OUTPUT_DIR / "final_summary_zh.md",
        "manifest": OUTPUT_DIR / "manifest.json",
    }
    state.to_csv(paths["state"], index=False, encoding="utf-8-sig")
    comparison_cols = [c for c in ["state_machine_variant", "signal_date", "pool_snapshot_date", "incumbent_ticker_before", "incumbent_score_percentile", "incumbent_deterioration_confirmed", "challenger_ticker", "challenger_score_percentile", "challenger_score_edge_pct_points", "challenger_confirmation_days", "selected_ticker_after", "decision_reason", "transition_type"] if c in state.columns]
    state[comparison_cols].to_csv(paths["comparison"], index=False, encoding="utf-8-sig")
    transitions.to_csv(paths["transitions"], index=False, encoding="utf-8-sig")
    policy.to_csv(paths["policy"], index=False, encoding="utf-8-sig")
    price_need.to_csv(paths["price_gap"], index=False, encoding="utf-8-sig")
    pd.DataFrame([{"transition_cost_key": key, **value, "cost_model_status": "EP05_TaiwanCostModel_unit_notional_hook", "diagnostic_only": True, **FLAGS} for key, value in TRANSITION_COSTS.items()]).to_csv(paths["cost"], index=False, encoding="utf-8-sig")
    coverage.to_csv(paths["coverage"], index=False, encoding="utf-8-sig")
    blocked.to_csv(paths["blocked"], index=False, encoding="utf-8-sig")
    future.to_csv(paths["future"], index=False, encoding="utf-8-sig")
    paths["readiness"].write_text(json.dumps(readiness, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_summary(paths["summary"], readiness)
    manifest = {
        "task_id": TASK_ID,
        "output_dir": str(OUTPUT_DIR),
        "inputs": {"layer4_pool": str(LAYER4_POOL), "market_fields": str(MARKET_FIELDS), "exact_trigger": str(EXACT_TRIGGER), "r6_unified": str(R6_UNIFIED), "revenue_integrated": str(REVENUE_INTEGRATED)},
        "artifacts": [{"path": str(path), "sha256": _sha256(path), "bytes": path.stat().st_size} for key, path in paths.items() if key != "manifest"],
        "readiness": readiness,
        "created_at": datetime.now(timezone.utc).isoformat(),
        **FLAGS,
    }
    paths["manifest"].write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(readiness, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
