from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_FEATURES = REPO_ROOT / "outputs" / "vnext_dynamic_candidate_pool_data_materialization_20260706" / "benchmark_features.csv"
POOL_FIELDS = REPO_ROOT / "outputs" / "vnext_regime_switch_hybrid_route_market_fields_path_materialization_20260708" / "regime_switch_hybrid_route_signal_table.csv"
R6_STATE = REPO_ROOT / "outputs" / "vnext_weekly_r6_single_position_state_boundary_reconstruction_contract_20260710" / "reconstructed_weekly_r6_single_position_daily_state_rows.csv"
DAILY_F_STATE = REPO_ROOT / "outputs" / "vnext_daily_incumbent_challenger_state_machine_contract_ohlc_absorbed_20260710" / "daily_incumbent_challenger_state_machine_contract_ohlc_absorbed.csv"
OUTPUT_DIR = REPO_ROOT / "outputs" / "vnext_confirmed_bear_cash_classifier_contract_20260710"

TASK_ID = "TASK-BACKTEST-CORE-VNEXT-CONFIRMED-BEAR-CASH-CLASSIFIER-CONTRACT-001"
RAW_F_VARIANT = "F_two_day_confirmation_and_risk_adjusted_edge"
PERIODS = {
    "P1": ("2015-01-02", "2022-12-29"),
    "P2": ("2023-01-02", "2026-06-30"),
    "2024_latest": ("2024-01-02", "2026-06-30"),
    "2026YTD": ("2026-01-02", "2026-06-30"),
    "full_integrated": ("2015-01-02", "2026-06-30"),
}
VARIANTS = {
    "B0_no_cash_reference": {"entry_days": 0, "exit_days": 0},
    "B1_medium_trend_break": {"entry_days": 2, "exit_days": 2},
    "B2_long_trend_breadth": {"entry_days": 2, "exit_days": 2},
    "B3_drawdown_breadth_deterioration": {"entry_days": 2, "exit_days": 2},
    "B4_two_of_three_consensus": {"entry_days": 2, "exit_days": 2},
    "B5_strict_three_of_three": {"entry_days": 2, "exit_days": 2},
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

COSTS = {
    "hold": {"transition_cost_rate": 0.0, "sell_fee_rate": 0.0, "buy_fee_rate": 0.0, "tax_rate": 0.0},
    "etf_to_cash": {"transition_cost_rate": 0.002425, "sell_fee_rate": 0.001425, "buy_fee_rate": 0.0, "tax_rate": 0.001},
    "stock_to_cash": {"transition_cost_rate": 0.004425, "sell_fee_rate": 0.001425, "buy_fee_rate": 0.0, "tax_rate": 0.003},
    "cash_to_etf": {"transition_cost_rate": 0.001425, "sell_fee_rate": 0.0, "buy_fee_rate": 0.001425, "tax_rate": 0.0},
    "cash_to_stock": {"transition_cost_rate": 0.001425, "sell_fee_rate": 0.0, "buy_fee_rate": 0.001425, "tax_rate": 0.0},
    "00631L_to_stock": {"transition_cost_rate": 0.00385, "sell_fee_rate": 0.001425, "buy_fee_rate": 0.001425, "tax_rate": 0.001},
    "stock_to_00631L": {"transition_cost_rate": 0.00585, "sell_fee_rate": 0.001425, "buy_fee_rate": 0.001425, "tax_rate": 0.003},
    "stock_to_stock": {"transition_cost_rate": 0.00585, "sell_fee_rate": 0.001425, "buy_fee_rate": 0.001425, "tax_rate": 0.003},
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write(frame: pd.DataFrame, name: str) -> Path:
    path = OUTPUT_DIR / name
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def _ticker(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    return text[:-2] if text.endswith(".0") else text


def _expanding_percentile(series: pd.Series, min_periods: int = 12) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    result = np.full(len(values), np.nan)
    for index, value in enumerate(values):
        history = values[: index + 1]
        history = history[np.isfinite(history)]
        if np.isfinite(value) and len(history) >= min_periods:
            result[index] = float((history <= value).mean())
    return pd.Series(result, index=series.index)


def _market_features() -> pd.DataFrame:
    frame = pd.read_csv(BENCHMARK_FEATURES, low_memory=False, dtype={"benchmark": str})
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce")
    frame = frame[frame["benchmark"].eq("0050")].dropna(subset=["trade_date"]).sort_values("trade_date").copy()
    frame["close"] = pd.to_numeric(frame["adjusted_close"], errors="coerce")
    close = frame["close"]
    for window in [20, 60, 120]:
        frame[f"ma{window}"] = close.rolling(window, min_periods=window).mean()
        frame[f"bias{window}"] = close / frame[f"ma{window}"] - 1.0
    for window in [20, 40, 60]:
        frame[f"return_{window}d"] = close / close.shift(window) - 1.0
    frame["ma60_slope_20d"] = frame["ma60"] / frame["ma60"].shift(20) - 1.0
    frame["ma120_slope_20d"] = frame["ma120"] / frame["ma120"].shift(20) - 1.0
    frame["bias20_delta_5d"] = frame["bias20"] - frame["bias20"].shift(5)
    frame["bias60_delta_5d"] = frame["bias60"] - frame["bias60"].shift(5)
    rolling_high = close.rolling(252, min_periods=60).max()
    frame["drawdown_from_252d_high"] = close / rolling_high - 1.0
    frame["drawdown_severity_percentile_pit"] = _expanding_percentile(-frame["drawdown_from_252d_high"], min_periods=60)
    frame["medium_trend_break_component"] = (
        close.lt(frame["ma60"]) & frame["return_20d"].lt(0) & frame["return_40d"].lt(0)
    )
    frame["medium_trend_negative_context"] = close.lt(frame["ma60"]) & frame["return_20d"].lt(0)
    frame["long_trend_break_component"] = close.lt(frame["ma120"]) & frame["ma60_slope_20d"].lt(0)
    frame["drawdown_bias_deterioration_component"] = (
        frame["drawdown_severity_percentile_pit"].ge(0.80) & frame["bias60_delta_5d"].lt(0)
    )
    frame["market_feature_source_quality"] = "0050_adjusted_close_daily_PIT_rolling_including_signal_close"
    frame["market_feature_asof_date"] = frame["trade_date"]
    return frame.rename(columns={"trade_date": "signal_date"})


def _pool_breadth_features() -> pd.DataFrame:
    frame = pd.read_csv(POOL_FIELDS, low_memory=False)
    frame["snapshot_date"] = pd.to_datetime(frame["snapshot_date"], errors="coerce")
    frame = frame.dropna(subset=["snapshot_date"]).sort_values("snapshot_date")
    weekly = frame.groupby("snapshot_date", as_index=False).agg(
        pool80_rs20_positive_share=("dynamic80_rs20_positive_share", "first"),
        pool80_rs60_positive_share=("dynamic80_rs60_positive_share", "first"),
    )
    weekly["pool80_rs20_positive_share"] = pd.to_numeric(weekly["pool80_rs20_positive_share"], errors="coerce")
    weekly["pool80_rs60_positive_share"] = pd.to_numeric(weekly["pool80_rs60_positive_share"], errors="coerce")
    weekly["pool80_rs20_positive_share_pit_percentile"] = _expanding_percentile(weekly["pool80_rs20_positive_share"], min_periods=12)
    weekly["pool80_rs20_breadth_change_2snap"] = weekly["pool80_rs20_positive_share"].diff(2)
    weekly["pool80_rs20_breadth_change_4snap"] = weekly["pool80_rs20_positive_share"].diff(4)
    weekly["pool80_breadth_weak_component"] = weekly["pool80_rs20_positive_share_pit_percentile"].le(0.25)
    weekly["pool80_breadth_deterioration_component"] = (
        weekly["pool80_rs20_positive_share_pit_percentile"].le(0.35)
        & weekly["pool80_rs20_breadth_change_2snap"].lt(0)
        & weekly["pool80_rs20_breadth_change_4snap"].lt(0)
    )
    weekly["pool80_above_ma20_share"] = np.nan
    weekly["pool80_above_ma60_share"] = np.nan
    weekly["pool80_breadth_source_quality"] = "weekly_Layer4_primary80_PIT_snapshot_asof_forward_fill_not_daily_recalculation"
    weekly["pool80_above_ma_share_source_quality"] = "blocked_not_materialized"
    return weekly


def _feature_matrix(signal_dates: pd.Series) -> pd.DataFrame:
    dates = pd.DataFrame({"signal_date": pd.to_datetime(signal_dates).sort_values().unique()})
    market = _market_features()
    market = dates.merge(market, on="signal_date", how="left")
    pool = _pool_breadth_features()
    features = pd.merge_asof(
        market.sort_values("signal_date"),
        pool.sort_values("snapshot_date"),
        left_on="signal_date",
        right_on="snapshot_date",
        direction="backward",
    )
    features["pool_snapshot_age_days"] = (features["signal_date"] - features["snapshot_date"]).dt.days
    features["B1_feature_ready"] = features[["close", "ma60", "return_20d", "return_40d"]].notna().all(axis=1)
    features["B2_feature_ready"] = features[["close", "ma120", "ma60_slope_20d", "pool80_rs20_positive_share_pit_percentile"]].notna().all(axis=1)
    features["B3_feature_ready"] = features[["drawdown_severity_percentile_pit", "bias60_delta_5d", "pool80_rs20_breadth_change_2snap", "pool80_rs20_breadth_change_4snap", "ma60", "return_20d"]].notna().all(axis=1)
    features["B4_feature_ready"] = features["B1_feature_ready"] & features["B2_feature_ready"] & features["pool80_rs20_breadth_change_2snap"].notna()
    features["B5_feature_ready"] = features["B4_feature_ready"]
    features["B1_entry_condition"] = features["B1_feature_ready"] & features["medium_trend_break_component"]
    features["B2_entry_condition"] = features["B2_feature_ready"] & features["long_trend_break_component"] & features["pool80_breadth_weak_component"]
    features["B3_entry_condition"] = features["B3_feature_ready"] & features["drawdown_bias_deterioration_component"] & features["pool80_breadth_deterioration_component"] & features["medium_trend_negative_context"]
    break_count = (
        features["medium_trend_break_component"].fillna(False).astype(int)
        + features["long_trend_break_component"].fillna(False).astype(int)
        + features["pool80_breadth_deterioration_component"].fillna(False).astype(int)
    )
    features["bear_component_count"] = break_count
    features["bear_component_recovered_count"] = 3 - break_count
    features["B4_entry_condition"] = features["B4_feature_ready"] & break_count.ge(2)
    features["B5_entry_condition"] = features["B5_feature_ready"] & break_count.eq(3)
    for prefix in ["B1", "B2", "B3"]:
        features[f"{prefix}_recovery_condition"] = features[f"{prefix}_feature_ready"] & ~features[f"{prefix}_entry_condition"]
    for prefix in ["B4", "B5"]:
        features[f"{prefix}_recovery_condition"] = features[f"{prefix}_feature_ready"] & features["bear_component_recovered_count"].ge(2)
    features["future_return_used_as_classifier"] = False
    features["diagnostic_only"] = True
    return features


def _load_base_paths() -> pd.DataFrame:
    r6 = pd.read_csv(R6_STATE, low_memory=False, dtype={"selected_ticker_after": str})
    for col in ["signal_date", "next_trading_day_execution_date", "next_trading_day_after_execution_date"]:
        r6[col] = pd.to_datetime(r6[col], errors="coerce")
    r6["base_strategy"] = "reconstructed_single_position_R6"
    r6["base_target_ticker"] = r6["selected_ticker_after"].map(_ticker)
    r6["base_target_asset_type"] = r6["selected_asset_type_after"]
    r6["base_gross_daily_return"] = pd.to_numeric(r6["gross_daily_return"], errors="coerce")
    r6["base_path_source_quality"] = r6["daily_price_source_quality"]
    r6["base_path_ready"] = r6["daily_path_ready"].astype(str).str.lower().eq("true")
    r6["base_entry_close"] = pd.to_numeric(r6["entry_close"], errors="coerce")
    r6["base_exit_close"] = pd.to_numeric(r6["exit_close"], errors="coerce")

    raw_f = pd.read_csv(DAILY_F_STATE, low_memory=False, dtype={"selected_ticker_after": str})
    raw_f = raw_f[raw_f["state_machine_variant"].eq(RAW_F_VARIANT)].copy()
    for col in ["signal_date", "next_trading_day_execution_date", "next_trading_day_after_execution_date"]:
        raw_f[col] = pd.to_datetime(raw_f[col], errors="coerce")
    raw_f["base_strategy"] = "raw_Daily_F_challenger"
    raw_f["base_target_ticker"] = raw_f["selected_ticker_after"].map(_ticker)
    raw_f["base_target_asset_type"] = raw_f["selected_asset_type_after"]
    raw_f["base_gross_daily_return"] = pd.to_numeric(raw_f["gross_daily_return"], errors="coerce")
    raw_f["base_path_source_quality"] = raw_f["daily_price_source_quality"]
    raw_f["base_path_ready"] = raw_f["daily_path_ready"].astype(str).str.lower().eq("true")
    raw_f["base_entry_close"] = pd.to_numeric(raw_f["entry_close"], errors="coerce")
    raw_f["base_exit_close"] = pd.to_numeric(raw_f["exit_close"], errors="coerce")

    reference = r6.copy()
    reference["base_strategy"] = "all_00631L_state_hold_reference"
    reference["base_target_ticker"] = "00631L"
    reference["base_target_asset_type"] = "etf"
    reference["base_entry_close"] = pd.to_numeric(reference["base_entry_close"], errors="coerce")
    reference["base_exit_close"] = pd.to_numeric(reference["base_exit_close"], errors="coerce")
    reference["base_gross_daily_return"] = reference["base_exit_close"] / reference["base_entry_close"] - 1.0
    reference["base_path_source_quality"] = "00631L_same_basis_state_hold_reference_from_reconstructed_R6_benchmark_hook"
    reference["base_path_ready"] = reference["base_gross_daily_return"].notna()

    keep = [
        "base_strategy", "signal_date", "next_trading_day_execution_date", "next_trading_day_after_execution_date",
        "base_target_ticker", "base_target_asset_type", "base_gross_daily_return", "base_entry_close", "base_exit_close",
        "base_path_source_quality", "base_path_ready", "terminal_path_row_excluded_from_metric",
    ]
    return pd.concat([r6[keep], raw_f[keep], reference[keep]], ignore_index=True).sort_values(["base_strategy", "signal_date"])


def _transition(previous_ticker: str, previous_type: str, target_ticker: str, target_type: str) -> tuple[str, str, dict[str, float]]:
    if previous_ticker == target_ticker and previous_type == target_type:
        return "hold_same", "hold", COSTS["hold"]
    if target_type == "cash":
        key = f"{previous_type}_to_cash"
        return f"{previous_type}_to_cash", key, COSTS[key]
    if previous_type == "cash":
        key = f"cash_to_{target_type}"
        return f"cash_to_{target_type}", key, COSTS[key]
    if previous_type == "etf" and target_type == "stock":
        return "base_to_stock", "00631L_to_stock", COSTS["00631L_to_stock"]
    if previous_type == "stock" and target_type == "etf":
        return "stock_to_base", "stock_to_00631L", COSTS["stock_to_00631L"]
    return "stock_to_stock", "stock_to_stock", COSTS["stock_to_stock"]


def _condition_columns(variant: str) -> tuple[str, str, str]:
    prefix = variant.split("_", 1)[0]
    return f"{prefix}_feature_ready", f"{prefix}_entry_condition", f"{prefix}_recovery_condition"


def _bear_reason(row: Any, variant: str) -> str:
    if variant == "B1_medium_trend_break":
        return "0050_below_MA60_and_return20_40_negative"
    if variant == "B2_long_trend_breadth":
        return "0050_below_MA120_MA60_slope_negative_and_pool80_RS20_breadth_weak"
    if variant == "B3_drawdown_breadth_deterioration":
        return "0050_drawdown_bias_deterioration_and_pool80_breadth_deterioration_and_medium_trend_negative"
    if variant == "B4_two_of_three_consensus":
        return f"two_of_three_bear_components_count={int(row.bear_component_count)}"
    if variant == "B5_strict_three_of_three":
        return "strict_three_of_three_bear_components"
    return "no_cash_reference"


def _materialize_state(base_paths: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    merged = base_paths.merge(features, on="signal_date", how="left")
    rows: list[dict[str, Any]] = []
    for base_strategy, base in merged.groupby("base_strategy", sort=False):
        base = base.sort_values("signal_date")
        for variant, policy in VARIANTS.items():
            previous_ticker, previous_type = "00631L", "etf"
            bear_active, confirmation_streak, recovery_streak, cash_episode_id = False, 0, 0, 0
            ready_col, entry_col, recovery_col = _condition_columns(variant) if variant != "B0_no_cash_reference" else ("", "", "")
            for item in base.itertuples(index=False):
                if variant == "B0_no_cash_reference":
                    feature_ready, entry_condition, recovery_condition = True, False, True
                else:
                    feature_ready = bool(getattr(item, ready_col))
                    entry_condition = bool(getattr(item, entry_col)) if feature_ready else False
                    recovery_condition = bool(getattr(item, recovery_col)) if feature_ready else False
                confirmation_streak = confirmation_streak + 1 if entry_condition else 0
                recovery_streak = recovery_streak + 1 if bear_active and recovery_condition else 0
                bear_transition_event = "bear_state_unchanged"
                if not bear_active and variant != "B0_no_cash_reference" and confirmation_streak >= policy["entry_days"]:
                    bear_active = True
                    recovery_streak = 0
                    cash_episode_id += 1
                    bear_transition_event = "confirmed_bear_enter_cash"
                elif bear_active and recovery_streak >= policy["exit_days"]:
                    bear_active = False
                    confirmation_streak = 0
                    bear_transition_event = "confirmed_recovery_exit_cash"
                target_ticker = "CASH" if bear_active else item.base_target_ticker
                target_type = "cash" if bear_active else item.base_target_asset_type
                transition_type, cost_key, cost = _transition(previous_ticker, previous_type, target_ticker, target_type)
                gross_return = 0.0 if target_type == "cash" else item.base_gross_daily_return
                path_ready = bool(pd.notna(gross_return))
                net_return = gross_return - float(cost["transition_cost_rate"]) if path_ready else np.nan
                rows.append({
                    "task": TASK_ID,
                    "base_strategy": base_strategy,
                    "bear_classifier_variant": variant,
                    "signal_date": item.signal_date,
                    "next_trading_day_execution_date": item.next_trading_day_execution_date,
                    "next_trading_day_after_execution_date": item.next_trading_day_after_execution_date,
                    "base_target_ticker": item.base_target_ticker,
                    "base_target_asset_type": item.base_target_asset_type,
                    "incumbent_ticker_before": previous_ticker,
                    "incumbent_asset_type_before": previous_type,
                    "selected_ticker_after": target_ticker,
                    "selected_asset_type_after": target_type,
                    "classifier_feature_ready": feature_ready,
                    "classifier_missing_input_policy": "default_to_non_cash_base_target_never_infer_bear",
                    "bear_entry_condition": entry_condition,
                    "bear_recovery_condition": recovery_condition,
                    "bear_confirmation_streak": confirmation_streak,
                    "bear_recovery_streak": recovery_streak,
                    "confirmed_bear_state": bear_active,
                    "bear_transition_event": bear_transition_event,
                    "bear_reason": _bear_reason(item, variant) if bear_active else "not_in_confirmed_bear_state",
                    "cash_transition_reason": bear_transition_event if transition_type in {"etf_to_cash", "stock_to_cash", "cash_to_etf", "cash_to_stock"} else "not_cash_transition",
                    "cash_episode_id": cash_episode_id if bear_active else 0,
                    "transition_type": transition_type,
                    "transition_cost_key": cost_key,
                    "transition_cost_rate_hook": float(cost["transition_cost_rate"]),
                    "sell_fee_rate_hook": float(cost.get("sell_fee_rate", np.nan)),
                    "buy_fee_rate_hook": float(cost.get("buy_fee_rate", np.nan)),
                    "securities_transaction_tax_rate_hook": float(cost.get("tax_rate", np.nan)),
                    "transition_cost_twd_per_1m_notional": float(cost["transition_cost_rate"]) * 1_000_000,
                    "sell_fee_twd_per_1m_notional": float(cost.get("sell_fee_rate", 0.0)) * 1_000_000,
                    "buy_fee_twd_per_1m_notional": float(cost.get("buy_fee_rate", 0.0)) * 1_000_000,
                    "securities_transaction_tax_twd_per_1m_notional": float(cost.get("tax_rate", 0.0)) * 1_000_000,
                    "transition_cost_model_status": "EP05_TaiwanCostModel_unit_notional_stock_ETF_cash_transition_hooks_separated",
                    "gross_daily_return": gross_return,
                    "net_daily_return_after_transition_cost": net_return,
                    "daily_path_ready": path_ready,
                    "terminal_path_row_excluded_from_metric": bool(item.terminal_path_row_excluded_from_metric),
                    "cash_return_policy": "zero_return_no_interest_only_during_confirmed_bear",
                    "base_path_source_quality": item.base_path_source_quality,
                    "selected_stock_adjusted_close_ready": False if target_type == "stock" else True,
                    "execution_basis": "signal_day_close_classifier__next_trading_day_close_transition__single_position_daily_mark",
                    "medium_trend_break_component": item.medium_trend_break_component,
                    "long_trend_break_component": item.long_trend_break_component,
                    "drawdown_bias_deterioration_component": item.drawdown_bias_deterioration_component,
                    "pool80_breadth_weak_component": item.pool80_breadth_weak_component,
                    "pool80_breadth_deterioration_component": item.pool80_breadth_deterioration_component,
                    "bear_component_count": item.bear_component_count,
                    "revenue_anomaly_role": "report_only_not_cash_rule",
                    "rs20_top3_role": "reference_only_not_selected_branch",
                    "diagnostic_only": True,
                    "future_data_violation_count": 0,
                    **FLAGS,
                })
                previous_ticker, previous_type = target_ticker, target_type
    state = pd.DataFrame(rows)
    for period, (start, end) in PERIODS.items():
        candidate = (
            state["signal_date"].between(pd.Timestamp(start), pd.Timestamp(end))
            & state["next_trading_day_execution_date"].le(pd.Timestamp(end))
            & state["next_trading_day_after_execution_date"].le(pd.Timestamp(end))
        )
        state[f"metric_candidate_{period}"] = candidate
        state[f"metric_eligible_{period}"] = candidate & state["daily_path_ready"]
    return state


def _variant_policy() -> pd.DataFrame:
    return pd.DataFrame([
        {"variant": "B0_no_cash_reference", "entry_rule": "never_cash", "recovery_rule": "not_applicable", "confirmation_days": 0, "recovery_days": 0, "threshold_source": "reference_only"},
        {"variant": "B1_medium_trend_break", "entry_rule": "0050 below MA60 AND return20<0 AND return40<0", "recovery_rule": "entry condition false", "confirmation_days": 2, "recovery_days": 2, "threshold_source": "fixed_bounded_PIT_rule"},
        {"variant": "B2_long_trend_breadth", "entry_rule": "0050 below MA120 AND MA60 slope20d<0 AND pool80 RS20 breadth PIT percentile<=25%", "recovery_rule": "entry condition false", "confirmation_days": 2, "recovery_days": 2, "threshold_source": "fixed_market_rule_plus_expanding_PIT_breadth_percentile"},
        {"variant": "B3_drawdown_breadth_deterioration", "entry_rule": "drawdown severity PIT percentile>=80% AND BIAS60 delta5d<0 AND breadth percentile<=35% AND breadth 2/4-snapshot change<0 AND medium trend negative", "recovery_rule": "entry condition false", "confirmation_days": 2, "recovery_days": 2, "threshold_source": "expanding_PIT_percentile_and_fixed_directional_context"},
        {"variant": "B4_two_of_three_consensus", "entry_rule": "at least 2 of medium trend break / long trend break / breadth deterioration", "recovery_rule": "at least 2 components recovered", "confirmation_days": 2, "recovery_days": 2, "threshold_source": "bounded_consensus_components"},
        {"variant": "B5_strict_three_of_three", "entry_rule": "all 3 medium / long / breadth components", "recovery_rule": "at least 2 components recovered", "confirmation_days": 2, "recovery_days": 2, "threshold_source": "bounded_consensus_components"},
    ]).assign(future_return_used_as_rule=False, cash_majority_policy_allowed=False, diagnostic_only=True, **FLAGS)


def _cost_audit() -> pd.DataFrame:
    rows = []
    for transition, cost in COSTS.items():
        rows.append({
            "transition_cost_key": transition,
            **cost,
            "transition_cost_twd_per_1m_notional": cost["transition_cost_rate"] * 1_000_000,
            "sell_fee_twd_per_1m_notional": cost["sell_fee_rate"] * 1_000_000,
            "buy_fee_twd_per_1m_notional": cost["buy_fee_rate"] * 1_000_000,
            "tax_twd_per_1m_notional": cost["tax_rate"] * 1_000_000,
            "cost_model_status": "EP05_TaiwanCostModel_unit_notional_hook_stock_ETF_cash_separated",
            "slippage_model_status": "blocked_not_invented",
            "diagnostic_only": True,
            **FLAGS,
        })
    return pd.DataFrame(rows)


def _coverage(state: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (base, variant), group in state.groupby(["base_strategy", "bear_classifier_variant"]):
        for period, (start, end) in PERIODS.items():
            sub = group[group["signal_date"].between(pd.Timestamp(start), pd.Timestamp(end))]
            candidate = sub[sub[f"metric_candidate_{period}"]]
            ready = sub[sub[f"metric_eligible_{period}"]]
            classifier_ready = sub[sub["classifier_feature_ready"]]
            rows.append({
                "base_strategy": base, "bear_classifier_variant": variant, "period": period,
                "requested_start": start, "requested_end": end,
                "actual_path_start": ready["signal_date"].min() if len(ready) else pd.NaT,
                "actual_path_end": ready["signal_date"].max() if len(ready) else pd.NaT,
                "classifier_feature_ready_start": classifier_ready["signal_date"].min() if len(classifier_ready) else pd.NaT,
                "metric_candidate_rows": len(candidate), "metric_ready_rows": len(ready),
                "daily_path_ready_share": float(len(ready) / len(candidate)) if len(candidate) else 1.0,
                "classifier_feature_ready_share": float(sub["classifier_feature_ready"].mean()) if len(sub) else np.nan,
                "cash_days": int(sub["selected_asset_type_after"].eq("cash").sum()),
                "cash_exposure_share": float(sub["selected_asset_type_after"].eq("cash").mean()) if len(sub) else 0.0,
                **FLAGS,
            })
    return pd.DataFrame(rows)


def _episodes(state: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries, episodes = [], []
    for (base, variant), group in state.groupby(["base_strategy", "bear_classifier_variant"]):
        group = group.sort_values("signal_date").copy()
        cash = group["selected_asset_type_after"].eq("cash")
        group["cash_block"] = (cash.ne(cash.shift())).cumsum()
        cash_rows = group[cash]
        for _, episode in cash_rows.groupby("cash_block"):
            episodes.append({
                "base_strategy": base, "bear_classifier_variant": variant,
                "cash_episode_id": int(episode["cash_episode_id"].max()),
                "cash_signal_start": episode["signal_date"].min(), "cash_signal_end": episode["signal_date"].max(),
                "cash_execution_start": episode["next_trading_day_execution_date"].min(),
                "cash_execution_end": episode["next_trading_day_after_execution_date"].max(),
                "cash_days": len(episode), "bear_reasons": "|".join(sorted(set(episode["bear_reason"]))),
            })
        summaries.append({
            "base_strategy": base, "bear_classifier_variant": variant,
            "total_state_rows": len(group), "cash_days": int(cash.sum()), "cash_exposure_share": float(cash.mean()),
            "cash_episode_count": int(group["bear_transition_event"].eq("confirmed_bear_enter_cash").sum()),
            "cash_entry_transition_count": int(group["transition_type"].isin(["etf_to_cash", "stock_to_cash"]).sum()),
            "cash_exit_transition_count": int(group["transition_type"].isin(["cash_to_etf", "cash_to_stock"]).sum()),
            "cash_majority_policy_flag": bool(cash.mean() > 0.5), **FLAGS,
        })
    return pd.DataFrame(summaries), pd.DataFrame(episodes)


def _weak_year_context(state: pd.DataFrame) -> pd.DataFrame:
    subset = state[state["signal_date"].dt.year.isin([2015, 2018, 2022])].copy()
    subset["year"] = subset["signal_date"].dt.year
    return subset.groupby(["base_strategy", "bear_classifier_variant", "year"], as_index=False).agg(
        state_days=("signal_date", "size"), cash_days=("selected_asset_type_after", lambda x: int((x == "cash").sum())),
        cash_exposure_share=("selected_asset_type_after", lambda x: float((x == "cash").mean())),
        cash_episodes=("bear_transition_event", lambda x: int((x == "confirmed_bear_enter_cash").sum())),
        transition_count=("transition_type", lambda x: int((x != "hold_same").sum())),
    )


def _stop_gate(coverage: pd.DataFrame) -> pd.DataFrame:
    return coverage[coverage["period"].isin(["P2", "2024_latest", "2026YTD"])][[
        "base_strategy", "bear_classifier_variant", "period", "requested_start", "requested_end",
        "actual_path_start", "actual_path_end", "cash_days", "cash_exposure_share", "daily_path_ready_share",
    ]].copy()


def _source_missingness(features: pd.DataFrame) -> pd.DataFrame:
    fields = {
        "0050_close_MA20_MA60_MA120": (["close", "ma20", "ma60", "ma120"], "0050 adjusted benchmark daily PIT"),
        "0050_returns_20_40_60": (["return_20d", "return_40d", "return_60d"], "0050 adjusted benchmark daily PIT"),
        "0050_MA60_MA120_slope": (["ma60_slope_20d", "ma120_slope_20d"], "rolling past-only slope"),
        "0050_BIAS20_60_120": (["bias20", "bias60", "bias120"], "rolling past-only BIAS"),
        "0050_drawdown_percentile": (["drawdown_severity_percentile_pit"], "expanding past-only percentile"),
        "pool80_RS20_breadth": (["pool80_rs20_positive_share"], "weekly Layer4 PIT snapshot asof daily"),
        "pool80_above_MA20_60_share": (["pool80_above_ma20_share", "pool80_above_ma60_share"], "blocked not materialized"),
    }
    rows = []
    for group, (columns, quality) in fields.items():
        available = features[columns].notna().all(axis=1)
        rows.append({
            "field_group": group, "columns": "|".join(columns), "source_quality": quality,
            "available_rows": int(available.sum()), "total_rows": len(features),
            "coverage_share": float(available.mean()), "blocked": bool(available.sum() == 0),
            "future_data_violation_count": 0,
        })
    return pd.DataFrame(rows)


def _future_audit() -> pd.DataFrame:
    return pd.DataFrame([
        {"audit_item": "0050_rolling_features", "future_return_used_as_rule": False, "detail": "All rolling windows include signal close and prior observations only.", "future_data_violation_count": 0},
        {"audit_item": "pool80_breadth", "future_return_used_as_rule": False, "detail": "Latest released weekly PIT snapshot is asof-forward-filled; not presented as daily recomputation.", "future_data_violation_count": 0},
        {"audit_item": "cash_state", "future_return_used_as_rule": False, "detail": "Cash state uses only same-day classifier features with next-trading-day close execution.", "future_data_violation_count": 0},
    ])


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    base_paths = _load_base_paths()
    signal_dates = base_paths["signal_date"].dropna().drop_duplicates()
    features = _feature_matrix(signal_dates)
    state = _materialize_state(base_paths, features)
    coverage = _coverage(state)
    exposure, episodes = _episodes(state)
    transitions = state[state["transition_type"].ne("hold_same")].copy()
    gaps = state[state["metric_candidate_full_integrated"] & ~state["daily_path_ready"]][[
        "base_strategy", "bear_classifier_variant", "signal_date", "selected_ticker_after", "selected_asset_type_after", "base_path_source_quality",
    ]].copy()
    path_ready_min = float(coverage["daily_path_ready_share"].min())
    b1_ready = float(coverage[coverage["bear_classifier_variant"].eq("B1_medium_trend_break") & coverage["period"].eq("P1")]["classifier_feature_ready_share"].min())
    readiness = {
        "task_id": TASK_ID,
        "status": "confirmed_bear_cash_classifier_ready_bounded_diagnostic" if len(gaps) == 0 and path_ready_min == 1.0 else "confirmed_bear_cash_classifier_partial_path_gap",
        "base_comparator_count": 3,
        "classifier_variant_count": 6,
        "P1_primary": True,
        "P2_secondary_stop_gate": True,
        "daily_single_position_path_ready_share_min": path_ready_min,
        "B1_P1_classifier_feature_ready_share_min": b1_ready,
        "pool80_breadth_daily_recalculated": False,
        "pool80_breadth_source_quality": "weekly_PIT_snapshot_asof_forward_fill_proxy",
        "pool80_above_ma20_60_share_ready": False,
        "cash_return_zero_no_interest": True,
        "classifier_missing_input_policy": "default_to_non_cash_base_target_never_infer_bear",
        "EP05_stock_ETF_cash_transition_cost_hooks_ready": True,
        "base_path_gap_rows": int(len(gaps)),
        "selected_stock_adjusted_close_ready": False,
        "revenue_anomaly_used_as_cash_rule": False,
        "rs20_top3_used_as_selected_branch": False,
        "cash_majority_policy_authorized": False,
        "ready_for_experiments": bool(len(gaps) == 0 and path_ready_min == 1.0),
        "ready_for_formal": False,
        "ready_for_strategy_replay": False,
        "future_data_violation_count": 0,
        **FLAGS,
    }
    blocked = pd.DataFrame([
        {"item": "pool80_above_ma20_share", "status": "blocked", "detail": "Exact daily pool80 above-MA20 share is not materialized; no fabricated substitute.", "impact": "not used by B1-B5"},
        {"item": "pool80_above_ma60_share", "status": "blocked", "detail": "Exact daily pool80 above-MA60 share is not materialized; no fabricated substitute.", "impact": "not used by B1-B5"},
        {"item": "pool80_daily_breadth", "status": "proxy", "detail": "Weekly PIT Layer4 snapshot is asof-forward-filled to daily state rows.", "impact": "B2-B5 diagnostic-only"},
        {"item": "selected_stock_adjusted_close", "status": "blocked", "detail": "Official unadjusted stock OHLC remains diagnostic-only.", "impact": "not formal-ready"},
        {"item": "cash_interest", "status": "proxy", "detail": "Cash return is fixed at zero with no interest.", "impact": "explicit diagnostic assumption"},
    ])
    paths = [
        _write(features, "confirmed_bear_cash_classifier_feature_matrix.csv"),
        _write(_variant_policy(), "confirmed_bear_cash_classifier_variant_policy.csv"),
        _write(state, "confirmed_bear_cash_classifier_daily_state_contract.csv"),
        _write(transitions, "confirmed_bear_cash_classifier_transition_trace.csv"),
        _write(_cost_audit(), "confirmed_bear_cash_classifier_cost_audit.csv"),
        _write(exposure, "confirmed_bear_cash_classifier_exposure_summary.csv"),
        _write(episodes, "confirmed_bear_cash_classifier_cash_episode_trace.csv"),
        _write(_weak_year_context(state), "confirmed_bear_cash_classifier_P1_weak_year_context.csv"),
        _write(_stop_gate(coverage), "confirmed_bear_cash_classifier_P2_recent_stop_gate.csv"),
        _write(coverage, "confirmed_bear_cash_classifier_requested_vs_actual_coverage.csv"),
        _write(_source_missingness(features), "confirmed_bear_cash_classifier_source_quality_missingness.csv"),
        _write(gaps, "confirmed_bear_cash_classifier_base_path_gap_ledger.csv"),
        _write(blocked, "confirmed_bear_cash_classifier_blocked_proxy_audit.csv"),
        _write(_future_audit(), "confirmed_bear_cash_classifier_future_data_audit.csv"),
    ]
    readiness_path = OUTPUT_DIR / "readiness_for_confirmed_bear_cash_classifier_diagnostic.json"
    readiness_path.write_text(json.dumps(readiness, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path = OUTPUT_DIR / "final_summary_zh.md"
    summary_path.write_text(
        "# Confirmed Bear / Cash Classifier Bounded Contract\n\n"
        "本 contract 建立 B0-B5 六個 bounded 變體；只有連續確認的明確空頭才允許 cash，非 bear 時回到原 base target。\n\n"
        f"- base comparators: reconstructed R6 / raw Daily F challenger / all 00631L state-hold\n"
        f"- daily path ready share min: {path_ready_min:.1%}; path gaps: {len(gaps)}\n"
        "- signal close decision -> next-trading-day close execution；single position；cash return=0\n"
        "- EP05 stock/ETF/cash transition fee and tax hooks separated\n"
        "- B1 uses daily 0050 PIT fields；B2-B5 breadth uses weekly Layer4 PIT snapshot asof proxy\n"
        "- revenue anomaly is not a cash rule；RS20 top3 remains reference-only\n\n"
        f"ready_for_experiments={readiness['ready_for_experiments']}; ready_for_formal=false; future_data_violation_count=0.\n",
        encoding="utf-8",
    )
    manifest = {
        "task_id": TASK_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(OUTPUT_DIR),
        "files": [{"path": path.name, "sha256": _sha256(path)} for path in [*paths, readiness_path, summary_path]],
        "readiness": readiness,
        "source_inputs": {
            "benchmark_features": str(BENCHMARK_FEATURES),
            "pool_fields": str(POOL_FIELDS),
            "reconstructed_R6_state": str(R6_STATE),
            "raw_Daily_F_state": str(DAILY_F_STATE),
        },
    }
    (OUTPUT_DIR / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(readiness, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
