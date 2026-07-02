from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd

from backtest_lab.costs import TaiwanCostModel
from backtest_lab.bear_defense_backtest import risk_on_for_rule
from backtest_lab.market_regime import classify_market_regime
from backtest_lab.portfolio import Trade
from backtest_lab.regime_aware_simulation import _Account, _market_value, _rebalance
from backtest_lab.simulation import BacktestResult, _common_trade_dates, _date_str, _max_drawdown
from backtest_lab.strategies import (
    dual_momentum_vol_control,
    previous_available_date,
    relative_strength_scores,
    relative_strength_top1,
)


MODE_DAILY = "daily_strength"
MODE_WEEKLY = "weekly_rotation"
MODE_0050_DEFENSE = "0050_defense"
MODE_CASH = "cash"
MODE_HOLD = "hold"

_DAILY_HEALTH_CACHE: dict[tuple, bool] = {}


@dataclass(frozen=True)
class RegimeModeSwitchVariant:
    name: str
    regime_modes: dict[str, str]
    regime_exposures: dict[str, float]
    weekly_signal_weekday: int | None = None
    normal_rebalance_weekday: int | None = None
    normal_rebalance_weekday_by_regime: dict[str, int] | None = None
    normal_rebalance_min_regime_streak_days: int = 0
    normal_rebalance_last_trading_day_of_week: bool = False
    normal_rebalance_last_trading_day_regimes: tuple[str, ...] = ()
    state_evaluation_weekday: int | None = None
    defer_rebalance_on_adverse_open_gap_pct: float | None = None
    early_rebalance_regimes: tuple[str, ...] = ()
    early_rebalance_min_gap_over_current: float | None = None
    early_rebalance_min_top_gap: float | None = None
    portfolio_stop_drawdown_pct: float | None = None
    portfolio_stop_cooldown_days: int = 0
    portfolio_stop_latch_mode: str | None = None
    portfolio_stop_latch_defense_rule: str = "ma245"
    portfolio_stop_release_filter: str | None = None
    portfolio_stop_release_confirmation_days: int = 0
    portfolio_stop_release_health_lookback_days: int | None = None
    portfolio_stop_release_health_min_excess_return: float = 0.0
    candidate_trend_filter: str | None = None
    market_risk_off_filter: str | None = None
    market_risk_off_mode: str = MODE_CASH
    market_risk_off_defense_rule: str = "ma245"
    market_risk_off_defense_ticker: str | None = None
    market_risk_off_exposure: float | None = None
    market_risk_off_exposure_selector: str | None = None
    market_risk_off_exit_confirmation_days: int = 0
    market_risk_off_only_when_attack_gate_inactive: bool = False
    market_risk_off_only_before_first_attack_activation: bool = False
    fallback_ticker: str | None = None
    fallback_exposure: float = 1.0
    relative_score_fallback_ticker: str | None = None
    relative_score_fallback_defense_rule: str | None = None
    min_score_over_fallback: float | None = None
    min_score_over_fallback_by_regime: dict[str, float] | None = None
    margin_gate_ticker: str | None = None
    gated_on_margin: float | None = None
    gated_off_margin: float | None = None
    defense_anchor_ticker: str = "0050.TW"
    defense_anchor_ticker_by_regime: dict[str, str] | None = None
    defense_rule_by_regime: dict[str, str] | None = None
    defense_risk_off_exposure_by_regime: dict[str, float] | None = None
    daily_health_lookback_days: int | None = None
    daily_health_min_excess_return: float = 0.0
    daily_health_benchmark_ticker: str = "0050.TW"
    daily_health_fail_mode: str = MODE_0050_DEFENSE
    daily_health_fail_defense_rule: str = "ma245"
    daily_health_recovery_confirmation_days: int = 0
    attack_gate_margin_over_fallback: float | None = None
    attack_gate_fallback_ticker: str = "0050.TW"
    attack_gate_defense_rule: str | None = "ma245"
    attack_gate_defense_rule_by_regime: dict[str, str] | None = None
    attack_gate_defense_exposure_by_regime: dict[str, float] | None = None
    attack_gate_activation_confirmation_days: int = 1
    attack_gate_persistence_lookback_days: int = 0
    attack_gate_min_top_days: int = 0
    attack_gate_min_short_to_medium_momentum_ratio: float | None = None
    attack_gate_reentry_margin_over_fallback: float | None = None
    attack_gate_reentry_min_short_to_medium_momentum_ratio: float | None = None
    attack_gate_initialize_history_days: int = 0
    attack_gate_initialize_active_from_history: bool = False
    attack_selection_exclude_tickers: tuple[str, ...] = ()
    attack_gate_exclude_tickers: tuple[str, ...] = ()
    attack_gate_stop_latch_ticker: str | None = None
    attack_gate_stop_latch_rule: str | None = None
    attack_gate_stop_release_filter: str | None = None
    attack_gate_stop_release_confirmation_days: int = 0


@dataclass(frozen=True)
class FrozenCycleProvenTop1Spec:
    name: str = "frozen_cycle_proven_top1_v1"
    base_family: str = "cycle_proven_preproof_exposure_variants"
    base_selector: str = "market_risk_off_exposure_25pct"
    market_risk_off_exposure: float = 0.25
    defense_anchor_ticker: str = "00631L.TW"
    attack_selection_exclude_tickers: tuple[str, ...] = ("0050.TW", "00631L.TW")


@dataclass(frozen=True)
class ExposureOverlayDecision:
    adjusted_exposure: float
    risk_flag: bool = False
    reason: str = ""
    signal_date: str = ""


ExposureOverlay = Callable[[str | None, pd.Timestamp, pd.Timestamp, float], ExposureOverlayDecision]


@dataclass(frozen=True)
class TargetSelectionOverlayDecision:
    target: str | None
    reason: str = ""
    signal_date: str = ""


TargetSelectionOverlay = Callable[
    [str, dict[str, pd.DataFrame], pd.Timestamp, pd.Timestamp, str, RegimeModeSwitchVariant, str | None, float],
    TargetSelectionOverlayDecision,
]


@dataclass
class RegimeModeSwitchState:
    account_cash: float
    account_ticker: str | None = None
    account_shares: int = 0
    last_week_key_year: int | None = None
    last_week_key_week: int | None = None
    peak_signal_value: float = 0.0
    cooldown_until_date: str = ""
    risk_off_active: bool = False
    risk_off_clear_streak: int = 0
    daily_health_active: bool = True
    daily_health_recovery_streak: int = 0
    stop_latch_active: bool = False
    stop_release_streak: int = 0
    attack_gate_active: bool = False
    attack_gate_activation_streak: int = 0
    attack_gate_ever_activated: bool = False
    attack_gate_stop_latch_active: bool = False
    attack_gate_stop_release_streak: int = 0
    current_regime: str | None = None
    regime_streak_days: int = 0


def default_mode_switch_variants() -> tuple[RegimeModeSwitchVariant, ...]:
    return (
        RegimeModeSwitchVariant(
            name="daily_margin0050_25_ma250_defense",
            regime_modes={
                "strong_bull": MODE_DAILY,
                "recovery_bull": MODE_DAILY,
                "range_bound": MODE_DAILY,
                "correction_bear": MODE_DAILY,
                "systemic_bear": MODE_CASH,
            },
            regime_exposures={
                "strong_bull": 1.0,
                "recovery_bull": 1.0,
                "range_bound": 1.0,
                "correction_bear": 1.0,
                "systemic_bear": 0.0,
            },
            relative_score_fallback_ticker="0050.TW",
            relative_score_fallback_defense_rule="ma250",
            min_score_over_fallback=0.25,
        ),
        RegimeModeSwitchVariant(
            name="daily_margin0050_30_ma250_defense",
            regime_modes={
                "strong_bull": MODE_DAILY,
                "recovery_bull": MODE_DAILY,
                "range_bound": MODE_DAILY,
                "correction_bear": MODE_DAILY,
                "systemic_bear": MODE_CASH,
            },
            regime_exposures={
                "strong_bull": 1.0,
                "recovery_bull": 1.0,
                "range_bound": 1.0,
                "correction_bear": 1.0,
                "systemic_bear": 0.0,
            },
            relative_score_fallback_ticker="0050.TW",
            relative_score_fallback_defense_rule="ma250",
            min_score_over_fallback=0.30,
        ),
        RegimeModeSwitchVariant(
            name="strong_daily_health40_min0_else_ma245",
            regime_modes={
                "strong_bull": MODE_DAILY,
                "recovery_bull": MODE_0050_DEFENSE,
                "range_bound": MODE_0050_DEFENSE,
                "correction_bear": MODE_0050_DEFENSE,
                "systemic_bear": MODE_CASH,
            },
            regime_exposures={
                "strong_bull": 1.0,
                "recovery_bull": 1.0,
                "range_bound": 1.0,
                "correction_bear": 1.0,
                "systemic_bear": 0.0,
            },
            defense_rule_by_regime={"recovery_bull": "ma245", "range_bound": "ma245", "correction_bear": "ma245"},
            defense_risk_off_exposure_by_regime={"recovery_bull": 0.0, "range_bound": 0.0, "correction_bear": 0.0},
            daily_health_lookback_days=40,
            daily_health_min_excess_return=0.0,
            daily_health_fail_defense_rule="ma245",
        ),
        RegimeModeSwitchVariant(
            name="strong_daily_health60_min0_else_ma245",
            regime_modes={
                "strong_bull": MODE_DAILY,
                "recovery_bull": MODE_0050_DEFENSE,
                "range_bound": MODE_0050_DEFENSE,
                "correction_bear": MODE_0050_DEFENSE,
                "systemic_bear": MODE_CASH,
            },
            regime_exposures={
                "strong_bull": 1.0,
                "recovery_bull": 1.0,
                "range_bound": 1.0,
                "correction_bear": 1.0,
                "systemic_bear": 0.0,
            },
            defense_rule_by_regime={"recovery_bull": "ma245", "range_bound": "ma245", "correction_bear": "ma245"},
            defense_risk_off_exposure_by_regime={"recovery_bull": 0.0, "range_bound": 0.0, "correction_bear": 0.0},
            daily_health_lookback_days=60,
            daily_health_min_excess_return=0.0,
            daily_health_fail_defense_rule="ma245",
        ),
        RegimeModeSwitchVariant(
            name="strong_daily_health60_min5_else_ma245",
            regime_modes={
                "strong_bull": MODE_DAILY,
                "recovery_bull": MODE_0050_DEFENSE,
                "range_bound": MODE_0050_DEFENSE,
                "correction_bear": MODE_0050_DEFENSE,
                "systemic_bear": MODE_CASH,
            },
            regime_exposures={
                "strong_bull": 1.0,
                "recovery_bull": 1.0,
                "range_bound": 1.0,
                "correction_bear": 1.0,
                "systemic_bear": 0.0,
            },
            defense_rule_by_regime={"recovery_bull": "ma245", "range_bound": "ma245", "correction_bear": "ma245"},
            defense_risk_off_exposure_by_regime={"recovery_bull": 0.0, "range_bound": 0.0, "correction_bear": 0.0},
            daily_health_lookback_days=60,
            daily_health_min_excess_return=0.05,
            daily_health_fail_defense_rule="ma245",
        ),
        RegimeModeSwitchVariant(
            name="daily_health20_min0_ma245_defense",
            regime_modes={
                "strong_bull": MODE_DAILY,
                "recovery_bull": MODE_DAILY,
                "range_bound": MODE_DAILY,
                "correction_bear": MODE_DAILY,
                "systemic_bear": MODE_CASH,
            },
            regime_exposures={
                "strong_bull": 1.0,
                "recovery_bull": 1.0,
                "range_bound": 1.0,
                "correction_bear": 1.0,
                "systemic_bear": 0.0,
            },
            daily_health_lookback_days=20,
            daily_health_min_excess_return=0.0,
            daily_health_fail_defense_rule="ma245",
        ),
        RegimeModeSwitchVariant(
            name="daily_health20_min5_ma245_defense",
            regime_modes={
                "strong_bull": MODE_DAILY,
                "recovery_bull": MODE_DAILY,
                "range_bound": MODE_DAILY,
                "correction_bear": MODE_DAILY,
                "systemic_bear": MODE_CASH,
            },
            regime_exposures={
                "strong_bull": 1.0,
                "recovery_bull": 1.0,
                "range_bound": 1.0,
                "correction_bear": 1.0,
                "systemic_bear": 0.0,
            },
            daily_health_lookback_days=20,
            daily_health_min_excess_return=0.05,
            daily_health_fail_defense_rule="ma245",
        ),
        RegimeModeSwitchVariant(
            name="daily_health40_min0_ma245_defense",
            regime_modes={
                "strong_bull": MODE_DAILY,
                "recovery_bull": MODE_DAILY,
                "range_bound": MODE_DAILY,
                "correction_bear": MODE_DAILY,
                "systemic_bear": MODE_CASH,
            },
            regime_exposures={
                "strong_bull": 1.0,
                "recovery_bull": 1.0,
                "range_bound": 1.0,
                "correction_bear": 1.0,
                "systemic_bear": 0.0,
            },
            daily_health_lookback_days=40,
            daily_health_min_excess_return=0.0,
            daily_health_fail_defense_rule="ma245",
        ),
        RegimeModeSwitchVariant(
            name="daily_health40_min5_ma245_defense",
            regime_modes={
                "strong_bull": MODE_DAILY,
                "recovery_bull": MODE_DAILY,
                "range_bound": MODE_DAILY,
                "correction_bear": MODE_DAILY,
                "systemic_bear": MODE_CASH,
            },
            regime_exposures={
                "strong_bull": 1.0,
                "recovery_bull": 1.0,
                "range_bound": 1.0,
                "correction_bear": 1.0,
                "systemic_bear": 0.0,
            },
            daily_health_lookback_days=40,
            daily_health_min_excess_return=0.05,
            daily_health_fail_defense_rule="ma245",
        ),
        RegimeModeSwitchVariant(
            name="daily_health60_min0_ma245_defense",
            regime_modes={
                "strong_bull": MODE_DAILY,
                "recovery_bull": MODE_DAILY,
                "range_bound": MODE_DAILY,
                "correction_bear": MODE_DAILY,
                "systemic_bear": MODE_CASH,
            },
            regime_exposures={
                "strong_bull": 1.0,
                "recovery_bull": 1.0,
                "range_bound": 1.0,
                "correction_bear": 1.0,
                "systemic_bear": 0.0,
            },
            daily_health_lookback_days=60,
            daily_health_min_excess_return=0.0,
            daily_health_fail_defense_rule="ma245",
        ),
        RegimeModeSwitchVariant(
            name="daily_health60_min5_ma245_defense",
            regime_modes={
                "strong_bull": MODE_DAILY,
                "recovery_bull": MODE_DAILY,
                "range_bound": MODE_DAILY,
                "correction_bear": MODE_DAILY,
                "systemic_bear": MODE_CASH,
            },
            regime_exposures={
                "strong_bull": 1.0,
                "recovery_bull": 1.0,
                "range_bound": 1.0,
                "correction_bear": 1.0,
                "systemic_bear": 0.0,
            },
            daily_health_lookback_days=60,
            daily_health_min_excess_return=0.05,
            daily_health_fail_defense_rule="ma245",
        ),
        RegimeModeSwitchVariant(
            name="fast_risk_ma60_ret20_margin40_ma245",
            regime_modes={
                "strong_bull": MODE_DAILY,
                "recovery_bull": MODE_DAILY,
                "range_bound": MODE_DAILY,
                "correction_bear": MODE_DAILY,
                "systemic_bear": MODE_CASH,
            },
            regime_exposures={
                "strong_bull": 1.0,
                "recovery_bull": 1.0,
                "range_bound": 1.0,
                "correction_bear": 1.0,
                "systemic_bear": 0.0,
            },
            market_risk_off_filter="ma60_ret20",
            market_risk_off_mode=MODE_0050_DEFENSE,
            market_risk_off_defense_rule="ma245",
            relative_score_fallback_ticker="0050.TW",
            relative_score_fallback_defense_rule="ma245",
            min_score_over_fallback=0.40,
        ),
        RegimeModeSwitchVariant(
            name="fast_risk_dd5_ma60_ret20_margin40_ma245",
            regime_modes={
                "strong_bull": MODE_DAILY,
                "recovery_bull": MODE_DAILY,
                "range_bound": MODE_DAILY,
                "correction_bear": MODE_DAILY,
                "systemic_bear": MODE_CASH,
            },
            regime_exposures={
                "strong_bull": 1.0,
                "recovery_bull": 1.0,
                "range_bound": 1.0,
                "correction_bear": 1.0,
                "systemic_bear": 0.0,
            },
            market_risk_off_filter="dd5_ma60_ret20",
            market_risk_off_mode=MODE_0050_DEFENSE,
            market_risk_off_defense_rule="ma245",
            relative_score_fallback_ticker="0050.TW",
            relative_score_fallback_defense_rule="ma245",
            min_score_over_fallback=0.40,
        ),
        RegimeModeSwitchVariant(
            name="fast_risk_dd8_ma60_ret20_margin40_ma245",
            regime_modes={
                "strong_bull": MODE_DAILY,
                "recovery_bull": MODE_DAILY,
                "range_bound": MODE_DAILY,
                "correction_bear": MODE_DAILY,
                "systemic_bear": MODE_CASH,
            },
            regime_exposures={
                "strong_bull": 1.0,
                "recovery_bull": 1.0,
                "range_bound": 1.0,
                "correction_bear": 1.0,
                "systemic_bear": 0.0,
            },
            market_risk_off_filter="dd8_ma60_ret20",
            market_risk_off_mode=MODE_0050_DEFENSE,
            market_risk_off_defense_rule="ma245",
            relative_score_fallback_ticker="0050.TW",
            relative_score_fallback_defense_rule="ma245",
            min_score_over_fallback=0.40,
        ),
        RegimeModeSwitchVariant(
            name="fast_risk_2of3_margin40_ma245",
            regime_modes={
                "strong_bull": MODE_DAILY,
                "recovery_bull": MODE_DAILY,
                "range_bound": MODE_DAILY,
                "correction_bear": MODE_DAILY,
                "systemic_bear": MODE_CASH,
            },
            regime_exposures={
                "strong_bull": 1.0,
                "recovery_bull": 1.0,
                "range_bound": 1.0,
                "correction_bear": 1.0,
                "systemic_bear": 0.0,
            },
            market_risk_off_filter="risk_2of3",
            market_risk_off_mode=MODE_0050_DEFENSE,
            market_risk_off_defense_rule="ma245",
            relative_score_fallback_ticker="0050.TW",
            relative_score_fallback_defense_rule="ma245",
            min_score_over_fallback=0.40,
        ),
        RegimeModeSwitchVariant(
            name="daily_margin0050_40_ma240_defense",
            regime_modes={
                "strong_bull": MODE_DAILY,
                "recovery_bull": MODE_DAILY,
                "range_bound": MODE_DAILY,
                "correction_bear": MODE_DAILY,
                "systemic_bear": MODE_CASH,
            },
            regime_exposures={
                "strong_bull": 1.0,
                "recovery_bull": 1.0,
                "range_bound": 1.0,
                "correction_bear": 1.0,
                "systemic_bear": 0.0,
            },
            relative_score_fallback_ticker="0050.TW",
            relative_score_fallback_defense_rule="ma240",
            min_score_over_fallback=0.40,
        ),
        RegimeModeSwitchVariant(
            name="daily_margin0050_40_ma245_defense",
            regime_modes={
                "strong_bull": MODE_DAILY,
                "recovery_bull": MODE_DAILY,
                "range_bound": MODE_DAILY,
                "correction_bear": MODE_DAILY,
                "systemic_bear": MODE_CASH,
            },
            regime_exposures={
                "strong_bull": 1.0,
                "recovery_bull": 1.0,
                "range_bound": 1.0,
                "correction_bear": 1.0,
                "systemic_bear": 0.0,
            },
            relative_score_fallback_ticker="0050.TW",
            relative_score_fallback_defense_rule="ma245",
            min_score_over_fallback=0.40,
        ),
        RegimeModeSwitchVariant(
            name="daily_margin0050_40_ma250_defense",
            regime_modes={
                "strong_bull": MODE_DAILY,
                "recovery_bull": MODE_DAILY,
                "range_bound": MODE_DAILY,
                "correction_bear": MODE_DAILY,
                "systemic_bear": MODE_CASH,
            },
            regime_exposures={
                "strong_bull": 1.0,
                "recovery_bull": 1.0,
                "range_bound": 1.0,
                "correction_bear": 1.0,
                "systemic_bear": 0.0,
            },
            relative_score_fallback_ticker="0050.TW",
            relative_score_fallback_defense_rule="ma250",
            min_score_over_fallback=0.40,
        ),
        RegimeModeSwitchVariant(
            name="daily_margin0050_50_ma250_defense",
            regime_modes={
                "strong_bull": MODE_DAILY,
                "recovery_bull": MODE_DAILY,
                "range_bound": MODE_DAILY,
                "correction_bear": MODE_DAILY,
                "systemic_bear": MODE_CASH,
            },
            regime_exposures={
                "strong_bull": 1.0,
                "recovery_bull": 1.0,
                "range_bound": 1.0,
                "correction_bear": 1.0,
                "systemic_bear": 0.0,
            },
            relative_score_fallback_ticker="0050.TW",
            relative_score_fallback_defense_rule="ma250",
            min_score_over_fallback=0.50,
        ),
        RegimeModeSwitchVariant(
            name="daily_adaptive_margin0050_5_20_40_ma250_defense",
            regime_modes={
                "strong_bull": MODE_DAILY,
                "recovery_bull": MODE_DAILY,
                "range_bound": MODE_DAILY,
                "correction_bear": MODE_DAILY,
                "systemic_bear": MODE_CASH,
            },
            regime_exposures={
                "strong_bull": 1.0,
                "recovery_bull": 1.0,
                "range_bound": 1.0,
                "correction_bear": 1.0,
                "systemic_bear": 0.0,
            },
            relative_score_fallback_ticker="0050.TW",
            relative_score_fallback_defense_rule="ma250",
            min_score_over_fallback_by_regime={
                "strong_bull": 0.05,
                "recovery_bull": 0.20,
                "range_bound": 0.40,
                "correction_bear": 0.40,
                "systemic_bear": 0.40,
            },
        ),
        RegimeModeSwitchVariant(
            name="daily_adaptive_margin0050_10_25_40_ma250_defense",
            regime_modes={
                "strong_bull": MODE_DAILY,
                "recovery_bull": MODE_DAILY,
                "range_bound": MODE_DAILY,
                "correction_bear": MODE_DAILY,
                "systemic_bear": MODE_CASH,
            },
            regime_exposures={
                "strong_bull": 1.0,
                "recovery_bull": 1.0,
                "range_bound": 1.0,
                "correction_bear": 1.0,
                "systemic_bear": 0.0,
            },
            relative_score_fallback_ticker="0050.TW",
            relative_score_fallback_defense_rule="ma250",
            min_score_over_fallback_by_regime={
                "strong_bull": 0.10,
                "recovery_bull": 0.25,
                "range_bound": 0.40,
                "correction_bear": 0.40,
                "systemic_bear": 0.40,
            },
        ),
        RegimeModeSwitchVariant(
            name="daily_adaptive_margin0050_20_30_40_ma250_defense",
            regime_modes={
                "strong_bull": MODE_DAILY,
                "recovery_bull": MODE_DAILY,
                "range_bound": MODE_DAILY,
                "correction_bear": MODE_DAILY,
                "systemic_bear": MODE_CASH,
            },
            regime_exposures={
                "strong_bull": 1.0,
                "recovery_bull": 1.0,
                "range_bound": 1.0,
                "correction_bear": 1.0,
                "systemic_bear": 0.0,
            },
            relative_score_fallback_ticker="0050.TW",
            relative_score_fallback_defense_rule="ma250",
            min_score_over_fallback_by_regime={
                "strong_bull": 0.20,
                "recovery_bull": 0.30,
                "range_bound": 0.40,
                "correction_bear": 0.40,
                "systemic_bear": 0.40,
            },
        ),
        RegimeModeSwitchVariant(
            name="five_regime_v1_daily_weekly_ma250_cash",
            regime_modes={
                "strong_bull": MODE_DAILY,
                "recovery_bull": MODE_DAILY,
                "range_bound": MODE_WEEKLY,
                "correction_bear": MODE_0050_DEFENSE,
                "systemic_bear": MODE_CASH,
            },
            regime_exposures={
                "strong_bull": 1.0,
                "recovery_bull": 1.0,
                "range_bound": 1.0,
                "correction_bear": 1.0,
                "systemic_bear": 0.0,
            },
            defense_rule_by_regime={"correction_bear": "ma250"},
            defense_risk_off_exposure_by_regime={"correction_bear": 0.0},
        ),
        RegimeModeSwitchVariant(
            name="five_regime_v2_strong_daily_else_ma250",
            regime_modes={
                "strong_bull": MODE_DAILY,
                "recovery_bull": MODE_0050_DEFENSE,
                "range_bound": MODE_0050_DEFENSE,
                "correction_bear": MODE_0050_DEFENSE,
                "systemic_bear": MODE_CASH,
            },
            regime_exposures={
                "strong_bull": 1.0,
                "recovery_bull": 1.0,
                "range_bound": 1.0,
                "correction_bear": 1.0,
                "systemic_bear": 0.0,
            },
            defense_rule_by_regime={"recovery_bull": "ma250", "range_bound": "ma250", "correction_bear": "ma250"},
            defense_risk_off_exposure_by_regime={"recovery_bull": 0.0, "range_bound": 0.0, "correction_bear": 0.0},
        ),
        RegimeModeSwitchVariant(
            name="five_regime_v2_strong_weekly_else_ma250",
            regime_modes={
                "strong_bull": MODE_WEEKLY,
                "recovery_bull": MODE_0050_DEFENSE,
                "range_bound": MODE_0050_DEFENSE,
                "correction_bear": MODE_0050_DEFENSE,
                "systemic_bear": MODE_CASH,
            },
            regime_exposures={
                "strong_bull": 1.0,
                "recovery_bull": 1.0,
                "range_bound": 1.0,
                "correction_bear": 1.0,
                "systemic_bear": 0.0,
            },
            defense_rule_by_regime={"recovery_bull": "ma250", "range_bound": "ma250", "correction_bear": "ma250"},
            defense_risk_off_exposure_by_regime={"recovery_bull": 0.0, "range_bound": 0.0, "correction_bear": 0.0},
        ),
        RegimeModeSwitchVariant(
            name="five_regime_v2_all_ma240_until_systemic",
            regime_modes={
                "strong_bull": MODE_0050_DEFENSE,
                "recovery_bull": MODE_0050_DEFENSE,
                "range_bound": MODE_0050_DEFENSE,
                "correction_bear": MODE_0050_DEFENSE,
                "systemic_bear": MODE_CASH,
            },
            regime_exposures={
                "strong_bull": 1.0,
                "recovery_bull": 1.0,
                "range_bound": 1.0,
                "correction_bear": 1.0,
                "systemic_bear": 0.0,
            },
            defense_rule_by_regime={
                "strong_bull": "ma240",
                "recovery_bull": "ma240",
                "range_bound": "ma240",
                "correction_bear": "ma240",
            },
            defense_risk_off_exposure_by_regime={
                "strong_bull": 0.0,
                "recovery_bull": 0.0,
                "range_bound": 0.0,
                "correction_bear": 0.0,
            },
        ),
        RegimeModeSwitchVariant(
            name="five_regime_v2_all_ma245_until_systemic",
            regime_modes={
                "strong_bull": MODE_0050_DEFENSE,
                "recovery_bull": MODE_0050_DEFENSE,
                "range_bound": MODE_0050_DEFENSE,
                "correction_bear": MODE_0050_DEFENSE,
                "systemic_bear": MODE_CASH,
            },
            regime_exposures={
                "strong_bull": 1.0,
                "recovery_bull": 1.0,
                "range_bound": 1.0,
                "correction_bear": 1.0,
                "systemic_bear": 0.0,
            },
            defense_rule_by_regime={
                "strong_bull": "ma245",
                "recovery_bull": "ma245",
                "range_bound": "ma245",
                "correction_bear": "ma245",
            },
            defense_risk_off_exposure_by_regime={
                "strong_bull": 0.0,
                "recovery_bull": 0.0,
                "range_bound": 0.0,
                "correction_bear": 0.0,
            },
        ),
        RegimeModeSwitchVariant(
            name="five_regime_v2_all_ma250_until_systemic",
            regime_modes={
                "strong_bull": MODE_0050_DEFENSE,
                "recovery_bull": MODE_0050_DEFENSE,
                "range_bound": MODE_0050_DEFENSE,
                "correction_bear": MODE_0050_DEFENSE,
                "systemic_bear": MODE_CASH,
            },
            regime_exposures={
                "strong_bull": 1.0,
                "recovery_bull": 1.0,
                "range_bound": 1.0,
                "correction_bear": 1.0,
                "systemic_bear": 0.0,
            },
            defense_rule_by_regime={
                "strong_bull": "ma250",
                "recovery_bull": "ma250",
                "range_bound": "ma250",
                "correction_bear": "ma250",
            },
            defense_risk_off_exposure_by_regime={
                "strong_bull": 0.0,
                "recovery_bull": 0.0,
                "range_bound": 0.0,
                "correction_bear": 0.0,
            },
        ),
        RegimeModeSwitchVariant(
            name="five_regime_v1_daily_weekly_ma220_cash",
            regime_modes={
                "strong_bull": MODE_DAILY,
                "recovery_bull": MODE_DAILY,
                "range_bound": MODE_WEEKLY,
                "correction_bear": MODE_0050_DEFENSE,
                "systemic_bear": MODE_CASH,
            },
            regime_exposures={
                "strong_bull": 1.0,
                "recovery_bull": 1.0,
                "range_bound": 1.0,
                "correction_bear": 1.0,
                "systemic_bear": 0.0,
            },
            defense_rule_by_regime={"correction_bear": "ma220"},
            defense_risk_off_exposure_by_regime={"correction_bear": 0.0},
        ),
        RegimeModeSwitchVariant(
            name="five_regime_v1_daily_ma250_range_bear",
            regime_modes={
                "strong_bull": MODE_DAILY,
                "recovery_bull": MODE_DAILY,
                "range_bound": MODE_0050_DEFENSE,
                "correction_bear": MODE_0050_DEFENSE,
                "systemic_bear": MODE_CASH,
            },
            regime_exposures={
                "strong_bull": 1.0,
                "recovery_bull": 1.0,
                "range_bound": 1.0,
                "correction_bear": 1.0,
                "systemic_bear": 0.0,
            },
            defense_rule_by_regime={"range_bound": "ma250", "correction_bear": "ma250"},
            defense_risk_off_exposure_by_regime={"range_bound": 0.0, "correction_bear": 0.0},
        ),
        RegimeModeSwitchVariant(
            name="five_regime_v1_daily_ma250_10pct_range_bear",
            regime_modes={
                "strong_bull": MODE_DAILY,
                "recovery_bull": MODE_DAILY,
                "range_bound": MODE_0050_DEFENSE,
                "correction_bear": MODE_0050_DEFENSE,
                "systemic_bear": MODE_CASH,
            },
            regime_exposures={
                "strong_bull": 1.0,
                "recovery_bull": 1.0,
                "range_bound": 1.0,
                "correction_bear": 1.0,
                "systemic_bear": 0.0,
            },
            defense_rule_by_regime={"range_bound": "ma250", "correction_bear": "ma250"},
            defense_risk_off_exposure_by_regime={"range_bound": 0.1, "correction_bear": 0.1},
        ),
        RegimeModeSwitchVariant(
            name="daily_until_bear_cash",
            regime_modes={
                "strong_bull": MODE_DAILY,
                "recovery_bull": MODE_DAILY,
                "range_bound": MODE_DAILY,
                "correction_bear": MODE_CASH,
                "systemic_bear": MODE_CASH,
            },
            regime_exposures={
                "strong_bull": 1.0,
                "recovery_bull": 1.0,
                "range_bound": 1.0,
                "correction_bear": 0.0,
                "systemic_bear": 0.0,
            },
        ),
        RegimeModeSwitchVariant(
            name="daily_bull_weekly_range_cash_bear",
            regime_modes={
                "strong_bull": MODE_DAILY,
                "recovery_bull": MODE_DAILY,
                "range_bound": MODE_WEEKLY,
                "correction_bear": MODE_CASH,
                "systemic_bear": MODE_CASH,
            },
            regime_exposures={
                "strong_bull": 1.0,
                "recovery_bull": 1.0,
                "range_bound": 1.0,
                "correction_bear": 0.0,
                "systemic_bear": 0.0,
            },
        ),
        RegimeModeSwitchVariant(
            name="daily_strong_weekly_recovery_cash_bear",
            regime_modes={
                "strong_bull": MODE_DAILY,
                "recovery_bull": MODE_WEEKLY,
                "range_bound": MODE_WEEKLY,
                "correction_bear": MODE_CASH,
                "systemic_bear": MODE_CASH,
            },
            regime_exposures={
                "strong_bull": 1.0,
                "recovery_bull": 1.0,
                "range_bound": 1.0,
                "correction_bear": 0.0,
                "systemic_bear": 0.0,
            },
        ),
        RegimeModeSwitchVariant(
            name="daily_bull_cash_range_bear",
            regime_modes={
                "strong_bull": MODE_DAILY,
                "recovery_bull": MODE_DAILY,
                "range_bound": MODE_CASH,
                "correction_bear": MODE_CASH,
                "systemic_bear": MODE_CASH,
            },
            regime_exposures={
                "strong_bull": 1.0,
                "recovery_bull": 1.0,
                "range_bound": 0.0,
                "correction_bear": 0.0,
                "systemic_bear": 0.0,
            },
        ),
        RegimeModeSwitchVariant(
            name="daily_strong_only_cash_other",
            regime_modes={
                "strong_bull": MODE_DAILY,
                "recovery_bull": MODE_CASH,
                "range_bound": MODE_CASH,
                "correction_bear": MODE_CASH,
                "systemic_bear": MODE_CASH,
            },
            regime_exposures={
                "strong_bull": 1.0,
                "recovery_bull": 0.0,
                "range_bound": 0.0,
                "correction_bear": 0.0,
                "systemic_bear": 0.0,
            },
        ),
        RegimeModeSwitchVariant(
            name="daily_strong_else_0050",
            regime_modes={
                "strong_bull": MODE_DAILY,
                "recovery_bull": MODE_CASH,
                "range_bound": MODE_CASH,
                "correction_bear": MODE_CASH,
                "systemic_bear": MODE_CASH,
            },
            regime_exposures={
                "strong_bull": 1.0,
                "recovery_bull": 0.0,
                "range_bound": 0.0,
                "correction_bear": 0.0,
                "systemic_bear": 0.0,
            },
            fallback_ticker="0050.TW",
            fallback_exposure=1.0,
        ),
        RegimeModeSwitchVariant(
            name="daily_bull_else_0050",
            regime_modes={
                "strong_bull": MODE_DAILY,
                "recovery_bull": MODE_DAILY,
                "range_bound": MODE_CASH,
                "correction_bear": MODE_CASH,
                "systemic_bear": MODE_CASH,
            },
            regime_exposures={
                "strong_bull": 1.0,
                "recovery_bull": 1.0,
                "range_bound": 0.0,
                "correction_bear": 0.0,
                "systemic_bear": 0.0,
            },
            fallback_ticker="0050.TW",
            fallback_exposure=1.0,
        ),
        RegimeModeSwitchVariant(
            name="daily_until_bear_else_0050",
            regime_modes={
                "strong_bull": MODE_DAILY,
                "recovery_bull": MODE_DAILY,
                "range_bound": MODE_DAILY,
                "correction_bear": MODE_CASH,
                "systemic_bear": MODE_CASH,
            },
            regime_exposures={
                "strong_bull": 1.0,
                "recovery_bull": 1.0,
                "range_bound": 1.0,
                "correction_bear": 0.0,
                "systemic_bear": 0.0,
            },
            fallback_ticker="0050.TW",
            fallback_exposure=1.0,
        ),
        RegimeModeSwitchVariant(
            name="daily_margin0050_5",
            regime_modes={
                "strong_bull": MODE_DAILY,
                "recovery_bull": MODE_DAILY,
                "range_bound": MODE_DAILY,
                "correction_bear": MODE_DAILY,
                "systemic_bear": MODE_CASH,
            },
            regime_exposures={
                "strong_bull": 1.0,
                "recovery_bull": 1.0,
                "range_bound": 1.0,
                "correction_bear": 1.0,
                "systemic_bear": 0.0,
            },
            fallback_ticker="0050.TW",
            fallback_exposure=1.0,
            relative_score_fallback_ticker="0050.TW",
            min_score_over_fallback=0.05,
        ),
        RegimeModeSwitchVariant(
            name="daily_margin0050_10",
            regime_modes={
                "strong_bull": MODE_DAILY,
                "recovery_bull": MODE_DAILY,
                "range_bound": MODE_DAILY,
                "correction_bear": MODE_DAILY,
                "systemic_bear": MODE_CASH,
            },
            regime_exposures={
                "strong_bull": 1.0,
                "recovery_bull": 1.0,
                "range_bound": 1.0,
                "correction_bear": 1.0,
                "systemic_bear": 0.0,
            },
            fallback_ticker="0050.TW",
            fallback_exposure=1.0,
            relative_score_fallback_ticker="0050.TW",
            min_score_over_fallback=0.10,
        ),
        RegimeModeSwitchVariant(
            name="daily_margin0050_15",
            regime_modes={
                "strong_bull": MODE_DAILY,
                "recovery_bull": MODE_DAILY,
                "range_bound": MODE_DAILY,
                "correction_bear": MODE_DAILY,
                "systemic_bear": MODE_CASH,
            },
            regime_exposures={
                "strong_bull": 1.0,
                "recovery_bull": 1.0,
                "range_bound": 1.0,
                "correction_bear": 1.0,
                "systemic_bear": 0.0,
            },
            fallback_ticker="0050.TW",
            fallback_exposure=1.0,
            relative_score_fallback_ticker="0050.TW",
            min_score_over_fallback=0.15,
        ),
        RegimeModeSwitchVariant(
            name="daily_margin0050_20",
            regime_modes={
                "strong_bull": MODE_DAILY,
                "recovery_bull": MODE_DAILY,
                "range_bound": MODE_DAILY,
                "correction_bear": MODE_DAILY,
                "systemic_bear": MODE_CASH,
            },
            regime_exposures={
                "strong_bull": 1.0,
                "recovery_bull": 1.0,
                "range_bound": 1.0,
                "correction_bear": 1.0,
                "systemic_bear": 0.0,
            },
            fallback_ticker="0050.TW",
            fallback_exposure=1.0,
            relative_score_fallback_ticker="0050.TW",
            min_score_over_fallback=0.20,
        ),
        RegimeModeSwitchVariant(
            name="daily_margin0050_25",
            regime_modes={
                "strong_bull": MODE_DAILY,
                "recovery_bull": MODE_DAILY,
                "range_bound": MODE_DAILY,
                "correction_bear": MODE_DAILY,
                "systemic_bear": MODE_CASH,
            },
            regime_exposures={
                "strong_bull": 1.0,
                "recovery_bull": 1.0,
                "range_bound": 1.0,
                "correction_bear": 1.0,
                "systemic_bear": 0.0,
            },
            fallback_ticker="0050.TW",
            fallback_exposure=1.0,
            relative_score_fallback_ticker="0050.TW",
            min_score_over_fallback=0.25,
        ),
        RegimeModeSwitchVariant(
            name="daily_margin0050_30",
            regime_modes={
                "strong_bull": MODE_DAILY,
                "recovery_bull": MODE_DAILY,
                "range_bound": MODE_DAILY,
                "correction_bear": MODE_DAILY,
                "systemic_bear": MODE_CASH,
            },
            regime_exposures={
                "strong_bull": 1.0,
                "recovery_bull": 1.0,
                "range_bound": 1.0,
                "correction_bear": 1.0,
                "systemic_bear": 0.0,
            },
            fallback_ticker="0050.TW",
            fallback_exposure=1.0,
            relative_score_fallback_ticker="0050.TW",
            min_score_over_fallback=0.30,
        ),
        RegimeModeSwitchVariant(
            name="daily_adaptive_margin0050_5_15_30",
            regime_modes={
                "strong_bull": MODE_DAILY,
                "recovery_bull": MODE_DAILY,
                "range_bound": MODE_DAILY,
                "correction_bear": MODE_DAILY,
                "systemic_bear": MODE_CASH,
            },
            regime_exposures={
                "strong_bull": 1.0,
                "recovery_bull": 1.0,
                "range_bound": 1.0,
                "correction_bear": 1.0,
                "systemic_bear": 0.0,
            },
            fallback_ticker="0050.TW",
            fallback_exposure=1.0,
            relative_score_fallback_ticker="0050.TW",
            min_score_over_fallback_by_regime={
                "strong_bull": 0.05,
                "recovery_bull": 0.15,
                "range_bound": 0.30,
                "correction_bear": 0.30,
                "systemic_bear": 0.30,
            },
        ),
        RegimeModeSwitchVariant(
            name="daily_adaptive_margin0050_10_20_30",
            regime_modes={
                "strong_bull": MODE_DAILY,
                "recovery_bull": MODE_DAILY,
                "range_bound": MODE_DAILY,
                "correction_bear": MODE_DAILY,
                "systemic_bear": MODE_CASH,
            },
            regime_exposures={
                "strong_bull": 1.0,
                "recovery_bull": 1.0,
                "range_bound": 1.0,
                "correction_bear": 1.0,
                "systemic_bear": 0.0,
            },
            fallback_ticker="0050.TW",
            fallback_exposure=1.0,
            relative_score_fallback_ticker="0050.TW",
            min_score_over_fallback_by_regime={
                "strong_bull": 0.10,
                "recovery_bull": 0.20,
                "range_bound": 0.30,
                "correction_bear": 0.30,
                "systemic_bear": 0.30,
            },
        ),
        RegimeModeSwitchVariant(
            name="daily_adaptive_margin0050_15_25_30",
            regime_modes={
                "strong_bull": MODE_DAILY,
                "recovery_bull": MODE_DAILY,
                "range_bound": MODE_DAILY,
                "correction_bear": MODE_DAILY,
                "systemic_bear": MODE_CASH,
            },
            regime_exposures={
                "strong_bull": 1.0,
                "recovery_bull": 1.0,
                "range_bound": 1.0,
                "correction_bear": 1.0,
                "systemic_bear": 0.0,
            },
            fallback_ticker="0050.TW",
            fallback_exposure=1.0,
            relative_score_fallback_ticker="0050.TW",
            min_score_over_fallback_by_regime={
                "strong_bull": 0.15,
                "recovery_bull": 0.25,
                "range_bound": 0.30,
                "correction_bear": 0.30,
                "systemic_bear": 0.30,
            },
        ),
        RegimeModeSwitchVariant(
            name="daily_00631L_gate_margin5_30",
            regime_modes={
                "strong_bull": MODE_DAILY,
                "recovery_bull": MODE_DAILY,
                "range_bound": MODE_DAILY,
                "correction_bear": MODE_DAILY,
                "systemic_bear": MODE_CASH,
            },
            regime_exposures={
                "strong_bull": 1.0,
                "recovery_bull": 1.0,
                "range_bound": 1.0,
                "correction_bear": 1.0,
                "systemic_bear": 0.0,
            },
            fallback_ticker="0050.TW",
            fallback_exposure=1.0,
            relative_score_fallback_ticker="0050.TW",
            margin_gate_ticker="00631L.TW",
            gated_on_margin=0.05,
            gated_off_margin=0.30,
        ),
        RegimeModeSwitchVariant(
            name="daily_00631L_gate_margin10_30",
            regime_modes={
                "strong_bull": MODE_DAILY,
                "recovery_bull": MODE_DAILY,
                "range_bound": MODE_DAILY,
                "correction_bear": MODE_DAILY,
                "systemic_bear": MODE_CASH,
            },
            regime_exposures={
                "strong_bull": 1.0,
                "recovery_bull": 1.0,
                "range_bound": 1.0,
                "correction_bear": 1.0,
                "systemic_bear": 0.0,
            },
            fallback_ticker="0050.TW",
            fallback_exposure=1.0,
            relative_score_fallback_ticker="0050.TW",
            margin_gate_ticker="00631L.TW",
            gated_on_margin=0.10,
            gated_off_margin=0.30,
        ),
        RegimeModeSwitchVariant(
            name="daily_until_systemic_cash",
            regime_modes={
                "strong_bull": MODE_DAILY,
                "recovery_bull": MODE_DAILY,
                "range_bound": MODE_DAILY,
                "correction_bear": MODE_DAILY,
                "systemic_bear": MODE_CASH,
            },
            regime_exposures={
                "strong_bull": 1.0,
                "recovery_bull": 1.0,
                "range_bound": 1.0,
                "correction_bear": 1.0,
                "systemic_bear": 0.0,
            },
        ),
        RegimeModeSwitchVariant(
            name="daily_until_systemic_stop12_cd5",
            regime_modes={
                "strong_bull": MODE_DAILY,
                "recovery_bull": MODE_DAILY,
                "range_bound": MODE_DAILY,
                "correction_bear": MODE_DAILY,
                "systemic_bear": MODE_CASH,
            },
            regime_exposures={
                "strong_bull": 1.0,
                "recovery_bull": 1.0,
                "range_bound": 1.0,
                "correction_bear": 1.0,
                "systemic_bear": 0.0,
            },
            portfolio_stop_drawdown_pct=0.12,
            portfolio_stop_cooldown_days=5,
        ),
        RegimeModeSwitchVariant(
            name="daily_until_systemic_stop12_cd5_trend60",
            regime_modes={
                "strong_bull": MODE_DAILY,
                "recovery_bull": MODE_DAILY,
                "range_bound": MODE_DAILY,
                "correction_bear": MODE_DAILY,
                "systemic_bear": MODE_CASH,
            },
            regime_exposures={
                "strong_bull": 1.0,
                "recovery_bull": 1.0,
                "range_bound": 1.0,
                "correction_bear": 1.0,
                "systemic_bear": 0.0,
            },
            portfolio_stop_drawdown_pct=0.12,
            portfolio_stop_cooldown_days=5,
            candidate_trend_filter="trend60_positive20",
        ),
        RegimeModeSwitchVariant(
            name="daily_until_systemic_stop12_cd5_trend120",
            regime_modes={
                "strong_bull": MODE_DAILY,
                "recovery_bull": MODE_DAILY,
                "range_bound": MODE_DAILY,
                "correction_bear": MODE_DAILY,
                "systemic_bear": MODE_CASH,
            },
            regime_exposures={
                "strong_bull": 1.0,
                "recovery_bull": 1.0,
                "range_bound": 1.0,
                "correction_bear": 1.0,
                "systemic_bear": 0.0,
            },
            portfolio_stop_drawdown_pct=0.12,
            portfolio_stop_cooldown_days=5,
            candidate_trend_filter="trend120_positive20",
        ),
        RegimeModeSwitchVariant(
            name="daily_market_guard_dd5_ma60",
            regime_modes={
                "strong_bull": MODE_DAILY,
                "recovery_bull": MODE_DAILY,
                "range_bound": MODE_DAILY,
                "correction_bear": MODE_DAILY,
                "systemic_bear": MODE_CASH,
            },
            regime_exposures={
                "strong_bull": 1.0,
                "recovery_bull": 1.0,
                "range_bound": 1.0,
                "correction_bear": 1.0,
                "systemic_bear": 0.0,
            },
            market_risk_off_filter="dd5_ma60_ret20",
        ),
        RegimeModeSwitchVariant(
            name="daily_market_guard_dd8_ma60",
            regime_modes={
                "strong_bull": MODE_DAILY,
                "recovery_bull": MODE_DAILY,
                "range_bound": MODE_DAILY,
                "correction_bear": MODE_DAILY,
                "systemic_bear": MODE_CASH,
            },
            regime_exposures={
                "strong_bull": 1.0,
                "recovery_bull": 1.0,
                "range_bound": 1.0,
                "correction_bear": 1.0,
                "systemic_bear": 0.0,
            },
            market_risk_off_filter="dd8_ma60_ret20",
        ),
        RegimeModeSwitchVariant(
            name="daily_market_guard_ma120",
            regime_modes={
                "strong_bull": MODE_DAILY,
                "recovery_bull": MODE_DAILY,
                "range_bound": MODE_DAILY,
                "correction_bear": MODE_DAILY,
                "systemic_bear": MODE_CASH,
            },
            regime_exposures={
                "strong_bull": 1.0,
                "recovery_bull": 1.0,
                "range_bound": 1.0,
                "correction_bear": 1.0,
                "systemic_bear": 0.0,
            },
            market_risk_off_filter="ma120_ret20",
        ),
        RegimeModeSwitchVariant(
            name="daily_until_bear_stop12_cd5",
            regime_modes={
                "strong_bull": MODE_DAILY,
                "recovery_bull": MODE_DAILY,
                "range_bound": MODE_DAILY,
                "correction_bear": MODE_CASH,
                "systemic_bear": MODE_CASH,
            },
            regime_exposures={
                "strong_bull": 1.0,
                "recovery_bull": 1.0,
                "range_bound": 1.0,
                "correction_bear": 0.0,
                "systemic_bear": 0.0,
            },
            portfolio_stop_drawdown_pct=0.12,
            portfolio_stop_cooldown_days=5,
        ),
        RegimeModeSwitchVariant(
            name="daily_until_bear_stop8_cd3",
            regime_modes={
                "strong_bull": MODE_DAILY,
                "recovery_bull": MODE_DAILY,
                "range_bound": MODE_DAILY,
                "correction_bear": MODE_CASH,
                "systemic_bear": MODE_CASH,
            },
            regime_exposures={
                "strong_bull": 1.0,
                "recovery_bull": 1.0,
                "range_bound": 1.0,
                "correction_bear": 0.0,
                "systemic_bear": 0.0,
            },
            portfolio_stop_drawdown_pct=0.08,
            portfolio_stop_cooldown_days=3,
        ),
        RegimeModeSwitchVariant(
            name="daily_bear_10pct_cash_systemic",
            regime_modes={
                "strong_bull": MODE_DAILY,
                "recovery_bull": MODE_DAILY,
                "range_bound": MODE_DAILY,
                "correction_bear": MODE_DAILY,
                "systemic_bear": MODE_CASH,
            },
            regime_exposures={
                "strong_bull": 1.0,
                "recovery_bull": 1.0,
                "range_bound": 1.0,
                "correction_bear": 0.1,
                "systemic_bear": 0.0,
            },
        ),
    )


def asymmetric_attack_defense_variants() -> tuple[RegimeModeSwitchVariant, ...]:
    daily_modes = {
        "strong_bull": MODE_DAILY,
        "recovery_bull": MODE_DAILY,
        "range_bound": MODE_DAILY,
        "correction_bear": MODE_DAILY,
        "systemic_bear": MODE_CASH,
    }
    full_exposure = {
        "strong_bull": 1.0,
        "recovery_bull": 1.0,
        "range_bound": 1.0,
        "correction_bear": 1.0,
        "systemic_bear": 0.0,
    }
    variants: list[RegimeModeSwitchVariant] = []
    for confirmation_days in (0, 3, 5):
        variants.extend(
            [
                RegimeModeSwitchVariant(
                    name=f"asym_daily_fastcash_reentry{confirmation_days}",
                    regime_modes=daily_modes,
                    regime_exposures=full_exposure,
                    market_risk_off_filter="ma60_ret20",
                    market_risk_off_mode=MODE_CASH,
                    market_risk_off_exit_confirmation_days=confirmation_days,
                ),
                RegimeModeSwitchVariant(
                    name=f"asym_daily_fastdefense245_reentry{confirmation_days}",
                    regime_modes=daily_modes,
                    regime_exposures=full_exposure,
                    market_risk_off_filter="ma60_ret20",
                    market_risk_off_mode=MODE_0050_DEFENSE,
                    market_risk_off_defense_rule="ma245",
                    market_risk_off_exit_confirmation_days=confirmation_days,
                ),
            ]
        )

    for confirmation_days in (3, 5):
        variants.extend(
            [
                RegimeModeSwitchVariant(
                    name=f"asym_adaptive_0_5_30_40_def245_reentry{confirmation_days}",
                    regime_modes=daily_modes,
                    regime_exposures=full_exposure,
                    market_risk_off_filter="ma60_ret20",
                    market_risk_off_mode=MODE_0050_DEFENSE,
                    market_risk_off_defense_rule="ma245",
                    market_risk_off_exit_confirmation_days=confirmation_days,
                    relative_score_fallback_ticker="0050.TW",
                    relative_score_fallback_defense_rule="ma245",
                    min_score_over_fallback_by_regime={
                        "strong_bull": 0.0,
                        "recovery_bull": 0.05,
                        "range_bound": 0.30,
                        "correction_bear": 0.40,
                        "systemic_bear": 0.40,
                    },
                ),
                RegimeModeSwitchVariant(
                    name=f"asym_adaptive_0_10_40_40_def245_reentry{confirmation_days}",
                    regime_modes=daily_modes,
                    regime_exposures=full_exposure,
                    market_risk_off_filter="ma60_ret20",
                    market_risk_off_mode=MODE_0050_DEFENSE,
                    market_risk_off_defense_rule="ma245",
                    market_risk_off_exit_confirmation_days=confirmation_days,
                    relative_score_fallback_ticker="0050.TW",
                    relative_score_fallback_defense_rule="ma245",
                    min_score_over_fallback_by_regime={
                        "strong_bull": 0.0,
                        "recovery_bull": 0.10,
                        "range_bound": 0.40,
                        "correction_bear": 0.40,
                        "systemic_bear": 0.40,
                    },
                ),
            ]
        )
    return tuple(variants)


def strategy_health_attack_defense_variants() -> tuple[RegimeModeSwitchVariant, ...]:
    daily_modes = {
        "strong_bull": MODE_DAILY,
        "recovery_bull": MODE_DAILY,
        "range_bound": MODE_DAILY,
        "correction_bear": MODE_DAILY,
        "systemic_bear": MODE_CASH,
    }
    full_exposure = {
        "strong_bull": 1.0,
        "recovery_bull": 1.0,
        "range_bound": 1.0,
        "correction_bear": 1.0,
        "systemic_bear": 0.0,
    }
    variants: list[RegimeModeSwitchVariant] = []
    for lookback_days, minimum_excess in ((40, 0.0), (60, 0.0), (60, 0.05)):
        for confirmation_days in (3, 5, 10):
            variants.append(
                RegimeModeSwitchVariant(
                    name=(
                        f"health{lookback_days}_min{int(minimum_excess * 100)}"
                        f"_fastdef245_recovery{confirmation_days}"
                    ),
                    regime_modes=daily_modes,
                    regime_exposures=full_exposure,
                    market_risk_off_filter="ma60_ret20",
                    market_risk_off_mode=MODE_0050_DEFENSE,
                    market_risk_off_defense_rule="ma245",
                    market_risk_off_exit_confirmation_days=3,
                    daily_health_lookback_days=lookback_days,
                    daily_health_min_excess_return=minimum_excess,
                    daily_health_fail_mode=MODE_0050_DEFENSE,
                    daily_health_fail_defense_rule="ma245",
                    daily_health_recovery_confirmation_days=confirmation_days,
                )
            )
    return tuple(variants)


def stop_latch_attack_defense_variants() -> tuple[RegimeModeSwitchVariant, ...]:
    daily_modes = {
        "strong_bull": MODE_DAILY,
        "recovery_bull": MODE_DAILY,
        "range_bound": MODE_DAILY,
        "correction_bear": MODE_DAILY,
        "systemic_bear": MODE_CASH,
    }
    full_exposure = {
        "strong_bull": 1.0,
        "recovery_bull": 1.0,
        "range_bound": 1.0,
        "correction_bear": 1.0,
        "systemic_bear": 0.0,
    }
    variants: list[RegimeModeSwitchVariant] = []
    for stop_pct in (0.08, 0.10, 0.12):
        for release_days in (3, 5):
            for latch_mode in (MODE_CASH, MODE_0050_DEFENSE):
                mode_label = "cash" if latch_mode == MODE_CASH else "def245"
                variants.append(
                    RegimeModeSwitchVariant(
                        name=f"stop{int(stop_pct * 100)}_latch_{mode_label}_strong{release_days}",
                        regime_modes=daily_modes,
                        regime_exposures=full_exposure,
                        portfolio_stop_drawdown_pct=stop_pct,
                        portfolio_stop_latch_mode=latch_mode,
                        portfolio_stop_latch_defense_rule="ma245",
                        portfolio_stop_release_filter="strong_bull",
                        portfolio_stop_release_confirmation_days=release_days,
                    )
                )
    return tuple(variants)


def stop_latch_defense_sweep_variants() -> tuple[RegimeModeSwitchVariant, ...]:
    daily_modes = {
        "strong_bull": MODE_DAILY,
        "recovery_bull": MODE_DAILY,
        "range_bound": MODE_DAILY,
        "correction_bear": MODE_DAILY,
        "systemic_bear": MODE_CASH,
    }
    full_exposure = {
        "strong_bull": 1.0,
        "recovery_bull": 1.0,
        "range_bound": 1.0,
        "correction_bear": 1.0,
        "systemic_bear": 0.0,
    }
    variants: list[RegimeModeSwitchVariant] = []
    for stop_pct in (0.10, 0.12):
        for defense_window in (120, 180, 200, 220, 245):
            variants.append(
                RegimeModeSwitchVariant(
                    name=f"stop{int(stop_pct * 100)}_latch_def{defense_window}_strong5",
                    regime_modes=daily_modes,
                    regime_exposures=full_exposure,
                    portfolio_stop_drawdown_pct=stop_pct,
                    portfolio_stop_latch_mode=MODE_0050_DEFENSE,
                    portfolio_stop_latch_defense_rule=f"ma{defense_window}",
                    portfolio_stop_release_filter="strong_bull",
                    portfolio_stop_release_confirmation_days=5,
                )
            )
    variants.append(
        RegimeModeSwitchVariant(
            name="stop12_latch_cash_strong5_reference",
            regime_modes=daily_modes,
            regime_exposures=full_exposure,
            portfolio_stop_drawdown_pct=0.12,
            portfolio_stop_latch_mode=MODE_CASH,
            portfolio_stop_release_filter="strong_bull",
            portfolio_stop_release_confirmation_days=5,
        )
    )
    return tuple(variants)


def stop_latch_health_release_variants() -> tuple[RegimeModeSwitchVariant, ...]:
    daily_modes = {
        "strong_bull": MODE_DAILY,
        "recovery_bull": MODE_DAILY,
        "range_bound": MODE_DAILY,
        "correction_bear": MODE_DAILY,
        "systemic_bear": MODE_CASH,
    }
    full_exposure = {
        "strong_bull": 1.0,
        "recovery_bull": 1.0,
        "range_bound": 1.0,
        "correction_bear": 1.0,
        "systemic_bear": 0.0,
    }
    variants: list[RegimeModeSwitchVariant] = []
    for stop_pct in (0.08, 0.10, 0.12):
        for defense_window in (200, 245):
            for health_lookback in (40, 60):
                variants.append(
                    RegimeModeSwitchVariant(
                        name=(
                            f"stop{int(stop_pct * 100)}_def{defense_window}"
                            f"_release_health{health_lookback}_confirm3"
                        ),
                        regime_modes=daily_modes,
                        regime_exposures=full_exposure,
                        portfolio_stop_drawdown_pct=stop_pct,
                        portfolio_stop_latch_mode=MODE_0050_DEFENSE,
                        portfolio_stop_latch_defense_rule=f"ma{defense_window}",
                        portfolio_stop_release_filter="daily_health",
                        portfolio_stop_release_confirmation_days=3,
                        portfolio_stop_release_health_lookback_days=health_lookback,
                        portfolio_stop_release_health_min_excess_return=0.0,
                    )
                )
    return tuple(variants)


def attack_gate_latch_variants() -> tuple[RegimeModeSwitchVariant, ...]:
    daily_modes = {
        "strong_bull": MODE_DAILY,
        "recovery_bull": MODE_DAILY,
        "range_bound": MODE_DAILY,
        "correction_bear": MODE_DAILY,
        "systemic_bear": MODE_CASH,
    }
    full_exposure = {
        "strong_bull": 1.0,
        "recovery_bull": 1.0,
        "range_bound": 1.0,
        "correction_bear": 1.0,
        "systemic_bear": 0.0,
    }
    variants: list[RegimeModeSwitchVariant] = []
    for margin in (0.20, 0.30, 0.40):
        for confirmation_days in (1, 3):
            for stop_pct in (0.10, 0.12):
                variants.append(
                    RegimeModeSwitchVariant(
                        name=(
                            f"attack_gate_m{int(margin * 100)}_confirm{confirmation_days}"
                            f"_stop{int(stop_pct * 100)}_def245"
                        ),
                        regime_modes=daily_modes,
                        regime_exposures=full_exposure,
                        portfolio_stop_drawdown_pct=stop_pct,
                        attack_gate_margin_over_fallback=margin,
                        attack_gate_fallback_ticker="0050.TW",
                        attack_gate_defense_rule="ma245",
                        attack_gate_activation_confirmation_days=confirmation_days,
                    )
                )
    return tuple(variants)


def attack_gate_fine_sweep_variants() -> tuple[RegimeModeSwitchVariant, ...]:
    daily_modes = {
        "strong_bull": MODE_DAILY,
        "recovery_bull": MODE_DAILY,
        "range_bound": MODE_DAILY,
        "correction_bear": MODE_DAILY,
        "systemic_bear": MODE_CASH,
    }
    full_exposure = {
        "strong_bull": 1.0,
        "recovery_bull": 1.0,
        "range_bound": 1.0,
        "correction_bear": 1.0,
        "systemic_bear": 0.0,
    }
    variants: list[RegimeModeSwitchVariant] = []
    for margin in (0.20, 0.22, 0.24, 0.26, 0.28, 0.30):
        for confirmation_days in (1, 2):
            variants.append(
                RegimeModeSwitchVariant(
                    name=f"attack_gate_fine_m{int(margin * 100)}_confirm{confirmation_days}_stop12_def245",
                    regime_modes=daily_modes,
                    regime_exposures=full_exposure,
                    portfolio_stop_drawdown_pct=0.12,
                    attack_gate_margin_over_fallback=margin,
                    attack_gate_fallback_ticker="0050.TW",
                    attack_gate_defense_rule="ma245",
                    attack_gate_activation_confirmation_days=confirmation_days,
                )
            )
    return tuple(variants)


def attack_gate_persistence_variants() -> tuple[RegimeModeSwitchVariant, ...]:
    daily_modes = {
        "strong_bull": MODE_DAILY,
        "recovery_bull": MODE_DAILY,
        "range_bound": MODE_DAILY,
        "correction_bear": MODE_DAILY,
        "systemic_bear": MODE_CASH,
    }
    full_exposure = {
        "strong_bull": 1.0,
        "recovery_bull": 1.0,
        "range_bound": 1.0,
        "correction_bear": 1.0,
        "systemic_bear": 0.0,
    }
    variants: list[RegimeModeSwitchVariant] = []
    for margin in (0.20, 0.22):
        for min_top_days in (6, 8, 10):
            variants.append(
                RegimeModeSwitchVariant(
                    name=f"attack_gate_m{int(margin * 100)}_persist{min_top_days}of10_stop12_def245",
                    regime_modes=daily_modes,
                    regime_exposures=full_exposure,
                    portfolio_stop_drawdown_pct=0.12,
                    attack_gate_margin_over_fallback=margin,
                    attack_gate_fallback_ticker="0050.TW",
                    attack_gate_defense_rule="ma245",
                    attack_gate_activation_confirmation_days=1,
                    attack_gate_persistence_lookback_days=10,
                    attack_gate_min_top_days=min_top_days,
                )
            )
    return tuple(variants)


def attack_gate_acceleration_variants() -> tuple[RegimeModeSwitchVariant, ...]:
    daily_modes = {
        "strong_bull": MODE_DAILY,
        "recovery_bull": MODE_DAILY,
        "range_bound": MODE_DAILY,
        "correction_bear": MODE_DAILY,
        "systemic_bear": MODE_CASH,
    }
    full_exposure = {
        "strong_bull": 1.0,
        "recovery_bull": 1.0,
        "range_bound": 1.0,
        "correction_bear": 1.0,
        "systemic_bear": 0.0,
    }
    variants: list[RegimeModeSwitchVariant] = []
    for margin in (0.20, 0.22):
        for ratio in (0.40, 0.50, 0.60):
            variants.append(
                RegimeModeSwitchVariant(
                    name=f"attack_gate_m{int(margin * 100)}_accel{int(ratio * 100)}_stop12_def245",
                    regime_modes=daily_modes,
                    regime_exposures=full_exposure,
                    portfolio_stop_drawdown_pct=0.12,
                    attack_gate_margin_over_fallback=margin,
                    attack_gate_fallback_ticker="0050.TW",
                    attack_gate_defense_rule="ma245",
                    attack_gate_activation_confirmation_days=1,
                    attack_gate_min_short_to_medium_momentum_ratio=ratio,
                )
            )
    return tuple(variants)


def attack_gate_leveraged_fallback_variants() -> tuple[RegimeModeSwitchVariant, ...]:
    """Use leveraged market beta until a durable stock leader is confirmed."""
    daily_modes = {
        "strong_bull": MODE_DAILY,
        "recovery_bull": MODE_DAILY,
        "range_bound": MODE_DAILY,
        "correction_bear": MODE_DAILY,
        "systemic_bear": MODE_CASH,
    }
    full_exposure = {
        "strong_bull": 1.0,
        "recovery_bull": 1.0,
        "range_bound": 1.0,
        "correction_bear": 1.0,
        "systemic_bear": 0.0,
    }
    variants: list[RegimeModeSwitchVariant] = []
    for margin in (0.20, 0.22):
        for min_top_days in (8, 10):
            for defense_rule in ("ma60", "ma120", "ma200"):
                variants.append(
                    RegimeModeSwitchVariant(
                        name=(
                            f"selector_m{int(margin * 100)}_persist{min_top_days}of10"
                            f"_00631l_{defense_rule}_stop12"
                        ),
                        regime_modes=daily_modes,
                        regime_exposures=full_exposure,
                        portfolio_stop_drawdown_pct=0.12,
                        attack_gate_margin_over_fallback=margin,
                        attack_gate_fallback_ticker="0050.TW",
                        attack_gate_defense_rule=defense_rule,
                        attack_gate_activation_confirmation_days=1,
                        attack_gate_persistence_lookback_days=10,
                        attack_gate_min_top_days=min_top_days,
                        defense_anchor_ticker="00631L.TW",
                    )
                )
    return tuple(variants)


def asymmetric_strategy_selector_variants() -> tuple[RegimeModeSwitchVariant, ...]:
    """Select aggressive beta in bull regimes and defensive beta elsewhere."""
    daily_modes = {
        "strong_bull": MODE_DAILY,
        "recovery_bull": MODE_DAILY,
        "range_bound": MODE_DAILY,
        "correction_bear": MODE_DAILY,
        "systemic_bear": MODE_CASH,
    }
    full_exposure = {
        "strong_bull": 1.0,
        "recovery_bull": 1.0,
        "range_bound": 1.0,
        "correction_bear": 1.0,
        "systemic_bear": 0.0,
    }
    variants: list[RegimeModeSwitchVariant] = []
    for margin in (0.20, 0.22):
        for min_top_days in (8, 10):
            for bull_rule in ("ma120", "ma200"):
                variants.append(
                    RegimeModeSwitchVariant(
                        name=(
                            f"asym_selector_m{int(margin * 100)}_persist{min_top_days}of10"
                            f"_bull00631l_{bull_rule}_bear0050_ma245_stop12"
                        ),
                        regime_modes=daily_modes,
                        regime_exposures=full_exposure,
                        portfolio_stop_drawdown_pct=0.12,
                        attack_gate_margin_over_fallback=margin,
                        attack_gate_fallback_ticker="0050.TW",
                        attack_gate_defense_rule=None,
                        attack_gate_defense_rule_by_regime={
                            "strong_bull": bull_rule,
                            "recovery_bull": bull_rule,
                            "range_bound": "ma245",
                            "correction_bear": "ma245",
                        },
                        attack_gate_activation_confirmation_days=1,
                        attack_gate_persistence_lookback_days=10,
                        attack_gate_min_top_days=min_top_days,
                        defense_anchor_ticker_by_regime={
                            "strong_bull": "00631L.TW",
                            "recovery_bull": "00631L.TW",
                            "range_bound": "0050.TW",
                            "correction_bear": "0050.TW",
                        },
                    )
                )
    return tuple(variants)


def stop_latched_strategy_selector_variants() -> tuple[RegimeModeSwitchVariant, ...]:
    """Keep aggressive beta available, but latch into 0050 defense after a stop."""
    daily_modes = {
        "strong_bull": MODE_DAILY,
        "recovery_bull": MODE_DAILY,
        "range_bound": MODE_DAILY,
        "correction_bear": MODE_DAILY,
        "systemic_bear": MODE_CASH,
    }
    full_exposure = {
        "strong_bull": 1.0,
        "recovery_bull": 1.0,
        "range_bound": 1.0,
        "correction_bear": 1.0,
        "systemic_bear": 0.0,
    }
    variants: list[RegimeModeSwitchVariant] = []
    for margin in (0.20, 0.22):
        for min_top_days in (10,):
            for release_filter in ("strong_bull_fallback_risk_on", "strong_or_recovery_bull_fallback_risk_on"):
                for confirmation_days in (1, 3):
                    release_name = "strong" if release_filter.startswith("strong_bull") else "bull"
                    variants.append(
                        RegimeModeSwitchVariant(
                            name=(
                                f"latched_selector_m{int(margin * 100)}_persist{min_top_days}of10"
                                f"_00631l_ma200_stop12_0050ma245_release{release_name}{confirmation_days}"
                            ),
                            regime_modes=daily_modes,
                            regime_exposures=full_exposure,
                            portfolio_stop_drawdown_pct=0.12,
                            attack_gate_margin_over_fallback=margin,
                            attack_gate_fallback_ticker="0050.TW",
                            attack_gate_defense_rule="ma200",
                            attack_gate_activation_confirmation_days=1,
                            attack_gate_persistence_lookback_days=10,
                            attack_gate_min_top_days=min_top_days,
                            defense_anchor_ticker="00631L.TW",
                            attack_gate_stop_latch_ticker="0050.TW",
                            attack_gate_stop_latch_rule="ma245",
                            attack_gate_stop_release_filter=release_filter,
                            attack_gate_stop_release_confirmation_days=confirmation_days,
                        )
                    )
    return tuple(variants)


def fast_risk_strategy_selector_variants() -> tuple[RegimeModeSwitchVariant, ...]:
    """Use a fast market-risk overlay before the slower regime classifier reacts."""
    daily_modes = {
        "strong_bull": MODE_DAILY,
        "recovery_bull": MODE_DAILY,
        "range_bound": MODE_DAILY,
        "correction_bear": MODE_DAILY,
        "systemic_bear": MODE_CASH,
    }
    full_exposure = {
        "strong_bull": 1.0,
        "recovery_bull": 1.0,
        "range_bound": 1.0,
        "correction_bear": 1.0,
        "systemic_bear": 0.0,
    }
    variants: list[RegimeModeSwitchVariant] = []
    for margin in (0.20, 0.22):
        for risk_filter in ("ma60_ret20", "risk_2of3"):
            for exit_confirmation_days in (3, 5):
                variants.append(
                    RegimeModeSwitchVariant(
                        name=(
                            f"fast_risk_selector_m{int(margin * 100)}_persist10of10"
                            f"_00631l_ma200_{risk_filter}_to0050ma245_exit{exit_confirmation_days}_stop12"
                        ),
                        regime_modes=daily_modes,
                        regime_exposures=full_exposure,
                        portfolio_stop_drawdown_pct=0.12,
                        market_risk_off_filter=risk_filter,
                        market_risk_off_mode=MODE_0050_DEFENSE,
                        market_risk_off_defense_rule="ma245",
                        market_risk_off_defense_ticker="0050.TW",
                        market_risk_off_exit_confirmation_days=exit_confirmation_days,
                        attack_gate_margin_over_fallback=margin,
                        attack_gate_fallback_ticker="0050.TW",
                        attack_gate_defense_rule="ma200",
                        attack_gate_activation_confirmation_days=1,
                        attack_gate_persistence_lookback_days=10,
                        attack_gate_min_top_days=10,
                        defense_anchor_ticker="00631L.TW",
                    )
                )
    return tuple(variants)


def fallback_only_risk_selector_variants() -> tuple[RegimeModeSwitchVariant, ...]:
    """Protect leveraged fallback without interrupting a confirmed leader attack."""
    daily_modes = {
        "strong_bull": MODE_DAILY,
        "recovery_bull": MODE_DAILY,
        "range_bound": MODE_DAILY,
        "correction_bear": MODE_DAILY,
        "systemic_bear": MODE_CASH,
    }
    full_exposure = {
        "strong_bull": 1.0,
        "recovery_bull": 1.0,
        "range_bound": 1.0,
        "correction_bear": 1.0,
        "systemic_bear": 0.0,
    }
    variants: list[RegimeModeSwitchVariant] = []
    for risk_filter in ("ma60_ret20", "risk_2of3"):
        for exit_confirmation_days in (3, 5):
            variants.append(
                RegimeModeSwitchVariant(
                    name=(
                        "fallback_risk_selector_m22_persist10of10_00631l_ma200"
                        f"_{risk_filter}_to0050ma245_exit{exit_confirmation_days}_stop12"
                    ),
                    regime_modes=daily_modes,
                    regime_exposures=full_exposure,
                    portfolio_stop_drawdown_pct=0.12,
                    market_risk_off_filter=risk_filter,
                    market_risk_off_mode=MODE_0050_DEFENSE,
                    market_risk_off_defense_rule="ma245",
                    market_risk_off_defense_ticker="0050.TW",
                    market_risk_off_exit_confirmation_days=exit_confirmation_days,
                    market_risk_off_only_when_attack_gate_inactive=True,
                    attack_gate_margin_over_fallback=0.22,
                    attack_gate_fallback_ticker="0050.TW",
                    attack_gate_defense_rule="ma200",
                    attack_gate_activation_confirmation_days=1,
                    attack_gate_persistence_lookback_days=10,
                    attack_gate_min_top_days=10,
                    defense_anchor_ticker="00631L.TW",
                )
            )
    return tuple(variants)


def two_stage_attack_selector_variants() -> tuple[RegimeModeSwitchVariant, ...]:
    """Require durable leadership once, then permit faster post-stop re-entry."""
    daily_modes = {
        "strong_bull": MODE_DAILY,
        "recovery_bull": MODE_DAILY,
        "range_bound": MODE_DAILY,
        "correction_bear": MODE_DAILY,
        "systemic_bear": MODE_CASH,
    }
    full_exposure = {
        "strong_bull": 1.0,
        "recovery_bull": 1.0,
        "range_bound": 1.0,
        "correction_bear": 1.0,
        "systemic_bear": 0.0,
    }
    variants: list[RegimeModeSwitchVariant] = []
    for reentry_ratio in (0.40, 0.60):
        for use_fallback_risk in (False, True):
            risk_name = "risk2of3_exit5" if use_fallback_risk else "norisk"
            variants.append(
                RegimeModeSwitchVariant(
                    name=(
                        "two_stage_m22_persist10of10"
                        f"_reentry_m20_accel{int(reentry_ratio * 100)}"
                        f"_00631l_ma200_{risk_name}_stop12"
                    ),
                    regime_modes=daily_modes,
                    regime_exposures=full_exposure,
                    portfolio_stop_drawdown_pct=0.12,
                    market_risk_off_filter="risk_2of3" if use_fallback_risk else None,
                    market_risk_off_mode=MODE_0050_DEFENSE,
                    market_risk_off_defense_rule="ma245",
                    market_risk_off_defense_ticker="0050.TW",
                    market_risk_off_exit_confirmation_days=5,
                    market_risk_off_only_when_attack_gate_inactive=use_fallback_risk,
                    attack_gate_margin_over_fallback=0.22,
                    attack_gate_fallback_ticker="0050.TW",
                    attack_gate_defense_rule="ma200",
                    attack_gate_activation_confirmation_days=1,
                    attack_gate_persistence_lookback_days=10,
                    attack_gate_min_top_days=10,
                    attack_gate_reentry_margin_over_fallback=0.20,
                    attack_gate_reentry_min_short_to_medium_momentum_ratio=reentry_ratio,
                    defense_anchor_ticker="00631L.TW",
                )
            )
    return tuple(variants)


def two_stage_fast_guard_variants() -> tuple[RegimeModeSwitchVariant, ...]:
    """Sweep fast guards only for the leveraged-beta waiting state."""
    daily_modes = {
        "strong_bull": MODE_DAILY,
        "recovery_bull": MODE_DAILY,
        "range_bound": MODE_DAILY,
        "correction_bear": MODE_DAILY,
        "systemic_bear": MODE_CASH,
    }
    full_exposure = {
        "strong_bull": 1.0,
        "recovery_bull": 1.0,
        "range_bound": 1.0,
        "correction_bear": 1.0,
        "systemic_bear": 0.0,
    }
    variants: list[RegimeModeSwitchVariant] = []
    for risk_filter in ("risk_2of3", "ma20_ret10", "ret20_dd5", "ma60_ret20"):
        variants.append(
            RegimeModeSwitchVariant(
                name=(
                    "two_stage_fast_guard_m22_persist10of10_reentry_m20_accel40"
                    f"_00631l_ma200_{risk_filter}_to0050ma245_exit5_stop12"
                ),
                regime_modes=daily_modes,
                regime_exposures=full_exposure,
                portfolio_stop_drawdown_pct=0.12,
                market_risk_off_filter=risk_filter,
                market_risk_off_mode=MODE_0050_DEFENSE,
                market_risk_off_defense_rule="ma245",
                market_risk_off_defense_ticker="0050.TW",
                market_risk_off_exit_confirmation_days=5,
                market_risk_off_only_when_attack_gate_inactive=True,
                attack_gate_margin_over_fallback=0.22,
                attack_gate_fallback_ticker="0050.TW",
                attack_gate_defense_rule="ma200",
                attack_gate_activation_confirmation_days=1,
                attack_gate_persistence_lookback_days=10,
                attack_gate_min_top_days=10,
                attack_gate_reentry_margin_over_fallback=0.20,
                attack_gate_reentry_min_short_to_medium_momentum_ratio=0.40,
                defense_anchor_ticker="00631L.TW",
            )
        )
    return tuple(variants)


def two_stage_cash_guard_variants() -> tuple[RegimeModeSwitchVariant, ...]:
    """Test whether a confirmed waiting-state risk event should go fully to cash."""
    daily_modes = {
        "strong_bull": MODE_DAILY,
        "recovery_bull": MODE_DAILY,
        "range_bound": MODE_DAILY,
        "correction_bear": MODE_DAILY,
        "systemic_bear": MODE_CASH,
    }
    full_exposure = {
        "strong_bull": 1.0,
        "recovery_bull": 1.0,
        "range_bound": 1.0,
        "correction_bear": 1.0,
        "systemic_bear": 0.0,
    }
    variants: list[RegimeModeSwitchVariant] = []
    for risk_filter in ("risk_2of3", "ma20_ret10", "ret20_dd5"):
        variants.append(
            RegimeModeSwitchVariant(
                name=(
                    "two_stage_cash_guard_m22_persist10of10_reentry_m20_accel40"
                    f"_00631l_ma200_{risk_filter}_to_cash_exit5_stop12"
                ),
                regime_modes=daily_modes,
                regime_exposures=full_exposure,
                portfolio_stop_drawdown_pct=0.12,
                market_risk_off_filter=risk_filter,
                market_risk_off_mode=MODE_CASH,
                market_risk_off_exit_confirmation_days=5,
                market_risk_off_only_when_attack_gate_inactive=True,
                attack_gate_margin_over_fallback=0.22,
                attack_gate_fallback_ticker="0050.TW",
                attack_gate_defense_rule="ma200",
                attack_gate_activation_confirmation_days=1,
                attack_gate_persistence_lookback_days=10,
                attack_gate_min_top_days=10,
                attack_gate_reentry_margin_over_fallback=0.20,
                attack_gate_reentry_min_short_to_medium_momentum_ratio=0.40,
                defense_anchor_ticker="00631L.TW",
            )
        )
    return tuple(variants)


def cycle_proven_selector_variants() -> tuple[RegimeModeSwitchVariant, ...]:
    """Defend before a cycle proves leadership, then use faster attack re-entry."""
    daily_modes = {
        "strong_bull": MODE_DAILY,
        "recovery_bull": MODE_DAILY,
        "range_bound": MODE_DAILY,
        "correction_bear": MODE_DAILY,
        "systemic_bear": MODE_CASH,
    }
    full_exposure = {
        "strong_bull": 1.0,
        "recovery_bull": 1.0,
        "range_bound": 1.0,
        "correction_bear": 1.0,
        "systemic_bear": 0.0,
    }
    variants: list[RegimeModeSwitchVariant] = []
    for risk_mode in (MODE_CASH, MODE_0050_DEFENSE):
        risk_name = "cash" if risk_mode == MODE_CASH else "0050ma245"
        variants.append(
            RegimeModeSwitchVariant(
                name=(
                    "cycle_proven_m22_persist10of10_reentry_m20_accel40"
                    f"_00631l_ma200_preproof_risk2of3_to_{risk_name}_exit5_stop12"
                ),
                regime_modes=daily_modes,
                regime_exposures=full_exposure,
                portfolio_stop_drawdown_pct=0.12,
                market_risk_off_filter="risk_2of3",
                market_risk_off_mode=risk_mode,
                market_risk_off_defense_rule="ma245",
                market_risk_off_defense_ticker="0050.TW",
                market_risk_off_exit_confirmation_days=5,
                market_risk_off_only_before_first_attack_activation=True,
                attack_gate_margin_over_fallback=0.22,
                attack_gate_fallback_ticker="0050.TW",
                attack_gate_defense_rule="ma200",
                attack_gate_activation_confirmation_days=1,
                attack_gate_persistence_lookback_days=10,
                attack_gate_min_top_days=10,
                attack_gate_reentry_margin_over_fallback=0.20,
                attack_gate_reentry_min_short_to_medium_momentum_ratio=0.40,
                defense_anchor_ticker="00631L.TW",
            )
        )
    return tuple(variants)


def cycle_proven_robustness_variants() -> tuple[RegimeModeSwitchVariant, ...]:
    """Parameter-neighborhood checks for the current cycle-proven candidate."""
    daily_modes = {
        "strong_bull": MODE_DAILY,
        "recovery_bull": MODE_DAILY,
        "range_bound": MODE_DAILY,
        "correction_bear": MODE_DAILY,
        "systemic_bear": MODE_CASH,
    }
    full_exposure = {
        "strong_bull": 1.0,
        "recovery_bull": 1.0,
        "range_bound": 1.0,
        "correction_bear": 1.0,
        "systemic_bear": 0.0,
    }
    variants: list[RegimeModeSwitchVariant] = []
    for initial_margin in (0.20, 0.22, 0.24):
        for min_top_days in (8, 10):
            for reentry_ratio in (0.40, 0.60):
                variants.append(
                    RegimeModeSwitchVariant(
                        name=(
                            f"cycle_robust_m{int(initial_margin * 100)}_persist{min_top_days}of10"
                            f"_reentry_m20_accel{int(reentry_ratio * 100)}"
                            "_00631l_ma200_preproof_risk2of3_to_cash_exit5_stop12"
                        ),
                        regime_modes=daily_modes,
                        regime_exposures=full_exposure,
                        portfolio_stop_drawdown_pct=0.12,
                        market_risk_off_filter="risk_2of3",
                        market_risk_off_mode=MODE_CASH,
                        market_risk_off_exit_confirmation_days=5,
                        market_risk_off_only_before_first_attack_activation=True,
                        attack_gate_margin_over_fallback=initial_margin,
                        attack_gate_fallback_ticker="0050.TW",
                        attack_gate_defense_rule="ma200",
                        attack_gate_activation_confirmation_days=1,
                        attack_gate_persistence_lookback_days=10,
                        attack_gate_min_top_days=min_top_days,
                        attack_gate_reentry_margin_over_fallback=0.20,
                        attack_gate_reentry_min_short_to_medium_momentum_ratio=reentry_ratio,
                        defense_anchor_ticker="00631L.TW",
                    )
                )
    return tuple(variants)


def cycle_proven_history_init_variants() -> tuple[RegimeModeSwitchVariant, ...]:
    """Restore cycle-proof state from information available before the start date."""
    base = cycle_proven_selector_variants()[0]
    return tuple(
        RegimeModeSwitchVariant(
            **{
                **base.__dict__,
                "name": f"cycle_proven_history_init_{history_days}d",
                "attack_gate_initialize_history_days": history_days,
                "attack_gate_initialize_active_from_history": True,
            }
        )
        for history_days in (60, 120, 252)
    )


def cycle_proven_preproof_exposure_variants() -> tuple[RegimeModeSwitchVariant, ...]:
    """Vary only the residual leveraged-beta exposure during pre-proof risk-off."""
    base = cycle_proven_history_init_variants()[0]
    variants: list[RegimeModeSwitchVariant] = []
    for exposure in (0.0, 0.25, 0.50, 0.75):
        is_cash = exposure == 0.0
        variants.append(
            RegimeModeSwitchVariant(
                **{
                    **base.__dict__,
                    "name": f"cycle_preproof_risk_exposure_{int(exposure * 100)}pct",
                    "market_risk_off_mode": MODE_CASH if is_cash else MODE_0050_DEFENSE,
                    "market_risk_off_defense_ticker": None if is_cash else "00631L.TW",
                    "market_risk_off_defense_rule": "ma200",
                    "market_risk_off_exposure": exposure,
                }
            )
        )
    return tuple(variants)


def cycle_proven_preproof_dynamic_exposure_variants() -> tuple[RegimeModeSwitchVariant, ...]:
    """Dynamically choose pre-proof residual leveraged exposure by market health."""
    base = cycle_proven_history_init_variants()[0]
    variants: list[RegimeModeSwitchVariant] = []
    for selector in (
        "00631l_above_ma120_to_75pct_else_cash",
        "00631l_above_ma200_to_75pct_else_cash",
        "0050_above_ma120_to_75pct_else_cash",
        "0050_above_ma200_to_75pct_else_cash",
        "0050_above_ma200_to_50pct_else_cash",
        "0050_above_ma200_bull_regime_to_25pct_else_cash",
        "0050_above_ma200_bull_regime_to_50pct_else_cash",
        "0050_above_ma200_bull_regime_to_75pct_else_cash",
    ):
        variants.append(
            RegimeModeSwitchVariant(
                **{
                    **base.__dict__,
                    "name": f"cycle_preproof_dynamic_{selector}",
                    "market_risk_off_mode": MODE_0050_DEFENSE,
                    "market_risk_off_defense_ticker": "00631L.TW",
                    "market_risk_off_defense_rule": "ma200",
                    "market_risk_off_exposure": None,
                    "market_risk_off_exposure_selector": selector,
                }
            )
        )
    for exposure in (25, 50, 75):
        selector = f"0050_above_ma200_bull_regime_to_{exposure}pct_else_cash"
        variants.append(
            RegimeModeSwitchVariant(
                **{
                    **base.__dict__,
                    "name": f"cycle_preproof_dynamic_m20_{selector}",
                    "market_risk_off_mode": MODE_0050_DEFENSE,
                    "market_risk_off_defense_ticker": "00631L.TW",
                    "market_risk_off_defense_rule": "ma200",
                    "market_risk_off_exposure": None,
                    "market_risk_off_exposure_selector": selector,
                    "attack_gate_margin_over_fallback": 0.20,
                }
            )
        )
    for acceleration in (0.40, 0.60, 0.80):
        selector = "0050_above_ma200_bull_regime_to_75pct_else_cash"
        variants.append(
            RegimeModeSwitchVariant(
                **{
                    **base.__dict__,
                    "name": (
                        f"cycle_preproof_dynamic_m20_initial_accel{int(acceleration * 100)}_"
                        f"{selector}"
                    ),
                    "market_risk_off_mode": MODE_0050_DEFENSE,
                    "market_risk_off_defense_ticker": "00631L.TW",
                    "market_risk_off_defense_rule": "ma200",
                    "market_risk_off_exposure": None,
                    "market_risk_off_exposure_selector": selector,
                    "attack_gate_margin_over_fallback": 0.20,
                    "attack_gate_min_short_to_medium_momentum_ratio": acceleration,
                }
            )
        )
    return tuple(variants)


def frozen_cycle_proven_top1_v1_spec() -> FrozenCycleProvenTop1Spec:
    """Return the named production-baseline spec without traversing research results."""
    return FrozenCycleProvenTop1Spec()


def build_frozen_cycle_proven_top1_v1_variant(
    spec: FrozenCycleProvenTop1Spec | None = None,
) -> RegimeModeSwitchVariant:
    """Build the frozen baseline from a named spec while preserving behavior."""
    spec = spec or frozen_cycle_proven_top1_v1_spec()
    matching = [
        variant
        for variant in cycle_proven_preproof_exposure_variants()
        if variant.market_risk_off_exposure == spec.market_risk_off_exposure
        and variant.defense_anchor_ticker == spec.defense_anchor_ticker
    ]
    if not matching:
        raise RuntimeError(f"Missing frozen baseline base variant for spec: {spec}")
    base = matching[0]
    return RegimeModeSwitchVariant(
        **{
            **base.__dict__,
            "name": spec.name,
            "attack_selection_exclude_tickers": spec.attack_selection_exclude_tickers,
        }
    )


def frozen_cycle_proven_top1_v1_variant() -> RegimeModeSwitchVariant:
    """Return the immutable production baseline used by reports and challengers."""
    return build_frozen_cycle_proven_top1_v1_variant()


def ai_theme_large_cap_v20260613_variant() -> RegimeModeSwitchVariant:
    """Return the current AI large-cap challenger used for three-perspective voting."""
    target_name = "cycle_mature_bull_cadence_bull_wed_after20d_breakout_gap10"
    for variant in cycle_proven_mature_bull_cadence_variants():
        if variant.name == target_name:
            return RegimeModeSwitchVariant(
                **{
                    **variant.__dict__,
                    "name": "ai_theme_large_cap_v20260613",
                }
            )
    raise RuntimeError(f"Missing regime variant: {target_name}")


def cycle_proven_cadence_variants() -> tuple[RegimeModeSwitchVariant, ...]:
    """Compare daily analysis with daily-risk/weekly-rotation and full-weekly operation."""
    base = cycle_proven_preproof_exposure_variants()[1]
    base_values = {
        **base.__dict__,
        "attack_selection_exclude_tickers": ("0050.TW", "00631L.TW"),
    }
    variants = [
        RegimeModeSwitchVariant(
            **{
                **base_values,
                "name": "cycle_cadence_daily_analysis",
            }
        )
    ]
    weekday_names = ("mon", "tue", "wed", "thu", "fri")
    for weekday, weekday_name in enumerate(weekday_names):
        variants.append(
            RegimeModeSwitchVariant(
                **{
                    **base_values,
                    "name": f"cycle_cadence_daily_risk_weekly_rotation_{weekday_name}",
                    "normal_rebalance_weekday": weekday,
                }
            )
        )
        variants.append(
            RegimeModeSwitchVariant(
                **{
                    **base_values,
                    "name": f"cycle_cadence_full_weekly_{weekday_name}",
                    "normal_rebalance_weekday": weekday,
                    "state_evaluation_weekday": weekday,
                }
            )
        )
    variants.extend(
        [
            RegimeModeSwitchVariant(
                **{
                    **base_values,
                    "name": "cycle_cadence_daily_risk_weekly_rotation_last_trading_day",
                    "normal_rebalance_last_trading_day_of_week": True,
                }
            ),
            RegimeModeSwitchVariant(
                **{
                    **base_values,
                    "name": "cycle_cadence_daily_risk_weekly_rotation_fri_gap2_guard",
                    "normal_rebalance_weekday": 4,
                    "defer_rebalance_on_adverse_open_gap_pct": 0.02,
                }
            ),
            RegimeModeSwitchVariant(
                **{
                    **base_values,
                    "name": "cycle_cadence_daily_risk_weekly_rotation_last_trading_day_gap2_guard",
                    "normal_rebalance_last_trading_day_of_week": True,
                    "defer_rebalance_on_adverse_open_gap_pct": 0.02,
                }
            ),
            RegimeModeSwitchVariant(
                **{
                    **base_values,
                    "name": "cycle_cadence_daily_risk_weekly_rotation_last_trading_day_gap3_guard",
                    "normal_rebalance_last_trading_day_of_week": True,
                    "defer_rebalance_on_adverse_open_gap_pct": 0.03,
                }
            ),
        ]
    )
    return tuple(variants)


def cycle_proven_adaptive_cadence_variants() -> tuple[RegimeModeSwitchVariant, ...]:
    """Switch only the normal rotation cadence by market regime."""
    base = cycle_proven_preproof_exposure_variants()[1]
    base_values = {
        **base.__dict__,
        "attack_selection_exclude_tickers": ("0050.TW", "00631L.TW"),
    }
    regime_sets = (
        ("strong_bull_wed_rest_daily", {"strong_bull": 2}),
        ("bull_wed_rest_daily", {"strong_bull": 2, "recovery_bull": 2}),
        ("non_bear_wed_rest_daily", {"strong_bull": 2, "recovery_bull": 2, "range_bound": 2}),
        (
            "bull_wed_range_last_day_rest_daily",
            {"strong_bull": 2, "recovery_bull": 2},
        ),
    )
    variants: list[RegimeModeSwitchVariant] = []
    for name, weekday_by_regime in regime_sets:
        values = {
            **base_values,
            "name": f"cycle_adaptive_cadence_{name}",
            "normal_rebalance_weekday_by_regime": weekday_by_regime,
        }
        if name == "bull_wed_range_last_day_rest_daily":
            values["normal_rebalance_last_trading_day_regimes"] = ("range_bound",)
        variants.append(RegimeModeSwitchVariant(**values))
    variants.append(
        RegimeModeSwitchVariant(
            **{
                **base_values,
                "name": "cycle_adaptive_cadence_bull_wed_rest_daily_gap2_guard",
                "normal_rebalance_weekday_by_regime": {"strong_bull": 2, "recovery_bull": 2},
                "defer_rebalance_on_adverse_open_gap_pct": 0.02,
            }
        )
    )
    return tuple(variants)


def cycle_proven_adaptive_cadence_breakout_variants() -> tuple[RegimeModeSwitchVariant, ...]:
    """Allow an off-cadence switch when leadership clearly changes."""
    base = cycle_proven_preproof_exposure_variants()[1]
    base_values = {
        **base.__dict__,
        "attack_selection_exclude_tickers": ("0050.TW", "00631L.TW"),
    }
    variants: list[RegimeModeSwitchVariant] = []
    templates = (
        ("strong_bull_wed_rest_daily", {"strong_bull": 2}, ("strong_bull",)),
        ("bull_wed_rest_daily", {"strong_bull": 2, "recovery_bull": 2}, ("strong_bull", "recovery_bull")),
    )
    for template_name, weekday_by_regime, early_regimes in templates:
        for gap in (0.05, 0.10, 0.15, 0.20, 0.25):
            variants.append(
                RegimeModeSwitchVariant(
                    **{
                        **base_values,
                        "name": f"cycle_adaptive_breakout_{template_name}_gap{int(gap * 100)}",
                        "normal_rebalance_weekday_by_regime": weekday_by_regime,
                        "early_rebalance_regimes": early_regimes,
                        "early_rebalance_min_gap_over_current": gap,
                        "early_rebalance_min_top_gap": gap,
                    }
                )
            )
    return tuple(variants)


def cycle_proven_mature_bull_cadence_variants() -> tuple[RegimeModeSwitchVariant, ...]:
    """Keep daily rotation early in a bull regime, then switch to weekly cadence after it matures."""
    base = cycle_proven_preproof_exposure_variants()[1]
    base_values = {
        **base.__dict__,
        "attack_selection_exclude_tickers": ("0050.TW", "00631L.TW"),
    }
    variants: list[RegimeModeSwitchVariant] = []
    templates = (
        ("strong_bull_wed", {"strong_bull": 2}),
        ("bull_wed", {"strong_bull": 2, "recovery_bull": 2}),
    )
    for name, weekday_by_regime in templates:
        for streak_days in (10, 20, 30):
            variants.append(
                RegimeModeSwitchVariant(
                    **{
                        **base_values,
                        "name": f"cycle_mature_bull_cadence_{name}_after{streak_days}d",
                        "normal_rebalance_weekday_by_regime": weekday_by_regime,
                        "normal_rebalance_min_regime_streak_days": streak_days,
                    }
                )
            )
    for gap in (0.10, 0.15):
        variants.append(
            RegimeModeSwitchVariant(
                **{
                    **base_values,
                    "name": f"cycle_mature_bull_cadence_bull_wed_after20d_breakout_gap{int(gap * 100)}",
                    "normal_rebalance_weekday_by_regime": {"strong_bull": 2, "recovery_bull": 2},
                    "normal_rebalance_min_regime_streak_days": 20,
                    "early_rebalance_regimes": ("strong_bull", "recovery_bull"),
                    "early_rebalance_min_gap_over_current": gap,
                    "early_rebalance_min_top_gap": gap,
                }
            )
        )
    return tuple(variants)


def cycle_proven_asset_role_variants() -> tuple[RegimeModeSwitchVariant, ...]:
    """Test whether broad-market ETFs should compete with stock leaders."""
    base = cycle_proven_preproof_exposure_variants()[1]
    role_definitions = (
        ("unrestricted_9asset_ranking", (), ()),
        ("00631l_gate_signal_not_attack_holding", ("00631L.TW",), ()),
        ("etfs_signal_not_attack_holding", ("0050.TW", "00631L.TW"), ()),
        ("strict_etfs_role_separated", ("0050.TW", "00631L.TW"), ("00631L.TW",)),
    )
    return tuple(
        RegimeModeSwitchVariant(
            **{
                **base.__dict__,
                "name": f"cycle_asset_role_{name}",
                "attack_selection_exclude_tickers": selection_exclusions,
                "attack_gate_exclude_tickers": gate_exclusions,
            }
        )
        for name, selection_exclusions, gate_exclusions in role_definitions
    )


def cycle_proven_market_exposure_ladder_variants() -> tuple[RegimeModeSwitchVariant, ...]:
    """Vary broad-market instruments and exposure only while the attack gate is inactive."""
    base = cycle_proven_preproof_exposure_variants()[1]
    base_values = {
        **base.__dict__,
        "attack_selection_exclude_tickers": ("0050.TW", "00631L.TW"),
    }
    ladders = (
        (
            "frozen_baseline_25pct_00631l",
            {
                "strong_bull": "00631L.TW",
                "recovery_bull": "00631L.TW",
                "range_bound": "00631L.TW",
                "correction_bear": "00631L.TW",
            },
            {
                "strong_bull": 1.0,
                "recovery_bull": 1.0,
                "range_bound": 1.0,
                "correction_bear": 1.0,
            },
            "00631L.TW",
            0.25,
        ),
        (
            "balanced_00631l_0050_cash",
            {
                "strong_bull": "00631L.TW",
                "recovery_bull": "0050.TW",
                "range_bound": "0050.TW",
                "correction_bear": "0050.TW",
            },
            {
                "strong_bull": 1.0,
                "recovery_bull": 1.0,
                "range_bound": 0.50,
                "correction_bear": 0.10,
            },
            "0050.TW",
            0.25,
        ),
        (
            "aggressive_00631l_partial_0050",
            {
                "strong_bull": "00631L.TW",
                "recovery_bull": "00631L.TW",
                "range_bound": "0050.TW",
                "correction_bear": "0050.TW",
            },
            {
                "strong_bull": 1.0,
                "recovery_bull": 0.75,
                "range_bound": 0.75,
                "correction_bear": 0.25,
            },
            "0050.TW",
            0.25,
        ),
        (
            "defensive_00631l_then_0050_cash",
            {
                "strong_bull": "00631L.TW",
                "recovery_bull": "0050.TW",
                "range_bound": "0050.TW",
                "correction_bear": "0050.TW",
            },
            {
                "strong_bull": 1.0,
                "recovery_bull": 0.75,
                "range_bound": 0.25,
                "correction_bear": 0.0,
            },
            None,
            0.0,
        ),
        (
            "full_0050_below_strong_bull",
            {
                "strong_bull": "00631L.TW",
                "recovery_bull": "0050.TW",
                "range_bound": "0050.TW",
                "correction_bear": "0050.TW",
            },
            {
                "strong_bull": 1.0,
                "recovery_bull": 1.0,
                "range_bound": 1.0,
                "correction_bear": 1.0,
            },
            "0050.TW",
            0.25,
        ),
    )
    return tuple(
        RegimeModeSwitchVariant(
            **{
                **base_values,
                "name": f"cycle_market_ladder_{name}",
                "defense_anchor_ticker_by_regime": tickers,
                "attack_gate_defense_exposure_by_regime": exposures,
                "market_risk_off_mode": MODE_CASH if risk_ticker is None else MODE_0050_DEFENSE,
                "market_risk_off_defense_ticker": risk_ticker,
                "market_risk_off_defense_rule": "ma200",
                "market_risk_off_exposure": risk_exposure,
            }
        )
        for name, tickers, exposures, risk_ticker, risk_exposure in ladders
    )


def simulate_regime_mode_switch(
    *,
    name: str,
    prices_by_ticker: dict[str, pd.DataFrame],
    asset_types: dict[str, str],
    market_prices: pd.DataFrame,
    start_date: str,
    end_date: str,
    initial_cash: float,
    cost_model: TaiwanCostModel,
    variant: RegimeModeSwitchVariant,
    dividend_series_by_ticker: dict[str, pd.Series] | None = None,
    exposure_overlay: ExposureOverlay | None = None,
    target_selection_overlay: TargetSelectionOverlay | None = None,
    candidate_universe_tickers: tuple[str, ...] | None = None,
    candidate_universe_by_date: dict[str, tuple[str, ...]] | None = None,
    initial_state: RegimeModeSwitchState | None = None,
) -> BacktestResult:
    trade_dates = (
        _dynamic_candidate_universe_trade_dates(
            prices_by_ticker=prices_by_ticker,
            market_prices=market_prices,
            start_date=start_date,
            end_date=end_date,
            variant=variant,
            candidate_universe_tickers=candidate_universe_tickers,
            candidate_universe_by_date=candidate_universe_by_date,
        )
        if candidate_universe_by_date
        else _common_trade_dates(prices_by_ticker, start_date, end_date)
    )
    if not trade_dates:
        raise ValueError(f"No common trade dates between {start_date} and {end_date}")
    account = (
        _Account(
            cash=float(initial_state.account_cash),
            ticker=initial_state.account_ticker,
            shares=int(initial_state.account_shares),
        )
        if initial_state is not None
        else _Account(cash=float(initial_cash))
    )
    trades: list[Trade] = []
    equity_rows: list[dict] = []
    last_week_key: tuple[int, int] | None = (
        (initial_state.last_week_key_year, initial_state.last_week_key_week)
        if initial_state is not None
        and initial_state.last_week_key_year is not None
        and initial_state.last_week_key_week is not None
        else None
    )
    peak_signal_value = float(initial_state.peak_signal_value) if initial_state is not None else float(initial_cash)
    cooldown_until_index = _cooldown_index_from_state(initial_state, trade_dates)
    risk_off_active = bool(initial_state.risk_off_active) if initial_state is not None else False
    risk_off_clear_streak = int(initial_state.risk_off_clear_streak) if initial_state is not None else 0
    daily_health_active = bool(initial_state.daily_health_active) if initial_state is not None else True
    daily_health_recovery_streak = int(initial_state.daily_health_recovery_streak) if initial_state is not None else 0
    stop_latch_active = bool(initial_state.stop_latch_active) if initial_state is not None else False
    stop_release_streak = int(initial_state.stop_release_streak) if initial_state is not None else 0
    if initial_state is None:
        prior_attack_activation = _had_prior_attack_gate_activation(
            prices_by_ticker=prices_by_ticker,
            first_trade_date=trade_dates[0],
            variant=variant,
            candidate_universe_tickers=candidate_universe_tickers,
            candidate_universe_by_date=candidate_universe_by_date,
        )
        attack_gate_active = variant.attack_gate_margin_over_fallback is None or (
            variant.attack_gate_initialize_active_from_history and prior_attack_activation
        )
        attack_gate_activation_streak = 0
        attack_gate_ever_activated = attack_gate_active or prior_attack_activation
        attack_gate_stop_latch_active = False
        attack_gate_stop_release_streak = 0
        current_regime: str | None = None
        regime_streak_days = 0
    else:
        attack_gate_active = bool(initial_state.attack_gate_active)
        attack_gate_activation_streak = int(initial_state.attack_gate_activation_streak)
        attack_gate_ever_activated = bool(initial_state.attack_gate_ever_activated)
        attack_gate_stop_latch_active = bool(initial_state.attack_gate_stop_latch_active)
        attack_gate_stop_release_streak = int(initial_state.attack_gate_stop_release_streak)
        current_regime = initial_state.current_regime
        regime_streak_days = int(initial_state.regime_streak_days)

    for index, trade_date in enumerate(trade_dates):
        stop_triggered_today = False
        overlay_risk_flag = False
        overlay_reason = ""
        overlay_signal_date = ""
        target_overlay_baseline_target = ""
        target_overlay_target = ""
        target_overlay_reason = ""
        target_overlay_signal_date = ""
        target_overlay_changed = False
        if account.ticker is not None and dividend_series_by_ticker is not None:
            dividend = float(dividend_series_by_ticker[account.ticker].get(trade_date, 0.0))
            if dividend > 0:
                account.cash += account.shares * dividend
                trades.append(
                    Trade(
                        _date_str(trade_date),
                        account.ticker,
                        "dividend",
                        account.shares,
                        dividend,
                        account.shares * dividend,
                        0,
                        account.cash,
                        "cash_dividend",
                    )
                )

        trade_universe_tickers = _candidate_universe_for_date(
            trade_date,
            candidate_universe_tickers=candidate_universe_tickers,
            candidate_universe_by_date=candidate_universe_by_date,
        )
        active_prices_for_trade_date = _active_prices_for_candidate_universe(
            prices_by_ticker=prices_by_ticker,
            signal_date=trade_date,
            variant=variant,
            candidate_universe_tickers=trade_universe_tickers,
            current_ticker=account.ticker,
        )
        signal_date = previous_available_date({**active_prices_for_trade_date, "__market__": market_prices}, trade_date)
        signal_universe_tickers = _candidate_universe_for_date(
            signal_date,
            candidate_universe_tickers=trade_universe_tickers,
            candidate_universe_by_date=candidate_universe_by_date,
        )
        active_prices = _active_prices_for_candidate_universe(
            prices_by_ticker=prices_by_ticker,
            signal_date=signal_date,
            variant=variant,
            candidate_universe_tickers=signal_universe_tickers,
            current_ticker=account.ticker,
        )
        state_evaluation_day = (
            variant.state_evaluation_weekday is None
            or signal_date.weekday() == variant.state_evaluation_weekday
        )
        signal_prices = {ticker: _signal_close(prices, signal_date) for ticker, prices in active_prices.items()}
        if variant.portfolio_stop_drawdown_pct is not None and state_evaluation_day:
            signal_value = _market_value(account, signal_prices)
            peak_signal_value = max(peak_signal_value, signal_value)
            if (
                account.ticker is not None
                and peak_signal_value > 0
                and signal_value / peak_signal_value - 1 <= -variant.portfolio_stop_drawdown_pct
            ):
                _rebalance(
                    account=account,
                    trades=trades,
                    trade_date=trade_date,
                    target=None,
                    target_exposure=0.0,
                    prices_by_ticker=active_prices_for_trade_date,
                    asset_types=asset_types,
                    cost_model=cost_model,
                    reason=f"regime_mode_switch_portfolio_stop_{variant.name}",
                )
                peak_signal_value = account.cash
                cooldown_until_index = max(cooldown_until_index, index + variant.portfolio_stop_cooldown_days)
                if variant.portfolio_stop_latch_mode is not None:
                    stop_latch_active = True
                    stop_release_streak = 0
                    stop_triggered_today = True
                if variant.attack_gate_margin_over_fallback is not None:
                    attack_gate_active = False
                    attack_gate_activation_streak = 0
                    if variant.attack_gate_stop_latch_ticker is not None:
                        attack_gate_stop_latch_active = True
                        attack_gate_stop_release_streak = 0
                    stop_triggered_today = True

        regime = classify_market_regime(market_prices, signal_date, universe_prices=active_prices).regime
        if regime == current_regime:
            regime_streak_days += 1
        else:
            current_regime = regime
            regime_streak_days = 1
        mode = variant.regime_modes[regime]
        exposure = variant.regime_exposures[regime]
        defense_override_rule: str | None = None
        defense_override_ticker: str | None = None
        if stop_latch_active and not stop_triggered_today:
            raw_release = _portfolio_stop_release_condition(
                market_prices,
                signal_date,
                regime,
                variant.portfolio_stop_release_filter,
                prices_by_ticker=active_prices,
                variant=variant,
            )
            stop_latch_active, stop_release_streak = _update_latch_release_state(
                active=stop_latch_active,
                release_streak=stop_release_streak,
                raw_release=raw_release,
                release_confirmation_days=variant.portfolio_stop_release_confirmation_days,
            )
        if stop_latch_active:
            mode = variant.portfolio_stop_latch_mode or MODE_CASH
            exposure = 1.0 if mode == MODE_0050_DEFENSE else 0.0
            defense_override_rule = variant.portfolio_stop_latch_defense_rule
        if variant.attack_gate_margin_over_fallback is not None:
            if state_evaluation_day:
                if regime == "systemic_bear":
                    attack_gate_active = False
                    attack_gate_activation_streak = 0
                    if variant.attack_gate_reentry_margin_over_fallback is not None:
                        attack_gate_ever_activated = False
                elif not attack_gate_active and not stop_triggered_today:
                    raw_attack_gate = _attack_gate_passes(
                        active_prices,
                        signal_date,
                        variant,
                        use_reentry_rules=attack_gate_ever_activated,
                        candidate_universe_tickers=signal_universe_tickers,
                        candidate_universe_by_date=candidate_universe_by_date,
                    )
                    attack_gate_active, attack_gate_activation_streak = _update_attack_gate_state(
                        active=attack_gate_active,
                        activation_streak=attack_gate_activation_streak,
                        raw_activation=raw_attack_gate,
                        activation_confirmation_days=variant.attack_gate_activation_confirmation_days,
                    )
                    attack_gate_ever_activated = attack_gate_ever_activated or attack_gate_active
                if attack_gate_active:
                    attack_gate_stop_latch_active = False
                    attack_gate_stop_release_streak = 0
                elif attack_gate_stop_latch_active and not stop_triggered_today:
                    raw_stop_release = _attack_gate_stop_release_condition(
                        prices_by_ticker=prices_by_ticker,
                        signal_date=signal_date,
                        regime=regime,
                        variant=variant,
                    )
                    attack_gate_stop_latch_active, attack_gate_stop_release_streak = _update_latch_release_state(
                        active=attack_gate_stop_latch_active,
                        release_streak=attack_gate_stop_release_streak,
                        raw_release=raw_stop_release,
                        release_confirmation_days=variant.attack_gate_stop_release_confirmation_days,
                    )
            if not attack_gate_active and mode != MODE_CASH:
                mode = MODE_0050_DEFENSE
                exposure = (variant.attack_gate_defense_exposure_by_regime or {}).get(regime, 1.0)
                if attack_gate_stop_latch_active:
                    defense_override_ticker = variant.attack_gate_stop_latch_ticker
                    defense_override_rule = variant.attack_gate_stop_latch_rule
                else:
                    defense_override_rule = (variant.attack_gate_defense_rule_by_regime or {}).get(
                        regime,
                        variant.attack_gate_defense_rule,
                    )
        if state_evaluation_day:
            if (
                variant.market_risk_off_only_when_attack_gate_inactive
                and attack_gate_active
            ) or (
                variant.market_risk_off_only_before_first_attack_activation
                and attack_gate_ever_activated
            ):
                risk_off_active = False
                risk_off_clear_streak = 0
            else:
                raw_risk_off = bool(
                    variant.market_risk_off_filter
                    and _market_risk_off(market_prices, signal_date, variant.market_risk_off_filter)
                )
                risk_off_active, risk_off_clear_streak = _update_risk_off_state(
                    active=risk_off_active,
                    clear_streak=risk_off_clear_streak,
                    raw_risk_off=raw_risk_off,
                    exit_confirmation_days=variant.market_risk_off_exit_confirmation_days,
                )
        if risk_off_active:
            mode = variant.market_risk_off_mode
            exposure = (
                _market_risk_off_dynamic_exposure(active_prices, signal_date, regime, variant)
                if variant.market_risk_off_exposure_selector
                else variant.market_risk_off_exposure
                if variant.market_risk_off_exposure is not None
                or variant.market_risk_off_exposure_selector
                else 1.0 if mode == MODE_0050_DEFENSE else 0.0
            )
            defense_override_rule = variant.market_risk_off_defense_rule
            defense_override_ticker = variant.market_risk_off_defense_ticker
        if mode == MODE_DAILY and variant.daily_health_lookback_days is not None:
            raw_daily_health = _daily_health_gate_passes(active_prices, signal_date, variant)
            daily_health_active, daily_health_recovery_streak = _update_health_gate_state(
                active=daily_health_active,
                recovery_streak=daily_health_recovery_streak,
                raw_healthy=raw_daily_health,
                recovery_confirmation_days=variant.daily_health_recovery_confirmation_days,
            )
            if not daily_health_active:
                mode = variant.daily_health_fail_mode
                exposure = 1.0 if mode == MODE_0050_DEFENSE else 0.0
        should_check = mode in {MODE_DAILY, MODE_CASH, MODE_0050_DEFENSE}
        normal_rebalance_rule_applied = False
        if (
            mode == MODE_DAILY
            and variant.normal_rebalance_weekday_by_regime is not None
            and regime in variant.normal_rebalance_weekday_by_regime
            and regime_streak_days >= variant.normal_rebalance_min_regime_streak_days
        ):
            should_check = signal_date.weekday() == variant.normal_rebalance_weekday_by_regime[regime]
            normal_rebalance_rule_applied = True
        elif mode == MODE_DAILY and variant.normal_rebalance_weekday is not None:
            should_check = signal_date.weekday() == variant.normal_rebalance_weekday
            normal_rebalance_rule_applied = True
        if (
            mode == MODE_DAILY
            and not normal_rebalance_rule_applied
            and (
                variant.normal_rebalance_last_trading_day_of_week
                or regime in variant.normal_rebalance_last_trading_day_regimes
            )
        ):
            should_check = _is_last_trading_day_of_week(trade_dates, signal_date)
        if mode == MODE_WEEKLY:
            week_key = (signal_date.isocalendar().year, signal_date.isocalendar().week)
            should_check = week_key != last_week_key
            if variant.weekly_signal_weekday is not None:
                should_check = signal_date.weekday() == variant.weekly_signal_weekday and week_key != last_week_key
            if should_check:
                last_week_key = week_key

        if index <= cooldown_until_index:
            should_check = False
        if variant.state_evaluation_weekday is not None and not state_evaluation_day:
            should_check = False
        if (
            not should_check
            and mode == MODE_DAILY
            and regime in variant.early_rebalance_regimes
            and _early_rebalance_signal(
                prices_by_ticker=prices_by_ticker,
                signal_date=signal_date,
                current_ticker=account.ticker,
                variant=variant,
                candidate_universe_tickers=signal_universe_tickers,
            )
        ):
            should_check = True
        if (
            should_check
            and mode == MODE_DAILY
            and variant.defer_rebalance_on_adverse_open_gap_pct is not None
            and _has_adverse_open_gap(
                market_prices=market_prices,
                signal_date=signal_date,
                trade_date=trade_date,
                threshold_pct=variant.defer_rebalance_on_adverse_open_gap_pct,
            )
        ):
            should_check = False
        if exposure_overlay is not None and account.ticker is not None:
            signal_value = _market_value(account, signal_prices)
            current_position_value = account.shares * signal_prices[account.ticker]
            current_exposure = current_position_value / signal_value if signal_value > 0 else 0.0
            current_overlay = exposure_overlay(account.ticker, trade_date, signal_date, current_exposure)
            if current_overlay.risk_flag:
                overlay_risk_flag = True
                overlay_reason = current_overlay.reason
                overlay_signal_date = current_overlay.signal_date
                if current_overlay.adjusted_exposure < current_exposure - 0.02:
                    should_check = True

        if should_check:
            if mode == MODE_0050_DEFENSE:
                target, target_exposure = _defense_target_and_exposure(
                    active_prices,
                    signal_date,
                    regime,
                    exposure,
                    variant,
                    override_rule=defense_override_rule,
                    override_ticker=defense_override_ticker,
                )
            else:
                target = _select_target(
                    mode,
                    active_prices,
                    signal_date,
                    regime,
                    variant,
                    candidate_universe_tickers=signal_universe_tickers,
                )
                target_exposure = exposure if target else 0.0
            if target is None and mode == MODE_CASH and variant.fallback_ticker:
                target = variant.fallback_ticker
                target_exposure = variant.fallback_exposure
            if target_selection_overlay is not None:
                baseline_target = target
                selection_decision = target_selection_overlay(
                    mode,
                    prices_by_ticker,
                    trade_date,
                    signal_date,
                    regime,
                    variant,
                    baseline_target,
                    target_exposure,
                )
                target = selection_decision.target
                if target is None:
                    target_exposure = 0.0
                target_overlay_baseline_target = baseline_target or "cash"
                target_overlay_target = target or "cash"
                target_overlay_reason = selection_decision.reason
                target_overlay_signal_date = selection_decision.signal_date
                target_overlay_changed = target != baseline_target
            overlay_decision: ExposureOverlayDecision | None = None
            if exposure_overlay is not None and target is not None and target_exposure > 0:
                overlay_decision = exposure_overlay(target, trade_date, signal_date, target_exposure)
                if overlay_decision.risk_flag:
                    overlay_risk_flag = True
                    overlay_reason = overlay_decision.reason
                    overlay_signal_date = overlay_decision.signal_date
                    target_exposure = min(target_exposure, overlay_decision.adjusted_exposure)
            _rebalance(
                account=account,
                trades=trades,
                trade_date=trade_date,
                target=target,
                target_exposure=target_exposure,
                prices_by_ticker=active_prices_for_trade_date,
                asset_types=asset_types,
                cost_model=cost_model,
                reason=(
                    f"regime_mode_switch_{regime}_{mode}_{variant.name}"
                    + (f"_overlay_{overlay_reason}" if overlay_decision and overlay_decision.risk_flag else "")
                ),
            )

        close_prices = {
            ticker: float(prices.loc[trade_date, "close"])
            for ticker, prices in active_prices_for_trade_date.items()
            if trade_date in prices.index
        }
        total_value = _market_value(account, close_prices)
        position_value = account.shares * close_prices[account.ticker] if account.ticker else 0.0
        equity_row = {
            "date": trade_date,
            "total_value": total_value,
            "current_ticker": account.ticker or "cash",
            "current_shares": account.shares,
            "cash_balance": account.cash,
            "current_exposure": position_value / total_value if total_value > 0 else 0.0,
            "regime": regime,
            "regime_streak_days": regime_streak_days,
            "mode": mode,
            "risk_off_active": risk_off_active,
            "risk_off_clear_streak": risk_off_clear_streak,
            "daily_health_active": daily_health_active,
            "daily_health_recovery_streak": daily_health_recovery_streak,
            "stop_latch_active": stop_latch_active,
            "stop_release_streak": stop_release_streak,
            "attack_gate_active": attack_gate_active,
            "attack_gate_activation_streak": attack_gate_activation_streak,
            "attack_gate_ever_activated": attack_gate_ever_activated,
            "attack_gate_stop_latch_active": attack_gate_stop_latch_active,
            "attack_gate_stop_release_streak": attack_gate_stop_release_streak,
        }
        if exposure_overlay is not None:
            equity_row.update(
                {
                    "overlay_risk_flag": overlay_risk_flag,
                    "overlay_reason": overlay_reason,
                    "overlay_signal_date": overlay_signal_date,
                }
            )
        if target_selection_overlay is not None:
            equity_row.update(
                {
                    "target_overlay_baseline_target": target_overlay_baseline_target,
                    "target_overlay_target": target_overlay_target,
                    "target_overlay_reason": target_overlay_reason,
                    "target_overlay_signal_date": target_overlay_signal_date,
                    "target_overlay_changed": target_overlay_changed,
                }
            )
        equity_rows.append(equity_row)

    equity_curve = pd.DataFrame(equity_rows).set_index("date")
    final_value = float(equity_curve["total_value"].iloc[-1])
    result = BacktestResult(
        name=name,
        final_value=final_value,
        total_return=final_value / initial_cash - 1,
        max_drawdown=_max_drawdown(equity_curve["total_value"]),
        trades=trades,
        equity_curve=equity_curve,
    )
    result.regime_mode_switch_state = RegimeModeSwitchState(
        account_cash=float(account.cash),
        account_ticker=account.ticker,
        account_shares=int(account.shares),
        last_week_key_year=last_week_key[0] if last_week_key else None,
        last_week_key_week=last_week_key[1] if last_week_key else None,
        peak_signal_value=float(peak_signal_value),
        cooldown_until_date=_cooldown_date_from_index(cooldown_until_index, trade_dates),
        risk_off_active=bool(risk_off_active),
        risk_off_clear_streak=int(risk_off_clear_streak),
        daily_health_active=bool(daily_health_active),
        daily_health_recovery_streak=int(daily_health_recovery_streak),
        stop_latch_active=bool(stop_latch_active),
        stop_release_streak=int(stop_release_streak),
        attack_gate_active=bool(attack_gate_active),
        attack_gate_activation_streak=int(attack_gate_activation_streak),
        attack_gate_ever_activated=bool(attack_gate_ever_activated),
        attack_gate_stop_latch_active=bool(attack_gate_stop_latch_active),
        attack_gate_stop_release_streak=int(attack_gate_stop_release_streak),
        current_regime=current_regime,
        regime_streak_days=int(regime_streak_days),
    )
    return result


def _select_target(
    mode: str,
    prices_by_ticker: dict[str, pd.DataFrame],
    signal_date: pd.Timestamp,
    regime: str,
    variant: RegimeModeSwitchVariant,
    *,
    candidate_universe_tickers: tuple[str, ...] | None = None,
) -> str | None:
    if mode == MODE_DAILY:
        min_margin = _min_score_margin_for_regime(variant, regime, prices_by_ticker, signal_date)
        if (
            variant.candidate_trend_filter is None
            and min_margin is None
            and not variant.attack_selection_exclude_tickers
            and candidate_universe_tickers is None
        ):
            return relative_strength_top1(prices_by_ticker, signal_date)
        scores = relative_strength_scores(prices_by_ticker, signal_date)
        fallback_score = scores.get(variant.relative_score_fallback_ticker or "", None)
        eligible = {
            ticker: score
            for ticker, score in scores.items()
            if (
                _candidate_universe_allows(ticker, candidate_universe_tickers)
                and
                ticker not in variant.attack_selection_exclude_tickers
                and (
                variant.candidate_trend_filter is None
                or _passes_candidate_filter(prices_by_ticker[ticker], signal_date, variant.candidate_trend_filter)
                )
            )
        }
        if not eligible:
            return None
        top_ticker, top_score = max(eligible.items(), key=lambda item: (item[1], item[0]))
        if (
            top_ticker == variant.relative_score_fallback_ticker
            and variant.relative_score_fallback_defense_rule
            and not risk_on_for_rule(prices_by_ticker[top_ticker], signal_date, variant.relative_score_fallback_defense_rule)
        ):
            return None
        if min_margin is not None and fallback_score is not None:
            if top_ticker != variant.relative_score_fallback_ticker and top_score - fallback_score < min_margin:
                fallback = variant.relative_score_fallback_ticker
                if (
                    fallback
                    and variant.relative_score_fallback_defense_rule
                    and not risk_on_for_rule(prices_by_ticker[fallback], signal_date, variant.relative_score_fallback_defense_rule)
                ):
                    return None
                return fallback
        return top_ticker
    if mode == MODE_WEEKLY:
        return dual_momentum_vol_control(prices_by_ticker, signal_date)
    if mode == MODE_0050_DEFENSE:
        anchor = (variant.defense_anchor_ticker_by_regime or {}).get(regime, variant.defense_anchor_ticker)
        if anchor not in prices_by_ticker:
            return None
        rule = (variant.defense_rule_by_regime or {}).get(regime)
        if rule is None:
            return anchor
        return anchor if risk_on_for_rule(prices_by_ticker[anchor], signal_date, rule) else None
    if mode in {MODE_CASH, MODE_HOLD}:
        return None
    raise ValueError(f"Unsupported mode: {mode}")


def _early_rebalance_signal(
    *,
    prices_by_ticker: dict[str, pd.DataFrame],
    signal_date: pd.Timestamp,
    current_ticker: str | None,
    variant: RegimeModeSwitchVariant,
    candidate_universe_tickers: tuple[str, ...] | None = None,
) -> bool:
    if current_ticker is None:
        return False
    scores = relative_strength_scores(prices_by_ticker, signal_date)
    eligible = {
        ticker: score
        for ticker, score in scores.items()
        if (
            _candidate_universe_allows(ticker, candidate_universe_tickers)
            and
            ticker not in variant.attack_selection_exclude_tickers
            and (
                variant.candidate_trend_filter is None
                or _passes_candidate_filter(prices_by_ticker[ticker], signal_date, variant.candidate_trend_filter)
            )
        )
    }
    if len(eligible) < 2:
        return False
    ranked = sorted(eligible.items(), key=lambda item: (item[1], item[0]), reverse=True)
    top_ticker, top_score = ranked[0]
    if top_ticker == current_ticker:
        return False
    second_score = ranked[1][1]
    min_top_gap = variant.early_rebalance_min_top_gap
    if min_top_gap is not None and top_score - second_score < min_top_gap:
        return False
    min_gap_over_current = variant.early_rebalance_min_gap_over_current
    if min_gap_over_current is None:
        return True
    current_score = eligible.get(current_ticker)
    if current_score is None:
        return top_score - second_score >= min_gap_over_current
    return top_score - current_score >= min_gap_over_current


def _defense_target_and_exposure(
    prices_by_ticker: dict[str, pd.DataFrame],
    signal_date: pd.Timestamp,
    regime: str,
    risk_on_exposure: float,
    variant: RegimeModeSwitchVariant,
    override_rule: str | None = None,
    override_ticker: str | None = None,
) -> tuple[str | None, float]:
    anchor = override_ticker or (variant.defense_anchor_ticker_by_regime or {}).get(regime, variant.defense_anchor_ticker)
    if anchor not in prices_by_ticker:
        return None, 0.0
    rule = override_rule or (variant.defense_rule_by_regime or {}).get(regime)
    if rule is None and variant.daily_health_lookback_days is not None:
        rule = variant.daily_health_fail_defense_rule
    if rule is None or risk_on_for_rule(prices_by_ticker[anchor], signal_date, rule):
        return anchor, risk_on_exposure
    risk_off_exposure = (variant.defense_risk_off_exposure_by_regime or {}).get(regime, 0.0)
    return (anchor, risk_off_exposure) if risk_off_exposure > 0 else (None, 0.0)


def _daily_health_gate_passes(
    prices_by_ticker: dict[str, pd.DataFrame],
    signal_date: pd.Timestamp,
    variant: RegimeModeSwitchVariant,
) -> bool:
    lookback = variant.daily_health_lookback_days
    if lookback is None:
        return True
    return _daily_health_gate_passes_values(
        prices_by_ticker=prices_by_ticker,
        signal_date=signal_date,
        lookback=lookback,
        minimum_excess=variant.daily_health_min_excess_return,
        benchmark=variant.daily_health_benchmark_ticker,
    )


def _daily_health_gate_passes_values(
    *,
    prices_by_ticker: dict[str, pd.DataFrame],
    signal_date: pd.Timestamp,
    lookback: int,
    minimum_excess: float,
    benchmark: str,
) -> bool:
    if benchmark not in prices_by_ticker:
        return False
    cache_key = (
        tuple(sorted((ticker, id(frame)) for ticker, frame in prices_by_ticker.items())),
        pd.Timestamp(signal_date).value,
        lookback,
        minimum_excess,
        benchmark,
    )
    if cache_key in _DAILY_HEALTH_CACHE:
        return _DAILY_HEALTH_CACHE[cache_key]
    trade_dates = _common_dates_until(prices_by_ticker, signal_date)
    if len(trade_dates) < lookback + 2:
        return False
    window_dates = trade_dates[-(lookback + 1) :]
    strategy_value = 1.0
    benchmark_value = 1.0
    for index in range(1, len(window_dates)):
        decision_date = window_dates[index - 1]
        current_date = window_dates[index]
        scores = relative_strength_scores(prices_by_ticker, decision_date)
        if scores:
            target = max(scores.items(), key=lambda item: (item[1], item[0]))[0]
            strategy_value *= _period_return(prices_by_ticker[target], decision_date, current_date)
        benchmark_value *= _period_return(prices_by_ticker[benchmark], decision_date, current_date)
    passes = strategy_value / benchmark_value - 1 >= minimum_excess
    _DAILY_HEALTH_CACHE[cache_key] = passes
    return passes


def _common_dates_until(prices_by_ticker: dict[str, pd.DataFrame], signal_date: pd.Timestamp) -> list[pd.Timestamp]:
    common: set[pd.Timestamp] | None = None
    for prices in prices_by_ticker.values():
        dates = set(prices.index[prices.index <= signal_date])
        common = dates if common is None else common & dates
    return sorted(common or set())


def _period_return(prices: pd.DataFrame, start_date: pd.Timestamp, end_date: pd.Timestamp) -> float:
    start = float(prices.loc[start_date, "adj_close"])
    end = float(prices.loc[end_date, "adj_close"])
    if start <= 0:
        return 1.0
    return end / start


def _signal_close(prices: pd.DataFrame, signal_date: pd.Timestamp) -> float:
    history = prices.loc[prices.index <= signal_date, "close"]
    if history.empty:
        return float(prices["close"].iloc[0])
    return float(history.iloc[-1])


def _passes_candidate_filter(prices: pd.DataFrame, signal_date: pd.Timestamp, filter_name: str) -> bool:
    history = prices.loc[prices.index <= signal_date, "adj_close"].dropna()
    if filter_name == "trend60_positive20":
        return _trend_filter(history, trend_window=60, return_window=20)
    if filter_name == "trend120_positive20":
        return _trend_filter(history, trend_window=120, return_window=20)
    raise ValueError(f"Unsupported candidate trend filter: {filter_name}")


def _trend_filter(history: pd.Series, *, trend_window: int, return_window: int) -> bool:
    if len(history) <= max(trend_window, return_window):
        return False
    close = float(history.iloc[-1])
    trend_average = float(history.iloc[-trend_window:].mean())
    recent_return = float(history.iloc[-1] / history.iloc[-return_window] - 1)
    return close > trend_average and recent_return > 0


def _market_risk_off(market_prices: pd.DataFrame, signal_date: pd.Timestamp, filter_name: str) -> bool:
    history = market_prices.loc[market_prices.index <= signal_date, "adj_close"].dropna()
    if len(history) < 130:
        return False
    close = float(history.iloc[-1])
    ma60 = float(history.iloc[-60:].mean())
    ma120 = float(history.iloc[-120:].mean())
    ma20 = float(history.iloc[-20:].mean())
    return_10 = float(history.iloc[-1] / history.iloc[-10] - 1) if len(history) > 10 else 0.0
    return_20 = float(history.iloc[-1] / history.iloc[-20] - 1) if len(history) > 20 else 0.0
    high_60 = float(history.iloc[-60:].max())
    drawdown_60 = close / high_60 - 1
    if filter_name == "dd5_ma60_ret20":
        return close < ma60 and return_20 < 0 and drawdown_60 <= -0.05
    if filter_name == "dd8_ma60_ret20":
        return close < ma60 and return_20 < 0 and drawdown_60 <= -0.08
    if filter_name == "ma60_ret20":
        return close < ma60 and return_20 < 0
    if filter_name == "risk_2of3":
        return sum((close < ma60, return_20 < 0, drawdown_60 <= -0.05)) >= 2
    if filter_name == "ma20_ret10":
        return close < ma20 and return_10 < 0
    if filter_name == "ret20_dd5":
        return return_20 < 0 and drawdown_60 <= -0.05
    if filter_name == "ma120_ret20":
        return close < ma120 and return_20 < 0
    raise ValueError(f"Unsupported market risk-off filter: {filter_name}")


def _market_risk_off_dynamic_exposure(
    prices_by_ticker: dict[str, pd.DataFrame],
    signal_date: pd.Timestamp,
    regime: str,
    variant: RegimeModeSwitchVariant,
) -> float:
    selector = variant.market_risk_off_exposure_selector
    if selector is None:
        return variant.market_risk_off_exposure or 0.0

    selector_parts = selector.split("_to_")
    if len(selector_parts) != 2:
        raise ValueError(f"Unsupported market risk-off exposure selector: {selector}")
    condition_name, exposure_name = selector_parts
    exposure_token = exposure_name.removesuffix("_else_cash")
    if not exposure_token.endswith("pct"):
        raise ValueError(f"Unsupported market risk-off exposure selector: {selector}")
    risk_on_exposure = float(exposure_token.removesuffix("pct")) / 100

    if condition_name == "0050_above_ma200_bull_regime":
        return risk_on_exposure if regime in {"strong_bull", "recovery_bull"} and _ticker_above_ma(
            prices_by_ticker, "0050.TW", signal_date, 200
        ) else 0.0
    if condition_name == "0050_above_ma200":
        return risk_on_exposure if _ticker_above_ma(prices_by_ticker, "0050.TW", signal_date, 200) else 0.0
    if condition_name == "0050_above_ma120":
        return risk_on_exposure if _ticker_above_ma(prices_by_ticker, "0050.TW", signal_date, 120) else 0.0
    if condition_name == "00631l_above_ma200":
        return risk_on_exposure if _ticker_above_ma(prices_by_ticker, "00631L.TW", signal_date, 200) else 0.0
    if condition_name == "00631l_above_ma120":
        return risk_on_exposure if _ticker_above_ma(prices_by_ticker, "00631L.TW", signal_date, 120) else 0.0
    raise ValueError(f"Unsupported market risk-off exposure selector: {selector}")


def _ticker_above_ma(
    prices_by_ticker: dict[str, pd.DataFrame],
    ticker: str,
    signal_date: pd.Timestamp,
    window: int,
) -> bool:
    prices = prices_by_ticker.get(ticker)
    if prices is None:
        return False
    history = prices.loc[prices.index <= signal_date, "adj_close"].dropna()
    if len(history) < window:
        return False
    return float(history.iloc[-1]) > float(history.iloc[-window:].mean())


def _is_last_trading_day_of_week(trade_dates: list[pd.Timestamp], signal_date: pd.Timestamp) -> bool:
    signal_week = (signal_date.isocalendar().year, signal_date.isocalendar().week)
    later_dates = [date for date in trade_dates if date > signal_date]
    if not later_dates:
        return True
    next_trade_date = later_dates[0]
    next_week = (next_trade_date.isocalendar().year, next_trade_date.isocalendar().week)
    return next_week != signal_week


def _has_adverse_open_gap(
    *,
    market_prices: pd.DataFrame,
    signal_date: pd.Timestamp,
    trade_date: pd.Timestamp,
    threshold_pct: float,
) -> bool:
    if signal_date not in market_prices.index or trade_date not in market_prices.index:
        return False
    signal_close = float(market_prices.loc[signal_date, "adj_close"])
    open_column = "open" if "open" in market_prices.columns else "adj_close"
    trade_open = float(market_prices.loc[trade_date, open_column])
    if signal_close <= 0:
        return False
    open_gap = trade_open / signal_close - 1
    return open_gap <= -abs(threshold_pct)


def _update_risk_off_state(
    *,
    active: bool,
    clear_streak: int,
    raw_risk_off: bool,
    exit_confirmation_days: int,
) -> tuple[bool, int]:
    if raw_risk_off:
        return True, 0
    if not active:
        return False, 0
    if exit_confirmation_days <= 0:
        return False, 0
    clear_streak += 1
    if clear_streak >= exit_confirmation_days:
        return False, 0
    return True, clear_streak


def _update_health_gate_state(
    *,
    active: bool,
    recovery_streak: int,
    raw_healthy: bool,
    recovery_confirmation_days: int,
) -> tuple[bool, int]:
    if not raw_healthy:
        return False, 0
    if active or recovery_confirmation_days <= 0:
        return True, 0
    recovery_streak += 1
    if recovery_streak >= recovery_confirmation_days:
        return True, 0
    return False, recovery_streak


def _portfolio_stop_release_condition(
    market_prices: pd.DataFrame,
    signal_date: pd.Timestamp,
    regime: str,
    filter_name: str | None,
    *,
    prices_by_ticker: dict[str, pd.DataFrame] | None = None,
    variant: RegimeModeSwitchVariant | None = None,
) -> bool:
    if filter_name is None:
        return True
    if filter_name == "strong_bull":
        return regime == "strong_bull"
    if filter_name == "strong_or_recovery_bull":
        return regime in {"strong_bull", "recovery_bull"}
    if filter_name == "ma60_ret20_positive":
        history = market_prices.loc[market_prices.index <= signal_date, "adj_close"].dropna()
        if len(history) < 60:
            return False
        return float(history.iloc[-1]) > float(history.iloc[-60:].mean()) and _period_return_from_series(history, 20) > 0
    if filter_name in {"daily_health", "strong_bull_and_daily_health"}:
        if prices_by_ticker is None or variant is None:
            return False
        if filter_name == "strong_bull_and_daily_health" and regime != "strong_bull":
            return False
        lookback = variant.portfolio_stop_release_health_lookback_days
        if lookback is None:
            return False
        return _daily_health_gate_passes_values(
            prices_by_ticker=prices_by_ticker,
            signal_date=signal_date,
            lookback=lookback,
            minimum_excess=variant.portfolio_stop_release_health_min_excess_return,
            benchmark=variant.daily_health_benchmark_ticker,
        )
    raise ValueError(f"Unsupported portfolio stop release filter: {filter_name}")


def _update_latch_release_state(
    *,
    active: bool,
    release_streak: int,
    raw_release: bool,
    release_confirmation_days: int,
) -> tuple[bool, int]:
    if not active:
        return False, 0
    if not raw_release:
        return True, 0
    if release_confirmation_days <= 0:
        return False, 0
    release_streak += 1
    if release_streak >= release_confirmation_days:
        return False, 0
    return True, release_streak


def _period_return_from_series(history: pd.Series, window: int) -> float:
    if len(history) <= window:
        return 0.0
    return float(history.iloc[-1] / history.iloc[-window] - 1)


def _candidate_universe_allows(ticker: str, candidate_universe_tickers: tuple[str, ...] | None) -> bool:
    return candidate_universe_tickers is None or ticker in candidate_universe_tickers


def _candidate_universe_for_date(
    signal_date: pd.Timestamp,
    *,
    candidate_universe_tickers: tuple[str, ...] | None,
    candidate_universe_by_date: dict[str, tuple[str, ...]] | None,
) -> tuple[str, ...] | None:
    if not candidate_universe_by_date:
        return candidate_universe_tickers
    return candidate_universe_by_date.get(_date_str(pd.Timestamp(signal_date)), candidate_universe_tickers)


def _required_tickers_for_candidate_universe(
    *,
    variant: RegimeModeSwitchVariant,
    candidate_universe_tickers: tuple[str, ...] | None,
    current_ticker: str | None = None,
) -> set[str]:
    required = set(candidate_universe_tickers or ())
    for ticker in (
        variant.relative_score_fallback_ticker,
        variant.attack_gate_fallback_ticker,
        variant.defense_anchor_ticker,
        variant.fallback_ticker,
        current_ticker,
    ):
        if ticker:
            required.add(ticker)
    required.update(variant.attack_selection_exclude_tickers)
    required.update(variant.attack_gate_exclude_tickers)
    for mapping in (variant.defense_anchor_ticker_by_regime,):
        if mapping:
            required.update(ticker for ticker in mapping.values() if ticker)
    if variant.market_risk_off_defense_ticker:
        required.add(variant.market_risk_off_defense_ticker)
    if variant.attack_gate_stop_latch_ticker:
        required.add(variant.attack_gate_stop_latch_ticker)
    return required


def _active_prices_for_candidate_universe(
    *,
    prices_by_ticker: dict[str, pd.DataFrame],
    signal_date: pd.Timestamp,
    variant: RegimeModeSwitchVariant,
    candidate_universe_tickers: tuple[str, ...] | None,
    current_ticker: str | None = None,
) -> dict[str, pd.DataFrame]:
    if candidate_universe_tickers is None:
        return prices_by_ticker
    required = _required_tickers_for_candidate_universe(
        variant=variant,
        candidate_universe_tickers=candidate_universe_tickers,
        current_ticker=current_ticker,
    )
    if not required:
        return prices_by_ticker
    return {
        ticker: prices
        for ticker, prices in prices_by_ticker.items()
        if ticker in required and not prices.loc[prices.index <= signal_date].empty
    }


def _has_trade_price(prices: pd.DataFrame, trade_date: pd.Timestamp) -> bool:
    if trade_date not in prices.index:
        return False
    row = prices.loc[trade_date]
    for column in ("open", "close", "adj_close"):
        if column in row and pd.to_numeric(row[column], errors="coerce") <= 0:
            return False
    return True


def _dynamic_candidate_universe_trade_dates(
    *,
    prices_by_ticker: dict[str, pd.DataFrame],
    market_prices: pd.DataFrame,
    start_date: str,
    end_date: str,
    variant: RegimeModeSwitchVariant,
    candidate_universe_tickers: tuple[str, ...] | None,
    candidate_universe_by_date: dict[str, tuple[str, ...]] | None,
) -> list[pd.Timestamp]:
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    market_dates = [
        date
        for date in market_prices.index
        if start <= date <= end and _has_trade_price(market_prices, pd.Timestamp(date))
    ]
    trade_dates: list[pd.Timestamp] = []
    for trade_date in market_dates:
        universe = _candidate_universe_for_date(
            pd.Timestamp(trade_date),
            candidate_universe_tickers=candidate_universe_tickers,
            candidate_universe_by_date=candidate_universe_by_date,
        )
        required = _required_tickers_for_candidate_universe(
            variant=variant,
            candidate_universe_tickers=universe,
        )
        if not required:
            continue
        available = True
        for ticker in required:
            prices = prices_by_ticker.get(ticker)
            if prices is None or not _has_trade_price(prices, pd.Timestamp(trade_date)):
                available = False
                break
        if available:
            trade_dates.append(pd.Timestamp(trade_date))
    return trade_dates


def _cooldown_index_from_state(initial_state: RegimeModeSwitchState | None, trade_dates: list[pd.Timestamp]) -> int:
    if initial_state is None or not initial_state.cooldown_until_date:
        return -1
    cooldown_until_date = pd.Timestamp(initial_state.cooldown_until_date)
    matching = [index for index, trade_date in enumerate(trade_dates) if trade_date <= cooldown_until_date]
    return max(matching) if matching else -1


def _cooldown_date_from_index(cooldown_until_index: int, trade_dates: list[pd.Timestamp]) -> str:
    if cooldown_until_index < 0 or not trade_dates:
        return ""
    index = min(cooldown_until_index, len(trade_dates) - 1)
    return _date_str(trade_dates[index])


def _attack_gate_passes(
    prices_by_ticker: dict[str, pd.DataFrame],
    signal_date: pd.Timestamp,
    variant: RegimeModeSwitchVariant,
    *,
    use_reentry_rules: bool = False,
    candidate_universe_tickers: tuple[str, ...] | None = None,
    candidate_universe_by_date: dict[str, tuple[str, ...]] | None = None,
) -> bool:
    margin = (
        variant.attack_gate_reentry_margin_over_fallback
        if use_reentry_rules and variant.attack_gate_reentry_margin_over_fallback is not None
        else variant.attack_gate_margin_over_fallback
    )
    fallback = variant.attack_gate_fallback_ticker
    if margin is None or fallback not in prices_by_ticker:
        return True
    scores = relative_strength_scores(prices_by_ticker, signal_date)
    fallback_score = scores.get(fallback)
    if fallback_score is None or not scores:
        return False
    eligible_scores = {
        ticker: score
        for ticker, score in scores.items()
        if (
            _candidate_universe_allows(ticker, candidate_universe_tickers)
            and ticker != fallback
            and ticker not in variant.attack_gate_exclude_tickers
        )
    }
    if not eligible_scores:
        return False
    top_ticker, top_score = max(eligible_scores.items(), key=lambda item: (item[1], item[0]))
    if top_ticker == fallback or top_score - fallback_score < margin:
        return False
    minimum_acceleration = (
        variant.attack_gate_reentry_min_short_to_medium_momentum_ratio
        if use_reentry_rules and variant.attack_gate_reentry_min_short_to_medium_momentum_ratio is not None
        else variant.attack_gate_min_short_to_medium_momentum_ratio
    )
    if minimum_acceleration is not None:
        top_history = prices_by_ticker[top_ticker].loc[
            prices_by_ticker[top_ticker].index <= signal_date,
            "adj_close",
        ].dropna()
        short_return = _period_return_from_series(top_history, 20)
        medium_return = _period_return_from_series(top_history, 60)
        if medium_return <= 0 or short_return <= 0 or short_return / medium_return < minimum_acceleration:
            return False
    lookback = 0 if use_reentry_rules and variant.attack_gate_reentry_margin_over_fallback is not None else variant.attack_gate_persistence_lookback_days
    min_top_days = 0 if use_reentry_rules and variant.attack_gate_reentry_margin_over_fallback is not None else variant.attack_gate_min_top_days
    if lookback <= 0 or min_top_days <= 0:
        return True
    common_dates = _common_dates_until(prices_by_ticker, signal_date)
    prior_dates = common_dates[-(lookback + 1) : -1]
    if len(prior_dates) < lookback:
        return False
    prior_top_days = 0
    for prior_date in prior_dates:
        prior_universe_tickers = _candidate_universe_for_date(
            prior_date,
            candidate_universe_tickers=candidate_universe_tickers,
            candidate_universe_by_date=candidate_universe_by_date,
        )
        prior_scores = relative_strength_scores(prices_by_ticker, prior_date)
        prior_eligible_scores = {
            ticker: score
            for ticker, score in prior_scores.items()
            if (
                _candidate_universe_allows(ticker, prior_universe_tickers)
                and ticker != fallback
                and ticker not in variant.attack_gate_exclude_tickers
            )
        }
        if (
            prior_eligible_scores
            and max(prior_eligible_scores.items(), key=lambda item: (item[1], item[0]))[0] == top_ticker
        ):
            prior_top_days += 1
    return prior_top_days >= min_top_days


def _had_prior_attack_gate_activation(
    *,
    prices_by_ticker: dict[str, pd.DataFrame],
    first_trade_date: pd.Timestamp,
    variant: RegimeModeSwitchVariant,
    candidate_universe_tickers: tuple[str, ...] | None = None,
    candidate_universe_by_date: dict[str, tuple[str, ...]] | None = None,
) -> bool:
    history_days = variant.attack_gate_initialize_history_days
    if history_days <= 0 or variant.attack_gate_margin_over_fallback is None:
        return False
    common_dates = _common_dates_until(prices_by_ticker, first_trade_date)
    prior_dates = [date for date in common_dates if date < first_trade_date][-history_days:]
    return any(
        _attack_gate_passes(
            prices_by_ticker,
            date,
            variant,
            candidate_universe_tickers=_candidate_universe_for_date(
                date,
                candidate_universe_tickers=candidate_universe_tickers,
                candidate_universe_by_date=candidate_universe_by_date,
            ),
            candidate_universe_by_date=candidate_universe_by_date,
        )
        for date in prior_dates
    )


def _attack_gate_stop_release_condition(
    *,
    prices_by_ticker: dict[str, pd.DataFrame],
    signal_date: pd.Timestamp,
    regime: str,
    variant: RegimeModeSwitchVariant,
) -> bool:
    filter_name = variant.attack_gate_stop_release_filter
    if filter_name is None:
        return True
    fallback = variant.defense_anchor_ticker
    if fallback not in prices_by_ticker or variant.attack_gate_defense_rule is None:
        return False
    fallback_risk_on = risk_on_for_rule(prices_by_ticker[fallback], signal_date, variant.attack_gate_defense_rule)
    if filter_name == "strong_bull_fallback_risk_on":
        return regime == "strong_bull" and fallback_risk_on
    if filter_name == "strong_or_recovery_bull_fallback_risk_on":
        return regime in {"strong_bull", "recovery_bull"} and fallback_risk_on
    raise ValueError(f"Unsupported attack gate stop release filter: {filter_name}")


def _update_attack_gate_state(
    *,
    active: bool,
    activation_streak: int,
    raw_activation: bool,
    activation_confirmation_days: int,
) -> tuple[bool, int]:
    if active:
        return True, 0
    if not raw_activation:
        return False, 0
    if activation_confirmation_days <= 1:
        return True, 0
    activation_streak += 1
    if activation_streak >= activation_confirmation_days:
        return True, 0
    return False, activation_streak


def _min_score_margin_for_regime(
    variant: RegimeModeSwitchVariant,
    regime: str,
    prices_by_ticker: dict[str, pd.DataFrame],
    signal_date: pd.Timestamp,
) -> float | None:
    if variant.margin_gate_ticker is not None:
        gate_prices = prices_by_ticker.get(variant.margin_gate_ticker)
        gate_on = _leveraged_gate_on(gate_prices, signal_date) if gate_prices is not None else False
        return variant.gated_on_margin if gate_on else variant.gated_off_margin
    if variant.min_score_over_fallback_by_regime is not None:
        return variant.min_score_over_fallback_by_regime.get(regime, variant.min_score_over_fallback)
    return variant.min_score_over_fallback


def _leveraged_gate_on(prices: pd.DataFrame, signal_date: pd.Timestamp) -> bool:
    history = prices.loc[prices.index <= signal_date, "adj_close"].dropna()
    if len(history) <= 60:
        return False
    close = float(history.iloc[-1])
    ma60 = float(history.iloc[-60:].mean())
    return_20 = float(history.iloc[-1] / history.iloc[-20] - 1)
    return_60 = float(history.iloc[-1] / history.iloc[-60] - 1)
    return close > ma60 and return_20 > 0 and return_60 > 0
