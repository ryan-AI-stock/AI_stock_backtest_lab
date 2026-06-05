from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from backtest_lab.costs import TaiwanCostModel
from backtest_lab.market_regime import classify_market_regime
from backtest_lab.regime_policy import StrategyPolicy, policy_for
from backtest_lab.simulation import BacktestResult, _common_trade_dates, _date_str, _max_drawdown
from backtest_lab.strategies import (
    dual_momentum_scores,
    previous_available_date,
    relative_strength_scores,
)
from backtest_lab.portfolio import Trade


@dataclass(frozen=True)
class RegimePolicyVariant:
    name: str
    exposure_multiplier: float = 1.0
    strong_bull_multiplier: float | None = None
    recovery_bull_multiplier: float | None = None
    range_bound_multiplier: float | None = None
    correction_bear_multiplier: float | None = None
    systemic_bear_multiplier: float | None = None
    min_exposure_cap: float = 0.0
    score_margin_add: float = 0.0
    min_score_add: float = 0.0
    daily_risk_exit: bool = False
    holding_trend_window: int | None = None
    portfolio_stop_drawdown_pct: float | None = None
    portfolio_stop_cooldown_days: int = 0
    fallback_ticker: str | None = None
    fallback_regimes: tuple[str, ...] = ()
    fallback_exposure: float = 1.0
    weekly_signal_weekday: int | None = None


@dataclass
class _Account:
    cash: float
    ticker: str | None = None
    shares: int = 0


def simulate_regime_aware_strategy(
    *,
    name: str,
    strategy_id: str,
    prices_by_ticker: dict[str, pd.DataFrame],
    asset_types: dict[str, str],
    market_prices: pd.DataFrame,
    start_date: str,
    end_date: str,
    initial_cash: float,
    cost_model: TaiwanCostModel,
    variant: RegimePolicyVariant = RegimePolicyVariant(name="balanced"),
) -> BacktestResult:
    trade_dates = _common_trade_dates(prices_by_ticker, start_date, end_date)
    if not trade_dates:
        raise ValueError(f"No common trade dates between {start_date} and {end_date}")

    account = _Account(cash=float(initial_cash))
    trades: list[Trade] = []
    equity_rows: list[dict] = []
    last_week_key: tuple[int, int] | None = None
    weekly_check_count = 0
    peak_signal_value = float(initial_cash)
    cooldown_until_index = -1

    for index, trade_date in enumerate(trade_dates):
        signal_date = previous_available_date({**prices_by_ticker, "__market__": market_prices}, trade_date)
        should_check = strategy_id == "daily_strength"
        if strategy_id == "weekly_rotation":
            week_key = (signal_date.isocalendar().year, signal_date.isocalendar().week)
            if variant.weekly_signal_weekday is not None:
                should_check = signal_date.weekday() == variant.weekly_signal_weekday
                if should_check and week_key != last_week_key:
                    weekly_check_count += 1
                    last_week_key = week_key
                elif should_check:
                    should_check = False
            elif week_key != last_week_key:
                weekly_check_count += 1
                last_week_key = week_key
                should_check = True
        if variant.portfolio_stop_drawdown_pct is not None:
            signal_prices = {ticker: float(prices.loc[signal_date, "close"]) for ticker, prices in prices_by_ticker.items()}
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
                    prices_by_ticker=prices_by_ticker,
                    asset_types=asset_types,
                    cost_model=cost_model,
                    reason=f"portfolio_stop_{strategy_id}_{variant.name}",
                )
                peak_signal_value = account.cash
                should_check = False
                cooldown_until_index = max(cooldown_until_index, index + variant.portfolio_stop_cooldown_days)
        if variant.daily_risk_exit and account.ticker is not None:
            regime = classify_market_regime(market_prices, signal_date, universe_prices=prices_by_ticker)
            if _should_daily_exit(account.ticker, prices_by_ticker, signal_date, regime.regime, variant):
                _rebalance(
                    account=account,
                    trades=trades,
                    trade_date=trade_date,
                    target=None,
                    target_exposure=0.0,
                    prices_by_ticker=prices_by_ticker,
                    asset_types=asset_types,
                    cost_model=cost_model,
                    reason=f"daily_risk_exit_{strategy_id}_{regime.regime}_{variant.name}",
                )
                should_check = False
        if index <= cooldown_until_index:
            should_check = False

        if should_check:
            regime = classify_market_regime(market_prices, signal_date, universe_prices=prices_by_ticker)
            base_policy = policy_for(strategy_id, regime.regime)
            policy = _variant_policy(base_policy, variant)
            if strategy_id == "weekly_rotation" and policy.rebalance_frequency == "biweekly" and weekly_check_count % 2 == 0:
                target = account.ticker
            else:
                target = _select_target(strategy_id, prices_by_ticker, signal_date, policy)
                target_exposure = policy.max_equity_exposure if target else 0.0
                if target is None and variant.fallback_ticker and regime.regime in variant.fallback_regimes:
                    target = variant.fallback_ticker
                    target_exposure = min(variant.fallback_exposure, 1.0)
                _rebalance(
                    account=account,
                    trades=trades,
                    trade_date=trade_date,
                    target=target,
                    target_exposure=target_exposure,
                    prices_by_ticker=prices_by_ticker,
                    asset_types=asset_types,
                    cost_model=cost_model,
                    reason=f"regime_aware_{strategy_id}_{regime.regime}_{variant.name}",
                )

        close_prices = {ticker: float(prices.loc[trade_date, "close"]) for ticker, prices in prices_by_ticker.items()}
        equity_rows.append(
            {
                "date": trade_date,
                "total_value": _market_value(account, close_prices),
                "current_ticker": account.ticker or "cash",
            }
        )

    equity_curve = pd.DataFrame(equity_rows).set_index("date")
    final_value = float(equity_curve["total_value"].iloc[-1])
    return BacktestResult(
        name=name,
        final_value=final_value,
        total_return=final_value / initial_cash - 1,
        max_drawdown=_max_drawdown(equity_curve["total_value"]),
        trades=trades,
        equity_curve=equity_curve,
    )


def default_policy_variants() -> tuple[RegimePolicyVariant, ...]:
    return (
        RegimePolicyVariant(name="defensive", exposure_multiplier=0.65, score_margin_add=0.02, min_score_add=0.01),
        RegimePolicyVariant(name="balanced", exposure_multiplier=1.0),
        RegimePolicyVariant(name="aggressive", exposure_multiplier=1.2, min_exposure_cap=0.2, score_margin_add=-0.01),
        RegimePolicyVariant(
            name="bear_guard",
            exposure_multiplier=1.0,
            range_bound_multiplier=0.5,
            correction_bear_multiplier=0.0,
            systemic_bear_multiplier=0.0,
            score_margin_add=0.02,
            min_score_add=0.01,
        ),
        RegimePolicyVariant(
            name="bull_attack_guarded",
            exposure_multiplier=1.0,
            strong_bull_multiplier=1.0,
            recovery_bull_multiplier=1.0,
            range_bound_multiplier=0.7,
            correction_bear_multiplier=0.2,
            systemic_bear_multiplier=0.0,
            score_margin_add=0.0,
            min_score_add=0.0,
        ),
        RegimePolicyVariant(
            name="bear_guard_daily_stop",
            exposure_multiplier=1.0,
            range_bound_multiplier=0.5,
            correction_bear_multiplier=0.0,
            systemic_bear_multiplier=0.0,
            score_margin_add=0.02,
            min_score_add=0.01,
            daily_risk_exit=True,
            holding_trend_window=60,
        ),
        RegimePolicyVariant(
            name="range_cash_guard",
            exposure_multiplier=1.0,
            strong_bull_multiplier=1.0,
            recovery_bull_multiplier=1.0,
            range_bound_multiplier=0.0,
            correction_bear_multiplier=0.0,
            systemic_bear_multiplier=0.0,
            score_margin_add=0.02,
            min_score_add=0.01,
        ),
        RegimePolicyVariant(
            name="strong_only_guard",
            exposure_multiplier=1.0,
            strong_bull_multiplier=1.0,
            recovery_bull_multiplier=0.5,
            range_bound_multiplier=0.0,
            correction_bear_multiplier=0.0,
            systemic_bear_multiplier=0.0,
            score_margin_add=0.03,
            min_score_add=0.02,
        ),
        RegimePolicyVariant(
            name="strong_only_stop_8",
            exposure_multiplier=1.0,
            strong_bull_multiplier=1.0,
            recovery_bull_multiplier=0.5,
            range_bound_multiplier=0.0,
            correction_bear_multiplier=0.0,
            systemic_bear_multiplier=0.0,
            score_margin_add=0.03,
            min_score_add=0.02,
            portfolio_stop_drawdown_pct=0.08,
            portfolio_stop_cooldown_days=5,
        ),
        RegimePolicyVariant(
            name="strong_only_stop_12",
            exposure_multiplier=1.0,
            strong_bull_multiplier=1.0,
            recovery_bull_multiplier=0.5,
            range_bound_multiplier=0.0,
            correction_bear_multiplier=0.0,
            systemic_bear_multiplier=0.0,
            score_margin_add=0.03,
            min_score_add=0.02,
            portfolio_stop_drawdown_pct=0.12,
            portfolio_stop_cooldown_days=5,
        ),
        RegimePolicyVariant(
            name="bull_attack_stop_10",
            exposure_multiplier=1.0,
            strong_bull_multiplier=1.0,
            recovery_bull_multiplier=1.0,
            range_bound_multiplier=0.3,
            correction_bear_multiplier=0.0,
            systemic_bear_multiplier=0.0,
            score_margin_add=0.02,
            min_score_add=0.01,
            portfolio_stop_drawdown_pct=0.10,
            portfolio_stop_cooldown_days=3,
        ),
        RegimePolicyVariant(
            name="systemic_only_guard",
            exposure_multiplier=1.0,
            strong_bull_multiplier=1.0,
            recovery_bull_multiplier=1.0,
            range_bound_multiplier=1.0,
            correction_bear_multiplier=1.0,
            systemic_bear_multiplier=0.0,
            score_margin_add=0.0,
            min_score_add=0.0,
        ),
        RegimePolicyVariant(
            name="bear_only_guard",
            exposure_multiplier=1.0,
            strong_bull_multiplier=1.0,
            recovery_bull_multiplier=1.0,
            range_bound_multiplier=1.0,
            correction_bear_multiplier=0.0,
            systemic_bear_multiplier=0.0,
            score_margin_add=0.0,
            min_score_add=0.0,
        ),
        RegimePolicyVariant(
            name="bear_only_stop_12_cd3",
            exposure_multiplier=1.0,
            strong_bull_multiplier=1.0,
            recovery_bull_multiplier=1.0,
            range_bound_multiplier=1.0,
            correction_bear_multiplier=0.0,
            systemic_bear_multiplier=0.0,
            score_margin_add=0.0,
            min_score_add=0.0,
            portfolio_stop_drawdown_pct=0.12,
            portfolio_stop_cooldown_days=3,
        ),
        RegimePolicyVariant(
            name="bear_only_stop_8_cd3",
            exposure_multiplier=1.0,
            strong_bull_multiplier=1.0,
            recovery_bull_multiplier=1.0,
            range_bound_multiplier=1.0,
            correction_bear_multiplier=0.0,
            systemic_bear_multiplier=0.0,
            score_margin_add=0.0,
            min_score_add=0.0,
            portfolio_stop_drawdown_pct=0.08,
            portfolio_stop_cooldown_days=3,
        ),
        RegimePolicyVariant(
            name="correction_half_guard",
            exposure_multiplier=1.0,
            strong_bull_multiplier=1.0,
            recovery_bull_multiplier=1.0,
            range_bound_multiplier=1.0,
            correction_bear_multiplier=0.5,
            systemic_bear_multiplier=0.0,
            score_margin_add=0.0,
            min_score_add=0.0,
        ),
        RegimePolicyVariant(
            name="full_attack_bear_guard",
            exposure_multiplier=1.0,
            strong_bull_multiplier=1.0,
            recovery_bull_multiplier=1.5,
            range_bound_multiplier=2.5,
            correction_bear_multiplier=0.0,
            systemic_bear_multiplier=0.0,
            score_margin_add=0.0,
            min_score_add=0.0,
        ),
        RegimePolicyVariant(
            name="full_attack_systemic_guard",
            exposure_multiplier=1.0,
            strong_bull_multiplier=1.0,
            recovery_bull_multiplier=1.5,
            range_bound_multiplier=2.5,
            correction_bear_multiplier=5.0,
            systemic_bear_multiplier=0.0,
            score_margin_add=0.0,
            min_score_add=0.0,
        ),
        RegimePolicyVariant(
            name="full_attack_bear_stop_12_cd3",
            exposure_multiplier=1.0,
            strong_bull_multiplier=1.0,
            recovery_bull_multiplier=1.5,
            range_bound_multiplier=2.5,
            correction_bear_multiplier=0.0,
            systemic_bear_multiplier=0.0,
            score_margin_add=0.0,
            min_score_add=0.0,
            portfolio_stop_drawdown_pct=0.12,
            portfolio_stop_cooldown_days=3,
        ),
        RegimePolicyVariant(
            name="full_attack_bear_stop_8_cd3",
            exposure_multiplier=1.0,
            strong_bull_multiplier=1.0,
            recovery_bull_multiplier=1.5,
            range_bound_multiplier=2.5,
            correction_bear_multiplier=0.0,
            systemic_bear_multiplier=0.0,
            score_margin_add=0.0,
            min_score_add=0.0,
            portfolio_stop_drawdown_pct=0.08,
            portfolio_stop_cooldown_days=3,
        ),
        RegimePolicyVariant(
            name="pure_strong_guard",
            exposure_multiplier=1.0,
            strong_bull_multiplier=1.0,
            recovery_bull_multiplier=0.0,
            range_bound_multiplier=0.0,
            correction_bear_multiplier=0.0,
            systemic_bear_multiplier=0.0,
            score_margin_add=0.04,
            min_score_add=0.03,
        ),
        RegimePolicyVariant(
            name="ultra_strong_guard",
            exposure_multiplier=1.0,
            strong_bull_multiplier=0.8,
            recovery_bull_multiplier=0.0,
            range_bound_multiplier=0.0,
            correction_bear_multiplier=0.0,
            systemic_bear_multiplier=0.0,
            score_margin_add=0.06,
            min_score_add=0.05,
        ),
        RegimePolicyVariant(
            name="ultra_strong_stop_8",
            exposure_multiplier=1.0,
            strong_bull_multiplier=0.8,
            recovery_bull_multiplier=0.0,
            range_bound_multiplier=0.0,
            correction_bear_multiplier=0.0,
            systemic_bear_multiplier=0.0,
            score_margin_add=0.06,
            min_score_add=0.05,
            portfolio_stop_drawdown_pct=0.08,
            portfolio_stop_cooldown_days=5,
        ),
        RegimePolicyVariant(
            name="ultra_strong_stop_8_cd20",
            exposure_multiplier=1.0,
            strong_bull_multiplier=0.8,
            recovery_bull_multiplier=0.0,
            range_bound_multiplier=0.0,
            correction_bear_multiplier=0.0,
            systemic_bear_multiplier=0.0,
            score_margin_add=0.06,
            min_score_add=0.05,
            portfolio_stop_drawdown_pct=0.08,
            portfolio_stop_cooldown_days=20,
        ),
        RegimePolicyVariant(
            name="ultra_strong_stop_6",
            exposure_multiplier=1.0,
            strong_bull_multiplier=0.8,
            recovery_bull_multiplier=0.0,
            range_bound_multiplier=0.0,
            correction_bear_multiplier=0.0,
            systemic_bear_multiplier=0.0,
            score_margin_add=0.06,
            min_score_add=0.05,
            portfolio_stop_drawdown_pct=0.06,
            portfolio_stop_cooldown_days=5,
        ),
        RegimePolicyVariant(
            name="ultra_strong_stop_6_cd20",
            exposure_multiplier=1.0,
            strong_bull_multiplier=0.8,
            recovery_bull_multiplier=0.0,
            range_bound_multiplier=0.0,
            correction_bear_multiplier=0.0,
            systemic_bear_multiplier=0.0,
            score_margin_add=0.06,
            min_score_add=0.05,
            portfolio_stop_drawdown_pct=0.06,
            portfolio_stop_cooldown_days=20,
        ),
        RegimePolicyVariant(
            name="ultra_strong_stop_6_cd10",
            exposure_multiplier=1.0,
            strong_bull_multiplier=0.8,
            recovery_bull_multiplier=0.0,
            range_bound_multiplier=0.0,
            correction_bear_multiplier=0.0,
            systemic_bear_multiplier=0.0,
            score_margin_add=0.06,
            min_score_add=0.05,
            portfolio_stop_drawdown_pct=0.06,
            portfolio_stop_cooldown_days=10,
        ),
        RegimePolicyVariant(
            name="ultra_strong_stop_7_cd10",
            exposure_multiplier=1.0,
            strong_bull_multiplier=0.8,
            recovery_bull_multiplier=0.0,
            range_bound_multiplier=0.0,
            correction_bear_multiplier=0.0,
            systemic_bear_multiplier=0.0,
            score_margin_add=0.06,
            min_score_add=0.05,
            portfolio_stop_drawdown_pct=0.07,
            portfolio_stop_cooldown_days=10,
        ),
        RegimePolicyVariant(
            name="ultra_score20_stop_6_cd10",
            exposure_multiplier=1.0,
            strong_bull_multiplier=0.8,
            recovery_bull_multiplier=0.0,
            range_bound_multiplier=0.0,
            correction_bear_multiplier=0.0,
            systemic_bear_multiplier=0.0,
            score_margin_add=0.06,
            min_score_add=0.20,
            portfolio_stop_drawdown_pct=0.06,
            portfolio_stop_cooldown_days=10,
        ),
        RegimePolicyVariant(
            name="ultra_score25_stop_6_cd10",
            exposure_multiplier=1.0,
            strong_bull_multiplier=0.8,
            recovery_bull_multiplier=0.0,
            range_bound_multiplier=0.0,
            correction_bear_multiplier=0.0,
            systemic_bear_multiplier=0.0,
            score_margin_add=0.06,
            min_score_add=0.25,
            portfolio_stop_drawdown_pct=0.06,
            portfolio_stop_cooldown_days=10,
        ),
        RegimePolicyVariant(
            name="ultra_score30_stop_6_cd10",
            exposure_multiplier=1.0,
            strong_bull_multiplier=0.8,
            recovery_bull_multiplier=0.0,
            range_bound_multiplier=0.0,
            correction_bear_multiplier=0.0,
            systemic_bear_multiplier=0.0,
            score_margin_add=0.06,
            min_score_add=0.30,
            portfolio_stop_drawdown_pct=0.06,
            portfolio_stop_cooldown_days=10,
        ),
        RegimePolicyVariant(
            name="ultra_strong_stop_12",
            exposure_multiplier=1.0,
            strong_bull_multiplier=0.8,
            recovery_bull_multiplier=0.0,
            range_bound_multiplier=0.0,
            correction_bear_multiplier=0.0,
            systemic_bear_multiplier=0.0,
            score_margin_add=0.06,
            min_score_add=0.05,
            portfolio_stop_drawdown_pct=0.12,
            portfolio_stop_cooldown_days=5,
        ),
        RegimePolicyVariant(
            name="ultra_strong_full_stop_12",
            exposure_multiplier=1.0,
            strong_bull_multiplier=1.0,
            recovery_bull_multiplier=0.0,
            range_bound_multiplier=0.0,
            correction_bear_multiplier=0.0,
            systemic_bear_multiplier=0.0,
            score_margin_add=0.06,
            min_score_add=0.05,
            portfolio_stop_drawdown_pct=0.12,
            portfolio_stop_cooldown_days=5,
        ),
        RegimePolicyVariant(
            name="ultra_strong_full_stop_8",
            exposure_multiplier=1.0,
            strong_bull_multiplier=1.0,
            recovery_bull_multiplier=0.0,
            range_bound_multiplier=0.0,
            correction_bear_multiplier=0.0,
            systemic_bear_multiplier=0.0,
            score_margin_add=0.06,
            min_score_add=0.05,
            portfolio_stop_drawdown_pct=0.08,
            portfolio_stop_cooldown_days=5,
        ),
        RegimePolicyVariant(
            name="anchor_0050_guard",
            exposure_multiplier=1.0,
            strong_bull_multiplier=1.0,
            recovery_bull_multiplier=0.7,
            range_bound_multiplier=0.0,
            correction_bear_multiplier=0.0,
            systemic_bear_multiplier=0.0,
            score_margin_add=0.04,
            min_score_add=0.03,
            fallback_ticker="0050.TW",
            fallback_regimes=("recovery_bull", "range_bound"),
            fallback_exposure=1.0,
        ),
        RegimePolicyVariant(
            name="anchor_0050_stop_12",
            exposure_multiplier=1.0,
            strong_bull_multiplier=1.0,
            recovery_bull_multiplier=0.7,
            range_bound_multiplier=0.0,
            correction_bear_multiplier=0.0,
            systemic_bear_multiplier=0.0,
            score_margin_add=0.04,
            min_score_add=0.03,
            portfolio_stop_drawdown_pct=0.12,
            portfolio_stop_cooldown_days=5,
            fallback_ticker="0050.TW",
            fallback_regimes=("recovery_bull", "range_bound"),
            fallback_exposure=1.0,
        ),
    )


def _select_target(
    strategy_id: str,
    prices_by_ticker: dict[str, pd.DataFrame],
    signal_date: pd.Timestamp,
    policy: StrategyPolicy,
) -> str | None:
    if not policy.allow_new_entry:
        return None
    if strategy_id == "daily_strength":
        scores = relative_strength_scores(prices_by_ticker, signal_date)
    elif strategy_id == "weekly_rotation":
        scores = dual_momentum_scores(prices_by_ticker, signal_date)
    else:
        raise ValueError(f"Unsupported strategy_id: {strategy_id}")
    if not scores:
        return None
    ordered = sorted(scores.items(), key=lambda item: (item[1], item[0]), reverse=True)
    top_ticker, top_score = ordered[0]
    if policy.min_candidate_score is not None and top_score < policy.min_candidate_score:
        return None
    if policy.switch_score_margin is not None and len(ordered) > 1:
        if top_score - ordered[1][1] < policy.switch_score_margin:
            return None
    return top_ticker


def _variant_policy(policy: StrategyPolicy, variant: RegimePolicyVariant) -> StrategyPolicy:
    exposure = max(policy.max_equity_exposure * _regime_multiplier(policy.regime, variant), variant.min_exposure_cap)
    exposure = min(exposure, 1.0)
    margin = policy.switch_score_margin
    if margin is not None:
        margin = max(margin + variant.score_margin_add, 0.0)
    min_score = policy.min_candidate_score
    if min_score is not None:
        min_score = max(min_score + variant.min_score_add, 0.0)
    return StrategyPolicy(
        strategy_id=policy.strategy_id,
        regime=policy.regime,
        allow_new_entry=policy.allow_new_entry,
        allow_rebalance=policy.allow_rebalance,
        max_equity_exposure=round(exposure, 4),
        min_cash_ratio=round(1.0 - exposure, 4),
        rebalance_frequency=policy.rebalance_frequency,
        switch_score_margin=margin,
        min_candidate_score=min_score,
        product_mode=f"{policy.product_mode}/{variant.name}",
        risk_message=policy.risk_message,
    )


def _regime_multiplier(regime: str, variant: RegimePolicyVariant) -> float:
    overrides = {
        "strong_bull": variant.strong_bull_multiplier,
        "recovery_bull": variant.recovery_bull_multiplier,
        "range_bound": variant.range_bound_multiplier,
        "correction_bear": variant.correction_bear_multiplier,
        "systemic_bear": variant.systemic_bear_multiplier,
    }
    value = overrides.get(regime)
    return variant.exposure_multiplier if value is None else value


def _should_daily_exit(
    ticker: str,
    prices_by_ticker: dict[str, pd.DataFrame],
    signal_date: pd.Timestamp,
    regime: str,
    variant: RegimePolicyVariant,
) -> bool:
    if regime == "systemic_bear":
        return True
    if regime == "correction_bear":
        return True
    if variant.holding_trend_window is None:
        return False
    prices = prices_by_ticker[ticker]
    history = prices.loc[prices.index <= signal_date, "adj_close"].dropna()
    if len(history) < variant.holding_trend_window:
        return False
    close = float(history.iloc[-1])
    trend_average = float(history.iloc[-variant.holding_trend_window :].mean())
    return close < trend_average


def _rebalance(
    *,
    account: _Account,
    trades: list[Trade],
    trade_date: pd.Timestamp,
    target: str | None,
    target_exposure: float,
    prices_by_ticker: dict[str, pd.DataFrame],
    asset_types: dict[str, str],
    cost_model: TaiwanCostModel,
    reason: str,
) -> None:
    date = _date_str(trade_date)
    open_prices = {ticker: float(prices.loc[trade_date, "open"]) for ticker, prices in prices_by_ticker.items()}
    if target is None or target_exposure <= 0:
        _sell_shares(account, trades, date, account.shares, open_prices, asset_types, cost_model, reason)
        return
    if account.ticker and account.ticker != target:
        _sell_shares(account, trades, date, account.shares, open_prices, asset_types, cost_model, reason)

    total_value = _market_value(account, open_prices)
    target_value = total_value * target_exposure
    target_price = open_prices[target]
    current_shares = account.shares if account.ticker == target else 0
    desired_shares = int(target_value // target_price)
    diff = desired_shares - current_shares
    if account.ticker == target and abs(diff * target_price) < total_value * 0.05:
        return
    if diff > 0:
        _buy_shares(account, trades, date, target, diff, target_price, cost_model, reason)
    elif diff < 0:
        _sell_shares(account, trades, date, abs(diff), open_prices, asset_types, cost_model, reason)


def _buy_shares(
    account: _Account,
    trades: list[Trade],
    date: str,
    ticker: str,
    shares: int,
    price: float,
    cost_model: TaiwanCostModel,
    reason: str,
) -> None:
    while shares > 0:
        gross = shares * price
        costs = cost_model.buy_cost(gross)
        if gross + costs <= account.cash:
            break
        shares -= 1
    if shares <= 0:
        return
    gross = shares * price
    costs = cost_model.buy_cost(gross)
    account.cash -= gross + costs
    account.ticker = ticker
    account.shares += shares
    trades.append(Trade(date, ticker, "buy", shares, price, gross, costs, account.cash, reason))


def _sell_shares(
    account: _Account,
    trades: list[Trade],
    date: str,
    shares: int,
    open_prices: dict[str, float],
    asset_types: dict[str, str],
    cost_model: TaiwanCostModel,
    reason: str,
) -> None:
    if account.ticker is None or account.shares <= 0 or shares <= 0:
        return
    shares = min(shares, account.shares)
    ticker = account.ticker
    price = open_prices[ticker]
    gross = shares * price
    costs = cost_model.sell_cost(gross, asset_types[ticker])
    account.cash += gross - costs
    account.shares -= shares
    if account.shares == 0:
        account.ticker = None
    trades.append(Trade(date, ticker, "sell", shares, price, gross, costs, account.cash, reason))


def _market_value(account: _Account, prices: dict[str, float]) -> float:
    value = account.cash
    if account.ticker and account.shares > 0:
        value += account.shares * prices[account.ticker]
    return value
