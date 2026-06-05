from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from backtest_lab.bear_defense_backtest import risk_on_for_rule
from backtest_lab.costs import TaiwanCostModel
from backtest_lab.market_regime import classify_market_regime
from backtest_lab.portfolio import Trade
from backtest_lab.regime_mode_switch import (
    _attack_gate_passes,
    _had_prior_attack_gate_activation,
    _market_risk_off,
    _signal_close,
    _update_attack_gate_state,
    _update_risk_off_state,
    cycle_proven_preproof_exposure_variants,
)
from backtest_lab.simulation import BacktestResult, _common_trade_dates, _date_str, _max_drawdown
from backtest_lab.strategies import previous_available_date, relative_strength_scores


@dataclass(frozen=True)
class DiversificationVariant:
    name: str
    top_n: int
    weighting: str


@dataclass
class _MultiAccount:
    cash: float
    positions: dict[str, int] = field(default_factory=dict)


def diversification_variants() -> tuple[DiversificationVariant, ...]:
    return (
        DiversificationVariant("top1_frozen_baseline", 1, "rank"),
        DiversificationVariant("top1_90_top2_10", 2, "rank_90_10"),
        DiversificationVariant("top1_80_top2_20", 2, "rank_80_20"),
        DiversificationVariant("top2_equal", 2, "equal"),
        DiversificationVariant("top2_inverse_vol", 2, "inverse_vol"),
        DiversificationVariant("top3_inverse_vol", 3, "inverse_vol"),
    )


def simulate_cycle_proven_diversification(
    *,
    name: str,
    prices_by_ticker: dict[str, pd.DataFrame],
    asset_types: dict[str, str],
    market_prices: pd.DataFrame,
    start_date: str,
    end_date: str,
    initial_cash: float,
    cost_model: TaiwanCostModel,
    diversification: DiversificationVariant,
    dividend_series_by_ticker: dict[str, pd.Series] | None = None,
) -> BacktestResult:
    trade_dates = _common_trade_dates(prices_by_ticker, start_date, end_date)
    if not trade_dates:
        raise ValueError(f"No common trade dates between {start_date} and {end_date}")
    strategy = cycle_proven_preproof_exposure_variants()[1]
    strategy = type(strategy)(
        **{
            **strategy.__dict__,
            "attack_selection_exclude_tickers": ("0050.TW", "00631L.TW"),
        }
    )
    account = _MultiAccount(float(initial_cash))
    trades: list[Trade] = []
    equity_rows: list[dict] = []
    peak_signal_value = float(initial_cash)
    risk_off_active = False
    risk_off_clear_streak = 0
    prior_activation = _had_prior_attack_gate_activation(
        prices_by_ticker=prices_by_ticker,
        first_trade_date=trade_dates[0],
        variant=strategy,
    )
    attack_active = prior_activation
    attack_ever_activated = prior_activation
    attack_streak = 0

    for trade_date in trade_dates:
        _credit_dividends(account, trades, trade_date, dividend_series_by_ticker)
        signal_date = previous_available_date({**prices_by_ticker, "__market__": market_prices}, trade_date)
        signal_prices = {ticker: _signal_close(prices, signal_date) for ticker, prices in prices_by_ticker.items()}
        signal_value = _market_value(account, signal_prices)
        peak_signal_value = max(peak_signal_value, signal_value)
        stop_triggered = bool(
            account.positions
            and peak_signal_value > 0
            and signal_value / peak_signal_value - 1 <= -0.12
        )
        if stop_triggered:
            _rebalance_weights(
                account,
                trades,
                trade_date,
                {},
                prices_by_ticker,
                asset_types,
                cost_model,
                f"cycle_diversification_portfolio_stop_{diversification.name}",
            )
            peak_signal_value = account.cash
            attack_active = False
            attack_streak = 0

        regime = classify_market_regime(market_prices, signal_date, universe_prices=prices_by_ticker).regime
        if regime == "systemic_bear":
            attack_active = False
            attack_streak = 0
            attack_ever_activated = False
        elif not attack_active and not stop_triggered:
            raw_attack = _attack_gate_passes(
                prices_by_ticker,
                signal_date,
                strategy,
                use_reentry_rules=attack_ever_activated,
            )
            attack_active, attack_streak = _update_attack_gate_state(
                active=attack_active,
                activation_streak=attack_streak,
                raw_activation=raw_attack,
                activation_confirmation_days=strategy.attack_gate_activation_confirmation_days,
            )
            attack_ever_activated = attack_ever_activated or attack_active

        if attack_ever_activated:
            risk_off_active = False
            risk_off_clear_streak = 0
        else:
            risk_off_active, risk_off_clear_streak = _update_risk_off_state(
                active=risk_off_active,
                clear_streak=risk_off_clear_streak,
                raw_risk_off=_market_risk_off(market_prices, signal_date, "risk_2of3"),
                exit_confirmation_days=5,
            )

        if stop_triggered:
            target_weights = {}
            mode = "portfolio_stop"
        elif regime == "systemic_bear":
            target_weights: dict[str, float] = {}
            mode = "cash"
        elif attack_active:
            target_weights = _attack_weights(prices_by_ticker, signal_date, diversification)
            mode = diversification.name
        elif risk_off_active:
            target_weights = (
                {"00631L.TW": 0.25}
                if risk_on_for_rule(prices_by_ticker["00631L.TW"], signal_date, "ma200")
                else {}
            )
            mode = "preproof_risk_off"
        else:
            target_weights = (
                {"00631L.TW": 1.0}
                if risk_on_for_rule(prices_by_ticker["00631L.TW"], signal_date, "ma200")
                else {}
            )
            mode = "preproof_waiting"

        _rebalance_weights(
            account,
            trades,
            trade_date,
            target_weights,
            prices_by_ticker,
            asset_types,
            cost_model,
            f"cycle_diversification_{regime}_{mode}",
        )
        close_prices = {ticker: float(prices.loc[trade_date, "close"]) for ticker, prices in prices_by_ticker.items()}
        equity_rows.append(
            {
                "date": trade_date,
                "total_value": _market_value(account, close_prices),
                "current_ticker": ",".join(sorted(account.positions)) if account.positions else "cash",
                "regime": regime,
                "mode": mode,
                "attack_gate_active": attack_active,
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


def _attack_weights(
    prices_by_ticker: dict[str, pd.DataFrame],
    signal_date: pd.Timestamp,
    diversification: DiversificationVariant,
) -> dict[str, float]:
    scores = relative_strength_scores(prices_by_ticker, signal_date)
    ranked = sorted(
        (
            (ticker, score)
            for ticker, score in scores.items()
            if ticker not in {"0050.TW", "00631L.TW"}
        ),
        key=lambda item: (item[1], item[0]),
        reverse=True,
    )[: diversification.top_n]
    if not ranked:
        return {}
    tickers = [ticker for ticker, _ in ranked]
    if diversification.weighting == "rank":
        return {tickers[0]: 1.0}
    if diversification.weighting == "rank_90_10":
        return {tickers[0]: 0.90, tickers[1]: 0.10}
    if diversification.weighting == "rank_80_20":
        return {tickers[0]: 0.80, tickers[1]: 0.20}
    if diversification.weighting == "equal":
        return {ticker: 1.0 / len(tickers) for ticker in tickers}
    if diversification.weighting == "inverse_vol":
        inverse_vols: dict[str, float] = {}
        for ticker in tickers:
            history = prices_by_ticker[ticker].loc[
                prices_by_ticker[ticker].index <= signal_date,
                "adj_close",
            ].dropna()
            volatility = float(history.pct_change().dropna().iloc[-20:].std())
            inverse_vols[ticker] = 1.0 / max(volatility, 0.0001)
        total = sum(inverse_vols.values())
        return {ticker: value / total for ticker, value in inverse_vols.items()}
    raise ValueError(f"Unsupported diversification weighting: {diversification.weighting}")


def _rebalance_weights(
    account: _MultiAccount,
    trades: list[Trade],
    trade_date: pd.Timestamp,
    target_weights: dict[str, float],
    prices_by_ticker: dict[str, pd.DataFrame],
    asset_types: dict[str, str],
    cost_model: TaiwanCostModel,
    reason: str,
) -> None:
    target_weights = {ticker: weight for ticker, weight in target_weights.items() if weight > 0}
    date = _date_str(trade_date)
    open_prices = {ticker: float(prices.loc[trade_date, "open"]) for ticker, prices in prices_by_ticker.items()}
    total_value = _market_value(account, open_prices)
    desired_shares = {
        ticker: int(total_value * weight // open_prices[ticker])
        for ticker, weight in target_weights.items()
    }
    for ticker, shares in list(account.positions.items()):
        shares_to_sell = shares - desired_shares.get(ticker, 0)
        if shares_to_sell <= 0:
            continue
        if ticker in target_weights and shares_to_sell * open_prices[ticker] < total_value * 0.05:
            continue
        gross = shares_to_sell * open_prices[ticker]
        costs = cost_model.sell_cost(gross, asset_types[ticker])
        account.cash += gross - costs
        account.positions[ticker] -= shares_to_sell
        if account.positions[ticker] == 0:
            del account.positions[ticker]
        trades.append(
            Trade(date, ticker, "sell", shares_to_sell, open_prices[ticker], gross, costs, account.cash, reason)
        )
    for ticker, weight in sorted(target_weights.items(), key=lambda item: item[1], reverse=True):
        price = open_prices[ticker]
        shares_to_buy = desired_shares[ticker] - account.positions.get(ticker, 0)
        if shares_to_buy <= 0:
            continue
        if ticker in account.positions and shares_to_buy * price < total_value * 0.05:
            continue
        while shares_to_buy > 0:
            gross = shares_to_buy * price
            costs = cost_model.buy_cost(gross)
            if gross + costs <= account.cash:
                break
            shares_to_buy -= 1
        if shares_to_buy <= 0:
            continue
        gross = shares_to_buy * price
        costs = cost_model.buy_cost(gross)
        account.cash -= gross + costs
        account.positions[ticker] = account.positions.get(ticker, 0) + shares_to_buy
        trades.append(Trade(date, ticker, "buy", shares_to_buy, price, gross, costs, account.cash, reason))


def _credit_dividends(
    account: _MultiAccount,
    trades: list[Trade],
    trade_date: pd.Timestamp,
    dividend_series_by_ticker: dict[str, pd.Series] | None,
) -> None:
    if dividend_series_by_ticker is None:
        return
    for ticker, shares in account.positions.items():
        dividend = float(dividend_series_by_ticker[ticker].get(trade_date, 0.0))
        if dividend <= 0:
            continue
        amount = shares * dividend
        account.cash += amount
        trades.append(
            Trade(_date_str(trade_date), ticker, "dividend", shares, dividend, amount, 0, account.cash, "cash_dividend")
        )


def _market_value(account: _MultiAccount, prices: dict[str, float]) -> float:
    return account.cash + sum(shares * prices[ticker] for ticker, shares in account.positions.items())
