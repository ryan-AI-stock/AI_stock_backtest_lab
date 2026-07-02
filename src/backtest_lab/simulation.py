from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from backtest_lab.costs import TaiwanCostModel
from backtest_lab.portfolio import Portfolio, Trade
from backtest_lab.strategies import (
    dual_momentum_vol_control,
    previous_available_date,
    relative_strength_top1,
    theme_enhanced_dual_momentum,
)


@dataclass
class BacktestResult:
    name: str
    final_value: float
    total_return: float
    max_drawdown: float
    trades: list[Trade]
    equity_curve: pd.DataFrame


def simulate_buy_and_hold(
    name: str,
    ticker: str,
    asset_type: str,
    prices: pd.DataFrame,
    start_date: str,
    end_date: str,
    initial_cash: float,
    cost_model: TaiwanCostModel,
    dividend_series: pd.Series | None = None,
) -> BacktestResult:
    trade_dates = _trade_dates(prices, start_date, end_date)
    if not trade_dates:
        raise ValueError(f"No trade dates for {ticker} between {start_date} and {end_date}")
    portfolio = Portfolio(initial_cash, cost_model)
    entry_date = trade_dates[0]
    entry_price = float(prices.loc[entry_date, "open"])
    portfolio.buy_max(_date_str(entry_date), ticker, asset_type, entry_price, "benchmark_initial_entry")
    equity_curve = _single_asset_equity_curve(portfolio, ticker, prices, trade_dates, entry_date, dividend_series)
    return _result(name, initial_cash, portfolio.trades, equity_curve)


def simulate_relative_strength_top1(
    name: str,
    prices_by_ticker: dict[str, pd.DataFrame],
    asset_types: dict[str, str],
    start_date: str,
    end_date: str,
    initial_cash: float,
    cost_model: TaiwanCostModel,
    dividend_series_by_ticker: dict[str, pd.Series] | None = None,
) -> BacktestResult:
    trade_dates = _common_trade_dates(prices_by_ticker, start_date, end_date)
    if not trade_dates:
        raise ValueError(f"No common trade dates between {start_date} and {end_date}")
    portfolio = Portfolio(initial_cash, cost_model)
    equity_rows = []

    for trade_date in trade_dates:
        current_before_trade = portfolio.current_ticker()
        if current_before_trade is not None and dividend_series_by_ticker is not None:
            dividend = float(dividend_series_by_ticker[current_before_trade].get(trade_date, 0.0))
            portfolio.credit_dividend(_date_str(trade_date), current_before_trade, dividend)

        signal_date = previous_available_date(prices_by_ticker, trade_date)
        target = relative_strength_top1(prices_by_ticker, signal_date)
        current = portfolio.current_ticker()
        if current != target and current is not None:
            sell_price = float(prices_by_ticker[current].loc[trade_date, "open"])
            portfolio.sell_all(
                _date_str(trade_date),
                current,
                asset_types[current],
                sell_price,
                "relative_strength_rebalance",
            )
        if current != target:
            buy_price = float(prices_by_ticker[target].loc[trade_date, "open"])
            portfolio.buy_max(
                _date_str(trade_date),
                target,
                asset_types[target],
                buy_price,
                "relative_strength_initial_entry" if current is None else "relative_strength_rebalance",
            )
        close_prices = {
            ticker: float(prices.loc[trade_date, "close"])
            for ticker, prices in prices_by_ticker.items()
        }
        equity_rows.append(
            {
                "date": trade_date,
                "total_value": portfolio.market_value(close_prices),
                "current_ticker": portfolio.current_ticker() or "cash",
            }
        )

    equity_curve = pd.DataFrame(equity_rows).set_index("date")
    return _result(name, initial_cash, portfolio.trades, equity_curve)


def simulate_dual_momentum_vol_control(
    name: str,
    prices_by_ticker: dict[str, pd.DataFrame],
    asset_types: dict[str, str],
    start_date: str,
    end_date: str,
    initial_cash: float,
    cost_model: TaiwanCostModel,
    dividend_series_by_ticker: dict[str, pd.Series] | None = None,
    momentum_windows: tuple[int, int] = (63, 126),
    trend_window: int = 126,
    volatility_window: int = 20,
    rebalance_frequency: str = "weekly",
    signal_weekday: int | None = None,
) -> BacktestResult:
    trade_dates = _common_trade_dates(prices_by_ticker, start_date, end_date)
    if not trade_dates:
        raise ValueError(f"No common trade dates between {start_date} and {end_date}")
    portfolio = Portfolio(initial_cash, cost_model)
    equity_rows = []
    last_signal_week_key: tuple[int, int] | None = None

    for index, trade_date in enumerate(trade_dates):
        current_before_trade = portfolio.current_ticker()
        if current_before_trade is not None and dividend_series_by_ticker is not None:
            dividend = float(dividend_series_by_ticker[current_before_trade].get(trade_date, 0.0))
            portfolio.credit_dividend(_date_str(trade_date), current_before_trade, dividend)

        signal_date = previous_available_date(prices_by_ticker, trade_date)
        should_rebalance = index == 0 or _is_rebalance_date(trade_dates, index, rebalance_frequency)
        if signal_weekday is not None:
            signal_week_key = (signal_date.isocalendar().year, signal_date.isocalendar().week)
            should_rebalance = signal_date.weekday() == signal_weekday and signal_week_key != last_signal_week_key
            if should_rebalance:
                last_signal_week_key = signal_week_key

        if should_rebalance:
            target = dual_momentum_vol_control(
                prices_by_ticker,
                signal_date,
                momentum_windows=momentum_windows,
                trend_window=trend_window,
                volatility_window=volatility_window,
            )
            current = portfolio.current_ticker()
            if target != current:
                if current is not None:
                    sell_price = float(prices_by_ticker[current].loc[trade_date, "open"])
                    portfolio.sell_all(
                        _date_str(trade_date),
                        current,
                        asset_types[current],
                        sell_price,
                        "dual_momentum_rebalance",
                    )
                if target is not None:
                    buy_price = float(prices_by_ticker[target].loc[trade_date, "open"])
                    portfolio.buy_max(
                        _date_str(trade_date),
                        target,
                        asset_types[target],
                        buy_price,
                        "dual_momentum_initial_entry" if current is None else "dual_momentum_rebalance",
                    )

        close_prices = {
            ticker: float(prices.loc[trade_date, "close"])
            for ticker, prices in prices_by_ticker.items()
        }
        equity_rows.append(
            {
                "date": trade_date,
                "total_value": portfolio.market_value(close_prices),
                "current_ticker": portfolio.current_ticker() or "cash",
            }
        )

    equity_curve = pd.DataFrame(equity_rows).set_index("date")
    return _result(name, initial_cash, portfolio.trades, equity_curve)


def simulate_theme_enhanced_dual_momentum(
    name: str,
    prices_by_ticker: dict[str, pd.DataFrame],
    asset_types: dict[str, str],
    theme_by_ticker: dict[str, tuple[str, ...]],
    start_date: str,
    end_date: str,
    initial_cash: float,
    cost_model: TaiwanCostModel,
    dividend_series_by_ticker: dict[str, pd.Series] | None = None,
    rebalance_frequency: str = "weekly",
) -> BacktestResult:
    trade_dates = _common_trade_dates(prices_by_ticker, start_date, end_date)
    if not trade_dates:
        raise ValueError(f"No common trade dates between {start_date} and {end_date}")
    portfolio = Portfolio(initial_cash, cost_model)
    equity_rows = []

    for index, trade_date in enumerate(trade_dates):
        current_before_trade = portfolio.current_ticker()
        if current_before_trade is not None and dividend_series_by_ticker is not None:
            dividend = float(dividend_series_by_ticker[current_before_trade].get(trade_date, 0.0))
            portfolio.credit_dividend(_date_str(trade_date), current_before_trade, dividend)

        if index == 0 or _is_rebalance_date(trade_dates, index, rebalance_frequency):
            signal_date = previous_available_date(prices_by_ticker, trade_date)
            target = theme_enhanced_dual_momentum(prices_by_ticker, signal_date, theme_by_ticker)
            current = portfolio.current_ticker()
            if target != current:
                if current is not None:
                    sell_price = float(prices_by_ticker[current].loc[trade_date, "open"])
                    portfolio.sell_all(
                        _date_str(trade_date),
                        current,
                        asset_types[current],
                        sell_price,
                        "theme_enhanced_rebalance",
                    )
                if target is not None:
                    buy_price = float(prices_by_ticker[target].loc[trade_date, "open"])
                    portfolio.buy_max(
                        _date_str(trade_date),
                        target,
                        asset_types[target],
                        buy_price,
                        "theme_enhanced_initial_entry" if current is None else "theme_enhanced_rebalance",
                    )

        close_prices = {
            ticker: float(prices.loc[trade_date, "close"])
            for ticker, prices in prices_by_ticker.items()
        }
        equity_rows.append(
            {
                "date": trade_date,
                "total_value": portfolio.market_value(close_prices),
                "current_ticker": portfolio.current_ticker() or "cash",
            }
        )

    equity_curve = pd.DataFrame(equity_rows).set_index("date")
    return _result(name, initial_cash, portfolio.trades, equity_curve)


def _trade_dates(prices: pd.DataFrame, start_date: str, end_date: str) -> list[pd.Timestamp]:
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    frame = prices.loc[(prices.index >= start) & (prices.index <= end)].copy()
    for column in ("open", "close", "adj_close"):
        if column not in frame.columns:
            continue
        frame = frame[pd.to_numeric(frame[column], errors="coerce") > 0]
    return list(frame.index)


def _common_trade_dates(
    prices_by_ticker: dict[str, pd.DataFrame],
    start_date: str,
    end_date: str,
) -> list[pd.Timestamp]:
    common: set[pd.Timestamp] | None = None
    for prices in prices_by_ticker.values():
        dates = set(_trade_dates(prices, start_date, end_date))
        common = dates if common is None else common & dates
    return sorted(common or set())


def _is_rebalance_date(trade_dates: list[pd.Timestamp], index: int, frequency: str) -> bool:
    if index == 0:
        return True
    if frequency == "daily":
        return True
    if frequency == "weekly":
        return _is_first_trading_day_of_week(trade_dates, index)
    if frequency == "monthly":
        return trade_dates[index].month != trade_dates[index - 1].month
    raise ValueError(f"Unsupported rebalance frequency: {frequency}")


def _is_first_trading_day_of_week(trade_dates: list[pd.Timestamp], index: int) -> bool:
    if index == 0:
        return True
    current = trade_dates[index].isocalendar()
    previous = trade_dates[index - 1].isocalendar()
    return (current.year, current.week) != (previous.year, previous.week)


def _single_asset_equity_curve(
    portfolio: Portfolio,
    ticker: str,
    prices: pd.DataFrame,
    trade_dates: list[pd.Timestamp],
    entry_date: pd.Timestamp,
    dividend_series: pd.Series | None,
) -> pd.DataFrame:
    rows = []
    for date in trade_dates:
        if date > entry_date and dividend_series is not None:
            dividend = float(dividend_series.get(date, 0.0))
            portfolio.credit_dividend(_date_str(date), ticker, dividend)
        close_prices = {ticker: float(prices.loc[date, "close"])}
        rows.append({"date": date, "total_value": portfolio.market_value(close_prices), "current_ticker": ticker})
    return pd.DataFrame(rows).set_index("date")


def _result(
    name: str,
    initial_cash: float,
    trades: list[Trade],
    equity_curve: pd.DataFrame,
) -> BacktestResult:
    final_value = float(equity_curve["total_value"].iloc[-1])
    total_return = final_value / initial_cash - 1
    max_drawdown = _max_drawdown(equity_curve["total_value"])
    return BacktestResult(
        name=name,
        final_value=final_value,
        total_return=total_return,
        max_drawdown=max_drawdown,
        trades=trades,
        equity_curve=equity_curve,
    )


def _max_drawdown(values: pd.Series) -> float:
    peaks = values.cummax()
    drawdowns = values / peaks - 1
    return float(drawdowns.min())


def _date_str(date: pd.Timestamp) -> str:
    return date.strftime("%Y-%m-%d")
