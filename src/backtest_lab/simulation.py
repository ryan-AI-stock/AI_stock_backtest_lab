from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from backtest_lab.costs import TaiwanCostModel
from backtest_lab.portfolio import Portfolio, Trade
from backtest_lab.strategies import previous_available_date, relative_strength_top1


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
) -> BacktestResult:
    trade_dates = _trade_dates(prices, start_date, end_date)
    if not trade_dates:
        raise ValueError(f"No trade dates for {ticker} between {start_date} and {end_date}")
    portfolio = Portfolio(initial_cash, cost_model)
    entry_date = trade_dates[0]
    entry_price = float(prices.loc[entry_date, "open"])
    portfolio.buy_max(_date_str(entry_date), ticker, asset_type, entry_price, "benchmark_initial_entry")
    equity_curve = _single_asset_equity_curve(portfolio, ticker, prices, trade_dates)
    return _result(name, initial_cash, portfolio.trades, equity_curve)


def simulate_relative_strength_top1(
    name: str,
    prices_by_ticker: dict[str, pd.DataFrame],
    asset_types: dict[str, str],
    start_date: str,
    end_date: str,
    initial_cash: float,
    cost_model: TaiwanCostModel,
) -> BacktestResult:
    trade_dates = _common_trade_dates(prices_by_ticker, start_date, end_date)
    if not trade_dates:
        raise ValueError(f"No common trade dates between {start_date} and {end_date}")
    portfolio = Portfolio(initial_cash, cost_model)
    equity_rows = []

    for trade_date in trade_dates:
        signal_date = previous_available_date(prices_by_ticker, trade_date)
        target = relative_strength_top1(prices_by_ticker, signal_date)
        current = portfolio.current_ticker()
        if current == target:
            continue
        if current is not None:
            sell_price = float(prices_by_ticker[current].loc[trade_date, "open"])
            portfolio.sell_all(
                _date_str(trade_date),
                current,
                asset_types[current],
                sell_price,
                "relative_strength_rebalance",
            )
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
        equity_rows.append({"date": trade_date, "total_value": portfolio.market_value(close_prices)})

    equity_curve = pd.DataFrame(equity_rows).set_index("date")
    return _result(name, initial_cash, portfolio.trades, equity_curve)


def _trade_dates(prices: pd.DataFrame, start_date: str, end_date: str) -> list[pd.Timestamp]:
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    return list(prices.loc[(prices.index >= start) & (prices.index <= end)].index)


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


def _single_asset_equity_curve(
    portfolio: Portfolio,
    ticker: str,
    prices: pd.DataFrame,
    trade_dates: list[pd.Timestamp],
) -> pd.DataFrame:
    rows = []
    for date in trade_dates:
        close_prices = {ticker: float(prices.loc[date, "close"])}
        rows.append({"date": date, "total_value": portfolio.market_value(close_prices)})
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
