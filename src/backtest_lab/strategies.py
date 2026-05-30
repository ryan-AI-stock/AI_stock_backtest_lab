from __future__ import annotations

import pandas as pd


def previous_available_date(prices_by_ticker: dict[str, pd.DataFrame], trade_date: pd.Timestamp) -> pd.Timestamp:
    candidates: list[pd.Timestamp] = []
    for prices in prices_by_ticker.values():
        available = prices.index[prices.index < trade_date]
        if len(available):
            candidates.append(available.max())
    if not candidates:
        raise ValueError(f"No signal date available before {trade_date.date()}")
    return min(candidates)


def relative_strength_top1(
    prices_by_ticker: dict[str, pd.DataFrame],
    signal_date: pd.Timestamp,
    windows: tuple[int, int] = (20, 60),
) -> str:
    scores: dict[str, float] = {}
    for ticker, prices in prices_by_ticker.items():
        history = prices.loc[prices.index <= signal_date, "adj_close"].dropna()
        if len(history) <= max(windows):
            continue
        short_return = history.iloc[-1] / history.iloc[-windows[0]] - 1
        long_return = history.iloc[-1] / history.iloc[-windows[1]] - 1
        scores[ticker] = (0.4 * short_return) + (0.6 * long_return)
    if not scores:
        raise ValueError(f"No ticker had enough warmup data for signal date {signal_date.date()}")
    return max(scores.items(), key=lambda item: (item[1], item[0]))[0]


def dual_momentum_vol_control(
    prices_by_ticker: dict[str, pd.DataFrame],
    signal_date: pd.Timestamp,
    momentum_windows: tuple[int, int] = (63, 126),
    trend_window: int = 126,
    volatility_window: int = 20,
) -> str | None:
    scores: dict[str, float] = {}
    required = max(max(momentum_windows), trend_window, volatility_window) + 1
    for ticker, prices in prices_by_ticker.items():
        history = prices.loc[prices.index <= signal_date, "adj_close"].dropna()
        if len(history) < required:
            continue
        current = float(history.iloc[-1])
        trend_average = float(history.iloc[-trend_window:].mean())
        medium_return = current / float(history.iloc[-momentum_windows[0]]) - 1
        long_return = current / float(history.iloc[-momentum_windows[1]]) - 1
        if current <= trend_average or medium_return <= 0 or long_return <= 0:
            continue
        daily_vol = history.pct_change().dropna().iloc[-volatility_window:].std()
        annual_vol = float(daily_vol * (252**0.5))
        scores[ticker] = (0.45 * medium_return) + (0.45 * long_return) - (0.10 * annual_vol)
    if not scores:
        return None
    return max(scores.items(), key=lambda item: (item[1], item[0]))[0]
