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

