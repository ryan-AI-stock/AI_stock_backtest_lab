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
    scores = relative_strength_scores(prices_by_ticker, signal_date, windows)
    if not scores:
        raise ValueError(f"No ticker had enough warmup data for signal date {signal_date.date()}")
    return max(scores.items(), key=lambda item: (item[1], item[0]))[0]


def relative_strength_scores(
    prices_by_ticker: dict[str, pd.DataFrame],
    signal_date: pd.Timestamp,
    windows: tuple[int, int] = (20, 60),
) -> dict[str, float]:
    scores: dict[str, float] = {}
    for ticker, prices in prices_by_ticker.items():
        history = prices.loc[prices.index <= signal_date, "adj_close"].dropna()
        if len(history) <= max(windows):
            continue
        short_return = history.iloc[-1] / history.iloc[-windows[0]] - 1
        long_return = history.iloc[-1] / history.iloc[-windows[1]] - 1
        scores[ticker] = (0.4 * short_return) + (0.6 * long_return)
    return scores


def dual_momentum_vol_control(
    prices_by_ticker: dict[str, pd.DataFrame],
    signal_date: pd.Timestamp,
    momentum_windows: tuple[int, int] = (63, 126),
    trend_window: int = 126,
    volatility_window: int = 20,
) -> str | None:
    scores = dual_momentum_scores(
        prices_by_ticker,
        signal_date,
        momentum_windows=momentum_windows,
        trend_window=trend_window,
        volatility_window=volatility_window,
    )
    if not scores:
        return None
    return max(scores.items(), key=lambda item: (item[1], item[0]))[0]


def theme_enhanced_dual_momentum(
    prices_by_ticker: dict[str, pd.DataFrame],
    signal_date: pd.Timestamp,
    theme_by_ticker: dict[str, tuple[str, ...]],
    momentum_windows: tuple[int, int] = (63, 126),
    trend_window: int = 126,
    volatility_window: int = 20,
    theme_window: int = 20,
    theme_weight: float = 0.25,
) -> str | None:
    base_scores = dual_momentum_scores(
        prices_by_ticker,
        signal_date,
        momentum_windows=momentum_windows,
        trend_window=trend_window,
        volatility_window=volatility_window,
    )
    if not base_scores:
        return None
    theme_scores = _theme_scores(prices_by_ticker, signal_date, theme_by_ticker, theme_window)
    adjusted_scores: dict[str, float] = {}
    for ticker, score in base_scores.items():
        themes = theme_by_ticker.get(ticker, ())
        theme_bonus = max((theme_scores.get(theme, 0.0) for theme in themes), default=0.0)
        adjusted_scores[ticker] = score + (theme_weight * theme_bonus)
    return max(adjusted_scores.items(), key=lambda item: (item[1], item[0]))[0]


def dual_momentum_scores(
    prices_by_ticker: dict[str, pd.DataFrame],
    signal_date: pd.Timestamp,
    momentum_windows: tuple[int, int] = (63, 126),
    trend_window: int = 126,
    volatility_window: int = 20,
) -> dict[str, float]:
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
    return scores


def _theme_scores(
    prices_by_ticker: dict[str, pd.DataFrame],
    signal_date: pd.Timestamp,
    theme_by_ticker: dict[str, tuple[str, ...]],
    theme_window: int,
) -> dict[str, float]:
    values: dict[str, list[float]] = {}
    for ticker, themes in theme_by_ticker.items():
        if ticker not in prices_by_ticker:
            continue
        history = prices_by_ticker[ticker].loc[prices_by_ticker[ticker].index <= signal_date, "adj_close"].dropna()
        if len(history) <= theme_window:
            continue
        theme_return = float(history.iloc[-1] / history.iloc[-theme_window] - 1)
        for theme in themes:
            values.setdefault(theme, []).append(theme_return)
    scores: dict[str, float] = {}
    for theme, returns in values.items():
        if not returns:
            continue
        average_return = sum(returns) / len(returns)
        strong_ratio = sum(1 for value in returns if value > 0) / len(returns)
        scores[theme] = (0.7 * average_return) + (0.3 * strong_ratio)
    return scores
