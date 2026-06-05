from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd


REGIME_LABELS = {
    "strong_bull": "強多頭",
    "recovery_bull": "修復多頭",
    "range_bound": "震盪盤",
    "correction_bear": "修正空頭",
    "systemic_bear": "系統性空頭",
}


@dataclass(frozen=True)
class MarketRegimeSnapshot:
    signal_date: str
    regime: str
    regime_label: str
    confidence: float
    close: float
    ma60: float
    ma120: float
    ma200: float
    ma200_slope_20d: float
    return_20d: float
    return_60d: float
    return_120d: float
    drawdown_from_252d_high: float
    volatility_20d: float
    breadth_60d: float | None = None
    foreign_net_buy_20d: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def classify_market_regime(
    market_prices: pd.DataFrame,
    signal_date: str | pd.Timestamp,
    universe_prices: dict[str, pd.DataFrame] | None = None,
) -> MarketRegimeSnapshot:
    signal_ts = pd.Timestamp(signal_date).normalize()
    history = market_prices.loc[market_prices.index <= signal_ts, "adj_close"].dropna()
    if history.empty or history.index.max() != signal_ts:
        latest = history.index.max().date() if not history.empty else "none"
        raise ValueError(f"No market data for signal date {signal_ts.date()}; latest available is {latest}")
    if len(history) < 253:
        raise ValueError(f"Need at least 253 trading rows for market regime, got {len(history)}")

    close = float(history.iloc[-1])
    ma60 = float(history.iloc[-60:].mean())
    ma120 = float(history.iloc[-120:].mean())
    ma200_series = history.rolling(200).mean().dropna()
    ma200 = float(ma200_series.iloc[-1])
    ma200_slope_20d = float(ma200_series.iloc[-1] / ma200_series.iloc[-21] - 1) if len(ma200_series) > 20 else 0.0
    return_20d = _window_return(history, 20)
    return_60d = _window_return(history, 60)
    return_120d = _window_return(history, 120)
    high_252 = float(history.iloc[-252:].max())
    drawdown = close / high_252 - 1
    volatility_20d = float(history.pct_change().dropna().iloc[-20:].std() * (252**0.5))
    breadth_60d = _breadth_above_ma(universe_prices, signal_ts, 60) if universe_prices else None

    regime, confidence = _classify(
        close=close,
        ma60=ma60,
        ma120=ma120,
        ma200=ma200,
        ma200_slope_20d=ma200_slope_20d,
        return_20d=return_20d,
        return_60d=return_60d,
        return_120d=return_120d,
        drawdown=drawdown,
        volatility_20d=volatility_20d,
        breadth_60d=breadth_60d,
    )

    return MarketRegimeSnapshot(
        signal_date=signal_ts.strftime("%Y-%m-%d"),
        regime=regime,
        regime_label=REGIME_LABELS[regime],
        confidence=round(confidence, 4),
        close=round(close, 4),
        ma60=round(ma60, 4),
        ma120=round(ma120, 4),
        ma200=round(ma200, 4),
        ma200_slope_20d=round(ma200_slope_20d, 6),
        return_20d=round(return_20d, 6),
        return_60d=round(return_60d, 6),
        return_120d=round(return_120d, 6),
        drawdown_from_252d_high=round(drawdown, 6),
        volatility_20d=round(volatility_20d, 6),
        breadth_60d=round(breadth_60d, 6) if breadth_60d is not None else None,
    )


def latest_available_date(frame: pd.DataFrame) -> pd.Timestamp | None:
    if frame.empty:
        return None
    return pd.Timestamp(frame.index.max()).normalize()


def has_data_for_date(frame: pd.DataFrame, signal_date: str | pd.Timestamp) -> bool:
    signal_ts = pd.Timestamp(signal_date).normalize()
    latest = latest_available_date(frame)
    return latest == signal_ts


def _classify(
    *,
    close: float,
    ma60: float,
    ma120: float,
    ma200: float,
    ma200_slope_20d: float,
    return_20d: float,
    return_60d: float,
    return_120d: float,
    drawdown: float,
    volatility_20d: float,
    breadth_60d: float | None,
) -> tuple[str, float]:
    score = 0
    score += 1 if close > ma60 else -1
    score += 1 if close > ma120 else -1
    score += 1 if close > ma200 else -1
    score += 1 if ma200_slope_20d > 0 else -1
    score += 1 if return_60d > 0 else -1
    score += 1 if return_120d > 0 else -1
    if breadth_60d is not None:
        score += 1 if breadth_60d >= 0.6 else -1 if breadth_60d <= 0.4 else 0

    if close < ma200 and ma200_slope_20d < 0 and return_60d < 0 and return_120d < 0 and drawdown <= -0.20:
        return "systemic_bear", 0.9
    if close < ma200 and (return_120d < 0 or drawdown <= -0.10):
        return "correction_bear", 0.75
    if breadth_60d is not None and breadth_60d <= 0.4 and close < ma60:
        return "range_bound", 0.7
    if close < ma60 and return_20d < 0 and drawdown <= -0.03:
        return "range_bound", 0.68
    if (
        close > ma60
        and close > ma200
        and return_20d > 0
        and return_60d > 0
        and return_120d > 0
        and ma200_slope_20d > 0
        and drawdown > -0.10
        and (breadth_60d is None or breadth_60d >= 0.5)
    ):
        return "strong_bull", 0.85
    if (close > ma120 or close > ma200) and return_60d > 0 and drawdown > -0.20:
        return "recovery_bull", 0.7
    if abs(return_60d) <= 0.05 or volatility_20d > 0.28 or -3 <= score <= 3:
        return "range_bound", 0.65
    if score > 3:
        return "recovery_bull", 0.6
    return "correction_bear", 0.6


def _window_return(history: pd.Series, window: int) -> float:
    if len(history) <= window:
        return 0.0
    return float(history.iloc[-1] / history.iloc[-window] - 1)


def _breadth_above_ma(
    universe_prices: dict[str, pd.DataFrame] | None,
    signal_date: pd.Timestamp,
    window: int,
) -> float | None:
    if not universe_prices:
        return None
    values: list[bool] = []
    for prices in universe_prices.values():
        history = prices.loc[prices.index <= signal_date, "adj_close"].dropna()
        if len(history) < window:
            continue
        values.append(float(history.iloc[-1]) > float(history.iloc[-window:].mean()))
    if not values:
        return None
    return sum(values) / len(values)
