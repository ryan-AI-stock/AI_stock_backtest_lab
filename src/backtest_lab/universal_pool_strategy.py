from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


POOL_LARGE_LIQUID = "large_liquid"
POOL_MID_SMALL_LIQUID = "mid_small_liquid"
POOL_THIN_OR_MIXED = "thin_or_mixed"


@dataclass(frozen=True)
class PoolProfile:
    pool_type: str
    ticker_count: int
    median_turnover_twd: float
    has_theme_map: bool


@dataclass(frozen=True)
class UniversalPoolParameters:
    min_avg_turnover_twd: float
    min_stock_score: float
    overheated_20d_return: float
    require_ma60: bool = True
    score_mode: str = "risk_adjusted"
    max_stock_drawdown_20d: float = -0.25


@dataclass(frozen=True)
class UniversalCandidateScore:
    ticker: str
    score: float
    ret20: float
    ret60: float
    ret120: float
    vol20: float
    avg_turnover_twd: float
    drawdown20: float
    passed: bool
    reason: str = ""


def infer_pool_profile(
    prices_by_ticker: dict[str, pd.DataFrame],
    signal_date: pd.Timestamp,
    *,
    theme_by_ticker: dict[str, str] | None = None,
) -> PoolProfile:
    turnovers = []
    for prices in prices_by_ticker.values():
        history = prices.loc[prices.index <= signal_date].dropna(subset=["close"])
        if len(history) < 20 or "volume" not in history.columns:
            continue
        turnovers.append(float((history["close"] * history["volume"].fillna(0)).tail(20).mean()))
    median_turnover = float(pd.Series(turnovers).median()) if turnovers else 0.0
    ticker_count = len(prices_by_ticker)
    has_theme_map = bool(theme_by_ticker)

    if ticker_count <= 12 and median_turnover >= 1_000_000_000:
        pool_type = POOL_LARGE_LIQUID
    elif median_turnover >= 50_000_000:
        pool_type = POOL_MID_SMALL_LIQUID
    else:
        pool_type = POOL_THIN_OR_MIXED
    return PoolProfile(
        pool_type=pool_type,
        ticker_count=ticker_count,
        median_turnover_twd=median_turnover,
        has_theme_map=has_theme_map,
    )


def default_parameters_for_profile(profile: PoolProfile) -> UniversalPoolParameters:
    if profile.pool_type == POOL_LARGE_LIQUID:
        return UniversalPoolParameters(
            min_avg_turnover_twd=0.0,
            min_stock_score=0.0,
            overheated_20d_return=0.90,
            require_ma60=True,
            score_mode="relative_strength",
            max_stock_drawdown_20d=-0.30,
        )
    if profile.pool_type == POOL_MID_SMALL_LIQUID:
        return UniversalPoolParameters(
            min_avg_turnover_twd=60_000_000,
            min_stock_score=0.0,
            overheated_20d_return=0.62,
            require_ma60=True,
            score_mode="risk_adjusted",
            max_stock_drawdown_20d=-0.25,
        )
    return UniversalPoolParameters(
        min_avg_turnover_twd=120_000_000,
        min_stock_score=0.08,
        overheated_20d_return=0.55,
        require_ma60=True,
        score_mode="risk_adjusted",
        max_stock_drawdown_20d=-0.20,
    )


def score_universal_candidates(
    prices_by_ticker: dict[str, pd.DataFrame],
    signal_date: pd.Timestamp,
    params: UniversalPoolParameters,
    *,
    conviction_by_ticker: dict[str, float] | None = None,
) -> dict[str, UniversalCandidateScore]:
    scores: dict[str, UniversalCandidateScore] = {}
    conviction_by_ticker = conviction_by_ticker or {}
    for ticker, prices in prices_by_ticker.items():
        scores[ticker] = score_universal_candidate(
            ticker=ticker,
            prices=prices,
            signal_date=signal_date,
            params=params,
            conviction_bonus=conviction_by_ticker.get(ticker, 0.0),
        )
    return scores


def score_universal_candidate(
    *,
    ticker: str,
    prices: pd.DataFrame,
    signal_date: pd.Timestamp,
    params: UniversalPoolParameters,
    conviction_bonus: float = 0.0,
) -> UniversalCandidateScore:
    history = prices.loc[prices.index <= signal_date].dropna(subset=["adj_close"])
    if len(history) < 126:
        return _candidate_reject(ticker, "warmup不足")

    close = float(history["adj_close"].iloc[-1])
    ma20 = float(history["adj_close"].iloc[-20:].mean())
    ma60 = float(history["adj_close"].iloc[-60:].mean())
    if close < ma20:
        return _candidate_reject(ticker, "跌破20日均線")
    if params.require_ma60 and close < ma60:
        return _candidate_reject(ticker, "跌破60日均線")

    volume = history["volume"].fillna(0) if "volume" in history.columns else pd.Series(0, index=history.index)
    avg_turnover = float((history["close"] * volume).tail(20).mean()) if "close" in history.columns else 0.0
    if avg_turnover < params.min_avg_turnover_twd:
        return _candidate_reject(ticker, "流動性不足", avg_turnover_twd=avg_turnover)

    ret20 = window_return(history["adj_close"], 20)
    ret60 = window_return(history["adj_close"], 60)
    ret120 = window_return(history["adj_close"], 120)
    drawdown20 = close / float(history["adj_close"].iloc[-20:].max()) - 1
    if ret20 > params.overheated_20d_return:
        return _candidate_reject(
            ticker,
            "20日漲幅過熱",
            ret20=ret20,
            ret60=ret60,
            ret120=ret120,
            avg_turnover_twd=avg_turnover,
            drawdown20=drawdown20,
        )
    if drawdown20 < params.max_stock_drawdown_20d:
        return _candidate_reject(
            ticker,
            "20日回撤過深",
            ret20=ret20,
            ret60=ret60,
            ret120=ret120,
            avg_turnover_twd=avg_turnover,
            drawdown20=drawdown20,
        )

    vol20 = float(history["adj_close"].pct_change().dropna().iloc[-20:].std() * (252**0.5))
    score = universal_stock_score(
        ret20=ret20,
        ret60=ret60,
        ret120=ret120,
        vol20=vol20,
        mode=params.score_mode,
        conviction_bonus=conviction_bonus,
    )
    if score < params.min_stock_score:
        return UniversalCandidateScore(
            ticker=ticker,
            score=score,
            ret20=ret20,
            ret60=ret60,
            ret120=ret120,
            vol20=vol20,
            avg_turnover_twd=avg_turnover,
            drawdown20=drawdown20,
            passed=False,
            reason="分數未達門檻",
        )
    return UniversalCandidateScore(
        ticker=ticker,
        score=score,
        ret20=ret20,
        ret60=ret60,
        ret120=ret120,
        vol20=vol20,
        avg_turnover_twd=avg_turnover,
        drawdown20=drawdown20,
        passed=True,
    )


def universal_stock_score(
    *,
    ret20: float,
    ret60: float,
    ret120: float,
    vol20: float,
    mode: str,
    conviction_bonus: float = 0.0,
) -> float:
    if mode == "relative_strength":
        return (0.4 * ret20) + (0.6 * ret60) + conviction_bonus
    if mode == "momentum":
        return (0.35 * ret20) + (0.35 * ret60) + (0.15 * ret120) - (0.10 * vol20) + conviction_bonus
    if mode == "raw_momentum":
        return (0.40 * ret20) + (0.40 * ret60) + (0.20 * ret120) + conviction_bonus
    if mode == "trend_momentum":
        return (0.20 * ret20) + (0.45 * ret60) + (0.25 * ret120) - (0.08 * vol20) + conviction_bonus
    if mode == "short_momentum":
        return (0.55 * ret20) + (0.25 * ret60) + (0.05 * ret120) - (0.10 * vol20) + conviction_bonus
    if mode == "risk_adjusted":
        return (0.30 * ret20) + (0.40 * ret60) + (0.20 * ret120) - (0.22 * vol20) + conviction_bonus
    raise ValueError(f"Unsupported universal score mode: {mode}")


def window_return(series: pd.Series, window: int) -> float:
    if len(series) <= window:
        return 0.0
    return float(series.iloc[-1] / series.iloc[-window] - 1)


def _candidate_reject(
    ticker: str,
    reason: str,
    *,
    ret20: float = 0.0,
    ret60: float = 0.0,
    ret120: float = 0.0,
    vol20: float = 0.0,
    avg_turnover_twd: float = 0.0,
    drawdown20: float = 0.0,
) -> UniversalCandidateScore:
    return UniversalCandidateScore(
        ticker=ticker,
        score=0.0,
        ret20=ret20,
        ret60=ret60,
        ret120=ret120,
        vol20=vol20,
        avg_turnover_twd=avg_turnover_twd,
        drawdown20=drawdown20,
        passed=False,
        reason=reason,
    )
