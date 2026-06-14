from __future__ import annotations

from dataclasses import dataclass, replace

import pandas as pd

from backtest_lab.risk_factor_source import RiskFactorSignal
from backtest_lab.valuation_source import ValuationSignal


POOL_HIGH_LIQUIDITY = "high_liquidity"
POOL_STANDARD_LIQUIDITY = "standard_liquidity"
POOL_LOW_LIQUIDITY_OR_MIXED = "low_liquidity_or_mixed"
SIZE_LARGE_CAP = "large_cap"
SIZE_MID_CAP = "mid_cap"
SIZE_SMALL_CAP = "small_cap"
SIZE_MICRO_CAP = "micro_cap"
SIZE_UNKNOWN = "unknown_size"


@dataclass(frozen=True)
class PoolProfile:
    pool_type: str
    ticker_count: int
    median_turnover_twd: float
    has_theme_map: bool
    classification_basis: str = "liquidity"


@dataclass(frozen=True)
class UniversalPoolParameters:
    min_avg_turnover_twd: float
    min_stock_score: float
    overheated_20d_return: float
    require_ma60: bool = True
    score_mode: str = "risk_adjusted"
    max_stock_drawdown_20d: float = -0.25
    risk_signal_weight: float = 0.0
    valuation_signal_weight: float = 0.0
    require_valuation_gate: bool = False


@dataclass(frozen=True)
class SizeClassificationThresholds:
    large_min_twd: float = 500_000_000_000
    mid_min_twd: float = 50_000_000_000
    small_min_twd: float = 5_000_000_000


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
    liquidity_profile: str = ""
    size_profile: str = SIZE_UNKNOWN
    market_cap_twd: float = 0.0
    size_basis: str = ""
    profile_type: str = ""
    applied_score_mode: str = ""
    flow_risk_score: float = 0.0
    institutional_risk: float = 0.0
    margin_risk: float = 0.0
    borrow_risk: float = 0.0
    day_trading_risk: float = 0.0
    sentiment_risk: float = 0.0
    bullish_flow_score: float = 0.0
    sentiment_score: float = 0.0
    flow_score_adjustment: float = 0.0
    flow_risk_reasons: str = ""
    flow_source_dates: str = ""
    flow_source_kinds: str = ""
    valuation_score_adjustment: float = 0.0
    valuation_gate_passed: bool = True
    valuation_safety_margin_pct: float = 0.0
    valuation_fair_price: float = 0.0
    valuation_buy_price: float = 0.0
    valuation_reason: str = ""
    valuation_source_date: str = ""


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
        pool_type = POOL_HIGH_LIQUIDITY
    elif median_turnover >= 50_000_000:
        pool_type = POOL_STANDARD_LIQUIDITY
    else:
        pool_type = POOL_LOW_LIQUIDITY_OR_MIXED
    return PoolProfile(
        pool_type=pool_type,
        ticker_count=ticker_count,
        median_turnover_twd=median_turnover,
        has_theme_map=has_theme_map,
    )


def default_parameters_for_profile(profile: PoolProfile) -> UniversalPoolParameters:
    if profile.pool_type == POOL_HIGH_LIQUIDITY:
        return UniversalPoolParameters(
            min_avg_turnover_twd=0.0,
            min_stock_score=0.0,
            overheated_20d_return=0.90,
            require_ma60=True,
            score_mode="relative_strength",
            max_stock_drawdown_20d=-0.30,
        )
    if profile.pool_type == POOL_STANDARD_LIQUIDITY:
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


def core_defensive_parameters_for_profile(profile: PoolProfile) -> UniversalPoolParameters:
    base = default_parameters_for_profile(profile)
    return replace(
        base,
        min_avg_turnover_twd=max(base.min_avg_turnover_twd, 60_000_000),
        min_stock_score=max(base.min_stock_score, 0.02),
        overheated_20d_return=min(base.overheated_20d_return, 0.45),
        require_ma60=True,
        score_mode="risk_adjusted",
        max_stock_drawdown_20d=max(base.max_stock_drawdown_20d, -0.18),
        risk_signal_weight=0.20,
    )


def score_universal_candidates(
    prices_by_ticker: dict[str, pd.DataFrame],
    signal_date: pd.Timestamp,
    params: UniversalPoolParameters,
    *,
    conviction_by_ticker: dict[str, float] | None = None,
    market_cap_by_ticker: dict[str, float] | None = None,
    risk_signal_by_ticker: dict[str, RiskFactorSignal] | None = None,
    valuation_signal_by_ticker: dict[str, ValuationSignal] | None = None,
    size_thresholds: SizeClassificationThresholds = SizeClassificationThresholds(),
    enforce_pool_parameters: bool = False,
) -> dict[str, UniversalCandidateScore]:
    scores: dict[str, UniversalCandidateScore] = {}
    conviction_by_ticker = conviction_by_ticker or {}
    market_cap_by_ticker = market_cap_by_ticker or {}
    risk_signal_by_ticker = risk_signal_by_ticker or {}
    valuation_signal_by_ticker = valuation_signal_by_ticker or {}
    for ticker, prices in prices_by_ticker.items():
        liquidity_profile = classify_candidate_liquidity_profile(prices, signal_date)
        size_profile, market_cap_twd, size_basis = classify_candidate_size_profile(
            prices,
            signal_date,
            market_cap_twd=market_cap_by_ticker.get(ticker),
            thresholds=size_thresholds,
        )
        candidate_params = parameters_for_candidate_route(size_profile, liquidity_profile)
        if enforce_pool_parameters:
            candidate_params = replace(
                candidate_params,
                min_avg_turnover_twd=max(candidate_params.min_avg_turnover_twd, params.min_avg_turnover_twd),
                min_stock_score=max(candidate_params.min_stock_score, params.min_stock_score),
                overheated_20d_return=min(candidate_params.overheated_20d_return, params.overheated_20d_return),
                require_ma60=candidate_params.require_ma60 or params.require_ma60,
                score_mode=params.score_mode,
                max_stock_drawdown_20d=max(candidate_params.max_stock_drawdown_20d, params.max_stock_drawdown_20d),
                risk_signal_weight=params.risk_signal_weight,
            )
        else:
            candidate_params = replace(candidate_params, risk_signal_weight=params.risk_signal_weight)
        scores[ticker] = score_universal_candidate(
            ticker=ticker,
            prices=prices,
            signal_date=signal_date,
            params=candidate_params,
            conviction_bonus=conviction_by_ticker.get(ticker, 0.0),
            risk_signal=risk_signal_by_ticker.get(ticker),
            valuation_signal=valuation_signal_by_ticker.get(ticker),
            liquidity_profile=liquidity_profile,
            size_profile=size_profile,
            market_cap_twd=market_cap_twd,
            size_basis=size_basis,
        )
    return scores


def score_universal_candidate(
    *,
    ticker: str,
    prices: pd.DataFrame,
    signal_date: pd.Timestamp,
    params: UniversalPoolParameters,
    conviction_bonus: float = 0.0,
    risk_signal: RiskFactorSignal | None = None,
    valuation_signal: ValuationSignal | None = None,
    liquidity_profile: str = "",
    size_profile: str = SIZE_UNKNOWN,
    market_cap_twd: float = 0.0,
    size_basis: str = "",
) -> UniversalCandidateScore:
    history = prices.loc[prices.index <= signal_date].dropna(subset=["adj_close"])
    if len(history) < 126:
        return _candidate_reject(
            ticker,
            "warmup不足",
            liquidity_profile=liquidity_profile,
            size_profile=size_profile,
            market_cap_twd=market_cap_twd,
            size_basis=size_basis,
            applied_score_mode=params.score_mode,
            risk_signal=risk_signal,
            valuation_signal=valuation_signal,
        )

    close = float(history["adj_close"].iloc[-1])
    ma20 = float(history["adj_close"].iloc[-20:].mean())
    ma60 = float(history["adj_close"].iloc[-60:].mean())
    if close < ma20:
        return _candidate_reject(
            ticker,
            "跌破20日均線",
            liquidity_profile=liquidity_profile,
            size_profile=size_profile,
            market_cap_twd=market_cap_twd,
            size_basis=size_basis,
            applied_score_mode=params.score_mode,
            risk_signal=risk_signal,
            valuation_signal=valuation_signal,
        )
    if params.require_ma60 and close < ma60:
        return _candidate_reject(
            ticker,
            "跌破60日均線",
            liquidity_profile=liquidity_profile,
            size_profile=size_profile,
            market_cap_twd=market_cap_twd,
            size_basis=size_basis,
            applied_score_mode=params.score_mode,
            risk_signal=risk_signal,
            valuation_signal=valuation_signal,
        )

    volume = history["volume"].fillna(0) if "volume" in history.columns else pd.Series(0, index=history.index)
    avg_turnover = float((history["close"] * volume).tail(20).mean()) if "close" in history.columns else 0.0
    if avg_turnover < params.min_avg_turnover_twd:
        return _candidate_reject(
            ticker,
            "流動性不足",
            avg_turnover_twd=avg_turnover,
            liquidity_profile=liquidity_profile,
            size_profile=size_profile,
            market_cap_twd=market_cap_twd,
            size_basis=size_basis,
            applied_score_mode=params.score_mode,
            risk_signal=risk_signal,
            valuation_signal=valuation_signal,
        )

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
            liquidity_profile=liquidity_profile,
            size_profile=size_profile,
            market_cap_twd=market_cap_twd,
            size_basis=size_basis,
            applied_score_mode=params.score_mode,
            risk_signal=risk_signal,
            valuation_signal=valuation_signal,
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
            liquidity_profile=liquidity_profile,
            size_profile=size_profile,
            market_cap_twd=market_cap_twd,
            size_basis=size_basis,
            applied_score_mode=params.score_mode,
            risk_signal=risk_signal,
            valuation_signal=valuation_signal,
        )
    if params.require_valuation_gate and valuation_signal and not valuation_signal.gate_passed:
        return _candidate_reject(
            ticker,
            "估值安全邊際不足",
            ret20=ret20,
            ret60=ret60,
            ret120=ret120,
            avg_turnover_twd=avg_turnover,
            drawdown20=drawdown20,
            liquidity_profile=liquidity_profile,
            size_profile=size_profile,
            market_cap_twd=market_cap_twd,
            size_basis=size_basis,
            applied_score_mode=params.score_mode,
            risk_signal=risk_signal,
            valuation_signal=valuation_signal,
        )

    vol20 = float(history["adj_close"].pct_change().dropna().iloc[-20:].std() * (252**0.5))
    flow_score_adjustment = (risk_signal.score_adjustment * params.risk_signal_weight) if risk_signal else 0.0
    valuation_score_adjustment = (
        valuation_signal.score_adjustment * params.valuation_signal_weight
        if valuation_signal
        else 0.0
    )
    score = universal_stock_score(
        ret20=ret20,
        ret60=ret60,
        ret120=ret120,
        vol20=vol20,
        mode=params.score_mode,
        conviction_bonus=conviction_bonus,
    ) + flow_score_adjustment + valuation_score_adjustment
    if score < params.min_stock_score:
        return _attach_risk_signal(
            UniversalCandidateScore(
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
                liquidity_profile=liquidity_profile,
                size_profile=size_profile,
                market_cap_twd=market_cap_twd,
                size_basis=size_basis,
                profile_type=liquidity_profile,
                applied_score_mode=params.score_mode,
                flow_score_adjustment=flow_score_adjustment,
                valuation_score_adjustment=valuation_score_adjustment,
            ),
            risk_signal,
            valuation_signal,
        )
    return _attach_risk_signal(
        UniversalCandidateScore(
            ticker=ticker,
            score=score,
            ret20=ret20,
            ret60=ret60,
            ret120=ret120,
            vol20=vol20,
            avg_turnover_twd=avg_turnover,
            drawdown20=drawdown20,
            passed=True,
            liquidity_profile=liquidity_profile,
            size_profile=size_profile,
            market_cap_twd=market_cap_twd,
            size_basis=size_basis,
            profile_type=liquidity_profile,
            applied_score_mode=params.score_mode,
            flow_score_adjustment=flow_score_adjustment,
            valuation_score_adjustment=valuation_score_adjustment,
        ),
        risk_signal,
        valuation_signal,
    )


def classify_candidate_liquidity_profile(prices: pd.DataFrame, signal_date: pd.Timestamp) -> str:
    history = prices.loc[prices.index <= signal_date].dropna(subset=["close"])
    if len(history) < 20 or "volume" not in history.columns:
        return POOL_LOW_LIQUIDITY_OR_MIXED
    volume = history["volume"].fillna(0)
    avg_turnover = float((history["close"] * volume).tail(20).mean())
    if avg_turnover >= 1_000_000_000:
        return POOL_HIGH_LIQUIDITY
    if avg_turnover >= 60_000_000:
        return POOL_STANDARD_LIQUIDITY
    return POOL_LOW_LIQUIDITY_OR_MIXED


def classify_candidate_profile(prices: pd.DataFrame, signal_date: pd.Timestamp) -> str:
    return classify_candidate_liquidity_profile(prices, signal_date)


def classify_candidate_size_profile(
    prices: pd.DataFrame,
    signal_date: pd.Timestamp,
    *,
    market_cap_twd: float | None = None,
    thresholds: SizeClassificationThresholds = SizeClassificationThresholds(),
) -> tuple[str, float, str]:
    cap = _resolve_market_cap_twd(prices, signal_date, market_cap_twd=market_cap_twd)
    if cap <= 0:
        return SIZE_UNKNOWN, 0.0, ""
    if cap >= thresholds.large_min_twd:
        return SIZE_LARGE_CAP, cap, "market_cap_twd"
    if cap >= thresholds.mid_min_twd:
        return SIZE_MID_CAP, cap, "market_cap_twd"
    if cap >= thresholds.small_min_twd:
        return SIZE_SMALL_CAP, cap, "market_cap_twd"
    return SIZE_MICRO_CAP, cap, "market_cap_twd"


def _resolve_market_cap_twd(
    prices: pd.DataFrame,
    signal_date: pd.Timestamp,
    *,
    market_cap_twd: float | None = None,
) -> float:
    if market_cap_twd is not None and pd.notna(market_cap_twd):
        return float(market_cap_twd)
    history = prices.loc[prices.index <= signal_date]
    for column in ("free_float_market_cap_twd", "market_cap_twd"):
        if column not in history.columns:
            continue
        series = pd.to_numeric(history[column], errors="coerce").dropna()
        if not series.empty and float(series.iloc[-1]) > 0:
            return float(series.iloc[-1])
    return 0.0


def parameters_for_candidate_route(size_profile: str, liquidity_profile: str) -> UniversalPoolParameters:
    if size_profile == SIZE_LARGE_CAP and liquidity_profile != POOL_LOW_LIQUIDITY_OR_MIXED:
        return parameters_for_liquidity_profile(POOL_HIGH_LIQUIDITY)
    if size_profile in {SIZE_SMALL_CAP, SIZE_MICRO_CAP}:
        return parameters_for_liquidity_profile(POOL_LOW_LIQUIDITY_OR_MIXED)
    if size_profile == SIZE_MID_CAP:
        return parameters_for_liquidity_profile(POOL_STANDARD_LIQUIDITY)
    return parameters_for_liquidity_profile(liquidity_profile)


def parameters_for_liquidity_profile(liquidity_profile: str) -> UniversalPoolParameters:
    return default_parameters_for_profile(
        PoolProfile(
            pool_type=liquidity_profile,
            ticker_count=1,
            median_turnover_twd=0.0,
            has_theme_map=False,
        )
    )


def parameters_for_candidate_profile(profile_type: str) -> UniversalPoolParameters:
    return parameters_for_liquidity_profile(profile_type)


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
    liquidity_profile: str = "",
    size_profile: str = SIZE_UNKNOWN,
    market_cap_twd: float = 0.0,
    size_basis: str = "",
    profile_type: str = "",
    applied_score_mode: str = "",
    risk_signal: RiskFactorSignal | None = None,
    valuation_signal: ValuationSignal | None = None,
) -> UniversalCandidateScore:
    profile_type = profile_type or liquidity_profile
    return _attach_risk_signal(
        UniversalCandidateScore(
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
            liquidity_profile=liquidity_profile,
            size_profile=size_profile,
            market_cap_twd=market_cap_twd,
            size_basis=size_basis,
            profile_type=profile_type,
            applied_score_mode=applied_score_mode,
        ),
        risk_signal,
        valuation_signal,
    )


def _attach_risk_signal(
    candidate: UniversalCandidateScore,
    risk_signal: RiskFactorSignal | None,
    valuation_signal: ValuationSignal | None = None,
) -> UniversalCandidateScore:
    candidate = _attach_valuation_signal(candidate, valuation_signal)
    if risk_signal is None:
        return candidate
    return UniversalCandidateScore(
        ticker=candidate.ticker,
        score=candidate.score,
        ret20=candidate.ret20,
        ret60=candidate.ret60,
        ret120=candidate.ret120,
        vol20=candidate.vol20,
        avg_turnover_twd=candidate.avg_turnover_twd,
        drawdown20=candidate.drawdown20,
        passed=candidate.passed,
        reason=candidate.reason,
        liquidity_profile=candidate.liquidity_profile,
        size_profile=candidate.size_profile,
        market_cap_twd=candidate.market_cap_twd,
        size_basis=candidate.size_basis,
        profile_type=candidate.profile_type,
        applied_score_mode=candidate.applied_score_mode,
        flow_risk_score=risk_signal.total_risk_score,
        institutional_risk=risk_signal.institutional_risk,
        margin_risk=risk_signal.margin_risk,
        borrow_risk=risk_signal.borrow_risk,
        day_trading_risk=risk_signal.day_trading_risk,
        sentiment_risk=risk_signal.sentiment_risk,
        bullish_flow_score=risk_signal.bullish_flow_score,
        sentiment_score=risk_signal.sentiment_score,
        flow_score_adjustment=candidate.flow_score_adjustment,
        flow_risk_reasons=risk_signal.reason_text,
        flow_source_dates=",".join(risk_signal.source_dates),
        flow_source_kinds=",".join(risk_signal.source_kinds),
        valuation_score_adjustment=candidate.valuation_score_adjustment,
        valuation_gate_passed=candidate.valuation_gate_passed,
        valuation_safety_margin_pct=candidate.valuation_safety_margin_pct,
        valuation_fair_price=candidate.valuation_fair_price,
        valuation_buy_price=candidate.valuation_buy_price,
        valuation_reason=candidate.valuation_reason,
        valuation_source_date=candidate.valuation_source_date,
    )


def _attach_valuation_signal(
    candidate: UniversalCandidateScore,
    valuation_signal: ValuationSignal | None,
) -> UniversalCandidateScore:
    if valuation_signal is None:
        return candidate
    return UniversalCandidateScore(
        ticker=candidate.ticker,
        score=candidate.score,
        ret20=candidate.ret20,
        ret60=candidate.ret60,
        ret120=candidate.ret120,
        vol20=candidate.vol20,
        avg_turnover_twd=candidate.avg_turnover_twd,
        drawdown20=candidate.drawdown20,
        passed=candidate.passed,
        reason=candidate.reason,
        liquidity_profile=candidate.liquidity_profile,
        size_profile=candidate.size_profile,
        market_cap_twd=candidate.market_cap_twd,
        size_basis=candidate.size_basis,
        profile_type=candidate.profile_type,
        applied_score_mode=candidate.applied_score_mode,
        flow_risk_score=candidate.flow_risk_score,
        institutional_risk=candidate.institutional_risk,
        margin_risk=candidate.margin_risk,
        borrow_risk=candidate.borrow_risk,
        day_trading_risk=candidate.day_trading_risk,
        sentiment_risk=candidate.sentiment_risk,
        bullish_flow_score=candidate.bullish_flow_score,
        sentiment_score=candidate.sentiment_score,
        flow_score_adjustment=candidate.flow_score_adjustment,
        flow_risk_reasons=candidate.flow_risk_reasons,
        flow_source_dates=candidate.flow_source_dates,
        flow_source_kinds=candidate.flow_source_kinds,
        valuation_score_adjustment=candidate.valuation_score_adjustment,
        valuation_gate_passed=valuation_signal.gate_passed,
        valuation_safety_margin_pct=valuation_signal.safety_margin_pct,
        valuation_fair_price=valuation_signal.fair_price,
        valuation_buy_price=valuation_signal.buy_price,
        valuation_reason=valuation_signal.reason,
        valuation_source_date=valuation_signal.signal_date,
    )
