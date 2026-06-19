from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import pandas as pd

from backtest_lab.costs import TaiwanCostModel
from backtest_lab.market_regime import classify_market_regime
from backtest_lab.sector_dynamic_pool_backtest import MultiPositionAccount, ThemeMember
from backtest_lab.simulation import BacktestResult, _date_str, _max_drawdown
from backtest_lab.universal_pool_strategy import (
    default_parameters_for_profile,
    infer_pool_profile,
    score_universal_candidate,
    universal_stock_score,
    window_return,
)


@dataclass(frozen=True)
class RadarCoreVariant:
    name: str = "radar_core_pool_v1"
    label: str = "雷達核心成員池 v1"
    rebalance_frequency: str = "weekly"
    top_theme_count: int = 2
    max_stocks_per_theme: int = 2
    max_single_weight: float = 0.25
    min_avg_turnover_twd: float = 80_000_000
    min_theme_score: float = 0.12
    min_stock_score: float = 0.05
    overheated_20d_return: float = 0.55
    max_stock_drawdown_20d: float = -0.25
    rebalance_band: float = 0.05
    strong_bull_exposure: float = 0.90
    recovery_bull_exposure: float = 0.65
    range_bound_exposure: float = 0.35
    correction_bear_exposure: float = 0.10
    systemic_bear_exposure: float = 0.0
    daily_blowoff_exit: bool = False
    blowoff_min_runup_20d: float = 0.25
    blowoff_drawdown_10d: float = -0.08
    blowoff_volume_ratio_5d_over_60d: float = 1.5
    selection_mode: str = "per_theme"
    max_total_stocks: int = 0
    weak_theme_full_score: float = 0.0
    weak_theme_exposure_multiplier: float = 1.0
    hold_existing_score_ratio: float = 0.0
    daily_trend_exit: bool = False
    trend_exit_ma_window: int = 20
    trend_exit_drawdown_20d: float = -0.12
    portfolio_trailing_stop: bool = False
    portfolio_stop_drawdown: float = -0.12
    portfolio_stop_cooldown_days: int = 5
    portfolio_stop_min_gain: float = 0.0
    position_trailing_stop: bool = False
    position_stop_min_runup: float = 0.30
    position_stop_drawdown: float = -0.15
    position_stop_cooldown_days: int = 5
    stock_score_mode: str = "momentum"
    require_ma60: bool = True
    use_pool_profile_defaults: bool = False


@dataclass
class RadarCoreResult:
    result: BacktestResult
    theme_log: pd.DataFrame
    stock_log: pd.DataFrame
    holdings: pd.DataFrame


def radar_core_mid_small_calibrated_v1_variant() -> RadarCoreVariant:
    """Calibrated radar mid/small-cap preset found from the 2022-2023 research cycle."""
    return RadarCoreVariant(
        name="radar_core_v1_score_risk_stock00_turnover60m_overheat62",
        label="雷達核心成員池 v1 風險調整分數 + 流動性60M + 過熱62",
        top_theme_count=1,
        max_stocks_per_theme=1,
        max_single_weight=1.0,
        min_avg_turnover_twd=60_000_000,
        min_theme_score=0.20,
        min_stock_score=0.00,
        overheated_20d_return=0.62,
        strong_bull_exposure=1.0,
        recovery_bull_exposure=1.0,
        range_bound_exposure=1.0,
        correction_bear_exposure=1.0,
        systemic_bear_exposure=0.0,
        rebalance_band=0.08,
        stock_score_mode="risk_adjusted",
    )


def load_radar_core_members(path: str | Path) -> list[ThemeMember]:
    csv_path = Path(path)
    ticker_overrides = _load_radar_ticker_overrides(csv_path)
    frame = pd.read_csv(csv_path, dtype={"symbol": str}).fillna("")
    frame = frame[frame["primary"].astype(str).str.lower() != "no"]
    members: list[ThemeMember] = []
    seen: set[tuple[str, str]] = set()
    for _, row in frame.iterrows():
        symbol = str(row["symbol"]).strip()
        theme = str(row["theme"]).strip()
        if not symbol or not theme or (theme, symbol) in seen:
            continue
        seen.add((theme, symbol))
        members.append(
            ThemeMember(
                theme=theme,
                ticker=ticker_overrides.get(symbol, f"{symbol}.TW"),
                symbol=symbol,
                name=str(row["name"]).strip(),
                role=str(row.get("role", "")).strip(),
                conviction=str(row.get("conviction", "")).strip(),
            )
        )
    return members


def _load_radar_ticker_overrides(theme_map_path: Path) -> dict[str, str]:
    """Use RADAR's formal symbol exchange map when it is available."""
    data_root = theme_map_path.parent
    candidates = [
        data_root / "formal_sources" / "date_aware_theme_membership_full_2022_2023.csv",
        data_root / "formal_sources" / "date_aware_theme_membership_full_2022_2023_gap.csv",
    ]
    overrides: dict[str, str] = {}
    for candidate in candidates:
        if not candidate.exists():
            continue
        frame = pd.read_csv(candidate, dtype={"symbol": str}).fillna("")
        if "symbol" not in frame.columns or "ticker" not in frame.columns:
            continue
        for _, row in frame.iterrows():
            symbol = str(row["symbol"]).strip()
            ticker = str(row["ticker"]).strip()
            if symbol and ticker:
                overrides.setdefault(symbol, ticker)
    market_universe = data_root / "market_universe.generated.csv"
    if market_universe.exists():
        frame = pd.read_csv(market_universe, dtype={"symbol": str}).fillna("")
        if "symbol" in frame.columns and "market" in frame.columns:
            for _, row in frame.iterrows():
                symbol = str(row["symbol"]).strip()
                market = str(row["market"]).strip().upper()
                if not symbol:
                    continue
                if market == "TPEX":
                    overrides.setdefault(symbol, f"{symbol}.TWO")
                elif market == "TWSE":
                    overrides.setdefault(symbol, f"{symbol}.TW")
    return overrides


def simulate_radar_core_pool(
    *,
    name: str,
    prices_by_ticker: dict[str, pd.DataFrame],
    members_by_ticker: dict[str, ThemeMember],
    asset_types: dict[str, str],
    market_prices: pd.DataFrame,
    start_date: str,
    end_date: str,
    initial_cash: float,
    cost_model: TaiwanCostModel,
    variant: RadarCoreVariant = RadarCoreVariant(),
    dividend_series_by_ticker: dict[str, pd.Series] | None = None,
) -> RadarCoreResult:
    trade_dates = _market_trade_dates(market_prices, start_date, end_date)
    if not trade_dates:
        raise ValueError(f"No market trade dates between {start_date} and {end_date}")
    variant = resolve_variant_for_pool(
        variant=variant,
        prices_by_ticker=prices_by_ticker,
        members_by_ticker=members_by_ticker,
        signal_date=_previous_market_signal_date(market_prices, trade_dates[0]),
    )
    account = MultiPositionAccount(initial_cash, cost_model)
    equity_rows: list[dict] = []
    holding_rows: list[dict] = []
    theme_rows: list[dict] = []
    stock_rows: list[dict] = []
    peak_signal_value = initial_cash
    stop_until_index = -1
    position_entry_prices: dict[str, float] = {}
    position_peak_prices: dict[str, float] = {}
    ticker_stop_until_index: dict[str, int] = {}

    for index, trade_date in enumerate(trade_dates):
        if dividend_series_by_ticker is not None:
            for ticker in list(account.positions):
                dividend_series = dividend_series_by_ticker.get(ticker)
                if dividend_series is not None:
                    account.credit_dividend(trade_date, ticker, float(dividend_series.get(trade_date, 0.0)))

        signal_date = _previous_market_signal_date(market_prices, trade_date)
        forced_exits: set[str] = set()
        portfolio_stop_active = index <= stop_until_index
        if variant.portfolio_trailing_stop and not account.positions and not portfolio_stop_active:
            peak_signal_value = account.cash
        if variant.portfolio_trailing_stop and account.positions:
            signal_prices = _latest_price_lookup(prices_by_ticker, signal_date, "close", set(account.positions))
            if set(account.positions) <= set(signal_prices):
                signal_value = account.value(signal_prices)
                peak_signal_value = max(peak_signal_value, signal_value)
                if _portfolio_stop_triggered(signal_value, peak_signal_value, variant, initial_cash):
                    forced_exits.update(ticker for ticker, shares in account.positions.items() if shares > 0)
                    stop_until_index = max(stop_until_index, index + max(0, variant.portfolio_stop_cooldown_days))
                    portfolio_stop_active = True
        if variant.position_trailing_stop and account.positions:
            held_tickers = {ticker for ticker, shares in account.positions.items() if shares > 0}
            for ticker in set(position_entry_prices) - held_tickers:
                position_entry_prices.pop(ticker, None)
                position_peak_prices.pop(ticker, None)
            signal_prices = _latest_price_lookup(prices_by_ticker, signal_date, "close", held_tickers)
            for ticker in sorted(held_tickers):
                signal_price = signal_prices.get(ticker)
                if signal_price is None:
                    continue
                position_entry_prices.setdefault(ticker, signal_price)
                position_peak_prices[ticker] = max(position_peak_prices.get(ticker, signal_price), signal_price)
                if _position_stop_triggered(
                    signal_price=signal_price,
                    entry_price=position_entry_prices[ticker],
                    peak_price=position_peak_prices[ticker],
                    variant=variant,
                ):
                    forced_exits.add(ticker)
                    ticker_stop_until_index[ticker] = max(
                        ticker_stop_until_index.get(ticker, -1),
                        index + max(0, variant.position_stop_cooldown_days),
                    )
        if variant.daily_blowoff_exit:
            forced_exits.update(_daily_blowoff_exit_tickers(
                positions=account.positions,
                prices_by_ticker=prices_by_ticker,
                signal_date=signal_date,
                variant=variant,
            ))
        if variant.daily_trend_exit:
            forced_exits.update(_daily_trend_exit_tickers(
                positions=account.positions,
                prices_by_ticker=prices_by_ticker,
                signal_date=signal_date,
                variant=variant,
            ))
        if forced_exits:
            open_prices = _exact_price_lookup(prices_by_ticker, trade_date, "open", forced_exits)
            for ticker in sorted(forced_exits):
                if ticker in open_prices:
                    account.sell(
                        trade_date,
                        ticker,
                        account.shares(ticker),
                        open_prices[ticker],
                        asset_types[ticker],
                        f"{variant.name}_daily_forced_exit",
                    )

        target_weights: dict[str, float] | None = None
        if index == 0 or _is_rebalance_date(trade_dates, index, variant.rebalance_frequency):
            regime = classify_market_regime(market_prices, signal_date, universe_prices=prices_by_ticker)
            exposure = _exposure_for_regime(regime.regime, variant)
            theme_scores = score_themes(prices_by_ticker, members_by_ticker, signal_date)
            exposure = _theme_strength_adjusted_exposure(exposure, theme_scores, variant)
            stock_scores = score_stocks(prices_by_ticker, members_by_ticker, signal_date, variant)
            if portfolio_stop_active:
                target_weights = {}
            else:
                target_weights = target_weights_from_radar_scores(
                    theme_scores=theme_scores,
                    stock_scores=stock_scores,
                    members_by_ticker=members_by_ticker,
                    exposure=exposure,
                    variant=variant,
                    current_tickers={ticker for ticker, shares in account.positions.items() if shares > 0},
                )
            if ticker_stop_until_index:
                target_weights = {
                    ticker: weight
                    for ticker, weight in target_weights.items()
                    if index > ticker_stop_until_index.get(ticker, -1)
                }
            if forced_exits:
                target_weights = {ticker: weight for ticker, weight in target_weights.items() if ticker not in forced_exits}
            tradable_tickers = _tradable_tickers(prices_by_ticker, trade_date)
            target_weights = {ticker: weight for ticker, weight in target_weights.items() if ticker in tradable_tickers}
            theme_rows.extend(_theme_log_rows(trade_date, signal_date, regime.regime, theme_scores))
            stock_rows.extend(_stock_log_rows(trade_date, signal_date, stock_scores, members_by_ticker))
            required_tickers = set(target_weights) | {ticker for ticker, shares in account.positions.items() if shares > 0}
            open_prices = _exact_price_lookup(prices_by_ticker, trade_date, "open", required_tickers)
            close_prices = _exact_price_lookup(prices_by_ticker, trade_date, "close", required_tickers)
            if required_tickers <= set(open_prices) and required_tickers <= set(close_prices):
                current_value = account.value(close_prices)
                target_weights = _apply_rebalance_band(
                    positions=account.positions,
                    close_prices=close_prices,
                    total_value=current_value,
                    target_weights=target_weights,
                    band=variant.rebalance_band,
                )
                account.rebalance(
                    date=trade_date,
                    target_weights=target_weights,
                    open_prices=open_prices,
                    close_prices=close_prices,
                    asset_types=asset_types,
                    reason=f"{variant.name}_{regime.regime}",
                )

        held_tickers = {ticker for ticker, shares in account.positions.items() if shares > 0}
        close_prices = _latest_price_lookup(prices_by_ticker, trade_date, "close", held_tickers)
        total_value = account.value(close_prices)
        equity_rows.append(
            {
                "date": trade_date,
                "total_value": total_value,
                "cash": account.cash,
                "market_exposure": 1 - account.cash / total_value if total_value else 0.0,
                "current_ticker": "|".join(sorted(ticker for ticker, shares in account.positions.items() if shares > 0)) or "cash",
            }
        )
        for ticker, shares in sorted(account.positions.items()):
            if shares <= 0:
                continue
            holding_rows.append(
                {
                    "date": _date_str(trade_date),
                    "ticker": ticker,
                    "theme": members_by_ticker[ticker].theme,
                    "shares": shares,
                    "close": close_prices[ticker],
                    "weight": shares * close_prices[ticker] / total_value if total_value else 0.0,
                }
            )

    equity_curve = pd.DataFrame(equity_rows).set_index("date")
    result = BacktestResult(
        name=name,
        final_value=float(equity_curve["total_value"].iloc[-1]),
        total_return=float(equity_curve["total_value"].iloc[-1] / initial_cash - 1),
        max_drawdown=_max_drawdown(equity_curve["total_value"]),
        trades=account.trades,
        equity_curve=equity_curve,
    )
    return RadarCoreResult(
        result=result,
        theme_log=pd.DataFrame(theme_rows),
        stock_log=pd.DataFrame(stock_rows),
        holdings=pd.DataFrame(holding_rows),
    )


def resolve_variant_for_pool(
    *,
    variant: RadarCoreVariant,
    prices_by_ticker: dict[str, pd.DataFrame],
    members_by_ticker: dict[str, ThemeMember],
    signal_date: pd.Timestamp,
) -> RadarCoreVariant:
    if not variant.use_pool_profile_defaults:
        return variant
    theme_by_ticker = {ticker: member.theme for ticker, member in members_by_ticker.items()}
    params = default_parameters_for_profile(
        infer_pool_profile(prices_by_ticker, signal_date, theme_by_ticker=theme_by_ticker)
    )
    return replace(
        variant,
        min_avg_turnover_twd=params.min_avg_turnover_twd,
        min_stock_score=params.min_stock_score,
        overheated_20d_return=params.overheated_20d_return,
        max_stock_drawdown_20d=params.max_stock_drawdown_20d,
        require_ma60=params.require_ma60,
        stock_score_mode=params.score_mode,
    )


def score_themes(
    prices_by_ticker: dict[str, pd.DataFrame],
    members_by_ticker: dict[str, ThemeMember],
    signal_date: pd.Timestamp,
) -> dict[str, float]:
    theme_values: dict[str, list[dict[str, float]]] = {}
    for ticker, prices in prices_by_ticker.items():
        history = prices.loc[prices.index <= signal_date].dropna(subset=["adj_close"])
        if len(history) < 126:
            continue
        ret20 = _window_return(history["adj_close"], 20)
        ret60 = _window_return(history["adj_close"], 60)
        ret120 = _window_return(history["adj_close"], 120)
        vol20 = float(history["adj_close"].pct_change().dropna().iloc[-20:].std() * (252**0.5))
        theme_values.setdefault(members_by_ticker[ticker].theme, []).append(
            {"ret20": ret20, "ret60": ret60, "ret120": ret120, "vol20": vol20}
        )
    scores: dict[str, float] = {}
    for theme, values in theme_values.items():
        if len(values) < 2:
            continue
        strong_ratio = sum(1 for item in values if item["ret20"] > 0 and item["ret60"] > 0) / len(values)
        avg20 = sum(item["ret20"] for item in values) / len(values)
        avg60 = sum(item["ret60"] for item in values) / len(values)
        avg120 = sum(item["ret120"] for item in values) / len(values)
        avg_vol = sum(item["vol20"] for item in values) / len(values)
        scores[theme] = (0.35 * avg20) + (0.35 * avg60) + (0.15 * avg120) + (0.25 * strong_ratio) - (0.08 * avg_vol)
    return scores


def score_stocks(
    prices_by_ticker: dict[str, pd.DataFrame],
    members_by_ticker: dict[str, ThemeMember],
    signal_date: pd.Timestamp,
    variant: RadarCoreVariant,
) -> dict[str, float]:
    scores: dict[str, float] = {}
    theme_relative_metrics = _theme_relative_metrics(prices_by_ticker, members_by_ticker, signal_date)
    for ticker, prices in prices_by_ticker.items():
        history = prices.loc[prices.index <= signal_date].dropna(subset=["adj_close"])
        if len(history) < 126:
            continue
        close = float(history["adj_close"].iloc[-1])
        ma20 = float(history["adj_close"].iloc[-20:].mean())
        ma60 = float(history["adj_close"].iloc[-60:].mean())
        if close < ma20 or (variant.require_ma60 and close < ma60):
            continue
        avg_turnover = float((history["close"] * history.get("volume", 0)).iloc[-20:].mean())
        if avg_turnover < variant.min_avg_turnover_twd:
            continue
        ret20 = _window_return(history["adj_close"], 20)
        ret60 = _window_return(history["adj_close"], 60)
        ret120 = _window_return(history["adj_close"], 120)
        drawdown20 = close / float(history["adj_close"].iloc[-20:].max()) - 1
        if ret20 > variant.overheated_20d_return or drawdown20 < variant.max_stock_drawdown_20d:
            continue
        if variant.daily_blowoff_exit and _blowoff_risk(prices, signal_date, variant):
            continue
        vol20 = float(history["adj_close"].pct_change().dropna().iloc[-20:].std() * (252**0.5))
        conviction_bonus = 0.03 if members_by_ticker[ticker].conviction.lower() == "high" else 0.0
        score = _stock_score(
            history=history,
            ret20=ret20,
            ret60=ret60,
            ret120=ret120,
            vol20=vol20,
            theme_relative=theme_relative_metrics.get(ticker, {}),
            conviction_bonus=conviction_bonus,
            variant=variant,
        )
        if score >= variant.min_stock_score:
            scores[ticker] = score
    return scores


def _stock_score(
    *,
    history: pd.DataFrame,
    ret20: float,
    ret60: float,
    ret120: float,
    vol20: float,
    theme_relative: dict[str, float],
    conviction_bonus: float,
    variant: RadarCoreVariant,
) -> float:
    if variant.stock_score_mode == "acceleration":
        ret5 = _window_return(history["adj_close"], 5)
        turnover = history["close"] * history.get("volume", 0).fillna(0)
        avg5 = float(turnover.tail(5).mean())
        avg60 = float(turnover.tail(60).mean())
        turnover_surge = min(max(avg5 / avg60 - 1, -1.0), 2.0) if avg60 > 0 else 0.0
        acceleration = ret20 - ret60
        return (
            (0.30 * ret5)
            + (0.40 * ret20)
            + (0.18 * ret60)
            + (0.18 * turnover_surge)
            + (0.12 * acceleration)
            - (0.08 * vol20)
            + conviction_bonus
        )
    if variant.stock_score_mode in {"short_momentum", "trend_momentum", "risk_adjusted", "raw_momentum"}:
        return universal_stock_score(
            ret20=ret20,
            ret60=ret60,
            ret120=ret120,
            vol20=vol20,
            mode=variant.stock_score_mode,
            conviction_bonus=conviction_bonus,
        )
    if variant.stock_score_mode == "risk_adjusted_theme_relative":
        return _risk_adjusted_theme_relative_score(
            ret20=ret20,
            ret60=ret60,
            ret120=ret120,
            vol20=vol20,
            theme_relative=theme_relative,
            conviction_bonus=conviction_bonus,
            relative_weight=0.12,
            turnover_weight=0.00,
        )
    if variant.stock_score_mode == "risk_adjusted_capital_flow":
        return _risk_adjusted_theme_relative_score(
            ret20=ret20,
            ret60=ret60,
            ret120=ret120,
            vol20=vol20,
            theme_relative=theme_relative,
            conviction_bonus=conviction_bonus,
            relative_weight=0.10,
            turnover_weight=0.06,
        )
    if variant.stock_score_mode == "risk_adjusted_theme_leader":
        return _risk_adjusted_theme_relative_score(
            ret20=ret20,
            ret60=ret60,
            ret120=ret120,
            vol20=vol20,
            theme_relative=theme_relative,
            conviction_bonus=conviction_bonus,
            relative_weight=0.18,
            turnover_weight=0.03,
        )
    return (0.35 * ret20) + (0.35 * ret60) + (0.15 * ret120) - (0.10 * vol20) + conviction_bonus


def _risk_adjusted_theme_relative_score(
    *,
    ret20: float,
    ret60: float,
    ret120: float,
    vol20: float,
    theme_relative: dict[str, float],
    conviction_bonus: float,
    relative_weight: float,
    turnover_weight: float,
) -> float:
    base = (0.30 * ret20) + (0.40 * ret60) + (0.20 * ret120) - (0.22 * vol20) + conviction_bonus
    relative_strength = (0.55 * theme_relative.get("ret20_relative", 0.0)) + (
        0.45 * theme_relative.get("ret60_relative", 0.0)
    )
    turnover_lead = theme_relative.get("turnover_relative", 0.0)
    return base + (relative_weight * relative_strength) + (turnover_weight * turnover_lead)


def _theme_relative_metrics(
    prices_by_ticker: dict[str, pd.DataFrame],
    members_by_ticker: dict[str, ThemeMember],
    signal_date: pd.Timestamp,
) -> dict[str, dict[str, float]]:
    theme_values: dict[str, list[dict[str, float | str]]] = {}
    for ticker, prices in prices_by_ticker.items():
        member = members_by_ticker.get(ticker)
        if member is None:
            continue
        history = prices.loc[prices.index <= signal_date].dropna(subset=["adj_close", "close"])
        if len(history) < 126:
            continue
        turnover = history["close"] * history.get("volume", 0).fillna(0)
        theme_values.setdefault(member.theme, []).append(
            {
                "ticker": ticker,
                "ret20": _window_return(history["adj_close"], 20),
                "ret60": _window_return(history["adj_close"], 60),
                "turnover20": float(turnover.tail(20).mean()),
            }
        )

    metrics: dict[str, dict[str, float]] = {}
    for values in theme_values.values():
        if not values:
            continue
        avg_ret20 = sum(float(item["ret20"]) for item in values) / len(values)
        avg_ret60 = sum(float(item["ret60"]) for item in values) / len(values)
        avg_turnover = sum(float(item["turnover20"]) for item in values) / len(values)
        for item in values:
            turnover_relative = 0.0
            if avg_turnover > 0:
                turnover_relative = min(max(float(item["turnover20"]) / avg_turnover - 1.0, -1.0), 2.0)
            metrics[str(item["ticker"])] = {
                "ret20_relative": float(item["ret20"]) - avg_ret20,
                "ret60_relative": float(item["ret60"]) - avg_ret60,
                "turnover_relative": turnover_relative,
            }
    return metrics


def _daily_blowoff_exit_tickers(
    *,
    positions: dict[str, int],
    prices_by_ticker: dict[str, pd.DataFrame],
    signal_date: pd.Timestamp,
    variant: RadarCoreVariant,
) -> set[str]:
    return {
        ticker
        for ticker, shares in positions.items()
        if shares > 0 and _blowoff_risk(prices_by_ticker.get(ticker), signal_date, variant)
    }


def _blowoff_risk(
    prices: pd.DataFrame | None,
    signal_date: pd.Timestamp,
    variant: RadarCoreVariant,
) -> bool:
    if prices is None or prices.empty:
        return False
    history = prices.loc[prices.index <= signal_date].dropna(subset=["adj_close", "close"])
    if len(history) < 61:
        return False
    close = float(history["adj_close"].iloc[-1])
    ret20 = close / float(history["adj_close"].iloc[-20]) - 1
    drawdown10 = close / float(history["adj_close"].tail(10).max()) - 1
    if "volume" in history.columns:
        turnover = history["close"] * history["volume"].fillna(0)
        avg5 = float(turnover.tail(5).mean())
        avg60 = float(turnover.tail(60).mean())
        volume_ratio = avg5 / avg60 if avg60 > 0 else 0.0
    else:
        volume_ratio = 0.0
    return (
        ret20 >= variant.blowoff_min_runup_20d
        and drawdown10 <= variant.blowoff_drawdown_10d
        and volume_ratio >= variant.blowoff_volume_ratio_5d_over_60d
    )


def _daily_trend_exit_tickers(
    *,
    positions: dict[str, int],
    prices_by_ticker: dict[str, pd.DataFrame],
    signal_date: pd.Timestamp,
    variant: RadarCoreVariant,
) -> set[str]:
    return {
        ticker
        for ticker, shares in positions.items()
        if shares > 0 and _trend_exit_risk(prices_by_ticker.get(ticker), signal_date, variant)
    }


def _trend_exit_risk(
    prices: pd.DataFrame | None,
    signal_date: pd.Timestamp,
    variant: RadarCoreVariant,
) -> bool:
    if prices is None or prices.empty:
        return False
    window = max(2, variant.trend_exit_ma_window)
    history = prices.loc[prices.index <= signal_date].dropna(subset=["adj_close"])
    if len(history) < max(20, window):
        return False
    close = float(history["adj_close"].iloc[-1])
    ma = float(history["adj_close"].tail(window).mean())
    drawdown20 = close / float(history["adj_close"].tail(20).max()) - 1
    return close < ma or drawdown20 <= variant.trend_exit_drawdown_20d


def _portfolio_stop_triggered(
    signal_value: float,
    peak_signal_value: float,
    variant: RadarCoreVariant,
    initial_value: float = 0.0,
) -> bool:
    if peak_signal_value <= 0:
        return False
    if variant.portfolio_stop_min_gain > 0 and initial_value > 0:
        if peak_signal_value / initial_value - 1 < variant.portfolio_stop_min_gain:
            return False
    return signal_value / peak_signal_value - 1 <= variant.portfolio_stop_drawdown


def _position_stop_triggered(
    *,
    signal_price: float,
    entry_price: float,
    peak_price: float,
    variant: RadarCoreVariant,
) -> bool:
    if signal_price <= 0 or entry_price <= 0 or peak_price <= 0:
        return False
    if peak_price / entry_price - 1 < variant.position_stop_min_runup:
        return False
    return signal_price / peak_price - 1 <= variant.position_stop_drawdown


def target_weights_from_radar_scores(
    *,
    theme_scores: dict[str, float],
    stock_scores: dict[str, float],
    members_by_ticker: dict[str, ThemeMember],
    exposure: float,
    variant: RadarCoreVariant,
    current_tickers: set[str] | None = None,
) -> dict[str, float]:
    selected_themes = [
        theme
        for theme, score in sorted(theme_scores.items(), key=lambda item: item[1], reverse=True)
        if score >= variant.min_theme_score
    ][: variant.top_theme_count]
    if variant.hold_existing_score_ratio > 0 and current_tickers:
        selected = _existing_leaders_to_hold(
            selected_themes=selected_themes,
            stock_scores=stock_scores,
            members_by_ticker=members_by_ticker,
            variant=variant,
            current_tickers=current_tickers,
        )
        if selected:
            per_stock = min(variant.max_single_weight, exposure / len(selected))
            return {ticker: per_stock for ticker in selected}

    selected: list[str] = []
    if variant.selection_mode == "global_leaders":
        max_total = variant.max_total_stocks or max(1, variant.top_theme_count * variant.max_stocks_per_theme)
        selected = [
            ticker
            for ticker, score in sorted(stock_scores.items(), key=lambda item: item[1], reverse=True)
            if members_by_ticker[ticker].theme in selected_themes and score >= variant.min_stock_score
        ][:max_total]
    else:
        for theme in selected_themes:
            theme_candidates = [
                ticker
                for ticker, score in sorted(stock_scores.items(), key=lambda item: item[1], reverse=True)
                if members_by_ticker[ticker].theme == theme and score >= variant.min_stock_score
            ]
            selected.extend(theme_candidates[: variant.max_stocks_per_theme])
    if not selected or exposure <= 0:
        return {}
    per_stock = min(variant.max_single_weight, exposure / len(selected))
    return {ticker: per_stock for ticker in selected}


def _existing_leaders_to_hold(
    *,
    selected_themes: list[str],
    stock_scores: dict[str, float],
    members_by_ticker: dict[str, ThemeMember],
    variant: RadarCoreVariant,
    current_tickers: set[str],
) -> list[str]:
    candidates = [
        ticker
        for ticker, score in stock_scores.items()
        if members_by_ticker[ticker].theme in selected_themes and score >= variant.min_stock_score
    ]
    if not candidates:
        return []
    best_score = max(stock_scores[ticker] for ticker in candidates)
    hold_candidates = [
        ticker
        for ticker in current_tickers
        if ticker in candidates and stock_scores[ticker] >= best_score * variant.hold_existing_score_ratio
    ]
    hold_limit = variant.max_total_stocks or max(1, variant.top_theme_count * variant.max_stocks_per_theme)
    return sorted(hold_candidates, key=lambda ticker: stock_scores[ticker], reverse=True)[:hold_limit]


def _exposure_for_regime(regime: str, variant: RadarCoreVariant) -> float:
    return {
        "strong_bull": variant.strong_bull_exposure,
        "recovery_bull": variant.recovery_bull_exposure,
        "range_bound": variant.range_bound_exposure,
        "correction_bear": variant.correction_bear_exposure,
        "systemic_bear": variant.systemic_bear_exposure,
    }.get(regime, 0.0)


def _theme_strength_adjusted_exposure(
    exposure: float,
    theme_scores: dict[str, float],
    variant: RadarCoreVariant,
) -> float:
    if exposure <= 0 or variant.weak_theme_full_score <= 0 or variant.weak_theme_exposure_multiplier >= 1:
        return exposure
    eligible_scores = [score for score in theme_scores.values() if score >= variant.min_theme_score]
    if not eligible_scores:
        return exposure
    if max(eligible_scores) < variant.weak_theme_full_score:
        return exposure * max(0.0, variant.weak_theme_exposure_multiplier)
    return exposure


def _theme_log_rows(trade_date, signal_date, regime: str, scores: dict[str, float]) -> list[dict]:
    return [
        {
            "date": _date_str(trade_date),
            "signal_date": _date_str(signal_date),
            "market_regime": regime,
            "rank": rank,
            "theme": theme,
            "score": score,
        }
        for rank, (theme, score) in enumerate(sorted(scores.items(), key=lambda item: item[1], reverse=True), start=1)
    ]


def _stock_log_rows(trade_date, signal_date, scores: dict[str, float], members_by_ticker: dict[str, ThemeMember]) -> list[dict]:
    return [
        {
            "date": _date_str(trade_date),
            "signal_date": _date_str(signal_date),
            "rank": rank,
            "ticker": ticker,
            "theme": members_by_ticker[ticker].theme,
            "name": members_by_ticker[ticker].name,
            "score": score,
        }
        for rank, (ticker, score) in enumerate(sorted(scores.items(), key=lambda item: item[1], reverse=True), start=1)
    ]


def _market_trade_dates(market_prices: pd.DataFrame, start_date: str, end_date: str) -> list[pd.Timestamp]:
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    return list(market_prices.loc[(market_prices.index >= start) & (market_prices.index <= end)].index)


def _previous_market_signal_date(market_prices: pd.DataFrame, trade_date: pd.Timestamp) -> pd.Timestamp:
    available = market_prices.index[market_prices.index < trade_date]
    if len(available) == 0:
        raise ValueError(f"No market signal date available before {trade_date.date()}")
    return available.max()


def _tradable_tickers(prices_by_ticker: dict[str, pd.DataFrame], trade_date: pd.Timestamp) -> set[str]:
    return {ticker for ticker, prices in prices_by_ticker.items() if trade_date in prices.index}


def _exact_price_lookup(
    prices_by_ticker: dict[str, pd.DataFrame],
    trade_date: pd.Timestamp,
    field: str,
    tickers: set[str],
) -> dict[str, float]:
    values: dict[str, float] = {}
    for ticker in tickers:
        prices = prices_by_ticker.get(ticker)
        if prices is None or trade_date not in prices.index:
            continue
        value = prices.loc[trade_date, field]
        if pd.notna(value):
            values[ticker] = float(value)
    return values


def _latest_price_lookup(
    prices_by_ticker: dict[str, pd.DataFrame],
    trade_date: pd.Timestamp,
    field: str,
    tickers: set[str],
) -> dict[str, float]:
    values: dict[str, float] = {}
    for ticker in tickers:
        prices = prices_by_ticker.get(ticker)
        if prices is None:
            continue
        history = prices.loc[prices.index <= trade_date, field].dropna()
        if not history.empty:
            values[ticker] = float(history.iloc[-1])
    return values


def _apply_rebalance_band(
    *,
    positions: dict[str, int],
    close_prices: dict[str, float],
    total_value: float,
    target_weights: dict[str, float],
    band: float,
) -> dict[str, float]:
    if total_value <= 0 or band <= 0:
        return target_weights
    adjusted = dict(target_weights)
    for ticker, target_weight in target_weights.items():
        shares = positions.get(ticker, 0)
        if shares <= 0:
            continue
        current_price = close_prices.get(ticker)
        if current_price is None:
            continue
        current_weight = shares * current_price / total_value
        if abs(current_weight - target_weight) <= band:
            adjusted[ticker] = current_weight
    return adjusted


def _is_rebalance_date(trade_dates: list[pd.Timestamp], index: int, frequency: str) -> bool:
    if index == 0 or frequency == "daily":
        return True
    if frequency == "weekly":
        current = trade_dates[index].isocalendar()
        previous = trade_dates[index - 1].isocalendar()
        return (current.year, current.week) != (previous.year, previous.week)
    if frequency.startswith("weekly_"):
        target_weekday = {
            "weekly_mon": 0,
            "weekly_tue": 1,
            "weekly_wed": 2,
            "weekly_thu": 3,
            "weekly_fri": 4,
        }.get(frequency)
        if target_weekday is None:
            raise ValueError(f"Unsupported rebalance frequency: {frequency}")
        current = trade_dates[index]
        previous = trade_dates[index - 1]
        if current.weekday() < target_weekday:
            return False
        current_week = current.isocalendar()
        previous_week = previous.isocalendar()
        if (current_week.year, current_week.week) != (previous_week.year, previous_week.week):
            return True
        return previous.weekday() < target_weekday
    if frequency == "biweekly":
        current = trade_dates[index].isocalendar()
        previous = trade_dates[index - 1].isocalendar()
        current_key = current.year * 53 + current.week
        previous_key = previous.year * 53 + previous.week
        return current_key // 2 != previous_key // 2
    if frequency == "monthly":
        return trade_dates[index].month != trade_dates[index - 1].month
    raise ValueError(f"Unsupported rebalance frequency: {frequency}")


def _window_return(series: pd.Series, window: int) -> float:
    return window_return(series, window)
