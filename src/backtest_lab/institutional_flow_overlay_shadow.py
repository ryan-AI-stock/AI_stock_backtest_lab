from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from backtest_lab.config import load_config
from backtest_lab.data import download_yfinance_prices, split_adjusted_dividends
from backtest_lab.regime_aware_backtest import PERIODS
from backtest_lab.regime_mode_switch import frozen_cycle_proven_top1_v1_variant, simulate_regime_mode_switch
from backtest_lab.regime_mode_switch_backtest import _load_sufficient_cache_prices
from backtest_lab.simulation import BacktestResult, _max_drawdown, simulate_buy_and_hold


DEFAULT_FLOW_SOURCE = (
    "C:/Users/zergv/Documents/Codex/2026-05-23/ai-stock-rotation-radar-https-docs/"
    "data/formal_sources/chip_flow_overlay_2021_2023/institutional_flows_daily_2021_2023.csv"
)
DEFAULT_MARGIN_SOURCE = (
    "C:/Users/zergv/Documents/Codex/2026-05-23/ai-stock-rotation-radar-https-docs/"
    "data/formal_sources/chip_flow_overlay_2021_2023/margin_short_daily_2021_2023.csv"
)
DEFAULT_DAY_TRADING_SOURCE = (
    "C:/Users/zergv/Documents/Codex/2026-05-23/ai-stock-rotation-radar-https-docs/"
    "data/formal_sources/chip_flow_overlay_2021_2023/day_trading_daily_2021_2023.csv"
)
DEFAULT_PERIODS = "bear_2022,year_2023"
ETF_TICKERS = {"0050.TW", "00631L.TW"}


@dataclass(frozen=True)
class InstitutionalOverlayRule:
    name: str
    foreign_sell_days: int
    trust_sell_days: int
    exposure_cap: float
    require_negative_total_shares: bool = False
    stock_only: bool = True


@dataclass(frozen=True)
class ChipFlowOverlayRule:
    name: str
    exposure_cap: float
    institutional_foreign_sell_days: int | None = None
    institutional_trust_sell_days: int | None = None
    use_margin_overheat: bool = False
    use_short_pressure: bool = False
    use_day_trading_overheat: bool = False
    min_day_trading_ratio: float | None = None
    require_two_signals: bool = False
    price_confirmation: str | None = None
    stock_only: bool = True


def default_overlay_rules() -> tuple[InstitutionalOverlayRule, ...]:
    return (
        InstitutionalOverlayRule(
            name="foreign3_or_trust2_reduce50",
            foreign_sell_days=3,
            trust_sell_days=2,
            exposure_cap=0.50,
        ),
        InstitutionalOverlayRule(
            name="foreign5_or_trust3_reduce50",
            foreign_sell_days=5,
            trust_sell_days=3,
            exposure_cap=0.50,
        ),
        InstitutionalOverlayRule(
            name="foreign3_or_trust2_cash",
            foreign_sell_days=3,
            trust_sell_days=2,
            exposure_cap=0.0,
        ),
        InstitutionalOverlayRule(
            name="foreign3_or_trust2_negative_total_reduce50",
            foreign_sell_days=3,
            trust_sell_days=2,
            exposure_cap=0.50,
            require_negative_total_shares=True,
        ),
    )


def default_chip_flow_rules() -> tuple[ChipFlowOverlayRule, ...]:
    return (
        ChipFlowOverlayRule(
            name="chip_any_inst3_margin_day_reduce50",
            exposure_cap=0.50,
            institutional_foreign_sell_days=3,
            institutional_trust_sell_days=2,
            use_margin_overheat=True,
            use_short_pressure=True,
            use_day_trading_overheat=True,
        ),
        ChipFlowOverlayRule(
            name="chip_two_signal_reduce50",
            exposure_cap=0.50,
            institutional_foreign_sell_days=3,
            institutional_trust_sell_days=2,
            use_margin_overheat=True,
            use_short_pressure=True,
            use_day_trading_overheat=True,
            require_two_signals=True,
        ),
        ChipFlowOverlayRule(
            name="chip_margin_or_short_reduce50",
            exposure_cap=0.50,
            use_margin_overheat=True,
            use_short_pressure=True,
        ),
        ChipFlowOverlayRule(
            name="chip_day_ratio35_reduce50",
            exposure_cap=0.50,
            min_day_trading_ratio=35.0,
        ),
        ChipFlowOverlayRule(
            name="chip_two_signal_cash",
            exposure_cap=0.0,
            institutional_foreign_sell_days=3,
            institutional_trust_sell_days=2,
            use_margin_overheat=True,
            use_short_pressure=True,
            use_day_trading_overheat=True,
            require_two_signals=True,
        ),
        ChipFlowOverlayRule(
            name="chip_two_signal_price_ma10_reduce50",
            exposure_cap=0.50,
            institutional_foreign_sell_days=3,
            institutional_trust_sell_days=2,
            use_margin_overheat=True,
            use_short_pressure=True,
            use_day_trading_overheat=True,
            require_two_signals=True,
            price_confirmation="below_ma10",
        ),
        ChipFlowOverlayRule(
            name="chip_two_signal_price_ma20_reduce50",
            exposure_cap=0.50,
            institutional_foreign_sell_days=3,
            institutional_trust_sell_days=2,
            use_margin_overheat=True,
            use_short_pressure=True,
            use_day_trading_overheat=True,
            require_two_signals=True,
            price_confirmation="below_ma20",
        ),
        ChipFlowOverlayRule(
            name="chip_two_signal_price_ret5neg_reduce50",
            exposure_cap=0.50,
            institutional_foreign_sell_days=3,
            institutional_trust_sell_days=2,
            use_margin_overheat=True,
            use_short_pressure=True,
            use_day_trading_overheat=True,
            require_two_signals=True,
            price_confirmation="ret5_negative",
        ),
        ChipFlowOverlayRule(
            name="chip_two_signal_price_dd10_reduce50",
            exposure_cap=0.50,
            institutional_foreign_sell_days=3,
            institutional_trust_sell_days=2,
            use_margin_overheat=True,
            use_short_pressure=True,
            use_day_trading_overheat=True,
            require_two_signals=True,
            price_confirmation="drawdown10_over8",
        ),
        ChipFlowOverlayRule(
            name="chip_two_signal_price_dd10_reduce75",
            exposure_cap=0.75,
            institutional_foreign_sell_days=3,
            institutional_trust_sell_days=2,
            use_margin_overheat=True,
            use_short_pressure=True,
            use_day_trading_overheat=True,
            require_two_signals=True,
            price_confirmation="drawdown10_over8",
        ),
        ChipFlowOverlayRule(
            name="chip_two_signal_price_dd10_reduce25",
            exposure_cap=0.25,
            institutional_foreign_sell_days=3,
            institutional_trust_sell_days=2,
            use_margin_overheat=True,
            use_short_pressure=True,
            use_day_trading_overheat=True,
            require_two_signals=True,
            price_confirmation="drawdown10_over8",
        ),
        ChipFlowOverlayRule(
            name="chip_two_signal_price_dd10_cash",
            exposure_cap=0.0,
            institutional_foreign_sell_days=3,
            institutional_trust_sell_days=2,
            use_margin_overheat=True,
            use_short_pressure=True,
            use_day_trading_overheat=True,
            require_two_signals=True,
            price_confirmation="drawdown10_over8",
        ),
    )


def load_institutional_flows(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"symbol": str, "ticker": str}, parse_dates=["date"])
    required = {
        "date",
        "ticker",
        "foreign_net_buy_shares",
        "investment_trust_net_buy_shares",
        "dealer_net_buy_shares",
        "foreign_consecutive_sell_days",
        "trust_consecutive_sell_days",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Institutional flow source missing columns: {', '.join(missing)}")
    for column in required - {"date", "ticker"}:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    frame["date"] = frame["date"].dt.normalize()
    return frame.sort_values(["date", "ticker"]).reset_index(drop=True)


def load_margin_short(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"symbol": str, "ticker": str}, parse_dates=["date"])
    required = {
        "date",
        "ticker",
        "margin_balance_5d_change_pct",
        "margin_balance_20d_change_pct",
        "short_balance_5d_change_pct",
        "short_balance_20d_change_pct",
        "margin_overheat_flag",
        "short_lending_pressure_flag",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Margin/short source missing columns: {', '.join(missing)}")
    for column in required - {"date", "ticker", "margin_overheat_flag", "short_lending_pressure_flag"}:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["date"] = frame["date"].dt.normalize()
    return frame.sort_values(["date", "ticker"]).reset_index(drop=True)


def load_day_trading(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"symbol": str, "ticker": str}, parse_dates=["date"])
    required = {
        "date",
        "ticker",
        "day_trading_volume_ratio",
        "day_trading_ratio_5d_avg",
        "day_trading_ratio_20d_avg",
        "day_trading_overheat_flag",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Day-trading source missing columns: {', '.join(missing)}")
    for column in required - {"date", "ticker", "day_trading_overheat_flag"}:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["date"] = frame["date"].dt.normalize()
    return frame.sort_values(["date", "ticker"]).reset_index(drop=True)


def apply_institutional_overlay(
    *,
    baseline: BacktestResult,
    flow_frame: pd.DataFrame,
    rule: InstitutionalOverlayRule,
    initial_cash: float,
) -> pd.DataFrame:
    """Post-process baseline daily returns with a same-day-safe institutional risk cap.

    This is a shadow approximation. It scales the frozen model's realized daily
    return when the previous available institutional-flow record flags risk for
    the currently held stock. It does not replace the formal execution engine.
    """

    flow_lookup = {
        (pd.Timestamp(row.date), str(row.ticker)): row
        for row in flow_frame.itertuples(index=False)
    }
    flow_dates = sorted(pd.Timestamp(date) for date in flow_frame["date"].drop_duplicates())
    baseline_curve = baseline.equity_curve.copy()
    baseline_returns = baseline_curve["total_value"].pct_change().fillna(
        baseline_curve["total_value"].iloc[0] / initial_cash - 1
    )

    value = float(initial_cash)
    rows: list[dict] = []
    for date, baseline_row in baseline_curve.iterrows():
        trade_date = pd.Timestamp(date).normalize()
        ticker = str(baseline_row.get("current_ticker", "cash"))
        baseline_exposure = float(baseline_row.get("current_exposure", 0.0))
        signal_date = previous_flow_date(flow_dates, trade_date)
        risk_flag, risk_reason = institutional_risk_flag(
            flow_lookup.get((signal_date, ticker)) if signal_date is not None else None,
            ticker=ticker,
            rule=rule,
        )
        overlay_exposure = min(baseline_exposure, rule.exposure_cap) if risk_flag else baseline_exposure
        exposure_ratio = overlay_exposure / baseline_exposure if baseline_exposure > 0 else 0.0
        baseline_return = float(baseline_returns.loc[date])
        overlay_return = baseline_return * exposure_ratio if risk_flag else baseline_return
        value *= 1 + overlay_return
        rows.append(
            {
                "date": trade_date,
                "total_value": value,
                "baseline_total_value": float(baseline_row["total_value"]),
                "daily_return_pct": overlay_return * 100,
                "baseline_daily_return_pct": float(baseline_returns.loc[date]) * 100,
                "current_ticker": ticker,
                "baseline_exposure": baseline_exposure,
                "overlay_exposure": overlay_exposure,
                "risk_flag": risk_flag,
                "risk_reason": risk_reason,
                "flow_signal_date": signal_date.strftime("%Y-%m-%d") if signal_date is not None else "",
                "regime": baseline_row.get("regime", ""),
                "mode": baseline_row.get("mode", ""),
            }
        )
    return pd.DataFrame(rows).set_index("date")


def apply_chip_flow_overlay(
    *,
    baseline: BacktestResult,
    institutional_frame: pd.DataFrame,
    margin_frame: pd.DataFrame,
    day_trading_frame: pd.DataFrame,
    prices_by_ticker: dict[str, pd.DataFrame],
    rule: ChipFlowOverlayRule,
    initial_cash: float,
) -> pd.DataFrame:
    institutional_lookup, institutional_dates = _lookup_and_dates(institutional_frame)
    margin_lookup, margin_dates = _lookup_and_dates(margin_frame)
    day_lookup, day_dates = _lookup_and_dates(day_trading_frame)
    baseline_curve = baseline.equity_curve.copy()
    baseline_returns = baseline_curve["total_value"].pct_change().fillna(
        baseline_curve["total_value"].iloc[0] / initial_cash - 1
    )

    value = float(initial_cash)
    rows: list[dict] = []
    for date, baseline_row in baseline_curve.iterrows():
        trade_date = pd.Timestamp(date).normalize()
        ticker = str(baseline_row.get("current_ticker", "cash"))
        baseline_exposure = float(baseline_row.get("current_exposure", 0.0))
        institutional_signal_date = previous_flow_date(institutional_dates, trade_date)
        margin_signal_date = previous_flow_date(margin_dates, trade_date)
        day_signal_date = previous_flow_date(day_dates, trade_date)
        risk_flag, risk_reason = chip_flow_risk_flag(
            institutional_lookup.get((institutional_signal_date, ticker)) if institutional_signal_date else None,
            margin_lookup.get((margin_signal_date, ticker)) if margin_signal_date else None,
            day_lookup.get((day_signal_date, ticker)) if day_signal_date else None,
            ticker=ticker,
            rule=rule,
        )
        price_signal_date = _previous_price_date(prices_by_ticker.get(ticker), trade_date)
        price_confirmed, price_reason = price_confirmation_flag(
            prices_by_ticker.get(ticker),
            price_signal_date,
            rule.price_confirmation,
        )
        if risk_flag and rule.price_confirmation and not price_confirmed:
            risk_flag = False
            risk_reason = ""
        elif risk_flag and price_reason:
            risk_reason = f"{risk_reason}+{price_reason}"
        overlay_exposure = min(baseline_exposure, rule.exposure_cap) if risk_flag else baseline_exposure
        exposure_ratio = overlay_exposure / baseline_exposure if baseline_exposure > 0 else 0.0
        baseline_return = float(baseline_returns.loc[date])
        overlay_return = baseline_return * exposure_ratio if risk_flag else baseline_return
        value *= 1 + overlay_return
        rows.append(
            {
                "date": trade_date,
                "total_value": value,
                "baseline_total_value": float(baseline_row["total_value"]),
                "daily_return_pct": overlay_return * 100,
                "baseline_daily_return_pct": float(baseline_returns.loc[date]) * 100,
                "current_ticker": ticker,
                "baseline_exposure": baseline_exposure,
                "overlay_exposure": overlay_exposure,
                "risk_flag": risk_flag,
                "risk_reason": risk_reason,
                "flow_signal_date": institutional_signal_date.strftime("%Y-%m-%d") if institutional_signal_date else "",
                "margin_signal_date": margin_signal_date.strftime("%Y-%m-%d") if margin_signal_date else "",
                "day_trading_signal_date": day_signal_date.strftime("%Y-%m-%d") if day_signal_date else "",
                "price_signal_date": price_signal_date.strftime("%Y-%m-%d") if price_signal_date is not None else "",
                "regime": baseline_row.get("regime", ""),
                "mode": baseline_row.get("mode", ""),
            }
        )
    return pd.DataFrame(rows).set_index("date")


def _lookup_and_dates(frame: pd.DataFrame) -> tuple[dict[tuple[pd.Timestamp, str], object], list[pd.Timestamp]]:
    lookup = {(pd.Timestamp(row.date), str(row.ticker)): row for row in frame.itertuples(index=False)}
    dates = sorted(pd.Timestamp(date) for date in frame["date"].drop_duplicates())
    return lookup, dates


def _previous_price_date(prices: pd.DataFrame | None, trade_date: pd.Timestamp) -> pd.Timestamp | None:
    if prices is None or prices.empty:
        return None
    prior = prices.index[prices.index < trade_date]
    return pd.Timestamp(prior[-1]) if len(prior) else None


def previous_flow_date(flow_dates: list[pd.Timestamp], trade_date: pd.Timestamp) -> pd.Timestamp | None:
    prior = [date for date in flow_dates if date < trade_date]
    return prior[-1] if prior else None


def institutional_risk_flag(
    flow_row: object | None,
    *,
    ticker: str,
    rule: InstitutionalOverlayRule,
) -> tuple[bool, str]:
    if ticker == "cash" or (rule.stock_only and ticker in ETF_TICKERS):
        return False, ""
    if flow_row is None:
        return False, "missing_flow"
    foreign_sell_days = int(getattr(flow_row, "foreign_consecutive_sell_days", 0) or 0)
    trust_sell_days = int(getattr(flow_row, "trust_consecutive_sell_days", 0) or 0)
    total_net_shares = (
        float(getattr(flow_row, "foreign_net_buy_shares", 0) or 0)
        + float(getattr(flow_row, "investment_trust_net_buy_shares", 0) or 0)
        + float(getattr(flow_row, "dealer_net_buy_shares", 0) or 0)
    )
    streak_risk = foreign_sell_days >= rule.foreign_sell_days or trust_sell_days >= rule.trust_sell_days
    if rule.require_negative_total_shares and total_net_shares >= 0:
        return False, ""
    if not streak_risk:
        return False, ""
    reasons = []
    if foreign_sell_days >= rule.foreign_sell_days:
        reasons.append(f"foreign_sell_{foreign_sell_days}d")
    if trust_sell_days >= rule.trust_sell_days:
        reasons.append(f"trust_sell_{trust_sell_days}d")
    if total_net_shares < 0:
        reasons.append("total_net_sell_shares")
    return True, "+".join(reasons)


def chip_flow_risk_flag(
    institutional_row: object | None,
    margin_row: object | None,
    day_trading_row: object | None,
    *,
    ticker: str,
    rule: ChipFlowOverlayRule,
) -> tuple[bool, str]:
    if ticker == "cash" or (rule.stock_only and ticker in ETF_TICKERS):
        return False, ""
    signals: list[str] = []
    if rule.institutional_foreign_sell_days is not None and rule.institutional_trust_sell_days is not None:
        flagged, reason = institutional_risk_flag(
            institutional_row,
            ticker=ticker,
            rule=InstitutionalOverlayRule(
                name=rule.name,
                foreign_sell_days=rule.institutional_foreign_sell_days,
                trust_sell_days=rule.institutional_trust_sell_days,
                exposure_cap=rule.exposure_cap,
                stock_only=rule.stock_only,
            ),
        )
        if flagged:
            signals.append(f"institutional:{reason}")
    if margin_row is not None:
        if rule.use_margin_overheat and _bool_field(getattr(margin_row, "margin_overheat_flag", False)):
            signals.append("margin_overheat")
        if rule.use_short_pressure and _bool_field(getattr(margin_row, "short_lending_pressure_flag", False)):
            signals.append("short_pressure")
    if day_trading_row is not None:
        day_ratio = _float_or_none(getattr(day_trading_row, "day_trading_volume_ratio", None))
        if rule.use_day_trading_overheat and _bool_field(getattr(day_trading_row, "day_trading_overheat_flag", False)):
            signals.append("day_trading_overheat")
        if rule.min_day_trading_ratio is not None and day_ratio is not None and day_ratio >= rule.min_day_trading_ratio:
            signals.append(f"day_ratio_{day_ratio:.2f}")
    if rule.require_two_signals:
        return (len(signals) >= 2, "+".join(signals) if len(signals) >= 2 else "")
    return (bool(signals), "+".join(signals))


def price_confirmation_flag(
    prices: pd.DataFrame | None,
    signal_date: pd.Timestamp | None,
    confirmation: str | None,
) -> tuple[bool, str]:
    if confirmation is None:
        return True, ""
    if prices is None or prices.empty or signal_date is None or signal_date not in prices.index:
        return False, "missing_price_confirmation"
    history = prices.loc[prices.index <= signal_date]
    close = float(history["close"].iloc[-1])
    if confirmation == "below_ma10":
        if len(history) < 10:
            return False, "price_history_lt10"
        ma10 = float(history["close"].tail(10).mean())
        return close < ma10, "price_below_ma10" if close < ma10 else ""
    if confirmation == "below_ma20":
        if len(history) < 20:
            return False, "price_history_lt20"
        ma20 = float(history["close"].tail(20).mean())
        return close < ma20, "price_below_ma20" if close < ma20 else ""
    if confirmation == "ret5_negative":
        if len(history) < 6:
            return False, "price_history_lt6"
        ret5 = close / float(history["close"].iloc[-6]) - 1
        return ret5 < 0, "price_ret5_negative" if ret5 < 0 else ""
    if confirmation == "drawdown10_over8":
        if len(history) < 10:
            return False, "price_history_lt10"
        drawdown = close / float(history["close"].tail(10).max()) - 1
        return drawdown <= -0.08, "price_dd10_over8" if drawdown <= -0.08 else ""
    raise ValueError(f"Unsupported price confirmation: {confirmation}")


def _bool_field(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _float_or_none(value: object) -> float | None:
    number = pd.to_numeric(value, errors="coerce")
    if pd.isna(number):
        return None
    return float(number)


def run_overlay_shadow(
    *,
    config_path: str,
    group_id: str,
    cache_dir: str,
    output_dir: str,
    flow_source: str,
    margin_source: str | None,
    day_trading_source: str | None,
    period_ids: list[str],
    market_proxy: str,
) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "current_step.txt").write_text("loading_inputs\n", encoding="utf-8")

    config = load_config(config_path)
    group = next(group for group in config.groups if group.group_id == group_id)
    asset_types = {asset.ticker: asset.asset_type for asset in group.assets}
    labels = {asset.ticker: asset.label for asset in group.assets}
    tickers = sorted({asset.ticker for asset in group.assets} | {market_proxy})
    periods = _selected_periods(period_ids)
    start_for_download = min(pd.Timestamp(start) for start, _, _ in periods.values())
    end_for_download = max(pd.Timestamp(end) for _, end, _ in periods.values())
    cached_prices = _load_sufficient_cache_prices(
        tickers,
        cache_dir,
        required_start=start_for_download,
        required_end=end_for_download,
    )
    missing_tickers = [ticker for ticker in tickers if ticker not in cached_prices]
    prices = download_yfinance_prices(
        tickers=missing_tickers,
        start_date=(start_for_download - pd.DateOffset(years=2)).strftime("%Y-%m-%d"),
        end_date=end_for_download.strftime("%Y-%m-%d"),
        cache_dir=cache_dir,
    )
    prices.update(cached_prices)
    dividends = {
        ticker: split_adjusted_dividends(prices[ticker], config.manual_splits.get(ticker, ())) for ticker in tickers
    }
    group_prices = {asset.ticker: prices[asset.ticker] for asset in group.assets}
    flow_frame = load_institutional_flows(flow_source)
    margin_frame = load_margin_short(margin_source) if margin_source else None
    day_trading_frame = load_day_trading(day_trading_source) if day_trading_source else None

    summary_rows: list[dict] = []
    daily_rows: list[dict] = []
    run_log_rows: list[dict] = []
    rules = default_overlay_rules()
    variant = frozen_cycle_proven_top1_v1_variant()

    for period_id, (start, end, period_label) in periods.items():
        (output_path / "current_step.txt").write_text(f"running_{period_id}\n", encoding="utf-8")
        available_prices = {ticker: frame for ticker, frame in group_prices.items() if _covers_period(frame, start, end)}
        available_dividends = {ticker: dividends[ticker] for ticker in available_prices}
        baseline = simulate_regime_mode_switch(
            name=variant.name,
            prices_by_ticker=available_prices,
            asset_types=asset_types,
            market_prices=prices[market_proxy],
            start_date=start,
            end_date=end,
            initial_cash=config.initial_cash_twd,
            cost_model=config.cost_model,
            variant=variant,
            dividend_series_by_ticker=available_dividends,
        )
        summary_rows.append(_summary_row(period_id, period_label, "best_v20260605", "最佳版 v20260605", baseline))
        daily_rows.extend(_daily_rows(period_id, "best_v20260605", "最佳版 v20260605", baseline.equity_curve, labels))
        run_log_rows.append({"period_id": period_id, "candidate_id": "best_v20260605", "status": "completed"})

        for ticker, label in (("0050.TW", "0050買進持有"), ("00631L.TW", "0050正二買進持有")):
            benchmark = simulate_buy_and_hold(
                name=f"{ticker}_buy_and_hold",
                ticker=ticker,
                asset_type=asset_types[ticker],
                prices=available_prices[ticker],
                start_date=start,
                end_date=end,
                initial_cash=config.initial_cash_twd,
                cost_model=config.cost_model,
                dividend_series=dividends[ticker],
            )
            candidate_id = f"benchmark_{ticker.replace('.', '_')}"
            summary_rows.append(_summary_row(period_id, period_label, candidate_id, label, benchmark))
            daily_rows.extend(_daily_rows(period_id, candidate_id, label, benchmark.equity_curve, labels))

        for rule in rules:
            overlay_curve = apply_institutional_overlay(
                baseline=baseline,
                flow_frame=flow_frame,
                rule=rule,
                initial_cash=config.initial_cash_twd,
            )
            candidate_id = f"institutional_overlay_{rule.name}"
            display_name = f"法人籌碼shadow_{rule.name}"
            summary_rows.append(_summary_from_curve(period_id, period_label, candidate_id, display_name, overlay_curve, config.initial_cash_twd))
            daily_rows.extend(_daily_rows(period_id, candidate_id, display_name, overlay_curve, labels))
            run_log_rows.append(
                {
                    "period_id": period_id,
                    "candidate_id": candidate_id,
                    "status": "completed",
                    "risk_flag_days": int(overlay_curve["risk_flag"].sum()),
                }
            )
            pd.DataFrame(run_log_rows).to_csv(output_path / "run_log.csv", index=False, encoding="utf-8-sig")
        if margin_frame is not None and day_trading_frame is not None:
            for rule in default_chip_flow_rules():
                overlay_curve = apply_chip_flow_overlay(
                    baseline=baseline,
                    institutional_frame=flow_frame,
                    margin_frame=margin_frame,
                    day_trading_frame=day_trading_frame,
                    prices_by_ticker=available_prices,
                    rule=rule,
                    initial_cash=config.initial_cash_twd,
                )
                candidate_id = f"chip_flow_overlay_{rule.name}"
                display_name = f"籌碼三層shadow_{rule.name}"
                summary_rows.append(
                    _summary_from_curve(
                        period_id,
                        period_label,
                        candidate_id,
                        display_name,
                        overlay_curve,
                        config.initial_cash_twd,
                    )
                )
                daily_rows.extend(_daily_rows(period_id, candidate_id, display_name, overlay_curve, labels))
                run_log_rows.append(
                    {
                        "period_id": period_id,
                        "candidate_id": candidate_id,
                        "status": "completed",
                        "risk_flag_days": int(overlay_curve["risk_flag"].sum()),
                    }
                )
                pd.DataFrame(run_log_rows).to_csv(output_path / "run_log.csv", index=False, encoding="utf-8-sig")

    summary = pd.DataFrame(summary_rows)
    daily = pd.DataFrame(daily_rows)
    summary.to_csv(output_path / "institutional_flow_overlay_shadow_summary.csv", index=False, encoding="utf-8-sig")
    daily.to_csv(output_path / "institutional_flow_overlay_shadow_daily.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(run_log_rows).to_csv(output_path / "run_log.csv", index=False, encoding="utf-8-sig")
    _write_report(output_path / "institutional_flow_overlay_shadow_report.md", summary)
    (output_path / "metadata.json").write_text(
        json.dumps(
            {
                "model": "institutional_flow_overlay_shadow_v1",
                "baseline": "最佳版 v20260605 / frozen_cycle_proven_top1_v1",
                "flow_source": str(Path(flow_source).resolve()),
                "margin_source": str(Path(margin_source).resolve()) if margin_source else None,
                "day_trading_source": str(Path(day_trading_source).resolve()) if day_trading_source else None,
                "periods": periods,
                "note": "AI輔助回測與策略驗證；shadow approximation 不是正式交易引擎，也不是投資建議。",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (output_path / "current_step.txt").write_text("completed\n", encoding="utf-8")
    return output_path


def _selected_periods(period_ids: list[str]) -> dict[str, tuple[str, str, str]]:
    selected: dict[str, tuple[str, str, str]] = {}
    for period_id in period_ids:
        if period_id not in PERIODS:
            raise ValueError(f"Unsupported period id: {period_id}")
        selected[period_id] = PERIODS[period_id]
    return selected


def _covers_period(frame: pd.DataFrame, start: str, end: str) -> bool:
    if frame.empty:
        return False
    first = frame.index.min()
    last = frame.index.max()
    return (first - pd.Timestamp(start)).days <= 10 and (pd.Timestamp(end) - last).days <= 10


def _summary_row(
    period_id: str,
    period_label: str,
    candidate_id: str,
    strategy_name: str,
    result: BacktestResult,
) -> dict:
    return {
        "period_id": period_id,
        "period_label": period_label,
        "candidate_id": candidate_id,
        "strategy_name": strategy_name,
        "final_value_twd": round(result.final_value, 2),
        "total_return_pct": round(result.total_return * 100, 2),
        "max_drawdown_pct": round(result.max_drawdown * 100, 2),
        "trade_count": sum(1 for trade in result.trades if trade.action in {"buy", "sell"}),
    }


def _summary_from_curve(
    period_id: str,
    period_label: str,
    candidate_id: str,
    strategy_name: str,
    curve: pd.DataFrame,
    initial_cash: float,
) -> dict:
    final_value = float(curve["total_value"].iloc[-1])
    return {
        "period_id": period_id,
        "period_label": period_label,
        "candidate_id": candidate_id,
        "strategy_name": strategy_name,
        "final_value_twd": round(final_value, 2),
        "total_return_pct": round((final_value / initial_cash - 1) * 100, 2),
        "max_drawdown_pct": round(_max_drawdown(curve["total_value"]) * 100, 2),
        "trade_count": "",
        "risk_flag_days": int(curve.get("risk_flag", pd.Series(dtype=bool)).sum()),
    }


def _daily_rows(
    period_id: str,
    candidate_id: str,
    strategy_name: str,
    curve: pd.DataFrame,
    labels: dict[str, str],
) -> list[dict]:
    rows = []
    for date, row in curve.iterrows():
        ticker = str(row.get("current_ticker", "cash"))
        rows.append(
            {
                "period_id": period_id,
                "candidate_id": candidate_id,
                "strategy_name": strategy_name,
                "date": pd.Timestamp(date).strftime("%Y-%m-%d"),
                "total_value_twd": round(float(row["total_value"]), 2),
                "current_ticker": ticker,
                "current_label": labels.get(ticker, ticker),
                "regime": row.get("regime", ""),
                "mode": row.get("mode", ""),
                "risk_flag": row.get("risk_flag", ""),
                "risk_reason": row.get("risk_reason", ""),
                "flow_signal_date": row.get("flow_signal_date", ""),
                "margin_signal_date": row.get("margin_signal_date", ""),
                "day_trading_signal_date": row.get("day_trading_signal_date", ""),
                "price_signal_date": row.get("price_signal_date", ""),
                "baseline_total_value_twd": round(float(row.get("baseline_total_value", row["total_value"])), 2),
                "baseline_exposure": row.get("baseline_exposure", ""),
                "overlay_exposure": row.get("overlay_exposure", ""),
            }
        )
    return rows


def _write_report(path: Path, summary: pd.DataFrame) -> None:
    lines = [
        "# 籌碼資金 overlay shadow v2",
        "",
        "定位：診斷型 shadow 回測，檢查法人買賣超、融資融券/借券、當沖比例是否能改善最佳版 v20260605。",
        "",
        "限制：這不是正式交易引擎，只是用最佳版每日淨值報酬做曝險縮放；若結果有價值，下一步才應整合進正式交易模擬並重算成本。此結果只作 AI 輔助回測與策略驗證，不是投資建議。",
        "",
    ]
    for period_id, frame in summary.groupby("period_id", sort=False):
        lines.append(f"## {period_id}")
        ordered = frame.sort_values("total_return_pct", ascending=False)
        for row in ordered.itertuples(index=False):
            lines.append(
                f"- {row.strategy_name}: final {row.final_value_twd:,.0f}, "
                f"return {row.total_return_pct:.2f}%, maxDD {row.max_drawdown_pct:.2f}%"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run institutional-flow shadow overlays on best v20260605.")
    parser.add_argument("--config", default="configs/ep05_universe.json")
    parser.add_argument("--group-id", default="group_c_0050_00631l_plus_mega_caps")
    parser.add_argument("--cache-dir", default="backtest_cache")
    parser.add_argument("--output-dir", default="outputs/chip_flow_overlay_shadow_v2/latest")
    parser.add_argument("--flow-source", default=DEFAULT_FLOW_SOURCE)
    parser.add_argument("--margin-source", default=DEFAULT_MARGIN_SOURCE)
    parser.add_argument("--day-trading-source", default=DEFAULT_DAY_TRADING_SOURCE)
    parser.add_argument("--periods", default=DEFAULT_PERIODS)
    parser.add_argument("--market-proxy", default="0050.TW")
    args = parser.parse_args()
    output_path = run_overlay_shadow(
        config_path=args.config,
        group_id=args.group_id,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        flow_source=args.flow_source,
        margin_source=args.margin_source,
        day_trading_source=args.day_trading_source,
        period_ids=[period.strip() for period in args.periods.split(",") if period.strip()],
        market_proxy=args.market_proxy,
    )
    print(f"OUTPUT_DIR={output_path.resolve()}")


if __name__ == "__main__":
    main()
