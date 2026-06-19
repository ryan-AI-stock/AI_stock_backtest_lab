from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd

from backtest_lab.config import load_config
from backtest_lab.data import load_price_csv, split_adjusted_dividends
from backtest_lab.institutional_flow_overlay_shadow import (
    DEFAULT_DAY_TRADING_SOURCE,
    DEFAULT_FLOW_SOURCE,
    DEFAULT_MARGIN_SOURCE,
    _covers_period,
    _lookup_and_dates,
    _summary_from_curve,
    load_day_trading,
    load_institutional_flows,
    load_margin_short,
    previous_flow_date,
)
from backtest_lab.regime_aware_backtest import PERIODS
from backtest_lab.regime_mode_switch import (
    ExposureOverlayDecision,
    frozen_cycle_proven_top1_v1_variant,
    simulate_regime_mode_switch,
)
from backtest_lab.strategies import previous_available_date
from backtest_lab.valuation_source import load_valuation_signals


DEFAULT_PERIODS = "bear_2022,year_2023,ep05_2024_2026"
DEFAULT_CACHE_DIRS = (
    "backtest_cache/ad_hoc_20260612_daily_targets_filled,"
    "backtest_cache/unified_9_asset_full,"
    "backtest_cache/three_model_2018_2023_warmup"
)
DEFAULT_VALUATION_SOURCE = "data/valuation_signals.manual.csv"
ETF_TICKERS = {"0050.TW", "00631L.TW"}
HYPOTHESES = (
    "post_profit_chip_failure",
    "stock_specific_breakdown",
    "crowding_without_price_failure",
    "valuation_entry_block",
    "institutional_divergence",
)


@dataclass(frozen=True)
class FactorLookups:
    institutional: dict[tuple[pd.Timestamp, str], object]
    institutional_dates: list[pd.Timestamp]
    margin: dict[tuple[pd.Timestamp, str], object]
    margin_dates: list[pd.Timestamp]
    day_trading: dict[tuple[pd.Timestamp, str], object]
    day_trading_dates: list[pd.Timestamp]


@dataclass(frozen=True)
class HypothesisConfig:
    hypothesis_id: str
    decision_layer: str
    formal_action: str
    exposure_cap: float | None = None


FORMAL_CHALLENGERS = (
    HypothesisConfig(
        hypothesis_id="post_profit_chip_failure",
        decision_layer="formal_challenger",
        formal_action="cap_exposure_75pct_when_profit_then_factor_failure",
        exposure_cap=0.75,
    ),
    HypothesisConfig(
        hypothesis_id="stock_specific_breakdown",
        decision_layer="formal_challenger",
        formal_action="cap_exposure_75pct_when_stock_lags_market_with_factor_risk",
        exposure_cap=0.75,
    ),
)
DIAGNOSTIC_ONLY = {
    "crowding_without_price_failure": "Crowding without price failure is a warning layer, not a reduction rule.",
    "valuation_entry_block": "Valuation data is entry-specific and currently has insufficient historical coverage.",
    "institutional_divergence": "Institutional divergence needs interpretation before it can become a risk rule.",
}


def run_chip_valuation_event_study(
    *,
    config_path: str,
    group_id: str,
    cache_dirs: list[str],
    output_dir: str,
    flow_source: str,
    margin_source: str,
    day_trading_source: str,
    valuation_source: str | None,
    period_ids: list[str],
    market_proxy: str,
) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    _write_text(output_path / "current_step.txt", "loading_inputs")
    run_log_rows: list[dict] = []

    def log(step: str, status: str, detail: str = "") -> None:
        run_log_rows.append(
            {
                "timestamp": pd.Timestamp.now(tz="Asia/Taipei").strftime("%Y-%m-%d %H:%M:%S%z"),
                "step": step,
                "status": status,
                "detail": detail,
            }
        )
        pd.DataFrame(run_log_rows).to_csv(output_path / "run_log.csv", index=False, encoding="utf-8-sig")
        _write_text(output_path / "current_step.txt", step)

    config = load_config(config_path)
    group = config.group_by_id(group_id)
    asset_types = {asset.ticker: asset.asset_type for asset in group.assets}
    tickers = sorted({asset.ticker for asset in group.assets} | {market_proxy})
    selected_periods = _selected_periods(period_ids)
    log("load_prices", "started", ",".join(cache_dirs))
    prices = load_merged_cache_prices(tickers, cache_dirs)
    missing_prices = sorted(ticker for ticker in tickers if ticker not in prices)
    if missing_prices:
        raise ValueError(f"Missing price cache for: {', '.join(missing_prices)}")
    dividends = {
        ticker: split_adjusted_dividends(prices[ticker], config.manual_splits.get(ticker, ())) for ticker in tickers
    }
    log("load_prices", "completed", f"loaded={len(prices)} missing={len(missing_prices)}")

    log("load_factors", "started", "")
    institutional_frame = _load_optional_factor(flow_source, load_institutional_flows)
    margin_frame = _load_optional_factor(margin_source, load_margin_short)
    day_trading_frame = _load_optional_factor(day_trading_source, load_day_trading)
    lookups = FactorLookups(
        *_lookup_and_dates(institutional_frame) if not institutional_frame.empty else ({}, []),
        *_lookup_and_dates(margin_frame) if not margin_frame.empty else ({}, []),
        *_lookup_and_dates(day_trading_frame) if not day_trading_frame.empty else ({}, []),
    )
    log(
        "load_factors",
        "completed",
        f"institutional={len(institutional_frame)} margin={len(margin_frame)} day={len(day_trading_frame)}",
    )

    variant = frozen_cycle_proven_top1_v1_variant()
    panel_rows: list[dict] = []
    challenger_rows: list[dict] = []
    baseline_rows: list[dict] = []
    failed_rows: list[dict] = []

    for period_id, (start, end, label) in selected_periods.items():
        log(f"baseline_{period_id}", "started", f"{start}~{end}")
        group_prices = {asset.ticker: prices[asset.ticker] for asset in group.assets if _covers_period(prices[asset.ticker], start, end)}
        period_asset_types = {ticker: asset_types[ticker] for ticker in group_prices}
        period_dividends = {ticker: dividends[ticker] for ticker in group_prices}
        baseline = simulate_regime_mode_switch(
            name=variant.name,
            prices_by_ticker=group_prices,
            asset_types=period_asset_types,
            market_prices=prices[market_proxy],
            start_date=start,
            end_date=end,
            initial_cash=config.initial_cash_twd,
            cost_model=config.cost_model,
            variant=variant,
            dividend_series_by_ticker=period_dividends,
        )
        baseline_summary = _summary_from_curve(
            period_id,
            label,
            "best_v20260605",
            "最佳版 v20260605",
            baseline.equity_curve,
            config.initial_cash_twd,
        )
        baseline_rows.append(baseline_summary)
        panel_rows.extend(
            build_factor_event_panel(
                period_id=period_id,
                baseline_curve=baseline.equity_curve,
                prices_by_ticker=group_prices,
                market_prices=prices[market_proxy],
                lookups=lookups,
                valuation_source=valuation_source,
            )
        )
        log(f"baseline_{period_id}", "completed", f"final={baseline.final_value:.2f}")

        for challenger in FORMAL_CHALLENGERS:
            log(f"challenger_{period_id}_{challenger.hypothesis_id}", "started", "")
            try:
                overlay = build_hypothesis_exposure_overlay(
                    hypothesis_id=challenger.hypothesis_id,
                    exposure_cap=challenger.exposure_cap or 1.0,
                    prices_by_ticker=group_prices,
                    market_prices=prices[market_proxy],
                    lookups=lookups,
                    valuation_source=valuation_source,
                )
                result = simulate_regime_mode_switch(
                    name=f"{variant.name}_{challenger.hypothesis_id}",
                    prices_by_ticker=group_prices,
                    asset_types=period_asset_types,
                    market_prices=prices[market_proxy],
                    start_date=start,
                    end_date=end,
                    initial_cash=config.initial_cash_twd,
                    cost_model=config.cost_model,
                    variant=variant,
                    dividend_series_by_ticker=period_dividends,
                    exposure_overlay=overlay,
                )
                summary = _summary_from_curve(
                    period_id,
                    label,
                    f"challenger_{challenger.hypothesis_id}",
                    challenger.formal_action,
                    result.equity_curve,
                    config.initial_cash_twd,
                )
                summary["hypothesis_id"] = challenger.hypothesis_id
                summary["decision_layer"] = challenger.decision_layer
                summary["formal_action"] = challenger.formal_action
                summary["overlay_trigger_days"] = _overlay_trigger_days(result.equity_curve)
                challenger_rows.append(summary)
                log(f"challenger_{period_id}_{challenger.hypothesis_id}", "completed", f"final={result.final_value:.2f}")
            except Exception as exc:  # pragma: no cover - failure is recorded for resume/debug.
                failed_rows.append({"period_id": period_id, "hypothesis_id": challenger.hypothesis_id, "error": str(exc)})
                log(f"challenger_{period_id}_{challenger.hypothesis_id}", "failed", str(exc))

    panel = pd.DataFrame(panel_rows)
    panel_path = output_path / "factor_event_panel.csv"
    panel.to_csv(panel_path, index=False, encoding="utf-8-sig")

    event_summary = summarize_factor_events(panel)
    event_summary.to_csv(output_path / "factor_event_summary.csv", index=False, encoding="utf-8-sig")

    baseline_frame = pd.DataFrame(baseline_rows)
    challenger_summary = build_formal_challenger_summary(
        baseline_frame=baseline_frame,
        challenger_frame=pd.DataFrame(challenger_rows),
        event_summary=event_summary,
    )
    challenger_summary.to_csv(output_path / "formal_challenger_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(failed_rows).to_csv(output_path / "failed.csv", index=False, encoding="utf-8-sig")

    metadata = {
        "task_id": "TASK-BACKTEST-CORE-CHIP-VALUATION-HYPOTHESIS-001",
        "model": "chip_valuation_event_study_v0",
        "baseline": "best_v20260605 / frozen_cycle_proven_top1_v1",
        "formal_model_changed": False,
        "periods": selected_periods,
        "hypotheses": list(HYPOTHESES),
        "formal_challengers": [config.__dict__ for config in FORMAL_CHALLENGERS],
        "diagnostic_only": DIAGNOSTIC_ONLY,
        "price_coverage": _price_coverage(prices),
        "factor_coverage": {
            "institutional": _frame_coverage(institutional_frame),
            "margin_short": _frame_coverage(margin_frame),
            "day_trading": _frame_coverage(day_trading_frame),
            "valuation_source": str(valuation_source or ""),
        },
        "outputs": [
            "factor_event_panel.csv",
            "factor_event_summary.csv",
            "hypothesis_review.md",
            "formal_challenger_summary.csv",
            "metadata.json",
            "run_log.csv",
        ],
        "note": "Outputs are validation evidence only. No shadow or challenger is promoted into the formal daily model here.",
    }
    _write_json(output_path / "metadata.json", metadata)
    _write_text(output_path / "hypothesis_review.md", build_hypothesis_review(event_summary, challenger_summary, metadata))
    _write_text(output_path / "completed.txt", "completed")
    _write_text(output_path / "current_step.txt", "completed")
    log("completed", "completed", str(output_path.resolve()))
    return output_path


def load_merged_cache_prices(tickers: list[str], cache_dirs: list[str]) -> dict[str, pd.DataFrame]:
    prices: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        frames = []
        file_name = f"{ticker.replace('.', '_')}.csv"
        for cache_dir in cache_dirs:
            path = Path(cache_dir) / file_name
            if path.exists():
                frames.append(load_price_csv(path))
        if frames:
            frame = pd.concat(frames).sort_index()
            frame = frame[~frame.index.duplicated(keep="last")]
            prices[ticker] = frame
    return prices


def build_factor_event_panel(
    *,
    period_id: str,
    baseline_curve: pd.DataFrame,
    prices_by_ticker: dict[str, pd.DataFrame],
    market_prices: pd.DataFrame,
    lookups: FactorLookups,
    valuation_source: str | None,
) -> list[dict]:
    rows: list[dict] = []
    current_ticker: str | None = None
    entry_price = 0.0
    entry_date: pd.Timestamp | None = None
    holding_peak_price = 0.0
    all_prices = {**prices_by_ticker, "__market__": market_prices}
    for trade_date, equity_row in baseline_curve.iterrows():
        trade_ts = pd.Timestamp(trade_date).normalize()
        try:
            signal_date = previous_available_date(all_prices, trade_ts)
        except ValueError:
            continue
        ticker = str(equity_row.get("current_ticker") or "cash")
        if ticker == "cash" or ticker not in prices_by_ticker:
            current_ticker = ticker
            rows.append(_cash_panel_row(period_id, trade_ts, signal_date, equity_row))
            continue
        ticker_prices = prices_by_ticker[ticker]
        if signal_date not in ticker_prices.index:
            continue
        close = float(ticker_prices.loc[signal_date, "close"])
        if ticker != current_ticker:
            current_ticker = ticker
            entry_date = signal_date
            entry_price = close
            holding_peak_price = close
        else:
            holding_peak_price = max(holding_peak_price, close)
        feature_row = factor_event_features(
            ticker=ticker,
            signal_date=signal_date,
            trade_date=trade_ts,
            ticker_prices=ticker_prices,
            market_prices=market_prices,
            lookups=lookups,
            valuation_source=valuation_source,
            entry_price=entry_price,
            entry_date=entry_date,
            holding_peak_price=holding_peak_price,
        )
        feature_row.update(
            {
                "period_id": period_id,
                "trade_date": trade_ts.strftime("%Y-%m-%d"),
                "signal_date": signal_date.strftime("%Y-%m-%d"),
                "total_value_twd": round(float(equity_row.get("total_value", 0.0)), 2),
                "current_exposure": round(float(equity_row.get("current_exposure", 0.0)), 4),
                "regime": equity_row.get("regime", ""),
                "mode": equity_row.get("mode", ""),
                "attack_gate_active": bool(equity_row.get("attack_gate_active", False)),
                "decision_layer": "diagnostic",
            }
        )
        rows.append(feature_row)
    return rows


def factor_event_features(
    *,
    ticker: str,
    signal_date: pd.Timestamp,
    trade_date: pd.Timestamp,
    ticker_prices: pd.DataFrame,
    market_prices: pd.DataFrame,
    lookups: FactorLookups,
    valuation_source: str | None,
    entry_price: float = 0.0,
    entry_date: pd.Timestamp | None = None,
    holding_peak_price: float = 0.0,
) -> dict:
    history = ticker_prices.loc[ticker_prices.index <= signal_date]
    market_history = market_prices.loc[market_prices.index <= signal_date]
    close = float(history["close"].iloc[-1])
    entry_price = entry_price if entry_price > 0 else close
    holding_peak_price = max(holding_peak_price, close)
    institutional_row, institutional_date = _latest_factor_row(lookups.institutional, lookups.institutional_dates, trade_date, ticker)
    margin_row, margin_date = _latest_factor_row(lookups.margin, lookups.margin_dates, trade_date, ticker)
    day_row, day_date = _latest_factor_row(lookups.day_trading, lookups.day_trading_dates, trade_date, ticker)
    valuation = load_valuation_signals(
        valuation_source,
        signal_date=signal_date,
        current_price_by_ticker={ticker: close},
    ).get(ticker)
    feature = {
        "ticker": ticker,
        "close": round(close, 4),
        "entry_date": entry_date.strftime("%Y-%m-%d") if entry_date is not None else "",
        "entry_price": round(entry_price, 4),
        "holding_return_pct": round(close / entry_price - 1, 6) if entry_price > 0 else 0.0,
        "holding_peak_gain_pct": round(holding_peak_price / entry_price - 1, 6) if entry_price > 0 else 0.0,
        "ret_5d_pct": _past_return(history, 5),
        "ret_10d_pct": _past_return(history, 10),
        "ret_20d_pct": _past_return(history, 20),
        "ret_60d_pct": _past_return(history, 60),
        "drawdown_10d_pct": _drawdown_from_high(history, 10),
        "below_ma10": _below_ma(history, 10),
        "below_ma20": _below_ma(history, 20),
        "relative_ret_10d_vs_market_pct": round(_past_return(history, 10) - _past_return(market_history, 10), 6),
        "future_ret_5d_pct": _future_return(ticker_prices, signal_date, 5),
        "future_ret_10d_pct": _future_return(ticker_prices, signal_date, 10),
        "future_ret_20d_pct": _future_return(ticker_prices, signal_date, 20),
        "future_rel_ret_20d_vs_market_pct": round(
            _future_return(ticker_prices, signal_date, 20) - _future_return(market_prices, signal_date, 20),
            6,
        ),
        "future_max_adverse_20d_pct": _future_max_adverse(ticker_prices, signal_date, 20),
        "institutional_signal_date": institutional_date.strftime("%Y-%m-%d") if institutional_date else "",
        "margin_signal_date": margin_date.strftime("%Y-%m-%d") if margin_date else "",
        "day_trading_signal_date": day_date.strftime("%Y-%m-%d") if day_date else "",
        "valuation_signal_date": valuation.signal_date if valuation else "",
    }
    feature.update(_institutional_features(institutional_row))
    feature.update(_margin_features(margin_row))
    feature.update(_day_trading_features(day_row))
    feature.update(_valuation_features(valuation))
    feature.update(classify_hypotheses(feature))
    return feature


def classify_hypotheses(feature: dict) -> dict:
    stock = feature["ticker"] not in ETF_TICKERS
    institutional_risk = bool(feature.get("institutional_sync_sell"))
    margin_or_short_risk = bool(feature.get("margin_overheat_flag")) or bool(feature.get("short_lending_pressure_flag"))
    day_risk = bool(feature.get("day_trading_overheat_flag")) or float(feature.get("day_trading_volume_ratio", 0.0) or 0.0) >= 35.0
    factor_risk_count = int(institutional_risk) + int(margin_or_short_risk) + int(day_risk)
    price_failure = bool(feature.get("below_ma20")) or float(feature.get("drawdown_10d_pct", 0.0)) <= -0.08
    crowding = margin_or_short_risk or day_risk
    output = {
        "factor_risk_count": factor_risk_count,
        "post_profit_chip_failure": bool(
            stock
            and float(feature.get("holding_peak_gain_pct", 0.0)) >= 0.25
            and price_failure
            and factor_risk_count >= 2
        ),
        "stock_specific_breakdown": bool(
            stock
            and float(feature.get("relative_ret_10d_vs_market_pct", 0.0)) <= -0.05
            and price_failure
            and factor_risk_count >= 1
        ),
        "crowding_without_price_failure": bool(
            stock
            and crowding
            and not price_failure
            and float(feature.get("relative_ret_10d_vs_market_pct", 0.0)) >= 0.0
        ),
        "valuation_entry_block": bool(stock and not bool(feature.get("valuation_gate_passed", True))),
        "institutional_divergence": bool(
            stock
            and int(feature.get("foreign_consecutive_sell_days", 0) or 0) >= 3
            and float(feature.get("investment_trust_net_buy_shares", 0.0) or 0.0) > 0
        ),
    }
    return output


def build_hypothesis_exposure_overlay(
    *,
    hypothesis_id: str,
    exposure_cap: float,
    prices_by_ticker: dict[str, pd.DataFrame],
    market_prices: pd.DataFrame,
    lookups: FactorLookups,
    valuation_source: str | None,
) -> Callable[[str | None, pd.Timestamp, pd.Timestamp, float], ExposureOverlayDecision]:
    def overlay(
        ticker: str | None,
        trade_date: pd.Timestamp,
        signal_date: pd.Timestamp,
        proposed_exposure: float,
    ) -> ExposureOverlayDecision:
        if ticker is None or ticker == "cash" or ticker in ETF_TICKERS or ticker not in prices_by_ticker:
            return ExposureOverlayDecision(adjusted_exposure=proposed_exposure)
        ticker_prices = prices_by_ticker[ticker]
        if signal_date not in ticker_prices.index:
            return ExposureOverlayDecision(adjusted_exposure=proposed_exposure)
        signal_close = float(ticker_prices.loc[signal_date, "close"])
        history = ticker_prices.loc[ticker_prices.index <= signal_date]
        entry_proxy = signal_close / (1 + max(_past_return(history, 60), -0.95))
        holding_peak_proxy = float(history["close"].tail(60).max()) if len(history) else signal_close
        feature = factor_event_features(
            ticker=ticker,
            signal_date=signal_date,
            trade_date=trade_date,
            ticker_prices=ticker_prices,
            market_prices=market_prices,
            lookups=lookups,
            valuation_source=valuation_source,
            entry_price=entry_proxy,
            entry_date=signal_date,
            holding_peak_price=holding_peak_proxy,
        )
        if bool(feature.get(hypothesis_id, False)):
            return ExposureOverlayDecision(
                adjusted_exposure=min(proposed_exposure, exposure_cap),
                risk_flag=True,
                reason=hypothesis_id,
                signal_date=signal_date.strftime("%Y-%m-%d"),
            )
        return ExposureOverlayDecision(adjusted_exposure=proposed_exposure)

    return overlay


def summarize_factor_events(panel: pd.DataFrame) -> pd.DataFrame:
    if panel.empty:
        return pd.DataFrame()
    rows: list[dict] = []
    for period_id, period_frame in panel.groupby("period_id"):
        total_rows = len(period_frame)
        for hypothesis_id in HYPOTHESES:
            if hypothesis_id not in period_frame.columns:
                continue
            events = period_frame[period_frame[hypothesis_id].fillna(False).astype(bool)].copy()
            rows.append(
                {
                    "period_id": period_id,
                    "hypothesis_id": hypothesis_id,
                    "decision_layer": _decision_layer_for_hypothesis(hypothesis_id),
                    "sample_rows": total_rows,
                    "event_count": len(events),
                    "event_rate_pct": round(len(events) / total_rows, 6) if total_rows else 0.0,
                    "avg_future_ret_5d_pct": _mean(events, "future_ret_5d_pct"),
                    "avg_future_ret_10d_pct": _mean(events, "future_ret_10d_pct"),
                    "avg_future_ret_20d_pct": _mean(events, "future_ret_20d_pct"),
                    "avg_future_rel_ret_20d_vs_market_pct": _mean(events, "future_rel_ret_20d_vs_market_pct"),
                    "negative_future_rel_20d_rate_pct": _negative_rate(events, "future_rel_ret_20d_vs_market_pct"),
                    "avg_future_max_adverse_20d_pct": _mean(events, "future_max_adverse_20d_pct"),
                    "data_readiness": _data_readiness(period_frame, hypothesis_id),
                }
            )
    return pd.DataFrame(rows)


def build_formal_challenger_summary(
    *,
    baseline_frame: pd.DataFrame,
    challenger_frame: pd.DataFrame,
    event_summary: pd.DataFrame,
) -> pd.DataFrame:
    if challenger_frame.empty:
        return pd.DataFrame(
            [
                {
                    "hypothesis_id": hypothesis_id,
                    "formal_promotion_status": "not_promoted_no_challenger_result",
                    "reason": "No challenger result was produced.",
                }
                for hypothesis_id in HYPOTHESES
            ]
        )
    baseline = baseline_frame[
        ["period_id", "total_return_pct", "max_drawdown_pct"]
    ].rename(
        columns={
            "total_return_pct": "baseline_total_return_pct",
            "max_drawdown_pct": "baseline_max_drawdown_pct",
        }
    )
    merged = challenger_frame.merge(baseline, on="period_id", how="left")
    merged["return_diff_pct"] = (merged["total_return_pct"] - merged["baseline_total_return_pct"]).round(4)
    merged["max_drawdown_diff_pct"] = (merged["max_drawdown_pct"] - merged["baseline_max_drawdown_pct"]).round(4)
    coverage = event_summary.pivot_table(
        index="hypothesis_id",
        values="event_count",
        aggfunc="sum",
    ).rename(columns={"event_count": "total_event_count"})
    merged = merged.merge(coverage, on="hypothesis_id", how="left")
    if {"period_id", "hypothesis_id", "event_count"}.issubset(event_summary.columns):
        period_coverage = event_summary[["period_id", "hypothesis_id", "event_count"]].rename(
            columns={"event_count": "period_event_count"}
        )
        merged = merged.merge(period_coverage, on=["period_id", "hypothesis_id"], how="left")
    else:
        merged["period_event_count"] = 0
    merged["total_event_count"] = merged["total_event_count"].fillna(0).astype(int)
    merged["period_event_count"] = merged["period_event_count"].fillna(0).astype(int)
    status_by_hypothesis: dict[str, str] = {}
    reason_by_hypothesis: dict[str, str] = {}
    for hypothesis_id, group in merged.groupby("hypothesis_id"):
        returns_ok = (group["return_diff_pct"] >= 0).all()
        drawdown_ok = (group["max_drawdown_diff_pct"] <= 0).all()
        has_events = int(group["total_event_count"].max()) > 0
        has_all_periods = set(group["period_id"]) >= {"bear_2022", "year_2023", "ep05_2024_2026"}
        if returns_ok and drawdown_ok and has_events and has_all_periods:
            status = "candidate_for_next_validation"
            reason = "Challenger did not trail baseline in return or drawdown across configured periods."
        else:
            status = "not_promoted"
            reason = "No robust increment over baseline, insufficient events, or incomplete period coverage."
        status_by_hypothesis[hypothesis_id] = status
        reason_by_hypothesis[hypothesis_id] = reason
    merged["formal_promotion_status"] = merged["hypothesis_id"].map(status_by_hypothesis)
    merged["promotion_reason"] = merged["hypothesis_id"].map(reason_by_hypothesis)
    diagnostic_rows = [
        {
            "period_id": "",
            "candidate_id": f"diagnostic_{hypothesis_id}",
            "candidate_label": reason,
            "final_value_twd": "",
            "total_return_pct": "",
            "max_drawdown_pct": "",
            "trade_count": "",
            "hypothesis_id": hypothesis_id,
            "decision_layer": "diagnostic",
            "formal_action": "not_tested_as_trade_rule",
            "baseline_total_return_pct": "",
            "baseline_max_drawdown_pct": "",
            "return_diff_pct": "",
            "max_drawdown_diff_pct": "",
            "total_event_count": int(
                event_summary.loc[event_summary["hypothesis_id"] == hypothesis_id, "event_count"].sum()
            )
            if not event_summary.empty
            else 0,
            "period_event_count": "",
            "overlay_trigger_days": "",
            "formal_promotion_status": "not_promoted_diagnostic_only",
            "promotion_reason": reason,
        }
        for hypothesis_id, reason in DIAGNOSTIC_ONLY.items()
    ]
    if diagnostic_rows:
        merged = pd.concat([merged, pd.DataFrame(diagnostic_rows)], ignore_index=True)
    return merged


def build_hypothesis_review(event_summary: pd.DataFrame, challenger_summary: pd.DataFrame, metadata: dict) -> str:
    lines = [
        "# 籌碼/融資/當沖/估值假設驗證摘要",
        "",
        f"- 任務：{metadata['task_id']}",
        f"- baseline：{metadata['baseline']}",
        "- 結論定位：事件研究與 versioned challenger，尚未升級正式每日模型。",
        "",
        "## 事件研究",
    ]
    if event_summary.empty:
        lines.append("- 沒有產生事件樣本。")
    else:
        for row in event_summary.itertuples(index=False):
            lines.append(
                "- "
                f"{row.period_id} / {row.hypothesis_id}: "
                f"事件 {row.event_count} 筆，20日相對報酬均值 {row.avg_future_rel_ret_20d_vs_market_pct:.4f}，"
                f"資料狀態 {row.data_readiness}。"
            )
    lines.extend(["", "## 正式 challenger", ""])
    if challenger_summary.empty:
        lines.append("- 沒有 challenger 結果。")
    else:
        promoted = challenger_summary[
            challenger_summary["formal_promotion_status"].astype(str).str.contains("candidate", na=False)
        ]
        if promoted.empty:
            lines.append("- 本輪沒有任何假設可直接升正式模型。")
        else:
            lines.append("- 以下假設可進下一輪驗證，不代表已升正式：")
            for hypothesis in sorted(promoted["hypothesis_id"].dropna().unique()):
                lines.append(f"  - {hypothesis}")
    lines.extend(
        [
            "",
            "## 主要限制",
            "",
            "- 2024-2026 的正式籌碼/融資/當沖歷史資料覆蓋不足時，只能標為 data_gap，不能拿來證明正式有效。",
            "- 估值資料目前多為手動快照，不足以支撐 2022/2023 的歷史驗證。",
            "- 本檔案不包含買進、賣出或保證績效語氣。",
        ]
    )
    return "\n".join(lines) + "\n"


def _selected_periods(period_ids: list[str]) -> dict[str, tuple[str, str, str]]:
    selected: dict[str, tuple[str, str, str]] = {}
    for period_id in period_ids:
        if period_id not in PERIODS:
            raise KeyError(f"Unknown period id: {period_id}")
        selected[period_id] = PERIODS[period_id]
    return selected


def _load_optional_factor(path: str | Path, loader: Callable[[str | Path], pd.DataFrame]) -> pd.DataFrame:
    source = Path(path)
    if not source.exists():
        return pd.DataFrame()
    return loader(source)


def _latest_factor_row(
    lookup: dict[tuple[pd.Timestamp, str], object],
    dates: list[pd.Timestamp],
    trade_date: pd.Timestamp,
    ticker: str,
    *,
    max_lag_days: int = 7,
) -> tuple[object | None, pd.Timestamp | None]:
    factor_date = previous_flow_date(dates, trade_date)
    if factor_date is None:
        return None, None
    if (trade_date.normalize() - factor_date.normalize()).days > max_lag_days:
        return None, None
    return lookup.get((factor_date, ticker)), factor_date


def _institutional_features(row: object | None) -> dict:
    if row is None:
        return {
            "institutional_data_available": False,
            "foreign_net_buy_shares": 0.0,
            "investment_trust_net_buy_shares": 0.0,
            "dealer_net_buy_shares": 0.0,
            "total_institutional_net_buy_shares": 0.0,
            "foreign_consecutive_sell_days": 0,
            "trust_consecutive_sell_days": 0,
            "institutional_sync_sell": False,
        }
    foreign = float(getattr(row, "foreign_net_buy_shares", 0.0) or 0.0)
    trust = float(getattr(row, "investment_trust_net_buy_shares", 0.0) or 0.0)
    dealer = float(getattr(row, "dealer_net_buy_shares", 0.0) or 0.0)
    foreign_sell = int(getattr(row, "foreign_consecutive_sell_days", 0) or 0)
    trust_sell = int(getattr(row, "trust_consecutive_sell_days", 0) or 0)
    total = foreign + trust + dealer
    return {
        "institutional_data_available": True,
        "foreign_net_buy_shares": foreign,
        "investment_trust_net_buy_shares": trust,
        "dealer_net_buy_shares": dealer,
        "total_institutional_net_buy_shares": total,
        "foreign_consecutive_sell_days": foreign_sell,
        "trust_consecutive_sell_days": trust_sell,
        "institutional_sync_sell": (foreign_sell >= 3 or trust_sell >= 2) and total < 0,
    }


def _margin_features(row: object | None) -> dict:
    if row is None:
        return {
            "margin_data_available": False,
            "margin_balance_5d_change_pct": 0.0,
            "margin_balance_20d_change_pct": 0.0,
            "short_balance_5d_change_pct": 0.0,
            "margin_overheat_flag": False,
            "short_lending_pressure_flag": False,
        }
    return {
        "margin_data_available": True,
        "margin_balance_5d_change_pct": _float(getattr(row, "margin_balance_5d_change_pct", 0.0)),
        "margin_balance_20d_change_pct": _float(getattr(row, "margin_balance_20d_change_pct", 0.0)),
        "short_balance_5d_change_pct": _float(getattr(row, "short_balance_5d_change_pct", 0.0)),
        "margin_overheat_flag": _bool(getattr(row, "margin_overheat_flag", False)),
        "short_lending_pressure_flag": _bool(getattr(row, "short_lending_pressure_flag", False)),
    }


def _day_trading_features(row: object | None) -> dict:
    if row is None:
        return {
            "day_trading_data_available": False,
            "day_trading_volume_ratio": 0.0,
            "day_trading_ratio_5d_avg": 0.0,
            "day_trading_overheat_flag": False,
        }
    return {
        "day_trading_data_available": True,
        "day_trading_volume_ratio": _float(getattr(row, "day_trading_volume_ratio", 0.0)),
        "day_trading_ratio_5d_avg": _float(getattr(row, "day_trading_ratio_5d_avg", 0.0)),
        "day_trading_overheat_flag": _bool(getattr(row, "day_trading_overheat_flag", False)),
    }


def _valuation_features(signal: object | None) -> dict:
    if signal is None:
        return {
            "valuation_data_available": False,
            "valuation_gate_passed": True,
            "valuation_safety_margin_pct": 0.0,
            "valuation_reason": "",
        }
    return {
        "valuation_data_available": True,
        "valuation_gate_passed": bool(signal.gate_passed),
        "valuation_safety_margin_pct": round(float(signal.safety_margin_pct), 6),
        "valuation_reason": signal.reason,
    }


def _cash_panel_row(period_id: str, trade_date: pd.Timestamp, signal_date: pd.Timestamp, equity_row: pd.Series) -> dict:
    row = {
        "period_id": period_id,
        "trade_date": trade_date.strftime("%Y-%m-%d"),
        "signal_date": signal_date.strftime("%Y-%m-%d"),
        "ticker": "cash",
        "total_value_twd": round(float(equity_row.get("total_value", 0.0)), 2),
        "current_exposure": 0.0,
        "decision_layer": "diagnostic",
    }
    row.update({hypothesis: False for hypothesis in HYPOTHESES})
    return row


def _past_return(frame: pd.DataFrame, days: int) -> float:
    if len(frame) <= days:
        return 0.0
    return round(float(frame["close"].iloc[-1]) / float(frame["close"].iloc[-days - 1]) - 1, 6)


def _future_return(frame: pd.DataFrame, signal_date: pd.Timestamp, days: int) -> float:
    if signal_date not in frame.index:
        return 0.0
    index = frame.index.get_loc(signal_date)
    if isinstance(index, slice):
        index = index.stop - 1
    target_index = int(index) + days
    if target_index >= len(frame):
        return 0.0
    return round(float(frame["close"].iloc[target_index]) / float(frame["close"].iloc[int(index)]) - 1, 6)


def _future_max_adverse(frame: pd.DataFrame, signal_date: pd.Timestamp, days: int) -> float:
    if signal_date not in frame.index:
        return 0.0
    index = frame.index.get_loc(signal_date)
    if isinstance(index, slice):
        index = index.stop - 1
    index = int(index)
    future = frame.iloc[index + 1 : min(index + days + 1, len(frame))]
    if future.empty:
        return 0.0
    close = float(frame["close"].iloc[index])
    return round(float(future["close"].min()) / close - 1, 6)


def _drawdown_from_high(frame: pd.DataFrame, days: int) -> float:
    if len(frame) < days:
        return 0.0
    close = float(frame["close"].iloc[-1])
    high = float(frame["close"].tail(days).max())
    return round(close / high - 1, 6) if high > 0 else 0.0


def _below_ma(frame: pd.DataFrame, days: int) -> bool:
    if len(frame) < days:
        return False
    return float(frame["close"].iloc[-1]) < float(frame["close"].tail(days).mean())


def _mean(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame:
        return 0.0
    return round(float(pd.to_numeric(frame[column], errors="coerce").dropna().mean() or 0.0), 6)


def _negative_rate(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame:
        return 0.0
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if values.empty:
        return 0.0
    return round(float((values < 0).mean()), 6)


def _data_readiness(frame: pd.DataFrame, hypothesis_id: str) -> str:
    if frame.empty:
        return "empty"
    eligible = frame[
        ~frame.get("ticker", pd.Series("", index=frame.index)).astype(str).isin(ETF_TICKERS | {"cash"})
    ].copy()
    if eligible.empty:
        return "no_stock_exposure"
    if hypothesis_id == "valuation_entry_block":
        coverage = _coverage_ratio(eligible, "valuation_data_available")
        if coverage >= 0.8:
            return f"ready_{coverage:.2f}"
        if coverage > 0:
            return f"partial_{coverage:.2f}"
        return "data_gap"
    required = ["institutional_data_available", "margin_data_available", "day_trading_data_available"]
    coverages = [_coverage_ratio(eligible, column) for column in required]
    min_coverage = min(coverages) if coverages else 0.0
    if min_coverage >= 0.8:
        return f"ready_{min_coverage:.2f}"
    if min_coverage > 0:
        return f"partial_{min_coverage:.2f}"
    return "data_gap"


def _coverage_ratio(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame:
        return 0.0
    return round(float(frame[column].fillna(False).astype(bool).mean()), 4)


def _decision_layer_for_hypothesis(hypothesis_id: str) -> str:
    if hypothesis_id in {config.hypothesis_id for config in FORMAL_CHALLENGERS}:
        return "formal_challenger"
    return "diagnostic"


def _overlay_trigger_days(curve: pd.DataFrame) -> int:
    if "overlay_risk_flag" not in curve.columns:
        return 0
    return int(curve["overlay_risk_flag"].fillna(False).astype(bool).sum())


def _price_coverage(prices: dict[str, pd.DataFrame]) -> dict[str, dict[str, str | int]]:
    return {
        ticker: {
            "start": frame.index.min().strftime("%Y-%m-%d") if not frame.empty else "",
            "end": frame.index.max().strftime("%Y-%m-%d") if not frame.empty else "",
            "rows": int(len(frame)),
        }
        for ticker, frame in prices.items()
    }


def _frame_coverage(frame: pd.DataFrame) -> dict[str, str | int]:
    if frame.empty or "date" not in frame.columns:
        return {"start": "", "end": "", "rows": 0, "tickers": 0}
    return {
        "start": pd.Timestamp(frame["date"].min()).strftime("%Y-%m-%d"),
        "end": pd.Timestamp(frame["date"].max()).strftime("%Y-%m-%d"),
        "rows": int(len(frame)),
        "tickers": int(frame["ticker"].nunique()) if "ticker" in frame.columns else 0,
    }


def _float(value: object) -> float:
    number = pd.to_numeric(value, errors="coerce")
    return 0.0 if pd.isna(number) else float(number)


def _bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _write_text(path: Path, text: str) -> None:
    path.write_text(text + ("" if text.endswith("\n") else "\n"), encoding="utf-8")


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run chip/valuation factor event study and versioned challengers.")
    parser.add_argument("--config", default="configs/ep05_universe.json")
    parser.add_argument("--group-id", default="group_c_0050_00631l_plus_mega_caps")
    parser.add_argument("--cache-dirs", default=DEFAULT_CACHE_DIRS)
    parser.add_argument("--output-dir", default="outputs/chip_valuation_event_study")
    parser.add_argument("--flow-source", default=DEFAULT_FLOW_SOURCE)
    parser.add_argument("--margin-source", default=DEFAULT_MARGIN_SOURCE)
    parser.add_argument("--day-trading-source", default=DEFAULT_DAY_TRADING_SOURCE)
    parser.add_argument("--valuation-source", default=DEFAULT_VALUATION_SOURCE)
    parser.add_argument("--periods", default=DEFAULT_PERIODS)
    parser.add_argument("--market-proxy", default="0050.TW")
    args = parser.parse_args()

    output_path = run_chip_valuation_event_study(
        config_path=args.config,
        group_id=args.group_id,
        cache_dirs=[item.strip() for item in args.cache_dirs.split(",") if item.strip()],
        output_dir=args.output_dir,
        flow_source=args.flow_source,
        margin_source=args.margin_source,
        day_trading_source=args.day_trading_source,
        valuation_source=args.valuation_source,
        period_ids=[item.strip() for item in args.periods.split(",") if item.strip()],
        market_proxy=args.market_proxy,
    )
    print(f"OUTPUT_DIR={output_path.resolve()}")


if __name__ == "__main__":
    main()
