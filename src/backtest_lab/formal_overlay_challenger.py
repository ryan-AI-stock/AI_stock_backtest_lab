from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from backtest_lab.config import load_config
from backtest_lab.data import download_yfinance_prices, split_adjusted_dividends
from backtest_lab.institutional_flow_overlay_shadow import (
    DEFAULT_DAY_TRADING_SOURCE,
    DEFAULT_FLOW_SOURCE,
    DEFAULT_MARGIN_SOURCE,
    DEFAULT_PERIODS,
    ChipFlowOverlayRule,
    _covers_period,
    _lookup_and_dates,
    _previous_price_date,
    _selected_periods,
    _summary_from_curve,
    chip_flow_risk_flag,
    default_chip_flow_rules,
    load_day_trading,
    load_institutional_flows,
    load_margin_short,
    previous_flow_date,
    price_confirmation_flag,
)
from backtest_lab.regime_mode_switch import (
    ExposureOverlayDecision,
    frozen_cycle_proven_top1_v1_variant,
    simulate_regime_mode_switch,
)
from backtest_lab.regime_mode_switch_backtest import _load_sufficient_cache_prices


FORMAL_CHALLENGER_RULE_NAMES = {
    "chip_two_signal_price_dd10_cash",
    "chip_two_signal_price_dd10_reduce25",
    "chip_two_signal_price_dd10_reduce50",
    "chip_two_signal_price_dd10_reduce75",
}


def selected_formal_challenger_rules() -> tuple[ChipFlowOverlayRule, ...]:
    rules = [rule for rule in default_chip_flow_rules() if rule.name in FORMAL_CHALLENGER_RULE_NAMES]
    return tuple(sorted(rules, key=lambda rule: (rule.exposure_cap, rule.name)))


def build_chip_flow_exposure_overlay(
    *,
    institutional_frame: pd.DataFrame,
    margin_frame: pd.DataFrame,
    day_trading_frame: pd.DataFrame,
    prices_by_ticker: dict[str, pd.DataFrame],
    rule: ChipFlowOverlayRule,
):
    institutional_lookup, institutional_dates = _lookup_and_dates(institutional_frame)
    margin_lookup, margin_dates = _lookup_and_dates(margin_frame)
    day_lookup, day_dates = _lookup_and_dates(day_trading_frame)

    def overlay(
        ticker: str | None,
        trade_date: pd.Timestamp,
        signal_date: pd.Timestamp,
        proposed_exposure: float,
    ) -> ExposureOverlayDecision:
        if ticker is None:
            return ExposureOverlayDecision(adjusted_exposure=proposed_exposure)
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
        adjusted = min(proposed_exposure, rule.exposure_cap) if risk_flag else proposed_exposure
        return ExposureOverlayDecision(
            adjusted_exposure=adjusted,
            risk_flag=risk_flag,
            reason=risk_reason,
            signal_date=signal_date.strftime("%Y-%m-%d"),
        )

    return overlay


def run_formal_overlay_challengers(
    *,
    config_path: str,
    group_id: str,
    cache_dir: str,
    output_dir: str,
    flow_source: str,
    margin_source: str,
    day_trading_source: str,
    period_ids: list[str],
    market_proxy: str,
) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "current_step.txt").write_text("loading_inputs\n", encoding="utf-8")

    config = load_config(config_path)
    group = next(group for group in config.groups if group.group_id == group_id)
    asset_types = {asset.ticker: asset.asset_type for asset in group.assets}
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
    institutional_frame = load_institutional_flows(flow_source)
    margin_frame = load_margin_short(margin_source)
    day_trading_frame = load_day_trading(day_trading_source)
    variant = frozen_cycle_proven_top1_v1_variant()

    summary_rows: list[dict] = []
    daily_rows: list[dict] = []
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
        summary_rows.append(
            _summary_from_curve(
                period_id,
                period_label,
                "best_v20260605",
                "最佳版 v20260605",
                baseline.equity_curve,
                config.initial_cash_twd,
            )
        )
        daily_rows.extend(_daily_rows(period_id, "best_v20260605", baseline.equity_curve))
        for rule in selected_formal_challenger_rules():
            overlay = build_chip_flow_exposure_overlay(
                institutional_frame=institutional_frame,
                margin_frame=margin_frame,
                day_trading_frame=day_trading_frame,
                prices_by_ticker=available_prices,
                rule=rule,
            )
            candidate_id = f"formal_engine_{rule.name}"
            result = simulate_regime_mode_switch(
                name=candidate_id,
                prices_by_ticker=available_prices,
                asset_types=asset_types,
                market_prices=prices[market_proxy],
                start_date=start,
                end_date=end,
                initial_cash=config.initial_cash_twd,
                cost_model=config.cost_model,
                variant=variant,
                dividend_series_by_ticker=available_dividends,
                exposure_overlay=overlay,
            )
            summary_rows.append(
                _summary_from_curve(
                    period_id,
                    period_label,
                    candidate_id,
                    f"正式引擎候選_{rule.name}",
                    result.equity_curve,
                    config.initial_cash_twd,
                )
            )
            daily_rows.extend(_daily_rows(period_id, candidate_id, result.equity_curve))

    summary = pd.DataFrame(summary_rows)
    baseline = summary.loc[
        summary["candidate_id"] == "best_v20260605",
        ["period_id", "total_return_pct", "max_drawdown_pct"],
    ].rename(
        columns={
            "total_return_pct": "baseline_total_return_pct",
            "max_drawdown_pct": "baseline_max_drawdown_pct",
        }
    )
    summary = summary.merge(baseline, on="period_id", how="left")
    summary["return_diff_pct"] = (summary["total_return_pct"] - summary["baseline_total_return_pct"]).round(4)
    summary["max_drawdown_diff_pct"] = (summary["max_drawdown_pct"] - summary["baseline_max_drawdown_pct"]).round(4)
    summary["decision_layer"] = "formal_trade_signal"
    summary["formal_promotion_status"] = "challenger_only"
    summary.to_csv(output_path / "formal_overlay_challenger_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(daily_rows).to_csv(output_path / "formal_overlay_challenger_daily.csv", index=False, encoding="utf-8-sig")
    (output_path / "metadata.json").write_text(
        json.dumps(
            {
                "model": "formal_overlay_challenger_v1",
                "baseline": "best_v20260605 / frozen_cycle_proven_top1_v1",
                "decision_layer": "formal_trade_signal",
                "formal_promotion_status": "challenger_only",
                "rule_names": sorted(FORMAL_CHALLENGER_RULE_NAMES),
                "periods": periods,
                "note": "Versioned challenger evidence only; not promoted into the daily formal model.",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (output_path / "current_step.txt").write_text("completed\n", encoding="utf-8")
    return output_path


def _daily_rows(period_id: str, candidate_id: str, curve: pd.DataFrame) -> list[dict]:
    rows = []
    for date, row in curve.iterrows():
        rows.append(
            {
                "period_id": period_id,
                "candidate_id": candidate_id,
                "date": pd.Timestamp(date).strftime("%Y-%m-%d"),
                "total_value_twd": round(float(row["total_value"]), 2),
                "current_ticker": row.get("current_ticker", "cash"),
                "current_exposure": round(float(row.get("current_exposure", 0.0)), 4),
                "overlay_risk_flag": bool(row.get("overlay_risk_flag", False)),
                "overlay_reason": row.get("overlay_reason", ""),
                "overlay_signal_date": row.get("overlay_signal_date", ""),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run versioned formal-engine overlay challengers.")
    parser.add_argument("--config", default="configs/ep05_universe.json")
    parser.add_argument("--group-id", default="group_c_0050_00631l_plus_mega_caps")
    parser.add_argument("--cache-dir", default="backtest_cache")
    parser.add_argument("--output-dir", default="outputs/formal_overlay_challenger")
    parser.add_argument("--flow-source", default=DEFAULT_FLOW_SOURCE)
    parser.add_argument("--margin-source", default=DEFAULT_MARGIN_SOURCE)
    parser.add_argument("--day-trading-source", default=DEFAULT_DAY_TRADING_SOURCE)
    parser.add_argument("--periods", default=DEFAULT_PERIODS)
    parser.add_argument("--market-proxy", default="0050.TW")
    args = parser.parse_args()

    output_path = run_formal_overlay_challengers(
        config_path=args.config,
        group_id=args.group_id,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        flow_source=args.flow_source,
        margin_source=args.margin_source,
        day_trading_source=args.day_trading_source,
        period_ids=[item.strip() for item in args.periods.split(",") if item.strip()],
        market_proxy=args.market_proxy,
    )
    print(f"OUTPUT_DIR={output_path.resolve()}")


if __name__ == "__main__":
    main()
