from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from backtest_lab.config import GroupConfig, load_config
from backtest_lab.data import download_yfinance_prices, split_adjusted_dividends
from backtest_lab.portfolio import Trade
from backtest_lab.simulation import (
    BacktestResult,
    simulate_buy_and_hold,
    simulate_dual_momentum_vol_control,
    simulate_relative_strength_top1,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run EP05 v0 backtests.")
    parser.add_argument("--config", default="configs/ep05_universe.json")
    parser.add_argument("--cache-dir", default="backtest_cache")
    parser.add_argument("--output-dir", default="backtest_outputs")
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tickers = sorted({asset.ticker for group in config.groups for asset in group.assets})
    prices = download_yfinance_prices(
        tickers=tickers,
        start_date=config.warmup_start_date,
        end_date=config.end_date,
        cache_dir=args.cache_dir,
    )
    dividends = {
        ticker: split_adjusted_dividends(prices[ticker], config.manual_splits.get(ticker, ()))
        for ticker in tickers
    }

    results: list[BacktestResult] = []
    reconciliation_rows: list[dict] = []
    for group in config.groups:
        benchmark_without_dividend = _run_benchmark(group, prices, config, dividend_series=None)
        benchmark_with_dividend = _run_benchmark(group, prices, config, dividend_series=dividends[group.benchmark])
        results.append(benchmark_with_dividend)
        reconciliation_rows.append(
            _benchmark_reconciliation_row(
                group,
                benchmark_without_dividend,
                benchmark_with_dividend,
                config.reference_values.get(group.benchmark),
            )
        )
        group_prices = {asset.ticker: prices[asset.ticker] for asset in group.assets}
        asset_types = {asset.ticker: asset.asset_type for asset in group.assets}
        group_dividends = {asset.ticker: dividends[asset.ticker] for asset in group.assets}
        results.append(
            simulate_relative_strength_top1(
                name=f"{group.group_id}__relative_strength_top1",
                prices_by_ticker=group_prices,
                asset_types=asset_types,
                start_date=config.start_date,
                end_date=config.end_date,
                initial_cash=config.initial_cash_twd,
                cost_model=config.cost_model,
                dividend_series_by_ticker=group_dividends,
            )
        )
        results.append(
            simulate_dual_momentum_vol_control(
                name=f"{group.group_id}__dual_momentum_vol_control",
                prices_by_ticker=group_prices,
                asset_types=asset_types,
                start_date=config.start_date,
                end_date=config.end_date,
                initial_cash=config.initial_cash_twd,
                cost_model=config.cost_model,
                dividend_series_by_ticker=group_dividends,
            )
        )

    _write_summary(results, output_dir)
    _write_benchmark_reconciliation(reconciliation_rows, output_dir)
    _write_trades(results, output_dir)
    _write_equity_curves(results, output_dir)
    _write_holding_exposure(results, output_dir)
    _write_robustness_summary(_run_robustness_checks(config, prices, dividends), output_dir)
    _write_video_summary(results, output_dir)


def _run_benchmark(
    group: GroupConfig,
    prices: dict[str, pd.DataFrame],
    config,
    dividend_series: pd.Series | None,
) -> BacktestResult:
    return simulate_buy_and_hold(
        name=f"{group.group_id}__benchmark__{group.benchmark}",
        ticker=group.benchmark,
        asset_type=group.asset_type(group.benchmark),
        prices=prices[group.benchmark],
        start_date=config.start_date,
        end_date=config.end_date,
        initial_cash=config.initial_cash_twd,
        cost_model=config.cost_model,
        dividend_series=dividend_series,
    )


def _write_summary(results: list[BacktestResult], output_dir: Path) -> None:
    rows = [
        {
            "strategy": result.name,
            "final_value": round(result.final_value, 2),
            "total_return_pct": round(result.total_return * 100, 2),
            "max_drawdown_pct": round(result.max_drawdown * 100, 2),
            "trade_count": _execution_trade_count(result),
            "event_count": len(result.trades),
        }
        for result in results
    ]
    pd.DataFrame(rows).to_csv(output_dir / "strategies_summary.csv", index=False, encoding="utf-8-sig")


def _benchmark_reconciliation_row(
    group: GroupConfig,
    without_dividend: BacktestResult,
    with_dividend: BacktestResult,
    reference: dict[str, float | str] | None,
) -> dict:
    reference_final = float(reference["final_value"]) if reference and "final_value" in reference else None
    reference_diff = (
        round(with_dividend.final_value - reference_final, 2)
        if reference_final is not None
        else None
    )
    return {
        "group": group.group_id,
        "benchmark": group.benchmark,
        "without_dividend_final_value": round(without_dividend.final_value, 2),
        "with_adjusted_dividend_final_value": round(with_dividend.final_value, 2),
        "dividend_cash_added": round(with_dividend.final_value - without_dividend.final_value, 2),
        "reference_source": reference.get("source") if reference else "",
        "reference_final_value": reference_final,
        "diff_vs_reference": reference_diff,
    }


def _write_benchmark_reconciliation(rows: list[dict], output_dir: Path) -> None:
    pd.DataFrame(rows).to_csv(
        output_dir / "benchmark_reconciliation.csv",
        index=False,
        encoding="utf-8-sig",
    )


def _write_trades(results: list[BacktestResult], output_dir: Path) -> None:
    rows = []
    for result in results:
        for trade in result.trades:
            rows.append({"strategy": result.name, **_trade_row(trade)})
    pd.DataFrame(rows).to_csv(output_dir / "trade_log.csv", index=False, encoding="utf-8-sig")


def _write_equity_curves(results: list[BacktestResult], output_dir: Path) -> None:
    frames = []
    for result in results:
        frame = result.equity_curve.copy()
        frame["strategy"] = result.name
        frames.append(frame.reset_index())
    pd.concat(frames, ignore_index=True).to_csv(
        output_dir / "daily_equity_curve.csv",
        index=False,
        encoding="utf-8-sig",
    )


def _write_holding_exposure(results: list[BacktestResult], output_dir: Path) -> None:
    rows = []
    for result in results:
        if "current_ticker" not in result.equity_curve.columns:
            continue
        counts = result.equity_curve["current_ticker"].value_counts()
        total = int(counts.sum())
        for ticker, days in counts.items():
            rows.append(
                {
                    "strategy": result.name,
                    "ticker": ticker,
                    "days": int(days),
                    "day_share_pct": round(days / total * 100, 2),
                }
            )
    pd.DataFrame(rows).to_csv(output_dir / "holding_exposure.csv", index=False, encoding="utf-8-sig")


def _run_robustness_checks(config, prices: dict[str, pd.DataFrame], dividends: dict[str, pd.Series]) -> list[dict]:
    variants = [
        {
            "variant": "base_weekly_63_126",
            "momentum_windows": (63, 126),
            "trend_window": 126,
            "volatility_window": 20,
            "rebalance_frequency": "weekly",
            "start_date": config.start_date,
            "end_date": config.end_date,
            "exclude": None,
        },
        {
            "variant": "short_weekly_42_84",
            "momentum_windows": (42, 84),
            "trend_window": 84,
            "volatility_window": 20,
            "rebalance_frequency": "weekly",
            "start_date": config.start_date,
            "end_date": config.end_date,
            "exclude": None,
        },
        {
            "variant": "long_weekly_84_168",
            "momentum_windows": (84, 168),
            "trend_window": 168,
            "volatility_window": 20,
            "rebalance_frequency": "weekly",
            "start_date": config.start_date,
            "end_date": config.end_date,
            "exclude": None,
        },
        {
            "variant": "base_monthly_63_126",
            "momentum_windows": (63, 126),
            "trend_window": 126,
            "volatility_window": 20,
            "rebalance_frequency": "monthly",
            "start_date": config.start_date,
            "end_date": config.end_date,
            "exclude": None,
        },
        {
            "variant": "exclude_2454",
            "momentum_windows": (63, 126),
            "trend_window": 126,
            "volatility_window": 20,
            "rebalance_frequency": "weekly",
            "start_date": config.start_date,
            "end_date": config.end_date,
            "exclude": "2454.TW",
        },
        {
            "variant": "exclude_6669",
            "momentum_windows": (63, 126),
            "trend_window": 126,
            "volatility_window": 20,
            "rebalance_frequency": "weekly",
            "start_date": config.start_date,
            "end_date": config.end_date,
            "exclude": "6669.TW",
        },
        {
            "variant": "validation_2026_only",
            "momentum_windows": (63, 126),
            "trend_window": 126,
            "volatility_window": 20,
            "rebalance_frequency": "weekly",
            "start_date": "2026-01-01",
            "end_date": config.end_date,
            "exclude": None,
        },
    ]
    rows = []
    for group in config.groups:
        asset_types = {asset.ticker: asset.asset_type for asset in group.assets}
        for variant in variants:
            included_assets = [asset for asset in group.assets if asset.ticker != variant["exclude"]]
            if not included_assets:
                continue
            group_prices = {asset.ticker: prices[asset.ticker] for asset in included_assets}
            group_dividends = {asset.ticker: dividends[asset.ticker] for asset in included_assets}
            result = simulate_dual_momentum_vol_control(
                name=f"{group.group_id}__dual_momentum__{variant['variant']}",
                prices_by_ticker=group_prices,
                asset_types=asset_types,
                start_date=variant["start_date"],
                end_date=variant["end_date"],
                initial_cash=config.initial_cash_twd,
                cost_model=config.cost_model,
                dividend_series_by_ticker=group_dividends,
                momentum_windows=variant["momentum_windows"],
                trend_window=variant["trend_window"],
                volatility_window=variant["volatility_window"],
                rebalance_frequency=variant["rebalance_frequency"],
            )
            rows.append(
                {
                    "group": group.group_id,
                    "variant": variant["variant"],
                    "start_date": variant["start_date"],
                    "end_date": variant["end_date"],
                    "excluded_ticker": variant["exclude"] or "",
                    "final_value": round(result.final_value, 2),
                    "total_return_pct": round(result.total_return * 100, 2),
                    "max_drawdown_pct": round(result.max_drawdown * 100, 2),
                    "trade_count": _execution_trade_count(result),
                    "first_trade_ticker": _first_execution_ticker(result),
                }
            )
    return rows


def _write_robustness_summary(rows: list[dict], output_dir: Path) -> None:
    pd.DataFrame(rows).to_csv(output_dir / "robustness_summary.csv", index=False, encoding="utf-8-sig")


def _write_video_summary(results: list[BacktestResult], output_dir: Path) -> None:
    payload = {
        "episode": "EP05",
        "framing": "AI 輔助回測與策略驗證，非投資建議",
        "results": [
            {
                "strategy": result.name,
                "final_value": round(result.final_value, 2),
                "total_return_pct": round(result.total_return * 100, 2),
                "max_drawdown_pct": round(result.max_drawdown * 100, 2),
                "trade_count": _execution_trade_count(result),
                "event_count": len(result.trades),
                "first_trade": _trade_row(result.trades[0]) if result.trades else None,
            }
            for result in results
        ],
    }
    (output_dir / "video_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _trade_row(trade: Trade) -> dict:
    row = asdict(trade)
    row["gross_amount"] = round(row["gross_amount"], 2)
    row["cash_after"] = round(row["cash_after"], 2)
    return row


def _execution_trade_count(result: BacktestResult) -> int:
    return sum(1 for trade in result.trades if trade.action in {"buy", "sell"})


def _first_execution_ticker(result: BacktestResult) -> str:
    for trade in result.trades:
        if trade.action in {"buy", "sell"}:
            return trade.ticker
    return "cash"


if __name__ == "__main__":
    main()
