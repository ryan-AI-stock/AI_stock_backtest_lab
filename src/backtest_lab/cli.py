from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from backtest_lab.config import GroupConfig, load_config
from backtest_lab.data import download_yfinance_prices
from backtest_lab.portfolio import Trade
from backtest_lab.simulation import (
    BacktestResult,
    simulate_buy_and_hold,
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

    results: list[BacktestResult] = []
    for group in config.groups:
        results.append(_run_benchmark(group, prices, config))
        group_prices = {asset.ticker: prices[asset.ticker] for asset in group.assets}
        asset_types = {asset.ticker: asset.asset_type for asset in group.assets}
        results.append(
            simulate_relative_strength_top1(
                name=f"{group.group_id}__relative_strength_top1",
                prices_by_ticker=group_prices,
                asset_types=asset_types,
                start_date=config.start_date,
                end_date=config.end_date,
                initial_cash=config.initial_cash_twd,
                cost_model=config.cost_model,
            )
        )

    _write_summary(results, output_dir)
    _write_trades(results, output_dir)
    _write_equity_curves(results, output_dir)
    _write_video_summary(results, output_dir)


def _run_benchmark(group: GroupConfig, prices: dict[str, pd.DataFrame], config) -> BacktestResult:
    return simulate_buy_and_hold(
        name=f"{group.group_id}__benchmark__{group.benchmark}",
        ticker=group.benchmark,
        asset_type=group.asset_type(group.benchmark),
        prices=prices[group.benchmark],
        start_date=config.start_date,
        end_date=config.end_date,
        initial_cash=config.initial_cash_twd,
        cost_model=config.cost_model,
    )


def _write_summary(results: list[BacktestResult], output_dir: Path) -> None:
    rows = [
        {
            "strategy": result.name,
            "final_value": round(result.final_value, 2),
            "total_return_pct": round(result.total_return * 100, 2),
            "max_drawdown_pct": round(result.max_drawdown * 100, 2),
            "trade_count": len(result.trades),
        }
        for result in results
    ]
    pd.DataFrame(rows).to_csv(output_dir / "strategies_summary.csv", index=False, encoding="utf-8-sig")


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
                "trade_count": len(result.trades),
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


if __name__ == "__main__":
    main()

