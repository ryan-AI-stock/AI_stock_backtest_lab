from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from backtest_lab.config import load_config
from backtest_lab.cycle_proven_diversification import diversification_variants, simulate_cycle_proven_diversification
from backtest_lab.data import download_yfinance_prices, split_adjusted_dividends
from backtest_lab.regime_aware_backtest import PERIODS
from backtest_lab.regime_mode_switch_backtest import _load_sufficient_cache_prices


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest cycle-proven diversified attack portfolios.")
    parser.add_argument("--config", default="configs/ep05_universe.json")
    parser.add_argument("--cache-dir", default="backtest_cache/unified_9_asset_full")
    parser.add_argument("--output-dir", default="outputs/cycle_proven_diversification")
    parser.add_argument("--periods", default="period_2021_2022,period_2023_2024")
    parser.add_argument("--exclude-tickers", default="")
    args = parser.parse_args()

    config = load_config(args.config)
    group = config.active_group
    asset_types = {asset.ticker: asset.asset_type for asset in group.assets}
    tickers = sorted(asset_types)
    periods = {period: PERIODS[period] for period in args.periods.split(",") if period}
    start = min(pd.Timestamp(values[0]) for values in periods.values())
    end = max(pd.Timestamp(values[1]) for values in periods.values())
    cached = _load_sufficient_cache_prices(tickers, args.cache_dir, required_start=start, required_end=end)
    prices = download_yfinance_prices(
        [ticker for ticker in tickers if ticker not in cached],
        (start - pd.DateOffset(years=2)).strftime("%Y-%m-%d"),
        end.strftime("%Y-%m-%d"),
        args.cache_dir,
    )
    prices.update(cached)
    excluded = {ticker.strip() for ticker in args.exclude_tickers.split(",") if ticker.strip()}
    group_prices = {ticker: frame for ticker, frame in prices.items() if ticker not in excluded}
    dividends = {
        ticker: split_adjusted_dividends(frame, config.manual_splits.get(ticker, ()))
        for ticker, frame in group_prices.items()
    }
    rows: list[dict] = []
    trade_rows: list[dict] = []
    for period_id, (period_start, period_end, label) in periods.items():
        for variant in diversification_variants():
            result = simulate_cycle_proven_diversification(
                name=variant.name,
                prices_by_ticker=group_prices,
                asset_types=asset_types,
                market_prices=prices["0050.TW"],
                start_date=period_start,
                end_date=period_end,
                initial_cash=config.initial_cash_twd,
                cost_model=config.cost_model,
                diversification=variant,
                dividend_series_by_ticker=dividends,
            )
            rows.append(
                {
                    "period_id": period_id,
                    "period_label": label,
                    "candidate_id": variant.name,
                    "final_value_twd": round(result.final_value, 2),
                    "total_return_pct": round(result.total_return * 100, 2),
                    "max_drawdown_pct": round(result.max_drawdown * 100, 2),
                    "trade_count": sum(trade.action in {"buy", "sell"} for trade in result.trades),
                }
            )
            for sequence, trade in enumerate(result.trades, start=1):
                trade_rows.append(
                    {
                        "period_id": period_id,
                        "candidate_id": variant.name,
                        "sequence": sequence,
                        **trade.__dict__,
                    }
                )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "diversification_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(trade_rows).to_csv(output_dir / "diversification_trades.csv", index=False, encoding="utf-8-sig")
    (output_dir / "metadata.json").write_text(
        json.dumps({"periods": periods, "excluded_tickers": sorted(excluded)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"OUTPUT_DIR={output_dir.resolve()}")


if __name__ == "__main__":
    main()
