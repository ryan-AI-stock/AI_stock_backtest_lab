from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import pandas as pd

from backtest_lab.config import load_config
from backtest_lab.data import download_yfinance_prices, split_adjusted_dividends
from backtest_lab.regime_aware_backtest import PERIODS
from backtest_lab.regime_aware_simulation import default_policy_variants, simulate_regime_aware_strategy
from backtest_lab.simulation import BacktestResult, simulate_dual_momentum_vol_control


WEEKDAYS = {
    0: ("mon", "週一收盤判斷，次交易日開盤執行"),
    1: ("tue", "週二收盤判斷，次交易日開盤執行"),
    2: ("wed", "週三收盤判斷，次交易日開盤執行"),
    3: ("thu", "週四收盤判斷，次交易日開盤執行"),
    4: ("fri", "週五收盤判斷，次交易日開盤執行"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest weekly strategies by signal weekday.")
    parser.add_argument("--config", default="configs/ep05_universe.json")
    parser.add_argument("--group-id", default="group_c_0050_00631l_plus_mega_caps")
    parser.add_argument("--cache-dir", default="backtest_cache/regime_aware_full")
    parser.add_argument("--output-dir", default="outputs/weekly_signal_day_backtest")
    parser.add_argument("--market-proxy", default="0050.TW")
    parser.add_argument("--periods", default="period_2021_2022,period_2023_2024")
    parser.add_argument(
        "--variant-names",
        default="balanced,bear_guard,strong_only_guard,ultra_score25_stop_6_cd10",
        help="Comma-separated regime-aware weekly variants to sweep.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    group = next(group for group in config.groups if group.group_id == args.group_id)
    labels = {asset.ticker: asset.label for asset in group.assets}
    asset_types = {asset.ticker: asset.asset_type for asset in group.assets}
    tickers = sorted({asset.ticker for asset in group.assets} | {args.market_proxy})
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_periods = _selected_periods(args.periods)
    start_for_download = min(pd.Timestamp(start) for start, _, _ in selected_periods.values())
    end_for_download = max(pd.Timestamp(end) for _, end, _ in selected_periods.values())
    prices = download_yfinance_prices(
        tickers=tickers,
        start_date=(start_for_download - pd.DateOffset(years=2)).strftime("%Y-%m-%d"),
        end_date=end_for_download.strftime("%Y-%m-%d"),
        cache_dir=args.cache_dir,
    )
    dividends = {ticker: split_adjusted_dividends(prices[ticker], config.manual_splits.get(ticker, ())) for ticker in tickers}
    group_prices = {asset.ticker: prices[asset.ticker] for asset in group.assets}
    variants_by_name = {variant.name: variant for variant in default_policy_variants()}
    selected_variant_names = [name.strip() for name in args.variant_names.split(",") if name.strip()]

    summary_rows: list[dict] = []
    trade_rows: list[dict] = []
    for period_id, (start, end, period_label) in selected_periods.items():
        for weekday_index, (weekday_id, weekday_label) in WEEKDAYS.items():
            baseline = simulate_dual_momentum_vol_control(
                name=f"baseline_weekly_{weekday_id}",
                prices_by_ticker=group_prices,
                asset_types=asset_types,
                start_date=start,
                end_date=end,
                initial_cash=config.initial_cash_twd,
                cost_model=config.cost_model,
                dividend_series_by_ticker={ticker: dividends[ticker] for ticker in group_prices},
                signal_weekday=weekday_index,
            )
            display_name = f"原始週頻輪動_{weekday_label}"
            summary_rows.append(
                _summary_row(period_id, period_label, display_name, "baseline_weekly", "baseline", weekday_index, weekday_label, baseline)
            )
            trade_rows.extend(_trade_rows(period_id, display_name, baseline, labels))

            for variant_name in selected_variant_names:
                if variant_name not in variants_by_name:
                    raise ValueError(f"Unknown variant name: {variant_name}")
                variant = replace(
                    variants_by_name[variant_name],
                    name=f"{variant_name}_{weekday_id}",
                    weekly_signal_weekday=weekday_index,
                )
                result = simulate_regime_aware_strategy(
                    name=f"regime_aware_weekly_{variant.name}",
                    strategy_id="weekly_rotation",
                    prices_by_ticker=group_prices,
                    asset_types=asset_types,
                    market_prices=prices[args.market_proxy],
                    start_date=start,
                    end_date=end,
                    initial_cash=config.initial_cash_twd,
                    cost_model=config.cost_model,
                    variant=variant,
                )
                display_name = f"環境版週頻輪動_{variant_name}_{weekday_label}"
                summary_rows.append(
                    _summary_row(period_id, period_label, display_name, "regime_aware_weekly", variant_name, weekday_index, weekday_label, result)
                )
                trade_rows.extend(_trade_rows(period_id, display_name, result, labels))

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "weekly_signal_day_summary.csv", index=False, encoding="utf-8-sig")
    trades = pd.DataFrame(trade_rows)
    trades.to_csv(output_dir / "weekly_signal_day_trades.csv", index=False, encoding="utf-8-sig")
    ranking = _ranking(summary)
    ranking.to_csv(output_dir / "weekly_signal_day_ranking.csv", index=False, encoding="utf-8-sig")
    _write_report(output_dir / "weekly_signal_day_report.md", summary, ranking)
    (output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "group_id": args.group_id,
                "market_proxy": args.market_proxy,
                "periods": selected_periods,
                "variant_names": selected_variant_names,
                "note": "AI輔助回測與策略驗證，不是投資建議。",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"OUTPUT_DIR={output_dir.resolve()}")


def _selected_periods(periods_arg: str) -> dict[str, tuple[str, str, str]]:
    selected: dict[str, tuple[str, str, str]] = {}
    for period_id in [item.strip() for item in periods_arg.split(",") if item.strip()]:
        if period_id not in PERIODS:
            raise ValueError(f"Unsupported period id: {period_id}")
        selected[period_id] = PERIODS[period_id]
    if not selected:
        raise ValueError("At least one period is required")
    return selected


def _summary_row(
    period_id: str,
    period_label: str,
    strategy_name: str,
    strategy_id: str,
    variant: str,
    signal_weekday: int,
    signal_weekday_label: str,
    result: BacktestResult,
) -> dict:
    return {
        "period_id": period_id,
        "period_label": period_label,
        "strategy_name": strategy_name,
        "strategy_id": strategy_id,
        "variant": variant,
        "signal_weekday": signal_weekday,
        "signal_weekday_label": signal_weekday_label,
        "final_value_twd": round(result.final_value, 2),
        "total_return_pct": round(result.total_return * 100, 2),
        "max_drawdown_pct": round(result.max_drawdown * 100, 2),
        "trade_count": sum(1 for trade in result.trades if trade.action in {"buy", "sell"}),
    }


def _trade_rows(period_id: str, strategy_name: str, result: BacktestResult, labels: dict[str, str]) -> list[dict]:
    rows = []
    for index, trade in enumerate(result.trades, start=1):
        rows.append(
            {
                "period_id": period_id,
                "strategy_name": strategy_name,
                "sequence": index,
                "date": trade.date,
                "ticker": trade.ticker,
                "label": labels.get(trade.ticker, trade.ticker),
                "action": trade.action,
                "shares": trade.shares,
                "price": round(trade.price, 4),
                "gross_amount_twd": round(trade.gross_amount, 2),
                "costs_twd": trade.costs,
                "cash_after_twd": round(trade.cash_after, 2),
                "reason": trade.reason,
            }
        )
    return rows


def _ranking(summary: pd.DataFrame) -> pd.DataFrame:
    pivot = summary.pivot_table(
        index=["strategy_name", "strategy_id", "variant", "signal_weekday", "signal_weekday_label"],
        columns="period_id",
        values=["total_return_pct", "max_drawdown_pct", "trade_count"],
        aggfunc="first",
    )
    pivot.columns = [f"{metric}_{period}" for metric, period in pivot.columns]
    pivot = pivot.reset_index()
    return pivot.sort_values(
        [column for column in pivot.columns if column.startswith("total_return_pct_")],
        ascending=False,
    )


def _write_report(path: Path, summary: pd.DataFrame, ranking: pd.DataFrame) -> None:
    lines = [
        "# 週頻訊號星期幾回測",
        "",
        "比較週一到週五收盤判斷、次交易日開盤執行的週頻策略結果。這是 AI 輔助回測與策略驗證，不是投資建議。",
        "",
        "## 分段結果",
        "",
        _markdown_table(summary),
        "",
        "## 排名",
        "",
        _markdown_table(ranking),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _markdown_table(frame: pd.DataFrame) -> str:
    headers = list(frame.columns)
    rows = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in frame.iterrows():
        rows.append("| " + " | ".join(str(row[col]) for col in headers) + " |")
    return "\n".join(rows)


if __name__ == "__main__":
    main()
