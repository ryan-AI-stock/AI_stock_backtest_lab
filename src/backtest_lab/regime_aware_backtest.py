from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from backtest_lab.config import load_config
from backtest_lab.data import download_yfinance_prices, split_adjusted_dividends
from backtest_lab.regime_aware_simulation import default_policy_variants, simulate_regime_aware_strategy
from backtest_lab.simulation import (
    BacktestResult,
    simulate_buy_and_hold,
    simulate_dual_momentum_vol_control,
    simulate_relative_strength_top1,
)


PERIODS = {
    "full_2020_2026": ("2020-01-01", "2026-05-26", "2020至2026完整樣本"),
    "ep05_2024_2026": ("2024-01-02", "2026-05-26", "EP05同區間多頭樣本"),
    "year_2021": ("2021-01-01", "2021-12-31", "2021多頭與突發風險診斷"),
    "period_2021_2022": ("2021-01-01", "2022-12-31", "2021至2022兩年壓力測試"),
    "period_2023_2024": ("2023-01-01", "2024-12-31", "2023至2024兩年修復與多頭測試"),
    "bear_2022": ("2022-01-01", "2022-12-31", "空頭壓力測試"),
    "year_2023": ("2023-01-01", "2023-12-31", "2023領漲行情診斷"),
    "year_2024": ("2024-01-01", "2024-12-31", "2024多頭延續診斷"),
    "bull_2025": ("2025-01-01", "2025-12-31", "多頭攻擊測試"),
    "recent_2025_2026": ("2025-01-01", "2026-05-26", "2025至2026近期實戰樣本"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run regime-aware strategy candidate backtests.")
    parser.add_argument("--config", default="configs/ep05_universe.json")
    parser.add_argument("--group-id", default="group_c_0050_00631l_plus_mega_caps")
    parser.add_argument("--cache-dir", default="backtest_cache/regime_aware")
    parser.add_argument("--output-dir", default="outputs/regime_aware_backtest")
    parser.add_argument("--market-proxy", default="0050.TW")
    parser.add_argument(
        "--periods",
        default="period_2021_2022,period_2023_2024",
        help=(
            "Comma-separated period ids. Available: "
            "full_2020_2026,period_2021_2022,period_2023_2024,bear_2022,bull_2025"
        ),
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

    summary_rows: list[dict] = []
    trade_rows: list[dict] = []
    for period_id, (start, end, period_label) in selected_periods.items():
        group_prices = {asset.ticker: prices[asset.ticker] for asset in group.assets}
        benchmark = simulate_buy_and_hold(
            name="0050_buy_and_hold",
            ticker=group.benchmark,
            asset_type=group.asset_type(group.benchmark),
            prices=prices[group.benchmark],
            start_date=start,
            end_date=end,
            initial_cash=config.initial_cash_twd,
            cost_model=config.cost_model,
            dividend_series=dividends[group.benchmark],
        )
        baseline_daily = simulate_relative_strength_top1(
            name="baseline_daily_strength",
            prices_by_ticker=group_prices,
            asset_types=asset_types,
            start_date=start,
            end_date=end,
            initial_cash=config.initial_cash_twd,
            cost_model=config.cost_model,
            dividend_series_by_ticker={ticker: dividends[ticker] for ticker in group_prices},
        )
        baseline_weekly = simulate_dual_momentum_vol_control(
            name="baseline_weekly_rotation",
            prices_by_ticker=group_prices,
            asset_types=asset_types,
            start_date=start,
            end_date=end,
            initial_cash=config.initial_cash_twd,
            cost_model=config.cost_model,
            dividend_series_by_ticker={ticker: dividends[ticker] for ticker in group_prices},
        )
        baselines = [
            ("0050買進持有", "benchmark", "baseline", benchmark),
            ("原始每日追強勢", "daily_strength", "baseline", baseline_daily),
            ("原始週頻輪動", "weekly_rotation", "baseline", baseline_weekly),
        ]
        for display_name, strategy_id, variant, result in baselines:
            summary_rows.append(_summary_row(period_id, period_label, display_name, strategy_id, variant, result))
            trade_rows.extend(_trade_rows(period_id, display_name, result, labels))

        for strategy_id, display_prefix in (
            ("daily_strength", "環境版每日追強勢"),
            ("weekly_rotation", "環境版週頻輪動"),
        ):
            for variant in default_policy_variants():
                result = simulate_regime_aware_strategy(
                    name=f"regime_aware_{strategy_id}_{variant.name}",
                    strategy_id=strategy_id,
                    prices_by_ticker=group_prices,
                    asset_types=asset_types,
                    market_prices=prices[args.market_proxy],
                    start_date=start,
                    end_date=end,
                    initial_cash=config.initial_cash_twd,
                    cost_model=config.cost_model,
                    variant=variant,
                )
                display_name = f"{display_prefix}_{variant.name}"
                summary_rows.append(_summary_row(period_id, period_label, display_name, strategy_id, variant.name, result))
                trade_rows.extend(_trade_rows(period_id, display_name, result, labels))

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "regime_aware_summary.csv", index=False, encoding="utf-8-sig")
    trades = pd.DataFrame(trade_rows)
    trades.to_csv(output_dir / "trade_log.csv", index=False, encoding="utf-8-sig")
    ranking = _aggregate_ranking(summary)
    ranking.to_csv(output_dir / "aggregate_ranking.csv", index=False, encoding="utf-8-sig")
    _write_report(output_dir, summary, ranking)
    (output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "market_proxy": args.market_proxy,
                "group_id": args.group_id,
                "periods": PERIODS,
                "selected_periods": selected_periods,
                "note": "AI輔助回測與策略驗證，不是投資建議。",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"OUTPUT_DIR={output_dir.resolve()}")


def _summary_row(
    period_id: str,
    period_label: str,
    display_name: str,
    strategy_id: str,
    variant: str,
    result: BacktestResult,
) -> dict:
    trade_count = sum(1 for trade in result.trades if trade.action in {"buy", "sell"})
    return {
        "period_id": period_id,
        "period_label": period_label,
        "strategy_name": display_name,
        "strategy_id": strategy_id,
        "variant": variant,
        "final_value_twd": round(result.final_value, 2),
        "total_return_pct": round(result.total_return * 100, 2),
        "max_drawdown_pct": round(result.max_drawdown * 100, 2),
        "trade_count": trade_count,
    }


def _selected_periods(periods_arg: str) -> dict[str, tuple[str, str, str]]:
    selected: dict[str, tuple[str, str, str]] = {}
    for period_id in [item.strip() for item in periods_arg.split(",") if item.strip()]:
        if period_id not in PERIODS:
            raise ValueError(f"Unsupported period id: {period_id}")
        selected[period_id] = PERIODS[period_id]
    if not selected:
        raise ValueError("At least one period is required")
    return selected


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


def _aggregate_ranking(summary: pd.DataFrame) -> pd.DataFrame:
    pivot = summary.pivot_table(
        index=["strategy_name", "strategy_id", "variant"],
        columns="period_id",
        values=["total_return_pct", "max_drawdown_pct", "trade_count"],
        aggfunc="first",
    )
    pivot.columns = [f"{metric}_{period}" for metric, period in pivot.columns]
    pivot = pivot.reset_index()
    period_ids = [column.removeprefix("total_return_pct_") for column in pivot.columns if column.startswith("total_return_pct_")]
    defensive_period = _first_available_period(period_ids, ("bear_2022", "period_2021_2022"))
    offensive_period = _first_available_period(period_ids, ("bull_2025", "period_2023_2024"))
    for col in tuple(_score_columns(defensive_period, offensive_period)):
        if col not in pivot:
            pivot[col] = 0.0
    pivot["defensive_score"] = pivot[f"total_return_pct_{defensive_period}"] + (
        0.8 * pivot[f"max_drawdown_pct_{defensive_period}"]
    )
    pivot["offensive_score"] = pivot[f"total_return_pct_{offensive_period}"] + (
        0.25 * pivot[f"max_drawdown_pct_{offensive_period}"]
    )
    pivot["bear_score"] = pivot["defensive_score"]
    pivot["bull_score"] = pivot["offensive_score"]
    pivot["composite_score"] = pivot["bear_score"] + pivot["bull_score"]
    return pivot.sort_values("composite_score", ascending=False)


def _first_available_period(period_ids: list[str], preferred: tuple[str, ...]) -> str:
    for period_id in preferred:
        if period_id in period_ids:
            return period_id
    if not period_ids:
        raise ValueError("No periods available for aggregate ranking")
    return period_ids[0]


def _score_columns(defensive_period: str, offensive_period: str) -> list[str]:
    return [
        f"total_return_pct_{defensive_period}",
        f"max_drawdown_pct_{defensive_period}",
        f"total_return_pct_{offensive_period}",
        f"max_drawdown_pct_{offensive_period}",
    ]


def _write_report(output_dir: Path, summary: pd.DataFrame, ranking: pd.DataFrame) -> None:
    period_labels = summary[["period_id", "period_label"]].drop_duplicates()
    period_text = "、".join(f"{row.period_label}（{row.period_id}）" for row in period_labels.itertuples())
    lines = [
        "# 市場環境版策略候選回測",
        "",
        f"本報告使用 {period_text} 做候選策略比較。這是 AI 輔助回測與策略驗證，不是投資建議。",
        "",
        "## 分段績效",
        "",
        _markdown_table(summary),
        "",
        "## 綜合排名",
        "",
        _markdown_table(ranking[_ranking_report_columns(ranking)]),
        "",
        "評分只是第一版排序工具：第一段偏防守，第二段偏進攻。正式採用前仍需用更多期間與六週 shadow mode 驗證。",
        "",
    ]
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def _ranking_report_columns(ranking: pd.DataFrame) -> list[str]:
    preferred = [
        "strategy_name",
        "variant",
        "total_return_pct_bear_2022",
        "max_drawdown_pct_bear_2022",
        "total_return_pct_bull_2025",
        "max_drawdown_pct_bull_2025",
        "total_return_pct_full_2020_2026",
        "max_drawdown_pct_full_2020_2026",
        "defensive_score",
        "offensive_score",
        "composite_score",
    ]
    return [column for column in preferred if column in ranking.columns]


def _markdown_table(frame: pd.DataFrame) -> str:
    headers = list(frame.columns)
    rows = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in frame.iterrows():
        rows.append("| " + " | ".join(str(row[col]) for col in headers) + " |")
    return "\n".join(rows)


if __name__ == "__main__":
    main()
