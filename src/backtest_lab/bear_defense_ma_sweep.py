from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from backtest_lab.bear_defense_backtest import (
    DEFENSE_TICKER,
    DefenseVariant,
    _load_best_cached_prices,
    simulate_0050_defense,
)
from backtest_lab.config import load_config
from backtest_lab.data import download_yfinance_prices, split_adjusted_dividends
from backtest_lab.regime_aware_backtest import PERIODS
from backtest_lab.simulation import BacktestResult, simulate_buy_and_hold


DEFAULT_PERIODS = "period_2021_2022,period_2023_2024,ep05_2024_2026"


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep 0050 moving-average bear-defense rules.")
    parser.add_argument("--config", default="configs/ep05_universe.json")
    parser.add_argument("--cache-dir", default="backtest_cache/regime_aware_full")
    parser.add_argument("--output-dir", default="outputs/bear_defense_ma_sweep")
    parser.add_argument("--periods", default=DEFAULT_PERIODS)
    parser.add_argument("--ma-start", type=int, default=180)
    parser.add_argument("--ma-end", type=int, default=320)
    parser.add_argument("--ma-step", type=int, default=10)
    args = parser.parse_args()

    config = load_config(args.config)
    selected_periods = _selected_periods(args.periods)
    start_for_download = min(pd.Timestamp(start) for start, _, _ in selected_periods.values())
    end_for_download = max(pd.Timestamp(end) for _, end, _ in selected_periods.values())
    download_start = (start_for_download - pd.DateOffset(years=2)).strftime("%Y-%m-%d")
    end = end_for_download.strftime("%Y-%m-%d")
    prices = _load_best_cached_prices(args.cache_dir, download_start, end)
    if prices is None:
        prices_by_ticker = download_yfinance_prices(
            tickers=[DEFENSE_TICKER],
            start_date=download_start,
            end_date=end,
            cache_dir=args.cache_dir,
        )
        prices = prices_by_ticker[DEFENSE_TICKER]
    dividends = split_adjusted_dividends(prices, config.manual_splits.get(DEFENSE_TICKER, ()))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    variants = _ma_variants(args.ma_start, args.ma_end, args.ma_step)
    summary_rows: list[dict] = []
    for period_id, (start, period_end, period_label) in selected_periods.items():
        benchmark = simulate_buy_and_hold(
            name="0050_buy_and_hold",
            ticker=DEFENSE_TICKER,
            asset_type="etf",
            prices=prices,
            start_date=start,
            end_date=period_end,
            initial_cash=config.initial_cash_twd,
            cost_model=config.cost_model,
            dividend_series=dividends,
        )
        summary_rows.append(_summary_row(period_id, period_label, "0050買進持有", "buy_and_hold", benchmark))
        for variant in variants:
            result = simulate_0050_defense(
                name=variant.name,
                prices=prices,
                start_date=start,
                end_date=period_end,
                initial_cash=config.initial_cash_twd,
                cost_model=config.cost_model,
                variant=variant,
                dividend_series=dividends,
            )
            summary_rows.append(_summary_row(period_id, period_label, f"0050防守_{variant.name}", variant.name, result))

    summary = pd.DataFrame(summary_rows)
    ranking = _ranking(summary)
    summary.to_csv(output_dir / "ma_sweep_summary.csv", index=False, encoding="utf-8-sig")
    ranking.to_csv(output_dir / "ma_sweep_ranking.csv", index=False, encoding="utf-8-sig")
    _write_report(output_dir / "ma_sweep_report.md", summary, ranking)
    (output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "ticker": DEFENSE_TICKER,
                "periods": selected_periods,
                "ma_start": args.ma_start,
                "ma_end": args.ma_end,
                "ma_step": args.ma_step,
                "note": "AI輔助回測與策略驗證，不是投資建議。",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"OUTPUT_DIR={output_dir.resolve()}")


def _ma_variants(start: int, end: int, step: int) -> tuple[DefenseVariant, ...]:
    if step <= 0:
        raise ValueError("ma-step must be positive")
    return tuple(
        DefenseVariant(name=f"ma{window}_cash", rule=f"ma{window}", risk_off_exposure=0.0)
        for window in range(start, end + 1, step)
    )


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
    variant: str,
    result: BacktestResult,
) -> dict:
    return {
        "period_id": period_id,
        "period_label": period_label,
        "strategy_name": strategy_name,
        "variant": variant,
        "final_value_twd": round(result.final_value, 2),
        "total_return_pct": round(result.total_return * 100, 2),
        "max_drawdown_pct": round(result.max_drawdown * 100, 2),
        "trade_count": sum(1 for trade in result.trades if trade.action in {"buy", "sell"}),
    }


def _ranking(summary: pd.DataFrame) -> pd.DataFrame:
    pivot = summary.pivot_table(
        index=["strategy_name", "variant"],
        columns="period_id",
        values=["total_return_pct", "max_drawdown_pct", "trade_count"],
        aggfunc="first",
    )
    pivot.columns = [f"{metric}_{period}" for metric, period in pivot.columns]
    pivot = pivot.reset_index()
    if "total_return_pct_period_2021_2022" in pivot and "max_drawdown_pct_period_2021_2022" in pivot:
        pivot["small_bear_score"] = pivot["total_return_pct_period_2021_2022"] + (
            0.6 * pivot["max_drawdown_pct_period_2021_2022"]
        )
        return pivot.sort_values(["small_bear_score", "total_return_pct_period_2021_2022"], ascending=False)
    return pivot


def _write_report(path: Path, summary: pd.DataFrame, ranking: pd.DataFrame) -> None:
    lines = [
        "# 0050 MA 空頭防守參數掃描",
        "",
        "本報告掃描 0050 長期均線防守參數，重點檢查 2021-2022 小空頭是否有穩定區間，而不是單一參數剛好漂亮。這是 AI 輔助回測與策略驗證，不是投資建議。",
        "",
        "## 小空頭優先排名",
        "",
        _markdown_table(ranking),
        "",
        "## 分段結果",
        "",
        _markdown_table(summary),
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
