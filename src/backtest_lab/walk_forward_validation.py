from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from backtest_lab.config import load_config
from backtest_lab.cycle_proven_diversification import diversification_variants, simulate_cycle_proven_diversification
from backtest_lab.data import download_yfinance_prices, split_adjusted_dividends
from backtest_lab.regime_mode_switch_backtest import _load_sufficient_cache_prices
from backtest_lab.simulation import simulate_buy_and_hold


@dataclass(frozen=True)
class WalkForwardFold:
    fold_id: str
    train_start: str
    train_end: str
    test_start: str
    test_end: str


def walk_forward_folds() -> tuple[WalkForwardFold, ...]:
    return (
        WalkForwardFold("train_2020_2021_test_2022", "2020-01-01", "2021-12-31", "2022-01-01", "2022-12-31"),
        WalkForwardFold("train_2021_2022_test_2023", "2021-01-01", "2022-12-31", "2023-01-01", "2023-12-31"),
        WalkForwardFold("train_2022_2023_test_2024", "2022-01-01", "2023-12-31", "2024-01-01", "2024-12-31"),
        WalkForwardFold("train_2023_2024_test_2025", "2023-01-01", "2024-12-31", "2025-01-01", "2025-12-31"),
        WalkForwardFold(
            "train_2024_2025_test_2026_partial",
            "2024-01-01",
            "2025-12-31",
            "2026-01-01",
            "2026-05-26",
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run rolling two-year train and next-year test validation.")
    parser.add_argument("--config", default="configs/ep05_universe.json")
    parser.add_argument("--cache-dir", default="backtest_cache/unified_9_asset_full")
    parser.add_argument("--output-dir", default="outputs/walk_forward_validation")
    args = parser.parse_args()

    config = load_config(args.config)
    group = config.active_group
    asset_types = {asset.ticker: asset.asset_type for asset in group.assets}
    tickers = sorted(asset_types)
    folds = walk_forward_folds()
    start = min(pd.Timestamp(fold.train_start) for fold in folds)
    end = max(pd.Timestamp(fold.test_end) for fold in folds)
    cached = _load_sufficient_cache_prices(tickers, args.cache_dir, required_start=start, required_end=end)
    prices = download_yfinance_prices(
        [ticker for ticker in tickers if ticker not in cached],
        (start - pd.DateOffset(years=2)).strftime("%Y-%m-%d"),
        end.strftime("%Y-%m-%d"),
        args.cache_dir,
    )
    prices.update(cached)
    dividends = {
        ticker: split_adjusted_dividends(frame, config.manual_splits.get(ticker, ()))
        for ticker, frame in prices.items()
    }
    variants = diversification_variants()
    candidate_rows: list[dict] = []
    selected_rows: list[dict] = []

    for fold in folds:
        fold_results: dict[str, dict[str, float]] = {}
        for variant in variants:
            train = simulate_cycle_proven_diversification(
                name=f"{variant.name}_{fold.fold_id}_train",
                prices_by_ticker=prices,
                asset_types=asset_types,
                market_prices=prices["0050.TW"],
                start_date=fold.train_start,
                end_date=fold.train_end,
                initial_cash=config.initial_cash_twd,
                cost_model=config.cost_model,
                diversification=variant,
                dividend_series_by_ticker=dividends,
            )
            test = simulate_cycle_proven_diversification(
                name=f"{variant.name}_{fold.fold_id}_test",
                prices_by_ticker=prices,
                asset_types=asset_types,
                market_prices=prices["0050.TW"],
                start_date=fold.test_start,
                end_date=fold.test_end,
                initial_cash=config.initial_cash_twd,
                cost_model=config.cost_model,
                diversification=variant,
                dividend_series_by_ticker=dividends,
            )
            train_return = train.total_return * 100
            train_drawdown = train.max_drawdown * 100
            robust_score = train_return + train_drawdown
            fold_results[variant.name] = {
                "train_return_pct": train_return,
                "train_max_drawdown_pct": train_drawdown,
                "train_robust_score": robust_score,
                "test_return_pct": test.total_return * 100,
                "test_max_drawdown_pct": test.max_drawdown * 100,
            }
            candidate_rows.append(
                {
                    "fold_id": fold.fold_id,
                    "candidate_id": variant.name,
                    **fold_results[variant.name],
                }
            )
        simplicity_order = {variant.name: index for index, variant in enumerate(variants)}
        selected = max(
            fold_results.items(),
            key=lambda item: (item[1]["train_robust_score"], -simplicity_order[item[0]]),
        )
        benchmark_rows = {}
        for ticker in ("0050.TW", "00631L.TW"):
            benchmark = simulate_buy_and_hold(
                name=f"{ticker}_{fold.fold_id}_test",
                ticker=ticker,
                asset_type=asset_types[ticker],
                prices=prices[ticker],
                start_date=fold.test_start,
                end_date=fold.test_end,
                initial_cash=config.initial_cash_twd,
                cost_model=config.cost_model,
                dividend_series=dividends[ticker],
            )
            benchmark_rows[ticker] = benchmark.total_return * 100
        selected_rows.append(
            {
                "fold_id": fold.fold_id,
                "selected_candidate": selected[0],
                **selected[1],
                "test_0050_return_pct": benchmark_rows["0050.TW"],
                "test_00631l_return_pct": benchmark_rows["00631L.TW"],
                "beats_0050": selected[1]["test_return_pct"] > benchmark_rows["0050.TW"],
                "beats_00631l": selected[1]["test_return_pct"] > benchmark_rows["00631L.TW"],
            }
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = pd.DataFrame(candidate_rows)
    selected = pd.DataFrame(selected_rows)
    candidates.to_csv(output_dir / "walk_forward_candidates.csv", index=False, encoding="utf-8-sig")
    selected.to_csv(output_dir / "walk_forward_selected.csv", index=False, encoding="utf-8-sig")
    _write_report(output_dir / "walk_forward_report.md", candidates, selected)
    print(f"OUTPUT_DIR={output_dir.resolve()}")


def _write_report(path: Path, candidates: pd.DataFrame, selected: pd.DataFrame) -> None:
    lines = [
        "# Walk-forward 穩定性驗證",
        "",
        "固定使用前兩年比較候選，再把訓練期 robust score 最高者套用到下一年。這是歷史穩定性檢查；因規則研發已看過部分歷史資料，不宣稱為完全未見資料。",
        "",
        "robust score = 訓練期報酬率 + 訓練期最大回撤率。",
        "",
        "## 每折選中結果",
        "",
        _markdown_table(selected),
        "",
        "## 全候選結果",
        "",
        _markdown_table(candidates),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _markdown_table(frame: pd.DataFrame) -> str:
    headers = list(frame.columns)
    rows = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in frame.iterrows():
        rows.append("| " + " | ".join(str(row[column]) for column in headers) + " |")
    return "\n".join(rows)


if __name__ == "__main__":
    main()
