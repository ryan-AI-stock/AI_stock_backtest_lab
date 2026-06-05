from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import pandas as pd

from backtest_lab.config import load_config
from backtest_lab.data import download_yfinance_prices, load_price_csv, split_adjusted_dividends
from backtest_lab.regime_aware_backtest import PERIODS
from backtest_lab.regime_aware_simulation import default_policy_variants, simulate_regime_aware_strategy
from backtest_lab.simulation import (
    BacktestResult,
    simulate_buy_and_hold,
    simulate_dual_momentum_vol_control,
    simulate_relative_strength_top1,
)
from backtest_lab.trade_diagnostics import build_closed_trade_diagnostics, summarize_closed_trades
from backtest_lab.weekly_signal_day_backtest import WEEKDAYS


DEFAULT_PERIODS = "period_2021_2022,period_2023_2024"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a unified strategy validation matrix.")
    parser.add_argument("--config", default="configs/ep05_universe.json")
    parser.add_argument("--group-id", default="group_c_0050_00631l_plus_mega_caps")
    parser.add_argument("--cache-dir", default="backtest_cache/regime_aware_full")
    parser.add_argument("--output-dir", default="outputs/strategy_validation_matrix")
    parser.add_argument("--market-proxy", default="0050.TW")
    parser.add_argument(
        "--reference-benchmarks",
        default="00631L.TW:0050正二買進持有",
        help="Comma-separated ticker:label references used for benchmark gates.",
    )
    parser.add_argument("--periods", default=DEFAULT_PERIODS)
    parser.add_argument(
        "--variant-names",
        default="all",
        help="Comma-separated regime-aware variants, or 'all'.",
    )
    parser.add_argument(
        "--weekly-weekdays",
        default="all",
        help="Comma-separated weekday indexes 0-4, 'all', or 'none'.",
    )
    parser.add_argument("--top-diagnostics", type=int, default=12)
    args = parser.parse_args()

    config = load_config(args.config)
    group = next(group for group in config.groups if group.group_id == args.group_id)
    labels = {asset.ticker: asset.label for config_group in config.groups for asset in config_group.assets}
    asset_types = {asset.ticker: asset.asset_type for config_group in config.groups for asset in config_group.assets}
    reference_benchmarks = _parse_reference_benchmarks(args.reference_benchmarks)
    labels.update({ticker: label for ticker, label in reference_benchmarks})
    asset_types.update({ticker: "etf" for ticker, _ in reference_benchmarks if ticker not in asset_types})
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
    prices.update(_load_reference_prices(reference_benchmarks, args.cache_dir))
    dividends = {
        ticker: split_adjusted_dividends(frame, config.manual_splits.get(ticker, ())) for ticker, frame in prices.items()
    }
    group_prices = {asset.ticker: prices[asset.ticker] for asset in group.assets}
    variants = _selected_variants(args.variant_names)
    weekly_weekdays = _selected_weekdays(args.weekly_weekdays)

    summary_rows: list[dict] = []
    trade_rows: list[dict] = []
    for period_id, (start, end, period_label) in selected_periods.items():
        period_results = _run_period(
            period_id=period_id,
            period_label=period_label,
            start=start,
            end=end,
            group_prices=group_prices,
            all_prices=prices,
            market_prices=prices[args.market_proxy],
            benchmark_ticker=group.benchmark,
            asset_types=asset_types,
            initial_cash=config.initial_cash_twd,
            cost_model=config.cost_model,
            dividends=dividends,
            variants=variants,
            weekly_weekdays=weekly_weekdays,
            reference_benchmarks=reference_benchmarks,
        )
        for candidate, result in period_results:
            summary_rows.append(_summary_row(period_id, period_label, candidate, result))
            trade_rows.extend(_trade_rows(period_id, candidate, result, labels))

    summary = pd.DataFrame(summary_rows)
    trades = pd.DataFrame(trade_rows)
    ranking = _rank_candidates(summary)
    diagnostics = _top_candidate_diagnostics(trades, ranking, args.top_diagnostics)

    summary.to_csv(output_dir / "strategy_validation_summary.csv", index=False, encoding="utf-8-sig")
    ranking.to_csv(output_dir / "strategy_validation_ranking.csv", index=False, encoding="utf-8-sig")
    trades.to_csv(output_dir / "strategy_validation_trades.csv", index=False, encoding="utf-8-sig")
    diagnostics.to_csv(output_dir / "top_closed_trade_summary.csv", index=False, encoding="utf-8-sig")
    _write_report(output_dir / "strategy_validation_report.md", summary, ranking, diagnostics)
    (output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "group_id": args.group_id,
                "market_proxy": args.market_proxy,
                "reference_benchmarks": reference_benchmarks,
                "periods": selected_periods,
                "variant_names": [variant.name for variant in variants],
                "weekly_weekdays": weekly_weekdays,
                "note": "AI輔助回測與策略驗證，不是投資建議。",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"OUTPUT_DIR={output_dir.resolve()}")


def _run_period(
    *,
    period_id: str,
    period_label: str,
    start: str,
    end: str,
    group_prices: dict[str, pd.DataFrame],
    all_prices: dict[str, pd.DataFrame],
    market_prices: pd.DataFrame,
    benchmark_ticker: str,
    asset_types: dict[str, str],
    initial_cash: float,
    cost_model,
    dividends: dict[str, pd.Series],
    variants,
    weekly_weekdays: tuple[int, ...],
    reference_benchmarks: tuple[tuple[str, str], ...],
) -> list[tuple[dict, BacktestResult]]:
    results: list[tuple[dict, BacktestResult]] = []
    benchmark = simulate_buy_and_hold(
        name="0050_buy_and_hold",
        ticker=benchmark_ticker,
        asset_type=asset_types[benchmark_ticker],
        prices=group_prices[benchmark_ticker],
        start_date=start,
        end_date=end,
        initial_cash=initial_cash,
        cost_model=cost_model,
        dividend_series=dividends[benchmark_ticker],
    )
    results.append((_candidate("benchmark_0050", "0050買進持有", "benchmark", "baseline", None), benchmark))
    for ticker, label in reference_benchmarks:
        if ticker not in all_prices or not _covers_period(all_prices[ticker], start, end):
            continue
        reference = simulate_buy_and_hold(
            name=f"{ticker}_buy_and_hold",
            ticker=ticker,
            asset_type=asset_types[ticker],
            prices=all_prices[ticker],
            start_date=start,
            end_date=end,
            initial_cash=initial_cash,
            cost_model=cost_model,
            dividend_series=dividends[ticker],
        )
        results.append(
            (
                _candidate(
                    f"benchmark_{ticker.replace('.', '_')}",
                    label,
                    "benchmark",
                    "baseline",
                    None,
                ),
                reference,
            )
        )

    daily = simulate_relative_strength_top1(
        name="baseline_daily_strength",
        prices_by_ticker=group_prices,
        asset_types=asset_types,
        start_date=start,
        end_date=end,
        initial_cash=initial_cash,
        cost_model=cost_model,
        dividend_series_by_ticker={ticker: dividends[ticker] for ticker in group_prices},
    )
    results.append((_candidate("baseline_daily_strength", "原始每日追強勢", "daily_strength", "baseline", None), daily))

    weekly = simulate_dual_momentum_vol_control(
        name="baseline_weekly_rotation",
        prices_by_ticker=group_prices,
        asset_types=asset_types,
        start_date=start,
        end_date=end,
        initial_cash=initial_cash,
        cost_model=cost_model,
        dividend_series_by_ticker={ticker: dividends[ticker] for ticker in group_prices},
    )
    results.append((_candidate("baseline_weekly_rotation", "原始週頻輪動_預設週初檢查", "weekly_rotation", "baseline", None), weekly))

    for weekday in weekly_weekdays:
        weekday_id, weekday_label = WEEKDAYS[weekday]
        weekday_result = simulate_dual_momentum_vol_control(
            name=f"baseline_weekly_rotation_{weekday_id}",
            prices_by_ticker=group_prices,
            asset_types=asset_types,
            start_date=start,
            end_date=end,
            initial_cash=initial_cash,
            cost_model=cost_model,
            dividend_series_by_ticker={ticker: dividends[ticker] for ticker in group_prices},
            signal_weekday=weekday,
        )
        results.append(
            (
                _candidate(
                    f"baseline_weekly_rotation_{weekday_id}",
                    f"原始週頻輪動_{weekday_label}",
                    "weekly_rotation",
                    "baseline",
                    weekday,
                ),
                weekday_result,
            )
        )

    for variant in variants:
        daily_variant = simulate_regime_aware_strategy(
            name=f"regime_daily_{variant.name}",
            strategy_id="daily_strength",
            prices_by_ticker=group_prices,
            asset_types=asset_types,
            market_prices=market_prices,
            start_date=start,
            end_date=end,
            initial_cash=initial_cash,
            cost_model=cost_model,
            variant=variant,
        )
        results.append(
            (
                _candidate(
                    f"regime_daily_{variant.name}",
                    f"環境版每日追強勢_{variant.name}",
                    "daily_strength",
                    variant.name,
                    None,
                ),
                daily_variant,
            )
        )

        weekly_variant = simulate_regime_aware_strategy(
            name=f"regime_weekly_{variant.name}",
            strategy_id="weekly_rotation",
            prices_by_ticker=group_prices,
            asset_types=asset_types,
            market_prices=market_prices,
            start_date=start,
            end_date=end,
            initial_cash=initial_cash,
            cost_model=cost_model,
            variant=variant,
        )
        results.append(
            (
                _candidate(
                    f"regime_weekly_{variant.name}",
                    f"環境版週頻輪動_{variant.name}_預設週初檢查",
                    "weekly_rotation",
                    variant.name,
                    None,
                ),
                weekly_variant,
            )
        )
        for weekday in weekly_weekdays:
            weekday_id, weekday_label = WEEKDAYS[weekday]
            weekday_variant = replace(variant, name=f"{variant.name}_{weekday_id}", weekly_signal_weekday=weekday)
            weekday_result = simulate_regime_aware_strategy(
                name=f"regime_weekly_{weekday_variant.name}",
                strategy_id="weekly_rotation",
                prices_by_ticker=group_prices,
                asset_types=asset_types,
                market_prices=market_prices,
                start_date=start,
                end_date=end,
                initial_cash=initial_cash,
                cost_model=cost_model,
                variant=weekday_variant,
            )
            results.append(
                (
                    _candidate(
                        f"regime_weekly_{variant.name}_{weekday_id}",
                        f"環境版週頻輪動_{variant.name}_{weekday_label}",
                        "weekly_rotation",
                        variant.name,
                        weekday,
                    ),
                    weekday_result,
                )
            )
    return results


def _candidate(
    candidate_id: str,
    strategy_name: str,
    strategy_id: str,
    variant: str,
    signal_weekday: int | None,
) -> dict:
    return {
        "candidate_id": candidate_id,
        "strategy_name": strategy_name,
        "strategy_id": strategy_id,
        "variant": variant,
        "signal_weekday": str(signal_weekday) if signal_weekday is not None else "default",
        "signal_weekday_label": WEEKDAYS[signal_weekday][1] if signal_weekday is not None else "預設",
    }


def _summary_row(period_id: str, period_label: str, candidate: dict, result: BacktestResult) -> dict:
    row = {
        "period_id": period_id,
        "period_label": period_label,
        **candidate,
        "final_value_twd": round(result.final_value, 2),
        "total_return_pct": round(result.total_return * 100, 2),
        "max_drawdown_pct": round(result.max_drawdown * 100, 2),
        "trade_count": sum(1 for trade in result.trades if trade.action in {"buy", "sell"}),
    }
    return row


def _trade_rows(period_id: str, candidate: dict, result: BacktestResult, labels: dict[str, str]) -> list[dict]:
    rows = []
    for index, trade in enumerate(result.trades, start=1):
        rows.append(
            {
                "period_id": period_id,
                "strategy_name": candidate["candidate_id"],
                "display_name": candidate["strategy_name"],
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


def _rank_candidates(summary: pd.DataFrame) -> pd.DataFrame:
    index_cols = ["candidate_id", "strategy_name", "strategy_id", "variant", "signal_weekday", "signal_weekday_label"]
    pivot = summary.pivot_table(
        index=index_cols,
        columns="period_id",
        values=["total_return_pct", "max_drawdown_pct", "trade_count"],
        aggfunc="first",
    )
    pivot.columns = [f"{metric}_{period}" for metric, period in pivot.columns]
    pivot = pivot.reset_index()
    pivot = _add_missing_metric_columns(pivot)
    benchmark = _benchmark_lookup(summary)
    leveraged_0050 = benchmark.get("00631L.TW", {}).get("ep05_2024_2026")
    original_weekly_ep05 = benchmark.get("baseline_weekly_rotation", {}).get("ep05_2024_2026")
    original_daily_ep05 = benchmark.get("baseline_daily_strength", {}).get("ep05_2024_2026")
    pivot["beats_0050_2021_2022"] = pivot["total_return_pct_period_2021_2022"] > benchmark.get("period_2021_2022", -9999)
    pivot["beats_0050_2023_2024"] = pivot["total_return_pct_period_2023_2024"] > benchmark.get("period_2023_2024", -9999)
    pivot["beats_0050_ep05"] = pivot["total_return_pct_ep05_2024_2026"] > benchmark.get("ep05_2024_2026", -9999)
    pivot["beats_00631L_ep05"] = (
        pivot["total_return_pct_ep05_2024_2026"] > leveraged_0050 if leveraged_0050 is not None else False
    )
    pivot["beats_original_weekly_ep05"] = (
        pivot["total_return_pct_ep05_2024_2026"] > original_weekly_ep05 if original_weekly_ep05 is not None else False
    )
    pivot["beats_original_daily_ep05"] = (
        pivot["total_return_pct_ep05_2024_2026"] > original_daily_ep05 if original_daily_ep05 is not None else False
    )
    pivot["defensive_raw"] = pivot["total_return_pct_period_2021_2022"] + pivot["max_drawdown_pct_period_2021_2022"]
    pivot["offensive_raw"] = pivot["total_return_pct_period_2023_2024"] + 0.25 * pivot["max_drawdown_pct_period_2023_2024"]
    pivot["ep05_raw"] = pivot["total_return_pct_ep05_2024_2026"] + 0.25 * pivot["max_drawdown_pct_ep05_2024_2026"]
    pivot["full_raw"] = pivot["total_return_pct_full_2020_2026"] + 0.35 * pivot["max_drawdown_pct_full_2020_2026"]
    for raw_column in ("defensive_raw", "offensive_raw", "ep05_raw", "full_raw"):
        rank_column = raw_column.replace("_raw", "_rank")
        pivot[rank_column] = pivot[raw_column].rank(pct=True) * 100
    pivot["trade_penalty"] = (pivot["trade_count_full_2020_2026"].fillna(0) / 250).clip(upper=1) * 5
    pivot["composite_score"] = (
        0.35 * pivot["defensive_rank"]
        + 0.30 * pivot["offensive_rank"]
        + 0.20 * pivot["ep05_rank"]
        + 0.15 * pivot["full_rank"]
        - pivot["trade_penalty"]
    )
    pivot["is_reference"] = pivot["strategy_id"].isin(("benchmark",)) | pivot["candidate_id"].isin(
        ("baseline_daily_strength", "baseline_weekly_rotation")
    )
    pivot["primary_eligible"] = (
        ~pivot["is_reference"] & pivot["beats_00631L_ep05"] & pivot["beats_original_weekly_ep05"]
    )
    pivot["stretch_eligible"] = pivot["primary_eligible"] & pivot["beats_original_daily_ep05"]
    pivot["primary_score"] = pivot["composite_score"].where(pivot["primary_eligible"], -9999.0)
    pivot["stretch_score"] = pivot["composite_score"].where(pivot["stretch_eligible"], -9999.0)
    return pivot.sort_values(["stretch_score", "primary_score", "composite_score"], ascending=False)


def _benchmark_lookup(summary: pd.DataFrame) -> dict:
    rows = summary.loc[summary["candidate_id"] == "benchmark_0050"]
    values: dict = {str(row.period_id): float(row.total_return_pct) for row in rows.itertuples(index=False)}
    for row in summary.loc[summary["strategy_id"] == "benchmark"].itertuples(index=False):
        ticker = str(row.candidate_id).removeprefix("benchmark_").replace("_", ".")
        values.setdefault(ticker, {})[str(row.period_id)] = float(row.total_return_pct)
    for candidate_id in ("baseline_weekly_rotation", "baseline_daily_strength"):
        rows = summary.loc[summary["candidate_id"] == candidate_id]
        values[candidate_id] = {str(row.period_id): float(row.total_return_pct) for row in rows.itertuples(index=False)}
    return values


def _add_missing_metric_columns(frame: pd.DataFrame) -> pd.DataFrame:
    periods = ("period_2021_2022", "period_2023_2024", "ep05_2024_2026", "full_2020_2026")
    for period in periods:
        for metric in ("total_return_pct", "max_drawdown_pct", "trade_count"):
            column = f"{metric}_{period}"
            if column not in frame:
                frame[column] = 0.0
    return frame


def _top_candidate_diagnostics(trades: pd.DataFrame, ranking: pd.DataFrame, top_n: int) -> pd.DataFrame:
    rows = []
    if trades.empty or top_n <= 0:
        return pd.DataFrame(rows)
    for candidate in ranking.head(top_n).itertuples(index=False):
        diagnostics = build_closed_trade_diagnostics(trades, str(candidate.candidate_id))
        if diagnostics.empty:
            continue
        summary = summarize_closed_trades(diagnostics)
        for period_summary in summary.itertuples(index=False):
            rows.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "strategy_name": candidate.strategy_name,
                    "composite_score": round(float(candidate.composite_score), 4),
                    **period_summary._asdict(),
                }
            )
    return pd.DataFrame(rows)


def _selected_periods(periods_arg: str) -> dict[str, tuple[str, str, str]]:
    selected: dict[str, tuple[str, str, str]] = {}
    for period_id in [item.strip() for item in periods_arg.split(",") if item.strip()]:
        if period_id not in PERIODS:
            raise ValueError(f"Unsupported period id: {period_id}")
        selected[period_id] = PERIODS[period_id]
    if not selected:
        raise ValueError("At least one period is required")
    return selected


def _selected_variants(variant_arg: str):
    all_variants = default_policy_variants()
    if variant_arg.strip().lower() == "all":
        return all_variants
    variants_by_name = {variant.name: variant for variant in all_variants}
    selected = []
    for name in [item.strip() for item in variant_arg.split(",") if item.strip()]:
        if name not in variants_by_name:
            raise ValueError(f"Unknown variant name: {name}")
        selected.append(variants_by_name[name])
    if not selected:
        raise ValueError("At least one variant is required")
    return tuple(selected)


def _selected_weekdays(weekday_arg: str) -> tuple[int, ...]:
    value = weekday_arg.strip().lower()
    if value == "none":
        return ()
    if value == "all":
        return tuple(WEEKDAYS)
    selected = []
    for item in [part.strip() for part in weekday_arg.split(",") if part.strip()]:
        weekday = int(item)
        if weekday not in WEEKDAYS:
            raise ValueError(f"Unsupported weekday: {weekday}")
        selected.append(weekday)
    return tuple(selected)


def _parse_reference_benchmarks(value: str) -> tuple[tuple[str, str], ...]:
    if not value.strip():
        return ()
    benchmarks: list[tuple[str, str]] = []
    for item in [part.strip() for part in value.split(",") if part.strip()]:
        if ":" in item:
            ticker, label = item.split(":", 1)
        else:
            ticker, label = item, item
        ticker = ticker.strip()
        label = label.strip()
        if not ticker:
            raise ValueError(f"Invalid reference benchmark: {item}")
        benchmarks.append((ticker, label or ticker))
    return tuple(benchmarks)


def _load_reference_prices(reference_benchmarks: tuple[tuple[str, str], ...], cache_dir: str) -> dict[str, pd.DataFrame]:
    prices: dict[str, pd.DataFrame] = {}
    for ticker, _ in reference_benchmarks:
        for directory in (Path(cache_dir), Path("backtest_cache")):
            csv_path = directory / f"{ticker.replace('.', '_')}.csv"
            if csv_path.exists():
                prices[ticker] = load_price_csv(csv_path)
                break
    return prices


def _covers_period(frame: pd.DataFrame, start: str, end: str) -> bool:
    if frame.empty:
        return False
    start_date = pd.Timestamp(start)
    end_date = pd.Timestamp(end)
    first = frame.index.min()
    last = frame.index.max()
    return (first - start_date).days <= 10 and (end_date - last).days <= 10


def _write_report(path: Path, summary: pd.DataFrame, ranking: pd.DataFrame, diagnostics: pd.DataFrame) -> None:
    top_columns = [
        "candidate_id",
        "strategy_name",
        "composite_score",
        "total_return_pct_period_2021_2022",
        "max_drawdown_pct_period_2021_2022",
        "total_return_pct_period_2023_2024",
        "max_drawdown_pct_period_2023_2024",
        "total_return_pct_ep05_2024_2026",
        "max_drawdown_pct_ep05_2024_2026",
        "total_return_pct_full_2020_2026",
        "max_drawdown_pct_full_2020_2026",
        "trade_count_full_2020_2026",
        "beats_00631L_ep05",
        "beats_original_weekly_ep05",
        "beats_original_daily_ep05",
        "primary_eligible",
        "stretch_eligible",
    ]
    lines = [
        "# 策略驗證矩陣",
        "",
        "本報告用同一套評分口徑比較候選策略。這是 AI 輔助回測與策略驗證，不是投資建議。",
        "",
        "## 評分口徑",
        "",
        "- 防守段：2021-2022 報酬與最大回撤，權重 35%。",
        "- 修復/多頭段：2023-2024 報酬與最大回撤，權重 30%。",
        "- EP05 同區間：2024-01-02 到 2026-05-26，權重 20%。",
        "- 2020-2026 完整樣本：長樣本穩定性，權重 15%。",
        "- 交易次數過高會小幅扣分，避免選出難以執行的高換手候選。",
        "",
        "## 綜合排名 Top 20",
        "",
        _markdown_table(ranking[top_columns].head(20)),
        "",
        "## Top 候選 FIFO 閉合交易摘要",
        "",
        _markdown_table(diagnostics),
        "",
        "## 原始分段結果",
        "",
        _markdown_table(summary),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_無資料_"
    headers = list(frame.columns)
    rows = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in frame.iterrows():
        rows.append("| " + " | ".join(str(row[col]) for col in headers) + " |")
    return "\n".join(rows)


if __name__ == "__main__":
    main()
