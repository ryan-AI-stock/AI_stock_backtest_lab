from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from backtest_lab.config import load_config
from backtest_lab.data import download_yfinance_prices, load_price_csv, split_adjusted_dividends
from backtest_lab.regime_aware_backtest import PERIODS
from backtest_lab.regime_mode_switch import (
    asymmetric_attack_defense_variants,
    asymmetric_strategy_selector_variants,
    attack_gate_latch_variants,
    attack_gate_fine_sweep_variants,
    attack_gate_leveraged_fallback_variants,
    attack_gate_persistence_variants,
    attack_gate_acceleration_variants,
    default_mode_switch_variants,
    simulate_regime_mode_switch,
    stop_latch_attack_defense_variants,
    stop_latch_defense_sweep_variants,
    stop_latch_health_release_variants,
    stop_latched_strategy_selector_variants,
    fast_risk_strategy_selector_variants,
    fallback_only_risk_selector_variants,
    two_stage_attack_selector_variants,
    two_stage_fast_guard_variants,
    two_stage_cash_guard_variants,
    cycle_proven_selector_variants,
    cycle_proven_robustness_variants,
    cycle_proven_history_init_variants,
    cycle_proven_preproof_exposure_variants,
    cycle_proven_preproof_dynamic_exposure_variants,
    cycle_proven_cadence_variants,
    cycle_proven_asset_role_variants,
    cycle_proven_market_exposure_ladder_variants,
    strategy_health_attack_defense_variants,
)
from backtest_lab.simulation import (
    BacktestResult,
    simulate_buy_and_hold,
    simulate_dual_momentum_vol_control,
    simulate_relative_strength_top1,
)


DEFAULT_PERIODS = "period_2021_2022,period_2023_2024"


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest explicit market-regime mode switching strategies.")
    parser.add_argument("--config", default="configs/ep05_universe.json")
    parser.add_argument("--group-id", default="group_c_0050_00631l_plus_mega_caps")
    parser.add_argument("--cache-dir", default="backtest_cache")
    parser.add_argument("--output-dir", default="outputs/regime_mode_switch_backtest")
    parser.add_argument("--market-proxy", default="0050.TW")
    parser.add_argument("--periods", default=DEFAULT_PERIODS)
    parser.add_argument("--candidate-filter", default="")
    parser.add_argument("--exclude-tickers", default="")
    parser.add_argument(
        "--variant-set",
        choices=(
            "default",
            "asymmetric",
            "strategy-health",
            "stop-latch",
            "stop-latch-defense-sweep",
            "stop-latch-health-release",
            "attack-gate-latch",
            "attack-gate-fine-sweep",
            "attack-gate-persistence",
            "attack-gate-acceleration",
            "attack-gate-leveraged-fallback",
            "asymmetric-strategy-selector",
            "stop-latched-strategy-selector",
            "fast-risk-strategy-selector",
            "fallback-only-risk-selector",
            "two-stage-attack-selector",
            "two-stage-fast-guard",
            "two-stage-cash-guard",
            "cycle-proven-selector",
            "cycle-proven-robustness",
            "cycle-proven-history-init",
            "cycle-proven-preproof-exposure",
            "cycle-proven-preproof-dynamic-exposure",
            "cycle-proven-cadence",
            "cycle-proven-asset-role",
            "cycle-proven-market-exposure-ladder",
        ),
        default="default",
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
    cached_prices = _load_sufficient_cache_prices(
        tickers,
        args.cache_dir,
        required_start=start_for_download,
        required_end=end_for_download,
    )
    download_tickers = [ticker for ticker in tickers if ticker not in cached_prices]
    prices = download_yfinance_prices(
        tickers=download_tickers,
        start_date=(start_for_download - pd.DateOffset(years=2)).strftime("%Y-%m-%d"),
        end_date=end_for_download.strftime("%Y-%m-%d"),
        cache_dir=args.cache_dir,
    )
    prices.update(cached_prices)
    dividends = {
        ticker: split_adjusted_dividends(prices[ticker], config.manual_splits.get(ticker, ())) for ticker in tickers
    }
    group_prices = {asset.ticker: prices[asset.ticker] for asset in group.assets}
    excluded_tickers = {ticker.strip() for ticker in args.exclude_tickers.split(",") if ticker.strip()}
    group_prices = {ticker: frame for ticker, frame in group_prices.items() if ticker not in excluded_tickers}

    summary_rows: list[dict] = []
    trade_rows: list[dict] = []
    daily_rows: list[dict] = []
    for period_id, (start, end, period_label) in selected_periods.items():
        period_results = _run_period(
            period_id=period_id,
            period_label=period_label,
            start=start,
            end=end,
            group_prices=group_prices,
            market_prices=prices[args.market_proxy],
            asset_types=asset_types,
            initial_cash=config.initial_cash_twd,
            cost_model=config.cost_model,
            dividends=dividends,
            candidate_filter=args.candidate_filter,
            variant_set=args.variant_set,
        )
        for candidate, result in period_results:
            summary_rows.append(_summary_row(period_id, period_label, candidate, result))
            trade_rows.extend(_trade_rows(period_id, candidate, result, labels))
            daily_rows.extend(_daily_rows(period_id, candidate, result, labels))

    summary = pd.DataFrame(summary_rows)
    trades = pd.DataFrame(trade_rows)
    daily = pd.DataFrame(daily_rows)
    ranking = _ranking(summary)
    summary.to_csv(output_dir / "regime_mode_switch_summary.csv", index=False, encoding="utf-8-sig")
    ranking.to_csv(output_dir / "regime_mode_switch_ranking.csv", index=False, encoding="utf-8-sig")
    trades.to_csv(output_dir / "regime_mode_switch_trades.csv", index=False, encoding="utf-8-sig")
    daily.to_csv(output_dir / "regime_mode_switch_daily.csv", index=False, encoding="utf-8-sig")
    _write_report(output_dir / "regime_mode_switch_report.md", summary, ranking)
    (output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "group_id": args.group_id,
                "market_proxy": args.market_proxy,
                "periods": selected_periods,
                "excluded_tickers": sorted(excluded_tickers),
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
    market_prices: pd.DataFrame,
    asset_types: dict[str, str],
    initial_cash: float,
    cost_model,
    dividends: dict[str, pd.Series],
    candidate_filter: str = "",
    variant_set: str = "default",
) -> list[tuple[dict, BacktestResult]]:
    results: list[tuple[dict, BacktestResult]] = []
    available_prices = {ticker: frame for ticker, frame in group_prices.items() if _covers_period(frame, start, end)}
    available_dividends = {ticker: dividends[ticker] for ticker in available_prices}
    # Reference baselines are always included so every optimization run can be
    # compared directly with the original daily and weekly prototypes.
    include_baselines = True
    for ticker, label in (("0050.TW", "0050買進持有"), ("00631L.TW", "0050正二買進持有")):
        if not include_baselines:
            continue
        if ticker in available_prices:
            result = simulate_buy_and_hold(
                name=f"{ticker}_buy_and_hold",
                ticker=ticker,
                asset_type=asset_types[ticker],
                prices=available_prices[ticker],
                start_date=start,
                end_date=end,
                initial_cash=initial_cash,
                cost_model=cost_model,
                dividend_series=dividends[ticker],
            )
            results.append((_candidate(f"benchmark_{ticker.replace('.', '_')}", label, "benchmark"), result))

    if include_baselines:
        daily = simulate_relative_strength_top1(
            name="baseline_daily_strength_combined_pool",
            prices_by_ticker=available_prices,
            asset_types=asset_types,
            start_date=start,
            end_date=end,
            initial_cash=initial_cash,
            cost_model=cost_model,
            dividend_series_by_ticker=available_dividends,
        )
        results.append((_candidate("baseline_daily_strength_combined_pool", "9檔池原始每日追強勢", "baseline"), daily))

        weekly = simulate_dual_momentum_vol_control(
            name="baseline_weekly_rotation_combined_pool",
            prices_by_ticker=available_prices,
            asset_types=asset_types,
            start_date=start,
            end_date=end,
            initial_cash=initial_cash,
            cost_model=cost_model,
            dividend_series_by_ticker=available_dividends,
        )
        results.append((_candidate("baseline_weekly_rotation_combined_pool", "9檔池原始週頻輪動", "baseline"), weekly))

    if variant_set == "asymmetric":
        variants = asymmetric_attack_defense_variants()
    elif variant_set == "strategy-health":
        variants = strategy_health_attack_defense_variants()
    elif variant_set == "stop-latch":
        variants = stop_latch_attack_defense_variants()
    elif variant_set == "stop-latch-defense-sweep":
        variants = stop_latch_defense_sweep_variants()
    elif variant_set == "stop-latch-health-release":
        variants = stop_latch_health_release_variants()
    elif variant_set == "attack-gate-latch":
        variants = attack_gate_latch_variants()
    elif variant_set == "attack-gate-fine-sweep":
        variants = attack_gate_fine_sweep_variants()
    elif variant_set == "attack-gate-persistence":
        variants = attack_gate_persistence_variants()
    elif variant_set == "attack-gate-acceleration":
        variants = attack_gate_acceleration_variants()
    elif variant_set == "attack-gate-leveraged-fallback":
        variants = attack_gate_leveraged_fallback_variants()
    elif variant_set == "asymmetric-strategy-selector":
        variants = asymmetric_strategy_selector_variants()
    elif variant_set == "stop-latched-strategy-selector":
        variants = stop_latched_strategy_selector_variants()
    elif variant_set == "fast-risk-strategy-selector":
        variants = fast_risk_strategy_selector_variants()
    elif variant_set == "fallback-only-risk-selector":
        variants = fallback_only_risk_selector_variants()
    elif variant_set == "two-stage-attack-selector":
        variants = two_stage_attack_selector_variants()
    elif variant_set == "two-stage-fast-guard":
        variants = two_stage_fast_guard_variants()
    elif variant_set == "two-stage-cash-guard":
        variants = two_stage_cash_guard_variants()
    elif variant_set == "cycle-proven-selector":
        variants = cycle_proven_selector_variants()
    elif variant_set == "cycle-proven-robustness":
        variants = cycle_proven_robustness_variants()
    elif variant_set == "cycle-proven-history-init":
        variants = cycle_proven_history_init_variants()
    elif variant_set == "cycle-proven-preproof-exposure":
        variants = cycle_proven_preproof_exposure_variants()
    elif variant_set == "cycle-proven-preproof-dynamic-exposure":
        variants = cycle_proven_preproof_dynamic_exposure_variants()
    elif variant_set == "cycle-proven-cadence":
        variants = cycle_proven_cadence_variants()
    elif variant_set == "cycle-proven-asset-role":
        variants = cycle_proven_asset_role_variants()
    elif variant_set == "cycle-proven-market-exposure-ladder":
        variants = cycle_proven_market_exposure_ladder_variants()
    else:
        variants = default_mode_switch_variants()
    for variant in variants:
        if candidate_filter and candidate_filter not in variant.name:
            continue
        result = simulate_regime_mode_switch(
            name=f"regime_mode_switch_{variant.name}",
            prices_by_ticker=available_prices,
            asset_types=asset_types,
            market_prices=market_prices,
            start_date=start,
            end_date=end,
            initial_cash=initial_cash,
            cost_model=cost_model,
            variant=variant,
            dividend_series_by_ticker=available_dividends,
        )
        results.append((_candidate(f"regime_mode_switch_{variant.name}", f"市場環境切換_{variant.name}", "mode_switch"), result))
    return results


def _candidate(candidate_id: str, strategy_name: str, strategy_id: str) -> dict:
    return {"candidate_id": candidate_id, "strategy_name": strategy_name, "strategy_id": strategy_id}


def _summary_row(period_id: str, period_label: str, candidate: dict, result: BacktestResult) -> dict:
    return {
        "period_id": period_id,
        "period_label": period_label,
        **candidate,
        "final_value_twd": round(result.final_value, 2),
        "total_return_pct": round(result.total_return * 100, 2),
        "max_drawdown_pct": round(result.max_drawdown * 100, 2),
        "trade_count": sum(1 for trade in result.trades if trade.action in {"buy", "sell"}),
    }


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


def _daily_rows(period_id: str, candidate: dict, result: BacktestResult, labels: dict[str, str]) -> list[dict]:
    frame = result.equity_curve.copy()
    frame["daily_return_pct"] = frame["total_value"].pct_change().fillna(0.0) * 100
    rows = []
    for date, row in frame.iterrows():
        ticker = str(row.get("current_ticker", ""))
        rows.append(
            {
                "period_id": period_id,
                "candidate_id": candidate["candidate_id"],
                "strategy_name": candidate["strategy_name"],
                "date": pd.Timestamp(date).strftime("%Y-%m-%d"),
                "total_value_twd": round(float(row["total_value"]), 2),
                "daily_return_pct": round(float(row["daily_return_pct"]), 6),
                "current_ticker": ticker,
                "current_label": labels.get(ticker, ticker),
                "regime": row.get("regime", ""),
                "mode": row.get("mode", ""),
            }
        )
    return rows


def _ranking(summary: pd.DataFrame) -> pd.DataFrame:
    pivot = summary.pivot_table(
        index=["candidate_id", "strategy_name", "strategy_id"],
        columns="period_id",
        values=["total_return_pct", "max_drawdown_pct", "trade_count"],
        aggfunc="first",
    )
    pivot.columns = [f"{metric}_{period}" for metric, period in pivot.columns]
    pivot = pivot.reset_index()
    preferred = (
        "total_return_pct_period_2023_2024",
        "total_return_pct_bear_2022",
        "total_return_pct_period_2021_2022",
        "total_return_pct_year_2021",
    )
    sort_columns = [column for column in preferred if column in pivot.columns]
    return pivot.sort_values(sort_columns, ascending=False) if sort_columns else pivot


def _selected_periods(periods_arg: str) -> dict[str, tuple[str, str, str]]:
    selected: dict[str, tuple[str, str, str]] = {}
    for period_id in [item.strip() for item in periods_arg.split(",") if item.strip()]:
        if period_id not in PERIODS:
            raise ValueError(f"Unsupported period id: {period_id}")
        selected[period_id] = PERIODS[period_id]
    if not selected:
        raise ValueError("At least one period is required")
    return selected


def _covers_period(frame: pd.DataFrame, start: str, end: str) -> bool:
    if frame.empty:
        return False
    start_date = pd.Timestamp(start)
    end_date = pd.Timestamp(end)
    first = frame.index.min()
    last = frame.index.max()
    return (first - start_date).days <= 10 and (end_date - last).days <= 10


def _load_sufficient_cache_prices(
    tickers: list[str],
    cache_dir: str,
    *,
    required_start: pd.Timestamp,
    required_end: pd.Timestamp,
    warmup_calendar_days: int = 365,
) -> dict[str, pd.DataFrame]:
    prices: dict[str, pd.DataFrame] = {}
    warmup_start = required_start - pd.Timedelta(days=warmup_calendar_days)
    for ticker in tickers:
        for directory in (Path(cache_dir), Path("backtest_cache")):
            csv_path = directory / f"{ticker.replace('.', '_')}.csv"
            if not csv_path.exists():
                continue
            frame = load_price_csv(csv_path)
            if (
                (frame.index.min() - warmup_start).days <= 10
                and (required_end - frame.index.max()).days <= 10
            ):
                prices[ticker] = frame
                break
    return prices


def _write_report(path: Path, summary: pd.DataFrame, ranking: pd.DataFrame) -> None:
    lines = [
        "# 市場環境切換策略回測",
        "",
        "股票池：0050、0050正二、七檔指定權值/AI相關標的。這是 AI 輔助回測與策略驗證，不是投資建議。",
        "",
        "## 排名",
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
