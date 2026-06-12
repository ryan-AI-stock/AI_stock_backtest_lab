from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from backtest_lab.config import load_config
from backtest_lab.data import load_price_csv, split_adjusted_dividends
from backtest_lab.radar_snapshot_v2_source import load_radar_snapshot_history, select_radar_snapshot_candidates
from backtest_lab.regime_mode_switch import frozen_cycle_proven_top1_v1_variant, simulate_regime_mode_switch
from backtest_lab.sector_dynamic_pool_backtest import RadarSnapshotPoolVariant, simulate_radar_snapshot_pool
from backtest_lab.simulation import simulate_buy_and_hold


DEFAULT_SYMBOL_LABELS = {
    "2327": "國巨*",
    "2344": "華邦電",
    "2368": "金像電",
    "2408": "南亞科",
    "2492": "華新科",
    "3037": "欣興",
    "3163": "波若威",
    "8046": "南電",
    "8299": "群聯",
}


@dataclass(frozen=True)
class ResolvedPrices:
    prices: dict[str, pd.DataFrame]
    symbol_to_ticker: dict[str, str]
    labels: dict[str, str]
    skipped_symbols: list[str]


def run_policy_sweep(
    *,
    snapshot_dir: str | Path,
    cache_dirs: list[str | Path],
    output_dir: str | Path,
    start_date: str,
    end_date: str,
    initial_cash: float,
    config_path: str = "configs/ep05_universe.json",
    theme: str = "",
    date_aware_membership_path: str | Path | None = None,
    date_aware_gap_path: str | Path | None = None,
    top_ns: tuple[int, ...] = (1, 2, 3, 4),
    policies: tuple[str, ...] = ("cash", "hold"),
) -> pd.DataFrame:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    history = load_radar_snapshot_history(snapshot_dir)
    source_notes: list[str] = []
    if theme:
        before = len(history)
        history = history.loc[history["theme"] == theme].copy()
        source_notes.append(f"Theme filter: {theme}; rows {before:,} -> {len(history):,}.")
    if date_aware_membership_path is not None:
        allowed_symbols = load_date_aware_membership_symbols(date_aware_membership_path)
        before = len(history)
        history = history.loc[history["symbol"].astype(str).isin(allowed_symbols)].copy()
        source_notes.append(
            f"Date-aware membership filter: {len(allowed_symbols)} symbols; rows {before:,} -> {len(history):,}."
        )
    if date_aware_gap_path is not None:
        gap_symbols = load_date_aware_gap_symbols(date_aware_gap_path)
        if gap_symbols:
            source_notes.append(f"Excluded static-only membership gaps: {', '.join(gap_symbols)}.")
    if history.empty:
        raise ValueError("No snapshot rows remain after theme/date-aware membership filters.")
    symbols = candidate_symbols(history)
    resolved = resolve_cached_prices(symbols, cache_dirs)
    asset_types = {ticker: "stock" for ticker in resolved.prices}
    config = load_config(config_path)

    rows = []
    dividends = {
        ticker: split_adjusted_dividends(frame, tuple(config.manual_splits.get(ticker, ())))
        for ticker, frame in resolved.prices.items()
    }
    for variant in policy_variants(top_ns=top_ns, policies=policies):
        result = simulate_radar_snapshot_pool(
            variant=variant,
            snapshot_history=history,
            prices_by_ticker=resolved.prices,
            symbol_to_ticker=resolved.symbol_to_ticker,
            labels=resolved.labels,
            asset_types=asset_types,
            start_date=start_date,
            end_date=end_date,
            initial_cash=initial_cash,
            cost_model=config.cost_model,
            dividend_series_by_ticker=dividends,
        )
        result.result.equity_curve.to_csv(output / f"{variant.variant_id}_equity_curve.csv", encoding="utf-8-sig")
        result.holdings.to_csv(output / f"{variant.variant_id}_holdings.csv", index=False, encoding="utf-8-sig")
        result.score_log.to_csv(output / f"{variant.variant_id}_score_log.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame([trade.__dict__ for trade in result.result.trades]).to_csv(
            output / f"{variant.variant_id}_trades.csv",
            index=False,
            encoding="utf-8-sig",
        )
        rows.append(
            {
                "strategy": variant.variant_id,
                "top_n": variant.top_n,
                "max_single_weight": variant.max_single_weight,
                "empty_candidate_policy": variant.empty_candidate_policy,
                "final_value_twd": round(result.result.final_value, 2),
                "total_return_pct": round(result.result.total_return * 100, 2),
                "max_drawdown_pct": round(result.result.max_drawdown * 100, 2),
                "trade_count": len([trade for trade in result.result.trades if trade.action in {"buy", "sell"}]),
                "first_holding": str(result.result.equity_curve["current_ticker"].iloc[0]),
                "last_holding": str(result.result.equity_curve["current_ticker"].iloc[-1]),
            }
        )

    benchmark_rows = benchmark_summary(
        cache_dirs=cache_dirs,
        config_path=config_path,
        start_date=start_date,
        end_date=end_date,
        initial_cash=initial_cash,
    )
    summary = pd.concat([pd.DataFrame(rows), pd.DataFrame(benchmark_rows)], ignore_index=True)
    summary = summary.sort_values("final_value_twd", ascending=False)
    summary.to_csv(output / "policy_vs_benchmark_summary.csv", index=False, encoding="utf-8-sig")
    (output / "policy_vs_benchmark_summary.json").write_text(
        json.dumps(summary.to_dict(orient="records"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output / "policy_vs_benchmark_report.md").write_text(
        policy_sweep_markdown(
            summary,
            start_date,
            end_date,
            initial_cash,
            sorted(resolved.symbol_to_ticker),
            resolved.skipped_symbols,
            source_notes=source_notes,
        ),
        encoding="utf-8",
    )
    return summary


def policy_variants(
    *,
    top_ns: tuple[int, ...],
    policies: tuple[str, ...],
) -> tuple[RadarSnapshotPoolVariant, ...]:
    variants: list[RadarSnapshotPoolVariant] = []
    for top_n in top_ns:
        for policy in policies:
            cap = 0.40 if top_n <= 3 else 0.25
            variants.append(
                RadarSnapshotPoolVariant(
                    f"radar_snapshot_v2_top{top_n}_cap{int(cap * 100)}_weekly_{policy}",
                    f"雷達 snapshot v2 Top{top_n} 單檔{int(cap * 100)}% 週輪動 {policy}",
                    top_n,
                    cap,
                    empty_candidate_policy=policy,
                )
            )
    variants.extend(
        [
            RadarSnapshotPoolVariant(
                "radar_snapshot_v2_top1_cap100_weekly_cash_ma20",
                "雷達 snapshot v2 Top1 滿倉 週輪動 MA20確認",
                1,
                1.00,
                empty_candidate_policy="cash",
                price_trend_rule="ma20",
            ),
            RadarSnapshotPoolVariant(
                "radar_snapshot_v2_top1_cap100_weekly_cash_ma60",
                "雷達 snapshot v2 Top1 滿倉 週輪動 MA60確認",
                1,
                1.00,
                empty_candidate_policy="cash",
                price_trend_rule="ma60",
            ),
            RadarSnapshotPoolVariant(
                "radar_snapshot_v2_top1_cap80_weekly_cash_ma20_persist3of5",
                "雷達 snapshot v2 Top1 80% 週輪動 MA20且5日3次入池",
                1,
                0.80,
                empty_candidate_policy="cash",
                price_trend_rule="ma20",
                recent_candidate_lookback_days=5,
                min_recent_candidate_days=3,
            ),
            RadarSnapshotPoolVariant(
                "radar_snapshot_v2_top2_cap50_weekly_cash_ma20_persist3of5",
                "雷達 snapshot v2 Top2 單檔50% 週輪動 MA20且5日3次入池",
                2,
                0.50,
                empty_candidate_policy="cash",
                price_trend_rule="ma20",
                recent_candidate_lookback_days=5,
                min_recent_candidate_days=3,
            ),
            RadarSnapshotPoolVariant(
                "radar_snapshot_v2_top1_cap100_monthly_cash_ma20",
                "雷達 snapshot v2 Top1 滿倉 月輪動 MA20確認",
                1,
                1.00,
                rebalance_frequency="monthly",
                empty_candidate_policy="cash",
                price_trend_rule="ma20",
            ),
            RadarSnapshotPoolVariant(
                "radar_snapshot_v2_top2_cap50_biweekly_cash_ma20_persist3of5",
                "雷達 snapshot v2 Top2 單檔50% 雙週輪動 MA20且5日3次入池",
                2,
                0.50,
                rebalance_frequency="biweekly",
                empty_candidate_policy="cash",
                price_trend_rule="ma20",
                recent_candidate_lookback_days=5,
                min_recent_candidate_days=3,
            ),
            RadarSnapshotPoolVariant(
                "radar_snapshot_v2_top2_cap50_weekly_cash_ma20_persist3of5_risk55",
                "雷達 snapshot v2 Top2 單檔50% 週輪動 MA20 5日3次入池 風險熱度<=55",
                2,
                0.50,
                empty_candidate_policy="cash",
                price_trend_rule="ma20",
                recent_candidate_lookback_days=5,
                min_recent_candidate_days=3,
                max_risk_heat=55,
            ),
            RadarSnapshotPoolVariant(
                "radar_snapshot_v2_top2_cap50_weekly_cash_ma20_persist3of5_turnover25",
                "雷達 snapshot v2 Top2 單檔50% 週輪動 MA20 5日3次入池 成交占比>=25",
                2,
                0.50,
                empty_candidate_policy="cash",
                price_trend_rule="ma20",
                recent_candidate_lookback_days=5,
                min_recent_candidate_days=3,
                min_turnover_share_in_theme=25,
            ),
            RadarSnapshotPoolVariant(
                "radar_snapshot_v2_top2_cap50_weekly_cash_ma20_persist3of5_stock58",
                "雷達 snapshot v2 Top2 單檔50% 週輪動 MA20 5日3次入池 股票分數>=58",
                2,
                0.50,
                empty_candidate_policy="cash",
                price_trend_rule="ma20",
                recent_candidate_lookback_days=5,
                min_recent_candidate_days=3,
                min_stock_score=58,
            ),
            RadarSnapshotPoolVariant(
                "radar_snapshot_v2_top1_cap100_weekly_cash_ma20_persist3of5_stock58",
                "雷達 snapshot v2 Top1 滿倉 週輪動 MA20 5日3次入池 股票分數>=58",
                1,
                1.00,
                empty_candidate_policy="cash",
                price_trend_rule="ma20",
                recent_candidate_lookback_days=5,
                min_recent_candidate_days=3,
                min_stock_score=58,
            ),
        ]
    )
    return tuple(variants)


def candidate_symbols(history: pd.DataFrame) -> list[str]:
    symbols: set[str] = set()
    for date in sorted(history["date"].dropna().unique()):
        candidates = select_radar_snapshot_candidates(history, date)
        symbols.update(str(symbol).strip() for symbol in candidates.rows["symbol"])
    return sorted(symbols)


def load_date_aware_membership_symbols(path: str | Path) -> set[str]:
    frame = pd.read_csv(path, dtype={"symbol": str}).fillna("")
    required = {"symbol", "effective_start", "source_date", "source_url", "confidence"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{Path(path).name} missing columns: {', '.join(sorted(missing))}")
    usable = frame.loc[
        frame["symbol"].astype(str).str.strip().ne("")
        & frame["effective_start"].astype(str).str.strip().ne("")
        & frame["source_date"].astype(str).str.strip().ne("")
        & frame["source_url"].astype(str).str.strip().ne("")
        & frame["confidence"].astype(str).str.strip().str.lower().isin({"high", "medium", "low"})
    ].copy()
    if usable.empty:
        raise ValueError(f"No usable date-aware membership rows in {path}")
    return set(usable["symbol"].astype(str).str.strip())


def load_date_aware_gap_symbols(path: str | Path) -> list[str]:
    gap_path = Path(path)
    if not gap_path.exists():
        return []
    frame = pd.read_csv(gap_path, dtype={"symbol": str}).fillna("")
    if "symbol" not in frame.columns:
        return []
    return sorted(set(frame["symbol"].astype(str).str.strip()) - {""})


def resolve_cached_prices(symbols: list[str], cache_dirs: list[str | Path]) -> ResolvedPrices:
    prices: dict[str, pd.DataFrame] = {}
    symbol_to_ticker: dict[str, str] = {}
    labels: dict[str, str] = {}
    skipped: list[str] = []
    for symbol in symbols:
        resolved = _find_cached_price(symbol, cache_dirs)
        if resolved is None:
            skipped.append(symbol)
            continue
        ticker, path = resolved
        prices[ticker] = load_price_csv(path)
        symbol_to_ticker[symbol] = ticker
        labels[ticker] = DEFAULT_SYMBOL_LABELS.get(symbol, symbol)
    return ResolvedPrices(prices=prices, symbol_to_ticker=symbol_to_ticker, labels=labels, skipped_symbols=skipped)


def benchmark_summary(
    *,
    cache_dirs: list[str | Path],
    config_path: str,
    start_date: str,
    end_date: str,
    initial_cash: float,
) -> list[dict[str, object]]:
    config = load_config(config_path)
    rows = []
    for name, ticker, asset_type in (("0050 買進持有", "0050.TW", "etf"), ("0050正二 買進持有", "00631L.TW", "etf")):
        path = _find_cached_ticker(ticker, cache_dirs)
        if path is None:
            continue
        prices = load_price_csv(path)
        dividends = split_adjusted_dividends(prices, tuple(config.manual_splits.get(ticker, ())))
        result = simulate_buy_and_hold(
            name,
            ticker,
            asset_type,
            prices,
            start_date,
            end_date,
            initial_cash,
            config.cost_model,
            dividend_series=dividends,
        )
        rows.append(
            {
                "strategy": name,
                "top_n": "",
                "max_single_weight": "",
                "empty_candidate_policy": "benchmark",
                "final_value_twd": round(result.final_value, 2),
                "total_return_pct": round(result.total_return * 100, 2),
                "max_drawdown_pct": round(result.max_drawdown * 100, 2),
                "trade_count": len([trade for trade in result.trades if trade.action in {"buy", "sell"}]),
                "first_holding": ticker,
                "last_holding": ticker,
            }
        )
    frozen_row = frozen_best_benchmark_summary(
        config=config,
        cache_dirs=cache_dirs,
        start_date=start_date,
        end_date=end_date,
        initial_cash=initial_cash,
    )
    if frozen_row is not None:
        rows.append(frozen_row)
    return rows


def frozen_best_benchmark_summary(
    *,
    config,
    cache_dirs: list[str | Path],
    start_date: str,
    end_date: str,
    initial_cash: float,
) -> dict[str, object] | None:
    group = config.group_by_id("group_c_0050_00631l_plus_mega_caps")
    prices: dict[str, pd.DataFrame] = {}
    asset_types: dict[str, str] = {}
    dividends: dict[str, pd.Series] = {}
    for asset in group.assets:
        path = _find_cached_ticker(asset.ticker, cache_dirs)
        if path is None:
            return None
        frame = load_price_csv(path)
        prices[asset.ticker] = frame
        asset_types[asset.ticker] = asset.asset_type
        dividends[asset.ticker] = split_adjusted_dividends(frame, tuple(config.manual_splits.get(asset.ticker, ())))

    result = simulate_regime_mode_switch(
        name="AI中大型權值股池最佳版 v20260605",
        prices_by_ticker=prices,
        asset_types=asset_types,
        market_prices=prices["0050.TW"],
        start_date=start_date,
        end_date=end_date,
        initial_cash=initial_cash,
        cost_model=config.cost_model,
        variant=frozen_cycle_proven_top1_v1_variant(),
        dividend_series_by_ticker=dividends,
    )
    return {
        "strategy": result.name,
        "top_n": "",
        "max_single_weight": "",
        "empty_candidate_policy": "frozen_best_benchmark",
        "final_value_twd": round(result.final_value, 2),
        "total_return_pct": round(result.total_return * 100, 2),
        "max_drawdown_pct": round(result.max_drawdown * 100, 2),
        "trade_count": len([trade for trade in result.trades if trade.action in {"buy", "sell"}]),
        "first_holding": str(result.equity_curve["current_ticker"].iloc[0]),
        "last_holding": str(result.equity_curve["current_ticker"].iloc[-1]),
    }


def policy_sweep_markdown(
    summary: pd.DataFrame,
    start_date: str,
    end_date: str,
    initial_cash: float,
    used_symbols: list[str],
    skipped_symbols: list[str],
    source_notes: list[str] | None = None,
) -> str:
    lines = [
        "# Radar Snapshot v2 Policy vs Benchmark Replay Backtest",
        "",
        f"- Period: {start_date} to {end_date}",
        f"- Initial cash: {initial_cash:,.0f} TWD",
        f"- Used candidate symbols: {', '.join(used_symbols) or 'none'}",
        f"- Skipped candidate symbols: {', '.join(skipped_symbols) or 'none'}",
    ]
    for note in source_notes or []:
        lines.append(f"- {note}")
    lines.extend(
        [
            "",
            "| rank | strategy | final value | return | max drawdown | trades | last holding |",
            "| ---: | --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for rank, (_, row) in enumerate(summary.iterrows(), start=1):
        lines.append(
            f"| {rank} | {row['strategy']} | {row['final_value_twd']:,.0f} | "
            f"{row['total_return_pct']:.2f}% | {row['max_drawdown_pct']:.2f}% | "
            f"{int(row['trade_count'])} | {row['last_holding']} |"
        )
    lines.extend(
        [
            "",
            "Interpretation: historical replay backtest only. These snapshots are reconstructed for strategy testing and should not be described as real-time daily radar records.",
        ]
    )
    return "\n".join(lines) + "\n"


def _find_cached_price(symbol: str, cache_dirs: list[str | Path]) -> tuple[str, Path] | None:
    for suffix in ("TW", "TWO"):
        ticker = f"{symbol}.{suffix}"
        path = _find_cached_ticker(ticker, cache_dirs)
        if path is not None:
            return ticker, path
    return None


def _find_cached_ticker(ticker: str, cache_dirs: list[str | Path]) -> Path | None:
    file_name = f"{ticker.replace('.', '_')}.csv"
    for directory in cache_dirs:
        path = Path(directory) / file_name
        if path.exists():
            return path
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Run radar snapshot v2 policy sweep smoke.")
    parser.add_argument("--snapshot-dir", required=True)
    parser.add_argument("--cache-dir", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--initial-cash", type=float, default=1_000_000)
    parser.add_argument("--config", default="configs/ep05_universe.json")
    parser.add_argument("--theme", default="")
    parser.add_argument("--date-aware-membership", default="")
    parser.add_argument("--date-aware-gap", default="")
    args = parser.parse_args()

    summary = run_policy_sweep(
        snapshot_dir=args.snapshot_dir,
        cache_dirs=args.cache_dir,
        output_dir=args.output_dir,
        start_date=args.start_date,
        end_date=args.end_date,
        initial_cash=args.initial_cash,
        config_path=args.config,
        theme=args.theme,
        date_aware_membership_path=args.date_aware_membership or None,
        date_aware_gap_path=args.date_aware_gap or None,
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
