from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import pandas as pd

from backtest_lab.costs import TaiwanCostModel
from backtest_lab.portfolio import Portfolio, Trade
from backtest_lab.simulation import BacktestResult, _common_trade_dates, _date_str, _max_drawdown
from backtest_lab.stock_pool_observation import _current_close_by_ticker, _load_observation_price_frames
from backtest_lab.stock_pool_store import StockPoolStore
from backtest_lab.universal_pool_strategy import (
    UniversalPoolParameters,
    default_parameters_for_profile,
    infer_pool_profile,
    score_universal_candidates,
)
from backtest_lab.valuation_source import load_valuation_signals


@dataclass(frozen=True)
class ValuationShadowVariant:
    name: str
    label: str
    valuation_signal_weight: float = 0.0
    require_valuation_gate: bool = False


DEFAULT_VARIANTS = (
    ValuationShadowVariant("baseline", "原始通用排序"),
    ValuationShadowVariant("valuation_weighted_1x", "估值加權 1x", valuation_signal_weight=1.0),
    ValuationShadowVariant("valuation_gate", "估值買點硬閘門", require_valuation_gate=True),
)


def simulate_valuation_shadow_strategy(
    *,
    name: str,
    prices_by_ticker: dict[str, pd.DataFrame],
    asset_types: dict[str, str],
    start_date: str,
    end_date: str,
    initial_cash: float,
    cost_model: TaiwanCostModel,
    base_params: UniversalPoolParameters | None = None,
    valuation_data: str | Path | None = None,
    variant: ValuationShadowVariant = DEFAULT_VARIANTS[0],
    rebalance_frequency: str = "weekly",
) -> tuple[BacktestResult, dict[str, Any]]:
    trade_dates = _common_trade_dates(prices_by_ticker, start_date, end_date)
    if not trade_dates:
        raise ValueError(f"No common trade dates between {start_date} and {end_date}")
    profile = infer_pool_profile(
        {ticker: frame.loc[frame.index <= pd.Timestamp(end_date)] for ticker, frame in prices_by_ticker.items()},
        trade_dates[0],
    )
    params = base_params or default_parameters_for_profile(profile)
    params = replace(
        params,
        valuation_signal_weight=variant.valuation_signal_weight,
        require_valuation_gate=variant.require_valuation_gate,
    )
    portfolio = Portfolio(initial_cash, cost_model)
    equity_rows: list[dict[str, Any]] = []
    signal_rows: list[dict[str, Any]] = []
    valuation_hit_count = 0
    rebalance_count = 0
    last_week_key: tuple[int, int] | None = None

    for index, trade_date in enumerate(trade_dates):
        signal_date = _previous_available_date(prices_by_ticker, trade_date)
        should_rebalance = index == 0 or rebalance_frequency == "daily"
        if rebalance_frequency == "weekly":
            week_key = (trade_date.isocalendar().year, trade_date.isocalendar().week)
            should_rebalance = should_rebalance or week_key != last_week_key
            if should_rebalance:
                last_week_key = week_key
        if should_rebalance:
            rebalance_count += 1
            valuation_signals = load_valuation_signals(
                valuation_data,
                signal_date=signal_date,
                current_price_by_ticker=_current_close_by_ticker(prices_by_ticker, signal_date),
            )
            valuation_hit_count += len(valuation_signals)
            scores = score_universal_candidates(
                prices_by_ticker,
                signal_date,
                params,
                valuation_signal_by_ticker=valuation_signals,
            )
            ranked = sorted(scores.values(), key=lambda item: (item.passed, item.score, item.ret20, item.ticker), reverse=True)
            target = next((candidate.ticker for candidate in ranked if candidate.passed), None)
            current = portfolio.current_ticker()
            if current != target and current is not None:
                portfolio.sell_all(
                    _date_str(trade_date),
                    current,
                    asset_types.get(current, "stock"),
                    float(prices_by_ticker[current].loc[trade_date, "open"]),
                    f"{variant.name}_rebalance",
                )
            if target and current != target:
                portfolio.buy_max(
                    _date_str(trade_date),
                    target,
                    asset_types.get(target, "stock"),
                    float(prices_by_ticker[target].loc[trade_date, "open"]),
                    f"{variant.name}_initial_entry" if current is None else f"{variant.name}_rebalance",
                )
            top = ranked[0] if ranked else None
            signal_rows.append(
                {
                    "trade_date": _date_str(trade_date),
                    "signal_date": _date_str(signal_date),
                    "target": target or "cash",
                    "top_ranked": top.ticker if top else "",
                    "top_score": top.score if top else 0.0,
                    "top_reason": top.reason if top else "",
                    "top_valuation_reason": top.valuation_reason if top else "",
                    "top_valuation_safety_margin_pct": top.valuation_safety_margin_pct if top else 0.0,
                }
            )
        close_prices = {ticker: float(frame.loc[trade_date, "close"]) for ticker, frame in prices_by_ticker.items()}
        equity_rows.append(
            {
                "date": trade_date,
                "total_value": portfolio.market_value(close_prices),
                "current_ticker": portfolio.current_ticker() or "cash",
            }
        )

    equity_curve = pd.DataFrame(equity_rows).set_index("date")
    result = BacktestResult(
        name=name,
        final_value=float(equity_curve["total_value"].iloc[-1]),
        total_return=float(equity_curve["total_value"].iloc[-1] / initial_cash - 1),
        max_drawdown=_max_drawdown(equity_curve["total_value"]),
        trades=portfolio.trades,
        equity_curve=equity_curve,
    )
    diagnostics = {
        "variant": asdict(variant),
        "rebalance_frequency": rebalance_frequency,
        "rebalance_count": rebalance_count,
        "valuation_signal_total_hits": valuation_hit_count,
        "valuation_signal_avg_hits_per_rebalance": valuation_hit_count / rebalance_count if rebalance_count else 0.0,
        "signal_rows": signal_rows,
    }
    return result, diagnostics


def run_valuation_shadow_backtest(
    *,
    pool: dict[str, Any],
    start_date: str,
    end_date: str,
    initial_cash: float,
    cost_model: TaiwanCostModel,
    cache_dir: str | Path,
    output_dir: str | Path,
    valuation_data: str | Path | None,
    variants: tuple[ValuationShadowVariant, ...] = DEFAULT_VARIANTS,
    rebalance_frequency: str = "weekly",
) -> dict[str, Any]:
    tickers = [symbol["ticker"] for symbol in pool.get("resolved_symbols", [])]
    if not tickers:
        raise ValueError("Pool has no resolved_symbols")
    prices, missing = _load_observation_price_frames(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        cache_dir=cache_dir,
    )
    if missing:
        raise ValueError(f"Missing price data: {', '.join(missing)}")
    asset_types = {symbol["ticker"]: symbol.get("asset_type", "stock") for symbol in pool.get("resolved_symbols", [])}
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, Any]] = []
    manifest: dict[str, Any] = {
        "pool_id": pool.get("pool_id", ""),
        "pool_name": pool.get("name", ""),
        "start_date": start_date,
        "end_date": end_date,
        "initial_cash": initial_cash,
        "valuation_data": str(valuation_data or ""),
        "rebalance_frequency": rebalance_frequency,
        "missing_price_tickers": missing,
        "variants": [],
    }
    for variant in variants:
        result, diagnostics = simulate_valuation_shadow_strategy(
            name=variant.name,
            prices_by_ticker=prices,
            asset_types=asset_types,
            start_date=start_date,
            end_date=end_date,
            initial_cash=initial_cash,
            cost_model=cost_model,
            valuation_data=valuation_data,
            variant=variant,
            rebalance_frequency=rebalance_frequency,
        )
        variant_dir = output / variant.name
        variant_dir.mkdir(parents=True, exist_ok=True)
        result.equity_curve.to_csv(variant_dir / "equity_curve.csv", encoding="utf-8-sig")
        pd.DataFrame([trade.__dict__ for trade in result.trades]).to_csv(
            variant_dir / "trades.csv",
            index=False,
            encoding="utf-8-sig",
        )
        pd.DataFrame(diagnostics["signal_rows"]).to_csv(
            variant_dir / "signals.csv",
            index=False,
            encoding="utf-8-sig",
        )
        row = {
            "variant": variant.name,
            "label": variant.label,
            "final_value": round(result.final_value, 2),
            "total_return_pct": round(result.total_return * 100, 2),
            "max_drawdown_pct": round(result.max_drawdown * 100, 2),
            "trade_count": len(result.trades),
            "valuation_signal_avg_hits_per_rebalance": round(diagnostics["valuation_signal_avg_hits_per_rebalance"], 2),
            "output_dir": str(variant_dir),
        }
        summary_rows.append(row)
        manifest["variants"].append({**row, "diagnostics": {k: v for k, v in diagnostics.items() if k != "signal_rows"}})
    pd.DataFrame(summary_rows).to_csv(output / "valuation_shadow_summary.csv", index=False, encoding="utf-8-sig")
    (output / "valuation_shadow_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_report(output / "valuation_shadow_report.md", manifest, summary_rows)
    return manifest


def _write_report(path: Path, manifest: dict[str, Any], summary_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# 估值濾網 Shadow 回測",
        "",
        f"- 股票池：{manifest.get('pool_name')} ({manifest.get('pool_id')})",
        f"- 區間：{manifest.get('start_date')} ~ {manifest.get('end_date')}",
        f"- 估值資料：{manifest.get('valuation_data') or '未提供'}",
        f"- 頻率：{manifest.get('rebalance_frequency')}",
        "",
        "本報告用來檢查 EPS / 合理價 / 買點濾網是否改善策略，不代表正式模型已替換。",
        "",
        "| 版本 | 期末淨值 | 報酬率 | 最大回撤 | 交易數 | 平均估值命中 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['label']} | {row['final_value']:,.0f} | {row['total_return_pct']:.2f}% | "
            f"{row['max_drawdown_pct']:.2f}% | {row['trade_count']} | {row['valuation_signal_avg_hits_per_rebalance']:.2f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _previous_available_date(prices_by_ticker: dict[str, pd.DataFrame], trade_date: pd.Timestamp) -> pd.Timestamp:
    common = None
    for frame in prices_by_ticker.values():
        dates = set(frame.index[frame.index < trade_date])
        common = dates if common is None else common & dates
    if not common:
        return trade_date
    return max(common)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run valuation-filter shadow backtests for a stock pool.")
    parser.add_argument("--pool-store", default="work/stock_pools/stock_pools.json")
    parser.add_argument("--pool-id", required=True)
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--initial-cash", type=float, default=1_000_000)
    parser.add_argument("--cache-dir", default="backtest_cache/stock_pool_observations")
    parser.add_argument("--output-dir", default="outputs/valuation_shadow_backtest")
    parser.add_argument("--valuation-data", default="")
    parser.add_argument("--rebalance-frequency", choices=("daily", "weekly"), default="weekly")
    args = parser.parse_args()
    pool = next((item for item in StockPoolStore(args.pool_store).list_pools() if item["pool_id"] == args.pool_id), None)
    if pool is None:
        raise ValueError(f"Unknown pool_id: {args.pool_id}")
    manifest = run_valuation_shadow_backtest(
        pool=pool,
        start_date=args.start_date,
        end_date=args.end_date,
        initial_cash=args.initial_cash,
        cost_model=TaiwanCostModel(),
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        valuation_data=args.valuation_data or None,
        rebalance_frequency=args.rebalance_frequency,
    )
    print(f"VALUATION_SHADOW_OUTPUT={Path(args.output_dir).resolve()}")
    print(json.dumps({k: manifest[k] for k in ("pool_id", "start_date", "end_date", "valuation_data")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
