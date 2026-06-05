from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from backtest_lab.config import load_config
from backtest_lab.costs import TaiwanCostModel
from backtest_lab.data import download_yfinance_prices, load_price_csv, split_adjusted_dividends
from backtest_lab.portfolio import Trade
from backtest_lab.regime_aware_backtest import PERIODS
from backtest_lab.regime_aware_simulation import _Account, _market_value, _rebalance
from backtest_lab.simulation import BacktestResult, _date_str, _max_drawdown, _trade_dates, simulate_buy_and_hold
from backtest_lab.strategies import previous_available_date


DEFAULT_PERIODS = "period_2021_2022,period_2023_2024,ep05_2024_2026"
DEFENSE_TICKER = "0050.TW"


@dataclass(frozen=True)
class DefenseVariant:
    name: str
    rule: str
    risk_on_exposure: float = 1.0
    risk_off_exposure: float = 0.0


def default_defense_variants() -> tuple[DefenseVariant, ...]:
    return (
        DefenseVariant(name="ma60_cash", rule="ma60", risk_off_exposure=0.0),
        DefenseVariant(name="ma120_cash", rule="ma120", risk_off_exposure=0.0),
        DefenseVariant(name="ma150_cash", rule="ma150", risk_off_exposure=0.0),
        DefenseVariant(name="ma180_cash", rule="ma180", risk_off_exposure=0.0),
        DefenseVariant(name="ma200_cash", rule="ma200", risk_off_exposure=0.0),
        DefenseVariant(name="ma220_cash", rule="ma220", risk_off_exposure=0.0),
        DefenseVariant(name="ma250_cash", rule="ma250", risk_off_exposure=0.0),
        DefenseVariant(name="ma60_10pct_keep", rule="ma60", risk_off_exposure=0.1),
        DefenseVariant(name="ma120_10pct_keep", rule="ma120", risk_off_exposure=0.1),
        DefenseVariant(name="ma150_10pct_keep", rule="ma150", risk_off_exposure=0.1),
        DefenseVariant(name="ma180_10pct_keep", rule="ma180", risk_off_exposure=0.1),
        DefenseVariant(name="ma200_10pct_keep", rule="ma200", risk_off_exposure=0.1),
        DefenseVariant(name="ma220_10pct_keep", rule="ma220", risk_off_exposure=0.1),
        DefenseVariant(name="ma250_10pct_keep", rule="ma250", risk_off_exposure=0.1),
        DefenseVariant(name="ma60_ret20_cash", rule="ma60_ret20", risk_off_exposure=0.0),
        DefenseVariant(name="ma120_ret20_cash", rule="ma120_ret20", risk_off_exposure=0.0),
        DefenseVariant(name="ma60_ret20_10pct_keep", rule="ma60_ret20", risk_off_exposure=0.1),
        DefenseVariant(name="ma120_ret20_10pct_keep", rule="ma120_ret20", risk_off_exposure=0.1),
        DefenseVariant(name="panic_ma60_dd8_cash", rule="panic_ma60_dd8", risk_off_exposure=0.0),
        DefenseVariant(name="panic_ma60_dd8_10pct_keep", rule="panic_ma60_dd8", risk_off_exposure=0.1),
        DefenseVariant(name="panic_ma120_dd10_cash", rule="panic_ma120_dd10", risk_off_exposure=0.0),
        DefenseVariant(name="panic_ma120_dd10_10pct_keep", rule="panic_ma120_dd10", risk_off_exposure=0.1),
    )


def simulate_0050_defense(
    *,
    name: str,
    prices: pd.DataFrame,
    start_date: str,
    end_date: str,
    initial_cash: float,
    cost_model: TaiwanCostModel,
    variant: DefenseVariant,
    dividend_series: pd.Series | None = None,
) -> BacktestResult:
    trade_dates = _trade_dates(prices, start_date, end_date)
    if not trade_dates:
        raise ValueError(f"No trade dates for {DEFENSE_TICKER} between {start_date} and {end_date}")

    account = _Account(cash=float(initial_cash))
    trades: list[Trade] = []
    equity_rows: list[dict] = []
    prices_by_ticker = {DEFENSE_TICKER: prices}
    asset_types = {DEFENSE_TICKER: "etf"}

    for trade_date in trade_dates:
        if account.ticker is not None and dividend_series is not None:
            dividend = float(dividend_series.get(trade_date, 0.0))
            if dividend > 0:
                amount = account.shares * dividend
                account.cash += amount
                trades.append(
                    Trade(
                        date=_date_str(trade_date),
                        ticker=DEFENSE_TICKER,
                        action="dividend",
                        shares=account.shares,
                        price=dividend,
                        gross_amount=amount,
                        costs=0,
                        cash_after=account.cash,
                        reason="cash_dividend",
                    )
                )

        signal_date = previous_available_date(prices_by_ticker, trade_date)
        target_exposure = variant.risk_on_exposure if risk_on_for_rule(prices, signal_date, variant.rule) else variant.risk_off_exposure
        target = DEFENSE_TICKER if target_exposure > 0 else None
        _rebalance(
            account=account,
            trades=trades,
            trade_date=trade_date,
            target=target,
            target_exposure=target_exposure,
            prices_by_ticker=prices_by_ticker,
            asset_types=asset_types,
            cost_model=cost_model,
            reason=f"0050_bear_defense_{variant.name}",
        )

        close_price = float(prices.loc[trade_date, "close"])
        equity_rows.append(
            {
                "date": trade_date,
                "total_value": _market_value(account, {DEFENSE_TICKER: close_price}),
                "current_ticker": account.ticker or "cash",
                "target_exposure": target_exposure,
                "signal_date": signal_date.strftime("%Y-%m-%d"),
            }
        )

    equity_curve = pd.DataFrame(equity_rows).set_index("date")
    final_value = float(equity_curve["total_value"].iloc[-1])
    return BacktestResult(
        name=name,
        final_value=final_value,
        total_return=final_value / initial_cash - 1,
        max_drawdown=_max_drawdown(equity_curve["total_value"]),
        trades=trades,
        equity_curve=equity_curve,
    )


def risk_on_for_rule(prices: pd.DataFrame, signal_date: pd.Timestamp, rule: str) -> bool:
    history = prices.loc[prices.index <= signal_date, "adj_close"].dropna()
    if rule.startswith("ma") and rule[2:].isdigit():
        return _above_ma(history, int(rule[2:]))
    if rule == "ma60_ret20":
        return _above_ma(history, 60) and _return(history, 20) > 0
    if rule == "ma120_ret20":
        return _above_ma(history, 120) and _return(history, 20) > 0
    if rule == "panic_ma60_dd8":
        return not (_below_ma(history, 60) and _return(history, 20) < 0 and _drawdown(history, 60) <= -0.08)
    if rule == "panic_ma120_dd10":
        return not (_below_ma(history, 120) and _return(history, 20) < 0 and _drawdown(history, 120) <= -0.10)
    raise ValueError(f"Unsupported defense rule: {rule}")


def _risk_on(prices: pd.DataFrame, signal_date: pd.Timestamp, rule: str) -> bool:
    return risk_on_for_rule(prices, signal_date, rule)


def _above_ma(history: pd.Series, window: int) -> bool:
    if len(history) < window:
        return True
    return float(history.iloc[-1]) >= float(history.iloc[-window:].mean())


def _below_ma(history: pd.Series, window: int) -> bool:
    if len(history) < window:
        return False
    return float(history.iloc[-1]) < float(history.iloc[-window:].mean())


def _return(history: pd.Series, window: int) -> float:
    if len(history) <= window:
        return 0.0
    base = float(history.iloc[-window - 1])
    if base <= 0:
        return 0.0
    return float(history.iloc[-1]) / base - 1


def _drawdown(history: pd.Series, window: int) -> float:
    if len(history) < window:
        return 0.0
    recent = history.iloc[-window:]
    peak = float(recent.max())
    if peak <= 0:
        return 0.0
    return float(recent.iloc[-1]) / peak - 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest 0050 bear-defense anchor candidates.")
    parser.add_argument("--config", default="configs/ep05_universe.json")
    parser.add_argument("--cache-dir", default="backtest_cache")
    parser.add_argument("--output-dir", default="outputs/bear_defense_backtest_v1")
    parser.add_argument("--periods", default=DEFAULT_PERIODS)
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_periods = _selected_periods(args.periods)
    start_for_download = min(pd.Timestamp(start) for start, _, _ in selected_periods.values())
    end_for_download = max(pd.Timestamp(end) for _, end, _ in selected_periods.values())
    download_start = (start_for_download - pd.DateOffset(years=2)).strftime("%Y-%m-%d")
    prices = _load_best_cached_prices(args.cache_dir, download_start, end_for_download.strftime("%Y-%m-%d"))
    if prices is None:
        prices_by_ticker = download_yfinance_prices(
            tickers=[DEFENSE_TICKER],
            start_date=download_start,
            end_date=end_for_download.strftime("%Y-%m-%d"),
            cache_dir=args.cache_dir,
        )
        prices = prices_by_ticker[DEFENSE_TICKER]
    dividends = split_adjusted_dividends(prices, config.manual_splits.get(DEFENSE_TICKER, ()))

    summary_rows: list[dict] = []
    trade_rows: list[dict] = []
    equity_rows: list[dict] = []
    for period_id, (start, end, period_label) in selected_periods.items():
        benchmark = simulate_buy_and_hold(
            name="0050_buy_and_hold",
            ticker=DEFENSE_TICKER,
            asset_type="etf",
            prices=prices,
            start_date=start,
            end_date=end,
            initial_cash=config.initial_cash_twd,
            cost_model=config.cost_model,
            dividend_series=dividends,
        )
        candidates: list[tuple[str, DefenseVariant | None, BacktestResult]] = [("0050買進持有", None, benchmark)]
        for variant in default_defense_variants():
            result = simulate_0050_defense(
                name=variant.name,
                prices=prices,
                start_date=start,
                end_date=end,
                initial_cash=config.initial_cash_twd,
                cost_model=config.cost_model,
                variant=variant,
                dividend_series=dividends,
            )
            candidates.append((f"0050防守_{variant.name}", variant, result))
        for display_name, variant, result in candidates:
            summary_rows.append(_summary_row(period_id, period_label, display_name, variant, result))
            trade_rows.extend(_trade_rows(period_id, display_name, result))
            equity_rows.extend(_equity_rows(period_id, display_name, result))

    summary = pd.DataFrame(summary_rows)
    ranking = _ranking(summary)
    trades = pd.DataFrame(trade_rows)
    equity = pd.DataFrame(equity_rows)
    summary.to_csv(output_dir / "bear_defense_summary.csv", index=False, encoding="utf-8-sig")
    ranking.to_csv(output_dir / "bear_defense_ranking.csv", index=False, encoding="utf-8-sig")
    trades.to_csv(output_dir / "bear_defense_trades.csv", index=False, encoding="utf-8-sig")
    equity.to_csv(output_dir / "bear_defense_equity_curve.csv", index=False, encoding="utf-8-sig")
    _write_report(output_dir / "bear_defense_report.md", summary, ranking)
    (output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "ticker": DEFENSE_TICKER,
                "periods": selected_periods,
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


def _load_best_cached_prices(cache_dir: str, start_date: str, end_date: str) -> pd.DataFrame | None:
    ticker_file = f"{DEFENSE_TICKER.replace('.', '_')}.csv"
    candidates = [Path(cache_dir) / ticker_file, *Path(cache_dir).rglob(ticker_file)]
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    usable: list[pd.DataFrame] = []
    for path in candidates:
        if not path.exists():
            continue
        frame = load_price_csv(path)
        if frame.index.min() <= start and frame.index.max() >= end:
            usable.append(frame)
    if not usable:
        return None
    return max(usable, key=lambda frame: len(frame))


def _summary_row(
    period_id: str,
    period_label: str,
    display_name: str,
    variant: DefenseVariant | None,
    result: BacktestResult,
) -> dict:
    return {
        "period_id": period_id,
        "period_label": period_label,
        "strategy_name": display_name,
        "variant": variant.name if variant else "buy_and_hold",
        "rule": variant.rule if variant else "buy_and_hold",
        "risk_off_exposure_pct": round((variant.risk_off_exposure if variant else 1.0) * 100, 2),
        "final_value_twd": round(result.final_value, 2),
        "total_return_pct": round(result.total_return * 100, 2),
        "max_drawdown_pct": round(result.max_drawdown * 100, 2),
        "trade_count": sum(1 for trade in result.trades if trade.action in {"buy", "sell"}),
    }


def _trade_rows(period_id: str, strategy_name: str, result: BacktestResult) -> list[dict]:
    rows = []
    for index, trade in enumerate(result.trades, start=1):
        rows.append(
            {
                "period_id": period_id,
                "strategy_name": strategy_name,
                "sequence": index,
                "date": trade.date,
                "ticker": trade.ticker,
                "label": "0050",
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


def _equity_rows(period_id: str, strategy_name: str, result: BacktestResult) -> list[dict]:
    rows = []
    for date, row in result.equity_curve.iterrows():
        rows.append(
            {
                "period_id": period_id,
                "strategy_name": strategy_name,
                "date": date.strftime("%Y-%m-%d"),
                "total_value_twd": round(float(row["total_value"]), 2),
                "current_ticker": row.get("current_ticker", ""),
                "target_exposure": row.get("target_exposure", ""),
                "signal_date": row.get("signal_date", ""),
            }
        )
    return rows


def _ranking(summary: pd.DataFrame) -> pd.DataFrame:
    pivot = summary.pivot_table(
        index=["strategy_name", "variant", "rule", "risk_off_exposure_pct"],
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
        "# 0050 空頭防守錨點回測",
        "",
        "本報告只測 0050 防守錨點：用前一交易日收盤後已知的 0050 趨勢，決定隔日開盤持有 0050、降到 10% 留倉，或全現金。這是 AI 輔助回測與策略驗證，不是投資建議。",
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
