from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt

from backtest_lab.config import load_config
from backtest_lab.data import download_yfinance_prices, load_price_csv, split_adjusted_dividends
from backtest_lab.frozen_report_pdf import _configure_chinese_font


DEFAULT_OUTPUT_DIR = "outputs/ep10_0050_0056_cashflow_growth_material_pack_202105_202605"
DEFAULT_CACHE_DIR = "backtest_cache/ep10_0050_0056_cashflow_growth"
DEFAULT_CONFIG = "configs/ep05_universe.json"
DEFAULT_END_DATE = "2026-05-26"
TICKERS = ("0050.TW", "0056.TW")
LABELS = {"0050.TW": "0050", "0056.TW": "0056"}
INITIAL_CAPITALS = (500_000, 1_000_000, 5_000_000)
ALLOCATIONS = (
    ("0050_100", 1.00, 0.00),
    ("0050_75_0056_25", 0.75, 0.25),
    ("0050_50_0056_50", 0.50, 0.50),
    ("0050_25_0056_75", 0.25, 0.75),
    ("0056_100", 0.00, 1.00),
)


@dataclass(frozen=True)
class PeriodSpec:
    period_id: str
    label: str
    start_date: str
    end_date: str


@dataclass
class HoldingState:
    ticker: str
    shares: int = 0
    cash: float = 0.0
    withdrawn_income: float = 0.0
    reinvested_dividend: float = 0.0
    withdrawn_dividend: float = 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Build EP10 0050/0056 cashflow-growth material pack.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--end-date", default=DEFAULT_END_DATE)
    parser.add_argument("--download-start-date", default="2008-01-01")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_step(output_dir, "loading_inputs")

    config = load_config(args.config)
    raw_prices = _load_or_download_prices(
        tickers=TICKERS,
        cache_dir=Path(args.cache_dir),
        start_date=args.download_start_date,
        end_date=args.end_date,
    )
    prices, price_repair_notes = _repair_price_frames(raw_prices)
    dividends = {
        ticker: split_adjusted_dividends(raw_prices[ticker], config.manual_splits.get(ticker, ()))
        for ticker in TICKERS
    }
    periods = _build_periods(prices, args.end_date)
    broker_fee_rate = config.cost_model.broker_fee_rate
    broker_fee_discount = config.cost_model.broker_fee_discount
    minimum_fee_twd = config.cost_model.minimum_fee_twd

    summary_rows: list[dict] = []
    cashflow_rows: list[dict] = []
    equity_frames: list[pd.DataFrame] = []
    trade_rows: list[dict] = []
    run_rows: list[dict] = []

    for period in periods:
        for initial_capital in INITIAL_CAPITALS:
            for allocation_id, weight_0050, weight_0056 in ALLOCATIONS:
                variant_id = f"{period.period_id}__{initial_capital:.0f}__{allocation_id}"
                _write_step(output_dir, f"running_{variant_id}")
                try:
                    result = simulate_cashflow_growth_portfolio(
                        prices_by_ticker=prices,
                        dividend_by_ticker=dividends,
                        start_date=period.start_date,
                        end_date=period.end_date,
                        initial_capital=initial_capital,
                        weight_0050=weight_0050,
                        weight_0056=weight_0056,
                        broker_fee_rate=broker_fee_rate,
                        broker_fee_discount=broker_fee_discount,
                        minimum_fee_twd=minimum_fee_twd,
                    )
                except Exception as exc:
                    run_rows.append({"variant_id": variant_id, "status": "failed", "message": str(exc)})
                    _write_run_log(output_dir, run_rows)
                    continue

                summary_rows.append(
                    {
                        "period_id": period.period_id,
                        "period_label": period.label,
                        "start_date": period.start_date,
                        "end_date": period.end_date,
                        "initial_capital_twd": initial_capital,
                        "allocation_id": allocation_id,
                        "0050_weight_pct": round(weight_0050 * 100, 2),
                        "0056_weight_pct": round(weight_0056 * 100, 2),
                        **_summary_from_result(result),
                    }
                )
                cashflow_rows.extend(result["cashflow_rows"])
                trade_rows.extend(result["trade_rows"])
                curve = result["equity_curve"].copy()
                curve.insert(0, "allocation_id", allocation_id)
                curve.insert(0, "initial_capital_twd", initial_capital)
                curve.insert(0, "period_label", period.label)
                curve.insert(0, "period_id", period.period_id)
                equity_frames.append(curve)
                run_rows.append({"variant_id": variant_id, "status": "completed", "message": ""})
                _write_run_log(output_dir, run_rows)

    _write_step(output_dir, "writing_outputs")
    summary = pd.DataFrame(summary_rows)
    cashflows = pd.DataFrame(cashflow_rows)
    trades = pd.DataFrame(trade_rows)
    equity = pd.concat(equity_frames, ignore_index=True) if equity_frames else pd.DataFrame()

    summary.to_csv(output_dir / "strategy_summary.csv", index=False, encoding="utf-8-sig")
    cashflows.to_csv(output_dir / "cashflow_summary.csv", index=False, encoding="utf-8-sig")
    equity.to_csv(output_dir / "equity_curves.csv", index=False, encoding="utf-8-sig")
    trades.to_csv(output_dir / "trade_log.csv", index=False, encoding="utf-8-sig")

    chart_paths = _write_charts(output_dir, summary, cashflows, equity)
    _write_readme(output_dir, summary, chart_paths, price_repair_notes)
    _write_source_notes(output_dir, summary, chart_paths)
    _write_planning_handoff_prompt(output_dir, summary, chart_paths)
    _write_first_12_pages_regen_check(output_dir)
    _write_manifest(output_dir, args, periods, chart_paths, price_repair_notes)
    _write_step(output_dir, "completed")


def simulate_cashflow_growth_portfolio(
    *,
    prices_by_ticker: dict[str, pd.DataFrame],
    dividend_by_ticker: dict[str, pd.Series],
    start_date: str,
    end_date: str,
    initial_capital: float,
    weight_0050: float,
    weight_0056: float,
    broker_fee_rate: float,
    broker_fee_discount: float,
    minimum_fee_twd: int,
) -> dict:
    data_by_ticker = {
        ticker: prices.loc[(prices.index >= pd.Timestamp(start_date)) & (prices.index <= pd.Timestamp(end_date))].copy()
        for ticker, prices in prices_by_ticker.items()
    }
    if any(frame.empty for frame in data_by_ticker.values()):
        raise ValueError(f"Missing price data in {start_date}~{end_date}")
    common_dates = sorted(set(data_by_ticker["0050.TW"].index).intersection(data_by_ticker["0056.TW"].index))
    if not common_dates:
        raise ValueError(f"No common trading dates in {start_date}~{end_date}")

    states = {"0050.TW": HoldingState("0050.TW"), "0056.TW": HoldingState("0056.TW")}
    residual_cash = 0.0
    trade_rows: list[dict] = []
    cashflow_rows: list[dict] = []
    equity_rows: list[dict] = []

    weights = {"0050.TW": weight_0050, "0056.TW": weight_0056}
    first_date = common_dates[0]
    for ticker, weight in weights.items():
        budget = initial_capital * weight
        if budget <= 0:
            continue
        price = _valuation_price(data_by_ticker[ticker], ticker, first_date, prefer_open=True)
        buy = _buy_with_budget(
            date=first_date,
            ticker=ticker,
            budget=budget,
            cash=budget,
            price=price,
            broker_fee_rate=broker_fee_rate,
            broker_fee_discount=broker_fee_discount,
            minimum_fee_twd=minimum_fee_twd,
            reason="initial_allocation",
        )
        if buy is None:
            states[ticker].cash += budget
            continue
        states[ticker].shares += int(buy["shares"])
        states[ticker].cash += budget - buy["gross_amount"] - buy["cost"]
        trade_rows.append(buy)

    for current_date in common_dates:
        for ticker, state in states.items():
            dividend = float(dividend_by_ticker[ticker].get(current_date, 0.0))
            if state.shares <= 0 or dividend <= 0:
                continue
            amount = state.shares * dividend
            if ticker == "0050.TW":
                # 0050 uses adj_close as a total-return proxy, so dividends are
                # already reflected as reinvested return and must not be credited again.
                continue
            if ticker == "0056.TW":
                state.withdrawn_income += amount
                state.withdrawn_dividend += amount
                cashflow_rows.append(_cashflow_row(current_date, ticker, amount, "withdrawn"))

        invested_value = residual_cash
        market_values: dict[str, float] = {}
        for ticker, state in states.items():
            close = _valuation_price(data_by_ticker[ticker], ticker, current_date)
            market_value = state.shares * close + state.cash
            market_values[ticker] = market_value
            invested_value += market_value
        withdrawn_income = sum(state.withdrawn_income for state in states.values())
        total_wealth = invested_value + withdrawn_income
        equity_rows.append(
            {
                "date": current_date.strftime("%Y-%m-%d"),
                "invested_value_twd": invested_value,
                "total_wealth_with_withdrawn_income_twd": total_wealth,
                "withdrawn_income_twd": withdrawn_income,
                "0050_value_twd": market_values["0050.TW"],
                "0056_value_twd": market_values["0056.TW"],
                "0050_shares": states["0050.TW"].shares,
                "0056_shares": states["0056.TW"].shares,
            }
        )

    equity = pd.DataFrame(equity_rows)
    stats = _portfolio_stats(equity["invested_value_twd"])
    total_stats = _portfolio_stats(equity["total_wealth_with_withdrawn_income_twd"])
    return {
        "initial_capital": initial_capital,
        "final_invested_value": float(equity["invested_value_twd"].iloc[-1]),
        "final_total_wealth": float(equity["total_wealth_with_withdrawn_income_twd"].iloc[-1]),
        "withdrawn_income": float(equity["withdrawn_income_twd"].iloc[-1]),
        "reinvested_dividend": states["0050.TW"].reinvested_dividend,
        "withdrawn_dividend": states["0056.TW"].withdrawn_dividend,
        "max_drawdown_invested": stats["max_drawdown"],
        "max_drawdown_total": total_stats["max_drawdown"],
        "equity_curve": equity,
        "trade_rows": trade_rows,
        "cashflow_rows": cashflow_rows,
    }


def _load_or_download_prices(
    *,
    tickers: tuple[str, ...],
    cache_dir: Path,
    start_date: str,
    end_date: str,
) -> dict[str, pd.DataFrame]:
    prices: dict[str, pd.DataFrame] = {}
    missing: list[str] = []
    for ticker in tickers:
        path = cache_dir / f"{ticker.replace('.', '_')}.csv"
        if path.exists():
            try:
                prices[ticker] = load_price_csv(path)
                continue
            except Exception:
                missing.append(ticker)
                continue
        missing.append(ticker)
    if missing:
        downloaded = download_yfinance_prices(list(tickers), start_date, end_date, cache_dir)
        prices.update(downloaded)
    return prices


def _repair_price_frames(prices: dict[str, pd.DataFrame]) -> tuple[dict[str, pd.DataFrame], list[dict]]:
    repaired: dict[str, pd.DataFrame] = {}
    notes: list[dict] = []
    for ticker, frame in prices.items():
        if ticker != "0050.TW":
            repaired[ticker] = frame
            continue
        fixed, ticker_notes = repair_split_like_price_jumps(frame, ticker=ticker)
        repaired[ticker] = fixed
        notes.extend(ticker_notes)
    return repaired, notes


def repair_split_like_price_jumps(frame: pd.DataFrame, *, ticker: str) -> tuple[pd.DataFrame, list[dict]]:
    adjusted = frame.copy()
    notes: list[dict] = []
    returns = adjusted["close"].astype(float).pct_change()
    for jump_date, daily_return in returns[returns <= -0.5].items():
        previous_position = adjusted.index.get_loc(jump_date) - 1
        if previous_position < 0:
            continue
        prev_date = adjusted.index[previous_position]
        prev_close = float(adjusted.loc[prev_date, "close"])
        current_close = float(adjusted.loc[jump_date, "close"])
        if current_close <= 0:
            continue
        raw_ratio = prev_close / current_close
        rounded_ratio = round(raw_ratio)
        if rounded_ratio < 2 or abs(raw_ratio - rounded_ratio) / rounded_ratio > 0.08:
            continue
        mask = adjusted.index < jump_date
        for column in ("open", "high", "low", "close", "adj_close"):
            if column in adjusted.columns:
                adjusted.loc[mask, column] = adjusted.loc[mask, column].astype(float) / rounded_ratio
        notes.append(
            {
                "ticker": ticker,
                "jump_date": pd.Timestamp(jump_date).strftime("%Y-%m-%d"),
                "previous_date": pd.Timestamp(prev_date).strftime("%Y-%m-%d"),
                "raw_return_pct": round(float(daily_return) * 100, 4),
                "applied_ratio": rounded_ratio,
                "reason": "close series contained a split-like discontinuity not represented in stock_split.",
            }
        )
    return adjusted, notes


def _valuation_price(frame: pd.DataFrame, ticker: str, date: pd.Timestamp, prefer_open: bool = False) -> float:
    row = frame.loc[date]
    if ticker == "0050.TW":
        return float(row["adj_close"])
    if prefer_open:
        return float(row["open"])
    return float(row["close"])


def _build_periods(prices: dict[str, pd.DataFrame], end_date: str) -> tuple[PeriodSpec, ...]:
    end = _last_common_date(prices, pd.Timestamp(end_date))
    first_common = max(frame.index.min() for frame in prices.values())
    starts = {
        "full": first_common,
        "15y": end - pd.DateOffset(years=15),
        "10y": end - pd.DateOffset(years=10),
        "5y": end - pd.DateOffset(years=5),
        "2022": pd.Timestamp("2022-01-03"),
    }
    labels = {
        "full": "完整可用樣本",
        "15y": "近15年",
        "10y": "近10年",
        "5y": "近5年",
        "2022": "2022壓力測試",
    }
    periods: list[PeriodSpec] = []
    for period_id, start in starts.items():
        actual_start = _first_common_date(prices, max(first_common, start))
        actual_end = _last_common_date(prices, pd.Timestamp("2022-12-30") if period_id == "2022" else end)
        periods.append(
            PeriodSpec(period_id, labels[period_id], actual_start.strftime("%Y-%m-%d"), actual_end.strftime("%Y-%m-%d"))
        )
    return tuple(periods)


def _first_common_date(prices: dict[str, pd.DataFrame], start: pd.Timestamp) -> pd.Timestamp:
    common = set(prices["0050.TW"].index)
    for frame in prices.values():
        common &= set(frame.index)
    candidates = sorted(date for date in common if date >= start)
    if not candidates:
        raise ValueError(f"No common date on or after {start.date()}")
    return candidates[0]


def _last_common_date(prices: dict[str, pd.DataFrame], end: pd.Timestamp) -> pd.Timestamp:
    common = set(prices["0050.TW"].index)
    for frame in prices.values():
        common &= set(frame.index)
    candidates = sorted((date for date in common if date <= end), reverse=True)
    if not candidates:
        raise ValueError(f"No common date on or before {end.date()}")
    return candidates[0]


def _summary_from_result(result: dict) -> dict:
    initial = float(result["initial_capital"])
    years = max(1e-9, len(result["equity_curve"]) / 252)
    return {
        "final_invested_value_twd": round(result["final_invested_value"], 2),
        "withdrawn_income_twd": round(result["withdrawn_income"], 2),
        "final_total_wealth_twd": round(result["final_total_wealth"], 2),
        "invested_return_pct": round((result["final_invested_value"] / initial - 1) * 100, 2),
        "total_wealth_return_pct": round((result["final_total_wealth"] / initial - 1) * 100, 2),
        "avg_annual_withdrawn_income_twd": round(result["withdrawn_income"] / years, 2),
        "income_to_initial_pct": round(result["withdrawn_income"] / initial * 100, 2),
        "0050_reinvested_dividend_twd": round(result["reinvested_dividend"], 2),
        "0056_withdrawn_dividend_twd": round(result["withdrawn_dividend"], 2),
        "max_drawdown_invested_pct": round(result["max_drawdown_invested"] * 100, 2),
        "max_drawdown_total_wealth_pct": round(result["max_drawdown_total"] * 100, 2),
    }


def _buy_with_budget(
    *,
    date: pd.Timestamp,
    ticker: str,
    budget: float,
    cash: float,
    price: float,
    broker_fee_rate: float,
    broker_fee_discount: float,
    minimum_fee_twd: int,
    reason: str,
) -> dict | None:
    if budget <= 0 or cash <= 0 or price <= 0:
        return None
    shares = int(min(budget, cash) // price)
    while shares > 0:
        gross = shares * price
        cost = max(minimum_fee_twd, int(round(gross * broker_fee_rate * broker_fee_discount)))
        if gross + cost <= cash and gross + cost <= budget + minimum_fee_twd:
            return {
                "date": date.strftime("%Y-%m-%d"),
                "ticker": ticker,
                "asset_label": LABELS[ticker],
                "action": "buy",
                "shares": shares,
                "price": round(price, 4),
                "gross_amount": round(gross, 2),
                "cost": cost,
                "reason": reason,
            }
        shares -= 1
    return None


def _cashflow_row(date: pd.Timestamp, ticker: str, amount: float, treatment: str) -> dict:
    return {
        "date": date.strftime("%Y-%m-%d"),
        "ticker": ticker,
        "asset_label": LABELS[ticker],
        "dividend_amount_twd": round(amount, 2),
        "treatment": treatment,
    }


def _portfolio_stats(values: pd.Series) -> dict:
    series = values.astype(float).reset_index(drop=True)
    peaks = series.cummax()
    drawdowns = series / peaks - 1
    return {"max_drawdown": float(drawdowns.min())}


def _write_charts(
    output_dir: Path,
    summary: pd.DataFrame,
    cashflows: pd.DataFrame,
    equity: pd.DataFrame,
) -> list[str]:
    _configure_chinese_font()
    chart_dir = output_dir / "charts"
    chart_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []

    for period_id in ("5y", "10y", "15y", "2022"):
        period_summary = summary[(summary["period_id"] == period_id) & (summary["initial_capital_twd"] == 1_000_000)]
        if period_summary.empty:
            continue
        labels = period_summary["allocation_id"].map(_allocation_label)
        fig, ax = plt.subplots(figsize=(10.5, 5.8))
        ax.bar(labels, period_summary["final_total_wealth_twd"] / 10_000, color="#b42318")
        ax.set_title(f"100萬投入：0050/0056 配置期末總財富 - {period_summary['period_label'].iloc[0]}")
        ax.set_ylabel("期末總財富（萬元，含0056已領出現金流）")
        ax.tick_params(axis="x", labelrotation=20)
        ax.grid(axis="y", alpha=0.25)
        path = chart_dir / f"{period_id}_allocation_total_wealth_1m.png"
        fig.tight_layout()
        fig.savefig(path, dpi=170)
        plt.close(fig)
        paths.append(str(path))

    five_year_equity = equity[
        (equity["period_id"] == "5y")
        & (equity["initial_capital_twd"] == 1_000_000)
        & (equity["allocation_id"].isin(["0050_100", "0050_50_0056_50", "0056_100"]))
    ].copy()
    if not five_year_equity.empty:
        fig, ax = plt.subplots(figsize=(10.5, 5.8))
        for allocation_id, rows in five_year_equity.groupby("allocation_id"):
            rows = rows.sort_values("date")
            ax.plot(
                pd.to_datetime(rows["date"]),
                rows["total_wealth_with_withdrawn_income_twd"] / 10_000,
                label=_allocation_label(allocation_id),
                linewidth=2.1,
            )
        ax.set_title("近5年：總財富曲線（100萬起算，含0056已領現金流）")
        ax.set_ylabel("萬元")
        ax.grid(True, alpha=0.25)
        ax.legend()
        ax.tick_params(axis="x", labelrotation=20)
        path = chart_dir / "5y_key_allocations_total_wealth_curve.png"
        fig.tight_layout()
        fig.savefig(path, dpi=170)
        plt.close(fig)
        paths.append(str(path))

    income = (
        summary[(summary["period_id"] == "5y") & (summary["initial_capital_twd"] == 1_000_000)]
        .sort_values("0056_weight_pct")
        .copy()
    )
    if not income.empty:
        fig, ax = plt.subplots(figsize=(10, 5.2))
        ax.bar(income["allocation_id"].map(_allocation_label), income["avg_annual_withdrawn_income_twd"] / 10_000, color="#c77917")
        ax.set_title("近5年：0056領出現金流平均每年金額（100萬起算）")
        ax.set_ylabel("萬元／年")
        ax.tick_params(axis="x", labelrotation=20)
        ax.grid(axis="y", alpha=0.25)
        path = chart_dir / "5y_avg_annual_income_1m.png"
        fig.tight_layout()
        fig.savefig(path, dpi=170)
        plt.close(fig)
        paths.append(str(path))

    return paths


def _allocation_label(allocation_id: str) -> str:
    labels = {
        "0050_100": "100% 0050",
        "0050_75_0056_25": "75% 0050 / 25% 0056",
        "0050_50_0056_50": "50% 0050 / 50% 0056",
        "0050_25_0056_75": "25% 0050 / 75% 0056",
        "0056_100": "100% 0056",
    }
    return labels.get(allocation_id, allocation_id)


def _write_readme(output_dir: Path, summary: pd.DataFrame, chart_paths: list[str], price_repair_notes: list[dict]) -> None:
    recent = summary[summary["period_id"] == "5y"].copy()
    recent_1m = recent[recent["initial_capital_twd"] == 1_000_000].copy()
    recent_1m = recent_1m.sort_values("final_total_wealth_twd", ascending=False)
    display_cols = [
        "allocation_id",
        "final_invested_value_twd",
        "withdrawn_income_twd",
        "final_total_wealth_twd",
        "total_wealth_return_pct",
        "avg_annual_withdrawn_income_twd",
        "max_drawdown_invested_pct",
    ]
    lines = [
        "# EP10 0050＋0056 家庭現金流與資產成長回測素材包",
        "",
        "定位：AI 輔助資產配置情境回測，不是投資建議。",
        "",
        "本版主素材口徑：只使用近五年區間，約 2021/5 到 2026/5。長週期與 2022 壓力測試資料仍保留在 CSV 中備查，但不作為 EP10 主敘事。",
        "",
        "## 回測口徑",
        "",
        "- 一次投入，不做定期定額。",
        "- 配置比例只在期初設定，期間不再平衡。",
        "- 0050 使用 adj_close 作為配息再投入後的總報酬近似口徑。",
        "- 0056 配息以資料源 dividend date 近似，視為領出補貼家用，不再投入。",
        "- 交易成本使用台灣券商手續費口徑；0050 因使用總報酬 proxy，僅期初買進扣手續費，不逐次模擬配息再投入手續費。",
        "- 若資料源出現未記錄的拆分型價格斷層，runner 會自動以接近整數倍的跳點修正前段價格。",
        "",
        "## 近五年主摘要：100 萬起算",
        "",
        _markdown_table(recent_1m[display_cols]),
        "",
        "## 近五年主摘要：全部起始金",
        "",
        _markdown_table(recent[["initial_capital_twd", *display_cols]]),
        "",
        "## 圖表清單",
        "",
        *[f"- `{path}`" for path in chart_paths if "\\5y_" in path or "/5y_" in path],
        "",
        "## 資料修正紀錄",
        "",
        _markdown_table(pd.DataFrame(price_repair_notes)) if price_repair_notes else "無。",
        "",
        "## 使用邊界",
        "",
        "- 本素材包只呈現歷史回測，不代表未來會重複。",
        "- 配息日以資料源欄位近似，不等同精準現金入帳日。",
        "- 0056 現金流領出後若實際花掉，生活現金流提高，但帳戶內可複利資金會下降。",
    ]
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def _write_source_notes(output_dir: Path, summary: pd.DataFrame, chart_paths: list[str]) -> None:
    recent = summary[(summary["period_id"] == "5y") & (summary["initial_capital_twd"] == 1_000_000)].copy()
    recent = recent.sort_values("final_total_wealth_twd", ascending=False)
    lines = [
        "EP10 sourceNotes：0050＋0056，家庭現金流與資產成長能不能兼顧？",
        "",
        "這份資料包可供企劃、旁白、GPT 繪圖與剪輯使用。內容定位是 AI 輔助回測與資產配置情境試算，不是投資建議。",
        "",
        "核心設定：0050 使用配息再投入總報酬近似，0056 配息領出補貼家用；主敘事使用近五年，約 2021/5 到 2026/5，並以 100 萬起算的五種配置比例作為主要畫面素材。",
        "",
        "近5年 100萬起算排序：",
    ]
    for _, row in recent.iterrows():
        lines.append(
            f"- {_allocation_label(row.allocation_id)}：期末總財富 {row.final_total_wealth_twd:,.0f} 元，"
            f"已領現金流 {row.withdrawn_income_twd:,.0f} 元，"
            f"平均每年現金流 {row.avg_annual_withdrawn_income_twd:,.0f} 元，"
            f"帳戶回撤 {row.max_drawdown_invested_pct:+.2f}%。"
        )
    lines.extend(
        [
            "",
            "主圖表檔案：",
            *[f"- {path}" for path in chart_paths if "\\5y_" in path or "/5y_" in path],
        ]
    )
    (output_dir / "creatorflow_source_notes.md").write_text("\n".join(lines), encoding="utf-8")


def _write_planning_handoff_prompt(output_dir: Path, summary: pd.DataFrame, chart_paths: list[str]) -> None:
    recent = summary[(summary["period_id"] == "5y") & (summary["initial_capital_twd"] == 1_000_000)].copy()
    recent = recent.sort_values("final_total_wealth_twd", ascending=False)
    bullets = []
    for _, row in recent.iterrows():
        bullets.append(
            f"- {_allocation_label(row.allocation_id)}：期末總財富 {row.final_total_wealth_twd:,.0f} 元；"
            f"已領現金流 {row.withdrawn_income_twd:,.0f} 元；"
            f"平均每年現金流 {row.avg_annual_withdrawn_income_twd:,.0f} 元；"
            f"最大回撤 {row.max_drawdown_invested_pct:+.2f}%。"
        )

    lines = [
        "# 給另一個 Chat 的 EP10 企劃草稿重製提示詞",
        "",
        "請讀取以下 EP10 素材包，重新生成企劃草稿。重點是修正原企劃中可能使用舊區間或舊敘事的頁面。",
        "",
        f"素材包位置：`{output_dir}`",
        "",
        "## 主題",
        "",
        "房貸車貸生活每月開銷 8-9 萬，該買 0050 還是 0056？我讓 AI 回測：配息補家用，資產還能長大嗎？",
        "",
        "## 必須使用的最新口徑",
        "",
        "- 回測區間改為近五年：約 2021/5 到 2026/5。",
        "- 0050 使用配息再投入的總報酬近似口徑。",
        "- 0056 的配息視為領出補貼家用，不再投入。",
        "- 這是 AI 輔助資產配置情境回測，不是投資建議。",
        "- 旁白要用第一人稱、普通家庭現金流壓力的角度，不要用創作者個人資產規模當主軸。",
        "",
        "## 近五年 100 萬起算核心數據",
        "",
        *bullets,
        "",
        "## 可使用圖表",
        "",
        *[f"- `{path}`" for path in chart_paths if "\\5y_" in path or "/5y_" in path],
        "",
        "## 請輸出",
        "",
        "1. 長影片完整旁白。",
        "2. 長影片 30 頁分頁繪圖提示詞。",
        "3. 短影片旁白。",
        "4. 短影片 7 頁分頁繪圖提示詞。",
        "5. YouTube 長短影片標題、說明、標籤、關鍵字、時間軸、置頂留言。",
        "",
        "## 注意",
        "",
        "- 若原企劃前 12 頁已有使用舊區間、完整樣本、10 年/15 年，或以 50 萬/100 萬/500 萬資金級距當開場鉤子，請改成近五年與家庭月開銷壓力敘事。",
        "- 數據頁請直接使用上方數字，避免重新估算。",
        "- 圖像提示詞要留白給報告或數據卡，不要把小字塞滿畫面。",
    ]
    (output_dir / "ep10_replanning_prompt_for_chatgpt.md").write_text("\n".join(lines), encoding="utf-8")


def _write_first_12_pages_regen_check(output_dir: Path) -> None:
    lines = [
        "# EP10 前 12 頁重生圖檢查建議",
        "",
        "因目前只掌握素材包與最新口徑，未直接讀取已生成的 12 張圖片；以下是給人工 QA 的重製判斷規則。",
        "",
        "## 建議必查頁面",
        "",
        "- P1-P3：若開場仍是 50萬、100萬、500萬資金級距，建議重製；新鉤子應改為家庭每月 8-9 萬開銷、配息能否補家用。",
        "- P4-P8：若畫面文字出現完整樣本、10年、15年、2009 起算，建議重製；EP10 主素材改用近五年 2021/5-2026/5。",
        "- P9-P12：若圖表或數據卡使用舊資料，建議重製；只保留近五年 100 萬起算五種配置比較。",
        "",
        "## 可沿用條件",
        "",
        "- 若畫面只是家庭現金流壓力、0050/0056 概念、AI 回測流程，且沒有舊區間或舊數字，可以沿用。",
        "- 若畫面有留白可後製貼新數據卡，也可沿用底圖，只重貼數據。",
        "",
        "## 新數據來源",
        "",
        "- `creatorflow_source_notes.md`：給企劃與旁白使用。",
        "- `ep10_replanning_prompt_for_chatgpt.md`：給另一個 Chat 重製企劃草稿使用。",
        "- `charts/5y_allocation_total_wealth_1m.png`、`charts/5y_avg_annual_income_1m.png`、`charts/5y_key_allocations_total_wealth_curve.png`：給剪輯與圖像重製使用。",
    ]
    (output_dir / "first_12_pages_regen_check.md").write_text("\n".join(lines), encoding="utf-8")


def _write_manifest(
    output_dir: Path,
    args: argparse.Namespace,
    periods: tuple[PeriodSpec, ...],
    chart_paths: list[str],
    price_repair_notes: list[dict],
) -> None:
    manifest = {
        "task_id": "TASK-EP10-001",
        "output_dir": str(output_dir),
        "cache_dir": args.cache_dir,
        "tickers": list(TICKERS),
        "initial_capitals": list(INITIAL_CAPITALS),
        "allocations": [
            {"allocation_id": allocation_id, "0050_weight": w50, "0056_weight": w56}
            for allocation_id, w50, w56 in ALLOCATIONS
        ],
        "periods": [period.__dict__ for period in periods],
        "price_repair_notes": price_repair_notes,
        "charts": chart_paths,
        "content_boundary": "AI輔助資產配置回測，不是投資建議。",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def _markdown_table(frame: pd.DataFrame) -> str:
    headers = [str(column) for column in frame.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in frame.iterrows():
        values = [_format_markdown_cell(row[column]) for column in frame.columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _format_markdown_cell(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:,.2f}"
    return str(value).replace("|", "／")


def _write_step(output_dir: Path, step: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "current_step.txt").write_text(step + "\n", encoding="utf-8")


def _write_run_log(output_dir: Path, rows: list[dict]) -> None:
    with (output_dir / "run_log.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["variant_id", "status", "message"])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
