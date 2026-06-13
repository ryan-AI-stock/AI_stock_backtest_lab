from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt

from backtest_lab.config import load_config
from backtest_lab.data import load_price_csv, split_adjusted_dividends
from backtest_lab.frozen_report_pdf import _configure_chinese_font


DEFAULT_OUTPUT_DIR = "outputs/ep09_dip_buying_material_pack"
DEFAULT_CACHE_DIR = "backtest_cache/unified_9_asset_full"
DEFAULT_INITIAL_CASH = 1_000_000


@dataclass(frozen=True)
class StrategySpec:
    strategy_id: str
    label: str
    description: str


@dataclass(frozen=True)
class PeriodSpec:
    period_id: str
    label: str
    start_date: str
    end_date: str


STRATEGIES = (
    StrategySpec(
        "blind_dip_ladder",
        "無腦越跌越買",
        "只要從近120日高點跌到階梯就加碼，不看趨勢是否轉弱。",
    ),
    StrategySpec(
        "fixed_time_ladder",
        "固定分批",
        "從起始日開始每20個交易日投入一批，不判斷當下漲跌。",
    ),
    StrategySpec(
        "ai_risk_filtered_ladder",
        "AI風險過濾後才加碼",
        "先等跌幅階梯被觸發；若長期趨勢轉弱則暫停，等風險改善才允許下一批。",
    ),
)

ASSETS = {
    "0050.TW": "0050",
    "00631L.TW": "0050正二",
}

DIP_THRESHOLDS = (-0.05, -0.10, -0.15, -0.20)
TRANCHE_COUNT = 5
FIXED_INTERVAL_TRADING_DAYS = 20


def main() -> None:
    parser = argparse.ArgumentParser(description="Build EP09 dip-buying backtest material pack.")
    parser.add_argument("--config", default="configs/ep05_universe.json")
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--initial-cash", type=float, default=DEFAULT_INITIAL_CASH)
    parser.add_argument("--start-date", default="2019-01-02")
    parser.add_argument("--end-date", default="2026-05-26")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_step(output_dir, "loading_inputs")

    config = load_config(args.config)
    cache_dir = Path(args.cache_dir)
    prices_by_ticker = {
        ticker: load_price_csv(cache_dir / f"{ticker.replace('.', '_')}.csv")
        for ticker in ASSETS
    }
    dividend_by_ticker = {
        ticker: split_adjusted_dividends(prices, config.manual_splits.get(ticker, ()))
        for ticker, prices in prices_by_ticker.items()
    }
    asset_types = {asset.ticker: asset.asset_type for group in config.groups for asset in group.assets}
    periods = _periods(args.start_date, args.end_date)

    summary_rows: list[dict] = []
    trade_rows: list[dict] = []
    equity_frames: list[pd.DataFrame] = []
    run_rows: list[dict] = []

    for period in periods:
        for ticker, label in ASSETS.items():
            for strategy in STRATEGIES:
                variant_id = f"{period.period_id}__{ticker.replace('.TW', '')}__{strategy.strategy_id}"
                _write_step(output_dir, f"running_{variant_id}")
                try:
                    result = simulate_ladder_strategy(
                        ticker=ticker,
                        asset_label=label,
                        asset_type=asset_types[ticker],
                        prices=prices_by_ticker[ticker],
                        dividend_series=dividend_by_ticker[ticker],
                        start_date=period.start_date,
                        end_date=period.end_date,
                        initial_cash=args.initial_cash,
                        strategy=strategy,
                        broker_fee_rate=config.cost_model.broker_fee_rate,
                        broker_fee_discount=config.cost_model.broker_fee_discount,
                        minimum_fee_twd=config.cost_model.minimum_fee_twd,
                        sell_tax_rate=(
                            config.cost_model.etf_sell_tax_rate
                            if asset_types[ticker] == "etf"
                            else config.cost_model.stock_sell_tax_rate
                        ),
                    )
                except Exception as exc:
                    run_rows.append(
                        {
                            "variant_id": variant_id,
                            "status": "failed",
                            "message": str(exc),
                        }
                    )
                    _write_run_log(output_dir, run_rows)
                    continue

                summary_rows.append(
                    {
                        "period_id": period.period_id,
                        "period_label": period.label,
                        "start_date": period.start_date,
                        "end_date": period.end_date,
                        "ticker": ticker,
                        "asset_label": label,
                        "strategy_id": strategy.strategy_id,
                        "strategy_label": strategy.label,
                        "final_value_twd": round(result["final_value"], 2),
                        "total_return_pct": round(result["total_return"] * 100, 2),
                        "max_drawdown_pct": round(result["max_drawdown"] * 100, 2),
                        "longest_underwater_trading_days": result["longest_underwater_days"],
                        "buy_count": result["buy_count"],
                        "blocked_buy_days": result["blocked_buy_days"],
                        "most_painful_start": result["most_painful_start"],
                        "most_painful_end": result["most_painful_end"],
                    }
                )
                trade_rows.extend(result["trades"])
                curve = result["equity_curve"].copy()
                curve.insert(0, "strategy_label", strategy.label)
                curve.insert(0, "strategy_id", strategy.strategy_id)
                curve.insert(0, "asset_label", label)
                curve.insert(0, "ticker", ticker)
                curve.insert(0, "period_id", period.period_id)
                equity_frames.append(curve)
                run_rows.append({"variant_id": variant_id, "status": "completed", "message": ""})
                _write_run_log(output_dir, run_rows)

    _write_step(output_dir, "writing_outputs")
    summary = pd.DataFrame(summary_rows)
    trades = pd.DataFrame(trade_rows)
    equity = pd.concat(equity_frames, ignore_index=True) if equity_frames else pd.DataFrame()

    summary.to_csv(output_dir / "strategy_summary.csv", index=False, encoding="utf-8-sig")
    trades.to_csv(output_dir / "trade_log.csv", index=False, encoding="utf-8-sig")
    equity.to_csv(output_dir / "equity_curves.csv", index=False, encoding="utf-8-sig")

    chart_paths = _write_charts(output_dir, summary, equity, trades)
    _write_report(output_dir, summary, chart_paths, args.initial_cash)
    _write_source_notes(output_dir, summary, chart_paths)
    _write_manifest(output_dir, args, periods, chart_paths)
    _write_step(output_dir, "completed")


def simulate_ladder_strategy(
    *,
    ticker: str,
    asset_label: str,
    asset_type: str,
    prices: pd.DataFrame,
    dividend_series: pd.Series,
    start_date: str,
    end_date: str,
    initial_cash: float,
    strategy: StrategySpec,
    broker_fee_rate: float,
    broker_fee_discount: float,
    minimum_fee_twd: int,
    sell_tax_rate: float,
) -> dict:
    del asset_type, sell_tax_rate
    data = prices.loc[(prices.index >= pd.Timestamp(start_date)) & (prices.index <= pd.Timestamp(end_date))].copy()
    if data.empty:
        raise ValueError(f"No price data for {ticker} in {start_date}~{end_date}")

    cash = float(initial_cash)
    shares = 0
    tranche_index = 0
    pending_buy_reason: str | None = "initial_tranche"
    touched_threshold_index: int | None = None
    blocked_buy_days = 0
    trades: list[dict] = []
    rows: list[dict] = []

    for trade_index, (date, row) in enumerate(data.iterrows()):
        dividend = float(dividend_series.get(date, 0.0))
        if shares > 0 and dividend > 0:
            cash += shares * dividend
            trades.append(
                {
                    "date": date.strftime("%Y-%m-%d"),
                    "ticker": ticker,
                    "asset_label": asset_label,
                    "strategy_id": strategy.strategy_id,
                    "strategy_label": strategy.label,
                    "action": "dividend",
                    "shares": shares,
                    "price": round(dividend, 4),
                    "gross_amount": round(shares * dividend, 2),
                    "cost": 0,
                    "reason": "cash_dividend",
                }
            )

        if pending_buy_reason and tranche_index < TRANCHE_COUNT and cash > 0:
            buy_budget = cash if tranche_index == TRANCHE_COUNT - 1 else min(cash, initial_cash / TRANCHE_COUNT)
            buy = _buy_with_budget(
                date=date,
                ticker=ticker,
                asset_label=asset_label,
                strategy=strategy,
                price=float(row["open"]),
                budget=buy_budget,
                cash=cash,
                reason=pending_buy_reason,
                broker_fee_rate=broker_fee_rate,
                broker_fee_discount=broker_fee_discount,
                minimum_fee_twd=minimum_fee_twd,
            )
            if buy is not None:
                cash -= buy["gross_amount"] + buy["cost"]
                shares += int(buy["shares"])
                tranche_index += 1
                trades.append(buy)
            pending_buy_reason = None

        close = float(row["close"])
        total_value = cash + shares * close
        rows.append(
            {
                "date": date.strftime("%Y-%m-%d"),
                "close": close,
                "shares": shares,
                "cash": cash,
                "total_value": total_value,
                "invested_pct": 1 - cash / initial_cash,
            }
        )

        if tranche_index >= TRANCHE_COUNT or cash <= 0:
            continue

        history = prices.loc[:date]
        if strategy.strategy_id == "fixed_time_ladder":
            if tranche_index > 0 and trade_index > 0 and trade_index % FIXED_INTERVAL_TRADING_DAYS == 0:
                pending_buy_reason = f"fixed_every_{FIXED_INTERVAL_TRADING_DAYS}_trading_days"
            continue

        drawdown = _rolling_high_drawdown(history, window=120)
        next_threshold_index = min(tranche_index - 1, len(DIP_THRESHOLDS) - 1)
        next_threshold = DIP_THRESHOLDS[next_threshold_index]
        if drawdown > next_threshold and touched_threshold_index is None:
            continue
        if drawdown <= next_threshold:
            touched_threshold_index = next_threshold_index

        if strategy.strategy_id == "blind_dip_ladder":
            pending_buy_reason = f"drawdown_ladder_{next_threshold:.0%}"
            touched_threshold_index = None
        elif strategy.strategy_id == "ai_risk_filtered_ladder":
            allowed, reason = _ai_risk_allows_add(history)
            if allowed:
                pending_buy_reason = f"risk_filtered_{next_threshold:.0%}_{reason}"
                touched_threshold_index = None
            else:
                blocked_buy_days += 1

    equity_curve = pd.DataFrame(rows)
    stats = _portfolio_stats(equity_curve)
    return {
        "final_value": float(equity_curve["total_value"].iloc[-1]),
        "total_return": float(equity_curve["total_value"].iloc[-1] / initial_cash - 1),
        "max_drawdown": stats["max_drawdown"],
        "longest_underwater_days": stats["longest_underwater_days"],
        "most_painful_start": stats["most_painful_start"],
        "most_painful_end": stats["most_painful_end"],
        "buy_count": sum(1 for trade in trades if trade["action"] == "buy"),
        "blocked_buy_days": blocked_buy_days,
        "trades": trades,
        "equity_curve": equity_curve,
    }


def _buy_with_budget(
    *,
    date: pd.Timestamp,
    ticker: str,
    asset_label: str,
    strategy: StrategySpec,
    price: float,
    budget: float,
    cash: float,
    reason: str,
    broker_fee_rate: float,
    broker_fee_discount: float,
    minimum_fee_twd: int,
) -> dict | None:
    if price <= 0:
        return None
    shares = int(budget // price)
    while shares > 0:
        gross = shares * price
        cost = _buy_cost(gross, broker_fee_rate, broker_fee_discount, minimum_fee_twd)
        if gross + cost <= cash and gross + cost <= budget + minimum_fee_twd:
            return {
                "date": date.strftime("%Y-%m-%d"),
                "ticker": ticker,
                "asset_label": asset_label,
                "strategy_id": strategy.strategy_id,
                "strategy_label": strategy.label,
                "action": "buy",
                "shares": shares,
                "price": round(price, 4),
                "gross_amount": round(gross, 2),
                "cost": cost,
                "reason": reason,
            }
        shares -= 1
    return None


def _buy_cost(gross: float, broker_fee_rate: float, broker_fee_discount: float, minimum_fee_twd: int) -> int:
    if gross <= 0:
        return 0
    return max(minimum_fee_twd, int(round(gross * broker_fee_rate * broker_fee_discount)))


def _rolling_high_drawdown(history: pd.DataFrame, window: int) -> float:
    segment = history.tail(window)
    high = float(segment["close"].max())
    close = float(segment["close"].iloc[-1])
    return close / high - 1 if high > 0 else 0.0


def _ai_risk_allows_add(history: pd.DataFrame) -> tuple[bool, str]:
    close = float(history["close"].iloc[-1])
    ma120 = float(history["close"].tail(120).mean()) if len(history) >= 120 else close
    ma200 = float(history["close"].tail(200).mean()) if len(history) >= 200 else close
    ma200_prev = float(history["close"].iloc[-220:-20].mean()) if len(history) >= 220 else ma200
    ret20 = _return(history, 20)
    ret60 = _return(history, 60)
    drawdown120 = _rolling_high_drawdown(history, 120)

    high_risk = close < ma200 and ma200 < ma200_prev and ret60 < 0
    if high_risk:
        return False, "below_ma200_slope_down"
    if close > ma120 and ret20 > 0:
        return True, "recovered_ma120_ret20_positive"
    if close > ma200 and ret60 > -0.05:
        return True, "above_ma200"
    if drawdown120 > -0.10 and ret20 > 0:
        return True, "shallow_pullback_recovery"
    return False, "risk_not_recovered"


def _return(history: pd.DataFrame, days: int) -> float:
    if len(history) <= days:
        return 0.0
    start = float(history["close"].iloc[-days - 1])
    end = float(history["close"].iloc[-1])
    return end / start - 1 if start > 0 else 0.0


def _portfolio_stats(equity_curve: pd.DataFrame) -> dict:
    values = equity_curve["total_value"].astype(float)
    peaks = values.cummax()
    drawdowns = values / peaks - 1
    max_drawdown = float(drawdowns.min())
    trough_index = int(drawdowns.idxmin())
    peak_index = int(values.loc[:trough_index].idxmax()) if trough_index > 0 else 0

    longest = 0
    current = 0
    for value, peak in zip(values, peaks):
        if value < peak * 0.999999:
            current += 1
            longest = max(longest, current)
        else:
            current = 0

    return {
        "max_drawdown": max_drawdown,
        "longest_underwater_days": longest,
        "most_painful_start": equity_curve.loc[peak_index, "date"],
        "most_painful_end": equity_curve.loc[trough_index, "date"],
    }


def _periods(start_date: str, end_date: str) -> tuple[PeriodSpec, ...]:
    return (
        PeriodSpec("full", "完整樣本", start_date, end_date),
        PeriodSpec("2022", "2022 空頭段", "2022-01-03", "2022-12-30"),
        PeriodSpec("2023", "2023 反彈多頭段", "2023-01-03", "2023-12-29"),
    )


def _write_charts(output_dir: Path, summary: pd.DataFrame, equity: pd.DataFrame, trades: pd.DataFrame) -> list[str]:
    _configure_chinese_font()
    chart_dir = output_dir / "charts"
    chart_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    colors = {
        "無腦越跌越買": "#b42318",
        "固定分批": "#2457a7",
        "AI風險過濾後才加碼": "#13795b",
    }
    for period_id in ("full", "2022", "2023"):
        for ticker, asset_label in ASSETS.items():
            subset = equity[(equity["period_id"] == period_id) & (equity["ticker"] == ticker)]
            if subset.empty:
                continue
            fig, ax = plt.subplots(figsize=(10, 5.8))
            for strategy_label, rows in subset.groupby("strategy_label"):
                rows = rows.sort_values("date")
                ax.plot(
                    pd.to_datetime(rows["date"]),
                    rows["total_value"] / 10_000,
                    label=strategy_label,
                    linewidth=2.1,
                    color=colors.get(strategy_label, "#52616b"),
                )
            ax.set_title(f"{asset_label} 三種下跌加碼策略資產曲線 - {period_id}")
            ax.set_ylabel("帳面淨值（萬元）")
            ax.grid(True, alpha=0.25)
            ax.legend()
            ax.tick_params(axis="x", labelrotation=20)
            path = chart_dir / f"{period_id}_{ticker.replace('.TW', '')}_equity.png"
            fig.tight_layout()
            fig.savefig(path, dpi=170)
            plt.close(fig)
            paths.append(str(path))

            fig, ax = plt.subplots(figsize=(10, 4.8))
            for strategy_label, rows in subset.groupby("strategy_label"):
                rows = rows.sort_values("date")
                values = rows["total_value"].astype(float)
                drawdown = values / values.cummax() - 1
                ax.plot(
                    pd.to_datetime(rows["date"]),
                    drawdown * 100,
                    label=strategy_label,
                    linewidth=2.0,
                    color=colors.get(strategy_label, "#52616b"),
                )
            ax.set_title(f"{asset_label} 回撤曲線 - {period_id}")
            ax.set_ylabel("回撤（%）")
            ax.grid(True, alpha=0.25)
            ax.legend()
            ax.tick_params(axis="x", labelrotation=20)
            path = chart_dir / f"{period_id}_{ticker.replace('.TW', '')}_drawdown.png"
            fig.tight_layout()
            fig.savefig(path, dpi=170)
            plt.close(fig)
            paths.append(str(path))

    pivot = summary[summary["period_id"].isin(["full", "2022", "2023"])].copy()
    pivot["name"] = pivot["asset_label"] + "｜" + pivot["strategy_label"] + "｜" + pivot["period_label"]
    pivot = pivot.sort_values("total_return_pct", ascending=True)
    fig, ax = plt.subplots(figsize=(11, 9))
    ax.barh(pivot["name"], pivot["total_return_pct"], color=["#b42318" if x >= 0 else "#13795b" for x in pivot["total_return_pct"]])
    ax.set_title("EP09 三策略報酬率總覽（台股色：漲紅跌綠）")
    ax.set_xlabel("報酬率（%）")
    ax.grid(axis="x", alpha=0.25)
    path = chart_dir / "strategy_return_overview.png"
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)
    paths.append(str(path))

    _write_buy_marker_tables(output_dir, trades)
    return paths


def _write_buy_marker_tables(output_dir: Path, trades: pd.DataFrame) -> None:
    buys = trades[trades["action"] == "buy"].copy() if not trades.empty else pd.DataFrame()
    if buys.empty:
        return
    buys.to_csv(output_dir / "buy_markers.csv", index=False, encoding="utf-8-sig")


def _write_report(output_dir: Path, summary: pd.DataFrame, chart_paths: list[str], initial_cash: float) -> None:
    lines = [
        "# EP09 下跌加碼回測素材包",
        "",
        "定位：AI 輔助回測與風險觀察，不是投資建議。",
        f"初始資金：{initial_cash:,.0f} 元。",
        "",
        "## 策略定義",
    ]
    for strategy in STRATEGIES:
        lines.append(f"- {strategy.label}：{strategy.description}")
    lines.extend(["", "## 主要結果", ""])
    display_cols = [
        "period_label",
        "asset_label",
        "strategy_label",
        "final_value_twd",
        "total_return_pct",
        "max_drawdown_pct",
        "longest_underwater_trading_days",
        "buy_count",
        "blocked_buy_days",
        "most_painful_start",
        "most_painful_end",
    ]
    lines.append(_markdown_table(summary[display_cols]))
    lines.extend(
        [
            "",
            "## 圖表清單",
            "",
            *[f"- `{path}`" for path in chart_paths],
            "",
            "## 使用邊界",
            "",
            "- 本資料包測的是下跌時分批投入規則，不代表未來市場會重複歷史。",
            "- 0050正二為槓桿 ETF，回撤與波動可能遠高於 0050。",
            "- 圖卡與旁白應使用「AI 輔助觀察、回測、風險」語氣，不得寫成買賣建議。",
        ]
    )
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def _write_source_notes(output_dir: Path, summary: pd.DataFrame, chart_paths: list[str]) -> None:
    full = summary[summary["period_id"] == "full"].copy()
    best_by_asset = full.sort_values("final_value_twd", ascending=False).groupby("asset_label").head(1)
    notes = [
        "EP09 sourceNotes：市場下跌時，越跌越買到底是紀律，還是在接刀？",
        "",
        "本集素材來自 AI_stock_backtest_lab 的 0050/0050正二下跌加碼回測。內容定位是 AI 輔助回測與風險觀察，不是投資建議。",
        "",
        "三種策略：",
        "1. 無腦越跌越買：跌到階梯就買，不看趨勢是否轉弱。",
        "2. 固定分批：固定每 20 個交易日投入一批。",
        "3. AI風險過濾後才加碼：跌幅階梯先觸發，但長期趨勢轉弱時暫停，等風險改善才允許下一批。",
        "",
        "完整樣本最佳結果：",
    ]
    for _, row in best_by_asset.iterrows():
        notes.append(
            f"- {row.asset_label}：{row.strategy_label}，期末 {row.final_value_twd:,.0f} 元，"
            f"報酬率 {row.total_return_pct:+.2f}%，最大回撤 {row.max_drawdown_pct:+.2f}%，"
            f"最長套牢 {int(row.longest_underwater_trading_days)} 個交易日。"
        )
    notes.extend(
        [
            "",
            "影片敘事重點：不要先預設越跌越買一定是紀律，也不要先預設它一定是接刀。用 0050 與 0050正二 的回測結果，拆開看報酬、最大回撤、套牢時間、加碼次數與最痛苦區間。",
            "",
            "建議圖卡：",
            "- 第一張：三策略總表，直接問觀眾「跌越多買越多，真的比較安全嗎？」",
            "- 第二張：0050 資產曲線，呈現分批節奏差異。",
            "- 第三張：0050正二回撤曲線，強調槓桿商品在下跌段的心理壓力。",
            "- 第四張：2022 空頭段比較，聚焦最大回撤與最長套牢。",
            "",
            "圖表檔案：",
            *[f"- {path}" for path in chart_paths],
        ]
    )
    (output_dir / "creatorflow_source_notes.md").write_text("\n".join(notes), encoding="utf-8")


def _markdown_table(frame: pd.DataFrame) -> str:
    headers = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
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


def _write_manifest(output_dir: Path, args: argparse.Namespace, periods: tuple[PeriodSpec, ...], chart_paths: list[str]) -> None:
    manifest = {
        "task_id": "TASK-EP09-DIP-BUYING-001",
        "output_dir": str(output_dir),
        "initial_cash": args.initial_cash,
        "cache_dir": args.cache_dir,
        "periods": [period.__dict__ for period in periods],
        "assets": ASSETS,
        "strategies": [strategy.__dict__ for strategy in STRATEGIES],
        "charts": chart_paths,
        "content_boundary": "AI輔助回測/風險觀察，不是投資建議。",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


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
