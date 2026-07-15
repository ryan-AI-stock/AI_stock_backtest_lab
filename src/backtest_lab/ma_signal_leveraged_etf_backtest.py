"""Backtest 0050 moving-average signals executed through 00631L.

This is a bounded diagnostic.  It intentionally excludes market-regime,
institutional-flow, chip, and Layer0-4 candidate-selection inputs.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from backtest_lab.costs import COST_MODEL_VERSION, TaiwanCostModel


TASK_ID = "TASK-BACKTEST-EXPERIMENTS-P1-P2-0050-SIGNAL-00631L-BELOW-MA-SLOPE-EXIT-DIAGNOSTIC-002"
INITIAL_CAPITAL = 1_000_000.0
SLIPPAGE_PER_SIDE = 0.001
PERIODS = (
    {"period": "P1", "requested_start": "2015-01-02", "requested_end": "2022-12-29"},
    {"period": "P2", "requested_start": "2023-01-02", "requested_end": "2026-06-30"},
)
EXIT_WINDOWS = (3, 5, 7, 10)


@dataclass(frozen=True)
class SimulationResult:
    daily: pd.DataFrame
    trades: pd.DataFrame
    summary: dict[str, Any]


def load_price(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"date", "close", "adj_close"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing price columns: {sorted(missing)}")
    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    for column in ("close", "adj_close"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.dropna(subset=["date", "close", "adj_close"]).sort_values("date").drop_duplicates("date")


def add_signals(frame: pd.DataFrame, exit_window: int) -> pd.DataFrame:
    out = frame.copy().sort_values("date")
    analysis = out["0050_adj_close"]
    out["ma5"] = analysis.rolling(5, min_periods=5).mean()
    out["buy_signal"] = (analysis > out["ma5"]) & (analysis > analysis.shift(4))
    exit_ma = analysis.rolling(exit_window, min_periods=exit_window).mean()
    out["exit_ma"] = exit_ma
    out["exit_ma_slope"] = exit_ma - exit_ma.shift(1)
    out["sell_signal"] = (analysis < exit_ma) & (out["exit_ma_slope"] < 0)
    return out


def simulate(
    common: pd.DataFrame,
    *,
    period: dict[str, str],
    exit_window: int,
    after_cost: bool,
    initial_capital: float = INITIAL_CAPITAL,
) -> SimulationResult:
    signaled = add_signals(common, exit_window)
    start = pd.Timestamp(period["requested_start"])
    end = pd.Timestamp(period["requested_end"])
    frame = signaled[(signaled["date"] >= start) & (signaled["date"] <= end)].copy().reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"no common coverage for {period['period']}")

    model = TaiwanCostModel()
    cash = float(initial_capital)
    shares = 0
    daily_rows: list[dict[str, Any]] = []
    trade_rows: list[dict[str, Any]] = []
    total_cost = 0.0

    for index, row in frame.iterrows():
        action = "hold_stock" if shares else "hold_cash"
        signal_date = pd.NaT
        signal_reason = ""
        if index > 0:
            previous = frame.iloc[index - 1]
            signal_date = previous["date"]
            price = float(row["00631L_adj_close"])
            if shares == 0 and bool(previous["buy_signal"]):
                execution_price = price * (1.0 + SLIPPAGE_PER_SIDE if after_cost else 1.0)
                shares = int(cash // execution_price)
                while shares > 0:
                    gross = shares * execution_price
                    fee = model.buy_cost(gross) if after_cost else 0
                    if gross + fee <= cash:
                        break
                    shares -= 1
                if shares > 0:
                    gross = shares * execution_price
                    fee = model.buy_cost(gross) if after_cost else 0
                    cash -= gross + fee
                    total_cost += fee + (shares * price * SLIPPAGE_PER_SIDE if after_cost else 0.0)
                    action = "buy"
                    signal_reason = "0050_close_above_MA5_and_above_close_4TD_ago"
                    trade_rows.append(_trade_row(period, exit_window, row["date"], signal_date, action, price, execution_price, shares, gross, fee, 0, signal_reason))
            elif shares > 0 and bool(previous["sell_signal"]):
                execution_price = price * (1.0 - SLIPPAGE_PER_SIDE if after_cost else 1.0)
                gross = shares * execution_price
                breakdown = model.sell_cost_breakdown(gross, "etf") if after_cost else {"sell_fee": 0, "securities_transaction_tax": 0, "total_transaction_cost": 0}
                cash += gross - breakdown["total_transaction_cost"]
                total_cost += breakdown["total_transaction_cost"] + (shares * price * SLIPPAGE_PER_SIDE if after_cost else 0.0)
                action = "sell"
                signal_reason = f"0050_close_below_MA{exit_window}_while_MA{exit_window}_slope_down"
                trade_rows.append(_trade_row(period, exit_window, row["date"], signal_date, action, price, execution_price, shares, gross, breakdown["sell_fee"], breakdown["securities_transaction_tax"], signal_reason))
                shares = 0

        equity = cash + shares * float(row["00631L_adj_close"])
        daily_rows.append({
            "period": period["period"],
            "exit_window": exit_window,
            "cost_basis": "after_cost_10bp_side" if after_cost else "gross_no_cost",
            "date": row["date"].date().isoformat(),
            "signal_date": signal_date.date().isoformat() if pd.notna(signal_date) else "",
            "0050_adj_close": float(row["0050_adj_close"]),
            "00631L_adj_close": float(row["00631L_adj_close"]),
            "ma5": float(row["ma5"]) if pd.notna(row["ma5"]) else None,
            "exit_ma": float(row["exit_ma"]) if pd.notna(row["exit_ma"]) else None,
            "exit_ma_slope": float(row["exit_ma_slope"]) if pd.notna(row["exit_ma_slope"]) else None,
            "buy_signal": bool(row["buy_signal"]),
            "sell_signal": bool(row["sell_signal"]),
            "action": action,
            "signal_reason": signal_reason,
            "cash": cash,
            "shares": shares,
            "equity": equity,
            "stock_exposure": int(shares > 0),
        })

    daily = pd.DataFrame(daily_rows)
    trades = pd.DataFrame(trade_rows)
    daily["drawdown_pct"] = (daily["equity"] / daily["equity"].cummax() - 1.0) * 100.0
    years = _years(daily.iloc[0]["date"], daily.iloc[-1]["date"])
    final_equity = float(daily.iloc[-1]["equity"])
    summary = {
        "strategy": f"exit_MA{exit_window}_slope_down",
        "period": period["period"],
        "requested_start": period["requested_start"],
        "requested_end": period["requested_end"],
        "actual_start": daily.iloc[0]["date"],
        "actual_end": daily.iloc[-1]["date"],
        "trading_days": int(len(daily)),
        "cost_basis": "after_cost_10bp_side" if after_cost else "gross_no_cost",
        "initial_capital": initial_capital,
        "final_equity": final_equity,
        "total_return_pct": (final_equity / initial_capital - 1.0) * 100.0,
        "cagr_pct": ((final_equity / initial_capital) ** (1.0 / years) - 1.0) * 100.0,
        "max_drawdown_pct": float(daily["drawdown_pct"].min()),
        "buy_count": int(trades["side"].eq("buy").sum()) if not trades.empty else 0,
        "sell_count": int(trades["side"].eq("sell").sum()) if not trades.empty else 0,
        "transitions": int(len(trades)),
        "stock_exposure_pct": float(daily["stock_exposure"].mean() * 100.0),
        "cash_exposure_pct": float((1.0 - daily["stock_exposure"].mean()) * 100.0),
        "total_cost_and_slippage_twd": total_cost,
        "open_position_at_end": bool(shares > 0),
    }
    return SimulationResult(daily=daily, trades=trades, summary=summary)


def buy_hold_summary(price: pd.DataFrame, period: dict[str, str], *, after_cost: bool) -> dict[str, Any]:
    start = pd.Timestamp(period["requested_start"])
    end = pd.Timestamp(period["requested_end"])
    frame = price[(price["date"] >= start) & (price["date"] <= end)].copy().reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"no 00631L coverage for {period['period']}")
    initial = INITIAL_CAPITAL
    first = float(frame.iloc[0]["adj_close"])
    if after_cost:
        execution = first * (1.0 + SLIPPAGE_PER_SIDE)
        model = TaiwanCostModel()
        shares = int(initial // execution)
        while shares > 0 and shares * execution + model.buy_cost(shares * execution) > initial:
            shares -= 1
        gross = shares * execution
        fee = model.buy_cost(gross)
        cash = initial - gross - fee
        equity = cash + shares * frame["adj_close"].astype(float)
        total_cost = fee + shares * first * SLIPPAGE_PER_SIDE
    else:
        shares = initial / first
        equity = shares * frame["adj_close"].astype(float)
        total_cost = 0.0
    years = _years(frame.iloc[0]["date"], frame.iloc[-1]["date"])
    final_equity = float(equity.iloc[-1])
    drawdown = (equity / equity.cummax() - 1.0) * 100.0
    return {
        "strategy": "00631L_buy_and_hold",
        "period": period["period"],
        "requested_start": period["requested_start"],
        "requested_end": period["requested_end"],
        "actual_start": frame.iloc[0]["date"].date().isoformat(),
        "actual_end": frame.iloc[-1]["date"].date().isoformat(),
        "trading_days": int(len(frame)),
        "cost_basis": "after_cost_10bp_buy_side" if after_cost else "gross_no_cost",
        "initial_capital": initial,
        "final_equity": final_equity,
        "total_return_pct": (final_equity / initial - 1.0) * 100.0,
        "cagr_pct": ((final_equity / initial) ** (1.0 / years) - 1.0) * 100.0,
        "max_drawdown_pct": float(drawdown.min()),
        "buy_count": 1,
        "sell_count": 0,
        "transitions": 1,
        "stock_exposure_pct": 100.0,
        "cash_exposure_pct": 0.0,
        "total_cost_and_slippage_twd": total_cost,
        "open_position_at_end": True,
    }


def run_backtest(
    *,
    price_0050: str | Path,
    price_00631l: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    p0050 = load_price(price_0050).rename(columns={"adj_close": "0050_adj_close"})
    p00631 = load_price(price_00631l).rename(columns={"adj_close": "00631L_adj_close"})
    common = p0050[["date", "0050_adj_close"]].merge(p00631[["date", "00631L_adj_close"]], on="date", how="inner")

    summaries: list[dict[str, Any]] = []
    daily_parts: list[pd.DataFrame] = []
    trade_parts: list[pd.DataFrame] = []
    for period in PERIODS:
        summaries.append(buy_hold_summary(load_price(price_00631l), period, after_cost=False))
        summaries.append(buy_hold_summary(load_price(price_00631l), period, after_cost=True))
        for window in EXIT_WINDOWS:
            for after_cost in (False, True):
                result = simulate(common, period=period, exit_window=window, after_cost=after_cost)
                summaries.append(result.summary)
                daily_parts.append(result.daily)
                if not result.trades.empty:
                    trade_parts.append(result.trades)

    summary = pd.DataFrame(summaries)
    daily = pd.concat(daily_parts, ignore_index=True)
    trades = pd.concat(trade_parts, ignore_index=True) if trade_parts else pd.DataFrame()
    annual = _annual_returns(daily)
    interpretation = pd.DataFrame([
        {"exit_window": 3, "institutional_reading_zh": "最敏感的短線跌破：價格跌破3日均線且均線下彎，代表短線承接轉弱；反應快，但容易被正常震盪洗出。"},
        {"exit_window": 5, "institutional_reading_zh": "一週趨勢跌破：價格跌破5日平均成本且均線下彎，代表短波段資金開始撤退；仍可能出現假跌破。"},
        {"exit_window": 7, "institutional_reading_zh": "約一週半趨勢確認：要求價格與均線方向同步轉弱，較能過濾短暫震盪，但退出會比MA3/5慢。"},
        {"exit_window": 10, "institutional_reading_zh": "兩週趨勢確認：較接近中短期資金撤退後的結構破壞，假訊號較少，但急跌時會承受較多回吐。"},
    ])

    summary.to_csv(output / "p1_p2_strategy_summary.csv", index=False, encoding="utf-8-sig", float_format="%.6f")
    daily.to_csv(output / "p1_p2_daily_nav_ledger.csv", index=False, encoding="utf-8-sig", float_format="%.8f")
    trades.to_csv(output / "p1_p2_trade_ledger.csv", index=False, encoding="utf-8-sig", float_format="%.8f")
    annual.to_csv(output / "p1_p2_annual_returns.csv", index=False, encoding="utf-8-sig", float_format="%.6f")
    interpretation.to_csv(output / "exit_rule_institutional_interpretation_zh.csv", index=False, encoding="utf-8-sig")
    coverage = _coverage(common)
    coverage.to_csv(output / "requested_vs_actual_coverage.csv", index=False, encoding="utf-8-sig")
    future_audit = pd.DataFrame([{"audit": "signal_date_strictly_before_execution_date", "violation_count": int((pd.to_datetime(trades.get("signal_date")) >= pd.to_datetime(trades.get("execution_date"))).sum()) if not trades.empty else 0}])
    future_audit.to_csv(output / "future_data_audit.csv", index=False, encoding="utf-8-sig")

    manifest = {
        "task_id": TASK_ID,
        "status": "completed_diagnostic",
        "initial_capital_twd": INITIAL_CAPITAL,
        "signal_asset": "0050.TW",
        "execution_asset": "00631L.TW",
        "signal_price_basis": "adjusted_close",
        "execution_and_holding_basis": "provider_adjusted_close_total_return_research_proxy",
        "execution_timing": "next_common_trading_day_close",
        "entry_rule": "0050 adjusted close > MA5 and > adjusted close 4 trading days ago",
        "exit_rules": [f"0050 adjusted close < MA{window} and MA{window} slope < 0" for window in EXIT_WINDOWS],
        "supersedes_incorrect_exit_semantics_output": "vnext_p1_p2_0050_signal_00631l_ma_slope_exit_diagnostic_20260715",
        "market_environment_risk_used": False,
        "layer0_4_candidate_pool_used": False,
        "cost_model_version": COST_MODEL_VERSION,
        "slippage_per_side": SLIPPAGE_PER_SIDE,
        "price_0050": str(Path(price_0050).resolve()),
        "price_00631l": str(Path(price_00631l).resolve()),
        "future_data_violation_count": int(future_audit.iloc[0]["violation_count"]),
        "formal_model_changed": False,
        "trade_decision_changed": False,
        "active_in_trade_decision": False,
        "report_changed": False,
        "ready_for_formal": False,
        "not_live_rule": True,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "final_summary_zh.md").write_text(_summary_text(summary), encoding="utf-8")
    return manifest


def _trade_row(period: dict[str, str], window: int, execution_date: pd.Timestamp, signal_date: pd.Timestamp, side: str, reference_price: float, execution_price: float, shares: int, gross: float, fee: float, tax: float, reason: str) -> dict[str, Any]:
    return {
        "period": period["period"], "exit_window": window, "cost_basis": "after_cost_10bp_side" if execution_price != reference_price else "gross_no_cost",
        "signal_date": signal_date.date().isoformat(), "execution_date": execution_date.date().isoformat(), "side": side,
        "reference_adj_close": reference_price, "execution_price_after_slippage": execution_price, "shares": shares,
        "gross_amount": gross, "broker_fee": fee, "etf_sell_tax": tax, "signal_reason": reason,
    }


def _years(start: str | pd.Timestamp, end: str | pd.Timestamp) -> float:
    days = (pd.Timestamp(end) - pd.Timestamp(start)).days
    return max(days / 365.2425, 1.0 / 365.2425)


def _annual_returns(daily: pd.DataFrame) -> pd.DataFrame:
    frame = daily.copy()
    frame["year"] = pd.to_datetime(frame["date"]).dt.year
    rows = []
    for keys, group in frame.groupby(["period", "exit_window", "cost_basis", "year"], sort=True):
        start_equity = float(group.iloc[0]["equity"])
        end_equity = float(group.iloc[-1]["equity"])
        rows.append({"period": keys[0], "exit_window": keys[1], "cost_basis": keys[2], "year": keys[3], "start_equity": start_equity, "end_equity": end_equity, "return_pct": (end_equity / start_equity - 1.0) * 100.0})
    return pd.DataFrame(rows)


def _coverage(common: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for period in PERIODS:
        frame = common[(common["date"] >= pd.Timestamp(period["requested_start"])) & (common["date"] <= pd.Timestamp(period["requested_end"]))]
        rows.append({"period": period["period"], "requested_start": period["requested_start"], "requested_end": period["requested_end"], "actual_start": frame["date"].min().date().isoformat(), "actual_end": frame["date"].max().date().isoformat(), "common_trading_days": len(frame)})
    return pd.DataFrame(rows)


def _summary_text(summary: pd.DataFrame) -> str:
    buy_hold = summary[summary["cost_basis"].eq("after_cost_10bp_buy_side")]
    primary = summary[summary["cost_basis"].eq("after_cost_10bp_side")]
    lines = ["# 0050訊號／00631L執行 P1-P2 診斷", "", "主結果使用next-day close、ETF費稅與10bp/side滑價。", ""]
    for row in buy_hold.to_dict(orient="records"):
        lines.append(f"- {row['period']} 00631L buy-and-hold: net {row['total_return_pct']:.2f}%, CAGR {row['cagr_pct']:.2f}%, MDD {row['max_drawdown_pct']:.2f}%")
    lines.append("")
    for row in primary.to_dict(orient="records"):
        lines.append(f"- {row['period']} {row['strategy']}: net {row['total_return_pct']:.2f}%, CAGR {row['cagr_pct']:.2f}%, MDD {row['max_drawdown_pct']:.2f}%, transitions {row['transitions']}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--price-0050", required=True)
    parser.add_argument("--price-00631l", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    manifest = run_backtest(price_0050=args.price_0050, price_00631l=args.price_00631l, output_dir=args.output_dir)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
