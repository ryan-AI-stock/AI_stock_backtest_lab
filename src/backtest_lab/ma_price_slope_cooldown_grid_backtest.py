"""Run the bounded 0050 signal / 00631L execution grid with a CD5 gate."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

import pandas as pd

from backtest_lab.costs import COST_MODEL_VERSION, TaiwanCostModel
from backtest_lab.ma_signal_leveraged_etf_backtest import (
    INITIAL_CAPITAL,
    PERIODS,
    SLIPPAGE_PER_SIDE,
    buy_hold_summary,
    load_price,
)


TASK_ID = "TASK-BACKTEST-EXPERIMENTS-P1-P2-0050-SIGNAL-00631L-MA-PRICE-SLOPE-CD5-GRID-DIAGNOSTIC-001"
COOLDOWN_DAYS = 5


@dataclass(frozen=True)
class SignalRule:
    rule_id: str
    ma_window: int
    slope_window: int


BUY_RULES = tuple(
    SignalRule(f"B_MA{ma_window}_S{slope_window}", ma_window, slope_window)
    for ma_window in (4, 7)
    for slope_window in (7, 10, 20)
)
SELL_RULES = tuple(
    SignalRule(f"X_MA{ma_window}_S{slope_window}", ma_window, slope_window)
    for ma_window in (4, 10)
    for slope_window in (7, 10, 20)
)


@dataclass(frozen=True)
class SimulationResult:
    daily: pd.DataFrame
    trades: pd.DataFrame
    summary: dict[str, Any]


def combination_matrix() -> pd.DataFrame:
    rows = []
    for buy_rule, sell_rule in product(BUY_RULES, SELL_RULES):
        rows.append(
            {
                "strategy": f"{buy_rule.rule_id}__{sell_rule.rule_id}__CD{COOLDOWN_DAYS}",
                "buy_rule": buy_rule.rule_id,
                "buy_ma_window": buy_rule.ma_window,
                "buy_slope_window": buy_rule.slope_window,
                "sell_rule": sell_rule.rule_id,
                "sell_ma_window": sell_rule.ma_window,
                "sell_slope_window": sell_rule.slope_window,
                "cooldown_days": COOLDOWN_DAYS,
            }
        )
    return pd.DataFrame(rows)


def add_signals(frame: pd.DataFrame, buy_rule: SignalRule, sell_rule: SignalRule) -> pd.DataFrame:
    out = frame.copy().sort_values("date")
    analysis = out["0050_adj_close"]
    out["buy_ma"] = analysis.rolling(buy_rule.ma_window, min_periods=buy_rule.ma_window).mean()
    out["buy_slope"] = analysis - analysis.shift(buy_rule.slope_window - 1)
    out["buy_signal"] = (analysis > out["buy_ma"]) & (out["buy_slope"] > 0)
    out["sell_ma"] = analysis.rolling(sell_rule.ma_window, min_periods=sell_rule.ma_window).mean()
    out["sell_slope"] = analysis - analysis.shift(sell_rule.slope_window - 1)
    out["sell_signal"] = (analysis < out["sell_ma"]) & (out["sell_slope"] < 0)
    return out


def cooldown_complete(current_index: int, last_execution_index: int | None, cooldown_days: int = COOLDOWN_DAYS) -> bool:
    if last_execution_index is None:
        return True
    return current_index - last_execution_index > cooldown_days


def simulate(
    common: pd.DataFrame,
    *,
    period: dict[str, str],
    buy_rule: SignalRule,
    sell_rule: SignalRule,
    after_cost: bool,
    cooldown_days: int = COOLDOWN_DAYS,
    initial_capital: float = INITIAL_CAPITAL,
) -> SimulationResult:
    signaled = add_signals(common, buy_rule, sell_rule)
    start = pd.Timestamp(period["requested_start"])
    end = pd.Timestamp(period["requested_end"])
    frame = signaled[(signaled["date"] >= start) & (signaled["date"] <= end)].copy().reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"no common coverage for {period['period']}")

    model = TaiwanCostModel()
    strategy = f"{buy_rule.rule_id}__{sell_rule.rule_id}__CD{cooldown_days}"
    cash = float(initial_capital)
    shares = 0
    last_execution_index: int | None = None
    total_cost = 0.0
    daily_rows: list[dict[str, Any]] = []
    trade_rows: list[dict[str, Any]] = []

    for index, row in frame.iterrows():
        action = "hold_stock" if shares else "hold_cash"
        reason = ""
        signal_date = pd.NaT
        blocked_by_cooldown = index > 0 and not cooldown_complete(index, last_execution_index, cooldown_days)

        if index > 0:
            previous = frame.iloc[index - 1]
            signal_date = previous["date"]
            if blocked_by_cooldown:
                reason = "cooldown_blocked"
            elif shares == 0 and bool(previous["buy_signal"]):
                price = float(row["00631L_adj_close"])
                execution_price = price * (1.0 + SLIPPAGE_PER_SIDE if after_cost else 1.0)
                shares = int(cash // execution_price)
                while shares > 0:
                    gross = shares * execution_price
                    fee = model.buy_cost(gross) if after_cost else 0.0
                    if gross + fee <= cash:
                        break
                    shares -= 1
                if shares > 0:
                    gross = shares * execution_price
                    fee = model.buy_cost(gross) if after_cost else 0.0
                    cash -= gross + fee
                    total_cost += fee + (shares * price * SLIPPAGE_PER_SIDE if after_cost else 0.0)
                    action = "buy"
                    reason = f"0050_close_above_MA{buy_rule.ma_window}_and_{buy_rule.slope_window}TD_price_slope_up"
                    last_execution_index = index
                    trade_rows.append(
                        _trade_row(
                            period,
                            strategy,
                            buy_rule,
                            sell_rule,
                            row["date"],
                            signal_date,
                            index,
                            action,
                            price,
                            execution_price,
                            shares,
                            gross,
                            fee,
                            0.0,
                            reason,
                            after_cost,
                        )
                    )
            elif shares > 0 and bool(previous["sell_signal"]):
                price = float(row["00631L_adj_close"])
                execution_price = price * (1.0 - SLIPPAGE_PER_SIDE if after_cost else 1.0)
                gross = shares * execution_price
                breakdown = (
                    model.sell_cost_breakdown(gross, "etf")
                    if after_cost
                    else {"sell_fee": 0.0, "securities_transaction_tax": 0.0, "total_transaction_cost": 0.0}
                )
                cash += gross - breakdown["total_transaction_cost"]
                total_cost += breakdown["total_transaction_cost"] + (
                    shares * price * SLIPPAGE_PER_SIDE if after_cost else 0.0
                )
                action = "sell"
                reason = f"0050_close_below_MA{sell_rule.ma_window}_and_{sell_rule.slope_window}TD_price_slope_down"
                last_execution_index = index
                trade_rows.append(
                    _trade_row(
                        period,
                        strategy,
                        buy_rule,
                        sell_rule,
                        row["date"],
                        signal_date,
                        index,
                        action,
                        price,
                        execution_price,
                        shares,
                        gross,
                        breakdown["sell_fee"],
                        breakdown["securities_transaction_tax"],
                        reason,
                        after_cost,
                    )
                )
                shares = 0

        equity = cash + shares * float(row["00631L_adj_close"])
        cooldown_remaining = (
            max(0, cooldown_days - (index - last_execution_index)) if last_execution_index is not None else 0
        )
        daily_rows.append(
            {
                "period": period["period"],
                "strategy": strategy,
                "buy_rule": buy_rule.rule_id,
                "sell_rule": sell_rule.rule_id,
                "cost_basis": "after_cost_10bp_side" if after_cost else "gross_no_cost",
                "date": row["date"].date().isoformat(),
                "signal_date": signal_date.date().isoformat() if pd.notna(signal_date) else "",
                "0050_adj_close": float(row["0050_adj_close"]),
                "00631L_adj_close": float(row["00631L_adj_close"]),
                "buy_ma": float(row["buy_ma"]) if pd.notna(row["buy_ma"]) else None,
                "buy_slope": float(row["buy_slope"]) if pd.notna(row["buy_slope"]) else None,
                "sell_ma": float(row["sell_ma"]) if pd.notna(row["sell_ma"]) else None,
                "sell_slope": float(row["sell_slope"]) if pd.notna(row["sell_slope"]) else None,
                "buy_signal": bool(row["buy_signal"]),
                "sell_signal": bool(row["sell_signal"]),
                "cooldown_blocked": blocked_by_cooldown,
                "cooldown_remaining": cooldown_remaining,
                "action": action,
                "reason": reason,
                "cash": cash,
                "shares": shares,
                "equity": equity,
                "stock_exposure": int(shares > 0),
            }
        )

    daily = pd.DataFrame(daily_rows)
    trades = pd.DataFrame(trade_rows)
    daily["drawdown_pct"] = (daily["equity"] / daily["equity"].cummax() - 1.0) * 100.0
    final_equity = float(daily.iloc[-1]["equity"])
    years = _years(daily.iloc[0]["date"], daily.iloc[-1]["date"])
    execution_gaps = trades["execution_index"].diff().dropna() if not trades.empty else pd.Series(dtype=float)
    summary = {
        "strategy": strategy,
        "period": period["period"],
        "requested_start": period["requested_start"],
        "requested_end": period["requested_end"],
        "actual_start": daily.iloc[0]["date"],
        "actual_end": daily.iloc[-1]["date"],
        "trading_days": int(len(daily)),
        "buy_rule": buy_rule.rule_id,
        "buy_ma_window": buy_rule.ma_window,
        "buy_slope_window": buy_rule.slope_window,
        "sell_rule": sell_rule.rule_id,
        "sell_ma_window": sell_rule.ma_window,
        "sell_slope_window": sell_rule.slope_window,
        "cooldown_days": cooldown_days,
        "cost_basis": "after_cost_10bp_side" if after_cost else "gross_no_cost",
        "initial_capital": initial_capital,
        "final_equity": final_equity,
        "total_return_pct": (final_equity / initial_capital - 1.0) * 100.0,
        "cagr_pct": ((final_equity / initial_capital) ** (1.0 / years) - 1.0) * 100.0,
        "max_drawdown_pct": float(daily["drawdown_pct"].min()),
        "buy_count": int(trades["side"].eq("buy").sum()) if not trades.empty else 0,
        "sell_count": int(trades["side"].eq("sell").sum()) if not trades.empty else 0,
        "transitions": int(len(trades)),
        "minimum_execution_gap_td": int(execution_gaps.min()) if not execution_gaps.empty else None,
        "cooldown_blocked_days": int(daily["cooldown_blocked"].sum()),
        "stock_exposure_pct": float(daily["stock_exposure"].mean() * 100.0),
        "cash_exposure_pct": float((1.0 - daily["stock_exposure"].mean()) * 100.0),
        "total_cost_and_slippage_twd": total_cost,
        "open_position_at_end": bool(shares > 0),
    }
    return SimulationResult(daily=daily, trades=trades, summary=summary)


def run_backtest(*, price_0050: str | Path, price_00631l: str | Path, output_dir: str | Path) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    p0050_raw = load_price(price_0050)
    p00631_raw = load_price(price_00631l)
    p0050 = p0050_raw.rename(columns={"adj_close": "0050_adj_close"})
    p00631 = p00631_raw.rename(columns={"adj_close": "00631L_adj_close"})
    common = p0050[["date", "0050_adj_close"]].merge(
        p00631[["date", "00631L_adj_close"]], on="date", how="inner"
    )

    combinations = combination_matrix()
    if len(combinations) != 36 or combinations["strategy"].nunique() != 36:
        raise AssertionError("expected exactly 36 unique buy/sell combinations")

    summaries: list[dict[str, Any]] = []
    daily_parts: list[pd.DataFrame] = []
    trade_parts: list[pd.DataFrame] = []
    for period in PERIODS:
        summaries.append(buy_hold_summary(p00631_raw, period, after_cost=False))
        summaries.append(buy_hold_summary(p00631_raw, period, after_cost=True))
        for buy_rule, sell_rule in product(BUY_RULES, SELL_RULES):
            for after_cost in (False, True):
                result = simulate(
                    common,
                    period=period,
                    buy_rule=buy_rule,
                    sell_rule=sell_rule,
                    after_cost=after_cost,
                )
                summaries.append(result.summary)
                daily_parts.append(result.daily)
                if not result.trades.empty:
                    trade_parts.append(result.trades)

    summary = pd.DataFrame(summaries)
    daily = pd.concat(daily_parts, ignore_index=True)
    trades = pd.concat(trade_parts, ignore_index=True) if trade_parts else pd.DataFrame()
    future_violations = (
        int((pd.to_datetime(trades["signal_date"]) >= pd.to_datetime(trades["execution_date"])).sum())
        if not trades.empty
        else 0
    )
    cooldown_violations = _cooldown_violation_count(trades)

    summary.to_csv(output / "p1_p2_cd5_grid_summary.csv", index=False, encoding="utf-8-sig", float_format="%.6f")
    daily.to_csv(output / "p1_p2_cd5_daily_nav_ledger.csv", index=False, encoding="utf-8-sig", float_format="%.8f")
    trades.to_csv(output / "p1_p2_cd5_trade_ledger.csv", index=False, encoding="utf-8-sig", float_format="%.8f")
    combinations.to_csv(output / "buy_sell_combination_matrix.csv", index=False, encoding="utf-8-sig")
    _coverage(common).to_csv(output / "requested_vs_actual_coverage.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(
        [
            {"audit": "signal_date_strictly_before_execution_date", "violation_count": future_violations},
            {"audit": "execution_gap_strictly_greater_than_CD5", "violation_count": cooldown_violations},
        ]
    ).to_csv(output / "execution_and_future_data_audit.csv", index=False, encoding="utf-8-sig")

    manifest = {
        "task_id": TASK_ID,
        "status": "completed_diagnostic",
        "combination_count": int(len(combinations)),
        "buy_rule_count": len(BUY_RULES),
        "sell_rule_count": len(SELL_RULES),
        "cooldown_days_after_each_execution": COOLDOWN_DAYS,
        "earliest_next_execution": "sixth_common_trading_day_after_execution",
        "signal_asset": "0050.TW",
        "execution_asset": "00631L.TW",
        "signal_price_basis": "adjusted_close",
        "execution_and_holding_basis": "provider_adjusted_close_total_return_research_proxy",
        "execution_timing": "next_common_trading_day_close_if_not_in_cooldown",
        "price_slope_definition": "current adjusted close minus adjusted close N-1 trading days ago",
        "market_environment_risk_used": False,
        "layer0_4_candidate_pool_used": False,
        "cost_model_version": COST_MODEL_VERSION,
        "slippage_per_side": SLIPPAGE_PER_SIDE,
        "future_data_violation_count": future_violations,
        "cooldown_violation_count": cooldown_violations,
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


def _trade_row(
    period: dict[str, str],
    strategy: str,
    buy_rule: SignalRule,
    sell_rule: SignalRule,
    execution_date: pd.Timestamp,
    signal_date: pd.Timestamp,
    execution_index: int,
    side: str,
    reference_price: float,
    execution_price: float,
    shares: int,
    gross: float,
    fee: float,
    tax: float,
    reason: str,
    after_cost: bool,
) -> dict[str, Any]:
    return {
        "period": period["period"],
        "strategy": strategy,
        "buy_rule": buy_rule.rule_id,
        "sell_rule": sell_rule.rule_id,
        "cost_basis": "after_cost_10bp_side" if after_cost else "gross_no_cost",
        "signal_date": signal_date.date().isoformat(),
        "execution_date": execution_date.date().isoformat(),
        "execution_index": execution_index,
        "side": side,
        "reference_adj_close": reference_price,
        "execution_price_after_slippage": execution_price,
        "shares": shares,
        "gross_amount": gross,
        "broker_fee": fee,
        "etf_sell_tax": tax,
        "signal_reason": reason,
    }


def _cooldown_violation_count(trades: pd.DataFrame) -> int:
    if trades.empty:
        return 0
    ordered = trades.sort_values(["period", "strategy", "cost_basis", "execution_index"])
    gaps = ordered.groupby(["period", "strategy", "cost_basis"])["execution_index"].diff()
    return int((gaps.dropna() <= COOLDOWN_DAYS).sum())


def _years(start: str | pd.Timestamp, end: str | pd.Timestamp) -> float:
    days = (pd.Timestamp(end) - pd.Timestamp(start)).days
    return max(days / 365.2425, 1.0 / 365.2425)


def _coverage(common: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for period in PERIODS:
        frame = common[
            (common["date"] >= pd.Timestamp(period["requested_start"]))
            & (common["date"] <= pd.Timestamp(period["requested_end"]))
        ]
        rows.append(
            {
                "period": period["period"],
                "requested_start": period["requested_start"],
                "requested_end": period["requested_end"],
                "actual_start": frame["date"].min().date().isoformat(),
                "actual_end": frame["date"].max().date().isoformat(),
                "common_trading_days": int(len(frame)),
            }
        )
    return pd.DataFrame(rows)


def _summary_text(summary: pd.DataFrame) -> str:
    buy_hold = summary[summary["cost_basis"].eq("after_cost_10bp_buy_side")]
    primary = summary[summary["cost_basis"].eq("after_cost_10bp_side")]
    lines = [
        "# 0050 MA/price-slope CD5 full grid diagnostic",
        "",
        "Primary results use next-day close, ETF fee/tax, 10bp/side slippage and CD5.",
        "",
    ]
    for row in buy_hold.to_dict(orient="records"):
        lines.append(
            f"- {row['period']} 00631L buy-and-hold: net {row['total_return_pct']:.2f}%, "
            f"CAGR {row['cagr_pct']:.2f}%, MDD {row['max_drawdown_pct']:.2f}%"
        )
    lines.append("")
    for period in ("P1", "P2"):
        ranked = primary[primary["period"].eq(period)].sort_values("total_return_pct", ascending=False)
        for row in ranked.head(10).to_dict(orient="records"):
            lines.append(
                f"- {period} {row['strategy']}: net {row['total_return_pct']:.2f}%, "
                f"CAGR {row['cagr_pct']:.2f}%, MDD {row['max_drawdown_pct']:.2f}%, "
                f"transitions {row['transitions']}"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--price-0050", required=True)
    parser.add_argument("--price-00631l", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    manifest = run_backtest(
        price_0050=args.price_0050,
        price_00631l=args.price_00631l,
        output_dir=args.output_dir,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
