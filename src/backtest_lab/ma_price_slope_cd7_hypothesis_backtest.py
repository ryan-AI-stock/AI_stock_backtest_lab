"""Evaluate 35 structured CD7 extensions around the current best MA rule."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
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


TASK_ID = "TASK-BACKTEST-EXPERIMENTS-P1-P2-0050-SIGNAL-00631L-CD7-STRUCTURED-HYPOTHESIS-DIAGNOSTIC-001"
COOLDOWN_DAYS = 7
BASELINE_STRATEGY = "E0_BASE_MA4_S7__X0_BASE_MA10_S20__CD7"


@dataclass(frozen=True)
class HypothesisRule:
    rule_id: str
    description: str


ENTRY_RULES = (
    HypothesisRule("E0_BASE_MA4_S7", "close>MA4 and 7TD price slope>0"),
    HypothesisRule("E1_MA4_CROSS_S7", "cross above MA4 today and 7TD price slope>0"),
    HypothesisRule("E2_MA4_HOLD2_S7", "close>MA4 for two days and 7TD price slope>0"),
    HypothesisRule("E3_MA4_S7_S3", "close>MA4 and both 7TD/3TD price slopes>0"),
    HypothesisRule("E4_MA4_S7_MA4UP3", "close>MA4, 7TD slope>0 and MA4 rising over 3TD"),
    HypothesisRule("E5_MA4_MA10_S7", "close>MA4 and MA10 with 7TD price slope>0"),
)

EXIT_RULES = (
    HypothesisRule("X0_BASE_MA10_S20", "close<MA10 and 20TD price slope<0"),
    HypothesisRule("X1_MA10_CROSS_S20", "cross below MA10 today and 20TD price slope<0"),
    HypothesisRule("X2_MA10_HOLD2_S20", "close<MA10 for two days and 20TD price slope<0"),
    HypothesisRule("X3_MA10_S20_S5", "close<MA10 and both 20TD/5TD price slopes<0"),
    HypothesisRule("X4_MA10_S20_MA10DN3", "close<MA10, 20TD slope<0 and MA10 falling over 3TD"),
    HypothesisRule("X5_MA10_MA20_S20", "close<MA10 and MA20 with 20TD price slope<0"),
)


@dataclass(frozen=True)
class SimulationResult:
    daily: pd.DataFrame
    trades: pd.DataFrame
    summary: dict[str, Any]


def strategy_id(entry_rule: HypothesisRule, exit_rule: HypothesisRule) -> str:
    return f"{entry_rule.rule_id}__{exit_rule.rule_id}__CD{COOLDOWN_DAYS}"


def hypothesis_matrix() -> pd.DataFrame:
    rows = []
    for entry_rule in ENTRY_RULES:
        for exit_rule in EXIT_RULES:
            strategy = strategy_id(entry_rule, exit_rule)
            rows.append(
                {
                    "strategy": strategy,
                    "entry_rule": entry_rule.rule_id,
                    "entry_description": entry_rule.description,
                    "exit_rule": exit_rule.rule_id,
                    "exit_description": exit_rule.description,
                    "cooldown_days": COOLDOWN_DAYS,
                    "is_existing_baseline": strategy == BASELINE_STRATEGY,
                }
            )
    return pd.DataFrame(rows)


def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy().sort_values("date").reset_index(drop=True)
    close = out["0050_adj_close"]
    for window in (4, 10, 20):
        out[f"ma{window}"] = close.rolling(window, min_periods=window).mean()
    for window in (3, 5, 7, 20):
        out[f"slope{window}"] = close - close.shift(window - 1)
    out["ma4_up3"] = out["ma4"] > out["ma4"].shift(2)
    out["ma10_down3"] = out["ma10"] < out["ma10"].shift(2)
    out["cross_above_ma4"] = (close > out["ma4"]) & (close.shift(1) <= out["ma4"].shift(1))
    out["cross_below_ma10"] = (close < out["ma10"]) & (close.shift(1) >= out["ma10"].shift(1))
    out["above_ma4_two_days"] = (close > out["ma4"]) & (close.shift(1) > out["ma4"].shift(1))
    out["below_ma10_two_days"] = (close < out["ma10"]) & (close.shift(1) < out["ma10"].shift(1))
    return out


def add_rule_signals(frame: pd.DataFrame, entry_rule: HypothesisRule, exit_rule: HypothesisRule) -> pd.DataFrame:
    out = frame.copy()
    close = out["0050_adj_close"]
    base_entry = (close > out["ma4"]) & (out["slope7"] > 0)
    base_exit = (close < out["ma10"]) & (out["slope20"] < 0)
    entry_map = {
        "E0_BASE_MA4_S7": base_entry,
        "E1_MA4_CROSS_S7": out["cross_above_ma4"] & (out["slope7"] > 0),
        "E2_MA4_HOLD2_S7": out["above_ma4_two_days"] & (out["slope7"] > 0),
        "E3_MA4_S7_S3": base_entry & (out["slope3"] > 0),
        "E4_MA4_S7_MA4UP3": base_entry & out["ma4_up3"],
        "E5_MA4_MA10_S7": base_entry & (close > out["ma10"]),
    }
    exit_map = {
        "X0_BASE_MA10_S20": base_exit,
        "X1_MA10_CROSS_S20": out["cross_below_ma10"] & (out["slope20"] < 0),
        "X2_MA10_HOLD2_S20": out["below_ma10_two_days"] & (out["slope20"] < 0),
        "X3_MA10_S20_S5": base_exit & (out["slope5"] < 0),
        "X4_MA10_S20_MA10DN3": base_exit & out["ma10_down3"],
        "X5_MA10_MA20_S20": base_exit & (close < out["ma20"]),
    }
    out["buy_signal"] = entry_map[entry_rule.rule_id].fillna(False)
    out["sell_signal"] = exit_map[exit_rule.rule_id].fillna(False)
    return out


def cooldown_complete(current_index: int, last_execution_index: int | None) -> bool:
    return last_execution_index is None or current_index - last_execution_index > COOLDOWN_DAYS


def simulate(
    featured: pd.DataFrame,
    *,
    period: dict[str, str],
    entry_rule: HypothesisRule,
    exit_rule: HypothesisRule,
    after_cost: bool,
    initial_capital: float = INITIAL_CAPITAL,
) -> SimulationResult:
    signaled = add_rule_signals(featured, entry_rule, exit_rule)
    frame = signaled[
        (signaled["date"] >= pd.Timestamp(period["requested_start"]))
        & (signaled["date"] <= pd.Timestamp(period["requested_end"]))
    ].copy().reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"no common coverage for {period['period']}")

    model = TaiwanCostModel()
    strategy = strategy_id(entry_rule, exit_rule)
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
        blocked_by_cooldown = index > 0 and not cooldown_complete(index, last_execution_index)
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
                    reason = entry_rule.description
                    last_execution_index = index
                    trade_rows.append(_trade_row(period, strategy, entry_rule, exit_rule, row["date"], signal_date, index, action, price, execution_price, shares, gross, fee, 0.0, reason, after_cost))
            elif shares > 0 and bool(previous["sell_signal"]):
                price = float(row["00631L_adj_close"])
                execution_price = price * (1.0 - SLIPPAGE_PER_SIDE if after_cost else 1.0)
                gross = shares * execution_price
                breakdown = model.sell_cost_breakdown(gross, "etf") if after_cost else {"sell_fee": 0.0, "securities_transaction_tax": 0.0, "total_transaction_cost": 0.0}
                cash += gross - breakdown["total_transaction_cost"]
                total_cost += breakdown["total_transaction_cost"] + (shares * price * SLIPPAGE_PER_SIDE if after_cost else 0.0)
                action = "sell"
                reason = exit_rule.description
                last_execution_index = index
                trade_rows.append(_trade_row(period, strategy, entry_rule, exit_rule, row["date"], signal_date, index, action, price, execution_price, shares, gross, breakdown["sell_fee"], breakdown["securities_transaction_tax"], reason, after_cost))
                shares = 0

        equity = cash + shares * float(row["00631L_adj_close"])
        daily_rows.append(
            {
                "period": period["period"], "strategy": strategy,
                "entry_rule": entry_rule.rule_id, "exit_rule": exit_rule.rule_id,
                "cost_basis": "after_cost_10bp_side" if after_cost else "gross_no_cost",
                "date": row["date"].date().isoformat(),
                "signal_date": signal_date.date().isoformat() if pd.notna(signal_date) else "",
                "0050_adj_close": float(row["0050_adj_close"]), "00631L_adj_close": float(row["00631L_adj_close"]),
                "buy_signal": bool(row["buy_signal"]), "sell_signal": bool(row["sell_signal"]),
                "cooldown_blocked": blocked_by_cooldown, "action": action, "reason": reason,
                "cash": cash, "shares": shares, "equity": equity, "stock_exposure": int(shares > 0),
            }
        )

    daily = pd.DataFrame(daily_rows)
    trades = pd.DataFrame(trade_rows)
    daily["drawdown_pct"] = (daily["equity"] / daily["equity"].cummax() - 1.0) * 100.0
    final_equity = float(daily.iloc[-1]["equity"])
    years = max((pd.Timestamp(daily.iloc[-1]["date"]) - pd.Timestamp(daily.iloc[0]["date"])).days / 365.2425, 1.0 / 365.2425)
    execution_gaps = trades["execution_index"].diff().dropna() if not trades.empty else pd.Series(dtype=float)
    summary = {
        "strategy": strategy, "period": period["period"],
        "requested_start": period["requested_start"], "requested_end": period["requested_end"],
        "actual_start": daily.iloc[0]["date"], "actual_end": daily.iloc[-1]["date"], "trading_days": int(len(daily)),
        "entry_rule": entry_rule.rule_id, "entry_description": entry_rule.description,
        "exit_rule": exit_rule.rule_id, "exit_description": exit_rule.description,
        "cooldown_days": COOLDOWN_DAYS, "cost_basis": "after_cost_10bp_side" if after_cost else "gross_no_cost",
        "initial_capital": initial_capital, "final_equity": final_equity,
        "total_return_pct": (final_equity / initial_capital - 1.0) * 100.0,
        "cagr_pct": ((final_equity / initial_capital) ** (1.0 / years) - 1.0) * 100.0,
        "max_drawdown_pct": float(daily["drawdown_pct"].min()),
        "buy_count": int(trades["side"].eq("buy").sum()) if not trades.empty else 0,
        "sell_count": int(trades["side"].eq("sell").sum()) if not trades.empty else 0,
        "transitions": int(len(trades)),
        "minimum_execution_gap_td": int(execution_gaps.min()) if not execution_gaps.empty else None,
        "stock_exposure_pct": float(daily["stock_exposure"].mean() * 100.0),
        "total_cost_and_slippage_twd": total_cost, "open_position_at_end": bool(shares > 0),
        "is_existing_baseline": strategy == BASELINE_STRATEGY,
    }
    return SimulationResult(daily=daily, trades=trades, summary=summary)


def run_backtest(*, price_0050: str | Path, price_00631l: str | Path, output_dir: str | Path) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    p0050_raw = load_price(price_0050)
    p00631_raw = load_price(price_00631l)
    common = p0050_raw.rename(columns={"adj_close": "0050_adj_close"})[["date", "0050_adj_close"]].merge(
        p00631_raw.rename(columns={"adj_close": "00631L_adj_close"})[["date", "00631L_adj_close"]], on="date", how="inner"
    )
    featured = add_features(common)
    matrix = hypothesis_matrix()
    if len(matrix) != 36 or int((~matrix["is_existing_baseline"]).sum()) != 35:
        raise AssertionError("expected one baseline plus exactly 35 new hypotheses")

    summaries: list[dict[str, Any]] = []
    daily_parts: list[pd.DataFrame] = []
    trade_parts: list[pd.DataFrame] = []
    for period in PERIODS:
        summaries.append(buy_hold_summary(p00631_raw, period, after_cost=True))
        for entry_rule in ENTRY_RULES:
            for exit_rule in EXIT_RULES:
                for after_cost in (False, True):
                    result = simulate(featured, period=period, entry_rule=entry_rule, exit_rule=exit_rule, after_cost=after_cost)
                    summaries.append(result.summary)
                    daily_parts.append(result.daily)
                    if not result.trades.empty:
                        trade_parts.append(result.trades)

    summary = pd.DataFrame(summaries)
    daily = pd.concat(daily_parts, ignore_index=True)
    trades = pd.concat(trade_parts, ignore_index=True) if trade_parts else pd.DataFrame()
    future_violations = int((pd.to_datetime(trades["signal_date"]) >= pd.to_datetime(trades["execution_date"])).sum()) if not trades.empty else 0
    ordered = trades.sort_values(["period", "strategy", "cost_basis", "execution_index"])
    gaps = ordered.groupby(["period", "strategy", "cost_basis"])["execution_index"].diff()
    cooldown_violations = int((gaps.dropna() <= COOLDOWN_DAYS).sum())

    primary = summary[summary["cost_basis"].eq("after_cost_10bp_side")].copy()
    baseline = primary[primary["is_existing_baseline"].eq(True)][["period", "total_return_pct", "max_drawdown_pct"]].rename(columns={"total_return_pct": "baseline_return_pct", "max_drawdown_pct": "baseline_mdd_pct"})
    benchmark = summary[summary["cost_basis"].eq("after_cost_10bp_buy_side")][
        ["period", "total_return_pct", "max_drawdown_pct"]
    ].rename(
        columns={
            "total_return_pct": "benchmark_00631l_return_pct",
            "max_drawdown_pct": "benchmark_00631l_mdd_pct",
        }
    )
    comparison = primary.merge(baseline, on="period", how="left").merge(benchmark, on="period", how="left")
    comparison["excess_vs_baseline_pp"] = comparison["total_return_pct"] - comparison["baseline_return_pct"]
    comparison["mdd_delta_vs_baseline_pp"] = comparison["max_drawdown_pct"] - comparison["baseline_mdd_pct"]
    comparison["excess_vs_00631l_pp"] = comparison["total_return_pct"] - comparison["benchmark_00631l_return_pct"]
    new_only = comparison[comparison["is_existing_baseline"].eq(False)]
    pass_both = new_only.pivot(index="strategy", columns="period", values="excess_vs_baseline_pp").dropna()
    pass_both_count = int(((pass_both.get("P1", -1) > 0) & (pass_both.get("P2", -1) > 0)).sum()) if not pass_both.empty else 0
    pass_benchmark = new_only.pivot(index="strategy", columns="period", values="excess_vs_00631l_pp").dropna()
    pass_benchmark_count = int(
        ((pass_benchmark.get("P1", -1) > 0) & (pass_benchmark.get("P2", -1) > 0)).sum()
    ) if not pass_benchmark.empty else 0

    summary.to_csv(output / "p1_p2_cd7_hypothesis_summary.csv", index=False, encoding="utf-8-sig", float_format="%.6f")
    comparison.to_csv(output / "p1_p2_cd7_comparison_vs_baseline.csv", index=False, encoding="utf-8-sig", float_format="%.6f")
    daily.to_csv(output / "p1_p2_cd7_hypothesis_daily_nav.csv", index=False, encoding="utf-8-sig", float_format="%.8f")
    trades.to_csv(output / "p1_p2_cd7_hypothesis_trades.csv", index=False, encoding="utf-8-sig", float_format="%.8f")
    matrix.to_csv(output / "structured_hypothesis_matrix.csv", index=False, encoding="utf-8-sig")
    _coverage(common).to_csv(output / "requested_vs_actual_coverage.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{"audit": "signal_date_strictly_before_execution_date", "violation_count": future_violations}, {"audit": "execution_gap_strictly_greater_than_CD7", "violation_count": cooldown_violations}]).to_csv(output / "execution_and_future_data_audit.csv", index=False, encoding="utf-8-sig")

    manifest = {
        "task_id": TASK_ID, "status": "completed_diagnostic", "tested_strategy_count": 36,
        "new_hypothesis_count": 35, "baseline_strategy": BASELINE_STRATEGY,
        "cooldown_days_after_each_execution": COOLDOWN_DAYS,
        "signal_asset": "0050.TW", "execution_asset": "00631L.TW",
        "signal_price_basis": "adjusted_close", "execution_and_holding_basis": "provider_adjusted_close_total_return_research_proxy",
        "execution_timing": "next_common_trading_day_close_if_not_in_cooldown",
        "cost_model_version": COST_MODEL_VERSION, "slippage_per_side": SLIPPAGE_PER_SIDE,
        "new_hypotheses_beating_baseline_both_periods": pass_both_count,
        "new_hypotheses_beating_00631l_both_periods": pass_benchmark_count,
        "future_data_violation_count": future_violations, "cooldown_violation_count": cooldown_violations,
        "formal_model_changed": False, "trade_decision_changed": False, "active_in_trade_decision": False,
        "report_changed": False, "ready_for_formal": False, "not_live_rule": True,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "final_summary_zh.md").write_text(
        _summary_text(comparison, pass_both_count, pass_benchmark_count), encoding="utf-8"
    )
    return manifest


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


def _summary_text(comparison: pd.DataFrame, pass_both_count: int, pass_benchmark_count: int) -> str:
    baseline = comparison[comparison["is_existing_baseline"].eq(True)].sort_values("period")
    new_only = comparison[comparison["is_existing_baseline"].eq(False)].copy()
    wide = new_only.pivot(index="strategy", columns="period", values="excess_vs_baseline_pp").dropna()
    wide["minimum_period_excess"] = wide[["P1", "P2"]].min(axis=1)
    nearest_strategy = wide["minimum_period_excess"].idxmax()
    nearest = new_only[new_only["strategy"].eq(nearest_strategy)].sort_values("period")
    lines = [
        "# CD7 structured hypothesis diagnostic",
        "",
        "這次測試包含既有基準與35個結構性延伸，訊號使用0050，實際持有00631L。",
        "主結果使用next-day close、ETF手續費/交易稅及10bp/side滑價。",
        "",
        f"- 同時打敗既有CD7基準的新規則：{pass_both_count}。",
        f"- 同時打敗00631L買進持有的新規則：{pass_benchmark_count}。",
        "",
        "## 既有基準",
    ]
    for row in baseline.to_dict(orient="records"):
        lines.append(
            f"- {row['period']}: net {row['total_return_pct']:.2f}%, MDD {row['max_drawdown_pct']:.2f}%, "
            f"transitions {int(row['transitions'])}."
        )
    lines.extend(["", "## 最接近的新規則", f"- {nearest_strategy}"])
    for row in nearest.to_dict(orient="records"):
        lines.append(
            f"- {row['period']}: net {row['total_return_pct']:.2f}%, "
            f"excess vs baseline {row['excess_vs_baseline_pp']:.2f}pp, MDD {row['max_drawdown_pct']:.2f}%."
        )
    return "\n".join(lines) + "\n"


def _trade_row(period: dict[str, str], strategy: str, entry_rule: HypothesisRule, exit_rule: HypothesisRule, execution_date: pd.Timestamp, signal_date: pd.Timestamp, execution_index: int, side: str, reference_price: float, execution_price: float, shares: int, gross: float, fee: float, tax: float, reason: str, after_cost: bool) -> dict[str, Any]:
    return {
        "period": period["period"], "strategy": strategy, "entry_rule": entry_rule.rule_id, "exit_rule": exit_rule.rule_id,
        "cost_basis": "after_cost_10bp_side" if after_cost else "gross_no_cost",
        "signal_date": signal_date.date().isoformat(), "execution_date": execution_date.date().isoformat(), "execution_index": execution_index,
        "side": side, "reference_adj_close": reference_price, "execution_price_after_slippage": execution_price,
        "shares": shares, "gross_amount": gross, "broker_fee": fee, "etf_sell_tax": tax, "signal_reason": reason,
    }


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
