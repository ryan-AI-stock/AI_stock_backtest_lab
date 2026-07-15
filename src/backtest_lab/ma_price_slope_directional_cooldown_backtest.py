"""Evaluate symmetric and directional cooldowns for the two frozen MA rules."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from backtest_lab.costs import COST_MODEL_VERSION, TaiwanCostModel
from backtest_lab.ma_price_slope_cd7_hypothesis_backtest import (
    ENTRY_RULES,
    EXIT_RULES,
    HypothesisRule,
    add_features,
    add_rule_signals,
)
from backtest_lab.ma_signal_leveraged_etf_backtest import (
    INITIAL_CAPITAL,
    PERIODS,
    buy_hold_summary,
    load_price,
)


TASK_ID = "TASK-BACKTEST-EXPERIMENTS-P1-P2-0050-SIGNAL-00631L-DIRECTIONAL-COOLDOWN-DIAGNOSTIC-001"
PRIMARY_SLIPPAGE = 0.001
SLIPPAGE_SCENARIOS = (0.0005, 0.001, 0.002)
ROLLING_WINDOW_TD = 504


@dataclass(frozen=True)
class SignalPair:
    pair_id: str
    entry_rule: HypothesisRule
    exit_rule: HypothesisRule


@dataclass(frozen=True)
class CooldownScenario:
    scenario_id: str
    post_buy_exit_lock_td: int
    post_sell_entry_lock_td: int


@dataclass(frozen=True)
class SimulationResult:
    daily: pd.DataFrame
    trades: pd.DataFrame
    summary: dict[str, Any]


SIGNAL_PAIRS = (
    SignalPair("S0_RETURN_BASE", ENTRY_RULES[0], EXIT_RULES[0]),
    SignalPair("S1_LOW_MDD_EXIT", ENTRY_RULES[0], EXIT_RULES[3]),
)

COOLDOWN_SCENARIOS = (
    CooldownScenario("L6_6", 6, 6),
    CooldownScenario("L7_7_REFERENCE", 7, 7),
    CooldownScenario("L8_8", 8, 8),
    CooldownScenario("L0_5_EXIT_UNLOCKED", 0, 5),
    CooldownScenario("L0_7_EXIT_UNLOCKED", 0, 7),
    CooldownScenario("L0_10_EXIT_UNLOCKED", 0, 10),
    CooldownScenario("L7_0_REENTRY_UNLOCKED", 7, 0),
    CooldownScenario("L7_5_REENTRY5", 7, 5),
    CooldownScenario("L7_10_REENTRY10", 7, 10),
    CooldownScenario("L0_0_BOTH_UNLOCKED", 0, 0),
)


def experiment_matrix() -> pd.DataFrame:
    rows = []
    for pair in SIGNAL_PAIRS:
        for scenario in COOLDOWN_SCENARIOS:
            rows.append(
                {
                    "strategy": strategy_id(pair, scenario),
                    "signal_pair": pair.pair_id,
                    "entry_rule": pair.entry_rule.rule_id,
                    "exit_rule": pair.exit_rule.rule_id,
                    "cooldown_scenario": scenario.scenario_id,
                    "post_buy_exit_lock_td": scenario.post_buy_exit_lock_td,
                    "post_sell_entry_lock_td": scenario.post_sell_entry_lock_td,
                    "is_current_reference": pair.pair_id == "S0_RETURN_BASE"
                    and scenario.scenario_id == "L7_7_REFERENCE",
                }
            )
    return pd.DataFrame(rows)


def strategy_id(pair: SignalPair, scenario: CooldownScenario) -> str:
    return f"{pair.pair_id}__{scenario.scenario_id}"


def opposite_action_allowed(
    current_index: int,
    *,
    holding_stock: bool,
    last_buy_index: int | None,
    last_sell_index: int | None,
    scenario: CooldownScenario,
) -> bool:
    if holding_stock:
        return last_buy_index is None or current_index - last_buy_index > scenario.post_buy_exit_lock_td
    return last_sell_index is None or current_index - last_sell_index > scenario.post_sell_entry_lock_td


def simulate(
    featured: pd.DataFrame,
    *,
    period: dict[str, str],
    pair: SignalPair,
    scenario: CooldownScenario,
    slippage_per_side: float,
    initial_capital: float = INITIAL_CAPITAL,
) -> SimulationResult:
    signaled = add_rule_signals(featured, pair.entry_rule, pair.exit_rule)
    frame = signaled[
        (signaled["date"] >= pd.Timestamp(period["requested_start"]))
        & (signaled["date"] <= pd.Timestamp(period["requested_end"]))
    ].copy().reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"no common coverage for {period['period']}")

    model = TaiwanCostModel()
    strategy = strategy_id(pair, scenario)
    cost_basis = f"after_cost_{int(slippage_per_side * 10000)}bp_side"
    cash = float(initial_capital)
    shares = 0
    last_buy_index: int | None = None
    last_sell_index: int | None = None
    total_cost = 0.0
    daily_rows: list[dict[str, Any]] = []
    trade_rows: list[dict[str, Any]] = []

    for index, row in frame.iterrows():
        action = "hold_stock" if shares else "hold_cash"
        reason = ""
        signal_date = pd.NaT
        exit_blocked = False
        entry_blocked = False

        if index > 0:
            previous = frame.iloc[index - 1]
            signal_date = previous["date"]
            allowed = opposite_action_allowed(
                index,
                holding_stock=shares > 0,
                last_buy_index=last_buy_index,
                last_sell_index=last_sell_index,
                scenario=scenario,
            )
            if shares == 0 and bool(previous["buy_signal"]):
                entry_blocked = not allowed
                if allowed:
                    price = float(row["00631L_adj_close"])
                    execution_price = price * (1.0 + slippage_per_side)
                    shares = int(cash // execution_price)
                    while shares > 0:
                        gross = shares * execution_price
                        fee = model.buy_cost(gross)
                        if gross + fee <= cash:
                            break
                        shares -= 1
                    if shares > 0:
                        gross = shares * execution_price
                        fee = model.buy_cost(gross)
                        cash_before = cash
                        cash -= gross + fee
                        slippage_cost = shares * price * slippage_per_side
                        total_cost += fee + slippage_cost
                        action = "buy"
                        reason = pair.entry_rule.description
                        last_buy_index = index
                        trade_rows.append(
                            _trade_row(
                                period, strategy, pair, scenario, cost_basis, row["date"], signal_date,
                                index, action, price, execution_price, shares, gross, fee, 0.0,
                                slippage_cost, cash_before, cash, reason,
                            )
                        )
            elif shares > 0 and bool(previous["sell_signal"]):
                exit_blocked = not allowed
                if allowed:
                    price = float(row["00631L_adj_close"])
                    execution_price = price * (1.0 - slippage_per_side)
                    gross = shares * execution_price
                    breakdown = model.sell_cost_breakdown(gross, "etf")
                    cash_before = cash
                    cash += gross - breakdown["total_transaction_cost"]
                    slippage_cost = shares * price * slippage_per_side
                    total_cost += breakdown["total_transaction_cost"] + slippage_cost
                    action = "sell"
                    reason = pair.exit_rule.description
                    last_sell_index = index
                    trade_rows.append(
                        _trade_row(
                            period, strategy, pair, scenario, cost_basis, row["date"], signal_date,
                            index, action, price, execution_price, shares, gross,
                            breakdown["sell_fee"], breakdown["securities_transaction_tax"],
                            slippage_cost, cash_before, cash, reason,
                        )
                    )
                    shares = 0

        equity = cash + shares * float(row["00631L_adj_close"])
        daily_rows.append(
            {
                "period": period["period"],
                "strategy": strategy,
                "signal_pair": pair.pair_id,
                "cooldown_scenario": scenario.scenario_id,
                "post_buy_exit_lock_td": scenario.post_buy_exit_lock_td,
                "post_sell_entry_lock_td": scenario.post_sell_entry_lock_td,
                "cost_basis": cost_basis,
                "date": row["date"].date().isoformat(),
                "signal_date": signal_date.date().isoformat() if pd.notna(signal_date) else "",
                "0050_adj_close": float(row["0050_adj_close"]),
                "00631L_adj_close": float(row["00631L_adj_close"]),
                "buy_signal": bool(row["buy_signal"]),
                "sell_signal": bool(row["sell_signal"]),
                "entry_blocked": entry_blocked,
                "exit_blocked": exit_blocked,
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
    years = max(
        (pd.Timestamp(daily.iloc[-1]["date"]) - pd.Timestamp(daily.iloc[0]["date"])).days / 365.2425,
        1.0 / 365.2425,
    )
    summary = {
        "strategy": strategy,
        "period": period["period"],
        "signal_pair": pair.pair_id,
        "entry_rule": pair.entry_rule.rule_id,
        "exit_rule": pair.exit_rule.rule_id,
        "cooldown_scenario": scenario.scenario_id,
        "post_buy_exit_lock_td": scenario.post_buy_exit_lock_td,
        "post_sell_entry_lock_td": scenario.post_sell_entry_lock_td,
        "requested_start": period["requested_start"],
        "requested_end": period["requested_end"],
        "actual_start": daily.iloc[0]["date"],
        "actual_end": daily.iloc[-1]["date"],
        "trading_days": int(len(daily)),
        "cost_basis": cost_basis,
        "initial_capital": initial_capital,
        "final_equity": final_equity,
        "total_return_pct": (final_equity / initial_capital - 1.0) * 100.0,
        "cagr_pct": ((final_equity / initial_capital) ** (1.0 / years) - 1.0) * 100.0,
        "max_drawdown_pct": float(daily["drawdown_pct"].min()),
        "buy_count": int(trades["side"].eq("buy").sum()) if not trades.empty else 0,
        "sell_count": int(trades["side"].eq("sell").sum()) if not trades.empty else 0,
        "transitions": int(len(trades)),
        "stock_exposure_pct": float(daily["stock_exposure"].mean() * 100.0),
        "entry_blocked_signal_days": int(daily["entry_blocked"].sum()),
        "exit_blocked_signal_days": int(daily["exit_blocked"].sum()),
        "total_cost_and_slippage_twd": total_cost,
        "open_position_at_end": bool(shares > 0),
        "is_current_reference": pair.pair_id == "S0_RETURN_BASE"
        and scenario.scenario_id == "L7_7_REFERENCE",
    }
    return SimulationResult(daily=daily, trades=trades, summary=summary)


def run_backtest(*, price_0050: str | Path, price_00631l: str | Path, output_dir: str | Path) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    p0050_raw = load_price(price_0050)
    p00631_raw = load_price(price_00631l)
    common = p0050_raw.rename(columns={"adj_close": "0050_adj_close"})[["date", "0050_adj_close"]].merge(
        p00631_raw.rename(columns={"adj_close": "00631L_adj_close"})[["date", "00631L_adj_close"]],
        on="date",
        how="inner",
    )
    featured = add_features(common)
    matrix = experiment_matrix()
    if len(matrix) != 20 or matrix["strategy"].nunique() != 20:
        raise AssertionError("expected exactly 20 directional cooldown paths")

    summaries: list[dict[str, Any]] = []
    primary_daily_parts: list[pd.DataFrame] = []
    primary_trade_parts: list[pd.DataFrame] = []
    for period in PERIODS:
        for pair in SIGNAL_PAIRS:
            for scenario in COOLDOWN_SCENARIOS:
                for slippage in SLIPPAGE_SCENARIOS:
                    result = simulate(
                        featured,
                        period=period,
                        pair=pair,
                        scenario=scenario,
                        slippage_per_side=slippage,
                    )
                    summaries.append(result.summary)
                    if slippage == PRIMARY_SLIPPAGE:
                        primary_daily_parts.append(result.daily)
                        if not result.trades.empty:
                            primary_trade_parts.append(result.trades)

    summary = pd.DataFrame(summaries)
    primary_daily = pd.concat(primary_daily_parts, ignore_index=True)
    primary_trades = pd.concat(primary_trade_parts, ignore_index=True)
    primary = summary[summary["cost_basis"].eq("after_cost_10bp_side")].copy()
    benchmark = pd.DataFrame([buy_hold_summary(p00631_raw, period, after_cost=True) for period in PERIODS])
    reference = primary[primary["is_current_reference"].eq(True)][
        ["period", "total_return_pct", "max_drawdown_pct"]
    ].rename(
        columns={"total_return_pct": "reference_return_pct", "max_drawdown_pct": "reference_mdd_pct"}
    )
    benchmark_merge = benchmark[["period", "total_return_pct", "max_drawdown_pct"]].rename(
        columns={
            "total_return_pct": "benchmark_00631l_return_pct",
            "max_drawdown_pct": "benchmark_00631l_mdd_pct",
        }
    )
    comparison = primary.merge(reference, on="period", how="left").merge(
        benchmark_merge, on="period", how="left"
    )
    comparison["excess_vs_reference_pp"] = comparison["total_return_pct"] - comparison["reference_return_pct"]
    comparison["mdd_delta_vs_reference_pp"] = comparison["max_drawdown_pct"] - comparison["reference_mdd_pct"]
    comparison["excess_vs_00631l_pp"] = comparison["total_return_pct"] - comparison["benchmark_00631l_return_pct"]

    annual = _annual_metrics(primary_daily)
    rolling = _rolling_metrics(primary_daily)
    concentration = _episode_concentration(primary_trades)
    divergence = _path_divergence(primary_daily)
    future_violations = int(
        (pd.to_datetime(primary_trades["signal_date"]) >= pd.to_datetime(primary_trades["execution_date"])).sum()
    )
    directional_violations = _directional_cooldown_violations(primary_trades)

    summary.to_csv(output / "p1_p2_directional_cd_summary.csv", index=False, encoding="utf-8-sig", float_format="%.6f")
    comparison.to_csv(output / "p1_p2_directional_cd_comparison.csv", index=False, encoding="utf-8-sig", float_format="%.6f")
    benchmark.to_csv(output / "p1_p2_00631l_buy_hold_reference.csv", index=False, encoding="utf-8-sig", float_format="%.6f")
    primary_daily.to_csv(output / "p1_p2_directional_cd_daily_nav.csv", index=False, encoding="utf-8-sig", float_format="%.8f")
    primary_trades.to_csv(output / "p1_p2_directional_cd_trades.csv", index=False, encoding="utf-8-sig", float_format="%.8f")
    matrix.to_csv(output / "directional_cooldown_matrix.csv", index=False, encoding="utf-8-sig")
    annual.to_csv(output / "directional_cd_annual_metrics.csv", index=False, encoding="utf-8-sig", float_format="%.6f")
    rolling.to_csv(output / "directional_cd_rolling_504td_metrics.csv", index=False, encoding="utf-8-sig", float_format="%.6f")
    concentration.to_csv(output / "directional_cd_episode_concentration_proxy.csv", index=False, encoding="utf-8-sig", float_format="%.6f")
    divergence.to_csv(output / "directional_cd_path_divergence_vs_same_signal_reference.csv", index=False, encoding="utf-8-sig", float_format="%.6f")
    _coverage(common).to_csv(output / "requested_vs_actual_coverage.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(
        [
            {"audit": "signal_date_strictly_before_execution_date", "violation_count": future_violations},
            {"audit": "directional_cooldown_execution_gap", "violation_count": directional_violations},
        ]
    ).to_csv(output / "execution_and_future_data_audit.csv", index=False, encoding="utf-8-sig")

    wide = comparison.pivot(index="strategy", columns="period", values="excess_vs_reference_pp").dropna()
    pass_both_count = int(((wide["P1"] > 0) & (wide["P2"] > 0)).sum())
    benchmark_wide = comparison.pivot(index="strategy", columns="period", values="excess_vs_00631l_pp").dropna()
    pass_benchmark_count = int(((benchmark_wide["P1"] > 0) & (benchmark_wide["P2"] > 0)).sum())
    manifest = {
        "task_id": TASK_ID,
        "status": "completed_diagnostic",
        "signal_pair_count": len(SIGNAL_PAIRS),
        "cooldown_scenario_count": len(COOLDOWN_SCENARIOS),
        "primary_path_count": len(matrix),
        "slippage_scenarios_bp_side": [5, 10, 20],
        "post_buy_exit_lock_and_post_sell_entry_lock_separated": True,
        "zero_lock_means_next_trading_day_opposite_execution_allowed": True,
        "strategies_beating_current_reference_both_periods": pass_both_count,
        "strategies_beating_00631l_both_periods": pass_benchmark_count,
        "signal_asset": "0050.TW",
        "execution_asset": "00631L.TW",
        "execution_timing": "prior-day signal to next common trading-day close",
        "execution_and_holding_basis": "provider_adjusted_close_total_return_research_proxy",
        "rolling_00631l_comparison_basis": "gross_adjusted_close_504td_slice_proxy",
        "episode_concentration_is_exact_rechain": False,
        "cost_model_version": COST_MODEL_VERSION,
        "future_data_violation_count": future_violations,
        "directional_cooldown_violation_count": directional_violations,
        "formal_model_changed": False,
        "trade_decision_changed": False,
        "active_in_trade_decision": False,
        "report_changed": False,
        "ready_for_formal": False,
        "not_live_rule": True,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "final_summary_zh.md").write_text(
        _summary_text(comparison, pass_both_count, pass_benchmark_count), encoding="utf-8"
    )
    return manifest


def _annual_metrics(daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in daily.groupby(["period", "strategy", "signal_pair", "cooldown_scenario"]):
        frame = group.sort_values("date").copy()
        frame["year"] = pd.to_datetime(frame["date"]).dt.year
        frame["daily_return"] = frame["equity"].pct_change().fillna(0.0)
        for year, year_frame in frame.groupby("year"):
            year_drawdown = year_frame["equity"] / year_frame["equity"].cummax() - 1.0
            rows.append(
                {
                    "period": keys[0],
                    "strategy": keys[1],
                    "signal_pair": keys[2],
                    "cooldown_scenario": keys[3],
                    "year": int(year),
                    "annual_return_pct": ((1.0 + year_frame["daily_return"]).prod() - 1.0) * 100.0,
                    "annual_mdd_pct": float(year_drawdown.min() * 100.0),
                    "trading_days": int(len(year_frame)),
                }
            )
    return pd.DataFrame(rows)


def _summary_text(comparison: pd.DataFrame, pass_reference_count: int, pass_benchmark_count: int) -> str:
    cooldown_alternatives = comparison[
        comparison["signal_pair"].eq("S0_RETURN_BASE") & comparison["is_current_reference"].eq(False)
    ]
    wide = cooldown_alternatives.pivot(
        index="strategy", columns="period", values="excess_vs_reference_pp"
    ).dropna()
    wide["minimum_period_excess"] = wide[["P1", "P2"]].min(axis=1)
    best_strategy = wide["minimum_period_excess"].idxmax()
    best = comparison[comparison["strategy"].eq(best_strategy)].sort_values("period")
    lines = [
        "# Directional cooldown diagnostic",
        "",
        "固定兩套訊號，交互測試買後賣出鎖定與賣後買進鎖定。",
        "CD0表示沒有額外鎖定，仍使用前一日訊號、下一交易日成交。",
        "",
        f"- 同時打敗目前CD7報酬基準：{pass_reference_count}條。",
        f"- 同時打敗00631L買進持有：{pass_benchmark_count}條。",
        f"- 最佳非基準CD替代方案：{best_strategy}。",
    ]
    for row in best.to_dict(orient="records"):
        lines.append(
            f"- {row['period']}: net {row['total_return_pct']:.2f}%, "
            f"vs reference {row['excess_vs_reference_pp']:.2f}pp, "
            f"MDD {row['max_drawdown_pct']:.2f}%."
        )
    return "\n".join(lines) + "\n"


def _rolling_metrics(daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in daily.groupby(["period", "strategy", "signal_pair", "cooldown_scenario"]):
        frame = group.sort_values("date").copy()
        strategy_return = frame["equity"] / frame["equity"].shift(ROLLING_WINDOW_TD - 1) - 1.0
        benchmark_return = (
            frame["00631L_adj_close"] / frame["00631L_adj_close"].shift(ROLLING_WINDOW_TD - 1) - 1.0
        )
        valid = pd.DataFrame({"strategy": strategy_return, "benchmark": benchmark_return}).dropna()
        excess = (valid["strategy"] - valid["benchmark"]) * 100.0
        rows.append(
            {
                "period": keys[0],
                "strategy": keys[1],
                "signal_pair": keys[2],
                "cooldown_scenario": keys[3],
                "rolling_window_td": ROLLING_WINDOW_TD,
                "window_count": int(len(valid)),
                "median_strategy_return_pct": float(valid["strategy"].median() * 100.0) if not valid.empty else None,
                "minimum_strategy_return_pct": float(valid["strategy"].min() * 100.0) if not valid.empty else None,
                "median_excess_vs_00631l_pp": float(excess.median()) if not valid.empty else None,
                "positive_excess_window_share": float((excess > 0).mean()) if not valid.empty else None,
            }
        )
    return pd.DataFrame(rows)


def _episode_concentration(trades: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in trades.groupby(["period", "strategy", "signal_pair", "cooldown_scenario"]):
        ordered = group.sort_values("execution_index").reset_index(drop=True)
        episode_returns = []
        pending_buy_outflow: float | None = None
        for trade in ordered.to_dict(orient="records"):
            if trade["side"] == "buy":
                pending_buy_outflow = float(trade["gross_amount"] + trade["broker_fee"])
            elif trade["side"] == "sell" and pending_buy_outflow:
                sell_net = float(trade["gross_amount"] - trade["broker_fee"] - trade["etf_sell_tax"])
                episode_returns.append((sell_net / pending_buy_outflow - 1.0) * 100.0)
                pending_buy_outflow = None
        positive = [value for value in episode_returns if value > 0]
        rows.append(
            {
                "period": keys[0],
                "strategy": keys[1],
                "signal_pair": keys[2],
                "cooldown_scenario": keys[3],
                "closed_episode_count": len(episode_returns),
                "best_episode_return_pct": max(episode_returns) if episode_returns else None,
                "worst_episode_return_pct": min(episode_returns) if episode_returns else None,
                "best_episode_share_of_positive_return_proxy": (
                    max(positive) / sum(positive) if positive and sum(positive) else None
                ),
                "proxy_not_exact_leave_one_episode_rechain": True,
            }
        )
    return pd.DataFrame(rows)


def _path_divergence(daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (period, signal_pair), pair_frame in daily.groupby(["period", "signal_pair"]):
        reference = pair_frame[pair_frame["cooldown_scenario"].eq("L7_7_REFERENCE")][
            ["date", "action", "stock_exposure", "equity"]
        ].rename(
            columns={
                "action": "reference_action",
                "stock_exposure": "reference_exposure",
                "equity": "reference_equity",
            }
        )
        for scenario, scenario_frame in pair_frame.groupby("cooldown_scenario"):
            alternative = scenario_frame[["date", "action", "stock_exposure", "equity"]].rename(
                columns={
                    "action": "alternative_action",
                    "stock_exposure": "alternative_exposure",
                    "equity": "alternative_equity",
                }
            )
            merged = reference.merge(alternative, on="date", how="inner")
            differences = merged[
                (merged["reference_action"] != merged["alternative_action"])
                | (merged["reference_exposure"] != merged["alternative_exposure"])
            ]
            rows.append(
                {
                    "period": period,
                    "signal_pair": signal_pair,
                    "cooldown_scenario": scenario,
                    "different_exposure_days": int(
                        (merged["reference_exposure"] != merged["alternative_exposure"]).sum()
                    ),
                    "different_action_days": int(
                        (merged["reference_action"] != merged["alternative_action"]).sum()
                    ),
                    "first_difference_date": differences.iloc[0]["date"] if not differences.empty else "",
                    "final_nav_delta_twd": float(
                        merged.iloc[-1]["alternative_equity"] - merged.iloc[-1]["reference_equity"]
                    ),
                }
            )
    return pd.DataFrame(rows)


def _directional_cooldown_violations(trades: pd.DataFrame) -> int:
    violations = 0
    for _, group in trades.sort_values("execution_index").groupby(["period", "strategy", "cost_basis"]):
        previous: dict[str, Any] | None = None
        for current in group.to_dict(orient="records"):
            if previous is not None:
                gap = int(current["execution_index"] - previous["execution_index"])
                required = (
                    int(current["post_buy_exit_lock_td"])
                    if previous["side"] == "buy"
                    else int(current["post_sell_entry_lock_td"])
                )
                if gap <= required:
                    violations += 1
            previous = current
    return violations


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


def _trade_row(
    period: dict[str, str],
    strategy: str,
    pair: SignalPair,
    scenario: CooldownScenario,
    cost_basis: str,
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
    slippage_cost: float,
    cash_before: float,
    cash_after: float,
    reason: str,
) -> dict[str, Any]:
    return {
        "period": period["period"],
        "strategy": strategy,
        "signal_pair": pair.pair_id,
        "cooldown_scenario": scenario.scenario_id,
        "post_buy_exit_lock_td": scenario.post_buy_exit_lock_td,
        "post_sell_entry_lock_td": scenario.post_sell_entry_lock_td,
        "cost_basis": cost_basis,
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
        "slippage_cost": slippage_cost,
        "cash_before": cash_before,
        "cash_after": cash_after,
        "signal_reason": reason,
    }


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
