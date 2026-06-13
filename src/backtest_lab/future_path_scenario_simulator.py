from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from matplotlib import pyplot as plt

from backtest_lab.frozen_strategy_engine import load_frozen_strategy_context_from_cache, simulate_frozen_baseline
from backtest_lab.simulation import _max_drawdown
from backtest_lab.strategies import relative_strength_scores


DEFAULT_OUTPUT_DIR = "outputs/ad_hoc/future_path_scenario_simulator"
DEFAULT_GROUP_ID = "group_c_0050_00631l_plus_mega_caps"
DEFAULT_SIGNAL_DATE = "2026-06-12"
DEFAULT_SCENARIO_START = "2026-06-15"
DEFAULT_SCENARIO_END = "2026-12-31"
DEFAULT_HISTORY_START = "2024-01-02"
DEFAULT_REPLAY_START = "2020-01-02"
DEFAULT_CACHE_DIR = "backtest_cache/ad_hoc_20260612_daily_targets_filled"
DEFAULT_INITIAL_CAPITAL = 1_328_709.0


@dataclass(frozen=True)
class ScenarioInputs:
    signal_date: str
    scenario_start: str
    scenario_end: str
    history_start: str
    replay_start: str
    initial_capital: float
    horizon_days: int
    target_ticker: str
    target_label: str
    current_regime: str
    current_mode: str
    current_attack_gate_active: bool
    current_top_stock: str
    current_top_stock_label: str
    current_top_stock_score: float
    current_target_score: float


def run_future_path_scenario(
    *,
    config_path: str | Path = "configs/ep05_universe.json",
    group_id: str = DEFAULT_GROUP_ID,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    signal_date: str = DEFAULT_SIGNAL_DATE,
    scenario_start: str = DEFAULT_SCENARIO_START,
    scenario_end: str = DEFAULT_SCENARIO_END,
    history_start: str = DEFAULT_HISTORY_START,
    replay_start: str = DEFAULT_REPLAY_START,
    initial_capital: float = DEFAULT_INITIAL_CAPITAL,
    min_analogs: int = 8,
) -> dict[str, Any]:
    context = load_frozen_strategy_context_from_cache(
        config_path=config_path,
        group_id=group_id,
        cache_dir=cache_dir,
    )
    labels = context.labels
    prices = context.prices_by_ticker
    result = simulate_frozen_baseline(
        context=context,
        name="future_path_base_history",
        start_date=replay_start,
        end_date=signal_date,
        initial_cash=1_000_000,
    )
    equity = result.equity_curve.sort_index()
    signal_ts = pd.Timestamp(signal_date)
    if signal_ts not in equity.index:
        raise ValueError(f"Signal date is not in simulated equity curve: {signal_date}")
    horizon_days = _business_horizon_days(scenario_start, scenario_end)
    inputs = _build_scenario_inputs(
        equity=equity,
        prices=prices,
        labels=labels,
        signal_ts=signal_ts,
        scenario_start=scenario_start,
        scenario_end=scenario_end,
        history_start=history_start,
        replay_start=replay_start,
        initial_capital=initial_capital,
        horizon_days=horizon_days,
    )
    analogs = _find_analog_cases(
        equity=equity,
        prices=prices,
        labels=labels,
        current=inputs,
        history_start=history_start,
        horizon_days=horizon_days,
        min_analogs=min_analogs,
    )
    paths = _build_normalized_paths(
        equity=equity,
        analogs=analogs,
        initial_capital=initial_capital,
        horizon_days=horizon_days,
    )
    hold_paths = _build_hold_counterfactual_paths(
        prices=prices,
        analogs=analogs,
        ticker=inputs.current_top_stock,
        initial_capital=initial_capital,
        horizon_days=horizon_days,
    )
    summary = _summarize_paths(paths, inputs)
    hold_summary = _summarize_hold_paths(hold_paths, inputs)
    trades = _trade_examples(result.trades, analogs, labels, horizon_days)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    analogs.to_csv(out / "analog_cases.csv", index=False, encoding="utf-8-sig")
    paths.to_csv(out / "scenario_paths.csv", index=False, encoding="utf-8-sig")
    hold_paths.to_csv(out / "hold_top_stock_paths.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out / "scenario_summary.csv", index=False, encoding="utf-8-sig")
    hold_summary.to_csv(out / "hold_top_stock_summary.csv", index=False, encoding="utf-8-sig")
    trades.to_csv(out / "trade_path_examples.csv", index=False, encoding="utf-8-sig")
    payload = {
        "status": "ready",
        "scenario_inputs": asdict(inputs),
        "analog_count": int(len(analogs)),
        "strict_analog_count": int((analogs["match_tier"] == "strict").sum()) if not analogs.empty else 0,
        "hold_counterfactual_ticker": inputs.current_top_stock,
        "hold_counterfactual_label": inputs.current_top_stock_label,
        "output_dir": str(out),
    }
    (out / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "README.md").write_text(_markdown_report(inputs, analogs, summary, hold_summary), encoding="utf-8")
    if not paths.empty:
        _write_path_chart(out / "scenario_paths.png", paths)
    return payload


def _business_horizon_days(start: str, end: str) -> int:
    dates = pd.bdate_range(pd.Timestamp(start), pd.Timestamp(end))
    return max(len(dates), 1)


def _build_scenario_inputs(
    *,
    equity: pd.DataFrame,
    prices: dict[str, pd.DataFrame],
    labels: dict[str, str],
    signal_ts: pd.Timestamp,
    scenario_start: str,
    scenario_end: str,
    history_start: str,
    replay_start: str,
    initial_capital: float,
    horizon_days: int,
) -> ScenarioInputs:
    row = equity.loc[signal_ts]
    scores = relative_strength_scores(prices, signal_ts)
    top_stock = _top_attack_stock(scores)
    return ScenarioInputs(
        signal_date=signal_ts.strftime("%Y-%m-%d"),
        scenario_start=scenario_start,
        scenario_end=scenario_end,
        history_start=history_start,
        replay_start=replay_start,
        initial_capital=float(initial_capital),
        horizon_days=int(horizon_days),
        target_ticker=str(row["current_ticker"]),
        target_label=_label(str(row["current_ticker"]), labels),
        current_regime=str(row["regime"]),
        current_mode=str(row["mode"]),
        current_attack_gate_active=bool(row["attack_gate_active"]),
        current_top_stock=top_stock,
        current_top_stock_label=_label(top_stock, labels),
        current_top_stock_score=float(scores.get(top_stock, 0.0)),
        current_target_score=float(scores.get(str(row["current_ticker"]), 0.0)),
    )


def _find_analog_cases(
    *,
    equity: pd.DataFrame,
    prices: dict[str, pd.DataFrame],
    labels: dict[str, str],
    current: ScenarioInputs,
    history_start: str,
    horizon_days: int,
    min_analogs: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    dates = list(equity.index)
    last_start_index = max(0, len(dates) - horizon_days - 1)
    current_ts = pd.Timestamp(current.signal_date)
    for index, date in enumerate(dates[: last_start_index + 1]):
        if date < pd.Timestamp(history_start) or date >= current_ts:
            continue
        row = equity.loc[date]
        scores = relative_strength_scores(prices, date)
        top_stock = _top_attack_stock(scores)
        analog = {
            "analog_start": date.strftime("%Y-%m-%d"),
            "target_ticker": str(row["current_ticker"]),
            "target_label": _label(str(row["current_ticker"]), labels),
            "target_score": round(float(scores.get(str(row["current_ticker"]), 0.0)), 6),
            "top_stock": top_stock,
            "top_stock_label": _label(top_stock, labels),
            "top_stock_score": round(float(scores.get(top_stock, 0.0)), 6),
            "regime": str(row["regime"]),
            "mode": str(row["mode"]),
            "attack_gate_active": bool(row["attack_gate_active"]),
            "attack_gate_ever_activated": bool(row["attack_gate_ever_activated"]),
            "similarity_score": 0.0,
            "match_tier": "",
            "forward_days_available": min(horizon_days, len(dates) - index - 1),
        }
        analog["similarity_score"] = round(_similarity_score(current, analog), 6)
        analog["match_tier"] = _match_tier(current, analog)
        rows.append(analog)
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    strict = frame[frame["match_tier"] == "strict"].copy()
    if len(strict) >= min_analogs:
        return strict.sort_values(["similarity_score", "analog_start"], ascending=[False, False]).head(30)
    relaxed = frame[frame["match_tier"].isin(["strict", "same_target_regime", "same_target"])].copy()
    if len(relaxed) >= min_analogs:
        return relaxed.sort_values(["match_tier", "similarity_score"], ascending=[True, False]).head(30)
    return frame.sort_values("similarity_score", ascending=False).head(30)


def _similarity_score(current: ScenarioInputs, analog: dict[str, Any]) -> float:
    score = 0.0
    if analog["target_ticker"] == current.target_ticker:
        score += 4.0
    if analog["regime"] == current.current_regime:
        score += 2.0
    if analog["mode"] == current.current_mode:
        score += 1.0
    if bool(analog["attack_gate_active"]) == current.current_attack_gate_active:
        score += 1.0
    score -= min(abs(float(analog["top_stock_score"]) - current.current_top_stock_score), 1.0)
    score -= min(abs(float(analog["target_score"]) - current.current_target_score), 1.0)
    return score


def _match_tier(current: ScenarioInputs, analog: dict[str, Any]) -> str:
    if (
        analog["target_ticker"] == current.target_ticker
        and analog["regime"] == current.current_regime
        and analog["mode"] == current.current_mode
        and bool(analog["attack_gate_active"]) == current.current_attack_gate_active
    ):
        return "strict"
    if analog["target_ticker"] == current.target_ticker and analog["regime"] == current.current_regime:
        return "same_target_regime"
    if analog["target_ticker"] == current.target_ticker:
        return "same_target"
    return "nearest"


def _build_normalized_paths(
    *,
    equity: pd.DataFrame,
    analogs: pd.DataFrame,
    initial_capital: float,
    horizon_days: int,
) -> pd.DataFrame:
    if analogs.empty:
        return pd.DataFrame()
    dates = list(equity.index)
    rows: list[dict[str, Any]] = []
    for _, analog in analogs.iterrows():
        start = pd.Timestamp(analog["analog_start"])
        start_index = dates.index(start)
        start_value = float(equity.loc[start, "total_value"])
        if start_value <= 0:
            continue
        for step in range(0, min(horizon_days, len(dates) - start_index - 1) + 1):
            date = dates[start_index + step]
            row = equity.loc[date]
            value = float(row["total_value"]) / start_value * initial_capital
            rows.append(
                {
                    "analog_start": analog["analog_start"],
                    "step": step,
                    "historical_date": date.strftime("%Y-%m-%d"),
                    "projected_value": round(value, 2),
                    "return_pct": round(value / initial_capital - 1, 6),
                    "current_ticker": row["current_ticker"],
                    "regime": row["regime"],
                    "match_tier": analog["match_tier"],
                    "similarity_score": analog["similarity_score"],
                }
            )
    return pd.DataFrame(rows)


def _build_hold_counterfactual_paths(
    *,
    prices: dict[str, pd.DataFrame],
    analogs: pd.DataFrame,
    ticker: str,
    initial_capital: float,
    horizon_days: int,
) -> pd.DataFrame:
    if analogs.empty or ticker not in prices:
        return pd.DataFrame()
    frame = prices[ticker].sort_index()
    dates = list(frame.index)
    rows: list[dict[str, Any]] = []
    for _, analog in analogs.iterrows():
        start = pd.Timestamp(analog["analog_start"])
        if start not in frame.index:
            continue
        start_index = dates.index(start)
        start_price = float(frame.loc[start, "adj_close"])
        if start_price <= 0:
            continue
        for step in range(0, min(horizon_days, len(dates) - start_index - 1) + 1):
            date = dates[start_index + step]
            value = float(frame.loc[date, "adj_close"]) / start_price * initial_capital
            rows.append(
                {
                    "analog_start": analog["analog_start"],
                    "step": step,
                    "historical_date": date.strftime("%Y-%m-%d"),
                    "projected_value": round(value, 2),
                    "return_pct": round(value / initial_capital - 1, 6),
                    "ticker": ticker,
                    "match_tier": analog["match_tier"],
                    "similarity_score": analog["similarity_score"],
                }
            )
    return pd.DataFrame(rows)


def _summarize_paths(paths: pd.DataFrame, current: ScenarioInputs) -> pd.DataFrame:
    if paths.empty:
        return pd.DataFrame()
    final_rows = paths.sort_values("step").groupby("analog_start").tail(1).copy()
    path_stats = []
    for analog_start, group in paths.groupby("analog_start"):
        values = group.sort_values("step")["projected_value"]
        path_stats.append(
            {
                "analog_start": analog_start,
                "final_value": float(values.iloc[-1]),
                "total_return_pct": float(values.iloc[-1] / current.initial_capital - 1),
                "max_drawdown_pct": _max_drawdown(values),
                "ending_ticker": str(group.sort_values("step").iloc[-1]["current_ticker"]),
                "match_tier": str(group.iloc[0]["match_tier"]),
                "similarity_score": float(group.iloc[0]["similarity_score"]),
            }
        )
    stats = pd.DataFrame(path_stats)
    rows = [
        _summary_row("optimistic_p90", stats, 0.90),
        _summary_row("upper_p75", stats, 0.75),
        _summary_row("median_p50", stats, 0.50),
        _summary_row("lower_p25", stats, 0.25),
        _summary_row("stress_p10", stats, 0.10),
    ]
    rows.append(
        {
            "scenario": "best_analog",
            "final_value": round(float(final_rows.loc[final_rows["projected_value"].idxmax(), "projected_value"]), 2),
            "total_return_pct": round(float(final_rows["projected_value"].max() / current.initial_capital - 1), 6),
            "max_drawdown_pct": round(float(stats.loc[stats["final_value"].idxmax(), "max_drawdown_pct"]), 6),
            "analog_count": int(len(stats)),
            "note": "highest final value among selected analog paths",
        }
    )
    rows.append(
        {
            "scenario": "worst_analog",
            "final_value": round(float(final_rows.loc[final_rows["projected_value"].idxmin(), "projected_value"]), 2),
            "total_return_pct": round(float(final_rows["projected_value"].min() / current.initial_capital - 1), 6),
            "max_drawdown_pct": round(float(stats.loc[stats["final_value"].idxmin(), "max_drawdown_pct"]), 6),
            "analog_count": int(len(stats)),
            "note": "lowest final value among selected analog paths",
        }
    )
    return pd.DataFrame(rows)


def _summarize_hold_paths(paths: pd.DataFrame, current: ScenarioInputs) -> pd.DataFrame:
    if paths.empty:
        return pd.DataFrame()
    path_stats = []
    for analog_start, group in paths.groupby("analog_start"):
        values = group.sort_values("step")["projected_value"]
        path_stats.append(
            {
                "analog_start": analog_start,
                "final_value": float(values.iloc[-1]),
                "total_return_pct": float(values.iloc[-1] / current.initial_capital - 1),
                "max_drawdown_pct": _max_drawdown(values),
                "match_tier": str(group.iloc[0]["match_tier"]),
                "similarity_score": float(group.iloc[0]["similarity_score"]),
            }
        )
    stats = pd.DataFrame(path_stats)
    rows = [
        _summary_row("hold_optimistic_p90", stats, 0.90),
        _summary_row("hold_upper_p75", stats, 0.75),
        _summary_row("hold_median_p50", stats, 0.50),
        _summary_row("hold_lower_p25", stats, 0.25),
        _summary_row("hold_stress_p10", stats, 0.10),
    ]
    rows.append(
        {
            "scenario": "hold_best_analog",
            "final_value": round(float(stats["final_value"].max()), 2),
            "total_return_pct": round(float(stats["total_return_pct"].max()), 6),
            "max_drawdown_pct": round(float(stats.loc[stats["final_value"].idxmax(), "max_drawdown_pct"]), 6),
            "analog_count": int(len(stats)),
            "note": "highest final value among selected hold paths",
        }
    )
    rows.append(
        {
            "scenario": "hold_worst_analog",
            "final_value": round(float(stats["final_value"].min()), 2),
            "total_return_pct": round(float(stats["total_return_pct"].min()), 6),
            "max_drawdown_pct": round(float(stats.loc[stats["final_value"].idxmin(), "max_drawdown_pct"]), 6),
            "analog_count": int(len(stats)),
            "note": "lowest final value among selected hold paths",
        }
    )
    return pd.DataFrame(rows)


def _summary_row(name: str, stats: pd.DataFrame, quantile: float) -> dict[str, Any]:
    return {
        "scenario": name,
        "final_value": round(float(stats["final_value"].quantile(quantile)), 2),
        "total_return_pct": round(float(stats["total_return_pct"].quantile(quantile)), 6),
        "max_drawdown_pct": round(float(stats["max_drawdown_pct"].quantile(quantile)), 6),
        "analog_count": int(len(stats)),
        "note": f"quantile {quantile:.0%} of selected analog paths",
    }


def _trade_examples(trades: list[Any], analogs: pd.DataFrame, labels: dict[str, str], horizon_days: int) -> pd.DataFrame:
    if analogs.empty:
        return pd.DataFrame()
    analog_starts = {str(value) for value in analogs["analog_start"].tolist()}
    rows = []
    for trade in trades:
        trade_date = pd.Timestamp(trade.date)
        for analog_start in analog_starts:
            start = pd.Timestamp(analog_start)
            if start <= trade_date <= start + pd.offsets.BDay(horizon_days):
                rows.append(
                    {
                        "analog_start": analog_start,
                        "trade_date": trade.date,
                        "ticker": trade.ticker,
                        "label": _label(trade.ticker, labels),
                        "action": trade.action,
                        "shares": trade.shares,
                        "price": trade.price,
                        "reason": trade.reason,
                    }
                )
    return pd.DataFrame(rows).sort_values(["analog_start", "trade_date"]) if rows else pd.DataFrame()


def _write_path_chart(path: Path, paths: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    for _, group in paths.groupby("analog_start"):
        group = group.sort_values("step")
        ax.plot(group["step"], group["projected_value"] / 10_000, color="#7aa6c2", alpha=0.28, linewidth=1)
    median = paths.groupby("step")["projected_value"].median().reset_index()
    ax.plot(median["step"], median["projected_value"] / 10_000, color="#0f766e", linewidth=2.5, label="median")
    ax.set_title("Future Path Scenario Simulator")
    ax.set_xlabel("Trading days from scenario start")
    ax.set_ylabel("Projected value (TWD 10k)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _markdown_report(
    inputs: ScenarioInputs,
    analogs: pd.DataFrame,
    summary: pd.DataFrame,
    hold_summary: pd.DataFrame,
) -> str:
    lines = [
        "# Future Path Scenario Simulator",
        "",
        "This is a historical analog scenario tool, not a forecast and not investment advice.",
        "",
        "## Inputs",
        "",
        f"- Signal date: {inputs.signal_date}",
        f"- Scenario window: {inputs.scenario_start} to {inputs.scenario_end}",
        f"- Initial capital: {inputs.initial_capital:,.0f}",
        f"- Current model target: {inputs.target_label} ({inputs.target_ticker})",
        f"- Current regime/mode: {inputs.current_regime} / {inputs.current_mode}",
        f"- Top stock score: {inputs.current_top_stock_label} {inputs.current_top_stock_score:.4f}",
        f"- Target score: {inputs.target_label} {inputs.current_target_score:.4f}",
        "",
        "## Analog Cases",
        "",
        f"- Selected analog cases: {len(analogs)}",
        f"- Strict matches: {int((analogs['match_tier'] == 'strict').sum()) if not analogs.empty else 0}",
        "",
        "## Scenario Summary",
        "",
    ]
    if summary.empty:
        lines.append("No scenario paths were available.")
    else:
        lines.extend(
            [
                "| Scenario | Final value | Return | Max drawdown |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for _, row in summary.iterrows():
            lines.append(
                f"| {row['scenario']} | {float(row['final_value']):,.0f} | "
                f"{float(row['total_return_pct']):+.2%} | {float(row['max_drawdown_pct']):+.2%} |"
            )
    lines.extend(
        [
            "",
            f"## Hold {inputs.current_top_stock_label} Counterfactual",
            "",
        ]
    )
    if hold_summary.empty:
        lines.append("No hold counterfactual paths were available.")
    else:
        lines.extend(
            [
                "| Scenario | Final value | Return | Max drawdown |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for _, row in hold_summary.iterrows():
            lines.append(
                f"| {row['scenario']} | {float(row['final_value']):,.0f} | "
                f"{float(row['total_return_pct']):+.2%} | {float(row['max_drawdown_pct']):+.2%} |"
            )
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- analog_cases.csv",
            "- scenario_paths.csv",
            "- scenario_summary.csv",
            "- hold_top_stock_paths.csv",
            "- hold_top_stock_summary.csv",
            "- trade_path_examples.csv",
            "- scenario_paths.png",
        ]
    )
    return "\n".join(lines)


def _top_attack_stock(scores: dict[str, float]) -> str:
    if not scores:
        return "cash"
    attack_scores = {
        ticker: score
        for ticker, score in scores.items()
        if ticker not in {"0050.TW", "00631L.TW"}
    }
    if not attack_scores:
        return max(scores.items(), key=lambda item: (item[1], item[0]))[0]
    return max(attack_scores.items(), key=lambda item: (item[1], item[0]))[0]


def _label(ticker: str, labels: dict[str, str]) -> str:
    if ticker == "cash":
        return "cash"
    return labels.get(ticker, ticker)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run historical analog future-path scenarios for the frozen best model.")
    parser.add_argument("--config", default="configs/ep05_universe.json")
    parser.add_argument("--group-id", default=DEFAULT_GROUP_ID)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--signal-date", default=DEFAULT_SIGNAL_DATE)
    parser.add_argument("--scenario-start", default=DEFAULT_SCENARIO_START)
    parser.add_argument("--scenario-end", default=DEFAULT_SCENARIO_END)
    parser.add_argument("--history-start", default=DEFAULT_HISTORY_START)
    parser.add_argument("--replay-start", default=DEFAULT_REPLAY_START)
    parser.add_argument("--initial-capital", type=float, default=DEFAULT_INITIAL_CAPITAL)
    parser.add_argument("--min-analogs", type=int, default=8)
    args = parser.parse_args()

    payload = run_future_path_scenario(
        config_path=args.config,
        group_id=args.group_id,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        signal_date=args.signal_date,
        scenario_start=args.scenario_start,
        scenario_end=args.scenario_end,
        history_start=args.history_start,
        replay_start=args.replay_start,
        initial_capital=args.initial_capital,
        min_analogs=args.min_analogs,
    )
    print(f"FUTURE_PATH_SCENARIO_DIR={Path(payload['output_dir']).resolve()}")


if __name__ == "__main__":
    main()
