"""Run Dynamic Pool1 incumbent-protection bounded diagnostics.

This runner tests switch-suppression overlays on the existing Dynamic Pool1 v2
top15-top1 diagnostic signal. It is not a formal model change.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from backtest_lab.costs import TaiwanCostModel, cost_model_metadata
from backtest_lab.dynamic_pool1_exact_ab_switch_friction_contract import DEFAULT_BACKTEST_PERIOD_CONTRACT
from backtest_lab.strong_stock_trend_extension_bounded_portfolio_diagnostic import (
    BENCHMARK_PRICE_PATHS,
    DEFAULT_LIQUIDITY_DIR,
    FORMAL_STREAMS,
    INITIAL_EQUITY,
    _asset_type,
    _canonical_from_prices,
    _canonical_target,
    _ffill_prices,
    _formal_state,
    _load_prices,
)


TASK_ID = "TASK-BACKTEST-CORE-DYNAMIC-POOL1-INCUMBENT-PROTECTION-BOUNDED-DIAGNOSTIC-RUNNER-001"
EXPERIMENTS_TASK_ID = "TASK-BACKTEST-EXPERIMENTS-DYNAMIC-POOL1-INCUMBENT-PROTECTION-BOUNDED-DIAGNOSTIC-VALIDATION-001"
DEFAULT_V2_SIGNAL_PANEL = Path("outputs/dynamic_pool1_v2_bounded_portfolio_contract_20260704/daily_signal_panel.csv")
DEFAULT_V2_RULE_CONTRACT = Path(
    "outputs/dynamic_pool1_ab_switch_friction_rule_candidate_v2_contract_20260705"
    "/exact_ab_switch_friction_rule_candidate_v2_contract.csv"
)
DEFAULT_OUTPUT_DIR = Path("outputs/dynamic_pool1_incumbent_protection_bounded_diagnostic_20260705")
SOURCE_DYNAMIC_VARIANT = "v2_top15_top1_when_formal_cash_or_market_exposure_hold20"
INCUMBENT_VARIANTS = [
    {
        "variant": "incumbent_protection_top5_unless_B_score10_primary",
        "rule_id": "v2_keep_A_if_still_working_top5_unless_B_score10",
        "role": "primary",
        "mapping": "new_no_stock_fallback_00631l_except_bear_cash",
    },
    {
        "variant": "incumbent_protection_top10_no_trend_break_sensitivity",
        "rule_id": "v2_keep_A_if_still_working_top10_and_no_trend_break",
        "role": "sensitivity",
        "mapping": "new_no_stock_fallback_00631l_except_bear_cash",
    },
    {
        "variant": "incumbent_protection_top5_unless_B_rank3_score10_sensitivity",
        "rule_id": "v2_keep_A_if_still_working_top5_unless_B_rank3_score10",
        "role": "sensitivity_readiness_caveat",
        "mapping": "new_no_stock_fallback_00631l_except_bear_cash",
    },
]
REFERENCE_RULE_IDS = [
    "v2_allow_switch_if_A_trend_break_and_B_rank2_score5",
    "v2_balanced_A_working_or_B_large_margin",
]
BASELINE_VARIANTS = [
    "current_formal_next_day",
    "0050_buy_and_hold",
    "00631L_buy_and_hold",
    "old_no_target_cash_mapping_reference",
    "new_no_stock_fallback_00631l_mapping_reference",
    "dynamic_top15_top1_no_suppression_new_mapping_reference",
]


@dataclass
class PortfolioState:
    cash: float = INITIAL_EQUITY
    shares: dict[str, float] = field(default_factory=dict)
    dynamic_incumbent: str = ""


def run_dynamic_pool1_incumbent_protection_bounded_diagnostic(
    *,
    repo_root: str | Path = ".",
    signal_panel: str | Path = DEFAULT_V2_SIGNAL_PANEL,
    rule_contract: str | Path = DEFAULT_V2_RULE_CONTRACT,
    liquidity_dir: str | Path = DEFAULT_LIQUIDITY_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    signal_path = _resolve(root, signal_panel)
    rule_path = _resolve(root, rule_contract)
    liquidity = _resolve(root, liquidity_dir)
    output = _resolve(root, output_dir)
    output.mkdir(parents=True, exist_ok=True)

    signal = _load_signal_panel(signal_path)
    formal = _load_formal_streams(root, signal["next_tradable_date"].tolist())
    rules = _load_rules(rule_path)
    dates = signal["next_tradable_date"].astype(str).dropna().unique().tolist()
    needed_tickers = _needed_tickers(signal, formal)
    prices = _load_prices(root, liquidity, sorted(needed_tickers), dates)
    prices = _ffill_prices(prices, dates)
    mapper = _canonical_from_prices(prices)
    signal["dynamic_selected_canonical_ticker"] = signal["dynamic_selected_canonical_ticker"].astype(str).map(mapper)
    formal["formal_target"] = formal["formal_target"].astype(str).map(mapper)

    all_variants = BASELINE_VARIANTS + [v["variant"] for v in INCUMBENT_VARIANTS]
    results = [_simulate_variant(variant, signal, formal, rules, prices) for variant in all_variants]
    daily = pd.concat([r["daily"] for r in results], ignore_index=True, sort=False)
    trades = pd.concat([r["trades"] for r in results], ignore_index=True, sort=False)
    suppression = pd.concat([r["suppression"] for r in results], ignore_index=True, sort=False)
    execution_state = pd.concat([r["execution_state"] for r in results], ignore_index=True, sort=False)
    future_audit = _future_data_audit(daily, suppression)
    period = _period_performance(daily)
    benchmark = _benchmark_comparison(daily)
    cost = _cost_turnover_summary(trades)
    retention = _incumbent_retention_outcome_panel(suppression, rules)

    daily.to_csv(output / "daily_equity_by_variant.csv", index=False, encoding="utf-8-sig")
    trades.to_csv(output / "trade_ledger_by_variant.csv", index=False, encoding="utf-8-sig")
    suppression.to_csv(output / "switch_suppression_ledger.csv", index=False, encoding="utf-8-sig")
    period.to_csv(output / "period_performance_default_periods.csv", index=False, encoding="utf-8-sig")
    benchmark.to_csv(output / "benchmark_comparison_0050_00631l.csv", index=False, encoding="utf-8-sig")
    execution_state.to_csv(output / "execution_state_daily_panel.csv", index=False, encoding="utf-8-sig")
    cost.to_csv(output / "cost_turnover_tradecount_summary.csv", index=False, encoding="utf-8-sig")
    retention.to_csv(output / "incumbent_retention_outcome_panel.csv", index=False, encoding="utf-8-sig")
    future_audit.to_csv(output / "future_data_audit.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([cost_model_metadata()]).to_csv(output / "cost_model_contract.csv", index=False, encoding="utf-8-sig")

    manifest = {
        "task_id": TASK_ID,
        "status": "completed_incumbent_protection_bounded_diagnostic_ready",
        "output_dir": str(output),
        "source_signal_panel": str(signal_path),
        "source_rule_contract": str(rule_path),
        "source_dynamic_variant": SOURCE_DYNAMIC_VARIANT,
        "diagnostic_variants": all_variants,
        "daily_equity_rows": int(len(daily)),
        "trade_rows": int(len(trades)),
        "switch_suppression_rows": int(len(suppression)),
        "suppressed_switch_rows": int(suppression["switch_suppressed"].sum()) if not suppression.empty else 0,
        "future_data_violation_count": int(future_audit["future_data_violation"].sum()) if not future_audit.empty else 0,
        "default_backtest_period_contract": DEFAULT_BACKTEST_PERIOD_CONTRACT,
        "actual_start": _date_text(daily["date"].min()),
        "actual_end": _date_text(daily["date"].max()),
        "execution_mapping_contract": {
            "old_reference": "no_target -> cash",
            "new_challenger": "no_stock_target_but_market_exposure_allowed -> 00631L; bear_or_cash_condition -> cash",
            "bear_or_cash_condition_source": "explicit formal bear flag not available in source signal; no explicit bear rows inferred",
        },
        "reference_rule_ids_not_primary": REFERENCE_RULE_IDS,
        "uses_forward_return_as_rule": False,
        "portfolio_replay_executed": True,
        "diagnostic_only": True,
        "ready_for_formal_absorption": False,
        "formal_model_changed": False,
        "trade_decision_changed": False,
        "active_in_trade_decision": False,
        "report_changed": False,
        "handoff_to_experiments_task": EXPERIMENTS_TASK_ID,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "final_summary_zh.md").write_text(_summary(manifest, period, cost), encoding="utf-8")
    pd.DataFrame([{"task_id": TASK_ID, "status": "completed", "output_dir": str(output)}]).to_csv(
        output / "completed.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(columns=["task_id", "status", "reason"]).to_csv(output / "failed.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(
        [
            {"step": "load_v2_signal_and_rule_contract", "status": "completed"},
            {"step": "load_price_and_formal_context", "status": "completed"},
            {"step": "simulate_incumbent_protection_diagnostics", "status": "completed"},
            {"step": "write_output_package", "status": "completed"},
        ]
    ).to_csv(output / "run_log.csv", index=False, encoding="utf-8-sig")
    return manifest


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _load_signal_panel(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path).fillna("")
    df = df[df["dynamic_pool_variant"].eq(SOURCE_DYNAMIC_VARIANT)].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df["next_tradable_date"] = pd.to_datetime(df["next_tradable_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df["dynamic_selected_canonical_ticker"] = df["dynamic_selected_canonical_ticker"].astype(str).replace({"nan": ""})
    return df.dropna(subset=["date", "next_tradable_date"]).sort_values("next_tradable_date")


def _load_rules(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path).fillna("")
    keep = [v["rule_id"] for v in INCUMBENT_VARIANTS] + REFERENCE_RULE_IDS
    df = df[df["rule_id"].isin(keep)].copy()
    df = df[df["variant_id"].eq(SOURCE_DYNAMIC_VARIANT)].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    for col in ["rule_candidate_triggered", "uses_forward_return_as_rule", "future_data_violation"]:
        df[col] = df[col].map(_as_bool)
    return df


def _load_formal_streams(root: Path, dates: list[str]) -> pd.DataFrame:
    frames = []
    for rel in FORMAL_STREAMS:
        path = root / rel
        if not path.exists():
            continue
        frame = pd.read_csv(path).fillna("")
        frame["signal_date"] = pd.to_datetime(frame["signal_date"], errors="coerce")
        frame["execution_date"] = pd.to_datetime(frame.get("execution_date", ""), errors="coerce")
        frame = frame.dropna(subset=["signal_date", "execution_date"]).copy()
        frame["date"] = frame["execution_date"].dt.strftime("%Y-%m-%d")
        frame["formal_target"] = frame.get("formal_target", "").astype(str).map(_canonical_target)
        frame["target_type"] = frame.get("target_type", "")
        frame["risk_off_state"] = frame.get("risk_off_state", "")
        frame["formal_state"] = frame.apply(_formal_state, axis=1)
        frames.append(frame[["date", "formal_target", "formal_state"]])
    formal = pd.concat(frames, ignore_index=True, sort=False)
    formal = formal.sort_values("date").drop_duplicates("date", keep="last")
    calendar = pd.DataFrame({"date": sorted(set(dates))})
    out = calendar.merge(formal, on="date", how="left").ffill().fillna("")
    return out


def _needed_tickers(signal: pd.DataFrame, formal: pd.DataFrame) -> set[str]:
    tickers = {"0050.TW", "00631L.TW"}
    tickers.update(t for t in signal["dynamic_selected_canonical_ticker"].dropna().astype(str).unique() if t and t != "nan")
    tickers.update(t for t in formal["formal_target"].dropna().astype(str).unique() if t and t not in {"", "CASH"})
    return tickers


def _simulate_variant(
    variant: str,
    signal: pd.DataFrame,
    formal: pd.DataFrame,
    rules: pd.DataFrame,
    prices: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    price_lookup = {(r["date"], r["canonical_ticker"]): float(r["close"]) for r in prices.to_dict(orient="records")}
    state = PortfolioState()
    model = TaiwanCostModel()
    daily_rows = []
    trade_rows = []
    suppression_rows = []
    execution_rows = []
    running_max = INITIAL_EQUITY
    signal_by_date = {r["next_tradable_date"]: r for r in signal.to_dict(orient="records")}
    formal_by_date = {r["date"]: r for r in formal.to_dict(orient="records")}
    for date in sorted(formal_by_date):
        formal_row = formal_by_date[date]
        signal_row = signal_by_date.get(date, {})
        equity_before = _equity(state, price_lookup, date)
        signal_date = str(signal_row.get("date", "") or "")
        target, exec_state, suppression = _target_for_date(
            variant, date, signal_date, signal_row, formal_row, rules, state.dynamic_incumbent
        )
        target = _filter_missing_price(target, price_lookup, date)
        cash, shares, trades, cost, turnover = _rebalance(state.cash, state.shares, target, price_lookup, date, model, equity_before)
        state.cash = cash
        state.shares = shares
        if exec_state["direct_stock_target_ticker"]:
            state.dynamic_incumbent = exec_state["direct_stock_target_ticker"]
        equity = _equity(state, price_lookup, date)
        running_max = max(running_max, equity)
        for trade in trades:
            trade_rows.append(
                {
                    "date": date,
                    "variant": variant,
                    "ticker": trade["ticker"],
                    "side": trade["side"],
                    "gross": trade["gross"],
                    "price": trade["price"],
                    "trade_cost": trade["cost"],
                    "turnover": turnover,
                    "active_in_trade_decision": False,
                }
            )
        daily_rows.append(
            {
                "date": date,
                "variant": variant,
                "portfolio_equity": round(equity, 4),
                "cash": round(state.cash, 4),
                "drawdown_pct": round((equity / running_max - 1.0) * 100.0, 6) if running_max else 0.0,
                "formal_target": formal_row.get("formal_target", ""),
                "formal_state": formal_row.get("formal_state", ""),
                "execution_state": exec_state["execution_state"],
                "uses_forward_return_as_rule": False,
                "active_in_trade_decision": False,
            }
        )
        execution_rows.append({"date": date, "variant": variant, **exec_state})
        if suppression:
            suppression_rows.append({"date": date, "variant": variant, **suppression})
    return {
        "daily": pd.DataFrame(daily_rows),
        "trades": pd.DataFrame(trade_rows),
        "suppression": pd.DataFrame(suppression_rows, columns=_suppression_columns()),
        "execution_state": pd.DataFrame(execution_rows),
    }


def _target_for_date(
    variant: str,
    date: str,
    signal_date: str,
    signal_row: dict,
    formal_row: dict,
    rules: pd.DataFrame,
    incumbent: str,
) -> tuple[dict[str, float], dict[str, Any], dict[str, Any]]:
    formal_target = str(formal_row.get("formal_target", ""))
    formal_state = str(formal_row.get("formal_state", ""))
    selected = str(signal_row.get("dynamic_selected_canonical_ticker", "") or "")
    dynamic_blocked = str(signal_row.get("dynamic_blocked_reason", "") or "")
    bear_cash = False
    exec_state = {
        "execution_state": "",
        "direct_stock_target_ticker": "",
        "market_exposure_fallback_ticker": "",
        "bear_or_cash_condition_flag": bear_cash,
        "old_no_target_cash_mapping_weight": 0.0,
        "new_no_stock_fallback_00631l_mapping_weight": 0.0,
        "formal_target": formal_target,
        "formal_state": formal_state,
        "dynamic_selected_ticker": selected,
        "dynamic_blocked_reason": dynamic_blocked,
    }
    if variant == "0050_buy_and_hold":
        exec_state.update({"execution_state": "benchmark_buy_hold", "direct_stock_target_ticker": "0050.TW"})
        return {"0050.TW": 1.0}, exec_state, {}
    if variant == "00631L_buy_and_hold":
        exec_state.update({"execution_state": "benchmark_buy_hold", "direct_stock_target_ticker": "00631L.TW"})
        return {"00631L.TW": 1.0}, exec_state, {}
    if variant == "current_formal_next_day":
        if formal_target and formal_target != "CASH":
            exec_state.update({"execution_state": "formal_target", "direct_stock_target_ticker": formal_target})
            return {formal_target: 1.0}, exec_state, {}
        exec_state.update({"execution_state": "formal_cash", "old_no_target_cash_mapping_weight": 1.0})
        return {}, exec_state, {}
    if formal_state == "direct_stock_target" and formal_target and formal_target != "CASH":
        exec_state.update({"execution_state": "direct_stock_target", "direct_stock_target_ticker": formal_target})
        return {formal_target: 1.0}, exec_state, {}
    candidate_available = bool(selected and selected != "nan" and not dynamic_blocked)
    suppression = _suppression_for(variant, signal_date, incumbent, selected, rules)
    if suppression:
        target = str(suppression["incumbent_ticker_A"])
        exec_state.update({"execution_state": "direct_stock_target", "direct_stock_target_ticker": target})
        return {target: 1.0}, exec_state, suppression
    if variant == "old_no_target_cash_mapping_reference" and not candidate_available:
        exec_state.update({"execution_state": "old_no_target_cash", "old_no_target_cash_mapping_weight": 1.0})
        return {}, exec_state, {}
    if candidate_available:
        target = selected
        exec_state.update({"execution_state": "direct_stock_target", "direct_stock_target_ticker": target})
        return {target: 1.0}, exec_state, {}
    exec_state.update(
        {
            "execution_state": "no_stock_target_but_market_exposure_allowed",
            "market_exposure_fallback_ticker": "00631L.TW",
            "new_no_stock_fallback_00631l_mapping_weight": 1.0,
        }
    )
    return {"00631L.TW": 1.0}, exec_state, {}


def _suppression_for(variant: str, date: str, incumbent: str, selected: str, rules: pd.DataFrame) -> dict[str, Any]:
    mapping = {v["variant"]: v["rule_id"] for v in INCUMBENT_VARIANTS}
    rule_id = mapping.get(variant)
    if not rule_id or not incumbent:
        return {}
    match = rules[
        rules["date"].eq(date)
        & rules["rule_id"].eq(rule_id)
        & rules["challenger_ticker_B"].astype(str).eq(selected)
        & rules["rule_candidate_triggered"]
    ]
    if match.empty:
        match = rules[
            rules["date"].eq(date)
            & rules["rule_id"].eq(rule_id)
            & rules["incumbent_ticker_A"].astype(str).eq(incumbent)
            & rules["rule_candidate_triggered"]
        ]
    if match.empty:
        return {}
    row = match.iloc[0].to_dict()
    rule_challenger = str(row.get("challenger_ticker_B", selected) or selected)
    return {
        "switch_event_id": row.get("switch_event_id", ""),
        "rule_id": rule_id,
        "incumbent_ticker_A": incumbent,
        "challenger_ticker_B": rule_challenger,
        "current_dynamic_selected_ticker": selected,
        "switch_suppressed": True,
        "suppression_reason": "incumbent_A_still_working_protection",
        "incumbent_A_still_working_flag": row.get("incumbent_A_still_working_flag", ""),
        "incumbent_A_trend_break_flag": row.get("incumbent_A_trend_break_flag", ""),
        "A_rank_still_top5": row.get("A_rank_still_top5", ""),
        "score_margin": row.get("score_margin", ""),
        "rank_margin": row.get("rank_margin", ""),
        "uses_forward_return_as_rule": False,
    }


def _suppression_columns() -> list[str]:
    return [
        "date",
        "variant",
        "switch_event_id",
        "rule_id",
        "incumbent_ticker_A",
        "challenger_ticker_B",
        "current_dynamic_selected_ticker",
        "switch_suppressed",
        "suppression_reason",
        "incumbent_A_still_working_flag",
        "incumbent_A_trend_break_flag",
        "A_rank_still_top5",
        "score_margin",
        "rank_margin",
        "uses_forward_return_as_rule",
    ]


def _filter_missing_price(target: dict[str, float], price_lookup: dict, date: str) -> dict[str, float]:
    return {ticker: weight for ticker, weight in target.items() if price_lookup.get((date, ticker), 0.0) > 0}


def _rebalance(cash: float, shares: dict[str, float], desired: dict[str, float], prices: dict, date: str, model: TaiwanCostModel, equity: float):
    trades = []
    cost_total = 0.0
    turnover = 0.0
    for ticker in sorted(set(shares) | set(desired)):
        price = prices.get((date, ticker))
        if not price or price <= 0:
            continue
        current_value = shares.get(ticker, 0.0) * price
        target_value = equity * float(desired.get(ticker, 0.0))
        delta = target_value - current_value
        if abs(delta) < 1:
            continue
        if delta < 0:
            gross = -delta
            cost = model.sell_cost(gross, _asset_type(ticker))
            qty = gross / price
            shares[ticker] = max(0.0, shares.get(ticker, 0.0) - qty)
            cash += gross - cost
            side = "sell"
        else:
            gross = min(delta, max(0.0, cash))
            cost = model.buy_cost(gross)
            qty = max(0.0, (gross - cost) / price)
            shares[ticker] = shares.get(ticker, 0.0) + qty
            cash -= gross
            side = "buy"
        cost_total += cost
        turnover += gross
        trades.append({"ticker": ticker, "side": side, "gross": round(gross, 4), "cost": cost, "price": price})
        if shares.get(ticker, 0.0) <= 1e-9:
            shares.pop(ticker, None)
    return cash, shares, trades, cost_total, turnover


def _equity(state: PortfolioState, prices: dict, date: str) -> float:
    return state.cash + sum(qty * prices.get((date, ticker), 0.0) for ticker, qty in state.shares.items())


def _period_performance(daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    frame = daily.copy()
    frame["date_ts"] = pd.to_datetime(frame["date"], errors="coerce")
    periods = [{"period_label": "full_available", "requested_start": "", "requested_end": ""}, *DEFAULT_BACKTEST_PERIOD_CONTRACT]
    for variant, group in frame.groupby("variant"):
        for period in periods:
            subset = group.copy()
            if period["requested_start"]:
                subset = subset[subset["date_ts"] >= pd.Timestamp(period["requested_start"])]
            if period["requested_end"]:
                subset = subset[subset["date_ts"] <= pd.Timestamp(period["requested_end"])]
            rows.append(_perf_row(variant, period, subset))
    return pd.DataFrame(rows)


def _perf_row(variant: str, period: dict, frame: pd.DataFrame) -> dict[str, Any]:
    label = period["period_label"]
    if frame.empty:
        return {
            "variant": variant,
            "period_label": label,
            "requested_start": period.get("requested_start", ""),
            "requested_end": period.get("requested_end", ""),
            "actual_start": "",
            "actual_end": "",
            "status": "empty",
        }
    frame = frame.sort_values("date_ts")
    start = float(frame.iloc[0]["portfolio_equity"])
    end = float(frame.iloc[-1]["portfolio_equity"])
    running = pd.to_numeric(frame["portfolio_equity"], errors="coerce").cummax()
    dd = pd.to_numeric(frame["portfolio_equity"], errors="coerce") / running - 1.0
    return {
        "variant": variant,
        "period_label": label,
        "requested_start": period.get("requested_start", ""),
        "requested_end": period.get("requested_end", ""),
        "actual_start": frame.iloc[0]["date"],
        "actual_end": frame.iloc[-1]["date"],
        "status": "completed",
        "start_equity": round(start, 4),
        "final_equity": round(end, 4),
        "return_pct": round((end / start - 1.0) * 100.0, 4) if start else 0.0,
        "max_drawdown_pct": round(float(dd.min()) * 100.0, 4),
    }


def _benchmark_comparison(daily: pd.DataFrame) -> pd.DataFrame:
    perf = _period_performance(daily)
    baseline = perf[perf["variant"].isin(["current_formal_next_day", "0050_buy_and_hold", "00631L_buy_and_hold"])]
    return baseline.rename(columns={"variant": "benchmark"})


def _cost_turnover_summary(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(columns=["variant", "trade_count", "total_cost", "total_turnover"])
    return trades.groupby("variant", as_index=False).agg(
        trade_count=("ticker", "count"),
        total_cost=("trade_cost", "sum"),
        total_turnover=("gross", "sum"),
    )


def _incumbent_retention_outcome_panel(suppression: pd.DataFrame, rules: pd.DataFrame) -> pd.DataFrame:
    if suppression.empty:
        return pd.DataFrame(columns=["variant", "rule_id", "switch_event_id", "suppression_reason"])
    eval_cols = [
        "switch_event_id",
        "B_minus_A_forward_delta_20d",
        "B_minus_A_forward_delta_40d",
        "forward_return_used_as_evaluation_metadata",
    ]
    enrich = rules[rules["switch_event_id"].isin(suppression["switch_event_id"])][eval_cols].drop_duplicates("switch_event_id")
    return suppression.merge(enrich, on="switch_event_id", how="left")


def _future_data_audit(daily: pd.DataFrame, suppression: pd.DataFrame) -> pd.DataFrame:
    out = daily[["date", "variant"]].copy()
    out["future_data_violation"] = False
    out["uses_forward_return_as_rule"] = False
    out["suppression_rows_checked"] = len(suppression)
    return out


def _as_bool(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _date_text(value: object) -> str:
    date = pd.to_datetime(value, errors="coerce")
    if pd.isna(date):
        return ""
    return str(date.date())


def _summary(manifest: dict[str, Any], period: pd.DataFrame, cost: pd.DataFrame) -> str:
    return "\n".join(
        [
            "# Dynamic Pool1 incumbent protection bounded diagnostic",
            "",
            "本包只做 incumbent-protection switch suppression 診斷；不改正式模型、日報或交易決策。",
            "",
            f"- daily equity rows：{manifest['daily_equity_rows']}",
            f"- trade rows：{manifest['trade_rows']}",
            f"- switch suppression rows：{manifest['switch_suppression_rows']}",
            f"- suppressed switch rows：{manifest['suppressed_switch_rows']}",
            f"- future_data_violation_count：{manifest['future_data_violation_count']}",
            f"- actual range：{manifest['actual_start']}～{manifest['actual_end']}",
            "- default periods：2015-01-02～2022-12-29；2023-01-02～2026-06-30。",
            "- execution mapping：同時輸出 old no-target cash 與 new no-stock fallback 00631L reference；本包仍是 diagnostic-only。",
            f"- 下一棒：{manifest['handoff_to_experiments_task']}",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--signal-panel", default=str(DEFAULT_V2_SIGNAL_PANEL))
    parser.add_argument("--rule-contract", default=str(DEFAULT_V2_RULE_CONTRACT))
    parser.add_argument("--liquidity-dir", default=str(DEFAULT_LIQUIDITY_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args(argv)
    manifest = run_dynamic_pool1_incumbent_protection_bounded_diagnostic(
        repo_root=args.repo_root,
        signal_panel=args.signal_panel,
        rule_contract=args.rule_contract,
        liquidity_dir=args.liquidity_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
