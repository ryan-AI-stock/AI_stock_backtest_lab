from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from backtest_lab.costs import TaiwanCostModel
from backtest_lab.data import load_price_csv
from backtest_lab.execution_layer_diagnostic import (
    build_execution_event_study_panel,
    build_formal_target_change_panel,
)


DEFAULT_OUTPUT_DIR = "outputs/partial_execution_ledger_20260625"
DEFAULT_PRICE_CACHE_DIR = "backtest_cache/stock_pool_triad_v1_corrected"
DEFAULT_INITIAL_CASH = 1_000_000.0
MARKET_EXPOSURE_TICKERS = {"0050.TW", "00631L.TW"}


@dataclass(frozen=True)
class ExecutionVariant:
    variant_id: str
    family: str
    partial_weight: float | None = None
    subset: str = "global"
    minimum_hold_days: int | None = None
    cooldown_days: int | None = None
    blocked: bool = False
    blocked_reason: str = ""


@dataclass
class LedgerAccount:
    cash: float
    positions: dict[str, int]


def run_partial_execution_ledger(
    *,
    formal_daily_path: str | Path,
    price_cache_dir: str | Path = DEFAULT_PRICE_CACHE_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    initial_cash: float = DEFAULT_INITIAL_CASH,
) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    run_log: list[dict[str, str]] = []

    def log(step: str, status: str, detail: str = "") -> None:
        run_log.append(
            {
                "timestamp": pd.Timestamp.now(tz="Asia/Taipei").strftime("%Y-%m-%d %H:%M:%S%z"),
                "step": step,
                "status": status,
                "detail": detail,
            }
        )
        pd.DataFrame(run_log).to_csv(output / "run_log.csv", index=False, encoding="utf-8-sig")
        (output / "current_step.txt").write_text(step, encoding="utf-8")

    try:
        log("load_inputs", "started", str(formal_daily_path))
        formal_daily = pd.read_csv(formal_daily_path).fillna("")
        _validate_formal_daily(formal_daily)
        frame = _normalize_formal_daily(formal_daily)
        variants = _variant_matrix()
        runnable = [variant for variant in variants if not variant.blocked]
        blocked = [variant for variant in variants if variant.blocked]

        log("load_prices", "started", str(price_cache_dir))
        prices = _load_prices(frame, Path(price_cache_dir))
        if not prices:
            raise ValueError("no prices loaded for partial execution ledger")

        log("build_event_context", "started", "")
        target_change = build_formal_target_change_panel(formal_daily)
        event_study = build_execution_event_study_panel(formal_daily, target_change, prices)
        event_context = _build_event_context(frame, event_study)

        log("simulate_variants", "started", f"variants={len(runnable)}")
        daily_frames: list[pd.DataFrame] = []
        trade_frames: list[pd.DataFrame] = []
        for variant in runnable:
            daily, trades = _simulate_variant(
                frame=frame,
                prices=prices,
                event_context=event_context,
                variant=variant,
                initial_cash=initial_cash,
            )
            daily_frames.append(daily)
            trade_frames.append(trades)
        daily_ledger = pd.concat(daily_frames, ignore_index=True) if daily_frames else pd.DataFrame()
        trade_ledger = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()

        log("build_summaries", "started", "")
        variant_df = _variant_parameter_matrix(variants)
        period_perf = _period_performance(daily_ledger, prices)
        cost_summary = _cost_turnover_summary(daily_ledger, trade_ledger)
        drawdown_summary = _drawdown_summary(daily_ledger)
        conflict_summary = _conflict_subset_summary(daily_ledger)
        min_hold_summary = _minimum_hold_cooldown_subset_summary(daily_ledger, trade_ledger)
        blocked_df = _blocked_variants(blocked)
        baseline_alignment = _baseline_alignment(frame, daily_ledger, trade_ledger)

        log("write_outputs", "started", "")
        variant_df.to_csv(output / "variant_parameter_matrix.csv", index=False, encoding="utf-8-sig")
        daily_ledger.to_csv(output / "partial_execution_daily_ledger.csv", index=False, encoding="utf-8-sig")
        trade_ledger.to_csv(output / "partial_execution_trade_ledger.csv", index=False, encoding="utf-8-sig")
        period_perf.to_csv(output / "partial_execution_period_performance.csv", index=False, encoding="utf-8-sig")
        cost_summary.to_csv(output / "partial_execution_cost_turnover_summary.csv", index=False, encoding="utf-8-sig")
        drawdown_summary.to_csv(output / "partial_execution_drawdown_summary.csv", index=False, encoding="utf-8-sig")
        conflict_summary.to_csv(output / "partial_execution_conflict_subset_summary.csv", index=False, encoding="utf-8-sig")
        min_hold_summary.to_csv(output / "minimum_hold_cooldown_subset_summary.csv", index=False, encoding="utf-8-sig")
        blocked_df.to_csv(output / "blocked_variants.csv", index=False, encoding="utf-8-sig")
        (output / "baseline_vs_partial_execution_summary_zh.md").write_text(
            _summary_markdown(baseline_alignment, cost_summary, drawdown_summary),
            encoding="utf-8",
        )
        manifest = {
            "schema_version": 1,
            "task_id": "TASK-BACKTEST-CORE-PARTIAL-EXECUTION-LEDGER-001",
            "model": "partial_execution_ledger_diagnostic_only",
            "status": "completed",
            "formal_daily_path": str(formal_daily_path),
            "price_cache_dir": str(price_cache_dir),
            "initial_cash": initial_cash,
            "formal_model_changed": False,
            "trade_decision_changed": False,
            "active_in_trade_decision": False,
            "execution_diagnostic_active_in_trade_decision": False,
            "not_formal_execution_layer": True,
            "proxy_performance": False,
            "baseline_alignment": baseline_alignment,
            "outputs": {
                "variant_parameter_matrix": "variant_parameter_matrix.csv",
                "partial_execution_daily_ledger": "partial_execution_daily_ledger.csv",
                "partial_execution_trade_ledger": "partial_execution_trade_ledger.csv",
                "partial_execution_period_performance": "partial_execution_period_performance.csv",
                "partial_execution_cost_turnover_summary": "partial_execution_cost_turnover_summary.csv",
                "partial_execution_drawdown_summary": "partial_execution_drawdown_summary.csv",
                "partial_execution_conflict_subset_summary": "partial_execution_conflict_subset_summary.csv",
                "minimum_hold_cooldown_subset_summary": "minimum_hold_cooldown_subset_summary.csv",
                "blocked_variants": "blocked_variants.csv",
                "summary": "baseline_vs_partial_execution_summary_zh.md",
            },
        }
        (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        pd.DataFrame([{"status": "completed", "output_dir": str(output.resolve())}]).to_csv(
            output / "completed.csv",
            index=False,
            encoding="utf-8-sig",
        )
        pd.DataFrame(columns=["step", "error"]).to_csv(output / "failed.csv", index=False, encoding="utf-8-sig")
        log("completed", "completed", str(output.resolve()))
        (output / "current_step.txt").write_text("completed\n", encoding="utf-8")
        return output
    except Exception as exc:
        pd.DataFrame([{"step": "run_partial_execution_ledger", "error": str(exc)}]).to_csv(
            output / "failed.csv",
            index=False,
            encoding="utf-8-sig",
        )
        log("failed", "failed", str(exc))
        raise


def _simulate_variant(
    *,
    frame: pd.DataFrame,
    prices: dict[str, pd.Series],
    event_context: dict[str, dict[str, Any]],
    variant: ExecutionVariant,
    initial_cash: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    account = LedgerAccount(cash=float(initial_cash), positions={})
    cost_model = TaiwanCostModel()
    daily_rows: list[dict[str, Any]] = []
    trade_rows: list[dict[str, Any]] = []
    previous_formal_target = ""
    last_trade_date: pd.Timestamp | None = None
    running_max = float(initial_cash)
    previous_equity = float(initial_cash)

    for index, item in frame.iterrows():
        date = str(item["date"])
        date_ts = pd.Timestamp(item["date_ts"])
        formal_target = str(item.get("formal_target", "")).strip()
        previous_top_holding = _top_holding(account, prices, date)
        target_changed = bool(formal_target and formal_target != previous_formal_target)
        context = event_context.get(date, {})
        is_rapid = bool(context.get("is_rapid_reversal", False))
        is_selector = str(context.get("conflict_source_label", "")) == "formal_selector_conflict"
        equity_before = _account_value(account, prices, date, fallback=previous_equity)
        high_turnover = bool(context.get("is_high_turnover_event", False))
        is_risk_off = _asset_role(formal_target) in {"cash", "none"} or not formal_target
        blocked_reason = ""
        action_type = "hold"

        target_weights = _current_weights(account, prices, date, equity_before)
        if formal_target and target_changed:
            target_weights, blocked_reason, action_type = _target_weights_for_variant(
                account=account,
                prices=prices,
                date=date,
                formal_target=formal_target,
                target_changed=target_changed,
                variant=variant,
                context=context,
                last_trade_date=last_trade_date,
                date_ts=date_ts,
                equity_before=equity_before,
            )
        trades = []
        if formal_target and target_changed and not blocked_reason:
            trades = _rebalance_to_weights(
                account=account,
                prices=prices,
                date=date,
                target_weights=target_weights,
                cost_model=cost_model,
                reason=f"{variant.variant_id}:{action_type}",
            )
        if trades:
            last_trade_date = date_ts
        trade_cost = sum(trade["transaction_cost"] for trade in trades)
        turnover = sum(trade["gross_amount"] for trade in trades)
        trade_rows.extend(
            {
                **trade,
                "variant_id": variant.variant_id,
                "execution_diagnostic_active_in_trade_decision": False,
            }
            for trade in trades
        )
        close_prices = _close_prices_for_positions(account.positions, prices, date)
        equity_after = _account_value(account, prices, date, fallback=previous_equity)
        previous_equity = equity_after
        running_max = max(running_max, equity_after)
        weights = _current_weights(account, prices, date, equity_after)
        old_holding_weight = 0.0 if not previous_top_holding else float(weights.get(previous_top_holding, 0.0))
        target_weight = 0.0 if not formal_target else float(weights.get(formal_target, 0.0))
        daily_return = equity_after / daily_rows[-1]["portfolio_equity_after"] - 1 if daily_rows else 0.0
        daily_rows.append(
            {
                "date": date,
                "period": item.get("period", ""),
                "variant_id": variant.variant_id,
                "baseline_formal_target": formal_target,
                "previous_formal_target": previous_formal_target,
                "current_holding_ticker": previous_top_holding or "cash",
                "new_target_ticker": formal_target or "none",
                "target_weight": round(target_weight, 8),
                "old_holding_weight": round(old_holding_weight, 8),
                "cash_weight": round(account.cash / equity_after, 8) if equity_after else 0.0,
                "weight_sum": round(sum(weights.values()) + (account.cash / equity_after if equity_after else 0.0), 8),
                "target_shares": int(account.positions.get(formal_target, 0)) if formal_target else 0,
                "old_holding_shares": int(account.positions.get(previous_top_holding, 0)) if previous_top_holding else 0,
                "cash_value": round(account.cash, 2),
                "buy_value": round(sum(trade["gross_amount"] for trade in trades if trade["action"] == "buy"), 2),
                "sell_value": round(sum(trade["gross_amount"] for trade in trades if trade["action"] == "sell"), 2),
                "transaction_cost": round(trade_cost, 2),
                "turnover": round(turnover, 2),
                "portfolio_equity_before": round(equity_before, 2),
                "portfolio_equity_after": round(equity_after, 2),
                "daily_return": round(float(daily_return), 8),
                "drawdown": round(equity_after / running_max - 1, 8) if running_max else 0.0,
                "is_target_change": target_changed,
                "is_rapid_reversal": is_rapid,
                "is_formal_selector_conflict": is_selector,
                "is_high_turnover_event": high_turnover,
                "is_risk_off_inferred": is_risk_off,
                "execution_action_type": action_type if trades or blocked_reason else "hold",
                "blocked_reason": blocked_reason,
                "close_price_count": len(close_prices),
                "execution_diagnostic_active_in_trade_decision": False,
            }
        )
        previous_formal_target = formal_target or previous_formal_target
    return pd.DataFrame(daily_rows), pd.DataFrame(trade_rows)


def _target_weights_for_variant(
    *,
    account: LedgerAccount,
    prices: dict[str, pd.Series],
    date: str,
    formal_target: str,
    target_changed: bool,
    variant: ExecutionVariant,
    context: dict[str, Any],
    last_trade_date: pd.Timestamp | None,
    date_ts: pd.Timestamp,
    equity_before: float,
) -> tuple[dict[str, float], str, str]:
    current = _current_weights(account, prices, date, equity_before)
    if not target_changed:
        return current, "", "hold"
    condition = _variant_condition_matches(variant, context)
    days_since_trade = 999 if last_trade_date is None else int((date_ts - last_trade_date).days)
    if variant.minimum_hold_days is not None and condition and days_since_trade < variant.minimum_hold_days:
        return current, f"minimum_hold_{variant.minimum_hold_days}_blocked", "minimum_hold_blocked"
    if variant.cooldown_days is not None and condition and days_since_trade < variant.cooldown_days:
        return current, f"cooldown_{variant.cooldown_days}_blocked", "cooldown_blocked"
    if variant.partial_weight is None or not condition:
        return {formal_target: 1.0}, "", "full_rotation"
    top = _top_holding(account, prices, date)
    if not top or top == formal_target:
        return {formal_target: 1.0}, "", "partial_degenerated_to_full"
    old_weight = max(0.0, 1.0 - float(variant.partial_weight))
    return {formal_target: float(variant.partial_weight), top: old_weight}, "", "partial_switch"


def _variant_condition_matches(variant: ExecutionVariant, context: dict[str, Any]) -> bool:
    if variant.subset == "global":
        return True
    rapid = bool(context.get("is_rapid_reversal", False))
    selector = str(context.get("conflict_source_label", "")) == "formal_selector_conflict"
    high_turnover = bool(context.get("is_high_turnover_event", False))
    if variant.subset == "rapid_reversal_only":
        return rapid
    if variant.subset == "formal_selector_conflict_only":
        return selector
    if variant.subset == "high_turnover_only":
        return high_turnover
    if variant.subset == "rapid_reversal_or_selector_conflict":
        return rapid or selector
    return False


def _rebalance_to_weights(
    *,
    account: LedgerAccount,
    prices: dict[str, pd.Series],
    date: str,
    target_weights: dict[str, float],
    cost_model: TaiwanCostModel,
    reason: str,
) -> list[dict[str, Any]]:
    trades: list[dict[str, Any]] = []
    target_weights = {ticker: max(0.0, float(weight)) for ticker, weight in target_weights.items() if ticker and weight > 0}
    total = sum(target_weights.values())
    if total > 1.0:
        target_weights = {ticker: weight / total for ticker, weight in target_weights.items()}
    prices_today = _close_prices_for_tickers(set(account.positions) | set(target_weights), prices, date)
    equity = account.cash + sum(account.positions.get(ticker, 0) * prices_today.get(ticker, 0.0) for ticker in account.positions)
    if equity <= 0:
        return trades

    # Sell positions above target first to fund buys without margin.
    for ticker, shares in list(account.positions.items()):
        price = prices_today.get(ticker)
        if not price or shares <= 0:
            continue
        desired_value = equity * target_weights.get(ticker, 0.0)
        desired_shares = int(desired_value // price)
        shares_to_sell = max(0, shares - desired_shares)
        if shares_to_sell <= 0:
            continue
        gross = shares_to_sell * price
        costs = cost_model.sell_cost(gross, _asset_type(ticker))
        account.cash += gross - costs
        account.positions[ticker] = shares - shares_to_sell
        trades.append(_trade_row(date, ticker, "sell", shares_to_sell, price, gross, costs, account.cash, reason))

    for ticker, weight in target_weights.items():
        price = prices_today.get(ticker)
        if not price:
            continue
        current_value = account.positions.get(ticker, 0) * price
        desired_value = equity * weight
        gross_budget = max(0.0, desired_value - current_value)
        shares = int(gross_budget // price)
        while shares > 0:
            gross = shares * price
            costs = cost_model.buy_cost(gross)
            if gross + costs <= account.cash:
                break
            shares -= 1
        if shares <= 0:
            continue
        gross = shares * price
        costs = cost_model.buy_cost(gross)
        account.cash -= gross + costs
        account.positions[ticker] = account.positions.get(ticker, 0) + shares
        trades.append(_trade_row(date, ticker, "buy", shares, price, gross, costs, account.cash, reason))
    account.positions = {ticker: shares for ticker, shares in account.positions.items() if shares > 0}
    return trades


def _trade_row(
    date: str,
    ticker: str,
    action: str,
    shares: int,
    price: float,
    gross: float,
    costs: int,
    cash_after: float,
    reason: str,
) -> dict[str, Any]:
    return {
        "date": date,
        "ticker": ticker,
        "action": action,
        "shares": int(shares),
        "price": round(float(price), 4),
        "gross_amount": round(float(gross), 2),
        "transaction_cost": int(costs),
        "cash_after": round(float(cash_after), 2),
        "reason": reason,
    }


def _variant_matrix() -> list[ExecutionVariant]:
    return [
        ExecutionVariant("baseline_full_rotation", "baseline"),
        ExecutionVariant("partial_switch_25_global_diagnostic", "partial_switch", partial_weight=0.25),
        ExecutionVariant("partial_switch_50_global_diagnostic", "partial_switch", partial_weight=0.50),
        ExecutionVariant("partial_switch_75_global_diagnostic", "partial_switch", partial_weight=0.75),
        ExecutionVariant(
            "partial_switch_50_rapid_reversal_only",
            "partial_switch",
            partial_weight=0.50,
            subset="rapid_reversal_only",
        ),
        ExecutionVariant(
            "partial_switch_50_formal_selector_conflict_only",
            "partial_switch",
            partial_weight=0.50,
            subset="formal_selector_conflict_only",
        ),
        ExecutionVariant(
            "partial_switch_50_high_turnover_only",
            "partial_switch",
            partial_weight=0.50,
            subset="high_turnover_only",
        ),
        ExecutionVariant(
            "partial_switch_50_rapid_reversal_or_selector_conflict",
            "partial_switch",
            partial_weight=0.50,
            subset="rapid_reversal_or_selector_conflict",
        ),
        ExecutionVariant(
            "minimum_hold_2_rapid_reversal_only",
            "minimum_hold",
            subset="rapid_reversal_only",
            minimum_hold_days=2,
        ),
        ExecutionVariant(
            "minimum_hold_3_rapid_reversal_only",
            "minimum_hold",
            subset="rapid_reversal_only",
            minimum_hold_days=3,
        ),
        ExecutionVariant(
            "cooldown_2_formal_selector_conflict_only",
            "cooldown",
            subset="formal_selector_conflict_only",
            cooldown_days=2,
        ),
        ExecutionVariant(
            "cooldown_3_formal_selector_conflict_only",
            "cooldown",
            subset="formal_selector_conflict_only",
            cooldown_days=3,
        ),
        ExecutionVariant(
            "sell_first_then_buy_global",
            "sell_first_then_buy",
            blocked=True,
            blocked_reason="global cash wait underperformed new target in 5/20/60d diagnostics",
        ),
        ExecutionVariant(
            "pause_on_conflict",
            "pause_on_conflict",
            blocked=True,
            blocked_reason="final_decision_diagnostic and pool3_selector_veto fields are not in formal daily stream",
        ),
    ]


def _variant_parameter_matrix(variants: list[ExecutionVariant]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "variant_id": variant.variant_id,
                "family": variant.family,
                "partial_weight": variant.partial_weight if variant.partial_weight is not None else "",
                "subset": variant.subset,
                "minimum_hold_days": variant.minimum_hold_days if variant.minimum_hold_days is not None else "",
                "cooldown_days": variant.cooldown_days if variant.cooldown_days is not None else "",
                "status": "blocked" if variant.blocked else "enabled_diagnostic",
                "blocked_reason": variant.blocked_reason,
                "execution_diagnostic_active_in_trade_decision": False,
            }
            for variant in variants
        ]
    )


def _period_performance(daily: pd.DataFrame, prices: dict[str, pd.Series]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    periods = {
        "2022": ("2022-01-01", "2022-12-31"),
        "2023": ("2023-01-01", "2023-12-31"),
        "2024_now": ("2024-01-01", None),
        "2024_hard_gate": ("2024-01-01", "2024-12-31"),
        "full_2022_2026": (None, None),
    }
    for variant_id, group in daily.groupby("variant_id", dropna=False):
        frame = group.copy()
        frame["date_ts"] = pd.to_datetime(frame["date"])
        for period, (start, end) in periods.items():
            subset = frame.copy()
            if start:
                subset = subset[subset["date_ts"] >= pd.Timestamp(start)]
            if end:
                subset = subset[subset["date_ts"] <= pd.Timestamp(end)]
            if subset.empty:
                continue
            start_equity = float(subset["portfolio_equity_after"].iloc[0])
            final_equity = float(subset["portfolio_equity_after"].iloc[-1])
            running_max = subset["portfolio_equity_after"].cummax()
            mdd = float((subset["portfolio_equity_after"] / running_max - 1).min())
            benchmark_0050 = _buy_hold_return(prices.get("0050.TW"), subset["date"].iloc[0], subset["date"].iloc[-1])
            benchmark_00631l = _buy_hold_return(prices.get("00631L.TW"), subset["date"].iloc[0], subset["date"].iloc[-1])
            rows.append(
                {
                    "variant_id": variant_id,
                    "period": period,
                    "start_date": subset["date"].iloc[0],
                    "end_date": subset["date"].iloc[-1],
                    "start_equity": round(start_equity, 2),
                    "final_equity": round(final_equity, 2),
                    "total_return_pct": round((final_equity / start_equity - 1) * 100, 4) if start_equity else "",
                    "max_drawdown_pct": round(mdd * 100, 4),
                    "trade_days": int((pd.to_numeric(subset["turnover"], errors="coerce") > 0).sum()),
                    "total_turnover": round(float(pd.to_numeric(subset["turnover"], errors="coerce").sum()), 2),
                    "total_transaction_cost": round(float(pd.to_numeric(subset["transaction_cost"], errors="coerce").sum()), 2),
                    "benchmark_0050_return_pct": _pct_or_blank(benchmark_0050),
                    "benchmark_00631l_return_pct": _pct_or_blank(benchmark_00631l),
                    "execution_diagnostic_active_in_trade_decision": False,
                }
            )
    baseline = pd.DataFrame(rows)
    full_baseline = baseline[baseline["variant_id"].eq("baseline_full_rotation")][
        ["period", "final_equity", "total_return_pct", "max_drawdown_pct"]
    ].rename(
        columns={
            "final_equity": "baseline_final_equity",
            "total_return_pct": "baseline_total_return_pct",
            "max_drawdown_pct": "baseline_max_drawdown_pct",
        }
    )
    if not baseline.empty and not full_baseline.empty:
        baseline = baseline.merge(full_baseline, on="period", how="left")
        baseline["return_delta_vs_baseline_pp"] = pd.to_numeric(baseline["total_return_pct"], errors="coerce") - pd.to_numeric(
            baseline["baseline_total_return_pct"], errors="coerce"
        )
        baseline["mdd_delta_vs_baseline_pp"] = pd.to_numeric(baseline["max_drawdown_pct"], errors="coerce") - pd.to_numeric(
            baseline["baseline_max_drawdown_pct"], errors="coerce"
        )
    return baseline


def _cost_turnover_summary(daily: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant_id, group in daily.groupby("variant_id", dropna=False):
        variant_trades = trades[trades["variant_id"].eq(variant_id)] if not trades.empty else pd.DataFrame()
        rows.append(
            {
                "variant_id": variant_id,
                "trade_count": int(len(variant_trades)),
                "trade_days": int((pd.to_numeric(group["turnover"], errors="coerce") > 0).sum()),
                "total_turnover": round(float(pd.to_numeric(group["turnover"], errors="coerce").sum()), 2),
                "total_transaction_cost": round(float(pd.to_numeric(group["transaction_cost"], errors="coerce").sum()), 2),
                "rapid_reversal_cost": round(
                    float(pd.to_numeric(group.loc[group["is_rapid_reversal"].astype(bool), "transaction_cost"], errors="coerce").sum()),
                    2,
                ),
                "underexposure_days": int((pd.to_numeric(group["cash_weight"], errors="coerce") > 0.05).sum()),
                "cash_drag_proxy": round(float(pd.to_numeric(group["cash_weight"], errors="coerce").mean()), 8),
                "execution_diagnostic_active_in_trade_decision": False,
            }
        )
    return pd.DataFrame(rows)


def _drawdown_summary(daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant_id, group in daily.groupby("variant_id", dropna=False):
        drawdown = pd.to_numeric(group["drawdown"], errors="coerce")
        rows.append(
            {
                "variant_id": variant_id,
                "max_drawdown_pct": round(float(drawdown.min()) * 100, 4) if drawdown.notna().any() else "",
                "drawdown_days_below_20pct": int((drawdown <= -0.2).sum()),
                "drawdown_days_below_40pct": int((drawdown <= -0.4).sum()),
                "execution_diagnostic_active_in_trade_decision": False,
            }
        )
    return pd.DataFrame(rows)


def _conflict_subset_summary(daily: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    flags = {
        "rapid_reversal": "is_rapid_reversal",
        "formal_selector_conflict": "is_formal_selector_conflict",
        "high_turnover": "is_high_turnover_event",
    }
    for variant_id, group in daily.groupby("variant_id", dropna=False):
        for subset_name, column in flags.items():
            subset = group[group[column].astype(bool)]
            rows.append(
                {
                    "variant_id": variant_id,
                    "subset": subset_name,
                    "event_rows": int(len(subset)),
                    "trade_rows": int((pd.to_numeric(subset.get("turnover", pd.Series(dtype=float)), errors="coerce") > 0).sum()),
                    "transaction_cost": round(float(pd.to_numeric(subset.get("transaction_cost", pd.Series(dtype=float)), errors="coerce").sum()), 2),
                    "turnover": round(float(pd.to_numeric(subset.get("turnover", pd.Series(dtype=float)), errors="coerce").sum()), 2),
                    "execution_diagnostic_active_in_trade_decision": False,
                }
            )
    return pd.DataFrame(rows)


def _minimum_hold_cooldown_subset_summary(daily: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    target_variants = daily[daily["variant_id"].astype(str).str.startswith(("minimum_hold_", "cooldown_"))]
    if target_variants.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for variant_id, group in target_variants.groupby("variant_id", dropna=False):
        blocked = group[group["blocked_reason"].astype(str).str.strip().ne("")]
        rows.append(
            {
                "variant_id": variant_id,
                "blocked_change_days": int(len(blocked)),
                "trade_count": int(len(trades[trades["variant_id"].eq(variant_id)])) if not trades.empty else 0,
                "final_equity": round(float(group["portfolio_equity_after"].iloc[-1]), 2),
                "max_drawdown_pct": round(float(pd.to_numeric(group["drawdown"], errors="coerce").min()) * 100, 4),
                "execution_diagnostic_active_in_trade_decision": False,
            }
        )
    return pd.DataFrame(rows)


def _blocked_variants(variants: list[ExecutionVariant]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "variant_id": variant.variant_id,
                "family": variant.family,
                "blocked_reason": variant.blocked_reason,
                "execution_diagnostic_active_in_trade_decision": False,
            }
            for variant in variants
        ]
    )


def _baseline_alignment(frame: pd.DataFrame, daily: pd.DataFrame, trades: pd.DataFrame) -> dict[str, Any]:
    baseline = daily[daily["variant_id"].eq("baseline_full_rotation")].copy()
    baseline_trades = trades[trades["variant_id"].eq("baseline_full_rotation")].copy() if not trades.empty else pd.DataFrame()
    if baseline.empty:
        return {"status": "missing_baseline"}
    formal_final = float(pd.to_numeric(frame["equity"], errors="coerce").iloc[-1])
    simulated_final = float(baseline["portfolio_equity_after"].iloc[-1])
    formal_mdd = float(pd.to_numeric(frame["drawdown"], errors="coerce").min())
    simulated_mdd = float(pd.to_numeric(baseline["drawdown"], errors="coerce").min())
    formal_trade_days = int(frame["action"].astype(str).isin(["buy", "switch"]).sum())
    simulated_trade_days = int((pd.to_numeric(baseline["turnover"], errors="coerce") > 0).sum())
    return {
        "status": "completed",
        "formal_final_equity": round(formal_final, 2),
        "simulated_final_equity": round(simulated_final, 2),
        "final_equity_diff": round(simulated_final - formal_final, 2),
        "formal_mdd": round(formal_mdd, 8),
        "simulated_mdd": round(simulated_mdd, 8),
        "mdd_diff": round(simulated_mdd - formal_mdd, 8),
        "formal_trade_days": formal_trade_days,
        "simulated_trade_days": simulated_trade_days,
        "simulated_trade_count": int(len(baseline_trades)),
        "alignment_note": "baseline is replayed from formal target stream with integer shares and TaiwanCostModel",
    }


def _build_event_context(frame: pd.DataFrame, event_study: pd.DataFrame) -> dict[str, dict[str, Any]]:
    context = {
        str(row["date"]): {
            "is_rapid_reversal": bool(row.get("reversal_within_3_trading_rows", False)),
            "conflict_source_label": str(row.get("conflict_source_label", "")),
        }
        for row in event_study.to_dict(orient="records")
    }
    turnover = pd.to_numeric(frame.get("turnover", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    threshold = float(turnover[turnover > 0].quantile(0.75)) if (turnover > 0).any() else 0.0
    for row in frame.to_dict(orient="records"):
        date = str(row["date"])
        context.setdefault(date, {})
        context[date]["is_high_turnover_event"] = bool(threshold and float(pd.to_numeric(pd.Series([row.get("turnover")]), errors="coerce").fillna(0).iloc[0]) >= threshold)
    return context


def _load_prices(frame: pd.DataFrame, cache_dir: Path) -> dict[str, pd.Series]:
    tickers = sorted(
        {
            str(value).strip()
            for column in ("winner_ticker", "position_ticker", "pool1_vote", "pool2_vote", "pool3_vote")
            if column in frame.columns
            for value in frame[column].tolist()
            if str(value).strip() and str(value).strip() not in {"cash", "none"}
        }
        | {"0050.TW", "00631L.TW"}
    )
    prices: dict[str, pd.Series] = {}
    for ticker in tickers:
        path = cache_dir / f"{ticker.replace('.', '_')}.csv"
        if not path.exists():
            continue
        price = load_price_csv(path)
        prices[ticker] = pd.to_numeric(price["adj_close"], errors="coerce").dropna()
    return prices


def _normalize_formal_daily(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy().fillna("")
    normalized["date"] = normalized["date"].astype(str)
    normalized["date_ts"] = pd.to_datetime(normalized["date"], errors="coerce")
    normalized = normalized[normalized["date_ts"].notna()].sort_values("date_ts").reset_index(drop=True)
    normalized["formal_target"] = normalized["winner_ticker"].astype(str).str.strip()
    return normalized


def _validate_formal_daily(frame: pd.DataFrame) -> None:
    required = {"date", "winner_ticker", "position_ticker", "equity", "drawdown", "action", "turnover", "transaction_cost"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError("missing formal daily columns: " + ",".join(sorted(missing)))


def _current_weights(account: LedgerAccount, prices: dict[str, pd.Series], date: str, equity: float) -> dict[str, float]:
    if equity <= 0:
        return {}
    close_prices = _close_prices_for_positions(account.positions, prices, date)
    return {
        ticker: shares * close_prices[ticker] / equity
        for ticker, shares in account.positions.items()
        if shares > 0 and ticker in close_prices
    }


def _top_holding(account: LedgerAccount, prices: dict[str, pd.Series], date: str) -> str:
    close_prices = _close_prices_for_positions(account.positions, prices, date)
    values = {
        ticker: shares * close_prices[ticker]
        for ticker, shares in account.positions.items()
        if shares > 0 and ticker in close_prices
    }
    if not values:
        return ""
    return max(values, key=values.get)


def _account_value(account: LedgerAccount, prices: dict[str, pd.Series], date: str, *, fallback: float) -> float:
    close_prices = _close_prices_for_positions(account.positions, prices, date)
    if not close_prices and account.positions:
        return fallback
    return float(account.cash + sum(shares * close_prices.get(ticker, 0.0) for ticker, shares in account.positions.items()))


def _close_prices_for_positions(positions: dict[str, int], prices: dict[str, pd.Series], date: str) -> dict[str, float]:
    return _close_prices_for_tickers(set(positions), prices, date)


def _close_prices_for_tickers(tickers: set[str], prices: dict[str, pd.Series], date: str) -> dict[str, float]:
    close: dict[str, float] = {}
    for ticker in tickers:
        price = _price_on_or_before(prices.get(ticker, pd.Series(dtype=float)), date)
        if price is not None:
            close[ticker] = price
    return close


def _price_on_or_before(series: pd.Series, date: str) -> float | None:
    if series.empty:
        return None
    clipped = series.loc[series.index <= pd.Timestamp(date)]
    if clipped.empty:
        return None
    return float(clipped.iloc[-1])


def _buy_hold_return(series: pd.Series | None, start: str, end: str) -> float | None:
    if series is None or series.empty:
        return None
    start_price = _price_on_or_before(series, start)
    end_price = _price_on_or_before(series, end)
    if not start_price or not end_price:
        return None
    return end_price / start_price - 1


def _pct_or_blank(value: float | None) -> float | str:
    return "" if value is None else round(value * 100, 4)


def _asset_type(ticker: str) -> str:
    return "etf" if ticker in MARKET_EXPOSURE_TICKERS else "stock"


def _asset_role(ticker: str) -> str:
    if not ticker:
        return "none"
    if ticker == "cash":
        return "cash"
    if ticker == "00631L.TW":
        return "leveraged_market_exposure"
    if ticker == "0050.TW":
        return "market_exposure"
    return "stock_attack"


def _summary_markdown(baseline_alignment: dict[str, Any], cost_summary: pd.DataFrame, drawdown_summary: pd.DataFrame) -> str:
    top_cost = cost_summary.sort_values("total_transaction_cost").head(3) if not cost_summary.empty else pd.DataFrame()
    lines = [
        "# Partial Execution Ledger / Execution A-B Diagnostic",
        "",
        "本輸出是 execution layer 的 diagnostic ledger，不是正式 execution / exit layer。",
        "",
        "## 邊界",
        "",
        "- formal_model_changed=false",
        "- trade_decision_changed=false",
        "- active_in_trade_decision=false",
        "- execution_diagnostic_active_in_trade_decision=false",
        "- 不改 formal selector / formal vote / formal target / formal trade action",
        "",
        "## Baseline 對齊",
        "",
        f"- formal final equity：{baseline_alignment.get('formal_final_equity')}",
        f"- ledger baseline final equity：{baseline_alignment.get('simulated_final_equity')}",
        f"- diff：{baseline_alignment.get('final_equity_diff')}",
        f"- formal MDD：{baseline_alignment.get('formal_mdd')}",
        f"- ledger MDD：{baseline_alignment.get('simulated_mdd')}",
        "",
        "## 成本較低的 diagnostic variants",
        "",
    ]
    for row in top_cost.to_dict(orient="records"):
        lines.append(
            f"- {row.get('variant_id')}：成本 {row.get('total_transaction_cost')}，turnover {row.get('total_turnover')}"
        )
    if not drawdown_summary.empty:
        best_mdd = drawdown_summary.sort_values("max_drawdown_pct", ascending=False).head(1).iloc[0]
        lines.extend(["", f"- MDD 較淺 variant：{best_mdd.get('variant_id')}，MDD {best_mdd.get('max_drawdown_pct')}%"])
    lines.extend(["", "## 下一步", "", "交由 Experiments 驗證各 variant 是否有跨期間、非 proxy 的診斷價值。"])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build partial execution ledger diagnostic runner outputs.")
    parser.add_argument("--formal-daily", required=True)
    parser.add_argument("--price-cache-dir", default=DEFAULT_PRICE_CACHE_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--initial-cash", type=float, default=DEFAULT_INITIAL_CASH)
    args = parser.parse_args()
    output = run_partial_execution_ledger(
        formal_daily_path=args.formal_daily,
        price_cache_dir=args.price_cache_dir,
        output_dir=args.output_dir,
        initial_cash=args.initial_cash,
    )
    print(f"OUTPUT_DIR={output.resolve()}")


if __name__ == "__main__":
    main()
