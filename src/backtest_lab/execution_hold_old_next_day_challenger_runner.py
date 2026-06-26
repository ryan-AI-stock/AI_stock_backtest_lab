from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from backtest_lab.execution_layer_next_day_ab_pool1_pool2_formal import (
    Account,
    FORMAL_MODEL_TARGET,
    INITIAL_CASH,
    _alignment_max_diff,
    _asset_type,
    _close_prices,
    _cost_turnover_summary,
    _current_weights,
    _drawdown_summary,
    _execute_order,
    _load_prices,
    _normalize_stream,
    _parse_weights,
    _period_performance,
    _top_holding,
    _validate_stream,
)


DEFAULT_REVIEW_DIR = "outputs/execution_layer_review_pool1_pool2_formal_20260626"
DEFAULT_PRICE_CACHE_DIR = "backtest_cache/stock_pool_triad_v1_corrected"
DEFAULT_OUTPUT_DIR = "outputs/execution_hold_old_next_day_challenger_20260626"


@dataclass(frozen=True)
class HoldOldVariant:
    variant_id: str
    fill_delay_days: int
    hold_old_rule: bool = False
    description: str = ""


VARIANTS = (
    HoldOldVariant(
        "selector_full_switch_same_day_reference",
        0,
        False,
        "同日 full switch alignment reference；不得作正式 execution performance。",
    ),
    HoldOldVariant(
        "selector_full_switch_next_day_baseline",
        1,
        False,
        "正式 target stream 的 next-day full rotation baseline。",
    ),
    HoldOldVariant(
        "hold_old_if_still_valid_ma20_next_day",
        1,
        True,
        "target change 時，舊持股仍在 20 日均線上且 20 日報酬非負時先保留舊持股；否則 next-day 切換。",
    ),
)


def run_execution_hold_old_next_day_challenger_runner(
    *,
    review_dir: str | Path = DEFAULT_REVIEW_DIR,
    price_cache_dir: str | Path = DEFAULT_PRICE_CACHE_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    initial_cash: float = INITIAL_CASH,
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
        review = Path(review_dir)
        log("load_formal_target_stream", "started", str(review))
        stream = pd.read_csv(review / "formal_target_stream_adapter.csv").fillna("")
        _validate_stream(stream)
        frame = _normalize_stream(stream)

        log("load_price_cache", "started", str(price_cache_dir))
        prices = _load_prices(frame, Path(price_cache_dir))
        if not prices:
            raise ValueError("no prices loaded for hold-old challenger")

        daily_frames: list[pd.DataFrame] = []
        weight_frames: list[pd.DataFrame] = []
        trade_frames: list[pd.DataFrame] = []
        cash_frames: list[pd.DataFrame] = []
        blocked_frames: list[pd.DataFrame] = []
        hold_event_frames: list[pd.DataFrame] = []
        reason_trace_frames: list[pd.DataFrame] = []

        log("simulate_variants", "started", f"variants={len(VARIANTS)}")
        for variant in VARIANTS:
            daily, weights, trades, cash, blocked, hold_events, reason_trace = _simulate_variant(frame, prices, variant, initial_cash)
            daily_frames.append(daily)
            weight_frames.append(weights)
            trade_frames.append(trades)
            cash_frames.append(cash)
            blocked_frames.append(blocked)
            hold_event_frames.append(hold_events)
            reason_trace_frames.append(reason_trace)

        daily = pd.concat(daily_frames, ignore_index=True)
        weights = pd.concat(weight_frames, ignore_index=True)
        trades = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()
        cash = pd.concat(cash_frames, ignore_index=True)
        blocked = pd.concat(blocked_frames, ignore_index=True) if blocked_frames else pd.DataFrame()
        hold_events = pd.concat(hold_event_frames, ignore_index=True) if hold_event_frames else pd.DataFrame()
        reason_trace = pd.concat(reason_trace_frames, ignore_index=True)

        log("build_reports", "started", "")
        variant_matrix = _variant_matrix()
        performance = _period_performance(daily)
        alignment = _same_day_vs_next_day_alignment(frame, daily)
        quality = _execution_quality_scorecard(daily, trades, blocked, hold_events)
        stability = _target_stability(reason_trace)
        concentration = _contribution_concentration(daily, weights)
        leave_one = _leave_one_period_report(daily)
        hard_gate = _hard_gate_caveat(performance)
        exposure = _exposure_integrity(daily)
        cost_turnover = _cost_turnover_summary(daily, trades)
        drawdown = _drawdown_summary(daily)

        log("write_outputs", "started", "")
        variant_matrix.to_csv(output / "variant_parameter_matrix.csv", index=False, encoding="utf-8-sig")
        daily.to_csv(output / "daily_equity_by_variant.csv", index=False, encoding="utf-8-sig")
        weights.to_csv(output / "portfolio_weight_ledger_by_variant.csv", index=False, encoding="utf-8-sig")
        trades.to_csv(output / "trade_ledger_by_variant.csv", index=False, encoding="utf-8-sig")
        cash.to_csv(output / "cash_ledger_by_variant.csv", index=False, encoding="utf-8-sig")
        blocked.to_csv(output / "blocked_fill_events.csv", index=False, encoding="utf-8-sig")
        hold_events.to_csv(output / "hold_old_event_panel.csv", index=False, encoding="utf-8-sig")
        reason_trace.to_csv(output / "old_holding_still_valid_reason_trace.csv", index=False, encoding="utf-8-sig")
        performance.to_csv(output / "period_performance_by_variant.csv", index=False, encoding="utf-8-sig")
        alignment.to_csv(output / "same_day_vs_next_day_alignment.csv", index=False, encoding="utf-8-sig")
        quality.to_csv(output / "execution_quality_scorecard.csv", index=False, encoding="utf-8-sig")
        stability.to_csv(output / "target_stability_by_variant.csv", index=False, encoding="utf-8-sig")
        concentration.to_csv(output / "contribution_concentration_by_variant.csv", index=False, encoding="utf-8-sig")
        leave_one.to_csv(output / "leave_one_period_by_variant.csv", index=False, encoding="utf-8-sig")
        hard_gate.to_csv(output / "hard_gate_2024_benchmark_caveat.csv", index=False, encoding="utf-8-sig")
        exposure.to_csv(output / "exposure_integrity_checks.csv", index=False, encoding="utf-8-sig")
        cost_turnover.to_csv(output / "cost_turnover_summary.csv", index=False, encoding="utf-8-sig")
        drawdown.to_csv(output / "drawdown_report.csv", index=False, encoding="utf-8-sig")
        (output / "execution_hold_old_next_day_challenger_summary_zh.md").write_text(
            _summary_markdown(performance, quality, hard_gate),
            encoding="utf-8",
        )

        manifest = {
            "schema_version": 1,
            "task_id": "TASK-BACKTEST-CORE-EXECUTION-HOLD-OLD-NEXT-DAY-CHALLENGER-001",
            "status": "completed_challenger_evidence",
            "formal_model_target": FORMAL_MODEL_TARGET,
            "formal_model_route": "pool1_primary_pool2_confirmation_cap40",
            "review_dir": str(review),
            "price_cache_dir": str(price_cache_dir),
            "start_date": str(frame["date"].iloc[0]) if not frame.empty else "",
            "latest_complete_common_date": str(frame["date"].iloc[-1]) if not frame.empty else "",
            "formal_model_changed": False,
            "trade_decision_changed": False,
            "formal_execution_layer_activated": False,
            "execution_layer_status": "challenger",
            "active_in_trade_decision": False,
            "execution_diagnostic_active_in_trade_decision": False,
            "uses_forward_return_as_rule": False,
            "pool3_shadow_used": False,
            "final_decision_label_used": False,
            "rr_partial_switch_used": False,
            "valuation_used": False,
            "h3_used": False,
            "production_grade_next_day_ledger": True,
            "simplified_experiments_ledger_used_for_formal_performance": False,
            "same_day_reference_used_for_formal_execution_performance": False,
            "same_day_and_next_day_mixed": False,
            "same_day_alignment_max_abs_diff": _alignment_max_diff(alignment),
            "hard_gate_2024_caveat_retained": True,
            "outputs": {
                "variant_parameter_matrix": "variant_parameter_matrix.csv",
                "daily_equity": "daily_equity_by_variant.csv",
                "portfolio_weight_ledger": "portfolio_weight_ledger_by_variant.csv",
                "trade_ledger": "trade_ledger_by_variant.csv",
                "cash_ledger": "cash_ledger_by_variant.csv",
                "blocked_fill_events": "blocked_fill_events.csv",
                "hold_old_event_panel": "hold_old_event_panel.csv",
                "reason_trace": "old_holding_still_valid_reason_trace.csv",
                "period_performance": "period_performance_by_variant.csv",
                "alignment": "same_day_vs_next_day_alignment.csv",
                "quality": "execution_quality_scorecard.csv",
                "concentration": "contribution_concentration_by_variant.csv",
                "summary": "execution_hold_old_next_day_challenger_summary_zh.md",
            },
        }
        (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        pd.DataFrame([{"status": "completed", "output_dir": str(output.resolve())}]).to_csv(output / "completed.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(columns=["step", "error"]).to_csv(output / "failed.csv", index=False, encoding="utf-8-sig")
        log("completed", "completed", str(output.resolve()))
        (output / "current_step.txt").write_text("completed\n", encoding="utf-8")
        return output
    except Exception as exc:
        pd.DataFrame([{"step": "run_execution_hold_old_next_day_challenger_runner", "error": str(exc)}]).to_csv(output / "failed.csv", index=False, encoding="utf-8-sig")
        log("failed", "failed", str(exc))
        raise


def _simulate_variant(
    frame: pd.DataFrame,
    prices: dict[str, pd.Series],
    variant: HoldOldVariant,
    initial_cash: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    account = Account(cash=float(initial_cash), positions={})
    pending: list[dict[str, Any]] = []
    accepted_target: dict[str, float] = {}
    last_formal_target_key = ""
    running_max = float(initial_cash)
    previous_equity = float(initial_cash)
    daily_rows: list[dict[str, Any]] = []
    weight_rows: list[dict[str, Any]] = []
    trade_rows: list[dict[str, Any]] = []
    cash_rows: list[dict[str, Any]] = []
    blocked_rows: list[dict[str, Any]] = []
    hold_event_rows: list[dict[str, Any]] = []
    reason_trace_rows: list[dict[str, Any]] = []

    for index, item in frame.iterrows():
        date = str(item["date"])
        due = [order for order in pending if int(order["fill_index"]) == index]
        pending = [order for order in pending if int(order["fill_index"]) != index]
        for order in due:
            trades, blocked_reason = _execute_order(account, prices, date, order["target_weights"], _cost_model(), order["reason"])
            if blocked_reason:
                blocked_rows.append(_blocked_row(variant, order["signal_date"], date, order["target_weights"], blocked_reason))
            else:
                accepted_target = dict(order["target_weights"])
                trade_rows.extend({**trade, "variant_id": variant.variant_id, "signal_date": order["signal_date"]} for trade in trades)

        signal_weights = _parse_weights(item.get("target_weights"))
        signal_key = _target_key(signal_weights)
        current_weights = _account_weights(account, prices, date, previous_equity)
        current_top = _top_holding(account, prices, date) or ""
        formal_target_changed = signal_key != last_formal_target_key
        policy_weights = signal_weights
        policy_state = "pass_through"
        policy_reason = ""
        old_valid = False
        if variant.hold_old_rule and formal_target_changed and signal_weights and current_top:
            old_valid, valid_reason = _old_holding_valid(current_top, prices, date)
            if old_valid:
                policy_weights = accepted_target if accepted_target else current_weights
                policy_state = "hold_old_still_valid"
                policy_reason = valid_reason
            else:
                policy_reason = valid_reason
        elif variant.hold_old_rule and formal_target_changed and not signal_weights:
            policy_state = "forced_exit_allowed"
            policy_reason = "formal target disappeared or risk-off/no-target"

        if variant.hold_old_rule:
            reason_trace_rows.append(
                {
                    "variant_id": variant.variant_id,
                    "date": date,
                    "formal_target": _target_key(signal_weights) or "cash",
                    "old_holding": current_top or "cash",
                    "old_holding_valid": old_valid,
                    "policy_state": policy_state,
                    "policy_reason": policy_reason,
                    "active_in_trade_decision": False,
                }
            )

        target_changed = policy_weights != accepted_target
        pending_same_target = pending and policy_weights == dict(pending[-1]["target_weights"])
        source_trade_signal = str(item.get("action", "")).strip() in {"buy", "switch", "sell"} or _number(item.get("turnover")) > 0
        should_schedule = ((target_changed and not pending_same_target) or source_trade_signal) and policy_state != "hold_old_still_valid"
        if should_schedule:
            fill_index = index + variant.fill_delay_days
            if fill_index >= len(frame):
                blocked_rows.append(_blocked_row(variant, date, "", policy_weights, "missing_future_fill_row"))
            elif variant.fill_delay_days == 0:
                trades, blocked_reason = _execute_order(account, prices, date, policy_weights, _cost_model(), "same_day_fill")
                if blocked_reason:
                    blocked_rows.append(_blocked_row(variant, date, date, policy_weights, blocked_reason))
                else:
                    accepted_target = dict(policy_weights)
                    trade_rows.extend({**trade, "variant_id": variant.variant_id, "signal_date": date} for trade in trades)
            else:
                pending = [{"fill_index": fill_index, "signal_date": date, "target_weights": dict(policy_weights), "reason": "next_day_fill"}]

        if variant.hold_old_rule and policy_state == "hold_old_still_valid":
            hold_event_rows.append(
                {
                    "variant_id": variant.variant_id,
                    "signal_date": date,
                    "held_old_ticker": current_top,
                    "formal_target": _target_key(signal_weights) or "cash",
                    "policy_target": _target_key(policy_weights) or "cash",
                    "reason": policy_reason,
                    "active_in_trade_decision": False,
                }
            )
        last_formal_target_key = signal_key

        prices_today = _close_prices(set(account.positions), prices, date)
        equity = account.cash + sum(account.positions.get(ticker, 0) * prices_today.get(ticker, 0.0) for ticker in account.positions)
        if equity <= 0:
            equity = previous_equity
        running_max = max(running_max, equity)
        daily_return = equity / previous_equity - 1 if previous_equity else 0.0
        previous_equity = equity
        weights = _current_weights(account, prices, date, equity)
        weight_sum = sum(weights.values())
        cash_weight = account.cash / equity if equity else 0.0
        daily_rows.append(
            {
                "variant_id": variant.variant_id,
                "date": date,
                "period": item.get("period", ""),
                "formal_target_weights": json.dumps(signal_weights, sort_keys=True),
                "execution_target_weights": json.dumps(accepted_target, sort_keys=True),
                "policy_target_weights": json.dumps(policy_weights, sort_keys=True),
                "policy_state": policy_state,
                "policy_reason": policy_reason,
                "pending_order_count": len(pending),
                "top_holding": _top_holding(account, prices, date) or "cash",
                "cash": round(account.cash, 2),
                "cash_weight": round(cash_weight, 8),
                "position_weight_sum": round(weight_sum, 8),
                "weight_sum": round(weight_sum + cash_weight, 8),
                "portfolio_equity": round(equity, 2),
                "daily_return": round(float(daily_return), 8),
                "drawdown": round(equity / running_max - 1, 8) if running_max else 0.0,
                "execution_diagnostic_active_in_trade_decision": False,
            }
        )
        cash_rows.append(
            {
                "variant_id": variant.variant_id,
                "date": date,
                "cash": round(account.cash, 2),
                "cash_weight": round(cash_weight, 8),
                "cash_non_negative": account.cash >= -0.01,
                "active_in_trade_decision": False,
            }
        )
        for ticker, weight in weights.items():
            weight_rows.append(
                {
                    "variant_id": variant.variant_id,
                    "date": date,
                    "ticker": ticker,
                    "weight": round(weight, 8),
                    "shares": int(account.positions.get(ticker, 0)),
                    "asset_type": _asset_type(ticker),
                    "active_in_trade_decision": False,
                }
            )
        if not weights:
            weight_rows.append({"variant_id": variant.variant_id, "date": date, "ticker": "cash", "weight": round(cash_weight, 8), "shares": 0, "asset_type": "cash", "active_in_trade_decision": False})

    return (
        pd.DataFrame(daily_rows),
        pd.DataFrame(weight_rows),
        pd.DataFrame(trade_rows),
        pd.DataFrame(cash_rows),
        pd.DataFrame(blocked_rows),
        pd.DataFrame(hold_event_rows),
        pd.DataFrame(reason_trace_rows),
    )


def _account_weights(account: Account, prices: dict[str, pd.Series], date: str, equity: float) -> dict[str, float]:
    if equity <= 0:
        return {}
    return _current_weights(account, prices, date, equity)


def _old_holding_valid(ticker: str, prices: dict[str, pd.Series], date: str) -> tuple[bool, str]:
    series = prices.get(ticker)
    if series is None or series.empty:
        return False, "missing_old_holding_price"
    valid = series.loc[series.index <= pd.Timestamp(date)]
    if len(valid) < 20:
        return False, "old_holding_less_than_20_price_rows"
    close = float(valid.iloc[-1])
    ma20 = float(valid.tail(20).mean())
    base = float(valid.iloc[-20])
    ret20 = close / base - 1 if base else -1.0
    if close >= ma20 and ret20 >= 0:
        return True, f"close_above_ma20_and_ret20_non_negative;ret20={ret20:.6f}"
    return False, f"old_holding_invalid;close={close:.4f};ma20={ma20:.4f};ret20={ret20:.6f}"


def _target_key(weights: dict[str, float]) -> str:
    if not weights:
        return ""
    return "|".join(f"{ticker}:{weight:.6f}" for ticker, weight in sorted(weights.items()))


def _blocked_row(variant: HoldOldVariant, signal_date: str, fill_date: str, weights: dict[str, float], reason: str) -> dict[str, Any]:
    return {
        "variant_id": variant.variant_id,
        "signal_date": signal_date,
        "intended_fill_date": fill_date,
        "target_weights": json.dumps(weights, sort_keys=True),
        "blocked_reason": reason,
        "diagnostic_only": True,
        "active_in_trade_decision": False,
    }


def _cost_model():
    from backtest_lab.costs import TaiwanCostModel

    return TaiwanCostModel()


def _number(value: object) -> float:
    return float(pd.to_numeric(pd.Series([value]), errors="coerce").fillna(0.0).iloc[0])


def _variant_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "variant_id": variant.variant_id,
                "fill_delay_days": variant.fill_delay_days,
                "hold_old_rule": variant.hold_old_rule,
                "description": variant.description,
                "formal_model_changed": False,
                "trade_decision_changed": False,
                "active_in_trade_decision": False,
            }
            for variant in VARIANTS
        ]
    )


def _same_day_vs_next_day_alignment(stream: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if "equity" in stream.columns:
        reference = daily[daily["variant_id"] == "selector_full_switch_same_day_reference"][["date", "portfolio_equity"]]
        merged = stream[["date", "equity"]].merge(reference, on="date", how="inner")
        if not merged.empty:
            diff = pd.to_numeric(merged["portfolio_equity"], errors="coerce") - pd.to_numeric(merged["equity"], errors="coerce")
            rows.append(
                {
                    "variant_id": "selector_full_switch_same_day_reference",
                    "alignment_state": "passed" if float(diff.abs().max()) <= 0.01 else "diff_detected",
                    "max_abs_diff": round(float(diff.abs().max()), 6),
                    "final_equity_diff": round(float(diff.iloc[-1]), 6),
                    "row_count": int(len(merged)),
                    "same_day_reference_not_formal_execution": True,
                    "active_in_trade_decision": False,
                }
            )
    baseline = daily[daily["variant_id"] == "selector_full_switch_next_day_baseline"]
    rows.append(
        {
            "variant_id": "selector_full_switch_next_day_baseline",
            "alignment_state": "next_day_baseline_completed" if not baseline.empty else "blocked_no_rows",
            "max_abs_diff": "",
            "final_equity_diff": "",
            "row_count": int(len(baseline)),
            "same_day_reference_not_formal_execution": False,
            "active_in_trade_decision": False,
        }
    )
    return pd.DataFrame(rows)


def _execution_quality_scorecard(daily: pd.DataFrame, trades: pd.DataFrame, blocked: pd.DataFrame, hold_events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant, group in daily.groupby("variant_id"):
        trade_group = trades[trades["variant_id"].eq(variant)] if not trades.empty else pd.DataFrame()
        blocked_group = blocked[blocked["variant_id"].eq(variant)] if not blocked.empty else pd.DataFrame()
        hold_group = hold_events[hold_events["variant_id"].eq(variant)] if not hold_events.empty else pd.DataFrame()
        rows.append(
            {
                "variant_id": variant,
                "final_equity": round(float(group["portfolio_equity"].iloc[-1]), 2),
                "max_drawdown_pct": round(float(pd.to_numeric(group["drawdown"], errors="coerce").min()) * 100, 4),
                "trade_rows": int(len(trade_group)),
                "blocked_fill_count": int(len(blocked_group)),
                "hold_old_event_count": int(len(hold_group)),
                "average_cash_weight": round(float(pd.to_numeric(group["cash_weight"], errors="coerce").mean()), 6),
                "ready_for_formal_activation": False,
                "active_in_trade_decision": False,
            }
        )
    return pd.DataFrame(rows)


def _target_stability(reason_trace: pd.DataFrame) -> pd.DataFrame:
    if reason_trace.empty:
        return pd.DataFrame(columns=["variant_id", "hold_old_event_count", "old_holding_valid_rate", "active_in_trade_decision"])
    rows = []
    for variant, group in reason_trace.groupby("variant_id"):
        rows.append(
            {
                "variant_id": variant,
                "hold_old_event_count": int(group["policy_state"].eq("hold_old_still_valid").sum()),
                "old_holding_valid_rate": round(float(group["old_holding_valid"].astype(bool).mean()), 6),
                "active_in_trade_decision": False,
            }
        )
    return pd.DataFrame(rows)


def _contribution_concentration(daily: pd.DataFrame, weights: pd.DataFrame) -> pd.DataFrame:
    rows = []
    full = daily.sort_values("date")
    for variant, group in full.groupby("variant_id"):
        weight_group = weights[(weights["variant_id"].eq(variant)) & (~weights["ticker"].eq("cash"))]
        ticker_share = weight_group.groupby("ticker")["weight"].sum() if not weight_group.empty else pd.Series(dtype=float)
        month = group.assign(month=pd.to_datetime(group["date"]).dt.to_period("M").astype(str))
        quarter = group.assign(quarter=pd.to_datetime(group["date"]).dt.to_period("Q").astype(str))
        rows.append(
            {
                "variant_id": variant,
                "top_ticker": ticker_share.idxmax() if not ticker_share.empty else "",
                "top_ticker_weight_share": round(float(ticker_share.max() / ticker_share.sum()), 6) if not ticker_share.empty and float(ticker_share.sum()) else 0.0,
                "top_month_share": _period_return_share(month, "month"),
                "top_quarter_share": _period_return_share(quarter, "quarter"),
                "active_in_trade_decision": False,
            }
        )
    return pd.DataFrame(rows)


def _period_return_share(frame: pd.DataFrame, column: str) -> float:
    grouped = frame.groupby(column)["daily_return"].sum()
    positives = grouped[grouped > 0]
    if positives.empty or float(positives.sum()) == 0:
        return 0.0
    return round(float(positives.max() / positives.sum()), 6)


def _leave_one_period_report(daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    frame = daily.copy()
    frame["year"] = pd.to_datetime(frame["date"]).dt.year.astype(str)
    for variant, group in frame.groupby("variant_id"):
        full_start = float(group["portfolio_equity"].iloc[0])
        for year in sorted(group["year"].unique()):
            kept = group[group["year"].ne(year)]
            if kept.empty:
                continue
            rows.append(
                {
                    "variant_id": variant,
                    "excluded_period_type": "year",
                    "excluded_period": year,
                    "return_pct_excluding_period": round((float(kept["portfolio_equity"].iloc[-1]) / full_start - 1) * 100, 4) if full_start else "",
                    "active_in_trade_decision": False,
                }
            )
    return pd.DataFrame(rows)


def _hard_gate_caveat(performance: pd.DataFrame) -> pd.DataFrame:
    segment = performance[performance["period_label"].eq("2024_hard_gate")].copy()
    return pd.DataFrame(
        [
            {
                "variant_id": row.get("variant_id", ""),
                "period_label": "2024_hard_gate",
                "return_pct": row.get("return_pct", ""),
                "benchmark": "0050/00631L/0050x2",
                "caveat_state": "retained_not_resolved",
                "caveat_reason": "2024 權值槓桿行情機會成本需由 Experiments 對照；本 runner 不宣稱 resolved。",
                "active_in_trade_decision": False,
            }
            for _, row in segment.iterrows()
        ]
    )


def _exposure_integrity(daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant, group in daily.groupby("variant_id"):
        weights = pd.to_numeric(group["weight_sum"], errors="coerce")
        cash = pd.to_numeric(group["cash"], errors="coerce")
        rows.append(
            {
                "variant_id": variant,
                "max_weight_sum": round(float(weights.max()), 8),
                "min_weight_sum": round(float(weights.min()), 8),
                "negative_cash_count": int((cash < -0.01).sum()),
                "exposure_over_100_count": int((weights > 1.0001).sum()),
                "integrity_pass": bool((cash >= -0.01).all() and (weights <= 1.0001).all()),
                "active_in_trade_decision": False,
            }
        )
    return pd.DataFrame(rows)


def _summary_markdown(performance: pd.DataFrame, quality: pd.DataFrame, caveat: pd.DataFrame) -> str:
    main = performance[(performance["variant_id"] == "hold_old_if_still_valid_ma20_next_day") & (performance["period_label"] == "full")]
    baseline = performance[(performance["variant_id"] == "selector_full_switch_next_day_baseline") & (performance["period_label"] == "full")]
    main_row = main.iloc[0].to_dict() if not main.empty else {}
    base_row = baseline.iloc[0].to_dict() if not baseline.empty else {}
    return "\n".join(
        [
            "# Hold-old execution challenger runner",
            "",
            "本輸出使用 production-grade next-day ledger 口徑，只驗證 `hold_old_if_still_valid_ma20`，不啟用正式 execution layer。",
            "",
            "## 主候選",
            "- 主候選：`hold_old_if_still_valid_ma20_next_day`",
            f"- next-day baseline full return：{base_row.get('return_pct', '')}%",
            f"- hold-old full return：{main_row.get('return_pct', '')}%",
            f"- hold-old MDD：{main_row.get('max_drawdown_pct', '')}%",
            "",
            "## 邊界",
            "- formal_model_changed=false",
            "- trade_decision_changed=false",
            "- formal_execution_layer_activated=false",
            "- same-day reference 不作正式 execution performance",
            "- 2024 hard gate / 0050x2 caveat 保留，不宣稱 resolved",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run hold-old next-day execution challenger panels.")
    parser.add_argument("--review-dir", default=DEFAULT_REVIEW_DIR)
    parser.add_argument("--price-cache-dir", default=DEFAULT_PRICE_CACHE_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    run_execution_hold_old_next_day_challenger_runner(
        review_dir=args.review_dir,
        price_cache_dir=args.price_cache_dir,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
