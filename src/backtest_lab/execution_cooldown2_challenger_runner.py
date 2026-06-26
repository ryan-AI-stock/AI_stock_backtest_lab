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
    _trade_row,
    _validate_stream,
)


DEFAULT_REVIEW_DIR = "outputs/execution_layer_review_pool1_pool2_formal_20260626"
DEFAULT_PRICE_CACHE_DIR = "backtest_cache/stock_pool_triad_v1_corrected"
DEFAULT_OUTPUT_DIR = "outputs/execution_cooldown2_challenger_runner_20260626"


@dataclass(frozen=True)
class CooldownVariant:
    variant_id: str
    fill_delay_days: int
    cooldown_after_switch_rows: int | None = None
    description: str = ""


VARIANTS = (
    CooldownVariant(
        "selector_full_switch_same_day_reference",
        0,
        None,
        "同日 full switch alignment reference；不得作正式 execution performance。",
    ),
    CooldownVariant(
        "selector_full_switch_next_day_baseline",
        1,
        None,
        "正式 target stream 的 next-day full rotation baseline。",
    ),
    CooldownVariant(
        "cooldown_after_switch_2",
        1,
        2,
        "非現金 target 成交切換後 2 個交易列內不接受短期反向切換；cash / forced exit 不阻擋。",
    ),
    CooldownVariant(
        "cooldown_after_switch_3",
        1,
        3,
        "非現金 target 成交切換後 3 個交易列內不接受短期反向切換；作 sensitivity。",
    ),
)


def run_execution_cooldown2_challenger_runner(
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
            raise ValueError("no prices loaded for cooldown challenger")

        daily_frames: list[pd.DataFrame] = []
        weight_frames: list[pd.DataFrame] = []
        trade_frames: list[pd.DataFrame] = []
        cash_frames: list[pd.DataFrame] = []
        blocked_frames: list[pd.DataFrame] = []
        cooldown_event_frames: list[pd.DataFrame] = []
        cooldown_state_frames: list[pd.DataFrame] = []

        log("simulate_variants", "started", f"variants={len(VARIANTS)}")
        for variant in VARIANTS:
            daily, weights, trades, cash, blocked, cooldown_events, cooldown_state = _simulate_cooldown_variant(
                frame,
                prices,
                variant,
                initial_cash,
            )
            daily_frames.append(daily)
            weight_frames.append(weights)
            trade_frames.append(trades)
            cash_frames.append(cash)
            blocked_frames.append(blocked)
            cooldown_event_frames.append(cooldown_events)
            cooldown_state_frames.append(cooldown_state)

        daily = pd.concat(daily_frames, ignore_index=True)
        weights = pd.concat(weight_frames, ignore_index=True)
        trades = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()
        cash = pd.concat(cash_frames, ignore_index=True)
        blocked = pd.concat(blocked_frames, ignore_index=True) if blocked_frames else pd.DataFrame()
        cooldown_events = pd.concat(cooldown_event_frames, ignore_index=True) if cooldown_event_frames else pd.DataFrame()
        cooldown_state = pd.concat(cooldown_state_frames, ignore_index=True)

        log("build_reports", "started", "")
        variant_matrix = _variant_matrix()
        performance = _period_performance(daily)
        alignment = _same_day_vs_next_day_alignment(frame, daily)
        quality = _execution_quality_scorecard(daily, trades, blocked, cooldown_events)
        stability = _target_stability(daily, cooldown_state)
        entry_without_exit = _entry_without_exit(daily, frame)
        caveat = _hard_gate_caveat(performance)
        cost_sensitivity = _cost_sensitivity(daily, trades)
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
        cooldown_events.to_csv(output / "cooldown_event_panel.csv", index=False, encoding="utf-8-sig")
        cooldown_state.to_csv(output / "cooldown_state_daily_panel.csv", index=False, encoding="utf-8-sig")
        performance.to_csv(output / "period_performance_by_variant.csv", index=False, encoding="utf-8-sig")
        alignment.to_csv(output / "same_day_vs_next_day_alignment.csv", index=False, encoding="utf-8-sig")
        quality.to_csv(output / "execution_quality_scorecard.csv", index=False, encoding="utf-8-sig")
        stability.to_csv(output / "target_stability_by_variant.csv", index=False, encoding="utf-8-sig")
        entry_without_exit.to_csv(output / "entry_without_exit_by_variant.csv", index=False, encoding="utf-8-sig")
        caveat.to_csv(output / "hard_gate_2024_benchmark_caveat.csv", index=False, encoding="utf-8-sig")
        cost_sensitivity.to_csv(output / "cost_sensitivity_by_variant.csv", index=False, encoding="utf-8-sig")
        exposure.to_csv(output / "exposure_integrity_checks.csv", index=False, encoding="utf-8-sig")
        cost_turnover.to_csv(output / "cost_turnover_summary.csv", index=False, encoding="utf-8-sig")
        drawdown.to_csv(output / "drawdown_report.csv", index=False, encoding="utf-8-sig")
        (output / "execution_cooldown2_challenger_summary_zh.md").write_text(
            _summary_markdown(performance, quality, caveat),
            encoding="utf-8",
        )
        manifest = {
            "schema_version": 1,
            "task_id": "TASK-BACKTEST-CORE-EXECUTION-COOLDOWN2-CHALLENGER-RUNNER-001",
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
                "cooldown_event_panel": "cooldown_event_panel.csv",
                "cooldown_state_daily_panel": "cooldown_state_daily_panel.csv",
                "period_performance": "period_performance_by_variant.csv",
                "alignment": "same_day_vs_next_day_alignment.csv",
                "quality": "execution_quality_scorecard.csv",
                "hard_gate_caveat": "hard_gate_2024_benchmark_caveat.csv",
                "summary": "execution_cooldown2_challenger_summary_zh.md",
            },
        }
        (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        pd.DataFrame([{"status": "completed", "output_dir": str(output.resolve())}]).to_csv(
            output / "completed.csv", index=False, encoding="utf-8-sig"
        )
        pd.DataFrame(columns=["step", "error"]).to_csv(output / "failed.csv", index=False, encoding="utf-8-sig")
        log("completed", "completed", str(output.resolve()))
        (output / "current_step.txt").write_text("completed\n", encoding="utf-8")
        return output
    except Exception as exc:
        pd.DataFrame([{"step": "run_execution_cooldown2_challenger_runner", "error": str(exc)}]).to_csv(
            output / "failed.csv", index=False, encoding="utf-8-sig"
        )
        log("failed", "failed", str(exc))
        raise


def _simulate_cooldown_variant(
    frame: pd.DataFrame,
    prices: dict[str, pd.Series],
    variant: CooldownVariant,
    initial_cash: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    account = Account(cash=float(initial_cash), positions={})
    pending: list[dict[str, Any]] = []
    accepted_target: dict[str, float] = {}
    accepted_target_key = ""
    cooldown_until_index = -1
    cooldown_source_fill_date = ""
    running_max = float(initial_cash)
    previous_equity = float(initial_cash)
    daily_rows: list[dict[str, Any]] = []
    weight_rows: list[dict[str, Any]] = []
    trade_rows: list[dict[str, Any]] = []
    cash_rows: list[dict[str, Any]] = []
    blocked_rows: list[dict[str, Any]] = []
    cooldown_event_rows: list[dict[str, Any]] = []
    cooldown_state_rows: list[dict[str, Any]] = []

    for index, item in frame.iterrows():
        date = str(item["date"])
        due = [order for order in pending if int(order["fill_index"]) == index]
        pending = [order for order in pending if int(order["fill_index"]) != index]
        for order in due:
            trades, blocked_reason = _execute_order(account, prices, date, order["target_weights"], _cost_model(), order["reason"])
            if blocked_reason:
                blocked_rows.append(_blocked_row(variant, order["signal_date"], date, order["target_weights"], blocked_reason))
            else:
                previous_key = accepted_target_key
                accepted_target = dict(order["target_weights"])
                accepted_target_key = _target_key(accepted_target)
                trade_rows.extend({**trade, "variant_id": variant.variant_id, "signal_date": order["signal_date"]} for trade in trades)
                if _starts_cooldown(previous_key, accepted_target_key, variant):
                    cooldown_until_index = index + int(variant.cooldown_after_switch_rows or 0)
                    cooldown_source_fill_date = date
                    cooldown_event_rows.append(
                        {
                            "variant_id": variant.variant_id,
                            "cooldown_start_signal_date": order["signal_date"],
                            "cooldown_start_fill_date": date,
                            "cooldown_end_date": str(frame.iloc[min(cooldown_until_index, len(frame) - 1)]["date"]),
                            "cooldown_after_switch_rows": variant.cooldown_after_switch_rows or "",
                            "from_target": previous_key or "cash",
                            "to_target": accepted_target_key or "cash",
                            "state": "started",
                            "active_in_trade_decision": False,
                        }
                    )

        signal_weights = _parse_weights(item.get("target_weights"))
        policy_weights, policy_state, policy_reason = _cooldown_policy(
            signal_weights=signal_weights,
            accepted_target=accepted_target,
            row_index=index,
            cooldown_until_index=cooldown_until_index,
            variant=variant,
        )
        if policy_state == "blocked_by_cooldown":
            cooldown_event_rows.append(
                {
                    "variant_id": variant.variant_id,
                    "cooldown_start_signal_date": "",
                    "cooldown_start_fill_date": cooldown_source_fill_date,
                    "cooldown_end_date": str(frame.iloc[min(cooldown_until_index, len(frame) - 1)]["date"]) if cooldown_until_index >= 0 else "",
                    "cooldown_after_switch_rows": variant.cooldown_after_switch_rows or "",
                    "from_target": accepted_target_key or "cash",
                    "to_target": _target_key(signal_weights) or "cash",
                    "state": "blocked_switch",
                    "active_in_trade_decision": False,
                }
            )

        target_changed = policy_weights != accepted_target
        pending_same_target = pending and policy_weights == dict(pending[-1]["target_weights"])
        source_trade_signal = str(item.get("action", "")).strip() in {"buy", "switch", "sell"} or _number(item.get("turnover")) > 0
        should_schedule = (target_changed and not pending_same_target) or source_trade_signal
        if should_schedule and policy_state != "blocked_by_cooldown":
            fill_index = index + variant.fill_delay_days
            if fill_index >= len(frame):
                blocked_rows.append(_blocked_row(variant, date, "", policy_weights, "missing_future_fill_row"))
            elif variant.fill_delay_days == 0:
                trades, blocked_reason = _execute_order(account, prices, date, policy_weights, _cost_model(), "same_day_fill")
                if blocked_reason:
                    blocked_rows.append(_blocked_row(variant, date, date, policy_weights, blocked_reason))
                else:
                    previous_key = accepted_target_key
                    accepted_target = dict(policy_weights)
                    accepted_target_key = _target_key(accepted_target)
                    trade_rows.extend({**trade, "variant_id": variant.variant_id, "signal_date": date} for trade in trades)
                    if _starts_cooldown(previous_key, accepted_target_key, variant):
                        cooldown_until_index = index + int(variant.cooldown_after_switch_rows or 0)
                        cooldown_source_fill_date = date
            else:
                fill_date = str(frame.iloc[fill_index]["date"])
                pending = [{"fill_index": fill_index, "signal_date": date, "target_weights": dict(policy_weights), "reason": "next_day_fill"}]

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
                "cooldown_active": bool(variant.cooldown_after_switch_rows is not None and index <= cooldown_until_index),
                "cooldown_until_date": str(frame.iloc[min(cooldown_until_index, len(frame) - 1)]["date"]) if cooldown_until_index >= index else "",
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
            weight_rows.append(
                {
                    "variant_id": variant.variant_id,
                    "date": date,
                    "ticker": "cash",
                    "weight": round(cash_weight, 8),
                    "shares": 0,
                    "asset_type": "cash",
                    "active_in_trade_decision": False,
                }
            )
        cooldown_state_rows.append(
            {
                "variant_id": variant.variant_id,
                "date": date,
                "formal_target": _target_key(signal_weights) or "cash",
                "execution_target": accepted_target_key or "cash",
                "cooldown_active": bool(variant.cooldown_after_switch_rows is not None and index <= cooldown_until_index),
                "cooldown_until_index": cooldown_until_index if cooldown_until_index >= index else "",
                "policy_state": policy_state,
                "policy_reason": policy_reason,
                "active_in_trade_decision": False,
            }
        )

    return (
        pd.DataFrame(daily_rows),
        pd.DataFrame(weight_rows),
        pd.DataFrame(trade_rows),
        pd.DataFrame(cash_rows),
        pd.DataFrame(blocked_rows),
        pd.DataFrame(cooldown_event_rows),
        pd.DataFrame(cooldown_state_rows),
    )


def _cooldown_policy(
    *,
    signal_weights: dict[str, float],
    accepted_target: dict[str, float],
    row_index: int,
    cooldown_until_index: int,
    variant: CooldownVariant,
) -> tuple[dict[str, float], str, str]:
    if variant.cooldown_after_switch_rows is None:
        return signal_weights, "pass_through", ""
    if not signal_weights:
        return {}, "exit_allowed", "cash_or_no_target_not_blocked_by_cooldown"
    if row_index <= cooldown_until_index and signal_weights != accepted_target:
        return dict(accepted_target), "blocked_by_cooldown", f"cooldown_after_switch_{variant.cooldown_after_switch_rows}_active"
    return signal_weights, "pass_through", ""


def _starts_cooldown(previous_key: str, current_key: str, variant: CooldownVariant) -> bool:
    return bool(variant.cooldown_after_switch_rows is not None and previous_key and current_key and previous_key != current_key)


def _target_key(weights: dict[str, float]) -> str:
    if not weights:
        return ""
    return "|".join(f"{ticker}:{weight:.6f}" for ticker, weight in sorted(weights.items()))


def _blocked_row(variant: CooldownVariant, signal_date: str, fill_date: str, weights: dict[str, float], reason: str) -> dict[str, Any]:
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
                "cooldown_after_switch_rows": variant.cooldown_after_switch_rows or "",
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


def _execution_quality_scorecard(daily: pd.DataFrame, trades: pd.DataFrame, blocked: pd.DataFrame, cooldown_events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant, group in daily.groupby("variant_id"):
        trade_group = trades[trades["variant_id"].eq(variant)] if not trades.empty else pd.DataFrame()
        blocked_group = blocked[blocked["variant_id"].eq(variant)] if not blocked.empty else pd.DataFrame()
        cooldown_group = cooldown_events[cooldown_events["variant_id"].eq(variant)] if not cooldown_events.empty else pd.DataFrame()
        rows.append(
            {
                "variant_id": variant,
                "final_equity": round(float(group["portfolio_equity"].iloc[-1]), 2),
                "max_drawdown_pct": round(float(pd.to_numeric(group["drawdown"], errors="coerce").min()) * 100, 4),
                "trade_rows": int(len(trade_group)),
                "blocked_fill_count": int(len(blocked_group)),
                "cooldown_blocked_switch_count": int(cooldown_group["state"].eq("blocked_switch").sum()) if not cooldown_group.empty else 0,
                "average_cash_weight": round(float(pd.to_numeric(group["cash_weight"], errors="coerce").mean()), 6),
                "ready_for_formal_activation": False,
                "active_in_trade_decision": False,
            }
        )
    return pd.DataFrame(rows)


def _target_stability(daily: pd.DataFrame, cooldown_state: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant, group in cooldown_state.groupby("variant_id"):
        targets = group.sort_values("date")["execution_target"].astype(str).tolist()
        changes = sum(1 for prev, cur in zip(targets, targets[1:]) if prev != cur)
        rows.append(
            {
                "variant_id": variant,
                "execution_target_change_count": changes,
                "target_changed_within_1d_rate": _changed_within_rate(targets, 1),
                "target_changed_within_3d_rate": _changed_within_rate(targets, 3),
                "rapid_flip_same_target_window_1_3d_rate": _rapid_flip_rate(targets),
                "possible_execution_layer_issue_rate": _changed_within_rate(targets, 3),
                "active_in_trade_decision": False,
            }
        )
    return pd.DataFrame(rows)


def _entry_without_exit(daily: pd.DataFrame, stream: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant, group in daily.groupby("variant_id"):
        merged = group[["date", "top_holding"]].merge(stream[["date", "target_weights"]], on="date", how="left")
        entries = 0
        without_exit = 0
        previous = "cash"
        for _, row in merged.iterrows():
            current = str(row.get("top_holding") or "cash")
            signal = _parse_weights(row.get("target_weights"))
            if previous == "cash" and current != "cash":
                entries += 1
                if signal:
                    without_exit += 1
            previous = current
        rows.append(
            {
                "variant_id": variant,
                "entry_count": entries,
                "entry_without_exit_confirmation_count": without_exit,
                "entry_without_exit_confirmation_rate": round(without_exit / entries, 6) if entries else 0.0,
                "active_in_trade_decision": False,
            }
        )
    return pd.DataFrame(rows)


def _hard_gate_caveat(performance: pd.DataFrame) -> pd.DataFrame:
    segment = performance[performance["period_label"].eq("2024_hard_gate")].copy()
    rows = []
    for _, row in segment.iterrows():
        rows.append(
            {
                "variant_id": row.get("variant_id", ""),
                "period_label": "2024_hard_gate",
                "return_pct": row.get("return_pct", ""),
                "benchmark": "0050x2",
                "caveat_state": "retained_not_resolved",
                "caveat_reason": "2024 權值槓桿行情機會成本仍需由 Experiments 驗證；本 runner 不宣稱 resolved。",
                "active_in_trade_decision": False,
            }
        )
    return pd.DataFrame(rows)


def _cost_sensitivity(daily: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    base = _cost_turnover_summary(daily, trades)
    rows = []
    for _, row in base.iterrows():
        cost = float(row.get("total_transaction_cost") or 0.0)
        rows.append({**row.to_dict(), "cost_multiplier": 1.0, "estimated_cost": round(cost, 2)})
        rows.append({**row.to_dict(), "cost_multiplier": 2.0, "estimated_cost": round(cost * 2, 2)})
    return pd.DataFrame(rows)


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


def _changed_within_rate(targets: list[str], window: int) -> float:
    if len(targets) <= 1:
        return 0.0
    flags = []
    for index in range(len(targets)):
        current = targets[index]
        changed = any(targets[j] != current for j in range(index + 1, min(len(targets), index + window + 1)))
        flags.append(changed)
    return round(sum(flags) / len(flags), 6)


def _rapid_flip_rate(targets: list[str]) -> float:
    if len(targets) < 3:
        return 0.0
    count = 0
    for index in range(len(targets) - 2):
        if targets[index] == targets[index + 2] and targets[index] != targets[index + 1]:
            count += 1
    return round(count / len(targets), 6)


def _summary_markdown(performance: pd.DataFrame, quality: pd.DataFrame, caveat: pd.DataFrame) -> str:
    main = performance[(performance["variant_id"] == "cooldown_after_switch_2") & (performance["period_label"] == "full")]
    baseline = performance[(performance["variant_id"] == "selector_full_switch_next_day_baseline") & (performance["period_label"] == "full")]
    main_row = main.iloc[0].to_dict() if not main.empty else {}
    base_row = baseline.iloc[0].to_dict() if not baseline.empty else {}
    return "\n".join(
        [
            "# Cooldown-after-switch execution challenger runner",
            "",
            "本輸出使用 production-grade next-day ledger 口徑，只作 execution challenger evidence，不啟用正式 execution layer。",
            "",
            "## 主候選",
            "- 主候選：`cooldown_after_switch_2`",
            "- sensitivity：`cooldown_after_switch_3`",
            f"- next-day baseline full return：{base_row.get('return_pct', '')}%",
            f"- cooldown2 full return：{main_row.get('return_pct', '')}%",
            f"- cooldown2 MDD：{main_row.get('max_drawdown_pct', '')}%",
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
    parser = argparse.ArgumentParser(description="Run production-grade cooldown-after-switch execution challenger panels.")
    parser.add_argument("--review-dir", default=DEFAULT_REVIEW_DIR)
    parser.add_argument("--price-cache-dir", default=DEFAULT_PRICE_CACHE_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    run_execution_cooldown2_challenger_runner(
        review_dir=args.review_dir,
        price_cache_dir=args.price_cache_dir,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
