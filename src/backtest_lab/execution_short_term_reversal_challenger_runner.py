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
DEFAULT_OUTPUT_DIR = "outputs/execution_short_term_reversal_challenger_20260626"


@dataclass(frozen=True)
class ReversalVariant:
    variant_id: str
    fill_delay_days: int
    mode: str
    confirm_rows: int = 1
    partial_weight: float = 1.0
    description: str = ""


VARIANTS = (
    ReversalVariant(
        "selector_full_switch_same_day_reference",
        0,
        "baseline",
        description="同日 full switch alignment reference；不得作正式 execution performance。",
    ),
    ReversalVariant(
        "selector_full_switch_next_day_baseline",
        1,
        "baseline",
        description="正式 target stream 的 next-day full rotation baseline。",
    ),
    ReversalVariant(
        "confirm_2_before_switch_next_day",
        1,
        "confirm_before_switch",
        confirm_rows=2,
        description="非現金 target 切換需連續 2 個交易列確認；cash / forced exit 不延後。",
    ),
    ReversalVariant(
        "confirm_3_before_switch_next_day",
        1,
        "confirm_before_switch",
        confirm_rows=3,
        description="非現金 target 切換需連續 3 個交易列確認；作較保守 sensitivity。",
    ),
    ReversalVariant(
        "partial_50_until_confirm_2_next_day",
        1,
        "partial_until_confirm",
        confirm_rows=2,
        partial_weight=0.5,
        description="新 target 第一天先切 50%，連續 2 個交易列確認後補到 100%。",
    ),
    ReversalVariant(
        "partial_75_until_confirm_2_next_day",
        1,
        "partial_until_confirm",
        confirm_rows=2,
        partial_weight=0.75,
        description="新 target 第一天先切 75%，連續 2 個交易列確認後補到 100%。",
    ),
)


def run_execution_short_term_reversal_challenger_runner(
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
            raise ValueError("no prices loaded for short-term reversal challenger")

        reversal_events = _formal_reversal_events(frame)
        daily_frames: list[pd.DataFrame] = []
        weight_frames: list[pd.DataFrame] = []
        trade_frames: list[pd.DataFrame] = []
        cash_frames: list[pd.DataFrame] = []
        blocked_frames: list[pd.DataFrame] = []
        policy_frames: list[pd.DataFrame] = []

        log("simulate_variants", "started", f"variants={len(VARIANTS)}")
        for variant in VARIANTS:
            daily, weights, trades, cash, blocked, policy = _simulate_variant(frame, prices, variant, initial_cash)
            daily_frames.append(daily)
            weight_frames.append(weights)
            trade_frames.append(trades)
            cash_frames.append(cash)
            blocked_frames.append(blocked)
            policy_frames.append(policy)

        daily = pd.concat(daily_frames, ignore_index=True)
        weights = pd.concat(weight_frames, ignore_index=True)
        trades = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()
        cash = pd.concat(cash_frames, ignore_index=True)
        blocked = pd.concat(blocked_frames, ignore_index=True) if blocked_frames else pd.DataFrame()
        policy = pd.concat(policy_frames, ignore_index=True)

        log("build_reports", "started", "")
        variant_matrix = _variant_matrix()
        performance = _period_performance(daily)
        alignment = _same_day_vs_next_day_alignment(frame, daily)
        quality = _execution_quality_scorecard(daily, trades, blocked, policy)
        stability = _target_stability(policy)
        concentration = _contribution_concentration(daily, weights)
        hard_gate = _hard_gate_caveat(performance)
        exposure = _exposure_integrity(daily)
        cost_turnover = _cost_turnover_summary(daily, trades)
        drawdown = _drawdown_summary(daily)

        log("write_outputs", "started", "")
        variant_matrix.to_csv(output / "variant_parameter_matrix.csv", index=False, encoding="utf-8-sig")
        reversal_events.to_csv(output / "short_term_reversal_event_panel.csv", index=False, encoding="utf-8-sig")
        daily.to_csv(output / "daily_equity_by_variant.csv", index=False, encoding="utf-8-sig")
        weights.to_csv(output / "portfolio_weight_ledger_by_variant.csv", index=False, encoding="utf-8-sig")
        trades.to_csv(output / "trade_ledger_by_variant.csv", index=False, encoding="utf-8-sig")
        cash.to_csv(output / "cash_ledger_by_variant.csv", index=False, encoding="utf-8-sig")
        blocked.to_csv(output / "blocked_fill_events.csv", index=False, encoding="utf-8-sig")
        policy.to_csv(output / "reversal_policy_daily_panel.csv", index=False, encoding="utf-8-sig")
        performance.to_csv(output / "period_performance_by_variant.csv", index=False, encoding="utf-8-sig")
        alignment.to_csv(output / "same_day_vs_next_day_alignment.csv", index=False, encoding="utf-8-sig")
        quality.to_csv(output / "execution_quality_scorecard.csv", index=False, encoding="utf-8-sig")
        stability.to_csv(output / "target_stability_by_variant.csv", index=False, encoding="utf-8-sig")
        concentration.to_csv(output / "contribution_concentration_by_variant.csv", index=False, encoding="utf-8-sig")
        hard_gate.to_csv(output / "hard_gate_2024_benchmark_caveat.csv", index=False, encoding="utf-8-sig")
        exposure.to_csv(output / "exposure_integrity_checks.csv", index=False, encoding="utf-8-sig")
        cost_turnover.to_csv(output / "cost_turnover_summary.csv", index=False, encoding="utf-8-sig")
        drawdown.to_csv(output / "drawdown_report.csv", index=False, encoding="utf-8-sig")
        (output / "execution_short_term_reversal_challenger_summary_zh.md").write_text(
            _summary_markdown(performance, quality, reversal_events),
            encoding="utf-8",
        )

        manifest = {
            "schema_version": 1,
            "task_id": "TASK-BACKTEST-CORE-EXECUTION-SHORT-TERM-REVERSAL-CHALLENGER-001",
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
                "reversal_event_panel": "short_term_reversal_event_panel.csv",
                "daily_equity": "daily_equity_by_variant.csv",
                "portfolio_weight_ledger": "portfolio_weight_ledger_by_variant.csv",
                "trade_ledger": "trade_ledger_by_variant.csv",
                "cash_ledger": "cash_ledger_by_variant.csv",
                "policy_panel": "reversal_policy_daily_panel.csv",
                "period_performance": "period_performance_by_variant.csv",
                "quality": "execution_quality_scorecard.csv",
                "summary": "execution_short_term_reversal_challenger_summary_zh.md",
            },
        }
        (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        pd.DataFrame([{"status": "completed", "output_dir": str(output.resolve())}]).to_csv(output / "completed.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(columns=["step", "error"]).to_csv(output / "failed.csv", index=False, encoding="utf-8-sig")
        log("completed", "completed", str(output.resolve()))
        (output / "current_step.txt").write_text("completed\n", encoding="utf-8")
        return output
    except Exception as exc:
        pd.DataFrame([{"step": "run_execution_short_term_reversal_challenger_runner", "error": str(exc)}]).to_csv(output / "failed.csv", index=False, encoding="utf-8-sig")
        log("failed", "failed", str(exc))
        raise


def _simulate_variant(
    frame: pd.DataFrame,
    prices: dict[str, pd.Series],
    variant: ReversalVariant,
    initial_cash: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    account = Account(cash=float(initial_cash), positions={})
    pending: list[dict[str, Any]] = []
    accepted_target: dict[str, float] = {}
    candidate_key = ""
    candidate_count = 0
    running_max = float(initial_cash)
    previous_equity = float(initial_cash)
    daily_rows: list[dict[str, Any]] = []
    weight_rows: list[dict[str, Any]] = []
    trade_rows: list[dict[str, Any]] = []
    cash_rows: list[dict[str, Any]] = []
    blocked_rows: list[dict[str, Any]] = []
    policy_rows: list[dict[str, Any]] = []

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
        accepted_key = _target_key(accepted_target)
        signal_key = _target_key(signal_weights)
        current_weights = _account_weights(account, prices, date, previous_equity)
        policy_weights, policy_state, policy_reason, candidate_key, candidate_count = _policy_weights(
            signal_weights=signal_weights,
            accepted_target=accepted_target,
            current_weights=current_weights,
            candidate_key=candidate_key,
            candidate_count=candidate_count,
            variant=variant,
        )

        target_changed = policy_weights != accepted_target
        pending_same_target = pending and policy_weights == dict(pending[-1]["target_weights"])
        source_trade_signal = str(item.get("action", "")).strip() in {"buy", "switch", "sell"} or _number(item.get("turnover")) > 0
        should_schedule = (target_changed and not pending_same_target) or (source_trade_signal and variant.mode == "baseline")
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
                "candidate_key": candidate_key,
                "candidate_count": candidate_count,
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
        policy_rows.append(
            {
                "variant_id": variant.variant_id,
                "date": date,
                "formal_target": signal_key or "cash",
                "accepted_target_before_policy": accepted_key or "cash",
                "policy_target": _target_key(policy_weights) or "cash",
                "policy_state": policy_state,
                "policy_reason": policy_reason,
                "candidate_key": candidate_key or "cash",
                "candidate_count": candidate_count,
                "active_in_trade_decision": False,
            }
        )

    return pd.DataFrame(daily_rows), pd.DataFrame(weight_rows), pd.DataFrame(trade_rows), pd.DataFrame(cash_rows), pd.DataFrame(blocked_rows), pd.DataFrame(policy_rows)


def _policy_weights(
    *,
    signal_weights: dict[str, float],
    accepted_target: dict[str, float],
    current_weights: dict[str, float],
    candidate_key: str,
    candidate_count: int,
    variant: ReversalVariant,
) -> tuple[dict[str, float], str, str, str, int]:
    signal_key = _target_key(signal_weights)
    accepted_key = _target_key(accepted_target)
    if variant.mode == "baseline":
        return signal_weights, "pass_through", "", candidate_key, candidate_count
    if not signal_weights:
        return {}, "forced_exit_allowed", "cash_or_no_target_not_delayed", "", 0
    if not accepted_target:
        return signal_weights, "entry_allowed", "no_existing_target_to_protect", signal_key, 1
    if signal_key == accepted_key:
        return signal_weights, "maintain_target", "formal_target_matches_execution_target", "", 0
    if signal_key == candidate_key:
        candidate_count += 1
    else:
        candidate_key = signal_key
        candidate_count = 1
    if variant.mode == "confirm_before_switch":
        if candidate_count >= variant.confirm_rows:
            return signal_weights, "confirmed_switch", f"candidate_confirmed_{candidate_count}_rows", candidate_key, candidate_count
        return accepted_target, "wait_for_confirmation", f"candidate_only_{candidate_count}_of_{variant.confirm_rows}_rows", candidate_key, candidate_count
    if variant.mode == "partial_until_confirm":
        if candidate_count >= variant.confirm_rows:
            return signal_weights, "confirmed_switch", f"candidate_confirmed_{candidate_count}_rows", candidate_key, candidate_count
        blended = _blend_weights(current_weights or accepted_target, signal_weights, variant.partial_weight)
        return blended, "partial_until_confirmed", f"partial_{int(variant.partial_weight * 100)}_candidate_{candidate_count}_of_{variant.confirm_rows}_rows", candidate_key, candidate_count
    return signal_weights, "pass_through", "", candidate_key, candidate_count


def _blend_weights(current: dict[str, float], target: dict[str, float], target_fraction: float) -> dict[str, float]:
    residual = max(0.0, 1.0 - target_fraction)
    merged: dict[str, float] = {}
    for ticker, weight in current.items():
        merged[ticker] = merged.get(ticker, 0.0) + weight * residual
    for ticker, weight in target.items():
        merged[ticker] = merged.get(ticker, 0.0) + weight * target_fraction
    return _normalize_weights(merged)


def _normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    cleaned = {ticker: max(0.0, float(weight)) for ticker, weight in weights.items() if ticker and float(weight) > 0}
    total = sum(cleaned.values())
    if total > 1.0:
        return {ticker: weight / total for ticker, weight in cleaned.items()}
    return cleaned


def _account_weights(account: Account, prices: dict[str, pd.Series], date: str, equity: float) -> dict[str, float]:
    if equity <= 0:
        return {}
    return _current_weights(account, prices, date, equity)


def _formal_reversal_events(frame: pd.DataFrame, max_window: int = 3) -> pd.DataFrame:
    keys = [_target_key(_parse_weights(value)) or "cash" for value in frame["target_weights"].tolist()]
    rows: list[dict[str, Any]] = []
    for index in range(1, len(keys)):
        old_key = keys[index - 1]
        new_key = keys[index]
        if new_key == old_key:
            continue
        for offset in range(1, max_window + 1):
            check = index + offset
            if check >= len(keys):
                continue
            if keys[check] == old_key:
                rows.append(
                    {
                        "switch_date": str(frame.iloc[index]["date"]),
                        "reversal_date": str(frame.iloc[check]["date"]),
                        "from_target": old_key,
                        "temporary_target": new_key,
                        "window_rows": offset,
                        "active_in_trade_decision": False,
                    }
                )
                break
    return pd.DataFrame(rows)


def _target_key(weights: dict[str, float]) -> str:
    if not weights:
        return ""
    return "|".join(f"{ticker}:{weight:.6f}" for ticker, weight in sorted(weights.items()))


def _blocked_row(variant: ReversalVariant, signal_date: str, fill_date: str, weights: dict[str, float], reason: str) -> dict[str, Any]:
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
                "mode": variant.mode,
                "confirm_rows": variant.confirm_rows,
                "partial_weight": variant.partial_weight,
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


def _execution_quality_scorecard(daily: pd.DataFrame, trades: pd.DataFrame, blocked: pd.DataFrame, policy: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant, group in daily.groupby("variant_id"):
        trade_group = trades[trades["variant_id"].eq(variant)] if not trades.empty else pd.DataFrame()
        blocked_group = blocked[blocked["variant_id"].eq(variant)] if not blocked.empty else pd.DataFrame()
        policy_group = policy[policy["variant_id"].eq(variant)] if not policy.empty else pd.DataFrame()
        rows.append(
            {
                "variant_id": variant,
                "final_equity": round(float(group["portfolio_equity"].iloc[-1]), 2),
                "max_drawdown_pct": round(float(pd.to_numeric(group["drawdown"], errors="coerce").min()) * 100, 4),
                "trade_rows": int(len(trade_group)),
                "blocked_fill_count": int(len(blocked_group)),
                "wait_confirmation_rows": int(policy_group["policy_state"].eq("wait_for_confirmation").sum()) if not policy_group.empty else 0,
                "partial_until_confirmed_rows": int(policy_group["policy_state"].eq("partial_until_confirmed").sum()) if not policy_group.empty else 0,
                "confirmed_switch_rows": int(policy_group["policy_state"].eq("confirmed_switch").sum()) if not policy_group.empty else 0,
                "average_cash_weight": round(float(pd.to_numeric(group["cash_weight"], errors="coerce").mean()), 6),
                "ready_for_formal_activation": False,
                "active_in_trade_decision": False,
            }
        )
    return pd.DataFrame(rows)


def _target_stability(policy: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant, group in policy.groupby("variant_id"):
        targets = group.sort_values("date")["policy_target"].astype(str).tolist()
        changes = sum(1 for prev, cur in zip(targets, targets[1:]) if prev != cur)
        rows.append(
            {
                "variant_id": variant,
                "policy_target_change_count": changes,
                "target_changed_within_1d_rate": _changed_within_rate(targets, 1),
                "target_changed_within_3d_rate": _changed_within_rate(targets, 3),
                "rapid_flip_same_target_window_1_3d_rate": _rapid_flip_rate(targets),
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


def _summary_markdown(performance: pd.DataFrame, quality: pd.DataFrame, reversal_events: pd.DataFrame) -> str:
    baseline = performance[(performance["variant_id"] == "selector_full_switch_next_day_baseline") & (performance["period_label"] == "full")]
    confirm = performance[(performance["variant_id"] == "confirm_2_before_switch_next_day") & (performance["period_label"] == "full")]
    partial = performance[(performance["variant_id"] == "partial_50_until_confirm_2_next_day") & (performance["period_label"] == "full")]
    base_row = baseline.iloc[0].to_dict() if not baseline.empty else {}
    confirm_row = confirm.iloc[0].to_dict() if not confirm.empty else {}
    partial_row = partial.iloc[0].to_dict() if not partial.empty else {}
    return "\n".join(
        [
            "# Short-term reversal execution challenger runner",
            "",
            "本輸出只處理 1-3 交易列內快速反轉情境，測試確認後切換與先半倉再確認；不啟用正式 execution layer。",
            "",
            f"- detected short-term reversal events: {len(reversal_events)}",
            f"- next-day baseline full return：{base_row.get('return_pct', '')}%",
            f"- confirm2 full return：{confirm_row.get('return_pct', '')}%",
            f"- partial50 confirm2 full return：{partial_row.get('return_pct', '')}%",
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
    parser = argparse.ArgumentParser(description="Run short-term reversal execution challenger panels.")
    parser.add_argument("--review-dir", default=DEFAULT_REVIEW_DIR)
    parser.add_argument("--price-cache-dir", default=DEFAULT_PRICE_CACHE_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    run_execution_short_term_reversal_challenger_runner(
        review_dir=args.review_dir,
        price_cache_dir=args.price_cache_dir,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
