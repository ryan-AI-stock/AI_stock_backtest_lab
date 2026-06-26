from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from backtest_lab.costs import TaiwanCostModel
from backtest_lab.data import load_price_csv
from backtest_lab.execution_layer_next_day_ab_pool1_pool2_formal import FORMAL_MODEL_TARGET, INITIAL_CASH
from backtest_lab.partial_execution_ledger import LedgerAccount, _account_value, _rebalance_to_weights


DEFAULT_FORMAL_TARGET_STREAM = "outputs/execution_layer_review_pool1_pool2_formal_20260626/formal_target_stream_adapter.csv"
DEFAULT_PRICE_CACHE_DIR = "backtest_cache/stock_pool_triad_v1_corrected"
DEFAULT_OUTPUT_DIR = "outputs/execution_position_transition_runner_20260626"


@dataclass(frozen=True)
class TransitionVariant:
    variant_id: str
    mode: str
    first_step_weight: float | None = None
    confirm_days: int = 0
    description: str = ""


VARIANTS = (
    TransitionVariant("full_switch_100_same_day_baseline", "full", description="沿用 formal replay：target change 時 100% 切換。"),
    TransitionVariant("no_action_hold_current", "no_action", description="target change 時不動，用來衡量不換倉機會成本。"),
    *(
        TransitionVariant(
            f"partial_switch_{weight:02d}_on_change",
            "partial",
            first_step_weight=weight / 100,
            description=f"target change 時先換 {weight}%，保留 {100 - weight}% 既有持股或現金。",
        )
        for weight in range(10, 101, 10)
    ),
    *(
        TransitionVariant(
            f"cash_buffer_{weight:02d}_on_change",
            "cash_buffer",
            first_step_weight=weight / 100,
            description=f"target change 時只配置 {weight}% 新 target，其餘留現金。",
        )
        for weight in (25, 50, 75)
    ),
    *(
        TransitionVariant(
            f"staged_switch_{weight:02d}_then_100_confirm_2",
            "staged",
            first_step_weight=weight / 100,
            confirm_days=2,
            description=f"先換 {weight}%，新 target 連續 2 個交易列後補到 100%。",
        )
        for weight in (30, 50, 75)
    ),
    TransitionVariant("hold_old_if_still_valid_ma20", "hold_old_if_valid", description="舊持股仍在 20 日均線上且 20 日報酬非負時先不賣。"),
)


def run_execution_position_transition_runner(
    *,
    formal_target_stream: str | Path = DEFAULT_FORMAL_TARGET_STREAM,
    price_cache_dir: str | Path = DEFAULT_PRICE_CACHE_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    initial_cash: float = INITIAL_CASH,
) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    run_log: list[dict[str, str]] = []

    def log(step: str, status: str, detail: str = "") -> None:
        run_log.append({"timestamp": pd.Timestamp.now(tz="Asia/Taipei").strftime("%Y-%m-%d %H:%M:%S%z"), "step": step, "status": status, "detail": detail})
        pd.DataFrame(run_log).to_csv(output / "run_log.csv", index=False, encoding="utf-8-sig")
        (output / "current_step.txt").write_text(step, encoding="utf-8")

    try:
        log("load_formal_target_stream", "started", str(formal_target_stream))
        stream = _normalize_stream(pd.read_csv(formal_target_stream).fillna(""))
        prices = _load_prices(stream, Path(price_cache_dir))
        if not prices:
            raise ValueError("no prices loaded for position transition runner")

        daily_frames: list[pd.DataFrame] = []
        trade_frames: list[pd.DataFrame] = []
        transition_frames: list[pd.DataFrame] = []
        log("simulate_variants", "started", f"variants={len(VARIANTS)}")
        for variant in VARIANTS:
            daily, trades, transitions = _simulate_variant(stream, prices, variant, initial_cash)
            daily_frames.append(daily)
            trade_frames.append(trades)
            transition_frames.append(transitions)

        daily = pd.concat(daily_frames, ignore_index=True)
        trades = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()
        transitions = pd.concat(transition_frames, ignore_index=True)

        log("build_reports", "started", "")
        variant_matrix = _variant_matrix()
        performance = _period_performance(daily)
        risk = _risk_summary(daily)
        costs = _cost_summary(trades)
        stability = _transition_stability(transitions)
        exposure = _exposure_integrity(daily)
        comparison = _baseline_comparison(performance)
        sizing_curve = _sizing_curve_report(performance, costs)

        log("write_outputs", "started", "")
        variant_matrix.to_csv(output / "variant_parameter_matrix.csv", index=False, encoding="utf-8-sig")
        daily.to_csv(output / "position_transition_daily_ledger.csv", index=False, encoding="utf-8-sig")
        trades.to_csv(output / "position_transition_trade_ledger.csv", index=False, encoding="utf-8-sig")
        transitions.to_csv(output / "position_transition_event_panel.csv", index=False, encoding="utf-8-sig")
        performance.to_csv(output / "period_performance_by_variant.csv", index=False, encoding="utf-8-sig")
        risk.to_csv(output / "drawdown_and_risk_by_variant.csv", index=False, encoding="utf-8-sig")
        costs.to_csv(output / "cost_turnover_by_variant.csv", index=False, encoding="utf-8-sig")
        stability.to_csv(output / "transition_stability_by_variant.csv", index=False, encoding="utf-8-sig")
        exposure.to_csv(output / "exposure_integrity_checks.csv", index=False, encoding="utf-8-sig")
        comparison.to_csv(output / "baseline_comparison_by_variant.csv", index=False, encoding="utf-8-sig")
        sizing_curve.to_csv(output / "sizing_curve_report.csv", index=False, encoding="utf-8-sig")
        (output / "execution_position_transition_summary_zh.md").write_text(_summary_markdown(performance, comparison), encoding="utf-8")

        manifest = {
            "schema_version": 1,
            "task_id": "TASK-BACKTEST-CORE-EXECUTION-POSITION-TRANSITION-LAYER-001",
            "status": "completed_diagnostic_runner",
            "formal_model_target": FORMAL_MODEL_TARGET,
            "formal_model_route": "pool1_primary_pool2_confirmation_cap40",
            "execution_layer_status": "diagnostic_challenger",
            "formal_target_stream": str(formal_target_stream),
            "price_cache_dir": str(price_cache_dir),
            "start_date": str(stream["date"].iloc[0]) if not stream.empty else "",
            "latest_complete_common_date": str(stream["date"].iloc[-1]) if not stream.empty else "",
            "replay_timing_basis": "same_day_formal_replay_口徑",
            "formal_model_changed": False,
            "trade_decision_changed": False,
            "formal_execution_layer_activated": False,
            "active_in_trade_decision": False,
            "execution_diagnostic_active_in_trade_decision": False,
            "uses_forward_return_as_rule": False,
            "pool3_shadow_used": False,
            "rr_partial_switch_used": False,
            "valuation_used": False,
            "h3_used": False,
            "outputs": {
                "variant_parameter_matrix": "variant_parameter_matrix.csv",
                "daily_ledger": "position_transition_daily_ledger.csv",
                "trade_ledger": "position_transition_trade_ledger.csv",
                "event_panel": "position_transition_event_panel.csv",
                "period_performance": "period_performance_by_variant.csv",
                "baseline_comparison": "baseline_comparison_by_variant.csv",
                "sizing_curve": "sizing_curve_report.csv",
                "summary": "execution_position_transition_summary_zh.md",
            },
        }
        (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        pd.DataFrame([{"status": "completed", "output_dir": str(output.resolve())}]).to_csv(output / "completed.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(columns=["step", "error"]).to_csv(output / "failed.csv", index=False, encoding="utf-8-sig")
        log("completed", "completed", str(output.resolve()))
        (output / "current_step.txt").write_text("completed\n", encoding="utf-8")
        return output
    except Exception as exc:
        pd.DataFrame([{"step": "run_execution_position_transition_runner", "error": str(exc)}]).to_csv(output / "failed.csv", index=False, encoding="utf-8-sig")
        log("failed", "failed", str(exc))
        raise


def _simulate_variant(frame: pd.DataFrame, prices: dict[str, pd.Series], variant: TransitionVariant, initial_cash: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    account = LedgerAccount(cash=float(initial_cash), positions={})
    cost_model = TaiwanCostModel()
    daily_rows: list[dict[str, Any]] = []
    trade_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    previous_equity = float(initial_cash)
    running_max = float(initial_cash)
    last_formal_target_key = ""
    pending_stage_target_key = ""
    pending_stage_count = 0

    for _, item in frame.iterrows():
        date = str(item["date"])
        formal_weights = _parse_weights(item.get("target_weights"))
        formal_key = _target_key(formal_weights)
        current_weights = _current_account_weights(account, prices, date, previous_equity)
        current_top = _top_holding(current_weights)
        target_changed = formal_key != last_formal_target_key
        action = "hold"
        reason = ""
        desired = current_weights
        if not formal_weights:
            desired = {}
            action = "forced_exit_to_cash"
            reason = "formal target disappeared or risk-off/no-target"
            pending_stage_target_key = ""
            pending_stage_count = 0
        elif variant.mode == "no_action":
            desired = current_weights
            action = "no_action"
            reason = "hold current position regardless of formal target change"
        elif variant.mode == "full":
            desired = formal_weights
            action = "full_switch_100"
            reason = "baseline formal replay full switch"
        elif variant.mode == "partial":
            desired = _blend_current_and_target(current_weights, formal_weights, float(variant.first_step_weight or 1.0)) if target_changed else formal_weights
            action = f"partial_switch_{int((variant.first_step_weight or 0) * 100)}" if target_changed else "maintain_target"
            reason = "partial switch on target change"
        elif variant.mode == "cash_buffer":
            desired = _scale_weights(formal_weights, float(variant.first_step_weight or 1.0)) if target_changed else formal_weights
            action = "cash_buffer_on_change" if target_changed else "maintain_target"
            reason = "keep residual cash on target change"
        elif variant.mode == "staged":
            if target_changed:
                pending_stage_target_key = formal_key
                pending_stage_count = 1
                desired = _blend_current_and_target(current_weights, formal_weights, float(variant.first_step_weight or 0.5))
                action = f"staged_first_{int((variant.first_step_weight or 0) * 100)}"
                reason = "first stage after target change"
            elif pending_stage_target_key == formal_key and pending_stage_count < variant.confirm_days:
                pending_stage_count += 1
                desired = current_weights
                action = "staged_wait_confirmation"
                reason = f"waiting {variant.confirm_days} rows confirmation"
            elif pending_stage_target_key == formal_key:
                desired = formal_weights
                action = "staged_complete_100"
                reason = "target persisted; complete to 100%"
                pending_stage_target_key = ""
                pending_stage_count = 0
            else:
                desired = formal_weights
                action = "maintain_target"
        elif variant.mode == "hold_old_if_valid":
            if target_changed and current_top and _old_holding_valid(current_top, prices, date):
                desired = current_weights
                action = "hold_old_still_valid"
                reason = "old holding above ma20 and ret20 non-negative"
            else:
                desired = formal_weights
                action = "full_switch_100"
                reason = "old holding invalid or no old holding"

        equity_before = _account_value(account, prices, date, fallback=previous_equity)
        trades = _rebalance_to_weights(account=account, prices=prices, date=date, target_weights=desired, cost_model=cost_model, reason=f"{variant.variant_id}:{action}")
        trade_rows.extend({**trade, "variant_id": variant.variant_id, "signal_date": date, "execution_action_type": action, "execution_diagnostic_active_in_trade_decision": False} for trade in trades)
        equity_after = _account_value(account, prices, date, fallback=previous_equity)
        previous_equity = equity_after
        running_max = max(running_max, equity_after)
        actual_weights = _current_account_weights(account, prices, date, equity_after)
        daily_rows.append(
            {
                "variant_id": variant.variant_id,
                "date": date,
                "period": item.get("period", ""),
                "formal_target_weights": json.dumps(formal_weights, sort_keys=True),
                "execution_target_weights": json.dumps(desired, sort_keys=True),
                "top_holding": _top_holding(actual_weights) or "cash",
                "cash": round(account.cash, 2),
                "cash_weight": round(account.cash / equity_after, 8) if equity_after else 0.0,
                "position_weight_sum": round(sum(actual_weights.values()), 8),
                "weight_sum": round(sum(actual_weights.values()) + (account.cash / equity_after if equity_after else 0.0), 8),
                "portfolio_equity": round(equity_after, 2),
                "daily_return": round(equity_after / equity_before - 1, 8) if equity_before else 0.0,
                "drawdown": round(equity_after / running_max - 1, 8) if running_max else 0.0,
                "execution_action_type": action,
                "execution_reason": reason,
                "target_changed": target_changed,
                "execution_diagnostic_active_in_trade_decision": False,
            }
        )
        event_rows.append(
            {
                "variant_id": variant.variant_id,
                "date": date,
                "formal_target_key": formal_key or "cash",
                "previous_formal_target_key": last_formal_target_key or "cash",
                "target_changed": target_changed,
                "execution_action_type": action,
                "execution_reason": reason,
                "trade_count": len(trades),
                "active_in_trade_decision": False,
            }
        )
        last_formal_target_key = formal_key

    return pd.DataFrame(daily_rows), pd.DataFrame(trade_rows), pd.DataFrame(event_rows)


def _normalize_stream(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "period", "target_weights"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError("missing formal target stream columns: " + ",".join(sorted(missing)))
    output = frame.copy().fillna("")
    output["date_ts"] = pd.to_datetime(output["date"], errors="coerce")
    return output[output["date_ts"].notna()].sort_values("date_ts").reset_index(drop=True)


def _load_prices(frame: pd.DataFrame, cache_dir: Path) -> dict[str, pd.Series]:
    tickers: set[str] = set()
    for value in frame["target_weights"].tolist():
        tickers.update(_parse_weights(value))
    prices: dict[str, pd.Series] = {}
    for ticker in sorted(tickers):
        path = cache_dir / f"{ticker.replace('.', '_')}.csv"
        if not path.exists():
            path = cache_dir / f"{ticker}.csv"
        if path.exists():
            prices[ticker] = pd.to_numeric(load_price_csv(path)["adj_close"], errors="coerce").dropna()
    return prices


def _parse_weights(value: object) -> dict[str, float]:
    text = str(value or "").strip()
    if not text:
        return {}
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        return {}
    return _normalize_weights({str(k): float(v) for k, v in parsed.items() if str(k).strip() and float(v) > 0})


def _normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    cleaned = {ticker: max(0.0, float(weight)) for ticker, weight in weights.items() if ticker and float(weight) > 0}
    total = sum(cleaned.values())
    if total > 1.0:
        return {ticker: weight / total for ticker, weight in cleaned.items()}
    return cleaned


def _blend_current_and_target(current: dict[str, float], target: dict[str, float], target_fraction: float) -> dict[str, float]:
    residual = max(0.0, 1.0 - target_fraction)
    merged: dict[str, float] = {}
    for ticker, weight in current.items():
        merged[ticker] = merged.get(ticker, 0.0) + weight * residual
    for ticker, weight in target.items():
        merged[ticker] = merged.get(ticker, 0.0) + weight * target_fraction
    return _normalize_weights(merged)


def _scale_weights(weights: dict[str, float], scale: float) -> dict[str, float]:
    return {ticker: weight * scale for ticker, weight in weights.items()}


def _current_account_weights(account: LedgerAccount, prices: dict[str, pd.Series], date: str, equity: float) -> dict[str, float]:
    if equity <= 0:
        return {}
    ts = pd.Timestamp(date)
    result: dict[str, float] = {}
    for ticker, shares in account.positions.items():
        series = prices.get(ticker)
        if series is None or series.empty:
            continue
        valid = series.loc[series.index <= ts]
        if valid.empty:
            continue
        result[ticker] = shares * float(valid.iloc[-1]) / equity
    return result


def _top_holding(weights: dict[str, float]) -> str:
    return max(weights, key=weights.get) if weights else ""


def _target_key(weights: dict[str, float]) -> str:
    if not weights:
        return ""
    return "|".join(f"{ticker}:{weight:.6f}" for ticker, weight in sorted(weights.items()))


def _old_holding_valid(ticker: str, prices: dict[str, pd.Series], date: str) -> bool:
    series = prices.get(ticker)
    if series is None or series.empty:
        return False
    valid = series.loc[series.index <= pd.Timestamp(date)]
    if len(valid) < 20:
        return False
    close = float(valid.iloc[-1])
    ma20 = float(valid.tail(20).mean())
    ret20 = close / float(valid.iloc[-20]) - 1 if float(valid.iloc[-20]) else -1.0
    return close >= ma20 and ret20 >= 0


def _variant_matrix() -> pd.DataFrame:
    return pd.DataFrame([{**variant.__dict__, "active_in_trade_decision": False} for variant in VARIANTS])


def _period_performance(daily: pd.DataFrame) -> pd.DataFrame:
    periods = {
        "full": (None, None),
        "2022": ("2022-01-01", "2022-12-31"),
        "2023": ("2023-01-01", "2023-12-31"),
        "2024_now": ("2024-01-01", None),
        "2024_hard_gate": ("2024-01-01", "2024-12-31"),
    }
    rows: list[dict[str, Any]] = []
    frame = daily.copy()
    frame["date_ts"] = pd.to_datetime(frame["date"])
    for variant, group in frame.groupby("variant_id"):
        for label, (start, end) in periods.items():
            segment = group.copy()
            if start:
                segment = segment[segment["date_ts"] >= pd.Timestamp(start)]
            if end:
                segment = segment[segment["date_ts"] <= pd.Timestamp(end)]
            if segment.empty:
                rows.append({"variant_id": variant, "period_label": label, "status": "no_rows"})
                continue
            start_equity = float(segment["portfolio_equity"].iloc[0])
            final = float(segment["portfolio_equity"].iloc[-1])
            rows.append(
                {
                    "variant_id": variant,
                    "period_label": label,
                    "status": "completed",
                    "start_date": str(segment["date"].iloc[0]),
                    "end_date": str(segment["date"].iloc[-1]),
                    "start_equity": round(start_equity, 2),
                    "final_equity": round(final, 2),
                    "return_pct": round((final / start_equity - 1) * 100, 4) if start_equity else "",
                    "max_drawdown_pct": round(float(pd.to_numeric(segment["drawdown"], errors="coerce").min()) * 100, 4),
                    "active_in_trade_decision": False,
                }
            )
    return pd.DataFrame(rows)


def _risk_summary(daily: pd.DataFrame) -> pd.DataFrame:
    return daily.groupby("variant_id", as_index=False).agg(
        max_drawdown=("drawdown", "min"),
        average_cash_weight=("cash_weight", "mean"),
        target_change_days=("target_changed", "sum"),
    )


def _cost_summary(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(columns=["variant_id", "trade_rows", "total_turnover", "total_transaction_cost", "active_in_trade_decision"])
    grouped = trades.groupby("variant_id", as_index=False).agg(
        trade_rows=("action", "count"),
        total_turnover=("gross_amount", "sum"),
        total_transaction_cost=("transaction_cost", "sum"),
    )
    grouped["active_in_trade_decision"] = False
    return grouped


def _transition_stability(events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant, group in events.groupby("variant_id"):
        rows.append(
            {
                "variant_id": variant,
                "target_change_count": int(group["target_changed"].sum()),
                "trade_event_count": int((group["trade_count"] > 0).sum()),
                "no_action_count": int(group["execution_action_type"].eq("no_action").sum()),
                "partial_or_staged_count": int(group["execution_action_type"].astype(str).str.contains("partial|staged", regex=True).sum()),
                "active_in_trade_decision": False,
            }
        )
    return pd.DataFrame(rows)


def _exposure_integrity(daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant, group in daily.groupby("variant_id"):
        weights = pd.to_numeric(group["weight_sum"], errors="coerce")
        cash = pd.to_numeric(group["cash"], errors="coerce")
        rows.append(
            {
                "variant_id": variant,
                "negative_cash_count": int((cash < -0.01).sum()),
                "exposure_over_100_count": int((weights > 1.0001).sum()),
                "integrity_pass": bool((cash >= -0.01).all() and (weights <= 1.0001).all()),
                "active_in_trade_decision": False,
            }
        )
    return pd.DataFrame(rows)


def _baseline_comparison(performance: pd.DataFrame) -> pd.DataFrame:
    full = performance[performance["period_label"].eq("full")].copy()
    baseline = full[full["variant_id"].eq("full_switch_100_same_day_baseline")]
    if baseline.empty:
        return pd.DataFrame()
    base_return = float(baseline.iloc[0]["return_pct"])
    base_mdd = float(baseline.iloc[0]["max_drawdown_pct"])
    rows = []
    for _, row in full.iterrows():
        rows.append(
            {
                "variant_id": row["variant_id"],
                "return_delta_pp_vs_full_switch": round(float(row["return_pct"]) - base_return, 4),
                "mdd_delta_pp_vs_full_switch": round(float(row["max_drawdown_pct"]) - base_mdd, 4),
                "active_in_trade_decision": False,
            }
        )
    return pd.DataFrame(rows)


def _sizing_curve_report(performance: pd.DataFrame, costs: pd.DataFrame) -> pd.DataFrame:
    full = performance[performance["period_label"].eq("full")].copy()
    full["sizing_family"] = full["variant_id"].map(_sizing_family)
    full["switch_pct"] = full["variant_id"].map(_switch_pct)
    merged = full.merge(costs, on="variant_id", how="left")
    return merged[
        [
            "variant_id",
            "sizing_family",
            "switch_pct",
            "return_pct",
            "max_drawdown_pct",
            "trade_rows",
            "total_turnover",
            "total_transaction_cost",
        ]
    ].sort_values(["sizing_family", "switch_pct", "variant_id"])


def _sizing_family(variant_id: str) -> str:
    if variant_id.startswith("partial_switch_"):
        return "partial_switch_curve"
    if variant_id.startswith("cash_buffer_"):
        return "cash_buffer_sensitivity"
    if variant_id.startswith("staged_switch_"):
        return "staged_switch_sensitivity"
    if variant_id.startswith("full_switch_"):
        return "full_switch_baseline"
    return "other"


def _switch_pct(variant_id: str) -> int:
    parts = variant_id.split("_")
    for part in parts:
        if part.isdigit():
            return int(part)
    return 0


def _summary_markdown(performance: pd.DataFrame, comparison: pd.DataFrame) -> str:
    full = performance[performance["period_label"].eq("full")][["variant_id", "return_pct", "max_drawdown_pct"]]
    lines = [
        "# Execution position transition diagnostic",
        "",
        "本輸出回到換倉執行層主幹：是否動作、換倉比例、分批切換、保留舊持股與現金緩衝。這是 diagnostic，不啟用正式 execution layer。",
        "",
        "## Full period",
    ]
    for _, row in full.iterrows():
        lines.append(f"- {row['variant_id']}：return {row['return_pct']}%，MDD {row['max_drawdown_pct']}%")
    lines.extend(
        [
            "",
            "## 邊界",
            "- formal_model_changed=false",
            "- trade_decision_changed=false",
            "- formal_execution_layer_activated=false",
            "- uses_forward_return_as_rule=false",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run execution position transition diagnostic variants.")
    parser.add_argument("--formal-target-stream", default=DEFAULT_FORMAL_TARGET_STREAM)
    parser.add_argument("--price-cache-dir", default=DEFAULT_PRICE_CACHE_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    run_execution_position_transition_runner(
        formal_target_stream=args.formal_target_stream,
        price_cache_dir=args.price_cache_dir,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
