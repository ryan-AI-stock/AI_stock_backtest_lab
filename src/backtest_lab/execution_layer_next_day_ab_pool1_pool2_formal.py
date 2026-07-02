from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from backtest_lab.costs import COST_MODEL_VERSION, TaiwanCostModel
from backtest_lab.data import load_price_csv


DEFAULT_REVIEW_DIR = "outputs/execution_layer_review_pool1_pool2_formal_20260626"
DEFAULT_PRICE_CACHE_DIR = "backtest_cache/stock_pool_triad_v1_corrected"
DEFAULT_OUTPUT_DIR = "outputs/execution_layer_next_day_ab_pool1_pool2_formal_20260626"
INITIAL_CASH = 1_000_000.0
FORMAL_MODEL_TARGET = "combined_cap40_confirmation1_base"


@dataclass(frozen=True)
class VariantSpec:
    variant_id: str
    fill_delay_days: int
    minimum_hold_rows: int | None = None
    cooldown_after_exit_rows: int | None = None
    no_formal_target_policy: str = "hold_previous"
    description: str = ""


@dataclass
class Account:
    cash: float
    positions: dict[str, int]


VARIANTS = (
    VariantSpec("same_day_full_rotation_reference", 0, description="同日成交參考，不是新正式規則。"),
    VariantSpec("next_day_full_rotation", 1, description="訊號日後下一個交易日成交。"),
    VariantSpec("next_day_minimum_hold_2", 1, minimum_hold_rows=2, description="新 target 連續 2 個交易列才排入下一日成交；退出現金例外。"),
    VariantSpec("next_day_minimum_hold_3", 1, minimum_hold_rows=3, description="新 target 連續 3 個交易列才排入下一日成交；退出現金例外。"),
    VariantSpec("next_day_minimum_hold_5", 1, minimum_hold_rows=5, description="新 target 連續 5 個交易列才排入下一日成交；退出現金例外。"),
    VariantSpec("next_day_cooldown_after_exit_to_cash_2", 1, cooldown_after_exit_rows=2, description="退出現金後 2 個交易列內不重新進場。"),
    VariantSpec("next_day_cooldown_after_exit_to_cash_3", 1, cooldown_after_exit_rows=3, description="退出現金後 3 個交易列內不重新進場。"),
    VariantSpec("next_day_cooldown_after_exit_to_cash_5", 1, cooldown_after_exit_rows=5, description="退出現金後 5 個交易列內不重新進場。"),
)


def run_execution_layer_next_day_ab_pool1_pool2_formal(
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
        log("load_inputs", "started", str(review))
        stream = pd.read_csv(review / "formal_target_stream_adapter.csv").fillna("")
        _validate_stream(stream)
        frame = _normalize_stream(stream)

        log("load_prices", "started", str(price_cache_dir))
        prices = _load_prices(frame, Path(price_cache_dir))
        if not prices:
            raise ValueError("no prices loaded for next-day execution A/B")

        log("simulate_variants", "started", f"variants={len(VARIANTS)}")
        daily_frames: list[pd.DataFrame] = []
        trade_frames: list[pd.DataFrame] = []
        event_frames: list[pd.DataFrame] = []
        blocked_frames: list[pd.DataFrame] = []
        for variant in VARIANTS:
            daily, trades, events, blocked = _simulate_variant(frame, prices, variant, initial_cash)
            daily_frames.append(daily)
            trade_frames.append(trades)
            event_frames.append(events)
            blocked_frames.append(blocked)

        daily_ledger = pd.concat(daily_frames, ignore_index=True)
        trade_ledger = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()
        fill_events = pd.concat(event_frames, ignore_index=True) if event_frames else pd.DataFrame()
        blocked_events = pd.concat(blocked_frames, ignore_index=True) if blocked_frames else pd.DataFrame()

        log("build_reports", "started", "")
        variant_matrix = _variant_matrix()
        performance = _period_performance(daily_ledger)
        baseline_alignment = _baseline_alignment(frame, daily_ledger)
        ab_summary = _ab_summary(daily_ledger, fill_events, blocked_events)
        cost_turnover = _cost_turnover_summary(daily_ledger, trade_ledger)
        drawdown = _drawdown_summary(daily_ledger)
        readiness = _readiness_report(ab_summary, blocked_events)

        log("write_outputs", "started", "")
        variant_matrix.to_csv(output / "variant_parameter_matrix.csv", index=False, encoding="utf-8-sig")
        daily_ledger.to_csv(output / "next_day_fill_full_equity_ledger.csv", index=False, encoding="utf-8-sig")
        trade_ledger.to_csv(output / "next_day_fill_trade_ledger.csv", index=False, encoding="utf-8-sig")
        fill_events.to_csv(output / "fill_event_panel.csv", index=False, encoding="utf-8-sig")
        blocked_events.to_csv(output / "blocked_execution_events.csv", index=False, encoding="utf-8-sig")
        performance.to_csv(output / "period_performance_by_variant.csv", index=False, encoding="utf-8-sig")
        baseline_alignment.to_csv(output / "baseline_alignment.csv", index=False, encoding="utf-8-sig")
        ab_summary.to_csv(output / "minimum_hold_cooldown_ab_summary.csv", index=False, encoding="utf-8-sig")
        cost_turnover.to_csv(output / "cost_turnover_summary.csv", index=False, encoding="utf-8-sig")
        drawdown.to_csv(output / "drawdown_summary.csv", index=False, encoding="utf-8-sig")
        readiness.to_csv(output / "execution_ab_readiness_report.csv", index=False, encoding="utf-8-sig")
        (output / "execution_layer_next_day_ab_summary_zh.md").write_text(
            _summary_markdown(performance, ab_summary, readiness),
            encoding="utf-8",
        )
        manifest = {
            "schema_version": 1,
            "task_id": "TASK-BACKTEST-CORE-EXECUTION-LAYER-NEXT-DAY-AB-POOL1-POOL2-FORMAL-001",
            "status": "completed_diagnostic_ab",
            "formal_model_target": FORMAL_MODEL_TARGET,
            "formal_model_route": "pool1_primary_pool2_confirmation_cap40",
            "review_dir": str(review),
            "price_cache_dir": str(price_cache_dir),
            "start_date": str(frame["date"].iloc[0]) if not frame.empty else "",
            "latest_complete_common_date": str(frame["date"].iloc[-1]) if not frame.empty else "",
            "formal_model_changed": False,
            "trade_decision_changed": False,
            "active_in_trade_decision": False,
            "execution_diagnostic_active_in_trade_decision": False,
            "formal_execution_layer_activated": False,
            "pool3_shadow_used": False,
            "final_decision_label_used": False,
            "rr_partial_switch_used": False,
            "uses_forward_return_as_rule": False,
            "valuation_used": False,
            "h3_used": False,
            "same_day_and_next_day_mixed": False,
            "same_day_reference_max_abs_diff_vs_formal_stream": _alignment_max_diff(baseline_alignment),
            "outputs": {
                "variant_parameter_matrix": "variant_parameter_matrix.csv",
                "daily_ledger": "next_day_fill_full_equity_ledger.csv",
                "trade_ledger": "next_day_fill_trade_ledger.csv",
                "fill_event_panel": "fill_event_panel.csv",
                "blocked_events": "blocked_execution_events.csv",
                "period_performance": "period_performance_by_variant.csv",
                "baseline_alignment": "baseline_alignment.csv",
                "ab_summary": "minimum_hold_cooldown_ab_summary.csv",
                "readiness": "execution_ab_readiness_report.csv",
                "summary": "execution_layer_next_day_ab_summary_zh.md",
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
        pd.DataFrame([{"step": "run_execution_layer_next_day_ab_pool1_pool2_formal", "error": str(exc)}]).to_csv(
            output / "failed.csv", index=False, encoding="utf-8-sig"
        )
        log("failed", "failed", str(exc))
        raise


def _simulate_variant(
    frame: pd.DataFrame,
    prices: dict[str, pd.Series],
    variant: VariantSpec,
    initial_cash: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    account = Account(cash=float(initial_cash), positions={})
    cost_model = TaiwanCostModel()
    pending: list[dict[str, Any]] = []
    accepted_target: dict[str, float] = {}
    last_signal_target: dict[str, float] = {}
    consecutive_signal_rows = 0
    exit_to_cash_index: int | None = None
    running_max = float(initial_cash)
    previous_equity = float(initial_cash)
    daily_rows: list[dict[str, Any]] = []
    trade_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    blocked_rows: list[dict[str, Any]] = []

    for index, item in frame.iterrows():
        date = str(item["date"])
        due = [order for order in pending if int(order["fill_index"]) == index]
        pending = [order for order in pending if int(order["fill_index"]) != index]
        for order in due:
            trades, blocked = _execute_order(account, prices, date, order["target_weights"], cost_model, order["reason"])
            if blocked:
                blocked_rows.append(
                    {
                        "variant_id": variant.variant_id,
                        "signal_date": order["signal_date"],
                        "intended_fill_date": date,
                        "target_weights": json.dumps(order["target_weights"], sort_keys=True),
                        "blocked_reason": blocked,
                        "diagnostic_only": True,
                        "active_in_trade_decision": False,
                    }
                )
            else:
                accepted_target = dict(order["target_weights"])
                trade_rows.extend({**trade, "variant_id": variant.variant_id, "signal_date": order["signal_date"]} for trade in trades)

        signal_weights = _parse_weights(item.get("target_weights"))
        if signal_weights == last_signal_target:
            consecutive_signal_rows += 1
        else:
            consecutive_signal_rows = 1
        last_signal_target = dict(signal_weights)

        policy_weights, policy_blocked_reason = _policy_target_weights(
            signal_weights=signal_weights,
            accepted_target=accepted_target,
            variant=variant,
            consecutive_signal_rows=consecutive_signal_rows,
            row_index=index,
            exit_to_cash_index=exit_to_cash_index,
        )
        source_trade_signal = str(item.get("action", "")).strip() in {"buy", "switch"} or _number(item.get("turnover")) > 0
        target_changed = policy_weights != accepted_target
        pending_same_target = policy_weights == _pending_target(pending)
        should_schedule = (target_changed and not pending_same_target) or source_trade_signal
        if should_schedule:
            fill_index = index + variant.fill_delay_days
            if policy_blocked_reason:
                blocked_rows.append(
                    {
                        "variant_id": variant.variant_id,
                        "signal_date": date,
                        "intended_fill_date": "",
                        "target_weights": json.dumps(signal_weights, sort_keys=True),
                        "blocked_reason": policy_blocked_reason,
                        "diagnostic_only": True,
                        "active_in_trade_decision": False,
                    }
                )
            elif fill_index >= len(frame):
                blocked_rows.append(
                    {
                        "variant_id": variant.variant_id,
                        "signal_date": date,
                        "intended_fill_date": "",
                        "target_weights": json.dumps(policy_weights, sort_keys=True),
                        "blocked_reason": "missing_future_fill_row",
                        "diagnostic_only": True,
                        "active_in_trade_decision": False,
                    }
                )
            elif variant.fill_delay_days == 0:
                trades, blocked = _execute_order(account, prices, date, policy_weights, cost_model, "same_day_fill")
                if blocked:
                    blocked_rows.append(
                        {
                            "variant_id": variant.variant_id,
                            "signal_date": date,
                            "intended_fill_date": date,
                            "target_weights": json.dumps(policy_weights, sort_keys=True),
                            "blocked_reason": blocked,
                            "diagnostic_only": True,
                            "active_in_trade_decision": False,
                        }
                    )
                else:
                    accepted_target = dict(policy_weights)
                    trade_rows.extend({**trade, "variant_id": variant.variant_id, "signal_date": date} for trade in trades)
                    event_rows.append(_fill_event(variant, date, date, policy_weights, "same_day_fill", policy_blocked_reason))
                    if not policy_weights:
                        exit_to_cash_index = index
            else:
                fill_date = str(frame.iloc[fill_index]["date"])
                pending = [{"fill_index": fill_index, "signal_date": date, "target_weights": dict(policy_weights), "reason": "next_day_fill"}]
                event_rows.append(_fill_event(variant, date, fill_date, policy_weights, "next_day_fill", policy_blocked_reason))
                if not policy_weights:
                    exit_to_cash_index = index

        prices_today = _close_prices(set(account.positions), prices, date)
        equity = account.cash + sum(account.positions.get(ticker, 0) * prices_today.get(ticker, 0.0) for ticker in account.positions)
        if equity <= 0:
            equity = previous_equity
        running_max = max(running_max, equity)
        daily_return = equity / previous_equity - 1 if previous_equity else 0.0
        previous_equity = equity
        weights = _current_weights(account, prices, date, equity)
        daily_rows.append(
            {
                "variant_id": variant.variant_id,
                "date": date,
                "period": item.get("period", ""),
                "signal_target_weights": json.dumps(signal_weights, sort_keys=True),
                "accepted_target_weights": json.dumps(accepted_target, sort_keys=True),
                "pending_order_count": len(pending),
                "top_holding": _top_holding(account, prices, date) or "cash",
                "cash": round(account.cash, 2),
                "cash_weight": round(account.cash / equity, 8) if equity else 0.0,
                "position_weight_sum": round(sum(weights.values()), 8),
                "weight_sum": round(sum(weights.values()) + (account.cash / equity if equity else 0.0), 8),
                "portfolio_equity": round(equity, 2),
                "daily_return": round(float(daily_return), 8),
                "drawdown": round(equity / running_max - 1, 8) if running_max else 0.0,
                "consecutive_signal_rows": consecutive_signal_rows,
                "minimum_hold_rows": variant.minimum_hold_rows or "",
                "cooldown_after_exit_rows": variant.cooldown_after_exit_rows or "",
                "execution_diagnostic_active_in_trade_decision": False,
            }
        )

    return pd.DataFrame(daily_rows), pd.DataFrame(trade_rows), pd.DataFrame(event_rows), pd.DataFrame(blocked_rows)


def _policy_target_weights(
    *,
    signal_weights: dict[str, float],
    accepted_target: dict[str, float],
    variant: VariantSpec,
    consecutive_signal_rows: int,
    row_index: int,
    exit_to_cash_index: int | None,
) -> tuple[dict[str, float], str]:
    if not signal_weights:
        if variant.no_formal_target_policy == "exit_to_cash":
            return {}, "no_formal_target_exit_to_cash"
        return dict(accepted_target), "no_formal_target_hold_previous"
    if variant.minimum_hold_rows is not None and signal_weights != accepted_target:
        if consecutive_signal_rows < variant.minimum_hold_rows:
            return accepted_target, f"minimum_hold_{variant.minimum_hold_rows}_waiting"
    if variant.cooldown_after_exit_rows is not None and exit_to_cash_index is not None and signal_weights != accepted_target:
        rows_since_exit = row_index - exit_to_cash_index
        if rows_since_exit <= variant.cooldown_after_exit_rows:
            return accepted_target, f"cooldown_after_exit_{variant.cooldown_after_exit_rows}_waiting"
    return signal_weights, ""


def _execute_order(
    account: Account,
    prices: dict[str, pd.Series],
    date: str,
    target_weights: dict[str, float],
    cost_model: TaiwanCostModel,
    reason: str,
) -> tuple[list[dict[str, Any]], str]:
    trades: list[dict[str, Any]] = []
    tickers = set(account.positions) | set(target_weights)
    price_map = _close_prices(tickers, prices, date)
    missing = sorted(ticker for ticker in tickers if ticker and ticker not in price_map)
    if missing:
        return [], "missing_price:" + ",".join(missing)
    equity = account.cash + sum(account.positions.get(ticker, 0) * price_map.get(ticker, 0.0) for ticker in account.positions)
    if equity <= 0:
        return [], "non_positive_equity"
    normalized = _normalize_weights(target_weights)

    # Match the formal challenger ledger口徑: every execution event is a full
    # rebalance, so existing positions are sold first and target weights are
    # bought from the pre-trade equity budget.
    for ticker, shares in list(account.positions.items()):
        price = price_map.get(ticker, 0.0)
        if not price or shares <= 0:
            continue
        gross = shares * price
        breakdown = cost_model.sell_cost_breakdown(gross, _asset_type(ticker))
        cost = breakdown["total_transaction_cost"]
        account.cash += gross - cost
        account.positions[ticker] = 0
        account.positions.pop(ticker, None)
        trades.append(_trade_row(date, ticker, "sell", shares, price, gross, cost, account.cash, reason, breakdown))

    for ticker, weight in normalized.items():
        price = price_map.get(ticker, 0.0)
        if price <= 0:
            continue
        desired_value = equity * weight
        shares = int(max(0.0, desired_value) // price)
        while shares > 0:
            gross = shares * price
            cost = cost_model.buy_cost(gross)
            if gross + cost <= account.cash:
                break
            shares -= 1
        if shares <= 0:
            continue
        gross = shares * price
        breakdown = cost_model.buy_cost_breakdown(gross)
        cost = breakdown["total_transaction_cost"]
        account.cash -= gross + cost
        account.positions[ticker] = account.positions.get(ticker, 0) + shares
        trades.append(_trade_row(date, ticker, "buy", shares, price, gross, cost, account.cash, reason, breakdown))
    return trades, ""


def _validate_stream(frame: pd.DataFrame) -> None:
    required = {"date", "period", "target_weights"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"formal target stream missing columns: {missing}")


def _normalize_stream(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy().fillna("")
    output["date_ts"] = pd.to_datetime(output["date"], errors="coerce")
    output = output[output["date_ts"].notna()].sort_values("date_ts").reset_index(drop=True)
    return output


def _number(value: object) -> float:
    return float(pd.to_numeric(pd.Series([value]), errors="coerce").fillna(0.0).iloc[0])


def _variant_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "variant_id": variant.variant_id,
                "fill_delay_days": variant.fill_delay_days,
                "minimum_hold_rows": variant.minimum_hold_rows or "",
                "cooldown_after_exit_rows": variant.cooldown_after_exit_rows or "",
                "description": variant.description,
                "formal_model_changed": False,
                "trade_decision_changed": False,
                "active_in_trade_decision": False,
            }
            for variant in VARIANTS
        ]
    )


def _load_prices(frame: pd.DataFrame, price_cache_dir: Path) -> dict[str, pd.Series]:
    tickers: set[str] = set()
    for value in frame["target_weights"].tolist():
        tickers.update(_parse_weights(value))
    prices: dict[str, pd.Series] = {}
    for ticker in sorted(tickers):
        path = price_cache_dir / f"{ticker.replace('.', '_')}.csv"
        if not path.exists():
            path = price_cache_dir / f"{ticker}.csv"
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
    return _normalize_weights({str(key).strip(): float(weight) for key, weight in parsed.items() if str(key).strip() and float(weight) > 0})


def _normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    cleaned = {ticker: max(0.0, float(weight)) for ticker, weight in weights.items() if ticker and float(weight) > 0}
    total = sum(cleaned.values())
    if total > 1.0:
        return {ticker: weight / total for ticker, weight in cleaned.items()}
    return cleaned


def _pending_target(pending: list[dict[str, Any]]) -> dict[str, float]:
    return dict(pending[-1]["target_weights"]) if pending else {}


def _close_prices(tickers: set[str], prices: dict[str, pd.Series], date: str) -> dict[str, float]:
    output: dict[str, float] = {}
    ts = pd.Timestamp(date)
    for ticker in tickers:
        series = prices.get(ticker)
        if series is None or series.empty:
            continue
        valid = series.loc[series.index <= ts]
        if valid.empty:
            continue
        output[ticker] = float(valid.iloc[-1])
    return output


def _current_weights(account: Account, prices: dict[str, pd.Series], date: str, equity: float) -> dict[str, float]:
    if equity <= 0:
        return {}
    price_map = _close_prices(set(account.positions), prices, date)
    return {
        ticker: account.positions.get(ticker, 0) * price / equity
        for ticker, price in price_map.items()
        if account.positions.get(ticker, 0) > 0 and price > 0
    }


def _top_holding(account: Account, prices: dict[str, pd.Series], date: str) -> str:
    price_map = _close_prices(set(account.positions), prices, date)
    values = {ticker: account.positions.get(ticker, 0) * price for ticker, price in price_map.items()}
    return max(values.items(), key=lambda item: item[1])[0] if values else ""


def _asset_type(ticker: str) -> str:
    return "etf" if ticker in {"0050.TW", "00631L.TW"} else "stock"


def _trade_row(
    date: str,
    ticker: str,
    action: str,
    shares: int,
    price: float,
    gross: float,
    cost: float,
    cash_after: float,
    reason: str,
    breakdown: dict[str, Any] | None = None,
) -> dict[str, Any]:
    parts = breakdown or {}
    return {
        "date": date,
        "ticker": ticker,
        "action": action,
        "shares": int(shares),
        "price": round(float(price), 6),
        "gross_amount": round(float(gross), 2),
        "transaction_cost": round(float(cost), 2),
        "buy_fee": round(float(parts.get("buy_fee", cost if action == "buy" else 0)), 2),
        "sell_fee": round(float(parts.get("sell_fee", cost if action == "sell" else 0)), 2),
        "securities_transaction_tax": round(float(parts.get("securities_transaction_tax", 0)), 2),
        "total_transaction_cost": round(float(parts.get("total_transaction_cost", cost)), 2),
        "cost_model_version": COST_MODEL_VERSION,
        "cash_after": round(float(cash_after), 2),
        "reason": reason,
        "execution_diagnostic_active_in_trade_decision": False,
    }


def _fill_event(variant: VariantSpec, signal_date: str, fill_date: str, weights: dict[str, float], fill_mode: str, blocked_reason: str) -> dict[str, Any]:
    return {
        "variant_id": variant.variant_id,
        "signal_date": signal_date,
        "fill_date": fill_date,
        "target_weights": json.dumps(weights, sort_keys=True),
        "fill_delay_days": variant.fill_delay_days,
        "minimum_hold_rows": variant.minimum_hold_rows or "",
        "cooldown_after_exit_rows": variant.cooldown_after_exit_rows or "",
        "fill_mode": fill_mode,
        "blocked_reason": blocked_reason,
        "diagnostic_only": True,
        "active_in_trade_decision": False,
    }


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
            end_equity = float(segment["portfolio_equity"].iloc[-1])
            rows.append(
                {
                    "variant_id": variant,
                    "period_label": label,
                    "status": "completed",
                    "start_date": str(segment["date"].iloc[0]),
                    "end_date": str(segment["date"].iloc[-1]),
                    "start_equity": round(start_equity, 2),
                    "final_equity": round(end_equity, 2),
                    "return_pct": round((end_equity / start_equity - 1) * 100, 4) if start_equity else "",
                    "max_drawdown_pct": round(float(pd.to_numeric(segment["drawdown"], errors="coerce").min()) * 100, 4),
                    "trade_signal_rows": int((segment["pending_order_count"] > 0).sum()),
                    "diagnostic_only": True,
                    "active_in_trade_decision": False,
                }
            )
    return pd.DataFrame(rows)


def _baseline_alignment(stream: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    if "equity" not in stream.columns:
        return pd.DataFrame(
            [
                {
                    "alignment_state": "blocked_missing_formal_stream_equity",
                    "max_abs_diff": "",
                    "final_equity_diff": "",
                    "row_count": 0,
                    "diagnostic_only": True,
                    "active_in_trade_decision": False,
                }
            ]
        )
    reference = daily[daily["variant_id"] == "same_day_full_rotation_reference"][["date", "portfolio_equity"]].copy()
    expected = stream[["date", "equity"]].copy()
    merged = expected.merge(reference, on="date", how="inner")
    if merged.empty:
        return pd.DataFrame(
            [
                {
                    "alignment_state": "blocked_no_overlap",
                    "max_abs_diff": "",
                    "final_equity_diff": "",
                    "row_count": 0,
                    "diagnostic_only": True,
                    "active_in_trade_decision": False,
                }
            ]
        )
    diff = pd.to_numeric(merged["portfolio_equity"], errors="coerce") - pd.to_numeric(merged["equity"], errors="coerce")
    max_abs = float(diff.abs().max())
    final_diff = float(diff.iloc[-1])
    return pd.DataFrame(
        [
            {
                "alignment_state": "passed" if max_abs <= 0.01 else "diff_detected",
                "max_abs_diff": round(max_abs, 6),
                "final_equity_diff": round(final_diff, 6),
                "row_count": int(len(merged)),
                "diagnostic_only": True,
                "active_in_trade_decision": False,
            }
        ]
    )


def _alignment_max_diff(alignment: pd.DataFrame) -> float | str:
    if alignment.empty or "max_abs_diff" not in alignment.columns:
        return ""
    value = pd.to_numeric(alignment["max_abs_diff"], errors="coerce")
    return round(float(value.iloc[0]), 6) if value.notna().any() else ""


def _ab_summary(daily: pd.DataFrame, fill_events: pd.DataFrame, blocked_events: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for variant, group in daily.groupby("variant_id"):
        fills = fill_events[fill_events["variant_id"].eq(variant)] if not fill_events.empty else pd.DataFrame()
        blocked = blocked_events[blocked_events["variant_id"].eq(variant)] if not blocked_events.empty else pd.DataFrame()
        rows.append(
            {
                "variant_id": variant,
                "fill_event_count": int(len(fills)),
                "blocked_event_count": int(len(blocked)),
                "final_equity": round(float(group["portfolio_equity"].iloc[-1]), 2),
                "max_drawdown_pct": round(float(pd.to_numeric(group["drawdown"], errors="coerce").min()) * 100, 4),
                "average_cash_weight": round(float(pd.to_numeric(group["cash_weight"], errors="coerce").mean()), 6),
                "max_pending_order_count": int(pd.to_numeric(group["pending_order_count"], errors="coerce").max()),
                "ready_for_formal_activation": False,
                "diagnostic_only": True,
                "active_in_trade_decision": False,
            }
        )
    return pd.DataFrame(rows)


def _cost_turnover_summary(daily: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for variant, group in trades.groupby("variant_id") if not trades.empty else []:
        rows.append(
            {
                "variant_id": variant,
                "trade_rows": int(len(group)),
                "total_turnover": round(float(pd.to_numeric(group["gross_amount"], errors="coerce").fillna(0.0).sum()), 2),
                "total_transaction_cost": round(float(pd.to_numeric(group["transaction_cost"], errors="coerce").fillna(0.0).sum()), 2),
                "diagnostic_only": True,
                "active_in_trade_decision": False,
            }
        )
    return pd.DataFrame(rows)


def _drawdown_summary(daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant, group in daily.groupby("variant_id"):
        rows.append(
            {
                "variant_id": variant,
                "max_drawdown_pct": round(float(pd.to_numeric(group["drawdown"], errors="coerce").min()) * 100, 4),
                "final_equity": round(float(group["portfolio_equity"].iloc[-1]), 2),
                "diagnostic_only": True,
                "active_in_trade_decision": False,
            }
        )
    return pd.DataFrame(rows)


def _readiness_report(ab_summary: pd.DataFrame, blocked_events: pd.DataFrame) -> pd.DataFrame:
    blocked_count = len(blocked_events) if not blocked_events.empty else 0
    return pd.DataFrame(
        [
            {
                "readiness_item": "next_day_fill_full_equity_ledger",
                "readiness_state": "completed_diagnostic" if blocked_count == 0 else "completed_with_blocked_events",
                "blocked_event_count": int(blocked_count),
                "formal_activation_ready": False,
                "next_step": "experiments_validate_ab_variants",
                "diagnostic_only": True,
                "active_in_trade_decision": False,
            }
        ]
    )


def _summary_markdown(performance: pd.DataFrame, ab_summary: pd.DataFrame, readiness: pd.DataFrame) -> str:
    main = performance[(performance["variant_id"] == "next_day_full_rotation") & (performance["period_label"] == "full")]
    ref = performance[(performance["variant_id"] == "same_day_full_rotation_reference") & (performance["period_label"] == "full")]
    main_row = main.iloc[0].to_dict() if not main.empty else {}
    ref_row = ref.iloc[0].to_dict() if not ref.empty else {}
    return "\n".join(
        [
            "# Pool1+Pool2 換倉執行層 next-day / minimum-hold A/B 診斷",
            "",
            "本輸出基於 `combined_cap40_confirmation1_base` formal target stream。所有結果仍是 execution diagnostic，不改正式 selector、formal target 或正式交易行為。",
            "",
            "## 重點",
            f"- same-day reference full return：{ref_row.get('return_pct', '')}%",
            f"- next-day full rotation full return：{main_row.get('return_pct', '')}%",
            f"- next-day full rotation MDD：{main_row.get('max_drawdown_pct', '')}%",
            f"- readiness：{readiness.iloc[0]['readiness_state'] if not readiness.empty else ''}",
            "",
            "## 邊界",
            "- formal_model_changed=false",
            "- trade_decision_changed=false",
            "- active_in_trade_decision=false",
            "- forward return 未作規則；Pool3、final decision label、RR partial switch、valuation/H3 均未使用。",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run next-day fill and min-hold/cooldown A/B diagnostics for the absorbed Pool1+Pool2 target stream.")
    parser.add_argument("--review-dir", default=DEFAULT_REVIEW_DIR)
    parser.add_argument("--price-cache-dir", default=DEFAULT_PRICE_CACHE_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    run_execution_layer_next_day_ab_pool1_pool2_formal(
        review_dir=args.review_dir,
        price_cache_dir=args.price_cache_dir,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
