from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from backtest_lab.costs import COST_MODEL_VERSION, cost_model_metadata
from backtest_lab.execution_layer_next_day_ab_pool1_pool2_formal import (
    INITIAL_CASH,
    VariantSpec as ExecutionVariantSpec,
    _simulate_variant as simulate_next_day_variant,
)
from backtest_lab.frozen_strategy_engine import load_frozen_strategy_context_from_cache, simulate_frozen_baseline
from backtest_lab.strategies import previous_available_date


MODEL_ID = "best_v20260605"
SELECTOR_ID = "frozen_cycle_proven_top1_v1"
DEFAULT_PRICE_CACHE_DIR = "backtest_cache/stock_pool_observations"
DEFAULT_OUTPUT_DIR = "outputs/previous_best_next_day_replay_20260630"
DEFAULT_SIGNAL_START = "2024-01-02"


def run_previous_best_next_day_replay(
    *,
    price_cache_dir: str | Path = DEFAULT_PRICE_CACHE_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    signal_start_date: str = DEFAULT_SIGNAL_START,
    signal_end_date: str | None = None,
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
        log("load_frozen_context", "started", str(price_cache_dir))
        context = load_frozen_strategy_context_from_cache(cache_dir=price_cache_dir)
        common_dates = _common_dates(context.prices_by_ticker)
        window = _resolve_signal_window(common_dates, signal_start_date, signal_end_date)

        log("simulate_frozen_baseline", "started", f"{window['execution_start']}..{window['execution_end']}")
        result = simulate_frozen_baseline(
            context=context,
            start_date=str(window["execution_start"]),
            end_date=str(window["execution_end"]),
            initial_cash=initial_cash,
            name=SELECTOR_ID,
        )

        log("build_daily_target_stream", "started", "")
        target_stream = _build_daily_target_stream(
            result.equity_curve,
            context.prices_by_ticker,
            signal_start=window["signal_start"],
            signal_end=window["signal_end"],
        )
        if target_stream.empty:
            raise ValueError("previous best target stream is empty")
        execution_frame = _execution_frame_from_target_stream(
            target_stream,
            terminal_execution_date=window["terminal_execution_date"],
        )

        log("simulate_next_day_ledger", "started", "")
        prices = {
            ticker: pd.to_numeric(frame["adj_close"], errors="coerce").dropna()
            for ticker, frame in context.prices_by_ticker.items()
        }
        daily, trades, events, blocked = simulate_next_day_variant(
            execution_frame,
            prices,
            ExecutionVariantSpec(
                "previous_best_next_day_full_rotation",
                1,
                description="previous best frozen baseline target stream with next-day full-rotation execution",
            ),
            initial_cash,
        )
        daily = _normalize_daily_ledger(daily)
        trades = _normalize_trade_ledger(trades)
        blockers = _data_blockers(context.prices_by_ticker, target_stream, blocked)
        performance = _period_performance(daily)

        log("write_outputs", "started", "")
        target_stream.to_csv(output / "previous_best_daily_target_stream.csv", index=False, encoding="utf-8-sig")
        daily.to_csv(output / "previous_best_next_day_daily_ledger.csv", index=False, encoding="utf-8-sig")
        trades.to_csv(output / "previous_best_next_day_trade_ledger.csv", index=False, encoding="utf-8-sig")
        events.to_csv(output / "previous_best_next_day_fill_events.csv", index=False, encoding="utf-8-sig")
        blocked.to_csv(output / "previous_best_next_day_blocked_events.csv", index=False, encoding="utf-8-sig")
        blockers.to_csv(output / "data_blockers.csv", index=False, encoding="utf-8-sig")
        performance.to_csv(output / "previous_best_next_day_period_performance.csv", index=False, encoding="utf-8-sig")
        (output / "final_summary_zh.md").write_text(
            _summary_markdown(target_stream, daily, trades, blockers, performance),
            encoding="utf-8",
        )
        manifest = {
            "schema_version": 1,
            "task_id": "TASK-BACKTEST-CORE-PREVIOUS-BEST-NEXT-DAY-REPLAY-RUNNER-001",
            "status": "completed_next_day_replay_package" if blockers.empty else "completed_with_blockers",
            "model_id": MODEL_ID,
            "selector_id": SELECTOR_ID,
            "execution_basis": "next_day",
            "initial_capital_twd": float(initial_cash),
            "taiwan_cost_model": True,
            "cost_model_version": COST_MODEL_VERSION,
            "cost_model": cost_model_metadata(),
            "required_cost_fields": [
                "buy_fee",
                "sell_fee",
                "securities_transaction_tax",
                "total_transaction_cost",
            ],
            "price_cache_dir": str(price_cache_dir),
            "signal_start_date": str(target_stream["signal_date"].iloc[0]),
            "signal_end_date": str(target_stream["signal_date"].iloc[-1]),
            "execution_start_date": str(window["execution_start"].date()),
            "execution_end_date": str(window["execution_end"].date()),
            "latest_price_common_date": str(window["latest_price_common_date"].date()),
            "latest_complete_signal_date": str(window["signal_end"].date()),
            "same_day_result_not_used_as_next_day_proof": True,
            "production_grade_next_day_ledger": True,
            "formal_model_changed": False,
            "trade_decision_changed": False,
            "active_in_trade_decision": False,
            "data_blocker_count": int(len(blockers)),
            "outputs": {
                "daily_target_stream": "previous_best_daily_target_stream.csv",
                "daily_ledger": "previous_best_next_day_daily_ledger.csv",
                "trade_ledger": "previous_best_next_day_trade_ledger.csv",
                "data_blockers": "data_blockers.csv",
                "summary": "final_summary_zh.md",
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
        pd.DataFrame([{"step": "run_previous_best_next_day_replay", "error": str(exc)}]).to_csv(
            output / "failed.csv", index=False, encoding="utf-8-sig"
        )
        log("failed", "failed", str(exc))
        raise


def _common_dates(prices_by_ticker: dict[str, pd.DataFrame]) -> list[pd.Timestamp]:
    common: set[pd.Timestamp] | None = None
    for frame in prices_by_ticker.values():
        dates = set(pd.to_datetime(frame.index).normalize())
        common = dates if common is None else common & dates
    return sorted(common or set())


def _resolve_signal_window(
    common_dates: list[pd.Timestamp],
    signal_start_date: str,
    signal_end_date: str | None,
) -> dict[str, pd.Timestamp]:
    if len(common_dates) < 3:
        raise ValueError("not enough common price dates for next-day replay")
    requested_start = pd.Timestamp(signal_start_date)
    possible_starts = [date for date in common_dates[:-1] if date >= requested_start]
    if not possible_starts:
        raise ValueError(f"no common signal date on or after {signal_start_date}")
    signal_start = possible_starts[0]
    requested_end = pd.Timestamp(signal_end_date) if signal_end_date else common_dates[-2]
    possible_ends = [date for date in common_dates[:-1] if signal_start <= date <= requested_end]
    if not possible_ends:
        raise ValueError(f"no complete signal date through {requested_end.date()}")
    signal_end = possible_ends[-1]
    start_index = common_dates.index(signal_start)
    end_index = common_dates.index(signal_end)
    return {
        "signal_start": signal_start,
        "signal_end": signal_end,
        "execution_start": common_dates[start_index + 1],
        "execution_end": common_dates[end_index + 1],
        "terminal_execution_date": common_dates[end_index + 1],
        "latest_price_common_date": common_dates[-1],
    }


def _build_daily_target_stream(
    equity_curve: pd.DataFrame,
    prices_by_ticker: dict[str, pd.DataFrame],
    *,
    signal_start: pd.Timestamp,
    signal_end: pd.Timestamp,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    previous_weights: dict[str, float] = {}
    for trade_date, row in equity_curve.iterrows():
        execution_date = pd.Timestamp(trade_date).normalize()
        signal_date = previous_available_date(prices_by_ticker, execution_date).normalize()
        if signal_date < signal_start or signal_date > signal_end:
            continue
        ticker = str(row.get("current_ticker", "") or "").strip()
        raw_exposure = float(pd.to_numeric(pd.Series([row.get("current_exposure", 0.0)]), errors="coerce").fillna(0.0).iloc[0])
        target_exposure = _intended_policy_exposure(ticker, raw_exposure)
        weights = {} if target_exposure <= 0 else {ticker: target_exposure}
        action = _action_from_weights(previous_weights, weights)
        rows.append(
            {
                "signal_date": signal_date.strftime("%Y-%m-%d"),
                "date": signal_date.strftime("%Y-%m-%d"),
                "next_trade_date": execution_date.strftime("%Y-%m-%d"),
                "execution_date": execution_date.strftime("%Y-%m-%d"),
                "model_id": MODEL_ID,
                "selector_id": SELECTOR_ID,
                "target_ticker": next(iter(weights), ""),
                "target_weights": json.dumps(weights, sort_keys=True),
                "cash_no_target_state": "target" if weights else "cash_or_no_target",
                "raw_decision_before_execution": _raw_decision(row, execution_date),
                "action": action,
                "turnover": 1.0 if action in {"buy", "sell", "switch"} else 0.0,
                "period": _period_label(signal_date),
                "target_exposure": round(sum(weights.values()), 8),
                "raw_current_exposure_after_strategy_execution": round(raw_exposure, 8),
                "regime": str(row.get("regime", "")),
                "mode": str(row.get("mode", "")),
                "risk_off_active": bool(row.get("risk_off_active", False)),
                "attack_gate_active": bool(row.get("attack_gate_active", False)),
                "taiwan_cost_model": True,
                "initial_capital_twd": INITIAL_CASH,
                "same_day_result_not_used_as_next_day_proof": True,
            }
        )
        previous_weights = weights
    return pd.DataFrame(rows)


def _execution_frame_from_target_stream(target_stream: pd.DataFrame, terminal_execution_date: pd.Timestamp) -> pd.DataFrame:
    frame = target_stream.copy()
    terminal = frame.iloc[-1].copy()
    terminal["signal_date"] = terminal_execution_date.strftime("%Y-%m-%d")
    terminal["date"] = terminal_execution_date.strftime("%Y-%m-%d")
    terminal["next_trade_date"] = ""
    terminal["execution_date"] = ""
    terminal["action"] = "hold"
    terminal["turnover"] = 0.0
    terminal["period"] = _period_label(terminal_execution_date)
    terminal["is_terminal_execution_row"] = True
    frame["is_terminal_execution_row"] = False
    return pd.concat([frame, pd.DataFrame([terminal])], ignore_index=True)


def _action_from_weights(previous: dict[str, float], current: dict[str, float]) -> str:
    if previous == current:
        return "hold"
    if not previous and current:
        return "buy"
    if previous and not current:
        return "sell"
    return "switch"


def _intended_policy_exposure(ticker: str, raw_exposure: float) -> float:
    if ticker in {"", "cash", "None"} or raw_exposure <= 0.02:
        return 0.0
    # frozen_cycle_proven_top1_v1 has two non-cash policy exposure states:
    # 25% risk-off/preproof exposure and 100% normal target exposure. The raw
    # equity curve exposure drifts with cash rounding and dividends, so using it
    # directly would create artificial daily rebalances in the next-day ledger.
    if raw_exposure <= 0.375:
        return 0.25
    return 1.0


def _raw_decision(row: pd.Series, execution_date: pd.Timestamp) -> str:
    payload = {
        "execution_date": execution_date.strftime("%Y-%m-%d"),
        "current_ticker_after_raw_strategy_execution": str(row.get("current_ticker", "")),
        "current_exposure_after_raw_strategy_execution": float(row.get("current_exposure", 0.0)),
        "regime": str(row.get("regime", "")),
        "mode": str(row.get("mode", "")),
        "risk_off_active": bool(row.get("risk_off_active", False)),
        "attack_gate_active": bool(row.get("attack_gate_active", False)),
        "attack_gate_ever_activated": bool(row.get("attack_gate_ever_activated", False)),
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _period_label(date: pd.Timestamp) -> str:
    return str(pd.Timestamp(date).year)


def _normalize_daily_ledger(daily: pd.DataFrame) -> pd.DataFrame:
    output = daily.copy()
    output["model_id"] = MODEL_ID
    output["selector_id"] = SELECTOR_ID
    output["execution_basis"] = "next_day"
    output["signal_date"] = output["date"]
    output["initial_capital_twd"] = INITIAL_CASH
    output["taiwan_cost_model"] = True
    output["same_day_result_not_used_as_next_day_proof"] = True
    output["formal_model_changed"] = False
    output["trade_decision_changed"] = False
    return output


def _normalize_trade_ledger(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(
            columns=[
                "model_id",
                "selector_id",
                "execution_basis",
                "signal_date",
                "execution_date",
                "ticker",
                "action",
                "shares",
                "price",
                "gross_amount",
                "transaction_cost",
                "cash_after",
                "reason",
            ]
        )
    output = trades.copy()
    output["model_id"] = MODEL_ID
    output["selector_id"] = SELECTOR_ID
    output["execution_basis"] = "next_day"
    output["execution_date"] = output["date"]
    output["taiwan_cost_model"] = True
    output["initial_capital_twd"] = INITIAL_CASH
    output["same_day_result_not_used_as_next_day_proof"] = True
    output["formal_model_changed"] = False
    output["trade_decision_changed"] = False
    return output


def _data_blockers(
    prices_by_ticker: dict[str, pd.DataFrame],
    target_stream: pd.DataFrame,
    blocked_execution: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for ticker, frame in prices_by_ticker.items():
        if frame.empty:
            rows.append({"blocker": "empty_price_frame", "ticker": ticker, "severity": "blocking", "detail": ""})
    if target_stream.empty:
        rows.append({"blocker": "empty_daily_target_stream", "ticker": "", "severity": "blocking", "detail": ""})
    if not blocked_execution.empty:
        for _, item in blocked_execution.iterrows():
            rows.append(
                {
                    "blocker": "blocked_next_day_fill",
                    "ticker": "",
                    "severity": "blocking",
                    "detail": json.dumps(item.to_dict(), ensure_ascii=False, default=str),
                }
            )
    return pd.DataFrame(rows, columns=["blocker", "ticker", "severity", "detail"])


def _period_performance(daily: pd.DataFrame) -> pd.DataFrame:
    periods = {
        "full": (None, None),
        "2024_now": ("2024-01-01", None),
        "2024": ("2024-01-01", "2024-12-31"),
        "2025": ("2025-01-01", "2025-12-31"),
        "2026_ytd": ("2026-01-01", None),
    }
    frame = daily.copy()
    frame["date_ts"] = pd.to_datetime(frame["date"])
    rows: list[dict[str, Any]] = []
    for label, (start, end) in periods.items():
        segment = frame.copy()
        if start:
            segment = segment[segment["date_ts"] >= pd.Timestamp(start)]
        if end:
            segment = segment[segment["date_ts"] <= pd.Timestamp(end)]
        if segment.empty:
            rows.append({"period_label": label, "status": "no_rows"})
            continue
        start_equity = float(segment["portfolio_equity"].iloc[0])
        final_equity = float(segment["portfolio_equity"].iloc[-1])
        rows.append(
            {
                "period_label": label,
                "status": "completed",
                "start_date": str(segment["date"].iloc[0]),
                "end_date": str(segment["date"].iloc[-1]),
                "start_equity": round(start_equity, 2),
                "final_equity": round(final_equity, 2),
                "return_pct": round((final_equity / start_equity - 1) * 100, 4) if start_equity else "",
                "max_drawdown_pct": round(float(pd.to_numeric(segment["drawdown"], errors="coerce").min()) * 100, 4),
                "trade_signal_rows": int((pd.to_numeric(segment.get("pending_order_count", 0), errors="coerce").fillna(0) > 0).sum()),
            }
        )
    return pd.DataFrame(rows)


def _summary_markdown(
    target_stream: pd.DataFrame,
    daily: pd.DataFrame,
    trades: pd.DataFrame,
    blockers: pd.DataFrame,
    performance: pd.DataFrame,
) -> str:
    full = performance.loc[performance["period_label"] == "full"].iloc[0].to_dict() if not performance.empty else {}
    trade_cost = float(pd.to_numeric(trades.get("transaction_cost", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum()) if not trades.empty else 0.0
    return "\n".join(
        [
            "# Previous best next-day replay package",
            "",
            f"- 模型：`{MODEL_ID}` / `{SELECTOR_ID}`。",
            "- 口徑：先重建 daily target stream，再用 production-grade next-day full-rotation ledger 重算，不拿 same-day 當 next-day 證據。",
            f"- 訊號區間：{target_stream['signal_date'].iloc[0]} 到 {target_stream['signal_date'].iloc[-1]}。",
            f"- 每日權益區間：{daily['date'].iloc[0]} 到 {daily['date'].iloc[-1]}。",
            f"- full return：{full.get('return_pct', '')}%；MDD：{full.get('max_drawdown_pct', '')}%。",
            f"- trade rows：{len(trades)}；transaction cost：{round(trade_cost, 2)} TWD。",
            f"- data blockers：{len(blockers)}。",
            "",
            "此輸出只供 Experiments 做 current formal vs previous best apples-to-apples 驗收；不改正式日報模型，也不代表 previous best 已回退上線。",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build previous best daily target stream and next-day replay ledgers.")
    parser.add_argument("--price-cache-dir", default=DEFAULT_PRICE_CACHE_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--signal-start-date", default=DEFAULT_SIGNAL_START)
    parser.add_argument("--signal-end-date", default=None)
    parser.add_argument("--initial-cash", type=float, default=INITIAL_CASH)
    args = parser.parse_args()
    output = run_previous_best_next_day_replay(
        price_cache_dir=args.price_cache_dir,
        output_dir=args.output_dir,
        signal_start_date=args.signal_start_date,
        signal_end_date=args.signal_end_date,
        initial_cash=args.initial_cash,
    )
    print(f"OUTPUT_DIR={output.resolve()}")


if __name__ == "__main__":
    main()
