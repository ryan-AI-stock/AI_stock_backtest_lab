"""Build the canonical current-formal next-day baseline ledger.

This runner is baseline governance only.  It replays the current formal target
stream exactly as a next-day execution ledger with CASH as cash, and writes
period reset / continuous slice summaries so later diagnostics stop mixing
different equity bases.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from backtest_lab.costs import COST_MODEL_VERSION, TaiwanCostModel
from backtest_lab.fallback_boundary_00631l_except_bear_cash_contract import (
    DEFAULT_2022_LATEST_FORMAL_STREAM,
    DEFAULT_FORMAL_STREAM,
    _canonical_target,
)


TASK_ID = "TASK-BACKTEST-CORE-CANONICAL-FORMAL-BASELINE-LEDGER-001"
EXPERIMENTS_TASK_ID = "TASK-BACKTEST-EXPERIMENTS-CANONICAL-FORMAL-BASELINE-LEDGER-VALIDATION-001"
DEFAULT_OUTPUT_DIR = Path("outputs/canonical_formal_baseline_ledger_20260705")
PRICE_CACHE_DIR = Path("backtest_cache/stock_pool_observations")
INITIAL_EQUITY = 1_000_000.0
BENCHMARK_TICKERS = ["0050.TW", "00631L.TW"]
PERIOD_CONTRACT = [
    {"period_label": "P1", "requested_start": "2015-01-02", "requested_end": "2022-12-29"},
    {"period_label": "P2", "requested_start": "2023-01-02", "requested_end": "2026-06-30"},
    {"period_label": "2024_latest", "requested_start": "2024-01-02", "requested_end": "2026-06-30"},
    {"period_label": "2026YTD", "requested_start": "2026-01-02", "requested_end": "2026-06-30"},
]


def run_canonical_formal_baseline_ledger(
    *,
    repo_root: str | Path = ".",
    formal_streams: list[str | Path] | None = None,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    price_cache_dir: str | Path = PRICE_CACHE_DIR,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    streams = formal_streams or [DEFAULT_FORMAL_STREAM, DEFAULT_2022_LATEST_FORMAL_STREAM]
    resolved_streams = [_resolve(root, path) for path in streams]
    output = _resolve(root, output_dir)
    output.mkdir(parents=True, exist_ok=True)
    price_dir = _resolve(root, price_cache_dir)

    formal = _load_canonical_formal_streams(resolved_streams)
    formal = _prepare_formal_stream(formal)
    formal = _apply_execution_calendar(formal, price_dir)
    prices, price_sources = _load_prices(price_dir, sorted(set(formal["formal_target"]) | set(BENCHMARK_TICKERS)))
    daily, trades = _simulate_next_day_ledger(formal, prices, initial_equity=INITIAL_EQUITY)
    period_reset = _period_reset_summary(formal, prices)
    continuous = _continuous_slice_summary(daily)
    benchmark = _benchmark_summary(prices, daily)
    mapping_audit = _no_target_mapping_audit(formal, resolved_streams)
    cost_summary = _cost_trade_summary(daily, trades)
    period_validation = _period_contract_validation(formal, daily, prices, price_sources)
    reconciliation = _prior_number_reconciliation()
    future = _future_data_audit(formal, daily)

    daily.to_csv(output / "canonical_formal_daily_ledger.csv", index=False, encoding="utf-8-sig")
    period_reset.to_csv(output / "canonical_period_reset_summary.csv", index=False, encoding="utf-8-sig")
    continuous.to_csv(output / "canonical_continuous_slice_summary.csv", index=False, encoding="utf-8-sig")
    benchmark.to_csv(output / "canonical_benchmark_0050_00631L_summary.csv", index=False, encoding="utf-8-sig")
    mapping_audit.to_csv(output / "canonical_no_target_mapping_audit.csv", index=False, encoding="utf-8-sig")
    cost_summary.to_csv(output / "canonical_cost_trade_summary.csv", index=False, encoding="utf-8-sig")
    period_validation.to_csv(output / "canonical_period_contract_validation.csv", index=False, encoding="utf-8-sig")
    reconciliation.to_csv(output / "canonical_reconciliation_to_prior_numbers.csv", index=False, encoding="utf-8-sig")
    future.to_csv(output / "future_data_audit.csv", index=False, encoding="utf-8-sig")

    future_count = int(future["future_data_violation"].sum()) if len(future) else 0
    missing_price_rows = int(daily["price_ready"].eq(False).sum())
    manifest: dict[str, Any] = {
        "task_id": TASK_ID,
        "status": "completed_canonical_formal_baseline_ledger",
        "output_dir": str(output),
        "source_formal_streams": [str(path) for path in resolved_streams],
        "formal_stream_rows": int(len(formal)),
        "canonical_daily_ledger_rows": int(len(daily)),
        "trade_rows": int(len(trades)),
        "execution_basis": "next_day",
        "canonical_current_formal_mapping": "formal_target as emitted by current formal stream; CASH remains cash_all/no-target cash",
        "old_no_target_cash_mapping_reference": "same as canonical for current formal stream rows",
        "no_target_cash_all_active_reference": "combined stream manifest 20260702 plus 2022-latest formal stream source_decision with no_target_cash_all",
        "default_requested_periods": PERIOD_CONTRACT,
        "actual_execution_start": _date_text(daily["date"].min()),
        "actual_execution_end": _date_text(daily["date"].max()),
        "initial_equity_twd": INITIAL_EQUITY,
        "equity_bases": ["period_reset_1m", "continuous_full_with_slices"],
        "benchmark_tickers": BENCHMARK_TICKERS,
        "missing_price_rows": missing_price_rows,
        "future_data_violation_count": future_count,
        "formal_model_changed": False,
        "trade_decision_changed": False,
        "active_in_trade_decision": False,
        "report_changed": False,
        "ready_for_experiments": bool(future_count == 0 and missing_price_rows == 0 and len(daily) > 0),
        "handoff_to_experiments_task": EXPERIMENTS_TASK_ID,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "final_summary_zh.md").write_text(_summary(manifest, period_reset, continuous), encoding="utf-8")
    pd.DataFrame([{"task_id": TASK_ID, "status": "completed", "output_dir": str(output)}]).to_csv(
        output / "completed.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(columns=["task_id", "status", "reason"]).to_csv(output / "failed.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(
        [
            {"step": "load_current_formal_streams", "status": "completed"},
            {"step": "simulate_canonical_next_day_ledger", "status": "completed"},
            {"step": "write_period_and_benchmark_summaries", "status": "completed"},
            {"step": "write_contract_package", "status": "completed"},
        ]
    ).to_csv(output / "run_log.csv", index=False, encoding="utf-8-sig")
    return manifest


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _prepare_formal_stream(formal: pd.DataFrame) -> pd.DataFrame:
    out = formal.copy()
    out["signal_date"] = pd.to_datetime(out["signal_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    out["execution_date"] = pd.to_datetime(out["execution_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    out["formal_target"] = out["formal_target"].map(_canonical_target)
    out = out.dropna(subset=["signal_date", "execution_date"]).sort_values("execution_date")
    out = out.drop_duplicates("execution_date", keep="last").reset_index(drop=True)
    out["raw_execution_date"] = out["execution_date"]
    out["execution_basis"] = "next_day"
    out["canonical_mapping"] = out["formal_target"].map(lambda target: "cash_all" if target == "CASH" else "target_100pct")
    return out


def _apply_execution_calendar(formal: pd.DataFrame, price_dir: Path) -> pd.DataFrame:
    out = formal.copy()
    out["calendar_adjusted"] = False
    out["calendar_adjustment_reason"] = ""
    calendar_by_ticker = {
        ticker: _load_price_calendar(price_dir, ticker)
        for ticker in sorted(set(out["formal_target"].astype(str)) | {"0050.TW"})
        if ticker and ticker != "CASH"
    }
    adjusted_dates: list[str] = []
    adjusted_flags: list[bool] = []
    reasons: list[str] = []
    for row in out.to_dict(orient="records"):
        raw = str(row["raw_execution_date"])
        signal = str(row["signal_date"])
        target = str(row.get("formal_target", ""))
        calendar_ticker = target if target and target != "CASH" else "0050.TW"
        calendar_dates = calendar_by_ticker.get(calendar_ticker, [])
        if not calendar_dates:
            adjusted_dates.append(raw)
            adjusted_flags.append(False)
            reasons.append(f"{calendar_ticker}_calendar_missing_calendar_not_adjusted")
            continue
        candidates = [date for date in calendar_dates if date >= raw and date > signal]
        adjusted = candidates[0] if candidates else raw
        adjusted_dates.append(adjusted)
        changed = adjusted != raw
        adjusted_flags.append(changed)
        reasons.append(f"raw_execution_date_not_{calendar_ticker}_trading_day_adjusted_to_next_price_date" if changed else "")
    out["execution_date"] = adjusted_dates
    out["calendar_adjusted"] = adjusted_flags
    out["calendar_adjustment_reason"] = reasons
    return out.sort_values("execution_date").drop_duplicates("execution_date", keep="last").reset_index(drop=True)


def _load_price_calendar(price_dir: Path, ticker: str) -> list[str]:
    path = price_dir / _price_file_name(ticker)
    if not path.exists():
        return []
    calendar = pd.read_csv(path, usecols=["date"])
    return sorted(pd.to_datetime(calendar["date"], errors="coerce").dropna().dt.strftime("%Y-%m-%d").unique())


def _load_canonical_formal_streams(paths: list[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in paths:
        if not path.exists():
            continue
        frame = pd.read_csv(path).fillna("")
        frame["source_stream"] = path.as_posix()
        frame["source_stream_priority"] = 2 if "formal_long_range_signal_reconstruction" in path.as_posix() else 1
        for column in ["signal_date", "execution_date", "formal_target"]:
            if column not in frame.columns:
                frame[column] = ""
        frames.append(frame)
    if not frames:
        raise FileNotFoundError("No formal streams found for canonical baseline ledger.")
    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined["signal_date"] = pd.to_datetime(combined["signal_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    combined["execution_date"] = pd.to_datetime(combined["execution_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    combined = combined.dropna(subset=["signal_date", "execution_date"])
    combined["formal_target"] = combined["formal_target"].map(_canonical_target)
    return combined.sort_values(["signal_date", "source_stream_priority"]).drop_duplicates("signal_date", keep="last")


def _load_prices(price_dir: Path, tickers: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    sources: list[dict[str, Any]] = []
    for ticker in tickers:
        if not ticker or ticker == "CASH":
            continue
        path = price_dir / _price_file_name(ticker)
        if not path.exists():
            sources.append({"ticker": ticker, "price_source_path": str(path), "exists": False, "rows": 0})
            continue
        frame = pd.read_csv(path, usecols=lambda col: col in {"date", "close", "adj_close"})
        close_col = "adj_close" if "adj_close" in frame.columns else "close"
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        frame["price"] = pd.to_numeric(frame[close_col], errors="coerce")
        frame["ticker"] = ticker
        frame["price_source_path"] = str(path)
        frame["price_field"] = close_col
        frame = frame.dropna(subset=["date", "price"]).drop_duplicates(["date", "ticker"])
        frames.append(frame[["date", "ticker", "price", "price_source_path", "price_field"]])
        sources.append(
            {
                "ticker": ticker,
                "price_source_path": str(path),
                "exists": True,
                "rows": int(len(frame)),
                "actual_start": _date_text(frame["date"].min()),
                "actual_end": _date_text(frame["date"].max()),
                "price_field": close_col,
            }
        )
    prices = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    return prices, pd.DataFrame(sources)


def _price_file_name(ticker: str) -> str:
    return ticker.replace(".TW", "_TW").replace(".TWO", "_TWO") + ".csv"


def _simulate_next_day_ledger(formal: pd.DataFrame, prices: pd.DataFrame, *, initial_equity: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    price_lookup = {(row["date"], row["ticker"]): float(row["price"]) for row in prices.to_dict(orient="records")}
    source_lookup = {(row["date"], row["ticker"]): row for row in prices.to_dict(orient="records")}
    model = TaiwanCostModel()
    cash = initial_equity
    shares: dict[str, float] = {}
    current_target = ""
    running_max = initial_equity
    daily_rows: list[dict[str, Any]] = []
    trade_rows: list[dict[str, Any]] = []
    for row in formal.to_dict(orient="records"):
        date = str(row["execution_date"])
        target = str(row["formal_target"])
        equity_before = _equity(cash, shares, price_lookup, date)
        price_ready = target == "CASH" or (date, target) in price_lookup
        trade_cost = 0.0
        turnover = 0.0
        if target != current_target and price_ready:
            cash, shares, trades, trade_cost, turnover = _rebalance_to_target(cash, shares, target, price_lookup, date, model, equity_before)
            current_target = target
            trade_rows.extend(
                {
                    **trade,
                    "date": date,
                    "signal_date": row["signal_date"],
                    "target_after_trade": target,
                    "cost_model_version": COST_MODEL_VERSION,
                }
                for trade in trades
            )
        equity = _equity(cash, shares, price_lookup, date)
        running_max = max(running_max, equity)
        held_ticker = "CASH" if not shares else "|".join(sorted(shares))
        source = source_lookup.get((date, target), {}) if target != "CASH" else {}
        daily_rows.append(
            {
                "date": date,
                "signal_date": row["signal_date"],
                "raw_execution_date": row.get("raw_execution_date", date),
                "calendar_adjusted": bool(row.get("calendar_adjusted", False)),
                "calendar_adjustment_reason": row.get("calendar_adjustment_reason", ""),
                "execution_basis": "next_day",
                "formal_target": target,
                "canonical_current_formal_mapping": "cash_all" if target == "CASH" else "target_100pct",
                "held_ticker": held_ticker,
                "cash": round(cash, 4),
                "shares": round(sum(shares.values()), 8),
                "equity": round(equity, 4),
                "running_max_equity": round(running_max, 4),
                "drawdown_pct": round((equity / running_max - 1.0) * 100.0, 6) if running_max else 0.0,
                "trade_cost": round(trade_cost, 4),
                "turnover": round(turnover, 4),
                "price_ready": bool(price_ready),
                "price_source_path": source.get("price_source_path", ""),
                "price_field": source.get("price_field", ""),
                "formal_model_changed": False,
                "trade_decision_changed": False,
                "active_in_trade_decision": False,
                "report_changed": False,
            }
        )
    return pd.DataFrame(daily_rows), pd.DataFrame(trade_rows)


def _rebalance_to_target(
    cash: float,
    shares: dict[str, float],
    target: str,
    prices: dict[tuple[str, str], float],
    date: str,
    model: TaiwanCostModel,
    equity: float,
) -> tuple[float, dict[str, float], list[dict[str, Any]], float, float]:
    trades: list[dict[str, Any]] = []
    cost_total = 0.0
    turnover = 0.0
    for ticker, qty in list(shares.items()):
        price = prices.get((date, ticker))
        if price is None or qty <= 0:
            continue
        gross = qty * price
        cost = model.sell_cost(gross, _asset_type(ticker))
        cash += gross - cost
        cost_total += cost
        turnover += gross
        trades.append({"ticker": ticker, "side": "sell", "price": price, "gross": round(gross, 4), "trade_cost": cost})
        shares.pop(ticker, None)
    if target and target != "CASH":
        price = prices.get((date, target))
        if price and price > 0:
            gross = min(equity, cash)
            cost = model.buy_cost(gross)
            qty = max(0.0, (gross - cost) / price)
            shares[target] = qty
            cash -= gross
            cost_total += cost
            turnover += gross
            trades.append({"ticker": target, "side": "buy", "price": price, "gross": round(gross, 4), "trade_cost": cost})
    return cash, shares, trades, cost_total, turnover


def _equity(cash: float, shares: dict[str, float], prices: dict[tuple[str, str], float], date: str) -> float:
    return cash + sum(qty * prices.get((date, ticker), 0.0) for ticker, qty in shares.items())


def _asset_type(ticker: str) -> str:
    return "etf" if ticker.startswith("00") else "stock"


def _period_reset_summary(formal: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for period in PERIOD_CONTRACT:
        frame = formal[(formal["execution_date"] >= period["requested_start"]) & (formal["execution_date"] <= period["requested_end"])]
        if frame.empty:
            rows.append(_empty_summary_row("period_reset_1m", period))
            continue
        daily, trades = _simulate_next_day_ledger(frame, prices, initial_equity=INITIAL_EQUITY)
        rows.append(_performance_row("period_reset_1m", period, daily, trades))
    return pd.DataFrame(rows)


def _continuous_slice_summary(daily: pd.DataFrame) -> pd.DataFrame:
    full_period = {
        "period_label": "full_available",
        "requested_start": _date_text(daily["date"].min()) if not daily.empty else "",
        "requested_end": _date_text(daily["date"].max()) if not daily.empty else "",
    }
    rows = [_slice_row("continuous_full_with_slices", full_period, daily)]
    for period in PERIOD_CONTRACT:
        frame = daily[(daily["date"] >= period["requested_start"]) & (daily["date"] <= period["requested_end"])]
        rows.append(_slice_row("continuous_full_with_slices", period, frame))
    return pd.DataFrame(rows)


def _benchmark_summary(prices: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for ticker in BENCHMARK_TICKERS:
        for period in PERIOD_CONTRACT:
            rows.append(_benchmark_period_row(ticker, "period_reset_1m", period, prices))
        full_period = {
            "period_label": "full_available",
            "requested_start": str(daily["date"].min()) if not daily.empty else "",
            "requested_end": str(daily["date"].max()) if not daily.empty else "",
        }
        rows.append(_benchmark_period_row(ticker, "continuous_full_with_slices", full_period, prices))
    return pd.DataFrame(rows)


def _benchmark_period_row(ticker: str, basis: str, period: dict[str, str], prices: pd.DataFrame) -> dict[str, Any]:
    series = prices[prices["ticker"].eq(ticker)].sort_values("date")
    frame = series[(series["date"] >= period["requested_start"]) & (series["date"] <= period["requested_end"])]
    if frame.empty:
        return {
            "benchmark_ticker": ticker,
            "equity_basis": basis,
            "period_label": period["period_label"],
            "requested_start": period["requested_start"],
            "requested_end": period["requested_end"],
            "actual_start": "",
            "actual_end": "",
            "rows": 0,
            "total_return_pct": pd.NA,
            "final_equity": pd.NA,
        }
    start = float(frame.iloc[0]["price"])
    shares = INITIAL_EQUITY / start
    equity = frame["price"].astype(float) * shares
    return {
        "benchmark_ticker": ticker,
        "equity_basis": basis,
        "period_label": period["period_label"],
        "requested_start": period["requested_start"],
        "requested_end": period["requested_end"],
        "actual_start": str(frame.iloc[0]["date"]),
        "actual_end": str(frame.iloc[-1]["date"]),
        "rows": int(len(frame)),
        "total_return_pct": round((float(equity.iloc[-1]) / INITIAL_EQUITY - 1.0) * 100.0, 4),
        "final_equity": round(float(equity.iloc[-1]), 2),
        "max_drawdown_pct": round(_max_drawdown(equity), 4),
    }


def _performance_row(equity_basis: str, period: dict[str, str], daily: pd.DataFrame, trades: pd.DataFrame) -> dict[str, Any]:
    if daily.empty:
        return _empty_summary_row(equity_basis, period)
    start_equity = INITIAL_EQUITY
    end_equity = float(daily.iloc[-1]["equity"])
    return {
        "equity_basis": equity_basis,
        "period_label": period["period_label"],
        "requested_start": period.get("requested_start", ""),
        "requested_end": period.get("requested_end", ""),
        "actual_start": str(daily.iloc[0]["date"]),
        "actual_end": str(daily.iloc[-1]["date"]),
        "rows": int(len(daily)),
        "start_equity": round(start_equity, 2),
        "final_equity": round(end_equity, 2),
        "total_return_pct": round((end_equity / start_equity - 1.0) * 100.0, 4),
        "max_drawdown_pct": round(float(pd.to_numeric(daily["drawdown_pct"], errors="coerce").min()), 4),
        "trade_rows": int(len(trades)),
        "total_transaction_cost": round(float(pd.to_numeric(trades.get("trade_cost", pd.Series(dtype=float)), errors="coerce").sum()), 2),
        "turnover": round(float(pd.to_numeric(trades.get("gross", pd.Series(dtype=float)), errors="coerce").sum()), 2),
    }


def _slice_row(equity_basis: str, period: dict[str, str], daily: pd.DataFrame) -> dict[str, Any]:
    if daily.empty:
        return _empty_summary_row(equity_basis, period)
    start = float(daily.iloc[0]["equity"])
    end = float(daily.iloc[-1]["equity"])
    drawdown = (daily["equity"].astype(float) / daily["equity"].astype(float).cummax() - 1.0) * 100.0
    return {
        "equity_basis": equity_basis,
        "period_label": period["period_label"],
        "requested_start": period["requested_start"],
        "requested_end": period["requested_end"],
        "actual_start": str(daily.iloc[0]["date"]),
        "actual_end": str(daily.iloc[-1]["date"]),
        "rows": int(len(daily)),
        "start_equity": round(start, 2),
        "final_equity": round(end, 2),
        "total_return_pct": round((end / start - 1.0) * 100.0, 4) if start else pd.NA,
        "max_drawdown_pct": round(float(drawdown.min()), 4),
        "trade_rows": pd.NA,
        "total_transaction_cost": round(float(pd.to_numeric(daily["trade_cost"], errors="coerce").sum()), 2),
        "turnover": round(float(pd.to_numeric(daily["turnover"], errors="coerce").sum()), 2),
    }


def _empty_summary_row(equity_basis: str, period: dict[str, str]) -> dict[str, Any]:
    return {
        "equity_basis": equity_basis,
        "period_label": period["period_label"],
        "requested_start": period.get("requested_start", ""),
        "requested_end": period.get("requested_end", ""),
        "actual_start": "",
        "actual_end": "",
        "rows": 0,
        "status": "no_actual_coverage",
    }


def _max_drawdown(equity: pd.Series) -> float:
    drawdown = (equity / equity.cummax() - 1.0) * 100.0
    return float(drawdown.min())


def _no_target_mapping_audit(formal: pd.DataFrame, source_paths: list[Path]) -> pd.DataFrame:
    cash = formal[formal["formal_target"].eq("CASH")]
    return pd.DataFrame(
        [
            {
                "mapping_id": "canonical_current_formal_mapping",
                "mapping_rule": "formal_target CASH remains cash_all; non-CASH target is held 100%",
                "source": "current formal target streams",
                "rows": int(len(formal)),
                "cash_rows": int(len(cash)),
                "canonical": True,
            },
            {
                "mapping_id": "old_no_target_cash_mapping_reference",
                "mapping_rule": "same as canonical for current formal streams",
                "source": "fallback boundary v2 / Experiments rerun old cash reference",
                "rows": int(len(formal)),
                "cash_rows": int(len(cash)),
                "canonical": False,
            },
            {
                "mapping_id": "no_target_cash_all_active_reference",
                "mapping_rule": "no_target_cash_all from 20260702 long-range stream generation; not a new patch in this runner",
                "source": ";".join(str(path) for path in source_paths),
                "rows": int(len(cash)),
                "cash_rows": int(len(cash)),
                "canonical": False,
            },
        ]
    )


def _cost_trade_summary(daily: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(
            [{"scope": "canonical_formal_next_day", "trade_rows": 0, "total_transaction_cost": 0, "turnover": 0}]
        )
    return pd.DataFrame(
        [
            {
                "scope": "canonical_formal_next_day",
                "cost_model_version": COST_MODEL_VERSION,
                "trade_rows": int(len(trades)),
                "buy_rows": int(trades["side"].eq("buy").sum()),
                "sell_rows": int(trades["side"].eq("sell").sum()),
                "total_transaction_cost": round(float(pd.to_numeric(trades["trade_cost"], errors="coerce").sum()), 2),
                "turnover": round(float(pd.to_numeric(trades["gross"], errors="coerce").sum()), 2),
                "daily_rows": int(len(daily)),
            }
        ]
    )


def _period_contract_validation(formal: pd.DataFrame, daily: pd.DataFrame, prices: pd.DataFrame, sources: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for period in PERIOD_CONTRACT:
        signal = formal[(formal["signal_date"] >= period["requested_start"]) & (formal["signal_date"] <= period["requested_end"])]
        execution = daily[(daily["date"] >= period["requested_start"]) & (daily["date"] <= period["requested_end"])]
        rows.append(_validation_row("formal_signal_stream", period, signal, "signal_date"))
        rows.append(_validation_row("canonical_daily_ledger_execution", period, execution, "date"))
        for ticker in BENCHMARK_TICKERS:
            benchmark = prices[
                (prices["ticker"].eq(ticker))
                & (prices["date"] >= period["requested_start"])
                & (prices["date"] <= period["requested_end"])
            ]
            row = _validation_row(f"benchmark_{ticker}", period, benchmark, "date")
            source = sources[sources["ticker"].eq(ticker)]
            row["price_source_path"] = str(source.iloc[0]["price_source_path"]) if not source.empty else ""
            rows.append(row)
    return pd.DataFrame(rows)


def _validation_row(layer: str, period: dict[str, str], frame: pd.DataFrame, date_col: str) -> dict[str, Any]:
    return {
        "layer": layer,
        "period_label": period["period_label"],
        "requested_start": period["requested_start"],
        "requested_end": period["requested_end"],
        "actual_start": _date_text(frame[date_col].min()) if not frame.empty else "",
        "actual_end": _date_text(frame[date_col].max()) if not frame.empty else "",
        "rows": int(len(frame)),
        "status": "actual_coverage_present" if not frame.empty else "no_actual_coverage",
    }


def _prior_number_reconciliation() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "prior_number": "+3452.89%",
                "source": "Experiments fallback boundary diagnostic rerun P2 old cash",
                "reason_not_canonical_mixed_use": "P2-only replay using fallback diagnostic package and old no-target cash reference",
                "canonical_handling": "compare only via canonical_period_reset_summary / canonical_continuous_slice_summary",
            },
            {
                "prior_number": "+1437.36%",
                "source": "Strong-stock trend-extension bounded portfolio diagnostic validation current formal baseline",
                "reason_not_canonical_mixed_use": "different runner and actual period; not the canonical full formal ledger",
                "canonical_handling": "listed as prior-number source, not reused as canonical baseline",
            },
            {
                "prior_number": "+700%~800%",
                "source": "Prior current-period / fallback / challenger diagnostics depending on runner and period",
                "reason_not_canonical_mixed_use": "range came from different period slices or benchmark/fallback basis",
                "canonical_handling": "must not be mixed without matching equity_basis and requested/actual period",
            },
        ]
    )


def _future_data_audit(formal: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    merged = daily[["date", "signal_date"]].copy()
    merged["future_data_violation"] = pd.to_datetime(merged["signal_date"]) >= pd.to_datetime(merged["date"])
    return pd.DataFrame(
        [
            {
                "audit_item": "canonical_formal_daily_ledger_next_day_execution",
                "rows": int(len(daily)),
                "future_data_violation": bool(merged["future_data_violation"].any()),
                "reason": "signal_date must be earlier than execution date",
            },
            {
                "audit_item": "formal_target_stream",
                "rows": int(len(formal)),
                "future_data_violation": False,
                "reason": "runner consumes existing formal target stream and does not alter selector",
            },
        ]
    )


def _summary(manifest: dict[str, Any], reset: pd.DataFrame, continuous: pd.DataFrame) -> str:
    reset_text = reset[["period_label", "actual_start", "actual_end", "total_return_pct"]].to_string(index=False)
    cont_text = continuous[["period_label", "actual_start", "actual_end", "total_return_pct"]].to_string(index=False)
    return (
        "# Canonical formal baseline ledger\n\n"
        "## 結論\n\n"
        "- 本包建立唯一 canonical current-formal next-day daily ledger；沒有改 selector、target、report 或 trade decision。\n"
        f"- execution_basis：{manifest['execution_basis']}\n"
        f"- daily rows：{manifest['canonical_daily_ledger_rows']}\n"
        f"- trade rows：{manifest['trade_rows']}\n"
        f"- actual execution：{manifest['actual_execution_start']}～{manifest['actual_execution_end']}\n"
        f"- missing price rows：{manifest['missing_price_rows']}\n"
        f"- ready_for_experiments：{manifest['ready_for_experiments']}\n\n"
        "## Period Reset 1m\n\n"
        f"```\n{reset_text}\n```\n\n"
        "## Continuous Full With Slices\n\n"
        f"```\n{cont_text}\n```\n\n"
        "## 邊界\n\n"
        "- canonical current formal mapping：CASH 仍為 cash_all；非 CASH formal target 以 100% 目標 replay。\n"
        "- prior +3452.89%、+1437.36%、+700%~800% 只列 reconciliation，不當作 canonical baseline 混用。\n"
    )


def _date_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value)[:10]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build canonical formal next-day baseline ledger.")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()
    manifest = run_canonical_formal_baseline_ledger(repo_root=args.repo_root, output_dir=args.output_dir)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
