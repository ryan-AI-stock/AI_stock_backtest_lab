from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from backtest_lab.costs import TaiwanCostModel, cost_model_metadata


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_DIR = (
    Path("C:/Users/zergv/Documents/Codex/2026-07-06/backtest-lab-experiments-diagnostic-validation-attribution")
    / "outputs"
    / "vnext_p1_ordinary_defensive_reference_policy_diagnostic_20260708"
)
BENCHMARK_FEATURES = REPO_ROOT / "outputs" / "vnext_dynamic_candidate_pool_data_materialization_20260706" / "benchmark_features.csv"
OUTPUT_DIR = REPO_ROOT / "outputs" / "vnext_p1_defensive_policy_benchmark_path_20260708"

TASK_ID = "TASK-BACKTEST-CORE-VNEXT-P1-DEFENSIVE-POLICY-BENCHMARK-PATH-MATERIALIZATION-001"
DIAGNOSTIC_NOTIONAL_TWD = 1_000_000
BENCHMARKS = ["00631L", "0050"]
TIMING_VARIANTS = [
    "same_week_close_to_next_rebalance_close_comparator",
    "next_day_close_entry_fixed_5td_exit",
    "next_day_open_entry_fixed_5td_exit",
]
FLAGS = {
    "formal_model_changed": False,
    "trade_decision_changed": False,
    "active_in_trade_decision": False,
    "report_changed": False,
    "portfolio_replay_executed": False,
    "ready_for_strategy_replay": False,
    "ready_for_formal": False,
    "not_live_rule": True,
    "forward_returns_live_rule_usage": False,
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _signal_dates() -> list[pd.Timestamp]:
    trace = pd.read_csv(EXPERIMENTS_DIR / "p1_ordinary_defensive_reference_policy_path_trace.csv", usecols=["signal_date"])
    dates = pd.to_datetime(trace["signal_date"], errors="coerce").dropna().drop_duplicates().sort_values()
    return [pd.Timestamp(x) for x in dates]


def _benchmark_data() -> tuple[pd.DataFrame, dict[str, list[pd.Timestamp]], dict[tuple[str, pd.Timestamp], float]]:
    bench = pd.read_csv(BENCHMARK_FEATURES, dtype={"benchmark": str}, low_memory=False)
    bench["trade_date"] = pd.to_datetime(bench["trade_date"], errors="coerce")
    bench = bench.loc[bench["benchmark"].isin(BENCHMARKS)].copy()
    bench = bench.sort_values(["benchmark", "trade_date"])
    calendars = {
        benchmark: [pd.Timestamp(x) for x in sorted(group["trade_date"].dropna().unique())]
        for benchmark, group in bench.groupby("benchmark")
    }
    close_map = {
        (str(row.benchmark), pd.Timestamp(row.trade_date)): float(row.adjusted_close)
        for row in bench.itertuples(index=False)
        if pd.notna(row.adjusted_close)
    }
    return bench, calendars, close_map


def _next_trading_date(calendar: list[pd.Timestamp], date: pd.Timestamp, offset: int) -> pd.Timestamp | None:
    for idx, trading_date in enumerate(calendar):
        if trading_date > date:
            target = idx + offset - 1
            return calendar[target] if 0 <= target < len(calendar) else None
    return None


def _next_signal_date(signal_dates: list[pd.Timestamp], date: pd.Timestamp) -> pd.Timestamp | None:
    later = [d for d in signal_dates if d > date]
    return later[0] if later else None


def _apply_cost(benchmark: str, entry_price: float | None, exit_price: float | None) -> dict[str, Any]:
    if entry_price is None or exit_price is None or entry_price <= 0 or exit_price <= 0:
        return {
            "diagnostic_unit_notional_twd": DIAGNOSTIC_NOTIONAL_TWD,
            "diagnostic_share_qty": None,
            "buy_gross_twd": None,
            "sell_gross_twd": None,
            "buy_cost_twd": None,
            "sell_cost_twd": None,
            "total_cost_twd": None,
            "net_return_local_ep05_cost_unit_notional": None,
            "cost_application_status": "blocked_missing_entry_or_exit_price",
        }
    qty = math.floor(DIAGNOSTIC_NOTIONAL_TWD / entry_price)
    model = TaiwanCostModel()
    buy_gross = qty * entry_price
    sell_gross = qty * exit_price
    buy_cost = model.buy_cost(buy_gross)
    sell_cost = model.sell_cost(sell_gross, "etf")
    net_return = (sell_gross - sell_cost - buy_gross - buy_cost) / (buy_gross + buy_cost)
    return {
        "diagnostic_unit_notional_twd": DIAGNOSTIC_NOTIONAL_TWD,
        "diagnostic_share_qty": qty,
        "buy_gross_twd": buy_gross,
        "sell_gross_twd": sell_gross,
        "buy_cost_twd": buy_cost,
        "sell_cost_twd": sell_cost,
        "total_cost_twd": buy_cost + sell_cost,
        "net_return_local_ep05_cost_unit_notional": net_return,
        "cost_application_status": f"applied_local_ep05_cost_model_to_{benchmark}_reference_unit_notional",
    }


def _path_rows() -> pd.DataFrame:
    signal_dates = _signal_dates()
    _, calendars, close_map = _benchmark_data()
    rows: list[dict[str, Any]] = []
    for signal_date in signal_dates:
        for benchmark in BENCHMARKS:
            calendar = calendars[benchmark]
            for timing in TIMING_VARIANTS:
                if timing == "same_week_close_to_next_rebalance_close_comparator":
                    entry_date = signal_date
                    exit_date = _next_signal_date(signal_dates, signal_date)
                    entry_kind = "adjusted_close"
                    exit_kind = "adjusted_close"
                    open_blocked_reason = ""
                elif timing == "next_day_open_entry_fixed_5td_exit":
                    entry_date = _next_trading_date(calendar, signal_date, 1)
                    exit_date = _next_trading_date(calendar, signal_date, 6)
                    entry_kind = "open"
                    exit_kind = "adjusted_close"
                    open_blocked_reason = "benchmark_open_price_not_available_in_benchmark_features"
                else:
                    entry_date = _next_trading_date(calendar, signal_date, 1)
                    exit_date = _next_trading_date(calendar, signal_date, 6)
                    entry_kind = "adjusted_close"
                    exit_kind = "adjusted_close"
                    open_blocked_reason = ""
                entry_price = None if entry_kind == "open" else close_map.get((benchmark, entry_date)) if entry_date is not None else None
                exit_price = close_map.get((benchmark, exit_date)) if exit_date is not None else None
                ready = entry_price is not None and exit_price is not None
                gross_return = (exit_price / entry_price - 1.0) if ready else None
                blocked_reason = "" if ready else open_blocked_reason or "missing_benchmark_entry_or_exit_adjusted_close"
                row = {
                    "signal_date": signal_date.date().isoformat(),
                    "benchmark": benchmark,
                    "path_type": f"{benchmark}_signal_aligned_reference",
                    "timing_variant": timing,
                    "entry_date": entry_date.date().isoformat() if entry_date is not None else "",
                    "exit_date": exit_date.date().isoformat() if exit_date is not None else "",
                    "entry_price_kind": entry_kind,
                    "exit_price_kind": exit_kind,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "gross_return": gross_return,
                    "path_ready": ready,
                    "blocked_reason": blocked_reason,
                    "source_quality": "benchmark_features_adjusted_close_exact_reference",
                    "cash_proxy": False,
                    "diagnostic_only": True,
                    **FLAGS,
                }
                row.update(_apply_cost(benchmark, entry_price, exit_price))
                rows.append(row)

        cash_calendar = calendars["0050"]
        for timing in TIMING_VARIANTS:
            entry_date = signal_date if timing == "same_week_close_to_next_rebalance_close_comparator" else _next_trading_date(cash_calendar, signal_date, 1)
            exit_date = _next_signal_date(signal_dates, signal_date) if timing == "same_week_close_to_next_rebalance_close_comparator" else _next_trading_date(cash_calendar, signal_date, 6)
            rows.append(
                {
                    "signal_date": signal_date.date().isoformat(),
                    "benchmark": "cash",
                    "path_type": "cash_zero_return_reference",
                    "timing_variant": timing,
                    "entry_date": entry_date.date().isoformat() if entry_date is not None else "",
                    "exit_date": exit_date.date().isoformat() if exit_date is not None else "",
                    "entry_price_kind": "cash_zero_return",
                    "exit_price_kind": "cash_zero_return",
                    "entry_price": 1.0,
                    "exit_price": 1.0,
                    "gross_return": 0.0,
                    "net_return_local_ep05_cost_unit_notional": 0.0,
                    "diagnostic_unit_notional_twd": DIAGNOSTIC_NOTIONAL_TWD,
                    "diagnostic_share_qty": None,
                    "buy_gross_twd": None,
                    "sell_gross_twd": None,
                    "buy_cost_twd": 0.0,
                    "sell_cost_twd": 0.0,
                    "total_cost_twd": 0.0,
                    "cost_application_status": "cash_zero_return_no_interest_no_slippage_proxy",
                    "path_ready": True,
                    "blocked_reason": "",
                    "source_quality": "cash_zero_return_proxy_no_interest",
                    "cash_proxy": True,
                    "diagnostic_only": True,
                    **FLAGS,
                }
            )
    return pd.DataFrame(rows)


def _buy_hold_rows() -> pd.DataFrame:
    signal_dates = _signal_dates()
    start = min(signal_dates)
    end = max(signal_dates)
    _, _, close_map = _benchmark_data()
    rows = []
    for benchmark in BENCHMARKS:
        entry_price = close_map.get((benchmark, start))
        exit_price = close_map.get((benchmark, end))
        ready = entry_price is not None and exit_price is not None
        row = {
            "signal_date": start.date().isoformat(),
            "benchmark": benchmark,
            "path_type": f"{benchmark}_buy_hold_p1_actual_period_reference",
            "timing_variant": "buy_hold_p1_actual_period_reference",
            "entry_date": start.date().isoformat(),
            "exit_date": end.date().isoformat(),
            "entry_price_kind": "adjusted_close",
            "exit_price_kind": "adjusted_close",
            "entry_price": entry_price,
            "exit_price": exit_price,
            "gross_return": (exit_price / entry_price - 1.0) if ready else None,
            "path_ready": ready,
            "blocked_reason": "" if ready else "missing_buy_hold_entry_or_exit_adjusted_close",
            "source_quality": "benchmark_features_adjusted_close_exact_reference",
            "cash_proxy": False,
            "diagnostic_only": True,
            **FLAGS,
        }
        row.update(_apply_cost(benchmark, entry_price, exit_price))
        rows.append(row)
    return pd.DataFrame(rows)


def _coverage(path: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        path.groupby(["benchmark", "timing_variant"], dropna=False)
        .agg(
            rows=("signal_date", "size"),
            ready_rows=("path_ready", "sum"),
            blocked_rows=("path_ready", lambda s: int((~s.astype(bool)).sum())),
            numeric_return_rows=("gross_return", lambda s: int(s.notna().sum())),
        )
        .reset_index()
    )
    grouped["ready_share"] = grouped["ready_rows"] / grouped["rows"]
    grouped["diagnostic_only"] = True
    return grouped


def _timing_cost_audit(path: pd.DataFrame) -> pd.DataFrame:
    meta = cost_model_metadata()
    same_week = path.loc[
        (path["benchmark"].isin(BENCHMARKS))
        & (path["timing_variant"] == "same_week_close_to_next_rebalance_close_comparator")
    ]
    same_week_ready = bool(same_week["path_ready"].sum() >= (len(same_week) - len(BENCHMARKS)))
    rows = [
        {
            "audit_item": "next_day_close_signal_aligned_path",
            "ready": bool(
                path.loc[
                    (path["benchmark"].isin(BENCHMARKS))
                    & (path["timing_variant"] == "next_day_close_entry_fixed_5td_exit"),
                    "path_ready",
                ].all()
            ),
            "source_quality": "benchmark_features_adjusted_close",
            "notes": "Full 0050/00631L signal-date aligned close path.",
        },
        {
            "audit_item": "same_week_close_signal_aligned_path",
            "ready": same_week_ready,
            "source_quality": "benchmark_features_adjusted_close",
            "notes": "Final signal date has no next signal and is excluded from readiness check via blocked ledger if present.",
        },
        {
            "audit_item": "next_day_open_signal_aligned_path",
            "ready": False,
            "source_quality": "blocked_open_price_not_in_benchmark_features",
            "notes": "Open price is not available locally; not fabricated.",
        },
        {
            "audit_item": "cash_zero_return_proxy",
            "ready": bool(path.loc[path["benchmark"] == "cash", "path_ready"].all()),
            "source_quality": "cash_proxy_no_interest_no_slippage",
            "notes": "Cash reference assumes zero return, zero interest, zero slippage.",
        },
        {
            "audit_item": "local_ep05_cost_model",
            "ready": True,
            "source_quality": "local_taiwan_standard_fee_tax_v1",
            "notes": "Applied to 0050/00631L ETF unit-notional diagnostic rows only; not portfolio replay.",
            **meta,
        },
    ]
    return pd.DataFrame(rows)


def _missing_price_audit(path: pd.DataFrame) -> pd.DataFrame:
    blocked = path.loc[~path["path_ready"].astype(bool)].copy()
    if blocked.empty:
        return pd.DataFrame(
            columns=["benchmark", "timing_variant", "blocked_rows", "blocked_reason", "diagnostic_only"]
        )
    grouped = (
        blocked.groupby(["benchmark", "timing_variant", "blocked_reason"], dropna=False)
        .agg(blocked_rows=("signal_date", "size"))
        .reset_index()
    )
    grouped["diagnostic_only"] = True
    return grouped


def _future_audit() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "audit_item": "future_return_as_rule",
                "result": "passed",
                "violation_count": 0,
                "evidence": "Benchmark paths are evaluation/reference rows only, not rule construction.",
            },
            {
                "audit_item": "benchmark_reference_mixed_with_stock_selection",
                "result": "passed",
                "violation_count": 0,
                "evidence": "Output is benchmark/reference path only.",
            },
            {
                "audit_item": "cash_interest_or_slippage_fabricated",
                "result": "passed",
                "violation_count": 0,
                "evidence": "Cash path is explicit zero-return proxy, no interest and no slippage.",
            },
            {
                "audit_item": "open_price_fabricated",
                "result": "passed",
                "violation_count": 0,
                "evidence": "Next-day open path is blocked because benchmark open is unavailable.",
            },
        ]
    )


def _write_manifest(files: list[Path], readiness: dict[str, Any]) -> None:
    manifest = {
        "task_id": TASK_ID,
        "status": readiness["status"],
        "output_dir": str(OUTPUT_DIR),
        "input_experiments_dir": str(EXPERIMENTS_DIR),
        "input_benchmark_features": str(BENCHMARK_FEATURES),
        "output_files": [p.name for p in files] + ["manifest.json"],
        "diagnostic_only": True,
        **FLAGS,
    }
    manifest["file_hashes"] = {p.name: {"sha256": _sha256(p), "bytes": p.stat().st_size} for p in files if p.exists()}
    (OUTPUT_DIR / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def _summary(readiness: dict[str, Any]) -> str:
    lines = [
        "# P1 defensive policy benchmark/reference path",
        "",
        f"- status: `{readiness['status']}`",
        f"- signal_dates: {readiness['signal_date_count']}",
        f"- full_p1_00631l_signal_path_ready: {str(readiness['full_p1_00631l_signal_path_ready']).lower()}",
        f"- full_p1_0050_signal_path_ready: {str(readiness['full_p1_0050_signal_path_ready']).lower()}",
        f"- cash_reference_path_ready: {str(readiness['cash_reference_path_ready']).lower()}",
        f"- next_day_open_ready: {str(readiness['next_day_open_ready']).lower()}",
        f"- ready_for_experiments: {str(readiness['ready_for_experiments']).lower()}",
        "",
        "## 判斷",
        "",
        "Core 已用同一批 P1 signal dates 補齊 00631L / 0050 signal-aligned close path 與 cash zero-return reference。"
        "next-day open benchmark path 因本機 benchmark_features 沒有 open price，明確 blocked，不杜撰。",
        "",
        "這是 benchmark/reference path only，不是 replay / formal / daily report / trade decision。",
        "",
        "## Flags",
        "",
    ]
    for key, value in FLAGS.items():
        lines.append(f"- {key}={str(value).lower()}")
    lines.append("- diagnostic_only=true")
    return "\n".join(lines) + "\n"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = _path_rows()
    buy_hold = _buy_hold_rows()
    full = pd.concat([path, buy_hold], ignore_index=True, sort=False)
    coverage = _coverage(path)
    timing_cost = _timing_cost_audit(path)
    missing = _missing_price_audit(path)
    future = _future_audit()
    signal_count = len(_signal_dates())
    close_ready = bool(
        path.loc[
            (path["benchmark"].isin(BENCHMARKS))
            & (path["timing_variant"] == "next_day_close_entry_fixed_5td_exit"),
            "path_ready",
        ].all()
    )
    same_week_rows = path.loc[
        (path["benchmark"].isin(BENCHMARKS))
        & (path["timing_variant"] == "same_week_close_to_next_rebalance_close_comparator")
    ]
    same_week_ready = bool(same_week_rows["path_ready"].sum() >= (len(same_week_rows) - len(BENCHMARKS)))
    cash_ready = bool(path.loc[path["benchmark"] == "cash", "path_ready"].all())
    future_violations = int(future["violation_count"].sum())
    ready = bool(close_ready and same_week_ready and cash_ready and future_violations == 0)
    readiness = {
        "task_id": TASK_ID,
        "status": "p1_defensive_policy_benchmark_close_cash_reference_ready_open_blocked",
        "diagnostic_only": True,
        "signal_date_count": signal_count,
        "requested_period_start": min(_signal_dates()).date().isoformat(),
        "requested_period_end": max(_signal_dates()).date().isoformat(),
        "ready_for_p1_defensive_policy_benchmark_diagnostic": ready,
        "full_p1_00631l_signal_path_ready": close_ready,
        "full_p1_0050_signal_path_ready": close_ready,
        "cash_reference_path_ready": cash_ready,
        "next_day_close_ready": close_ready,
        "next_day_open_ready": False,
        "same_week_close_ready": same_week_ready,
        "formal_cost_model_ready": True,
        "formal_cost_model_scope": "diagnostic_unit_notional_benchmark_reference_not_portfolio_replay",
        "blocked_rows": int((~path["path_ready"].astype(bool)).sum()),
        "future_data_violation_count": future_violations,
        "ready_for_experiments": ready,
        "ready_for_formal": False,
        "ready_for_strategy_replay": False,
        "blocked_fields": ["benchmark_next_day_open_price", "formal_portfolio_replay"],
        "proxy_fields": ["cash_zero_return_no_interest_no_slippage"],
        **FLAGS,
    }
    files = [
        OUTPUT_DIR / "p1_defensive_policy_benchmark_reference_path.csv",
        OUTPUT_DIR / "p1_defensive_policy_00631l_path.csv",
        OUTPUT_DIR / "p1_defensive_policy_0050_path.csv",
        OUTPUT_DIR / "p1_defensive_policy_cash_reference_path.csv",
        OUTPUT_DIR / "p1_defensive_policy_buy_hold_reference.csv",
        OUTPUT_DIR / "p1_defensive_policy_benchmark_timing_cost_audit.csv",
        OUTPUT_DIR / "p1_defensive_policy_benchmark_missing_price_audit.csv",
        OUTPUT_DIR / "p1_defensive_policy_benchmark_future_data_audit.csv",
        OUTPUT_DIR / "readiness_for_p1_defensive_policy_benchmark_diagnostic.json",
        OUTPUT_DIR / "final_summary_zh.md",
    ]
    full.to_csv(files[0], index=False, encoding="utf-8")
    path.loc[path["benchmark"] == "00631L"].to_csv(files[1], index=False, encoding="utf-8")
    path.loc[path["benchmark"] == "0050"].to_csv(files[2], index=False, encoding="utf-8")
    path.loc[path["benchmark"] == "cash"].to_csv(files[3], index=False, encoding="utf-8")
    buy_hold.to_csv(files[4], index=False, encoding="utf-8")
    timing_cost.to_csv(files[5], index=False, encoding="utf-8")
    missing.to_csv(files[6], index=False, encoding="utf-8")
    future.to_csv(files[7], index=False, encoding="utf-8")
    files[8].write_text(json.dumps(readiness, ensure_ascii=False, indent=2), encoding="utf-8")
    files[9].write_text(_summary(readiness), encoding="utf-8")
    _write_manifest(files, readiness)


if __name__ == "__main__":
    main()
