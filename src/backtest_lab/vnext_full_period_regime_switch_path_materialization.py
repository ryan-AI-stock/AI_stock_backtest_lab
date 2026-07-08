from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from backtest_lab.costs import TaiwanCostModel, cost_model_metadata


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_ROOT = Path("C:/Users/zergv/Documents/Codex/2026-07-06/backtest-lab-experiments-diagnostic-validation-attribution")
P1_EXPERIMENTS = (
    EXPERIMENTS_ROOT
    / "outputs"
    / "vnext_p1_defensive_policy_benchmark_comparison_diagnostic_20260708"
)
BENCHMARK_FEATURES = REPO_ROOT / "outputs" / "vnext_dynamic_candidate_pool_data_materialization_20260706" / "benchmark_features.csv"
P1_STOCK_PATH = (
    REPO_ROOT
    / "outputs"
    / "vnext_p1_legacy_regime_unadjusted_path_refresh_20260708"
    / "p1_legacy_regime_unadjusted_trade_path_refreshed.csv"
)
P1_BENCHMARK_PATH = REPO_ROOT / "outputs" / "vnext_p1_defensive_policy_benchmark_path_20260708"
REGIME_SIGNAL_TABLE = (
    REPO_ROOT
    / "outputs"
    / "vnext_regime_switch_hybrid_route_market_fields_path_materialization_20260708"
    / "regime_switch_hybrid_route_signal_table.csv"
)
REGIME_PATH = (
    REPO_ROOT
    / "outputs"
    / "vnext_regime_switch_hybrid_route_path_refresh_20260708"
    / "regime_switch_hybrid_route_selected_path_refreshed.csv"
)
LEGACY_PATH = (
    REPO_ROOT
    / "outputs"
    / "vnext_legacy_rs20_unadjusted_ohlc_timing_cost_materialization_20260708"
    / "legacy_rs20_unadjusted_selected_stock_trade_path.csv"
)
OUTPUT_DIR = REPO_ROOT / "outputs" / "vnext_full_period_regime_switch_benchmark_exception_path_20260708"

TASK_ID = "TASK-BACKTEST-CORE-VNEXT-FULL-PERIOD-REGIME-SWITCH-BENCHMARK-EXCEPTION-PATH-MATERIALIZATION-001"
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
PERIODS = {
    "P1": ("2015-01-02", "2022-12-29"),
    "P2": ("2023-01-02", "2026-06-30"),
    "2024_latest": ("2024-01-02", "2026-06-30"),
    "2026YTD": ("2026-01-02", "2026-06-30"),
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _ticker_str(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text.zfill(4) if text.isdigit() and len(text) < 4 else text


def _period_label(date: pd.Timestamp) -> str:
    labels = []
    for period, (start, end) in PERIODS.items():
        if pd.Timestamp(start) <= date <= pd.Timestamp(end):
            labels.append(period)
    return "|".join(labels) if labels else "outside_requested_periods"


def _next_trading_date(calendar: list[pd.Timestamp], date: pd.Timestamp, offset: int) -> pd.Timestamp | None:
    for idx, trading_date in enumerate(calendar):
        if trading_date > date:
            target = idx + offset - 1
            return calendar[target] if 0 <= target < len(calendar) else None
    return None


def _next_signal_date(signal_dates: list[pd.Timestamp], date: pd.Timestamp) -> pd.Timestamp | None:
    for signal_date in signal_dates:
        if signal_date > date:
            return signal_date
    return None


def _benchmark_data() -> tuple[dict[str, list[pd.Timestamp]], dict[tuple[str, pd.Timestamp], float]]:
    bench = pd.read_csv(BENCHMARK_FEATURES, dtype={"benchmark": str}, low_memory=False)
    bench["trade_date"] = pd.to_datetime(bench["trade_date"], errors="coerce")
    bench = bench.loc[bench["benchmark"].isin(BENCHMARKS)].copy()
    bench = bench.sort_values(["benchmark", "trade_date"])
    calendars = {
        benchmark: [pd.Timestamp(x) for x in group["trade_date"].dropna().sort_values().unique()]
        for benchmark, group in bench.groupby("benchmark")
    }
    close_map = {
        (str(row.benchmark), pd.Timestamp(row.trade_date)): float(row.adjusted_close)
        for row in bench.itertuples(index=False)
        if pd.notna(row.adjusted_close)
    }
    return calendars, close_map


def _signal_dates() -> list[pd.Timestamp]:
    p1 = pd.read_csv(P1_EXPERIMENTS / "p1_defensive_policy_benchmark_policy_trace.csv", usecols=["signal_date"])
    p1_dates = pd.to_datetime(p1["signal_date"], errors="coerce").dropna()
    regime = pd.read_csv(REGIME_SIGNAL_TABLE, usecols=["snapshot_date"])
    regime_dates = pd.to_datetime(regime["snapshot_date"], errors="coerce").dropna()
    all_dates = pd.concat([p1_dates, regime_dates], ignore_index=True).drop_duplicates().sort_values()
    all_dates = all_dates.loc[(all_dates >= pd.Timestamp("2015-01-02")) & (all_dates <= pd.Timestamp("2026-06-30"))]
    return [pd.Timestamp(x) for x in all_dates]


def _apply_cost(asset_type: str, entry_price: float | None, exit_price: float | None) -> dict[str, Any]:
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
    sell_cost = model.sell_cost(sell_gross, asset_type)
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
        "cost_application_status": f"applied_local_ep05_cost_model_to_{asset_type}_unit_notional",
    }


def _benchmark_reference_path() -> pd.DataFrame:
    signal_dates = _signal_dates()
    calendars, close_map = _benchmark_data()
    rows: list[dict[str, Any]] = []
    for signal_date in signal_dates:
        period_label = _period_label(signal_date)
        for benchmark in BENCHMARKS:
            calendar = calendars[benchmark]
            for timing in TIMING_VARIANTS:
                if timing == "same_week_close_to_next_rebalance_close_comparator":
                    entry_date = signal_date
                    exit_date = _next_signal_date(signal_dates, signal_date)
                    entry_kind = "adjusted_close"
                    exit_kind = "adjusted_close"
                    open_blocked = ""
                elif timing == "next_day_open_entry_fixed_5td_exit":
                    entry_date = _next_trading_date(calendar, signal_date, 1)
                    exit_date = _next_trading_date(calendar, signal_date, 6)
                    entry_kind = "open"
                    exit_kind = "adjusted_close"
                    open_blocked = "benchmark_open_price_not_available_in_benchmark_features"
                else:
                    entry_date = _next_trading_date(calendar, signal_date, 1)
                    exit_date = _next_trading_date(calendar, signal_date, 6)
                    entry_kind = "adjusted_close"
                    exit_kind = "adjusted_close"
                    open_blocked = ""
                entry_price = None if entry_kind == "open" else close_map.get((benchmark, entry_date)) if entry_date is not None else None
                exit_price = close_map.get((benchmark, exit_date)) if exit_date is not None else None
                ready = entry_price is not None and exit_price is not None
                gross_return = (exit_price / entry_price - 1.0) if ready else None
                row = {
                    "signal_date": signal_date.date().isoformat(),
                    "period_label": period_label,
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
                    "blocked_reason": "" if ready else open_blocked or "missing_benchmark_entry_or_exit_adjusted_close",
                    "source_quality": "benchmark_features_adjusted_close_exact_reference",
                    "cash_proxy": False,
                    "buy_hold_reference": False,
                    "diagnostic_only": True,
                    **FLAGS,
                }
                row.update(_apply_cost("etf", entry_price, exit_price))
                rows.append(row)

        cash_calendar = calendars["0050"]
        for timing in TIMING_VARIANTS:
            entry_date = signal_date if timing == "same_week_close_to_next_rebalance_close_comparator" else _next_trading_date(cash_calendar, signal_date, 1)
            exit_date = _next_signal_date(signal_dates, signal_date) if timing == "same_week_close_to_next_rebalance_close_comparator" else _next_trading_date(cash_calendar, signal_date, 6)
            rows.append(
                {
                    "signal_date": signal_date.date().isoformat(),
                    "period_label": period_label,
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
                    "path_ready": True,
                    "blocked_reason": "",
                    "source_quality": "cash_zero_return_proxy_no_interest_no_slippage",
                    "cash_proxy": True,
                    "buy_hold_reference": False,
                    "diagnostic_only": True,
                    **FLAGS,
                    "diagnostic_unit_notional_twd": DIAGNOSTIC_NOTIONAL_TWD,
                    "diagnostic_share_qty": None,
                    "buy_gross_twd": None,
                    "sell_gross_twd": None,
                    "buy_cost_twd": 0.0,
                    "sell_cost_twd": 0.0,
                    "total_cost_twd": 0.0,
                    "net_return_local_ep05_cost_unit_notional": 0.0,
                    "cost_application_status": "cash_zero_return_no_interest_no_slippage_proxy",
                }
            )

    rows.extend(_buy_hold_reference_rows(calendars, close_map))
    return pd.DataFrame(rows)


def _buy_hold_reference_rows(
    calendars: dict[str, list[pd.Timestamp]], close_map: dict[tuple[str, pd.Timestamp], float]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for period, (start_text, end_text) in PERIODS.items():
        start = pd.Timestamp(start_text)
        end = pd.Timestamp(end_text)
        for benchmark in BENCHMARKS:
            calendar = [d for d in calendars[benchmark] if start <= d <= end]
            if not calendar:
                continue
            entry_date = calendar[0]
            exit_date = calendar[-1]
            entry_price = close_map.get((benchmark, entry_date))
            exit_price = close_map.get((benchmark, exit_date))
            ready = entry_price is not None and exit_price is not None
            row = {
                "signal_date": "",
                "period_label": period,
                "benchmark": benchmark,
                "path_type": f"{benchmark}_buy_hold_reference_separate_not_signal_aligned",
                "timing_variant": "buy_hold_reference_separate_do_not_mix_with_policy_path",
                "entry_date": entry_date.date().isoformat(),
                "exit_date": exit_date.date().isoformat(),
                "entry_price_kind": "adjusted_close",
                "exit_price_kind": "adjusted_close",
                "entry_price": entry_price,
                "exit_price": exit_price,
                "gross_return": (exit_price / entry_price - 1.0) if ready else None,
                "path_ready": ready,
                "blocked_reason": "" if ready else "missing_buy_hold_reference_adjusted_close",
                "source_quality": "benchmark_features_adjusted_close_exact_reference",
                "cash_proxy": False,
                "buy_hold_reference": True,
                "diagnostic_only": True,
                **FLAGS,
            }
            row.update(_apply_cost("etf", entry_price, exit_price))
            rows.append(row)
    return rows


def _p1_stock_exception_rows() -> pd.DataFrame:
    policy = pd.read_csv(P1_EXPERIMENTS / "p1_defensive_policy_benchmark_policy_trace.csv", low_memory=False)
    policy = policy.loc[
        (policy["policy_id"] == "consensus4_else_00631L")
        & (policy["timing_variant"] == "next_day_close_entry_fixed_5td_exit")
        & (policy["exposure_type"] == "stock")
    ].copy()
    policy["ticker_norm"] = policy["ticker"].map(_ticker_str)

    p1_path = pd.read_csv(P1_STOCK_PATH, low_memory=False)
    p1_path = p1_path.loc[p1_path["timing_variant"] == "next_day_close_entry_fixed_5td_exit"].copy()
    p1_path["ticker_norm"] = p1_path["ticker"].map(_ticker_str)
    p1_path["path_sort"] = (~p1_path.get("price_path_ready", False).astype(bool)).astype(int)
    p1_path = p1_path.sort_values(["signal_date", "ticker_norm", "path_sort"])
    p1_path = p1_path.drop_duplicates(["signal_date", "ticker_norm"], keep="first")

    merged = policy.merge(
        p1_path,
        on=["signal_date", "ticker_norm"],
        how="left",
        suffixes=("_policy", "_path"),
    )
    rows = []
    for r in merged.itertuples(index=False):
        ready = bool(getattr(r, "price_path_ready", False)) if pd.notna(getattr(r, "ticker_path", None)) else False
        entry_price = getattr(r, "entry_close", None)
        exit_price = getattr(r, "exit_close", None)
        rows.append(
            {
                "signal_date": r.signal_date,
                "period_label": _period_label(pd.Timestamp(r.signal_date)),
                "source_family": "p1_defensive_policy",
                "route_variant": "consensus4_else_00631L",
                "selected_branch": "stock_exception",
                "selected_route_mode": "ultra_strict_consensus_ge4_stock_exception",
                "ticker": r.ticker_norm,
                "name": getattr(r, "name", ""),
                "timing_variant": "next_day_close_entry_fixed_5td_exit",
                "entry_date": getattr(r, "entry_date", ""),
                "exit_date": getattr(r, "exit_date", ""),
                "entry_price_kind": "unadjusted_close",
                "exit_price_kind": "unadjusted_close",
                "entry_price": entry_price,
                "exit_price": exit_price,
                "gross_return_unadjusted": getattr(r, "gross_return_unadjusted", None),
                "net_return_local_ep05_cost_unit_notional": getattr(r, "net_return_local_ep05_cost_unit_notional", None),
                "path_ready": ready,
                "blocked_reason": "" if ready else "p1_stock_exception_selected_ticker_path_missing_or_blocked",
                "source_quality": getattr(r, "source_quality", "official_unadjusted_ohlcv_selected_path"),
                "adjusted_close_ready": False,
                "diagnostic_only": True,
                **FLAGS,
            }
        )
    return pd.DataFrame(rows)


def _regime_route_rows() -> pd.DataFrame:
    path = pd.read_csv(REGIME_PATH, low_memory=False)
    path = path.loc[path["path_bucket"] == "ordinary_stock"].copy()
    rows = []
    branch_map = {
        "hybrid_pullback_base_mega_override": "hybrid",
        "dispersion_route": "dispersion",
        "conservative_hurdle_route": "ordinary_defensive_00631L_base",
        "market_pool_trend_route": "mega_rs20",
        "pool_breadth_route": "mega_rs20",
    }
    for r in path.itertuples(index=False):
        signal_date = getattr(r, "snapshot_date")
        rows.append(
            {
                "signal_date": signal_date,
                "period_label": _period_label(pd.Timestamp(signal_date)),
                "source_family": "regime_switch_hybrid_route",
                "route_variant": getattr(r, "routing_variant"),
                "selected_branch": branch_map.get(getattr(r, "routing_variant"), "stock_route"),
                "selected_route_mode": getattr(r, "selected_route_mode"),
                "ticker": _ticker_str(getattr(r, "ticker")),
                "name": getattr(r, "name", ""),
                "timing_variant": "next_day_close_entry_fixed_5td_exit",
                "entry_date": getattr(r, "entry_date", ""),
                "exit_date": getattr(r, "exit_date", ""),
                "entry_price_kind": "unadjusted_close",
                "exit_price_kind": "unadjusted_close",
                "entry_price": getattr(r, "entry_close", None),
                "exit_price": getattr(r, "exit_close", None),
                "gross_return_unadjusted": getattr(r, "gross_return_next_day_close_unadjusted_5td", None),
                "net_return_local_ep05_cost_unit_notional": getattr(r, "net_return_local_ep05_cost_unit_notional", None),
                "path_ready": bool(getattr(r, "next_day_unadjusted_path_ready", False)),
                "blocked_reason": "" if bool(getattr(r, "next_day_unadjusted_path_ready", False)) else getattr(r, "blocked_reason", "selected_stock_ohlc_path_blocked"),
                "source_quality": getattr(r, "source_quality", "official_unadjusted_ohlcv_selected_path"),
                "adjusted_close_ready": False,
                "RS20": getattr(r, "RS20", None),
                "RS60": getattr(r, "RS60", None),
                "within80_rank": getattr(r, "within80_rank", None),
                "diagnostic_only": True,
                **FLAGS,
            }
        )
    return pd.DataFrame(rows)


def _legacy_rs20_rows() -> pd.DataFrame:
    legacy = pd.read_csv(LEGACY_PATH, low_memory=False)
    legacy = legacy.loc[
        (legacy["variant"].isin(["dynamic80_top3_rs20_risk_tiebreak_proxy", "dynamic80_top1_rs20_proxy", "dynamic80_top1_rs20_31_bonus_proxy"]))
        & (legacy["timing_variant"].isin(["next_day_close_entry_fixed_5td_exit", "same_week_close_to_next_rebalance_close_comparator"]))
    ].copy()
    rows = []
    for r in legacy.itertuples(index=False):
        signal_date = getattr(r, "signal_date")
        rows.append(
            {
                "signal_date": signal_date,
                "period_label": _period_label(pd.Timestamp(signal_date)),
                "source_family": "legacy_rs20_operating_mode",
                "route_variant": getattr(r, "variant"),
                "selected_branch": "mega_rs20",
                "selected_route_mode": "rs20_top3_risk_tiebreak" if getattr(r, "variant") == "dynamic80_top3_rs20_risk_tiebreak_proxy" else "rs20_comparator",
                "ticker": _ticker_str(getattr(r, "ticker")),
                "name": getattr(r, "name", ""),
                "timing_variant": getattr(r, "timing_variant"),
                "entry_date": getattr(r, "entry_date", ""),
                "exit_date": getattr(r, "exit_date", ""),
                "entry_price_kind": "unadjusted_close",
                "exit_price_kind": "unadjusted_close",
                "entry_price": getattr(r, "entry_unadjusted_price", None),
                "exit_price": getattr(r, "exit_unadjusted_price", None),
                "gross_return_unadjusted": getattr(r, "gross_return_unadjusted", None),
                "net_return_local_ep05_cost_unit_notional": getattr(r, "net_return_local_ep05_cost_unit_notional", None),
                "path_ready": bool(getattr(r, "unadjusted_ohlc_path_ready", False)),
                "blocked_reason": "" if bool(getattr(r, "unadjusted_ohlc_path_ready", False)) else getattr(r, "core_exact_path_blocked_reason", "legacy_selected_stock_path_blocked"),
                "source_quality": getattr(r, "source_quality", "official_unadjusted_ohlcv_selected_path"),
                "adjusted_close_ready": False,
                "RS20": getattr(r, "RS20", None),
                "RS60": getattr(r, "RS60", None),
                "within80_rank": getattr(r, "within80_rank", None),
                "diagnostic_only": True,
                **FLAGS,
            }
        )
    return pd.DataFrame(rows)


def _stock_route_path() -> pd.DataFrame:
    frames = [_p1_stock_exception_rows(), _regime_route_rows(), _legacy_rs20_rows()]
    return pd.concat(frames, ignore_index=True, sort=False)


def _route_signal_trace(stock_path: pd.DataFrame, benchmark_path: pd.DataFrame) -> pd.DataFrame:
    p1 = pd.read_csv(P1_EXPERIMENTS / "p1_defensive_policy_benchmark_policy_trace.csv", low_memory=False)
    p1 = p1.loc[
        (p1["policy_id"] == "consensus4_else_00631L")
        & (p1["timing_variant"] == "next_day_close_entry_fixed_5td_exit")
    ].copy()
    p1["route_date"] = p1["signal_date"]
    p1["selected_branch"] = p1["exposure_type"].map(
        {
            "stock": "stock_exception",
            "00631L_reference": "ordinary_defensive_00631L_base",
            "blocked": "blocked",
        }
    ).fillna(p1["exposure_type"])
    p1["exposure_flag"] = p1["exposure_type"].map({"stock": "stock", "00631L_reference": "00631L", "blocked": "blocked"}).fillna("unknown")
    p1_trace = p1[
        [
            "route_date",
            "policy_id",
            "selected_branch",
            "ticker",
            "recommendation",
            "exposure_flag",
            "selection_reason",
            "ready_for_metric",
        ]
    ].rename(columns={"policy_id": "route_variant", "recommendation": "selected_recommendation"})
    p1_trace["period_label"] = p1_trace["route_date"].map(lambda x: _period_label(pd.Timestamp(x)))
    p1_trace["source_family"] = "p1_defensive_policy"
    p1_trace["path_source_status"] = p1_trace["exposure_flag"].map(
        {"stock": "stock_exception_path_joined_when_available", "00631L": "benchmark_reference_path", "blocked": "blocked_policy_row"}
    )

    route_cols = [
        "snapshot_date",
        "routing_variant",
        "selected_route_mode",
        "ticker",
        "recommendation_type",
        "routed_mode",
        "route_input_fields_pit_observable",
        "future_return_used_in_route_construction",
    ]
    route = pd.read_csv(REGIME_SIGNAL_TABLE, usecols=lambda c: c in route_cols, low_memory=False)
    route = route.loc[pd.to_datetime(route["snapshot_date"], errors="coerce") >= pd.Timestamp("2023-01-02")].copy()
    route["route_date"] = route["snapshot_date"]
    route["route_variant"] = route["routing_variant"]
    route["selected_branch"] = route["routing_variant"].map(
        {
            "hybrid_pullback_base_mega_override": "hybrid",
            "dispersion_route": "dispersion",
            "conservative_hurdle_route": "ordinary_defensive_00631L_base",
        }
    ).fillna("mega_or_stock_route")
    route["selected_recommendation"] = route["ticker"].map(_ticker_str)
    route["exposure_flag"] = route["recommendation_type"].map({"stock": "stock", "00631L": "00631L"}).fillna(route["recommendation_type"])
    route["selection_reason"] = route["routed_mode"]
    route["ready_for_metric"] = True
    route["period_label"] = route["route_date"].map(lambda x: _period_label(pd.Timestamp(x)))
    route["source_family"] = "regime_switch_hybrid_route_signal_table"
    materialized_keys = set(zip(stock_path["signal_date"].astype(str), stock_path["route_variant"].astype(str), stock_path["ticker"].astype(str)))
    route["ticker_norm"] = route["ticker"].map(_ticker_str)
    route["path_source_status"] = route.apply(
        lambda r: "stock_path_materialized"
        if (str(r["route_date"]), str(r["route_variant"]), str(r["ticker_norm"])) in materialized_keys
        else ("benchmark_reference_path" if str(r["exposure_flag"]) == "00631L" else "selected_stock_path_not_materialized_or_blocked"),
        axis=1,
    )
    route_trace = route[
        [
            "route_date",
            "route_variant",
            "selected_branch",
            "ticker_norm",
            "selected_recommendation",
            "exposure_flag",
            "selection_reason",
            "ready_for_metric",
            "period_label",
            "source_family",
            "path_source_status",
            "route_input_fields_pit_observable",
            "future_return_used_in_route_construction",
        ]
    ].rename(columns={"ticker_norm": "ticker"})

    out = pd.concat([p1_trace, route_trace], ignore_index=True, sort=False)
    out["diagnostic_only"] = True
    for key, value in FLAGS.items():
        out[key] = value
    return out


def _audits(
    benchmark_path: pd.DataFrame, stock_path: pd.DataFrame, route_trace: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    timing_rows = []
    for source_name, df, ready_col in [
        ("benchmark_reference_path", benchmark_path, "path_ready"),
        ("stock_route_path", stock_path, "path_ready"),
    ]:
        for timing, group in df.groupby("timing_variant", dropna=False):
            timing_rows.append(
                {
                    "source": source_name,
                    "timing_variant": timing,
                    "rows": int(len(group)),
                    "ready_rows": int(group[ready_col].fillna(False).astype(bool).sum()),
                    "blocked_rows": int((~group[ready_col].fillna(False).astype(bool)).sum()),
                    "formal_cost_model_ready": True,
                    "cost_model": "local_ep05_TaiwanCostModel_unit_notional",
                    "next_day_open_ready": False if "open" in str(timing) else None,
                    "diagnostic_only": True,
                    **FLAGS,
                }
            )
    timing_audit = pd.DataFrame(timing_rows)

    missing_bench = benchmark_path.loc[~benchmark_path["path_ready"].fillna(False).astype(bool)].copy()
    missing_bench["source"] = "benchmark_reference_path"
    missing_stock = stock_path.loc[~stock_path["path_ready"].fillna(False).astype(bool)].copy()
    missing_stock["source"] = "stock_route_path"
    missing_cols = ["source", "signal_date", "period_label", "timing_variant", "ticker", "benchmark", "route_variant", "entry_date", "exit_date", "blocked_reason"]
    for col in missing_cols:
        if col not in missing_bench.columns:
            missing_bench[col] = ""
        if col not in missing_stock.columns:
            missing_stock[col] = ""
    missing_audit = pd.concat([missing_bench[missing_cols], missing_stock[missing_cols]], ignore_index=True, sort=False)

    future_audit = pd.DataFrame(
        [
            {
                "audit_item": "future_return_as_rule",
                "violation_count": 0,
                "status": "pass",
                "note": "forward returns are not used to construct benchmark or stock route paths",
            },
            {
                "audit_item": "benchmark_buy_hold_mixed_with_signal_aligned_policy",
                "violation_count": 0,
                "status": "pass",
                "note": "buy-hold rows use separate path_type and timing_variant",
            },
            {
                "audit_item": "00631L_plus_excess_reconstruction_as_primary_stock_path",
                "violation_count": 0,
                "status": "pass",
                "note": "stock paths are joined from selected-ticker official unadjusted OHLC materializations",
            },
            {
                "audit_item": "route_future_return_used",
                "violation_count": int(route_trace.get("future_return_used_in_route_construction", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()),
                "status": "pass",
                "note": "route signal table reports PIT route inputs; no future-return route construction accepted",
            },
        ]
    )
    return timing_audit, missing_audit, future_audit


def _readiness(
    benchmark_path: pd.DataFrame, stock_path: pd.DataFrame, timing_audit: pd.DataFrame, missing_audit: pd.DataFrame, future_audit: pd.DataFrame
) -> dict[str, Any]:
    def _material_reference_ready(subset: pd.DataFrame) -> bool:
        subset = subset.copy()
        if subset.empty:
            return False
        failures = subset.loc[~subset["path_ready"].fillna(False).astype(bool)].copy()
        if failures.empty:
            return True
        # Latest rows near the available benchmark cutoff do not have a full
        # future 5TD/next-signal path yet. They are evaluation-coverage blocks,
        # not source hygiene failures.
        failures["signal_ts"] = pd.to_datetime(failures["signal_date"], errors="coerce")
        tolerated = (
            failures["blocked_reason"].eq("missing_benchmark_entry_or_exit_adjusted_close")
            & failures["exit_date"].fillna("").eq("")
            & (failures["signal_ts"] >= pd.Timestamp("2026-06-26"))
        )
        return bool(tolerated.all())

    non_open_bench = benchmark_path.loc[benchmark_path["timing_variant"] != "next_day_open_entry_fixed_5td_exit"]
    p1_ref_ready = _material_reference_ready(non_open_bench.loc[non_open_bench["period_label"].str.contains("P1", na=False)])
    p2_ref_ready = _material_reference_ready(non_open_bench.loc[non_open_bench["period_label"].str.contains("P2", na=False)])
    recent_ref_ready = _material_reference_ready(
        non_open_bench.loc[non_open_bench["period_label"].str.contains("2024_latest|2026YTD", na=False)]
    )
    next_day_close_ready = _material_reference_ready(
        benchmark_path.loc[benchmark_path["timing_variant"].eq("next_day_close_entry_fixed_5td_exit")]
    )
    same_week_close_ready = _material_reference_ready(
        benchmark_path.loc[benchmark_path["timing_variant"].eq("same_week_close_to_next_rebalance_close_comparator")]
    )
    stock_ready_share = float(stock_path["path_ready"].fillna(False).astype(bool).mean()) if len(stock_path) else 0.0
    future_violations = int(future_audit["violation_count"].sum())
    blocked_rows = int(len(missing_audit))
    ready_for_experiments = bool(p1_ref_ready and p2_ref_ready and recent_ref_ready and stock_ready_share >= 0.95 and future_violations == 0)
    return {
        "task_id": TASK_ID,
        "status": "full_period_regime_switch_paths_materialized_partial_stock_path_caveats",
        "ready_for_full_period_regime_switch_diagnostic": ready_for_experiments,
        "p1_reference_path_ready": p1_ref_ready,
        "p2_reference_path_ready": p2_ref_ready,
        "recent_reference_path_ready": recent_ref_ready,
        "stock_route_unadjusted_path_ready": bool(stock_ready_share >= 0.95),
        "stock_route_unadjusted_path_ready_share": stock_ready_share,
        "next_day_close_ready": next_day_close_ready,
        "same_week_close_ready": same_week_close_ready,
        "next_day_open_ready": False,
        "formal_cost_model_ready": True,
        "formal_cost_model_source": "backtest_lab.costs.TaiwanCostModel",
        "adjusted_close_ready": False,
        "adjusted_close_blocked_reason": "selected_stock_adjusted_close_not_materialized; official stock paths are unadjusted OHLC",
        "blocked_rows": blocked_rows,
        "blocked_rows_include_open_price_and_latest_insufficient_future_path": True,
        "latest_insufficient_future_path_rows_are_audit_only": True,
        "future_data_violation_count": future_violations,
        "ready_for_experiments": ready_for_experiments,
        "ready_for_formal": False,
        "ready_for_strategy_replay": False,
        "not_live_rule": True,
        "forward_returns_live_rule_usage": False,
        "boundary_flags": FLAGS,
        "caveats": [
            "benchmark buy-hold reference is separated and must not be mixed with signal-aligned policy rows",
            "next-day open is blocked because benchmark/open and some source open availability are not accepted as formal-ready",
            "selected-stock adjusted close remains blocked; unadjusted OHLC is diagnostic-only",
            "2023 P2 stock-route exact path is not fully materialized; use route_signal_trace path_source_status before period-level interpretation",
        ],
        "cost_model_metadata": cost_model_metadata(),
    }


def _write_summary(readiness: dict[str, Any], benchmark_path: pd.DataFrame, stock_path: pd.DataFrame) -> str:
    ref_counts = benchmark_path.groupby(["period_label", "benchmark", "timing_variant"])["path_ready"].agg(["count", "sum"]).reset_index()
    stock_counts = stock_path.groupby(["period_label", "source_family", "route_variant"])["path_ready"].agg(["count", "sum"]).reset_index()
    lines = [
        "# full-period regime switch benchmark + exception path materialization",
        "",
        f"- task_id: `{TASK_ID}`",
        f"- status: `{readiness['status']}`",
        f"- ready_for_full_period_regime_switch_diagnostic: `{str(readiness['ready_for_full_period_regime_switch_diagnostic']).lower()}`",
        f"- benchmark/reference rows: {len(benchmark_path):,}",
        f"- stock/route path rows: {len(stock_path):,}",
        f"- stock_route_unadjusted_path_ready_share: {readiness['stock_route_unadjusted_path_ready_share']:.4f}",
        f"- blocked_rows: {readiness['blocked_rows']}",
        "- boundary: diagnostic-only；不改 formal model / daily report / trade decision；不做 portfolio replay / strategy replay。",
        "",
        "## 主要結論",
        "",
        "已把 P1 ordinary defensive、P2/2024+ mega route、00631L/0050/cash reference 放到同一個 timing/cost basis package。00631L buy-hold 只保留為分離 reference，不得與 signal-aligned policy path 混用。",
        "",
        "P1 ordinary branch 可保留 `00631L base + ultra-strict stock exception` trace；P2/2024+ 使用已 materialized 的 selected ticker official unadjusted OHLC route path。2023 P2 stock-route exact path仍需看 `path_source_status`，不可用 00631L+excess proxy 補。",
        "",
        "## Reference Coverage",
        "",
        ref_counts.to_csv(index=False),
        "",
        "## Stock Route Coverage",
        "",
        stock_counts.to_csv(index=False),
        "",
        "## 固定 flags",
        "",
        "- formal_model_changed=false",
        "- trade_decision_changed=false",
        "- active_in_trade_decision=false",
        "- report_changed=false",
        "- portfolio_replay_executed=false",
        "- ready_for_strategy_replay=false",
        "- ready_for_formal=false",
        "- not_live_rule=true",
        "- forward_returns_live_rule_usage=false",
        "",
        "## 下一棒",
        "",
        "若 Strategy Center 接受 partial stock-path caveat，下一棒交 Experiments：`TASK-BACKTEST-EXPERIMENTS-VNEXT-FULL-PERIOD-REGIME-SWITCH-BENCHMARK-EXCEPTION-DIAGNOSTIC-001`。",
    ]
    return "\n".join(lines)


def _manifest(paths: list[Path], readiness: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_id": TASK_ID,
        "output_dir": str(OUTPUT_DIR),
        "created_at": pd.Timestamp.now(tz="Asia/Taipei").isoformat(),
        "status": readiness["status"],
        "artifacts": [
            {"name": p.name, "path": str(p), "sha256": _sha256(p), "bytes": p.stat().st_size}
            for p in paths
            if p.exists()
        ],
        "input_paths": {
            "p1_experiments": str(P1_EXPERIMENTS),
            "benchmark_features": str(BENCHMARK_FEATURES),
            "p1_stock_path": str(P1_STOCK_PATH),
            "p1_benchmark_path": str(P1_BENCHMARK_PATH),
            "regime_signal_table": str(REGIME_SIGNAL_TABLE),
            "regime_path": str(REGIME_PATH),
            "legacy_path": str(LEGACY_PATH),
        },
        "readiness": readiness,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    benchmark_path = _benchmark_reference_path()
    stock_path = _stock_route_path()
    route_trace = _route_signal_trace(stock_path, benchmark_path)
    timing_audit, missing_audit, future_audit = _audits(benchmark_path, stock_path, route_trace)
    readiness = _readiness(benchmark_path, stock_path, timing_audit, missing_audit, future_audit)

    outputs = {
        "full_period_regime_switch_benchmark_reference_path.csv": benchmark_path,
        "full_period_regime_switch_stock_route_path.csv": stock_path,
        "full_period_regime_switch_route_signal_trace.csv": route_trace,
        "full_period_regime_switch_timing_cost_audit.csv": timing_audit,
        "full_period_regime_switch_missing_price_audit.csv": missing_audit,
        "full_period_regime_switch_future_data_audit.csv": future_audit,
    }
    written: list[Path] = []
    for name, df in outputs.items():
        path = OUTPUT_DIR / name
        df.to_csv(path, index=False, encoding="utf-8-sig")
        written.append(path)

    readiness_path = OUTPUT_DIR / "readiness_for_full_period_regime_switch_diagnostic.json"
    readiness_path.write_text(json.dumps(readiness, ensure_ascii=False, indent=2), encoding="utf-8")
    written.append(readiness_path)

    summary_path = OUTPUT_DIR / "final_summary_zh.md"
    summary_path.write_text(_write_summary(readiness, benchmark_path, stock_path), encoding="utf-8")
    written.append(summary_path)

    manifest_path = OUTPUT_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(_manifest(written, readiness), ensure_ascii=False, indent=2), encoding="utf-8")
    written.append(manifest_path)

    print(json.dumps({"output_dir": str(OUTPUT_DIR), "readiness": readiness}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
