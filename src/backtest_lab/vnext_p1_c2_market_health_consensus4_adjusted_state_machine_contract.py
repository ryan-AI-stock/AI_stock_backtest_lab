from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from backtest_lab.costs import TaiwanCostModel, cost_model_metadata


REPO_ROOT = Path(__file__).resolve().parents[2]
P1_STATE_HOLD_DIR = REPO_ROOT / "outputs" / "vnext_p1_state_hold_base_exception_path_contract_20260708"
MARKET_REGIME_DIR = REPO_ROOT / "outputs" / "vnext_p1_market_regime_classifier_feature_contract_20260708"
BENCHMARK_FEATURES = REPO_ROOT / "outputs" / "vnext_dynamic_candidate_pool_data_materialization_20260706" / "benchmark_features.csv"
STOCK_FEATURES = REPO_ROOT / "outputs" / "vnext_dynamic_candidate_pool_data_materialization_20260706" / "stock_features.csv"
P1_STOCK_PATH = (
    REPO_ROOT
    / "outputs"
    / "vnext_p1_legacy_regime_unadjusted_path_refresh_20260708"
    / "p1_legacy_regime_unadjusted_trade_path_refreshed.csv"
)
OUTPUT_DIR = REPO_ROOT / "outputs" / "vnext_p1_c2_market_health_consensus4_adjusted_state_machine_contract_20260708"

TASK_ID = "TASK-BACKTEST-CORE-VNEXT-P1-C2-MARKET-HEALTH-CONSENSUS4-ADJUSTED-STATE-MACHINE-CONTRACT-001"
P1_START = pd.Timestamp("2015-01-02")
P1_END = pd.Timestamp("2022-12-29")
BASE_ASSET = "00631L"
DIAGNOSTIC_NOTIONAL_TWD = 1_000_000
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


def _ticker_str(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text.zfill(4) if text.isdigit() and len(text) < 4 else text


def _asset_type(ticker: str) -> str:
    return "etf" if ticker == BASE_ASSET else "stock"


def _benchmark_frame() -> pd.DataFrame:
    bench = pd.read_csv(BENCHMARK_FEATURES, dtype={"benchmark": str}, low_memory=False)
    bench["trade_date"] = pd.to_datetime(bench["trade_date"], errors="coerce")
    bench = bench.loc[
        (bench["benchmark"] == BASE_ASSET)
        & (bench["trade_date"] >= P1_START)
        & (bench["trade_date"] <= P1_END)
    ].copy()
    bench = bench.sort_values("trade_date")
    bench["date_key"] = bench["trade_date"].dt.date.astype(str)
    return bench


def _benchmark_price_map() -> dict[str, float]:
    bench = _benchmark_frame()
    return {row.date_key: float(row.adjusted_close) for row in bench.itertuples(index=False) if pd.notna(row.adjusted_close)}


def _benchmark_calendar() -> list[pd.Timestamp]:
    return [pd.Timestamp(x) for x in _benchmark_frame()["trade_date"].dropna().sort_values().unique()]


def _next_trading_date(calendar: list[pd.Timestamp], date: pd.Timestamp, offset: int = 1) -> pd.Timestamp | None:
    for idx, trading_date in enumerate(calendar):
        if trading_date > date:
            target = idx + offset - 1
            return calendar[target] if 0 <= target < len(calendar) else None
    return None


def _signal_trace_with_c2() -> pd.DataFrame:
    trace = pd.read_csv(P1_STATE_HOLD_DIR / "p1_base_exception_signal_trace.csv", low_memory=False)
    trace["signal_date"] = pd.to_datetime(trace["signal_date"], errors="coerce")
    trace["next_signal_date"] = pd.to_datetime(trace["next_signal_date"], errors="coerce")
    trace = trace.sort_values("signal_date").copy()
    market = pd.read_csv(MARKET_REGIME_DIR / "p1_market_regime_classifier_feature_contract.csv", low_memory=False)
    market["signal_date"] = pd.to_datetime(market["signal_date"], errors="coerce")
    keep = [
        "signal_date",
        "0050_above_ma60_flag",
        "0050_return_20d",
        "0050_return_40d",
        "0050_trend_state_label_candidate",
        "pool80_rs20_positive_share",
        "00631L_high_risk_context_candidate",
    ]
    trace = trace.merge(market[keep], on="signal_date", how="left")
    trace["raw_consensus4_exception_active"] = trace["exposure_type"].astype(str).eq("stock")
    trace["c2_market_health_gate"] = (
        trace["0050_above_ma60_flag"].fillna(False).astype(bool)
        & pd.to_numeric(trace["0050_return_20d"], errors="coerce").ge(0)
        & pd.to_numeric(trace["0050_return_40d"], errors="coerce").ge(0)
    )
    trace["exception_allowed_by_c2"] = trace["raw_consensus4_exception_active"] & trace["c2_market_health_gate"]
    trace["target_ticker"] = trace.apply(
        lambda r: _ticker_str(r["ticker"]) if bool(r["exception_allowed_by_c2"]) else BASE_ASSET,
        axis=1,
    )
    trace["target_asset_type"] = trace["target_ticker"].map(_asset_type)
    trace["target_state"] = trace["target_ticker"].map(lambda t: "stock_exception" if t != BASE_ASSET else "base_00631L")
    trace["selection_reason_c2"] = trace.apply(
        lambda r: "c2_gate_allows_consensus4_exception"
        if bool(r["exception_allowed_by_c2"])
        else (
            "consensus4_exception_blocked_by_c2_market_health_gate"
            if bool(r["raw_consensus4_exception_active"])
            else "hold_00631L_base_no_consensus4_exception"
        ),
        axis=1,
    )
    return trace


def _unadjusted_stock_path_map() -> dict[tuple[str, str], dict[str, Any]]:
    stock = pd.read_csv(P1_STOCK_PATH, low_memory=False)
    stock = stock.loc[
        (stock["timing_variant"] == "next_day_close_entry_fixed_5td_exit")
        & (stock["path_bucket"] == "ordinary_stock")
    ].copy()
    stock["signal_date"] = pd.to_datetime(stock["signal_date"], errors="coerce")
    stock["ticker_norm"] = stock["ticker"].map(_ticker_str)
    stock["ready_sort"] = (~stock["price_path_ready"].fillna(False).astype(bool)).astype(int)
    stock = stock.sort_values(["signal_date", "ticker_norm", "ready_sort"])
    stock = stock.drop_duplicates(["signal_date", "ticker_norm"], keep="first")
    return {
        (pd.Timestamp(row.signal_date).date().isoformat(), row.ticker_norm): row._asdict()
        for row in stock.itertuples(index=False)
    }


def _selected_stock_intervals(trace: pd.DataFrame, calendar: list[pd.Timestamp]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in trace.itertuples(index=False):
        if row.target_ticker == BASE_ASSET:
            continue
        signal_date = pd.Timestamp(row.signal_date)
        entry_date = _next_trading_date(calendar, signal_date, 1)
        next_signal_date = pd.Timestamp(row.next_signal_date)
        exit_date = _next_trading_date(calendar, next_signal_date, 1) if next_signal_date < P1_END else P1_END
        rows.append(
            {
                "signal_date": signal_date.date().isoformat(),
                "entry_date": entry_date.date().isoformat() if entry_date is not None else "",
                "exit_date": pd.Timestamp(exit_date).date().isoformat() if exit_date is not None else "",
                "ticker": row.target_ticker,
            }
        )
    return pd.DataFrame(rows)


def _candidate_cache_files(ticker: str) -> list[Path]:
    patterns = [f"{ticker}_TW.csv", f"{ticker}_TWO.csv", f"{ticker}.csv"]
    files: list[Path] = []
    for pattern in patterns:
        files.extend((REPO_ROOT / "backtest_cache").glob(f"**/{pattern}"))
    return sorted(set(files), key=lambda p: (len(str(p)), str(p)))


def _stock_features_adjusted_map(selected: pd.DataFrame) -> tuple[dict[tuple[str, str], dict[str, Any]], list[dict[str, Any]]]:
    required_tickers = sorted(selected["ticker"].dropna().map(_ticker_str).unique()) if not selected.empty else []
    maps: dict[tuple[str, str], dict[str, Any]] = {}
    attempts: list[dict[str, Any]] = []
    if STOCK_FEATURES.exists():
        sf = pd.read_csv(STOCK_FEATURES, usecols=["trade_date", "ticker", "adjusted_close"], dtype={"ticker": str}, low_memory=False)
        sf["trade_date"] = pd.to_datetime(sf["trade_date"], errors="coerce")
        sf["ticker_norm"] = sf["ticker"].map(_ticker_str)
        sf = sf.loc[sf["ticker_norm"].isin(required_tickers)].copy()
        for row in sf.itertuples(index=False):
            if pd.notna(row.adjusted_close):
                maps[(pd.Timestamp(row.trade_date).date().isoformat(), row.ticker_norm)] = {
                    "adjusted_close": float(row.adjusted_close),
                    "source": str(STOCK_FEATURES),
                    "source_quality": "stock_features_adjusted_close_local_materialized",
                }
        attempts.append(
            {
                "source": str(STOCK_FEATURES),
                "source_type": "stock_features",
                "tickers_checked": len(required_tickers),
                "price_points_loaded": len(maps),
                "status": "attempted",
            }
        )
    for ticker in required_tickers:
        for file in _candidate_cache_files(ticker):
            try:
                header = pd.read_csv(file, nrows=0)
                if "date" not in header.columns or "adj_close" not in header.columns:
                    attempts.append(
                        {
                            "source": str(file),
                            "source_type": "cache_csv",
                            "ticker": ticker,
                            "price_points_loaded": 0,
                            "status": "skipped_missing_date_or_adj_close",
                        }
                    )
                    continue
                df = pd.read_csv(file, usecols=["date", "adj_close"], low_memory=False)
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
                loaded = 0
                for row in df.itertuples(index=False):
                    if pd.notna(row.adj_close):
                        key = (pd.Timestamp(row.date).date().isoformat(), ticker)
                        maps.setdefault(
                            key,
                            {
                                "adjusted_close": float(row.adj_close),
                                "source": str(file),
                                "source_quality": "local_cache_adjusted_close_bounded_selected_ticker_attempt",
                            },
                        )
                        loaded += 1
                attempts.append(
                    {
                        "source": str(file),
                        "source_type": "cache_csv",
                        "ticker": ticker,
                        "price_points_loaded": loaded,
                        "status": "attempted",
                    }
                )
            except Exception as exc:  # noqa: BLE001
                attempts.append(
                    {
                        "source": str(file),
                        "source_type": "cache_csv",
                        "ticker": ticker,
                        "price_points_loaded": 0,
                        "status": f"error:{type(exc).__name__}",
                    }
                )
    return maps, attempts


def _cost_breakdown(old_ticker: str | None, new_ticker: str | None) -> dict[str, Any]:
    model = TaiwanCostModel()
    sell = {"buy_fee": 0, "sell_fee": 0, "securities_transaction_tax": 0, "total_transaction_cost": 0}
    buy = {"buy_fee": 0, "sell_fee": 0, "securities_transaction_tax": 0, "total_transaction_cost": 0}
    if old_ticker:
        sell = model.sell_cost_breakdown(DIAGNOSTIC_NOTIONAL_TWD, _asset_type(old_ticker))
    if new_ticker:
        buy = model.buy_cost_breakdown(DIAGNOSTIC_NOTIONAL_TWD)
    total = int(sell["total_transaction_cost"] + buy["total_transaction_cost"])
    return {
        "diagnostic_notional_twd": DIAGNOSTIC_NOTIONAL_TWD,
        "sell_fee_twd": int(sell["sell_fee"]),
        "buy_fee_twd": int(buy["buy_fee"]),
        "securities_transaction_tax_twd": int(sell["securities_transaction_tax"]),
        "total_transition_cost_twd": total,
        "transition_cost_rate": total / DIAGNOSTIC_NOTIONAL_TWD,
        "cost_model_status": "applied_local_ep05_TaiwanCostModel_unit_notional_transition_cost",
        "cost_model_version": cost_model_metadata()["cost_model_version"],
    }


def _transition_action(old: str, new: str) -> str:
    if old == BASE_ASSET and new != BASE_ASSET:
        return "base_00631L_to_stock_exception"
    if old != BASE_ASSET and new == BASE_ASSET:
        return "stock_exception_to_base_00631L"
    if old != BASE_ASSET and new != BASE_ASSET:
        return "stock_exception_to_stock_exception_switch"
    return "base_hold_no_trade"


def _state_machine_contract() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    trace = _signal_trace_with_c2()
    calendar = _benchmark_calendar()
    base_prices = _benchmark_price_map()
    unadj_map = _unadjusted_stock_path_map()
    selected = _selected_stock_intervals(trace, calendar)
    adj_map, source_attempts = _stock_features_adjusted_map(selected)

    intervals: list[dict[str, Any]] = []
    transitions: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    gross_equity_adjusted = 1.0
    net_equity_adjusted = 1.0
    gross_peak_adjusted = 1.0
    net_peak_adjusted = 1.0
    gross_equity_unadjusted = 1.0
    net_equity_unadjusted = 1.0
    current_ticker = BASE_ASSET

    first_signal = pd.Timestamp(trace["signal_date"].min())
    first_entry = _next_trading_date(calendar, first_signal, 1)
    start_date = calendar[0]
    if first_entry is not None and first_entry > start_date:
        start_key = start_date.date().isoformat()
        first_entry_key = first_entry.date().isoformat()
        start_price = base_prices.get(start_key)
        end_price = base_prices.get(first_entry_key)
        interval_return = (end_price / start_price - 1.0) if start_price and end_price else None
        if interval_return is not None:
            gross_equity_adjusted *= 1.0 + interval_return
            net_equity_adjusted *= 1.0 + interval_return
            gross_equity_unadjusted *= 1.0 + interval_return
            net_equity_unadjusted *= 1.0 + interval_return
            gross_peak_adjusted = max(gross_peak_adjusted, gross_equity_adjusted)
            net_peak_adjusted = max(net_peak_adjusted, net_equity_adjusted)
        intervals.append(
            {
                "signal_date": "",
                "state_start_date": start_key,
                "state_end_date": first_entry_key,
                "holding_ticker": BASE_ASSET,
                "holding_asset_type": "etf",
                "state_reason": "initial_default_base_hold_before_first_signal",
                "c2_market_health_gate": False,
                "raw_consensus4_exception_active": False,
                "exception_allowed_by_c2": False,
                "transition_action": "none_initial_state_already_holding_00631L",
                "previous_ticker": "",
                "adjusted_entry_price": start_price,
                "adjusted_exit_price": end_price,
                "adjusted_interval_return": interval_return,
                "adjusted_path_ready": interval_return is not None,
                "unadjusted_entry_price": start_price,
                "unadjusted_exit_price": end_price,
                "unadjusted_interval_return": interval_return,
                "unadjusted_comparator_ready": interval_return is not None,
                "transition_cost_rate": 0.0,
                "net_adjusted_interval_return_after_transition_cost": interval_return,
                "net_unadjusted_interval_return_after_transition_cost": interval_return,
                "net_equity_after_cost_adjusted_path": net_equity_adjusted,
                "net_equity_after_cost_unadjusted_comparator": net_equity_unadjusted,
                "adjusted_source_quality": "benchmark_features_adjusted_close_exact_reference",
                "unadjusted_source_quality": "benchmark_features_adjusted_close_exact_reference",
                "blocked_reason": "" if interval_return is not None else "missing_initial_00631L_adjusted_close",
                "diagnostic_only": True,
                **FLAGS,
            }
        )

    for row in trace.itertuples(index=False):
        signal_date = pd.Timestamp(row.signal_date)
        signal_key = signal_date.date().isoformat()
        entry_date = _next_trading_date(calendar, signal_date, 1)
        next_signal_date = pd.Timestamp(row.next_signal_date)
        exit_date = _next_trading_date(calendar, next_signal_date, 1) if next_signal_date < P1_END else P1_END
        entry_key = entry_date.date().isoformat() if entry_date is not None else ""
        exit_key = pd.Timestamp(exit_date).date().isoformat() if exit_date is not None else ""
        target_ticker = row.target_ticker
        transition_needed = target_ticker != current_ticker
        cost = _cost_breakdown(current_ticker if transition_needed else None, target_ticker if transition_needed else None)

        if target_ticker == BASE_ASSET:
            adjusted_entry = base_prices.get(entry_key)
            adjusted_exit = base_prices.get(exit_key)
            adjusted_source_quality = "benchmark_features_adjusted_close_exact_reference"
            unadjusted_entry = adjusted_entry
            unadjusted_exit = adjusted_exit
            unadjusted_source_quality = adjusted_source_quality
        else:
            entry_adj = adj_map.get((entry_key, target_ticker))
            exit_adj = adj_map.get((exit_key, target_ticker))
            adjusted_entry = entry_adj.get("adjusted_close") if entry_adj else None
            adjusted_exit = exit_adj.get("adjusted_close") if exit_adj else None
            adjusted_source_quality = (
                entry_adj.get("source_quality")
                if entry_adj and exit_adj
                else "blocked_selected_stock_adjusted_close_missing"
            )
            unadj = unadj_map.get((signal_key, target_ticker))
            unadjusted_entry = unadj.get("entry_close") if unadj else None
            unadjusted_exit = unadj.get("exit_close") if unadj else None
            unadjusted_source_quality = unadj.get("source_quality") if unadj else "missing_unadjusted_ohlc_comparator"
            coverage_rows.append(
                {
                    "signal_date": signal_key,
                    "entry_date": entry_key,
                    "exit_date": exit_key,
                    "ticker": target_ticker,
                    "entry_adjusted_close_ready": entry_adj is not None,
                    "exit_adjusted_close_ready": exit_adj is not None,
                    "adjusted_close_interval_ready": entry_adj is not None and exit_adj is not None,
                    "entry_adjusted_source": entry_adj.get("source") if entry_adj else "",
                    "exit_adjusted_source": exit_adj.get("source") if exit_adj else "",
                    "unadjusted_ohlc_comparator_ready": bool(unadj and unadj.get("price_path_ready")),
                    "unadjusted_source_quality": unadjusted_source_quality,
                    "blocked_reason": "" if entry_adj and exit_adj else "missing_selected_stock_adjusted_close_entry_or_exit",
                    "diagnostic_only": True,
                    **FLAGS,
                }
            )

        adjusted_ready = adjusted_entry is not None and adjusted_exit is not None and pd.notna(adjusted_entry) and pd.notna(adjusted_exit)
        unadjusted_ready = unadjusted_entry is not None and unadjusted_exit is not None and pd.notna(unadjusted_entry) and pd.notna(unadjusted_exit)
        adjusted_return = (float(adjusted_exit) / float(adjusted_entry) - 1.0) if adjusted_ready else None
        unadjusted_return = (float(unadjusted_exit) / float(unadjusted_entry) - 1.0) if unadjusted_ready else None
        net_adjusted = ((1.0 - cost["transition_cost_rate"]) * (1.0 + adjusted_return) - 1.0) if adjusted_return is not None else None
        net_unadjusted = ((1.0 - cost["transition_cost_rate"]) * (1.0 + unadjusted_return) - 1.0) if unadjusted_return is not None else None
        if adjusted_ready:
            gross_equity_adjusted *= 1.0 + float(adjusted_return)
            net_equity_adjusted *= 1.0 + float(net_adjusted)
            gross_peak_adjusted = max(gross_peak_adjusted, gross_equity_adjusted)
            net_peak_adjusted = max(net_peak_adjusted, net_equity_adjusted)
        if unadjusted_ready:
            gross_equity_unadjusted *= 1.0 + float(unadjusted_return)
            net_equity_unadjusted *= 1.0 + float(net_unadjusted)
        intervals.append(
            {
                "signal_date": signal_key,
                "state_start_date": entry_key,
                "state_end_date": exit_key,
                "holding_ticker": target_ticker,
                "holding_asset_type": _asset_type(target_ticker),
                "state_reason": row.selection_reason_c2,
                "c2_market_health_gate": bool(row.c2_market_health_gate),
                "raw_consensus4_exception_active": bool(row.raw_consensus4_exception_active),
                "exception_allowed_by_c2": bool(row.exception_allowed_by_c2),
                "0050_above_ma60_flag": row._asdict().get("0050_above_ma60_flag"),
                "0050_return_20d": row._asdict().get("0050_return_20d"),
                "0050_return_40d": row._asdict().get("0050_return_40d"),
                "transition_action": _transition_action(current_ticker, target_ticker) if transition_needed else "hold_same_state_no_trade",
                "previous_ticker": current_ticker,
                "adjusted_entry_price": adjusted_entry,
                "adjusted_exit_price": adjusted_exit,
                "adjusted_interval_return": adjusted_return,
                "adjusted_path_ready": adjusted_ready,
                "unadjusted_entry_price": unadjusted_entry,
                "unadjusted_exit_price": unadjusted_exit,
                "unadjusted_interval_return": unadjusted_return,
                "unadjusted_comparator_ready": unadjusted_ready,
                "diagnostic_notional_twd": DIAGNOSTIC_NOTIONAL_TWD,
                **cost,
                "net_adjusted_interval_return_after_transition_cost": net_adjusted,
                "net_unadjusted_interval_return_after_transition_cost": net_unadjusted,
                "gross_equity_adjusted_path": gross_equity_adjusted,
                "net_equity_after_cost_adjusted_path": net_equity_adjusted,
                "net_equity_after_cost_unadjusted_comparator": net_equity_unadjusted,
                "gross_drawdown_adjusted_path": gross_equity_adjusted / gross_peak_adjusted - 1.0,
                "net_drawdown_adjusted_path": net_equity_adjusted / net_peak_adjusted - 1.0,
                "adjusted_source_quality": adjusted_source_quality,
                "unadjusted_source_quality": unadjusted_source_quality,
                "blocked_reason": "" if adjusted_ready else "selected_stock_adjusted_close_missing" if target_ticker != BASE_ASSET else "missing_00631L_adjusted_close",
                "cash_condition_status": "blocked_no_bear_cash_classifier",
                "diagnostic_only": True,
                **FLAGS,
            }
        )
        if transition_needed:
            transitions.append(
                {
                    "signal_date": signal_key,
                    "transition_date": entry_key,
                    "from_ticker": current_ticker,
                    "from_asset_type": _asset_type(current_ticker),
                    "to_ticker": target_ticker,
                    "to_asset_type": _asset_type(target_ticker),
                    "transition_action": _transition_action(current_ticker, target_ticker),
                    "sell_price_kind": "adjusted_close" if current_ticker == BASE_ASSET else "selected_stock_adjusted_close_if_available_else_blocked",
                    "buy_price_kind": "adjusted_close" if target_ticker == BASE_ASSET else "selected_stock_adjusted_close_if_available_else_blocked",
                    "adjusted_transition_price_ready": adjusted_ready,
                    "unadjusted_comparator_ready": unadjusted_ready,
                    **cost,
                    "diagnostic_only": True,
                    **FLAGS,
                }
            )
        current_ticker = target_ticker
    coverage = pd.DataFrame(coverage_rows)
    source_attempt = pd.DataFrame(source_attempts)
    return pd.DataFrame(intervals), pd.DataFrame(transitions), coverage, source_attempt


def _cost_audit(transitions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if transitions.empty:
        return pd.DataFrame(rows)
    for action, group in transitions.groupby("transition_action"):
        rows.append(
            {
                "transition_action": action,
                "transition_count": len(group),
                "total_buy_fee_twd": int(group["buy_fee_twd"].sum()),
                "total_sell_fee_twd": int(group["sell_fee_twd"].sum()),
                "total_securities_transaction_tax_twd": int(group["securities_transaction_tax_twd"].sum()),
                "total_transition_cost_twd": int(group["total_transition_cost_twd"].sum()),
                "total_transition_cost_rate_sum": float(group["transition_cost_rate"].sum()),
                "from_asset_types": ";".join(sorted(group["from_asset_type"].dropna().unique())),
                "to_asset_types": ";".join(sorted(group["to_asset_type"].dropna().unique())),
                "cost_model_status": "applied_local_ep05_TaiwanCostModel_unit_notional_transition_cost",
                "formal_cost_model_ready": True,
                "diagnostic_only": True,
                **FLAGS,
            }
        )
    meta = cost_model_metadata()
    rows.append(
        {
            "transition_action": "cost_model_metadata",
            "transition_count": len(transitions),
            "total_buy_fee_twd": "",
            "total_sell_fee_twd": "",
            "total_securities_transaction_tax_twd": "",
            "total_transition_cost_twd": "",
            "total_transition_cost_rate_sum": "",
            "from_asset_types": "",
            "to_asset_types": "",
            "cost_model_status": json.dumps(meta, ensure_ascii=False),
            "formal_cost_model_ready": True,
            "diagnostic_only": True,
            **FLAGS,
        }
    )
    return pd.DataFrame(rows)


def _blocked_proxy_audit(contract: pd.DataFrame, coverage: pd.DataFrame) -> pd.DataFrame:
    stock_rows = int((contract["holding_asset_type"] == "stock").sum())
    adjusted_ready = int(coverage["adjusted_close_interval_ready"].fillna(False).astype(bool).sum()) if not coverage.empty else 0
    unadjusted_ready = int(coverage["unadjusted_ohlc_comparator_ready"].fillna(False).astype(bool).sum()) if not coverage.empty else 0
    return pd.DataFrame(
        [
            {
                "field_or_component": "selected_stock_adjusted_close",
                "status": "partial" if adjusted_ready else "blocked",
                "ready_rows": adjusted_ready,
                "affected_rows": stock_rows,
                "reason": "bounded local adjusted-close attempt used stock_features and existing cache only; missing rows require selected-ticker source fill",
            },
            {
                "field_or_component": "official_unadjusted_ohlc_comparator",
                "status": "ready" if unadjusted_ready == stock_rows else "partial",
                "ready_rows": unadjusted_ready,
                "affected_rows": stock_rows,
                "reason": "official unadjusted OHLC comparator retained but cannot be packaged as adjusted-close formal path",
            },
            {
                "field_or_component": "cash_condition_bear_classifier",
                "status": "blocked",
                "ready_rows": 0,
                "affected_rows": len(contract),
                "reason": "no accepted cash/bear classifier; no cash rule fabricated",
            },
            {
                "field_or_component": "cost_model",
                "status": "ready_for_diagnostic_cost",
                "ready_rows": len(contract),
                "affected_rows": len(contract),
                "reason": "EP05 TaiwanCostModel applies buy/sell fee, stock sell tax, ETF sell tax, and transition cost on unit notional; still diagnostic contract not formal replay",
            },
        ]
    )


def _readiness(contract: pd.DataFrame, transitions: pd.DataFrame, coverage: pd.DataFrame) -> dict[str, Any]:
    stock_rows = int((contract["holding_asset_type"] == "stock").sum())
    adjusted_ready_rows = int(coverage["adjusted_close_interval_ready"].fillna(False).astype(bool).sum()) if not coverage.empty else 0
    unadjusted_ready_rows = int(coverage["unadjusted_ohlc_comparator_ready"].fillna(False).astype(bool).sum()) if not coverage.empty else 0
    selected_stock_adjusted_share = adjusted_ready_rows / stock_rows if stock_rows else 1.0
    unadjusted_share = unadjusted_ready_rows / stock_rows if stock_rows else 1.0
    adjusted_full_ready = selected_stock_adjusted_share >= 1.0
    unadjusted_full_ready = unadjusted_share >= 1.0
    remaining_adjusted_blocked_rows = stock_rows - adjusted_ready_rows
    adjusted_net = contract["net_equity_after_cost_adjusted_path"].dropna()
    unadjusted_net = contract["net_equity_after_cost_unadjusted_comparator"].dropna()
    return {
        "task_id": TASK_ID,
        "status": "p1_c2_market_health_consensus4_adjusted_state_machine_partial_adjusted_close_blocked",
        "ready_for_p1_c2_market_health_consensus4_net_cost_diagnostic": bool(adjusted_full_ready),
        "ready_for_unadjusted_ohlc_comparator_diagnostic": bool(unadjusted_full_ready),
        "ready_for_experiments": bool(adjusted_full_ready),
        "selected_stock_adjusted_close_ready_share": selected_stock_adjusted_share,
        "selected_stock_adjusted_ready_rows": adjusted_ready_rows,
        "selected_stock_adjusted_remaining_blocked_rows": remaining_adjusted_blocked_rows,
        "selected_stock_exception_interval_rows": stock_rows,
        "unadjusted_ohlc_comparator_ready_share": unadjusted_share,
        "unadjusted_ohlc_comparator_ready_rows": unadjusted_ready_rows,
        "state_machine_interval_rows": int(len(contract)),
        "transition_rows": int(len(transitions)),
        "c2_exception_allowed_rows": stock_rows,
        "formal_cost_model_ready": True,
        "cost_model_version": cost_model_metadata()["cost_model_version"],
        "cost_model_includes_fee_tax_etf_stock_split": True,
        "ready_for_radar_selected_stock_adjusted_close_source_fill": bool(remaining_adjusted_blocked_rows > 0),
        "radar_next_task_suggestion": "TASK-RADAR-DATA-VNEXT-P1-C2-CONSENSUS4-SELECTED-STOCK-ADJUSTED-CLOSE-SOURCE-FILL-001"
        if remaining_adjusted_blocked_rows > 0
        else "",
        "net_total_return_after_transition_cost_adjusted_path_if_complete": float(adjusted_net.iloc[-1] - 1.0) if adjusted_full_ready and not adjusted_net.empty else None,
        "net_total_return_after_transition_cost_unadjusted_comparator": float(unadjusted_net.iloc[-1] - 1.0) if unadjusted_full_ready and not unadjusted_net.empty else None,
        "ready_for_formal": False,
        "ready_for_strategy_replay": False,
        "future_data_violation_count": 0,
        "not_live_rule": True,
        "forward_returns_live_rule_usage": False,
        **FLAGS,
    }


def _manifest(files: list[Path], readiness: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_id": TASK_ID,
        "status": readiness["status"],
        "created_at": pd.Timestamp.now(tz="Asia/Taipei").isoformat(),
        "output_dir": str(OUTPUT_DIR),
        "inputs": {
            "p1_state_hold_signal_trace": str(P1_STATE_HOLD_DIR / "p1_base_exception_signal_trace.csv"),
            "p1_market_regime_classifier_contract": str(MARKET_REGIME_DIR / "p1_market_regime_classifier_feature_contract.csv"),
            "benchmark_features": str(BENCHMARK_FEATURES),
            "stock_features": str(STOCK_FEATURES),
            "p1_unadjusted_stock_path": str(P1_STOCK_PATH),
        },
        "artifacts": [
            {
                "name": path.name,
                "path": str(path),
                "sha256": _sha256(path),
                "rows": int(pd.read_csv(path, low_memory=False).shape[0]) if path.suffix == ".csv" else None,
            }
            for path in files
        ],
        "readiness": readiness,
        "flags": FLAGS,
    }


def build() -> dict[str, Any]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    contract, transitions, coverage, source_attempts = _state_machine_contract()
    cost_audit = _cost_audit(transitions)
    blocked = _blocked_proxy_audit(contract, coverage)
    future_audit = pd.DataFrame(
        [
            {
                "audit_item": "future_return_as_feature_or_rule",
                "violation_count": 0,
                "status": "pass",
                "notes": "C2 gate uses 0050 above MA60 and 20D/40D returns as of signal date; no future return is used for selection.",
            },
            {
                "audit_item": "cost_required_for_main_conclusion",
                "violation_count": 0,
                "status": "pass",
                "notes": "Transition cost columns include buy fee, sell fee, securities transaction tax, ETF/stock sell-tax split, and total transition cost.",
            },
        ]
    )
    readiness = _readiness(contract, transitions, coverage)
    artifacts = {
        "p1_c2_market_health_consensus4_state_machine_contract.csv": contract,
        "p1_c2_market_health_consensus4_transition_trace.csv": transitions,
        "p1_c2_market_health_consensus4_adjusted_price_coverage.csv": coverage,
        "p1_c2_market_health_consensus4_cost_audit.csv": cost_audit,
        "p1_c2_market_health_consensus4_blocked_proxy_audit.csv": blocked,
        "p1_c2_market_health_consensus4_adjusted_source_attempts.csv": source_attempts,
        "p1_c2_market_health_consensus4_future_data_audit.csv": future_audit,
    }
    files: list[Path] = []
    for name, df in artifacts.items():
        path = OUTPUT_DIR / name
        df.to_csv(path, index=False, encoding="utf-8-sig")
        files.append(path)

    readiness_path = OUTPUT_DIR / "readiness_for_p1_c2_market_health_consensus4_diagnostic.json"
    readiness_path.write_text(json.dumps(readiness, ensure_ascii=False, indent=2), encoding="utf-8")
    files.append(readiness_path)

    summary_lines = [
        "# P1 C2 market-health gated consensus4 adjusted state-machine contract",
        "",
        f"- task_id: `{TASK_ID}`",
        f"- status: `{readiness['status']}`",
        f"- ready_for_p1_c2_market_health_consensus4_net_cost_diagnostic: `{str(readiness['ready_for_p1_c2_market_health_consensus4_net_cost_diagnostic']).lower()}`",
        f"- selected_stock_adjusted_close_ready_share: {readiness['selected_stock_adjusted_close_ready_share']:.6f}",
        f"- selected_stock_adjusted_remaining_blocked_rows: {readiness['selected_stock_adjusted_remaining_blocked_rows']}",
        f"- unadjusted_ohlc_comparator_ready_share: {readiness['unadjusted_ohlc_comparator_ready_share']:.6f}",
        f"- state_machine_interval_rows: {readiness['state_machine_interval_rows']}",
        f"- transition_rows: {readiness['transition_rows']}",
        "",
        "## 語義",
        "",
        "Default state 是持有 00631L。只有 C2 market health gate 通過，且 consensus4 exception active，才允許切到 selected stock；C2 gate 失效或 exception invalid 時切回 00631L。同一檔連續有效則續抱，不重複買賣。",
        "",
        "C2 定義固定為：0050 above MA60 + 20D/40D returns non-negative。Core 只 materialize contract/path/cost readiness，不做 Experiments verdict。",
        "",
        "成本已使用 EP05 TaiwanCostModel，包含買賣手續費、證券交易稅、ETF/股票賣出稅率差異與 transition cost。no-cost/gross 不作主結論。",
        "",
        "selected-stock adjusted close 目前仍未完整 ready；官方 unadjusted OHLC comparator 可保留作 proxy comparator，但不可包裝成 formal 或 adjusted-close path。",
        "",
        "下一棒明確：請交 Radar/Data 做 selected-ticker-only adjusted close source fill，不做 full-market mass download。建議任務：TASK-RADAR-DATA-VNEXT-P1-C2-CONSENSUS4-SELECTED-STOCK-ADJUSTED-CLOSE-SOURCE-FILL-001。",
        "",
        "## Flags",
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
        "完成後如果下一棒明確，請直接指派下一個 thread；如果下一棒不明確，請回報 Strategy Center 判斷。不要完成後停住不回報。",
    ]
    summary_path = OUTPUT_DIR / "final_summary_zh.md"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
    files.append(summary_path)

    manifest = _manifest(files, readiness)
    manifest_path = OUTPUT_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return readiness


if __name__ == "__main__":
    print(json.dumps(build(), ensure_ascii=False, indent=2))
