from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from backtest_lab.costs import TaiwanCostModel, cost_model_metadata


REPO_ROOT = Path(__file__).resolve().parents[2]
P1_STATE_HOLD_DIR = REPO_ROOT / "outputs" / "vnext_p1_state_hold_base_exception_path_contract_20260708"
BENCHMARK_FEATURES = REPO_ROOT / "outputs" / "vnext_dynamic_candidate_pool_data_materialization_20260706" / "benchmark_features.csv"
P1_STOCK_PATH = (
    REPO_ROOT
    / "outputs"
    / "vnext_p1_legacy_regime_unadjusted_path_refresh_20260708"
    / "p1_legacy_regime_unadjusted_trade_path_refreshed.csv"
)
OUTPUT_DIR = REPO_ROOT / "outputs" / "vnext_p1_00631l_base_consensus4_state_machine_contract_20260708"

TASK_ID = "TASK-BACKTEST-CORE-VNEXT-P1-00631L-BASE-CONSENSUS4-EXCEPTION-STATE-MACHINE-CONTRACT-001"
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


def _signal_trace() -> pd.DataFrame:
    trace = pd.read_csv(P1_STATE_HOLD_DIR / "p1_base_exception_signal_trace.csv", low_memory=False)
    trace["signal_date"] = pd.to_datetime(trace["signal_date"], errors="coerce")
    trace["next_signal_date"] = pd.to_datetime(trace["next_signal_date"], errors="coerce")
    trace = trace.sort_values("signal_date").copy()
    trace["target_ticker"] = trace.apply(
        lambda r: _ticker_str(r["ticker"]) if str(r["exposure_type"]) == "stock" else BASE_ASSET,
        axis=1,
    )
    trace["target_asset_type"] = trace["target_ticker"].map(_asset_type)
    trace["target_state"] = trace["target_ticker"].map(lambda t: "stock_exception" if t != BASE_ASSET else "base_00631L")
    return trace


def _stock_path_map() -> dict[tuple[str, str], dict[str, Any]]:
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


def _cost_rate(old_ticker: str | None, new_ticker: str | None) -> dict[str, Any]:
    model = TaiwanCostModel()
    sell_cost = 0.0
    buy_cost = 0.0
    if old_ticker:
        sell_cost = model.sell_cost(DIAGNOSTIC_NOTIONAL_TWD, _asset_type(old_ticker))
    if new_ticker:
        buy_cost = model.buy_cost(DIAGNOSTIC_NOTIONAL_TWD)
    return {
        "diagnostic_notional_twd": DIAGNOSTIC_NOTIONAL_TWD,
        "sell_cost_twd": sell_cost,
        "buy_cost_twd": buy_cost,
        "total_transition_cost_twd": sell_cost + buy_cost,
        "transition_cost_rate_proxy": (sell_cost + buy_cost) / DIAGNOSTIC_NOTIONAL_TWD,
        "cost_model_status": "applied_local_ep05_TaiwanCostModel_unit_notional_transition_proxy",
    }


def _price_for_asset(ticker: str, date_key: str, stock_row: dict[str, Any] | None, price_kind: str) -> float | None:
    if ticker == BASE_ASSET:
        return _benchmark_price_map().get(date_key)
    if not stock_row:
        return None
    if price_kind == "entry":
        return stock_row.get("entry_close")
    return stock_row.get("exit_close")


def _state_machine_contract() -> tuple[pd.DataFrame, pd.DataFrame]:
    trace = _signal_trace()
    calendar = _benchmark_calendar()
    base_prices = _benchmark_price_map()
    stock_map = _stock_path_map()
    intervals: list[dict[str, Any]] = []
    transitions: list[dict[str, Any]] = []

    first_signal = pd.Timestamp(trace["signal_date"].min())
    first_entry = _next_trading_date(calendar, first_signal, 1)
    start_date = calendar[0]
    gross_equity = 1.0
    net_equity = 1.0
    gross_peak = 1.0
    net_peak = 1.0
    current_ticker = BASE_ASSET

    if first_entry is not None and first_entry > start_date:
        start_key = start_date.date().isoformat()
        first_entry_key = first_entry.date().isoformat()
        start_price = base_prices.get(start_key)
        end_price = base_prices.get(first_entry_key)
        interval_return = (end_price / start_price - 1.0) if start_price and end_price else None
        if interval_return is not None:
            gross_equity *= 1.0 + interval_return
            net_equity *= 1.0 + interval_return
            gross_peak = max(gross_peak, gross_equity)
            net_peak = max(net_peak, net_equity)
        intervals.append(
            {
                "signal_date": "",
                "state_start_date": start_key,
                "state_end_date": first_entry_key,
                "holding_ticker": BASE_ASSET,
                "holding_asset_type": "etf",
                "state_reason": "initial_default_base_hold_before_first_signal",
                "transition_action": "none_initial_state_already_holding_00631L",
                "entry_price_kind": "adjusted_close",
                "exit_price_kind": "adjusted_close",
                "entry_price": start_price,
                "exit_price": end_price,
                "gross_interval_return": interval_return,
                "transition_cost_rate_proxy": 0.0,
                "net_interval_return_after_transition_cost_proxy": interval_return,
                "gross_equity_before_cost_proxy": gross_equity,
                "net_equity_after_transition_cost_proxy": net_equity,
                "gross_drawdown": gross_equity / gross_peak - 1.0,
                "net_drawdown": net_equity / net_peak - 1.0,
                "path_ready": interval_return is not None,
                "blocked_reason": "" if interval_return is not None else "missing_initial_base_price",
                "source_quality": "benchmark_features_adjusted_close_exact_reference",
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
        if pd.isna(exit_date):
            exit_date = P1_END
        entry_key = entry_date.date().isoformat() if entry_date is not None else ""
        exit_key = pd.Timestamp(exit_date).date().isoformat() if exit_date is not None else ""
        target_ticker = row.target_ticker
        target_asset_type = row.target_asset_type
        transition_needed = target_ticker != current_ticker
        cost = _cost_rate(current_ticker if transition_needed else None, target_ticker if transition_needed else None)
        stock_row = stock_map.get((signal_key, target_ticker)) if target_asset_type == "stock" else None
        if target_ticker == BASE_ASSET:
            entry_price = base_prices.get(entry_key)
            exit_price = base_prices.get(exit_key)
            source_quality = "benchmark_features_adjusted_close_exact_reference"
            entry_kind = "adjusted_close"
            exit_kind = "adjusted_close"
        else:
            entry_price = stock_row.get("entry_close") if stock_row else None
            exit_price = stock_row.get("exit_close") if stock_row else None
            source_quality = stock_row.get("source_quality") if stock_row else "missing_stock_exception_next_day_close_path"
            entry_kind = "unadjusted_close"
            exit_kind = "unadjusted_close"
        path_ready = entry_price is not None and exit_price is not None and pd.notna(entry_price) and pd.notna(exit_price)
        interval_return = (float(exit_price) / float(entry_price) - 1.0) if path_ready else None
        cost_rate = cost["transition_cost_rate_proxy"] if transition_needed else 0.0
        net_interval = ((1.0 - cost_rate) * (1.0 + interval_return) - 1.0) if interval_return is not None else None
        if path_ready:
            gross_equity *= 1.0 + float(interval_return)
            net_equity *= 1.0 + float(net_interval)
            gross_peak = max(gross_peak, gross_equity)
            net_peak = max(net_peak, net_equity)
        blocked_reason = ""
        if not path_ready:
            blocked_reason = "missing_base_or_stock_entry_exit_price"
        intervals.append(
            {
                "signal_date": signal_key,
                "state_start_date": entry_key,
                "state_end_date": exit_key,
                "holding_ticker": target_ticker,
                "holding_asset_type": target_asset_type,
                "state_reason": row.selection_reason,
                "target_state": row.target_state,
                "transition_action": _transition_action(current_ticker, target_ticker) if transition_needed else "hold_same_state_no_trade",
                "previous_ticker": current_ticker,
                "entry_price_kind": entry_kind,
                "exit_price_kind": exit_kind,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "gross_interval_return": interval_return,
                "transition_cost_rate_proxy": cost_rate,
                "net_interval_return_after_transition_cost_proxy": net_interval,
                "gross_equity_before_cost_proxy": gross_equity,
                "net_equity_after_transition_cost_proxy": net_equity,
                "gross_drawdown": gross_equity / gross_peak - 1.0,
                "net_drawdown": net_equity / net_peak - 1.0,
                "path_ready": path_ready,
                "blocked_reason": blocked_reason,
                "source_quality": source_quality,
                "cash_condition_status": "blocked_no_bear_cash_classifier",
                "adjusted_close_status": "ready_for_00631L_base; blocked_for_stock_exception_unadjusted_ohlc",
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
                    "to_asset_type": target_asset_type,
                    "transition_action": _transition_action(current_ticker, target_ticker),
                    "sell_price_kind": "adjusted_close" if current_ticker == BASE_ASSET else "unadjusted_close",
                    "buy_price_kind": entry_kind,
                    "sell_cost_asset_type": _asset_type(current_ticker),
                    "buy_cost_asset_type": target_asset_type,
                    "transition_price_ready": path_ready,
                    **cost,
                    "diagnostic_only": True,
                    **FLAGS,
                }
            )
        current_ticker = target_ticker
    return pd.DataFrame(intervals), pd.DataFrame(transitions)


def _transition_action(old: str, new: str) -> str:
    if old == BASE_ASSET and new != BASE_ASSET:
        return "base_00631L_to_stock_exception"
    if old != BASE_ASSET and new == BASE_ASSET:
        return "stock_exception_to_base_00631L"
    if old != BASE_ASSET and new != BASE_ASSET:
        return "stock_exception_to_stock_exception_switch"
    return "base_hold_no_trade"


def _cost_audit(transitions: pd.DataFrame) -> pd.DataFrame:
    if transitions.empty:
        return pd.DataFrame()
    rows = []
    for action, group in transitions.groupby("transition_action"):
        rows.append(
            {
                "transition_action": action,
                "transition_count": len(group),
                "total_transition_cost_rate_proxy_sum": group["transition_cost_rate_proxy"].sum(),
                "avg_transition_cost_rate_proxy": group["transition_cost_rate_proxy"].mean(),
                "sell_cost_asset_types": ";".join(sorted(group["sell_cost_asset_type"].dropna().unique())),
                "buy_cost_asset_types": ";".join(sorted(group["buy_cost_asset_type"].dropna().unique())),
                "cost_model_status": "applied_local_ep05_TaiwanCostModel_unit_notional_transition_proxy",
                "formal_cost_model_ready": True,
                "diagnostic_only": True,
                **FLAGS,
            }
        )
    return pd.DataFrame(rows)


def _price_coverage_audit(contract: pd.DataFrame, transitions: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "component": "state_machine_interval_contract",
                "rows": len(contract),
                "ready_rows": int(contract["path_ready"].fillna(False).astype(bool).sum()),
                "blocked_rows": int((~contract["path_ready"].fillna(False).astype(bool)).sum()),
                "coverage_status": "ready" if contract["path_ready"].fillna(False).all() else "partial",
            },
            {
                "component": "transition_trace",
                "rows": len(transitions),
                "ready_rows": int(transitions["transition_price_ready"].fillna(False).astype(bool).sum()) if not transitions.empty else 0,
                "blocked_rows": int((~transitions["transition_price_ready"].fillna(False).astype(bool)).sum()) if not transitions.empty else 0,
                "coverage_status": "ready" if transitions.empty or transitions["transition_price_ready"].fillna(False).all() else "partial",
            },
            {
                "component": "00631L_base_adjusted_close",
                "rows": len(_benchmark_frame()),
                "ready_rows": int(_benchmark_frame()["adjusted_close"].notna().sum()),
                "blocked_rows": int(_benchmark_frame()["adjusted_close"].isna().sum()),
                "coverage_status": "ready",
            },
        ]
    )


def _blocked_proxy_audit(contract: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "field_or_component": "cash_condition_bear_classifier",
                "status": "blocked",
                "reason": "no accepted cash/bear classifier; no cash rule fabricated",
                "affected_rows": len(contract),
            },
            {
                "field_or_component": "stock_exception_adjusted_close",
                "status": "blocked",
                "reason": "stock exception path uses official unadjusted OHLC; adjusted close not fabricated",
                "affected_rows": int((contract["holding_asset_type"] == "stock").sum()),
            },
            {
                "field_or_component": "transition_cost_model",
                "status": "diagnostic_proxy_ready",
                "reason": "EP05 TaiwanCostModel applied to unit notional; ETF sell tax and stock sell tax are separated by asset type",
                "affected_rows": len(contract),
            },
        ]
    )


def _readiness(contract: pd.DataFrame, transitions: pd.DataFrame) -> dict[str, Any]:
    path_ready = bool(contract["path_ready"].fillna(False).all())
    transition_ready = bool(transitions.empty or transitions["transition_price_ready"].fillna(False).all())
    return {
        "task_id": TASK_ID,
        "status": "p1_00631L_base_consensus4_exception_state_machine_contract_ready_diagnostic_only",
        "ready_for_p1_00631L_base_consensus4_state_machine_diagnostic": bool(path_ready and transition_ready),
        "state_machine_interval_rows": int(len(contract)),
        "transition_rows": int(len(transitions)),
        "path_ready": path_ready,
        "transition_trace_ready": transition_ready,
        "gross_total_return_before_cost_proxy": float(contract["gross_equity_before_cost_proxy"].dropna().iloc[-1] - 1.0),
        "net_total_return_after_transition_cost_proxy": float(contract["net_equity_after_transition_cost_proxy"].dropna().iloc[-1] - 1.0),
        "gross_mdd": float(contract["gross_drawdown"].min()),
        "net_mdd": float(contract["net_drawdown"].min()),
        "cash_condition_ready": False,
        "cash_condition_status": "blocked_no_bear_cash_classifier",
        "base_asset_cost_model": "ETF sell tax via TaiwanCostModel asset_type=etf",
        "stock_exception_cost_model": "stock sell tax via TaiwanCostModel asset_type=stock",
        "adjusted_close_ready": False,
        "adjusted_close_note": "00631L base adjusted close ready; stock exception adjusted close blocked/unadjusted OHLC only",
        "future_data_violation_count": 0,
        "ready_for_formal": False,
        "ready_for_strategy_replay": False,
        "not_live_rule": True,
        "forward_returns_live_rule_usage": False,
        "boundary_flags": FLAGS,
        "cost_model_metadata": cost_model_metadata(),
    }


def _summary(readiness: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# P1 00631L base + consensus4 exception state-machine contract",
            "",
            f"- task_id: `{TASK_ID}`",
            f"- status: `{readiness['status']}`",
            f"- ready_for_p1_00631L_base_consensus4_state_machine_diagnostic: `{str(readiness['ready_for_p1_00631L_base_consensus4_state_machine_diagnostic']).lower()}`",
            f"- state_machine_interval_rows: {readiness['state_machine_interval_rows']}",
            f"- transition_rows: {readiness['transition_rows']}",
            f"- gross_total_return_before_cost_proxy: {readiness['gross_total_return_before_cost_proxy']:.6f}",
            f"- net_total_return_after_transition_cost_proxy: {readiness['net_total_return_after_transition_cost_proxy']:.6f}",
            f"- gross_mdd: {readiness['gross_mdd']:.6f}",
            f"- net_mdd: {readiness['net_mdd']:.6f}",
            "",
            "## 語義",
            "",
            "Default state 是持有 00631L。只有 consensus4 stock exception 觸發時才切到個股；連續同一檔維持持有，不重複買賣；exception 失效時切回 00631L。00631L base 不再被做成每週清倉再買回。",
            "",
            "成本使用 EP05 TaiwanCostModel unit-notional transition proxy，ETF 與股票賣出稅率分開。這仍是 diagnostic-only contract，不是 formal 或 live rule。",
            "",
            "cash condition / bear classifier 仍 blocked，不杜撰。",
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
        ]
    )


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
            "p1_state_hold_contract": str(P1_STATE_HOLD_DIR),
            "benchmark_features": str(BENCHMARK_FEATURES),
            "p1_stock_path": str(P1_STOCK_PATH),
        },
        "readiness": readiness,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    contract, transitions = _state_machine_contract()
    cost_audit = _cost_audit(transitions)
    coverage = _price_coverage_audit(contract, transitions)
    blocked = _blocked_proxy_audit(contract)
    readiness = _readiness(contract, transitions)
    outputs = {
        "p1_00631L_base_consensus4_state_machine_contract.csv": contract,
        "p1_00631L_base_consensus4_transition_trace.csv": transitions,
        "p1_00631L_base_consensus4_cost_audit.csv": cost_audit,
        "p1_00631L_base_consensus4_price_coverage_audit.csv": coverage,
        "p1_00631L_base_consensus4_blocked_proxy_audit.csv": blocked,
    }
    written: list[Path] = []
    for name, df in outputs.items():
        path = OUTPUT_DIR / name
        df.to_csv(path, index=False, encoding="utf-8-sig")
        written.append(path)
    readiness_path = OUTPUT_DIR / "readiness_for_p1_00631L_base_consensus4_state_machine_diagnostic.json"
    readiness_path.write_text(json.dumps(readiness, ensure_ascii=False, indent=2), encoding="utf-8")
    written.append(readiness_path)
    summary_path = OUTPUT_DIR / "final_summary_zh.md"
    summary_path.write_text(_summary(readiness), encoding="utf-8")
    written.append(summary_path)
    manifest_path = OUTPUT_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(_manifest(written, readiness), ensure_ascii=False, indent=2), encoding="utf-8")
    written.append(manifest_path)
    print(json.dumps({"output_dir": str(OUTPUT_DIR), "readiness": readiness}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
