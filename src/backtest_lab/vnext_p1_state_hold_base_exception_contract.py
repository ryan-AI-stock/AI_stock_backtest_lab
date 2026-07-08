from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_ROOT = Path("C:/Users/zergv/Documents/Codex/2026-07-06/backtest-lab-experiments-diagnostic-validation-attribution")
P1_DEFENSIVE_DIR = EXPERIMENTS_ROOT / "outputs" / "vnext_p1_defensive_policy_benchmark_comparison_diagnostic_20260708"
BENCHMARK_FEATURES = REPO_ROOT / "outputs" / "vnext_dynamic_candidate_pool_data_materialization_20260706" / "benchmark_features.csv"
P1_STOCK_PATH = (
    REPO_ROOT
    / "outputs"
    / "vnext_p1_legacy_regime_unadjusted_path_refresh_20260708"
    / "p1_legacy_regime_unadjusted_trade_path_refreshed.csv"
)
OUTPUT_DIR = REPO_ROOT / "outputs" / "vnext_p1_state_hold_base_exception_path_contract_20260708"

TASK_ID = "TASK-BACKTEST-CORE-VNEXT-P1-STATE-HOLD-BASE-EXCEPTION-PATH-CONTRACT-001"
P1_START = pd.Timestamp("2015-01-02")
P1_END = pd.Timestamp("2022-12-29")
BASE_ASSETS = ["0050", "00631L"]
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


def _benchmark_frame(benchmark: str) -> pd.DataFrame:
    df = pd.read_csv(BENCHMARK_FEATURES, dtype={"benchmark": str}, low_memory=False)
    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
    df = df.loc[
        (df["benchmark"] == benchmark)
        & (df["trade_date"] >= P1_START)
        & (df["trade_date"] <= P1_END)
    ].copy()
    df = df.sort_values("trade_date")
    start_price = float(df["adjusted_close"].dropna().iloc[0])
    df["state_hold_path_type"] = "buy_hold_daily_state_hold_reference"
    df["daily_return"] = df["adjusted_close"].pct_change().fillna(0.0)
    df["cumulative_return"] = df["adjusted_close"] / start_price - 1.0
    df["running_peak_cumulative_value"] = (1.0 + df["cumulative_return"]).cummax()
    df["drawdown"] = (1.0 + df["cumulative_return"]) / df["running_peak_cumulative_value"] - 1.0
    df["mdd_to_date"] = df["drawdown"].cummin()
    df["source_quality"] = "benchmark_features_adjusted_close_exact_reference"
    df["cash_proxy"] = False
    df["diagnostic_only"] = True
    for key, value in FLAGS.items():
        df[key] = value
    return df[
        [
            "trade_date",
            "benchmark",
            "state_hold_path_type",
            "adjusted_close",
            "daily_return",
            "cumulative_return",
            "drawdown",
            "mdd_to_date",
            "source_quality",
            "cash_proxy",
            "diagnostic_only",
            *FLAGS.keys(),
        ]
    ]


def _policy_trace() -> pd.DataFrame:
    trace = pd.read_csv(P1_DEFENSIVE_DIR / "p1_defensive_policy_benchmark_policy_trace.csv", low_memory=False)
    trace = trace.loc[
        (trace["policy_id"] == "consensus4_else_00631L")
        & (trace["timing_variant"] == "next_day_close_entry_fixed_5td_exit")
    ].copy()
    trace["signal_date"] = pd.to_datetime(trace["signal_date"], errors="coerce")
    trace = trace.sort_values("signal_date")
    trace["ticker"] = trace["ticker"].map(_ticker_str)
    trace["recommendation"] = trace["recommendation"].map(_ticker_str)
    next_signal = trace["signal_date"].shift(-1)
    trace["next_signal_date"] = next_signal.fillna(P1_END)
    trace["state_hold_policy_note"] = "base asset is held continuously until stock exception; not signal-aligned all-base weekly rebuy"
    trace["cash_condition_status"] = "blocked_no_bear_cash_classifier"
    trace["diagnostic_only"] = True
    for key, value in FLAGS.items():
        trace[key] = value
    return trace


def _stock_exception_same_week_map() -> dict[tuple[str, str], dict[str, Any]]:
    stock = pd.read_csv(P1_STOCK_PATH, low_memory=False)
    stock = stock.loc[stock["timing_variant"] == "same_week_close_to_next_rebalance_close_comparator"].copy()
    stock["signal_date"] = pd.to_datetime(stock["signal_date"], errors="coerce")
    stock["ticker_norm"] = stock["ticker"].map(_ticker_str)
    stock["ready_sort"] = (~stock["price_path_ready"].fillna(False).astype(bool)).astype(int)
    stock = stock.sort_values(["signal_date", "ticker_norm", "ready_sort"])
    stock = stock.drop_duplicates(["signal_date", "ticker_norm"], keep="first")
    return {
        (pd.Timestamp(row.signal_date).date().isoformat(), row.ticker_norm): row._asdict()
        for row in stock.itertuples(index=False)
    }


def _benchmark_price_maps() -> dict[str, dict[str, float]]:
    maps = {}
    for benchmark in BASE_ASSETS:
        df = _benchmark_frame(benchmark)
        maps[benchmark] = {
            pd.Timestamp(row.trade_date).date().isoformat(): float(row.adjusted_close)
            for row in df.itertuples(index=False)
            if pd.notna(row.adjusted_close)
        }
    return maps


def _base_exception_contract(trace: pd.DataFrame) -> pd.DataFrame:
    stock_map = _stock_exception_same_week_map()
    bench_maps = _benchmark_price_maps()
    rows: list[dict[str, Any]] = []
    for base_asset in BASE_ASSETS:
        cumulative_value = 1.0
        running_peak = 1.0
        for r in trace.itertuples(index=False):
            signal_date = pd.Timestamp(r.signal_date).date().isoformat()
            exit_date = pd.Timestamp(r.next_signal_date).date().isoformat()
            is_stock = str(r.exposure_type) == "stock"
            selected_ticker = _ticker_str(r.ticker if is_stock else base_asset)
            base_entry_price = bench_maps[base_asset].get(signal_date)
            base_exit_price = bench_maps[base_asset].get(exit_date)
            base_return = (base_exit_price / base_entry_price - 1.0) if base_entry_price and base_exit_price else None
            selected_return = base_return
            selected_entry_price = base_entry_price
            selected_exit_price = base_exit_price
            selected_source_quality = "benchmark_features_adjusted_close_exact_reference"
            path_ready = base_return is not None
            blocked_reason = "" if path_ready else "missing_base_asset_entry_or_exit_adjusted_close"
            stock_row = None
            if is_stock:
                stock_row = stock_map.get((signal_date, selected_ticker))
                path_ready = bool(stock_row and stock_row.get("price_path_ready"))
                selected_return = stock_row.get("gross_return_unadjusted") if stock_row else None
                selected_entry_price = stock_row.get("entry_close") if stock_row else None
                selected_exit_price = stock_row.get("exit_close") if stock_row else None
                selected_source_quality = stock_row.get("source_quality") if stock_row else "missing_stock_exception_path"
                blocked_reason = "" if path_ready else "missing_or_blocked_stock_exception_same_week_path"
            if path_ready and selected_return is not None:
                cumulative_value *= 1.0 + float(selected_return)
                running_peak = max(running_peak, cumulative_value)
            drawdown = cumulative_value / running_peak - 1.0
            rows.append(
                {
                    "signal_date": signal_date,
                    "exit_date": exit_date,
                    "base_asset": base_asset,
                    "policy_id": "consensus4_else_base",
                    "exception_definition": "consensus4_ultra_strict_stock_exception",
                    "state_action": "stock_exception" if is_stock else "base_hold",
                    "selected_asset_type": "stock" if is_stock else "base_asset",
                    "selected_ticker": selected_ticker,
                    "original_consensus4_recommendation": r.recommendation,
                    "original_exposure_type": r.exposure_type,
                    "path_type": "signal_date_interval_state_hold_approximation",
                    "state_hold_semantics": "base is continuously held between exceptions; interval rows decompose path for Experiments",
                    "base_entry_adjusted_close": base_entry_price,
                    "base_exit_adjusted_close": base_exit_price,
                    "base_interval_return": base_return,
                    "selected_entry_price": selected_entry_price,
                    "selected_exit_price": selected_exit_price,
                    "selected_interval_return": selected_return,
                    "cumulative_return_before_cost_proxy": cumulative_value - 1.0,
                    "drawdown_before_cost_proxy": drawdown,
                    "path_ready": path_ready,
                    "blocked_reason": blocked_reason,
                    "source_quality": selected_source_quality,
                    "stock_exception_cost_field_available": bool(is_stock and stock_row is not None and pd.notna(stock_row.get("net_return_local_ep05_cost_unit_notional"))),
                    "stock_exception_net_return_local_ep05_cost_unit_notional": stock_row.get("net_return_local_ep05_cost_unit_notional") if stock_row else None,
                    "cash_condition_status": "blocked_no_bear_cash_classifier",
                    "diagnostic_only": True,
                    **FLAGS,
                }
            )
    return pd.DataFrame(rows)


def _coverage_audit(contract: pd.DataFrame, bench_paths: dict[str, pd.DataFrame], trace: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for benchmark, df in bench_paths.items():
        rows.append(
            {
                "artifact": f"p1_state_hold_benchmark_path_{benchmark}",
                "rows": len(df),
                "ready_rows": int(df["adjusted_close"].notna().sum()),
                "first_date": df["trade_date"].min().date().isoformat(),
                "last_date": df["trade_date"].max().date().isoformat(),
                "coverage_status": "ready",
            }
        )
    for base_asset, group in contract.groupby("base_asset"):
        rows.append(
            {
                "artifact": f"{base_asset}_base_plus_consensus4_stock_exception",
                "rows": len(group),
                "ready_rows": int(group["path_ready"].fillna(False).astype(bool).sum()),
                "first_date": group["signal_date"].min(),
                "last_date": group["signal_date"].max(),
                "coverage_status": "ready" if group["path_ready"].fillna(False).all() else "partial",
            }
        )
    rows.append(
        {
            "artifact": "p1_base_exception_signal_trace",
            "rows": len(trace),
            "ready_rows": int(trace["ready_for_metric"].fillna(False).astype(bool).sum()),
            "first_date": trace["signal_date"].min().date().isoformat(),
            "last_date": trace["signal_date"].max().date().isoformat(),
            "coverage_status": "ready",
        }
    )
    return pd.DataFrame(rows)


def _blocked_proxy_audit(contract: pd.DataFrame) -> pd.DataFrame:
    blocked_rows = int((~contract["path_ready"].fillna(False).astype(bool)).sum())
    return pd.DataFrame(
        [
            {
                "field_or_component": "cash_condition_classifier",
                "status": "blocked",
                "reason": "no accepted bear/cash classifier in this contract",
                "affected_rows": len(contract),
            },
            {
                "field_or_component": "state_hold_interval_contract",
                "status": "diagnostic_approximation",
                "reason": "interval rows decompose state-hold path for Experiments; benchmark daily buy-hold path is authoritative for base MDD",
                "affected_rows": len(contract),
            },
            {
                "field_or_component": "stock_exception_same_week_path",
                "status": "ready" if blocked_rows == 0 else "partial",
                "reason": "stock exception path is joined from selected-stock official unadjusted OHLC same-week comparator",
                "affected_rows": blocked_rows,
            },
            {
                "field_or_component": "adjusted_close_for_stock_exception",
                "status": "blocked",
                "reason": "stock exception adjusted close not materialized; unadjusted OHLC only",
                "affected_rows": len(contract.loc[contract["state_action"] == "stock_exception"]),
            },
        ]
    )


def _readiness(contract: pd.DataFrame, bench_paths: dict[str, pd.DataFrame]) -> dict[str, Any]:
    bench_ready = all(df["adjusted_close"].notna().all() and len(df) > 0 for df in bench_paths.values())
    contract_ready = bool(contract["path_ready"].fillna(False).all())
    return {
        "task_id": TASK_ID,
        "status": "p1_state_hold_base_exception_path_contract_ready_diagnostic_only",
        "ready_for_p1_base_exception_diagnostic": bool(bench_ready and contract_ready),
        "benchmark_daily_state_hold_paths_ready": bench_ready,
        "base_exception_interval_contract_ready": contract_ready,
        "p1_requested_start": P1_START.date().isoformat(),
        "p1_requested_end": P1_END.date().isoformat(),
        "p1_actual_0050_start": bench_paths["0050"]["trade_date"].min().date().isoformat(),
        "p1_actual_0050_end": bench_paths["0050"]["trade_date"].max().date().isoformat(),
        "p1_actual_00631L_start": bench_paths["00631L"]["trade_date"].min().date().isoformat(),
        "p1_actual_00631L_end": bench_paths["00631L"]["trade_date"].max().date().isoformat(),
        "contract_rows": int(len(contract)),
        "blocked_contract_rows": int((~contract["path_ready"].fillna(False).astype(bool)).sum()),
        "cash_condition_ready": False,
        "cash_condition_status": "blocked_no_bear_cash_classifier",
        "state_hold_approximation_used": True,
        "signal_aligned_proxy_used": False,
        "future_data_violation_count": 0,
        "ready_for_formal": False,
        "ready_for_strategy_replay": False,
        "not_live_rule": True,
        "forward_returns_live_rule_usage": False,
        "boundary_flags": FLAGS,
    }


def _summary(readiness: dict[str, Any], contract: pd.DataFrame, bench_paths: dict[str, pd.DataFrame]) -> str:
    metrics = []
    for benchmark, df in bench_paths.items():
        metrics.append(
            f"- {benchmark}: total_return={df['cumulative_return'].iloc[-1]:.6f}, MDD={df['drawdown'].min():.6f}, rows={len(df)}"
        )
    by_policy = contract.groupby("base_asset").tail(1)[["base_asset", "cumulative_return_before_cost_proxy", "drawdown_before_cost_proxy"]]
    return "\n".join(
        [
            "# P1 state-hold benchmark / base+exception path contract",
            "",
            f"- task_id: `{TASK_ID}`",
            f"- status: `{readiness['status']}`",
            f"- ready_for_p1_base_exception_diagnostic: `{str(readiness['ready_for_p1_base_exception_diagnostic']).lower()}`",
            f"- contract_rows: {readiness['contract_rows']}",
            f"- blocked_contract_rows: {readiness['blocked_contract_rows']}",
            "",
            "## Benchmark Daily State-Hold",
            "",
            *metrics,
            "",
            "## Base+Exception Interval Proxy",
            "",
            by_policy.to_csv(index=False),
            "",
            "## 語義",
            "",
            "`signal-aligned all 00631L` 不再被包裝成 live fallback/base rule。本包把 0050 / 00631L buy-hold daily state-hold path 與 signal-date interval contract 分開；base asset 是持續持有，只有 consensus4 stock exception 觸發時切到個股。",
            "",
            "cash condition / bear classifier 尚未 ready，本包只列 blocked，不杜撰 cash rule。",
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
            "p1_defensive_policy_trace": str(P1_DEFENSIVE_DIR / "p1_defensive_policy_benchmark_policy_trace.csv"),
            "benchmark_features": str(BENCHMARK_FEATURES),
            "p1_stock_path": str(P1_STOCK_PATH),
        },
        "readiness": readiness,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    bench_paths = {benchmark: _benchmark_frame(benchmark) for benchmark in BASE_ASSETS}
    trace = _policy_trace()
    contract = _base_exception_contract(trace)
    coverage = _coverage_audit(contract, bench_paths, trace)
    blocked = _blocked_proxy_audit(contract)
    readiness = _readiness(contract, bench_paths)
    outputs = {
        "p1_state_hold_base_exception_contract.csv": contract,
        "p1_state_hold_benchmark_path_0050.csv": bench_paths["0050"],
        "p1_state_hold_benchmark_path_00631L.csv": bench_paths["00631L"],
        "p1_base_exception_signal_trace.csv": trace,
        "p1_state_hold_coverage_audit.csv": coverage,
        "p1_state_hold_blocked_proxy_audit.csv": blocked,
    }
    written: list[Path] = []
    for name, df in outputs.items():
        path = OUTPUT_DIR / name
        df.to_csv(path, index=False, encoding="utf-8-sig")
        written.append(path)
    readiness_path = OUTPUT_DIR / "readiness_for_p1_state_hold_base_exception_path_contract.json"
    readiness_path.write_text(json.dumps(readiness, ensure_ascii=False, indent=2), encoding="utf-8")
    written.append(readiness_path)
    summary_path = OUTPUT_DIR / "final_summary_zh.md"
    summary_path.write_text(_summary(readiness, contract, bench_paths), encoding="utf-8")
    written.append(summary_path)
    manifest_path = OUTPUT_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(_manifest(written, readiness), ensure_ascii=False, indent=2), encoding="utf-8")
    written.append(manifest_path)
    print(json.dumps({"output_dir": str(OUTPUT_DIR), "readiness": readiness}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
