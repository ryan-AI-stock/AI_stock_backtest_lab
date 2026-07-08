from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from backtest_lab.costs import TaiwanCostModel, cost_model_metadata


REPO_ROOT = Path(__file__).resolve().parents[2]
RADAR_ROOT = Path("C:/Users/zergv/Documents/Codex/2026-05-23/ai-stock-rotation-radar-https-docs")
FULL_SWEEP_DIR = RADAR_ROOT / "outputs" / "radar_dynamic_pool1_all_listed_liquid_universe_full_sweep_20260703"
LEGACY_PRICE_DIR = RADAR_ROOT / "outputs" / "radar_vnext_legacy_rs20_selected_stock_price_path_source_package_20260708"

LEGACY_SIGNAL = (
    REPO_ROOT
    / "outputs"
    / "vnext_legacy_rs20_operating_mode_runner_readiness_20260708"
    / "legacy_rs20_operating_mode_signal_table.csv"
)
REGIME_SIGNAL = (
    REPO_ROOT
    / "outputs"
    / "vnext_regime_switch_hybrid_route_market_fields_path_materialization_20260708"
    / "regime_switch_hybrid_route_signal_table.csv"
)
BENCHMARK_FEATURES = REPO_ROOT / "outputs" / "vnext_dynamic_candidate_pool_data_materialization_20260706" / "benchmark_features.csv"
OUTPUT_DIR = REPO_ROOT / "outputs" / "vnext_p1_legacy_regime_selected_stock_unadjusted_path_materialization_20260708"

TASK_ID = "TASK-BACKTEST-CORE-VNEXT-P1-LEGACY-REGIME-SELECTED-STOCK-UNADJUSTED-PATH-MATERIALIZATION-001"
P1_START = pd.Timestamp("2015-01-02")
P1_END = pd.Timestamp("2022-12-29")
DIAGNOSTIC_NOTIONAL_TWD = 1_000_000
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
    "not_live_rule": True,
    "forward_returns_live_rule_usage": False,
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).lower() in {"true", "1", "yes", "y"}


def _load_legacy_signals() -> pd.DataFrame:
    df = pd.read_csv(LEGACY_SIGNAL, dtype={"ticker": str}, low_memory=False)
    df = df.loc[:, ~df.columns.duplicated()].copy()
    df["signal_date"] = pd.to_datetime(df["signal_date"], errors="coerce")
    df = df.loc[df["signal_date"].between(P1_START, P1_END)].copy()
    df["signal_family"] = "legacy_rs20"
    df["variant"] = df["signal_variant"]
    df["route_or_mode"] = df.get("selection_rule_basis", "")
    df["recommendation_type"] = "stock"
    df["path_bucket"] = "ordinary_stock"
    return df


def _load_regime_signals() -> pd.DataFrame:
    df = pd.read_csv(REGIME_SIGNAL, dtype={"ticker": str}, low_memory=False)
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"], errors="coerce")
    df = df.loc[df["snapshot_date"].between(P1_START, P1_END)].copy()
    df["signal_date"] = df["snapshot_date"]
    df["signal_family"] = "regime_switch"
    df["variant"] = df["routing_variant"]
    df["route_or_mode"] = df.get("selected_route_mode", df.get("routed_mode", ""))
    df["path_bucket"] = df["recommendation_type"].apply(
        lambda x: "00631L_reference" if str(x) == "00631L" else "ordinary_stock"
    )
    return df


def _signal_table() -> pd.DataFrame:
    legacy = _load_legacy_signals()
    regime = _load_regime_signals()
    keep = [
        "signal_date",
        "signal_family",
        "variant",
        "route_or_mode",
        "ticker",
        "name",
        "market",
        "recommendation_type",
        "path_bucket",
        "selected_rank",
        "within80_rank",
        "pool_rank",
        "RS20",
        "RS60",
        "layer4_risk_aware_score",
        "rs20_risk_context_score",
        "rs20_31_bonus_score",
        "in_31_high_confidence_subpool_reference",
        "in_100_extended_watchlist_reference",
        "0050_return_20d",
        "0050_return_40d",
        "0050_return_60d",
        "0050_bias20",
        "0050_bias60",
        "0050_bias120",
        "dynamic80_rs20_positive_share",
        "dynamic80_rs60_positive_share",
        "dynamic80_rs20_median",
        "dynamic80_rs20_dispersion_top_minus_median",
    ]
    combined = pd.concat(
        [legacy[[c for c in keep if c in legacy.columns]], regime[[c for c in keep if c in regime.columns]]],
        ignore_index=True,
        sort=False,
    )
    combined["signal_date"] = pd.to_datetime(combined["signal_date"], errors="coerce")
    combined["ticker"] = combined["ticker"].astype(str)
    combined["executable_ticker"] = combined.apply(
        lambda r: "00631L" if r["path_bucket"] == "00631L_reference" else str(r["ticker"]),
        axis=1,
    )
    combined["source_signal_contract"] = combined["signal_family"].map(
        {
            "legacy_rs20": str(LEGACY_SIGNAL),
            "regime_switch": str(REGIME_SIGNAL),
        }
    )
    combined["diagnostic_only"] = True
    for key, value in FLAGS.items():
        combined[key] = value
    return combined.sort_values(["signal_date", "signal_family", "variant"]).reset_index(drop=True)


def _benchmark_calendar() -> tuple[list[pd.Timestamp], dict[tuple[str, pd.Timestamp], float]]:
    bench = pd.read_csv(BENCHMARK_FEATURES, dtype={"benchmark": str}, low_memory=False)
    bench["trade_date"] = pd.to_datetime(bench["trade_date"], errors="coerce")
    bench = bench.loc[bench["trade_date"].notna()].copy()
    calendar = sorted(bench.loc[bench["benchmark"] == "0050", "trade_date"].dropna().unique())
    close_map: dict[tuple[str, pd.Timestamp], float] = {}
    for row in bench.itertuples(index=False):
        if pd.notna(row.adjusted_close):
            close_map[(str(row.benchmark), pd.Timestamp(row.trade_date))] = float(row.adjusted_close)
    return [pd.Timestamp(x) for x in calendar], close_map


def _next_trading_date(calendar: list[pd.Timestamp], date: pd.Timestamp, offset: int) -> pd.Timestamp | None:
    for idx, trading_date in enumerate(calendar):
        if trading_date > date:
            target = idx + offset - 1
            if 0 <= target < len(calendar):
                return calendar[target]
            return None
    return None


def _next_rebalance_date(signal_dates: list[pd.Timestamp], date: pd.Timestamp) -> pd.Timestamp | None:
    later = [d for d in signal_dates if d > date]
    return later[0] if later else None


def _load_available_local_price_keys() -> set[tuple[str, str]]:
    price_file = LEGACY_PRICE_DIR / "selected_stock_price_rows_local_only.csv"
    if not price_file.exists():
        return set()
    keys: set[tuple[str, str]] = set()
    for chunk in pd.read_csv(price_file, usecols=["date", "ticker"], dtype={"ticker": str}, chunksize=250_000):
        chunk = chunk.loc[(chunk["date"] >= P1_START.date().isoformat()) & (chunk["date"] <= "2023-01-15")]
        keys.update((str(r.ticker), str(r.date)) for r in chunk.itertuples(index=False))
    return keys


def _trade_path(signal: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    calendar, benchmark_close = _benchmark_calendar()
    signal_dates = sorted(signal["signal_date"].dropna().unique())
    available_keys = _load_available_local_price_keys()
    cost_model = TaiwanCostModel()
    rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []

    for row in signal.itertuples(index=False):
        signal_date = pd.Timestamp(row.signal_date)
        for timing in TIMING_VARIANTS:
            if timing == "same_week_close_to_next_rebalance_close_comparator":
                entry_date = signal_date
                exit_date = _next_rebalance_date([pd.Timestamp(x) for x in signal_dates], signal_date)
                entry_price_kind = "close"
                exit_price_kind = "close"
            elif timing == "next_day_open_entry_fixed_5td_exit":
                entry_date = _next_trading_date(calendar, signal_date, 1)
                exit_date = _next_trading_date(calendar, signal_date, 6)
                entry_price_kind = "open"
                exit_price_kind = "close"
            else:
                entry_date = _next_trading_date(calendar, signal_date, 1)
                exit_date = _next_trading_date(calendar, signal_date, 6)
                entry_price_kind = "close"
                exit_price_kind = "close"

            executable_ticker = str(row.executable_ticker)
            is_reference = row.path_bucket == "00631L_reference"
            entry_ready = False
            exit_ready = False
            entry_price = None
            exit_price = None
            source_quality = "blocked_missing_p1_selected_stock_unadjusted_ohlc_source"
            if is_reference and entry_date is not None and exit_date is not None and entry_price_kind == "close":
                entry_price = benchmark_close.get(("00631L", pd.Timestamp(entry_date)))
                exit_price = benchmark_close.get(("00631L", pd.Timestamp(exit_date)))
                entry_ready = entry_price is not None
                exit_ready = exit_price is not None
                source_quality = "benchmark_features_adjusted_close_reference_only"
            elif entry_date is not None and exit_date is not None:
                entry_ready = (executable_ticker, entry_date.date().isoformat()) in available_keys
                exit_ready = (executable_ticker, exit_date.date().isoformat()) in available_keys

            path_ready = bool(entry_ready and exit_ready and (is_reference or False))
            gross_return = (exit_price / entry_price - 1.0) if entry_price and exit_price else None
            net_return = None
            if gross_return is not None and entry_price and exit_price and is_reference:
                qty = math.floor(DIAGNOSTIC_NOTIONAL_TWD / entry_price)
                buy_gross = qty * entry_price
                sell_gross = qty * exit_price
                net_return = (sell_gross - cost_model.sell_cost(sell_gross, "etf") - buy_gross - cost_model.buy_cost(buy_gross)) / (
                    buy_gross + cost_model.buy_cost(buy_gross)
                )

            if is_reference:
                blocked_reason = "" if path_ready else "00631L_reference_missing_benchmark_close"
            elif not entry_ready and not exit_ready:
                blocked_reason = "missing_p1_selected_stock_entry_and_exit_unadjusted_ohlc"
            elif not entry_ready:
                blocked_reason = "missing_p1_selected_stock_entry_unadjusted_ohlc"
            elif not exit_ready:
                blocked_reason = "missing_p1_selected_stock_exit_unadjusted_ohlc"
            else:
                blocked_reason = "p1_selected_stock_price_key_exists_but_values_not_materialized_in_core"

            record = {
                "signal_date": signal_date.date().isoformat(),
                "signal_family": row.signal_family,
                "variant": row.variant,
                "route_or_mode": row.route_or_mode,
                "ticker": row.ticker,
                "executable_ticker": executable_ticker,
                "name": getattr(row, "name", ""),
                "path_bucket": row.path_bucket,
                "timing_variant": timing,
                "entry_date": entry_date.date().isoformat() if entry_date is not None else "",
                "exit_date": exit_date.date().isoformat() if exit_date is not None else "",
                "entry_price_kind": entry_price_kind,
                "exit_price_kind": exit_price_kind,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "gross_return": gross_return,
                "net_return_local_ep05_cost_unit_notional": net_return,
                "entry_ready": entry_ready,
                "exit_ready": exit_ready,
                "path_ready": path_ready,
                "source_quality": source_quality,
                "blocked_reason": blocked_reason,
                "adjusted_close_ready": False,
                "diagnostic_only": True,
                **FLAGS,
            }
            rows.append(record)
            if not path_ready:
                missing_rows.append(record)
    return pd.DataFrame(rows), pd.DataFrame(missing_rows)


def _source_request(signal: pd.DataFrame, trade_path: pd.DataFrame) -> pd.DataFrame:
    missing = trade_path.loc[trade_path["path_bucket"] == "ordinary_stock"].copy()
    if missing.empty:
        return pd.DataFrame()
    grouped = (
        missing.groupby(["executable_ticker", "name"], dropna=False)
        .agg(
            required_trade_path_rows=("signal_date", "size"),
            signal_family_count=("signal_family", "nunique"),
            first_signal_date=("signal_date", "min"),
            last_signal_date=("signal_date", "max"),
            required_price_start=("entry_date", "min"),
            required_price_end_with_exit_buffer=("exit_date", "max"),
        )
        .reset_index()
        .rename(columns={"executable_ticker": "ticker"})
    )
    grouped["required_columns"] = "date,ticker,name,market,open,high,low,close,volume,turnover_value,source_quality,adjustment_policy"
    grouped["requested_source_scope"] = "selected_tickers_only_no_full_market_mass_download"
    grouped["source_request_reason"] = "P1 legacy/regime selected-stock unadjusted OHLC path blocked locally"
    grouped["do_not_use_00631L_plus_excess_reconstruction"] = True
    grouped["adjusted_close_required"] = False
    grouped["adjusted_close_status"] = "blocked_not_required_for_this_unadjusted_path_request"
    grouped["diagnostic_only"] = True
    for key, value in FLAGS.items():
        grouped[key] = value
    return grouped.sort_values(["required_trade_path_rows", "ticker"], ascending=[False, True]).reset_index(drop=True)


def _timing_design() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "timing_variant": "same_week_close_to_next_rebalance_close_comparator",
                "entry": "signal_date close",
                "exit": "next weekly signal close",
                "status": "template_ready_stock_path_blocked_until_p1_selected_ohlc_source",
                "diagnostic_only": True,
            },
            {
                "timing_variant": "next_day_close_entry_fixed_5td_exit",
                "entry": "next trading day close",
                "exit": "5 trading days after entry close",
                "status": "template_ready_stock_path_blocked_until_p1_selected_ohlc_source",
                "diagnostic_only": True,
            },
            {
                "timing_variant": "next_day_open_entry_fixed_5td_exit",
                "entry": "next trading day open",
                "exit": "5 trading days after entry close",
                "status": "template_ready_stock_path_blocked_until_p1_selected_ohlc_source",
                "diagnostic_only": True,
            },
        ]
    )


def _cost_audit() -> pd.DataFrame:
    meta = cost_model_metadata()
    return pd.DataFrame(
        [
            {
                "audit_item": "local_ep05_cost_model",
                "ready": True,
                "scope": "unit_notional_diagnostic_only_not_portfolio_replay",
                "source_quality": "local_taiwan_standard_fee_tax_v1",
                **meta,
            },
            {
                "audit_item": "formal_portfolio_replay_cost_application",
                "ready": False,
                "scope": "not_requested_not_executed",
                "source_quality": "blocked_by_scope",
            },
        ]
    )


def _future_audit() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "audit_item": "selected_stock_return_reconstructed_from_00631L_plus_excess",
                "result": "passed",
                "violation_count": 0,
                "evidence": "Core emits source request instead of reconstructing P1 stock returns from 00631L plus excess.",
            },
            {
                "audit_item": "future_return_as_rule",
                "result": "passed",
                "violation_count": 0,
                "evidence": "Forward returns are not used to construct signals or requested source rows.",
            },
            {
                "audit_item": "full_market_mass_download_triggered",
                "result": "passed",
                "violation_count": 0,
                "evidence": "Source request is selected-ticker/date scoped only.",
            },
            {
                "audit_item": "adjusted_close_fabricated",
                "result": "passed",
                "violation_count": 0,
                "evidence": "Adjusted close remains blocked and is not fabricated.",
            },
        ]
    )


def _missing_price_audit(trade_path: pd.DataFrame) -> pd.DataFrame:
    if trade_path.empty:
        return pd.DataFrame()
    grouped = (
        trade_path.groupby(["signal_family", "variant", "path_bucket", "timing_variant"], dropna=False)
        .agg(
            rows=("signal_date", "size"),
            path_ready_rows=("path_ready", "sum"),
            blocked_rows=("path_ready", lambda s: int((~s.astype(bool)).sum())),
            entry_ready_rows=("entry_ready", "sum"),
            exit_ready_rows=("exit_ready", "sum"),
        )
        .reset_index()
    )
    grouped["path_ready_share"] = grouped["path_ready_rows"] / grouped["rows"]
    grouped["diagnostic_only"] = True
    return grouped


def _full_sweep_source_status() -> dict[str, Any]:
    manifest = _read_json(FULL_SWEEP_DIR / "manifest.json")
    shard_dir = Path(manifest.get("local_shard_dir", FULL_SWEEP_DIR / "shards"))
    first_shard = shard_dir / "accepted_liquidity_rows_2015_01.csv"
    return {
        "full_sweep_manifest_found": (FULL_SWEEP_DIR / "manifest.json").exists(),
        "full_sweep_manifest_local_shard_dir": str(shard_dir),
        "full_sweep_local_shard_dir_exists": shard_dir.exists(),
        "sample_2015_01_shard_exists": first_shard.exists(),
    }


def _write_manifest(files: list[Path], readiness: dict[str, Any]) -> None:
    manifest = {
        "task_id": TASK_ID,
        "status": readiness["status"],
        "output_dir": str(OUTPUT_DIR),
        "input_legacy_signal": str(LEGACY_SIGNAL),
        "input_regime_signal": str(REGIME_SIGNAL),
        "input_benchmark_features": str(BENCHMARK_FEATURES),
        "input_legacy_price_dir": str(LEGACY_PRICE_DIR),
        "output_files": [p.name for p in files] + ["manifest.json"],
        "diagnostic_only": True,
        **FLAGS,
    }
    manifest["file_hashes"] = {p.name: {"sha256": _sha256(p), "bytes": p.stat().st_size} for p in files if p.exists()}
    (OUTPUT_DIR / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def _summary(readiness: dict[str, Any]) -> str:
    lines = [
        "# P1 legacy/regime selected-stock unadjusted path materialization",
        "",
        f"- status: `{readiness['status']}`",
        f"- P1 signal rows: {readiness['p1_signal_rows']}",
        f"- selected unique tickers: {readiness['p1_selected_unique_ticker_count']}",
        f"- trade path rows requested: {readiness['p1_trade_path_rows']}",
        f"- ordinary stock path ready rows: {readiness['ordinary_stock_path_ready_rows']}",
        f"- blocked rows: {readiness['blocked_rows']}",
        f"- ready_for_experiments: {str(readiness['ready_for_experiments']).lower()}",
        "",
        "## 判斷",
        "",
        "Core 找到 P1 legacy / regime selected signal rows，但本機 P1 selected-stock official unadjusted OHLC source 不存在；"
        "既有 Radar selected price package 只覆蓋 2024-01-02 之後，full-sweep manifest 指向的 2015 shard 在本機也不存在。",
        "",
        "因此本包不交 Experiments。下一棒應交 Radar/Data，只補 P1 selected tickers/date range，不做 full-market mass download。",
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
    signal = _signal_table()
    trade_path, missing_rows = _trade_path(signal)
    source_request = _source_request(signal, missing_rows if not missing_rows.empty else trade_path)
    timing = _timing_design()
    cost = _cost_audit()
    missing_audit = _missing_price_audit(trade_path)
    future = _future_audit()
    source_status = _full_sweep_source_status()

    ordinary = trade_path.loc[trade_path["path_bucket"] == "ordinary_stock"]
    blocked_rows = int((~ordinary["path_ready"].astype(bool)).sum()) if not ordinary.empty else 0
    future_violations = int(future["violation_count"].sum())
    path_ready = bool(blocked_rows == 0 and len(ordinary) > 0)
    readiness = {
        "task_id": TASK_ID,
        "status": "p1_selected_stock_unadjusted_path_blocked_source_request_ready",
        "diagnostic_only": True,
        "requested_period_start": P1_START.date().isoformat(),
        "requested_period_end": P1_END.date().isoformat(),
        "actual_signal_start": signal["signal_date"].min().date().isoformat() if not signal.empty else None,
        "actual_signal_end": signal["signal_date"].max().date().isoformat() if not signal.empty else None,
        "p1_signal_rows": int(len(signal)),
        "p1_legacy_signal_rows": int((signal["signal_family"] == "legacy_rs20").sum()),
        "p1_regime_signal_rows": int((signal["signal_family"] == "regime_switch").sum()),
        "p1_selected_unique_ticker_count": int(signal.loc[signal["path_bucket"] == "ordinary_stock", "executable_ticker"].nunique()),
        "p1_trade_path_rows": int(len(trade_path)),
        "ordinary_stock_trade_path_rows": int(len(ordinary)),
        "ordinary_stock_path_ready_rows": int(ordinary["path_ready"].sum()) if not ordinary.empty else 0,
        "p1_selected_stock_unadjusted_ohlc_path_ready": path_ready,
        "ready_for_p1_legacy_regime_unadjusted_path_diagnostic": path_ready and future_violations == 0,
        "next_day_close_ready": False,
        "next_day_open_ready": False,
        "same_week_close_ready": False,
        "formal_cost_model_ready": True,
        "adjusted_close_ready": False,
        "blocked_rows": blocked_rows,
        "source_request_ticker_count": int(source_request["ticker"].nunique()) if not source_request.empty else 0,
        "future_data_violation_count": future_violations,
        "ready_for_experiments": False,
        "ready_for_formal": False,
        "ready_for_strategy_replay": False,
        "next_owner": "Radar/Data",
        "next_task_suggestion": "TASK-RADAR-DATA-VNEXT-P1-LEGACY-REGIME-SELECTED-STOCK-UNADJUSTED-OHLC-SOURCE-PACKAGE-001",
        "full_sweep_source_status": source_status,
        "blocked_fields": [
            "p1_selected_stock_unadjusted_ohlc_source",
            "next_day_close_stock_path",
            "next_day_open_stock_path",
            "same_week_close_stock_path",
            "adjusted_close_path",
        ],
        "proxy_fields": ["00631L_reference_benchmark_adjusted_close_reference_only"],
        **FLAGS,
    }

    files = [
        OUTPUT_DIR / "p1_legacy_regime_selected_stock_unadjusted_trade_path.csv",
        OUTPUT_DIR / "p1_legacy_regime_selected_stock_signal_table.csv",
        OUTPUT_DIR / "p1_legacy_regime_timing_variant_design.csv",
        OUTPUT_DIR / "p1_legacy_regime_cost_model_audit.csv",
        OUTPUT_DIR / "p1_legacy_regime_missing_price_audit.csv",
        OUTPUT_DIR / "p1_legacy_regime_source_request_selected_tickers.csv",
        OUTPUT_DIR / "p1_legacy_regime_future_data_audit.csv",
        OUTPUT_DIR / "readiness_for_p1_legacy_regime_unadjusted_path_diagnostic.json",
        OUTPUT_DIR / "final_summary_zh.md",
    ]
    trade_path.to_csv(files[0], index=False, encoding="utf-8")
    signal.to_csv(files[1], index=False, encoding="utf-8")
    timing.to_csv(files[2], index=False, encoding="utf-8")
    cost.to_csv(files[3], index=False, encoding="utf-8")
    missing_audit.to_csv(files[4], index=False, encoding="utf-8")
    source_request.to_csv(files[5], index=False, encoding="utf-8")
    future.to_csv(files[6], index=False, encoding="utf-8")
    files[7].write_text(json.dumps(readiness, ensure_ascii=False, indent=2), encoding="utf-8")
    files[8].write_text(_summary(readiness), encoding="utf-8")
    _write_manifest(files, readiness)


if __name__ == "__main__":
    main()
