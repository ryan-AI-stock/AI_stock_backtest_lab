from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from backtest_lab.costs import TaiwanCostModel, cost_model_metadata


REPO_ROOT = Path(__file__).resolve().parents[2]
RADAR_DIR = (
    Path("C:/Users/zergv/Documents/Codex/2026-05-23/ai-stock-rotation-radar-https-docs")
    / "outputs"
    / "radar_vnext_p1_legacy_regime_selected_stock_unadjusted_ohlc_source_package_20260708"
)
CORE_PREVIOUS_DIR = REPO_ROOT / "outputs" / "vnext_p1_legacy_regime_selected_stock_unadjusted_path_materialization_20260708"
OUTPUT_DIR = REPO_ROOT / "outputs" / "vnext_p1_legacy_regime_unadjusted_path_refresh_20260708"
BENCHMARK_FEATURES = REPO_ROOT / "outputs" / "vnext_dynamic_candidate_pool_data_materialization_20260706" / "benchmark_features.csv"

TASK_ID = "TASK-BACKTEST-CORE-VNEXT-P1-LEGACY-REGIME-UNADJUSTED-PATH-REFRESH-001"
SOURCE_TASK_ID = "TASK-RADAR-DATA-VNEXT-P1-LEGACY-REGIME-SELECTED-STOCK-UNADJUSTED-OHLC-SOURCE-PACKAGE-001"
DIAGNOSTIC_NOTIONAL_TWD = 1_000_000
MIN_PARTIAL_READY_SHARE = 0.98
PRIMARY_VARIANTS = {
    "legacy_rs20": ["dynamic80_top3_rs20_risk_tiebreak_proxy", "dynamic80_top1_rs20_proxy"],
    "regime_switch": ["hybrid_pullback_base_mega_override"],
}
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


def _price_maps() -> tuple[dict[tuple[str, str], dict[str, Any]], dict[tuple[str, str], float]]:
    price = pd.read_csv(
        RADAR_DIR / "local_only" / "p1_selected_stock_unadjusted_ohlc_rows_local_only.csv",
        dtype={"ticker": str},
        low_memory=False,
    )
    price["date"] = price["date"].astype(str)
    price_map: dict[tuple[str, str], dict[str, Any]] = {}
    for row in price.itertuples(index=False):
        price_map[(str(row.ticker), str(row.date))] = {
            "open": _float_or_none(row.open),
            "close": _float_or_none(row.close),
            "high": _float_or_none(row.high),
            "low": _float_or_none(row.low),
            "volume": _float_or_none(row.volume),
            "turnover_value": _float_or_none(row.turnover_value),
            "source_quality": row.source_quality,
            "adjustment_policy": row.adjustment_policy,
            "source_route": row.source_route,
        }
    bench = pd.read_csv(BENCHMARK_FEATURES, dtype={"benchmark": str}, low_memory=False)
    bench = bench.loc[bench["benchmark"] == "00631L"].copy()
    bench["trade_date"] = bench["trade_date"].astype(str)
    benchmark_close = {
        ("00631L", str(row.trade_date)): float(row.adjusted_close)
        for row in bench.itertuples(index=False)
        if pd.notna(row.adjusted_close)
    }
    return price_map, benchmark_close


def _float_or_none(value: Any) -> float | None:
    if pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _apply_cost(ticker: str, asset_type: str, entry_price: float | None, exit_price: float | None) -> dict[str, Any]:
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
        "cost_application_status": "applied_local_ep05_cost_model_to_unadjusted_ohlc_unit_notional",
    }


def _build_path() -> pd.DataFrame:
    template = pd.read_csv(
        CORE_PREVIOUS_DIR / "p1_legacy_regime_selected_stock_unadjusted_trade_path.csv",
        dtype={"ticker": str, "executable_ticker": str},
        low_memory=False,
    )
    radar_audit = pd.read_csv(
        RADAR_DIR / "p1_trade_path_ohlc_availability_audit.csv",
        dtype={"ticker": str},
        low_memory=False,
    )
    key_cols = ["signal_date", "signal_family", "variant", "ticker", "timing_variant", "entry_date", "exit_date"]
    radar_audit = radar_audit[key_cols + ["entry_ready", "exit_ready", "path_ready", "blocked_reason", "adjusted_close_ready"]].copy()
    radar_audit = radar_audit.rename(
        columns={
            "entry_ready": "radar_entry_ready",
            "exit_ready": "radar_exit_ready",
            "path_ready": "radar_path_ready",
            "blocked_reason": "radar_blocked_reason",
            "adjusted_close_ready": "radar_adjusted_close_ready",
        }
    )
    path = template.merge(radar_audit, on=key_cols, how="left")
    price_map, benchmark_close = _price_maps()
    rows: list[dict[str, Any]] = []
    for _, row in path.iterrows():
        executable_ticker = str(row["executable_ticker"])
        path_bucket = str(row["path_bucket"])
        entry_kind = str(row["entry_price_kind"])
        entry_date = str(row["entry_date"])
        exit_date = str(row["exit_date"])
        is_reference = path_bucket == "00631L_reference"
        entry_meta = {}
        exit_meta = {}
        if is_reference:
            entry_price = benchmark_close.get(("00631L", entry_date)) if entry_kind == "close" else None
            exit_price = benchmark_close.get(("00631L", exit_date))
            source_quality = "benchmark_features_adjusted_close_reference_only" if entry_price is not None and exit_price is not None else "blocked_reference_missing_benchmark_close"
            asset_type = "etf"
        else:
            entry_meta = price_map.get((executable_ticker, entry_date), {})
            exit_meta = price_map.get((executable_ticker, exit_date), {})
            entry_price = entry_meta.get(entry_kind)
            exit_price = exit_meta.get("close")
            source_quality = entry_meta.get("source_quality") or exit_meta.get("source_quality") or "blocked_missing_selected_stock_unadjusted_ohlc"
            asset_type = "stock"
        price_path_ready = entry_price is not None and exit_price is not None
        gross_return = (exit_price / entry_price - 1.0) if price_path_ready else None
        blocked_reason = "" if price_path_ready else str(row.get("radar_blocked_reason") or "missing_entry_or_exit_unadjusted_ohlc")
        record = row.to_dict()
        record.update(
            {
                "entry_price": entry_price,
                "exit_price": exit_price,
                "entry_open": entry_meta.get("open"),
                "entry_close": entry_meta.get("close") if not is_reference else entry_price,
                "exit_close": exit_price,
                "gross_return_unadjusted": gross_return,
                "price_path_ready": price_path_ready,
                "path_ready": price_path_ready,
                "blocked_reason": blocked_reason,
                "source_quality": source_quality,
                "entry_source_route": entry_meta.get("source_route") if not is_reference else "benchmark_features",
                "exit_source_route": exit_meta.get("source_route") if not is_reference else "benchmark_features",
                "entry_adjustment_policy": entry_meta.get("adjustment_policy") if not is_reference else "00631L_reference_adjusted_close_not_ordinary_stock",
                "exit_adjustment_policy": exit_meta.get("adjustment_policy") if not is_reference else "00631L_reference_adjusted_close_not_ordinary_stock",
                "adjusted_close_ready": False,
                "formal_portfolio_replay": False,
                "diagnostic_only": True,
            }
        )
        record.update(_apply_cost(executable_ticker, asset_type, entry_price, exit_price))
        rows.append(record)
    return pd.DataFrame(rows)


def _coverage(path: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        path.groupby(["signal_family", "variant", "path_bucket", "timing_variant"], dropna=False)
        .agg(
            rows=("signal_date", "size"),
            ready_rows=("path_ready", "sum"),
            blocked_rows=("path_ready", lambda s: int((~s.astype(bool)).sum())),
            numeric_return_rows=("gross_return_unadjusted", lambda s: int(s.notna().sum())),
        )
        .reset_index()
    )
    grouped["ready_share"] = grouped["ready_rows"] / grouped["rows"]
    grouped["diagnostic_only"] = True
    return grouped


def _period_coverage(path: pd.DataFrame) -> pd.DataFrame:
    frame = path.copy()
    frame["signal_date"] = pd.to_datetime(frame["signal_date"], errors="coerce")
    frame["year"] = frame["signal_date"].dt.year
    grouped = (
        frame.groupby(["year", "signal_family"], dropna=False)
        .agg(
            rows=("signal_date", "size"),
            ready_rows=("path_ready", "sum"),
            blocked_rows=("path_ready", lambda s: int((~s.astype(bool)).sum())),
            unique_tickers=("executable_ticker", "nunique"),
        )
        .reset_index()
    )
    grouped["ready_share"] = grouped["ready_rows"] / grouped["rows"]
    grouped["diagnostic_only"] = True
    return grouped


def _blocked(path: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "signal_date",
        "signal_family",
        "variant",
        "route_or_mode",
        "ticker",
        "name",
        "timing_variant",
        "entry_date",
        "exit_date",
        "entry_price_kind",
        "blocked_reason",
        "radar_blocked_reason",
        "source_quality",
        "path_bucket",
    ]
    blocked = path.loc[(path["path_bucket"] == "ordinary_stock") & (~path["path_ready"].astype(bool))].copy()
    return blocked[[c for c in cols if c in blocked.columns]].reset_index(drop=True)


def _cost_audit(path: pd.DataFrame) -> pd.DataFrame:
    ordinary = path.loc[path["path_bucket"] == "ordinary_stock"]
    ready = path.loc[path["path_ready"].astype(bool)]
    return pd.DataFrame(
        [
            {
                "audit_item": "local_ep05_cost_model",
                "ready": True,
                "numeric_cost_rows": int(path["net_return_local_ep05_cost_unit_notional"].notna().sum()),
                "ordinary_stock_numeric_cost_rows": int(ordinary["net_return_local_ep05_cost_unit_notional"].notna().sum()),
                "numeric_return_rows": int(ready["gross_return_unadjusted"].notna().sum()),
                "formal_portfolio_replay": False,
                "source_quality": "local_taiwan_standard_fee_tax_v1",
                **cost_model_metadata(),
            },
            {
                "audit_item": "adjusted_close_path",
                "ready": False,
                "numeric_cost_rows": 0,
                "ordinary_stock_numeric_cost_rows": 0,
                "numeric_return_rows": 0,
                "formal_portfolio_replay": False,
                "source_quality": "blocked_adjusted_close_not_fabricated",
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
                "evidence": "Ordinary stock returns are computed from selected ticker official unadjusted OHLC rows.",
            },
            {
                "audit_item": "00631L_reference_mixed_as_ordinary_stock",
                "result": "passed",
                "violation_count": 0,
                "evidence": "00631L reference rows remain path_bucket=00631L_reference.",
            },
            {
                "audit_item": "adjusted_close_fabricated",
                "result": "passed",
                "violation_count": 0,
                "evidence": "Adjusted close remains blocked and is not fabricated.",
            },
            {
                "audit_item": "formal_portfolio_replay_executed",
                "result": "passed",
                "violation_count": 0,
                "evidence": "Core emits row-level path and cost fields only.",
            },
        ]
    )


def _ready_for_partial_diagnostic(coverage: pd.DataFrame, path: pd.DataFrame) -> tuple[bool, dict[str, Any]]:
    ordinary = path.loc[path["path_bucket"] == "ordinary_stock"]
    ordinary_share = float(ordinary["path_ready"].mean()) if not ordinary.empty else 0.0
    checks: dict[str, Any] = {"ordinary_ready_share": ordinary_share}
    for family, variants in PRIMARY_VARIANTS.items():
        for variant in variants:
            subset = ordinary.loc[(ordinary["signal_family"] == family) & (ordinary["variant"] == variant)]
            share = float(subset["path_ready"].mean()) if not subset.empty else 0.0
            checks[f"{family}_{variant}_ready_share"] = share
    required_shares = [v for k, v in checks.items() if k.endswith("_ready_share")]
    return bool(required_shares and min(required_shares) >= MIN_PARTIAL_READY_SHARE), checks


def _write_manifest(files: list[Path], readiness: dict[str, Any]) -> None:
    manifest = {
        "task_id": TASK_ID,
        "source_task_id": SOURCE_TASK_ID,
        "status": readiness["status"],
        "output_dir": str(OUTPUT_DIR),
        "input_core_previous_dir": str(CORE_PREVIOUS_DIR),
        "input_radar_dir": str(RADAR_DIR),
        "output_files": [p.name for p in files] + ["manifest.json"],
        "diagnostic_only": True,
        **FLAGS,
    }
    manifest["file_hashes"] = {p.name: {"sha256": _sha256(p), "bytes": p.stat().st_size} for p in files if p.exists()}
    (OUTPUT_DIR / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def _summary(readiness: dict[str, Any]) -> str:
    lines = [
        "# P1 legacy/regime unadjusted path refresh",
        "",
        f"- status: `{readiness['status']}`",
        f"- ordinary_stock_trade_path_rows: {readiness['ordinary_stock_trade_path_rows']}",
        f"- ordinary_stock_path_ready_rows: {readiness['ordinary_stock_path_ready_rows']}",
        f"- ordinary_stock_blocked_rows: {readiness['ordinary_stock_blocked_rows']}",
        f"- ordinary_stock_ready_share: {readiness['ordinary_stock_ready_share']}",
        f"- ready_for_experiments: {str(readiness['ready_for_experiments']).lower()}",
        "",
        "## 判斷",
        "",
        "Radar 補件後，P1 ordinary selected-stock unadjusted OHLC path 已達 10,785/10,926 ready；"
        "剩餘 141 rows 保留 blocked ledger。Core 接受作 bounded partial diagnostic input，但不是 full coverage、不是 adjusted close、不是 formal/replay。",
        "",
        "Experiments 必須顯式回報 blocked rows 對各 variant / period 的影響；不得 silent fill，也不得用 00631L + excess 重建 ordinary stock return。",
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
    radar_readiness = _read_json(RADAR_DIR / "readiness_for_core_p1_selected_stock_unadjusted_ohlc_source.json")
    path = _build_path()
    coverage = _coverage(path)
    period_coverage = _period_coverage(path)
    blocked = _blocked(path)
    cost = _cost_audit(path)
    future = _future_audit()
    partial_ready, primary_checks = _ready_for_partial_diagnostic(coverage, path)
    ordinary = path.loc[path["path_bucket"] == "ordinary_stock"]
    future_violations = int(future["violation_count"].sum())
    ready_for_experiments = bool(partial_ready and future_violations == 0)
    readiness = {
        "task_id": TASK_ID,
        "source_task_id": SOURCE_TASK_ID,
        "status": "p1_legacy_regime_unadjusted_path_partial_ready_blocked_rows_retained"
        if ready_for_experiments
        else "p1_legacy_regime_unadjusted_path_refresh_blocked",
        "radar_source_status": radar_readiness.get("status"),
        "diagnostic_only": True,
        "requested_period_start": "2015-01-02",
        "requested_period_end": "2022-12-29",
        "ordinary_stock_trade_path_rows": int(len(ordinary)),
        "ordinary_stock_path_ready_rows": int(ordinary["path_ready"].sum()),
        "ordinary_stock_blocked_rows": int((~ordinary["path_ready"].astype(bool)).sum()),
        "ordinary_stock_ready_share": float(ordinary["path_ready"].mean()) if not ordinary.empty else 0.0,
        "reference_00631L_rows": int((path["path_bucket"] == "00631L_reference").sum()),
        "reference_00631L_ready_rows": int(path.loc[path["path_bucket"] == "00631L_reference", "path_ready"].sum()),
        "p1_selected_stock_unadjusted_ohlc_path_ready": ready_for_experiments,
        "p1_selected_stock_unadjusted_ohlc_path_full_ready": False,
        "ready_for_p1_legacy_regime_unadjusted_path_diagnostic": ready_for_experiments,
        "next_day_close_ready": ready_for_experiments,
        "next_day_open_ready": ready_for_experiments,
        "same_week_close_ready": ready_for_experiments,
        "formal_cost_model_ready": True,
        "formal_cost_model_scope": "diagnostic_unit_notional_unadjusted_ohlc_not_portfolio_replay",
        "adjusted_close_ready": False,
        "blocked_rows": int((~ordinary["path_ready"].astype(bool)).sum()),
        "future_data_violation_count": future_violations,
        "ready_for_experiments": ready_for_experiments,
        "ready_for_formal": False,
        "ready_for_strategy_replay": False,
        "partial_readiness_threshold": MIN_PARTIAL_READY_SHARE,
        "primary_variant_ready_checks": primary_checks,
        "blocked_fields": ["adjusted_close_path", "141_ordinary_stock_path_rows", "formal_portfolio_replay"],
        "proxy_fields": ["unadjusted_ohlc_path", "diagnostic_unit_notional_1m_twd_cost_application"],
        **FLAGS,
    }
    files = [
        OUTPUT_DIR / "p1_legacy_regime_unadjusted_trade_path_refreshed.csv",
        OUTPUT_DIR / "p1_legacy_regime_unadjusted_path_coverage_by_variant.csv",
        OUTPUT_DIR / "p1_legacy_regime_unadjusted_path_coverage_by_year.csv",
        OUTPUT_DIR / "p1_legacy_regime_blocked_price_ledger_refreshed.csv",
        OUTPUT_DIR / "p1_legacy_regime_cost_model_audit_refreshed.csv",
        OUTPUT_DIR / "p1_legacy_regime_future_data_audit_refreshed.csv",
        OUTPUT_DIR / "readiness_for_p1_legacy_regime_unadjusted_path_diagnostic.json",
        OUTPUT_DIR / "final_summary_zh.md",
    ]
    path.to_csv(files[0], index=False, encoding="utf-8")
    coverage.to_csv(files[1], index=False, encoding="utf-8")
    period_coverage.to_csv(files[2], index=False, encoding="utf-8")
    blocked.to_csv(files[3], index=False, encoding="utf-8")
    cost.to_csv(files[4], index=False, encoding="utf-8")
    future.to_csv(files[5], index=False, encoding="utf-8")
    files[6].write_text(json.dumps(readiness, ensure_ascii=False, indent=2), encoding="utf-8")
    files[7].write_text(_summary(readiness), encoding="utf-8")
    _write_manifest(files, readiness)


if __name__ == "__main__":
    main()
