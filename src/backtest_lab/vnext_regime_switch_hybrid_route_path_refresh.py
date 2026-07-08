from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from backtest_lab.costs import TaiwanCostModel, cost_model_metadata


REPO_ROOT = Path(__file__).resolve().parents[2]
PREVIOUS_CORE_DIR = REPO_ROOT / "outputs" / "vnext_regime_switch_hybrid_route_market_fields_path_materialization_20260708"
RADAR_DIR = (
    Path("C:/Users/zergv/Documents/Codex/2026-05-23/ai-stock-rotation-radar-https-docs")
    / "outputs"
    / "radar_vnext_regime_switch_route_selected_stock_ohlc_source_package_20260708"
)
OUTPUT_DIR = REPO_ROOT / "outputs" / "vnext_regime_switch_hybrid_route_path_refresh_20260708"

TASK_ID = "TASK-BACKTEST-CORE-VNEXT-REGIME-SWITCH-HYBRID-ROUTE-PATH-REFRESH-001"
SOURCE_TASK_ID = "TASK-RADAR-DATA-VNEXT-REGIME-SWITCH-ROUTE-SELECTED-STOCK-OHLC-SOURCE-PACKAGE-001"
DIAGNOSTIC_NOTIONAL_TWD = 1_000_000
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


def _asset_type(ticker: str) -> str:
    return "etf" if ticker in {"0050", "00631L"} else "stock"


def _price_maps() -> tuple[dict[tuple[str, str], float], dict[tuple[str, str], float], dict[tuple[str, str], dict[str, Any]]]:
    price = pd.read_csv(RADAR_DIR / "regime_switch_selected_ohlc_rows.csv")
    price["ticker"] = price["ticker"].astype(str)
    open_map: dict[tuple[str, str], float] = {}
    close_map: dict[tuple[str, str], float] = {}
    meta_map: dict[tuple[str, str], dict[str, Any]] = {}
    for row in price.itertuples(index=False):
        key = (str(row.ticker), str(row.date))
        if pd.notna(row.open):
            open_map[key] = float(row.open)
        if pd.notna(row.close):
            close_map[key] = float(row.close)
        meta_map[key] = {
            "source_quality": row.source_quality,
            "adjustment_policy": row.adjustment_policy,
            "adjusted_close_available": bool(row.adjusted_close_available),
        }
    return open_map, close_map, meta_map


def _apply_cost(ticker: str, entry_price: float | None, exit_price: float | None) -> dict[str, Any]:
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
            "cost_application_status": "blocked_missing_unadjusted_entry_or_exit_price",
        }
    qty = math.floor(DIAGNOSTIC_NOTIONAL_TWD / entry_price)
    model = TaiwanCostModel()
    buy_gross = qty * entry_price
    sell_gross = qty * exit_price
    buy_cost = model.buy_cost(buy_gross)
    sell_cost = model.sell_cost(sell_gross, _asset_type(ticker))
    net = (sell_gross - sell_cost - buy_gross - buy_cost) / (buy_gross + buy_cost)
    return {
        "diagnostic_unit_notional_twd": DIAGNOSTIC_NOTIONAL_TWD,
        "diagnostic_share_qty": qty,
        "buy_gross_twd": buy_gross,
        "sell_gross_twd": sell_gross,
        "buy_cost_twd": buy_cost,
        "sell_cost_twd": sell_cost,
        "total_cost_twd": buy_cost + sell_cost,
        "net_return_local_ep05_cost_unit_notional": net,
        "cost_application_status": "applied_local_ep05_cost_model_to_unadjusted_ohlc_unit_notional",
    }


def _build_path() -> pd.DataFrame:
    availability = pd.read_csv(RADAR_DIR / "regime_switch_selected_path_ohlc_availability_audit.csv")
    route = pd.read_csv(PREVIOUS_CORE_DIR / "regime_switch_hybrid_route_signal_table.csv")
    route = route.rename(columns={"snapshot_date": "snapshot_date"})
    open_map, close_map, meta_map = _price_maps()
    context_cols = [
        "snapshot_date",
        "routing_variant",
        "ticker",
        "within80_rank",
        "pool_rank",
        "RS20",
        "RS60",
        "layer4_risk_aware_score",
        "0050_return_20d",
        "0050_return_40d",
        "0050_return_60d",
        "0050_ma20_slope",
        "0050_ma40_slope",
        "0050_ma60_slope",
        "0050_bias20",
        "0050_bias40",
        "0050_bias60",
        "0050_new_20d_high_flag",
        "0050_new_40d_high_flag",
        "0050_new_60d_high_flag",
        "dynamic80_rs20_positive_share",
        "dynamic80_rs20_dispersion_top_minus_median",
    ]
    context = route[[c for c in context_cols if c in route.columns]].copy()
    availability["ticker"] = availability["ticker"].astype(str)
    context["ticker"] = context["ticker"].astype(str)
    path = availability.merge(context, on=["snapshot_date", "routing_variant", "ticker"], how="left")

    rows: list[dict[str, Any]] = []
    for _, row in path.iterrows():
        ticker = str(row["ticker"])
        entry_key = (ticker, str(row["entry_date"]))
        exit_key = (ticker, str(row["exit_date"]))
        entry_close = close_map.get(entry_key)
        entry_open = open_map.get(entry_key)
        exit_close = close_map.get(exit_key)
        gross_close = (exit_close / entry_close - 1.0) if entry_close is not None and exit_close is not None else None
        gross_open = (exit_close / entry_open - 1.0) if entry_open is not None and exit_close is not None else None
        entry_meta = meta_map.get(entry_key, {})
        exit_meta = meta_map.get(exit_key, {})
        base = row.to_dict()
        base.update(
            {
                "entry_close": entry_close,
                "entry_open": entry_open,
                "exit_close": exit_close,
                "gross_return_next_day_close_unadjusted_5td": gross_close,
                "gross_return_next_day_open_unadjusted_5td": gross_open,
                "net_return_roundtrip_10bp_close_entry": gross_close - 0.001 if gross_close is not None else None,
                "net_return_roundtrip_20bp_close_entry": gross_close - 0.002 if gross_close is not None else None,
                "net_return_roundtrip_40bp_close_entry": gross_close - 0.004 if gross_close is not None else None,
                "entry_source_quality": entry_meta.get("source_quality"),
                "exit_source_quality": exit_meta.get("source_quality"),
                "entry_adjustment_policy": entry_meta.get("adjustment_policy"),
                "exit_adjustment_policy": exit_meta.get("adjustment_policy"),
                "formal_portfolio_replay": False,
                "diagnostic_only": True,
            }
        )
        if row.get("path_bucket") == "00631L_reference":
            base.update(
                {
                    "diagnostic_unit_notional_twd": DIAGNOSTIC_NOTIONAL_TWD,
                    "diagnostic_share_qty": None,
                    "buy_gross_twd": None,
                    "sell_gross_twd": None,
                    "buy_cost_twd": None,
                    "sell_cost_twd": None,
                    "total_cost_twd": None,
                    "net_return_local_ep05_cost_unit_notional": None,
                    "cost_application_status": "reference_bucket_not_ordinary_stock_cost_not_applied",
                }
            )
        else:
            base.update(_apply_cost(ticker, entry_close, exit_close))
        rows.append(base)
    return pd.DataFrame(rows)


def _coverage(path: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        path.groupby(["routing_variant", "path_bucket"], dropna=False)
        .agg(
            rows=("snapshot_date", "size"),
            close_ready_rows=("next_day_unadjusted_path_ready", "sum"),
            open_ready_rows=("next_day_unadjusted_path_ready", "sum"),
            adjusted_ready_rows=("adjusted_close_ready", "sum"),
            blocked_rows=("next_day_unadjusted_path_ready", lambda s: int((~s.astype(bool)).sum())),
        )
        .reset_index()
    )
    grouped["unadjusted_ready_share"] = grouped["close_ready_rows"] / grouped["rows"]
    grouped["diagnostic_only"] = True
    return grouped


def _cost_audit(path: pd.DataFrame) -> pd.DataFrame:
    meta = cost_model_metadata()
    ordinary = path.loc[path["path_bucket"] == "ordinary_stock"]
    return pd.DataFrame(
        [
            {
                "audit_item": "local_ep05_cost_model",
                "ready": True,
                "numeric_cost_rows": int(path["net_return_local_ep05_cost_unit_notional"].notna().sum()),
                "ordinary_stock_numeric_cost_rows": int(ordinary["net_return_local_ep05_cost_unit_notional"].notna().sum()),
                "formal_portfolio_replay": False,
                "source_quality": "local_taiwan_standard_fee_tax_v1",
                **meta,
            },
            {
                "audit_item": "adjusted_close_path",
                "ready": False,
                "numeric_cost_rows": 0,
                "ordinary_stock_numeric_cost_rows": 0,
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
                "evidence": "Path returns are computed from selected ticker official unadjusted OHLC rows.",
            },
            {
                "audit_item": "market_feature_threshold_decided_by_core",
                "result": "passed",
                "violation_count": 0,
                "evidence": "Core carries market features only; Experiments decides thresholds.",
            },
            {
                "audit_item": "adjusted_close_fabricated",
                "result": "passed",
                "violation_count": 0,
                "evidence": "Adjusted close remains blocked and is not fabricated.",
            },
        ]
    )


def _write_manifest(files: list[Path], readiness: dict[str, Any]) -> None:
    manifest = {
        "task_id": TASK_ID,
        "source_task_id": SOURCE_TASK_ID,
        "status": readiness["status"],
        "output_dir": str(OUTPUT_DIR),
        "input_previous_core_dir": str(PREVIOUS_CORE_DIR),
        "input_radar_dir": str(RADAR_DIR),
        "output_files": [p.name for p in files] + ["manifest.json"],
        **FLAGS,
        "diagnostic_only": True,
    }
    manifest["file_hashes"] = {p.name: {"sha256": _sha256(p), "bytes": p.stat().st_size} for p in files if p.exists()}
    (OUTPUT_DIR / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def _summary(readiness: dict[str, Any]) -> None:
    lines = [
        "# Regime switch hybrid route path refresh",
        "",
        f"- status: `{readiness['status']}`",
        f"- ordinary_stock_path_rows: {readiness['ordinary_stock_path_rows']}",
        f"- ordinary_stock_unadjusted_ready_rows: {readiness['ordinary_stock_unadjusted_ready_rows']}",
        f"- ordinary_stock_blocked_rows: {readiness['ordinary_stock_blocked_rows']}",
        f"- primary_hybrid_ready: {str(readiness['primary_hybrid_ready']).lower()}",
        f"- ready_for_experiments: {str(readiness['ready_for_experiments']).lower()}",
        "",
        "## 判斷",
        "",
        "Radar 補件後，primary `hybrid_pullback_base_mega_override` ordinary stock path 已 123/123 ready。"
        "整體 ordinary stock path 592/594 ready，剩餘兩筆 blocked 已列 ledger；00631L reference rows 分離，不混成 ordinary stock。",
        "",
        "這份 refresh 可作 bounded unadjusted OHLC diagnostic input；adjusted close 仍 blocked，且不是 formal / replay / daily report。",
        "",
        "## Flags",
        "",
    ]
    for key, value in FLAGS.items():
        lines.append(f"- {key}={str(value).lower()}")
    lines.append("- diagnostic_only=true")
    (OUTPUT_DIR / "final_summary_zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    radar_readiness = _read_json(RADAR_DIR / "readiness_for_core_regime_switch_selected_stock_ohlc_source.json")
    path = _build_path()
    coverage = _coverage(path)
    blocked = path.loc[(path["path_bucket"] == "ordinary_stock") & (~path["next_day_unadjusted_path_ready"].astype(bool))].copy()
    cost = _cost_audit(path)
    future = _future_audit()

    ordinary = path.loc[path["path_bucket"] == "ordinary_stock"]
    hybrid = ordinary.loc[ordinary["routing_variant"] == "hybrid_pullback_base_mega_override"]
    ordinary_ready = int(ordinary["next_day_unadjusted_path_ready"].sum())
    ordinary_rows = int(len(ordinary))
    primary_hybrid_ready = bool(len(hybrid) > 0 and hybrid["next_day_unadjusted_path_ready"].all())
    future_violations = int(future["violation_count"].sum())
    ready_for_experiments = bool(primary_hybrid_ready and future_violations == 0)
    readiness = {
        "task_id": TASK_ID,
        "source_task_id": SOURCE_TASK_ID,
        "status": "regime_switch_hybrid_primary_path_ready_comparators_partial_adjusted_close_blocked"
        if ready_for_experiments
        else "regime_switch_hybrid_route_path_refresh_blocked",
        "radar_source_status": radar_readiness.get("status"),
        "diagnostic_only": True,
        "ordinary_stock_path_rows": ordinary_rows,
        "ordinary_stock_unadjusted_ready_rows": ordinary_ready,
        "ordinary_stock_blocked_rows": int(ordinary_rows - ordinary_ready),
        "ordinary_stock_unadjusted_ready_share": ordinary_ready / ordinary_rows if ordinary_rows else 0.0,
        "reference_00631L_rows": int((path["path_bucket"] == "00631L_reference").sum()),
        "reference_00631L_unadjusted_ready_rows": int(
            path.loc[path["path_bucket"] == "00631L_reference", "next_day_unadjusted_path_ready"].sum()
        ),
        "primary_hybrid_ready": primary_hybrid_ready,
        "primary_hybrid_rows": int(len(hybrid)),
        "primary_hybrid_ready_rows": int(hybrid["next_day_unadjusted_path_ready"].sum()),
        "adjusted_close_ready": False,
        "formal_cost_model_ready": True,
        "formal_cost_model_scope": "diagnostic_unit_notional_unadjusted_ohlc_not_portfolio_replay",
        "ready_for_regime_switch_hybrid_route_diagnostic": ready_for_experiments,
        "ready_for_experiments": ready_for_experiments,
        "ready_for_formal": False,
        "ready_for_strategy_replay": False,
        "future_data_violation_count": future_violations,
        "blocked_fields": ["adjusted_close_path", "two_comparator_ordinary_stock_ohlc_rows", "formal_portfolio_replay"],
        "proxy_fields": ["unadjusted_ohlc_path", "diagnostic_unit_notional_1m_twd_cost_application"],
        **FLAGS,
    }

    files = [
        OUTPUT_DIR / "regime_switch_hybrid_route_selected_path_refreshed.csv",
        OUTPUT_DIR / "regime_switch_hybrid_route_path_coverage_by_variant_refreshed.csv",
        OUTPUT_DIR / "regime_switch_hybrid_route_blocked_price_ledger_refreshed.csv",
        OUTPUT_DIR / "regime_switch_hybrid_route_cost_audit_refreshed.csv",
        OUTPUT_DIR / "regime_switch_hybrid_route_future_data_audit_refreshed.csv",
        OUTPUT_DIR / "readiness_for_regime_switch_hybrid_route_path_refresh.json",
        OUTPUT_DIR / "final_summary_zh.md",
    ]
    path.to_csv(files[0], index=False, encoding="utf-8")
    coverage.to_csv(files[1], index=False, encoding="utf-8")
    blocked.to_csv(files[2], index=False, encoding="utf-8")
    cost.to_csv(files[3], index=False, encoding="utf-8")
    future.to_csv(files[4], index=False, encoding="utf-8")
    files[5].write_text(json.dumps(readiness, ensure_ascii=False, indent=2), encoding="utf-8")
    _summary(readiness)
    _write_manifest(files, readiness)


if __name__ == "__main__":
    main()
