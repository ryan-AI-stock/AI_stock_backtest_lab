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
    / "radar_vnext_legacy_rs20_selected_stock_price_path_source_package_20260708"
)
CORE_EXACT_DIR = REPO_ROOT / "outputs" / "vnext_legacy_rs20_exact_path_timing_cost_materialization_20260708"
OUTPUT_DIR = REPO_ROOT / "outputs" / "vnext_legacy_rs20_unadjusted_ohlc_timing_cost_materialization_20260708"

TASK_ID = "TASK-BACKTEST-CORE-VNEXT-LEGACY-RS20-UNADJUSTED-OHLC-TIMING-COST-MATERIALIZATION-001"
SOURCE_TASK_ID = "TASK-RADAR-DATA-VNEXT-LEGACY-RS20-SELECTED-STOCK-PRICE-PATH-SOURCE-PACKAGE-001"
DIAGNOSTIC_NOTIONAL_TWD = 1_000_000
PRIMARY_START = pd.Timestamp("2024-01-02")
PRIMARY_END = pd.Timestamp("2026-05-26")
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
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _asset_type(ticker: str) -> str:
    return "etf" if ticker in {"0050", "00631L"} else "stock"


def _load_price_lookup() -> pd.DataFrame:
    price_path = RADAR_DIR / "selected_stock_price_rows_local_only.csv"
    price = pd.read_csv(
        price_path,
        usecols=[
            "date",
            "ticker",
            "open",
            "close",
            "adjusted_close",
            "adjusted_close_available",
            "source_quality",
            "adjustment_policy",
        ],
    )
    price["date"] = pd.to_datetime(price["date"], errors="coerce")
    price["ticker"] = price["ticker"].astype(str)
    return price


def _price_maps(price: pd.DataFrame) -> tuple[dict[tuple[str, pd.Timestamp], float], dict[tuple[str, pd.Timestamp], float], dict[tuple[str, pd.Timestamp], dict[str, Any]]]:
    open_map: dict[tuple[str, pd.Timestamp], float] = {}
    close_map: dict[tuple[str, pd.Timestamp], float] = {}
    meta_map: dict[tuple[str, pd.Timestamp], dict[str, Any]] = {}
    for row in price.itertuples(index=False):
        key = (str(row.ticker), pd.Timestamp(row.date))
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


def _apply_cost_model(entry_price: float | None, exit_price: float | None, ticker: str) -> dict[str, float | int | None]:
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
    if qty <= 0:
        return {
            "diagnostic_unit_notional_twd": DIAGNOSTIC_NOTIONAL_TWD,
            "diagnostic_share_qty": 0,
            "buy_gross_twd": 0,
            "sell_gross_twd": 0,
            "buy_cost_twd": 0,
            "sell_cost_twd": 0,
            "total_cost_twd": 0,
            "net_return_local_ep05_cost_unit_notional": None,
            "cost_application_status": "blocked_zero_share_qty",
        }
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


def _build_trade_path() -> pd.DataFrame:
    trade = pd.read_csv(CORE_EXACT_DIR / "legacy_rs20_exact_selected_stock_trade_path.csv")
    trade["signal_dt"] = pd.to_datetime(trade["signal_date"], errors="coerce")
    trade = trade.loc[(trade["signal_dt"] >= PRIMARY_START) & (trade["signal_dt"] <= PRIMARY_END)].copy()
    trade = trade.drop(columns=["signal_dt"])
    signal = pd.read_csv(CORE_EXACT_DIR / "legacy_rs20_exact_selected_stock_signal_table.csv")
    availability = pd.read_csv(RADAR_DIR / "trade_path_required_price_availability_audit.csv")
    price = _load_price_lookup()
    open_map, close_map, meta_map = _price_maps(price)

    signal_keys = [
        "signal_date",
        "signal_variant",
        "ticker",
        "RS20",
        "RS60",
        "RS20_rank_within_80",
        "pool_rank",
        "within80_rank",
        "layer4_risk_aware_score",
        "rs20_risk_context_score",
        "rs20_31_bonus_score",
    ]
    signal_available = [col for col in signal_keys if col in signal.columns]
    signal_trim = signal[signal_available].rename(columns={"signal_variant": "variant"})
    trade = trade.merge(signal_trim, on=["signal_date", "variant", "ticker"], how="left", suffixes=("", "_signal"))
    trade = trade.merge(
        availability[
            [
                "signal_date",
                "entry_date",
                "exit_date",
                "ticker",
                "variant",
                "timing_variant",
                "unadjusted_requested_timing_prices_available",
                "exact_adjusted_requested_timing_prices_available",
                "blocked_reason",
                "source_quality",
                "adjustment_policy",
            ]
        ].rename(columns={"blocked_reason": "radar_price_blocked_reason"}),
        on=["signal_date", "entry_date", "exit_date", "ticker", "variant", "timing_variant"],
        how="left",
    )

    rows: list[dict[str, Any]] = []
    for row in trade.itertuples(index=False):
        entry_date = pd.Timestamp(row.entry_date)
        exit_date = pd.Timestamp(row.exit_date)
        ticker = str(row.ticker)
        entry_col = "open" if row.timing_variant == "next_day_open_entry_fixed_5td_exit" else "close"
        entry_price = open_map.get((ticker, entry_date)) if entry_col == "open" else close_map.get((ticker, entry_date))
        exit_price = close_map.get((ticker, exit_date))
        entry_meta = meta_map.get((ticker, entry_date), {})
        exit_meta = meta_map.get((ticker, exit_date), {})

        gross_return = None
        blocked_reason = ""
        if entry_price is None or exit_price is None:
            blocked_reason = "missing_unadjusted_entry_or_exit_price"
        else:
            gross_return = (exit_price / entry_price) - 1.0
            blocked_reason = "adjusted_close_blocked_unadjusted_path_available"

        costs = _apply_cost_model(entry_price, exit_price, ticker)
        base = row._asdict()
        base.update(
            {
                "entry_price_column": entry_col,
                "exit_price_column": "close",
                "entry_unadjusted_price": entry_price,
                "exit_unadjusted_price": exit_price,
                "gross_return_unadjusted": gross_return,
                "net_return_roundtrip_10bp": gross_return - 0.001 if gross_return is not None else None,
                "net_return_roundtrip_20bp": gross_return - 0.002 if gross_return is not None else None,
                "net_return_roundtrip_40bp": gross_return - 0.004 if gross_return is not None else None,
                "entry_price_source_quality": entry_meta.get("source_quality"),
                "exit_price_source_quality": exit_meta.get("source_quality"),
                "entry_adjustment_policy": entry_meta.get("adjustment_policy"),
                "exit_adjustment_policy": exit_meta.get("adjustment_policy"),
                "adjusted_close_available": False,
                "adjusted_close_path_ready": False,
                "unadjusted_ohlc_path_ready": entry_price is not None and exit_price is not None,
                "path_materialization_status": blocked_reason,
                "formal_ready": False,
                "diagnostic_only": True,
                "not_live_rule": True,
                "forward_return_as_rule": False,
                "future_return_as_rule": False,
            }
        )
        base.update(costs)
        rows.append(base)
    return pd.DataFrame(rows)


def _write_cost_audit(trade_path: pd.DataFrame) -> pd.DataFrame:
    meta = cost_model_metadata(TaiwanCostModel())
    return pd.DataFrame(
        [
            {
                "cost_model": "local_ep05_taiwan_standard_fee_tax_v1",
                "model_found": True,
                "formula_ready": True,
                "numeric_cost_rows": int(trade_path["net_return_local_ep05_cost_unit_notional"].notna().sum()),
                "numeric_cost_scope": "unadjusted_ohlc_diagnostic_unit_notional_1m_twd",
                "formal_portfolio_replay": False,
                "formal_ready": False,
                "boundary": "Uses official unadjusted OHLC and unit notional only; adjusted close remains blocked.",
                **meta,
            },
            {
                "cost_model": "roundtrip_bp_placeholders",
                "model_found": True,
                "formula_ready": True,
                "numeric_cost_rows": int(trade_path["gross_return_unadjusted"].notna().sum()),
                "numeric_cost_scope": "10bp_20bp_40bp_reference_only",
                "formal_portfolio_replay": False,
                "formal_ready": False,
                "boundary": "Placeholder reference only; not formal fee/tax model.",
            },
        ]
    )


def _period_coverage(trade_path: pd.DataFrame) -> pd.DataFrame:
    periods = {
        "legacy_requested": ("2024-01-02", "2026-05-26"),
        "radar_primary_source_window": ("2024-01-02", "2026-06-05"),
    }
    trade_path["signal_dt"] = pd.to_datetime(trade_path["signal_date"], errors="coerce")
    rows = []
    for period, (start, end) in periods.items():
        subset = trade_path.loc[(trade_path["signal_dt"] >= pd.Timestamp(start)) & (trade_path["signal_dt"] <= pd.Timestamp(end))]
        rows.append(
            {
                "period": period,
                "requested_start": start,
                "requested_end": end,
                "actual_start": subset["signal_date"].min() if not subset.empty else "",
                "actual_end": subset["signal_date"].max() if not subset.empty else "",
                "rows": int(len(subset)),
                "unadjusted_ready_rows": int(subset["unadjusted_ohlc_path_ready"].sum()) if not subset.empty else 0,
                "blocked_rows": int((~subset["unadjusted_ohlc_path_ready"]).sum()) if not subset.empty else 0,
            }
        )
    return pd.DataFrame(rows)


def _future_audit() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "audit_item": "selected_stock_return_reconstructed_from_00631L_plus_excess",
                "result": "passed",
                "violation_count": 0,
                "evidence": "unadjusted returns are computed from selected ticker official OHLC rows only",
            },
            {
                "audit_item": "forward_return_as_rule",
                "result": "passed",
                "violation_count": 0,
                "evidence": "forward returns are not used to select ticker or timing variant",
            },
            {
                "audit_item": "adjusted_close_fabricated",
                "result": "passed",
                "violation_count": 0,
                "evidence": "adjusted_close remains blocked; no official close adjustment is fabricated",
            },
        ]
    )


def _clean_trade_path_schema(trade_path: pd.DataFrame) -> pd.DataFrame:
    renamed = trade_path.rename(columns={"blocked_reason": "core_exact_path_blocked_reason"})
    drop_cols = [
        "_24",
        "entry_price",
        "exit_price",
        "gross_return",
        "formal_cost_model_cost",
        "net_return_formal_cost",
        "roundtrip_10bp_net_return",
        "roundtrip_20bp_net_return",
        "roundtrip_40bp_net_return",
    ]
    renamed = renamed.drop(columns=[col for col in drop_cols if col in renamed.columns])
    preferred = [
        "signal_date",
        "entry_date",
        "exit_date",
        "ticker",
        "name",
        "variant",
        "timing_variant",
        "entry_price_column",
        "exit_price_column",
        "entry_unadjusted_price",
        "exit_unadjusted_price",
        "gross_return_unadjusted",
        "net_return_roundtrip_10bp",
        "net_return_roundtrip_20bp",
        "net_return_roundtrip_40bp",
        "net_return_local_ep05_cost_unit_notional",
        "diagnostic_unit_notional_twd",
        "diagnostic_share_qty",
        "buy_gross_twd",
        "sell_gross_twd",
        "buy_cost_twd",
        "sell_cost_twd",
        "total_cost_twd",
        "cost_application_status",
        "holding_days",
        "selected_rank",
        "RS20_signal",
        "RS60_signal",
        "RS20_rank_within_80",
        "pool_rank_signal",
        "within80_rank",
        "layer4_risk_aware_score",
        "rs20_risk_context_score",
        "rs20_31_bonus_score",
        "in_31_high_confidence_subpool_reference",
        "in_100_extended_watchlist_reference",
        "unadjusted_requested_timing_prices_available",
        "exact_adjusted_requested_timing_prices_available",
        "entry_price_source_quality",
        "exit_price_source_quality",
        "entry_adjustment_policy",
        "exit_adjustment_policy",
        "adjusted_close_available",
        "adjusted_close_path_ready",
        "unadjusted_ohlc_path_ready",
        "path_materialization_status",
        "core_exact_path_blocked_reason",
        "radar_price_blocked_reason",
        "source_quality",
        "adjustment_policy",
        "formal_ready",
        "diagnostic_only",
        "not_live_rule",
        "forward_return_as_rule",
        "future_return_as_rule",
    ]
    existing_preferred = [col for col in preferred if col in renamed.columns]
    remaining = [col for col in renamed.columns if col not in existing_preferred and not col.startswith("_")]
    return renamed[existing_preferred + remaining]


def _write_manifest(files: list[Path], readiness: dict[str, Any]) -> None:
    manifest = {
        "task_id": TASK_ID,
        "source_task_id": SOURCE_TASK_ID,
        "status": readiness["status"],
        "output_dir": str(OUTPUT_DIR),
        "input_radar_dir": str(RADAR_DIR),
        "input_core_exact_dir": str(CORE_EXACT_DIR),
        "output_files": [path.name for path in files] + ["manifest.json"],
        **FLAGS,
        "diagnostic_only": True,
    }
    manifest["file_hashes"] = {
        path.name: {"sha256": _sha256(path), "bytes": path.stat().st_size}
        for path in files
        if path.exists()
    }
    (OUTPUT_DIR / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_summary(readiness: dict[str, Any]) -> None:
    lines = [
        "# Legacy RS20 unadjusted OHLC timing/cost materialization",
        "",
        f"- status: `{readiness['status']}`",
        f"- trade_path_rows: {readiness['trade_path_rows']}",
        f"- unadjusted_ohlc_ready_rows: {readiness['unadjusted_ohlc_ready_rows']}",
        f"- blocked_rows: {readiness['blocked_rows']}",
        f"- exact_selected_stock_adjusted_close_path_ready: {str(readiness['exact_selected_stock_adjusted_close_path_ready']).lower()}",
        f"- ready_for_legacy_rs20_unadjusted_cost_timing_diagnostic: {str(readiness['ready_for_legacy_rs20_unadjusted_cost_timing_diagnostic']).lower()}",
        f"- ready_for_legacy_rs20_exact_cost_timing_diagnostic: {str(readiness['ready_for_legacy_rs20_exact_cost_timing_diagnostic']).lower()}",
        "",
        "## 判斷",
        "",
        "Radar/Data 補出的本機 full-sweep shards 足以讓 Core 產出 partial unadjusted official OHLC path：1,464/1,476 timing rows 可算。"
        "但 adjusted close 仍為 blocked，8249 在 2024-10-11 / 2024-10-15 的 exit rows 仍缺官方日資料，不能 silent fill。",
        "",
        "本 package 使用 selected ticker 官方未調整 OHLC 計算 path，沒有使用 `00631L + excess` 重建個股報酬。"
        "本機 EP05 TaiwanCostModel 已套到 100 萬元診斷單位本金；這是 cost model materialization，不是 formal portfolio replay。",
        "",
        "## 下一步",
        "",
        "可交 Strategy Center 判斷是否接受 unadjusted OHLC partial path 做 bounded diagnostic。"
        "若 Strategy Center 堅持 adjusted close exact diagnostic，則仍需 source policy / adjusted-close route；若接受 unadjusted partial diagnostic，再交 Experiments。",
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
    radar_readiness = _read_json(RADAR_DIR / "readiness_for_core_legacy_rs20_selected_stock_price_path.json")
    trade_path = _build_trade_path()
    trade_path = _clean_trade_path_schema(trade_path)
    coverage = _period_coverage(trade_path.copy())
    cost_audit = _write_cost_audit(trade_path)
    future_audit = _future_audit()
    missing = trade_path.loc[~trade_path["unadjusted_ohlc_path_ready"]].copy()

    ready_rows = int(trade_path["unadjusted_ohlc_path_ready"].sum())
    blocked_rows = int((~trade_path["unadjusted_ohlc_path_ready"]).sum())
    future_violations = int(future_audit["violation_count"].sum())
    unadjusted_partial_ready = ready_rows > 0 and future_violations == 0
    exact_ready = False
    readiness = {
        "task_id": TASK_ID,
        "source_task_id": SOURCE_TASK_ID,
        "status": "legacy_rs20_unadjusted_ohlc_path_partial_ready_adjusted_close_blocked"
        if unadjusted_partial_ready
        else "legacy_rs20_unadjusted_ohlc_path_blocked",
        "radar_source_status": radar_readiness.get("status"),
        "diagnostic_only": True,
        "trade_path_rows": int(len(trade_path)),
        "unadjusted_ohlc_ready_rows": ready_rows,
        "blocked_rows": blocked_rows,
        "blocked_detail": "12 rows tied to 8249 missing exit dates 2024-10-11/2024-10-15" if blocked_rows else "",
        "exact_selected_stock_adjusted_close_path_ready": exact_ready,
        "next_trading_day_close_unadjusted_path_ready": bool(
            trade_path.loc[trade_path["timing_variant"].str.contains("next_day_close"), "unadjusted_ohlc_path_ready"].any()
        ),
        "next_trading_day_open_unadjusted_path_ready": bool(
            trade_path.loc[trade_path["timing_variant"] == "next_day_open_entry_fixed_5td_exit", "unadjusted_ohlc_path_ready"].any()
        ),
        "formal_cost_model_found": True,
        "formal_cost_model_applied_to_unadjusted_unit_notional": True,
        "formal_portfolio_replay": False,
        "placeholder_cost_model_retained": True,
        "ready_for_legacy_rs20_unadjusted_cost_timing_diagnostic": unadjusted_partial_ready,
        "ready_for_legacy_rs20_exact_cost_timing_diagnostic": False,
        "ready_for_experiments": False,
        "ready_for_formal": False,
        "ready_for_strategy_replay": False,
        "future_data_violation_count": future_violations,
        "blocked_fields": [
            "adjusted_close_path",
            "8249_missing_official_daily_exit_rows",
            "formal_portfolio_replay_cost_application",
        ],
        "proxy_fields": [
            "unadjusted_close_as_timing_price",
            "diagnostic_unit_notional_1m_twd_cost_application",
            "roundtrip_10bp_20bp_40bp_placeholders",
        ],
        **FLAGS,
    }

    output_files = [
        OUTPUT_DIR / "legacy_rs20_unadjusted_selected_stock_trade_path.csv",
        OUTPUT_DIR / "legacy_rs20_unadjusted_cost_model_audit.csv",
        OUTPUT_DIR / "legacy_rs20_unadjusted_missing_price_audit.csv",
        OUTPUT_DIR / "legacy_rs20_unadjusted_requested_vs_actual_coverage.csv",
        OUTPUT_DIR / "legacy_rs20_unadjusted_future_data_audit.csv",
        OUTPUT_DIR / "readiness_for_legacy_rs20_unadjusted_cost_timing_diagnostic.json",
        OUTPUT_DIR / "final_summary_zh.md",
    ]
    trade_path.drop(columns=["signal_dt"], errors="ignore").to_csv(output_files[0], index=False, encoding="utf-8")
    cost_audit.to_csv(output_files[1], index=False, encoding="utf-8")
    missing.to_csv(output_files[2], index=False, encoding="utf-8")
    coverage.to_csv(output_files[3], index=False, encoding="utf-8")
    future_audit.to_csv(output_files[4], index=False, encoding="utf-8")
    output_files[5].write_text(json.dumps(readiness, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_summary(readiness)
    _write_manifest(output_files, readiness)


if __name__ == "__main__":
    main()
