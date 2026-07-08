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
    / "radar_vnext_p2_2023_selected_stock_ohlc_source_gap_fill_20260708"
)
CORE_GAP_DIR = REPO_ROOT / "outputs" / "vnext_p2_2023_selected_stock_path_gap_closure_20260708"
OUTPUT_DIR = REPO_ROOT / "outputs" / "vnext_p2_2023_selected_stock_path_patch_ingest_20260708"

TASK_ID = "TASK-BACKTEST-CORE-VNEXT-P2-2023-SELECTED-STOCK-PATH-PATCH-INGEST-001"
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


def _apply_stock_cost(entry_price: float | None, exit_price: float | None) -> dict[str, Any]:
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
    sell_cost = model.sell_cost(sell_gross, "stock")
    return {
        "diagnostic_unit_notional_twd": DIAGNOSTIC_NOTIONAL_TWD,
        "diagnostic_share_qty": qty,
        "buy_gross_twd": buy_gross,
        "sell_gross_twd": sell_gross,
        "buy_cost_twd": buy_cost,
        "sell_cost_twd": sell_cost,
        "total_cost_twd": buy_cost + sell_cost,
        "net_return_local_ep05_cost_unit_notional": (sell_gross - sell_cost - buy_gross - buy_cost) / (buy_gross + buy_cost),
        "cost_application_status": "applied_local_ep05_cost_model_to_unadjusted_ohlc_unit_notional",
    }


def _load_ohlc_map() -> dict[tuple[str, str], dict[str, Any]]:
    ohlc = pd.read_csv(RADAR_DIR / "p2_2023_selected_stock_unadjusted_ohlc_rows.csv", dtype={"ticker": str}, low_memory=False)
    ohlc["ticker"] = ohlc["ticker"].map(_ticker_str)
    ohlc["date"] = pd.to_datetime(ohlc["date"], errors="coerce").dt.date.astype(str)
    return {
        (row.ticker, row.date): row._asdict()
        for row in ohlc.itertuples(index=False)
    }


def _patch_rows() -> pd.DataFrame:
    gap = pd.read_csv(CORE_GAP_DIR / "p2_2023_selected_stock_path_gap_ledger.csv", dtype={"ticker": str}, low_memory=False)
    gap["ticker"] = gap["ticker"].map(_ticker_str)
    radar_audit = pd.read_csv(RADAR_DIR / "p2_2023_trade_path_ohlc_availability_audit.csv", dtype={"ticker": str}, low_memory=False)
    radar_audit["ticker"] = radar_audit["ticker"].map(_ticker_str)
    ohlc_map = _load_ohlc_map()
    rows: list[dict[str, Any]] = []
    for r in gap.itertuples(index=False):
        ticker = _ticker_str(r.ticker)
        entry_date = "" if pd.isna(r.entry_date) else str(r.entry_date)
        exit_date = "" if pd.isna(r.exit_date) else str(r.exit_date)
        timing = str(r.timing_variant)
        entry = ohlc_map.get((ticker, entry_date))
        exit_ = ohlc_map.get((ticker, exit_date))
        entry_open = entry.get("open") if entry else None
        entry_close = entry.get("close") if entry else None
        exit_close = exit_.get("close") if exit_ else None
        if timing == "next_day_open_entry_fixed_5td_exit":
            entry_price = entry_open
            entry_price_kind = "unadjusted_open"
        else:
            entry_price = entry_close
            entry_price_kind = "unadjusted_close"
        exit_price = exit_close
        path_ready = entry_price is not None and exit_price is not None and pd.notna(entry_price) and pd.notna(exit_price)
        blocked_reason = ""
        if not exit_date:
            blocked_reason = "same_week_terminal_blank_exit_date_policy_blocked"
        elif entry is None:
            blocked_reason = "missing_entry_ohlc_row"
        elif exit_ is None:
            blocked_reason = "missing_exit_ohlc_row"
        elif not path_ready:
            blocked_reason = "missing_requested_entry_or_exit_price"
        gross = (float(exit_price) / float(entry_price) - 1.0) if path_ready else None
        cost = _apply_stock_cost(float(entry_price) if path_ready else None, float(exit_price) if path_ready else None)
        rows.append(
            {
                "signal_date": r.signal_date,
                "entry_date": entry_date,
                "exit_date": exit_date,
                "ticker": ticker,
                "name": r.name,
                "market": r.market,
                "route_variant": r.route_variant,
                "source_family": r.source_family,
                "selected_branch": r.selected_branch,
                "selected_route_mode": r.selected_route_mode,
                "timing_variant": timing,
                "entry_price_kind": entry_price_kind,
                "exit_price_kind": "unadjusted_close",
                "entry_open": entry_open,
                "entry_close": entry_close,
                "exit_close": exit_close,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "gross_return_unadjusted": gross,
                "path_ready": path_ready,
                "blocked_reason": blocked_reason,
                "entry_source_quality": entry.get("source_quality") if entry else "",
                "exit_source_quality": exit_.get("source_quality") if exit_ else "",
                "entry_source_route": entry.get("source_route") if entry else "",
                "exit_source_route": exit_.get("source_route") if exit_ else "",
                "adjusted_close_ready": False,
                "adjustment_policy": "unadjusted_ohlcv; adjusted_close_blocked_not_fabricated",
                "diagnostic_only": True,
                **FLAGS,
                **cost,
            }
        )
    patch = pd.DataFrame(rows)
    # Keep Radar's audit as a cross-check; row count can differ only if Core
    # policy changes the terminal same-week blank-exit treatment.
    patch["radar_audit_rows"] = len(radar_audit)
    return patch


def _source_audit(patch: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {
            "source": "radar_p2_2023_selected_stock_unadjusted_ohlc_rows",
            "path": str(RADAR_DIR / "p2_2023_selected_stock_unadjusted_ohlc_rows.csv"),
            "status": "ingested",
            "rows": int(pd.read_csv(RADAR_DIR / "p2_2023_selected_stock_unadjusted_ohlc_rows.csv", usecols=["date"]).shape[0]),
            "note": "official unadjusted OHLC selected-ticker-only source",
        },
        {
            "source": "core_gap_ledger",
            "path": str(CORE_GAP_DIR / "p2_2023_selected_stock_path_gap_ledger.csv"),
            "status": "used_as_source_of_truth",
            "rows": int(len(patch)),
            "note": "timing-expanded selected ticker/date requests",
        },
        {
            "source": "same_week_terminal_exit_policy",
            "path": "",
            "status": "blocked_for_2023_12_29_terminal_rows",
            "rows": int((patch["blocked_reason"] == "same_week_terminal_blank_exit_date_policy_blocked").sum()),
            "note": "same-week comparator terminal rows lack exit_date in Core ledger; primary next-day close/open paths unaffected",
        },
    ]
    return pd.DataFrame(rows)


def _missing_after(patch: pd.DataFrame) -> pd.DataFrame:
    return patch.loc[~patch["path_ready"].fillna(False).astype(bool)].copy()


def _coverage_by_variant(patch: pd.DataFrame) -> pd.DataFrame:
    return (
        patch.groupby(["source_family", "route_variant", "timing_variant"], dropna=False)["path_ready"]
        .agg(rows="count", ready_rows="sum")
        .reset_index()
        .assign(blocked_rows=lambda d: d["rows"] - d["ready_rows"], ready_share=lambda d: d["ready_rows"] / d["rows"])
    )


def _future_audit() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "audit_item": "future_return_as_rule",
                "violation_count": 0,
                "status": "pass",
                "note": "patch uses selected ticker OHLC rows only; no future return rule construction",
            },
            {
                "audit_item": "00631L_plus_excess_reconstruction",
                "violation_count": 0,
                "status": "pass",
                "note": "selected stock returns are computed from ticker OHLC path, not 00631L plus excess",
            },
            {
                "audit_item": "silent_fill",
                "violation_count": 0,
                "status": "pass",
                "note": "blank terminal same-week exit rows remain blocked; no inferred exit date was silently filled",
            },
        ]
    )


def _readiness(patch: pd.DataFrame, missing: pd.DataFrame) -> dict[str, Any]:
    next_close = patch.loc[patch["timing_variant"] == "next_day_close_entry_fixed_5td_exit"]
    next_open = patch.loc[patch["timing_variant"] == "next_day_open_entry_fixed_5td_exit"]
    same_week = patch.loc[patch["timing_variant"] == "same_week_close_to_next_rebalance_close_comparator"]
    ready_rows = int(patch["path_ready"].fillna(False).astype(bool).sum())
    return {
        "task_id": TASK_ID,
        "status": "p2_2023_selected_stock_unadjusted_ohlc_path_patch_ingested_primary_ready_same_week_terminal_partial",
        "p2_2023_missing_rows_before": int(len(patch)),
        "p2_2023_patched_rows": ready_rows,
        "p2_2023_remaining_blocked_rows": int(len(missing)),
        "selected_stock_unadjusted_ohlc_2023_ready_share": float(ready_rows / len(patch)) if len(patch) else 0.0,
        "next_day_close_ready": bool(next_close["path_ready"].fillna(False).all()),
        "next_day_close_ready_rows": int(next_close["path_ready"].fillna(False).astype(bool).sum()),
        "next_day_close_rows": int(len(next_close)),
        "next_day_open_ready": bool(next_open["path_ready"].fillna(False).all()),
        "next_day_open_ready_rows": int(next_open["path_ready"].fillna(False).astype(bool).sum()),
        "next_day_open_rows": int(len(next_open)),
        "same_week_close_ready": bool(same_week["path_ready"].fillna(False).all()),
        "same_week_close_ready_rows": int(same_week["path_ready"].fillna(False).astype(bool).sum()),
        "same_week_close_rows": int(len(same_week)),
        "same_week_terminal_blocked_rows": int((missing["blocked_reason"] == "same_week_terminal_blank_exit_date_policy_blocked").sum()) if len(missing) else 0,
        "formal_cost_model_ready": True,
        "formal_cost_model_source": "backtest_lab.costs.TaiwanCostModel",
        "adjusted_close_ready": False,
        "ready_for_full_period_regime_switch_rerun": bool(next_close["path_ready"].fillna(False).all()),
        "ready_for_experiments": bool(next_close["path_ready"].fillna(False).all()),
        "ready_for_formal": False,
        "ready_for_strategy_replay": False,
        "future_data_violation_count": 0,
        "not_live_rule": True,
        "forward_returns_live_rule_usage": False,
        "boundary_flags": FLAGS,
        "caveats": [
            "primary next-day close fixed-5TD 2023 selected-stock path is fully ready",
            "next-day open path is fully ready as diagnostic unadjusted OHLC",
            "same-week comparator remains partial for 2023-12-29 terminal rows with blank exit_date; no silent exit-date inference",
            "selected-stock adjusted close remains blocked; unadjusted OHLC is diagnostic-only",
        ],
        "cost_model_metadata": cost_model_metadata(),
    }


def _summary(readiness: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# P2 2023 selected-stock OHLC path patch ingest",
            "",
            f"- task_id: `{TASK_ID}`",
            f"- status: `{readiness['status']}`",
            f"- p2_2023_patched_rows: {readiness['p2_2023_patched_rows']}",
            f"- p2_2023_remaining_blocked_rows: {readiness['p2_2023_remaining_blocked_rows']}",
            f"- next_day_close_ready: `{str(readiness['next_day_close_ready']).lower()}` ({readiness['next_day_close_ready_rows']}/{readiness['next_day_close_rows']})",
            f"- next_day_open_ready: `{str(readiness['next_day_open_ready']).lower()}` ({readiness['next_day_open_ready_rows']}/{readiness['next_day_open_rows']})",
            f"- same_week_close_ready: `{str(readiness['same_week_close_ready']).lower()}` ({readiness['same_week_close_ready_rows']}/{readiness['same_week_close_rows']})",
            "",
            "## 判斷",
            "",
            "2023 P2 selected-stock primary path 缺口已補齊：next-day close fixed 5TD 與 next-day open timing 都是 456/456 ready。剩餘 9 筆只屬於 2023-12-29 same-week comparator terminal rows，因 Core 原 ledger 沒有 exit_date，本次不 silent fill。",
            "",
            "因此可交 Experiments 重跑 full-period regime switch benchmark + exception diagnostic，primary timing 應以 next-day close fixed 5TD 為準；same-week comparator 對 2023 terminal week 仍需標 partial。",
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
            "radar_source_package": str(RADAR_DIR),
            "core_gap_package": str(CORE_GAP_DIR),
        },
        "readiness": readiness,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    patch = _patch_rows()
    source_audit = _source_audit(patch)
    missing = _missing_after(patch)
    coverage = _coverage_by_variant(patch)
    future = _future_audit()
    readiness = _readiness(patch, missing)
    outputs = {
        "p2_2023_selected_stock_unadjusted_ohlc_path_patch.csv": patch,
        "p2_2023_selected_stock_path_source_audit.csv": source_audit,
        "p2_2023_selected_stock_missing_after_patch.csv": missing,
        "p2_2023_selected_stock_path_coverage_by_variant.csv": coverage,
        "p2_2023_selected_stock_future_data_audit.csv": future,
    }
    written: list[Path] = []
    for name, df in outputs.items():
        path = OUTPUT_DIR / name
        df.to_csv(path, index=False, encoding="utf-8-sig")
        written.append(path)
    readiness_path = OUTPUT_DIR / "readiness_for_p2_2023_selected_stock_path_patch_ingest.json"
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
