from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from backtest_lab.costs import cost_model_metadata


REPO_ROOT = Path(__file__).resolve().parents[2]
RADAR_ROOT = Path("C:/Users/zergv/Documents/Codex/2026-05-23/ai-stock-rotation-radar-https-docs")
FULL_PERIOD_DIR = REPO_ROOT / "outputs" / "vnext_full_period_regime_switch_benchmark_exception_path_20260708"
REGIME_SIGNAL_TABLE = (
    REPO_ROOT
    / "outputs"
    / "vnext_regime_switch_hybrid_route_market_fields_path_materialization_20260708"
    / "regime_switch_hybrid_route_signal_table.csv"
)
LEGACY_SIGNAL_TABLE = (
    REPO_ROOT
    / "outputs"
    / "vnext_legacy_rs20_operating_mode_runner_readiness_20260708"
    / "legacy_rs20_operating_mode_signal_table.csv"
)
BENCHMARK_FEATURES = REPO_ROOT / "outputs" / "vnext_dynamic_candidate_pool_data_materialization_20260706" / "benchmark_features.csv"
RADAR_FULL_SWEEP = RADAR_ROOT / "outputs" / "radar_dynamic_pool1_all_listed_liquid_universe_full_sweep_20260703"
RADAR_REGIME_SOURCE = RADAR_ROOT / "outputs" / "radar_vnext_regime_switch_route_selected_stock_ohlc_source_package_20260708"
OUTPUT_DIR = REPO_ROOT / "outputs" / "vnext_p2_2023_selected_stock_path_gap_closure_20260708"

TASK_ID = "TASK-BACKTEST-CORE-VNEXT-P2-2023-SELECTED-STOCK-PATH-GAP-CLOSURE-001"
RADAR_HANDOFF_TASK_ID = "TASK-RADAR-DATA-VNEXT-P2-2023-SELECTED-STOCK-OHLC-SOURCE-GAP-FILL-001"
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
TIMING_VARIANTS = [
    "next_day_close_entry_fixed_5td_exit",
    "same_week_close_to_next_rebalance_close_comparator",
    "next_day_open_entry_fixed_5td_exit",
]


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


def _benchmark_calendar() -> list[pd.Timestamp]:
    bench = pd.read_csv(BENCHMARK_FEATURES, dtype={"benchmark": str}, usecols=["trade_date", "benchmark"])
    bench = bench.loc[bench["benchmark"] == "0050"].copy()
    dates = pd.to_datetime(bench["trade_date"], errors="coerce").dropna().drop_duplicates().sort_values()
    return [pd.Timestamp(x) for x in dates]


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


def _missing_route_rows() -> pd.DataFrame:
    full_trace = pd.read_csv(FULL_PERIOD_DIR / "full_period_regime_switch_route_signal_trace.csv", low_memory=False)
    full_trace["route_date"] = pd.to_datetime(full_trace["route_date"], errors="coerce")
    regime_missing = full_trace.loc[
        (full_trace["route_date"] >= pd.Timestamp("2023-01-02"))
        & (full_trace["route_date"] <= pd.Timestamp("2023-12-29"))
        & (full_trace["exposure_flag"] == "stock")
        & (full_trace["path_source_status"] == "selected_stock_path_not_materialized_or_blocked")
    ].copy()
    regime_missing["source_family"] = "regime_switch_hybrid_route"
    regime_missing["selected_route_mode"] = regime_missing["selection_reason"]
    regime_missing["name"] = ""

    regime_source = pd.read_csv(REGIME_SIGNAL_TABLE, low_memory=False)
    regime_source["snapshot_date"] = pd.to_datetime(regime_source["snapshot_date"], errors="coerce")
    regime_source["ticker_norm"] = regime_source["ticker"].map(_ticker_str)
    source_cols = [
        "snapshot_date",
        "routing_variant",
        "ticker_norm",
        "name",
        "selected_route_mode",
        "market",
        "RS20",
        "RS60",
        "within80_rank",
    ]
    regime_missing["ticker_norm"] = regime_missing["ticker"].map(_ticker_str)
    regime_missing = regime_missing.merge(
        regime_source[source_cols].drop_duplicates(["snapshot_date", "routing_variant", "ticker_norm"]),
        left_on=["route_date", "route_variant", "ticker_norm"],
        right_on=["snapshot_date", "routing_variant", "ticker_norm"],
        how="left",
        suffixes=("", "_source"),
    )
    regime_missing["name"] = regime_missing["name_source"].fillna("")
    regime_missing["selected_route_mode"] = regime_missing["selected_route_mode_source"].fillna(regime_missing["selected_route_mode"])

    legacy = pd.read_csv(LEGACY_SIGNAL_TABLE, low_memory=False)
    legacy = legacy.loc[:, ~legacy.columns.duplicated()].copy()
    legacy["signal_date"] = pd.to_datetime(legacy["signal_date"], errors="coerce")
    legacy = legacy.loc[
        (legacy["signal_date"] >= pd.Timestamp("2023-01-02"))
        & (legacy["signal_date"] <= pd.Timestamp("2023-12-29"))
        & (
            legacy["signal_variant"].isin(
                [
                    "dynamic80_top3_rs20_risk_tiebreak_proxy",
                    "dynamic80_top1_rs20_proxy",
                    "dynamic80_top1_rs20_31_bonus_proxy",
                    "dynamic80_top1_rs20_7core_context_proxy",
                ]
            )
        )
    ].copy()
    legacy_missing = pd.DataFrame(
        {
            "route_date": legacy["signal_date"],
            "route_variant": legacy["signal_variant"],
            "selected_branch": "mega_rs20",
            "ticker_norm": legacy["ticker"].map(_ticker_str),
            "ticker": legacy["ticker"].map(_ticker_str),
            "selected_recommendation": legacy["ticker"].map(_ticker_str),
            "exposure_flag": "stock",
            "selection_reason": legacy["selection_rule_basis"].fillna("legacy_rs20_signal"),
            "ready_for_metric": False,
            "period_label": "P2",
            "source_family": "legacy_rs20_operating_mode",
            "path_source_status": "selected_stock_path_not_materialized_or_blocked",
            "selected_route_mode": legacy["signal_variant"],
            "name": legacy["name"],
            "market": legacy["market"],
            "RS20": legacy["RS20"],
            "RS60": legacy["RS60"],
            "within80_rank": legacy["within80_rank"],
        }
    )

    combined = pd.concat(
        [
            regime_missing[
                [
                    "route_date",
                    "route_variant",
                    "selected_branch",
                    "ticker_norm",
                    "ticker",
                    "selected_recommendation",
                    "exposure_flag",
                    "selection_reason",
                    "ready_for_metric",
                    "period_label",
                    "source_family",
                    "path_source_status",
                    "selected_route_mode",
                    "name",
                    "market",
                    "RS20",
                    "RS60",
                    "within80_rank",
                ]
            ],
            legacy_missing,
        ],
        ignore_index=True,
        sort=False,
    )
    combined["ticker"] = combined["ticker_norm"].fillna(combined["ticker"].map(_ticker_str))
    combined = combined.drop(columns=["ticker_norm"])
    combined = combined.drop_duplicates(["route_date", "route_variant", "ticker", "source_family"])
    return combined.sort_values(["route_date", "source_family", "route_variant", "ticker"])


def _gap_ledger() -> pd.DataFrame:
    base = _missing_route_rows()
    signal_dates = [pd.Timestamp(x) for x in sorted(base["route_date"].dropna().unique())]
    calendar = _benchmark_calendar()
    rows: list[dict[str, Any]] = []
    for r in base.itertuples(index=False):
        signal_date = pd.Timestamp(getattr(r, "route_date"))
        for timing in TIMING_VARIANTS:
            if timing == "same_week_close_to_next_rebalance_close_comparator":
                entry_date = signal_date
                exit_date = _next_signal_date(signal_dates, signal_date)
                missing_field = "entry_close;exit_close"
                requested_price_fields = "open;high;low;close;volume;traded_value"
            elif timing == "next_day_open_entry_fixed_5td_exit":
                entry_date = _next_trading_date(calendar, signal_date, 1)
                exit_date = _next_trading_date(calendar, signal_date, 6)
                missing_field = "entry_open;exit_close"
                requested_price_fields = "open;high;low;close;volume;traded_value"
            else:
                entry_date = _next_trading_date(calendar, signal_date, 1)
                exit_date = _next_trading_date(calendar, signal_date, 6)
                missing_field = "entry_close;exit_close"
                requested_price_fields = "open;high;low;close;volume;traded_value"
            rows.append(
                {
                    "signal_date": signal_date.date().isoformat(),
                    "entry_date": entry_date.date().isoformat() if entry_date is not None else "",
                    "exit_date": exit_date.date().isoformat() if exit_date is not None else "",
                    "ticker": getattr(r, "ticker"),
                    "name": getattr(r, "name", ""),
                    "market": getattr(r, "market", ""),
                    "route_variant": getattr(r, "route_variant"),
                    "source_family": getattr(r, "source_family"),
                    "selected_branch": getattr(r, "selected_branch"),
                    "selected_route_mode": getattr(r, "selected_route_mode"),
                    "timing_variant": timing,
                    "missing_field": missing_field,
                    "requested_price_fields": requested_price_fields,
                    "adjusted_close_required": False,
                    "adjusted_close_status": "blocked_not_required_for_unadjusted_gap_fill",
                    "source_attempted": "core_full_period_route_trace;core_2024plus_selected_path_packages;radar_full_sweep_manifest_without_local_shards",
                    "local_materialization_status": "blocked_no_local_2023_selected_ohlc_source_rows",
                    "next_owner": "Radar/Data",
                    "next_task_id": RADAR_HANDOFF_TASK_ID,
                    "no_silent_fill": True,
                    "diagnostic_only": True,
                    **FLAGS,
                }
            )
    return pd.DataFrame(rows)


def _empty_patch() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "signal_date",
            "entry_date",
            "exit_date",
            "ticker",
            "route_variant",
            "timing_variant",
            "entry_open",
            "entry_close",
            "exit_close",
            "gross_return_unadjusted",
            "net_return_local_ep05_cost_unit_notional",
            "path_ready",
            "blocked_reason",
        ]
    )


def _source_audit(gap: pd.DataFrame) -> pd.DataFrame:
    full_sweep_manifest = RADAR_FULL_SWEEP / "accepted_liquidity_shard_manifest.csv"
    full_sweep_shards = RADAR_FULL_SWEEP / "shards"
    source_rows = [
        {
            "source": "full_period_regime_switch_route_signal_trace",
            "path": str(FULL_PERIOD_DIR / "full_period_regime_switch_route_signal_trace.csv"),
            "status": "available",
            "use": "identified_2023_selected_stock_path_missing_rows",
            "rows_relevant": int(len(gap)),
        },
        {
            "source": "radar_full_sweep_manifest",
            "path": str(full_sweep_manifest),
            "status": "available" if full_sweep_manifest.exists() else "missing",
            "use": "confirms official daily OHLCV sweep candidate exists",
            "rows_relevant": None,
        },
        {
            "source": "radar_full_sweep_local_shards",
            "path": str(full_sweep_shards),
            "status": "missing_or_not_available_to_core" if not full_sweep_shards.exists() else "available",
            "use": "would be required for Core local patch materialization",
            "rows_relevant": None,
        },
        {
            "source": "radar_regime_selected_ohlc_source_package",
            "path": str(RADAR_REGIME_SOURCE),
            "status": "available_2024plus_only",
            "use": "prior selected-stock package starts 2024 and does not close 2023 P2 gap",
            "rows_relevant": None,
        },
    ]
    return pd.DataFrame(source_rows)


def _readiness(gap: pd.DataFrame, patch: pd.DataFrame, source_audit: pd.DataFrame) -> dict[str, Any]:
    base_rows = gap.drop_duplicates(["signal_date", "ticker", "route_variant", "source_family"])
    patched_rows = int(len(patch.loc[patch.get("path_ready", pd.Series(dtype=bool)).fillna(False).astype(bool)])) if not patch.empty else 0
    remaining = int(len(gap))
    return {
        "task_id": TASK_ID,
        "status": "p2_2023_selected_stock_path_gap_identified_core_local_source_blocked_handoff_radar_required",
        "p2_2023_missing_rows_before": int(len(base_rows)),
        "p2_2023_missing_timing_rows_before": int(len(gap)),
        "p2_2023_patched_rows": patched_rows,
        "p2_2023_remaining_blocked_rows": remaining,
        "selected_stock_unadjusted_ohlc_2023_ready_share": 0.0,
        "next_day_close_ready": False,
        "same_week_close_ready": False,
        "next_day_open_ready": False,
        "formal_cost_model_ready": True,
        "formal_cost_model_source": "backtest_lab.costs.TaiwanCostModel_available_after_price_source_fill",
        "adjusted_close_ready": False,
        "adjusted_close_status": "blocked_not_fabricated",
        "ready_for_full_period_regime_switch_rerun": False,
        "ready_for_experiments": False,
        "ready_for_formal": False,
        "ready_for_strategy_replay": False,
        "ready_for_radar_data_source_gap_fill": True,
        "radar_handoff_task_id": RADAR_HANDOFF_TASK_ID,
        "bounded_source_scope": {
            "unique_tickers": int(gap["ticker"].nunique()),
            "unique_signal_dates": int(gap["signal_date"].nunique()),
            "timing_variants": sorted(gap["timing_variant"].dropna().unique().tolist()),
            "route_variants": sorted(gap["route_variant"].dropna().unique().tolist()),
        },
        "future_data_violation_count": 0,
        "not_live_rule": True,
        "forward_returns_live_rule_usage": False,
        "boundary_flags": FLAGS,
        "cost_model_metadata": cost_model_metadata(),
        "source_audit_summary": json.loads(source_audit.where(pd.notna(source_audit), None).to_json(orient="records")),
    }


def _future_audit() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "audit_item": "future_return_as_rule",
                "violation_count": 0,
                "status": "pass",
                "note": "ledger uses PIT route signal dates and requested entry/exit dates only; no future returns used as rule",
            },
            {
                "audit_item": "00631L_plus_excess_reconstruction",
                "violation_count": 0,
                "status": "pass",
                "note": "explicitly prohibited; no selected stock return patch materialized from excess reconstruction",
            },
            {
                "audit_item": "silent_fill",
                "violation_count": 0,
                "status": "pass",
                "note": "Core did not fabricate missing OHLC; remaining rows handed to Radar/Data",
            },
        ]
    )


def _summary(readiness: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# P2 2023 selected-stock path gap closure",
            "",
            f"- task_id: `{TASK_ID}`",
            f"- status: `{readiness['status']}`",
            f"- p2_2023_missing_rows_before: {readiness['p2_2023_missing_rows_before']}",
            f"- p2_2023_missing_timing_rows_before: {readiness['p2_2023_missing_timing_rows_before']}",
            f"- p2_2023_patched_rows: {readiness['p2_2023_patched_rows']}",
            f"- p2_2023_remaining_blocked_rows: {readiness['p2_2023_remaining_blocked_rows']}",
            f"- unique_tickers: {readiness['bounded_source_scope']['unique_tickers']}",
            f"- unique_signal_dates: {readiness['bounded_source_scope']['unique_signal_dates']}",
            "",
            "## 判斷",
            "",
            "Core 已精準列出 2023 P2 selected-stock path 缺口，但目前本機沒有可直接 materialize 的 2023 selected-stock OHLC source rows。Radar full-sweep manifest 可見，但 local shard 實體不在目前 Core 可用路徑；既有 regime selected OHLC package 主要是 2024+，不能補 2023。",
            "",
            "因此本包不把 partial caveat 留在 Core，而是把 bounded selected-ticker-only ledger 交 Radar/Data 補 source acquisition。不得做 full-market mass download；只補 ledger 中的 ticker/date/timing 所需 official unadjusted OHLC。",
            "",
            "## 下一棒",
            "",
            f"直接交 Radar/Data：`{RADAR_HANDOFF_TASK_ID}`。",
            "",
            "完成後如果下一棒明確，請直接指派下一個 thread；如果下一棒不明確，請回報 Strategy Center 判斷。不要完成後停住不回報。",
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
            "full_period_package": str(FULL_PERIOD_DIR),
            "regime_signal_table": str(REGIME_SIGNAL_TABLE),
            "legacy_signal_table": str(LEGACY_SIGNAL_TABLE),
            "benchmark_features": str(BENCHMARK_FEATURES),
            "radar_full_sweep": str(RADAR_FULL_SWEEP),
            "radar_regime_source": str(RADAR_REGIME_SOURCE),
        },
        "readiness": readiness,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    gap = _gap_ledger()
    patch = _empty_patch()
    source_audit = _source_audit(gap)
    missing_after = gap.copy()
    future_audit = _future_audit()
    readiness = _readiness(gap, patch, source_audit)

    outputs = {
        "p2_2023_selected_stock_path_gap_ledger.csv": gap,
        "p2_2023_selected_stock_unadjusted_ohlc_path_patch.csv": patch,
        "p2_2023_selected_stock_path_source_audit.csv": source_audit,
        "p2_2023_selected_stock_missing_after_patch.csv": missing_after,
        "p2_2023_selected_stock_future_data_audit.csv": future_audit,
    }
    written: list[Path] = []
    for name, df in outputs.items():
        path = OUTPUT_DIR / name
        df.to_csv(path, index=False, encoding="utf-8-sig")
        written.append(path)

    readiness_path = OUTPUT_DIR / "readiness_for_p2_2023_selected_stock_path_gap_closure.json"
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
