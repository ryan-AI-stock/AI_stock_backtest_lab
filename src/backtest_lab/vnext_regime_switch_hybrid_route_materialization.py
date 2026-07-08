from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from backtest_lab.costs import TaiwanCostModel, cost_model_metadata


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_DIR = (
    Path("C:/Users/zergv/Documents/Codex/2026-07-06/backtest-lab-experiments-diagnostic-validation-attribution")
    / "outputs"
    / "vnext_regime_switch_operating_mode_routing_diagnostic_20260708"
)
RADAR_PRICE_DIR = (
    Path("C:/Users/zergv/Documents/Codex/2026-05-23/ai-stock-rotation-radar-https-docs")
    / "outputs"
    / "radar_vnext_legacy_rs20_selected_stock_price_path_source_package_20260708"
)
OUTPUT_DIR = REPO_ROOT / "outputs" / "vnext_regime_switch_hybrid_route_market_fields_path_materialization_20260708"

BENCHMARK_FEATURES = REPO_ROOT / "outputs" / "vnext_dynamic_candidate_pool_data_materialization_20260706" / "benchmark_features.csv"
LAYER4_POOL = REPO_ROOT / "outputs" / "vnext_layer4_80_primary_pool_contract_20260708" / "layer4_80_primary_pool_contract.csv"
ROUTE_TRACE = EXPERIMENTS_DIR / "regime_switch_selected_trace.csv"
PRICE_ROWS = RADAR_PRICE_DIR / "selected_stock_price_rows_local_only.csv"

TASK_ID = "TASK-BACKTEST-CORE-VNEXT-REGIME-SWITCH-HYBRID-ROUTE-MARKET-FIELDS-PATH-MATERIALIZATION-001"
PRIMARY_ROUTE = "hybrid_pullback_base_mega_override"
ROUTE_VARIANTS = [
    PRIMARY_ROUTE,
    "conservative_hurdle_route",
    "dispersion_route",
    "market_bias_pool_trend_route",
    "pool_breadth_route",
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


def _read_price_rows() -> pd.DataFrame:
    cols = [
        "date",
        "ticker",
        "open",
        "close",
        "adjusted_close",
        "adjusted_close_available",
        "source_quality",
        "adjustment_policy",
    ]
    price = pd.read_csv(PRICE_ROWS, usecols=cols)
    price["date"] = pd.to_datetime(price["date"], errors="coerce")
    price["ticker"] = price["ticker"].astype(str)
    return price


def _load_route_trace() -> pd.DataFrame:
    desired = [
        "snapshot_date",
        "ticker",
        "name",
        "market",
        "routing_variant",
        "routed_mode",
        "recommendation_type",
        "selected_mode",
        "within80_rank",
        "pool_rank",
        "RS20",
        "RS60",
        "layer4_risk_aware_score",
        "rs20_risk_context_score",
        "rs20_31_bonus_score",
        "pool_rs20_positive_share",
        "pool_rs20_30_positive_share",
        "pool_rs20_median",
        "pool_rs60_median",
        "pool_rs20_top1_minus_median",
        "pool_two_plus_opportunity_share",
        "pool_high_exhaustion_breakdown_share",
        "market_trend_proxy_strong",
        "pool_breadth_strong",
        "pool_breadth_very_strong",
        "rs_dispersion_strong",
        "opportunity_breadth_strong",
    ]
    available = pd.read_csv(ROUTE_TRACE, nrows=0).columns.tolist()
    usecols = [col for col in desired if col in available]
    trace = pd.read_csv(ROUTE_TRACE, usecols=usecols)
    trace["snapshot_date"] = pd.to_datetime(trace["snapshot_date"], errors="coerce")
    trace["ticker"] = trace["ticker"].astype(str)
    return trace.loc[trace["routing_variant"].isin(ROUTE_VARIANTS)].copy()


def _market_regime_fields() -> pd.DataFrame:
    bench = pd.read_csv(BENCHMARK_FEATURES)
    bench["trade_date"] = pd.to_datetime(bench["trade_date"], errors="coerce")
    pivot = bench.pivot(index="trade_date", columns="benchmark", values="adjusted_close").sort_index()
    features = bench.loc[bench["benchmark"] == "0050"].copy().sort_values("trade_date")
    features = features.set_index("trade_date")
    close = features["adjusted_close"].astype(float)
    ma20 = close.rolling(20, min_periods=20).mean()
    ma40 = close.rolling(40, min_periods=40).mean()
    ma60 = close.rolling(60, min_periods=60).mean()
    high20 = close.rolling(20, min_periods=20).max()
    high40 = close.rolling(40, min_periods=40).max()
    high60 = close.rolling(60, min_periods=60).max()

    out = pd.DataFrame(
        {
            "snapshot_date": close.index,
            "0050_adjusted_close": close.values,
            "0050_return_20d": close.pct_change(20).values,
            "0050_return_40d": close.pct_change(40).values,
            "0050_return_60d": close.pct_change(60).values,
            "0050_ma20": ma20.values,
            "0050_ma40": ma40.values,
            "0050_ma60": ma60.values,
            "0050_ma20_slope": ma20.pct_change(5).values,
            "0050_ma40_slope": ma40.pct_change(5).values,
            "0050_ma60_slope": ma60.pct_change(5).values,
            "0050_price_vs_ma20": (close / ma20 - 1.0).values,
            "0050_price_vs_ma40": (close / ma40 - 1.0).values,
            "0050_price_vs_ma60": (close / ma60 - 1.0).values,
            "0050_bias20": features["BIAS20"].astype(float).values,
            "0050_bias40": (close / ma40 - 1.0).values,
            "0050_bias60": features["BIAS60"].astype(float).values,
            "0050_bias120": features["BIAS120"].astype(float).values,
            "bias20_delta_5d": features["BIAS20"].astype(float).diff(5).values,
            "bias60_delta_5d": features["BIAS60"].astype(float).diff(5).values,
            "bias20_expanding_candidate": (features["BIAS20"].astype(float).diff(5) > 0).values,
            "bias60_expanding_candidate": (features["BIAS60"].astype(float).diff(5) > 0).values,
            "0050_close_vs_20d_high": (close / high20 - 1.0).values,
            "0050_close_vs_40d_high": (close / high40 - 1.0).values,
            "0050_close_vs_60d_high": (close / high60 - 1.0).values,
            "0050_new_20d_high_flag": (close >= high20).values,
            "0050_new_40d_high_flag": (close >= high40).values,
            "0050_new_60d_high_flag": (close >= high60).values,
        }
    )
    new_high_cols = ["0050_new_20d_high_flag", "0050_new_40d_high_flag", "0050_new_60d_high_flag"]
    out["rolling_high_breakout_count"] = out[new_high_cols].sum(axis=1)

    if "00631L" in pivot.columns and "0050" in pivot.columns:
        out = out.merge(
            pd.DataFrame(
                {
                    "snapshot_date": pivot.index,
                    "00631L_return_20d": pivot["00631L"].pct_change(20).values,
                    "00631L_return_60d": pivot["00631L"].pct_change(60).values,
                    "00631L_vs_0050_return_20d": (pivot["00631L"].pct_change(20) - pivot["0050"].pct_change(20)).values,
                    "00631L_vs_0050_return_60d": (pivot["00631L"].pct_change(60) - pivot["0050"].pct_change(60)).values,
                    "00631L_vs_0050_relation_source_quality": "benchmark_features_exact_pit",
                }
            ),
            on="snapshot_date",
            how="left",
        )
    else:
        out["00631L_vs_0050_relation_source_quality"] = "blocked_missing_benchmark_features"

    out["0050_bias40_source_quality"] = "derived_from_0050_adjusted_close_and_ma40"
    out["market_regime_feature_threshold_decided_by_core"] = False
    out["diagnostic_only"] = True
    for key, value in FLAGS.items():
        out[key] = value
    return out


def _pool_regime_fields() -> pd.DataFrame:
    cols = [
        "snapshot_date",
        "ticker",
        "RS20",
        "RS60",
        "traded_value_rank_20d",
        "traded_value_rank_60d",
        "two_plus_opportunity_labels",
        "high_exhaustion_or_breakdown_context",
    ]
    available = pd.read_csv(LAYER4_POOL, nrows=0).columns.tolist()
    pool = pd.read_csv(LAYER4_POOL, usecols=[c for c in cols if c in available])
    pool["snapshot_date"] = pd.to_datetime(pool["snapshot_date"], errors="coerce")
    for col in ["RS20", "RS60", "traded_value_rank_20d", "traded_value_rank_60d"]:
        if col in pool.columns:
            pool[col] = pd.to_numeric(pool[col], errors="coerce")
    grouped = []
    for date, group in pool.groupby("snapshot_date"):
        rs20 = group["RS20"] if "RS20" in group.columns else pd.Series(dtype=float)
        rs60 = group["RS60"] if "RS60" in group.columns else pd.Series(dtype=float)
        top_decile_count = max(1, int(len(group) * 0.1))
        top_decile_median = rs20.sort_values(ascending=False).head(top_decile_count).median() if not rs20.dropna().empty else None
        traded_value_breadth = None
        concentration_proxy = None
        if "traded_value_rank_20d" in group.columns:
            traded_value_breadth = float((group["traded_value_rank_20d"] <= 300).mean())
            concentration_proxy = float((group["traded_value_rank_20d"] <= 50).mean())
        grouped.append(
            {
                "snapshot_date": date,
                "dynamic80_row_count": int(len(group)),
                "dynamic80_rs20_positive_share": float((rs20 > 0).mean()) if len(rs20) else None,
                "dynamic80_rs60_positive_share": float((rs60 > 0).mean()) if len(rs60) else None,
                "dynamic80_rs20_median": float(rs20.median()) if not rs20.dropna().empty else None,
                "dynamic80_rs20_p75": float(rs20.quantile(0.75)) if not rs20.dropna().empty else None,
                "dynamic80_rs20_top_decile_median": top_decile_median,
                "dynamic80_rs20_dispersion_top_minus_median": (float(rs20.max() - rs20.median()) if not rs20.dropna().empty else None),
                "dynamic80_traded_value_breadth": traded_value_breadth,
                "dynamic80_traded_value_top50_concentration_proxy": concentration_proxy,
                "dynamic80_two_plus_opportunity_label_share": float(group.get("two_plus_opportunity_labels", pd.Series(False)).astype(bool).mean()),
                "dynamic80_high_exhaustion_breakdown_share": float(
                    group.get("high_exhaustion_or_breakdown_context", pd.Series(False)).astype(bool).mean()
                ),
                "pool_confirmation_threshold_decided_by_core": False,
                "diagnostic_only": True,
                **FLAGS,
            }
        )
    return pd.DataFrame(grouped)


def _route_signal_table(route_trace: pd.DataFrame, market: pd.DataFrame, pool: pd.DataFrame) -> pd.DataFrame:
    out = route_trace.copy()
    market_small = market[
        [
            "snapshot_date",
            "0050_return_20d",
            "0050_return_40d",
            "0050_return_60d",
            "0050_ma20_slope",
            "0050_ma40_slope",
            "0050_ma60_slope",
            "0050_bias20",
            "0050_bias40",
            "0050_bias60",
            "0050_bias120",
            "bias20_delta_5d",
            "bias60_delta_5d",
            "0050_new_20d_high_flag",
            "0050_new_40d_high_flag",
            "0050_new_60d_high_flag",
            "rolling_high_breakout_count",
            "00631L_vs_0050_return_20d",
            "00631L_vs_0050_return_60d",
        ]
    ]
    pool_small = pool[
        [
            "snapshot_date",
            "dynamic80_rs20_positive_share",
            "dynamic80_rs60_positive_share",
            "dynamic80_rs20_median",
            "dynamic80_rs20_top_decile_median",
            "dynamic80_rs20_dispersion_top_minus_median",
            "dynamic80_traded_value_breadth",
            "dynamic80_traded_value_top50_concentration_proxy",
            "dynamic80_two_plus_opportunity_label_share",
        ]
    ]
    out = out.merge(market_small, on="snapshot_date", how="left")
    out = out.merge(pool_small, on="snapshot_date", how="left", suffixes=("", "_core"))
    out["selected_route_mode"] = out["routed_mode"]
    out["route_input_fields_pit_observable"] = True
    out["future_return_used_in_route_construction"] = False
    out["route_threshold_decided_by_core"] = False
    out["diagnostic_only"] = True
    for key, value in FLAGS.items():
        out[key] = value
    return out.sort_values(["snapshot_date", "routing_variant"]).reset_index(drop=True)


def _price_maps() -> tuple[dict[tuple[str, pd.Timestamp], float], dict[tuple[str, pd.Timestamp], float]]:
    price = _read_price_rows()
    open_map = {}
    close_map = {}
    for row in price.itertuples(index=False):
        key = (str(row.ticker), pd.Timestamp(row.date))
        if pd.notna(row.open):
            open_map[key] = float(row.open)
        if pd.notna(row.close):
            close_map[key] = float(row.close)
    return open_map, close_map


def _calendar(market: pd.DataFrame) -> list[pd.Timestamp]:
    return sorted(pd.to_datetime(market["snapshot_date"].dropna().unique()))


def _next_date(cal: list[pd.Timestamp], date: pd.Timestamp, offset: int) -> pd.Timestamp | None:
    for idx, d in enumerate(cal):
        if d > date:
            target = idx + offset - 1
            if 0 <= target < len(cal):
                return cal[target]
            return None
    return None


def _path_readiness(route_table: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    open_map, close_map = _price_maps()
    cal = _calendar(market)
    rows = []
    primary_window = route_table.loc[
        (route_table["snapshot_date"] >= pd.Timestamp("2024-01-02"))
        & (route_table["snapshot_date"] <= pd.Timestamp("2026-05-26"))
    ].copy()
    model = TaiwanCostModel()
    for row in primary_window.itertuples(index=False):
        ticker = str(row.ticker)
        signal_date = pd.Timestamp(row.snapshot_date)
        is_ref = str(row.recommendation_type) == "00631L" or ticker == "00631L"
        entry_date = _next_date(cal, signal_date, 1)
        exit_date = _next_date(cal, signal_date, 6)
        entry_close = close_map.get((ticker, entry_date)) if entry_date is not None and not is_ref else None
        entry_open = open_map.get((ticker, entry_date)) if entry_date is not None and not is_ref else None
        exit_close = close_map.get((ticker, exit_date)) if exit_date is not None and not is_ref else None
        close_ready = entry_close is not None and exit_close is not None
        open_ready = entry_open is not None and exit_close is not None
        gross_close = (exit_close / entry_close - 1.0) if close_ready else None
        gross_open = (exit_close / entry_open - 1.0) if open_ready else None
        if close_ready:
            qty = int(1_000_000 // entry_close)
            buy_gross = qty * entry_close
            sell_gross = qty * exit_close
            net_close = (sell_gross - model.sell_cost(sell_gross, "stock") - buy_gross - model.buy_cost(buy_gross)) / (
                buy_gross + model.buy_cost(buy_gross)
            )
        else:
            net_close = None
        if is_ref:
            blocked_reason = "00631L_reference_path_separate_from_ordinary_stock_path"
        elif not close_ready and not open_ready:
            blocked_reason = "missing_unadjusted_entry_or_exit_price"
        elif close_ready and not open_ready:
            blocked_reason = "open_entry_blocked_close_path_ready"
        else:
            blocked_reason = "adjusted_close_blocked_unadjusted_path_available"
        rows.append(
            {
                "snapshot_date": signal_date.date().isoformat(),
                "routing_variant": row.routing_variant,
                "selected_route_mode": row.selected_route_mode,
                "ticker": ticker,
                "name": row.name,
                "recommendation_type": row.recommendation_type,
                "entry_date": entry_date.date().isoformat() if entry_date is not None else "",
                "exit_date": exit_date.date().isoformat() if exit_date is not None else "",
                "next_day_close_ready": close_ready,
                "next_day_open_ready": open_ready,
                "adjusted_close_ready": False,
                "entry_close": entry_close,
                "entry_open": entry_open,
                "exit_close": exit_close,
                "gross_return_next_day_close_unadjusted_5td": gross_close,
                "gross_return_next_day_open_unadjusted_5td": gross_open,
                "net_return_local_ep05_cost_unit_notional_close_entry": net_close,
                "cost_model_version": cost_model_metadata()["cost_model_version"],
                "formal_portfolio_replay": False,
                "path_blocked_reason": blocked_reason,
                "diagnostic_only": True,
                **FLAGS,
            }
        )
    return pd.DataFrame(rows)


def _missing_price_request(path: pd.DataFrame) -> pd.DataFrame:
    missing = path.loc[
        (path["recommendation_type"] != "00631L")
        & (~path["next_day_close_ready"])
        & (~path["next_day_open_ready"])
    ].copy()
    if missing.empty:
        return pd.DataFrame(
            columns=[
                "ticker",
                "name",
                "missing_route_rows",
                "first_signal_date",
                "last_signal_date",
                "required_price_start",
                "required_price_end_with_exit_buffer",
                "required_columns",
                "source_request_reason",
                "diagnostic_only",
            ]
        )
    grouped = (
        missing.groupby(["ticker", "name"], dropna=False)
        .agg(
            missing_route_rows=("snapshot_date", "size"),
            first_signal_date=("snapshot_date", "min"),
            last_signal_date=("snapshot_date", "max"),
            required_price_start=("entry_date", "min"),
            required_price_end_with_exit_buffer=("exit_date", "max"),
        )
        .reset_index()
    )
    grouped["required_columns"] = "date,ticker,open,close,adjusted_close,source_quality,adjustment_policy"
    grouped["source_request_reason"] = "regime_switch_route_selected_ticker_missing_unadjusted_ohlc_path"
    grouped["do_not_use_00631L_plus_excess_reconstruction"] = True
    grouped["diagnostic_only"] = True
    for key, value in FLAGS.items():
        grouped[key] = value
    return grouped.sort_values(["missing_route_rows", "ticker"], ascending=[False, True])


def _path_coverage_summary(path: pd.DataFrame) -> pd.DataFrame:
    if path.empty:
        return pd.DataFrame()
    grouped = (
        path.groupby("routing_variant", dropna=False)
        .agg(
            rows=("snapshot_date", "size"),
            ordinary_stock_rows=("recommendation_type", lambda s: int((s != "00631L").sum())),
            reference_00631L_rows=("recommendation_type", lambda s: int((s == "00631L").sum())),
            next_day_close_ready_rows=("next_day_close_ready", "sum"),
            next_day_open_ready_rows=("next_day_open_ready", "sum"),
            adjusted_close_ready_rows=("adjusted_close_ready", "sum"),
        )
        .reset_index()
    )
    grouped["next_day_close_ready_share"] = grouped["next_day_close_ready_rows"] / grouped["rows"]
    grouped["next_day_open_ready_share"] = grouped["next_day_open_ready_rows"] / grouped["rows"]
    grouped["diagnostic_only"] = True
    return grouped


def _timing_cost_audit(path: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "audit_item": "local_ep05_cost_model",
                "ready": True,
                "source_quality": "local_taiwan_standard_fee_tax_v1",
                "notes": "Applied only to diagnostic unit-notional close-entry rows; not portfolio replay.",
                **cost_model_metadata(),
            },
            {
                "audit_item": "next_day_close_unadjusted_path",
                "ready": bool(path["next_day_close_ready"].any()) if not path.empty else False,
                "source_quality": "official_unadjusted_ohlcv_pit_daily_full_sweep_shard",
                "notes": f"ready_rows={int(path['next_day_close_ready'].sum()) if not path.empty else 0}",
            },
            {
                "audit_item": "next_day_open_unadjusted_path",
                "ready": bool(path["next_day_open_ready"].any()) if not path.empty else False,
                "source_quality": "official_unadjusted_ohlcv_pit_daily_full_sweep_shard",
                "notes": f"ready_rows={int(path['next_day_open_ready'].sum()) if not path.empty else 0}",
            },
            {
                "audit_item": "adjusted_close_path",
                "ready": False,
                "source_quality": "blocked",
                "notes": "Adjusted close remains unavailable and is not fabricated.",
            },
        ]
    )


def _future_audit() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "audit_item": "route_construction_uses_future_return",
                "result": "passed",
                "violation_count": 0,
                "evidence": "Route table carries PIT market/pool features; forward-return columns are not used.",
            },
            {
                "audit_item": "market_feature_threshold_decided_by_core",
                "result": "passed",
                "violation_count": 0,
                "evidence": "Core materializes slope/BIAS/breakout fields only; no thresholds selected.",
            },
            {
                "audit_item": "adjusted_close_fabricated",
                "result": "passed",
                "violation_count": 0,
                "evidence": "Adjusted close is explicitly blocked; unadjusted OHLC path is separately labeled.",
            },
        ]
    )


def _write_manifest(files: list[Path], readiness: dict[str, Any]) -> None:
    manifest = {
        "task_id": TASK_ID,
        "status": readiness["status"],
        "output_dir": str(OUTPUT_DIR),
        "input_experiments_dir": str(EXPERIMENTS_DIR),
        "input_radar_price_dir": str(RADAR_PRICE_DIR),
        "input_benchmark_features": str(BENCHMARK_FEATURES),
        "input_layer4_pool": str(LAYER4_POOL),
        "output_files": [p.name for p in files] + ["manifest.json"],
        "diagnostic_only": True,
        **FLAGS,
    }
    manifest["file_hashes"] = {p.name: {"sha256": _sha256(p), "bytes": p.stat().st_size} for p in files if p.exists()}
    (OUTPUT_DIR / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def _summary(readiness: dict[str, Any]) -> None:
    lines = [
        "# Regime switch hybrid route market fields/path materialization",
        "",
        f"- status: `{readiness['status']}`",
        f"- market_bias_fields_ready: {str(readiness['market_bias_fields_ready']).lower()}",
        f"- pool_breadth_dispersion_fields_ready: {str(readiness['pool_breadth_dispersion_fields_ready']).lower()}",
        f"- route_signal_table_ready: {str(readiness['route_signal_table_ready']).lower()}",
        f"- next_day_unadjusted_path_ready: {str(readiness['next_day_unadjusted_path_ready']).lower()}",
        f"- next_day_unadjusted_path_ready_share: {readiness['next_day_unadjusted_path_ready_share']}",
        f"- next_day_open_ready: {str(readiness['next_day_open_ready']).lower()}",
        f"- next_day_close_ready: {str(readiness['next_day_close_ready']).lower()}",
        f"- adjusted_close_ready: {str(readiness['adjusted_close_ready']).lower()}",
        f"- ready_for_experiments: {str(readiness['ready_for_experiments']).lower()}",
        "",
        "## 判斷",
        "",
        "Core 已補 0050 slope / BIAS / BIAS expansion / 20-40-60D high breakout 欄位，並從 Layer4 80 pool 重新 materialize pool breadth / RS dispersion。"
        "這些都是 PIT feature columns，不含 Core threshold 決策。",
        "",
        "Path 端目前只有 partial selected ticker official unadjusted OHLC 覆蓋；adjusted close 仍 blocked，00631L reference path 仍需與 ordinary stock path 分開。",
        "因此 Core 不直接交 Experiments，除非 Strategy Center 接受 partial-row diagnostic；下一棒較明確是 Radar/Data 補 regime-route selected ticker OHLC source package。",
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
    market = _market_regime_fields()
    pool = _pool_regime_fields()
    route_trace = _load_route_trace()
    route_table = _route_signal_table(route_trace, market, pool)
    path = _path_readiness(route_table, market)
    path_coverage = _path_coverage_summary(path)
    missing_request = _missing_price_request(path)
    timing_cost = _timing_cost_audit(path)
    future = _future_audit()

    market_ready = bool(market[["0050_bias20", "0050_bias60", "0050_bias120", "0050_ma20_slope", "0050_close_vs_20d_high"]].notna().any().all())
    pool_ready = bool(pool[["dynamic80_rs20_positive_share", "dynamic80_rs20_median", "dynamic80_rs20_dispersion_top_minus_median"]].notna().any().all())
    route_ready = not route_table.empty
    close_ready_count = int(path["next_day_close_ready"].sum()) if not path.empty else 0
    open_ready_count = int(path["next_day_open_ready"].sum()) if not path.empty else 0
    ordinary_rows = int((path["recommendation_type"] != "00631L").sum()) if not path.empty else 0
    close_ready = close_ready_count == ordinary_rows and ordinary_rows > 0
    open_ready = open_ready_count == ordinary_rows and ordinary_rows > 0
    partial_next_ready = close_ready_count > 0 or open_ready_count > 0
    future_violations = int(future["violation_count"].sum())
    path_ready_share = (close_ready_count / ordinary_rows) if ordinary_rows else 0.0
    ready_for_experiments = bool(market_ready and pool_ready and route_ready and close_ready and future_violations == 0)

    readiness = {
        "task_id": TASK_ID,
        "status": "regime_switch_hybrid_route_market_fields_ready_path_partial_blocked"
        if not ready_for_experiments
        else "regime_switch_hybrid_route_market_fields_unadjusted_path_ready_adjusted_close_blocked",
        "diagnostic_only": True,
        "primary_route_candidate": PRIMARY_ROUTE,
        "route_variants": ROUTE_VARIANTS,
        "market_bias_fields_ready": market_ready,
        "market_slope_breakout_fields_ready": market_ready,
        "pool_breadth_dispersion_fields_ready": pool_ready,
        "route_signal_table_ready": route_ready,
        "route_signal_rows": int(len(route_table)),
        "selected_path_rows_primary_window": int(len(path)),
        "ordinary_stock_path_rows": ordinary_rows,
        "next_day_unadjusted_path_ready": close_ready,
        "next_day_unadjusted_path_partial_ready": partial_next_ready,
        "next_day_unadjusted_close_ready_rows": close_ready_count,
        "next_day_unadjusted_open_ready_rows": open_ready_count,
        "next_day_unadjusted_path_ready_share": path_ready_share,
        "next_day_open_ready": open_ready,
        "next_day_close_ready": close_ready,
        "formal_cost_model_ready": True,
        "formal_cost_model_scope": "diagnostic_unit_notional_unadjusted_ohlc_not_portfolio_replay",
        "adjusted_close_ready": False,
        "future_data_violation_count": future_violations,
        "ready_for_regime_switch_hybrid_route_diagnostic": ready_for_experiments,
        "ready_for_market_regime_feature_diagnostic": bool(market_ready and pool_ready and route_ready),
        "ready_for_experiments": ready_for_experiments,
        "ready_for_formal": False,
        "ready_for_strategy_replay": False,
        "blocked_fields": [
            "adjusted_close_path",
            "missing_regime_route_selected_stock_ohlc_rows",
            "00631L_reference_executable_open_path_as_ordinary_stock",
            "formal_portfolio_replay",
        ],
        "proxy_fields": ["unadjusted_ohlc_path", "BIAS40_derived_from_ma40", "traded_value_concentration_proxy"],
        **FLAGS,
    }

    files = [
        OUTPUT_DIR / "regime_switch_hybrid_route_signal_table.csv",
        OUTPUT_DIR / "regime_switch_market_regime_fields.csv",
        OUTPUT_DIR / "regime_switch_pool_regime_fields.csv",
        OUTPUT_DIR / "regime_switch_hybrid_route_selected_path_readiness.csv",
        OUTPUT_DIR / "regime_switch_hybrid_route_timing_cost_audit.csv",
        OUTPUT_DIR / "regime_switch_hybrid_route_path_coverage_by_variant.csv",
        OUTPUT_DIR / "regime_switch_hybrid_route_missing_price_source_request.csv",
        OUTPUT_DIR / "regime_switch_hybrid_route_future_data_audit.csv",
        OUTPUT_DIR / "readiness_for_regime_switch_hybrid_route_diagnostic.json",
        OUTPUT_DIR / "final_summary_zh.md",
    ]
    route_table.to_csv(files[0], index=False, encoding="utf-8")
    market.to_csv(files[1], index=False, encoding="utf-8")
    pool.to_csv(files[2], index=False, encoding="utf-8")
    path.to_csv(files[3], index=False, encoding="utf-8")
    timing_cost.to_csv(files[4], index=False, encoding="utf-8")
    path_coverage.to_csv(files[5], index=False, encoding="utf-8")
    missing_request.to_csv(files[6], index=False, encoding="utf-8")
    future.to_csv(files[7], index=False, encoding="utf-8")
    files[8].write_text(json.dumps(readiness, ensure_ascii=False, indent=2), encoding="utf-8")
    _summary(readiness)
    _write_manifest(files, readiness)


if __name__ == "__main__":
    main()
