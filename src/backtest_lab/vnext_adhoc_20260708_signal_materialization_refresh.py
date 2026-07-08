from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
RADAR_ROOT = Path("C:/Users/zergv/Documents/Codex/2026-05-23/ai-stock-rotation-radar-https-docs/outputs")
DEFAULT_RADAR_WINDOW = RADAR_ROOT / "radar_vnext_adhoc_20260708_eod_historical_window_source_fill_20260708"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "vnext_adhoc_20260708_eod_signal_materialization_refresh_20260708"

DAILY_MARKET = REPO_ROOT / "outputs" / "vnext_dynamic_candidate_pool_data_materialization_20260706" / "daily_market_features.csv"
BENCHMARK_FEATURES = REPO_ROOT / "outputs" / "vnext_dynamic_candidate_pool_data_materialization_20260706" / "benchmark_features.csv"
LAYER4_PRIMARY80 = REPO_ROOT / "outputs" / "vnext_layer4_80_primary_pool_contract_20260708" / "layer4_80_primary_pool_contract.csv"
EXACT_TRIGGER = REPO_ROOT / "outputs" / "vnext_full_period_exact_consensus_trigger_contract_20260708" / "full_period_exact_consensus_trigger_contract.csv"
ROUTE_SUPPORT_MAX1 = REPO_ROOT / "outputs" / "vnext_route_support_max1_full_period_same_basis_contract_20260708" / "route_support_max1_full_period_same_basis_modelization_contract.csv"

TASK_ID = "TASK-BACKTEST-CORE-VNEXT-ADHOC-20260708-EOD-SIGNAL-MATERIALIZATION-REFRESH-001"
REQUESTED_DATE = "2026-07-08"
FLAGS = {
    "formal_model_changed": False,
    "trade_decision_changed": False,
    "active_in_trade_decision": False,
    "report_changed": True,
    "portfolio_replay_executed": False,
    "ready_for_strategy_replay": False,
    "ready_for_formal": False,
    "not_live_rule": True,
    "forward_returns_live_rule_usage": False,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize bounded vNext 2026-07-08 EOD signal refresh.")
    parser.add_argument("--radar-window-dir", default=str(DEFAULT_RADAR_WINDOW))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--as-of-date", default=REQUESTED_DATE)
    args = parser.parse_args()
    build_package(
        radar_window_dir=Path(args.radar_window_dir),
        output_dir=Path(args.output_dir),
        requested_date=args.as_of_date,
    )


def build_package(*, radar_window_dir: Path, output_dir: Path, requested_date: str = REQUESTED_DATE) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    source_rows = radar_window_dir / "vnext_adhoc_20260708_historical_window_scoped_common_stock_etf_rows.csv"
    window_readiness = radar_window_dir / "readiness_for_core_vnext_adhoc_20260708_historical_window_source_fill.json"

    market = load_market_with_patch(source_rows)
    feature = compute_current_features(market, requested_date)
    benchmarks = compute_benchmark_features(source_rows, requested_date)
    layer0 = build_layer0_compact(feature)
    rs20_top3 = build_rs20_top3(feature, benchmarks)
    market_regime = build_market_regime_partial(benchmarks)
    snapshot = build_signal_snapshot(requested_date, market_regime, rs20_top3)
    blocked = build_blocked_proxy_audit(market_regime)
    coverage = build_coverage_audit(source_rows, window_readiness, feature, layer0, rs20_top3, market_regime)
    future = pd.DataFrame(
        [
            {
                "audit_item": "future_return_as_rule",
                "used": False,
                "future_data_violation_count": 0,
            },
            {
                "audit_item": "retrieval_time_as_market_date",
                "used": False,
                "future_data_violation_count": 0,
            },
        ]
    )

    layer0.to_csv(output_dir / "vnext_adhoc_20260708_layer0_compact_active_universe.csv", index=False, encoding="utf-8-sig")
    rs20_top3.to_csv(output_dir / "vnext_adhoc_20260708_rs20_top3_reference.csv", index=False, encoding="utf-8-sig")
    market_regime.to_csv(output_dir / "vnext_adhoc_20260708_0050_market_regime_partial.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([snapshot]).to_csv(output_dir / "vnext_adhoc_20260708_signal_snapshot_zh.csv", index=False, encoding="utf-8-sig")
    blocked.to_csv(output_dir / "vnext_adhoc_20260708_signal_blocked_proxy_audit.csv", index=False, encoding="utf-8-sig")
    coverage.to_csv(output_dir / "vnext_adhoc_20260708_signal_coverage_audit.csv", index=False, encoding="utf-8-sig")
    future.to_csv(output_dir / "vnext_adhoc_20260708_signal_future_data_audit.csv", index=False, encoding="utf-8-sig")

    readiness = {
        "task": TASK_ID,
        "status": "partial_materialized_rs20_layer0_ready_c2_exact_and_route_support_blocked",
        "requested_date": requested_date,
        "actual_eod_source_date": requested_date,
        "layer0_compact_active_universe_ready": True,
        "rs20_top3_reference_ready": True,
        "c2_market_health_exact_ready": False,
        "exact_consensus_trigger_ready": False,
        "route_support_max1_selected_signal_ready": False,
        "ready_for_vnext_daily_report_selected_signal_publish": False,
        "ready_for_live_publish": False,
        "ready_for_experiments": False,
        "ready_for_formal": False,
        "ready_for_strategy_replay": False,
        "future_data_violation_count": 0,
        "blocking_summary": "2026-07-08 official EOD/window source is ready, but Core lacks exact 0050 ETF 2026-07-01/2026-07-02 rows for MA60 and lacks 2026-07-08 exact consensus trigger / Layer4 primary80 / route_support state-machine materialization.",
        **FLAGS,
    }
    write_json(output_dir / "readiness_for_vnext_adhoc_20260708_signal_materialization_refresh.json", readiness)
    write_json(output_dir / "vnext_adhoc_20260708_signal_snapshot.json", snapshot)
    write_summary(output_dir / "final_summary_zh.md", snapshot, readiness)
    write_json(
        output_dir / "manifest.json",
        {
            "task": TASK_ID,
            "output_dir": str(output_dir),
            "source_radar_window_dir": str(radar_window_dir),
            "artifacts": [
                "vnext_adhoc_20260708_signal_snapshot.json",
                "vnext_adhoc_20260708_signal_snapshot_zh.csv",
                "vnext_adhoc_20260708_layer0_compact_active_universe.csv",
                "vnext_adhoc_20260708_rs20_top3_reference.csv",
                "vnext_adhoc_20260708_0050_market_regime_partial.csv",
                "vnext_adhoc_20260708_signal_blocked_proxy_audit.csv",
                "vnext_adhoc_20260708_signal_coverage_audit.csv",
                "vnext_adhoc_20260708_signal_future_data_audit.csv",
                "readiness_for_vnext_adhoc_20260708_signal_materialization_refresh.json",
                "final_summary_zh.md",
            ],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "report_changed": True,
            "formal_model_changed": False,
            "trade_decision_changed": False,
        },
    )
    return readiness


def load_market_with_patch(source_rows: Path) -> pd.DataFrame:
    usecols = ["trade_date", "ticker", "name", "market", "adjusted_close", "volume", "traded_value"]
    base = pd.read_csv(DAILY_MARKET, usecols=usecols, dtype={"ticker": str})
    base = base.rename(columns={"trade_date": "date", "adjusted_close": "close", "traded_value": "turnover_value"})
    patch = pd.read_csv(
        source_rows,
        usecols=["date", "ticker", "name", "market", "close", "volume", "turnover_value", "instrument_type_candidate"],
        dtype={"ticker": str},
    )
    patch = patch[patch["instrument_type_candidate"].isin(["common_stock_candidate", "required_benchmark_or_fallback_etf"])].copy()
    patch = patch.drop(columns=["instrument_type_candidate"])
    out = pd.concat([base, patch], ignore_index=True, sort=False)
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["ticker"] = out["ticker"].map(normalize_ticker)
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    out["turnover_value"] = pd.to_numeric(out["turnover_value"], errors="coerce")
    out["volume"] = pd.to_numeric(out["volume"], errors="coerce")
    out = out.dropna(subset=["date", "ticker", "close"]).sort_values(["ticker", "date"])
    return out.drop_duplicates(["date", "ticker"], keep="last")


def compute_current_features(market: pd.DataFrame, requested_date: str) -> pd.DataFrame:
    common = market[~market["ticker"].isin(["0050", "00631L"])].copy()
    common = common.sort_values(["ticker", "date"])
    for window in [5, 10, 20, 40, 60]:
        common[f"close_lag_{window}"] = common.groupby("ticker")["close"].shift(window)
        common[f"return_{window}d"] = common["close"] / common[f"close_lag_{window}"] - 1.0
    common["ma20"] = common.groupby("ticker")["close"].transform(lambda s: s.rolling(20, min_periods=20).mean())
    common["bias20"] = common["close"] / common["ma20"] - 1.0
    for window in [5, 20, 60]:
        common[f"turnover_{window}d"] = common.groupby("ticker")["turnover_value"].transform(lambda s: s.rolling(window, min_periods=window).mean())
    current = common[common["date"].dt.strftime("%Y-%m-%d").eq(requested_date)].copy()
    if current.empty:
        return current
    for col in ["turnover_5d", "turnover_20d", "turnover_60d"]:
        rank_col = col.replace("turnover_", "traded_value_rank_")
        current[rank_col] = current[col].rank(method="first", ascending=False)
    current["in_5d_top300_today"] = current["traded_value_rank_5d"] <= 300
    current["layer0_core_top250"] = current["traded_value_rank_5d"] <= 250
    current["layer0_buffer_candidate_251_300"] = current["traded_value_rank_5d"].between(251, 300, inclusive="both")
    recent = common[common["date"] <= pd.Timestamp(requested_date)].copy()
    recent["rank5_tmp"] = recent.groupby("date")["turnover_5d"].rank(method="first", ascending=False)
    recent["in_5d_top300"] = recent["rank5_tmp"] <= 300
    last20_dates = sorted(recent["date"].dropna().unique())[-20:]
    recent20 = recent[recent["date"].isin(last20_dates)]
    counts = recent20.groupby("ticker")["in_5d_top300"].sum().rename("top300_count_last20_trading_days")
    current = current.merge(counts, on="ticker", how="left")
    current["top300_count_last20_trading_days"] = current["top300_count_last20_trading_days"].fillna(0)
    current["buffer_confirmation"] = (
        (current["top300_count_last20_trading_days"] >= 2)
        | (current["traded_value_rank_20d"] <= 300)
        | (current["traded_value_rank_60d"] <= 300)
    )
    current["layer0_active_scope"] = current["layer0_core_top250"] | (
        current["layer0_buffer_candidate_251_300"] & current["buffer_confirmation"]
    )
    return current


def mark_top300(group: pd.DataFrame) -> pd.DataFrame:
    group = group.copy()
    group["rank5_tmp"] = group["turnover_5d"].rank(method="first", ascending=False)
    group["in_5d_top300"] = group["rank5_tmp"] <= 300
    return group


def compute_benchmark_features(source_rows: Path, requested_date: str) -> pd.DataFrame:
    base = pd.read_csv(BENCHMARK_FEATURES, dtype={"benchmark": str})
    base = base[base["benchmark"].isin(["0050", "00631L"])].copy()
    base = base.rename(columns={"trade_date": "date", "benchmark": "ticker", "adjusted_close": "close"})
    base = base[["date", "ticker", "close", "source_quality", "benchmark_data_blocked"]]
    patch = pd.read_csv(
        source_rows,
        usecols=["date", "ticker", "close", "source_quality"],
        dtype={"ticker": str},
    )
    patch = patch[patch["ticker"].isin(["0050", "00631L"])].copy()
    patch["benchmark_data_blocked"] = False
    bench = pd.concat([base, patch], ignore_index=True, sort=False)
    bench["date"] = pd.to_datetime(bench["date"], errors="coerce")
    bench["ticker"] = bench["ticker"].map(normalize_ticker)
    bench["close"] = pd.to_numeric(bench["close"], errors="coerce")
    bench = bench.dropna(subset=["date", "ticker", "close"]).sort_values(["ticker", "date"])
    bench = bench.drop_duplicates(["date", "ticker"], keep="last")
    trade_dates = sorted(bench[bench["ticker"].eq("0050")]["date"].unique())
    row_dates = {d.strftime("%Y-%m-%d") for d in trade_dates}
    required_missing = [d for d in ["2026-07-01", "2026-07-02"] if d not in row_dates]
    rows = []
    for ticker, group in bench.groupby("ticker"):
        group = group.sort_values("date").copy()
        current = group[group["date"].dt.strftime("%Y-%m-%d").eq(requested_date)]
        if current.empty:
            continue
        current_row = current.iloc[-1]
        result: dict[str, Any] = {
            "date": requested_date,
            "ticker": ticker,
            "close": float(current_row["close"]),
            "source_quality": current_row.get("source_quality"),
            "missing_intermediate_dates_for_ma": "|".join(required_missing),
        }
        for window in [20, 40, 60]:
            dates = sorted(group["date"].unique())
            current_idx = dates.index(pd.Timestamp(requested_date).to_datetime64()) if pd.Timestamp(requested_date).to_datetime64() in dates else None
            if current_idx is not None and current_idx >= window:
                past_date = pd.Timestamp(dates[current_idx - window])
                past_close = float(group[group["date"].eq(past_date)]["close"].iloc[-1])
                result[f"return_{window}d"] = result["close"] / past_close - 1.0
                result[f"return_{window}d_source_quality"] = "current_and_lag_close_official_or_cache_exact"
            else:
                result[f"return_{window}d"] = None
                result[f"return_{window}d_source_quality"] = "blocked_insufficient_calendar"
        if required_missing:
            result["ma60"] = None
            result["bias60"] = None
            result["ma60_source_quality"] = "blocked_missing_0050_00631L_2026_07_01_2026_07_02_for_exact_ma60"
        else:
            result["ma60"] = group["close"].rolling(60, min_periods=60).mean().iloc[-1]
            result["bias60"] = result["close"] / result["ma60"] - 1.0
            result["ma60_source_quality"] = "exact_60_trading_day_window"
        rows.append(result)
    return pd.DataFrame(rows)


def build_layer0_compact(feature: pd.DataFrame) -> pd.DataFrame:
    if feature.empty:
        return pd.DataFrame()
    cols = [
        "date",
        "ticker",
        "name",
        "market",
        "close",
        "turnover_value",
        "turnover_5d",
        "turnover_20d",
        "turnover_60d",
        "traded_value_rank_5d",
        "traded_value_rank_20d",
        "traded_value_rank_60d",
        "top300_count_last20_trading_days",
        "layer0_core_top250",
        "layer0_buffer_candidate_251_300",
        "buffer_confirmation",
        "layer0_active_scope",
    ]
    out = feature[cols].copy()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    out["diagnostic_only"] = True
    for key, value in FLAGS.items():
        out[key] = value
    return out.sort_values(["layer0_active_scope", "traded_value_rank_5d"], ascending=[False, True])


def build_rs20_top3(feature: pd.DataFrame, benchmarks: pd.DataFrame) -> pd.DataFrame:
    if feature.empty or benchmarks.empty:
        return pd.DataFrame()
    b0050 = benchmarks[benchmarks["ticker"].eq("0050")]
    if b0050.empty or pd.isna(b0050["return_20d"].iloc[0]):
        return pd.DataFrame()
    ret20 = float(b0050["return_20d"].iloc[0])
    active = feature[feature["layer0_active_scope"]].copy()
    active["RS20"] = active["return_20d"] - ret20
    active["risk_tiebreak_bias20_abs"] = active["bias20"].abs()
    active = active.sort_values(
        ["RS20", "traded_value_rank_20d", "risk_tiebreak_bias20_abs", "ticker"],
        ascending=[False, True, True, True],
    )
    top = active.head(3).copy()
    top["rs20_rank"] = range(1, len(top) + 1)
    top["date"] = top["date"].dt.strftime("%Y-%m-%d")
    top["0050_return_20d_used"] = ret20
    top["source_quality"] = "official_unadjusted_close_rs20_reference_layer0_active_scope; risk_tiebreak=RS20_then_20d_turnover_rank_then_abs_bias20"
    top["diagnostic_only"] = True
    for key, value in FLAGS.items():
        top[key] = value
    return top[
        [
            "date",
            "rs20_rank",
            "ticker",
            "name",
            "market",
            "close",
            "return_20d",
            "0050_return_20d_used",
            "RS20",
            "traded_value_rank_20d",
            "bias20",
            "source_quality",
            "diagnostic_only",
            *FLAGS.keys(),
        ]
    ]


def build_market_regime_partial(benchmarks: pd.DataFrame) -> pd.DataFrame:
    if benchmarks.empty:
        return pd.DataFrame()
    out = benchmarks.copy()
    out["c2_return_20d_nonnegative"] = pd.to_numeric(out["return_20d"], errors="coerce") >= 0
    out["c2_return_40d_nonnegative"] = pd.to_numeric(out["return_40d"], errors="coerce") >= 0
    out["c2_above_ma60_ready"] = out["ma60"].notna()
    out["c2_above_ma60"] = False
    ready = out["c2_above_ma60_ready"]
    out.loc[ready, "c2_above_ma60"] = out.loc[ready, "close"] >= out.loc[ready, "ma60"]
    out["c2_market_health_gate_ready"] = out["ticker"].eq("0050") & out["c2_return_20d_nonnegative"].notna() & out["c2_return_40d_nonnegative"].notna() & out["c2_above_ma60_ready"]
    out["c2_market_health_gate"] = out["ticker"].eq("0050") & out["c2_return_20d_nonnegative"] & out["c2_return_40d_nonnegative"] & out["c2_above_ma60"]
    out["c2_definition"] = "0050 above MA60 + 0050 20D/40D returns non-negative"
    out["diagnostic_only"] = True
    for key, value in FLAGS.items():
        out[key] = value
    return out


def build_signal_snapshot(requested_date: str, market_regime: pd.DataFrame, rs20_top3: pd.DataFrame) -> dict[str, Any]:
    m0050 = market_regime[market_regime["ticker"].eq("0050")].iloc[0].to_dict() if not market_regime[market_regime["ticker"].eq("0050")].empty else {}
    rs_rows = rs20_top3.to_dict("records") if not rs20_top3.empty else []
    c2_ready = bool(m0050.get("c2_market_health_gate_ready", False))
    c2_pass = bool(m0050.get("c2_market_health_gate", False)) if c2_ready else False
    return {
        "as_of_requested_date": requested_date,
        "as_of_data_date": requested_date,
        "market_data_ready": True,
        "c2_gate_ready": c2_ready,
        "c2_gate_pass": c2_pass,
        "consensus_trigger_ready": False,
        "consensus_trigger_pass": False,
        "c2_selected_asset_type": "blocked",
        "c2_selected_ticker": "",
        "c2_selected_name": "",
        "c2_blocked_reason": "C2 exact MA60 and exact consensus trigger / route_support max1 not fully materialized for 2026-07-08",
        "c2_reference_stock_top1_ticker": rs_rows[0]["ticker"] if rs_rows else "",
        "c2_reference_stock_top1_name": rs_rows[0]["name"] if rs_rows else "",
        "rs20_top1_ticker": rs_rows[0]["ticker"] if rs_rows else "",
        "rs20_top1_name": rs_rows[0]["name"] if rs_rows else "",
        "rs20_top3_tickers": "|".join(str(r["ticker"]) for r in rs_rows),
        "same_top1_flag": False,
        "0050_return_20d": m0050.get("return_20d"),
        "0050_return_40d": m0050.get("return_40d"),
        "0050_ma60_source_quality": m0050.get("ma60_source_quality"),
        "diagnostic_only": True,
        "not_live_trade_decision": True,
        **FLAGS,
    }


def build_blocked_proxy_audit(market_regime: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {
            "field": "0050_ma60_exact",
            "field_as_of_date": REQUESTED_DATE,
            "blocked_reason": "Core benchmark_features lacks official 0050/00631L rows for 2026-07-01 and 2026-07-02; MA60 cannot be exact with gap",
            "proxy_policy": "do not use approximate MA60 to pass C2 gate",
            "next_owner": "Radar/Data bounded ETF rows or Core benchmark absorption if source exists",
        },
        {
            "field": "exact_consensus_trigger_20260708",
            "field_as_of_date": "2026-06-29",
            "blocked_reason": "exact consensus trigger contract latest row is 2026-06-29",
            "proxy_policy": "do not substitute route_support>=4 or RS20 as trigger",
            "next_owner": "Core/Data exact trigger materialization after route source fields ready",
        },
        {
            "field": "layer4_primary80_20260708",
            "field_as_of_date": "2026-06-29",
            "blocked_reason": "Layer4 primary80 contract latest row is 2026-06-29",
            "proxy_policy": "Layer0 and RS20 reference are materialized; not selected pool",
            "next_owner": "Core/Data bounded Layer4 refresh",
        },
        {
            "field": "route_support_max1_20260708",
            "field_as_of_date": "2026-06-29",
            "blocked_reason": "route_support max1 state-machine latest row is 2026-06-29",
            "proxy_policy": "do not publish selected top1 until C2 + exact trigger + Layer4 + route_support are same-date",
            "next_owner": "Core/Data after trigger and Layer4 refresh",
        },
        {
            "field": "selected_stock_adjusted_close",
            "field_as_of_date": "",
            "blocked_reason": "adjusted close remains blocked",
            "proxy_policy": "official unadjusted OHLC only for diagnostic/reference",
            "next_owner": "Strategy Center policy or Radar/Data adjusted source route",
        },
    ]
    if not market_regime.empty:
        rows.append(
            {
                "field": "0050_return_20d_40d",
                "field_as_of_date": REQUESTED_DATE,
                "blocked_reason": "",
                "proxy_policy": "materialized as feature only; cannot alone decide C2 without exact MA60",
                "next_owner": "Core/Data",
            }
        )
    return pd.DataFrame(rows)


def build_coverage_audit(source_rows: Path, window_readiness: Path, feature: pd.DataFrame, layer0: pd.DataFrame, rs20: pd.DataFrame, market_regime: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {
            "item": "radar_historical_window_source",
            "requested_date_ready": True,
            "actual_date": REQUESTED_DATE,
            "row_count": int(pd.read_csv(source_rows, usecols=["date"]).shape[0]) if source_rows.exists() else 0,
            "source_path": str(source_rows),
        },
        {
            "item": "common_stock_feature_refresh",
            "requested_date_ready": not feature.empty,
            "actual_date": REQUESTED_DATE if not feature.empty else "",
            "row_count": len(feature),
            "source_path": str(DAILY_MARKET),
        },
        {
            "item": "layer0_compact_active_universe",
            "requested_date_ready": not layer0.empty,
            "actual_date": REQUESTED_DATE if not layer0.empty else "",
            "row_count": int(layer0["layer0_active_scope"].sum()) if not layer0.empty else 0,
            "source_path": str(source_rows),
        },
        {
            "item": "rs20_top3_reference",
            "requested_date_ready": not rs20.empty,
            "actual_date": REQUESTED_DATE if not rs20.empty else "",
            "row_count": len(rs20),
            "source_path": str(source_rows),
        },
        {
            "item": "c2_market_health_gate_exact",
            "requested_date_ready": bool(not market_regime.empty and market_regime[market_regime["ticker"].eq("0050")]["c2_market_health_gate_ready"].any()),
            "actual_date": REQUESTED_DATE,
            "row_count": int(len(market_regime)),
            "source_path": str(BENCHMARK_FEATURES),
        },
        {
            "item": "window_readiness_json",
            "requested_date_ready": window_readiness.exists(),
            "actual_date": REQUESTED_DATE,
            "row_count": 1 if window_readiness.exists() else 0,
            "source_path": str(window_readiness),
        },
    ]
    return pd.DataFrame(rows)


def normalize_ticker(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text.zfill(4) if text.isdigit() and len(text) < 4 else text


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_summary(path: Path, snapshot: dict[str, Any], readiness: dict[str, Any]) -> None:
    path.write_text(
        f"""# vNext 2026-07-08 EOD signal materialization refresh

## 結論

- 2026-07-08 官方 EOD historical window source 已吸收為 Core bounded materialization input。
- Layer0 compact active universe 與 RS20 top3 reference 已可用 2026-07-08 官方未調整 OHLCV 重算。
- C2 / route_support selected signal 仍不可發布：0050 MA60 exact 缺 2026-07-01 / 2026-07-02 ETF rows，且 exact consensus trigger / Layer4 primary80 / route_support max1 尚未 materialized 到 2026-07-08。

## 今日參考結果

- RS20 top1 reference: `{snapshot.get('rs20_top1_ticker')} {snapshot.get('rs20_top1_name')}`。
- RS20 top3 reference: `{snapshot.get('rs20_top3_tickers')}`。
- C2 gate ready: `{snapshot.get('c2_gate_ready')}`；C2 gate pass: `{snapshot.get('c2_gate_pass')}`。
- c2_selected_asset_type: `{snapshot.get('c2_selected_asset_type')}`。

## Readiness

- ready_for_vnext_daily_report_selected_signal_publish=false。
- ready_for_live_publish=false。
- ready_for_experiments=false。
- future_data_violation_count=0。

## Flags

- formal_model_changed=false
- trade_decision_changed=false
- active_in_trade_decision=false
- report_changed=true
- portfolio_replay_executed=false
- ready_for_strategy_replay=false
- ready_for_formal=false
- not_live_rule=true
- forward_returns_live_rule_usage=false
""",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
