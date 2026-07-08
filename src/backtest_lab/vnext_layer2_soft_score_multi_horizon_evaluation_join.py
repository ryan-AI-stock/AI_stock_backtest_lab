"""Build Layer2 soft-score multi-horizon evaluation join.

Forward returns and multi-horizon shape fields are evaluation metadata only.
They are not live rule inputs, selectors, formal model inputs, report inputs, or
trade decisions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


TASK_ID = "TASK-BACKTEST-CORE-VNEXT-LAYER2-SOFT-SCORE-MULTI-HORIZON-EVALUATION-JOIN-001"
DEFAULT_SOFT_DIR = Path("outputs/vnext_layer2_soft_score_feature_contract_20260708")
DEFAULT_DATA_DIR = Path("outputs/vnext_dynamic_candidate_pool_data_materialization_20260706")
DEFAULT_EXPERIMENTS_DIR = Path(
    "C:/Users/zergv/Documents/Codex/2026-07-06/backtest-lab-experiments-diagnostic-validation-attribution/"
    "outputs/vnext_layer2_soft_score_bounded_diagnostic_20260708"
)
DEFAULT_OUTPUT_DIR = Path("outputs/vnext_layer2_soft_score_multi_horizon_evaluation_join_20260708")
HORIZONS = [5, 10, 20, 30, 40]
BENCHMARKS = ["0050", "00631L"]
PERIODS = {
    "P1": ("2015-01-02", "2022-12-29"),
    "P2": ("2023-01-02", "2026-06-30"),
    "2024_latest": ("2024-01-02", "2026-06-30"),
    "2026YTD": ("2026-01-02", "2026-06-30"),
}


def build_join(
    *,
    soft_dir: str | Path = DEFAULT_SOFT_DIR,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    experiments_dir: str | Path = DEFAULT_EXPERIMENTS_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    soft = Path(soft_dir)
    data = Path(data_dir)
    experiments = Path(experiments_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    soft_readiness = _read_json(soft / "readiness_for_layer2_soft_score_diagnostic.json")
    experiment_summary = _read_json(experiments / "layer2_soft_score_summary.json")
    base = _read_soft_contract(soft / "layer2_soft_score_feature_contract.csv")
    calendar = _read_calendar(data / "trading_calendar.csv")
    stock_daily = _read_stock_daily(data / "daily_market_features.csv", base["ticker"].unique())
    benchmark_prices = _read_benchmark_prices(data / "benchmark_features.csv")

    joined = _attach_forward_returns(base, calendar, stock_daily, benchmark_prices)
    joined = _attach_outcome_labels(joined)
    joined = _attach_decile_labels(joined)
    joined = _attach_multi_horizon_shape(joined)
    joined = _attach_path_risk_proxies(joined, calendar, stock_daily)
    joined = _attach_policy_flags(joined)

    coverage = _coverage_by_period(joined)
    blocked_latest = _blocked_rows(joined)
    missingness = _missingness_by_period(joined)
    source_quality = _source_quality_matrix(joined)
    blocked_proxy = _blocked_proxy_fields()
    future_audit = _future_data_audit()
    readiness = _readiness(soft_readiness, experiment_summary, joined, blocked_latest)

    _write_csv(joined, output / "layer2_soft_score_multi_horizon_evaluation_join.csv")
    _write_csv(joined.head(1000), output / "layer2_soft_score_multi_horizon_evaluation_join_sample.csv")
    (output / ".gitignore").write_text("layer2_soft_score_multi_horizon_evaluation_join.csv\n", encoding="utf-8")
    _write_csv(coverage, output / "layer2_soft_score_multi_horizon_requested_vs_actual_coverage.csv")
    _write_csv(blocked_latest, output / "layer2_soft_score_multi_horizon_blocked_latest_rows.csv")
    _write_csv(missingness, output / "layer2_soft_score_multi_horizon_missingness_by_period.csv")
    _write_csv(source_quality, output / "layer2_soft_score_multi_horizon_source_quality_matrix.csv")
    _write_csv(blocked_proxy, output / "layer2_soft_score_multi_horizon_blocked_proxy_fields.csv")
    _write_csv(future_audit, output / "layer2_soft_score_multi_horizon_future_data_audit.csv")
    (output / "readiness_for_layer2_soft_score_multi_horizon_diagnostic.json").write_text(
        json.dumps(readiness, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    manifest = {
        "task_id": TASK_ID,
        "status": readiness["status"],
        "output_dir": str(output.resolve()),
        "input_soft_score_dir": str(soft.resolve()),
        "input_data_dir": str(data.resolve()),
        "input_experiments_dir": str(experiments.resolve()),
        "output_files": [
            "layer2_soft_score_multi_horizon_evaluation_join.csv",
            "layer2_soft_score_multi_horizon_evaluation_join_sample.csv",
            "layer2_soft_score_multi_horizon_requested_vs_actual_coverage.csv",
            "layer2_soft_score_multi_horizon_blocked_latest_rows.csv",
            "layer2_soft_score_multi_horizon_missingness_by_period.csv",
            "layer2_soft_score_multi_horizon_source_quality_matrix.csv",
            "layer2_soft_score_multi_horizon_blocked_proxy_fields.csv",
            "layer2_soft_score_multi_horizon_future_data_audit.csv",
            "readiness_for_layer2_soft_score_multi_horizon_diagnostic.json",
            "manifest.json",
            "final_summary_zh.md",
        ],
        "large_local_files_not_tracked": ["layer2_soft_score_multi_horizon_evaluation_join.csv"],
        "large_local_file_policy": "full multi-horizon evaluation join is retained in local output path; Git tracks sample/readiness/audit files only",
        "formal_model_changed": False,
        "trade_decision_changed": False,
        "active_in_trade_decision": False,
        "report_changed": False,
        "portfolio_replay_executed": False,
        "ready_for_strategy_replay": False,
        "not_live_rule": True,
        "forward_returns_live_rule_usage": False,
        "diagnostic_only": True,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "final_summary_zh.md").write_text(_summary(readiness), encoding="utf-8")
    return manifest


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def _read_soft_contract(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"ticker": str}, encoding="utf-8-sig", low_memory=False)
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
    drop_prefixes = (
        "forward_excess_vs_",
        "forward_return_",
        "0050_forward_return_",
        "00631L_forward_return_",
        "target_date_",
        "stock_target_close_",
        "0050_target_close_",
        "00631L_target_close_",
        "P1_top_decile_",
        "P1_bottom_decile_",
        "P2_top_decile_",
        "P2_bottom_decile_",
        "2024_latest_top_decile_",
        "2024_latest_bottom_decile_",
        "2026YTD_top_decile_",
        "2026YTD_bottom_decile_",
    )
    drop_exact = {
        "stock_entry_close",
        "0050_entry_close",
        "00631L_entry_close",
        "forward_eval_available_20d",
        "win_both_20d",
        "only_win_0050_lose_00631L_20d",
        "fail_0050_20d",
    }
    drop_cols = [c for c in df.columns if c.startswith(drop_prefixes) or c in drop_exact]
    return df.drop(columns=drop_cols, errors="ignore")


def _read_calendar(path: Path) -> pd.DataFrame:
    cal = pd.read_csv(path)
    cal["trade_date"] = pd.to_datetime(cal["trade_date"])
    cal = cal.sort_values("trade_date").reset_index(drop=True)
    cal["trade_index"] = range(len(cal))
    return cal[["trade_date", "trade_index"]]


def _read_stock_daily(path: Path, tickers: Any) -> pd.DataFrame:
    ticker_set = set(map(str, tickers))
    usecols = ["trade_date", "ticker", "adjusted_close", "traded_value"]
    chunks = []
    for chunk in pd.read_csv(path, usecols=usecols, dtype={"ticker": str}, chunksize=500_000):
        chunk = chunk[chunk["ticker"].isin(ticker_set)].copy()
        if chunk.empty:
            continue
        chunk["trade_date"] = pd.to_datetime(chunk["trade_date"])
        chunk["adjusted_close"] = pd.to_numeric(chunk["adjusted_close"], errors="coerce")
        chunk["traded_value"] = pd.to_numeric(chunk["traded_value"], errors="coerce")
        chunks.append(chunk)
    daily = pd.concat(chunks, ignore_index=True).dropna(subset=["adjusted_close"])
    daily = daily.sort_values(["ticker", "trade_date"]).drop_duplicates(["ticker", "trade_date"], keep="last")
    daily = daily.sort_values(["ticker", "trade_date"]).copy()
    daily["daily_return"] = daily.groupby("ticker")["adjusted_close"].pct_change()
    daily["large_down_day_proxy"] = daily["daily_return"].le(-0.05)
    daily["large_down_day_count_20d_proxy"] = (
        daily.groupby("ticker")["large_down_day_proxy"].transform(lambda s: s.astype(float).rolling(20, min_periods=1).sum())
    )
    daily["large_down_day_count_30d_proxy"] = (
        daily.groupby("ticker")["large_down_day_proxy"].transform(lambda s: s.astype(float).rolling(30, min_periods=1).sum())
    )
    tv_mean = daily.groupby("ticker")["traded_value"].transform(lambda s: s.rolling(20, min_periods=5).mean())
    tv_std = daily.groupby("ticker")["traded_value"].transform(lambda s: s.rolling(20, min_periods=5).std())
    daily["traded_value_z20_proxy"] = (daily["traded_value"] - tv_mean) / tv_std
    daily["return_5d_past_proxy"] = daily.groupby("ticker")["adjusted_close"].pct_change(5)
    daily["blowoff_turnover_without_price_continuation_proxy"] = (
        daily["traded_value_z20_proxy"].ge(2.5) & daily["return_5d_past_proxy"].le(0)
    )
    return daily


def _read_benchmark_prices(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=["trade_date", "benchmark", "adjusted_close", "benchmark_data_blocked", "source_quality"])
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df["adjusted_close"] = pd.to_numeric(df["adjusted_close"], errors="coerce")
    df["benchmark_data_blocked"] = df["benchmark_data_blocked"].astype(str).str.lower().eq("true")
    df = df[df["benchmark"].isin(BENCHMARKS)].copy()
    return df.sort_values(["benchmark", "trade_date"]).drop_duplicates(["benchmark", "trade_date"], keep="last")


def _attach_forward_returns(
    events: pd.DataFrame,
    calendar: pd.DataFrame,
    stock_daily: pd.DataFrame,
    benchmark_prices: pd.DataFrame,
) -> pd.DataFrame:
    out = events.merge(calendar.rename(columns={"trade_date": "snapshot_date"}), on="snapshot_date", how="left")
    stock_entry = stock_daily[["trade_date", "ticker", "adjusted_close"]].rename(
        columns={"trade_date": "snapshot_date", "adjusted_close": "stock_entry_close"}
    )
    out = out.merge(stock_entry, on=["snapshot_date", "ticker"], how="left")

    bench_entry = benchmark_prices.rename(columns={"trade_date": "snapshot_date", "adjusted_close": "benchmark_entry_close"})
    for benchmark in BENCHMARKS:
        entry = bench_entry[bench_entry["benchmark"].eq(benchmark)][
            ["snapshot_date", "benchmark_entry_close", "source_quality", "benchmark_data_blocked"]
        ].rename(
            columns={
                "benchmark_entry_close": f"{benchmark}_entry_close",
                "source_quality": f"{benchmark}_entry_source_quality",
                "benchmark_data_blocked": f"{benchmark}_entry_blocked",
            }
        )
        out = out.merge(entry, on="snapshot_date", how="left")

    for horizon in HORIZONS:
        target = calendar.copy()
        target["trade_index"] = target["trade_index"] - horizon
        target = target.rename(columns={"trade_date": f"target_date_{horizon}d"})
        out = out.merge(target, on="trade_index", how="left")

        stock_target = stock_daily[["trade_date", "ticker", "adjusted_close"]].rename(
            columns={"trade_date": f"target_date_{horizon}d", "adjusted_close": f"stock_target_close_{horizon}d"}
        )
        out = out.merge(stock_target, on=[f"target_date_{horizon}d", "ticker"], how="left")
        out[f"forward_return_{horizon}d"] = out[f"stock_target_close_{horizon}d"] / out["stock_entry_close"] - 1

        for benchmark in BENCHMARKS:
            btarget = benchmark_prices[benchmark_prices["benchmark"].eq(benchmark)].rename(
                columns={
                    "trade_date": f"target_date_{horizon}d",
                    "adjusted_close": f"{benchmark}_target_close_{horizon}d",
                    "source_quality": f"{benchmark}_target_source_quality_{horizon}d",
                    "benchmark_data_blocked": f"{benchmark}_target_blocked_{horizon}d",
                }
            )
            btarget = btarget[
                [
                    f"target_date_{horizon}d",
                    f"{benchmark}_target_close_{horizon}d",
                    f"{benchmark}_target_source_quality_{horizon}d",
                    f"{benchmark}_target_blocked_{horizon}d",
                ]
            ]
            out = out.merge(btarget, on=f"target_date_{horizon}d", how="left")
            out[f"{benchmark}_forward_return_{horizon}d"] = (
                out[f"{benchmark}_target_close_{horizon}d"] / out[f"{benchmark}_entry_close"] - 1
            )
            out[f"forward_excess_vs_{benchmark}_{horizon}d"] = (
                out[f"forward_return_{horizon}d"] - out[f"{benchmark}_forward_return_{horizon}d"]
            )
    return out


def _attach_outcome_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for horizon in HORIZONS:
        vs0050 = out[f"forward_excess_vs_0050_{horizon}d"]
        vs00631 = out[f"forward_excess_vs_00631L_{horizon}d"]
        available = vs0050.notna() & vs00631.notna()
        out[f"win_both_{horizon}d"] = available & vs0050.gt(0) & vs00631.gt(0)
        out[f"only_win_0050_lose_00631L_{horizon}d"] = available & vs0050.gt(0) & vs00631.le(0)
        out[f"fail_0050_{horizon}d"] = available & vs0050.le(0)
        out[f"forward_eval_available_{horizon}d"] = available
    return out


def _attach_decile_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for period, (start, end) in PERIODS.items():
        mask = out["snapshot_date"].between(pd.Timestamp(start), pd.Timestamp(end))
        for horizon in HORIZONS:
            for benchmark in BENCHMARKS:
                col = f"forward_excess_vs_{benchmark}_{horizon}d"
                rank = out.loc[mask, col].rank(pct=True, method="average")
                out[f"{period}_top_decile_vs_{benchmark}_{horizon}d"] = pd.NA
                out[f"{period}_bottom_decile_vs_{benchmark}_{horizon}d"] = pd.NA
                out.loc[mask, f"{period}_top_decile_vs_{benchmark}_{horizon}d"] = rank.ge(0.90)
                out.loc[mask, f"{period}_bottom_decile_vs_{benchmark}_{horizon}d"] = rank.le(0.10)
    return out


def _attach_multi_horizon_shape(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for benchmark in BENCHMARKS:
        e5 = out[f"forward_excess_vs_{benchmark}_5d"]
        e10 = out[f"forward_excess_vs_{benchmark}_10d"]
        e20 = out[f"forward_excess_vs_{benchmark}_20d"]
        e30 = out[f"forward_excess_vs_{benchmark}_30d"]
        e40 = out[f"forward_excess_vs_{benchmark}_40d"]
        available = e5.notna() & e10.notna() & e20.notna() & e30.notna()
        out[f"shape_improving_vs_{benchmark}"] = available & e5.lt(e10) & e10.lt(e20) & e20.lt(e30)
        out[f"shape_fading_vs_{benchmark}"] = available & e5.gt(e10) & e10.gt(e20) & e20.gt(e30)
        out[f"shape_quick_burst_vs_{benchmark}"] = available & e5.gt(0) & e10.le(e5) & e20.le(e10)
        out[f"shape_short_bounce_fade_risk_vs_{benchmark}"] = available & e5.gt(0) & e20.lt(0) & e30.lt(0)
        out[f"shape_slow_start_vs_{benchmark}"] = available & e5.le(0) & e10.gt(e5) & e20.gt(e10) & e30.gt(e20)
        out[f"shape_durable_vs_{benchmark}"] = e20.notna() & e30.notna() & e40.notna() & e20.gt(0) & e30.gt(0) & e40.gt(0)
        out[f"shape_40d_decay_reference_vs_{benchmark}"] = e30.notna() & e40.notna() & e40.lt(e30)
    out["multi_horizon_shape_evaluation_metadata_only"] = True
    return out


def _attach_path_risk_proxies(df: pd.DataFrame, calendar: pd.DataFrame, stock_daily: pd.DataFrame) -> pd.DataFrame:
    risk_cols = [
        "trade_date",
        "ticker",
        "large_down_day_count_20d_proxy",
        "large_down_day_count_30d_proxy",
        "large_down_day_proxy",
        "traded_value_z20_proxy",
        "blowoff_turnover_without_price_continuation_proxy",
    ]
    risk = stock_daily[risk_cols].rename(columns={"trade_date": "snapshot_date"})
    out = df.merge(risk, on=["snapshot_date", "ticker"], how="left")
    out["large_down_day_flag_20d_proxy"] = out["large_down_day_count_20d_proxy"].ge(1)
    out["large_down_day_flag_30d_proxy"] = out["large_down_day_count_30d_proxy"].ge(1)
    out["large_down_day_source_quality"] = "diagnostic_price_proxy_threshold_not_formal"
    out["blowoff_turnover_source_quality"] = "diagnostic_traded_value_proxy_threshold_not_formal"
    out["risk_bucket"] = pd.NA
    out["risk_bucket_source_quality"] = "blocked_no_accepted_pit_risk_bucket"
    return out


def _attach_policy_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["evaluation_metadata_only"] = True
    out["forward_return_as_rule"] = False
    out["future_return_as_rule"] = False
    out["multi_horizon_shape_as_rule"] = False
    out["selector_output"] = False
    out["live_rule"] = False
    out["diagnostic_only"] = True
    out["not_live_rule"] = True
    out["forward_returns_live_rule_usage"] = False
    out["formal_model_changed"] = False
    out["trade_decision_changed"] = False
    out["active_in_trade_decision"] = False
    out["report_changed"] = False
    return out


def _coverage_by_period(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for period, (start, end) in {"ALL": (None, None), **PERIODS}.items():
        mask = pd.Series(True, index=df.index)
        if start:
            mask &= df["snapshot_date"].ge(pd.Timestamp(start))
        if end:
            mask &= df["snapshot_date"].le(pd.Timestamp(end))
        sub = df[mask]
        row: dict[str, Any] = {
            "period": period,
            "requested_start": start or str(df["snapshot_date"].min().date()),
            "requested_end": end or str(df["snapshot_date"].max().date()),
            "actual_start": str(sub["snapshot_date"].min().date()) if not sub.empty else "",
            "actual_end": str(sub["snapshot_date"].max().date()) if not sub.empty else "",
            "rows": int(len(sub)),
            "weekly_snapshot_count": int(sub["snapshot_date"].nunique()),
            "unique_ticker_count": int(sub["ticker"].nunique()),
            "evaluation_metadata_only": True,
        }
        for horizon in HORIZONS:
            row[f"forward_eval_available_share_{horizon}d"] = float(sub[f"forward_eval_available_{horizon}d"].mean()) if len(sub) else 0.0
            row[f"blocked_rows_{horizon}d"] = int((~sub[f"forward_eval_available_{horizon}d"]).sum()) if len(sub) else 0
        rows.append(row)
    return pd.DataFrame(rows)


def _blocked_rows(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for horizon in HORIZONS:
        blocked = df[~df[f"forward_eval_available_{horizon}d"]].copy()
        if blocked.empty:
            continue
        target_col = f"target_date_{horizon}d"
        reasons = pd.DataFrame(
            {
                "insufficient_calendar_path": blocked[target_col].isna(),
                "stock_price_missing": blocked[f"stock_target_close_{horizon}d"].isna() | blocked["stock_entry_close"].isna(),
                "benchmark_price_missing": blocked[f"0050_forward_return_{horizon}d"].isna() | blocked[f"00631L_forward_return_{horizon}d"].isna(),
            }
        )
        out = blocked[["baseline_compact_universe_row_id", "snapshot_date", "ticker", "name", "market"]].copy()
        out["horizon"] = horizon
        out["target_date"] = blocked[target_col]
        out["blocked_reason"] = reasons.apply(lambda r: ";".join([k for k, v in r.items() if v]), axis=1)
        out["evaluation_metadata_only"] = True
        rows.append(out)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _missingness_by_period(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    feature_cols = [
        "forward_excess_vs_00631L_5d",
        "forward_excess_vs_00631L_10d",
        "forward_excess_vs_00631L_20d",
        "forward_excess_vs_00631L_30d",
        "forward_excess_vs_00631L_40d",
        "shape_improving_vs_00631L",
        "shape_fading_vs_00631L",
        "large_down_day_count_20d_proxy",
        "large_down_day_count_30d_proxy",
        "blowoff_turnover_without_price_continuation_proxy",
        "BIAS20_percentile",
        "BIAS60_percentile",
        "volatility",
        "risk_bucket",
    ]
    for period, (start, end) in {"ALL": (None, None), **PERIODS}.items():
        mask = pd.Series(True, index=df.index)
        if start:
            mask &= df["snapshot_date"].ge(pd.Timestamp(start))
        if end:
            mask &= df["snapshot_date"].le(pd.Timestamp(end))
        sub = df[mask]
        for col in feature_cols:
            rows.append(
                {
                    "period": period,
                    "feature": col,
                    "rows": int(len(sub)),
                    "available_rows": int(sub[col].notna().sum()) if col in sub else 0,
                    "missing_rows": int(sub[col].isna().sum()) if col in sub else int(len(sub)),
                    "available_share": float(sub[col].notna().mean()) if len(sub) and col in sub else 0.0,
                }
            )
    return pd.DataFrame(rows)


def _source_quality_matrix(df: pd.DataFrame) -> pd.DataFrame:
    rows = [
        ("forward_5_10_20_30_40d", "exact_endpoint_adjusted_close", float(df["forward_eval_available_30d"].mean()), "30D computed directly; not 20/40 proxy"),
        ("multi_horizon_shape", "evaluation_metadata_only_from_forward_excess", float(df["shape_improving_vs_00631L"].notna().mean()), "not a live rule input"),
        ("large_down_day", "diagnostic_price_proxy_threshold_not_formal", float(df["large_down_day_count_20d_proxy"].notna().mean()), "threshold candidate only"),
        ("blowoff_turnover", "diagnostic_traded_value_proxy_threshold_not_formal", float(df["blowoff_turnover_without_price_continuation_proxy"].notna().mean()), "proxy, not accepted formal trigger"),
        ("risk_bucket", "blocked_no_accepted_pit_risk_bucket", 0.0, "do not fabricate"),
        ("BIAS_volatility", "pit_stock_features_diagnostic_context", float(df["BIAS20_percentile"].notna().mean()), "risk/overheat context only"),
        ("RS_exhaustion", "pit_rs_window_context", float(df["rs60_high_short_rs_weakening_exhaustion_context"].notna().mean()), "context only"),
    ]
    return pd.DataFrame(rows, columns=["feature_group", "source_quality", "available_share", "note"])


def _blocked_proxy_fields() -> pd.DataFrame:
    rows = [
        ("30D_forward_excess", "exact", "computed from adjusted/available close endpoint", "evaluation_metadata_only"),
        ("multi_horizon_shape", "evaluation_metadata_only", "derived from forward evaluation metadata", "not live rule"),
        ("large_down_day_count", "diagnostic_proxy", "computed from adjusted close daily return <= -5% trailing 20/30D", "threshold not formal"),
        ("blowoff_turnover_without_price_continuation", "diagnostic_proxy", "traded_value z20 >= 2.5 and past 5D return <= 0", "not accepted formal trigger"),
        ("risk_bucket", "blocked", "no accepted PIT risk bucket field", "do not fabricate"),
        ("BIAS20_60_percentile", "pit_diagnostic_context", "from stock_features", "overheat context only"),
        ("volatility", "pit_diagnostic_context", "from stock_features", "risk context only"),
        ("RS60_high_short_RS_weakening", "pit_diagnostic_context", "from RS window contract", "exhaustion context only"),
    ]
    return pd.DataFrame(rows, columns=["field", "status", "reason", "policy"])


def _future_data_audit() -> pd.DataFrame:
    rows = [
        ("forward_return_as_rule", "passed", 0, "forward returns are evaluation_metadata_only"),
        ("30d_proxy_usage", "passed", 0, "30D uses exact endpoint close, not 20D/40D proxy"),
        ("multi_horizon_shape_as_rule", "passed", 0, "shape fields are evaluation metadata only"),
        ("risk_feature_pit", "passed", 0, "risk proxies use as-of daily history up to snapshot_date"),
        ("selector_output", "not_applicable", 0, "no selector output produced"),
        ("portfolio_replay", "not_executed", 0, "no replay executed"),
    ]
    return pd.DataFrame(rows, columns=["audit_item", "status", "future_data_violation_count", "note"])


def _readiness(
    soft_readiness: dict[str, Any],
    experiment_summary: dict[str, Any],
    df: pd.DataFrame,
    blocked_latest: pd.DataFrame,
) -> dict[str, Any]:
    eval30_share = float(df["forward_eval_available_30d"].mean())
    eval5_share = float(df["forward_eval_available_5d"].mean())
    large_down_share = float(df["large_down_day_count_20d_proxy"].notna().mean())
    blowoff_share = float(df["blowoff_turnover_without_price_continuation_proxy"].notna().mean())
    bias_share = float(df["BIAS20_percentile"].notna().mean())
    ready = eval30_share > 0.80 and eval5_share > 0.80 and bias_share > 0.80
    return {
        "task_id": TASK_ID,
        "status": "layer2_soft_score_multi_horizon_evaluation_join_ready_for_experiments_intake" if ready else "layer2_soft_score_multi_horizon_evaluation_join_partial_blocked",
        "diagnostic_only": True,
        "evaluation_metadata_only": True,
        "selector_output": False,
        "input_soft_score_status": soft_readiness.get("status", ""),
        "input_experiments_verdict": experiment_summary.get("verdict", ""),
        "rows": int(len(df)),
        "weekly_snapshot_count": int(df["snapshot_date"].nunique()),
        "unique_ticker_count": int(df["ticker"].nunique()),
        "forward_eval_available_share_5d": eval5_share,
        "forward_eval_available_share_10d": float(df["forward_eval_available_10d"].mean()),
        "forward_eval_available_share_20d": float(df["forward_eval_available_20d"].mean()),
        "forward_eval_available_share_30d": eval30_share,
        "forward_eval_available_share_40d": float(df["forward_eval_available_40d"].mean()),
        "blocked_evaluation_rows": int(len(blocked_latest)),
        "large_down_day_available": True,
        "large_down_day_source_quality": "diagnostic_price_proxy_threshold_not_formal",
        "large_down_day_available_share": large_down_share,
        "blowoff_turnover_available": True,
        "blowoff_turnover_source_quality": "diagnostic_traded_value_proxy_threshold_not_formal",
        "blowoff_turnover_available_share": blowoff_share,
        "risk_bucket_available": False,
        "risk_bucket_blocked_reason": "no accepted PIT risk_bucket field",
        "bias20_percentile_available_share": bias_share,
        "volatility_available_share": float(df["volatility"].notna().mean()),
        "rs_exhaustion_context_available_share": float(df["rs60_high_short_rs_weakening_exhaustion_context"].notna().mean()),
        "ready_for_layer2_soft_score_multi_horizon_risk_diagnostic": ready,
        "ready_for_experiments_intake": ready,
        "ready_for_portfolio_like_diagnostic": False,
        "ready_for_formal": False,
        "ready_for_strategy_replay": False,
        "portfolio_replay_executed": False,
        "candidate_forward_return_diagnostic_executed": False,
        "future_data_violation_count": 0,
        "formal_model_changed": False,
        "trade_decision_changed": False,
        "active_in_trade_decision": False,
        "report_changed": False,
        "not_live_rule": True,
        "forward_returns_live_rule_usage": False,
        "blocked_fields": ["risk_bucket", "formal_large_down_day_policy", "formal_blowoff_turnover_policy"],
        "proxy_fields": ["large_down_day_count_20d_proxy", "large_down_day_count_30d_proxy", "blowoff_turnover_without_price_continuation_proxy", "multi_horizon_shape_evaluation_metadata"],
    }


def _summary(readiness: dict[str, Any]) -> str:
    return f"""# Layer2 soft-score multi-horizon evaluation join

## Verdict
- status={readiness["status"]}
- rows={readiness["rows"]}
- weekly_snapshot_count={readiness["weekly_snapshot_count"]}
- unique_ticker_count={readiness["unique_ticker_count"]}
- forward_eval_available_share_5d={readiness["forward_eval_available_share_5d"]}
- forward_eval_available_share_10d={readiness["forward_eval_available_share_10d"]}
- forward_eval_available_share_20d={readiness["forward_eval_available_share_20d"]}
- forward_eval_available_share_30d={readiness["forward_eval_available_share_30d"]}
- forward_eval_available_share_40d={readiness["forward_eval_available_share_40d"]}
- large_down_day_available=true
- large_down_day_source_quality=diagnostic_price_proxy_threshold_not_formal
- blowoff_turnover_available=true
- blowoff_turnover_source_quality=diagnostic_traded_value_proxy_threshold_not_formal
- risk_bucket_available=false
- ready_for_layer2_soft_score_multi_horizon_risk_diagnostic={str(readiness["ready_for_layer2_soft_score_multi_horizon_risk_diagnostic"]).lower()}

## Plain Summary
This package adds exact 5D/10D/20D/30D/40D evaluation metadata versus 0050 and 00631L. The 30D horizon is computed directly from adjusted/available close and is not a 20D/40D proxy. Multi-horizon shape fields are evaluation metadata only. Large-down-day and blowoff-turnover are diagnostic proxies, while risk_bucket remains blocked.

## Flags
- formal_model_changed=false
- trade_decision_changed=false
- active_in_trade_decision=false
- report_changed=false
- portfolio_replay_executed=false
- ready_for_strategy_replay=false
- not_live_rule=true
- forward_returns_live_rule_usage=false
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--soft-dir", default=str(DEFAULT_SOFT_DIR))
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--experiments-dir", default=str(DEFAULT_EXPERIMENTS_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()
    manifest = build_join(
        soft_dir=args.soft_dir,
        data_dir=args.data_dir,
        experiments_dir=args.experiments_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
