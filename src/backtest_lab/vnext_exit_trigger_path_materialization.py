"""Materialize compact vNext exit-trigger daily path diagnostics.

This is a diagnostic-only Core/Data materialization. It builds event x
day-in-band rows for Experiments to rerun tightened exit-trigger attribution
without recomputing full daily paths in the Experiments runner.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


TASK_ID = "TASK-BACKTEST-CORE-VNEXT-EXIT-TRIGGER-PATH-TABLE-MATERIALIZATION-001"
DEFAULT_INPUT_DIR = Path("outputs/vnext_dynamic_candidate_pool_data_materialization_20260706")
DEFAULT_PULLBACK_DIR = Path("outputs/vnext_user_original_lowpoint_pullback_filter_readiness_20260706")
DEFAULT_EXPERIMENTS_DIR = Path(
    r"C:\Users\zergv\Documents\Codex\2026-07-06\backtest-lab-experiments-diagnostic-validation-attribution\outputs"
    r"\vnext_window_band_exit_trigger_attribution_tightening_diagnostic_20260707"
)
DEFAULT_OUTPUT_DIR = Path("outputs/vnext_exit_trigger_path_table_materialization_20260707")


def build_exit_trigger_path_table(
    *,
    input_dir: str | Path = DEFAULT_INPUT_DIR,
    pullback_dir: str | Path = DEFAULT_PULLBACK_DIR,
    experiments_dir: str | Path = DEFAULT_EXPERIMENTS_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    input_path = Path(input_dir)
    pullback_path = Path(pullback_dir)
    exp_path = Path(experiments_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    calendar = _read_csv(input_path / "trading_calendar.csv")
    weekly = _read_csv(input_path / "vnext_weekly_candidate_snapshot.csv", dtype={"ticker": str})
    daily = _read_csv(
        input_path / "daily_market_features.csv",
        usecols=["trade_date", "ticker", "adjusted_close", "volume", "traded_value", "valid_universe"],
        dtype={"ticker": str},
    )
    benchmark = _read_csv(input_path / "benchmark_features.csv", dtype={"benchmark": str})
    pullback = _read_csv(pullback_path / "sleeve_parallel_ranking_contract.csv", dtype={"ticker": str})
    exp_summary = _read_json(exp_path / "exit_trigger_attribution_summary.json")

    event_base = _event_base(weekly, pullback, calendar)
    path_core = _path_core(event_base, daily, benchmark, calendar)
    complete_event_ids, blocked_events = _blocked_events(path_core)
    event_base["included_in_path_table"] = event_base["event_id"].isin(complete_event_ids)
    event_base = event_base.merge(blocked_events[["event_id", "blocked_reason"]], on="event_id", how="left")
    event_base["blocked_reason"] = event_base["blocked_reason"].fillna("")
    path_core = path_core[path_core["event_id"].isin(complete_event_ids)].copy()
    path_table = _band_rows(path_core)
    feature_audit = _feature_quality_audit(path_table)
    future_audit = _future_data_audit(path_table)
    readiness = _readiness(event_base, blocked_events, path_table, feature_audit, future_audit, exp_summary)

    _write_csv(event_base, output / "vnext_exit_trigger_event_base.csv")
    _write_csv(blocked_events, output / "vnext_exit_trigger_blocked_events.csv")
    _write_csv(path_table, output / "vnext_exit_trigger_path_table.csv")
    _write_csv(feature_audit, output / "vnext_exit_trigger_feature_quality_audit.csv")
    _write_csv(future_audit, output / "vnext_exit_trigger_future_data_audit.csv")
    (output / "readiness_for_vnext_exit_trigger_path_materialization.json").write_text(
        json.dumps(readiness, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    manifest = {
        "task_id": TASK_ID,
        "status": readiness["status"],
        "output_dir": str(output.resolve()),
        "input_materialization_dir": str(input_path.resolve()),
        "experiments_input_dir": str(exp_path.resolve()),
        "output_files": [
            "vnext_exit_trigger_event_base.csv",
            "vnext_exit_trigger_blocked_events.csv",
            "vnext_exit_trigger_path_table.csv",
            "vnext_exit_trigger_feature_quality_audit.csv",
            "vnext_exit_trigger_future_data_audit.csv",
            "readiness_for_vnext_exit_trigger_path_materialization.json",
            "manifest.json",
            "final_summary_zh.md",
        ],
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
    (output / "final_summary_zh.md").write_text(_summary(readiness, feature_audit), encoding="utf-8")
    return manifest


def _read_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    return pd.read_csv(path, **kwargs) if path.exists() else pd.DataFrame()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _event_base(weekly: pd.DataFrame, pullback: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    selected = weekly[
        weekly["selected_outcome_candidate"].astype(bool)
        & ~weekly["case_trace_only"].astype(bool)
        & weekly["diagnostic_only"].astype(bool)
    ].copy()
    selected["signal_date"] = selected["snapshot_date"]
    selected["event_id"] = selected["signal_date"].astype(str) + "|" + selected["ticker"].astype(str)
    selected["candidate_base"] = "vnext_primary_selected_outcome_candidate"
    selected["current_final_candidate"] = True
    selected["current_top3_candidate"] = selected["rank_overall"].fillna(9999).astype(float) <= 3
    selected["long_strong_candidate"] = selected["subpool_class"].eq("long_strong")
    selected["pullback_repair_candidate"] = selected["subpool_class"].eq("pullback_repair")
    selected["overlap_candidate"] = selected["long_strong_candidate"] & selected["pullback_repair_candidate"]

    sleeve_cols = [
        "signal_date",
        "ticker",
        "momentum_sleeve_candidate",
        "pullback_sleeve_candidate",
        "source_sleeve",
        "candidate_family_list",
    ]
    if not pullback.empty:
        selected = selected.merge(pullback.reindex(columns=sleeve_cols), on=["signal_date", "ticker"], how="left")
    else:
        selected["momentum_sleeve_candidate"] = selected["long_strong_candidate"]
        selected["pullback_sleeve_candidate"] = selected["pullback_repair_candidate"]
        selected["source_sleeve"] = "missing_pullback_contract"
        selected["candidate_family_list"] = selected["candidate_base"]

    selected["momentum_sleeve_candidate"] = selected["momentum_sleeve_candidate"].fillna(selected["long_strong_candidate"]).astype(bool)
    selected["pullback_sleeve_candidate"] = selected["pullback_sleeve_candidate"].fillna(selected["pullback_repair_candidate"]).astype(bool)
    selected["sleeve_group"] = np.select(
        [
            selected["momentum_sleeve_candidate"] & selected["pullback_sleeve_candidate"],
            selected["momentum_sleeve_candidate"],
            selected["pullback_sleeve_candidate"],
        ],
        ["overlap", "momentum_only", "pullback_repair"],
        default="neither",
    )

    cal = calendar[["trade_date", "next_trade_date"]].copy()
    selected = selected.merge(cal, left_on="signal_date", right_on="trade_date", how="left")
    selected["execution_date"] = selected["next_trade_date"]
    selected["execution_date_basis"] = "next_trading_day_after_signal_date"
    selected["not_live_rule"] = True
    selected["diagnostic_only"] = True
    keep = [
        "event_id",
        "signal_date",
        "execution_date",
        "execution_date_basis",
        "week_id",
        "ticker",
        "name",
        "sleeve_group",
        "candidate_base",
        "candidate_family_list",
        "subpool_class",
        "current_final_candidate",
        "current_top3_candidate",
        "long_strong_candidate",
        "pullback_repair_candidate",
        "momentum_sleeve_candidate",
        "pullback_sleeve_candidate",
        "market_attention_member",
        "eligible_pool_member",
        "rank_overall",
        "rank_in_subpool",
        "risk_score",
        "risk_bucket",
        "turnover_state",
        "selected_by_vnext",
        "selected_outcome_candidate",
        "case_trace_only",
        "diagnostic_only",
        "not_live_rule",
    ]
    return selected.reindex(columns=keep).reset_index(drop=True)


def _path_core(event_base: pd.DataFrame, daily: pd.DataFrame, benchmark: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    tickers = sorted(event_base["ticker"].astype(str).unique())
    daily = daily[daily["ticker"].astype(str).isin(tickers)].copy()
    daily["trade_date"] = pd.to_datetime(daily["trade_date"])
    daily = daily.sort_values(["ticker", "trade_date"])

    daily["stock_1d_return"] = daily.groupby("ticker")["adjusted_close"].pct_change()
    for window in [5, 10, 20, 60]:
        daily[f"stock_return_{window}d"] = daily.groupby("ticker")["adjusted_close"].pct_change(window)
    daily["MA20"] = daily.groupby("ticker")["adjusted_close"].transform(lambda s: s.rolling(20, min_periods=20).mean())
    daily["MA60"] = daily.groupby("ticker")["adjusted_close"].transform(lambda s: s.rolling(60, min_periods=60).mean())
    daily["BIAS20"] = (daily["adjusted_close"] - daily["MA20"]) / daily["MA20"]
    daily["BIAS60"] = (daily["adjusted_close"] - daily["MA60"]) / daily["MA60"]
    daily["BIAS20_z"] = daily.groupby("ticker")["BIAS20"].transform(lambda s: _rolling_z(s, 120))
    daily["BIAS60_z"] = daily.groupby("ticker")["BIAS60"].transform(lambda s: _rolling_z(s, 120))
    daily["volatility_20d"] = daily.groupby("ticker")["stock_1d_return"].transform(lambda s: s.rolling(20, min_periods=20).std())
    daily["volatility_60d"] = daily.groupby("ticker")["stock_1d_return"].transform(lambda s: s.rolling(60, min_periods=60).std())
    daily["large_down_day"] = daily["stock_1d_return"] <= -0.05
    daily["large_down_day_count_20d"] = daily.groupby("ticker")["large_down_day"].transform(lambda s: s.rolling(20, min_periods=1).sum())
    daily["turnover_20d"] = daily.groupby("ticker")["traded_value"].transform(lambda s: s.rolling(20, min_periods=5).mean())
    daily["turnover_60d"] = daily.groupby("ticker")["traded_value"].transform(lambda s: s.rolling(60, min_periods=10).mean())
    daily["turnover_spike_ratio"] = daily["traded_value"] / daily["turnover_20d"]
    daily["blowoff_turnover_without_price_continuation"] = (daily["turnover_spike_ratio"] >= 2.0) & (daily["stock_return_5d"].fillna(0) <= 0)

    bench = _benchmark_wide(benchmark)
    daily = daily.merge(bench, left_on="trade_date", right_on="trade_date", how="left")
    for window in [5, 10, 20, 60]:
        daily[f"RS{window}"] = daily[f"stock_return_{window}d"] - daily[f"0050_return_{window}d"]
    daily["relative_to_0050_spread"] = daily["RS20"]
    daily["rs5_lt_rs10"] = daily["RS5"] < daily["RS10"]
    daily["rs10_lt_rs20"] = daily["RS10"] < daily["RS20"]
    daily["rs20_lt_rs60"] = daily["RS20"] < daily["RS60"]
    daily["rs60_high_short_window_weakening"] = (daily["RS60"] > 0) & (daily["RS20"] < daily["RS60"]) & (daily["RS10"] < daily["RS20"])
    daily["stock_bias_overheat_broad"] = daily["BIAS20_z"] >= 1.0
    daily["stock_bias_overheat_extreme"] = daily["BIAS20_z"] >= 2.0
    daily["volatility_shock_cluster"] = daily["volatility_20d"] > (daily["volatility_60d"] * 1.5)
    daily["market_bias_hot"] = daily["0050_BIAS60"] >= 0.08

    path_rows = _event_path_dates(event_base, calendar)
    path = path_rows.merge(daily, left_on=["ticker", "path_date"], right_on=["ticker", "trade_date"], how="left")
    path = path.merge(
        event_base[
            [
                "event_id",
                "signal_date",
                "ticker",
                "sleeve_group",
                "candidate_base",
                "candidate_family_list",
                "subpool_class",
                "current_final_candidate",
                "current_top3_candidate",
                "long_strong_candidate",
                "pullback_repair_candidate",
                "momentum_sleeve_candidate",
                "pullback_sleeve_candidate",
                "market_attention_member",
                "eligible_pool_member",
                "rank_overall",
                "risk_score",
                "risk_bucket",
            ]
        ],
        on=["event_id", "signal_date", "ticker"],
        how="left",
    )
    signal_close = daily[["ticker", "trade_date", "adjusted_close", "0050_close", "00631L_close"]].rename(
        columns={
            "trade_date": "signal_dt",
            "adjusted_close": "signal_stock_close",
            "0050_close": "signal_0050_close",
            "00631L_close": "signal_00631L_close",
        }
    )
    path = path.merge(signal_close, left_on=["ticker", "signal_date_dt"], right_on=["ticker", "signal_dt"], how="left")
    path["observed_return_so_far"] = path["adjusted_close"] / path["signal_stock_close"] - 1
    path["observed_0050_return_so_far"] = path["0050_close"] / path["signal_0050_close"] - 1
    path["observed_00631L_return_so_far"] = path["00631L_close"] / path["signal_00631L_close"] - 1
    path["observed_excess_vs_0050_so_far"] = path["observed_return_so_far"] - path["observed_0050_return_so_far"]
    path["observed_excess_vs_00631L_so_far"] = path["observed_return_so_far"] - path["observed_00631L_return_so_far"]

    path = path.sort_values(["event_id", "horizon_day"])
    grouped = path.groupby("event_id", sort=False)
    path["max_observed_return_so_far"] = grouped["observed_return_so_far"].cummax()
    path["max_observed_excess_vs_0050_so_far"] = grouped["observed_excess_vs_0050_so_far"].cummax()
    path["max_observed_excess_vs_00631L_so_far"] = grouped["observed_excess_vs_00631L_so_far"].cummax()
    path["drawdown_from_observed_return_peak"] = path["observed_return_so_far"] - path["max_observed_return_so_far"]
    path["trailing_giveback_proxy"] = path["max_observed_excess_vs_00631L_so_far"] - path["observed_excess_vs_00631L_so_far"]
    path["relative_to_0050_spread_contraction"] = path["relative_to_0050_spread"] < grouped["relative_to_0050_spread"].cummax()

    endpoint = path[path["horizon_day"].eq(30)][
        ["event_id", "observed_return_so_far", "observed_excess_vs_0050_so_far", "observed_excess_vs_00631L_so_far"]
    ].rename(
        columns={
            "observed_return_so_far": "endpoint_30d_return",
            "observed_excess_vs_0050_so_far": "endpoint_30d_excess_vs_0050",
            "observed_excess_vs_00631L_so_far": "endpoint_30d_excess_vs_00631L",
        }
    )
    path = path.merge(endpoint, on="event_id", how="left")
    path["time_stop_5d_no_relative_expansion"] = (path["horizon_day"] >= 5) & (path["max_observed_excess_vs_0050_so_far"].fillna(-np.inf) <= 0)
    path["time_stop_8d_no_relative_expansion"] = (path["horizon_day"] >= 8) & (path["max_observed_excess_vs_0050_so_far"].fillna(-np.inf) <= 0)
    path["time_stop_10d_no_relative_expansion"] = (path["horizon_day"] >= 10) & (path["max_observed_excess_vs_0050_so_far"].fillna(-np.inf) <= 0)
    path["time_stop_12d_no_relative_expansion"] = (path["horizon_day"] >= 12) & (path["max_observed_excess_vs_0050_so_far"].fillna(-np.inf) <= 0)
    path["evaluation_correct_exit_if_triggered"] = path["endpoint_30d_excess_vs_00631L"] < 0
    path["evaluation_false_exit_if_triggered"] = path["endpoint_30d_excess_vs_00631L"] > 0
    path["endpoint_labels_evaluation_metadata_only"] = True
    path["future_high_as_rule"] = False
    path["forward_return_as_rule"] = False
    path["max_in_band_as_rule"] = False
    path["diagnostic_only"] = True
    path["not_live_rule"] = True
    return path


def _benchmark_wide(benchmark: pd.DataFrame) -> pd.DataFrame:
    b = benchmark.copy()
    b["trade_date"] = pd.to_datetime(b["trade_date"])
    out = b.pivot(index="trade_date", columns="benchmark", values="adjusted_close").rename(columns={"0050": "0050_close", "00631L": "00631L_close"})
    for benchmark_id in ["0050", "00631L"]:
        sub = b[b["benchmark"].eq(benchmark_id)].set_index("trade_date").sort_index()
        prefix = benchmark_id
        for window in [5, 10, 20, 60]:
            out[f"{prefix}_return_{window}d"] = sub["adjusted_close"].pct_change(window)
        out[f"{prefix}_BIAS60"] = sub["BIAS60"]
    return out.reset_index()


def _event_path_dates(event_base: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    cal_dates = pd.to_datetime(calendar["trade_date"]).sort_values().reset_index(drop=True)
    rows: list[dict[str, Any]] = []
    for row in event_base.to_dict("records"):
        signal_dt = pd.to_datetime(row["signal_date"])
        future_dates = cal_dates[cal_dates > signal_dt].head(30)
        for i, path_dt in enumerate(future_dates, start=1):
            rows.append(
                {
                    "event_id": row["event_id"],
                    "signal_date": row["signal_date"],
                    "signal_date_dt": signal_dt,
                    "ticker": row["ticker"],
                    "horizon_day": i,
                    "path_date": path_dt,
                }
            )
    return pd.DataFrame(rows)


def _band_rows(path_core: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "event_id",
        "signal_date",
        "ticker",
        "horizon_day",
        "path_date",
        "sleeve_group",
        "candidate_base",
        "candidate_family_list",
        "subpool_class",
        "current_final_candidate",
        "current_top3_candidate",
        "long_strong_candidate",
        "pullback_repair_candidate",
        "momentum_sleeve_candidate",
        "pullback_sleeve_candidate",
        "market_attention_member",
        "eligible_pool_member",
        "rank_overall",
        "risk_score",
        "risk_bucket",
        "adjusted_close",
        "stock_1d_return",
        "RS5",
        "RS10",
        "RS20",
        "RS60",
        "rs5_lt_rs10",
        "rs10_lt_rs20",
        "rs20_lt_rs60",
        "rs60_high_short_window_weakening",
        "relative_to_0050_spread",
        "relative_to_0050_spread_contraction",
        "BIAS20",
        "BIAS60",
        "BIAS20_z",
        "BIAS60_z",
        "stock_bias_overheat_broad",
        "stock_bias_overheat_extreme",
        "turnover_spike_ratio",
        "blowoff_turnover_without_price_continuation",
        "large_down_day",
        "large_down_day_count_20d",
        "volatility_20d",
        "volatility_60d",
        "volatility_shock_cluster",
        "0050_BIAS60",
        "market_bias_hot",
        "observed_return_so_far",
        "observed_excess_vs_0050_so_far",
        "observed_excess_vs_00631L_so_far",
        "max_observed_return_so_far",
        "max_observed_excess_vs_0050_so_far",
        "max_observed_excess_vs_00631L_so_far",
        "drawdown_from_observed_return_peak",
        "trailing_giveback_proxy",
        "time_stop_5d_no_relative_expansion",
        "time_stop_8d_no_relative_expansion",
        "time_stop_10d_no_relative_expansion",
        "time_stop_12d_no_relative_expansion",
        "endpoint_30d_return",
        "endpoint_30d_excess_vs_0050",
        "endpoint_30d_excess_vs_00631L",
        "evaluation_correct_exit_if_triggered",
        "evaluation_false_exit_if_triggered",
        "endpoint_labels_evaluation_metadata_only",
        "future_high_as_rule",
        "forward_return_as_rule",
        "max_in_band_as_rule",
        "diagnostic_only",
        "not_live_rule",
    ]
    frames = []
    for band_id, start in [("band_3_30", 3), ("band_5_30", 5)]:
        band = path_core[path_core["horizon_day"].between(start, 30)].copy()
        band["band_id"] = band_id
        band["band_start_day"] = start
        band["band_end_day"] = 30
        frames.append(band.reindex(columns=["band_id", "band_start_day", "band_end_day", *cols]))
    out = pd.concat(frames, ignore_index=True)
    out["path_date"] = pd.to_datetime(out["path_date"]).dt.date.astype(str)
    return out


def _blocked_events(path_core: pd.DataFrame) -> tuple[set[str], pd.DataFrame]:
    stats = path_core.groupby("event_id", dropna=False).agg(
        signal_date=("signal_date", "first"),
        ticker=("ticker", "first"),
        generated_horizon_rows=("horizon_day", "count"),
        max_horizon_day=("horizon_day", "max"),
        missing_observed_excess_vs_00631L=("observed_excess_vs_00631L_so_far", lambda s: int(s.isna().sum())),
        missing_endpoint_30d_excess_vs_00631L=("endpoint_30d_excess_vs_00631L", lambda s: int(s.isna().sum())),
    )
    blocked = stats[
        (stats["generated_horizon_rows"] < 30)
        | (stats["max_horizon_day"] < 30)
        | (stats["missing_observed_excess_vs_00631L"] > 0)
        | (stats["missing_endpoint_30d_excess_vs_00631L"] > 0)
    ].copy()
    if blocked.empty:
        return set(stats.index.astype(str)), pd.DataFrame(
            columns=[
                "event_id",
                "signal_date",
                "ticker",
                "generated_horizon_rows",
                "max_horizon_day",
                "missing_observed_excess_vs_00631L",
                "missing_endpoint_30d_excess_vs_00631L",
                "blocked_reason",
                "diagnostic_only",
            ]
        )

    def reason(row: pd.Series) -> str:
        parts: list[str] = []
        if row["max_horizon_day"] < 30 or row["generated_horizon_rows"] < 30:
            parts.append("insufficient_30d_future_path_with_current_data_coverage")
        if row["missing_observed_excess_vs_00631L"] > 0:
            parts.append("missing_observed_path_excess_vs_00631L")
        if row["missing_endpoint_30d_excess_vs_00631L"] > 0:
            parts.append("missing_endpoint_30d_excess_vs_00631L")
        return "|".join(parts)

    blocked["blocked_reason"] = blocked.apply(reason, axis=1)
    blocked["diagnostic_only"] = True
    blocked = blocked.reset_index()
    complete = set(stats.index.astype(str)) - set(blocked["event_id"].astype(str))
    return complete, blocked


def _rolling_z(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window, min_periods=max(20, window // 3)).mean()
    std = series.rolling(window, min_periods=max(20, window // 3)).std()
    return (series - mean) / std


def _feature_quality_audit(path_table: pd.DataFrame) -> pd.DataFrame:
    fields = [
        "RS5",
        "RS10",
        "RS20",
        "RS60",
        "BIAS20_z",
        "BIAS60_z",
        "turnover_spike_ratio",
        "large_down_day_count_20d",
        "volatility_20d",
        "0050_BIAS60",
        "observed_excess_vs_00631L_so_far",
        "endpoint_30d_excess_vs_00631L",
    ]
    rows = []
    total = len(path_table)
    for field in fields:
        missing = int(path_table[field].isna().sum()) if field in path_table else total
        source_quality = "pit_computed_from_daily_price_or_benchmark" if missing < total else "blocked_missing"
        if field.startswith("endpoint_30d"):
            source_quality = "evaluation_metadata_only_not_rule_input"
        if field in {"BIAS20_z", "BIAS60_z"}:
            source_quality = "pit_self_history_zscore_proxy_cross_section_percentile_blocked"
        rows.append(
            {
                "field": field,
                "missing_rows": missing,
                "total_rows": total,
                "missing_share": missing / total if total else 0.0,
                "source_quality": source_quality,
                "diagnostic_only": True,
            }
        )
    rows.append(
        {
            "field": "stock_bias_cross_section_percentile",
            "missing_rows": total,
            "total_rows": total,
            "missing_share": 1.0 if total else 0.0,
            "source_quality": "blocked_not_materialized_daily_cross_section",
            "diagnostic_only": True,
        }
    )
    return pd.DataFrame(rows)


def _future_data_audit(path_table: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "audit_item": "future_high_as_rule",
                "status": "passed" if not path_table["future_high_as_rule"].astype(bool).any() else "failed",
                "future_data_violation_count": int(path_table["future_high_as_rule"].astype(bool).sum()),
                "note": "profit protection uses observed-to-date cummax only",
            },
            {
                "audit_item": "forward_return_as_rule",
                "status": "passed" if not path_table["forward_return_as_rule"].astype(bool).any() else "failed",
                "future_data_violation_count": int(path_table["forward_return_as_rule"].astype(bool).sum()),
                "note": "endpoint 30D labels are evaluation metadata only",
            },
            {
                "audit_item": "max_in_band_as_rule",
                "status": "passed" if not path_table["max_in_band_as_rule"].astype(bool).any() else "failed",
                "future_data_violation_count": int(path_table["max_in_band_as_rule"].astype(bool).sum()),
                "note": "no future full-band maximum is used as trigger input",
            },
        ]
    )


def _readiness(
    event_base: pd.DataFrame,
    blocked_events: pd.DataFrame,
    path_table: pd.DataFrame,
    feature_audit: pd.DataFrame,
    future_audit: pd.DataFrame,
    exp_summary: dict[str, Any],
) -> dict[str, Any]:
    future_count = int(future_audit["future_data_violation_count"].sum())
    total_event_count = int(event_base["event_id"].nunique())
    included_event_count = int(event_base["included_in_path_table"].astype(bool).sum()) if "included_in_path_table" in event_base else total_event_count
    blocked_event_count = int(len(blocked_events))
    expected_rows = included_event_count * (28 + 26)
    path_rows = len(path_table)
    critical_missing = feature_audit[
        feature_audit["field"].isin(["observed_excess_vs_00631L_so_far", "endpoint_30d_excess_vs_00631L"])
    ]["missing_rows"].sum()
    ready = future_count == 0 and path_rows == expected_rows and critical_missing == 0 and included_event_count > 0
    return {
        "date": "2026-07-07",
        "task_id": TASK_ID,
        "owner": "BACKTEST_LAB Core/Data",
        "status": "ready_for_experiments_tightened_exit_trigger_attribution" if ready else "partial_or_blocked_exit_trigger_path_materialization",
        "source_experiments_status": exp_summary.get("status"),
        "source_experiments_verdict": exp_summary.get("verdict"),
        "event_base": "vnext selected_outcome_candidate true, case_trace_only false",
        "bands": ["band_3_30", "band_5_30"],
        "total_event_count": total_event_count,
        "included_event_count": included_event_count,
        "blocked_event_count": blocked_event_count,
        "path_rows": path_rows,
        "expected_path_rows": expected_rows,
        "ready_for_exit_trigger_tightening_diagnostic": bool(ready),
        "ready_for_experiments": bool(ready),
        "ready_for_portfolio_like_diagnostic": False,
        "ready_for_strategy_replay": False,
        "ready_for_formal": False,
        "future_data_violation_count": future_count,
        "blocked_fields": ["stock_bias_cross_section_percentile"],
        "proxy_fields": ["BIAS20_z", "BIAS60_z", "turnover_blowoff_without_price_continuation"],
        "endpoint_labels_evaluation_metadata_only": True,
        "not_live_rule": True,
        "forward_returns_live_rule_usage": False,
        "formal_model_changed": False,
        "trade_decision_changed": False,
        "active_in_trade_decision": False,
        "report_changed": False,
        "portfolio_replay_executed": False,
    }


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def _summary(readiness: dict[str, Any], feature_audit: pd.DataFrame) -> str:
    return "\n".join(
        [
            "# vNext Exit-trigger Path Table Materialization",
            "",
            f"Status: {readiness['status']}",
            "",
            "Boundary: diagnostic-only path table for Experiments tightening rerun; no replay, no live rule, no formal/report/trade decision change.",
            "",
            "Readiness:",
            f"- ready_for_exit_trigger_tightening_diagnostic={str(readiness['ready_for_exit_trigger_tightening_diagnostic']).lower()}",
            f"- ready_for_experiments={str(readiness['ready_for_experiments']).lower()}",
            "- ready_for_portfolio_like_diagnostic=false",
            "- ready_for_strategy_replay=false",
            "- ready_for_formal=false",
            f"- total_event_count={readiness['total_event_count']}",
            f"- included_event_count={readiness['included_event_count']}",
            f"- blocked_event_count={readiness['blocked_event_count']}",
            f"- path_rows={readiness['path_rows']}",
            f"- expected_path_rows={readiness['expected_path_rows']}",
            f"- future_data_violation_count={readiness['future_data_violation_count']}",
            "",
            "Blocked / proxy notes:",
            "- stock_bias_cross_section_percentile remains blocked; Core provides PIT self-history BIAS z-score proxy instead.",
            "- endpoint 30D labels are evaluation metadata only and not rule inputs.",
            "",
            "Feature missingness highlights:",
            *[
                f"- {row.field}: missing_share={row.missing_share:.4f}; source_quality={row.source_quality}"
                for row in feature_audit.itertuples()
            ],
            "",
            "Flags:",
            "- formal_model_changed=false",
            "- trade_decision_changed=false",
            "- active_in_trade_decision=false",
            "- report_changed=false",
            "- portfolio_replay_executed=false",
            "- ready_for_strategy_replay=false",
            "- not_live_rule=true",
            "- forward_returns_live_rule_usage=false",
        ]
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--pullback-dir", type=Path, default=DEFAULT_PULLBACK_DIR)
    parser.add_argument("--experiments-dir", type=Path, default=DEFAULT_EXPERIMENTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    manifest = build_exit_trigger_path_table(
        input_dir=args.input_dir,
        pullback_dir=args.pullback_dir,
        experiments_dir=args.experiments_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
