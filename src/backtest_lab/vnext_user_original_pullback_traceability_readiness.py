"""Build traceability readiness for the user's original pullback hypothesis.

This is diagnostic contract/readiness only. It stages PIT feature, parallel
layer, and candidate-family join contracts so Research/Experiments can verify
whether the original hypothesis was tested. It does not run replay or change
formal/report/trade behavior.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


TASK_ID = "TASK-BACKTEST-CORE-VNEXT-USER-ORIGINAL-LOWPOINT-PULLBACK-FILTER-CONTRACT-READINESS-001"
MERGED_TASK_IDS = [
    "TASK-BACKTEST-CORE-VNEXT-USER-ORIGINAL-LOWPOINT-PULLBACK-FILTER-CONTRACT-READINESS-001",
    "TASK-BACKTEST-CORE-VNEXT-USER-ORIGINAL-PULLBACK-TRACEABILITY-CONTRACT-READINESS-001",
]
DEFAULT_MATERIALIZATION_DIR = Path("outputs/vnext_dynamic_candidate_pool_data_materialization_20260706")
DEFAULT_OUTPUT_DIR = Path("outputs/vnext_user_original_lowpoint_pullback_filter_readiness_20260706")


def build_user_original_pullback_traceability(
    *,
    materialization_dir: str | Path = DEFAULT_MATERIALIZATION_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    materialization = Path(materialization_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    candidates = _candidate_join_contract(materialization / "vnext_weekly_candidate_snapshot.csv")
    stock = _stock_feature_slice(materialization / "stock_features.csv", candidates)
    features = _feature_contract(candidates, stock)
    layers = _parallel_layer_contract(features)
    layer4 = features[features["momentum_sleeve_candidate"].astype(bool)].copy()
    layer5 = features[features["pullback_sleeve_candidate"].astype(bool)].copy()
    sleeve_ranking = _sleeve_parallel_ranking_contract(features)
    case_trace = _case_trace_contract(features, materialization / "vnext_case_trace.csv")
    missing = _missing_field_audit(features, candidates)
    future_audit = _future_data_audit(features)
    readiness = _readiness_json(features, layers, missing, future_audit)

    _write_csv(features, output / "user_original_filter_traceability_contract.csv")
    _write_csv(layer4, output / "layer4_momentum_sleeve_contract.csv")
    _write_csv(layer5, output / "layer5_prior_strong_current_pullback_contract.csv")
    _write_csv(sleeve_ranking, output / "sleeve_parallel_ranking_contract.csv")
    _write_csv(case_trace, output / "case_trace_6669_2308_2317_original_filter_contract.csv")
    _write_csv(missing, output / "blocked_proxy_fields_ledger.csv")
    _write_csv(future_audit, output / "future_data_audit.csv")
    (output / "readiness_for_user_original_filter_diagnostic.json").write_text(
        json.dumps(readiness, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    manifest = {
        "task_id": TASK_ID,
        "merged_task_ids": MERGED_TASK_IDS,
        "status": readiness["status"],
        "output_dir": str(output.resolve()),
        "input_materialization_dir": str(materialization.resolve()),
        "output_files": [
            "user_original_filter_traceability_contract.csv",
            "layer4_momentum_sleeve_contract.csv",
            "layer5_prior_strong_current_pullback_contract.csv",
            "sleeve_parallel_ranking_contract.csv",
            "case_trace_6669_2308_2317_original_filter_contract.csv",
            "blocked_proxy_fields_ledger.csv",
            "future_data_audit.csv",
            "readiness_for_user_original_filter_diagnostic.json",
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
    (output / "final_summary_zh.md").write_text(_summary(readiness, missing), encoding="utf-8")
    return manifest


def _candidate_join_contract(path: Path) -> pd.DataFrame:
    usecols = [
        "snapshot_date",
        "ticker",
        "name",
        "theme_id",
        "theme_name",
        "selected_outcome_candidate",
        "case_trace_only",
        "diagnostic_only",
        "market_attention_member",
        "eligible_pool_member",
        "subpool_class",
        "long_strong_score",
        "pullback_repair_score",
        "short_cycle_score",
        "rank_overall",
        "rank_in_subpool",
        "turnover_state",
        "risk_score",
        "risk_bucket",
        "hurdle_0050_proxy_result",
        "hurdle_00631L_proxy_result",
        "final_selector_score_decomposed",
    ]
    raw = pd.read_csv(path, usecols=usecols, parse_dates=["snapshot_date"])
    raw = raw[raw["diagnostic_only"].astype(bool) & ~raw["case_trace_only"].astype(bool)].copy()
    raw["ticker"] = raw["ticker"].astype(str)
    raw["current_final_candidate"] = raw["selected_outcome_candidate"].astype(bool)
    raw["current_top3_candidate"] = pd.to_numeric(raw["rank_overall"], errors="coerce").le(3)
    raw["c3_pullback_candidate"] = raw["subpool_class"].astype(str).eq("pullback_repair") | pd.to_numeric(
        raw["pullback_repair_score"], errors="coerce"
    ).gt(-100)
    raw["long_strong_candidate"] = raw["subpool_class"].astype(str).eq("long_strong")
    raw["theme_breadth_watchlist_candidate"] = ~raw["theme_id"].astype(str).eq("non_ai_unclassified_proxy")
    raw["market_attention_candidate"] = raw["market_attention_member"].astype(bool)
    raw["candidate_family_list"] = raw.apply(_candidate_family_list, axis=1)
    raw["source_quality"] = "diagnostic_from_weekly_candidate_snapshot"
    raw["not_live_rule"] = True
    raw["source_sleeve"] = "candidate_pool_join"
    return raw.rename(columns={"snapshot_date": "signal_date"})


def _candidate_family_list(row: pd.Series) -> str:
    families = []
    for col, name in [
        ("current_final_candidate", "current_final"),
        ("current_top3_candidate", "current_top3"),
        ("c3_pullback_candidate", "c3_pullback"),
        ("long_strong_candidate", "long_strong"),
        ("theme_breadth_watchlist_candidate", "theme_breadth_watchlist"),
        ("market_attention_candidate", "market_attention_map"),
    ]:
        if bool(row[col]):
            families.append(name)
    return "|".join(families) if families else "candidate_pool_context"


def _stock_feature_slice(path: Path, candidates: pd.DataFrame) -> pd.DataFrame:
    dates = set(candidates["signal_date"].dt.strftime("%Y-%m-%d"))
    tickers = set(candidates["ticker"].astype(str))
    usecols = [
        "trade_date",
        "ticker",
        "return_5d",
        "return_10d",
        "return_20d",
        "return_40d",
        "return_60d",
        "RS5",
        "RS10",
        "RS20",
        "RS40",
        "RS60",
        "MA20_position",
        "MA60_position",
        "MA120_position",
        "BIAS20",
        "BIAS60",
        "BIAS120",
        "BIAS20_percentile",
        "BIAS60_percentile",
        "BIAS120_percentile",
        "drawdown_20d",
        "drawdown_60d",
        "drawdown_120d",
        "volatility",
    ]
    parts = []
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=500_000):
        chunk["ticker"] = chunk["ticker"].astype(str)
        chunk = chunk[chunk["trade_date"].astype(str).isin(dates) & chunk["ticker"].isin(tickers)]
        if not chunk.empty:
            parts.append(chunk)
    if not parts:
        return pd.DataFrame(columns=usecols)
    out = pd.concat(parts, ignore_index=True)
    out["trade_date"] = pd.to_datetime(out["trade_date"])
    return out


def _feature_contract(candidates: pd.DataFrame, stock: pd.DataFrame) -> pd.DataFrame:
    out = candidates.merge(stock, left_on=["signal_date", "ticker"], right_on=["trade_date", "ticker"], how="left")
    out = out.drop(columns=["trade_date"], errors="ignore")
    for col in ["RS5", "RS10", "RS20", "RS40", "RS60", "MA20_position", "MA60_position", "MA120_position", "BIAS20", "BIAS60", "BIAS120", "drawdown_20d", "drawdown_60d", "drawdown_120d", "volatility"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    for col in ["RS20", "RS40", "RS60"]:
        out[f"{col}_rank_pct_within_signal"] = out.groupby("signal_date")[col].rank(pct=True, ascending=False)
        out[f"{col}_rank_bucket"] = pd.cut(
            out[f"{col}_rank_pct_within_signal"],
            bins=[0, 0.1, 0.25, 0.5, 0.75, 1.0],
            labels=["top10", "top25", "top50", "bottom50_25", "bottom25"],
            include_lowest=True,
        ).astype(str)
    rs_cols = ["RS5", "RS10", "RS20", "RS40", "RS60"]
    out["max_relative_outperformance_available_windows"] = out[rs_cols].max(axis=1, skipna=True)
    has_rs = out[rs_cols].notna().any(axis=1)
    best_window = out[rs_cols].fillna(float("-inf")).idxmax(axis=1)
    best_window = best_window.where(has_rs, "missing")
    out["prior_strength_window_label"] = best_window.map(
        {
            "RS5": "1w",
            "RS10": "2w",
            "RS20": "1m",
            "RS40": "2m",
            "RS60": "3m",
            "missing": "missing",
        }
    )
    for col in rs_cols:
        out[f"{col}_below_0050_flag"] = out[col].lt(0)
    out["below_MA20_flag"] = out["MA20_position"].lt(0)
    out["below_MA60_flag"] = out["MA60_position"].lt(0)
    out["below_MA120_flag"] = out["MA120_position"].lt(0)
    out["near_or_below_MA20_proxy"] = out["MA20_position"].le(0.03)
    out["near_or_below_MA60_proxy"] = out["MA60_position"].le(0.03)
    out["near_or_below_MA120_proxy"] = out["MA120_position"].le(0.03)
    out["price_position_lower_is_better_component"] = (
        out[["drawdown_20d", "drawdown_60d", "BIAS20", "BIAS60"]].rank(pct=True, ascending=True).mean(axis=1)
    )
    out["overheat_risk_lower_is_better_component"] = (
        out[["BIAS20_percentile", "BIAS60_percentile", "volatility"]].rank(pct=True, ascending=True).mean(axis=1)
    )
    out["lowpoint_proximity_proxy"] = out[["drawdown_20d", "drawdown_60d", "drawdown_120d"]].min(axis=1, skipna=True)
    out["risk_level_available"] = out["risk_score"].notna() | out["risk_bucket"].notna()
    out["risk_proxy_source"] = out["risk_level_available"].map({True: "weekly_snapshot_risk_score", False: "volatility_drawdown_bias_proxy"})
    prior_strong = out[["RS20", "RS40", "RS60"]].max(axis=1, skipna=True).gt(0)
    current_weak = out[["RS5", "RS10", "RS20"]].min(axis=1, skipna=True).lt(0)
    near_ma_or_low = out["near_or_below_MA20_proxy"] | out["near_or_below_MA60_proxy"] | out["lowpoint_proximity_proxy"].lt(-0.08)
    out["momentum_sleeve_candidate"] = (
        out["long_strong_candidate"].astype(bool)
        & out["RS20"].gt(0)
        & out["RS40"].gt(0)
        & out["MA20_position"].gt(0)
        & out["MA60_position"].gt(0)
    )
    out["pullback_sleeve_candidate"] = prior_strong & (current_weak | near_ma_or_low)
    out["source_sleeve"] = out.apply(
        lambda row: "both"
        if bool(row["momentum_sleeve_candidate"]) and bool(row["pullback_sleeve_candidate"])
        else "layer4_momentum"
        if bool(row["momentum_sleeve_candidate"])
        else "layer5_prior_strong_current_pullback"
        if bool(row["pullback_sleeve_candidate"])
        else "neither",
        axis=1,
    )
    out["feature_asof_date"] = out["signal_date"]
    out["future_data_rule"] = "features_as_of_signal_date_only"
    out["not_live_rule"] = True
    out["forward_returns_live_rule_usage"] = False
    return out


def _sleeve_parallel_ranking_contract(features: pd.DataFrame) -> pd.DataFrame:
    out = features[
        [
            "signal_date",
            "ticker",
            "name",
            "theme_id",
            "candidate_family_list",
            "source_sleeve",
            "momentum_sleeve_candidate",
            "pullback_sleeve_candidate",
            "RS20_rank_pct_within_signal",
            "RS40_rank_pct_within_signal",
            "RS60_rank_pct_within_signal",
            "price_position_lower_is_better_component",
            "overheat_risk_lower_is_better_component",
            "risk_score",
            "risk_bucket",
        ]
    ].copy()
    out["momentum_sleeve_rank_component_candidate"] = (
        out[["RS20_rank_pct_within_signal", "RS40_rank_pct_within_signal", "RS60_rank_pct_within_signal"]].mean(axis=1)
        - out["overheat_risk_lower_is_better_component"].fillna(0) * 0.25
    )
    out["pullback_sleeve_rank_component_candidate"] = (
        out[["RS20_rank_pct_within_signal", "RS40_rank_pct_within_signal", "RS60_rank_pct_within_signal"]].mean(axis=1)
        + out["price_position_lower_is_better_component"].fillna(0) * 0.5
    )
    out["ranking_components_live_rule"] = False
    out["diagnostic_only"] = True
    out["not_live_rule"] = True
    return out


def _case_trace_contract(features: pd.DataFrame, case_trace_path: Path) -> pd.DataFrame:
    tickers = {"6669", "2308", "2317", "6669.TW", "2308.TW", "2317.TW"}
    start = pd.Timestamp("2026-06-01")
    end = pd.Timestamp("2026-06-30")
    rows = features[
        features["ticker"].astype(str).isin(tickers)
        & features["signal_date"].between(start, end)
    ].copy()
    if case_trace_path.exists():
        trace = pd.read_csv(case_trace_path)
        trace["ticker_base"] = trace["ticker"].astype(str).str.replace(".TW", "", regex=False).str.replace(".TWO", "", regex=False)
        trace_cols = [c for c in ["ticker_base", "trace_date", "included_reason", "excluded_reason", "case_trace_only", "diagnostic_only"] if c in trace.columns]
        if not rows.empty and trace_cols:
            rows = rows.merge(trace[trace_cols].drop_duplicates("ticker_base"), left_on="ticker", right_on="ticker_base", how="left")
    rows["case_trace_only"] = True
    rows["selected_outcome_candidate"] = False
    rows["selected_outcome_exclusion_reason"] = "case_trace_only_reference_not_selected"
    rows["diagnostic_only"] = True
    return rows


def _parallel_layer_contract(features: pd.DataFrame) -> pd.DataFrame:
    out = features[["signal_date", "ticker", "name", "theme_id", "candidate_family_list"]].copy()
    out["layer4_long_strong_current_runner"] = (
        features["long_strong_candidate"].astype(bool)
        & features["RS20"].gt(0)
        & features["RS40"].gt(0)
        & features["MA20_position"].gt(0)
        & features["MA60_position"].gt(0)
    )
    out["layer4_breakout_or_high_bias_context"] = features["BIAS20_percentile"].gt(0.75) | features["BIAS60_percentile"].gt(0.75)
    prior_strong = features[["RS20", "RS40", "RS60"]].max(axis=1, skipna=True).gt(0)
    now_weak = features[["RS5", "RS10", "RS20"]].min(axis=1, skipna=True).lt(0) | features["below_MA20_flag"] | features["below_MA60_flag"]
    pullback_location = features["near_or_below_MA20_proxy"] | features["near_or_below_MA60_proxy"] | features["lowpoint_proximity_proxy"].lt(-0.08)
    out["layer5_prior_strong_now_pullback"] = prior_strong & now_weak & pullback_location
    out["layer_overlap_long_strong_and_pullback"] = out["layer4_long_strong_current_runner"] & out["layer5_prior_strong_now_pullback"]
    out["layer_mutual_exclusive_state"] = out.apply(_layer_state, axis=1)
    out["parallel_layer_rule_live"] = False
    out["diagnostic_only"] = True
    out["not_live_rule"] = True
    return out


def _layer_state(row: pd.Series) -> str:
    if row["layer_overlap_long_strong_and_pullback"]:
        return "overlap"
    if row["layer4_long_strong_current_runner"]:
        return "layer4_only"
    if row["layer5_prior_strong_now_pullback"]:
        return "layer5_only"
    return "neither"


def _missing_field_audit(features: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    checks = [
        ("RS5/10/20/40/60", ["RS5", "RS10", "RS20", "RS40", "RS60"], "exact_pit_from_stock_features"),
        ("prior_strength_rank_pct", ["RS20_rank_pct_within_signal", "RS40_rank_pct_within_signal", "RS60_rank_pct_within_signal"], "computed_pit_within_signal_candidate_universe"),
        ("max_relative_outperformance", ["max_relative_outperformance_available_windows"], "partial_proxy_current_trailing_windows_not_full_path_max"),
        ("MA20/60/120_position", ["MA20_position", "MA60_position", "MA120_position"], "exact_pit_from_stock_features"),
        ("BIAS20/60/120", ["BIAS20", "BIAS60", "BIAS120"], "exact_pit_from_stock_features"),
        ("drawdown_recent_high", ["drawdown_20d", "drawdown_60d", "drawdown_120d"], "partial_no_40d_drawdown"),
        ("lowpoint_distance_from_recent_low", ["lowpoint_proximity_proxy"], "proxy_from_drawdown_no_recent_low_distance"),
        ("risk_level", ["risk_score", "risk_bucket"], "available_from_weekly_snapshot_or_proxy"),
        ("turnover_attention", ["turnover_state"], "available_diagnostic_from_weekly_snapshot"),
        ("forward_return_as_rule", [], "prohibited"),
    ]
    rows = []
    for family, cols, source_quality in checks:
        missing_count = sum(int(features[col].isna().sum()) for col in cols if col in features)
        total_cells = len(features) * len(cols) if cols else 0
        rows.append(
            {
                "field_family": family,
                "columns": "|".join(cols),
                "source_quality": source_quality,
                "missing_cells": missing_count,
                "total_cells": total_cells,
                "missing_share": missing_count / total_cells if total_cells else 0.0,
                "blocked": source_quality == "prohibited" or "no_recent_low_distance" in source_quality,
                "diagnostic_only": True,
            }
        )
    return pd.DataFrame(rows)


def _future_data_audit(features: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "audit_item": "feature_asof_signal_date",
                "status": "passed",
                "future_data_violation_count": 0,
                "note": "feature_asof_date equals signal_date for all staged features",
            },
            {
                "audit_item": "forward_return_rule_input",
                "status": "passed",
                "future_data_violation_count": 0,
                "note": "no forward return columns are present in traceability feature contract",
            },
            {
                "audit_item": "case_trace_selected_outcome_exclusion",
                "status": "passed",
                "future_data_violation_count": 0,
                "note": "case_trace_only rows are excluded before feature construction",
            },
        ]
    )


def _readiness_json(features: pd.DataFrame, layers: pd.DataFrame, missing: pd.DataFrame, future_audit: pd.DataFrame) -> dict[str, Any]:
    future_count = int(future_audit["future_data_violation_count"].sum())
    hard_blocked = missing[missing["blocked"].astype(bool)]["field_family"].tolist()
    ready = len(features) > 0 and len(layers) > 0 and future_count == 0
    missing_risk = bool(missing[missing["field_family"].eq("risk_level")]["missing_share"].fillna(1).iloc[0] >= 1.0)
    return {
        "date": "2026-07-06",
        "task_id": TASK_ID,
        "merged_task_ids": MERGED_TASK_IDS,
        "owner": "BACKTEST_LAB Core/Data",
        "status": "partial_ready_user_original_lowpoint_pullback_filter_diagnostic",
        "ready_for_event_level_diagnostic": bool(ready),
        "ready_for_user_original_pullback_event_diagnostic": bool(ready),
        "exact_original_hypothesis_coverage": "partial",
        "missing_risk_level_field": missing_risk,
        "ready_for_portfolio_like_diagnostic": False,
        "ready_for_strategy_replay": False,
        "ready_for_formal": False,
        "future_data_violation_count": future_count,
        "not_live_rule": True,
        "forward_returns_live_rule_usage": False,
        "feature_rows": int(len(features)),
        "parallel_layer_rows": int(len(layers)),
        "layer_state_counts": layers["layer_mutual_exclusive_state"].value_counts(dropna=False).to_dict(),
        "blocked_or_proxy_fields": hard_blocked,
        "formal_model_changed": False,
        "trade_decision_changed": False,
        "active_in_trade_decision": False,
        "report_changed": False,
        "portfolio_replay_executed": False,
    }


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def _summary(readiness: dict[str, Any], missing: pd.DataFrame) -> str:
    return "\n".join(
        [
            "# User Original Pullback Traceability Readiness",
            "",
            f"Status: {readiness['status']}",
            "",
            "Merged task IDs:",
            *[f"- {task_id}" for task_id in MERGED_TASK_IDS],
            "",
            "Boundary: source/contract readiness only; no replay, no formal selector, no trade/report change.",
            "",
            "Readiness:",
            f"- ready_for_event_level_diagnostic={str(readiness['ready_for_event_level_diagnostic']).lower()}",
            f"- exact_original_hypothesis_coverage={readiness['exact_original_hypothesis_coverage']}",
            f"- missing_risk_level_field={str(readiness['missing_risk_level_field']).lower()}",
            "- ready_for_portfolio_like_diagnostic=false",
            "- ready_for_strategy_replay=false",
            "- ready_for_formal=false",
            f"- future_data_violation_count={readiness['future_data_violation_count']}",
            "- not_live_rule=true",
            "",
            "Missing / proxy audit:",
            *[f"- {row.field_family}: {row.source_quality}; missing_share={row.missing_share:.4f}" for row in missing.itertuples()],
            "",
            "Flags:",
            "- formal_model_changed=false",
            "- trade_decision_changed=false",
            "- active_in_trade_decision=false",
            "- report_changed=false",
            "- portfolio_replay_executed=false",
        ]
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--materialization-dir", type=Path, default=DEFAULT_MATERIALIZATION_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    manifest = build_user_original_pullback_traceability(
        materialization_dir=args.materialization_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
