"""Build Layer5 lifecycle-state single-stock selector candidate contract.

This is diagnostic/readiness only. It materializes PIT lifecycle/state
candidate designs inside the Layer4 80-stock pool. It does not authorize a
live Layer5 rule, fallback rule, portfolio replay, formal model, daily report,
or trade decision.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


TASK_ID = "TASK-BACKTEST-CORE-VNEXT-LAYER5-LIFECYCLE-STATE-SELECTOR-CANDIDATE-CONTRACT-001"
DEFAULT_WITHIN80_DIR = Path("outputs/vnext_layer5_within80_daily_rank_context_contract_20260708")
DEFAULT_INCUMBENT_DIR = Path("outputs/vnext_layer5_incumbent_aware_selector_candidate_contract_20260708")
DEFAULT_HURDLE_DIR = Path("outputs/vnext_layer5_stock_vs_00631l_hurdle_context_contract_20260708")
DEFAULT_EXPERIMENTS_DIR = Path(
    "C:/Users/zergv/Documents/Codex/2026-07-06/backtest-lab-experiments-diagnostic-validation-attribution/"
    "outputs/vnext_layer5_stock_vs_00631l_hurdle_fallback_diagnostic_20260708"
)
DEFAULT_OUTPUT_DIR = Path("outputs/vnext_layer5_lifecycle_state_selector_candidate_contract_20260708")
PERIODS = {
    "P1": ("2015-01-02", "2022-12-29"),
    "P2": ("2023-01-02", "2026-06-30"),
    "2024_latest": ("2024-01-02", "2026-06-30"),
    "2026YTD": ("2026-01-02", "2026-06-30"),
}
EVAL_HORIZONS = [5, 10, 20, 30, 40]
LIFECYCLE_VARIANTS = [
    "lifecycle_top10_clean_state_selector",
    "pullback_repair_reacceleration_selector",
    "reentry_confirmed_lifecycle_selector",
    "strengthening_not_overheated_selector",
    "clean_trend_low_risk_selector",
    "high_confidence_bonus_lifecycle_selector",
    "raw_top1_baseline",
    "incumbent_reentry_baseline",
    "stock_vs_00631L_best_baseline",
]


def build_contract(
    *,
    within80_dir: str | Path = DEFAULT_WITHIN80_DIR,
    incumbent_dir: str | Path = DEFAULT_INCUMBENT_DIR,
    hurdle_dir: str | Path = DEFAULT_HURDLE_DIR,
    experiments_dir: str | Path = DEFAULT_EXPERIMENTS_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    within80_path = Path(within80_dir)
    incumbent_path = Path(incumbent_dir)
    hurdle_path = Path(hurdle_dir)
    experiments_path = Path(experiments_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    source_readiness = _read_json(within80_path / "readiness_for_layer5_within80_daily_rank_context_diagnostic.json")
    incumbent_readiness = _read_json(incumbent_path / "readiness_for_layer5_incumbent_aware_single_stock_diagnostic.json")
    hurdle_readiness = _read_json(hurdle_path / "readiness_for_layer5_stock_vs_00631l_hurdle_diagnostic.json")
    experiment_summary = _read_json(experiments_path / "layer5_stock_vs_00631l_summary.json")
    within80 = _read_context(within80_path / "layer5_within80_daily_rank_context_contract.csv")
    incumbent = _read_optional_context(incumbent_path / "layer5_incumbent_aware_selector_candidate_contract.csv")
    hurdle = _read_optional_context(hurdle_path / "layer5_stock_vs_00631l_hurdle_context_contract.csv")

    features = _attach_lifecycle_features(within80)
    contract = _build_lifecycle_candidates(features, incumbent, hurdle)
    feature_design = _lifecycle_state_feature_design()
    variant_design = _selector_variant_design()
    source_quality = _source_quality_matrix()
    coverage = _coverage_by_period(contract)
    missingness = _missingness_by_period(contract)
    blocked_proxy = _blocked_proxy_ledger()
    future_audit = _future_data_audit(contract)
    requested_actual = _requested_vs_actual_coverage(contract)
    readiness = _readiness(
        source_readiness,
        incumbent_readiness,
        hurdle_readiness,
        experiment_summary,
        contract,
        coverage,
        future_audit,
    )

    _write_csv(contract, output / "layer5_lifecycle_state_selector_candidate_contract.csv")
    _write_csv(feature_design, output / "layer5_lifecycle_state_feature_design.csv")
    _write_csv(variant_design, output / "layer5_lifecycle_selector_variant_design.csv")
    _write_csv(source_quality, output / "layer5_lifecycle_source_quality_matrix.csv")
    _write_csv(coverage, output / "layer5_lifecycle_coverage_by_period.csv")
    _write_csv(missingness, output / "layer5_lifecycle_missingness_by_period.csv")
    _write_csv(blocked_proxy, output / "layer5_lifecycle_blocked_proxy_ledger.csv")
    _write_csv(future_audit, output / "layer5_lifecycle_future_data_audit.csv")
    _write_csv(requested_actual, output / "layer5_lifecycle_requested_vs_actual_coverage.csv")
    (output / "readiness_for_layer5_lifecycle_state_selector_diagnostic.json").write_text(
        json.dumps(readiness, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    manifest = {
        "task_id": TASK_ID,
        "status": readiness["status"],
        "output_dir": str(output.resolve()),
        "input_within80_dir": str(within80_path.resolve()),
        "input_incumbent_dir": str(incumbent_path.resolve()),
        "input_hurdle_dir": str(hurdle_path.resolve()),
        "input_experiments_dir": str(experiments_path.resolve()),
        "output_files": [
            "layer5_lifecycle_state_selector_candidate_contract.csv",
            "layer5_lifecycle_state_feature_design.csv",
            "layer5_lifecycle_selector_variant_design.csv",
            "layer5_lifecycle_source_quality_matrix.csv",
            "layer5_lifecycle_coverage_by_period.csv",
            "layer5_lifecycle_missingness_by_period.csv",
            "layer5_lifecycle_blocked_proxy_ledger.csv",
            "layer5_lifecycle_future_data_audit.csv",
            "layer5_lifecycle_requested_vs_actual_coverage.csv",
            "readiness_for_layer5_lifecycle_state_selector_diagnostic.json",
            "manifest.json",
            "final_summary_zh.md",
        ],
        **_fixed_flags(),
        "diagnostic_only": True,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "final_summary_zh.md").write_text(_summary(readiness), encoding="utf-8")
    return manifest


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _read_context(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"ticker": str}, encoding="utf-8-sig", low_memory=False)
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
    return df


def _read_optional_context(path: Path) -> pd.DataFrame:
    return _read_context(path) if path.exists() else pd.DataFrame()


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def _fixed_flags() -> dict[str, bool]:
    return {
        "formal_model_changed": False,
        "trade_decision_changed": False,
        "active_in_trade_decision": False,
        "report_changed": False,
        "portfolio_replay_executed": False,
        "ready_for_strategy_replay": False,
        "not_live_rule": True,
        "forward_returns_live_rule_usage": False,
    }


def _bool(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df:
        return pd.Series(False, index=df.index)
    series = df[col]
    if series.dtype == bool:
        return series.fillna(False)
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).ne(0)
    return series.astype(str).str.lower().isin(["true", "1", "yes", "y"])


def _num(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df:
        return pd.Series(default, index=df.index)
    return pd.to_numeric(df[col], errors="coerce").fillna(default)


def _attach_lifecycle_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out[_bool(out, "is_layer4_primary_pool") & _num(out, "within80_rank").between(1, 80)].copy()
    out["within80_top10_candidate_scope"] = _num(out, "within80_rank").le(10)
    late_stage = (
        _bool(out, "rs60_high_short_rs_weakening_exhaustion_context")
        | _bool(out, "rs_exhaustion_warning_context")
        | _bool(out, "bias_overheat_penalty_context")
        | _bool(out, "volatility_high_context")
        | _bool(out, "blowoff_turnover_without_price_continuation_proxy")
        | _bool(out, "large_down_day_flag_20d_proxy")
        | _bool(out, "exhaustion_risk_medium_or_high_confidence")
        | _bool(out, "breakdown_risk_medium_or_high_confidence")
    )
    out["avoid_late_stage_flag"] = late_stage
    out["not_overheated_or_breakdown"] = ~late_stage
    out["strengthening_not_overheated_flag"] = (
        _bool(out, "rs20_30_primary_momentum_stable")
        & ~_bool(out, "rs_short_deterioration_flag")
        & out["not_overheated_or_breakdown"]
    )
    out["pullback_repair_reacceleration_flag"] = (
        (_bool(out, "lifecycle_pullback_reacceleration_context") | _bool(out, "pullback_repair_medium_or_high_confidence"))
        & (_bool(out, "rs_short_acceleration_flag") | _bool(out, "shape_slow_start_vs_0050"))
        & ~_bool(out, "breakdown_risk_medium_or_high_confidence")
    )
    out["reentry_after_watchlist_confirmation_flag"] = (
        _bool(out, "extended_100_to_80_reentry_context")
        & (_bool(out, "came_from_100_extended_only_recent_4w") | _bool(out, "capital_reasonable_band_4w_persistent"))
    )
    out["clean_trend_low_risk_flag"] = (
        _bool(out, "rs20_30_primary_momentum_stable")
        & _bool(out, "capital_reasonable_band_4w_persistent")
        & out["not_overheated_or_breakdown"]
        & ~_bool(out, "risk_overheat_penalty_context")
    )
    out["high_confidence_bonus_flag"] = _bool(out, "in_31_high_confidence_subpool_reference")
    out["strengthening_not_overheated_score"] = (
        0.30 * _bool(out, "rs20_30_primary_momentum_stable").astype(float)
        + 0.20 * _bool(out, "rs_short_acceleration_flag").astype(float)
        + 0.20 * _bool(out, "capital_reasonable_band_4w_persistent").astype(float)
        + 0.15 * out["not_overheated_or_breakdown"].astype(float)
        + 0.15 * _num(out, "within80_rank_score")
    )
    out["pullback_repair_reacceleration_score"] = (
        0.30 * _num(out, "pullback_repair_score")
        + 0.20 * _bool(out, "pullback_repair_reacceleration_flag").astype(float)
        + 0.20 * _bool(out, "rs_short_acceleration_flag").astype(float)
        + 0.15 * out["not_overheated_or_breakdown"].astype(float)
        + 0.15 * _num(out, "within80_rank_score")
    )
    out["reentry_after_watchlist_confirmation_score"] = (
        0.35 * _bool(out, "reentry_after_watchlist_confirmation_flag").astype(float)
        + 0.20 * _bool(out, "capital_reasonable_band_4w_persistent").astype(float)
        + 0.20 * _bool(out, "rs20_30_primary_momentum_stable").astype(float)
        + 0.15 * out["not_overheated_or_breakdown"].astype(float)
        + 0.10 * _num(out, "within80_rank_score")
    )
    out["clean_trend_low_risk_score"] = (
        0.30 * _num(out, "layer4_risk_aware_score")
        + 0.25 * _bool(out, "clean_trend_low_risk_flag").astype(float)
        + 0.20 * _bool(out, "rs20_30_primary_momentum_stable").astype(float)
        + 0.15 * _bool(out, "capital_reasonable_band_4w_persistent").astype(float)
        + 0.10 * _num(out, "within80_rank_score")
    )
    out["high_confidence_bonus_lifecycle_score"] = (
        0.35 * _bool(out, "high_confidence_bonus_flag").astype(float)
        + 0.20 * _num(out, "layer4_risk_aware_score")
        + 0.20 * _num(out, "final_selector_lifecycle_context_score")
        + 0.15 * out["not_overheated_or_breakdown"].astype(float)
        + 0.10 * _num(out, "within80_rank_score")
    )
    out["avoid_late_stage_score"] = (
        0.25 * _bool(out, "rs60_high_short_rs_weakening_exhaustion_context").astype(float)
        + 0.20 * _bool(out, "bias_overheat_penalty_context").astype(float)
        + 0.20 * _bool(out, "volatility_high_context").astype(float)
        + 0.20 * _bool(out, "blowoff_turnover_without_price_continuation_proxy").astype(float)
        + 0.15 * _bool(out, "large_down_day_flag_20d_proxy").astype(float)
    )
    out["lifecycle_composite_score"] = (
        0.24 * out["strengthening_not_overheated_score"]
        + 0.20 * out["pullback_repair_reacceleration_score"]
        + 0.18 * out["reentry_after_watchlist_confirmation_score"]
        + 0.20 * out["clean_trend_low_risk_score"]
        + 0.10 * out["high_confidence_bonus_lifecycle_score"]
        - 0.12 * out["avoid_late_stage_score"]
    )
    out["lifecycle_state_contract_only"] = True
    out["lifecycle_selector_output"] = False
    return out


def _build_lifecycle_candidates(features: pd.DataFrame, incumbent: pd.DataFrame, hurdle: pd.DataFrame) -> pd.DataFrame:
    rows = []
    inc_lookup = _variant_lookup(incumbent, "selector_candidate_variant", "reentry_confirmed_selector")
    hurdle_lookup = _variant_lookup(hurdle, "decision_candidate_variant", "stock_if_incumbent_still_valid_else_00631L_or_best_confirmed_challenger")
    for date, week in features.groupby("snapshot_date", sort=True):
        top10 = week[_bool(week, "within80_top10_candidate_scope")].copy()
        if top10.empty:
            top10 = week.nsmallest(10, "within80_rank").copy()
        selections = {
            "lifecycle_top10_clean_state_selector": _select(top10, "lifecycle_composite_score", ~_bool(top10, "avoid_late_stage_flag")),
            "pullback_repair_reacceleration_selector": _select(top10, "pullback_repair_reacceleration_score", _bool(top10, "pullback_repair_reacceleration_flag")),
            "reentry_confirmed_lifecycle_selector": _select(top10, "reentry_after_watchlist_confirmation_score", _bool(top10, "reentry_after_watchlist_confirmation_flag")),
            "strengthening_not_overheated_selector": _select(top10, "strengthening_not_overheated_score", _bool(top10, "strengthening_not_overheated_flag")),
            "clean_trend_low_risk_selector": _select(top10, "clean_trend_low_risk_score", _bool(top10, "clean_trend_low_risk_flag")),
            "high_confidence_bonus_lifecycle_selector": _select(top10, "high_confidence_bonus_lifecycle_score", _bool(top10, "high_confidence_bonus_flag")),
            "raw_top1_baseline": top10.nsmallest(1, "within80_rank").iloc[0],
        }
        if date in inc_lookup:
            selections["incumbent_reentry_baseline"] = _coerce_to_feature_row(inc_lookup[date], features, date)
        else:
            selections["incumbent_reentry_baseline"] = selections["reentry_confirmed_lifecycle_selector"]
        if date in hurdle_lookup and bool(hurdle_lookup[date].get("uses_stock_candidate", False)):
            selections["stock_vs_00631L_best_baseline"] = _coerce_to_feature_row(hurdle_lookup[date], features, date)
        else:
            selections["stock_vs_00631L_best_baseline"] = selections["incumbent_reentry_baseline"]

        for variant, selected in selections.items():
            row = selected.copy()
            row["lifecycle_selector_candidate_variant"] = variant
            row["selector_candidate_family"] = _variant_family(variant)
            row["selected_candidate_id"] = row.get("ticker")
            row["recommended_asset_type"] = "stock"
            row["00631L_reference_only"] = True
            row["00631L_ordinary_stock_pool_member"] = False
            row["fallback_00631L_metadata_only"] = True
            row["fallback_trading_rule_output"] = False
            row["state_match_for_variant"] = _state_match(row, variant)
            row["variant_score_used"] = _variant_score(row, variant)
            row["variant_selection_basis"] = _variant_basis(variant)
            row["top10_lifecycle_scope"] = True
            row["single_stock_candidate_design_only"] = True
            row["live_layer5_rule_output"] = False
            row["trade_decision_output"] = False
            row["daily_report_output"] = False
            row["portfolio_like_execution_output"] = False
            row["ab_switch_rule_output"] = False
            row["second_stock_allocation_output"] = False
            row["future_return_as_rule"] = False
            row["forward_return_as_rule"] = False
            row["max_in_band_as_rule"] = False
            row["evaluation_metadata_only"] = True
            row["next_day_entry_assumption"] = "diagnostic_only_next_trading_session_after_rank_context_date"
            row["turnover_cost_placeholder"] = "blocked_no_accepted_cost_model"
            for key, value in _fixed_flags().items():
                row[key] = value
            rows.append(row)
    out = pd.DataFrame(rows).sort_values(["lifecycle_selector_candidate_variant", "snapshot_date"]).reset_index(drop=True)
    out = _attach_path_state(out)
    return out


def _variant_lookup(frame: pd.DataFrame, col: str, value: str) -> dict[pd.Timestamp, pd.Series]:
    if frame.empty or col not in frame:
        return {}
    subset = frame[frame[col].eq(value)].copy()
    return {pd.Timestamp(row["snapshot_date"]): row for _, row in subset.iterrows()}


def _coerce_to_feature_row(row: pd.Series, features: pd.DataFrame, date: pd.Timestamp) -> pd.Series:
    ticker = str(row.get("ticker", row.get("recommended_ticker", "")))
    match = features[(features["snapshot_date"].eq(date)) & (features["ticker"].astype(str).eq(ticker))]
    if not match.empty:
        return match.iloc[0].copy()
    top10 = features[features["snapshot_date"].eq(date)].nsmallest(10, "within80_rank")
    return top10.iloc[0].copy()


def _select(frame: pd.DataFrame, score_col: str, mask: pd.Series) -> pd.Series:
    candidates = frame[mask].copy()
    if candidates.empty:
        candidates = frame.copy()
    return candidates.sort_values([score_col, "within80_rank", "ticker"], ascending=[False, True, True]).iloc[0]


def _state_match(row: pd.Series, variant: str) -> bool:
    if variant == "pullback_repair_reacceleration_selector":
        return bool(row.get("pullback_repair_reacceleration_flag", False))
    if variant == "reentry_confirmed_lifecycle_selector":
        return bool(row.get("reentry_after_watchlist_confirmation_flag", False))
    if variant == "strengthening_not_overheated_selector":
        return bool(row.get("strengthening_not_overheated_flag", False))
    if variant == "clean_trend_low_risk_selector":
        return bool(row.get("clean_trend_low_risk_flag", False))
    if variant == "high_confidence_bonus_lifecycle_selector":
        return bool(row.get("high_confidence_bonus_flag", False))
    if variant == "lifecycle_top10_clean_state_selector":
        return not bool(row.get("avoid_late_stage_flag", False))
    return True


def _variant_score(row: pd.Series, variant: str) -> float:
    mapping = {
        "lifecycle_top10_clean_state_selector": "lifecycle_composite_score",
        "pullback_repair_reacceleration_selector": "pullback_repair_reacceleration_score",
        "reentry_confirmed_lifecycle_selector": "reentry_after_watchlist_confirmation_score",
        "strengthening_not_overheated_selector": "strengthening_not_overheated_score",
        "clean_trend_low_risk_selector": "clean_trend_low_risk_score",
        "high_confidence_bonus_lifecycle_selector": "high_confidence_bonus_lifecycle_score",
        "raw_top1_baseline": "within80_rank_score",
        "incumbent_reentry_baseline": "reentry_after_watchlist_confirmation_score",
        "stock_vs_00631L_best_baseline": "lifecycle_composite_score",
    }
    return float(row.get(mapping.get(variant, "lifecycle_composite_score"), 0.0))


def _variant_basis(variant: str) -> str:
    mapping = {
        "lifecycle_top10_clean_state_selector": "top10 lifecycle composite, clean-state preferred",
        "pullback_repair_reacceleration_selector": "top10 prior-strong pullback repair and reacceleration context",
        "reentry_confirmed_lifecycle_selector": "top10 100-to-80 reentry with multi-snapshot confirmation context",
        "strengthening_not_overheated_selector": "top10 RS20/30 strengthening without overheat context",
        "clean_trend_low_risk_selector": "top10 stable trend and lower risk context",
        "high_confidence_bonus_lifecycle_selector": "top10 lifecycle score with 31-subpool confidence bonus",
        "raw_top1_baseline": "within80 rank top1 baseline only",
        "incumbent_reentry_baseline": "prior incumbent-aware reentry baseline only",
        "stock_vs_00631L_best_baseline": "prior stock-vs-00631L best stock context baseline only",
    }
    return mapping[variant]


def _variant_family(variant: str) -> str:
    if "baseline" in variant:
        return "baseline_only"
    if "pullback" in variant:
        return "pullback_repair_lifecycle"
    if "reentry" in variant:
        return "reentry_lifecycle"
    if "strengthening" in variant:
        return "strengthening_lifecycle"
    if "clean_trend" in variant:
        return "clean_trend_low_risk"
    if "high_confidence" in variant:
        return "confidence_bonus_lifecycle"
    return "composite_lifecycle"


def _attach_path_state(contract: pd.DataFrame) -> pd.DataFrame:
    out = contract.copy()
    out["selector_changed_from_previous_signal"] = False
    out["consecutive_same_recommendation_snapshots_proxy"] = 1
    for variant, idx in out.groupby("lifecycle_selector_candidate_variant", sort=False).groups.items():
        previous = None
        count = 0
        for row_index in idx:
            ticker = out.at[row_index, "ticker"]
            if ticker == previous:
                count += 1
                changed = False
            else:
                count = 1
                changed = previous is not None
            out.at[row_index, "selector_changed_from_previous_signal"] = changed
            out.at[row_index, "consecutive_same_recommendation_snapshots_proxy"] = count
            previous = ticker
    out["consecutive_same_recommendation_days_proxy"] = out["consecutive_same_recommendation_snapshots_proxy"] * 5
    return out


def _lifecycle_state_feature_design() -> pd.DataFrame:
    rows = [
        ("strengthening_not_overheated", "RS20/30 support, RS5/10 not deteriorating, BIAS/volatility not hot", "PIT/proxy context"),
        ("pullback_repair_reacceleration", "Prior strength plus correction and current reacceleration, not breakdown", "PIT/proxy context"),
        ("reentry_after_watchlist_confirmation", "100 extended watchlist to 80 primary with multi-snapshot or capital confirmation", "PIT context"),
        ("clean_trend_low_risk", "Stable RS/capital/trend context without high risk/exhaustion", "PIT/proxy context"),
        ("high_confidence_bonus", "31 high-confidence subpool overlap as bonus only", "reference context"),
        ("avoid_late_stage", "RS60 high + short RS weakening, high BIAS, blowoff/large-down proxies", "diagnostic proxy"),
        ("forward_evaluation_metadata", "5D/10D/20D/30D primary, 40D decay reference", "evaluation-only"),
    ]
    return pd.DataFrame(rows, columns=["lifecycle_state_feature", "definition", "source_quality"])


def _selector_variant_design() -> pd.DataFrame:
    rows = [
        ("lifecycle_top10_clean_state_selector", "top10 lifecycle composite, clean-state preferred", "candidate design only"),
        ("pullback_repair_reacceleration_selector", "top10 prior-strong repair/reacceleration state", "candidate design only"),
        ("reentry_confirmed_lifecycle_selector", "top10 reentry after watchlist confirmation", "candidate design only"),
        ("strengthening_not_overheated_selector", "top10 strengthening but not overheated", "candidate design only"),
        ("clean_trend_low_risk_selector", "top10 stable trend, lower risk", "candidate design only"),
        ("high_confidence_bonus_lifecycle_selector", "top10 lifecycle plus 31 subpool bonus, not mandatory filter", "candidate design only"),
        ("raw_top1_baseline", "within80 raw top1 baseline", "baseline only"),
        ("incumbent_reentry_baseline", "prior incumbent-aware/reentry baseline", "baseline only"),
        ("stock_vs_00631L_best_baseline", "prior stock-vs-00631L best stock/hurdle context baseline", "baseline only"),
    ]
    return pd.DataFrame(rows, columns=["lifecycle_selector_candidate_variant", "definition", "status"])


def _source_quality_matrix() -> pd.DataFrame:
    rows = [
        ("Layer4_80_primary_pool", "PIT materialized Core contract", "base universe"),
        ("within80_rank_top10", "PIT context", "candidate scope, not top1 rule"),
        ("RS5_10_20_30_60", "RS30 proxy where exact unavailable", "lifecycle context"),
        ("BIAS_volatility_large_down_blowoff", "diagnostic proxy thresholds", "risk/late-stage context"),
        ("31_high_confidence_subpool", "reference only", "bonus, not mandatory filter"),
        ("100_extended_watchlist", "reference only", "reentry context"),
        ("00631L", "benchmark/fallback metadata only", "not ordinary stock row"),
        ("real_current_holder_state", "blocked", "no formal live holder state"),
        ("cash_bear_classifier", "blocked", "no cash/fallback trading rule"),
        ("forward_returns", "evaluation_metadata_only", "not rule construction"),
    ]
    return pd.DataFrame(rows, columns=["field_group", "source_quality", "contract_role"])


def _coverage_by_period(contract: pd.DataFrame) -> pd.DataFrame:
    frame = contract.copy()
    frame["period"] = frame["snapshot_date"].map(_period_label)
    rows = []
    for (period, variant), group in frame.groupby(["period", "lifecycle_selector_candidate_variant"], dropna=False):
        rows.append(
            {
                "period": period,
                "lifecycle_selector_candidate_variant": variant,
                "row_count": len(group),
                "weekly_snapshot_count": int(group["snapshot_date"].nunique()),
                "state_match_share": _share(_bool(group, "state_match_for_variant").sum(), len(group)),
                "top10_scope_share": _share(_bool(group, "top10_lifecycle_scope").sum(), len(group)),
                "avoid_late_stage_share": _share(_bool(group, "avoid_late_stage_flag").sum(), len(group)),
                "forward_eval_available_20d_share": _share(_bool(group, "forward_eval_available_20d").sum(), len(group)),
                "switch_rate_proxy": _share(_bool(group, "selector_changed_from_previous_signal").sum(), len(group)),
                "avg_consecutive_same_recommendation_snapshots_proxy": float(_num(group, "consecutive_same_recommendation_snapshots_proxy").mean()),
            }
        )
    return pd.DataFrame(rows)


def _missingness_by_period(contract: pd.DataFrame) -> pd.DataFrame:
    fields = [
        "RS5",
        "RS10",
        "RS20",
        "RS30_proxy",
        "RS60",
        "BIAS20",
        "BIAS60",
        "volatility",
        "risk_bucket",
        "large_down_day_count_20d_proxy",
        "blowoff_turnover_without_price_continuation_proxy",
        "lifecycle_composite_score",
    ] + [f"forward_excess_vs_0050_{horizon}d" for horizon in EVAL_HORIZONS] + [
        f"forward_excess_vs_00631L_{horizon}d" for horizon in EVAL_HORIZONS
    ]
    frame = contract.copy()
    frame["period"] = frame["snapshot_date"].map(_period_label)
    rows = []
    for period, group in frame.groupby("period", dropna=False):
        for field in fields:
            missing = int(group[field].isna().sum()) if field in group else len(group)
            rows.append({"period": period, "field": field, "row_count": len(group), "missing_count": missing, "missing_share": _share(missing, len(group))})
    return pd.DataFrame(rows)


def _blocked_proxy_ledger() -> pd.DataFrame:
    rows = [
        ("risk_bucket", "proxy_or_blocked", "No accepted formal risk bucket; use source_quality field when present"),
        ("large_down_day_count_20d_proxy", "diagnostic_proxy", "PIT proxy threshold, not formal"),
        ("blowoff_turnover_without_price_continuation_proxy", "diagnostic_proxy", "PIT proxy threshold, not formal"),
        ("BIAS_volatility_thresholds", "diagnostic_proxy", "Used only as lifecycle context"),
        ("RS30", "proxy", "RS30_proxy used where exact unavailable"),
        ("real_current_holder_state", "blocked", "No live/formal holder state contract"),
        ("cash_bear_classifier", "blocked", "No cash rule in this package"),
        ("00631L_fallback_rule", "blocked", "00631L is benchmark/reference/fallback metadata only"),
        ("A_B_switch_or_second_stock_allocation", "blocked", "Not authorized"),
        ("portfolio_replay", "blocked", "Not authorized"),
        ("latest_forward_path", "blocked_partial", "Latest rows may lack future evaluation path by horizon"),
    ]
    return pd.DataFrame(rows, columns=["field_or_policy", "status", "reason"])


def _future_data_audit(contract: pd.DataFrame) -> pd.DataFrame:
    rows = [
        ("future_return_as_rule", "passed", int(_bool(contract, "future_return_as_rule").sum()), "false for all rows"),
        ("forward_return_as_rule", "passed", int(_bool(contract, "forward_return_as_rule").sum()), "false for all rows"),
        ("max_in_band_as_rule", "passed", int(_bool(contract, "max_in_band_as_rule").sum()), "false for all rows"),
        ("live_layer5_rule_output", "passed", int(_bool(contract, "live_layer5_rule_output").sum()), "no live rule output"),
        ("trade_decision_output", "passed", int(_bool(contract, "trade_decision_output").sum()), "no trade decision output"),
        ("portfolio_like_execution_output", "passed", int(_bool(contract, "portfolio_like_execution_output").sum()), "no portfolio-like execution output"),
        ("00631L_ordinary_stock_pool_member", "passed", int(_bool(contract, "00631L_ordinary_stock_pool_member").sum()), "00631L is reference/fallback metadata only"),
    ]
    return pd.DataFrame(rows, columns=["audit_item", "status", "violation_count", "evidence"])


def _requested_vs_actual_coverage(contract: pd.DataFrame) -> pd.DataFrame:
    actual_start = contract["snapshot_date"].min()
    actual_end = contract["snapshot_date"].max()
    rows = []
    for period, (requested_start, requested_end) in PERIODS.items():
        start = pd.Timestamp(requested_start)
        end = pd.Timestamp(requested_end)
        in_period = contract[contract["snapshot_date"].between(start, end)]
        rows.append(
            {
                "period": period,
                "requested_start": requested_start,
                "requested_end": requested_end,
                "actual_contract_start": actual_start.date().isoformat(),
                "actual_contract_end": actual_end.date().isoformat(),
                "rows_in_requested_period": len(in_period),
                "weekly_snapshots_in_requested_period": int(in_period["snapshot_date"].nunique()),
                "rows_before_requested_start": int((contract["snapshot_date"] < start).sum()),
                "rows_after_requested_end": int((contract["snapshot_date"] > end).sum()),
            }
        )
    return pd.DataFrame(rows)


def _period_label(value: Any) -> str:
    date = pd.to_datetime(value)
    hits = []
    for label, (start, end) in PERIODS.items():
        if pd.Timestamp(start) <= date <= pd.Timestamp(end):
            hits.append(label)
    return "|".join(hits) if hits else "outside_requested_periods"


def _share(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def _readiness(
    source_readiness: dict[str, Any],
    incumbent_readiness: dict[str, Any],
    hurdle_readiness: dict[str, Any],
    experiment_summary: dict[str, Any],
    contract: pd.DataFrame,
    coverage: pd.DataFrame,
    future_audit: pd.DataFrame,
) -> dict[str, Any]:
    future_violations = int(future_audit["violation_count"].sum())
    ready = future_violations == 0 and len(contract) > 0
    return {
        "task_id": TASK_ID,
        "status": "layer5_lifecycle_state_selector_candidate_contract_ready_for_experiments_intake"
        if ready
        else "layer5_lifecycle_state_selector_candidate_contract_blocked",
        "diagnostic_only": True,
        "input_within80_status": source_readiness.get("status"),
        "input_incumbent_status": incumbent_readiness.get("status"),
        "input_hurdle_status": hurdle_readiness.get("status"),
        "input_experiments_verdict": experiment_summary.get("verdict"),
        "row_count": int(len(contract)),
        "weekly_snapshot_count": int(contract["snapshot_date"].nunique()),
        "lifecycle_selector_candidate_variant_count": int(contract["lifecycle_selector_candidate_variant"].nunique()),
        "lifecycle_selector_candidate_variants": sorted(contract["lifecycle_selector_candidate_variant"].unique().tolist()),
        "avg_state_match_share_by_variant": coverage.groupby("lifecycle_selector_candidate_variant")["state_match_share"].mean().to_dict(),
        "avg_switch_rate_proxy_by_variant": coverage.groupby("lifecycle_selector_candidate_variant")["switch_rate_proxy"].mean().to_dict(),
        "ready_for_layer5_lifecycle_state_single_stock_selector_diagnostic": ready,
        "ready_for_experiments_intake": ready,
        "ready_for_live_layer5_rule": False,
        "ready_for_stock_vs_00631l_fallback_rule": False,
        "ready_for_portfolio_like_diagnostic": False,
        "ready_for_formal": False,
        "ready_for_strategy_replay": False,
        "future_data_violation_count": future_violations,
        "blocked_fields": ["real_current_holder_state", "cash_bear_classifier", "00631L_fallback_rule", "turnover_cost_model", "portfolio_replay"],
        "proxy_fields": ["risk_bucket", "large_down_day_count_20d_proxy", "blowoff_turnover_without_price_continuation_proxy", "RS30_proxy"],
        **_fixed_flags(),
    }


def _summary(readiness: dict[str, Any]) -> str:
    return f"""# Layer5 lifecycle-state selector candidate contract

## Verdict
- status={readiness['status']}
- diagnostic_only=true
- formal_model_changed=false
- trade_decision_changed=false
- active_in_trade_decision=false
- report_changed=false
- portfolio_replay_executed=false
- ready_for_strategy_replay=false
- not_live_rule=true
- forward_returns_live_rule_usage=false

## Scope
- Base universe: Layer4 80 primary pool with within-80 top10 lifecycle candidate scope.
- 100 extended watchlist and 31 high-confidence subpool are context/bonus only.
- 00631L is benchmark/reference/fallback metadata only and is not an ordinary stock row.
- No A/B switch, second-stock allocation, cash rule, live Layer5 rule, portfolio replay, formal model, daily report, or trade decision.
- 這一輪測的是每天只選一檔時，是否能用 lifecycle/state 條件改善個股 selector，而不是靠單日排名或 00631L fallback 避險。

## Candidate variants
- lifecycle_selector_candidate_variant_count={readiness['lifecycle_selector_candidate_variant_count']}
- variants={', '.join(readiness['lifecycle_selector_candidate_variants'])}
- row_count={readiness['row_count']}
- weekly_snapshot_count={readiness['weekly_snapshot_count']}

## Blocked / proxy
- blocked_fields={', '.join(readiness['blocked_fields'])}
- proxy_fields={', '.join(readiness['proxy_fields'])}

## Next
If accepted, hand off to Experiments:
`TASK-BACKTEST-EXPERIMENTS-VNEXT-LAYER5-LIFECYCLE-STATE-SINGLE-STOCK-SELECTOR-DIAGNOSTIC-001`.
完成後如果下一棒明確，請直接指派下一個 thread；如果下一棒不明確，請回報 Strategy Center 判斷。不要完成後停住不回報。
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--within80-dir", default=str(DEFAULT_WITHIN80_DIR))
    parser.add_argument("--incumbent-dir", default=str(DEFAULT_INCUMBENT_DIR))
    parser.add_argument("--hurdle-dir", default=str(DEFAULT_HURDLE_DIR))
    parser.add_argument("--experiments-dir", default=str(DEFAULT_EXPERIMENTS_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()
    manifest = build_contract(
        within80_dir=args.within80_dir,
        incumbent_dir=args.incumbent_dir,
        hurdle_dir=args.hurdle_dir,
        experiments_dir=args.experiments_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
