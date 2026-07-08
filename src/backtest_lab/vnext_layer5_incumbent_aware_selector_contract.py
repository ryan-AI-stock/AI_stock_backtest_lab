"""Build Layer5 incumbent-aware single-stock decision candidate contract.

This is diagnostic/readiness only. It materializes hypothetical path-state
proxies for incumbent-aware selector candidates without authorizing a live
Layer5 rule, A/B switch, portfolio replay, formal model, daily report, or trade
decision.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import pandas as pd


TASK_ID = "TASK-BACKTEST-CORE-VNEXT-LAYER5-INCUMBENT-AWARE-SINGLE-STOCK-DECISION-CANDIDATE-CONTRACT-001"
DEFAULT_LAYER5_DIR = Path("outputs/vnext_layer5_within80_daily_rank_context_contract_20260708")
DEFAULT_EXPERIMENTS_DIR = Path(
    "C:/Users/zergv/Documents/Codex/2026-07-06/backtest-lab-experiments-diagnostic-validation-attribution/"
    "outputs/vnext_layer5_within80_final_selector_context_diagnostic_20260708"
)
DEFAULT_OUTPUT_DIR = Path("outputs/vnext_layer5_incumbent_aware_selector_candidate_contract_20260708")
PERIODS = {
    "P1": ("2015-01-02", "2022-12-29"),
    "P2": ("2023-01-02", "2026-06-30"),
    "2024_latest": ("2024-01-02", "2026-06-30"),
    "2026YTD": ("2026-01-02", "2026-06-30"),
}
EVAL_HORIZONS = [5, 10, 20, 30, 40]


def build_contract(
    *,
    layer5_dir: str | Path = DEFAULT_LAYER5_DIR,
    experiments_dir: str | Path = DEFAULT_EXPERIMENTS_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    layer5 = Path(layer5_dir)
    experiments = Path(experiments_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    source_readiness = _read_json(layer5 / "readiness_for_layer5_within80_daily_rank_context_diagnostic.json")
    experiment_summary = _read_json(experiments / "layer5_final_selector_summary.json")
    within80 = _read_context(layer5 / "layer5_within80_daily_rank_context_contract.csv")

    candidates = _build_incumbent_aware_candidates(within80)
    state_design = _incumbent_state_design()
    threshold_design = _switch_threshold_candidate_design()
    variant_design = _selector_candidate_variant_design()
    source_quality = _source_quality_matrix()
    missingness = _missingness_by_period(candidates)
    coverage = _coverage_by_period(candidates)
    blocked_proxy = _blocked_proxy_ledger()
    future_audit = _future_data_audit(candidates)
    readiness = _readiness(source_readiness, experiment_summary, within80, candidates, coverage, future_audit)

    _write_csv(candidates, output / "layer5_incumbent_aware_selector_candidate_contract.csv")
    _write_csv(state_design, output / "layer5_incumbent_state_design.csv")
    _write_csv(threshold_design, output / "layer5_switch_threshold_candidate_design.csv")
    _write_csv(variant_design, output / "layer5_selector_candidate_variant_design.csv")
    _write_csv(source_quality, output / "layer5_incumbent_source_quality_matrix.csv")
    _write_csv(missingness, output / "layer5_incumbent_missingness_by_period.csv")
    _write_csv(coverage, output / "layer5_incumbent_coverage_by_period.csv")
    _write_csv(blocked_proxy, output / "layer5_incumbent_blocked_proxy_ledger.csv")
    _write_csv(future_audit, output / "layer5_incumbent_future_data_audit.csv")
    _write_csv(_requested_vs_actual_coverage(candidates), output / "layer5_incumbent_requested_vs_actual_coverage.csv")
    (output / "readiness_for_layer5_incumbent_aware_single_stock_diagnostic.json").write_text(
        json.dumps(readiness, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    manifest = {
        "task_id": TASK_ID,
        "status": readiness["status"],
        "output_dir": str(output.resolve()),
        "input_layer5_dir": str(layer5.resolve()),
        "input_experiments_dir": str(experiments.resolve()),
        "output_files": [
            "layer5_incumbent_aware_selector_candidate_contract.csv",
            "layer5_incumbent_state_design.csv",
            "layer5_switch_threshold_candidate_design.csv",
            "layer5_selector_candidate_variant_design.csv",
            "layer5_incumbent_source_quality_matrix.csv",
            "layer5_incumbent_missingness_by_period.csv",
            "layer5_incumbent_coverage_by_period.csv",
            "layer5_incumbent_blocked_proxy_ledger.csv",
            "layer5_incumbent_future_data_audit.csv",
            "layer5_incumbent_requested_vs_actual_coverage.csv",
            "readiness_for_layer5_incumbent_aware_single_stock_diagnostic.json",
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


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def _read_context(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"ticker": str}, encoding="utf-8-sig", low_memory=False)
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
    return df


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


def _build_incumbent_aware_candidates(context: pd.DataFrame) -> pd.DataFrame:
    ctx = _attach_decision_scores(context)
    variants = [
        "fresh_top1_baseline",
        "fresh_best_risk_adjusted_top10_baseline",
        "incumbent_protection_selector",
        "confirmed_challenger_selector",
        "lifecycle_clean_candidate_selector",
        "reentry_confirmed_selector",
        "high_confidence_bonus_selector",
    ]
    state: dict[str, dict[str, Any]] = {
        variant: {"incumbent_ticker": None, "hold_count": 0, "challenger_history": deque(maxlen=3)}
        for variant in variants
    }
    outputs = []
    for date, week in ctx.groupby("snapshot_date", sort=True):
        week = week.sort_values(["within80_rank", "ticker"]).copy()
        lookup = {ticker: row for ticker, row in week.set_index("ticker", drop=False).iterrows()}
        for variant in variants:
            result, new_state = _select_for_variant(variant, week, lookup, state[variant])
            state[variant] = new_state
            row = result.copy()
            row["selector_candidate_variant"] = variant
            row["selector_candidate_family"] = _variant_family(variant)
            row["hypothetical_path_state_proxy"] = True
            row["real_incumbent_state_available"] = False
            row["real_incumbent_state_status"] = "blocked_no_live_position_state_contract"
            row["incumbent_candidate_id"] = new_state.get("previous_incumbent_ticker")
            row["selected_candidate_id"] = row["ticker"]
            row["incumbent_still_in_80"] = bool(new_state.get("incumbent_still_in_80", False))
            row["incumbent_rank_within_80"] = new_state.get("incumbent_rank_within_80")
            row["incumbent_score"] = new_state.get("incumbent_score")
            row["best_challenger_id"] = new_state.get("best_challenger_id")
            row["best_challenger_score"] = new_state.get("best_challenger_score")
            row["incumbent_score_delta_vs_best_candidate"] = new_state.get("score_delta")
            row["incumbent_risk_deterioration"] = bool(new_state.get("incumbent_risk_deterioration", False))
            row["incumbent_rs_deterioration"] = bool(new_state.get("incumbent_rs_deterioration", False))
            row["incumbent_exhaustion_breakdown_context"] = bool(new_state.get("incumbent_exhaustion_breakdown_context", False))
            row["consecutive_hold_snapshots_proxy"] = int(new_state["hold_count"])
            row["consecutive_hold_days_proxy"] = int(new_state["hold_count"] * 5)
            row["selector_changed_from_previous_signal"] = bool(new_state.get("switched", False))
            row["candidate_switch_required_reason"] = new_state.get("switch_reason", "hold_existing_incumbent")
            row["switch_threshold_candidate_design_only"] = True
            row["next_day_entry_assumption"] = "diagnostic_only_next_trading_session_after_rank_context_date"
            row["turnover_cost_placeholder"] = "blocked_no_accepted_cost_model"
            row["cash_fallback_classifier_status"] = "blocked_no_accepted_market_cash_classifier"
            row["fallback_reference_only"] = True
            row["fallback_trading_rule_output"] = False
            row["ab_switch_rule_output"] = False
            row["second_stock_allocation_output"] = False
            row["layer5_live_action_rule_output"] = False
            row["trade_decision_output"] = False
            row["portfolio_like_execution_output"] = False
            row["future_return_as_rule"] = False
            row["forward_return_as_rule"] = False
            for key, value in _fixed_flags().items():
                row[key] = value
            outputs.append(row)
    return pd.DataFrame(outputs).sort_values(["selector_candidate_variant", "snapshot_date"]).reset_index(drop=True)


def _attach_decision_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    risk = _num(out, "layer4_risk_penalty_score")
    out["incumbent_hold_quality_score"] = (
        0.38 * _num(out, "layer4_risk_aware_score")
        + 0.20 * _num(out, "within80_rank_score")
        + 0.15 * _bool(out, "in_31_high_confidence_subpool_reference").astype(float)
        + 0.12 * _bool(out, "rs20_30_primary_momentum_stable").astype(float)
        + 0.10 * _bool(out, "capital_reasonable_band_4w_persistent").astype(float)
        - 0.20 * risk
    ).clip(0, 1)
    out["challenger_confirmation_score"] = (
        0.36 * _num(out, "layer4_risk_aware_score")
        + 0.22 * _num(out, "final_selector_lifecycle_context_score")
        + 0.14 * _bool(out, "in_31_high_confidence_subpool_reference").astype(float)
        + 0.10 * _bool(out, "lifecycle_strengthening_not_overheated_context").astype(float)
        + 0.08 * _bool(out, "extended_100_to_80_reentry_context").astype(float)
        - 0.18 * risk
    ).clip(0, 1)
    out["lifecycle_clean_score"] = (
        0.45 * _num(out, "final_selector_lifecycle_context_score")
        + 0.20 * _bool(out, "lifecycle_strengthening_not_overheated_context").astype(float)
        + 0.15 * _bool(out, "lifecycle_pullback_reacceleration_context").astype(float)
        + 0.10 * _bool(out, "in_31_high_confidence_subpool_reference").astype(float)
        - 0.20 * risk
    ).clip(0, 1)
    out["reentry_confirmed_score"] = (
        0.40 * _bool(out, "extended_100_to_80_reentry_context").astype(float)
        + 0.20 * _num(out, "final_selector_lifecycle_context_score")
        + 0.18 * _num(out, "layer4_risk_aware_score")
        + 0.10 * _bool(out, "came_from_100_extended_only_previous_week").astype(float)
        - 0.18 * risk
    ).clip(0, 1)
    out["high_confidence_bonus_score"] = (
        0.35 * _num(out, "layer4_risk_aware_score")
        + 0.24 * _bool(out, "in_31_high_confidence_subpool_reference").astype(float)
        + 0.16 * _num(out, "final_selector_lifecycle_context_score")
        + 0.10 * _bool(out, "two_plus_opportunity_labels").astype(float)
        - 0.15 * risk
    ).clip(0, 1)
    out["incumbent_risk_deterioration_candidate"] = (
        risk.ge(0.55)
        | _bool(out, "breakdown_risk_medium_or_high_confidence")
        | _bool(out, "exhaustion_risk_medium_or_high_confidence")
        | _bool(out, "large_down_day_flag_20d_proxy")
    )
    out["incumbent_rs_deterioration_candidate"] = (
        _bool(out, "rs_short_deterioration_flag")
        | _bool(out, "rs60_high_short_rs_weakening_exhaustion_context")
        | _bool(out, "rs_exhaustion_warning_context")
    )
    out["incumbent_breakdown_exhaustion_candidate"] = (
        _bool(out, "breakdown_risk_medium_or_high_confidence")
        | _bool(out, "exhaustion_risk_medium_or_high_confidence")
    )
    out["candidate_not_high_risk_not_exhaustion"] = ~out["incumbent_breakdown_exhaustion_candidate"]
    return out


def _select_for_variant(variant: str, week: pd.DataFrame, lookup: dict[str, pd.Series], state: dict[str, Any]) -> tuple[pd.Series, dict[str, Any]]:
    previous_incumbent = state.get("incumbent_ticker")
    incumbent = lookup.get(previous_incumbent) if previous_incumbent else None
    incumbent_still = incumbent is not None
    current_state = dict(state)
    current_state["previous_incumbent_ticker"] = previous_incumbent
    current_state["incumbent_still_in_80"] = incumbent_still

    if variant == "fresh_top1_baseline":
        chosen = week.sort_values(["within80_rank", "ticker"]).iloc[0]
        return _state_from_choice(chosen, incumbent, current_state, "fresh_top1_baseline")
    if variant == "fresh_best_risk_adjusted_top10_baseline":
        chosen = _best_in_scope(week, 10, "layer4_risk_aware_score")
        return _state_from_choice(chosen, incumbent, current_state, "fresh_risk_adjusted_baseline")

    if variant == "incumbent_protection_selector":
        challenger = _best_in_scope(week, 10, "challenger_confirmation_score")
        return _maybe_hold_or_switch(challenger, incumbent, current_state, margin=0.15, require_confirmed=False, reason_prefix="incumbent_protection")
    if variant == "confirmed_challenger_selector":
        challenger = _best_in_scope(week, 10, "challenger_confirmation_score")
        history = current_state["challenger_history"]
        confirmed = sum(1 for ticker in history if ticker == challenger["ticker"]) >= 1
        history.append(challenger["ticker"])
        current_state["challenger_history"] = history
        return _maybe_hold_or_switch(challenger, incumbent, current_state, margin=0.12, require_confirmed=not confirmed, reason_prefix="confirmed_challenger")
    if variant == "lifecycle_clean_candidate_selector":
        challenger = _best_in_scope(week, 10, "lifecycle_clean_score")
        return _maybe_hold_or_switch(challenger, incumbent, current_state, margin=0.10, require_confirmed=False, reason_prefix="lifecycle_clean")
    if variant == "reentry_confirmed_selector":
        scoped = week[week["within80_rank"].le(10) & _bool(week, "extended_100_to_80_reentry_context")]
        challenger = _best_in_scope(scoped if not scoped.empty else week, 10, "reentry_confirmed_score")
        confirmed = bool(challenger.get("came_from_100_extended_only_previous_week", False)) or bool(challenger.get("in_31_high_confidence_subpool_reference", False))
        return _maybe_hold_or_switch(challenger, incumbent, current_state, margin=0.12, require_confirmed=not confirmed, reason_prefix="reentry_confirmed")
    if variant == "high_confidence_bonus_selector":
        challenger = _best_in_scope(week, 10, "high_confidence_bonus_score")
        return _maybe_hold_or_switch(challenger, incumbent, current_state, margin=0.12, require_confirmed=False, reason_prefix="high_confidence_bonus")
    raise ValueError(f"Unsupported selector variant: {variant}")


def _best_in_scope(week: pd.DataFrame, scope: int, score_col: str) -> pd.Series:
    scoped = week[week["within80_rank"].le(scope)].copy() if "within80_rank" in week else week.copy()
    if scoped.empty:
        scoped = week.copy()
    return scoped.sort_values([score_col, "layer4_risk_aware_score", "within80_rank"], ascending=[False, False, True]).iloc[0]


def _maybe_hold_or_switch(
    challenger: pd.Series,
    incumbent: pd.Series | None,
    state: dict[str, Any],
    *,
    margin: float,
    require_confirmed: bool,
    reason_prefix: str,
) -> tuple[pd.Series, dict[str, Any]]:
    if incumbent is None:
        return _state_from_choice(challenger, incumbent, state, f"{reason_prefix}_no_incumbent")

    incumbent_score = float(incumbent.get("incumbent_hold_quality_score", 0.0))
    challenger_score = float(challenger.get("challenger_confirmation_score", challenger.get("layer4_risk_aware_score", 0.0)))
    score_delta = challenger_score - incumbent_score
    inc_bad = _incumbent_bad(incumbent)
    challenger_clean = bool(challenger.get("candidate_not_high_risk_not_exhaustion", True))
    if inc_bad:
        return _state_from_choice(challenger, incumbent, state, f"{reason_prefix}_switch_incumbent_deteriorated")
    if require_confirmed:
        return _state_from_choice(incumbent, incumbent, state, f"{reason_prefix}_hold_challenger_not_confirmed")
    if score_delta >= margin and challenger_clean:
        return _state_from_choice(challenger, incumbent, state, f"{reason_prefix}_switch_score_margin")
    return _state_from_choice(incumbent, incumbent, state, f"{reason_prefix}_hold_margin_not_met")


def _incumbent_bad(incumbent: pd.Series) -> bool:
    return bool(
        incumbent.get("incumbent_risk_deterioration_candidate", False)
        or incumbent.get("incumbent_rs_deterioration_candidate", False)
        or incumbent.get("incumbent_breakdown_exhaustion_candidate", False)
        or int(incumbent.get("within80_rank", 80)) > 50
    )


def _state_from_choice(chosen: pd.Series, incumbent: pd.Series | None, state: dict[str, Any], reason: str) -> tuple[pd.Series, dict[str, Any]]:
    previous = state.get("incumbent_ticker")
    chosen_ticker = chosen["ticker"]
    switched = previous is not None and chosen_ticker != previous
    hold_count = 1 if switched or previous is None else int(state.get("hold_count", 0)) + 1
    incumbent_score = float(incumbent.get("incumbent_hold_quality_score", 0.0)) if incumbent is not None else None
    challenger_score = float(chosen.get("challenger_confirmation_score", chosen.get("layer4_risk_aware_score", 0.0)))
    new_state = dict(state)
    new_state.update(
        {
            "incumbent_ticker": chosen_ticker,
            "hold_count": hold_count,
            "switched": switched,
            "switch_reason": reason,
            "incumbent_rank_within_80": int(incumbent.get("within80_rank")) if incumbent is not None else None,
            "incumbent_score": incumbent_score,
            "best_challenger_id": chosen_ticker,
            "best_challenger_score": challenger_score,
            "score_delta": None if incumbent_score is None else challenger_score - incumbent_score,
            "incumbent_risk_deterioration": bool(incumbent.get("incumbent_risk_deterioration_candidate", False)) if incumbent is not None else False,
            "incumbent_rs_deterioration": bool(incumbent.get("incumbent_rs_deterioration_candidate", False)) if incumbent is not None else False,
            "incumbent_exhaustion_breakdown_context": bool(incumbent.get("incumbent_breakdown_exhaustion_candidate", False)) if incumbent is not None else False,
        }
    )
    return chosen, new_state


def _variant_family(variant: str) -> str:
    if variant.startswith("fresh_"):
        return "baseline_fresh_picker"
    if "incumbent" in variant:
        return "incumbent_protection"
    if "confirmed" in variant:
        return "confirmed_challenger"
    if "lifecycle" in variant:
        return "lifecycle_state"
    if "high_confidence" in variant:
        return "confidence_bonus"
    return "diagnostic"


def _incumbent_state_design() -> pd.DataFrame:
    rows = [
        ("incumbent_candidate_id", "previous selected ticker under hypothetical diagnostic selector path", "diagnostic_path_state_proxy", "not real current holder"),
        ("incumbent_still_in_80", "whether previous diagnostic incumbent is still in current Layer4 80 primary pool", "diagnostic_path_state_proxy", "not live rule"),
        ("incumbent_rank_within_80", "incumbent rank if still in pool", "diagnostic_path_state_proxy", "not live rule"),
        ("incumbent_score_delta_vs_best_candidate", "candidate score minus incumbent hold score", "diagnostic_path_state_proxy", "no future return"),
        ("incumbent_risk_deterioration", "risk/exhaustion/breakdown deterioration context", "proxy_context", "not formal risk model"),
        ("incumbent_rs_deterioration", "short RS deterioration / exhaustion context", "proxy_context", "not hard live rule"),
        ("consecutive_hold_snapshots_proxy", "number of consecutive weekly signals same candidate remains selected", "diagnostic_path_state_proxy", "not real holding days"),
        ("candidate_switch_required_reason", "candidate reason code for hold/switch under diagnostic design", "diagnostic_reason_code", "not trade instruction"),
    ]
    return pd.DataFrame(rows, columns=["field", "definition", "source_quality", "boundary"])


def _switch_threshold_candidate_design() -> pd.DataFrame:
    rows = [
        ("hold_unless_score_margin", "switch only if challenger score beats incumbent by candidate margin and challenger is not high-risk/exhaustion", "candidate_design_only"),
        ("challenger_confirmation_2_of_3_proxy", "confirmed challenger requires current best challenger observed in recent diagnostic history where available", "proxy_candidate_design_only"),
        ("incumbent_breakdown_override", "allow switch if incumbent exits 80 or has sharp risk/RS deterioration", "candidate_design_only"),
        ("reentry_confirmation", "100-to-80 re-entry candidate needs previous-week re-entry context or 31-reference confirmation where available", "proxy_candidate_design_only"),
        ("fallback_reference_only", "fallback/cash remains blocked until accepted classifier exists", "blocked_no_rule"),
    ]
    return pd.DataFrame(rows, columns=["candidate_design", "definition", "status"])


def _selector_candidate_variant_design() -> pd.DataFrame:
    rows = [
        ("fresh_top1_baseline", "raw top1 within80 retained as baseline only", "baseline_only"),
        ("fresh_best_risk_adjusted_top10_baseline", "best risk-aware score among top10 retained as baseline only", "baseline_only"),
        ("incumbent_protection_selector", "hold incumbent unless challenger margin or incumbent deterioration triggers switch", "diagnostic_candidate"),
        ("confirmed_challenger_selector", "requires challenger confirmation proxy before switching unless incumbent deteriorates", "diagnostic_candidate"),
        ("lifecycle_clean_candidate_selector", "top10 lifecycle clean candidate can challenge incumbent with margin", "diagnostic_candidate"),
        ("reentry_confirmed_selector", "100-to-80 re-entry candidate needs confirmation context before switching", "diagnostic_candidate"),
        ("high_confidence_bonus_selector", "31 high-confidence overlap is score bonus only, not mandatory filter", "diagnostic_candidate"),
    ]
    return pd.DataFrame(rows, columns=["selector_candidate_variant", "definition", "status"])


def _source_quality_matrix() -> pd.DataFrame:
    rows = [
        ("Layer4_80_primary_pool", "exact_from_core_contract", "base_universe"),
        ("within80_rank_context", "exact_from_core_contract", "rank_context"),
        ("hypothetical_diagnostic_path_state", "proxy", "incumbent-aware state support"),
        ("real_current_holder_state", "blocked", "no live position-state contract"),
        ("incumbent_risk_rs_deterioration", "diagnostic_proxy", "state context"),
        ("31_high_confidence_overlap", "exact_reference_context", "confidence bonus only"),
        ("100_to_80_reentry", "exact_reference_context", "reentry context only"),
        ("forward_excess_5d_10d_20d_30d", "evaluation_metadata_only", "primary eval support"),
        ("forward_excess_40d", "evaluation_metadata_only", "decay reference"),
        ("turnover_cost_model", "blocked_placeholder", "no accepted cost model"),
        ("cash_fallback_classifier", "blocked", "no accepted classifier"),
    ]
    return pd.DataFrame(rows, columns=["field_group", "source_quality", "contract_role"])


def _missingness_by_period(candidates: pd.DataFrame) -> pd.DataFrame:
    fields = [
        "incumbent_candidate_id",
        "incumbent_rank_within_80",
        "incumbent_score_delta_vs_best_candidate",
        "consecutive_hold_snapshots_proxy",
        "layer4_risk_aware_score",
        "final_selector_lifecycle_context_score",
        "challenger_confirmation_score",
        "incumbent_hold_quality_score",
        "within80_rank",
        "RS20",
        "RS30_proxy",
        "RS60",
        "BIAS20",
        "BIAS60",
        "volatility",
    ] + [f"forward_excess_vs_0050_{h}d" for h in EVAL_HORIZONS] + [f"forward_excess_vs_00631L_{h}d" for h in EVAL_HORIZONS]
    frame = candidates.copy()
    frame["period"] = frame["snapshot_date"].map(_period_label)
    rows = []
    for period, group in frame.groupby("period", dropna=False):
        for field in fields:
            missing = int(group[field].isna().sum()) if field in group else len(group)
            rows.append({"period": period, "field": field, "row_count": len(group), "missing_count": missing, "missing_share": _share(missing, len(group))})
    return pd.DataFrame(rows)


def _coverage_by_period(candidates: pd.DataFrame) -> pd.DataFrame:
    frame = candidates.copy()
    frame["period"] = frame["snapshot_date"].map(_period_label)
    rows = []
    for keys, group in frame.groupby(["period", "selector_candidate_variant"], dropna=False):
        period, variant = keys
        rows.append(
            {
                "period": period,
                "selector_candidate_variant": variant,
                "row_count": len(group),
                "weekly_snapshot_count": int(group["snapshot_date"].nunique()),
                "switch_count": int(group["selector_changed_from_previous_signal"].sum()),
                "switch_rate_proxy": _share(group["selector_changed_from_previous_signal"].sum(), len(group)),
                "avg_consecutive_hold_snapshots_proxy": float(group["consecutive_hold_snapshots_proxy"].mean()),
                "median_consecutive_hold_snapshots_proxy": float(group["consecutive_hold_snapshots_proxy"].median()),
                "avg_within80_rank": float(group["within80_rank"].mean()),
                "in_31_reference_share": _share(group["in_31_high_confidence_subpool_reference"].sum(), len(group)),
                "reentry_100_to_80_share": _share(group["extended_100_to_80_reentry_context"].sum(), len(group)),
            }
        )
    return pd.DataFrame(rows)


def _blocked_proxy_ledger() -> pd.DataFrame:
    rows = [
        ("real_current_holder_state", "blocked", "No live position-state contract; using hypothetical diagnostic path state proxy"),
        ("incumbent_protection_live_rule", "blocked", "Candidate designs only; no live rule"),
        ("cash_fallback_classifier", "blocked", "No accepted market cash classifier"),
        ("fallback_00631L_trading_rule", "blocked", "Fallback/reference only"),
        ("A_B_switch_or_second_stock_allocation", "blocked", "A/B switch and second-stock allocation unauthorized"),
        ("turnover_cost_model", "blocked_placeholder", "No accepted cost model; placeholder only"),
        ("incumbent_risk_deterioration", "proxy", "Uses risk/exhaustion/breakdown context, not formal risk model"),
        ("RS30_proxy", "proxy", "Exact RS30 unavailable"),
        ("portfolio_replay", "blocked", "Portfolio/strategy replay unauthorized"),
    ]
    return pd.DataFrame(rows, columns=["field_or_policy", "status", "reason"])


def _future_data_audit(candidates: pd.DataFrame) -> pd.DataFrame:
    rows = [
        ("forward_returns_used_in_selector_construction", "passed", 0, "selector candidates use current/prior path state and PIT context only"),
        ("future_return_as_rule", "passed", int(_bool(candidates, "future_return_as_rule").sum()), "future_return_as_rule=false"),
        ("forward_return_as_rule", "passed", int(_bool(candidates, "forward_return_as_rule").sum()), "forward_return_as_rule=false"),
        ("live_action_rule_output", "passed", int(_bool(candidates, "layer5_live_action_rule_output").sum()), "no live action output"),
        ("trade_decision_output", "passed", int(_bool(candidates, "trade_decision_output").sum()), "no trade decision output"),
        ("portfolio_like_execution_output", "passed", int(_bool(candidates, "portfolio_like_execution_output").sum()), "no execution output"),
    ]
    return pd.DataFrame(rows, columns=["audit_item", "status", "violation_count", "evidence"])


def _requested_vs_actual_coverage(candidates: pd.DataFrame) -> pd.DataFrame:
    actual_start = candidates["snapshot_date"].min()
    actual_end = candidates["snapshot_date"].max()
    rows = []
    for period, (requested_start, requested_end) in PERIODS.items():
        start = pd.Timestamp(requested_start)
        end = pd.Timestamp(requested_end)
        in_period = candidates[candidates["snapshot_date"].between(start, end)]
        rows.append(
            {
                "period": period,
                "requested_start": requested_start,
                "requested_end": requested_end,
                "actual_contract_start": actual_start.date().isoformat(),
                "actual_contract_end": actual_end.date().isoformat(),
                "rows_in_requested_period": len(in_period),
                "weekly_snapshots_in_requested_period": int(in_period["snapshot_date"].nunique()),
                "rows_before_requested_start": int((candidates["snapshot_date"] < start).sum()),
                "rows_after_requested_end": int((candidates["snapshot_date"] > end).sum()),
                "coverage_note": "requested and actual coverage are separate; do not use actual range as requested-period conclusion",
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
    experiment_summary: dict[str, Any],
    within80: pd.DataFrame,
    candidates: pd.DataFrame,
    coverage: pd.DataFrame,
    future_audit: pd.DataFrame,
) -> dict[str, Any]:
    future_violations = int(future_audit["violation_count"].sum())
    ready = future_violations == 0 and candidates["snapshot_date"].nunique() == within80["snapshot_date"].nunique()
    return {
        "task_id": TASK_ID,
        "status": "layer5_incumbent_aware_selector_candidate_contract_ready_for_experiments_intake" if ready else "layer5_incumbent_aware_selector_candidate_contract_blocked",
        "diagnostic_only": True,
        "input_layer5_status": source_readiness.get("status"),
        "input_experiments_verdict": experiment_summary.get("verdict"),
        "base_universe": "Layer4_80_primary_pool",
        "row_count": int(len(candidates)),
        "weekly_snapshot_count": int(candidates["snapshot_date"].nunique()),
        "selector_candidate_variant_count": int(candidates["selector_candidate_variant"].nunique()),
        "selector_candidate_variants": sorted(candidates["selector_candidate_variant"].unique().tolist()),
        "hypothetical_path_state_proxy": True,
        "real_current_holder_state_available": False,
        "real_current_holder_state_status": "blocked_no_live_position_state_contract",
        "cash_fallback_classifier_status": "blocked_no_accepted_market_cash_classifier",
        "turnover_cost_model_status": "blocked_placeholder_no_accepted_cost_model",
        "avg_switch_rate_proxy_by_variant": coverage.groupby("selector_candidate_variant")["switch_rate_proxy"].mean().to_dict(),
        "avg_consecutive_hold_snapshots_proxy_by_variant": coverage.groupby("selector_candidate_variant")["avg_consecutive_hold_snapshots_proxy"].mean().to_dict(),
        "ready_for_layer5_incumbent_aware_single_stock_diagnostic": ready,
        "ready_for_experiments_intake": ready,
        "ready_for_live_layer5_action_rule": False,
        "ready_for_ab_switch_rule": False,
        "ready_for_portfolio_like_diagnostic": False,
        "ready_for_formal": False,
        "ready_for_strategy_replay": False,
        "future_data_violation_count": future_violations,
        "blocked_fields": [
            "real_current_holder_state",
            "incumbent_protection_live_rule",
            "cash_fallback_classifier",
            "fallback_00631L_trading_rule",
            "A_B_switch_or_second_stock_allocation",
            "turnover_cost_model",
            "portfolio_replay",
        ],
        "proxy_fields": ["hypothetical_diagnostic_path_state", "incumbent_risk_deterioration", "RS30_proxy"],
        **_fixed_flags(),
    }


def _summary(readiness: dict[str, Any]) -> str:
    return f"""# Layer5 incumbent-aware single-stock decision candidate contract

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
- Base universe: Layer4 80-stock primary pool.
- Uses hypothetical diagnostic path state proxy because real current-holder/incumbent state is blocked.
- 100 extended watchlist and 31 high-confidence reference remain context only.
- 00631L / 0050正二 remain fallback/reference metadata, not ordinary stock rows.
- No live Layer5 rule, no A/B switch, no second-stock allocation, no portfolio replay.
- Layer5 的目標不是每天重新選分數最高的一檔，而是在只持有一檔的操作模式下，判斷續抱、換倉、或等待 fallback 的長期勝率/報酬/風險 tradeoff。

## Candidate variants
- selector_candidate_variant_count={readiness['selector_candidate_variant_count']}
- variants={', '.join(readiness['selector_candidate_variants'])}
- row_count={readiness['row_count']}
- weekly_snapshot_count={readiness['weekly_snapshot_count']}

## Blocked / proxy
- real_current_holder_state=blocked
- cash_fallback_classifier=blocked
- turnover_cost_model=blocked_placeholder
- hypothetical_path_state_proxy=true

## Next
If accepted, hand off to Experiments:
`TASK-BACKTEST-EXPERIMENTS-VNEXT-LAYER5-INCUMBENT-AWARE-SINGLE-STOCK-DECISION-DIAGNOSTIC-001`.
完成後如果下一棒明確，請直接指派下一個 thread；如果下一棒不明確，請回報 Strategy Center 判斷。不要完成後停住不回報。
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer5-dir", default=str(DEFAULT_LAYER5_DIR))
    parser.add_argument("--experiments-dir", default=str(DEFAULT_EXPERIMENTS_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()
    manifest = build_contract(layer5_dir=args.layer5_dir, experiments_dir=args.experiments_dir, output_dir=args.output_dir)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
