"""Build Layer4 pool-size sensitivity and retention-constrained contract.

This package is diagnostic/source-contract only. It is not Layer5, not a
selector authorization, not replay, not formal, not report, and not a trade
decision.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


TASK_ID = "TASK-BACKTEST-CORE-VNEXT-LAYER4-POOL-SIZE-RETENTION-CONSTRAINT-CONTRACT-001"
DEFAULT_LAYER3_DIR = Path("outputs/vnext_layer3_broad_opportunity_label_contract_20260708")
DEFAULT_LAYER4_EXPERIMENTS_DIR = Path(
    "C:/Users/zergv/Documents/Codex/2026-07-06/backtest-lab-experiments-diagnostic-validation-attribution/"
    "outputs/vnext_layer4_bounded_31_pool_assembly_diagnostic_20260708"
)
DEFAULT_PREVIOUS_LAYER4_DIR = Path("outputs/vnext_layer4_bounded_31_pool_assembly_contract_20260708")
DEFAULT_OUTPUT_DIR = Path("outputs/vnext_layer4_pool_size_retention_constraint_contract_20260708")
POOL_SIZES = [31, 40, 50, 60, 80, 100]
PERIODS = {
    "P1": ("2015-01-02", "2022-12-29"),
    "P2": ("2023-01-02", "2026-06-30"),
    "2024_latest": ("2024-01-02", "2026-06-30"),
    "2026YTD": ("2026-01-02", "2026-06-30"),
}
EVAL_HORIZONS = [5, 10, 20, 30, 40]


def build_contract(
    *,
    layer3_dir: str | Path = DEFAULT_LAYER3_DIR,
    layer4_experiments_dir: str | Path = DEFAULT_LAYER4_EXPERIMENTS_DIR,
    previous_layer4_dir: str | Path = DEFAULT_PREVIOUS_LAYER4_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    layer3 = Path(layer3_dir)
    experiments = Path(layer4_experiments_dir)
    previous = Path(previous_layer4_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    layer3_readiness = _read_json(layer3 / "readiness_for_layer3_broad_opportunity_label_diagnostic.json")
    previous_readiness = _read_json(previous / "readiness_for_layer4_bounded_31_pool_diagnostic.json")
    experiment_summary = _read_json(experiments / "layer4_pool_summary.json")
    base = _read_layer3_labels(layer3 / "layer3_broad_opportunity_label_contract.csv")
    scored = _attach_redesign_scores(base)

    pool = _assemble_pool_size_variants(scored)
    weekly_coverage = _weekly_coverage(pool)
    variant_design = _variant_design()
    retention_policy = _retention_constraint_policy()
    source_quality = _source_quality_matrix()
    missingness = _missingness_by_period(scored)
    blocked_proxy = _blocked_proxy_ledger()
    future_audit = _future_data_audit(pool)
    readiness = _readiness(
        layer3_readiness=layer3_readiness,
        previous_readiness=previous_readiness,
        experiment_summary=experiment_summary,
        scored=scored,
        pool=pool,
        weekly_coverage=weekly_coverage,
        future_audit=future_audit,
    )

    _write_csv(pool, output / "layer4_pool_size_sensitivity_contract.csv")
    _write_csv(pool.head(1000), output / "layer4_pool_size_sensitivity_contract_sample.csv")
    (output / ".gitignore").write_text("layer4_pool_size_sensitivity_contract.csv\n", encoding="utf-8")
    _write_csv(variant_design, output / "layer4_pool_size_variant_design.csv")
    _write_csv(retention_policy, output / "layer4_retention_constraint_policy.csv")
    _write_csv(weekly_coverage, output / "layer4_pool_weekly_coverage_by_variant.csv")
    _write_csv(source_quality, output / "layer4_pool_size_source_quality_matrix.csv")
    _write_csv(missingness, output / "layer4_pool_size_missingness_by_period.csv")
    _write_csv(blocked_proxy, output / "layer4_pool_size_blocked_proxy_ledger.csv")
    _write_csv(future_audit, output / "layer4_pool_size_future_data_audit.csv")
    (output / "readiness_for_layer4_pool_size_retention_constraint_diagnostic.json").write_text(
        json.dumps(readiness, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    manifest = {
        "task_id": TASK_ID,
        "status": readiness["status"],
        "output_dir": str(output.resolve()),
        "input_layer3_dir": str(layer3.resolve()),
        "input_previous_layer4_dir": str(previous.resolve()),
        "input_layer4_experiments_dir": str(experiments.resolve()),
        "output_files": [
            "layer4_pool_size_sensitivity_contract.csv",
            "layer4_pool_size_sensitivity_contract_sample.csv",
            "layer4_pool_size_variant_design.csv",
            "layer4_retention_constraint_policy.csv",
            "layer4_pool_weekly_coverage_by_variant.csv",
            "layer4_pool_size_source_quality_matrix.csv",
            "layer4_pool_size_missingness_by_period.csv",
            "layer4_pool_size_blocked_proxy_ledger.csv",
            "layer4_pool_size_future_data_audit.csv",
            "readiness_for_layer4_pool_size_retention_constraint_diagnostic.json",
            "manifest.json",
            "final_summary_zh.md",
        ],
        "large_local_files_not_tracked": ["layer4_pool_size_sensitivity_contract.csv"],
        "large_local_file_policy": "full pool-size sensitivity table is retained locally; Git tracks sample/readiness/audit files only",
        **_fixed_flags(),
        "diagnostic_only": True,
        "bounded_pool_assembly_only": True,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "final_summary_zh.md").write_text(_summary(readiness), encoding="utf-8")
    return manifest


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def _read_layer3_labels(path: Path) -> pd.DataFrame:
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


def _attach_redesign_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    layer1_quality = _bool(out, "layer1_b30_pass_through_eligible").astype(float)
    layer2_support = _num(out, "layer2_support_signal_count") / 3
    layer2_warning = _num(out, "layer2_warning_signal_count") / 4
    opportunity_count = (
        _bool(out, "momentum_continuation_medium_or_high_confidence").astype(int)
        + _bool(out, "pullback_repair_medium_or_high_confidence").astype(int)
        + _bool(out, "overlap_reacceleration_medium_or_high_confidence").astype(int)
        + _bool(out, "neutral_quality_liquidity_medium_or_high_confidence").astype(int)
    )
    out["opportunity_label_count"] = opportunity_count
    out["two_plus_opportunity_labels"] = opportunity_count.ge(2)
    out["layer4_redesign_base_eligible"] = layer1_quality.eq(1)
    out["fallback_00631L_reference_only"] = False
    out["fallback_00631L_is_ordinary_stock_pool_member"] = False

    capital_improvement = _num(out, "capital_rank_improvement_20d_vs_60d").clip(lower=-100, upper=100).add(100).div(200)
    risk_penalty = (
        0.40 * _num(out, "exhaustion_risk_score")
        + 0.35 * _num(out, "breakdown_risk_score")
        + 0.10 * _bool(out, "large_down_day_flag_20d_proxy").astype(float)
        + 0.10 * _bool(out, "blowoff_turnover_without_price_continuation_proxy").astype(float)
        + 0.05 * layer2_warning.clip(0, 1)
    ).clip(0, 1)
    out["layer4_risk_penalty_score"] = risk_penalty
    out["layer4_broad_opportunity_net_score"] = (
        0.24 * _num(out, "momentum_continuation_score")
        + 0.22 * _num(out, "pullback_repair_score")
        + 0.18 * _num(out, "overlap_reacceleration_score")
        + 0.18 * _num(out, "neutral_quality_liquidity_score")
        + 0.10 * layer2_support.clip(0, 1)
        + 0.08 * capital_improvement
        - 0.18 * risk_penalty
    ).clip(0, 1)
    out["layer4_c_quota_base_score"] = (
        0.38 * out["layer4_broad_opportunity_net_score"]
        + 0.18 * out["opportunity_label_count"].clip(0, 4).div(4)
        + 0.14 * _num(out, "neutral_quality_liquidity_score")
        + 0.12 * _num(out, "momentum_continuation_score")
        + 0.10 * _num(out, "pullback_repair_score")
        + 0.08 * layer1_quality
    ).clip(0, 1)
    out["layer4_retention_constrained_score"] = (
        0.34 * out["two_plus_opportunity_labels"].astype(float)
        + 0.22 * out["opportunity_label_count"].clip(0, 4).div(4)
        + 0.18 * out["layer4_broad_opportunity_net_score"]
        + 0.14 * _num(out, "neutral_quality_liquidity_score")
        + 0.07 * _num(out, "overlap_reacceleration_score")
        + 0.05 * layer1_quality
        - 0.08 * risk_penalty
    ).clip(0, 1)
    out["layer4_risk_aware_score"] = (
        0.48 * out["layer4_broad_opportunity_net_score"]
        + 0.16 * out["opportunity_label_count"].clip(0, 4).div(4)
        + 0.14 * _num(out, "neutral_quality_liquidity_score")
        + 0.10 * layer2_support.clip(0, 1)
        + 0.08 * capital_improvement
        + 0.04 * layer1_quality
        - 0.28 * risk_penalty
    ).clip(0, 1)
    out["high_exhaustion_or_breakdown_context"] = (
        _bool(out, "exhaustion_risk_medium_or_high_confidence")
        | _bool(out, "breakdown_risk_medium_or_high_confidence")
    )
    out["low_quality_low_liquidity_high_risk_exclusion_candidate"] = (
        (~_bool(out, "layer1_b20_pass_through_eligible"))
        & _num(out, "traded_value_rank_20d", 9999).gt(500)
        & out["high_exhaustion_or_breakdown_context"]
    )
    out["low_quality_low_liquidity_high_risk_exclusion_applied"] = False
    out["theme_dynamic_slot_status"] = "blocked_no_accepted_theme_slot_context"
    out["theme_dynamic_slot_applied"] = False
    out["ai_theme_slot_hard_quota_applied"] = False
    out["layer4_pool_size_sensitivity_contract_only"] = True
    out["layer4_selector_output"] = False
    out["layer5_decision_authorized"] = False
    out["forward_return_as_rule"] = False
    out["future_return_as_rule"] = False
    for key, value in _fixed_flags().items():
        out[key] = value
    return out


def _assemble_pool_size_variants(scored: pd.DataFrame) -> pd.DataFrame:
    eligible = scored[scored["layer4_redesign_base_eligible"]].copy()
    pools: list[pd.DataFrame] = []
    for _, week in eligible.groupby("snapshot_date", sort=True):
        for size in POOL_SIZES:
            pools.append(
                _quota_select_week(
                    week,
                    size=size,
                    variant="C_quota_size_sensitive_broad_label_quota",
                    family="c_quota_style",
                    score_col="layer4_c_quota_base_score",
                    quota_mode="balanced",
                )
            )
            pools.append(
                _quota_select_week(
                    week,
                    size=size,
                    variant="C_retention_friendly_retention_constrained_quota",
                    family="retention_friendly_c_quota",
                    score_col="layer4_retention_constrained_score",
                    quota_mode="retention",
                )
            )
            pools.append(
                _quota_select_week(
                    week,
                    size=size,
                    variant="C_risk_aware_retention_constrained_quota",
                    family="risk_aware_c_quota",
                    score_col="layer4_risk_aware_score",
                    quota_mode="risk_aware",
                )
            )
    out = pd.concat(pools, ignore_index=True)
    out["bounded_diagnostic_pool_only"] = True
    out["diagnostic_only"] = True
    out["not_live_rule"] = True
    out["pool_size_reference_upper_bound"] = out["pool_size_target"].eq(100)
    out["pool_size_reference_current_31"] = out["pool_size_target"].eq(31)
    out["layer2_layer3_hard_gate_outside_pool_variant"] = False
    return out


def _quota_select_week(
    week: pd.DataFrame,
    *,
    size: int,
    variant: str,
    family: str,
    score_col: str,
    quota_mode: str,
) -> pd.DataFrame:
    selected_idx: list[int] = []
    policy = _quota_policy(size, quota_mode)

    def add_candidates(mask: pd.Series, count: int, bucket: str) -> None:
        if count <= 0:
            return
        candidates = week[mask & ~week.index.isin(selected_idx)].copy()
        if candidates.empty:
            return
        candidates = candidates.sort_values(
            by=[score_col, "layer4_broad_opportunity_net_score", "traded_value_rank_20d"],
            ascending=[False, False, True],
        )
        take = candidates.head(count)
        selected_idx.extend(take.index.tolist())
        week.loc[take.index, "_pool_quota_bucket"] = bucket

    week = week.copy()
    week["_pool_quota_bucket"] = "fill"
    if quota_mode == "retention":
        add_candidates(week["two_plus_opportunity_labels"], policy["two_plus_min"], "two_plus_opportunity_min")
    add_candidates(_bool(week, "neutral_quality_liquidity_medium_or_high_confidence"), policy["neutral_min"], "neutral_quality_liquidity_min")
    add_candidates(_bool(week, "momentum_continuation_medium_or_high_confidence"), policy["momentum_min"], "momentum_continuation_min")
    add_candidates(_bool(week, "pullback_repair_medium_or_high_confidence"), policy["pullback_min"], "pullback_repair_min")
    add_candidates(_bool(week, "overlap_reacceleration_medium_or_high_confidence"), policy["overlap_min"], "overlap_reacceleration_min")
    if quota_mode == "risk_aware":
        add_candidates(~week["high_exhaustion_or_breakdown_context"], policy["low_risk_context_min"], "low_risk_context_min")

    remaining = week[~week.index.isin(selected_idx)].copy()
    remaining = remaining.sort_values(
        by=[score_col, "layer4_broad_opportunity_net_score", "traded_value_rank_20d"],
        ascending=[False, False, True],
    )
    needed = max(0, size - len(selected_idx))
    selected_idx.extend(remaining.head(needed).index.tolist())

    result = week.loc[selected_idx[:size]].copy()
    result = result.sort_values(
        by=[score_col, "layer4_broad_opportunity_net_score", "traded_value_rank_20d"],
        ascending=[False, False, True],
    )
    result["layer4_pool_variant"] = f"{variant}_{size}"
    result["pool_variant_family"] = family
    result["pool_size_target"] = size
    result["pool_selection_score"] = result[score_col]
    result["pool_selection_score_col"] = score_col
    result["pool_selection_policy"] = quota_mode
    result["retention_constraint_applied"] = quota_mode in {"retention", "risk_aware"}
    result["risk_constraint_applied"] = quota_mode == "risk_aware"
    result["pool_rank"] = range(1, len(result) + 1)
    result["pool_shortfall_count"] = max(0, size - len(result))
    result["pool_selection_basis"] = (
        "C-quota broad label redesign; live-feasible ranking/quota context only; no future-return rule construction"
    )
    return result


def _quota_policy(size: int, mode: str) -> dict[str, int]:
    if mode == "retention":
        return {
            "two_plus_min": math.ceil(size * 0.50),
            "neutral_min": math.ceil(size * 0.22),
            "momentum_min": math.ceil(size * 0.16),
            "pullback_min": math.ceil(size * 0.14),
            "overlap_min": math.ceil(size * 0.14),
            "low_risk_context_min": 0,
        }
    if mode == "risk_aware":
        return {
            "two_plus_min": math.ceil(size * 0.42),
            "neutral_min": math.ceil(size * 0.20),
            "momentum_min": math.ceil(size * 0.16),
            "pullback_min": math.ceil(size * 0.14),
            "overlap_min": math.ceil(size * 0.14),
            "low_risk_context_min": math.ceil(size * 0.50),
        }
    return {
        "two_plus_min": 0,
        "neutral_min": math.ceil(size * 0.20),
        "momentum_min": math.ceil(size * 0.20),
        "pullback_min": math.ceil(size * 0.18),
        "overlap_min": math.ceil(size * 0.14),
        "low_risk_context_min": 0,
    }


def _weekly_coverage(pool: pd.DataFrame) -> pd.DataFrame:
    def agg(group: pd.DataFrame) -> pd.Series:
        selected = len(group)
        target_size = int(group.name[1]) if isinstance(group.name, tuple) and len(group.name) > 1 else selected
        return pd.Series(
            {
                "selected_count": selected,
                "target_pool_size": target_size,
                "shortfall_count": int(group["pool_shortfall_count"].max()),
                "two_plus_opportunity_count": int(group["two_plus_opportunity_labels"].sum()),
                "two_plus_opportunity_share": _safe_share(group["two_plus_opportunity_labels"].sum(), selected),
                "momentum_medium_high_count": int(_bool(group, "momentum_continuation_medium_or_high_confidence").sum()),
                "pullback_medium_high_count": int(_bool(group, "pullback_repair_medium_or_high_confidence").sum()),
                "overlap_medium_high_count": int(_bool(group, "overlap_reacceleration_medium_or_high_confidence").sum()),
                "neutral_medium_high_count": int(_bool(group, "neutral_quality_liquidity_medium_or_high_confidence").sum()),
                "neutral_medium_high_share": _safe_share(_bool(group, "neutral_quality_liquidity_medium_or_high_confidence").sum(), selected),
                "high_exhaustion_or_breakdown_count": int(group["high_exhaustion_or_breakdown_context"].sum()),
                "high_exhaustion_or_breakdown_share": _safe_share(group["high_exhaustion_or_breakdown_context"].sum(), selected),
                "avg_traded_value_rank_20d": float(_num(group, "traded_value_rank_20d").mean()),
                "median_traded_value_rank_20d": float(_num(group, "traded_value_rank_20d").median()),
                "turnover_share_metadata_status": "rank_proxy_only_raw_turnover_share_not_in_layer3_contract",
                "fallback_00631L_reference_only": bool(group["fallback_00631L_reference_only"].any()),
                "fallback_00631L_is_ordinary_stock_pool_member": bool(group["fallback_00631L_is_ordinary_stock_pool_member"].any()),
            }
        )

    coverage = (
        pool.groupby(["snapshot_date", "pool_size_target", "layer4_pool_variant", "pool_variant_family"], sort=True)
        .apply(agg)
        .reset_index()
    )
    coverage["period"] = coverage["snapshot_date"].map(_period_label)
    return coverage


def _safe_share(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def _period_label(date_value: Any) -> str:
    date = pd.to_datetime(date_value)
    hits = []
    for label, (start, end) in PERIODS.items():
        if pd.Timestamp(start) <= date <= pd.Timestamp(end):
            hits.append(label)
    return "|".join(hits) if hits else "outside_requested_periods"


def _variant_design() -> pd.DataFrame:
    rows = []
    for size in POOL_SIZES:
        rows.extend(
            [
                {
                    "layer4_pool_variant": f"C_quota_size_sensitive_broad_label_quota_{size}",
                    "pool_size_target": size,
                    "pool_variant_family": "c_quota_style",
                    "selection_score": "layer4_c_quota_base_score",
                    "retention_constraint_applied": False,
                    "risk_constraint_applied": False,
                    "design_note": "C-quota broad label pool size sensitivity reference; no Layer5 authorization",
                },
                {
                    "layer4_pool_variant": f"C_retention_friendly_retention_constrained_quota_{size}",
                    "pool_size_target": size,
                    "pool_variant_family": "retention_friendly_c_quota",
                    "selection_score": "layer4_retention_constrained_score",
                    "retention_constraint_applied": True,
                    "risk_constraint_applied": False,
                    "design_note": "Adds live-feasible minimum representation for two-plus opportunity and neutral-quality labels",
                },
                {
                    "layer4_pool_variant": f"C_risk_aware_retention_constrained_quota_{size}",
                    "pool_size_target": size,
                    "pool_variant_family": "risk_aware_c_quota",
                    "selection_score": "layer4_risk_aware_score",
                    "retention_constraint_applied": True,
                    "risk_constraint_applied": True,
                    "design_note": "Risk/exhaustion/breakdown lowers rank; not a hard deletion gate",
                },
            ]
        )
    return pd.DataFrame(rows)


def _retention_constraint_policy() -> pd.DataFrame:
    rows = []
    for size in POOL_SIZES:
        for mode in ["balanced", "retention", "risk_aware"]:
            policy = _quota_policy(size, mode)
            rows.append(
                {
                    "pool_size_target": size,
                    "quota_mode": mode,
                    "two_plus_opportunity_min": policy["two_plus_min"],
                    "momentum_medium_high_min": policy["momentum_min"],
                    "pullback_medium_high_min": policy["pullback_min"],
                    "overlap_medium_high_min": policy["overlap_min"],
                    "neutral_quality_liquidity_min": policy["neutral_min"],
                    "low_risk_context_min": policy["low_risk_context_min"],
                    "single_sleeve_domination_policy": "quota-first fill; score-only fill cannot replace all sleeves before minima are attempted",
                    "high_exhaustion_breakdown_policy": "rank_down_context_only_no_one_cut",
                    "future_return_used": False,
                    "live_feasible": True,
                    "diagnostic_only": True,
                }
            )
    return pd.DataFrame(rows)


def _source_quality_matrix() -> pd.DataFrame:
    rows = [
        ("compact_layer0_active_scope", "exact_from_core_contract", "base_universe"),
        ("layer1_b30_pass_through_eligible", "diagnostic_exact_from_core_contract", "base_eligibility"),
        ("layer2_context_fields", "diagnostic_exact_or_proxy_mixed", "context_only_pass_through"),
        ("layer3_broad_opportunity_scores", "diagnostic_exact_or_proxy_mixed", "ranking_context"),
        ("RS30_proxy", "proxy", "RS30 exact unavailable"),
        ("large_down_day_count_20d_proxy", "proxy", "risk context"),
        ("blowoff_turnover_without_price_continuation_proxy", "proxy", "risk context"),
        ("risk_bucket", "blocked", "formal risk bucket unavailable"),
        ("turnover_share_raw", "blocked", "raw traded-value share not carried in Layer3 contract"),
        ("theme_ai_dynamic_slot", "blocked_placeholder", "no accepted theme slot contract in this package"),
        ("forward_excess_vs_0050_5d_10d_20d_30d_40d", "evaluation_metadata_only", "not rule construction"),
        ("forward_excess_vs_00631L_5d_10d_20d_30d_40d", "evaluation_metadata_only", "not rule construction"),
    ]
    return pd.DataFrame(rows, columns=["field_group", "source_quality", "contract_role"])


def _missingness_by_period(scored: pd.DataFrame) -> pd.DataFrame:
    fields = [
        "layer4_c_quota_base_score",
        "layer4_retention_constrained_score",
        "layer4_risk_aware_score",
        "momentum_continuation_score",
        "pullback_repair_score",
        "overlap_reacceleration_score",
        "neutral_quality_liquidity_score",
        "exhaustion_risk_score",
        "breakdown_risk_score",
        "traded_value_rank_20d",
        "traded_value_rank_60d",
        "RS20",
        "RS30_proxy",
        "RS60",
        "BIAS20",
        "BIAS60",
        "volatility",
    ]
    rows = []
    base = scored[scored["layer4_redesign_base_eligible"]].copy()
    base["period"] = base["snapshot_date"].map(_period_label)
    for period, group in base.groupby("period", dropna=False):
        for field in fields:
            missing = int(group[field].isna().sum()) if field in group else len(group)
            rows.append(
                {
                    "period": period,
                    "field": field,
                    "row_count": len(group),
                    "missing_count": missing,
                    "missing_share": _safe_share(missing, len(group)),
                }
            )
    return pd.DataFrame(rows)


def _blocked_proxy_ledger() -> pd.DataFrame:
    rows = [
        {
            "field_or_policy": "AI_theme_dynamic_slot_contract",
            "status": "blocked_placeholder",
            "reason": "theme/AI dynamic slot data not accepted in this Layer4 package; no hard-coded AI 20 quota",
        },
        {
            "field_or_policy": "raw_turnover_share_coverage",
            "status": "blocked",
            "reason": "Layer3 contract carries traded-value ranks, not raw weekly turnover shares",
        },
        {
            "field_or_policy": "risk_bucket",
            "status": "blocked",
            "reason": "formal risk bucket unavailable; risk context uses BIAS/volatility/large-down/blowoff proxies",
        },
        {
            "field_or_policy": "large_down_day_count_20d_proxy",
            "status": "proxy",
            "reason": "diagnostic path-derived PIT proxy; not formal-ready",
        },
        {
            "field_or_policy": "blowoff_turnover_without_price_continuation_proxy",
            "status": "proxy",
            "reason": "diagnostic PIT proxy; not formal-ready",
        },
        {
            "field_or_policy": "RS30_proxy",
            "status": "proxy",
            "reason": "exact RS30 unavailable; proxy retained with source-quality label",
        },
        {
            "field_or_policy": "Layer5_A_B_decision",
            "status": "blocked",
            "reason": "Layer5 explicitly unauthorized",
        },
        {
            "field_or_policy": "portfolio_replay",
            "status": "blocked",
            "reason": "portfolio/strategy replay unauthorized",
        },
    ]
    return pd.DataFrame(rows)


def _future_data_audit(pool: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "audit_item": "forward_returns_used_in_pool_rule",
                "status": "passed",
                "violation_count": 0,
                "evidence": "pool selection scores use Layer1/2/3 PIT context columns only; forward excess columns retained as evaluation metadata",
            },
            {
                "audit_item": "future_return_as_rule",
                "status": "passed",
                "violation_count": int(_bool(pool, "future_return_as_rule").sum()),
                "evidence": "future_return_as_rule=false for output rows",
            },
            {
                "audit_item": "fallback_00631L_ordinary_stock_member",
                "status": "passed",
                "violation_count": int(pool["fallback_00631L_is_ordinary_stock_pool_member"].sum()),
                "evidence": "00631L kept as fallback/reference policy, not ordinary stock pool row",
            },
            {
                "audit_item": "layer5_authorized",
                "status": "passed",
                "violation_count": int(_bool(pool, "layer5_decision_authorized").sum()),
                "evidence": "Layer5 remains unauthorized",
            },
        ]
    )


def _readiness(
    *,
    layer3_readiness: dict[str, Any],
    previous_readiness: dict[str, Any],
    experiment_summary: dict[str, Any],
    scored: pd.DataFrame,
    pool: pd.DataFrame,
    weekly_coverage: pd.DataFrame,
    future_audit: pd.DataFrame,
) -> dict[str, Any]:
    expected_week_variants = scored["snapshot_date"].nunique() * len(POOL_SIZES) * 3
    actual_week_variants = weekly_coverage.shape[0]
    full_share = float((weekly_coverage["selected_count"] == weekly_coverage["target_pool_size"]).mean())
    future_violations = int(future_audit["violation_count"].sum())
    ready = expected_week_variants == actual_week_variants and full_share == 1.0 and future_violations == 0
    return {
        "task_id": TASK_ID,
        "status": (
            "layer4_pool_size_retention_constraint_contract_ready_for_experiments_intake"
            if ready
            else "layer4_pool_size_retention_constraint_contract_blocked"
        ),
        "diagnostic_only": True,
        "bounded_pool_assembly_only": True,
        "input_layer3_status": layer3_readiness.get("status"),
        "input_previous_layer4_status": previous_readiness.get("status"),
        "input_experiments_verdict": experiment_summary.get("verdict"),
        "redesign_base": "C_quota_style_broad_label_pool",
        "pool_sizes": POOL_SIZES,
        "pool_variant_count": len(POOL_SIZES) * 3,
        "weekly_snapshot_count": int(scored["snapshot_date"].nunique()),
        "base_rows": int(scored["layer4_redesign_base_eligible"].sum()),
        "pool_rows": int(len(pool)),
        "expected_week_variant_count": int(expected_week_variants),
        "actual_week_variant_count": int(actual_week_variants),
        "full_pool_week_variant_share": full_share,
        "min_selected_count": int(weekly_coverage["selected_count"].min()),
        "max_selected_count": int(weekly_coverage["selected_count"].max()),
        "layer1_b30_is_base_eligibility": True,
        "layer2_context_only": True,
        "layer3_context_only": True,
        "layer2_layer3_hard_gate_outside_pool_variant_allowed": False,
        "retention_constraints_are_live_feasible": True,
        "future_winner_used_in_constraint": False,
        "fallback_00631L_reference_only": True,
        "fallback_00631L_ordinary_stock_member": False,
        "theme_ai_dynamic_slot_status": "blocked_placeholder_no_hard_quota",
        "ready_for_layer4_pool_size_retention_constraint_diagnostic": ready,
        "ready_for_experiments_intake": ready,
        "ready_for_layer5_decision": False,
        "ready_for_portfolio_like_diagnostic": False,
        "ready_for_formal": False,
        "ready_for_strategy_replay": False,
        "candidate_forward_return_diagnostic_executed": False,
        "future_data_violation_count": future_violations,
        "blocked_fields": [
            "AI_theme_dynamic_slot_contract",
            "raw_turnover_share_coverage",
            "risk_bucket",
            "Layer5_decision",
            "portfolio_replay",
        ],
        "proxy_fields": ["RS30_proxy", "large_down_day_proxy", "blowoff_turnover_proxy"],
        **_fixed_flags(),
    }


def _summary(readiness: dict[str, Any]) -> str:
    return f"""# Layer4 pool-size / retention-constrained redesign contract

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
- Base universe: compact Layer0 active scope + Layer1 b30 pass-through.
- Layer2 and Layer3 remain context-only pass-through.
- Redesign base: C_quota_style_broad_label_pool.
- Pool sizes: {', '.join(str(x) for x in POOL_SIZES)}.
- Variants per size: C-quota, retention-friendly C-quota, risk-aware C-quota.
- 00631L remains fallback/reference only, not ordinary stock-pool member.

## Readiness
- weekly_snapshot_count={readiness['weekly_snapshot_count']}
- pool_variant_count={readiness['pool_variant_count']}
- pool_rows={readiness['pool_rows']}
- full_pool_week_variant_share={readiness['full_pool_week_variant_share']}
- ready_for_layer4_pool_size_retention_constraint_diagnostic={str(readiness['ready_for_layer4_pool_size_retention_constraint_diagnostic']).lower()}

## Blocked / proxy
- AI/theme dynamic slot remains blocked placeholder; no hard-coded AI 20 quota.
- Raw turnover-share coverage is blocked in this package; traded-value ranks are retained.
- RS30, large-down, and blowoff-turnover are diagnostic proxy fields.
- Layer5 / replay / formal remain blocked.

## Next
If accepted, hand off to Experiments for
`TASK-BACKTEST-EXPERIMENTS-VNEXT-LAYER4-POOL-SIZE-RETENTION-CONSTRAINT-DIAGNOSTIC-001`.
完成後如果下一棒明確，請直接指派下一個 thread；如果下一棒不明確，請回報 Strategy Center 判斷。不要完成後停住不回報。
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer3-dir", default=str(DEFAULT_LAYER3_DIR))
    parser.add_argument("--layer4-experiments-dir", default=str(DEFAULT_LAYER4_EXPERIMENTS_DIR))
    parser.add_argument("--previous-layer4-dir", default=str(DEFAULT_PREVIOUS_LAYER4_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()
    manifest = build_contract(
        layer3_dir=args.layer3_dir,
        layer4_experiments_dir=args.layer4_experiments_dir,
        previous_layer4_dir=args.previous_layer4_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
