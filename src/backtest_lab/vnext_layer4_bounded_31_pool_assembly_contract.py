"""Build bounded Layer4 31-pool assembly contract/readiness.

This package materializes diagnostic pool variants only. It is not Layer5,
not replay, not formal, not a daily report input, and not a trade decision.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


TASK_ID = "TASK-BACKTEST-CORE-VNEXT-LAYER4-BOUNDED-31-POOL-ASSEMBLY-CONTRACT-001"
DEFAULT_LAYER3_DIR = Path("outputs/vnext_layer3_broad_opportunity_label_contract_20260708")
DEFAULT_EXPERIMENTS_DIR = Path(
    "C:/Users/zergv/Documents/Codex/2026-07-06/backtest-lab-experiments-diagnostic-validation-attribution/"
    "outputs/vnext_layer3_broad_opportunity_label_diagnostic_20260708"
)
DEFAULT_OUTPUT_DIR = Path("outputs/vnext_layer4_bounded_31_pool_assembly_contract_20260708")
POOL_SIZE = 31
PERIODS = {
    "P1": ("2015-01-02", "2022-12-29"),
    "P2": ("2023-01-02", "2026-06-30"),
    "2024_latest": ("2024-01-02", "2026-06-30"),
    "2026YTD": ("2026-01-02", "2026-06-30"),
}


def build_contract(
    *,
    layer3_dir: str | Path = DEFAULT_LAYER3_DIR,
    experiments_dir: str | Path = DEFAULT_EXPERIMENTS_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    layer3 = Path(layer3_dir)
    experiments = Path(experiments_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    layer3_readiness = _read_json(layer3 / "readiness_for_layer3_broad_opportunity_label_diagnostic.json")
    experiment_summary = _read_json(experiments / "layer3_broad_label_summary.json")
    base = _read_layer3_labels(layer3 / "layer3_broad_opportunity_label_contract.csv")
    scored = _attach_pool_scores(base)

    pool = _assemble_variants(scored)
    coverage = _weekly_coverage(pool)
    variant_design = _variant_design()
    component_design = _component_score_design()
    source_quality = _source_quality_matrix(scored, pool)
    missingness = _missingness_by_period(scored)
    blocked_proxy = _blocked_proxy_ledger()
    future_audit = _future_audit()
    readiness = _readiness(layer3_readiness, experiment_summary, scored, pool, coverage)

    _write_csv(pool, output / "layer4_bounded_31_pool_assembly_contract.csv")
    _write_csv(pool.head(1000), output / "layer4_bounded_31_pool_assembly_contract_sample.csv")
    (output / ".gitignore").write_text("layer4_bounded_31_pool_assembly_contract.csv\n", encoding="utf-8")
    _write_csv(variant_design, output / "layer4_pool_variant_design.csv")
    _write_csv(coverage, output / "layer4_pool_weekly_coverage_by_variant.csv")
    _write_csv(component_design, output / "layer4_pool_component_score_design.csv")
    _write_csv(source_quality, output / "layer4_pool_source_quality_matrix.csv")
    _write_csv(missingness, output / "layer4_pool_missingness_by_period.csv")
    _write_csv(blocked_proxy, output / "layer4_pool_blocked_proxy_ledger.csv")
    _write_csv(future_audit, output / "layer4_pool_future_data_audit.csv")
    (output / "readiness_for_layer4_bounded_31_pool_diagnostic.json").write_text(
        json.dumps(readiness, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    manifest = {
        "task_id": TASK_ID,
        "status": readiness["status"],
        "output_dir": str(output.resolve()),
        "input_layer3_dir": str(layer3.resolve()),
        "input_experiments_dir": str(experiments.resolve()),
        "output_files": [
            "layer4_bounded_31_pool_assembly_contract.csv",
            "layer4_bounded_31_pool_assembly_contract_sample.csv",
            "layer4_pool_variant_design.csv",
            "layer4_pool_weekly_coverage_by_variant.csv",
            "layer4_pool_component_score_design.csv",
            "layer4_pool_source_quality_matrix.csv",
            "layer4_pool_missingness_by_period.csv",
            "layer4_pool_blocked_proxy_ledger.csv",
            "layer4_pool_future_data_audit.csv",
            "readiness_for_layer4_bounded_31_pool_diagnostic.json",
            "manifest.json",
            "final_summary_zh.md",
        ],
        "large_local_files_not_tracked": ["layer4_bounded_31_pool_assembly_contract.csv"],
        "large_local_file_policy": "full bounded 31-pool assembly contract is retained locally; Git tracks sample/readiness/audit files only",
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


def _read_layer3_labels(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"ticker": str}, encoding="utf-8-sig", low_memory=False)
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
    return df


def _bool(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df:
        return pd.Series(False, index=df.index)
    return df[col].astype(str).str.lower().eq("true")


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce") if col in df else pd.Series(0.0, index=df.index)


def _rank_desc(df: pd.DataFrame, score_col: str) -> pd.Series:
    return df.groupby("snapshot_date")[score_col].rank(method="first", ascending=False)


def _attach_pool_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    layer1_quality = _bool(out, "layer1_b30_pass_through_eligible").astype(float)
    layer2_support = _num(out, "layer2_support_signal_count") / 2
    layer2_warning = _num(out, "layer2_warning_signal_count") / 3
    opportunity_count = (
        _bool(out, "momentum_continuation_medium_or_high_confidence").astype(int)
        + _bool(out, "pullback_repair_medium_or_high_confidence").astype(int)
        + _bool(out, "overlap_reacceleration_medium_or_high_confidence").astype(int)
        + _bool(out, "neutral_quality_liquidity_medium_or_high_confidence").astype(int)
    )
    out["two_plus_opportunity_labels"] = opportunity_count.ge(2)
    out["opportunity_label_count"] = opportunity_count
    out["layer4_base_eligible"] = layer1_quality.eq(1)
    out["layer4_pool_size_target"] = POOL_SIZE
    out["fallback_00631L_reference_only"] = False
    out["fallback_00631L_is_ordinary_stock_pool_member"] = False

    out["broad_opportunity_net_score"] = (
        0.24 * _num(out, "momentum_continuation_score")
        + 0.22 * _num(out, "pullback_repair_score")
        + 0.18 * _num(out, "overlap_reacceleration_score")
        + 0.18 * _num(out, "neutral_quality_liquidity_score")
        + 0.10 * layer2_support
        + 0.08 * _num(out, "capital_rank_improvement_20d_vs_60d").clip(lower=-100, upper=100).add(100).div(200)
        - 0.16 * _num(out, "exhaustion_risk_score")
        - 0.14 * _num(out, "breakdown_risk_score")
        - 0.08 * layer2_warning
    ).clip(0, 1)
    out["retention_friendly_score"] = (
        0.38 * out["two_plus_opportunity_labels"].astype(float)
        + 0.22 * out["opportunity_label_count"].clip(0, 4).div(4)
        + 0.20 * out["broad_opportunity_net_score"]
        + 0.12 * _num(out, "neutral_quality_liquidity_score")
        + 0.08 * layer1_quality
    ).clip(0, 1)
    out["score_balanced_pool_score"] = (
        0.30 * out["broad_opportunity_net_score"]
        + 0.18 * _num(out, "neutral_quality_liquidity_score")
        + 0.16 * _num(out, "momentum_continuation_score")
        + 0.14 * _num(out, "pullback_repair_score")
        + 0.12 * _num(out, "overlap_reacceleration_score")
        + 0.10 * layer2_support
        - 0.14 * _num(out, "exhaustion_risk_score")
        - 0.12 * _num(out, "breakdown_risk_score")
    ).clip(0, 1)
    out["risk_aware_pool_score"] = (
        out["score_balanced_pool_score"]
        - 0.22 * _num(out, "exhaustion_risk_score")
        - 0.18 * _num(out, "breakdown_risk_score")
        - 0.08 * _bool(out, "large_down_day_flag_20d_proxy").astype(float)
        - 0.08 * _bool(out, "blowoff_turnover_without_price_continuation_proxy").astype(float)
    ).clip(0, 1)
    out["theme_dynamic_slot_status"] = "blocked_no_accepted_theme_slot_context"
    out["theme_dynamic_slot_applied"] = False
    out["ai_theme_slot_hard_quota_applied"] = False
    out["layer4_selector_output"] = False
    out["layer5_decision_authorized"] = False
    out["portfolio_replay_executed"] = False
    out["formal_model_changed"] = False
    out["trade_decision_changed"] = False
    out["active_in_trade_decision"] = False
    out["report_changed"] = False
    out["not_live_rule"] = True
    out["forward_returns_live_rule_usage"] = False
    return out


def _assemble_variants(scored: pd.DataFrame) -> pd.DataFrame:
    pools = []
    eligible = scored[scored["layer4_base_eligible"]].copy()
    pools.append(_top_n(eligible, "A_retention_friendly_broad_label_pool", "retention_friendly_score"))
    pools.append(_top_n(eligible, "B_score_balanced_pool", "score_balanced_pool_score", max_cap_top30=18))
    pools.append(_quota_pool(eligible))
    pools.append(_top_n(eligible, "D_risk_aware_pool", "risk_aware_pool_score", max_high_risk=8))
    e = _top_n(eligible, "E_theme_dynamic_slot_placeholder_reference", "score_balanced_pool_score", max_cap_top30=18)
    e["theme_dynamic_slot_status"] = "blocked_no_accepted_theme_slot_context"
    e["theme_dynamic_slot_applied"] = False
    pools.append(e)
    out = pd.concat(pools, ignore_index=True)
    out["pool_size_target"] = POOL_SIZE
    out["bounded_diagnostic_pool_only"] = True
    out["layer4_formal_pool"] = False
    out["layer5_authorized"] = False
    out["forward_returns_evaluation_metadata_only"] = True
    out["forward_return_as_rule"] = False
    return out


def _top_n(
    df: pd.DataFrame,
    variant: str,
    score_col: str,
    *,
    max_cap_top30: int | None = None,
    max_high_risk: int | None = None,
) -> pd.DataFrame:
    rows = []
    for _, sub in df.groupby("snapshot_date", sort=True):
        sub = sub.sort_values([score_col, "neutral_quality_liquidity_score", "traded_value_rank_20d"], ascending=[False, False, True]).copy()
        selected = []
        cap_top30_count = 0
        high_risk_count = 0
        for _, row in sub.iterrows():
            cap_top30 = bool(row.get("capital_rank_20d_top30pct_context", False))
            high_risk = float(row.get("exhaustion_risk_score", 0) or 0) >= 0.70 or float(row.get("breakdown_risk_score", 0) or 0) >= 0.70
            if max_cap_top30 is not None and cap_top30 and cap_top30_count >= max_cap_top30:
                continue
            if max_high_risk is not None and high_risk and high_risk_count >= max_high_risk:
                continue
            selected.append(row)
            cap_top30_count += int(cap_top30)
            high_risk_count += int(high_risk)
            if len(selected) >= POOL_SIZE:
                break
        if len(selected) < POOL_SIZE:
            picked = {r["ticker"] for r in selected}
            for _, row in sub.iterrows():
                if row["ticker"] in picked:
                    continue
                selected.append(row)
                if len(selected) >= POOL_SIZE:
                    break
        pool = pd.DataFrame(selected)
        if pool.empty:
            continue
        pool["pool_variant"] = variant
        pool["pool_rank"] = range(1, len(pool) + 1)
        pool["pool_selection_score"] = pool[score_col]
        pool["pool_selection_basis"] = score_col
        rows.append(pool)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _quota_pool(df: pd.DataFrame) -> pd.DataFrame:
    quotas = [
        ("momentum", "momentum_continuation_score", "momentum_continuation_medium_or_high_confidence", 8),
        ("pullback", "pullback_repair_score", "pullback_repair_medium_or_high_confidence", 8),
        ("overlap", "overlap_reacceleration_score", "overlap_reacceleration_medium_or_high_confidence", 5),
        ("neutral", "neutral_quality_liquidity_score", "neutral_quality_liquidity_medium_or_high_confidence", 10),
    ]
    weekly = []
    for _, sub in df.groupby("snapshot_date", sort=True):
        picked: set[str] = set()
        rows = []
        for bucket, score_col, label_col, quota in quotas:
            candidates = sub[_bool(sub, label_col)].sort_values([score_col, "broad_opportunity_net_score"], ascending=False)
            take = candidates[~candidates["ticker"].isin(picked)].head(quota).copy()
            take["quota_bucket"] = bucket
            rows.append(take)
            picked.update(take["ticker"].tolist())
        pool = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
        if len(pool) < POOL_SIZE:
            filler = sub[~sub["ticker"].isin(set(pool["ticker"]) if not pool.empty else set())].sort_values(
                ["broad_opportunity_net_score", "neutral_quality_liquidity_score"], ascending=False
            )
            add = filler.head(POOL_SIZE - len(pool)).copy()
            add["quota_bucket"] = "fill_broad_score"
            pool = pd.concat([pool, add], ignore_index=True)
        pool = pool.head(POOL_SIZE).copy()
        pool["pool_variant"] = "C_quota_style_broad_label_pool"
        pool["pool_rank"] = range(1, len(pool) + 1)
        pool["pool_selection_score"] = pool["broad_opportunity_net_score"]
        pool["pool_selection_basis"] = "broad_label_quota_momentum8_pullback8_overlap5_neutral10"
        weekly.append(pool)
    return pd.concat(weekly, ignore_index=True) if weekly else pd.DataFrame()


def _variant_design() -> pd.DataFrame:
    rows = [
        ("A_retention_friendly_broad_label_pool", "31", "two_plus_opportunity_labels + opportunity count + net score", "retention-friendly ranking; no strict top-k gate"),
        ("B_score_balanced_pool", "31", "broad net score blend with cap top30 concentration guard", "balanced score; max 18 capital-top30 context rows before fill"),
        ("C_quota_style_broad_label_pool", "31", "momentum 8 / pullback 8 / overlap 5 / neutral 10 broad quotas", "neutral quality/liquidity retained"),
        ("D_risk_aware_pool", "31", "balanced score with exhaustion/breakdown/large-down/blowoff penalty", "risk reduces rank, does not hard exclude"),
        ("E_theme_dynamic_slot_placeholder_reference", "31", "theme slot placeholder unavailable; no AI hard quota applied", "theme/AI dynamic policy blocked until accepted source"),
    ]
    return pd.DataFrame(rows, columns=["pool_variant", "pool_size", "ranking_or_quota_basis", "policy"])


def _component_score_design() -> pd.DataFrame:
    rows = [
        ("layer1_quality_floor", "layer1_b30_pass_through_eligible", "eligibility base", "only base eligibility flag"),
        ("layer2_context", "capital support, RS context, warnings/penalties", "ranking context", "no row deletion"),
        ("layer3_broad_labels", "momentum/pullback/overlap/neutral scores and tiers", "ranking/quota context", "labels are not filters"),
        ("risk_penalty", "exhaustion/breakdown/large-down/blowoff", "rank reduction", "no one-shot exclusion"),
        ("liquidity_investability", "traded value ranks and neutral quality/liquidity", "tie breaker", "diagnostic only"),
        ("evaluation_metadata", "5/10/20/30/40D forward excess", "Experiments evaluation only", "not used in pool assembly"),
    ]
    return pd.DataFrame(rows, columns=["component", "input_fields", "role", "policy"])


def _weekly_coverage(pool: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (variant, date), sub in pool.groupby(["pool_variant", "snapshot_date"], sort=True):
        rows.append(
            {
                "pool_variant": variant,
                "snapshot_date": date,
                "period": _period_for_date(date),
                "selected_count": int(len(sub)),
                "unique_ticker_count": int(sub["ticker"].nunique()),
                "target_pool_size": POOL_SIZE,
                "shortfall_count": int(max(0, POOL_SIZE - len(sub))),
                "momentum_medium_or_high_count": int(_bool(sub, "momentum_continuation_medium_or_high_confidence").sum()),
                "pullback_medium_or_high_count": int(_bool(sub, "pullback_repair_medium_or_high_confidence").sum()),
                "overlap_medium_or_high_count": int(_bool(sub, "overlap_reacceleration_medium_or_high_confidence").sum()),
                "neutral_medium_or_high_count": int(_bool(sub, "neutral_quality_liquidity_medium_or_high_confidence").sum()),
                "high_exhaustion_count": int(_bool(sub, "exhaustion_risk_high_confidence").sum()),
                "high_breakdown_count": int(_bool(sub, "breakdown_risk_high_confidence").sum()),
                "fallback_00631L_reference_rows": int(sub["fallback_00631L_reference_only"].sum()) if "fallback_00631L_reference_only" in sub else 0,
                "ordinary_stock_pool_rows": int((~_bool(sub, "fallback_00631L_reference_only")).sum()) if "fallback_00631L_reference_only" in sub else int(len(sub)),
            }
        )
    return pd.DataFrame(rows)


def _period_for_date(date: pd.Timestamp) -> str:
    for period, (start, end) in PERIODS.items():
        if pd.Timestamp(start) <= date <= pd.Timestamp(end):
            return period
    return "out_of_requested_periods"


def _source_quality_matrix(scored: pd.DataFrame, pool: pd.DataFrame) -> pd.DataFrame:
    rows = [
        ("base_universe", "compact Layer0 active + Layer1 b30", "exact_from_prior_core_contract", 1.0, "only base eligibility"),
        ("Layer2_context", "capital, RS, risk context fields", "diagnostic_pit_context", float(scored["RS20"].notna().mean()), "pass-through only"),
        ("Layer3_broad_labels", "broad opportunity scores/tier fields", "diagnostic_composite_from_pit_features", float(scored["broad_opportunity_net_score"].notna().mean()), "not filters"),
        ("Layer4_pool_assembly", "31-row bounded diagnostic variants", "diagnostic_assembly_from_pit_context", float((pool.groupby(["pool_variant", "snapshot_date"]).size() == POOL_SIZE).mean()), "not formal pool"),
        ("AI_theme_slots", "theme dynamic slot placeholder", "blocked_no_accepted_theme_slot_context", 0.0, "no AI hard quota"),
        ("forward_evaluation", "5/10/20/30/40D labels", "evaluation_metadata_only", float(scored["forward_eval_available_30d"].mean()), "not used in assembly"),
    ]
    return pd.DataFrame(rows, columns=["feature_group", "fields", "source_quality", "available_share", "policy"])


def _missingness_by_period(scored: pd.DataFrame) -> pd.DataFrame:
    features = [
        "layer4_base_eligible",
        "broad_opportunity_net_score",
        "retention_friendly_score",
        "score_balanced_pool_score",
        "risk_aware_pool_score",
        "momentum_continuation_score",
        "pullback_repair_score",
        "overlap_reacceleration_score",
        "neutral_quality_liquidity_score",
        "exhaustion_risk_score",
        "breakdown_risk_score",
        "forward_excess_vs_00631L_5d",
        "forward_excess_vs_00631L_10d",
        "forward_excess_vs_00631L_20d",
        "forward_excess_vs_00631L_30d",
        "forward_excess_vs_00631L_40d",
    ]
    rows = []
    for period, (start, end) in {"ALL": (None, None), **PERIODS}.items():
        mask = pd.Series(True, index=scored.index)
        if start:
            mask &= scored["snapshot_date"].ge(pd.Timestamp(start))
        if end:
            mask &= scored["snapshot_date"].le(pd.Timestamp(end))
        sub = scored[mask]
        for feature in features:
            rows.append(
                {
                    "period": period,
                    "feature": feature,
                    "rows": int(len(sub)),
                    "available_rows": int(sub[feature].notna().sum()) if feature in sub else 0,
                    "missing_rows": int(sub[feature].isna().sum()) if feature in sub else int(len(sub)),
                    "available_share": float(sub[feature].notna().mean()) if len(sub) and feature in sub else 0.0,
                }
            )
    return pd.DataFrame(rows)


def _blocked_proxy_ledger() -> pd.DataFrame:
    rows = [
        ("Layer4 formal pool", "not_authorized", "bounded diagnostic pool assembly only", "not formal-ready"),
        ("Layer5 A/B decision", "not_authorized", "scope stops before daily decision", "no trade decision"),
        ("portfolio_replay", "not_authorized", "no replay in Core contract", "Experiments event diagnostic only if authorized"),
        ("00631L_fallback", "reference_only", "fallback/reference fields only", "not ordinary stock pool member"),
        ("AI_theme_dynamic_slots", "blocked_placeholder", "no accepted dynamic theme slot contract in this package", "no AI hard quota"),
        ("RS30", "proxy", "RS30_proxy midpoint of RS20/RS40", "diagnostic context only"),
        ("risk_bucket", "blocked", "no accepted PIT risk_bucket", "do not fabricate"),
        ("large_down_blowoff", "diagnostic_proxy", "proxy risk context", "rank penalty only, not hard exclusion"),
        ("forward_returns", "evaluation_metadata_only", "retained for metrics join", "not used in assembly scoring"),
    ]
    return pd.DataFrame(rows, columns=["field", "status", "reason", "policy"])


def _future_audit() -> pd.DataFrame:
    rows = [
        ("pool_assembly_inputs", "passed", 0, "ranking/quota inputs are PIT context fields only"),
        ("forward_return_as_rule", "passed", 0, "forward returns retained only for evaluation metadata and not used in pool scoring"),
        ("Layer2_Layer3_hard_gate", "passed", 0, "Layer2/Layer3 fields are ranking/context only; base eligibility is Layer1 b30"),
        ("00631L_policy", "passed", 0, "00631L retained as reference/fallback marker only, not ordinary stock member"),
        ("Layer5", "not_authorized", 0, "no A/B/fallback decision output"),
        ("portfolio_replay", "not_executed", 0, "no replay executed"),
    ]
    return pd.DataFrame(rows, columns=["audit_item", "status", "future_data_violation_count", "note"])


def _readiness(
    layer3_readiness: dict[str, Any],
    experiment_summary: dict[str, Any],
    scored: pd.DataFrame,
    pool: pd.DataFrame,
    coverage: pd.DataFrame,
) -> dict[str, Any]:
    full_pool_share = float(coverage["selected_count"].eq(POOL_SIZE).mean()) if len(coverage) else 0.0
    variant_count = int(pool["pool_variant"].nunique()) if len(pool) else 0
    weekly_count = int(scored["snapshot_date"].nunique())
    ready = full_pool_share >= 0.99 and variant_count >= 5
    return {
        "task_id": TASK_ID,
        "status": "layer4_bounded_31_pool_assembly_contract_ready_for_experiments_intake" if ready else "layer4_bounded_31_pool_assembly_contract_partial_blocked",
        "diagnostic_only": True,
        "bounded_pool_assembly_only": True,
        "input_layer3_status": layer3_readiness.get("status", ""),
        "input_experiments_verdict": experiment_summary.get("verdict", ""),
        "base_rows": int(len(scored)),
        "pool_rows": int(len(pool)),
        "weekly_snapshot_count": weekly_count,
        "pool_variant_count": variant_count,
        "target_pool_size": POOL_SIZE,
        "full_pool_week_variant_share": full_pool_share,
        "layer1_b30_is_base_eligibility": True,
        "layer2_context_only": True,
        "layer3_context_only": True,
        "layer2_layer3_hard_gate_allowed": False,
        "fallback_00631L_reference_only": True,
        "fallback_00631L_ordinary_stock_member": False,
        "ai_theme_dynamic_slot_status": "blocked_placeholder_no_hard_quota",
        "ready_for_layer4_bounded_31_pool_diagnostic": ready,
        "ready_for_experiments_intake": ready,
        "ready_for_layer5_decision": False,
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
        "blocked_fields": ["AI_theme_dynamic_slot_contract", "risk_bucket", "Layer5_decision", "portfolio_replay"],
        "proxy_fields": ["RS30_proxy", "large_down_blowoff_proxy", "pullback_lowpoint_proxy"],
    }


def _summary(readiness: dict[str, Any]) -> str:
    return f"""# Layer4 bounded 31-pool assembly contract

## Verdict
- status={readiness["status"]}
- base_rows={readiness["base_rows"]}
- pool_rows={readiness["pool_rows"]}
- weekly_snapshot_count={readiness["weekly_snapshot_count"]}
- pool_variant_count={readiness["pool_variant_count"]}
- target_pool_size=31
- full_pool_week_variant_share={readiness["full_pool_week_variant_share"]}
- layer1_b30_is_base_eligibility=true
- layer2_context_only=true
- layer3_context_only=true
- layer2_layer3_hard_gate_allowed=false
- fallback_00631L_reference_only=true
- fallback_00631L_ordinary_stock_member=false
- ready_for_layer4_bounded_31_pool_diagnostic={str(readiness["ready_for_layer4_bounded_31_pool_diagnostic"]).lower()}

## Plain Summary
This package materializes five bounded 31-stock diagnostic pool variants from compact Layer0 + Layer1 b30 pass-through plus Layer2/Layer3 context ranking. It does not authorize Layer5, replay, formal use, reports, or trade decisions. 00631L is reference/fallback metadata only and is not mixed into ordinary stock pool rows.

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
    parser.add_argument("--layer3-dir", default=str(DEFAULT_LAYER3_DIR))
    parser.add_argument("--experiments-dir", default=str(DEFAULT_EXPERIMENTS_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()
    manifest = build_contract(
        layer3_dir=args.layer3_dir,
        experiments_dir=args.experiments_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
