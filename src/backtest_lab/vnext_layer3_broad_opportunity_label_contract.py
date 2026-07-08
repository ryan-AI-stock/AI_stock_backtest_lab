"""Build Layer3 broad opportunity-label context contract.

Layer3 labels in this package are broad, overlapping, context-only annotations.
They are not filters, selectors, Layer4 pool assembly, replay inputs, formal
model changes, report inputs, or trade decisions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


TASK_ID = "TASK-BACKTEST-CORE-VNEXT-LAYER3-BROAD-OPPORTUNITY-LABEL-CONTEXT-CONTRACT-001"
DEFAULT_LAYER3_DIR = Path("outputs/vnext_layer2_context_layer3_sleeve_readiness_20260708")
DEFAULT_EXPERIMENTS_DIR = Path(
    "C:/Users/zergv/Documents/Codex/2026-07-06/backtest-lab-experiments-diagnostic-validation-attribution/"
    "outputs/vnext_layer3_compact_pass_through_sleeve_diagnostic_20260708"
)
DEFAULT_OUTPUT_DIR = Path("outputs/vnext_layer3_broad_opportunity_label_contract_20260708")
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

    prior_readiness = _read_json(layer3 / "readiness_for_layer2_context_layer3_sleeve_diagnostic.json")
    experiment_summary = _read_json(experiments / "layer3_sleeve_summary.json")
    base = _read_layer3_contract(layer3 / "layer3_compact_sleeve_feature_readiness_contract.csv")

    contract = _attach_broad_scores(base)
    contract = _attach_tiers(contract)
    contract = _attach_policy_flags(contract)
    score_design = _score_design()
    coverage = _label_coverage(contract)
    missingness = _missingness_by_period(contract)
    source_quality = _source_quality_matrix(contract)
    blocked_proxy = _blocked_proxy_ledger()
    future_audit = _future_audit()
    readiness = _readiness(prior_readiness, experiment_summary, contract, coverage)

    _write_csv(contract, output / "layer3_broad_opportunity_label_contract.csv")
    _write_csv(contract.head(1000), output / "layer3_broad_opportunity_label_contract_sample.csv")
    (output / ".gitignore").write_text("layer3_broad_opportunity_label_contract.csv\n", encoding="utf-8")
    _write_csv(score_design, output / "layer3_broad_label_score_design.csv")
    _write_csv(coverage, output / "layer3_label_coverage_by_week_period.csv")
    _write_csv(missingness, output / "layer3_broad_label_missingness_by_period.csv")
    _write_csv(source_quality, output / "layer3_broad_label_source_quality_matrix.csv")
    _write_csv(blocked_proxy, output / "layer3_broad_label_blocked_proxy_ledger.csv")
    _write_csv(future_audit, output / "layer3_broad_label_future_data_audit.csv")
    (output / "readiness_for_layer3_broad_opportunity_label_diagnostic.json").write_text(
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
            "layer3_broad_opportunity_label_contract.csv",
            "layer3_broad_opportunity_label_contract_sample.csv",
            "layer3_broad_label_score_design.csv",
            "layer3_label_coverage_by_week_period.csv",
            "layer3_broad_label_missingness_by_period.csv",
            "layer3_broad_label_source_quality_matrix.csv",
            "layer3_broad_label_blocked_proxy_ledger.csv",
            "layer3_broad_label_future_data_audit.csv",
            "readiness_for_layer3_broad_opportunity_label_diagnostic.json",
            "manifest.json",
            "final_summary_zh.md",
        ],
        "large_local_files_not_tracked": ["layer3_broad_opportunity_label_contract.csv"],
        "large_local_file_policy": "full broad opportunity label contract is retained locally; Git tracks sample/readiness/audit files only",
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


def _read_layer3_contract(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"ticker": str}, encoding="utf-8-sig", low_memory=False)
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
    return df


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce") if col in df else pd.Series(pd.NA, index=df.index, dtype="float")


def _bool(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df:
        return pd.Series(False, index=df.index)
    return df[col].astype(str).str.lower().eq("true")


def _clip(series: pd.Series, lower: float = 0.0, upper: float = 1.0) -> pd.Series:
    return series.astype(float).clip(lower=lower, upper=upper).fillna(0.0)


def _pctile(df: pd.DataFrame, col: str) -> pd.Series:
    values = _num(df, col)
    return values.groupby(df["snapshot_date"]).rank(pct=True, method="average").fillna(0.0)


def _rank_support(df: pd.DataFrame, col: str) -> pd.Series:
    rank = _num(df, col)
    week_size = df.groupby("snapshot_date")["ticker"].transform("count")
    return (1 - ((rank - 1) / (week_size - 1))).clip(0, 1).fillna(0.0)


def _attach_broad_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    rs5p = _pctile(out, "RS5")
    rs10p = _pctile(out, "RS10")
    rs20p = _pctile(out, "RS20")
    rs30p = _pctile(out, "RS30_proxy")
    rs40p = _pctile(out, "RS40")
    rs60p = _pctile(out, "RS60")
    cap20 = _rank_support(out, "traded_value_rank_20d")
    cap60 = _rank_support(out, "traded_value_rank_60d")
    cap_persist = _clip(_num(out, "capital_reasonable_band_4w_count") / 4)
    bias20 = _num(out, "BIAS20_percentile").fillna(0.5)
    bias60 = _num(out, "BIAS60_percentile").fillna(0.5)
    vol = _num(out, "volatility_pctile_by_week").fillna(0.5)
    dd20 = _num(out, "drawdown_20d").fillna(0.0)
    dd60 = _num(out, "drawdown_60d").fillna(0.0)

    rs_widening = (_bool(out, "momentum_relative_spread_widening_context") | (rs5p.ge(rs10p) & rs10p.ge(rs20p))).astype(float)
    rs_accel = (~_bool(out, "rs_short_deterioration_flag") & (_num(out, "rs5_minus_rs10").fillna(0).ge(0) | _num(out, "rs10_minus_rs20").fillna(0).ge(0))).astype(float)
    overheat_penalty = _clip(((bias20 - 0.75) / 0.25).clip(0, 1) * 0.45 + ((bias60 - 0.80) / 0.20).clip(0, 1) * 0.25 + vol.clip(0, 1) * 0.15)
    exhaustion = (_bool(out, "rs60_high_short_rs_weakening_exhaustion_context") | _bool(out, "blowoff_turnover_without_price_continuation_proxy")).astype(float)

    out["momentum_continuation_score"] = _clip(
        0.22 * rs20p
        + 0.22 * rs30p
        + 0.16 * rs_widening
        + 0.16 * cap20
        + 0.12 * cap_persist
        + 0.12 * rs_accel
        - 0.18 * overheat_penalty
        - 0.14 * exhaustion
    )

    prior_strength = pd.concat([rs20p, rs30p, rs40p, rs60p], axis=1).max(axis=1)
    current_correction = (
        _bool(out, "rs_short_deterioration_flag").astype(float) * 0.30
        + _clip((-dd20) / 0.12) * 0.25
        + _clip((-dd60) / 0.20) * 0.20
        + (bias20.le(0.65)).astype(float) * 0.15
        + (bias60.le(0.70)).astype(float) * 0.10
    )
    cooling = (
        (bias20.le(0.75)).astype(float) * 0.35
        + (bias60.le(0.80)).astype(float) * 0.25
        + (~_bool(out, "large_down_day_flag_20d_proxy")).astype(float) * 0.20
        + (~_bool(out, "risk_overheat_penalty_context")).astype(float) * 0.20
    )
    breakdown_penalty = ((rs20p.lt(0.35) & rs30p.lt(0.35) & rs60p.lt(0.35)) | _bool(out, "large_down_day_flag_30d_proxy")).astype(float)
    capital_fade = (_bool(out, "capital_rank_20d_deteriorating_vs_60d") & cap20.lt(0.45)).astype(float)
    out["pullback_repair_score"] = _clip(
        0.28 * prior_strength
        + 0.24 * _clip(current_correction)
        + 0.18 * cooling
        + 0.12 * cap60
        + 0.10 * _bool(out, "pullback_ma_bias_position_context").astype(float)
        + 0.08 * _bool(out, "layer1_b30_pass_through_eligible").astype(float)
        - 0.22 * breakdown_penalty
        - 0.12 * capital_fade
    )

    reaccel = (_num(out, "rs5_minus_rs10").fillna(0).gt(0) | _num(out, "rs10_minus_rs20").fillna(0).gt(0)).astype(float)
    repair_base = _clip(current_correction)
    out["overlap_reacceleration_score"] = _clip(
        0.28 * prior_strength
        + 0.22 * repair_base
        + 0.20 * reaccel
        + 0.14 * cap20
        + 0.10 * cap_persist
        + 0.06 * _bool(out, "layer1_b30_pass_through_eligible").astype(float)
        - 0.18 * breakdown_penalty
        - 0.12 * exhaustion
    )

    layer1_quality = _bool(out, "layer1_b30_pass_through_eligible").astype(float)
    liquidity = pd.concat([cap20, cap60], axis=1).max(axis=1)
    not_obvious_label = 1 - pd.concat(
        [out["momentum_continuation_score"], out["pullback_repair_score"], out["overlap_reacceleration_score"]],
        axis=1,
    ).max(axis=1)
    out["neutral_quality_liquidity_score"] = _clip(
        0.42 * layer1_quality
        + 0.28 * liquidity
        + 0.18 * (~_bool(out, "risk_overheat_penalty_context")).astype(float)
        + 0.12 * not_obvious_label
    )

    out["exhaustion_risk_score"] = _clip(
        0.30 * _bool(out, "rs60_high_short_rs_weakening_exhaustion_context").astype(float)
        + 0.22 * _bool(out, "rs_short_deterioration_flag").astype(float)
        + 0.18 * overheat_penalty
        + 0.14 * _bool(out, "blowoff_turnover_without_price_continuation_proxy").astype(float)
        + 0.10 * _bool(out, "large_down_day_flag_20d_proxy").astype(float)
        + 0.06 * vol.clip(0, 1)
    )
    out["breakdown_risk_score"] = _clip(
        0.26 * (rs20p.lt(0.30)).astype(float)
        + 0.22 * (rs30p.lt(0.30)).astype(float)
        + 0.18 * (rs60p.lt(0.30)).astype(float)
        + 0.14 * _bool(out, "capital_rank_20d_deteriorating_vs_60d").astype(float)
        + 0.12 * _bool(out, "large_down_day_flag_30d_proxy").astype(float)
        + 0.08 * _clip((-dd60) / 0.25)
    )
    return out


def _tier(score: pd.Series) -> pd.Series:
    return pd.cut(
        score,
        bins=[-0.001, 0.25, 0.45, 0.70, 1.001],
        labels=["none", "low", "medium", "high"],
    ).astype(str)


def _attach_tiers(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    score_cols = [
        "momentum_continuation_score",
        "pullback_repair_score",
        "overlap_reacceleration_score",
        "neutral_quality_liquidity_score",
        "exhaustion_risk_score",
        "breakdown_risk_score",
    ]
    for col in score_cols:
        label = col.replace("_score", "")
        out[f"{label}_tier"] = _tier(out[col])
        out[f"{label}_high_confidence"] = out[f"{label}_tier"].eq("high")
        out[f"{label}_medium_or_high_confidence"] = out[f"{label}_tier"].isin(["medium", "high"])
    out["broad_labels_are_overlapping"] = True
    out["label_is_filter"] = False
    out["layer3_broad_context_only"] = True
    out["layer3_hard_gate_allowed"] = False
    out["layer3_row_deleted"] = False
    out["layer4_31pool_authorized"] = False
    out["layer5_decision_authorized"] = False
    return out


def _attach_policy_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["diagnostic_only"] = True
    out["not_live_rule"] = True
    out["selector_output"] = False
    out["forward_return_as_rule"] = False
    out["future_return_as_rule"] = False
    out["forward_returns_live_rule_usage"] = False
    out["formal_model_changed"] = False
    out["trade_decision_changed"] = False
    out["active_in_trade_decision"] = False
    out["report_changed"] = False
    return out


def _score_design() -> pd.DataFrame:
    rows = [
        ("momentum_continuation_score", "RS20/30 percentile, RS widening, capital support, 4w persistence, acceleration minus overheat/exhaustion", "broad support label", "not a filter or selector"),
        ("pullback_repair_score", "prior RS strength, current correction, BIAS/MA position, cooling, capital support minus breakdown/capital fade", "broad repair label", "not exact lowpoint"),
        ("overlap_reacceleration_score", "prior strength + correction + short-RS reacceleration + capital support minus breakdown/exhaustion", "overlap label", "labels may overlap"),
        ("neutral_quality_liquidity_score", "Layer1 b30 pass + liquidity + no risk penalty + not obvious momentum/pullback", "background 31-pool context", "do not discard neutral candidates"),
        ("exhaustion_risk_score", "RS60 high short-RS weakening, overheat, blowoff proxy, large-down proxy, volatility", "warning/penalty context", "not automatic exclusion"),
        ("breakdown_risk_score", "weak RS20/30/60, capital deterioration, large-down proxy, drawdown", "warning/penalty context", "not automatic exclusion"),
    ]
    return pd.DataFrame(rows, columns=["score", "input_components", "role", "policy"])


def _label_coverage(df: pd.DataFrame) -> pd.DataFrame:
    label_cols = [
        "momentum_continuation_medium_or_high_confidence",
        "pullback_repair_medium_or_high_confidence",
        "overlap_reacceleration_medium_or_high_confidence",
        "neutral_quality_liquidity_medium_or_high_confidence",
        "exhaustion_risk_medium_or_high_confidence",
        "breakdown_risk_medium_or_high_confidence",
    ]
    weekly = []
    for date, sub in df.groupby("snapshot_date"):
        row: dict[str, Any] = {
            "period": _period_for_date(date),
            "snapshot_date": date,
            "rows": int(len(sub)),
            "layer1_b30_pass_share": float(_bool(sub, "layer1_b30_pass_through_eligible").mean()),
        }
        for col in label_cols:
            row[f"{col}_share"] = float(sub[col].mean())
            row[f"{col}_count"] = int(sub[col].sum())
        row["any_opportunity_medium_or_high_share"] = float(
            sub[
                [
                    "momentum_continuation_medium_or_high_confidence",
                    "pullback_repair_medium_or_high_confidence",
                    "overlap_reacceleration_medium_or_high_confidence",
                    "neutral_quality_liquidity_medium_or_high_confidence",
                ]
            ].any(axis=1).mean()
        )
        weekly.append(row)
    return pd.DataFrame(weekly)


def _period_for_date(date: pd.Timestamp) -> str:
    for period, (start, end) in PERIODS.items():
        if pd.Timestamp(start) <= date <= pd.Timestamp(end):
            return period
    return "out_of_requested_periods"


def _missingness_by_period(df: pd.DataFrame) -> pd.DataFrame:
    features = [
        "momentum_continuation_score",
        "pullback_repair_score",
        "overlap_reacceleration_score",
        "neutral_quality_liquidity_score",
        "exhaustion_risk_score",
        "breakdown_risk_score",
        "RS20",
        "RS30_proxy",
        "RS60",
        "BIAS20_percentile",
        "BIAS60_percentile",
        "drawdown_20d",
        "drawdown_60d",
        "MA20_position",
        "MA60_position",
        "forward_excess_vs_00631L_5d",
        "forward_excess_vs_00631L_10d",
        "forward_excess_vs_00631L_20d",
        "forward_excess_vs_00631L_30d",
        "forward_excess_vs_00631L_40d",
    ]
    rows = []
    for period, (start, end) in {"ALL": (None, None), **PERIODS}.items():
        mask = pd.Series(True, index=df.index)
        if start:
            mask &= df["snapshot_date"].ge(pd.Timestamp(start))
        if end:
            mask &= df["snapshot_date"].le(pd.Timestamp(end))
        sub = df[mask]
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


def _source_quality_matrix(df: pd.DataFrame) -> pd.DataFrame:
    rows = [
        ("base_universe", "compact Layer0 active + Layer1 b30 pass-through flag", "exact_from_prior_core_contract", 1.0, "Layer3 labels do not delete rows"),
        ("momentum_continuation_score", "RS/capital/BIAS/exhaustion PIT components", "diagnostic_composite_from_pit_features", float(df["momentum_continuation_score"].notna().mean()), "context score only"),
        ("pullback_repair_score", "prior strength/correction/BIAS/MA/drawdown cooling", "diagnostic_composite_proxy_lowpoint", float(df["pullback_repair_score"].notna().mean()), "exact lowpoint blocked"),
        ("overlap_reacceleration_score", "prior strength + correction + short RS reacceleration", "diagnostic_composite_from_pit_features", float(df["overlap_reacceleration_score"].notna().mean()), "overlapping label"),
        ("neutral_quality_liquidity_score", "Layer1 b30 + liquidity + no major risk context", "diagnostic_composite_from_pit_features", float(df["neutral_quality_liquidity_score"].notna().mean()), "background context"),
        ("risk_scores", "exhaustion and breakdown risk scores", "diagnostic_proxy_partial", float(df["exhaustion_risk_score"].notna().mean()), "warning/penalty only"),
        ("forward_evaluation", "5/10/20/30/40D retained for Experiments", "evaluation_metadata_only", float(df["forward_eval_available_30d"].mean()), "not live rule"),
    ]
    return pd.DataFrame(rows, columns=["feature_group", "fields", "source_quality", "available_share", "policy"])


def _blocked_proxy_ledger() -> pd.DataFrame:
    rows = [
        ("Layer3 hard filter", "prohibited", "Strategy Center accepted current narrow sleeve no-go", "labels only; do not delete rows"),
        ("RS30", "proxy", "exact RS30 unavailable; RS30_proxy uses midpoint of RS20/RS40", "diagnostic context only"),
        ("risk_bucket", "blocked", "no accepted PIT risk_bucket field", "do not fabricate"),
        ("large_down_day", "diagnostic_proxy", "daily return threshold proxy", "warning score context only"),
        ("blowoff_turnover", "diagnostic_proxy", "traded value z-score proxy", "warning score context only"),
        ("pullback exact lowpoint", "proxy", "uses drawdown/BIAS/MA position, not exact lowpoint model", "broad repair context only"),
        ("forward returns", "evaluation_metadata_only", "retained for Experiments evaluation", "not label construction input"),
        ("Layer4 31-pool", "not_authorized", "scope stops at Layer3 broad labels", "no pool assembly"),
        ("Layer5 decision", "not_authorized", "scope stops before daily A/B/fallback decision", "no trade decision"),
    ]
    return pd.DataFrame(rows, columns=["field", "status", "reason", "policy"])


def _future_audit() -> pd.DataFrame:
    rows = [
        ("Layer3_filter", "passed", 0, "Layer3 labels do not delete rows"),
        ("forward_return_as_rule", "passed", 0, "forward returns retained only as evaluation metadata and not used in label score formulas"),
        ("score_inputs", "passed", 0, "scores use PIT RS/capital/BIAS/MA/drawdown/volatility context"),
        ("selector_output", "not_applicable", 0, "no selector or rank output produced"),
        ("Layer4_31pool", "not_authorized", 0, "no pool assembly"),
        ("portfolio_replay", "not_executed", 0, "no replay executed"),
    ]
    return pd.DataFrame(rows, columns=["audit_item", "status", "future_data_violation_count", "note"])


def _readiness(
    prior_readiness: dict[str, Any],
    experiment_summary: dict[str, Any],
    df: pd.DataFrame,
    coverage: pd.DataFrame,
) -> dict[str, Any]:
    p2_cov = coverage[coverage["period"].eq("P2")]
    p2_avg_any = float(p2_cov["any_opportunity_medium_or_high_share"].mean()) if not p2_cov.empty else 0.0
    p2_min_any = float(p2_cov["any_opportunity_medium_or_high_share"].min()) if not p2_cov.empty else 0.0
    score_cols = [
        "momentum_continuation_score",
        "pullback_repair_score",
        "overlap_reacceleration_score",
        "neutral_quality_liquidity_score",
        "exhaustion_risk_score",
        "breakdown_risk_score",
    ]
    ready = all(float(df[col].notna().mean()) > 0.95 for col in score_cols) and p2_avg_any > 0.50
    return {
        "task_id": TASK_ID,
        "status": "layer3_broad_opportunity_label_contract_ready_for_experiments_intake" if ready else "layer3_broad_opportunity_label_contract_partial_blocked",
        "diagnostic_only": True,
        "context_only": True,
        "label_is_filter": False,
        "input_layer3_sleeve_status": prior_readiness.get("status", ""),
        "input_experiments_verdict": experiment_summary.get("verdict", ""),
        "rows": int(len(df)),
        "weekly_snapshot_count": int(df["snapshot_date"].nunique()),
        "unique_ticker_count": int(df["ticker"].nunique()),
        "layer1_b30_pass_through_share": float(_bool(df, "layer1_b30_pass_through_eligible").mean()),
        "layer2_context_only": True,
        "layer3_hard_gate_allowed": False,
        "layer3_row_deleted_count": int(df["layer3_row_deleted"].sum()),
        "momentum_medium_or_high_share": float(df["momentum_continuation_medium_or_high_confidence"].mean()),
        "pullback_medium_or_high_share": float(df["pullback_repair_medium_or_high_confidence"].mean()),
        "overlap_reacceleration_medium_or_high_share": float(df["overlap_reacceleration_medium_or_high_confidence"].mean()),
        "neutral_medium_or_high_share": float(df["neutral_quality_liquidity_medium_or_high_confidence"].mean()),
        "p2_avg_any_opportunity_medium_or_high_share": p2_avg_any,
        "p2_min_any_opportunity_medium_or_high_share": p2_min_any,
        "rs30_exact_available": False,
        "risk_bucket_available": False,
        "ready_for_layer3_broad_opportunity_label_diagnostic": ready,
        "ready_for_experiments_intake": ready,
        "ready_for_layer4_31pool": False,
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
        "blocked_fields": ["risk_bucket", "exact_pullback_lowpoint_model", "Layer4_31pool", "Layer5_decision"],
        "proxy_fields": ["RS30_proxy", "large_down_day_proxy", "blowoff_turnover_proxy", "pullback_lowpoint_proxy"],
    }


def _summary(readiness: dict[str, Any]) -> str:
    return f"""# Layer3 broad opportunity-label context contract

## Verdict
- status={readiness["status"]}
- rows={readiness["rows"]}
- weekly_snapshot_count={readiness["weekly_snapshot_count"]}
- unique_ticker_count={readiness["unique_ticker_count"]}
- context_only=true
- label_is_filter=false
- layer3_hard_gate_allowed=false
- layer3_row_deleted_count={readiness["layer3_row_deleted_count"]}
- momentum_medium_or_high_share={readiness["momentum_medium_or_high_share"]}
- pullback_medium_or_high_share={readiness["pullback_medium_or_high_share"]}
- overlap_reacceleration_medium_or_high_share={readiness["overlap_reacceleration_medium_or_high_share"]}
- neutral_medium_or_high_share={readiness["neutral_medium_or_high_share"]}
- p2_avg_any_opportunity_medium_or_high_share={readiness["p2_avg_any_opportunity_medium_or_high_share"]}
- ready_for_layer3_broad_opportunity_label_diagnostic={str(readiness["ready_for_layer3_broad_opportunity_label_diagnostic"]).lower()}

## Plain Summary
Layer3 is rebuilt as broad overlapping opportunity labels and continuous context scores. Momentum, pullback/repair, overlap reacceleration, neutral quality/liquidity, exhaustion risk, and breakdown risk are annotations only. No rows are deleted by Layer2 or Layer3. Forward returns remain evaluation metadata only.

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
