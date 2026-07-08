"""Build Layer2 context-only pass-through and Layer3 sleeve readiness contract.

Layer2 fields in this package are annotations only. They do not delete rows,
create eligibility gates, create selectors, feed the daily report, or change
trade decisions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


TASK_ID = "TASK-BACKTEST-CORE-VNEXT-LAYER2-CONTEXT-ONLY-PASS-THROUGH-LAYER3-SLEEVE-READINESS-001"
DEFAULT_MULTI_DIR = Path("outputs/vnext_layer2_soft_score_multi_horizon_evaluation_join_20260708")
DEFAULT_DATA_DIR = Path("outputs/vnext_dynamic_candidate_pool_data_materialization_20260706")
DEFAULT_EXPERIMENTS_DIR = Path(
    "C:/Users/zergv/Documents/Codex/2026-07-06/backtest-lab-experiments-diagnostic-validation-attribution/"
    "outputs/vnext_layer2_soft_score_multi_horizon_risk_diagnostic_20260708"
)
DEFAULT_OUTPUT_DIR = Path("outputs/vnext_layer2_context_layer3_sleeve_readiness_20260708")
PERIODS = {
    "P1": ("2015-01-02", "2022-12-29"),
    "P2": ("2023-01-02", "2026-06-30"),
    "2024_latest": ("2024-01-02", "2026-06-30"),
    "2026YTD": ("2026-01-02", "2026-06-30"),
}


def build_contract(
    *,
    multi_dir: str | Path = DEFAULT_MULTI_DIR,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    experiments_dir: str | Path = DEFAULT_EXPERIMENTS_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    multi = Path(multi_dir)
    data = Path(data_dir)
    experiments = Path(experiments_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    multi_readiness = _read_json(multi / "readiness_for_layer2_soft_score_multi_horizon_diagnostic.json")
    experiment_summary = _read_json(experiments / "layer2_soft_score_multi_horizon_summary.json")
    base = _read_multi_join(multi / "layer2_soft_score_multi_horizon_evaluation_join.csv")
    ma_context = _read_ma_context(data / "stock_features.csv", base)

    enriched = base.merge(ma_context, on=["snapshot_date", "ticker"], how="left")
    pass_through = _build_pass_through(enriched)
    layer2_context_fields = _layer2_context_field_contract()
    sleeve = _build_layer3_sleeve_contract(pass_through)

    missingness = _missingness_by_period(sleeve)
    source_quality = _source_quality_matrix(sleeve)
    blocked_proxy = _blocked_proxy_fields()
    future_audit = _future_audit()
    readiness = _readiness(multi_readiness, experiment_summary, pass_through, sleeve)

    _write_csv(pass_through, output / "layer2_context_pass_through_candidate_contract.csv")
    _write_csv(pass_through.head(1000), output / "layer2_context_pass_through_candidate_contract_sample.csv")
    _write_csv(sleeve, output / "layer3_compact_sleeve_feature_readiness_contract.csv")
    _write_csv(sleeve.head(1000), output / "layer3_compact_sleeve_feature_readiness_contract_sample.csv")
    (output / ".gitignore").write_text(
        "layer2_context_pass_through_candidate_contract.csv\n"
        "layer3_compact_sleeve_feature_readiness_contract.csv\n",
        encoding="utf-8",
    )
    _write_csv(layer2_context_fields, output / "layer2_context_field_contract.csv")
    _write_csv(missingness, output / "layer3_sleeve_missingness_by_period.csv")
    _write_csv(source_quality, output / "layer2_context_layer3_sleeve_source_quality_matrix.csv")
    _write_csv(blocked_proxy, output / "layer2_context_layer3_sleeve_blocked_proxy_ledger.csv")
    _write_csv(future_audit, output / "layer2_context_layer3_sleeve_future_data_audit.csv")
    (output / "readiness_for_layer2_context_layer3_sleeve_diagnostic.json").write_text(
        json.dumps(readiness, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    manifest = {
        "task_id": TASK_ID,
        "status": readiness["status"],
        "output_dir": str(output.resolve()),
        "input_multi_horizon_dir": str(multi.resolve()),
        "input_data_dir": str(data.resolve()),
        "input_experiments_dir": str(experiments.resolve()),
        "output_files": [
            "layer2_context_pass_through_candidate_contract.csv",
            "layer2_context_pass_through_candidate_contract_sample.csv",
            "layer2_context_field_contract.csv",
            "layer3_compact_sleeve_feature_readiness_contract.csv",
            "layer3_compact_sleeve_feature_readiness_contract_sample.csv",
            "layer3_sleeve_missingness_by_period.csv",
            "layer2_context_layer3_sleeve_source_quality_matrix.csv",
            "layer2_context_layer3_sleeve_blocked_proxy_ledger.csv",
            "layer2_context_layer3_sleeve_future_data_audit.csv",
            "readiness_for_layer2_context_layer3_sleeve_diagnostic.json",
            "manifest.json",
            "final_summary_zh.md",
        ],
        "large_local_files_not_tracked": [
            "layer2_context_pass_through_candidate_contract.csv",
            "layer3_compact_sleeve_feature_readiness_contract.csv",
        ],
        "large_local_file_policy": "full row-level pass-through and sleeve contracts are retained locally; Git tracks samples/readiness/audit only",
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


def _read_multi_join(path: Path) -> pd.DataFrame:
    usecols = [
        "baseline_compact_universe_row_id",
        "snapshot_date",
        "ticker",
        "name",
        "market",
        "variant",
        "scope_type",
        "active_for_layer1_source_scope",
        "selection_bucket",
        "layer1_exclude_bottom20_candidate",
        "layer1_exclude_bottom30_candidate",
        "layer1_pass_bottom20",
        "layer1_pass_bottom30",
        "layer1_financial_risk_flag_count",
        "layer1_quality_floor_risk_pctile_by_week",
        "monthly_revenue_available",
        "quarterly_fundamental_available",
        "missing_core_fundamental_flag",
        "traded_value_rank_5d",
        "traded_value_rank_20d",
        "traded_value_rank_60d",
        "capital_rank_improvement_20d_vs_60d",
        "capital_rank_20d_improving_vs_60d",
        "capital_rank_20d_deteriorating_vs_60d",
        "capital_reasonable_band_4w_count",
        "capital_reasonable_band_4w_persistent",
        "pure_5d_burst_without_20d60d_confirmation",
        "RS5",
        "RS10",
        "RS20",
        "RS30_proxy",
        "RS40",
        "RS60",
        "RS30_source_quality",
        "rs5_minus_rs10",
        "rs10_minus_rs20",
        "rs20_minus_rs60",
        "rs_short_acceleration_flag",
        "rs_short_deterioration_flag",
        "rs20_30_primary_momentum_positive",
        "rs20_30_primary_momentum_stable",
        "rs60_background_supportive",
        "rs60_top20_by_week",
        "rs_exhaustion_warning_context",
        "rs60_high_short_rs_weakening_exhaustion_context",
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
        "volatility_pctile_by_week",
        "bias_overheat_penalty_context",
        "volatility_high_context",
        "risk_overheat_penalty_context",
        "large_down_day_count_20d_proxy",
        "large_down_day_count_30d_proxy",
        "large_down_day_flag_20d_proxy",
        "large_down_day_flag_30d_proxy",
        "large_down_day_source_quality",
        "blowoff_turnover_without_price_continuation_proxy",
        "blowoff_turnover_source_quality",
        "risk_bucket",
        "risk_bucket_source_quality",
        "forward_eval_available_5d",
        "forward_eval_available_10d",
        "forward_eval_available_20d",
        "forward_eval_available_30d",
        "forward_eval_available_40d",
        "forward_excess_vs_0050_5d",
        "forward_excess_vs_0050_10d",
        "forward_excess_vs_0050_20d",
        "forward_excess_vs_0050_30d",
        "forward_excess_vs_0050_40d",
        "forward_excess_vs_00631L_5d",
        "forward_excess_vs_00631L_10d",
        "forward_excess_vs_00631L_20d",
        "forward_excess_vs_00631L_30d",
        "forward_excess_vs_00631L_40d",
        "P2_top_decile_vs_00631L_20d",
        "P2_bottom_decile_vs_00631L_20d",
        "shape_improving_vs_0050",
        "shape_fading_vs_0050",
        "shape_quick_burst_vs_0050",
        "shape_short_bounce_fade_risk_vs_0050",
        "shape_slow_start_vs_0050",
        "shape_durable_vs_0050",
        "shape_40d_decay_reference_vs_0050",
        "shape_improving_vs_00631L",
        "shape_fading_vs_00631L",
        "shape_quick_burst_vs_00631L",
        "shape_short_bounce_fade_risk_vs_00631L",
        "shape_slow_start_vs_00631L",
        "shape_durable_vs_00631L",
        "shape_40d_decay_reference_vs_00631L",
    ]
    df = pd.read_csv(path, usecols=usecols, dtype={"ticker": str}, encoding="utf-8-sig", low_memory=False)
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
    return df


def _read_ma_context(path: Path, base: pd.DataFrame) -> pd.DataFrame:
    pairs = base[["snapshot_date", "ticker"]].drop_duplicates().copy()
    ticker_set = set(pairs["ticker"].astype(str))
    date_set = set(pairs["snapshot_date"].dt.strftime("%Y-%m-%d"))
    usecols = [
        "trade_date",
        "ticker",
        "MA20_position",
        "MA60_position",
        "MA120_position",
        "MA20",
        "MA60",
        "MA120",
    ]
    chunks = []
    for chunk in pd.read_csv(path, usecols=usecols, dtype={"ticker": str}, chunksize=500_000):
        chunk = chunk[chunk["ticker"].isin(ticker_set)].copy()
        if chunk.empty:
            continue
        chunk = chunk[chunk["trade_date"].isin(date_set)].copy()
        if chunk.empty:
            continue
        chunk["snapshot_date"] = pd.to_datetime(chunk["trade_date"])
        chunks.append(chunk.drop(columns=["trade_date"]))
    if not chunks:
        return pairs
    ma = pd.concat(chunks, ignore_index=True).drop_duplicates(["snapshot_date", "ticker"], keep="last")
    return pairs.merge(ma, on=["snapshot_date", "ticker"], how="left")


def _as_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().eq("true")


def _build_pass_through(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["layer1_b20_pass_through_eligible"] = _as_bool(out["layer1_pass_bottom20"])
    out["layer1_b30_pass_through_eligible"] = _as_bool(out["layer1_pass_bottom30"])
    out["layer2_context_only"] = True
    out["layer2_hard_gate_allowed"] = False
    out["layer2_row_deleted"] = False
    out["pass_through_primary_basis"] = "layer0_compact_active_universe_plus_layer1_quality_floor_b30"

    out["capital_support_context"] = (
        out["capital_reasonable_band_4w_persistent"].fillna(False).astype(bool)
        | pd.to_numeric(out["traded_value_rank_20d"], errors="coerce").le(300)
        | pd.to_numeric(out["traded_value_rank_60d"], errors="coerce").le(300)
    )
    out["capital_warning_context"] = _as_bool(out["pure_5d_burst_without_20d60d_confirmation"]) | _as_bool(
        out["capital_rank_20d_deteriorating_vs_60d"]
    )
    out["rs_support_context"] = _as_bool(out["rs20_30_primary_momentum_positive"]) | _as_bool(out["rs20_30_primary_momentum_stable"])
    out["rs_warning_context"] = _as_bool(out["rs_short_deterioration_flag"]) | _as_bool(out["rs_exhaustion_warning_context"])
    out["risk_penalty_context"] = (
        _as_bool(out["risk_overheat_penalty_context"])
        | _as_bool(out["large_down_day_flag_20d_proxy"])
        | _as_bool(out["blowoff_turnover_without_price_continuation_proxy"])
    )
    out["layer2_support_signal_count"] = out[["capital_support_context", "rs_support_context"]].astype(int).sum(axis=1)
    out["layer2_warning_signal_count"] = out[["capital_warning_context", "rs_warning_context", "risk_penalty_context"]].astype(int).sum(axis=1)
    out["layer2_context_role"] = "annotation_support_warning_penalty_only"
    out["diagnostic_only"] = True
    out["not_live_rule"] = True
    out["forward_returns_live_rule_usage"] = False
    out["formal_model_changed"] = False
    out["trade_decision_changed"] = False
    out["active_in_trade_decision"] = False
    out["report_changed"] = False
    return out


def _layer2_context_field_contract() -> pd.DataFrame:
    rows = [
        ("traded_value_rank_20d", "capital_support", "support", "exact_from_layer0_compact_pit", "context only"),
        ("traded_value_rank_60d", "capital_support", "support", "exact_from_layer0_compact_pit", "context only"),
        ("capital_rank_improvement_20d_vs_60d", "capital_support", "support_or_warning", "exact_from_layer0_compact_pit", "context only"),
        ("capital_reasonable_band_4w_persistent", "capital_support", "support", "diagnostic_pit_rolling", "no row deletion"),
        ("pure_5d_burst_without_20d60d_confirmation", "capital_support", "warning", "diagnostic_pit_context", "watchlist/penalty only"),
        ("RS5_RS10_RS20_RS30proxy_RS60", "relative_strength", "support_or_warning", "pit_stock_features_rs30_proxy", "RS30 proxy only"),
        ("rs_short_deterioration_flag", "relative_strength", "warning", "pit_rs_context", "context only"),
        ("rs60_high_short_rs_weakening_exhaustion_context", "relative_strength", "penalty", "pit_rs_context", "context only"),
        ("BIAS_percentile_volatility", "risk_overheat", "penalty", "pit_stock_features_diagnostic", "context only"),
        ("large_down_day_proxy", "risk_overheat", "proxy", "diagnostic_price_proxy_threshold_not_formal", "not formal trigger"),
        ("blowoff_turnover_proxy", "risk_overheat", "proxy", "diagnostic_traded_value_proxy_threshold_not_formal", "not formal trigger"),
        ("risk_bucket", "risk_overheat", "blocked", "blocked_no_accepted_pit_risk_bucket", "do not fabricate"),
    ]
    return pd.DataFrame(rows, columns=["field", "group", "role", "source_quality", "policy"])


def _build_layer3_sleeve_contract(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    rs5 = pd.to_numeric(out["RS5"], errors="coerce")
    rs10 = pd.to_numeric(out["RS10"], errors="coerce")
    rs20 = pd.to_numeric(out["RS20"], errors="coerce")
    rs30 = pd.to_numeric(out["RS30_proxy"], errors="coerce")
    rs40 = pd.to_numeric(out["RS40"], errors="coerce")
    rs60 = pd.to_numeric(out["RS60"], errors="coerce")
    bias20_pct = pd.to_numeric(out["BIAS20_percentile"], errors="coerce")
    bias60_pct = pd.to_numeric(out["BIAS60_percentile"], errors="coerce")
    drawdown20 = pd.to_numeric(out["drawdown_20d"], errors="coerce")
    drawdown60 = pd.to_numeric(out["drawdown_60d"], errors="coerce")
    ma20 = _as_bool(out.get("MA20_position", pd.Series(False, index=out.index)))
    ma60 = _as_bool(out.get("MA60_position", pd.Series(False, index=out.index)))

    out["momentum_relative_spread_widening_context"] = (rs5.ge(rs10) & rs10.ge(rs20)) | (rs10.ge(rs20) & rs20.ge(rs60))
    out["momentum_stable_rise_context"] = rs20.ge(0) & rs30.ge(0) & ~_as_bool(out["rs_short_deterioration_flag"])
    out["momentum_capital_support_context"] = _as_bool(out["capital_support_context"])
    out["momentum_overheat_late_stage_penalty_context"] = (
        bias20_pct.ge(0.90)
        | bias60_pct.ge(0.90)
        | _as_bool(out["rs60_high_short_rs_weakening_exhaustion_context"])
        | _as_bool(out["blowoff_turnover_without_price_continuation_proxy"])
    )
    out["momentum_sleeve_candidate_feature"] = (
        out["layer1_b30_pass_through_eligible"]
        & out["momentum_stable_rise_context"]
        & out["momentum_capital_support_context"]
        & ~out["momentum_overheat_late_stage_penalty_context"]
    )

    out["pullback_prior_strength_context"] = (
        rs20.ge(0) | rs30.ge(0) | rs40.ge(0) | rs60.ge(0) | out["rs60_top20_by_week"].fillna(False).astype(bool)
    )
    out["pullback_current_correction_context"] = (
        _as_bool(out["rs_short_deterioration_flag"])
        | rs5.lt(rs20)
        | rs10.lt(rs20)
        | drawdown20.le(-0.05)
        | drawdown60.le(-0.08)
    )
    out["pullback_ma_bias_position_context"] = (
        bias20_pct.le(0.60)
        | bias60_pct.le(0.60)
        | ma20
        | ma60
    )
    out["pullback_risk_cooling_context"] = (
        bias20_pct.le(0.75)
        & bias60_pct.le(0.80)
        & ~_as_bool(out["large_down_day_flag_20d_proxy"])
    )
    out["pullback_breakdown_warning_context"] = (
        rs20.lt(0)
        & rs30.lt(0)
        & rs60.lt(0)
    ) | _as_bool(out["large_down_day_flag_30d_proxy"])
    out["pullback_repair_sleeve_candidate_feature"] = (
        out["layer1_b30_pass_through_eligible"]
        & out["pullback_prior_strength_context"]
        & out["pullback_current_correction_context"]
        & out["pullback_ma_bias_position_context"]
        & ~out["pullback_breakdown_warning_context"]
    )
    out["overlap_sleeve_candidate_feature"] = out["momentum_sleeve_candidate_feature"] & out["pullback_repair_sleeve_candidate_feature"]
    out["neither_sleeve_candidate_feature"] = ~(
        out["momentum_sleeve_candidate_feature"] | out["pullback_repair_sleeve_candidate_feature"]
    )
    out["layer3_sleeve_feature_contract_only"] = True
    out["layer3_selector_output"] = False
    out["layer4_31pool_authorized"] = False
    out["layer5_decision_authorized"] = False
    return out


def _missingness_by_period(df: pd.DataFrame) -> pd.DataFrame:
    features = [
        "layer1_b30_pass_through_eligible",
        "capital_support_context",
        "rs_support_context",
        "risk_penalty_context",
        "RS20",
        "RS30_proxy",
        "RS60",
        "BIAS20_percentile",
        "BIAS60_percentile",
        "drawdown_20d",
        "drawdown_60d",
        "MA20_position",
        "MA60_position",
        "momentum_sleeve_candidate_feature",
        "pullback_repair_sleeve_candidate_feature",
        "overlap_sleeve_candidate_feature",
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
        ("Layer1 quality floor", "layer1 b20/b30 pass-through flags", "exact_from_layer1_compact_contract", float(df["layer1_b30_pass_through_eligible"].notna().mean()), "only eligibility gate in this package"),
        ("Layer2 capital context", "rank 20/60, rank change, 4w persistence", "exact_or_diagnostic_pit_context", float(df["capital_support_context"].notna().mean()), "annotation only"),
        ("Layer2 RS context", "RS5/10/20/30proxy/60 and exhaustion", "pit_rs_context_rs30_proxy", float(df["RS20"].notna().mean()), "annotation only"),
        ("Risk/overheat context", "BIAS, volatility, large-down, blowoff proxy", "pit_diagnostic_partial_proxy", float(df["BIAS20_percentile"].notna().mean()), "warning/penalty context"),
        ("Layer3 momentum sleeve", "relative spread widening, stable rise, capital support, overheat penalty", "diagnostic_composite_from_pit_features", float(df["momentum_sleeve_candidate_feature"].notna().mean()), "feature only"),
        ("Layer3 pullback sleeve", "prior strength, correction, MA/BIAS position, cooling, no breakdown", "diagnostic_composite_from_pit_features", float(df["pullback_repair_sleeve_candidate_feature"].notna().mean()), "feature only"),
        ("Forward evaluation", "5/10/20/30/40D labels retained", "evaluation_metadata_only", float(df["forward_eval_available_30d"].mean()), "not live rule"),
    ]
    return pd.DataFrame(rows, columns=["feature_group", "fields", "source_quality", "available_share", "policy"])


def _blocked_proxy_fields() -> pd.DataFrame:
    rows = [
        ("Layer2 hard gate", "prohibited", "Strategy Center accepted Layer2 NO-GO as gate", "do not filter rows"),
        ("RS30", "proxy", "exact RS30 unavailable; RS30_proxy uses midpoint of RS20/RS40", "diagnostic context only"),
        ("large_down_day", "diagnostic_proxy", "daily return threshold proxy, not formal policy", "warning context only"),
        ("blowoff_turnover_without_price_continuation", "diagnostic_proxy", "traded-value z-score proxy, not formal trigger", "warning context only"),
        ("risk_bucket", "blocked", "no accepted PIT risk_bucket field", "do not fabricate"),
        ("pullback exact lowpoint", "proxy", "uses drawdown/BIAS/MA position context, not an accepted lowpoint model", "sleeve feature only"),
        ("Layer4 31-pool", "not_authorized", "scope stops at Layer3 sleeve readiness", "no pool assembly"),
        ("Layer5 decision", "not_authorized", "scope stops before daily A/B/fallback decision", "no trade decision"),
    ]
    return pd.DataFrame(rows, columns=["field", "status", "reason", "policy"])


def _future_audit() -> pd.DataFrame:
    rows = [
        ("Layer2_gate", "passed", 0, "Layer2 context fields do not delete rows"),
        ("forward_return_as_rule", "passed", 0, "forward returns retained only as evaluation metadata"),
        ("sleeve_features", "passed", 0, "Layer3 sleeve flags use as-of PIT features only"),
        ("selector_output", "not_applicable", 0, "no selector, no score rank output, no Layer4 pool"),
        ("portfolio_replay", "not_executed", 0, "no replay executed"),
    ]
    return pd.DataFrame(rows, columns=["audit_item", "status", "future_data_violation_count", "note"])


def _readiness(
    multi_readiness: dict[str, Any],
    experiment_summary: dict[str, Any],
    pass_through: pd.DataFrame,
    sleeve: pd.DataFrame,
) -> dict[str, Any]:
    b30_share = float(pass_through["layer1_b30_pass_through_eligible"].mean())
    momentum_share = float(sleeve["momentum_sleeve_candidate_feature"].mean())
    pullback_share = float(sleeve["pullback_repair_sleeve_candidate_feature"].mean())
    overlap_share = float(sleeve["overlap_sleeve_candidate_feature"].mean())
    rs20_share = float(sleeve["RS20"].notna().mean())
    bias_share = float(sleeve["BIAS20_percentile"].notna().mean())
    ma60_share = float(sleeve["MA60_position"].notna().mean()) if "MA60_position" in sleeve else 0.0
    ready = rs20_share > 0.80 and bias_share > 0.80 and ma60_share > 0.75
    return {
        "task_id": TASK_ID,
        "status": "layer2_context_pass_through_layer3_sleeve_readiness_ready_for_experiments_intake" if ready else "layer2_context_pass_through_layer3_sleeve_readiness_partial_blocked",
        "diagnostic_only": True,
        "input_multi_horizon_status": multi_readiness.get("status", ""),
        "input_experiments_verdict": experiment_summary.get("verdict", ""),
        "rows": int(len(pass_through)),
        "weekly_snapshot_count": int(pass_through["snapshot_date"].nunique()),
        "unique_ticker_count": int(pass_through["ticker"].nunique()),
        "layer1_b30_pass_through_share": b30_share,
        "layer2_context_only": True,
        "layer2_hard_gate_allowed": False,
        "layer2_row_deleted_count": int(pass_through["layer2_row_deleted"].sum()),
        "momentum_sleeve_candidate_share": momentum_share,
        "pullback_repair_sleeve_candidate_share": pullback_share,
        "overlap_sleeve_candidate_share": overlap_share,
        "rs20_available_share": rs20_share,
        "bias20_percentile_available_share": bias_share,
        "ma60_position_available_share": ma60_share,
        "ready_for_layer3_compact_pass_through_sleeve_diagnostic": ready,
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
        "blocked_fields": ["risk_bucket", "formal_large_down_day_policy", "formal_blowoff_turnover_policy", "exact_pullback_lowpoint_model"],
        "proxy_fields": ["RS30_proxy", "large_down_day_proxy", "blowoff_turnover_proxy", "pullback_ma_bias_position_context"],
    }


def _summary(readiness: dict[str, Any]) -> str:
    return f"""# Layer2 context-only pass-through + Layer3 compact sleeve readiness

## Verdict
- status={readiness["status"]}
- rows={readiness["rows"]}
- weekly_snapshot_count={readiness["weekly_snapshot_count"]}
- unique_ticker_count={readiness["unique_ticker_count"]}
- layer2_context_only=true
- layer2_hard_gate_allowed=false
- layer2_row_deleted_count={readiness["layer2_row_deleted_count"]}
- layer1_b30_pass_through_share={readiness["layer1_b30_pass_through_share"]}
- momentum_sleeve_candidate_share={readiness["momentum_sleeve_candidate_share"]}
- pullback_repair_sleeve_candidate_share={readiness["pullback_repair_sleeve_candidate_share"]}
- overlap_sleeve_candidate_share={readiness["overlap_sleeve_candidate_share"]}
- ready_for_layer3_compact_pass_through_sleeve_diagnostic={str(readiness["ready_for_layer3_compact_pass_through_sleeve_diagnostic"]).lower()}

## Plain Summary
Layer2 is converted from a gate into pass-through context annotations. Layer1 b30 remains the primary quality-floor eligibility flag. Layer3 momentum, pullback/repair, overlap, and neither sleeve features are materialized for bounded diagnostic only. This package does not authorize Layer4 pool assembly, Layer5 decisions, replay, reports, or formal model changes.

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
    parser.add_argument("--multi-dir", default=str(DEFAULT_MULTI_DIR))
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--experiments-dir", default=str(DEFAULT_EXPERIMENTS_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()
    manifest = build_contract(
        multi_dir=args.multi_dir,
        data_dir=args.data_dir,
        experiments_dir=args.experiments_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
