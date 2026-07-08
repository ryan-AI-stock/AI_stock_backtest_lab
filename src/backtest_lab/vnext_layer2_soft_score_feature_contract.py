"""Build Layer2 compact soft-score feature contract/readiness.

This is a diagnostic feature contract only. It does not create a selector,
portfolio replay, daily report input, trade decision, or formal model change.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


TASK_ID = "TASK-BACKTEST-CORE-VNEXT-LAYER2-SOFT-SCORE-FEATURE-CONTRACT-READINESS-001"
DEFAULT_LAYER1_DIR = Path("outputs/vnext_layer1_compact_reduced_universe_interim_contract_20260707")
DEFAULT_RS_JOIN_DIR = Path("outputs/vnext_layer2_compact_rs_window_evaluation_join_20260707")
DEFAULT_DATA_DIR = Path("outputs/vnext_dynamic_candidate_pool_data_materialization_20260706")
DEFAULT_EXPERIMENTS_DIR = Path(
    "C:/Users/zergv/Documents/Codex/2026-07-06/backtest-lab-experiments-diagnostic-validation-attribution/"
    "outputs/vnext_layer2_compact_missed_winner_quality_attribution_rerun_20260708"
)
DEFAULT_OUTPUT_DIR = Path("outputs/vnext_layer2_soft_score_feature_contract_20260708")
PERIODS = {
    "P1": ("2015-01-02", "2022-12-29"),
    "P2": ("2023-01-02", "2026-06-30"),
    "2024_latest": ("2024-01-02", "2026-06-30"),
    "2026YTD": ("2026-01-02", "2026-06-30"),
}


def build_contract(
    *,
    layer1_dir: str | Path = DEFAULT_LAYER1_DIR,
    rs_join_dir: str | Path = DEFAULT_RS_JOIN_DIR,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    experiments_dir: str | Path = DEFAULT_EXPERIMENTS_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    layer1 = Path(layer1_dir)
    rs_join = Path(rs_join_dir)
    data = Path(data_dir)
    experiments = Path(experiments_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    rs_readiness = _read_json(rs_join / "readiness_for_layer2_compact_rs_window_evaluation_join.json")
    missed_summary = _read_json(experiments / "missed_winner_quality_summary.json")
    base = _read_rs_join(rs_join / "layer2_compact_rs_window_evaluation_join.csv")
    layer1_extra = _read_layer1_extra(layer1 / "layer1_compact_reduced_universe_interim_contract.csv")
    risk_context = _read_risk_context(data / "stock_features.csv", base)

    contract = base.merge(layer1_extra, on=["snapshot_date", "ticker"], how="left")
    contract = contract.merge(risk_context, on=["snapshot_date", "ticker"], how="left")
    contract = _attach_capital_features(contract)
    contract = _attach_rs_soft_features(contract)
    contract = _attach_stable_strong_features(contract)
    contract = _attach_risk_features(contract)
    contract = _attach_policy_flags(contract)

    missingness = _missingness_by_period(contract)
    blocked_proxy = _blocked_proxy_fields()
    source_quality = _source_quality_matrix(contract)
    component_design = _suggested_component_design()
    future_audit = _future_data_audit()
    readiness = _readiness(rs_readiness, missed_summary, contract, missingness)

    _write_csv(contract, output / "layer2_soft_score_feature_contract.csv")
    _write_csv(contract.head(1000), output / "layer2_soft_score_feature_contract_sample.csv")
    (output / ".gitignore").write_text("layer2_soft_score_feature_contract.csv\n", encoding="utf-8")
    _write_csv(missingness, output / "layer2_soft_score_missingness_by_period.csv")
    _write_csv(blocked_proxy, output / "layer2_soft_score_blocked_proxy_fields.csv")
    _write_csv(source_quality, output / "layer2_soft_score_source_quality_matrix.csv")
    _write_csv(component_design, output / "layer2_soft_score_suggested_component_design.csv")
    _write_csv(future_audit, output / "layer2_soft_score_future_data_audit.csv")
    (output / "readiness_for_layer2_soft_score_diagnostic.json").write_text(
        json.dumps(readiness, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    manifest = {
        "task_id": TASK_ID,
        "status": readiness["status"],
        "output_dir": str(output.resolve()),
        "input_layer1_dir": str(layer1.resolve()),
        "input_rs_join_dir": str(rs_join.resolve()),
        "input_data_dir": str(data.resolve()),
        "input_experiments_dir": str(experiments.resolve()),
        "output_files": [
            "layer2_soft_score_feature_contract.csv",
            "layer2_soft_score_feature_contract_sample.csv",
            "layer2_soft_score_missingness_by_period.csv",
            "layer2_soft_score_blocked_proxy_fields.csv",
            "layer2_soft_score_source_quality_matrix.csv",
            "layer2_soft_score_suggested_component_design.csv",
            "layer2_soft_score_future_data_audit.csv",
            "readiness_for_layer2_soft_score_diagnostic.json",
            "manifest.json",
            "final_summary_zh.md",
        ],
        "large_local_files_not_tracked": ["layer2_soft_score_feature_contract.csv"],
        "large_local_file_policy": "full soft-score feature contract is retained in local output path; Git tracks sample/readiness/audit files only",
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


def _read_rs_join(path: Path) -> pd.DataFrame:
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
        "traded_value_rank_5d",
        "traded_value_rank_20d",
        "traded_value_rank_60d",
        "layer1_financial_risk_flag_count",
        "layer1_quality_floor_risk_pctile_by_week",
        "layer1_exclude_bottom20_candidate",
        "layer1_exclude_bottom30_candidate",
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
        "rs20_primary_positive_flag",
        "rs30_proxy_positive_flag",
        "rs60_medium_context_positive_flag",
        "rs60_pctile_by_week",
        "rs60_top20_by_week",
        "rs60_top10_by_week",
        "rs60_high_short_rs_weakening_exhaustion_context",
        "forward_excess_vs_0050_20d",
        "forward_excess_vs_00631L_20d",
        "forward_eval_available_20d",
        "P2_top_decile_vs_00631L_20d",
        "P2_bottom_decile_vs_00631L_20d",
        "diagnostic_only",
        "not_live_rule",
        "forward_returns_live_rule_usage",
    ]
    df = pd.read_csv(path, usecols=usecols, dtype={"ticker": str}, encoding="utf-8-sig", low_memory=False)
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
    for col in [
        "traded_value_rank_5d",
        "traded_value_rank_20d",
        "traded_value_rank_60d",
        "RS5",
        "RS10",
        "RS20",
        "RS30_proxy",
        "RS40",
        "RS60",
        "rs5_minus_rs10",
        "rs10_minus_rs20",
        "rs20_minus_rs60",
        "rs60_pctile_by_week",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _read_layer1_extra(path: Path) -> pd.DataFrame:
    usecols = [
        "snapshot_date",
        "ticker",
        "traded_value_5d",
        "traded_value_20d",
        "traded_value_60d",
        "rank_improvement_5d_vs_60d",
        "top300_5d_count_last4w",
        "buffer_candidate_rank_251_300",
        "buffer_included_by_2in4",
        "buffer_included_by_20d60d",
        "pure_5d_burst_watchlist_only",
        "monthly_revenue_available",
        "quarterly_fundamental_available",
        "negative_revenue_yoy_flag",
        "negative_revenue_3m_yoy_flag",
        "negative_eps_flag",
        "negative_operating_income_flag",
        "low_gross_margin_flag",
        "missing_core_fundamental_flag",
    ]
    df = pd.read_csv(path, usecols=usecols, dtype={"ticker": str}, encoding="utf-8-sig", low_memory=False)
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
    return df


def _read_risk_context(path: Path, base: pd.DataFrame) -> pd.DataFrame:
    pairs = base[["snapshot_date", "ticker"]].drop_duplicates().copy()
    ticker_set = set(pairs["ticker"].astype(str))
    date_set = set(pairs["snapshot_date"].dt.strftime("%Y-%m-%d"))
    usecols = [
        "trade_date",
        "ticker",
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
    chunks = []
    for chunk in pd.read_csv(path, usecols=usecols, dtype={"ticker": str}, chunksize=500_000):
        chunk = chunk[chunk["ticker"].isin(ticker_set)].copy()
        if chunk.empty:
            continue
        chunk = chunk[chunk["trade_date"].isin(date_set)].copy()
        if chunk.empty:
            continue
        chunk["snapshot_date"] = pd.to_datetime(chunk["trade_date"])
        for col in usecols:
            if col not in {"trade_date", "ticker"}:
                chunk[col] = pd.to_numeric(chunk[col], errors="coerce")
        chunks.append(chunk.drop(columns=["trade_date"]))
    if not chunks:
        return pairs
    return pairs.merge(pd.concat(chunks, ignore_index=True), on=["snapshot_date", "ticker"], how="left")


def _attach_capital_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    week_size = out.groupby("snapshot_date")["ticker"].transform("count")
    for window in [20, 60]:
        rank_col = f"traded_value_rank_{window}d"
        out[f"capital_rank_{window}d_pctile_by_week"] = 1 - ((out[rank_col] - 1) / (week_size - 1))
        out[f"capital_rank_{window}d_reasonable_band_300"] = out[rank_col].le(300)
        out[f"capital_rank_{window}d_top30pct_context"] = out[f"capital_rank_{window}d_pctile_by_week"].ge(0.70)
    out["capital_rank_improvement_20d_vs_60d"] = out["traded_value_rank_60d"] - out["traded_value_rank_20d"]
    out["capital_rank_20d_improving_vs_60d"] = out["capital_rank_improvement_20d_vs_60d"].gt(0)
    out["capital_rank_20d_deteriorating_vs_60d"] = out["capital_rank_improvement_20d_vs_60d"].lt(0)
    out = out.sort_values(["ticker", "snapshot_date"]).copy()
    out["capital_reasonable_band_4w_count"] = (
        out.groupby("ticker")["capital_rank_20d_reasonable_band_300"]
        .transform(lambda s: s.astype(float).rolling(4, min_periods=1).sum())
    )
    out["capital_reasonable_band_4w_persistent"] = out["capital_reasonable_band_4w_count"].ge(2)
    out["pure_5d_burst_without_20d60d_confirmation"] = out["pure_5d_burst_watchlist_only"].astype(str).str.lower().eq("true")
    return out


def _attach_rs_soft_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["rs20_30_primary_momentum_positive"] = out["RS20"].gt(0) | out["RS30_proxy"].gt(0)
    out["rs20_30_primary_momentum_stable"] = out["rs20_30_primary_momentum_positive"] & ~out["rs_short_deterioration_flag"].astype(bool)
    out["rs60_background_supportive"] = out["RS60"].gt(0) | out["rs60_pctile_by_week"].ge(0.50)
    out["rs_acceleration_warning_context"] = out["rs_short_acceleration_flag"].astype(bool)
    out["rs_deterioration_warning_context"] = out["rs_short_deterioration_flag"].astype(bool)
    out["rs_exhaustion_warning_context"] = out["rs60_high_short_rs_weakening_exhaustion_context"].astype(bool)
    return out


def _attach_stable_strong_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["layer1_pass_bottom20"] = ~out["layer1_exclude_bottom20_candidate"].astype(str).str.lower().eq("true")
    out["layer1_pass_bottom30"] = ~out["layer1_exclude_bottom30_candidate"].astype(str).str.lower().eq("true")
    out["capital_support_reasonable_not_topk_required"] = (
        out["capital_rank_20d_reasonable_band_300"] | out["capital_rank_60d_reasonable_band_300"]
    )
    out["stable_strong_protection_candidate"] = (
        out["layer1_pass_bottom20"]
        & out["capital_support_reasonable_not_topk_required"]
        & out["rs20_30_primary_momentum_stable"]
        & out["capital_reasonable_band_4w_persistent"]
        & ~out["rs_exhaustion_warning_context"]
    )
    out["multi_week_consistency_candidate"] = out["capital_reasonable_band_4w_persistent"] & out["rs20_30_primary_momentum_stable"]
    return out


def _attach_risk_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["bias20_overheat_context"] = out["BIAS20_percentile"].ge(0.90)
    out["bias60_overheat_context"] = out["BIAS60_percentile"].ge(0.90)
    out["bias_overheat_penalty_context"] = out["bias20_overheat_context"] | out["bias60_overheat_context"]
    out["volatility_available"] = out["volatility"].notna()
    out["volatility_pctile_by_week"] = out.groupby("snapshot_date")["volatility"].rank(pct=True, method="average")
    out["volatility_high_context"] = out["volatility_pctile_by_week"].ge(0.90)
    out["layer1_risk_high_context"] = out["layer1_quality_floor_risk_pctile_by_week"].ge(0.80)
    out["risk_overheat_penalty_context"] = (
        out["bias_overheat_penalty_context"] | out["volatility_high_context"] | out["layer1_risk_high_context"]
    )
    out["large_down_day_available"] = False
    out["blowoff_turnover_available"] = False
    out["risk_bucket_available"] = False
    return out


def _attach_policy_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["soft_score_feature_contract_only"] = True
    out["suggested_component_design_only"] = True
    out["selector_output"] = False
    out["live_rule"] = False
    out["diagnostic_only"] = True
    out["evaluation_metadata_only"] = True
    out["not_live_rule"] = True
    out["forward_return_as_rule"] = False
    out["future_return_as_rule"] = False
    out["forward_returns_live_rule_usage"] = False
    out["formal_model_changed"] = False
    out["trade_decision_changed"] = False
    out["active_in_trade_decision"] = False
    out["report_changed"] = False
    return out


def _missingness_by_period(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    feature_cols = [
        "traded_value_rank_20d",
        "traded_value_rank_60d",
        "capital_reasonable_band_4w_count",
        "RS20",
        "RS30_proxy",
        "RS60",
        "BIAS20_percentile",
        "BIAS60_percentile",
        "volatility",
        "stable_strong_protection_candidate",
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


def _blocked_proxy_fields() -> pd.DataFrame:
    rows = [
        ("RS30", "proxy", "exact RS30 unavailable; RS30_proxy uses midpoint of RS20 and RS40", "diagnostic only"),
        ("large_down_day_count", "blocked", "not materialized in compact Layer2 PIT contract", "do not fabricate"),
        ("blowoff_turnover_without_price_continuation", "blocked", "requires path/price continuation trigger contract", "do not fabricate"),
        ("risk_bucket", "blocked", "no accepted PIT risk_bucket field in current compact package", "use Layer1 risk pctile only as proxy/context"),
        ("market_cap_exact", "blocked", "exact daily market cap still blocked", "not used in soft score"),
        ("free_float_market_cap", "blocked", "free float market cap unavailable", "not used in soft score"),
        ("BIAS_percentile", "exact_or_pit_diagnostic", "BIAS percentile from stock_features as-of snapshot_date", "risk/overheat context only"),
        ("volatility", "exact_or_pit_diagnostic", "volatility from stock_features as-of snapshot_date", "risk context only"),
        ("forward_returns", "evaluation_metadata_only", "retained only for Experiments evaluation", "not live rule"),
    ]
    return pd.DataFrame(rows, columns=["field", "status", "blocked_or_proxy_reason", "policy"])


def _source_quality_matrix(df: pd.DataFrame) -> pd.DataFrame:
    rows = [
        ("capital_support", "traded_value_rank_20d/60d, rank change, 4w persistence", "exact_from_layer0_compact_pit", float(df["traded_value_rank_20d"].notna().mean())),
        ("relative_strength", "RS5/10/20/40/60 vs 0050", "exact_from_stock_features_pit", float(df["RS20"].notna().mean())),
        ("rs30_context", "RS30_proxy", "proxy_midpoint_rs20_rs40", float(df["RS30_proxy"].notna().mean())),
        ("stable_strong_protection", "Layer1 pass + capital reasonable + RS stable + no exhaustion", "diagnostic_composite_from_pit_features", float(df["stable_strong_protection_candidate"].notna().mean())),
        ("overheat_risk", "BIAS percentiles, volatility pctile, Layer1 risk pctile", "diagnostic_pit_context_partial", float(df["BIAS20_percentile"].notna().mean())),
        ("forward_evaluation", "20D excess vs 0050/00631L and decile labels", "evaluation_metadata_only", float(df["forward_eval_available_20d"].mean())),
    ]
    return pd.DataFrame(rows, columns=["feature_group", "fields", "source_quality", "available_share"])


def _suggested_component_design() -> pd.DataFrame:
    rows = [
        ("capital_support_component", "traded_value_rank_20d/60d percentile, 20d-vs-60d improvement, 4w reasonable-band persistence", "positive soft contribution", "do not hard-filter top-k; retain score features for Experiments sweep"),
        ("relative_strength_component", "RS20/RS30 primary momentum, RS60 background, RS5/10 acceleration", "positive soft contribution plus warning flags", "RS60 not universal hard gate"),
        ("stable_strong_protection_component", "Layer1 pass, capital reasonable, RS stable, multi-week consistency", "protect potential winners from single-week top-k miss", "candidate feature only"),
        ("risk_overheat_penalty_component", "BIAS percentile, volatility percentile, Layer1 risk pctile, RS exhaustion warning", "soft penalty context", "not automatic exclusion"),
        ("watchlist_burst_penalty_context", "pure 5D burst without 20D/60D confirmation", "watchlist or mild penalty context", "not direct elimination"),
    ]
    return pd.DataFrame(rows, columns=["component", "input_features", "suggested_role", "policy"])


def _future_data_audit() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("soft_score_feature_contract", "passed", 0, "features are as-of snapshot_date PIT fields"),
            ("forward_return_as_rule", "passed", 0, "forward returns retained only as evaluation_metadata_only"),
            ("selector_output", "not_applicable", 0, "no final score/selector output produced"),
            ("portfolio_replay", "not_executed", 0, "no replay executed"),
        ],
        columns=["audit_item", "status", "future_data_violation_count", "note"],
    )


def _readiness(
    rs_readiness: dict[str, Any],
    missed_summary: dict[str, Any],
    df: pd.DataFrame,
    missingness: pd.DataFrame,
) -> dict[str, Any]:
    rs20_share = float(df["RS20"].notna().mean())
    cap_share = float(df["traded_value_rank_20d"].notna().mean())
    bias_share = float(df["BIAS20_percentile"].notna().mean())
    volatility_share = float(df["volatility"].notna().mean())
    ready = rs20_share > 0.80 and cap_share > 0.95
    main_variant = missed_summary.get("main_variant_summary", {})
    return {
        "task_id": TASK_ID,
        "status": "layer2_soft_score_feature_contract_ready_for_experiments_planning" if ready else "layer2_soft_score_feature_contract_partial_blocked",
        "diagnostic_only": True,
        "soft_score_feature_contract_only": True,
        "suggested_component_design_only": True,
        "selector_output": False,
        "input_rs_join_status": rs_readiness.get("status", ""),
        "rows": int(len(df)),
        "weekly_snapshot_count": int(df["snapshot_date"].nunique()),
        "unique_ticker_count": int(df["ticker"].nunique()),
        "capital_support_available_share": cap_share,
        "rs20_available_share": rs20_share,
        "rs30_exact_available": False,
        "rs30_proxy_available_share": float(df["RS30_proxy"].notna().mean()),
        "bias20_percentile_available_share": bias_share,
        "volatility_available_share": volatility_share,
        "large_down_day_available": False,
        "blowoff_turnover_available": False,
        "risk_bucket_available": False,
        "missed_winner_attribution_context": {
            "variant": main_variant.get("variant", ""),
            "missed_top_decile_rate": main_variant.get("missed_top_decile_rate"),
            "bad_missed_winner_rate": main_variant.get("bad_missed_winner_rate"),
        },
        "ready_for_layer2_soft_score_bounded_diagnostic": ready,
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
        "blocked_fields": ["large_down_day_count", "blowoff_turnover_without_price_continuation", "risk_bucket", "exact_market_cap", "free_float_market_cap"],
        "proxy_fields": ["RS30_proxy", "BIAS_percentile_risk_context", "volatility_percentile_context"],
        "missingness_rows": int(len(missingness)),
    }


def _summary(readiness: dict[str, Any]) -> str:
    return f"""# Layer2 soft-score feature contract readiness

## Verdict
- status={readiness["status"]}
- rows={readiness["rows"]}
- weekly_snapshot_count={readiness["weekly_snapshot_count"]}
- unique_ticker_count={readiness["unique_ticker_count"]}
- capital_support_available_share={readiness["capital_support_available_share"]}
- rs20_available_share={readiness["rs20_available_share"]}
- rs30_exact_available=false
- rs30_proxy_available_share={readiness["rs30_proxy_available_share"]}
- bias20_percentile_available_share={readiness["bias20_percentile_available_share"]}
- volatility_available_share={readiness["volatility_available_share"]}
- ready_for_layer2_soft_score_bounded_diagnostic={str(readiness["ready_for_layer2_soft_score_bounded_diagnostic"]).lower()}
- ready_for_formal=false
- portfolio_replay_executed=false

## Plain Summary
This package converts Layer2 hard-filter inputs into a diagnostic soft-score feature contract. It keeps capital support, RS windows, stable-strong protection, and risk/overheat context as features only. It does not output a selector, live rule, replay, daily report, or formal model change.

## Blocked / Proxy
- RS30 is proxy only.
- large_down_day_count, blowoff_turnover, and risk_bucket remain blocked.
- BIAS and volatility are diagnostic PIT context only.

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
    parser.add_argument("--layer1-dir", default=str(DEFAULT_LAYER1_DIR))
    parser.add_argument("--rs-join-dir", default=str(DEFAULT_RS_JOIN_DIR))
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--experiments-dir", default=str(DEFAULT_EXPERIMENTS_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()
    manifest = build_contract(
        layer1_dir=args.layer1_dir,
        rs_join_dir=args.rs_join_dir,
        data_dir=args.data_dir,
        experiments_dir=args.experiments_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
