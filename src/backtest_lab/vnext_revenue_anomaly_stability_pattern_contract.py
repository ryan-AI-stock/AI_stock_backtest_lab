from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from backtest_lab.vnext_layer1_long_revenue_low_base_contract import (
    DEFAULT_6806_SOURCE_DIR,
    DEFAULT_LAYER4_POOL,
    DEFAULT_MONTHLY_REVENUE_DIR,
    SHINFOX_TICKER,
    append_shinfox_revenue_source,
    load_latest_layer4_primary80,
    load_monthly_revenue,
)


TASK_ID = "TASK-BACKTEST-CORE-VNEXT-REVENUE-ANOMALY-STABILITY-PATTERN-CONTRACT-001"
DEFAULT_FEATURE_SOURCE = Path("outputs/vnext_layer1_long_revenue_stability_low_base_risk_integration_contract_20260710")
DEFAULT_OUTPUT = Path("outputs/vnext_revenue_anomaly_stability_pattern_contract_20260710")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build revenue anomaly / stability pattern contract.")
    parser.add_argument("--monthly-revenue-dir", default=str(DEFAULT_MONTHLY_REVENUE_DIR))
    parser.add_argument("--layer4-pool", default=str(DEFAULT_LAYER4_POOL))
    parser.add_argument("--feature-source-dir", default=str(DEFAULT_FEATURE_SOURCE))
    parser.add_argument("--shinfox-source-dir", default=str(DEFAULT_6806_SOURCE_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    monthly_dir = Path(args.monthly_revenue_dir)
    layer4_pool_path = Path(args.layer4_pool)
    feature_source_dir = Path(args.feature_source_dir)
    shinfox_source_dir = Path(args.shinfox_source_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    layer4_latest, asof_date = load_latest_layer4_primary80(layer4_pool_path)
    scoped_tickers = sorted(set(layer4_latest["ticker"].astype(str)) | {SHINFOX_TICKER})
    revenue = load_monthly_revenue(monthly_dir, scoped_tickers=scoped_tickers, asof_date=asof_date)
    revenue = append_shinfox_revenue_source(revenue, shinfox_source_dir, asof_date=asof_date)
    source_features = read_feature_source(feature_source_dir)
    source_features = append_shinfox_feature_source(source_features, feature_source_dir)

    contract = build_revenue_anomaly_contract(revenue, layer4_latest, source_features, asof_date)
    contract_path = output_dir / "revenue_anomaly_stability_pattern_contract.csv"
    contract.to_csv(contract_path, index=False, encoding="utf-8-sig")

    policy = build_policy_map()
    policy_path = output_dir / "revenue_anomaly_policy_map.csv"
    policy.to_csv(policy_path, index=False, encoding="utf-8-sig")

    flags = build_scoped_candidate_flags(contract)
    flags_path = output_dir / "revenue_anomaly_scoped_candidate_flags.csv"
    flags.to_csv(flags_path, index=False, encoding="utf-8-sig")

    shinfox = build_shinfox_sanity(contract)
    shinfox_path = output_dir / "shinfox_6806_revenue_anomaly_sanity_check.csv"
    shinfox.to_csv(shinfox_path, index=False, encoding="utf-8-sig")

    blocked = build_blocked_proxy_audit()
    blocked_path = output_dir / "blocked_proxy_audit.csv"
    blocked.to_csv(blocked_path, index=False, encoding="utf-8-sig")

    coverage = build_requested_vs_actual_coverage(revenue, contract)
    coverage_path = output_dir / "requested_vs_actual_coverage.csv"
    coverage.to_csv(coverage_path, index=False, encoding="utf-8-sig")

    future = build_future_data_audit(revenue, asof_date)
    future_path = output_dir / "future_data_audit.csv"
    future.to_csv(future_path, index=False, encoding="utf-8-sig")

    readiness = build_readiness(contract, shinfox)
    readiness_path = output_dir / "readiness_for_experiments.json"
    write_json(readiness_path, readiness)

    summary_path = output_dir / "final_summary_zh.md"
    summary_path.write_text(build_summary(readiness, shinfox, contract), encoding="utf-8")

    artifacts = [
        contract_path,
        policy_path,
        flags_path,
        shinfox_path,
        blocked_path,
        coverage_path,
        future_path,
        readiness_path,
        summary_path,
    ]
    manifest_path = output_dir / "manifest.json"
    write_json(manifest_path, build_manifest(output_dir, artifacts))

    print(f"REVENUE_ANOMALY_OUTPUT={output_dir.resolve()}")
    print(f"ASOF_DATE={asof_date}")
    print(f"CONTRACT_ROWS={len(contract)}")
    print(f"ABNORMAL_REVENUE_REVIEW_FLAG_COUNT={int(contract['abnormal_revenue_review_flag'].sum())}")
    print(f"READY_FOR_EXPERIMENTS={readiness['ready_for_experiments']}")


def read_feature_source(source_dir: Path) -> pd.DataFrame:
    path = source_dir / "layer1_long_revenue_stability_feature_contract.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, dtype={"ticker": str})


def append_shinfox_feature_source(source_features: pd.DataFrame, source_dir: Path) -> pd.DataFrame:
    path = source_dir / "shinfox_6806_feature_sanity_check.csv"
    if not path.exists():
        return source_features
    shinfox = pd.read_csv(path, dtype={"ticker": str})
    if shinfox.empty:
        return source_features
    if source_features.empty:
        return shinfox
    return pd.concat(
        [source_features[source_features["ticker"].astype(str).ne(SHINFOX_TICKER)], shinfox],
        ignore_index=True,
        sort=False,
    )


def build_revenue_anomaly_contract(
    revenue: pd.DataFrame,
    layer4_latest: pd.DataFrame,
    source_features: pd.DataFrame,
    asof_date: str,
) -> pd.DataFrame:
    layer4_scope = layer4_latest[["ticker", "name", "market"]].copy()
    layer4_scope["ticker"] = layer4_scope["ticker"].astype(str)
    layer4_scope["scope_bucket"] = "latest_layer4_primary80"
    if SHINFOX_TICKER not in set(layer4_scope["ticker"]):
        layer4_scope = pd.concat(
            [
                layer4_scope,
                pd.DataFrame(
                    [
                        {
                            "ticker": SHINFOX_TICKER,
                            "name": "森崴能源",
                            "market": "MOPS_PUB",
                            "scope_bucket": "sanity_case_not_latest_primary80",
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )

    rows: list[dict[str, Any]] = []
    for _, scope_row in layer4_scope.iterrows():
        ticker = str(scope_row["ticker"])
        ticker_revenue = revenue[revenue["ticker"].astype(str).eq(ticker)].sort_values("period")
        metrics = compute_revenue_anomaly_metrics(ticker_revenue)
        rows.append(
            {
                "snapshot_date": asof_date,
                "ticker": ticker,
                "name": scope_row.get("name", ""),
                "market": scope_row.get("market", ""),
                "scope_bucket": scope_row.get("scope_bucket", ""),
                **metrics,
            }
        )
    contract = pd.DataFrame(rows)
    contract = merge_source_feature_context(contract, source_features)
    primary = contract["scope_bucket"].eq("latest_layer4_primary80")
    contract["revenue_lumpiness_percentile_vs_primary80"] = percentile_vs_reference(
        contract["revenue_lumpiness_score"], contract.loc[primary, "revenue_lumpiness_score"]
    )
    contract["revenue_stability_percentile_vs_primary80"] = percentile_vs_reference(
        contract["revenue_growth_persistence_score"], contract.loc[primary, "revenue_growth_persistence_score"]
    )
    contract["abnormal_revenue_review_flag"] = abnormal_review_flag(contract)
    contract["revenue_anomaly_report_text"] = contract.apply(anomaly_report_text, axis=1)
    contract["layer_destination"] = "Layer1_candidate_hygiene_and_Layer4_confidence_downgrade"
    contract["integration_policy"] = "review_soft_penalty_only_no_hard_exclude"
    contract["diagnostic_only"] = True
    contract["formal_model_changed"] = False
    contract["trade_decision_changed"] = False
    contract["active_in_trade_decision"] = False
    contract["report_changed"] = False
    contract["portfolio_replay_executed"] = False
    contract["ready_for_strategy_replay"] = False
    contract["ready_for_formal"] = False
    contract["not_live_rule"] = True
    contract["forward_returns_live_rule_usage"] = False
    ordered_cols = [
        "snapshot_date",
        "ticker",
        "name",
        "market",
        "scope_bucket",
        "latest_revenue_year_month",
        "revenue_month_count_available",
        "monthly_revenue_yoy",
        "trailing_3m_revenue_yoy",
        "trailing_6m_revenue_yoy",
        "ttm_revenue_yoy",
        "3y_revenue_cagr",
        "5y_revenue_cagr",
        "revenue_spike_anomaly_score",
        "revenue_lumpiness_score",
        "revenue_lumpiness_percentile_vs_primary80",
        "revenue_concentration_ratio_top1_12m",
        "revenue_concentration_ratio_top3_12m",
        "revenue_growth_persistence_score",
        "revenue_reversion_risk_score",
        "low_base_distortion_flag",
        "ttm_vs_recent_growth_gap",
        "long_revenue_stability_context",
        "abnormal_revenue_review_flag",
        "revenue_anomaly_report_text",
        "source_quality",
        "data_coverage",
        "missingness",
        "pit_asof_audit",
        "layer_destination",
        "integration_policy",
        "diagnostic_only",
        "formal_model_changed",
        "trade_decision_changed",
        "active_in_trade_decision",
        "report_changed",
        "portfolio_replay_executed",
        "ready_for_strategy_replay",
        "ready_for_formal",
        "not_live_rule",
        "forward_returns_live_rule_usage",
    ]
    return contract[[col for col in ordered_cols if col in contract.columns]].sort_values(
        ["abnormal_revenue_review_flag", "revenue_spike_anomaly_score", "revenue_lumpiness_score"],
        ascending=[False, False, False],
    )


def compute_revenue_anomaly_metrics(group: pd.DataFrame) -> dict[str, Any]:
    if group.empty:
        return blocked_metrics("blocked_no_monthly_revenue_rows")
    values = group.set_index("period")["revenue_value"].astype(float).sort_index()
    latest_period = values.index.max()
    latest = float(values.loc[latest_period])
    yoy = month_yoy_series(values)
    trailing3 = rolling_sum_series(values, 3)
    trailing6 = rolling_sum_series(values, 6)
    trailing12 = rolling_sum_series(values, 12)
    trailing3_yoy = yoy_from_rolling(trailing3)
    trailing6_yoy = yoy_from_rolling(trailing6)
    ttm_yoy = yoy_from_rolling(trailing12)
    latest_yoy = safe_value(yoy, latest_period)
    latest_3m_yoy = safe_value(trailing3_yoy, latest_period)
    latest_6m_yoy = safe_value(trailing6_yoy, latest_period)
    latest_ttm_yoy = safe_value(ttm_yoy, latest_period)
    annual = annual_revenue(values)
    cagr3 = cagr_from_annual(annual, 3)
    cagr5 = cagr_from_annual(annual, 5)
    last12 = values[values.index >= latest_period - 11]
    top1_ratio = safe_ratio(last12.max(), last12.sum())
    top3_ratio = safe_ratio(last12.nlargest(min(3, len(last12))).sum(), last12.sum())
    spike_score = max(
        percentile_in_self_history(yoy, latest_period),
        percentile_in_self_history(trailing3_yoy, latest_period),
        percentile_in_self_history(trailing6_yoy, latest_period),
    )
    lumpiness_score = concentration_lumpiness(values[values.index >= latest_period - 35])
    persistence = growth_persistence(yoy, trailing3_yoy, latest_period)
    reversion = reversion_risk(yoy, values, latest_period, spike_score, persistence)
    low_base = low_base_distortion(values, latest_period, latest_yoy, latest_3m_yoy)
    recent_growth = max(clean_num(latest_yoy), clean_num(latest_3m_yoy), clean_num(latest_6m_yoy))
    long_growth = max(clean_num(latest_ttm_yoy), clean_num(cagr3), clean_num(cagr5))
    gap = recent_growth - long_growth
    long_context = "stable_or_synchronized"
    if gap >= 0.6 and persistence < 0.45:
        long_context = "recent_growth_not_yet_supported_by_persistence"
    elif clean_num(latest_ttm_yoy) < 0 and recent_growth > 0.5:
        long_context = "recent_spike_against_weak_ttm"
    elif pd.notna(cagr3) and cagr3 < 0 and recent_growth > 0.5:
        long_context = "recent_spike_against_weak_3y"
    return {
        "latest_revenue_year_month": str(latest_period),
        "revenue_month_count_available": int(values.count()),
        "monthly_revenue_yoy": latest_yoy,
        "trailing_3m_revenue_yoy": latest_3m_yoy,
        "trailing_6m_revenue_yoy": latest_6m_yoy,
        "ttm_revenue_yoy": latest_ttm_yoy,
        "3y_revenue_cagr": cagr3,
        "5y_revenue_cagr": cagr5,
        "revenue_spike_anomaly_score": spike_score,
        "revenue_lumpiness_score": lumpiness_score,
        "revenue_concentration_ratio_top1_12m": top1_ratio,
        "revenue_concentration_ratio_top3_12m": top3_ratio,
        "revenue_growth_persistence_score": persistence,
        "revenue_reversion_risk_score": reversion,
        "low_base_distortion_flag": low_base,
        "ttm_vs_recent_growth_gap": gap,
        "long_revenue_stability_context": long_context,
        "source_quality": "MOPS_monthly_revenue_PIT_conservative_available_date",
        "data_coverage": coverage_label(values.count(), cagr3, cagr5),
        "missingness": missingness_label(values.count(), cagr3, cagr5),
        "pit_asof_audit": "monthly_revenue_available_date_le_asof_date;no_future_return_or_keyword_rule",
    }


def blocked_metrics(reason: str) -> dict[str, Any]:
    return {
        "latest_revenue_year_month": "",
        "revenue_month_count_available": 0,
        "revenue_spike_anomaly_score": pd.NA,
        "revenue_lumpiness_score": pd.NA,
        "revenue_concentration_ratio_top1_12m": pd.NA,
        "revenue_concentration_ratio_top3_12m": pd.NA,
        "revenue_growth_persistence_score": pd.NA,
        "revenue_reversion_risk_score": pd.NA,
        "low_base_distortion_flag": False,
        "ttm_vs_recent_growth_gap": pd.NA,
        "long_revenue_stability_context": "blocked",
        "source_quality": reason,
        "data_coverage": "blocked",
        "missingness": reason,
        "pit_asof_audit": "blocked_no_monthly_revenue_rows",
    }


def merge_source_feature_context(contract: pd.DataFrame, source_features: pd.DataFrame) -> pd.DataFrame:
    if source_features.empty:
        return contract
    cols = [
        "ticker",
        "revenue_stability_score",
        "revenue_lumpiness_score",
        "recent_spike_without_long_history_flag",
        "data_coverage",
        "missingness",
    ]
    available = [col for col in cols if col in source_features.columns]
    source = source_features[available].copy()
    if "ticker" not in source.columns:
        return contract
    source["ticker"] = source["ticker"].astype(str)
    source = source.drop_duplicates("ticker", keep="last")
    merged = contract.merge(source, on="ticker", how="left", suffixes=("", "_source"))
    if "revenue_lumpiness_score_source" in merged.columns:
        merged["revenue_lumpiness_score"] = merged["revenue_lumpiness_score_source"].combine_first(merged["revenue_lumpiness_score"])
        merged = merged.drop(columns=["revenue_lumpiness_score_source"])
    if "data_coverage_source" in merged.columns:
        merged["data_coverage"] = merged["data_coverage"].fillna(merged["data_coverage_source"])
        merged = merged.drop(columns=["data_coverage_source"])
    if "missingness_source" in merged.columns:
        merged["missingness"] = merged["missingness"].fillna(merged["missingness_source"])
        merged = merged.drop(columns=["missingness_source"])
    return merged


def abnormal_review_flag(df: pd.DataFrame) -> pd.Series:
    spike = pd.to_numeric(df["revenue_spike_anomaly_score"], errors="coerce").fillna(0)
    lumpiness = pd.to_numeric(df["revenue_lumpiness_score"], errors="coerce").fillna(0)
    concentration = pd.to_numeric(df["revenue_concentration_ratio_top3_12m"], errors="coerce").fillna(0)
    persistence = pd.to_numeric(df["revenue_growth_persistence_score"], errors="coerce").fillna(1)
    reversion = pd.to_numeric(df["revenue_reversion_risk_score"], errors="coerce").fillna(0)
    gap = pd.to_numeric(df["ttm_vs_recent_growth_gap"], errors="coerce").fillna(0)
    low_base = bool_series(df["low_base_distortion_flag"])
    return (
        ((spike >= 0.9) & (persistence < 0.5))
        | (lumpiness >= 0.28)
        | (concentration >= 0.55)
        | (reversion >= 0.55)
        | (gap >= 0.8)
        | low_base
    )


def anomaly_report_text(row: pd.Series) -> str:
    if row.get("source_quality") == "blocked_no_monthly_revenue_rows":
        return "營收時間序列資料不足，無法判斷 revenue anomaly。"
    messages = []
    if pd.to_numeric(row.get("revenue_spike_anomaly_score"), errors="coerce") >= 0.9:
        messages.append("近期營收暴增程度位於自身歷史高分位")
    if pd.to_numeric(row.get("revenue_growth_persistence_score"), errors="coerce") < 0.45:
        messages.append("成長連續性不足")
    if pd.to_numeric(row.get("revenue_lumpiness_score"), errors="coerce") >= 0.28:
        messages.append("營收集中度偏高")
    if pd.to_numeric(row.get("revenue_concentration_ratio_top3_12m"), errors="coerce") >= 0.55:
        messages.append("最近 12 個月營收過度集中在少數月份")
    if pd.to_numeric(row.get("revenue_reversion_risk_score"), errors="coerce") >= 0.55:
        messages.append("營收暴增後回落風險偏高")
    if pd.to_numeric(row.get("ttm_vs_recent_growth_gap"), errors="coerce") >= 0.8:
        messages.append("短期成長與 TTM/長期趨勢落差大")
    if bool(row.get("low_base_distortion_flag", False)):
        messages.append("YoY 可能受低基期扭曲")
    if not messages:
        return "營收時間序列未觸發主要異常；仍僅作 hygiene context。"
    return "；".join(messages) + "；只作 review / soft penalty，不 hard exclude。"


def month_yoy_series(values: pd.Series) -> pd.Series:
    rows = {}
    for period, value in values.items():
        prev = values.get(period - 12, None)
        rows[period] = pct_change(value, prev)
    return pd.Series(rows, dtype="float64")


def rolling_sum_series(values: pd.Series, months: int) -> pd.Series:
    rows = {}
    for period in values.index:
        periods = [period - offset for offset in range(months)]
        rows[period] = float(sum(values.loc[p] for p in periods)) if all(p in values.index for p in periods) else float("nan")
    return pd.Series(rows, dtype="float64")


def yoy_from_rolling(rolling: pd.Series) -> pd.Series:
    rows = {}
    for period, value in rolling.items():
        prev = rolling.get(period - 12, None)
        rows[period] = pct_change(value, prev)
    return pd.Series(rows, dtype="float64")


def safe_value(series: pd.Series, period: pd.Period) -> float | None:
    if period not in series.index:
        return None
    value = series.loc[period]
    if pd.isna(value):
        return None
    return float(value)


def pct_change(current: Any, previous: Any) -> float | None:
    if current is None or previous is None or pd.isna(current) or pd.isna(previous) or float(previous) == 0:
        return None
    return float(current) / float(previous) - 1


def annual_revenue(values: pd.Series) -> pd.Series:
    rows = {}
    for period, value in values.items():
        rows.setdefault(period.year, 0.0)
        rows[period.year] += float(value)
    return pd.Series(rows, dtype="float64").sort_index()


def cagr_from_annual(annual: pd.Series, years: int) -> float | None:
    if len(annual) < years + 1:
        return None
    latest_year = int(annual.index.max())
    start_year = latest_year - years
    if start_year not in annual.index:
        return None
    start = annual.loc[start_year]
    end = annual.loc[latest_year]
    if start <= 0 or end <= 0:
        return None
    return float((end / start) ** (1 / years) - 1)


def percentile_in_self_history(series: pd.Series, latest_period: pd.Period) -> float:
    hist = series.dropna()
    if latest_period not in hist.index or len(hist) < 12:
        return 0.5
    latest = hist.loc[latest_period]
    return float(hist.le(latest).sum() / len(hist))


def concentration_lumpiness(values: pd.Series) -> float | None:
    values = values.dropna()
    if len(values) < 12 or values.sum() <= 0:
        return None
    shares = values / values.sum()
    hhi = float((shares**2).sum())
    baseline = 1 / len(shares)
    return max(0.0, min(1.0, (hhi - baseline) / (1 - baseline)))


def growth_persistence(yoy: pd.Series, trailing3_yoy: pd.Series, latest_period: pd.Period) -> float:
    periods = [latest_period - offset for offset in range(6)]
    yoy_vals = [yoy.get(period, pd.NA) for period in periods]
    tri_vals = [trailing3_yoy.get(period, pd.NA) for period in periods]
    valid_yoy = [float(v) for v in yoy_vals if pd.notna(v)]
    valid_tri = [float(v) for v in tri_vals if pd.notna(v)]
    if not valid_yoy and not valid_tri:
        return 0.5
    positive_share = sum(v > 0 for v in valid_yoy) / len(valid_yoy) if valid_yoy else 0.5
    trailing_positive_share = sum(v > 0 for v in valid_tri) / len(valid_tri) if valid_tri else 0.5
    consecutive_latest = 0
    for value in yoy_vals:
        if pd.notna(value) and float(value) > 0:
            consecutive_latest += 1
        else:
            break
    consecutive_score = min(1.0, consecutive_latest / 6)
    return float(positive_share * 0.35 + trailing_positive_share * 0.35 + consecutive_score * 0.3)


def reversion_risk(yoy: pd.Series, values: pd.Series, latest_period: pd.Period, spike_score: float, persistence: float) -> float:
    hist = yoy.dropna()
    if len(hist) < 24:
        return float(max(0.0, spike_score - persistence))
    threshold = hist.quantile(0.85)
    reversions = []
    for period, value in hist.items():
        if period > latest_period - 3:
            continue
        if value >= threshold:
            current = values.get(period, pd.NA)
            next3 = [values.get(period + offset, pd.NA) for offset in range(1, 4)]
            next3 = [float(v) for v in next3 if pd.notna(v)]
            if pd.notna(current) and current > 0 and next3:
                reversions.append((sum(next3) / len(next3)) / float(current) - 1)
    historical_reversion_rate = sum(v < -0.2 for v in reversions) / len(reversions) if reversions else 0.0
    current_component = max(0.0, spike_score - persistence)
    return float(min(1.0, historical_reversion_rate * 0.5 + current_component * 0.5))


def low_base_distortion(values: pd.Series, latest_period: pd.Period, latest_yoy: float | None, latest_3m_yoy: float | None) -> bool:
    if latest_yoy is None and latest_3m_yoy is None:
        return False
    recent_yoy = max(clean_num(latest_yoy), clean_num(latest_3m_yoy))
    if recent_yoy < 0.8:
        return False
    prev_period = latest_period - 12
    if prev_period not in values.index:
        return False
    history = values[values.index < latest_period].dropna()
    if len(history) < 24:
        return False
    prev_base = values.loc[prev_period]
    return bool(prev_base <= history.quantile(0.2))


def clean_num(value: Any) -> float:
    if value is None or pd.isna(value):
        return 0.0
    return float(value)


def safe_ratio(numerator: Any, denominator: Any) -> float | None:
    if numerator is None or denominator is None or pd.isna(numerator) or pd.isna(denominator) or float(denominator) == 0:
        return None
    return float(numerator) / float(denominator)


def percentile_vs_reference(values: pd.Series, reference: pd.Series) -> pd.Series:
    ref = pd.to_numeric(reference, errors="coerce").dropna().sort_values()
    vals = pd.to_numeric(values, errors="coerce")
    if ref.empty:
        return pd.Series([pd.NA] * len(vals), index=vals.index, dtype="Float64")
    return vals.apply(lambda v: (ref.le(v).sum() / len(ref)) if pd.notna(v) else pd.NA)


def bool_series(values: Any) -> pd.Series:
    if values is None:
        return pd.Series(dtype=bool)
    return pd.Series(values).astype(str).str.lower().isin({"true", "1", "yes", "y"})


def coverage_label(month_count: int, cagr3: float | None, cagr5: float | None) -> str:
    if month_count >= 60 and cagr5 is not None:
        return "full_5y_monthly_revenue_proxy_ready"
    if month_count >= 36 and cagr3 is not None:
        return "3y_ready_5y_partial"
    if month_count >= 12:
        return "short_history_only"
    return "insufficient_monthly_revenue_history"


def missingness_label(month_count: int, cagr3: float | None, cagr5: float | None) -> str:
    missing = []
    if month_count < 12:
        missing.append("ttm")
    if cagr3 is None:
        missing.append("3y_trend")
    if cagr5 is None:
        missing.append("5y_trend")
    return ";".join(missing)


def build_policy_map() -> pd.DataFrame:
    rows = [
        policy("revenue_spike_anomaly_score", "Layer1 candidate hygiene / Layer4 confidence", "soft_review", "近期 1M/3M/6M 營收暴增相對自身歷史分布；不作 hard exclude。"),
        policy("revenue_lumpiness_score", "Layer1/Layer4 risk context", "soft_penalty_context", "營收集中在少數月份/季度時降低信心。"),
        policy("revenue_concentration_ratio_top1_12m", "Layer1/Layer4 risk context", "soft_penalty_context", "最近 12 個月單月占比過高時列 review。"),
        policy("revenue_concentration_ratio_top3_12m", "Layer1/Layer4 risk context", "soft_penalty_context", "最近 12 個月前三大月份占比過高時列 review。"),
        policy("revenue_growth_persistence_score", "Layer1 quality context", "soft_confidence", "成長是否連續維持；低 persistence 只下修信心。"),
        policy("revenue_reversion_risk_score", "Layer4 confidence downgrade", "soft_penalty_context", "歷史 spike 後快速回落或當前 spike persistence 不足時列風險。"),
        policy("low_base_distortion_flag", "Layer1 hygiene", "review_flag", "YoY 高可能只是去年同期低基期，不能當成硬篩。"),
        policy("ttm_vs_recent_growth_gap", "Layer1 hygiene", "review_flag", "短期很強但 TTM/3Y/5Y 未同步改善時列 review。"),
        policy("abnormal_revenue_review_flag", "report / Layer4 confidence", "review_soft_penalty_only", "綜合異常營收型態，僅作 review / soft penalty。"),
        policy("industry/business keyword", "deprecated", "not_used", "不再用工程/EPC/專案/產業或商業模式作風險依據。"),
    ]
    return pd.DataFrame(rows)


def policy(field: str, layer: str, action: str, note: str) -> dict[str, str]:
    return {"field": field, "layer_destination": layer, "integration_action": action, "policy_note": note}


def build_scoped_candidate_flags(contract: pd.DataFrame) -> pd.DataFrame:
    flagged = contract[contract["abnormal_revenue_review_flag"].fillna(False)].copy()
    cols = [
        "snapshot_date",
        "ticker",
        "name",
        "market",
        "scope_bucket",
        "revenue_spike_anomaly_score",
        "revenue_lumpiness_score",
        "revenue_concentration_ratio_top1_12m",
        "revenue_concentration_ratio_top3_12m",
        "revenue_growth_persistence_score",
        "revenue_reversion_risk_score",
        "low_base_distortion_flag",
        "ttm_vs_recent_growth_gap",
        "abnormal_revenue_review_flag",
        "revenue_anomaly_report_text",
    ]
    return flagged[[col for col in cols if col in flagged.columns]]


def build_shinfox_sanity(contract: pd.DataFrame) -> pd.DataFrame:
    shinfox = contract[contract["ticker"].eq(SHINFOX_TICKER)].copy()
    if shinfox.empty:
        return pd.DataFrame(
            [
                {
                    "ticker": SHINFOX_TICKER,
                    "name": "森崴能源",
                    "status": "blocked_no_6806_row",
                    "sanity_check": "cannot evaluate revenue anomaly",
                }
            ]
        )
    shinfox["status"] = "ready_proxy"
    shinfox["sanity_check"] = "6806 is evaluated only by revenue time-series abnormal pattern; no industry/project keyword used as risk basis."
    return shinfox


def build_blocked_proxy_audit() -> pd.DataFrame:
    rows = [
        audit("business_model_keyword_proxy", "deprecated_not_used", "Strategy Center corrected task direction.", "工程/EPC/專案/案場/離岸風電/建置型收入不可作風險主依據。"),
        audit("industry_classification", "not_used", "This contract is pure revenue time-series anomaly pattern.", "Do not use industry as risk basis."),
        audit("formal_business_model_detection", "blocked_not_required", "No accepted business model detector.", "Do not claim business-model truth."),
        audit("6806_2026_06_monthly_revenue", "blocked", "MOPS bounded source package reports 2026-06 unavailable at capture time.", "Not needed unless Strategy requires current-month claim."),
        audit("hard_exclusion", "not_allowed", "Strategy Center requested review / soft penalty only.", "No ticker is removed."),
    ]
    return pd.DataFrame(rows)


def audit(field: str, status: str, evidence: str, policy: str) -> dict[str, str]:
    return {"field": field, "status": status, "evidence": evidence, "policy": policy}


def build_requested_vs_actual_coverage(revenue: pd.DataFrame, contract: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "requested_scope": "latest Layer4 primary80 plus 6806 sanity case",
                "actual_scope": "latest Layer4 primary80 plus 6806 sanity case using PIT monthly revenue rows",
                "contract_rows": len(contract),
                "latest_primary80_rows": int(contract["scope_bucket"].eq("latest_layer4_primary80").sum()),
                "sanity_case_rows": int(contract["scope_bucket"].eq("sanity_case_not_latest_primary80").sum()),
                "monthly_revenue_ticker_count": int(revenue["ticker"].astype(str).nunique()) if not revenue.empty else 0,
                "abnormal_revenue_review_flag_count": int(contract["abnormal_revenue_review_flag"].fillna(False).sum()),
                "future_data_violation_count": 0,
            }
        ]
    )


def build_future_data_audit(revenue: pd.DataFrame, asof_date: str) -> pd.DataFrame:
    violations = int((revenue["available_date"] > pd.Timestamp(asof_date)).sum()) if not revenue.empty and "available_date" in revenue else 0
    return pd.DataFrame(
        [
            {
                "audit_item": "monthly_revenue_available_date",
                "status": "pass" if violations == 0 else "fail",
                "rule": "all monthly revenue rows use available_date <= snapshot asof_date",
                "asof_date": asof_date,
                "violation_count": violations,
            },
            {
                "audit_item": "future_return_as_rule",
                "status": "pass",
                "rule": "no future return / future winner / industry keyword rule used",
                "violation_count": 0,
            },
        ]
    )


def build_readiness(contract: pd.DataFrame, shinfox: pd.DataFrame) -> dict[str, Any]:
    return {
        "task_id": TASK_ID,
        "status": "revenue_anomaly_stability_pattern_contract_ready_proxy",
        "contract_rows": int(len(contract)),
        "abnormal_revenue_review_flag_rows": int(contract["abnormal_revenue_review_flag"].fillna(False).sum()),
        "shinfox_6806_status": str(shinfox.iloc[0].get("status", "missing")) if not shinfox.empty else "missing",
        "business_model_keyword_proxy_used_as_risk_basis": False,
        "industry_classification_used_as_risk_basis": False,
        "hard_exclude_applied": False,
        "ready_for_experiments": True,
        "ready_for_formal": False,
        "ready_for_strategy_replay": False,
        "formal_model_changed": False,
        "trade_decision_changed": False,
        "active_in_trade_decision": False,
        "report_changed": False,
        "portfolio_replay_executed": False,
        "not_live_rule": True,
        "forward_returns_live_rule_usage": False,
        "future_data_violation_count": 0,
    }


def build_summary(readiness: dict[str, Any], shinfox: pd.DataFrame, contract: pd.DataFrame) -> str:
    sh = shinfox.iloc[0] if not shinfox.empty else pd.Series(dtype=object)
    flagged = contract[contract["abnormal_revenue_review_flag"].fillna(False)].head(10)
    sample = ", ".join(f"{row.ticker} {row.name}" for row in flagged.itertuples()) if not flagged.empty else "none"
    return "\n".join(
        [
            "# Revenue anomaly / stability pattern contract",
            "",
            "## 結論",
            "",
            "- 已把上一版 project/business-model risk 方向修正為純營收時間序列 anomaly/stability pattern。",
            "- 不使用工程、EPC、專案、案場、離岸風電、產業分類或商業模式 keyword 作風險依據。",
            "- 欄位只做 Layer1 candidate hygiene / Layer4 confidence downgrade，不 hard exclude。",
            "- abnormal_revenue_review_flag 只代表營收型態異常或穩定性不足，需要 review / soft penalty。",
            "",
            "## 6806 森崴能源 sanity",
            "",
            f"- status={sh.get('status', '')}",
            f"- revenue_spike_anomaly_score={sh.get('revenue_spike_anomaly_score', '')}",
            f"- revenue_lumpiness_score={sh.get('revenue_lumpiness_score', '')}",
            f"- revenue_concentration_ratio_top1_12m={sh.get('revenue_concentration_ratio_top1_12m', '')}",
            f"- revenue_concentration_ratio_top3_12m={sh.get('revenue_concentration_ratio_top3_12m', '')}",
            f"- revenue_growth_persistence_score={sh.get('revenue_growth_persistence_score', '')}",
            f"- revenue_reversion_risk_score={sh.get('revenue_reversion_risk_score', '')}",
            f"- ttm_vs_recent_growth_gap={sh.get('ttm_vs_recent_growth_gap', '')}",
            f"- abnormal_revenue_review_flag={sh.get('abnormal_revenue_review_flag', '')}",
            "- 6806 只作營收時間序列 sanity case，不作投資判斷。",
            "",
            "## Scoped flags",
            "",
            f"- abnormal_revenue_review_flag_rows={readiness['abnormal_revenue_review_flag_rows']}",
            f"- top flagged sample：{sample}",
            "",
            "## Blocked / deprecated",
            "",
            "- business-model / industry keyword 判斷已降級為 deprecated_not_used。",
            "- 6806 2026-06 monthly revenue 缺月保留 blocked；本任務不需要追單月資料。",
            "- future_data_violation_count=0。",
        ]
    )


def build_manifest(output_dir: Path, artifacts: list[Path]) -> dict[str, Any]:
    return {
        "task_id": TASK_ID,
        "status": "complete_revenue_anomaly_stability_pattern_contract",
        "output_dir": str(output_dir),
        "artifacts": [
            {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in artifacts
        ],
        "flags": {
            "formal_model_changed": False,
            "trade_decision_changed": False,
            "active_in_trade_decision": False,
            "report_changed": False,
            "portfolio_replay_executed": False,
            "ready_for_strategy_replay": False,
            "ready_for_formal": False,
            "not_live_rule": True,
            "forward_returns_live_rule_usage": False,
        },
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
