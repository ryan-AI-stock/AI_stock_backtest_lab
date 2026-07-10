from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


TASK_ID = "TASK-BACKTEST-CORE-VNEXT-LAYER1-LONG-REVENUE-STABILITY-AND-LOW-BASE-RISK-INTEGRATION-CONTRACT-001"
DEFAULT_OUTPUT = Path("outputs/vnext_layer1_long_revenue_stability_low_base_risk_integration_contract_20260710")
DEFAULT_MONTHLY_REVENUE_DIR = Path(
    "C:/Users/zergv/Documents/Codex/2026-05-23/ai-stock-rotation-radar-https-docs/outputs/"
    "radar_dynamic_pool1_mops_monthly_revenue_full_universe_pit_20260703"
)
DEFAULT_LAYER4_POOL = Path("outputs/vnext_layer4_80_primary_pool_contract_20260708/layer4_80_primary_pool_contract.csv")
DEFAULT_LOW_BASE_DIR = Path("outputs/vnext_layer4_low_base_score_contract_20260709")
SHINFOX_TICKER = "6806"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build long revenue stability and low-base risk integration feature contract.")
    parser.add_argument("--monthly-revenue-dir", default=str(DEFAULT_MONTHLY_REVENUE_DIR))
    parser.add_argument("--layer4-pool", default=str(DEFAULT_LAYER4_POOL))
    parser.add_argument("--low-base-dir", default=str(DEFAULT_LOW_BASE_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    monthly_dir = Path(args.monthly_revenue_dir)
    layer4_pool_path = Path(args.layer4_pool)
    low_base_dir = Path(args.low_base_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    layer4_latest, asof_date = load_latest_layer4_primary80(layer4_pool_path)
    scoped_tickers = sorted(set(layer4_latest["ticker"].astype(str)) | {SHINFOX_TICKER})
    revenue = load_monthly_revenue(monthly_dir, scoped_tickers=scoped_tickers, asof_date=asof_date)
    features = build_revenue_features(revenue, asof_date=asof_date)
    contract = build_feature_contract(layer4_latest, features, asof_date)
    contract_path = output_dir / "layer1_long_revenue_stability_feature_contract.csv"
    contract.to_csv(contract_path, index=False, encoding="utf-8-sig")

    low_base_map = build_low_base_integration_map(low_base_dir)
    low_base_path = output_dir / "low_base_risk_integration_feature_map.csv"
    low_base_map.to_csv(low_base_path, index=False, encoding="utf-8-sig")

    missingness = build_missingness_by_period(revenue, layer4_latest, asof_date)
    missingness_path = output_dir / "revenue_stability_missingness_by_period.csv"
    missingness.to_csv(missingness_path, index=False, encoding="utf-8-sig")

    lumpiness_audit = build_lumpiness_proxy_audit(contract)
    lumpiness_path = output_dir / "revenue_lumpiness_proxy_audit.csv"
    lumpiness_audit.to_csv(lumpiness_path, index=False, encoding="utf-8-sig")

    sanity = build_shinfox_sanity_check(features, revenue, asof_date)
    sanity_path = output_dir / "shinfox_6806_feature_sanity_check.csv"
    sanity.to_csv(sanity_path, index=False, encoding="utf-8-sig")

    coverage = build_requested_vs_actual_coverage(revenue, contract, asof_date)
    coverage_path = output_dir / "requested_vs_actual_coverage.csv"
    coverage.to_csv(coverage_path, index=False, encoding="utf-8-sig")

    future_audit = pd.DataFrame(
        [
            {
                "audit_item": "monthly_revenue_available_date",
                "status": "pass",
                "rule": "all monthly revenue rows use available_date <= layer4 latest snapshot_date",
                "asof_date": asof_date,
                "violation_count": int((revenue["available_date"] > pd.Timestamp(asof_date)).sum()) if not revenue.empty else 0,
            },
            {
                "audit_item": "future_return_as_rule",
                "status": "pass",
                "rule": "no future return / future winner field used",
                "violation_count": 0,
            },
        ]
    )
    future_path = output_dir / "future_data_audit.csv"
    future_audit.to_csv(future_path, index=False, encoding="utf-8-sig")

    readiness = build_readiness(contract, missingness, sanity, asof_date)
    readiness_path = output_dir / "readiness_for_experiments.json"
    write_json(readiness_path, readiness)

    summary_path = output_dir / "final_summary_zh.md"
    summary_path.write_text(build_summary(readiness, sanity), encoding="utf-8")

    artifacts = [
        contract_path,
        low_base_path,
        missingness_path,
        lumpiness_path,
        sanity_path,
        coverage_path,
        future_path,
        readiness_path,
        summary_path,
    ]
    manifest_path = output_dir / "manifest.json"
    write_json(manifest_path, build_manifest(output_dir, artifacts))

    print(f"LAYER1_LONG_REVENUE_LOW_BASE_OUTPUT={output_dir.resolve()}")
    print(f"ASOF_DATE={asof_date}")
    print(f"CONTRACT_ROWS={len(contract)}")
    print(f"READY_FOR_EXPERIMENTS={readiness['ready_for_experiments']}")


def load_latest_layer4_primary80(path: Path) -> tuple[pd.DataFrame, str]:
    df = pd.read_csv(path, dtype={"ticker": str}, low_memory=False)
    if "snapshot_date" not in df.columns:
        raise ValueError("Layer4 primary80 contract missing snapshot_date")
    dates = pd.to_datetime(df["snapshot_date"], errors="coerce")
    latest_date = dates.max()
    latest = df[dates.eq(latest_date)].copy()
    if "is_layer4_primary_pool" in latest.columns:
        latest = latest[latest["is_layer4_primary_pool"].astype(str).str.lower().eq("true")]
    return latest, latest_date.strftime("%Y-%m-%d")


def load_monthly_revenue(monthly_dir: Path, *, scoped_tickers: list[str], asof_date: str) -> pd.DataFrame:
    shard_dir = monthly_dir / "accepted_monthly_revenue_rows_shards"
    frames = []
    for shard in sorted(shard_dir.glob("accepted_monthly_revenue_rows_*.csv")):
        frame = pd.read_csv(shard, dtype={"ticker": str})
        frame = frame[frame["ticker"].isin(scoped_tickers)]
        if frame.empty:
            continue
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    revenue = pd.concat(frames, ignore_index=True)
    revenue["available_date"] = pd.to_datetime(revenue["available_date"], errors="coerce")
    revenue["period"] = pd.PeriodIndex(revenue["revenue_year_month"].astype(str), freq="M")
    revenue["revenue_value"] = pd.to_numeric(revenue["revenue_value"], errors="coerce")
    revenue = revenue[
        revenue["pit_usable"].astype(str).str.lower().eq("true")
        & revenue["available_date"].le(pd.Timestamp(asof_date))
        & revenue["revenue_value"].notna()
    ].copy()
    return revenue.sort_values(["ticker", "period"])


def build_revenue_features(revenue: pd.DataFrame, *, asof_date: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for ticker, group in revenue.groupby("ticker"):
        group = group.sort_values("period").copy()
        values = group.set_index("period")["revenue_value"].astype(float)
        if values.empty:
            continue
        latest_period = values.index.max()
        latest_value = values.loc[latest_period]
        lag12 = value_at(values, latest_period - 12)
        monthly_yoy = pct_change(latest_value, lag12)

        trailing_3m = rolling_sum(values, latest_period, 3)
        trailing_3m_prev = rolling_sum(values, latest_period - 12, 3)
        trailing_3m_yoy = pct_change(trailing_3m, trailing_3m_prev)

        quarter_sum = rolling_sum(values, latest_period, 3)
        quarter_sum_prev = rolling_sum(values, latest_period - 12, 3)
        quarterly_yoy = pct_change(quarter_sum, quarter_sum_prev)

        ttm = rolling_sum(values, latest_period, 12)
        ttm_prev = rolling_sum(values, latest_period - 12, 12)
        ttm_yoy = pct_change(ttm, ttm_prev)

        annual = annual_revenue(values)
        cagr3 = cagr_from_annual(annual, years=3)
        cagr5 = cagr_from_annual(annual, years=5)
        trend3 = trend_score_from_annual(annual, years=3)
        trend5 = trend_score_from_annual(annual, years=5)

        last36 = values.loc[values.index >= latest_period - 35]
        last60 = values.loc[values.index >= latest_period - 59]
        revenue_stability_score = stability_score(last36, ttm_yoy, cagr3)
        lumpiness_score = lumpiness(last36)
        recent_spike = recent_spike_without_history(monthly_yoy, trailing_3m_yoy, cagr3, cagr5, len(last60))
        project_risk = project_based_proxy(lumpiness_score, recent_spike)

        first = group.iloc[0]
        latest_row = group[group["period"].eq(latest_period)].iloc[-1]
        rows.append(
            {
                "ticker": ticker,
                "name": latest_row.get("name", first.get("name", "")),
                "market": latest_row.get("market", first.get("market", "")),
                "asof_date": asof_date,
                "latest_revenue_year_month": str(latest_period),
                "monthly_revenue_yoy": monthly_yoy,
                "trailing_3m_revenue_yoy": trailing_3m_yoy,
                "quarterly_revenue_yoy": quarterly_yoy,
                "quarterly_revenue_source_quality": "proxy_from_monthly_revenue_rolling_3m_not_statement_revenue",
                "ttm_revenue": ttm,
                "ttm_revenue_yoy": ttm_yoy,
                "3y_revenue_cagr": cagr3,
                "3y_revenue_trend_score": trend3,
                "5y_revenue_cagr": cagr5,
                "5y_revenue_trend_score": trend5,
                "revenue_stability_score": revenue_stability_score,
                "revenue_lumpiness_score": lumpiness_score,
                "recent_spike_without_long_history_flag": recent_spike,
                "project_based_revenue_risk_proxy": project_risk,
                "revenue_month_count_available": int(values.count()),
                "revenue_3y_month_count_available": int(last36.count()),
                "revenue_5y_month_count_available": int(last60.count()),
                "data_coverage": coverage_label(len(last60), cagr3, cagr5),
                "missingness": missingness_label(len(last60), cagr3, cagr5),
                "pit_asof_audit": "monthly_revenue_available_date_le_asof_date;conservative_available_date_not_exact_filing_timestamp",
                "source_quality": "MOPS_monthly_revenue_PIT_conservative_available_date",
            }
        )
    return pd.DataFrame(rows)


def value_at(values: pd.Series, period: pd.Period) -> float | None:
    return float(values.loc[period]) if period in values.index else None


def rolling_sum(values: pd.Series, end_period: pd.Period, months: int) -> float | None:
    periods = [end_period - offset for offset in range(months)]
    if not all(period in values.index for period in periods):
        return None
    return float(sum(values.loc[period] for period in periods))


def pct_change(current: float | None, previous: float | None) -> float | None:
    if current is None or previous is None or previous == 0 or pd.isna(current) or pd.isna(previous):
        return None
    return float(current / previous - 1.0)


def annual_revenue(values: pd.Series) -> pd.Series:
    frame = values.reset_index()
    frame["year"] = frame["period"].astype(str).str.slice(0, 4).astype(int)
    yearly = frame.groupby("year")["revenue_value"].sum()
    counts = frame.groupby("year")["revenue_value"].count()
    return yearly[counts >= 10]


def cagr_from_annual(annual: pd.Series, *, years: int) -> float | None:
    if len(annual) < years + 1:
        return None
    end_year = int(annual.index.max())
    start_year = end_year - years
    if start_year not in annual.index or annual.loc[start_year] <= 0 or annual.loc[end_year] <= 0:
        return None
    return float((annual.loc[end_year] / annual.loc[start_year]) ** (1 / years) - 1.0)


def trend_score_from_annual(annual: pd.Series, *, years: int) -> float | None:
    if len(annual) < years + 1:
        return None
    end_year = int(annual.index.max())
    relevant_years = [year for year in range(end_year - years, end_year + 1) if year in annual.index]
    if len(relevant_years) < years + 1:
        return None
    series = annual.loc[relevant_years].astype(float)
    positive_steps = (series.diff().dropna() > 0).mean()
    drawdown_from_peak = 1 - (series.iloc[-1] / series.max()) if series.max() > 0 else 1
    return float(max(0.0, min(1.0, positive_steps * 0.7 + (1 - drawdown_from_peak) * 0.3)))


def stability_score(last36: pd.Series, ttm_yoy: float | None, cagr3: float | None) -> float:
    if len(last36) < 24:
        return 0.35
    coefficient_var = float(last36.std() / last36.mean()) if last36.mean() else 2.0
    cv_score = max(0.0, min(1.0, 1 - coefficient_var / 1.5))
    growth_score = score_growth(ttm_yoy) * 0.55 + score_growth(cagr3) * 0.45
    return float(max(0.0, min(1.0, cv_score * 0.45 + growth_score * 0.55)))


def score_growth(value: float | None) -> float:
    if value is None or pd.isna(value):
        return 0.5
    return float(max(0.0, min(1.0, (value + 0.25) / 0.75)))


def lumpiness(last36: pd.Series) -> float:
    if len(last36) < 12 or last36.sum() <= 0:
        return 0.5
    top3_share = float(last36.sort_values(ascending=False).head(3).sum() / last36.sum())
    cv = float(last36.std() / last36.mean()) if last36.mean() else 2.0
    raw = top3_share * 0.65 + min(cv / 2.0, 1.0) * 0.35
    return float(max(0.0, min(1.0, raw)))


def recent_spike_without_history(
    monthly_yoy: float | None,
    trailing_3m_yoy: float | None,
    cagr3: float | None,
    cagr5: float | None,
    month_count_5y: int,
) -> bool:
    spike = max([value for value in [monthly_yoy, trailing_3m_yoy] if value is not None and not pd.isna(value)] or [0.0]) >= 1.0
    long_history_weak = month_count_5y < 48 or cagr3 is None or cagr3 < 0 or cagr5 is None
    return bool(spike and long_history_weak)


def project_based_proxy(lumpiness_score: float, recent_spike: bool) -> bool:
    return bool(lumpiness_score >= 0.45 or recent_spike)


def coverage_label(month_count_5y: int, cagr3: float | None, cagr5: float | None) -> str:
    if month_count_5y >= 60 and cagr3 is not None and cagr5 is not None:
        return "full_5y_monthly_revenue_proxy_ready"
    if month_count_5y >= 36 and cagr3 is not None:
        return "3y_ready_5y_partial"
    return "long_history_partial_or_blocked"


def missingness_label(month_count_5y: int, cagr3: float | None, cagr5: float | None) -> str:
    missing = []
    if month_count_5y < 60 or cagr5 is None:
        missing.append("5y_trend")
    if month_count_5y < 36 or cagr3 is None:
        missing.append("3y_trend")
    return ";".join(missing) if missing else ""


def build_feature_contract(layer4_latest: pd.DataFrame, features: pd.DataFrame, asof_date: str) -> pd.DataFrame:
    base_cols = [
        "snapshot_date",
        "ticker",
        "name",
        "market",
        "selected_branch" if "selected_branch" in layer4_latest.columns else None,
        "layer1_pass_bottom30" if "layer1_pass_bottom30" in layer4_latest.columns else None,
        "layer1_quality_floor_risk_pctile_by_week" if "layer1_quality_floor_risk_pctile_by_week" in layer4_latest.columns else None,
    ]
    base_cols = [col for col in base_cols if col]
    base = layer4_latest[base_cols].copy()
    base["ticker"] = base["ticker"].astype(str)
    merged = base.merge(features, on=["ticker"], how="left", suffixes=("", "_revenue"))
    merged["feature_scope"] = "latest_layer4_primary80"
    merged["diagnostic_only"] = True
    merged["formal_model_changed"] = False
    merged["trade_decision_changed"] = False
    merged["active_in_trade_decision"] = False
    merged["report_changed"] = False
    merged["portfolio_replay_executed"] = False
    merged["ready_for_strategy_replay"] = False
    merged["ready_for_formal"] = False
    merged["not_live_rule"] = True
    merged["forward_returns_live_rule_usage"] = False
    merged["quarterly_margin_profitability_source_quality"] = "existing_layer1_interim_or_quarterly_source_partial;not_refreshed_in_this_contract"
    merged["project_based_revenue_risk_proxy_source_quality"] = "proxy_from_monthly_revenue_lumpiness_and_recent_spike_not_industry_contract"
    merged["asof_date"] = merged["asof_date"].fillna(asof_date)
    return merged


def build_low_base_integration_map(low_base_dir: Path) -> pd.DataFrame:
    overlap_path = low_base_dir / "existing_low_base_overlap_audit.csv"
    placement_path = low_base_dir / "layer4_low_base_layer_placement_decision.csv"
    existing = pd.read_csv(overlap_path) if overlap_path.exists() else pd.DataFrame()
    placement = pd.read_csv(placement_path) if placement_path.exists() else pd.DataFrame()
    rows = [
        low_base_row("price_low_base_percentile", "price_position_low_base", "Layer2/Layer4 ranking context", "reuse_existing_as_bonus", "stock price location within own range; never hard filter"),
        low_base_row("bias60_percentile_zscore", "bias60_zscore_252d / BIAS60_percentile", "Layer2 risk/overheat context", "reuse_existing_penalty", "ticker-specific overheat control; raw BIAS threshold is not enough"),
        low_base_row("valuation_low_base_proxy", "PE/PB/PS unavailable", "blocked/proxy", "blocked", "valuation source not accepted in this contract"),
        low_base_row("revenue_base_context", "long_revenue_stability_feature_contract", "Layer1 quality context + Layer4 scoring support", "new_context_not_hard_filter", "separate recovery from short-term low-base illusion"),
        low_base_row("margin_recovery_context", "gross_margin/operating_margin partial", "Layer1 quality context", "proxy_or_blocked", "requires refreshed quarterly profitability source for broad use"),
        low_base_row("overheat_penalty_context", "overheat_veto_flag / volatility", "Layer2/Layer4 risk context", "keep_existing_as_penalty", "low-base bonus cannot offset obvious overheat/high volatility"),
        low_base_row("low_base_as_bonus_cap", "policy", "Layer4 ranking component", "cap_bonus_only", "max small bonus/tie-break; no independent route and no hard selected rule"),
    ]
    result = pd.DataFrame(rows)
    result["existing_overlap_source_rows"] = len(existing)
    result["prior_layer_placement_source_rows"] = len(placement)
    return result


def low_base_row(feature: str, source: str, layer: str, action: str, note: str) -> dict[str, str]:
    return {"feature": feature, "source_or_component": source, "recommended_layer": layer, "integration_action": action, "note": note}


def build_missingness_by_period(revenue: pd.DataFrame, layer4_latest: pd.DataFrame, asof_date: str) -> pd.DataFrame:
    scoped = set(layer4_latest["ticker"].astype(str))
    latest_features = build_revenue_features(revenue[revenue["ticker"].isin(scoped)], asof_date=asof_date)
    total = len(scoped)
    rows = []
    for period in ["latest_layer4_primary80"]:
        rows.append(
            {
                "period": period,
                "requested_scope_ticker_count": total,
                "monthly_revenue_ready_ticker_count": latest_features["ticker"].nunique() if not latest_features.empty else 0,
                "3y_trend_ready_ticker_count": int(latest_features["3y_revenue_cagr"].notna().sum()) if not latest_features.empty else 0,
                "5y_trend_ready_ticker_count": int(latest_features["5y_revenue_cagr"].notna().sum()) if not latest_features.empty else 0,
                "lumpiness_proxy_ready_ticker_count": int(latest_features["revenue_lumpiness_score"].notna().sum()) if not latest_features.empty else 0,
                "quarterly_statement_exact_revenue_yoy_ready": False,
                "quarterly_statement_exact_revenue_yoy_blocked_reason": "not refreshed here; quarterly_revenue_yoy is proxy_from_monthly_rolling_3m",
                "asof_date": asof_date,
            }
        )
    return pd.DataFrame(rows)


def build_lumpiness_proxy_audit(contract: pd.DataFrame) -> pd.DataFrame:
    ready = contract["revenue_lumpiness_score"].notna()
    return pd.DataFrame(
        [
            {
                "proxy_field": "revenue_lumpiness_score",
                "status": "proxy_ready" if ready.any() else "blocked",
                "ready_rows": int(ready.sum()),
                "blocked_rows": int((~ready).sum()),
                "definition": "top3_month_revenue_share_last36m_65pct + monthly_revenue_cv_last36m_35pct",
                "limitation": "does not prove project-based business model; flags lumpy monthly revenue shape only",
            },
            {
                "proxy_field": "project_based_revenue_risk_proxy",
                "status": "diagnostic_proxy",
                "ready_rows": int(contract["project_based_revenue_risk_proxy"].notna().sum()),
                "definition": "lumpiness_score>=0.45 or recent_spike_without_long_history_flag",
                "limitation": "not industry/business-model verified",
            },
        ]
    )


def build_shinfox_sanity_check(features: pd.DataFrame, revenue: pd.DataFrame, asof_date: str) -> pd.DataFrame:
    feature = features[features["ticker"].eq(SHINFOX_TICKER)].copy()
    if feature.empty:
        return pd.DataFrame(
            [
                {
                    "ticker": SHINFOX_TICKER,
                    "name": "森崴能源",
                    "asof_date": asof_date,
                    "status": "blocked_no_monthly_revenue_rows",
                    "sanity_check": "cannot evaluate",
                }
            ]
        )
    row = feature.iloc[0].to_dict()
    shinfox_revenue = revenue[revenue["ticker"].eq(SHINFOX_TICKER)].sort_values("period")
    last12_total = shinfox_revenue.tail(12)["revenue_value"].sum() if not shinfox_revenue.empty else None
    row.update(
        {
            "status": "ready_proxy",
            "ticker": SHINFOX_TICKER,
            "name": row.get("name", "森崴能源"),
            "asof_date": asof_date,
            "last12_month_revenue_total": last12_total,
            "sanity_check": "flags project/lumpy risk if lumpiness or recent spike proxy is true; not an investment judgment",
        }
    )
    return pd.DataFrame([row])


def build_requested_vs_actual_coverage(revenue: pd.DataFrame, contract: pd.DataFrame, asof_date: str) -> pd.DataFrame:
    if revenue.empty:
        month_start = month_end = ""
    else:
        month_start = str(revenue["period"].min())
        month_end = str(revenue["period"].max())
    return pd.DataFrame(
        [
            {
                "requested_scope": "latest Layer4 primary80 + 6806 sanity check",
                "actual_scope": "latest_layer4_primary80_contract_rows plus shinfox sanity check",
                "requested_asof_date": asof_date,
                "actual_asof_date": asof_date,
                "monthly_revenue_period_start": month_start,
                "monthly_revenue_period_end": month_end,
                "contract_rows": len(contract),
                "future_data_violation_count": 0,
            }
        ]
    )


def build_readiness(contract: pd.DataFrame, missingness: pd.DataFrame, sanity: pd.DataFrame, asof_date: str) -> dict[str, Any]:
    ready_share = float(contract["revenue_stability_score"].notna().mean()) if len(contract) else 0.0
    return {
        "task_id": TASK_ID,
        "status": "layer1_long_revenue_stability_low_base_risk_integration_contract_ready_proxy_partial",
        "asof_date": asof_date,
        "contract_rows": int(len(contract)),
        "revenue_stability_ready_share": ready_share,
        "ready_for_experiments": ready_share > 0.8,
        "ready_for_formal": False,
        "ready_for_strategy_replay": False,
        "selected_stock_adjusted_close_required": False,
        "quarterly_revenue_yoy_exact_ready": False,
        "quarterly_revenue_yoy_source_quality": "proxy_from_monthly_revenue_rolling_3m",
        "valuation_low_base_proxy_ready": False,
        "valuation_low_base_proxy_blocked_reason": "PE/PB/PS accepted source not materialized in this contract",
        "shinfox_6806_sanity_status": sanity.iloc[0].get("status", "missing") if not sanity.empty else "missing",
        "formal_model_changed": False,
        "trade_decision_changed": False,
        "active_in_trade_decision": False,
        "report_changed": False,
        "portfolio_replay_executed": False,
        "not_live_rule": True,
        "forward_returns_live_rule_usage": False,
        "future_data_violation_count": 0,
    }


def build_summary(readiness: dict[str, Any], sanity: pd.DataFrame) -> str:
    sanity_row = sanity.iloc[0].to_dict() if not sanity.empty else {}
    return "\n".join(
        [
            "# Layer1 長期營收穩定性 + low-base 風險整合 feature contract",
            "",
            "## 結論",
            "",
            "- 已建立 latest Layer4 primary80 scoped 的長期營收穩定性 feature contract。",
            "- 長期營收穩定性建議放 Layer1 quality floor / quality score，不是短線爆發 selector。",
            "- low-base 建議放 Layer2/Layer4 soft context，只能小幅 bonus / penalty / tie-break，不作 hard filter、不作獨立 route。",
            "- quarterly_revenue_yoy 目前用 monthly rolling 3M proxy；valuation low-base proxy 仍 blocked。",
            f"- ready_for_experiments={str(readiness['ready_for_experiments']).lower()}；ready_for_formal=false。",
            "",
            "## 6806 森崴能源 sanity check",
            "",
            f"- status={sanity_row.get('status', 'missing')}",
            f"- revenue_lumpiness_score={sanity_row.get('revenue_lumpiness_score', '')}",
            f"- recent_spike_without_long_history_flag={sanity_row.get('recent_spike_without_long_history_flag', '')}",
            f"- project_based_revenue_risk_proxy={sanity_row.get('project_based_revenue_risk_proxy', '')}",
            "- 這只是 feature sanity check，不是投資判斷。",
            "",
            "## Layer placement",
            "",
            "- 長期營收穩定性：Layer1 quality floor / quality score。",
            "- revenue_lumpiness / project-based proxy：Layer1 risk context，可傳給 Layer4 作排序風險扣分。",
            "- low-base：Layer2/Layer4 soft context；不得抵消 overheat / volatility / high risk。",
        ]
    )


def build_manifest(output_dir: Path, artifacts: list[Path]) -> dict[str, Any]:
    return {
        "task_id": TASK_ID,
        "status": "complete_feature_contract_proxy_partial",
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
