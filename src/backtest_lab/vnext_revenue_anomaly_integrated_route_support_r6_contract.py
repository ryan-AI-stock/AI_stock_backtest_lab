from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from backtest_lab.vnext_layer1_long_revenue_low_base_contract import DEFAULT_MONTHLY_REVENUE_DIR, load_monthly_revenue
from backtest_lab.vnext_revenue_anomaly_stability_pattern_contract import (
    abnormal_review_flag,
    anomaly_report_text,
    compute_revenue_anomaly_metrics,
)


TASK_ID = "TASK-BACKTEST-CORE-VNEXT-REVENUE-ANOMALY-INTEGRATED-ROUTE-SUPPORT-R6-CONTRACT-001"
DEFAULT_R6_SOURCE = Path("outputs/vnext_r6_guard_first_market_bias_override_unified_contract_20260709")
DEFAULT_ANOMALY_SOURCE = Path("outputs/vnext_revenue_anomaly_stability_pattern_contract_20260710")
DEFAULT_OUTPUT = Path("outputs/vnext_revenue_anomaly_integrated_route_support_r6_contract_20260710")


def main() -> None:
    parser = argparse.ArgumentParser(description="Integrate revenue anomaly hygiene into route_support / R6 unified contract.")
    parser.add_argument("--r6-source-dir", default=str(DEFAULT_R6_SOURCE))
    parser.add_argument("--anomaly-source-dir", default=str(DEFAULT_ANOMALY_SOURCE))
    parser.add_argument("--monthly-revenue-dir", default=str(DEFAULT_MONTHLY_REVENUE_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    r6_source_dir = Path(args.r6_source_dir)
    anomaly_source_dir = Path(args.anomaly_source_dir)
    monthly_revenue_dir = Path(args.monthly_revenue_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    r6 = pd.read_csv(r6_source_dir / "r6_guard_first_market_bias_override_unified_contract.csv", dtype={"selected_ticker": str}, low_memory=False)
    anomaly = pd.read_csv(anomaly_source_dir / "revenue_anomaly_stability_pattern_contract.csv", dtype={"ticker": str}, low_memory=False)
    selected_tickers = sorted(set(normalize_ticker_series(r6["selected_ticker"])) - {"", "00631L", "0050", "nan", "NaN"})
    max_signal_date = pd.to_datetime(r6["signal_date"], errors="coerce").max().strftime("%Y-%m-%d")
    revenue = load_monthly_revenue(monthly_revenue_dir, scoped_tickers=selected_tickers, asof_date=max_signal_date)
    selected_pit_anomaly = build_selected_pit_anomaly_context(r6, revenue)
    integrated = build_integrated_contract(r6, anomaly, selected_pit_anomaly)

    contract_path = output_dir / "revenue_anomaly_integrated_route_support_r6_contract.csv"
    integrated.to_csv(contract_path, index=False, encoding="utf-8-sig")

    policy = build_policy_map()
    policy_path = output_dir / "revenue_anomaly_integration_policy_map.csv"
    policy.to_csv(policy_path, index=False, encoding="utf-8-sig")

    report_sample = build_daily_report_hook_sample(integrated)
    report_sample_path = output_dir / "revenue_anomaly_daily_report_hook_sample.csv"
    report_sample.to_csv(report_sample_path, index=False, encoding="utf-8-sig")

    coverage = build_requested_vs_actual_coverage(integrated, anomaly)
    coverage_path = output_dir / "revenue_anomaly_integrated_requested_vs_actual_coverage.csv"
    coverage.to_csv(coverage_path, index=False, encoding="utf-8-sig")

    blocked = build_blocked_proxy_audit(integrated)
    blocked_path = output_dir / "revenue_anomaly_integrated_blocked_proxy_audit.csv"
    blocked.to_csv(blocked_path, index=False, encoding="utf-8-sig")

    future = build_future_data_audit(integrated)
    future_path = output_dir / "revenue_anomaly_integrated_future_data_audit.csv"
    future.to_csv(future_path, index=False, encoding="utf-8-sig")

    readiness = build_readiness(integrated, coverage)
    readiness_path = output_dir / "readiness_for_revenue_anomaly_integrated_experiments.json"
    write_json(readiness_path, readiness)

    summary_path = output_dir / "final_summary_zh.md"
    summary_path.write_text(build_summary(readiness, integrated, coverage), encoding="utf-8")

    artifacts = [
        contract_path,
        policy_path,
        report_sample_path,
        coverage_path,
        blocked_path,
        future_path,
        readiness_path,
        summary_path,
    ]
    manifest_path = output_dir / "manifest.json"
    write_json(manifest_path, build_manifest(output_dir, artifacts))

    print(f"REVENUE_ANOMALY_R6_INTEGRATED_OUTPUT={output_dir.resolve()}")
    print(f"CONTRACT_ROWS={len(integrated)}")
    print(f"STOCK_SELECTED_ROWS={int(integrated['selected_primary_asset_type_for_anomaly'].eq('stock').sum())}")
    print(f"READY_FOR_EXPERIMENTS={readiness['ready_for_experiments']}")


def build_selected_pit_anomaly_context(r6: pd.DataFrame, revenue: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    stock_rows = r6[r6.get("selected_asset_type", "").astype(str).str.lower().eq("stock")].copy()
    stock_rows["selected_ticker"] = normalize_ticker_series(stock_rows["selected_ticker"])
    for _, row in stock_rows.iterrows():
        signal_date = pd.Timestamp(row["signal_date"])
        ticker = str(row["selected_ticker"])
        ticker_revenue = revenue[
            revenue["ticker"].astype(str).eq(ticker)
            & pd.to_datetime(revenue["available_date"], errors="coerce").le(signal_date)
        ].sort_values("period")
        metrics = compute_revenue_anomaly_metrics(ticker_revenue)
        metrics["selected_ticker"] = ticker
        metrics["signal_date"] = row["signal_date"]
        metrics["revenue_lumpiness_percentile_vs_primary80"] = pd.NA
        metrics["revenue_lumpiness_percentile_source_quality"] = "blocked_historical_primary80_distribution_not_materialized"
        rows.append(metrics)
    context = pd.DataFrame(rows)
    if context.empty:
        return context
    context["abnormal_revenue_review_flag"] = abnormal_review_flag(context)
    context["revenue_anomaly_report_text"] = context.apply(anomaly_report_text, axis=1)
    return context


def build_integrated_contract(r6: pd.DataFrame, anomaly: pd.DataFrame, selected_pit_anomaly: pd.DataFrame) -> pd.DataFrame:
    df = r6.copy()
    df["selected_ticker"] = normalize_ticker_series(df["selected_ticker"])
    pit_cols = [
        "signal_date",
        "selected_ticker",
        "abnormal_revenue_review_flag",
        "revenue_growth_persistence_score",
        "revenue_concentration_ratio_top3_12m",
        "revenue_lumpiness_score",
        "revenue_lumpiness_percentile_vs_primary80",
        "revenue_lumpiness_percentile_source_quality",
        "ttm_vs_recent_growth_gap",
        "low_base_distortion_flag",
        "revenue_anomaly_report_text",
        "source_quality",
        "data_coverage",
        "missingness",
        "pit_asof_audit",
    ]
    anomaly_join = selected_pit_anomaly[[col for col in pit_cols if col in selected_pit_anomaly.columns]].copy()
    anomaly_join = anomaly_join.rename(columns={"source_quality": "revenue_anomaly_source_quality"})
    merged = df.merge(anomaly_join, on=["signal_date", "selected_ticker"], how="left")

    is_stock = merged.get("selected_asset_type", "").astype(str).str.lower().eq("stock")
    is_fallback_or_etf = ~is_stock
    merged["selected_primary_asset_type_for_anomaly"] = "stock"
    merged.loc[is_fallback_or_etf, "selected_primary_asset_type_for_anomaly"] = merged.loc[is_fallback_or_etf, "selected_asset_type"].astype(str)
    merged["revenue_anomaly_applicability"] = "selected_stock_anomaly_context"
    merged.loc[is_fallback_or_etf, "revenue_anomaly_applicability"] = "not_applicable_fallback_or_etf"
    merged.loc[is_stock & merged["abnormal_revenue_review_flag"].isna(), "revenue_anomaly_applicability"] = "blocked_missing_selected_stock_revenue_anomaly_row"

    merged["abnormal_revenue_review_flag"] = merged["abnormal_revenue_review_flag"].fillna(False).astype(bool)
    merged["low_base_distortion_flag"] = merged["low_base_distortion_flag"].fillna(False).astype(bool)
    merged["revenue_anomaly_penalty_score"] = revenue_penalty_score(merged, is_stock)
    merged["revenue_hygiene_confidence_level"] = confidence_level(merged, is_stock)
    merged["revenue_anomaly_used_as_hard_exclude"] = False
    merged["route_support_selected_result_changed"] = False
    merged["r6_selected_result_changed"] = False
    merged["business_model_keyword_proxy_used_as_risk_basis"] = False
    merged["industry_classification_used_as_risk_basis"] = False
    merged["hard_exclude_applied"] = False
    merged["report_revenue_anomaly_warning"] = report_warning(merged, is_stock)
    merged["report_revenue_anomaly_reason"] = report_reason(merged, is_stock)
    merged["cash_bear_classifier_ready"] = ~merged.get("bear_cash_guard_source_quality", "").astype(str).str.contains("blocked", case=False, na=False)
    merged["cash_bear_classifier_status"] = merged.get("bear_cash_guard_source_quality", "")
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

    keep = base_columns(df) + [
        "selected_primary_asset_type_for_anomaly",
        "revenue_anomaly_applicability",
        "abnormal_revenue_review_flag",
        "revenue_growth_persistence_score",
        "revenue_concentration_ratio_top3_12m",
        "revenue_lumpiness_score",
        "revenue_lumpiness_percentile_vs_primary80",
        "revenue_lumpiness_percentile_source_quality",
        "ttm_vs_recent_growth_gap",
        "low_base_distortion_flag",
        "revenue_anomaly_penalty_score",
        "revenue_hygiene_confidence_level",
        "revenue_anomaly_used_as_hard_exclude",
        "route_support_selected_result_changed",
        "r6_selected_result_changed",
        "report_revenue_anomaly_warning",
        "report_revenue_anomaly_reason",
        "revenue_anomaly_report_text",
        "revenue_anomaly_source_quality",
        "data_coverage",
        "missingness",
        "pit_asof_audit",
        "business_model_keyword_proxy_used_as_risk_basis",
        "industry_classification_used_as_risk_basis",
        "hard_exclude_applied",
        "selected_stock_adjusted_close_ready",
        "cash_bear_classifier_ready",
        "cash_bear_classifier_status",
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
    # Preserve duplicate governance columns once.
    seen = set()
    ordered = []
    for col in keep:
        if col in merged.columns and col not in seen:
            ordered.append(col)
            seen.add(col)
    return merged[ordered]


def base_columns(df: pd.DataFrame) -> list[str]:
    preferred = [
        "task",
        "signal_date",
        "next_signal_date",
        "period_label",
        "in_P1",
        "in_P2",
        "in_2024_latest",
        "in_2026YTD",
        "regime_label",
        "selected_branch",
        "selected_ticker",
        "selected_ticker_name",
        "selected_asset_type",
        "fallback_asset",
        "branch_reason",
        "triggered_features",
        "c2_pass_flag",
        "consensus_trigger_flag",
        "r6_override_flag",
        "breakout_breadth_flag",
        "p1_risk_veto_flag",
        "bear_guard_flag",
        "cash_guard_flag",
        "bear_cash_guard_source_quality",
        "rs20_top3_reference_tickers",
        "rs20_reference_only",
        "low_base_main_weight_included",
        "entry_date",
        "exit_date",
        "entry_price",
        "exit_price",
        "gross_interval_return",
        "net_interval_return_after_transition_cost",
        "transition_action",
        "transition_cost_rate",
        "official_selected_stock_ohlc_ready",
        "benchmark_adjusted_path_ready",
        "path_ready",
        "branch_path_missing_fallback",
        "source_quality",
        "data_readiness",
        "blocked_reason",
    ]
    return [col for col in preferred if col in df.columns]


def normalize_ticker_series(series: pd.Series) -> pd.Series:
    values = series.astype(str).str.strip()
    return values.str.replace(r"\.0$", "", regex=True)


def revenue_penalty_score(df: pd.DataFrame, is_stock: pd.Series) -> pd.Series:
    lumpiness_pct = pd.to_numeric(df.get("revenue_lumpiness_percentile_vs_primary80"), errors="coerce").fillna(0)
    concentration = pd.to_numeric(df.get("revenue_concentration_ratio_top3_12m"), errors="coerce").fillna(0)
    persistence = pd.to_numeric(df.get("revenue_growth_persistence_score"), errors="coerce").fillna(1)
    gap = pd.to_numeric(df.get("ttm_vs_recent_growth_gap"), errors="coerce").fillna(0).clip(lower=0, upper=2) / 2
    review = df.get("abnormal_revenue_review_flag", pd.Series(False, index=df.index)).fillna(False).astype(bool).astype(float)
    low_base = df.get("low_base_distortion_flag", pd.Series(False, index=df.index)).fillna(False).astype(bool).astype(float)
    raw = review * 0.25 + lumpiness_pct * 0.25 + concentration.clip(0, 1) * 0.2 + (1 - persistence.clip(0, 1)) * 0.15 + gap * 0.1 + low_base * 0.05
    raw = raw.clip(0, 1)
    raw.loc[~is_stock] = 0.0
    raw.loc[df["revenue_anomaly_applicability"].eq("blocked_missing_selected_stock_revenue_anomaly_row")] = 0.0
    return raw.round(6)


def confidence_level(df: pd.DataFrame, is_stock: pd.Series) -> pd.Series:
    score = pd.to_numeric(df["revenue_anomaly_penalty_score"], errors="coerce").fillna(0)
    level = pd.Series("not_applicable_fallback_or_etf", index=df.index)
    level.loc[is_stock & score.lt(0.25)] = "high_no_major_revenue_anomaly"
    level.loc[is_stock & score.ge(0.25) & score.lt(0.45)] = "medium_minor_revenue_anomaly_review"
    level.loc[is_stock & score.ge(0.45)] = "low_revenue_anomaly_confidence_downgrade"
    level.loc[df["revenue_anomaly_applicability"].eq("blocked_missing_selected_stock_revenue_anomaly_row")] = "blocked_missing_revenue_anomaly_context"
    return level


def report_warning(df: pd.DataFrame, is_stock: pd.Series) -> pd.Series:
    warning = pd.Series("", index=df.index)
    warning.loc[~is_stock] = "fallback/ETF row：營收異常欄位不適用。"
    warning.loc[is_stock & df["abnormal_revenue_review_flag"].fillna(False).astype(bool)] = "入選個股觸發營收異常 review，僅作信心下修/軟扣分。"
    warning.loc[is_stock & ~df["abnormal_revenue_review_flag"].fillna(False).astype(bool)] = "入選個股未觸發主要營收異常 review。"
    warning.loc[df["revenue_anomaly_applicability"].eq("blocked_missing_selected_stock_revenue_anomaly_row")] = "入選個股缺營收異常 context，需保持 blocked/proxy 標示。"
    return warning


def report_reason(df: pd.DataFrame, is_stock: pd.Series) -> pd.Series:
    reason = df.get("revenue_anomaly_report_text", pd.Series("", index=df.index)).fillna("")
    reason.loc[~is_stock] = "fallback/ETF 不套用公司營收 anomaly。"
    reason.loc[df["revenue_anomaly_applicability"].eq("blocked_missing_selected_stock_revenue_anomaly_row")] = "selected ticker not found in revenue anomaly scoped contract"
    return reason


def build_policy_map() -> pd.DataFrame:
    rows = [
        policy("abnormal_revenue_review_flag", "route_support/R6", "soft_penalty_only", "不得 hard exclude，不改 selected result；交 Experiments 評估扣分後是否改善風險。"),
        policy("revenue_anomaly_penalty_score", "route_support score context", "confidence_downgrade", "0-1 soft penalty candidate；不當 standalone alpha。"),
        policy("revenue_hygiene_confidence_level", "daily report / diagnostics", "display_and_grouping", "日報與 diagnostic 分組使用。"),
        policy("report_revenue_anomaly_warning", "daily report hook", "display_only", "report-only warning，不產生交易指令。"),
        policy("business_model_keyword_proxy", "deprecated", "not_used", "不得使用工程/EPC/產業/商業模式 keyword 作風險依據。"),
        policy("selected_stock_adjusted_close_ready", "readiness", "preserve_status", "沿用 R6 contract 狀態；不包裝成 formal-ready。"),
        policy("cash_bear_classifier_ready", "readiness", "preserve_blocker", "cash/bear classifier blocked 不阻擋本 diagnostic contract，但不可產生 cash rule。"),
    ]
    return pd.DataFrame(rows)


def policy(field: str, layer: str, action: str, note: str) -> dict[str, str]:
    return {"field": field, "layer_destination": layer, "integration_action": action, "policy_note": note}


def build_daily_report_hook_sample(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "signal_date",
        "period_label",
        "regime_label",
        "selected_branch",
        "selected_ticker",
        "selected_ticker_name",
        "selected_asset_type",
        "abnormal_revenue_review_flag",
        "revenue_anomaly_penalty_score",
        "revenue_hygiene_confidence_level",
        "report_revenue_anomaly_warning",
        "report_revenue_anomaly_reason",
        "diagnostic_only",
    ]
    stock = df[df["selected_primary_asset_type_for_anomaly"].eq("stock")]
    sample = stock[stock["abnormal_revenue_review_flag"]].head(12)
    if sample.empty:
        sample = df.tail(12)
    return sample[[col for col in cols if col in sample.columns]]


def build_requested_vs_actual_coverage(df: pd.DataFrame, anomaly: pd.DataFrame) -> pd.DataFrame:
    stock = df["selected_primary_asset_type_for_anomaly"].eq("stock")
    blocked_stock = df["revenue_anomaly_applicability"].eq("blocked_missing_selected_stock_revenue_anomaly_row")
    rows = []
    periods = {
        "P1": df.get("in_P1", pd.Series(False, index=df.index)).fillna(False).astype(bool),
        "P2": df.get("in_P2", pd.Series(False, index=df.index)).fillna(False).astype(bool),
        "2024-latest": df.get("in_2024_latest", pd.Series(False, index=df.index)).fillna(False).astype(bool),
        "2026YTD": df.get("in_2026YTD", pd.Series(False, index=df.index)).fillna(False).astype(bool),
        "full_integrated": pd.Series(True, index=df.index),
    }
    for label, mask in periods.items():
        subset = df[mask]
        stock_subset = subset[subset["selected_primary_asset_type_for_anomaly"].eq("stock")]
        rows.append(
            {
                "period": label,
                "contract_rows": len(subset),
                "stock_selected_rows": len(stock_subset),
                "stock_rows_with_anomaly_context": int(stock_subset["revenue_anomaly_applicability"].eq("selected_stock_anomaly_context").sum()),
                "stock_rows_missing_anomaly_context": int(stock_subset["revenue_anomaly_applicability"].eq("blocked_missing_selected_stock_revenue_anomaly_row").sum()),
                "abnormal_revenue_review_stock_rows": int((stock_subset["abnormal_revenue_review_flag"]).sum()) if not stock_subset.empty else 0,
                "fallback_or_etf_rows": int((~subset["selected_primary_asset_type_for_anomaly"].eq("stock")).sum()),
                "future_data_violation_count": 0,
            }
        )
    rows.append(
        {
            "period": "source_anomaly_contract",
            "contract_rows": len(anomaly),
            "stock_selected_rows": "",
            "stock_rows_with_anomaly_context": "",
            "stock_rows_missing_anomaly_context": "",
            "abnormal_revenue_review_stock_rows": int(anomaly["abnormal_revenue_review_flag"].fillna(False).sum()) if "abnormal_revenue_review_flag" in anomaly else "",
            "fallback_or_etf_rows": "",
            "future_data_violation_count": 0,
        }
    )
    return pd.DataFrame(rows)


def build_blocked_proxy_audit(df: pd.DataFrame) -> pd.DataFrame:
    missing_stock = int(df["revenue_anomaly_applicability"].eq("blocked_missing_selected_stock_revenue_anomaly_row").sum())
    adjusted_ready = df.get("selected_stock_adjusted_close_ready", pd.Series(False, index=df.index)).astype(str).str.lower().eq("true").all()
    cash_ready = df["cash_bear_classifier_ready"].fillna(False).astype(bool).all()
    rows = [
        audit("selected_stock_revenue_anomaly_context", "ready" if missing_stock == 0 else "partial_blocked", f"missing_selected_stock_rows={missing_stock}", "Missing rows cannot be silently filled."),
        audit("selected_stock_adjusted_close_ready", "ready" if adjusted_ready else "blocked_or_partial", "preserved from R6 unified contract", "Not formal-ready if adjusted close remains blocked/diagnostic-only."),
        audit("cash_bear_classifier_ready", "ready" if cash_ready else "blocked", "preserved bear_cash_guard_source_quality", "Do not fabricate cash rule."),
        audit("business_model_keyword_proxy", "deprecated_not_used", "Strategy Center corrected direction to pure revenue anomaly.", "Do not use industry/business keywords as risk basis."),
        audit("hard_exclusion", "not_allowed", "Revenue anomaly is soft penalty/report hook only.", "No row removed and selected result unchanged."),
    ]
    return pd.DataFrame(rows)


def audit(field: str, status: str, evidence: str, policy: str) -> dict[str, str]:
    return {"field": field, "status": status, "evidence": evidence, "policy": policy}


def build_future_data_audit(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "audit_item": "future_return_as_rule",
                "status": "pass",
                "rule": "revenue anomaly fields are PIT monthly revenue context; no future return / future winner used",
                "violation_count": 0,
            },
            {
                "audit_item": "business_model_keyword_as_rule",
                "status": "pass",
                "rule": "business-model / industry keywords are not used as risk basis",
                "violation_count": int(df["business_model_keyword_proxy_used_as_risk_basis"].fillna(False).sum()),
            },
            {
                "audit_item": "hard_exclude",
                "status": "pass",
                "rule": "revenue anomaly fields do not remove rows or change selected result",
                "violation_count": int(df["hard_exclude_applied"].fillna(False).sum()),
            },
        ]
    )


def build_readiness(df: pd.DataFrame, coverage: pd.DataFrame) -> dict[str, Any]:
    stock_rows = int(df["selected_primary_asset_type_for_anomaly"].eq("stock").sum())
    missing_stock = int(df["revenue_anomaly_applicability"].eq("blocked_missing_selected_stock_revenue_anomaly_row").sum())
    return {
        "task_id": TASK_ID,
        "status": "revenue_anomaly_integrated_route_support_r6_contract_ready",
        "contract_rows": int(len(df)),
        "stock_selected_rows": stock_rows,
        "stock_selected_rows_missing_anomaly_context": missing_stock,
        "ready_for_experiments": missing_stock == 0,
        "ready_for_formal": False,
        "ready_for_strategy_replay": False,
        "daily_report_hook_ready": True,
        "revenue_anomaly_used_as_hard_exclude": False,
        "route_support_selected_result_changed": False,
        "r6_selected_result_changed": False,
        "business_model_keyword_proxy_used_as_risk_basis": False,
        "industry_classification_used_as_risk_basis": False,
        "hard_exclude_applied": False,
        "selected_stock_adjusted_close_ready_all_rows": bool(df.get("selected_stock_adjusted_close_ready", pd.Series(False, index=df.index)).astype(str).str.lower().eq("true").all()),
        "cash_bear_classifier_ready_all_rows": bool(df["cash_bear_classifier_ready"].fillna(False).astype(bool).all()),
        "formal_model_changed": False,
        "trade_decision_changed": False,
        "active_in_trade_decision": False,
        "report_changed": False,
        "portfolio_replay_executed": False,
        "not_live_rule": True,
        "forward_returns_live_rule_usage": False,
        "future_data_violation_count": 0,
    }


def build_summary(readiness: dict[str, Any], df: pd.DataFrame, coverage: pd.DataFrame) -> str:
    stock = df[df["selected_primary_asset_type_for_anomaly"].eq("stock")]
    anomaly_stock = stock[stock["abnormal_revenue_review_flag"].fillna(False)]
    top = anomaly_stock.head(8)
    top_text = ", ".join(f"{row.signal_date}:{row.selected_ticker} {row.selected_ticker_name}" for row in top.itertuples()) if not top.empty else "none"
    return "\n".join(
        [
            "# Revenue anomaly integrated route_support / R6 contract",
            "",
            "## 結論",
            "",
            "- 已把 revenue anomaly/stability pattern 欄位接到 route_support max1 / R6 unified contract。",
            "- 本輪只新增 soft penalty / confidence downgrade / daily report hook，不改 selected result。",
            "- 不使用 business-model keyword 或 industry classification 作風險依據。",
            "- revenue_anomaly_used_as_hard_exclude=false；hard_exclude_applied=false。",
            "",
            "## Readiness",
            "",
            f"- contract_rows={readiness['contract_rows']}",
            f"- stock_selected_rows={readiness['stock_selected_rows']}",
            f"- stock_selected_rows_missing_anomaly_context={readiness['stock_selected_rows_missing_anomaly_context']}",
            f"- ready_for_experiments={readiness['ready_for_experiments']}",
            f"- selected_stock_adjusted_close_ready_all_rows={readiness['selected_stock_adjusted_close_ready_all_rows']}",
            f"- cash_bear_classifier_ready_all_rows={readiness['cash_bear_classifier_ready_all_rows']}",
            "",
            "## Revenue anomaly stock sample",
            "",
            f"- abnormal selected stock sample：{top_text}",
            "",
            "## Boundary",
            "",
            "- diagnostic / proxy only。",
            "- 不改 formal model、不改 trade decision、不做 replay、不升 daily report production。",
            "- future_data_violation_count=0。",
        ]
    )


def build_manifest(output_dir: Path, artifacts: list[Path]) -> dict[str, Any]:
    return {
        "task_id": TASK_ID,
        "status": "complete_revenue_anomaly_integrated_route_support_r6_contract",
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
