from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from backtest_lab.vnext_layer1_long_revenue_low_base_contract import DEFAULT_MONTHLY_REVENUE_DIR, load_monthly_revenue
from backtest_lab.vnext_revenue_anomaly_stability_pattern_contract import abnormal_review_flag, anomaly_report_text, compute_revenue_anomaly_metrics

TASK_ID = "TASK-BACKTEST-CORE-VNEXT-REVENUE-ANOMALY-SOFT-PENALTY-RERANK-CONTRACT-001"
DEFAULT_INTEGRATED = Path("outputs/vnext_revenue_anomaly_integrated_route_support_r6_contract_20260710")
DEFAULT_LAYER4 = Path("outputs/vnext_layer4_80_primary_pool_contract_20260708/layer4_80_primary_pool_contract.csv")
DEFAULT_OUTPUT = Path("outputs/vnext_revenue_anomaly_soft_penalty_rerank_contract_20260710")
VARIANTS = {
    "baseline_no_revenue_penalty": 0.0,
    "mild_revenue_penalty": 0.15,
    "balanced_revenue_penalty": 0.30,
    "strict_revenue_penalty": 0.50,
}
TOP_N = 10


def main() -> None:
    parser = argparse.ArgumentParser(description="Build revenue anomaly soft-penalty rerank contract.")
    parser.add_argument("--integrated-dir", default=str(DEFAULT_INTEGRATED))
    parser.add_argument("--layer4-pool", default=str(DEFAULT_LAYER4))
    parser.add_argument("--monthly-revenue-dir", default=str(DEFAULT_MONTHLY_REVENUE_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    integrated_dir = Path(args.integrated_dir)
    layer4_path = Path(args.layer4_pool)
    monthly_dir = Path(args.monthly_revenue_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline = pd.read_csv(integrated_dir / "revenue_anomaly_integrated_route_support_r6_contract.csv", dtype={"selected_ticker": str}, low_memory=False)
    layer4 = pd.read_csv(layer4_path, dtype={"ticker": str}, low_memory=False)
    candidate_dates = sorted(set(baseline["signal_date"].astype(str)))
    layer4 = layer4[layer4["snapshot_date"].astype(str).isin(candidate_dates)].copy()
    layer4 = layer4[layer4.get("is_layer4_primary_pool", True).astype(str).str.lower().eq("true")].copy()
    topn_base = build_topn_base(layer4, baseline)
    revenue = load_monthly_revenue(monthly_dir, scoped_tickers=sorted(topn_base["ticker"].astype(str).unique()), asof_date=max(candidate_dates))
    candidate_matrix = attach_pit_anomaly(topn_base, revenue)
    selected = build_rerank_contract(baseline, candidate_matrix)
    gap = build_ohlc_gap_ledger(selected)

    paths = {}
    paths["contract"] = output_dir / "revenue_anomaly_soft_penalty_rerank_contract.csv"
    selected.to_csv(paths["contract"], index=False, encoding="utf-8-sig")
    paths["candidate_topn"] = output_dir / "revenue_anomaly_soft_penalty_candidate_topn.csv"
    candidate_matrix.to_csv(paths["candidate_topn"], index=False, encoding="utf-8-sig")
    paths["policy"] = output_dir / "revenue_anomaly_soft_penalty_variant_policy.csv"
    build_policy().to_csv(paths["policy"], index=False, encoding="utf-8-sig")
    paths["coverage"] = output_dir / "revenue_anomaly_soft_penalty_requested_vs_actual_coverage.csv"
    build_coverage(selected, candidate_matrix, gap).to_csv(paths["coverage"], index=False, encoding="utf-8-sig")
    paths["gap"] = output_dir / "revenue_anomaly_soft_penalty_selected_ohlc_gap_ledger.csv"
    gap.to_csv(paths["gap"], index=False, encoding="utf-8-sig")
    paths["blocked"] = output_dir / "revenue_anomaly_soft_penalty_blocked_proxy_audit.csv"
    build_blocked(gap).to_csv(paths["blocked"], index=False, encoding="utf-8-sig")
    paths["future"] = output_dir / "revenue_anomaly_soft_penalty_future_data_audit.csv"
    build_future().to_csv(paths["future"], index=False, encoding="utf-8-sig")
    readiness = build_readiness(selected, candidate_matrix, gap)
    paths["readiness"] = output_dir / "readiness_for_revenue_anomaly_soft_penalty_rerank.json"
    write_json(paths["readiness"], readiness)
    paths["summary"] = output_dir / "final_summary_zh.md"
    paths["summary"].write_text(build_summary(readiness, gap), encoding="utf-8")
    paths["manifest"] = output_dir / "manifest.json"
    write_json(paths["manifest"], build_manifest(output_dir, [p for k, p in paths.items() if k != "manifest"]))

    print(f"REVENUE_ANOMALY_SOFT_PENALTY_RERANK_OUTPUT={output_dir.resolve()}")
    print(f"CONTRACT_ROWS={len(selected)}")
    print(f"CANDIDATE_ROWS={len(candidate_matrix)}")
    print(f"OHLC_GAP_ROWS={len(gap)}")
    print(f"READY_FOR_EXPERIMENTS={readiness['ready_for_experiments']}")


def norm(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)


def build_topn_base(layer4: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    keep = ["snapshot_date", "ticker", "name", "market", "pool_rank", "pool_selection_score", "layer4_risk_aware_score", "layer4_broad_opportunity_net_score", "layer4_c_quota_base_score", "route_support_variant_count", "route_support_mode_count", "two_plus_opportunity_labels"]
    for col in keep:
        if col not in layer4.columns:
            layer4[col] = pd.NA
    l4 = layer4[keep].copy()
    l4["signal_date"] = l4["snapshot_date"].astype(str)
    l4["ticker"] = norm(l4["ticker"])
    l4["baseline_candidate_score"] = pd.to_numeric(l4["pool_selection_score"], errors="coerce").fillna(pd.to_numeric(l4["layer4_risk_aware_score"], errors="coerce")).fillna(0)
    l4 = l4.sort_values(["signal_date", "baseline_candidate_score", "pool_rank"], ascending=[True, False, True]).groupby("signal_date", as_index=False).head(TOP_N).copy()
    l4["candidate_source"] = "layer4_primary80_top10_pool_selection_score"

    selected_stock = baseline[baseline["selected_primary_asset_type_for_anomaly"].eq("stock")].copy()
    selected_stock["signal_date"] = selected_stock["signal_date"].astype(str)
    selected_stock["ticker"] = norm(selected_stock["selected_ticker"])
    selected_rows = selected_stock[["signal_date", "ticker", "selected_ticker_name", "selected_branch"]].drop_duplicates()
    selected_rows = selected_rows.rename(columns={"selected_ticker_name": "name"})
    selected_rows["market"] = ""
    selected_rows["pool_rank"] = 0
    selected_rows["pool_selection_score"] = pd.NA
    selected_rows["layer4_risk_aware_score"] = pd.NA
    selected_rows["layer4_broad_opportunity_net_score"] = pd.NA
    selected_rows["layer4_c_quota_base_score"] = pd.NA
    selected_rows["route_support_variant_count"] = pd.NA
    selected_rows["route_support_mode_count"] = pd.NA
    selected_rows["two_plus_opportunity_labels"] = pd.NA
    selected_rows["snapshot_date"] = selected_rows["signal_date"]
    selected_rows["baseline_candidate_score"] = pd.NA
    selected_rows["candidate_source"] = "original_selected_stock_injected_if_missing_from_top10"
    merged_key = set(zip(l4["signal_date"], l4["ticker"]))
    add = selected_rows[[k not in merged_key for k in zip(selected_rows["signal_date"], selected_rows["ticker"])]].copy()
    if not add.empty:
        min_scores = l4.groupby("signal_date")["baseline_candidate_score"].min().to_dict()
        add["baseline_candidate_score"] = add["signal_date"].map(min_scores).fillna(0)
        l4 = pd.concat([l4, add[l4.columns]], ignore_index=True, sort=False)
    l4["candidate_rank_before_penalty"] = l4.groupby("signal_date")["baseline_candidate_score"].rank(method="first", ascending=False).astype(int)
    return l4


def attach_pit_anomaly(candidates: pd.DataFrame, revenue: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in candidates.iterrows():
        signal_date = pd.Timestamp(row["signal_date"])
        ticker = str(row["ticker"])
        ticker_revenue = revenue[(revenue["ticker"].astype(str).eq(ticker)) & (pd.to_datetime(revenue["available_date"], errors="coerce").le(signal_date))].sort_values("period")
        metrics = compute_revenue_anomaly_metrics(ticker_revenue)
        payload = row.to_dict()
        payload.update(metrics)
        rows.append(payload)
    df = pd.DataFrame(rows)
    df["abnormal_revenue_review_flag"] = abnormal_review_flag(df)
    df["revenue_anomaly_report_text"] = df.apply(anomaly_report_text, axis=1)
    df["revenue_anomaly_penalty_score"] = penalty_score(df)
    df["revenue_hygiene_confidence_level"] = confidence(df)
    df["business_model_keyword_proxy_used_as_risk_basis"] = False
    df["industry_classification_used_as_risk_basis"] = False
    df["hard_exclude_applied"] = False
    df["future_data_violation_count"] = 0
    return df


def penalty_score(df: pd.DataFrame) -> pd.Series:
    review = df["abnormal_revenue_review_flag"].fillna(False).astype(bool).astype(float)
    concentration = pd.to_numeric(df.get("revenue_concentration_ratio_top3_12m"), errors="coerce").fillna(0).clip(0, 1)
    persistence = pd.to_numeric(df.get("revenue_growth_persistence_score"), errors="coerce").fillna(1).clip(0, 1)
    lumpiness = pd.to_numeric(df.get("revenue_lumpiness_score"), errors="coerce").fillna(0).clip(0, 1)
    gap = pd.to_numeric(df.get("ttm_vs_recent_growth_gap"), errors="coerce").fillna(0).clip(lower=0, upper=2) / 2
    low_base = df.get("low_base_distortion_flag", pd.Series(False, index=df.index)).fillna(False).astype(bool).astype(float)
    return (review * 0.25 + concentration * 0.2 + (1 - persistence) * 0.15 + lumpiness * 0.25 + gap * 0.1 + low_base * 0.05).clip(0, 1).round(6)


def confidence(df: pd.DataFrame) -> pd.Series:
    score = pd.to_numeric(df["revenue_anomaly_penalty_score"], errors="coerce").fillna(0)
    out = pd.Series("high_no_major_revenue_anomaly", index=df.index)
    out.loc[score.ge(0.25) & score.lt(0.45)] = "medium_minor_revenue_anomaly_review"
    out.loc[score.ge(0.45)] = "low_revenue_anomaly_confidence_downgrade"
    out.loc[df["source_quality"].astype(str).eq("blocked_no_monthly_revenue_rows")] = "blocked_missing_revenue_anomaly_context"
    return out


def build_rerank_contract(baseline: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    rows = []
    candidate_groups = {k: v.copy() for k, v in candidates.groupby("signal_date")}
    for _, base in baseline.iterrows():
        signal_date = str(base["signal_date"])
        before = str(base["selected_ticker"]).replace(".0", "")
        is_stock = str(base.get("selected_primary_asset_type_for_anomaly", base.get("selected_asset_type", ""))).lower() == "stock"
        date_candidates = candidate_groups.get(signal_date, pd.DataFrame())
        for variant, weight in VARIANTS.items():
            selected = choose_candidate(date_candidates, before, is_stock, variant, weight)
            rows.append(build_selected_row(base, selected, variant, weight, before, is_stock))
        selected = choose_veto_candidate(date_candidates, before, is_stock)
        rows.append(build_selected_row(base, selected, "anomaly_veto_only_when_alternative_available", 1.0, before, is_stock))
    return pd.DataFrame(rows)


def choose_candidate(candidates: pd.DataFrame, before: str, is_stock: bool, variant: str, weight: float) -> pd.Series | None:
    if not is_stock or candidates.empty:
        return None
    if variant == "baseline_no_revenue_penalty":
        selected = candidates[candidates["ticker"].astype(str).eq(before)]
        return selected.iloc[0] if not selected.empty else None
    pool = candidates.copy()
    pool["rerank_score"] = pd.to_numeric(pool["baseline_candidate_score"], errors="coerce").fillna(0) - weight * pd.to_numeric(pool["revenue_anomaly_penalty_score"], errors="coerce").fillna(0)
    return pool.sort_values(["rerank_score", "baseline_candidate_score"], ascending=[False, False]).iloc[0]


def choose_veto_candidate(candidates: pd.DataFrame, before: str, is_stock: bool) -> pd.Series | None:
    if not is_stock or candidates.empty:
        return None
    before_row = candidates[candidates["ticker"].astype(str).eq(before)]
    if before_row.empty:
        return choose_candidate(candidates, before, is_stock, "balanced_revenue_penalty", 0.3)
    before_penalty = float(before_row.iloc[0].get("revenue_anomaly_penalty_score", 0) or 0)
    if not bool(before_row.iloc[0].get("abnormal_revenue_review_flag", False)) and before_penalty < 0.45:
        return before_row.iloc[0]
    alternatives = candidates[~candidates["ticker"].astype(str).eq(before)].copy()
    alternatives = alternatives[pd.to_numeric(alternatives["revenue_anomaly_penalty_score"], errors="coerce").fillna(1).lt(before_penalty)]
    if alternatives.empty:
        return before_row.iloc[0]
    alternatives["rerank_score"] = pd.to_numeric(alternatives["baseline_candidate_score"], errors="coerce").fillna(0) - 0.3 * pd.to_numeric(alternatives["revenue_anomaly_penalty_score"], errors="coerce").fillna(0)
    return alternatives.sort_values(["rerank_score", "baseline_candidate_score"], ascending=[False, False]).iloc[0]


def build_selected_row(base: pd.Series, selected: pd.Series | None, variant: str, weight: float, before: str, is_stock: bool) -> dict[str, Any]:
    after = before if selected is None else str(selected.get("ticker", before))
    changed = is_stock and after != before
    selected_name_after = base.get("selected_ticker_name", "") if selected is None else selected.get("name", "")
    reason = "non_stock_or_fallback_no_rerank" if not is_stock else "baseline_preserved"
    if changed:
        reason = "revenue_anomaly_soft_penalty_rerank_selected_alternative"
    elif variant != "baseline_no_revenue_penalty" and is_stock:
        reason = "rerank_evaluated_original_retained"
    path_ready = bool(base.get("path_ready", False)) if not changed else False
    official_ready = bool(base.get("official_selected_stock_ohlc_ready", False)) if not changed else False
    return {
        "signal_date": base.get("signal_date"),
        "next_signal_date": base.get("next_signal_date"),
        "period_label": base.get("period_label"),
        "in_P1": base.get("in_P1"),
        "in_P2": base.get("in_P2"),
        "in_2024_latest": base.get("in_2024_latest"),
        "in_2026YTD": base.get("in_2026YTD"),
        "rerank_variant": variant,
        "revenue_penalty_weight": weight,
        "selected_branch_before": base.get("selected_branch"),
        "selected_ticker_before": before,
        "selected_ticker_name_before": base.get("selected_ticker_name"),
        "selected_asset_type_before": base.get("selected_asset_type"),
        "selected_ticker_after": after,
        "selected_ticker_name_after": selected_name_after,
        "selected_asset_type_after": base.get("selected_asset_type") if is_stock else base.get("selected_asset_type"),
        "selected_result_changed": changed,
        "changed_reason": reason,
        "candidate_rank_before_penalty": selected.get("candidate_rank_before_penalty", pd.NA) if selected is not None else pd.NA,
        "candidate_source": selected.get("candidate_source", "") if selected is not None else "",
        "baseline_candidate_score": selected.get("baseline_candidate_score", pd.NA) if selected is not None else pd.NA,
        "rerank_score": selected.get("rerank_score", selected.get("baseline_candidate_score", pd.NA)) if selected is not None else pd.NA,
        "abnormal_revenue_review_flag": selected.get("abnormal_revenue_review_flag", False) if selected is not None else base.get("abnormal_revenue_review_flag", False),
        "revenue_growth_persistence_score": selected.get("revenue_growth_persistence_score", pd.NA) if selected is not None else base.get("revenue_growth_persistence_score", pd.NA),
        "revenue_concentration_ratio_top3_12m": selected.get("revenue_concentration_ratio_top3_12m", pd.NA) if selected is not None else base.get("revenue_concentration_ratio_top3_12m", pd.NA),
        "revenue_lumpiness_score": selected.get("revenue_lumpiness_score", pd.NA) if selected is not None else base.get("revenue_lumpiness_score", pd.NA),
        "ttm_vs_recent_growth_gap": selected.get("ttm_vs_recent_growth_gap", pd.NA) if selected is not None else base.get("ttm_vs_recent_growth_gap", pd.NA),
        "low_base_distortion_flag": selected.get("low_base_distortion_flag", False) if selected is not None else base.get("low_base_distortion_flag", False),
        "revenue_anomaly_penalty_score": selected.get("revenue_anomaly_penalty_score", 0) if selected is not None else base.get("revenue_anomaly_penalty_score", 0),
        "revenue_hygiene_confidence_level": selected.get("revenue_hygiene_confidence_level", "") if selected is not None else base.get("revenue_hygiene_confidence_level", ""),
        "report_revenue_anomaly_warning": "rerank candidate revenue anomaly context; diagnostic only" if is_stock else "fallback/ETF row not reranked",
        "report_revenue_anomaly_reason": selected.get("revenue_anomaly_report_text", "") if selected is not None else base.get("report_revenue_anomaly_reason", ""),
        "entry_date": base.get("entry_date"),
        "exit_date": base.get("exit_date"),
        "official_unadjusted_ohlc_ready": official_ready,
        "path_ready": path_ready,
        "selected_stock_adjusted_close_ready": bool(base.get("selected_stock_adjusted_close_ready", False)) if not changed else False,
        "transition_cost_rate_source": "preserve_baseline_if_unchanged_else_requires_path_refresh",
        "future_data_violation_count": 0,
        "business_model_keyword_proxy_used_as_risk_basis": False,
        "industry_classification_used_as_risk_basis": False,
        "hard_exclude_applied": False,
        "diagnostic_only": True,
        "formal_model_changed": False,
        "trade_decision_changed": False,
        "active_in_trade_decision": False,
        "report_changed": False,
        "portfolio_replay_executed": False,
        "ready_for_strategy_replay": False,
        "ready_for_formal": False,
        "not_live_rule": True,
        "forward_returns_live_rule_usage": False,
    }


def build_ohlc_gap_ledger(selected: pd.DataFrame) -> pd.DataFrame:
    gap = selected[(selected["selected_result_changed"]) & (~selected["official_unadjusted_ohlc_ready"])].copy()
    cols = ["signal_date", "entry_date", "exit_date", "rerank_variant", "selected_ticker_before", "selected_ticker_after", "selected_ticker_name_after", "changed_reason"]
    if gap.empty:
        return pd.DataFrame(columns=cols + ["missing_field", "next_owner"])
    gap = gap[cols].drop_duplicates()
    gap["missing_field"] = "entry_open/entry_close/exit_close official unadjusted OHLC for reranked selected ticker"
    gap["next_owner"] = "Radar/Data bounded selected-ticker OHLC gap fill"
    return gap


def build_policy() -> pd.DataFrame:
    return pd.DataFrame([
        {"variant": "baseline_no_revenue_penalty", "policy": "preserve original route_support/R6 selected result", "hard_exclude": False},
        {"variant": "mild_revenue_penalty", "policy": "baseline_score - 0.15 * revenue_anomaly_penalty_score", "hard_exclude": False},
        {"variant": "balanced_revenue_penalty", "policy": "baseline_score - 0.30 * revenue_anomaly_penalty_score", "hard_exclude": False},
        {"variant": "strict_revenue_penalty", "policy": "baseline_score - 0.50 * revenue_anomaly_penalty_score", "hard_exclude": False},
        {"variant": "anomaly_veto_only_when_alternative_available", "policy": "switch only when original has worse anomaly penalty and cleaner alternative exists", "hard_exclude": False},
    ])


def build_coverage(selected: pd.DataFrame, candidates: pd.DataFrame, gap: pd.DataFrame) -> pd.DataFrame:
    rows = []
    periods = {"P1": "in_P1", "P2": "in_P2", "2024-latest": "in_2024_latest", "2026YTD": "in_2026YTD", "full_integrated": None}
    for label, col in periods.items():
        subset = selected if col is None else selected[selected[col].fillna(False).astype(bool)]
        rows.append({
            "period": label,
            "contract_rows": len(subset),
            "selected_result_changed_rows": int(subset["selected_result_changed"].sum()),
            "unique_signal_dates": subset["signal_date"].nunique(),
            "ohlc_gap_rows": int(gap["signal_date"].isin(subset["signal_date"]).sum()) if not gap.empty else 0,
            "candidate_topn_rows": int(candidates["signal_date"].isin(subset["signal_date"]).sum()),
            "future_data_violation_count": 0,
        })
    return pd.DataFrame(rows)


def build_blocked(gap: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([
        {"field": "reranked_selected_stock_ohlc_path", "status": "ready" if gap.empty else "blocked_partial", "evidence": f"gap_rows={len(gap)}", "policy": "Do not run exact net-cost diagnostic until bounded OHLC gap fill is absorbed."},
        {"field": "adjusted_close", "status": "blocked", "evidence": "selected_stock_adjusted_close remains blocked in upstream", "policy": "diagnostic unadjusted OHLC only, not formal."},
        {"field": "business_model_keyword_proxy", "status": "deprecated_not_used", "evidence": "not used in scoring", "policy": "No industry/business keyword risk basis."},
        {"field": "hard_exclude", "status": "not_allowed", "evidence": "all variants are rerank/substitute only", "policy": "No hard delete."},
    ])


def build_future() -> pd.DataFrame:
    return pd.DataFrame([
        {"audit_item": "future_return_as_rule", "status": "pass", "violation_count": 0},
        {"audit_item": "business_model_keyword_as_rule", "status": "pass", "violation_count": 0},
        {"audit_item": "hard_exclude", "status": "pass", "violation_count": 0},
    ])


def build_readiness(selected: pd.DataFrame, candidates: pd.DataFrame, gap: pd.DataFrame) -> dict[str, Any]:
    return {
        "task_id": TASK_ID,
        "status": "revenue_anomaly_soft_penalty_rerank_contract_ready_path_blocked" if not gap.empty else "revenue_anomaly_soft_penalty_rerank_contract_ready",
        "contract_rows": int(len(selected)),
        "candidate_topn_rows": int(len(candidates)),
        "selected_result_changed_rows": int(selected["selected_result_changed"].sum()),
        "reranked_selected_ohlc_gap_rows": int(len(gap)),
        "ready_for_experiments": bool(gap.empty),
        "ready_for_radar_data_gap_fill": bool(not gap.empty),
        "ready_for_formal": False,
        "ready_for_strategy_replay": False,
        "business_model_keyword_proxy_used_as_risk_basis": False,
        "industry_classification_used_as_risk_basis": False,
        "hard_exclude_applied": False,
        "formal_model_changed": False,
        "trade_decision_changed": False,
        "active_in_trade_decision": False,
        "report_changed": False,
        "portfolio_replay_executed": False,
        "not_live_rule": True,
        "forward_returns_live_rule_usage": False,
        "future_data_violation_count": 0,
    }


def build_summary(readiness: dict[str, Any], gap: pd.DataFrame) -> str:
    return "\n".join([
        "# Revenue anomaly soft-penalty rerank contract",
        "",
        "## 結論",
        "",
        "- 已建立 route_support / R6 revenue anomaly soft-penalty rerank contract。",
        "- rerank 使用 Layer4 primary80 每週 PIT topN 候選，不使用 future return。",
        "- revenue anomaly 只作 soft penalty / substitute ranking，不作 standalone alpha、不 hard exclude。",
        f"- selected_result_changed_rows={readiness['selected_result_changed_rows']}",
        f"- reranked_selected_ohlc_gap_rows={readiness['reranked_selected_ohlc_gap_rows']}",
        f"- ready_for_experiments={readiness['ready_for_experiments']}",
        "",
        "## Next",
        "",
        "- 若 gap_rows > 0，需先交 Radar/Data 補 reranked selected ticker official OHLC path，再回 Core refresh readiness。",
        "- business_model / industry keyword 已 deprecated，不作風險依據。",
    ])


def build_manifest(output_dir: Path, artifacts: list[Path]) -> dict[str, Any]:
    return {"task_id": TASK_ID, "status": "complete_revenue_anomaly_soft_penalty_rerank_contract", "output_dir": str(output_dir), "artifacts": [{"path": str(p), "sha256": sha256_file(p), "bytes": p.stat().st_size} for p in artifacts], "flags": {"formal_model_changed": False, "trade_decision_changed": False, "active_in_trade_decision": False, "report_changed": False, "portfolio_replay_executed": False, "ready_for_strategy_replay": False, "ready_for_formal": False, "not_live_rule": True, "forward_returns_live_rule_usage": False}}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


if __name__ == "__main__":
    main()

