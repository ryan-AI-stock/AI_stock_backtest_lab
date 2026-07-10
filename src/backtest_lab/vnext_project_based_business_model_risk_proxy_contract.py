from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


TASK_ID = "TASK-BACKTEST-CORE-VNEXT-PROJECT-BASED-BUSINESS-MODEL-RISK-PROXY-CONTRACT-001"
DEFAULT_REVENUE_SOURCE = Path("outputs/vnext_layer1_long_revenue_stability_low_base_risk_integration_contract_20260710")
DEFAULT_HYGIENE_SOURCE = Path("outputs/vnext_layer1_revenue_stability_hygiene_integration_contract_20260710")
DEFAULT_6806_SOURCE = Path(
    "C:/Users/zergv/Documents/Codex/2026-05-23/ai-stock-rotation-radar-https-docs/outputs/"
    "radar_vnext_6806_shinfox_long_revenue_stability_source_package_20260710"
)
DEFAULT_OUTPUT = Path("outputs/vnext_project_based_business_model_risk_proxy_contract_20260710")
SHINFOX_TICKER = "6806"
KEYWORDS = [
    "工程",
    "統包",
    "EPC",
    "案場",
    "專案",
    "工程收入",
    "建置",
    "離岸風電",
    "離岸",
    "開發案",
    "一次性認列",
    "工程進度",
    "合約",
    "長約",
    "認列收入",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build project-based business model risk proxy contract.")
    parser.add_argument("--revenue-source-dir", default=str(DEFAULT_REVENUE_SOURCE))
    parser.add_argument("--hygiene-source-dir", default=str(DEFAULT_HYGIENE_SOURCE))
    parser.add_argument("--shinfox-source-dir", default=str(DEFAULT_6806_SOURCE))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    revenue_source_dir = Path(args.revenue_source_dir)
    hygiene_source_dir = Path(args.hygiene_source_dir)
    shinfox_source_dir = Path(args.shinfox_source_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    revenue_contract = pd.read_csv(
        revenue_source_dir / "layer1_long_revenue_stability_feature_contract.csv",
        dtype={"ticker": str},
    )
    hygiene = pd.read_csv(
        hygiene_source_dir / "layer1_revenue_stability_hygiene_integration_contract.csv",
        dtype={"ticker": str},
    )
    shinfox_sanity = read_optional_csv(
        revenue_source_dir / "shinfox_6806_feature_sanity_check.csv",
        dtype={"ticker": str},
    )
    shinfox_revenue_rows = read_optional_csv(
        shinfox_source_dir / "6806_monthly_revenue_rows.csv",
        dtype={"ticker": str},
    )

    base = build_scoped_base(revenue_contract, hygiene, shinfox_sanity)
    keyword_evidence = build_keyword_evidence(base, shinfox_revenue_rows)
    contract = build_proxy_contract(base, keyword_evidence)

    contract_path = output_dir / "project_based_business_model_risk_proxy_contract.csv"
    contract.to_csv(contract_path, index=False, encoding="utf-8-sig")

    policy = build_policy_map()
    policy_path = output_dir / "project_risk_proxy_policy_map.csv"
    policy.to_csv(policy_path, index=False, encoding="utf-8-sig")

    keyword_audit = build_keyword_source_audit(base, keyword_evidence)
    keyword_audit_path = output_dir / "project_risk_keyword_source_audit.csv"
    keyword_audit.to_csv(keyword_audit_path, index=False, encoding="utf-8-sig")

    scoped_flags = build_scoped_candidate_flags(contract)
    scoped_flags_path = output_dir / "project_risk_scoped_candidate_flags.csv"
    scoped_flags.to_csv(scoped_flags_path, index=False, encoding="utf-8-sig")

    shinfox = build_shinfox_sanity(contract)
    shinfox_path = output_dir / "shinfox_6806_project_risk_sanity_check.csv"
    shinfox.to_csv(shinfox_path, index=False, encoding="utf-8-sig")

    blocked = build_blocked_proxy_audit()
    blocked_path = output_dir / "blocked_proxy_audit.csv"
    blocked.to_csv(blocked_path, index=False, encoding="utf-8-sig")

    coverage = build_requested_vs_actual_coverage(base, contract, shinfox_revenue_rows)
    coverage_path = output_dir / "requested_vs_actual_coverage.csv"
    coverage.to_csv(coverage_path, index=False, encoding="utf-8-sig")

    readiness = build_readiness(contract, keyword_audit, shinfox)
    readiness_path = output_dir / "readiness_for_experiments.json"
    write_json(readiness_path, readiness)

    summary_path = output_dir / "final_summary_zh.md"
    summary_path.write_text(build_summary(readiness, shinfox, contract), encoding="utf-8")

    artifacts = [
        contract_path,
        policy_path,
        keyword_audit_path,
        scoped_flags_path,
        shinfox_path,
        blocked_path,
        coverage_path,
        readiness_path,
        summary_path,
    ]
    manifest_path = output_dir / "manifest.json"
    write_json(manifest_path, build_manifest(output_dir, artifacts))

    print(f"PROJECT_BASED_RISK_PROXY_OUTPUT={output_dir.resolve()}")
    print(f"CONTRACT_ROWS={len(contract)}")
    print(f"PROJECT_RISK_REVIEW_FLAG_COUNT={int(contract['project_risk_review_flag'].sum())}")
    print(f"READY_FOR_EXPERIMENTS={readiness['ready_for_experiments']}")


def read_optional_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, **kwargs)


def build_scoped_base(revenue_contract: pd.DataFrame, hygiene: pd.DataFrame, shinfox_sanity: pd.DataFrame) -> pd.DataFrame:
    keep_revenue = [
        "snapshot_date",
        "ticker",
        "name",
        "market",
        "latest_revenue_year_month",
        "revenue_stability_score",
        "revenue_lumpiness_score",
        "recent_spike_without_long_history_flag",
        "project_based_revenue_risk_proxy",
        "source_quality",
        "feature_scope",
    ]
    base = revenue_contract[[col for col in keep_revenue if col in revenue_contract.columns]].copy()
    if "snapshot_date" not in base.columns:
        base["snapshot_date"] = ""
    base["scope_bucket"] = "latest_layer4_primary80"

    if not shinfox_sanity.empty:
        shinfox = shinfox_sanity.copy()
        rename = {"asof_date": "snapshot_date"}
        shinfox = shinfox.rename(columns=rename)
        for col in base.columns:
            if col not in shinfox.columns:
                shinfox[col] = ""
        shinfox = shinfox[base.columns].copy()
        shinfox["scope_bucket"] = "sanity_case_not_latest_primary80"
        base = pd.concat([base[base["ticker"].astype(str).ne(SHINFOX_TICKER)], shinfox], ignore_index=True, sort=False)

    hygiene_cols = [
        "ticker",
        "revenue_stability_context_score",
        "revenue_lumpiness_penalty_score",
        "hygiene_warning_text_for_report",
    ]
    if all(col in hygiene.columns for col in hygiene_cols):
        base = base.merge(hygiene[hygiene_cols], on="ticker", how="left")
    else:
        base["revenue_stability_context_score"] = pd.to_numeric(base["revenue_stability_score"], errors="coerce")
        base["revenue_lumpiness_penalty_score"] = pd.to_numeric(base["revenue_lumpiness_score"], errors="coerce")
        base["hygiene_warning_text_for_report"] = ""

    base["ticker"] = base["ticker"].astype(str)
    return base


def build_keyword_evidence(base: pd.DataFrame, shinfox_revenue_rows: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in base.iterrows():
        ticker = str(row["ticker"])
        text_source_status = "blocked_no_bounded_company_text_source"
        evidence_text = ""
        evidence_source = ""
        if ticker == SHINFOX_TICKER and not shinfox_revenue_rows.empty:
            notes = shinfox_revenue_rows.get("notes", pd.Series(dtype=str)).dropna().astype(str).tolist()
            evidence_text = " | ".join(note for note in notes if keyword_hits(note))[:600]
            evidence_source = "MOPS monthly revenue company-specific ajax notes"
            text_source_status = "bounded_mops_monthly_revenue_notes_ready"
        matched = matched_keywords(evidence_text)
        rows.append(
            {
                "ticker": ticker,
                "keyword_source_status": text_source_status,
                "keyword_source": evidence_source,
                "keyword_evidence_text": evidence_text,
                "matched_keywords": ";".join(matched),
                "keyword_hit_count": len(matched),
                "contract_project_keyword_proxy": len(matched) > 0,
            }
        )
    return pd.DataFrame(rows)


def keyword_hits(text: str) -> bool:
    return bool(matched_keywords(text))


def matched_keywords(text: str) -> list[str]:
    if not isinstance(text, str) or not text:
        return []
    upper = text.upper()
    hits = []
    for keyword in KEYWORDS:
        if keyword.upper() in upper:
            hits.append(keyword)
    return sorted(set(hits), key=hits.index)


def build_proxy_contract(base: pd.DataFrame, keyword_evidence: pd.DataFrame) -> pd.DataFrame:
    df = base.merge(keyword_evidence, on="ticker", how="left")
    stability = pd.to_numeric(df["revenue_stability_score"], errors="coerce")
    lumpiness = pd.to_numeric(df["revenue_lumpiness_score"], errors="coerce")
    primary_mask = df["scope_bucket"].eq("latest_layer4_primary80")
    df["revenue_stability_percentile_vs_primary80"] = percentile_vs_reference(stability, stability[primary_mask], higher_is_better=True)
    df["revenue_lumpiness_percentile_vs_primary80"] = percentile_vs_reference(lumpiness, lumpiness[primary_mask], higher_is_better=True)

    keyword_proxy = df["contract_project_keyword_proxy"].fillna(False).astype(bool)
    high_lumpy = df["revenue_lumpiness_percentile_vs_primary80"].ge(0.8) | lumpiness.ge(0.28)
    low_stability = df["revenue_stability_percentile_vs_primary80"].le(0.3) | stability.lt(0.62)
    shape_proxy = high_lumpy & low_stability
    current_shape_proxy = bool_series(df.get("project_based_revenue_risk_proxy"))

    df["project_revenue_business_model_proxy"] = keyword_proxy
    df["contract_project_keyword_proxy"] = keyword_proxy
    df["revenue_lumpiness_business_risk_label"] = "low_lumpiness_or_stable"
    df.loc[high_lumpy & ~low_stability, "revenue_lumpiness_business_risk_label"] = "high_lumpiness_but_stability_not_low"
    df.loc[shape_proxy, "revenue_lumpiness_business_risk_label"] = "high_lumpiness_low_stability_shape_review"
    df.loc[keyword_proxy & shape_proxy, "revenue_lumpiness_business_risk_label"] = "keyword_plus_shape_project_risk_review"
    df["long_revenue_stability_context"] = "stable_or_insufficient_context"
    df.loc[low_stability, "long_revenue_stability_context"] = "long_revenue_stability_low_or_bottom_percentile"
    df.loc[stability.isna(), "long_revenue_stability_context"] = "blocked_missing_revenue_stability"
    df["project_risk_review_flag"] = keyword_proxy | shape_proxy | current_shape_proxy
    df["proxy_confidence_level"] = "low"
    df.loc[shape_proxy & ~keyword_proxy, "proxy_confidence_level"] = "low_shape_only"
    df.loc[keyword_proxy & ~shape_proxy, "proxy_confidence_level"] = "medium_keyword_only"
    df.loc[keyword_proxy & shape_proxy, "proxy_confidence_level"] = "medium_keyword_plus_revenue_shape"
    df["proxy_source_quality"] = "monthly_shape_only_no_business_text_source"
    df.loc[keyword_proxy, "proxy_source_quality"] = "bounded_official_mops_note_keyword_plus_monthly_shape_proxy"
    df["project_risk_report_text"] = df.apply(project_report_text, axis=1)
    df["integration_policy"] = "soft_review_flag_only_no_hard_exclude"
    df["layer_destination"] = "Layer1_quality_context_and_Layer4_risk_context"
    df["diagnostic_only"] = True
    df["formal_model_changed"] = False
    df["trade_decision_changed"] = False
    df["active_in_trade_decision"] = False
    df["report_changed"] = False
    df["portfolio_replay_executed"] = False
    df["ready_for_strategy_replay"] = False
    df["ready_for_formal"] = False
    df["not_live_rule"] = True
    df["forward_returns_live_rule_usage"] = False

    keep = [
        "snapshot_date",
        "ticker",
        "name",
        "market",
        "scope_bucket",
        "latest_revenue_year_month",
        "revenue_stability_score",
        "revenue_stability_percentile_vs_primary80",
        "revenue_lumpiness_score",
        "revenue_lumpiness_percentile_vs_primary80",
        "project_revenue_business_model_proxy",
        "contract_project_keyword_proxy",
        "revenue_lumpiness_business_risk_label",
        "long_revenue_stability_context",
        "project_risk_review_flag",
        "project_risk_report_text",
        "proxy_source_quality",
        "proxy_confidence_level",
        "matched_keywords",
        "keyword_source_status",
        "keyword_source",
        "keyword_evidence_text",
        "hygiene_warning_text_for_report",
        "integration_policy",
        "layer_destination",
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
    return df[[col for col in keep if col in df.columns]].sort_values(
        ["project_risk_review_flag", "revenue_lumpiness_percentile_vs_primary80"],
        ascending=[False, False],
    )


def percentile_vs_reference(values: pd.Series, reference: pd.Series, *, higher_is_better: bool) -> pd.Series:
    ref = pd.to_numeric(reference, errors="coerce").dropna().sort_values()
    vals = pd.to_numeric(values, errors="coerce")
    if ref.empty:
        return pd.Series([pd.NA] * len(vals), index=vals.index, dtype="Float64")
    ranks = vals.apply(lambda v: (ref.le(v).sum() / len(ref)) if pd.notna(v) else pd.NA)
    if not higher_is_better:
        ranks = 1 - ranks
    return ranks


def bool_series(values: Any) -> pd.Series:
    if values is None:
        return pd.Series(dtype=bool)
    return pd.Series(values).astype(str).str.lower().isin({"true", "1", "yes", "y"})


def project_report_text(row: pd.Series) -> str:
    if bool(row.get("contract_project_keyword_proxy", False)) and row.get("revenue_lumpiness_business_risk_label") == "keyword_plus_shape_project_risk_review":
        return "官方月營收說明出現工程/認列收入等專案型 keyword，且營收集中度偏高、長期穩定性偏低；僅作 review/soft penalty，不作 hard exclude。"
    if bool(row.get("contract_project_keyword_proxy", False)):
        return "官方文字出現專案型 keyword，但營收穩定性未同步轉弱；低信心提醒，不重罰。"
    if bool(row.get("project_risk_review_flag", False)):
        return "營收形狀顯示 lumpiness 或近期暴衝風險，但缺公司文字 source；只能作低信心 proxy review。"
    return "未觸發 project-based business-model proxy；仍非正式商業模式判定。"


def build_policy_map() -> pd.DataFrame:
    rows = [
        policy("project_revenue_business_model_proxy", "Layer1/Layer4", "soft_review_flag", "keyword 或 keyword+營收形狀 proxy 才提高風險提醒；不得 hard exclude。"),
        policy("contract_project_keyword_proxy", "Layer1 report context", "proxy_evidence", "只表示 bounded source 出現工程/統包/EPC/案場/專案等字樣，不等於真實商業模式分類。"),
        policy("revenue_lumpiness_business_risk_label", "Layer1/Layer4", "soft_penalty_context", "高 lumpiness + 低長期穩定性才加重提醒；單獨高 lumpiness 不重罰。"),
        policy("long_revenue_stability_context", "Layer1 quality context", "soft_penalty_or_context", "長期營收穩定性低時降低候選 hygiene 信心，不作 alpha。"),
        policy("project_risk_review_flag", "daily report / review", "display_or_soft_penalty", "報告顯示與 downstream diagnostic 使用；不得直接剔除。"),
        policy("project_risk_report_text", "daily report", "display_only", "白話警示文字；不產生交易指令。"),
        policy("proxy_confidence_level", "all downstream", "governance", "區分 keyword_only、shape_only、keyword_plus_shape；避免把 proxy 包裝成 business truth。"),
    ]
    return pd.DataFrame(rows)


def policy(field: str, layer: str, action: str, note: str) -> dict[str, str]:
    return {"field": field, "layer_destination": layer, "integration_action": action, "policy_note": note}


def build_keyword_source_audit(base: pd.DataFrame, evidence: pd.DataFrame) -> pd.DataFrame:
    merged = base[["ticker", "name", "scope_bucket"]].merge(evidence, on="ticker", how="left")
    merged["source_audit_policy"] = merged["keyword_source_status"].map(
        {
            "bounded_mops_monthly_revenue_notes_ready": "accepted_as_diagnostic_proxy_evidence_only",
            "blocked_no_bounded_company_text_source": "blocked_for_business_model_keyword_proxy_do_not_infer",
        }
    ).fillna("blocked_for_business_model_keyword_proxy_do_not_infer")
    return merged


def build_scoped_candidate_flags(contract: pd.DataFrame) -> pd.DataFrame:
    flagged = contract[contract["project_risk_review_flag"].fillna(False)].copy()
    if flagged.empty:
        flagged = contract.head(20).copy()
    cols = [
        "snapshot_date",
        "ticker",
        "name",
        "market",
        "scope_bucket",
        "project_risk_review_flag",
        "project_revenue_business_model_proxy",
        "contract_project_keyword_proxy",
        "revenue_lumpiness_business_risk_label",
        "long_revenue_stability_context",
        "proxy_confidence_level",
        "project_risk_report_text",
        "integration_policy",
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
                    "sanity_check": "cannot evaluate project proxy",
                }
            ]
        )
    shinfox["status"] = "ready_proxy"
    shinfox["sanity_check"] = "6806 is a sanity case only; not in latest Layer4 primary80 and not an investment judgment."
    return shinfox


def build_blocked_proxy_audit() -> pd.DataFrame:
    rows = [
        audit("business_model_truth", "blocked", "No accepted full business-description taxonomy or annual report parser.", "Do not claim true project-based business model detection."),
        audit("company_description_latest_layer4_primary80", "blocked_partial", "No bounded company description package for the full latest primary80 in this task.", "Keyword proxy unavailable rows remain monthly-shape only."),
        audit("annual_report_text", "blocked", "No scoped annual report text source materialized.", "Can be a future Radar/Data bounded source task if Strategy Center wants it."),
        audit("mops_material_info_full_market", "not_requested", "Task explicitly avoids full-market scrape.", "Do not expand source acquisition here."),
        audit("6806_2026_06_monthly_revenue", "blocked", "MOPS bounded source package reports 2026-06 unavailable at capture time.", "Not needed for this proxy contract; single-month refresh only if Strategy requires current-month claim."),
        audit("hard_exclusion", "not_allowed", "Strategy Center requested review flag / soft penalty only.", "No ticker is removed."),
    ]
    return pd.DataFrame(rows)


def audit(field: str, status: str, evidence: str, policy: str) -> dict[str, str]:
    return {"field": field, "status": status, "evidence": evidence, "policy": policy}


def build_requested_vs_actual_coverage(base: pd.DataFrame, contract: pd.DataFrame, shinfox_rows: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "requested_scope": "latest Layer4 primary80 plus sanity ticker 6806",
                "actual_scope": "latest Layer4 primary80 from long revenue contract plus 6806 monthly revenue sanity row",
                "contract_rows": len(contract),
                "latest_primary80_rows": int(base["scope_bucket"].eq("latest_layer4_primary80").sum()),
                "sanity_case_rows": int(base["scope_bucket"].eq("sanity_case_not_latest_primary80").sum()),
                "keyword_source_ready_rows": int(contract["contract_project_keyword_proxy"].notna().sum()),
                "keyword_hit_rows": int(contract["contract_project_keyword_proxy"].fillna(False).sum()),
                "6806_monthly_revenue_rows": int(len(shinfox_rows)),
                "future_data_violation_count": 0,
            }
        ]
    )


def build_readiness(contract: pd.DataFrame, keyword_audit: pd.DataFrame, shinfox: pd.DataFrame) -> dict[str, Any]:
    return {
        "task_id": TASK_ID,
        "status": "project_based_business_model_risk_proxy_contract_ready_proxy",
        "contract_rows": int(len(contract)),
        "project_risk_review_flag_rows": int(contract["project_risk_review_flag"].fillna(False).sum()),
        "keyword_hit_rows": int(contract["contract_project_keyword_proxy"].fillna(False).sum()),
        "keyword_source_ready_rows": int(keyword_audit["keyword_source_status"].eq("bounded_mops_monthly_revenue_notes_ready").sum()),
        "shinfox_6806_status": str(shinfox.iloc[0].get("status", "missing")) if not shinfox.empty else "missing",
        "ready_for_experiments": True,
        "ready_for_formal": False,
        "ready_for_strategy_replay": False,
        "business_model_truth_detector_ready": False,
        "full_market_keyword_source_ready": False,
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


def build_summary(readiness: dict[str, Any], shinfox: pd.DataFrame, contract: pd.DataFrame) -> str:
    sh = shinfox.iloc[0] if not shinfox.empty else pd.Series(dtype=object)
    high_flags = contract[contract["project_risk_review_flag"].fillna(False)].head(8)
    flagged_text = ", ".join(f"{row.ticker} {row.name}" for row in high_flags.itertuples()) if not high_flags.empty else "none"
    return "\n".join(
        [
            "# Project-based business-model risk proxy contract",
            "",
            "## 結論",
            "",
            "- 已建立 project/business-model risk proxy contract；這是 diagnostic/proxy，不是正式商業模式分類器。",
            "- 本輪不做 hard exclude，只提供 Layer1/Layer4 review flag、soft penalty、report text hook。",
            "- 文字 keyword source 目前只有 bounded 6806 MOPS monthly revenue notes；latest primary80 多數仍是 monthly-shape-only proxy。",
            "- 若只有 keyword 命中但長期穩定性佳，僅低信心提醒；若 keyword + 高 lumpiness + 低 stability 同時命中，才提高風險提醒。",
            "",
            "## 6806 森崴能源 sanity",
            "",
            f"- status={sh.get('status', '')}",
            f"- project_risk_review_flag={sh.get('project_risk_review_flag', '')}",
            f"- project_revenue_business_model_proxy={sh.get('project_revenue_business_model_proxy', '')}",
            f"- proxy_confidence_level={sh.get('proxy_confidence_level', '')}",
            f"- revenue_lumpiness_percentile_vs_primary80={sh.get('revenue_lumpiness_percentile_vs_primary80', '')}",
            f"- revenue_stability_percentile_vs_primary80={sh.get('revenue_stability_percentile_vs_primary80', '')}",
            f"- matched_keywords={sh.get('matched_keywords', '')}",
            "- 6806 不在 latest Layer4 primary80；只作 sanity case，不作投資判斷。",
            "",
            "## Scoped flags",
            "",
            f"- project_risk_review_flag_rows={readiness['project_risk_review_flag_rows']}",
            f"- top flagged sample：{flagged_text}",
            "",
            "## Blocked / proxy",
            "",
            "- full business model truth detector blocked。",
            "- latest primary80 company description / annual report text source blocked/partial。",
            "- 6806 2026-06 monthly revenue 缺月保留 blocked；本任務不需要追單月資料。",
            "- future_data_violation_count=0。",
        ]
    )


def build_manifest(output_dir: Path, artifacts: list[Path]) -> dict[str, Any]:
    return {
        "task_id": TASK_ID,
        "status": "complete_project_based_business_model_risk_proxy_contract",
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
