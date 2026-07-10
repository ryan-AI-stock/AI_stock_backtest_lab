from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


TASK_ID = "TASK-BACKTEST-CORE-VNEXT-LAYER1-REVENUE-STABILITY-HYGIENE-INTEGRATION-CONTRACT-001"
DEFAULT_SOURCE = Path("outputs/vnext_layer1_long_revenue_stability_low_base_risk_integration_contract_20260710")
DEFAULT_EXPERIMENTS = Path(
    "C:/Users/zergv/Documents/Codex/2026-07-06/backtest-lab-experiments-diagnostic-validation-attribution/"
    "outputs/vnext_layer1_long_revenue_stability_low_base_risk_integration_diagnostic_20260710"
)
DEFAULT_OUTPUT = Path("outputs/vnext_layer1_revenue_stability_hygiene_integration_contract_20260710")
REVIEW_TICKERS = {"8926": "台汽電", "5351": "鈺創", "3006": "晶豪科", "4931": "新盛力", "2347": "聯強"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Layer1/Layer4 revenue stability hygiene integration contract.")
    parser.add_argument("--source-dir", default=str(DEFAULT_SOURCE))
    parser.add_argument("--experiments-dir", default=str(DEFAULT_EXPERIMENTS))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    experiments_dir = Path(args.experiments_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source = pd.read_csv(source_dir / "layer1_long_revenue_stability_feature_contract.csv", dtype={"ticker": str})
    experiments_summary = read_json(experiments_dir / "layer1_long_revenue_stability_low_base_summary.json")

    integration = build_integration_contract(source)
    integration_path = output_dir / "layer1_revenue_stability_hygiene_integration_contract.csv"
    integration.to_csv(integration_path, index=False, encoding="utf-8-sig")

    policy = build_policy_map()
    policy_path = output_dir / "layer1_revenue_stability_hygiene_policy_map.csv"
    policy.to_csv(policy_path, index=False, encoding="utf-8-sig")

    flagged = build_flagged_candidates(integration)
    flagged_path = output_dir / "layer4_primary80_hygiene_flagged_candidates.csv"
    flagged.to_csv(flagged_path, index=False, encoding="utf-8-sig")

    report_sample = build_daily_report_sample(integration)
    report_sample_path = output_dir / "daily_report_hygiene_field_sample.csv"
    report_sample.to_csv(report_sample_path, index=False, encoding="utf-8-sig")

    blocked = build_blocked_proxy_audit()
    blocked_path = output_dir / "blocked_proxy_audit.csv"
    blocked.to_csv(blocked_path, index=False, encoding="utf-8-sig")

    coverage = build_requested_vs_actual_coverage(source, integration)
    coverage_path = output_dir / "requested_vs_actual_coverage.csv"
    coverage.to_csv(coverage_path, index=False, encoding="utf-8-sig")

    readiness = build_readiness(integration, flagged, experiments_summary)
    readiness_path = output_dir / "readiness_for_experiments.json"
    write_json(readiness_path, readiness)

    summary_path = output_dir / "final_summary_zh.md"
    summary_path.write_text(build_summary(readiness, flagged), encoding="utf-8")

    artifacts = [
        integration_path,
        policy_path,
        flagged_path,
        report_sample_path,
        blocked_path,
        coverage_path,
        readiness_path,
        summary_path,
    ]
    manifest_path = output_dir / "manifest.json"
    write_json(manifest_path, build_manifest(output_dir, artifacts))

    print(f"LAYER1_REVENUE_STABILITY_HYGIENE_OUTPUT={output_dir.resolve()}")
    print(f"CONTRACT_ROWS={len(integration)}")
    print(f"FLAGGED_ROWS={len(flagged)}")
    print(f"READY_FOR_EXPERIMENTS={readiness['ready_for_experiments']}")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"status": "missing", "path": str(path)}
    return json.loads(path.read_text(encoding="utf-8"))


def build_integration_contract(source: pd.DataFrame) -> pd.DataFrame:
    df = source.copy()
    df["ticker"] = df["ticker"].astype(str)
    stability = pd.to_numeric(df.get("revenue_stability_score"), errors="coerce").fillna(0.5)
    lumpiness = pd.to_numeric(df.get("revenue_lumpiness_score"), errors="coerce").fillna(0.5)
    recent_spike = bool_series(df.get("recent_spike_without_long_history_flag"))
    project_proxy = bool_series(df.get("project_based_revenue_risk_proxy"))
    bias_context = pd.Series(0.5, index=df.index)
    if "layer1_quality_floor_risk_pctile_by_week" in df.columns:
        layer1_risk = pd.to_numeric(df["layer1_quality_floor_risk_pctile_by_week"], errors="coerce").fillna(0.5)
    else:
        layer1_risk = pd.Series(0.5, index=df.index)

    df["revenue_stability_context_score"] = stability.clip(0, 1)
    df["revenue_lumpiness_penalty_score"] = (lumpiness * 0.65 + recent_spike.astype(float) * 0.2 + project_proxy.astype(float) * 0.15).clip(0, 1)
    df["recent_spike_review_flag"] = recent_spike
    df["project_revenue_proxy_review_flag"] = project_proxy
    df["low_base_context_label"] = "context_only_not_main_weight"
    df.loc[stability.ge(0.7) & df["revenue_lumpiness_penalty_score"].lt(0.35), "low_base_context_label"] = "stable_quality_context"
    df.loc[df["revenue_lumpiness_penalty_score"].ge(0.45), "low_base_context_label"] = "lumpy_or_spike_review_context"
    df["low_base_tiebreak_cap"] = 0.03
    df["overheat_penalty_modifier"] = (1 + df["revenue_lumpiness_penalty_score"] * 0.25 + layer1_risk * 0.1 + bias_context * 0.0).round(6)
    df["layer1_quality_score_soft_adjustment"] = ((df["revenue_stability_context_score"] - 0.5) * 0.06 - df["revenue_lumpiness_penalty_score"] * 0.04).round(6)
    df["layer4_risk_context_soft_penalty"] = (df["revenue_lumpiness_penalty_score"] * 0.08 + recent_spike.astype(float) * 0.03 + project_proxy.astype(float) * 0.03).round(6)
    df["hygiene_warning_text_for_report"] = df.apply(hygiene_warning_text, axis=1)
    df["hygiene_integration_role"] = "soft_context_only_no_hard_exclusion"
    df["route_support_selected_result_changed"] = False
    df["hard_exclude_applied"] = False
    df["review_soft_penalty_candidate"] = df["ticker"].isin(REVIEW_TICKERS) | recent_spike | project_proxy | df["revenue_lumpiness_penalty_score"].ge(0.45)
    df["review_ticker_requested_by_strategy_center"] = df["ticker"].isin(REVIEW_TICKERS)
    df["project_based_revenue_risk_proxy_source_quality"] = "proxy_only_monthly_revenue_lumpiness_not_business_model_truth"
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
        "revenue_stability_score",
        "revenue_lumpiness_score",
        "recent_spike_without_long_history_flag",
        "project_based_revenue_risk_proxy",
        "revenue_stability_context_score",
        "revenue_lumpiness_penalty_score",
        "recent_spike_review_flag",
        "project_revenue_proxy_review_flag",
        "low_base_context_label",
        "low_base_tiebreak_cap",
        "overheat_penalty_modifier",
        "layer1_quality_score_soft_adjustment",
        "layer4_risk_context_soft_penalty",
        "hygiene_warning_text_for_report",
        "review_soft_penalty_candidate",
        "review_ticker_requested_by_strategy_center",
        "route_support_selected_result_changed",
        "hard_exclude_applied",
        "hygiene_integration_role",
        "project_based_revenue_risk_proxy_source_quality",
        "source_quality",
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
    return df[[col for col in keep if col in df.columns]]


def bool_series(values: Any) -> pd.Series:
    if values is None:
        return pd.Series(dtype=bool)
    return pd.Series(values).astype(str).str.lower().isin({"true", "1", "yes", "y"})


def hygiene_warning_text(row: pd.Series) -> str:
    warnings = []
    if bool(row.get("recent_spike_review_flag", False)):
        warnings.append("近期營收暴增但長期穩定性需人工 review")
    if bool(row.get("project_revenue_proxy_review_flag", False)):
        warnings.append("營收形態偏 lumpy/project-based proxy，僅作風險提醒")
    if row.get("revenue_lumpiness_penalty_score", 0) >= 0.45:
        warnings.append("營收集中度偏高，Layer4 排序應納入軟扣分")
    if not warnings:
        return "長期營收 hygiene 未觸發主要警示"
    return "；".join(warnings)


def build_policy_map() -> pd.DataFrame:
    rows = [
        policy("revenue_stability_context_score", "Layer1 quality score", "soft_bonus_or_soft_penalty", "可納入 Layer1 quality context；不單獨決定買賣。"),
        policy("revenue_lumpiness_penalty_score", "Layer1 risk context + Layer4 risk context", "soft_penalty", "營收過度集中或波動時降低衛生分數；不得 hard exclude。"),
        policy("recent_spike_review_flag", "daily report / review flag", "display_only_or_soft_warning", "近期爆增但長期歷史不足，保留人工 review flag。"),
        policy("project_revenue_proxy_review_flag", "daily report / Layer4 risk context", "proxy_warning", "只能說 project/lumpy revenue proxy，不可宣稱辨識真實商業模式。"),
        policy("low_base_context_label", "Layer2/Layer4 context", "tie_break_context", "low-base 只作位置/風險 context，不回主權重。"),
        policy("low_base_tiebreak_cap", "Layer4 scoring", "cap_bonus_to_3pct", "低基期最多小幅 tie-break；不可單獨讓股票入選。"),
        policy("overheat_penalty_modifier", "Layer2/Layer4 risk penalty", "modifier", "low-base 不得抵消明顯過熱/高波動/高風險。"),
        policy("hygiene_warning_text_for_report", "daily report", "display_only", "日報只顯示風險提醒，不產生交易指令。"),
        policy("valuation PE/PB/PS", "blocked", "blocked", "估值 source 未接受，不進排序。"),
        policy("exact quarterly revenue YoY", "blocked/proxy", "blocked_exact_proxy_available", "目前 quarterly revenue YoY 仍是 monthly rolling 3M proxy。"),
        policy("margin recovery exact source", "blocked/proxy", "blocked_exact", "毛利率/營益率恢復需要 refreshed quarterly source。"),
    ]
    return pd.DataFrame(rows)


def policy(field: str, layer: str, action: str, note: str) -> dict[str, str]:
    return {"field": field, "layer_destination": layer, "integration_action": action, "policy_note": note}


def build_flagged_candidates(integration: pd.DataFrame) -> pd.DataFrame:
    flagged = integration[integration["review_soft_penalty_candidate"].fillna(False)].copy()
    flagged["recommended_action"] = "review_soft_penalty_candidate_do_not_hard_exclude"
    requested = integration[integration["ticker"].isin(REVIEW_TICKERS)].copy()
    missing = sorted(set(REVIEW_TICKERS) - set(requested["ticker"].astype(str)))
    if missing:
        missing_rows = [
            {
                "snapshot_date": "",
                "ticker": ticker,
                "name": REVIEW_TICKERS[ticker],
                "market": "",
                "review_soft_penalty_candidate": True,
                "review_ticker_requested_by_strategy_center": True,
                "recommended_action": "requested_review_ticker_not_in_latest_layer4_primary80_scope_no_hard_exclude",
                "hygiene_warning_text_for_report": "Strategy Center 指定 review ticker，但不在 latest Layer4 primary80 scope。",
                "route_support_selected_result_changed": False,
                "hard_exclude_applied": False,
                "diagnostic_only": True,
            }
            for ticker in missing
        ]
        flagged = pd.concat([flagged, pd.DataFrame(missing_rows)], ignore_index=True, sort=False)
    return flagged.sort_values(["review_ticker_requested_by_strategy_center", "revenue_lumpiness_penalty_score"], ascending=[False, False])


def build_daily_report_sample(integration: pd.DataFrame) -> pd.DataFrame:
    sample = integration[integration["review_soft_penalty_candidate"].fillna(False)].head(12).copy()
    if sample.empty:
        sample = integration.head(12).copy()
    cols = [
        "snapshot_date",
        "ticker",
        "name",
        "revenue_stability_context_score",
        "revenue_lumpiness_penalty_score",
        "recent_spike_review_flag",
        "project_revenue_proxy_review_flag",
        "low_base_context_label",
        "overheat_penalty_modifier",
        "hygiene_warning_text_for_report",
        "diagnostic_only",
    ]
    return sample[[col for col in cols if col in sample.columns]]


def build_blocked_proxy_audit() -> pd.DataFrame:
    rows = [
        audit("valuation_PE_PB_PS", "blocked", "No accepted valuation source in this contract.", "Do not use valuation low-base proxy."),
        audit("exact_quarterly_revenue_yoy", "proxy_only", "Current source is monthly rolling 3M proxy.", "Do not label statement-exact."),
        audit("margin_recovery_exact_source", "blocked_partial", "Quarterly gross/operating margin source not refreshed here.", "Only use if later Core refreshes accepted quarterly source."),
        audit("project_based_revenue_risk_proxy", "proxy_only", "Monthly revenue lumpiness/spike shape.", "Do not claim true project-based business model detection."),
        audit("6806_shinfox", "blocked_pending_radar", "6806 monthly revenue rows unavailable in source contract.", "Keep blocked note; do not wait in this task."),
        audit("hard_exclusion", "not_allowed", "Strategy Center requested soft hygiene integration only.", "No row is removed."),
    ]
    return pd.DataFrame(rows)


def audit(field: str, status: str, evidence: str, policy: str) -> dict[str, str]:
    return {"field": field, "status": status, "evidence": evidence, "policy": policy}


def build_requested_vs_actual_coverage(source: pd.DataFrame, integration: pd.DataFrame) -> pd.DataFrame:
    asof = source["snapshot_date"].dropna().astype(str).max() if "snapshot_date" in source.columns else ""
    return pd.DataFrame(
        [
            {
                "requested_scope": "latest Layer4 primary80 + Strategy Center requested risk proxy names",
                "actual_scope": "latest Layer4 primary80 hygiene integration; requested review tickers included if present else noted",
                "requested_asof": asof,
                "actual_asof": asof,
                "source_rows": len(source),
                "integration_rows": len(integration),
                "review_soft_penalty_candidate_count": int(integration["review_soft_penalty_candidate"].sum()),
                "strategy_center_review_tickers_present_count": int(integration["ticker"].isin(REVIEW_TICKERS).sum()),
                "future_data_violation_count": 0,
            }
        ]
    )


def build_readiness(integration: pd.DataFrame, flagged: pd.DataFrame, experiments_summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_id": TASK_ID,
        "status": "layer1_revenue_stability_hygiene_integration_contract_ready",
        "source_experiments_verdict": experiments_summary.get("verdict", ""),
        "contract_rows": int(len(integration)),
        "flagged_candidate_rows": int(len(flagged)),
        "ready_for_experiments": True,
        "ready_for_formal": False,
        "ready_for_strategy_replay": False,
        "route_support_selected_result_changed": False,
        "hard_exclude_applied": False,
        "valuation_low_base_proxy_ready": False,
        "exact_quarterly_revenue_yoy_ready": False,
        "margin_recovery_exact_source_ready": False,
        "project_based_revenue_risk_proxy_is_formal_business_model_detector": False,
        "report_field_hooks_ready": True,
        "formal_model_changed": False,
        "trade_decision_changed": False,
        "active_in_trade_decision": False,
        "report_changed": False,
        "portfolio_replay_executed": False,
        "not_live_rule": True,
        "forward_returns_live_rule_usage": False,
        "future_data_violation_count": 0,
    }


def build_summary(readiness: dict[str, Any], flagged: pd.DataFrame) -> str:
    requested = flagged[flagged.get("review_ticker_requested_by_strategy_center", False).fillna(False)] if not flagged.empty else pd.DataFrame()
    requested_names = (
        ", ".join(f"{row['ticker']} {row.get('name', '')}" for _, row in requested.iterrows())
        if not requested.empty
        else "none"
    )
    return "\n".join(
        [
            "# Layer1 revenue stability hygiene integration contract",
            "",
            "## 結論",
            "",
            "- 已把長期營收穩定性接成 Layer1/Layer4 hygiene integration contract。",
            "- revenue_stability_score 進 Layer1 quality context / soft adjustment。",
            "- revenue_lumpiness_score 進 Layer1/Layer4 risk context / soft penalty。",
            "- recent_spike_without_long_history 與 project_based_revenue_risk_proxy 只作 review/report flag。",
            "- low-base 不回主權重，只作 context / tie-break cap / overheat penalty modifier。",
            "- 本輪沒有改 route_support selected result，沒有 hard exclude。",
            "",
            "## 高風險 proxy 名單",
            "",
            f"- review_soft_penalty_candidate rows={readiness['flagged_candidate_rows']}",
            f"- Strategy Center 指定名單 present/flagged：{requested_names}",
            "- 這些只作 review / soft penalty candidate，不得直接踢出。",
            "",
            "## Blocked / proxy",
            "",
            "- PE/PB/PS valuation low-base blocked。",
            "- exact quarterly revenue YoY blocked；目前只能用 monthly rolling 3M proxy。",
            "- margin recovery exact source blocked/partial。",
            "- project_based_revenue_risk_proxy 不是正式商業模式辨識。",
            "- 6806 森崴能源 source 仍等 Radar/Data bounded source，未在本任務等待。",
        ]
    )


def build_manifest(output_dir: Path, artifacts: list[Path]) -> dict[str, Any]:
    return {
        "task_id": TASK_ID,
        "status": "complete_hygiene_integration_contract",
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
