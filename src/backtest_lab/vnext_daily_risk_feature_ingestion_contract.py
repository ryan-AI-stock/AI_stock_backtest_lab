from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import pandas as pd


TASK_ID = "TASK-BACKTEST-CORE-VNEXT-DAILY-RISK-FEATURE-INGESTION-CONTRACT-001"
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO_ROOT / "outputs/vnext_daily_risk_feature_ingestion_contract_20260711"
FLAGS = {
    "formal_model_changed": False, "trade_decision_changed": False,
    "active_in_trade_decision": False, "report_changed": False,
    "portfolio_replay_executed": False, "ready_for_strategy_replay": False,
    "ready_for_formal": False, "not_live_rule": True,
    "forward_returns_live_rule_usage": False,
}


FAMILIES = [
    ("official_raw_execution_OHLCV", "P3-1|P3-2", True, "ticker_daily", "same trading day post-close", 0, "execution/mark/cost", "official"),
    ("event_adjusted_analysis_OHLC", "P3-1|P3-2", True, "ticker_daily", "decision cutoff before scoring", 0, "KD/MA/BIAS/RS", "trusted_nonofficial_research_grade"),
    ("corporate_action_event_guard", "P3-1|P3-2", True, "ticker_event", "announcement/effective time PIT", 0, "analysis-price contamination guard", "official_or_explicit_block"),
    ("foreign_ownership_ratio_change", "P3-1|P3-2", False, "ticker_daily_or_latest", "published before cutoff", 5, "proxy score confidence", "source_specific"),
    ("three_institutional_flow_5_10_20D", "P3-1|P3-2", True, "ticker_daily", "same day post-close", 0, "flow/continuity proxy", "official"),
    ("margin_short_lending_crowding", "P3-1|P3-2", True, "ticker_daily", "same day post-close or documented lag", 1, "crowding/risk proxy", "official"),
    ("TDCC_holder_bucket_weekly_change", "P3-2", True, "ticker_weekly", "release_at plus documented publication lag", 10, "large/small holder proxy", "official"),
    ("TAIFEX_OI_foreign_net", "P3-1|P3-2", True, "market_daily", "Taiwan cutoff before decision", 1, "market threshold context", "official"),
    ("global_market_context", "P3-1|P3-2", False, "market_session", "latest completed session before Taiwan decision cutoff", 4, "market threshold confidence", "trusted_market_data"),
]


def _csv(rows: list[dict], path: Path) -> None:
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def _family_rows() -> list[dict]:
    return [{"family": f, "contract": c, "mandatory": m, "grain": g, "PIT_release_policy": p,
             "max_stale_calendar_days": s, "allowed_use": u, "required_source_quality": q,
             "weight_fixed": False, "silent_fill_allowed": False} for f, c, m, g, p, s, u, q in FAMILIES]


def _load_manifest(path: Path | None, radar_repo: Path | None, radar_git_ref: str, radar_manifest_repo_path: str) -> dict | pd.DataFrame | None:
    if path is not None:
        if path.suffix.lower() == ".json":
            return json.loads(path.read_text(encoding="utf-8-sig"))
        return pd.read_csv(path, dtype=str).fillna("")
    if radar_repo is not None and radar_git_ref:
        raw = subprocess.run(
            ["git", "show", f"{radar_git_ref}:{radar_manifest_repo_path}"], cwd=radar_repo,
            check=True, capture_output=True, text=True, encoding="utf-8",
        ).stdout
        return json.loads(raw)
    return None


def _evaluate_manifest(payload: dict | pd.DataFrame | None) -> tuple[pd.DataFrame, str, dict]:
    schema = pd.DataFrame(_family_rows())
    audit = {"requested_date": "", "source_manifest_status": "not_supplied", "calendar_state": "", "market_closed": False, "source_rows": 0}
    if payload is None:
        schema["availability"] = "awaiting_radar_manifest"
        schema["confidence"] = "unknown"
        schema["blocked_reason"] = "source package not handed off"
        return schema, "ready_for_radar_handoff", audit
    if isinstance(payload, dict):
        audit.update({"requested_date": payload.get("requested_date", ""), "source_manifest_status": payload.get("status", ""),
                      "calendar_state": payload.get("calendar_state", ""), "market_closed": payload.get("status") == "skipped_market_closed",
                      "source_rows": len(payload.get("sources", [])), "future_data_violation_count": payload.get("future_data_violation_count", 0)})
        manifest = pd.DataFrame(payload.get("sources", [])).fillna("")
        if audit["market_closed"]:
            valid_no_rows = (len(manifest) == 2 and set(manifest.get("market", [])) == {"TWSE", "TPEx"}
                             and manifest.get("family", pd.Series(dtype=str)).eq("official_raw_execution_ohlcv").all()
                             and manifest.get("status", pd.Series(dtype=str)).eq("no_rows").all()
                             and pd.to_numeric(manifest.get("http_status", pd.Series(dtype=float)), errors="coerce").eq(200).all())
            if not valid_no_rows:
                raise ValueError("market-closed manifest lacks valid TWSE/TPEx official no_rows evidence")
            schema["availability"] = "not_applicable_market_closed"
            schema["confidence"] = "high"
            schema["blocked_reason"] = ""
            return schema, "market_closed_no_signal", audit
        manifest = manifest.rename(columns={"requested_date": "market_date", "actual_source_date": "source_date", "retrieved_at_utc": "retrieved_at"})
        if "release_at" not in manifest: manifest["release_at"] = ""
        if "source_quality" not in manifest: manifest["source_quality"] = "source_manifest_lineage"
    else:
        manifest = payload.copy()
    required = {"family", "market_date", "source_date", "release_at", "retrieved_at", "source_quality", "status"}
    absent = sorted(required - set(manifest.columns))
    if absent:
        raise ValueError(f"Radar daily manifest missing columns: {absent}")
    latest = manifest.sort_values(["family", "market_date", "retrieved_at"]).drop_duplicates("family", keep="last")
    joined = schema.merge(latest, on="family", how="left")
    joined["availability"] = joined["status"].replace("", "missing").fillna("missing")
    joined["confidence"] = joined.apply(lambda r: "high" if r.availability == "ready" and r.source_quality else "reduced" if r.availability == "partial" else "none", axis=1)
    joined["blocked_reason"] = joined.apply(lambda r: "" if r.availability == "ready" else f"{r.family}:{r.availability}", axis=1)
    mandatory_blocked = joined.mandatory & joined.availability.ne("ready")
    state = "blocked" if mandatory_blocked.any() else "partial" if joined.availability.ne("ready").any() else "data_ready"
    return joined, state, audit


def run(output_dir: Path = DEFAULT_OUTPUT, radar_manifest: Path | None = None, radar_repo: Path | None = None,
        radar_git_ref: str = "", radar_manifest_repo_path: str = "") -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "current_step.txt").write_text("building_daily_risk_ingestion_schema", encoding="utf-8")
    payload = _load_manifest(radar_manifest, radar_repo, radar_git_ref, radar_manifest_repo_path)
    availability, state, ingest_audit = _evaluate_manifest(payload)
    availability.to_csv(output_dir / "daily_risk_feature_family_availability_contract.csv", index=False, encoding="utf-8-sig")
    _csv([ingest_audit], output_dir / "daily_risk_manifest_ingestion_audit.csv")
    _csv([{
        "radar_pipeline_commit": "09aaeef", "radar_persisted_manifest_commit": "7bb60f5",
        "actions_run_url": "https://github.com/ryan-AI-stock/AI_stock_rotation_radar/actions/runs/29147702136",
        "actions_conclusion": "success", "validated_path": "market_closed_no_signal",
        "open_trading_day_full_family_validation_ready": False,
    }], output_dir / "radar_daily_risk_pipeline_source_audit.csv")

    _csv([
        {"canonical_name": "institutional_large_holder_chip_proxy_score", "display_name_zh": "法人／大戶籌碼代理分數", "exact_institution_vs_retail_ratio": False, "weight_fixed": False,
         "components": "foreign_ownership_ratio_change|three_institutional_flow_5_10_20D|margin_short_lending_crowding|TDCC_holder_bucket_weekly_change(P3-2_only)",
         "claim_boundary": "proxy only; must not claim precise institutional versus retail ownership ratio"}
    ], output_dir / "institutional_large_holder_chip_proxy_schema.csv")

    component_rows = [
        ("foreign_ownership_ratio_change", "P3-1|P3-2", "optional", "ownership context", "source-specific PIT release"),
        ("three_institutional_flow_5_10_20D", "P3-1|P3-2", "mandatory", "flow and continuity", "rolling windows use rows released by cutoff"),
        ("margin_short_lending_crowding", "P3-1|P3-2", "mandatory", "crowding/risk", "documented release lag"),
        ("TDCC_holder_bucket_weekly_change", "P3-2", "mandatory", "large/small bucket proxy", "weekly release_at; never backdate to observation week"),
    ]
    pd.DataFrame(component_rows, columns=["component", "contract", "requirement", "meaning", "PIT_policy"]).assign(weight="unset").to_csv(
        output_dir / "chip_proxy_component_schema.csv", index=False, encoding="utf-8-sig")

    market = ["TAIFEX_OI_foreign_net", "^DJI", "^IXIC", "^N225", "^KS11", "TWD=X", "^VIX"]
    _csv([{"series": item, "use": "market_threshold_context_only", "session_policy": "latest completed session before Taiwan decision cutoff",
           "missing_policy": "TAIFEX mandatory blocked; global series optional confidence downgrade", "weight_fixed": False} for item in market],
         output_dir / "market_threshold_context_schema.csv")

    _csv([
        {"field": "market_date", "meaning": "target Taiwan market date", "mandatory": True},
        {"field": "source_date", "meaning": "economic/market observation date", "mandatory": True},
        {"field": "release_at", "meaning": "first legally observable timestamp", "mandatory": True},
        {"field": "retrieved_at", "meaning": "cache retrieval metadata only", "mandatory": True},
        {"field": "decision_cutoff_at", "meaning": "Taiwan decision cutoff", "mandatory": True},
        {"field": "stale", "meaning": "source exceeds family stale policy", "mandatory": True},
        {"field": "blocked", "meaning": "mandatory unavailable or PIT invalid", "mandatory": True},
        {"field": "source_quality", "meaning": "official/trusted/proxy/blocked", "mandatory": True},
    ], output_dir / "daily_freshness_gate_schema.csv")

    _csv([
        {"rule": "release_cutoff", "condition": "release_at <= decision_cutoff_at", "failure": "blocked", "notes": "retrieved_at never substitutes release_at"},
        {"rule": "analysis_execution_separation", "condition": "analysis_price and raw_execution_price are separate columns", "failure": "blocked", "notes": "KD/MA/BIAS/RS use analysis price only"},
        {"rule": "mandatory_missing", "condition": "mandatory family ready", "failure": "blocked", "notes": "no silent fill"},
        {"rule": "optional_missing", "condition": "optional family may be unavailable", "failure": "partial", "notes": "lower confidence and report reason"},
        {"rule": "TDCC_lag", "condition": "TDCC row available only after actual release_at", "failure": "blocked_for_P3_2", "notes": "P3-1 excludes TDCC by contract"},
        {"rule": "global_session", "condition": "session completed before Taiwan cutoff", "failure": "partial", "notes": "never use later same-calendar-date close"},
    ], output_dir / "daily_risk_PIT_policy.csv")

    _csv([
        {"contract": "P3-1", "TDCC_used": False, "comparison_role": "full lifecycle without TDCC", "same_period_AB_required": False},
        {"contract": "P3-2_no_TDCC_arm", "TDCC_used": False, "comparison_role": "same-period A arm", "same_period_AB_required": True},
        {"contract": "P3-2_with_TDCC_arm", "TDCC_used": True, "comparison_role": "same-period B arm", "same_period_AB_required": True},
    ], output_dir / "p3_1_p3_2_AB_contract.csv")

    _csv([
        {"field": "data_status", "value_source": "data_ready|partial|blocked"},
        {"field": "chip_proxy_label", "value_source": "法人／大戶籌碼代理分數"},
        {"field": "chip_proxy_confidence", "value_source": "component availability and PIT quality"},
        {"field": "analysis_price_source_quality", "value_source": "adjusted analysis manifest"},
        {"field": "corporate_action_event_status", "value_source": "event guard"},
        {"field": "stale_or_blocked_families", "value_source": "freshness gate"},
        {"field": "diagnostic_warning", "value_source": "not live rule; no trade decision"},
    ], output_dir / "daily_report_risk_feature_hooks.csv")

    blocked = availability[~availability.availability.isin(["ready", "not_applicable_market_closed"])].copy()
    blocked.to_csv(output_dir / "daily_risk_feature_blocked_ledger.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(columns=["market_date", "family", "violation_reason"]).to_csv(output_dir / "future_data_audit.csv", index=False, encoding="utf-8-sig")
    readiness = {"task_id": TASK_ID, "status": state, "ready_for_radar_handoff": True,
                 "ready_for_daily_manifest_absorption": True, "daily_manifest_supplied": payload is not None,
                 "requested_date": ingest_audit.get("requested_date", ""), "market_closed": ingest_audit.get("market_closed", False),
                 "market_closed_no_signal": state == "market_closed_no_signal",
                 "radar_actions_manual_validation_success": True,
                 "open_trading_day_full_family_validation_ready": False,
                 "weights_fixed": False, "partial_backtest_executed": False, "ready_for_experiments": False,
                 "precise_institution_retail_ratio_claimed": False, "P3_2_same_period_TDCC_AB_required": True,
                 "future_data_violation_count": 0, **FLAGS}
    (output_dir / "readiness_for_daily_risk_feature_ingestion_contract.json").write_text(json.dumps(readiness, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "final_summary_zh.md").write_text(
        "# Daily risk feature ingestion contract\n\n"
        f"- Status: `{state}`；目前只建立 schema/PIT/freshness gate，未 materialize、未回測。\n"
        "- 正式名稱為「法人／大戶籌碼代理分數」，不宣稱精確法人與散戶比例，權重未設定。\n"
        "- P3-1 不含 TDCC；P3-2 必須同期間有/無 TDCC A/B，TDCC 依實際 release_at 對齊。\n"
        "- Analysis price 用於 KD/MA/BIAS/RS；官方 raw price 僅用 execution，兩者不可混欄。\n"
        "- Mandatory 缺欄/PIT 無效為 blocked；optional 缺欄為 partial 並降低 confidence。\n",
        encoding="utf-8")
    files = sorted(p for p in output_dir.iterdir() if p.is_file() and p.name != "manifest.json")
    manifest = {"task_id": TASK_ID, "runner": str(Path(__file__).resolve()), "input_radar_manifest": str(radar_manifest or ""),
                "input_radar_git_ref": radar_git_ref, "input_radar_manifest_repo_path": radar_manifest_repo_path,
                "readiness": readiness, "files": [{"name": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in files]}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "current_step.txt").write_text("ready_for_radar_complete_family_handoff_no_partial_backtest", encoding="utf-8")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--radar-manifest", type=Path)
    parser.add_argument("--radar-repo", type=Path)
    parser.add_argument("--radar-git-ref", default="")
    parser.add_argument("--radar-manifest-repo-path", default="")
    args = parser.parse_args()
    print(run(args.output_dir, args.radar_manifest, args.radar_repo, args.radar_git_ref, args.radar_manifest_repo_path))


if __name__ == "__main__":
    main()
