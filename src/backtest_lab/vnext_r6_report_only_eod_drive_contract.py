from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


TASK_ID = "TASK-BACKTEST-CORE-VNEXT-R6-REPORT-ONLY-DAILY-EOD-REFRESH-AND-DRIVE-OVERWRITE-CONTRACT-001"
DEFAULT_REPORT_DATE = "2026-07-10"
DEFAULT_REQUESTED_DATE = "2026-07-10"
DEFAULT_OUTPUT = Path("outputs/vnext_r6_report_only_daily_eod_refresh_drive_overwrite_contract_20260710")
DEFAULT_DRAFT = Path("outputs/vnext_r6_daily_report_only_contract_20260710")
DEFAULT_BLOCKER_MAP = Path(
    "C:/Users/zergv/Documents/Codex/2026-07-06/strategy-center-core-experiments-research-materials/"
    "outputs/vnext_r6_daily_report_readiness_blocker_map_20260710"
)
DEFAULT_EXPERIMENTS = Path(
    "C:/Users/zergv/Documents/Codex/2026-07-06/backtest-lab-experiments-diagnostic-validation-attribution/"
    "outputs/vnext_r6_guard_first_market_bias_override_unified_diagnostic_20260710"
)
DEFAULT_LAYER1_REVENUE = Path(
    "C:/Users/zergv/Documents/Codex/2026-07-06/backtest-lab-experiments-diagnostic-validation-attribution/"
    "outputs/vnext_layer1_revenue_horizon_extension_diagnostic_20260710"
)
DRIVE_FOLDER_ID = "16SmfPgMMIs7MWteeX1h2EkhSIEaGvpHn"
DRIVE_FOLDER_URL = f"https://drive.google.com/drive/u/0/folders/{DRIVE_FOLDER_ID}"
DRIVE_FILENAME = "vNext台股AI模型訊號追蹤_每日報告.pdf"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build R6 report-only daily EOD refresh and Drive overwrite contract.")
    parser.add_argument("--draft-dir", default=str(DEFAULT_DRAFT))
    parser.add_argument("--blocker-map-dir", default=str(DEFAULT_BLOCKER_MAP))
    parser.add_argument("--experiments-dir", default=str(DEFAULT_EXPERIMENTS))
    parser.add_argument("--layer1-revenue-dir", default=str(DEFAULT_LAYER1_REVENUE))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--report-date", default=DEFAULT_REPORT_DATE)
    parser.add_argument("--requested-date", default=DEFAULT_REQUESTED_DATE)
    args = parser.parse_args()

    draft_dir = Path(args.draft_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    draft_latest = pd.read_csv(draft_dir / "r6_daily_report_latest_row.csv").iloc[0].to_dict()
    draft_readiness = read_json(draft_dir / "readiness_for_r6_daily_report_only_contract.json")
    experiments = read_json(Path(args.experiments_dir) / "r6_unified_diagnostic_summary.json")

    latest_row = build_latest_eod_row(draft_latest, draft_readiness, args.report_date, args.requested_date)
    latest_path = output_dir / "r6_report_only_daily_row_latest.csv"
    pd.DataFrame([latest_row]).to_csv(latest_path, index=False, encoding="utf-8-sig")

    drive_policy = build_drive_overwrite_policy_contract()
    drive_path = output_dir / "r6_report_only_drive_overwrite_readiness.csv"
    drive_policy.to_csv(drive_path, index=False, encoding="utf-8-sig")
    drive_json_path = output_dir / "r6_report_only_drive_overwrite_readiness.json"
    write_json(drive_json_path, build_drive_readiness_payload())

    schedule_contract = build_schedule_rules_integration_contract(latest_row)
    schedule_path = output_dir / "r6_report_only_schedule_rules_integration_contract.csv"
    schedule_contract.to_csv(schedule_path, index=False, encoding="utf-8-sig")

    coverage = build_requested_vs_actual_coverage(latest_row)
    coverage_path = output_dir / "r6_report_only_daily_requested_vs_actual_coverage.csv"
    coverage.to_csv(coverage_path, index=False, encoding="utf-8-sig")

    blocked = build_blocked_proxy_audit(latest_row)
    blocked_path = output_dir / "r6_report_only_daily_blocked_proxy_audit.csv"
    blocked.to_csv(blocked_path, index=False, encoding="utf-8-sig")

    model_note = build_model_status_note(experiments, args.layer1_revenue_dir)
    model_note_path = output_dir / "r6_report_only_model_status_note.csv"
    model_note.to_csv(model_note_path, index=False, encoding="utf-8-sig")

    readiness = build_readiness(latest_row, draft_readiness, experiments)
    readiness_path = output_dir / "readiness_for_r6_report_only_daily_eod_refresh.json"
    write_json(readiness_path, readiness)

    html_path = output_dir / "r6_report_only_daily_sample.html"
    html_path.write_text(build_html_sample(latest_row, readiness), encoding="utf-8")

    summary_path = output_dir / "final_summary_zh.md"
    summary_path.write_text(build_summary(latest_row, readiness), encoding="utf-8")

    artifacts = [
        latest_path,
        drive_path,
        drive_json_path,
        schedule_path,
        coverage_path,
        blocked_path,
        model_note_path,
        readiness_path,
        html_path,
        summary_path,
    ]
    manifest_path = output_dir / "manifest.json"
    write_json(manifest_path, build_manifest(output_dir, artifacts))

    print(f"R6_REPORT_ONLY_EOD_DRIVE_CONTRACT={output_dir.resolve()}")
    print(f"READY_FOR_REPORT_ONLY_PDF_GENERATION={readiness['ready_for_report_only_pdf_generation']}")
    print(f"READY_FOR_DRIVE_OVERWRITE={readiness['ready_for_drive_overwrite']}")
    print(f"READY_FOR_SCHEDULE_INTEGRATION={readiness['ready_for_schedule_integration']}")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"status": "missing", "path": str(path)}
    return json.loads(path.read_text(encoding="utf-8"))


def build_latest_eod_row(
    draft: dict[str, Any],
    draft_readiness: dict[str, Any],
    report_date: str,
    requested_date: str,
) -> dict[str, Any]:
    actual_signal_date = str(draft.get("signal_date", ""))
    actual_report_date = report_date
    fallback_reason = ""
    if actual_signal_date != requested_date:
        fallback_reason = "latest_available_unified_contract_row_used;daily_eod_refresh_materialization_not_executed_in_this_contract"

    return {
        "task": TASK_ID,
        "report_date": report_date,
        "signal_date": actual_signal_date,
        "requested_date": requested_date,
        "actual_report_date": actual_report_date,
        "actual_signal_date": actual_signal_date,
        "data_asof_date": str(draft.get("data_asof_date", actual_signal_date)),
        "fallback_reason": fallback_reason,
        "regime_label": draft.get("regime_label", ""),
        "selected_branch": draft.get("selected_branch", ""),
        "branch_reason": draft.get("branch_reason", ""),
        "triggered_features": draft.get("triggered_features", ""),
        "c2_pass_flag": bool_field(draft.get("c2_pass_flag")),
        "consensus_trigger_flag": bool_field(draft.get("consensus_trigger_flag")),
        "r6_override_flag": bool_field(draft.get("r6_override_flag")),
        "p1_risk_veto_flag": bool_field(draft.get("p1_risk_veto_flag")),
        "selected_primary_asset_type": draft.get("selected_primary_asset_type", ""),
        "selected_ticker": draft.get("selected_ticker", ""),
        "selected_name": draft.get("selected_name", ""),
        "fallback_asset": draft.get("fallback_asset", ""),
        "rs20_top3_reference_tickers": draft.get("rs20_top3_reference_tickers", ""),
        "rs20_reference_only": bool_field(draft.get("rs20_reference_only", True)),
        "data_readiness": draft.get("data_readiness", ""),
        "blocked_reason": draft.get("blocked_reason", ""),
        "cost_model_status": draft.get("cost_model_status", ""),
        "diagnostic_warning": "report_only_not_live_trade_decision;no_drive_write;no_strategy_replay;not_formal",
        "source_draft_status": draft_readiness.get("status", ""),
        "latest_row_source_quality": "draft_contract_latest_available_row",
        "report_only": True,
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


def bool_field(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def build_drive_overwrite_policy_contract() -> pd.DataFrame:
    rows = [
        policy_row("target_folder_id", DRIVE_FOLDER_ID, "configured", "Use this single folder only; do not create dated folders."),
        policy_row("target_folder_url", DRIVE_FOLDER_URL, "configured", "Strategy Center provided target Drive folder."),
        policy_row("target_filename", DRIVE_FILENAME, "configured", "Fixed human-readable name; overwrite same file each trading day."),
        policy_row("filename_reason", "report-only signal tracking wording", "configured", "Avoids implying formal live trade decision while remaining understandable."),
        policy_row("overwrite_semantics", "prefer_file_id_update_then_update_by_name_else_create_once", "draft", "Use file-id if configured; otherwise find same name in folder and update; create only if no existing file."),
        policy_row("no_new_dated_folder", "true", "required", "Daily report must not create a folder per date."),
        policy_row("publish_authorization", "blocked_until_strategy_center_authorizes", "blocked", "This task does not upload to Drive."),
        policy_row("credential_policy", "reuse_existing_drive_publish_auth", "draft", "Use existing backtest_lab.drive_publish auth/env conventions."),
        policy_row("remote_mimetype", "application/pdf", "configured", "Only PDF overwrite is in scope."),
        policy_row("drive_write_this_task", "false", "required", "No Drive write performed."),
    ]
    return pd.DataFrame(rows)


def build_drive_readiness_payload() -> dict[str, Any]:
    return {
        "target_folder_id": DRIVE_FOLDER_ID,
        "target_folder_url": DRIVE_FOLDER_URL,
        "target_filename": DRIVE_FILENAME,
        "overwrite_semantics": "prefer_file_id_update_then_update_by_name_else_create_once",
        "no_new_dated_folder": True,
        "drive_write_this_task": False,
        "ready_for_drive_overwrite": False,
        "blocked_reason": "Strategy Center has not authorized Drive write; schedule/publish is contract-only in this task.",
        "formal_model_changed": False,
        "trade_decision_changed": False,
        "active_in_trade_decision": False,
        "report_changed": False,
    }


def build_schedule_rules_integration_contract(latest_row: dict[str, Any]) -> pd.DataFrame:
    rows = [
        schedule_row("trading_day_gate", "required", "Use AI_stock_schedule_rules trading day gate before running EOD report.", "draft_only"),
        schedule_row("run_timing", "required", "Run after Taiwan market close and after official EOD source availability.", "draft_only"),
        schedule_row("manual_rerun_semantics", "required", "Manual rerun must preserve requested_date and record actual_report_date / actual_signal_date / fallback_reason.", "draft_only"),
        schedule_row("requested_date", latest_row["requested_date"], "The date requested by schedule/manual rerun.", "materialized_latest_row_contract"),
        schedule_row("actual_report_date", latest_row["actual_report_date"], "Date this report-only row contract was generated.", "materialized_latest_row_contract"),
        schedule_row("actual_signal_date", latest_row["actual_signal_date"], "Latest model signal date used; may lag requested_date.", "materialized_latest_row_contract"),
        schedule_row("fallback_reason", latest_row["fallback_reason"], "Required whenever actual_signal_date differs from requested_date.", "materialized_latest_row_contract"),
        schedule_row("failure_policy", "no_silent_date_substitution", "If EOD fields are missing, write blocked/proxy audit instead of pretending requested date is ready.", "required"),
        schedule_row("production_status", "not_integrated", "No workflow/schedule changed in this task.", "blocked"),
    ]
    return pd.DataFrame(rows)


def build_requested_vs_actual_coverage(latest_row: dict[str, Any]) -> pd.DataFrame:
    rows = [
        {
            "requested_date": latest_row["requested_date"],
            "actual_report_date": latest_row["actual_report_date"],
            "actual_signal_date": latest_row["actual_signal_date"],
            "data_asof_date": latest_row["data_asof_date"],
            "requested_equals_actual_signal": latest_row["requested_date"] == latest_row["actual_signal_date"],
            "fallback_reason": latest_row["fallback_reason"],
            "data_readiness": latest_row["data_readiness"],
            "report_only": True,
            "no_silent_date_substitution": True,
        }
    ]
    return pd.DataFrame(rows)


def policy_row(field: str, value: str, status: str, note: str) -> dict[str, str]:
    return {"field": field, "value": value, "status": status, "note": note}


def schedule_row(item: str, value: str, note: str, status: str) -> dict[str, str]:
    return {"item": item, "value": value, "note": note, "status": status}


def build_blocked_proxy_audit(latest_row: dict[str, Any]) -> pd.DataFrame:
    rows = [
        audit_row("selected_stock_adjusted_close", "blocked", "selected_stock_adjusted_close_ready=false", "Allowed for report-only warning; blocks formal/replay."),
        audit_row("cash_bear_classifier", "blocked", "cash_bear_classifier_ready=false", "No live cash rule can be fabricated."),
        audit_row("rs20_top3_reference_only", "enforced", str(latest_row["rs20_reference_only"]), "RS20 top3 remains reference-only unless regime switch formally authorizes it."),
        audit_row("drive_upload", "not_executed", DRIVE_FOLDER_ID, "No Drive write in this task."),
        audit_row("drive_target_filename", "configured_not_uploaded", DRIVE_FILENAME, "Fixed report-only filename for future overwrite."),
        audit_row("schedule_rules_integration", "not_integrated", "draft only", "No workflow or production schedule changed."),
        audit_row("daily_eod_refresh_execution", "not_executed_in_this_contract", latest_row["fallback_reason"], "This contract specifies refresh semantics but does not acquire new EOD data."),
        audit_row("formal_status", "not_formal", "ready_for_formal=false", "Report-only pipeline must not be packaged as formal-ready."),
        audit_row("trade_decision", "not_live_decision", "trade_decision_changed=false", "No live trade instruction."),
    ]
    return pd.DataFrame(rows)


def audit_row(item: str, status: str, evidence: str, note: str) -> dict[str, str]:
    return {"item": item, "status": status, "evidence": evidence, "note": note}


def build_model_status_note(experiments: dict[str, Any], layer1_revenue_dir: str) -> pd.DataFrame:
    layer1_summary = discover_layer1_revenue_summary(Path(layer1_revenue_dir))
    rows = [
        {
            "component": "R6 unified",
            "status": experiments.get("verdict", experiments.get("status", "PARTIAL_R6_UNIFIED_REVIEW_CANDIDATE_NOT_FORMAL")),
            "policy": "current regime switch review candidate; diagnostic/report-only, not formal.",
            "main_weight_change": "none",
        },
        {
            "component": "Layer1 revenue horizon",
            "status": "PARTIAL only",
            "policy": "monthly + TTM revenue has tiny P1 improvement; keep as soft context / attribution, do not increase main Layer1 weight.",
            "main_weight_change": "none",
            "source": str(layer1_revenue_dir),
            "source_detected_files": layer1_summary,
        },
    ]
    return pd.DataFrame(rows)


def discover_layer1_revenue_summary(path: Path) -> str:
    if not path.exists():
        return "source_dir_missing"
    names = sorted(item.name for item in path.iterdir() if item.is_file())
    return ";".join(names[:8]) if names else "source_dir_empty"


def build_readiness(
    latest_row: dict[str, Any],
    draft_readiness: dict[str, Any],
    experiments: dict[str, Any],
) -> dict[str, Any]:
    return {
        "task_id": TASK_ID,
        "status": "r6_report_only_daily_eod_refresh_drive_overwrite_contract_ready_no_publish",
        "report_date": latest_row["report_date"],
        "requested_date": latest_row["requested_date"],
        "actual_report_date": latest_row["actual_report_date"],
        "signal_date": latest_row["signal_date"],
        "actual_signal_date": latest_row["actual_signal_date"],
        "data_asof_date": latest_row["data_asof_date"],
        "fallback_reason": latest_row["fallback_reason"],
        "ready_for_report_only_pdf_generation": bool(draft_readiness.get("sample_pdf_text_verified", False)),
        "ready_for_daily_eod_refresh_execution": False,
        "ready_for_drive_overwrite": False,
        "ready_for_schedule_integration": False,
        "drive_folder_id": DRIVE_FOLDER_ID,
        "drive_folder_url": DRIVE_FOLDER_URL,
        "drive_filename": DRIVE_FILENAME,
        "drive_overwrite_policy_ready": True,
        "schedule_rules_integration_contract_ready": True,
        "latest_eod_report_row_contract_ready": True,
        "selected_stock_adjusted_close_ready": False,
        "cash_bear_classifier_ready": False,
        "rs20_top3_reference_only_enforced": True,
        "report_only": True,
        "ready_for_daily_report_review": True,
        "daily_report_ready": False,
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
        "source_draft_status": draft_readiness.get("status", ""),
        "experiments_verdict": experiments.get("verdict", experiments.get("status", "")),
        "next_owner_recommendation": "Strategy Center review; after authorization Core can wire Drive overwrite and schedule integration.",
    }


def build_html_sample(latest_row: dict[str, Any], readiness: dict[str, Any]) -> str:
    def row(label: str, value: Any) -> str:
        return f"<tr><th>{escape_html(label)}</th><td>{escape_html(value)}</td></tr>"

    rows = "\n".join(
        [
            row("report_date", latest_row["report_date"]),
            row("requested_date", latest_row["requested_date"]),
            row("actual_signal_date", latest_row["actual_signal_date"]),
            row("data_asof_date", latest_row["data_asof_date"]),
            row("regime_label", latest_row["regime_label"]),
            row("selected_branch", latest_row["selected_branch"]),
            row("selected", f"{latest_row['selected_ticker']} {latest_row['selected_name']}"),
            row("C2 / consensus / R6", f"{latest_row['c2_pass_flag']} / {latest_row['consensus_trigger_flag']} / {latest_row['r6_override_flag']}"),
            row("RS20 top3", "reference-only"),
            row("daily_report_ready", readiness["daily_report_ready"]),
            row("drive_overwrite_ready", readiness["ready_for_drive_overwrite"]),
            row("diagnostic_warning", latest_row["diagnostic_warning"]),
        ]
    )
    return f"""<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8">
  <title>vNext R6 report-only daily sample</title>
  <style>
    body {{ font-family: "Microsoft JhengHei", Arial, sans-serif; margin: 32px; color: #17212a; }}
    .banner {{ background: #17212a; color: white; padding: 18px 22px; border-radius: 8px; }}
    table {{ border-collapse: collapse; margin-top: 20px; width: 100%; }}
    th, td {{ border: 1px solid #d9e0e5; padding: 10px; text-align: left; vertical-align: top; }}
    th {{ width: 240px; background: #f4f6f8; }}
    .warn {{ margin-top: 18px; color: #9a3412; }}
  </style>
</head>
<body>
  <div class="banner">
    <h1>vNext 台股 AI 模型訊號追蹤（日報樣稿）</h1>
    <div>Report-only / diagnostic. Not formal. Not a live trade decision.</div>
  </div>
  <table>{rows}</table>
  <p class="warn">selected_stock_adjusted_close 與 cash/bear classifier 仍 blocked；RS20 top3 僅 reference-only。</p>
</body>
</html>
"""


def escape_html(value: Any) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def build_summary(latest_row: dict[str, Any], readiness: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# R6 report-only daily EOD refresh + Drive overwrite contract",
            "",
            "## 結論",
            "",
            "- 已建立 latest EOD report row contract、Drive overwrite policy contract、schedule_rules integration contract。",
            "- 本任務沒有上傳 Drive、沒有接排程、沒有改正式模型、沒有改交易決策。",
            f"- requested_date={latest_row['requested_date']}；actual_signal_date={latest_row['actual_signal_date']}。",
            f"- selected={latest_row['selected_ticker']} {latest_row['selected_name']}；branch={latest_row['selected_branch']}。",
            f"- C2={latest_row['c2_pass_flag']}；consensus={latest_row['consensus_trigger_flag']}；R6={latest_row['r6_override_flag']}。",
            "",
            "## Readiness",
            "",
            f"- ready_for_report_only_pdf_generation={str(readiness['ready_for_report_only_pdf_generation']).lower()}",
            f"- ready_for_drive_overwrite={str(readiness['ready_for_drive_overwrite']).lower()}",
            f"- ready_for_schedule_integration={str(readiness['ready_for_schedule_integration']).lower()}",
            f"- Drive target：{DRIVE_FOLDER_URL}",
            f"- Drive filename：{DRIVE_FILENAME}",
            "- selected_stock_adjusted_close remains blocked。",
            "- cash_bear_classifier remains blocked；不可杜撰空手規則。",
            "- RS20 top3 reference-only enforced。",
            "",
            "## Model Status Note",
            "",
            "- Layer1 revenue horizon diagnostic = PARTIAL only；monthly + TTM revenue 只保留 soft context / attribution，不提高主 Layer1 權重。",
        ]
    )


def build_manifest(output_dir: Path, artifacts: list[Path]) -> dict[str, Any]:
    return {
        "task_id": TASK_ID,
        "status": "complete_contract_ready_no_publish",
        "output_dir": str(output_dir),
        "artifacts": [
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size if path.exists() else 0,
            }
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
