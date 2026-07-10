from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


TASK_ID = "TASK-BACKTEST-CORE-VNEXT-R6-DAILY-REPORT-ONLY-CONTRACT-PDF-PIPELINE-DRAFT-001"
DEFAULT_CORE_INPUT = Path("outputs/vnext_r6_guard_first_market_bias_override_unified_contract_20260709")
DEFAULT_EXPERIMENTS_INPUT = Path(
    "C:/Users/zergv/Documents/Codex/2026-07-06/"
    "backtest-lab-experiments-diagnostic-validation-attribution/outputs/"
    "vnext_r6_guard_first_market_bias_override_unified_diagnostic_20260710"
)
DEFAULT_OUTPUT = Path("outputs/vnext_r6_daily_report_only_contract_20260710")
DEFAULT_REPORT_DATE = "2026-07-10"
REMOTE_FOLDER_ID = "1O6Se-HfI7ZDTQ-LWeAO6f8vtvoLcCzIj"
REMOTE_FILENAME = "AI台股新模型每日收盤報告.pdf"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build R6 vNext report-only daily PDF contract draft.")
    parser.add_argument("--core-input", default=str(DEFAULT_CORE_INPUT))
    parser.add_argument("--experiments-input", default=str(DEFAULT_EXPERIMENTS_INPUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--report-date", default=DEFAULT_REPORT_DATE)
    args = parser.parse_args()

    core_input = Path(args.core_input)
    experiments_input = Path(args.experiments_input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    contract_path = core_input / "r6_guard_first_market_bias_override_unified_contract.csv"
    readiness_path = core_input / "readiness_for_r6_guard_first_market_bias_override_unified_contract.json"
    diagnostic_path = experiments_input / "r6_unified_diagnostic_summary.json"

    contract = pd.read_csv(contract_path)
    latest = latest_signal_row(contract)
    core_readiness = read_json(readiness_path)
    diagnostic = read_json(diagnostic_path)
    latest_report = build_latest_report_row(latest, args.report_date)

    artifacts: list[Path] = []
    artifacts.append(write_schema(output_dir / "r6_daily_report_only_contract_schema.json"))
    latest_csv = output_dir / "r6_daily_report_latest_row.csv"
    pd.DataFrame([latest_report]).to_csv(latest_csv, index=False, encoding="utf-8-sig")
    artifacts.append(latest_csv)

    publish_audit = build_publish_readiness_audit(latest_report)
    publish_csv = output_dir / "r6_daily_report_publish_readiness_audit.csv"
    publish_audit.to_csv(publish_csv, index=False, encoding="utf-8-sig")
    artifacts.append(publish_csv)

    blocked_audit = build_blocked_proxy_audit(latest_report)
    blocked_csv = output_dir / "r6_daily_report_blocked_proxy_audit.csv"
    blocked_audit.to_csv(blocked_csv, index=False, encoding="utf-8-sig")
    artifacts.append(blocked_csv)

    pdf_path = output_dir / "r6_daily_report_sample.pdf"
    write_sample_pdf(pdf_path, latest_report, diagnostic, core_readiness)
    artifacts.append(pdf_path)

    readiness = build_readiness(latest_report, core_readiness, diagnostic, pdf_path)
    readiness_path_out = output_dir / "readiness_for_r6_daily_report_only_contract.json"
    readiness_path_out.write_text(json.dumps(readiness, ensure_ascii=False, indent=2), encoding="utf-8")
    artifacts.append(readiness_path_out)

    summary_path = output_dir / "final_summary_zh.md"
    summary_path.write_text(build_final_summary(latest_report, readiness), encoding="utf-8")
    artifacts.append(summary_path)

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(build_manifest(output_dir, core_input, experiments_input, artifacts), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"R6_DAILY_REPORT_ONLY_OUTPUT={output_dir.resolve()}")
    print(f"R6_DAILY_REPORT_SAMPLE_PDF={pdf_path.resolve()}")
    print(f"R6_DAILY_REPORT_READY_FOR_DAILY_REPORT={readiness['ready_for_daily_report']}")


def latest_signal_row(contract: pd.DataFrame) -> pd.Series:
    if "signal_date" not in contract.columns:
        raise ValueError("unified contract is missing signal_date")
    frame = contract.copy()
    frame["_signal_date"] = pd.to_datetime(frame["signal_date"], errors="coerce")
    frame = frame.dropna(subset=["_signal_date"]).sort_values("_signal_date")
    if frame.empty:
        raise ValueError("unified contract has no valid signal_date rows")
    return frame.iloc[-1]


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"status": "missing", "path": str(path)}
    return json.loads(path.read_text(encoding="utf-8"))


def value(row: pd.Series, key: str, default: Any = "") -> Any:
    if key not in row.index:
        return default
    raw = row[key]
    if pd.isna(raw):
        return default
    return raw


def bool_value(row: pd.Series, key: str) -> bool:
    raw = value(row, key, False)
    if isinstance(raw, bool):
        return raw
    return str(raw).strip().lower() in {"true", "1", "yes", "y"}


def build_latest_report_row(row: pd.Series, report_date: str) -> dict[str, Any]:
    selected_ticker = str(value(row, "selected_ticker", "")).strip()
    selected_name = str(value(row, "selected_ticker_name", "")).strip()
    if selected_ticker == "00631L" and not selected_name:
        selected_name = "00631L / 0050正二"

    selected_asset_type = str(value(row, "selected_asset_type", "")).strip().lower()
    if selected_ticker == "00631L":
        primary_type = "00631L"
    elif selected_asset_type in {"stock", "ordinary_stock"}:
        primary_type = "stock"
    else:
        primary_type = selected_asset_type or "blocked"

    blocked_reason = str(value(row, "blocked_reason", "")).strip()
    if not blocked_reason:
        blocked_reason = "production_daily_report_not_authorized;selected_stock_adjusted_close_blocked;cash_bear_classifier_blocked"

    return {
        "task": TASK_ID,
        "report_date": report_date,
        "signal_date": str(value(row, "signal_date", "")),
        "data_asof_date": str(value(row, "signal_date", "")),
        "selected_primary_asset_type": primary_type,
        "selected_ticker": selected_ticker,
        "selected_name": selected_name,
        "selected_branch": str(value(row, "selected_branch", "")),
        "regime_label": str(value(row, "regime_label", "")),
        "branch_reason": str(value(row, "branch_reason", "")),
        "triggered_features": str(value(row, "triggered_features", "")),
        "c2_pass_flag": bool_value(row, "c2_pass_flag"),
        "consensus_trigger_flag": bool_value(row, "consensus_trigger_flag"),
        "r6_override_flag": bool_value(row, "r6_override_flag"),
        "p1_risk_veto_flag": bool_value(row, "p1_risk_veto_flag"),
        "fallback_asset": str(value(row, "fallback_asset", "")),
        "rs20_top3_reference_tickers": str(value(row, "rs20_top3_reference_tickers", "")),
        "rs20_reference_only": bool_value(row, "rs20_reference_only"),
        "data_readiness": str(value(row, "data_readiness", "")),
        "blocked_reason": blocked_reason,
        "cost_model_status": "EP05 transition cost hooks available in unified diagnostic contract; report draft only",
        "diagnostic_warning": "report_only_draft_not_live_trade_decision_not_formal",
        "report_only": True,
        "not_live_trade_decision": True,
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


def write_schema(path: Path) -> Path:
    schema = {
        "task": TASK_ID,
        "status": "report_only_contract_schema_draft",
        "description": "R6 vNext daily report-only row schema. This schema is for local PDF/readiness draft only.",
        "fields": [
            field("report_date", "string", "daily_report_runner", True, "PDF report date."),
            field("signal_date", "string", "r6_unified_contract", True, "Model signal date used by the row."),
            field("data_asof_date", "string", "r6_unified_contract", True, "Data as-of date; currently same as signal_date."),
            field("selected_primary_asset_type", "enum", "derived", True, "stock / 00631L / cash_blocked."),
            field("selected_ticker", "string", "r6_unified_contract", True, "Selected ticker or fallback reference ticker."),
            field("selected_name", "string", "r6_unified_contract", False, "Ticker display name if available."),
            field("selected_branch", "string", "r6_unified_contract", True, "route_support / R6 override / fallback branch label."),
            field("regime_label", "string", "r6_unified_contract", True, "Human-readable regime label."),
            field("branch_reason", "string", "r6_unified_contract", True, "Why this branch was chosen."),
            field("triggered_features", "string", "r6_unified_contract", True, "PIT feature trace; no future returns."),
            field("c2_pass_flag", "boolean", "r6_unified_contract", True, "C2 market health flag."),
            field("consensus_trigger_flag", "boolean", "r6_unified_contract", True, "Consensus trigger flag."),
            field("r6_override_flag", "boolean", "r6_unified_contract", True, "R6 breakout+breadth override flag."),
            field("p1_risk_veto_flag", "boolean", "r6_unified_contract", True, "P1-like risk veto flag."),
            field("fallback_asset", "string", "r6_unified_contract", False, "Fallback asset when stock branch is not active."),
            field("rs20_top3_reference_tickers", "string", "r6_unified_contract", False, "Reference-only RS20 top3 tickers."),
            field("data_readiness", "string", "r6_unified_contract", True, "ready / proxy / blocked label."),
            field("blocked_reason", "string", "derived", True, "Production blockers and report-only caveats."),
            field("cost_model_status", "string", "derived", True, "Transaction-cost status for report wording."),
            field("diagnostic_warning", "string", "derived", True, "Report-only warning."),
        ],
        "policy": {
            "report_only": True,
            "no_drive_write": True,
            "no_schedule_integration": True,
            "no_live_trade_instruction": True,
            "selected_stock_adjusted_close_remains_blocked": True,
            "cash_bear_classifier_remains_blocked": True,
            "forward_returns_live_rule_usage": False,
        },
    }
    path.write_text(json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def field(name: str, type_name: str, source: str, required: bool, description: str) -> dict[str, Any]:
    return {
        "name": name,
        "type": type_name,
        "source": source,
        "required": required,
        "description": description,
    }


def build_publish_readiness_audit(report: dict[str, Any]) -> pd.DataFrame:
    rows = [
        audit_row("drive_folder_id_path_config", "configured_not_published", REMOTE_FOLDER_ID, "Folder id recorded only; no Drive write in this task."),
        audit_row("drive_remote_filename", "configured_not_published", REMOTE_FILENAME, "Filename policy draft only."),
        audit_row("drive_overwrite_semantics", "design_required", "update_by_name_or_create_once", "Need implementation/verification before production."),
        audit_row("schedule_rules_integration_point", "blocked_not_integrated", "AI_stock_schedule_rules reuse point not wired", "Strategy Center authorization required."),
        audit_row("exact_latest_daily_data_update_source", "blocked_for_production", str(report["data_asof_date"]), "This sample uses latest available unified contract row, not a live EOD refresh pipeline."),
        audit_row("selected_stock_adjusted_close", "blocked", "adjusted_close_ready=false", "Do not promote to formal-ready."),
        audit_row("cash_bear_classifier", "blocked", "no accepted classifier", "cash/bear path must remain blocked/proxy."),
        audit_row("pdf_overwrite_execution", "not_executed", "local sample only", "No Google Drive upload/update."),
        audit_row("report_wording_review", "draft_ready_needs_review", "report-only warning included", "Strategy Center should approve public wording."),
    ]
    return pd.DataFrame(rows)


def build_blocked_proxy_audit(report: dict[str, Any]) -> pd.DataFrame:
    rows = [
        audit_row("selected_stock_adjusted_close", "blocked", "selected_stock_adjusted_close_ready=false", "Official unadjusted diagnostic path exists; adjusted close is not fabricated."),
        audit_row("cash_bear_classifier", "blocked", "cash/bear classifier not accepted", "cash_blocked must not be converted into a live cash rule."),
        audit_row("drive_publish", "blocked_not_requested", REMOTE_FOLDER_ID, "No Drive write in report-only draft task."),
        audit_row("schedule_integration", "blocked_not_authorized", "not integrated", "No production scheduler change."),
        audit_row("formal_model", "blocked_by_policy", "formal_model_changed=false", "Diagnostic/report-only candidate, not formal."),
        audit_row("trade_decision", "blocked_by_policy", "trade_decision_changed=false", "No live trade instruction."),
        audit_row("latest_eod_refresh", "sample_only", str(report["data_asof_date"]), "PDF sample uses latest row in unified contract."),
    ]
    return pd.DataFrame(rows)


def audit_row(item: str, status: str, evidence: str, note: str) -> dict[str, str]:
    return {"item": item, "status": status, "evidence": evidence, "note": note}


def write_sample_pdf(path: Path, report: dict[str, Any], diagnostic: dict[str, Any], core_readiness: dict[str, Any]) -> None:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
    doc = SimpleDocTemplate(
        str(path),
        pagesize=A4,
        rightMargin=1.4 * cm,
        leftMargin=1.4 * cm,
        topMargin=1.2 * cm,
        bottomMargin=1.2 * cm,
        title="vNext 每日模型報告（樣稿）",
    )
    base = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "zh_title",
        parent=base["Title"],
        fontName="STSong-Light",
        fontSize=20,
        leading=26,
        textColor=colors.HexColor("#17212a"),
    )
    subtitle_style = ParagraphStyle(
        "zh_subtitle",
        parent=base["Normal"],
        fontName="STSong-Light",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#66737d"),
    )
    heading_style = ParagraphStyle(
        "zh_heading",
        parent=base["Heading2"],
        fontName="STSong-Light",
        fontSize=14,
        leading=19,
        textColor=colors.HexColor("#17212a"),
        spaceBefore=12,
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "zh_body",
        parent=base["BodyText"],
        fontName="STSong-Light",
        fontSize=10.5,
        leading=16,
        textColor=colors.HexColor("#27323a"),
    )
    warning_style = ParagraphStyle(
        "zh_warning",
        parent=body_style,
        textColor=colors.HexColor("#9a3412"),
    )

    story = [
        Paragraph("vNext 每日模型報告（樣稿）", title_style),
        Paragraph(
            f"報告日 {report['report_date']}｜訊號日 {report['signal_date']}｜Report-only draft，不是正式交易決策",
            subtitle_style,
        ),
        Spacer(1, 0.35 * cm),
    ]
    summary_rows = [
        ["欄位", "內容"],
        ["regime_label", report["regime_label"]],
        ["selected_branch", report["selected_branch"]],
        ["主資產", f"{report['selected_ticker']} {report['selected_name']}"],
        ["C2 / consensus / R6", f"{yn(report['c2_pass_flag'])} / {yn(report['consensus_trigger_flag'])} / {yn(report['r6_override_flag'])}"],
        ["fallback_asset", report["fallback_asset"] or "無"],
    ]
    story.append(report_table(summary_rows))
    story.append(Paragraph("模型狀態", heading_style))
    for line in [
        f"selected_primary_asset_type：{report['selected_primary_asset_type']}",
        f"branch_reason：{report['branch_reason']}",
        f"triggered_features：{clip(report['triggered_features'], 160)}",
    ]:
        story.append(Paragraph(f"• {line}", body_style))
    story.append(Paragraph("診斷證據", heading_style))
    for line in [
        f"R6 contract rows：{core_readiness.get('contract_rows', '')}；path_ready_share：{core_readiness.get('path_ready_share', '')}",
        f"R6 override count：{core_readiness.get('r6_override_count', '')}",
        f"Experiments verdict：{diagnostic.get('verdict', diagnostic.get('status', 'missing'))}",
        "RS20 top3：reference-only，不作主推薦。",
    ]:
        story.append(Paragraph(f"• {line}", body_style))
    story.append(Paragraph("發布與使用邊界", heading_style))
    for line in [
        "本 PDF 是本機樣稿，不是正式每日報告，不會上傳 Google Drive。",
        "selected_stock_adjusted_close 仍 blocked；cash/bear classifier 仍 blocked。",
        "此報告不改正式模型、不改交易決策、不作 live trade instruction。",
    ]:
        story.append(Paragraph(f"• {line}", warning_style))
    story.append(Spacer(1, 0.35 * cm))
    story.append(
        Paragraph(
            "Report-only / diagnostic draft. No formal model change. No trade decision. No Drive publish.",
            subtitle_style,
        )
    )
    doc.build(story)


def report_table(rows: list[list[str]]):
    from reportlab.lib import colors
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import Paragraph, Table, TableStyle

    cell = ParagraphStyle("cell", fontName="STSong-Light", fontSize=10, leading=14)
    data = [[Paragraph(str(item), cell) for item in row] for row in rows]
    table = Table(data, colWidths=[4.2 * cm, 11.2 * cm])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17212a")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("BACKGROUND", (0, 1), (-1, -1), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#d9e0e5")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return table


def wrap_line(text: str, width: int) -> list[str]:
    import textwrap

    return textwrap.wrap(str(text), width=width) or [""]


def yn(flag: Any) -> str:
    return "通過" if bool(flag) else "未觸發"


def clip(text: Any, limit: int) -> str:
    value = str(text)
    return value if len(value) <= limit else value[: limit - 1] + "…"


def build_readiness(
    report: dict[str, Any],
    core_readiness: dict[str, Any],
    diagnostic: dict[str, Any],
    pdf_path: Path,
) -> dict[str, Any]:
    pdf_check = verify_pdf_text(pdf_path)
    return {
        "task_id": TASK_ID,
        "status": "r6_daily_report_only_contract_pdf_draft_created_not_published",
        "report_date": report["report_date"],
        "latest_signal_date": report["signal_date"],
        "data_asof_date": report["data_asof_date"],
        "sample_pdf_created": pdf_path.exists(),
        "sample_pdf_path": str(pdf_path),
        "sample_pdf_text_verified": pdf_check["text_verified"],
        "sample_pdf_page_count": pdf_check["page_count"],
        "sample_pdf_render_verified": False,
        "sample_pdf_render_blocked_reason": "local pdftoppm wrapper exists but native poppler binary path is missing in this runtime",
        "report_only": True,
        "report_only_draft_created": True,
        "ready_for_daily_report": False,
        "ready_for_drive_publish": False,
        "ready_for_schedule_integration": False,
        "drive_folder_id": REMOTE_FOLDER_ID,
        "drive_remote_filename": REMOTE_FILENAME,
        "core_unified_contract_status": core_readiness.get("status", ""),
        "experiments_verdict": diagnostic.get("verdict", diagnostic.get("status", "")),
        "selected_stock_adjusted_close_ready": False,
        "cash_bear_classifier_ready": False,
        "formal_model_changed": False,
        "trade_decision_changed": False,
        "active_in_trade_decision": False,
        "report_changed": False,
        "portfolio_replay_executed": False,
        "ready_for_strategy_replay": False,
        "ready_for_formal": False,
        "not_live_rule": True,
        "forward_returns_live_rule_usage": False,
        "future_data_violation_count": 0,
    }


def verify_pdf_text(pdf_path: Path) -> dict[str, Any]:
    try:
        from pypdf import PdfReader

        reader = PdfReader(str(pdf_path))
        text = "\n".join((page.extract_text() or "") for page in reader.pages)
        return {
            "text_verified": "vNext 每日模型報告" in text and "Report-only" in text,
            "page_count": len(reader.pages),
        }
    except Exception as exc:  # pragma: no cover - defensive readiness audit
        return {
            "text_verified": False,
            "page_count": 0,
            "error": str(exc),
        }


def build_final_summary(report: dict[str, Any], readiness: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# R6 vNext 每日報告 report-only contract / PDF pipeline draft",
            "",
            "## 結論",
            "",
            "- 已建立 R6 daily report-only contract schema 與本機 PDF 樣稿。",
            "- 本任務沒有上傳 Google Drive、沒有接 schedule、沒有改正式模型、沒有改交易決策。",
            f"- 最新可用 unified contract 訊號日：{report['signal_date']}。",
            f"- 樣稿列出的主分支：{report['selected_branch']}；主資產：{report['selected_ticker']} {report['selected_name']}。",
            f"- C2：{yn(report['c2_pass_flag'])}；consensus trigger：{yn(report['consensus_trigger_flag'])}；R6 override：{yn(report['r6_override_flag'])}。",
            "",
            "## 發布狀態",
            "",
            f"- report_only_draft_created={str(readiness['report_only_draft_created']).lower()}",
            f"- ready_for_daily_report={str(readiness['ready_for_daily_report']).lower()}",
            f"- ready_for_drive_publish={str(readiness['ready_for_drive_publish']).lower()}",
            "- selected_stock_adjusted_close 仍 blocked。",
            "- cash/bear classifier 仍 blocked。",
            "",
            "## 下一步",
            "",
            "- 若 Strategy Center 之後授權，下一步才是 Drive overwrite 實作與 schedule_rules integration。",
            "- 在授權前，此 artifact 只能作 report-only draft review。",
        ]
    )


def build_manifest(
    output_dir: Path,
    core_input: Path,
    experiments_input: Path,
    artifacts: list[Path],
) -> dict[str, Any]:
    return {
        "task_id": TASK_ID,
        "status": "complete_report_only_draft",
        "output_dir": str(output_dir),
        "inputs": {
            "core_unified_contract_dir": str(core_input),
            "experiments_diagnostic_dir": str(experiments_input),
        },
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
            "report_only_draft_created": True,
            "portfolio_replay_executed": False,
            "ready_for_strategy_replay": False,
            "ready_for_formal": False,
            "not_live_rule": True,
            "forward_returns_live_rule_usage": False,
        },
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
