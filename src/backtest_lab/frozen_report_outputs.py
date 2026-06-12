from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import pandas as pd

from backtest_lab.market_regime import latest_available_date


MarkdownBuilder = Callable[[object], str]
PdfWriter = Callable[[Path, object], None]
SignalClassifier = Callable[[object], str]
PersonalSummaryBuilder = Callable[[object], dict]


def write_report_outputs(
    output_dir: Path,
    signal: object,
    *,
    report_name: str,
    report_version: str,
    drive_folder_url: str,
    markdown_report: MarkdownBuilder,
    write_signal_pdf: PdfWriter,
    report_mode: SignalClassifier,
    personal_exposure_summary: PersonalSummaryBuilder,
) -> None:
    waiting_status = output_dir / "waiting_status.json"
    if waiting_status.exists():
        waiting_status.unlink()
    payload = {
        "status": "ready",
        "report_name": report_name,
        "report_mode": report_mode(signal),
        "drive_folder_url": drive_folder_url,
        "signal": signal.to_dict(),
        "disclaimer": "AI 輔助市場觀察、回測與紀律提醒，不是投資建議。",
    }
    (output_dir / "frozen_strategy_signal.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    personal_summary = personal_exposure_summary(signal) if signal.personal_portfolio else None
    pd.DataFrame([daily_status_row(signal, report_mode(signal), personal_summary)]).to_csv(
        output_dir / "frozen_strategy_daily_status.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(signal.ranking).to_csv(output_dir / "frozen_strategy_ranking.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(signal.projected_trades).to_csv(
        output_dir / "frozen_strategy_projected_trades.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(shadow_mode_rows(signal)).to_csv(
        output_dir / "shadow_mode_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )
    report = markdown_report(signal)
    dated_md = output_dir / report_filename(report_name, report_version, "md", signal.signal_date)
    latest_md = output_dir / report_filename(report_name, report_version, "md", latest=True)
    dated_md.write_text(report, encoding="utf-8")
    latest_md.write_text(report, encoding="utf-8")
    write_signal_pdf(output_dir / report_filename(report_name, report_version, "pdf", signal.signal_date), signal)
    write_signal_pdf(output_dir / report_filename(report_name, report_version, "pdf", latest=True), signal)


def daily_status_row(signal: object, report_mode_value: str, personal_summary: dict | None) -> dict:
    return {
        "strategy_id": signal.strategy_id,
        "signal_date": signal.signal_date,
        "market_regime": signal.market_regime,
        "market_regime_label": signal.market_regime_label,
        "action": signal.action,
        "current_ticker": signal.current_ticker,
        "current_label": signal.current_label,
        "current_exposure": signal.current_exposure,
        "target_ticker": signal.target_ticker,
        "target_label": signal.target_label,
        "target_exposure": signal.target_exposure,
        "target_is_actionable": signal.target_is_actionable,
        "model_target_status": signal.model_target_status,
        "cash_account_reference": signal.cash_account_reference,
        "attack_gate_active": signal.attack_gate_active,
        "attack_gate_ever_activated": signal.attack_gate_ever_activated,
        "risk_off_active": signal.risk_off_active,
        "shadow_mode_count": len(signal.shadow_modes or []),
        "report_mode": report_mode_value,
        "personal_portfolio_attached": signal.personal_portfolio is not None,
        "personal_total_value_twd": personal_summary["total_value_twd"] if personal_summary else "",
        "personal_cash_exposure": personal_summary["cash_exposure"] if personal_summary else "",
        "personal_market_exposure": personal_summary["market_exposure"] if personal_summary else "",
        "personal_target_actual_exposure": personal_summary["target_actual_exposure"] if personal_summary else "",
        "personal_target_gap_exposure": personal_summary["target_gap_exposure"] if personal_summary else "",
    }


def shadow_mode_rows(signal: object) -> list[dict]:
    rows = []
    for rank, shadow in enumerate(signal.shadow_modes or [], start=1):
        rows.append(
            {
                "rank": rank,
                "signal_date": signal.signal_date,
                "shadow_id": shadow.shadow_id,
                "shadow_label": shadow.shadow_label,
                "strategy_name": shadow.strategy_name,
                "current_ticker": shadow.current_ticker,
                "current_label": shadow.current_label,
                "current_exposure": shadow.current_exposure,
                "target_ticker": shadow.target_ticker,
                "target_label": shadow.target_label,
                "target_exposure": shadow.target_exposure,
                "action": shadow.action,
                "model_target_status": shadow.model_target_status,
                "model_total_value_twd": shadow.model_total_value_twd,
                "primary_model_total_value_twd": signal.model_total_value_twd,
                "value_diff_twd": shadow.value_diff_twd,
                "value_diff_pct": shadow.value_diff_pct,
                "target_differs_from_primary": shadow.target_ticker != signal.target_ticker
                or abs(shadow.target_exposure - signal.target_exposure) >= 0.02,
            }
        )
    return rows


def report_filename(
    report_name: str,
    report_version: str,
    extension: str,
    signal_date: str | None = None,
    *,
    latest: bool = False,
) -> str:
    if latest:
        suffix = "最新版"
    elif signal_date:
        suffix = signal_date
    else:
        raise ValueError("signal_date is required when latest is false.")
    return f"{report_name}_{suffix}_{report_version}.{extension}"


def write_skip_status(
    output_dir: Path,
    signal_date: str,
    prices_by_ticker: dict[str, pd.DataFrame],
    incomplete: list[str],
) -> None:
    payload = {
        "status": "waiting_for_complete_market_data",
        "signal_date": signal_date,
        "incomplete_tickers": incomplete,
        "latest_available_dates": {
            ticker: (
                latest_available_date(frame).strftime("%Y-%m-%d")
                if latest_available_date(frame) is not None
                else ""
            )
            for ticker, frame in prices_by_ticker.items()
        },
    }
    (output_dir / "waiting_status.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_download_waiting_status(output_dir: Path, signal_date: str, error: Exception) -> None:
    payload = {
        "status": "waiting_for_market_data_download",
        "signal_date": signal_date,
        "error": str(error),
    }
    (output_dir / "waiting_status.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
