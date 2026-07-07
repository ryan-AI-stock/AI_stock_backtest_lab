"""Build Layer 1 t05st01 official announcement as-of contract.

This stages market-available-at policy and join design for t164 financial
statement features. It does not perform full ingest, experiments, or replay.
"""

from __future__ import annotations

import argparse
import json
from datetime import time
from pathlib import Path
from typing import Any

import pandas as pd


TASK_ID = "TASK-BACKTEST-CORE-VNEXT-LAYER1-T05ST01-OFFICIAL-ANNOUNCEMENT-ASOF-CONTRACT-001"
DEFAULT_RADAR_DIR = Path(
    r"C:\Users\zergv\Documents\Codex\2026-05-23\ai-stock-rotation-radar-https-docs\outputs"
    r"\radar_vnext_layer1_exact_filing_timestamp_route_search_20260707"
)
DEFAULT_OUTPUT_DIR = Path("outputs/vnext_layer1_t05st01_official_announcement_asof_contract_20260707")


def build_t05st01_asof_contract(
    *,
    radar_dir: str | Path = DEFAULT_RADAR_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    radar = Path(radar_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    readiness_in = _read_json(radar / "readiness_for_core_exact_filing_asof_join.json")
    route_inventory = _read_csv(radar / "exact_filing_timestamp_source_route_inventory.csv")
    sample = _read_csv(radar / "mops_financial_report_announcement_route_probe.csv")
    policy = _read_csv(radar / "official_announcement_date_vs_timestamp_policy_ledger.csv")
    blocked_in = _read_csv(radar / "blocked_or_partial_asof_fields.csv")

    announcement_contract = _announcement_asof_contract(sample)
    join_policy = _join_policy(route_inventory, policy)
    eligibility_policy = _signal_eligibility_policy(announcement_contract)
    blocked = _blocked_prohibited_fields(blocked_in)
    readiness = _readiness_json(readiness_in, announcement_contract, join_policy, eligibility_policy, blocked)

    _write_csv(announcement_contract, output / "layer1_t05st01_financial_report_announcement_asof_contract.csv")
    _write_csv(join_policy, output / "layer1_t164_t05st01_asof_join_policy.csv")
    _write_csv(eligibility_policy, output / "layer1_market_available_at_signal_eligibility_policy.csv")
    _write_csv(blocked, output / "layer1_asof_blocked_prohibited_fields.csv")
    (output / "readiness_for_layer1_t164_official_asof_join.json").write_text(
        json.dumps(readiness, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    manifest = {
        "task_id": TASK_ID,
        "status": readiness["status"],
        "output_dir": str(output.resolve()),
        "radar_input_dir": str(radar.resolve()),
        "radar_exact_filing_timestamp_route_search_commit": "a8a15b0",
        "output_files": [
            "layer1_t05st01_financial_report_announcement_asof_contract.csv",
            "layer1_t164_t05st01_asof_join_policy.csv",
            "layer1_market_available_at_signal_eligibility_policy.csv",
            "layer1_asof_blocked_prohibited_fields.csv",
            "readiness_for_layer1_t164_official_asof_join.json",
            "manifest.json",
            "final_summary_zh.md",
        ],
        "formal_model_changed": False,
        "trade_decision_changed": False,
        "active_in_trade_decision": False,
        "report_changed": False,
        "portfolio_replay_executed": False,
        "ready_for_strategy_replay": False,
        "not_live_rule": True,
        "forward_returns_live_rule_usage": False,
        "diagnostic_only": True,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "final_summary_zh.md").write_text(_summary(readiness, blocked), encoding="utf-8")
    return manifest


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _announcement_asof_contract(sample: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in sample.itertuples(index=False):
        timestamp = _roc_datetime_to_iso(row.announcement_date, row.announcement_time)
        period = _infer_report_period(row.period_text, row.matched_subject)
        after_close = _is_after_regular_close(row.announcement_time)
        rows.append(
            {
                "ticker": row.ticker,
                "market": row.market,
                "route_id": row.route_id,
                "detail_api": row.detail_api,
                "announcement_subject": row.matched_subject,
                "report_period": period,
                "source_period_text": row.period_text,
                "announcement_date": row.announcement_date,
                "announcement_time": row.announcement_time,
                "official_announcement_timestamp": timestamp,
                "market_available_at": timestamp,
                "market_available_at_source": "t05st01_public_material_information_announcement_timestamp",
                "exact_internal_filing_upload_timestamp_found": False,
                "after_regular_close": after_close,
                "signal_eligibility_basis": "next_trading_day_if_after_regular_close_else_same_trading_day_after_timestamp",
                "source_quality": row.source_quality,
                "accepted_for_formal": False,
                "diagnostic_only": True,
                "not_live_rule": True,
            }
        )
    return pd.DataFrame(rows)


def _roc_datetime_to_iso(roc_date: str, time_text: str) -> str:
    parts = str(roc_date).split("/")
    year = int(parts[0]) + 1911
    return f"{year:04d}-{int(parts[1]):02d}-{int(parts[2]):02d}T{time_text}"


def _infer_report_period(period_text: str, subject: str) -> str:
    text = f"{period_text} {subject}"
    if "115" in text and ("第1季" in text or "第一季" in text or "115/01/01" in text):
        return "2026Q1"
    if "114" in text and "第4季" in text:
        return "2025Q4"
    return "unknown_requires_parser"


def _is_after_regular_close(time_text: str) -> bool:
    hour, minute, second = [int(part) for part in str(time_text).split(":")]
    return time(hour, minute, second) >= time(13, 30, 0)


def _join_policy(route_inventory: pd.DataFrame, policy: pd.DataFrame) -> pd.DataFrame:
    rows = [
        (
            "join_key_company",
            "ticker/companyId",
            "required",
            "Match t164sb05/t164sb03 companyId to t05st01 companyId.",
        ),
        (
            "join_key_report_period",
            "ROC year + season inferred from t164 payload and t05st01 subject/period text",
            "required_with_human_review",
            "Use subject regex for 董事會通過 + fiscal year/quarter; keep unmatched rows blocked.",
        ),
        (
            "join_key_market",
            "marketKind sii/otc from t05st01_detail + t164 market sample",
            "required",
            "Use market for detail payload and TPEx/TWSE audit; bounded samples passed both markets.",
        ),
        (
            "join_timestamp",
            "official_announcement_timestamp",
            "required",
            "Set market_available_at to public material-information 發言日期+發言時間.",
        ),
        (
            "internal_filing_upload_timestamp",
            "not available",
            "blocked_not_required_for_market_available_at_policy",
            "Keep exact_internal_filing_upload_timestamp_found=false.",
        ),
        (
            "prohibited_available_date_sources",
            "API response datetime / quarter-end date / conservative filing-deadline proxy",
            "prohibited",
            "Cannot replace official announcement timestamp when route match exists.",
        ),
    ]
    return pd.DataFrame(
        [
            {
                "policy_item": item,
                "field_or_rule": field,
                "requirement": requirement,
                "policy_detail": detail,
                "ready_for_contract_design": requirement != "prohibited",
                "ready_for_full_ingest": False,
                "diagnostic_only": True,
            }
            for item, field, requirement, detail in rows
        ]
    )


def _signal_eligibility_policy(contract: pd.DataFrame) -> pd.DataFrame:
    rows = [
        (
            "market_available_at_before_or_during_session",
            "same trading day only after timestamp and only if signal construction time is after market_available_at",
            "avoid same-day pre-announcement leakage",
        ),
        (
            "market_available_at_after_regular_close",
            "next trading day eligibility",
            "sample announcements after 13:30 cannot be used by same-day signal",
        ),
        (
            "weekly_signal_date",
            "eligible only if market_available_at <= signal_timestamp; if signal timestamp unknown, use next trading day after announcement",
            "conservative PIT policy",
        ),
        (
            "announcement_time_missing",
            "blocked until timestamp resolved",
            "date-only route is not enough",
        ),
    ]
    sample_after_close = int(contract["after_regular_close"].sum()) if not contract.empty else 0
    return pd.DataFrame(
        [
            {
                "eligibility_case": case,
                "signal_eligibility_rule": rule,
                "reason": reason,
                "sample_after_close_rows": sample_after_close,
                "same_day_pre_announcement_use_allowed": False,
                "diagnostic_only": True,
                "not_live_rule": True,
            }
            for case, rule, reason in rows
        ]
    )


def _blocked_prohibited_fields(blocked_in: pd.DataFrame) -> pd.DataFrame:
    rows = [
        (
            "exact_internal_filing_upload_timestamp",
            "blocked",
            "No route found; Strategy accepts public announcement timestamp as market_available_at but internal upload remains false.",
        ),
        ("api_response_datetime", "prohibited", "Query-time response datetime is not historical availability."),
        ("quarter_end_date", "prohibited", "Quarter-end precedes disclosure; forbidden as available_date."),
        ("conservative_filing_deadline_proxy", "superseded_for_matched_rows", "Do not replace official timestamp when t05st01 match exists."),
        ("date_only_announcement", "insufficient", "Use 發言日期+發言時間, not date-only."),
    ]
    if not blocked_in.empty:
        for row in blocked_in.itertuples(index=False):
            rows.append((row.field, row.status, row.impact))
    return pd.DataFrame(
        [
            {
                "field": field,
                "status": status,
                "reason": reason,
                "ready_for_asof_join": status in {"superseded_for_matched_rows"},
                "ready_for_formal": False,
                "diagnostic_only": True,
            }
            for field, status, reason in rows
        ]
    )


def _readiness_json(
    readiness_in: dict[str, Any],
    contract: pd.DataFrame,
    join_policy: pd.DataFrame,
    eligibility_policy: pd.DataFrame,
    blocked: pd.DataFrame,
) -> dict[str, Any]:
    contract_ready = bool(readiness_in.get("ready_for_core_official_announcement_timestamp_asof_contract", False)) and not contract.empty
    return {
        "date": "2026-07-07",
        "task_id": TASK_ID,
        "owner": "BACKTEST_LAB Core/Data",
        "status": "official_announcement_asof_contract_ready_sample_design_not_full_ingest" if contract_ready else "blocked_official_announcement_asof_contract",
        "ready_for_core_official_announcement_timestamp_asof_contract": contract_ready,
        "ready_for_core_exact_filing_asof_join": False,
        "ready_for_t164_interim_official_asof_join_design": contract_ready,
        "ready_for_full_ingest": False,
        "ready_for_experiments": False,
        "ready_for_formal": False,
        "ready_for_strategy_replay": False,
        "official_announcement_timestamp_route_found": bool(readiness_in.get("official_announcement_timestamp_route_found", False)),
        "market_available_at_source": "t05st01_public_material_information_announcement_timestamp",
        "exact_internal_filing_upload_timestamp_found": False,
        "after_close_next_trading_day_policy_required": True,
        "future_data_violation_count": int(readiness_in.get("future_data_violation_count", 0)),
        "sample_contract_rows": int(len(contract)),
        "join_policy_rows": int(len(join_policy)),
        "eligibility_policy_rows": int(len(eligibility_policy)),
        "blocked_or_prohibited_fields": blocked["field"].tolist(),
        "required_next_step": "vNext Research judge whether t164 cashflow/inventory/receivable can enter interim Layer1 diagnostic package; Core full ingest still needs explicit authorization",
        "formal_model_changed": False,
        "trade_decision_changed": False,
        "active_in_trade_decision": False,
        "report_changed": False,
        "portfolio_replay_executed": False,
        "not_live_rule": True,
        "forward_returns_live_rule_usage": False,
    }


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def _summary(readiness: dict[str, Any], blocked: pd.DataFrame) -> str:
    return "\n".join(
        [
            "# Layer1 t05st01 Official Announcement Asof Contract",
            "",
            f"Status: {readiness['status']}",
            "",
            "Boundary: asof contract/sample design only; no full ingest, no Experiments, no replay, no formal/report/trade change.",
            "",
            "Readiness:",
            f"- ready_for_core_official_announcement_timestamp_asof_contract={str(readiness['ready_for_core_official_announcement_timestamp_asof_contract']).lower()}",
            "- ready_for_core_exact_filing_asof_join=false",
            "- ready_for_full_ingest=false",
            "- ready_for_experiments=false",
            "- ready_for_formal=false",
            "- ready_for_strategy_replay=false",
            f"- market_available_at_source={readiness['market_available_at_source']}",
            "- exact_internal_filing_upload_timestamp_found=false",
            "- after_close_next_trading_day_policy_required=true",
            f"- future_data_violation_count={readiness['future_data_violation_count']}",
            "",
            "Blocked / prohibited fields:",
            *[f"- {row.field}: {row.status}; {row.reason}" for row in blocked.itertuples()],
            "",
            "Flags:",
            "- formal_model_changed=false",
            "- trade_decision_changed=false",
            "- active_in_trade_decision=false",
            "- report_changed=false",
            "- portfolio_replay_executed=false",
            "- ready_for_strategy_replay=false",
            "- not_live_rule=true",
            "- forward_returns_live_rule_usage=false",
        ]
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--radar-dir", type=Path, default=DEFAULT_RADAR_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    manifest = build_t05st01_asof_contract(radar_dir=args.radar_dir, output_dir=args.output_dir)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
