"""Build Layer 1 t164sb05/t164sb03 parser contract design readiness.

This consumes Radar/Data browser-route capture samples and emits Core-side
parser contract design artifacts. It does not perform full ingest or replay.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


TASK_ID = "TASK-BACKTEST-CORE-VNEXT-LAYER1-T164SB05-T164SB03-PARSER-CONTRACT-DESIGN-001"
DEFAULT_RADAR_DIR = Path(
    r"C:\Users\zergv\Documents\Codex\2026-05-23\ai-stock-rotation-radar-https-docs\outputs"
    r"\radar_vnext_layer1_cashflow_inventory_receivable_browser_route_capture_20260707"
)
DEFAULT_OUTPUT_DIR = Path("outputs/vnext_layer1_t164sb05_t164sb03_parser_contract_design_20260707")


def build_parser_contract_design(
    *,
    radar_dir: str | Path = DEFAULT_RADAR_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    radar = Path(radar_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    readiness_in = _read_json(radar / "readiness_for_core_layer1_cashflow_inventory_receivable_ingest.json")
    route_ledger = _read_csv(radar / "browser_route_capture_ledger.csv")
    mapping = _read_csv(radar / "field_mapping_cashflow_inventory_receivable.csv")
    cashflow_sample = _read_csv(radar / "cashflow_route_payload_response_sample.csv")
    balance_sample = _read_csv(radar / "balance_sheet_detail_payload_response_sample.csv")
    timing = _read_csv(radar / "pit_timing_capture_audit.csv")
    radar_blocked = _read_csv(radar / "blocked_security_or_policy_ledger.csv")

    cashflow_design = _cashflow_contract_design(cashflow_sample, mapping, route_ledger)
    balance_design = _balance_sheet_contract_design(balance_sample, mapping, route_ledger)
    normalized_schema = _normalized_schema(cashflow_design, balance_design)
    pit_timing = _pit_timing_requirements(timing)
    payload_prereqs = _payload_prerequisites(route_ledger)
    blocked = _blocked_prerequisites(radar_blocked)
    readiness = _readiness_json(readiness_in, cashflow_design, balance_design, normalized_schema, pit_timing, blocked)

    _write_csv(cashflow_design, output / "layer1_t164sb05_cashflow_parser_contract_design.csv")
    _write_csv(balance_design, output / "layer1_t164sb03_balance_sheet_parser_contract_design.csv")
    _write_csv(normalized_schema, output / "layer1_cashflow_inventory_receivable_normalized_schema.csv")
    _write_csv(pit_timing, output / "layer1_cashflow_inventory_receivable_pit_timing_requirements.csv")
    _write_csv(payload_prereqs, output / "layer1_browser_equivalent_payload_prerequisites.csv")
    _write_csv(blocked, output / "blocked_prerequisites_ledger.csv")
    (output / "readiness_for_layer1_cashflow_inventory_receivable_ingest.json").write_text(
        json.dumps(readiness, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    manifest = {
        "task_id": TASK_ID,
        "status": readiness["status"],
        "output_dir": str(output.resolve()),
        "radar_input_dir": str(radar.resolve()),
        "radar_browser_route_capture_commit": "ba63444",
        "output_files": [
            "layer1_t164sb05_cashflow_parser_contract_design.csv",
            "layer1_t164sb03_balance_sheet_parser_contract_design.csv",
            "layer1_cashflow_inventory_receivable_normalized_schema.csv",
            "layer1_cashflow_inventory_receivable_pit_timing_requirements.csv",
            "layer1_browser_equivalent_payload_prerequisites.csv",
            "blocked_prerequisites_ledger.csv",
            "readiness_for_layer1_cashflow_inventory_receivable_ingest.json",
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


def _route_row(route_ledger: pd.DataFrame, route_id: str) -> dict[str, Any]:
    if route_ledger.empty:
        return {}
    row = route_ledger[route_ledger["route_id"].eq(route_id)]
    if row.empty:
        return {}
    return row.iloc[0].to_dict()


def _cashflow_contract_design(cashflow_sample: pd.DataFrame, mapping: pd.DataFrame, route_ledger: pd.DataFrame) -> pd.DataFrame:
    route_id = "mops_spa_t164sb05_cashflow_latest_ui"
    route = _route_row(route_ledger, route_id)
    sample = cashflow_sample.iloc[0].to_dict() if not cashflow_sample.empty else {}
    rows = [
        (
            "operating_cash_flow",
            sample.get("operating_cash_flow_label", "營業活動之淨現金流入（流出）"),
            "operating_cash_flow_current",
            "TWD_thousand",
            "cash_flow_quality_input",
            "sample_unlocked_not_full_contract",
            "OCF value can support OCF positive floor or OCF/net_income after PIT statement ingest",
        ),
        (
            "investing_cash_flow",
            sample.get("investing_cash_flow_label", "投資活動之淨現金流入（流出）"),
            "investing_cash_flow_current",
            "TWD_thousand",
            "context_only",
            "sample_unlocked_not_full_contract",
            "context for cash-flow profile; not a standalone eligibility rule",
        ),
        (
            "capex_proxy",
            sample.get("capex_proxy_label", "取得不動產、廠房及設備"),
            "capex_proxy_current",
            "TWD_thousand",
            "free_cash_flow_proxy_input",
            "sample_unlocked_proxy_policy_required",
            "FCF proxy policy required because capex labels vary by company/profile",
        ),
        (
            "inventory_change_cashflow",
            sample.get("inventory_change_label", "存貨（增加）減少"),
            "inventory_change_current",
            "TWD_thousand",
            "inventory_context_input",
            "sample_unlocked_not_full_contract",
            "cash-flow statement working-capital change; not balance-sheet inventory level",
        ),
        (
            "receivable_change_cashflow",
            sample.get("receivable_change_label", "應收帳款（增加）減少"),
            "receivable_change_current",
            "TWD_thousand",
            "receivable_context_input",
            "sample_unlocked_not_full_contract",
            "cash-flow statement working-capital change; label taxonomy must be reviewed",
        ),
    ]
    return pd.DataFrame(
        [
            {
                "route_id": route_id,
                "endpoint_candidate": route.get("endpoint_candidate", "https://mops.twse.com.tw/mops/api/t164sb05"),
                "method": route.get("method", "POST"),
                "normalized_field": field,
                "source_label": label,
                "sample_column": column,
                "unit": unit,
                "field_role": role,
                "source_quality": quality,
                "contract_note": note,
                "sample_ticker": sample.get("ticker"),
                "sample_period_label": sample.get("period_label"),
                "sample_schema_status": sample.get("schema_status"),
                "accepted_for_formal": False,
                "human_review_required": True,
                "diagnostic_only": True,
                "not_live_rule": True,
            }
            for field, label, column, unit, role, quality, note in rows
        ]
    )


def _balance_sheet_contract_design(balance_sample: pd.DataFrame, mapping: pd.DataFrame, route_ledger: pd.DataFrame) -> pd.DataFrame:
    route_id = "mops_spa_t164sb03_balance_sheet_latest_ui"
    route = _route_row(route_ledger, route_id)
    sample = balance_sample.iloc[0].to_dict() if not balance_sample.empty else {}
    rows = [
        ("inventory", sample.get("inventory_label", "存貨"), "inventory_current", "TWD_thousand", "inventory_risk_input", "sample_unlocked_not_full_contract", "inventory/current_assets or inventory growth after PIT panel is materialized"),
        ("receivable_trade", sample.get("receivable_trade_label", "應收帳款淨額"), "receivable_trade_current", "TWD_thousand", "receivable_risk_input", "sample_unlocked_not_full_contract", "part of receivable basket; label taxonomy required"),
        ("notes_receivable", sample.get("notes_receivable_label", "應收票據淨額"), "notes_receivable_current", "TWD_thousand", "receivable_risk_input", "sample_unlocked_not_full_contract", "part of receivable basket; label taxonomy required"),
        ("other_receivable", sample.get("other_receivable_label", "其他應收款淨額"), "other_receivable_current", "TWD_thousand", "receivable_risk_input", "sample_unlocked_not_full_contract", "part of receivable basket; policy required"),
        ("current_assets", sample.get("current_assets_label", "流動資產合計"), "current_assets_current", "TWD_thousand", "denominator_and_current_ratio_input", "sample_unlocked_not_full_contract", "current ratio denominator and inventory/receivable denominator candidate"),
        ("current_liabilities", sample.get("current_liabilities_label", "流動負債合計"), "current_liabilities_current", "TWD_thousand", "current_ratio_input", "sample_unlocked_not_full_contract", "current ratio denominator candidate"),
    ]
    return pd.DataFrame(
        [
            {
                "route_id": route_id,
                "endpoint_candidate": route.get("endpoint_candidate", "https://mops.twse.com.tw/mops/api/t164sb03"),
                "method": route.get("method", "POST"),
                "normalized_field": field,
                "source_label": label,
                "sample_column": column,
                "unit": unit,
                "field_role": role,
                "source_quality": quality,
                "contract_note": note,
                "sample_ticker": sample.get("ticker"),
                "sample_period_label": sample.get("period_label"),
                "sample_schema_status": sample.get("schema_status"),
                "accepted_for_formal": False,
                "human_review_required": True,
                "diagnostic_only": True,
                "not_live_rule": True,
            }
            for field, label, column, unit, role, quality, note in rows
        ]
    )


def _normalized_schema(cashflow_design: pd.DataFrame, balance_design: pd.DataFrame) -> pd.DataFrame:
    rows = [
        ("statement_source", "enum", "t164sb05_cashflow|t164sb03_balance_sheet", "required", "route source"),
        ("ticker", "string", "TW ticker", "required", "join key"),
        ("market", "string", "TWSE/TPEx if available", "required", "coverage audit key"),
        ("report_period", "string", "fiscal period, e.g. 2026Q1", "required", "period key"),
        ("period_label_raw", "string", "民國115年第1季", "required", "source trace"),
        ("source_label", "string", "raw MOPS label", "required", "parser trace"),
        ("normalized_field", "string", "one row per mapped field", "required", "canonical field"),
        ("value", "decimal", "numeric value", "required", "TWD thousand unless unit says otherwise"),
        ("unit", "string", "TWD_thousand", "required", "unit audit"),
        ("available_datetime", "datetime", "MOPS disclosure/filing datetime", "blocked_prerequisite", "PIT as-of"),
        ("source_capture_method", "enum", "browser_ui|browser_equivalent_payload|direct_api", "required", "route replay audit"),
        ("payload_contract_version", "string", "versioned sanitized payload", "blocked_prerequisite", "replay audit"),
        ("source_quality", "enum", "sample_unlocked|PIT_ready|proxy|blocked", "required", "readiness audit"),
        ("human_review_required", "boolean", "true", "required", "label taxonomy control"),
        ("diagnostic_only", "boolean", "true", "required", "boundary flag"),
    ]
    return pd.DataFrame(
        [
            {
                "column": column,
                "dtype": dtype,
                "allowed_or_example": example,
                "requirement_level": level,
                "purpose": purpose,
                "diagnostic_only": True,
            }
            for column, dtype, example, level, purpose in rows
        ]
    )


def _pit_timing_requirements(timing: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in timing.itertuples(index=False):
        rows.append(
            {
                "route_id": row.route_id,
                "period_observed": row.period_observed,
                "observed_current_date": row.observed_current_date,
                "required_available_datetime_source": "MOPS disclosure/financial-report filing datetime, not browser latest page date",
                "historical_asof_rule": "available_datetime <= signal_date; if exact filing timestamp unavailable, conservative statutory release date must be explicit",
                "current_status": "blocked_for_full_ingest",
                "future_data_violation_risk": row.future_data_violation_risk,
                "future_data_violation_count": row.future_data_violation_count,
                "ready_for_core_ingest": False,
                "diagnostic_only": True,
            }
        )
    return pd.DataFrame(rows)


def _payload_prerequisites(route_ledger: pd.DataFrame) -> pd.DataFrame:
    rows = [
        (
            "browser_equivalent_payload_replay",
            "blocked",
            "direct POST returned code=500 parameter exception without full browser context",
            "Capture full SPA payload/context or browser-equivalent request transformation for /mops/api/t164sb05 and /mops/api/t164sb03",
        ),
        (
            "payload_versioning",
            "blocked",
            "latest UI sample only; no versioned reusable payload contract",
            "Persist sanitized payload schema including companyId/dataType/year/season/subsidiaryCompanyId and any SPA context fields",
        ),
        (
            "session_cookie_policy",
            "blocked",
            "cookie/session not saved in Radar package",
            "Define allowed browser-context replay policy without bypassing security blocks",
        ),
        (
            "TPEx_sample_confirmation",
            "blocked",
            "TPEx multi-sample automation unstable; not evidence of universal route readiness",
            "Run bounded stable TPEx sample after payload replay is solved",
        ),
        (
            "legacy_ajax_route",
            "blocked",
            "legacy ajax routes returned security block pages",
            "Do not use or bypass legacy ajax security block; prefer SPA routes",
        ),
    ]
    return pd.DataFrame(
        [
            {
                "prerequisite": name,
                "status": status,
                "evidence": evidence,
                "next_action": action,
                "ready_for_full_ingest": False,
                "diagnostic_only": True,
            }
            for name, status, evidence, action in rows
        ]
    )


def _blocked_prerequisites(radar_blocked: pd.DataFrame) -> pd.DataFrame:
    base = [
        ("MOPS_disclosure_datetime_asof_join", "blocked", "browser latest page alone is not sufficient for historical PIT", "Acquire/join MOPS filing/disclosure datetime per ticker/report period"),
        ("direct_browser_equivalent_payload_replay", "blocked", "direct API minimal payload returns code=500", "Capture SPA-equivalent payload/context"),
        ("TPEx_sample_confirmation", "blocked", "multi-sample automation unstable", "Run bounded TPEx sample with stable browser/session or solved payload replay"),
        ("full_universe_ingest_runner", "blocked", "only 1101 latest browser sample exists", "Build resumable full-universe runner only after payload/asof prerequisites pass"),
        ("label_taxonomy_policy", "blocked", "capex/receivable labels vary by company/profile", "Human-review normalized label taxonomy before diagnostic ingest"),
    ]
    rows = [
        {
            "blocked_item": item,
            "status": status,
            "blocked_reason": reason,
            "next_action": action,
            "source": "Core contract design",
            "ready_for_full_ingest": False,
            "diagnostic_only": True,
        }
        for item, status, reason, action in base
    ]
    if not radar_blocked.empty:
        for row in radar_blocked.itertuples(index=False):
            rows.append(
                {
                    "blocked_item": row.field_or_route,
                    "status": row.status,
                    "blocked_reason": row.blocked_reason,
                    "next_action": row.next_step,
                    "source": "Radar blocked ledger",
                    "ready_for_full_ingest": False,
                    "diagnostic_only": True,
                }
            )
    return pd.DataFrame(rows)


def _readiness_json(
    readiness_in: dict[str, Any],
    cashflow_design: pd.DataFrame,
    balance_design: pd.DataFrame,
    normalized_schema: pd.DataFrame,
    pit_timing: pd.DataFrame,
    blocked: pd.DataFrame,
) -> dict[str, Any]:
    future_count = int(readiness_in.get("future_data_violation_count", 0))
    design_ready = bool(readiness_in.get("ready_for_core_parser_contract_design", False)) and not cashflow_design.empty and not balance_design.empty
    full_ingest_ready = False
    return {
        "date": "2026-07-07",
        "task_id": TASK_ID,
        "owner": "BACKTEST_LAB Core/Data",
        "status": "parser_contract_design_ready_not_full_ingest" if design_ready else "blocked_parser_contract_design",
        "ready_for_core_parser_contract_design": design_ready,
        "ready_for_core_layer1_cashflow_inventory_receivable_ingest": full_ingest_ready,
        "ready_for_core_rerun": False,
        "ready_for_experiments": False,
        "ready_for_formal": False,
        "ready_for_strategy_replay": False,
        "ready_for_radar_payload_asof_tpex_followup": design_ready,
        "future_data_violation_count": future_count,
        "cashflow_design_rows": int(len(cashflow_design)),
        "balance_sheet_design_rows": int(len(balance_design)),
        "normalized_schema_rows": int(len(normalized_schema)),
        "pit_timing_requirement_rows": int(len(pit_timing)),
        "blocked_prerequisite_rows": int(len(blocked)),
        "blocking_prerequisites": blocked["blocked_item"].dropna().tolist(),
        "required_next_step": "Radar/Data payload/asof/TPEx follow-up before Core full ingest",
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
            "# Layer1 t164sb05/t164sb03 Parser Contract Design",
            "",
            f"Status: {readiness['status']}",
            "",
            "Boundary: parser contract design only; no full ingest, no Experiments, no replay, no formal/report/trade change.",
            "",
            "Readiness:",
            f"- ready_for_core_parser_contract_design={str(readiness['ready_for_core_parser_contract_design']).lower()}",
            "- ready_for_core_layer1_cashflow_inventory_receivable_ingest=false",
            "- ready_for_core_rerun=false",
            "- ready_for_experiments=false",
            "- ready_for_formal=false",
            "- ready_for_strategy_replay=false",
            f"- future_data_violation_count={readiness['future_data_violation_count']}",
            "- not_live_rule=true",
            "",
            "Blocked prerequisites:",
            *[f"- {row.blocked_item}: {row.blocked_reason}" for row in blocked.itertuples()],
            "",
            "Next handoff:",
            "- Radar/Data should capture browser-equivalent payload/context, MOPS disclosure/asof join, and stable TPEx samples before Core full ingest.",
            "",
            "Flags:",
            "- formal_model_changed=false",
            "- trade_decision_changed=false",
            "- active_in_trade_decision=false",
            "- report_changed=false",
            "- portfolio_replay_executed=false",
            "- ready_for_strategy_replay=false",
        ]
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--radar-dir", type=Path, default=DEFAULT_RADAR_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    manifest = build_parser_contract_design(radar_dir=args.radar_dir, output_dir=args.output_dir)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
