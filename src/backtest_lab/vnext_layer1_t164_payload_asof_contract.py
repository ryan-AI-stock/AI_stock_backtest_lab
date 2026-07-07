"""Build t164 payload replay and diagnostic-only as-of policy contract.

This stages Core/Data contract artifacts from Radar/Data payload/asof/TPEx
follow-up. It does not execute full-universe ingest or any experiment/replay.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


TASK_ID = "TASK-BACKTEST-CORE-VNEXT-LAYER1-T164-PAYLOAD-REPLAY-CONSERVATIVE-ASOF-PROXY-CONTRACT-001"
DEFAULT_RADAR_DIR = Path(
    r"C:\Users\zergv\Documents\Codex\2026-05-23\ai-stock-rotation-radar-https-docs\outputs"
    r"\radar_vnext_layer1_t164_payload_asof_tpex_followup_20260707"
)
DEFAULT_OUTPUT_DIR = Path("outputs/vnext_layer1_t164_payload_replay_conservative_asof_contract_20260707")


def build_t164_payload_asof_contract(
    *,
    radar_dir: str | Path = DEFAULT_RADAR_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    radar = Path(radar_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    readiness_in = _read_json(radar / "readiness_for_core_t164_cashflow_inventory_receivable_full_ingest.json")
    payload_capture = _read_csv(radar / "t164_browser_equivalent_payload_capture.csv")
    replay_ledger = _read_csv(radar / "t164_direct_replay_feasibility_ledger.csv")
    asof_routes = _read_csv(radar / "mops_disclosure_asof_source_route.csv")
    taxonomy = _read_csv(radar / "t164_label_taxonomy_policy.csv")
    tpex_samples = _read_csv(radar / "t164_tpex_bounded_sample_confirmation.csv")
    blocked_in = _read_csv(radar / "blocked_prerequisites_ledger.csv")

    payload_contract = _payload_replay_contract(payload_capture, replay_ledger)
    asof_policy = _asof_policy_matrix(asof_routes)
    taxonomy_review = _label_taxonomy_review(taxonomy)
    cashflow_inventory_payload = _cashflow_inventory_payload_contract(payload_capture, tpex_samples)
    blocked = _blocked_prohibited_ledger(asof_policy, blocked_in)
    readiness = _readiness_json(readiness_in, payload_contract, asof_policy, taxonomy_review, tpex_samples, blocked)

    _write_csv(payload_contract, output / "layer1_t164_payload_replay_contract.csv")
    _write_csv(asof_policy, output / "layer1_t164_conservative_asof_policy_matrix.csv")
    _write_csv(taxonomy_review, output / "layer1_t164_label_taxonomy_human_review.csv")
    _write_csv(cashflow_inventory_payload, output / "layer1_t164_cashflow_inventory_receivable_payload_contract.csv")
    _write_csv(blocked, output / "layer1_t164_blocked_prohibited_asof_ledger.csv")
    (output / "readiness_for_layer1_t164_diagnostic_only_contract.json").write_text(
        json.dumps(readiness, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    manifest = {
        "task_id": TASK_ID,
        "status": readiness["status"],
        "output_dir": str(output.resolve()),
        "radar_input_dir": str(radar.resolve()),
        "radar_payload_asof_tpex_commit": "70b798b",
        "output_files": [
            "layer1_t164_payload_replay_contract.csv",
            "layer1_t164_conservative_asof_policy_matrix.csv",
            "layer1_t164_label_taxonomy_human_review.csv",
            "layer1_t164_cashflow_inventory_receivable_payload_contract.csv",
            "layer1_t164_blocked_prohibited_asof_ledger.csv",
            "readiness_for_layer1_t164_diagnostic_only_contract.json",
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


def _payload_replay_contract(payload_capture: pd.DataFrame, replay_ledger: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in payload_capture.itertuples(index=False):
        payload = json.loads(row.sanitized_payload_schema)
        rows.append(
            {
                "api": row.api,
                "endpoint": row.endpoint,
                "method": row.method,
                "payload_case": row.payload_case,
                "dataType": payload.get("dataType"),
                "year_required": "year" in payload,
                "year_value": payload.get("year"),
                "season_required": "season" in payload,
                "season_value": payload.get("season"),
                "subsidiaryCompanyId_required": "subsidiaryCompanyId" in payload,
                "subsidiaryCompanyId_value": payload.get("subsidiaryCompanyId"),
                "payload_rule": _payload_rule(payload),
                "required_context": row.required_context,
                "bounded_result": row.result,
                "direct_browser_equivalent_replay_feasible": bool(row.direct_browser_equivalent_replay_feasible),
                "tpex_universal_ready": False,
                "accepted_for_formal": False,
                "human_review_required": True,
                "diagnostic_only": True,
                "not_live_rule": True,
            }
        )
    return pd.DataFrame(rows)


def _payload_rule(payload: dict[str, Any]) -> str:
    if payload.get("dataType") == "1":
        return 'latest query must preserve year="", season="", subsidiaryCompanyId=""'
    return 'custom period uses dataType=2, ROC year, season 1-4, subsidiaryCompanyId=""'


def _asof_policy_matrix(asof_routes: pd.DataFrame) -> pd.DataFrame:
    rows = [
        (
            "exact_official_filing_timestamp",
            "blocked",
            "exact",
            "Required for exact PIT/full ingest/formal; Radar route t163sb01 has announcement text but no exact datetime",
            "not_ready",
        ),
        (
            "conservative_filing_deadline_proxy",
            "diagnostic_only_candidate",
            "proxy",
            "Strategy Center accepted for diagnostic-only staging; use statutory filing deadline per report period, never quarter end",
            "ready_for_diagnostic_only_contract",
        ),
        (
            "quarter_end_date",
            "prohibited",
            "invalid",
            "Quarter-end is before disclosure and would introduce future-data risk",
            "never_use",
        ),
        (
            "query_response_datetime",
            "prohibited",
            "invalid",
            "Response datetime is query time, not historical availability",
            "never_use",
        ),
    ]
    return pd.DataFrame(
        [
            {
                "asof_policy": policy,
                "status": status,
                "source_quality": quality,
                "policy_detail": detail,
                "readiness": readiness,
                "ready_for_full_ingest": False,
                "ready_for_formal": False,
                "diagnostic_only": status == "diagnostic_only_candidate",
            }
            for policy, status, quality, detail, readiness in rows
        ]
    )


def _label_taxonomy_review(taxonomy: pd.DataFrame) -> pd.DataFrame:
    if taxonomy.empty:
        return pd.DataFrame()
    out = taxonomy.copy()
    out["core_policy_status"] = out["canonical_field"].map(
        {
            "operating_cash_flow": "narrow_mapping_candidate",
            "investing_cash_flow": "narrow_mapping_candidate",
            "capex_proxy": "human_policy_required",
            "inventory": "narrow_mapping_candidate",
            "receivables_basket": "human_policy_required",
        }
    ).fillna("human_policy_required")
    out["ready_for_formal"] = False
    out["diagnostic_only"] = True
    return out


def _cashflow_inventory_payload_contract(payload_capture: pd.DataFrame, tpex_samples: pd.DataFrame) -> pd.DataFrame:
    tpex_confirmed = bool(not tpex_samples.empty and tpex_samples["bounded_sample_confirmation"].astype(bool).all())
    rows = []
    for api in ["t164sb05", "t164sb03"]:
        api_rows = payload_capture[payload_capture["api"].eq(api)]
        for row in api_rows.itertuples(index=False):
            rows.append(
                {
                    "api": row.api,
                    "endpoint": row.endpoint,
                    "payload_case": row.payload_case,
                    "payload_schema": row.sanitized_payload_schema,
                    "field_family": "cashflow" if api == "t164sb05" else "inventory_receivable_balance_sheet",
                    "bounded_twse_sample_confirmed": True,
                    "bounded_tpex_sample_confirmed": tpex_confirmed,
                    "tpex_universal_ready": False,
                    "asof_policy_allowed": "conservative_filing_deadline_proxy_diagnostic_only",
                    "exact_filing_timestamp_blocked": True,
                    "ready_for_full_ingest": False,
                    "ready_for_experiments": False,
                    "diagnostic_only": True,
                    "not_live_rule": True,
                }
            )
    return pd.DataFrame(rows)


def _blocked_prohibited_ledger(asof_policy: pd.DataFrame, blocked_in: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in asof_policy.itertuples(index=False):
        if row.status in {"blocked", "prohibited"}:
            rows.append(
                {
                    "item": row.asof_policy,
                    "status": row.status,
                    "reason": row.policy_detail,
                    "next_action": "Radar/Data exact filing timestamp route required" if row.status == "blocked" else "Do not use",
                    "ready_for_full_ingest": False,
                    "diagnostic_only": False,
                }
            )
    if not blocked_in.empty:
        for row in blocked_in.itertuples(index=False):
            rows.append(
                {
                    "item": row.prerequisite,
                    "status": row.status,
                    "reason": row.impact,
                    "next_action": row.next_action,
                    "ready_for_full_ingest": False,
                    "diagnostic_only": True,
                }
            )
    return pd.DataFrame(rows)


def _readiness_json(
    readiness_in: dict[str, Any],
    payload_contract: pd.DataFrame,
    asof_policy: pd.DataFrame,
    taxonomy_review: pd.DataFrame,
    tpex_samples: pd.DataFrame,
    blocked: pd.DataFrame,
) -> dict[str, Any]:
    payload_ready = bool(readiness_in.get("direct_browser_equivalent_payload_replay", False)) and not payload_contract.empty
    tpex_sample = bool(readiness_in.get("tpex_sample_confirmation", False))
    diagnostic_asof = bool(asof_policy["status"].eq("diagnostic_only_candidate").any())
    ready_diag = payload_ready and diagnostic_asof
    return {
        "date": "2026-07-07",
        "task_id": TASK_ID,
        "owner": "BACKTEST_LAB Core/Data",
        "status": "diagnostic_only_payload_replay_contract_ready_exact_asof_blocked" if ready_diag else "blocked_t164_payload_replay_contract",
        "ready_for_layer1_t164_diagnostic_only_contract": ready_diag,
        "ready_for_core_payload_replay_contract": payload_ready,
        "ready_for_core_t164_cashflow_inventory_receivable_full_ingest": False,
        "ready_for_experiments": False,
        "ready_for_formal": False,
        "ready_for_strategy_replay": False,
        "direct_browser_equivalent_payload_replay": payload_ready,
        "tpex_sample_confirmation": tpex_sample,
        "tpex_universal_ready": False,
        "mops_disclosure_datetime_asof_join": False,
        "exact_official_filing_timestamp_status": "blocked",
        "conservative_filing_deadline_proxy_status": "diagnostic_only_candidate",
        "quarter_end_date_status": "prohibited",
        "query_response_datetime_status": "prohibited",
        "label_taxonomy_policy_ready_for_human_review": bool(readiness_in.get("label_taxonomy_policy_ready_for_human_review", False)),
        "human_policy_required_fields": taxonomy_review[taxonomy_review["core_policy_status"].eq("human_policy_required")]["canonical_field"].tolist() if not taxonomy_review.empty else [],
        "future_data_violation_count": int(readiness_in.get("future_data_violation_count", 0)),
        "blocked_items": blocked["item"].tolist(),
        "required_next_step": "vNext Research/Strategy judge interim diagnostic-only package vs wait for exact asof; Radar still needed for exact filing timestamp",
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
            "# Layer1 t164 Payload Replay Conservative Asof Contract",
            "",
            f"Status: {readiness['status']}",
            "",
            "Boundary: diagnostic-only payload/asof contract staging; no full ingest, no Experiments, no replay, no formal/report/trade change.",
            "",
            "Readiness:",
            f"- ready_for_layer1_t164_diagnostic_only_contract={str(readiness['ready_for_layer1_t164_diagnostic_only_contract']).lower()}",
            "- ready_for_core_t164_cashflow_inventory_receivable_full_ingest=false",
            "- ready_for_experiments=false",
            "- ready_for_formal=false",
            "- ready_for_strategy_replay=false",
            "- exact_official_filing_timestamp_status=blocked",
            "- conservative_filing_deadline_proxy_status=diagnostic_only_candidate",
            "- quarter_end_date_status=prohibited",
            "- query_response_datetime_status=prohibited",
            f"- future_data_violation_count={readiness['future_data_violation_count']}",
            "",
            "Blocked / prohibited:",
            *[f"- {row.item}: {row.status}; {row.reason}" for row in blocked.itertuples()],
            "",
            "Next handoff:",
            "- vNext Research / Strategy should decide whether to include this interim diagnostic-only asof package or wait for exact official filing timestamp.",
            "- Radar/Data still needed for exact filing timestamp route.",
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
    manifest = build_t164_payload_asof_contract(radar_dir=args.radar_dir, output_dir=args.output_dir)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
