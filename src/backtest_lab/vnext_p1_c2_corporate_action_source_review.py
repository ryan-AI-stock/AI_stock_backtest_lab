from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
RADAR_DIR = Path(
    "C:/Users/zergv/Documents/Codex/2026-05-23/ai-stock-rotation-radar-https-docs/outputs/"
    "radar_vnext_p1_c2_consensus4_selected_stock_corporate_action_source_package_20260708"
)
OUTPUT_DIR = REPO_ROOT / "outputs" / "vnext_p1_c2_corporate_action_source_review_20260708"

TASK_ID = "TASK-BACKTEST-CORE-VNEXT-P1-C2-CORPORATE-ACTION-SOURCE-PACKAGE-REVIEW-001"
UPSTREAM_RADAR_TASK_ID = "TASK-RADAR-DATA-VNEXT-P1-C2-CONSENSUS4-SELECTED-STOCK-CORPORATE-ACTION-SOURCE-PACKAGE-001"
FLAGS = {
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


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv_safe(path: Path, columns: list[str]) -> pd.DataFrame:
    try:
        return pd.read_csv(path, low_memory=False)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=columns)


def build() -> dict[str, Any]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    readiness_in = _read_json(RADAR_DIR / "readiness_for_core_p1_c2_corporate_action_adjustment_contract.json")
    source_manifest = pd.read_csv(RADAR_DIR / "p1_c2_selected_stock_corporate_action_source_manifest.csv", low_memory=False)
    blocked = pd.read_csv(RADAR_DIR / "p1_c2_selected_stock_corporate_action_blocked_ledger.csv", low_memory=False)
    events = _read_csv_safe(
        RADAR_DIR / "p1_c2_selected_stock_corporate_action_events.csv",
        ["signal_date", "ticker", "event_date", "event_type", "source_quality"],
    )
    factors = _read_csv_safe(
        RADAR_DIR / "p1_c2_selected_stock_adjustment_factor_source_candidates.csv",
        ["signal_date", "ticker", "factor_source", "source_quality"],
    )

    review = pd.DataFrame(
        [
            {
                "core_task_id": TASK_ID,
                "upstream_radar_task_id": UPSTREAM_RADAR_TASK_ID,
                "radar_output": str(RADAR_DIR),
                "source_manifest_rows": readiness_in["coverage"]["source_manifest_rows"],
                "corporate_action_event_candidate_rows": readiness_in["coverage"]["corporate_action_event_candidate_rows"],
                "adjustment_factor_source_candidate_rows": readiness_in["coverage"]["adjustment_factor_source_candidate_rows"],
                "blocked_no_candidate_rows": readiness_in["coverage"]["blocked_no_candidate_rows"],
                "core_review_status": "blocked_no_core_adjustment_contract_from_bounded_no_candidate_package",
                "can_conclude_no_adjustment_needed": False,
                "reason": "MOPS bounded material-information windows found no candidates, but historical TWSE/TPEx ex-right/capital-change route is not unlocked; no-event inference is not defensible.",
                "ready_for_core_p1_c2_corporate_action_adjustment_contract": False,
                "ready_for_p1_c2_market_health_consensus4_net_cost_diagnostic": False,
                "ready_for_experiments": False,
                "ready_for_formal": False,
                "future_data_violation_count": 0,
                **FLAGS,
            }
        ]
    )
    policy = pd.DataFrame(
        [
            {
                "option": "stop_adjusted_close_route_keep_blocked",
                "owner": "Strategy Center",
                "status": "policy_decision",
                "effect": "P1 C2 adjusted net-cost diagnostic remains blocked; unadjusted comparator can remain proxy-only.",
            },
            {
                "option": "authorize_broader_official_historical_exright_capital_change_route_unlock",
                "owner": "Radar/Data",
                "status": "larger_source_acquisition_required",
                "effect": "Try to unlock official historical ex-right/dividend/capital-change source beyond MOPS t05st01 bounded windows.",
            },
            {
                "option": "authorize_licensed_adjusted_close_source",
                "owner": "Strategy Center / Radar/Data",
                "status": "requires_source_policy_or_license",
                "effect": "Could fill selected-stock adjusted close directly if license/access accepted.",
            },
        ]
    )
    future = pd.DataFrame(
        [
            {
                "audit_item": "corporate_action_no_candidate_inference",
                "status": "blocked",
                "violation_count": 0,
                "notes": "No candidate rows from bounded MOPS windows are not enough to infer no adjustment is needed.",
            },
            {
                "audit_item": "future_data_usage",
                "status": "pass",
                "violation_count": 0,
                "notes": "Core review did not calculate adjusted close or use future returns.",
            },
        ]
    )

    readiness = {
        "task_id": TASK_ID,
        "status": "p1_c2_corporate_action_source_review_blocked_no_adjustment_contract",
        "upstream_radar_task_id": UPSTREAM_RADAR_TASK_ID,
        "ready_for_core_p1_c2_corporate_action_adjustment_contract": False,
        "ready_for_p1_c2_market_health_consensus4_net_cost_diagnostic": False,
        "ready_for_experiments": False,
        "ready_for_formal": False,
        "ready_for_strategy_replay": False,
        "corporate_action_event_candidate_rows": int(readiness_in["coverage"]["corporate_action_event_candidate_rows"]),
        "adjustment_factor_source_candidate_rows": int(readiness_in["coverage"]["adjustment_factor_source_candidate_rows"]),
        "blocked_no_candidate_rows": int(readiness_in["coverage"]["blocked_no_candidate_rows"]),
        "can_conclude_no_adjustment_needed": False,
        "future_data_violation_count": 0,
        "next_owner": "Strategy Center",
        "not_live_rule": True,
        "forward_returns_live_rule_usage": False,
        **FLAGS,
    }

    artifacts = {
        "p1_c2_corporate_action_source_review.csv": review,
        "p1_c2_corporate_action_source_manifest.csv": source_manifest,
        "p1_c2_corporate_action_blocked_ledger.csv": blocked,
        "p1_c2_corporate_action_events.csv": events,
        "p1_c2_adjustment_factor_source_candidates.csv": factors,
        "p1_c2_corporate_action_policy_options.csv": policy,
        "p1_c2_corporate_action_review_future_data_audit.csv": future,
    }
    files: list[Path] = []
    for name, df in artifacts.items():
        path = OUTPUT_DIR / name
        df.to_csv(path, index=False, encoding="utf-8-sig")
        files.append(path)

    readiness_path = OUTPUT_DIR / "readiness_for_p1_c2_corporate_action_source_review.json"
    readiness_path.write_text(json.dumps(readiness, ensure_ascii=False, indent=2), encoding="utf-8")
    files.append(readiness_path)

    summary = "\n".join(
        [
            "# P1 C2 corporate-action source review",
            "",
            "- Radar bounded corporate-action source package returned 0 event candidates and 0 adjustment factor candidates.",
            "- Core does not accept this as proof that no adjustment is needed, because official historical ex-right/capital-change routes remain unavailable.",
            "- ready_for_core_p1_c2_corporate_action_adjustment_contract=false.",
            "- ready_for_p1_c2_market_health_consensus4_net_cost_diagnostic=false.",
            "- No adjusted close calculation, no silent fill, no Experiments handoff.",
            "",
            "下一步需要 Strategy Center 決策：停止 adjusted close route and keep blocked、授權更大的官方 historical ex-right/capital-change route unlock，或授權 licensed adjusted-close source。",
            "",
            "Flags: formal_model_changed=false; trade_decision_changed=false; active_in_trade_decision=false; report_changed=false; portfolio_replay_executed=false; ready_for_strategy_replay=false; ready_for_formal=false; not_live_rule=true; forward_returns_live_rule_usage=false.",
            "",
            "完成後如果下一棒明確，請直接指派下一個 thread；如果下一棒不明確，請回報 Strategy Center 判斷。不要完成後停住不回報。",
        ]
    )
    summary_path = OUTPUT_DIR / "final_summary_zh.md"
    summary_path.write_text(summary, encoding="utf-8")
    files.append(summary_path)

    manifest = {
        "task_id": TASK_ID,
        "status": readiness["status"],
        "created_at": pd.Timestamp.now(tz="Asia/Taipei").isoformat(),
        "output_dir": str(OUTPUT_DIR),
        "inputs": {"radar_output": str(RADAR_DIR)},
        "artifacts": [
            {
                "name": path.name,
                "path": str(path),
                "sha256": _sha256(path),
                "rows": int(pd.read_csv(path, low_memory=False).shape[0]) if path.suffix == ".csv" else None,
            }
            for path in files
        ],
        "readiness": readiness,
        "flags": FLAGS,
    }
    manifest_path = OUTPUT_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return readiness


if __name__ == "__main__":
    print(json.dumps(build(), ensure_ascii=False, indent=2))
