from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
RADAR_DIR = Path(
    "C:/Users/zergv/Documents/Codex/2026-05-23/ai-stock-rotation-radar-https-docs/outputs/"
    "radar_vnext_p1_c2_consensus4_selected_stock_adjusted_close_source_fill_20260708"
)
CORE_C2_DIR = REPO_ROOT / "outputs" / "vnext_p1_c2_market_health_consensus4_adjusted_state_machine_contract_20260708"
OUTPUT_DIR = REPO_ROOT / "outputs" / "vnext_p1_c2_consensus4_adjusted_close_source_fill_absorption_20260708"

TASK_ID = "TASK-BACKTEST-CORE-VNEXT-P1-C2-CONSENSUS4-ADJUSTED-CLOSE-SOURCE-FILL-ABSORPTION-001"
UPSTREAM_RADAR_TASK_ID = "TASK-RADAR-DATA-VNEXT-P1-C2-CONSENSUS4-SELECTED-STOCK-ADJUSTED-CLOSE-SOURCE-FILL-001"

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


def _policy_options() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "policy_option": "approve_licensed_third_party_adjusted_close_source",
                "owner": "Strategy Center / Radar/Data",
                "status": "requires_user_or_strategy_approval",
                "expected_effect": "could unlock exact selected-stock adjusted close path if source terms and coverage are acceptable",
                "risk_or_cost": "third-party license/access cost and source governance required",
                "core_position": "do_not_assume_without_policy_approval",
            },
            {
                "policy_option": "build_official_corporate_action_adjustment_contract",
                "owner": "Radar/Data -> Core/Data",
                "status": "larger_engineering_route",
                "expected_effect": "could derive adjusted close from official OHLC plus ex-right/dividend/capital-change events",
                "risk_or_cost": "higher cost; requires PIT corporate-action event coverage and adjustment validation",
                "core_position": "valid_next_route_if_Strategy_Center_accepts_scope",
            },
            {
                "policy_option": "keep_adjusted_state_machine_blocked_use_unadjusted_proxy_only",
                "owner": "Strategy Center",
                "status": "policy_decision_required",
                "expected_effect": "allows qualitative/proxy comparator only; does not satisfy adjusted net-cost diagnostic",
                "risk_or_cost": "proxy caveat remains; cannot promote to formal/replay/main verdict",
                "core_position": "acceptable_only_as_diagnostic_proxy_boundary",
            },
        ]
    )


def build() -> dict[str, Any]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    radar_readiness = _read_json(RADAR_DIR / "readiness_for_core_p1_c2_consensus4_adjusted_close_source_fill.json")
    core_readiness = _read_json(CORE_C2_DIR / "readiness_for_p1_c2_market_health_consensus4_diagnostic.json")
    patch = pd.read_csv(RADAR_DIR / "p1_c2_consensus4_selected_stock_adjusted_close_patch_rows.csv", low_memory=False)
    blocked = pd.read_csv(RADAR_DIR / "p1_c2_consensus4_selected_stock_adjusted_close_blocked_ledger.csv", low_memory=False)
    attempts = pd.read_csv(RADAR_DIR / "adjusted_close_source_route_attempts.csv", low_memory=False)
    missing_scope = pd.read_csv(RADAR_DIR / "p1_c2_consensus4_missing_scope.csv", low_memory=False)

    absorption = pd.DataFrame(
        [
            {
                "core_task_id": TASK_ID,
                "upstream_radar_task_id": UPSTREAM_RADAR_TASK_ID,
                "core_prior_output": str(CORE_C2_DIR),
                "radar_source_fill_output": str(RADAR_DIR),
                "patched_interval_rows": radar_readiness["coverage"]["patched_interval_rows"],
                "remaining_blocked_interval_rows": radar_readiness["coverage"]["remaining_blocked_interval_rows"],
                "entry_exit_adjusted_price_values_requested": radar_readiness["coverage"]["entry_exit_adjusted_price_values_requested"],
                "entry_exit_adjusted_price_values_filled": radar_readiness["coverage"]["entry_exit_adjusted_price_values_filled"],
                "selected_stock_adjusted_close_ready_share_after_radar": core_readiness["selected_stock_adjusted_close_ready_share"],
                "ready_for_p1_c2_market_health_consensus4_net_cost_diagnostic": False,
                "ready_for_unadjusted_ohlc_comparator_diagnostic": True,
                "ready_for_experiments": False,
                "ready_for_formal": False,
                "ready_for_strategy_replay": False,
                "future_data_violation_count": 0,
                "core_absorption_status": "blocked_no_refresh_no_accepted_adjusted_close_patch_rows",
                "next_owner": "Strategy Center",
                "next_decision_required": "choose licensed adjusted-close source, official corporate-action adjustment project, or keep adjusted diagnostic blocked with unadjusted proxy only",
                **FLAGS,
            }
        ]
    )

    policy = _policy_options()
    future_audit = pd.DataFrame(
        [
            {
                "audit_item": "radar_patch_ingest",
                "status": "blocked_no_rows_to_ingest",
                "violation_count": 0,
                "notes": "Radar returned 0 adjusted-close patch rows; Core did not fabricate or silent-fill adjusted prices.",
            },
            {
                "audit_item": "unadjusted_proxy_boundary",
                "status": "preserved",
                "violation_count": 0,
                "notes": "Unadjusted OHLC comparator remains proxy-only and is not promoted to adjusted-close path or formal evidence.",
            },
            {
                "audit_item": "future_return_as_rule",
                "status": "pass",
                "violation_count": 0,
                "notes": "No future return, 00631L+excess reconstruction, or hindsight price fill was used.",
            },
        ]
    )

    readiness = {
        "task_id": TASK_ID,
        "status": "p1_c2_consensus4_adjusted_close_source_fill_absorbed_blocked_no_refresh",
        "upstream_radar_task_id": UPSTREAM_RADAR_TASK_ID,
        "ready_for_p1_c2_market_health_consensus4_net_cost_diagnostic": False,
        "ready_for_core_p1_c2_consensus4_adjusted_state_machine_refresh": False,
        "ready_for_unadjusted_ohlc_comparator_diagnostic": True,
        "ready_for_experiments": False,
        "ready_for_formal": False,
        "ready_for_strategy_replay": False,
        "patched_interval_rows": int(radar_readiness["coverage"]["patched_interval_rows"]),
        "remaining_blocked_interval_rows": int(radar_readiness["coverage"]["remaining_blocked_interval_rows"]),
        "entry_exit_adjusted_price_values_requested": int(radar_readiness["coverage"]["entry_exit_adjusted_price_values_requested"]),
        "entry_exit_adjusted_price_values_filled": int(radar_readiness["coverage"]["entry_exit_adjusted_price_values_filled"]),
        "selected_stock_adjusted_close_ready_share": float(core_readiness["selected_stock_adjusted_close_ready_share"]),
        "adjusted_close_ready": False,
        "future_data_violation_count": 0,
        "next_owner": "Strategy Center",
        "next_decision_required": True,
        "not_live_rule": True,
        "forward_returns_live_rule_usage": False,
        **FLAGS,
    }

    artifacts = {
        "p1_c2_consensus4_adjusted_close_source_fill_absorption.csv": absorption,
        "p1_c2_consensus4_adjusted_close_remaining_blocked_ledger.csv": blocked,
        "p1_c2_consensus4_adjusted_close_source_route_attempts.csv": attempts,
        "p1_c2_consensus4_adjusted_close_missing_scope.csv": missing_scope,
        "p1_c2_consensus4_adjusted_close_patch_rows.csv": patch,
        "p1_c2_consensus4_adjusted_close_policy_options.csv": policy,
        "p1_c2_consensus4_adjusted_close_absorption_future_data_audit.csv": future_audit,
    }
    files: list[Path] = []
    for name, df in artifacts.items():
        path = OUTPUT_DIR / name
        df.to_csv(path, index=False, encoding="utf-8-sig")
        files.append(path)

    readiness_path = OUTPUT_DIR / "readiness_for_p1_c2_consensus4_adjusted_close_absorption.json"
    readiness_path.write_text(json.dumps(readiness, ensure_ascii=False, indent=2), encoding="utf-8")
    files.append(readiness_path)

    summary = "\n".join(
        [
            "# P1 C2 consensus4 adjusted close source fill absorption",
            "",
            f"- task_id: `{TASK_ID}`",
            "- status: `p1_c2_consensus4_adjusted_close_source_fill_absorbed_blocked_no_refresh`",
            "- Radar patch result: patched_interval_rows=0, remaining_blocked_interval_rows=16.",
            "- Core adjusted state-machine refresh is not possible because there are no accepted adjusted-close patch rows.",
            "- ready_for_p1_c2_market_health_consensus4_net_cost_diagnostic=false.",
            "- unadjusted OHLC comparator remains ready as proxy-only, not adjusted-close evidence.",
            "",
            "## Core judgment",
            "",
            "這不是 Core ingest blocker，而是 source / policy blocker：目前沒有可接受的 selected-stock adjusted close source route。不得 silent fill，也不得把 unadjusted OHLC comparator 包裝成 adjusted-close path。",
            "",
            "## Strategy Center decision needed",
            "",
            "1. 核准 licensed third-party adjusted close source route；或",
            "2. 啟動 official corporate-action adjustment contract，從官方除權息/減資事件建立調整價；或",
            "3. 接受 adjusted net-cost diagnostic 繼續 blocked，只保留 unadjusted comparator 作 proxy diagnostic。",
            "",
            "## Flags",
            "",
            "- formal_model_changed=false",
            "- trade_decision_changed=false",
            "- active_in_trade_decision=false",
            "- report_changed=false",
            "- portfolio_replay_executed=false",
            "- ready_for_strategy_replay=false",
            "- ready_for_formal=false",
            "- not_live_rule=true",
            "- forward_returns_live_rule_usage=false",
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
        "inputs": {
            "radar_output": str(RADAR_DIR),
            "core_prior_c2_contract": str(CORE_C2_DIR),
        },
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
