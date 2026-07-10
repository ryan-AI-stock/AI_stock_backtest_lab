from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


TASK_ID = "TASK-BACKTEST-CORE-VNEXT-SELECTED-STOCK-TOTAL-RETURN-SOURCE-ESCALATION-CLOSURE-001"
REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = REPO_ROOT / "outputs" / "vnext_selected_path_trusted_corporate_action_absorption_20260710"
OUTPUT_DIR = REPO_ROOT / "outputs" / "vnext_selected_stock_total_return_source_escalation_closure_20260710"

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
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write(frame: pd.DataFrame, name: str) -> Path:
    path = OUTPUT_DIR / name
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    source_readiness = json.loads(
        (SOURCE_DIR / "readiness_for_selected_path_trusted_corporate_action_absorption.json").read_text(encoding="utf-8")
    )
    blocked = pd.read_csv(
        SOURCE_DIR / "selected_path_remaining_28_trusted_source_blocked_ledger.csv",
        dtype={"ticker": str},
        low_memory=False,
    )
    providers = pd.read_csv(SOURCE_DIR / "selected_path_licensed_provider_options_absorbed.csv", low_memory=False)
    blocked["source_escalation_status"] = "closed_by_strategy_center"
    blocked["blocker_disposition"] = "preserved_unresolved"
    blocked["allowed_path_usage"] = "official_unadjusted_ohlc_diagnostic_only"
    blocked["formal_use_allowed"] = False
    blocked["strategy_replay_allowed"] = False
    blocked["daily_trade_decision_use_allowed"] = False
    blocked["future_data_violation_count"] = 0
    providers["purchase_or_inquiry_authorized"] = False
    providers["source_escalation_status"] = "closed_by_strategy_center"
    providers["closure_reason"] = (
        "TWSE T48 starts 2019-12-23 and cannot close P1 2015-2019; "
        "TPEx period/terms are unknown and completeness is not assured"
    )
    policy = pd.DataFrame([
        {
            "policy_item": "selected_stock_price_path",
            "status": "allowed_diagnostic_only",
            "value": "official_unadjusted_OHLC",
            "detail": "Must be labeled diagnostic-only in every downstream artifact.",
        },
        {
            "policy_item": "selected_stock_adjusted_close",
            "status": "blocked_closed_no_more_source_search",
            "value": False,
            "detail": "No adjusted close may be fabricated or inferred from unaccepted evidence.",
        },
        {
            "policy_item": "selected_stock_total_return",
            "status": "blocked_closed_no_more_source_search",
            "value": False,
            "detail": "Dividend/share conversion completeness remains unresolved for 28 rows.",
        },
        {
            "policy_item": "licensed_source",
            "status": "not_authorized",
            "value": False,
            "detail": "No purchase or provider inquiry is authorized.",
        },
        {
            "policy_item": "formal_strategy_replay_daily_decision",
            "status": "prohibited",
            "value": False,
            "detail": "Diagnostic results cannot be promoted to formal, replay, or daily trade decision.",
        },
        {
            "policy_item": "next_research_focus",
            "status": "return_to_strategy_center",
            "value": "P1_strategy_robustness_validation",
            "detail": "Source escalation is closed; model work continues on P1 robustness.",
        },
    ])
    closure_audit = pd.DataFrame([
        {"audit_item": "unresolved_rows_preserved", "status": "pass", "rows": len(blocked), "detail": "No blocker was silently marked resolved."},
        {"audit_item": "licensed_purchase", "status": "not_authorized", "rows": len(providers), "detail": "All provider options retained for audit only."},
        {"audit_item": "trusted_nonofficial_formal_use", "status": "prohibited", "rows": 1, "detail": "2330 cross-validated metadata remains diagnostic-only."},
        {"audit_item": "future_data", "status": "pass", "rows": 0, "detail": "No future data or future return used."},
    ])
    readiness = {
        "task_id": TASK_ID,
        "status": "source_escalation_closed_unadjusted_diagnostic_only_boundary_locked",
        "source_escalation_closed": True,
        "licensed_source_purchase_authorized": False,
        "licensed_source_inquiry_authorized": False,
        "remaining_blocked_rows": len(blocked),
        "remaining_yahoo_only_inferred_rows": int(blocked["resolution_status"].eq("blocked_yahoo_only_inferred").sum()),
        "remaining_missing_holder_ratio_effective_date_rows": int(blocked["resolution_status"].eq("blocked_missing_holder_ratio_effective_date").sum()),
        "remaining_no_trusted_structured_candidate_rows": int(blocked["resolution_status"].eq("blocked_no_trusted_structured_candidate").sum()),
        "complete_ticker_months": source_readiness["complete_ticker_months"],
        "blocked_ticker_months": source_readiness["blocked_ticker_months"],
        "complete_holding_intervals": source_readiness["complete_holding_intervals"],
        "blocked_holding_intervals": source_readiness["blocked_holding_intervals"],
        "official_unadjusted_ohlc_diagnostic_only": True,
        "selected_stock_total_return_complete": False,
        "selected_stock_adjusted_close_ready": False,
        "ready_for_experiments_on_adjusted_total_return_basis": False,
        "ready_for_formal": False,
        "ready_for_strategy_replay": False,
        "ready_for_daily_trade_decision": False,
        "future_data_violation_count": 0,
        "next_owner": "Strategy Center P1 strategy robustness validation",
        **FLAGS,
    }
    future_audit = pd.DataFrame([
        {"audit_item": "closure_decision", "future_data_used": False, "detail": "Governance decision only; no return/path values changed.", "future_data_violation_count": 0},
        {"audit_item": "blocked_ledger", "future_data_used": False, "detail": "All 28 unresolved rows preserved.", "future_data_violation_count": 0},
        {"audit_item": "downstream_usage", "future_data_used": False, "detail": "Only official unadjusted OHLC diagnostic use is allowed.", "future_data_violation_count": 0},
    ])
    paths = [
        _write(blocked, "selected_stock_total_return_closed_blocked_ledger.csv"),
        _write(providers, "selected_stock_licensed_source_options_closed_audit.csv"),
        _write(policy, "selected_stock_total_return_source_escalation_closure_policy.csv"),
        _write(closure_audit, "selected_stock_total_return_source_escalation_closure_audit.csv"),
        _write(future_audit, "selected_stock_total_return_source_escalation_future_data_audit.csv"),
    ]
    readiness_path = OUTPUT_DIR / "readiness_for_selected_stock_total_return_source_escalation_closure.json"
    readiness_path.write_text(json.dumps(readiness, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path = OUTPUT_DIR / "final_summary_zh.md"
    summary_path.write_text(
        "# Selected-stock Total-return Source Escalation Closure\n\n"
        "- Strategy Center has not authorized licensed-source purchase or inquiry.\n"
        "- Source escalation is closed; no further official/nonofficial/paid source search will be assigned.\n"
        f"- unresolved rows preserved: {len(blocked)}；complete ticker-months: {source_readiness['complete_ticker_months']}/42；complete holding intervals: {source_readiness['complete_holding_intervals']}/279。\n"
        "- selected-stock adjusted close / total-return remain blocked.\n"
        "- downstream path label is fixed: official unadjusted OHLC diagnostic-only.\n"
        "- formal / strategy replay / daily trade decision promotion is prohibited.\n"
        "- next research focus returns to P1 strategy robustness validation.\n\n"
        "結論：這是 source escalation closure，不是 blocker resolution。所有 ledger、source audit與runner保留；模型研究繼續，但不得再為 adjusted-close路線開新資料取得任務。\n",
        encoding="utf-8",
    )
    manifest = {
        "task_id": TASK_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(OUTPUT_DIR),
        "source_input": str(SOURCE_DIR),
        "files": [{"path": p.name, "sha256": _sha256(p)} for p in [*paths, readiness_path, summary_path]],
        "readiness": readiness,
        "source_readiness": source_readiness,
    }
    (OUTPUT_DIR / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(readiness, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
