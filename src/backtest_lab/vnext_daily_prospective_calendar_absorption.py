from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


TASK_ID = "TASK-BACKTEST-CORE-VNEXT-DAILY-PROSPECTIVE-CORPORATE-ACTION-MARKET-CALENDAR-ABSORPTION-001"
REPO_ROOT = Path(__file__).resolve().parents[2]
CONTRACT_DIR = REPO_ROOT / "outputs" / "vnext_daily_prospective_corporate_action_event_guard_contract_20260710"
RADAR_DIR = Path(
    r"C:\Users\zergv\Documents\Codex\2026-05-23\ai-stock-rotation-radar-https-docs\outputs"
    r"\radar_vnext_daily_prospective_corporate_action_market_calendar_source_package_20260710"
)
OUTPUT_DIR = REPO_ROOT / "outputs" / "vnext_daily_prospective_corporate_action_market_calendar_absorption_20260710"

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
    radar_readiness = json.loads(
        (RADAR_DIR / "readiness_for_core_daily_prospective_market_calendar_absorption.json").read_text(encoding="utf-8")
    )
    guard_readiness = json.loads(
        (CONTRACT_DIR / "readiness_for_daily_prospective_corporate_action_event_guard.json").read_text(encoding="utf-8")
    )
    calendar = pd.read_csv(RADAR_DIR / "daily_prospective_market_calendar_canonical_rows.csv", dtype={"ticker": str}, low_memory=False)
    blocked_source = pd.read_csv(RADAR_DIR / "daily_prospective_market_calendar_blocked_ledger.csv", low_memory=False)
    source_manifest = pd.read_csv(RADAR_DIR / "daily_prospective_market_calendar_source_manifest.csv", low_memory=False)
    calendar["effective_date"] = pd.to_datetime(calendar["effective_date"], errors="coerce")
    calendar["calendar_discovery_ready"] = True
    calendar["market_available_at_ready"] = pd.to_datetime(calendar["market_available_at"], errors="coerce").notna()
    calendar["event_terms_ready"] = False
    calendar["event_detail_required_after_score_universe_hit"] = True
    calendar["score_universe_intersection_status"] = "not_run_waiting_current_price_scoring_universe"
    calendar["affected_ticker_detail_query_authorized"] = False
    calendar["analysis_price_factor_ready"] = False
    calendar["price_scoring_ready"] = False
    calendar["selected_ticker_exact_audit_ready"] = False
    calendar["calendar_role"] = "current_prospective_candidate_discovery_only"
    calendar["historical_252session_use_allowed"] = False
    calendar["future_data_violation_count"] = 0
    retention = pd.DataFrame([
        {
            "cache_stage": "daily_market_calendar_snapshot",
            "write_policy": "append one canonical snapshot per market trading day",
            "dedupe_key": "market|ticker|event_type|effective_date|source_hash",
            "retention_policy": "retain snapshots covering at least 252 trading sessions; operational buffer may use 400 calendar days",
            "historical_backfill_allowed": False,
            "historical_adjusted_close_escalation_reopened": False,
            "current_status": "day_1_TWSE_seed_only",
        },
        {
            "cache_stage": "affected_ticker_detail",
            "write_policy": "query/cache only after same-day scoring-universe intersection",
            "dedupe_key": "detail_cache_key|source_hash",
            "retention_policy": "retain while any 252-session analysis window references the event",
            "historical_backfill_allowed": False,
            "historical_adjusted_close_escalation_reopened": False,
            "current_status": "zero_queries_until_intersection",
        },
    ]).assign(future_data_violation_count=0)
    intersection = pd.DataFrame([
        {"step": 1, "operation": "load_current_price_scoring_universe", "input": "same-day Layer2/Layer4 price-scoring ticker set", "query_count": 0, "output": "scoring_tickers"},
        {"step": 2, "operation": "join_calendar", "input": "scoring_tickers INNER JOIN calendar by market+ticker", "query_count": 0, "output": "affected_scoring_tickers"},
        {"step": 3, "operation": "query_detail_cache_miss_only", "input": "unique detail_cache_key for affected_scoring_tickers", "query_count": "A cache misses", "output": "event effective terms and PIT timestamp"},
        {"step": 4, "operation": "construct_event_adjusted_analysis_price", "input": "raw price + resolved event terms", "query_count": 0, "output": "event_adjusted_analysis_price or blocked"},
        {"step": 5, "operation": "selected_top1_exact_audit", "input": "selected ticker and reused detail cache", "query_count": "0 if fresh; max 1 exact refresh", "output": "pass|next-ranked hook|fallback hook|blocked"},
    ]).assign(
        raw_execution_price_preserved=True,
        top250_per_ticker_history_query=False,
        live_trade_decision_authorized=False,
        future_data_violation_count=0,
    )
    query_storage = pd.DataFrame([
        {
            "audit_scope": "observed_Radar_package",
            "market_level_queries": radar_readiness["coverage"]["market_level_queries"],
            "TWSE_calendar_rows": len(calendar),
            "TPEx_calendar_rows": 0,
            "affected_ticker_detail_queries": radar_readiness["coverage"]["affected_ticker_detail_queries"],
            "observed_raw_bytes": radar_readiness["coverage"]["observed_raw_bytes"],
            "route_error_count": radar_readiness["coverage"]["route_error_count"],
            "query_semantics": "one batch per market source; zero per-ticker query before score-universe intersection",
        }
    ]).assign(future_data_violation_count=0)
    readiness = {
        "task_id": TASK_ID,
        "status": "TWSE_current_calendar_absorbed_TPEx_and_retained_history_blocked",
        "TWSE_current_calendar_rows_absorbed": len(calendar),
        "TWSE_current_calendar_candidate_discovery_ready": True,
        "TWSE_market_available_at_ready_rows": int(calendar["market_available_at_ready"].sum()),
        "TWSE_analysis_price_factor_ready_rows": int(calendar["analysis_price_factor_ready"].sum()),
        "TPEx_current_calendar_rows_absorbed": 0,
        "TPEx_current_calendar_ready": False,
        "retained_252session_calendar_history_ready": False,
        "forward_daily_cache_retention_contract_ready": True,
        "current_price_scoring_universe_intersection_ready": False,
        "affected_ticker_detail_queries_executed": 0,
        "selected_ticker_exact_audit_ready": False,
        "ready_for_TWSE_partial_guard_candidate_discovery": True,
        "ready_for_all_market_daily_prospective_guard": False,
        "historical_adjusted_close_escalation_reopened": False,
        "historical_backtest_path_policy": "official_unadjusted_OHLC_diagnostic_only",
        "ready_for_experiments": False,
        "ready_for_formal": False,
        "ready_for_strategy_replay": False,
        "ready_for_daily_trade_decision": False,
        "future_data_violation_count": 0,
        "next_owner": "Radar/Data bounded TPEx current browser-equivalent calendar capture; Core later joins same-day scoring universe",
        **FLAGS,
    }
    blocked = pd.DataFrame([
        {"item": "TPEx_current_market_calendar", "status": "blocked", "detail": "Public page identified but direct replay returned 404; no TPEx rows materialized.", "next_owner": "Radar/Data browser-equivalent bounded current capture"},
        {"item": "retained_252session_calendar_history", "status": "blocked_bootstrap", "detail": "Current snapshot cannot backfill history; build prospectively by daily cache retention.", "next_owner": "Core daily pipeline over time"},
        {"item": "market_available_at", "status": "detail_required", "detail": "Retrieval time is metadata only; issuer announcement timestamp is required after score-universe hit.", "next_owner": "Radar/Data bounded detail query after Core intersection"},
        {"item": "price_scoring_universe_intersection", "status": "not_run", "detail": "No same-day scoring ticker set was supplied; detail query count correctly remains zero.", "next_owner": "Core daily Layer2/Layer4 materialization"},
        {"item": "historical_adjusted_close", "status": "closed_not_in_scope", "detail": "No P1 historical source escalation is reopened.", "next_owner": "none"},
    ])
    future_audit = pd.DataFrame([
        {"audit_item": "current_calendar", "future_data_used": False, "detail": "Current prospective events are candidate discovery; PIT availability remains unresolved until detail.", "future_data_violation_count": 0},
        {"audit_item": "retrieval_timestamp", "future_data_used": False, "detail": "Retrieval time is not used as market availability time.", "future_data_violation_count": 0},
        {"audit_item": "retained_history", "future_data_used": False, "detail": "Current snapshot is not represented as 252-session history.", "future_data_violation_count": 0},
        {"audit_item": "detail_queries", "future_data_used": False, "detail": "No ticker detail queried before score-universe intersection.", "future_data_violation_count": 0},
    ])
    paths = [
        _write(calendar, "daily_prospective_TWSE_calendar_absorbed.csv"),
        _write(intersection, "daily_prospective_score_universe_intersection_detail_query_contract.csv"),
        _write(retention, "daily_prospective_calendar_forward_cache_retention_contract.csv"),
        _write(query_storage, "daily_prospective_calendar_observed_query_storage_audit.csv"),
        _write(source_manifest, "daily_prospective_calendar_source_manifest_absorbed.csv"),
        _write(blocked_source, "daily_prospective_calendar_radar_blocked_ledger_absorbed.csv"),
        _write(blocked, "daily_prospective_calendar_core_blocked_proxy_audit.csv"),
        _write(future_audit, "daily_prospective_calendar_absorption_future_data_audit.csv"),
    ]
    readiness_path = OUTPUT_DIR / "readiness_for_daily_prospective_calendar_absorption.json"
    readiness_path.write_text(json.dumps(readiness, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path = OUTPUT_DIR / "final_summary_zh.md"
    summary_path.write_text(
        "# Daily Prospective Corporate-action Calendar Absorption\n\n"
        f"- TWSE current calendar absorbed: {len(calendar)} rows; one TWSE batch, no per-TOP250 ticker history queries.\n"
        "- Calendar rows are candidate discovery only: market_available_at and complete event terms require detail after scoring-universe intersection.\n"
        "- affected ticker detail queries remain 0 because no current scoring-universe intersection was supplied.\n"
        "- TPEx current rows remain blocked; retained 252-session history must accumulate prospectively from daily snapshots.\n"
        "- Current snapshot is not used as historical coverage and does not reopen adjusted-close escalation.\n"
        "- raw execution and event-adjusted analysis price semantics remain separate.\n\n"
        "結論：TWSE partial candidate discovery ready；all-market daily guard、PIT detail、252-session retained history與selected exact audit尚未ready。\n",
        encoding="utf-8",
    )
    manifest = {
        "task_id": TASK_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(OUTPUT_DIR),
        "source_inputs": {"Core_guard_contract": str(CONTRACT_DIR), "Radar_calendar": str(RADAR_DIR)},
        "files": [{"path": p.name, "sha256": _sha256(p)} for p in [*paths, readiness_path, summary_path]],
        "readiness": readiness,
        "Radar_readiness": radar_readiness,
        "guard_readiness": guard_readiness,
    }
    (OUTPUT_DIR / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(readiness, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
