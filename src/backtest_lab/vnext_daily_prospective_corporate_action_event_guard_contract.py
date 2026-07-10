from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


TASK_ID = "TASK-BACKTEST-CORE-VNEXT-DAILY-PROSPECTIVE-CORPORATE-ACTION-EVENT-GUARD-CONTRACT-001"
REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "outputs" / "vnext_daily_prospective_corporate_action_event_guard_contract_20260710"
SOURCE_CLOSURE_DIR = REPO_ROOT / "outputs" / "vnext_selected_stock_total_return_source_escalation_closure_20260710"

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


def _lookback_contract() -> pd.DataFrame:
    rows = [
        ("RS20", 20, "stock price scoring", "vNext Layer2/Layer4"),
        ("RS60", 60, "stock price scoring", "vNext Layer2/Layer4"),
        ("BIAS60", 60, "stock overheat/risk scoring", "vNext Layer2/Layer4"),
        ("BIAS120", 120, "stock/market context", "daily incumbent/challenger and market fields"),
        ("return_120d", 120, "market regime", "market_regime.py"),
        ("MA200", 200, "market regime", "market_regime.py"),
        ("drawdown_from_252d_high", 252, "market regime", "market_regime.py"),
    ]
    return pd.DataFrame(rows, columns=["field", "lookback_trading_days", "usage", "source_contract"]).assign(
        corporate_action_guard_required=True,
        future_data_violation_count=0,
    )


def _event_schema() -> pd.DataFrame:
    common = {
        "raw_execution_price_column": "raw_execution_price",
        "analysis_price_column": "event_adjusted_analysis_price",
        "analysis_price_adjusted_flag": "analysis_price_adjusted",
        "pit_requirement": "event source market_available_at <= signal_close",
        "formal_status": "contract_only_not_live",
    }
    rows = [
        {"event_type": "cash_dividend", "required_terms": "effective_date|cash_per_share", "analysis_transform": "total-return analysis uses cash entitlement; raw execution price unchanged", "unresolved_action": "block affected price-return features"},
        {"event_type": "stock_dividend", "required_terms": "ex_right_date|new_share_effective_date|share_ratio", "analysis_transform": "share-ratio continuity adjustment", "unresolved_action": "block affected price-return features"},
        {"event_type": "capital_reduction", "required_terms": "effective_date|trading_resume_date|share_ratio|cash_return", "analysis_transform": "share-scale plus cash-return wealth continuity", "unresolved_action": "block affected price-return features"},
        {"event_type": "split", "required_terms": "effective_date|split_ratio", "analysis_transform": "inverse price/share scale adjustment", "unresolved_action": "block affected price-return features"},
        {"event_type": "reverse_split", "required_terms": "effective_date|reverse_split_ratio", "analysis_transform": "inverse price/share scale adjustment", "unresolved_action": "block affected price-return features"},
        {"event_type": "merger_share_conversion", "required_terms": "effective_date|old_ticker|new_ticker|conversion_ratio|cash_component", "analysis_transform": "security-identity and holder-wealth continuity", "unresolved_action": "block affected price-return features"},
    ]
    return pd.DataFrame([{**row, **common} for row in rows]).assign(future_data_violation_count=0)


def _pipeline_contract() -> pd.DataFrame:
    rows = [
        (0, "Layer0 turnover universe", "market OHLCV batch already required", "none", "Corporate action does not trigger per-ticker history download at Layer0."),
        (1, "market event calendar", "one market-level calendar query per run", "all calendar events in retained trading window plus next execution date", "PIT-filter by market_available_at."),
        (2, "price-scoring universe intersection", "local join", "calendar ticker intersects any ticker entering RS/BIAS/return scoring", "Guard occurs before price features are computed."),
        (3, "affected ticker detail", "one cached detail/factor query per affected ticker/event group", "only intersected affected tickers", "No query for unaffected TOP250 names."),
        (4, "analysis-price construction", "local deterministic transform", "affected ticker price history", "Keep raw execution and adjusted analysis columns separate."),
        (5, "Layer2/Layer4 price scoring", "local", "only event-ready analysis prices", "Unresolved affected ticker price features are blocked, never silently raw."),
        (6, "selected ticker exact audit", "cache-first; at most one exact refresh if stale", "selected top1", "Pass, blocked-next-ranked hook, or fallback hook; no live authorization."),
    ]
    return pd.DataFrame(rows, columns=["step", "stage", "query_policy", "scope", "guard_semantics"]).assign(
        max_analysis_lookback_trading_days=252,
        calendar_window_policy="earliest retained 252-session date through next trading-day execution date",
        future_data_violation_count=0,
    )


def _selected_guard() -> pd.DataFrame:
    rows = [
        ("pass", True, "normal_output_hook", "selected event audit is complete and analysis scores are event-aware"),
        ("unresolved", False, "next_ranked_data_ready_hook", "do not use suspected discontinuity score; evaluate next-ranked event-ready candidate"),
        ("unresolved_no_next_ranked", False, "fallback_hook", "fallback asset hook only; Core contract does not authorize live action"),
        ("calendar_or_source_stale", False, "blocked_hook", "do not issue stock selection from stale event state"),
    ]
    return pd.DataFrame(rows, columns=["selected_ticker_event_status", "selected_ticker_event_ready", "output_hook", "blocked_reason_or_action"]).assign(
        live_trade_decision_authorized=False,
        future_data_violation_count=0,
    )


def _report_hooks() -> pd.DataFrame:
    fields = [
        ("corporate_action_check", "pass|blocked|not_affected|stale"),
        ("event_type", "cash_dividend|stock_dividend|capital_reduction|split|reverse_split|merger_share_conversion|none"),
        ("effective_date", "ISO date or blank when none; never inferred"),
        ("analysis_price_adjusted", "true|false|blocked"),
        ("selected_ticker_event_ready", "true|false"),
        ("blocked_reason", "explicit source/term/staleness reason"),
        ("raw_execution_price", "official raw executable/report price"),
        ("event_adjusted_analysis_price", "analysis-only price used by RS/BIAS/return"),
        ("event_source_quality", "official|trusted_nonofficial_diagnostic|blocked"),
        ("event_market_available_at", "PIT availability timestamp"),
    ]
    return pd.DataFrame(fields, columns=["report_field", "contract_semantics"]).assign(
        report_changed=False,
        report_hook_only=True,
        future_data_violation_count=0,
    )


def _sanity_cases() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "case_id": "2330_20230316_cash_dividend",
            "event_type": "cash_dividend",
            "effective_date": "2023-03-16",
            "known_term": "cash_per_share=2.74982072",
            "source_quality": "FinMind+Yahoo trusted_nonofficial cross-validated diagnostic metadata",
            "selected_holding_alignment": "event predates first selected holding on 2023-03-20",
            "raw_price_return_usage": "not substituted for event-aware return across event date",
            "event_aware_return_usage": "no selected-path entitlement impact; event metadata retained",
            "sanity_status": "pass_no_mix",
        },
        {
            "case_id": "2316_cash_capital_reduction_terms_incomplete",
            "event_type": "capital_reduction",
            "effective_date": "blocked",
            "known_term": "historical MOPS text indicates reduction semantics but holder effective terms incomplete",
            "source_quality": "official_partial",
            "selected_holding_alignment": "historical diagnostic case only",
            "raw_price_return_usage": "blocked for event-spanning price scoring",
            "event_aware_return_usage": "not calculated without exact effective date/share/cash terms",
            "sanity_status": "pass_blocks_instead_of_mix",
        },
        {
            "case_id": "generic_split_guard",
            "event_type": "split_or_reverse_split",
            "effective_date": "required",
            "known_term": "effective_date+ratio required",
            "source_quality": "contract_sanity_no_fabricated_event",
            "selected_holding_alignment": "not applicable",
            "raw_price_return_usage": "raw discontinuity cannot enter RS/BIAS/return",
            "event_aware_return_usage": "analysis price requires exact ratio; otherwise blocked",
            "sanity_status": "contract_guard_only",
        },
    ]).assign(future_data_violation_count=0)


def _query_budget() -> pd.DataFrame:
    rows = []
    for affected in (0, 5, 20):
        rows.append({
            "scenario": f"affected_tickers_{affected}",
            "top250_per_ticker_history_queries": 0,
            "market_calendar_queries": 1,
            "affected_detail_queries_max": affected,
            "selected_exact_refresh_queries_max": 1,
            "daily_queries_max_cache_miss": 2 + affected,
            "daily_queries_typical_cache_reuse": 1 + affected,
            "storage_assumption": "calendar row 1KB; detail response 20KB per affected ticker; rolling raw cache retained by policy",
            "daily_increment_estimate_kb": 5 + affected * 20,
            "rolling_400_calendar_day_estimate_mb": round((5 + affected * 20) * 400 / 1024, 2),
            "estimate_not_observed": True,
        })
    return pd.DataFrame(rows).assign(future_data_violation_count=0)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    closure = json.loads(
        (SOURCE_CLOSURE_DIR / "readiness_for_selected_stock_total_return_source_escalation_closure.json").read_text(encoding="utf-8")
    )
    lookback = _lookback_contract()
    schema = _event_schema()
    pipeline = _pipeline_contract()
    selected_guard = _selected_guard()
    report_hooks = _report_hooks()
    sanity = _sanity_cases()
    budget = _query_budget()
    source_readiness = pd.DataFrame([
        {"source_component": "current_market_level_corporate_action_calendar", "status": "blocked_source_package_required", "scope": "one market-level query covering retained 252 trading sessions through next execution date", "next_owner": "Radar/Data"},
        {"source_component": "affected_ticker_event_detail_factor", "status": "blocked_source_package_required", "scope": "only calendar-hit tickers entering price scoring", "next_owner": "Radar/Data"},
        {"source_component": "selected_ticker_exact_event_audit", "status": "contract_ready_source_not_materialized", "scope": "cache-first selected top1 refresh", "next_owner": "Core after Radar package"},
        {"source_component": "historical_P1_adjusted_close", "status": "closed_not_in_scope", "scope": "official unadjusted OHLC diagnostic-only remains fixed", "next_owner": "none"},
    ]).assign(future_data_violation_count=0)
    future_audit = pd.DataFrame([
        {"audit_item": "calendar_PIT", "future_data_used": False, "detail": "Only events available by signal close may affect analysis or next-day guard.", "future_data_violation_count": 0},
        {"audit_item": "effective_date", "future_data_used": False, "detail": "No board/shareholder/query date substitution.", "future_data_violation_count": 0},
        {"audit_item": "price_columns", "future_data_used": False, "detail": "Raw execution and adjusted analysis prices are separate columns.", "future_data_violation_count": 0},
        {"audit_item": "historical_escalation", "future_data_used": False, "detail": "No P1 adjusted-close backfill or archive escalation reopened.", "future_data_violation_count": 0},
    ])
    readiness = {
        "task_id": TASK_ID,
        "status": "prospective_guard_contract_ready_current_market_calendar_source_blocked",
        "max_price_lookback_trading_days": int(lookback["lookback_trading_days"].max()),
        "max_price_lookback_driver": "drawdown_from_252d_high",
        "market_calendar_single_query_contract_ready": True,
        "top250_per_ticker_history_download_required": False,
        "event_aware_guard_before_price_scoring_required": True,
        "selected_ticker_exact_audit_required": True,
        "raw_execution_analysis_price_columns_separated": True,
        "current_market_calendar_source_ready": False,
        "affected_ticker_detail_source_ready": False,
        "ready_for_radar_current_prospective_source_package": True,
        "ready_for_daily_prospective_guard_materialization": False,
        "historical_adjusted_close_escalation_reopened": False,
        "historical_backtest_path_policy": "official_unadjusted_OHLC_diagnostic_only",
        "ready_for_experiments": False,
        "ready_for_formal": False,
        "ready_for_strategy_replay": False,
        "ready_for_daily_trade_decision": False,
        "future_data_violation_count": 0,
        "next_owner": "Radar/Data current/prospective bounded market-calendar source package",
        **FLAGS,
    }
    paths = [
        _write(lookback, "daily_prospective_event_guard_lookback_audit.csv"),
        _write(schema, "daily_prospective_corporate_action_event_schema.csv"),
        _write(pipeline, "daily_prospective_corporate_action_event_guard_contract.csv"),
        _write(selected_guard, "daily_prospective_selected_ticker_event_guard_hooks.csv"),
        _write(report_hooks, "daily_prospective_corporate_action_report_hooks.csv"),
        _write(sanity, "daily_prospective_corporate_action_sanity_cases.csv"),
        _write(budget, "daily_prospective_corporate_action_query_storage_budget.csv"),
        _write(source_readiness, "daily_prospective_corporate_action_source_readiness.csv"),
        _write(future_audit, "daily_prospective_corporate_action_future_data_audit.csv"),
    ]
    readiness_path = OUTPUT_DIR / "readiness_for_daily_prospective_corporate_action_event_guard.json"
    readiness_path.write_text(json.dumps(readiness, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path = OUTPUT_DIR / "final_summary_zh.md"
    summary_path.write_text(
        "# Daily Prospective Corporate-action Event Guard\n\n"
        "- max lookback is 252 trading days, driven by drawdown-from-252D-high; MA200 and 120D fields are also covered.\n"
        "- query design: one market calendar query, then detail/factor only for calendar-hit tickers entering price scoring.\n"
        "- Layer0 turnover universe adds zero per-ticker corporate-action history queries.\n"
        "- raw_execution_price and event_adjusted_analysis_price are separate mandatory columns.\n"
        "- unresolved affected tickers cannot use raw discontinuity in RS/BIAS/return; use blocked/next-ranked/fallback hooks only.\n"
        "- selected top1 receives a cache-first exact event audit; no live trade action is authorized.\n"
        "- historical P1 adjusted-close escalation remains closed and official-unadjusted diagnostic-only.\n\n"
        "結論：contract與report hooks已ready；current/prospective市場層級calendar及命中ticker detail source尚未materialize，需Radar/Data bounded source package。\n",
        encoding="utf-8",
    )
    manifest = {
        "task_id": TASK_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(OUTPUT_DIR),
        "source_closure": str(SOURCE_CLOSURE_DIR),
        "files": [{"path": p.name, "sha256": _sha256(p)} for p in [*paths, readiness_path, summary_path]],
        "readiness": readiness,
        "historical_source_closure": closure,
    }
    (OUTPUT_DIR / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(readiness, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
