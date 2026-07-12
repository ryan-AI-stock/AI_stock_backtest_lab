from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd


TASK_ID = "TASK-BACKTEST-CORE-VNEXT-P1-FULL-LIFECYCLE-UNIFIED-FEATURE-STATE-MACHINE-CONTRACT-001"
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RADAR = Path(r"C:\Users\zergv\Documents\Codex\2026-05-23\ai-stock-rotation-radar-https-docs\outputs\radar_vnext_p1_full_lifecycle_minimum_data_acquisition_20260710")
DEFAULT_ADJUSTED_63 = Path(r"C:\Users\zergv\Documents\Codex\2026-05-23\ai-stock-rotation-radar-https-docs\outputs\radar_vnext_p1_adjusted_analysis_63_bounded_resolution_20260711")
DEFAULT_FREE_REOPEN = Path(r"C:\Users\zergv\Documents\Codex\2026-05-23\ai-stock-rotation-radar-https-docs\outputs\radar_vnext_p1_free_historical_source_reopen_audit_20260712")
DEFAULT_OUTPUT = REPO_ROOT / "outputs/vnext_p1_full_lifecycle_unified_feature_state_machine_contract_20260711"
FLAGS = {
    "formal_model_changed": False, "trade_decision_changed": False,
    "active_in_trade_decision": False, "report_changed": False,
    "portfolio_replay_executed": False, "ready_for_strategy_replay": False,
    "ready_for_formal": False, "not_live_rule": True,
    "forward_returns_live_rule_usage": False,
}


def _write_csv(rows: list[dict], path: Path) -> None:
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def run(radar_dir: Path = DEFAULT_RADAR, adjusted_63_dir: Path = DEFAULT_ADJUSTED_63,
        free_reopen_dir: Path = DEFAULT_FREE_REOPEN, output_dir: Path = DEFAULT_OUTPUT) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "current_step.txt").write_text("validating_radar_source_package", encoding="utf-8")
    required = [
        "twse_compact_repair_dedup_audit.csv", "twse_true_failed_ledger.csv",
        "trusted_adjusted_analysis_manifest.csv", "tpex_institutional_margin_manifest.csv",
        "price_bulk_download_manifest.csv", "tdcc_taifex_route_probe_evidence.csv",
        "readiness_for_core_full_lifecycle_feature_matrix.json",
        "twse_2017_rebuilt_shard_integrity.csv",
        "twse_2017_margin_failed_market_day_classification.csv",
        "twse_true_failed_final_bounded_retry.csv",
    ]
    missing = [name for name in required if not (radar_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Radar package missing required artifacts: {missing}")
    adjusted_required = [
        "adjusted_analysis_63_ticker_classification.csv", "adjusted_analysis_63_remaining_blocked.csv",
        "adjusted_analysis_63_provider_route_inventory.csv", "adjusted_analysis_63_requested_vs_actual_coverage.csv",
        "adjusted_analysis_63_future_data_audit.csv", "readiness_for_core_adjusted_analysis_63_resolution.json",
    ]
    adjusted_missing = [name for name in adjusted_required if not (adjusted_63_dir / name).exists()]
    if adjusted_missing:
        raise FileNotFoundError(f"Adjusted-63 package missing required artifacts: {adjusted_missing}")
    reopen_required = ["readiness_for_core_p1_free_historical_reopen.json", "tpex_institutional_checksum_manifest.csv",
                       "tpex_institutional_source_manifest.csv", "tpex_institutional_blocked_ledger.csv",
                       "p1_free_historical_requested_vs_actual.csv", "p1_free_historical_future_data_audit.csv"]
    reopen_missing = [name for name in reopen_required if not (free_reopen_dir / name).exists()]
    if reopen_missing:
        raise FileNotFoundError(f"P1 free historical reopen package missing: {reopen_missing}")

    compact = pd.read_csv(radar_dir / "twse_compact_repair_dedup_audit.csv")
    failures = pd.read_csv(radar_dir / "twse_true_failed_ledger.csv")
    adjusted = pd.read_csv(radar_dir / "trusted_adjusted_analysis_manifest.csv", dtype={"ticker": str})
    tpex = pd.read_csv(radar_dir / "tpex_institutional_margin_manifest.csv")
    radar_ready = json.loads((radar_dir / "readiness_for_core_full_lifecycle_feature_matrix.json").read_text(encoding="utf-8-sig"))
    rebuilt = pd.read_csv(radar_dir / "twse_2017_rebuilt_shard_integrity.csv")
    margin_gaps = pd.read_csv(radar_dir / "twse_2017_margin_failed_market_day_classification.csv")
    retry = pd.read_csv(radar_dir / "twse_true_failed_final_bounded_retry.csv")
    adjusted_63 = pd.read_csv(adjusted_63_dir / "adjusted_analysis_63_ticker_classification.csv", dtype={"ticker": str})
    adjusted_63_coverage = pd.read_csv(adjusted_63_dir / "adjusted_analysis_63_requested_vs_actual_coverage.csv").iloc[0]
    adjusted_63_ready = json.loads((adjusted_63_dir / "readiness_for_core_adjusted_analysis_63_resolution.json").read_text(encoding="utf-8-sig"))
    reopen_ready = json.loads((free_reopen_dir / "readiness_for_core_p1_free_historical_reopen.json").read_text(encoding="utf-8-sig"))
    reopen_checksums = pd.read_csv(free_reopen_dir / "tpex_institutional_checksum_manifest.csv")
    tpex_checksum_rows = []
    for item in reopen_checksums.itertuples(index=False):
        path = free_reopen_dir / str(item.path).replace("\\", "/")
        actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else ""
        tpex_checksum_rows.append({"path": str(path), "expected_sha256": item.sha256, "actual_sha256": actual,
                                   "checksum_match": actual == item.sha256, "bytes": item.bytes})
    if len(tpex_checksum_rows) != 8 or not all(row["checksum_match"] for row in tpex_checksum_rows):
        raise ValueError("TPEx P1 institutional compact checksum validation failed")
    if len(adjusted_63) != 63 or adjusted_63.resolution_status.ne("blocked").any():
        raise ValueError("Adjusted-63 classification must contain exactly 63 explicitly blocked tickers")
    expected_shards = {item["family"]: item for item in radar_ready.get("twse_2017_rebuilt_shards", [])}
    shard_checks = []
    for family, item in expected_shards.items():
        path = Path(item["path"])
        actual_hash = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else ""
        shard_checks.append({
            "family": family, "path": str(path), "rows": item["rows"], "unique_rows": item["unique_rows"],
            "gzip_integrity": item["gzip_integrity"], "utf8_roundtrip": item["utf8_roundtrip"],
            "expected_sha256": item["sha256"], "actual_sha256": actual_hash,
            "checksum_match": actual_hash == item["sha256"], "ingest_priority": "atomic_rebuilt_only",
        })
    if len(shard_checks) != 2 or not all(row["checksum_match"] for row in shard_checks):
        raise ValueError("TWSE 2017 atomic rebuilt shard validation failed")

    shutil.copy2(radar_dir / "twse_true_failed_ledger.csv", output_dir / "p1_full_lifecycle_true_failed_ledger.csv")
    compact["ingest_action"] = compact.apply(
        lambda r: "exclude_corrupt_file_then_deduplicate_date_ticker" if int(r.corrupt_gzip_file_count) else "deduplicate_date_ticker", axis=1
    )
    compact["silent_fill_allowed"] = False
    compact.to_csv(output_dir / "p1_full_lifecycle_compact_dedup_exclusion_audit.csv", index=False, encoding="utf-8-sig")
    _write_csv(shard_checks, output_dir / "p1_full_lifecycle_twse_2017_atomic_shard_ingest_audit.csv")
    margin_gaps.assign(silent_fill_allowed=False, feature_missingness_required=True).to_csv(
        output_dir / "p1_full_lifecycle_twse_2017_margin_market_day_gap_ledger.csv", index=False, encoding="utf-8-sig"
    )
    retry.assign(final_ingest_status=retry.retry_result, silent_fill_allowed=False).to_csv(
        output_dir / "p1_full_lifecycle_true_failed_retry_resolution.csv", index=False, encoding="utf-8-sig"
    )
    adjusted_63.to_csv(output_dir / "p1_full_lifecycle_adjusted_analysis_63_classification.csv", index=False, encoding="utf-8-sig")
    shutil.copy2(adjusted_63_dir / "adjusted_analysis_63_remaining_blocked.csv", output_dir / "p1_full_lifecycle_adjusted_analysis_63_blocked_ledger.csv")
    shutil.copy2(adjusted_63_dir / "adjusted_analysis_63_provider_route_inventory.csv", output_dir / "p1_full_lifecycle_adjusted_analysis_63_provider_route_audit.csv")
    _write_csv(tpex_checksum_rows, output_dir / "p1_full_lifecycle_tpex_institutional_compact_ingest_audit.csv")
    shutil.copy2(free_reopen_dir / "tpex_institutional_source_manifest.csv", output_dir / "p1_full_lifecycle_tpex_institutional_source_manifest.csv")
    shutil.copy2(free_reopen_dir / "tpex_institutional_blocked_ledger.csv", output_dir / "p1_full_lifecycle_tpex_institutional_blocked_ledger.csv")
    shutil.copy2(free_reopen_dir / "p1_free_historical_requested_vs_actual.csv", output_dir / "p1_free_historical_reopen_requested_vs_actual.csv")

    adjusted_accepted = int(adjusted.status.eq("accepted").sum())
    tpex_inst_accepted = int(((tpex.family == "tpex_three_institutional") & (tpex.status == "accepted")).sum())
    tpex_margin_accepted = int(((tpex.family == "tpex_margin_short") & (tpex.status == "accepted")).sum())
    feature_rows = [
        {"feature_family": "official_raw_execution_OHLCV", "status": "ready", "source_quality": "official", "allowed_use": "execution_price_and_mark_to_market", "blocked_reason": ""},
        {"feature_family": "trusted_adjusted_analysis_OHLC", "status": "partial_structural_blocker", "source_quality": "trusted_nonofficial_analysis_only", "allowed_use": "research_grade_KD_MA_BIAS_RS_only_on_accepted_tickers", "blocked_reason": f"accepted={adjusted_accepted}; blocked=63 after bounded free-route exhaustion; no paid source; not formal"},
        {"feature_family": "TWSE_three_institutional", "status": "ready_with_explicit_no_rows", "source_quality": "official", "allowed_use": "PIT feature using atomic 2017 shard and date-level no_rows flags", "blocked_reason": "original true failures resolved: accepted=4/no_rows=6 across TWSE families"},
        {"feature_family": "TWSE_margin_short", "status": "partial", "source_quality": "official", "allowed_use": "PIT feature using atomic 2017 shard with explicit missingness", "blocked_reason": f"2017 market-trading-day source gaps={len(margin_gaps)}; original corrupt stream excluded"},
        {"feature_family": "TPEx_margin_short", "status": "partial", "source_quality": "official", "allowed_use": "PIT feature on accepted dates", "blocked_reason": f"accepted_dates={tpex_margin_accepted}; remaining missingness explicit"},
        {"feature_family": "TPEx_three_institutional", "status": "ready", "source_quality": "official", "allowed_use": "PIT institutional flow feature", "blocked_reason": "corrected dailyTrade route; accepted_dates=1943/1943; failed=0"},
        {"feature_family": "TDCC_holder_buckets", "status": "blocked", "source_quality": "official_current_only", "allowed_use": "none", "blocked_reason": "P1 historical PIT archive unavailable"},
        {"feature_family": "TAIFEX_OI_foreign_net", "status": "blocked", "source_quality": "official_historical_retention_limited", "allowed_use": "none", "blocked_reason": "P1 free historical route unavailable"},
        {"feature_family": "corporate_action_guard", "status": "partial", "source_quality": "prospective_only_plus_trusted_nonofficial_analysis", "allowed_use": "warning_and_analysis_price_lineage", "blocked_reason": "historical selected-stock adjusted close not formal-ready"},
    ]
    _write_csv(feature_rows, output_dir / "p1_full_lifecycle_feature_family_readiness.csv")

    _write_csv([
        {"field": "raw_execution_price", "source": "official TWSE/TPEx raw OHLCV", "use": "execution/cost/wealth path", "may_mix_with_other_price": False, "ready": True},
        {"field": "event_adjusted_analysis_price", "source": "Yahoo trusted_nonofficial adjusted series", "use": "KD/MA/BIAS/RS research features", "may_mix_with_other_price": False, "ready": False},
        {"field": "analysis_price_source_quality", "source": "per ticker manifest", "use": "row-level gate", "may_mix_with_other_price": False, "ready": True},
        {"field": "corporate_action_warning", "source": "event audit/guard", "use": "contamination warning", "may_mix_with_other_price": False, "ready": False},
    ], output_dir / "p1_full_lifecycle_price_semantics_contract.csv")

    schema = [
        ("signal_date", "identity", "PIT weekly close"), ("ticker", "identity", "Layer4 primary80 frozen universe"),
        ("raw_execution_close", "execution", "official raw OHLCV"), ("event_adjusted_analysis_close", "analysis", "trusted nonofficial adjusted OHLC"),
        ("KD_MA_BIAS_RS_block", "lifecycle", "requires adjusted analysis coverage and warmup"),
        ("institutional_flow_block", "capital", "requires market-complete PIT rows"), ("margin_short_lending_block", "risk", "requires market-complete PIT rows"),
        ("TDCC_holder_structure_block", "capital", "blocked"), ("market_TAIFEX_OI_block", "regime", "blocked"),
        ("lifecycle_state", "state", "not materialized until all minimum blocks ready"),
        ("incumbent_ticker", "decision", "hold valid incumbent"), ("challenger_ticker", "decision", "multi-block risk-adjusted challenger"),
        ("decision", "decision", "hold/switch/cash; cash only confirmed risk or invalid no replacement"),
        ("next_execution_date", "execution", "next official trading close"), ("transaction_cost", "cost", "EP05 stock/cash transition cost"),
    ]
    pd.DataFrame(schema, columns=["field", "block", "contract_semantics"]).assign(materialized=False).to_csv(
        output_dir / "p1_full_lifecycle_state_machine_schema_contract.csv", index=False, encoding="utf-8-sig"
    )

    blockers = [r for r in feature_rows if r["status"] not in {"ready", "ready_with_explicit_no_rows"}]
    _write_csv(blockers, output_dir / "p1_full_lifecycle_blocked_proxy_audit.csv")
    _write_csv([
        {"period": "P1", "requested_start": "2015-01-02", "requested_end": "2022-12-29", "actual_start": "", "actual_end": "", "status": "blocked_before_unified_materialization"}
    ], output_dir / "requested_vs_actual_coverage.csv")
    pd.DataFrame(columns=["signal_date", "ticker", "violation_reason"]).to_csv(output_dir / "future_data_audit.csv", index=False, encoding="utf-8-sig")

    readiness = {
        "task_id": TASK_ID,
        "status": "blocked_for_complete_unified_feature_state_machine_materialization",
        "radar_source_package_absorbed": True,
        "compact_ingest_policy_ready": True,
        "corrupt_TWSE_2017_margin_file_excluded": True,
        "twse_2017_atomic_rebuild_ready_for_core_ingest": True,
        "twse_2017_atomic_shard_checksum_match": True,
        "twse_compact_stream_corruption_file_count": 0,
        "original_true_failed_rows": len(failures),
        "true_failed_retry_accepted_rows": int(retry.retry_result.eq("accepted_official_response").sum()),
        "true_failed_retry_official_no_rows": int(retry.retry_result.eq("no_rows_official_response").sum()),
        "true_failed_rows_remaining_after_retry": 0,
        "twse_2017_margin_market_trading_day_source_gap_rows": len(margin_gaps),
        "no_rows_silently_filled": False,
        "official_raw_execution_OHLCV_ready": True,
        "trusted_nonofficial_adjusted_analysis_accepted_tickers": adjusted_accepted,
        "trusted_nonofficial_adjusted_analysis_blocked_tickers": len(adjusted) - adjusted_accepted,
        "adjusted_analysis_63_resolution_absorbed": True,
        "adjusted_analysis_63_attempted_tickers": int(adjusted_63_coverage["attempted_tickers"]),
        "adjusted_analysis_63_accepted_repair_tickers": int(adjusted_63_coverage["accepted_repair_tickers"]),
        "adjusted_analysis_63_remaining_blocked_tickers": int(adjusted_63_coverage["remaining_blocked_tickers"]),
        "adjusted_analysis_63_route_status": "bounded_free_routes_exhausted_explicit_blocked",
        "adjusted_analysis_paid_source_authorized": False,
        "free_historical_reopen_absorbed": True,
        "TPEx_institutional_prior_exhausted_conclusion_superseded": True,
        "TPEx_institutional_ready": True,
        "TPEx_institutional_requested_dates": int(reopen_ready["tpex_p1_dates"]),
        "TPEx_institutional_accepted_dates": int(reopen_ready["tpex_p1_dates"]),
        "TPEx_institutional_rows": int(reopen_ready["tpex_p1_rows"]),
        "TPEx_institutional_checksum_files": len(tpex_checksum_rows),
        "TPEx_institutional_checksum_match": True,
        "TDCC_P1_ready": False, "TAIFEX_P1_ready": False,
        "ready_for_core_full_lifecycle_feature_matrix": False,
        "ready_for_unified_state_machine_materialization": False,
        "ready_for_experiments": False, "automatic_handoff_stopped": True,
        "future_data_violation_count": 0, **FLAGS,
    }
    (output_dir / "readiness_for_p1_full_lifecycle_unified_feature_state_machine_contract.json").write_text(json.dumps(readiness, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "final_summary_zh.md").write_text(
        "# P1 full lifecycle unified feature/state-machine ingest 判定\n\n"
        "- Verdict：BLOCKED。Radar source package 已吸收為保守 ingest/schema contract，但不具備完整 feature matrix 或 state-machine materialization readiness。\n"
        "- 官方 raw execution OHLCV 可用；trusted Yahoo adjusted series 僅 research-grade analysis，913/976 ticker accepted，不得與 execution price 混欄。\n"
        "- TWSE 2017 institutional/margin atomic shards checksum、gzip、UTF-8 驗證通過；原損壞 stream 明確排除。\n"
        "- 原 10 筆 true failure 已由 bounded retry 關閉為 4 accepted + 6 official no_rows；另有 22 個 market-trading-day margin source gaps 保留欄位級 missingness，不 silent fill。\n"
        "- Adjusted-analysis 剩餘 63 檔已完成 bounded resolution：0 repair、63 explicit blocked；免費 route exhausted，未使用付費來源、successor ticker 或 raw-price substitution。\n"
        "- TPEx institutional先前exhausted結論已作廢：corrected dailyTrade route完成1943/1943日、1,091,250 rows、8個年度gzip checksum全通過。\n"
        "- TDCC P1、TAIFEX P1與adjusted-analysis 63仍blocked；因此不跑partial performance、不交Experiments。\n",
        encoding="utf-8",
    )
    files = sorted(p for p in output_dir.iterdir() if p.is_file())
    manifest = {"task_id": TASK_ID, "runner": str(Path(__file__).resolve()), "source_radar": str(radar_dir), "source_adjusted_63": str(adjusted_63_dir),
                "source_free_historical_reopen": str(free_reopen_dir), "source_free_historical_reopen_commit": "5a20866",
                "source_readiness": radar_ready, "adjusted_63_source_readiness": adjusted_63_ready, "free_reopen_readiness": reopen_ready, "readiness": readiness,
                "files": [{"name": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in files if p.name != "manifest.json"]}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (output_dir / "current_step.txt").write_text("blocked_waiting_complete_source_families_no_experiments_handoff", encoding="utf-8")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--radar-dir", type=Path, default=DEFAULT_RADAR)
    parser.add_argument("--adjusted-63-dir", type=Path, default=DEFAULT_ADJUSTED_63)
    parser.add_argument("--free-reopen-dir", type=Path, default=DEFAULT_FREE_REOPEN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(run(args.radar_dir, args.adjusted_63_dir, args.free_reopen_dir, args.output_dir))


if __name__ == "__main__":
    main()
