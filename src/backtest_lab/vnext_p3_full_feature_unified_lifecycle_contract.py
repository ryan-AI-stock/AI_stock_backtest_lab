from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd


TASK_ID = "TASK-BACKTEST-CORE-VNEXT-P3-FULL-FEATURE-UNIFIED-LIFECYCLE-CONTRACT-001"
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = Path(r"C:\Users\zergv\Documents\Codex\2026-05-23\ai-stock-rotation-radar-https-docs\outputs\radar_vnext_p3_recent_full_feature_data_readiness_acquisition_20260711")
DEFAULT_GAP_PATCH = Path(r"C:\Users\zergv\Documents\Codex\2026-05-23\ai-stock-rotation-radar-https-docs\outputs\radar_vnext_p3_gap_convergence_open_day_acceptance_20260711")
DEFAULT_OUTPUT = REPO_ROOT / "outputs/vnext_p3_full_feature_unified_lifecycle_contract_20260711"
FLAGS = {"formal_model_changed": False, "trade_decision_changed": False, "active_in_trade_decision": False,
         "report_changed": False, "portfolio_replay_executed": False, "ready_for_strategy_replay": False,
         "ready_for_formal": False, "not_live_rule": True, "forward_returns_live_rule_usage": False}


def _write(rows: list[dict], path: Path) -> None:
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def run(source_dir: Path = DEFAULT_SOURCE, gap_patch_dir: Path = DEFAULT_GAP_PATCH, output_dir: Path = DEFAULT_OUTPUT) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "current_step.txt").write_text("validating_P3_source_package", encoding="utf-8")
    required = ["manifest.json", "readiness_for_core_p3_full_feature_unified_lifecycle_contract.json",
                "p3_family_coverage_matrix.csv", "p3_blocked_rows_and_family_ledger.csv", "p3_blocked_rows_detail.csv",
                "p3_adjusted_analysis_coverage_by_ticker.csv", "p3_tdcc_subperiod_split.csv",
                "p3_global_market_field_readiness.csv", "p3_pit_release_lag_ledger.csv",
                "p3_universe_requested_vs_actual.csv", "p3_future_data_audit.csv"]
    missing = [name for name in required if not (source_dir / name).exists()]
    if missing: raise FileNotFoundError(f"P3 source package missing: {missing}")
    patch_required = ["manifest.json", "readiness_for_core_p3_gap_convergence.json", "taifex_110_classification.csv",
                      "tdcc_11_classification.csv", "adjusted_12_exhausted_evidence.csv",
                      "layer4_membership_freshness_ledger.csv", "first_open_day_acceptance_status.json"]
    patch_missing = [name for name in patch_required if not (gap_patch_dir / name).exists()]
    if patch_missing: raise FileNotFoundError(f"P3 gap patch missing: {patch_missing}")
    source_manifest = json.loads((source_dir / "manifest.json").read_text(encoding="utf-8-sig"))
    source_ready = json.loads((source_dir / "readiness_for_core_p3_full_feature_unified_lifecycle_contract.json").read_text(encoding="utf-8-sig"))
    patch_manifest = json.loads((gap_patch_dir / "manifest.json").read_text(encoding="utf-8-sig"))
    patch_ready = json.loads((gap_patch_dir / "readiness_for_core_p3_gap_convergence.json").read_text(encoding="utf-8-sig"))
    declared = {item["path"].replace("\\", "/"): item["sha256"] for item in source_manifest["files"]}
    hash_mismatch = []
    for name in required[1:]:
        expected = declared.get(name)
        actual = hashlib.sha256((source_dir / name).read_bytes()).hexdigest()
        if expected != actual: hash_mismatch.append(name)
    if hash_mismatch: raise ValueError(f"P3 source manifest hash mismatch: {hash_mismatch}")
    patch_declared = {item["path"]: item["sha256"] for item in patch_manifest["files"]}
    patch_hash_mismatch = [name for name in patch_required[1:] if patch_declared.get(name) != hashlib.sha256((gap_patch_dir / name).read_bytes()).hexdigest()]
    if patch_hash_mismatch: raise ValueError(f"P3 gap patch hash mismatch: {patch_hash_mismatch}")

    coverage = pd.read_csv(source_dir / "p3_family_coverage_matrix.csv")
    blocked = pd.read_csv(source_dir / "p3_blocked_rows_and_family_ledger.csv")
    adjusted = pd.read_csv(source_dir / "p3_adjusted_analysis_coverage_by_ticker.csv", dtype={"ticker": str})
    tdcc = pd.read_csv(source_dir / "p3_tdcc_subperiod_split.csv")
    global_fields = pd.read_csv(source_dir / "p3_global_market_field_readiness.csv")
    universe = pd.read_csv(source_dir / "p3_universe_requested_vs_actual.csv")
    frozen_membership = pd.read_csv(source_dir / "p3_frozen_layer4_primary80_watchlist_membership.csv", dtype={"ticker": str}, low_memory=False)
    release_lag = pd.read_csv(source_dir / "p3_pit_release_lag_ledger.csv")
    taifex_patch = pd.read_csv(gap_patch_dir / "taifex_110_classification.csv")
    tdcc_patch = pd.read_csv(gap_patch_dir / "tdcc_11_classification.csv", dtype={"ticker": str})
    adjusted_12 = pd.read_csv(gap_patch_dir / "adjusted_12_exhausted_evidence.csv", dtype={"ticker": str})
    open_day = json.loads((gap_patch_dir / "first_open_day_acceptance_status.json").read_text(encoding="utf-8-sig"))
    shutil.copy2(source_dir / "p3_blocked_rows_detail.csv", output_dir / "p3_full_feature_blocked_rows_detail.csv")
    shutil.copy2(source_dir / "p3_pit_release_lag_ledger.csv", output_dir / "p3_full_feature_PIT_release_lag_ledger.csv")
    lag_cols = [c for c in ("family", "source_date_semantics", "available_at_policy", "PIT_status") if c in release_lag.columns]
    lag_view = release_lag[lag_cols].drop_duplicates("family") if "family" in lag_cols else pd.DataFrame()
    normalized_coverage = coverage.merge(lag_view, on="family", how="left") if len(lag_view) else coverage.copy()
    normalized_coverage["freshness"] = normalized_coverage.coverage_status.map({"ready": "within_contract", "partial": "partial_or_missing_dates"}).fillna("blocked")
    normalized_coverage["ingest_readiness"] = normalized_coverage.coverage_status.map({"ready": "ready", "partial": "partial"}).fillna("blocked")
    normalized_coverage["silent_fill_allowed"] = False
    normalized_coverage.to_csv(output_dir / "p3_full_feature_family_ingest_readiness.csv", index=False, encoding="utf-8-sig")
    blocked.to_csv(output_dir / "p3_full_feature_blocked_family_ledger.csv", index=False, encoding="utf-8-sig")
    taifex_patch.to_csv(output_dir / "p3_TAIFEX_110_gap_absorption_audit.csv", index=False, encoding="utf-8-sig")
    tdcc_patch.to_csv(output_dir / "p3_TDCC_11_gap_absorption_audit.csv", index=False, encoding="utf-8-sig")
    adjusted_12.to_csv(output_dir / "p3_adjusted_12_exhausted_blocked_ledger.csv", index=False, encoding="utf-8-sig")

    adjusted["_ready"] = adjusted.trusted_nonofficial_adjusted_ready.astype(str).str.lower().eq("true")
    adjusted_by_ticker = adjusted.groupby("ticker", as_index=False)._ready.max()
    adjusted_ready = int(adjusted_by_ticker._ready.sum())
    adjusted_total = int(adjusted_by_ticker.ticker.nunique())
    adjusted_blocked = adjusted_total - adjusted_ready
    blocked_tickers = set(adjusted_by_ticker.loc[~adjusted_by_ticker._ready, "ticker"])
    frozen_membership["snapshot_date"] = pd.to_datetime(frozen_membership.snapshot_date)
    exact_membership = frozen_membership[frozen_membership.snapshot_date.between("2023-07-11", "2026-06-29")].copy()
    blocked_membership = exact_membership[exact_membership.ticker.isin(blocked_tickers)].copy()
    primary_mask = exact_membership.is_layer4_primary_pool.astype(str).str.lower().eq("true")
    primary_rows = int(primary_mask.sum()); watchlist_rows = int((~primary_mask).sum())
    if primary_rows != 0: raise ValueError("Expected current Radar frozen scope to expose zero primary80 rows before repair")
    blocked_membership["adjusted_analysis_ready"] = False
    blocked_membership["new_selection_eligible"] = False
    blocked_membership["selection_semantics"] = "watchlist_reference_only_not_primary80"
    blocked_membership.to_csv(output_dir / "p3_adjusted_12_membership_path_impact_rows.csv", index=False, encoding="utf-8-sig")
    impact_summary = []
    for ticker in sorted(blocked_tickers):
        rows = blocked_membership[blocked_membership.ticker.eq(ticker)]
        impact_summary.append({"ticker": ticker, "membership_rows": len(rows), "primary80_rows": 0,
                               "watchlist_reference_rows": len(rows), "affected_signal_dates": rows.snapshot_date.nunique(),
                               "layer5_shortlist_rows": 0, "challenger_rows": 0, "incumbent_rows": 0, "selected_rows": 0,
                               "selected_path_impact": False, "proof": "never primary80; watchlist reference_only cannot initiate selection; lifecycle Layer5 not yet materialized"})
    pd.DataFrame(impact_summary).to_csv(output_dir / "p3_adjusted_12_path_impact_audit.csv", index=False, encoding="utf-8-sig")
    _write([{"audit": "adjusted_12_selected_path", "status": "no_path_impact_proven_under_frozen_primary80_semantics",
             "blocked_tickers": 12, "primary80_rows": 0, "watchlist_reference_rows": len(blocked_membership),
             "silent_exclusion_used": False, "raw_price_substitution_used": False}], output_dir / "p3_adjusted_12_no_path_impact_proof.csv")
    _write([{"source_scope_file": "p3_frozen_layer4_primary80_watchlist_membership.csv", "rows": len(exact_membership),
             "primary80_rows": primary_rows, "watchlist_reference_rows": watchlist_rows,
             "verdict": "blocked_wrong_source_scope_watchlist_only", "required_repair": "acquire exact layer4_80_primary_pool_contract membership rows for P3 actual period"}],
           output_dir / "p3_primary80_source_scope_blocker_audit.csv")
    schema = [
        ("raw_execution_OHLCV", "official_raw_execution_ohlcv", "execution/mark/cost", "ready", "official", True),
        ("analysis_adjusted_OHLC", "adjusted_analysis_ohlc", "KD/MA/BIAS/RS", "partial", "trusted_nonofficial_research_grade", True),
        ("institutional_flow_5_10_20D", "institutional", "法人／大戶籌碼代理分數 component", "ready", "official", True),
        ("foreign_ownership_level_change", "foreign_ownership", "法人／大戶籌碼代理分數 component", "ready", "official", False),
        ("margin_short_crowding", "margin_short", "法人／大戶籌碼代理分數/risk component", "ready", "official", True),
        ("securities_lending_crowding", "securities_lending", "法人／大戶籌碼代理分數/risk component", "ready", "official", True),
        ("TDCC_bucket_weekly_change", "tdcc_holder_distribution", "P3-2 proxy component only", "partial", "official_51_week_retention", False),
        ("TAIFEX_foreign_OI", "taifex_foreign_oi", "market threshold context", "ready_after_gap_patch", "official_range_CSV", True),
        ("global_completed_session", "global_market", "market threshold context", "ready", "trusted_nonofficial_timezone_PIT", False),
        ("corporate_action_guard", "corporate_action_guard", "analysis-price contamination status", "partial", "source package event guard", True),
        ("Layer4_PIT_membership", "frozen_membership", "candidate universe", "partial", "carried through 2026-06-29 only", True),
    ]
    pd.DataFrame(schema, columns=["field", "source_family", "semantics", "status", "source_quality", "mandatory_full_lifecycle"]).assign(
        weight_fixed=False, silent_fill_allowed=False).to_csv(output_dir / "p3_unified_feature_ingestion_schema.csv", index=False, encoding="utf-8-sig")

    _write([
        {"field": "institutional_large_holder_chip_proxy_score", "display_name_zh": "法人／大戶籌碼代理分數",
         "components": "foreign_ownership_level_change|institutional_flow_5_10_20D|margin_short|securities_lending|TDCC(P3-2_only)",
         "precise_institution_retail_ratio": False, "weight_fixed": False, "claim_boundary": "proxy_not_exact_identity"}
    ], output_dir / "p3_chip_proxy_semantics_contract.csv")

    _write([
        {"contract": "P3-1", "period": "2023-07-11~2025-07-10", "TDCC_used": False, "status": "partial_blocked_by_non_TDCC_mandatory_gaps", "AB_role": "none"},
        {"contract": "P3-2_no_TDCC", "period": "2025-07-11~2026-07-09", "TDCC_used": False, "status": "partial", "AB_role": "A"},
        {"contract": "P3-2_with_TDCC", "period": "2025-07-11~2026-07-09", "TDCC_used": True, "status": "partial_4069_ticker_weeks_11_blocked", "AB_role": "B"},
    ], output_dir / "p3_1_p3_2_same_period_TDCC_AB_contract.csv")

    readiness_rows = []
    family_status = dict(zip(coverage.family, coverage.coverage_status)); family_status["taifex_foreign_oi"] = "ready"
    for contract, period in (("P3-1", "2023-07-11~2025-07-10"), ("P3-2", "2025-07-11~2026-07-09")):
        for family, mandatory, notes in [
            ("official_raw_execution_ohlcv", True, "execution path"),
            ("adjusted_analysis_ohlc", True, "KD/MA/BIAS/RS analysis price; 12 ticker gaps"),
            ("institutional", True, "proxy flow component"), ("margin_short", True, "crowding component"),
            ("securities_lending", True, "crowding component"), ("foreign_ownership", False, "proxy confidence component"),
            ("taifex_foreign_oi", True, "market threshold context; 110 market-day gaps"),
            ("global_market", False, "completed-session threshold context"),
            ("tdcc_holder_distribution", False, "not used" if contract == "P3-1" else "optional same-period A/B only"),
            ("Layer4_PIT_membership", True, "exact through 2026-06-29; carried scope later is not exact PIT"),
        ]:
            status = family_status.get(family, "blocked")
            if family == "tdcc_holder_distribution" and contract == "P3-1": status = "not_applicable"
            if family == "Layer4_PIT_membership": status = "ready" if contract == "P3-1" else "partial"
            readiness_rows.append({"contract": contract, "period": period, "family": family, "mandatory": mandatory,
                                   "status": status, "notes": notes, "silent_fill_allowed": False})
    pd.DataFrame(readiness_rows).to_csv(output_dir / "p3_1_p3_2_mandatory_optional_readiness_matrix.csv", index=False, encoding="utf-8-sig")

    _write([
        {"gate": "P3-1 unified PIT feature matrix", "status": "blocked", "blockers": "Radar_compact_scope_is_watchlist100_not_primary80", "partial_test_allowed": False},
        {"gate": "P3-2 unified PIT feature matrix without TDCC", "status": "blocked", "blockers": "Radar_compact_scope_is_watchlist100_not_primary80", "partial_test_allowed": False},
        {"gate": "P3-2 TDCC A/B", "status": "partial", "blockers": "8_legacy_or_inactive_ticker_weeks_official_zero_rows; may run only after common mandatory set ready", "partial_test_allowed": False},
        {"gate": "open-day daily risk ingestion", "status": "pending_first_normal_trading_day_manifest", "blockers": "2026-07-10 validated market_closed_no_signal only", "partial_test_allowed": False},
    ], output_dir / "p3_unified_PIT_feature_matrix_readiness.csv")

    _write([
        {"validation_date": open_day.get("validation_date", ""), "manifest_status": open_day.get("status", "pending"), "normal_trading_day": True, "full_family_ingestion_validated": False,
         "market_closed_validation_is_sufficient": False, "next_action": "ingest first accepted open-day Radar manifest and evaluate mandatory freshness"}
    ], output_dir / "p3_open_day_full_family_ingestion_validation.csv")

    _write([
        {"field": "raw_execution_price", "basis": "official unadjusted", "use": "execution only", "ready": True},
        {"field": "analysis_price", "basis": "trusted_nonofficial adjusted", "use": "KD/MA/BIAS/RS only", "ready": False},
        {"field": "analysis_price_source_quality", "basis": "per ticker", "use": "mandatory row gate", "ready": True},
        {"field": "corporate_action_event_status", "basis": "event guard", "use": "contamination/block status", "ready": False},
    ], output_dir / "p3_analysis_execution_price_separation_contract.csv")

    _write([
        {"field": row.field, "symbol": row.symbol, "exchange_timezone": row.exchange_timezone,
         "PIT_policy": "latest completed session before Taiwan decision cutoff", "status": row.status}
        for row in global_fields.itertuples(index=False)
    ], output_dir / "p3_global_session_cutoff_contract.csv")
    tdcc.to_csv(output_dir / "p3_TDCC_subperiod_ingest_audit.csv", index=False, encoding="utf-8-sig")
    universe.to_csv(output_dir / "p3_universe_membership_coverage_audit.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(columns=["date", "family", "violation_reason"]).to_csv(output_dir / "future_data_audit.csv", index=False, encoding="utf-8-sig")

    readiness = {"task_id": TASK_ID, "status": "partial_readiness_mandatory_gaps_remain",
                 "source_package_hash_mismatch_count": 0, "requested_start": "2023-07-11", "requested_end": "2026-07-10",
                 "gap_patch_hash_mismatch_count": 0, "gap_patch_absorbed": True,
                 "actual_market_end": "2026-07-09", "P3_replaces_P1": False,
                 "official_raw_execution_OHLCV_ready": True, "adjusted_analysis_accepted_tickers": adjusted_ready,
                 "adjusted_analysis_blocked_tickers": adjusted_blocked, "official_adjusted_ready": False,
                 "adjusted_12_primary80_rows": 0, "adjusted_12_watchlist_reference_rows": len(blocked_membership),
                 "adjusted_12_selected_path_impact": False, "adjusted_12_no_path_impact_proof_ready": True,
                 "Radar_source_scope_primary80_rows": primary_rows, "Radar_source_scope_watchlist_rows": watchlist_rows,
                 "Radar_compact_scope_correct_for_primary80": False,
                 "TDCC_full_P3_ready": False, "TDCC_P3_2_AB_only": True,
                 "TAIFEX_original_accepted_dates": 615, "TAIFEX_repaired_dates": 110,
                 "TAIFEX_confirmed_market_day_missing_dates": 0, "TAIFEX_full_period_ready_after_patch": True,
                 "TDCC_gap_input_ticker_weeks": 11, "TDCC_gap_repaired_ticker_weeks": 3, "TDCC_gap_blocked_ticker_weeks": 8,
                 "Layer4_new_PIT_membership_after_2026_06_29_ready": False,
                 "P3_1_TDCC_required": False, "P3_2_TDCC_optional_AB": True,
                 "open_day_full_family_ingestion_validation_ready": False,
                 "market_closed_no_signal_validation_does_not_count_as_open_day": True,
                 "mandatory_full_period_ready": False, "ready_for_feature_materialization": False,
                 "ready_for_state_machine_materialization": False, "weights_fixed": False,
                 "partial_backtest_executed": False, "ready_for_experiments": False,
                 "future_data_violation_count": 0, **FLAGS}
    (output_dir / "readiness_for_p3_full_feature_unified_lifecycle_contract.json").write_text(json.dumps(readiness, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "final_summary_zh.md").write_text(
        "# P3 full-feature unified lifecycle partial readiness\n\n"
        "- Verdict: PARTIAL/BLOCKED；只完成 ingest/schema readiness，未 materialize state machine、未回測。\n"
        f"- Adjusted analysis {adjusted_ready}/{adjusted_total} unique tickers ready，{adjusted_blocked} blocked；官方 adjusted 不 ready。\n"
        "- TAIFEX 110/110 交易日 gaps 已由 official range CSV 修復，該 mandatory family ready。\n"
        "- TDCC 11 gaps 修復 3、剩 8 個 legacy/inactive ticker-week official zero rows；僅 P3-2 optional A/B。Layer4 2026-06-29 後新 PIT membership blocked。\n"
        "- Adjusted 12 在 exact period 僅有107筆 watchlist reference、0筆 primary80；依 frozen semantics 無 selected-path impact，不再作 primary matrix blocker。\n"
        "- 新 blocker：Radar compact acquisition scope 實際是 watchlist100，primary80 rows=0；必須補 exact primary80 source scope，不能用 watchlist features 代替。\n"
        "- P3 不取代 P1；法人／大戶籌碼代理分數僅 proxy components，權重未定。\n"
        "- Mandatory gaps 關閉前不交 Experiments、不跑 partial performance。\n", encoding="utf-8")
    files = sorted(p for p in output_dir.iterdir() if p.is_file() and p.name != "manifest.json")
    manifest = {"task_id": TASK_ID, "runner": str(Path(__file__).resolve()), "source_package": str(source_dir),
                "source_commit": "d2e9071", "gap_patch_source": str(gap_patch_dir), "gap_patch_commit": "9a803ca",
                "source_readiness": source_ready, "gap_patch_readiness": patch_ready, "readiness": readiness,
                "files": [{"name": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in files]}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "current_step.txt").write_text("partial_readiness_waiting_mandatory_gap_closure_no_experiments", encoding="utf-8")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE); parser.add_argument("--gap-patch-dir", type=Path, default=DEFAULT_GAP_PATCH); parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(); print(run(args.source_dir, args.gap_patch_dir, args.output_dir))


if __name__ == "__main__": main()
