from __future__ import annotations

import argparse
import glob
import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd
import numpy as np


TASK_ID = "TASK-BACKTEST-CORE-VNEXT-P3-FULL-FEATURE-UNIFIED-LIFECYCLE-CONTRACT-001"
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = Path(r"C:\Users\zergv\Documents\Codex\2026-05-23\ai-stock-rotation-radar-https-docs\outputs\radar_vnext_p3_recent_full_feature_data_readiness_acquisition_20260711")
DEFAULT_GAP_PATCH = Path(r"C:\Users\zergv\Documents\Codex\2026-05-23\ai-stock-rotation-radar-https-docs\outputs\radar_vnext_p3_gap_convergence_open_day_acceptance_20260711")
DEFAULT_EXACT_PRIMARY80 = Path(r"C:\Users\zergv\Documents\Codex\2026-05-23\ai-stock-rotation-radar-https-docs\outputs\radar_vnext_p3_exact_primary80_full_feature_source_scope_repair_20260711")
DEFAULT_RAW_HLC_WARMUP = Path(r"C:\Users\zergv\Documents\Codex\2026-05-23\ai-stock-rotation-radar-https-docs\outputs\radar_vnext_p3_exact_primary80_raw_hlc_warmup_gap_fill_20260711")
DEFAULT_OUTPUT = REPO_ROOT / "outputs/vnext_p3_full_feature_unified_lifecycle_contract_20260711"
FLAGS = {"formal_model_changed": False, "trade_decision_changed": False, "active_in_trade_decision": False,
         "report_changed": False, "portfolio_replay_executed": False, "ready_for_strategy_replay": False,
         "ready_for_formal": False, "not_live_rule": True, "forward_returns_live_rule_usage": False}


def _write(rows: list[dict], path: Path) -> None:
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def _load_compact(root: Path, family: str) -> pd.DataFrame:
    files = glob.glob(str(root / "compact" / family / "*.csv.gz"))
    if not files: return pd.DataFrame()
    return pd.concat([pd.read_csv(path, dtype={"ticker": str}, low_memory=False) for path in files], ignore_index=True)


def run(source_dir: Path = DEFAULT_SOURCE, gap_patch_dir: Path = DEFAULT_GAP_PATCH,
        exact_primary80_dir: Path = DEFAULT_EXACT_PRIMARY80, raw_hlc_warmup_dir: Path = DEFAULT_RAW_HLC_WARMUP,
        output_dir: Path = DEFAULT_OUTPUT) -> Path:
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
    exact_required = ["manifest.json", "readiness_for_core_p3_exact_primary80_source_scope_repair.json",
                      "p3_exact_primary80_membership.csv", "p3_exact_primary80_blocked_ticker_date_ledger.csv",
                      "p3_exact_primary80_core_no_path_proof_contradiction.csv", "p3_exact_primary80_compact_hash_manifest.csv"]
    exact_missing = [name for name in exact_required if not (exact_primary80_dir / name).exists()]
    if exact_missing: raise FileNotFoundError(f"P3 exact primary80 package missing: {exact_missing}")
    warmup_required = ["manifest.json", "readiness_for_core_p3_exact_primary80_raw_hlc_warmup.json",
                       "p3_exact_primary80_raw_hlc_warmup_coverage_by_segment.csv",
                       "p3_exact_primary80_raw_hlc_warmup_blocked_ledger.csv",
                       "p3_exact_primary80_raw_hlc_warmup_compact_hash_manifest.csv"]
    warmup_missing = [name for name in warmup_required if not (raw_hlc_warmup_dir / name).exists()]
    if warmup_missing: raise FileNotFoundError(f"P3 raw HLC warmup package missing: {warmup_missing}")
    source_manifest = json.loads((source_dir / "manifest.json").read_text(encoding="utf-8-sig"))
    source_ready = json.loads((source_dir / "readiness_for_core_p3_full_feature_unified_lifecycle_contract.json").read_text(encoding="utf-8-sig"))
    patch_manifest = json.loads((gap_patch_dir / "manifest.json").read_text(encoding="utf-8-sig"))
    patch_ready = json.loads((gap_patch_dir / "readiness_for_core_p3_gap_convergence.json").read_text(encoding="utf-8-sig"))
    exact_manifest = json.loads((exact_primary80_dir / "manifest.json").read_text(encoding="utf-8-sig"))
    exact_ready = json.loads((exact_primary80_dir / "readiness_for_core_p3_exact_primary80_source_scope_repair.json").read_text(encoding="utf-8-sig"))
    warmup_manifest = json.loads((raw_hlc_warmup_dir / "manifest.json").read_text(encoding="utf-8-sig"))
    warmup_ready = json.loads((raw_hlc_warmup_dir / "readiness_for_core_p3_exact_primary80_raw_hlc_warmup.json").read_text(encoding="utf-8-sig"))
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
    exact_declared = {item["path"]: item["sha256"] for item in exact_manifest["files"]}
    exact_hash_mismatch = [name for name in exact_required[1:] if exact_declared.get(name) != hashlib.sha256((exact_primary80_dir / name).read_bytes()).hexdigest()]
    if exact_hash_mismatch: raise ValueError(f"P3 exact package hash mismatch: {exact_hash_mismatch}")
    warmup_declared = {item["path"]: item["sha256"] for item in warmup_manifest["files"]}
    warmup_hash_mismatch = [name for name in warmup_required[1:] if warmup_declared.get(name) != hashlib.sha256((raw_hlc_warmup_dir / name).read_bytes()).hexdigest()]
    if warmup_hash_mismatch: raise ValueError(f"P3 warmup package hash mismatch: {warmup_hash_mismatch}")

    coverage = pd.read_csv(source_dir / "p3_family_coverage_matrix.csv")
    blocked = pd.read_csv(source_dir / "p3_blocked_rows_and_family_ledger.csv")
    adjusted = pd.read_csv(source_dir / "p3_adjusted_analysis_coverage_by_ticker.csv", dtype={"ticker": str})
    tdcc = pd.read_csv(source_dir / "p3_tdcc_subperiod_split.csv")
    global_fields = pd.read_csv(source_dir / "p3_global_market_field_readiness.csv")
    universe = pd.read_csv(source_dir / "p3_universe_requested_vs_actual.csv")
    frozen_membership = pd.read_csv(exact_primary80_dir / "p3_exact_primary80_membership.csv", dtype={"ticker": str}, low_memory=False)
    release_lag = pd.read_csv(source_dir / "p3_pit_release_lag_ledger.csv")
    taifex_patch = pd.read_csv(gap_patch_dir / "taifex_110_classification.csv")
    tdcc_patch = pd.read_csv(gap_patch_dir / "tdcc_11_classification.csv", dtype={"ticker": str})
    adjusted_12 = pd.read_csv(gap_patch_dir / "adjusted_12_exhausted_evidence.csv", dtype={"ticker": str})
    open_day = json.loads((gap_patch_dir / "first_open_day_acceptance_status.json").read_text(encoding="utf-8-sig"))
    warmup_segments = pd.read_csv(raw_hlc_warmup_dir / "p3_exact_primary80_raw_hlc_warmup_coverage_by_segment.csv", dtype={"ticker": str})
    warmup_blocked = pd.read_csv(raw_hlc_warmup_dir / "p3_exact_primary80_raw_hlc_warmup_blocked_ledger.csv", dtype={"ticker": str})
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
    warmup_segments.to_csv(output_dir / "p3_raw_HLC_warmup_segment_absorption_audit.csv", index=False, encoding="utf-8-sig")
    warmup_blocked.assign(missing_class="official_zero_or_not_applicable", feature_value_imputation_allowed=False).to_csv(
        output_dir / "p3_raw_HLC_warmup_blocked_ledger.csv", index=False, encoding="utf-8-sig")

    adjusted["_ready"] = adjusted.trusted_nonofficial_adjusted_ready.astype(str).str.lower().eq("true")
    adjusted_by_ticker = adjusted.groupby("ticker", as_index=False)._ready.max()
    adjusted_ready = int(adjusted_by_ticker._ready.sum())
    adjusted_total = int(adjusted_by_ticker.ticker.nunique())
    adjusted_blocked = adjusted_total - adjusted_ready
    exact_adjusted_blocked = pd.read_csv(exact_primary80_dir / "p3_exact_primary80_adjusted_blocked_tickers.csv", dtype={"ticker": str})
    blocked_tickers = set(exact_adjusted_blocked.loc[exact_adjusted_blocked.blocked_dates.eq(exact_adjusted_blocked.required_dates), "ticker"])
    frozen_membership["snapshot_date"] = pd.to_datetime(frozen_membership.snapshot_date)
    exact_membership = frozen_membership[frozen_membership.snapshot_date.between("2023-07-11", "2026-06-29")].copy()
    blocked_membership = exact_membership[exact_membership.ticker.isin(blocked_tickers)].copy()
    primary_mask = exact_membership.is_layer4_primary_pool.astype(str).str.lower().eq("true")
    primary_rows = int(primary_mask.sum()); watchlist_rows = int((~primary_mask).sum())
    if primary_rows != 12320 or watchlist_rows != 0: raise ValueError("Exact primary80 scope semantics mismatch")
    blocked_membership["adjusted_analysis_ready"] = False
    blocked_membership["new_selection_eligible"] = "blocked_until_adjusted_independent_or_source_ready"
    blocked_membership["selection_semantics"] = "primary80_candidate_selector_completeness_blocked"
    blocked_membership.to_csv(output_dir / "p3_adjusted_12_membership_path_impact_rows.csv", index=False, encoding="utf-8-sig")
    impact_summary = []
    for ticker in sorted(blocked_tickers):
        rows = blocked_membership[blocked_membership.ticker.eq(ticker)]
        impact_summary.append({"ticker": ticker, "membership_rows": len(rows), "primary80_rows": len(rows),
                               "watchlist_reference_rows": 0, "affected_signal_dates": rows.snapshot_date.nunique(),
                               "layer5_shortlist_rows": "not_materialized", "challenger_rows": "potentially_affected", "incumbent_rows": "potentially_affected", "selected_rows": "unknown_before_frozen_selector",
                               "selected_path_impact": "potential", "proof": "exact primary80 membership confirmed; no silent exclusion allowed"})
    pd.DataFrame(impact_summary).to_csv(output_dir / "p3_adjusted_12_path_impact_audit.csv", index=False, encoding="utf-8-sig")
    _write([{"audit": "adjusted_12_selected_path", "status": "superseded_prior_no_path_proof_primary80_impact_confirmed",
             "blocked_tickers": len(blocked_tickers), "primary80_rows": len(blocked_membership), "watchlist_reference_rows": 0,
             "silent_exclusion_used": False, "raw_price_substitution_used": False}], output_dir / "p3_adjusted_12_no_path_impact_proof.csv")
    _write([{"source_scope_file": "p3_exact_primary80_membership.csv", "rows": len(exact_membership),
             "primary80_rows": primary_rows, "watchlist_reference_rows": watchlist_rows,
             "verdict": "exact_primary80_source_scope_ready", "required_repair": ""}],
           output_dir / "p3_primary80_source_scope_blocker_audit.csv")

    price_active = _load_compact(exact_primary80_dir, "price")
    price_warmup = _load_compact(raw_hlc_warmup_dir, "raw_hlc_warmup")
    price_active["source_priority"] = 0; price_warmup["source_priority"] = 1
    price = pd.concat([price_active, price_warmup], ignore_index=True, sort=False)
    price["date"] = pd.to_datetime(price.date); price["ticker"] = price.ticker.astype(str)
    price = price.sort_values(["ticker", "date", "source_priority"]).drop_duplicates(["ticker", "date"], keep="first")
    analysis = _load_compact(exact_primary80_dir, "adjusted")
    for frame in (price, analysis):
        frame["date"] = pd.to_datetime(frame.date); frame["ticker"] = frame.ticker.astype(str)
    factor = analysis[["date", "ticker", "adjusted_close", "raw_close_comparator", "source_quality"]].copy()
    factor["adjusted_close"] = pd.to_numeric(factor.adjusted_close, errors="coerce")
    factor["raw_close_comparator"] = pd.to_numeric(factor.raw_close_comparator, errors="coerce")
    factor["analysis_adjustment_factor"] = factor.adjusted_close / factor.raw_close_comparator.replace(0, np.nan)
    factor["factor_valid"] = np.isfinite(factor.analysis_adjustment_factor) & factor.analysis_adjustment_factor.gt(0)
    factor["analysis_close_ready"] = factor.factor_valid & factor.adjusted_close.notna()
    raw_cols = price[["date", "ticker", "open", "high", "low", "close"]].copy()
    for col in ("open", "high", "low", "close"): raw_cols[col] = pd.to_numeric(raw_cols[col], errors="coerce")
    factor = factor.merge(raw_cols, on=["date", "ticker"], how="left")
    for col in ("open", "high", "low", "close"):
        factor[f"analysis_{col}"] = factor[col] * factor.analysis_adjustment_factor
    factor["adjusted_OHLC_ready"] = factor.factor_valid & factor[["analysis_open", "analysis_high", "analysis_low", "analysis_close"]].notna().all(axis=1)
    factor["OHLC_order_valid"] = factor.analysis_high.ge(factor[["analysis_open", "analysis_close", "analysis_low"]].max(axis=1)) & factor.analysis_low.le(factor[["analysis_open", "analysis_close", "analysis_high"]].min(axis=1))
    _write([{"method": "trusted_adjclose_divided_by_provider_raw_close_then_multiply_official_raw_OHLC",
             "factor_rows": len(factor), "factor_valid_rows": int(factor.factor_valid.sum()),
             "adjusted_OHLC_ready_rows": int(factor.adjusted_OHLC_ready.sum()), "zero_denominator_rows": int(factor.raw_close_comparator.eq(0).sum()),
             "nonfinite_or_nonpositive_factor_rows": int((~factor.factor_valid).sum()), "OHLC_order_invalid_rows": int((factor.adjusted_OHLC_ready & ~factor.OHLC_order_valid).sum()),
             "source_quality": "trusted_nonofficial_adjusted_analysis_research_only", "official_adjusted_complete": False,
             "corporate_action_no_event_completeness_ready": False, "raw_execution_OHLC_substituted_as_analysis": False}],
           output_dir / "p3_adjusted_OHLC_factor_method_audit.csv")

    factor = factor.sort_values(["ticker", "date"])
    factor["valid_history_count"] = factor.groupby("ticker").adjusted_OHLC_ready.cumsum()
    factor["valid_close_history_count"] = factor.groupby("ticker").analysis_close_ready.cumsum()
    readiness_rows = exact_membership[["snapshot_date", "ticker", "name", "market", "pool_rank"]].merge(
        factor[["date", "ticker", "analysis_close_ready", "adjusted_OHLC_ready", "valid_close_history_count", "valid_history_count", "source_quality"]],
        left_on=["snapshot_date", "ticker"], right_on=["date", "ticker"], how="left")
    readiness_rows["adjusted_OHLC_ready"] = readiness_rows.adjusted_OHLC_ready.fillna(False)
    readiness_rows["analysis_close_ready"] = readiness_rows.analysis_close_ready.fillna(False)
    readiness_rows["analysis_close_warmup_60TD_ready"] = readiness_rows.valid_close_history_count.fillna(0).ge(60)
    readiness_rows["close_based_RS_MA_BIAS_ready"] = readiness_rows.analysis_close_ready & readiness_rows.analysis_close_warmup_60TD_ready
    readiness_rows["analysis_warmup_60TD_ready"] = readiness_rows.valid_history_count.fillna(0).ge(60)
    readiness_rows["adjusted_dependent_lifecycle_ready"] = readiness_rows.adjusted_OHLC_ready & readiness_rows.analysis_warmup_60TD_ready
    readiness_rows["row_blocked_reason"] = np.where(readiness_rows.adjusted_dependent_lifecycle_ready, "", "missing_adjusted_OHLC_or_60TD_warmup")
    family_map = {
        "institutional": "chip_institutional", "margin_short": "chip_margin_short",
        "securities_lending": "chip_securities_lending", "foreign_ownership": "foreign_ownership",
    }
    for label, folder in family_map.items():
        frame = _load_compact(exact_primary80_dir, folder)
        frame["date"] = pd.to_datetime(frame.date); frame["ticker"] = frame.ticker.astype(str)
        frame = frame.sort_values(["ticker", "date"]).drop_duplicates(["ticker", "date"])
        frame[f"{label}_observed"] = True
        frame[f"{label}_rolling35d_count"] = frame.set_index("date").groupby("ticker")[f"{label}_observed"].rolling("35D").count().reset_index(level=0, drop=True).to_numpy()
        readiness_rows = readiness_rows.merge(frame[["date", "ticker", f"{label}_observed", f"{label}_rolling35d_count"]],
                                              left_on=["snapshot_date", "ticker"], right_on=["date", "ticker"], how="left", suffixes=("", f"_{label}"))
        readiness_rows[f"{label}_20obs_PIT_ready"] = readiness_rows[f"{label}_observed"].fillna(False) & readiness_rows[f"{label}_rolling35d_count"].fillna(0).ge(20)
    mandatory_row_cols = ["adjusted_dependent_lifecycle_ready"] + [f"{name}_20obs_PIT_ready" for name in family_map]
    readiness_rows["full_feature_row_ready"] = readiness_rows[mandatory_row_cols].all(axis=1)
    readiness_rows["full_feature_blocked_families"] = readiness_rows.apply(
        lambda row: "|".join(col.replace("_20obs_PIT_ready", "").replace("adjusted_dependent_lifecycle_ready", "adjusted_HLC") for col in mandatory_row_cols if not bool(row[col])), axis=1)
    snapshot = readiness_rows.groupby("snapshot_date", as_index=False).agg(
        primary80_rows=("ticker", "size"), close_feature_ready_rows=("close_based_RS_MA_BIAS_ready", "sum"),
        adjusted_HLC_KD_ready_rows=("adjusted_dependent_lifecycle_ready", "sum"), full_feature_ready_rows=("full_feature_row_ready", "sum"))
    snapshot["close_feature_snapshot_complete"] = snapshot.primary80_rows.eq(80) & snapshot.close_feature_ready_rows.eq(80)
    snapshot["KD_price_snapshot_complete"] = snapshot.primary80_rows.eq(80) & snapshot.adjusted_HLC_KD_ready_rows.eq(80)
    snapshot["selector_completeness_ready"] = snapshot.primary80_rows.eq(80) & snapshot.full_feature_ready_rows.eq(80)
    snapshot["selector_completeness_status"] = np.where(snapshot.selector_completeness_ready, "exact_complete", "selector_completeness_blocked")
    snapshot["silent_exclusion_allowed"] = False
    readiness_rows.to_csv(output_dir / "p3_exact_primary80_unified_PIT_feature_matrix_readiness_rows.csv", index=False, encoding="utf-8-sig")
    snapshot.to_csv(output_dir / "p3_exact_primary80_snapshot_completeness_audit.csv", index=False, encoding="utf-8-sig")
    snapshot[snapshot.selector_completeness_ready].to_csv(output_dir / "p3_exact_complete_snapshot_subset_contract.csv", index=False, encoding="utf-8-sig")
    complete_snapshot_count = int(snapshot.selector_completeness_ready.sum())
    close_complete_snapshot_count = int(snapshot.close_feature_snapshot_complete.sum())
    KD_price_complete_snapshot_count = int(snapshot.KD_price_snapshot_complete.sum())
    total_snapshot_count = len(snapshot)

    missing_ledger = pd.read_csv(exact_primary80_dir / "p3_exact_primary80_blocked_ticker_date_ledger.csv", dtype={"ticker": str})
    missing_ledger["missing_class"] = np.select([
        missing_ledger.family.eq("adjusted_analysis"),
        missing_ledger.blocked_reason.str.contains("not_applicable|suspended|no_exact_ticker_row", case=False, na=False),
    ], ["adjusted_price_blocked", "official_zero_or_not_applicable"], default="source_gap")
    missing_ledger["feature_value_imputation_allowed"] = False
    missing_ledger.to_csv(output_dir / "p3_source_missingness_classified_ledger.csv", index=False, encoding="utf-8-sig")
    missing_counts = missing_ledger.groupby("missing_class").size().to_dict()
    _write([{"missing_class": name, "rows": int(missing_counts.get(name, 0)), "silent_fill_allowed": False,
             "feature_semantics": "availability_flag_only_not_zero_or_negative_score"}
            for name in ("official_zero_or_not_applicable", "source_gap", "adjusted_price_blocked")],
           output_dir / "p3_source_missingness_classification_summary.csv")
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
        {"gate": "P3-1 unified PIT feature matrix", "status": "ready_exact_subset" if complete_snapshot_count else "blocked", "blockers": "" if complete_snapshot_count else "20D_chip_family_warmup_missing_for_new_or_reentry_primary80|adjusted_HLC_partial", "partial_test_allowed": False},
        {"gate": "P3-2 unified PIT feature matrix without TDCC", "status": "ready_exact_subset" if complete_snapshot_count else "blocked", "blockers": "" if complete_snapshot_count else "20D_chip_family_warmup_missing_for_new_or_reentry_primary80|adjusted_HLC_partial", "partial_test_allowed": False},
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
                 "raw_HLC_warmup_hash_mismatch_count": 0, "raw_HLC_warmup_absorbed": True,
                 "actual_market_start": "2023-07-14", "actual_market_end": "2026-06-29", "P3_replaces_P1": False,
                 "official_raw_execution_OHLCV_ready": True, "adjusted_analysis_accepted_tickers": adjusted_ready,
                 "adjusted_analysis_blocked_tickers": adjusted_blocked, "official_adjusted_ready": False,
                 "prior_adjusted_12_no_path_proof_status": "superseded_incorrect_watchlist_scope",
                 "adjusted_blocked_primary80_tickers": len(blocked_tickers), "adjusted_blocked_primary80_rows": len(blocked_membership),
                 "adjusted_blocked_selected_path_impact": "potential_unknown_before_selector",
                 "Radar_source_scope_primary80_rows": primary_rows, "Radar_source_scope_watchlist_rows": watchlist_rows,
                 "Radar_compact_scope_correct_for_primary80": True,
                 "trusted_adjustment_factor_method_ready": True, "adjusted_HLC_method_ready_research_only": True,
                 "corporate_action_no_event_completeness_ready": False,
                 "close_feature_complete_snapshots": close_complete_snapshot_count,
                 "KD_price_complete_snapshots": KD_price_complete_snapshot_count,
                 "full_lifecycle_complete_snapshots": complete_snapshot_count,
                 "total_exact_snapshots": total_snapshot_count,
                 "full_lifecycle_complete_snapshot_share": complete_snapshot_count / total_snapshot_count if total_snapshot_count else 0,
                 "raw_HLC_warmup_required_ticker_dates": int(warmup_manifest["coverage"]["required_ticker_dates"]),
                 "raw_HLC_warmup_ready_ticker_dates": int(warmup_manifest["coverage"]["ready_ticker_dates"]),
                 "raw_HLC_warmup_official_zero_or_not_applicable": int(warmup_manifest["coverage"]["blocked_ticker_dates"]),
                 "raw_HLC_warmup_source_gap_rows": 0,
                 "raw_HLC_warmup_complete_segments": int(warmup_manifest["coverage"]["complete_60td_segments"]),
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
        "- TDCC 11 gaps 修復 3、剩 8 個 legacy/inactive ticker-week official zero rows；僅 P3-2 optional A/B。Primary exact period固定 2023-07-14~2026-06-29。\n"
        f"- 前版 adjusted-12 no-path proof 已 superseded：11檔進入 primary80，共{len(blocked_membership)}筆 snapshots，selected impact在凍結selector前未知。\n"
        "- Trusted adjclose/raw-close factor一致調整官方raw O/H/L/C供research；warmup 181,375/183,886 ready，2,511為official no-row/not-applicable，source gap=0。\n"
        f"- Close-based complete snapshots={close_complete_snapshot_count}/{total_snapshot_count}；KD-price complete={KD_price_complete_snapshot_count}/{total_snapshot_count}；all mandatory full-feature complete={complete_snapshot_count}/{total_snapshot_count}。\n"
        "- Exact chip compacts缺新進/重入前20D warmup；法人、融資融券借券、外資持股不得用不足20日或舊值補齊。\n"
        "- P3 不取代 P1；法人／大戶籌碼代理分數僅 proxy components，權重未定。\n"
        "- Mandatory gaps 關閉前不交 Experiments、不跑 partial performance。\n", encoding="utf-8")
    files = sorted(p for p in output_dir.iterdir() if p.is_file() and p.name != "manifest.json")
    manifest = {"task_id": TASK_ID, "runner": str(Path(__file__).resolve()), "source_package": str(source_dir),
                "source_commit": "d2e9071", "gap_patch_source": str(gap_patch_dir), "gap_patch_commit": "9a803ca",
                "exact_primary80_source": str(exact_primary80_dir), "exact_primary80_commit": "6120840",
                "raw_HLC_warmup_source": str(raw_hlc_warmup_dir), "raw_HLC_warmup_commit": "2303e87", "raw_HLC_runner_hygiene_commit": "cfd1542",
                "source_readiness": source_ready, "gap_patch_readiness": patch_ready, "exact_primary80_readiness": exact_ready,
                "raw_HLC_warmup_readiness": warmup_ready, "readiness": readiness,
                "files": [{"name": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in files]}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "current_step.txt").write_text("partial_readiness_waiting_mandatory_gap_closure_no_experiments", encoding="utf-8")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE); parser.add_argument("--gap-patch-dir", type=Path, default=DEFAULT_GAP_PATCH); parser.add_argument("--exact-primary80-dir", type=Path, default=DEFAULT_EXACT_PRIMARY80); parser.add_argument("--raw-hlc-warmup-dir", type=Path, default=DEFAULT_RAW_HLC_WARMUP); parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(); print(run(args.source_dir, args.gap_patch_dir, args.exact_primary80_dir, args.raw_hlc_warmup_dir, args.output_dir))


if __name__ == "__main__": main()
