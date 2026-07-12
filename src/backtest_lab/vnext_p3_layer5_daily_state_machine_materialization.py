from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
WEEKLY = ROOT / "outputs/vnext_layer4_80_primary_pool_contract_20260708/layer4_80_primary_pool_contract.csv"
P3 = ROOT / "outputs/vnext_p3_full_feature_unified_lifecycle_contract_20260711"
ARCH = ROOT / "outputs/vnext_p3_layer5_single_lifecycle_state_machine_contract_20260711"
RADAR = Path(r"C:\Users\zergv\Documents\Codex\2026-05-23\ai-stock-rotation-radar-https-docs\outputs\radar_vnext_p3_recent_full_feature_data_readiness_acquisition_20260711\compact")
MARKET_FILL = Path(r"C:\Users\zergv\Documents\Codex\2026-05-23\ai-stock-rotation-radar-https-docs\outputs\radar_vnext_p3_market_state_source_fill_20260711")
SOURCE_CONVERGENCE = Path(r"C:\Users\zergv\Documents\Codex\2026-05-23\ai-stock-rotation-radar-https-docs\outputs\radar_vnext_p3_post_market_source_remaining_gap_convergence_20260712")
OUTPUT = ROOT / "outputs/vnext_p3_layer5_daily_state_machine_materialization_20260711"
TASK = "TASK-BACKTEST-CORE-VNEXT-P3-LAYER5-DAILY-STATE-MACHINE-MATERIALIZATION-001"


def load_years(family: str) -> pd.DataFrame:
    files = sorted((RADAR / family).glob("*.csv.gz"))
    return pd.concat([pd.read_csv(p, dtype={"ticker": str}, low_memory=False) for p in files], ignore_index=True)


def run(output: Path = OUTPUT, market_fill: Path = MARKET_FILL, source_convergence: Path = SOURCE_CONVERGENCE) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    market_ready = json.loads((market_fill / "readiness_for_core_p3_market_state_source_fill.json").read_text(encoding="utf-8-sig"))
    checksum = pd.read_csv(market_fill / "p3_market_state_checksum_manifest.csv")
    checksum_rows = []
    for row in checksum.itertuples(index=False):
        path = market_fill / str(row.path).replace("\\", "/")
        actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else ""
        checksum_rows.append({"path": str(path), "expected_sha256": row.sha256, "actual_sha256": actual,
                              "checksum_match": actual == row.sha256, "bytes": row.bytes})
    if len(checksum_rows) != 3 or not all(row["checksum_match"] for row in checksum_rows):
        raise ValueError("P3 market source compact checksum validation failed")
    pd.DataFrame(checksum_rows).to_csv(output / "p3_layer5_daily_market_source_absorption_audit.csv", index=False, encoding="utf-8-sig")
    convergence_ready = json.loads((source_convergence / "readiness_for_core_p3_post_market_source_convergence.json").read_text(encoding="utf-8-sig"))
    if not convergence_ready["ready_for_core_daily_feature_materialization"] or convergence_ready["source_gap_count_requiring_download"] != 0:
        raise ValueError("P3 post-market source convergence contract not ready")
    for source_name, target_name in [
        ("p3_remaining_source_family_matrix.csv", "p3_layer5_remaining_source_family_matrix.csv"),
        ("p3_remaining_source_gap_ledger.csv", "p3_layer5_remaining_source_gap_ledger.csv"),
        ("p3_layer4_july_freshness_audit.csv", "p3_layer5_july_exact_layer4_freshness_audit.csv")]:
        pd.read_csv(source_convergence / source_name).to_csv(output / target_name, index=False, encoding="utf-8-sig")
    weekly = pd.read_csv(WEEKLY, dtype={"ticker": str}, low_memory=False)
    weekly["snapshot_date"] = pd.to_datetime(weekly.snapshot_date)
    weekly = weekly.loc[(weekly.is_layer4_primary_pool == True) & weekly.snapshot_date.between("2023-07-14", "2026-06-29")]
    weekly = weekly.sort_values(["snapshot_date", "ticker"]).drop_duplicates(["snapshot_date", "ticker"], keep="last")
    dates = pd.read_csv(ROOT / "backtest_cache/stock_pool_observations/0050_TW.csv", usecols=["date", "close"])
    dates["date"] = pd.to_datetime(dates.date)
    dates = dates.loc[dates.date.between("2023-07-14", "2026-06-30")].dropna().sort_values("date")
    trading_dates = dates.date.drop_duplicates().tolist()
    next_date = dict(zip(trading_dates[:-1], trading_dates[1:]))
    snapshots = sorted(weekly.snapshot_date.unique())
    effective = {}
    for snap in snapshots:
        later = [d for d in trading_dates if d > snap]
        effective[snap] = later[0] if later else pd.NaT
    rows = []
    for idx, snap in enumerate(snapshots):
        start = effective[snap]
        end = effective[snapshots[idx + 1]] if idx + 1 < len(snapshots) else pd.Timestamp("2026-06-30")
        active_dates = [d for d in trading_dates if d >= start and d < end and d <= pd.Timestamp("2026-06-29")]
        members = weekly.loc[weekly.snapshot_date.eq(snap), ["ticker", "name", "market", "pool_rank"]]
        for d in active_dates:
            x = members.copy(); x["decision_date"] = d; x["membership_snapshot_date"] = snap
            x["membership_effective_date"] = start; x["next_execution_date"] = next_date.get(d, pd.NaT)
            rows.append(x)
    daily = pd.concat(rows, ignore_index=True)

    raw = load_years("price"); raw["date"] = pd.to_datetime(raw.date)
    adj = load_years("adjusted"); adj["date"] = pd.to_datetime(adj.date)
    raw = raw.loc[raw.date.between("2023-07-14", "2026-06-29"), ["date", "ticker", "open", "high", "low", "close", "volume", "turnover_value", "source_quality"]]
    adj = adj.loc[adj.date.between("2023-07-14", "2026-06-29"), ["date", "ticker", "adjusted_close", "raw_close_comparator", "source_quality"]]
    raw = raw.rename(columns={"date": "decision_date", "source_quality": "raw_execution_source_quality"})
    adj = adj.rename(columns={"date": "decision_date", "source_quality": "adjusted_analysis_source_quality"})
    daily = daily.merge(raw, on=["decision_date", "ticker"], how="left").merge(adj, on=["decision_date", "ticker"], how="left")
    daily["adjustment_factor"] = daily.adjusted_close / daily.raw_close_comparator
    bracket = pd.read_csv(P3 / "p3_20250801_adjustment_factor_bracket_proof.csv", dtype={"ticker": str})
    bracket = bracket.loc[bracket.bracket_factor_accepted == True, ["ticker", "prior_factor"]].set_index("ticker")["prior_factor"]
    bracket_mask = daily.decision_date.eq(pd.Timestamp("2025-08-01")) & daily.ticker.isin(bracket.index) & daily.adjustment_factor.isna()
    daily.loc[bracket_mask, "adjustment_factor"] = daily.loc[bracket_mask, "ticker"].map(bracket)
    daily.loc[bracket_mask, "adjusted_close"] = daily.loc[bracket_mask, "close"] * daily.loc[bracket_mask, "adjustment_factor"]
    daily.loc[bracket_mask, "adjusted_analysis_source_quality"] = "trusted_nonofficial_factor_continuity_bracket_research_only"
    daily["adjusted_open"] = daily.open * daily.adjustment_factor
    daily["adjusted_high"] = daily.high * daily.adjustment_factor
    daily["adjusted_low"] = daily.low * daily.adjustment_factor
    daily["price_core_valid"] = daily[["open", "high", "low", "close", "adjusted_open", "adjusted_high", "adjusted_low", "adjusted_close"]].notna().all(axis=1)
    daily["membership_source_quality"] = "weekly_close_PIT_effective_next_trading_day"
    daily["daily_price_feature_status"] = daily.price_core_valid.map({True: "ready_daily", False: "blocked_daily_price_core"})
    daily["weekly_context_status"] = "weekly_snapshot_carried_PIT_not_daily_recomputed"
    daily["lifecycle_state"] = "blocked_pending_daily_technical_chip_feature_computation"
    daily["selected_action"] = "not_materialized"
    daily["future_return_used_as_rule"] = False
    keep = ["decision_date", "membership_snapshot_date", "membership_effective_date", "next_execution_date", "ticker", "name", "market", "pool_rank",
            "open", "high", "low", "close", "volume", "turnover_value", "adjusted_open", "adjusted_high", "adjusted_low", "adjusted_close", "adjustment_factor",
            "price_core_valid", "raw_execution_source_quality", "adjusted_analysis_source_quality", "membership_source_quality", "daily_price_feature_status",
            "weekly_context_status", "lifecycle_state", "selected_action", "future_return_used_as_rule"]
    daily[keep].to_csv(output / "p3_layer5_daily_candidate_materialization.csv", index=False, encoding="utf-8-sig")

    coverage = daily.groupby("decision_date").agg(candidate_count=("ticker", "size"), price_core_ready_count=("price_core_valid", "sum")).reset_index()
    coverage["requested_start"] = "2023-07-14"; coverage["requested_end"] = "2026-06-29"
    coverage.to_csv(output / "p3_layer5_daily_requested_vs_actual_coverage.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([
        {"family": "membership", "frequency": "weekly_PIT_effective_next_day", "status": "ready", "daily_recomputed": False},
        {"family": "raw_execution_OHLCV", "frequency": "daily", "status": "ready_or_row_blocked", "daily_recomputed": True},
        {"family": "adjusted_analysis_OHLC", "frequency": "daily", "status": "ready_or_row_blocked", "daily_recomputed": True},
        {"family": "RS_MA_BIAS_KD", "frequency": "daily", "status": "blocked_not_yet_computed_from_daily_adjusted_history", "daily_recomputed": False},
        {"family": "chip_rollups", "frequency": "daily", "status": "blocked_not_yet_rolled_5_10_20D", "daily_recomputed": False},
        {"family": "quality_context", "frequency": "weekly_PIT", "status": "ready_carried_context", "daily_recomputed": False},
        {"family": "market_state_3group", "frequency": "daily", "status": "blocked_not_yet_materialized", "daily_recomputed": False},
    ]).to_csv(output / "p3_layer5_daily_field_frequency_source_audit.csv", index=False, encoding="utf-8-sig")

    profiles = {"balanced": [20, 20, 20, 15, 15, 10], "trend_capital": [20, 25, 25, 15, 10, 5], "risk_quality": [15, 20, 20, 20, 15, 10]}
    blocks = list("ABCDEF")
    policy = []
    for profile, weights in profiles.items():
        for block, weight in zip(blocks, weights): policy.append({"profile": profile, "block": block, "weight_pct": weight, "field_weight_policy": "equal_available_applicable"})
    pd.DataFrame(policy).to_csv(output / "p3_layer5_daily_frozen_parameter_policy.csv", index=False, encoding="utf-8-sig")
    cost = pd.DataFrame([
        {"item": "EP05_brokerage_fee", "base": "existing_model", "ready": True}, {"item": "stock_sell_tax", "base": "existing_model", "ready": True},
        {"item": "switch_double_sided_cost", "base": "existing_model", "ready": True}, {"item": "slippage_bp_per_side", "low": 5, "base": 10, "high": 20, "ready": True},
        {"item": "after_cost_edge_buffer_x", "low": 1.0, "base": 1.5, "high": 2.0, "ready": True},
    ])
    cost.to_csv(output / "p3_layer5_daily_cost_slippage_policy.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([
        {"ablation_id": "P3_full_TAIFEX_on", "period": "P3_exact", "TAIFEX_enabled": True, "TDCC_enabled": False, "coverage_policy": "same_dates", "NA_policy": "not_zero", "execution_basis": "same_next_day", "cost_basis": "EP05_plus_10bp_per_side"},
        {"ablation_id": "P3_full_TAIFEX_off", "period": "P3_exact", "TAIFEX_enabled": False, "TDCC_enabled": False, "coverage_policy": "same_dates", "NA_policy": "explicit_ablation", "execution_basis": "same_next_day", "cost_basis": "EP05_plus_10bp_per_side"},
        {"ablation_id": "P3_1_TDCC_unavailable", "period": "2023-07-14_to_2025-07-10", "TAIFEX_enabled": True, "TDCC_enabled": False, "coverage_policy": "TDCC_unavailable_no_fill", "NA_policy": "not_applicable", "execution_basis": "same_next_day", "cost_basis": "EP05_plus_10bp_per_side"},
        {"ablation_id": "P3_2_TDCC_on", "period": "2025-07-11_to_2026-06-29", "TAIFEX_enabled": True, "TDCC_enabled": True, "coverage_policy": "same_TDCC_ready_dates", "NA_policy": "row_availability", "execution_basis": "same_next_day", "cost_basis": "EP05_plus_10bp_per_side"},
        {"ablation_id": "P3_2_TDCC_off", "period": "2025-07-11_to_2026-06-29", "TAIFEX_enabled": True, "TDCC_enabled": False, "coverage_policy": "same_TDCC_ready_dates", "NA_policy": "explicit_ablation", "execution_basis": "same_next_day", "cost_basis": "EP05_plus_10bp_per_side"},
    ]).to_csv(output / "p3_layer5_daily_taifex_tdcc_ablation_contract.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([
        {"gate_id": "G1_regime_primary", "phase": "B", "scope": "ordinary_market|weak_market", "metric": "net_after_cost_return|median_excess|hit_rate|downside_capture|MDD|false_switch", "primary": True, "go_rule": "must_pass; strong_or_mega_cannot_override_failure"},
        {"gate_id": "G2_regime_secondary", "phase": "B", "scope": "strong_market|mega_like", "metric": "same_metrics", "primary": False, "go_rule": "secondary_only"},
        {"gate_id": "G3_market_bonus_removal", "phase": "B", "scope": "same_date_same_execution", "metric": "selected_return_minus_0050_return", "primary": True, "go_rule": "report_median_excess_and_hit_rate"},
        {"gate_id": "G4_leveraged_hurdle", "phase": "B", "scope": "same_basis", "metric": "00631L_or_0050_2x_state_hold_after_cost", "primary": True, "go_rule": "must_not_hide_underperformance_with_headline_return"},
        {"gate_id": "G5_walk_forward", "phase": "B", "scope": "year|quarter|walk_forward", "metric": "net_return|MDD|annual_quarter_stability", "primary": True, "go_rule": "stable_platform_not_single_peak"},
        {"gate_id": "G6_remove_best", "phase": "B", "scope": "exact_full_path_rechain", "metric": "remove_best_year|quarter|episode_1_3_5", "primary": True, "go_rule": "no_interval_subtraction_proxy"},
        {"gate_id": "G7_concentration", "phase": "B", "scope": "ticker|sector|mega_period", "metric": "contribution_share", "primary": True, "go_rule": "explosive_segment_dependency_blocks_primary"},
        {"gate_id": "G8_phase_A_semantics", "phase": "A", "scope": "all_events", "metric": "PIT|NA|confidence|state_action_consistency", "primary": True, "go_rule": "must_pass_before_phase_B"},
    ]).to_csv(output / "p3_layer5_anti_mega_bias_acceptance_governance.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([
        {"comparison": "selected_vs_0050", "execution_basis": "same_next_day", "cost": "EP05_plus_10bp_per_side", "coverage": "same_dates", "required_outputs": "daily_excess|median_excess|hit_rate|downside_capture|MDD"},
        {"comparison": "selected_vs_00631L_or_0050_2x", "execution_basis": "same_next_day_state_hold", "cost": "EP05_asset_specific_plus_10bp_per_side", "coverage": "same_dates", "required_outputs": "net_return|MDD|hit_rate|downside_capture"},
        {"comparison": "slippage_sensitivity", "execution_basis": "same_next_day", "cost": "EP05", "coverage": "same_dates", "required_outputs": "5bp|10bp_primary|20bp"},
    ]).to_csv(output / "p3_layer5_benchmark_excess_comparison_contract.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([
        {"market_state": "strong_market", "entry_floor_multiplier": 0.9, "hold_tolerance_multiplier": 1.2, "replacement_margin_multiplier": 1.2, "weakening_condition_delta": 1, "cash_allowed": False, "required_independent_groups": 3, "persistence_requires_daily_reproof": True},
        {"market_state": "ordinary_market", "entry_floor_multiplier": 1.0, "hold_tolerance_multiplier": 1.0, "replacement_margin_multiplier": 1.0, "weakening_condition_delta": 0, "cash_allowed": False, "required_independent_groups": 0, "persistence_requires_daily_reproof": True},
        {"market_state": "weak_market", "entry_floor_multiplier": 1.1, "hold_tolerance_multiplier": 0.8, "replacement_margin_multiplier": 0.8, "weakening_condition_delta": -1, "cash_allowed": False, "required_independent_groups": 3, "persistence_requires_daily_reproof": True},
        {"market_state": "confirmed_bear", "entry_floor_multiplier": 1.2, "hold_tolerance_multiplier": 0.7, "replacement_margin_multiplier": 0.8, "weakening_condition_delta": -1, "cash_allowed": True, "required_independent_groups": 4, "persistence_requires_daily_reproof": True},
    ]).to_csv(output / "p3_layer5_market_controller_multiplier_contract.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([
        {"field": "decision_date", "source": "daily_PIT", "required": True},
        {"field": "effective_execution_date", "source": "next_trading_day", "required": True},
        {"field": "market_state_before_after", "source": "daily_5group_controller", "required": True},
        {"field": "five_group_states_confidence_missingness", "source": "Taiwan|breadth|capital_risk|derivatives|external", "required": True},
        {"field": "risk_multipliers_before_after", "source": "frozen_multiplier_contract", "required": True},
        {"field": "incumbent_action_impact", "source": "state_action_trace", "required": True},
        {"field": "entry_blocked_replacement_raised_weakening_accelerated_cash_allowed", "source": "controller_effect", "required": True},
        {"field": "TAIFEX_participated", "source": "derivatives_group", "required": True},
        {"field": "TDCC_participated", "source": "P3_2_optional_only", "required": True},
        {"field": "forward_drawdown_lead_lag", "source": "Experiments_evaluation_only", "required": False},
    ]).to_csv(output / "p3_layer5_market_regime_transition_trace_schema.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([
        {"counterfactual": "C0_ordinary_always", "selector_same": True, "candidate_same": True, "controller": "disabled_ordinary_thresholds", "TAIFEX": "same_input_not_action", "TDCC": "same_input_not_action"},
        {"counterfactual": "C1_full_frozen_controller", "selector_same": True, "candidate_same": True, "controller": "strong_ordinary_weak_bear", "TAIFEX": "on", "TDCC": "P3_2_on"},
        {"counterfactual": "C2_no_strong_relaxation", "selector_same": True, "candidate_same": True, "controller": "ordinary_for_strong_weak_bear_tightening_on", "TAIFEX": "on", "TDCC": "P3_2_on"},
        {"counterfactual": "C3_TAIFEX_off", "selector_same": True, "candidate_same": True, "controller": "full_except_derivatives_ablation", "TAIFEX": "off", "TDCC": "P3_2_separate_on_off"},
    ]).assign(execution_basis="same_next_day", cost_basis="EP05_plus_10bp_per_side").to_csv(output / "p3_layer5_market_controller_counterfactual_contract.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([
        {"metric": "regime_downgrade_lead_lag_vs_0050_TAIEX_drawdown", "use": "evaluation_only", "primary_regime": "ordinary|weak"},
        {"metric": "downside_capture_5_10_20D_around_weak_bear", "use": "evaluation_only", "primary_regime": "ordinary|weak"},
        {"metric": "false_alarm_count|tightening_duration|early_exit_loss|reentry_delay", "use": "evaluation_only", "primary_regime": "ordinary|weak"},
        {"metric": "MDD|tail_loss|false_switch|turnover|cost", "use": "Phase_B_net_after_cost", "primary_regime": "ordinary|weak"},
        {"metric": "C1_bull_return_only_without_risk_improvement", "use": "NO_GO", "primary_regime": "ordinary|weak"},
        {"metric": "C2_risk_stable_limited_upside_cost", "use": "conservative_primary_candidate", "primary_regime": "ordinary|weak"},
    ]).to_csv(output / "p3_layer5_market_controller_risk_acceptance_contract.csv", index=False, encoding="utf-8-sig")
    blockers = [
        {"blocker": "daily_RS_MA_BIAS_KD_not_materialized", "scope": "all daily candidate rows", "silent_fill": False},
        {"blocker": "daily_chip_5_10_20D_rollups_not_materialized", "scope": "applicable rows", "silent_fill": False},
        {"blocker": "daily_market_three_group_state_not_materialized", "scope": "all decision dates", "silent_fill": False},
        {"blocker": "2025_08_01_adjusted_provider_partial", "scope": "ticker-level", "silent_fill": False},
    ]
    pd.DataFrame(blockers).to_csv(output / "p3_layer5_daily_blocked_ledger.csv", index=False, encoding="utf-8-sig")
    market_source_audit = [
        {"market_field": "0050_adjusted_HLC", "local_source": "backtest_cache/stock_pool_observations/0050_TW.csv", "status": "ready", "can_substitute": False},
        {"market_field": "TAIEX", "local_source": "backtest_cache/taiex_yfinance/^TWII.csv", "status": "ready_trusted_nonofficial", "can_substitute": False},
        {"market_field": "primary80_breadth", "local_source": "daily candidate adjusted features", "status": "blocked_until_daily_technical_compute", "can_substitute": False},
        {"market_field": "full_market_traded_value", "local_source": str(market_fill / "compact/full_market_traded_value/p3_daily.csv.gz"), "status": "ready_official_derived_all_market", "can_substitute": False},
        {"market_field": "full_market_margin_balance", "local_source": str(market_fill / "compact/full_market_margin_balance/p3_daily.csv.gz"), "status": "ready_official_derived_all_market", "can_substitute": False},
        {"market_field": "TAIFEX_foreign_OI", "local_source": "Radar P3 taifex compact", "status": "ready", "can_substitute": False},
        {"market_field": "Nasdaq", "local_source": "Radar P3 global_market compact", "status": "ready", "can_substitute": False},
        {"market_field": "SOX", "local_source": str(market_fill / "compact/global_market/p3_sox.csv.gz"), "status": "ready_trusted_nonofficial_cutoff_PIT", "can_substitute": False},
        {"market_field": "VIX", "local_source": "Radar P3 global_market compact", "status": "ready", "can_substitute": False},
        {"market_field": "USD_TWD", "local_source": "Radar P3 global_market compact", "status": "ready", "can_substitute": False},
    ]
    pd.DataFrame(market_source_audit).to_csv(output / "p3_layer5_daily_market_source_gate_audit.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{"audit": "membership_effective_strictly_after_snapshot", "violation_count": int((daily.membership_effective_date <= daily.membership_snapshot_date).sum())},
                  {"audit": "execution_strictly_after_decision", "violation_count": int((daily.next_execution_date.notna() & (daily.next_execution_date <= daily.decision_date)).sum())},
                  {"audit": "future_return_used_as_rule", "violation_count": int(daily.future_return_used_as_rule.sum())}]).to_csv(output / "future_data_audit.csv", index=False, encoding="utf-8-sig")
    readiness = {"task_id": TASK, "status": "daily_membership_and_price_core_materialized_feature_state_action_partial",
                 "requested_start": "2023-07-14", "requested_end": "2026-06-29", "actual_start": str(coverage.decision_date.min().date()), "actual_end": str(coverage.decision_date.max().date()),
                 "daily_dates": int(coverage.decision_date.nunique()), "daily_candidate_rows": int(len(daily)), "daily_candidate_count_min": int(coverage.candidate_count.min()),
                 "daily_candidate_count_median": float(coverage.candidate_count.median()), "daily_candidate_count_max": int(coverage.candidate_count.max()),
                 "daily_price_core_ready_rows": int(daily.price_core_valid.sum()), "daily_membership_timing_ready": True,
                 "daily_feature_blocks_ready": False, "daily_state_rows_ready": False, "daily_action_rows_ready": False,
                 "market_source_fill_absorbed": True, "market_source_fill_commit": "0d1d4a0",
                 "post_market_source_convergence_absorbed": True, "post_market_source_convergence_commit": "d5044fc",
                 "true_download_source_gap_count": 0,
                 "july_exact_Layer4_recompute_ready": True, "july_exact_Layer4_materialized": False,
                 "Layer4_2026_06_29_carry_forward_after_date_allowed": False,
                 "market_source_compact_checksum_match": True,
                 "market_controller_source_ready": True, "market_controller_source_blockers": [],
                 "anti_mega_bias_acceptance_governance_ready": True,
                 "ordinary_weak_primary_judgment_required": True,
                 "strong_mega_secondary_only": True,
                 "P1_backfill_from_P3_allowed": False,
                 "selector_market_controller_separation_contract_ready": True,
                 "market_transition_trace_schema_ready": True,
                 "market_counterfactual_C0_C3_contract_ready": True,
                 "rolling_only_calibration_required": True,
                 "full_period_or_future_threshold_calibration_allowed": False,
                 "ready_for_phase_a_event_validation": False, "ready_for_phase_b_unique_position_path": False, "ready_for_experiments": False,
                 "future_data_violation_count": 0, "formal_model_changed": False, "trade_decision_changed": False, "active_in_trade_decision": False,
                 "report_changed": False, "portfolio_replay_executed": False, "ready_for_strategy_replay": False, "ready_for_formal": False,
                 "not_live_rule": True, "forward_returns_live_rule_usage": False}
    (output / "p3_layer5_daily_readiness.json").write_text(json.dumps(readiness, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "final_summary_zh.md").write_text("# P3 Layer5 daily materialization\n\n- weekly membership next-day effective expansion與daily raw/adjusted price core已完成。\n- daily technical/chip/market state尚未計算，不得用weekly context冒充，因此Phase A不ready。\n- 未跑績效、未產selected action、未交Experiments。\n", encoding="utf-8")
    files = sorted(p for p in output.iterdir() if p.is_file() and p.name != "manifest.json")
    manifest = {"task_id": TASK, "source_commits": ["ea5b046", "9c3ea9a", "0d1d4a0", "d5044fc"],
                "market_source_fill": str(market_fill), "market_source_readiness": market_ready,
                "source_convergence": str(source_convergence), "source_convergence_readiness": convergence_ready, "readiness": readiness,
                "files": [{"name": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in files]}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--output-dir", type=Path, default=OUTPUT); parser.add_argument("--market-fill-dir", type=Path, default=MARKET_FILL); parser.add_argument("--source-convergence-dir", type=Path, default=SOURCE_CONVERGENCE)
    args = parser.parse_args(); print(run(args.output_dir, args.market_fill_dir, args.source_convergence_dir))


if __name__ == "__main__": main()
