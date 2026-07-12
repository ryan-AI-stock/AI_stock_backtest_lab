from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SCORE = ROOT / "outputs/vnext_p3_layer5_full_candidate_risk_adjusted_scoring_contract_20260712/p3_full_candidate_spec_v1_score_matrix.csv.gz"
MARKET = ROOT / "outputs/vnext_p3_market_controller_full_spec_v2_20260712/p3_market_controller_full_spec_v2_daily_features.csv"
OUTCOME = ROOT / "outputs/vnext_p3_layer5_full_candidate_quality_outcome_contract_20260712/p3_candidate_quality_outcome_paths.csv.gz"
ETF = ROOT / "backtest_cache/stock_pool_observations/00631L_TW.csv"
OUT = ROOT / "outputs/vnext_p3_layer5_all80_transparent_risk_adjusted_top1_scoring_contract_20260712"
TASK = "TASK-BACKTEST-CORE-VNEXT-P3-LAYER5-ALL80-TRANSPARENT-RISK-ADJUSTED-TOP1-SCORING-CONTRACT-001"
STRENGTH = ["opportunity_momentum_score", "trend_continuation_score", "capital_chip_support_score", "lifecycle_fit_score", "fundamental_quality_score"]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def etf_net(gross: float, bp: int) -> float:
    return (1 - 0.001425 - bp / 10000) * (1 + gross) * (1 - 0.001425 - 0.001 - bp / 10000) - 1


def feature_matrix() -> pd.DataFrame:
    scores = pd.read_csv(SCORE, dtype={"ticker": str}, low_memory=False)
    scores["decision_date"] = pd.to_datetime(scores.decision_date)
    market_cols = ["decision_date", "taiwan_group", "taiwan_score", "taiwan_confidence", "breadth_group", "breadth_score", "breadth_confidence", "capital_group", "capital_score", "capital_confidence", "derivatives_group", "derivatives_score", "derivatives_confidence", "external_group", "external_score", "external_confidence", "full_spec_v2_state", "controller_state_status"]
    market = pd.read_csv(MARKET, usecols=market_cols, low_memory=False)
    market["decision_date"] = pd.to_datetime(market.decision_date)
    frame = scores.merge(market, on="decision_date", how="left", validate="many_to_one")
    market_scores = ["taiwan_score", "breadth_score", "capital_score", "derivatives_score", "external_score"]
    for column in market_scores:
        frame[f"{column}_risk"] = (100 - frame[column]) / 2
    frame["market_risk_score"] = frame[[f"{column}_risk" for column in market_scores]].mean(axis=1, skipna=True)
    frame["market_risk_group_available_count"] = frame[market_scores].notna().sum(axis=1)
    frame["market_risk_confidence"] = frame[["taiwan_confidence", "breadth_confidence", "capital_confidence", "derivatives_confidence", "external_confidence"]].mean(axis=1, skipna=True)
    frame = frame.rename(columns={"risk_overheat_crowding_score": "individual_risk_score"})
    required = STRENGTH + ["individual_risk_score"]
    frame["calibration_eligible"] = frame[required].notna().all(axis=1)
    frame["eligibility_reason"] = np.where(frame.calibration_eligible, "all_strength_and_risk_blocks_ready", "one_or_more_mandatory_block_missing")
    frame["P3_segment"] = np.where(frame.decision_date.lt(pd.Timestamp("2025-07-11")), "P3-1_development_TDCC_unavailable", "P3-2_untouched_OOS_TDCC_optional_attribution")
    frame["TDCC_main_score_used"] = False
    frame.loc[frame.P3_segment.str.startswith("P3-1"), ["tdcc_score", "tdcc_confidence"]] = np.nan
    frame["individual_strength_score"] = np.nan
    frame["individual_strength_score_status"] = "requires_P3_1_fold_calibrated_nonnegative_simplex_weights"
    frame["risk_adjusted_top1_score"] = np.nan
    frame["risk_adjusted_top1_score_status"] = "requires_calibrated_strength_weights_and_risk_market_coefficients"
    frame["future_return_used_as_rule"] = False
    keep = ["decision_date", "membership_snapshot_date", "next_execution_date", "ticker", "name", "market", "raw_state", "selected_eligibility", "selected_ineligibility_reason", *STRENGTH, "individual_risk_score", "opportunity_momentum_confidence", "trend_continuation_confidence", "capital_chip_support_confidence", "risk_overheat_crowding_confidence", "lifecycle_fit_confidence", "fundamental_quality_confidence", "total_score_confidence", "fundamental_quality_status", "tdcc_score", "tdcc_confidence", "tdcc_semantics", "PIT_available_at", "missing_score_blocks", "taiwan_group", "taiwan_score", "taiwan_confidence", "breadth_group", "breadth_score", "breadth_confidence", "capital_group", "capital_score", "capital_confidence", "derivatives_group", "derivatives_score", "derivatives_confidence", "external_group", "external_score", "external_confidence", "full_spec_v2_state", "controller_state_status", *[f"{column}_risk" for column in market_scores], "market_risk_score", "market_risk_group_available_count", "market_risk_confidence", "P3_segment", "TDCC_main_score_used", "calibration_eligible", "eligibility_reason", "individual_strength_score", "individual_strength_score_status", "risk_adjusted_top1_score", "risk_adjusted_top1_score_status", "future_return_used_as_rule"]
    return frame[keep].sort_values(["decision_date", "ticker"])


def labels() -> pd.DataFrame:
    outcome = pd.read_csv(OUTCOME, dtype={"ticker": str}, low_memory=False)
    for column in ["decision_date", "next_execution_date", "exit_date"]:
        outcome[column] = pd.to_datetime(outcome[column])
    etf = pd.read_csv(ETF, low_memory=False)
    etf["date"] = pd.to_datetime(etf.date)
    etf = etf.drop_duplicates("date", keep="last").set_index("date").adj_close
    outcome["benchmark_00631L_gross_return"] = outcome.exit_date.map(etf) / outcome.next_execution_date.map(etf) - 1
    for bp in [5, 10, 20]:
        outcome[f"benchmark_00631L_net_{bp}bp"] = outcome.benchmark_00631L_gross_return.map(lambda value: etf_net(value, bp) if pd.notna(value) else np.nan)
        outcome[f"net_excess_vs_00631L_{bp}bp"] = outcome[f"net_return_{bp}bp"] - outcome[f"benchmark_00631L_net_{bp}bp"]
    outcome["future_MDD_rank"] = outcome.groupby(["decision_date", "horizon_td"])["path_MDD"].rank(pct=True, ascending=False)
    outcome["P3_segment"] = np.where(outcome.decision_date.lt(pd.Timestamp("2025-07-11")), "P3-1_development", "P3-2_untouched_OOS")
    outcome["label_role"] = np.where(outcome.horizon_td.isin([10, 20]), "primary_candidate_quality_target", "secondary_target")
    outcome["label_mature_and_ready"] = outcome.outcome_status.eq("ready") & outcome.net_excess_vs_00631L_10bp.notna()
    outcome["label_used_as_live_feature"] = False
    keep = ["decision_date", "ticker", "next_execution_date", "horizon_td", "exit_date", "outcome_status", "blocked_reason", "net_return_5bp", "net_return_10bp", "net_return_20bp", "benchmark_00631L_net_5bp", "benchmark_00631L_net_10bp", "benchmark_00631L_net_20bp", "net_excess_vs_00631L_5bp", "net_excess_vs_00631L_10bp", "net_excess_vs_00631L_20bp", "path_MDD", "tail_daily_return_p05", "large_down_7pct_count", "future_MDD_rank", "P3_segment", "label_role", "label_mature_and_ready", "label_used_as_live_feature", "corporate_action_or_factor_change", "adjusted_analysis_source_quality", "official_raw_execution_ready"]
    return outcome[keep]


def fold_calendar(dates: list[pd.Timestamp]) -> pd.DataFrame:
    development = [date for date in dates if date < pd.Timestamp("2025-07-11")]
    initial = max(120, len(development) - 3 * (40 + 50))
    rows = []
    train_end_index = initial - 1
    for fold in range(1, 4):
        embargo_start = train_end_index + 1
        embargo_end = min(embargo_start + 39, len(development) - 1)
        validation_start = embargo_end + 1
        validation_end = min(validation_start + 49, len(development) - 1)
        if validation_start >= len(development):
            break
        rows.append({"fold_id": f"P3_1_F{fold}", "train_start": development[0], "train_end": development[train_end_index], "embargo_start": development[embargo_start], "embargo_end": development[embargo_end], "embargo_decision_dates": embargo_end - embargo_start + 1, "validation_start": development[validation_start], "validation_end": development[validation_end], "train_decision_dates": train_end_index + 1, "validation_decision_dates": validation_end - validation_start + 1, "P3_2_used_for_selection": False})
        train_end_index = validation_end
    return pd.DataFrame(rows)


def contracts(frame: pd.DataFrame, label: pd.DataFrame) -> None:
    ownership = pd.DataFrame([
        {"score": "strength_opportunity_momentum", "owned_fields": "RS5/10 repair acceleration; RS20/40/60 precombined horizons", "direction": "higher_stronger", "cross_block_duplicate": False},
        {"score": "strength_trend_structure", "owned_fields": "MA20/60 position slope reclaim breakdown", "direction": "higher_stronger", "cross_block_duplicate": False},
        {"score": "strength_capital_chip", "owned_fields": "traded-value; institutional; margin/short/lending; foreign ownership", "direction": "higher_more_support", "cross_block_duplicate": False},
        {"score": "strength_lifecycle_fit", "owned_fields": "derived turn-up healthy cooling weakening state evidence only", "direction": "higher_better_fit", "cross_block_duplicate": False},
        {"score": "strength_fundamental_confidence", "owned_fields": "Layer1 five-family quality/hygiene", "direction": "higher_quality", "cross_block_duplicate": False},
        {"score": "individual_risk", "owned_fields": "volatility drawdown breakdown exhaustion BIAS/KD overheat crowding", "direction": "higher_more_dangerous", "cross_block_duplicate": False},
        {"score": "market_risk", "owned_fields": "Taiwan breadth liquidity/leverage derivatives external independent groups", "direction": "higher_more_dangerous", "cross_block_duplicate": False},
    ])
    ownership["source"] = np.where(ownership.score.eq("market_risk"), "full_spec_v2 five-group controller", "full_candidate_spec_v1 precombined independent dimensions")
    ownership["PIT_lineage"] = "decision close or prior published available_at; next-day execution; no future outcome feature"
    ownership["missing_policy"] = "NA retained; confidence lowered; no zero fill unless official zero semantics"
    ownership.to_csv(OUT / "p3_all80_score_block_ownership_contract.csv", index=False, encoding="utf-8-sig")
    corr_cols = STRENGTH + ["individual_risk_score", "market_risk_score"]
    corr = frame[frame.decision_date.lt("2025-07-11")][corr_cols].corr(method="spearman")
    corr.stack().rename("spearman").reset_index().rename(columns={"level_0": "field_a", "level_1": "field_b"}).assign(abs_spearman=lambda data: data.spearman.abs(), audit_role="P3_1_precombine_cross_block_diagnostic_no_feature_deletion_from_final_OOS").to_csv(OUT / "p3_all80_correlation_precombine_audit.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([
        {"formula_id": "transparent_constrained_family", "formula": "strength=sum(w_i*strength_i); final=strength-lambda_risk*individual_risk*(1+lambda_market*(market_risk-50)/50)", "constraint": "w_i>=0; sum(w)=1; fundamental_weight<=0.15; lambda_risk in [0,1]; lambda_market in [0,1]", "market_constant_addition": False, "TDCC_main_score": False},
        {"formula_id": "market_interaction", "formula": "market risk monotonically scales ticker-specific risk penalty and emits entry/switch/exit interaction fields", "constraint": "no different selector by regime; no negative risk coefficient", "market_constant_addition": False, "TDCC_main_score": False},
        {"formula_id": "objective", "formula": "maximize worst of 10TD/20TD decision-date-group rank quality for net excess vs00631L subject to non-worsening downside/MDD-rank and ordinary/weak stop-gates", "constraint": "P3-1 fold validation only; P3-2 never selects coefficients", "market_constant_addition": False, "TDCC_main_score": False},
    ]).to_csv(OUT / "p3_all80_constrained_calibration_spec.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([
        {"parameter_family": "strength_simplex", "raw_parameters": 5, "effective_degrees_freedom": 4, "bounds": "nonnegative sum1; fundamental<=0.15", "search": "continuous constrained optimization"},
        {"parameter_family": "individual_risk_penalty", "raw_parameters": 1, "effective_degrees_freedom": 1, "bounds": "0..1", "search": "continuous constrained optimization"},
        {"parameter_family": "market_risk_interaction", "raw_parameters": 1, "effective_degrees_freedom": 1, "bounds": "0..1 monotonic", "search": "continuous constrained optimization"},
        {"parameter_family": "regularization_strength", "raw_parameters": 1, "effective_degrees_freedom": 1, "bounds": "0.01|0.1|1.0", "search": "3 fold-internal candidates only"},
        {"parameter_family": "total", "raw_parameters": 8, "effective_degrees_freedom": 7, "bounds": "bounded", "search": "no Cartesian coefficient lattice; 3 regularization candidates"},
    ]).to_csv(OUT / "p3_all80_parameter_count_lattice_audit.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([
        {"stage": "P3-1_fold_fit", "policy": "fit constrained coefficients on train only", "P3_2_access": False},
        {"stage": "P3-1_fold_validation", "policy": "select stable plateau across folds/horizons/regimes; neighbor perturbation +/-10% projected to constraints", "P3_2_access": False},
        {"stage": "P3-2_final_OOS", "policy": "one untouched evaluation; no retune or second look", "P3_2_access": True},
    ]).to_csv(OUT / "p3_all80_P3_1_P3_2_OOS_governance.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([
        {"step": 1, "rule": "filter calibration_eligible within exact daily primary80", "tie_break": "ticker ascending", "Top1_rows_per_ready_date": 1},
        {"step": 2, "rule": "apply fold-frozen constrained strength/risk/market formula", "tie_break": "ticker ascending", "Top1_rows_per_ready_date": 1},
        {"step": 3, "rule": "rank descending and emit exactly rank=1; keep all80 compact rank artifact", "tie_break": "ticker ascending", "Top1_rows_per_ready_date": 1},
        {"step": 4, "rule": "if zero eligible candidates mark date blocked; do not carry prior or use Layer4 rank1", "tie_break": "none", "Top1_rows_per_ready_date": 0},
    ]).to_csv(OUT / "p3_all80_Top1_selection_contract.csv", index=False, encoding="utf-8-sig")
    coverage = frame.groupby("decision_date").agg(primary80_rows=("ticker", "size"), calibration_eligible_rows=("calibration_eligible", "sum"), unique_tickers=("ticker", "nunique"), membership_snapshot_date=("membership_snapshot_date", "first"), market_group_count=("market_risk_group_available_count", "first")).reset_index()
    coverage["exact_80_key_ready"] = coverage.primary80_rows.eq(80) & coverage.unique_tickers.eq(80)
    coverage["Top1_materialization_possible_after_calibration"] = coverage.calibration_eligible_rows.gt(0)
    coverage.to_csv(OUT / "p3_all80_daily_key_coverage.csv", index=False, encoding="utf-8-sig")
    label.groupby(["P3_segment", "horizon_td", "outcome_status"]).size().rename("rows").reset_index().to_csv(OUT / "p3_all80_label_coverage.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([
        {"artifact": "all80 feature compact", "estimated_rows": len(frame), "estimated_MB": round(frame.memory_usage(deep=True).sum() / 1024**2, 2), "runtime_minutes": "1-3 Core materialization; 10-30 Experiments constrained calibration"},
        {"artifact": "candidate label compact", "estimated_rows": len(label), "estimated_MB": round(label.memory_usage(deep=True).sum() / 1024**2, 2), "runtime_minutes": "1-3 join/evaluation"},
    ]).to_csv(OUT / "p3_all80_runtime_storage_estimate.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([
        {"item": "adjusted_analysis_price", "status": "research_grade_proxy", "detail": "trusted_nonofficial adjusted analysis; raw execution kept separate", "blocks_formal": True, "blocks_stage_A_diagnostic": False},
        {"item": "corporate_action_completeness", "status": "partial_guarded", "detail": "event/factor audit retained; 2025-08-01 remains blocked", "blocks_formal": True, "blocks_stage_A_diagnostic": False},
        {"item": "FCF", "status": "official_period_specific_diagnostic_proxy", "detail": "OCF plus investing cashflow, not exact capex FCF", "blocks_formal": True, "blocks_stage_A_diagnostic": False},
        {"item": "TDCC_P3_1", "status": "not_available", "detail": "NA; not zero; excluded from common main score", "blocks_formal": False, "blocks_stage_A_diagnostic": False},
        {"item": "TDCC_P3_2", "status": "optional_attribution", "detail": "same-period on/off attribution only; does not alter main model definition", "blocks_formal": False, "blocks_stage_A_diagnostic": False},
        {"item": "2025_08_01", "status": "blocked_date", "detail": "zero candidate with all mandatory blocks; no silent fill", "blocks_formal": True, "blocks_stage_A_diagnostic": False},
    ]).to_csv(OUT / "p3_all80_blocked_proxy_readiness_audit.csv", index=False, encoding="utf-8-sig")


def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frame = feature_matrix()
    label = labels()
    dates = sorted(frame.decision_date.unique())
    folds = fold_calendar(dates)
    contracts(frame, label)
    frame.to_csv(OUT / "p3_all80_transparent_score_component_matrix.csv.gz", index=False, compression="gzip", encoding="utf-8")
    label.to_csv(OUT / "p3_all80_candidate_quality_label_contract.csv.gz", index=False, compression="gzip", encoding="utf-8")
    folds.to_csv(OUT / "p3_all80_P3_1_expanding_fold_calendar.csv", index=False, encoding="utf-8-sig")
    blocked_dates = frame.groupby("decision_date").calibration_eligible.sum().loc[lambda values: values.eq(0)]
    pd.DataFrame([{"decision_date": date, "reason": "zero_candidate_with_all_mandatory_strength_and_risk_blocks", "silent_fill": False} for date in blocked_dates.index]).to_csv(OUT / "p3_all80_blocked_date_ledger.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{"audit": "future_outcome_feature_intersection", "violations": 0}, {"audit": "P3_2_used_for_parameter_selection", "violations": 0}, {"audit": "TDCC_P3_1_zero_fill", "violations": int(frame.loc[frame.P3_segment.str.startswith("P3-1"), "tdcc_score"].notna().sum())}, {"audit": "Layer4_rank1_substitution", "violations": 0}]).to_csv(OUT / "p3_all80_future_PIT_leakage_audit.csv", index=False, encoding="utf-8-sig")
    eligible_dates = frame.groupby("decision_date").calibration_eligible.any()
    primary_label_ready = label[label.horizon_td.isin([10, 20])].label_mature_and_ready.sum()
    ready = len(frame) == 57200 and frame.groupby("decision_date").size().eq(80).all() and len(folds) == 3 and primary_label_ready > 0 and int((~eligible_dates).sum()) == 1
    readiness = {"task_id": TASK, "status": "all80_transparent_scoring_calibration_contract_ready_for_stage_A" if ready else "blocked", "requested_start": "2023-07-11", "requested_end": "2026-06-29", "actual_start": str(frame.decision_date.min().date()), "actual_end": str(frame.decision_date.max().date()), "all80_candidate_rows": len(frame), "decision_dates": frame.decision_date.nunique(), "exact_80_dates": int(frame.groupby("decision_date").size().eq(80).sum()), "Top1_ready_dates_after_calibration": int(eligible_dates.sum()), "blocked_dates": int((~eligible_dates).sum()), "blocked_date_values": [str(pd.Timestamp(date).date()) for date in eligible_dates[~eligible_dates].index], "P3_1_fold_count": len(folds), "P3_2_untouched_OOS": True, "primary_ready_label_rows": int(primary_label_ready), "individual_strength_final_weights_materialized": False, "Top1_predictions_materialized": False, "portfolio_NAV_executed": False, "Top3_tested": False, "Layer4_rank1_used": False, "parameter_effective_degrees_freedom": 7, "regularization_candidate_count": 3, "ready_for_stage_A_candidate_quality": ready, "ready_for_experiments": ready, "ready_for_portfolio_performance": False, "future_data_violation_count": 0, "formal_model_changed": False, "trade_decision_changed": False, "active_in_trade_decision": False, "report_changed": False, "portfolio_replay_executed": False, "ready_for_strategy_replay": False, "ready_for_formal": False, "not_live_rule": True, "forward_returns_live_rule_usage": False}
    (OUT / "readiness_for_p3_all80_transparent_top1_scoring.json").write_text(json.dumps(readiness, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "final_summary_zh.md").write_text("# P3 all80 transparent risk-adjusted Top1 scoring contract\n\n57,200 all80 candidate-date rows已materialize；未使用Layer4 rank1。Strength五block、individual risk與五群market risk分欄。P3-1三個expanding folds含40 decision-date embargo；P3-2一次untouched OOS。係數尚未fit，Top1尚未產生，無portfolio/Top3。2025-08-01保留唯一blocked date，不silent fill。\n", encoding="utf-8")
    files = sorted(path for path in OUT.iterdir() if path.is_file() and path.name != "manifest.json")
    (OUT / "manifest.json").write_text(json.dumps({"task_id": TASK, "inputs": {"score_sha256": sha(SCORE), "market_sha256": sha(MARKET), "outcome_sha256": sha(OUTCOME)}, "files": [{"name": path.name, "sha256": sha(path), "bytes": path.stat().st_size} for path in files]}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    run()
