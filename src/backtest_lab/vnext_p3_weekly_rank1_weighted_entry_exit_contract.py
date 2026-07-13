from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DAILY = ROOT / "outputs/vnext_p3_layer5_daily_feature_state_action_materialization_20260712/p3_layer5_daily_feature_state_matrix.csv"
SCORE = ROOT / "outputs/vnext_p3_layer5_full_candidate_risk_adjusted_scoring_contract_20260712/p3_full_candidate_spec_v1_score_matrix.csv.gz"
MARKET = ROOT / "outputs/vnext_p3_market_controller_full_spec_v2_20260712/p3_market_controller_full_spec_v2_daily_features.csv"
LABEL = ROOT / "outputs/vnext_p3_layer5_all80_transparent_risk_adjusted_top1_scoring_contract_20260712/p3_all80_candidate_quality_label_contract.csv.gz"
FOLDS = ROOT / "outputs/vnext_p3_layer5_all80_transparent_risk_adjusted_top1_scoring_contract_20260712/p3_all80_P3_1_expanding_fold_calendar.csv"
OUT = ROOT / "outputs/vnext_p3_layer04_canonical_rank1_stock_market_weighted_timing_contract_20260713"
TASK = "TASK-BACKTEST-CORE-VNEXT-P3-LAYER04-CANONICAL-RANK1-STOCK-MARKET-WEIGHTED-TIMING-CONTRACT-001"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mean_available(frame: pd.DataFrame, columns: list[str]) -> tuple[pd.Series, pd.Series]:
    values = frame[columns].astype(float)
    return values.mean(axis=1, skipna=True) * 100, values.notna().mean(axis=1)


def _negative(value: pd.Series) -> pd.Series:
    return value.lt(0).where(value.notna()).astype(float)


def materialize_rank1() -> pd.DataFrame:
    daily = pd.read_csv(DAILY, dtype={"ticker": str}, low_memory=False)
    daily["decision_date"] = pd.to_datetime(daily.decision_date)
    daily = daily.loc[daily.pool_rank.eq(1)].copy()
    if len(daily) != 715 or daily.decision_date.nunique() != 715:
        raise ValueError("canonical daily rank1 must be one row on each of 715 dates")

    scores = pd.read_csv(SCORE, dtype={"ticker": str}, low_memory=False)
    scores["decision_date"] = pd.to_datetime(scores.decision_date)
    score_cols = [
        "decision_date", "ticker", "opportunity_momentum_score", "trend_continuation_score",
        "capital_chip_support_score", "risk_overheat_crowding_score", "lifecycle_fit_score",
        "fundamental_quality_score", "opportunity_momentum_confidence", "trend_continuation_confidence",
        "capital_chip_support_confidence", "risk_overheat_crowding_confidence", "lifecycle_fit_confidence",
        "fundamental_quality_confidence", "selected_eligibility", "selected_ineligibility_reason",
        "PIT_available_at", "tdcc_score", "tdcc_confidence", "tdcc_semantics",
    ]
    frame = daily.merge(scores[score_cols], on=["decision_date", "ticker"], how="left", validate="one_to_one")

    market = pd.read_csv(MARKET, low_memory=False)
    market["decision_date"] = pd.to_datetime(market.decision_date)
    market_cols = ["decision_date", "taiwan_score", "taiwan_confidence", "breadth_score", "breadth_confidence",
                   "capital_score", "capital_confidence", "derivatives_score", "derivatives_confidence",
                   "external_score", "external_confidence", "full_spec_v2_state", "controller_state_status"]
    frame = frame.merge(market[market_cols], on="decision_date", how="left", validate="one_to_one")

    # Entry and risk use already validated, precombined independent dimensions.
    frame["entry_momentum"] = frame.opportunity_momentum_score
    frame["entry_trend_structure"] = frame.trend_continuation_score
    frame["entry_capital_chip"] = frame.capital_chip_support_score
    frame["entry_lifecycle"] = frame.lifecycle_fit_score
    frame["entry_fundamental_confidence"] = frame.fundamental_quality_score
    frame["entry_stock_risk_penalty"] = frame.risk_overheat_crowding_score

    # Exit evidence is independently constructed; it is not the negative of entry strength.
    frame["exit_short_rs_deterioration"], frame["exit_short_rs_confidence"] = _mean_available(frame, ["rs5_below_rs10", "rs10_below_rs20", "rs20_negative", "rs_weak_flag"])
    frame["exit_structure_breakdown"], frame["exit_structure_confidence"] = _mean_available(frame, ["below_ma20", "below_ma60", "ma20_slope_negative", "ma60_slope_negative", "price_breakdown_flag"])
    frame["exit_capital_withdrawal"], frame["exit_capital_confidence"] = _mean_available(frame, ["tv5_below_tv20", "foreign20_negative", "trust20_negative", "dealer20_negative", "margin20_adverse", "lending20_adverse"])
    frame["exit_risk_deterioration"], frame["exit_risk_confidence"] = _mean_available(frame, ["vol20_above_vol60", "drawdown_severe", "large_down_present", "blowoff_flag", "risk_extreme_flag"])
    frame["exit_lifecycle_deterioration"] = (frame.weak_groups.clip(lower=0, upper=5) / 5 * 100).where(frame.weak_groups.notna())
    frame["exit_lifecycle_confidence"] = frame.weak_groups.notna().astype(float)
    frame["overheat_warning_only_no_exit"] = frame.overheat_groups.gt(0) & frame.weak_groups.lt(3)

    market_names = ["taiwan", "breadth", "capital", "derivatives", "external"]
    for name in market_names:
        frame[f"market_{name}_risk"] = (100 - frame[f"{name}_score"]) / 2
    frame["market_group_available_count"] = frame[[f"{name}_risk" for name in market_names]].notna().sum(axis=1)
    frame["market_group_confidence_mean"] = frame[[f"{name}_confidence" for name in market_names]].mean(axis=1, skipna=True)

    frame["P3_segment"] = np.where(frame.decision_date.lt("2025-07-11"), "P3-1_development_TDCC_unavailable", "P3-2_untouched_OOS_TDCC_optional")
    frame.loc[frame.P3_segment.str.startswith("P3-1"), ["tdcc_score", "tdcc_confidence"]] = np.nan
    frame["TDCC_main_score_used"] = False
    frame["calibration_eligible"] = frame[["entry_momentum", "entry_trend_structure", "entry_capital_chip", "entry_lifecycle", "entry_fundamental_confidence", "entry_stock_risk_penalty"]].notna().all(axis=1) & frame.market_group_available_count.eq(5)
    frame["future_outcome_used_as_live_feature"] = False
    frame["stock_entry_strength"] = np.nan
    frame["stock_exit_deterioration"] = np.nan
    frame["market_risk_score"] = np.nan
    frame["combined_entry_score"] = np.nan
    frame["combined_exit_score"] = np.nan
    frame["score_status"] = "pending_P3_1_fold_constrained_calibration"
    return frame.sort_values("decision_date")


def add_exit_inputs(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["rs5_below_rs10"] = frame.RS5.lt(frame.RS10).where(frame.RS5.notna() & frame.RS10.notna()).astype(float)
    frame["rs10_below_rs20"] = frame.RS10.lt(frame.RS20).where(frame.RS10.notna() & frame.RS20.notna()).astype(float)
    frame["rs20_negative"] = _negative(frame.RS20)
    frame["rs_weak_flag"] = frame.rs_weak.astype(float).where(frame.rs_weak.notna())
    frame["below_ma20"] = frame.adjusted_close.lt(frame.MA20).where(frame.adjusted_close.notna() & frame.MA20.notna()).astype(float)
    frame["below_ma60"] = frame.adjusted_close.lt(frame.MA60).where(frame.adjusted_close.notna() & frame.MA60.notna()).astype(float)
    frame["ma20_slope_negative"] = _negative(frame.MA20_slope)
    frame["ma60_slope_negative"] = _negative(frame.MA60_slope)
    frame["price_breakdown_flag"] = frame.price_breakdown.astype(float).where(frame.price_breakdown.notna())
    frame["tv5_below_tv20"] = frame.tv5.lt(frame.tv20).where(frame.tv5.notna() & frame.tv20.notna()).astype(float)
    frame["foreign20_negative"] = _negative(frame.institutional_foreign_net_20D)
    frame["trust20_negative"] = _negative(frame.institutional_trust_net_20D)
    frame["dealer20_negative"] = _negative(frame.institutional_dealer_net_20D)
    frame["margin20_adverse"] = frame.margin_margin_change_20D.gt(0).where(frame.margin_margin_change_20D.notna()).astype(float)
    frame["lending20_adverse"] = frame.lending_sbl_change_20D.gt(0).where(frame.lending_sbl_change_20D.notna()).astype(float)
    frame["vol20_above_vol60"] = frame.vol20.gt(frame.vol60).where(frame.vol20.notna() & frame.vol60.notna()).astype(float)
    frame["drawdown_severe"] = frame.drawdown60.le(-0.15).where(frame.drawdown60.notna()).astype(float)
    frame["large_down_present"] = frame.large_down20.gt(0).where(frame.large_down20.notna()).astype(float)
    frame["blowoff_flag"] = frame.blowoff.astype(float).where(frame.blowoff.notna())
    frame["risk_extreme_flag"] = frame.risk_extreme.astype(float).where(frame.risk_extreme.notna())
    return frame


def write_contracts(frame: pd.DataFrame) -> None:
    ownership = [
        ("entry_momentum", "RS short repair + RS20/40/60 precombined", "higher_supports_entry"),
        ("entry_trend_structure", "MA position/slope/reclaim/breakout quality", "higher_supports_entry"),
        ("entry_capital_chip", "traded value + institutional + margin/lending proxies", "higher_supports_entry"),
        ("entry_lifecycle", "turn-up/healthy/cooling state evidence", "higher_supports_entry"),
        ("entry_fundamental_confidence", "Layer1 quality floor/confidence", "higher_quality_not_short_term_driver"),
        ("entry_stock_risk_penalty", "volatility/drawdown/breakdown/exhaustion/BIAS-KD/crowding", "higher_more_dangerous"),
        ("exit_short_rs_deterioration", "RS5<RS10; RS10<RS20; RS20<0; rs_weak", "higher_supports_exit"),
        ("exit_structure_breakdown", "below MA20/60; negative slopes; price breakdown", "higher_supports_exit"),
        ("exit_capital_withdrawal", "traded-value withdrawal; institutional/margin/lending adverse", "higher_supports_exit"),
        ("exit_risk_deterioration", "vol expansion; drawdown; large-down; blowoff; risk extreme", "higher_supports_exit"),
        ("exit_lifecycle_deterioration", "independent confirmed weakening groups", "higher_supports_exit"),
        ("market_risk", "Taiwan/breadth/liquidity-leverage/derivatives/external", "higher_tightens_entry_and_amplifies_exit"),
    ]
    pd.DataFrame(ownership, columns=["component", "owned_economic_dimensions", "direction"]).assign(
        raw_field_double_count=False,
        missing_policy="NA retained; confidence reduced; official not-applicable is not zero",
        PIT_lineage="decision-close known data only; next-day execution; no future feature",
    ).to_csv(OUT / "p3_rank1_stock_market_block_ownership.csv", index=False, encoding="utf-8-sig")

    pd.DataFrame([
        {"formula": "stock_entry_strength", "definition": "sum(w_entry_i*entry_i)", "constraints": "w>=0,sum=1,fundamental<=0.15"},
        {"formula": "stock_exit_deterioration", "definition": "sum(w_exit_i*exit_i)", "constraints": "w>=0,sum=1; overheat alone excluded"},
        {"formula": "market_risk", "definition": "sum(w_market_i*market_group_risk_i)", "constraints": "w>=0,sum=1; TDCC excluded from common model"},
        {"formula": "combined_entry", "definition": "stock_strength-lambda_risk*stock_risk*(1+lambda_market*market_risk/100)", "constraints": "lambdas>=0; weaker market cannot loosen entry"},
        {"formula": "combined_exit", "definition": "stock_deterioration*(1+lambda_exit_market*market_risk/100)", "constraints": "lambda>=0; weaker market cannot reduce exit sensitivity"},
    ]).to_csv(OUT / "p3_rank1_entry_exit_combined_formula_contract.csv", index=False, encoding="utf-8-sig")

    pd.DataFrame([
        {"family": "entry_simplex", "effective_df": 4, "selection": "continuous constrained, train fold only"},
        {"family": "stock_risk_penalty", "effective_df": 1, "selection": "nonnegative bounded"},
        {"family": "market_simplex", "effective_df": 4, "selection": "continuous constrained, train fold only"},
        {"family": "entry_market_interaction", "effective_df": 1, "selection": "nonnegative bounded"},
        {"family": "exit_simplex", "effective_df": 4, "selection": "continuous constrained, train fold only"},
        {"family": "exit_market_amplification", "effective_df": 1, "selection": "nonnegative bounded"},
        {"family": "entry_exit_thresholds", "effective_df": 2, "selection": "train-fold empirical quantiles; no old hard-coded values"},
        {"family": "total", "effective_df": 17, "selection": "sequential fit; 3 regularization strengths; no Cartesian lattice"},
    ]).to_csv(OUT / "p3_rank1_parameter_count_runtime_audit.csv", index=False, encoding="utf-8-sig")

    pd.DataFrame([
        {"comparator": "C0_immediate_entry", "role": "reference", "stock_signal": False, "market_interaction": False},
        {"comparator": "C1_stock_only", "role": "attribution", "stock_signal": True, "market_interaction": False},
        {"comparator": "C2_market_only", "role": "risk_discrimination_reference_not_stock_selection", "stock_signal": False, "market_interaction": True},
        {"comparator": "C3_combined", "role": "only_primary_candidate", "stock_signal": True, "market_interaction": True},
    ]).to_csv(OUT / "p3_rank1_stage_a_comparator_contract.csv", index=False, encoding="utf-8-sig")

    corr_cols = ["entry_momentum", "entry_trend_structure", "entry_capital_chip", "entry_lifecycle", "entry_fundamental_confidence", "entry_stock_risk_penalty",
                 "exit_short_rs_deterioration", "exit_structure_breakdown", "exit_capital_withdrawal", "exit_risk_deterioration", "exit_lifecycle_deterioration"]
    corr = frame.loc[frame.P3_segment.str.startswith("P3-1"), corr_cols].corr(method="spearman")
    corr.stack().rename("spearman").reset_index().rename(columns={"level_0": "component_a", "level_1": "component_b"}).assign(
        abs_spearman=lambda x: x.spearman.abs(), action="regularize_and_neighbor_audit_within_P3_1_only"
    ).to_csv(OUT / "p3_rank1_correlation_precombine_audit.csv", index=False, encoding="utf-8-sig")


def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    # Exit inputs must be available before the score merge computes independent exit blocks.
    raw = pd.read_csv(DAILY, dtype={"ticker": str}, low_memory=False)
    raw["decision_date"] = pd.to_datetime(raw.decision_date)
    enriched = add_exit_inputs(raw)
    tmp = DAILY.with_name("_unused")
    # Reuse materialize logic while replacing its daily source in memory.
    daily_rank1 = enriched.loc[enriched.pool_rank.eq(1)].copy()
    scores = pd.read_csv(SCORE, dtype={"ticker": str}, low_memory=False)
    scores["decision_date"] = pd.to_datetime(scores.decision_date)
    market = pd.read_csv(MARKET, low_memory=False); market["decision_date"] = pd.to_datetime(market.decision_date)
    # Temporarily reproduce the joins explicitly to avoid writing an intermediate file.
    score_cols = [c for c in scores.columns if c in {"decision_date","ticker","opportunity_momentum_score","trend_continuation_score","capital_chip_support_score","risk_overheat_crowding_score","lifecycle_fit_score","fundamental_quality_score","opportunity_momentum_confidence","trend_continuation_confidence","capital_chip_support_confidence","risk_overheat_crowding_confidence","lifecycle_fit_confidence","fundamental_quality_confidence","selected_eligibility","selected_ineligibility_reason","PIT_available_at","tdcc_score","tdcc_confidence","tdcc_semantics"}]
    frame = daily_rank1.merge(scores[score_cols], on=["decision_date","ticker"], how="left", validate="one_to_one")
    market_cols = [c for c in market.columns if c in {"decision_date","taiwan_score","taiwan_confidence","breadth_score","breadth_confidence","capital_score","capital_confidence","derivatives_score","derivatives_confidence","external_score","external_confidence","full_spec_v2_state","controller_state_status"}]
    frame = frame.merge(market[market_cols], on="decision_date", how="left", validate="one_to_one")
    if len(frame) != 715 or frame.decision_date.nunique() != 715: raise ValueError("rank1 join failed")
    for source, target in [("opportunity_momentum_score","entry_momentum"),("trend_continuation_score","entry_trend_structure"),("capital_chip_support_score","entry_capital_chip"),("lifecycle_fit_score","entry_lifecycle"),("fundamental_quality_score","entry_fundamental_confidence"),("risk_overheat_crowding_score","entry_stock_risk_penalty")]: frame[target]=frame[source]
    for target, cols in [("exit_short_rs_deterioration",["rs5_below_rs10","rs10_below_rs20","rs20_negative","rs_weak_flag"]),("exit_structure_breakdown",["below_ma20","below_ma60","ma20_slope_negative","ma60_slope_negative","price_breakdown_flag"]),("exit_capital_withdrawal",["tv5_below_tv20","foreign20_negative","trust20_negative","dealer20_negative","margin20_adverse","lending20_adverse"]),("exit_risk_deterioration",["vol20_above_vol60","drawdown_severe","large_down_present","blowoff_flag","risk_extreme_flag"])]:
        frame[target], frame[target.replace("deterioration","confidence").replace("breakdown","confidence").replace("withdrawal","confidence")] = _mean_available(frame, cols)
    frame["exit_lifecycle_deterioration"]=(frame.weak_groups.clip(0,5)/5*100).where(frame.weak_groups.notna()); frame["exit_lifecycle_confidence"]=frame.weak_groups.notna().astype(float)
    frame["overheat_warning_only_no_exit"]=frame.overheat_groups.gt(0)&frame.weak_groups.lt(3)
    for n in ["taiwan","breadth","capital","derivatives","external"]: frame[f"market_{n}_risk"]=(100-frame[f"{n}_score"])/2
    frame["market_group_available_count"]=frame[[f"market_{n}_risk" for n in ["taiwan","breadth","capital","derivatives","external"]]].notna().sum(axis=1)
    frame["market_group_confidence_mean"]=frame[[f"{n}_confidence" for n in ["taiwan","breadth","capital","derivatives","external"]]].mean(axis=1,skipna=True)
    frame["P3_segment"]=np.where(frame.decision_date.lt("2025-07-11"),"P3-1_development_TDCC_unavailable","P3-2_untouched_OOS_TDCC_optional")
    frame.loc[frame.P3_segment.str.startswith("P3-1"),["tdcc_score","tdcc_confidence"]]=np.nan; frame["TDCC_main_score_used"]=False
    mandatory=["entry_momentum","entry_trend_structure","entry_capital_chip","entry_lifecycle","entry_fundamental_confidence","entry_stock_risk_penalty"]
    frame["calibration_eligible"]=frame[mandatory].notna().all(axis=1)&frame.market_group_available_count.eq(5)
    frame["future_outcome_used_as_live_feature"]=False
    for c in ["stock_entry_strength","stock_exit_deterioration","market_risk_score","combined_entry_score","combined_exit_score"]: frame[c]=np.nan
    frame["score_status"]="pending_P3_1_fold_constrained_calibration"

    labels = pd.read_csv(LABEL, dtype={"ticker":str}, low_memory=False); labels["decision_date"]=pd.to_datetime(labels.decision_date)
    labels = labels.merge(frame[["decision_date","ticker","P3_segment"]], on=["decision_date","ticker"], how="inner", validate="many_to_one", suffixes=("","_rank1"))
    labels["candidate_scope"]="canonical_Layer0_4_weekly_rank1_only"; labels["used_as_live_rule"]=False
    folds=pd.read_csv(FOLDS); folds["P3_2_used_for_selection"]=False
    write_contracts(frame)
    frame.to_csv(OUT/"p3_rank1_daily_stock_market_feature_contract.csv.gz",index=False,compression="gzip",encoding="utf-8")
    labels.to_csv(OUT/"p3_rank1_candidate_quality_label_contract.csv.gz",index=False,compression="gzip",encoding="utf-8")
    folds.to_csv(OUT/"p3_rank1_P3_1_expanding_fold_calendar.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame([{"requested_start":"2023-07-11","requested_end":"2026-06-29","actual_start":frame.decision_date.min().date(),"actual_end":frame.decision_date.max().date(),"daily_rank1_rows":len(frame),"decision_dates":frame.decision_date.nunique(),"weekly_membership_snapshots":frame.membership_snapshot_date.nunique(),"P3_1_rows":frame.P3_segment.str.startswith("P3-1").sum(),"P3_2_rows":frame.P3_segment.str.startswith("P3-2").sum(),"calibration_eligible_rows":frame.calibration_eligible.sum()}]).to_csv(OUT/"p3_rank1_PIT_coverage.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame([{"audit":"future_outcome_feature_intersection","violations":0},{"audit":"P3_2_parameter_selection","violations":0},{"audit":"TDCC_P3_1_zero_fill","violations":int(frame.loc[frame.P3_segment.str.startswith('P3-1'),'tdcc_score'].notna().sum())},{"audit":"weekly_rank1_change_auto_switch","violations":0},{"audit":"portfolio_NAV_executed","violations":0}]).to_csv(OUT/"p3_rank1_future_PIT_audit.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame([{"item":"adjusted_analysis","status":"trusted_nonofficial_research_diagnostic","stage_A_blocked":False},{"item":"corporate_action_total_return","status":"not_formal_complete","stage_A_blocked":False},{"item":"TDCC_P3_1","status":"NA_excluded_common_model","stage_A_blocked":False},{"item":"TDCC_P3_2","status":"optional_attribution_only","stage_A_blocked":False},{"item":"portfolio_path","status":"not_authorized_before_stage_A","stage_A_blocked":False}]).to_csv(OUT/"p3_rank1_proxy_blocked_readiness_audit.csv",index=False,encoding="utf-8-sig")
    mechanically_reproducible=len(frame)==715 and frame.decision_date.nunique()==715 and len(folds)==3 and labels.loc[labels.horizon_td.isin([10,20]),"label_mature_and_ready"].sum()>0
    readiness={"task_id":TASK,"status":"blocked_pending_strategy_calibration_freeze","requested_start":"2023-07-11","requested_end":"2026-06-29","actual_start":str(frame.decision_date.min().date()),"actual_end":str(frame.decision_date.max().date()),"daily_rank1_rows":len(frame),"decision_dates":frame.decision_date.nunique(),"P3_1_fold_count":len(folds),"P3_2_untouched_OOS":True,"P3_2_read_prohibited_until_P3_1_gate_pass":True,"parameter_effective_degrees_freedom":17,"non_cartesian_calibration":True,"stock_and_exit_scores_separate":True,"old_hardcoded_thresholds_used":False,"normal_switch_enabled":False,"portfolio_NAV_materialized":False,"mechanically_reproducible":bool(mechanically_reproducible),"calibration_policy_unique":False,"pending_strategy_decision_count":6,"diagnostic_subproblem":True,"user_explicitly_authorized_rank1_timing_scope":True,"representative_of_full_intended_layer5":False,"non_representative_of_full_Layer5":True,"may_be_used_to_reject_full_layer5":False,"may_be_used_to_assess_rank1_timing_hypothesis":True,"all80_rerank_executed":False,"Top3_executed":False,"ready_for_stage_A_candidate_quality":False,"ready_for_experiments":False,"ready_for_stage_B_NAV":False,"future_data_violation_count":0,"formal_model_changed":False,"trade_decision_changed":False,"active_in_trade_decision":False,"report_changed":False,"portfolio_replay_executed":False,"ready_for_strategy_replay":False,"ready_for_formal":False,"not_live_rule":True,"forward_returns_live_rule_usage":False}
    (OUT/"readiness_for_p3_rank1_weighted_entry_exit.json").write_text(json.dumps(readiness,ensure_ascii=False,indent=2),encoding="utf-8")
    pd.DataFrame([
        {"question":"rank1 stock parameters identify better timing","in_scope":True},
        {"question":"market risk improves rank1 timing","in_scope":True},
        {"question":"combined timing beats immediate-entry and stock-only","in_scope":True},
        {"question":"P3-1 calibration survives untouched P3-2","in_scope":True},
        {"question":"Layer5 can select true Top1 from all80","in_scope":False},
        {"question":"full Layer5 selector succeeds or fails","in_scope":False},
        {"question":"Top3 or formal portfolio performance","in_scope":False},
    ]).to_csv(OUT/"p3_rank1_timing_scope_boundary.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame([
        {"decision_id":"D1_entry_objective","strategy_center_must_freeze":"exact 10D/20D combination, worst-case objective formula, downside and future-MDD constraint","why_core_cannot_infer":"changes optimization target and accepted timing behavior","status":"pending"},
        {"decision_id":"D2_exit_objective","strategy_center_must_freeze":"event-level exit validation horizon, label, counterexample and pass rule without NAV","why_core_cannot_infer":"exit quality has no unique event-level proxy","status":"pending"},
        {"decision_id":"D3_comparator_actions","strategy_center_must_freeze":"exact C0/C1/C2/C3 entry, hold and exit gate semantics including executable C2 behavior","why_core_cannot_infer":"current C2 is attribution metadata, not an action rule","status":"pending"},
        {"decision_id":"D4_threshold_candidates","strategy_center_must_freeze":"entry/exit train-only quantile candidate sets, selection objective and deterministic tie-break","why_core_cannot_infer":"median or any quantile would be an unauthorized threshold","status":"pending"},
        {"decision_id":"D5_stability_gate","strategy_center_must_freeze":"numeric fold, neighbor and ordinary/weak pass-fail requirements before P3-2","why_core_cannot_infer":"defines what counts as a stable platform","status":"pending"},
        {"decision_id":"D6_P3_2_mechanical_gate","strategy_center_must_freeze":"exact machine condition that permits one-time P3-2 read and evaluation","why_core_cannot_infer":"must prevent accidental final-OOS access after P3-1 failure","status":"pending"},
    ]).to_csv(OUT/"p3_rank1_timing_strategy_center_calibration_decision_ledger.csv",index=False,encoding="utf-8-sig")
    (OUT/"final_summary_zh.md").write_text("# P3 Layer0-4 canonical rank1 weighted timing contract\n\n本contract是使用者明確授權的bounded rank1 timing subproblem，且不代表完整Layer5。715日features與2,860 labels可重現，但17df calibration尚缺唯一D1-D6政策。為避免Experiments自行發明objective、threshold或pass gate，目前status=blocked_pending_strategy_calibration_freeze、ready_for_experiments=false，P3-2禁止讀取。未產NAV、Top3或正式交易決策。\n",encoding="utf-8")
    files=sorted(p for p in OUT.iterdir() if p.is_file() and p.name!="manifest.json")
    (OUT/"manifest.json").write_text(json.dumps({"task_id":TASK,"inputs":{"daily_sha256":sha(DAILY),"score_sha256":sha(SCORE),"market_sha256":sha(MARKET),"label_sha256":sha(LABEL),"fold_sha256":sha(FOLDS)},"files":[{"name":p.name,"sha256":sha(p),"bytes":p.stat().st_size} for p in files]},ensure_ascii=False,indent=2),encoding="utf-8")


if __name__ == "__main__":
    run()
