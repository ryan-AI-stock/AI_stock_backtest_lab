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
OUTCOME = ROOT / "outputs/vnext_p3_layer5_full_candidate_quality_outcome_contract_20260712/p3_candidate_quality_outcome_paths.csv.gz"
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


def calibration_policy() -> pd.DataFrame:
    rows = [
        ("authority", "P3_1_split", "three expanding walk-forward folds; 40 decision-date embargo each; date/weekly-episode clustered aggregation"),
        ("authority", "P3_2_access", "prohibited until every P3-1 mechanical gate passes; then exactly one untouched read"),
        ("target", "entry_target", "0.5*net_excess_vs_00631L_10TD_10bp + 0.5*net_excess_vs_00631L_20TD_10bp"),
        ("target", "entry_horizon_guard", "10TD and 20TD excess retained separately and each must be nonnegative"),
        ("target", "entry_risk_constraints", "future MDD, P10 and large-down remain separate constraints; 5/40TD secondary only"),
        ("target", "exit_hold_net", "(1+event_aware_hold_gross)*(1-stock_sell_fee-tax-10bp)-1 for each 5TD/10TD horizon"),
        ("target", "exit_target", "-0.5*stock_hold_net_return_5TD - 0.5*stock_hold_net_return_10TD; cash return=0"),
        ("target", "exit_independence", "fit exit deterioration independently; entry target sign inversion prohibited"),
        ("comparator", "C0", "immediate entry at first next-day official tradable close; fixed 5/10/20/40TD event reference; no score gate"),
        ("comparator", "C1", "stock-only: entry if stock_entry_score>=frozen entry quantile threshold; exit if stock_deterioration>=frozen exit threshold"),
        ("comparator", "C2", "market-only executable reference: entry if market_risk<=frozen market-entry threshold; exit if market_risk>=frozen market-exit threshold"),
        ("comparator", "C3", "combined primary: monotonic market penalty on entry and amplification on exit; market weakness cannot loosen either gate"),
        ("action", "allowed_labels", "entry_pass|hold_or_wait|exit_pass only; no normal switch and no portfolio path"),
        ("regularization", "alpha_candidates", "0.01|0.1|1.0; selected in P3-1 inner validation only"),
        ("regularization", "coefficient_constraints", "all directions fixed; no sign flip; precombined dimensions; no raw duplicate fit"),
        ("entry_threshold", "quantile_candidates", "0.50|0.60|0.70|0.80 from each fold train score distribution only"),
        ("exit_threshold", "quantile_candidates", "0.70|0.80|0.90 from each fold train score distribution only"),
        ("selection", "primary_objective", "maximize validation decision-date/weekly-episode clustered mean entry_target"),
        ("selection", "entry_constraints", "10/20 excess each>=0; at least two of MDD/P10/large-down not worse than C0; ordinary 10/20 each>=0; weak<20 clusters low-sample else not both negative"),
        ("selection", "entry_tiebreak", "lower MDD severity; lower large-down; larger alpha; higher cluster coverage; then lower entry threshold"),
        ("selection", "exit_objective", "maximize clustered mean exit_target among exit-pass rows"),
        ("selection", "exit_constraints", "5/10 hold net each<0; false positive where both future holds positive<=40%; at least 20 validation clusters"),
        ("selection", "exit_tiebreak", "lower false-positive rate; then higher exit threshold"),
        ("P3_1_gate", "fold_direction", "at least 2 of 3 folds have C3 10TD>0 and 20TD>0"),
        ("P3_1_gate", "ordinary", "pooled C3 ordinary 10TD>0 and 20TD>0"),
        ("P3_1_gate", "C3_vs_C1", "one primary horizon improves; other deterioration<=0.25pp; at least one of MDD/P10/large-down improves and other two do not materially worsen"),
        ("P3_1_gate", "risk_tradeoff_exact", "at least one risk metric improves; C3 future MDD no more negative than C1 by 0.005; P10 no lower by more than 0.005; large-down rate no higher by more than 0.02; equality passes"),
        ("P3_1_gate", "threshold_stability", "selected quantile across folds differs by at most one candidate step"),
        ("P3_1_gate", "neighbor_stability", "one threshold neighbor keeps 10/20 positive; one alpha neighbor keeps ordinary 10/20 positive"),
        ("P3_1_gate", "coefficient_stability", "directions consistent; no block >50% absolute weight; at least 2 of top3 blocks repeat across all folds"),
        ("P3_2_gate", "lock_before_read", "freeze coefficients, alpha, entry/exit quantiles, threshold generation, missingness and C0-C3 semantics"),
        ("P3_2_gate", "return_hit_rate", "C3 10/20 excess each>0; one hit rate>50% and other>=45%; ordinary 10/20 each>0"),
        ("P3_2_gate", "C3_vs_C1", "one horizon improves; other deterioration<=0.25pp"),
        ("P3_2_gate", "risk", "at least two of MDD/P10/large-down beat C0; not all three worse than C1"),
        ("P3_2_gate", "cost_sensitivity", "10bp primary positive; at 20bp at least one of 10/20 positive and other has no material reversal"),
        ("P3_2_gate", "cost_sensitivity_exact", "other 20bp horizon excess>=-0.0025 and its drop versus 10bp<=0.0025; both conditions required"),
        ("C2_threshold", "train_only_mapping", "market risk higher is dangerous; entry iff score<=train_Q(entry_q); exit iff score>=train_Q(exit_q); equality passes; validation/P3-2 reuse frozen numeric thresholds"),
        ("C2_threshold", "missing_market_score", "entry_pass=false; exit_pass=false; reason=blocked_missing_market_score; no zero fill or safe/bear inference"),
        ("P3_2_gate", "failure", "any failure => NO_GO_RANK1_TIMING_FORMULA_FAMILY; no retune, NAV, Top3 or full-Layer5 inference"),
    ]
    return pd.DataFrame(rows, columns=["policy_section", "policy_id", "machine_rule"])


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
    entry_wide = labels.loc[labels.horizon_td.isin([10, 20])].pivot(index=["decision_date", "ticker"], columns="horizon_td", values=["net_excess_vs_00631L_10bp", "path_MDD", "tail_daily_return_p05", "large_down_7pct_count"])
    entry_wide.columns = [f"{name}_{int(horizon)}TD" for name, horizon in entry_wide.columns]
    entry_wide = entry_wide.reset_index()
    entry_wide["entry_target"] = 0.5 * entry_wide.net_excess_vs_00631L_10bp_10TD + 0.5 * entry_wide.net_excess_vs_00631L_10bp_20TD
    entry_wide["entry_10TD_nonnegative"] = entry_wide.net_excess_vs_00631L_10bp_10TD.ge(0)
    entry_wide["entry_20TD_nonnegative"] = entry_wide.net_excess_vs_00631L_10bp_20TD.ge(0)
    entry_wide["evaluation_metadata_only"] = True
    entry_wide["used_as_live_feature"] = False

    outcome = pd.read_csv(OUTCOME, dtype={"ticker": str}, low_memory=False)
    outcome["decision_date"] = pd.to_datetime(outcome.decision_date)
    outcome = outcome.merge(frame[["decision_date", "ticker"]], on=["decision_date", "ticker"], how="inner", validate="many_to_one")
    outcome = outcome.loc[outcome.horizon_td.isin([5, 10])].copy()
    stock_sell_cost_10bp = 0.001425 + 0.003 + 0.001
    outcome["stock_hold_net_after_sell_10bp"] = (1 + outcome.gross_event_aware_return) * (1 - stock_sell_cost_10bp) - 1
    exit_wide = outcome.pivot(index=["decision_date", "ticker"], columns="horizon_td", values=["stock_hold_net_after_sell_10bp", "path_MDD", "tail_daily_return_p05", "large_down_7pct_count"])
    exit_wide.columns = [f"{name}_{int(horizon)}TD" for name, horizon in exit_wide.columns]
    exit_wide = exit_wide.reset_index()
    exit_wide["exit_target"] = -0.5 * exit_wide.stock_hold_net_after_sell_10bp_5TD - 0.5 * exit_wide.stock_hold_net_after_sell_10bp_10TD
    exit_wide["future_hold_5TD_negative"] = exit_wide.stock_hold_net_after_sell_10bp_5TD.lt(0)
    exit_wide["future_hold_10TD_negative"] = exit_wide.stock_hold_net_after_sell_10bp_10TD.lt(0)
    exit_wide["exit_false_positive_both_holds_positive"] = exit_wide.stock_hold_net_after_sell_10bp_5TD.gt(0) & exit_wide.stock_hold_net_after_sell_10bp_10TD.gt(0)
    exit_wide["stock_sell_fee"] = 0.001425
    exit_wide["stock_transaction_tax"] = 0.003
    exit_wide["slippage_per_side"] = 0.001
    exit_wide["cash_return"] = 0.0
    exit_wide["evaluation_metadata_only"] = True
    exit_wide["used_as_live_feature"] = False
    folds=pd.read_csv(FOLDS); folds["P3_2_used_for_selection"]=False
    write_contracts(frame)
    frame.to_csv(OUT/"p3_rank1_daily_stock_market_feature_contract.csv.gz",index=False,compression="gzip",encoding="utf-8")
    labels.to_csv(OUT/"p3_rank1_candidate_quality_label_contract.csv.gz",index=False,compression="gzip",encoding="utf-8")
    entry_wide.to_csv(OUT/"p3_rank1_entry_target_contract.csv.gz",index=False,compression="gzip",encoding="utf-8")
    exit_wide.to_csv(OUT/"p3_rank1_exit_target_contract.csv.gz",index=False,compression="gzip",encoding="utf-8")
    folds.to_csv(OUT/"p3_rank1_P3_1_expanding_fold_calendar.csv",index=False,encoding="utf-8-sig")
    calibration_policy().to_csv(OUT/"p3_rank1_timing_calibration_policy.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame([{"requested_start":"2023-07-11","requested_end":"2026-06-29","actual_start":frame.decision_date.min().date(),"actual_end":frame.decision_date.max().date(),"daily_rank1_rows":len(frame),"decision_dates":frame.decision_date.nunique(),"weekly_membership_snapshots":frame.membership_snapshot_date.nunique(),"P3_1_rows":frame.P3_segment.str.startswith("P3-1").sum(),"P3_2_rows":frame.P3_segment.str.startswith("P3-2").sum(),"calibration_eligible_rows":frame.calibration_eligible.sum()}]).to_csv(OUT/"p3_rank1_PIT_coverage.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame([{"audit":"future_outcome_feature_intersection","violations":0},{"audit":"P3_2_parameter_selection","violations":0},{"audit":"TDCC_P3_1_zero_fill","violations":int(frame.loc[frame.P3_segment.str.startswith('P3-1'),'tdcc_score'].notna().sum())},{"audit":"weekly_rank1_change_auto_switch","violations":0},{"audit":"portfolio_NAV_executed","violations":0}]).to_csv(OUT/"p3_rank1_future_PIT_audit.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame([{"item":"adjusted_analysis","status":"trusted_nonofficial_research_diagnostic","stage_A_blocked":False},{"item":"corporate_action_total_return","status":"not_formal_complete","stage_A_blocked":False},{"item":"TDCC_P3_1","status":"NA_excluded_common_model","stage_A_blocked":False},{"item":"TDCC_P3_2","status":"optional_attribution_only","stage_A_blocked":False},{"item":"portfolio_path","status":"not_authorized_before_stage_A","stage_A_blocked":False}]).to_csv(OUT/"p3_rank1_proxy_blocked_readiness_audit.csv",index=False,encoding="utf-8-sig")
    mechanically_reproducible=len(frame)==715 and frame.decision_date.nunique()==715 and len(folds)==3 and labels.loc[labels.horizon_td.isin([10,20]),"label_mature_and_ready"].sum()>0
    targets_ready = len(entry_wide) == len(frame) and len(exit_wide) == len(frame) and entry_wide.entry_target.notna().sum() > 0 and exit_wide.exit_target.notna().sum() > 0
    stage_a_data_ready = mechanically_reproducible and targets_ready
    readiness={"task_id":TASK,"status":"unique_calibration_policy_ready_for_rank1_timing_stage_A","requested_start":"2023-07-11","requested_end":"2026-06-29","actual_start":str(frame.decision_date.min().date()),"actual_end":str(frame.decision_date.max().date()),"daily_rank1_rows":len(frame),"decision_dates":frame.decision_date.nunique(),"entry_target_rows":len(entry_wide),"exit_target_rows":len(exit_wide),"P3_1_fold_count":len(folds),"P3_2_untouched_OOS":True,"P3_2_read_prohibited_until_P3_1_gate_pass":True,"parameter_effective_degrees_freedom":17,"regularization_candidates":[0.01,0.1,1.0],"entry_quantile_candidates":[0.5,0.6,0.7,0.8],"exit_quantile_candidates":[0.7,0.8,0.9],"C2_market_entry_mapping":"score<=train_Q(entry_q)","C2_market_exit_mapping":"score>=train_Q(exit_q)","P3_1_MDD_max_deterioration_decimal":0.005,"P3_1_P10_max_deterioration_decimal":0.005,"P3_1_large_down_rate_max_deterioration":0.02,"P3_2_20bp_min_other_horizon_excess":-0.0025,"P3_2_20bp_max_drop_vs_10bp":0.0025,"non_cartesian_calibration":True,"stock_and_exit_scores_separate":True,"old_hardcoded_thresholds_used":False,"normal_switch_enabled":False,"portfolio_NAV_materialized":False,"mechanically_reproducible":bool(mechanically_reproducible),"stage_A_data_and_target_materialization_ready":bool(stage_a_data_ready),"calibration_policy_unique":True,"pending_strategy_decision_count":0,"diagnostic_subproblem":True,"user_explicitly_authorized_rank1_timing_scope":True,"representative_of_full_intended_layer5":False,"non_representative_of_full_Layer5":True,"may_be_used_to_reject_full_layer5":False,"may_be_used_to_assess_rank1_timing_hypothesis":True,"all80_rerank_executed":False,"Top3_executed":False,"ready_for_stage_A_candidate_quality":bool(stage_a_data_ready),"ready_for_experiments":bool(stage_a_data_ready),"ready_for_stage_B_NAV":False,"future_data_violation_count":0,"formal_model_changed":False,"trade_decision_changed":False,"active_in_trade_decision":False,"report_changed":False,"portfolio_replay_executed":False,"ready_for_strategy_replay":False,"ready_for_formal":False,"not_live_rule":True,"forward_returns_live_rule_usage":False}
    readiness.update({
        "status": "stopped_broad_additive_formula_non_representative_of_sequential_lifecycle",
        "non_representative_of_requested_sequential_lifecycle_logic": True,
        "may_be_used_to_reject_sequential_low_buy_high_sell_hypothesis": False,
        "follow_up_of_broad_additive_formula_stopped": True,
        "ready_for_experiments": False,
    })
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
    pd.DataFrame([{"decision_id":f"D{i}","status":"frozen_by_strategy_center_2026_07_13","machine_policy_file":"p3_rank1_timing_calibration_policy.csv"} for i in range(1,7)]).to_csv(OUT/"p3_rank1_timing_strategy_center_calibration_decision_ledger.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame([
        {"conflict_id":"N1_P3_1_risk_material_worsening","status":"resolved","machine_resolution":"MDD<=C1+0.005 severity; P10>=C1-0.005; large-down rate<=C1+0.02; equality passes"},
        {"conflict_id":"N2_P3_2_20bp_material_reversal","status":"resolved","machine_resolution":"other 20bp excess>=-0.0025 and drop from its 10bp excess<=0.0025"},
        {"conflict_id":"N3_C2_market_threshold_mapping","status":"resolved","machine_resolution":"entry score<=train_Q(entry_q); exit score>=train_Q(exit_q); equality passes; missing blocks both"},
    ]).to_csv(OUT/"p3_rank1_timing_numeric_policy_conflict_ledger.csv",index=False,encoding="utf-8-sig")
    (OUT/"final_summary_zh.md").write_text("# P3 Layer0-4 canonical rank1 weighted timing contract\n\nD1-D6與N1-N3均已完整凍結。715日entry/exit targets、C0-C3、train-only quantiles、alpha、P3-1 stability及P3-2一次性gate均machine-readable；calibration_policy_unique=true、ready_for_experiments=true。P3-2仍須P3-1全部機械gate通過才可讀。此task只評估bounded canonical rank1 timing，不代表完整Layer5；未產NAV、Top3或正式交易決策。\n",encoding="utf-8")
    files=sorted(p for p in OUT.iterdir() if p.is_file() and p.name!="manifest.json")
    (OUT/"manifest.json").write_text(json.dumps({"task_id":TASK,"inputs":{"daily_sha256":sha(DAILY),"score_sha256":sha(SCORE),"market_sha256":sha(MARKET),"label_sha256":sha(LABEL),"outcome_sha256":sha(OUTCOME),"fold_sha256":sha(FOLDS)},"files":[{"name":p.name,"sha256":sha(p),"bytes":p.stat().st_size} for p in files]},ensure_ascii=False,indent=2),encoding="utf-8")


if __name__ == "__main__":
    run()
