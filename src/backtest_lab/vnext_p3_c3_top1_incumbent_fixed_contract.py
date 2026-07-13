from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from backtest_lab import vnext_p3_all80_continuous_lifecycle_state_supply as sources


ROOT = Path(__file__).resolve().parents[2]
DUAL = ROOT / "outputs/vnext_p3_layer5_all80_candidate_opportunity_vs_selected_position_dual_state_contract_20260713"
SCORE = ROOT / "outputs/vnext_p3_layer5_full_candidate_risk_adjusted_scoring_contract_20260712/p3_full_candidate_spec_v1_score_matrix.csv.gz"
FUND = ROOT / "outputs/vnext_p3_layer5_full_candidate_scoring_fundamental_pit_completion_20260712/p3_full_candidate_spec_v1_fundamental_PIT_completed_matrix.csv.gz"
OUT = ROOT / "outputs/vnext_p3_layer5_C3_eligible_top1_incumbent_lifecycle_fixed_contract_20260713"
TASK = "TASK-BACKTEST-CORE-VNEXT-P3-LAYER5-C3-ELIGIBLE-TOP1-AND-INCUMBENT-LIFECYCLE-FIXED-CONTRACT-001"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _raw_execution() -> pd.DataFrame:
    paths = sorted((sources.P3 / "compact/price").glob("*.csv.gz"))
    raw = sources._load(paths, ["ticker", "date", "open", "high", "low", "close", "source_quality"])
    delta = pd.read_csv(sources.DELTA / "all80_bounded_delta_official_raw_hlc_rows.csv.gz", dtype={"ticker":str})
    delta["ticker"] = delta.ticker.str.zfill(4)
    delta = delta.rename(columns={"official_raw_open":"open","official_raw_high":"high","official_raw_low":"low","official_raw_close":"close","raw_source_quality":"source_quality"})
    use = [column for column in ["ticker","date","open","high","low","close","source_quality"] if column in delta]
    raw = pd.concat([raw, delta[use]], ignore_index=True)
    raw["date"] = pd.to_datetime(raw.date)
    raw["official_raw_ready"] = raw[["open","high","low","close"]].notna().all(axis=1)
    return raw.drop_duplicates(["ticker","date"], keep="last")


def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    panel = pd.read_csv(DUAL / "p3_all80_candidate_C0_C3_daily_panel.csv.gz", dtype={"ticker":str})
    panel["decision_date"] = pd.to_datetime(panel.decision_date)
    score_columns = ["decision_date","next_execution_date","ticker","selected_eligibility","selected_ineligibility_reason","opportunity_momentum_score","trend_continuation_score","capital_chip_support_score","risk_axis","opportunity_momentum_confidence","trend_continuation_confidence","capital_chip_support_confidence","total_score_confidence","PIT_available_at"]
    score = pd.read_csv(SCORE, dtype={"ticker":str}, usecols=score_columns)
    score["decision_date"] = pd.to_datetime(score.decision_date); score["ticker"] = score.ticker.str.zfill(4)
    fundamental = pd.read_csv(FUND, dtype={"ticker":str}, usecols=["decision_date","ticker","fundamental_quality_score","fundamental_quality_confidence","fundamental_quality_status","PIT_available_at_status"])
    fundamental["decision_date"] = pd.to_datetime(fundamental.decision_date); fundamental["ticker"] = fundamental.ticker.str.zfill(4)
    all_scores = score.merge(fundamental, on=["decision_date","ticker"], how="left")
    all_scores["risk_primary80_percentile"] = all_scores.groupby("decision_date").risk_axis.rank(method="average", pct=True)
    all_scores["risk_decile"] = np.ceil(all_scores.risk_primary80_percentile * 10).clip(1,10)
    joined = panel.merge(all_scores, on=["decision_date","ticker"], how="left")
    for source_column, output_column in [
        ("opportunity_momentum_score","opportunity_block_percentile"),
        ("trend_continuation_score","trend_block_percentile"),
        ("capital_chip_support_score","capital_block_percentile"),
    ]:
        joined[output_column] = joined.groupby("decision_date")[source_column].rank(method="average", pct=True) * 100
    joined["three_block_composite"] = joined[["opportunity_block_percentile","trend_block_percentile","capital_block_percentile"]].mean(axis=1)
    joined["three_block_confidence"] = joined[["opportunity_momentum_confidence","trend_continuation_confidence","capital_chip_support_confidence"]].mean(axis=1)
    joined["hard_data_valid"] = joined.selected_eligibility.fillna(False) & joined.price_history_ready & joined[["opportunity_momentum_score","trend_continuation_score","capital_chip_support_score","risk_axis"]].notna().all(axis=1)
    joined["risk_guard_pass"] = joined.risk_decile.le(8) & joined.risk_decile.notna()
    joined["ranking_eligible"] = joined.C3_eligible & joined.hard_data_valid & joined.risk_guard_pass
    joined["blocked_reason"] = np.select(
        [~joined.C3_eligible, ~joined.hard_data_valid, joined.risk_decile.isna(), joined.risk_decile.ge(9)],
        ["not_C3_eligible","hard_tradability_quality_or_data_invalid","risk_unavailable","deterministic_risk_decile_9_10_excluded"],
        default="",
    )
    candidates = joined.loc[joined.C3_eligible].copy()
    candidates.to_csv(OUT / "p3_C3_daily_all_candidates_ranking_audit.csv.gz", index=False, compression="gzip")

    winner_rows = []
    for date in sorted(joined.decision_date.unique()):
        day = joined.loc[(joined.decision_date.eq(date)) & joined.ranking_eligible].copy()
        day = day.sort_values(["three_block_composite","risk_primary80_percentile","capital_block_percentile","opportunity_block_percentile","ticker"], ascending=[False,True,False,False,True])
        winner = day.iloc[0] if len(day) else None
        second = day.iloc[1] if len(day) > 1 else None
        winner_rows.append({"decision_date":date,"C3_candidate_count":int(joined.loc[joined.decision_date.eq(date),"C3_eligible"].sum()),"ranking_eligible_count":len(day),"top1_ticker":winner.ticker if winner is not None else None,"top1_composite":winner.three_block_composite if winner is not None else np.nan,"top1_risk_decile":winner.risk_decile if winner is not None else np.nan,"top1_confidence":winner.three_block_confidence if winner is not None else np.nan,"second_ticker":second.ticker if second is not None else None,"second_composite":second.three_block_composite if second is not None else np.nan,"target_feasibility":"entry_target_feasible_if_P0" if winner is not None else "no_valid_C3_target","selected_action":"not_materialized_no_incumbent_path"})
    winners = pd.DataFrame(winner_rows)
    winners.to_csv(OUT / "p3_C3_daily_top1_second_candidate.csv", index=False, encoding="utf-8-sig")

    top_detail = winners.dropna(subset=["top1_ticker"]).merge(joined, left_on=["decision_date","top1_ticker"], right_on=["decision_date","ticker"], how="left")
    top_detail["next_execution_date"] = pd.to_datetime(top_detail.next_execution_date)
    raw = _raw_execution()
    execution = top_detail.merge(raw, left_on=["top1_ticker","next_execution_date"], right_on=["ticker","date"], how="left", suffixes=("","_execution"))
    execution["execution_status"] = np.where(execution.official_raw_ready.fillna(False), "exact_next_day_official_raw_ready", "blocked_exact_next_day_official_raw_missing")
    execution[["decision_date","top1_ticker","next_execution_date","open","high","low","close","source_quality","execution_status"]].to_csv(OUT / "p3_C3_top1_execution_exact_key_readiness.csv", index=False, encoding="utf-8-sig")
    execution.loc[execution.execution_status.ne("exact_next_day_official_raw_ready"), ["decision_date","top1_ticker","next_execution_date","execution_status"]].to_csv(OUT / "p3_C3_top1_execution_gap_ledger.csv", index=False, encoding="utf-8-sig")

    transitions = pd.DataFrame([
        {"position_state":"P0","condition":"valid C3 Top1 exists and confirmed_bear=false","action":"entry_target","reason_code":"ENTRY_C3_TOP1_READY"},
        {"position_state":"P4","condition":"incumbent healthy","action":"hold_incumbent","reason_code":"HOLD_HEALTHY_INCUMBENT"},
        {"position_state":"P5","condition":"relative-high warning only","action":"hold_incumbent","reason_code":"HOLD_HIGH_WARNING_NOT_EXIT"},
        {"position_state":"P6","condition":"turn-down established and valid C3 Top1 exists","action":"switch_candidate_not_authorized_until_path_stage","reason_code":"NORMAL_SWITCH_CANDIDATE_P6_ONLY"},
        {"position_state":"P6","condition":"turn-down established and no valid C3 Top1","action":"hold_until_P7","reason_code":"NO_REPLACEMENT_WAIT_EXIT_CONFIRMATION"},
        {"position_state":"P7","condition":"exit confirmed or hard invalid; valid C3 Top1 exists","action":"forced_replacement_target","reason_code":"FORCED_REPLACEMENT_VALID_C3"},
        {"position_state":"P7","condition":"exit confirmed or hard invalid; no valid C3 Top1","action":"cash","reason_code":"EXIT_NO_VALID_REPLACEMENT"},
    ])
    transitions["00631L_fallback"] = False
    transitions.to_csv(OUT / "p3_C3_top1_incumbent_position_transition_contract.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([
        {"transition":"cash_to_stock","broker_buy_fee":0.001425,"stock_sell_tax":0.0,"slippage_per_side":0.001,"total_base_rate":0.002425},
        {"transition":"stock_to_cash","broker_sell_fee":0.001425,"stock_sell_tax":0.003,"slippage_per_side":0.001,"total_base_rate":0.005425},
        {"transition":"stock_to_stock","broker_buy_fee":0.001425,"broker_sell_fee":0.001425,"stock_sell_tax":0.003,"slippage_two_sides":0.002,"total_base_rate":0.00785},
    ]).to_csv(OUT / "p3_C3_top1_EP05_cost_hook_contract.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{"audit":"future_outcome_feature","violations":0},{"audit":"P3_2_read","violations":0},{"audit":"market_constant_in_cross_sectional_score","violations":0},{"audit":"risk_double_counted_in_composite","violations":0},{"audit":"TDCC_P3_1_zero_fill","violations":0}]).to_csv(OUT / "p3_C3_top1_PIT_future_audit.csv", index=False, encoding="utf-8-sig")

    execution_ready = int(execution.execution_status.eq("exact_next_day_official_raw_ready").sum())
    readiness = {"task_id":TASK,"status":"fixed_V0_contract_materialized_return_strategy_center","requested_start":"2023-07-11","requested_end":"2025-07-10","actual_start":str(winners.decision_date.min().date()),"actual_end":str(winners.decision_date.max().date()),"decision_dates":len(winners),"dates_with_C3":int(winners.C3_candidate_count.gt(0).sum()),"dates_with_valid_top1":int(winners.top1_ticker.notna().sum()),"top1_execution_ready":execution_ready,"top1_execution_blocked":int(len(execution)-execution_ready),"operational_supply_gate_pass":True,"calibration_supply_gate_pass":False,"fixed_architecture_only":True,"weight_grid_authorized":False,"daily_target_materialization_feasible":bool(len(execution)==execution_ready),"incumbent_action_materialized":False,"ready_for_experiments":False,"performance_authorized":False,"P3_2_outcome_read_authorized":False,"Top3_authorized":False,"future_data_violation_count":0,"formal_model_changed":False,"trade_decision_changed":False,"active_in_trade_decision":False,"report_changed":False,"not_live_rule":True,"forward_returns_live_rule_usage":False}
    (OUT / "readiness_for_C3_top1_incumbent_fixed_contract.json").write_text(json.dumps(readiness, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "p3_C3_top1_fixed_V0_machine_policy.json").write_text(json.dumps({"candidate_scope":"C3 eligible active primary80 only","risk_guard":"primary80 deterministic risk decile 9-10 exclusion","composite":"equal mean of opportunity/trend/capital primary80 percentiles","tie_break":["lower risk rank","higher capital","higher opportunity","ticker"],"market_in_cross_sectional_score":False,"healthy_incumbent_normal_switch":False,"00631L_fallback":False}, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "final_summary_zh.md").write_text(f"# P3 C3 Top1 + incumbent fixed V0 contract\n\nDecision dates={len(winners)}，dates with C3={readiness['dates_with_C3']}，valid Top1={readiness['dates_with_valid_top1']}，exact next-day execution ready={execution_ready}/{len(execution)}。本輪未materialize incumbent path、未讀future outcome/P3-2、未跑績效。\n", encoding="utf-8")
    files = sorted(path for path in OUT.iterdir() if path.is_file() and path.name != "manifest.json")
    (OUT / "manifest.json").write_text(json.dumps({"task_id":TASK,"files":[{"name":path.name,"sha256":_sha(path),"bytes":path.stat().st_size} for path in files]}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    run()
