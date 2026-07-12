from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "outputs/vnext_p3_layer5_weekly_rank1_single_candidate_minimum_contract_20260712"
FROZEN = ROOT / "outputs/vnext_p3_layer5_single_lifecycle_state_machine_contract_20260711"
OUT = ROOT / "outputs/vnext_p3_layer5_weekly_rank1_challenger_frozen_state_machine_contract_20260712"
TASK = "TASK-BACKTEST-CORE-VNEXT-P3-LAYER5-WEEKLY-RANK1-CHALLENGER-FROZEN-STATE-MACHINE-CONTRACT-001"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rank1 = pd.read_csv(SOURCE / "p3_weekly_rank1_single_candidate_feature_contract.csv", dtype={"ticker": str}, low_memory=False)
    rank1["decision_date"] = pd.to_datetime(rank1.decision_date)
    if len(rank1) != 154 or rank1.decision_date.nunique() != 154 or not rank1.canonical_rank1_lineage_ready.all():
        raise RuntimeError("canonical rank1 source contract invalid")

    rules = pd.DataFrame([
        {"action": "entry_rank1", "required_conditions": "rank1 state turning_up OR healthy_rise; price core valid; quality/risk veto absent; confidence pass; market-adjusted entry gate pass", "multi_evidence": "turning_up >=3/5 independent groups including RS repair and MA reclaim or capital improvement; healthy_rise >=3/4 groups", "single_signal_prohibited": "KD|BIAS|RS alone", "PIT_source": "weekly rank1 same-day feature contract + full_spec_v2 market controller", "missing_policy": "blocked_not_assume_neutral", "unique_without_strategy_decision": False},
        {"action": "hold_incumbent", "required_conditions": "incumbent valid AND state healthy_rise OR overheat_warning without confirmed_weakening", "multi_evidence": "overheat warning alone never exits", "single_signal_prohibited": "single KD/BIAS/RS/score decline", "PIT_source": "incumbent same-day Layer5 feature row", "missing_policy": "price/tradability invalid can force; optional chip NA lowers confidence", "unique_without_strategy_decision": True},
        {"action": "normal_switch_to_rank1", "required_conditions": "rank1 challenger margin pass AND >=3/6 block wins incl >=1 of momentum/trend/capital AND incumbent weakened AND after-cost edge pass AND confidence/price/quality/risk pass", "multi_evidence": "all clauses required", "single_signal_prohibited": "rank1 status alone; overheat alone", "PIT_source": "rank1 and incumbent same-date six-block evidence", "missing_policy": "missing comparable block cannot count as win", "unique_without_strategy_decision": False},
        {"action": "forced_replacement", "required_conditions": "incumbent confirmed_weakening OR formal stop OR price/tradability/core invalid OR severe quality/risk veto", "multi_evidence": "confirmed_weakening >=3/5 groups including price/trend breakdown and RS weakening or capital withdrawal", "single_signal_prohibited": "single-day move; single overheat", "PIT_source": "incumbent same-day feature and formal stop contract", "missing_policy": "hard mandatory invalid is explicit; optional NA is not invalid", "unique_without_strategy_decision": False},
        {"action": "no_position", "required_conditions": "confirmed_bear OR forced exit with no valid rank1 replacement", "multi_evidence": "confirmed bear full_spec_v2 cross-group two-day rule", "single_signal_prohibited": "no challenger; no new target", "PIT_source": "full_spec_v2 + incumbent invalidity", "missing_policy": "controller warmup/low-confidence cannot be mapped ordinary", "unique_without_strategy_decision": True},
    ])
    rules.to_csv(OUT / "p3_rank1_challenger_action_condition_contract.csv", index=False, encoding="utf-8-sig")

    evidence = pd.DataFrame([
        {"condition_family": "lifecycle_entry", "fields": "raw_state; RS repair; MA reclaim; capital improvement; risk; quality", "source": "rank1 feature contract", "PIT_ready": "131/154 full; 23/154 Layer4 partial", "missing_handling": "partial rows cannot assert absent evidence", "parameter_decision_needed": "confirmation count/time basis"},
        {"condition_family": "incumbent_validity", "fields": "price/tradability; confirmed weakening; severe risk/quality veto", "source": "daily Layer5 matrix for held ticker", "PIT_ready": "requires path-dependent incumbent audit", "missing_handling": "optional chip NA not invalid", "parameter_decision_needed": "formal stop applicability and severe-veto mapping"},
        {"condition_family": "challenger_superiority", "fields": "six independent block scores/confidence", "source": "full_candidate_spec_v1 same-day rows", "PIT_ready": "rank1 full rows 131/154; incumbent coverage not yet materialized", "missing_handling": "NA block cannot win", "parameter_decision_needed": "single accepted margin metric/value"},
        {"condition_family": "after_cost_edge", "fields": "EP05 fee; sell tax; 10bp/side; expected edge", "source": "cost contract plus unspecified ex-ante edge mapping", "PIT_ready": "cost ready; expected-edge mapping blocked", "missing_handling": "cannot substitute realized return", "parameter_decision_needed": "define ex-ante expected edge units and pass rule"},
        {"condition_family": "market_tightness", "fields": "full_spec_v2 five groups/state/confidence", "source": "full_spec_v2", "PIT_ready": "ready after warmup; P3 has zero confirmed bear", "missing_handling": "warmup metric-ineligible; TDCC not part common mandatory", "parameter_decision_needed": "weekly action uses Friday state or daily controller between snapshots"},
        {"condition_family": "TDCC", "fields": "tdcc score/confidence", "source": "P3-2 optional A/B", "PIT_ready": "P3-1 unavailable; P3-2 partial", "missing_handling": "NA not zero or neutral", "parameter_decision_needed": "none for common path; optional attribution only"},
    ])
    evidence.to_csv(OUT / "p3_rank1_challenger_condition_source_readiness.csv", index=False, encoding="utf-8-sig")

    decisions = pd.DataFrame([
        {"decision_id": "D1_confirmation_time_basis", "question": "多項確認是同一週截面成立、連續2個交易日，或連續2週？", "existing_evidence": "frozen lattice has 2/3/4 but base_selected=false; prior daily implementation used current-day groups without accepted persistence", "minimum_options": "same_decision_multi_group|2_daily_closes", "impact": "entry/weakening timing and coverage", "core_default_applied": False},
        {"decision_id": "D2_challenger_margin", "question": "正常換倉使用哪個可比較分數與唯一margin？", "existing_evidence": "lattice 3/5/8 all base_selected=false; simplified code hardcoded balanced score 5 but later selector was invalidated", "minimum_options": "accept full_candidate six-block score margin 5|define no normal switch until calibrated", "impact": "normal switch eligibility", "core_default_applied": False},
        {"decision_id": "D3_after_cost_edge_mapping", "question": "如何把事前score edge換算為足以覆蓋EP05+slippage的預期edge？", "existing_evidence": "cost is exact; no accepted score-to-return calibration; realized outcome prohibited", "minimum_options": "explicit score-to-return calibration contract|disable normal switch", "impact": "required switch clause currently non-computable", "core_default_applied": False},
        {"decision_id": "D4_forced_stop_veto", "question": "本輪是否移植formal 12% portfolio stop，及哪些quality/risk欄屬severe hard veto？", "existing_evidence": "frozen policy names formal stop/severe veto but does not uniquely map current fields", "minimum_options": "no portfolio stop and explicit hard invalid only|approved formal stop plus veto crosswalk", "impact": "forced replacement/no-position", "core_default_applied": False},
        {"decision_id": "D5_market_action_frequency", "question": "週頻rank1決策間，full_spec_v2日頻controller可否每日觸發forced exit/cash？", "existing_evidence": "market controller is daily; challenger is weekly", "minimum_options": "weekly Friday only|daily risk exits with weekly entries/switches", "impact": "state-machine frequency and OHLC path", "core_default_applied": False},
    ])
    decisions.to_csv(OUT / "p3_rank1_challenger_minimum_strategy_center_decision_table.csv", index=False, encoding="utf-8-sig")

    changes = rank1.ticker.ne(rank1.ticker.shift())
    scope = pd.DataFrame([{
        "requested_weekly_snapshots": 154,
        "canonical_rank1_rows": len(rank1),
        "unique_rank1_tickers": rank1.ticker.nunique(),
        "rank1_change_events_upper_bound": int(changes.iloc[1:].sum()),
        "state_machine_decision_events_upper_bound": 154,
        "transition_upper_bound": 153,
        "same_day_full_rank1_feature_rows": int(rank1.same_day_full_layer5_feature_ready.sum()),
        "same_day_partial_rank1_rows": int((~rank1.same_day_full_layer5_feature_ready).sum()),
        "OHLC_scope": "rank1 union plus path-dependent incumbent holdings; official next-day execution and daily event-aware marks",
        "bounded": True,
        "all80_rerank": False,
        "Top3": False,
        "Ridge_GBDT": False,
    }])
    scope.to_csv(OUT / "p3_rank1_challenger_bounded_path_scope_estimate.csv", index=False, encoding="utf-8-sig")

    rank1_trace = rank1[["decision_date", "ticker", "name", "pool_rank", "pool_selection_score", "pool_selection_score_col", "pool_selection_policy", "P3_segment", "same_day_full_layer5_feature_ready", "stock_strength_confidence", "market_risk_available_group_count", "market_risk_confidence", "full_spec_v2_state", "controller_state_status"]].copy()
    rank1_trace["state_machine_action"] = "blocked_pending_strategy_center_threshold_decisions"
    rank1_trace["future_return_used_as_rule"] = False
    rank1_trace.to_csv(OUT / "p3_rank1_challenger_weekly_input_trace.csv", index=False, encoding="utf-8-sig")

    readiness = {
        "task_id": TASK,
        "status": "contract_materialized_blocked_minimum_strategy_center_decisions_required",
        "canonical_rank1_ready": True,
        "weekly_rows": len(rank1),
        "condition_contract_ready": True,
        "source_PIT_readiness_audited": True,
        "spec_unique": False,
        "unresolved_strategy_decisions": len(decisions),
        "bounded_path_scope_ready": True,
        "state_machine_actions_materialized": False,
        "portfolio_path_executed": False,
        "ready_for_experiments": False,
        "blocker": "D1-D5 must be decided without using prior realized rank1 outcomes",
        "future_data_violation_count": 0,
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
    (OUT / "readiness_for_p3_rank1_challenger_state_machine.json").write_text(json.dumps(readiness, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "final_summary_zh.md").write_text("# P3 weekly rank1 challenger frozen state-machine contract\n\nCanonical rank1與有限判斷表已建立，但既有lattice沒有選定唯一門檻，after-cost ex-ante edge也無accepted mapping。為避免用上一輪realized outcome倒推規則，本包只列D1-D5最小Strategy Center裁決，不產action/path/performance。\n", encoding="utf-8")
    files = sorted(path for path in OUT.iterdir() if path.is_file() and path.name != "manifest.json")
    (OUT / "manifest.json").write_text(json.dumps({"task_id": TASK, "inputs": {"rank1_contract_sha256": sha(SOURCE / "p3_weekly_rank1_single_candidate_feature_contract.csv"), "frozen_parameter_lattice_sha256": sha(FROZEN / "p3_layer5_parameter_lattice.csv")}, "files": [{"name": path.name, "sha256": sha(path), "bytes": path.stat().st_size} for path in files]}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    run()
