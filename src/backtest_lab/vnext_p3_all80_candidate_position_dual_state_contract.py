from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from backtest_lab import vnext_p3_all80_entry_funnel_setup_latch_audit as audit
from backtest_lab import vnext_p3_all80_l3_10td_latch_state_supply as frozen


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/vnext_p3_layer5_all80_candidate_opportunity_vs_selected_position_dual_state_contract_20260713"
TASK = "TASK-BACKTEST-CORE-VNEXT-P3-LAYER5-ALL80-CANDIDATE-OPPORTUNITY-VS-SELECTED-POSITION-DUAL-STATE-CONTRACT-001"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frame = audit._base().sort_values(["ticker", "decision_date"]).copy()
    dates = sorted(frame.decision_date.unique())
    market_index = {pd.Timestamp(date): index for index, date in enumerate(dates)}
    fold_map = audit._fold_map(dates)
    frame["market_day_index"] = frame.decision_date.map(market_index)
    frame["fold"] = frame.decision_date.map(fold_map)
    frame["relative_low"] = frame.price_pct_6M.le(frozen.LOW) & (frame.K_pct_6M.le(frozen.LOW) | frame.BIAS_pct_6M.le(frozen.LOW))
    up = ["kd_up", "rs_repair", "ma_up", "capital_improve", "risk_ok"]
    for column in up:
        # Candidate eligibility is active-primary80 only, but ticker evidence memory is not reset by rank/membership changes.
        frame[f"{column}_3of5"] = frame.groupby("ticker", sort=False)[column].transform(
            lambda values: values.fillna(False).astype(int).rolling(frozen.WINDOW, min_periods=frozen.WINDOW).sum().ge(frozen.NEED)
        )
    frame["up_group_count"] = frame[[f"{column}_3of5" for column in up]].sum(axis=1)
    frame["entry_required_groups"] = frame.market_state.map({"strong_market":3,"ordinary_market":3,"weak_market":4,"confirmed_bear":6}).fillna(3)
    frame["entry_evidence_ready"] = frame.up_group_count.ge(frame.entry_required_groups) & frame.market_state.ne("confirmed_bear")
    frame["capital_confirmation_ready"] = frame.capital_improve_3of5

    rows, episodes = [], []
    for ticker, group in frame.groupby("ticker", sort=False):
        state = "C0"
        last_low = None
        episode_open = False
        episode_number = 0
        episode_record = None
        for row in group.itertuples():
            day = row.market_day_index
            if row.relative_low:
                if not episode_open:
                    episode_number += 1; episode_open = True
                    episode_record = {"ticker":ticker,"episode_id":f"{ticker}-CLOW{episode_number}","first_qualifying_date":row.decision_date,"last_qualifying_date":row.decision_date,"grace_expiry_index":day + frozen.GRACE}
                    episodes.append(episode_record)
                last_low = day
                episode_record["last_qualifying_date"] = row.decision_date
                episode_record["grace_expiry_index"] = day + frozen.GRACE
            elif episode_open and last_low is not None and day - last_low > frozen.GRACE:
                episode_open = False
            setup_active = episode_open and last_low is not None and day - last_low <= frozen.GRACE
            prior = state
            if not row.price_history_ready:
                state = "BLOCKED"
            elif state == "BLOCKED":
                state = "C1" if setup_active else "C0"
            elif not setup_active:
                state = "C0"
            elif state == "C0":
                state = "C1"
            elif state == "C1":
                state = "C2" if row.entry_evidence_ready else "C1"
            elif state == "C2":
                state = "C3" if row.entry_evidence_ready and row.capital_confirmation_ready else "C1"
            elif state == "C3":
                state = "C3" if row.entry_evidence_ready and row.capital_confirmation_ready else "C1"
            rows.append({"decision_date":row.decision_date,"ticker":ticker,"pool_rank":row.pool_rank,"fold":row.fold,"prior_candidate_state":prior,"candidate_state":state,"setup_episode_id":episode_record["episode_id"] if setup_active else None,"relative_low":row.relative_low,"setup_active":setup_active,"up_group_count":row.up_group_count,"entry_required_groups":row.entry_required_groups,"entry_evidence_ready":row.entry_evidence_ready,"capital_confirmation_ready":row.capital_confirmation_ready,"market_state":row.market_state,"price_history_ready":row.price_history_ready,"chip_confidence":row.chip_confidence})

    panel = pd.DataFrame(rows).sort_values(["ticker", "decision_date"])
    panel["C3_eligible"] = panel.candidate_state.eq("C3")
    panel["C3_cluster_start"] = panel.C3_eligible & ~panel.groupby("ticker").C3_eligible.shift(fill_value=False)
    panel.to_csv(OUT / "p3_all80_candidate_C0_C3_daily_panel.csv.gz", index=False, compression="gzip")
    pd.DataFrame(episodes).to_csv(OUT / "p3_all80_candidate_low_setup_episodes.csv.gz", index=False, compression="gzip")

    folds = panel.groupby("fold").agg(unique_C3_clusters=("C3_cluster_start","sum"), C3_eligible_dates=("decision_date", lambda s: s[panel.loc[s.index,"C3_eligible"]].nunique()), C3_eligible_rows=("C3_eligible","sum")).reindex([1,2,3], fill_value=0).reset_index()
    folds["cluster_gate_pass"] = folds.unique_C3_clusters.ge(20)
    folds["date_gate_pass"] = folds.C3_eligible_dates.ge(20)
    folds.to_csv(OUT / "p3_all80_candidate_C3_fold_supply_gate.csv", index=False, encoding="utf-8-sig")

    daily = panel.groupby("decision_date").C3_eligible.sum().reindex(dates, fill_value=0)
    no_c3 = daily.eq(0)
    longest = current = 0
    for missing in no_c3:
        current = current + 1 if missing else 0; longest = max(longest, current)
    distribution = {"mean":float(daily.mean()),"median":float(daily.median()),"p75":float(daily.quantile(.75)),"max":int(daily.max()),"no_C3_dates":int(no_c3.sum()),"longest_no_C3_trading_days":longest}
    pd.DataFrame([distribution]).to_csv(OUT / "p3_all80_candidate_C3_daily_count_distribution.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"decision_date":dates,"C3_candidate_count":daily.values,"no_C3":no_c3.values}).to_csv(OUT / "p3_all80_candidate_C3_daily_supply.csv", index=False, encoding="utf-8-sig")

    rearm = panel.groupby("ticker").agg(C3_clusters=("C3_cluster_start","sum"), C3_eligible_rows=("C3_eligible","sum"), setup_episodes=("setup_episode_id", lambda s: s.dropna().nunique())).reset_index()
    rearm["requires_prior_high_exit_cycle"] = False
    rearm["unselected_C3_enters_position_lane"] = False
    rearm.to_csv(OUT / "p3_all80_candidate_rearm_audit.csv", index=False, encoding="utf-8-sig")

    position_rules = pd.DataFrame([
        {"from_state":"P0","to_state":"P3","condition":"C3 selected by future Top1 selector and entry threshold passes","evaluable_now":False},
        {"from_state":"P3","to_state":"P4","condition":"official entry completes; incumbent healthy","evaluable_now":False},
        {"from_state":"P4","to_state":"P5","condition":"selected incumbent relative-high warning","evaluable_now":False},
        {"from_state":"P5","to_state":"P6","condition":"selected incumbent turn-down established","evaluable_now":False},
        {"from_state":"P6","to_state":"P7","condition":"selected incumbent exit confirmed","evaluable_now":False},
        {"from_state":"P7","to_state":"P0","condition":"official exit completes","evaluable_now":False},
    ])
    position_rules["supply_status"] = "not_evaluable_until_selector_and_incumbent_path"
    position_rules.to_csv(OUT / "p3_selected_position_P0_P7_transition_contract.csv", index=False, encoding="utf-8-sig")

    prior_audit = pd.read_csv(ROOT / "outputs/vnext_p3_layer5_all80_sequential_entry_funnel_lead_lag_setup_latch_audit_20260713/p3_all80_setup_latch_supply_counterfactual.csv")
    prior_l3 = prior_audit.loc[(prior_audit.platform.eq("L3")) & (prior_audit.persistence.eq("3of5")) & (prior_audit.latch_window_TD.eq(10))]
    reconciliation = folds[["fold","unique_C3_clusters"]].merge(prior_l3[["fold","entry_clusters"]].rename(columns={"entry_clusters":"prior_independent_setup_entry_clusters"}), on="fold", how="left")
    reconciliation["authority_change"] = "new Candidate Lane C3 persists and rearms without requiring P4-P7; no actual trade implied"
    reconciliation.to_csv(OUT / "p3_candidate_lane_vs_prior_event_counterfactual_reconciliation.csv", index=False, encoding="utf-8-sig")

    gate_pass = bool(folds.cluster_gate_pass.all() and folds.date_gate_pass.all() and folds.C3_eligible_rows.gt(0).all())
    policy = {"candidate_lane":["C0","C1","C2","C3"],"position_lane":["P0","P3","P4","P5","P6","P7"],"candidate_rearm_requires_position_exit":False,"C3_means_trade":False,"L3_low":frozen.LOW,"setup_grace_TD":frozen.GRACE,"persistence":"3of5","candidate_and_position_states_separated":True}
    (OUT / "p3_candidate_position_dual_state_machine_policy.json").write_text(json.dumps(policy, ensure_ascii=False, indent=2), encoding="utf-8")
    readiness = {"task_id":TASK,"status":"candidate_supply_gate_pass_return_strategy_center" if gate_pass else "candidate_supply_gate_failed_return_strategy_center","P3_1_dates":len(dates),"candidate_rows":len(panel),"C3_clusters_by_fold":folds.unique_C3_clusters.tolist(),"C3_eligible_dates_by_fold":folds.C3_eligible_dates.tolist(),"C3_daily_distribution":distribution,"candidate_supply_gate_pass":gate_pass,"candidate_and_position_states_separated":True,"represents_intended_all80_layer5_candidate_generation":True,"position_lane_supply_status":"not_evaluable_until_selector_and_incumbent_path","ready_for_experiments":False,"performance_authorized":False,"P3_2_outcome_read_authorized":False,"Top3_authorized":False,"future_data_violation_count":0,"formal_model_changed":False,"trade_decision_changed":False,"active_in_trade_decision":False,"report_changed":False,"not_live_rule":True,"forward_returns_live_rule_usage":False}
    (OUT / "readiness_for_candidate_position_dual_state.json").write_text(json.dumps(readiness, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "final_summary_zh.md").write_text(f"# P3 all80 Candidate vs Position dual-state contract\n\nCandidate C3 clusters={readiness['C3_clusters_by_fold']}，eligible dates={readiness['C3_eligible_dates_by_fold']}，gate={gate_pass}。Position P4-P7未materialize，因selector/incumbent尚未凍結。本輪未讀future outcome/P3-2或績效。\n", encoding="utf-8")
    files = sorted(path for path in OUT.iterdir() if path.is_file() and path.name != "manifest.json")
    (OUT / "manifest.json").write_text(json.dumps({"task_id":TASK,"files":[{"name":path.name,"sha256":_sha(path),"bytes":path.stat().st_size} for path in files]}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    run()
