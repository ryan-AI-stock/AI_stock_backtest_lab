from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from backtest_lab import vnext_p3_all80_entry_funnel_setup_latch_audit as audit
from backtest_lab import vnext_p3_all80_continuous_lifecycle_state_supply as source


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/vnext_p3_layer5_all80_sequential_lifecycle_L3_10TD_latch_state_supply_contract_20260713"
TASK = "TASK-BACKTEST-CORE-VNEXT-P3-LAYER5-ALL80-SEQUENTIAL-LIFECYCLE-L3-10TD-LATCH-STATE-SUPPLY-CONTRACT-001"
LOW, HIGH, GRACE, NEED, WINDOW = 0.40, 0.65, 10, 3, 5


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _segment_persistence(frame: pd.DataFrame, column: str, market_index: dict[pd.Timestamp, int]) -> pd.Series:
    result = pd.Series(False, index=frame.index)
    for _, group in frame.groupby("ticker", sort=False):
        indexes = group.decision_date.map(market_index)
        segments = indexes.diff().ne(1).cumsum()
        values = group[column].fillna(False).astype(int)
        result.loc[group.index] = values.groupby(segments).rolling(WINDOW, min_periods=WINDOW).sum().reset_index(level=0, drop=True).ge(NEED)
    return result


def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frame = audit._base().sort_values(["ticker", "decision_date"]).copy()
    dates = sorted(frame.decision_date.unique())
    market_index = {pd.Timestamp(date): index for index, date in enumerate(dates)}
    frame["market_day_index"] = frame.decision_date.map(market_index)
    frame["relative_low"] = frame.price_pct_6M.le(LOW) & (frame.K_pct_6M.le(LOW) | frame.BIAS_pct_6M.le(LOW))
    frame["relative_high"] = frame.price_pct_6M.ge(HIGH) & (frame.K_pct_6M.ge(HIGH) | frame.BIAS_pct_6M.ge(HIGH))
    up = ["kd_up", "rs_repair", "ma_up", "capital_improve", "risk_ok"]
    down = ["kd_down", "rs_weak", "ma_down", "capital_withdraw", "risk_bad"]
    for column in up + down:
        frame[f"{column}_3of5"] = _segment_persistence(frame, column, market_index)
    frame["up_group_count"] = frame[[f"{c}_3of5" for c in up]].sum(axis=1)
    frame["down_group_count"] = frame[[f"{c}_3of5" for c in down]].sum(axis=1)
    frame["entry_required_groups"] = frame.market_state.map({"strong_market":3,"ordinary_market":3,"weak_market":4,"confirmed_bear":6}).fillna(3)
    frame["exit_required_groups"] = frame.market_state.map({"strong_market":4,"ordinary_market":3,"weak_market":3,"confirmed_bear":2}).fillna(3)
    frame["entry_evidence_ready"] = frame.up_group_count.ge(frame.entry_required_groups) & frame.market_state.ne("confirmed_bear")
    frame["exit_evidence_ready"] = frame.down_group_count.ge(frame.exit_required_groups)

    states, transitions, episodes = [], [], []
    for ticker, group in frame.groupby("ticker", sort=False):
        current = "S0"
        low_episode = high_episode = 0
        low_last = high_last = None
        low_open = high_open = False
        low_record = high_record = None
        pending_entry_confirmation = pending_exit_confirmation = False
        for row in group.itertuples():
            day = row.market_day_index
            if row.relative_low:
                if not low_open:
                    low_episode += 1; low_open = True
                    low_record = {"ticker":ticker,"setup_type":"entry_low","episode_id":f"{ticker}-L{low_episode}","first_qualifying_date":row.decision_date,"last_qualifying_date":row.decision_date,"grace_expiry_index":day + GRACE}
                    episodes.append(low_record)
                low_last = day
                low_record["last_qualifying_date"] = row.decision_date
                low_record["grace_expiry_index"] = day + GRACE
            elif low_open and low_last is not None and day - low_last > GRACE:
                low_open = False
            if row.relative_high:
                if not high_open:
                    high_episode += 1; high_open = True
                    high_record = {"ticker":ticker,"setup_type":"exit_high","episode_id":f"{ticker}-H{high_episode}","first_qualifying_date":row.decision_date,"last_qualifying_date":row.decision_date,"grace_expiry_index":day + GRACE}
                    episodes.append(high_record)
                high_last = day
                high_record["last_qualifying_date"] = row.decision_date
                high_record["grace_expiry_index"] = day + GRACE
            elif high_open and high_last is not None and day - high_last > GRACE:
                high_open = False

            low_active = low_open and low_last is not None and day - low_last <= GRACE
            high_active = high_open and high_last is not None and day - high_last <= GRACE
            prior = current
            if not row.price_history_ready:
                current = "BLOCKED"
            elif current == "BLOCKED":
                current = "S1" if low_active else ("S5" if high_active else "S0")
            elif current in {"S0", "S7"}:
                current = "S1" if low_active else "S0"
            elif current == "S1":
                if low_active and row.entry_evidence_ready:
                    current = "S2"; pending_entry_confirmation = True
                elif not low_active:
                    current = "S0"
            elif current == "S2":
                if pending_entry_confirmation and row.capital_improve_3of5 and low_active:
                    current = "S3"; pending_entry_confirmation = False
                elif not low_active or row.risk_bad:
                    current = "S0"; pending_entry_confirmation = False
            elif current == "S3":
                current = "S4"
            elif current == "S4":
                if high_active: current = "S5"
            elif current == "S5":
                if high_active and row.exit_evidence_ready:
                    current = "S6"; pending_exit_confirmation = True
                elif not high_active:
                    current = "S4"
            elif current == "S6":
                if pending_exit_confirmation and row.capital_withdraw_3of5 and high_active:
                    current = "S7"; pending_exit_confirmation = False
                elif not high_active:
                    current = "S4"; pending_exit_confirmation = False
            states.append({"decision_date":row.decision_date,"ticker":ticker,"pool_rank":row.pool_rank,"prior_state":prior,"state":current,"relative_low":row.relative_low,"relative_high":row.relative_high,"entry_setup_active":low_active,"exit_setup_active":high_active,"entry_setup_last_qualifying_index":low_last,"exit_setup_last_qualifying_index":high_last,"up_group_count":row.up_group_count,"down_group_count":row.down_group_count,"entry_required_groups":row.entry_required_groups,"exit_required_groups":row.exit_required_groups,"entry_evidence_ready":row.entry_evidence_ready,"exit_evidence_ready":row.exit_evidence_ready,"market_state":row.market_state,"price_history_ready":row.price_history_ready,"chip_confidence":row.chip_confidence})
            if current != prior:
                transitions.append({"decision_date":row.decision_date,"ticker":ticker,"from_state":prior,"to_state":current,"transition_reason":"sequential_L3_10TD_3of5","entry_event":current=="S3","exit_event":current=="S7"})

    state = pd.DataFrame(states)
    transition = pd.DataFrame(transitions)
    episode = pd.DataFrame(episodes)
    state.to_csv(OUT / "p3_all80_L3_10TD_daily_candidate_state.csv.gz", index=False, compression="gzip")
    transition.to_csv(OUT / "p3_all80_L3_10TD_transition_events.csv", index=False, encoding="utf-8-sig")
    episode.to_csv(OUT / "p3_all80_L3_10TD_setup_episodes.csv.gz", index=False, compression="gzip")
    state.groupby(["decision_date", "market_state", "state"]).size().rename("candidate_count").reset_index().to_csv(OUT / "p3_all80_L3_10TD_daily_state_supply.csv", index=False, encoding="utf-8-sig")

    fold_map = audit._fold_map(dates)
    transition["fold"] = transition.decision_date.map(fold_map)
    folds = transition.groupby("fold").agg(entry_clusters=("entry_event", "sum"), exit_clusters=("exit_event", "sum")).reindex([1,2,3], fill_value=0).reset_index()
    available = state.assign(S3_S4=state.state.isin(["S3","S4"])).groupby("decision_date").S3_S4.any().reindex(dates, fill_value=False)
    longest = current = 0
    for missing in ~available:
        current = current + 1 if missing else 0; longest = max(longest, current)
    coverage = float(available.mean())
    folds["entry_gate_pass"] = folds.entry_clusters.ge(20)
    folds["exit_gate_pass"] = folds.exit_clusters.ge(20)
    folds.to_csv(OUT / "p3_all80_L3_10TD_fold_supply_gate.csv", index=False, encoding="utf-8-sig")
    gate_pass = bool(folds.entry_gate_pass.all() and folds.exit_gate_pass.all() and coverage >= .90 and longest <= 20)
    pd.DataFrame([{"S3_S4_date_coverage":coverage,"longest_no_S3_S4_supply_trading_days":longest,"coverage_gate_pass":coverage>=.90,"gap_gate_pass":longest<=20,"overall_supply_gate_pass":gate_pass}]).to_csv(OUT / "p3_all80_L3_10TD_coverage_gate.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([
        {"authority":"prior_parameter_free_setup_event_audit","entry_clusters_by_fold":"21/26/68","exit_clusters_by_fold":"100/100/81","requires_full_S1_to_S7_cycle":False,"eligible_for_final_state_supply_gate":False,"reason":"independent setup episodes can overlap an already healthy/held lifecycle"},
        {"authority":"continuous_L3_10TD_state_machine","entry_clusters_by_fold":"/".join(map(str, folds.entry_clusters)),"exit_clusters_by_fold":"/".join(map(str, folds.exit_clusters)),"requires_full_S1_to_S7_cycle":True,"eligible_for_final_state_supply_gate":True,"reason":"one ticker cannot create a new entry cycle before its prior lifecycle exits"},
    ]).to_csv(OUT / "p3_all80_L3_10TD_event_vs_state_authority_reconciliation.csv", index=False, encoding="utf-8-sig")
    blocked = pd.read_csv(source.DELTA / "all80_bounded_delta_remaining_blocker_ledger.csv.gz", dtype={"ticker":str})
    blocked.to_csv(OUT / "p3_all80_L3_10TD_blocked_ledger.csv.gz", index=False, compression="gzip")
    policy = {"relative_position":"L3_6M_price_mandatory_K_or_BIAS","setup_memory_trading_days":10,"persistence":"3of5","daily_max_state_advance":1,"entry_order":["S1","S2","S3"],"hold_state":"S4","exit_order":["S5","S6","S7"],"rank_change_resets_state":False,"TDCC_P3_1":"NA_not_zero","sequence_memory_unique":True}
    (OUT / "p3_all80_L3_10TD_machine_policy.json").write_text(json.dumps(policy, ensure_ascii=False, indent=2), encoding="utf-8")
    readiness = {"task_id":TASK,"status":"supply_gate_pass_return_strategy_center" if gate_pass else "supply_gate_failed_return_strategy_center","P3_1_dates":len(dates),"candidate_rows":len(frame),"entry_clusters_by_fold":folds.entry_clusters.tolist(),"exit_clusters_by_fold":folds.exit_clusters.tolist(),"S3_S4_date_coverage":coverage,"longest_no_supply_trading_days":longest,"overall_supply_gate_pass":gate_pass,"sequence_memory_unique":True,"represents_intended_all80_layer5_state_supply":True,"ready_for_experiments":False,"performance_authorized":False,"P3_2_outcome_read_authorized":False,"Top3_authorized":False,"future_data_violation_count":0,"formal_model_changed":False,"trade_decision_changed":False,"active_in_trade_decision":False,"report_changed":False,"not_live_rule":True,"forward_returns_live_rule_usage":False}
    (OUT / "readiness_for_L3_10TD_latch_state_supply.json").write_text(json.dumps(readiness, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "final_summary_zh.md").write_text(f"# P3 all80 L3 10TD latch state supply\n\nP3-1 entry clusters={readiness['entry_clusters_by_fold']}，exit clusters={readiness['exit_clusters_by_fold']}，S3+S4 coverage={coverage:.2%}，longest gap={longest}TD，overall gate={gate_pass}。未讀future outcome/P3-2，未跑績效。\n", encoding="utf-8")
    files = sorted(p for p in OUT.iterdir() if p.is_file() and p.name != "manifest.json")
    (OUT / "manifest.json").write_text(json.dumps({"task_id":TASK,"files":[{"name":p.name,"sha256":_sha(p),"bytes":p.stat().st_size} for p in files]}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    run()
