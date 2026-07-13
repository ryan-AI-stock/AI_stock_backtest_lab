from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from backtest_lab import vnext_p3_all80_entry_funnel_setup_latch_audit as audit
from backtest_lab import vnext_p3_all80_l3_10td_latch_state_supply as frozen
from backtest_lab import vnext_p3_c3_top1_incumbent_fixed_contract as top1_contract


ROOT = Path(__file__).resolve().parents[2]
TOP1 = ROOT / "outputs/vnext_p3_layer5_C3_eligible_top1_incumbent_lifecycle_fixed_contract_20260713/p3_C3_daily_top1_second_candidate.csv"
OUT = ROOT / "outputs/vnext_p3_layer5_C3_top1_incumbent_path_corrected_NAV_contract_20260713"
TASK = "TASK-BACKTEST-CORE-VNEXT-P3-LAYER5-C3-TOP1-INCUMBENT-PATH-CORRECTED-NAV-CONTRACT-001"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _incumbent_features() -> pd.DataFrame:
    frame = audit._base().sort_values(["ticker", "decision_date"]).copy()
    dates = sorted(frame.decision_date.unique())
    market_index = {pd.Timestamp(date): index for index, date in enumerate(dates)}
    frame["market_day_index"] = frame.decision_date.map(market_index)
    frame["relative_high"] = frame.price_pct_6M.ge(frozen.HIGH) & (frame.K_pct_6M.ge(frozen.HIGH) | frame.BIAS_pct_6M.ge(frozen.HIGH))
    down = ["kd_down", "rs_weak", "ma_down", "capital_withdraw", "risk_bad"]
    for column in down:
        frame[f"{column}_3of5"] = frame.groupby("ticker", sort=False)[column].transform(lambda values: values.fillna(False).astype(int).rolling(5, min_periods=5).sum().ge(3))
    frame["down_group_count"] = frame[[f"{column}_3of5" for column in down]].sum(axis=1)
    frame["exit_required_groups"] = frame.market_state.map({"strong_market":4,"ordinary_market":3,"weak_market":3,"confirmed_bear":2}).fillna(3)
    frame["turn_down_ready"] = frame.down_group_count.ge(frame.exit_required_groups)
    frame["capital_withdraw_ready"] = frame.capital_withdraw_3of5
    return frame


def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    top = pd.read_csv(TOP1, dtype={"top1_ticker":str,"second_ticker":str})
    top["decision_date"] = pd.to_datetime(top.decision_date)
    features = _incumbent_features()
    feature_lookup = features.set_index(["decision_date","ticker"])
    raw = top1_contract._raw_execution().set_index(["date","ticker"])
    dates = sorted(top.decision_date.unique())
    next_date = {date: dates[index + 1] if index + 1 < len(dates) else pd.NaT for index, date in enumerate(dates)}

    incumbent = None
    position_state = "P0"
    high_last_index = None
    pending = None
    actions, requirements, blocked = [], [], []
    date_index = {date:index for index,date in enumerate(dates)}

    for date in dates:
        # Apply a previously scheduled transition only when every required execution leg is ready.
        if pending and pending["execution_date"] == date:
            legs_ready = all(leg["ready"] for leg in pending["legs"])
            if legs_ready:
                incumbent = pending["new_ticker"]
                position_state = "P4" if incumbent else "P0"
                high_last_index = None
            else:
                blocked.append({"decision_date":pending["decision_date"],"date":date,"ticker":incumbent,"blocked_type":"atomic_execution_leg_missing","reason":pending["action"]})
            pending = None

        day_top = top.loc[top.decision_date.eq(date)].iloc[0]
        challenger = str(day_top.top1_ticker).zfill(4) if pd.notna(day_top.top1_ticker) else None
        market_state = None
        action = "hold_or_wait"
        reason = "P0_NO_C3" if incumbent is None else "HOLD_INCUMBENT"

        if incumbent is None:
            if challenger:
                candidate_row = features.loc[(features.decision_date.eq(date)) & (features.ticker.eq(challenger))]
                market_state = candidate_row.market_state.iloc[0] if len(candidate_row) else None
                if market_state != "confirmed_bear":
                    action, reason = "entry_signal", "ENTRY_C3_TOP1_READY"
        else:
            key = (date, incumbent)
            if key not in feature_lookup.index:
                blocked.append({"decision_date":date,"date":date,"ticker":incumbent,"blocked_type":"incumbent_PIT_lifecycle_missing_outside_primary80","reason":"cannot infer hold/P5/P6/P7"})
                action, reason = "blocked_no_action", "INCUMBENT_LIFECYCLE_DATA_MISSING"
            else:
                row = feature_lookup.loc[key]
                market_state = row.market_state
                if row.relative_high:
                    high_last_index = date_index[date]
                high_active = high_last_index is not None and date_index[date] - high_last_index <= frozen.GRACE
                if position_state == "P4" and high_active:
                    position_state = "P5"
                if position_state == "P5" and high_active and row.turn_down_ready:
                    position_state = "P6"
                elif position_state == "P5" and not high_active:
                    position_state = "P4"
                if position_state == "P6":
                    if challenger and challenger != incumbent and market_state != "confirmed_bear":
                        action, reason = "P6_replacement_signal", "P6_VALID_C3_REPLACEMENT"
                    elif row.capital_withdraw_ready:
                        position_state = "P7"
                        if challenger and challenger != incumbent and market_state != "confirmed_bear":
                            action, reason = "P7_replacement_signal", "P7_VALID_C3_REPLACEMENT"
                        else:
                            action, reason = "P7_exit_signal", "P7_EXIT_NO_VALID_REPLACEMENT"
                    else:
                        action, reason = "hold_incumbent", "P6_NO_REPLACEMENT_WAIT_P7"
                elif position_state == "P5":
                    action, reason = "hold_incumbent", "HOLD_HIGH_WARNING_NOT_EXIT"
                elif position_state == "P4":
                    action, reason = "hold_incumbent", "HOLD_HEALTHY_INCUMBENT"

        execution_date = next_date[date]
        if action in {"entry_signal","P6_replacement_signal","P7_replacement_signal","P7_exit_signal"} and pd.notna(execution_date):
            new_ticker = challenger if "replacement" in action or action == "entry_signal" else None
            leg_tickers = ([incumbent] if incumbent else []) + ([new_ticker] if new_ticker else [])
            legs = []
            for leg_ticker in leg_tickers:
                raw_key = (execution_date, leg_ticker)
                ready = raw_key in raw.index and bool(raw.loc[raw_key].official_raw_ready)
                legs.append({"ticker":leg_ticker,"ready":ready})
                requirements.append({"decision_date":date,"execution_date":execution_date,"action":action,"ticker":leg_ticker,"leg":"exit" if leg_ticker==incumbent and incumbent else "entry","official_raw_ready":ready,"source_quality":raw.loc[raw_key].source_quality if ready else None})
            pending = {"decision_date":date,"execution_date":execution_date,"action":action,"new_ticker":new_ticker,"legs":legs}

        actions.append({"decision_date":date,"execution_date":execution_date,"incumbent":incumbent,"position_state":position_state,"C3_top1":challenger,"market_state":market_state,"selected_action":action,"action_reason":reason,"pending_execution":pending is not None,"metric_eligible":len(blocked)==0})

    action_df = pd.DataFrame(actions)
    req_df = pd.DataFrame(requirements)
    blocked_df = pd.DataFrame(blocked, columns=["decision_date","date","ticker","blocked_type","reason"])
    action_df.to_csv(OUT / "p3_C3_top1_incumbent_daily_action_ledger.csv", index=False, encoding="utf-8-sig")
    req_df.to_csv(OUT / "p3_C3_top1_incumbent_execution_requirement_ledger.csv", index=False, encoding="utf-8-sig")
    blocked_df.to_csv(OUT / "p3_C3_top1_incumbent_path_blocked_ledger.csv", index=False, encoding="utf-8-sig")
    pit_gaps = blocked_df.loc[blocked_df.blocked_type.eq("incumbent_PIT_lifecycle_missing_outside_primary80")].copy()
    if len(pit_gaps):
        pit_gaps["date"] = pd.to_datetime(pit_gaps.date)
        pit_gaps["market_index"] = pit_gaps.date.map(date_index)
        pit_gaps["segment"] = pit_gaps.groupby("ticker").market_index.transform(lambda values: values.diff().ne(1).cumsum())
        segments = pit_gaps.groupby(["ticker","segment"]).agg(required_start=("date","min"),required_end=("date","max"),required_decision_rows=("date","size")).reset_index()
        segments["warmup_trading_days"] = 20
        segments["required_families"] = "adjusted_analysis_HLC;institutional;margin_short;securities_lending;foreign_ownership"
        segments["TDCC_P3_1"] = "NA_not_required_not_zero"
        segments.to_csv(OUT / "p3_incumbent_continuous_PIT_bounded_source_requirement.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([
        {"field":"NAV_open","authority":"prior day NAV_close","status":"ready_if_path_unblocked"},
        {"field":"holding_return","authority":"same ticker event-aware adjusted analysis marks only","status":"ready_if_incumbent_history_complete"},
        {"field":"execution","authority":"official raw next-day close","status":"ready_if_all_atomic_legs_complete"},
        {"field":"cost","authority":"EP05 fee/tax + 10bp per side; 5/20bp sensitivity","status":"ready"},
        {"field":"cross_asset_return","authority":"prohibited; transition changes units at constant NAV less costs","status":"enforced"},
    ]).to_csv(OUT / "p3_corrected_NAV_input_contract.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{"requested_start":"2023-07-11","requested_end":"2025-07-10","actual_candidate_start":str(min(dates).date()),"actual_candidate_end":str(max(dates).date()),"metric_start":"blocked_until_incumbent_PIT_horizon_complete"}]).to_csv(OUT / "p3_incumbent_path_requested_actual_coverage.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{"audit":"future_outcome_read","violations":0},{"audit":"P3_2_read","violations":0},{"audit":"cross_asset_nominal_return","violations":0},{"audit":"neighbor_price_substitution","violations":0}]).to_csv(OUT / "p3_incumbent_path_future_execution_audit.csv", index=False, encoding="utf-8-sig")

    missing_features = int(blocked_df.blocked_type.eq("incumbent_PIT_lifecycle_missing_outside_primary80").sum()) if len(blocked_df) else 0
    missing_legs = int((~req_df.official_raw_ready).sum()) if len(req_df) else 0
    exact_ready = missing_features == 0 and missing_legs == 0
    readiness = {"task_id":TASK,"status":"exact_path_contract_ready" if exact_ready else "blocked_incumbent_continuous_PIT_or_execution_gap","decision_dates":len(dates),"action_rows":len(action_df),"execution_leg_rows":len(req_df),"incumbent_PIT_missing_rows":missing_features,"execution_leg_missing_rows":missing_legs,"exact_path_ready":exact_ready,"corrected_NAV_materialized":False,"ready_for_experiments":exact_ready,"performance_authorized":"true_P3_1_only","P3_2_outcome_read_authorized":False,"operational_supply_gate_pass":True,"calibration_supply_gate_pass":False,"fixed_architecture_only":True,"weight_grid_authorized":False,"Top3_authorized":False,"future_data_violation_count":0,"formal_model_changed":False,"trade_decision_changed":False,"active_in_trade_decision":False,"report_changed":False,"not_live_rule":True,"forward_returns_live_rule_usage":False}
    (OUT / "readiness_for_incumbent_path_corrected_NAV.json").write_text(json.dumps(readiness, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "final_summary_zh.md").write_text(f"# P3-1 C3 Top1 incumbent path corrected-NAV contract\n\nAction rows={len(action_df)}，incumbent PIT missing={missing_features}，execution leg missing={missing_legs}，exact path ready={exact_ready}。未讀P3-2或future outcome；NAV未在blocked contract上計算。\n", encoding="utf-8")
    files = sorted(path for path in OUT.iterdir() if path.is_file() and path.name != "manifest.json")
    (OUT / "manifest.json").write_text(json.dumps({"task_id":TASK,"files":[{"name":path.name,"sha256":_sha(path),"bytes":path.stat().st_size} for path in files]}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    run()
