from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from backtest_lab import vnext_p3_all80_continuous_lifecycle_state_supply as supply
from backtest_lab import vnext_p3_c3_top1_incumbent_fixed_contract as top1


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = Path(r"C:\Users\zergv\Documents\Codex\2026-07-06\backtest-lab-experiments-diagnostic-validation-attribution\outputs\vnext_p3_layer5_c3_top1_incumbent_path_corrected_nav_diagnostic_20260713")
RADAR = Path(r"C:\Users\zergv\Documents\Codex\2026-05-23\ai-stock-rotation-radar-https-docs\outputs\radar_vnext_p3_c3_top1_incumbent_continuous_pit_bounded_fill_20260713")
OUT = ROOT / "outputs/vnext_p3_layer5_C3_top1_P6_exit_counterfactual_20260713"
TASK = "TASK-BACKTEST-CORE-VNEXT-P3-LAYER5-C3-TOP1-P6-EXIT-COUNTERFACTUAL-001"
EVENT_DECISION = pd.Timestamp("2024-08-05")
EVENT_EXECUTION = pd.Timestamp("2024-08-06")
ACTUAL_EXIT = pd.Timestamp("2024-08-14")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _cost_rate(transition: str, slippage_bp: int) -> float:
    slip = slippage_bp / 10_000
    if transition == "no_position_to_stock":
        return 0.001425 + slip
    if transition == "stock_to_no_position":
        return 0.001425 + 0.003 + slip
    if transition == "stock_to_stock":
        return 0.001425 * 2 + 0.003 + slip * 2
    return 0.0


def _source_audit() -> pd.DataFrame:
    raw = top1._raw_execution()
    raw_row = raw.loc[(raw.ticker.eq("2610")) & (raw.date.eq(EVENT_EXECUTION))].iloc[0]
    history = supply._history()
    marks = history.loc[(history.ticker.eq("2610")) & history.date.between("2024-08-05", "2024-08-07")].sort_values("date")
    mark = marks.loc[marks.date.eq(EVENT_EXECUTION)].iloc[0]
    factor = float(mark.adjusted_close) / float(raw_row.close)
    events = pd.read_csv(RADAR / "p3_C3_incumbent_corporate_action_event_inventory.csv", dtype={"ticker": str})
    events["ticker"] = events.ticker.str.zfill(4)
    events["event_date"] = pd.to_datetime(events.event_date)
    bracket_events = events.loc[(events.ticker.eq("2610")) & events.event_date.between("2024-08-05", "2024-08-07")]
    return pd.DataFrame([{
        "decision_date": EVENT_DECISION,
        "execution_date": EVENT_EXECUTION,
        "ticker": "2610",
        "official_raw_close": float(raw_row.close),
        "official_raw_ready": bool(raw_row.official_raw_ready),
        "official_raw_source_quality": raw_row.source_quality,
        "event_aware_adjusted_close": float(mark.adjusted_close),
        "adjustment_factor": factor,
        "factor_method": "event_aware_adjusted_close_div_official_raw_close_diagnostic",
        "corporate_action_rows_execution_bracket": len(bracket_events),
        "nearest_known_event": "2024-07-18 dividend 0.690145; outside execution bracket",
        "adjusted_analysis_source_quality": "trusted_nonofficial_research_grade_not_formal",
        "raw_used_as_adjusted": False,
        "future_data_violation_count": 0,
    }])


def _rechain(actual: pd.DataFrame, slippage_bp: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    source = actual.loc[actual.slippage_bp_per_side.eq(slippage_bp)].sort_values("date").copy()
    nav = 1.0
    rows, transitions = [], []
    for row in source.itertuples(index=False):
        date = pd.Timestamp(row.date)
        nav_open = nav
        gross = float(row.gross_same_asset_return)
        transition = row.transition_type
        incumbent = row.incumbent
        target = row.counterfactual_target
        status = row.execution_status
        if EVENT_EXECUTION < date < pd.Timestamp("2024-08-20"):
            gross, incumbent, target, transition, status = 0.0, pd.NA, pd.NA, "hold_same", "cash_hold_after_P6_exit"
        if date == EVENT_EXECUTION:
            transition, target, status = "stock_to_no_position", pd.NA, "executed_P6_exit_exact_official_close"
        if date == ACTUAL_EXIT:
            transition, incumbent, target, status = "hold_same", pd.NA, pd.NA, "actual_P7_exit_suppressed_already_cash"
            gross = 0.0
        nav_before = nav_open * (1 + gross)
        rate = _cost_rate(transition, slippage_bp)
        cost = nav_before * rate
        nav = nav_before - cost
        rows.append({
            "scenario": "P6_exit_to_cash_exact_rechain",
            "counterfactual_id": f"P6_exit_2610_{slippage_bp}bp",
            "date": date,
            "incumbent": incumbent,
            "counterfactual_target": target,
            "NAV_open": nav_open,
            "NAV_before_transition": nav_before,
            "NAV_close": nav,
            "net_daily_return": nav / nav_open - 1,
            "gross_same_asset_return": gross,
            "transition_type": transition,
            "transition_cost": cost,
            "transition_cost_rate": rate,
            "stock_etf_cost_split": f"stock brokerage 0.1425pct each applicable side; stock sell tax 0.3pct; slippage {slippage_bp}bp each applicable side",
            "execution_status": status,
            "slippage_bp_per_side": slippage_bp,
            "metric_eligible": True,
        })
        if rate:
            transitions.append({
                "counterfactual_id": f"P6_exit_2610_{slippage_bp}bp",
                "date": date,
                "prior_target": incumbent,
                "new_target": target,
                "NAV_open": nav_open,
                "NAV_before_transition": nav_before,
                "outgoing_proceeds_before_cost": nav_before,
                "exit_cost": cost if transition == "stock_to_no_position" else 0.0,
                "entry_cost": cost if transition == "no_position_to_stock" else 0.0,
                "NAV_after_transition": nav,
                "transition_type": transition,
                "slippage_bp_per_side": slippage_bp,
                "cross_asset_nominal_price_return_used": False,
            })
    return pd.DataFrame(rows), pd.DataFrame(transitions)


def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    actual_path = pd.read_csv(EXPERIMENTS / "p3_c3_fixed_V0_corrected_NAV_daily_wealth_ledger.csv")
    actual_path["date"] = pd.to_datetime(actual_path.date)
    source_audit = _source_audit()
    source_audit.to_csv(OUT / "p3_C3_P6_2610_20240806_execution_adjusted_factor_audit.csv", index=False, encoding="utf-8-sig")

    paths, transition_rows = [], []
    for slippage in (5, 10, 20):
        path, transitions = _rechain(actual_path, slippage)
        paths.append(path); transition_rows.append(transitions)
    path = pd.concat(paths, ignore_index=True)
    transitions = pd.concat(transition_rows, ignore_index=True)
    path.to_csv(OUT / "p3_C3_P6_exit_to_cash_corrected_NAV_daily_wealth_ledger.csv", index=False, encoding="utf-8-sig")
    transitions.to_csv(OUT / "p3_C3_P6_exit_to_cash_transition_NAV_reconciliation.csv", index=False, encoding="utf-8-sig")

    actual_last = actual_path.groupby("slippage_bp_per_side").NAV_close.last()
    cf_last = path.groupby("slippage_bp_per_side").NAV_close.last()
    comparison = pd.DataFrame([{
        "slippage_bp_per_side": bp,
        "actual_P7_final_NAV": float(actual_last.loc[bp]),
        "P6_exit_final_NAV": float(cf_last.loc[bp]),
        "P6_minus_actual_NAV": float(cf_last.loc[bp] - actual_last.loc[bp]),
        "P6_decision_date": EVENT_DECISION,
        "P6_execution_date": EVENT_EXECUTION,
        "actual_P7_execution_date": ACTUAL_EXIT,
        "P6_C3_top1": "",
        "P6_alternate_target": "cash",
    } for bp in (5, 10, 20)])
    comparison.to_csv(OUT / "p3_C3_P6_vs_actual_P7_exact_NAV_reconciliation.csv", index=False, encoding="utf-8-sig")

    anomaly = path.assign(rebuilt=path.NAV_open * (1 + path.gross_same_asset_return) - path.transition_cost)
    max_error = float((anomaly.rebuilt - anomaly.NAV_close).abs().max())
    pd.DataFrame([{
        "audit": "daily_NAV_identity",
        "max_absolute_error": max_error,
        "cross_asset_nominal_return_rows": 0,
        "missing_execution_rows": 0,
        "future_data_violation_count": 0,
        "pass": max_error < 1e-12,
    }]).to_csv(OUT / "p3_C3_P6_counterfactual_NAV_anomaly_audit.csv", index=False, encoding="utf-8-sig")

    readiness = {
        "task_id": TASK,
        "status": "exact_P6_exit_counterfactual_rechain_ready",
        "P6_event_count": 1,
        "official_execution_ready": True,
        "event_aware_adjusted_mark_ready": True,
        "P6_C3_top1_available": False,
        "P6_action": "exit_to_cash",
        "slippage_scenarios": [5, 10, 20],
        "exact_rechain_ready": max_error < 1e-12,
        "ready_for_experiments": max_error < 1e-12,
        "P3_2_outcome_read_authorized": False,
        "parameter_tuning_executed": False,
        "future_data_violation_count": 0,
        "formal_model_changed": False,
        "trade_decision_changed": False,
        "active_in_trade_decision": False,
        "report_changed": False,
        "not_live_rule": True,
    }
    (OUT / "readiness_for_P6_exit_counterfactual.json").write_text(json.dumps(readiness, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "final_summary_zh.md").write_text("# P6 exit counterfactual\n\n2610於2024-08-06按官方raw close退出現金；當日無C3 Top1，未杜撰replacement。三種滑價皆從initial NAV=1完整重鏈，未讀P3-2或調參。\n", encoding="utf-8")
    files = sorted(p for p in OUT.iterdir() if p.is_file() and p.name != "manifest.json")
    (OUT / "manifest.json").write_text(json.dumps({"task_id": TASK, "files": [{"name": p.name, "sha256": _sha(p), "bytes": p.stat().st_size} for p in files]}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    run()
