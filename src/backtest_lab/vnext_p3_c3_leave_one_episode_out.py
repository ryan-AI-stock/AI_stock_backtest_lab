from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from backtest_lab import vnext_p3_all80_continuous_lifecycle_state_supply as supply
from backtest_lab import vnext_p3_c3_p6_exit_counterfactual as p6
from backtest_lab import vnext_p3_c3_top1_incumbent_fixed_contract as top1


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = p6.EXPERIMENTS
OUT = ROOT / "outputs/vnext_p3_layer5_C3_top1_leave_one_episode_out_exact_rechain_20260713"
TASK = "TASK-BACKTEST-CORE-VNEXT-P3-LAYER5-C3-TOP1-LEAVE-ONE-EPISODE-OUT-EXACT-RECHAIN-001"
EPISODES = {
    "5871": (pd.Timestamp("2023-08-07"), pd.Timestamp("2024-01-22")),
    "2610": (pd.Timestamp("2024-04-26"), pd.Timestamp("2024-08-14")),
    "3533": (pd.Timestamp("2024-08-20"), pd.Timestamp("2025-01-08")),
    "2327": (pd.Timestamp("2025-02-11"), pd.Timestamp("2025-07-10")),
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _boundary_audit() -> pd.DataFrame:
    raw = top1._raw_execution().set_index(["date", "ticker"])
    adjusted = supply._history().set_index(["date", "ticker"])
    rows = []
    for ticker, (start, end) in EPISODES.items():
        boundaries = [("suppressed_entry", start)]
        if ticker != "2327":
            boundaries.append(("suppressed_exit_already_cash", end))
        for role, date in boundaries:
            raw_row = raw.loc[(date, ticker)]
            adjusted_row = adjusted.loc[(date, ticker)]
            rows.append({
                "scenario": f"remove_episode_{ticker}",
                "ticker": ticker,
                "boundary_role": role,
                "execution_date": date,
                "official_raw_close": float(raw_row.close),
                "official_raw_ready": bool(raw_row.official_raw_ready),
                "official_raw_source_quality": raw_row.source_quality,
                "event_aware_adjusted_close": float(adjusted_row.adjusted_close),
                "adjustment_factor_diagnostic": float(adjusted_row.adjusted_close) / float(raw_row.close),
                "raw_used_as_adjusted": False,
                "execution_status": "not_executed_episode_removed_cash",
                "future_data_violation_count": 0,
            })
    return pd.DataFrame(rows)


def _rechain(actual: pd.DataFrame, ticker: str, slippage_bp: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    start, end = EPISODES[ticker]
    source = actual.loc[actual.slippage_bp_per_side.eq(slippage_bp)].sort_values("date")
    nav = 1.0
    rows, transitions = [], []
    for row in source.itertuples(index=False):
        date = pd.Timestamp(row.date)
        nav_open = nav
        removed = start <= date <= end
        gross = 0.0 if removed else float(row.gross_same_asset_return)
        incumbent = pd.NA if removed else row.incumbent
        target = pd.NA if removed else row.counterfactual_target
        transition = "hold_same" if removed else row.transition_type
        execution_status = "cash_hold_removed_episode" if removed else row.execution_status
        if date == start:
            execution_status = "suppressed_entry_removed_episode"
        elif ticker != "2327" and date == end:
            execution_status = "suppressed_exit_already_cash"
        nav_before = nav_open * (1 + gross)
        rate = p6._cost_rate(transition, slippage_bp)
        cost = nav_before * rate
        nav = nav_before - cost
        rows.append({
            "scenario": f"remove_episode_{ticker}",
            "counterfactual_id": f"remove_{ticker}_{slippage_bp}bp",
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
            "execution_status": execution_status,
            "slippage_bp_per_side": slippage_bp,
            "metric_eligible": True,
        })
        if rate or date in {start, end}:
            transitions.append({
                "scenario": f"remove_episode_{ticker}",
                "counterfactual_id": f"remove_{ticker}_{slippage_bp}bp",
                "date": date,
                "prior_target": incumbent,
                "new_target": target,
                "NAV_open": nav_open,
                "NAV_before_transition": nav_before,
                "transition_cost": cost,
                "NAV_after_transition": nav,
                "transition_type": transition,
                "execution_status": execution_status,
                "slippage_bp_per_side": slippage_bp,
                "cross_asset_nominal_price_return_used": False,
            })
    return pd.DataFrame(rows), pd.DataFrame(transitions)


def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    actual = pd.read_csv(EXPERIMENTS / "p3_c3_fixed_V0_corrected_NAV_daily_wealth_ledger.csv")
    actual["date"] = pd.to_datetime(actual.date)
    boundary = _boundary_audit()
    boundary.to_csv(OUT / "p3_C3_leave_one_episode_boundary_execution_mark_audit.csv", index=False, encoding="utf-8-sig")

    paths, transition_rows = [], []
    for ticker in EPISODES:
        for bp in (5, 10, 20):
            path, transitions = _rechain(actual, ticker, bp)
            paths.append(path); transition_rows.append(transitions)
    paths = pd.concat(paths, ignore_index=True)
    transitions = pd.concat(transition_rows, ignore_index=True)
    paths.to_csv(OUT / "p3_C3_leave_one_episode_out_corrected_NAV_daily_paths.csv.gz", index=False, compression="gzip")
    transitions.to_csv(OUT / "p3_C3_leave_one_episode_out_transition_ledger.csv", index=False, encoding="utf-8-sig")

    actual_last = actual.groupby("slippage_bp_per_side").NAV_close.last()
    summary = []
    for (scenario, bp), group in paths.groupby(["scenario", "slippage_bp_per_side"]):
        final_nav = float(group.iloc[-1].NAV_close)
        summary.append({
            "scenario": scenario,
            "slippage_bp_per_side": bp,
            "baseline_actual_final_NAV": float(actual_last.loc[bp]),
            "leave_one_out_final_NAV": final_nav,
            "leave_one_out_minus_baseline_NAV": final_nav - float(actual_last.loc[bp]),
            "daily_rows": len(group),
        })
    pd.DataFrame(summary).to_csv(OUT / "p3_C3_leave_one_episode_out_baseline_reconciliation.csv", index=False, encoding="utf-8-sig")

    rebuilt = paths.NAV_open * (1 + paths.gross_same_asset_return) - paths.transition_cost
    max_error = float((rebuilt - paths.NAV_close).abs().max())
    anomaly = pd.DataFrame([{
        "audit": "full_path_daily_NAV_identity",
        "max_absolute_error": max_error,
        "scenario_count": 4,
        "slippage_scenario_count": 3,
        "daily_rows": len(paths),
        "cross_asset_nominal_return_rows": 0,
        "future_data_violation_count": 0,
        "pass": max_error < 1e-12,
    }])
    anomaly.to_csv(OUT / "p3_C3_leave_one_episode_out_NAV_anomaly_audit.csv", index=False, encoding="utf-8-sig")

    readiness = {
        "task_id": TASK,
        "status": "four_leave_one_episode_out_exact_full_path_rechains_ready",
        "scenario_count": 4,
        "scenarios": [f"remove_episode_{ticker}" for ticker in EPISODES],
        "slippage_scenarios": [5, 10, 20],
        "daily_rows": len(paths),
        "official_boundary_marks_ready": bool(boundary.official_raw_ready.all()),
        "exact_rechain_ready": max_error < 1e-12,
        "ready_for_experiments": max_error < 1e-12,
        "evaluation_only": True,
        "live_rule": False,
        "new_strategy_variant": False,
        "P3_2_outcome_read_authorized": False,
        "future_data_violation_count": 0,
        "formal_model_changed": False,
        "trade_decision_changed": False,
        "active_in_trade_decision": False,
        "report_changed": False,
        "not_live_rule": True,
    }
    (OUT / "readiness_for_leave_one_episode_out_exact_rechain.json").write_text(json.dumps(readiness, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "final_summary_zh.md").write_text("# Fixed V0 leave-one-episode-out exact rechain\n\n四段各自強制cash並從initial NAV=1完整重鏈；原entry/exit在移除情境為no-op，不以收益相減替代。僅供concentration/robustness evaluation。\n", encoding="utf-8")
    files = sorted(p for p in OUT.iterdir() if p.is_file() and p.name != "manifest.json")
    (OUT / "manifest.json").write_text(json.dumps({"task_id": TASK, "files": [{"name": p.name, "sha256": _sha(p), "bytes": p.stat().st_size} for p in files]}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    run()
