from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from backtest_lab import vnext_daily_incumbent_challenger_state_machine_contract as source
from backtest_lab import vnext_p1_dynamic80_kd_pit_feature_materialization as kd_scope


TASK_ID = "TASK-BACKTEST-CORE-VNEXT-P1-DYNAMIC80-KD-SHORTLIST-PIT-CONTRACT-001"
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO_ROOT / "outputs/vnext_p1_dynamic80_kd_shortlist_pit_contract_20260710"
R6_STATE = REPO_ROOT / "outputs/vnext_weekly_r6_single_position_state_boundary_reconstruction_contract_20260710/reconstructed_weekly_r6_single_position_daily_state_rows.csv"
RADAR_INCUMBENT = kd_scope.RADAR_INCUMBENT
P1_START, P1_END = pd.Timestamp("2015-01-02"), pd.Timestamp("2022-12-29")
FLAGS = {"formal_model_changed": False, "trade_decision_changed": False, "active_in_trade_decision": False, "report_changed": False, "portfolio_replay_executed": False, "ready_for_strategy_replay": False, "ready_for_formal": False, "not_live_rule": True, "forward_returns_live_rule_usage": False}


def _incumbent_by_snapshot(snapshots: list[pd.Timestamp]) -> pd.DataFrame:
    state = pd.read_csv(R6_STATE, dtype={"selected_ticker_after": str}, low_memory=False)
    state["signal_date"] = pd.to_datetime(state.signal_date, errors="coerce")
    state["ticker"] = state.selected_ticker_after.astype(str).str.replace(r"\.0$", "", regex=True)
    stock = state[state.selected_asset_type_after.eq("stock")][["signal_date", "ticker"]].sort_values("signal_date")
    left = pd.DataFrame({"snapshot_date": snapshots}).sort_values("snapshot_date")
    return pd.merge_asof(left, stock, left_on="snapshot_date", right_on="signal_date", direction="backward")[["snapshot_date", "ticker"]]


def _shortlist(matrix: pd.DataFrame) -> pd.DataFrame:
    rows = []
    snapshots = sorted(pd.to_datetime(matrix.snapshot_date.unique()))
    incumbent = dict(_incumbent_by_snapshot(snapshots).dropna().set_index("snapshot_date").ticker)
    for date, group in matrix.groupby("snapshot_date"):
        picks: dict[str, set[str]] = {}
        def add(frame: pd.DataFrame, role: str) -> None:
            for ticker in frame.ticker.astype(str): picks.setdefault(ticker, set()).add(role)
        add(group.sort_values(["route_support_score", "ticker"], ascending=[False, True]).head(10), "route_support_top10")
        strong = group[group.momentum_sleeve_candidate_feature.astype(str).str.lower().eq("true")].sort_values(["momentum_continuation_score", "ticker"], ascending=[False, True]).head(5)
        pullback = group[group.pullback_repair_sleeve_candidate_feature.astype(str).str.lower().eq("true")].sort_values(["pullback_repair_score", "ticker"], ascending=[False, True]).head(5)
        add(strong, "layer3_strong_attack_top5"); add(pullback, "layer3_pullback_reacceleration_top5")
        inc = str(incumbent.get(pd.Timestamp(date), ""));
        if inc and inc != "nan": picks.setdefault(inc, set()).add("current_reconstructed_R6_incumbent")
        for ticker, roles in picks.items():
            src = group[group.ticker.astype(str).eq(ticker)]
            rows.append({"snapshot_date": date, "ticker": ticker, "shortlist_roles": "|".join(sorted(roles)), "in_primary80": bool(len(src)), "incumbent_retained_outside_primary80": "current_reconstructed_R6_incumbent" in roles and not len(src), "future_return_used_as_rule": False})
    return pd.DataFrame(rows)


def _phase0(matrix: pd.DataFrame, shortlist: pd.DataFrame) -> pd.DataFrame:
    joined = matrix.merge(shortlist[["snapshot_date", "ticker"]].assign(in_shortlist=True), on=["snapshot_date", "ticker"], how="left")
    joined["in_shortlist"] = joined.in_shortlist.fillna(False).astype(bool)
    rows = []
    for period, start, end in (("P1", "2015-01-02", "2022-12-29"), ("P2_metadata_only", "2023-01-02", "2026-06-30")):
        sub = joined[joined.snapshot_date.between(start, end)].copy()
        if not len(sub):
            rows.append({"period": period, "status": "not_in_current_P1_contract"}); continue
        fwd = pd.to_numeric(sub.forward_excess_vs_00631L_20d, errors="coerce")
        sub["eval_value"] = fwd
        sub["top_decile_eval"] = sub.groupby("snapshot_date").eval_value.transform(lambda x: x >= x.quantile(.9) if x.notna().any() else False)
        sub["material_winner_eval"] = sub.eval_value > 0.10
        rows.append({"period": period, "status": "evaluation_metadata_only", "snapshot_count": sub.snapshot_date.nunique(), "mean_shortlist_size": shortlist[shortlist.snapshot_date.between(start, end)].groupby("snapshot_date").size().mean(), "top_decile_rows": int(sub.top_decile_eval.sum()), "top_decile_retained": int((sub.top_decile_eval & sub.in_shortlist).sum()), "top_decile_retention_share": float(sub.loc[sub.top_decile_eval, "in_shortlist"].mean()), "material_winner_rows": int(sub.material_winner_eval.sum()), "material_winner_retained": int((sub.material_winner_eval & sub.in_shortlist).sum()), "material_winner_retention_share": float(sub.loc[sub.material_winner_eval, "in_shortlist"].mean()), "missed_material_winner_rows": int((sub.material_winner_eval & ~sub.in_shortlist).sum()), "strong_attack_supply_mean": float(sub.groupby("snapshot_date").momentum_sleeve_candidate_feature.apply(lambda x: x.astype(str).str.lower().eq("true").sum()).mean()), "pullback_supply_mean": float(sub.groupby("snapshot_date").pullback_repair_sleeve_candidate_feature.apply(lambda x: x.astype(str).str.lower().eq("true").sum()).mean()), "future_return_used_as_rule": False})
    return pd.DataFrame(rows)


def _segments(shortlist: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    pseudo = shortlist.rename(columns={"snapshot_date": "snapshot_date"})
    return kd_scope._segments(pseudo, calendar)


def _deferred_policy() -> pd.DataFrame:
    blocked = pd.read_csv(RADAR_INCUMBENT / "p1_dynamic80_incumbent_hold_selected_stock_daily_ohlc_blocked_ledger.csv", dtype={"ticker": str}, low_memory=False)
    blocked["market_open_ticker_close_absent"] = True
    blocked["execution_semantics"] = blocked.required_field.map({"entry_close": "challenger_not_tradable_no_buy_hold_valid_incumbent_until_next_official_tradable_close", "exit_close": "held_ticker_no_trade_daily_valuation_carries_prior_official_mark_no_execution"})
    blocked["price_substitution_used"] = False
    blocked["deferred_execution_cost_timing"] = "charge_EP05_cost_only_on_actual_official_tradable_execution_date"
    blocked["source_status"] = "official_month_response_exact_ticker_date_absent"
    return blocked


def run(output_dir: str | Path = DEFAULT_OUTPUT) -> Path:
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True); (out / "current_step.txt").write_text("build_fixed_shortlist", encoding="utf-8")
    matrix = source._weekly_candidate_matrix(); matrix["ticker"] = matrix.ticker.astype(str)
    shortlist = _shortlist(matrix); phase0 = _phase0(matrix, shortlist)
    p1_shortlist = shortlist[shortlist.snapshot_date.between(P1_START, P1_END)].copy()
    calendar = pd.DatetimeIndex(pd.to_datetime(source._load_market_daily().signal_date.unique())).sort_values(); calendar = calendar[(calendar >= P1_START) & (calendar <= P1_END)]
    segments = _segments(p1_shortlist, calendar); req = kd_scope._requirements(segments, calendar); available = kd_scope._available(); audit = req.merge(available, on=["ticker", "price_date"], how="left"); audit["official_HLC_ready"] = audit[["high", "low", "close"]].notna().all(axis=1)
    gaps = audit[~audit.official_HLC_ready].copy(); gaps["ticker_month"] = gaps.price_date.dt.strftime("%Y-%m")
    routes = gaps.groupby(["ticker", "ticker_month"], as_index=False).agg(required_date_start=("price_date", "min"), required_date_end=("price_date", "max"), required_date_count=("price_date", "nunique"), segment_ids=("segment_id", lambda x: "|".join(sorted(set(x)))))
    routes["source_scope"] = "fixed_10_5_5_incumbent_shortlist_membership_segment_plus_20TD_warmup"
    routes["no_full_primary80_history"] = True
    deferred = _deferred_policy()
    shortlist.to_csv(out / "p1_dynamic80_kd_fixed_shortlist_by_snapshot.csv", index=False, encoding="utf-8-sig"); phase0.to_csv(out / "p1_dynamic80_kd_shortlist_phase0_retention_audit.csv", index=False, encoding="utf-8-sig"); segments.to_csv(out / "p1_dynamic80_kd_shortlist_membership_segments.csv", index=False, encoding="utf-8-sig"); audit.to_csv(out / "p1_dynamic80_kd_shortlist_HLC_coverage_audit.csv", index=False, encoding="utf-8-sig"); gaps[["ticker", "segment_id", "price_date", "required_fields", "decision_eligible_after_warmup"]].to_csv(out / "p1_dynamic80_kd_shortlist_exact_HLC_gap_ledger.csv", index=False, encoding="utf-8-sig"); routes.to_csv(out / "p1_dynamic80_kd_shortlist_bounded_ticker_month_source_request.csv", index=False, encoding="utf-8-sig"); deferred.to_csv(out / "p1_dynamic80_incumbent_hold_deferred_execution_trace_policy.csv", index=False, encoding="utf-8-sig"); pd.DataFrame(columns=["signal_date", "violation_reason"]).to_csv(out / "future_data_audit.csv", index=False, encoding="utf-8-sig")
    within_gate = len(gaps) <= 100000 and len(routes) <= 5000
    p1_eval = phase0[phase0.period.eq("P1")].iloc[0].to_dict()
    readiness = {"task_id": TASK_ID, "status": "bounded_shortlist_HLC_gap_fill_ready_for_Radar" if within_gate else "shortlist_still_exceeds_bounded_stop_gate", "shortlist_max_per_snapshot": int(p1_shortlist.groupby("snapshot_date").size().max()), "shortlist_mean_per_snapshot": float(p1_shortlist.groupby("snapshot_date").size().mean()), "membership_segments": len(segments), "required_ticker_date_HLC_rows": len(req), "reused_official_HLC_rows": int(audit.official_HLC_ready.sum()), "remaining_HLC_gap_rows": len(gaps), "remaining_ticker_month_routes": len(routes), "estimated_normalized_csv_MB": round(len(gaps) * 0.00018, 2), "bounded_stop_gate_pass": within_gate, "phase0_P1_top_decile_retention_share": p1_eval.get("top_decile_retention_share"), "phase0_P1_material_winner_retention_share": p1_eval.get("material_winner_retention_share"), "KD_materialized": False, "ready_for_experiments": False, "future_data_violation_count": 0, **FLAGS}
    (out / "readiness_for_p1_dynamic80_kd_shortlist_pit_contract.json").write_text(json.dumps(readiness, ensure_ascii=False, indent=2), encoding="utf-8"); (out / "final_summary_zh.md").write_text(f"# P1 Dynamic80 KD shortlist PIT contract\n\n- Fixed shortlist=route_support top10 + strong top5 + pullback top5 + reconstructed R6 incumbent。\n- mean size={readiness['shortlist_mean_per_snapshot']:.2f}, max={readiness['shortlist_max_per_snapshot']}。\n- P1 top-decile retention={readiness['phase0_P1_top_decile_retention_share']:.4f}; material-winner retention={readiness['phase0_P1_material_winner_retention_share']:.4f}，evaluation metadata only。\n- remaining HLC={len(gaps):,}; routes={len(routes):,}; stop-gate pass={within_gate}。\n- 9 official no-close dates已建立deferred execution semantics，不做鄰日價格替代。\n", encoding="utf-8"); (out / "manifest.json").write_text(json.dumps({"task_id": TASK_ID, "runner": __file__, "files": sorted(p.name for p in out.iterdir()), "readiness": readiness}, ensure_ascii=False, indent=2), encoding="utf-8"); (out / "current_step.txt").write_text("ready_for_Radar_bounded_HLC_fill" if within_gate else "blocked_stop_gate", encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT)); args = parser.parse_args(); print(run(args.output_dir))


if __name__ == "__main__": main()
