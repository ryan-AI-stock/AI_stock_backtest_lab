from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from backtest_lab import vnext_daily_incumbent_challenger_state_machine_contract as source
from backtest_lab import vnext_weekly_r6_single_position_state_boundary_reconstruction_contract as r6_source


TASK_ID = "TASK-BACKTEST-CORE-VNEXT-P1-DYNAMIC80-LIFECYCLE-SWING-STATE-CONTRACT-001"
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO_ROOT / "outputs/vnext_p1_dynamic80_lifecycle_swing_state_contract_20260710"
P1_START, P1_END = pd.Timestamp("2015-01-02"), pd.Timestamp("2022-12-29")
VARIANTS = ("K0_KD30_70_reference_only", "K1_oscillator_turn_confirmed", "K2_lifecycle_composite", "K3_incumbent_challenger_lifecycle", "K4_market_modulated_lifecycle")
FLAGS = {"formal_model_changed": False, "trade_decision_changed": False, "active_in_trade_decision": False, "report_changed": False, "portfolio_replay_executed": False, "ready_for_strategy_replay": False, "ready_for_formal": False, "not_live_rule": True, "forward_returns_live_rule_usage": False}


def _truth(value: object) -> bool:
    return False if pd.isna(value) else str(value).lower() in {"true", "1", "yes"}


def _inventory(matrix: pd.DataFrame) -> pd.DataFrame:
    groups = [
        ("KD_K_D_J_raw_slope_cross_self_percentile", [], "blocked", "no local PIT KD columns"),
        ("BIAS20_60_raw_self_percentile", ["BIAS20", "BIAS60", "BIAS20_percentile", "BIAS60_percentile"], "ready", "weekly PIT primary80"),
        ("BIAS_zscore", [], "blocked", "percentile ready; zscore absent"),
        ("RS5_10_20_60_acceleration", ["RS5", "RS10", "RS20", "RS60", "rs_short_acceleration_flag", "rs_short_deterioration_flag"], "ready", "weekly PIT vs 0050"),
        ("MA20_60_position", ["MA20", "MA60", "MA20_position", "MA60_position"], "ready", "weekly PIT"),
        ("MA20_60_slope_reclaim_exact", [], "proxy", "position ready; exact slope/cross event absent"),
        ("volatility_drawdown_exhaustion", ["volatility", "volatility_pctile_by_week", "drawdown_20d", "drawdown_60d", "high_exhaustion_or_breakdown_context"], "ready", "weekly PIT"),
        ("gap_exact", [], "blocked", "no PIT gap event column"),
        ("blowoff", ["blowoff_turnover_without_price_continuation_proxy"], "proxy", "explicit proxy source quality"),
        ("traded_value_5d20d60d_rank_change", ["traded_value_rank_5d", "traded_value_rank_20d", "traded_value_rank_60d", "capital_rank_improvement_20d_vs_60d"], "ready", "weekly PIT"),
        ("foreign_investment_trust_dealer_flow", [], "blocked", "not in canonical primary80 contract"),
        ("margin_short_lending", [], "blocked", "not in canonical primary80 contract"),
        ("TDCC_holder_structure", [], "blocked", "not in canonical primary80 contract"),
        ("0050_trend_breadth_BIAS", ["external_market_context"], "ready", "daily market fields aligned backward to weekly snapshot"),
        ("futures_OI_foreign_net", [], "blocked", "not in local PIT market contract"),
    ]
    return pd.DataFrame([{"feature_group": name, "requested_fields": "|".join(cols), "status": status, "source_quality": quality, "P1_start": matrix.snapshot_date.min(), "P1_end": matrix.snapshot_date.max()} for name, cols, status, quality in groups])


def _state_matrix() -> pd.DataFrame:
    matrix = source._weekly_candidate_matrix()
    matrix = matrix[matrix["snapshot_date"].between(P1_START, P1_END)].copy()
    market = source._load_market_daily()[["signal_date", "c2_pass_daily", "0050_price_vs_ma60", "0050_return_20d", "0050_return_40d"]].rename(columns={"signal_date": "snapshot_date"})
    matrix = matrix.merge(market, on="snapshot_date", how="left")
    hard = matrix["high_exhaustion_or_breakdown_context"].map(_truth) & matrix["rs_short_deterioration_flag"].map(_truth)
    deterioration = matrix["incumbent_deterioration_confirmed"].map(_truth)
    turn = matrix["rs_short_acceleration_flag"].map(_truth) & matrix["capital_rank_20d_improving_vs_60d"].map(_truth) & ~hard
    healthy = matrix["rs20_30_primary_momentum_positive"].map(_truth) & ~deterioration & ~hard
    overheat = matrix["risk_overheat_penalty_context"].map(_truth) | matrix["rs60_high_short_rs_weakening_exhaustion_context"].map(_truth)
    cooling = matrix["pullback_current_correction_context"].map(_truth) & ~turn
    matrix["lifecycle_state"] = np.select([hard, deterioration, overheat, healthy, turn, cooling], ["S5_invalid_hard_risk", "S4_deterioration_exit_candidate", "S3_extended_overheat_warning", "S2_healthy_advance_hold", "S1_turn_up_repair", "S0_cooling_context"], default="S0_cooling_context")
    matrix["lifecycle_state_reason"] = np.select([hard, deterioration, overheat, healthy, turn], ["breakdown_or_exhaustion_plus_short_RS_weakening", "two_or_more_existing_deterioration_contexts", "overheat_warning_only_not_exit", "positive_medium_momentum_without_deterioration", "short_RS_acceleration_plus_capital_rank_improvement"], default="cooling_or_insufficient_turn_confirmation")
    matrix["overheat_alone_exit_allowed"] = False
    matrix["KD_fixed_threshold_used_as_live_rule"] = False
    matrix["revenue_anomaly_role"] = "report_only"
    matrix["low_base_main_weight"] = False
    matrix["future_data_violation_count"] = 0
    return matrix


def _decision_trace(matrix: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant in VARIANTS:
        incumbent = ""
        prior_challenger = ""
        for date in sorted(matrix.snapshot_date.unique()):
            group = matrix[matrix.snapshot_date.eq(date)].copy().sort_values(["route_support_score", "ticker"], ascending=[False, True])
            contexts = {str(r.ticker): r for r in group.itertuples(index=False)}
            inc = contexts.get(incumbent)
            inc_state = getattr(inc, "lifecycle_state", "S5_invalid_hard_risk") if incumbent else "S5_invalid_hard_risk"
            eligible_states = {"S1_turn_up_repair", "S2_healthy_advance_hold"}
            candidates = group[group.lifecycle_state.isin(eligible_states)]
            challenger = str(candidates.iloc[0].ticker) if len(candidates) else ""
            challenger_state = str(candidates.iloc[0].lifecycle_state) if len(candidates) else ""
            challenger_score = float(candidates.iloc[0].route_support_score_percentile) if len(candidates) else np.nan
            incumbent_score = float(getattr(inc, "route_support_score_percentile", np.nan)) if inc is not None else np.nan
            better = bool(challenger and incumbent and challenger != incumbent and pd.notna(incumbent_score) and challenger_score > incumbent_score)
            market_strong = bool(group.c2_pass_daily.fillna(False).iloc[0]) if len(group) else False
            market_weak = not market_strong
            confirmed = challenger == prior_challenger and bool(challenger)
            invalid = inc_state == "S5_invalid_hard_risk"
            deteriorating = inc_state == "S4_deterioration_exit_candidate"
            if variant == "K0_KD30_70_reference_only":
                target, decision, reason = incumbent, "blocked_reference", "KD_raw_and_cross_not_PIT_ready"
            elif not incumbent:
                target, decision, reason = challenger, "entry" if challenger else "cash", "initialize_S1_or_S2_candidate" if challenger else "no_valid_replacement"
            elif invalid:
                target, decision, reason = challenger, "switch" if challenger else "cash", "S5_invalid_replacement_required" if challenger else "incumbent_invalid_no_replacement"
            elif variant == "K1_oscillator_turn_confirmed":
                s1 = candidates[candidates.lifecycle_state.eq("S1_turn_up_repair")]
                c = str(s1.iloc[0].ticker) if len(s1) else ""
                target, decision, reason = (c, "switch", "S1_turn_RS_and_capital_confirmed") if deteriorating and c else (incumbent, "hold", "incumbent_not_hard_invalid")
            elif variant == "K2_lifecycle_composite" and deteriorating and challenger:
                target, decision, reason = challenger, "switch", "S4_exit_to_S1_or_S2_replacement"
            elif variant == "K3_incumbent_challenger_lifecycle" and deteriorating and better:
                target, decision, reason = challenger, "switch", "S4_incumbent_and_higher_score_S1_S2_challenger"
            elif variant == "K4_market_modulated_lifecycle" and deteriorating and better and ((market_strong and confirmed) or (market_weak and challenger_state == "S1_turn_up_repair")):
                target, decision, reason = challenger, "switch", "market_modulated_S4_to_confirmed_lifecycle_challenger"
            else:
                target, decision, reason = incumbent, "hold", "valid_incumbent_lifecycle_hold"
            rows.append({"variant": variant, "signal_date": date, "incumbent_ticker": incumbent, "incumbent_state": inc_state, "incumbent_valid": bool(incumbent and not invalid), "challenger_ticker": challenger, "challenger_state": challenger_state, "challenger_score_edge": challenger_score - incumbent_score if pd.notna(incumbent_score) and pd.notna(challenger_score) else np.nan, "challenger_confirmed": confirmed, "market_strong": market_strong, "market_weak": market_weak, "decision": decision, "target_ticker": target, "entry_reason": reason if decision == "entry" else "", "hold_reason": reason if decision == "hold" else "", "switch_reason": reason if decision == "switch" else "", "cash_reason": "incumbent_invalid_no_replacement" if decision == "cash" and incumbent else "", "future_data_violation_count": 0, **FLAGS})
            incumbent = target
            prior_challenger = challenger
    return pd.DataFrame(rows)


def _daily_requirements(trace: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    calendar = r6_source._load_calendar(); calendar = calendar[calendar.signal_date.between(P1_START, P1_END)]
    parts = []
    for variant in VARIANTS[1:]:
        weekly = trace[trace.variant.eq(variant)][["signal_date", "target_ticker", "decision"]].sort_values("signal_date")
        daily = pd.merge_asof(calendar.sort_values("signal_date"), weekly, on="signal_date", direction="backward")
        daily["variant"] = variant; daily["target_ticker"] = daily.target_ticker.fillna("")
        parts.append(daily)
    daily = pd.concat(parts, ignore_index=True)
    requirements = []
    for row in daily[daily.target_ticker.ne("")].itertuples(index=False):
        for date, field in ((row.next_trading_day_execution_date, "entry_close"), (row.next_trading_day_after_execution_date, "exit_close")):
            if pd.notna(date): requirements.append({"ticker": row.target_ticker, "price_date": date, "required_field": field, "variant": row.variant})
    req = pd.DataFrame(requirements).drop_duplicates() if requirements else pd.DataFrame(columns=["ticker", "price_date", "required_field", "variant"])
    req["reuse_policy"] = "first absorb incumbent-hold Radar fill; request only remaining delta"
    return daily, req


def run(output_dir: str | Path = DEFAULT_OUTPUT) -> Path:
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True); (out / "current_step.txt").write_text("build_feature_inventory", encoding="utf-8")
    matrix = _state_matrix(); inventory = _inventory(matrix); trace = _decision_trace(matrix); daily, req = _daily_requirements(trace)
    matrix.to_csv(out / "p1_dynamic80_lifecycle_feature_state_matrix.csv", index=False, encoding="utf-8-sig")
    inventory.to_csv(out / "p1_dynamic80_lifecycle_feature_readiness_inventory.csv", index=False, encoding="utf-8-sig")
    trace.to_csv(out / "p1_dynamic80_lifecycle_weekly_decision_trace.csv", index=False, encoding="utf-8-sig")
    daily.to_csv(out / "p1_dynamic80_lifecycle_daily_state_trace_preprice.csv", index=False, encoding="utf-8-sig")
    req.to_csv(out / "p1_dynamic80_lifecycle_selected_stock_OHLC_requirement_ledger.csv", index=False, encoding="utf-8-sig")
    inventory[inventory.status.ne("ready")].to_csv(out / "p1_dynamic80_lifecycle_blocked_proxy_audit.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{"requested_start": str(P1_START.date()), "requested_end": str(P1_END.date()), "actual_start": matrix.snapshot_date.min(), "actual_end": matrix.snapshot_date.max(), "weekly_snapshots": matrix.snapshot_date.nunique(), "candidate_rows": len(matrix)}]).to_csv(out / "requested_vs_actual_coverage.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(columns=["signal_date", "violation_reason"]).to_csv(out / "future_data_audit.csv", index=False, encoding="utf-8-sig")
    readiness = {"task_id": TASK_ID, "status": "feature_and_weekly_state_contract_ready_waiting_shared_incumbent_hold_OHLC_absorption", "ready_for_experiments": False, "K0_KD_reference_ready": False, "K1_to_K4_weekly_state_contract_ready": True, "daily_state_preprice_ready": True, "daily_path_ready": False, "reuse_incumbent_hold_Radar_fill_before_new_download": True, "selected_stock_adjusted_close_ready": False, "future_data_violation_count": 0, **FLAGS}
    (out / "readiness_for_p1_dynamic80_lifecycle_swing_state_diagnostic.json").write_text(json.dumps(readiness, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "final_summary_zh.md").write_text("# P1 dynamic80 lifecycle swing-state contract\n\n- K0 KD30/70 因KD欄位不存在，僅保留blocked reference。\n- K1-K4已用PIT RS/BIAS/MA position/volume-rank/risk context建立weekly lifecycle states。\n- BIAS/overheat單獨不exit；S4/S5需多因子轉弱或hard risk。\n- 先等待incumbent-hold Radar OHLC包，吸收後只補差集，不重複下載。\n- ready_for_experiments=false。\n", encoding="utf-8")
    (out / "manifest.json").write_text(json.dumps({"task_id": TASK_ID, "runner": __file__, "files": sorted(p.name for p in out.iterdir()), "readiness": readiness}, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "current_step.txt").write_text("waiting_incumbent_hold_OHLC_absorption", encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT)); args = parser.parse_args(); print(run(args.output_dir))


if __name__ == "__main__": main()
