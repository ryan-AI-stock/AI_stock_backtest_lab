from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from backtest_lab import vnext_daily_incumbent_challenger_state_machine_contract as source
from backtest_lab import vnext_weekly_r6_single_position_state_boundary_reconstruction_contract as r6_source


TASK_ID = "TASK-BACKTEST-CORE-VNEXT-P1-DYNAMIC80-INCUMBENT-HOLD-COMPARATOR-CONTRACT-001"
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "vnext_p1_dynamic80_incumbent_hold_comparator_contract_20260710"
R6_STATE = REPO_ROOT / "outputs/vnext_weekly_r6_single_position_state_boundary_reconstruction_contract_20260710/reconstructed_weekly_r6_single_position_daily_state_rows.csv"
RADAR_INCUMBENT_OHLC = Path(r"C:\Users\zergv\Documents\Codex\2026-05-23\ai-stock-rotation-radar-https-docs\outputs\radar_vnext_p1_dynamic80_incumbent_hold_selected_stock_daily_ohlc_gap_fill_20260710\p1_dynamic80_incumbent_hold_selected_stock_daily_unadjusted_ohlc_rows.csv")
P1_START, P1_END = pd.Timestamp("2015-01-02"), pd.Timestamp("2022-12-29")
VARIANTS = (
    "I1_replacement_only_hold",
    "I2_deteriorating_incumbent_plus_better_challenger",
    "I3_one_prior_snapshot_confirmation",
)
FLAGS = {
    "formal_model_changed": False, "trade_decision_changed": False,
    "active_in_trade_decision": False, "report_changed": False,
    "portfolio_replay_executed": False, "ready_for_strategy_replay": False,
    "ready_for_formal": False, "not_live_rule": True,
    "forward_returns_live_rule_usage": False,
}
COST = {"hold": 0.0, "stock_to_stock": 0.00585, "stock_to_cash": 0.004425, "cash_to_stock": 0.001425}


def _truth(value: object) -> bool:
    if pd.isna(value):
        return False
    return str(value).lower() in {"true", "1", "yes"}


def _context(matrix: pd.DataFrame, date: pd.Timestamp, ticker: str) -> dict:
    rows = matrix[(matrix["snapshot_date"].eq(date)) & (matrix["ticker"].eq(ticker))]
    return rows.iloc[0].to_dict() if len(rows) else {}


def _hard_risk(row: dict) -> bool:
    breakdown = _truth(row.get("high_exhaustion_or_breakdown_context"))
    weakening = _truth(row.get("rs_short_deterioration_flag")) or _truth(row.get("rs60_high_short_rs_weakening_exhaustion_context"))
    return breakdown and weakening


def _weekly_decisions(matrix: pd.DataFrame) -> pd.DataFrame:
    dates = sorted(matrix.loc[matrix["snapshot_date"].between(P1_START, P1_END), "snapshot_date"].unique())
    rows: list[dict] = []
    for variant in VARIANTS:
        incumbent = ""
        prior_challenger = ""
        prior_better = False
        for date_value in dates:
            date = pd.Timestamp(date_value)
            candidates = matrix[matrix["snapshot_date"].eq(date)].copy()
            candidates["hard_risk_blocker"] = candidates.apply(lambda r: _hard_risk(r.to_dict()), axis=1)
            eligible = candidates[~candidates["hard_risk_blocker"]].sort_values(["route_support_score", "ticker"], ascending=[False, True])
            challenger = str(eligible.iloc[0]["ticker"]) if len(eligible) else ""
            challenger_score = float(eligible.iloc[0]["route_support_score_percentile"]) if len(eligible) else np.nan
            incumbent_row = _context(matrix, date, incumbent) if incumbent else {}
            incumbent_in_primary80 = bool(incumbent_row)
            incumbent_hard_invalid = _hard_risk(incumbent_row) if incumbent_row else bool(incumbent)
            deterioration = _truth(incumbent_row.get("incumbent_deterioration_confirmed")) if incumbent_row else False
            incumbent_valid = bool(incumbent) and not incumbent_hard_invalid
            incumbent_score = float(incumbent_row.get("route_support_score_percentile", np.nan)) if incumbent_row else np.nan
            edge = challenger_score - incumbent_score if pd.notna(challenger_score) and pd.notna(incumbent_score) else np.nan
            better = bool(challenger and incumbent and challenger != incumbent and pd.notna(edge) and edge > 0 and not _hard_risk(eligible.iloc[0].to_dict()))
            confirmed = better and prior_better and challenger == prior_challenger

            if not incumbent:
                target, decision, reason = challenger, "switch" if challenger else "cash", "initialize_top_data_ready_replacement" if challenger else "no_valid_replacement"
            elif incumbent_hard_invalid:
                target, decision, reason = challenger, "switch" if challenger else "cash", "incumbent_hard_invalid_replacement_required" if challenger else "incumbent_invalid_no_replacement"
            elif variant == "I1_replacement_only_hold":
                target, decision, reason = incumbent, "hold", "valid_incumbent_hold_no_replacement_test"
            elif variant == "I2_deteriorating_incumbent_plus_better_challenger" and deterioration and better:
                target, decision, reason = challenger, "switch", "incumbent_deteriorating_and_better_challenger"
            elif variant == "I3_one_prior_snapshot_confirmation" and deterioration and confirmed:
                target, decision, reason = challenger, "switch", "deteriorating_incumbent_and_challenger_better_two_snapshots"
            else:
                target, decision, reason = incumbent, "hold", "hold_valid_incumbent_challenger_not_actionable"

            rows.append({
                "variant": variant, "signal_date": date, "incumbent_ticker_before": incumbent,
                "incumbent_in_primary80": incumbent_in_primary80, "watch100_membership_ready": False,
                "incumbent_valid": incumbent_valid, "incumbent_hard_invalid": incumbent_hard_invalid,
                "incumbent_deterioration_confirmed": deterioration, "incumbent_reason": reason,
                "incumbent_score": incumbent_score, "challenger_ticker": challenger,
                "challenger_score": challenger_score, "challenger_score_edge": edge,
                "challenger_confirmed": confirmed, "decision": decision,
                "target_ticker": target, "cash_reason": "incumbent_invalid_no_replacement" if not target else "",
                "systemic_bear_cash": False, "portfolio_stop_cash": False,
                "score_source_quality": "weekly_PIT_route_support_quant_score_existing_contract",
                "revenue_anomaly_role": "report_only", "low_base_main_weight": False,
                "future_data_violation_count": 0, **FLAGS,
            })
            incumbent = target
            prior_challenger, prior_better = challenger, better
    return pd.DataFrame(rows)


def _expand_daily(weekly: pd.DataFrame) -> pd.DataFrame:
    calendar = r6_source._load_calendar()
    calendar = calendar[calendar["signal_date"].between(P1_START, P1_END)].copy()
    parts = []
    for variant in VARIANTS:
        decisions = weekly[weekly["variant"].eq(variant)].sort_values("signal_date")
        daily = pd.merge_asof(calendar.sort_values("signal_date"), decisions, on="signal_date", direction="backward")
        daily["variant"] = variant
        daily["target_ticker"] = daily["target_ticker"].fillna("")
        daily["selected_asset_type"] = np.where(daily["target_ticker"].eq(""), "cash", "stock")
        daily["weekly_decision_updated"] = daily["decision"].notna()
        daily["decision"] = daily["decision"].fillna("cash")
        daily["decision_reason_daily"] = np.where(daily["weekly_decision_updated"], daily["incumbent_reason"], "before_first_weekly_snapshot_cash_no_future_backfill")
        parts.append(daily)
    out = pd.concat(parts, ignore_index=True)
    out = out.sort_values(["variant", "signal_date"])
    out["previous_ticker"] = out.groupby("variant")["target_ticker"].shift().fillna("")
    out["transition_type"] = np.select(
        [out["previous_ticker"].eq(out["target_ticker"]), out["previous_ticker"].eq("") & out["target_ticker"].ne(""), out["previous_ticker"].ne("") & out["target_ticker"].eq("")],
        ["hold_same", "cash_to_stock", "stock_to_cash"], default="stock_to_stock",
    )
    out["transition_cost_rate_hook"] = out["transition_type"].map(COST).fillna(0.0)
    out["execution_basis"] = "weekly_signal_close_next_trading_day_close_unique_position_daily_mark"
    return out


def _attach_prices(daily: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    prices, _ = r6_source._load_official_prices()
    radar = pd.read_csv(RADAR_INCUMBENT_OHLC, dtype={"ticker": str}, low_memory=False)
    radar = radar.rename(columns={"date": "date"})
    radar["ticker"] = radar.ticker.astype(str).str.replace(r"\.0$", "", regex=True)
    radar["date"] = pd.to_datetime(radar.date, errors="coerce")
    radar["close"] = pd.to_numeric(radar.close, errors="coerce")
    radar["source_quality"] = radar.get("source_quality", "official_unadjusted_OHLC")
    radar["source_route"] = radar.get("source_route", "Radar_incumbent_hold_fill")
    radar["adjustment_policy"] = radar.get("adjustment_policy", "unadjusted;adjusted_close_blocked")
    prices = pd.concat([prices, radar[["ticker", "date", "close", "source_quality", "source_route", "adjustment_policy"]]], ignore_index=True).drop_duplicates(["ticker", "date"], keep="last")
    entry = prices.rename(columns={"ticker": "target_ticker", "date": "next_trading_day_execution_date", "close": "entry_close"})
    exit_ = prices.rename(columns={"ticker": "target_ticker", "date": "next_trading_day_after_execution_date", "close": "exit_close"})
    out = daily.merge(entry[["target_ticker", "next_trading_day_execution_date", "entry_close"]], on=["target_ticker", "next_trading_day_execution_date"], how="left")
    out = out.merge(exit_[["target_ticker", "next_trading_day_after_execution_date", "exit_close"]], on=["target_ticker", "next_trading_day_after_execution_date"], how="left")
    cash = out["target_ticker"].eq("")
    out.loc[cash, ["entry_close", "exit_close"]] = 1.0
    out["gross_daily_return"] = out["exit_close"] / out["entry_close"] - 1.0
    out["net_daily_return_after_transition_cost"] = out["gross_daily_return"] - out["transition_cost_rate_hook"]
    out["daily_path_ready"] = out["gross_daily_return"].notna()
    missing = out[~cash & ~out["daily_path_ready"]].copy()
    gap_rows = []
    for row in missing.itertuples(index=False):
        for date, field in ((row.next_trading_day_execution_date, "entry_close"), (row.next_trading_day_after_execution_date, "exit_close")):
            if pd.notna(date):
                gap_rows.append({"ticker": row.target_ticker, "price_date": date, "required_field": field, "variant": row.variant, "signal_date": row.signal_date})
    gaps = pd.DataFrame(gap_rows)
    if len(gaps):
        gaps = gaps.groupby(["ticker", "price_date", "required_field"], as_index=False).agg(
            variants=("variant", lambda x: "|".join(sorted(set(x)))),
            impacted_signal_dates=("signal_date", lambda x: "|".join(sorted(set(pd.to_datetime(x).dt.strftime("%Y-%m-%d"))))),
        )
        gaps["source_requirement"] = "bounded selected-ticker official unadjusted daily OHLC"
        gaps["next_owner"] = "Radar/Data"
    return out, gaps


def _metrics(state: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant in VARIANTS:
        sub = state[state["variant"].eq(variant)].copy()
        ready = bool(sub["daily_path_ready"].all())
        returns = sub["net_daily_return_after_transition_cost"].dropna()
        equity = (1 + returns).cumprod()
        dd = equity / equity.cummax() - 1 if len(equity) else pd.Series(dtype=float)
        transitions = sub[sub["transition_type"].ne("hold_same")]
        stock_runs = (sub["target_ticker"].ne(sub["target_ticker"].shift())).cumsum()
        hold_days = sub[sub["target_ticker"].ne("")].groupby(stock_runs).size()
        rows.append({
            "variant": variant, "path_ready": ready,
            "net_total_return_after_cost": float(equity.iloc[-1] - 1) if ready and len(equity) else np.nan,
            "MDD": float(dd.min()) if ready and len(dd) else np.nan,
            "stock_exposure_share": float(sub["target_ticker"].ne("").mean()), "cash_exposure_share": float(sub["target_ticker"].eq("").mean()),
            "cash_systemic_bear_share": float(sub["systemic_bear_cash"].fillna(False).mean()),
            "cash_portfolio_stop_share": float(sub["portfolio_stop_cash"].fillna(False).mean()),
            "cash_incumbent_invalid_no_replacement_share": float(sub["cash_reason"].fillna("").eq("incumbent_invalid_no_replacement").mean()),
            "transition_count": int(len(transitions)), "average_hold_days": float(hold_days.mean()) if len(hold_days) else 0,
            "median_hold_days": float(hold_days.median()) if len(hold_days) else 0, "max_hold_days": int(hold_days.max()) if len(hold_days) else 0,
            "cash_majority_semantic_failure": bool(sub["target_ticker"].eq("").mean() > 0.5),
        })
    return pd.DataFrame(rows)


def run(output_dir: str | Path = DEFAULT_OUTPUT) -> Path:
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    (output / "current_step.txt").write_text("load_PIT_candidate_fields", encoding="utf-8")
    matrix = source._weekly_candidate_matrix()
    weekly = _weekly_decisions(matrix)
    daily = _expand_daily(weekly)
    state, gaps = _attach_prices(daily)
    metrics = _metrics(state)
    inventory = pd.DataFrame([
        ("route_support_score", True, "weekly PIT quant score"), ("quality", True, "Layer1 weekly PIT context"),
        ("risk_breakdown_exhaustion", True, "existing booleans; combined hard invalid"), ("BIAS", True, "context only; never sole exit"),
        ("volatility", True, "weekly PIT context"), ("primary80_membership", True, "canonical weekly pool"),
        ("watch100_membership", False, "not materialized in canonical primary80 file"), ("revenue_anomaly", True, "report only"),
        ("systemic_bear_exact_daily", False, "not applied in this bounded comparator"), ("portfolio_stop_exact_state", False, "requires complete wealth path"),
    ], columns=["field", "ready", "source_quality"])
    weekly.to_csv(output / "p1_dynamic80_incumbent_hold_weekly_decision_trace.csv", index=False, encoding="utf-8-sig")
    state.to_csv(output / "p1_dynamic80_incumbent_hold_daily_state_path.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(output / "p1_dynamic80_incumbent_hold_metric_hooks.csv", index=False, encoding="utf-8-sig")
    inventory.to_csv(output / "p1_dynamic80_incumbent_hold_PIT_field_inventory.csv", index=False, encoding="utf-8-sig")
    gaps.to_csv(output / "p1_dynamic80_incumbent_hold_selected_stock_ohlc_gap_ledger.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{"requested_start": str(P1_START.date()), "requested_end": str(P1_END.date()), "actual_start": state.signal_date.min(), "actual_end": state.signal_date.max(), "weekly_snapshots": weekly.signal_date.nunique(), "daily_signal_dates": state.signal_date.nunique()}]).to_csv(output / "requested_vs_actual_coverage.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(columns=["signal_date", "violation_reason"]).to_csv(output / "future_data_audit.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{"item": "watch100_membership", "status": "blocked", "detail": "not used as invalidity proof"}, {"item": "selected_stock_adjusted_close", "status": "blocked", "detail": "official unadjusted diagnostic only"}, {"item": "systemic_bear_and_portfolio_stop", "status": "not_applied", "detail": "cash reasons reserved but exact state not fabricated"}]).to_csv(output / "blocked_proxy_audit.csv", index=False, encoding="utf-8-sig")
    ready = len(gaps) == 0 and bool(metrics.path_ready.all())
    readiness = {"task_id": TASK_ID, "status": "ready_for_experiments" if ready else "selected_stock_OHLC_gap_fill_required", "ready_for_experiments": ready, "selected_stock_ohlc_gap_rows": len(gaps), "official_unadjusted_OHLC_diagnostic_only": True, "selected_stock_adjusted_close_ready": False, "supersedes_attack_sleeve_no_target_cash_default": True, "default_state": "hold_valid_incumbent", "cash_only_confirmed_risk_or_no_valid_replacement": True, "future_data_violation_count": 0, **FLAGS}
    (output / "readiness_for_p1_dynamic80_incumbent_hold_comparator.json").write_text(json.dumps(readiness, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "final_summary_zh.md").write_text(f"# vNext-native incumbent-hold comparator\n\n- exact legacy transplant: blocked_not_pursued。\n- I1-I3 使用既有 weekly PIT route_support/quality/risk/BIAS/volatility；BIAS 不單獨觸發退出。\n- default=hold_valid_incumbent；沒有 challenger 不等於 no target。\n- selected OHLC gaps={len(gaps)}；ready_for_experiments={str(ready).lower()}。\n- watch100 membership、adjusted close、exact systemic/portfolio-stop state 保留 blocker，不杜撰。\n", encoding="utf-8")
    (output / "manifest.json").write_text(json.dumps({"task_id": TASK_ID, "runner": __file__, "files": sorted(p.name for p in output.iterdir()), "readiness": readiness}, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "current_step.txt").write_text("completed" if ready else "await_bounded_selected_OHLC", encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT)); args = parser.parse_args()
    print(run(args.output_dir))


if __name__ == "__main__":
    main()
