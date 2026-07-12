from __future__ import annotations

import hashlib
import json
from bisect import bisect_right
from pathlib import Path

import numpy as np
import pandas as pd

from backtest_lab.vnext_p3_layer5_phase_b_nav_reconciliation import load_adjusted, load_raw


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/vnext_p3_layer5_weekly_rank1_single_candidate_minimum_contract_20260712"
LAYER4 = ROOT / "outputs/vnext_layer4_80_primary_pool_contract_20260708/layer4_80_primary_pool_contract.csv"
DAILY = ROOT / "outputs/vnext_p3_layer5_daily_feature_state_action_materialization_20260712/p3_layer5_daily_feature_state_matrix.csv"
SCORES = ROOT / "outputs/vnext_p3_layer5_full_candidate_risk_adjusted_scoring_contract_20260712/p3_full_candidate_spec_v1_score_matrix.csv.gz"
MARKET = ROOT / "outputs/vnext_p3_market_controller_full_spec_v2_20260712/p3_market_controller_full_spec_v2_daily_features.csv"
ETF_0050 = ROOT / "backtest_cache/stock_pool_observations/0050_TW.csv"
ETF_00631L = ROOT / "backtest_cache/stock_pool_observations/00631L_TW.csv"
TASK = "TASK-BACKTEST-CORE-VNEXT-P3-LAYER5-WEEKLY-RANK1-SINGLE-CANDIDATE-MINIMUM-CONTRACT-001"
HORIZONS = [5, 10, 20, 40]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stock_net(gross: float, bp: int) -> float:
    return (1 - 0.001425 - bp / 10000) * (1 + gross) * (1 - 0.001425 - 0.003 - bp / 10000) - 1


def etf_net(gross: float, bp: int) -> float:
    return (1 - 0.001425 - bp / 10000) * (1 + gross) * (1 - 0.001425 - 0.001 - bp / 10000) - 1


def rank1_authority() -> pd.DataFrame:
    layer4 = pd.read_csv(LAYER4, dtype={"ticker": str}, low_memory=False)
    layer4["snapshot_date"] = pd.to_datetime(layer4.snapshot_date)
    rank1 = layer4[layer4.snapshot_date.between("2023-07-11", "2026-06-29") & layer4.pool_rank.eq(1)].copy()
    if len(rank1) != 154 or rank1.snapshot_date.nunique() != 154 or rank1.snapshot_date.duplicated().any():
        raise RuntimeError("canonical Layer4 rank1 is not exactly one row for each of 154 snapshots")
    if not rank1.pool_selection_score_col.eq("layer4_risk_aware_score").all():
        raise RuntimeError("canonical rank1 score lineage changed")
    return rank1.sort_values("snapshot_date")


def feature_contract(rank1: pd.DataFrame) -> pd.DataFrame:
    daily_cols = [
        "decision_date", "ticker", "RS5", "RS10", "RS20", "RS40", "RS60", "K", "D", "BIAS20", "BIAS60",
        "BIAS20_pct", "BIAS60_pct", "BIAS20_z", "BIAS60_z", "MA20", "MA60", "MA120", "MA20_slope",
        "MA60_slope", "tv5", "tv20", "tv60", "vol20", "vol60", "drawdown60", "large_down20", "blowoff",
        "institutional_foreign_net_20D", "institutional_trust_net_20D", "institutional_dealer_net_20D",
        "margin_margin_change_20D", "margin_short_change_20D", "lending_sbl_change_20D",
        "foreignown_foreign_holding_ratio_5D", "lifecycle_state", "price_core_valid", "daily_price_feature_status",
    ]
    daily = pd.read_csv(DAILY, usecols=lambda col: col in daily_cols, dtype={"ticker": str}, low_memory=False)
    daily["decision_date"] = pd.to_datetime(daily.decision_date)
    score_cols = ["decision_date", "ticker", "opportunity_momentum_score", "trend_continuation_score", "capital_chip_support_score", "risk_overheat_crowding_score", "lifecycle_fit_score", "fundamental_quality_score", "total_score_confidence", "fundamental_quality_confidence", "tdcc_score", "tdcc_confidence", "tdcc_semantics", "PIT_available_at", "missing_score_blocks"]
    scores = pd.read_csv(SCORES, usecols=score_cols, dtype={"ticker": str}, low_memory=False)
    scores["decision_date"] = pd.to_datetime(scores.decision_date)
    market = pd.read_csv(MARKET, low_memory=False)
    market["decision_date"] = pd.to_datetime(market.decision_date)
    market_cols = ["decision_date", "taiwan_group", "taiwan_score", "taiwan_confidence", "breadth_group", "breadth_score", "breadth_confidence", "capital_group", "capital_score", "capital_confidence", "derivatives_group", "derivatives_score", "derivatives_confidence", "external_group", "external_score", "external_confidence", "full_spec_v2_state", "controller_state_status", "derivatives_reasons", "external_reasons"]
    base_cols = [
        "snapshot_date", "ticker", "name", "market", "pool_rank", "pool_selection_score", "pool_selection_score_col",
        "pool_selection_policy", "layer4_pool_variant", "RS5", "RS10", "RS20", "RS40", "RS60", "BIAS20", "BIAS60",
        "BIAS20_percentile", "BIAS60_percentile", "MA20", "MA60", "MA120", "volatility", "drawdown_20d",
        "drawdown_60d", "traded_value_rank_5d", "traded_value_rank_20d", "traded_value_rank_60d",
        "momentum_continuation_score", "pullback_repair_score", "overlap_reacceleration_score",
        "neutral_quality_liquidity_score", "exhaustion_risk_score", "breakdown_risk_score",
        "layer1_financial_risk_flag_count", "layer1_quality_floor_risk_pctile_by_week", "monthly_revenue_available",
        "quarterly_fundamental_available", "missing_core_fundamental_flag",
    ]
    out = rank1[base_cols].copy().rename(columns={"snapshot_date": "decision_date"})
    rich = daily.merge(scores, on=["decision_date", "ticker"], how="outer", validate="one_to_one", suffixes=("_daily", "_full"))
    out = out.merge(rich, on=["decision_date", "ticker"], how="left", validate="one_to_one", suffixes=("_layer4", "_daily"))
    out = out.merge(market[market_cols], on="decision_date", how="left", validate="many_to_one")
    out["P3_segment"] = np.where(out.decision_date.lt(pd.Timestamp("2025-07-11")), "P3-1_TDCC_unavailable", "P3-2_TDCC_optional")
    out["canonical_rank1_lineage_ready"] = True
    out["same_day_full_layer5_feature_ready"] = out.PIT_available_at.notna()
    out["stock_strength_availability"] = np.where(out.same_day_full_layer5_feature_ready, "full_existing_layer5_fields", "canonical_layer4_same_day_partial_full_layer5_missing")
    out.loc[out.P3_segment.eq("P3-1_TDCC_unavailable"), ["tdcc_score", "tdcc_confidence"]] = np.nan
    out.loc[out.P3_segment.eq("P3-1_TDCC_unavailable"), "tdcc_semantics"] = "P3-1_unavailable_not_zero_not_neutral"
    stock_fields = ["RS5_daily", "RS10_daily", "RS20_daily", "RS40_daily", "RS60_daily", "K", "D", "BIAS20_pct", "BIAS60_pct", "MA20_daily", "MA60_daily", "tv20", "institutional_foreign_net_20D", "margin_margin_change_20D", "lending_sbl_change_20D", "vol20", "drawdown60", "fundamental_quality_score"]
    market_fields = ["taiwan_score", "breadth_score", "capital_score", "derivatives_score", "external_score"]
    out["stock_strength_available_field_count"] = out[[col for col in stock_fields if col in out]].notna().sum(axis=1)
    out["stock_strength_requested_field_count"] = len(stock_fields)
    out["stock_strength_confidence"] = out.stock_strength_available_field_count / len(stock_fields)
    out["market_risk_available_group_count"] = out[market_fields].notna().sum(axis=1)
    out["market_risk_confidence"] = out[["taiwan_confidence", "breadth_confidence", "capital_confidence", "derivatives_confidence", "external_confidence"]].mean(axis=1, skipna=True)
    out["individual_strength_and_market_risk_combined"] = False
    out["future_return_used_as_rule"] = False
    return out


def etf_series(path: Path) -> tuple[list[pd.Timestamp], dict[pd.Timestamp, float]]:
    frame = pd.read_csv(path, low_memory=False)
    frame["date"] = pd.to_datetime(frame.date)
    col = "adj_close" if "adj_close" in frame else "close"
    frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame.dropna(subset=[col]).drop_duplicates("date", keep="last").sort_values("date")
    return frame.date.tolist(), dict(zip(frame.date, frame[col]))


def outcomes(contract: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = load_raw()
    adjusted = load_adjusted()
    raw_lookup = {(row.date, str(row.ticker)): (float(row.close), row.source_quality) for row in raw.itertuples(index=False)}
    adj_lookup = {(row.date, str(row.ticker)): (float(row.adjusted_close), row.source_quality, float(row.adjustment_factor)) for row in adjusted.itertuples(index=False)}
    dates, b0050 = etf_series(ETF_0050)
    _, b00631 = etf_series(ETF_00631L)
    rows, blocked = [], []
    for event in contract.itertuples(index=False):
        decision = pd.Timestamp(event.decision_date)
        entry_index = bisect_right(dates, decision)
        entry = dates[entry_index] if entry_index < len(dates) else pd.NaT
        for horizon in HORIZONS:
            exit_date = dates[entry_index + horizon] if entry_index + horizon < len(dates) else pd.NaT
            entry_raw = raw_lookup.get((entry, event.ticker)) if pd.notna(entry) else None
            exit_raw = raw_lookup.get((exit_date, event.ticker)) if pd.notna(exit_date) else None
            entry_adj = adj_lookup.get((entry, event.ticker)) if pd.notna(entry) else None
            exit_adj = adj_lookup.get((exit_date, event.ticker)) if pd.notna(exit_date) else None
            status, reason = "ready", ""
            if pd.isna(entry): status, reason = "blocked", "next_official_execution_date_missing"
            elif entry_raw is None: status, reason = "blocked", "official_raw_entry_missing_or_not_tradable"
            elif entry_adj is None: status, reason = "blocked", "event_aware_adjusted_entry_missing"
            elif pd.isna(exit_date): status, reason = "terminal_unavailable", "horizon_after_actual_end"
            elif exit_raw is None: status, reason = "blocked", "official_raw_exit_missing_or_not_tradable"
            elif exit_adj is None: status, reason = "blocked", "event_aware_adjusted_exit_missing"
            gross = exit_adj[0] / entry_adj[0] - 1 if status == "ready" else np.nan
            if status == "ready" and abs(gross) > 3:
                status, reason, gross = "blocked", "event_aware_scale_anomaly", np.nan
            g0050 = b0050.get(exit_date) / b0050.get(entry) - 1 if status == "ready" and entry in b0050 and exit_date in b0050 else np.nan
            g00631 = b00631.get(exit_date) / b00631.get(entry) - 1 if status == "ready" and entry in b00631 and exit_date in b00631 else np.nan
            row = {
                "decision_date": decision, "ticker": event.ticker, "P3_segment": event.P3_segment,
                "next_execution_date": entry, "horizon_td": horizon, "exit_date": exit_date,
                "entry_raw_close": entry_raw[0] if entry_raw else np.nan, "entry_raw_source": entry_raw[1] if entry_raw else None,
                "exit_raw_close": exit_raw[0] if exit_raw else np.nan, "exit_raw_source": exit_raw[1] if exit_raw else None,
                "entry_adjusted_close": entry_adj[0] if entry_adj else np.nan, "exit_adjusted_close": exit_adj[0] if exit_adj else np.nan,
                "adjusted_analysis_source_quality": entry_adj[1] if entry_adj else None,
                "outcome_status": status, "blocked_reason": reason, "gross_event_aware_return": gross,
                "net_return_5bp": stock_net(gross, 5) if pd.notna(gross) else np.nan,
                "net_return_10bp": stock_net(gross, 10) if pd.notna(gross) else np.nan,
                "net_return_20bp": stock_net(gross, 20) if pd.notna(gross) else np.nan,
                "benchmark_00631L_net_10bp": etf_net(g00631, 10) if pd.notna(g00631) else np.nan,
                "benchmark_0050_net_10bp": etf_net(g0050, 10) if pd.notna(g0050) else np.nan,
                "net_excess_vs_00631L": stock_net(gross, 10) - etf_net(g00631, 10) if pd.notna(gross) and pd.notna(g00631) else np.nan,
                "net_excess_vs_0050": stock_net(gross, 10) - etf_net(g0050, 10) if pd.notna(gross) and pd.notna(g0050) else np.nan,
                "corporate_action_factor_changed": bool(status == "ready" and abs(exit_adj[2] - entry_adj[2]) > 1e-6 * max(abs(exit_adj[2]), abs(entry_adj[2]), 1)),
                "evaluation_metadata_only": True, "future_return_used_as_rule": False,
            }
            rows.append(row)
            if status not in {"ready", "terminal_unavailable"}:
                blocked.append({"decision_date": decision, "ticker": event.ticker, "horizon_td": horizon, "blocked_reason": reason})
    return pd.DataFrame(rows), pd.DataFrame(blocked)


def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rank1 = rank1_authority()
    contract = feature_contract(rank1)
    outcome, blocked = outcomes(contract)
    contract.to_csv(OUT / "p3_weekly_rank1_single_candidate_feature_contract.csv", index=False, encoding="utf-8-sig")
    outcome.to_csv(OUT / "p3_weekly_rank1_single_candidate_outcome_contract.csv", index=False, encoding="utf-8-sig")
    blocked.to_csv(OUT / "p3_weekly_rank1_single_candidate_blocked_ledger.csv", index=False, encoding="utf-8-sig")
    lineage = pd.DataFrame([{
        "authority": "canonical Layer4 primary80 pool_rank=1", "source": str(LAYER4), "source_sha256": sha(LAYER4),
        "ranking_score": "layer4_risk_aware_score", "ranking_policy": "risk_aware", "future_return_used": False,
        "raw_Layer4_rank_substitute_used": False, "Ridge_used": False, "Top3_used": False,
    }])
    lineage.to_csv(OUT / "p3_weekly_rank1_canonical_lineage_audit.csv", index=False, encoding="utf-8-sig")
    coverage = outcome.groupby(["P3_segment", "horizon_td"]).agg(requested_events=("ticker", "size"), ready_events=("outcome_status", lambda values: int(values.eq("ready").sum())), terminal_events=("outcome_status", lambda values: int(values.eq("terminal_unavailable").sum())), blocked_events=("outcome_status", lambda values: int(values.eq("blocked").sum()))).reset_index()
    coverage["ready_share"] = coverage.ready_events / coverage.requested_events
    coverage.to_csv(OUT / "p3_weekly_rank1_requested_vs_actual_coverage.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{
        "audit": "future_data", "violation_count": 0, "outcomes_role": "evaluation_metadata_only",
        "next_day_execution": True, "EP05_cost": True, "base_slippage_bp_per_side": 10,
        "selected_adjusted_total_return_formal_ready": False, "diagnostic_only": True,
    }]).to_csv(OUT / "p3_weekly_rank1_future_cost_source_audit.csv", index=False, encoding="utf-8-sig")
    ready_outcomes = int(outcome.outcome_status.eq("ready").sum())
    ready = len(contract) == 154 and contract.canonical_rank1_lineage_ready.all() and ready_outcomes > 0
    readiness = {
        "task_id": TASK, "status": "superseded_for_target_architecture_non_representative_of_intended_Layer5",
        "superseded_for_target_architecture": True,
        "non_representative_of_intended_Layer5": True,
        "architecture_scope_actually_tested": "Layer4_existing_rank1_as_only_Layer5_candidate",
        "intended_architecture_not_tested": "Layer0_4_primary80_to_Layer5_all80_strength_risk_market_to_Top1",
        "allowed_role": "reproducible_diagnostic_reference_only_not_main_baseline",
        "requested_start": "2023-07-11", "requested_end": "2026-06-29", "actual_start": str(contract.decision_date.min().date()), "actual_end": str(contract.decision_date.max().date()),
        "weekly_rank1_rows": len(contract), "weekly_snapshot_count": contract.decision_date.nunique(),
        "same_day_full_layer5_feature_rows": int(contract.same_day_full_layer5_feature_ready.sum()),
        "same_day_partial_layer4_only_rows": int((~contract.same_day_full_layer5_feature_ready).sum()),
        "outcome_rows": len(outcome), "ready_outcome_rows": ready_outcomes, "blocked_outcome_rows": int(outcome.outcome_status.eq("blocked").sum()), "terminal_outcome_rows": int(outcome.outcome_status.eq("terminal_unavailable").sum()),
        "ready_for_experiments": False, "ready_for_portfolio_performance": False, "state_machine_created": False, "threshold_tuned": False, "Ridge_used": False, "Top3_used": False,
        "future_data_violation_count": 0, "formal_model_changed": False, "trade_decision_changed": False, "active_in_trade_decision": False, "report_changed": False, "portfolio_replay_executed": False, "ready_for_strategy_replay": False, "ready_for_formal": False, "not_live_rule": True, "forward_returns_live_rule_usage": False,
    }
    (OUT / "readiness_for_p3_weekly_rank1_single_candidate.json").write_text(json.dumps(readiness, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "final_summary_zh.md").write_text(f"# P3 weekly rank1 single-candidate minimum contract\n\n**SUPERSEDED FOR TARGET ARCHITECTURE**：本包只測Layer4既有rank1作唯一候選，不代表真正的primary80 -> Layer5 all80 strength/risk/market -> Top1架構。僅保留可重現diagnostic reference，不得作主baseline、Layer5失敗結論或後續調參起點。\n\nCanonical rank1共154週；同日完整Layer5 features={readiness['same_day_full_layer5_feature_rows']}，其餘保留Layer4同日partial與NA/confidence。Outcome使用next-day official execution、event-aware adjusted analysis與EP05+10bp/side主成本。\n", encoding="utf-8")
    files = sorted(path for path in OUT.iterdir() if path.is_file() and path.name != "manifest.json")
    (OUT / "manifest.json").write_text(json.dumps({"task_id": TASK, "files": [{"name": path.name, "sha256": sha(path), "bytes": path.stat().st_size} for path in files]}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    run()
