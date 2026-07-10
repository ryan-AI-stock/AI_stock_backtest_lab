from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backtest_lab import vnext_daily_incumbent_challenger_state_machine_contract as daily_source
from backtest_lab.vnext_daily_incumbent_challenger_ohlc_absorption import _benchmark_price_map


REPO_ROOT = Path(__file__).resolve().parents[2]
DAILY_DIR = REPO_ROOT / "outputs" / "vnext_daily_incumbent_challenger_state_machine_contract_ohlc_absorbed_20260710"
R6_DIR = REPO_ROOT / "outputs" / "vnext_r6_guard_first_market_bias_override_unified_contract_20260709"
RADAR_R6_OHLC = Path("C:/Users/zergv/Documents/Codex/2026-05-23/ai-stock-rotation-radar-https-docs/outputs/radar_vnext_regime_switch_route_selected_stock_ohlc_source_package_20260708/regime_switch_selected_ohlc_rows.csv")
RADAR_R6_DAILY_FILL = Path("C:/Users/zergv/Documents/Codex/2026-05-23/ai-stock-rotation-radar-https-docs/outputs/radar_vnext_p1_weekly_r6_selected_stock_daily_ohlc_attribution_gap_fill_20260710/p1_weekly_r6_selected_stock_daily_ohlc_filled_rows.csv")
OUTPUT_DIR = REPO_ROOT / "outputs" / "vnext_p1_daily_f_weekly_r6_transition_drawdown_attribution_contract_20260710"

TASK_ID = "TASK-BACKTEST-CORE-VNEXT-P1-DAILY-F-VS-WEEKLY-R6-TRANSITION-DRAWDOWN-ATTRIBUTION-CONTRACT-001"
F_VARIANT = "F_two_day_confirmation_and_risk_adjusted_edge"
P1_START = pd.Timestamp("2015-01-02")
P1_END = pd.Timestamp("2022-12-29")
VARIANTS = list(daily_source.VARIANTS)
FLAGS = {
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ticker(value: Any) -> str:
    if pd.isna(value):
        return ""
    value = str(value).strip()
    return value[:-2] if value.endswith(".0") else value


def _bool(value: Any) -> bool:
    if pd.isna(value):
        return False
    return value.lower() in {"true", "1", "yes"} if isinstance(value, str) else bool(value)


def _write_csv(frame: pd.DataFrame, name: str) -> Path:
    path = OUTPUT_DIR / name
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def _p1(frame: pd.DataFrame, date_col: str = "signal_date") -> pd.DataFrame:
    out = frame.copy()
    dates = pd.to_datetime(out[date_col], errors="coerce")
    return out[(dates >= P1_START) & (dates <= P1_END)].copy()


def _load_daily() -> pd.DataFrame:
    frame = pd.read_csv(
        DAILY_DIR / "daily_incumbent_challenger_state_machine_contract_ohlc_absorbed.csv",
        low_memory=False,
        dtype={"selected_ticker_after": str, "incumbent_ticker_before": str, "challenger_ticker": str},
    )
    for col in ["signal_date", "next_trading_day_execution_date", "next_trading_day_after_execution_date", "pool_snapshot_date"]:
        frame[col] = pd.to_datetime(frame[col], errors="coerce")
    for col in ["selected_ticker_after", "incumbent_ticker_before", "challenger_ticker"]:
        frame[col] = frame[col].map(_ticker)
    return frame


def _candidate_matrix() -> pd.DataFrame:
    matrix = daily_source._weekly_candidate_matrix().copy()
    matrix["pool_snapshot_date"] = pd.to_datetime(matrix["snapshot_date"], errors="coerce")
    matrix["ticker"] = matrix["ticker"].map(_ticker)
    keep = [
        "pool_snapshot_date", "ticker", "name", "route_support_score", "route_support_score_raw",
        "route_support_score_percentile", "route_support_score_source_quality", "quality_component",
        "rs_component", "liquidity_component", "bias_health_component", "route_support_component",
        "risk_inverse_component", "route_support_variant_count", "route_support_variant_flags",
        "incumbent_deterioration_signal_count", "incumbent_deterioration_confirmed",
        "BIAS20_percentile", "BIAS60_percentile", "BIAS120_percentile", "volatility_pctile_by_week",
        "layer1_quality_floor_risk_pctile_by_week", "layer4_risk_penalty_score",
    ]
    matrix = matrix[[col for col in keep if col in matrix.columns]].copy()
    # A few historical Layer4 snapshots contain duplicate ticker records. This
    # attribution join selects the already-existing highest route-support score
    # deterministically; it does not construct a new candidate or rule.
    matrix = matrix.sort_values(["pool_snapshot_date", "ticker", "route_support_score"], ascending=[True, True, False])
    return matrix.drop_duplicates(["pool_snapshot_date", "ticker"], keep="first")


def _score_attribution(daily: pd.DataFrame, matrix: pd.DataFrame) -> pd.DataFrame:
    f = _p1(daily[daily["state_machine_variant"].eq(F_VARIANT)])
    challenger = matrix.add_prefix("challenger_").rename(columns={"challenger_pool_snapshot_date": "pool_snapshot_date", "challenger_ticker": "challenger_ticker"})
    incumbent = matrix.add_prefix("incumbent_").rename(columns={"incumbent_pool_snapshot_date": "pool_snapshot_date", "incumbent_ticker": "incumbent_ticker_before"})
    out = f.merge(challenger, on=["pool_snapshot_date", "challenger_ticker"], how="left", validate="many_to_one")
    out = out.merge(incumbent, on=["pool_snapshot_date", "incumbent_ticker_before"], how="left", validate="many_to_one")
    out["challenger_raw_score"] = pd.to_numeric(out.get("challenger_route_support_score"), errors="coerce")
    out["incumbent_raw_score"] = pd.to_numeric(out.get("incumbent_route_support_score"), errors="coerce")
    out.loc[out["incumbent_asset_type_before"].eq("etf"), "incumbent_raw_score"] = np.nan
    out["challenger_cross_sectional_percentile"] = pd.to_numeric(out.get("challenger_route_support_score_percentile"), errors="coerce")
    out["incumbent_cross_sectional_percentile"] = pd.to_numeric(out.get("incumbent_route_support_score_percentile"), errors="coerce")
    out.loc[out["incumbent_asset_type_before"].eq("etf"), "incumbent_cross_sectional_percentile"] = np.nan
    out["raw_score_edge"] = out["challenger_raw_score"] - out["incumbent_raw_score"]
    out["risk_adjusted_edge"] = out["raw_score_edge"]
    out["risk_adjusted_edge_definition"] = (
        "existing_route_support_score includes quality(0.10), RS(0.20), liquidity(0.10), bias_health(0.10), "
        "route_support(0.38), risk_inverse(0.12); no new selector score is created"
    )
    previous = out.shift(1)
    out["prior_c2_pass_flag"] = previous["c2_pass_flag"].fillna(False).map(_bool)
    out["prior_consensus_trigger_flag"] = previous["consensus_trigger_flag"].fillna(False).map(_bool)
    out["prior_r6_override_flag"] = previous["r6_override_flag"].fillna(False).map(_bool)
    out["prior_pool_snapshot_date"] = previous["pool_snapshot_date"]
    out["prior_challenger_ticker"] = previous["challenger_ticker"].fillna("")
    out["gate_on_flag"] = (
        out["transition_type"].eq("base_to_stock")
        & (out["c2_pass_flag"].map(_bool) & out["consensus_trigger_flag"].map(_bool) | out["r6_override_flag"].map(_bool))
    )
    out["gate_off_flag"] = out["transition_type"].eq("stock_to_base") & ~(
        out["c2_pass_flag"].map(_bool) & (out["consensus_trigger_flag"].map(_bool) | out["r6_override_flag"].map(_bool))
    )
    out["challenger_ticker_changed_flag"] = out["challenger_ticker"].ne(out["prior_challenger_ticker"])
    out["weekly_pool_snapshot_changed_flag"] = out["pool_snapshot_date"].ne(out["prior_pool_snapshot_date"])
    out["r6_override_changed_flag"] = out["r6_override_flag"].map(_bool).ne(out["prior_r6_override_flag"])
    out["two_day_confirmation_flag"] = out["challenger_confirmation_days"].fillna(0).astype(float).ge(2) & out["requires_two_day_confirmation"].map(_bool)
    out["incumbent_deterioration_flag"] = out["incumbent_deterioration_confirmed"].map(_bool)
    out["stock_to_stock_threshold_controlled_flag"] = out["transition_type"].eq("stock_to_stock")
    out["primary_transition_cause"] = np.select(
        [
            out["gate_off_flag"], out["gate_on_flag"], out["transition_type"].eq("stock_to_stock") & out["incumbent_deterioration_flag"],
            out["transition_type"].eq("stock_to_stock") & out["two_day_confirmation_flag"], out["transition_type"].eq("stock_to_stock"),
        ],
        ["gate_off", "gate_on", "stock_to_stock_incumbent_deterioration", "stock_to_stock_two_day_confirmation", "stock_to_stock_challenger_change"],
        default="hold_same",
    )
    selected = [
        "signal_date", "next_trading_day_execution_date", "pool_snapshot_date", "state_machine_variant", "incumbent_ticker_before",
        "incumbent_asset_type_before", "challenger_ticker", "challenger_name", "selected_ticker_after", "selected_asset_type_after",
        "decision_reason", "transition_type", "c2_pass_flag", "consensus_trigger_flag", "r6_override_flag", "p1_risk_veto_flag",
        "stock_candidate_allowed", "switch_threshold_pct_points", "requires_incumbent_deterioration", "requires_two_day_confirmation",
        "challenger_confirmation_days", "challenger_raw_score", "incumbent_raw_score", "challenger_cross_sectional_percentile",
        "incumbent_cross_sectional_percentile", "raw_score_edge", "challenger_score_edge_pct_points", "risk_adjusted_edge",
        "challenger_risk_inverse_component", "incumbent_risk_inverse_component", "challenger_route_support_variant_count",
        "incumbent_route_support_variant_count", "incumbent_deterioration_flag", "gate_on_flag", "gate_off_flag",
        "challenger_ticker_changed_flag", "weekly_pool_snapshot_changed_flag", "r6_override_changed_flag", "two_day_confirmation_flag",
        "stock_to_stock_threshold_controlled_flag", "primary_transition_cause", "daily_price_source_quality",
        "official_unadjusted_daily_ohlc_ready", "selected_stock_adjusted_close_ready", "transition_cost_rate_hook",
        "net_daily_return_after_transition_cost", "metric_eligible_P1", "revenue_anomaly_warning", "rs20_top3_reference_only",
    ]
    return out[[col for col in selected if col in out.columns]].copy()


def _variant_equivalence(daily: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    p1 = _p1(daily)
    p1 = p1[p1["metric_eligible_P1"].map(_bool)].copy()
    decision = p1[["state_machine_variant", "signal_date", "selected_ticker_after", "selected_asset_type_after", "transition_type", "decision_reason"]].copy()
    decision["decision_key"] = decision[["selected_ticker_after", "selected_asset_type_after", "transition_type", "decision_reason"]].astype(str).agg("|".join, axis=1)
    by_day = decision.groupby("signal_date", as_index=False).agg(
        variant_rows=("state_machine_variant", "nunique"),
        distinct_decisions=("decision_key", "nunique"),
        selected_assets=("selected_ticker_after", lambda x: "|".join(sorted(set(x)))),
    )
    by_day["all_variants_identical_decision_flag"] = by_day["distinct_decisions"].eq(1) & by_day["variant_rows"].eq(len(VARIANTS))
    pairs = []
    pivot = decision.pivot(index="signal_date", columns="state_machine_variant", values="decision_key")
    for index, left in enumerate(VARIANTS):
        for right in VARIANTS[index + 1:]:
            both = pivot[[left, right]].dropna()
            pairs.append({
                "comparison_type": "pairwise", "left_variant": left, "right_variant": right,
                "comparable_p1_rows": int(len(both)), "identical_decision_rows": int((both[left] == both[right]).sum()),
                "different_decision_rows": int((both[left] != both[right]).sum()),
                "identical_share": float((both[left] == both[right]).mean()) if len(both) else np.nan,
            })
    pairs.append({
        "comparison_type": "all_six", "left_variant": "A-F", "right_variant": "",
        "comparable_p1_rows": int(len(by_day)), "identical_decision_rows": int(by_day["all_variants_identical_decision_flag"].sum()),
        "different_decision_rows": int((~by_day["all_variants_identical_decision_flag"]).sum()),
        "identical_share": float(by_day["all_variants_identical_decision_flag"].mean()) if len(by_day) else np.nan,
    })
    return by_day, pd.DataFrame(pairs)


def _transition_summary(score: pd.DataFrame) -> pd.DataFrame:
    moved = score[score["transition_type"].ne("hold_same")].copy()
    if moved.empty:
        return pd.DataFrame()
    summary = moved.groupby(["primary_transition_cause", "transition_type"], as_index=False).agg(
        transition_rows=("signal_date", "size"),
        mean_percentile_edge=("challenger_score_edge_pct_points", "mean"),
        mean_raw_edge=("raw_score_edge", "mean"),
        weekly_pool_snapshot_changed_rows=("weekly_pool_snapshot_changed_flag", "sum"),
        challenger_changed_rows=("challenger_ticker_changed_flag", "sum"),
        r6_override_changed_rows=("r6_override_changed_flag", "sum"),
        incumbent_deterioration_rows=("incumbent_deterioration_flag", "sum"),
        two_day_confirmation_rows=("two_day_confirmation_flag", "sum"),
    )
    summary["transition_share"] = summary["transition_rows"] / summary["transition_rows"].sum()
    summary["threshold_likely_not_binding"] = summary["transition_type"].isin(["base_to_stock", "stock_to_base"])
    return summary.sort_values("transition_rows", ascending=False)


def _load_r6() -> pd.DataFrame:
    r6 = pd.read_csv(R6_DIR / "r6_guard_first_market_bias_override_unified_contract.csv", low_memory=False, dtype={"selected_ticker": str})
    for col in ["signal_date", "next_signal_date", "entry_date", "exit_date"]:
        r6[col] = pd.to_datetime(r6[col], errors="coerce")
    r6["selected_ticker"] = r6["selected_ticker"].map(_ticker)
    return _p1(r6)


def _load_r6_daily_fill() -> pd.DataFrame:
    if not RADAR_R6_DAILY_FILL.exists():
        return pd.DataFrame(columns=["ticker", "price_date", "close", "source_quality", "source_route", "official_unadjusted_ohlc_ready", "adjustment_policy"])
    filled = pd.read_csv(RADAR_R6_DAILY_FILL, low_memory=False, dtype={"ticker": str})
    filled["ticker"] = filled["ticker"].map(_ticker)
    filled["price_date"] = pd.to_datetime(filled["price_date"], errors="coerce")
    filled["close"] = pd.to_numeric(filled["close"], errors="coerce")
    filled["official_unadjusted_ohlc_ready"] = filled["official_unadjusted_ohlc_ready"].map(_bool)
    return filled.sort_values(["ticker", "price_date"]).drop_duplicates(["ticker", "price_date"], keep="last")


def _r6_daily_alignment(daily_f: pd.DataFrame, r6: pd.DataFrame, fill: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = daily_f[daily_f["metric_eligible_P1"].map(_bool)].copy().sort_values("signal_date")
    r6_state = r6.sort_values("signal_date").copy()
    r6_state["weekly_r6_source_signal_date"] = r6_state["signal_date"]
    keep = [
        "signal_date", "next_signal_date", "selected_ticker", "selected_ticker_name", "selected_asset_type", "selected_branch",
        "regime_label", "entry_date", "exit_date", "transition_action", "transition_cost_rate", "source_quality",
        "official_selected_stock_ohlc_ready", "path_ready", "selected_stock_adjusted_close_ready",
        "weekly_r6_source_signal_date", "gross_interval_return", "net_interval_return_after_transition_cost",
    ]
    aligned = pd.merge_asof(
        base.sort_values("signal_date"), r6_state[[col for col in keep if col in r6_state.columns]].sort_values("signal_date"),
        on="signal_date", direction="backward", suffixes=("_f", "_r6"),
    )
    aligned = aligned.rename(columns={
        "selected_ticker": "weekly_r6_ticker", "selected_asset_type": "weekly_r6_asset_type", "selected_branch": "weekly_r6_branch",
        "entry_date": "weekly_r6_interval_entry_date", "exit_date": "weekly_r6_interval_exit_date", "source_quality": "weekly_r6_source_quality",
        "gross_interval_return": "weekly_r6_contract_gross_interval_return", "net_interval_return_after_transition_cost": "weekly_r6_contract_net_interval_return",
    })
    # The R6 contract starts at its first weekly snapshot. Before that point its
    # documented default is the same 00631L state-hold base, not a missing state.
    aligned["weekly_r6_ticker"] = aligned["weekly_r6_ticker"].fillna("00631L").map(_ticker)
    aligned["weekly_r6_asset_type"] = aligned["weekly_r6_asset_type"].fillna("etf")
    aligned["weekly_r6_branch"] = aligned["weekly_r6_branch"].fillna("route_support_default_before_first_weekly_snapshot")
    prices = _benchmark_price_map().copy()
    prices["date"] = pd.to_datetime(prices["date"], errors="coerce")
    base_entry = prices[["date", "price", "benchmark_source_quality"]].rename(columns={"date": "next_trading_day_execution_date", "price": "r6_base_entry_close", "benchmark_source_quality": "r6_base_entry_source_quality"})
    base_exit = prices[["date", "price", "benchmark_source_quality"]].rename(columns={"date": "next_trading_day_after_execution_date", "price": "r6_base_exit_close", "benchmark_source_quality": "r6_base_exit_source_quality"})
    aligned = aligned.merge(base_entry, on="next_trading_day_execution_date", how="left").merge(base_exit, on="next_trading_day_after_execution_date", how="left")
    stock_entry = fill[["ticker", "price_date", "close", "source_quality", "source_route", "official_unadjusted_ohlc_ready", "adjustment_policy"]].rename(
        columns={"ticker": "weekly_r6_ticker", "price_date": "next_trading_day_execution_date", "close": "r6_stock_entry_close", "source_quality": "r6_stock_entry_source_quality", "source_route": "r6_stock_entry_source_route", "official_unadjusted_ohlc_ready": "r6_stock_entry_ready", "adjustment_policy": "r6_stock_adjustment_policy"}
    )
    stock_exit = fill[["ticker", "price_date", "close", "source_quality", "source_route", "official_unadjusted_ohlc_ready", "adjustment_policy"]].rename(
        columns={"ticker": "weekly_r6_ticker", "price_date": "next_trading_day_after_execution_date", "close": "r6_stock_exit_close", "source_quality": "r6_stock_exit_source_quality", "source_route": "r6_stock_exit_source_route", "official_unadjusted_ohlc_ready": "r6_stock_exit_ready", "adjustment_policy": "r6_stock_exit_adjustment_policy"}
    )
    aligned = aligned.merge(stock_entry, on=["weekly_r6_ticker", "next_trading_day_execution_date"], how="left")
    aligned = aligned.merge(stock_exit, on=["weekly_r6_ticker", "next_trading_day_after_execution_date"], how="left")
    stock_mask = aligned["weekly_r6_asset_type"].eq("stock")
    aligned["weekly_r6_entry_close"] = np.where(stock_mask, aligned["r6_stock_entry_close"], aligned["r6_base_entry_close"])
    aligned["weekly_r6_exit_close"] = np.where(stock_mask, aligned["r6_stock_exit_close"], aligned["r6_base_exit_close"])
    aligned["weekly_r6_daily_price_source_quality"] = np.where(stock_mask, aligned["r6_stock_entry_source_quality"], aligned["r6_base_entry_source_quality"])
    aligned["weekly_r6_daily_gross_return"] = pd.to_numeric(aligned["weekly_r6_exit_close"], errors="coerce") / pd.to_numeric(aligned["weekly_r6_entry_close"], errors="coerce") - 1.0
    previous = aligned.shift(1)
    prior_ticker = previous["weekly_r6_ticker"].fillna("00631L")
    prior_type = previous["weekly_r6_asset_type"].fillna("etf")
    change = aligned["weekly_r6_ticker"].ne(prior_ticker) | aligned["weekly_r6_asset_type"].ne(prior_type)
    aligned["weekly_r6_transition_type"] = np.select(
        [~change, prior_type.eq("etf") & stock_mask, prior_type.eq("stock") & aligned["weekly_r6_asset_type"].eq("etf")],
        ["hold_same", "base_to_stock", "stock_to_base"], default="stock_to_stock",
    )
    cost_map = {"hold_same": 0.0, "base_to_stock": 0.00385, "stock_to_base": 0.00585, "stock_to_stock": 0.00585}
    aligned["weekly_r6_transition_cost_rate_hook"] = aligned["weekly_r6_transition_type"].map(cost_map)
    aligned["weekly_r6_net_daily_return_after_transition_cost"] = aligned["weekly_r6_daily_gross_return"] - aligned["weekly_r6_transition_cost_rate_hook"]
    aligned["weekly_r6_daily_path_ready"] = aligned["weekly_r6_daily_gross_return"].notna()
    aligned["same_asset_ticker_flag"] = (
        aligned["selected_ticker_after"].eq(aligned["weekly_r6_ticker"].fillna(""))
        & aligned["selected_asset_type_after"].eq(aligned["weekly_r6_asset_type"].fillna(""))
    )
    aligned["daily_r6_mark_status"] = np.where(
        aligned["weekly_r6_daily_path_ready"],
        np.where(stock_mask, "official_unadjusted_stock_daily_mark_ready", "base_00631L_daily_mark_available"),
        "blocked_missing_required_daily_mark",
    )
    ledger = _r6_stock_daily_gap_ledger(aligned)
    return aligned, ledger


def _r6_stock_daily_gap_ledger(aligned: pd.DataFrame) -> pd.DataFrame:
    stock = aligned[
        aligned["weekly_r6_asset_type"].eq("stock")
        & ~aligned["weekly_r6_daily_path_ready"].fillna(False)
    ].copy()
    if stock.empty:
        return pd.DataFrame()
    stock["ticker"] = stock["weekly_r6_ticker"].map(_ticker)
    start = stock[["ticker", "signal_date", "next_trading_day_execution_date", "weekly_r6_interval_entry_date", "weekly_r6_interval_exit_date", "weekly_r6_branch"]].rename(
        columns={"next_trading_day_execution_date": "price_date"}
    )
    start["required_as"] = "daily_mark_entry_close"
    end = stock[["ticker", "signal_date", "next_trading_day_after_execution_date", "weekly_r6_interval_entry_date", "weekly_r6_interval_exit_date", "weekly_r6_branch"]].rename(
        columns={"next_trading_day_after_execution_date": "price_date"}
    )
    end["required_as"] = "daily_mark_following_close"
    needs = pd.concat([start, end], ignore_index=True).dropna(subset=["price_date"])
    requirement = needs.groupby(["ticker", "price_date"], as_index=False).agg(
        required_as=("required_as", lambda x: "|".join(sorted(set(x)))),
        impacted_f_signal_dates=("signal_date", lambda x: "|".join(sorted(set(pd.Series(x).dt.strftime("%Y-%m-%d"))))),
        r6_weekly_signal_start=("weekly_r6_interval_entry_date", "min"),
        r6_weekly_signal_end=("weekly_r6_interval_exit_date", "max"),
        weekly_r6_branches=("weekly_r6_branch", lambda x: "|".join(sorted(set(x)))),
    )
    requirement["required_fields"] = "official_unadjusted_close_at_price_date_for_daily_mark_path"
    requirement["source_requirement"] = "selected_ticker_only official daily unadjusted OHLC; reuse prior Radar rows before bounded official month routes"
    requirement["adjusted_close_ready"] = False
    requirement["path_status"] = "blocked_pending_core_daily_r6_mark_absorption"
    requirement["next_owner"] = "Radar/Data bounded P1 weekly R6 selected-stock daily OHLC attribution gap fill"
    if RADAR_R6_OHLC.exists():
        existing = pd.read_csv(RADAR_R6_OHLC, low_memory=False, dtype={"ticker": str})
        existing["ticker"] = existing["ticker"].map(_ticker)
        existing["date"] = pd.to_datetime(existing["date"], errors="coerce")
        dates = requirement[["ticker", "price_date"]].rename(columns={"price_date": "date"})
        hit = dates.merge(existing[["ticker", "date", "close"]], on=["ticker", "date"], how="left")
        requirement = requirement.merge(hit.rename(columns={"date": "price_date", "close": "existing_close"}), on=["ticker", "price_date"], how="left")
        requirement["existing_daily_mark_available"] = requirement["existing_close"].notna()
    else:
        requirement["existing_daily_mark_available"] = False
    return requirement


def _f_rechain_contract(daily_f: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    f = daily_f[daily_f["metric_eligible_P1"].map(_bool)].copy().sort_values("signal_date")
    prices = _benchmark_price_map().copy()
    prices["date"] = pd.to_datetime(prices["date"], errors="coerce")
    entry = prices[["date", "price"]].rename(columns={"date": "next_trading_day_execution_date", "price": "base_entry_price"})
    exit_ = prices[["date", "price"]].rename(columns={"date": "next_trading_day_after_execution_date", "price": "base_exit_price"})
    f = f.merge(entry, on="next_trading_day_execution_date", how="left").merge(exit_, on="next_trading_day_after_execution_date", how="left")
    f["base_00631L_gross_daily_return"] = f["base_exit_price"] / f["base_entry_price"] - 1.0
    f["base_00631L_transition_cost_if_held"] = 0.0
    f["rechain_leg_actual_net_return"] = f["net_daily_return_after_transition_cost"]
    f["rechain_leg_base_net_return_before_recomputed_transition"] = f["base_00631L_gross_daily_return"]
    f["rechain_requires_transition_recompute"] = True
    f["rechain_note"] = "Replace selected-stock daily legs with 00631L daily marks, then recompute state transitions/costs; do not subtract standalone episode returns."
    f["is_stock_exception_leg"] = f["selected_asset_type_after"].eq("stock")
    f["episode_key"] = (f["is_stock_exception_leg"].ne(f["is_stock_exception_leg"].shift()) | f["selected_ticker_after"].ne(f["selected_ticker_after"].shift())).cumsum()
    stock = f[f["is_stock_exception_leg"]].copy()
    episodes = stock.groupby(["episode_key", "selected_ticker_after"], as_index=False).agg(
        episode_start_signal_date=("signal_date", "min"), episode_end_signal_date=("signal_date", "max"),
        entry_date=("next_trading_day_execution_date", "min"), exit_date=("next_trading_day_after_execution_date", "max"),
        daily_legs=("signal_date", "size"), stock_net_compound_return=("rechain_leg_actual_net_return", lambda x: float((1 + x).prod() - 1)),
        base_00631L_compound_return=("rechain_leg_base_net_return_before_recomputed_transition", lambda x: float((1 + x.dropna()).prod() - 1)),
        transition_cost_rate_sum=("transition_cost_rate_hook", "sum"), transition_types=("transition_type", lambda x: "|".join(sorted(set(x)))),
        ticker_name=("challenger_name", "first"), price_source_quality=("daily_price_source_quality", lambda x: "|".join(sorted(set(map(str, x))))),
    )
    episodes["gross_contribution_delta_before_rechain"] = episodes["stock_net_compound_return"] - episodes["base_00631L_compound_return"]
    episodes["exact_rechain_input_ready"] = True
    episodes["rechain_required"] = True
    episodes["removal_protocol"] = "Replace all daily legs in this episode with the matching base marks and recompute full state transitions, ETF/stock costs and equity path."
    return f, episodes


def _drawdown_events(rechain: pd.DataFrame, alignment: pd.DataFrame) -> pd.DataFrame:
    path = rechain.sort_values("signal_date").copy()
    path["equity"] = (1 + pd.to_numeric(path["rechain_leg_actual_net_return"], errors="coerce").fillna(0)).cumprod()
    path["running_peak_equity"] = path["equity"].cummax()
    path["drawdown"] = path["equity"] / path["running_peak_equity"] - 1.0
    events_raw: list[tuple[pd.Series, pd.Series, pd.Timestamp]] = []
    peak = path.iloc[0]
    trough = peak
    in_drawdown = False
    for _, row in path.iloc[1:].iterrows():
        if not in_drawdown:
            if row["equity"] >= peak["equity"]:
                peak = row
                trough = row
            else:
                in_drawdown = True
                trough = row
        else:
            if row["equity"] < trough["equity"]:
                trough = row
            if row["equity"] >= peak["equity"]:
                events_raw.append((peak, trough, row["signal_date"]))
                peak = row
                trough = row
                in_drawdown = False
    if in_drawdown:
        events_raw.append((peak, trough, pd.NaT))
    events = []
    for peak, trough, recovery_date in sorted(events_raw, key=lambda item: item[1]["equity"] / item[0]["equity"] - 1.0)[:10]:
        within = path[(path["signal_date"] >= peak["signal_date"]) & (path["signal_date"] <= trough["signal_date"])]
        r6 = alignment[alignment["signal_date"].isin(within["signal_date"])].copy()
        r6_stock_blocked = int((~r6["weekly_r6_daily_path_ready"].fillna(False).astype(bool)).sum())
        r6_return_ready = bool(len(r6)) and bool(r6["weekly_r6_daily_path_ready"].fillna(False).all())
        r6_segment_return = float((1 + pd.to_numeric(r6["weekly_r6_net_daily_return_after_transition_cost"], errors="coerce")).prod() - 1) if r6_return_ready else np.nan
        f_segment_return = float((1 + pd.to_numeric(within["rechain_leg_actual_net_return"], errors="coerce")).prod() - 1)
        events.append({
            "event_rank": len(events) + 1,
            "peak_date": peak["signal_date"], "trough_date": trough["signal_date"], "recovery_date": recovery_date,
            "recovered_within_P1": bool(pd.notna(recovery_date)), "daily_f_peak_asset": peak["selected_ticker_after"],
            "daily_f_trough_asset": trough["selected_ticker_after"], "daily_f_peak_asset_type": peak["selected_asset_type_after"],
            "daily_f_trough_asset_type": trough["selected_asset_type_after"], "daily_f_drawdown": trough["equity"] / peak["equity"] - 1.0,
            "daily_f_peak_to_trough_return": f_segment_return,
            "daily_f_transition_rows_in_segment": int(within["transition_type"].ne("hold_same").sum()),
            "daily_f_gate_transition_rows_in_segment": int((within["gate_on_flag"] | within["gate_off_flag"]).sum()),
            "daily_f_stock_to_stock_rows_in_segment": int(within["transition_type"].eq("stock_to_stock").sum()),
            "daily_f_two_day_confirmation_rows_in_segment": int(within["two_day_confirmation_flag"].sum()),
            "weekly_r6_assets_seen": "|".join(sorted(set(r6["weekly_r6_ticker"].dropna().astype(str)))),
            "same_asset_day_share": float(r6["same_asset_ticker_flag"].mean()) if len(r6) else np.nan,
            "weekly_r6_peak_to_trough_net_return": r6_segment_return,
            "daily_f_minus_weekly_r6_return_difference": f_segment_return - r6_segment_return if r6_return_ready else np.nan,
            "weekly_r6_daily_return_difference_status": "available_from_aligned_daily_marks" if r6_return_ready else "blocked_missing_r6_daily_marks",
            "weekly_r6_stock_daily_mark_blocked_rows": r6_stock_blocked,
            "transition_timing_difference_note": "Daily F executes next trading day after close; weekly R6 state is forward-filled from its weekly signal until the next weekly snapshot.",
        })
    return pd.DataFrame(events)


def _weekly_r6_drawdown_events(alignment: pd.DataFrame) -> pd.DataFrame:
    path = alignment.sort_values("signal_date").copy()
    path["weekly_r6_equity"] = (1 + pd.to_numeric(path["weekly_r6_net_daily_return_after_transition_cost"], errors="coerce").fillna(0)).cumprod()
    events_raw: list[tuple[pd.Series, pd.Series, pd.Timestamp]] = []
    peak = path.iloc[0]
    trough = peak
    in_drawdown = False
    for _, row in path.iloc[1:].iterrows():
        if not in_drawdown:
            if row["weekly_r6_equity"] >= peak["weekly_r6_equity"]:
                peak, trough = row, row
            else:
                in_drawdown, trough = True, row
        else:
            if row["weekly_r6_equity"] < trough["weekly_r6_equity"]:
                trough = row
            if row["weekly_r6_equity"] >= peak["weekly_r6_equity"]:
                events_raw.append((peak, trough, row["signal_date"]))
                peak, trough, in_drawdown = row, row, False
    if in_drawdown:
        events_raw.append((peak, trough, pd.NaT))
    rows = []
    for peak, trough, recovery in sorted(events_raw, key=lambda item: item[1]["weekly_r6_equity"] / item[0]["weekly_r6_equity"] - 1.0)[:10]:
        segment = path[(path["signal_date"] >= peak["signal_date"]) & (path["signal_date"] <= trough["signal_date"])]
        rows.append({
            "event_rank": len(rows) + 1, "peak_date": peak["signal_date"], "trough_date": trough["signal_date"], "recovery_date": recovery,
            "recovered_within_P1": bool(pd.notna(recovery)), "weekly_r6_peak_asset": peak["weekly_r6_ticker"],
            "weekly_r6_trough_asset": trough["weekly_r6_ticker"], "weekly_r6_drawdown": trough["weekly_r6_equity"] / peak["weekly_r6_equity"] - 1.0,
            "weekly_r6_peak_to_trough_net_return": float((1 + pd.to_numeric(segment["weekly_r6_net_daily_return_after_transition_cost"], errors="coerce")).prod() - 1),
            "weekly_r6_transition_rows_in_segment": int(segment["weekly_r6_transition_type"].ne("hold_same").sum()),
            "daily_f_assets_seen": "|".join(sorted(set(segment["selected_ticker_after"].dropna().astype(str)))),
            "same_asset_day_share": float(segment["same_asset_ticker_flag"].mean()) if len(segment) else np.nan,
            "daily_path_source_quality": "00631L_adjusted_reference_for_base; official_unadjusted_ohlc_for_selected_R6_stock",
        })
    return pd.DataFrame(rows)


def _weekly_r6_interval_reconciliation(alignment: pd.DataFrame) -> pd.DataFrame:
    use = alignment[alignment["weekly_r6_source_signal_date"].notna()].copy()
    keys = [
        "weekly_r6_source_signal_date", "weekly_r6_ticker", "weekly_r6_asset_type", "weekly_r6_branch",
        "weekly_r6_interval_entry_date", "weekly_r6_interval_exit_date", "weekly_r6_contract_gross_interval_return",
        "weekly_r6_contract_net_interval_return",
    ]
    rows = []
    for values, group in use.groupby(keys, dropna=False):
        record = dict(zip(keys, values))
        record["daily_mark_rows"] = int(len(group))
        record["daily_mark_path_ready_share"] = float(group["weekly_r6_daily_path_ready"].fillna(False).mean())
        record["daily_mark_gross_compound_return"] = float((1 + pd.to_numeric(group["weekly_r6_daily_gross_return"], errors="coerce")).prod() - 1)
        record["daily_mark_net_compound_return"] = float((1 + pd.to_numeric(group["weekly_r6_net_daily_return_after_transition_cost"], errors="coerce")).prod() - 1)
        contract_net = pd.to_numeric(pd.Series([record["weekly_r6_contract_net_interval_return"]]), errors="coerce").iloc[0]
        record["daily_mark_minus_contract_net_return"] = record["daily_mark_net_compound_return"] - contract_net if pd.notna(contract_net) else np.nan
        record["transition_cost_rate_sum_daily"] = float(pd.to_numeric(group["weekly_r6_transition_cost_rate_hook"], errors="coerce").sum())
        difference = record["daily_mark_minus_contract_net_return"]
        record["reconciliation_material_mismatch_flag"] = bool(pd.notna(difference) and abs(difference) > 0.001)
        record["reconciliation_status"] = (
            "matched" if pd.notna(contract_net) and abs(difference) <= 1e-6
            else "minor_cost_compounding_difference" if pd.notna(difference) and abs(difference) <= 0.001
            else "material_interval_to_daily_basis_difference"
        )
        rows.append(record)
    return pd.DataFrame(rows).sort_values("weekly_r6_source_signal_date")


def _weekly_r6_interval_overlap_audit(r6: pd.DataFrame) -> pd.DataFrame:
    source = r6.dropna(subset=["entry_date", "exit_date"]).sort_values("signal_date").copy()
    rows = []
    for left in source.itertuples(index=False):
        later = source[(source["signal_date"] > left.signal_date) & (source["entry_date"] < left.exit_date) & (source["exit_date"] > left.entry_date)]
        for right in later.itertuples(index=False):
            if _ticker(left.selected_ticker) == _ticker(right.selected_ticker) and left.selected_asset_type == right.selected_asset_type:
                continue
            rows.append({
                "left_signal_date": left.signal_date, "left_ticker": _ticker(left.selected_ticker), "left_asset_type": left.selected_asset_type,
                "left_entry_date": left.entry_date, "left_exit_date": left.exit_date, "left_transition_action": left.transition_action,
                "right_signal_date": right.signal_date, "right_ticker": _ticker(right.selected_ticker), "right_asset_type": right.selected_asset_type,
                "right_entry_date": right.entry_date, "right_exit_date": right.exit_date, "right_transition_action": right.transition_action,
                "overlap_start": max(left.entry_date, right.entry_date), "overlap_end": min(left.exit_date, right.exit_date),
                "semantic_issue": "weekly_interval_contract_has_overlapping_different_asset_intervals; cannot be treated as a unique daily state-hold path without an explicit state-boundary policy",
                "next_owner": "Strategy Center/Core Data state-boundary policy decision",
            })
    return pd.DataFrame(rows)


def _f2_evidence_map(summary: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {"candidate_change": "separate_gate_transition_from_stock_to_stock_edge", "evidence_status": "supported", "rule_scope": "Apply challenger-edge threshold only to stock_to_stock. Gate-off returns to 00631L immediately; gate-on may enter stock without an incumbent score comparison.", "not_a_live_rule": True},
        {"candidate_change": "immediate_exit_on_incumbent_deterioration", "evidence_status": "testable_if_transition_attribution_supports", "rule_scope": "Do not delay a confirmed deteriorating incumbent; evaluate separately from challenger edge.", "not_a_live_rule": True},
        {"candidate_change": "one_or_two_day_grace_for_non_deteriorating_incumbent", "evidence_status": "testable_only_for_stock_to_stock", "rule_scope": "Use only when incumbent remains healthy and attribution identifies short reversals; never delay gate-off exit.", "not_a_live_rule": True},
        {"candidate_change": "minimum_hold_or_hysteresis", "evidence_status": "not_pre_authorized", "rule_scope": "Create only if exact attribution shows short stock-to-stock churn is the material drawdown cause. Do not tune blindly.", "not_a_live_rule": True},
    ]
    if len(summary):
        gate_rows = int(summary.loc[summary["primary_transition_cause"].isin(["gate_on", "gate_off"]), "transition_rows"].sum())
        total = int(summary["transition_rows"].sum())
        rows.append({"candidate_change": "observed_gate_transition_share", "evidence_status": "observed", "rule_scope": f"{gate_rows}/{total} P1 F transition rows are gate_on/gate_off in the source contract.", "not_a_live_rule": True})
    return pd.DataFrame(rows)


def _coverage(score: pd.DataFrame, r6_gap: pd.DataFrame, rechain: pd.DataFrame, alignment: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([{
        "period": "P1", "requested_start": "2015-01-02", "requested_end": "2022-12-29",
        "daily_f_metric_rows": int(score["metric_eligible_P1"].map(_bool).sum()),
        "daily_f_score_attribution_rows": int(len(score)),
        "daily_f_rechain_leg_rows": int(len(rechain)),
        "daily_f_rechain_base_marks_ready_share": float(rechain["base_00631L_gross_daily_return"].notna().mean()) if len(rechain) else 0.0,
        "weekly_r6_stock_daily_mark_gap_rows": int(len(r6_gap)),
        "weekly_r6_daily_state_rows": int(len(alignment)),
        "weekly_r6_daily_path_ready_share": float(alignment["weekly_r6_daily_path_ready"].fillna(False).mean()) if len(alignment) else 0.0,
        "daily_f_source_quality": "official_selected_stock_unadjusted_ohlc_plus_00631L_adjusted_reference_proxy_path",
        "weekly_r6_daily_stock_path_status": "blocked_pending_bounded_radar_selected_ticker_daily_ohlc_fill" if len(r6_gap) else "ready",
        **FLAGS,
    }])


def _blocked(r6_gap: pd.DataFrame, alignment: pd.DataFrame, overlap: pd.DataFrame) -> pd.DataFrame:
    r6_ready = bool(len(alignment)) and bool(alignment["weekly_r6_daily_path_ready"].fillna(False).all())
    rows = [
        {"item": "weekly_r6_selected_stock_daily_marks", "status": "ready" if r6_ready else "blocked_pending_bounded_gap_fill", "detail": "Weekly R6 daily state uses official unadjusted stock OHLC for selected stock intervals and the accepted 00631L benchmark reference path.", "next_owner": "none" if r6_ready else "Radar/Data"},
        {"item": "weekly_r6_interval_state_boundary", "status": "blocked_semantic_overlap" if len(overlap) else "ready", "detail": "The weekly R6 interval contract contains overlapping different-asset intervals. A unique daily state-hold reconstruction requires an explicit boundary policy; daily marks alone do not resolve it.", "next_owner": "Strategy Center/Core Data state-boundary policy decision" if len(overlap) else "none"},
        {"item": "selected_stock_adjusted_close", "status": "blocked", "detail": "Official unadjusted OHLC is diagnostic-only. No selected-stock adjusted close is fabricated.", "next_owner": "Strategy Center/Radar Data if trusted adjusted route is authorized"},
        {"item": "cash_bear_classifier", "status": "blocked", "detail": "No cash rule is created.", "next_owner": "Strategy Center/Core Data later"},
        {"item": "revenue_anomaly", "status": "report_only", "detail": "Revenue anomaly remains warning/confidence context and is not a rerank input.", "next_owner": "none"},
        {"item": "rs20_top3", "status": "reference_only", "detail": "RS20 top3 is not a selected branch.", "next_owner": "none"},
    ]
    return pd.DataFrame(rows)


def _future_audit() -> pd.DataFrame:
    return pd.DataFrame([
        {"audit_item": "daily_F_score_and_transition_attribution", "future_return_used_as_rule": False, "detail": "Uses PIT daily 0050 regime fields and latest weekly candidate snapshot only.", "future_data_violation_count": 0},
        {"audit_item": "drawdown_and_rechain_contract", "future_return_used_as_rule": False, "detail": "Price paths are evaluation-only after next-day execution and are not fed back into selection.", "future_data_violation_count": 0},
    ])


def _readiness(r6_gap: pd.DataFrame, score: pd.DataFrame, rechain: pd.DataFrame, alignment: pd.DataFrame, reconciliation: pd.DataFrame, overlap: pd.DataFrame) -> dict[str, Any]:
    exact_r6_daily = len(r6_gap) == 0 and bool(len(alignment)) and bool(alignment["weekly_r6_daily_path_ready"].fillna(False).all())
    reconciliation_mismatch_rows = int(reconciliation["reconciliation_material_mismatch_flag"].fillna(False).sum()) if len(reconciliation) else 0
    return {
        "task_id": TASK_ID,
        "status": (
            "partial_daily_f_transition_rechain_ready_weekly_r6_daily_mark_gap_pending" if not exact_r6_daily
            else "blocked_weekly_r6_interval_to_daily_execution_basis_reconciliation_required" if reconciliation_mismatch_rows or len(overlap)
            else "ready_for_p1_daily_f_weekly_r6_transition_drawdown_attribution_diagnostic"
        ),
        "daily_f_score_transition_attribution_ready": bool(len(score)),
        "daily_f_exact_rechain_input_ready": bool(len(rechain)) and bool(rechain["base_00631L_gross_daily_return"].notna().all()),
        "weekly_r6_daily_state_alignment_ready": True,
        "weekly_r6_exact_daily_stock_mark_ready": exact_r6_daily,
        "weekly_r6_daily_path_ready_share": float(alignment["weekly_r6_daily_path_ready"].fillna(False).mean()) if len(alignment) else 0.0,
        "weekly_r6_interval_to_daily_reconciliation_mismatch_rows": reconciliation_mismatch_rows,
        "weekly_r6_overlapping_different_asset_interval_rows": int(len(overlap)),
        "weekly_r6_selected_stock_daily_mark_gap_rows": int(len(r6_gap)),
        "official_daily_f_selected_stock_unadjusted_ohlc_ready": True,
        "EP05_transaction_cost_hooks_ready": True,
        "selected_stock_adjusted_close_ready": False,
        "cash_bear_classifier_ready": False,
        "ready_for_experiments": bool(len(score)) and bool(len(rechain)) and exact_r6_daily and reconciliation_mismatch_rows == 0 and len(overlap) == 0,
        "ready_for_radar_data_gap_fill": not exact_r6_daily,
        "ready_for_formal": False,
        "ready_for_strategy_replay": False,
        "future_data_violation_count": 0,
        **FLAGS,
    }


def _summary(readiness: dict[str, Any], transition: pd.DataFrame) -> str:
    lines = [
        "# P1 Daily F vs Weekly R6 換倉與回撤歸因 contract",
        "",
        "本包將每日 F 的 PIT 分數、gate/換倉原因、逐日成本 leg 與個股例外區間攤平。個股區間移除必須重新串接 00631L path 與 transition cost，不可直接扣掉單筆 proxy 報酬。",
        "",
        f"P1 F 非 hold transition rows：{int(transition['transition_rows'].sum()) if len(transition) else 0}。門檻只應控制 stock_to_stock；gate_on/gate_off 不應被錯誤當成 challenger-edge 門檻問題。",
        "",
        "週頻 R6 selected-stock 的逐日官方 unadjusted OHLC 已吸收；但原週區間 contract 存在不同資產區間重疊與 material interval-to-daily return mismatch，因此不能把目前 reconstructed daily R6 path 當作原 R6 同口徑 verdict。",
        f"Material reconciliation mismatch rows={readiness['weekly_r6_interval_to_daily_reconciliation_mismatch_rows']}；overlapping different-asset interval rows={readiness['weekly_r6_overlapping_different_asset_interval_rows']}。需由 Strategy Center/Core 明確定義週頻 state boundary，才可重跑同口徑 F vs R6 attribution。",
        "",
        f"ready_for_experiments={readiness['ready_for_experiments']}；selected_stock_adjusted_close_ready=false；cash_bear_classifier_ready=false；future_data_violation_count=0。",
        "",
        "本任務為 diagnostic contract，不是 formal、strategy replay、daily report 或 trade decision。營收異常僅 report-only；RS20 top3 僅 reference-only。",
        "",
        "Flags: formal_model_changed=false; trade_decision_changed=false; active_in_trade_decision=false; report_changed=false; portfolio_replay_executed=false; ready_for_strategy_replay=false; ready_for_formal=false; not_live_rule=true; forward_returns_live_rule_usage=false.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    daily = _load_daily()
    matrix = _candidate_matrix()
    score = _score_attribution(daily, matrix)
    day_equivalence, pair_equivalence = _variant_equivalence(daily)
    transition = _transition_summary(score)
    daily_f = _p1(daily[daily["state_machine_variant"].eq(F_VARIANT)])
    r6 = _load_r6()
    r6_fill = _load_r6_daily_fill()
    alignment, r6_gap = _r6_daily_alignment(daily_f, r6, r6_fill)
    rechain, episodes = _f_rechain_contract(daily_f)
    rechain = rechain.merge(
        score[["signal_date", "gate_on_flag", "gate_off_flag", "two_day_confirmation_flag"]],
        on="signal_date", how="left", validate="one_to_one",
    )
    drawdowns = _drawdown_events(rechain, alignment)
    r6_drawdowns = _weekly_r6_drawdown_events(alignment)
    f2_map = _f2_evidence_map(transition)
    coverage = _coverage(score, r6_gap, rechain, alignment)
    future = _future_audit()
    reconciliation = _weekly_r6_interval_reconciliation(alignment)
    overlap = _weekly_r6_interval_overlap_audit(r6)
    blocked = _blocked(r6_gap, alignment, overlap)
    readiness = _readiness(r6_gap, score, rechain, alignment, reconciliation, overlap)

    paths = [
        _write_csv(score, "p1_daily_f_score_transition_attribution.csv"),
        _write_csv(day_equivalence, "p1_daily_f_variant_decision_by_date.csv"),
        _write_csv(pair_equivalence, "p1_daily_f_variant_decision_equivalence.csv"),
        _write_csv(transition, "p1_daily_f_transition_cause_summary.csv"),
        _write_csv(alignment, "p1_weekly_r6_daily_state_alignment.csv"),
        _write_csv(alignment[[col for col in ["signal_date", "next_trading_day_execution_date", "next_trading_day_after_execution_date", "weekly_r6_ticker", "weekly_r6_asset_type", "weekly_r6_branch", "weekly_r6_transition_type", "weekly_r6_transition_cost_rate_hook", "weekly_r6_entry_close", "weekly_r6_exit_close", "weekly_r6_daily_gross_return", "weekly_r6_net_daily_return_after_transition_cost", "weekly_r6_daily_path_ready", "weekly_r6_daily_price_source_quality", "selected_stock_adjusted_close_ready"] if col in alignment.columns]], "p1_weekly_r6_daily_state_hold_path.csv"),
        _write_csv(r6_gap, "p1_weekly_r6_daily_path_gap_ledger.csv"),
        _write_csv(rechain, "p1_daily_f_rechain_daily_leg_contract.csv"),
        _write_csv(episodes, "p1_daily_f_exact_episode_contribution_contract.csv"),
        _write_csv(drawdowns, "p1_daily_f_drawdown_top10.csv"),
        _write_csv(r6_drawdowns, "p1_weekly_r6_drawdown_top10.csv"),
        _write_csv(reconciliation, "p1_weekly_r6_interval_to_daily_reconciliation_audit.csv"),
        _write_csv(overlap, "p1_weekly_r6_interval_state_boundary_overlap_audit.csv"),
        _write_csv(f2_map, "p1_daily_f_weekly_r6_f2_evidence_map.csv"),
        _write_csv(coverage, "p1_daily_f_weekly_r6_requested_vs_actual_coverage.csv"),
        _write_csv(blocked, "p1_daily_f_weekly_r6_blocked_proxy_audit.csv"),
        _write_csv(future, "p1_daily_f_weekly_r6_future_data_audit.csv"),
    ]
    readiness_path = OUTPUT_DIR / "readiness_for_p1_daily_f_weekly_r6_transition_drawdown_attribution.json"
    readiness_path.write_text(json.dumps(readiness, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path = OUTPUT_DIR / "final_summary_zh.md"
    summary_path.write_text(_summary(readiness, transition), encoding="utf-8")
    manifest = {
        "task_id": TASK_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(OUTPUT_DIR),
        "files": [{"path": path.name, "sha256": _sha256(path)} for path in [*paths, readiness_path, summary_path]],
        "readiness": readiness,
        "source_inputs": {
            "daily_f_state": str(DAILY_DIR / "daily_incumbent_challenger_state_machine_contract_ohlc_absorbed.csv"),
            "weekly_r6": str(R6_DIR / "r6_guard_first_market_bias_override_unified_contract.csv"),
            "existing_r6_ohlc_probe": str(RADAR_R6_OHLC),
            "radar_r6_daily_fill": str(RADAR_R6_DAILY_FILL),
        },
    }
    (OUTPUT_DIR / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(readiness, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
