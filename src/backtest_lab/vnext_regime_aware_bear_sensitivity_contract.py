from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backtest_lab import vnext_confirmed_bear_cash_classifier_contract as bear_source


REPO_ROOT = Path(__file__).resolve().parents[2]
DAILY_CONTEXT = REPO_ROOT / "outputs" / "vnext_daily_incumbent_challenger_state_machine_contract_ohlc_absorbed_20260710" / "daily_incumbent_challenger_state_machine_contract_ohlc_absorbed.csv"
R6_UNIFIED = REPO_ROOT / "outputs" / "vnext_r6_guard_first_market_bias_override_unified_contract_20260709" / "r6_guard_first_market_bias_override_unified_contract.csv"
OUTPUT_DIR = REPO_ROOT / "outputs" / "vnext_regime_aware_bear_sensitivity_contract_20260710"

TASK_ID = "TASK-BACKTEST-CORE-VNEXT-REGIME-AWARE-BEAR-SENSITIVITY-CONTRACT-001"
RAW_F_VARIANT = "F_two_day_confirmation_and_risk_adjusted_edge"
BASES = ["reconstructed_single_position_R6", "raw_Daily_F_challenger"]
PERIODS = bear_source.PERIODS
VARIANTS = [
    "R0_no_cash_R6",
    "R1_universal_B1_reference",
    "R2_regime_tiered",
    "R3_regime_tiered_mega_hard_only",
    "R4_regime_tiered_bull_strict",
    "R5_low_churn_tiered",
]
FLAGS = bear_source.FLAGS


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write(frame: pd.DataFrame, name: str) -> Path:
    path = OUTPUT_DIR / name
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def _bool(value: Any) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, str):
        return value.lower() in {"true", "1", "yes"}
    return bool(value)


def _daily_regime(signal_dates: pd.Series) -> pd.DataFrame:
    context = pd.read_csv(DAILY_CONTEXT, low_memory=False)
    context = context[context["state_machine_variant"].eq(RAW_F_VARIANT)].copy()
    context["signal_date"] = pd.to_datetime(context["signal_date"], errors="coerce")
    context = context.dropna(subset=["signal_date"]).sort_values("signal_date").drop_duplicates("signal_date")
    context["bull_c2_market_health_flag"] = context["c2_pass_flag"].map(_bool)
    dates = pd.DataFrame({"signal_date": pd.to_datetime(signal_dates).sort_values().unique()})
    context = dates.merge(context[["signal_date", "bull_c2_market_health_flag"]], on="signal_date", how="left")
    r6 = pd.read_csv(R6_UNIFIED, low_memory=False)
    r6["r6_snapshot_date"] = pd.to_datetime(r6["signal_date"], errors="coerce")
    r6["mega_r6_override_flag"] = r6["r6_override_flag"].map(_bool)
    r6 = r6.dropna(subset=["r6_snapshot_date"]).sort_values("r6_snapshot_date").drop_duplicates("r6_snapshot_date")
    context = pd.merge_asof(
        context.sort_values("signal_date"),
        r6[["r6_snapshot_date", "mega_r6_override_flag"]],
        left_on="signal_date",
        right_on="r6_snapshot_date",
        direction="backward",
    )
    context["mega_r6_override_flag"] = context["mega_r6_override_flag"].fillna(False).astype(bool)
    context["regime_label"] = np.select(
        [context["mega_r6_override_flag"], context["bull_c2_market_health_flag"]],
        ["mega", "bull"],
        default="ordinary",
    )
    context["regime_priority"] = context["regime_label"].map({"mega": 1, "bull": 2, "ordinary": 3})
    context["regime_priority_policy"] = "mega_R6_override_first__bull_C2_second__ordinary_otherwise"
    context["mega_regime_source_quality"] = "accepted_R6_unified_breakout_breadth_override_weekly_PIT_snapshot_asof_daily"
    context["bull_regime_source_quality"] = "daily_0050_C2_close_above_MA60_return20_40_nonnegative_PIT"
    context["ordinary_regime_source_quality"] = "derived_non_mega_non_bull"
    context["regime_future_return_used"] = False
    keep = [
        "signal_date", "regime_label", "regime_priority", "regime_priority_policy",
        "mega_r6_override_flag", "bull_c2_market_health_flag",
        "mega_regime_source_quality", "bull_regime_source_quality", "ordinary_regime_source_quality",
        "regime_future_return_used",
    ]
    return context[keep]


def _policy(variant: str, regime: str, hard_recovery: bool) -> dict[str, Any]:
    if variant == "R0_no_cash_R6":
        return {"tier": "B0", "entry_days": 0, "recovery_days": 0, "policy_key": "R0"}
    if variant == "R1_universal_B1_reference":
        return {"tier": "B1", "entry_days": 2, "recovery_days": 2, "policy_key": "universal_B1"}
    mapping = {"ordinary": "B1", "bull": "B4", "mega": "B5"}
    if variant == "R4_regime_tiered_bull_strict":
        mapping = {"ordinary": "B1", "bull": "B5", "mega": "B5"}
    tier = mapping[regime]
    entry_days = 3 if regime == "mega" and variant in {"R3_regime_tiered_mega_hard_only", "R4_regime_tiered_bull_strict"} else 2
    recovery_days = 2
    if variant == "R5_low_churn_tiered":
        recovery_days = 2 if hard_recovery else 3
    return {
        "tier": tier,
        "entry_days": entry_days,
        "recovery_days": recovery_days,
        "policy_key": f"{regime}:{tier}:entry{entry_days}",
    }


def _tier_conditions(row: Any, tier: str) -> tuple[bool, bool, bool]:
    if tier == "B0":
        return True, False, True
    ready = bool(getattr(row, f"{tier}_feature_ready"))
    entry = bool(getattr(row, f"{tier}_entry_condition")) if ready else False
    recovery = bool(getattr(row, f"{tier}_recovery_condition")) if ready else False
    return ready, entry, recovery


def _reason(variant: str, regime: str, tier: str, active: bool) -> str:
    if not active:
        return "non_bear_hold_original_strategy_target"
    return f"confirmed_bear__variant={variant}__regime={regime}__sensitivity={tier}"


def _materialize() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base_paths = bear_source._load_base_paths()
    base_paths = base_paths[base_paths["base_strategy"].isin(BASES)].copy()
    features = bear_source._feature_matrix(base_paths["signal_date"].dropna().drop_duplicates())
    regimes = _daily_regime(base_paths["signal_date"])
    feature_matrix = features.merge(regimes, on="signal_date", how="left")
    merged = base_paths.merge(feature_matrix, on="signal_date", how="left")
    output: list[dict[str, Any]] = []
    for base_strategy, base in merged.groupby("base_strategy", sort=False):
        base = base.sort_values("signal_date")
        for variant in VARIANTS:
            previous_ticker, previous_type = "00631L", "etf"
            bear_active, confirmation_streak, recovery_streak, cash_episode_id = False, 0, 0, 0
            previous_policy_key = ""
            for item in base.itertuples(index=False):
                regime = str(item.regime_label)
                hard_recovery = bool(item.bull_c2_market_health_flag)
                policy = _policy(variant, regime, hard_recovery)
                policy_changed = bool(previous_policy_key and previous_policy_key != policy["policy_key"])
                if policy_changed:
                    confirmation_streak = 0
                    recovery_streak = 0
                feature_ready, entry_condition, recovery_condition = _tier_conditions(item, policy["tier"])
                confirmation_streak = confirmation_streak + 1 if entry_condition else 0
                recovery_streak = recovery_streak + 1 if bear_active and recovery_condition else 0
                event = "bear_state_unchanged"
                if not bear_active and policy["tier"] != "B0" and confirmation_streak >= policy["entry_days"]:
                    bear_active = True
                    recovery_streak = 0
                    cash_episode_id += 1
                    event = "confirmed_bear_enter_cash"
                elif bear_active and recovery_streak >= policy["recovery_days"]:
                    bear_active = False
                    confirmation_streak = 0
                    event = "confirmed_recovery_exit_cash"
                target_ticker = "CASH" if bear_active else item.base_target_ticker
                target_type = "cash" if bear_active else item.base_target_asset_type
                transition_type, cost_key, cost = bear_source._transition(previous_ticker, previous_type, target_ticker, target_type)
                gross_return = 0.0 if target_type == "cash" else item.base_gross_daily_return
                path_ready = pd.notna(gross_return)
                net_return = gross_return - float(cost["transition_cost_rate"]) if path_ready else np.nan
                output.append({
                    "task": TASK_ID,
                    "base_strategy": base_strategy,
                    "regime_bear_variant": variant,
                    "signal_date": item.signal_date,
                    "next_trading_day_execution_date": item.next_trading_day_execution_date,
                    "next_trading_day_after_execution_date": item.next_trading_day_after_execution_date,
                    "regime_label": regime,
                    "regime_priority": item.regime_priority,
                    "regime_priority_policy": item.regime_priority_policy,
                    "mega_r6_override_flag": item.mega_r6_override_flag,
                    "bull_c2_market_health_flag": item.bull_c2_market_health_flag,
                    "regime_source_quality": item.mega_regime_source_quality if regime == "mega" else item.bull_regime_source_quality if regime == "bull" else item.ordinary_regime_source_quality,
                    "bear_sensitivity_tier": policy["tier"],
                    "required_confirmation_days": policy["entry_days"],
                    "required_recovery_days": policy["recovery_days"],
                    "hard_trend_recovery_flag": hard_recovery,
                    "tier_policy_changed_flag": policy_changed,
                    "classifier_feature_ready": feature_ready,
                    "classifier_missing_input_policy": "default_to_non_cash_original_strategy_target_never_infer_bear",
                    "bear_entry_condition": entry_condition,
                    "bear_recovery_condition": recovery_condition,
                    "confirmation_streak": confirmation_streak,
                    "recovery_streak": recovery_streak,
                    "confirmed_bear_state": bear_active,
                    "bear_state_event": event,
                    "bear_reason": _reason(variant, regime, policy["tier"], bear_active),
                    "cash_episode_id": cash_episode_id if bear_active else 0,
                    "base_target_ticker": item.base_target_ticker,
                    "base_target_asset_type": item.base_target_asset_type,
                    "incumbent_ticker_before": previous_ticker,
                    "incumbent_asset_type_before": previous_type,
                    "selected_ticker_after": target_ticker,
                    "selected_asset_type_after": target_type,
                    "transition_type": transition_type,
                    "transition_cost_key": cost_key,
                    "transition_cost_rate_hook": float(cost["transition_cost_rate"]),
                    "sell_fee_rate_hook": float(cost.get("sell_fee_rate", 0.0)),
                    "buy_fee_rate_hook": float(cost.get("buy_fee_rate", 0.0)),
                    "securities_transaction_tax_rate_hook": float(cost.get("tax_rate", 0.0)),
                    "transition_cost_twd_per_1m_notional": float(cost["transition_cost_rate"]) * 1_000_000,
                    "transition_cost_model_status": "EP05_TaiwanCostModel_stock_ETF_cash_hooks_separated",
                    "gross_daily_return": gross_return,
                    "net_daily_return_after_transition_cost": net_return,
                    "daily_path_ready": bool(path_ready),
                    "terminal_path_row_excluded_from_metric": bool(item.terminal_path_row_excluded_from_metric),
                    "cash_return_policy": "zero_return_no_interest_only_during_confirmed_bear",
                    "base_path_source_quality": item.base_path_source_quality,
                    "medium_trend_break_component": item.medium_trend_break_component,
                    "long_trend_break_component": item.long_trend_break_component,
                    "pool80_breadth_deterioration_component": item.pool80_breadth_deterioration_component,
                    "bear_component_count": item.bear_component_count,
                    "revenue_anomaly_role": "report_only_not_cash_rule",
                    "rs20_top3_role": "reference_only_not_selected",
                    "selected_stock_adjusted_close_ready": False if target_type == "stock" else True,
                    "future_data_violation_count": 0,
                    "diagnostic_only": True,
                    **FLAGS,
                })
                previous_ticker, previous_type = target_ticker, target_type
                previous_policy_key = policy["policy_key"]
    state = pd.DataFrame(output)
    for period, (start, end) in PERIODS.items():
        candidate = (
            state["signal_date"].between(pd.Timestamp(start), pd.Timestamp(end))
            & state["next_trading_day_execution_date"].le(pd.Timestamp(end))
            & state["next_trading_day_after_execution_date"].le(pd.Timestamp(end))
        )
        state[f"metric_candidate_{period}"] = candidate
        state[f"metric_eligible_{period}"] = candidate & state["daily_path_ready"]
    return state, feature_matrix, regimes


def _metrics(state: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (base, variant), group in state.groupby(["base_strategy", "regime_bear_variant"]):
        group = group.sort_values("signal_date")
        for period in PERIODS:
            sub = group[group[f"metric_eligible_{period}"]].copy()
            if sub.empty:
                continue
            equity = (1.0 + sub["net_daily_return_after_transition_cost"]).cumprod()
            total_return = float(equity.iloc[-1] - 1.0)
            mdd = float((equity / equity.cummax() - 1.0).min())
            annualized = float(equity.iloc[-1] ** (252.0 / len(sub)) - 1.0)
            rows.append({
                "base_strategy": base, "regime_bear_variant": variant, "period": period,
                "actual_start": sub["signal_date"].min(), "actual_end": sub["signal_date"].max(), "daily_rows": len(sub),
                "net_total_return_after_transition_cost_hook": total_return,
                "net_MDD_hook": mdd,
                "annualized_net_return_hook": annualized,
                "calmar_like_hook": annualized / abs(mdd) if mdd < 0 else np.nan,
                "cash_days": int(sub["selected_asset_type_after"].eq("cash").sum()),
                "cash_exposure_share": float(sub["selected_asset_type_after"].eq("cash").mean()),
                "transition_count": int(sub["transition_type"].ne("hold_same").sum()),
                "metric_role": "Core_materialized_hook_for_Experiments_verdict_not_Core_judgment",
                **FLAGS,
            })
    return pd.DataFrame(rows)


def _episodes(state: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (base, variant), group in state.groupby(["base_strategy", "regime_bear_variant"]):
        group = group.sort_values("signal_date").copy()
        cash = group["selected_asset_type_after"].eq("cash")
        group["cash_block"] = cash.ne(cash.shift()).cumsum()
        for _, episode in group[cash].groupby("cash_block"):
            rows.append({
                "base_strategy": base, "regime_bear_variant": variant,
                "cash_episode_id": int(episode["cash_episode_id"].max()),
                "signal_start": episode["signal_date"].min(), "signal_end": episode["signal_date"].max(),
                "execution_start": episode["next_trading_day_execution_date"].min(), "execution_end": episode["next_trading_day_after_execution_date"].max(),
                "cash_days": len(episode), "regimes": "|".join(sorted(set(episode["regime_label"]))),
                "sensitivity_tiers": "|".join(sorted(set(episode["bear_sensitivity_tier"]))),
                "bear_reasons": "|".join(sorted(set(episode["bear_reason"]))),
            })
    return pd.DataFrame(rows)


def _exposure(state: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    by_regime = state.groupby(["base_strategy", "regime_bear_variant", "regime_label"], as_index=False).agg(
        state_days=("signal_date", "size"), cash_days=("selected_asset_type_after", lambda x: int((x == "cash").sum())),
        cash_exposure_share=("selected_asset_type_after", lambda x: float((x == "cash").mean())),
        cash_entry_count=("bear_state_event", lambda x: int((x == "confirmed_bear_enter_cash").sum())),
    )
    temp = state.copy()
    temp["year"] = temp["signal_date"].dt.year
    by_year = temp.groupby(["base_strategy", "regime_bear_variant", "year"], as_index=False).agg(
        state_days=("signal_date", "size"), cash_days=("selected_asset_type_after", lambda x: int((x == "cash").sum())),
        cash_exposure_share=("selected_asset_type_after", lambda x: float((x == "cash").mean())),
        transition_count=("transition_type", lambda x: int((x != "hold_same").sum())),
    )
    period_rows = []
    for (base, variant), group in state.groupby(["base_strategy", "regime_bear_variant"]):
        for period, (start, end) in PERIODS.items():
            sub = group[group["signal_date"].between(pd.Timestamp(start), pd.Timestamp(end))]
            period_rows.append({
                "base_strategy": base, "regime_bear_variant": variant, "period": period,
                "state_days": len(sub), "cash_days": int(sub["selected_asset_type_after"].eq("cash").sum()),
                "cash_exposure_share": float(sub["selected_asset_type_after"].eq("cash").mean()) if len(sub) else 0.0,
                "cash_majority_flag": bool(sub["selected_asset_type_after"].eq("cash").mean() > 0.5) if len(sub) else False,
            })
    return by_regime, by_year, pd.DataFrame(period_rows)


def _weak_year_attribution(metrics: pd.DataFrame, state: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (base, variant), group in state[state["signal_date"].dt.year.isin([2015, 2018, 2022])].groupby(["base_strategy", "regime_bear_variant"]):
        for year, sub in group.groupby(group["signal_date"].dt.year):
            sub = sub[sub["daily_path_ready"]].sort_values("signal_date")
            equity = (1 + sub["net_daily_return_after_transition_cost"]).cumprod()
            rows.append({
                "base_strategy": base, "regime_bear_variant": variant, "year": int(year),
                "net_return_after_cost_hook": float(equity.iloc[-1] - 1),
                "MDD_hook": float((equity / equity.cummax() - 1).min()),
                "cash_days": int(sub["selected_asset_type_after"].eq("cash").sum()),
                "cash_exposure_share": float(sub["selected_asset_type_after"].eq("cash").mean()),
                "transition_count": int(sub["transition_type"].ne("hold_same").sum()),
            })
    return pd.DataFrame(rows)


def _p2_damage(metrics: pd.DataFrame) -> pd.DataFrame:
    p2 = metrics[metrics["period"].eq("P2")].copy()
    baseline = p2[p2["regime_bear_variant"].eq("R0_no_cash_R6")][[
        "base_strategy", "net_total_return_after_transition_cost_hook", "net_MDD_hook", "transition_count",
    ]].rename(columns={
        "net_total_return_after_transition_cost_hook": "R0_P2_net_return",
        "net_MDD_hook": "R0_P2_MDD",
        "transition_count": "R0_P2_transition_count",
    })
    out = p2.merge(baseline, on="base_strategy", how="left")
    out["P2_return_damage_vs_R0"] = out["net_total_return_after_transition_cost_hook"] - out["R0_P2_net_return"]
    out["P2_MDD_change_vs_R0"] = out["net_MDD_hook"] - out["R0_P2_MDD"]
    out["P2_transition_change_vs_R0"] = out["transition_count"] - out["R0_P2_transition_count"]
    return out


def _coverage(state: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (base, variant), group in state.groupby(["base_strategy", "regime_bear_variant"]):
        for period, (start, end) in PERIODS.items():
            sub = group[group["signal_date"].between(pd.Timestamp(start), pd.Timestamp(end))]
            candidate = sub[sub[f"metric_candidate_{period}"]]
            ready = sub[sub[f"metric_eligible_{period}"]]
            rows.append({
                "base_strategy": base, "regime_bear_variant": variant, "period": period,
                "requested_start": start, "requested_end": end,
                "actual_start": ready["signal_date"].min() if len(ready) else pd.NaT,
                "actual_end": ready["signal_date"].max() if len(ready) else pd.NaT,
                "candidate_rows": len(candidate), "ready_rows": len(ready),
                "path_ready_share": float(len(ready) / len(candidate)) if len(candidate) else 1.0,
                "regime_ready_share": float(sub["regime_label"].isin(["ordinary", "bull", "mega"]).mean()) if len(sub) else np.nan,
                "classifier_feature_ready_share": float(sub["classifier_feature_ready"].mean()) if len(sub) else np.nan,
                **FLAGS,
            })
    return pd.DataFrame(rows)


def _policy_map() -> pd.DataFrame:
    rows = [
        {"variant": "R0_no_cash_R6", "ordinary": "B0", "bull": "B0", "mega": "B0", "entry": "never", "recovery": "n/a"},
        {"variant": "R1_universal_B1_reference", "ordinary": "B1", "bull": "B1", "mega": "B1", "entry": "2 days", "recovery": "2 days"},
        {"variant": "R2_regime_tiered", "ordinary": "B1", "bull": "B4", "mega": "B5", "entry": "2 days", "recovery": "2 days"},
        {"variant": "R3_regime_tiered_mega_hard_only", "ordinary": "B1", "bull": "B4", "mega": "B5", "entry": "ordinary/bull 2 days; mega 3 days", "recovery": "2 days"},
        {"variant": "R4_regime_tiered_bull_strict", "ordinary": "B1", "bull": "B5", "mega": "B5", "entry": "ordinary/bull 2 days; mega 3 days", "recovery": "2 days"},
        {"variant": "R5_low_churn_tiered", "ordinary": "B1", "bull": "B4", "mega": "B5", "entry": "2 days", "recovery": "3 days; C2 hard recovery 2 days"},
    ]
    return pd.DataFrame(rows).assign(
        regime_priority="mega_R6_override > bull_C2 > ordinary",
        policy_change_resets_streak=True,
        confirmed_bear_state_survives_single_day_regime_change=True,
        future_return_used=False,
        diagnostic_only=True,
        **FLAGS,
    )


def _missingness(feature_matrix: pd.DataFrame, regimes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, columns, source in [
        ("daily_regime", ["regime_label", "mega_r6_override_flag", "bull_c2_market_health_flag"], "daily PIT context; mega uses accepted R6 override asof"),
        ("B1_medium", ["B1_feature_ready", "B1_entry_condition"], "0050 adjusted daily PIT"),
        ("B4_consensus", ["B4_feature_ready", "B4_entry_condition"], "0050 daily PIT + weekly pool80 breadth asof proxy"),
        ("B5_strict", ["B5_feature_ready", "B5_entry_condition"], "0050 daily PIT + weekly pool80 breadth asof proxy"),
    ]:
        table = regimes if name == "daily_regime" else feature_matrix
        available = table[columns].notna().all(axis=1)
        rows.append({
            "field_group": name, "columns": "|".join(columns), "source_quality": source,
            "available_rows": int(available.sum()), "total_rows": len(table), "coverage_share": float(available.mean()),
            "future_data_violation_count": 0,
        })
    return pd.DataFrame(rows)


def _future_audit() -> pd.DataFrame:
    return pd.DataFrame([
        {"audit_item": "daily_regime_priority", "future_return_used_as_rule": False, "detail": "Mega uses existing R6 breakout/breadth PIT override; bull uses same-day C2; ordinary is residual.", "future_data_violation_count": 0},
        {"audit_item": "bear_sensitivity", "future_return_used_as_rule": False, "detail": "B1/B4/B5 are existing PIT trigger components; streaks use current and prior days only.", "future_data_violation_count": 0},
        {"audit_item": "execution", "future_return_used_as_rule": False, "detail": "Signal-close state executes at next-trading-day close; later prices are evaluation path only.", "future_data_violation_count": 0},
    ])


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    state, feature_matrix, regimes = _materialize()
    metrics = _metrics(state)
    episodes = _episodes(state)
    by_regime, by_year, by_period = _exposure(state)
    coverage = _coverage(state)
    gaps = state[state["metric_candidate_full_integrated"] & ~state["daily_path_ready"]][[
        "base_strategy", "regime_bear_variant", "signal_date", "selected_ticker_after", "base_path_source_quality",
    ]].copy()
    path_ready_min = float(coverage["path_ready_share"].min())
    regime_ready_min = float(coverage["regime_ready_share"].min())
    max_cash_share = float(by_period["cash_exposure_share"].max())
    readiness = {
        "task_id": TASK_ID,
        "status": "regime_aware_bear_sensitivity_ready_bounded_diagnostic" if not len(gaps) and path_ready_min == 1.0 and regime_ready_min == 1.0 else "regime_aware_bear_sensitivity_partial",
        "base_strategy_count": 2,
        "variant_count": 6,
        "P1_primary": True,
        "P2_secondary_stop_gate": True,
        "daily_regime_priority_ready": regime_ready_min == 1.0,
        "daily_regime_priority": "mega_R6_override > bull_C2 > ordinary",
        "daily_single_position_path_ready_share_min": path_ready_min,
        "base_path_gap_rows": len(gaps),
        "maximum_period_cash_exposure_share": max_cash_share,
        "cash_majority_variant_present": bool(max_cash_share > 0.5),
        "EP05_stock_ETF_cash_transition_cost_hooks_ready": True,
        "selected_stock_adjusted_close_ready": False,
        "revenue_anomaly_used_as_cash_rule": False,
        "rs20_top3_used_as_selected_branch": False,
        "ready_for_experiments": bool(not len(gaps) and path_ready_min == 1.0 and regime_ready_min == 1.0),
        "ready_for_formal": False,
        "ready_for_strategy_replay": False,
        "future_data_violation_count": 0,
        **FLAGS,
    }
    blocked = pd.DataFrame([
        {"item": "pool80_daily_breadth", "status": "proxy", "detail": "B4/B5 use latest weekly Layer4 PIT breadth snapshot asof daily; not daily recalculation.", "impact": "diagnostic-only"},
        {"item": "selected_stock_adjusted_close", "status": "blocked", "detail": "Official unadjusted stock OHLC remains diagnostic-only.", "impact": "not formal-ready"},
        {"item": "cash_interest", "status": "proxy", "detail": "Cash return fixed at zero without interest.", "impact": "explicit diagnostic assumption"},
        {"item": "slippage", "status": "blocked", "detail": "No slippage model invented beyond EP05 fee/tax hooks.", "impact": "Experiments must retain blocker"},
    ])
    output_paths = [
        _write(regimes, "regime_aware_bear_daily_regime_labels.csv"),
        _write(_policy_map(), "regime_aware_bear_sensitivity_policy_map.csv"),
        _write(feature_matrix, "regime_aware_bear_trigger_feature_matrix.csv"),
        _write(state, "regime_aware_bear_sensitivity_daily_state_contract.csv"),
        _write(state[state["transition_type"].ne("hold_same")], "regime_aware_bear_sensitivity_transition_trace.csv"),
        _write(episodes, "regime_aware_bear_sensitivity_cash_episode_trace.csv"),
        _write(by_regime, "regime_aware_bear_cash_exposure_by_regime.csv"),
        _write(by_year, "regime_aware_bear_cash_exposure_by_year.csv"),
        _write(by_period, "regime_aware_bear_cash_exposure_by_period.csv"),
        _write(metrics, "regime_aware_bear_net_mdd_calmar_hooks.csv"),
        _write(_weak_year_attribution(metrics, state), "regime_aware_bear_P1_weak_year_attribution.csv"),
        _write(_p2_damage(metrics), "regime_aware_bear_P2_cash_damage_attribution.csv"),
        _write(coverage, "regime_aware_bear_requested_vs_actual_coverage.csv"),
        _write(_missingness(feature_matrix, regimes), "regime_aware_bear_source_quality_missingness.csv"),
        _write(gaps, "regime_aware_bear_base_path_gap_ledger.csv"),
        _write(bear_source._cost_audit(), "regime_aware_bear_EP05_cost_audit.csv"),
        _write(blocked, "regime_aware_bear_blocked_proxy_audit.csv"),
        _write(_future_audit(), "regime_aware_bear_future_data_audit.csv"),
    ]
    readiness_path = OUTPUT_DIR / "readiness_for_regime_aware_bear_sensitivity_diagnostic.json"
    readiness_path.write_text(json.dumps(readiness, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path = OUTPUT_DIR / "final_summary_zh.md"
    summary_path.write_text(
        "# Regime-aware Bear Sensitivity Contract\n\n"
        "本 contract 只測 R0-R5 六個 bounded 變體，regime priority 固定為 mega > bull > ordinary。\n\n"
        f"- daily path coverage min: {path_ready_min:.1%}; regime coverage min: {regime_ready_min:.1%}; path gaps: {len(gaps)}\n"
        "- ordinary uses B1 sensitivity；bull uses B4/B5；mega uses B5 with optional 3-day confirmation\n"
        "- signal close decision -> next-trading-day close execution；single position；cash return=0\n"
        "- EP05 stock/ETF/cash transition costs included；gross/no-cost is not the primary hook\n"
        "- revenue anomaly report-only；RS20 top3 reference-only；adjusted stock close blocked\n\n"
        f"ready_for_experiments={readiness['ready_for_experiments']}; ready_for_formal=false; future_data_violation_count=0.\n",
        encoding="utf-8",
    )
    manifest = {
        "task_id": TASK_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(OUTPUT_DIR),
        "files": [{"path": path.name, "sha256": _sha256(path)} for path in [*output_paths, readiness_path, summary_path]],
        "readiness": readiness,
        "source_inputs": {
            "confirmed_bear_source_runner": str(Path(bear_source.__file__)),
            "daily_regime_context": str(DAILY_CONTEXT),
            "accepted_R6_unified_context": str(R6_UNIFIED),
        },
    }
    (OUTPUT_DIR / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(readiness, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
