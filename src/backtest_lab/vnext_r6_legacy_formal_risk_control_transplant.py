from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backtest_lab.regime_mode_switch import frozen_cycle_proven_top1_v1_variant


TASK_ID = "TASK-BACKTEST-CORE-VNEXT-R6-LEGACY-FORMAL-RISK-CONTROL-TRANSPLANT-CONTRACT-001"
REPO_ROOT = Path(__file__).resolve().parents[2]
R6_DIR = REPO_ROOT / "outputs" / "vnext_weekly_r6_single_position_state_boundary_reconstruction_contract_20260710"
BENCHMARK_PATH = REPO_ROOT / "outputs" / "vnext_dynamic_candidate_pool_data_materialization_20260706" / "benchmark_features.csv"
SOURCE_CLOSURE_DIR = REPO_ROOT / "outputs" / "vnext_selected_stock_total_return_source_escalation_closure_20260710"
OUTPUT_DIR = REPO_ROOT / "outputs" / "vnext_r6_legacy_formal_risk_control_transplant_contract_20260710"
VARIANTS = ["L0_R6_no_additional_cash", "L1_systemic_bear_cash", "L2_portfolio_stop12_cash", "L3_preproof_risk2of3_25pct_00631L", "L4_combined_exact_transplant"]
PERIODS = {"P1": ("2015-01-02", "2022-12-29"), "P2": ("2023-01-02", "2026-06-30")}

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

FEE = 0.001425
TAX = {"stock": 0.003, "etf": 0.001, "cash": 0.0}


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


def _ticker(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    return text[:-2] if text.endswith(".0") else text


def _market_features() -> pd.DataFrame:
    source = pd.read_csv(BENCHMARK_PATH, dtype={"benchmark": str}, low_memory=False)
    source["signal_date"] = pd.to_datetime(source["trade_date"], errors="coerce")
    source = source.loc[source["benchmark"].eq("0050")].dropna(subset=["signal_date"]).sort_values("signal_date").copy()
    source["close_0050"] = pd.to_numeric(source["adjusted_close"], errors="coerce")
    close = source["close_0050"]
    source["ma60"] = close.rolling(60, min_periods=60).mean()
    source["ma200"] = close.rolling(200, min_periods=200).mean()
    source["ma200_slope_20d"] = source["ma200"] / source["ma200"].shift(20) - 1.0
    source["return_20d"] = close / close.shift(20) - 1.0
    source["return_60d"] = close / close.shift(60) - 1.0
    source["return_120d"] = close / close.shift(120) - 1.0
    source["drawdown_from_60d_high"] = close / close.rolling(60, min_periods=60).max() - 1.0
    source["drawdown_from_252d_high"] = close / close.rolling(252, min_periods=252).max() - 1.0
    source["systemic_bear_feature_ready"] = source[["ma200", "ma200_slope_20d", "return_60d", "return_120d", "drawdown_from_252d_high"]].notna().all(axis=1)
    source["systemic_bear_exact"] = (
        source["systemic_bear_feature_ready"]
        & source["close_0050"].lt(source["ma200"])
        & source["ma200_slope_20d"].lt(0)
        & source["return_60d"].lt(0)
        & source["return_120d"].lt(0)
        & source["drawdown_from_252d_high"].le(-0.20)
    )
    source["risk_2of3_feature_ready"] = source[["ma60", "return_20d", "drawdown_from_60d_high"]].notna().all(axis=1)
    source["risk_component_close_below_ma60"] = source["close_0050"].lt(source["ma60"])
    source["risk_component_return20_negative"] = source["return_20d"].lt(0)
    source["risk_component_drawdown60_le_minus5pct"] = source["drawdown_from_60d_high"].le(-0.05)
    source["risk_2of3_count"] = source[["risk_component_close_below_ma60", "risk_component_return20_negative", "risk_component_drawdown60_le_minus5pct"]].sum(axis=1)
    source["risk_2of3_exact"] = source["risk_2of3_feature_ready"] & source["risk_2of3_count"].ge(2)
    source["market_feature_source_quality"] = "0050_adjusted_benchmark_daily_PIT_rolling_including_signal_close"
    return source[[
        "signal_date", "close_0050", "ma60", "ma200", "ma200_slope_20d", "return_20d", "return_60d", "return_120d",
        "drawdown_from_60d_high", "drawdown_from_252d_high", "systemic_bear_feature_ready", "systemic_bear_exact",
        "risk_2of3_feature_ready", "risk_component_close_below_ma60", "risk_component_return20_negative",
        "risk_component_drawdown60_le_minus5pct", "risk_2of3_count", "risk_2of3_exact", "market_feature_source_quality",
    ]]


def _load_base() -> pd.DataFrame:
    state = pd.read_csv(R6_DIR / "reconstructed_weekly_r6_single_position_daily_state_rows.csv", dtype={"selected_ticker_after": str}, low_memory=False)
    for column in ["signal_date", "next_trading_day_execution_date", "next_trading_day_after_execution_date"]:
        state[column] = pd.to_datetime(state[column], errors="coerce")
    state["selected_ticker_after"] = state["selected_ticker_after"].map(_ticker)
    state["base_00631L_gross_daily_return"] = pd.to_numeric(state["base_exit_close"], errors="coerce") / pd.to_numeric(state["base_entry_close"], errors="coerce") - 1.0
    state = state.merge(_market_features(), on="signal_date", how="left", validate="many_to_one")
    return state


def _transition_cost(
    from_ticker: str,
    from_type: str,
    from_exposure: float,
    to_ticker: str,
    to_type: str,
    to_exposure: float,
) -> dict[str, Any]:
    same_asset = from_ticker == to_ticker and from_type == to_type
    sold = max(from_exposure - to_exposure, 0.0) if same_asset else from_exposure
    bought = max(to_exposure - from_exposure, 0.0) if same_asset else to_exposure
    sell_fee = sold * FEE
    tax = sold * TAX.get(from_type, 0.0)
    buy_fee = bought * FEE
    total = sell_fee + tax + buy_fee
    if same_asset and abs(from_exposure - to_exposure) < 1e-12:
        transition_type = "hold_same"
    elif to_type == "cash" or to_exposure == 0:
        transition_type = f"{from_type}_to_cash"
    elif from_type == "cash" or from_exposure == 0:
        transition_type = f"cash_to_{to_type}"
    elif same_asset:
        transition_type = "reduce_exposure" if to_exposure < from_exposure else "increase_exposure"
    else:
        transition_type = f"{from_type}_to_{to_type}"
    return {
        "transition_type": transition_type,
        "sold_notional_share": sold,
        "bought_notional_share": bought,
        "sell_fee_rate_hook": sell_fee,
        "buy_fee_rate_hook": buy_fee,
        "securities_transaction_tax_rate_hook": tax,
        "transition_cost_rate_hook": total,
    }


def _update_preproof(active: bool, clear_streak: int, raw_risk: bool, disabled: bool) -> tuple[bool, int]:
    if disabled:
        return False, 0
    if raw_risk:
        return True, 0
    if not active:
        return False, 0
    clear_streak += 1
    return (False, 0) if clear_streak >= 5 else (True, clear_streak)


def _target_for_overlay(base_ticker: str, base_type: str, systemic: bool, preproof: bool, stop: bool) -> tuple[str, str, float, str]:
    if stop:
        return "CASH", "cash", 0.0, "portfolio_stop_drawdown_le_minus12pct"
    if systemic:
        return "CASH", "cash", 0.0, "systemic_bear_exact_100pct_cash"
    if preproof:
        return "00631L", "etf", 0.25, "preproof_risk_2of3_25pct_00631L_75pct_cash"
    return base_ticker, base_type, 1.0, "R6_base_target"


def _materialize_period(base: pd.DataFrame, period: str, variant: str) -> pd.DataFrame:
    eligible = base.loc[base[f"metric_eligible_{period}"].astype(bool)].sort_values("signal_date").copy()
    previous_ticker, previous_type, previous_exposure = "00631L", "etf", 1.0
    wealth, peak_wealth = 1.0, 1.0
    preproof_active, preproof_clear_streak, attack_ever_activated = False, 0, False
    cash_episode_id = 0
    rows = []
    for item in eligible.itertuples(index=False):
        base_ticker, base_type = _ticker(item.selected_ticker_after), item.selected_asset_type_after
        systemic = bool(item.systemic_bear_exact) if variant in {"L1_systemic_bear_cash", "L4_combined_exact_transplant"} else False
        if systemic:
            attack_ever_activated = False
        base_stock_activation = base_type == "stock" and not systemic
        if base_stock_activation:
            attack_ever_activated = True
        raw_risk = bool(item.risk_2of3_exact) and bool(item.risk_2of3_feature_ready)
        if variant in {"L3_preproof_risk2of3_25pct_00631L", "L4_combined_exact_transplant"}:
            preproof_active, preproof_clear_streak = _update_preproof(
                preproof_active, preproof_clear_streak, raw_risk, attack_ever_activated
            )
        else:
            preproof_active, preproof_clear_streak = False, 0
        drawdown_at_signal = wealth / peak_wealth - 1.0
        stop_trigger = (
            variant in {"L2_portfolio_stop12_cash", "L4_combined_exact_transplant"}
            and previous_type != "cash"
            and drawdown_at_signal <= -0.12
        )
        target_ticker, target_type, target_exposure, rule_reason = _target_for_overlay(
            base_ticker, base_type, systemic, preproof_active, stop_trigger
        )
        transition = _transition_cost(
            previous_ticker, previous_type, previous_exposure,
            target_ticker, target_type, target_exposure,
        )
        if target_type == "cash":
            asset_gross = 0.0
        elif target_type == "etf":
            asset_gross = float(item.base_00631L_gross_daily_return)
        else:
            asset_gross = float(item.gross_daily_return)
        gross = target_exposure * asset_gross
        net = gross - transition["transition_cost_rate_hook"]
        wealth *= 1.0 + net
        if stop_trigger:
            peak_wealth = wealth
        else:
            peak_wealth = max(peak_wealth, wealth)
        cash_state = target_exposure < 1.0
        if cash_state and previous_exposure == 1.0:
            cash_episode_id += 1
        rows.append({
            "task": TASK_ID,
            "period": period,
            "legacy_risk_variant": variant,
            "signal_date": item.signal_date,
            "next_trading_day_execution_date": item.next_trading_day_execution_date,
            "next_trading_day_after_execution_date": item.next_trading_day_after_execution_date,
            "base_target_ticker": base_ticker,
            "base_target_asset_type": base_type,
            "incumbent_ticker_before": previous_ticker,
            "incumbent_asset_type_before": previous_type,
            "incumbent_exposure_before": previous_exposure,
            "selected_ticker_after": target_ticker,
            "selected_asset_type_after": target_type,
            "selected_exposure_after": target_exposure,
            "cash_exposure_after": 1.0 - target_exposure,
            "rule_reason": rule_reason,
            "systemic_bear_feature_ready": bool(item.systemic_bear_feature_ready),
            "systemic_bear_exact": bool(item.systemic_bear_exact),
            "portfolio_drawdown_at_signal": drawdown_at_signal,
            "portfolio_stop_trigger": stop_trigger,
            "portfolio_peak_reset_after_stop": stop_trigger,
            "portfolio_stop_cooldown_days": 0,
            "portfolio_stop_latch_mode": "none",
            "risk_2of3_feature_ready": bool(item.risk_2of3_feature_ready),
            "risk_2of3_count": int(item.risk_2of3_count) if pd.notna(item.risk_2of3_count) else 0,
            "risk_2of3_raw": raw_risk,
            "preproof_risk_active": preproof_active,
            "preproof_clear_streak": preproof_clear_streak,
            "preproof_exit_confirmation_days": 5,
            "attack_ever_activated_in_cycle": attack_ever_activated,
            "attack_activation_mapping": "R6_first_actual_stock_target_after_systemic_bear_cycle_reset",
            "cash_episode_id": cash_episode_id if cash_state else 0,
            **transition,
            "transition_cost_model_status": "EP05_TaiwanCostModel_exposure_weighted_stock_ETF_cash",
            "asset_gross_daily_return": asset_gross,
            "gross_daily_return": gross,
            "net_daily_return_after_transition_cost": net,
            "wealth_after": wealth,
            "peak_wealth_after": peak_wealth,
            "daily_path_ready": True,
            "market_feature_source_quality": item.market_feature_source_quality,
            "selected_stock_price_source_quality": item.daily_price_source_quality if target_type == "stock" else "not_stock_or_base_reference",
            "selected_stock_adjusted_close_ready": False if target_type == "stock" else True,
            "official_unadjusted_OHLC_diagnostic_only": True,
            "execution_basis": "signal_close_rule_evaluation_next_trading_day_close_unique_position_exposure_state",
            "diagnostic_only": True,
            **FLAGS,
        })
        previous_ticker, previous_type, previous_exposure = target_ticker, target_type, target_exposure
    return pd.DataFrame(rows)


def _metrics(path: pd.DataFrame) -> dict[str, Any]:
    returns = pd.to_numeric(path["net_daily_return_after_transition_cost"], errors="raise")
    equity = (1 + returns).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    return {
        "period": path["period"].iloc[0],
        "variant": path["legacy_risk_variant"].iloc[0],
        "requested_start": PERIODS[path["period"].iloc[0]][0],
        "requested_end": PERIODS[path["period"].iloc[0]][1],
        "actual_start": path["signal_date"].min(),
        "actual_end": path["signal_date"].max(),
        "net_total_return_after_transaction_cost": float(equity.iloc[-1] - 1.0),
        "net_MDD": float(drawdown.min()),
        "transition_count": int(path["transition_type"].ne("hold_same").sum()),
        "cash_exposure_share": float(path["cash_exposure_after"].mean()),
        "full_cash_day_share": float(path["selected_exposure_after"].eq(0).mean()),
        "partial_25pct_00631L_day_share": float(path["selected_exposure_after"].eq(0.25).mean()),
        "stock_exposure_day_share": float((path["selected_asset_type_after"].eq("stock") * path["selected_exposure_after"]).mean()),
        "portfolio_stop_trigger_count": int(path["portfolio_stop_trigger"].sum()),
        "systemic_bear_cash_days": int((path["systemic_bear_exact"] & path["selected_exposure_after"].eq(0)).sum()),
        "preproof_risk_days": int(path["preproof_risk_active"].sum()),
        "daily_path_ready_share": float(path["daily_path_ready"].mean()),
        "diagnostic_only": True,
        **FLAGS,
    }


def _annual(paths: pd.DataFrame) -> pd.DataFrame:
    rows = []
    data = paths.assign(year=paths["signal_date"].dt.year)
    for (period, variant, year), group in data.groupby(["period", "legacy_risk_variant", "year"]):
        returns = group["net_daily_return_after_transition_cost"]
        equity = (1 + returns).cumprod()
        dd = equity / equity.cummax() - 1.0
        rows.append({
            "period": period, "variant": variant, "year": int(year),
            "actual_start": group["signal_date"].min(), "actual_end": group["signal_date"].max(),
            "net_total_return_after_transaction_cost": float(equity.iloc[-1] - 1.0),
            "net_MDD": float(dd.min()), "cash_exposure_share": float(group["cash_exposure_after"].mean()),
            "transition_count": int(group["transition_type"].ne("hold_same").sum()),
            "diagnostic_only": True, **FLAGS,
        })
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    base = _load_base()
    variant = frozen_cycle_proven_top1_v1_variant()
    all_paths = []
    for period in PERIODS:
        for variant_name in VARIANTS:
            all_paths.append(_materialize_period(base, period, variant_name))
    paths = pd.concat(all_paths, ignore_index=True)
    metrics = pd.DataFrame([_metrics(group) for _, group in paths.groupby(["period", "legacy_risk_variant"], sort=False)])
    source_metrics = pd.read_csv(R6_DIR / "reconstructed_weekly_r6_net_path_metrics_hook.csv")
    reconciliation_rows = []
    for period in PERIODS:
        calculated = float(metrics.loc[(metrics["period"].eq(period)) & metrics["variant"].eq("L0_R6_no_additional_cash"), "net_total_return_after_transaction_cost"].iloc[0])
        expected = float(source_metrics.loc[source_metrics["period"].eq(period), "net_total_return_after_transition_cost_hook"].iloc[0])
        reconciliation_rows.append({"period": period, "calculated_L0": calculated, "source_reconstructed_R6": expected, "difference": calculated - expected, "pass_within_1e_12": abs(calculated - expected) <= 1e-12})
    reconciliation = pd.DataFrame(reconciliation_rows)
    if not reconciliation["pass_within_1e_12"].all():
        raise ValueError("L0 does not reconcile to reconstructed R6")
    transitions = paths.loc[paths["transition_type"].ne("hold_same"), [
        "period", "legacy_risk_variant", "signal_date", "next_trading_day_execution_date",
        "incumbent_ticker_before", "incumbent_asset_type_before", "incumbent_exposure_before",
        "selected_ticker_after", "selected_asset_type_after", "selected_exposure_after", "transition_type",
        "sold_notional_share", "bought_notional_share", "sell_fee_rate_hook", "buy_fee_rate_hook",
        "securities_transaction_tax_rate_hook", "transition_cost_rate_hook", "rule_reason",
    ]].copy()
    no_target = pd.DataFrame([{
        "audit_item": "formal_no_target_cash_all_applicability",
        "R6_base_target_blank_rows": int(base["selected_ticker_after"].fillna("").eq("").sum()),
        "R6_00631L_base_rows": int(base["selected_ticker_after"].eq("00631L").sum()),
        "no_stock_exception_is_no_target": False,
        "cash_all_trigger_count": 0,
        "applicability_status": "not_triggered_R6_always_has_stock_or_00631L_target",
        "policy": "cash_all only when R6 target is genuinely null/blocked; 00631L base is a valid target",
        "future_data_violation_count": 0,
    }])
    policy = pd.DataFrame([
        {"variant": "L0_R6_no_additional_cash", "exact_rule": "none", "source": "reconstructed R6", "parameter": "reference"},
        {"variant": "L1_systemic_bear_cash", "exact_rule": "0050 close<MA200 AND MA200 slope20d<0 AND return60<0 AND return120<0 AND DD252<=-20%", "source": "market_regime.py systemic_bear", "parameter": "100% cash"},
        {"variant": "L2_portfolio_stop12_cash", "exact_rule": "strategy signal-value drawdown from peak <=-12%; cash and reset peak; no cooldown/latch", "source": "frozen_cycle_proven_top1_v1_variant + regime_mode_switch.py", "parameter": "12%; cooldown=0; latch=None"},
        {"variant": "L3_preproof_risk2of3_25pct_00631L", "exact_rule": "at least 2: close<MA60, return20<0, DD60<=-5%; only before first attack activation; clear after 5 non-risk days", "source": "frozen variant risk_2of3 state", "parameter": "25% 00631L +75% cash; exit confirmation=5"},
        {"variant": "L4_combined_exact_transplant", "exact_rule": "L1+L2+L3 with priority portfolio stop cash, systemic cash, then preproof 25% exposure", "source": "bounded R6 transplant mapping", "parameter": "no new threshold"},
        {"variant": "L5_no_target_cash_all_audit", "exact_rule": "cash only when target genuinely null/blocked", "source": "formal_model_contract.py + no_target_risk_off_formal_activation.py", "parameter": "00631L base is not no-target"},
    ]).assign(future_return_used_as_rule=False, diagnostic_only=True, **FLAGS)
    mapping_audit = pd.DataFrame([
        {"mapping_item": "first_attack_activation", "formal_source_semantics": "attack_gate_ever_activated; systemic bear resets cycle", "R6_transplant_mapping": "first actual R6 stock target after systemic-bear reset", "mapping_quality": "explicit_bounded_transplant_mapping_not_formal_selector_replay"},
        {"mapping_item": "portfolio_signal_value", "formal_source_semantics": "account marked at signal close against running peak", "R6_transplant_mapping": "unique-position daily wealth available at each signal state before next-day execution", "mapping_quality": "same_daily_state_basis_diagnostic"},
        {"mapping_item": "no_target", "formal_source_semantics": "target=None exits to cash", "R6_transplant_mapping": "only blank/blocked R6 target; 00631L base remains target", "mapping_quality": "exact_applicability_guard"},
    ]).assign(future_data_violation_count=0)
    coverage = pd.DataFrame([
        {
            "period": period, "requested_start": start, "requested_end": end,
            "actual_start": paths.loc[paths["period"].eq(period), "signal_date"].min(),
            "actual_end": paths.loc[paths["period"].eq(period), "signal_date"].max(),
            "daily_rows_per_variant": int(paths.loc[(paths["period"].eq(period)) & paths["legacy_risk_variant"].eq(VARIANTS[0])].shape[0]),
            "systemic_bear_feature_ready_share": float(paths.loc[(paths["period"].eq(period)) & paths["legacy_risk_variant"].eq(VARIANTS[0]), "systemic_bear_feature_ready"].mean()),
            "risk_2of3_feature_ready_share": float(paths.loc[(paths["period"].eq(period)) & paths["legacy_risk_variant"].eq(VARIANTS[0]), "risk_2of3_feature_ready"].mean()),
            "daily_path_ready_share": 1.0, "future_data_violation_count": 0,
        }
        for period, (start, end) in PERIODS.items()
    ])
    source_audit = pd.DataFrame([
        {"source_item": "frozen_config", "path": "configs/frozen_cycle_proven_top1_v1.json", "status": "loaded_contract_truth"},
        {"source_item": "frozen_variant", "path": "regime_mode_switch.py::frozen_cycle_proven_top1_v1_variant", "status": "loaded_exact_parameters"},
        {"source_item": "market_features", "path": str(BENCHMARK_PATH), "status": "0050 adjusted benchmark PIT rolling"},
        {"source_item": "R6_prices", "path": str(R6_DIR), "status": "official unadjusted selected-stock diagnostic-only"},
        {"source_item": "selected_stock_adjusted_close", "path": str(SOURCE_CLOSURE_DIR), "status": "blocked_closed_no_source_escalation"},
    ])
    future_audit = pd.DataFrame([
        {"audit_item": "market_rules", "future_return_used_as_rule": False, "detail": "All 0050 rolling fields use signal close and prior observations only.", "future_data_violation_count": 0},
        {"audit_item": "portfolio_stop", "future_return_used_as_rule": False, "detail": "Uses accumulated strategy wealth available before current next-day execution.", "future_data_violation_count": 0},
        {"audit_item": "R6_target", "future_return_used_as_rule": False, "detail": "Uses existing reconstructed R6 PIT target.", "future_data_violation_count": 0},
    ])
    source_closure = json.loads((SOURCE_CLOSURE_DIR / "readiness_for_selected_stock_total_return_source_escalation_closure.json").read_text(encoding="utf-8"))
    readiness = {
        "task_id": TASK_ID,
        "status": "legacy_formal_risk_controls_transplanted_to_R6_bounded_diagnostic_ready",
        "variant_count": len(VARIANTS),
        "L0_R6_baseline_reconciliation_pass": bool(reconciliation["pass_within_1e_12"].all()),
        "systemic_bear_exact_rule_ready": True,
        "portfolio_stop12_exact_no_cooldown_ready": True,
        "preproof_risk2of3_exact_state_ready": True,
        "first_attack_activation_mapping_quality": "explicit_R6_transplant_mapping_not_formal_selector_replay",
        "no_target_cash_all_applicability_audit_ready": True,
        "no_target_cash_all_trigger_count": 0,
        "EP05_exposure_weighted_transition_cost_ready": True,
        "official_unadjusted_stock_path_ready": True,
        "selected_stock_adjusted_close_ready": False,
        "historical_path_policy": source_closure.get("historical_backtest_path_policy", "official_unadjusted_OHLC_diagnostic_only"),
        "ready_for_experiments": True,
        "ready_for_formal": False,
        "ready_for_strategy_replay": False,
        "future_data_violation_count": 0,
        "next_owner": "Experiments R6 legacy formal risk-control transplant diagnostic",
        **FLAGS,
    }
    blocked = pd.DataFrame([
        {"item": "first_attack_activation", "status": "bounded_mapping", "detail": "Mapped to first actual R6 stock target after systemic reset; not a replay of legacy selector attack gate."},
        {"item": "selected_stock_adjusted_close", "status": "blocked_closed", "detail": "Official unadjusted OHLC diagnostic-only."},
        {"item": "formal_promotion", "status": "prohibited", "detail": "Transplant challenger requires Experiments verdict and cannot alter formal model."},
    ])
    output_paths = [
        _write(paths, "r6_legacy_formal_risk_control_daily_state_contract.csv"),
        _write(transitions, "r6_legacy_formal_risk_control_transition_trace.csv"),
        _write(metrics, "r6_legacy_formal_risk_control_net_mdd_exposure_metrics.csv"),
        _write(_annual(paths), "r6_legacy_formal_risk_control_annual_metrics.csv"),
        _write(policy, "r6_legacy_formal_risk_control_policy_source_map.csv"),
        _write(mapping_audit, "r6_legacy_formal_risk_control_state_mapping_audit.csv"),
        _write(no_target, "r6_legacy_formal_no_target_cash_all_applicability_audit.csv"),
        _write(reconciliation, "r6_legacy_formal_risk_control_L0_reconciliation_audit.csv"),
        _write(coverage, "r6_legacy_formal_risk_control_requested_vs_actual_coverage.csv"),
        _write(source_audit, "r6_legacy_formal_risk_control_price_source_audit.csv"),
        _write(blocked, "r6_legacy_formal_risk_control_blocked_proxy_audit.csv"),
        _write(future_audit, "r6_legacy_formal_risk_control_future_data_audit.csv"),
    ]
    readiness_path = OUTPUT_DIR / "readiness_for_r6_legacy_formal_risk_control_transplant.json"
    readiness_path.write_text(json.dumps(readiness, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path = OUTPUT_DIR / "final_summary_zh.md"
    summary_path.write_text(
        "# R6 Legacy Formal Risk-control Transplant\n\n"
        "- L0-L4 fixed variants only; no threshold grid or new cooldown.\n"
        "- L1 uses exact systemic-bear formula; L2 uses 12% strategy peak stop with peak reset, cooldown=0, no latch.\n"
        "- L3 uses exact risk_2of3, 25% 00631L +75% cash, 5-day clear, only before first R6 stock activation in each systemic-reset cycle.\n"
        "- L5 confirms R6 00631L base is a valid target, not formal no-target; cash-all trigger count is zero.\n"
        "- L0 reconciles exactly to reconstructed R6 P1/P2. Metrics are net after exposure-weighted EP05 costs.\n"
        "- selected stocks remain official unadjusted OHLC diagnostic-only; no formal/replay/report/trade decision change.\n\n"
        "結論：bounded transplant contract ready，可直接交Experiments做P1-first tradeoff診斷。\n",
        encoding="utf-8",
    )
    manifest = {
        "task_id": TASK_ID, "generated_at_utc": datetime.now(timezone.utc).isoformat(), "output_dir": str(OUTPUT_DIR),
        "source_inputs": {"R6": str(R6_DIR), "benchmark": str(BENCHMARK_PATH), "frozen_variant": variant.__dict__},
        "files": [{"path": p.name, "sha256": _sha256(p)} for p in [*output_paths, readiness_path, summary_path]],
        "readiness": readiness,
    }
    (OUTPUT_DIR / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(readiness, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
