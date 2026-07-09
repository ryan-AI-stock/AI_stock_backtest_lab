from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from backtest_lab.vnext_p1_c2_consensus_trigger_weighted_pool80_top5_contract import (
    FLAGS,
    LAYER4_POOL,
    PREV_COST_DESIGN,
    _load_triggers,
    _path_map,
    _route_support,
    _score_components,
    _ticker,
)
from backtest_lab.vnext_p1_c2_route_support_max1_modelization import (
    STATE_HOLD_DIR,
    TRANSITION_COSTS,
    _benchmark_maps,
    _calendar,
    _transition_action,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
WEIGHTED_ABSORPTION_DIR = REPO_ROOT / "outputs" / "vnext_p1_c2_weighted_pool80_top5_ohlc_absorption_20260708"
LOW_BASE_DIR = REPO_ROOT / "outputs" / "vnext_layer4_low_base_score_contract_20260709"
OUTPUT_DIR = REPO_ROOT / "outputs" / "vnext_low_base_integrated_c2_route_support_contract_20260709"

TASK_ID = "TASK-BACKTEST-CORE-VNEXT-LOW-BASE-INTEGRATED-C2-ROUTE-SUPPORT-CONTRACT-001"
DIAGNOSTIC_NOTIONAL = 1_000_000

VARIANT_WEIGHTS = {
    "baseline_route_support": {
        "route_support_base_score": 1.00,
    },
    "low_base_balanced": {
        "route_support_base_score": 0.72,
        "low_base_score": 0.20,
        "pool_persistence_component_proxy": 0.08,
    },
    "low_base_risk_aware": {
        "route_support_base_score": 0.60,
        "low_base_score": 0.22,
        "risk_inverse_component": 0.12,
        "overheat_inverse": 0.06,
    },
    "low_base_quality": {
        "route_support_base_score": 0.58,
        "low_base_score": 0.20,
        "quality_component": 0.16,
        "risk_inverse_component": 0.06,
    },
    "low_base_pullback_reacceleration": {
        "route_support_base_score": 0.55,
        "low_base_score": 0.17,
        "pullback_repair_score_norm": 0.14,
        "overlap_reacceleration_score_norm": 0.09,
        "risk_inverse_component": 0.05,
    },
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _num(df: pd.DataFrame, col: str, default: float = 0.5) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").fillna(default)


def _boolish(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(False, index=df.index)
    return df[col].astype(str).str.lower().isin(["true", "1", "yes"])


def _clip01(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").clip(0.0, 1.0)


def _load_path_candidates() -> pd.DataFrame:
    absorbed = WEIGHTED_ABSORPTION_DIR / "p1_c2_weighted_pool80_top5_contract_refreshed.csv"
    if absorbed.exists():
        df = pd.read_csv(absorbed, low_memory=False, dtype={"ticker": str})
        df["signal_date"] = pd.to_datetime(df["signal_date"], errors="coerce").dt.strftime("%Y-%m-%d")
        df["ticker"] = df["ticker"].map(_ticker)
        return df.sort_values(["signal_date", "ticker", "score_variant", "candidate_rank"]).drop_duplicates(
            ["signal_date", "ticker"]
        )
    return _path_map()


def _build_low_base_components(components: pd.DataFrame) -> pd.DataFrame:
    df = components.copy()
    draw60 = _num(df, "drawdown_60d", 0.0)
    draw120 = _num(df, "drawdown_120d", 0.0)
    df["price_position_low_base"] = (((-draw120).clip(0.03, 0.45) / 0.45) * 0.6 + ((-draw60).clip(0.02, 0.35) / 0.35) * 0.4).clip(0, 1)

    bias60_pct = _num(df, "BIAS60_percentile", 0.5)
    bias20_pct = _num(df, "BIAS20_percentile", 0.5)
    bias120_pct = _num(df, "BIAS120_percentile", 0.5)
    df["stock_specific_bias_score"] = (1.0 - (bias60_pct - 0.42).abs() * 1.8).clip(0, 1)
    df["recent_runup_penalty"] = (bias20_pct * 0.45 + bias60_pct * 0.4 + bias120_pct * 0.15).clip(0, 1)
    df["recent_runup_inverse"] = 1.0 - df["recent_runup_penalty"]

    rs20 = _num(df, "RS20", 0.0)
    rs60 = _num(df, "RS60", 0.0)
    rs20_rank = rs20.groupby(df["snapshot_date"]).rank(pct=True, ascending=True).fillna(0.5)
    rs60_rank = rs60.groupby(df["snapshot_date"]).rank(pct=True, ascending=True).fillna(0.5)
    rs_overheat = (rs60_rank - 0.82).clip(0, 1) * 0.35
    df["improving_rs_score"] = (rs20_rank * 0.65 + rs60_rank * 0.35 - rs_overheat).clip(0, 1)

    rank20 = _num(df, "traded_value_rank_20d", 80.0)
    rank60 = _num(df, "traded_value_rank_60d", 80.0)
    rank_improve = ((rank60 - rank20) / rank60.clip(lower=1)).clip(-1, 1)
    df["liquidity_improvement"] = (((rank_improve + 1.0) / 2.0) * 0.45 + (1.0 - ((rank20 - 1.0) / 80.0).clip(0, 1)) * 0.55).clip(0, 1)

    df["quality_support"] = _num(df, "quality_component", 0.5)
    overheat_flags = (
        _boolish(df, "bias_overheat_penalty_context")
        | _boolish(df, "volatility_high_context")
        | _boolish(df, "rs60_high_short_rs_weakening_exhaustion_context")
        | _boolish(df, "high_exhaustion_or_breakdown_context")
    )
    high_bias = bias60_pct.gt(0.9) | bias20_pct.gt(0.92)
    high_vol = _num(df, "volatility_pctile_by_week", 0.5).gt(0.9)
    df["overheat_veto_flag"] = overheat_flags | high_bias | high_vol
    df["overheat_inverse"] = (1.0 - overheat_flags.astype(float) * 0.35 - high_bias.astype(float) * 0.35 - high_vol.astype(float) * 0.30).clip(0, 1)

    low_parts = [
        df["price_position_low_base"] * 0.20,
        df["stock_specific_bias_score"] * 0.16,
        df["recent_runup_inverse"] * 0.14,
        df["improving_rs_score"] * 0.18,
        df["liquidity_improvement"] * 0.14,
        df["quality_support"] * 0.13,
        df["overheat_inverse"] * 0.05,
    ]
    df["low_base_score"] = sum(low_parts).clip(0, 1)
    df["route_support_base_score"] = (
        df["quality_component"] * 0.10
        + df["rs_component"] * 0.20
        + df["liquidity_component"] * 0.10
        + df["bias_health_component"] * 0.10
        + df["route_support_component"] * 0.38
        + df["risk_inverse_component"] * 0.12
    ).clip(0, 1)
    df["pullback_repair_score_norm"] = _num(df, "pullback_repair_score", 0.0).clip(0, 1)
    df["overlap_reacceleration_score_norm"] = _num(df, "overlap_reacceleration_score", 0.0).clip(0, 1)
    df["low_base_component_source_quality"] = "PIT Layer4 existing fields; no new hard filter; drawdown/BIAS/liquidity/quality are diagnostic components"
    return df


def _rank_variants(components: pd.DataFrame, path: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    selections: list[pd.DataFrame] = []
    all_ranked: list[pd.DataFrame] = []
    for variant, weights in VARIANT_WEIGHTS.items():
        df = components.copy()
        df["score_variant"] = variant
        df["integrated_score"] = sum(_num(df, col, 0.5) * weight for col, weight in weights.items()).clip(0, 1)
        df = df.sort_values(["snapshot_date", "integrated_score", "route_support_base_score", "ticker"], ascending=[True, False, False, True])
        df["integrated_rank"] = df.groupby("snapshot_date").cumcount() + 1
        all_ranked.append(df)
        selections.append(df[df["integrated_rank"].eq(1)].copy())
    selected = pd.concat(selections, ignore_index=True)
    selected = selected.merge(path, left_on=["snapshot_date", "ticker"], right_on=["signal_date", "ticker"], how="left", suffixes=("", "_path"))
    selected["signal_date"] = selected["snapshot_date"]
    selected["candidate_rank"] = 1
    selected["official_unadjusted_ohlc_path_ready"] = selected[["entry_close", "exit_close"]].notna().all(axis=1)
    selected["adjusted_close_ready"] = False
    selected["adjustment_policy"] = "official_unadjusted_ohlc_diagnostic_only_adjusted_close_blocked"
    selected["future_data_violation_count"] = 0
    selected["diagnostic_only"] = True
    for key, value in FLAGS.items():
        selected[key] = value
    return selected, pd.concat(all_ranked, ignore_index=True)


def _selection_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "signal_date",
        "score_variant",
        "candidate_rank",
        "integrated_rank",
        "ticker",
        "name",
        "market",
        "integrated_score",
        "route_support_base_score",
        "low_base_score",
        "price_position_low_base",
        "stock_specific_bias_score",
        "recent_runup_penalty",
        "recent_runup_inverse",
        "improving_rs_score",
        "liquidity_improvement",
        "quality_support",
        "overheat_veto_flag",
        "overheat_inverse",
        "quality_component",
        "rs_component",
        "liquidity_component",
        "bias_health_component",
        "route_support_component",
        "risk_inverse_component",
        "pool_persistence_component_proxy",
        "c2_market_health_gate",
        "consensus_trigger",
        "prior_c2_allowed_exception_ticker",
        "prior_single_exception_ticker",
        "route_support_variant_count",
        "route_support_mode_count",
        "route_support_variant_flags",
        "route_support_mode_flags",
        "layer1_quality_floor_risk_pctile_by_week",
        "layer1_pass_bottom30",
        "RS20",
        "RS40",
        "RS60",
        "RS30_proxy",
        "traded_value_rank_5d",
        "traded_value_rank_20d",
        "traded_value_rank_60d",
        "BIAS20",
        "BIAS60",
        "BIAS120",
        "BIAS20_percentile",
        "BIAS60_percentile",
        "BIAS120_percentile",
        "volatility_pctile_by_week",
        "exhaustion_risk_score",
        "breakdown_risk_score",
        "entry_date",
        "exit_date",
        "entry_open",
        "entry_close",
        "exit_close",
        "gross_return_unadjusted",
        "net_return_local_ep05_cost_unit_notional",
        "official_unadjusted_ohlc_path_ready",
        "adjusted_close_ready",
        "source_quality",
        "entry_source_route",
        "exit_source_route",
        "total_cost_twd",
        "cost_application_status",
        "adjustment_policy",
        "future_data_violation_count",
        "diagnostic_only",
        *FLAGS.keys(),
    ]
    return df[[c for c in cols if c in df.columns]]


def _build_state_contract(selections: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cal = _calendar()
    maps = _benchmark_maps()
    rows: list[dict[str, Any]] = []
    transitions: list[dict[str, Any]] = []
    for variant in VARIANT_WEIGHTS:
        selected = selections[selections["score_variant"].eq(variant)].copy()
        by_date = {row.signal_date: row._asdict() for row in selected.itertuples(index=False)}
        prev_ticker = "00631L"
        prev_type = "etf"
        for r in cal.itertuples(index=False):
            signal_date = r.signal_date
            next_signal = r.next_signal_date
            stock = by_date.get(signal_date)
            if stock is not None:
                target_ticker = _ticker(stock.get("ticker"))
                target_type = "stock"
                state_reason = f"c2_consensus_trigger_{variant}_stock_exception"
                entry_price = stock.get("entry_close")
                exit_price = stock.get("exit_close")
                interval_return = stock.get("gross_return_unadjusted")
                path_ready = bool(stock.get("official_unadjusted_ohlc_path_ready", False))
                source_quality = stock.get("source_quality", "official_unadjusted_ohlc")
                entry_date = stock.get("entry_date", "")
                exit_date = stock.get("exit_date", "")
                score = stock.get("integrated_score")
                low_base_score = stock.get("low_base_score")
            else:
                target_ticker = "00631L"
                target_type = "etf"
                state_reason = "default_00631L_base_no_c2_consensus_trigger"
                entry_price = maps["00631L"].get(signal_date)
                exit_price = maps["00631L"].get(next_signal)
                interval_return = (exit_price / entry_price - 1.0) if entry_price and exit_price else None
                path_ready = interval_return is not None
                source_quality = "benchmark_features_adjusted_close_exact_reference"
                entry_date = signal_date
                exit_date = next_signal
                score = None
                low_base_score = None

            action, cost_key = _transition_action(prev_ticker, prev_type, target_ticker, target_type)
            cost = TRANSITION_COSTS[cost_key]
            net_return = (float(interval_return) - cost["transition_cost_rate"]) if interval_return is not None else None
            rows.append(
                {
                    "signal_date": signal_date,
                    "next_signal_date": next_signal,
                    "score_variant": variant,
                    "selected_ticker": target_ticker,
                    "selected_asset_type": target_type,
                    "state_reason": state_reason,
                    "integrated_score": score,
                    "low_base_score": low_base_score,
                    "entry_date": entry_date,
                    "exit_date": exit_date,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "gross_interval_return": interval_return,
                    "transition_action": action,
                    "transition_cost_rate": cost["transition_cost_rate"],
                    "net_interval_return_after_transition_cost": net_return,
                    "official_unadjusted_ohlc_ready": path_ready if target_type == "stock" else True,
                    "benchmark_adjusted_path_ready": path_ready if target_type == "etf" else True,
                    "adjusted_close_ready": target_type == "etf",
                    "source_quality": source_quality,
                    "cash_condition_status": "blocked_no_bear_cash_classifier",
                    "diagnostic_only": True,
                    **FLAGS,
                }
            )
            if action != "hold_same_state_no_trade":
                transitions.append(
                    {
                        "signal_date": signal_date,
                        "transition_date": entry_date,
                        "score_variant": variant,
                        "from_ticker": prev_ticker,
                        "from_asset_type": prev_type,
                        "to_ticker": target_ticker,
                        "to_asset_type": target_type,
                        "transition_action": action,
                        "diagnostic_notional_twd": DIAGNOSTIC_NOTIONAL,
                        **cost,
                        "cost_model_status": "applied_local_ep05_TaiwanCostModel_unit_notional_transition_cost",
                        "cost_model_version": "taiwan_standard_fee_tax_v1",
                        "diagnostic_only": True,
                        **FLAGS,
                    }
                )
            prev_ticker, prev_type = target_ticker, target_type
    return pd.DataFrame(rows), pd.DataFrame(transitions)


def _variant_definitions() -> pd.DataFrame:
    rows = []
    descriptions = {
        "baseline_route_support": "current C2 + consensus trigger + route_support max1 baseline",
        "low_base_balanced": "route_support with low_base bonus and pool persistence; no hard filter",
        "low_base_risk_aware": "route_support with low_base bonus, BIAS/volatility overheat penalty, risk inverse support",
        "low_base_quality": "route_support with low_base bonus plus Layer1 quality/risk support",
        "low_base_pullback_reacceleration": "optional overlap with Layer3 pullback/reacceleration scores as soft components only",
    }
    for variant, weights in VARIANT_WEIGHTS.items():
        row = {
            "score_variant": variant,
            "description": descriptions[variant],
            "formula": "sum(component * weight)",
            "hard_filter_added": False,
            "future_return_used": False,
            "component_policy": "low_base is ranking bonus/penalty/tie-break component only; C2+consensus trigger still controls stock-exception eligibility",
        }
        row.update(weights)
        rows.append(row)
    return pd.DataFrame(rows)


def _coverage(selection: pd.DataFrame, state_contract: pd.DataFrame) -> pd.DataFrame:
    stock_state = state_contract[state_contract["selected_asset_type"].eq("stock")].copy()
    rows = []
    for variant in VARIANT_WEIGHTS:
        sel = selection[selection["score_variant"].eq(variant)]
        st = stock_state[stock_state["score_variant"].eq(variant)]
        rows.append(
            {
                "score_variant": variant,
                "requested_period": "P1 2015-01-02 to 2022-12-29",
                "actual_signal_start": state_contract[state_contract["score_variant"].eq(variant)]["signal_date"].min(),
                "actual_signal_end": state_contract[state_contract["score_variant"].eq(variant)]["signal_date"].max(),
                "eligible_stock_signal_dates": int(sel["signal_date"].nunique()),
                "selected_unique_tickers": int(sel["ticker"].nunique()),
                "selected_stock_rows": int(len(sel)),
                "official_unadjusted_ohlc_ready_rows": int(sel["official_unadjusted_ohlc_path_ready"].fillna(False).astype(bool).sum()),
                "official_unadjusted_ohlc_ready_share": float(sel["official_unadjusted_ohlc_path_ready"].fillna(False).astype(bool).mean()) if len(sel) else 0.0,
                "state_contract_rows": int(len(state_contract[state_contract["score_variant"].eq(variant)])),
                "state_stock_rows": int(len(st)),
            }
        )
    return pd.DataFrame(rows)


def _gap_ledger(selection: pd.DataFrame) -> pd.DataFrame:
    rows = []
    missing = selection[~selection["official_unadjusted_ohlc_path_ready"].fillna(False).astype(bool)]
    timing_by_date = (
        selection[selection["official_unadjusted_ohlc_path_ready"].fillna(False).astype(bool)]
        .dropna(subset=["entry_date", "exit_date"])
        .drop_duplicates("signal_date")
        .set_index("signal_date")[["entry_date", "exit_date"]]
        .to_dict(orient="index")
    )
    for r in missing.itertuples(index=False):
        timing = timing_by_date.get(r.signal_date, {})
        rows.append(
            {
                "signal_date": r.signal_date,
                "entry_date": timing.get("entry_date", getattr(r, "entry_date", "")),
                "exit_date": timing.get("exit_date", getattr(r, "exit_date", "")),
                "score_variant": r.score_variant,
                "ticker": r.ticker,
                "name": getattr(r, "name", ""),
                "market": getattr(r, "market", ""),
                "timing_variant": "next_day_close_entry_fixed_5td_exit",
                "required_price_fields": "entry_open,entry_close,exit_close",
                "blocked_item": "selected_stock_official_unadjusted_ohlc_path",
                "blocked_reason": "new low_base integrated top1 not covered by prior selected-ticker OHLC path package",
                "next_owner": "Radar/Data bounded selected-ticker-only OHLC gap fill if Strategy Center wants this variant tested",
            }
        )
    return pd.DataFrame(rows)


def _blocked(selection: pd.DataFrame, gap_ledger: pd.DataFrame) -> pd.DataFrame:
    rows = gap_ledger.to_dict(orient="records")
    rows.extend(
        [
            {
                "signal_date": "",
                "score_variant": "all",
                "ticker": "",
                "name": "",
                "blocked_item": "selected_stock_adjusted_close",
                "blocked_reason": "historical adjusted-close route remains blocked; official unadjusted OHLC is diagnostic-only",
                "next_owner": "Strategy Center policy or Radar/Data adjusted-close source route",
            },
            {
                "signal_date": "",
                "score_variant": "all",
                "ticker": "",
                "name": "",
                "blocked_item": "cash_bear_classifier",
                "blocked_reason": "no accepted cash/bear classifier; default base remains 00631L",
                "next_owner": "Strategy Center/Core if cash branch is authorized later",
            },
            {
                "signal_date": "",
                "score_variant": "all",
                "ticker": "",
                "name": "",
                "blocked_item": "low_base_hard_filter",
                "blocked_reason": "explicitly not allowed; low_base is component only",
                "next_owner": "none",
            },
        ]
    )
    return pd.DataFrame(rows)


def _score_components_table(all_ranked: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "snapshot_date",
        "score_variant",
        "integrated_rank",
        "ticker",
        "name",
        "market",
        "integrated_score",
        "route_support_base_score",
        "low_base_score",
        "price_position_low_base",
        "stock_specific_bias_score",
        "recent_runup_penalty",
        "recent_runup_inverse",
        "improving_rs_score",
        "liquidity_improvement",
        "quality_support",
        "overheat_veto_flag",
        "overheat_inverse",
        "quality_component",
        "rs_component",
        "liquidity_component",
        "bias_health_component",
        "route_support_component",
        "risk_inverse_component",
        "pool_persistence_component_proxy",
        "route_support_variant_count",
        "route_support_variant_flags",
        "route_support_mode_flags",
        "layer1_quality_floor_risk_pctile_by_week",
        "layer1_pass_bottom30",
        "RS20",
        "RS40",
        "RS60",
        "RS30_proxy",
        "traded_value_rank_5d",
        "traded_value_rank_20d",
        "traded_value_rank_60d",
        "BIAS20",
        "BIAS60",
        "BIAS120",
        "BIAS20_percentile",
        "BIAS60_percentile",
        "BIAS120_percentile",
        "volatility_pctile_by_week",
        "exhaustion_risk_score",
        "breakdown_risk_score",
        "low_base_component_source_quality",
    ]
    return all_ranked[[c for c in cols if c in all_ranked.columns]].rename(columns={"snapshot_date": "signal_date"})


def _readiness(selection: pd.DataFrame, coverage: pd.DataFrame) -> dict[str, Any]:
    ready_share = float(selection["official_unadjusted_ohlc_path_ready"].fillna(False).astype(bool).mean()) if len(selection) else 0.0
    ready = ready_share == 1.0
    return {
        "task_id": TASK_ID,
        "status": "low_base_integrated_c2_route_support_contract_ready_unadjusted_diagnostic_adjusted_blocked" if ready else "low_base_integrated_c2_route_support_contract_partial_ohlc_blocked",
        "period_requested": "P1 2015-01-02 to 2022-12-29",
        "variant_count": int(len(VARIANT_WEIGHTS)),
        "eligible_signal_dates": int(selection["signal_date"].nunique()),
        "selected_stock_rows": int(len(selection)),
        "official_unadjusted_ohlc_ready_share": ready_share,
        "official_unadjusted_ohlc_ready_rows": int(selection["official_unadjusted_ohlc_path_ready"].fillna(False).astype(bool).sum()) if len(selection) else 0,
        "official_unadjusted_ohlc_blocked_rows": int((~selection["official_unadjusted_ohlc_path_ready"].fillna(False).astype(bool)).sum()) if len(selection) else 0,
        "transition_cost_fields_ready": True,
        "cost_model_ready": True,
        "cost_model_version": "taiwan_standard_fee_tax_v1",
        "adjusted_close_ready": False,
        "low_base_hard_filter_added": False,
        "ready_for_low_base_integrated_experiments": bool(ready),
        "ready_for_experiments": bool(ready),
        "ready_for_radar_selected_ohlc_gap_fill": not ready,
        "future_data_violation_count": 0,
        "ready_for_formal": False,
        "ready_for_strategy_replay": False,
        **FLAGS,
        "coverage_by_variant": coverage.to_dict(orient="records"),
    }


def _write_summary(path: Path, readiness: dict[str, Any], blocked: pd.DataFrame) -> None:
    next_step = (
        "下一棒：交 Experiments 執行 TASK-BACKTEST-EXPERIMENTS-VNEXT-P1-LOW-BASE-INTEGRATED-C2-ROUTE-SUPPORT-DIAGNOSTIC-001。"
        if readiness["ready_for_experiments"]
        else "下一棒：交 Radar/Data 做 bounded selected-ticker-only OHLC gap fill，補新 low_base integrated top1 缺價。"
    )
    path.write_text(
        "\n".join(
            [
                "# low_base integrated C2 / route_support contract",
                "",
                "## 結論",
                "",
                "- 已建立 P1 low_base_score 整合版 C2 + consensus trigger + route_support / Layer4 ranking contract。",
                "- low_base_score 只作 ranking bonus / penalty / tie-break component，沒有新增 hard filter。",
                "- baseline 保留現行 C2 + route_support max1；另輸出 low_base_balanced、low_base_risk_aware、low_base_quality、low_base_pullback_reacceleration。",
                f"- eligible signal dates = {readiness['eligible_signal_dates']}；selected stock rows = {readiness['selected_stock_rows']}。",
                f"- official unadjusted OHLC ready share = {readiness['official_unadjusted_ohlc_ready_share']:.4f}。",
                "- adjusted_close_ready=false；selected-stock unadjusted OHLC 仍是 diagnostic-only。",
                "- 後續主結論必須 net after transaction cost；gross/no-cost 只能 secondary。",
                "",
                "## Blocked / Proxy",
                "",
                blocked.to_csv(index=False).strip(),
                "",
                next_step,
                "",
                "完成後如果下一棒明確，請直接指派下一個 thread；如果下一棒不明確，請回報 Strategy Center 判斷。不要完成後停住不回報。",
                "",
                "Flags: formal_model_changed=false; trade_decision_changed=false; active_in_trade_decision=false; report_changed=false; portfolio_replay_executed=false; ready_for_strategy_replay=false; ready_for_formal=false; not_live_rule=true; forward_returns_live_rule_usage=false.",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pool = pd.read_csv(LAYER4_POOL, low_memory=False, dtype={"ticker": str})
    triggers = _load_triggers()
    support = _route_support()
    path = _load_path_candidates()
    components = _score_components(pool, support, triggers)
    components = _build_low_base_components(components)
    selection_full, all_ranked = _rank_variants(components, path)
    selection = _selection_columns(selection_full)
    state_contract, transition_trace = _build_state_contract(selection)
    coverage = _coverage(selection, state_contract)
    gap_ledger = _gap_ledger(selection)
    blocked = _blocked(selection, gap_ledger)
    variants = _variant_definitions()
    score_components = _score_components_table(all_ranked)
    future = pd.DataFrame(
        [
            {
                "audit_item": "low_base_integrated_score",
                "future_return_used_as_rule": False,
                "future_winner_used_as_rule": False,
                "future_data_violation_count": 0,
            },
            {
                "audit_item": "selected_ohlc_path",
                "future_return_used_as_rule": False,
                "source_policy": "entry/exit OHLC for diagnostic evaluation only, not score construction",
                "future_data_violation_count": 0,
            },
        ]
    )

    paths = {
        "contract": OUTPUT_DIR / "low_base_integrated_c2_route_support_contract.csv",
        "selection": OUTPUT_DIR / "low_base_integrated_variant_selection_map.csv",
        "components": OUTPUT_DIR / "low_base_integrated_score_components.csv",
        "ohlc": OUTPUT_DIR / "low_base_integrated_ohlc_readiness.csv",
        "coverage": OUTPUT_DIR / "requested_vs_actual_coverage.csv",
        "blocked": OUTPUT_DIR / "blocked_proxy_audit.csv",
        "gap_ledger": OUTPUT_DIR / "low_base_integrated_selected_ticker_ohlc_gap_ledger.csv",
        "future": OUTPUT_DIR / "future_data_audit.csv",
        "variants": OUTPUT_DIR / "low_base_integrated_variant_definitions.csv",
        "transition": OUTPUT_DIR / "low_base_integrated_transition_trace.csv",
        "cost": OUTPUT_DIR / "low_base_integrated_transition_cost_audit.csv",
        "readiness": OUTPUT_DIR / "readiness_for_low_base_integrated_experiments.json",
        "summary": OUTPUT_DIR / "final_summary_zh.md",
        "manifest": OUTPUT_DIR / "manifest.json",
    }
    state_contract.to_csv(paths["contract"], index=False, encoding="utf-8-sig")
    selection.to_csv(paths["selection"], index=False, encoding="utf-8-sig")
    score_components.to_csv(paths["components"], index=False, encoding="utf-8-sig")
    coverage.to_csv(paths["ohlc"], index=False, encoding="utf-8-sig")
    coverage.to_csv(paths["coverage"], index=False, encoding="utf-8-sig")
    blocked.to_csv(paths["blocked"], index=False, encoding="utf-8-sig")
    gap_ledger.to_csv(paths["gap_ledger"], index=False, encoding="utf-8-sig")
    future.to_csv(paths["future"], index=False, encoding="utf-8-sig")
    variants.to_csv(paths["variants"], index=False, encoding="utf-8-sig")
    transition_trace.to_csv(paths["transition"], index=False, encoding="utf-8-sig")
    if transition_trace.empty:
        pd.DataFrame(columns=["transition_action", "transition_count", "transition_cost_rate"]).to_csv(paths["cost"], index=False, encoding="utf-8-sig")
    else:
        transition_trace.groupby(["score_variant", "transition_action", "from_asset_type", "to_asset_type"], as_index=False).agg(
            transition_count=("transition_action", "size"),
            transition_cost_rate=("transition_cost_rate", "first"),
            total_transition_cost_twd_sum=("total_transition_cost_twd", "sum"),
            cost_model_status=("cost_model_status", "first"),
            cost_model_version=("cost_model_version", "first"),
        ).to_csv(paths["cost"], index=False, encoding="utf-8-sig")
    readiness = _readiness(selection, coverage)
    paths["readiness"].write_text(json.dumps(readiness, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_summary(paths["summary"], readiness, blocked)
    manifest = {
        "task_id": TASK_ID,
        "output_dir": str(OUTPUT_DIR),
        "inputs": {
            "layer4_pool": str(LAYER4_POOL),
            "weighted_pool80_absorbed_contract": str(WEIGHTED_ABSORPTION_DIR / "p1_c2_weighted_pool80_top5_contract_refreshed.csv"),
            "low_base_overlap_audit": str(LOW_BASE_DIR / "existing_low_base_overlap_audit.csv"),
            "state_hold_benchmark_dir": str(STATE_HOLD_DIR),
            "transition_cost_design": str(PREV_COST_DESIGN),
        },
        "artifacts": [
            {"path": str(p), "sha256": _sha256(p), "bytes": p.stat().st_size}
            for key, p in paths.items()
            if key != "manifest"
        ],
        "readiness": readiness,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "formal_model_changed": False,
        "trade_decision_changed": False,
        "active_in_trade_decision": False,
        "report_changed": False,
    }
    paths["manifest"].write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(readiness, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
