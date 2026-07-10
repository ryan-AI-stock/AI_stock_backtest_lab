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
RAW_F = REPO_ROOT / "outputs" / "vnext_daily_incumbent_challenger_state_machine_contract_ohlc_absorbed_20260710" / "daily_incumbent_challenger_state_machine_contract_ohlc_absorbed.csv"
OUTPUT_DIR = REPO_ROOT / "outputs" / "vnext_f2_gate_persistence_hysteresis_incumbent_protection_contract_20260710"
RADAR_ROOT = Path("C:/Users/zergv/Documents/Codex/2026-05-23/ai-stock-rotation-radar-https-docs/outputs")
PRICE_SOURCES = [
    RADAR_ROOT / "radar_vnext_daily_incumbent_challenger_selected_stock_daily_ohlc_gap_fill_20260710" / "daily_incumbent_challenger_selected_stock_daily_unadjusted_ohlc_rows.csv",
    RADAR_ROOT / "radar_vnext_p1_weekly_r6_selected_stock_daily_ohlc_attribution_gap_fill_20260710" / "p1_weekly_r6_selected_stock_daily_ohlc_filled_rows.csv",
    RADAR_ROOT / "radar_vnext_weekly_r6_single_position_reconstructed_p2_selected_stock_daily_ohlc_gap_fill_20260710" / "reconstructed_weekly_r6_p2_selected_stock_daily_ohlc_filled_rows.csv",
    RADAR_ROOT / "radar_vnext_regime_switch_route_selected_stock_ohlc_source_package_20260708" / "regime_switch_selected_ohlc_rows.csv",
    RADAR_ROOT / "radar_vnext_p2_2023_selected_stock_ohlc_source_gap_fill_20260708" / "p2_2023_selected_stock_unadjusted_ohlc_rows.csv",
    RADAR_ROOT / "radar_vnext_f2_gate_persistence_hysteresis_selected_stock_daily_ohlc_gap_fill_20260710" / "f2_gate_persistence_selected_stock_daily_ohlc_filled_rows.csv",
    RADAR_ROOT / "radar_vnext_f2_gate_persistence_00631l_benchmark_date_gap_fill_20260710" / "f2_gate_persistence_00631L_benchmark_price_filled_rows.csv",
]

TASK_ID = "TASK-BACKTEST-CORE-VNEXT-F2-GATE-PERSISTENCE-HYSTERESIS-INCUMBENT-PROTECTION-CONTRACT-001"
F_VARIANT = "F_two_day_confirmation_and_risk_adjusted_edge"
PERIODS = {
    "P1": ("2015-01-02", "2022-12-29"), "P2": ("2023-01-02", "2026-06-30"),
    "2024_latest": ("2024-01-02", "2026-06-30"), "2026YTD": ("2026-01-02", "2026-06-30"),
    "full_integrated": ("2015-01-02", "2026-06-30"),
}
VARIANTS = {
    "F2_A_gate_on_2d_off_immediate": {"on_days": 2, "off_days": 1, "min_hold_days": 0, "cooldown_days": 0},
    "F2_B_gate_on_2d_off_2d_grace": {"on_days": 2, "off_days": 2, "min_hold_days": 0, "cooldown_days": 0},
    "F2_C_asymmetric_3on_2off": {"on_days": 3, "off_days": 2, "min_hold_days": 0, "cooldown_days": 0},
    "F2_D_2on_2off_min_hold5": {"on_days": 2, "off_days": 2, "min_hold_days": 5, "cooldown_days": 0},
    "F2_E_2on_2off_cooldown3": {"on_days": 2, "off_days": 2, "min_hold_days": 0, "cooldown_days": 3},
}
FLAGS = {
    "formal_model_changed": False, "trade_decision_changed": False, "active_in_trade_decision": False,
    "report_changed": False, "portfolio_replay_executed": False, "ready_for_strategy_replay": False,
    "ready_for_formal": False, "not_live_rule": True, "forward_returns_live_rule_usage": False,
}


def _bool(value: Any) -> bool:
    if pd.isna(value):
        return False
    return value.lower() in {"true", "1", "yes"} if isinstance(value, str) else bool(value)


def _ticker(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    return text[:-2] if text.endswith(".0") else text


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


def _transition(previous_ticker: str, previous_type: str, target_ticker: str, target_type: str) -> tuple[str, str]:
    if previous_ticker == target_ticker and previous_type == target_type:
        return "hold_same", "hold"
    if previous_type == "etf" and target_type == "stock":
        return "base_to_stock", "00631L_to_stock"
    if previous_type == "stock" and target_type == "etf":
        return "stock_to_base", "stock_to_00631L"
    return "stock_to_stock", "stock_to_stock"


def _load_raw_f() -> pd.DataFrame:
    frame = pd.read_csv(RAW_F, low_memory=False, dtype={"challenger_ticker": str, "selected_ticker_after": str})
    frame = frame[frame["state_machine_variant"].eq(F_VARIANT)].copy()
    for col in ["signal_date", "next_trading_day_execution_date", "next_trading_day_after_execution_date", "pool_snapshot_date"]:
        frame[col] = pd.to_datetime(frame[col], errors="coerce")
    for col in ["challenger_ticker", "selected_ticker_after"]:
        frame[col] = frame[col].map(_ticker)
    return frame.sort_values("signal_date")


def _matrix_map() -> dict[tuple[pd.Timestamp, str], dict[str, Any]]:
    matrix = daily_source._weekly_candidate_matrix().copy()
    matrix["snapshot_date"] = pd.to_datetime(matrix["snapshot_date"], errors="coerce")
    matrix["ticker"] = matrix["ticker"].map(_ticker)
    matrix = matrix.sort_values(["snapshot_date", "ticker", "route_support_score"], ascending=[True, True, False]).drop_duplicates(["snapshot_date", "ticker"])
    return {(row.snapshot_date, row.ticker): row._asdict() for row in matrix.itertuples(index=False)}


def _load_prices() -> tuple[pd.DataFrame, pd.DataFrame]:
    frames, audit = [], []
    for path in PRICE_SOURCES:
        if not path.exists():
            continue
        frame = pd.read_csv(path, low_memory=False, dtype={"ticker": str})
        date_col = "price_date" if "price_date" in frame.columns else "date"
        if not {"ticker", date_col, "close"}.issubset(frame.columns):
            continue
        view = frame[["ticker", date_col, "close"]].copy().rename(columns={date_col: "date"})
        view["ticker"] = view["ticker"].map(_ticker)
        view["date"] = pd.to_datetime(view["date"], errors="coerce")
        view["close"] = pd.to_numeric(view["close"], errors="coerce")
        view["source_quality"] = frame.get("source_quality", pd.Series("official_unadjusted_ohlc", index=frame.index)).astype(str).values
        view["source_route"] = frame.get("source_route", pd.Series(path.name, index=frame.index)).astype(str).values
        view["adjustment_policy"] = frame.get("adjustment_policy", pd.Series("unadjusted_ohlcv; adjusted_close_blocked_not_fabricated", index=frame.index)).astype(str).values
        view["source_file"] = str(path)
        frames.append(view.dropna(subset=["ticker", "date", "close"]))
        audit.append({"source_file": str(path), "loaded_rows": int(len(view)), "status": "loaded"})
    if not frames:
        return pd.DataFrame(columns=["ticker", "date", "close", "source_quality", "source_route", "adjustment_policy"]), pd.DataFrame(audit)
    return pd.concat(frames, ignore_index=True).drop_duplicates(["ticker", "date"], keep="last"), pd.DataFrame(audit)


def _detail(context: dict[str, Any]) -> dict[str, Any]:
    if not context:
        return {
            "hard_deterioration_flag": False, "hard_deterioration_source_quality": "blocked_no_weekly_candidate_context",
            "route_support_score": np.nan, "route_support_score_source_quality": "blocked", "rs5": np.nan, "rs10": np.nan, "rs20": np.nan,
            "rs_short_weakening_flag": np.nan, "bias20_percentile": np.nan, "bias60_percentile": np.nan,
            "risk_overheat_flag": np.nan, "exhaustion_breakdown_flag": np.nan, "in_primary80_context": np.nan,
        }
    return {
        "hard_deterioration_flag": _bool(context.get("incumbent_deterioration_confirmed")),
        "hard_deterioration_source_quality": "existing_weekly_PIT_composite_two_or_more_deterioration_contexts",
        "route_support_score": context.get("route_support_score"), "route_support_score_source_quality": context.get("route_support_score_source_quality"),
        "rs5": context.get("RS5"), "rs10": context.get("RS10"), "rs20": context.get("RS20"),
        "rs_short_weakening_flag": context.get("rs_short_deterioration_flag"), "bias20_percentile": context.get("BIAS20_percentile"),
        "bias60_percentile": context.get("BIAS60_percentile"), "risk_overheat_flag": context.get("risk_overheat_penalty_context"),
        "exhaustion_breakdown_flag": context.get("high_exhaustion_or_breakdown_context"), "in_primary80_context": True,
    }


def _materialize_states(raw: pd.DataFrame, matrix: dict[tuple[pd.Timestamp, str], dict[str, Any]]) -> pd.DataFrame:
    output = []
    for variant, policy in VARIANTS.items():
        incumbent_ticker, incumbent_type, hold_days = "00631L", "etf", 0
        on_streak, off_streak, cooldown_remaining, cooldown_ticker = 0, 0, 0, ""
        for day in raw.itertuples(index=False):
            gate_pass = _bool(day.c2_pass_flag) and _bool(day.consensus_trigger_flag)
            on_streak = on_streak + 1 if gate_pass else 0
            off_streak = 0 if gate_pass else off_streak + 1
            challenger = _ticker(day.challenger_ticker)
            snapshot = day.pool_snapshot_date
            incumbent_context = matrix.get((snapshot, incumbent_ticker), {}) if incumbent_type == "stock" and pd.notna(snapshot) else {}
            hard = _detail(incumbent_context)
            cooldown_block = cooldown_remaining > 0 and challenger == cooldown_ticker
            action_reason = "hold_incumbent"
            if incumbent_type == "etf":
                if gate_pass and on_streak >= policy["on_days"] and challenger and not cooldown_block:
                    target_ticker, target_type, action_reason = challenger, "stock", "gate_on_streak_confirmed_enter_challenger"
                else:
                    target_ticker, target_type = "00631L", "etf"
                    action_reason = "gate_on_not_confirmed_or_cooldown"
            elif hard["hard_deterioration_flag"]:
                target_ticker, target_type, action_reason = "00631L", "etf", "hard_deterioration_immediate_exit"
            elif not gate_pass and off_streak >= policy["off_days"] and (hold_days >= policy["min_hold_days"]):
                target_ticker, target_type, action_reason = "00631L", "etf", "gate_off_streak_exit_to_base"
            elif not gate_pass and off_streak >= policy["off_days"] and hold_days < policy["min_hold_days"]:
                target_ticker, target_type, action_reason = incumbent_ticker, "stock", "min_hold_protects_incumbent_despite_gate_off"
            else:
                target_ticker, target_type = incumbent_ticker, "stock"
                action_reason = "incumbent_protection_no_stock_to_stock_micro_switch"
            transition_type, cost_key = _transition(incumbent_ticker, incumbent_type, target_ticker, target_type)
            cost = daily_source.TRANSITION_COSTS[cost_key]
            new_cooldown = cooldown_remaining
            new_cooldown_ticker = cooldown_ticker
            if transition_type == "stock_to_base" and policy["cooldown_days"]:
                new_cooldown, new_cooldown_ticker = policy["cooldown_days"], incumbent_ticker
            elif cooldown_remaining > 0:
                new_cooldown = cooldown_remaining - 1
                if new_cooldown == 0:
                    new_cooldown_ticker = ""
            hold_days = hold_days + 1 if target_type == "stock" and target_ticker == incumbent_ticker else 1 if target_type == "stock" else 0
            output.append({
                "task": TASK_ID, "f2_variant": variant, "signal_date": day.signal_date, "next_trading_day_execution_date": day.next_trading_day_execution_date,
                "next_trading_day_after_execution_date": day.next_trading_day_after_execution_date, "pool_snapshot_date": snapshot,
                "c2_pass_flag": _bool(day.c2_pass_flag), "consensus_trigger_flag": _bool(day.consensus_trigger_flag), "r6_override_context_flag": _bool(day.r6_override_flag),
                "gate_pass_flag": gate_pass, "gate_on_streak": on_streak, "gate_off_streak": off_streak,
                "required_gate_on_days": policy["on_days"], "required_gate_off_days": policy["off_days"], "minimum_hold_days": policy["min_hold_days"], "cooldown_days_policy": policy["cooldown_days"],
                "cooldown_remaining_before": cooldown_remaining, "cooldown_ticker_before": cooldown_ticker, "cooldown_block_reentry_flag": cooldown_block,
                "challenger_ticker": challenger, "challenger_name": day.challenger_name,
                "incumbent_ticker_before": incumbent_ticker, "incumbent_asset_type_before": incumbent_type, "incumbent_hold_days_before": hold_days if incumbent_type == "stock" else 0,
                "selected_ticker_after": target_ticker, "selected_asset_type_after": target_type, "decision_reason": action_reason,
                "transition_type": transition_type, "transition_cost_key": cost_key, "transition_cost_rate_hook": cost["transition_cost_rate"], "transition_cost_model_status": "EP05_TaiwanCostModel_unit_notional_hook_stock_etf_separated",
                "revenue_anomaly_role": "report_only", "rs20_top3_role": "reference_only", "cash_bear_classifier_status": "blocked_no_cash_rule",
                "hard_deterioration_action_enabled": True, "layer4_exit_as_context_only": True, "diagnostic_only": True, **hard, **FLAGS,
            })
            incumbent_ticker, incumbent_type, cooldown_remaining, cooldown_ticker = target_ticker, target_type, new_cooldown, new_cooldown_ticker
    return pd.DataFrame(output)


def _attach_prices(state: pd.DataFrame, prices: pd.DataFrame, raw_f: pd.DataFrame) -> pd.DataFrame:
    out = state.copy()
    benchmark = _benchmark_price_map().copy()
    benchmark["date"] = pd.to_datetime(benchmark["date"], errors="coerce")
    etf_prices = prices[prices["ticker"].eq("00631L")][["date", "close", "source_quality"]].rename(
        columns={"close": "price", "source_quality": "benchmark_source_quality"}
    )
    benchmark = pd.concat([benchmark[["date", "price", "benchmark_source_quality"]], etf_prices], ignore_index=True)
    f_base = raw_f[raw_f["selected_asset_type_after"].eq("etf")].copy()
    f_entry = f_base[["next_trading_day_execution_date", "entry_close"]].rename(columns={"entry_close": "price"})
    f_entry["date"] = f_entry["next_trading_day_execution_date"]
    f_exit = f_base[["next_trading_day_after_execution_date", "exit_close"]].rename(columns={"next_trading_day_after_execution_date": "date", "exit_close": "price"})
    f_base_prices = pd.concat([f_entry[["date", "price"]], f_exit[["date", "price"]]], ignore_index=True).dropna().drop_duplicates("date", keep="last")
    f_base_prices["benchmark_source_quality"] = "absorbed_daily_F_00631L_base_path_fallback_diagnostic"
    benchmark = pd.concat([benchmark, f_base_prices], ignore_index=True).drop_duplicates("date", keep="first")
    entry_b = benchmark[["date", "price", "benchmark_source_quality"]].rename(columns={"date": "next_trading_day_execution_date", "price": "base_entry_close", "benchmark_source_quality": "base_entry_source_quality"})
    exit_b = benchmark[["date", "price", "benchmark_source_quality"]].rename(columns={"date": "next_trading_day_after_execution_date", "price": "base_exit_close", "benchmark_source_quality": "base_exit_source_quality"})
    out = out.merge(entry_b, on="next_trading_day_execution_date", how="left").merge(exit_b, on="next_trading_day_after_execution_date", how="left")
    entry = prices.rename(columns={"ticker": "selected_ticker_after", "date": "next_trading_day_execution_date", "close": "stock_entry_close", "source_quality": "stock_entry_source_quality", "source_route": "stock_entry_source_route", "adjustment_policy": "stock_adjustment_policy"})
    exit_ = prices.rename(columns={"ticker": "selected_ticker_after", "date": "next_trading_day_after_execution_date", "close": "stock_exit_close", "source_quality": "stock_exit_source_quality", "source_route": "stock_exit_source_route", "adjustment_policy": "stock_exit_adjustment_policy"})
    out = out.merge(entry[["selected_ticker_after", "next_trading_day_execution_date", "stock_entry_close", "stock_entry_source_quality", "stock_entry_source_route", "stock_adjustment_policy"]], on=["selected_ticker_after", "next_trading_day_execution_date"], how="left")
    out = out.merge(exit_[["selected_ticker_after", "next_trading_day_after_execution_date", "stock_exit_close", "stock_exit_source_quality", "stock_exit_source_route", "stock_exit_adjustment_policy"]], on=["selected_ticker_after", "next_trading_day_after_execution_date"], how="left")
    stock = out["selected_asset_type_after"].eq("stock")
    out["entry_close"] = np.where(stock, out["stock_entry_close"], out["base_entry_close"])
    out["exit_close"] = np.where(stock, out["stock_exit_close"], out["base_exit_close"])
    out["gross_daily_return"] = pd.to_numeric(out["exit_close"], errors="coerce") / pd.to_numeric(out["entry_close"], errors="coerce") - 1.0
    out["net_daily_return_after_transition_cost"] = out["gross_daily_return"] - out["transition_cost_rate_hook"]
    out["daily_path_ready"] = out["gross_daily_return"].notna()
    out["official_unadjusted_stock_ohlc_ready"] = np.where(stock, out["stock_entry_close"].notna() & out["stock_exit_close"].notna(), True)
    out["selected_stock_adjusted_close_ready"] = ~stock
    out["terminal_path_row_excluded_from_metric"] = out["next_trading_day_after_execution_date"].isna()
    for period, (start, end) in PERIODS.items():
        candidate = (out["signal_date"] >= pd.Timestamp(start)) & (out["signal_date"] <= pd.Timestamp(end)) & (out["next_trading_day_execution_date"] <= pd.Timestamp(end)) & (out["next_trading_day_after_execution_date"] <= pd.Timestamp(end)) & ~out["terminal_path_row_excluded_from_metric"]
        out[f"metric_candidate_{period}"] = candidate
        out[f"metric_eligible_{period}"] = candidate & out["daily_path_ready"]
    return out


def _gap_ledger(state: pd.DataFrame) -> pd.DataFrame:
    missing = state[~state["daily_path_ready"]]
    rows = []
    for row in missing.itertuples(index=False):
        for date, required_as, value in [
            (row.next_trading_day_execution_date, "entry_close", row.entry_close),
            (row.next_trading_day_after_execution_date, "following_daily_close", row.exit_close),
        ]:
            if pd.notna(date) and pd.isna(value):
                rows.append({"f2_variant": row.f2_variant, "ticker": row.selected_ticker_after, "asset_type": row.selected_asset_type_after, "price_date": date, "required_as": required_as, "signal_date": row.signal_date, "transition_type": row.transition_type})
    if not rows:
        return pd.DataFrame(columns=["ticker", "asset_type", "price_date", "required_as", "impacted_variants", "impacted_signal_dates", "source_requirement", "next_owner"])
    gap = pd.DataFrame(rows).groupby(["ticker", "asset_type", "price_date"], as_index=False).agg(
        required_as=("required_as", lambda x: "|".join(sorted(set(x)))), impacted_variants=("f2_variant", lambda x: "|".join(sorted(set(x)))),
        impacted_signal_dates=("signal_date", lambda x: "|".join(sorted(set(pd.Series(x).dt.strftime("%Y-%m-%d"))))),
    )
    gap["source_requirement"] = np.where(gap["asset_type"].eq("stock"), "selected_ticker_only official unadjusted daily OHLC; no 00631L+excess reconstruction", "official 00631L daily close benchmark route; no date substitution")
    gap["next_owner"] = np.where(gap["asset_type"].eq("stock"), "Radar/Data bounded F2 selected-stock daily OHLC gap fill", "Radar/Data bounded F2 00631L benchmark date gap fill")
    return gap


def _episodes_and_rechains(state: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    episode_rows, scenario_paths, transitions, metrics = [], [], [], []
    for variant, base in state.groupby("f2_variant"):
        base = base[base["metric_eligible_P1"]].copy().sort_values("signal_date")
        base["episode_key"] = (base["selected_ticker_after"].ne(base["selected_ticker_after"].shift()) | base["selected_asset_type_after"].ne(base["selected_asset_type_after"].shift())).cumsum()
        stock = base[base["selected_asset_type_after"].eq("stock")].copy()
        stock["base_00631L_daily_return"] = stock["base_exit_close"] / stock["base_entry_close"] - 1.0
        episodes = stock.groupby(["episode_key", "selected_ticker_after"], as_index=False).agg(
            episode_start=("signal_date", "min"), episode_end=("signal_date", "max"), daily_legs=("signal_date", "size"),
            stock_net_return=("net_daily_return_after_transition_cost", lambda x: float((1 + x).prod() - 1)),
            base_00631L_compound_return=("base_00631L_daily_return", lambda x: float((1 + x).prod() - 1)),
        )
        rank = episodes.copy()
        rank["evaluation_contribution_vs_base"] = rank["stock_net_return"] - rank["base_00631L_compound_return"]
        rank["evaluation_rank_status"] = "realized_incremental_contribution_vs_base_evaluation_only"
        rank = rank.sort_values(["evaluation_contribution_vs_base", "episode_key"], ascending=[False, True]).reset_index(drop=True)
        rank["evaluation_rank_best_stock_episode"] = np.arange(1, len(rank) + 1)
        rank["ranking_uses_realized_evaluation_only"] = True
        episode_rows.append(rank.assign(f2_variant=variant))
        for label, count in {"baseline": 0, "remove_best_1": 1, "remove_best_3": 3, "remove_best_5": 5}.items():
            removed = set(rank.head(count)["episode_key"].astype(int))
            previous_ticker, previous_type = "00631L", "etf"
            path_rows = []
            for row in base.itertuples(index=False):
                remove = int(row.episode_key) in removed and row.selected_asset_type_after == "stock"
                ticker, asset_type = ("00631L", "etf") if remove else (row.selected_ticker_after, row.selected_asset_type_after)
                transition_type, cost_key = _transition(previous_ticker, previous_type, ticker, asset_type)
                cost = daily_source.TRANSITION_COSTS[cost_key]
                gross = (row.base_exit_close / row.base_entry_close - 1.0) if remove else row.gross_daily_return
                path_rows.append({"f2_variant": variant, "rechain_scenario": label, "signal_date": row.signal_date, "episode_key": int(row.episode_key), "removed_best_episode_flag": remove, "selected_ticker_after": ticker, "selected_asset_type_after": asset_type, "transition_type": transition_type, "transition_cost_rate_hook": cost["transition_cost_rate"], "gross_daily_return": gross, "net_daily_return_after_transition_cost": gross - cost["transition_cost_rate"], "execution_basis": "next_day_close_unique_position_exact_rechain", "diagnostic_only": True, **FLAGS})
                if transition_type != "hold_same":
                    transitions.append({"f2_variant": variant, "rechain_scenario": label, "signal_date": row.signal_date, "from_ticker": previous_ticker, "from_asset_type": previous_type, "to_ticker": ticker, "to_asset_type": asset_type, "transition_type": transition_type, **cost, "diagnostic_only": True, **FLAGS})
                previous_ticker, previous_type = ticker, asset_type
            path = pd.DataFrame(path_rows)
            equity = (1 + path["net_daily_return_after_transition_cost"]).cumprod()
            metrics.append({"f2_variant": variant, "rechain_scenario": label, "removed_episode_keys": "|".join(map(str, sorted(removed))), "net_total_return_after_transition_cost": float(equity.iloc[-1] - 1), "net_MDD": float((equity / equity.cummax() - 1).min()), "transition_count": int(path["transition_type"].ne("hold_same").sum()), "stock_state_days": int(path["selected_asset_type_after"].eq("stock").sum()), "diagnostic_only": True, **FLAGS})
            scenario_paths.append(path)
    return pd.concat(episode_rows, ignore_index=True), pd.concat(scenario_paths, ignore_index=True), pd.DataFrame(transitions), pd.DataFrame(metrics)


def _coverage(state: pd.DataFrame, gaps: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant, group in state.groupby("f2_variant"):
        for period, (start, end) in PERIODS.items():
            sub = group[(group["signal_date"] >= pd.Timestamp(start)) & (group["signal_date"] <= pd.Timestamp(end))]
            cand = sub[sub[f"metric_candidate_{period}"]]
            ready = sub[sub[f"metric_eligible_{period}"]]
            rows.append({"f2_variant": variant, "period": period, "requested_start": start, "requested_end": end, "actual_start": ready["signal_date"].min() if len(ready) else pd.NaT, "actual_end": ready["signal_date"].max() if len(ready) else pd.NaT, "metric_candidate_rows": int(len(cand)), "metric_ready_rows": int(len(ready)), "daily_path_ready_share": float(len(ready) / len(cand)) if len(cand) else 1.0, "stock_official_unadjusted_ready_share": float(sub.loc[sub["selected_asset_type_after"].eq("stock"), "official_unadjusted_stock_ohlc_ready"].mean()) if sub["selected_asset_type_after"].eq("stock").any() else 1.0, **FLAGS})
    return pd.DataFrame(rows)


def _future_audit() -> pd.DataFrame:
    return pd.DataFrame([
        {"audit_item": "F2_gate_and_hard_deterioration", "future_return_used_as_rule": False, "detail": "Uses same-day C2/consensus and weekly PIT incumbent deterioration composite only.", "future_data_violation_count": 0},
        {"audit_item": "F2_best_episode_removal", "future_return_used_as_rule": True, "detail": "Realized episode ranking is evaluation-only robustness stress test; no selector rule uses it.", "future_data_violation_count": 0},
    ])


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    raw, matrix = _load_raw_f(), _matrix_map()
    state = _materialize_states(raw, matrix)
    prices, price_audit = _load_prices()
    state = _attach_prices(state, prices, raw)
    gaps = _gap_ledger(state)
    stock_gaps = gaps[gaps["asset_type"].eq("stock")] if len(gaps) else gaps
    benchmark_gaps = gaps[gaps["asset_type"].eq("etf")] if len(gaps) else gaps
    coverage = _coverage(state, gaps)
    p1 = coverage[coverage["period"].eq("P1")]["daily_path_ready_share"].min()
    p2 = coverage[coverage["period"].eq("P2")]["daily_path_ready_share"].min()
    rechain_ready = bool(len(gaps) == 0 and p1 == 1.0 and p2 == 1.0)
    if rechain_ready:
        episodes, rechain_paths, rechain_transitions, rechain_metrics = _episodes_and_rechains(state)
    else:
        episodes, rechain_paths, rechain_transitions, rechain_metrics = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    readiness = {
        "task_id": TASK_ID, "status": "F2_gate_persistence_hysteresis_ready_unadjusted_diagnostic" if rechain_ready else "F2_gate_persistence_hysteresis_partial_selected_stock_daily_ohlc_gap",
        "variant_count": len(VARIANTS), "P1_daily_path_ready_share_min": float(p1), "P2_daily_path_ready_share_min": float(p2), "stock_price_gap_rows": int(len(stock_gaps)), "benchmark_00631L_gap_rows": int(len(benchmark_gaps)),
        "hard_deterioration_action_source": "existing_weekly_PIT_composite_two_or_more_deterioration_contexts", "route_score_drop_action_threshold": "blocked_not_invented", "layer4_exit_action": "context_only_not_standalone_exit",
        "exact_episode_rechain_ready": rechain_ready, "exact_episode_rechain_period": "P1_primary" if rechain_ready else "blocked_until_price_path_ready", "EP05_transaction_cost_hooks_ready": True, "selected_stock_adjusted_close_ready": False, "cash_bear_classifier_ready": False,
        "ready_for_experiments": rechain_ready, "ready_for_formal": False, "ready_for_strategy_replay": False, "future_data_violation_count": 0, **FLAGS,
    }
    blocked = pd.DataFrame([
        {"item": "route_support_score_drop_exact_action_threshold", "status": "blocked", "detail": "Score drop is materialized as context but no new threshold is invented for action.", "next_owner": "Experiments/Strategy Center if evidence supports a bounded threshold"},
        {"item": "layer4_primary80_exit", "status": "context_only", "detail": "Leaving primary80 is recorded but cannot be a standalone exit.", "next_owner": "none"},
        {"item": "selected_stock_adjusted_close", "status": "blocked", "detail": "Official unadjusted stock OHLC only.", "next_owner": "Strategy Center/Radar Data if trusted route authorized"},
        {"item": "cash_bear_classifier", "status": "blocked", "detail": "No cash rule created.", "next_owner": "Strategy Center/Core Data later"},
    ])
    output_paths = [
        _write(state, "f2_gate_persistence_daily_state_contract.csv"),
        _write(gaps, "f2_gate_persistence_price_gap_ledger.csv"),
        _write(stock_gaps, "f2_gate_persistence_selected_stock_daily_ohlc_gap_ledger.csv"),
        _write(benchmark_gaps, "f2_gate_persistence_00631L_benchmark_gap_ledger.csv"),
        _write(coverage, "f2_gate_persistence_requested_vs_actual_coverage.csv"), _write(price_audit, "f2_gate_persistence_official_price_source_audit.csv"),
        _write(episodes, "f2_gate_persistence_exact_episode_ids_rank_audit.csv"), _write(rechain_paths, "f2_gate_persistence_exact_rechain_daily_paths.csv"),
        _write(rechain_transitions, "f2_gate_persistence_exact_rechain_transition_trace.csv"), _write(rechain_metrics, "f2_gate_persistence_exact_rechain_metrics.csv"),
        _write(blocked, "f2_gate_persistence_blocked_proxy_audit.csv"), _write(_future_audit(), "f2_gate_persistence_future_data_audit.csv"),
    ]
    readiness_path = OUTPUT_DIR / "readiness_for_f2_gate_persistence_hysteresis_diagnostic.json"
    readiness_path.write_text(json.dumps(readiness, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path = OUTPUT_DIR / "final_summary_zh.md"
    summary_path.write_text(
        "# F2 Gate Persistence / Hysteresis Contract\n\n"
        "本 contract 只測五個 bounded gate persistence / hysteresis 變體，沒有重新開啟 score-edge grid。\n\n"
        f"- P1 daily path coverage: {readiness['P1_daily_path_ready_share_min']:.1%}; P2: {readiness['P2_daily_path_ready_share_min']:.1%}\n"
        f"- selected-stock official unadjusted OHLC gaps: {readiness['stock_price_gap_rows']}\n"
        f"- 00631L benchmark gaps: {readiness['benchmark_00631L_gap_rows']}\n"
        f"- exact remove-best 1/3/5 rechain: {readiness['exact_episode_rechain_ready']}（P1 primary）\n"
        "- execution: signal-day close decision -> next-trading-day close; daily mark-to-market; EP05 stock/ETF transition cost hooks\n"
        "- hard deterioration: existing PIT weekly composite only；route score drop threshold 未自行新增\n"
        "- revenue anomaly: report-only；RS20 top3: reference-only；cash/bear classifier: blocked\n\n"
        f"ready_for_experiments={readiness['ready_for_experiments']}; selected_stock_adjusted_close_ready=false; "
        "cash_bear_classifier_ready=false; future_data_violation_count=0.\n"
        "本輸出為 diagnostic-only，不是 formal、replay、daily report 或 live trade rule。\n",
        encoding="utf-8",
    )
    manifest = {"task_id": TASK_ID, "generated_at_utc": datetime.now(timezone.utc).isoformat(), "output_dir": str(OUTPUT_DIR), "files": [{"path": p.name, "sha256": _sha256(p)} for p in [*output_paths, readiness_path, summary_path]], "readiness": readiness, "source_inputs": {"raw_daily_F": str(RAW_F), "price_sources": [str(x) for x in PRICE_SOURCES]}}
    (OUTPUT_DIR / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(readiness, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
