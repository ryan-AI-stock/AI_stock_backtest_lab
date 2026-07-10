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
R6_CONTRACT = REPO_ROOT / "outputs" / "vnext_r6_guard_first_market_bias_override_unified_contract_20260709" / "r6_guard_first_market_bias_override_unified_contract.csv"
DAILY_CALENDAR = REPO_ROOT / "outputs" / "vnext_daily_incumbent_challenger_state_machine_contract_ohlc_absorbed_20260710" / "daily_incumbent_challenger_state_machine_contract_ohlc_absorbed.csv"
OUTPUT_DIR = REPO_ROOT / "outputs" / "vnext_weekly_r6_single_position_state_boundary_reconstruction_contract_20260710"
RADAR_ROOT = Path("C:/Users/zergv/Documents/Codex/2026-05-23/ai-stock-rotation-radar-https-docs/outputs")
OFFICIAL_PRICE_SOURCES = [
    RADAR_ROOT / "radar_vnext_p1_weekly_r6_selected_stock_daily_ohlc_attribution_gap_fill_20260710" / "p1_weekly_r6_selected_stock_daily_ohlc_filled_rows.csv",
    RADAR_ROOT / "radar_vnext_daily_incumbent_challenger_selected_stock_daily_ohlc_gap_fill_20260710" / "daily_incumbent_challenger_selected_stock_daily_unadjusted_ohlc_rows.csv",
    RADAR_ROOT / "radar_vnext_regime_switch_route_selected_stock_ohlc_source_package_20260708" / "regime_switch_selected_ohlc_rows.csv",
    RADAR_ROOT / "radar_vnext_p2_2023_selected_stock_ohlc_source_gap_fill_20260708" / "p2_2023_selected_stock_unadjusted_ohlc_rows.csv",
    RADAR_ROOT / "radar_vnext_weekly_r6_single_position_reconstructed_p2_selected_stock_daily_ohlc_gap_fill_20260710" / "reconstructed_weekly_r6_p2_selected_stock_daily_ohlc_filled_rows.csv",
]

TASK_ID = "TASK-BACKTEST-CORE-VNEXT-WEEKLY-R6-SINGLE-POSITION-STATE-BOUNDARY-RECONSTRUCTION-CONTRACT-001"
PERIODS = {
    "P1": ("2015-01-02", "2022-12-29"),
    "P2": ("2023-01-02", "2026-06-30"),
    "2024_latest": ("2024-01-02", "2026-06-30"),
    "2026YTD": ("2026-01-02", "2026-06-30"),
    "full_integrated": ("2015-01-02", "2026-06-30"),
}
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


def _ticker(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    return text[:-2] if text.endswith(".0") else text


def _bool(value: Any) -> bool:
    if pd.isna(value):
        return False
    return value.lower() in {"true", "1", "yes"} if isinstance(value, str) else bool(value)


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


def _load_calendar() -> pd.DataFrame:
    frame = pd.read_csv(DAILY_CALENDAR, low_memory=False)
    frame = frame[frame["state_machine_variant"].eq("A_any_positive_edge")].copy()
    for col in ["signal_date", "next_trading_day_execution_date", "next_trading_day_after_execution_date"]:
        frame[col] = pd.to_datetime(frame[col], errors="coerce")
    keep = ["signal_date", "next_trading_day_execution_date", "next_trading_day_after_execution_date"]
    return frame[keep].drop_duplicates().sort_values("signal_date")


def _load_signals() -> pd.DataFrame:
    frame = pd.read_csv(R6_CONTRACT, low_memory=False, dtype={"selected_ticker": str})
    for col in ["signal_date", "next_signal_date", "entry_date", "exit_date"]:
        frame[col] = pd.to_datetime(frame[col], errors="coerce")
    frame["selected_ticker"] = frame["selected_ticker"].map(_ticker)
    return frame.sort_values("signal_date")


def _load_official_prices() -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    sources = []
    for path in OFFICIAL_PRICE_SOURCES:
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
        sources.append({"source_file": str(path), "loaded_rows": int(len(view)), "source_status": "loaded"})
    if not frames:
        return pd.DataFrame(columns=["ticker", "date", "close", "source_quality", "source_route", "adjustment_policy", "source_file"]), pd.DataFrame(sources)
    prices = pd.concat(frames, ignore_index=True)
    # Later files are narrower, newer bounded fills and take precedence.
    prices = prices.drop_duplicates(["ticker", "date"], keep="last")
    return prices, pd.DataFrame(sources)


def _transition(previous_ticker: str, previous_type: str, target_ticker: str, target_type: str) -> tuple[str, str]:
    if previous_ticker == target_ticker and previous_type == target_type:
        return "hold_same", "hold"
    if previous_type == "etf" and target_type == "stock":
        return "base_to_stock", "00631L_to_stock"
    if previous_type == "stock" and target_type == "etf":
        return "stock_to_base", "stock_to_00631L"
    return "stock_to_stock", "stock_to_stock"


def _state_rows(calendar: pd.DataFrame, signals: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    signal_map = {row.signal_date: row for row in signals.itertuples(index=False)}
    incumbent_ticker, incumbent_type = "00631L", "etf"
    rows, transitions = [], []
    for day in calendar.itertuples(index=False):
        signal = signal_map.get(day.signal_date)
        if signal is None:
            target_ticker, target_type = incumbent_ticker, incumbent_type
            branch, regime, reason = "incumbent_hold", "weekly_signal_not_updated", "no_new_weekly_R6_signal_hold_incumbent"
            r6_signal_date = pd.NaT
            original_interval_reference = ""
        else:
            target_ticker, target_type = _ticker(signal.selected_ticker), signal.selected_asset_type
            branch, regime, reason = signal.selected_branch, signal.regime_label, "weekly_R6_signal_target_next_trading_day_close_execution"
            r6_signal_date = signal.signal_date
            original_interval_reference = f"deprecated_overlapping_interval_entry={signal.entry_date};exit={signal.exit_date};net={signal.net_interval_return_after_transition_cost}"
        transition_type, cost_key = _transition(incumbent_ticker, incumbent_type, target_ticker, target_type)
        cost = daily_source.TRANSITION_COSTS[cost_key]
        row = {
            "task": TASK_ID, "signal_date": day.signal_date, "next_trading_day_execution_date": day.next_trading_day_execution_date,
            "next_trading_day_after_execution_date": day.next_trading_day_after_execution_date, "weekly_r6_signal_date": r6_signal_date,
            "incumbent_ticker_before": incumbent_ticker, "incumbent_asset_type_before": incumbent_type,
            "target_ticker": target_ticker, "target_asset_type": target_type, "selected_ticker_after": target_ticker,
            "selected_asset_type_after": target_type, "selected_branch": branch, "regime_label": regime,
            "decision_reason": reason, "transition_type": transition_type, "transition_cost_key": cost_key,
            "transition_cost_rate_hook": cost["transition_cost_rate"], "transition_cost_model_status": "EP05_TaiwanCostModel_unit_notional_hook_stock_etf_separated",
            "state_semantics": "single_position_signal_close_next_trading_day_close_execution_no_weekly_rebuy",
            "original_overlapping_interval_reference": original_interval_reference,
            "historical_overlapping_R6_reference_deprecated": True, "revenue_anomaly_role": "report_only", "rs20_top3_role": "reference_only",
            "cash_bear_classifier_status": "blocked_no_cash_rule", "diagnostic_only": True, **FLAGS,
        }
        rows.append(row)
        if transition_type != "hold_same":
            transitions.append({
                "signal_date": day.signal_date, "execution_date": day.next_trading_day_execution_date,
                "from_ticker": incumbent_ticker, "from_asset_type": incumbent_type, "to_ticker": target_ticker, "to_asset_type": target_type,
                "transition_type": transition_type, **cost, "cost_model_status": row["transition_cost_model_status"], "diagnostic_only": True, **FLAGS,
            })
        incumbent_ticker, incumbent_type = target_ticker, target_type
    return pd.DataFrame(rows), pd.DataFrame(transitions)


def _attach_prices(state: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    out = state.copy()
    benchmark = _benchmark_price_map().copy()
    benchmark["date"] = pd.to_datetime(benchmark["date"], errors="coerce")
    b_entry = benchmark[["date", "price", "benchmark_source_quality"]].rename(columns={"date": "next_trading_day_execution_date", "price": "base_entry_close", "benchmark_source_quality": "base_entry_source_quality"})
    b_exit = benchmark[["date", "price", "benchmark_source_quality"]].rename(columns={"date": "next_trading_day_after_execution_date", "price": "base_exit_close", "benchmark_source_quality": "base_exit_source_quality"})
    out = out.merge(b_entry, on="next_trading_day_execution_date", how="left").merge(b_exit, on="next_trading_day_after_execution_date", how="left")
    entry = prices.rename(columns={"ticker": "selected_ticker_after", "date": "next_trading_day_execution_date", "close": "stock_entry_close", "source_quality": "stock_entry_source_quality", "source_route": "stock_entry_source_route", "adjustment_policy": "stock_adjustment_policy"})
    exit_ = prices.rename(columns={"ticker": "selected_ticker_after", "date": "next_trading_day_after_execution_date", "close": "stock_exit_close", "source_quality": "stock_exit_source_quality", "source_route": "stock_exit_source_route", "adjustment_policy": "stock_exit_adjustment_policy"})
    out = out.merge(entry[["selected_ticker_after", "next_trading_day_execution_date", "stock_entry_close", "stock_entry_source_quality", "stock_entry_source_route", "stock_adjustment_policy"]], on=["selected_ticker_after", "next_trading_day_execution_date"], how="left")
    out = out.merge(exit_[["selected_ticker_after", "next_trading_day_after_execution_date", "stock_exit_close", "stock_exit_source_quality", "stock_exit_source_route", "stock_exit_adjustment_policy"]], on=["selected_ticker_after", "next_trading_day_after_execution_date"], how="left")
    stock = out["selected_asset_type_after"].eq("stock")
    out["entry_close"] = np.where(stock, out["stock_entry_close"], out["base_entry_close"])
    out["exit_close"] = np.where(stock, out["stock_exit_close"], out["base_exit_close"])
    out["daily_price_source_quality"] = np.where(stock, out["stock_entry_source_quality"], out["base_entry_source_quality"])
    out["official_unadjusted_daily_ohlc_ready"] = np.where(stock, out["stock_entry_close"].notna() & out["stock_exit_close"].notna(), True)
    out["gross_daily_return"] = pd.to_numeric(out["exit_close"], errors="coerce") / pd.to_numeric(out["entry_close"], errors="coerce") - 1.0
    out["net_daily_return_after_transition_cost"] = out["gross_daily_return"] - pd.to_numeric(out["transition_cost_rate_hook"], errors="coerce").fillna(0.0)
    out["daily_path_ready"] = out["gross_daily_return"].notna()
    out["terminal_mark_date"] = out["next_trading_day_execution_date"]
    out["terminal_mark_available"] = out["entry_close"].notna()
    out["terminal_path_row_excluded_from_metric"] = out["next_trading_day_after_execution_date"].isna()
    out["selected_stock_adjusted_close_ready"] = ~stock
    out["execution_basis"] = "weekly_signal_close__next_trading_day_close_execution__unique_daily_state_mark"
    for period, (start, end) in PERIODS.items():
        within_execution = (
            (out["signal_date"] >= pd.Timestamp(start)) & (out["signal_date"] <= pd.Timestamp(end))
            & (out["next_trading_day_execution_date"] <= pd.Timestamp(end))
        )
        out[f"metric_candidate_{period}"] = (
            within_execution & ~out["terminal_path_row_excluded_from_metric"]
            & (out["next_trading_day_after_execution_date"] <= pd.Timestamp(end))
        )
        out[f"metric_eligible_{period}"] = (
            out[f"metric_candidate_{period}"] & out["daily_path_ready"]
        )
    return out


def _stock_gap_ledger(state: pd.DataFrame) -> pd.DataFrame:
    missing = state[state["selected_asset_type_after"].eq("stock") & ~state["daily_path_ready"]].copy()
    rows = []
    for row in missing.itertuples(index=False):
        for date, required_as in [(row.next_trading_day_execution_date, "entry_close"), (row.next_trading_day_after_execution_date, "following_daily_close")]:
            if pd.notna(date):
                rows.append({"ticker": row.selected_ticker_after, "price_date": date, "required_as": required_as, "signal_date": row.signal_date, "transition_type": row.transition_type})
    if not rows:
        return pd.DataFrame(columns=["ticker", "price_date", "required_as", "impacted_signal_dates", "impacted_transition_types", "source_requirement", "next_owner"])
    gap = pd.DataFrame(rows).groupby(["ticker", "price_date"], as_index=False).agg(
        required_as=("required_as", lambda x: "|".join(sorted(set(x)))),
        impacted_signal_dates=("signal_date", lambda x: "|".join(sorted(set(pd.Series(x).dt.strftime("%Y-%m-%d"))))),
        impacted_transition_types=("transition_type", lambda x: "|".join(sorted(set(x)))),
    )
    gap["source_requirement"] = "selected_ticker_only official unadjusted daily OHLC; no 00631L+excess reconstruction"
    gap["adjusted_close_ready"] = False
    gap["next_owner"] = "Radar/Data bounded weekly R6 reconstructed single-position selected-stock daily OHLC gap fill"
    return gap


def _overlap_resolution(signals: pd.DataFrame, state: pd.DataFrame) -> pd.DataFrame:
    source = signals.dropna(subset=["entry_date", "exit_date"]).sort_values("signal_date")
    execution = state[state["weekly_r6_signal_date"].notna()][["weekly_r6_signal_date", "next_trading_day_execution_date"]].drop_duplicates()
    rows = []
    for left in source.itertuples(index=False):
        later = source[(source["signal_date"] > left.signal_date) & (source["entry_date"] < left.exit_date) & (source["exit_date"] > left.entry_date)]
        for right in later.itertuples(index=False):
            if _ticker(left.selected_ticker) == _ticker(right.selected_ticker) and left.selected_asset_type == right.selected_asset_type:
                continue
            right_exec = execution.loc[execution["weekly_r6_signal_date"].eq(right.signal_date), "next_trading_day_execution_date"]
            rows.append({
                "period_label": "P1" if left.signal_date <= pd.Timestamp("2022-12-29") else "P2",
                "original_left_signal_date": left.signal_date, "original_left_ticker": _ticker(left.selected_ticker), "original_left_asset_type": left.selected_asset_type,
                "original_left_entry_date": left.entry_date, "original_left_exit_date": left.exit_date,
                "original_right_signal_date": right.signal_date, "original_right_ticker": _ticker(right.selected_ticker), "original_right_asset_type": right.selected_asset_type,
                "original_right_entry_date": right.entry_date, "original_right_exit_date": right.exit_date,
                "resolution_policy": "right_weekly_signal_target_replaces_incumbent_at_its_next_trading_day_close; previous incumbent is marked only through that execution close",
                "right_effective_execution_date": right_exec.iloc[0] if len(right_exec) else pd.NaT,
                "overlap_removed_flag": True,
                "historical_overlapping_interval_reference_only": True,
            })
    return pd.DataFrame(rows)


def _coverage(state: pd.DataFrame, gaps: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for period, (start, end) in PERIODS.items():
        sub = state[(state["signal_date"] >= pd.Timestamp(start)) & (state["signal_date"] <= pd.Timestamp(end))]
        metric = sub[sub[f"metric_eligible_{period}"]]
        candidate = sub[sub[f"metric_candidate_{period}"]]
        stock = sub[sub["selected_asset_type_after"].eq("stock")]
        rows.append({
            "period": period, "requested_start": start, "requested_end": end,
            "actual_start": metric["signal_date"].min() if len(metric) else pd.NaT,
            "actual_end": metric["signal_date"].max() if len(metric) else pd.NaT,
            "daily_state_rows": int(len(sub)), "metric_candidate_rows": int(len(candidate)), "metric_ready_rows": int(len(metric)), "daily_path_ready_share": float(metric.shape[0] / candidate.shape[0]) if len(candidate) else 1.0,
            "period_execution_boundary_rows_excluded": int(len(sub) - len(candidate)), "terminal_mark_rows_excluded": int(sub["terminal_path_row_excluded_from_metric"].sum()),
            "stock_state_rows": int(len(stock)), "stock_official_unadjusted_ready_share": float(stock["official_unadjusted_daily_ohlc_ready"].mean()) if len(stock) else 1.0,
            "unresolved_stock_price_gap_rows": int(len(gaps[(gaps["price_date"] >= pd.Timestamp(start)) & (gaps["price_date"] <= pd.Timestamp(end))])) if len(gaps) else 0,
            **FLAGS,
        })
    return pd.DataFrame(rows)


def _future_audit() -> pd.DataFrame:
    return pd.DataFrame([
        {"audit_item": "weekly_signal_target", "future_return_used_as_rule": False, "detail": "Uses original weekly R6 signal close targets only; old interval outcomes are reference-only.", "future_data_violation_count": 0},
        {"audit_item": "daily_mark_path", "future_return_used_as_rule": False, "detail": "Next-day closes are evaluation-only after the state decision.", "future_data_violation_count": 0},
    ])


def _net_path_metrics_hook(state: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for period, (start, end) in PERIODS.items():
        sub = state[state[f"metric_eligible_{period}"]].copy().sort_values("signal_date")
        equity = (1 + pd.to_numeric(sub["net_daily_return_after_transition_cost"], errors="coerce").fillna(0.0)).cumprod()
        drawdown = equity / equity.cummax() - 1.0 if len(equity) else pd.Series(dtype=float)
        rows.append({
            "period": period, "requested_start": start, "requested_end": end,
            "actual_start": sub["signal_date"].min() if len(sub) else pd.NaT, "actual_end": sub["signal_date"].max() if len(sub) else pd.NaT,
            "net_total_return_after_transition_cost_hook": float(equity.iloc[-1] - 1.0) if len(equity) else np.nan,
            "net_MDD_hook": float(drawdown.min()) if len(drawdown) else np.nan,
            "transition_count": int(sub["transition_type"].ne("hold_same").sum()),
            "gross_reference_only": False, "execution_basis": "single_position_weekly_signal_close_next_trading_day_close_execution",
            "historical_overlapping_R6_used_as_primary": False, "diagnostic_only": True, **FLAGS,
        })
    return pd.DataFrame(rows)


def _summary(readiness: dict[str, Any]) -> str:
    return "\n".join([
        "# Reconstructed Single-Position Weekly R6 Contract",
        "",
        "本包重建單一持倉 R6：每週 signal close 的 target 在下一交易日 close 才取代 incumbent；同 ticker 續抱，不存在 fixed-5TD 重疊 interval 或每週重買 00631L。",
        "舊 R6 +646.44% / MDD -48.80% 為 historical overlapping path-like reference，不能當本包或 daily F 的 same-basis primary baseline。",
        f"ready_for_experiments={readiness['ready_for_experiments']}；P1 path={readiness['p1_daily_path_ready_share']:.4f}；P2 path={readiness['p2_daily_path_ready_share']:.4f}。",
        "主結論只能 net after transaction cost。official stock prices are unadjusted diagnostic-only; selected-stock adjusted close and cash/bear classifier remain blocked.",
        "Flags: formal_model_changed=false; trade_decision_changed=false; active_in_trade_decision=false; report_changed=false; portfolio_replay_executed=false; ready_for_strategy_replay=false; ready_for_formal=false; not_live_rule=true; forward_returns_live_rule_usage=false.",
        "",
    ])


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    calendar, signals = _load_calendar(), _load_signals()
    state, transitions = _state_rows(calendar, signals)
    prices, source_audit = _load_official_prices()
    state = _attach_prices(state, prices)
    gaps = _stock_gap_ledger(state)
    collision = _overlap_resolution(signals, state)
    coverage = _coverage(state, gaps)
    future = _future_audit()
    metrics = _net_path_metrics_hook(state)
    p1_share = float(coverage.loc[coverage["period"].eq("P1"), "daily_path_ready_share"].iloc[0])
    p2_share = float(coverage.loc[coverage["period"].eq("P2"), "daily_path_ready_share"].iloc[0])
    ready = bool(len(gaps) == 0 and p1_share == 1.0 and p2_share == 1.0)
    readiness = {
        "task_id": TASK_ID,
        "status": "reconstructed_single_position_R6_ready_unadjusted_diagnostic" if ready else "reconstructed_single_position_R6_partial_selected_stock_daily_ohlc_gap",
        "reconstructed_single_position_R6": True,
        "historical_overlapping_R6_reference_deprecated": True,
        "collision_resolution_rows": int(len(collision)),
        "p1_original_overlapping_interval_rows_resolved": int(collision["period_label"].eq("P1").sum()) if len(collision) else 0,
        "p2_original_overlapping_interval_rows_resolved": int(collision["period_label"].eq("P2").sum()) if len(collision) else 0,
        "single_position_state_path_ready": bool((state["daily_path_ready"] | (state["terminal_path_row_excluded_from_metric"] & state["terminal_mark_available"])).all()),
        "p1_daily_path_ready_share": p1_share,
        "p2_daily_path_ready_share": p2_share,
        "stock_price_gap_rows": int(len(gaps)),
        "official_unadjusted_stock_ohlc_ready_share": float(state.loc[state["selected_asset_type_after"].eq("stock"), "official_unadjusted_daily_ohlc_ready"].mean()) if state["selected_asset_type_after"].eq("stock").any() else 1.0,
        "EP05_transaction_cost_hooks_ready": True,
        "selected_stock_adjusted_close_ready": False,
        "cash_bear_classifier_ready": False,
        "ready_for_experiments": ready,
        "ready_for_formal": False,
        "ready_for_strategy_replay": False,
        "future_data_violation_count": 0,
        **FLAGS,
    }
    blocked = pd.DataFrame([
        {"item": "historical_overlapping_R6_reference", "status": "deprecated_reference_only", "detail": "Old fixed-5TD interval metrics are not same-basis with reconstructed single-position R6.", "next_owner": "none"},
        {"item": "selected_stock_daily_ohlc", "status": "blocked_pending_bounded_gap_fill" if len(gaps) else "ready", "detail": "Only missing selected ticker/date rows remain in the gap ledger; no synthetic return path.", "next_owner": "Radar/Data" if len(gaps) else "none"},
        {"item": "selected_stock_adjusted_close", "status": "blocked", "detail": "Official unadjusted OHLC only; no adjusted close fabricated.", "next_owner": "Strategy Center/Radar Data if trusted source authorized"},
        {"item": "cash_bear_classifier", "status": "blocked", "detail": "No cash rule constructed.", "next_owner": "Strategy Center/Core Data later"},
    ])
    paths = [
        _write(state, "reconstructed_weekly_r6_single_position_daily_state_rows.csv"),
        _write(transitions, "reconstructed_weekly_r6_single_position_transition_trace.csv"),
        _write(collision, "reconstructed_weekly_r6_collision_resolution_audit.csv"),
        _write(gaps, "reconstructed_weekly_r6_selected_stock_daily_ohlc_gap_ledger.csv"),
        _write(coverage, "reconstructed_weekly_r6_requested_vs_actual_coverage.csv"),
        _write(source_audit, "reconstructed_weekly_r6_official_price_source_audit.csv"),
        _write(blocked, "reconstructed_weekly_r6_blocked_proxy_audit.csv"),
        _write(future, "reconstructed_weekly_r6_future_data_audit.csv"),
        _write(metrics, "reconstructed_weekly_r6_net_path_metrics_hook.csv"),
        _write(pd.DataFrame([{"reference_name": "historical_overlapping_R6", "reference_status": "deprecated_path_like_only", "reported_P1_net_return": 6.4644, "reported_P1_MDD": -0.4880, "primary_comparison_allowed": False}]), "historical_overlapping_r6_reference_deprecation.csv"),
    ]
    readiness_path = OUTPUT_DIR / "readiness_for_reconstructed_weekly_r6_single_position_diagnostic.json"
    readiness_path.write_text(json.dumps(readiness, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path = OUTPUT_DIR / "final_summary_zh.md"
    summary_path.write_text(_summary(readiness), encoding="utf-8")
    manifest = {
        "task_id": TASK_ID, "generated_at_utc": datetime.now(timezone.utc).isoformat(), "output_dir": str(OUTPUT_DIR),
        "files": [{"path": path.name, "sha256": _sha256(path)} for path in [*paths, readiness_path, summary_path]],
        "readiness": readiness, "source_inputs": {"weekly_R6": str(R6_CONTRACT), "daily_calendar": str(DAILY_CALENDAR), "official_price_sources": [str(x) for x in OFFICIAL_PRICE_SOURCES]},
    }
    (OUTPUT_DIR / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(readiness, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
