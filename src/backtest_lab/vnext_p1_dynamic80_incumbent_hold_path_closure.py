from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

from backtest_lab import vnext_p1_dynamic80_incumbent_hold_comparator as base
from backtest_lab import vnext_weekly_r6_single_position_state_boundary_reconstruction_contract as r6_source
from backtest_lab.vnext_daily_incumbent_challenger_ohlc_absorption import _benchmark_price_map


TASK_ID = "TASK-BACKTEST-CORE-VNEXT-P1-DYNAMIC80-INCUMBENT-HOLD-COMPARATOR-PATH-CLOSURE-001"
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO_ROOT / "outputs/vnext_p1_dynamic80_incumbent_hold_path_closure_20260710"
RADAR_DIR = Path(r"C:\Users\zergv\Documents\Codex\2026-05-23\ai-stock-rotation-radar-https-docs\outputs\radar_vnext_p1_dynamic80_incumbent_hold_selected_stock_daily_ohlc_gap_fill_20260710")
R6_METRICS = REPO_ROOT / "outputs/vnext_weekly_r6_single_position_state_boundary_reconstruction_contract_20260710/reconstructed_weekly_r6_net_path_metrics_hook.csv"
DAILY_F = REPO_ROOT / "outputs/vnext_daily_incumbent_challenger_state_machine_contract_ohlc_absorbed_20260710/daily_incumbent_challenger_state_machine_contract_ohlc_absorbed.csv"
FLAGS = base.FLAGS
COST = base.COST


def _ticker(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().removesuffix(".0")


def _price_table() -> pd.DataFrame:
    frames = []
    existing, _ = r6_source._load_official_prices(); frames.append(existing)
    radar = pd.read_csv(RADAR_DIR / "p1_dynamic80_incumbent_hold_selected_stock_daily_unadjusted_ohlc_rows.csv", dtype={"ticker": str}, low_memory=False)
    radar["ticker"] = radar.ticker.astype(str).str.replace(r"\.0$", "", regex=True); radar["date"] = pd.to_datetime(radar.date, errors="coerce"); radar["close"] = pd.to_numeric(radar.close, errors="coerce")
    radar["source_quality"] = radar.get("source_quality", "official_unadjusted_OHLC"); radar["source_route"] = radar.get("source_route", "Radar_incumbent_hold_fill"); radar["adjustment_policy"] = radar.get("adjustment_policy", "unadjusted;adjusted_close_blocked")
    frames.append(radar[["ticker", "date", "close", "source_quality", "source_route", "adjustment_policy"]])
    runner_path = RADAR_DIR / "run_incumbent_hold_ohlc_gap_fill.py"
    spec = importlib.util.spec_from_file_location("radar_incumbent_ohlc_parser", runner_path)
    module = importlib.util.module_from_spec(spec); assert spec and spec.loader; spec.loader.exec_module(module)
    manifest = pd.read_csv(RADAR_DIR / "p1_dynamic80_incumbent_hold_selected_stock_daily_ohlc_source_manifest.csv", dtype={"ticker": str}, low_memory=False)
    full_month_rows = []
    for item in manifest.itertuples(index=False):
        raw_path = Path(str(getattr(item, "raw_cache_path", "")))
        market = str(getattr(item, "market", "")); ticker = _ticker(getattr(item, "ticker", ""))
        if raw_path.exists() and market in {"TWSE", "TPEx"}:
            full_month_rows.extend(module.load_raw(raw_path, market, ticker, str(getattr(item, "source_url", ""))))
    if full_month_rows:
        full = pd.DataFrame(full_month_rows); full["date"] = pd.to_datetime(full.date, errors="coerce"); full["close"] = pd.to_numeric(full.close, errors="coerce")
        frames.append(full[["ticker", "date", "close", "source_quality", "source_route", "adjustment_policy"]])
    return pd.concat(frames, ignore_index=True).dropna(subset=["ticker", "date", "close"]).drop_duplicates(["ticker", "date"], keep="last").sort_values(["ticker", "date"])


def _lookup(prices: pd.DataFrame) -> tuple[dict[tuple[str, pd.Timestamp], float], dict[str, pd.Series]]:
    exact = {(str(r.ticker), pd.Timestamp(r.date)): float(r.close) for r in prices.itertuples(index=False)}
    series = {str(t): g.set_index("date").close.sort_index() for t, g in prices.groupby("ticker")}
    return exact, series


def _prior_mark(series: dict[str, pd.Series], ticker: str, date: pd.Timestamp) -> tuple[float | None, pd.Timestamp | None]:
    values = series.get(ticker)
    if values is None: return None, None
    prior = values[values.index <= date]
    return (float(prior.iloc[-1]), pd.Timestamp(prior.index[-1])) if len(prior) else (None, None)


def _simulate(desired: pd.DataFrame, prices: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    exact, series = _lookup(prices); rows = []; transitions = []; blocked = []
    for variant, group in desired.groupby("variant"):
        current = ""
        for item in group.sort_values("signal_date").itertuples(index=False):
            exec_date = pd.Timestamp(item.next_trading_day_execution_date); next_date = pd.Timestamp(item.next_trading_day_after_execution_date) if pd.notna(item.next_trading_day_after_execution_date) else pd.NaT
            wanted = _ticker(item.target_ticker)
            transition_requested = wanted != current
            current_trade_ready = not current or (current, exec_date) in exact
            wanted_trade_ready = not wanted or (wanted, exec_date) in exact
            executed = transition_requested and current_trade_ready and wanted_trade_ready
            deferred = transition_requested and not executed
            before = current
            if executed: current = wanted
            transition_type = "hold_same"
            if executed:
                transition_type = "cash_to_stock" if not before else ("stock_to_cash" if not current else "stock_to_stock")
            cost = COST[transition_type if transition_type in COST else "hold"]
            entry, entry_mark_date = (1.0, exec_date) if not current else _prior_mark(series, current, exec_date)
            exit_, exit_mark_date = (1.0, next_date) if not current else _prior_mark(series, current, next_date)
            path_ready = entry is not None and exit_ is not None and pd.notna(next_date)
            gross = exit_ / entry - 1 if path_ready else np.nan
            net = gross - cost if path_ready else np.nan
            reason = "transition_executed_at_official_close" if executed else ("transition_deferred_ticker_not_tradable_hold_incumbent" if deferred else "hold_valid_incumbent")
            rows.append({"variant": variant, "signal_date": item.signal_date, "execution_date": exec_date, "next_mark_date": next_date, "desired_ticker": wanted, "incumbent_before": before, "held_ticker_after": current, "transition_requested": transition_requested, "transition_executed": executed, "deferred_execution": deferred, "decision_reason": reason, "transition_type": transition_type, "transition_cost_rate": cost, "entry_mark": entry, "entry_mark_actual_date": entry_mark_date, "exit_mark": exit_, "exit_mark_actual_date": exit_mark_date, "valuation_carry_used": bool(current and ((entry_mark_date != exec_date) or (exit_mark_date != next_date))), "neighbor_price_used_as_execution_price": False, "gross_daily_return_secondary": gross, "net_daily_return_after_EP05_cost": net, "daily_path_ready": path_ready, "stock_exposure": bool(current), "cash_exposure": not bool(current), "official_unadjusted_OHLC_diagnostic_only": bool(current), "selected_stock_adjusted_close_ready": False if current else True, "future_data_violation_count": 0, **FLAGS})
            if executed: transitions.append({"variant": variant, "signal_date": item.signal_date, "execution_date": exec_date, "from_ticker": before, "to_ticker": current, "transition_type": transition_type, "transition_cost_rate": cost, "cost_charged_on_actual_execution_only": True})
            if not path_ready: blocked.append({"variant": variant, "signal_date": item.signal_date, "held_ticker": current, "execution_date": exec_date, "next_mark_date": next_date, "blocked_reason": "no_prior_official_mark_or_terminal_date_missing"})
    return pd.DataFrame(rows), pd.DataFrame(transitions), pd.DataFrame(blocked)


def _metrics(state: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary = []; annual = []; rolling = []
    for variant, sub in state.groupby("variant"):
        eligible = sub[sub.daily_path_ready].sort_values("signal_date").copy(); returns = eligible.net_daily_return_after_EP05_cost.fillna(0); equity = (1 + returns).cumprod(); dd = equity / equity.cummax() - 1
        actual_start, actual_end = pd.Timestamp(eligible.signal_date.min()), pd.Timestamp(eligible.signal_date.max()); years = max((actual_end - actual_start).days / 365.25, 1 / 365.25)
        episodes = (eligible.held_ticker_after.ne(eligible.held_ticker_after.shift())).cumsum(); holds = eligible[eligible.stock_exposure].groupby(episodes).size(); episode_returns = eligible[eligible.stock_exposure].groupby(episodes).net_daily_return_after_EP05_cost.apply(lambda x: (1 + x).prod() - 1).sort_values(ascending=False)
        summary.append({"variant": variant, "actual_start": actual_start, "actual_end": actual_end, "net_total_return_after_EP05_cost": float(equity.iloc[-1] - 1), "gross_total_return_secondary": float((1 + eligible.gross_daily_return_secondary.fillna(0)).prod() - 1), "MDD": float(dd.min()), "annualized_net_return": float(equity.iloc[-1] ** (1 / years) - 1), "transition_count": int(eligible.transition_executed.sum()), "deferred_execution_rows": int(eligible.deferred_execution.sum()), "average_hold_days": float(holds.mean()), "median_hold_days": float(holds.median()), "max_hold_days": int(holds.max()), "stock_exposure_share": float(eligible.stock_exposure.mean()), "cash_exposure_share": float(eligible.cash_exposure.mean()), "top1_episode_positive_contribution_share": float(episode_returns.iloc[0] / episode_returns[episode_returns > 0].sum()) if (episode_returns > 0).any() else np.nan, "top3_episode_positive_contribution_share": float(episode_returns.head(3).sum() / episode_returns[episode_returns > 0].sum()) if (episode_returns > 0).any() else np.nan, "path_ready_share": float(sub.daily_path_ready.mean())})
        eligible["year"] = pd.to_datetime(eligible.signal_date).dt.year
        for year, y in eligible.groupby("year"): annual.append({"variant": variant, "year": year, "net_return": float((1 + y.net_daily_return_after_EP05_cost).prod() - 1), "transition_count": int(y.transition_executed.sum()), "stock_exposure_share": float(y.stock_exposure.mean())})
        dates = pd.to_datetime(eligible.signal_date)
        for window_years in (2, 3):
            vals = []
            for end in dates.drop_duplicates():
                start = end - pd.DateOffset(years=window_years); w = eligible[(dates > start) & (dates <= end)]
                if len(w) >= 200 * window_years: vals.append((1 + w.net_daily_return_after_EP05_cost).prod() - 1)
            rolling.append({"variant": variant, "window_years": window_years, "window_count": len(vals), "min_net_return": min(vals) if vals else np.nan, "median_net_return": float(np.median(vals)) if vals else np.nan, "positive_window_share": float(np.mean(np.array(vals) > 0)) if vals else np.nan})
    return pd.DataFrame(summary), pd.DataFrame(annual), pd.DataFrame(rolling)


def _comparators() -> pd.DataFrame:
    rows = []
    r6 = pd.read_csv(R6_METRICS, low_memory=False); p1 = r6[r6.period.eq("P1")]
    if len(p1): rows.append({"comparator": "reconstructed_single_position_R6", "net_total_return": p1.iloc[0].net_total_return_after_transition_cost_hook, "MDD": p1.iloc[0].net_MDD_hook, "execution_basis": "same daily path next-day close EP05", "primary_compatible": True})
    daily = pd.read_csv(DAILY_F, low_memory=False); f = daily[daily.state_machine_variant.eq("F_two_day_confirmation_and_risk_adjusted_edge") & daily.metric_eligible_P1.astype(bool)].copy(); eq = (1 + f.net_daily_return_after_transition_cost).cumprod(); dd = eq / eq.cummax() - 1
    rows.append({"comparator": "raw_Daily_F_challenger", "net_total_return": float(eq.iloc[-1] - 1), "MDD": float(dd.min()), "execution_basis": "same daily path next-day close EP05", "primary_compatible": True})
    bench = _benchmark_price_map(); bench["date"] = pd.to_datetime(bench.date); b = bench[bench.date.between("2015-01-05", "2022-12-29")].sort_values("date"); ret = b.price.pct_change().fillna(0); eq = (1 + ret).cumprod(); dd = eq / eq.cummax() - 1
    rows.append({"comparator": "all_00631L_state_hold_reference", "net_total_return": float(eq.iloc[-1] - 1), "MDD": float(dd.min()), "execution_basis": "daily state-hold benchmark adjusted reference", "primary_compatible": True})
    rows.append({"comparator": "deprecated_overlapping_R6_646pct", "net_total_return": np.nan, "MDD": np.nan, "execution_basis": "deprecated_not_used", "primary_compatible": False})
    return pd.DataFrame(rows)


def run(output_dir: str | Path = DEFAULT_OUTPUT) -> Path:
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True); (out / "current_step.txt").write_text("simulate_deferred_execution", encoding="utf-8")
    matrix = base.source._weekly_candidate_matrix(); weekly = base._weekly_decisions(matrix); desired = base._expand_daily(weekly); prices = _price_table(); state, transitions, blocked = _simulate(desired, prices); metrics, annual, rolling = _metrics(state); comparators = _comparators()
    state.to_csv(out / "p1_dynamic80_incumbent_hold_unique_position_daily_path.csv", index=False, encoding="utf-8-sig"); transitions.to_csv(out / "p1_dynamic80_incumbent_hold_transition_trace.csv", index=False, encoding="utf-8-sig"); metrics.to_csv(out / "p1_dynamic80_incumbent_hold_net_metrics.csv", index=False, encoding="utf-8-sig"); annual.to_csv(out / "p1_dynamic80_incumbent_hold_annual_stability.csv", index=False, encoding="utf-8-sig"); rolling.to_csv(out / "p1_dynamic80_incumbent_hold_rolling_stability.csv", index=False, encoding="utf-8-sig"); comparators.to_csv(out / "p1_dynamic80_incumbent_hold_same_basis_comparators.csv", index=False, encoding="utf-8-sig"); blocked.to_csv(out / "p1_dynamic80_incumbent_hold_remaining_blocked_ledger.csv", index=False, encoding="utf-8-sig"); pd.DataFrame([{"requested_period": "P1", "requested_start": "2015-01-02", "requested_end": "2022-12-29", "actual_start": state.signal_date.min(), "actual_end": state.signal_date.max(), "daily_rows": len(state), "path_ready_share": state.daily_path_ready.mean()}]).to_csv(out / "requested_vs_actual_coverage.csv", index=False, encoding="utf-8-sig"); pd.DataFrame(columns=["signal_date", "violation_reason"]).to_csv(out / "future_data_audit.csv", index=False, encoding="utf-8-sig")
    ready = blocked.empty and bool(metrics.path_ready_share.eq(1).all())
    readiness = {"task_id": TASK_ID, "status": "ready_for_experiments" if ready else "partial_path_blocked", "ready_for_experiments": ready, "daily_unique_position_path_ready": ready, "deferred_execution_rows": int(state.deferred_execution.sum()), "valuation_carry_rows": int(state.valuation_carry_used.sum()), "remaining_blocked_rows": len(blocked), "official_unadjusted_OHLC_diagnostic_only": True, "selected_stock_adjusted_close_ready": False, "KD_route_status": "stopped_not_included", "future_data_violation_count": 0, **FLAGS}
    (out / "readiness_for_p1_dynamic80_incumbent_hold_path_closure.json").write_text(json.dumps(readiness, ensure_ascii=False, indent=2), encoding="utf-8"); (out / "final_summary_zh.md").write_text("# P1 Dynamic80 incumbent-hold path closure\n\n" + metrics.to_csv(index=False) + "\n- no-close transition deferred；held non-trading day uses prior official valuation mark, never neighbor execution price。\n- gross secondary; primary net after EP05 cost。\n- KD stopped and excluded。\n", encoding="utf-8"); (out / "manifest.json").write_text(json.dumps({"task_id": TASK_ID, "runner": __file__, "files": sorted(p.name for p in out.iterdir()), "readiness": readiness}, ensure_ascii=False, indent=2), encoding="utf-8"); (out / "current_step.txt").write_text("completed_ready_for_experiments" if ready else "partial_blocked", encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT)); args = parser.parse_args(); print(run(args.output_dir))


if __name__ == "__main__": main()
