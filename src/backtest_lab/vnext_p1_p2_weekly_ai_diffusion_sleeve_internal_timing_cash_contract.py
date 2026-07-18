from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from backtest_lab import (
    vnext_p1_p2_ai_concentration_diffusion_weekly_switch_exact_nav_contract as base,
)
from backtest_lab import (
    vnext_p1_p2_strict_ai_diffusion_weekly_switch_fixed_bear_cash_extension_contract as close_index,
)


OUT = base.ROOT / "outputs/vnext_p1_p2_weekly_ai_diffusion_sleeve_internal_timing_cash_contract_20260718"
AI = base.STATE_AI
DIFFUSION = base.STATE_DIFFUSION
AI_SLEEVE = "fixed7_S10_CD10"
DIFFUSION_SLEEVE = "0050_signal_to_00631L_MA4_S7_MA10_S20_CD7"
RADAR_SLEEVE_TIMING_CLOSE_FILL = Path(
    r"C:\Users\zergv\Documents\Codex\2026-05-23"
    r"\ai-stock-rotation-radar-https-docs\outputs"
    r"\radar_vnext_p1_p2_sleeve_internal_timing_close_fill_20260718"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_official_close_index() -> pd.DataFrame:
    parts = [close_index.load_official_close_index()]
    patch_path = RADAR_SLEEVE_TIMING_CLOSE_FILL / "sleeve_internal_timing_exact_close_patch.csv"
    if patch_path.exists():
        patch = pd.read_csv(patch_path, dtype={"ticker": str})
        patch["date"] = pd.to_datetime(patch["date"]).dt.tz_localize(None)
        patch["close"] = pd.to_numeric(patch["close"], errors="coerce")
        patch["official_source_role"] = "radar_sleeve_internal_timing_close_fill"
        parts.append(patch[["ticker", "date", "market", "close", "source_quality", "source_url", "source_hash", "official_source_role"]])
    return pd.concat(parts, ignore_index=True).dropna(subset=["ticker", "date", "close"]).sort_values(["ticker", "date", "official_source_role"]).drop_duplicates(["ticker", "date"], keep="last")


def load_no_trade_lookup() -> dict[tuple[str, pd.Timestamp], dict]:
    lookup = base.load_no_trade_lookup()
    path = RADAR_SLEEVE_TIMING_CLOSE_FILL / "sleeve_internal_timing_exact_close_no_trade.csv"
    if path.exists():
        frame = pd.read_csv(path, dtype={"ticker": str})
        frame["date"] = pd.to_datetime(frame["date"]).dt.tz_localize(None)
        lookup.update({(str(row.ticker), pd.Timestamp(row.date)): row._asdict() for row in frame.itertuples(index=False)})
    return lookup


def build_prices() -> tuple[pd.DataFrame, pd.DataFrame]:
    attack = base.load_prices().copy()
    market = base.read_price("0050").sort_values("date").copy()
    market["MA4"] = market["adj_close"].rolling(4, min_periods=4).mean()
    market["MA10"] = market["adj_close"].rolling(10, min_periods=10).mean()
    market["slope7"] = market["adj_close"] - market["adj_close"].shift(6)
    market["slope20"] = market["adj_close"] - market["adj_close"].shift(19)
    market["diffusion_buy_signal"] = market["adj_close"].gt(market["MA4"]) & market["slope7"].gt(0)
    market["diffusion_sell_signal"] = market["adj_close"].lt(market["MA10"]) & market["slope20"].lt(0)
    return attack, market.set_index("date")


def weekly_strict() -> pd.DataFrame:
    return base.load_stage_a_variant("strict_no_bear").reset_index(drop=True)


def regime_for_next_day(weekly: pd.DataFrame, next_date: pd.Timestamp) -> tuple[str, pd.Timestamp | None]:
    return base.state_for_date(weekly, next_date)


def ai_candidate(day: pd.DataFrame, next_date: pd.Timestamp, last_sold: str | None, by_ticker: dict[str, pd.DataFrame]) -> tuple[str | None, str]:
    eligible = []
    for ticker in base.FIXED7:
        if ticker == last_sold or ticker not in day.index or next_date not in by_ticker[ticker].index:
            continue
        row = day.loc[ticker]
        if pd.notna(row.get("MA10")) and pd.notna(row.get("slope20")) and float(row["adj_close"]) > float(row["MA10"]) and float(row["slope20"]) > 0:
            denominator = float(row["adj_close"] - row["slope20"])
            eligible.append({
                "ticker": ticker,
                "normalized_slope20": float(row["slope20"] / denominator) if denominator else np.nan,
                "distance_above_ma": float(row["adj_close"] / row["MA10"] - 1),
            })
    if not eligible:
        return None, "no_fixed7_positive_entry_signal"
    target = pd.DataFrame(eligible).sort_values(["normalized_slope20", "distance_above_ma", "ticker"], ascending=[False, True, True]).iloc[0]
    return str(target["ticker"]), "fixed7_ma10_slope20_entry_signal"


def ai_exit(day: pd.DataFrame, ticker: str, execution_index: int | None, current_index: int) -> bool:
    if ticker not in day.index or execution_index is None or current_index - execution_index <= base.FIXED7_SIGNAL.cooldown:
        return False
    row = day.loc[ticker]
    return bool(pd.notna(row.get("MA20")) and pd.notna(row.get("slope20")) and float(row["adj_close"]) < float(row["MA20"]) and float(row["slope20"]) < 0)


def materialize_virtual_path(period: str, prices: pd.DataFrame, market: pd.DataFrame, weekly: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # The local analysis cache contains a few provider Saturday rows. They are
    # not Taiwan trading sessions and cannot be decision or execution dates.
    calendar = [date for date in base.calendar_for_period(prices, period) if pd.Timestamp(date).dayofweek < 5]
    by_date = {date: group.set_index("ticker") for date, group in prices.groupby("date")}
    by_ticker = {ticker: group.set_index("date") for ticker, group in prices.groupby("ticker")}
    index_by_date = {date: index for index, date in enumerate(calendar)}
    holding: str | None = None
    sleeve: str | None = None
    ai_last_sold: str | None = None
    ai_entry_index: int | None = None
    diffusion_last_sell_index: int | None = None
    diffusion_entry_index: int | None = None
    pending: dict | None = None
    daily, actions, holding_marks = [], [], []

    for current_index, date in enumerate(calendar):
        executed = "hold_cash" if holding is None else "hold_position"
        execution_reason = ""
        if pending is not None and pending["execution_date"] == date:
            old, target = pending["old_ticker"], pending["target_ticker"]
            if old is not None:
                if sleeve == AI_SLEEVE:
                    ai_last_sold = old
                    ai_entry_index = None
                else:
                    diffusion_last_sell_index = current_index
                    diffusion_entry_index = None
            holding = target
            sleeve = pending["target_sleeve"] if target is not None else None
            if target is not None:
                if sleeve == AI_SLEEVE:
                    ai_entry_index = current_index
                else:
                    diffusion_entry_index = current_index
            executed = pending["action"]
            execution_reason = pending["reason"]
            actions.append({**pending, "period": period, "executed": True})
            pending = None

        if holding is not None:
            mark = by_ticker.get(holding)
            ready = mark is not None and date in mark.index
            holding_marks.append({"period": period, "date": date, "ticker": holding, "sleeve": sleeve, "analysis_mark_ready": ready})

        next_date = calendar[current_index + 1] if current_index + 1 < len(calendar) else None
        day = by_date.get(date)
        next_regime, regime_week = regime_for_next_day(weekly, next_date) if next_date is not None else (DIFFUSION, None)
        desired_ticker, desired_sleeve, decision_reason = None, None, "no_positive_entry_signal"
        if next_date is not None and day is not None:
            if next_regime == AI:
                if holding is not None and sleeve == AI_SLEEVE and not ai_exit(day, holding, ai_entry_index, current_index):
                    desired_ticker, desired_sleeve, decision_reason = holding, AI_SLEEVE, "ai_hold_no_exit_signal"
                else:
                    desired_ticker, decision_reason = ai_candidate(day, next_date, ai_last_sold, by_ticker)
                    desired_sleeve = AI_SLEEVE if desired_ticker is not None else None
            else:
                if holding == "00631L" and sleeve == DIFFUSION_SLEEVE and date in market.index:
                    signal = market.loc[date]
                    unlocked = diffusion_entry_index is not None and current_index - diffusion_entry_index > 7
                    if not (bool(signal["diffusion_sell_signal"]) and unlocked):
                        desired_ticker, desired_sleeve, decision_reason = "00631L", DIFFUSION_SLEEVE, "diffusion_hold_no_exit_signal"
                    else:
                        decision_reason = "diffusion_exit_signal"
                elif date in market.index and bool(market.loc[date, "diffusion_buy_signal"]):
                    unlocked = diffusion_last_sell_index is None or current_index - diffusion_last_sell_index > 7
                    if unlocked:
                        desired_ticker, desired_sleeve, decision_reason = "00631L", DIFFUSION_SLEEVE, "0050_ma4_slope7_entry_signal"
                    else:
                        decision_reason = "diffusion_post_sell_cd7_lock"

        if next_date is not None and pending is None and (holding != desired_ticker or sleeve != desired_sleeve):
            if holding is not None and desired_ticker is not None:
                action = "atomic_switch"
            elif holding is not None:
                action = "exit_to_cash"
            else:
                action = "entry_from_cash"
            pending = {
                "decision_date": date,
                "execution_date": next_date,
                "old_ticker": holding,
                "old_sleeve": sleeve,
                "target_ticker": desired_ticker,
                "target_sleeve": desired_sleeve,
                "regime_decision_week": regime_week,
                "regime_effective_state": next_regime,
                "action": action,
                "reason": decision_reason if holding is None or desired_ticker is None else f"weekly_regime_priority__{decision_reason}",
            }

        daily.append({
            "period": period, "date": date, "holding_ticker": holding, "holding_sleeve": sleeve,
            "executed_action": executed, "execution_reason": execution_reason,
            "next_effective_regime": next_regime, "regime_decision_week": regime_week,
            "strict_weekly_state_reason_code": "strict_no_bear_weekly_selector_unchanged",
            "pending_action": pending["action"] if pending else "", "cash_due_to_no_positive_signal": holding is None,
            "actual_target_ticker": holding if holding is not None else "cash",
            "reference_variant_id": "strict_no_bear_baseline",
            "reference_join_period": period, "reference_join_date": date,
        })
    return pd.DataFrame(daily), pd.DataFrame(actions), pd.DataFrame(holding_marks)


def requirements(actions: pd.DataFrame, official_lookup: dict, no_trade_lookup: dict) -> pd.DataFrame:
    rows = []
    for action in actions.itertuples(index=False):
        for role, ticker in (("sell", action.old_ticker), ("buy", action.target_ticker)):
            if ticker is None or pd.isna(ticker):
                continue
            key = (str(ticker), pd.Timestamp(action.execution_date))
            source, no_trade = official_lookup.get(key), no_trade_lookup.get(key)
            rows.append({
                "period": action.period, "decision_date": action.decision_date, "actual_execution_date": action.execution_date,
                "role": role, "ticker": ticker, "action": action.action, "reason": action.reason,
                "official_raw_ready": bool(source), "official_no_trade_ready": bool(no_trade),
                "official_raw_close": source.get("close", np.nan) if source else np.nan,
                "source_hash": source.get("source_hash", "") if source else "",
            })
    return pd.DataFrame(rows)


def weekday_calendar(prices: pd.DataFrame, period: str) -> list[pd.Timestamp]:
    return [date for date in base.calendar_for_period(prices, period) if pd.Timestamp(date).dayofweek < 5]


def materialize_nav_authority(
    period: str,
    prices: pd.DataFrame,
    actions: pd.DataFrame,
    official_lookup: dict,
    no_trade_lookup: dict,
    slippage: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    calendar = weekday_calendar(prices, period)
    by_ticker = {ticker: group.set_index("date") for ticker, group in prices.groupby("ticker")}
    action_rows = actions.loc[actions["period"].eq(period)].copy()
    action_rows["execution_date"] = pd.to_datetime(action_rows["execution_date"])
    action_rows = action_rows.sort_values("execution_date").to_dict("records")
    pending_index = 0
    active: dict | None = None
    ticker: str | None = None
    nav = base.INITIAL_CAPITAL
    prior_mark: float | None = None
    daily_rows, transition_rows, audit_rows = [], [], []

    for date in calendar:
        nav_open = nav
        gross_return = 0.0
        if ticker is not None and prior_mark is not None:
            mark = float(by_ticker[ticker].loc[date, "adj_close"])
            gross_return = mark / prior_mark - 1.0
            nav *= 1.0 + gross_return
            prior_mark = mark
        if active is None and pending_index < len(action_rows) and pd.Timestamp(action_rows[pending_index]["execution_date"]) <= date:
            active = action_rows[pending_index]
        action = "hold" if ticker is not None else "cash"
        status = "not_due"
        transition_cost = 0.0
        actual_legs = []
        if active is not None:
            legs = [("sell", active.get("old_ticker")), ("buy", active.get("target_ticker"))]
            legs = [(side, str(value)) for side, value in legs if pd.notna(value) and value is not None]
            sources = [official_lookup.get((leg_ticker, date)) for _, leg_ticker in legs]
            no_trade = [no_trade_lookup.get((leg_ticker, date)) for _, leg_ticker in legs]
            if all(source is not None for source in sources):
                before = nav
                for side, leg_ticker in legs:
                    cost = nav * base.tax_rate(leg_ticker, side, slippage)
                    nav -= cost
                    transition_cost += cost
                    src = official_lookup[(leg_ticker, date)]
                    actual_legs.append({"role": side, "ticker": leg_ticker, "official_raw_close": src["close"], "source_hash": src.get("source_hash", "")})
                ticker = str(active["target_ticker"]) if pd.notna(active.get("target_ticker")) else None
                prior_mark = float(by_ticker[ticker].loc[date, "adj_close"]) if ticker is not None else None
                action = str(active["action"])
                status = "executed" if date == pd.Timestamp(active["execution_date"]) else "deferred_executed"
                transition_rows.append({**active, "period": period, "actual_execution_date": date, "execution_status": status, "NAV_before_transition": before, "NAV_after_transition": nav, "transition_cost": transition_cost, "legs_json": json.dumps(actual_legs, ensure_ascii=False)})
                active = None
                pending_index += 1
            elif any(item is not None for item in no_trade):
                action = "deferred_no_trade"
                status = "deferred_waiting_for_all_legs_same_tradable_date"
            else:
                raise RuntimeError(f"unexpected unresolved execution leg on {date.date()} for {active}")
        nav_close = nav
        daily_rows.append({"period": period, "date": date, "slippage_bp_per_side": int(round(slippage * 10000)), "NAV_open": nav_open, "NAV_close": nav_close, "net_daily_return": nav_close / nav_open - 1 if nav_open else np.nan, "gross_holding_return": gross_return, "held_ticker": ticker or "cash", "action": action, "execution_status": status, "transition_cost": transition_cost, "analysis_mark": prior_mark if ticker is not None else np.nan})
        audit_rows.append({"period": period, "date": date, "nav_chain_residual": nav_close - nav_open * (1 + gross_return) + transition_cost, "cross_asset_nominal_return_used": False, "same_asset_mark_return_used": ticker is not None and gross_return != 0})
    if active is not None or pending_index != len(action_rows):
        raise RuntimeError("unresolved pending action after calendar end")
    return pd.DataFrame(daily_rows), pd.DataFrame(transition_rows), pd.DataFrame(audit_rows)


def same_coverage_benchmark(period: str, prices: pd.DataFrame, slippage: float) -> pd.DataFrame:
    calendar = weekday_calendar(prices, period)
    series = prices.loc[prices["ticker"].eq("00631L")].set_index("date").reindex(calendar)["adj_close"].astype(float)
    nav = base.INITIAL_CAPITAL * (1 - base.tax_rate("00631L", "buy", slippage)) * series / series.iloc[0]
    nav.iloc[-1] *= 1 - base.tax_rate("00631L", "sell", slippage)
    return pd.DataFrame({"period": period, "date": nav.index, "benchmark": "00631L_buyhold_same_coverage", "slippage_bp_per_side": int(round(slippage * 10000)), "NAV_open": nav.shift(1).fillna(base.INITIAL_CAPITAL), "NAV_close": nav.values})


def strict_reference_same_coverage(period: str, dates: pd.Series, slippage_bp: int) -> pd.DataFrame:
    path = close_index.OUT / "strict_bear_cash_corrected_NAV_daily_wealth_ledger.csv.gz"
    source = pd.read_csv(path, parse_dates=["date"])
    source = source.loc[(source["variant_id"].eq("strict_no_bear_baseline")) & (source["period"].eq(period)) & (source["slippage_bp_per_side"].eq(slippage_bp))].copy()
    source = source.set_index("date").reindex(pd.DatetimeIndex(dates))
    return pd.DataFrame({"period": period, "date": dates, "benchmark": "strict_no_bear_forced_hold_reference", "slippage_bp_per_side": slippage_bp, "NAV_close": source["NAV_close"].values, "reference_available": source["NAV_close"].notna().values})


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "current_step.txt").write_text("running_contract_readiness\n", encoding="utf-8")
    prices, market = build_prices()
    weekly = weekly_strict()
    official_lookup = base.build_source_lookup(load_official_close_index())
    no_trade_lookup = load_no_trade_lookup()
    daily_parts, action_parts, mark_parts = [], [], []
    for period in base.PERIODS:
        daily, actions, marks = materialize_virtual_path(period, prices, market, weekly)
        daily_parts.append(daily)
        action_parts.append(actions)
        mark_parts.append(marks)
    daily = pd.concat(daily_parts, ignore_index=True)
    actions = pd.concat(action_parts, ignore_index=True)
    marks = pd.concat(mark_parts, ignore_index=True)
    execution = requirements(actions, official_lookup, no_trade_lookup)
    missing = execution.loc[~(execution["official_raw_ready"] | execution["official_no_trade_ready"])].copy()
    union = missing[["ticker", "actual_execution_date"]].drop_duplicates().rename(columns={"actual_execution_date": "date"}).sort_values(["ticker", "date"])
    union["authorized_source_scope"] = "exact official raw close for fixed sleeve timing execution only"
    union["network_family_allowed"] = "official close/OHLC only; no market, bear, or other data family"
    per_period = []
    for period in base.PERIODS:
        req = execution.loc[execution["period"].eq(period)]
        held = marks.loc[marks["period"].eq(period)]
        per_period.append({"period": period, "execution_requirement_rows": len(req), "execution_missing_rows": len(req.loc[~(req["official_raw_ready"] | req["official_no_trade_ready"])]), "holding_mark_rows": len(held), "holding_mark_blocked_rows": int((~held["analysis_mark_ready"]).sum()), "exact_path_ready": bool((req["official_raw_ready"] | req["official_no_trade_ready"]).all() and held["analysis_mark_ready"].all())})
    readiness_frame = pd.DataFrame(per_period)
    ready = bool(readiness_frame["exact_path_ready"].all())
    nav_parts, transition_parts, nav_audit_parts, benchmark_parts, strict_reference_parts = [], [], [], [], []
    for slippage in base.SLIPPAGE_SENSITIVITY:
        slippage_bp = int(round(slippage * 10000))
        for period in base.PERIODS:
            nav_daily, nav_transitions, nav_audit = materialize_nav_authority(period, prices, actions, official_lookup, no_trade_lookup, slippage)
            nav_parts.append(nav_daily)
            transition_parts.append(nav_transitions)
            nav_audit_parts.append(nav_audit)
            benchmark_parts.append(same_coverage_benchmark(period, prices, slippage))
            strict_reference_parts.append(strict_reference_same_coverage(period, nav_daily["date"], slippage_bp))
    nav_daily_frame = pd.concat(nav_parts, ignore_index=True)
    nav_transition_frame = pd.concat(transition_parts, ignore_index=True)
    nav_audit_frame = pd.concat(nav_audit_parts, ignore_index=True)
    benchmark_frame = pd.concat(benchmark_parts, ignore_index=True)
    strict_reference_frame = pd.concat(strict_reference_parts, ignore_index=True)
    coverage = []
    for period, (start, end) in base.PERIODS.items():
        rows = daily.loc[daily["period"].eq(period)]
        coverage.append({"period": period, "requested_start": start, "requested_end": end, "actual_start": rows["date"].min(), "actual_end": rows["date"].max(), "actual_trading_days": len(rows)})
    policy = {"task_id": "TASK-BACKTEST-CORE-VNEXT-P1-P2-WEEKLY-AI-DIFFUSION-SLEEVE-INTERNAL-TIMING-CASH-CONTRACT-001", "diagnostic_only": True, "regime_selector": "strict_no_bear unchanged", "ai_sleeve": "fixed7 MA10+slope20 entry / MA20+slope20 exit / CD10", "diffusion_sleeve": "0050 MA4+slope7 entry -> 00631L / MA10+slope20 exit / CD7", "bear_classifier_used": False, "no_fallback": True, "no_grid": True, "formal_model_changed": False, "trade_decision_changed": False, "active_in_trade_decision": False, "report_changed": False, "not_live_rule": True, "future_data_violation_count": 0}
    daily.to_csv(OUT / "sleeve_internal_timing_daily_target_trace.csv.gz", index=False, compression="gzip")
    actions.to_csv(OUT / "sleeve_internal_timing_unique_action_ledger.csv", index=False, encoding="utf-8-sig")
    execution.to_csv(OUT / "sleeve_internal_timing_execution_requirement_ledger.csv", index=False, encoding="utf-8-sig")
    marks.to_csv(OUT / "sleeve_internal_timing_holding_mark_audit.csv.gz", index=False, compression="gzip")
    nav_daily_frame.to_csv(OUT / "sleeve_internal_timing_corrected_NAV_daily_wealth_ledger.csv.gz", index=False, compression="gzip")
    nav_transition_frame.to_csv(OUT / "sleeve_internal_timing_NAV_transition_ledger.csv", index=False, encoding="utf-8-sig")
    nav_audit_frame.to_csv(OUT / "sleeve_internal_timing_NAV_reconciliation.csv", index=False, encoding="utf-8-sig")
    benchmark_frame.to_csv(OUT / "sleeve_internal_timing_same_coverage_00631L_benchmark_daily_ledger.csv.gz", index=False, compression="gzip")
    strict_reference_frame.to_csv(OUT / "sleeve_internal_timing_same_coverage_strict_no_bear_reference_daily_ledger.csv.gz", index=False, compression="gzip")
    union.to_csv(OUT / "sleeve_internal_timing_path_independent_close_gap_union.csv", index=False, encoding="utf-8-sig")
    readiness_frame.to_csv(OUT / "sleeve_internal_timing_per_period_readiness.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(coverage).to_csv(OUT / "requested_vs_actual_coverage.csv", index=False, encoding="utf-8-sig")
    reference = daily[["period", "date", "reference_variant_id", "reference_join_period", "reference_join_date"]].copy()
    reference["reference_nav_authority"] = str(close_index.OUT / "strict_bear_cash_corrected_NAV_daily_wealth_ledger.csv.gz")
    reference["reference_join_slippage_bp_per_side"] = "5|10|20"
    reference.to_csv(OUT / "strict_no_bear_reference_join_keys.csv", index=False, encoding="utf-8-sig")
    (OUT / "sleeve_internal_timing_policy.json").write_text(json.dumps(policy, ensure_ascii=False, indent=2), encoding="utf-8")
    nav_reconciliation_pass = bool(np.isclose(nav_audit_frame["nav_chain_residual"], 0.0, atol=1e-6).all())
    reference_complete = bool(strict_reference_frame["reference_available"].all())
    ready = bool(ready and nav_reconciliation_pass and reference_complete)
    readiness = {"ready_for_experiments": ready, "execution_missing_rows": len(missing), "execution_missing_unique_keys": len(union), "holding_mark_blocked_rows": int((~marks["analysis_mark_ready"]).sum()), "nav_reconciliation_pass": nav_reconciliation_pass, "strict_no_bear_reference_complete": reference_complete, "one_shot_path_independent_close_union_required": bool(not union.empty), "bear_classifier_used": False, "future_data_violation_count": 0, "may_be_used_to_reject_strategy": False}
    (OUT / "readiness_for_experiments.json").write_text(json.dumps(readiness, ensure_ascii=False, indent=2), encoding="utf-8")
    files = ["sleeve_internal_timing_daily_target_trace.csv.gz", "sleeve_internal_timing_unique_action_ledger.csv", "sleeve_internal_timing_execution_requirement_ledger.csv", "sleeve_internal_timing_holding_mark_audit.csv.gz", "sleeve_internal_timing_corrected_NAV_daily_wealth_ledger.csv.gz", "sleeve_internal_timing_NAV_transition_ledger.csv", "sleeve_internal_timing_NAV_reconciliation.csv", "sleeve_internal_timing_same_coverage_00631L_benchmark_daily_ledger.csv.gz", "sleeve_internal_timing_same_coverage_strict_no_bear_reference_daily_ledger.csv.gz", "sleeve_internal_timing_path_independent_close_gap_union.csv", "sleeve_internal_timing_per_period_readiness.csv", "requested_vs_actual_coverage.csv", "strict_no_bear_reference_join_keys.csv", "sleeve_internal_timing_policy.json", "readiness_for_experiments.json"]
    (OUT / "manifest.json").write_text(json.dumps({"files": [{"path": f, "sha256": sha256(OUT / f)} for f in files]}, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "final_summary_zh.md").write_text("\n".join(["# Weekly AI/diffusion sleeve internal timing cash contract", "", f"- ready_for_experiments：{ready}", f"- exact execution missing unique keys：{len(union)}", f"- holding mark blocked rows：{int((~marks['analysis_mark_ready']).sum())}", f"- NAV reconciliation pass：{nav_reconciliation_pass}", f"- strict_no_bear reference complete：{reference_complete}", "- cash 僅由當期 sleeve 無正向 entry signal 或既有 exit 產生；未使用 bear classifier。", "- 本包 materialize fixed corrected NAV authority；Core不作策略結論。"])+"\n", encoding="utf-8")
    (OUT / "current_step.txt").write_text("completed_ready_for_experiments\n" if ready else "completed_one_shot_close_union_required\n", encoding="utf-8")


if __name__ == "__main__":
    main()
