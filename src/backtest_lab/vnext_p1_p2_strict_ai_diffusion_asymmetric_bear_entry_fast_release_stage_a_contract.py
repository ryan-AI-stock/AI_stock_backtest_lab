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
    vnext_p1_p2_strict_ai_diffusion_weekly_switch_fixed_bear_cash_extension_contract as prior,
)


OUT = (
    base.ROOT
    / "outputs/vnext_p1_p2_strict_ai_diffusion_asymmetric_bear_entry_fast_release_stage_a_contract_20260718"
)
STATE_BEAR = "確認空頭"
ENTRY_CONFIRMATION_WEEKS = 2
RELEASE_CONFIRMATION_WEEKS = 2


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_weekly_inputs() -> pd.DataFrame:
    path = base.STAGE_A_OUT / "three_state_weekly_feature_and_state_matrix.csv.gz"
    frame = pd.read_csv(path)
    frame = frame.loc[frame["variant_id"].eq("strict_no_bear")].copy()
    frame["decision_week"] = pd.to_datetime(frame["decision_week"]).dt.tz_localize(None)
    return frame.drop_duplicates("decision_week", keep="last").sort_values("decision_week").reset_index(drop=True)


def add_release_features(weekly: pd.DataFrame) -> pd.DataFrame:
    market = base.read_price("0050").sort_values("date").copy()
    market["ma20_raw_close"] = market["close"].rolling(20, min_periods=20).mean()
    market["raw_close_slope20"] = market["close"] - market["close"].shift(19)
    market["release_close_above_ma20"] = market["close"].gt(market["ma20_raw_close"])
    market["release_slope20_positive"] = market["raw_close_slope20"].gt(0)
    market["release_candidate"] = (
        market["release_close_above_ma20"] & market["release_slope20_positive"]
    )
    market = market.set_index("date")

    rows = []
    for row in weekly.itertuples(index=False):
        key = pd.Timestamp(row.decision_week)
        available = market.loc[market.index <= key]
        values = available.iloc[-1].to_dict() if not available.empty else {}
        rows.append(
            {
                "decision_week": key,
                "market_close_date": available.index[-1] if not available.empty else pd.NaT,
                "market_raw_close": values.get("close", np.nan),
                "market_ma20_raw_close": values.get("ma20_raw_close", np.nan),
                "market_raw_close_slope20": values.get("raw_close_slope20", np.nan),
                "release_close_above_ma20": values.get("release_close_above_ma20", np.nan),
                "release_slope20_positive": values.get("release_slope20_positive", np.nan),
                "release_candidate": values.get("release_candidate", np.nan),
            }
        )
    return weekly.merge(pd.DataFrame(rows), on="decision_week", how="left", validate="one_to_one")


def strict_target(row: pd.Series) -> str:
    value = str(row.get("decision_state", base.STATE_DIFFUSION))
    return value if value in {base.STATE_AI, base.STATE_DIFFUSION} else base.STATE_DIFFUSION


def materialize_state_trace(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    state = strict_target(result.iloc[0]) if not result.empty else base.STATE_DIFFUSION
    entry_streak = 0
    release_streak = 0
    rows = []

    for row in result.itertuples(index=False):
        item = pd.Series(row._asdict())
        trend_dd10 = bool(
            float(item["market_close_vs_ma120"]) < 0
            and float(item["market_ma120_slope20"]) < 0
            and float(item["market_drawdown120"]) <= -0.10
        )
        breadth_bearish = bool(item.get("bear_breadth_pass", False) == 1)
        entry_candidate = trend_dd10 and breadth_bearish
        release_ready = bool(item.get("release_candidate", False))
        target = strict_target(item)
        action = "hold_current_state"
        reason = "strict_target_unchanged"

        if state != STATE_BEAR:
            entry_streak = entry_streak + 1 if entry_candidate else 0
            release_streak = 0
            if entry_streak >= ENTRY_CONFIRMATION_WEEKS:
                state = STATE_BEAR
                action = "enter_cash_next_trading_day"
                reason = "trend_dd10_and_pit_breadth_bearish_confirmed_2w"
                entry_streak = 0
            else:
                state = target
                reason = "strict_ai_diffusion_target"
        else:
            release_streak = release_streak + 1 if release_ready else 0
            entry_streak = 0
            if release_streak >= RELEASE_CONFIRMATION_WEEKS:
                state = target
                action = "release_cash_to_strict_target_next_trading_day"
                reason = "0050_close_above_ma20_and_slope20_positive_confirmed_2w"
                release_streak = 0
            else:
                reason = "bear_cash_waiting_for_fast_release_confirmation"

        rows.append(
            {
                **item.to_dict(),
                "trend_dd10_candidate": trend_dd10,
                "pit_breadth_bearish": breadth_bearish,
                "bear_entry_candidate": entry_candidate,
                "bear_entry_confirmation_streak": entry_streak,
                "bear_release_candidate": release_ready,
                "bear_release_confirmation_streak": release_streak,
                "strict_target_state": target,
                "asymmetric_decision_state": state,
                "state_action": action,
                "state_reason_code": reason,
            }
        )
    return pd.DataFrame(rows)


def next_market_date(calendar: pd.DatetimeIndex, decision_week: pd.Timestamp) -> pd.Timestamp | pd.NaT:
    future = calendar[calendar > decision_week]
    return future[0] if len(future) else pd.NaT


def materialize_episodes(trace: pd.DataFrame, calendar: pd.DatetimeIndex) -> tuple[pd.DataFrame, pd.DataFrame]:
    events = trace.loc[trace["state_action"].ne("hold_current_state")].copy()
    events["requested_execution_date"] = events["decision_week"].map(lambda value: next_market_date(calendar, pd.Timestamp(value)))
    events["execution_semantics"] = "stage_a_requested_next_market_trading_day_only_no_execution_readiness_required"

    bear = trace["asymmetric_decision_state"].eq(STATE_BEAR)
    starts = trace.index[bear & ~bear.shift(fill_value=False)]
    rows = []
    for episode_id, start in enumerate(starts, start=1):
        end = start
        while end + 1 < len(trace) and bool(bear.iloc[end + 1]):
            end += 1
        start_week = pd.Timestamp(trace.at[start, "decision_week"])
        end_week = pd.Timestamp(trace.at[end, "decision_week"])
        release_rows = trace.loc[(trace.index > end) & trace["state_action"].eq("release_cash_to_strict_target_next_trading_day")]
        release_week = pd.Timestamp(release_rows.iloc[0]["decision_week"]) if not release_rows.empty else pd.NaT
        cash_start = next_market_date(calendar, start_week)
        cash_end = next_market_date(calendar, release_week) if pd.notna(release_week) else pd.NaT
        if pd.notna(cash_start) and pd.notna(cash_end):
            cash_td = int(((calendar >= cash_start) & (calendar < cash_end)).sum())
        elif pd.notna(cash_start):
            cash_td = int((calendar >= cash_start).sum())
        else:
            cash_td = 0
        rows.append(
            {
                "episode_id": f"asymmetric_{episode_id:03d}",
                "entry_decision_week": start_week,
                "last_bear_decision_week": end_week,
                "release_decision_week": release_week,
                "cash_requested_start": cash_start,
                "cash_requested_end_exclusive": cash_end,
                "cash_trading_days": cash_td,
                "entry_reason_code": "trend_dd10_and_pit_breadth_bearish_confirmed_2w",
                "release_reason_code": "0050_close_above_ma20_and_slope20_positive_confirmed_2w" if pd.notna(release_week) else "unreleased_at_actual_end",
            }
        )
    return pd.DataFrame(rows), events


def old_trend_dd10_episodes(calendar: pd.DatetimeIndex) -> pd.DataFrame:
    weekly = prior.load_weekly_path("trend_dd10").reset_index(drop=True)
    bear = weekly["decision_state"].eq(STATE_BEAR)
    starts = weekly.index[bear & ~bear.shift(fill_value=False)]
    rows = []
    for episode_id, start in enumerate(starts, start=1):
        end = start
        while end + 1 < len(weekly) and bool(bear.iloc[end + 1]):
            end += 1
        start_week = pd.Timestamp(weekly.at[start, "decision_week"])
        end_week = pd.Timestamp(weekly.at[end, "decision_week"])
        cash_start = next_market_date(calendar, start_week)
        next_non_bear = weekly.loc[(weekly.index > end) & weekly["decision_state"].ne(STATE_BEAR)]
        release_week = pd.Timestamp(next_non_bear.iloc[0]["decision_week"]) if not next_non_bear.empty else pd.NaT
        cash_end = next_market_date(calendar, release_week) if pd.notna(release_week) else pd.NaT
        cash_td = int(((calendar >= cash_start) & (calendar < cash_end)).sum()) if pd.notna(cash_start) and pd.notna(cash_end) else 0
        rows.append({"episode_id": f"old_trend_dd10_{episode_id:03d}", "entry_decision_week": start_week, "last_bear_decision_week": end_week, "release_decision_week": release_week, "cash_trading_days": cash_td})
    return pd.DataFrame(rows)


def period_summary(episodes: pd.DataFrame, period_label: str, start: pd.Timestamp, end: pd.Timestamp) -> dict:
    intersect = episodes.loc[
        episodes["entry_decision_week"].le(end)
        & (episodes["release_decision_week"].isna() | episodes["release_decision_week"].ge(start))
    ]
    return {"geometry_period": period_label, "episode_count": int(len(intersect)), "cash_trading_days": int(intersect["cash_trading_days"].sum()) if not intersect.empty else 0}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "current_step.txt").write_text("running_state_supply_audit\n", encoding="utf-8")
    source = add_release_features(load_weekly_inputs())
    trace = materialize_state_trace(source)
    market_calendar = pd.DatetimeIndex(base.read_price("0050")["date"].sort_values().unique())
    episodes, actions = materialize_episodes(trace, market_calendar)
    old_episodes = old_trend_dd10_episodes(market_calendar)

    old_lock = old_episodes.loc[
        old_episodes["entry_decision_week"].le(pd.Timestamp("2016-01-25"))
        & (old_episodes["release_decision_week"].isna() | old_episodes["release_decision_week"].ge(pd.Timestamp("2018-01-19")))
    ]
    # The gate is geometric, not merely an exact-end-date comparison. Releasing
    # a few sessions before 2018-01-19 would still leave the same multi-year
    # cash lock and therefore cannot pass the intended fast-release audit.
    new_lock = episodes.loc[
        episodes["entry_decision_week"].le(pd.Timestamp("2016-01-25"))
        & (episodes["release_decision_week"].isna() | episodes["release_decision_week"].ge(pd.Timestamp("2018-01-01")))
    ]
    bear_2022 = episodes.loc[
        episodes["entry_decision_week"].le(pd.Timestamp("2022-12-30"))
        & (episodes["release_decision_week"].isna() | episodes["release_decision_week"].ge(pd.Timestamp("2022-01-01")))
    ]
    release_complete = trace[["market_raw_close", "market_ma20_raw_close", "market_raw_close_slope20"]].notna().all(axis=1)
    requested_start = min(start for start, _ in base.PERIODS.values())
    requested_end = max(end for _, end in base.PERIODS.values())
    actual_start = pd.Timestamp(trace["decision_week"].min())
    actual_end = pd.Timestamp(trace["decision_week"].max())
    gate = {
        "no_2016_2018_single_cash_lock": bool(new_lock.empty),
        "long_cash_lock_episode_count": int(len(new_lock)),
        "confirmed_bear_episode_in_2022": bool(not bear_2022.empty),
        "release_pit_complete_for_all_weekly_rows": bool(release_complete.all()),
    }
    gate["stage_a_supply_gate_pass"] = bool(all(gate.values()))

    comparison = pd.DataFrame(
        [
            {"series": "old_trend_dd10", **period_summary(old_episodes, "2015_2018", pd.Timestamp("2015-01-01"), pd.Timestamp("2018-12-31"))},
            {"series": "asymmetric_fast_release", **period_summary(episodes, "2015_2018", pd.Timestamp("2015-01-01"), pd.Timestamp("2018-12-31"))},
            {"series": "old_trend_dd10", **period_summary(old_episodes, "2022", pd.Timestamp("2022-01-01"), pd.Timestamp("2022-12-30"))},
            {"series": "asymmetric_fast_release", **period_summary(episodes, "2022", pd.Timestamp("2022-01-01"), pd.Timestamp("2022-12-30"))},
            {"series": "old_trend_dd10", **period_summary(old_episodes, "2025", pd.Timestamp("2025-01-01"), pd.Timestamp("2025-12-31"))},
            {"series": "asymmetric_fast_release", **period_summary(episodes, "2025", pd.Timestamp("2025-01-01"), pd.Timestamp("2025-12-31"))},
        ]
    )
    comparison = comparison.loc[:, ~comparison.columns.duplicated()]

    trace.to_csv(OUT / "asymmetric_bear_weekly_state_action_trace.csv.gz", index=False, compression="gzip")
    episodes.to_csv(OUT / "asymmetric_bear_episode_ledger.csv", index=False, encoding="utf-8-sig")
    actions.to_csv(OUT / "asymmetric_bear_requested_action_ledger.csv", index=False, encoding="utf-8-sig")
    old_episodes.to_csv(OUT / "old_trend_dd10_episode_geometry.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv(OUT / "asymmetric_vs_old_trend_dd10_episode_geometry.csv", index=False, encoding="utf-8-sig")
    policy = {
        "task_id": "TASK-BACKTEST-CORE-VNEXT-P1-P2-STRICT-AI-DIFFUSION-ASYMMETRIC-BEAR-ENTRY-FAST-RELEASE-STAGE-A-CONTRACT-001",
        "diagnostic_only": True,
        "performance_authorized": False,
        "future_outcome_read": False,
        "strict_ai_diffusion_switch_unchanged": True,
        "bear_entry": "trend_dd10 unchanged plus existing PIT bear_breadth_pass, two weekly confirmations",
        "bear_release": "0050 raw close > PIT MA20 and 20TD raw close slope > 0, two weekly confirmations",
        "execution_readiness_required": False,
        "no_grid": True,
        "formal_model_changed": False,
        "trade_decision_changed": False,
        "active_in_trade_decision": False,
        "report_changed": False,
        "not_live_rule": True,
        "future_data_violation_count": 0,
    }
    (OUT / "asymmetric_bear_policy.json").write_text(json.dumps(policy, ensure_ascii=False, indent=2), encoding="utf-8")
    readiness = {
        "requested_weekly_start": str(requested_start.date()),
        "requested_weekly_end": str(requested_end.date()),
        "actual_weekly_start": str(actual_start.date()),
        "actual_weekly_end": str(actual_end.date()),
        "weekly_rows": int(len(trace)),
        "pit_release_input_complete_rows": int(release_complete.sum()),
        "pit_release_input_blocked_rows": int((~release_complete).sum()),
        "execution_readiness_required": False,
        "ready_for_experiments": False,
        "ready_for_strategy_center_episode_review": bool(gate["stage_a_supply_gate_pass"]),
        "future_data_violation_count": 0,
        **gate,
    }
    (OUT / "readiness_for_strategy_center.json").write_text(json.dumps(readiness, ensure_ascii=False, indent=2), encoding="utf-8")
    files = [
        "asymmetric_bear_weekly_state_action_trace.csv.gz",
        "asymmetric_bear_episode_ledger.csv",
        "asymmetric_bear_requested_action_ledger.csv",
        "old_trend_dd10_episode_geometry.csv",
        "asymmetric_vs_old_trend_dd10_episode_geometry.csv",
        "asymmetric_bear_policy.json",
        "readiness_for_strategy_center.json",
    ]
    (OUT / "manifest.json").write_text(
        json.dumps({"files": [{"path": name, "sha256": sha256(OUT / name)} for name in files]}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary = [
        "# Strict AI/diffusion asymmetric bear entry fast release Stage A",
        "",
        f"- stage_a_supply_gate_pass：{gate['stage_a_supply_gate_pass']}",
        f"- no_2016_2018_single_cash_lock：{gate['no_2016_2018_single_cash_lock']}",
        f"- long_cash_lock_episode_count：{gate['long_cash_lock_episode_count']}",
        f"- confirmed_bear_episode_in_2022：{gate['confirmed_bear_episode_in_2022']}",
        f"- release PIT complete：{int(release_complete.sum())}/{len(trace)}",
        "- 本包僅為 state/supply/episode geometry audit；未讀 future outcome，未跑 NAV 或績效。",
    ]
    (OUT / "final_summary_zh.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    (OUT / "current_step.txt").write_text("completed_strategy_center_episode_review\n", encoding="utf-8")


if __name__ == "__main__":
    main()
