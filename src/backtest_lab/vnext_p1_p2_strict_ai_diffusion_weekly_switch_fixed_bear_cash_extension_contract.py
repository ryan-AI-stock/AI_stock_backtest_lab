from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from backtest_lab import (
    vnext_p1_p2_ai_concentration_diffusion_weekly_switch_exact_nav_contract as base,
)


OUT = (
    base.ROOT
    / "outputs/vnext_p1_p2_strict_ai_diffusion_weekly_switch_fixed_bear_cash_extension_contract_20260718"
)
RADAR_STRICT_BEAR_CASH_CLOSE_FILL = (
    Path(
        r"C:\Users\zergv\Documents\Codex\2026-05-23"
        r"\ai-stock-rotation-radar-https-docs\outputs"
        r"\radar_vnext_p1_p2_strict_ai_diffusion_bear_cash_close_fill_20260718"
    )
    / "strict_bear_cash_exact_close_patch.csv"
)
RADAR_STRICT_BEAR_CASH_RECHAIN_CLOSE_FILL = (
    Path(
        r"C:\Users\zergv\Documents\Codex\2026-05-23"
        r"\ai-stock-rotation-radar-https-docs\outputs"
        r"\radar_vnext_p1_p2_strict_ai_diffusion_bear_cash_rechain_close_fill_20260718"
    )
    / "strict_bear_cash_rechain_exact_close_patch.csv"
)
RADAR_STRICT_BEAR_CASH_INCREMENTAL_CLOSE_FILL = (
    Path(
        r"C:\Users\zergv\Documents\Codex\2026-05-23"
        r"\ai-stock-rotation-radar-https-docs\outputs"
        r"\radar_vnext_p1_p2_strict_bear_cash_incremental_close_fill_20260718"
    )
    / "strict_bear_cash_incremental_exact_close_patch.csv"
)

STATE_BEAR = "確認空頭"
RULES = ["strict_no_bear_baseline", "current_score", "trend_dd10", "crash_dd20"]
CONFIRMATION_WEEKS = 3
MINIMUM_HOLD_WEEKS = 6


def bear_candidate(row: pd.Series, rule_id: str) -> bool:
    below = float(row["market_close_vs_ma120"]) < 0
    slope_down = float(row["market_ma120_slope20"]) < 0
    drawdown = float(row["market_drawdown120"])
    if rule_id == "strict_no_bear_baseline":
        return False
    if rule_id == "current_score":
        return int(row["bear_score"]) >= 3
    if rule_id == "trend_dd10":
        return below and slope_down and drawdown <= -0.10
    if rule_id == "crash_dd20":
        return below and drawdown <= -0.20
    raise ValueError(rule_id)


def load_official_close_index() -> pd.DataFrame:
    official = base.load_official_close_index()
    parts = [official]
    for patch_path, role in (
        (RADAR_STRICT_BEAR_CASH_CLOSE_FILL, "radar_strict_bear_cash_bounded_close_fill"),
        (RADAR_STRICT_BEAR_CASH_RECHAIN_CLOSE_FILL, "radar_strict_bear_cash_rechain_close_fill"),
        (RADAR_STRICT_BEAR_CASH_INCREMENTAL_CLOSE_FILL, "radar_strict_bear_cash_incremental_close_fill"),
    ):
        if not patch_path.exists():
            continue
        patch = pd.read_csv(patch_path, dtype={"ticker": str})
        patch["date"] = pd.to_datetime(patch["date"]).dt.tz_localize(None)
        patch["close"] = pd.to_numeric(patch["close"], errors="coerce")
        patch["official_source_role"] = role
        parts.append(
            patch[
                [
                    "ticker",
                    "date",
                    "market",
                    "close",
                    "source_quality",
                    "source_url",
                    "source_hash",
                    "official_source_role",
                ]
            ].copy()
        )
    merged = pd.concat(parts, ignore_index=True).dropna(subset=["ticker", "date", "close"])
    return merged.sort_values(["ticker", "date", "official_source_role"]).drop_duplicates(
        ["ticker", "date"], keep="last"
    )


def load_weekly_path(rule_id: str) -> pd.DataFrame:
    path = base.STAGE_A_OUT / "three_state_weekly_feature_and_state_matrix.csv.gz"
    frame = pd.read_csv(path)
    # Use strict_no_bear feature rows as the authority for holiday-aligned weekly
    # keys and AI score inputs. The bear overlay only changes candidate_state.
    frame = frame.loc[frame["variant_id"].eq("strict_no_bear")].copy()
    frame["decision_week"] = pd.to_datetime(frame["decision_week"]).dt.tz_localize(None)
    frame = frame.drop_duplicates("decision_week", keep="last").sort_values("decision_week")

    current = base.STATE_DIFFUSION
    pending = current
    pending_count = 0
    weeks_in_state = 0
    candidates: list[str] = []
    decisions: list[str] = []
    for row in frame.itertuples(index=False):
        series = pd.Series(row._asdict())
        if bear_candidate(series, rule_id):
            candidate = STATE_BEAR
        elif int(series["ai_score"]) >= 5:
            candidate = base.STATE_AI
        else:
            candidate = base.STATE_DIFFUSION
        candidates.append(candidate)
        weeks_in_state += 1
        if candidate == current:
            pending = current
            pending_count = 0
        elif candidate == pending:
            pending_count += 1
        else:
            pending = candidate
            pending_count = 1
        if (
            candidate != current
            and pending_count >= CONFIRMATION_WEEKS
            and weeks_in_state >= MINIMUM_HOLD_WEEKS
        ):
            current = candidate
            weeks_in_state = 0
            pending = current
            pending_count = 0
        decisions.append(current)
    frame["bear_rule_id"] = rule_id
    frame["candidate_state"] = candidates
    frame["decision_state"] = decisions
    return frame


def state_for_date(weekly: pd.DataFrame, date: pd.Timestamp) -> tuple[str, pd.Timestamp | None]:
    eligible = weekly.loc[weekly["decision_week"].lt(date)]
    if eligible.empty:
        return base.STATE_DIFFUSION, None
    row = eligible.iloc[-1]
    state = str(row["decision_state"])
    if state not in {base.STATE_AI, base.STATE_DIFFUSION, STATE_BEAR}:
        state = base.STATE_DIFFUSION
    return state, pd.Timestamp(row["decision_week"])


def simulate_period(
    rule_id: str,
    period: str,
    prices: pd.DataFrame,
    official_lookup: dict[tuple[str, pd.Timestamp], dict],
    no_trade_lookup: dict[tuple[str, pd.Timestamp], dict],
    slippage: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    calendar = base.calendar_for_period(prices, period)
    weekly = load_weekly_path(rule_id)
    fixed7 = base.fixed7_target_path(prices, calendar).set_index("date")
    by_ticker = {ticker: group.set_index("date") for ticker, group in prices.groupby("ticker")}

    nav = base.INITIAL_CAPITAL
    current_ticker: str | None = None
    previous_date: pd.Timestamp | None = None
    rows: list[dict] = []
    actions: list[dict] = []
    requirements: list[dict] = []
    blockers: list[dict] = []

    for date in calendar:
        terminal_blocked = False
        nav_open = nav
        gross_holding_return = 0.0
        if current_ticker is not None and previous_date is not None:
            series = by_ticker[current_ticker]
            prior_rows = series.loc[series.index <= previous_date]
            current_rows = series.loc[series.index <= date]
            if prior_rows.empty or current_rows.empty:
                blockers.append(
                    {
                        "variant_id": rule_id,
                        "period": period,
                        "date": date,
                        "ticker": current_ticker,
                        "blocker_class": "missing_adjusted_holding_mark",
                        "blocked_reason": "event-aware adjusted holding mark missing before/current date",
                    }
                )
            else:
                previous_mark = float(prior_rows.iloc[-1]["adj_close"])
                current_mark = float(current_rows.iloc[-1]["adj_close"])
                gross_holding_return = current_mark / previous_mark - 1
                nav *= 1 + gross_holding_return

        state, state_decision_week = state_for_date(weekly, date)
        if state == base.STATE_AI:
            target_ticker = fixed7.loc[date, "fixed7_ticker"] if date in fixed7.index else None
            target_strategy = "fixed7_S10_CD10"
        elif state == STATE_BEAR:
            target_ticker = None
            target_strategy = "cash"
        else:
            target_ticker = "00631L"
            target_strategy = "00631L_buyhold"
        if pd.isna(target_ticker):
            target_ticker = None

        action = "hold" if current_ticker else "cash"
        action_reason = ""
        transition_cost = 0.0
        source_ready = True
        no_trade_deferred = False
        if target_ticker != current_ticker:
            action = (
                "entry"
                if current_ticker is None and target_ticker is not None
                else "exit_to_cash"
                if current_ticker is not None and target_ticker is None
                else "atomic_switch"
            )
            action_reason = "weekly_state_fixed_bear_cash_or_fixed7_target_change"
            legs = []
            if current_ticker is not None:
                legs.append(("sell", current_ticker))
            if target_ticker is not None:
                legs.append(("buy", str(target_ticker)))
            for side, leg_ticker in legs:
                src = official_lookup.get((leg_ticker, date), {})
                cache = (
                    by_ticker.get(leg_ticker, pd.DataFrame()).loc[[date]]
                    if leg_ticker in by_ticker and date in by_ticker[leg_ticker].index
                    else pd.DataFrame()
                )
                raw_close = float(cache.iloc[0]["close"]) if not cache.empty else np.nan
                no_trade = no_trade_lookup.get((leg_ticker, date), {})
                official_ready = bool(src)
                no_trade_ready = bool(no_trade)
                if not official_ready:
                    if no_trade_ready:
                        no_trade_deferred = True
                    else:
                        source_ready = False
                        blockers.append(
                            {
                                "variant_id": rule_id,
                                "period": period,
                                "date": date,
                                "ticker": leg_ticker,
                                "role": side,
                                "blocker_class": "missing_official_raw_execution_close",
                                "blocked_reason": "no accepted official raw close source for exact execution leg",
                            }
                        )
                requirements.append(
                    {
                        "variant_id": rule_id,
                        "period": period,
                        "decision_date": state_decision_week,
                        "actual_execution_date": date,
                        "role": side,
                        "ticker": leg_ticker,
                        "raw_close_cache": raw_close,
                        "official_raw_close": src.get("close", np.nan) if src else np.nan,
                        "official_market": src.get("market", "") if src else "",
                        "official_source_quality": src.get("source_quality", "") if src else "",
                        "official_source_hash": src.get("source_hash", "") if src else "",
                        "official_raw_ready": official_ready,
                        "official_no_trade_ready": no_trade_ready,
                        "official_no_trade_classification": no_trade.get("classification", "") if no_trade else "",
                        "official_no_trade_reason": no_trade.get("reason", "") if no_trade else "",
                    }
                )
            if source_ready and not no_trade_deferred:
                if current_ticker is not None:
                    cost = nav * base.tax_rate(current_ticker, "sell", slippage)
                    nav -= cost
                    transition_cost += cost
                if target_ticker is not None:
                    cost = nav * base.tax_rate(str(target_ticker), "buy", slippage)
                    nav -= cost
                    transition_cost += cost
                actions.append(
                    {
                        "variant_id": rule_id,
                        "period": period,
                        "execution_date": date,
                        "prior_ticker": current_ticker,
                        "target_ticker": target_ticker,
                        "action": action,
                        "action_reason": action_reason,
                        "state": state,
                        "state_decision_week": state_decision_week,
                        "transition_cost": transition_cost,
                        "slippage_bp_per_side": int(round(slippage * 10000)),
                        "official_raw_all_legs_ready": True,
                        "execution_status": "executed",
                    }
                )
                current_ticker = str(target_ticker) if target_ticker is not None else None
            elif no_trade_deferred:
                action = "deferred_no_trade"
                action_reason = "official_no_trade_atomic_transition_not_executed"
                actions.append(
                    {
                        "variant_id": rule_id,
                        "period": period,
                        "execution_date": date,
                        "prior_ticker": current_ticker,
                        "target_ticker": target_ticker,
                        "action": action,
                        "action_reason": action_reason,
                        "state": state,
                        "state_decision_week": state_decision_week,
                        "transition_cost": 0.0,
                        "slippage_bp_per_side": int(round(slippage * 10000)),
                        "official_raw_all_legs_ready": False,
                        "execution_status": "deferred_no_trade",
                    }
                )
            else:
                action = "blocked_no_transition"
                action_reason = "official_raw_execution_close_missing_atomic_transition_not_applied"
                terminal_blocked = True
                actions.append(
                    {
                        "variant_id": rule_id,
                        "period": period,
                        "execution_date": date,
                        "prior_ticker": current_ticker,
                        "target_ticker": target_ticker,
                        "action": action,
                        "action_reason": action_reason,
                        "state": state,
                        "state_decision_week": state_decision_week,
                        "transition_cost": 0.0,
                        "slippage_bp_per_side": int(round(slippage * 10000)),
                        "official_raw_all_legs_ready": False,
                        "execution_status": "blocked_missing_official_raw_execution_close",
                    }
                )

        rows.append(
            {
                "variant_id": rule_id,
                "period": period,
                "date": date,
                "NAV_open": nav_open,
                "NAV_close": nav,
                "net_daily_return": nav / nav_open - 1 if nav_open else np.nan,
                "gross_holding_return": gross_holding_return,
                "state": state,
                "state_decision_week": state_decision_week,
                "target_strategy": target_strategy,
                "target_ticker": target_ticker,
                "held_ticker": current_ticker,
                "action": action,
                "action_reason": action_reason,
                "transition_cost": transition_cost,
                "slippage_bp_per_side": int(round(slippage * 10000)),
            }
        )
        previous_date = date
        if terminal_blocked:
            break

    return (
        pd.DataFrame(rows),
        pd.DataFrame(actions),
        pd.DataFrame(requirements),
        pd.DataFrame(blockers),
    )


def materialize_execution_requirement_union(
    prices: pd.DataFrame,
    official_lookup: dict[tuple[str, pd.Timestamp], dict],
    no_trade_lookup: dict[tuple[str, pd.Timestamp], dict],
) -> pd.DataFrame:
    rows: list[dict] = []
    for rule_id in RULES:
        weekly = load_weekly_path(rule_id)
        for period in base.PERIODS:
            calendar = base.calendar_for_period(prices, period)
            fixed7 = base.fixed7_target_path(prices, calendar).set_index("date")
            current_ticker: str | None = None
            for date in calendar:
                state, state_decision_week = state_for_date(weekly, date)
                if state == base.STATE_AI:
                    target_ticker = fixed7.loc[date, "fixed7_ticker"] if date in fixed7.index else None
                    target_strategy = "fixed7_S10_CD10"
                elif state == STATE_BEAR:
                    target_ticker = None
                    target_strategy = "cash"
                else:
                    target_ticker = "00631L"
                    target_strategy = "00631L_buyhold"
                if pd.isna(target_ticker):
                    target_ticker = None
                if target_ticker == current_ticker:
                    continue
                action = (
                    "entry"
                    if current_ticker is None and target_ticker is not None
                    else "exit_to_cash"
                    if current_ticker is not None and target_ticker is None
                    else "atomic_switch"
                )
                legs = []
                if current_ticker is not None:
                    legs.append(("sell", current_ticker))
                if target_ticker is not None:
                    legs.append(("buy", str(target_ticker)))
                for side, leg_ticker in legs:
                    src = official_lookup.get((leg_ticker, date), {})
                    no_trade = no_trade_lookup.get((leg_ticker, date), {})
                    rows.append(
                        {
                            "variant_id": rule_id,
                            "period": period,
                            "decision_date": state_decision_week,
                            "actual_execution_date": date,
                            "role": side,
                            "ticker": leg_ticker,
                            "action": action,
                            "target_strategy": target_strategy,
                            "prior_ticker_before_virtual_transition": current_ticker,
                            "target_ticker_after_virtual_transition": target_ticker,
                            "official_raw_close": src.get("close", np.nan) if src else np.nan,
                            "official_market": src.get("market", "") if src else "",
                            "official_source_quality": src.get("source_quality", "") if src else "",
                            "official_source_hash": src.get("source_hash", "") if src else "",
                            "official_raw_ready": bool(src),
                            "official_no_trade_ready": bool(no_trade),
                            "official_no_trade_classification": no_trade.get("classification", "") if no_trade else "",
                            "requirement_discovery_policy": "path_independent_virtual_transition_for_source_authority_only",
                        }
                    )
                current_ticker = str(target_ticker) if target_ticker is not None else None
    return pd.DataFrame(rows)


def metric_row(frame: pd.DataFrame, benchmark: pd.DataFrame) -> dict:
    row = base.metric_row(frame, benchmark)
    counts = frame["state"].value_counts()
    row.update(
        {
            "bear_days": int(counts.get(STATE_BEAR, 0)),
            "ai_days": int(counts.get(base.STATE_AI, 0)),
            "diffusion_days": int(counts.get(base.STATE_DIFFUSION, 0)),
        }
    )
    return row


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "current_step.txt").write_text("running\n", encoding="utf-8")
    prices = base.load_prices()
    official_lookup = base.build_source_lookup(load_official_close_index())
    no_trade_lookup = base.load_no_trade_lookup()

    daily_parts = []
    action_parts = []
    requirement_parts = []
    blocker_parts = []
    benchmark_parts = []
    weekly_parts = []
    metric_rows = []
    for rule_id in RULES:
        weekly = load_weekly_path(rule_id)
        weekly_parts.append(weekly)
    for slippage in base.SLIPPAGE_SENSITIVITY:
        for period in base.PERIODS:
            benchmark = base.build_benchmark(period, prices, slippage)
            benchmark_parts.append(benchmark)
            for rule_id in RULES:
                daily, actions, requirements, blockers = simulate_period(
                    rule_id, period, prices, official_lookup, no_trade_lookup, slippage
                )
                daily_parts.append(daily)
                if not actions.empty:
                    action_parts.append(actions)
                if not requirements.empty:
                    requirement_parts.append(requirements)
                if not blockers.empty:
                    blocker_parts.append(blockers)
                metric_rows.append(metric_row(daily, benchmark))

    daily_frame = pd.concat(daily_parts, ignore_index=True)
    action_frame = pd.concat(action_parts, ignore_index=True) if action_parts else pd.DataFrame()
    requirement_frame = pd.concat(requirement_parts, ignore_index=True) if requirement_parts else pd.DataFrame()
    blocker_frame = pd.concat(blocker_parts, ignore_index=True) if blocker_parts else pd.DataFrame(
        columns=["variant_id", "period", "date", "ticker", "role", "blocker_class", "blocked_reason"]
    )
    benchmark_frame = pd.concat(benchmark_parts, ignore_index=True)
    weekly_frame = pd.concat(weekly_parts, ignore_index=True)
    metrics = pd.DataFrame(metric_rows)
    requirement_union = materialize_execution_requirement_union(prices, official_lookup, no_trade_lookup)
    union_missing = requirement_union.loc[
        ~(requirement_union["official_raw_ready"] | requirement_union["official_no_trade_ready"])
    ].copy()

    readiness_rows = []
    for (variant_id, period), group in requirement_frame.groupby(["variant_id", "period"], dropna=False):
        blockers = blocker_frame.loc[blocker_frame["variant_id"].eq(variant_id) & blocker_frame["period"].eq(period)]
        readiness_rows.append(
            {
                "variant_id": variant_id,
                "period": period,
                "execution_requirement_rows": int(len(group)),
                "official_raw_ready_rows": int(group["official_raw_ready"].sum()),
                "execution_blocker_rows": int(len(blockers)),
                "official_no_trade_deferred_rows": int(group["official_no_trade_ready"].sum()),
                "exact_path_ready": bool(blockers.empty and (group["official_raw_ready"] | group["official_no_trade_ready"]).all()),
            }
        )
    readiness_frame = pd.DataFrame(readiness_rows)
    ready_for_experiments = bool(
        not readiness_frame.empty
        and readiness_frame.loc[readiness_frame["period"].isin(["P1", "P2"]), "exact_path_ready"].all()
    )

    if not union_missing.empty:
        bounded = (
            union_missing.rename(columns={"actual_execution_date": "date"})[
                ["ticker", "date"]
            ]
            .drop_duplicates()
            .sort_values(["ticker", "date"])
        )
        bounded["blocker_class"] = "missing_official_raw_execution_close"
        bounded["blocked_reason"] = "no accepted official raw close source for exact execution leg in full requirement union"
        bounded["authorized_source_scope"] = "exact official raw close for transition execution leg only"
        bounded["network_family_allowed"] = "official close/OHLC only; no non-close family"
    else:
        bounded = pd.DataFrame(columns=["ticker", "date", "blocker_class", "blocked_reason", "authorized_source_scope", "network_family_allowed"])

    daily_frame.to_csv(OUT / "strict_bear_cash_corrected_NAV_daily_wealth_ledger.csv.gz", index=False, compression="gzip")
    action_frame.to_csv(OUT / "strict_bear_cash_unique_position_action_ledger.csv", index=False, encoding="utf-8-sig")
    requirement_frame.to_csv(OUT / "strict_bear_cash_execution_requirement_ledger.csv", index=False, encoding="utf-8-sig")
    requirement_union.to_csv(OUT / "strict_bear_cash_path_independent_execution_requirement_union.csv", index=False, encoding="utf-8-sig")
    blocker_frame.to_csv(OUT / "strict_bear_cash_blocked_ledger.csv", index=False, encoding="utf-8-sig")
    bounded.to_csv(OUT / "strict_bear_cash_bounded_official_raw_execution_gap_ledger.csv", index=False, encoding="utf-8-sig")
    benchmark_frame.to_csv(OUT / "strict_bear_cash_00631L_benchmark_daily_ledger.csv.gz", index=False, compression="gzip")
    metrics.to_csv(OUT / "strict_bear_cash_exact_nav_metrics_preview.csv", index=False, encoding="utf-8-sig")
    readiness_frame.to_csv(OUT / "strict_bear_cash_per_variant_readiness.csv", index=False, encoding="utf-8-sig")
    weekly_frame.to_csv(OUT / "strict_bear_cash_weekly_state_matrix.csv.gz", index=False, compression="gzip")

    policy = {
        "task_id": "TASK-BACKTEST-CORE-VNEXT-P1-P2-STRICT-AI-DIFFUSION-WEEKLY-SWITCH-FIXED-BEAR-CASH-EXTENSION-CONTRACT-001",
        "diagnostic_only": True,
        "baseline": "strict_no_bear_baseline",
        "bear_rules": {
            "current_score": "bear_score>=3",
            "trend_dd10": "market_close_vs_ma120<0 and market_ma120_slope20<0 and market_drawdown120<=-0.10",
            "crash_dd20": "market_close_vs_ma120<0 and market_drawdown120<=-0.20",
        },
        "confirmation_weeks": CONFIRMATION_WEEKS,
        "minimum_hold_weeks": MINIMUM_HOLD_WEEKS,
        "ai_rule": "ai_score=5/5",
        "state_to_strategy": {
            base.STATE_AI: "fixed7_S10_CD10",
            base.STATE_DIFFUSION: "00631L_buyhold",
            STATE_BEAR: "cash",
        },
        "threshold_grid_authorized": False,
        "formal_model_changed": False,
        "trade_decision_changed": False,
        "active_in_trade_decision": False,
        "report_changed": False,
        "not_live_rule": True,
        "future_data_violation_count": 0,
    }
    (OUT / "strict_bear_cash_policy.json").write_text(json.dumps(policy, ensure_ascii=False, indent=2), encoding="utf-8")
    readiness = {
        "ready_for_experiments": ready_for_experiments,
        "ready_variant_period_rows": int(readiness_frame["exact_path_ready"].sum()) if not readiness_frame.empty else 0,
        "total_variant_period_rows": int(len(readiness_frame)),
        "execution_blocked_rows": int(len(blocker_frame)),
        "execution_blocked_unique_keys": int(blocker_frame[["ticker", "date"]].drop_duplicates().shape[0]) if not blocker_frame.empty else 0,
        "official_raw_requirement_rows": int(len(requirement_frame)),
        "official_raw_ready_rows": int(requirement_frame["official_raw_ready"].sum()) if not requirement_frame.empty else 0,
        "exact_bounded_delta_required": bool(len(blocker_frame) > 0),
        "path_independent_execution_requirement_rows": int(len(requirement_union)),
        "path_independent_execution_missing_rows": int(len(union_missing)),
        "path_independent_execution_missing_unique_keys": int(len(bounded)),
        "bounded_official_raw_execution_gap_unique_keys": int(len(bounded)),
        "data_readiness_blocked_only": bool(len(blocker_frame) > 0),
        "may_be_used_to_reject_strategy": False,
        "future_data_violation_count": 0,
    }
    (OUT / "readiness_for_experiments.json").write_text(json.dumps(readiness, ensure_ascii=False, indent=2), encoding="utf-8")

    file_names = [
        "strict_bear_cash_corrected_NAV_daily_wealth_ledger.csv.gz",
        "strict_bear_cash_unique_position_action_ledger.csv",
        "strict_bear_cash_execution_requirement_ledger.csv",
        "strict_bear_cash_path_independent_execution_requirement_union.csv",
        "strict_bear_cash_blocked_ledger.csv",
        "strict_bear_cash_bounded_official_raw_execution_gap_ledger.csv",
        "strict_bear_cash_00631L_benchmark_daily_ledger.csv.gz",
        "strict_bear_cash_exact_nav_metrics_preview.csv",
        "strict_bear_cash_per_variant_readiness.csv",
        "strict_bear_cash_weekly_state_matrix.csv.gz",
        "strict_bear_cash_policy.json",
        "readiness_for_experiments.json",
    ]
    manifest = {
        "output_dir": str(OUT),
        "source_contract": str(base.OUT),
        "files": [
            {"path": name, "sha256": base.sha256(OUT / name), "bytes": (OUT / name).stat().st_size}
            for name in file_names
            if (OUT / name).exists()
        ],
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# strict AI diffusion weekly switch fixed bear/cash extension",
        "",
        f"- ready_for_experiments：{ready_for_experiments}",
        f"- execution_blocked_rows：{len(blocker_frame)}",
        f"- bounded official raw execution gap unique keys：{len(bounded)}",
        "- variants：strict_no_bear_baseline / current_score / trend_dd10 / crash_dd20",
        "",
        "本包只 materialize 固定 bear/cash extension；不調 AI/擴散切換，不新增 threshold/grid，不作策略結論。",
    ]
    if ready_for_experiments:
        lines.append("目前 exact path ready，可交 Experiments 比較 bear overlay vs strict_no_bear 與 00631L。")
    else:
        lines.append("目前不可交 Experiments；只可依 bounded gap ledger 補 official close/OHLC execution legs。")
    (OUT / "final_summary_zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "current_step.txt").write_text(
        "completed_ready_for_experiments\n" if ready_for_experiments else "completed_blocked_exact_delta_required\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
