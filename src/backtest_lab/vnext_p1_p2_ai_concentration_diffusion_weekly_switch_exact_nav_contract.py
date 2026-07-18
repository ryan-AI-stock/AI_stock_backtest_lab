from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
STRATEGY_ROOT = Path(
    r"C:\Users\zergv\Documents\Codex\2026-07-06"
    r"\strategy-center-core-experiments-research-materials"
)
STAGE_A_OUT = STRATEGY_ROOT / "outputs/vnext_three_state_weekly_switch_stage_a_20260718"
PRICE_ROOT = ROOT / "backtest_cache/stock_pool_observations"
OFFICIAL_PRIMARY80_CLOSE = Path(
    r"C:\Users\zergv\Documents\Codex\2026-05-23"
    r"\ai-stock-rotation-radar-https-docs\outputs"
    r"\radar_vnext_p1_p2_primary80_path_independent_raw_close_bulk_fill_20260716"
    r"\path_independent_primary80_official_raw_close_compact.csv.gz"
)
OFFICIAL_00631L_2014_2015 = ROOT / "data/normalized_prices/00631L_twse_stock_day_201411_201512.csv"
OUT = ROOT / "outputs/vnext_p1_p2_ai_concentration_diffusion_weekly_switch_exact_nav_contract_20260718"

INITIAL_CAPITAL = 1_000_000.0
BROKERAGE = 0.001425
STOCK_TAX = 0.003
ETF_TAX = 0.001
SLIPPAGE_PRIMARY = 0.001
SLIPPAGE_SENSITIVITY = (0.0005, 0.001, 0.002)

STATE_AI = "AI集中行情"
STATE_DIFFUSION = "大盤擴散行情"
FIXED7 = ["2330", "2454", "2382", "2317", "6669", "3231", "2308"]
VARIANTS = ["balanced_no_bear", "strict_no_bear"]
PERIODS = {
    "P1": (pd.Timestamp("2015-01-02"), pd.Timestamp("2022-12-29")),
    "P2": (pd.Timestamp("2023-01-02"), pd.Timestamp("2026-06-29")),
}


@dataclass(frozen=True)
class Signal:
    entry_ma: int
    entry_slope: int
    exit_ma: int
    exit_slope: int
    cooldown: int


FIXED7_SIGNAL = Signal(entry_ma=10, entry_slope=20, exit_ma=20, exit_slope=20, cooldown=10)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_price(ticker: str) -> pd.DataFrame:
    path = PRICE_ROOT / f"{ticker}_TW.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing local price cache: {path}")
    frame = pd.read_csv(path)
    if "adj_close" not in frame and "adjusted_close" in frame:
        frame = frame.rename(columns={"adjusted_close": "adj_close"})
    frame["date"] = pd.to_datetime(frame["date"]).dt.tz_localize(None)
    frame["ticker"] = ticker
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame["adj_close"] = pd.to_numeric(frame["adj_close"], errors="coerce")
    return frame.dropna(subset=["date", "close", "adj_close"])[
        ["date", "ticker", "close", "adj_close"]
    ].copy()


def add_ma_slope(prices: pd.DataFrame) -> pd.DataFrame:
    frame = prices.sort_values(["ticker", "date"]).copy()
    grouped = frame.groupby("ticker", sort=False)
    for window in (10, 20):
        frame[f"MA{window}"] = grouped["adj_close"].transform(
            lambda values: values.rolling(window, min_periods=window).mean()
        )
    base = grouped["adj_close"].shift(FIXED7_SIGNAL.entry_slope - 1)
    frame[f"slope{FIXED7_SIGNAL.entry_slope}"] = frame["adj_close"] - base
    return frame


def load_prices() -> pd.DataFrame:
    tickers = sorted(set(FIXED7 + ["00631L"]))
    return add_ma_slope(pd.concat([read_price(ticker) for ticker in tickers], ignore_index=True))


def load_official_close_index() -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    if OFFICIAL_PRIMARY80_CLOSE.exists():
        official = pd.read_csv(OFFICIAL_PRIMARY80_CLOSE, dtype={"ticker": str})
        official["date"] = pd.to_datetime(official["date"]).dt.tz_localize(None)
        official["close"] = pd.to_numeric(official["close"], errors="coerce")
        official["official_source_role"] = "path_independent_primary80_official_raw_close"
        parts.append(
            official[
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
    if OFFICIAL_00631L_2014_2015.exists():
        etf = pd.read_csv(OFFICIAL_00631L_2014_2015, dtype={"ticker": str})
        etf["ticker"] = etf["ticker"].str.replace(".TW", "", regex=False)
        etf["date"] = pd.to_datetime(etf["date"]).dt.tz_localize(None)
        etf["market"] = "TWSE"
        etf["close"] = pd.to_numeric(etf["close"], errors="coerce")
        etf["source_quality"] = "official_twse_stock_day_backfill_201411_201512"
        etf["source_url"] = ""
        etf["source_hash"] = sha256(OFFICIAL_00631L_2014_2015)
        etf["official_source_role"] = "normalized_00631L_twse_stock_day_backfill"
        parts.append(
            etf[
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
    if not parts:
        return pd.DataFrame()
    index = pd.concat(parts, ignore_index=True).dropna(subset=["ticker", "date", "close"])
    index = index.sort_values(["ticker", "date", "official_source_role"]).drop_duplicates(
        ["ticker", "date"], keep="first"
    )
    return index


def first_stage_week_for_period(period: str) -> pd.Timestamp:
    requested_start, requested_end = PERIODS[period]
    frame = load_stage_a_variant(VARIANTS[0])
    in_period = frame.loc[frame["decision_week"].between(requested_start, requested_end)]
    if in_period.empty:
        raise RuntimeError(f"no Stage A weekly key for {period}")
    return pd.Timestamp(in_period["decision_week"].min())


def calendar_for_period(prices: pd.DataFrame, period: str) -> list[pd.Timestamp]:
    _, end = PERIODS[period]
    first_week = first_stage_week_for_period(period)
    calendar = (
        prices.loc[prices["ticker"].eq("00631L") & prices["date"].gt(first_week) & prices["date"].le(end), "date"]
        .drop_duplicates()
        .sort_values()
        .tolist()
    )
    if not calendar:
        raise RuntimeError(f"empty 00631L calendar for {period}")
    return calendar


def fixed7_target_path(prices: pd.DataFrame, calendar: list[pd.Timestamp]) -> pd.DataFrame:
    by_date = {date: group.set_index("ticker") for date, group in prices.groupby("date")}
    by_ticker = {ticker: group.set_index("date") for ticker, group in prices.groupby("ticker")}
    calendar_index = {date: idx for idx, date in enumerate(calendar)}
    ticker: str | None = None
    last_sold: str | None = None
    entry_execution_index: int | None = None
    pending: tuple[str, str, pd.Timestamp, pd.Timestamp, str] | None = None
    rows: list[dict] = []

    for date in calendar:
        action = "cash" if ticker is None else "hold"
        action_reason = ""
        if pending is not None and pending[2] == date:
            role, target, _, decision_date, reason = pending
            if role == "sell" and ticker == target:
                last_sold = ticker
                ticker = None
                entry_execution_index = None
                action = "sell"
                action_reason = reason
            elif role == "buy" and ticker is None:
                ticker = target
                entry_execution_index = calendar_index[date]
                action = "buy"
                action_reason = reason
            pending = None

        day = by_date.get(date)
        next_idx = calendar_index[date] + 1
        next_date = calendar[next_idx] if next_idx < len(calendar) else None
        if day is not None and next_date is not None and pending is None:
            if ticker is None:
                eligible_rows = []
                for candidate in FIXED7:
                    if candidate == last_sold or candidate not in day.index:
                        continue
                    row = day.loc[candidate]
                    if (
                        pd.notna(row.get("MA10"))
                        and pd.notna(row.get("slope20"))
                        and float(row["adj_close"]) > float(row["MA10"])
                        and float(row["slope20"]) > 0
                    ):
                        base = by_ticker.get(candidate)
                        if base is not None and next_date in base.index:
                            eligible_rows.append(
                                {
                                    "ticker": candidate,
                                    "normalized_slope20": float(row["slope20"] / (row["adj_close"] - row["slope20"]))
                                    if float(row["adj_close"] - row["slope20"]) != 0
                                    else np.nan,
                                    "distance_above_ma": float(row["adj_close"] / row["MA10"] - 1),
                                }
                            )
                if eligible_rows:
                    target = (
                        pd.DataFrame(eligible_rows)
                        .sort_values(
                            ["normalized_slope20", "distance_above_ma", "ticker"],
                            ascending=[False, True, True],
                        )
                        .iloc[0]["ticker"]
                    )
                    pending = ("buy", str(target), next_date, date, "fixed7_entry_signal")
            elif ticker in day.index:
                held = day.loc[ticker]
                sell_signal = bool(
                    pd.notna(held.get("MA20"))
                    and pd.notna(held.get("slope20"))
                    and float(held["adj_close"]) < float(held["MA20"])
                    and float(held["slope20"]) < 0
                )
                unlocked = (
                    entry_execution_index is not None
                    and next_idx - entry_execution_index > FIXED7_SIGNAL.cooldown
                )
                if sell_signal and unlocked:
                    pending = ("sell", ticker, next_date, date, "fixed7_exit_signal")

        rows.append(
            {
                "date": date,
                "fixed7_ticker": ticker,
                "fixed7_action": action,
                "fixed7_action_reason": action_reason,
            }
        )
    return pd.DataFrame(rows)


def load_stage_a_variant(variant_id: str) -> pd.DataFrame:
    path = STAGE_A_OUT / "three_state_weekly_feature_and_state_matrix.csv.gz"
    frame = pd.read_csv(path)
    frame = frame.loc[frame["variant_id"].eq(variant_id)].copy()
    frame["decision_week"] = pd.to_datetime(frame["decision_week"]).dt.tz_localize(None)
    return frame.sort_values("decision_week")


def state_for_date(weekly: pd.DataFrame, date: pd.Timestamp) -> tuple[str, pd.Timestamp | None]:
    eligible = weekly.loc[weekly["decision_week"].lt(date)]
    if eligible.empty:
        return STATE_DIFFUSION, None
    row = eligible.iloc[-1]
    state = str(row["decision_state"])
    if state not in {STATE_AI, STATE_DIFFUSION}:
        state = STATE_DIFFUSION
    return state, pd.Timestamp(row["decision_week"])


def tax_rate(ticker: str, side: str, slippage: float) -> float:
    if side == "buy":
        return BROKERAGE + slippage
    tax = ETF_TAX if ticker == "00631L" else STOCK_TAX
    return BROKERAGE + tax + slippage


def build_source_lookup(index: pd.DataFrame) -> dict[tuple[str, pd.Timestamp], dict]:
    if index.empty:
        return {}
    return {
        (str(row.ticker), pd.Timestamp(row.date)): row._asdict()
        for row in index.itertuples(index=False)
    }


def simulate_variant_period(
    variant_id: str,
    period: str,
    prices: pd.DataFrame,
    official_lookup: dict[tuple[str, pd.Timestamp], dict],
    slippage: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    calendar = calendar_for_period(prices, period)
    weekly = load_stage_a_variant(variant_id)
    fixed7 = fixed7_target_path(prices, calendar).set_index("date")
    by_ticker = {ticker: group.set_index("date") for ticker, group in prices.groupby("ticker")}

    nav = INITIAL_CAPITAL
    current_ticker: str | None = None
    previous_date: pd.Timestamp | None = None
    rows: list[dict] = []
    actions: list[dict] = []
    requirements: list[dict] = []
    blockers: list[dict] = []

    for date in calendar:
        nav_open = nav
        gross_holding_return = 0.0
        if current_ticker is not None and previous_date is not None:
            series = by_ticker[current_ticker]
            prior_rows = series.loc[series.index <= previous_date]
            current_rows = series.loc[series.index <= date]
            if prior_rows.empty or current_rows.empty:
                blockers.append(
                    {
                        "variant_id": variant_id,
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
        if state == STATE_AI:
            target_ticker = fixed7.loc[date, "fixed7_ticker"] if date in fixed7.index else None
            target_strategy = "fixed7_S10_CD10"
        else:
            target_ticker = "00631L"
            target_strategy = "00631L_buyhold"
        if pd.isna(target_ticker):
            target_ticker = None

        action = "hold" if current_ticker else "cash"
        action_reason = ""
        transition_cost = 0.0
        source_ready = True
        if target_ticker != current_ticker:
            action = (
                "entry"
                if current_ticker is None and target_ticker is not None
                else "exit_to_cash"
                if current_ticker is not None and target_ticker is None
                else "atomic_switch"
            )
            action_reason = "weekly_state_or_fixed7_target_change"
            legs = []
            if current_ticker is not None:
                legs.append(("sell", current_ticker))
            if target_ticker is not None:
                legs.append(("buy", str(target_ticker)))
            for side, leg_ticker in legs:
                src = official_lookup.get((leg_ticker, date), {})
                cache_row = by_ticker.get(leg_ticker, pd.DataFrame()).loc[[date]] if (
                    leg_ticker in by_ticker and date in by_ticker[leg_ticker].index
                ) else pd.DataFrame()
                raw_close = float(cache_row.iloc[0]["close"]) if not cache_row.empty else np.nan
                official_ready = bool(src)
                if not official_ready:
                    source_ready = False
                    blockers.append(
                        {
                            "variant_id": variant_id,
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
                        "variant_id": variant_id,
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
                    }
                )
            # Keep the path deterministic with local cache for planning, but do not
            # mark the contract exact-ready unless every transition leg has official
            # raw close authority. This prevents one missing source row from creating
            # a repeated daily phantom frontier.
            if current_ticker is not None:
                cost = nav * tax_rate(current_ticker, "sell", slippage)
                nav -= cost
                transition_cost += cost
            if target_ticker is not None:
                cost = nav * tax_rate(str(target_ticker), "buy", slippage)
                nav -= cost
                transition_cost += cost
            actions.append(
                {
                    "variant_id": variant_id,
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
                    "official_raw_all_legs_ready": source_ready,
                    "path_preview_basis": (
                        "official_raw_execution_ready"
                        if source_ready
                        else "local_cache_execution_preview_not_experiments_ready"
                    ),
                }
            )
            current_ticker = str(target_ticker) if target_ticker is not None else None

        rows.append(
            {
                "variant_id": variant_id,
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

    return (
        pd.DataFrame(rows),
        pd.DataFrame(actions),
        pd.DataFrame(requirements),
        pd.DataFrame(blockers),
    )


def build_benchmark(period: str, prices: pd.DataFrame, slippage: float) -> pd.DataFrame:
    calendar = calendar_for_period(prices, period)
    frame = prices.loc[prices["ticker"].eq("00631L")].set_index("date").reindex(calendar)
    frame = frame.dropna(subset=["adj_close"])
    gross = frame["adj_close"] / frame["adj_close"].iloc[0]
    nav = INITIAL_CAPITAL * (1 - tax_rate("00631L", "buy", slippage)) * gross
    nav.iloc[-1] *= 1 - tax_rate("00631L", "sell", slippage)
    return pd.DataFrame(
        {
            "period": period,
            "date": nav.index,
            "benchmark": "00631L_buy_and_hold",
            "slippage_bp_per_side": int(round(slippage * 10000)),
            "NAV_close": nav.values,
            "NAV_open": nav.shift(1).fillna(INITIAL_CAPITAL).values,
        }
    )


def metric_row(frame: pd.DataFrame, benchmark: pd.DataFrame) -> dict:
    nav = frame["NAV_close"]
    bench_nav = benchmark["NAV_close"]
    drawdown = nav / nav.cummax() - 1
    bench_drawdown = bench_nav / bench_nav.cummax() - 1
    days = (pd.Timestamp(frame["date"].iloc[-1]) - pd.Timestamp(frame["date"].iloc[0])).days
    years = days / 365.25 if days > 0 else np.nan
    total_return = float(nav.iloc[-1] / INITIAL_CAPITAL - 1)
    bench_return = float(bench_nav.iloc[-1] / INITIAL_CAPITAL - 1)
    return {
        "variant_id": frame["variant_id"].iloc[0],
        "period": frame["period"].iloc[0],
        "slippage_bp_per_side": int(frame["slippage_bp_per_side"].iloc[0]),
        "actual_start": pd.Timestamp(frame["date"].iloc[0]).date().isoformat(),
        "actual_end": pd.Timestamp(frame["date"].iloc[-1]).date().isoformat(),
        "rows": int(len(frame)),
        "final_NAV": float(nav.iloc[-1]),
        "net_total_return": total_return,
        "CAGR": float((1 + total_return) ** (1 / years) - 1) if years and years > 0 else np.nan,
        "MDD": float(drawdown.min()),
        "00631L_buyhold_return": bench_return,
        "00631L_buyhold_MDD": float(bench_drawdown.min()),
        "excess_vs_00631L": total_return - bench_return,
        "transition_count": int(frame["action"].isin(["entry", "exit_to_cash", "atomic_switch"]).sum()),
        "stock_or_etf_exposure_share": float(frame["held_ticker"].notna().mean()),
        "cash_exposure_share": float(frame["held_ticker"].isna().mean()),
        "total_cost": float(frame["transition_cost"].sum()),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "current_step.txt").write_text("running\n", encoding="utf-8")
    prices = load_prices()
    official_index = load_official_close_index()
    official_lookup = build_source_lookup(official_index)

    daily_parts = []
    action_parts = []
    requirement_parts = []
    blocker_parts = []
    benchmark_parts = []
    metric_rows = []
    for slippage in SLIPPAGE_SENSITIVITY:
        for period in PERIODS:
            benchmark = build_benchmark(period, prices, slippage)
            benchmark_parts.append(benchmark)
            for variant_id in VARIANTS:
                daily, actions, requirements, blockers = simulate_variant_period(
                    variant_id, period, prices, official_lookup, slippage
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
    requirement_frame = (
        pd.concat(requirement_parts, ignore_index=True) if requirement_parts else pd.DataFrame()
    )
    blocker_frame = pd.concat(blocker_parts, ignore_index=True) if blocker_parts else pd.DataFrame()
    benchmark_frame = pd.concat(benchmark_parts, ignore_index=True)
    metrics = pd.DataFrame(metric_rows)

    readiness_by_variant = []
    for (variant_id, period), group in requirement_frame.groupby(["variant_id", "period"], dropna=False):
        blockers = blocker_frame.loc[
            blocker_frame["variant_id"].eq(variant_id) & blocker_frame["period"].eq(period)
        ]
        readiness_by_variant.append(
            {
                "variant_id": variant_id,
                "period": period,
                "execution_requirement_rows": int(len(group)),
                "official_raw_ready_rows": int(group["official_raw_ready"].sum()),
                "execution_blocker_rows": int(len(blockers)),
                "exact_path_ready": bool(blockers.empty and group["official_raw_ready"].all()),
            }
        )
    readiness_frame = pd.DataFrame(readiness_by_variant)
    ready_for_experiments = bool(
        not readiness_frame.empty
        and readiness_frame.loc[
            readiness_frame["period"].isin(["P1", "P2"]), "exact_path_ready"
        ].all()
    )

    daily_frame.to_csv(OUT / "weekly_switch_corrected_NAV_daily_wealth_ledger.csv.gz", index=False, compression="gzip")
    action_frame.to_csv(OUT / "weekly_switch_unique_position_action_ledger.csv", index=False, encoding="utf-8-sig")
    requirement_frame.to_csv(OUT / "weekly_switch_execution_requirement_ledger.csv", index=False, encoding="utf-8-sig")
    blocker_frame.to_csv(OUT / "weekly_switch_blocked_ledger.csv", index=False, encoding="utf-8-sig")
    if not blocker_frame.empty:
        bounded = (
            blocker_frame[["ticker", "date", "blocker_class", "blocked_reason"]]
            .drop_duplicates()
            .sort_values(["ticker", "date"])
            .copy()
        )
        bounded["authorized_source_scope"] = "exact official raw close for transition execution leg only"
        bounded["network_family_allowed"] = "official close/OHLC only; no non-close family"
    else:
        bounded = pd.DataFrame(
            columns=[
                "ticker",
                "date",
                "blocker_class",
                "blocked_reason",
                "authorized_source_scope",
                "network_family_allowed",
            ]
        )
    bounded.to_csv(
        OUT / "weekly_switch_bounded_official_raw_execution_gap_ledger.csv",
        index=False,
        encoding="utf-8-sig",
    )
    benchmark_frame.to_csv(OUT / "weekly_switch_00631L_benchmark_daily_ledger.csv.gz", index=False, compression="gzip")
    metrics.to_csv(OUT / "weekly_switch_exact_nav_metrics_preview.csv", index=False, encoding="utf-8-sig")
    readiness_frame.to_csv(OUT / "weekly_switch_per_variant_readiness.csv", index=False, encoding="utf-8-sig")

    policy = {
        "task_id": "TASK-BACKTEST-CORE-VNEXT-P1-P2-AI-CONCENTRATION-DIFFUSION-WEEKLY-SWITCH-EXACT-NAV-CONTRACT-001",
        "diagnostic_only": True,
        "stage_a_proxy_superseded_for_performance": True,
        "bear_cash_module_used": False,
        "variants": {
            "balanced_no_bear": {
                "ai_score_rule": "ai_score>=4/5",
                "confirmation_weeks": 2,
                "minimum_hold_weeks": 4,
            },
            "strict_no_bear": {
                "ai_score_rule": "ai_score=5/5",
                "confirmation_weeks": 3,
                "minimum_hold_weeks": 6,
            },
        },
        "state_to_strategy": {
            STATE_AI: "fixed7_S10_CD10",
            STATE_DIFFUSION: "00631L_buyhold",
        },
        "execution_semantics": "weekly close decision or fixed7 internal execution target; transition at actual execution date; sell before buy for atomic switch",
        "analysis_mark_basis": "event-aware adjusted close from local stock_pool_observations cache; research diagnostic, not formal total return",
        "execution_price_basis": "official raw close required for transition legs; cache close retained only for audit",
        "cost_model": {
            "brokerage": BROKERAGE,
            "stock_sell_tax": STOCK_TAX,
            "etf_sell_tax": ETF_TAX,
            "primary_slippage_bp_per_side": 10,
            "sensitivity_slippage_bp_per_side": [5, 10, 20],
        },
        "formal_model_changed": False,
        "trade_decision_changed": False,
        "active_in_trade_decision": False,
        "report_changed": False,
        "not_live_rule": True,
        "future_data_violation_count": 0,
    }
    (OUT / "weekly_switch_policy.json").write_text(
        json.dumps(policy, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    readiness = {
        "ready_for_experiments": ready_for_experiments,
        "ready_variant_period_rows": int(readiness_frame["exact_path_ready"].sum()) if not readiness_frame.empty else 0,
        "total_variant_period_rows": int(len(readiness_frame)),
        "execution_blocked_rows": int(len(blocker_frame)),
        "execution_blocked_unique_keys": int(
            blocker_frame[["ticker", "date"]].drop_duplicates().shape[0]
        )
        if not blocker_frame.empty and {"ticker", "date"}.issubset(blocker_frame.columns)
        else 0,
        "official_raw_requirement_rows": int(len(requirement_frame)),
        "official_raw_ready_rows": int(requirement_frame["official_raw_ready"].sum())
        if not requirement_frame.empty
        else 0,
        "exact_bounded_delta_required": bool(len(blocker_frame) > 0),
        "bounded_official_raw_execution_gap_unique_keys": int(len(bounded)),
        "ready_for_radar_bounded_close_fill": bool(len(bounded) > 0),
        "radar_authority_file": "weekly_switch_bounded_official_raw_execution_gap_ledger.csv",
        "data_readiness_blocked_only": bool(len(blocker_frame) > 0),
        "may_be_used_to_reject_strategy": False,
        "future_data_violation_count": 0,
    }
    (OUT / "readiness_for_experiments.json").write_text(
        json.dumps(readiness, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    manifest_files = [
        "weekly_switch_corrected_NAV_daily_wealth_ledger.csv.gz",
        "weekly_switch_unique_position_action_ledger.csv",
        "weekly_switch_execution_requirement_ledger.csv",
        "weekly_switch_blocked_ledger.csv",
        "weekly_switch_bounded_official_raw_execution_gap_ledger.csv",
        "weekly_switch_00631L_benchmark_daily_ledger.csv.gz",
        "weekly_switch_exact_nav_metrics_preview.csv",
        "weekly_switch_per_variant_readiness.csv",
        "weekly_switch_policy.json",
        "readiness_for_experiments.json",
    ]
    manifest = {
        "output_dir": str(OUT),
        "stage_a_output": str(STAGE_A_OUT),
        "files": [
            {"path": name, "sha256": sha256(OUT / name), "bytes": (OUT / name).stat().st_size}
            for name in manifest_files
            if (OUT / name).exists()
        ],
    }
    (OUT / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "# AI集中／大盤擴散 weekly switch exact NAV contract",
        "",
        f"- ready_for_experiments：{ready_for_experiments}",
        f"- execution_blocked_rows：{len(blocker_frame)}",
        f"- bounded official raw execution gap unique keys：{len(bounded)}",
        f"- official_raw_requirement_rows：{len(requirement_frame)}",
        "",
        "本包只驗證 balanced_no_bear / strict_no_bear 的 no-bear 兩狀態 exact path；Stage A proxy 不作 exact 績效權威。",
        "目前不可交 Experiments；下一棒是 Radar/Data 只補 bounded ledger 內的 official close/OHLC execution legs，不得擴到非 close family。",
    ]
    (OUT / "final_summary_zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "current_step.txt").write_text(
        "completed_contract_ready\n" if ready_for_experiments else "completed_blocked_exact_delta_required\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
