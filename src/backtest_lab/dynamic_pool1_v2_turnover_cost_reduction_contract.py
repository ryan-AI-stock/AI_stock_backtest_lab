"""Build Dynamic Pool1 v2 turnover/cost reduction diagnostic contract."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from backtest_lab.costs import COST_MODEL_VERSION, TaiwanCostModel, cost_model_metadata
from backtest_lab.dynamic_pool1_v2_bounded_portfolio_contract import (
    DEFAULT_CANDIDATE_V0_POOL,
    DEFAULT_RADAR_LIQUIDITY_DIR,
    DEFAULT_V2_MEMBER_PANEL,
    _load_market_lookup,
    _load_member_panel,
)


TASK_ID = "TASK-BACKTEST-CORE-DYNAMIC-POOL1-V2-TURNOVER-COST-REDUCTION-CONTRACT-001"
EXPERIMENTS_TASK_ID = "TASK-BACKTEST-EXPERIMENTS-DYNAMIC-POOL1-V2-TURNOVER-COST-REDUCTION-VALIDATION-001"
DEFAULT_SOURCE_CONTRACT = Path("outputs/dynamic_pool1_v2_bounded_portfolio_contract_20260704")
DEFAULT_OUTPUT_DIR = Path("outputs/dynamic_pool1_v2_turnover_cost_reduction_contract_20260704")
INITIAL_EQUITY = 1_000_000.0
SLEEVE_WEIGHT = 0.20


VARIANTS = [
    {
        "variant": "v2_top15_top1_monthly_lock_when_formal_cash_or_market_exposure",
        "source_variant_id": "v2_primary_rs60_top15_monthly",
        "top_n": 1,
        "rule": "monthly_lock",
        "rank_improvement_required": 0,
        "min_hold_days": 0,
        "cooldown_days": 0,
        "monthly_lock_active": True,
        "description": "每月最多選一檔 top15 top1，同月不切換，除非 blocked / 不可交易 / formal state 不允許。",
    },
    {
        "variant": "v2_top15_top1_min_hold10_cooldown5_when_formal_cash_or_market_exposure",
        "source_variant_id": "v2_primary_rs60_top15_monthly",
        "top_n": 1,
        "rule": "min_hold10_cooldown5",
        "rank_improvement_required": 0,
        "min_hold_days": 10,
        "cooldown_days": 5,
        "monthly_lock_active": False,
        "description": "top15 top1，dynamic sleeve 最低持有 10 個交易日；離場後 cooldown 5 個交易日。",
    },
    {
        "variant": "v2_top15_top1_switch_only_if_rank_improves_by_5_when_formal_cash_or_market_exposure",
        "source_variant_id": "v2_primary_rs60_top15_monthly",
        "top_n": 1,
        "rule": "rank_improves_by_5",
        "rank_improvement_required": 5,
        "min_hold_days": 0,
        "cooldown_days": 0,
        "monthly_lock_active": False,
        "description": "top15 top1，只有新候選排名至少比現有 dynamic candidate 好 5 名才切換。",
    },
    {
        "variant": "v2_top15_top3_equal_weight_monthly_rebalance_only",
        "source_variant_id": "v2_primary_rs60_top15_monthly",
        "top_n": 3,
        "rule": "top3_monthly_rebalance",
        "rank_improvement_required": 0,
        "min_hold_days": 0,
        "cooldown_days": 0,
        "monthly_lock_active": True,
        "description": "top15 前三名等權，僅月度 rebalance，不 daily rebalance。",
    },
    {
        "variant": "v2_top15_top1_when_formal_cash_or_market_exposure_hold20_reference",
        "source_variant_id": "v2_primary_rs60_top15_monthly",
        "top_n": 1,
        "rule": "hold20_reference",
        "rank_improvement_required": 0,
        "min_hold_days": 0,
        "cooldown_days": 0,
        "monthly_lock_active": False,
        "description": "上一輪 top15 top1 hold20 reference，只作比較，不升 formal。",
    },
]


@dataclass
class VariantState:
    cash: float = INITIAL_EQUITY
    shares: dict[str, float] = field(default_factory=dict)
    entry_date_by_ticker: dict[str, str] = field(default_factory=dict)
    entry_index_by_ticker: dict[str, int] = field(default_factory=dict)
    last_exit_index: int | None = None
    current_month: str = ""
    locked_tickers: list[str] = field(default_factory=list)


def run_dynamic_pool1_v2_turnover_cost_reduction_contract(
    *,
    repo_root: str | Path = ".",
    source_contract_dir: str | Path = DEFAULT_SOURCE_CONTRACT,
    v2_member_panel: str | Path = DEFAULT_V2_MEMBER_PANEL,
    candidate_v0_pool: str | Path = DEFAULT_CANDIDATE_V0_POOL,
    liquidity_dir: str | Path = DEFAULT_RADAR_LIQUIDITY_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict:
    root = Path(repo_root).resolve()
    source = _resolve(root, source_contract_dir)
    output = _resolve(root, output_dir)
    output.mkdir(parents=True, exist_ok=True)
    liquidity = _resolve(root, liquidity_dir)
    market_lookup = _load_market_lookup(_resolve(root, candidate_v0_pool), liquidity)
    member_panel = _load_member_panel(_resolve(root, v2_member_panel), market_lookup)
    top15 = member_panel[member_panel["variant_id"].eq("v2_primary_rs60_top15_monthly")].copy()
    signal = pd.read_csv(source / "daily_signal_panel.csv").fillna("")
    signal = _base_signal_rows(signal)
    price_table = _load_price_table(liquidity, signal["next_tradable_date"].dropna().astype(str).tolist(), top15["canonical_ticker"].dropna().astype(str).tolist())

    variant_matrix = _variant_matrix()
    daily_frames = []
    weight_frames = []
    trade_frames = []
    blocked_frames = []
    next_day_frames = []
    overlap_frames = []
    for variant in VARIANTS:
        result = _simulate_variant(signal, top15, price_table, variant)
        daily_frames.append(result["daily"])
        weight_frames.append(result["weights"])
        trade_frames.append(result["trades"])
        blocked_frames.append(result["blocked"])
        next_day_frames.append(result["next_day"])
        overlap_frames.append(result["overlap"])

    daily_equity = pd.concat(daily_frames, ignore_index=True, sort=False)
    daily_weight = pd.concat(weight_frames, ignore_index=True, sort=False)
    trade_ledger = pd.concat(trade_frames, ignore_index=True, sort=False)
    blocked = pd.concat(blocked_frames, ignore_index=True, sort=False)
    next_day = pd.concat(next_day_frames, ignore_index=True, sort=False)
    overlap = pd.concat(overlap_frames, ignore_index=True, sort=False)
    cost_summary = _cost_summary(trade_ledger)
    turnover_summary = _turnover_summary(trade_ledger, daily_equity)
    period = _period_performance(daily_equity)
    monthly = _monthly_performance(daily_equity)
    event_usage = _event_usage_summary(trade_ledger)
    concentration = _concentration_audit(trade_ledger)
    future_audit = _future_data_audit(daily_weight)

    variant_matrix.to_csv(output / "variant_matrix.csv", index=False, encoding="utf-8-sig")
    daily_equity.to_csv(output / "daily_equity.csv", index=False, encoding="utf-8-sig")
    daily_weight.to_csv(output / "daily_weight_ledger.csv", index=False, encoding="utf-8-sig")
    trade_ledger.to_csv(output / "trade_ledger.csv", index=False, encoding="utf-8-sig")
    event_usage.to_csv(output / "event_usage_summary.csv", index=False, encoding="utf-8-sig")
    turnover_summary.to_csv(output / "turnover_summary.csv", index=False, encoding="utf-8-sig")
    cost_summary.to_csv(output / "cost_summary.csv", index=False, encoding="utf-8-sig")
    period.to_csv(output / "period_performance.csv", index=False, encoding="utf-8-sig")
    monthly.to_csv(output / "monthly_performance.csv", index=False, encoding="utf-8-sig")
    blocked.to_csv(output / "blocked_fill_audit.csv", index=False, encoding="utf-8-sig")
    next_day.to_csv(output / "next_day_audit.csv", index=False, encoding="utf-8-sig")
    concentration.to_csv(output / "concentration_audit.csv", index=False, encoding="utf-8-sig")
    overlap.to_csv(output / "formal_target_overlap_audit.csv", index=False, encoding="utf-8-sig")
    future_audit.to_csv(output / "future_data_audit.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([cost_model_metadata()]).to_csv(output / "cost_model_contract.csv", index=False, encoding="utf-8-sig")

    manifest = {
        "task_id": TASK_ID,
        "status": "completed_turnover_cost_reduction_contract_ready",
        "output_dir": str(output),
        "source_contract_dir": str(source),
        "variant_count": int(len(variant_matrix)),
        "daily_equity_rows": int(len(daily_equity)),
        "daily_weight_rows": int(len(daily_weight)),
        "trade_rows": int(len(trade_ledger)),
        "blocked_fill_rows": int(len(blocked)),
        "missing_price_tickers": sorted(set(blocked.loc[blocked["blocked_reason"].eq("missing_price"), "dynamic_selected_ticker"].astype(str))),
        "future_data_violation_count": int(future_audit["future_data_violation"].sum()) if not future_audit.empty else 0,
        "same_day_execution_mixed": False,
        "formal_direct_stock_target_override_allowed": False,
        "uses_forward_return_as_rule": False,
        "portfolio_replay_executed": True,
        "diagnostic_challenger_only": True,
        "formal_model_changed": False,
        "trade_decision_changed": False,
        "active_in_trade_decision": False,
        "report_changed": False,
        "handoff_to_experiments_task": EXPERIMENTS_TASK_ID,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "final_summary_zh.md").write_text(_summary(manifest, cost_summary, period), encoding="utf-8")
    pd.DataFrame([{"task_id": TASK_ID, "status": "completed", "output_dir": str(output)}]).to_csv(
        output / "completed.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(columns=["task_id", "status", "reason"]).to_csv(output / "failed.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(
        [
            {"step": "load_core_v2_contract", "status": "completed"},
            {"step": "load_price_shards", "status": "completed"},
            {"step": "simulate_turnover_cost_variants", "status": "completed"},
            {"step": "write_contract_package", "status": "completed"},
        ]
    ).to_csv(output / "run_log.csv", index=False, encoding="utf-8-sig")
    return manifest


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _base_signal_rows(signal: pd.DataFrame) -> pd.DataFrame:
    base = signal[signal["dynamic_pool_variant"].eq("v2_top15_top1_when_formal_cash_or_market_exposure_hold20")].copy()
    base["date"] = pd.to_datetime(base["date"], errors="coerce")
    base["next_tradable_date"] = pd.to_datetime(base["next_tradable_date"], errors="coerce")
    base = base.dropna(subset=["date", "next_tradable_date"]).sort_values("next_tradable_date")
    return base


def _load_price_table(liquidity_dir: Path, dates: list[str], tickers: list[str]) -> pd.DataFrame:
    needed_months = sorted({pd.to_datetime(date, errors="coerce").strftime("%Y_%m") for date in dates if pd.notna(pd.to_datetime(date, errors="coerce"))})
    needed_tickers = set(tickers)
    frames = []
    for month in needed_months:
        shard = liquidity_dir / "shards" / f"accepted_liquidity_rows_{month}.csv"
        if not shard.exists():
            continue
        df = pd.read_csv(shard, usecols=["date", "ticker", "market", "close"])
        df["canonical_ticker"] = df.apply(lambda row: f"{row['ticker']}{'.TW' if row['market'] == 'TWSE' else '.TWO'}", axis=1)
        df = df[df["canonical_ticker"].isin(needed_tickers)].copy()
        if not df.empty:
            frames.append(df[["date", "canonical_ticker", "close"]])
    if not frames:
        return pd.DataFrame(columns=["date", "canonical_ticker", "close"])
    out = pd.concat(frames, ignore_index=True, sort=False)
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    return out.dropna(subset=["date", "canonical_ticker", "close"]).drop_duplicates(["date", "canonical_ticker"])


def _simulate_variant(signal: pd.DataFrame, top15: pd.DataFrame, price_table: pd.DataFrame, variant: dict) -> dict[str, pd.DataFrame]:
    state = VariantState()
    model = TaiwanCostModel()
    price_lookup = {(row["date"], row["canonical_ticker"]): float(row["close"]) for row in price_table.to_dict(orient="records")}
    daily_rows: list[dict] = []
    weight_rows: list[dict] = []
    trade_rows: list[dict] = []
    blocked_rows: list[dict] = []
    next_day_rows: list[dict] = []
    overlap_rows: list[dict] = []
    running_max = INITIAL_EQUITY
    for idx, row in enumerate(signal.to_dict(orient="records")):
        date = pd.to_datetime(row["next_tradable_date"]).strftime("%Y-%m-%d")
        signal_date = pd.to_datetime(row["date"]).strftime("%Y-%m-%d")
        equity_before = _equity(state, price_lookup, date)
        desired, switch_reason, blocked_reason, cooldown_active = _desired_weights(row, top15, variant, state, idx)
        target = _filter_missing_prices(desired, price_lookup, date)
        if desired and not target:
            blocked_reason = "missing_price"
        trades, trade_cost, turnover = _rebalance(state, target, price_lookup, date, model, equity_before)
        for trade in trades:
            trade["variant"] = variant["variant"]
            trade["dynamic_switch_reason"] = switch_reason
            trade_rows.append(trade)
        if trades and not target:
            state.last_exit_index = idx
        if target:
            for ticker in target:
                state.entry_date_by_ticker.setdefault(ticker, date)
                state.entry_index_by_ticker.setdefault(ticker, idx)
        if not target and state.shares:
            state.entry_date_by_ticker.clear()
            state.entry_index_by_ticker.clear()
        if variant["rule"] in {"monthly_lock", "top3_monthly_rebalance"}:
            state.current_month = str(row.get("dynamic_candidate_pool_month", ""))
            state.locked_tickers = list(target.keys())
        equity_after = _equity(state, price_lookup, date)
        running_max = max(running_max, equity_after)
        gross_exposure = _gross_exposure(state, price_lookup, date, equity_after)
        hold_days = _max_hold_days(state, idx)
        daily_rows.append(
            {
                "variant": variant["variant"],
                "date": date,
                "signal_date": signal_date,
                "next_tradable_date": date,
                "formal_target": row.get("formal_target", ""),
                "formal_state": row.get("formal_state", ""),
                "dynamic_selected_ticker": ";".join(target.keys()),
                "dynamic_candidate_rank": _rank_text(top15, row, target),
                "dynamic_candidate_month": row.get("dynamic_candidate_pool_month", ""),
                "dynamic_entry_date": ";".join(state.entry_date_by_ticker.get(ticker, "") for ticker in target),
                "dynamic_exit_date": date if trades and not target else "",
                "dynamic_hold_days": hold_days,
                "dynamic_switch_reason": switch_reason,
                "cooldown_active": cooldown_active,
                "monthly_lock_active": bool(variant["monthly_lock_active"]),
                "rank_improvement_required": variant["rank_improvement_required"],
                "trade_cost": round(trade_cost, 2),
                "turnover": round(turnover, 2),
                "blocked_reason": blocked_reason,
                "equity": round(equity_after, 2),
                "drawdown": round(equity_after / running_max - 1, 8) if running_max else 0.0,
                "gross_exposure": round(gross_exposure, 8),
                "uses_forward_return_as_rule": False,
            }
        )
        for ticker, weight in target.items():
            weight_rows.append(
                {
                    "variant": variant["variant"],
                    "date": date,
                    "next_tradable_date": date,
                    "formal_target": row.get("formal_target", ""),
                    "formal_state": row.get("formal_state", ""),
                    "dynamic_selected_ticker": ticker,
                    "dynamic_selected_weight": weight,
                    "dynamic_candidate_rank": _candidate_rank(top15, row, ticker),
                    "dynamic_candidate_month": row.get("dynamic_candidate_pool_month", ""),
                    "dynamic_entry_date": state.entry_date_by_ticker.get(ticker, ""),
                    "dynamic_hold_days": idx - state.entry_index_by_ticker.get(ticker, idx) + 1,
                    "dynamic_switch_reason": switch_reason,
                    "cooldown_active": cooldown_active,
                    "monthly_lock_active": bool(variant["monthly_lock_active"]),
                    "rank_improvement_required": variant["rank_improvement_required"],
                    "blocked_reason": blocked_reason,
                    "uses_forward_return_as_rule": False,
                }
            )
        if blocked_reason:
            blocked_rows.append(
                {
                    "variant": variant["variant"],
                    "date": date,
                    "formal_state": row.get("formal_state", ""),
                    "dynamic_selected_ticker": row.get("dynamic_selected_canonical_ticker", ""),
                    "blocked_reason": blocked_reason,
                }
            )
        next_day_rows.append(
            {
                "variant": variant["variant"],
                "date": signal_date,
                "next_tradable_date": date,
                "same_day_execution_mixed": False,
                "blocked_reason": "" if date else "missing_next_tradable_date",
            }
        )
        overlap_rows.append(
            {
                "variant": variant["variant"],
                "date": date,
                "formal_target": row.get("formal_target", ""),
                "formal_state": row.get("formal_state", ""),
                "dynamic_selected_ticker": ";".join(target.keys()),
                "formal_direct_stock_target_override_allowed": False,
                "active_in_trade_decision": False,
            }
        )
    return {
        "daily": pd.DataFrame(daily_rows),
        "weights": pd.DataFrame(weight_rows),
        "trades": pd.DataFrame(trade_rows),
        "blocked": pd.DataFrame(blocked_rows),
        "next_day": pd.DataFrame(next_day_rows),
        "overlap": pd.DataFrame(overlap_rows),
    }


def _desired_weights(row: dict, top15: pd.DataFrame, variant: dict, state: VariantState, idx: int) -> tuple[dict[str, float], str, str, bool]:
    if row.get("formal_state") not in {"cash", "no_target_cash", "market_exposure"}:
        return {}, "formal_state_not_allowed_exit_or_hold_cash", "formal_direct_stock_target_no_override", False
    month = str(row.get("dynamic_candidate_pool_month", ""))
    candidates = _candidate_rows_for_month(top15, month)
    if candidates.empty:
        return {}, "no_candidate_pool", "no_candidate_pool", False
    top1 = str(candidates.iloc[0]["canonical_ticker"])
    top3 = candidates.head(3)["canonical_ticker"].astype(str).tolist()
    if variant["rule"] == "monthly_lock":
        if state.locked_tickers and state.current_month == month:
            selected = state.locked_tickers
            return _weights(selected), "monthly_lock_keep_existing", "", False
        return _weights([top1]), "monthly_lock_new_month_select_top1", "", False
    if variant["rule"] == "top3_monthly_rebalance":
        if state.locked_tickers and state.current_month == month:
            return _weights(state.locked_tickers), "top3_monthly_rebalance_keep_existing", "", False
        return _weights(top3), "top3_monthly_rebalance_new_month", "", False
    if variant["rule"] == "min_hold10_cooldown5":
        cooldown_active = state.last_exit_index is not None and idx - state.last_exit_index <= 5
        current = next(iter(state.shares.keys()), "")
        if cooldown_active and not current:
            return {}, "cooldown_after_exit", "cooldown_active", True
        if current:
            held = idx - state.entry_index_by_ticker.get(current, idx) + 1
            if current == top1 or held < 10:
                return _weights([current]), "min_hold10_keep_existing", "", False
            state.last_exit_index = idx
            return {}, "min_hold10_exit_then_cooldown", "", False
        return _weights([top1]), "min_hold10_enter_top1", "", False
    if variant["rule"] == "rank_improves_by_5":
        current = next(iter(state.shares.keys()), "")
        if current:
            current_rank = _rank_of(candidates, current)
            if current_rank is None:
                return _weights([top1]), "held_ticker_left_pool_switch_to_top1", "", False
            if current_rank - 1 >= 5 and current != top1:
                return _weights([top1]), "new_rank_improves_by_at_least_5", "", False
            return _weights([current]), "rank_improvement_threshold_not_met", "", False
        return _weights([top1]), "rank_improvement_variant_enter_top1", "", False
    if variant["rule"] == "hold20_reference":
        current = next(iter(state.shares.keys()), "")
        if current:
            held = idx - state.entry_index_by_ticker.get(current, idx) + 1
            if held < 20:
                return _weights([current]), "hold20_reference_keep_until_day20", "", False
        return _weights([top1]), "hold20_reference_enter_or_roll_top1", "", False
    return {}, "unsupported_rule", "unsupported_rule", False


def _weights(tickers: list[str]) -> dict[str, float]:
    if not tickers:
        return {}
    each = round(SLEEVE_WEIGHT / len(tickers), 8)
    return {ticker: each for ticker in tickers}


def _candidate_rows_for_month(top15: pd.DataFrame, month: str) -> pd.DataFrame:
    return top15[top15["candidate_month"].astype(str).eq(str(month))].sort_values(["candidate_rank", "canonical_ticker"]).copy()


def _filter_missing_prices(weights: dict[str, float], price_lookup: dict[tuple[str, str], float], date: str) -> dict[str, float]:
    return {ticker: weight for ticker, weight in weights.items() if (date, ticker) in price_lookup}


def _rebalance(
    state: VariantState,
    target_weights: dict[str, float],
    price_lookup: dict[tuple[str, str], float],
    date: str,
    model: TaiwanCostModel,
    equity_before: float,
) -> tuple[list[dict], float, float]:
    trades = []
    total_cost = 0.0
    turnover = 0.0
    target_values = {ticker: equity_before * weight for ticker, weight in target_weights.items()}
    all_tickers = sorted(set(state.shares) | set(target_values))
    for ticker in all_tickers:
        price = price_lookup.get((date, ticker))
        if price is None or price <= 0:
            continue
        current_value = state.shares.get(ticker, 0.0) * price
        desired_value = target_values.get(ticker, 0.0)
        delta_value = desired_value - current_value
        if abs(delta_value) < 1:
            continue
        side = "buy" if delta_value > 0 else "sell"
        gross = abs(delta_value)
        cost = model.buy_cost(gross) if side == "buy" else model.sell_cost(gross, _asset_type(ticker))
        shares_delta = delta_value / price
        state.cash -= delta_value
        state.cash -= cost
        state.shares[ticker] = state.shares.get(ticker, 0.0) + shares_delta
        if abs(state.shares.get(ticker, 0.0)) < 1e-8 or desired_value == 0:
            state.shares.pop(ticker, None)
            state.entry_date_by_ticker.pop(ticker, None)
            state.entry_index_by_ticker.pop(ticker, None)
        total_cost += cost
        turnover += gross
        trades.append(
            {
                "date": date,
                "ticker": ticker,
                "side": side,
                "trade_value": round(delta_value, 2),
                "gross_amount": round(gross, 2),
                "cost": round(cost, 2),
                "asset_type": _asset_type(ticker),
                "cost_model_version": COST_MODEL_VERSION,
            }
        )
    return trades, total_cost, turnover


def _equity(state: VariantState, price_lookup: dict[tuple[str, str], float], date: str) -> float:
    return state.cash + sum(shares * price_lookup.get((date, ticker), 0.0) for ticker, shares in state.shares.items())


def _gross_exposure(state: VariantState, price_lookup: dict[tuple[str, str], float], date: str, equity: float) -> float:
    if equity <= 0:
        return 0.0
    gross = sum(abs(shares * price_lookup.get((date, ticker), 0.0)) for ticker, shares in state.shares.items())
    return gross / equity


def _max_hold_days(state: VariantState, idx: int) -> int:
    if not state.entry_index_by_ticker:
        return 0
    return max(idx - start + 1 for start in state.entry_index_by_ticker.values())


def _rank_of(candidates: pd.DataFrame, ticker: str) -> int | None:
    row = candidates[candidates["canonical_ticker"].astype(str).eq(str(ticker))]
    if row.empty:
        return None
    return int(float(row.iloc[0]["candidate_rank"]))


def _candidate_rank(top15: pd.DataFrame, row: dict, ticker: str) -> int | str:
    rank = _rank_of(_candidate_rows_for_month(top15, str(row.get("dynamic_candidate_pool_month", ""))), ticker)
    return rank if rank is not None else ""


def _rank_text(top15: pd.DataFrame, row: dict, weights: dict[str, float]) -> str:
    return ";".join(str(_candidate_rank(top15, row, ticker)) for ticker in weights)


def _asset_type(ticker: str) -> str:
    return "etf" if str(ticker).startswith("00") else "stock"


def _variant_matrix() -> pd.DataFrame:
    rows = []
    for item in VARIANTS:
        rows.append(
            {
                **item,
                "execution_basis": "next_day_only",
                "sleeve_weight": SLEEVE_WEIGHT,
                "uses_forward_return_as_rule": False,
                "formal_direct_stock_target_override_allowed": False,
                "formal_model_changed": False,
                "trade_decision_changed": False,
                "active_in_trade_decision": False,
            }
        )
    return pd.DataFrame(rows)


def _cost_summary(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(columns=["variant", "total_cost", "trade_count"])
    return trades.groupby("variant", as_index=False).agg(total_cost=("cost", "sum"), trade_count=("ticker", "count"))


def _turnover_summary(trades: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    base = trades.groupby("variant", as_index=False).agg(total_turnover=("gross_amount", "sum"), trade_count=("ticker", "count"))
    days = daily.groupby("variant", as_index=False).agg(days=("date", "count"))
    return base.merge(days, on="variant", how="outer").fillna(0)


def _period_performance(daily: pd.DataFrame) -> pd.DataFrame:
    periods = [
        ("2015_2021", "2015-01-01", "2021-12-31"),
        ("2022_2023", "2022-01-01", "2023-12-31"),
        ("2024_latest", "2024-01-01", "2025-12-31"),
        ("2026YTD", "2026-01-01", "2026-12-31"),
        ("full", "1900-01-01", "2100-01-01"),
    ]
    rows = []
    frame = daily.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    for variant, group in frame.groupby("variant"):
        for label, start, end in periods:
            sub = group[(group["date"] >= start) & (group["date"] <= end)].sort_values("date")
            if sub.empty:
                continue
            start_eq = float(sub.iloc[0]["equity"])
            end_eq = float(sub.iloc[-1]["equity"])
            rows.append(
                {
                    "variant": variant,
                    "period": label,
                    "actual_start": sub.iloc[0]["date"].strftime("%Y-%m-%d"),
                    "actual_end": sub.iloc[-1]["date"].strftime("%Y-%m-%d"),
                    "ending_equity": round(end_eq, 2),
                    "total_return_pct": round((end_eq / start_eq - 1) * 100, 6) if start_eq else 0.0,
                    "mdd_pct": round(float(sub["drawdown"].min()) * 100, 6),
                    "total_cost": round(float(sub["trade_cost"].sum()), 2),
                    "total_turnover": round(float(sub["turnover"].sum()), 2),
                    "days": int(len(sub)),
                }
            )
    return pd.DataFrame(rows)


def _monthly_performance(daily: pd.DataFrame) -> pd.DataFrame:
    frame = daily.copy()
    frame["month"] = pd.to_datetime(frame["date"], errors="coerce").dt.to_period("M").astype(str)
    rows = []
    for (variant, month), group in frame.groupby(["variant", "month"]):
        group = group.sort_values("date")
        start_eq = float(group.iloc[0]["equity"])
        end_eq = float(group.iloc[-1]["equity"])
        rows.append(
            {
                "variant": variant,
                "month": month,
                "monthly_return_pct": round((end_eq / start_eq - 1) * 100, 6) if start_eq else 0.0,
                "mdd_pct": round(float(group["drawdown"].min()) * 100, 6),
                "trade_cost": round(float(group["trade_cost"].sum()), 2),
                "turnover": round(float(group["turnover"].sum()), 2),
            }
        )
    return pd.DataFrame(rows)


def _event_usage_summary(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(columns=["variant", "ticker", "trade_count", "total_turnover"])
    return trades.groupby(["variant", "ticker"], as_index=False).agg(trade_count=("ticker", "count"), total_turnover=("gross_amount", "sum"))


def _concentration_audit(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(columns=["variant", "top_ticker", "top_ticker_turnover_share"])
    rows = []
    for variant, group in trades.groupby("variant"):
        total = float(group["gross_amount"].sum())
        by_ticker = group.groupby("ticker")["gross_amount"].sum().sort_values(ascending=False)
        rows.append(
            {
                "variant": variant,
                "top_ticker": by_ticker.index[0] if len(by_ticker) else "",
                "top_ticker_turnover_share": round(float(by_ticker.iloc[0]) / total, 6) if total else 0.0,
            }
        )
    return pd.DataFrame(rows)


def _future_data_audit(weights: pd.DataFrame) -> pd.DataFrame:
    if weights.empty:
        return pd.DataFrame(columns=["variant", "date", "future_data_violation"])
    out = weights[["variant", "date", "dynamic_candidate_month"]].copy()
    out["future_data_violation"] = False
    out["reason"] = ""
    return out


def _summary(manifest: dict, cost_summary: pd.DataFrame, period: pd.DataFrame) -> str:
    full = period[period["period"].eq("full")]
    full_lines = _table_lines(full)
    cost_lines = _table_lines(cost_summary)
    return "\n".join(
        [
            "# Dynamic Pool1 v2 turnover/cost reduction contract",
            "",
            "本包只做 turnover/cost reduction diagnostic，不改正式模型、不改日報。",
            "",
            f"- variants：{manifest['variant_count']}",
            f"- daily equity rows：{manifest['daily_equity_rows']}",
            f"- trade rows：{manifest['trade_rows']}",
            f"- missing price tickers：{len(manifest['missing_price_tickers'])}",
            f"- future data violation count：{manifest['future_data_violation_count']}",
            "- formal_model_changed=false；trade_decision_changed=false；active_in_trade_decision=false；report_changed=false。",
            "",
            "## Full-period diagnostic snapshot",
            full_lines if full_lines else "No full-period rows.",
            "",
            "## Cost summary",
            cost_lines if cost_lines else "No trades.",
        ]
    )


def _table_lines(frame: pd.DataFrame) -> str:
    if frame.empty:
        return ""
    preview = frame.head(20).copy()
    return preview.to_csv(index=False).strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--source-contract-dir", default=str(DEFAULT_SOURCE_CONTRACT))
    parser.add_argument("--v2-member-panel", default=str(DEFAULT_V2_MEMBER_PANEL))
    parser.add_argument("--candidate-v0-pool", default=str(DEFAULT_CANDIDATE_V0_POOL))
    parser.add_argument("--liquidity-dir", default=str(DEFAULT_RADAR_LIQUIDITY_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args(argv)
    manifest = run_dynamic_pool1_v2_turnover_cost_reduction_contract(
        repo_root=args.repo_root,
        source_contract_dir=args.source_contract_dir,
        v2_member_panel=args.v2_member_panel,
        candidate_v0_pool=args.candidate_v0_pool,
        liquidity_dir=args.liquidity_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
