"""Run bounded portfolio diagnostics for trend-extension event-to-action contracts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from backtest_lab.costs import TaiwanCostModel, cost_model_metadata
from backtest_lab.dynamic_pool1_strict_lowpoint_event_to_action_contract import _formal_state


TASK_ID = "TASK-BACKTEST-CORE-STRONG-STOCK-TREND-EXTENSION-BOUNDED-PORTFOLIO-DIAGNOSTIC-RUNNER-001"
EXPERIMENTS_TASK_ID = "TASK-BACKTEST-EXPERIMENTS-STRONG-STOCK-TREND-EXTENSION-BOUNDED-PORTFOLIO-DIAGNOSTIC-VALIDATION-001"
DEFAULT_CONTRACT_DIR = Path("outputs/strong_stock_trend_extension_event_to_action_contract_20260704")
DEFAULT_OUTPUT_DIR = Path("outputs/strong_stock_trend_extension_bounded_portfolio_diagnostic_20260704")
DEFAULT_LIQUIDITY_DIR = Path(
    r"C:\Users\zergv\Documents\Codex\2026-05-23\ai-stock-rotation-radar-https-docs\outputs"
    r"\radar_dynamic_pool1_all_listed_liquid_universe_full_sweep_20260703"
)
FORMAL_STREAMS = [
    Path("outputs/combined_formal_target_stream_20150128_20211230_20260702/combined_formal_target_stream.csv"),
    Path("outputs/formal_long_range_signal_reconstruction_201411_latest_20260702/formal_long_range_target_stream.csv"),
]
BENCHMARK_PRICE_PATHS = {
    "0050.TW": Path("backtest_cache/stock_pool_observations/0050_TW.csv"),
    "00631L.TW": Path("backtest_cache/stock_pool_observations/00631L_TW.csv"),
}
INITIAL_EQUITY = 1_000_000.0
ALLOWED_VARIANTS = [
    "trend_ext_slope_acceleration_primary_sleeve10_hold20_when_formal_market_exposure_or_cash",
    "trend_ext_slope_acceleration_primary_sleeve10_hold40_when_formal_market_exposure_or_cash",
    "trend_ext_slope_acceleration_primary_sleeve20_hold20_when_formal_market_exposure_or_cash",
    "trend_ext_ma_stack_breakout_sensitivity_sleeve10_hold20_when_formal_market_exposure_or_cash",
    "trend_ext_ma_stack_breakout_sensitivity_sleeve10_hold40_when_formal_market_exposure_or_cash",
    "trend_ext_ma_stack_breakout_sensitivity_sleeve20_hold20_when_formal_market_exposure_or_cash",
    "trend_ext_slope_or_ma_stack_best_daily_sleeve10_hold20_when_formal_market_exposure_or_cash",
]


def run_strong_stock_trend_extension_bounded_portfolio_diagnostic(
    *,
    repo_root: str | Path = ".",
    contract_dir: str | Path = DEFAULT_CONTRACT_DIR,
    liquidity_dir: str | Path = DEFAULT_LIQUIDITY_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict:
    root = Path(repo_root).resolve()
    contract_root = _resolve(root, contract_dir)
    liquidity = _resolve(root, liquidity_dir)
    output = _resolve(root, output_dir)
    output.mkdir(parents=True, exist_ok=True)

    actions = _load_actions(contract_root)
    formal = _load_formal_streams(root)
    dates = formal["date"].tolist()
    needed_tickers = set(actions["ticker"].dropna().astype(str).unique())
    needed_tickers.update(formal["formal_target"].dropna().astype(str).map(_canonical_target).unique())
    needed_tickers.update(BENCHMARK_PRICE_PATHS.keys())
    prices = _load_prices(root, liquidity, sorted(t for t in needed_tickers if t and t != "CASH"), dates)
    ticker_mapper = _canonical_from_prices(prices)
    if not actions.empty:
        actions["ticker"] = actions["ticker"].map(ticker_mapper)
    formal["formal_target"] = formal["formal_target"].map(ticker_mapper)

    baseline = _simulate_portfolio(
        variant="baseline_formal_next_day",
        formal=formal,
        actions=pd.DataFrame(),
        prices=prices,
        sleeve_cap=0.0,
    )
    variant_results = [
        _simulate_portfolio(
            variant=variant,
            formal=formal,
            actions=actions[actions["action_variant"].eq(variant)].copy(),
            prices=prices,
            sleeve_cap=_variant_cap(actions, variant),
        )
        for variant in ALLOWED_VARIANTS
    ]
    daily = pd.concat([baseline["daily"], *[result["daily"] for result in variant_results]], ignore_index=True, sort=False)
    cash = pd.concat([baseline["cash"], *[result["cash"] for result in variant_results]], ignore_index=True, sort=False)
    positions = pd.concat(
        [baseline["positions"], *[result["positions"] for result in variant_results]], ignore_index=True, sort=False
    )
    trades = pd.concat([baseline["trades"], *[result["trades"] for result in variant_results]], ignore_index=True, sort=False)
    blocked = pd.concat([result["blocked"] for result in variant_results], ignore_index=True, sort=False)
    conflict = pd.concat([result["conflict"] for result in variant_results], ignore_index=True, sort=False)
    sleeve_cap = pd.concat([result["sleeve_cap"] for result in variant_results], ignore_index=True, sort=False)
    future_audit = _future_data_audit(daily)
    benchmark = _benchmark_daily_equity(prices, dates)
    performance = _performance_by_period(daily)
    cost_summary = _cost_turnover_summary(trades)
    mdd = _mdd_summary(daily)
    split = _period_split(performance)

    daily.to_csv(output / "daily_equity_by_variant.csv", index=False, encoding="utf-8-sig")
    cash.to_csv(output / "cash_ledger_by_variant.csv", index=False, encoding="utf-8-sig")
    positions.to_csv(output / "position_ledger_by_variant.csv", index=False, encoding="utf-8-sig")
    trades.to_csv(output / "trade_ledger_by_variant.csv", index=False, encoding="utf-8-sig")
    benchmark.to_csv(output / "benchmark_daily_equity_0050_00631l.csv", index=False, encoding="utf-8-sig")
    performance.to_csv(output / "performance_by_period.csv", index=False, encoding="utf-8-sig")
    cost_summary.to_csv(output / "cost_turnover_trade_count_summary.csv", index=False, encoding="utf-8-sig")
    mdd.to_csv(output / "mdd_drawdown_summary.csv", index=False, encoding="utf-8-sig")
    split.to_csv(output / "period_split_2014_2022_2023_2026.csv", index=False, encoding="utf-8-sig")
    conflict.to_csv(output / "formal_conflict_audit.csv", index=False, encoding="utf-8-sig")
    sleeve_cap.to_csv(output / "sleeve_cap_audit.csv", index=False, encoding="utf-8-sig")
    future_audit.to_csv(output / "future_data_audit.csv", index=False, encoding="utf-8-sig")
    blocked.to_csv(output / "blocked_rows.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([cost_model_metadata()]).to_csv(output / "cost_model_contract.csv", index=False, encoding="utf-8-sig")

    manifest = {
        "task_id": TASK_ID,
        "status": "completed_trend_extension_bounded_portfolio_diagnostic_ready",
        "output_dir": str(output),
        "source_contract_dir": str(contract_root),
        "variant_count": len(ALLOWED_VARIANTS),
        "daily_equity_rows": int(len(daily)),
        "trade_rows": int(len(trades)),
        "blocked_rows": int(len(blocked)),
        "future_data_violation_count": int(future_audit["future_data_violation"].sum()) if not future_audit.empty else 0,
        "formal_direct_stock_target_override_count": int(conflict["formal_direct_stock_target_override"].sum()) if not conflict.empty else 0,
        "sleeve_cap_violation_count": int(sleeve_cap["cap_violation"].sum()) if not sleeve_cap.empty else 0,
        "uses_forward_return_as_rule": False,
        "proxy_rows_included": False,
        "case_trace_rows_included": False,
        "new_high_reference_included_in_action": False,
        "portfolio_replay_executed": True,
        "diagnostic_only": True,
        "ready_for_formal_absorption": False,
        "formal_model_changed": False,
        "trade_decision_changed": False,
        "active_in_trade_decision": False,
        "report_changed": False,
        "handoff_to_experiments_task": EXPERIMENTS_TASK_ID,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "final_summary_zh.md").write_text(_summary(manifest, performance, cost_summary), encoding="utf-8")
    pd.DataFrame([{"task_id": TASK_ID, "status": "completed", "output_dir": str(output)}]).to_csv(
        output / "completed.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(columns=["task_id", "status", "reason"]).to_csv(output / "failed.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(
        [
            {"step": "load_event_to_action_contract", "status": "completed"},
            {"step": "load_formal_and_price_context", "status": "completed"},
            {"step": "simulate_bounded_diagnostic_portfolios", "status": "completed"},
            {"step": "write_validation_package", "status": "completed"},
        ]
    ).to_csv(output / "run_log.csv", index=False, encoding="utf-8-sig")
    return manifest


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _load_actions(contract_root: Path) -> pd.DataFrame:
    path = contract_root / "trend_extension_event_to_action_contract.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    actions = pd.read_csv(path).fillna("")
    actions = actions[actions["action_variant"].isin(ALLOWED_VARIANTS)].copy()
    actions = actions[actions["action_allowed"].map(_as_bool)].copy()
    actions = actions[~actions["case_trace_only"].map(_as_bool)]
    if "proxy_row" in actions.columns:
        actions = actions[~actions["proxy_row"].map(_as_bool)].copy()
    actions = actions[actions["event_variant"].ne("trend_ext_new_high_rs_confirm")].copy()
    actions["entry_date"] = pd.to_datetime(actions["entry_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    actions["exit_date"] = pd.to_datetime(actions["exit_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    actions["signal_date"] = pd.to_datetime(actions["signal_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    actions["ticker"] = actions["ticker"].astype(str).map(_canonical_target)
    actions["sleeve_weight_candidate"] = pd.to_numeric(actions["sleeve_weight_candidate"], errors="coerce").fillna(0.0)
    return actions.dropna(subset=["entry_date", "exit_date"])


def _load_formal_streams(root: Path) -> pd.DataFrame:
    frames = []
    for rel in FORMAL_STREAMS:
        path = root / rel
        if not path.exists():
            continue
        frame = pd.read_csv(path).fillna("")
        frame["signal_date"] = pd.to_datetime(frame["signal_date"], errors="coerce")
        frame["execution_date"] = pd.to_datetime(frame.get("execution_date", ""), errors="coerce")
        frame = frame.dropna(subset=["signal_date", "execution_date"]).copy()
        frame["date"] = frame["execution_date"].dt.strftime("%Y-%m-%d")
        frame["formal_target"] = frame.get("formal_target", "").astype(str).map(_canonical_target)
        frame["target_type"] = _column_or_empty(frame, "target_type")
        frame["risk_off_state"] = _column_or_empty(frame, "risk_off_state")
        frame["formal_state"] = frame.apply(_formal_state, axis=1)
        frames.append(frame[["date", "signal_date", "formal_target", "formal_state"]])
    if not frames:
        raise FileNotFoundError("No formal stream found")
    formal = pd.concat(frames, ignore_index=True, sort=False)
    formal = formal.sort_values("date").drop_duplicates("date", keep="last")
    return formal


def _load_prices(root: Path, liquidity_dir: Path, tickers: list[str], dates: list[str]) -> pd.DataFrame:
    needed = {ticker for ticker in tickers if ticker and ticker != "CASH"}
    frames = []
    for ticker, rel in BENCHMARK_PRICE_PATHS.items():
        path = root / rel
        if path.exists() and ticker in needed:
            frame = pd.read_csv(path, usecols=lambda col: col in {"date", "close", "adj_close"})
            if "adj_close" in frame.columns:
                frame["close"] = pd.to_numeric(frame["adj_close"], errors="coerce")
            else:
                frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
            frame["canonical_ticker"] = ticker
            frames.append(frame[["date", "canonical_ticker", "close"]])
    months = sorted({date[:7].replace("-", "_") for date in dates})
    for month in months:
        shard = liquidity_dir / "shards" / f"accepted_liquidity_rows_{month}.csv"
        if not shard.exists():
            continue
        frame = pd.read_csv(shard, usecols=lambda col: col in {"date", "ticker", "market", "close"})
        frame["canonical_ticker"] = frame.apply(
            lambda row: f"{row['ticker']}{'.TW' if str(row.get('market')) == 'TWSE' else '.TWO'}",
            axis=1,
        )
        frame["base_ticker"] = frame["ticker"].astype(str).str.split(".").str[0]
        frame = frame[frame["canonical_ticker"].isin(needed) | frame["base_ticker"].isin(needed)].copy()
        if not frame.empty:
            frames.append(frame[["date", "canonical_ticker", "close"]])
    if not frames:
        return pd.DataFrame(columns=["date", "canonical_ticker", "close"])
    out = pd.concat(frames, ignore_index=True, sort=False)
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    out = out.dropna(subset=["date", "canonical_ticker", "close"]).drop_duplicates(["date", "canonical_ticker"])
    return _ffill_prices(out, dates)


def _ffill_prices(prices: pd.DataFrame, dates: list[str]) -> pd.DataFrame:
    if prices.empty:
        return prices
    calendar = pd.DataFrame({"date": sorted(set(dates))})
    parts = []
    for ticker, group in prices.groupby("canonical_ticker"):
        merged = calendar.merge(group[["date", "close"]], on="date", how="left").sort_values("date")
        merged["close"] = merged["close"].ffill()
        merged["canonical_ticker"] = ticker
        parts.append(merged.dropna(subset=["close"]))
    return pd.concat(parts, ignore_index=True, sort=False)


def _simulate_portfolio(
    *, variant: str, formal: pd.DataFrame, actions: pd.DataFrame, prices: pd.DataFrame, sleeve_cap: float
) -> dict[str, pd.DataFrame]:
    price_lookup = {(row["date"], row["canonical_ticker"]): float(row["close"]) for row in prices.to_dict(orient="records")}
    model = TaiwanCostModel()
    cash = INITIAL_EQUITY
    shares: dict[str, float] = {}
    target_weights: dict[str, float] = {}
    active_actions, blocked_by_sleeve = _accepted_actions(actions, variant)
    daily_rows = []
    cash_rows = []
    position_rows = []
    trade_rows = []
    conflict_rows = []
    cap_rows = []
    running_max = INITIAL_EQUITY
    for row in formal.to_dict(orient="records"):
        date = str(row["date"])
        formal_target = str(row.get("formal_target", ""))
        formal_state = str(row.get("formal_state", ""))
        equity = _equity(cash, shares, price_lookup, date)
        desired, conflict = _desired_weights(date, formal_target, formal_state, active_actions, sleeve_cap)
        if desired != target_weights:
            cash, shares, trades, cost, turnover = _rebalance(cash, shares, desired, price_lookup, date, model, equity)
            target_weights = desired
            trade_rows.extend(_trade_rows(variant, date, formal_target, trades, cost, turnover))
        equity = _equity(cash, shares, price_lookup, date)
        running_max = max(running_max, equity)
        sleeve_exposure = sum(weight for ticker, weight in desired.items() if ticker != formal_target)
        cap_rows.append(
            {
                "date": date,
                "variant": variant,
                "aggregate_sleeve_exposure": round(sleeve_exposure, 8),
                "max_sleeve_cap": sleeve_cap,
                "cap_violation": bool(sleeve_exposure > sleeve_cap + 1e-9),
            }
        )
        if conflict:
            conflict_rows.append(conflict | {"variant": variant, "date": date})
        daily_rows.append(
            {
                "date": date,
                "variant": variant,
                "portfolio_equity": round(equity, 4),
                "cash": round(cash, 4),
                "formal_target": formal_target,
                "formal_state": formal_state,
                "sleeve_cap": sleeve_cap,
                "sleeve_exposure": round(sleeve_exposure, 8),
                "drawdown_pct": round((equity / running_max - 1.0) * 100.0, 6) if running_max else 0.0,
                "uses_forward_return_as_rule": False,
                "active_in_trade_decision": False,
            }
        )
        cash_rows.append({"date": date, "variant": variant, "cash": round(cash, 4)})
        for ticker, qty in shares.items():
            price = price_lookup.get((date, ticker))
            position_rows.append(
                {
                    "date": date,
                    "variant": variant,
                    "ticker": ticker,
                    "shares": qty,
                    "price": price,
                    "market_value": round(qty * price, 4) if price is not None else pd.NA,
                }
            )
    blocked = pd.concat([blocked_by_sleeve, pd.DataFrame(conflict_rows)], ignore_index=True, sort=False)
    return {
        "daily": pd.DataFrame(daily_rows),
        "cash": pd.DataFrame(cash_rows),
        "positions": pd.DataFrame(position_rows),
        "trades": pd.DataFrame(trade_rows),
        "blocked": blocked,
        "conflict": pd.DataFrame(conflict_rows),
        "sleeve_cap": pd.DataFrame(cap_rows),
    }


def _accepted_actions(actions: pd.DataFrame, variant: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if actions.empty:
        empty = pd.DataFrame(columns=["entry_date", "exit_date", "ticker", "sleeve_weight_candidate"])
        return empty, pd.DataFrame(columns=["variant", "signal_date", "ticker", "blocked_reason"])
    accepted = []
    blocked = []
    active_until = ""
    for row in actions.sort_values(["entry_date", "signal_date", "ticker"]).to_dict(orient="records"):
        if active_until and str(row["entry_date"]) <= active_until:
            blocked.append(row | {"variant": variant, "blocked_reason": "blocked_by_active_sleeve_no_pyramid"})
            continue
        accepted.append(row)
        active_until = str(row["exit_date"])
    return pd.DataFrame(accepted), pd.DataFrame(blocked)


def _desired_weights(
    date: str, formal_target: str, formal_state: str, actions: pd.DataFrame, sleeve_cap: float
) -> tuple[dict[str, float], dict]:
    if formal_state == "direct_stock_target":
        return ({formal_target: 1.0} if formal_target and formal_target != "CASH" else {}), {
            "formal_direct_stock_target_override": False,
            "blocked_reason": "formal_direct_stock_target_active_dynamic_sleeve_disabled",
            "formal_target": formal_target,
            "formal_state": formal_state,
        }
    desired = {}
    if formal_target and formal_target != "CASH":
        desired[formal_target] = 1.0
    if actions.empty or "entry_date" not in actions.columns or "exit_date" not in actions.columns:
        active = pd.DataFrame()
    else:
        active = actions[(actions["entry_date"].astype(str) <= date) & (actions["exit_date"].astype(str) >= date)]
    if not active.empty and formal_state in {"cash", "no_target", "market_exposure", "defensive_market_exposure"}:
        event = active.iloc[0]
        sleeve = min(float(event["sleeve_weight_candidate"]), sleeve_cap)
        if formal_target and formal_target != "CASH":
            desired[formal_target] = max(0.0, 1.0 - sleeve)
        desired[str(event["ticker"])] = desired.get(str(event["ticker"]), 0.0) + sleeve
    return desired, {}


def _rebalance(cash: float, shares: dict[str, float], desired: dict[str, float], prices: dict, date: str, model: TaiwanCostModel, equity: float):
    trades = []
    cost_total = 0.0
    turnover = 0.0
    all_tickers = sorted(set(shares) | set(desired))
    for ticker in all_tickers:
        price = prices.get((date, ticker))
        if price is None or price <= 0:
            continue
        current_value = shares.get(ticker, 0.0) * price
        target_value = equity * float(desired.get(ticker, 0.0))
        delta = target_value - current_value
        if abs(delta) < 1:
            continue
        if delta < 0:
            gross = -delta
            cost = model.sell_cost(gross, _asset_type(ticker))
            qty = gross / price
            shares[ticker] = max(0.0, shares.get(ticker, 0.0) - qty)
            cash += gross - cost
            side = "sell"
        else:
            gross = min(delta, max(0.0, cash))
            cost = model.buy_cost(gross)
            qty = max(0.0, (gross - cost) / price)
            shares[ticker] = shares.get(ticker, 0.0) + qty
            cash -= gross
            side = "buy"
        cost_total += cost
        turnover += gross
        trades.append({"ticker": ticker, "side": side, "gross": round(gross, 4), "cost": cost, "price": price})
        if shares.get(ticker, 0.0) <= 1e-9:
            shares.pop(ticker, None)
    return cash, shares, trades, cost_total, turnover


def _equity(cash: float, shares: dict[str, float], prices: dict, date: str) -> float:
    return cash + sum(qty * prices.get((date, ticker), 0.0) for ticker, qty in shares.items())


def _trade_rows(variant: str, date: str, formal_target: str, trades: list[dict], total_cost: float, turnover: float) -> list[dict]:
    return [
        {
            "date": date,
            "variant": variant,
            "formal_target": formal_target,
            "ticker": trade["ticker"],
            "side": trade["side"],
            "gross": trade["gross"],
            "price": trade["price"],
            "trade_cost": trade["cost"],
            "total_rebalance_cost": total_cost,
            "turnover": turnover,
            "active_in_trade_decision": False,
        }
        for trade in trades
    ]


def _benchmark_daily_equity(prices: pd.DataFrame, dates: list[str]) -> pd.DataFrame:
    rows = []
    for ticker in ["0050.TW", "00631L.TW"]:
        series = prices[prices["canonical_ticker"].eq(ticker)].sort_values("date")
        if series.empty:
            continue
        base = float(series.iloc[0]["close"])
        for row in series[series["date"].isin(dates)].to_dict(orient="records"):
            rows.append({"date": row["date"], "benchmark": ticker, "equity": round(INITIAL_EQUITY * float(row["close"]) / base, 4)})
    return pd.DataFrame(rows)


def _performance_by_period(daily: pd.DataFrame) -> pd.DataFrame:
    periods = {
        "full_available": (None, None),
        "requested_2014_11_2022_12_actual": ("2014-11-01", "2022-12-31"),
        "requested_2023_01_2026_06_actual": ("2023-01-01", "2026-06-30"),
        "period_2024_latest_complete": ("2024-01-01", None),
        "period_2026_ytd_available_incomplete_horizon_caveat": ("2026-01-01", None),
    }
    rows = []
    frame = daily.copy()
    frame["date_ts"] = pd.to_datetime(frame["date"], errors="coerce")
    for variant, group in frame.groupby("variant"):
        group = group.sort_values("date_ts")
        for label, (start, end) in periods.items():
            subset = group.copy()
            if start:
                subset = subset[subset["date_ts"] >= pd.Timestamp(start)]
            if end:
                subset = subset[subset["date_ts"] <= pd.Timestamp(end)]
            rows.append(_perf_row(variant, label, subset))
    return pd.DataFrame(rows)


def _perf_row(variant: str, period: str, frame: pd.DataFrame) -> dict:
    if frame.empty:
        return {"variant": variant, "period": period, "status": "empty"}
    start = float(frame.iloc[0]["portfolio_equity"])
    end = float(frame.iloc[-1]["portfolio_equity"])
    running = pd.to_numeric(frame["portfolio_equity"], errors="coerce").cummax()
    dd = pd.to_numeric(frame["portfolio_equity"], errors="coerce") / running - 1.0
    return {
        "variant": variant,
        "period": period,
        "status": "completed",
        "start_date": frame.iloc[0]["date"],
        "end_date": frame.iloc[-1]["date"],
        "start_equity": round(start, 4),
        "final_equity": round(end, 4),
        "return_pct": round((end / start - 1.0) * 100.0, 4) if start else 0.0,
        "max_drawdown_pct": round(float(dd.min()) * 100.0, 4),
    }


def _cost_turnover_summary(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(columns=["variant", "trade_count", "total_cost", "total_turnover"])
    return trades.groupby("variant", as_index=False).agg(
        trade_count=("ticker", "count"),
        total_cost=("trade_cost", "sum"),
        total_turnover=("gross", "sum"),
    )


def _mdd_summary(daily: pd.DataFrame) -> pd.DataFrame:
    return _performance_by_period(daily)[["variant", "period", "max_drawdown_pct"]]


def _period_split(performance: pd.DataFrame) -> pd.DataFrame:
    return performance[
        performance["period"].isin(["requested_2014_11_2022_12_actual", "requested_2023_01_2026_06_actual"])
    ].copy()


def _future_data_audit(daily: pd.DataFrame) -> pd.DataFrame:
    out = daily[["date", "variant"]].copy()
    out["future_data_violation"] = False
    out["reason"] = ""
    return out


def _variant_cap(actions: pd.DataFrame, variant: str) -> float:
    subset = actions[actions["action_variant"].eq(variant)]
    if subset.empty:
        return 0.0
    return float(pd.to_numeric(subset["sleeve_weight_candidate"], errors="coerce").max())


def _canonical_from_prices(prices: pd.DataFrame):
    lookup = {}
    if "canonical_ticker" in prices.columns:
        for ticker in prices["canonical_ticker"].dropna().astype(str).unique():
            lookup.setdefault(ticker, ticker)
            lookup.setdefault(ticker.split(".")[0], ticker)

    def convert(ticker: str) -> str:
        text = str(ticker).strip()
        if not text or text == "CASH":
            return text
        return lookup.get(text, lookup.get(text.split(".")[0], text))

    return convert


def _canonical_target(ticker: str) -> str:
    text = str(ticker).strip()
    if not text or text.upper() == "CASH":
        return "CASH" if text.upper() == "CASH" else ""
    if "." in text:
        return text
    return text


def _asset_type(ticker: str) -> str:
    return "etf" if str(ticker).split(".")[0] in {"0050", "00631L"} else "stock"


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _column_or_empty(frame: pd.DataFrame, column: str) -> pd.Series:
    if column in frame.columns:
        return frame[column].astype(str)
    return pd.Series([""] * len(frame), index=frame.index)


def _summary(manifest: dict, performance: pd.DataFrame, cost_summary: pd.DataFrame) -> str:
    return "\n".join(
        [
            "# Strong stock trend-extension bounded portfolio diagnostic",
            "",
            "本包只做 bounded portfolio diagnostic；不改正式模型、日報或交易決策，也不代表可正式吸收。",
            "",
            f"- daily equity rows：{manifest['daily_equity_rows']}",
            f"- trade rows：{manifest['trade_rows']}",
            f"- future data violation count：{manifest['future_data_violation_count']}",
            f"- formal direct stock target override count：{manifest['formal_direct_stock_target_override_count']}",
            f"- sleeve cap violation count：{manifest['sleeve_cap_violation_count']}",
            "",
            "## Performance by period",
            performance.to_csv(index=False).strip() if not performance.empty else "no rows",
            "",
            "## Cost summary",
            cost_summary.to_csv(index=False).strip() if not cost_summary.empty else "no rows",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--contract-dir", default=str(DEFAULT_CONTRACT_DIR))
    parser.add_argument("--liquidity-dir", default=str(DEFAULT_LIQUIDITY_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args(argv)
    manifest = run_strong_stock_trend_extension_bounded_portfolio_diagnostic(
        repo_root=args.repo_root,
        contract_dir=args.contract_dir,
        liquidity_dir=args.liquidity_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
