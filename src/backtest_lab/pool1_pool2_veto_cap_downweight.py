from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from backtest_lab.costs import TaiwanCostModel
from backtest_lab.data import load_price_csv


DEFAULT_FORMAL_REPLAY_DIR = "outputs/stock_pool_formal_daily_replay_pit_pool2_daily_final_combined_20260624"
DEFAULT_PRICE_CACHE_DIR = "backtest_cache/stock_pool_triad_v1_corrected"
DEFAULT_OUTPUT_DIR = "outputs/pool1_pool2_veto_cap_downweight_panels_20260626"
INITIAL_CASH = 1_000_000.0
HORIZONS = (20, 60, 120)
BENCHMARKS = {"0050": "0050.TW", "00631L": "00631L.TW"}


@dataclass(frozen=True)
class VariantSpec:
    variant: str
    pool2_policy: str
    cap_00631l: float | None = None
    downweight_00631l: float | None = None
    fallback: str = "cash"
    confirmation_days: int = 0


VARIANT_SPECS = [
    VariantSpec("pool1_pool2_veto_no_cap", "hard_veto"),
    VariantSpec("pool1_pool2_veto_00631L_cap_40", "hard_veto", cap_00631l=0.40),
    VariantSpec("pool1_pool2_veto_00631L_cap_30", "hard_veto", cap_00631l=0.30),
    VariantSpec("pool1_pool2_veto_00631L_downweight_50", "hard_veto", downweight_00631l=0.50),
    VariantSpec("pool1_pool2_veto_00631L_to_0050_fallback", "hard_veto", cap_00631l=0.0, fallback="0050.TW"),
    VariantSpec("pool1_pool2_veto_00631L_to_cash_fallback", "hard_veto", cap_00631l=0.0, fallback="cash"),
    VariantSpec("pool1_pool2_disagree_hard_veto", "hard_veto"),
    VariantSpec("pool1_pool2_disagree_downweight_50", "downweight_50"),
    VariantSpec("pool1_pool2_disagree_confirmation_1", "confirmation", confirmation_days=1),
    VariantSpec("pool1_pool2_disagree_confirmation_2", "confirmation", confirmation_days=2),
    VariantSpec("pool1_pool2_disagree_warning_only", "warning_only"),
    VariantSpec("pool1_primary_no_overlay", "warning_only"),
    VariantSpec("combined_cap40_hard_veto", "hard_veto", cap_00631l=0.40),
    VariantSpec("combined_cap40_downweight", "downweight_50", cap_00631l=0.40),
    VariantSpec("combined_cap40_confirmation1", "confirmation", cap_00631l=0.40, confirmation_days=1),
    VariantSpec("combined_no_cap_downweight", "downweight_50"),
    VariantSpec("combined_no_cap_warning_only", "warning_only"),
]


def run_pool1_pool2_veto_cap_downweight(
    *,
    formal_replay_dir: str | Path = DEFAULT_FORMAL_REPLAY_DIR,
    price_cache_dir: str | Path = DEFAULT_PRICE_CACHE_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    initial_cash: float = INITIAL_CASH,
) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    run_log: list[dict[str, str]] = []

    def log(step: str, status: str, detail: str = "") -> None:
        run_log.append({"timestamp": pd.Timestamp.now(tz="Asia/Taipei").strftime("%Y-%m-%d %H:%M:%S%z"), "step": step, "status": status, "detail": detail})
        pd.DataFrame(run_log).to_csv(output / "run_log.csv", index=False, encoding="utf-8-sig")
        (output / "current_step.txt").write_text(step, encoding="utf-8")

    try:
        root = Path(formal_replay_dir)
        log("load_inputs", "started", str(root))
        decision = pd.read_csv(root / "formal_three_pool_decision_panel.csv").fillna("")
        _validate_decision(decision)
        prices = _load_prices(_needed_tickers(decision), Path(price_cache_dir))

        log("simulate_variants", "started", "")
        all_daily: list[pd.DataFrame] = []
        all_trades: list[pd.DataFrame] = []
        all_events: list[pd.DataFrame] = []
        for spec in VARIANT_SPECS:
            target_panel = _build_target_weights(decision, spec)
            daily, trades, events = _simulate_weighted_variant(target_panel, prices, spec, initial_cash)
            all_daily.append(daily)
            all_trades.append(trades)
            all_events.append(events)
        daily_equity = pd.concat(all_daily, ignore_index=True)
        trade_ledger = pd.concat(all_trades, ignore_index=True)
        event_panel = pd.concat(all_events, ignore_index=True)

        log("build_reports", "started", "")
        perf = _period_performance(daily_equity)
        cap_true = _cap_true_equity(daily_equity)
        exposure = _exposure_by_variant(daily_equity)
        contribution = _contribution_by_variant(daily_equity)
        veto_forward = _vetoed_event_forward_outcome(event_panel, prices)
        missed = _missed_upside_attribution(veto_forward)
        oos = _oos_walk_forward(daily_equity)
        leave_one = _leave_one_period(daily_equity)
        concentration = _contribution_concentration(daily_equity)
        execution = _execution_diagnostics(daily_equity)

        log("write_outputs", "started", "")
        _variant_matrix().to_csv(output / "variant_parameter_matrix.csv", index=False, encoding="utf-8-sig")
        daily_equity.to_csv(output / "daily_equity_by_variant.csv", index=False, encoding="utf-8-sig")
        trade_ledger.to_csv(output / "trade_ledger_by_variant.csv", index=False, encoding="utf-8-sig")
        perf.to_csv(output / "period_performance_by_variant.csv", index=False, encoding="utf-8-sig")
        cap_true.to_csv(output / "00631L_cap_true_equity.csv", index=False, encoding="utf-8-sig")
        exposure.to_csv(output / "00631L_exposure_by_variant.csv", index=False, encoding="utf-8-sig")
        contribution.to_csv(output / "00631L_contribution_by_variant.csv", index=False, encoding="utf-8-sig")
        event_panel.to_csv(output / "pool2_disagreement_variant_events.csv", index=False, encoding="utf-8-sig")
        event_panel.to_csv(output / "veto_downweight_confirmation_event_panel.csv", index=False, encoding="utf-8-sig")
        veto_forward.to_csv(output / "vetoed_event_forward_outcome_by_variant.csv", index=False, encoding="utf-8-sig")
        missed.to_csv(output / "missed_upside_attribution.csv", index=False, encoding="utf-8-sig")
        oos.to_csv(output / "oos_walk_forward_by_variant.csv", index=False, encoding="utf-8-sig")
        leave_one.to_csv(output / "leave_one_period_by_variant.csv", index=False, encoding="utf-8-sig")
        concentration.to_csv(output / "contribution_concentration_by_variant.csv", index=False, encoding="utf-8-sig")
        execution.to_csv(output / "execution_diagnostics_by_variant.csv", index=False, encoding="utf-8-sig")
        (output / "pool1_pool2_veto_cap_downweight_summary_zh.md").write_text(_summary_markdown(perf, exposure), encoding="utf-8")
        manifest = {
            "schema_version": 1,
            "task_id": "TASK-BACKTEST-CORE-POOL1-POOL2-VETO-CAP-DOWNWEIGHT-PANELS-001",
            "model": "pool1_pool2_veto_cap_downweight",
            "status": "completed",
            "formal_replay_dir": str(root),
            "price_cache_dir": str(price_cache_dir),
            "start_date": str(decision["date"].iloc[0]),
            "latest_complete_common_date": str(decision["date"].iloc[-1]),
            "formal_model_changed": False,
            "trade_decision_changed": False,
            "formal_absorption_ready": False,
            "pool3_shadow_used_as_formal": False,
            "report_only_labels_used_in_performance": False,
            "rr_partial_switch_used_in_performance": False,
            "uses_forward_return_as_rule": False,
            "valuation_used": False,
            "h3_used": False,
            "cap_performance_recomputed": True,
            "same_date_range_for_variants": _same_date_range(daily_equity),
            "same_cost_model_for_variants": True,
        }
        (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        pd.DataFrame([{"status": "completed", "output_dir": str(output.resolve())}]).to_csv(output / "completed.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(columns=["step", "error"]).to_csv(output / "failed.csv", index=False, encoding="utf-8-sig")
        log("completed", "completed", str(output.resolve()))
        (output / "current_step.txt").write_text("completed\n", encoding="utf-8")
        return output
    except Exception as exc:
        pd.DataFrame([{"step": "run_pool1_pool2_veto_cap_downweight", "error": str(exc)}]).to_csv(output / "failed.csv", index=False, encoding="utf-8-sig")
        log("failed", "failed", str(exc))
        raise


def _build_target_weights(decision: pd.DataFrame, spec: VariantSpec) -> pd.DataFrame:
    rows = []
    pool1_history: list[str] = []
    for item in decision.to_dict(orient="records"):
        pool1 = _text(item.get("pool1_vote"))
        pool2 = _text(item.get("pool2_vote"))
        disagreement = bool(pool1 and pool2 and pool2 != pool1)
        confirmed = _confirmed(pool1_history, pool1, spec.confirmation_days)
        target_weights, reason = _target_weights(pool1, disagreement, confirmed, spec)
        rows.append({**item, "variant": spec.variant, "target_weights": json.dumps(target_weights, ensure_ascii=False), "pool2_disagreement": disagreement, "event_reason": reason, "pool3_shadow_used_as_formal": False})
        pool1_history.append(pool1)
    return pd.DataFrame(rows)


def _target_weights(pool1: str, disagreement: bool, confirmed: bool, spec: VariantSpec) -> tuple[dict[str, float], str]:
    if not pool1:
        return {}, "pool1_no_target"
    if disagreement:
        if spec.pool2_policy == "hard_veto":
            return {}, "pool2_disagrees_hard_veto"
        if spec.pool2_policy == "downweight_50":
            weights = {pool1: 0.50}
            return _apply_00631l_policy(weights, spec), "pool2_disagrees_downweight_50"
        if spec.pool2_policy == "confirmation" and not confirmed:
            return {}, f"pool2_disagrees_confirmation_{spec.confirmation_days}_not_met"
    weights = {pool1: 1.0}
    return _apply_00631l_policy(weights, spec), "pool1_primary"


def _apply_00631l_policy(weights: dict[str, float], spec: VariantSpec) -> dict[str, float]:
    if "00631L.TW" not in weights:
        return weights
    current = weights["00631L.TW"]
    cap = spec.cap_00631l if spec.cap_00631l is not None else None
    if spec.downweight_00631l is not None:
        cap = min(cap if cap is not None else 1.0, spec.downweight_00631l)
    if cap is None or current <= cap:
        return weights
    remainder = current - cap
    output = dict(weights)
    if cap <= 0:
        output.pop("00631L.TW", None)
    else:
        output["00631L.TW"] = cap
    if spec.fallback != "cash":
        output[spec.fallback] = output.get(spec.fallback, 0.0) + remainder
    return output


def _confirmed(history: list[str], target: str, days: int) -> bool:
    if days <= 0:
        return True
    if not target or len(history) < days:
        return False
    return all(item == target for item in history[-days:])


def _simulate_weighted_variant(targets: pd.DataFrame, prices: dict[str, pd.Series], spec: VariantSpec, initial_cash: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cost_model = TaiwanCostModel()
    cash = float(initial_cash)
    shares: dict[str, int] = {}
    rows: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    running_max = initial_cash
    for item in targets.to_dict(orient="records"):
        date = pd.Timestamp(item["date"]).strftime("%Y-%m-%d")
        weights = json.loads(item["target_weights"]) if _text(item.get("target_weights")) else {}
        close_prices = _prices_for_date(prices, date)
        equity_before = cash + sum(qty * close_prices.get(ticker, 0.0) for ticker, qty in shares.items())
        trade_cost = 0
        turnover = 0.0
        action = "hold"
        current_tickers = {ticker for ticker, qty in shares.items() if qty > 0}
        desired_tickers = set(weights)
        if current_tickers != desired_tickers or _needs_rebalance(shares, weights, close_prices, equity_before):
            for ticker in list(shares):
                if shares.get(ticker, 0) <= 0:
                    continue
                price = close_prices.get(ticker)
                if price is None:
                    continue
                gross = shares[ticker] * price
                costs = cost_model.sell_cost(gross, _asset_type(ticker))
                cash += gross - costs
                trade_cost += costs
                turnover += gross
                trades.append(_trade_row(spec.variant, date, ticker, "sell", shares[ticker], price, gross, costs, cash, "rebalance"))
                shares[ticker] = 0
            for ticker, weight in weights.items():
                price = close_prices.get(ticker)
                if price is None or weight <= 0:
                    continue
                budget = max(0.0, equity_before * float(weight))
                qty = int(budget // price)
                while qty > 0 and qty * price + cost_model.buy_cost(qty * price) > cash:
                    qty -= 1
                if qty <= 0:
                    continue
                gross = qty * price
                costs = cost_model.buy_cost(gross)
                cash -= gross + costs
                trade_cost += costs
                turnover += gross
                shares[ticker] = shares.get(ticker, 0) + qty
                trades.append(_trade_row(spec.variant, date, ticker, "buy", qty, price, gross, costs, cash, "target_weights"))
            action = "rebalance"
        equity = cash + sum(qty * close_prices.get(ticker, 0.0) for ticker, qty in shares.items())
        running_max = max(running_max, equity)
        rows.append({"variant": spec.variant, "date": date, "period": item.get("period", ""), "target_weights": item.get("target_weights", "{}"), "position_ticker": _position_label(shares), "cash": round(cash, 2), "equity": round(equity, 2), "drawdown": round(equity / running_max - 1, 8), "turnover": round(turnover, 2), "transaction_cost": trade_cost, "action": action, "data_status": "cap_downweight_challenger"})
        events.append({"variant": spec.variant, "date": date, "period": item.get("period", ""), "pool1_vote": item.get("pool1_vote", ""), "pool2_vote": item.get("pool2_vote", ""), "pool2_disagreement": item.get("pool2_disagreement", False), "event_reason": item.get("event_reason", ""), "target_weights": item.get("target_weights", "{}"), "uses_forward_return_as_rule": False})
    return pd.DataFrame(rows), pd.DataFrame(trades), pd.DataFrame(events)


def _needs_rebalance(shares: dict[str, int], weights: dict[str, float], prices: dict[str, float], equity: float) -> bool:
    if not weights:
        return any(qty > 0 for qty in shares.values())
    if equity <= 0:
        return False
    for ticker, weight in weights.items():
        current = shares.get(ticker, 0) * prices.get(ticker, 0.0) / equity
        if abs(current - weight) > 0.05:
            return True
    return False


def _variant_matrix() -> pd.DataFrame:
    return pd.DataFrame([{**spec.__dict__} for spec in VARIANT_SPECS])


def _period_performance(daily: pd.DataFrame) -> pd.DataFrame:
    periods = {"2022": ("2022-01-01", "2022-12-31"), "2023": ("2023-01-01", "2023-12-31"), "2024_now": ("2024-01-01", None), "2024_hard_gate": ("2024-01-01", "2024-12-31"), "full": (None, None)}
    rows = []
    frame = daily.copy()
    frame["date_ts"] = pd.to_datetime(frame["date"])
    for variant, group in frame.groupby("variant"):
        for label, (start, end) in periods.items():
            subset = group
            if start:
                subset = subset[subset["date_ts"] >= pd.Timestamp(start)]
            if end:
                subset = subset[subset["date_ts"] <= pd.Timestamp(end)]
            rows.append(_perf_row(variant, label, subset))
    return pd.DataFrame(rows)


def _perf_row(variant: str, label: str, frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {"variant": variant, "period_label": label, "status": "empty"}
    equity = pd.to_numeric(frame["equity"], errors="coerce")
    start = float(equity.iloc[0])
    end = float(equity.iloc[-1])
    return {"variant": variant, "period_label": label, "status": "completed", "start_date": frame["date"].iloc[0], "end_date": frame["date"].iloc[-1], "start_equity": round(start, 2), "final_equity": round(end, 2), "return_pct": round((end / start - 1) * 100, 4) if start else 0.0, "max_drawdown_pct": round(float(pd.to_numeric(frame["drawdown"], errors="coerce").min()) * 100, 4), "trade_days": int(frame["action"].astype(str).ne("hold").sum()), "total_transaction_cost": round(float(pd.to_numeric(frame["transaction_cost"], errors="coerce").sum()), 2)}


def _cap_true_equity(daily: pd.DataFrame) -> pd.DataFrame:
    frame = daily[daily["variant"].astype(str).str.contains("00631L_cap|00631L_downweight|00631L_to_", regex=True)].copy()
    frame["performance_recomputed"] = True
    return frame


def _exposure_by_variant(daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant, group in daily.groupby("variant"):
        active = group[~group["position_ticker"].eq("cash")]
        exposure = group[group["position_ticker"].astype(str).str.contains("00631L.TW", regex=False)]
        rows.append({"variant": variant, "active_days": len(active), "00631L_active_days": len(exposure), "00631L_position_day_share": round(len(exposure) / len(active), 6) if len(active) else 0.0})
    return pd.DataFrame(rows)


def _contribution_by_variant(daily: pd.DataFrame) -> pd.DataFrame:
    frame = daily.copy()
    frame["daily_return"] = frame.groupby("variant")["equity"].pct_change().fillna(0)
    return frame.groupby(["variant", "position_ticker"], dropna=False)["daily_return"].agg(["count", "sum", "mean"]).reset_index()


def _vetoed_event_forward_outcome_by_variant(events: pd.DataFrame, prices: dict[str, pd.Series]) -> pd.DataFrame:
    return _vetoed_event_forward_outcome(events, prices)


def _vetoed_event_forward_outcome(events: pd.DataFrame, prices: dict[str, pd.Series]) -> pd.DataFrame:
    rows = []
    for item in events[events["pool2_disagreement"].map(_truthy)].to_dict(orient="records"):
        date = _text(item.get("date"))
        target = _text(item.get("pool1_vote"))
        row = {"variant": item.get("variant", ""), "date": date, "period": item.get("period", ""), "pool2_disagreement": True, "event_reason": item.get("event_reason", ""), "vetoed_or_downweighted_target": target, "target_weights": item.get("target_weights", ""), "uses_forward_return_as_rule": False}
        for horizon in HORIZONS:
            target_ret = _forward_return(prices.get(target), date, horizon)
            row[f"target_forward_{horizon}d_return"] = _round(target_ret)
            for label, ticker in BENCHMARKS.items():
                row[f"target_excess_vs_{label}_{horizon}d"] = _round(_diff(target_ret, _forward_return(prices.get(ticker), date, horizon)))
            bench0050 = _forward_return(prices.get("0050.TW"), date, horizon)
            row[f"target_excess_vs_0050x2_{horizon}d"] = _round(_diff(target_ret, None if bench0050 is None else bench0050 * 2))
        rows.append(row)
    return pd.DataFrame(rows)


def _missed_upside_attribution(veto_forward: pd.DataFrame) -> pd.DataFrame:
    if veto_forward.empty:
        return pd.DataFrame()
    rows = []
    for variant, group in veto_forward.groupby("variant"):
        rows.append({"variant": variant, "event_count": len(group), "mean_target_forward_20d": _mean(group.get("target_forward_20d_return", pd.Series(dtype=float))), "mean_target_forward_60d": _mean(group.get("target_forward_60d_return", pd.Series(dtype=float))), "mean_target_forward_120d": _mean(group.get("target_forward_120d_return", pd.Series(dtype=float)))})
    return pd.DataFrame(rows)


def _oos_walk_forward(daily: pd.DataFrame) -> pd.DataFrame:
    tests = {"train_2022_2024": ("2022-01-01", "2024-12-31"), "test_2025_2026": ("2025-01-01", None), "train_2022_2023": ("2022-01-01", "2023-12-31"), "test_2024_now": ("2024-01-01", None)}
    rows = []
    frame = daily.copy()
    frame["date_ts"] = pd.to_datetime(frame["date"])
    for variant, group in frame.groupby("variant"):
        for label, (start, end) in tests.items():
            subset = group[group["date_ts"] >= pd.Timestamp(start)]
            if end:
                subset = subset[subset["date_ts"] <= pd.Timestamp(end)]
            rows.append(_perf_row(variant, label, subset))
    return pd.DataFrame(rows)


def _leave_one_period(daily: pd.DataFrame) -> pd.DataFrame:
    frame = daily.copy()
    frame["date_ts"] = pd.to_datetime(frame["date"])
    frame["year"] = frame["date_ts"].dt.year.astype(str)
    rows = []
    for variant, group in frame.groupby("variant"):
        for year in sorted(group["year"].unique()):
            rows.append(_perf_row(variant, f"leave_one_year_{year}", group[~group["year"].eq(year)]))
    return pd.DataFrame(rows)


def _contribution_concentration(daily: pd.DataFrame) -> pd.DataFrame:
    contrib = _contribution_by_variant(daily)
    rows = []
    for variant, group in contrib.groupby("variant"):
        total = group["sum"].abs().sum()
        top = group.sort_values("sum", ascending=False).iloc[0]
        rows.append({"variant": variant, "top_position": top["position_ticker"], "top_positive_contribution": round(float(top["sum"]), 8), "top_abs_contribution_share": round(abs(float(top["sum"])) / total, 6) if total else 0.0})
    return pd.DataFrame(rows)


def _execution_diagnostics(daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant, group in daily.groupby("variant"):
        values = group["position_ticker"].astype(str).tolist()
        rows.append({"variant": variant, "trade_days": int(group["action"].astype(str).ne("hold").sum()), "total_transaction_cost": round(float(pd.to_numeric(group["transaction_cost"], errors="coerce").sum()), 2), "rapid_flip_count": int(_rapid_flip(values).sum())})
    return pd.DataFrame(rows)


def _validate_decision(decision: pd.DataFrame) -> None:
    required = {"period", "date", "pool1_vote", "pool2_vote"}
    missing = sorted(required - set(decision.columns))
    if missing:
        raise ValueError(f"decision panel missing columns: {missing}")


def _needed_tickers(decision: pd.DataFrame) -> list[str]:
    tickers = set(BENCHMARKS.values())
    for col in ("pool1_vote", "pool2_vote", "pool3_vote"):
        if col in decision.columns:
            tickers.update(_text(value) for value in decision[col].tolist() if _text(value))
    tickers.add("0050.TW")
    return sorted(tickers)


def _load_prices(tickers: list[str], cache_dir: Path) -> dict[str, pd.Series]:
    prices = {}
    for ticker in tickers:
        path = cache_dir / f"{ticker.replace('.', '_')}.csv"
        if path.exists():
            frame = load_price_csv(path)
            prices[ticker] = pd.to_numeric(frame["adj_close"], errors="coerce").dropna()
    return prices


def _prices_for_date(prices: dict[str, pd.Series], date: str) -> dict[str, float]:
    result = {}
    for ticker, series in prices.items():
        price = _price_on_or_before(series, date)
        if price is not None:
            result[ticker] = price
    return result


def _price_on_or_before(series: pd.Series, date: str) -> float | None:
    if series.empty:
        return None
    clipped = series.loc[series.index <= pd.Timestamp(date)]
    if clipped.empty:
        return None
    return float(clipped.iloc[-1])


def _forward_return(series: pd.Series | None, date: str, horizon: int) -> float | None:
    if series is None or series.empty:
        return None
    future = series.loc[series.index >= pd.Timestamp(date)]
    if len(future) <= horizon:
        return None
    start = float(future.iloc[0])
    end = float(future.iloc[horizon])
    return end / start - 1 if start else None


def _asset_type(ticker: str) -> str:
    return "etf" if ticker.split(".")[0] in {"0050", "00631L"} else "stock"


def _trade_row(variant: str, date: str, ticker: str, action: str, shares: int, price: float, gross: float, costs: int, cash: float, reason: str) -> dict[str, Any]:
    return {"variant": variant, "date": date, "ticker": ticker, "action": action, "shares": shares, "price": round(price, 4), "gross_amount": round(gross, 2), "costs": costs, "cash_after": round(cash, 2), "reason": reason}


def _position_label(shares: dict[str, int]) -> str:
    active = [ticker for ticker, qty in shares.items() if qty > 0]
    return "cash" if not active else "+".join(sorted(active))


def _same_date_range(daily: pd.DataFrame) -> bool:
    ranges = daily.groupby("variant")["date"].agg(["min", "max", "count"])
    return bool(ranges["min"].nunique() == 1 and ranges["max"].nunique() == 1 and ranges["count"].nunique() == 1)


def _rapid_flip(values: list[str]) -> pd.Series:
    flags = []
    for index, value in enumerate(values):
        future = values[index + 1 : index + 4]
        flags.append(bool(value and value in future and any(item and item != value for item in future)))
    return pd.Series(flags)


def _mean(series: pd.Series) -> Any:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return round(float(values.mean()), 8) if len(values) else ""


def _diff(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return left - right


def _round(value: float | None) -> Any:
    return "" if value is None else round(float(value), 8)


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    return "" if text.lower() == "nan" else text.strip()


def _summary_markdown(perf: pd.DataFrame, exposure: pd.DataFrame) -> str:
    full = perf[perf["period_label"].eq("full")]
    lines = ["# Pool1 + Pool2 Veto Cap / Downweight Panels", "", "本輸出只做 challenger evidence，不改正式模型。", "", "## Full Period", ""]
    for row in full.to_dict(orient="records"):
        lines.append(f"- {row.get('variant')}: return {row.get('return_pct')}%, MDD {row.get('max_drawdown_pct')}%")
    lines.extend(["", "## 00631L Exposure", ""])
    for row in exposure.to_dict(orient="records"):
        lines.append(f"- {row.get('variant')}: share {row.get('00631L_position_day_share')}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Pool1+Pool2 veto cap/downweight challenger panels.")
    parser.add_argument("--formal-replay-dir", default=DEFAULT_FORMAL_REPLAY_DIR)
    parser.add_argument("--price-cache-dir", default=DEFAULT_PRICE_CACHE_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    output = run_pool1_pool2_veto_cap_downweight(formal_replay_dir=args.formal_replay_dir, price_cache_dir=args.price_cache_dir, output_dir=args.output_dir)
    print(f"OUTPUT_DIR={output.resolve()}")


if __name__ == "__main__":
    main()
