from __future__ import annotations

import json
import math
from pathlib import Path


def load_latest_signal(signal_root: str | Path) -> dict | None:
    root = Path(signal_root)
    candidates = sorted(root.glob("*/frozen_strategy_signal.json"), reverse=True)
    for path in candidates:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") == "ready" and payload.get("signal"):
            return payload["signal"]
    return None


def build_dashboard(user: dict, signal: dict | None, asset_types: dict[str, str], cost_model) -> dict:
    if signal is None:
        return {
            "user": user,
            "signal": None,
            "portfolio": _portfolio_summary(user, {}, asset_types, cost_model),
            "recommendations": [],
        }
    close_prices = {ticker: float(price) for ticker, price in signal.get("close_prices", {}).items()}
    portfolio = _portfolio_summary(user, close_prices, asset_types, cost_model)
    recommendations = _recommendations(user, signal, close_prices, asset_types, cost_model, portfolio["total_value_twd"])
    return {
        "user": user,
        "signal": signal,
        "portfolio": portfolio,
        "recommendations": recommendations,
    }


def _portfolio_summary(user: dict, close_prices: dict[str, float], asset_types: dict[str, str], cost_model) -> dict:
    rows = []
    market_value = 0.0
    unrealized = 0.0
    net_unrealized = 0.0
    estimated_exit_costs = 0.0
    for ticker, position in sorted(user["positions"].items()):
        price = close_prices.get(ticker, position["avg_cost"])
        value = position["shares"] * price
        pnl = (price - position["avg_cost"]) * position["shares"]
        exit_cost = cost_model.sell_cost(value, asset_types.get(ticker, "stock"))
        net_pnl = pnl - exit_cost
        market_value += value
        unrealized += pnl
        net_unrealized += net_pnl
        estimated_exit_costs += exit_cost
        rows.append(
            {
                "ticker": ticker,
                "shares": position["shares"],
                "avg_cost": position["avg_cost"],
                "reference_price": round(price, 4),
                "market_value_twd": round(value, 2),
                "unrealized_pnl_twd": round(pnl, 2),
                "estimated_exit_costs_twd": round(exit_cost, 2),
                "net_unrealized_pnl_twd": round(net_pnl, 2),
            }
        )
    total = user["cash_twd"] + market_value
    return {
        "cash_twd": round(user["cash_twd"], 2),
        "market_value_twd": round(market_value, 2),
        "total_value_twd": round(total, 2),
        "unrealized_pnl_twd": round(unrealized, 2),
        "estimated_exit_costs_twd": round(estimated_exit_costs, 2),
        "net_unrealized_pnl_twd": round(net_unrealized, 2),
        "positions": rows,
    }


def _recommendations(
    user: dict,
    signal: dict,
    close_prices: dict[str, float],
    asset_types: dict[str, str],
    cost_model,
    total_value: float,
) -> list[dict]:
    target = signal["target_ticker"]
    target_exposure = float(signal["target_exposure"])
    positions = user["positions"]
    rows: list[dict] = []
    projected_cash = float(user["cash_twd"])

    for ticker, position in sorted(positions.items()):
        if ticker == target:
            continue
        price = close_prices.get(ticker)
        if not price or position["shares"] <= 0:
            continue
        gross = position["shares"] * price
        costs = cost_model.sell_cost(gross, asset_types.get(ticker, "stock"))
        projected_cash += gross - costs
        rows.append(
            _recommendation_row(
                ticker=ticker,
                action="sell",
                shares=position["shares"],
                price=price,
                target_exposure=0.0,
                reason="模型目標已轉往其他標的或現金。",
            )
        )

    if target == "cash" or target not in close_prices:
        return rows

    price = close_prices[target]
    current_shares = int(positions.get(target, {}).get("shares", 0))
    desired_shares = max(0, math.floor(total_value * target_exposure / price))
    delta = desired_shares - current_shares
    if delta > 0:
        affordable = _max_affordable_shares(projected_cash, price, cost_model)
        suggested = min(delta, affordable)
        rows.append(
            _recommendation_row(
                ticker=target,
                action="buy" if suggested > 0 else "hold",
                shares=suggested,
                price=price,
                target_exposure=target_exposure,
                reason="依模型目標比例與完成其他建議調整後的可用現金估算。",
                desired_shares=desired_shares,
                immediately_buyable_shares=_max_affordable_shares(float(user["cash_twd"]), price, cost_model),
            )
        )
    elif delta < 0:
        rows.append(
            _recommendation_row(
                ticker=target,
                action="sell",
                shares=abs(delta),
                price=price,
                target_exposure=target_exposure,
                reason="目前持股超過模型目標比例。",
                desired_shares=desired_shares,
            )
        )
    else:
        rows.append(
            _recommendation_row(
                ticker=target,
                action="hold",
                shares=0,
                price=price,
                target_exposure=target_exposure,
                reason="目前股數已接近模型目標比例。",
                desired_shares=desired_shares,
            )
        )
    return rows


def _recommendation_row(
    *,
    ticker: str,
    action: str,
    shares: int,
    price: float,
    target_exposure: float,
    reason: str,
    desired_shares: int | None = None,
    immediately_buyable_shares: int | None = None,
) -> dict:
    return {
        "ticker": ticker,
        "action": action,
        "shares": shares,
        "reference_price": round(price, 4),
        "estimated_gross_twd": round(shares * price, 2),
        "target_exposure": target_exposure,
        "desired_total_shares": desired_shares,
        "immediately_buyable_shares": immediately_buyable_shares,
        "reason": reason,
    }


def _max_affordable_shares(cash: float, price: float, cost_model) -> int:
    low = 0
    high = max(0, int(cash // price))
    while low < high:
        middle = (low + high + 1) // 2
        if middle * price + cost_model.buy_cost(middle * price) <= cash:
            low = middle
        else:
            high = middle - 1
    return low
