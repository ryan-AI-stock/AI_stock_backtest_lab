from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from backtest_lab.costs import TaiwanCostModel


DEFAULT_OUTPUT_DIR = "outputs/pool3_radar_weighted_overlay_replay_20260623"


def run_pool3_radar_weighted_overlay_replay(
    *,
    weighted_basket_daily: str | Path,
    output_dir: str | Path,
    initial_cash: float = 1_000_000,
) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    source = pd.read_csv(weighted_basket_daily).fillna("")
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    run_log: list[dict[str, str]] = []

    def log(step: str, status: str, detail: str = "") -> None:
        run_log.append(
            {
                "timestamp": pd.Timestamp.now(tz="Asia/Taipei").strftime("%Y-%m-%d %H:%M:%S%z"),
                "step": step,
                "status": status,
                "detail": detail,
            }
        )
        pd.DataFrame(run_log).to_csv(output / "run_log.csv", index=False, encoding="utf-8-sig")
        (output / "current_step.txt").write_text(step, encoding="utf-8")

    log("replay_variants", "started", f"rows={len(source)}")
    for variant, frame in source.groupby("variant", dropna=False):
        daily, summary = _replay_variant(str(variant), frame, initial_cash=initial_cash)
        rows.extend(daily)
        summaries.append(summary)
    daily_frame = pd.DataFrame(rows)
    summary_frame = pd.DataFrame(summaries)
    daily_frame.to_csv(output / "pool3_radar_weighted_overlay_formal_daily.csv", index=False, encoding="utf-8-sig")
    summary_frame.to_csv(output / "pool3_radar_weighted_overlay_summary.csv", index=False, encoding="utf-8-sig")
    metadata = {
        "model": "pool3_radar_weighted_overlay_replay_v1",
        "status": "completed",
        "active_in_trade_decision": False,
        "formal_model_changed": False,
        "source": str(weighted_basket_daily),
        "initial_cash": initial_cash,
        "rebalance_policy": "rebalance when target basket composition or weights change; integer-share accounting with Taiwan cost model",
        "rows": {"daily": len(daily_frame), "summary": len(summary_frame)},
        "outputs": {
            "daily": "pool3_radar_weighted_overlay_formal_daily.csv",
            "summary": "pool3_radar_weighted_overlay_summary.csv",
        },
    }
    (output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "completed.txt").write_text("completed\n", encoding="utf-8")
    log("completed", "completed", str(output.resolve()))
    (output / "current_step.txt").write_text("completed\n", encoding="utf-8")
    return output


def _replay_variant(variant: str, frame: pd.DataFrame, *, initial_cash: float) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cash = float(initial_cash)
    shares: dict[str, int] = {}
    cost_model = TaiwanCostModel()
    previous_signature: tuple[tuple[str, float], ...] | None = None
    daily_rows: list[dict[str, Any]] = []
    equity_values: list[float] = []
    running_max = initial_cash
    total_transaction_cost = 0
    rebalance_days = 0
    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    for date, day in frame.sort_values("date").groupby("date", dropna=False):
        date_text = pd.Timestamp(date).strftime("%Y-%m-%d")
        prices = _prices(day)
        targets = _target_weights(day, prices)
        signature = tuple(sorted((ticker, round(weight, 6)) for ticker, weight in targets.items()))
        equity_before = _equity(cash, shares, prices)
        transaction_cost = 0
        turnover = 0.0
        action = "hold"
        if signature != previous_signature and (targets or shares):
            action = "rebalance"
            cash, shares, cost, gross = _rebalance(
                cash=cash,
                shares=shares,
                prices=prices,
                targets=targets,
                equity=equity_before,
                cost_model=cost_model,
                date=date_text,
            )
            transaction_cost += cost
            turnover += gross
            previous_signature = signature
            total_transaction_cost += cost
            rebalance_days += 1
        elif signature != previous_signature:
            previous_signature = signature
        equity_after = _equity(cash, shares, prices)
        running_max = max(running_max, equity_after)
        equity_values.append(equity_after)
        if shares:
            for row_index, (ticker, share_count) in enumerate(sorted(shares.items())):
                price = prices.get(ticker, 0.0)
                value = share_count * price
                daily_rows.append(
                    _daily_row(
                        date_text=date_text,
                        variant=variant,
                        ticker=ticker,
                        weight=(value / equity_after) if equity_after else 0.0,
                        shares=share_count,
                        price=price,
                        action=action,
                        cash=cash,
                        position_value=value,
                        transaction_cost=transaction_cost if row_index == 0 else 0,
                        equity=equity_after,
                        drawdown=(equity_after / running_max - 1) if running_max else 0.0,
                        theme=_theme_for(day, ticker),
                    )
                )
        else:
            daily_rows.append(
                _daily_row(
                    date_text=date_text,
                    variant=variant,
                    ticker="cash",
                    weight=1.0,
                    shares=0,
                    price=0.0,
                    action="hold_cash",
                    cash=cash,
                    position_value=0.0,
                    transaction_cost=0,
                    equity=equity_after,
                    drawdown=(equity_after / running_max - 1) if running_max else 0.0,
                    theme="cash",
                )
            )
    start = equity_values[0] if equity_values else initial_cash
    end = equity_values[-1] if equity_values else initial_cash
    summary = {
        "variant": variant,
        "start_equity": round(start, 2),
        "final_equity": round(end, 2),
        "total_return_pct": round((end / start - 1) * 100, 4) if start else 0.0,
        "max_drawdown_pct": round(min((value / max(equity_values[: idx + 1]) - 1) for idx, value in enumerate(equity_values)) * 100, 4) if equity_values else 0.0,
        "rebalance_days": rebalance_days,
        "total_transaction_cost": total_transaction_cost,
        "rows": len(daily_rows),
    }
    return daily_rows, summary


def _rebalance(
    *,
    cash: float,
    shares: dict[str, int],
    prices: dict[str, float],
    targets: dict[str, float],
    equity: float,
    cost_model: TaiwanCostModel,
    date: str,
) -> tuple[float, dict[str, int], int, float]:
    del date
    total_cost = 0
    total_gross = 0.0
    shares = dict(shares)
    for ticker, current_shares in list(shares.items()):
        price = prices.get(ticker)
        if not price:
            continue
        target_value = equity * targets.get(ticker, 0.0)
        current_value = current_shares * price
        if current_value <= target_value:
            continue
        sell_shares = min(current_shares, int((current_value - target_value) // price))
        if sell_shares <= 0:
            continue
        gross = sell_shares * price
        cost = cost_model.sell_cost(gross, _asset_type(ticker))
        cash += gross - cost
        shares[ticker] = current_shares - sell_shares
        total_cost += cost
        total_gross += gross
    for ticker, weight in sorted(targets.items()):
        price = prices.get(ticker)
        if not price or weight <= 0:
            continue
        current_value = shares.get(ticker, 0) * price
        target_value = equity * weight
        shortage = target_value - current_value
        if shortage <= price:
            continue
        buy_shares = int(shortage // price)
        while buy_shares > 0:
            gross = buy_shares * price
            cost = cost_model.buy_cost(gross)
            if gross + cost <= cash:
                break
            buy_shares -= 1
        if buy_shares <= 0:
            continue
        gross = buy_shares * price
        cost = cost_model.buy_cost(gross)
        cash -= gross + cost
        shares[ticker] = shares.get(ticker, 0) + buy_shares
        total_cost += cost
        total_gross += gross
    shares = {ticker: count for ticker, count in shares.items() if count > 0}
    return cash, shares, total_cost, total_gross


def _daily_row(
    *,
    date_text: str,
    variant: str,
    ticker: str,
    weight: float,
    shares: int,
    price: float,
    action: str,
    cash: float,
    position_value: float,
    transaction_cost: int,
    equity: float,
    drawdown: float,
    theme: str,
) -> dict[str, Any]:
    return {
        "date": date_text,
        "variant": variant,
        "pool3_formal_vote": "weighted_basket",
        "holding_ticker": ticker,
        "holding_name": ticker,
        "theme": theme,
        "weight": round(weight, 8),
        "shares": int(shares),
        "fill_action": action,
        "fill_price": round(price, 6),
        "cash": round(cash, 2),
        "position_value": round(position_value, 2),
        "transaction_cost": int(transaction_cost),
        "equity": round(equity, 2),
        "drawdown": round(drawdown, 8),
        "data_status": "formal_weighted_overlay_replay",
    }


def _prices(day: pd.DataFrame) -> dict[str, float]:
    prices: dict[str, float] = {}
    for row in day.to_dict(orient="records"):
        ticker = str(row.get("ticker") or "").strip()
        if not ticker or ticker == "cash":
            continue
        price = _number(row.get("close"))
        if price > 0:
            prices[ticker] = price
    return prices


def _target_weights(day: pd.DataFrame, prices: dict[str, float]) -> dict[str, float]:
    weights: dict[str, float] = {}
    for row in day.to_dict(orient="records"):
        ticker = str(row.get("ticker") or "").strip()
        if not ticker or ticker == "cash" or ticker not in prices:
            continue
        weight = _number(row.get("weight"))
        if weight > 0:
            weights[ticker] = weights.get(ticker, 0.0) + weight
    total = sum(weights.values())
    if total > 1.0:
        weights = {ticker: weight / total for ticker, weight in weights.items()}
    return weights


def _theme_for(day: pd.DataFrame, ticker: str) -> str:
    subset = day[day["ticker"].astype(str) == ticker]
    if subset.empty:
        return ""
    return str(subset.iloc[0].get("theme") or "")


def _equity(cash: float, shares: dict[str, int], prices: dict[str, float]) -> float:
    return cash + sum(count * prices.get(ticker, 0.0) for ticker, count in shares.items())


def _asset_type(ticker: str) -> str:
    symbol = ticker.split(".")[0]
    return "etf" if symbol in {"0050", "00631L"} else "stock"


def _number(value: object) -> float:
    try:
        text = str(value).replace(",", "").replace("%", "").strip()
        return float(text) if text else 0.0
    except (TypeError, ValueError):
        return 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay Pool3 Radar weighted overlay with holdings and Taiwan transaction costs.")
    parser.add_argument("--weighted-basket-daily", required=True)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--initial-cash", type=float, default=1_000_000)
    args = parser.parse_args()
    output = run_pool3_radar_weighted_overlay_replay(
        weighted_basket_daily=args.weighted_basket_daily,
        output_dir=args.output_dir,
        initial_cash=args.initial_cash,
    )
    print(f"OUTPUT_DIR={output.resolve()}")


if __name__ == "__main__":
    main()
