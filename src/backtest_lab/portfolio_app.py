from __future__ import annotations

import argparse
import json
import math
import threading
from datetime import date
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from backtest_lab.config import load_config


DEFAULT_USER_ID = "default"
DEFAULT_STORE_PATH = "work/portfolio_app/portfolio_store.json"
DEFAULT_SIGNAL_ROOT = "outputs/frozen_strategy_monitor"


class PortfolioStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def get_user(self, user_id: str = DEFAULT_USER_ID) -> dict:
        with self.lock:
            data = self._load()
            return json.loads(json.dumps(_ensure_user(data, user_id), ensure_ascii=False))

    def replace_portfolio(self, *, user_id: str, cash_twd: float, positions: list[dict]) -> dict:
        if cash_twd < 0:
            raise ValueError("可用現金不可小於 0。")
        cleaned: dict[str, dict] = {}
        for item in positions:
            ticker = str(item.get("ticker", "")).strip()
            shares = int(item.get("shares", 0))
            avg_cost = float(item.get("avg_cost", 0))
            if not ticker or shares < 0 or avg_cost < 0:
                raise ValueError("持倉資料格式錯誤。")
            if shares > 0:
                cleaned[ticker] = {"shares": shares, "avg_cost": round(avg_cost, 4)}
        with self.lock:
            data = self._load()
            user = _ensure_user(data, user_id)
            user["cash_twd"] = round(float(cash_twd), 2)
            user["positions"] = cleaned
            self._save(data)
            return json.loads(json.dumps(user, ensure_ascii=False))

    def record_trade(self, *, user_id: str, trade: dict, asset_types: dict[str, str], cost_model) -> dict:
        ticker = str(trade.get("ticker", "")).strip()
        side = str(trade.get("side", "")).lower()
        shares = int(trade.get("shares", 0))
        price = float(trade.get("price", 0))
        trade_date = str(trade.get("date") or date.today().isoformat())
        if ticker not in asset_types or side not in {"buy", "sell"} or shares <= 0 or price <= 0:
            raise ValueError("成交資料格式錯誤。")
        gross = shares * price
        supplied_costs = trade.get("costs_twd")
        if supplied_costs in (None, ""):
            costs = cost_model.buy_cost(gross) if side == "buy" else cost_model.sell_cost(gross, asset_types[ticker])
        else:
            costs = max(0, round(float(supplied_costs)))

        with self.lock:
            data = self._load()
            user = _ensure_user(data, user_id)
            position = user["positions"].get(ticker, {"shares": 0, "avg_cost": 0.0})
            if side == "buy":
                total_outflow = gross + costs
                if total_outflow > user["cash_twd"] + 1e-6:
                    raise ValueError("可用現金不足，無法登錄這筆買進。")
                previous_cost = position["shares"] * position["avg_cost"]
                new_shares = position["shares"] + shares
                position = {
                    "shares": new_shares,
                    "avg_cost": round((previous_cost + total_outflow) / new_shares, 4),
                }
                user["cash_twd"] = round(user["cash_twd"] - total_outflow, 2)
                realized_pnl = None
            else:
                if shares > position["shares"]:
                    raise ValueError("賣出股數大於目前持有股數。")
                user["cash_twd"] = round(user["cash_twd"] + gross - costs, 2)
                realized_pnl = round((price - position["avg_cost"]) * shares - costs, 2)
                position["shares"] -= shares
                if position["shares"] == 0:
                    position["avg_cost"] = 0.0

            if position["shares"] > 0:
                user["positions"][ticker] = position
            else:
                user["positions"].pop(ticker, None)
            user["trades"].append(
                {
                    "date": trade_date,
                    "ticker": ticker,
                    "side": side,
                    "shares": shares,
                    "price": round(price, 4),
                    "gross_amount_twd": round(gross, 2),
                    "costs_twd": costs,
                    "realized_pnl_twd": realized_pnl,
                }
            )
            self._save(data)
            return json.loads(json.dumps(user, ensure_ascii=False))

    def _load(self) -> dict:
        if not self.path.exists():
            return {"schema_version": 1, "users": {}}
        return json.loads(self.path.read_text(encoding="utf-8"))

    def _save(self, data: dict) -> None:
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(self.path)


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
            "portfolio": _portfolio_summary(user, {}),
            "recommendations": [],
        }
    close_prices = {ticker: float(price) for ticker, price in signal.get("close_prices", {}).items()}
    portfolio = _portfolio_summary(user, close_prices)
    recommendations = _recommendations(user, signal, close_prices, asset_types, cost_model, portfolio["total_value_twd"])
    return {
        "user": user,
        "signal": signal,
        "portfolio": portfolio,
        "recommendations": recommendations,
    }


def _portfolio_summary(user: dict, close_prices: dict[str, float]) -> dict:
    rows = []
    market_value = 0.0
    unrealized = 0.0
    for ticker, position in sorted(user["positions"].items()):
        price = close_prices.get(ticker, position["avg_cost"])
        value = position["shares"] * price
        pnl = (price - position["avg_cost"]) * position["shares"]
        market_value += value
        unrealized += pnl
        rows.append(
            {
                "ticker": ticker,
                "shares": position["shares"],
                "avg_cost": position["avg_cost"],
                "reference_price": round(price, 4),
                "market_value_twd": round(value, 2),
                "unrealized_pnl_twd": round(pnl, 2),
            }
        )
    total = user["cash_twd"] + market_value
    return {
        "cash_twd": round(user["cash_twd"], 2),
        "market_value_twd": round(market_value, 2),
        "total_value_twd": round(total, 2),
        "unrealized_pnl_twd": round(unrealized, 2),
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


def _ensure_user(data: dict, user_id: str) -> dict:
    users = data.setdefault("users", {})
    if user_id not in users:
        users[user_id] = {
            "user_id": user_id,
            "display_name": "主要使用者",
            "cash_twd": 0.0,
            "positions": {},
            "trades": [],
        }
    return users[user_id]


def create_handler(*, store: PortfolioStore, signal_root: str, asset_types: dict[str, str], cost_model):
    html = Path(__file__).with_name("portfolio_app.html").read_text(encoding="utf-8")

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            path = urlparse(self.path).path
            if path == "/":
                self._send(html, "text/html; charset=utf-8")
                return
            if path == "/api/state":
                user = store.get_user()
                signal = load_latest_signal(signal_root)
                self._json(build_dashboard(user, signal, asset_types, cost_model))
                return
            self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
            path = urlparse(self.path).path
            try:
                payload = self._read_json()
                user_id = str(payload.get("user_id") or DEFAULT_USER_ID)
                if path == "/api/portfolio":
                    store.replace_portfolio(
                        user_id=user_id,
                        cash_twd=float(payload.get("cash_twd", 0)),
                        positions=list(payload.get("positions", [])),
                    )
                elif path == "/api/trades":
                    store.record_trade(
                        user_id=user_id,
                        trade=payload,
                        asset_types=asset_types,
                        cost_model=cost_model,
                    )
                else:
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                user = store.get_user(user_id)
                signal = load_latest_signal(signal_root)
                self._json(build_dashboard(user, signal, asset_types, cost_model))
            except (ValueError, TypeError, json.JSONDecodeError) as error:
                self._json({"error": str(error)}, status=HTTPStatus.BAD_REQUEST)

        def _read_json(self) -> dict:
            length = int(self.headers.get("Content-Length", "0"))
            return json.loads(self.rfile.read(length).decode("utf-8"))

        def _json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
            self._send(json.dumps(payload, ensure_ascii=False), "application/json; charset=utf-8", status)

        def _send(self, payload: str, content_type: str, status: HTTPStatus = HTTPStatus.OK) -> None:
            body = payload.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args) -> None:
            print(f"portfolio_app: {format % args}")

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the private best-strategy portfolio workspace.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--store", default=DEFAULT_STORE_PATH)
    parser.add_argument("--signal-root", default=DEFAULT_SIGNAL_ROOT)
    parser.add_argument("--config", default="configs/ep05_universe.json")
    parser.add_argument("--group-id", default="group_c_0050_00631l_plus_mega_caps")
    args = parser.parse_args()

    config = load_config(args.config)
    group = config.group_by_id(args.group_id)
    asset_types = {asset.ticker: asset.asset_type for asset in group.assets}
    handler = create_handler(
        store=PortfolioStore(args.store),
        signal_root=args.signal_root,
        asset_types=asset_types,
        cost_model=config.cost_model,
    )
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"PORTFOLIO_APP_URL=http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
