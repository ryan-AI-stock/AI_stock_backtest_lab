from __future__ import annotations

import json
import threading
from datetime import date
from pathlib import Path

from backtest_lab.portfolio_app_settings import DEFAULT_USER_ID


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
