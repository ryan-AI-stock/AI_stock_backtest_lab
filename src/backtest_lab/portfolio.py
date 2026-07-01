from __future__ import annotations

from dataclasses import dataclass

from backtest_lab.costs import COST_MODEL_VERSION, TaiwanCostModel


@dataclass
class Trade:
    date: str
    ticker: str
    action: str
    shares: int
    price: float
    gross_amount: float
    costs: int
    cash_after: float
    reason: str
    buy_fee: int = 0
    sell_fee: int = 0
    securities_transaction_tax: int = 0
    total_transaction_cost: int = 0
    cost_model_version: str = COST_MODEL_VERSION


@dataclass
class Position:
    ticker: str
    shares: int = 0


class Portfolio:
    def __init__(self, cash: float, cost_model: TaiwanCostModel) -> None:
        self.cash = float(cash)
        self.cost_model = cost_model
        self.positions: dict[str, Position] = {}
        self.trades: list[Trade] = []

    def shares(self, ticker: str) -> int:
        position = self.positions.get(ticker)
        return position.shares if position else 0

    def current_ticker(self) -> str | None:
        active = [ticker for ticker, position in self.positions.items() if position.shares > 0]
        if not active:
            return None
        if len(active) > 1:
            raise ValueError(f"Expected single active position, got {active}")
        return active[0]

    def buy_max(self, date: str, ticker: str, asset_type: str, price: float, reason: str) -> Trade | None:
        shares = self._max_affordable_shares(price)
        if shares <= 0:
            return None
        gross = shares * price
        breakdown = self.cost_model.buy_cost_breakdown(gross)
        costs = breakdown["total_transaction_cost"]
        if gross + costs > self.cash:
            raise ValueError("Internal buy sizing error: gross plus costs exceeds cash")
        self.cash -= gross + costs
        self.positions[ticker] = Position(ticker=ticker, shares=self.shares(ticker) + shares)
        trade = Trade(
            date=date,
            ticker=ticker,
            action="buy",
            shares=shares,
            price=price,
            gross_amount=gross,
            costs=costs,
            cash_after=self.cash,
            reason=reason,
            buy_fee=breakdown["buy_fee"],
            sell_fee=breakdown["sell_fee"],
            securities_transaction_tax=breakdown["securities_transaction_tax"],
            total_transaction_cost=breakdown["total_transaction_cost"],
        )
        self.trades.append(trade)
        return trade

    def sell_all(self, date: str, ticker: str, asset_type: str, price: float, reason: str) -> Trade | None:
        shares = self.shares(ticker)
        if shares <= 0:
            return None
        gross = shares * price
        breakdown = self.cost_model.sell_cost_breakdown(gross, asset_type)
        costs = breakdown["total_transaction_cost"]
        self.cash += gross - costs
        self.positions[ticker] = Position(ticker=ticker, shares=0)
        trade = Trade(
            date=date,
            ticker=ticker,
            action="sell",
            shares=shares,
            price=price,
            gross_amount=gross,
            costs=costs,
            cash_after=self.cash,
            reason=reason,
            buy_fee=breakdown["buy_fee"],
            sell_fee=breakdown["sell_fee"],
            securities_transaction_tax=breakdown["securities_transaction_tax"],
            total_transaction_cost=breakdown["total_transaction_cost"],
        )
        self.trades.append(trade)
        return trade

    def credit_dividend(self, date: str, ticker: str, dividend_per_share: float) -> Trade | None:
        shares = self.shares(ticker)
        if shares <= 0 or dividend_per_share <= 0:
            return None
        amount = shares * dividend_per_share
        self.cash += amount
        trade = Trade(
            date=date,
            ticker=ticker,
            action="dividend",
            shares=shares,
            price=dividend_per_share,
            gross_amount=amount,
            costs=0,
            cash_after=self.cash,
            reason="cash_dividend",
        )
        self.trades.append(trade)
        return trade

    def market_value(self, close_prices: dict[str, float]) -> float:
        value = self.cash
        for ticker, position in self.positions.items():
            if position.shares <= 0:
                continue
            value += position.shares * close_prices[ticker]
        return value

    def _max_affordable_shares(self, price: float) -> int:
        if price <= 0:
            raise ValueError(f"Invalid buy price: {price}")
        shares = int(self.cash // price)
        while shares > 0:
            gross = shares * price
            if gross + self.cost_model.buy_cost(gross) <= self.cash:
                return shares
            shares -= 1
        return 0
