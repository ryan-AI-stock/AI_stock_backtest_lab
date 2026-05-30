from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TaiwanCostModel:
    broker_fee_rate: float = 0.001425
    broker_fee_discount: float = 1.0
    minimum_fee_twd: int = 20
    stock_sell_tax_rate: float = 0.003
    etf_sell_tax_rate: float = 0.001

    def broker_fee(self, gross_amount: float) -> int:
        if gross_amount <= 0:
            return 0
        fee = gross_amount * self.broker_fee_rate * self.broker_fee_discount
        return max(self.minimum_fee_twd, round(fee))

    def sell_tax(self, gross_amount: float, asset_type: str) -> int:
        if gross_amount <= 0:
            return 0
        rate = self.etf_sell_tax_rate if asset_type == "etf" else self.stock_sell_tax_rate
        return round(gross_amount * rate)

    def buy_cost(self, gross_amount: float) -> int:
        return self.broker_fee(gross_amount)

    def sell_cost(self, gross_amount: float, asset_type: str) -> int:
        return self.broker_fee(gross_amount) + self.sell_tax(gross_amount, asset_type)

