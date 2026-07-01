from __future__ import annotations

from dataclasses import dataclass


COST_MODEL_VERSION = "taiwan_standard_fee_tax_v1"


@dataclass(frozen=True)
class TaiwanCostModel:
    broker_fee_rate: float = 0.001425
    broker_fee_discount: float = 1.0
    minimum_fee_twd: int = 20
    stock_sell_tax_rate: float = 0.003
    etf_sell_tax_rate: float = 0.001

    def metadata(self) -> dict[str, float | int | str | bool]:
        return cost_model_metadata(self)

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
        return int(self.buy_cost_breakdown(gross_amount)["total_transaction_cost"])

    def sell_cost(self, gross_amount: float, asset_type: str) -> int:
        return int(self.sell_cost_breakdown(gross_amount, asset_type)["total_transaction_cost"])

    def buy_cost_breakdown(self, gross_amount: float) -> dict[str, int]:
        fee = self.broker_fee(gross_amount)
        return {
            "buy_fee": fee,
            "sell_fee": 0,
            "securities_transaction_tax": 0,
            "total_transaction_cost": fee,
        }

    def sell_cost_breakdown(self, gross_amount: float, asset_type: str) -> dict[str, int]:
        fee = self.broker_fee(gross_amount)
        tax = self.sell_tax(gross_amount, asset_type)
        return {
            "buy_fee": 0,
            "sell_fee": fee,
            "securities_transaction_tax": tax,
            "total_transaction_cost": fee + tax,
        }


def cost_model_metadata(model: TaiwanCostModel | None = None) -> dict[str, float | int | str | bool]:
    active = model or TaiwanCostModel()
    return {
        "cost_model_version": COST_MODEL_VERSION,
        "broker_fee_rate": active.broker_fee_rate,
        "broker_fee_discount": active.broker_fee_discount,
        "minimum_fee_twd": active.minimum_fee_twd,
        "stock_sell_tax_rate": active.stock_sell_tax_rate,
        "etf_sell_tax_rate": active.etf_sell_tax_rate,
        "broker_fee_applies_on": "buy_and_sell",
        "securities_transaction_tax_applies_on": "sell_only",
        "etf_and_stock_tax_split": True,
        "yuanta_actual_discount_known": active.broker_fee_discount != 1.0,
        "cost_model_boundary_zh": (
            "台股標準成本口徑：買進與賣出皆扣券商手續費，賣出另扣證券交易稅；"
            "ETF 賣出稅率 0.1%，個股賣出稅率 0.3%。若未提供元大實際折扣，先用未折扣標準手續費 0.1425%。"
        ),
    }
