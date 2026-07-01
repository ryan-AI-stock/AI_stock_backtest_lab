import unittest

import test_paths  # noqa: F401

from backtest_lab.costs import TaiwanCostModel


class TaiwanCostModelTest(unittest.TestCase):
    def test_broker_fee_uses_minimum_fee(self) -> None:
        model = TaiwanCostModel()
        self.assertEqual(model.broker_fee(1000), 20)

    def test_broker_fee_rounds_standard_rate(self) -> None:
        model = TaiwanCostModel()
        self.assertEqual(model.broker_fee(1_000_000), 1425)

    def test_stock_sell_cost_includes_stock_tax(self) -> None:
        model = TaiwanCostModel()
        self.assertEqual(model.sell_cost(1_000_000, "stock"), 4425)

    def test_etf_sell_cost_includes_etf_tax(self) -> None:
        model = TaiwanCostModel()
        self.assertEqual(model.sell_cost(1_000_000, "etf"), 2425)

    def test_cost_breakdown_splits_fee_and_tax(self) -> None:
        model = TaiwanCostModel()

        buy = model.buy_cost_breakdown(1_000_000)
        stock_sell = model.sell_cost_breakdown(1_000_000, "stock")
        etf_sell = model.sell_cost_breakdown(1_000_000, "etf")

        self.assertEqual(buy["buy_fee"], 1425)
        self.assertEqual(buy["securities_transaction_tax"], 0)
        self.assertEqual(stock_sell["sell_fee"], 1425)
        self.assertEqual(stock_sell["securities_transaction_tax"], 3000)
        self.assertEqual(stock_sell["total_transaction_cost"], 4425)
        self.assertEqual(etf_sell["sell_fee"], 1425)
        self.assertEqual(etf_sell["securities_transaction_tax"], 1000)
        self.assertEqual(etf_sell["total_transaction_cost"], 2425)


if __name__ == "__main__":
    unittest.main()
