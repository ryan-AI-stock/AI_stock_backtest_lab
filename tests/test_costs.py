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


if __name__ == "__main__":
    unittest.main()
