from __future__ import annotations

import unittest

import pandas as pd
import test_paths  # noqa: F401

from backtest_lab.bear_defense_backtest import _risk_on


class BearDefenseBacktestTest(unittest.TestCase):
    def test_long_ma_rule_turns_off_when_price_below_trend(self) -> None:
        dates = pd.date_range("2021-01-01", periods=260, freq="D")
        values = [100.0] * 259 + [80.0]
        prices = pd.DataFrame({"adj_close": values}, index=dates)

        self.assertFalse(_risk_on(prices, dates[-1], "ma250"))

    def test_panic_rule_stays_on_without_drawdown_pressure(self) -> None:
        dates = pd.date_range("2021-01-01", periods=130, freq="D")
        values = [100.0] * 130
        prices = pd.DataFrame({"adj_close": values}, index=dates)

        self.assertTrue(_risk_on(prices, dates[-1], "panic_ma120_dd10"))


if __name__ == "__main__":
    unittest.main()
