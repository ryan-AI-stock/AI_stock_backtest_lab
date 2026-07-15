from __future__ import annotations

import unittest

import pandas as pd

from backtest_lab.ma_price_slope_cooldown_grid_backtest import (
    BUY_RULES,
    SELL_RULES,
    SignalRule,
    add_signals,
    combination_matrix,
    cooldown_complete,
)


class MaPriceSlopeCooldownGridBacktestTests(unittest.TestCase):
    def test_all_36_combinations_are_unique(self) -> None:
        combinations = combination_matrix(7)
        self.assertEqual(len(BUY_RULES), 6)
        self.assertEqual(len(SELL_RULES), 6)
        self.assertEqual(len(combinations), 36)
        self.assertEqual(combinations["strategy"].nunique(), 36)
        self.assertTrue(combinations["strategy"].str.endswith("CD7").all())
        self.assertTrue(combinations["cooldown_days"].eq(7).all())

    def test_signal_uses_independent_ma_and_price_slope_windows(self) -> None:
        dates = pd.date_range("2020-01-01", periods=20, freq="B")
        prices = list(range(1, 21))
        frame = pd.DataFrame({"date": dates, "0050_adj_close": prices, "00631L_adj_close": [20] * 20})
        signaled = add_signals(frame, SignalRule("B", 4, 20), SignalRule("X", 10, 7))
        self.assertTrue(bool(signaled.iloc[-1]["buy_signal"]))
        self.assertFalse(bool(signaled.iloc[-1]["sell_signal"]))

    def test_seven_observation_slope_compares_six_trading_days_ago(self) -> None:
        dates = pd.date_range("2020-01-01", periods=20, freq="B")
        prices = [95] * 12 + [101, 99, 99, 99, 99, 99, 99, 100]
        frame = pd.DataFrame({"date": dates, "0050_adj_close": prices, "00631L_adj_close": [20] * 20})

        signaled = add_signals(frame, SignalRule("B", 4, 7), SignalRule("X", 10, 20))

        self.assertEqual(float(signaled.iloc[-1]["buy_slope"]), 1.0)
        self.assertTrue(bool(signaled.iloc[-1]["buy_signal"]))
        self.assertEqual(prices[-1] - prices[-8], -1)

    def test_twenty_observation_slope_compares_nineteen_trading_days_ago(self) -> None:
        dates = pd.date_range("2020-01-01", periods=20, freq="B")
        prices = [101] + [100] * 18 + [99]
        frame = pd.DataFrame({"date": dates, "0050_adj_close": prices, "00631L_adj_close": [20] * 20})

        signaled = add_signals(frame, SignalRule("B", 4, 7), SignalRule("X", 10, 20))

        self.assertEqual(float(signaled.iloc[-1]["sell_slope"]), -2.0)
        self.assertTrue(bool(signaled.iloc[-1]["sell_signal"]))

    def test_cd5_blocks_five_following_trading_days(self) -> None:
        self.assertFalse(cooldown_complete(11, 10, 5))
        self.assertFalse(cooldown_complete(15, 10, 5))
        self.assertTrue(cooldown_complete(16, 10, 5))


if __name__ == "__main__":
    unittest.main()
