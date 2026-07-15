from __future__ import annotations

import unittest

import pandas as pd

from backtest_lab.ma_signal_leveraged_etf_backtest import add_signals, simulate


class MaSignalLeveragedEtfBacktestTests(unittest.TestCase):
    def test_buy_signal_uses_only_current_and_prior_closes(self) -> None:
        dates = pd.date_range("2020-01-01", periods=7, freq="B")
        frame = pd.DataFrame({"date": dates, "0050_adj_close": [10, 9, 8, 9, 11, 50, 1], "00631L_adj_close": [20] * 7})
        signaled = add_signals(frame, 3)
        self.assertTrue(bool(signaled.iloc[4]["buy_signal"]))
        changed_future = frame.copy()
        changed_future.loc[5:, "0050_adj_close"] = [-50, -100]
        self.assertTrue(bool(add_signals(changed_future, 3).iloc[4]["buy_signal"]))

    def test_signal_executes_on_next_trading_day(self) -> None:
        dates = pd.date_range("2020-01-01", periods=8, freq="B")
        frame = pd.DataFrame({"date": dates, "0050_adj_close": [10, 9, 8, 9, 11, 12, 13, 14], "00631L_adj_close": [20, 20, 20, 20, 20, 21, 22, 23]})
        result = simulate(frame, period={"period": "T", "requested_start": "2020-01-01", "requested_end": "2020-01-31"}, exit_window=3, after_cost=False)
        buy = result.trades[result.trades["side"].eq("buy")].iloc[0]
        self.assertEqual(buy["signal_date"], dates[4].date().isoformat())
        self.assertEqual(buy["execution_date"], dates[5].date().isoformat())

    def test_exit_requires_price_below_declining_exit_ma(self) -> None:
        dates = pd.date_range("2020-01-01", periods=8, freq="B")
        frame = pd.DataFrame({"date": dates, "0050_adj_close": [10, 11, 12, 13, 14, 10, 11, 12], "00631L_adj_close": [20] * 8})
        signaled = add_signals(frame, 3)
        expected = (signaled["0050_adj_close"] < signaled["exit_ma"]) & (signaled["exit_ma_slope"] < 0)
        self.assertTrue(signaled["sell_signal"].equals(expected.fillna(False)))


if __name__ == "__main__":
    unittest.main()
