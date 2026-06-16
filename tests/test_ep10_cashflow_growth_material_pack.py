from __future__ import annotations

import unittest

import pandas as pd

import test_paths  # noqa: F401

from backtest_lab.ep10_cashflow_growth_material_pack import repair_split_like_price_jumps, simulate_cashflow_growth_portfolio


class Ep10CashflowGrowthMaterialPackTest(unittest.TestCase):
    def test_0050_dividend_reinvests_and_0056_dividend_is_withdrawn(self) -> None:
        dates = pd.bdate_range("2024-01-02", periods=8)
        prices = {
            "0050.TW": _price_frame(dates, 100.0),
            "0056.TW": _price_frame(dates, 50.0),
        }
        dividends = {
            "0050.TW": pd.Series([0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], index=dates),
            "0056.TW": pd.Series([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], index=dates),
        }

        result = simulate_cashflow_growth_portfolio(
            prices_by_ticker=prices,
            dividend_by_ticker=dividends,
            start_date="2024-01-02",
            end_date="2024-01-11",
            initial_capital=100_000,
            weight_0050=0.5,
            weight_0056=0.5,
            broker_fee_rate=0.0,
            broker_fee_discount=1.0,
            minimum_fee_twd=0,
        )

        treatments = {(row["ticker"], row["treatment"]) for row in result["cashflow_rows"]}
        self.assertIn(("0056.TW", "withdrawn"), treatments)
        self.assertEqual(result["reinvested_dividend"], 0)
        self.assertGreater(result["withdrawn_income"], 0)
        self.assertGreater(
            result["final_total_wealth"],
            result["final_invested_value"],
            "0056 cashflow should be counted outside the invested account.",
        )

    def test_repair_split_like_price_jumps_adjusts_prior_segment(self) -> None:
        dates = pd.to_datetime(["2013-12-31", "2014-01-02"])
        prices = _price_frame(dates, 60.0)
        prices.loc[dates[1], ["open", "high", "low", "close", "adj_close"]] = [15.0, 15.5, 14.5, 15.0, 15.0]

        repaired, notes = repair_split_like_price_jumps(prices, ticker="0050.TW")

        self.assertEqual(notes[0]["applied_ratio"], 4)
        self.assertAlmostEqual(repaired.loc[dates[0], "close"], 15.0)
        self.assertAlmostEqual(repaired.loc[dates[0], "adj_close"], 15.0)


def _price_frame(dates: pd.DatetimeIndex, close: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": [close] * len(dates),
            "high": [close * 1.01] * len(dates),
            "low": [close * 0.99] * len(dates),
            "close": [close] * len(dates),
            "adj_close": [close] * len(dates),
            "volume": [100_000] * len(dates),
            "dividend": [0.0] * len(dates),
            "stock_split": [0.0] * len(dates),
        },
        index=dates,
    )


if __name__ == "__main__":
    unittest.main()
