import unittest

import pandas as pd

import test_paths  # noqa: F401

from backtest_lab.data import split_adjusted_dividends


class DividendAdjustmentTest(unittest.TestCase):
    def test_pre_split_dividends_are_divided_for_split_adjusted_prices(self) -> None:
        prices = pd.DataFrame(
            {"dividend": [0.75, 0.36]},
            index=pd.to_datetime(["2024-01-17", "2025-07-21"]),
        )

        adjusted = split_adjusted_dividends(
            prices,
            ({"date": "2025-06-18", "ratio": 4.0},),
        )

        self.assertAlmostEqual(adjusted.loc[pd.Timestamp("2024-01-17")], 0.1875)
        self.assertAlmostEqual(adjusted.loc[pd.Timestamp("2025-07-21")], 0.36)


if __name__ == "__main__":
    unittest.main()
