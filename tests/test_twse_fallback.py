from __future__ import annotations

import unittest

import pandas as pd

import test_paths  # noqa: F401

from backtest_lab.frozen_strategy_monitor import (
    _roc_date_to_timestamp,
    _twse_float,
    fill_signal_date_from_twse,
)


class TwseFallbackTest(unittest.TestCase):
    def test_fill_signal_date_from_twse_appends_missing_latest_row(self) -> None:
        frame = pd.DataFrame(
            {
                "open": [100.0],
                "high": [101.0],
                "low": [99.0],
                "close": [100.5],
                "adj_close": [100.5],
                "volume": [1000.0],
                "dividend": [0.0],
                "stock_split": [0.0],
            },
            index=[pd.Timestamp("2026-06-03")],
        )

        filled = fill_signal_date_from_twse(
            {"0050.TW": frame},
            "2026-06-04",
            ["0050.TW"],
            fetcher=lambda ticker, signal_date: {
                "open": 101.0,
                "high": 102.0,
                "low": 100.0,
                "close": 101.5,
                "adj_close": 101.5,
                "volume": 2000.0,
                "dividend": 0.0,
                "stock_split": 0.0,
            },
        )

        self.assertIn(pd.Timestamp("2026-06-04"), filled["0050.TW"].index)
        self.assertEqual(filled["0050.TW"].loc[pd.Timestamp("2026-06-04"), "close"], 101.5)

    def test_twse_helpers_parse_roc_date_and_comma_number(self) -> None:
        self.assertEqual(_roc_date_to_timestamp("115/06/04"), pd.Timestamp("2026-06-04"))
        self.assertEqual(_twse_float("1,234.5"), 1234.5)


if __name__ == "__main__":
    unittest.main()
