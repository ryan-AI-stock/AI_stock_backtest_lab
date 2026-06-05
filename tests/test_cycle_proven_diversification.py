from __future__ import annotations

import unittest

import pandas as pd

from backtest_lab.cycle_proven_diversification import DiversificationVariant, _attack_weights, diversification_variants


class CycleProvenDiversificationTest(unittest.TestCase):
    def test_variants_include_top1_and_diversified_portfolios(self) -> None:
        variants = diversification_variants()

        self.assertEqual(len(variants), 6)
        self.assertEqual({variant.top_n for variant in variants}, {1, 2, 3})

    def test_attack_weights_exclude_broad_market_etfs(self) -> None:
        dates = pd.bdate_range("2025-01-01", periods=80)
        prices = {
            "0050.TW": self._frame(dates, 1.0),
            "00631L.TW": self._frame(dates, 2.0),
            "A.TW": self._frame(dates, 1.5),
            "B.TW": self._frame(dates, 1.2),
        }

        weights = _attack_weights(prices, dates[-1], DiversificationVariant("top2", 2, "equal"))

        self.assertEqual(set(weights), {"A.TW", "B.TW"})
        self.assertAlmostEqual(sum(weights.values()), 1.0)

    @staticmethod
    def _frame(dates: pd.DatetimeIndex, step: float) -> pd.DataFrame:
        values = [100 + step * index for index in range(len(dates))]
        return pd.DataFrame({"adj_close": values}, index=dates)


if __name__ == "__main__":
    unittest.main()
