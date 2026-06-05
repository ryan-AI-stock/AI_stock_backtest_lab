import unittest

import pandas as pd

import test_paths  # noqa: F401

from backtest_lab.market_regime import classify_market_regime, has_data_for_date


def frame_from_closes(closes: list[float], start: str = "2021-01-01") -> pd.DataFrame:
    dates = pd.bdate_range(start, periods=len(closes))
    return pd.DataFrame(
        {
            "open": closes,
            "close": closes,
            "adj_close": closes,
        },
        index=dates,
    )


class MarketRegimeTest(unittest.TestCase):
    def test_classifies_strong_bull_when_trend_and_momentum_are_positive(self) -> None:
        closes = [100 + index * 0.5 for index in range(320)]
        prices = frame_from_closes(closes)

        snapshot = classify_market_regime(prices, prices.index[-1])

        self.assertEqual(snapshot.regime, "strong_bull")
        self.assertEqual(snapshot.regime_label, "強多頭")
        self.assertGreater(snapshot.return_60d, 0)

    def test_classifies_systemic_bear_when_below_falling_long_trend(self) -> None:
        closes = [300 - index * 0.6 for index in range(320)]
        prices = frame_from_closes(closes)

        snapshot = classify_market_regime(prices, prices.index[-1])

        self.assertEqual(snapshot.regime, "systemic_bear")
        self.assertLess(snapshot.drawdown_from_252d_high, -0.2)

    def test_has_data_for_date_requires_exact_latest_date(self) -> None:
        prices = frame_from_closes([100, 101, 102])

        self.assertTrue(has_data_for_date(prices, prices.index[-1]))
        self.assertFalse(has_data_for_date(prices, prices.index[-1] + pd.Timedelta(days=1)))


if __name__ == "__main__":
    unittest.main()
