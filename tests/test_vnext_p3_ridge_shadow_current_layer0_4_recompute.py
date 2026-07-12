import unittest

import pandas as pd

from backtest_lab.vnext_p3_ridge_shadow_current_layer0_4_recompute import layer0_snapshot


class CurrentLayer0RecomputeTest(unittest.TestCase):
    def test_snapshot_never_uses_etf_ticker(self):
        dates = pd.to_datetime(["2026-06-12", "2026-06-19", "2026-06-26", "2026-07-03", "2026-07-09"])
        rows = []
        for date in dates:
            for ticker in [f"{value:04d}" for value in range(1000, 1301)] + ["0050"]:
                rows.append({"trade_date": date, "ticker": ticker, "name": ticker, "market": "TWSE", "traded_value_5d": 100000-int(ticker), "traded_value_20d": 200000-int(ticker), "traded_value_60d": 300000-int(ticker), "eligible": not ticker.startswith("00"), "listing_status": "listed", "liquidity_flag": True})
        snapshot = layer0_snapshot(pd.DataFrame(rows))
        self.assertNotIn("0050", set(snapshot.ticker))
        self.assertEqual(snapshot.snapshot_date.nunique(), 1)


if __name__ == "__main__":
    unittest.main()
