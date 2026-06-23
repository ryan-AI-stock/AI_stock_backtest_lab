from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

import test_paths  # noqa: F401

from backtest_lab.pool3_radar_weighted_overlay_replay import run_pool3_radar_weighted_overlay_replay


class Pool3RadarWeightedOverlayReplayTest(unittest.TestCase):
    def test_replays_weighted_basket_with_costs_and_holdings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "basket.csv"
            output = root / "out"
            pd.DataFrame(
                [
                    _row("2024-01-02", "variant_a", "00631L.TW", "market", 0.8, 10.0),
                    _row("2024-01-02", "variant_a", "3260.TW", "memory", 0.2, 100.0),
                    _row("2024-01-03", "variant_a", "00631L.TW", "market", 0.8, 11.0),
                    _row("2024-01-03", "variant_a", "3260.TW", "memory", 0.2, 110.0),
                    _row("2024-01-04", "variant_a", "00631L.TW", "market", 0.5, 12.0),
                    _row("2024-01-04", "variant_a", "3260.TW", "memory", 0.5, 120.0),
                ]
            ).to_csv(source, index=False)

            result = run_pool3_radar_weighted_overlay_replay(
                weighted_basket_daily=source,
                output_dir=output,
                initial_cash=100_000,
            )

            daily = pd.read_csv(result / "pool3_radar_weighted_overlay_formal_daily.csv")
            self.assertIn("holding_ticker", daily.columns)
            self.assertIn("transaction_cost", daily.columns)
            self.assertIn("equity", daily.columns)
            self.assertEqual(set(daily["pool3_formal_vote"]), {"weighted_basket"})
            first_day = daily[daily["date"] == "2024-01-02"]
            self.assertEqual(set(first_day["holding_ticker"]), {"00631L.TW", "3260.TW"})
            self.assertGreater(first_day["transaction_cost"].sum(), 0)
            third_day = daily[daily["date"] == "2024-01-04"]
            self.assertTrue((third_day["fill_action"] == "rebalance").any())
            summary = pd.read_csv(result / "pool3_radar_weighted_overlay_summary.csv")
            self.assertEqual(summary.iloc[0]["variant"], "variant_a")
            self.assertEqual(int(summary.iloc[0]["rebalance_days"]), 2)
            self.assertEqual(
                int(summary.iloc[0]["total_transaction_cost"]),
                int(pd.to_numeric(daily["transaction_cost"], errors="coerce").sum()),
            )
            self.assertTrue((result / "metadata.json").exists())


def _row(date: str, variant: str, ticker: str, theme: str, weight: float, close: float) -> dict:
    return {
        "date": date,
        "period": "2024",
        "variant": variant,
        "ticker": ticker,
        "theme": theme,
        "weight": weight,
        "close": close,
    }


if __name__ == "__main__":
    unittest.main()
