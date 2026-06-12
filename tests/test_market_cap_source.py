from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

import test_paths  # noqa: F401

from backtest_lab.market_cap_source import load_first_available_market_caps, load_market_cap_by_ticker


class MarketCapSourceTest(unittest.TestCase):
    def test_loads_latest_market_cap_not_after_signal_date(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "market_cap.latest.csv"
            pd.DataFrame(
                [
                    {"date": "2026-06-05", "symbol": "2330", "market_cap_twd": "1,000,000,000,000"},
                    {"date": "2026-06-12", "symbol": "2330", "market_cap_twd": "2,000,000,000,000"},
                    {"date": "2026-06-13", "symbol": "2330", "market_cap_twd": "9,000,000,000,000"},
                    {"date": "2026-06-12", "ticker": "2454.TW", "free_float_market_cap_twd": "800,000,000,000"},
                ]
            ).to_csv(path, index=False, encoding="utf-8-sig")

            caps = load_market_cap_by_ticker(path, signal_date="2026-06-12")

        self.assertEqual(caps["2330.TW"], 2_000_000_000_000)
        self.assertEqual(caps["2454.TW"], 800_000_000_000)

    def test_discovers_radar_stock_metrics_as_fallback_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            radar_data_dir = Path(tmp) / "radar_data"
            radar_data_dir.mkdir()
            pd.DataFrame(
                [
                    {
                        "report_date": "2026-06-12",
                        "symbol": "2408",
                        "name": "南亞科",
                        "market_cap_twd": 80_000_000_000,
                    }
                ]
            ).to_csv(radar_data_dir / "stock_metrics.refreshed.csv", index=False, encoding="utf-8-sig")

            caps, source = load_first_available_market_caps(
                signal_date="2026-06-12",
                radar_data_dir=radar_data_dir,
            )

        self.assertEqual(caps["2408.TW"], 80_000_000_000)
        self.assertTrue(source.endswith("stock_metrics.refreshed.csv"))


if __name__ == "__main__":
    unittest.main()
