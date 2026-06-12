from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from backtest_lab.radar_snapshot_v2_policy_sweep import resolve_cached_prices


class RadarSnapshotV2PolicySweepTest(unittest.TestCase):
    def test_resolve_cached_prices_handles_tw_and_two_suffixes(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write_price_cache(root / "2408_TW.csv")
            _write_price_cache(root / "8299_TWO.csv")

            resolved = resolve_cached_prices(["2408", "8299", "9999"], [root])

            self.assertEqual(resolved.symbol_to_ticker["2408"], "2408.TW")
            self.assertEqual(resolved.symbol_to_ticker["8299"], "8299.TWO")
            self.assertIn("9999", resolved.skipped_symbols)


def _write_price_cache(path: Path) -> None:
    dates = pd.bdate_range("2026-01-01", periods=3)
    pd.DataFrame(
        {
            "date": dates,
            "open": [1.0, 1.0, 1.0],
            "high": [1.0, 1.0, 1.0],
            "low": [1.0, 1.0, 1.0],
            "close": [1.0, 1.0, 1.0],
            "adj_close": [1.0, 1.0, 1.0],
            "volume": [1000, 1000, 1000],
            "dividend": [0.0, 0.0, 0.0],
        }
    ).to_csv(path, index=False)


if __name__ == "__main__":
    unittest.main()
