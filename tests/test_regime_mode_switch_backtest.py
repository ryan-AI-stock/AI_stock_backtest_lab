from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from backtest_lab.regime_mode_switch_backtest import _load_sufficient_cache_prices


class RegimeModeSwitchBacktestTest(unittest.TestCase):
    def test_loads_local_cache_when_it_covers_warmup_and_period(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir)
            dates = pd.bdate_range("2019-01-02", "2020-12-31")
            self._write_prices(cache_dir / "TEST_TW.csv", dates)

            prices = _load_sufficient_cache_prices(
                ["TEST.TW"],
                str(cache_dir),
                required_start=pd.Timestamp("2020-01-01"),
                required_end=pd.Timestamp("2020-12-31"),
            )

            self.assertIn("TEST.TW", prices)

    def test_rejects_cache_without_warmup_history(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir)
            dates = pd.bdate_range("2020-01-02", "2020-12-31")
            self._write_prices(cache_dir / "TEST_TW.csv", dates)

            prices = _load_sufficient_cache_prices(
                ["TEST.TW"],
                str(cache_dir),
                required_start=pd.Timestamp("2020-01-01"),
                required_end=pd.Timestamp("2020-12-31"),
            )

            self.assertNotIn("TEST.TW", prices)

    @staticmethod
    def _write_prices(path: Path, dates: pd.DatetimeIndex) -> None:
        pd.DataFrame(
            {
                "date": dates,
                "open": 100.0,
                "close": 100.0,
                "adj_close": 100.0,
            }
        ).to_csv(path, index=False)


if __name__ == "__main__":
    unittest.main()
