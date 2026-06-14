from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import test_paths  # noqa: F401

from backtest_lab.valuation_source import load_valuation_signals


class ValuationSourceTest(unittest.TestCase):
    def test_loads_latest_point_in_time_signal_without_future_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "valuation.csv"
            path.write_text(
                "source_date,symbol,eps_estimate_low,eps_estimate_high,fair_pe,buy_price\n"
                "2026-06-01,2317,18,20,14,252\n"
                "2026-06-20,2317,30,32,16,496\n",
                encoding="utf-8",
            )

            signals = load_valuation_signals(
                path,
                signal_date="2026-06-12",
                current_price_by_ticker={"2317.TW": 260},
            )

            signal = signals["2317.TW"]
            self.assertEqual(signal.signal_date, "2026-06-01")
            self.assertEqual(signal.buy_price, 252)
            self.assertFalse(signal.gate_passed)
            self.assertEqual(signal.reason, "現價高於合理買點")
            self.assertLess(signal.score_adjustment, 0)


if __name__ == "__main__":
    unittest.main()
