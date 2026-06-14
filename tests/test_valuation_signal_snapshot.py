from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import test_paths  # noqa: F401

from backtest_lab.valuation_signal_snapshot import (
    build_valuation_signal_snapshot,
    _load_price_frames_cache_first,
    write_valuation_signal_snapshot_outputs,
)


class ValuationSignalSnapshotTest(unittest.TestCase):
    def test_snapshot_marks_blocked_and_missing_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "valuation.csv"
            path.write_text(
                "source_date,symbol,eps_estimate_low,eps_estimate_high,fair_pe,buy_price,valuation_action\n"
                "2026-06-14,2454,75,80,,,cannot_buy\n"
                "2026-06-14,2317,18,20,14,252,buy_zone\n",
                encoding="utf-8",
            )

            snapshot = build_valuation_signal_snapshot(
                valuation_data=path,
                signal_date="2026-06-15",
                tickers=["2454.TW", "2317.TW", "2330.TW"],
                current_price_by_ticker={"2454.TW": 4200, "2317.TW": 250, "2330.TW": 1200},
            )

        rows = {row["ticker"]: row for row in snapshot["rows"]}
        self.assertEqual(snapshot["covered_ticker_count"], 2)
        self.assertEqual(snapshot["blocked_count"], 1)
        self.assertFalse(rows["2454.TW"]["gate_passed"])
        self.assertEqual(rows["2454.TW"]["reason"], "估值來源標示不能買")
        self.assertTrue(rows["2317.TW"]["gate_passed"])
        self.assertEqual(rows["2330.TW"]["valuation_status"], "missing")

    def test_writes_csv_json_and_markdown(self) -> None:
        snapshot = {
            "valuation_data": "valuation.csv",
            "signal_date": "2026-06-15",
            "expected_ticker_count": 1,
            "covered_ticker_count": 1,
            "coverage_ratio": 1.0,
            "blocked_count": 0,
            "passable_count": 1,
            "rows": [
                {
                    "ticker": "2317.TW",
                    "signal_date": "2026-06-14",
                    "current_price": 250.0,
                    "valuation_action": "buy_zone",
                    "valuation_status": "covered",
                    "gate_passed": True,
                    "score_adjustment": 0.01,
                    "eps_estimate_low": 18.0,
                    "eps_estimate_high": 20.0,
                    "fair_pe": 14.0,
                    "fair_price": 266.0,
                    "buy_price": 252.0,
                    "safety_margin_pct": 6.4,
                    "reason": "估值仍有安全邊際",
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            write_valuation_signal_snapshot_outputs(snapshot, output_dir=output)

            self.assertTrue((output / "valuation_signal_snapshot.csv").exists())
            self.assertTrue((output / "valuation_signal_snapshot.json").exists())
            self.assertIn("估值訊號快照", (output / "valuation_signal_snapshot.md").read_text(encoding="utf-8"))

    def test_cache_first_loader_reads_nested_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            nested = root / "nested"
            nested.mkdir()
            (nested / "2454_TW.csv").write_text(
                "date,open,high,low,close,adj_close,volume,dividend,stock_split\n"
                "2026-06-12,4410,4415,4165,4180,4180,7806177,0,0\n",
                encoding="utf-8",
            )

            prices, missing = _load_price_frames_cache_first(
                tickers=["2454.TW"],
                start_date="2026-06-01",
                end_date="2026-06-15",
                cache_dir=root,
            )

        self.assertEqual(missing, [])
        self.assertEqual(float(prices["2454.TW"].iloc[-1]["close"]), 4180.0)


if __name__ == "__main__":
    unittest.main()
