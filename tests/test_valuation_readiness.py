from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

import test_paths  # noqa: F401

from backtest_lab.valuation_readiness import build_valuation_readiness, write_valuation_readiness_outputs


class ValuationReadinessTest(unittest.TestCase):
    def test_reports_ready_when_point_in_time_coverage_is_complete(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "valuation.csv"
            _write_valuation(
                path,
                [
                    {"source_date": "2024-07-01", "symbol": "2317", "eps_estimate_low": 13, "eps_estimate_high": 15, "fair_pe": 14, "buy_price": 130},
                    {"source_date": "2024-07-01", "symbol": "2382", "eps_estimate_low": 20, "eps_estimate_high": 22, "fair_pe": 14, "buy_price": 500},
                ],
            )

            readiness = build_valuation_readiness(
                valuation_data=path,
                start_date="2024-07-02",
                end_date="2024-07-05",
                tickers=["2317.TW", "2382.TW"],
            )

        self.assertEqual(readiness["status"], "ready")
        self.assertEqual(readiness["expected_ticker_count"], 2)
        self.assertEqual(readiness["average_coverage_ratio"], 1.0)
        self.assertEqual(readiness["warnings"], [])

    def test_reports_partial_when_buy_price_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "valuation.csv"
            _write_valuation(
                path,
                [
                    {"source_date": "2024-07-01", "symbol": "2317", "fair_price": 196},
                ],
            )

            readiness = build_valuation_readiness(
                valuation_data=path,
                start_date="2024-07-02",
                end_date="2024-07-05",
                tickers=["2317.TW"],
            )

        self.assertEqual(readiness["status"], "partial")
        self.assertTrue(any("buy_price_missing" in warning for warning in readiness["warnings"]))

    def test_reports_not_ready_when_file_is_missing(self) -> None:
        readiness = build_valuation_readiness(
            valuation_data="missing.csv",
            start_date="2024-07-02",
            end_date="2024-07-05",
            tickers=["2317.TW"],
        )

        self.assertEqual(readiness["status"], "not_ready")
        self.assertIn("valuation_data_missing", readiness["warnings"])

    def test_writes_json_and_csv_outputs(self) -> None:
        readiness = {
            "status": "ready",
            "coverage": [{"signal_date": "2024-07-02", "coverage_ratio": 1.0}],
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            json_path = root / "valuation_readiness.json"
            csv_path = root / "valuation_coverage.csv"
            write_valuation_readiness_outputs(readiness, output_json=json_path, output_csv=csv_path)

            self.assertTrue(json_path.exists())
            self.assertTrue(csv_path.exists())
            self.assertIn("2024-07-02", csv_path.read_text(encoding="utf-8-sig"))


def _write_valuation(path: Path, rows: list[dict[str, object]]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    unittest.main()
