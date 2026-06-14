from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

import test_paths  # noqa: F401

from backtest_lab.valuation_signal_importer import import_valuation_signals, normalize_valuation_input_frame


class ValuationSignalImporterTest(unittest.TestCase):
    def test_normalizes_manual_analyst_style_rows(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "source_date": "2026/06/14",
                    "symbol": "2317",
                    "name": "鴻海",
                    "eps_estimate_range": "18~20",
                    "fair_pe": "14倍",
                    "buy_price": "252元",
                    "source_name": "manual",
                }
            ]
        )

        normalized, manifest = normalize_valuation_input_frame(frame)

        row = normalized.iloc[0].to_dict()
        self.assertEqual(manifest["output_rows"], 1)
        self.assertEqual(row["source_date"], "2026-06-14")
        self.assertEqual(row["ticker"], "2317.TW")
        self.assertEqual(row["eps_estimate_low"], "18")
        self.assertEqual(row["eps_estimate_high"], "20")
        self.assertEqual(row["fair_pe"], "14")
        self.assertEqual(row["fair_price"], "266")
        self.assertEqual(row["buy_price"], "252")

    def test_import_appends_and_deduplicates_by_source_date_and_ticker(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "manual.csv"
            output_path = root / "valuation_signals.csv"
            pd.DataFrame(
                [
                    {"source_date": "2026-06-14", "symbol": "2317", "eps_estimate_range": "18~20", "fair_pe": "14", "buy_price": "252"},
                    {"source_date": "2026-06-14", "symbol": "2317", "eps_estimate_range": "19~21", "fair_pe": "14", "buy_price": "260"},
                ]
            ).to_csv(input_path, index=False, encoding="utf-8-sig")

            manifest = import_valuation_signals(input_path=input_path, output_path=output_path)

            output = pd.read_csv(output_path, dtype=str).fillna("")
            self.assertEqual(manifest["output_rows"], 2)
            self.assertEqual(manifest["final_rows"], 1)
            self.assertEqual(output.iloc[0]["buy_price"], "260")

    def test_default_source_date_can_fill_missing_date(self) -> None:
        frame = pd.DataFrame([{"symbol": "2382", "eps_estimate_range": "85-90", "fair_pe": "14"}])

        normalized, manifest = normalize_valuation_input_frame(frame, default_source_date="2026-06-14")

        self.assertEqual(manifest["skipped_rows"], [])
        self.assertEqual(normalized.iloc[0]["source_date"], "2026-06-14")
        self.assertEqual(normalized.iloc[0]["ticker"], "2382.TW")


if __name__ == "__main__":
    unittest.main()
