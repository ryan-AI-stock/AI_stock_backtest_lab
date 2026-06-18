from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

import test_paths  # noqa: F401

from backtest_lab.decision_layers import DIAGNOSTIC
from backtest_lab.margin_short_ingestion_spec import (
    build_margin_short_readiness,
    default_margin_short_ingestion_spec,
    normalize_margin_short_frame,
    write_margin_short_spec_outputs,
)


class MarginShortIngestionSpecTest(unittest.TestCase):
    def test_spec_is_diagnostic_and_not_active_in_formal_trade(self) -> None:
        spec = default_margin_short_ingestion_spec()

        self.assertEqual(spec.decision_layer, DIAGNOSTIC)
        self.assertFalse(spec.active_in_trade_decision)
        self.assertIn("margin_balance", spec.required_columns)
        self.assertEqual(spec.canonical_output, "margin_short.latest.csv")

    def test_normalizes_twse_tpex_like_columns_and_flags_overheat(self) -> None:
        dates = pd.bdate_range("2026-06-01", periods=6)
        frame = pd.DataFrame(
            [
                {
                    "資料日期": date.strftime("%Y-%m-%d"),
                    "證券代號": "2454",
                    "證券名稱": "聯發科",
                    "融資餘額": 1000 + index * 50,
                    "融券餘額": 100 + index * 5,
                    "市場別": "TWSE",
                }
                for index, date in enumerate(dates)
            ]
        )
        frame.loc[len(frame) - 1, "融資餘額"] = 1300

        normalized = normalize_margin_short_frame(frame)

        self.assertEqual(normalized.loc[0, "ticker"], "2454.TW")
        self.assertIn("margin_balance_5d_change_pct", normalized.columns)
        self.assertGreaterEqual(normalized.loc[len(normalized) - 1, "margin_balance_5d_change_pct"], 12)
        self.assertTrue(bool(normalized.loc[len(normalized) - 1, "margin_overheat_flag"]))

        readiness = build_margin_short_readiness(frame, signal_date=dates[-1].strftime("%Y-%m-%d"))
        self.assertEqual(readiness["status"], "ready")
        self.assertEqual(readiness["decision_layer"], DIAGNOSTIC)
        self.assertFalse(readiness["active_in_trade_decision"])
        self.assertEqual(readiness["future_data_violation_count"], 0)

    def test_future_rows_block_readiness(self) -> None:
        frame = pd.DataFrame(
            [
                {"date": "2026-06-12", "ticker": "2454.TW", "margin_balance": 1000, "short_balance": 10},
                {"date": "2026-06-15", "ticker": "2454.TW", "margin_balance": 1300, "short_balance": 10},
            ]
        )

        readiness = build_margin_short_readiness(frame, signal_date="2026-06-12")

        self.assertEqual(readiness["status"], "blocked")
        self.assertEqual(readiness["future_data_violation_count"], 1)
        self.assertIn("future_data_violation", readiness["notes"])

    def test_writes_spec_only_without_raw_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            readiness = write_margin_short_spec_outputs(output_dir=output_dir, signal_date="2026-06-12")

            self.assertEqual(readiness["status"], "spec_only")
            self.assertTrue((output_dir / "margin_short_ingestion_spec.json").exists())
            self.assertTrue((output_dir / "margin_short_readiness.json").exists())


if __name__ == "__main__":
    unittest.main()
