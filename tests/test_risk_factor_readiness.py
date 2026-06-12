from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

import test_paths  # noqa: F401

from backtest_lab.risk_factor_readiness import build_risk_factor_readiness, write_readiness_outputs


class RiskFactorReadinessTest(unittest.TestCase):
    def test_reports_partial_when_required_sources_are_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pd.DataFrame(
                [
                    {
                        "date": "2026-06-12",
                        "symbol": "2454",
                        "foreign_consecutive_sell_days": 3,
                        "foreign_net_buy_shares": -1000,
                    }
                ]
            ).to_csv(root / "institutional_flows.latest.csv", index=False, encoding="utf-8-sig")

            readiness = build_risk_factor_readiness(signal_date="2026-06-12", radar_data_dir=root)

        self.assertEqual(readiness["status"], "partial")
        self.assertIn("institutional", readiness["available_risk_kinds"])
        self.assertIn("margin_short", readiness["missing_risk_kinds"])
        self.assertIn("market_cap_missing", readiness["notes"])
        self.assertGreater(readiness["risk_factor_nonzero_count"], 0)

    def test_writes_json_and_csv_outputs(self) -> None:
        readiness = {
            "status": "partial",
            "signals": [{"ticker": "2454.TW", "total_risk_score": 0.3}],
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            json_path = root / "readiness.json"
            csv_path = root / "signals.csv"
            write_readiness_outputs(readiness, output_json=json_path, output_csv=csv_path)

            self.assertTrue(json_path.exists())
            self.assertTrue(csv_path.exists())
            self.assertIn("2454.TW", csv_path.read_text(encoding="utf-8-sig"))


if __name__ == "__main__":
    unittest.main()
