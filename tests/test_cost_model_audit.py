import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

import test_paths  # noqa: F401

from backtest_lab.cost_model_audit import run_cost_model_audit


class CostModelAuditTest(unittest.TestCase):
    def test_audit_outputs_cost_coverage_package(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = run_cost_model_audit(output_dir=Path(tmp) / "cost_audit")

            self.assertTrue((output / "manifest.json").exists())
            self.assertTrue((output / "current_cost_coverage.csv").exists())
            self.assertTrue((output / "missing_cost_fields.csv").exists())
            self.assertTrue((output / "final_summary_zh.md").exists())

            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["cost_model_version"], "taiwan_standard_fee_tax_v1")
            self.assertFalse(manifest["formal_model_changed"])
            self.assertFalse(manifest["trade_decision_changed"])
            self.assertTrue(manifest["current_formal_next_day_cost_complete"])
            self.assertFalse(manifest["daily_report_has_profit_or_pnl_fields"])

            coverage = pd.read_csv(output / "current_cost_coverage.csv")
            scopes = set(coverage["scope"])
            self.assertIn("current_formal_next_day_replay", scopes)
            self.assertIn("stock_pool_observation_daily_report", scopes)

            daily = coverage[coverage["scope"] == "stock_pool_observation_daily_report"].iloc[0]
            self.assertEqual(daily["coverage_status"], "not_applicable_no_profit_or_pnl_fields")

            missing = pd.read_csv(output / "missing_cost_fields.csv")
            self.assertIn("historical_outputs_before_current_fix", set(missing["scope"]))


if __name__ == "__main__":
    unittest.main()
