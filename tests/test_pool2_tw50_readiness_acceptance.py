from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import test_paths  # noqa: F401

from backtest_lab.pool2_tw50_readiness_acceptance import (
    evaluate_pool2_tw50_readiness,
    run_pool2_tw50_readiness_acceptance,
)


class Pool2Tw50ReadinessAcceptanceTest(unittest.TestCase):
    def test_blocks_waiting_user_files_package(self) -> None:
        result = evaluate_pool2_tw50_readiness(
            {
                "status": "blocked_waiting_user_files",
                "exact_tw50_official_constituents_ready": False,
                "yuanta_0050_holdings_proxy_ready": False,
                "future_data_violation_count": 0,
                "accepted_proxy_rows": 0,
                "missing_priority_one_files": ["yuanta_0050_monthly_202201.pdf"],
                "core_coverage_summary": [
                    {"period": "2022", "checked_dates": "260", "ready_dates": "0", "gap_dates": "260", "coverage_ratio": "0.0"}
                ],
            }
        )

        self.assertEqual(result["acceptance_status"], "blocked_waiting_user_files")
        self.assertFalse(result["can_use_as_exact_tw50_constituents"])
        self.assertFalse(result["can_use_as_0050_holdings_proxy"])
        self.assertIn("no exact TW50 or proxy-specific accepted rows are ready", result["blockers"])

    def test_proxy_specific_ready_is_not_exact_tw50(self) -> None:
        result = evaluate_pool2_tw50_readiness(
            {
                "status": "proxy_ready",
                "formal_ready": False,
                "exact_tw50_official_constituents_ready": False,
                "yuanta_0050_holdings_proxy_ready": True,
                "is_proxy": True,
                "accepted_proxy_rows": 50,
                "future_data_violation_count": 0,
                "core_coverage_summary": [],
            }
        )

        self.assertEqual(result["acceptance_status"], "accepted_proxy_specific")
        self.assertFalse(result["can_use_as_exact_tw50_constituents"])
        self.assertTrue(result["can_use_as_0050_holdings_proxy"])
        self.assertTrue(any("proxy-specific" in warning for warning in result["warnings"]))

    def test_exact_tw50_requires_coverage_threshold(self) -> None:
        result = evaluate_pool2_tw50_readiness(
            {
                "status": "ready",
                "formal_ready": True,
                "exact_tw50_official_constituents_ready": True,
                "yuanta_0050_holdings_proxy_ready": False,
                "is_proxy": False,
                "future_data_violation_count": 0,
                "core_coverage_summary": [
                    {"period": "2022", "checked_dates": "260", "ready_dates": "260", "gap_dates": "0", "coverage_ratio": "1.0"},
                    {"period": "2023", "checked_dates": "259", "ready_dates": "200", "gap_dates": "59", "coverage_ratio": "0.7722"},
                ],
            }
        )

        self.assertEqual(result["acceptance_status"], "ready")
        self.assertFalse(result["can_use_as_exact_tw50_constituents"])
        self.assertTrue(any("coverage below threshold" in blocker for blocker in result["blockers"]))

    def test_runner_writes_acceptance_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            readiness = root / "readiness.json"
            readiness.write_text(
                json.dumps(
                    {
                        "status": "blocked_waiting_user_files",
                        "exact_tw50_official_constituents_ready": False,
                        "yuanta_0050_holdings_proxy_ready": False,
                        "future_data_violation_count": 0,
                        "accepted_proxy_rows": 0,
                        "core_coverage_summary": [],
                    }
                ),
                encoding="utf-8",
            )

            result = run_pool2_tw50_readiness_acceptance(readiness_path=readiness, output_dir=root / "out")

            self.assertEqual(result["acceptance_status"], "blocked_waiting_user_files")
            self.assertTrue((root / "out" / "pool2_tw50_readiness_acceptance.json").exists())
            self.assertTrue((root / "out" / "pool2_tw50_readiness_acceptance.md").exists())


if __name__ == "__main__":
    unittest.main()
