from __future__ import annotations

import unittest
from pathlib import Path


class StockPoolObservationWorkflowTest(unittest.TestCase):
    def test_workflow_checks_out_radar_repo_and_defaults_snapshot_history(self) -> None:
        workflow = Path(".github/workflows/stock_pool_observation.yml").read_text(encoding="utf-8")

        self.assertIn("repository: ryan-AI-stock/AI_stock_rotation_radar", workflow)
        self.assertIn("path: AI_stock_rotation_radar", workflow)
        self.assertIn("AI_stock_rotation_radar/data/history", workflow)
        self.assertIn("RADAR_SNAPSHOT_DIR", workflow)
        self.assertIn("python -m backtest_lab.drive_publish", workflow)
        self.assertIn("AI股票池觀察總覽_最新版_v20260612.pdf", workflow)
        self.assertIn("STOCK_POOL_OBSERVATION_DRIVE_FOLDER_ID", workflow)
        self.assertIn("--require-exact-signal-date", workflow)
        self.assertIn("steps.observation-pdf.outputs.exists == 'true'", workflow)


if __name__ == "__main__":
    unittest.main()
