from __future__ import annotations

import unittest
from pathlib import Path


class GithubActionsWorkflowPolicyTest(unittest.TestCase):
    def test_only_stock_pool_observation_workflow_remains(self) -> None:
        workflow_paths = sorted(path.name for path in Path(".github/workflows").glob("*.yml"))

        self.assertEqual(workflow_paths, ["stock_pool_observation.yml"])

    def test_stock_pool_observation_is_formal_daily_report_workflow(self) -> None:
        workflow = Path(".github/workflows/stock_pool_observation.yml").read_text(encoding="utf-8")

        self.assertIn("name: Stock Pool Observation", workflow)
        self.assertIn("workflow_dispatch:", workflow)
        self.assertIn("schedule:", workflow)
        self.assertIn("AI股票池觀察總覽_最新版_v20260612.pdf", workflow)
        self.assertIn("python -m backtest_lab.drive_publish", workflow)
        self.assertIn("STOCK_POOL_OBSERVATION_DRIVE_FOLDER_ID", workflow)
        self.assertIn("FORCE_JAVASCRIPT_ACTIONS_TO_NODE24: true", workflow)

    def test_removed_workflows_do_not_return(self) -> None:
        workflows = "\n".join(
            path.read_text(encoding="utf-8")
            for path in Path(".github/workflows").glob("*.yml")
        )

        forbidden = (
            "Best Strategy Daily Report",
            "Model Scorecard Delayed Report",
            "Live Path Tracker Disabled",
            "frozen_strategy_daily_report.yml",
            "model_scorecard_report.yml",
            "live_path_tracker.yml",
            "AI模型延遲公開成績單_最新版",
            "AI模型實戰路徑追蹤報告_最新版",
        )
        for text in forbidden:
            with self.subTest(text=text):
                self.assertNotIn(text, workflows)


if __name__ == "__main__":
    unittest.main()
