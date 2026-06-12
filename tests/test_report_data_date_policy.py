from __future__ import annotations

import unittest
from pathlib import Path


class ReportDataDatePolicyTest(unittest.TestCase):
    def test_only_public_scorecard_uses_delayed_public_cutoff(self) -> None:
        scorecard_workflow = Path(".github/workflows/model_scorecard_report.yml").read_text(encoding="utf-8")
        best_workflow = Path(".github/workflows/frozen_strategy_daily_report.yml").read_text(encoding="utf-8")
        observation_workflow = Path(".github/workflows/stock_pool_observation.yml").read_text(encoding="utf-8")

        self.assertIn("backtest_lab.model_scorecard_report", scorecard_workflow)
        self.assertIn("public scorecard is delayed by 7 calendar days", scorecard_workflow)
        self.assertIn("--report-date", scorecard_workflow)
        self.assertIn("AI模型延遲公開成績單_最新版_v20260612.pdf", scorecard_workflow)

        self.assertIn("--signal-date", best_workflow)
        self.assertNotIn("public scorecard is delayed", best_workflow)
        self.assertNotIn("--report-date", best_workflow)
        self.assertNotIn("schedule:", best_workflow)
        self.assertNotIn("drive_publish", best_workflow)
        self.assertNotIn("AI股票最佳策略每日觀察報告_最新版_v20260605.pdf", best_workflow)

        self.assertIn("--signal-date", observation_workflow)
        self.assertNotIn("public scorecard is delayed", observation_workflow)
        self.assertNotIn("--report-date", observation_workflow)
        self.assertIn("AI股票池觀察總覽_最新版_v20260612.pdf", observation_workflow)


if __name__ == "__main__":
    unittest.main()
