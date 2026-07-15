from __future__ import annotations

import unittest
from pathlib import Path


class GithubActionsWorkflowPolicyTest(unittest.TestCase):
    def test_only_authorized_workflows_remain(self) -> None:
        workflow_paths = sorted(path.name for path in Path(".github/workflows").glob("*.yml"))

        self.assertEqual(
            workflow_paths,
            ["stock_pool_observation.yml", "vnext-ridge-prospective-shadow.yml"],
        )

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

    def test_ridge_shadow_is_gated_blocking_and_append_only(self) -> None:
        workflow = Path(".github/workflows/vnext-ridge-prospective-shadow.yml").read_text(encoding="utf-8")

        self.assertIn("repository: ryan-AI-stock/AI_stock_schedule_rules", workflow)
        self.assertIn("python -m stock_schedule_rules.gate", workflow)
        self.assertIn("steps.schedule-gate.outputs.should_run", workflow)
        self.assertIn("steps.schedule-gate.outputs.target_date", workflow)
        self.assertIn("blocked_exact_current_layer0_4_input", workflow)
        self.assertIn("steps.readiness.outputs.ready == 'true'", workflow)
        self.assertIn("--require-hashes -r requirements-ml.lock", workflow)
        self.assertIn("git add data/vnext_shadow/predictions", workflow)
        self.assertIn("append-only", workflow)

    def test_ridge_lock_contains_windows_cp313_hashes(self) -> None:
        lock = Path("requirements-ml.lock").read_text(encoding="utf-8")

        for digest in (
            "c4fc99836233ea196540b17ab0983aff60ed07941751930f5f4d05bc3b3b7359",
            "f8bfc0e12dc78f777f323f55c58649591b2cd0c43534e8355c51d3fede5f4dee",
            "63a9afd6f7b229aad94618c01c252ce9e6fa97918c5ca19c9a17a087d819440c",
            "37425bc9175607b0268f493d79a292c39f9d001a357bebb6b88fdfaff13f6448",
        ):
            with self.subTest(digest=digest):
                self.assertIn(digest, lock)


if __name__ == "__main__":
    unittest.main()
