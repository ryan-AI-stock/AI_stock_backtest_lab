from __future__ import annotations

import unittest
from pathlib import Path

import test_paths  # noqa: F401


class LivePathTrackerWorkflowDisabledTest(unittest.TestCase):
    def test_workflow_removed_from_github_actions(self) -> None:
        workflow_path = Path(".github/workflows/live_path_tracker.yml")

        self.assertFalse(workflow_path.exists())

    def test_remaining_workflows_do_not_publish_live_path_report(self) -> None:
        workflows = "\n".join(
            path.read_text(encoding="utf-8")
            for path in Path(".github/workflows").glob("*.yml")
        )

        self.assertNotIn("Live Path Tracker Disabled", workflows)
        self.assertNotIn("python -m backtest_lab.live_path_tracker", workflows)
        self.assertNotIn("backtest_lab.live_path_tracker", workflows)
        self.assertNotIn("AI模型實戰路徑追蹤報告_最新版", workflows)
        self.assertNotIn("outputs/live_path_tracker/", workflows)


if __name__ == "__main__":
    unittest.main()
