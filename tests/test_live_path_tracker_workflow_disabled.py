from __future__ import annotations

import unittest
from pathlib import Path

import test_paths  # noqa: F401


class LivePathTrackerWorkflowDisabledTest(unittest.TestCase):
    def test_workflow_no_longer_generates_or_publishes_latest_report(self) -> None:
        workflow = Path(".github/workflows/live_path_tracker.yml").read_text(encoding="utf-8")

        self.assertIn("Live Path Tracker Disabled", workflow)
        self.assertNotIn("schedule:", workflow)
        self.assertNotIn("python -m backtest_lab.live_path_tracker", workflow)
        self.assertNotIn("backtest_lab.drive_publish", workflow)
        self.assertNotIn("AI模型實戰路徑追蹤報告_最新版", workflow)
        self.assertNotIn("outputs/live_path_tracker/", workflow)


if __name__ == "__main__":
    unittest.main()
