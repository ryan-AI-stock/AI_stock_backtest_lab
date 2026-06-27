from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path

from backtest_lab.core_self_check import (
    build_findings,
    check_required_files,
    collect_git_head,
    collect_git_status,
    run_core_self_check,
)


def fake_runner(stdout_by_command: dict[tuple[str, ...], str]):
    def _run(command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 0, stdout_by_command.get(tuple(command), ""), stderr="")

    return _run


class CoreSelfCheckTest(unittest.TestCase):
    def test_collect_git_status_uses_runner(self) -> None:
        runner = fake_runner({("git", "status", "--short"): " M src/example.py\n"})

        self.assertEqual(collect_git_status(Path("."), runner), "M src/example.py")

    def test_collect_git_head_uses_runner(self) -> None:
        runner = fake_runner({("git", "rev-parse", "--short", "HEAD"): "abc1234\n"})

        self.assertEqual(collect_git_head(Path("."), runner), "abc1234")

    def test_check_required_files_reports_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "exists.txt").write_text("ok", encoding="utf-8")

            rows = check_required_files(root, ("exists.txt", "missing.txt"))

        by_path = {row["path"]: row for row in rows}
        self.assertTrue(by_path["exists.txt"]["exists"])
        self.assertFalse(by_path["missing.txt"]["exists"])

    def test_build_findings_marks_dirty_worktree(self) -> None:
        findings = build_findings(git_status=" M src/x.py", missing_files=[], test_result="Ran 1 tests\nOK")

        self.assertEqual(findings[0]["area"], "worktree")
        self.assertEqual(findings[0]["severity"], "medium")

    def test_run_core_self_check_writes_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            root.mkdir()
            for relative in (
                "src/backtest_lab/decision_layers.py",
                "src/backtest_lab/regime_mode_switch.py",
                "src/backtest_lab/stock_pool_observation.py",
                "src/backtest_lab/stock_pool_consensus.py",
                "src/backtest_lab/margin_short_ingestion_spec.py",
                "src/backtest_lab/chip_valuation_event_study.py",
                ".github/workflows/stock_pool_observation.yml",
            ):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("ok", encoding="utf-8")
            output_dir = Path(tmp) / "out"
            runner = fake_runner(
                {
                    ("git", "status", "--short"): "",
                    ("git", "rev-parse", "--short", "HEAD"): "abc1234\n",
                }
            )

            run_core_self_check(
                repo_root=root,
                output_dir=output_dir,
                test_result="Ran 256 tests\nOK",
                runner=runner,
            )

            self.assertTrue((output_dir / "core_self_check.json").exists())
            self.assertTrue((output_dir / "core_self_check.md").exists())
            self.assertTrue((output_dir / "run_log.csv").exists())
            self.assertTrue((output_dir / "completed.txt").exists())


if __name__ == "__main__":
    unittest.main()
