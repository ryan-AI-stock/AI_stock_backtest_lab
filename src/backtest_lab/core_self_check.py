from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd


CommandRunner = Callable[[list[str], Path], subprocess.CompletedProcess[str]]


@dataclass(frozen=True)
class CoreSelfCheckContext:
    repo_root: Path
    output_dir: Path
    test_command: str = "PYTHONPATH=src python -m unittest discover -s tests"
    test_result: str = ""


REQUIRED_CORE_FILES = (
    "src/backtest_lab/decision_layers.py",
    "src/backtest_lab/regime_mode_switch.py",
    "src/backtest_lab/stock_pool_observation.py",
    "src/backtest_lab/stock_pool_consensus.py",
    "src/backtest_lab/margin_short_ingestion_spec.py",
    "src/backtest_lab/chip_valuation_event_study.py",
)

REQUIRED_WORKFLOWS = (
    ".github/workflows/frozen_strategy_daily_report.yml",
    ".github/workflows/model_scorecard_report.yml",
    ".github/workflows/stock_pool_observation.yml",
)


def default_runner(command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=False)


def run_core_self_check(
    *,
    repo_root: str | Path = ".",
    output_dir: str | Path = "outputs/core_self_check",
    test_result: str = "",
    runner: CommandRunner = default_runner,
) -> Path:
    context = CoreSelfCheckContext(
        repo_root=Path(repo_root).resolve(),
        output_dir=Path(output_dir).resolve(),
        test_result=test_result,
    )
    context.output_dir.mkdir(parents=True, exist_ok=True)
    run_log: list[dict] = []

    def log(step: str, status: str, detail: str = "") -> None:
        run_log.append(
            {
                "timestamp": pd.Timestamp.now(tz="Asia/Taipei").strftime("%Y-%m-%d %H:%M:%S%z"),
                "step": step,
                "status": status,
                "detail": detail,
            }
        )
        pd.DataFrame(run_log).to_csv(context.output_dir / "run_log.csv", index=False, encoding="utf-8-sig")
        (context.output_dir / "current_step.txt").write_text(step + "\n", encoding="utf-8")

    log("collect_git", "started")
    git_status = collect_git_status(context.repo_root, runner)
    git_head = collect_git_head(context.repo_root, runner)
    log("collect_git", "completed", git_head or "unknown")

    log("check_files", "started")
    file_checks = check_required_files(context.repo_root, REQUIRED_CORE_FILES + REQUIRED_WORKFLOWS)
    missing_files = [item["path"] for item in file_checks if not item["exists"]]
    log("check_files", "completed", f"missing={len(missing_files)}")

    findings = build_findings(
        git_status=git_status,
        missing_files=missing_files,
        test_result=test_result,
    )
    payload = {
        "task_type": "core_self_check",
        "repo_root": str(context.repo_root),
        "git_head": git_head,
        "git_status_short": git_status,
        "worktree_clean": git_status.strip() == "",
        "required_file_checks": file_checks,
        "test_command": context.test_command,
        "test_result": test_result,
        "findings": findings,
        "next_formal_engine_goal": next_formal_engine_goal(),
        "guardrails": [
            "No formal model mutation in this self-check.",
            "No Drive/PDF/LINE/workflow entrypoint change in this self-check.",
            "Research or Experiments output must pass Core validation before formal promotion.",
            "Frozen baseline remains protected.",
        ],
    }
    (context.output_dir / "core_self_check.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (context.output_dir / "core_self_check.md").write_text(render_markdown(payload), encoding="utf-8")
    (context.output_dir / "completed.txt").write_text("completed\n", encoding="utf-8")
    log("completed", "completed", str(context.output_dir))
    return context.output_dir


def collect_git_status(repo_root: Path, runner: CommandRunner = default_runner) -> str:
    result = runner(["git", "status", "--short"], repo_root)
    return (result.stdout or "").strip()


def collect_git_head(repo_root: Path, runner: CommandRunner = default_runner) -> str:
    result = runner(["git", "rev-parse", "--short", "HEAD"], repo_root)
    return (result.stdout or "").strip()


def check_required_files(repo_root: Path, paths: tuple[str, ...]) -> list[dict[str, object]]:
    rows = []
    for relative in paths:
        path = repo_root / relative
        rows.append(
            {
                "path": relative,
                "exists": path.exists(),
                "size_bytes": path.stat().st_size if path.exists() else 0,
            }
        )
    return rows


def build_findings(*, git_status: str, missing_files: list[str], test_result: str) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    if git_status.strip():
        findings.append(
            {
                "severity": "medium",
                "area": "worktree",
                "finding": "Worktree has uncommitted changes.",
                "recommended_action": "Review before any formal model or workflow change.",
            }
        )
    if missing_files:
        findings.append(
            {
                "severity": "high",
                "area": "guardrail",
                "finding": f"Missing required Core guardrail files: {', '.join(missing_files)}.",
                "recommended_action": "Restore missing files before changing model behavior.",
            }
        )
    if "OK" not in test_result and test_result:
        findings.append(
            {
                "severity": "high",
                "area": "tests",
                "finding": "Latest test evidence does not clearly show OK.",
                "recommended_action": "Rerun tests and inspect failures before proceeding.",
            }
        )
    if not findings:
        findings.append(
            {
                "severity": "low",
                "area": "health",
                "finding": "No blocking Core health issue found in this self-check.",
                "recommended_action": "Proceed with versioned challengers or data-readiness work only.",
            }
        )
    return findings


def next_formal_engine_goal() -> dict[str, str]:
    return {
        "goal": "Upgrade data-readiness and factor coverage before any new formal model promotion.",
        "reason": (
            "Recent chip/valuation event study found no robust improvement over baseline and exposed "
            "2024-2026 factor coverage gaps."
        ),
        "safe_next_step": (
            "Add coverage validators for factor/event-study inputs, then hand missing 2024-2026 "
            "official factor data requirements to Research/Radar before testing new challengers."
        ),
    }


def render_markdown(payload: dict) -> str:
    lines = [
        "# Core 自我檢測報告",
        "",
        f"- repo：`{payload['repo_root']}`",
        f"- git head：`{payload['git_head']}`",
        f"- worktree clean：`{payload['worktree_clean']}`",
        f"- test：`{payload['test_result'] or 'not_provided'}`",
        "",
        "## 發現",
    ]
    for finding in payload["findings"]:
        lines.append(
            f"- [{finding['severity']}] {finding['area']}：{finding['finding']} "
            f"下一步：{finding['recommended_action']}"
        )
    lines.extend(
        [
            "",
            "## 下一個正式工程目標",
            "",
            f"- 目標：{payload['next_formal_engine_goal']['goal']}",
            f"- 理由：{payload['next_formal_engine_goal']['reason']}",
            f"- 安全下一步：{payload['next_formal_engine_goal']['safe_next_step']}",
            "",
            "## 邊界",
        ]
    )
    lines.extend(f"- {guardrail}" for guardrail in payload["guardrails"])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a Core health/self-check report.")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--output-dir", default="outputs/core_self_check")
    parser.add_argument("--test-result", default="")
    args = parser.parse_args()
    output = run_core_self_check(
        repo_root=args.repo_root,
        output_dir=args.output_dir,
        test_result=args.test_result,
    )
    print(f"OUTPUT_DIR={output}")


if __name__ == "__main__":
    main()
