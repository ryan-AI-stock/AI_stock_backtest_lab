from __future__ import annotations

import subprocess
from pathlib import Path

from backtest_lab.portfolio_app_settings import (
    DEFAULT_GITHUB_REF,
    DEFAULT_GITHUB_REPO,
    DEFAULT_WORKFLOW_FILE,
    PORTFOLIO_SECRET_NAME,
)


def sync_portfolio_secret(
    *,
    store_path: str | Path,
    repo: str = DEFAULT_GITHUB_REPO,
    runner=subprocess.run,
) -> dict:
    path = Path(store_path)
    if not path.exists():
        raise ValueError("尚未找到本機持倉檔，請先儲存目前資產。")
    secret_body = path.read_text(encoding="utf-8")
    result = _run_gh(
        [
            "gh",
            "secret",
            "set",
            PORTFOLIO_SECRET_NAME,
            "--repo",
            repo,
        ],
        runner=runner,
        action="sync_secret",
        input_text=secret_body,
    )
    return {
        "action": "sync_secret",
        "repo": repo,
        "secret_name": PORTFOLIO_SECRET_NAME,
        "message": "已同步本機持倉到 GitHub repo secret。",
        **result,
    }


def trigger_report_workflow(
    *,
    signal_date: str,
    repo: str = DEFAULT_GITHUB_REPO,
    workflow_file: str = DEFAULT_WORKFLOW_FILE,
    ref: str = DEFAULT_GITHUB_REF,
    runner=subprocess.run,
) -> dict:
    if not signal_date:
        raise ValueError("尚未找到訊號日期，無法觸發 GitHub Action。")
    result = _run_gh(
        [
            "gh",
            "workflow",
            "run",
            workflow_file,
            "--repo",
            repo,
            "--ref",
            ref,
            "-f",
            f"signal_date={signal_date}",
        ],
        runner=runner,
        action="run_workflow",
    )
    return {
        "action": "run_workflow",
        "repo": repo,
        "workflow_file": workflow_file,
        "signal_date": signal_date,
        "message": "已觸發 GitHub Action，報告完成後會覆蓋 Drive 最新版 PDF。",
        **result,
    }


def _run_gh(args: list[str], *, runner, action: str, input_text: str | None = None) -> dict:
    completed = runner(args, input=input_text, capture_output=True, text=True, timeout=90)
    stdout = str(getattr(completed, "stdout", "") or "").strip()
    stderr = str(getattr(completed, "stderr", "") or "").strip()
    returncode = int(getattr(completed, "returncode", 0))
    if returncode != 0:
        detail = stderr or stdout or f"{action} failed with exit code {returncode}."
        raise ValueError(detail)
    return {"ok": True, "stdout": stdout}
