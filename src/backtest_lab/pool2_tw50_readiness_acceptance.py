from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_OUTPUT_DIR = "outputs/pool2_tw50_readiness_acceptance"
MIN_EXACT_COVERAGE = 0.95


def run_pool2_tw50_readiness_acceptance(
    *,
    readiness_path: str | Path,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    min_exact_coverage: float = MIN_EXACT_COVERAGE,
) -> dict[str, Any]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    current_step = root / "current_step.txt"
    run_log = root / "run_log.csv"
    completed = root / "completed.csv"
    failed = root / "failed.csv"
    _write_rows(run_log, [{"event": "started", "detail": str(readiness_path)}])
    current_step.write_text("loading_readiness\n", encoding="utf-8")

    source = Path(readiness_path)
    if not source.exists():
        result = _blocked_result(
            readiness_path=str(source),
            status="blocked_missing_readiness_file",
            blockers=[f"readiness file not found: {source}"],
        )
        _write_outputs(root, result, completed=False)
        current_step.write_text("blocked_missing_readiness_file\n", encoding="utf-8")
        return result

    payload = json.loads(source.read_text(encoding="utf-8"))
    current_step.write_text("evaluating_readiness_contract\n", encoding="utf-8")
    result = evaluate_pool2_tw50_readiness(payload, readiness_path=str(source), min_exact_coverage=min_exact_coverage)
    _write_outputs(root, result, completed=True)
    _write_rows(run_log, [{"event": "completed", "detail": result["acceptance_status"]}])
    _write_rows(failed, [] if result["acceptance_status"] in {"accepted_exact_tw50", "accepted_proxy_specific"} else [{"status": result["acceptance_status"], "blockers": "; ".join(result["blockers"])}])
    _write_rows(completed, [{"status": result["acceptance_status"], "formal_model_changed": False, "trade_decision_changed": False}])
    current_step.write_text("completed\n", encoding="utf-8")
    return result


def evaluate_pool2_tw50_readiness(
    payload: dict[str, Any],
    *,
    readiness_path: str = "",
    min_exact_coverage: float = MIN_EXACT_COVERAGE,
) -> dict[str, Any]:
    exact_ready = _truthy(payload.get("exact_tw50_official_constituents_ready"))
    proxy_ready = _truthy(payload.get("yuanta_0050_holdings_proxy_ready"))
    formal_ready = _truthy(payload.get("formal_ready"))
    is_proxy = _truthy(payload.get("is_proxy"))
    future_violations = int(payload.get("future_data_violation_count") or 0)
    accepted_proxy_rows = int(payload.get("accepted_proxy_rows") or 0)
    coverage_rows = _coverage_rows(payload.get("core_coverage_summary") or [])

    blockers: list[str] = []
    warnings: list[str] = []
    if future_violations:
        blockers.append(f"future_data_violation_count={future_violations}")
    if exact_ready and is_proxy:
        blockers.append("exact_tw50_official_constituents_ready cannot be true when is_proxy=true")
    if proxy_ready and accepted_proxy_rows <= 0:
        blockers.append("yuanta_0050_holdings_proxy_ready=true but accepted_proxy_rows=0")
    low_coverage = [
        f"{row['period']}={row['coverage_ratio']:.2%}"
        for row in coverage_rows
        if row["coverage_ratio"] < min_exact_coverage
    ]
    if exact_ready and low_coverage:
        blockers.append("exact TW50 readiness coverage below threshold: " + ", ".join(low_coverage))
    if not exact_ready and not proxy_ready:
        blockers.append("no exact TW50 or proxy-specific accepted rows are ready")
    if proxy_ready and not exact_ready:
        warnings.append("proxy-specific readiness only; Core must not treat proxy rows as exact TW50 constituents")
    if payload.get("status") == "blocked_waiting_user_files":
        warnings.append("waiting for manual source files")

    if blockers:
        acceptance_status = str(payload.get("status") or "blocked")
    elif exact_ready and formal_ready:
        acceptance_status = "accepted_exact_tw50"
    elif proxy_ready:
        acceptance_status = "accepted_proxy_specific"
    else:
        acceptance_status = "blocked"

    return {
        "schema_version": 1,
        "task_id": "TASK-BACKTEST-CORE-POOL2-PIT-REPLAY-COVERAGE-20260623",
        "decision_layer": "data_readiness",
        "active_in_trade_decision": False,
        "formal_model_changed": False,
        "trade_decision_changed": False,
        "readiness_path": readiness_path,
        "radar_status": str(payload.get("status") or ""),
        "acceptance_status": acceptance_status,
        "can_use_as_exact_tw50_constituents": acceptance_status == "accepted_exact_tw50",
        "can_use_as_0050_holdings_proxy": acceptance_status == "accepted_proxy_specific",
        "must_not_backfill_with_current_constituents": True,
        "exact_tw50_official_constituents_ready": exact_ready,
        "yuanta_0050_holdings_proxy_ready": proxy_ready,
        "formal_ready": formal_ready,
        "is_proxy": is_proxy,
        "accepted_proxy_rows": accepted_proxy_rows,
        "future_data_violation_count": future_violations,
        "min_exact_coverage": min_exact_coverage,
        "coverage": coverage_rows,
        "blockers": blockers,
        "warnings": warnings,
        "manual_pdf_dir": payload.get("manual_pdf_dir", ""),
        "missing_priority_one_files": payload.get("missing_priority_one_files", []),
        "next_actions": payload.get("next_actions", []),
    }


def _coverage_rows(raw_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in raw_rows:
        rows.append(
            {
                "period": str(raw.get("period") or ""),
                "checked_dates": int(raw.get("checked_dates") or 0),
                "ready_dates": int(raw.get("ready_dates") or 0),
                "gap_dates": int(raw.get("gap_dates") or 0),
                "coverage_ratio": float(raw.get("coverage_ratio") or 0.0),
                "first_ready_date": str(raw.get("first_ready_date") or ""),
                "last_ready_date": str(raw.get("last_ready_date") or ""),
            }
        )
    return rows


def _write_outputs(root: Path, result: dict[str, Any], *, completed: bool) -> None:
    (root / "pool2_tw50_readiness_acceptance.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame(result.get("coverage", [])).to_csv(
        root / "pool2_tw50_coverage_acceptance.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (root / "pool2_tw50_readiness_acceptance.md").write_text(_markdown(result), encoding="utf-8")
    if not completed:
        _write_rows(root / "failed.csv", [{"status": result["acceptance_status"], "blockers": "; ".join(result["blockers"])}])


def _markdown(result: dict[str, Any]) -> str:
    blocker_lines = [f"- {item}" for item in result.get("blockers", [])] or ["_無。_"]
    missing_file_lines = [f"- `{item}`" for item in result.get("missing_priority_one_files", [])] or ["_無。_"]
    lines = [
        "# Pool2 TW50/0050 Readiness Acceptance",
        "",
        f"- acceptance_status: `{result['acceptance_status']}`",
        f"- radar_status: `{result['radar_status']}`",
        f"- exact_tw50_official_constituents_ready: `{result['exact_tw50_official_constituents_ready']}`",
        f"- yuanta_0050_holdings_proxy_ready: `{result['yuanta_0050_holdings_proxy_ready']}`",
        f"- can_use_as_exact_tw50_constituents: `{result['can_use_as_exact_tw50_constituents']}`",
        f"- can_use_as_0050_holdings_proxy: `{result['can_use_as_0050_holdings_proxy']}`",
        f"- future_data_violation_count: `{result['future_data_violation_count']}`",
        "",
        "## Blockers",
        "",
        *blocker_lines,
        "",
        "## Missing Priority-1 Files",
        "",
        *missing_file_lines,
        "",
        "Core boundary: proxy-specific rows cannot be passed as exact official TW50 constituents.",
    ]
    return "\n".join(lines)


def _blocked_result(*, readiness_path: str, status: str, blockers: list[str]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "task_id": "TASK-BACKTEST-CORE-POOL2-PIT-REPLAY-COVERAGE-20260623",
        "decision_layer": "data_readiness",
        "active_in_trade_decision": False,
        "formal_model_changed": False,
        "trade_decision_changed": False,
        "readiness_path": readiness_path,
        "radar_status": "",
        "acceptance_status": status,
        "can_use_as_exact_tw50_constituents": False,
        "can_use_as_0050_holdings_proxy": False,
        "must_not_backfill_with_current_constituents": True,
        "blockers": blockers,
        "warnings": [],
        "coverage": [],
    }


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    columns = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Accept or block Pool2 TW50/0050 PIT readiness handoff.")
    parser.add_argument("--readiness-path", required=True)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-exact-coverage", type=float, default=MIN_EXACT_COVERAGE)
    args = parser.parse_args()
    result = run_pool2_tw50_readiness_acceptance(
        readiness_path=args.readiness_path,
        output_dir=args.output_dir,
        min_exact_coverage=args.min_exact_coverage,
    )
    print(json.dumps({key: result[key] for key in ("acceptance_status", "can_use_as_exact_tw50_constituents", "can_use_as_0050_holdings_proxy")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
