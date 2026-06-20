from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from backtest_lab.tw50_constituents import load_tw50_constituents_for_date


DEFAULT_PERIODS = {
    "2022": ("2022-01-03", "2022-12-30"),
    "2023": ("2023-01-03", "2023-12-29"),
    "2024_2026": ("2024-01-02", "2026-06-18"),
}
DEFAULT_OUTPUT_DIR = "outputs/tw50_constituent_coverage"
REQUIRED_COLUMNS = {"effective_date", "ticker"}


def run_tw50_constituent_coverage(
    *,
    constituent_path: str | Path = "data/tw50_constituents.csv",
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    periods: dict[str, tuple[str, str]] | None = None,
    date_stride: int = 1,
    minimum_active_count: int = 45,
) -> dict[str, Any]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    current_step = root / "current_step.txt"
    run_log = root / "run_log.csv"
    _write_csv(run_log, [{"event": "started", "detail": str(constituent_path)}])
    current_step.write_text("loading constituent source\n", encoding="utf-8")

    source = Path(constituent_path)
    rows: list[dict[str, Any]] = []
    gap_rows: list[dict[str, Any]] = []
    readiness_status = "blocked"
    source_columns: list[str] = []
    source_row_count = 0
    source_effective_min = ""
    source_effective_max = ""
    source_error = ""

    if not source.exists():
        source_error = f"TW50 constituent file not found: {source}"
    else:
        frame = pd.read_csv(source)
        source_columns = list(frame.columns)
        source_row_count = len(frame)
        missing_columns = REQUIRED_COLUMNS - set(frame.columns)
        if missing_columns:
            source_error = "missing_columns:" + ",".join(sorted(missing_columns))
        else:
            effective = pd.to_datetime(frame["effective_date"], errors="coerce")
            source_effective_min = "" if effective.dropna().empty else str(effective.min().date())
            source_effective_max = "" if effective.dropna().empty else str(effective.max().date())

            current_step.write_text("checking period coverage\n", encoding="utf-8")
            period_ranges = periods or DEFAULT_PERIODS
            for period_name, (start, end) in period_ranges.items():
                dates = _business_dates(start, end, stride=max(1, date_stride))
                checked = 0
                ready = 0
                min_active: int | None = None
                max_active = 0
                first_ready = ""
                last_ready = ""
                for signal_date in dates:
                    checked += 1
                    status = _date_status(source, signal_date, minimum_active_count)
                    active_count = int(status.get("active_count") or 0)
                    min_active = active_count if min_active is None else min(min_active, active_count)
                    max_active = max(max_active, active_count)
                    if status["ready"]:
                        ready += 1
                        if not first_ready:
                            first_ready = signal_date.strftime("%Y-%m-%d")
                        last_ready = signal_date.strftime("%Y-%m-%d")
                    else:
                        gap_rows.append(
                            {
                                "period": period_name,
                                "signal_date": signal_date.strftime("%Y-%m-%d"),
                                "gap_reason": status["reason"],
                                "active_count": active_count,
                            }
                        )
                coverage = round(ready / checked, 8) if checked else 0.0
                rows.append(
                    {
                        "period": period_name,
                        "start": start,
                        "end": end,
                        "checked_dates": checked,
                        "ready_dates": ready,
                        "gap_dates": checked - ready,
                        "coverage_ratio": coverage,
                        "minimum_active_count": minimum_active_count,
                        "min_active_count": min_active if min_active is not None else 0,
                        "max_active_count": max_active,
                        "first_ready_date": first_ready,
                        "last_ready_date": last_ready,
                    }
                )
            readiness_status = _overall_status(rows)

    _write_csv(root / "tw50_constituent_coverage_summary.csv", rows)
    _write_csv(root / "tw50_constituent_gap_dates.csv", gap_rows)
    report = _markdown_report(
        rows=rows,
        gap_rows=gap_rows,
        source_path=str(source),
        source_error=source_error,
        readiness_status=readiness_status,
        source_row_count=source_row_count,
        source_effective_min=source_effective_min,
        source_effective_max=source_effective_max,
    )
    (root / "tw50_constituent_coverage.md").write_text(report, encoding="utf-8")
    metadata = {
        "schema_version": 1,
        "status": "completed" if not source_error else "blocked",
        "readiness_status": readiness_status,
        "purpose": "validate_tw50_point_in_time_constituent_coverage_for_stock_pool_replay",
        "constituent_path": str(source),
        "source_error": source_error,
        "source_columns": source_columns,
        "source_row_count": source_row_count,
        "source_effective_min": source_effective_min,
        "source_effective_max": source_effective_max,
        "date_stride": date_stride,
        "minimum_active_count": minimum_active_count,
        "outputs": {
            "summary": str(root / "tw50_constituent_coverage_summary.csv"),
            "gaps": str(root / "tw50_constituent_gap_dates.csv"),
            "markdown": str(root / "tw50_constituent_coverage.md"),
            "run_log": str(run_log),
        },
        "rows": {
            "summary": len(rows),
            "gaps": len(gap_rows),
        },
    }
    (root / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    (root / "completed.txt").write_text("completed\n", encoding="utf-8")
    current_step.write_text("completed\n", encoding="utf-8")
    return metadata


def _date_status(source: Path, signal_date: pd.Timestamp, minimum_active_count: int) -> dict[str, Any]:
    try:
        symbols = load_tw50_constituents_for_date(source, signal_date)
    except (FileNotFoundError, ValueError) as exc:
        return {"ready": False, "reason": str(exc), "active_count": 0}
    active_count = len(symbols)
    if active_count < minimum_active_count:
        return {
            "ready": False,
            "reason": f"active_count_below_minimum:{active_count}<{minimum_active_count}",
            "active_count": active_count,
        }
    return {"ready": True, "reason": "ready", "active_count": active_count}


def _overall_status(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "blocked_no_source"
    if all(float(row["coverage_ratio"]) >= 0.95 for row in rows):
        return "ready"
    if any(float(row["coverage_ratio"]) > 0 for row in rows):
        return "partial_blocked"
    return "blocked_no_historical_coverage"


def _business_dates(start: str, end: str, *, stride: int) -> list[pd.Timestamp]:
    dates = list(pd.bdate_range(start, end))
    return [date.normalize() for offset, date in enumerate(dates) if offset % stride == 0]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def _markdown_report(
    *,
    rows: list[dict[str, Any]],
    gap_rows: list[dict[str, Any]],
    source_path: str,
    source_error: str,
    readiness_status: str,
    source_row_count: int,
    source_effective_min: str,
    source_effective_max: str,
) -> str:
    lines = [
        "# TW50 / 0050 成分股歷史覆蓋檢查",
        "",
        f"- readiness_status: `{readiness_status}`",
        f"- source_path: `{source_path}`",
        f"- source_row_count: `{source_row_count}`",
        f"- source_effective_range: `{source_effective_min}` ~ `{source_effective_max}`",
    ]
    if source_error:
        lines.append(f"- source_error: `{source_error}`")
    lines.extend(
        [
            "",
            "## 覆蓋摘要",
            "",
            _markdown_table(rows),
            "",
            "## 前 30 筆缺口",
            "",
            _markdown_table(gap_rows[:30]),
            "",
            "使用邊界：本檢查只驗證 point-in-time 0050 成分股資料是否足以支撐池2歷史 replay；不使用現代成分股硬回推歷史，不改正式模型。",
        ]
    )
    return "\n".join(lines)


def _markdown_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "_無資料。_"
    columns = list(rows[0].keys())
    output = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        output.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    return "\n".join(output)


def _parse_periods(values: list[str] | None) -> dict[str, tuple[str, str]]:
    if not values:
        return DEFAULT_PERIODS
    selected: dict[str, tuple[str, str]] = {}
    for value in values:
        if value in DEFAULT_PERIODS:
            selected[value] = DEFAULT_PERIODS[value]
            continue
        name, raw_range = value.split("=", 1)
        start, end = raw_range.split(":", 1)
        selected[name] = (start, end)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate TW50 point-in-time constituent coverage.")
    parser.add_argument("--constituent-path", default="data/tw50_constituents.csv")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--period", action="append", help="Known period key or name=start:end. Can repeat.")
    parser.add_argument("--date-stride", type=int, default=1)
    parser.add_argument("--minimum-active-count", type=int, default=45)
    args = parser.parse_args()
    metadata = run_tw50_constituent_coverage(
        constituent_path=args.constituent_path,
        output_dir=args.output_dir,
        periods=_parse_periods(args.period),
        date_stride=args.date_stride,
        minimum_active_count=args.minimum_active_count,
    )
    print(f"TW50_CONSTITUENT_COVERAGE={Path(metadata['outputs']['markdown']).resolve()}")


if __name__ == "__main__":
    main()
