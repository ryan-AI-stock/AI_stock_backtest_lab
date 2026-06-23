from __future__ import annotations

import argparse
import csv
import glob
import io
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from backtest_lab.stock_pool_store import normalize_ticker


EVENT_COLUMNS = [
    "effective_date",
    "event_type",
    "ticker",
    "name",
    "index_name",
    "source_date",
    "source_title",
    "source_url",
    "source_type",
    "exact_or_proxy",
    "accepted",
    "blocked_reason",
]

INTERVAL_COLUMNS = [
    "effective_date",
    "end_date",
    "ticker",
    "name",
    "source",
    "source_updated_at",
]


@dataclass(frozen=True)
class NoticeIngestionResult:
    output_dir: str
    event_rows: int
    accepted_event_rows: int
    blocked_event_rows: int
    readiness_status: str
    blockers: list[str]
    outputs: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_dir": self.output_dir,
            "event_rows": self.event_rows,
            "accepted_event_rows": self.accepted_event_rows,
            "blocked_event_rows": self.blocked_event_rows,
            "readiness_status": self.readiness_status,
            "blockers": self.blockers,
            "outputs": self.outputs,
            "formal_ready": False,
            "reason": "technical notice events are source evidence; full PIT intervals still require an official baseline snapshot on or before the first event",
        }


def parse_tw50_technical_notice_text(
    text: str,
    *,
    source_url: str = "",
    source_title: str = "",
) -> list[dict[str, Any]]:
    normalized = _normalize_text(text)
    title = source_title or _extract_title(normalized)
    source_date = _extract_first_date(normalized)
    effective_date = _extract_effective_date(normalized)
    rows: list[dict[str, Any]] = []

    if "臺灣50" not in normalized and "台灣50" not in normalized:
        return []

    if "延期" in normalized and not _contains_add_delete_section(normalized):
        rows.append(
            _event_row(
                effective_date=effective_date,
                event_type="blocked_notice",
                ticker="",
                name="",
                source_date=source_date,
                source_title=title,
                source_url=source_url,
                accepted=False,
                blocked_reason="postponed_or_non_constituent_change_notice",
            )
        )
        return rows

    section_events = _parse_review_sections(
        normalized,
        source_date=source_date,
        source_title=title,
        source_url=source_url,
        effective_date=effective_date,
    )
    if section_events:
        return section_events

    return _parse_table_lines(
        normalized,
        source_date=source_date,
        source_title=title,
        source_url=source_url,
        effective_date=effective_date,
    )


def parse_tw50_technical_notice_pdf(
    payload: bytes,
    *,
    source_url: str = "",
    source_title: str = "",
) -> list[dict[str, Any]]:
    text = ""
    try:
        import pdfplumber

        with pdfplumber.open(io.BytesIO(payload)) as pdf:
            text = "\n".join(page.extract_text() or "" for page in pdf.pages)
    except ImportError:
        text = ""
    except Exception:
        text = ""

    if text.strip():
        return parse_tw50_technical_notice_text(text, source_url=source_url, source_title=source_title)

    try:
        from pypdf import PdfReader
    except ImportError as error:  # pragma: no cover - dependency is present in normal runtime.
        raise ValueError("pypdf is required to parse Taiwan Index technical notice PDF.") from error

    reader = PdfReader(io.BytesIO(payload))
    text = "\n".join(page.extract_text(extraction_mode="layout") or "" for page in reader.pages)
    return parse_tw50_technical_notice_text(text, source_url=source_url, source_title=source_title)


def run_tw50_technical_notice_ingestion(
    *,
    input_paths: Iterable[str | Path],
    output_dir: str | Path,
    source_manifest: str | Path | None = None,
) -> NoticeIngestionResult:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    run_log = root / "run_log.csv"
    current_step = root / "current_step.txt"
    failed = root / "failed.csv"
    completed = root / "completed.csv"
    source_audit = root / "tw50_technical_notice_source_audit.csv"
    event_csv = root / "tw50_technical_notice_events.csv"
    metadata_path = root / "metadata.json"
    report_path = root / "tw50_technical_notice_events.md"

    _write_rows(run_log, [{"event": "started", "detail": ""}])
    current_step.write_text("loading source manifest\n", encoding="utf-8")
    manifest = _load_source_manifest(source_manifest)

    event_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for raw_path in _expand_input_paths(input_paths):
        path = Path(raw_path)
        current_step.write_text(f"parsing {path}\n", encoding="utf-8")
        source_info = manifest.get(path.name, {})
        try:
            payload = path.read_bytes()
            if payload.lstrip().startswith(b"%PDF"):
                rows = parse_tw50_technical_notice_pdf(
                    payload,
                    source_url=str(source_info.get("source_url", "")),
                    source_title=str(source_info.get("source_title", path.stem)),
                )
            else:
                rows = parse_tw50_technical_notice_text(
                    path.read_text(encoding="utf-8"),
                    source_url=str(source_info.get("source_url", "")),
                    source_title=str(source_info.get("source_title", path.stem)),
                )
            event_rows.extend(rows)
            audit_rows.append(
                {
                    "file": str(path),
                    "source_url": source_info.get("source_url", ""),
                    "source_title": source_info.get("source_title", path.stem),
                    "download_status": "provided_to_core",
                    "parse_status": "parsed" if rows else "parsed_no_tw50_event",
                    "event_rows": len(rows),
                    "accepted_event_rows": sum(1 for row in rows if _truthy(row.get("accepted"))),
                    "exact_or_proxy": "exact_candidate",
                }
            )
        except Exception as error:  # pragma: no cover - covered through CLI resilience.
            failures.append({"file": str(path), "error": str(error)})
            audit_rows.append(
                {
                    "file": str(path),
                    "source_url": source_info.get("source_url", ""),
                    "source_title": source_info.get("source_title", path.stem),
                    "download_status": "provided_to_core",
                    "parse_status": "failed",
                    "event_rows": 0,
                    "accepted_event_rows": 0,
                    "exact_or_proxy": "exact_candidate",
                }
            )

    event_frame = pd.DataFrame(event_rows, columns=EVENT_COLUMNS)
    event_frame.to_csv(event_csv, index=False, encoding="utf-8-sig")
    _write_rows(source_audit, audit_rows)
    _write_rows(failed, failures)

    accepted_count = int(event_frame["accepted"].astype(str).str.lower().isin({"true", "1", "yes"}).sum()) if not event_frame.empty else 0
    blockers: list[str] = []
    if accepted_count == 0:
        blockers.append("no accepted TW50 add/delete event rows parsed")
    blockers.append("PIT interval build still requires an official baseline constituent snapshot on or before the first event")
    readiness_status = "events_ready_pending_baseline_snapshot" if accepted_count else "blocked_no_accepted_events"

    result = NoticeIngestionResult(
        output_dir=str(root),
        event_rows=len(event_rows),
        accepted_event_rows=accepted_count,
        blocked_event_rows=len(event_rows) - accepted_count,
        readiness_status=readiness_status,
        blockers=blockers,
        outputs={
            "events": str(event_csv),
            "source_audit": str(source_audit),
            "metadata": str(metadata_path),
            "markdown": str(report_path),
            "run_log": str(run_log),
            "completed": str(completed),
            "failed": str(failed),
        },
    )
    metadata_path.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(_notice_markdown(result), encoding="utf-8")
    _write_rows(completed, [{"status": readiness_status, "event_rows": len(event_rows), "accepted_event_rows": accepted_count}])
    current_step.write_text("completed\n", encoding="utf-8")
    return result


def build_tw50_pit_intervals_from_events(
    *,
    baseline_snapshot_path: str | Path,
    event_rows_path: str | Path,
    output_path: str | Path,
    source_updated_at: str,
) -> dict[str, Any]:
    baseline = pd.read_csv(baseline_snapshot_path)
    events = pd.read_csv(event_rows_path)
    if baseline.empty:
        raise ValueError("baseline snapshot is empty.")
    if "effective_date" not in baseline.columns or "ticker" not in baseline.columns:
        raise ValueError("baseline snapshot must include effective_date and ticker columns.")
    if not set(EVENT_COLUMNS).issubset(events.columns):
        raise ValueError("event rows file does not match TW50 technical notice event schema.")

    baseline_effective = pd.to_datetime(baseline["effective_date"], errors="coerce").min()
    if pd.isna(baseline_effective):
        raise ValueError("baseline snapshot contains invalid effective_date values.")
    accepted_events = events[events["accepted"].astype(str).str.lower().isin({"true", "1", "yes"})].copy()
    if not accepted_events.empty:
        first_event = pd.to_datetime(accepted_events["effective_date"], errors="coerce").min()
        if pd.isna(first_event):
            raise ValueError("accepted events contain invalid effective_date values.")
        if first_event < baseline_effective:
            raise ValueError(
                "baseline snapshot starts after the first accepted event; cannot build point-in-time intervals without future leakage."
            )

    active: dict[str, dict[str, str]] = {}
    rows: list[dict[str, str]] = []
    for _, row in baseline.iterrows():
        ticker = normalize_ticker(str(row["ticker"]))
        item = {
            "effective_date": _date_to_str(row["effective_date"]),
            "end_date": "",
            "ticker": ticker,
            "name": _clean_name(str(row.get("name", ""))),
            "source": str(row.get("source", "official_baseline_snapshot")),
            "source_updated_at": source_updated_at,
        }
        active[ticker] = item
        rows.append(item)

    accepted_events = accepted_events.sort_values(["effective_date", "event_type", "ticker"])
    for _, event in accepted_events.iterrows():
        ticker = normalize_ticker(str(event["ticker"]))
        effective = pd.Timestamp(event["effective_date"])
        if str(event["event_type"]) == "delete":
            if ticker not in active:
                continue
            active[ticker]["end_date"] = (effective - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            active.pop(ticker, None)
        elif str(event["event_type"]) == "add":
            if ticker in active:
                continue
            item = {
                "effective_date": effective.strftime("%Y-%m-%d"),
                "end_date": "",
                "ticker": ticker,
                "name": _clean_name(str(event.get("name", ""))),
                "source": str(event.get("source_url") or event.get("source_title") or "official_technical_notice"),
                "source_updated_at": source_updated_at,
            }
            active[ticker] = item
            rows.append(item)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=INTERVAL_COLUMNS).to_csv(output, index=False, encoding="utf-8-sig")
    return {
        "output_path": str(output),
        "row_count": len(rows),
        "baseline_effective_date": baseline_effective.strftime("%Y-%m-%d"),
        "accepted_event_rows": len(accepted_events),
        "formal_ready": True,
    }


def build_tw50_pit_snapshots_from_current_and_events(
    *,
    current_snapshot_path: str | Path,
    snapshot_as_of: str,
    event_rows_path: str | Path,
    output_path: str | Path,
    history_start: str,
    source_updated_at: str,
) -> dict[str, Any]:
    """Reconstruct PIT snapshots backward from an official current TW50 snapshot.

    This is valid only when the current snapshot is official exact evidence and
    the event rows cover every add/delete change between history_start and
    snapshot_as_of. It writes full snapshots at history_start and every accepted
    event effective date, so the existing point-in-time loader can select the
    latest active basket without needing per-ticker end dates.
    """

    snapshot = _read_current_snapshot(current_snapshot_path)
    events = pd.read_csv(event_rows_path)
    if not set(EVENT_COLUMNS).issubset(events.columns):
        raise ValueError("event rows file does not match TW50 technical notice event schema.")
    accepted = events[events["accepted"].astype(str).str.lower().isin({"true", "1", "yes"})].copy()
    if accepted.empty:
        raise ValueError("no accepted TW50 event rows available.")

    snapshot_date = pd.Timestamp(snapshot_as_of).normalize()
    start_date = pd.Timestamp(history_start).normalize()
    accepted["effective_ts"] = pd.to_datetime(accepted["effective_date"], errors="coerce")
    if accepted["effective_ts"].isna().any():
        raise ValueError("accepted events contain invalid effective_date values.")
    usable_events = accepted[(accepted["effective_ts"] >= start_date) & (accepted["effective_ts"] <= snapshot_date)].copy()
    if usable_events.empty:
        raise ValueError("no accepted events fall within history_start and snapshot_as_of.")

    active: dict[str, str] = {}
    for _, row in snapshot.iterrows():
        ticker = normalize_ticker(str(row["ticker"]))
        active[ticker] = _clean_name(str(row.get("name", "")))
    rows: list[dict[str, str]] = []
    event_dates_desc = sorted({pd.Timestamp(value).normalize() for value in usable_events["effective_ts"]}, reverse=True)
    source = f"official_current_snapshot_reverse_events:{Path(current_snapshot_path).name}"

    for event_date in event_dates_desc:
        _append_snapshot_rows(
            rows,
            effective_date=event_date.strftime("%Y-%m-%d"),
            active=active,
            source=source,
            source_updated_at=source_updated_at,
        )
        group = usable_events[usable_events["effective_ts"] == event_date]
        for _, event in group.iterrows():
            event_type = str(event["event_type"])
            ticker = normalize_ticker(str(event["ticker"]))
            if event_type == "add":
                active.pop(ticker, None)
            elif event_type == "delete":
                active[ticker] = _clean_name(str(event.get("name", "")))

    _append_snapshot_rows(
        rows,
        effective_date=start_date.strftime("%Y-%m-%d"),
        active=active,
        source=source,
        source_updated_at=source_updated_at,
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows, columns=["effective_date", "ticker", "name", "source", "source_updated_at"])
    frame = frame.drop_duplicates(subset=["effective_date", "ticker"], keep="last").sort_values(
        ["effective_date", "ticker"]
    )
    frame.to_csv(output, index=False, encoding="utf-8-sig")
    counts = frame.groupby("effective_date")["ticker"].nunique()
    low_counts = {date: int(count) for date, count in counts.items() if count < 45}
    return {
        "output_path": str(output),
        "row_count": len(frame),
        "snapshot_count": int(counts.size),
        "snapshot_as_of": snapshot_date.strftime("%Y-%m-%d"),
        "history_start": start_date.strftime("%Y-%m-%d"),
        "accepted_event_rows_used": int(len(usable_events)),
        "min_snapshot_count": int(counts.min()) if not counts.empty else 0,
        "max_snapshot_count": int(counts.max()) if not counts.empty else 0,
        "low_snapshot_counts": low_counts,
        "formal_ready": not low_counts,
    }


def _read_current_snapshot(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    payload = source.read_bytes()
    if payload.lstrip().startswith(b"%PDF"):
        from backtest_lab.tw50_constituents_update import parse_ftse_tw50_pdf

        return parse_ftse_tw50_pdf(payload)
    return pd.read_csv(source)


def _append_snapshot_rows(
    rows: list[dict[str, str]],
    *,
    effective_date: str,
    active: dict[str, str],
    source: str,
    source_updated_at: str,
) -> None:
    for ticker, name in sorted(active.items()):
        rows.append(
            {
                "effective_date": effective_date,
                "ticker": ticker,
                "name": name,
                "source": source,
                "source_updated_at": source_updated_at,
            }
        )


def _parse_review_sections(
    text: str,
    *,
    source_date: str,
    source_title: str,
    source_url: str,
    effective_date: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for event_type, heading in (("add", "成分股納入"), ("delete", "成分股刪除")):
        section = _section_after_heading(text, heading)
        for item in _extract_name_code_pairs(section):
            rows.append(
                _event_row(
                    effective_date=effective_date,
                    event_type=event_type,
                    ticker=normalize_ticker(item["code"]),
                    name=item["name"],
                    source_date=source_date,
                    source_title=source_title,
                    source_url=source_url,
                    accepted=bool(effective_date),
                    blocked_reason="" if effective_date else "missing_effective_date",
                )
            )
    return rows


def _parse_table_lines(
    text: str,
    *,
    source_date: str,
    source_title: str,
    source_url: str,
    effective_date: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        if "臺灣50" not in line and "台灣50" not in line:
            continue
        if "延期" in line:
            rows.append(
                _event_row(
                    effective_date=effective_date or _extract_first_date(line),
                    event_type="blocked_notice",
                    ticker="",
                    name="",
                    source_date=source_date,
                    source_title=source_title,
                    source_url=source_url,
                    accepted=False,
                    blocked_reason="postponed_or_non_constituent_change_notice",
                )
            )
            continue
        for event_type, pattern in (
            ("add", r"(?:新增|納入)(?P<body>.*?)(?:刪除|剔除|$)"),
            ("delete", r"(?:刪除|剔除)(?P<body>.*)$"),
        ):
            match = re.search(pattern, line)
            if not match:
                continue
            for item in _extract_name_code_pairs(match.group("body")):
                rows.append(
                    _event_row(
                        effective_date=effective_date or _extract_first_date(line),
                        event_type=event_type,
                        ticker=normalize_ticker(item["code"]),
                        name=item["name"],
                        source_date=source_date,
                        source_title=source_title,
                        source_url=source_url,
                        accepted=bool(effective_date or _extract_first_date(line)),
                        blocked_reason="" if (effective_date or _extract_first_date(line)) else "missing_effective_date",
                    )
                )
    return rows


def _event_row(
    *,
    effective_date: str,
    event_type: str,
    ticker: str,
    name: str,
    source_date: str,
    source_title: str,
    source_url: str,
    accepted: bool,
    blocked_reason: str,
) -> dict[str, Any]:
    return {
        "effective_date": effective_date,
        "event_type": event_type,
        "ticker": ticker,
        "name": _clean_name(name),
        "index_name": "臺灣50指數",
        "source_date": source_date,
        "source_title": source_title,
        "source_url": source_url,
        "source_type": "official_technical_notice",
        "exact_or_proxy": "exact_candidate",
        "accepted": bool(accepted),
        "blocked_reason": blocked_reason,
    }


def _section_after_heading(text: str, heading: str) -> str:
    heading_pattern = rf"{re.escape(heading)}\s*(?:\(\d+\)|（\d+）)?\s*[:：]"
    pattern = (
        rf"{heading_pattern}(?P<body>.*?)"
        rf"(?=成分股(?:納入|刪除|候補名單)\s*(?:\(\d+\)|（\d+）)?\s*[:：]|[一二三四五六七八九十]+、|$)"
    )
    match = re.search(pattern, text, flags=re.DOTALL)
    return match.group("body") if match else ""


def _extract_name_code_pairs(text: str) -> list[dict[str, str]]:
    pairs: list[dict[str, str]] = []
    for match in re.finditer(r"(?P<name>[^\s\d,，;；:：()（）]{1,24})\s*(?P<code>\d{4})", text):
        name = _clean_name(match.group("name"))
        code = match.group("code")
        if name and code and _looks_like_tw_stock_code(code) and _looks_like_stock_name(name):
            pairs.append({"name": name, "code": code})
    return pairs


def _looks_like_tw_stock_code(code: str) -> bool:
    return bool(re.fullmatch(r"[1-9]\d{3}", code))


def _looks_like_stock_name(name: str) -> bool:
    if not name or name in {".", "-", "*"}:
        return False
    return bool(re.search(r"[\u4e00-\u9fffA-Za-z]", name))


def _extract_title(text: str) -> str:
    for line in text.splitlines():
        line = line.strip()
        if "臺灣50" in line or "台灣50" in line or "臺灣指數系列" in line:
            return line
    return ""


def _extract_first_date(text: str) -> str:
    match = re.search(r"(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日", text)
    return _date_match_to_str(match) if match else ""


def _extract_effective_date(text: str) -> str:
    candidates = re.findall(
        r"(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日[^。\n]{0,40}?(?:起生效|生效日|生效)",
        text,
    )
    if candidates:
        return _date_tuple_to_str(candidates[-1])
    table_date = re.search(r"生效日\s*.*?(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日", text, flags=re.DOTALL)
    return _date_match_to_str(table_date) if table_date else ""


def _date_match_to_str(match: re.Match[str]) -> str:
    return _date_tuple_to_str((match.group(1), match.group(2), match.group(3)))


def _date_tuple_to_str(parts: tuple[str, str, str] | tuple[Any, Any, Any]) -> str:
    year, month, day = parts
    return f"{int(year):04d}-{int(month):02d}-{int(day):02d}"


def _contains_add_delete_section(text: str) -> bool:
    return "成分股納入" in text or "成分股刪除" in text


def _normalize_text(text: str) -> str:
    lines = [" ".join(line.strip().split()) for line in text.replace("\r", "\n").splitlines()]
    return "\n".join(line for line in lines if line)


def _clean_name(value: str) -> str:
    return value.strip().strip("、,，;；。")


def _date_to_str(value: Any) -> str:
    parsed = pd.Timestamp(value)
    if pd.isna(parsed):
        raise ValueError(f"invalid date: {value}")
    return parsed.strftime("%Y-%m-%d")


def _load_source_manifest(path: str | Path | None) -> dict[str, dict[str, Any]]:
    if not path:
        return {}
    source = Path(path)
    if not source.exists():
        return {}
    if source.suffix.lower() == ".json":
        payload = json.loads(source.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return {str(item.get("file") or item.get("filename") or ""): item for item in payload}
        if isinstance(payload, dict):
            return {str(key): value for key, value in payload.items() if isinstance(value, dict)}
    rows: dict[str, dict[str, Any]] = {}
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            key = str(row.get("file") or row.get("filename") or "")
            if key:
                rows[key] = dict(row)
    return rows


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _notice_markdown(result: NoticeIngestionResult) -> str:
    return "\n".join(
        [
            "# TW50 Technical Notice Event Ingestion",
            "",
            f"- readiness_status: `{result.readiness_status}`",
            f"- event_rows: `{result.event_rows}`",
            f"- accepted_event_rows: `{result.accepted_event_rows}`",
            f"- blocked_event_rows: `{result.blocked_event_rows}`",
            "",
            "## Core Boundary",
            "",
            "- These rows are exact-candidate Taiwan Index technical notice events.",
            "- They are not sufficient for formal Pool2 replay until a point-in-time baseline snapshot exists on or before the first event.",
            "- Yuanta 0050 holdings/monthly reports remain proxy candidates and must not be mixed into exact TW50 constituents.",
            "",
            "## Blockers",
            "",
            *(f"- {item}" for item in result.blockers),
            "",
        ]
    )


def _expand_input_paths(input_paths: Iterable[str | Path]) -> list[Path]:
    expanded: list[Path] = []
    for raw_path in input_paths:
        text = str(raw_path)
        if any(char in text for char in "*?[]"):
            matches = [Path(item) for item in glob.glob(text)]
            expanded.extend(sorted(matches))
        else:
            expanded.append(Path(raw_path))
    return expanded


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse Taiwan Index TW50 technical notices into PIT event rows.")
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--output-dir", default="outputs/tw50_technical_notice_events")
    parser.add_argument("--source-manifest", default="")
    args = parser.parse_args()
    result = run_tw50_technical_notice_ingestion(
        input_paths=args.input,
        output_dir=args.output_dir,
        source_manifest=args.source_manifest or None,
    )
    print(json.dumps(result.to_dict(), ensure_ascii=False))


if __name__ == "__main__":
    main()
