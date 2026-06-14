from __future__ import annotations

import csv
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from backtest_lab.stock_pool_store import normalize_ticker

CSV_SOURCE_MODES = {"ai_theme_candidate_csv", "core_defensive_candidate_csv"}


def build_candidate_review_decision_draft(
    *,
    pools: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
) -> dict[str, Any]:
    pool_by_id = {str(pool.get("pool_id") or ""): pool for pool in pools}
    changes: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for decision in decisions:
        pool_id = str(decision.get("pool_id") or "")
        pool = pool_by_id.get(pool_id)
        if not pool:
            skipped.append({**_decision_ref(decision), "reason": "pool_not_found"})
            continue
        config = pool.get("candidate_review_config") or {}
        source_mode = str(config.get("source_mode") or "")
        source_path = Path(str(config.get("path") or ""))
        if source_mode not in CSV_SOURCE_MODES:
            skipped.append({**_decision_ref(decision), "reason": "unsupported_source_mode", "source_mode": source_mode})
            continue
        if not source_path.exists():
            skipped.append({**_decision_ref(decision), "reason": "source_missing", "source_path": str(source_path)})
            continue
        changes.append(_draft_csv_change(decision=decision, pool=pool, source_path=source_path))
    return {
        "status": "ready",
        "change_count": len(changes),
        "skipped_count": len(skipped),
        "changes": changes,
        "skipped": skipped,
    }


def apply_candidate_review_decision_draft(
    *,
    pools: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
    backup_root: str | Path,
) -> dict[str, Any]:
    draft = build_candidate_review_decision_draft(pools=pools, decisions=decisions)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = Path(backup_root) / timestamp
    backup_dir.mkdir(parents=True, exist_ok=True)
    applied: list[dict[str, Any]] = []
    grouped: dict[str, list[dict[str, Any]]] = {}
    for change in draft.get("changes", []):
        grouped.setdefault(str(change.get("source_path") or ""), []).append(change)
    for source_path_text, changes in grouped.items():
        source_path = Path(source_path_text)
        backup_path = backup_dir / source_path.name
        shutil.copy2(source_path, backup_path)
        rows = _read_rows(source_path)
        fieldnames = _fieldnames_for_apply(rows, changes)
        rows = _apply_changes_to_rows(rows, changes, fieldnames)
        _write_rows(source_path, rows, fieldnames)
        applied.append(
            {
                "source_path": str(source_path),
                "backup_path": str(backup_path),
                "change_count": len(changes),
                "tickers": [str(change.get("ticker") or "") for change in changes],
            }
        )
    result = {
        "status": "applied",
        "applied_at": timestamp,
        "applied_source_count": len(applied),
        "applied_change_count": sum(item["change_count"] for item in applied),
        "applied": applied,
        "skipped": draft.get("skipped", []),
    }
    (backup_dir / "candidate_review_apply_log.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result


def _draft_csv_change(*, decision: dict[str, Any], pool: dict[str, Any], source_path: Path) -> dict[str, Any]:
    rows = _read_rows(source_path)
    columns = list(rows[0].keys()) if rows else []
    ticker = normalize_ticker(str(decision.get("ticker") or ""))
    symbol = ticker.split(".")[0]
    existing = next((row for row in rows if normalize_ticker(str(row.get("ticker") or row.get("symbol") or "")) == ticker), None)
    draft_fields = _fields_for_decision(decision)
    action = "update_row" if existing else "append_row"
    row_preview = dict(existing or _new_row(columns, decision=decision, ticker=ticker, symbol=symbol))
    row_preview.update(draft_fields)
    if "review_reason" in row_preview and decision.get("note"):
        row_preview["review_reason"] = str(decision.get("note"))
    return {
        **_decision_ref(decision),
        "pool_name": pool.get("name", ""),
        "source_path": str(source_path),
        "action": action,
        "current_status": (existing or {}).get("review_status", ""),
        "draft_status": row_preview.get("review_status", ""),
        "draft_is_current_member": row_preview.get("is_current_member", ""),
        "fields_to_change": draft_fields,
        "row_preview": row_preview,
    }


def _fields_for_decision(decision: dict[str, Any]) -> dict[str, str]:
    action = str(decision.get("decision") or "")
    if action == "approve_add":
        return {"review_status": "active", "is_current_member": "true"}
    if action == "keep_current":
        return {"review_status": "active", "is_current_member": "true"}
    if action == "keep_watch":
        return {"review_status": "watch", "is_current_member": "false"}
    if action == "reject":
        return {"review_status": "rejected", "is_current_member": "false"}
    return {}


def _new_row(columns: list[str], *, decision: dict[str, Any], ticker: str, symbol: str) -> dict[str, str]:
    row = {column: "" for column in columns}
    row["effective_date"] = str(decision.get("signal_date") or "")
    row["ticker"] = ticker
    row["symbol"] = symbol
    row["name"] = _name_from_display(str(decision.get("display") or symbol))
    row["review_reason"] = str(decision.get("note") or "由月頻候選審核決策新增。")
    return row


def _decision_ref(decision: dict[str, Any]) -> dict[str, str]:
    return {
        "pool_id": str(decision.get("pool_id") or ""),
        "ticker": str(decision.get("ticker") or ""),
        "display": str(decision.get("display") or decision.get("ticker") or ""),
        "decision": str(decision.get("decision") or ""),
        "decision_label": str(decision.get("decision_label") or ""),
        "signal_date": str(decision.get("signal_date") or ""),
        "note": str(decision.get("note") or ""),
    }


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_rows(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
    temporary.replace(path)


def _fieldnames_for_apply(rows: list[dict[str, str]], changes: list[dict[str, Any]]) -> list[str]:
    fieldnames = list(rows[0].keys()) if rows else []
    for change in changes:
        for field in (change.get("row_preview") or {}).keys():
            if field not in fieldnames:
                fieldnames.append(field)
    return fieldnames


def _apply_changes_to_rows(
    rows: list[dict[str, str]],
    changes: list[dict[str, Any]],
    fieldnames: list[str],
) -> list[dict[str, str]]:
    output = [dict(row) for row in rows]
    for change in changes:
        ticker = normalize_ticker(str(change.get("ticker") or ""))
        row_preview = {field: str(value) for field, value in (change.get("row_preview") or {}).items()}
        existing_index = next(
            (
                index
                for index, row in enumerate(output)
                if normalize_ticker(str(row.get("ticker") or row.get("symbol") or "")) == ticker
            ),
            None,
        )
        row_preview = {field: row_preview.get(field, "") for field in fieldnames}
        if existing_index is None:
            output.append(row_preview)
        else:
            merged = {field: output[existing_index].get(field, "") for field in fieldnames}
            merged.update(row_preview)
            output[existing_index] = merged
    return output


def _name_from_display(display: str) -> str:
    return display.split("(")[0].strip() or display
