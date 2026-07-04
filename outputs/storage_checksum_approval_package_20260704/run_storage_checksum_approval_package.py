import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
AUDIT_DIR = REPO / "outputs" / "data_governance_storage_audit_20260704"
OUT_DIR = Path(__file__).resolve().parent
TASK_ID = "TASK-BACKTEST-CORE-STORAGE-CHECKSUM-APPROVAL-PACKAGE-20260704"

PROTECTED_KEYWORDS = (
    "0050",
    "tw50",
    "00631",
    "formal_next_day",
    "current_formal",
    "backtest_cache",
    "price",
    "liquidity",
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    current_step = OUT_DIR / "current_step.txt"
    current_step.write_text("running checksum approval package\n", encoding="utf-8")

    candidate_path = AUDIT_DIR / "disposable_rebuildable_candidates.csv"
    candidates = []
    with candidate_path.open("r", newline="", encoding="utf-8-sig") as fh:
        candidates = list(csv.DictReader(fh))

    checksum_rows = []
    approval_rows = []
    protected_review_rows = []
    total_bytes = 0

    for row in candidates:
        rel = row["path"]
        root = REPO / rel
        lower_rel = rel.lower()
        protected_hit = any(k in lower_rel for k in PROTECTED_KEYWORDS)
        if protected_hit:
            protected_review_rows.append(
                {
                    "path": rel,
                    "size_mb": row.get("size_mb", ""),
                    "reason": "path contains protected keyword; requires manual review before any cleanup",
                    "cleanup_allowed_without_review": False,
                }
            )
        file_rows = []
        if root.is_file():
            file_rows = [root]
        elif root.is_dir():
            file_rows = [p for p in root.rglob("*") if p.is_file()]
        for path in file_rows:
            size = path.stat().st_size
            total_bytes += size
            digest = sha256_file(path)
            checksum_rows.append(
                {
                    "candidate_path": rel,
                    "relative_file": path.relative_to(REPO).as_posix(),
                    "size_bytes": size,
                    "size_mb": round(size / 1024 / 1024, 6),
                    "sha256": digest,
                    "delete_executed": False,
                    "archive_executed": False,
                    "move_executed": False,
                    "approval_required": True,
                }
            )
        approval_rows.append(
            {
                "path": rel,
                "data_layer": row.get("data_layer", ""),
                "governance_bucket": row.get("governance_bucket", ""),
                "size_mb": row.get("size_mb", ""),
                "file_count": row.get("file_count", ""),
                "classification": row.get("recommended_action", ""),
                "why_rebuildable": row.get("reason", ""),
                "restore_or_rebuild_source": "original runner/source audit; verify before cleanup",
                "protected_keyword_hit": protected_hit,
                "approval_required": True,
                "delete_allowed_without_user_approval": False,
                "delete_executed": False,
                "archive_executed": False,
                "move_executed": False,
            }
        )

    write_csv(
        OUT_DIR / "file_checksum_manifest.csv",
        checksum_rows,
        [
            "candidate_path",
            "relative_file",
            "size_bytes",
            "size_mb",
            "sha256",
            "delete_executed",
            "archive_executed",
            "move_executed",
            "approval_required",
        ],
    )
    write_csv(
        OUT_DIR / "user_approval_table.csv",
        approval_rows,
        [
            "path",
            "data_layer",
            "governance_bucket",
            "size_mb",
            "file_count",
            "classification",
            "why_rebuildable",
            "restore_or_rebuild_source",
            "protected_keyword_hit",
            "approval_required",
            "delete_allowed_without_user_approval",
            "delete_executed",
            "archive_executed",
            "move_executed",
        ],
    )
    write_csv(
        OUT_DIR / "protected_keyword_review_rows.csv",
        protected_review_rows,
        ["path", "size_mb", "reason", "cleanup_allowed_without_review"],
    )

    manifest = {
        "task_id": TASK_ID,
        "status": "completed_approval_package_no_delete",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_audit": str(AUDIT_DIR),
        "candidate_rows": len(candidates),
        "checksum_rows": len(checksum_rows),
        "candidate_size_mb_from_audit": round(sum(float(r.get("size_mb") or 0) for r in candidates), 6),
        "hashed_size_mb": round(total_bytes / 1024 / 1024, 6),
        "protected_keyword_review_rows": len(protected_review_rows),
        "delete_executed": False,
        "archive_executed": False,
        "move_executed": False,
        "compress_executed": False,
        "formal_model_changed": False,
        "trade_decision_changed": False,
        "active_in_trade_decision": False,
        "user_approval_required_for_delete": True,
        "protected_boundaries": [
            "2014/11+ backtest-required data",
            "0050/TW50 PIT and 00631L related data",
            "TWSE/TPEx price and liquidity cache",
            "formal next-day ledgers and current formal outputs",
        ],
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "completed.csv").write_text("task_id,status\n%s,completed\n" % TASK_ID, encoding="utf-8")
    (OUT_DIR / "failed.csv").write_text("task_id,status,reason\n", encoding="utf-8")
    (OUT_DIR / "run_log.csv").write_text("step,status,rows\nchecksum_approval_package,completed,%d\n" % len(checksum_rows), encoding="utf-8")
    summary = f"""# Core storage checksum approval package

- 狀態：`completed_approval_package_no_delete`
- 來源 audit：`{AUDIT_DIR}`
- approval candidate rows：`{len(candidates)}`
- checksum rows：`{len(checksum_rows)}`
- audit candidate size：`{manifest['candidate_size_mb_from_audit']} MB`
- hashed size：`{manifest['hashed_size_mb']} MB`
- protected keyword review rows：`{len(protected_review_rows)}`

## 結論
- 本包只建立 checksum 與 approval table。
- 沒有刪除、搬移、壓縮或封存任何檔案。
- 所有 cleanup 都仍需使用者批准。

## 邊界
- `delete_executed=false`
- `archive_executed=false`
- `move_executed=false`
- `compress_executed=false`
- `formal_model_changed=false`
- `trade_decision_changed=false`
"""
    (OUT_DIR / "final_summary_zh.md").write_text(summary, encoding="utf-8")
    current_step.write_text("completed\n", encoding="utf-8")


if __name__ == "__main__":
    main()
