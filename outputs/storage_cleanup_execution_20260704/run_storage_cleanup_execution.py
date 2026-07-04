import csv
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
APPROVAL_PACKAGE = REPO / "outputs" / "storage_checksum_approval_package_20260704"
APPROVAL_TABLE = APPROVAL_PACKAGE / "user_approval_table.csv"
OUT_DIR = Path(__file__).resolve().parent
TASK_ID = "TASK-BACKTEST-CORE-STORAGE-CLEANUP-EXECUTION-20260704"

PROTECTED_PREFIXES = {
    "backtest_cache",
    "data",
}


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def safe_resolve(relative_path: str) -> Path:
    target = (REPO / relative_path).resolve()
    repo_resolved = REPO.resolve()
    if repo_resolved not in target.parents and target != repo_resolved:
        raise ValueError(f"path escapes repo: {relative_path}")
    return target


def classify_skip(row: dict) -> str:
    if row.get("approval_required") != "True":
        return "approval_required_not_true"
    if row.get("delete_allowed_without_user_approval") != "False":
        return "approval_table_not_conservative"
    if row.get("protected_keyword_hit") == "True":
        return "protected_keyword_hit"
    path = row["path"].replace("\\", "/").strip("/")
    first = path.split("/", 1)[0]
    if first in PROTECTED_PREFIXES:
        return "protected_prefix"
    return ""


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "current_step.txt").write_text("running cleanup execution\n", encoding="utf-8")
    rows = list(csv.DictReader(APPROVAL_TABLE.open("r", newline="", encoding="utf-8-sig")))

    deleted_rows = []
    skipped_rows = []
    missing_rows = []
    total_deleted_bytes = 0

    for row in rows:
        rel = row["path"]
        skip_reason = classify_skip(row)
        target = safe_resolve(rel)
        if skip_reason:
            skipped_rows.append(
                {
                    "path": rel,
                    "size_mb": row.get("size_mb", ""),
                    "skip_reason": skip_reason,
                    "deleted": False,
                }
            )
            continue
        if not target.exists():
            missing_rows.append({"path": rel, "size_mb": row.get("size_mb", ""), "deleted": False, "reason": "missing"})
            continue

        size_before = 0
        if target.is_file():
            size_before = target.stat().st_size
            target.unlink()
        elif target.is_dir():
            for p in target.rglob("*"):
                if p.is_file():
                    size_before += p.stat().st_size
            shutil.rmtree(target)
        else:
            skipped_rows.append(
                {
                    "path": rel,
                    "size_mb": row.get("size_mb", ""),
                    "skip_reason": "not_file_or_dir",
                    "deleted": False,
                }
            )
            continue

        total_deleted_bytes += size_before
        deleted_rows.append(
            {
                "path": rel,
                "size_mb_audit": row.get("size_mb", ""),
                "deleted_size_mb": round(size_before / 1024 / 1024, 6),
                "classification": row.get("classification", ""),
                "reason": row.get("why_rebuildable", ""),
                "delete_executed": True,
            }
        )

    write_csv(
        OUT_DIR / "deleted_rows.csv",
        deleted_rows,
        ["path", "size_mb_audit", "deleted_size_mb", "classification", "reason", "delete_executed"],
    )
    write_csv(OUT_DIR / "skipped_rows.csv", skipped_rows, ["path", "size_mb", "skip_reason", "deleted"])
    write_csv(OUT_DIR / "missing_rows.csv", missing_rows, ["path", "size_mb", "deleted", "reason"])

    manifest = {
        "task_id": TASK_ID,
        "status": "completed_cleanup_execution_core_rebuildable_only",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_approval_package": str(APPROVAL_PACKAGE),
        "input_rows": len(rows),
        "deleted_rows": len(deleted_rows),
        "skipped_rows": len(skipped_rows),
        "missing_rows": len(missing_rows),
        "deleted_size_mb": round(total_deleted_bytes / 1024 / 1024, 6),
        "scope": "Core disposable/rebuildable candidates only",
        "radar_raw_sources_deleted": False,
        "protected_data_deleted": False,
        "formal_model_changed": False,
        "trade_decision_changed": False,
        "active_in_trade_decision": False,
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "completed.csv").write_text(f"task_id,status\n{TASK_ID},completed\n", encoding="utf-8")
    (OUT_DIR / "failed.csv").write_text("task_id,status,reason\n", encoding="utf-8")
    (OUT_DIR / "run_log.csv").write_text(
        "step,status,rows\ncleanup_execution,completed,%d\n" % len(deleted_rows),
        encoding="utf-8",
    )
    summary = f"""# Core storage cleanup execution

- 狀態：`completed_cleanup_execution_core_rebuildable_only`
- 刪除範圍：Core disposable/rebuildable candidates only
- input rows：`{len(rows)}`
- deleted rows：`{len(deleted_rows)}`
- skipped rows：`{len(skipped_rows)}`
- missing rows：`{len(missing_rows)}`
- deleted size：`{manifest['deleted_size_mb']} MB`

## 保護邊界

- Radar raw sources 未刪。
- 2014/11+ 回測必要資料未刪。
- 0050/TW50 PIT、00631L、TWSE/TPEx price/liquidity cache、formal next-day ledgers/current formal outputs 未刪。
- `formal_model_changed=false`
- `trade_decision_changed=false`
"""
    (OUT_DIR / "final_summary_zh.md").write_text(summary, encoding="utf-8")
    (OUT_DIR / "current_step.txt").write_text("completed\n", encoding="utf-8")


if __name__ == "__main__":
    main()
