from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd


TASK_ID = "TASK-BACKTEST-CORE-STORAGE-CHECKSUM-APPROVAL-PACKAGE-20260704"
DEFAULT_AUDIT_DIR = "outputs/data_governance_storage_audit_20260704"
DEFAULT_OUTPUT_DIR = "outputs/storage_checksum_approval_package_20260704"
PROTECTED_KEYWORDS = [
    "2014/11+ backtest-required data",
    "0050/TW50 PIT",
    "00631L",
    "TWSE/TPEx price/liquidity cache",
    "formal next-day ledgers",
    "current formal outputs",
]


def run_storage_checksum_approval_package(
    *,
    repo_root: str | Path = ".",
    audit_dir: str | Path = DEFAULT_AUDIT_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> Path:
    root = Path(repo_root).resolve()
    audit = Path(audit_dir)
    if not audit.is_absolute():
        audit = root / audit
    output = Path(output_dir)
    if not output.is_absolute():
        output = root / output
    output.mkdir(parents=True, exist_ok=True)
    run_log: list[dict[str, str]] = []

    def log(step: str, status: str, detail: str = "") -> None:
        run_log.append(
            {
                "timestamp": pd.Timestamp.now(tz="Asia/Taipei").strftime("%Y-%m-%d %H:%M:%S%z"),
                "step": step,
                "status": status,
                "detail": detail,
            }
        )
        pd.DataFrame(run_log).to_csv(output / "run_log.csv", index=False, encoding="utf-8-sig")
        (output / "current_step.txt").write_text(f"{step}:{status}\n{detail}", encoding="utf-8")

    try:
        log("load_core_audit", "started", str(audit))
        audit_manifest = _load_json(audit / "manifest.json")
        disposable = _read_csv_required(audit / "disposable_rebuildable_candidates.csv")
        keep_required = _read_csv_required(audit / "keep_required_for_backtest.csv")

        log("build_checksums", "started", f"{len(disposable)} candidates")
        checksum = _build_checksum_manifest(root, disposable)
        approval = _build_approval_table(disposable, checksum)
        protected = _protected_confirmation(keep_required, audit_manifest)
        manifest = _manifest(output, audit, audit_manifest, checksum, approval, protected)

        log("write_outputs", "started", str(output))
        checksum.to_csv(output / "disposable_candidate_checksum_manifest.csv", index=False, encoding="utf-8-sig")
        approval.to_csv(output / "user_approval_table.csv", index=False, encoding="utf-8-sig")
        protected.to_csv(output / "protected_boundaries_confirmation.csv", index=False, encoding="utf-8-sig")
        (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        (output / "final_summary_zh.md").write_text(_summary_zh(manifest), encoding="utf-8")
        pd.DataFrame([{"step": TASK_ID, "status": "completed_checksum_approval_package", "output_dir": str(output)}]).to_csv(
            output / "completed.csv",
            index=False,
            encoding="utf-8-sig",
        )
        pd.DataFrame(columns=["step", "status", "reason"]).to_csv(output / "failed.csv", index=False, encoding="utf-8-sig")
        log("completed", "completed", str(output))
        return output
    except Exception as exc:
        pd.DataFrame([{"step": TASK_ID, "status": "failed", "reason": str(exc)}]).to_csv(
            output / "failed.csv",
            index=False,
            encoding="utf-8-sig",
        )
        log("failed", "failed", str(exc))
        raise


def _build_checksum_manifest(root: Path, candidates: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for item in candidates.to_dict(orient="records"):
        rel_path = str(item.get("path", ""))
        path = root / rel_path
        digest, file_count, size_bytes, skipped = _aggregate_sha256(path, root)
        rows.append(
            {
                "path": rel_path,
                "path_exists": path.exists(),
                "classification": item.get("governance_bucket", ""),
                "recommended_action": item.get("recommended_action", ""),
                "size_bytes_from_audit": int(item.get("size_bytes", 0) or 0),
                "size_mb_from_audit": float(item.get("size_mb", 0) or 0),
                "checksummed_file_count": file_count,
                "checksummed_size_bytes": size_bytes,
                "aggregate_sha256": digest,
                "checksum_skipped_files": skipped,
                "delete_executed": False,
                "move_executed": False,
                "compress_executed": False,
                "archive_executed": False,
            }
        )
    return pd.DataFrame(rows)


def _aggregate_sha256(path: Path, root: Path) -> tuple[str, int, int, int]:
    if not path.exists():
        return "", 0, 0, 0
    hasher = hashlib.sha256()
    file_count = 0
    size = 0
    skipped = 0
    files: list[Path] = []
    if path.is_file():
        files = [path]
    else:
        for dirpath, dirnames, filenames in os.walk(path):
            dirnames[:] = [name for name in dirnames if name != ".git"]
            for filename in filenames:
                files.append(Path(dirpath) / filename)
    for file_path in sorted(files, key=lambda item: item.relative_to(root).as_posix()):
        try:
            rel = file_path.relative_to(root).as_posix()
            stat = file_path.stat()
            file_hash = hashlib.sha256()
            with file_path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    file_hash.update(chunk)
            hasher.update(rel.encode("utf-8"))
            hasher.update(str(stat.st_size).encode("ascii"))
            hasher.update(file_hash.hexdigest().encode("ascii"))
            file_count += 1
            size += stat.st_size
        except OSError:
            skipped += 1
    return hasher.hexdigest(), file_count, size, skipped


def _build_approval_table(candidates: pd.DataFrame, checksum: pd.DataFrame) -> pd.DataFrame:
    merged = candidates.merge(
        checksum[["path", "aggregate_sha256", "checksummed_file_count", "checksummed_size_bytes", "checksum_skipped_files"]],
        on="path",
        how="left",
    )
    merged["why_rebuildable"] = merged["reason"].astype(str)
    merged["restore_or_rebuild_source"] = merged.apply(_restore_source, axis=1)
    merged["approval_required"] = True
    merged["approval_status"] = "pending_user_approval"
    merged["delete_executed"] = False
    merged["move_executed"] = False
    merged["compress_executed"] = False
    merged["archive_executed"] = False
    merged["classification"] = merged["governance_bucket"]
    return merged[
        [
            "path",
            "size_bytes",
            "size_mb",
            "classification",
            "governance_bucket",
            "recommended_action",
            "why_rebuildable",
            "restore_or_rebuild_source",
            "aggregate_sha256",
            "checksummed_file_count",
            "checksummed_size_bytes",
            "checksum_skipped_files",
            "approval_required",
            "approval_status",
            "delete_executed",
            "move_executed",
            "compress_executed",
            "archive_executed",
        ]
    ]


def _restore_source(row: pd.Series) -> str:
    path = str(row.get("path", ""))
    if path.startswith("outputs/"):
        return "Rebuild from committed runner/source manifest when needed; preserve current approval package before cleanup."
    if path.startswith("tmp") or path.startswith(".codex_tmp"):
        return "Temporary workspace artifact; no production restore expected, but user approval is required before deletion."
    if path.startswith("work"):
        return "Workspace generated artifact; review owner and runner before deletion."
    return "Review source manifest or runner before cleanup."


def _protected_confirmation(keep_required: pd.DataFrame, audit_manifest: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for keyword in PROTECTED_KEYWORDS:
        rows.append(
            {
                "protected_boundary": keyword,
                "protected_confirmed": True,
                "delete_executed": False,
                "move_executed": False,
                "compress_executed": False,
                "archive_executed": False,
                "evidence": "Core audit keep_required_for_backtest plus explicit boundary from integrated judgment.",
            }
        )
    rows.append(
        {
            "protected_boundary": "keep_required_for_backtest rows",
            "protected_confirmed": True,
            "delete_executed": False,
            "move_executed": False,
            "compress_executed": False,
            "archive_executed": False,
            "evidence": f"{len(keep_required)} rows / {audit_manifest.get('keep_required_size_mb', 0)} MB remain protected.",
        }
    )
    return pd.DataFrame(rows)


def _manifest(
    output: Path,
    audit: Path,
    audit_manifest: dict[str, Any],
    checksum: pd.DataFrame,
    approval: pd.DataFrame,
    protected: pd.DataFrame,
) -> dict[str, Any]:
    total_bytes = int(pd.to_numeric(approval["size_bytes"], errors="coerce").fillna(0).sum()) if not approval.empty else 0
    return {
        "schema_version": 1,
        "task_id": TASK_ID,
        "status": "completed_checksum_approval_package_no_action",
        "generated_at": pd.Timestamp.now(tz="Asia/Taipei").isoformat(),
        "output_dir": str(output),
        "source_core_audit": str(audit),
        "candidate_rows": int(len(approval)),
        "candidate_size_mb_from_audit": round(total_bytes / 1024 / 1024, 3),
        "checksum_rows": int(len(checksum)),
        "protected_boundary_rows": int(len(protected)),
        "delete_executed": False,
        "move_executed": False,
        "compress_executed": False,
        "archive_executed": False,
        "formal_model_changed": False,
        "trade_decision_changed": False,
        "active_in_trade_decision": False,
        "user_approval_required_for_any_action": True,
        "core_audit_total_scanned_mb": audit_manifest.get("total_scanned_size_mb", 0),
    }


def _summary_zh(manifest: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Core storage checksum approval package",
            "",
            f"- 狀態：{manifest['status']}",
            f"- approval candidates：{manifest['candidate_rows']} rows",
            f"- audit candidate size：約 {manifest['candidate_size_mb_from_audit']} MB",
            "- 本次只建立 checksum manifest 與 user approval table。",
            "- 沒有刪除、搬移、壓縮或封存任何資料。",
            "- 2014/11+、0050/TW50 PIT、00631L、TWSE/TPEx price/liquidity、formal ledgers/current outputs 仍維持 protected。",
        ]
    )


def _read_csv_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path).fillna("")


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build checksum approval package for Core disposable storage candidates.")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--audit-dir", default=DEFAULT_AUDIT_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    run_storage_checksum_approval_package(repo_root=args.repo_root, audit_dir=args.audit_dir, output_dir=args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
