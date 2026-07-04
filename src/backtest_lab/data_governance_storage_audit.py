from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd


TASK_ID = "TASK-BACKTEST-CORE-DATA-GOVERNANCE-STORAGE-AUDIT-20260704"
DEFAULT_OUTPUT_DIR = "outputs/data_governance_storage_audit_20260704"
SCAN_ROOTS = ["backtest_cache", "backtest_outputs", "data", "outputs", "tmp", "work", ".codex_tmp"]
PROTECTED_TERMS = [
    "0050",
    "00631l",
    "current_formal",
    "formal_long_range",
    "formal_daily",
    "formal_replay",
    "combined_formal_target_stream",
    "formal_target_stream",
    "long_range_data_completion",
    "pool1_full_state",
    "pool1_warmup",
    "pool2_date_batched",
    "persistence_reconstruction",
    "next_day",
    "previous_best_next_day",
    "no_target_cash_all",
    "price_coverage",
    "pit",
    "stock_pool_observations",
]
DIAGNOSTIC_TERMS = [
    "diagnostic",
    "readiness",
    "taxonomy",
    "dynamic_pool1",
    "short_cycle",
    "sector",
    "challenger",
    "evidence",
    "research",
    "robustness",
]
REBUILDABLE_TERMS = [
    "tmp",
    "debug",
    "smoke",
    "preliminary",
    "failed",
    "ad_hoc",
    "challenger_search_20260613",
    "__pycache__",
]
ARCHIVE_TERMS = ["raw_sources", "raw_source", "shards", "cache_compatible", "archive"]


def run_data_governance_storage_audit(
    *,
    repo_root: str | Path = ".",
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> Path:
    root = Path(repo_root).resolve()
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
        log("collect_tracked_files", "started", str(root))
        tracked = _git_tracked_files(root)
        log("scan_storage_inventory", "started", ",".join(SCAN_ROOTS))
        inventory = _build_inventory(root, tracked)
        keep_required = inventory[inventory["governance_bucket"].eq("keep_required_for_backtest")].copy()
        diagnostic_keep = inventory[inventory["governance_bucket"].eq("diagnostic_keep_optional")].copy()
        archive = inventory[inventory["governance_bucket"].eq("archive_or_compress_candidate")].copy()
        disposable = inventory[inventory["governance_bucket"].eq("disposable_rebuildable_candidate")].copy()
        approval = inventory[inventory["requires_user_approval_before_action"].eq(True)].copy()
        manifest = _manifest(output, inventory, keep_required, diagnostic_keep, archive, disposable, approval)

        log("write_outputs", "started", str(output))
        inventory.to_csv(output / "storage_inventory.csv", index=False, encoding="utf-8-sig")
        keep_required.to_csv(output / "keep_required_for_backtest.csv", index=False, encoding="utf-8-sig")
        diagnostic_keep.to_csv(output / "diagnostic_keep_optional.csv", index=False, encoding="utf-8-sig")
        archive.to_csv(output / "archive_or_compress_candidates.csv", index=False, encoding="utf-8-sig")
        disposable.to_csv(output / "disposable_rebuildable_candidates.csv", index=False, encoding="utf-8-sig")
        approval.to_csv(output / "deletion_requires_user_approval.csv", index=False, encoding="utf-8-sig")
        (output / "data_layer_governance_plan.md").write_text(_governance_plan(manifest), encoding="utf-8")
        (output / "risk_and_dependency_notes.md").write_text(_risk_notes(), encoding="utf-8")
        (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        (output / "final_summary_zh.md").write_text(_summary_zh(manifest), encoding="utf-8")
        pd.DataFrame([{"step": TASK_ID, "status": "completed_audit_only", "output_dir": str(output)}]).to_csv(
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


def _git_tracked_files(root: Path) -> set[str]:
    try:
        result = subprocess.run(
            ["git", "ls-files"],
            cwd=root,
            text=True,
            capture_output=True,
            check=False,
        )
    except Exception:
        return set()
    if result.returncode != 0:
        return set()
    return {line.strip().replace("\\", "/") for line in result.stdout.splitlines() if line.strip()}


def _build_inventory(root: Path, tracked: set[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for scan_root in SCAN_ROOTS:
        path = root / scan_root
        if not path.exists():
            continue
        if scan_root == "outputs":
            children = [item for item in path.iterdir() if item.is_dir()]
        elif scan_root in {"backtest_cache", "backtest_outputs", "data", "tmp", "work", ".codex_tmp"}:
            children = [path]
            children.extend([item for item in path.iterdir() if item.is_dir()])
        else:
            children = [path]
        seen: set[Path] = set()
        for child in children:
            if child in seen or not child.exists():
                continue
            seen.add(child)
            stats = _path_stats(child, root, tracked)
            bucket, action, reason = _classify(child.relative_to(root).as_posix(), stats)
            rows.append(
                {
                    "path": child.relative_to(root).as_posix(),
                    "data_layer": _data_layer(child.relative_to(root).as_posix()),
                    "governance_bucket": bucket,
                    "recommended_action": action,
                    "reason": reason,
                    **stats,
                    "requires_user_approval_before_action": action not in {"keep", "keep_optional"},
                    "delete_executed": False,
                    "formal_model_changed": False,
                    "trade_decision_changed": False,
                    "active_in_trade_decision": False,
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=_inventory_columns())
    frame = frame.sort_values("size_bytes", ascending=False).reset_index(drop=True)
    return frame[_inventory_columns()]


def _path_stats(path: Path, root: Path, tracked: set[str]) -> dict[str, Any]:
    size = 0
    file_count = 0
    tracked_count = 0
    newest_mtime = 0.0
    largest_file = ""
    largest_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        dirnames[:] = [name for name in dirnames if name not in {".git"}]
        for filename in filenames:
            file_path = Path(dirpath) / filename
            try:
                stat = file_path.stat()
            except OSError:
                continue
            rel = file_path.relative_to(root).as_posix()
            size += stat.st_size
            file_count += 1
            newest_mtime = max(newest_mtime, stat.st_mtime)
            if rel in tracked:
                tracked_count += 1
            if stat.st_size > largest_size:
                largest_size = stat.st_size
                largest_file = rel
    return {
        "size_bytes": int(size),
        "size_mb": round(size / 1024 / 1024, 3),
        "file_count": int(file_count),
        "git_tracked_file_count": int(tracked_count),
        "untracked_file_count_estimate": int(file_count - tracked_count),
        "newest_mtime": pd.Timestamp.fromtimestamp(newest_mtime).isoformat() if newest_mtime else "",
        "largest_file": largest_file,
        "largest_file_mb": round(largest_size / 1024 / 1024, 3),
    }


def _classify(path: str, stats: dict[str, Any]) -> tuple[str, str, str]:
    lower = path.lower()
    if any(term in lower for term in REBUILDABLE_TERMS):
        return (
            "disposable_rebuildable_candidate",
            "delete_or_archive_after_user_approval",
            "Temporary/debug/smoke/preliminary output; keep until user approves cleanup.",
        )
    if any(term in lower for term in ARCHIVE_TERMS):
        return (
            "archive_or_compress_candidate",
            "compress_or_move_to_archive_after_user_approval",
            "Raw/shard/cache-compatible source material should be archived or compressed, not silently deleted.",
        )
    if lower.startswith("backtest_cache") or any(term in lower for term in PROTECTED_TERMS):
        return (
            "keep_required_for_backtest",
            "keep",
            "Protected backtest/PIT/price/formal replay data.",
        )
    if any(term in lower for term in DIAGNOSTIC_TERMS):
        return (
            "diagnostic_keep_optional",
            "keep_optional",
            "Diagnostic/readiness/evidence output useful for audit but not active formal model input.",
        )
    if stats.get("git_tracked_file_count", 0):
        return (
            "diagnostic_keep_optional",
            "keep_optional",
            "Tracked output or docs; review before any cleanup.",
        )
    return (
        "disposable_rebuildable_candidate",
        "delete_or_archive_after_user_approval",
        "Untracked generated data; likely rebuildable but requires approval before cleanup.",
    )


def _data_layer(path: str) -> str:
    lower = path.lower()
    if "raw" in lower or "shard" in lower or "cache_compatible" in lower:
        return "raw_source_archive_or_large_shard"
    if "backtest_cache" in lower or "price" in lower:
        return "price_cache_or_market_data"
    if "pit" in lower or "0050" in lower or "tw50" in lower:
        return "pit_or_constituent_data"
    if "formal" in lower or "ledger" in lower or "replay" in lower:
        return "formal_replay_or_ledger"
    if "diagnostic" in lower or "readiness" in lower or "evidence" in lower:
        return "evidence_readiness_diagnostic"
    if "outputs" in lower:
        return "derived_output"
    return "workspace_generated"


def _manifest(
    output: Path,
    inventory: pd.DataFrame,
    keep_required: pd.DataFrame,
    diagnostic_keep: pd.DataFrame,
    archive: pd.DataFrame,
    disposable: pd.DataFrame,
    approval: pd.DataFrame,
) -> dict[str, Any]:
    total = int(inventory["size_bytes"].sum()) if not inventory.empty else 0
    archive_bytes = int(archive["size_bytes"].sum()) if not archive.empty else 0
    disposable_bytes = int(disposable["size_bytes"].sum()) if not disposable.empty else 0
    return {
        "schema_version": 1,
        "task_id": TASK_ID,
        "status": "completed_storage_audit_only_no_delete",
        "generated_at": pd.Timestamp.now(tz="Asia/Taipei").isoformat(),
        "output_dir": str(output),
        "scanned_roots": SCAN_ROOTS,
        "inventory_rows": int(len(inventory)),
        "total_scanned_size_bytes": total,
        "total_scanned_size_mb": round(total / 1024 / 1024, 3),
        "keep_required_size_mb": _sum_mb(keep_required),
        "diagnostic_keep_optional_size_mb": _sum_mb(diagnostic_keep),
        "archive_or_compress_candidate_size_mb": _sum_mb(archive),
        "disposable_rebuildable_candidate_size_mb": _sum_mb(disposable),
        "estimated_reclaimable_if_archived_or_deleted_mb": round((archive_bytes + disposable_bytes) / 1024 / 1024, 3),
        "requires_user_approval_rows": int(len(approval)),
        "delete_executed": False,
        "archive_executed": False,
        "formal_model_changed": False,
        "trade_decision_changed": False,
        "active_in_trade_decision": False,
        "protected_boundaries": [
            "2014/11+ backtest-required data",
            "0050/TW50 PIT and 00631L related data",
            "TWSE/TPEx price and liquidity cache",
            "formal next-day ledgers and current formal outputs",
        ],
    }


def _sum_mb(frame: pd.DataFrame) -> float:
    if frame.empty:
        return 0.0
    return round(float(frame["size_bytes"].sum()) / 1024 / 1024, 3)


def _governance_plan(manifest: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# BACKTEST_LAB local data governance plan",
            "",
            "## Current action",
            "",
            "- Audit only. No files were deleted, moved, or compressed.",
            "- All cleanup candidates require explicit user approval before action.",
            "",
            "## Proposed layers",
            "",
            "1. Raw source archive: keep provenance, but move large raw/shard folders to compressed archive after validation.",
            "2. Normalized data tables: keep compact source-backed tables needed for replay and future model search.",
            "3. Evidence/readiness ledger: keep optional diagnostic evidence for decisions; do not treat as formal model input.",
            "4. Derived/replay outputs: keep formal-ready ledgers and latest validation outputs; archive old superseded runs.",
            "5. Disposable intermediates: delete or archive smoke/debug/tmp/preliminary outputs only after approval.",
            "",
            "## Estimated scope",
            "",
            f"- Total scanned: {manifest['total_scanned_size_mb']} MB.",
            f"- Required keep: {manifest['keep_required_size_mb']} MB.",
            f"- Optional diagnostic keep: {manifest['diagnostic_keep_optional_size_mb']} MB.",
            f"- Archive/compress candidates: {manifest['archive_or_compress_candidate_size_mb']} MB.",
            f"- Disposable/rebuildable candidates: {manifest['disposable_rebuildable_candidate_size_mb']} MB.",
            f"- Potential size addressable after approval: {manifest['estimated_reclaimable_if_archived_or_deleted_mb']} MB.",
        ]
    )


def _risk_notes() -> str:
    return "\n".join(
        [
            "# Risk and dependency notes",
            "",
            "- Do not delete 2014/11+ PIT, price, liquidity, formal target stream, or next-day ledger data.",
            "- Do not delete raw/source-backed data until a normalized table and manifest have been validated.",
            "- Do not delete diagnostic evidence that is still referenced by Research/Experiments handoffs.",
            "- Prefer compression or external archive for large raw/shard packages before deletion.",
            "- Git-tracked files should normally stay in repo or be removed through a reviewed commit, not local cleanup.",
            "- This audit intentionally does not execute destructive commands.",
        ]
    )


def _summary_zh(manifest: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# BACKTEST_LAB 本機資料治理與瘦身盤點",
            "",
            f"- 狀態：{manifest['status']}",
            f"- 掃描總量：約 {manifest['total_scanned_size_mb']} MB",
            f"- 必留正式回測資料：約 {manifest['keep_required_size_mb']} MB",
            f"- diagnostic / evidence 可選留：約 {manifest['diagnostic_keep_optional_size_mb']} MB",
            f"- 可壓縮/封存候選：約 {manifest['archive_or_compress_candidate_size_mb']} MB",
            f"- 可重建/暫存候選：約 {manifest['disposable_rebuildable_candidate_size_mb']} MB",
            f"- 估計需批准後可處理：約 {manifest['estimated_reclaimable_if_archived_or_deleted_mb']} MB",
            "- 本次沒有刪除、移動或壓縮任何資料。",
            "- 0050/TW50 PIT、00631L、價格/流動性、formal next-day ledger、current formal output 都列為保護邊界。",
        ]
    )


def _inventory_columns() -> list[str]:
    return [
        "path",
        "data_layer",
        "governance_bucket",
        "recommended_action",
        "reason",
        "size_bytes",
        "size_mb",
        "file_count",
        "git_tracked_file_count",
        "untracked_file_count_estimate",
        "newest_mtime",
        "largest_file",
        "largest_file_mb",
        "requires_user_approval_before_action",
        "delete_executed",
        "formal_model_changed",
        "trade_decision_changed",
        "active_in_trade_decision",
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit BACKTEST_LAB local data storage without deleting files.")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    run_data_governance_storage_audit(repo_root=args.repo_root, output_dir=args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
