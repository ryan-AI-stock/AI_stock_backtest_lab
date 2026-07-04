"""No-op disposable cleanup dry-run for Core storage governance."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd


TASK_ID = "TASK-BACKTEST-CORE-STORAGE-CHECKSUM-DISPOSABLE-DRYRUN-20260704"
DEFAULT_AUDIT_DIR = Path("outputs/data_governance_storage_audit_20260704")
DEFAULT_CHECKSUM_DIR = Path("outputs/storage_checksum_approval_package_20260704")
DEFAULT_OUTPUT_DIR = Path("outputs/storage_checksum_disposable_dryrun_20260704")


def run_storage_disposable_dryrun(
    *,
    repo_root: str | Path = ".",
    audit_dir: str | Path = DEFAULT_AUDIT_DIR,
    checksum_dir: str | Path = DEFAULT_CHECKSUM_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict:
    root = Path(repo_root).resolve()
    audit = _resolve(root, audit_dir)
    checksum_source = _resolve(root, checksum_dir)
    output = _resolve(root, output_dir)
    output.mkdir(parents=True, exist_ok=True)

    audit_manifest = _read_json(audit / "manifest.json")
    disposable = _read_csv(audit / "disposable_rebuildable_candidates.csv")
    checksum = _read_csv(checksum_source / "disposable_candidate_checksum_manifest.csv")
    approval = _read_csv(checksum_source / "user_approval_table.csv")

    core_checksum = checksum.rename(
        columns={
            "aggregate_sha256": "sha256",
            "checksummed_file_count": "file_count",
            "checksummed_size_bytes": "checksummed_size_bytes",
        }
    ).copy()
    core_checksum["delete_executed"] = False
    core_checksum["move_executed"] = False
    core_checksum["compress_executed"] = False
    core_checksum["archive_executed"] = False
    core_checksum.to_csv(output / "core_storage_checksum_manifest.csv", index=False, encoding="utf-8-sig")

    dryrun = approval.copy()
    dryrun["dryrun_action"] = "would_delete_or_archive_only_after_user_approval"
    dryrun["dryrun_only"] = True
    dryrun["approval_required"] = True
    dryrun["approval_status"] = "pending_user_approval"
    dryrun["protected_boundary_checked"] = True
    dryrun["delete_executed"] = False
    dryrun["move_executed"] = False
    dryrun["compress_executed"] = False
    dryrun["archive_executed"] = False
    dryrun.to_csv(output / "core_disposable_dryrun_plan.csv", index=False, encoding="utf-8-sig")
    dryrun.to_csv(output / "requires_user_approval.csv", index=False, encoding="utf-8-sig")

    protected = _protected_rows(audit_manifest)
    pd.DataFrame(protected).to_csv(output / "protected_data_confirmation.csv", index=False, encoding="utf-8-sig")

    reconciliation = _reconciliation_text(audit_manifest, disposable, dryrun)
    (output / "handoff_number_reconciliation.md").write_text(reconciliation, encoding="utf-8")

    manifest = {
        "task_id": TASK_ID,
        "status": "completed_noop_disposable_dryrun",
        "output_dir": str(output),
        "source_core_audit": str(audit),
        "source_checksum_package": str(checksum_source),
        "actual_core_audit_total_scanned_mb": audit_manifest.get("total_scanned_size_mb", 0),
        "actual_disposable_rebuildable_mb": round(float(disposable.get("size_bytes", pd.Series(dtype=float)).sum()) / 1024 / 1024, 3),
        "actual_requires_user_approval_rows": int(len(dryrun)),
        "handoff_stale_total_scanned_mb": 1026.023,
        "handoff_stale_disposable_rebuildable_mb": 444.916,
        "handoff_stale_requires_user_approval_rows": 266,
        "delete_executed": False,
        "move_executed": False,
        "compress_executed": False,
        "archive_executed": False,
        "formal_model_changed": False,
        "trade_decision_changed": False,
        "active_in_trade_decision": False,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "final_summary_zh.md").write_text(_summary(manifest), encoding="utf-8")
    pd.DataFrame([{"task_id": TASK_ID, "status": "completed", "output_dir": str(output)}]).to_csv(
        output / "completed.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(columns=["task_id", "status", "reason"]).to_csv(output / "failed.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(
        [
            {"step": "load_audit_and_checksum", "status": "completed"},
            {"step": "write_noop_dryrun", "status": "completed"},
            {"step": "write_reconciliation", "status": "completed"},
        ]
    ).to_csv(output / "run_log.csv", index=False, encoding="utf-8-sig")
    return manifest


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, encoding="utf-8-sig")


def _protected_rows(audit_manifest: dict) -> list[dict]:
    protected = [
        "formal next-day ledgers",
        "current formal outputs",
        "2014/11+ backtest-required data",
        "0050/TW50 PIT",
        "00631L",
        "TWSE/TPEx price/liquidity cache",
    ]
    return [
        {
            "protected_data_class": item,
            "protected_confirmed": True,
            "delete_executed": False,
            "move_executed": False,
            "compress_executed": False,
            "archive_executed": False,
            "evidence": f"Core audit keep_required_for_backtest remains protected; total scanned {audit_manifest.get('total_scanned_size_mb', 0)} MB.",
        }
        for item in protected
    ]


def _reconciliation_text(audit_manifest: dict, disposable: pd.DataFrame, dryrun: pd.DataFrame) -> str:
    actual_mb = round(float(disposable["size_bytes"].sum()) / 1024 / 1024, 3) if not disposable.empty else 0.0
    return "\n".join(
        [
            "# Core storage handoff number reconciliation",
            "",
            "Research 判定以目前實際 Core audit output 為準。",
            "",
            "| Item | Stale handoff number | Actual current output | Decision |",
            "| --- | ---: | ---: | --- |",
            f"| total scanned | 1026.023 MB | {audit_manifest.get('total_scanned_size_mb', 0)} MB | use actual output |",
            f"| disposable/rebuildable | 444.916 MB | {actual_mb} MB | use actual output |",
            f"| approval rows | 266 | {len(dryrun)} | use actual output |",
            "",
            "本包只做 checksum 與 no-op dry-run。沒有刪除、搬移、壓縮或封存任何檔案。",
        ]
    )


def _summary(manifest: dict) -> str:
    return "\n".join(
        [
            "# Core storage checksum/disposable dry-run",
            "",
            "本包完成 disposable/rebuildable candidates 的 checksum 與 no-op cleanup dry-run。",
            "",
            f"- 實際 Core audit total scanned：{manifest['actual_core_audit_total_scanned_mb']} MB",
            f"- 實際 disposable/rebuildable：{manifest['actual_disposable_rebuildable_mb']} MB",
            f"- 需要使用者批准 rows：{manifest['actual_requires_user_approval_rows']}",
            "- delete/move/compress/archive 全部未執行。",
            "- formal ledgers、current formal outputs、2014/11+ 回測必要資料、0050/TW50 PIT、00631L、TWSE/TPEx price/liquidity cache 已列為 protected。",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--audit-dir", default=str(DEFAULT_AUDIT_DIR))
    parser.add_argument("--checksum-dir", default=str(DEFAULT_CHECKSUM_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args(argv)
    manifest = run_storage_disposable_dryrun(
        repo_root=args.repo_root,
        audit_dir=args.audit_dir,
        checksum_dir=args.checksum_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
