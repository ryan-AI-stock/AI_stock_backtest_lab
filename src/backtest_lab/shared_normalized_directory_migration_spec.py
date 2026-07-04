from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


TASK_ID = "TASK-BACKTEST-DATA-SHARED-NORMALIZED-DIRECTORY-MIGRATION-SPEC-20260704"
DEFAULT_OUTPUT_DIR = "outputs/shared_normalized_directory_migration_spec_20260704"
CANONICAL_ROOT = "C:/Users/zergv/Documents/Codex/shared_data/ai_stock/normalized"


def run_shared_normalized_directory_migration_spec(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    phases = _migration_phases()
    datasets = _dataset_candidates()
    manifests = _manifest_schema()
    references = _repo_reference_contract()
    approval = _approval_gates()
    rollback = _rollback_plan()
    manifest = _manifest(output, phases, datasets)

    phases.to_csv(output / "migration_phases.csv", index=False, encoding="utf-8-sig")
    datasets.to_csv(output / "dataset_migration_candidates.csv", index=False, encoding="utf-8-sig")
    manifests.to_csv(output / "shared_manifest_schema.csv", index=False, encoding="utf-8-sig")
    references.to_csv(output / "repo_reference_contract.csv", index=False, encoding="utf-8-sig")
    approval.to_csv(output / "user_approval_gates.csv", index=False, encoding="utf-8-sig")
    rollback.to_csv(output / "rollback_restore_plan.csv", index=False, encoding="utf-8-sig")
    (output / "shared_normalized_directory_migration_spec.md").write_text(_spec_md(), encoding="utf-8")
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "final_summary_zh.md").write_text(_summary_zh(manifest), encoding="utf-8")
    pd.DataFrame([{"step": TASK_ID, "status": "completed_manifest_only_spec", "output_dir": str(output)}]).to_csv(
        output / "completed.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(columns=["step", "status", "reason"]).to_csv(output / "failed.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{"step": "write_spec", "status": "completed"}]).to_csv(output / "run_log.csv", index=False, encoding="utf-8-sig")
    return output


def _migration_phases() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "phase": "phase0_manifest_only",
                "action": "write shared manifest references beside existing repo outputs",
                "destructive": False,
                "approval_required": False,
                "exit_criteria": "both repos can read manifest without changing existing runners",
            },
            {
                "phase": "phase1_copy_dry_run",
                "action": "copy selected normalized shards to shared root dry-run location",
                "destructive": False,
                "approval_required": True,
                "exit_criteria": "size/checksum/row counts match source",
            },
            {
                "phase": "phase2_checksum_validation",
                "action": "validate sha256, row counts, coverage, and future-data audit parity",
                "destructive": False,
                "approval_required": True,
                "exit_criteria": "all validation ledgers pass",
            },
            {
                "phase": "phase3_user_approved_move_or_archive",
                "action": "optionally replace duplicate repo-local normalized data with pointers or archive",
                "destructive": True,
                "approval_required": True,
                "exit_criteria": "manual approval and rollback package exist",
            },
        ]
    )


def _dataset_candidates() -> pd.DataFrame:
    rows = [
        ("tw50_0050_pit_monthly_anchor", "first_wave", "source-backed candidate, compact normalized table", "move_after_copy_validation"),
        ("all_listed_liquid_universe_pit_daily", "first_wave", "large normalized daily shards, high duplication risk", "copy_dry_run_first"),
        ("mops_monthly_revenue_pit", "first_wave", "full-universe PIT candidate with conservative available_date", "copy_dry_run_first"),
        ("quarterly_fundamentals_t163sb04_pit", "first_wave", "full sweep source candidate, formal_exact=false", "copy_dry_run_first"),
        ("tpex_market_cap_daily_candidate", "second_wave", "TPEx full source candidate, TWSE still proxy", "wait_for_market_cap_policy"),
        ("twse_capital_stock_proxy_market_cap", "second_wave", "proxy contract, not direct daily market cap", "diagnostic_only"),
        ("twse_sector_monthly_anchor", "second_wave", "TWSE-only monthly anchor diagnostic proxy", "diagnostic_only"),
        ("taxonomy_evidence_panel", "do_not_migrate_first", "small diagnostic evidence, not performance signal", "keep_in_repo_outputs"),
        ("formal_next_day_ledgers", "do_not_move_without_runner_update", "formal replay evidence must remain stable", "manifest_only_reference"),
        ("raw_pdf_html_json_archives", "archive_only", "raw source artifacts require checksum archive plan", "do_not_make_shared_normalized"),
    ]
    return pd.DataFrame(rows, columns=["dataset_id", "migration_priority", "reason", "recommended_action"])


def _manifest_schema() -> pd.DataFrame:
    fields = [
        ("dataset_id", "string", "required"),
        ("version", "string", "required"),
        ("canonical_path", "path", "required"),
        ("source_repo", "string", "required"),
        ("source_output_path", "path", "required"),
        ("source_commit", "string", "required_if_known"),
        ("schema_hash", "sha256", "required"),
        ("file_sha256", "sha256", "required_per_shard"),
        ("row_count", "integer", "required"),
        ("coverage_start", "date", "required_if_temporal"),
        ("coverage_end", "date", "required_if_temporal"),
        ("formal_exact", "boolean", "required"),
        ("diagnostic_only", "boolean", "required"),
        ("future_data_violation_count", "integer", "required"),
        ("restore_source", "string", "required"),
        ("migration_phase", "string", "required"),
    ]
    return pd.DataFrame(fields, columns=["field", "type", "requirement"])


def _repo_reference_contract() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "repo": "AI_stock_backtest_lab",
                "reference_method": "config path or manifest resolver",
                "compatibility_rule": "existing runners keep local defaults; shared path is opt-in until validation",
            },
            {
                "repo": "AI_stock_rotation_radar",
                "reference_method": "write normalized shards to shared root plus source manifest",
                "compatibility_rule": "Radar outputs retain source manifests and do not delete local package until approval",
            },
        ]
    )


def _approval_gates() -> pd.DataFrame:
    gates = [
        ("copy_dry_run", "user approves non-destructive copy size and target root"),
        ("checksum_validation", "user approves validation report before runner references shared path"),
        ("move_or_archive", "user explicitly approves move/archive/delete of repo-local duplicates"),
        ("rollback", "restore manifest and source package available before any destructive action"),
    ]
    return pd.DataFrame(gates, columns=["gate", "approval_requirement"])


def _rollback_plan() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "scenario": "shared path unavailable",
                "restore_action": "runner falls back to repo-local default paths",
                "requires_destructive_rollback": False,
            },
            {
                "scenario": "checksum mismatch after copy",
                "restore_action": "discard copied shared candidate; keep source repo package untouched",
                "requires_destructive_rollback": False,
            },
            {
                "scenario": "post-approval archive restore",
                "restore_action": "restore from archive path listed in shared manifest and rerun checksum validation",
                "requires_destructive_rollback": False,
            },
        ]
    )


def _manifest(output: Path, phases: pd.DataFrame, datasets: pd.DataFrame) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "task_id": TASK_ID,
        "status": "completed_manifest_only_migration_spec",
        "generated_at": pd.Timestamp.now(tz="Asia/Taipei").isoformat(),
        "output_dir": str(output),
        "canonical_shared_data_root": CANONICAL_ROOT,
        "phase_count": int(len(phases)),
        "dataset_candidate_count": int(len(datasets)),
        "delete_executed": False,
        "move_executed": False,
        "compress_executed": False,
        "archive_executed": False,
        "formal_model_changed": False,
        "trade_decision_changed": False,
        "active_in_trade_decision": False,
        "requires_radar_data_next_step": True,
    }


def _spec_md() -> str:
    return f"""# Shared normalized data directory migration spec

Canonical root proposal:

`{CANONICAL_ROOT}`

This task is manifest-only. It does not move, delete, compress, or archive files.

## Shard naming

`{{dataset_id}}/{{version}}/{{market_or_scope}}/{{frequency}}/{{year_or_period}}/{{dataset_id}}_{{period}}_{{shard_seq}}.csv`

Examples:

- `mops_monthly_revenue_pit/v0/full_universe/monthly/2024/mops_monthly_revenue_pit_2024_000.csv`
- `all_listed_liquid_universe_pit_daily/v0/twse_tpex/daily/2024/liquidity_pit_daily_2024_001.csv`

## Migration phases

1. Manifest only.
2. Copy dry-run.
3. Checksum validation.
4. User-approved move/archive.

## Boundaries

- No formal model change.
- No trade decision change.
- No report change.
- No destructive action without explicit user approval.
"""


def _summary_zh(manifest: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Shared normalized data directory migration spec",
            "",
            f"- 狀態：{manifest['status']}",
            f"- canonical root 建議：`{manifest['canonical_shared_data_root']}`",
            "- 本任務只做 spec，不搬檔、不刪檔、不壓縮。",
            "- 第一批適合：0050 PIT compact table、全市場 liquidity normalized shards、月營收、季財報。",
            "- 不適合第一批：formal ledgers、raw PDF/HTML/JSON archives、taxonomy 小型 evidence。",
            "- 需要 Radar/Data 下一棒配合 copy dry-run / checksum validation。",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write shared normalized data directory migration spec.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    run_shared_normalized_directory_migration_spec(output_dir=args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
