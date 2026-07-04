# Shared normalized data directory migration spec

Task ID: `TASK-BACKTEST-DATA-SHARED-NORMALIZED-DIRECTORY-MIGRATION-SPEC-20260704`

Status: `completed_spec_only_no_move`

## Purpose

BACKTEST_LAB and AI_stock_rotation_radar now both hold large normalized/cache-compatible tables. Some tables are necessary for future 2014/11+ backtests, but storing them under each repo output package will keep increasing local disk use.

This spec defines a shared normalized data root and staged migration plan. This task does not move, delete, compress, or rewrite any data.

## Proposed canonical root

`C:\Users\zergv\Documents\Codex\shared_stock_data`

Subdirectories:

- `normalized\twse_tpex_liquidity_daily`
- `normalized\twse_tpex_price_daily`
- `normalized\0050_pit_pcf`
- `normalized\pool1b_price_repair`
- `normalized\mops_monthly_revenue`
- `normalized\mops_quarterly_fundamentals`
- `normalized\listing_status_metadata`
- `manifests`
- `checksums`
- `restore_maps`

## Shard naming

Recommended pattern:

`{dataset_id}__{market_or_scope}__{period_grain}_{period}__v{version}.csv`

Examples:

- `twse_tpex_liquidity_daily__twse__month_2024-01__v1.csv`
- `mops_monthly_revenue__all_listed__year_2024__v1.csv`
- `0050_pcf_daily__yuanta__month_2023-12__v1.csv`

Rules:

- Keep shard grain stable within each dataset.
- Never overwrite old versions; create `v2`, `v3` when schema or source route changes.
- Each shard must have row count, byte size, sha256, source package, and generated timestamp in a manifest.

## Manifest schema

Required fields:

- `dataset_id`
- `dataset_version`
- `shard_path`
- `relative_shard_path`
- `source_repo`
- `source_output_package`
- `source_file`
- `period_start`
- `period_end`
- `market`
- `row_count`
- `size_bytes`
- `sha256`
- `schema_hash`
- `source_type`
- `formal_exact`
- `future_data_violation_count`
- `generated_at`
- `migration_status`
- `restore_source`

## Repo reference policy

Each repo should keep only lightweight references:

- `data_sources/shared_data_manifest.json`
- output package local `readme.md`
- compatibility stub CSV only if old runners require a path

No repo should silently depend on absolute paths without manifest validation.

Before reading a shared shard, runner must validate:

1. manifest exists;
2. shard exists;
3. sha256 matches;
4. dataset version is accepted;
5. date range covers requested replay window.

## Migration phases

### Phase 0: Manifest only

- Build manifests for current large normalized/cache-compatible tables.
- Compute checksums.
- No copy, move, delete, or compress.

### Phase 1: Copy dry-run

- Estimate copy targets and disk impact.
- Produce commands but do not execute them.
- Verify destination path policy.

### Phase 2: User-approved copy

- Copy selected shards to shared root.
- Validate checksum after copy.
- Keep original files untouched.

### Phase 3: Runner compatibility test

- Update one runner at a time to optionally read shared root.
- Default fallback stays local until validated.
- Run smoke tests.

### Phase 4: User-approved archive/move

- Only after successful shared readback and checksum validation.
- Archive or move duplicated local normalized files only with explicit user approval.
- Preserve restore map.

## Rollback / restore plan

Every migrated shard must be restorable by:

1. locating `restore_source`;
2. validating original checksum if still present;
3. copying from shared root back to original relative path if needed;
4. rerunning the package runner if source package is rebuildable.

No cleanup can proceed unless restore steps are written for that package.

## First batch candidates

Good candidates for migration spec phase, not immediate move:

- Radar `validated_daily_pcf_candidate.csv` and full-range 0050 PCF shards.
- Radar TWSE/TPEx liquidity daily monthly shards.
- Radar TPEx market cap full sweep shards.
- Radar MOPS monthly revenue annual shards.
- Radar MOPS quarterly fundamentals annual shards.
- Pool1B price repair cache-compatible outputs.

## Do not move in first batch

- Formal next-day ledgers used by current reports.
- Current formal report outputs.
- Files without checksum manifest.
- Files that are only present as source evidence and not normalized tables.
- raw PDF/HTML/JSON evidence before archive policy is approved.

## User approval gates

User approval is required before:

- deleting any local file;
- moving original files out of repo outputs/cache paths;
- compressing raw source evidence;
- replacing runner defaults with shared root only;
- archiving local duplicates after copy.

## Boundary

- `delete_executed=false`
- `move_executed=false`
- `compress_executed=false`
- `formal_model_changed=false`
- `trade_decision_changed=false`
- `active_in_trade_decision=false`

