# Core Backlog

## TASK-BACKTEST-CORE-LOCAL-DATASTORE-CONSOLIDATION-AFTER-DYNAMIC-POOL1-PIT-READY-001

Status: backlog / do not start until Dynamic Pool1 PIT readiness data work is sufficiently complete.

Goal: after 2014-11+ Taiwan market data is collected and validated for long-range backtests and dynamic Pool1 candidate-universe research, perform a local datastore consolidation pass to reduce duplicate or unnecessary local storage while preserving all replay-critical data.

Scope:
- Inventory Radar/Data and Backtest Lab local outputs, cache, raw archives, shards, normalized tables, and readiness/evidence ledgers.
- Classify files into raw source archive, normalized replay tables, readiness/evidence ledger, reproducible temporary output, duplicate cache, and safe-to-delete candidates.
- Preserve backtest-critical data for 2014-11 onward, including 0050/TW50 PIT candidates, all-market liquidity, monthly revenue, quarterly fundamentals, market-cap/capital-stock proxy, sector membership, and price cache.
- Produce a cleanup plan and manifest before deleting or moving anything.

Boundaries:
- Do not delete unvalidated, unbacked-up, or non-reproducible data.
- Do not start cleanup while Dynamic Pool1 readiness / TWSE sector diagnostic panel work is still in progress.
- Do not compress, shard, move, or delete data without a manifest and reviewable plan first.
- This is storage hygiene and data-contract consolidation only; it must not change formal model, selector, report, or trade decision.

Next trigger: start only after Dynamic Pool1 PIT readiness reaches an accepted/partial-ready handoff point and the remaining Radar/Data full sweeps are no longer actively writing local shards.
