# TW50 / 0050 Point-in-Time Constituents Data Contract

Purpose: support `tw50_dynamic_constituents_v0` historical replay without using
current constituents to backfill earlier dates.

## Why This Exists

The three-pool model needs pool 2 to represent the large-cap breadth view. If
pool 2 has no point-in-time constituents for a historical date, the replay must
skip pool 2 instead of pretending the current 0050 list existed in the past.

This protects the model from future-data leakage.

## Required CSV

Default path:

```text
data/tw50_constituents.csv
```

Required columns:

```text
effective_date,ticker
```

Recommended columns:

```text
effective_date,end_date,ticker,name,source,source_updated_at
```

Column meanings:

- `effective_date`: first date the ticker is valid as a Taiwan 50 / 0050
  constituent.
- `end_date`: optional last valid date. Empty means still active until the next
  point-in-time update or until explicitly ended.
- `ticker`: Taiwan ticker with suffix, for example `2330.TW`.
- `name`: display name.
- `source`: source label, for example `official_snapshot`, `manual_verified`,
  or `seed_snapshot`.
- `source_updated_at`: date this row was added or verified.

## Validation Rule

For a signal date to be ready:

1. At least one constituent snapshot must be active on that date.
2. The active constituent count should be at least 45. This allows normal Taiwan
   50 edge cases such as 49 names, but rejects incomplete lists.
3. The source must be point-in-time. A current snapshot may be used for current
   reports but not for historical replay before its `effective_date`.

## Current Limitation

As of 2026-06-20, `data/tw50_constituents.csv` contains one seed snapshot with
`effective_date=2025-06-23`. Therefore:

- 2022 coverage is 0%.
- 2023 coverage is 0%.
- 2024 to 2026 coverage starts only on 2025-06-23.

This means full three-pool historical diagnostics are currently blocked for
2022 through 2025-06-20.

## Validator

Run:

```powershell
$env:PYTHONPATH='src'
python -m backtest_lab.tw50_constituent_coverage --constituent-path data/tw50_constituents.csv --output-dir outputs/tw50_constituent_coverage_YYYYMMDD_full
```

Outputs:

- `tw50_constituent_coverage_summary.csv`
- `tw50_constituent_gap_dates.csv`
- `tw50_constituent_coverage.md`
- `metadata.json`
- `run_log.csv`
- `current_step.txt`
- `completed.txt`

Readiness statuses:

- `ready`: every requested period has at least 95% date coverage.
- `partial_blocked`: at least one period has data, but coverage is not enough
  for full historical replay.
- `blocked_no_historical_coverage`: no requested period has usable coverage.
- `blocked_no_source`: source file is missing or invalid.

## Acceptance Criteria for Full Three-Pool Diagnostics

Before using pool 2 in full historical diagnostics:

- 2022 coverage ratio >= 0.95.
- 2023 coverage ratio >= 0.95.
- 2024-2026 coverage ratio >= 0.95 for the chosen study period.
- Each ready signal date has at least 45 active tickers.
- The validator output is archived with the replay output.

Until this passes, pool 3 role diagnostics must be labeled as partial and
cannot be used as a final decision to remove or downgrade pool 3.
