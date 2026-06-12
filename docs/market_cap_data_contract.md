# Market Cap Data Contract

This project uses market capitalization only for stock size classification.
Liquidity remains a separate trading feasibility signal.

## Accepted Source Files

`stock_pool_observation` loads the first available source in this order:

1. CLI/env explicit path: `--market-cap-data` or `MARKET_CAP_DATA_PATH`
2. `RADAR_DATA_DIR/market_cap.latest.csv`
3. `RADAR_DATA_DIR/market_caps.latest.csv`
4. `RADAR_DATA_DIR/stock_metrics.refreshed.csv`
5. `RADAR_DATA_DIR/formal_radar_candidates.latest.csv`

## Required Columns

One ticker identifier:

- `ticker`, such as `2330.TW`
- or `symbol`, such as `2330`; optional `exchange` / `suffix` defaults to `TW`

One market cap value:

- `free_float_market_cap_twd` preferred when available
- `market_cap_twd` fallback

Optional date column:

- `date`
- `report_date`
- `source_date`

If a date column exists, rows after the signal date are ignored.
If no date column exists, the file is treated as the latest known snapshot and should only be used for current-day observation, not historical replay.

## Output Fields

Candidate outputs include:

- `size_profile`: `large_cap`, `mid_cap`, `small_cap`, `micro_cap`, or `unknown_size`
- `market_cap_twd`
- `size_basis`
- `liquidity_profile`

## Current Thresholds

The first engineering version uses configurable TWD thresholds:

- `large_cap`: at least 500,000,000,000
- `mid_cap`: at least 50,000,000,000
- `small_cap`: at least 5,000,000,000
- `micro_cap`: below 5,000,000,000

These thresholds are a practical first version, not an official TWSE/MSCI methodology.
Future versions may replace them with market-wide percentile ranking or free-float market-cap index methodology.
