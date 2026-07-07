# Layer0 weekly universe snapshot contract

## Verdict
- status=layer0_weekly_universe_snapshot_materialized_diagnostic_ready_not_experiments
- weekly_snapshot_count=592
- snapshot_rows=997574
- recommended_primary_variant=top300_buffer100
- top300_buffer100_average_ticker_count=400.0
- top300_buffer100_average_turnover_share_5d=0.9339828252604127
- ready_for_t164_mass_download=false
- ready_for_experiments=false
- ready_for_formal=false

## Core decision
Layer0 weekly snapshots are materialized from PIT daily traded value. Use top300+buffer100 as the primary reduced universe candidate for Layer1 source planning. This is not a trading rule and does not authorize t164 mass download.

## Storage note
Full `layer0_weekly_universe_snapshot.csv` is retained in the local output path but intentionally ignored by git because it is 288MB. The repo tracks `layer0_weekly_universe_snapshot_sample.csv`, coverage, summary, readiness, and manifest.

## Flags
- formal_model_changed=false
- trade_decision_changed=false
- active_in_trade_decision=false
- report_changed=false
- portfolio_replay_executed=false
- ready_for_strategy_replay=false
- not_live_rule=true
- forward_returns_live_rule_usage=false
