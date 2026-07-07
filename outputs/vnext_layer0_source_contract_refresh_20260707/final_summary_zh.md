# Layer0 source contract refresh

## Verdict
- status=layer0_source_contract_refreshed_traded_value_primary_ready_marketcap_event_instrument_partial
- primary_pruning_source=daily_per_stock_traded_value
- recommended_universe=top_200_to_300_by_recent_traded_value_plus_100_buffer_watchlist
- ready_for_layer0_materialized_weekly_universe_snapshot_contract=true
- ready_for_t164_mass_download=false
- ready_for_experiments=false
- ready_for_formal=false

## Core decision
Use traded_value as the primary Layer0 pruning source. Market-cap rank, full instrument master, and PIT disposition/full-delivery ledgers stay proxy/blocked. Do not resume large t164 downloads until a reduced Layer0 universe is materialized and accepted.

## Flags
- formal_model_changed=false
- trade_decision_changed=false
- active_in_trade_decision=false
- report_changed=false
- portfolio_replay_executed=false
- ready_for_strategy_replay=false
- not_live_rule=true
- forward_returns_live_rule_usage=false
