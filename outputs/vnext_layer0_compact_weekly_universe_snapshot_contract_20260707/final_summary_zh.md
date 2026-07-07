# Layer0 compact weekly universe snapshot contract

## Verdict
- status=layer0_compact_weekly_universe_snapshot_materialized_ready_for_layer1_compact_rebuild
- primary_variant=top250_core_conditional_buffer50
- p2_active_avg_weekly_count=289.71666666666664
- p2_active_unique_ticker_count=1181
- p2_active_avg_turnover_share_5d=0.8942194742569847
- p2_watchlist_avg_weekly_count=10.283333333333333
- p2_watchlist_unique_ticker_count=868
- watchlist_reference_excluded_from_layer1_source_scope=true
- ready_for_layer1_compact_reduced_universe_interim_contract_rebuild=true
- ready_for_experiments=false
- ready_for_formal=false

## Plain Summary
The compact Layer0 snapshot materializes Strategy Center's accepted top250_core_conditional_buffer50 policy. Active Layer1 source scope is separated from watchlist_reference, so one-week 5D bursts do not trigger high-cost Layer1 source work. Watchlist reference unique count is high because it is mostly 1-4 week bursts and must not be used as Layer1 source scope. This is still diagnostic/source readiness only and does not change formal model or trade decisions.

## Flags
- formal_model_changed=false
- trade_decision_changed=false
- active_in_trade_decision=false
- report_changed=false
- portfolio_replay_executed=false
- ready_for_strategy_replay=false
- not_live_rule=true
- forward_returns_live_rule_usage=false
