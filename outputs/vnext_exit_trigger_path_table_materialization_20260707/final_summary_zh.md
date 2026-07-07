# vNext Exit-trigger Path Table Materialization

Status: ready_for_experiments_tightened_exit_trigger_attribution

Boundary: diagnostic-only path table for Experiments tightening rerun; no replay, no live rule, no formal/report/trade decision change.

Readiness:
- ready_for_exit_trigger_tightening_diagnostic=true
- ready_for_experiments=true
- ready_for_portfolio_like_diagnostic=false
- ready_for_strategy_replay=false
- ready_for_formal=false
- total_event_count=509
- included_event_count=433
- blocked_event_count=76
- path_rows=23382
- expected_path_rows=23382
- future_data_violation_count=0

Blocked / proxy notes:
- stock_bias_cross_section_percentile remains blocked; Core provides PIT self-history BIAS z-score proxy instead.
- endpoint 30D labels are evaluation metadata only and not rule inputs.

Feature missingness highlights:
- RS5: missing_share=0.0000; source_quality=pit_computed_from_daily_price_or_benchmark
- RS10: missing_share=0.0000; source_quality=pit_computed_from_daily_price_or_benchmark
- RS20: missing_share=0.0000; source_quality=pit_computed_from_daily_price_or_benchmark
- RS60: missing_share=0.0060; source_quality=pit_computed_from_daily_price_or_benchmark
- BIAS20_z: missing_share=0.0057; source_quality=pit_self_history_zscore_proxy_cross_section_percentile_blocked
- BIAS60_z: missing_share=0.0195; source_quality=pit_self_history_zscore_proxy_cross_section_percentile_blocked
- turnover_spike_ratio: missing_share=0.0000; source_quality=pit_computed_from_daily_price_or_benchmark
- large_down_day_count_20d: missing_share=0.0000; source_quality=pit_computed_from_daily_price_or_benchmark
- volatility_20d: missing_share=0.0000; source_quality=pit_computed_from_daily_price_or_benchmark
- 0050_BIAS60: missing_share=0.0000; source_quality=pit_computed_from_daily_price_or_benchmark
- observed_excess_vs_00631L_so_far: missing_share=0.0000; source_quality=pit_computed_from_daily_price_or_benchmark
- endpoint_30d_excess_vs_00631L: missing_share=0.0000; source_quality=evaluation_metadata_only_not_rule_input
- stock_bias_cross_section_percentile: missing_share=1.0000; source_quality=blocked_not_materialized_daily_cross_section

Flags:
- formal_model_changed=false
- trade_decision_changed=false
- active_in_trade_decision=false
- report_changed=false
- portfolio_replay_executed=false
- ready_for_strategy_replay=false
- not_live_rule=true
- forward_returns_live_rule_usage=false