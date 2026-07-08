# full-period regime switch benchmark + exception path materialization

- task_id: `TASK-BACKTEST-CORE-VNEXT-FULL-PERIOD-REGIME-SWITCH-BENCHMARK-EXCEPTION-PATH-MATERIALIZATION-001`
- status: `full_period_regime_switch_paths_materialized_partial_stock_path_caveats`
- ready_for_full_period_regime_switch_diagnostic: `true`
- benchmark/reference rows: 5,336
- stock/route path rows: 1,373
- stock_route_unadjusted_path_ready_share: 0.9942
- blocked_rows: 1198
- boundary: diagnostic-only；不改 formal model / daily report / trade decision；不做 portfolio replay / strategy replay。

## 主要結論

已把 P1 ordinary defensive、P2/2024+ mega route、00631L/0050/cash reference 放到同一個 timing/cost basis package。00631L buy-hold 只保留為分離 reference，不得與 signal-aligned policy path 混用。

P1 ordinary branch 可保留 `00631L base + ultra-strict stock exception` trace；P2/2024+ 使用已 materialized 的 selected ticker official unadjusted OHLC route path。2023 P2 stock-route exact path仍需看 `path_source_status`，不可用 00631L+excess proxy 補。

## Reference Coverage

period_label,benchmark,timing_variant,count,sum
2024_latest,0050,buy_hold_reference_separate_do_not_mix_with_policy_path,1,1
2024_latest,00631L,buy_hold_reference_separate_do_not_mix_with_policy_path,1,1
2026YTD,0050,buy_hold_reference_separate_do_not_mix_with_policy_path,1,1
2026YTD,00631L,buy_hold_reference_separate_do_not_mix_with_policy_path,1,1
P1,0050,buy_hold_reference_separate_do_not_mix_with_policy_path,1,1
P1,0050,next_day_close_entry_fixed_5td_exit,411,411
P1,0050,next_day_open_entry_fixed_5td_exit,411,0
P1,0050,same_week_close_to_next_rebalance_close_comparator,411,411
P1,00631L,buy_hold_reference_separate_do_not_mix_with_policy_path,1,1
P1,00631L,next_day_close_entry_fixed_5td_exit,411,411
P1,00631L,next_day_open_entry_fixed_5td_exit,411,0
P1,00631L,same_week_close_to_next_rebalance_close_comparator,411,411
P1,cash,next_day_close_entry_fixed_5td_exit,411,411
P1,cash,next_day_open_entry_fixed_5td_exit,411,411
P1,cash,same_week_close_to_next_rebalance_close_comparator,411,411
P2,0050,buy_hold_reference_separate_do_not_mix_with_policy_path,1,1
P2,0050,next_day_close_entry_fixed_5td_exit,51,51
P2,0050,next_day_open_entry_fixed_5td_exit,51,0
P2,0050,same_week_close_to_next_rebalance_close_comparator,51,51
P2,00631L,buy_hold_reference_separate_do_not_mix_with_policy_path,1,1
P2,00631L,next_day_close_entry_fixed_5td_exit,51,51
P2,00631L,next_day_open_entry_fixed_5td_exit,51,0
P2,00631L,same_week_close_to_next_rebalance_close_comparator,51,51
P2,cash,next_day_close_entry_fixed_5td_exit,51,51
P2,cash,next_day_open_entry_fixed_5td_exit,51,51
P2,cash,same_week_close_to_next_rebalance_close_comparator,51,51
P2|2024_latest,0050,next_day_close_entry_fixed_5td_exit,103,103
P2|2024_latest,0050,next_day_open_entry_fixed_5td_exit,103,0
P2|2024_latest,0050,same_week_close_to_next_rebalance_close_comparator,103,103
P2|2024_latest,00631L,next_day_close_entry_fixed_5td_exit,103,103
P2|2024_latest,00631L,next_day_open_entry_fixed_5td_exit,103,0
P2|2024_latest,00631L,same_week_close_to_next_rebalance_close_comparator,103,103
P2|2024_latest,cash,next_day_close_entry_fixed_5td_exit,103,103
P2|2024_latest,cash,next_day_open_entry_fixed_5td_exit,103,103
P2|2024_latest,cash,same_week_close_to_next_rebalance_close_comparator,103,103
P2|2024_latest|2026YTD,0050,next_day_close_entry_fixed_5td_exit,26,24
P2|2024_latest|2026YTD,0050,next_day_open_entry_fixed_5td_exit,26,0
P2|2024_latest|2026YTD,0050,same_week_close_to_next_rebalance_close_comparator,26,25
P2|2024_latest|2026YTD,00631L,next_day_close_entry_fixed_5td_exit,26,24
P2|2024_latest|2026YTD,00631L,next_day_open_entry_fixed_5td_exit,26,0
P2|2024_latest|2026YTD,00631L,same_week_close_to_next_rebalance_close_comparator,26,25
P2|2024_latest|2026YTD,cash,next_day_close_entry_fixed_5td_exit,26,26
P2|2024_latest|2026YTD,cash,next_day_open_entry_fixed_5td_exit,26,26
P2|2024_latest|2026YTD,cash,same_week_close_to_next_rebalance_close_comparator,26,26
outside_requested_periods,0050,next_day_close_entry_fixed_5td_exit,1,1
outside_requested_periods,0050,next_day_open_entry_fixed_5td_exit,1,0
outside_requested_periods,0050,same_week_close_to_next_rebalance_close_comparator,1,1
outside_requested_periods,00631L,next_day_close_entry_fixed_5td_exit,1,1
outside_requested_periods,00631L,next_day_open_entry_fixed_5td_exit,1,0
outside_requested_periods,00631L,same_week_close_to_next_rebalance_close_comparator,1,1
outside_requested_periods,cash,next_day_close_entry_fixed_5td_exit,1,1
outside_requested_periods,cash,next_day_open_entry_fixed_5td_exit,1,1
outside_requested_periods,cash,same_week_close_to_next_rebalance_close_comparator,1,1


## Stock Route Coverage

period_label,source_family,route_variant,count,sum
P1,p1_defensive_policy,consensus4_else_00631L,41,41
P2|2024_latest,legacy_rs20_operating_mode,dynamic80_top1_rs20_31_bonus_proxy,206,204
P2|2024_latest,legacy_rs20_operating_mode,dynamic80_top1_rs20_proxy,206,204
P2|2024_latest,legacy_rs20_operating_mode,dynamic80_top3_rs20_risk_tiebreak_proxy,206,204
P2|2024_latest,regime_switch_hybrid_route,conservative_hurdle_route,86,85
P2|2024_latest,regime_switch_hybrid_route,dispersion_route,103,102
P2|2024_latest,regime_switch_hybrid_route,hybrid_pullback_base_mega_override,103,103
P2|2024_latest,regime_switch_hybrid_route,market_bias_pool_trend_route,103,103
P2|2024_latest,regime_switch_hybrid_route,pool_breadth_route,103,103
P2|2024_latest|2026YTD,legacy_rs20_operating_mode,dynamic80_top1_rs20_31_bonus_proxy,40,40
P2|2024_latest|2026YTD,legacy_rs20_operating_mode,dynamic80_top1_rs20_proxy,40,40
P2|2024_latest|2026YTD,legacy_rs20_operating_mode,dynamic80_top3_rs20_risk_tiebreak_proxy,40,40
P2|2024_latest|2026YTD,regime_switch_hybrid_route,conservative_hurdle_route,16,16
P2|2024_latest|2026YTD,regime_switch_hybrid_route,dispersion_route,20,20
P2|2024_latest|2026YTD,regime_switch_hybrid_route,hybrid_pullback_base_mega_override,20,20
P2|2024_latest|2026YTD,regime_switch_hybrid_route,market_bias_pool_trend_route,20,20
P2|2024_latest|2026YTD,regime_switch_hybrid_route,pool_breadth_route,20,20


## 固定 flags

- formal_model_changed=false
- trade_decision_changed=false
- active_in_trade_decision=false
- report_changed=false
- portfolio_replay_executed=false
- ready_for_strategy_replay=false
- ready_for_formal=false
- not_live_rule=true
- forward_returns_live_rule_usage=false

## 下一棒

若 Strategy Center 接受 partial stock-path caveat，下一棒交 Experiments：`TASK-BACKTEST-EXPERIMENTS-VNEXT-FULL-PERIOD-REGIME-SWITCH-BENCHMARK-EXCEPTION-DIAGNOSTIC-001`。