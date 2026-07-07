# Layer0 compact variant design

## Verdict
- status=layer0_compact_variant_design_ready_for_strategy_center_policy_review
- recommended_primary_candidate=top250_core_conditional_buffer50
- p2_recommended_avg_weekly_count=295.01666666666665
- p2_recommended_unique_ticker_count=1215
- p2_recommended_avg_turnover_share_5d=0.8971753917247257
- p2_baseline_avg_weekly_count=400.0
- p2_baseline_unique_ticker_count=1430
- p2_baseline_avg_turnover_share_5d=0.9367931468793661
- ready_for_layer0_policy_review=true
- ready_for_experiments=false
- ready_for_formal=false

## Plain Summary
The compact candidate that best matches the user's cost-control intent is top250_core_conditional_buffer50: keep the primary weekly universe around 250-300 names, allow a small buffer only when it repeats within four weeks or is confirmed by 20D/60D traded-value rank, and keep pure 5D bursts as watchlist-only. This is a Layer0 data-pruning refinement proposal, not a formal selector.

## Flags
- formal_model_changed=false
- trade_decision_changed=false
- active_in_trade_decision=false
- report_changed=false
- portfolio_replay_executed=false
- ready_for_strategy_replay=false
- not_live_rule=true
- forward_returns_live_rule_usage=false
