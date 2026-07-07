# Layer0 top300_buffer100 unique count audit

## Verdict
- status=layer0_top300_buffer100_unique_count_audit_completed_high_unique_mainly_churn_not_duplicate
- p2_unique_ticker_count=1430
- p2_weekly_snapshot_count=180
- p2_avg_weeks_per_ticker=50.34965034965035
- p2_median_weeks_per_ticker=25.0
- reference_variant_mixed=false
- weekly_duplicate_detected=false
- traded_value_unit_status=passed_traded_value_not_volume
- ready_for_layer0_policy_review=true
- ready_for_experiments=false
- ready_for_formal=false

## Plain Summary
The high P2 unique count is not explained by mixed reference variants or weekly duplicate rows. It is mainly a consequence of using weekly 5D traded-value ranks: many names briefly enter the top300/core or buffer during short turnover bursts. This is plausible for a broad data-pruning universe, but it is expensive for Layer1 source acquisition unless later source work is period-scoped. If Strategy Center wants a more stable Layer0, the clean alternatives are 20D/60D traded-value core ranks, a repeat-appearance requirement, or making surge exceptions watchlist-only.

## Flags
- formal_model_changed=false
- trade_decision_changed=false
- active_in_trade_decision=false
- report_changed=false
- portfolio_replay_executed=false
- ready_for_strategy_replay=false
- not_live_rule=true
- forward_returns_live_rule_usage=false
