# Layer5 lifecycle-state selector candidate contract

## Verdict
- status=layer5_lifecycle_state_selector_candidate_contract_ready_for_experiments_intake
- diagnostic_only=true
- formal_model_changed=false
- trade_decision_changed=false
- active_in_trade_decision=false
- report_changed=false
- portfolio_replay_executed=false
- ready_for_strategy_replay=false
- not_live_rule=true
- forward_returns_live_rule_usage=false

## Scope
- Base universe: Layer4 80 primary pool with within-80 top10 lifecycle candidate scope.
- 100 extended watchlist and 31 high-confidence subpool are context/bonus only.
- 00631L is benchmark/reference/fallback metadata only and is not an ordinary stock row.
- No A/B switch, second-stock allocation, cash rule, live Layer5 rule, portfolio replay, formal model, daily report, or trade decision.
- 這一輪測的是每天只選一檔時，是否能用 lifecycle/state 條件改善個股 selector，而不是靠單日排名或 00631L fallback 避險。

## Candidate variants
- lifecycle_selector_candidate_variant_count=9
- variants=clean_trend_low_risk_selector, high_confidence_bonus_lifecycle_selector, incumbent_reentry_baseline, lifecycle_top10_clean_state_selector, pullback_repair_reacceleration_selector, raw_top1_baseline, reentry_confirmed_lifecycle_selector, stock_vs_00631L_best_baseline, strengthening_not_overheated_selector
- row_count=5328
- weekly_snapshot_count=592

## Blocked / proxy
- blocked_fields=real_current_holder_state, cash_bear_classifier, 00631L_fallback_rule, turnover_cost_model, portfolio_replay
- proxy_fields=risk_bucket, large_down_day_count_20d_proxy, blowoff_turnover_without_price_continuation_proxy, RS30_proxy

## Next
If accepted, hand off to Experiments:
`TASK-BACKTEST-EXPERIMENTS-VNEXT-LAYER5-LIFECYCLE-STATE-SINGLE-STOCK-SELECTOR-DIAGNOSTIC-001`.
完成後如果下一棒明確，請直接指派下一個 thread；如果下一棒不明確，請回報 Strategy Center 判斷。不要完成後停住不回報。
