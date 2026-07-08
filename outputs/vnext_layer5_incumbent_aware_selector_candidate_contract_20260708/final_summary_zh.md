# Layer5 incumbent-aware single-stock decision candidate contract

## Verdict
- status=layer5_incumbent_aware_selector_candidate_contract_ready_for_experiments_intake
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
- Base universe: Layer4 80-stock primary pool.
- Uses hypothetical diagnostic path state proxy because real current-holder/incumbent state is blocked.
- 100 extended watchlist and 31 high-confidence reference remain context only.
- 00631L / 0050正二 remain fallback/reference metadata, not ordinary stock rows.
- No live Layer5 rule, no A/B switch, no second-stock allocation, no portfolio replay.
- Layer5 的目標不是每天重新選分數最高的一檔，而是在只持有一檔的操作模式下，判斷續抱、換倉、或等待 fallback 的長期勝率/報酬/風險 tradeoff。

## Candidate variants
- selector_candidate_variant_count=7
- variants=confirmed_challenger_selector, fresh_best_risk_adjusted_top10_baseline, fresh_top1_baseline, high_confidence_bonus_selector, incumbent_protection_selector, lifecycle_clean_candidate_selector, reentry_confirmed_selector
- row_count=4144
- weekly_snapshot_count=592

## Blocked / proxy
- real_current_holder_state=blocked
- cash_fallback_classifier=blocked
- turnover_cost_model=blocked_placeholder
- hypothetical_path_state_proxy=true

## Next
If accepted, hand off to Experiments:
`TASK-BACKTEST-EXPERIMENTS-VNEXT-LAYER5-INCUMBENT-AWARE-SINGLE-STOCK-DECISION-DIAGNOSTIC-001`.
完成後如果下一棒明確，請直接指派下一個 thread；如果下一棒不明確，請回報 Strategy Center 判斷。不要完成後停住不回報。
