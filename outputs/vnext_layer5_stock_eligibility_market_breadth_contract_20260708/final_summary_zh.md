# Layer5 stock-eligibility / market-breadth final decision architecture contract

## Verdict
- status=layer5_stock_eligibility_market_breadth_contract_ready_for_experiments_intake
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
- Fixed Layer0-Layer4 context; no upstream data/source acquisition.
- 00631L is fallback/reference candidate metadata, not an ordinary stock row.
- This package creates stock-eligibility/environment features and bounded decision-architecture candidate variants only.
- 固定 Layer0~4 後，Layer5 下一步不是再微調個股排序，而是判斷何時有足夠股票 edge 可以選個股；沒有股票 edge 時，00631L fallback 是候選決策之一。

## Candidate variants
- decision_architecture_variant_count=8
- variants=0050_reference, 00631L_always_baseline, always_stock_best_previous_baseline, hybrid_stock_eligibility_gate_then_best_incumbent_or_reentry_selector, stock_allowed_only_when_pool_breadth_positive_else_00631L, stock_allowed_when_31_confidence_breadth_positive_else_00631L, stock_allowed_when_reentry_breadth_confirmed_else_00631L, stock_allowed_when_top10_dispersion_clear_else_00631L
- row_count=4736
- weekly_snapshot_count=592

## Blocked / proxy
- blocked_fields=cash_bear_classifier, real_current_holder_state, 00631L_fallback_live_rule, turnover_cost_model, portfolio_replay
- proxy_fields=weak_stock_environment_flag, stock_eligibility_composite_flag, top10_dispersion_clear_flag

## Next
If accepted, hand off to Experiments:
`TASK-BACKTEST-EXPERIMENTS-VNEXT-LAYER5-STOCK-ELIGIBILITY-MARKET-BREADTH-FINAL-DECISION-DIAGNOSTIC-001`.
完成後如果下一棒明確，請直接指派下一個 thread；如果下一棒不明確，請回報 Strategy Center 判斷。不要完成後停住不回報。
