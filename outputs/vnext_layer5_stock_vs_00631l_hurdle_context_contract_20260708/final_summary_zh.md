# Layer5 stock-vs-00631L hurdle / fallback context contract

## Verdict
- status=layer5_stock_vs_00631l_hurdle_context_contract_ready_for_experiments_intake
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
- Base stock candidate line: incumbent-aware / reentry-confirmed selector context.
- 00631L is fallback candidate / hurdle reference, not an ordinary stock-pool row.
- 0050 is comparison reference only.
- Cash/bear classifier remains blocked; this package does not create a cash rule.
- Layer5 不應在沒有足夠股票 edge 時硬選個股；最終每日主推薦可以是個股，也可以是 00631L fallback。

## Candidate variants
- decision_candidate_variant_count=7
- variants=0050_reference, 00631L_always_baseline, always_stock_incumbent_reentry_baseline, stock_if_31_high_confidence_bonus_and_state_clean_else_00631L, stock_if_high_confidence_else_00631L, stock_if_incumbent_still_valid_else_00631L_or_best_confirmed_challenger, stock_if_reentry_confirmed_and_not_high_risk_else_00631L
- row_count=4144
- weekly_snapshot_count=592

## Next
If accepted, hand off to Experiments:
`TASK-BACKTEST-EXPERIMENTS-VNEXT-LAYER5-STOCK-VS-00631L-HURDLE-FALLBACK-DIAGNOSTIC-001`.
完成後如果下一棒明確，請直接指派下一個 thread；如果下一棒不明確，請回報 Strategy Center 判斷。不要完成後停住不回報。
