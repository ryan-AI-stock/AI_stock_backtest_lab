# Layer4 pool-size / retention-constrained redesign contract

## Verdict
- status=layer4_pool_size_retention_constraint_contract_ready_for_experiments_intake
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
- Base universe: compact Layer0 active scope + Layer1 b30 pass-through.
- Layer2 and Layer3 remain context-only pass-through.
- Redesign base: C_quota_style_broad_label_pool.
- Pool sizes: 31, 40, 50, 60, 80, 100.
- Variants per size: C-quota, retention-friendly C-quota, risk-aware C-quota.
- 00631L remains fallback/reference only, not ordinary stock-pool member.

## Readiness
- weekly_snapshot_count=592
- pool_variant_count=18
- pool_rows=641136
- full_pool_week_variant_share=1.0
- ready_for_layer4_pool_size_retention_constraint_diagnostic=true

## Blocked / proxy
- AI/theme dynamic slot remains blocked placeholder; no hard-coded AI 20 quota.
- Raw turnover-share coverage is blocked in this package; traded-value ranks are retained.
- RS30, large-down, and blowoff-turnover are diagnostic proxy fields.
- Layer5 / replay / formal remain blocked.

## Next
If accepted, hand off to Experiments for
`TASK-BACKTEST-EXPERIMENTS-VNEXT-LAYER4-POOL-SIZE-RETENTION-CONSTRAINT-DIAGNOSTIC-001`.
完成後如果下一棒明確，請直接指派下一個 thread；如果下一棒不明確，請回報 Strategy Center 判斷。不要完成後停住不回報。
