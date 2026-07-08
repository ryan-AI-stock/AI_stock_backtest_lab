# Layer4 80-stock primary pool contract refresh

## Verdict
- status=layer4_80_primary_pool_contract_ready_for_strategy_center_judgment
- diagnostic_only=true
- formal_model_changed=false
- trade_decision_changed=false
- active_in_trade_decision=false
- report_changed=false
- portfolio_replay_executed=false
- ready_for_strategy_replay=false
- not_live_rule=true
- forward_returns_live_rule_usage=false

## Primary Policy
- Layer4 primary weekly candidate pool = 80 stocks.
- Primary variant = `C_risk_aware_retention_constrained_quota_80`.
- 31-stock pool is downgraded to high-confidence subpool reference only.
- 100-stock pool is retained as extended/watchlist reference only.
- 00631L / 0050正二 remain fallback/reference metadata, not ordinary stock-pool members.
- AI/theme dynamic slot remains blocked placeholder; no hard-coded AI 20.

## Coverage
- weekly_snapshot_count=592
- primary_pool_rows=47360
- primary_selected_count_min=80
- primary_selected_count_max=80
- primary_shortfall_count=0
- reference_100_rows=59200
- reference_31_rows=18352

## Next
回 Strategy Center 判斷是否要開 Layer5 前置 `within-80 daily rank context diagnostic`。
不要自行交 Experiments 進 Layer5，除非 Strategy Center 明確授權。
完成後如果下一棒明確，請直接指派下一個 thread；如果下一棒不明確，請回報 Strategy Center 判斷。不要完成後停住不回報。
