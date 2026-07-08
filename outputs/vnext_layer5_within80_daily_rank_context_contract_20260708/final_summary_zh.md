# Layer5 within-80 daily rank context contract

## Verdict
- status=layer5_within80_daily_rank_context_contract_ready_for_experiments_intake
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
- 100 extended watchlist and 31 high-confidence subpool are context/reference flags only.
- rank_context_date_basis=weekly_signal_date_as_layer5_pre_action_context
- Top1/top2/top3/top5/top10 are diagnostic context groups only.
- Final single-stock selector candidates are diagnostic candidates only.
- No A/B switch, no fallback trading rule, no Layer5 action rule.
- Layer5 final selector 的目標不是保留所有 winner，而是在每個交易日只選一檔時，找出長期勝率/報酬/風險表現最好的決策模式。

## Coverage
- rows=47360
- weekly_snapshot_count=592
- selected_count_min=80
- selected_count_max=80
- top1_rows=592
- top3_rows=1776
- top5_rows=2960
- top10_rows=5920
- in_31_high_confidence_reference_rows=18352
- extended_100_to_80_reentry_recent_4w_rows=12522
- final_single_stock_selector_candidate_rows=2960
- final_single_stock_selector_candidate_variant_count=5

## Layer5 Metric Focus
- Primary: median/mean vs 0050 and 00631L, hit-rate, fail_0050 rate, path-like return proxy if available, downside risk proxy if available, turnover/churn proxy, period stability.
- Secondary only: top-decile retention / missed winner attribution.

## Next
If accepted, hand off to Experiments:
`TASK-BACKTEST-EXPERIMENTS-VNEXT-LAYER5-WITHIN80-DAILY-RANK-CONTEXT-DIAGNOSTIC-001`.
完成後如果下一棒明確，請直接指派下一個 thread；如果下一棒不明確，請回報 Strategy Center 判斷。不要完成後停住不回報。
