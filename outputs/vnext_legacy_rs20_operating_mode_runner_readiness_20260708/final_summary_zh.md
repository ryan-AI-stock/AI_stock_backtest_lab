# Legacy RS20 operating mode materialized runner readiness

## Verdict
- status=legacy_rs20_operating_mode_materialized_proxy_runner_ready_exact_path_blocked
- diagnostic_only=true
- exact_legacy_runner_found=false
- exact_legacy_runner_not_found=true
- same_week_close_forward_5td_proxy_ready=true
- next_trading_day_exact_path_ready=false
- formal_model_changed=false
- trade_decision_changed=false
- active_in_trade_decision=false
- report_changed=false
- portfolio_replay_executed=false
- ready_for_strategy_replay=false
- not_live_rule=true
- forward_returns_live_rule_usage=false

## Scope
- Main materialized candidate: weekly RS20 top1 within Layer4 80 primary pool.
- Comparators: top3 RS20 risk/context tie-break, 7-core flag context, 31 high-confidence bonus context.
- This package is runner/readiness only, not formal-ready and not a daily trade decision.
- Exact next-day adjusted-close path is blocked because selected dynamic80 stock price coverage is not materialized locally.

## Coverage
- materialized_signal_row_count=2368
- weekly_snapshot_count=592
- signal_variant_count=4
- selected_unique_ticker_count=388

## Blocked / proxy
- blocked_fields=exact_legacy_runner, full_dynamic80_selected_stock_adjusted_close_path, next_trading_day_entry_exact_return, rank_deterioration_exit_materialized_path, formal_cost_model_application
- proxy_fields=proxy_stock_forward_return_5d, same_week_close_to_next_5td_proxy, rs20_31_bonus_score, rs20_risk_context_score

## Next
If accepted, hand off to Experiments:
`TASK-BACKTEST-EXPERIMENTS-VNEXT-LEGACY-RS20-OPERATING-MODE-COST-TIMING-DIAGNOSTIC-001`.
完成後如果下一棒明確，請直接指派下一個 thread；如果下一棒不明確，請回報 Strategy Center 判斷。不要完成後停住不回報。
