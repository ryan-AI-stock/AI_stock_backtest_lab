# P1 C2 top5 OHLC absorption and exright review

- Top5 selected-ticker official unadjusted OHLC path is now ready: 1020/1020 C2-true candidate rows.
- Transition cost design from Core top5 contract is retained and must be used by Experiments; no-cost/gross only secondary reference.
- Adjusted close remains blocked. Radar found dividend candidates, but exact historical ex-date/capital-change route is incomplete, so Core does not compute adjustment factors.
- ready_for_p1_c2_top5_multi_stock_exception_count_diagnostic=true.
- ready_for_formal=false; ready_for_strategy_replay=false.

下一棒：交 Experiments rerun TASK-BACKTEST-EXPERIMENTS-VNEXT-P1-C2-MULTI-STOCK-EXCEPTION-COUNT-DIAGNOSTIC-001-RERUN-AFTER-TOP5-CONTRACT。

Flags: formal_model_changed=false; trade_decision_changed=false; active_in_trade_decision=false; report_changed=false; portfolio_replay_executed=false; ready_for_strategy_replay=false; ready_for_formal=false; not_live_rule=true; forward_returns_live_rule_usage=false.

完成後如果下一棒明確，請直接指派下一個 thread；如果下一棒不明確，請回報 Strategy Center 判斷。不要完成後停住不回報。