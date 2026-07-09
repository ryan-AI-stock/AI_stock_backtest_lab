# R6 guard-first market_bias override unified contract

## 結論

- 已 materialize R6_breakout_breadth_p1_risk_veto unified diagnostic contract。
- default branch = C2 / route_support baseline。
- override branch = R6 market_bias override，語義為 0050 突破前高 + pool80 breadth 擴散，且 P1-like risk veto 不成立。
- low_base 不進主權重；RS20 top3 只保留 reference，不作 selected branch。
- contract rows = 591；path_ready_share = 1.0000。
- R6 override count = 15。
- adjusted_close_ready=false；cash/bear classifier blocked；daily_report not authorized。
- 所有 branch path 仍是 diagnostic-only，不升 formal / replay / daily report / trade decision。

下一棒：交 Experiments 執行 TASK-BACKTEST-EXPERIMENTS-VNEXT-R6-GUARD-FIRST-MARKET-BIAS-OVERRIDE-UNIFIED-DIAGNOSTIC-001。

完成後如果下一棒明確，請直接指派下一個 thread；如果下一棒不明確，請回報 Strategy Center 判斷。不要完成後停住不回報。

Flags: formal_model_changed=false; trade_decision_changed=false; active_in_trade_decision=false; report_changed=false; portfolio_replay_executed=false; ready_for_strategy_replay=false; ready_for_formal=false; not_live_rule=true; forward_returns_live_rule_usage=false.