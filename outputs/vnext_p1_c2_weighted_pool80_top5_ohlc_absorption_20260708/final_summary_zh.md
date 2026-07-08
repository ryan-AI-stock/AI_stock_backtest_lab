# P1 C2 weighted pool80 top5 OHLC absorption

- Radar selected-ticker OHLC source fill absorbed into Core weighted pool80 top5 contract.
- official unadjusted OHLC ready rows = 630/630.
- adjusted_close_ready=false；official unadjusted OHLC 只能作 diagnostic path，不可 formal。
- transition cost fields are ready; Experiments main conclusion must be net after transaction cost.
- ready_for_p1_c2_weighted_pool80_top5_multi_stock_diagnostic=true.
- ready_for_formal=false；ready_for_strategy_replay=false。

下一棒：交 Experiments 做 P1 C2 weighted pool80 top5 multi-stock exception diagnostic。

Flags: formal_model_changed=false; trade_decision_changed=false; active_in_trade_decision=false; report_changed=false; portfolio_replay_executed=false; ready_for_strategy_replay=false; ready_for_formal=false; not_live_rule=true; forward_returns_live_rule_usage=false.

完成後如果下一棒明確，請直接指派下一個 thread；如果下一棒不明確，請回報 Strategy Center 判斷。不要完成後停住不回報。