# P1 C2 exact consensus4 top5 exception candidate contract

- 本輪已修正前一輪偏差：top1~top5 直接由原 consensus4 多 route 同股確認邏輯重建，不使用 Layer4 generic top5 proxy。
- exact top1 對齊 prior single consensus4 exception share = 1.0000。
- exact top1 對齊 C2 allowed prior exception share = 1.0000。
- C2=true rank<=5 official unadjusted OHLC ready share = 1.0000。
- adjusted_close_ready=false；official unadjusted OHLC 只能作 diagnostic path，不可 formal。
- 後續 Experiments 主結論必須 net after transaction cost；gross/no-cost 只能 secondary。
- ready_for_formal=false；ready_for_strategy_replay=false。

下一棒：交 Experiments rerun exact consensus4 top1~top5 multi-stock exception diagnostic。

Flags: formal_model_changed=false; trade_decision_changed=false; active_in_trade_decision=false; report_changed=false; portfolio_replay_executed=false; ready_for_strategy_replay=false; ready_for_formal=false; not_live_rule=true; forward_returns_live_rule_usage=false.

完成後如果下一棒明確，請直接指派下一個 thread；如果下一棒不明確，請回報 Strategy Center 判斷。不要完成後停住不回報。