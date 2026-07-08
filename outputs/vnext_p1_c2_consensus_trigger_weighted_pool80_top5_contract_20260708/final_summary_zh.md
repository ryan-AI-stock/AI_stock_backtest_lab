# P1 C2 consensus-trigger weighted pool80 top5 contract

- 本輪採兩層架構：C2 market health + exact consensus4 trigger 決定是否允許個股例外；通過後才在 Layer4 primary 80 pool 內做 weighted top5 排名。
- 排名不是 Layer4 generic rank，也不是 future-return rank；使用 Layer1 quality、RS、liquidity、BIAS health、route support、risk inverse 等 PIT quant components。
- prior single exception 只作 comparator/reference，不作校準目標。
- eligible signal dates = 21；contract rows = 630。
- official unadjusted OHLC ready share = 0.0460。
- adjusted_close_ready=false；unadjusted OHLC 只能作 diagnostic path，不可 formal。
- 後續 Experiments 主結論必須 net after transaction cost；gross/no-cost 只能 secondary。
- ready_for_formal=false；ready_for_strategy_replay=false。

下一棒：若 OHLC readiness pass，交 Experiments 做 P1 C2 weighted pool80 top5 multi-stock diagnostic；若 partial，交 Radar/Data 補 bounded selected-ticker OHLC path。

Flags: formal_model_changed=false; trade_decision_changed=false; active_in_trade_decision=false; report_changed=false; portfolio_replay_executed=false; ready_for_strategy_replay=false; ready_for_formal=false; not_live_rule=true; forward_returns_live_rule_usage=false.

完成後如果下一棒明確，請直接指派下一個 thread；如果下一棒不明確，請回報 Strategy Center 判斷。不要完成後停住不回報。