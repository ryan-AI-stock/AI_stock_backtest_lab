# P1 C2 route_support max1 modelization readiness

- 主線已收斂為單押：00631L state-hold base + C2 market health gate + consensus trigger + route_support quant score max1。
- State machine 已建立：同股續抱不重買；gate/trigger 失效回 00631L；top1 改變時計 stock-to-stock transition。
- stock exception official unadjusted OHLC ready share = 1.0000。
- Cost model ready：EP05 TaiwanCostModel unit-notional transition cost，ETF/stock transaction tax split retained。
- adjusted_close_ready=false；cash/bear classifier blocked；formal/replay blocked。
- 後續 Experiments 主結論必須 net after transaction cost；gross/no-cost 只能 secondary。

下一棒：交 Experiments 做 P1 C2 route_support max1 modelization diagnostic。

Flags: formal_model_changed=false; trade_decision_changed=false; active_in_trade_decision=false; report_changed=false; portfolio_replay_executed=false; ready_for_strategy_replay=false; ready_for_formal=false; not_live_rule=true; forward_returns_live_rule_usage=false.

完成後如果下一棒明確，請直接指派下一個 thread；如果下一棒不明確，請回報 Strategy Center 判斷。不要完成後停住不回報。