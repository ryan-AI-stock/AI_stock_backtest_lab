# route_support max1 full-period same-basis modelization contract

- 已 materialize P1/P2/2024-latest/2026YTD/full integrated state-machine contract。
- Default state = 00631L state-hold；C2 + consensus trigger 才允許 route_support quant score max1 stock exception。
- P1/P2/recent 均使用 full-period exact consensus trigger contract，不使用 route_support>=4 proxy trigger。
- official_unadjusted_ohlc_ready_share = 1.0000；adjusted_close_ready=false。
- p2_recent_proxy_trigger_stock_rows = 0。
- Cost model ready：EP05 TaiwanCostModel unit-notional transition cost；後續主結論必須 net after transaction cost。
- 若 readiness true，下一棒交 Experiments 做 full-period exact same-basis route_support max1 diagnostic。

下一棒：若 ready_for_experiments=true，交 Experiments；若 false，依 missing ledger 補 bounded selected-ticker path。

Flags: formal_model_changed=false; trade_decision_changed=false; active_in_trade_decision=false; report_changed=false; portfolio_replay_executed=false; ready_for_strategy_replay=false; ready_for_formal=false; not_live_rule=true; forward_returns_live_rule_usage=false.

完成後如果下一棒明確，請直接指派下一個 thread；如果下一棒不明確，請回報 Strategy Center 判斷。不要完成後停住不回報。