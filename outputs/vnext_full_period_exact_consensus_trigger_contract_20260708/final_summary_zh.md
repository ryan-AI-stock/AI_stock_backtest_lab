# full-period exact consensus trigger contract

- P1 exact consensus trigger 已明文化：同一 ticker 被五個 PIT source variants 中至少四個指向，即 exact_same_ticker_consensus_ge4。
- 排名依序為 consensus_count、route_count、ticker；不使用 future return，也不使用 route_support score threshold。
- p1_exact_trigger_match_share = 1.0000。
- p2_exact_trigger_ready = true；recent_exact_trigger_ready = true。
- route_support_ge4_proxy 保留為 rejected secondary option，不作 primary。
- 若 readiness true，下一步應刷新 route_support max1 full-period same-basis state-machine contract，再交 Experiments。

Flags: formal_model_changed=false; trade_decision_changed=false; active_in_trade_decision=false; report_changed=false; portfolio_replay_executed=false; ready_for_strategy_replay=false; ready_for_formal=false; not_live_rule=true; forward_returns_live_rule_usage=false.

完成後如果下一棒明確，請直接指派下一個 thread；如果下一棒不明確，請回報 Strategy Center 判斷。不要完成後停住不回報。