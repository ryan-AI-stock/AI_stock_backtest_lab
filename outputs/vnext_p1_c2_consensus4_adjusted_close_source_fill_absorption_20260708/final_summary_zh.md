# P1 C2 consensus4 adjusted close source fill absorption

- task_id: `TASK-BACKTEST-CORE-VNEXT-P1-C2-CONSENSUS4-ADJUSTED-CLOSE-SOURCE-FILL-ABSORPTION-001`
- status: `p1_c2_consensus4_adjusted_close_source_fill_absorbed_blocked_no_refresh`
- Radar patch result: patched_interval_rows=0, remaining_blocked_interval_rows=16.
- Core adjusted state-machine refresh is not possible because there are no accepted adjusted-close patch rows.
- ready_for_p1_c2_market_health_consensus4_net_cost_diagnostic=false.
- unadjusted OHLC comparator remains ready as proxy-only, not adjusted-close evidence.

## Core judgment

這不是 Core ingest blocker，而是 source / policy blocker：目前沒有可接受的 selected-stock adjusted close source route。不得 silent fill，也不得把 unadjusted OHLC comparator 包裝成 adjusted-close path。

## Strategy Center decision needed

1. 核准 licensed third-party adjusted close source route；或
2. 啟動 official corporate-action adjustment contract，從官方除權息/減資事件建立調整價；或
3. 接受 adjusted net-cost diagnostic 繼續 blocked，只保留 unadjusted comparator 作 proxy diagnostic。

## Flags

- formal_model_changed=false
- trade_decision_changed=false
- active_in_trade_decision=false
- report_changed=false
- portfolio_replay_executed=false
- ready_for_strategy_replay=false
- ready_for_formal=false
- not_live_rule=true
- forward_returns_live_rule_usage=false

完成後如果下一棒明確，請直接指派下一個 thread；如果下一棒不明確，請回報 Strategy Center 判斷。不要完成後停住不回報。