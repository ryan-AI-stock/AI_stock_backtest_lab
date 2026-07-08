# P1 defensive policy benchmark/reference path

- status: `p1_defensive_policy_benchmark_close_cash_reference_ready_open_blocked`
- signal_dates: 411
- full_p1_00631l_signal_path_ready: true
- full_p1_0050_signal_path_ready: true
- cash_reference_path_ready: true
- next_day_open_ready: false
- ready_for_experiments: true

## 判斷

Core 已用同一批 P1 signal dates 補齊 00631L / 0050 signal-aligned close path 與 cash zero-return reference。next-day open benchmark path 因本機 benchmark_features 沒有 open price，明確 blocked，不杜撰。

這是 benchmark/reference path only，不是 replay / formal / daily report / trade decision。

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
- diagnostic_only=true
