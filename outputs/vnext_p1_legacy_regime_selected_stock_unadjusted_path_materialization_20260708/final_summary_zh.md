# P1 legacy/regime selected-stock unadjusted path materialization

- status: `p1_selected_stock_unadjusted_path_blocked_source_request_ready`
- P1 signal rows: 3699
- selected unique tickers: 509
- trade path rows requested: 11097
- ordinary stock path ready rows: 0
- blocked rows: 10926
- ready_for_experiments: false

## 判斷

Core 找到 P1 legacy / regime selected signal rows，但本機 P1 selected-stock official unadjusted OHLC source 不存在；既有 Radar selected price package 只覆蓋 2024-01-02 之後，full-sweep manifest 指向的 2015 shard 在本機也不存在。

因此本包不交 Experiments。下一棒應交 Radar/Data，只補 P1 selected tickers/date range，不做 full-market mass download。

## Flags

- formal_model_changed=false
- trade_decision_changed=false
- active_in_trade_decision=false
- report_changed=false
- portfolio_replay_executed=false
- ready_for_strategy_replay=false
- not_live_rule=true
- forward_returns_live_rule_usage=false
- diagnostic_only=true
