# P1 state-hold benchmark / base+exception path contract

- task_id: `TASK-BACKTEST-CORE-VNEXT-P1-STATE-HOLD-BASE-EXCEPTION-PATH-CONTRACT-001`
- status: `p1_state_hold_base_exception_path_contract_ready_diagnostic_only`
- ready_for_p1_base_exception_diagnostic: `true`
- contract_rows: 822
- blocked_contract_rows: 0

## Benchmark Daily State-Hold

- 0050: total_return=1.113979, MDD=-0.338276, rows=1952
- 00631L: total_return=3.796028, MDD=-0.521306, rows=1956

## Base+Exception Interval Proxy

base_asset,cumulative_return_before_cost_proxy,drawdown_before_cost_proxy
0050,3.544228417281264,-0.3787521369830169
00631L,9.537008034630349,-0.4697093787020842


## 語義

`signal-aligned all 00631L` 不再被包裝成 live fallback/base rule。本包把 0050 / 00631L buy-hold daily state-hold path 與 signal-date interval contract 分開；base asset 是持續持有，只有 consensus4 stock exception 觸發時切到個股。

cash condition / bear classifier 尚未 ready，本包只列 blocked，不杜撰 cash rule。

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