# P1 00631L base + consensus4 exception state-machine contract

- task_id: `TASK-BACKTEST-CORE-VNEXT-P1-00631L-BASE-CONSENSUS4-EXCEPTION-STATE-MACHINE-CONTRACT-001`
- status: `p1_00631L_base_consensus4_exception_state_machine_contract_ready_diagnostic_only`
- ready_for_p1_00631L_base_consensus4_state_machine_diagnostic: `true`
- state_machine_interval_rows: 412
- transition_rows: 76
- gross_total_return_before_cost_proxy: 7.602204
- net_total_return_after_transition_cost_proxy: 4.920754
- gross_mdd: -0.460414
- net_mdd: -0.510318

## 語義

Default state 是持有 00631L。只有 consensus4 stock exception 觸發時才切到個股；連續同一檔維持持有，不重複買賣；exception 失效時切回 00631L。00631L base 不再被做成每週清倉再買回。

成本使用 EP05 TaiwanCostModel unit-notional transition proxy，ETF 與股票賣出稅率分開。這仍是 diagnostic-only contract，不是 formal 或 live rule。

cash condition / bear classifier 仍 blocked，不杜撰。

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