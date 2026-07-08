# P1 C2 market-health gated consensus4 adjusted state-machine contract

- task_id: `TASK-BACKTEST-CORE-VNEXT-P1-C2-MARKET-HEALTH-CONSENSUS4-ADJUSTED-STATE-MACHINE-CONTRACT-001`
- status: `p1_c2_market_health_consensus4_adjusted_state_machine_partial_adjusted_close_blocked`
- ready_for_p1_c2_market_health_consensus4_net_cost_diagnostic: `false`
- selected_stock_adjusted_close_ready_share: 0.238095
- selected_stock_adjusted_remaining_blocked_rows: 16
- unadjusted_ohlc_comparator_ready_share: 1.000000
- state_machine_interval_rows: 412
- transition_rows: 41

## 語義

Default state 是持有 00631L。只有 C2 market health gate 通過，且 consensus4 exception active，才允許切到 selected stock；C2 gate 失效或 exception invalid 時切回 00631L。同一檔連續有效則續抱，不重複買賣。

C2 定義固定為：0050 above MA60 + 20D/40D returns non-negative。Core 只 materialize contract/path/cost readiness，不做 Experiments verdict。

成本已使用 EP05 TaiwanCostModel，包含買賣手續費、證券交易稅、ETF/股票賣出稅率差異與 transition cost。no-cost/gross 不作主結論。

selected-stock adjusted close 目前仍未完整 ready；官方 unadjusted OHLC comparator 可保留作 proxy comparator，但不可包裝成 formal 或 adjusted-close path。

下一棒明確：請交 Radar/Data 做 selected-ticker-only adjusted close source fill，不做 full-market mass download。建議任務：TASK-RADAR-DATA-VNEXT-P1-C2-CONSENSUS4-SELECTED-STOCK-ADJUSTED-CLOSE-SOURCE-FILL-001。

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