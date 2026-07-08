# P1 legacy/regime unadjusted path refresh

- status: `p1_legacy_regime_unadjusted_path_partial_ready_blocked_rows_retained`
- ordinary_stock_trade_path_rows: 10926
- ordinary_stock_path_ready_rows: 10785
- ordinary_stock_blocked_rows: 141
- ordinary_stock_ready_share: 0.9870950027457441
- ready_for_experiments: true

## 判斷

Radar 補件後，P1 ordinary selected-stock unadjusted OHLC path 已達 10,785/10,926 ready；剩餘 141 rows 保留 blocked ledger。Core 接受作 bounded partial diagnostic input，但不是 full coverage、不是 adjusted close、不是 formal/replay。

Experiments 必須顯式回報 blocked rows 對各 variant / period 的影響；不得 silent fill，也不得用 00631L + excess 重建 ordinary stock return。

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
