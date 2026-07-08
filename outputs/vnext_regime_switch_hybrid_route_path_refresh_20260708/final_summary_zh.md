# Regime switch hybrid route path refresh

- status: `regime_switch_hybrid_primary_path_ready_comparators_partial_adjusted_close_blocked`
- ordinary_stock_path_rows: 594
- ordinary_stock_unadjusted_ready_rows: 592
- ordinary_stock_blocked_rows: 2
- primary_hybrid_ready: true
- ready_for_experiments: true

## 判斷

Radar 補件後，primary `hybrid_pullback_base_mega_override` ordinary stock path 已 123/123 ready。整體 ordinary stock path 592/594 ready，剩餘兩筆 blocked 已列 ledger；00631L reference rows 分離，不混成 ordinary stock。

這份 refresh 可作 bounded unadjusted OHLC diagnostic input；adjusted close 仍 blocked，且不是 formal / replay / daily report。

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
