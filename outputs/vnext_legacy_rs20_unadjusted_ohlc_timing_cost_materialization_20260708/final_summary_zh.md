# Legacy RS20 unadjusted OHLC timing/cost materialization

- status: `legacy_rs20_unadjusted_ohlc_path_partial_ready_adjusted_close_blocked`
- trade_path_rows: 1476
- unadjusted_ohlc_ready_rows: 1464
- blocked_rows: 12
- exact_selected_stock_adjusted_close_path_ready: false
- ready_for_legacy_rs20_unadjusted_cost_timing_diagnostic: true
- ready_for_legacy_rs20_exact_cost_timing_diagnostic: false

## 判斷

Radar/Data 補出的本機 full-sweep shards 足以讓 Core 產出 partial unadjusted official OHLC path：1,464/1,476 timing rows 可算。但 adjusted close 仍為 blocked，8249 在 2024-10-11 / 2024-10-15 的 exit rows 仍缺官方日資料，不能 silent fill。

本 package 使用 selected ticker 官方未調整 OHLC 計算 path，沒有使用 `00631L + excess` 重建個股報酬。本機 EP05 TaiwanCostModel 已套到 100 萬元診斷單位本金；這是 cost model materialization，不是 formal portfolio replay。

## 下一步

可交 Strategy Center 判斷是否接受 unadjusted OHLC partial path 做 bounded diagnostic。若 Strategy Center 堅持 adjusted close exact diagnostic，則仍需 source policy / adjusted-close route；若接受 unadjusted partial diagnostic，再交 Experiments。

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
