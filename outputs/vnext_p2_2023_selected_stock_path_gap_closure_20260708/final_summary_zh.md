# P2 2023 selected-stock path gap closure

- task_id: `TASK-BACKTEST-CORE-VNEXT-P2-2023-SELECTED-STOCK-PATH-GAP-CLOSURE-001`
- status: `p2_2023_selected_stock_path_gap_identified_core_local_source_blocked_handoff_radar_required`
- p2_2023_missing_rows_before: 456
- p2_2023_missing_timing_rows_before: 1368
- p2_2023_patched_rows: 0
- p2_2023_remaining_blocked_rows: 1368
- unique_tickers: 100
- unique_signal_dates: 51

## 判斷

Core 已精準列出 2023 P2 selected-stock path 缺口，但目前本機沒有可直接 materialize 的 2023 selected-stock OHLC source rows。Radar full-sweep manifest 可見，但 local shard 實體不在目前 Core 可用路徑；既有 regime selected OHLC package 主要是 2024+，不能補 2023。

因此本包不把 partial caveat 留在 Core，而是把 bounded selected-ticker-only ledger 交 Radar/Data 補 source acquisition。不得做 full-market mass download；只補 ledger 中的 ticker/date/timing 所需 official unadjusted OHLC。

## 下一棒

直接交 Radar/Data：`TASK-RADAR-DATA-VNEXT-P2-2023-SELECTED-STOCK-OHLC-SOURCE-GAP-FILL-001`。

完成後如果下一棒明確，請直接指派下一個 thread；如果下一棒不明確，請回報 Strategy Center 判斷。不要完成後停住不回報。

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