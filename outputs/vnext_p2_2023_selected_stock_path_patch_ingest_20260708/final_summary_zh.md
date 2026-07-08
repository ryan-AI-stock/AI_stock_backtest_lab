# P2 2023 selected-stock OHLC path patch ingest

- task_id: `TASK-BACKTEST-CORE-VNEXT-P2-2023-SELECTED-STOCK-PATH-PATCH-INGEST-001`
- status: `p2_2023_selected_stock_unadjusted_ohlc_path_patch_ingested_primary_ready_same_week_terminal_partial`
- p2_2023_patched_rows: 1359
- p2_2023_remaining_blocked_rows: 9
- next_day_close_ready: `true` (456/456)
- next_day_open_ready: `true` (456/456)
- same_week_close_ready: `false` (447/456)

## 判斷

2023 P2 selected-stock primary path 缺口已補齊：next-day close fixed 5TD 與 next-day open timing 都是 456/456 ready。剩餘 9 筆只屬於 2023-12-29 same-week comparator terminal rows，因 Core 原 ledger 沒有 exit_date，本次不 silent fill。

因此可交 Experiments 重跑 full-period regime switch benchmark + exception diagnostic，primary timing 應以 next-day close fixed 5TD 為準；same-week comparator 對 2023 terminal week 仍需標 partial。

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