# low_base integrated C2 / route_support contract

## 結論

- 已建立 P1 low_base_score 整合版 C2 + consensus trigger + route_support / Layer4 ranking contract。
- low_base_score 只作 ranking bonus / penalty / tie-break component，沒有新增 hard filter。
- baseline 保留現行 C2 + route_support max1；另輸出 low_base_balanced、low_base_risk_aware、low_base_quality、low_base_pullback_reacceleration。
- eligible signal dates = 21；selected stock rows = 105。
- official unadjusted OHLC ready share = 1.0000。
- adjusted_close_ready=false；selected-stock unadjusted OHLC 仍是 diagnostic-only。
- 後續主結論必須 net after transaction cost；gross/no-cost 只能 secondary。

## Blocked / Proxy

signal_date,score_variant,ticker,name,blocked_item,blocked_reason,next_owner
,all,,,selected_stock_adjusted_close,historical adjusted-close route remains blocked; official unadjusted OHLC is diagnostic-only,Strategy Center policy or Radar/Data adjusted-close source route
,all,,,cash_bear_classifier,no accepted cash/bear classifier; default base remains 00631L,Strategy Center/Core if cash branch is authorized later
,all,,,low_base_hard_filter,explicitly not allowed; low_base is component only,none

下一棒：交 Experiments 執行 TASK-BACKTEST-EXPERIMENTS-VNEXT-P1-LOW-BASE-INTEGRATED-C2-ROUTE-SUPPORT-DIAGNOSTIC-001。

完成後如果下一棒明確，請直接指派下一個 thread；如果下一棒不明確，請回報 Strategy Center 判斷。不要完成後停住不回報。

Flags: formal_model_changed=false; trade_decision_changed=false; active_in_trade_decision=false; report_changed=false; portfolio_replay_executed=false; ready_for_strategy_replay=false; ready_for_formal=false; not_live_rule=true; forward_returns_live_rule_usage=false.