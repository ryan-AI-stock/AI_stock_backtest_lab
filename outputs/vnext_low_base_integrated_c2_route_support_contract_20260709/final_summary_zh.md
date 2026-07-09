# low_base integrated C2 / route_support contract

## 結論

- 已建立 P1 low_base_score 整合版 C2 + consensus trigger + route_support / Layer4 ranking contract。
- low_base_score 只作 ranking bonus / penalty / tie-break component，沒有新增 hard filter。
- baseline 保留現行 C2 + route_support max1；另輸出 low_base_balanced、low_base_risk_aware、low_base_quality、low_base_pullback_reacceleration。
- eligible signal dates = 21；selected stock rows = 105。
- official unadjusted OHLC ready share = 0.9905。
- adjusted_close_ready=false；selected-stock unadjusted OHLC 仍是 diagnostic-only。
- 後續主結論必須 net after transaction cost；gross/no-cost 只能 secondary。

## Blocked / Proxy

signal_date,entry_date,exit_date,score_variant,ticker,name,market,timing_variant,required_price_fields,blocked_item,blocked_reason,next_owner
2018-03-16,2018-03-19,2018-03-26,low_base_risk_aware,8464,億豐,TWSE,next_day_close_entry_fixed_5td_exit,"entry_open,entry_close,exit_close",selected_stock_official_unadjusted_ohlc_path,new low_base integrated top1 not covered by prior selected-ticker OHLC path package,Radar/Data bounded selected-ticker-only OHLC gap fill if Strategy Center wants this variant tested
,,,all,,,,,,selected_stock_adjusted_close,historical adjusted-close route remains blocked; official unadjusted OHLC is diagnostic-only,Strategy Center policy or Radar/Data adjusted-close source route
,,,all,,,,,,cash_bear_classifier,no accepted cash/bear classifier; default base remains 00631L,Strategy Center/Core if cash branch is authorized later
,,,all,,,,,,low_base_hard_filter,explicitly not allowed; low_base is component only,none

下一棒：交 Radar/Data 做 bounded selected-ticker-only OHLC gap fill，補新 low_base integrated top1 缺價。

完成後如果下一棒明確，請直接指派下一個 thread；如果下一棒不明確，請回報 Strategy Center 判斷。不要完成後停住不回報。

Flags: formal_model_changed=false; trade_decision_changed=false; active_in_trade_decision=false; report_changed=false; portfolio_replay_executed=false; ready_for_strategy_replay=false; ready_for_formal=false; not_live_rule=true; forward_returns_live_rule_usage=false.