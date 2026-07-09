# P1 risk-adjusted RS20 OHLC absorption

## 結論

- 已吸收 Radar/Data 351 筆 selected-stock official unadjusted OHLC gap fill。
- P1 new RS20 path coverage 更新為 405/411。
- 仍有 6 筆 official target-missing rows，Core 決策：維持 explicit blocked，不改 timing、不 silent fill、不自行替換 00631L。
- 因 6 筆仍 blocked，full exact P1 comparison 尚未 ready；需要 Strategy Center 接受 partial-blocked policy，或另行授權 timing/fallback 政策後，才交 Experiments 產生主比較表。

## Remaining blocked rows

field,status,blocked_reason,core_policy,next_owner,signal_date,entry_date,exit_date,ticker,name,market
remaining6_official_ohlc_path,blocked_policy_retained,official selected-month route and bounded exact-day fallback both missing target row,maintain blocked; do not change timing; do not use neighboring date; do not fallback to 00631L without Strategy Center policy,Strategy Center policy decision before Experiments primary comparison,,,,,,
remaining6_official_ohlc_path,blocked,official_selected_month_and_exact_day_fallback_target_missing_no_silent_fill,explicit blocked ledger; no silent fill,Strategy Center policy decision,2015-07-09,2015-07-10,2015-07-17,6121,新普,TPEx
remaining6_official_ohlc_path,blocked,official_selected_month_and_exact_day_fallback_target_missing_no_silent_fill,explicit blocked ledger; no silent fill,Strategy Center policy decision,2015-09-18,2015-09-21,2015-09-29,3231,緯創,TWSE
remaining6_official_ohlc_path,blocked,official_selected_month_and_exact_day_fallback_target_missing_no_silent_fill,explicit blocked ledger; no silent fill,Strategy Center policy decision,2015-09-25,2015-09-29,2015-10-06,5490,同亨,TPEx
remaining6_official_ohlc_path,blocked,official_selected_month_and_exact_day_fallback_target_missing_no_silent_fill,explicit blocked ledger; no silent fill,Strategy Center policy decision,2016-07-07,2016-07-08,2016-07-15,2231,為升,TWSE
remaining6_official_ohlc_path,blocked,official_selected_month_and_exact_day_fallback_target_missing_no_silent_fill,explicit blocked ledger; no silent fill,Strategy Center policy decision,2017-09-15,2017-09-18,2017-09-25,2340,台亞,TWSE
remaining6_official_ohlc_path,blocked,official_selected_month_and_exact_day_fallback_target_missing_no_silent_fill,explicit blocked ledger; no silent fill,Strategy Center policy decision,2019-05-17,2019-05-20,2019-05-27,2912,統一超,TWSE


## Reference

- 00631L / 0050 buy-hold reference rows: 2
- cost_model_version: `taiwan_standard_fee_tax_v1`
- adjusted_close_ready=false for selected stocks。

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
