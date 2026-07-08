# vNext 2026-07-08 EOD signal materialization refresh

## 結論

- 2026-07-08 官方 EOD historical window source 已吸收為 Core bounded materialization input。
- Layer0 compact active universe 與 RS20 top3 reference 已可用 2026-07-08 官方未調整 OHLCV 重算。
- C2 market health gate 已可 exact 判定，且 2026-07-08 C2 通過。
- C2 / route_support 個股 selected signal 仍不可發布：exact consensus trigger / Layer4 primary80 / route_support max1 尚未 materialized 到 2026-07-08。

## 今日參考結果

- RS20 top1 reference: `8261 富鼎`。
- RS20 top3 reference: `8261|3055|6182`。
- C2 gate ready: `True`；C2 gate pass: `True`。
- c2_selected_asset_type: `blocked`。

## Readiness

- ready_for_vnext_daily_report_selected_signal_publish=false。
- ready_for_live_publish=false。
- ready_for_experiments=false。
- future_data_violation_count=0。

## Flags

- formal_model_changed=false
- trade_decision_changed=false
- active_in_trade_decision=false
- report_changed=true
- portfolio_replay_executed=false
- ready_for_strategy_replay=false
- ready_for_formal=false
- not_live_rule=true
- forward_returns_live_rule_usage=false
