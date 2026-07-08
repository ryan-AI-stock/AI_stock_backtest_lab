# Regime switch hybrid route market fields/path materialization

- status: `regime_switch_hybrid_route_market_fields_ready_path_partial_blocked`
- market_bias_fields_ready: true
- pool_breadth_dispersion_fields_ready: true
- route_signal_table_ready: true
- next_day_unadjusted_path_ready: false
- next_day_unadjusted_path_ready_share: 0.6094276094276094
- next_day_open_ready: false
- next_day_close_ready: false
- adjusted_close_ready: false
- ready_for_experiments: false

## 判斷

Core 已補 0050 slope / BIAS / BIAS expansion / 20-40-60D high breakout 欄位，並從 Layer4 80 pool 重新 materialize pool breadth / RS dispersion。這些都是 PIT feature columns，不含 Core threshold 決策。

Path 端目前只有 partial selected ticker official unadjusted OHLC 覆蓋；adjusted close 仍 blocked，00631L reference path 仍需與 ordinary stock path 分開。
因此 Core 不直接交 Experiments，除非 Strategy Center 接受 partial-row diagnostic；下一棒較明確是 Radar/Data 補 regime-route selected ticker OHLC source package。

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
