# P1 risk-adjusted RS20 branch contract

## 結論

- 已建立 P1 新版 risk-adjusted RS20 signal / score contract。
- 舊 RS20 對照使用既有 `dynamic80_top3_rs20_risk_tiebreak_proxy` + next-day close 5TD unadjusted OHLC path。
- 00631L / 0050 reference 使用 buy-hold / state-hold reference，不混 signal-aligned weekly rebuy。
- 新 RS20 selected-stock OHLC path ready share=0.1314；若小於 1，需 Radar/Data 補 bounded selected ticker path 後再交 Experiments。

## 新 RS20 score

`RS20 branch = RS20 動能 - BIAS60 過熱扣分 - 波動風險扣分 + low_base / quality / route_support 加分`

實作公式：
`0.45*RS20_rank_pct - 0.18*BIAS60_percentile - 0.12*volatility_pctile + 0.12*low_base + 0.08*quality_support + 0.05*route_support_or_neutral`

## Coverage

- weekly_signal_rows=411
- old_rs20_source_rows=411
- new_rs20_selected_path_ready_rows=54
- new_rs20_selected_path_missing_rows=357
- reference_rows=2

## Next owner

- ready_for_p1_rs20_comparison_experiments=False
- ready_for_radar_p1_risk_adjusted_rs20_selected_ohlc_gap_fill=True

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
