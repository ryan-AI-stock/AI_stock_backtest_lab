# Layer3 Momentum Strength Gross / Risk Penalty Readiness

Status: partial_ready_layer3_momentum_redesign_proxy_limited

Boundary: feature/contract readiness only; no selector, no live rule, no replay, no formal/report/trade change.

Readiness:
- ready_for_layer3_momentum_redesign_event_diagnostic=true
- rs30_exact_available=false
- rs30_proxy_available=true
- persistence_daily_winrate_available=false
- persistence_rs5_positive_share_proxy_available=true
- volatility_large_down_day_available=true
- blowoff_turnover_available=true
- blowoff_turnover_source_quality=proxy
- ready_for_portfolio_like_diagnostic=false
- ready_for_strategy_replay=false
- ready_for_formal=false
- future_data_violation_count=0

Blocked / proxy fields:
- RS30: proxy; RS30 exact not materialized; RS30_proxy=(RS20+RS40)/2
- persistence_daily_winrate: proxy; exact 1D beat-0050 daily history unavailable; RS5_positive_share_20d/30d used as proxy
- volatility_large_down_day: PIT-ready; computed from stock adjusted_close daily history only
- blowoff_turnover: proxy; turnover/value spike plus RS short-window deterioration proxy; not a formal blow-off definition
- MA20_MA60_slope: proxy; computed from MA level change over 20 trading days
- stock_BIAS_percentile: PIT-ready; materialized in stock_features
- risk_score_bucket: PIT-ready; existing vNext weekly snapshot diagnostic fields
- forward_return_as_rule: prohibited; forward returns are not used

Flags:
- formal_model_changed=false
- trade_decision_changed=false
- active_in_trade_decision=false
- report_changed=false
- portfolio_replay_executed=false
- ready_for_strategy_replay=false
- not_live_rule=true
- forward_returns_live_rule_usage=false