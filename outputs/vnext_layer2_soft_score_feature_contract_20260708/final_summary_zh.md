# Layer2 soft-score feature contract readiness

## Verdict
- status=layer2_soft_score_feature_contract_ready_for_experiments_planning
- rows=171499
- weekly_snapshot_count=592
- unique_ticker_count=1612
- capital_support_available_share=1.0
- rs20_available_share=0.9902273482644214
- rs30_exact_available=false
- rs30_proxy_available_share=0.9902273482644214
- bias20_percentile_available_share=0.9919766296013388
- volatility_available_share=0.9902273482644214
- ready_for_layer2_soft_score_bounded_diagnostic=true
- ready_for_formal=false
- portfolio_replay_executed=false

## Plain Summary
This package converts Layer2 hard-filter inputs into a diagnostic soft-score feature contract. It keeps capital support, RS windows, stable-strong protection, and risk/overheat context as features only. It does not output a selector, live rule, replay, daily report, or formal model change.

## Blocked / Proxy
- RS30 is proxy only.
- large_down_day_count, blowoff_turnover, and risk_bucket remain blocked.
- BIAS and volatility are diagnostic PIT context only.

## Flags
- formal_model_changed=false
- trade_decision_changed=false
- active_in_trade_decision=false
- report_changed=false
- portfolio_replay_executed=false
- ready_for_strategy_replay=false
- not_live_rule=true
- forward_returns_live_rule_usage=false
