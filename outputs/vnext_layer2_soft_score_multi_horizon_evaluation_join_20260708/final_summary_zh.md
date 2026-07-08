# Layer2 soft-score multi-horizon evaluation join

## Verdict
- status=layer2_soft_score_multi_horizon_evaluation_join_ready_for_experiments_intake
- rows=171499
- weekly_snapshot_count=592
- unique_ticker_count=1612
- forward_eval_available_share_5d=0.9911719601863568
- forward_eval_available_share_10d=0.9891369628977429
- forward_eval_available_share_20d=0.9856617239750669
- forward_eval_available_share_30d=0.982081528172176
- forward_eval_available_share_40d=0.9786354439384486
- large_down_day_available=true
- large_down_day_source_quality=diagnostic_price_proxy_threshold_not_formal
- blowoff_turnover_available=true
- blowoff_turnover_source_quality=diagnostic_traded_value_proxy_threshold_not_formal
- risk_bucket_available=false
- ready_for_layer2_soft_score_multi_horizon_risk_diagnostic=true

## Plain Summary
This package adds exact 5D/10D/20D/30D/40D evaluation metadata versus 0050 and 00631L. The 30D horizon is computed directly from adjusted/available close and is not a 20D/40D proxy. Multi-horizon shape fields are evaluation metadata only. Large-down-day and blowoff-turnover are diagnostic proxies, while risk_bucket remains blocked.

## Flags
- formal_model_changed=false
- trade_decision_changed=false
- active_in_trade_decision=false
- report_changed=false
- portfolio_replay_executed=false
- ready_for_strategy_replay=false
- not_live_rule=true
- forward_returns_live_rule_usage=false
