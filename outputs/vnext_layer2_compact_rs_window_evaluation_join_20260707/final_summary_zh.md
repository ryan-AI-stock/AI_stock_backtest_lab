# Layer2 compact RS-window evaluation join

## Verdict
- status=layer2_compact_rs_window_evaluation_join_ready_for_experiments_intake
- rows=171499
- weekly_snapshot_count=592
- unique_ticker_count=1612
- rs20_available_share=0.9902273482644214
- rs60_available_share=0.9719007107913166
- rs30_exact_available=false
- rs30_proxy_available_share=0.9902273482644214
- forward_eval_available_share_20d=0.9856617239750669
- blocked_evaluation_rows=9500
- ready_for_layer2_compact_rs_capital_interaction_diagnostic=true
- ready_for_experiments_intake=true
- ready_for_formal=false
- portfolio_replay_executed=false

## Plain Summary
This package adds PIT RS5/10/20/40/60, RS30 proxy, short-window acceleration/deterioration, and RS60-high short-RS weakening context to the compact Layer1 evaluation join. RS30 is explicitly proxy. Forward returns remain evaluation_metadata_only and are not rule inputs.

## Flags
- formal_model_changed=false
- trade_decision_changed=false
- active_in_trade_decision=false
- report_changed=false
- portfolio_replay_executed=false
- ready_for_strategy_replay=false
- not_live_rule=true
- forward_returns_live_rule_usage=false
