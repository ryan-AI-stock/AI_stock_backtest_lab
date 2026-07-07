# Layer1 compact candidate-quality evaluation join

## Verdict
- status=layer1_compact_candidate_quality_evaluation_join_ready_for_experiments_intake
- rows=171499
- weekly_snapshot_count=592
- unique_ticker_count=1612
- forward_eval_available_share_20d=0.9856617239750669
- blocked_evaluation_rows=9500
- ready_for_layer1_compact_candidate_quality_diagnostic=true
- ready_for_experiments_intake=true
- ready_for_formal=false
- portfolio_replay_executed=false

## Plain Summary
This package adds 5D/10D/20D/40D forward excess return metadata versus 0050 and 00631L for compact Layer1 candidate-quality evaluation. Forward returns are explicitly evaluation_metadata_only and not rule inputs. Latest rows without enough future trading path are listed in the blocked ledger.

## Flags
- formal_model_changed=false
- trade_decision_changed=false
- active_in_trade_decision=false
- report_changed=false
- portfolio_replay_executed=false
- ready_for_strategy_replay=false
- not_live_rule=true
- forward_returns_live_rule_usage=false
