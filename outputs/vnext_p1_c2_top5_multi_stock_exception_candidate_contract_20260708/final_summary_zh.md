# P1 C2 top5 multi-stock exception candidate contract

- task_id: `TASK-BACKTEST-CORE-VNEXT-P1-C2-TOP5-MULTI-STOCK-EXCEPTION-CANDIDATE-CONTRACT-001`
- status: `p1_c2_top5_exception_candidate_contract_ready_path_partial_blocked`
- candidate_rows: 2055
- c2_true_candidate_path_required_rows: 1020
- official_ohlc_ready_rows: 61
- official_ohlc_blocked_rows: 959
- candidate ranks use Layer4 80 primary pool PIT pool_rank / pool_selection_score. Existing consensus4 exact top2~top5 is blocked, so this is Layer4 high-confidence/risk-aware top5 exception candidate contract.
- transition cost design uses EP05 TaiwanCostModel with ETF/stock tax split and equal-weight sleeve cost basis for max1~max5.
- adjusted close remains blocked; official unadjusted OHLC is diagnostic-only where available.
- Not ready for Experiments rerun until Radar/Data fills missing official OHLC path rows for C2-true top5 candidates.

下一棒：交 Radar/Data 做 selected-ticker-only top5 candidate official OHLC source fill，不做 full-market mass download。

Flags: formal_model_changed=false; trade_decision_changed=false; active_in_trade_decision=false; report_changed=false; portfolio_replay_executed=false; ready_for_strategy_replay=false; ready_for_formal=false; not_live_rule=true; forward_returns_live_rule_usage=false.

完成後如果下一棒明確，請直接指派下一個 thread；如果下一棒不明確，請回報 Strategy Center 判斷。不要完成後停住不回報。