# Layer1 t164 Asof Alternate Route Patch Readiness

Status: partial_patch_ready_remaining_blocked_not_experiments_ready

Conclusion: Core accepts five strict official t05st01 alternate-route timestamps as a diagnostic patch, but the 40-row sample remains not Experiments-ready because one row is still blocked.

Readiness:
- patch_accepted_rows=5
- remaining_blocked_rows=1
- official_timestamp_matched_rows_after_patch=39
- official_timestamp_matched_share_after_patch=0.975
- ready_for_core_t164_asof_join_contract_refresh=false
- ready_for_core_t164_full_or_broader_ingest_contract=false
- ready_for_experiments=false
- ready_for_formal=false
- ready_for_strategy_replay=false
- future_data_violation_count=0

Remaining blocker:
- 6187 TPEx 114Q4 remains blocked_multiple_strict_candidates; no premeeting notice, query time, quarter end, or deadline proxy backfill.

Next step:
- Radar/Data should provide stricter detail/subject disambiguation for 6187 TPEx 114Q4, or Research/Strategy can accept matched-only partial policy review without Experiments.

Flags:
- formal_model_changed=false
- trade_decision_changed=false
- active_in_trade_decision=false
- report_changed=false
- portfolio_replay_executed=false
- ready_for_strategy_replay=false
- not_live_rule=true
- forward_returns_live_rule_usage=false