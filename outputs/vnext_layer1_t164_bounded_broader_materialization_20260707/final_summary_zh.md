# Layer1 t164 Bounded Broader Materialization

Status: bounded_broader_materialized_not_experiments_ready

Conclusion: Core materialized the bounded broader t164 source table from the approved contract. This remains source materialization only, not full universe, not Experiments-ready, and not formal-ready.

Readiness:
- ready_for_layer1_t164_bounded_interim_diagnostic_planning=true
- ready_for_core_t164_broader_or_full_ingest_next=false
- ready_for_experiments=false
- ready_for_formal=false
- ready_for_strategy_replay=false
- sample_rows=40
- ticker_count=20
- period_count=2
- statement_success_rows=40
- official_asof_matched_rows=40
- blocked_rows=0
- future_data_violation_count=0

Retained caveats:
- bounded materialization only, not full universe
- TPEx all-stock proof not complete
- full period range not complete
- capex_proxy / receivables_trade human-review proxy policy required

Flags:
- formal_model_changed=false
- trade_decision_changed=false
- active_in_trade_decision=false
- report_changed=false
- portfolio_replay_executed=false
- ready_for_strategy_replay=false
- not_live_rule=true
- forward_returns_live_rule_usage=false