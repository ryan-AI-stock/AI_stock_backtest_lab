# Layer1 t164 Bounded Broader Ingest Contract

Status: bounded_broader_ingest_contract_built_not_materialized

Conclusion: Core built a bounded broader ingest contract from the pruning v2 seed. This is contract readiness only, not broader materialization, not Experiments-ready, and not formal-ready.

Readiness:
- ready_for_core_t164_bounded_broader_materialization=true
- ready_for_experiments=false
- ready_for_formal=false
- ready_for_strategy_replay=false
- ready_for_full_universe=false
- sample_rows=40
- ticker_count=20
- period_count=2
- official_asof_matched_rows=40
- future_data_violation_count=0

Retained blockers:
- TPEx all-stock proof not complete
- full period range not complete
- full universe false
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