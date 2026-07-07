# Layer1 t164 Broader Source Package Review

Status: source_package_review_completed_pruning_v2_required

Conclusion: the broader seed package passed bounded route/asof hygiene, but Core should not build a broader ingest contract yet. The route fan-out is too high for full universe without candidate/detail pruning.

Readiness:
- sample_rows=40
- statement_success_rows=40
- official_asof_matched_rows=40
- cache_manifest_rows=1683
- seed_cache_rows_per_materialized_row=42.075
- ready_for_bounded_broader_ingest_contract=false
- ready_for_core_t164_broader_ingest_contract=false
- ready_for_core_t164_broader_materialization=false
- ready_for_radar_candidate_detail_pruning_runner_v2=true
- ready_for_experiments=false
- ready_for_formal=false
- ready_for_strategy_replay=false
- future_data_violation_count=0

Next step: Radar/Data should build candidate/detail pruning runner v2 before any all-stock or broader-period materialization.

Flags:
- formal_model_changed=false
- trade_decision_changed=false
- active_in_trade_decision=false
- report_changed=false
- portfolio_replay_executed=false
- ready_for_strategy_replay=false
- not_live_rule=true
- forward_returns_live_rule_usage=false