# Layer1 t164/t05st01 Interim Official-Asof Join Readiness

Status: interim_official_asof_join_sample_ready_not_full_ingest

Boundary: sample/interim official-asof join readiness only; no full ingest, no Experiments, no replay, no formal/report/trade change.

Readiness:
- ready_for_layer1_t164_interim_official_asof_event_diagnostic=true
- ready_for_full_ingest=false
- ready_for_experiments=false
- ready_for_formal=false
- ready_for_strategy_replay=false
- official_timestamp_matched_share=1.0
- unmatched_share=0.0
- after_close_policy_applied_count=9
- future_data_violation_count=0

Blocked / proxy fields:
- unmatched_official_timestamp_rows: blocked; unmatched rows must remain blocked or separate conservative-asof candidates
- conservative_filing_deadline_proxy: separate_candidate_only; do not silently backfill official timestamp
- capex_proxy: human_policy_required; FCF proxy label/policy requires human review
- receivables_basket: human_policy_required; receivables basket policy requires human review
- tpex_universal_ready: blocked; bounded sample confirmation only; not universal ready
- full_ingest: blocked; this package is sample/interim readiness only
- formal_selector: prohibited; no Layer1 selector created

Flags:
- formal_model_changed=false
- trade_decision_changed=false
- active_in_trade_decision=false
- report_changed=false
- portfolio_replay_executed=false
- ready_for_strategy_replay=false
- not_live_rule=true
- forward_returns_live_rule_usage=false