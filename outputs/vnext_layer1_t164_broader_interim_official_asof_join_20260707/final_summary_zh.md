# Layer1 t164 Broader Interim Official-Asof Join Readiness

Status: broader_interim_official_asof_join_ready_not_full_ingest

Boundary: bounded broader sample readiness only; no full ingest, no Experiments, no replay, no formal/report/trade change.

Readiness:
- ready_for_layer1_t164_broader_interim_official_asof_event_diagnostic=true
- ready_for_full_ingest=false
- ready_for_experiments=false
- ready_for_formal=false
- ready_for_strategy_replay=false
- official_timestamp_matched_share=1.0
- unmatched_share=0.0
- ticker_count=4
- market_count=2
- period_count=2
- after_close_policy_applied_count=8
- future_data_violation_count=0

Blocked / proxy fields:
- full_universe_runner: blocked; bounded broader sample only; no all-stock runner
- tpex_universal_ready: blocked; TPEx has bounded samples only, not universal readiness
- capex_proxy: human_review_required; FCF proxy label policy still needs review
- receivables_basket: human_review_required; receivables basket policy still needs review
- exact_upload_timestamp: not_found; market_available_at is public announcement timestamp, not internal upload timestamp
- conservative_asof_backfill: prohibited_for_matched_rows; official timestamp matched rows must not be backfilled by deadline proxy
- formal_selector: prohibited; no Layer1 selector created
- full_universe_materialization_runner: blocked_not_started; this task is bounded readiness; no full download/full sweep
- tpex_universal_ready: blocked_bounded_only; TPEx 6488/8299 samples pass but not all-stock coverage
- quarter_end_date: prohibited; quarter end precedes disclosure
- query_response_datetime: prohibited; API response datetime is query time
- capex_proxy_and_receivables_basket: human_review_required; label variants and basket definition require policy approval

Flags:
- formal_model_changed=false
- trade_decision_changed=false
- active_in_trade_decision=false
- report_changed=false
- portfolio_replay_executed=false
- ready_for_strategy_replay=false
- not_live_rule=true
- forward_returns_live_rule_usage=false