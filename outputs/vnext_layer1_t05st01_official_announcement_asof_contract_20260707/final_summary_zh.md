# Layer1 t05st01 Official Announcement Asof Contract

Status: official_announcement_asof_contract_ready_sample_design_not_full_ingest

Boundary: asof contract/sample design only; no full ingest, no Experiments, no replay, no formal/report/trade change.

Readiness:
- ready_for_core_official_announcement_timestamp_asof_contract=true
- ready_for_core_exact_filing_asof_join=false
- ready_for_full_ingest=false
- ready_for_experiments=false
- ready_for_formal=false
- ready_for_strategy_replay=false
- market_available_at_source=t05st01_public_material_information_announcement_timestamp
- exact_internal_filing_upload_timestamp_found=false
- after_close_next_trading_day_policy_required=true
- future_data_violation_count=0

Blocked / prohibited fields:
- exact_internal_filing_upload_timestamp: blocked; No route found; Strategy accepts public announcement timestamp as market_available_at but internal upload remains false.
- api_response_datetime: prohibited; Query-time response datetime is not historical availability.
- quarter_end_date: prohibited; Quarter-end precedes disclosure; forbidden as available_date.
- conservative_filing_deadline_proxy: superseded_for_matched_rows; Do not replace official timestamp when t05st01 match exists.
- date_only_announcement: insufficient; Use 發言日期+發言時間, not date-only.
- internal_exact_filing_upload_timestamp: blocked; formal exact filing timestamp remains blocked if policy requires upload timestamp rather than public announcement timestamp
- date_only_announcement_route: superseded_by_timestamp_route; date-only official route not needed for sampled material-information path
- query_datetime: forbidden; must not be used as PIT available_at
- quarter_end_date: forbidden; must not be used as PIT available_at

Flags:
- formal_model_changed=false
- trade_decision_changed=false
- active_in_trade_decision=false
- report_changed=false
- portfolio_replay_executed=false
- ready_for_strategy_replay=false
- not_live_rule=true
- forward_returns_live_rule_usage=false