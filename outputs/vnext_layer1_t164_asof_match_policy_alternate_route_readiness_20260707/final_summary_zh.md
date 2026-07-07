# Layer1 t164 Asof Match Policy / Alternate Route Readiness

Status: blocked_by_official_asof_match_gaps

Conclusion: t164 statement replay is stable in the larger bounded sample, but official-asof match coverage is incomplete. Core does not accept the package as broader ingest-ready or Experiments-ready.

Readiness:
- ready_for_core_t164_broader_interim_official_asof_join=false
- ready_for_experiments=false
- ready_for_formal=false
- ready_for_strategy_replay=false
- official_timestamp_matched_share=0.85
- matched_rows=34
- unmatched_rows=6
- future_data_violation_count=0

Unmatched failure attribution:
- 3008 TWSE 115Q1: t05st01_query_returned_no_financial_report_candidate
- 3008 TWSE 114Q4: t05st01_query_returned_no_financial_report_candidate
- 6669 TWSE 115Q1: t05st01_query_returned_no_financial_report_candidate
- 6669 TWSE 114Q4: t05st01_query_returned_no_financial_report_candidate
- 6187 TPEx 115Q1: t05st01_query_returned_no_financial_report_candidate
- 6187 TPEx 114Q4: t05st01_query_returned_no_financial_report_candidate

Alternate route / policy staging:
- current_t05st01_t05st01_detail_exact_subject_match: partial_blocked_by_unmatched_rows; can_fill_current_unmatched_rows=false
- broaden_t05st01_query_and_subject_policy: candidate_not_validated; can_fill_current_unmatched_rows=false
- conservative_filing_deadline_proxy: separate_proxy_candidate_only; can_fill_current_unmatched_rows=false
- quarter_end_date_or_query_response_datetime: prohibited; can_fill_current_unmatched_rows=false

Blocked rows policy:
- matched-only is acceptable only as a partial contract for policy review.
- unmatched rows remain blocked; no silent backfill.
- conservative filing-deadline proxy must remain separate and cannot replace official timestamp.
- quarter_end_date and query_response_datetime remain prohibited.

Flags:
- formal_model_changed=false
- trade_decision_changed=false
- active_in_trade_decision=false
- report_changed=false
- portfolio_replay_executed=false
- ready_for_strategy_replay=false
- not_live_rule=true
- forward_returns_live_rule_usage=false