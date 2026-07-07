# Layer1 Current Ratio Only PIT Contract Readiness

Status: sample_ready_current_ratio_only_contract_design_not_full_ingest

Boundary: current_ratio-only sample-backed contract design; no full Layer1 ingest, no Experiments, no replay, no formal/report/trade change.

Readiness:
- ready_for_current_ratio_contract_design_review=true
- ready_for_layer1_current_ratio_full_diagnostic=false
- ready_for_layer1_remaining_parser_ingest=false
- ready_for_merge_with_layer1_fuller_interim_diagnostic=false
- current_ratio_contract_scope=sample_only_not_full_universe
- sample_contract_rows=36
- candidate_join_current_ratio_available_rows=68
- future_data_violation_count=0

Blocked fields kept blocked:
- inventory_risk: blocked; not in standard t163sb05 summary sample
- receivable_risk: blocked; profile-specific or missing; no universal parser
- operating_cash_flow_quality: blocked; cash-flow route sample missing
- free_cash_flow_quality: blocked; depends on operating cash flow and capex fields
- free_float_market_cap: blocked; outside parser scope; no local official free-float route
- exact_market_cap: blocked; TWSE exact daily market cap still blocked
- full_sector_pit: blocked; TPEx all-stock sector PIT remains blocked
- forward_return_as_rule: prohibited; forward returns prohibited

Next handoff:
- vNext Research should decide whether to merge this current_ratio-only design with Layer1 fuller interim diagnostic, or wait for cash-flow/inventory/receivable unlock.

Flags:
- formal_model_changed=false
- trade_decision_changed=false
- active_in_trade_decision=false
- report_changed=false
- portfolio_replay_executed=false
- ready_for_strategy_replay=false
- not_live_rule=true
- forward_returns_live_rule_usage=false