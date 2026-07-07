# Layer1 t164sb05/t164sb03 Parser Contract Design

Status: parser_contract_design_ready_not_full_ingest

Boundary: parser contract design only; no full ingest, no Experiments, no replay, no formal/report/trade change.

Readiness:
- ready_for_core_parser_contract_design=true
- ready_for_core_layer1_cashflow_inventory_receivable_ingest=false
- ready_for_core_rerun=false
- ready_for_experiments=false
- ready_for_formal=false
- ready_for_strategy_replay=false
- future_data_violation_count=0
- not_live_rule=true

Blocked prerequisites:
- MOPS_disclosure_datetime_asof_join: browser latest page alone is not sufficient for historical PIT
- direct_browser_equivalent_payload_replay: direct API minimal payload returns code=500
- TPEx_sample_confirmation: multi-sample automation unstable
- full_universe_ingest_runner: only 1101 latest browser sample exists
- label_taxonomy_policy: capex/receivable labels vary by company/profile
- direct_api_replay_without_browser_context: missing SPA/browser-equivalent request context or full payload transformation
- TPEx multi-sample t164sb05/t164sb03: bounded browser control instability, not evidence that official route lacks TPEx support
- legacy_ajax_security_block: legacy ajax routes security protected; no bypass attempted

Next handoff:
- Radar/Data should capture browser-equivalent payload/context, MOPS disclosure/asof join, and stable TPEx samples before Core full ingest.

Flags:
- formal_model_changed=false
- trade_decision_changed=false
- active_in_trade_decision=false
- report_changed=false
- portfolio_replay_executed=false
- ready_for_strategy_replay=false