# Layer1 reduced-universe source planning from Layer0

## Verdict
- status=layer1_reduced_universe_source_planning_ready_existing_low_cost_fields_first
- layer0_variant=top300_buffer100
- weekly_snapshot_count=592
- event_rows=236800
- unique_ticker_count_all=1843
- average_weekly_ticker_count=400.0
- average_turnover_share_5d=0.9339828252604127
- all_period_unique_ticker_scope_warning=top300_buffer100 rolling universe touches 1843 names across full history; source acquisition should be period-scoped, not all-history unique-scoped
- ready_for_layer1_reduced_universe_interim_contract=true
- ready_for_t164_mass_download=false
- ready_for_experiments=false
- ready_for_formal=false

## Plain Summary
Layer0 top300_buffer100 turns the full market into a weekly universe of about 400 names on average. Across the full 2015-2026 rolling history, those weekly lists still touch many tickers, so the cost-saving policy must be period-scoped source acquisition rather than all-history unique-ticker acquisition. Layer1 should first use existing/low-cost monthly revenue, quarterly profitability/margins/EPS, liquidity, and listing status fields to build a quality-floor contract. High-cost t164 cashflow/current-ratio/inventory/receivables should only be scoped to active Layer0 passers after the interim contract is accepted.

## Flags
- formal_model_changed=false
- trade_decision_changed=false
- active_in_trade_decision=false
- report_changed=false
- portfolio_replay_executed=false
- ready_for_strategy_replay=false
- not_live_rule=true
- forward_returns_live_rule_usage=false
