# vNext Layer1 Fuller PIT Contract Readiness

Status: partial_ready_layer1_fuller_pit_contract_proxy_limited

Boundary: Layer1 quality-floor / eligibility-filter contract only; no Experiments, no replay, no formal/report/trade change.

Readiness:
- ready_for_funnel_layer1_fuller_quality_floor_diagnostic=true
- layer1_fuller_exact_coverage=partial
- layer1_source_upgrade_vs_refreshed=material
- ready_for_layer2_diagnostic=false
- ready_for_portfolio_like_diagnostic=false
- ready_for_strategy_replay=false
- ready_for_formal=false
- future_data_violation_count=0
- not_live_rule=true
- forward_returns_live_rule_usage=false

Blocked / proxy fields:
- monthly_revenue_growth: PIT-ready; MOPS monthly revenue as-of available_date; conservative timing
- quarterly_growth_margin_profitability: PIT-ready; quarterly fundamentals as-of available_date; exact filing timestamp unavailable
- tpex_market_cap_proxy: proxy; TPEx daily market cap proxy from official close * issued shares; TWSE exact daily market cap still blocked
- twse_capital_stock_shares_proxy: proxy; TWSE capital stock/shares quarterly proxy; not daily exact market cap
- debt_leverage_solvency: blocked; Radar route says derivable, but accepted quarterly total_assets/total_liabilities/equity are empty locally
- listing_board_status_partial_proxy: proxy; partial event/status source; master_ready=false
- twse_industry_diagnostic_proxy: proxy; TWSE official industry diagnostic route noted; no accepted PIT industry rows materialized in Core package
- average_traded_value: proxy; existing attention_features signal-date traded_value
- turnover: proxy; existing attention_features turnover windows
- free_float_market_cap: blocked; no local free-float shares/free-float cap route
- operating_cash_flow_quality: blocked; cash-flow statement full sweep not materialized
- free_cash_flow_quality: blocked; OCF/capex fields unavailable; do not proxy from profitability
- current_ratio: blocked; current assets/current liabilities not materialized
- inventory_risk: blocked; inventory detail not materialized
- receivable_risk: blocked; receivable detail not materialized
- tpex_all_stock_sector_pit: blocked; accepted TPEx sector rows=0
- twse_exact_daily_market_cap: blocked; TWSE direct daily market cap / issued shares route remains blocked
- forward_return_as_rule: prohibited; forward returns prohibited as Layer 1 rule inputs

Next handoff:
- vNext Research should judge whether this partial fuller package is enough to ask Experiments for Layer1 quality-floor diagnostic.
- Do not start Layer2 until Layer1 fuller diagnostic receives Research / Strategy Center GO.

Flags:
- formal_model_changed=false
- trade_decision_changed=false
- active_in_trade_decision=false
- report_changed=false
- portfolio_replay_executed=false
- ready_for_strategy_replay=false