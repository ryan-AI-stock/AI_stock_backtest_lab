# vNext Funnel Layer 1 Refreshed PIT Contract Readiness

Status: partial_ready_layer1_refreshed_pit_contract_proxy_limited
Merged task ids: TASK-BACKTEST-CORE-VNEXT-FUNNEL-LAYER1-REFRESHED-PIT-CONTRACT-FROM-RADAR-SOURCE-INVENTORY-001, TASK-BACKTEST-CORE-VNEXT-FUNNEL-LAYER1-REFRESHED-PIT-CONTRACT-READINESS-001

Boundary: source/contract readiness only; no Experiments replay, no selector, no formal/report/trade change.

Readiness:
- ready_for_funnel_layer1_refreshed_event_diagnostic=true
- ready_for_funnel_layer1_candidate_pool_quality_diagnostic=true
- layer1_refreshed_exact_coverage=partial
- layer1_source_upgrade_vs_previous=material
- ready_for_layer2_diagnostic=false
- ready_for_portfolio_like_diagnostic=false
- ready_for_strategy_replay=false
- ready_for_formal=false
- future_data_violation_count=0
- not_live_rule=true
- forward_returns_live_rule_usage=false

Blocked / proxy fields:
- monthly_revenue_yoy_mom_rolling3m: diagnostic; MOPS monthly revenue full-universe; conservative available_date; exact filing timestamp unavailable
- quarterly_revenue_eps_operating_income_growth: diagnostic; quarterly fundamentals full sweep; growth computed from past quarters only
- gross_operating_margin: diagnostic; quarterly fundamentals as-of join
- roe_roa: partial; quarterly fundamentals where profile supplies total assets/equity/ROE
- debt_to_equity_proxy: proxy; balance-sheet proxy; not current ratio or full leverage policy
- paid_in_capital_issued_shares_proxy: proxy; TWSE capital stock proxy; TPEx missing and no daily exact issued shares
- listing_status_partial_event: proxy; partial accepted listing events only; not full listing master
- listing_board_proxy: proxy; derived from MOPS market/source table and partial events
- average_traded_value: proxy; existing attention_features signal-date traded_value
- turnover: proxy; existing attention_features turnover windows
- industry_sector: blocked; industry/sector remains blocked/proxy; TPEx all-stock historical route not accepted
- market_cap_free_float_market_cap: blocked; full market cap and free-float market cap not materialized
- cash_flow_quality: blocked; cash-flow quality contract not present
- current_ratio: blocked; current ratio contract not present
- inventory_receivable_risk: blocked; inventory/receivable risk contract not present
- forward_return_as_rule: prohibited; forward returns are prohibited as Layer 1 rule inputs

Next handoff:
- vNext Research should judge whether this partial refreshed contract is enough for a bounded Layer 1 candidate-pool-quality diagnostic.
- Strategy Center should keep Layer 2 disabled until Layer 1 diagnostic receives a Research GO.

Flags:
- formal_model_changed=false
- trade_decision_changed=false
- active_in_trade_decision=false
- report_changed=false
- portfolio_replay_executed=false
- ready_for_strategy_replay=false