# Layer0 investable universe prefilter contract

## Verdict
- status=layer0_investable_universe_prefilter_design_ready_traded_value_ready_market_cap_partial
- recommended_name=Layer0 investable universe / data-pruning filter
- traded_value_prefilter_ready=true
- total_market_traded_value_ready=true
- market_cap_rank_prefilter_ready=false
- hybrid_prefilter_contract_ready=true
- recommended_initial_universe_size=200_to_500_with_buffer
- ready_for_experiments=false
- ready_for_formal=false

## Plain Summary
這應命名為 Layer0 investable universe / data-pruning filter，而不是 Layer1 selector。目的只是先用 PIT 可見的成交金額、流動性與標的類型，把 1900 檔縮成較可補基本面的 200-500 檔級距。

本機資料足以做 traded-value based prefilter 和 total market traded-value share estimate。Market-cap rank 版本仍需要 full daily market cap 或明確接受 proxy，否則只能標 proxy/blocked。

## Flags
- formal_model_changed=false
- trade_decision_changed=false
- active_in_trade_decision=false
- report_changed=false
- portfolio_replay_executed=false
- ready_for_strategy_replay=false
- not_live_rule=true
- forward_returns_live_rule_usage=false
