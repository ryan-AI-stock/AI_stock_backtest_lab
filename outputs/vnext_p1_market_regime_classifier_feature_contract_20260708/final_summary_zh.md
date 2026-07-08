# P1 market regime classifier feature contract

結論：本包已建立 P1 market regime classifier 的 feature/readiness contract，可交 Experiments 做 bounded diagnostic。

- signal rows: 411
- 0050 market regime features：BIAS20/40/60/120、MA above/below、20D/40D/60D return/slope、rolling-high breakout/drawdown 已 materialized。
- 00631L context：drawdown、volatility、high-risk candidate context 已 materialized，但只作 diagnostic context。
- dynamic80 pool features：RS20/60 breadth、median、dispersion、opportunity label share 與 traded-value proxy 已對齊 signal dates。
- consensus4 exception alignment：是否有 exception、ticker、連續 signal count、transition context 已對齊 P1 state-machine。
- blocked/proxy：pool RS40 exact、top10/top20 turnover churn、top10/top20 traded-value share、cash/bear classifier 仍不可宣稱 ready。
- trend state labels 是候選 label，不是 live rule；Core 不決定 threshold。
- Strategy Center 新成本規則已寫入 audit：後續回測主要結論必須含手續費、證交稅、ETF/股票成本差異與 transition cost；no-cost 只能當 secondary gross reference。

下一棒建議：交 Experiments 執行 TASK-BACKTEST-EXPERIMENTS-VNEXT-P1-MARKET-REGIME-CLASSIFIER-DIAGNOSTIC-001。

Flags: formal_model_changed=false; trade_decision_changed=false; active_in_trade_decision=false; report_changed=false; portfolio_replay_executed=false; ready_for_strategy_replay=false; ready_for_formal=false; not_live_rule=true; forward_returns_live_rule_usage=false.

完成後如果下一棒明確，請直接指派下一個 thread；如果下一棒不明確，請回報 Strategy Center 判斷。不要完成後停住不回報。