# Legacy RS20 exact path / timing / cost materialization

- status: `legacy_rs20_exact_path_timing_cost_blocked_missing_selected_stock_price_path`
- primary_variant: `dynamic80_top3_rs20_risk_tiebreak_proxy`
- selected_signal_rows: 1776
- selected_unique_ticker_count: 388
- exact_selected_stock_adjusted_close_path_ready: false
- next_trading_day_close_path_ready: false
- next_trading_day_open_path_ready: false
- local_ep05_cost_model_found: true
- formal_cost_model_ready: false
- ready_for_experiments: false

## 判斷

本輪已把 Legacy RS20 selected-stock exact path 的缺口 materialize 成可稽核 contract。本機找到 EP05 台股費稅模型，公式與參數可用；但 selected ticker 的 adjusted close / executable path 在本機 price registry 中沒有足夠覆蓋，因此無法產出 exact selected-stock gross/net return。

這份 package 不再使用 `00631L_forward_return_5d + forward_excess_vs_00631L_5d` 重建個股報酬。所有 exact trade path row 均只允許 selected ticker price path；缺價時明確 blocked。

## Blockers

- full selected-stock adjusted close path missing for dynamic80 RS20 selected tickers.
- next-day close/open entry timing cannot be exact until selected ticker price path is available.
- formal EP05 fee/tax model is found, but numeric formal-cost return needs entry/exit price and notional; currently blocked by missing exact price path.

## 下一棒

readiness 尚未通過 exact Experiments diagnostic。下一步應交 Radar/Data 或 Core source path owner 補 selected ticker adjusted close / open-close path，範圍先限 Legacy RS20 selected tickers 與 2024-01-02~2026-05-26 加 exit buffer，不要回到全市場 mass download。

## Flags

- formal_model_changed=false
- trade_decision_changed=false
- active_in_trade_decision=false
- report_changed=false
- portfolio_replay_executed=false
- ready_for_strategy_replay=false
- not_live_rule=true
- forward_returns_live_rule_usage=false
- diagnostic_only=true
