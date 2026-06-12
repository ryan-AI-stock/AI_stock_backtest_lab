# Risk Factor Data Contract

This project can read optional risk-factor data to annotate stock-pool candidates.
The first version is diagnostic-only: default strategy scores are not changed unless a future preset explicitly sets `risk_signal_weight`.

## Accepted Source Kinds

`stock_pool_observation` accepts explicit CLI/env paths:

- `--institutional-flow-data` / `INSTITUTIONAL_FLOW_DATA_PATH`
- `--margin-short-data` / `MARGIN_SHORT_DATA_PATH`
- `--borrow-lending-data` / `BORROW_LENDING_DATA_PATH`
- `--day-trading-data` / `DAY_TRADING_DATA_PATH`
- `--sentiment-data` / `SENTIMENT_DATA_PATH`

If explicit paths are not provided, the loader searches `RADAR_DATA_DIR` for:

- institutional: `institutional_flows.latest.csv`, `institutional_flow.latest.csv`, `institutional_flows.refreshed.csv`, `stock_metrics.refreshed.csv`
- margin/short: `margin_short.latest.csv`, `margin_short_daily.latest.csv`, `margin_short.refreshed.csv`, `stock_metrics.refreshed.csv`
- borrow/lending: `borrow_lending.latest.csv`, `securities_lending.latest.csv`, `short_lending.latest.csv`, `stock_metrics.refreshed.csv`
- day trading: `day_trading.latest.csv`, `day_trading_daily.latest.csv`, `day_trading.refreshed.csv`, `stock_metrics.refreshed.csv`
- sentiment: `sentiment.latest.csv`, `social_sentiment.latest.csv`, `sentiment.refreshed.csv`, `stock_metrics.refreshed.csv`

## Required Shared Columns

One ticker identifier:

- `ticker`, such as `2454.TW`
- or `symbol`, such as `2454`; optional `exchange` / `suffix` defaults to `TW`

Optional date column:

- `date`
- `report_date`
- `source_date`

Rows after the signal date are ignored. If multiple rows exist for the same ticker, the latest row not after the signal date is used.

## Institutional Columns

Preferred:

- `foreign_net_buy_shares`
- `investment_trust_net_buy_shares`
- `dealer_net_buy_shares`
- `foreign_consecutive_sell_days`
- `trust_consecutive_sell_days`

Fallback aliases:

- `foreign_5d`
- `trust_5d`

## Margin / Short Columns

Preferred:

- `margin_balance_5d_change_pct`
- `margin_balance_20d_change_pct`
- `short_balance_5d_change_pct`
- `margin_overheat_flag`
- `short_lending_pressure_flag`

Fallback aliases:

- `margin_change_5d`
- `margin_change_20d`

## Borrow / Lending Columns

Accepted:

- `borrow_balance_5d_change_pct`
- `securities_lending_5d_change_pct`
- `short_lending_5d_change_pct`
- `borrow_sell_ratio`
- `securities_lending_sell_ratio`
- `short_lending_ratio`
- `borrow_pressure_flag`
- `short_lending_pressure_flag`

## Day Trading Columns

Accepted:

- `day_trading_volume_ratio`
- `day_trading_ratio`
- `day_trading_ratio_5d_avg`
- `day_trading_overheat_flag`

## Sentiment Columns

Accepted:

- `sentiment_score`
- `social_sentiment_score`
- `social_heat_score`
- `message_heat_score`
- `sentiment_heat`
- `sentiment_overheat_flag`

## Candidate Output Fields

Candidate CSV/JSON outputs include:

- `flow_risk_score`
- `institutional_risk`
- `margin_risk`
- `borrow_risk`
- `day_trading_risk`
- `sentiment_risk`
- `bullish_flow_score`
- `sentiment_score`
- `flow_score_adjustment`
- `flow_risk_reasons`
- `flow_source_dates`
- `flow_source_kinds`

`flow_score_adjustment` is only applied when the selected strategy parameters set `risk_signal_weight` above zero.
