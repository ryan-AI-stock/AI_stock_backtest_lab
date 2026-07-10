# Revenue anomaly integrated route_support / R6 contract

## 結論

- 已把 revenue anomaly/stability pattern 欄位接到 route_support max1 / R6 unified contract。
- 本輪只新增 soft penalty / confidence downgrade / daily report hook，不改 selected result。
- 不使用 business-model keyword 或 industry classification 作風險依據。
- revenue_anomaly_used_as_hard_exclude=false；hard_exclude_applied=false。

## Readiness

- contract_rows=591
- stock_selected_rows=31
- stock_selected_rows_missing_anomaly_context=0
- ready_for_experiments=True
- selected_stock_adjusted_close_ready_all_rows=False
- cash_bear_classifier_ready_all_rows=False

## Revenue anomaly stock sample

- abnormal selected stock sample：2015-04-30:2316 nan, 2015-05-15:3008 nan, 2017-10-27:6121 nan, 2017-11-24:6573 nan, 2018-03-16:6150 nan, 2021-12-30:8478 nan

## Boundary

- diagnostic / proxy only。
- 不改 formal model、不改 trade decision、不做 replay、不升 daily report production。
- future_data_violation_count=0。