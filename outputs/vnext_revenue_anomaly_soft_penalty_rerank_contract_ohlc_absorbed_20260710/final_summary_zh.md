# Revenue anomaly soft-penalty rerank OHLC absorption

## 結論

- 已吸收 Radar/Data selected-ticker official unadjusted OHLC gap fill。
- radar_filled_rows_absorbed=92
- reranked_selected_ohlc_gap_rows_after_absorption=0
- official_unadjusted_ohlc_ready_share=1.000000
- ready_for_experiments=True

## 邊界

- revenue anomaly 只作 soft penalty / rerank，不作 standalone alpha。
- business-model / industry keyword 未作風險依據。
- hard_exclude_applied=false。
- selected-stock adjusted close 仍 blocked；本包為 official unadjusted OHLC diagnostic readiness。
- 不升 formal / replay / daily report / trade decision。