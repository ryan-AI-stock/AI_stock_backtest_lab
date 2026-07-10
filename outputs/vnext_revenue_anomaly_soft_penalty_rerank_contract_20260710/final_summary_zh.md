# Revenue anomaly soft-penalty rerank contract

## 結論

- 已建立 route_support / R6 revenue anomaly soft-penalty rerank contract。
- rerank 使用 Layer4 primary80 每週 PIT topN 候選，不使用 future return。
- revenue anomaly 只作 soft penalty / substitute ranking，不作 standalone alpha、不 hard exclude。
- selected_result_changed_rows=92
- reranked_selected_ohlc_gap_rows=92
- ready_for_experiments=False

## Next

- 若 gap_rows > 0，需先交 Radar/Data 補 reranked selected ticker official OHLC path，再回 Core refresh readiness。
- business_model / industry keyword 已 deprecated，不作風險依據。