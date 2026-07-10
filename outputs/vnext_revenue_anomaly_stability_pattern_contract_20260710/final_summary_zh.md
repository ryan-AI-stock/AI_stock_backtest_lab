# Revenue anomaly / stability pattern contract

## 結論

- 已把上一版 project/business-model risk 方向修正為純營收時間序列 anomaly/stability pattern。
- 不使用工程、EPC、專案、案場、離岸風電、產業分類或商業模式 keyword 作風險依據。
- 欄位只做 Layer1 candidate hygiene / Layer4 confidence downgrade，不 hard exclude。
- abnormal_revenue_review_flag 只代表營收型態異常或穩定性不足，需要 review / soft penalty。

## 6806 森崴能源 sanity

- status=ready_proxy
- revenue_spike_anomaly_score=0.35294117647058826
- revenue_lumpiness_score=0.2867231662208861
- revenue_concentration_ratio_top1_12m=0.24860061012139467
- revenue_concentration_ratio_top3_12m=0.5703473505709017
- revenue_growth_persistence_score=0.2833333333333333
- revenue_reversion_risk_score=0.15980392156862747
- ttm_vs_recent_growth_gap=0.3687504187518522
- abnormal_revenue_review_flag=True
- 6806 只作營收時間序列 sanity case，不作投資判斷。

## Scoped flags

- abnormal_revenue_review_flag_rows=13
- top flagged sample：8926 台汽電, 5289 宜鼎, 5351 鈺創, 8299 群聯, 3006 晶豪科, 8112 至上, 2885 元大金, 2347 聯強, 4931 新盛力, 6005 群益證

## Blocked / deprecated

- business-model / industry keyword 判斷已降級為 deprecated_not_used。
- 6806 2026-06 monthly revenue 缺月保留 blocked；本任務不需要追單月資料。
- future_data_violation_count=0。