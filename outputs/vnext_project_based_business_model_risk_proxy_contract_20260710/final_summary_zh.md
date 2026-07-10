# Project-based business-model risk proxy contract

## 結論

- 已建立 project/business-model risk proxy contract；這是 diagnostic/proxy，不是正式商業模式分類器。
- 本輪不做 hard exclude，只提供 Layer1/Layer4 review flag、soft penalty、report text hook。
- 文字 keyword source 目前只有 bounded 6806 MOPS monthly revenue notes；latest primary80 多數仍是 monthly-shape-only proxy。
- 若只有 keyword 命中但長期穩定性佳，僅低信心提醒；若 keyword + 高 lumpiness + 低 stability 同時命中，才提高風險提醒。

## 6806 森崴能源 sanity

- status=ready_proxy
- project_risk_review_flag=True
- project_revenue_business_model_proxy=True
- proxy_confidence_level=medium_keyword_plus_revenue_shape
- revenue_lumpiness_percentile_vs_primary80=0.9625
- revenue_stability_percentile_vs_primary80=0.2125
- matched_keywords=工程;離岸;認列收入
- 6806 不在 latest Layer4 primary80；只作 sanity case，不作投資判斷。

## Scoped flags

- project_risk_review_flag_rows=8
- top flagged sample：8926 台汽電, 5289 宜鼎, 5351 鈺創, 6806 森崴能源, 3006 晶豪科, 4931 新盛力, 2882 國泰金, 2347 聯強

## Blocked / proxy

- full business model truth detector blocked。
- latest primary80 company description / annual report text source blocked/partial。
- 6806 2026-06 monthly revenue 缺月保留 blocked；本任務不需要追單月資料。
- future_data_violation_count=0。