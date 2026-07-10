# Layer1 revenue stability hygiene integration contract

## 結論

- 已把長期營收穩定性接成 Layer1/Layer4 hygiene integration contract。
- revenue_stability_score 進 Layer1 quality context / soft adjustment。
- revenue_lumpiness_score 進 Layer1/Layer4 risk context / soft penalty。
- recent_spike_without_long_history 與 project_based_revenue_risk_proxy 只作 review/report flag。
- low-base 不回主權重，只作 context / tie-break cap / overheat penalty modifier。
- 本輪沒有改 route_support selected result，沒有 hard exclude。

## 高風險 proxy 名單

- review_soft_penalty_candidate rows=5
- Strategy Center 指定名單 present/flagged：5351 鈺創, 3006 晶豪科, 8926 台汽電, 4931 新盛力, 2347 聯強
- 這些只作 review / soft penalty candidate，不得直接踢出。

## Blocked / proxy

- PE/PB/PS valuation low-base blocked。
- exact quarterly revenue YoY blocked；目前只能用 monthly rolling 3M proxy。
- margin recovery exact source blocked/partial。
- project_based_revenue_risk_proxy 不是正式商業模式辨識。
- 6806 森崴能源 source 仍等 Radar/Data bounded source，未在本任務等待。