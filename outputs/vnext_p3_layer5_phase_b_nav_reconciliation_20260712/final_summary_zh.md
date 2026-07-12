# P3 Phase B NAV reconciliation

先前 Stage B A 績效因 NAV accounting 與 corporate-action scale 錯誤全數作廢，策略 NO_GO 含義已撤銷。

修正後以固定 portfolio NAV 換股：舊股仅使用自身 event-aware adjusted mark 計當日報酬，再扣 exit/entry EP05 與 10bp/side 滑價，最後以 after-cost NAV / 新股官方 raw entry close 重設股數。跨 ticker 名目價格報酬使用次數=0。

2025-08-01 缺日僅在前後「trusted adjusted close / 同日 official raw close」factor 於 1e-6 相對容差內一致時吸收，未使用鄰日價格。

最終 blocked=0，abs gross return >15%=0，可交 Experiments 重跑固定參數 Stage B A；本包未計算策略績效。
