# Daily risk feature ingestion contract

- Status: `ready_for_radar_handoff`；目前只建立 schema/PIT/freshness gate，未 materialize、未回測。
- 正式名稱為「法人／大戶籌碼代理分數」，不宣稱精確法人與散戶比例，權重未設定。
- P3-1 不含 TDCC；P3-2 必須同期間有/無 TDCC A/B，TDCC 依實際 release_at 對齊。
- Analysis price 用於 KD/MA/BIAS/RS；官方 raw price 僅用 execution，兩者不可混欄。
- Mandatory 缺欄/PIT 無效為 blocked；optional 缺欄為 partial 並降低 confidence。
