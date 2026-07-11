# P3 full-feature unified lifecycle partial readiness

- Verdict: PARTIAL/BLOCKED；只完成 ingest/schema readiness，未 materialize state machine、未回測。
- Adjusted analysis 764/776 unique tickers ready，12 blocked；官方 adjusted 不 ready。
- TAIFEX 110/110 交易日 gaps 已由 official range CSV 修復，該 mandatory family ready。
- TDCC 11 gaps 修復 3、剩 8 個 legacy/inactive ticker-week official zero rows；僅 P3-2 optional A/B。Primary exact period固定 2023-07-14~2026-06-29。
- 前版 adjusted-12 no-path proof 已 superseded：11檔進入 primary80，共87筆 snapshots，selected impact在凍結selector前未知。
- Trusted adjclose/raw-close factor一致調整官方raw O/H/L/C供research；warmup 181,375/183,886 ready，2,511為official no-row/not-applicable，source gap=0。
- Close-based complete snapshots=87/154；KD-price complete=63/154；all mandatory full-feature complete=0/154。
- Exact chip compacts缺新進/重入前20D warmup；法人、融資融券借券、外資持股不得用不足20日或舊值補齊。
- P3 不取代 P1；法人／大戶籌碼代理分數僅 proxy components，權重未定。
- Mandatory gaps 關閉前不交 Experiments、不跑 partial performance。
