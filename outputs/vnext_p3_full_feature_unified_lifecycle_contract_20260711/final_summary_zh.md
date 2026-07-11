# P3 full-feature unified lifecycle partial readiness

- Verdict: PARTIAL/BLOCKED；只完成 ingest/schema readiness，未 materialize state machine、未回測。
- Adjusted analysis 764/776 unique tickers ready，12 blocked；官方 adjusted 不 ready。
- TAIFEX 110/110 交易日 gaps 已由 official range CSV 修復，該 mandatory family ready。
- TDCC 11 gaps 修復 3、剩 8 個 legacy/inactive ticker-week official zero rows；僅 P3-2 optional A/B。Layer4 2026-06-29 後新 PIT membership blocked。
- Adjusted 12 在 exact period 僅有107筆 watchlist reference、0筆 primary80；依 frozen semantics 無 selected-path impact，不再作 primary matrix blocker。
- 新 blocker：Radar compact acquisition scope 實際是 watchlist100，primary80 rows=0；必須補 exact primary80 source scope，不能用 watchlist features 代替。
- P3 不取代 P1；法人／大戶籌碼代理分數僅 proxy components，權重未定。
- Mandatory gaps 關閉前不交 Experiments、不跑 partial performance。
