# P1 full lifecycle unified feature/state-machine ingest 判定

- Verdict：BLOCKED。Radar source package 已吸收為保守 ingest/schema contract，但不具備完整 feature matrix 或 state-machine materialization readiness。
- 官方 raw execution OHLCV 可用；trusted Yahoo adjusted series 僅 research-grade analysis，913/976 ticker accepted，不得與 execution price 混欄。
- TWSE 2017 institutional/margin atomic shards checksum、gzip、UTF-8 驗證通過；原損壞 stream 明確排除。
- 原 10 筆 true failure 已由 bounded retry 關閉為 4 accepted + 6 official no_rows；另有 22 個 market-trading-day margin source gaps 保留欄位級 missingness，不 silent fill。
- Adjusted-analysis 剩餘 63 檔已完成 bounded resolution：0 repair、63 explicit blocked；免費 route exhausted，未使用付費來源、successor ticker 或 raw-price substitution。
- TPEx institutional、TDCC P1、TAIFEX P1 blocked；因此不跑 partial performance、不交 Experiments。
