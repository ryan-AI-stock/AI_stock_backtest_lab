# P1 full lifecycle unified feature/state-machine ingest 判定

- Verdict：BLOCKED。Radar source package 已吸收為保守 ingest/schema contract，但不具備完整 feature matrix 或 state-machine materialization readiness。
- 官方 raw execution OHLCV 可用；trusted Yahoo adjusted series 僅 research-grade analysis，913/976 ticker accepted，不得與 execution price 混欄。
- TWSE compact 必須 date+ticker 去重；margin `TWSE_2017.csv.gz` 損壞檔明確排除。10 筆 true failure 保留，不 silent fill。
- TPEx institutional、TDCC P1、TAIFEX P1 blocked；因此不跑 partial performance、不交 Experiments。
