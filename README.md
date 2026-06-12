# AI_stock_backtest_lab

被AI研究所 EP05 回測程式基地。

本 repo 用來獨立承接大型回測實驗，避免把歷史資料、參數搜尋、影片資料包輸出塞進 `AI_stock_market_daily` 或 `AI_stock_rotation_radar` 的正式排程主線。

## EP05 第一版目標

驗證一個觀眾最在意的問題：

> 用 AI 訊號在 0050 / 0050 正二 / 七大中大型權值股之間低頻輪動，能不能打敗單純買進持有 0050 正二？

這不是投資建議，也不是保證獲利。這是 AI 輔助回測與策略驗證。

## 回測範圍

- 起始資金：新台幣 1,000,000 元
- 回測區間：2024-01-02 到 2026-05-26
- 暖身資料起日：2023-01-01，只供 2024-01-02 第一筆策略訊號計算使用，不列入績效統計或交易區間
- 訊號日與成交假設：T 日收盤後產生訊號，T+1 交易日成交
- 對照組初始建倉：0050 買進持有組在 2024-01-02 開盤買 0050；0050 正二買進持有組在 2024-01-02 開盤買 0050 正二
- 策略組初始建倉：2024-01-02 開盤第一筆就套用策略選股，不預設先買 0050 或 0050 正二
- 暖身資料：策略可使用回測起日前的歷史資料產生第一筆訊號，避免用 2024-01-02 收盤資料偷看未來；輸出的資產曲線與績效仍從 2024-01-02 開始
- 每日最多交易：2 筆，可以 0 筆
- 成本：台股現股交易成本，含券商手續費與賣出證交稅
- 股利：現金股利入帳後再投入可用資金
- 除權息：用調整後價格計算報酬與指標，用原始成交價模擬交易，避免把除權息價格落差誤判成崩跌

## 標的組

第一組：

- 0050
- 台積電
- 聯發科
- 台達電
- 鴻海
- 廣達
- 緯創
- 緯穎

第二組：

- 0050 正二
- 台積電
- 聯發科
- 台達電
- 鴻海
- 廣達
- 緯創
- 緯穎

## 評估基準

EP05 不再做單一個股跟自己的買進持有比較，主軸改成「八檔資產輪動組合」對抗：

- 0050 買進持有
- 0050 正二買進持有

這兩條是固定對照組，只用來衡量策略組表現；其他策略組、參考組與實驗組不使用固定第一天買 0050 或 0050 正二的規則。

核心成功標準：

- 扣除交易成本後，總報酬要超過 0050 正二買進持有
- 最大回撤不可明顯惡化
- 交易次數要維持低頻、真人可執行
- 結果不能只靠單一極端交易撐起來

## 初版策略候選

1. `relative_strength_top1`
   - 每日收盤後計算八檔資產相對強弱。
   - 第一筆建倉也使用相同排名邏輯，可能買 0050、台積電、緯穎或任一入選標的。
   - T+1 只持有排名最高標的。
   - 若新第一名沒有明顯超過目前持股，避免頻繁換股。

2. `risk_adjusted_top2`
   - 用動能、回撤、波動度組合成風險調整後分數。
   - 主動部位分散到前兩名，降低單檔誤判風險。
   - 保留 10% 基本倉位在防守錨定標的。

3. `regime_anchor_rotation`
   - 先判斷市場風險狀態，再決定積極輪動或降低主動部位。
   - 市場偏強時持有強勢標的；市場偏弱時只保留 10% 錨定倉位，其餘保留現金。
   - 初版錨定標的建議使用 0050，不建議用 0050 正二當防守倉位。

## 預期輸出

後續正式回測至少輸出：

- `video_summary.json`
- `strategies_summary.csv`
- `portfolio_results.csv`
- `trade_log.csv`
- `daily_equity_curve.csv`
- `top_winners_losers.csv`
- 4 到 8 張 PNG 圖表

這些輸出要能直接交給 ChatGPT 中控制作 EP05 影片企劃、頁序、旁白與圖卡素材。

## v0 執行方式

目前 v0 已可跑：

- 0050 買進持有對照組
- 0050 正二買進持有對照組
- `relative_strength_top1` 相對強弱第一名策略骨架
- `dual_momentum_vol_control` 雙動能波動控管策略
- `theme_enhanced_dual_momentum` 題材雷達 proxy 策略
- `benchmark_reconciliation.csv` 基準口徑調節表

執行：

```powershell
$env:PYTHONPATH='src'
python -m backtest_lab.cli --config configs/ep05_universe.json --cache-dir backtest_cache --output-dir backtest_outputs
```

測試：

```powershell
python -m unittest discover -s tests -p "test*.py" -v
```

## 最佳策略每日觀察報告

正式每日產品使用 `frozen_cycle_proven_top1_v1`。它在台股開盤日 15:00 開始檢查九標的當日資料；資料未完整時，每小時重試。報告提供下一交易日的 AI 輔助操作建議，由投資人自行決定是否執行，不會自動下單。

```powershell
$env:PYTHONPATH='src'
python -m backtest_lab.frozen_strategy_monitor --signal-date 2026-05-26 --cache-dir backtest_cache/unified_9_asset_full
```

Drive 固定更新檔名：

```text
AI股票最佳策略每日觀察報告_最新版_v20260605.pdf
```

`v20260605` 代表這版每日觀察報告產品線的完成/發布版本，不是每日訊號日期。每日留存檔會另外帶入訊號日期，例如 `AI股票最佳策略每日觀察報告_2026-06-05_v20260605.pdf`。

GitHub Actions 的 Drive 上傳比照既有 DAILY / WEEKLY 股票報告專案，優先使用 Google OAuth refresh token，不使用 service account 作為主流程。

需要的 repo 或 org secrets：

- `GOOGLE_OAUTH_CLIENT_ID`
- `GOOGLE_OAUTH_CLIENT_SECRET`
- `GOOGLE_OAUTH_REFRESH_TOKEN`

可選設定：

- `FROZEN_REPORT_DRIVE_FOLDER_ID`：覆蓋預設 Drive 目標資料夾。
- `FROZEN_REPORT_DRIVE_FILE_ID`：若要直接覆蓋同一個 Drive 檔案，可指定固定 file id。
- `PORTFOLIO_STORE_JSON`：可選。若要讓 GitHub Action 產出的 PDF 結合個人持倉，需把 `work/portfolio_app/portfolio_store.json` 的內容設成此 secret；未設定時，報告只顯示模型帳戶狀態。
- `GOOGLE_DRIVE_SERVICE_ACCOUNT_JSON`：只保留作為備援，不是建議主設定。

敏感憑證不可寫入 repo、README、workflow log 或測試資料。

## 股票池管理中控台

`127.0.0.1:8765` 現在作為股票池管理中控台，不再作為舊版「AI 股票紀律工作台」首頁。資料檔位於被 git 忽略的 `work/stock_pools/stock_pools.json`，保留日後接 GitHub Actions 或私有伺服器的設定入口。

```powershell
$env:PYTHONPATH='src'
python -m backtest_lab.portfolio_app --signal-root outputs/frozen_strategy_monitor
```

介面支援：

- 檢視內建股票池：`AI中大型權值股池最佳版 v20260605`，共 9 檔。
- 檢視內建股票池：`雷達中小型校準版`，由 radar snapshot 動態決定候選股，不寫死固定清單。
- 維護 `模型延遲公開成績單池`：0050、0050正二，加上跟隨大型權值股池最新模型第一名的第三檔股票。
- 新增、修改、刪除自訂股票池；手動輸入股票代號時一行一檔。
- 預留 `strategy_preset` 欄位，供每日報告、回測與未來私有部署依池執行。

舊版持倉 store 與同步函式仍保留在後端，供每日 PDF 個人化報告相容使用；但 8765 首頁已改為股票池管理。若之後要重新設計持倉功能，應以股票池中控台為入口重做，不再恢復舊版工作台畫面。

Codex 內部驗證原則：

- 不預設用 Codex 背景啟動長駐服務，避免工具呼叫卡住。
- HTTP 驗證使用短生命週期測試，測試內啟動 server、測完自動關閉。
- 若要人工操作網頁，再由使用者於本機終端手動執行上述啟動命令，並用 `Ctrl+C` 關閉。

## AI 模型延遲公開成績單

這條產品線用來公開「延遲一週」的模型歷史觀察結果，不公布即時訊號。第一版以 `2026-05-29` 作為暫定追蹤起點，初始資金 `1,328,709` 元，固定比較：

- AI中大型權值股池最佳版 v20260605
- AI 模型追蹤標的持有，標的會依最佳版模型狀態自動變更
- 0050 買進持有
- 0050正二買進持有

本機手動產出範例：

```powershell
$env:PYTHONPATH='src'
python -m backtest_lab.model_scorecard_report `
  --config configs/ep05_universe.json `
  --strategy-config configs/frozen_cycle_proven_top1_v1.json `
  --group-id group_c_0050_00631l_plus_mega_caps `
  --report-date 2026-06-12 `
  --tracking-start 2026-05-29 `
  --initial-cash 1328709 `
  --cache-dir backtest_cache/frozen_strategy_monitor `
  --output-root outputs/model_scorecard_report
```

固定最新版檔名：

```text
AI模型延遲公開成績單_最新版_v20260612.pdf
```

GitHub Actions workflow：`.github/workflows/model_scorecard_report.yml`。Drive 目標資料夾預設為 `1NDqeKNo3Sa08t0PUqWiSkCLQTZGfKHIe`，也可用 `SCORECARD_REPORT_DRIVE_FOLDER_ID` 覆蓋；若要固定覆蓋同一個 Drive file id，可設定 `SCORECARD_REPORT_DRIVE_FILE_ID`。

## 股票池觀察框架

`stock_pool_observation` 是統一股票池觀察輸出層，用來把大型權值股池、成績單池、雷達中小型池與自訂池整理成同一套 JSON/CSV schema。它只產出觀察排名與候選股分數，不直接取代最佳版正式持倉引擎。

本機批次產出範例：

```powershell
$env:PYTHONPATH='src'
python -m backtest_lab.stock_pool_observation `
  --pool-id all `
  --signal-date 2026-06-05 `
  --cache-dir backtest_cache/frozen_strategy_monitor `
  --output-root outputs/stock_pool_observations
```

若要讓 `雷達中小型校準版` 也產出結果，需提供 radar snapshot v2 原始/replay 資料夾：

```powershell
python -m backtest_lab.stock_pool_observation `
  --pool-id all `
  --signal-date 2026-06-05 `
  --radar-snapshot-dir <radar_snapshot資料夾> `
  --cache-dir backtest_cache/stock_pool_observations `
  --output-root outputs/stock_pool_observations
```

GitHub Actions workflow：`.github/workflows/stock_pool_observation.yml`。workflow 會嘗試 checkout `ryan-AI-stock/AI_stock_rotation_radar`，並預設使用 `AI_stock_rotation_radar/data/history` 作為正式日常 snapshot 來源；也可用 workflow dispatch input `radar_snapshot_dir` 或 repo variable `RADAR_SNAPSHOT_DIR` 覆蓋。若 RADAR repo checkout 失敗或未提供 snapshot 來源，雷達池會在 manifest 中標示 `missing_radar_snapshot_dir`，其餘可解析股票池仍會正常產出。

雷達候選股若只有少數個股缺價格資料，不會讓整個雷達池失敗；缺價代號會寫入 manifest 的 `missing_price_tickers`。

每次批次產出會寫入同一個日期資料夾：

- `stock_pool_observation_manifest.json`
- `stock_pool_observation_summary.csv`
- `stock_pool_observation_report.md`
- `AI股票池觀察總覽_最新版_v20260612.pdf`

Drive 固定更新檔名：

```text
AI股票池觀察總覽_最新版_v20260612.pdf
```

Drive 預設目標資料夾與最佳版每日觀察報告相同：`1O6Se-HfI7ZDTQ-LWeAO6f8vtvoLcCzIj`。可用以下 secrets 或 variables 覆蓋：

- `STOCK_POOL_OBSERVATION_DRIVE_FOLDER_ID`
- `STOCK_POOL_OBSERVATION_DRIVE_FILE_ID`

這條 workflow 是新的股票池觀察總覽入口，用來取代舊的「雷達動態題材池個股輪動回測」每日上傳需求；舊研究程式仍可能被回測測試引用，但不再有獨立每日上傳 workflow。

v0 限制：

- 已可記錄現金股利入帳，但尚未實作股利自動再投入。
- 0050 已加入 2025-06-18 一拆四調整，用來對齊 EP03 TWSE 口徑。
- 尚未實作一般化股票股利 / 分割股數調整。
- `relative_strength_top1` 尚未加冷卻天數、換股門檻或低頻限制，所以交易次數偏高。
- v0 數字只用來驗證工程規則，不應直接當成影片正式結論。

## 專業風格策略：dual_momentum_vol_control

這版不是宣稱「專業投資人都這樣做」，而是把投資實務與量化研究中常見的幾個規則合併成可回測版本：

- 相對動能：在候選標的中挑近期與中期表現較強者。
- 絕對趨勢濾網：價格需站上中期均線，且 3 個月與 6 個月動能為正。
- 波動度懲罰：強勢但波動過大的標的分數會被扣分。
- 週頻再平衡：只在每週第一個交易日檢查是否換股，避免每天追逐排名。
- 無合格標的時可保留現金，避免在趨勢轉弱時硬買。

這類設計接近常見的 tactical asset allocation、time-series momentum、cross-sectional momentum 與 volatility-aware momentum 思路，但仍只是 AI 輔助回測，不是投資建議。

## 題材雷達 proxy 策略

`theme_enhanced_dual_momentum` 會讀取 `AI_stock_rotation_radar` 的 `theme_map.csv`，將候選股對應到題材，例如 AI 伺服器 / ODM、電源 / BBU、ASIC / IP、車用電子等。

由於 `AI_stock_rotation_radar` 沒有從 2024-01-02 開始的完整每日雷達歷史分數，這裡不假裝使用不存在的歷史雷達報告；目前做法是用 rotation radar 的題材分類，再用歷史價格回推各題材在當時的強弱，形成 radar proxy。

正式影片可以說成：

> 這不是直接拿 EP04 當天的雷達結果倒回去用，而是用同一套題材分類邏輯，回頭檢查如果當時也用題材強弱輔助選股，結果會不會更好。

## 穩健性檢查輸出

新增兩個檢查檔：

- `robustness_summary.csv`
  - 日頻 / 週頻 / 月頻再平衡
  - 不同動能窗口
  - 排除聯發科
  - 排除緯穎
  - 2026 單獨驗證段
- `holding_exposure.csv`
  - 每個策略持有各標的的天數與占比

目前初步結果：base 週頻版本、排除聯發科、排除緯穎仍維持強勢，代表不是只靠單一股票；日頻版本交易次數大增且績效/回撤惡化，短週期與月頻版本也對結果影響明顯，代表策略仍有參數敏感度，不能直接當正式結論。

## 圖表輸出

CLI 會在 `backtest_outputs/charts/` 輸出：

- `strategy_final_values.png`
- `strategy_max_drawdowns.png`
- `equity_curves.png`
- `robustness_variants.png`
