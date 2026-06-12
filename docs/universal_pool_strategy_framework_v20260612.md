# 通用股票池策略框架 v20260612

## 目的

本框架的目標是把選股池、候選股評分、流動性門檻、過熱控制與風險調整集中管理，避免未來分成「只能跑 AI 大型權值股」或「只能跑雷達熱門族群中小型股」兩套互不相容的程式。

框架可以套用不同股票池，但不代表所有股票池都用同一組固定參數。正確做法是：

1. 先定義股票池。
2. 依股票池資料推估池類型。
3. 依池類型套用預設門檻。
4. 再用回測校準該股票池的最佳參數。
5. 正式採用前用 out-of-sample / shadow mode 驗證。

## 目前池類型

### large_liquid

適用：0050、0050正二、台積電、聯發科、台達電、鴻海、廣達、緯創、緯穎等大型高流動性池。

預設原則：

- 不使用額外流動性門檻。
- 評分偏相對強弱。
- 過熱門檻較寬。
- 0050、0050正二仍應視為市場曝險工具，不是普通個股。

### mid_small_liquid

適用：雷達題材池中的中小型核心成員。

預設原則：

- 使用 20 日均成交金額門檻。
- 評分偏風險調整後動能。
- 加入過熱限制。
- 避免只因題材熱門就買入流動性不足或短線過熱個股。

### thin_or_mixed

適用：流動性較低、資料不完整或混合型股票池。

預設原則：

- 使用更高流動性門檻。
- 使用更嚴格過熱與回撤限制。
- 預設不應直接進入正式策略，只適合作研究或篩選。

## 已完成整合

- `src/backtest_lab/universal_pool_strategy.py`
  - `infer_pool_profile`
  - `default_parameters_for_profile`
  - `universal_stock_score`
  - `score_universal_candidate`
  - `score_universal_candidates`

- `src/backtest_lab/strategies.py`
  - 大型權值股相對強弱分數改由 `universal_stock_score` 計算，公式不變。

- `src/backtest_lab/radar_core_pool_v1.py`
  - 新增 `RadarCoreVariant.use_pool_profile_defaults`。
  - 預設關閉，不影響目前最佳結果。
  - 開啟後會依股票池類型自動套用流動性、過熱、回撤、均線與評分模式。
  - 新增 `radar_core_mid_small_calibrated_v1_variant()`，正式封裝目前雷達中小型股最佳候選。

- `src/backtest_lab/radar_core_pool_runner.py`
  - 新增 `radar_core_v1_universal_profile_defaults`，用來驗證通用框架是否能直接套入雷達核心池。

## 框架驗證結果

驗證輸出：

- `outputs/sector_dynamic_pool/radar_core_pool_universal_profile_check_v1/latest`

期間：2022-01-03 到 2023-12-29。

結果：

- 通用股票池預設參數版：153.96 萬，報酬率 +53.96%。
- 0050正二買進持有：102.22 萬，報酬率 +2.22%。
- 0050買進持有：94.31 萬，報酬率 -5.69%。

結論：

通用框架可以正常套入不同股票池，但預設參數只是 baseline，不是正式最佳策略。真正有操作價值的版本仍須針對股票池校準，例如目前雷達核心池最佳候選是風險調整分數、流動性門檻與過熱控制共同作用後的版本。

## 雷達中小型校準版

正式 preset：

- `radar_core_mid_small_calibrated_v1_variant()`

對應變體：

- `radar_core_v1_score_risk_stock00_turnover60m_overheat62`

參數重點：

- 只選最強題材 1 個。
- 每次只選該題材中最強個股 1 檔。
- 最高單一持股 100%。
- 20 日均成交金額至少 60M。
- 個股分數門檻 0。
- 20 日漲幅過熱上限 62%。
- 評分使用風險調整動能。
- 強多、弱多、震盪、小空頭仍維持 100% 曝險，大空頭才降為 0%。

已驗證結果：

- 期間：2022-01-03 到 2023-12-29。
- 最終淨值：525.25 萬。
- 報酬率：+425.25%。
- 最大回撤：-28.51%。
- 交易次數：96。

這是目前雷達中小型股研究線的最佳候選，不等於通用 baseline。

## 刪資料原則

不要為了清爽而刪資料。只有合併後確定未來分析、策略搜尋、影片取證、報告、shadow mode 與交接都用不到的舊中間輸出，才可列入刪除。

必須保留：

- 價格快取與雷達 snapshot。
- 正式最佳版、候選版、shadow mode、每日報告輸出。
- 關鍵成功與失敗節點。
- 已用於或可能用於影片取證的 summary、CSV、圖表與報告。
- `outputs/sector_dynamic_pool/radar_core_pool_universal_profile_check_v1/latest`：通用框架 baseline 驗證。

## 後續方向

1. 大型權值股最佳版暫不直接改交易行為，只把共用分數與池型判斷納入基礎層。
2. 雷達中小型股可繼續用此框架找更高績效版本。
3. 法人、融資、當沖、情緒過熱與同族群資金轉移資料補齊後，應以 overlay / shadow 方式先測，不直接替換正式最佳版。
4. 未來若要產品化，應把「股票池輸入、池類型判斷、參數覆寫、每日報告」做成同一條 pipeline。

## 整合護欄

之後每一個整合步驟都必須檢查兩個核心基準：

1. AI 中大型權值股池最佳版 v20260605。
   - 基準檔：`outputs/sector_dynamic_pool/benchmark_v20260605_2022_2023/latest/summary.csv`
   - 期間：2022-01-03 到 2023-12-29。
   - 應維持：354.04 萬，+254.04%，最大回撤 -21.09%，22 次交易。

2. 雷達中小型校準版。
   - 基準檔：`outputs/sector_dynamic_pool/radar_core_pool_refactor_check_v1/latest/radar_core_pool_v1_summary.csv`
   - 期間：2022-01-03 到 2023-12-29。
   - 應維持：525.25 萬，+425.25%，最大回撤 -28.51%，96 次交易。

固定檢查指令：

```powershell
$env:PYTHONPATH='src'; python -m backtest_lab.integration_guardrails
```

若此檢查失敗，代表重構可能破壞了既有最佳策略口徑，必須先查明原因，不應繼續往下合併或刪資料。
