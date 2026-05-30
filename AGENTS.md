# AI_stock_backtest_lab 專案規則

本 repo 屬於「被AI研究所 / 個人 AI 工作室」的 S2 工程執行範圍。正式跨聊天室交接來源為共同 Google Sheet：

https://docs.google.com/spreadsheets/d/1vMv4rmHsmPonWwVXTlbYVvhcOw96VheFIPSExsvJTzA/edit

## 啟動順序

開始與本專案相關工作前，先讀：

1. `共通規則`
2. `AI工作室流程中控`
3. `工具專案總覽`
4. 最新 `交接紀錄`
5. 相關 `產出作品`、`查詢紀錄`、`草稿區`、`內容策略中控`

不要為了簽到而寫 Sheet。完成有意義工作後，才回寫工具狀態、完成內容、阻礙、下一步與相關連結。

## 分工邊界

ChatGPT 中控負責策略、內容、品牌、YouTube、LINE、變現與跨專案協調。

Codex 負責程式開發、測試、部署、GitHub Actions、Google Drive PDF、LINE Bot 接入與工具實作。

若任務跨 S1 / S2 / S3，本 repo 只處理可落地的工程規格與資料包；影片角度、公開敘事與發布節奏交由 ChatGPT 中控定案。

## 股票內容邊界

所有公開內容只能寫成 AI 輔助市場觀察、回測、紀律提醒、風險觀察與策略驗證。不可寫成投資建議、喊單、保證績效或穩賺。

回測數字需避免誤導。例如 `+200%` 需說明為 `100 萬變約 300 萬，報酬率約 +200%`。

## 安全規則

AI 股票 repo 優先建立在 GitHub Organization `ryan-AI-stock`。

不得把 SMTP 密碼、OAuth token、API key、LINE secret、Google credentials 或任何敏感資訊寫入記憶、程式碼、README、workflow 或 logs。

可用時使用 org secrets：`SMTP_USERNAME`、`SMTP_PASSWORD`、`REPORT_EMAIL_TO`。若 private repo 無法使用 org secrets，才改用 per-repo `gh secret set`。

