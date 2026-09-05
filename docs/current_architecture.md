# 現行架構與安全整併

## 不變邊界

V4-D 是 frozen formal baseline；C6 score0 是 research；C6 risk 是 challenger。實際持倉紀錄與模擬帳本隔離，實際成交僅可由使用者確認。此次重構不更改選股、交易、成本、提領或排程。

保留服務：AI_stock_market_daily、AI_stock_market_weekly、AI_stock_rotation_radar、AI_stock_schedule_rules、AI_action_orchestrator。保留 YouTube 專案。公開報告不得混入私人持股。

AI_stock_rotation_radar 的 V4-D main 與 C6 publish 工作目錄是同 repo 的 worktrees，不是兩個獨立服務。

## 責任不再等於 task

- Strategy Center：凍結規格、判讀結果、保存歷史結論、管理升級及 checkpoint。收斂為文件，不要求獨立 task。
- Core：PIT、資料契約、執行及事件帳本。功能保留。
- Experiments：回測、parity、成本與績效驗證。功能保留。
- Radar：官方資料取得與每日族群報告。服務與資料 lineage 保留。

同一個主 task 可完成以上流程。這是協作方式簡化，不是把不同交易語義混成一個引擎。

## 遷移狀態：partial

2026-09-05 對本機 Strategy/Core/Experiments/Radar 指定來源資料夾完成 652 檔盤點，67 檔有動態載入。僅解析語法與引用，未宣稱全測試通過。

Strategy/Core/Experiments 的 432 個來源檔均未出現在本 repo 49bbc8c 的相同相對路徑。這不證明沒有同內容異路徑檔案，也不代表已完成整合。已建立逐檔 SHA256 驗證的本機回復封存，原始程式與資料未刪除。

封存位於本次 task 的 outputs/refactor_20260905/local_research_sources.zip，manifest 為 source_recovery_manifest.json。封存只含上述來源，並不涵蓋 work 腳本、authority、資料或回測結果；這些全部保留原位。

目前僅加入現行入口、清冊及清冊驗證器。還需完成：遠端與本機異路徑依賴比對、三份報告測試基準、每個舊模型的證據卡、分批遷移及舊碼退役。未完成前不得刪除整個工作目錄。

## 歷史證據門檻

舊版刪碼前，必須保存規則、requested/actual coverage、成本及資金、訊號及成交時點、公司行動與提領語義、結論、優缺點、證據路徑及 hash、替代狀態。不存在原始證據時標 evidence_partial，不把舊台帳文字當成完整驗證。

舊 weekly R6 重疊持倉數字不能當 baseline；Raw RS20 Top3 僅作 reference。舊參數失敗不等於以現行700萬元和正式外殼重鏈也失敗。

C6 R3-T80 提領 4,240.02 萬元的完整驗證主張已撤回：事件驗收被 wrapper 跳過、估值 carry 未限定 official_no_trade、提領選槽口徑不同。不得以 blocker=0 宣稱完整 exact 驗收。

formal_model_changed=false；trade_decision_changed=false；active_in_trade_decision=false；report_changed=false。

## 首批引擎入版（2026-09-05）

已將本機 Core 的 run_c6_64_start_monthly_withdrawal_rechain.py 及相應測試納入 Git。
原工作目錄暫留相同修正版，避免破壞既有絕對路徑呼叫；尚未完成所有 caller 遷移。

修正：估值只接受 exact ticker/date 價格，或同 ticker/date 的 official_no_trade 證據才允許 carry。缺估值時停止 NAV 計算，禁止漏算持股或沿用未授權舊價。6項事件及估值測試通過，不等於公司行動coverage或完整模型回測驗收。

R3-T80 原提領路徑窄範圍稽核：2406個持股日期鍵；舊來源含2402個exact、1個official_no_trade，缺3個。
官方補查確認：3653於2025-10-02收盤2470元；2383與3044的2026-02-20是官方春節補假，不是應補成交價的日期。
因此要修正交易日曆並重新計算持有TD／訊號與帳本，不能只把兩列刪掉便沿用績效。
原始HTTP回應、SHA256及補件CSV保存於本task outputs/c6_risk_mark_audit_20260905/official_probe。
仍未驗收：全變體公司行動區間覆蓋、完整future-data audit、統一提領規則、重鏈與新風險假說比較。
