# vNext daily PDF report pipeline readiness

## 結論

- 已建立 vNext 新模型每日 PDF 報告 runner / CLI：`python -m backtest_lab.vnext_daily_report_pipeline`。
- 已產出 local sample PDF：`C:\Users\zergv\Documents\Codex\2026-05-30\ep05-chat-ai-stock-backtest-lab\outputs\vnext_daily_pdf_report_pipeline_readiness_no_publish_20260709\vnext_daily_report_sample_output.pdf`。
- Drive publish 使用既有 `backtest_lab.drive_publish.upsert_pdf`，語義是 update-by-file-id / update-by-name / create-once。
- 本地未實際上傳 Drive，因此不可宣稱已發布。

## 今日資料狀態

- requested date: `2026-07-08`。
- actual data date: `2026-07-08`。
- vNext signal actual date: `2026-07-08`。
- market_data_ready_for_requested_date: `False`。
- selected_branch: `blocked_consensus_trigger_missing`。
- branch_reason: `C2 exact MA60 and exact consensus trigger / route_support max1 not fully materialized for 2026-07-08`。
- 若 `market_data_ready_for_requested_date=false`，代表今日主推薦尚未可發布成 selected signal；reference-only 欄位不可包裝成交易建議。
- low_base_score status: `reference_only_layer4_primary80_blocked`。
- low_base_score 只作 Layer4 component / reference；不可包裝成 selected rule 或 hard filter。

## 報告口徑

- Default 主線：00631L state-hold base + C2 market health gate + consensus trigger + route_support max1。
- Regime branch：保留 market_bias override / G3 guard / M4 breakout+breadth as design context，尚未接成 formal branch。
- RS20 top3：只保留 extreme reference，不作主線 selected。

## Flags

- formal_model_changed=false
- trade_decision_changed=false
- active_in_trade_decision=false
- report_changed=true
- portfolio_replay_executed=false
- ready_for_strategy_replay=false
- ready_for_formal=false
- not_live_rule=true
- forward_returns_live_rule_usage=false
