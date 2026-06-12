# 本地資料清理候選 v20260612

本文件只列出清理候選，不代表已刪除。實際刪除需再次確認。

清理定義：程式合併完成後，凡是現在或未來持續分析、繼續找更強策略、正式每日報告、shadow mode、回測驗證、影片取證或跨專案交接仍會用到的資料，都必須保留。只有確定不再需要的舊參數搜尋中間產物、重複輸出、已被 summary / 文件取代的臨時結果，才列為可刪除候選。

## 保留

- `backtest_cache/`：價格與回測快取，未來繼續分析會用到。
- `outputs/handoffs/`：跨專案交接證據。
- `docs/`：影片企劃、設計紀錄與驗收說明。
- `outputs/sector_dynamic_pool/radar_core_pool_refactor_check_v1/latest`
- `outputs/sector_dynamic_pool/radar_core_pool_attack_v37_overheat_fine_58_64/latest`
- `outputs/sector_dynamic_pool/radar_core_pool_attack_v38_overheat62_2022_validation/latest`
- `outputs/sector_dynamic_pool/radar_core_pool_attack_v39_overheat62_2023_validation/latest`
- `outputs/sector_dynamic_pool/radar_core_pool_universal_profile_check_v1/latest`
- `outputs/frozen_strategy_monitor`
- `outputs/chip_flow_overlay_shadow_v2/latest`
- `outputs/chip_flow_overlay_shadow_v3_price_confirmed/latest`

## 可刪除或歸檔候選

這些多半是參數搜尋或舊實驗輸出。若要瘦身，可先確認是否仍會用於未來策略搜尋、同口徑比較或影片素材；若不會，才刪除本地輸出目錄，並保留對應 summary 或已寫入影片素材的結果。

- `outputs/sector_dynamic_pool/radar_core_pool_attack_v23_theme_relative`
- `outputs/sector_dynamic_pool/radar_core_pool_attack_v24_bear_exposure`
- `outputs/sector_dynamic_pool/radar_core_pool_attack_v25_weekday_raw`
- `outputs/sector_dynamic_pool/radar_core_pool_attack_v26_acceleration`
- `outputs/sector_dynamic_pool/radar_core_pool_attack_v27_hold_trend`
- `outputs/sector_dynamic_pool/radar_core_pool_attack_v28_risk_combos`
- `outputs/sector_dynamic_pool/radar_core_pool_attack_v29_thresholds`
- `outputs/sector_dynamic_pool/radar_core_pool_attack_v30_stock_threshold_fine`
- `outputs/sector_dynamic_pool/radar_core_pool_attack_v31_liquidity_diversify`
- `outputs/sector_dynamic_pool/radar_core_pool_attack_v32_liquidity_fine`
- `outputs/sector_dynamic_pool/radar_core_pool_attack_v33_2022_validation`
- `outputs/sector_dynamic_pool/radar_core_pool_attack_v34_2023_validation`
- `outputs/sector_dynamic_pool/radar_core_pool_attack_v35_score_stack`
- `outputs/sector_dynamic_pool/radar_core_pool_attack_v36_overheat_fine`
- 早期 `outputs/regime_mode_switch_backtest_v*` 參數搜尋目錄
- 早期 `outputs/strategy_validation_matrix*` 參數矩陣目錄

## 不可直接刪

- 原始資料、價格快取、雷達 snapshot、法人/融資/當沖資料。
- 最佳版、候選版、shadow mode、正式每日報告的最新輸出。
- 能證明策略演進的關鍵成功與失敗節點。
- 任何尚未寫入 summary / 文件 / 影片素材的唯一結果。

## 建議做法

1. 先只保留最新最佳候選、分年驗證、正式每日報告與交接證據。
2. 舊回測輸出若要刪，先確認是否已經有 summary 數字寫入影片素材或文件。
3. 不用 `git reset` 或大量刪除未追蹤檔，避免誤刪使用者或其他任務產物。
