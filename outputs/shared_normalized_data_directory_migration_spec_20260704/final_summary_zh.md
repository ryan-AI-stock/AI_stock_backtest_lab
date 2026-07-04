# Shared normalized data directory migration spec

- 狀態：`completed_spec_only_no_move`
- 建議共用根目錄：`C:\Users\zergv\Documents\Codex\shared_stock_data`
- 本任務只做規格，不搬檔、不刪檔、不壓縮。

## 下一步

1. 先做 Phase 0 manifest-only。
2. 對大型 normalized/cache-compatible tables 建 checksum。
3. 再做 copy dry-run。
4. 只有使用者批准後，才允許 copy、archive 或 move。

## 邊界

- `delete_executed=false`
- `move_executed=false`
- `compress_executed=false`
- `formal_model_changed=false`
- `trade_decision_changed=false`
