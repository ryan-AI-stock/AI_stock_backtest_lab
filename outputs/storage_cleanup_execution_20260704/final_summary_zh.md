# Core storage cleanup execution

- 狀態：`completed_cleanup_execution_core_rebuildable_only`
- 刪除範圍：Core disposable/rebuildable candidates only
- input rows：`266`
- deleted rows：`224`
- skipped rows：`15`
- missing rows：`27`
- deleted size：`420.600508 MB`

## 保護邊界

- Radar raw sources 未刪。
- 2014/11+ 回測必要資料未刪。
- 0050/TW50 PIT、00631L、TWSE/TPEx price/liquidity cache、formal next-day ledgers/current formal outputs 未刪。
- `formal_model_changed=false`
- `trade_decision_changed=false`
