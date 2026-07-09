# Layer4 low_base_score contract

## 結論

- 已建立 Layer4 `low_base_score` component / formula / sample contract。
- 已補 `existing_low_base_overlap_audit`，確認 low_base 不新增硬篩，只作 ranking/penalty/bonus component。
- 建議放置：Layer4 ranking component；BIAS/RS/risk 欄位重用 Layer2，pullback/reacceleration 語義保留 Layer3。
- 2026-07-08 exact Layer4 primary80 尚未 materialized，所以 top10 是 Layer0-active reference sample，不是 selected rule。
- ready_for_layer4_low_base_score_experiments_diagnostic=false，需等 exact Layer4 primary80 2026-07-08 或 historical panel 接上後再交 Experiments。

## Balanced top10 reference

low_base_rank,ticker,name,low_base_score
1,8070,長華*,0.7600956143156973
2,2231,為升,0.7525648639411326
3,1536,和大,0.74931308109266
4,8937,合騏*,0.7461970662237292
5,6153,嘉聯益,0.7450892523189472
6,2498,宏達電,0.7415610715103218
7,1460,宏遠,0.7413320322387347
8,6550,北極星藥業-KY,0.7379660717288723
9,1305,華夏,0.7374951337087967
10,6505,台塑化,0.7369767081687608


## Flags

- formal_model_changed=false
- trade_decision_changed=false
- active_in_trade_decision=false
- report_changed=false
- portfolio_replay_executed=false
- ready_for_strategy_replay=false
- ready_for_formal=false
- not_live_rule=true
- forward_returns_live_rule_usage=false
