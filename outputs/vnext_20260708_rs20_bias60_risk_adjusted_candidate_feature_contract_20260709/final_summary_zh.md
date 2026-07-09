# 2026-07-08 RS20 / BIAS60 risk-adjusted candidate feature contract

## 結論

- 已建立 same-date feature contract，as_of_date=`2026-07-08`。
- 這是 Layer0 active / candidate universe 的 risk-adjusted diagnostic support，不是 selected signal。
- exact Layer4 primary80、exact consensus trigger、route_support max1 仍 blocked 到 2026-06-29，因此本 package 不可產出主推薦。
- RS20 top3 仍是 reference；新增 BIAS60 percentile/zscore、volatility proxy、Layer1 proxy、low_base/risk-adjusted score 供 Experiments 做排序診斷。

## RS20 top3 audit support

ticker,name,RS20,bias60_raw,bias60_stock_specific_percentile_proxy,volatility_percentile_cross_section_proxy,risk_adjusted_rs20_rank_active
8261,富鼎,1.0908638385974188,0.9515842602112348,0.9999208203278624,0.9656762295081968,219.0
3055,蔚華科,0.8944967151959329,0.6939497009198603,0.9986651012732553,0.9375,245.0
6182,合晶,0.8102247041616806,0.9809948308416132,0.9954415527399966,0.9902663934426229,218.0


## Alternative candidate top10 support

ticker,name,RS20,bias60_stock_specific_percentile_proxy,risk_adjusted_rs20_score,risk_adjusted_rs20_rank_active
6223,旺矽,0.0730683511260512,0.09624083853207288,0.7796468207024451,1.0
3711,日月光投控,0.0737805965207203,0.27341901106692124,0.7775142243061348,2.0
2330,台積電,0.0447766355433998,0.3892038479360449,0.7658255141028122,3.0
6139,亞翔,0.1315705965441846,0.45315734939579067,0.7639158122522622,4.0
1785,光洋科,0.0543096872616324,0.2423210122693969,0.7623232294100146,5.0
6257,矽格,0.1480241893441769,0.5194285445997172,0.7596695403307833,6.0
1326,台化,0.3162949613276264,0.7639960412828475,0.7545695372106648,7.0
1301,台塑,0.2486883342083077,0.7916246875816957,0.7525095405770854,8.0
6505,台塑化,0.1089501051001218,0.6553918975315146,0.7492922069386317,9.0
5314,世紀*,0.1005709999758053,0.47092680325005265,0.747418251120076,10.0


## Readiness

- ready_for_rs20_bias60_risk_adjusted_candidate_diagnostic=True
- ready_for_selected_signal=false
- future_data_violation_count=0

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
