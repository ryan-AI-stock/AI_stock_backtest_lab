# Layer1 長期營收穩定性 + low-base 風險整合 feature contract

## 結論

- 已建立 latest Layer4 primary80 scoped 的長期營收穩定性 feature contract。
- 長期營收穩定性建議放 Layer1 quality floor / quality score，不是短線爆發 selector。
- low-base 建議放 Layer2/Layer4 soft context，只能小幅 bonus / penalty / tie-break，不作 hard filter、不作獨立 route。
- quarterly_revenue_yoy 目前用 monthly rolling 3M proxy；valuation low-base proxy 仍 blocked。
- ready_for_experiments=true；ready_for_formal=false。

## 6806 森崴能源 sanity check

- status=ready_proxy
- revenue_lumpiness_score=0.2867231662208861
- recent_spike_without_long_history_flag=False
- project_based_revenue_risk_proxy=False
- 這只是 feature sanity check，不是投資判斷。

## Layer placement

- 長期營收穩定性：Layer1 quality floor / quality score。
- revenue_lumpiness / project-based proxy：Layer1 risk context，可傳給 Layer4 作排序風險扣分。
- low-base：Layer2/Layer4 soft context；不得抵消 overheat / volatility / high risk。