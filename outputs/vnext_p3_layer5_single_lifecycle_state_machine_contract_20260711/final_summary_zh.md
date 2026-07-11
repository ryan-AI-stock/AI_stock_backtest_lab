# P3 Layer5 single lifecycle state-machine contract

- Architecture contract 已 materialize；Layer0~4 不變，P3 exact 為 2023-07-14~2026-06-29。
- 固定單一生命週期策略：valid incumbent 預設續抱；無 challenger 不等於 no target；00631L 不是日常 fallback。
- 六個 score blocks 已建立 exclusive raw-field ownership；跨 block 只允許 no-double-count derived flags。
- NA 不等於 0；not_applicable 不扣 confidence，applicable-but-missing 才降低 confidence。
- 正常換倉須同時通過 multi-block、incumbent weakening、confidence、price、quality/risk 與 after-cost edge。
- 市場環境只調門檻，不切換 selector；confirmed bear 才允許 no-position。
- 參數只建立每項最多三值 lattice，尚未選 base；block weights、state thresholds、slippage 亦未凍結。
- 12,320筆 candidate/source evidence rows 已 materialize；最終state/score/action仍因權重、evidence threshold與slippage未凍結而blocked。
- 因此 Phase A event validation不ready，Phase B path亦不ready；不交Experiments、不跑績效。
- C2/route_support/R6/RS20 top3 全部維持 reference-only。
