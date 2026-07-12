# P3 weekly rank1 entry/hold/exit state machine

D1-D5固定語義已materialize。Weekly rank1只負責entry；持股期間daily hold/risk exit；normal switch停用。NAV使用同資產event-aware adjusted marks、official execution與EP05+5/10/20bp；00631L只作hurdle，不作fallback。
