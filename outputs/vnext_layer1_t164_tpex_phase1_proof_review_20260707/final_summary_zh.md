# Layer1 t164 TPEx phase_1 proof review

## Verdict
- status=phase1_tpex_proof_reviewed_official_asof_blocked
- statement_success_rows=100/100
- official_asof_matched_rows=85/100
- blocked_or_ambiguous_rows=15
- accepted_candidate_count_0_rows=12
- accepted_candidate_count_2_rows=3
- ready_for_radar_official_asof_alternate_route_disambiguation=true
- ready_for_experiments=false
- ready_for_formal=false

## Core decision
Statement route and route cost are positive, but official-asof coverage is not clean enough to update TPEx all-stock proof readiness. The next owner is Radar/Data for bounded alternate official-asof route and detail/subject disambiguation on the 15 blocked rows.

## Boundaries
- No Experiments.
- No replay.
- No formal model/report/trade decision change.
- No silent official-asof backfill.

## Flags
- formal_model_changed=false
- trade_decision_changed=false
- active_in_trade_decision=false
- report_changed=false
- portfolio_replay_executed=false
- ready_for_strategy_replay=false
- not_live_rule=true
- forward_returns_live_rule_usage=false
