# Layer1 t164 TPEx phase-1 bounded proof runner contract

## Verdict
- status=phase1_tpex_50x2_runner_contract_ready_not_executed
- runner_contract_rows=100
- phase1_ticker_count=50
- phase1_period_count=2
- ready_for_radar_phase1_tpex_50x2_runner_execution=true
- ready_for_experiments=false
- ready_for_formal=false
- ready_for_strategy_replay=false

## Boundary
This is a diagnostic/source runner contract only. It does not execute t164 materialization, Experiments, portfolio replay, strategy replay, formal model changes, report changes, or trade decisions.

## Retained blockers
- TPEx historical all-stock universe remains blocked; current-or-carried universe is sampling-only.
- Full-period 891 x 46 expansion requires a separate checkpoint/resume runner contract.
- capex_proxy and receivables_trade remain diagnostic proxy / human-review required.

## Flags
- formal_model_changed=false
- trade_decision_changed=false
- active_in_trade_decision=false
- report_changed=false
- portfolio_replay_executed=false
- ready_for_strategy_replay=false
- not_live_rule=true
- forward_returns_live_rule_usage=false
