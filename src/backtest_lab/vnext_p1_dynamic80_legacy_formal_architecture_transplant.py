from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


TASK_ID = "TASK-BACKTEST-CORE-VNEXT-P1-DYNAMIC80-LEGACY-FORMAL-ARCHITECTURE-TRANSPLANT-CONTRACT-001"
DEFAULT_POOL = "outputs/vnext_layer4_80_primary_pool_contract_20260708/layer4_80_primary_pool_contract.csv"
DEFAULT_OUTPUT = "outputs/vnext_p1_dynamic80_legacy_formal_architecture_transplant_contract_20260710"
P1_START = "2015-01-02"
P1_END = "2022-12-29"


def run(*, pool_path: str | Path = DEFAULT_POOL, output_dir: str | Path = DEFAULT_OUTPUT) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    _checkpoint(output, "load_primary80")

    usecols = [
        "snapshot_date", "ticker", "name", "market", "pool_rank",
        "layer4_risk_aware_score", "RS20", "RS60", "BIAS60_percentile",
        "volatility_pctile_by_week", "layer1_quality_floor_risk_pctile_by_week",
        "risk_overheat_penalty_context", "layer1_pass_bottom20",
    ]
    pool = pd.read_csv(pool_path, usecols=usecols)
    pool["snapshot_date"] = pd.to_datetime(pool["snapshot_date"])
    pool = pool[pool["snapshot_date"].between(P1_START, P1_END)].copy()
    pool = pool.sort_values(["snapshot_date", "pool_rank", "ticker"])

    _checkpoint(output, "materialize_contract")
    matrix = _candidate_matrix(pool)
    matrix.to_csv(output / "p1_dynamic80_legacy_architecture_candidate_matrix.csv", index=False, encoding="utf-8-sig")
    _variant_contract().to_csv(output / "p1_dynamic80_legacy_architecture_variant_contract.csv", index=False, encoding="utf-8-sig")
    _state_mapping().to_csv(output / "p1_dynamic80_legacy_architecture_state_mapping_audit.csv", index=False, encoding="utf-8-sig")
    _source_quality().to_csv(output / "p1_dynamic80_legacy_architecture_source_quality_audit.csv", index=False, encoding="utf-8-sig")
    blocked = _blocked_ledger()
    blocked.to_csv(output / "p1_dynamic80_legacy_architecture_blocked_ledger.csv", index=False, encoding="utf-8-sig")
    coverage = _coverage(pool)
    coverage.to_csv(output / "requested_vs_actual_coverage.csv", index=False, encoding="utf-8-sig")
    _metric_hooks().to_csv(output / "p1_dynamic80_legacy_architecture_metric_hooks.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(columns=["variant", "episode_id", "episode_rank", "remove_best_1_ready", "remove_best_3_ready", "remove_best_5_ready"]).to_csv(
        output / "p1_dynamic80_legacy_architecture_episode_rechain_hooks.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(columns=["signal_date", "variant", "attack_target_available", "selected_ticker", "no_target_cash", "systemic_bear", "incumbent_valid", "challenger_edge", "reason"]).to_csv(
        output / "p1_dynamic80_legacy_architecture_daily_report_hooks.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(columns=["signal_date", "field", "violation_reason"]).to_csv(
        output / "future_data_audit.csv", index=False, encoding="utf-8-sig"
    )

    readiness = {
        "task_id": TASK_ID,
        "status": "blocked_exact_dynamic80_formal_selector_and_pool2_confirmation_not_source_ready",
        "ready_for_experiments": False,
        "p1_dynamic80_primary80_pit_universe_ready": True,
        "p1_dynamic80_exact_legacy_selector_ready": False,
        "p1_dynamic80_pool2_confirmation_ready": False,
        "p1_dynamic80_unique_position_daily_state_ready": False,
        "p1_dynamic80_official_unadjusted_selected_path_ready": False,
        "selected_stock_adjusted_close_ready": False,
        "cash_sleeve_semantics_ready": True,
        "supersedes_attack_sleeve_no_target_cash_default": True,
        "default_state": "hold_valid_incumbent",
        "cash_only_confirmed_risk_or_no_valid_replacement": True,
        "future_data_violation_count": 0,
        "ready_for_formal": False,
        "ready_for_strategy_replay": False,
        "formal_model_changed": False,
        "trade_decision_changed": False,
        "active_in_trade_decision": False,
        "report_changed": False,
        "portfolio_replay_executed": False,
        "not_live_rule": True,
        "forward_returns_live_rule_usage": False,
    }
    (output / "readiness_for_p1_dynamic80_legacy_formal_architecture_transplant.json").write_text(
        json.dumps(readiness, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "final_summary_zh.md").write_text(_summary(pool, blocked), encoding="utf-8")
    manifest = {
        "task_id": TASK_ID,
        "supersedes_attack_sleeve_no_target_cash_default": True,
        "default_state": "hold_valid_incumbent",
        "cash_only_confirmed_risk_or_no_valid_replacement": True,
        "runner": "src/backtest_lab/vnext_p1_dynamic80_legacy_formal_architecture_transplant.py",
        "canonical_primary80_source": str(pool_path),
        "formal_rule_sources": [
            "configs/frozen_cycle_proven_top1_v1.json",
            "src/backtest_lab/regime_mode_switch.py::frozen_cycle_proven_top1_v1_variant",
            "src/backtest_lab/formal_model_contract.py",
            "src/backtest_lab/pool1_pool2_veto_cap_downweight.py",
        ],
        "artifacts": sorted(path.name for path in output.iterdir() if path.is_file()),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    _checkpoint(output, "completed_blocked_contract")
    return output


def _candidate_matrix(pool: pd.DataFrame) -> pd.DataFrame:
    out = pool.copy()
    out["variant_B_exact_legacy_candidate_universe"] = True
    out["variant_C_risk_adjusted_context_available"] = True
    out["variant_D_incumbent_challenger_context_available"] = True
    out["exact_legacy_rs_score_ready"] = False
    out["exact_dynamic80_pool2_vote_ready"] = False
    out["selected_ticker_authorized"] = False
    out["source_quality"] = "weekly_PIT_primary80_context_only_not_exact_formal_daily_selector"
    out["future_return_used_as_rule"] = False
    return out


def _variant_contract() -> pd.DataFrame:
    rows = [
        ("A_fixed7_exact_formal_reference", "fixed7", "exact formal reference only", "blocked_same_basis_P1_full_coverage"),
        ("B_dynamic80_old_selector_incumbent_hold_reference", "PIT dynamic80", "legacy selector/state reference; no challenger means hold valid incumbent", "blocked_exact_analysis_price_and_dynamic80_pool2_vote"),
        ("C_dynamic80_risk_adjusted_incumbent_challenger", "PIT dynamic80", "risk-adjusted challenger must beat incumbent and justify cost", "blocked_until_same_basis_scores_and_state_ready"),
        ("D_dynamic80_incumbent_protection_replacement_required", "PIT dynamic80", "hold incumbent until invalid; replacement required before stock switch", "blocked_until_incumbent_validity_and_replacement_sources_ready"),
    ]
    return pd.DataFrame(rows, columns=["variant", "attack_universe", "intended_semantics", "readiness_status"])


def _state_mapping() -> pd.DataFrame:
    rows = [
        ("attack_target", "one_stock", "one_stock", "preserved"),
        ("no_new_challenger_incumbent_valid", "00631L waiting/no target", "hold incumbent stock", "superseding_strategy_center_semantics"),
        ("incumbent_invalid_replacement_ready", "not explicit", "switch to replacement stock", "replacement_required"),
        ("incumbent_invalid_no_replacement", "not explicit", "CASH", "cash_allowed_exception"),
        ("systemic_bear", "CASH", "CASH", "preserved"),
        ("portfolio_stop", "CASH", "CASH", "preserved"),
        ("preproof_risk_2of3", "25% 00631L + 75% cash", "reference_only_not_primary_transplant", "not_reused_as_cash_default"),
        ("0050", "market benchmark/fallback in score gate", "market benchmark only", "score benchmark preserved_not_sleeve_asset"),
    ]
    return pd.DataFrame(rows, columns=["state", "legacy_formal_semantics", "attack_sleeve_semantics", "mapping_status"])


def _source_quality() -> pd.DataFrame:
    rows = [
        ("dynamic80_universe", DEFAULT_POOL, "weekly PIT exact pool membership", True),
        ("legacy_selector", "regime_mode_switch.relative_strength_scores", "requires daily event-adjusted/adjusted 20D and 60D analysis price", False),
        ("attack_gate_persistence", "regime_mode_switch._attack_gate_passes", "requires daily top leader over prior 10 trading days", False),
        ("pool2_confirmation", "formal replay pool2_vote", "available for old formal universe only; no dynamic80 PIT vote", False),
        ("execution_price", "official selected-stock unadjusted OHLC", "can be bounded only after exact targets exist", False),
        ("transaction_cost", "EP05 TaiwanCostModel", "stock/cash transition hooks defined; no path yet", True),
    ]
    return pd.DataFrame(rows, columns=["field", "source", "source_quality", "ready"])


def _blocked_ledger() -> pd.DataFrame:
    rows = [
        ("B/C/D", "exact_selector_analysis_price", "Historical adjusted-price escalation stopped; official unadjusted rows cannot safely bridge corporate-action scale for RS20/RS60.", "Strategy Center"),
        ("B/C/D", "dynamic80_pool2_vote", "Existing Pool2 confirmation panel is tied to old formal universe and cannot be relabeled as dynamic80 confirmation.", "Strategy Center/Core architecture decision"),
        ("A", "same_basis_P1_reference", "Existing fixed7 verified metrics use different period/execution basis and cannot be mixed as same-basis P1 result.", "Core/Data"),
        ("B/C/D", "selected_stock_daily_OHLC", "Exact selected tickers do not exist until selector blockers close; bounded Radar request would be premature.", "Core/Data after selector readiness"),
        ("all", "slippage", "No accepted exact slippage model in this contract.", "Experiments proxy audit"),
        ("B/C/D", "incumbent_validity", "Exact PIT incumbent weakening/overheat/risk-invalidity rule has not been mapped from legacy formal architecture.", "Strategy Center/Core architecture decision"),
    ]
    return pd.DataFrame(rows, columns=["variant", "blocked_field", "blocked_reason", "next_owner"])


def _coverage(pool: pd.DataFrame) -> pd.DataFrame:
    counts = pool.groupby("snapshot_date")["ticker"].nunique()
    return pd.DataFrame([{
        "period": "P1",
        "requested_start": P1_START,
        "requested_end": P1_END,
        "actual_primary80_start": pool["snapshot_date"].min().strftime("%Y-%m-%d"),
        "actual_primary80_end": pool["snapshot_date"].max().strftime("%Y-%m-%d"),
        "weekly_snapshot_count": int(counts.size),
        "candidate_row_count": int(len(pool)),
        "min_tickers_per_snapshot": int(counts.min()),
        "max_tickers_per_snapshot": int(counts.max()),
        "exact_state_path_actual_start": "",
        "exact_state_path_actual_end": "",
        "coverage_status": "primary80_ready_exact_architecture_blocked",
    }])


def _metric_hooks() -> pd.DataFrame:
    columns = [
        "variant", "total_return_net_after_cost", "max_drawdown", "calmar_like",
        "cash_exposure", "stock_exposure", "cash_exposure_systemic_bear",
        "cash_exposure_portfolio_stop", "cash_exposure_incumbent_invalid_no_replacement",
        "transition_count", "average_holding_days", "max_holding_days", "cash_episode_count",
        "annual_metrics_ready", "rolling_2y_ready", "rolling_3y_ready",
        "wins_vs_0050_ready", "wins_vs_00631L_ready", "episode_concentration_ready", "metric_status",
    ]
    return pd.DataFrame(
        [[v] + [None] * (len(columns) - 2) + ["blocked_no_unique_position_path"] for v in ["A", "B", "C", "D"]],
        columns=columns,
    )


def _summary(pool: pd.DataFrame, blocked: pd.DataFrame) -> str:
    dates = pool["snapshot_date"].dt.strftime("%Y-%m-%d")
    return f"""# P1 dynamic80 legacy formal architecture transplant contract

## 結論
- verdict：`BLOCKED_EXACT_TRANSPLANT_NOT_READY_FOR_EXPERIMENTS`。
- canonical Layer4 primary80 PIT pool 已完整落入 contract：{len(pool):,} rows，{dates.nunique():,} weekly snapshots，actual {dates.min()}~{dates.max()}。
- 本版已覆蓋並廢棄前一版 `no target -> cash` 語義；`supersedes_attack_sleeve_no_target_cash_default=true`。
- 不以 Layer4 `pool_rank` 冒充 legacy formal daily RS selector，也不以舊 fixed-universe Pool2 vote 冒充 dynamic80 confirmation。
- 因此未產生 B/C/D selected ticker、daily wealth path 或績效；避免 mixed basis 與假精確。

## 關鍵 blocker
1. exact selector / 10-day leadership persistence 需要每日 20D/60D event-adjusted analysis price；selected-stock historical adjusted-close source escalation 已停止。
2. formal Pool2 confirmation panel 沒有 dynamic80 PIT vote 對應。
3. 在 selector 未成立前，無法合理派 Radar 補 selected-ticker-only OHLC，因 target ticker 尚不存在。

## Sleeve 語義
- `default_state=hold_valid_incumbent`：沒有新 challenger 時，續抱有效 incumbent，不回 cash、不回 00631L。
- Cash 只允許 systemic bear、portfolio stop、或 incumbent 已失效且沒有合格 replacement。
- 00631L 不作 sleeve fallback；0050 只作市場與 relative-score benchmark。
- 舊 `no_target_cash_all` 只可列 reference comparator，不可當本版主線。

## 下一步
- 這不是 source acquisition 可以單獨關閉的缺口，請 Strategy Center 判斷是否接受移除 Pool2 的 bounded architecture comparator，或停止 exact transplant route。
- 不交 Experiments，因 `ready_for_experiments=false`。

## 邊界
- diagnostic / contract only；official unadjusted OHLC diagnostic-only。
- formal_model_changed=false；trade_decision_changed=false；active_in_trade_decision=false；report_changed=false。
- ready_for_formal=false；ready_for_strategy_replay=false；not_live_rule=true；forward_returns_live_rule_usage=false。
"""


def _checkpoint(output: Path, step: str) -> None:
    (output / "current_step.txt").write_text(step, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool-path", default=DEFAULT_POOL)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(run(pool_path=args.pool_path, output_dir=args.output_dir))


if __name__ == "__main__":
    main()
