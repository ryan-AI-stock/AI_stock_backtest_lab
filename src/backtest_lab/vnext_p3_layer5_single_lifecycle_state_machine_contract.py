from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE = REPO_ROOT / "outputs/vnext_p3_full_feature_unified_lifecycle_contract_20260711"
OUTPUT = REPO_ROOT / "outputs/vnext_p3_layer5_single_lifecycle_state_machine_contract_20260711"
TASK_ID = "TASK-BACKTEST-CORE-VNEXT-P3-LAYER5-SINGLE-LIFECYCLE-STATE-MACHINE-CONTRACT-001"


def write_rows(path: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def run(source_dir: Path = SOURCE, output_dir: Path = OUTPUT) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    source_readiness = json.loads((source_dir / "readiness_for_p3_full_feature_unified_lifecycle_contract.json").read_text(encoding="utf-8-sig"))
    if not source_readiness["candidate_level_feature_matrix_ready"]:
        raise ValueError("P3 candidate-level matrix is not ready")

    lifecycle = [
        ("cooling_down", "降溫", "recent strength/attention plus RS5/10 weakening, short-MA weakness or capital support decline", "BIAS/KD cooling; volatility falling without breakdown; quality floor pass", "MA breakdown with RS20/40 negative; high risk; quality/trading block; price core missing", "watch_only"),
        ("turning_up", "轉強", "RS5/10 repair; MA20 reclaim/support; traded-value improvement; risk not worsening", "RS20 stabilizing; KD turn; applicable chip improvement; quality pass", "one-day volume only; falling knife; price core missing; quality blocked", "challenger_pool"),
        ("healthy_rise", "健康上升", "RS20/40 positive or improving; MA20/60 healthy; capital concentration; no extreme risk", "RS5/10 persistence; chip support; non-weak market; high quality", "RS60 high with short RS weakening; extreme BIAS plus blowoff; large-down cluster; price missing", "incumbent_hold_primary"),
        ("overheat_warning", "過熱警告", "high ticker-specific BIAS/KD/RS crowding or turnover blowoff", "strong market/breadth and persistent RS increase tolerance", "single KD/BIAS/RS signal cannot confirm exit", "confidence_down_and_challenger_bar_up"),
        ("confirmed_weakening", "確認轉弱", "multiple RS/MA/capital/risk deterioration conditions", "weak market/breadth; crowding deterioration; new quality risk", "single-day move or single BIAS/KD/RS signal", "forced_replacement_or_exit_candidate"),
    ]
    write_rows(output_dir / "p3_layer5_state_machine_contract.csv", [
        {"state_order": i + 1, "lifecycle_state": s, "label_zh": zh, "required_evidence": req,
         "supporting_evidence": bonus, "veto_or_prohibition": veto, "allowed_use": use,
         "price_core_required": True, "future_return_used_as_rule": False}
        for i, (s, zh, req, bonus, veto, use) in enumerate(lifecycle)
    ])

    ownership = {
        "A_momentum_opportunity": ["RS5", "RS10", "RS20", "RS5_10_acceleration", "RS20_change", "relative_0050_spread_change", "short_cycle_repair_trigger"],
        "B_trend_persistence": ["RS40", "RS60_context", "MA20_slope", "MA60_slope", "above_MA20", "above_MA60", "above_MA120", "RS_persistence", "return20_continuity", "return40_continuity"],
        "C_capital_chip_support": ["traded_value_rank_5D", "traded_value_rank_20D", "traded_value_rank_60D", "turnover_rank_change", "volume_concentration", "institutional_flow", "margin_balance_change", "short_balance_change", "securities_lending_change", "foreign_ownership_change", "TDCC_bucket_change_optional_P3_2"],
        "D_risk_overheat_crowding": ["BIAS20_self_percentile", "BIAS20_zscore", "BIAS60_self_percentile", "BIAS60_zscore", "volatility20", "volatility60", "drawdown", "large_down_day_count", "blowoff_turnover", "risk_score", "risk_bucket", "market_crowding_warning"],
        "E_lifecycle_fit": ["lifecycle_state", "state_transition_direction", "cooling_flag", "turning_flag", "healthy_flag", "overheat_flag", "weakening_flag", "state_confidence", "state_age"],
        "F_fundamental_quality": ["revenue_yoy_mom_3m", "quarterly_revenue_growth", "EPS", "profitability", "margin", "current_ratio", "operating_cashflow", "free_cashflow", "inventory_receivable_risk", "debt_leverage_solvency", "listing_status", "market_cap_investability", "PE_PB_PS_self_percentile_risk_context"],
    }
    rows = []
    for block, fields in ownership.items():
        for field in fields:
            rows.append({"raw_field": field, "owner_block": block, "ownership": "exclusive", "applicability_aware": block.startswith("C_"), "not_applicable_fill_zero": False, "double_count_allowed": False})
    rows += [
        {"raw_field": "overheat_warning_derived", "owner_block": "D_risk_overheat_crowding", "ownership": "derived_crosswalk_to_E", "applicability_aware": False, "not_applicable_fill_zero": False, "double_count_allowed": False, "source_block": "D_risk_overheat_crowding", "derived_flag": "overheat_flag", "no_double_count": True},
        {"raw_field": "short_cycle_repair_derived", "owner_block": "A_momentum_opportunity", "ownership": "derived_crosswalk_to_E", "applicability_aware": False, "not_applicable_fill_zero": False, "double_count_allowed": False, "source_block": "A_momentum_opportunity", "derived_flag": "turning_flag", "no_double_count": True},
    ]
    write_rows(output_dir / "p3_layer5_raw_field_block_ownership.csv", rows)

    applicability = [
        ("observed_value", "include value and weight", False, False),
        ("official_zero", "use zero only when official omission-means-zero semantics is proven", True, False),
        ("not_applicable", "keep NA; exclude field from applicable and available sums", False, False),
        ("source_gap", "keep NA; applicable but unavailable; lower confidence", False, True),
        ("adjusted_price_blocked", "candidate cannot be selected for that snapshot", False, True),
    ]
    write_rows(output_dir / "p3_layer5_applicability_confidence_policy.csv", [
        {"availability_state": s, "scoring_treatment": t, "zero_allowed": z, "missing_penalty_allowed": p,
         "block_score_raw": "sum(value*prefixed_field_weight for available applicable fields)",
         "block_applicable_weight_sum": "sum weights for applicable fields",
         "block_available_weight_sum": "sum weights for available applicable fields",
         "block_confidence": "available_weight_sum/applicable_weight_sum",
         "block_score_normalized": "block_score_raw/block_available_weight_sum",
         "total_score": "sum(block_score_normalized*block_weight*block_confidence)/sum(block_weight*block_confidence)",
         "total_confidence": "sum(block_confidence*block_weight)/sum(block_weight)", "NA_is_zero": False}
        for s, t, z, p in applicability
    ])

    policies = [
        ("hold_incumbent", "incumbent valid and healthy_rise, or overheat_warning without confirmed_weakening", "default; no challenger is not no target"),
        ("normal_switch", "margin pass AND >=3/6 block wins including A/B/C AND incumbent weakened AND after-cost edge pass AND confidence/price/quality/risk gates pass", "all conditions required"),
        ("forced_replacement", "confirmed_weakening OR formal stop OR invalid price/tradability/core data OR severe risk/quality veto", "independent from normal switch"),
        ("no_position_confirmed_bear", "confirmed_bear", "cash allowed"),
        ("watch_only", "forced exit and no valid replacement but no accepted cash rule", "Strategy Center review; do not fabricate action"),
    ]
    write_rows(output_dir / "p3_layer5_incumbent_challenger_policy.csv", [
        {"selected_action": a, "condition": c, "policy_note": n, "next_day_execution": True, "single_position": True,
         "00631L_daily_fallback": False, "reference_signals_can_select": False, "full_EP05_cost_required": True}
        for a, c, n in policies
    ])

    regimes = [
        ("strong_market", "loosen incumbent overheat tolerance; require stronger challenger edge", "same selector"),
        ("ordinary_market", "base thresholds", "same selector"),
        ("weak_market", "stricter entry/confidence; lower weakening tolerance", "same selector"),
        ("confirmed_bear", "cash allowed; only strongest valid holdings survive", "same selector"),
    ]
    write_rows(output_dir / "p3_layer5_market_tightness_policy.csv", [
        {"market_state": s, "threshold_effect": e, "selector_policy": p,
         "primary_context": "0050|TAIEX|breadth|market_value|volatility|margin|TAIFEX|USD_TWD|Nasdaq|SOX|VIX",
         "secondary_context": "SP500|US10Y", "reference_only": "Dow|Nikkei|KOSPI", "future_return_rule": False}
        for s, e, p in regimes
    ])

    lattice = {
        "challenger_margin": ["low:3pct_or_equivalent_z", "base:5pct_or_equivalent_z", "high:8pct_or_equivalent_z"],
        "confidence_floor": [0.60, 0.70, 0.80],
        "incumbent_weakening_threshold": ["mild", "base", "strict"],
        "market_tightening_multiplier": [0.8, 1.0, 1.2],
        "after_cost_edge_buffer": ["1x_round_trip_cost", "1.5x_round_trip_cost", "2x_round_trip_cost"],
        "overheat_tolerance_percentile": [80, 90, 95],
        "state_confirmation_count": [2, 3, 4],
    }
    write_rows(output_dir / "p3_layer5_parameter_lattice.csv", [
        {"parameter": key, "candidate_order": i + 1, "candidate_value": value, "base_selected": False,
         "max_values": 3, "large_grid_allowed": False, "selection_policy": "stable_plateau_P1_walkforward_years_not_single_peak"}
        for key, values in lattice.items() for i, value in enumerate(values)
    ])

    reasons = [
        "lifecycle_state", "lifecycle_state_confidence", "incumbent_valid", "incumbent_weakened", "challenger_valid",
        "challenger_margin_pass", "after_cost_edge_pass", "forced_replacement_reason", "market_tightness_state",
        "confidence_floor_pass", "price_core_valid", "quality_floor_pass", "risk_overheat_warning", "chip_data_applicability",
        "selected_action", "reference_only_flags", "今日進攻部位建議", "為什麼續抱_換倉_空手", "主要支持理由3條",
        "主要風險3條", "挑戰者是否明顯勝出", "是否足以覆蓋換倉成本", "大環境門檻狀態", "資料信心與缺口", "不可選原因",
    ]
    write_rows(output_dir / "p3_layer5_reason_code_contract.csv", [
        {"field": r, "required": True, "machine_or_plain": "plain_zh" if any(ord(ch) > 127 for ch in r) else "machine",
         "allowed_selected_action": "hold_incumbent|switch_to_challenger|forced_exit|no_position_confirmed_bear|watch_only",
         "reference_flags": "c2_reference|route_support_reference|r6_reference|rs20_top3_reference"}
        for r in reasons
    ])

    write_rows(output_dir / "p3_layer5_validation_phase_contract.csv", [
        {"phase": "A_event_acceptance", "portfolio_replay": False, "scope": "states|blocks|confidence|reason_codes|eligibility|missingness", "cost_boundary": "materialize EP05 and slippage hurdle fields", "ready_now": False},
        {"phase": "B_unique_position_path", "portfolio_replay": True, "scope": "single-position next-day path|hold-vs-switch|full costs", "cost_boundary": "brokerage|tax|switch|slippage", "ready_now": False},
        {"phase": "robustness", "portfolio_replay": True, "scope": "walk-forward|leave-one-year-out|remove-best episodes|P3-1|P3-2 TDCC A/B", "cost_boundary": "net after full cost", "ready_now": False},
    ])

    stop_gates = ["ordinary_years_weak_P2_hides", "one_episode_or_ticker_cluster_dependency", "remove_best_episode_collapses",
                  "after_cost_edge_disappears", "excessive_churn", "healthy_incumbent_replaced_without_multiblock_evidence",
                  "missing_data_benefit", "price_core_missing_selected", "reference_signal_selected_without_approval",
                  "00631L_hurdle_fails_after_cost", "2026YTD_severe_instability"]
    write_rows(output_dir / "p3_layer5_stop_gate_contract.csv", [
        {"stop_gate": s, "trigger_action": "stop_or_route_reset", "future_return_used_as_rule": False, "formal_change_allowed": False}
        for s in stop_gates
    ])

    readiness = {
        "task_id": TASK_ID, "status": "architecture_contract_materialized_parameter_selection_and_event_rows_pending",
        "source_candidate_level_feature_matrix_ready": True, "requested_start": "2023-07-11", "requested_end": "2026-07-10",
        "actual_exact_start": "2023-07-14", "actual_exact_end": "2026-06-29", "exact_snapshots": 154,
        "lifecycle_states_contract_ready": True, "six_block_ownership_contract_ready": True,
        "applicability_confidence_contract_ready": True, "incumbent_challenger_policy_ready": True,
        "market_tightness_policy_ready": True, "reason_code_contract_ready": True,
        "parameter_lattice_ready": True, "parameter_base_values_selected": False,
        "block_weights_selected": False, "state_thresholds_materialized": False, "event_rows_materialized": False,
        "EP05_cost_hooks_required": True, "slippage_assumption_ready": False,
        "ready_for_phase_a_event_validation": False, "ready_for_phase_b_unique_position_path": False,
        "ready_for_experiments": False, "future_data_violation_count": 0,
        "formal_model_changed": False, "trade_decision_changed": False, "active_in_trade_decision": False,
        "report_changed": False, "portfolio_replay_executed": False, "ready_for_strategy_replay": False,
        "ready_for_formal": False, "not_live_rule": True, "forward_returns_live_rule_usage": False,
    }
    (output_dir / "p3_layer5_readiness.json").write_text(json.dumps(readiness, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = """# P3 Layer5 single lifecycle state-machine contract

- Architecture contract 已 materialize；Layer0~4 不變，P3 exact 為 2023-07-14~2026-06-29。
- 固定單一生命週期策略：valid incumbent 預設續抱；無 challenger 不等於 no target；00631L 不是日常 fallback。
- 六個 score blocks 已建立 exclusive raw-field ownership；跨 block 只允許 no-double-count derived flags。
- NA 不等於 0；not_applicable 不扣 confidence，applicable-but-missing 才降低 confidence。
- 正常換倉須同時通過 multi-block、incumbent weakening、confidence、price、quality/risk 與 after-cost edge。
- 市場環境只調門檻，不切換 selector；confirmed bear 才允許 no-position。
- 參數只建立每項最多三值 lattice，尚未選 base；block weights、state thresholds、slippage 亦未凍結。
- 因此 Phase A event rows 尚不可 materialize，Phase B path 不 ready，不交 Experiments、不跑績效。
- C2/route_support/R6/RS20 top3 全部維持 reference-only。
"""
    (output_dir / "final_summary_zh.md").write_text(summary, encoding="utf-8")
    files = sorted(p for p in output_dir.iterdir() if p.is_file() and p.name != "manifest.json")
    manifest = {"task_id": TASK_ID, "source": str(source_dir), "source_commit": "ea5b046", "readiness": readiness,
                "files": [{"name": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in files]}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=SOURCE)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    print(run(args.source_dir, args.output_dir))


if __name__ == "__main__":
    main()
