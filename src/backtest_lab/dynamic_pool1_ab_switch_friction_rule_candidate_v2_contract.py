"""Build Dynamic Pool1 A/B switch friction rule-candidate v2 contract.

This runner reworks the exact A/B switch contract into separate rule families:
incumbent protection, challenger overheat discount, quality/RS superiority,
and a balanced combined candidate. It is not a portfolio replay.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from backtest_lab.dynamic_pool1_exact_ab_switch_friction_contract import DEFAULT_BACKTEST_PERIOD_CONTRACT


TASK_ID = "TASK-BACKTEST-CORE-DYNAMIC-POOL1-AB-SWITCH-FRICTION-RULE-CANDIDATE-V2-CONTRACT-001"
EXPERIMENTS_TASK_ID = "TASK-BACKTEST-EXPERIMENTS-DYNAMIC-POOL1-AB-SWITCH-FRICTION-RULE-CANDIDATE-V2-VALIDATION-001"
DEFAULT_EXACT_CONTRACT = Path(
    "outputs/dynamic_pool1_exact_ab_switch_friction_contract_20260705/exact_ab_switch_friction_contract.csv"
)
DEFAULT_OUTPUT_DIR = Path("outputs/dynamic_pool1_ab_switch_friction_rule_candidate_v2_contract_20260705")

RULES = [
    {
        "rule_id": "v2_keep_A_if_still_working_top5_unless_B_score10",
        "family": "incumbent_A_still_working",
        "required_features": ["incumbent_A_still_working_flag", "score_margin"],
        "fallback_policy": "A_top5_and_ma_state_from_exact_contract",
    },
    {
        "rule_id": "v2_keep_A_if_still_working_top5_unless_B_rank3_score10",
        "family": "incumbent_A_still_working",
        "required_features": ["incumbent_A_still_working_flag", "rank_margin", "score_margin"],
        "fallback_policy": "A_top5_and_ma_state_from_exact_contract",
    },
    {
        "rule_id": "v2_keep_A_if_still_working_top10_and_no_trend_break",
        "family": "incumbent_A_still_working",
        "required_features": ["A_rank_still_top10", "incumbent_A_trend_break_flag"],
        "fallback_policy": "A_rank_and_ma_state_from_exact_contract",
    },
    {
        "rule_id": "v2_allow_switch_if_A_trend_break_and_B_rank2_score5",
        "family": "A_break_then_B_superiority",
        "required_features": ["incumbent_A_trend_break_flag", "rank_margin", "score_margin"],
        "fallback_policy": "A_ma_break_and_rank_score_from_exact_contract",
    },
    {
        "rule_id": "v2_allow_switch_if_A_trend_break_and_B_quality_not_lower",
        "family": "A_break_then_B_superiority",
        "required_features": ["incumbent_A_trend_break_flag", "quality_margin"],
        "fallback_policy": "A_ma_break_and_quality_proxy_from_exact_contract",
    },
    {
        "rule_id": "v2_switch_when_A_working_only_if_B_large_margin",
        "family": "B_superiority_when_A_working",
        "required_features": ["incumbent_A_still_working_flag", "rank_margin", "score_margin"],
        "fallback_policy": "A_working_and_B_large_margin_from_exact_contract",
    },
    {
        "rule_id": "v2_switch_when_A_working_only_if_B_not_overheated_large_margin",
        "family": "B_superiority_when_A_working",
        "required_features": [
            "incumbent_A_still_working_flag",
            "rank_margin",
            "score_margin",
            "deviation_gap_B_minus_A_ma20",
            "deviation_gap_B_minus_A_ma60",
        ],
        "fallback_policy": "A_working_B_large_margin_and_not_more_overheated",
    },
    {
        "rule_id": "v2_keep_A_if_top5_cluster_stable",
        "family": "topk_stability_context",
        "required_features": ["top5_total_strength_score", "top_k_rank_stability_5d", "top_k_rank_stability_10d"],
        "fallback_policy": "blocked_until_same_day_topk_strength_panel",
    },
    {
        "rule_id": "v2_watch_B_if_top1_changed_but_top5_stable",
        "family": "topk_stability_context",
        "required_features": ["top5_total_strength_score", "top_k_rank_stability_5d", "top_k_rank_stability_10d"],
        "fallback_policy": "blocked_until_same_day_topk_strength_panel",
    },
    {
        "rule_id": "v2_balanced_A_working_or_B_large_margin",
        "family": "combined_balanced",
        "required_features": [
            "incumbent_A_still_working_flag",
            "incumbent_A_trend_break_flag",
            "rank_margin",
            "score_margin",
            "deviation_gap_B_minus_A_ma20",
            "deviation_gap_B_minus_A_ma60",
            "quality_margin",
            "rs60_B_minus_A_vs_0050",
            "rs60_B_minus_A_vs_00631l",
        ],
        "fallback_policy": "A_break_allows_rank2_score5; A_working_requires_large_margin_and_not_overheated",
    },
]

FUTURE_EXECUTION_STATE_NOTE = {
    "applies_to_future_portfolio_or_execution_contracts_only": True,
    "direct_stock_target": "individual stock signal strong enough; hold stock candidate",
    "no_stock_target_but_market_exposure_allowed": "if no individual stock is strong enough, challenger assumption may hold 00631L/0050 leveraged exposure",
    "bear_or_cash_condition": "cash only when explicit bear or formal cash condition is active",
    "not_applied_in_this_task": True,
    "required_future_mapping_fields": [
        "execution_state",
        "direct_stock_target_ticker",
        "market_exposure_fallback_ticker",
        "bear_or_cash_condition_flag",
        "old_no_target_cash_mapping_weight",
        "new_no_stock_fallback_00631l_mapping_weight",
    ],
}


def run_dynamic_pool1_ab_switch_friction_rule_candidate_v2_contract(
    *,
    repo_root: str | Path = ".",
    exact_contract: str | Path = DEFAULT_EXACT_CONTRACT,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    contract_path = _resolve(root, exact_contract)
    output = _resolve(root, output_dir)
    output.mkdir(parents=True, exist_ok=True)

    exact = pd.read_csv(contract_path)
    v2 = _build_v2_contract(exact)
    missing = _missing_feature_audit_v2(v2)
    future = _future_data_audit(v2)
    by_period = _distribution(v2, ["rule_id", "default_period_label"], "v2_rule_distribution_by_period")
    by_variant = _distribution(v2, ["rule_id", "variant_id"], "v2_rule_distribution_by_variant")
    by_incumbent_state = _distribution(
        v2, ["rule_id", "incumbent_working_state"], "v2_rule_distribution_by_incumbent_state"
    )
    by_topk_stability = _distribution(
        v2, ["rule_id", "topk_stability_state"], "v2_rule_distribution_by_topk_stability"
    )
    readiness = _readiness_by_rule(v2)
    incumbent_context = _incumbent_working_context(v2)
    topk_context = _top_k_strength_context(v2)

    v2.to_csv(output / "exact_ab_switch_friction_rule_candidate_v2_contract.csv", index=False, encoding="utf-8-sig")
    by_period.to_csv(output / "v2_rule_distribution_by_period.csv", index=False, encoding="utf-8-sig")
    by_variant.to_csv(output / "v2_rule_distribution_by_variant.csv", index=False, encoding="utf-8-sig")
    by_incumbent_state.to_csv(output / "v2_rule_distribution_by_incumbent_state.csv", index=False, encoding="utf-8-sig")
    by_topk_stability.to_csv(output / "v2_rule_distribution_by_topk_stability.csv", index=False, encoding="utf-8-sig")
    readiness.to_csv(output / "v2_readiness_by_rule.csv", index=False, encoding="utf-8-sig")
    missing.to_csv(output / "missing_feature_audit_v2.csv", index=False, encoding="utf-8-sig")
    future.to_csv(output / "future_data_audit.csv", index=False, encoding="utf-8-sig")
    incumbent_context.to_csv(output / "incumbent_working_context.csv", index=False, encoding="utf-8-sig")
    topk_context.to_csv(output / "top_k_strength_context.csv", index=False, encoding="utf-8-sig")

    balanced_rows = int(v2.loc[v2["rule_id"].eq("v2_balanced_A_working_or_B_large_margin"), "rule_candidate_triggered"].sum())
    old_strict_rows = int(v2.loc[v2["rule_id"].eq("v2_balanced_A_working_or_B_large_margin"), "old_combined_strict"].sum())
    future_count = int(future["future_data_violation"].sum()) if len(future) else 0
    manifest: dict[str, Any] = {
        "task_id": TASK_ID,
        "status": "completed_v2_rule_candidate_contract",
        "output_dir": str(output),
        "source_exact_contract": str(contract_path),
        "exact_switch_rows": int(len(exact)),
        "v2_contract_rows": int(len(v2)),
        "rule_count": int(len(RULES)),
        "balanced_rule_triggered_rows": balanced_rows,
        "old_combined_strict_rows": old_strict_rows,
        "balanced_rule_less_sparse_than_old_strict": balanced_rows > old_strict_rows,
        "future_data_violation_count": future_count,
        "default_backtest_period_contract": DEFAULT_BACKTEST_PERIOD_CONTRACT,
        "actual_switch_event_start": _date_text(pd.to_datetime(exact["date"], errors="coerce").min()),
        "actual_switch_event_end": _date_text(pd.to_datetime(exact["date"], errors="coerce").max()),
        "future_execution_state_note": FUTURE_EXECUTION_STATE_NOTE,
        "uses_forward_return_as_rule": False,
        "forward_return_used_as_evaluation_metadata": True,
        "formal_model_changed": False,
        "trade_decision_changed": False,
        "active_in_trade_decision": False,
        "report_changed": False,
        "portfolio_replay_executed": False,
        "ready_for_strategy_replay": False,
        "ready_for_formal_absorption": False,
        "handoff_to_experiments_task": EXPERIMENTS_TASK_ID,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "final_summary_zh.md").write_text(_summary_text(manifest, readiness), encoding="utf-8")
    pd.DataFrame([{"task_id": TASK_ID, "status": "completed", "output_dir": str(output)}]).to_csv(
        output / "completed.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(columns=["task_id", "status", "reason"]).to_csv(output / "failed.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(
        [
            {"step": "load_exact_ab_switch_contract", "status": "completed"},
            {"step": "build_v2_rule_families", "status": "completed"},
            {"step": "apply_readiness_gates", "status": "completed"},
            {"step": "write_outputs", "status": "completed"},
        ]
    ).to_csv(output / "run_log.csv", index=False, encoding="utf-8-sig")
    return manifest


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _build_v2_contract(exact: pd.DataFrame) -> pd.DataFrame:
    base = exact.copy()
    base["date"] = pd.to_datetime(base["date"], errors="coerce")
    _coerce_numeric(
        base,
        [
            "incumbent_holding_age_days",
            "rank_margin",
            "score_margin",
            "deviation_gap_B_minus_A_ma20",
            "deviation_gap_B_minus_A_ma60",
            "quality_margin",
            "rs60_B_minus_A_vs_0050",
            "rs60_B_minus_A_vs_00631l",
            "B_minus_A_forward_delta_20d",
            "B_minus_A_forward_delta_40d",
        ],
    )
    bool_cols = [
        "switch_margin_rank2_score5",
        "switch_margin_rank3_score10",
        "switch_no_short_heat_only",
        "switch_after_min_hold5",
        "short_heat_only",
        "medium_quality_confirmed",
        "rs_superiority",
        "quality_not_lower",
        "B_more_overheated_ma20",
        "B_more_overheated_ma60",
        "combined_ab_switch_friction_strict",
        "future_data_violation",
    ]
    for col in bool_cols:
        base[col] = _bool(base.get(col, False), index=base.index)
    base["default_period_label"] = base["date"].map(_default_period_label)
    base["rank2_score5"] = base["switch_margin_rank2_score5"]
    base["rank3_score10"] = base["switch_margin_rank3_score10"]
    base["score10"] = base["score_margin"] >= 0.10
    _ensure_delta(base, "rs20_B_minus_A_vs_0050", "rs20_B_vs_0050", "rs20_A_vs_0050")
    _ensure_delta(base, "rs60_B_minus_A_vs_0050", "rs60_B_vs_0050", "rs60_A_vs_0050")
    _ensure_delta(base, "rs20_B_minus_A_vs_00631l", "rs20_B_vs_00631l", "rs20_A_vs_00631l")
    _ensure_delta(base, "rs60_B_minus_A_vs_00631l", "rs60_B_vs_00631l", "rs60_A_vs_00631l")
    base["any_rs_or_quality_support"] = base["rs_superiority"] | base["quality_not_lower"]
    base["quality_rs_combo"] = base["rs_superiority"] & base["quality_not_lower"]
    base["support_for_balanced"] = base["rs_superiority"] | base["quality_not_lower"] | base["medium_quality_confirmed"]
    base["overheat_ma20_pass"] = base["deviation_gap_B_minus_A_ma20"].notna() & (base["deviation_gap_B_minus_A_ma20"] <= 5)
    base["overheat_ma60_pass"] = base["deviation_gap_B_minus_A_ma60"].notna() & (base["deviation_gap_B_minus_A_ma60"] <= 8)
    base = _add_a_still_working_fields(base)
    for col in ["candidate_as_of_date_A", "candidate_as_of_date_B"]:
        if col not in base.columns:
            base[col] = pd.NA

    rule_frames = []
    for rule in RULES:
        frame = base.copy()
        frame["rule_id"] = rule["rule_id"]
        frame["rule_family"] = rule["family"]
        frame["required_features"] = ";".join(rule["required_features"])
        frame["fallback_policy"] = rule["fallback_policy"]
        frame["rule_candidate_triggered"] = _evaluate_rule(frame, rule["rule_id"])
        frame["required_feature_ready_rate"] = _feature_ready_rate(frame, rule["required_features"])
        frame["required_feature_ready"] = frame["required_feature_ready_rate"] >= 0.50
        frame["old_combined_strict"] = frame["combined_ab_switch_friction_strict"]
        frame["forward_return_used_as_evaluation_metadata"] = True
        frame["uses_forward_return_as_rule"] = False
        frame["portfolio_replay_executed"] = False
        frame["formal_model_changed"] = False
        frame["trade_decision_changed"] = False
        frame["active_in_trade_decision"] = False
        frame["report_changed"] = False
        frame["ready_for_formal_absorption"] = False
        rule_frames.append(frame[_v2_columns()])
    return pd.concat(rule_frames, ignore_index=True)


def _ensure_delta(frame: pd.DataFrame, delta_col: str, b_col: str, a_col: str) -> None:
    if delta_col in frame.columns:
        frame[delta_col] = pd.to_numeric(frame[delta_col], errors="coerce")
        return
    if b_col in frame.columns and a_col in frame.columns:
        frame[delta_col] = pd.to_numeric(frame[b_col], errors="coerce") - pd.to_numeric(frame[a_col], errors="coerce")
    else:
        frame[delta_col] = pd.NA


def _add_a_still_working_fields(base: pd.DataFrame) -> pd.DataFrame:
    out = base.copy()
    _coerce_numeric(out, ["rank_A", "score_A", "close_vs_ma20_A", "close_vs_ma60_A"])
    out["A_rank_still_top3"] = out["rank_A"].notna() & (out["rank_A"] <= 3)
    out["A_rank_still_top5"] = out["rank_A"].notna() & (out["rank_A"] <= 5)
    out["A_rank_still_top10"] = out["rank_A"].notna() & (out["rank_A"] <= 10)
    out["A_rank_still_top_k"] = out["A_rank_still_top5"]
    out["incumbent_A_trend_break_flag"] = (
        (out["close_vs_ma20_A"].notna() & (out["close_vs_ma20_A"] < 0))
        | (out["close_vs_ma60_A"].notna() & (out["close_vs_ma60_A"] < 0))
    )
    out["incumbent_A_still_working_flag"] = out["A_rank_still_top5"] & ~out["incumbent_A_trend_break_flag"]
    out["B_large_margin_over_A"] = (out["rank_margin"] >= 3) & (out["score_margin"] >= 0.10)
    out["B_superiority_required_when_A_working"] = out["incumbent_A_still_working_flag"]
    out["switch_allowed_only_if_A_breaks_or_B_large_margin"] = out["incumbent_A_trend_break_flag"] | out["B_large_margin_over_A"]
    out["A_rs5_vs_0050"] = pd.NA
    out["A_rs20_vs_0050"] = out.get("rs20_A_vs_0050", pd.NA)
    out["A_rs60_vs_0050"] = out.get("rs60_A_vs_0050", pd.NA)
    out["A_rs5_vs_00631L"] = pd.NA
    out["A_rs20_vs_00631L"] = out.get("rs20_A_vs_00631l", pd.NA)
    out["A_rs60_vs_00631L"] = out.get("rs60_A_vs_00631l", pd.NA)
    out["A_close_vs_ma20_pct"] = out["close_vs_ma20_A"]
    out["A_close_vs_ma60_pct"] = out["close_vs_ma60_A"]
    # These require entry/peak state or full same-day top-k panels that are not
    # present in the exact A/B switch contract; keep them explicit and audited.
    out["A_score_decay_from_entry"] = pd.NA
    out["A_score_decay_from_recent_peak"] = pd.NA
    out["A_recent_return_5d"] = pd.NA
    out["A_recent_return_10d"] = pd.NA
    out["A_recent_return_20d"] = pd.NA
    out["A_drawdown_from_20d_high"] = pd.NA
    out["A_drawdown_from_60d_high"] = pd.NA
    out["A_drawdown_from_recent_high"] = pd.NA
    out["top3_total_strength_score"] = pd.NA
    out["top_k_strength_dispersion"] = pd.NA
    out["top5_total_strength_score"] = pd.NA
    out["top10_total_strength_score"] = pd.NA
    out["top_k_rank_stability_5d"] = pd.NA
    out["top_k_rank_stability_10d"] = pd.NA
    out["A_rank_stability_5d"] = pd.NA
    out["A_rank_stability_10d"] = pd.NA
    out["B_rank_stability_5d"] = pd.NA
    out["B_rank_stability_10d"] = pd.NA
    out["top5_cluster_stable"] = False
    out["top1_changed_but_top5_stable"] = False
    out["topk_stability_state"] = "blocked_missing_same_day_topk_panel"
    out["incumbent_working_state"] = "not_working_or_unknown"
    out.loc[out["incumbent_A_still_working_flag"], "incumbent_working_state"] = "A_still_working"
    out.loc[out["incumbent_A_trend_break_flag"], "incumbent_working_state"] = "A_trend_break"
    return out


def _evaluate_rule(frame: pd.DataFrame, rule_id: str) -> pd.Series:
    if rule_id == "v2_keep_A_if_still_working_top5_unless_B_score10":
        return frame["incumbent_A_still_working_flag"] & ~frame["score10"]
    if rule_id == "v2_keep_A_if_still_working_top5_unless_B_rank3_score10":
        return frame["incumbent_A_still_working_flag"] & ~frame["rank3_score10"]
    if rule_id == "v2_keep_A_if_still_working_top10_and_no_trend_break":
        return frame["A_rank_still_top10"] & ~frame["incumbent_A_trend_break_flag"]
    if rule_id == "v2_allow_switch_if_A_trend_break_and_B_rank2_score5":
        return frame["incumbent_A_trend_break_flag"] & frame["rank2_score5"]
    if rule_id == "v2_allow_switch_if_A_trend_break_and_B_quality_not_lower":
        return frame["incumbent_A_trend_break_flag"] & frame["quality_not_lower"]
    if rule_id == "v2_switch_when_A_working_only_if_B_large_margin":
        return frame["incumbent_A_still_working_flag"] & frame["B_large_margin_over_A"]
    if rule_id == "v2_switch_when_A_working_only_if_B_not_overheated_large_margin":
        return frame["incumbent_A_still_working_flag"] & frame["B_large_margin_over_A"] & frame["overheat_ma20_pass"] & frame["overheat_ma60_pass"]
    if rule_id == "v2_keep_A_if_top5_cluster_stable":
        return frame["top5_cluster_stable"]
    if rule_id == "v2_watch_B_if_top1_changed_but_top5_stable":
        return frame["top1_changed_but_top5_stable"]
    if rule_id == "v2_balanced_A_working_or_B_large_margin":
        return (
            (
                frame["incumbent_A_trend_break_flag"]
                & frame["rank2_score5"]
                & (frame["quality_not_lower"] | frame["rs_superiority"] | frame["medium_quality_confirmed"])
            )
            | (
                frame["incumbent_A_still_working_flag"]
                & frame["B_large_margin_over_A"]
                & frame["overheat_ma20_pass"]
                & frame["overheat_ma60_pass"]
                & (frame["quality_not_lower"] | frame["rs_superiority"] | frame["medium_quality_confirmed"])
            )
        )
    raise ValueError(f"Unknown rule_id: {rule_id}")


def _feature_ready_rate(frame: pd.DataFrame, features: list[str]) -> pd.Series:
    readiness = []
    boolean_ready_features = {
        "short_heat_only",
        "medium_quality_confirmed",
        "incumbent_A_still_working_flag",
        "incumbent_A_trend_break_flag",
        "A_rank_still_top3",
        "A_rank_still_top5",
        "A_rank_still_top10",
    }
    for feature in features:
        if feature in boolean_ready_features:
            readiness.append(pd.Series(True, index=frame.index))
        elif feature in frame.columns:
            readiness.append(frame[feature].notna())
        else:
            readiness.append(pd.Series(False, index=frame.index))
    if not readiness:
        return pd.Series(1.0, index=frame.index)
    return pd.concat(readiness, axis=1).mean(axis=1)


def _distribution(v2: pd.DataFrame, keys: list[str], source: str) -> pd.DataFrame:
    grouped = v2.groupby(keys, dropna=False).agg(
        total_rows=("switch_event_id", "count"),
        triggered_rows=("rule_candidate_triggered", "sum"),
        required_feature_ready_rows=("required_feature_ready", "sum"),
        future_data_violation_count=("future_data_violation", "sum"),
    )
    out = grouped.reset_index()
    out["triggered_rate"] = out["triggered_rows"] / out["total_rows"].replace(0, pd.NA)
    out["required_feature_ready_rate"] = out["required_feature_ready_rows"] / out["total_rows"].replace(0, pd.NA)
    out["source"] = source
    out["uses_forward_return_as_rule"] = False
    return out


def _readiness_by_rule(v2: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for rule in RULES:
        subset = v2[v2["rule_id"].eq(rule["rule_id"])]
        triggered = subset[subset["rule_candidate_triggered"]]
        period_rows = {
            period["period_label"]: int(
                triggered[
                    (triggered["date"] >= pd.to_datetime(period["requested_start"]))
                    & (triggered["date"] <= pd.to_datetime(period["requested_end"]))
                ].shape[0]
            )
            for period in DEFAULT_BACKTEST_PERIOD_CONTRACT
        }
        feature_ready_rate = float(triggered["required_feature_ready"].mean()) if len(triggered) else 0.0
        readiness_blockers = []
        if len(triggered) < 25:
            readiness_blockers.append("total_rows_lt25")
        if period_rows["default_backtest_period_2"] < 10:
            readiness_blockers.append("period2_actual_rows_lt10")
        if feature_ready_rate < 0.50 and rule["fallback_policy"] == "none_required":
            readiness_blockers.append("required_feature_readiness_lt50_without_fallback")
        replay_ready = not readiness_blockers
        rows.append(
            {
                "rule_id": rule["rule_id"],
                "rule_family": rule["family"],
                "triggered_rows": int(len(triggered)),
                "default_period_1_actual_rows": period_rows["default_backtest_period_1"],
                "default_period_2_actual_rows": period_rows["default_backtest_period_2"],
                "required_feature_ready_rate": round(feature_ready_rate, 6),
                "fallback_policy": rule["fallback_policy"],
                "candidate_replay_ready": replay_ready,
                "ready_for_strategy_replay": False,
                "readiness_status": "rule_candidate_ready_for_validation" if replay_ready else "rule_candidate_partial",
                "readiness_blocker": ";".join(readiness_blockers),
                "uses_forward_return_as_rule": False,
            }
        )
    return pd.DataFrame(rows)


def _missing_feature_audit_v2(v2: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for rule in RULES:
        subset = v2[v2["rule_id"].eq(rule["rule_id"])]
        for feature in rule["required_features"]:
            if feature in [
                "short_heat_only",
                "medium_quality_confirmed",
                "incumbent_A_still_working_flag",
                "incumbent_A_trend_break_flag",
                "A_rank_still_top3",
                "A_rank_still_top5",
                "A_rank_still_top10",
            ]:
                missing = 0
            elif feature in subset.columns:
                missing = int(subset[feature].isna().sum())
            else:
                missing = len(subset)
            rows.append(
                {
                    "rule_id": rule["rule_id"],
                    "feature": feature,
                    "missing_count": missing,
                    "total_rows": int(len(subset)),
                    "missing_rate": round(missing / len(subset), 6) if len(subset) else 0.0,
                    "fallback_policy": rule["fallback_policy"],
                }
            )
    supplemental_fields = [
        "incumbent_A_still_working_flag",
        "incumbent_A_trend_break_flag",
        "A_rank_still_top3",
        "A_rank_still_top5",
        "A_rank_still_top10",
        "A_score_decay_from_entry",
        "A_score_decay_from_recent_peak",
        "A_recent_return_5d",
        "A_recent_return_10d",
        "A_recent_return_20d",
        "A_rs5_vs_0050",
        "A_rs20_vs_0050",
        "A_rs60_vs_0050",
        "A_rs5_vs_00631L",
        "A_rs20_vs_00631L",
        "A_rs60_vs_00631L",
        "A_close_vs_ma20_pct",
        "A_close_vs_ma60_pct",
        "A_drawdown_from_20d_high",
        "A_drawdown_from_60d_high",
        "top_k_strength_dispersion",
        "top3_total_strength_score",
        "top5_total_strength_score",
        "top10_total_strength_score",
        "top_k_rank_stability_5d",
        "top_k_rank_stability_10d",
        "A_rank_stability_5d",
        "A_rank_stability_10d",
        "B_rank_stability_5d",
        "B_rank_stability_10d",
        "B_superiority_required_when_A_working",
        "switch_allowed_only_if_A_breaks_or_B_large_margin",
    ]
    one_rule = v2[v2["rule_id"].eq(RULES[0]["rule_id"])] if len(v2) else v2
    for feature in supplemental_fields:
        missing = int(one_rule[feature].isna().sum()) if feature in one_rule.columns else len(one_rule)
        rows.append(
            {
                "rule_id": "supplemental_a_still_working_topk_contract",
                "feature": feature,
                "missing_count": missing,
                "total_rows": int(len(one_rule)),
                "missing_rate": round(missing / len(one_rule), 6) if len(one_rule) else 0.0,
                "fallback_policy": "explicit_na_until_entry_peak_or_same_day_topk_panel_available",
            }
        )
    return pd.DataFrame(rows)


def _future_data_audit(v2: pd.DataFrame) -> pd.DataFrame:
    return v2[
        [
            "switch_event_id",
            "rule_id",
            "date",
            "default_period_label",
            "candidate_as_of_date_A",
            "candidate_as_of_date_B",
            "future_data_violation",
            "uses_forward_return_as_rule",
        ]
    ].copy()


def _incumbent_working_context(v2: pd.DataFrame) -> pd.DataFrame:
    fields = [
        "switch_event_id",
        "date",
        "variant_id",
        "incumbent_ticker_A",
        "challenger_ticker_B",
        "incumbent_A_still_working_flag",
        "incumbent_A_trend_break_flag",
        "A_rank_still_top3",
        "A_rank_still_top5",
        "A_rank_still_top10",
        "A_rs5_vs_0050",
        "A_rs20_vs_0050",
        "A_rs60_vs_0050",
        "A_rs5_vs_00631L",
        "A_rs20_vs_00631L",
        "A_rs60_vs_00631L",
        "A_close_vs_ma20_pct",
        "A_close_vs_ma60_pct",
        "A_score_decay_from_entry",
        "A_score_decay_from_recent_peak",
        "A_recent_return_5d",
        "A_recent_return_10d",
        "A_recent_return_20d",
        "A_drawdown_from_20d_high",
        "A_drawdown_from_60d_high",
        "B_superiority_required_when_A_working",
        "switch_allowed_only_if_A_breaks_or_B_large_margin",
    ]
    return v2[v2["rule_id"].eq(RULES[0]["rule_id"])][fields].copy()


def _top_k_strength_context(v2: pd.DataFrame) -> pd.DataFrame:
    fields = [
        "switch_event_id",
        "date",
        "variant_id",
        "incumbent_ticker_A",
        "challenger_ticker_B",
        "top3_total_strength_score",
        "top5_total_strength_score",
        "top10_total_strength_score",
        "top_k_strength_dispersion",
        "top_k_rank_stability_5d",
        "top_k_rank_stability_10d",
        "A_rank_stability_5d",
        "A_rank_stability_10d",
        "B_rank_stability_5d",
        "B_rank_stability_10d",
        "topk_stability_state",
    ]
    return v2[v2["rule_id"].eq(RULES[0]["rule_id"])][fields].copy()


def _coerce_numeric(frame: pd.DataFrame, cols: list[str]) -> None:
    for col in cols:
        frame[col] = pd.to_numeric(frame[col], errors="coerce") if col in frame.columns else pd.NA


def _bool(value: object, *, index: pd.Index) -> pd.Series:
    if isinstance(value, pd.Series):
        if value.dtype == bool:
            return value.fillna(False)
        return value.astype(str).str.lower().isin(["true", "1", "yes"])
    return pd.Series([bool(value)] * len(index), index=index)


def _default_period_label(date: object) -> str:
    when = pd.to_datetime(date, errors="coerce")
    if pd.isna(when):
        return "date_not_ready"
    for period in DEFAULT_BACKTEST_PERIOD_CONTRACT:
        if pd.to_datetime(period["requested_start"]) <= when <= pd.to_datetime(period["requested_end"]):
            return period["period_label"]
    return "outside_default_backtest_period"


def _date_text(value: object) -> str:
    date = pd.to_datetime(value, errors="coerce")
    if pd.isna(date):
        return ""
    return str(date.date())


def _v2_columns() -> list[str]:
    return [
        "switch_event_id",
        "date",
        "next_tradable_date",
        "default_period_label",
        "period_label",
        "variant_id",
        "top1_or_top3_source",
        "rule_id",
        "rule_family",
        "required_features",
        "fallback_policy",
        "rule_candidate_triggered",
        "required_feature_ready_rate",
        "required_feature_ready",
        "incumbent_ticker_A",
        "challenger_ticker_B",
        "candidate_as_of_date_A",
        "candidate_as_of_date_B",
        "incumbent_holding_age_days",
        "rank_A",
        "rank_B",
        "rank_margin",
        "score_A",
        "score_B",
        "score_margin",
        "rank2_score5",
        "rank3_score10",
        "score10",
        "rs20_B_minus_A_vs_0050",
        "rs60_B_minus_A_vs_0050",
        "rs20_B_minus_A_vs_00631l",
        "rs60_B_minus_A_vs_00631l",
        "rs_superiority",
        "quality_margin",
        "quality_not_lower",
        "any_rs_or_quality_support",
        "quality_rs_combo",
        "medium_quality_confirmed",
        "short_heat_only",
        "deviation_gap_B_minus_A_ma20",
        "deviation_gap_B_minus_A_ma60",
        "B_more_overheated_ma20",
        "B_more_overheated_ma60",
        "overheat_ma20_pass",
        "overheat_ma60_pass",
        "switch_after_min_hold5",
        "incumbent_A_still_working_flag",
        "incumbent_A_trend_break_flag",
        "A_rank_still_top_k",
        "A_rank_still_top3",
        "A_rank_still_top5",
        "A_rank_still_top10",
        "A_score_decay_from_entry",
        "A_score_decay_from_recent_peak",
        "A_recent_return_5d",
        "A_recent_return_10d",
        "A_recent_return_20d",
        "A_rs5_vs_0050",
        "A_rs20_vs_0050",
        "A_rs60_vs_0050",
        "A_rs5_vs_00631L",
        "A_rs20_vs_00631L",
        "A_rs60_vs_00631L",
        "A_close_vs_ma20_pct",
        "A_close_vs_ma60_pct",
        "A_drawdown_from_20d_high",
        "A_drawdown_from_60d_high",
        "A_drawdown_from_recent_high",
        "top3_total_strength_score",
        "top_k_strength_dispersion",
        "top5_total_strength_score",
        "top10_total_strength_score",
        "top_k_rank_stability_5d",
        "top_k_rank_stability_10d",
        "A_rank_stability_5d",
        "A_rank_stability_10d",
        "B_rank_stability_5d",
        "B_rank_stability_10d",
        "top5_cluster_stable",
        "top1_changed_but_top5_stable",
        "topk_stability_state",
        "incumbent_working_state",
        "B_large_margin_over_A",
        "B_superiority_required_when_A_working",
        "switch_allowed_only_if_A_breaks_or_B_large_margin",
        "old_combined_strict",
        "B_minus_A_forward_delta_20d",
        "B_minus_A_forward_delta_40d",
        "forward_return_used_as_evaluation_metadata",
        "uses_forward_return_as_rule",
        "future_data_violation",
        "formal_model_changed",
        "trade_decision_changed",
        "active_in_trade_decision",
        "report_changed",
        "portfolio_replay_executed",
        "ready_for_formal_absorption",
    ]


def _summary_text(manifest: dict[str, Any], readiness: pd.DataFrame) -> str:
    period_lines = [
        f"- {p['period_label']}：requested {p['requested_start']}～{p['requested_end']}"
        for p in manifest["default_backtest_period_contract"]
    ]
    ready_count = int(readiness["candidate_replay_ready"].sum())
    return "\n".join(
        [
            "# Dynamic Pool1 A/B switch friction rule-candidate v2 contract",
            "",
            "本包把 exact A/B switch friction 拆成 v2 rule-candidate families；只做候選規則驗證資料，不跑 portfolio，不改正式模型。",
            "",
            f"- exact switch rows：{manifest['exact_switch_rows']}",
            f"- v2 contract rows：{manifest['v2_contract_rows']}",
            f"- rule count：{manifest['rule_count']}",
            f"- balanced rule rows：{manifest['balanced_rule_triggered_rows']}",
            f"- old strict rows：{manifest['old_combined_strict_rows']}",
            f"- balanced less sparse than old strict：{str(manifest['balanced_rule_less_sparse_than_old_strict']).lower()}",
            f"- validation-ready rule candidates by row/readiness gate：{ready_count}",
            f"- future_data_violation_count：{manifest['future_data_violation_count']}",
            f"- actual switch-event range：{manifest['actual_switch_event_start']}～{manifest['actual_switch_event_end']}",
            "- default_backtest_period_contract：",
            *period_lines,
            "- 後續 portfolio/execution contract 備註：沒有個股值得切入時可研究 00631L market exposure fallback，但本包未套用、不產 execution ledger。",
            "- A still-working/top-k 補充：已提供 A trend break、A top3/top5/top10、B large margin 與 switch-only-if-A-breaks-or-B-large-margin 欄位；entry score decay、recent return、top-k dispersion 仍需額外 state/top-k panel。",
            "- uses_forward_return_as_rule=false；portfolio_replay_executed=false；ready_for_formal_absorption=false。",
            f"- 下一棒：{manifest['handoff_to_experiments_task']}",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--exact-contract", default=str(DEFAULT_EXACT_CONTRACT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args(argv)
    manifest = run_dynamic_pool1_ab_switch_friction_rule_candidate_v2_contract(
        repo_root=args.repo_root,
        exact_contract=args.exact_contract,
        output_dir=args.output_dir,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
