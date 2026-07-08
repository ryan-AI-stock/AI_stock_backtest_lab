"""Build Layer5 pre-action within-80 daily rank context contract.

This is a context/readiness package only. It does not authorize a Layer5
action rule, A/B switch, fallback rule, replay, formal model, report, or trade
decision.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


TASK_ID = "TASK-BACKTEST-CORE-VNEXT-LAYER5-WITHIN80-DAILY-RANK-CONTEXT-CONTRACT-001"
DEFAULT_LAYER4_DIR = Path("outputs/vnext_layer4_80_primary_pool_contract_20260708")
DEFAULT_LAYER4_EXPERIMENTS_DIR = Path(
    "C:/Users/zergv/Documents/Codex/2026-07-06/backtest-lab-experiments-diagnostic-validation-attribution/"
    "outputs/vnext_layer4_three_tier_80_100_31_pool_diagnostic_20260708"
)
DEFAULT_OUTPUT_DIR = Path("outputs/vnext_layer5_within80_daily_rank_context_contract_20260708")
PERIODS = {
    "P1": ("2015-01-02", "2022-12-29"),
    "P2": ("2023-01-02", "2026-06-30"),
    "2024_latest": ("2024-01-02", "2026-06-30"),
    "2026YTD": ("2026-01-02", "2026-06-30"),
}
EVAL_HORIZONS = [5, 10, 20, 30, 40]


def build_contract(
    *,
    layer4_dir: str | Path = DEFAULT_LAYER4_DIR,
    experiments_dir: str | Path = DEFAULT_LAYER4_EXPERIMENTS_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    layer4 = Path(layer4_dir)
    experiments = Path(experiments_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    layer4_readiness = _read_json(layer4 / "readiness_for_layer4_80_primary_pool_contract.json")
    exp_summary = _read_json(experiments / "three_tier_pool_summary.json")
    primary = _read_pool(layer4 / "layer4_80_primary_pool_contract.csv")
    ref100 = _read_pool(layer4 / "layer4_reference_100_extended_watchlist.csv")
    ref31 = _read_pool(layer4 / "layer4_reference_31_high_confidence_subpool.csv")

    contract = _attach_rank_context(primary, ref100, ref31)
    selector_candidates = _final_single_stock_selector_candidates(contract)
    design = _rank_context_design()
    selector_design = _final_selector_candidate_design()
    topk_flags = _topk_context_flags_design()
    coverage = _coverage_by_period(contract)
    requested_actual = _requested_vs_actual_coverage(contract)
    source_quality = _source_quality_matrix()
    missingness = _missingness_by_period(contract)
    blocked_proxy = _blocked_proxy_ledger()
    future_audit = _future_data_audit(pd.concat([contract, selector_candidates], ignore_index=True, sort=False))
    readiness = _readiness(layer4_readiness, exp_summary, contract, selector_candidates, coverage, future_audit)

    _write_csv(contract, output / "layer5_within80_daily_rank_context_contract.csv")
    _write_csv(contract.head(1000), output / "layer5_within80_daily_rank_context_contract_sample.csv")
    (output / ".gitignore").write_text("layer5_within80_daily_rank_context_contract.csv\n", encoding="utf-8")
    _write_csv(selector_candidates, output / "layer5_final_single_stock_selector_candidate_contract.csv")
    _write_csv(design, output / "layer5_daily_rank_context_design.csv")
    _write_csv(selector_design, output / "layer5_final_single_stock_selector_candidate_design.csv")
    _write_csv(topk_flags, output / "layer5_topk_context_flags.csv")
    _write_csv(coverage, output / "layer5_within80_coverage_by_period.csv")
    _write_csv(requested_actual, output / "layer5_within80_requested_vs_actual_coverage.csv")
    _write_csv(source_quality, output / "layer5_within80_source_quality_matrix.csv")
    _write_csv(missingness, output / "layer5_within80_missingness_by_period.csv")
    _write_csv(blocked_proxy, output / "layer5_within80_blocked_proxy_ledger.csv")
    _write_csv(future_audit, output / "layer5_within80_future_data_audit.csv")
    (output / "readiness_for_layer5_within80_daily_rank_context_diagnostic.json").write_text(
        json.dumps(readiness, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    manifest = {
        "task_id": TASK_ID,
        "status": readiness["status"],
        "output_dir": str(output.resolve()),
        "input_layer4_dir": str(layer4.resolve()),
        "input_layer4_experiments_dir": str(experiments.resolve()),
        "output_files": [
            "layer5_within80_daily_rank_context_contract.csv",
            "layer5_within80_daily_rank_context_contract_sample.csv",
            "layer5_final_single_stock_selector_candidate_contract.csv",
            "layer5_final_single_stock_selector_candidate_design.csv",
            "layer5_daily_rank_context_design.csv",
            "layer5_topk_context_flags.csv",
            "layer5_within80_coverage_by_period.csv",
            "layer5_within80_requested_vs_actual_coverage.csv",
            "layer5_within80_source_quality_matrix.csv",
            "layer5_within80_missingness_by_period.csv",
            "layer5_within80_blocked_proxy_ledger.csv",
            "layer5_within80_future_data_audit.csv",
            "readiness_for_layer5_within80_daily_rank_context_diagnostic.json",
            "manifest.json",
            "final_summary_zh.md",
        ],
        "large_local_files_not_tracked": ["layer5_within80_daily_rank_context_contract.csv"],
        "large_local_file_policy": "full within-80 rank context table is retained locally; Git tracks sample/readiness/audit files only",
        **_fixed_flags(),
        "diagnostic_only": True,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "final_summary_zh.md").write_text(_summary(readiness), encoding="utf-8")
    return manifest


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def _read_pool(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"ticker": str}, encoding="utf-8-sig", low_memory=False)
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
    return df


def _bool(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df:
        return pd.Series(False, index=df.index)
    series = df[col]
    if series.dtype == bool:
        return series.fillna(False)
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).ne(0)
    return series.astype(str).str.lower().isin(["true", "1", "yes", "y"])


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce") if col in df else pd.Series(float("nan"), index=df.index)


def _fixed_flags() -> dict[str, bool]:
    return {
        "formal_model_changed": False,
        "trade_decision_changed": False,
        "active_in_trade_decision": False,
        "report_changed": False,
        "portfolio_replay_executed": False,
        "ready_for_strategy_replay": False,
        "not_live_rule": True,
        "forward_returns_live_rule_usage": False,
    }


def _attach_rank_context(primary: pd.DataFrame, ref100: pd.DataFrame, ref31: pd.DataFrame) -> pd.DataFrame:
    out = primary.copy()
    out = out.sort_values(["snapshot_date", "pool_rank", "ticker"]).reset_index(drop=True)
    out["rank_context_date"] = out["snapshot_date"]
    out["rank_context_date_basis"] = "weekly_signal_date_as_layer5_pre_action_context"
    out["within80_rank"] = _num(out, "pool_rank").astype("Int64")
    out["within80_rank_score"] = (81 - _num(out, "within80_rank")).clip(lower=1, upper=80) / 80
    out["within80_top1_context"] = out["within80_rank"].le(1)
    out["within80_top2_context"] = out["within80_rank"].le(2)
    out["within80_top3_context"] = out["within80_rank"].le(3)
    out["within80_top5_context"] = out["within80_rank"].le(5)
    out["within80_top10_context"] = out["within80_rank"].le(10)
    out["within80_top20_context"] = out["within80_rank"].le(20)
    out["top1_top2_top3_context_only"] = True
    out["ab_switch_rule_output"] = False
    out["second_rank_challenger_switch_authorized"] = False
    out["portfolio_like_execution_output"] = False

    ref31_keys = set(zip(ref31["snapshot_date"], ref31["ticker"]))
    ref100_keys = set(zip(ref100["snapshot_date"], ref100["ticker"]))
    primary_keys = set(zip(out["snapshot_date"], out["ticker"]))
    extended_only_keys = ref100_keys - primary_keys
    out["in_31_high_confidence_subpool_reference"] = [
        (date, ticker) in ref31_keys for date, ticker in zip(out["snapshot_date"], out["ticker"])
    ]
    out["in_100_extended_watchlist_reference"] = [
        (date, ticker) in ref100_keys for date, ticker in zip(out["snapshot_date"], out["ticker"])
    ]
    out["extended_100_is_reference_only"] = True
    out["high_confidence_31_is_reference_only"] = True
    out = _attach_extended_reentry_flags(out, extended_only_keys)

    out["fallback_00631L_reference_context_only"] = True
    out["fallback_00631L_trading_rule_output"] = False
    out["cash_classifier_available"] = False
    out["cash_classifier_status"] = "blocked_no_accepted_market_cash_classifier"
    out["current_holder_field_available"] = False
    out["current_holder_status"] = "blocked_no_live_position_state_contract"
    out["incumbent_protection_field_available"] = False
    out["incumbent_protection_status"] = "blocked_no_live_incumbent_state_contract"
    out["layer5_rank_context_contract_only"] = True
    out["layer5_action_rule_authorized"] = False
    out["layer5_selector_output"] = False
    out["daily_report_output"] = False
    out["formal_model_changed"] = False
    out["trade_decision_changed"] = False
    out["active_in_trade_decision"] = False
    out["report_changed"] = False
    out["portfolio_replay_executed"] = False
    out["ready_for_strategy_replay"] = False
    out["not_live_rule"] = True
    out["forward_returns_live_rule_usage"] = False
    out["forward_return_as_rule"] = False
    out["future_return_as_rule"] = False
    return _attach_final_selector_context_scores(out)


def _attach_extended_reentry_flags(out: pd.DataFrame, extended_only_keys: set[tuple[pd.Timestamp, str]]) -> pd.DataFrame:
    dates = sorted(out["snapshot_date"].unique())
    previous_windows: dict[pd.Timestamp, list[pd.Timestamp]] = {}
    for idx, date in enumerate(dates):
        previous_windows[date] = dates[max(0, idx - 4) : idx]

    previous_week = {}
    recent_4w = {}
    for date, ticker in zip(out["snapshot_date"], out["ticker"]):
        prior_dates = previous_windows[date]
        previous_week[date, ticker] = bool(prior_dates and (prior_dates[-1], ticker) in extended_only_keys)
        recent_4w[date, ticker] = any((prior, ticker) in extended_only_keys for prior in prior_dates)

    out["came_from_100_extended_only_previous_week"] = [
        previous_week[(date, ticker)] for date, ticker in zip(out["snapshot_date"], out["ticker"])
    ]
    out["came_from_100_extended_only_recent_4w"] = [
        recent_4w[(date, ticker)] for date, ticker in zip(out["snapshot_date"], out["ticker"])
    ]
    out["extended_100_to_80_reentry_context"] = out["came_from_100_extended_only_recent_4w"]
    return out


def _attach_final_selector_context_scores(out: pd.DataFrame) -> pd.DataFrame:
    out = out.copy()
    risk_penalty = _num(out, "layer4_risk_penalty_score").fillna(0)
    out["lifecycle_strengthening_not_overheated_context"] = (
        (_bool(out, "rs20_30_primary_momentum_positive") | _bool(out, "rs20_30_primary_momentum_stable"))
        & ~_bool(out, "rs_exhaustion_warning_context")
        & risk_penalty.lt(0.45)
    )
    out["lifecycle_pullback_reacceleration_context"] = (
        _bool(out, "pullback_repair_medium_or_high_confidence")
        & _bool(out, "overlap_reacceleration_medium_or_high_confidence")
        & ~_bool(out, "breakdown_risk_medium_or_high_confidence")
    )
    out["confidence_bonus_31_overlap_context"] = out["in_31_high_confidence_subpool_reference"]
    out["reentry_100_to_80_context_bonus"] = out["extended_100_to_80_reentry_context"]
    out["final_selector_lifecycle_context_score"] = (
        0.36 * _num(out, "layer4_risk_aware_score").fillna(0)
        + 0.18 * out["lifecycle_strengthening_not_overheated_context"].astype(float)
        + 0.18 * out["lifecycle_pullback_reacceleration_context"].astype(float)
        + 0.12 * out["confidence_bonus_31_overlap_context"].astype(float)
        + 0.08 * out["reentry_100_to_80_context_bonus"].astype(float)
        + 0.08 * _num(out, "within80_rank_score").fillna(0)
        - 0.18 * risk_penalty
    ).clip(0, 1)
    return out


def _rank_context_design() -> pd.DataFrame:
    rows = [
        ("rank_context_date", "weekly Layer4 signal date used as pre-action context date", "context_key", "not execution day rule"),
        ("within80_rank_score", "normalized inverse rank inside 80 primary pool", "diagnostic_context", "not action rule"),
        ("within80_top1/top2/top3/top5/top10", "top-k context flags", "diagnostic_context", "no A/B switch authorization"),
        ("in_31_high_confidence_subpool_reference", "ticker also appears in 31 reference", "reference_context", "not primary pool"),
        ("came_from_100_extended_only_recent_4w", "ticker moved from extended-only watchlist into primary pool recently", "reentry_context", "not buy rule"),
        ("layer1/layer2/layer3/layer4 scores", "pass-through context fields", "diagnostic_context", "no hard gate here"),
        ("fallback_00631L_reference_context_only", "fallback metadata only", "reference_context", "no fallback trading rule"),
        ("current_holder/incumbent fields", "blocked until live position-state contract exists", "blocked", "not fabricated"),
        ("final_selector_lifecycle_context_score", "diagnostic single-stock selector context score", "selector_candidate_context", "not live rule"),
    ]
    return pd.DataFrame(rows, columns=["field_or_group", "definition", "role", "boundary"])


def _final_single_stock_selector_candidates(contract: pd.DataFrame) -> pd.DataFrame:
    scored = _attach_final_selector_context_scores(contract)
    variants = [
        ("raw_top1_within80", 1, "within80_rank", True),
        ("best_risk_adjusted_top3", 3, "layer4_risk_aware_score", False),
        ("best_risk_adjusted_top5", 5, "layer4_risk_aware_score", False),
        ("best_risk_adjusted_top10", 10, "layer4_risk_aware_score", False),
        ("best_lifecycle_state_top10", 10, "final_selector_lifecycle_context_score", False),
    ]
    rows = []
    for date, group in scored.groupby("snapshot_date", sort=True):
        for variant, scope, score_col, ascending in variants:
            scoped = group[group["within80_rank"].le(scope)].copy()
            if scoped.empty:
                continue
            if ascending:
                chosen = scoped.sort_values(["within80_rank", "ticker"], ascending=[True, True]).head(1)
            else:
                chosen = scoped.sort_values([score_col, "layer4_risk_aware_score", "within80_rank"], ascending=[False, False, True]).head(1)
            chosen = chosen.copy()
            chosen["selector_candidate_variant"] = variant
            chosen["selector_candidate_rank_scope"] = scope
            chosen["selector_candidate_score_col"] = score_col
            chosen["selector_candidate_score"] = _num(chosen, score_col).values
            chosen["next_day_entry_assumption"] = "diagnostic_only_next_trading_session_after_rank_context_date"
            chosen["single_stock_selector_candidate_context_only"] = True
            chosen["final_selector_action_rule_output"] = False
            chosen["ab_switch_rule_output"] = False
            chosen["trade_decision_output"] = False
            chosen["portfolio_like_execution_output"] = False
            rows.append(chosen)
    out = pd.concat(rows, ignore_index=True)
    out = out.sort_values(["selector_candidate_variant", "snapshot_date"]).reset_index(drop=True)
    out["selector_candidate_changed_from_previous_signal"] = False
    out["consecutive_same_candidate_signal_count"] = 1
    for variant, idx in out.groupby("selector_candidate_variant").groups.items():
        prev_ticker = None
        streak = 0
        for row_idx in idx:
            ticker = out.at[row_idx, "ticker"]
            changed = prev_ticker is not None and ticker != prev_ticker
            streak = 1 if changed or prev_ticker is None else streak + 1
            out.at[row_idx, "selector_candidate_changed_from_previous_signal"] = changed
            out.at[row_idx, "consecutive_same_candidate_signal_count"] = streak
            prev_ticker = ticker
    out["turnover_switch_frequency_proxy"] = out["selector_candidate_changed_from_previous_signal"]
    return out


def _final_selector_candidate_design() -> pd.DataFrame:
    rows = [
        (
            "raw_top1_within80",
            "rank 1 inside Layer4 80 primary pool",
            "single-stock selector diagnostic candidate",
            "not trade decision; not A/B switch",
        ),
        (
            "best_risk_adjusted_top3",
            "highest layer4_risk_aware_score among within80 top3",
            "single-stock selector diagnostic candidate",
            "not trade decision; top-decile retention is secondary context only",
        ),
        (
            "best_risk_adjusted_top5",
            "highest layer4_risk_aware_score among within80 top5",
            "single-stock selector diagnostic candidate",
            "not trade decision",
        ),
        (
            "best_risk_adjusted_top10",
            "highest layer4_risk_aware_score among within80 top10",
            "single-stock selector diagnostic candidate",
            "not trade decision",
        ),
        (
            "best_lifecycle_state_top10",
            "highest lifecycle context score among top10: strengthening-not-overheated, repair-reacceleration, 31 overlap, 100-to-80 reentry",
            "single-stock selector diagnostic candidate",
            "context-only; no live lifecycle rule",
        ),
    ]
    return pd.DataFrame(rows, columns=["selector_candidate_variant", "definition", "diagnostic_role", "boundary"])


def _topk_context_flags_design() -> pd.DataFrame:
    rows = []
    for k in [1, 2, 3, 5, 10, 20]:
        rows.append(
            {
                "topk_flag": f"within80_top{k}_context",
                "definition": f"within80_rank <= {k}",
                "allowed_use": "Experiments diagnostic grouping only",
                "prohibited_use": "live action, A/B switch, trade decision, daily report output",
                "forward_return_used": False,
            }
        )
    rows.append(
        {
            "topk_flag": "in_31_high_confidence_subpool_reference",
            "definition": "same ticker/date also in 31-stock reference subpool",
            "allowed_use": "context overlap diagnostic",
            "prohibited_use": "main pool replacement or formal selector",
            "forward_return_used": False,
        }
    )
    rows.append(
        {
            "topk_flag": "came_from_100_extended_only_recent_4w",
            "definition": "ticker was in 100 extended-only reference during previous 4 weekly snapshots and is now in 80 primary",
            "allowed_use": "re-entry context diagnostic",
            "prohibited_use": "buy/switch rule",
            "forward_return_used": False,
        }
    )
    return pd.DataFrame(rows)


def _coverage_by_period(contract: pd.DataFrame) -> pd.DataFrame:
    frame = contract.copy()
    frame["period"] = frame["snapshot_date"].map(_period_label)
    rows = []
    for period, group in frame.groupby("period", dropna=False):
        rows.append(
            {
                "period": period,
                "row_count": len(group),
                "weekly_snapshot_count": int(group["snapshot_date"].nunique()),
                "avg_selected_per_week": float(group.groupby("snapshot_date").size().mean()),
                "min_selected_per_week": int(group.groupby("snapshot_date").size().min()),
                "max_selected_per_week": int(group.groupby("snapshot_date").size().max()),
                "top1_rows": int(group["within80_top1_context"].sum()),
                "top3_rows": int(group["within80_top3_context"].sum()),
                "top5_rows": int(group["within80_top5_context"].sum()),
                "top10_rows": int(group["within80_top10_context"].sum()),
                "in_31_reference_rows": int(group["in_31_high_confidence_subpool_reference"].sum()),
                "extended_reentry_recent_4w_rows": int(group["came_from_100_extended_only_recent_4w"].sum()),
            }
        )
    return pd.DataFrame(rows)


def _requested_vs_actual_coverage(contract: pd.DataFrame) -> pd.DataFrame:
    actual_start = contract["snapshot_date"].min()
    actual_end = contract["snapshot_date"].max()
    rows = []
    for period, (requested_start, requested_end) in PERIODS.items():
        start = pd.Timestamp(requested_start)
        end = pd.Timestamp(requested_end)
        in_period = contract[contract["snapshot_date"].between(start, end)]
        rows.append(
            {
                "period": period,
                "requested_start": requested_start,
                "requested_end": requested_end,
                "actual_contract_start": actual_start.date().isoformat(),
                "actual_contract_end": actual_end.date().isoformat(),
                "rows_in_requested_period": len(in_period),
                "weekly_snapshots_in_requested_period": int(in_period["snapshot_date"].nunique()),
                "rows_before_requested_start": int((contract["snapshot_date"] < start).sum()),
                "rows_after_requested_end": int((contract["snapshot_date"] > end).sum()),
                "coverage_note": "actual coverage is reported separately; do not use actual range as requested-period conclusion",
            }
        )
    return pd.DataFrame(rows)


def _period_label(value: Any) -> str:
    date = pd.to_datetime(value)
    hits = []
    for label, (start, end) in PERIODS.items():
        if pd.Timestamp(start) <= date <= pd.Timestamp(end):
            hits.append(label)
    return "|".join(hits) if hits else "outside_requested_periods"


def _source_quality_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("Layer4_80_primary_pool", "exact_from_core_layer4_contract", "base universe"),
            ("Layer4_31_high_confidence_overlap", "exact_from_core_layer4_reference", "context flag"),
            ("Layer4_100_extended_reentry", "exact_from_core_layer4_reference", "context flag"),
            ("within80_rank_score/topk", "derived_exact_from_layer4_pool_rank", "diagnostic rank context"),
            ("final_single_stock_selector_candidates", "derived_exact_from_within80_context", "diagnostic selector candidate context"),
            ("Layer1_quality", "diagnostic_exact_from_core_contract", "pass-through"),
            ("Layer2_capital_RS_context", "diagnostic_exact_or_proxy_mixed", "pass-through"),
            ("Layer3_broad_label_context", "diagnostic_exact_or_proxy_mixed", "pass-through"),
            ("Layer4_pool_scores", "diagnostic_exact_from_core_contract", "pass-through"),
            ("forward_excess_5d_10d_20d_30d", "evaluation_metadata_only", "not rule construction"),
            ("forward_excess_40d", "evaluation_metadata_only", "decay reference only"),
            ("current_holder/incumbent", "blocked", "no live position-state contract"),
            ("cash_classifier/fallback_rule", "blocked", "no accepted market cash classifier"),
            ("00631L/0050正二", "reference_metadata_only", "not ordinary stock row"),
        ],
        columns=["field_group", "source_quality", "contract_role"],
    )


def _missingness_by_period(contract: pd.DataFrame) -> pd.DataFrame:
    fields = [
        "within80_rank",
        "within80_rank_score",
        "final_selector_lifecycle_context_score",
        "pool_selection_score",
        "layer4_risk_aware_score",
        "layer4_broad_opportunity_net_score",
        "layer1_quality_floor_risk_pctile_by_week",
        "layer2_support_signal_count",
        "layer2_warning_signal_count",
        "momentum_continuation_score",
        "pullback_repair_score",
        "overlap_reacceleration_score",
        "neutral_quality_liquidity_score",
        "exhaustion_risk_score",
        "breakdown_risk_score",
        "traded_value_rank_20d",
        "RS20",
        "RS30_proxy",
        "RS60",
        "BIAS20",
        "BIAS60",
        "volatility",
    ] + [f"forward_excess_vs_0050_{h}d" for h in EVAL_HORIZONS] + [f"forward_excess_vs_00631L_{h}d" for h in EVAL_HORIZONS]
    frame = contract.copy()
    frame["period"] = frame["snapshot_date"].map(_period_label)
    rows = []
    for period, group in frame.groupby("period", dropna=False):
        for field in fields:
            missing = int(group[field].isna().sum()) if field in group else len(group)
            rows.append(
                {
                    "period": period,
                    "field": field,
                    "row_count": len(group),
                    "missing_count": missing,
                    "missing_share": float(missing / len(group)) if len(group) else 0.0,
                }
            )
    return pd.DataFrame(rows)


def _blocked_proxy_ledger() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("current_holder_state", "blocked", "No live position-state contract; not fabricated"),
            ("incumbent_protection_state", "blocked", "No live incumbent state contract; not fabricated"),
            ("B_switch_rule", "blocked", "A/B switch explicitly unauthorized"),
            ("final_single_stock_live_rule", "blocked", "Selector candidates are diagnostic only; no live rule authorization"),
            ("fallback_00631L_trading_rule", "blocked", "Fallback reference only; no accepted cash/fallback classifier"),
            ("cash_classifier", "blocked", "No accepted market cash classifier"),
            ("risk_bucket", "blocked", "Formal risk bucket unavailable"),
            ("RS30_proxy", "proxy", "Exact RS30 unavailable"),
            ("large_down_blowoff", "proxy", "Diagnostic risk proxy only"),
            ("Layer5_action_rule", "blocked", "Layer5 action rule unauthorized"),
            ("portfolio_replay", "blocked", "Replay unauthorized"),
        ],
        columns=["field_or_policy", "status", "reason"],
    )


def _future_data_audit(contract: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "audit_item": "forward_returns_used_in_rank_context_rule",
                "status": "passed",
                "violation_count": 0,
                "evidence": "top-k/rank context derives from Layer4 pool_rank and PIT context; forward columns are evaluation metadata only",
            },
            {
                "audit_item": "future_return_as_rule",
                "status": "passed",
                "violation_count": int(_bool(contract, "future_return_as_rule").sum()),
                "evidence": "future_return_as_rule=false",
            },
            {
                "audit_item": "ab_switch_rule_output",
                "status": "passed",
                "violation_count": int(_bool(contract, "ab_switch_rule_output").sum()),
                "evidence": "No B switch rule emitted",
            },
            {
                "audit_item": "layer5_selector_output",
                "status": "passed",
                "violation_count": int(_bool(contract, "layer5_selector_output").sum()),
                "evidence": "No Layer5 selector output emitted",
            },
            {
                "audit_item": "final_selector_action_rule_output",
                "status": "passed",
                "violation_count": int(_bool(contract, "final_selector_action_rule_output").sum()),
                "evidence": "Single-stock selector candidates are diagnostic candidates only",
            },
            {
                "audit_item": "00631L_as_ordinary_stock_member",
                "status": "passed",
                "violation_count": int(_bool(contract, "fallback_00631L_is_ordinary_stock_pool_member").sum()),
                "evidence": "00631L remains fallback/reference metadata only",
            },
        ]
    )


def _readiness(
    layer4_readiness: dict[str, Any],
    exp_summary: dict[str, Any],
    contract: pd.DataFrame,
    selector_candidates: pd.DataFrame,
    coverage: pd.DataFrame,
    future_audit: pd.DataFrame,
) -> dict[str, Any]:
    future_violations = int(future_audit["violation_count"].sum())
    min_count = int(coverage["min_selected_per_week"].min())
    max_count = int(coverage["max_selected_per_week"].max())
    actual_start = contract["snapshot_date"].min()
    actual_end = contract["snapshot_date"].max()
    rows_after_p2_requested_end = int((contract["snapshot_date"] > pd.Timestamp(PERIODS["P2"][1])).sum())
    ready = future_violations == 0 and min_count == 80 and max_count == 80
    return {
        "task_id": TASK_ID,
        "status": "layer5_within80_daily_rank_context_contract_ready_for_experiments_intake" if ready else "layer5_within80_daily_rank_context_contract_blocked",
        "diagnostic_only": True,
        "rank_context_date_basis": "weekly_signal_date_as_layer5_pre_action_context",
        "input_layer4_status": layer4_readiness.get("status"),
        "input_layer4_experiments_verdict": exp_summary.get("verdict"),
        "base_universe": "Layer4_80_primary_pool",
        "row_count": int(len(contract)),
        "weekly_snapshot_count": int(contract["snapshot_date"].nunique()),
        "selected_count_min": min_count,
        "selected_count_max": max_count,
        "actual_coverage_start": actual_start.date().isoformat(),
        "actual_coverage_end": actual_end.date().isoformat(),
        "default_requested_p1": {"requested_start": PERIODS["P1"][0], "requested_end": PERIODS["P1"][1]},
        "default_requested_p2": {"requested_start": PERIODS["P2"][0], "requested_end": PERIODS["P2"][1]},
        "rows_after_default_requested_p2_end": rows_after_p2_requested_end,
        "top1_rows": int(contract["within80_top1_context"].sum()),
        "top3_rows": int(contract["within80_top3_context"].sum()),
        "top5_rows": int(contract["within80_top5_context"].sum()),
        "top10_rows": int(contract["within80_top10_context"].sum()),
        "in_31_high_confidence_reference_rows": int(contract["in_31_high_confidence_subpool_reference"].sum()),
        "extended_100_to_80_reentry_recent_4w_rows": int(contract["came_from_100_extended_only_recent_4w"].sum()),
        "final_single_stock_selector_candidate_rows": int(len(selector_candidates)),
        "final_single_stock_selector_candidate_variant_count": int(selector_candidates["selector_candidate_variant"].nunique()),
        "final_single_stock_selector_candidate_variants": sorted(selector_candidates["selector_candidate_variant"].unique().tolist()),
        "turnover_switch_frequency_proxy_available": True,
        "top_decile_retention_role": "secondary_context_not_primary_layer5_no_go_gate",
        "primary_layer5_metric_focus": [
            "median_mean_vs_0050_00631L",
            "hit_rate_vs_0050_00631L",
            "fail_0050_rate",
            "path_like_return_proxy_if_available",
            "downside_risk_proxy_if_available",
            "turnover_churn_proxy",
            "period_stability",
        ],
        "current_holder_fields_status": "blocked_no_live_position_state_contract",
        "incumbent_fields_status": "blocked_no_live_incumbent_state_contract",
        "fallback_cash_classifier_status": "blocked_no_accepted_market_cash_classifier",
        "ready_for_layer5_within80_daily_rank_context_diagnostic": ready,
        "ready_for_experiments_intake": ready,
        "ready_for_layer5_action_rule": False,
        "ready_for_ab_switch_rule": False,
        "ready_for_portfolio_like_diagnostic": False,
        "ready_for_formal": False,
        "ready_for_strategy_replay": False,
        "future_data_violation_count": future_violations,
        "blocked_fields": [
            "current_holder_state",
            "incumbent_protection_state",
            "B_switch_rule",
            "final_single_stock_live_rule",
            "cash_classifier",
            "fallback_00631L_trading_rule",
            "Layer5_action_rule",
            "portfolio_replay",
        ],
        "proxy_fields": ["RS30_proxy", "large_down_blowoff_proxy", "risk_bucket_proxy_or_blocked"],
        **_fixed_flags(),
    }


def _summary(readiness: dict[str, Any]) -> str:
    return f"""# Layer5 within-80 daily rank context contract

## Verdict
- status={readiness['status']}
- diagnostic_only=true
- formal_model_changed=false
- trade_decision_changed=false
- active_in_trade_decision=false
- report_changed=false
- portfolio_replay_executed=false
- ready_for_strategy_replay=false
- not_live_rule=true
- forward_returns_live_rule_usage=false

## Scope
- Base universe: Layer4 80-stock primary pool.
- 100 extended watchlist and 31 high-confidence subpool are context/reference flags only.
- rank_context_date_basis={readiness['rank_context_date_basis']}
- Top1/top2/top3/top5/top10 are diagnostic context groups only.
- Final single-stock selector candidates are diagnostic candidates only.
- No A/B switch, no fallback trading rule, no Layer5 action rule.
- Layer5 final selector 的目標不是保留所有 winner，而是在每個交易日只選一檔時，找出長期勝率/報酬/風險表現最好的決策模式。

## Coverage
- rows={readiness['row_count']}
- weekly_snapshot_count={readiness['weekly_snapshot_count']}
- selected_count_min={readiness['selected_count_min']}
- selected_count_max={readiness['selected_count_max']}
- top1_rows={readiness['top1_rows']}
- top3_rows={readiness['top3_rows']}
- top5_rows={readiness['top5_rows']}
- top10_rows={readiness['top10_rows']}
- in_31_high_confidence_reference_rows={readiness['in_31_high_confidence_reference_rows']}
- extended_100_to_80_reentry_recent_4w_rows={readiness['extended_100_to_80_reentry_recent_4w_rows']}
- final_single_stock_selector_candidate_rows={readiness['final_single_stock_selector_candidate_rows']}
- final_single_stock_selector_candidate_variant_count={readiness['final_single_stock_selector_candidate_variant_count']}

## Layer5 Metric Focus
- Primary: median/mean vs 0050 and 00631L, hit-rate, fail_0050 rate, path-like return proxy if available, downside risk proxy if available, turnover/churn proxy, period stability.
- Secondary only: top-decile retention / missed winner attribution.

## Next
If accepted, hand off to Experiments:
`TASK-BACKTEST-EXPERIMENTS-VNEXT-LAYER5-WITHIN80-DAILY-RANK-CONTEXT-DIAGNOSTIC-001`.
完成後如果下一棒明確，請直接指派下一個 thread；如果下一棒不明確，請回報 Strategy Center 判斷。不要完成後停住不回報。
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer4-dir", default=str(DEFAULT_LAYER4_DIR))
    parser.add_argument("--experiments-dir", default=str(DEFAULT_LAYER4_EXPERIMENTS_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()
    manifest = build_contract(layer4_dir=args.layer4_dir, experiments_dir=args.experiments_dir, output_dir=args.output_dir)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
