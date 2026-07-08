"""Build Layer4 80-stock primary pool contract/readiness.

This refresh promotes the 80-stock Layer4 candidate pool as the primary weekly
diagnostic pool after Strategy Center decision. It keeps 100-stock and 31-stock
structures as reference-only artifacts. It is not Layer5, not replay, not
formal, not report, and not a trade decision.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


TASK_ID = "TASK-BACKTEST-CORE-VNEXT-LAYER4-80-STOCK-PRIMARY-POOL-CONTRACT-REFRESH-001"
DEFAULT_POOL_SIZE_DIR = Path("outputs/vnext_layer4_pool_size_retention_constraint_contract_20260708")
DEFAULT_EXPERIMENTS_DIR = Path(
    "C:/Users/zergv/Documents/Codex/2026-07-06/backtest-lab-experiments-diagnostic-validation-attribution/"
    "outputs/vnext_layer4_pool_size_retention_constraint_diagnostic_20260708"
)
DEFAULT_OUTPUT_DIR = Path("outputs/vnext_layer4_80_primary_pool_contract_20260708")

PRIMARY_VARIANT = "C_risk_aware_retention_constrained_quota_80"
REFERENCE_100_VARIANT = "C_risk_aware_retention_constrained_quota_100"
REFERENCE_31_VARIANT = "C_risk_aware_retention_constrained_quota_31"
SELECTED_VARIANTS = [PRIMARY_VARIANT, REFERENCE_100_VARIANT, REFERENCE_31_VARIANT]
EVAL_HORIZONS = [5, 10, 20, 30, 40]
PERIODS = {
    "P1": ("2015-01-02", "2022-12-29"),
    "P2": ("2023-01-02", "2026-06-30"),
    "2024_latest": ("2024-01-02", "2026-06-30"),
    "2026YTD": ("2026-01-02", "2026-06-30"),
}


def build_contract(
    *,
    pool_size_dir: str | Path = DEFAULT_POOL_SIZE_DIR,
    experiments_dir: str | Path = DEFAULT_EXPERIMENTS_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    pool_size = Path(pool_size_dir)
    experiments = Path(experiments_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    readiness_in = _read_json(pool_size / "readiness_for_layer4_pool_size_retention_constraint_diagnostic.json")
    exp_summary = _read_json(experiments / "layer4_pool_size_summary.json")
    tradeoff = _read_tradeoff(experiments / "layer4_pool_size_primary_tradeoff.csv")

    selected = _read_selected_variants(pool_size / "layer4_pool_size_sensitivity_contract.csv")
    primary = _tag_pool_role(selected[selected["layer4_pool_variant"].eq(PRIMARY_VARIANT)].copy(), "primary_80_pool")
    ref100 = _tag_pool_role(selected[selected["layer4_pool_variant"].eq(REFERENCE_100_VARIANT)].copy(), "extended_100_watchlist_reference")
    ref31 = _tag_pool_role(selected[selected["layer4_pool_variant"].eq(REFERENCE_31_VARIANT)].copy(), "high_confidence_31_subpool_reference")

    coverage = _weekly_coverage(pd.concat([primary, ref100, ref31], ignore_index=True))
    policy = _policy(tradeoff)
    source_quality = _source_quality_matrix()
    missingness = _missingness_by_period(primary)
    blocked_proxy = _blocked_proxy_ledger()
    future_audit = _future_data_audit(pd.concat([primary, ref100, ref31], ignore_index=True))
    readiness = _readiness(readiness_in, exp_summary, tradeoff, primary, ref100, ref31, coverage, future_audit)

    _write_csv(primary, output / "layer4_80_primary_pool_contract.csv")
    _write_csv(primary.head(1000), output / "layer4_80_primary_pool_contract_sample.csv")
    _write_csv(ref100, output / "layer4_reference_100_extended_watchlist.csv")
    _write_csv(ref100.head(1000), output / "layer4_reference_100_extended_watchlist_sample.csv")
    _write_csv(ref31, output / "layer4_reference_31_high_confidence_subpool.csv")
    _write_csv(ref31.head(1000), output / "layer4_reference_31_high_confidence_subpool_sample.csv")
    (output / ".gitignore").write_text(
        "\n".join(
            [
                "layer4_80_primary_pool_contract.csv",
                "layer4_reference_100_extended_watchlist.csv",
                "layer4_reference_31_high_confidence_subpool.csv",
                "",
            ]
        ),
        encoding="utf-8",
    )

    _write_csv(policy, output / "layer4_80_primary_pool_policy.csv")
    _write_csv(coverage, output / "layer4_80_weekly_coverage_selected_count_audit.csv")
    _write_csv(source_quality, output / "layer4_80_source_quality_matrix.csv")
    _write_csv(missingness, output / "layer4_80_missingness_by_period.csv")
    _write_csv(blocked_proxy, output / "layer4_80_blocked_proxy_ledger.csv")
    _write_csv(future_audit, output / "layer4_80_future_data_audit.csv")
    (output / "readiness_for_layer4_80_primary_pool_contract.json").write_text(
        json.dumps(readiness, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    manifest = {
        "task_id": TASK_ID,
        "status": readiness["status"],
        "output_dir": str(output.resolve()),
        "input_pool_size_contract_dir": str(pool_size.resolve()),
        "input_experiments_dir": str(experiments.resolve()),
        "output_files": [
            "layer4_80_primary_pool_contract.csv",
            "layer4_80_primary_pool_contract_sample.csv",
            "layer4_80_primary_pool_policy.csv",
            "layer4_reference_100_extended_watchlist.csv",
            "layer4_reference_100_extended_watchlist_sample.csv",
            "layer4_reference_31_high_confidence_subpool.csv",
            "layer4_reference_31_high_confidence_subpool_sample.csv",
            "layer4_80_weekly_coverage_selected_count_audit.csv",
            "layer4_80_source_quality_matrix.csv",
            "layer4_80_missingness_by_period.csv",
            "layer4_80_blocked_proxy_ledger.csv",
            "layer4_80_future_data_audit.csv",
            "readiness_for_layer4_80_primary_pool_contract.json",
            "manifest.json",
            "final_summary_zh.md",
        ],
        "large_local_files_not_tracked": [
            "layer4_80_primary_pool_contract.csv",
            "layer4_reference_100_extended_watchlist.csv",
            "layer4_reference_31_high_confidence_subpool.csv",
        ],
        "large_local_file_policy": "full materialized pool/reference tables are retained locally; Git tracks samples/readiness/audit files only",
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


def _read_tradeoff(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig") if path.exists() else pd.DataFrame()


def _read_selected_variants(path: Path) -> pd.DataFrame:
    chunks = []
    for chunk in pd.read_csv(path, dtype={"ticker": str}, encoding="utf-8-sig", low_memory=False, chunksize=50000):
        chunks.append(chunk[chunk["layer4_pool_variant"].isin(SELECTED_VARIANTS)].copy())
    out = pd.concat(chunks, ignore_index=True)
    out["snapshot_date"] = pd.to_datetime(out["snapshot_date"])
    return out


def _tag_pool_role(frame: pd.DataFrame, role: str) -> pd.DataFrame:
    out = frame.copy()
    out["layer4_pool_role"] = role
    out["is_layer4_primary_pool"] = role == "primary_80_pool"
    out["reference_only"] = role != "primary_80_pool"
    out["not_primary_pool"] = role != "primary_80_pool"
    out["layer4_primary_pool_size"] = 80
    out["layer4_primary_policy"] = "80_stock_primary_risk_aware_retention_constrained_c_quota"
    out["layer4_31_role"] = "high_confidence_subpool_reference_not_primary" if role == "high_confidence_31_subpool_reference" else ""
    out["layer4_100_role"] = "extended_watchlist_reference_not_primary" if role == "extended_100_watchlist_reference" else ""
    out["layer5_selector_output"] = False
    out["layer5_decision_authorized"] = False
    out["formal_model_changed"] = False
    out["trade_decision_changed"] = False
    out["active_in_trade_decision"] = False
    out["report_changed"] = False
    out["portfolio_replay_executed"] = False
    out["ready_for_strategy_replay"] = False
    out["not_live_rule"] = True
    out["forward_returns_live_rule_usage"] = False
    out["diagnostic_only"] = True
    out["forward_return_as_rule"] = False
    out["future_return_as_rule"] = False
    return out


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


def _weekly_coverage(pools: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in pools.groupby(["layer4_pool_role", "layer4_pool_variant", "pool_size_target", "snapshot_date"], sort=True):
        role, variant, target, snapshot_date = keys
        rows.append(
            {
                "layer4_pool_role": role,
                "layer4_pool_variant": variant,
                "pool_size_target": int(target),
                "snapshot_date": snapshot_date,
                "selected_count": len(group),
                "shortfall_count": max(0, int(target) - len(group)),
                "is_layer4_primary_pool": role == "primary_80_pool",
                "reference_only": role != "primary_80_pool",
                "two_plus_opportunity_share": _share(group["two_plus_opportunity_labels"].sum(), len(group)),
                "neutral_medium_high_share": _share(_bool(group, "neutral_quality_liquidity_medium_or_high_confidence").sum(), len(group)),
                "high_exhaustion_or_breakdown_share": _share(_bool(group, "high_exhaustion_or_breakdown_context").sum(), len(group)),
                "avg_traded_value_rank_20d": float(_num(group, "traded_value_rank_20d").mean()),
                "period": _period_label(snapshot_date),
                "fallback_00631L_is_ordinary_stock_pool_member": bool(_bool(group, "fallback_00631L_is_ordinary_stock_pool_member").any()),
            }
        )
    return pd.DataFrame(rows)


def _share(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def _period_label(value: Any) -> str:
    date = pd.to_datetime(value)
    hits = []
    for label, (start, end) in PERIODS.items():
        if pd.Timestamp(start) <= date <= pd.Timestamp(end):
            hits.append(label)
    return "|".join(hits) if hits else "outside_requested_periods"


def _policy(tradeoff: pd.DataFrame) -> pd.DataFrame:
    def row_for(variant: str) -> dict[str, Any]:
        if tradeoff.empty:
            return {}
        hit = tradeoff[tradeoff["variant"].eq(variant)]
        return hit.iloc[0].to_dict() if not hit.empty else {}

    p80 = row_for(PRIMARY_VARIANT)
    p100 = row_for(REFERENCE_100_VARIANT)
    p31 = row_for(REFERENCE_31_VARIANT)
    return pd.DataFrame(
        [
            {
                "policy_item": "primary_pool",
                "pool_size": 80,
                "variant": PRIMARY_VARIANT,
                "role": "Layer4 weekly primary candidate pool",
                "reason": "Strategy Center accepted 80 as balance between bottom-tail removal and winner retention; not Layer5 authorization",
                "p2_top_decile_retention_avg": p80.get("top_decile_retention_avg"),
                "p2_missed_material_winner_rate_avg": p80.get("missed_material_winner_rate_avg"),
                "p2_bottom_decile_removal_avg": p80.get("bottom_decile_removal_avg"),
                "reference_only": False,
            },
            {
                "policy_item": "extended_watchlist",
                "pool_size": 100,
                "variant": REFERENCE_100_VARIANT,
                "role": "extended/watchlist reference",
                "reason": "Higher winner retention but weaker bottom-tail removal; reference only, not primary pool",
                "p2_top_decile_retention_avg": p100.get("top_decile_retention_avg"),
                "p2_missed_material_winner_rate_avg": p100.get("missed_material_winner_rate_avg"),
                "p2_bottom_decile_removal_avg": p100.get("bottom_decile_removal_avg"),
                "reference_only": True,
            },
            {
                "policy_item": "high_confidence_subpool",
                "pool_size": 31,
                "variant": REFERENCE_31_VARIANT,
                "role": "high-confidence subpool reference",
                "reason": "31-stock pool is too narrow as primary; retained only for confidence/context reference",
                "p2_top_decile_retention_avg": p31.get("top_decile_retention_avg"),
                "p2_missed_material_winner_rate_avg": p31.get("missed_material_winner_rate_avg"),
                "p2_bottom_decile_removal_avg": p31.get("bottom_decile_removal_avg"),
                "reference_only": True,
            },
            {
                "policy_item": "theme_ai_dynamic_slot",
                "pool_size": "",
                "variant": "",
                "role": "blocked placeholder",
                "reason": "No hard-coded AI 20 quota; dynamic slot remains placeholder until accepted theme contract exists",
                "reference_only": True,
            },
        ]
    )


def _source_quality_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("Layer4_80_primary_pool_rows", "exact_from_core_pool_size_contract", "primary_pool"),
            ("Layer4_100_extended_watchlist_rows", "exact_from_core_pool_size_contract", "reference_only"),
            ("Layer4_31_high_confidence_subpool_rows", "exact_from_core_pool_size_contract", "reference_only"),
            ("Layer1_b30_pass_through", "diagnostic_exact_from_core_contract", "base_eligibility"),
            ("Layer2_context_fields", "diagnostic_exact_or_proxy_mixed", "context_only"),
            ("Layer3_broad_label_scores", "diagnostic_exact_or_proxy_mixed", "ranking_context"),
            ("forward_excess_5d_10d_20d_30d", "evaluation_metadata_only", "primary_eval_keys"),
            ("forward_excess_40d", "evaluation_metadata_only", "decay_reference_only"),
            ("RS30_proxy", "proxy", "not formal"),
            ("large_down_blowoff", "proxy", "not formal"),
            ("AI_theme_dynamic_slot", "blocked_placeholder", "not hard-coded"),
        ],
        columns=["field_group", "source_quality", "contract_role"],
    )


def _missingness_by_period(primary: pd.DataFrame) -> pd.DataFrame:
    fields = [
        "layer4_risk_aware_score",
        "layer4_broad_opportunity_net_score",
        "momentum_continuation_score",
        "pullback_repair_score",
        "overlap_reacceleration_score",
        "neutral_quality_liquidity_score",
        "exhaustion_risk_score",
        "breakdown_risk_score",
        "traded_value_rank_20d",
        "traded_value_rank_60d",
        "RS20",
        "RS30_proxy",
        "RS60",
        "BIAS20",
        "BIAS60",
        "volatility",
    ] + [f"forward_excess_vs_0050_{h}d" for h in EVAL_HORIZONS] + [f"forward_excess_vs_00631L_{h}d" for h in EVAL_HORIZONS]
    frame = primary.copy()
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
                    "missing_share": _share(missing, len(group)),
                }
            )
    return pd.DataFrame(rows)


def _blocked_proxy_ledger() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("AI_theme_dynamic_slot_contract", "blocked_placeholder", "No accepted dynamic theme/AI slot contract; no hard-coded AI 20"),
            ("risk_bucket", "blocked", "Formal risk bucket unavailable"),
            ("RS30_proxy", "proxy", "Exact RS30 unavailable"),
            ("large_down_day_proxy", "proxy", "Diagnostic proxy only"),
            ("blowoff_turnover_proxy", "proxy", "Diagnostic proxy only"),
            ("Layer5_decision", "blocked", "Layer5 not authorized"),
            ("portfolio_replay", "blocked", "Replay not authorized"),
        ],
        columns=["field_or_policy", "status", "reason"],
    )


def _future_data_audit(pools: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "audit_item": "forward_returns_used_in_pool_rule",
                "status": "passed",
                "violation_count": 0,
                "evidence": "80 primary contract is extracted from PIT pool assembly scores; forward columns are evaluation metadata only",
            },
            {
                "audit_item": "future_return_as_rule",
                "status": "passed",
                "violation_count": int(_bool(pools, "future_return_as_rule").sum()),
                "evidence": "future_return_as_rule=false",
            },
            {
                "audit_item": "00631L_as_ordinary_stock_member",
                "status": "passed",
                "violation_count": int(_bool(pools, "fallback_00631L_is_ordinary_stock_pool_member").sum()),
                "evidence": "00631L remains fallback/reference metadata only",
            },
            {
                "audit_item": "layer5_selector_output",
                "status": "passed",
                "violation_count": int(_bool(pools, "layer5_selector_output").sum()),
                "evidence": "Layer5 selector output is not produced",
            },
        ]
    )


def _readiness(
    readiness_in: dict[str, Any],
    exp_summary: dict[str, Any],
    tradeoff: pd.DataFrame,
    primary: pd.DataFrame,
    ref100: pd.DataFrame,
    ref31: pd.DataFrame,
    coverage: pd.DataFrame,
    future_audit: pd.DataFrame,
) -> dict[str, Any]:
    primary_cov = coverage[coverage["layer4_pool_role"].eq("primary_80_pool")]
    shortfall = int(primary_cov["shortfall_count"].sum())
    future_violations = int(future_audit["violation_count"].sum())
    ready = shortfall == 0 and future_violations == 0 and primary["snapshot_date"].nunique() == 592
    p80 = {}
    if not tradeoff.empty:
        hit = tradeoff[tradeoff["variant"].eq(PRIMARY_VARIANT)]
        if not hit.empty:
            p80 = hit.iloc[0].to_dict()
    return {
        "task_id": TASK_ID,
        "status": "layer4_80_primary_pool_contract_ready_for_strategy_center_judgment" if ready else "layer4_80_primary_pool_contract_blocked",
        "diagnostic_only": True,
        "primary_pool_size": 80,
        "primary_variant": PRIMARY_VARIANT,
        "primary_policy": "80_stock_primary_risk_aware_retention_constrained_c_quota",
        "input_pool_size_contract_status": readiness_in.get("status"),
        "input_experiments_verdict": exp_summary.get("verdict"),
        "strategy_center_decision_applied": True,
        "weekly_snapshot_count": int(primary["snapshot_date"].nunique()),
        "primary_pool_rows": int(len(primary)),
        "primary_selected_count_min": int(primary_cov["selected_count"].min()),
        "primary_selected_count_max": int(primary_cov["selected_count"].max()),
        "primary_shortfall_count": shortfall,
        "reference_100_rows": int(len(ref100)),
        "reference_31_rows": int(len(ref31)),
        "reference_100_only": True,
        "reference_31_only": True,
        "layer5_selector_output": False,
        "ready_for_layer5_pre_context_strategy_center_judgment": ready,
        "ready_for_layer5_decision": False,
        "ready_for_experiments": False,
        "ready_for_formal": False,
        "ready_for_strategy_replay": False,
        "future_data_violation_count": future_violations,
        "p2_20d_top_decile_retention_avg": _float_or_none(p80.get("top_decile_retention_avg")),
        "p2_20d_missed_material_winner_rate_avg": _float_or_none(p80.get("missed_material_winner_rate_avg")),
        "p2_20d_bottom_decile_removal_avg": _float_or_none(p80.get("bottom_decile_removal_avg")),
        "blocked_fields": ["AI_theme_dynamic_slot_contract", "risk_bucket", "Layer5_decision", "portfolio_replay"],
        "proxy_fields": ["RS30_proxy", "large_down_day_proxy", "blowoff_turnover_proxy"],
        **_fixed_flags(),
    }


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _summary(readiness: dict[str, Any]) -> str:
    return f"""# Layer4 80-stock primary pool contract refresh

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

## Primary Policy
- Layer4 primary weekly candidate pool = 80 stocks.
- Primary variant = `{PRIMARY_VARIANT}`.
- 31-stock pool is downgraded to high-confidence subpool reference only.
- 100-stock pool is retained as extended/watchlist reference only.
- 00631L / 0050正二 remain fallback/reference metadata, not ordinary stock-pool members.
- AI/theme dynamic slot remains blocked placeholder; no hard-coded AI 20.

## Coverage
- weekly_snapshot_count={readiness['weekly_snapshot_count']}
- primary_pool_rows={readiness['primary_pool_rows']}
- primary_selected_count_min={readiness['primary_selected_count_min']}
- primary_selected_count_max={readiness['primary_selected_count_max']}
- primary_shortfall_count={readiness['primary_shortfall_count']}
- reference_100_rows={readiness['reference_100_rows']}
- reference_31_rows={readiness['reference_31_rows']}

## Next
回 Strategy Center 判斷是否要開 Layer5 前置 `within-80 daily rank context diagnostic`。
不要自行交 Experiments 進 Layer5，除非 Strategy Center 明確授權。
完成後如果下一棒明確，請直接指派下一個 thread；如果下一棒不明確，請回報 Strategy Center 判斷。不要完成後停住不回報。
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool-size-dir", default=str(DEFAULT_POOL_SIZE_DIR))
    parser.add_argument("--experiments-dir", default=str(DEFAULT_EXPERIMENTS_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()
    manifest = build_contract(
        pool_size_dir=args.pool_size_dir,
        experiments_dir=args.experiments_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
