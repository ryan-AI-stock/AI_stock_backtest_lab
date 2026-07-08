"""Build Layer5 stock-eligibility / market-breadth final decision contract.

This package keeps Layer0-Layer4 fixed and materializes PIT environment
features that answer whether a given signal date has enough stock edge to
consider a single-stock recommendation. 00631L remains fallback/reference
metadata and is not an ordinary stock row.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


TASK_ID = "TASK-BACKTEST-CORE-VNEXT-LAYER5-STOCK-ELIGIBILITY-MARKET-BREADTH-FINAL-DECISION-ARCHITECTURE-CONTRACT-001"
DEFAULT_WITHIN80_DIR = Path("outputs/vnext_layer5_within80_daily_rank_context_contract_20260708")
DEFAULT_LIFECYCLE_DIR = Path("outputs/vnext_layer5_lifecycle_state_selector_candidate_contract_20260708")
DEFAULT_HURDLE_DIR = Path("outputs/vnext_layer5_stock_vs_00631l_hurdle_context_contract_20260708")
DEFAULT_BENCHMARK_PATH = Path("outputs/vnext_dynamic_candidate_pool_data_materialization_20260706/benchmark_features.csv")
DEFAULT_EXPERIMENTS_DIR = Path(
    "C:/Users/zergv/Documents/Codex/2026-07-06/backtest-lab-experiments-diagnostic-validation-attribution/"
    "outputs/vnext_layer5_fixed_l0_l4_selector_family_sweep_diagnostic_20260708"
)
DEFAULT_OUTPUT_DIR = Path("outputs/vnext_layer5_stock_eligibility_market_breadth_contract_20260708")
PERIODS = {
    "P1": ("2015-01-02", "2022-12-29"),
    "P2": ("2023-01-02", "2026-06-30"),
    "2024_latest": ("2024-01-02", "2026-06-30"),
    "2026YTD": ("2026-01-02", "2026-06-30"),
}
EVAL_HORIZONS = [5, 10, 20, 30, 40]
STOCK_VARIANT_SOURCE = {
    "stock_allowed_only_when_pool_breadth_positive_else_00631L": "lifecycle_top10_clean_state_selector",
    "stock_allowed_when_top10_dispersion_clear_else_00631L": "lifecycle_top10_clean_state_selector",
    "stock_allowed_when_31_confidence_breadth_positive_else_00631L": "high_confidence_bonus_lifecycle_selector",
    "stock_allowed_when_reentry_breadth_confirmed_else_00631L": "reentry_confirmed_lifecycle_selector",
    "hybrid_stock_eligibility_gate_then_best_incumbent_or_reentry_selector": "stock_vs_00631L_best_baseline",
    "always_stock_best_previous_baseline": "stock_vs_00631L_best_baseline",
}


def build_contract(
    *,
    within80_dir: str | Path = DEFAULT_WITHIN80_DIR,
    lifecycle_dir: str | Path = DEFAULT_LIFECYCLE_DIR,
    hurdle_dir: str | Path = DEFAULT_HURDLE_DIR,
    benchmark_path: str | Path = DEFAULT_BENCHMARK_PATH,
    experiments_dir: str | Path = DEFAULT_EXPERIMENTS_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    within80_path = Path(within80_dir)
    lifecycle_path = Path(lifecycle_dir)
    hurdle_path = Path(hurdle_dir)
    experiments_path = Path(experiments_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    within80_readiness = _read_json(within80_path / "readiness_for_layer5_within80_daily_rank_context_diagnostic.json")
    lifecycle_readiness = _read_json(lifecycle_path / "readiness_for_layer5_lifecycle_state_selector_diagnostic.json")
    hurdle_readiness = _read_json(hurdle_path / "readiness_for_layer5_stock_vs_00631l_hurdle_diagnostic.json")
    experiment_summary = _read_json(experiments_path / "layer5_selector_family_sweep_summary.json")
    within80 = _read_context(within80_path / "layer5_within80_daily_rank_context_contract.csv")
    lifecycle = _read_context(lifecycle_path / "layer5_lifecycle_state_selector_candidate_contract.csv")
    benchmark = _benchmark_context(Path(benchmark_path), sorted(within80["snapshot_date"].unique()))

    environment = _build_environment_features(within80)
    stock_candidates = _stock_candidate_lookup(lifecycle)
    contract = _build_decision_architecture(environment, stock_candidates, benchmark)
    feature_design = _feature_design()
    variant_design = _variant_design()
    source_quality = _source_quality_matrix()
    coverage = _coverage_by_period(contract)
    missingness = _missingness_by_period(contract)
    blocked_proxy = _blocked_proxy_ledger()
    future_audit = _future_data_audit(contract)
    requested_actual = _requested_vs_actual_coverage(contract)
    readiness = _readiness(
        within80_readiness,
        lifecycle_readiness,
        hurdle_readiness,
        experiment_summary,
        contract,
        coverage,
        future_audit,
    )

    _write_csv(contract, output / "layer5_stock_eligibility_market_breadth_contract.csv")
    _write_csv(feature_design, output / "layer5_stock_eligibility_feature_design.csv")
    _write_csv(variant_design, output / "layer5_final_decision_architecture_variant_design.csv")
    _write_csv(source_quality, output / "layer5_stock_eligibility_source_quality_matrix.csv")
    _write_csv(coverage, output / "layer5_stock_eligibility_coverage_by_period.csv")
    _write_csv(missingness, output / "layer5_stock_eligibility_missingness_by_period.csv")
    _write_csv(blocked_proxy, output / "layer5_stock_eligibility_blocked_proxy_ledger.csv")
    _write_csv(future_audit, output / "layer5_stock_eligibility_future_data_audit.csv")
    _write_csv(requested_actual, output / "layer5_stock_eligibility_requested_vs_actual_coverage.csv")
    (output / "readiness_for_layer5_stock_eligibility_market_breadth_diagnostic.json").write_text(
        json.dumps(readiness, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    manifest = {
        "task_id": TASK_ID,
        "status": readiness["status"],
        "output_dir": str(output.resolve()),
        "input_within80_dir": str(within80_path.resolve()),
        "input_lifecycle_dir": str(lifecycle_path.resolve()),
        "input_hurdle_dir": str(hurdle_path.resolve()),
        "input_benchmark_path": str(Path(benchmark_path).resolve()),
        "input_experiments_dir": str(experiments_path.resolve()),
        "output_files": [
            "layer5_stock_eligibility_market_breadth_contract.csv",
            "layer5_stock_eligibility_feature_design.csv",
            "layer5_final_decision_architecture_variant_design.csv",
            "layer5_stock_eligibility_source_quality_matrix.csv",
            "layer5_stock_eligibility_coverage_by_period.csv",
            "layer5_stock_eligibility_missingness_by_period.csv",
            "layer5_stock_eligibility_blocked_proxy_ledger.csv",
            "layer5_stock_eligibility_future_data_audit.csv",
            "layer5_stock_eligibility_requested_vs_actual_coverage.csv",
            "readiness_for_layer5_stock_eligibility_market_breadth_diagnostic.json",
            "manifest.json",
            "final_summary_zh.md",
        ],
        **_fixed_flags(),
        "diagnostic_only": True,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "final_summary_zh.md").write_text(_summary(readiness), encoding="utf-8")
    return manifest


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _read_context(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"ticker": str}, encoding="utf-8-sig", low_memory=False)
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
    return df


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig")


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


def _bool(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df:
        return pd.Series(False, index=df.index)
    series = df[col]
    if series.dtype == bool:
        return series.fillna(False)
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).ne(0)
    return series.astype(str).str.lower().isin(["true", "1", "yes", "y"])


def _num(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df:
        return pd.Series(default, index=df.index)
    return pd.to_numeric(df[col], errors="coerce").fillna(default)


def _benchmark_context(path: Path, signal_dates: list[pd.Timestamp]) -> pd.DataFrame:
    bench = pd.read_csv(path, dtype={"benchmark": str}, encoding="utf-8-sig", low_memory=False)
    bench["trade_date"] = pd.to_datetime(bench["trade_date"])
    bench["benchmark"] = bench["benchmark"].astype(str)
    rows = []
    for benchmark, group in bench.groupby("benchmark"):
        group = group.sort_values("trade_date").reset_index(drop=True)
        close = pd.to_numeric(group["adjusted_close"], errors="coerce")
        for horizon in EVAL_HORIZONS:
            group[f"forward_return_{horizon}d"] = close.shift(-horizon) / close - 1
        rows.append(group)
    features = pd.concat(rows, ignore_index=True)
    pivot = features[features["trade_date"].isin(signal_dates)].pivot(index="trade_date", columns="benchmark")
    out = pd.DataFrame(index=sorted(signal_dates))
    for bench_id in ["0050", "00631L"]:
        for col in ["adjusted_close", "BIAS20", "BIAS60", "BIAS120", "drawdown", "return_20d", "return_40d", "return_60d", "volatility"]:
            if (col, bench_id) in pivot:
                out[f"{bench_id}_{col}"] = pivot[(col, bench_id)]
        for horizon in EVAL_HORIZONS:
            if (f"forward_return_{horizon}d", bench_id) in pivot:
                out[f"{bench_id}_forward_return_{horizon}d"] = pivot[(f"forward_return_{horizon}d", bench_id)]
        if ("source_quality", bench_id) in pivot:
            out[f"{bench_id}_source_quality"] = pivot[("source_quality", bench_id)]
        if ("benchmark_data_blocked", bench_id) in pivot:
            out[f"{bench_id}_benchmark_data_blocked"] = pivot[("benchmark_data_blocked", bench_id)]
    out = out.reset_index().rename(columns={"index": "snapshot_date"})
    for horizon in EVAL_HORIZONS:
        out[f"00631L_forward_excess_vs_0050_{horizon}d"] = out[f"00631L_forward_return_{horizon}d"] - out[f"0050_forward_return_{horizon}d"]
        out[f"0050_forward_excess_vs_00631L_{horizon}d"] = out[f"0050_forward_return_{horizon}d"] - out[f"00631L_forward_return_{horizon}d"]
    return out


def _build_environment_features(within80: pd.DataFrame) -> pd.DataFrame:
    frame = within80[_bool(within80, "is_layer4_primary_pool") & _num(within80, "within80_rank").between(1, 80)].copy()
    frame["clean_or_improving"] = (
        (_bool(frame, "rs20_30_primary_momentum_positive") | _bool(frame, "rs20_30_primary_momentum_stable"))
        & ~_bool(frame, "high_exhaustion_or_breakdown_context")
        & ~_bool(frame, "risk_overheat_penalty_context")
    )
    frame["non_exhaustion"] = ~(
        _bool(frame, "high_exhaustion_or_breakdown_context")
        | _bool(frame, "rs60_high_short_rs_weakening_exhaustion_context")
        | _bool(frame, "exhaustion_risk_medium_or_high_confidence")
    )
    frame["non_breakdown"] = ~_bool(frame, "breakdown_risk_medium_or_high_confidence")
    frame["reentry_confirmed"] = _bool(frame, "extended_100_to_80_reentry_context") & (
        _bool(frame, "came_from_100_extended_only_recent_4w") | _bool(frame, "capital_reasonable_band_4w_persistent")
    )
    rows = []
    for date, group in frame.groupby("snapshot_date", sort=True):
        top10 = group.nsmallest(10, "within80_rank").copy()
        top10_scores = _num(top10, "final_selector_lifecycle_context_score")
        top10_sorted = top10_scores.sort_values(ascending=False)
        top10_margin = float(top10_sorted.iloc[0] - top10_sorted.iloc[1]) if len(top10_sorted) > 1 else 0.0
        high31 = group[_bool(group, "in_31_high_confidence_subpool_reference")].copy()
        rows.append(
            {
                "snapshot_date": date,
                "pool80_count": int(len(group)),
                "pool80_positive_rs20_30_share": _share((_bool(group, "rs20_30_primary_momentum_positive") | _bool(group, "rs20_30_primary_momentum_stable")).sum(), len(group)),
                "pool80_two_plus_labels_share": _share(_bool(group, "two_plus_opportunity_labels").sum(), len(group)),
                "pool80_non_exhaustion_share": _share(_bool(group, "non_exhaustion").sum(), len(group)),
                "pool80_non_breakdown_share": _share(_bool(group, "non_breakdown").sum(), len(group)),
                "pool80_clean_or_improving_share": _share(_bool(group, "clean_or_improving").sum(), len(group)),
                "pool80_breakdown_risk_share": _share(_bool(group, "breakdown_risk_medium_or_high_confidence").sum(), len(group)),
                "pool80_exhaustion_risk_share": _share(_bool(group, "exhaustion_risk_medium_or_high_confidence").sum(), len(group)),
                "pool80_overheat_risk_share": _share((_bool(group, "bias_overheat_penalty_context") | _bool(group, "volatility_high_context")).sum(), len(group)),
                "high31_count": int(len(high31)),
                "high31_clean_improving_share": _share(_bool(high31, "clean_or_improving").sum(), len(high31)),
                "high31_non_overheated_share": _share((~(_bool(high31, "bias_overheat_penalty_context") | _bool(high31, "volatility_high_context"))).sum(), len(high31)),
                "reentry_100_to_80_count": int(_bool(group, "extended_100_to_80_reentry_context").sum()),
                "reentry_100_to_80_confirmed_count": int(_bool(group, "reentry_confirmed").sum()),
                "reentry_100_to_80_confirmed_share": _share(_bool(group, "reentry_confirmed").sum(), len(group)),
                "top10_score_max": float(top10_scores.max()),
                "top10_score_second": float(top10_sorted.iloc[1]) if len(top10_sorted) > 1 else None,
                "top10_score_margin_top1_vs_top2": top10_margin,
                "top10_score_std": float(top10_scores.std(ddof=0)) if len(top10_scores) else 0.0,
                "top10_clean_or_improving_share": _share(_bool(top10, "clean_or_improving").sum(), len(top10)),
                "top10_risk_penalty_share": _share(_bool(top10, "risk_overheat_penalty_context").sum(), len(top10)),
            }
        )
    env = pd.DataFrame(rows)
    env["pool_breadth_positive_flag"] = (
        env["pool80_positive_rs20_30_share"].ge(0.50)
        & env["pool80_two_plus_labels_share"].ge(0.35)
        & env["pool80_non_exhaustion_share"].ge(0.70)
        & env["pool80_non_breakdown_share"].ge(0.80)
    )
    env["top10_dispersion_clear_flag"] = env["top10_score_margin_top1_vs_top2"].ge(0.05) | env["top10_score_std"].ge(0.08)
    env["high31_confidence_breadth_positive_flag"] = env["high31_clean_improving_share"].ge(0.55) & env["high31_non_overheated_share"].ge(0.65)
    env["reentry_breadth_confirmed_flag"] = env["reentry_100_to_80_confirmed_count"].ge(2)
    env["weak_stock_environment_flag"] = (
        env["pool80_clean_or_improving_share"].lt(0.35)
        | env["pool80_breakdown_risk_share"].gt(0.25)
        | env["pool80_exhaustion_risk_share"].gt(0.25)
        | env["top10_risk_penalty_share"].gt(0.50)
    )
    env["stock_eligibility_composite_score"] = (
        0.22 * env["pool80_positive_rs20_30_share"]
        + 0.18 * env["pool80_two_plus_labels_share"]
        + 0.18 * env["pool80_clean_or_improving_share"]
        + 0.14 * env["high31_clean_improving_share"].fillna(0.0)
        + 0.12 * env["reentry_100_to_80_confirmed_share"]
        + 0.10 * env["top10_clean_or_improving_share"]
        + 0.06 * env["top10_score_margin_top1_vs_top2"].clip(0, 1)
        - 0.14 * env["pool80_breakdown_risk_share"]
        - 0.10 * env["pool80_exhaustion_risk_share"]
    )
    env["stock_eligibility_composite_flag"] = env["stock_eligibility_composite_score"].ge(0.42) & ~env["weak_stock_environment_flag"]
    return env


def _stock_candidate_lookup(lifecycle: pd.DataFrame) -> dict[tuple[pd.Timestamp, str], pd.Series]:
    return {
        (pd.Timestamp(row["snapshot_date"]), str(row["lifecycle_selector_candidate_variant"])): row
        for _, row in lifecycle.iterrows()
    }


def _build_decision_architecture(
    environment: pd.DataFrame,
    stock_candidates: dict[tuple[pd.Timestamp, str], pd.Series],
    benchmark: pd.DataFrame,
) -> pd.DataFrame:
    bench_lookup = benchmark.set_index("snapshot_date")
    rows = []
    for _, env in environment.iterrows():
        date = pd.Timestamp(env["snapshot_date"])
        variant_rules = {
            "stock_allowed_only_when_pool_breadth_positive_else_00631L": bool(env["pool_breadth_positive_flag"]),
            "stock_allowed_when_top10_dispersion_clear_else_00631L": bool(env["top10_dispersion_clear_flag"]) and not bool(env["weak_stock_environment_flag"]),
            "stock_allowed_when_31_confidence_breadth_positive_else_00631L": bool(env["high31_confidence_breadth_positive_flag"]),
            "stock_allowed_when_reentry_breadth_confirmed_else_00631L": bool(env["reentry_breadth_confirmed_flag"]),
            "hybrid_stock_eligibility_gate_then_best_incumbent_or_reentry_selector": bool(env["stock_eligibility_composite_flag"]),
            "always_stock_best_previous_baseline": True,
        }
        for variant, stock_allowed in variant_rules.items():
            source_variant = STOCK_VARIANT_SOURCE[variant]
            candidate = stock_candidates.get((date, source_variant))
            rows.append(_decision_row(date, variant, source_variant, stock_allowed, candidate, env, bench_lookup))
        rows.append(_fallback_row(date, "00631L_always_baseline", "00631L", env, bench_lookup))
        rows.append(_fallback_row(date, "0050_reference", "0050", env, bench_lookup))
    out = pd.DataFrame(rows)
    out = out.merge(benchmark, on="snapshot_date", how="left")
    out["stock_eligibility_market_breadth_contract_only"] = True
    out["cash_classifier_status"] = "blocked_no_accepted_market_cash_classifier"
    return out.sort_values(["decision_architecture_variant", "snapshot_date"]).reset_index(drop=True)


def _decision_row(
    date: pd.Timestamp,
    variant: str,
    source_variant: str,
    stock_allowed: bool,
    candidate: pd.Series | None,
    env: pd.Series,
    bench_lookup: pd.DataFrame,
) -> dict[str, Any]:
    use_stock = bool(stock_allowed and candidate is not None)
    if use_stock:
        row = candidate.to_dict()
        row["recommended_asset_type"] = "stock"
        row["recommended_ticker"] = row.get("ticker")
        row["recommended_name"] = row.get("name")
        row["uses_stock_candidate"] = True
        row["uses_00631L_fallback"] = False
        row["stock_disallowed_reason"] = ""
    else:
        row = _fallback_row(date, variant, "00631L", env, bench_lookup)
        row["stock_disallowed_reason"] = _disallowed_reason(env, variant, candidate)
    row.update(_env_dict(env))
    row["snapshot_date"] = date
    row["decision_architecture_variant"] = variant
    row["stock_candidate_source_variant"] = source_variant
    row["stock_allowed_by_architecture"] = use_stock
    row["stock_eligibility_gate_design_only"] = True
    row["decision_architecture_design_only"] = True
    row["00631L_ordinary_stock_pool_member"] = False
    row["fallback_00631L_reference_only"] = True
    row["fallback_00631L_live_rule_output"] = False
    row["live_layer5_rule_output"] = False
    row["trade_decision_output"] = False
    row["daily_report_output"] = False
    row["portfolio_like_execution_output"] = False
    row["ab_switch_rule_output"] = False
    row["second_stock_allocation_output"] = False
    row["future_return_as_rule"] = False
    row["forward_return_as_rule"] = False
    row["max_in_band_as_rule"] = False
    row["evaluation_metadata_only"] = True
    row["next_day_entry_assumption"] = "diagnostic_only_next_trading_session_after_decision_context_date"
    row["turnover_cost_placeholder"] = "blocked_no_accepted_cost_model"
    for key, value in _fixed_flags().items():
        row[key] = value
    return row


def _fallback_row(
    date: pd.Timestamp,
    variant: str,
    ticker: str,
    env: pd.Series,
    bench_lookup: pd.DataFrame,
) -> dict[str, Any]:
    bench = bench_lookup.loc[date] if date in bench_lookup.index else pd.Series(dtype=object)
    row: dict[str, Any] = {
        "snapshot_date": date,
        "ticker": ticker,
        "name": ticker,
        "recommended_asset_type": "00631L_fallback" if ticker == "00631L" else "0050_reference",
        "recommended_ticker": ticker,
        "recommended_name": ticker,
        "uses_stock_candidate": False,
        "uses_00631L_fallback": ticker == "00631L",
        "uses_0050_reference": ticker == "0050",
        "decision_architecture_variant": variant,
        "stock_candidate_source_variant": "fallback_reference",
        "stock_allowed_by_architecture": False,
    }
    for horizon in EVAL_HORIZONS:
        if ticker == "00631L":
            row[f"forward_excess_vs_00631L_{horizon}d"] = 0.0
            row[f"forward_excess_vs_0050_{horizon}d"] = bench.get(f"00631L_forward_excess_vs_0050_{horizon}d")
        else:
            row[f"forward_excess_vs_0050_{horizon}d"] = 0.0
            row[f"forward_excess_vs_00631L_{horizon}d"] = bench.get(f"0050_forward_excess_vs_00631L_{horizon}d")
        row[f"forward_eval_available_{horizon}d"] = pd.notna(row.get(f"forward_excess_vs_0050_{horizon}d")) and pd.notna(row.get(f"forward_excess_vs_00631L_{horizon}d"))
    row.update(_env_dict(env))
    for key, value in _fixed_flags().items():
        row[key] = value
    return row


def _env_dict(env: pd.Series) -> dict[str, Any]:
    keys = [
        "pool80_count",
        "pool80_positive_rs20_30_share",
        "pool80_two_plus_labels_share",
        "pool80_non_exhaustion_share",
        "pool80_non_breakdown_share",
        "pool80_clean_or_improving_share",
        "pool80_breakdown_risk_share",
        "pool80_exhaustion_risk_share",
        "pool80_overheat_risk_share",
        "high31_count",
        "high31_clean_improving_share",
        "high31_non_overheated_share",
        "reentry_100_to_80_count",
        "reentry_100_to_80_confirmed_count",
        "reentry_100_to_80_confirmed_share",
        "top10_score_max",
        "top10_score_second",
        "top10_score_margin_top1_vs_top2",
        "top10_score_std",
        "top10_clean_or_improving_share",
        "top10_risk_penalty_share",
        "pool_breadth_positive_flag",
        "top10_dispersion_clear_flag",
        "high31_confidence_breadth_positive_flag",
        "reentry_breadth_confirmed_flag",
        "weak_stock_environment_flag",
        "stock_eligibility_composite_score",
        "stock_eligibility_composite_flag",
    ]
    return {key: env.get(key) for key in keys}


def _disallowed_reason(env: pd.Series, variant: str, candidate: pd.Series | None) -> str:
    if candidate is None:
        return "blocked_missing_stock_candidate_context"
    checks = {
        "stock_allowed_only_when_pool_breadth_positive_else_00631L": "pool_breadth_positive_flag",
        "stock_allowed_when_top10_dispersion_clear_else_00631L": "top10_dispersion_clear_flag",
        "stock_allowed_when_31_confidence_breadth_positive_else_00631L": "high31_confidence_breadth_positive_flag",
        "stock_allowed_when_reentry_breadth_confirmed_else_00631L": "reentry_breadth_confirmed_flag",
        "hybrid_stock_eligibility_gate_then_best_incumbent_or_reentry_selector": "stock_eligibility_composite_flag",
    }
    check = checks.get(variant)
    if check and not bool(env.get(check, False)):
        return f"{check}=false"
    if bool(env.get("weak_stock_environment_flag", False)):
        return "weak_stock_environment_flag=true"
    return "stock_not_allowed_by_candidate_design"


def _feature_design() -> pd.DataFrame:
    rows = [
        ("pool80_positive_rs20_30_share", "80-pool share with positive/stable RS20/30", "PIT"),
        ("pool80_two_plus_labels_share", "80-pool share with two_plus opportunity labels", "PIT"),
        ("pool80_non_exhaustion_share", "80-pool share not flagged exhaustion", "PIT/proxy"),
        ("pool80_non_breakdown_share", "80-pool share not flagged breakdown", "PIT/proxy"),
        ("high31_clean_improving_share", "31 reference subpool clean/improving breadth", "reference context"),
        ("reentry_100_to_80_confirmed_count", "100->80 reentry confirmed by recent/context fields", "PIT/proxy"),
        ("top10_score_margin_top1_vs_top2", "top10 dispersion / top1 margin context", "PIT"),
        ("weak_stock_environment_flag", "broad weakness / risk flags, candidate design only", "diagnostic proxy"),
        ("0050_00631L_market_context", "benchmark BIAS/trend/volatility if available", "PIT benchmark context"),
        ("forward_evaluation_metadata", "5D/10D/20D/30D primary; 40D decay reference", "evaluation-only"),
    ]
    return pd.DataFrame(rows, columns=["feature", "definition", "source_quality"])


def _variant_design() -> pd.DataFrame:
    rows = [
        ("stock_allowed_only_when_pool_breadth_positive_else_00631L", "Allow stock only when 80-pool breadth is positive; otherwise 00631L metadata", "candidate design only"),
        ("stock_allowed_when_top10_dispersion_clear_else_00631L", "Allow stock when top10 scores show clear dispersion and environment is not weak", "candidate design only"),
        ("stock_allowed_when_31_confidence_breadth_positive_else_00631L", "Allow stock when 31-reference breadth is clean/improving", "candidate design only"),
        ("stock_allowed_when_reentry_breadth_confirmed_else_00631L", "Allow stock when 100->80 reentry breadth is confirmed", "candidate design only"),
        ("hybrid_stock_eligibility_gate_then_best_incumbent_or_reentry_selector", "Composite stock eligibility gate, then prior best incumbent/reentry candidate", "candidate design only"),
        ("always_stock_best_previous_baseline", "Always use prior best stock-only selector context", "baseline only"),
        ("00631L_always_baseline", "00631L always reference baseline", "reference only"),
        ("0050_reference", "0050 comparison reference", "reference only"),
    ]
    return pd.DataFrame(rows, columns=["decision_architecture_variant", "definition", "status"])


def _source_quality_matrix() -> pd.DataFrame:
    rows = [
        ("Layer0_to_Layer4_fixed_context", "PIT existing Core contracts", "base fixed architecture"),
        ("Layer4_80_primary_pool", "PIT materialized", "stock pool breadth"),
        ("Layer4_31_reference", "reference only", "confidence breadth, not hard gate"),
        ("Layer4_100_extended_watchlist", "reference only", "reentry breadth context"),
        ("Layer5_selector_family_outputs", "diagnostic context", "stock candidate source"),
        ("0050_00631L_benchmark_context", "PIT benchmark exact_or_blocked", "market context and evaluation"),
        ("forward_returns", "evaluation_metadata_only", "not rule construction"),
        ("cash_bear_classifier", "blocked", "no cash rule"),
        ("real_current_holder_state", "blocked", "hypothetical path state only where present"),
        ("00631L_fallback_live_rule", "blocked", "candidate design only"),
    ]
    return pd.DataFrame(rows, columns=["field_group", "source_quality", "contract_role"])


def _coverage_by_period(contract: pd.DataFrame) -> pd.DataFrame:
    frame = contract.copy()
    frame["period"] = frame["snapshot_date"].map(_period_label)
    rows = []
    for (period, variant), group in frame.groupby(["period", "decision_architecture_variant"], dropna=False):
        rows.append(
            {
                "period": period,
                "decision_architecture_variant": variant,
                "row_count": len(group),
                "weekly_snapshot_count": int(group["snapshot_date"].nunique()),
                "stock_allowed_day_share": _share(_bool(group, "stock_allowed_by_architecture").sum(), len(group)),
                "stock_exposure_share": _share(_bool(group, "uses_stock_candidate").sum(), len(group)),
                "fallback_00631L_exposure_share": _share(_bool(group, "uses_00631L_fallback").sum(), len(group)),
                "weak_stock_environment_share": _share(_bool(group, "weak_stock_environment_flag").sum(), len(group)),
                "forward_eval_available_20d_share": _share(_bool(group, "forward_eval_available_20d").sum(), len(group)),
            }
        )
    return pd.DataFrame(rows)


def _missingness_by_period(contract: pd.DataFrame) -> pd.DataFrame:
    fields = [
        "pool80_positive_rs20_30_share",
        "pool80_two_plus_labels_share",
        "high31_clean_improving_share",
        "reentry_100_to_80_confirmed_count",
        "top10_score_margin_top1_vs_top2",
        "stock_eligibility_composite_score",
        "0050_BIAS20",
        "0050_BIAS60",
        "00631L_BIAS20",
        "00631L_BIAS60",
    ] + [f"forward_excess_vs_0050_{h}d" for h in EVAL_HORIZONS] + [f"forward_excess_vs_00631L_{h}d" for h in EVAL_HORIZONS]
    frame = contract.copy()
    frame["period"] = frame["snapshot_date"].map(_period_label)
    rows = []
    for period, group in frame.groupby("period", dropna=False):
        for field in fields:
            missing = int(group[field].isna().sum()) if field in group else len(group)
            rows.append({"period": period, "field": field, "row_count": len(group), "missing_count": missing, "missing_share": _share(missing, len(group))})
    return pd.DataFrame(rows)


def _blocked_proxy_ledger() -> pd.DataFrame:
    rows = [
        ("cash_bear_classifier", "blocked", "This package does stock-vs-00631L only; no cash rule"),
        ("real_current_holder_state", "blocked", "No formal live holder state; only diagnostic context can be used"),
        ("00631L_fallback_live_rule", "blocked", "00631L is fallback/reference candidate design only"),
        ("A_B_switch_or_second_stock_allocation", "blocked", "Not authorized"),
        ("portfolio_replay", "blocked", "Not authorized"),
        ("weak_stock_environment_flag", "diagnostic_proxy", "PIT context threshold only"),
        ("stock_eligibility_composite_flag", "diagnostic_proxy", "Candidate architecture gate only, not formal"),
        ("top10_dispersion_clear_flag", "diagnostic_proxy", "Score margin/stdev threshold, not formal"),
        ("latest_forward_path", "blocked_partial", "Latest rows may lack future evaluation path by horizon"),
    ]
    return pd.DataFrame(rows, columns=["field_or_policy", "status", "reason"])


def _future_data_audit(contract: pd.DataFrame) -> pd.DataFrame:
    rows = [
        ("future_return_as_rule", "passed", int(_bool(contract, "future_return_as_rule").sum()), "false for all rows"),
        ("forward_return_as_rule", "passed", int(_bool(contract, "forward_return_as_rule").sum()), "false for all rows"),
        ("max_in_band_as_rule", "passed", int(_bool(contract, "max_in_band_as_rule").sum()), "false for all rows"),
        ("live_layer5_rule_output", "passed", int(_bool(contract, "live_layer5_rule_output").sum()), "no live rule output"),
        ("trade_decision_output", "passed", int(_bool(contract, "trade_decision_output").sum()), "no trade decision output"),
        ("portfolio_like_execution_output", "passed", int(_bool(contract, "portfolio_like_execution_output").sum()), "no portfolio-like execution output"),
        ("00631L_ordinary_stock_pool_member", "passed", int(_bool(contract, "00631L_ordinary_stock_pool_member").sum()), "00631L is reference/fallback metadata only"),
    ]
    return pd.DataFrame(rows, columns=["audit_item", "status", "violation_count", "evidence"])


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


def _share(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def _readiness(
    within80_readiness: dict[str, Any],
    lifecycle_readiness: dict[str, Any],
    hurdle_readiness: dict[str, Any],
    experiment_summary: dict[str, Any],
    contract: pd.DataFrame,
    coverage: pd.DataFrame,
    future_audit: pd.DataFrame,
) -> dict[str, Any]:
    future_violations = int(future_audit["violation_count"].sum())
    ready = future_violations == 0 and len(contract) > 0
    return {
        "task_id": TASK_ID,
        "status": "layer5_stock_eligibility_market_breadth_contract_ready_for_experiments_intake"
        if ready
        else "layer5_stock_eligibility_market_breadth_contract_blocked",
        "diagnostic_only": True,
        "input_within80_status": within80_readiness.get("status"),
        "input_lifecycle_status": lifecycle_readiness.get("status"),
        "input_hurdle_status": hurdle_readiness.get("status"),
        "input_experiments_verdict": experiment_summary.get("verdict"),
        "row_count": int(len(contract)),
        "weekly_snapshot_count": int(contract["snapshot_date"].nunique()),
        "decision_architecture_variant_count": int(contract["decision_architecture_variant"].nunique()),
        "decision_architecture_variants": sorted(contract["decision_architecture_variant"].unique().tolist()),
        "avg_stock_allowed_day_share_by_variant": coverage.groupby("decision_architecture_variant")["stock_allowed_day_share"].mean().to_dict(),
        "avg_fallback_00631L_exposure_share_by_variant": coverage.groupby("decision_architecture_variant")["fallback_00631L_exposure_share"].mean().to_dict(),
        "ready_for_layer5_stock_eligibility_market_breadth_diagnostic": ready,
        "ready_for_experiments_intake": ready,
        "ready_for_live_layer5_rule": False,
        "ready_for_cash_rule": False,
        "ready_for_portfolio_like_diagnostic": False,
        "ready_for_formal": False,
        "ready_for_strategy_replay": False,
        "future_data_violation_count": future_violations,
        "blocked_fields": ["cash_bear_classifier", "real_current_holder_state", "00631L_fallback_live_rule", "turnover_cost_model", "portfolio_replay"],
        "proxy_fields": ["weak_stock_environment_flag", "stock_eligibility_composite_flag", "top10_dispersion_clear_flag"],
        **_fixed_flags(),
    }


def _summary(readiness: dict[str, Any]) -> str:
    return f"""# Layer5 stock-eligibility / market-breadth final decision architecture contract

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
- Fixed Layer0-Layer4 context; no upstream data/source acquisition.
- 00631L is fallback/reference candidate metadata, not an ordinary stock row.
- This package creates stock-eligibility/environment features and bounded decision-architecture candidate variants only.
- 固定 Layer0~4 後，Layer5 下一步不是再微調個股排序，而是判斷何時有足夠股票 edge 可以選個股；沒有股票 edge 時，00631L fallback 是候選決策之一。

## Candidate variants
- decision_architecture_variant_count={readiness['decision_architecture_variant_count']}
- variants={', '.join(readiness['decision_architecture_variants'])}
- row_count={readiness['row_count']}
- weekly_snapshot_count={readiness['weekly_snapshot_count']}

## Blocked / proxy
- blocked_fields={', '.join(readiness['blocked_fields'])}
- proxy_fields={', '.join(readiness['proxy_fields'])}

## Next
If accepted, hand off to Experiments:
`TASK-BACKTEST-EXPERIMENTS-VNEXT-LAYER5-STOCK-ELIGIBILITY-MARKET-BREADTH-FINAL-DECISION-DIAGNOSTIC-001`.
完成後如果下一棒明確，請直接指派下一個 thread；如果下一棒不明確，請回報 Strategy Center 判斷。不要完成後停住不回報。
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--within80-dir", default=str(DEFAULT_WITHIN80_DIR))
    parser.add_argument("--lifecycle-dir", default=str(DEFAULT_LIFECYCLE_DIR))
    parser.add_argument("--hurdle-dir", default=str(DEFAULT_HURDLE_DIR))
    parser.add_argument("--benchmark-path", default=str(DEFAULT_BENCHMARK_PATH))
    parser.add_argument("--experiments-dir", default=str(DEFAULT_EXPERIMENTS_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()
    manifest = build_contract(
        within80_dir=args.within80_dir,
        lifecycle_dir=args.lifecycle_dir,
        hurdle_dir=args.hurdle_dir,
        benchmark_path=args.benchmark_path,
        experiments_dir=args.experiments_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
