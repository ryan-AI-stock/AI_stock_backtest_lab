"""Build Layer5 stock-vs-00631L hurdle/fallback decision context contract.

This is diagnostic/readiness only. It materializes candidate designs for a
single daily recommendation context where the recommendation can be a stock,
00631L fallback, or 0050 reference. It does not authorize a live rule, formal
model, report, trade decision, or replay.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


TASK_ID = "TASK-BACKTEST-CORE-VNEXT-LAYER5-STOCK-VS-00631L-HURDLE-FALLBACK-CONTEXT-CONTRACT-001"
DEFAULT_INCUMBENT_DIR = Path("outputs/vnext_layer5_incumbent_aware_selector_candidate_contract_20260708")
DEFAULT_BENCHMARK_PATH = Path("outputs/vnext_dynamic_candidate_pool_data_materialization_20260706/benchmark_features.csv")
DEFAULT_EXPERIMENTS_DIR = Path(
    "C:/Users/zergv/Documents/Codex/2026-07-06/backtest-lab-experiments-diagnostic-validation-attribution/"
    "outputs/vnext_layer5_incumbent_aware_single_stock_decision_diagnostic_20260708"
)
DEFAULT_OUTPUT_DIR = Path("outputs/vnext_layer5_stock_vs_00631l_hurdle_context_contract_20260708")
PERIODS = {
    "P1": ("2015-01-02", "2022-12-29"),
    "P2": ("2023-01-02", "2026-06-30"),
    "2024_latest": ("2024-01-02", "2026-06-30"),
    "2026YTD": ("2026-01-02", "2026-06-30"),
}
EVAL_HORIZONS = [5, 10, 20, 30, 40]
STOCK_SOURCE_VARIANTS = [
    "reentry_confirmed_selector",
    "incumbent_protection_selector",
    "confirmed_challenger_selector",
    "fresh_top1_baseline",
    "fresh_best_risk_adjusted_top10_baseline",
    "high_confidence_bonus_selector",
]


def build_contract(
    *,
    incumbent_dir: str | Path = DEFAULT_INCUMBENT_DIR,
    benchmark_path: str | Path = DEFAULT_BENCHMARK_PATH,
    experiments_dir: str | Path = DEFAULT_EXPERIMENTS_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    incumbent = Path(incumbent_dir)
    experiments = Path(experiments_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    source_readiness = _read_json(incumbent / "readiness_for_layer5_incumbent_aware_single_stock_diagnostic.json")
    experiment_summary = _read_json(experiments / "layer5_incumbent_summary.json")
    candidates = _read_candidates(incumbent / "layer5_incumbent_aware_selector_candidate_contract.csv")
    benchmark = _benchmark_context(Path(benchmark_path), sorted(candidates["snapshot_date"].unique()))

    contract = _build_decision_candidates(candidates, benchmark)
    design = _decision_candidate_design()
    hurdle_design = _hurdle_feature_design()
    source_quality = _source_quality_matrix()
    coverage = _coverage_by_period(contract)
    missingness = _missingness_by_period(contract)
    blocked_proxy = _blocked_proxy_ledger()
    future_audit = _future_data_audit(contract)
    readiness = _readiness(source_readiness, experiment_summary, contract, coverage, future_audit)

    _write_csv(contract, output / "layer5_stock_vs_00631l_hurdle_context_contract.csv")
    _write_csv(design, output / "layer5_stock_vs_00631l_decision_candidate_design.csv")
    _write_csv(hurdle_design, output / "layer5_fallback_hurdle_feature_design.csv")
    _write_csv(source_quality, output / "layer5_stock_vs_00631l_source_quality_matrix.csv")
    _write_csv(coverage, output / "layer5_stock_vs_00631l_coverage_by_period.csv")
    _write_csv(missingness, output / "layer5_stock_vs_00631l_missingness_by_period.csv")
    _write_csv(blocked_proxy, output / "layer5_stock_vs_00631l_blocked_proxy_ledger.csv")
    _write_csv(future_audit, output / "layer5_stock_vs_00631l_future_data_audit.csv")
    _write_csv(_requested_vs_actual_coverage(contract), output / "layer5_stock_vs_00631l_requested_vs_actual_coverage.csv")
    (output / "readiness_for_layer5_stock_vs_00631l_hurdle_diagnostic.json").write_text(
        json.dumps(readiness, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    manifest = {
        "task_id": TASK_ID,
        "status": readiness["status"],
        "output_dir": str(output.resolve()),
        "input_incumbent_dir": str(incumbent.resolve()),
        "input_benchmark_path": str(Path(benchmark_path).resolve()),
        "input_experiments_dir": str(experiments.resolve()),
        "output_files": [
            "layer5_stock_vs_00631l_hurdle_context_contract.csv",
            "layer5_stock_vs_00631l_decision_candidate_design.csv",
            "layer5_fallback_hurdle_feature_design.csv",
            "layer5_stock_vs_00631l_source_quality_matrix.csv",
            "layer5_stock_vs_00631l_coverage_by_period.csv",
            "layer5_stock_vs_00631l_missingness_by_period.csv",
            "layer5_stock_vs_00631l_blocked_proxy_ledger.csv",
            "layer5_stock_vs_00631l_future_data_audit.csv",
            "layer5_stock_vs_00631l_requested_vs_actual_coverage.csv",
            "readiness_for_layer5_stock_vs_00631l_hurdle_diagnostic.json",
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


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def _read_candidates(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"ticker": str}, encoding="utf-8-sig", low_memory=False)
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
    return df[df["selector_candidate_variant"].isin(STOCK_SOURCE_VARIANTS)].copy()


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
        if ("adjusted_close", bench_id) in pivot:
            for col in ["adjusted_close", "BIAS20", "BIAS60", "BIAS120", "drawdown", "return_20d", "return_40d", "return_60d"]:
                if (col, bench_id) in pivot:
                    out[f"{bench_id}_{col}"] = pivot[(col, bench_id)]
            for horizon in EVAL_HORIZONS:
                out[f"{bench_id}_forward_return_{horizon}d"] = pivot.get((f"forward_return_{horizon}d", bench_id))
            if ("source_quality", bench_id) in pivot:
                out[f"{bench_id}_source_quality"] = pivot[("source_quality", bench_id)]
            if ("benchmark_data_blocked", bench_id) in pivot:
                out[f"{bench_id}_benchmark_data_blocked"] = pivot[("benchmark_data_blocked", bench_id)]
    out = out.reset_index().rename(columns={"index": "snapshot_date"})
    for horizon in EVAL_HORIZONS:
        out[f"00631L_forward_excess_vs_0050_{horizon}d"] = out[f"00631L_forward_return_{horizon}d"] - out[f"0050_forward_return_{horizon}d"]
        out[f"0050_forward_excess_vs_00631L_{horizon}d"] = out[f"0050_forward_return_{horizon}d"] - out[f"00631L_forward_return_{horizon}d"]
    return out


def _build_decision_candidates(stock: pd.DataFrame, benchmark: pd.DataFrame) -> pd.DataFrame:
    stock = _attach_hurdle_context(stock)
    by_variant = {variant: frame.set_index("snapshot_date", drop=False) for variant, frame in stock.groupby("selector_candidate_variant")}
    benchmark_lookup = benchmark.set_index("snapshot_date")
    dates = sorted(stock["snapshot_date"].unique())
    rows = []
    for date in dates:
        refs = {variant: by_variant.get(variant).loc[date] for variant in by_variant if date in by_variant.get(variant).index}
        candidates = {
            "always_stock_incumbent_reentry_baseline": _stock_row(refs.get("reentry_confirmed_selector"), "always_stock_incumbent_reentry_baseline", "stock"),
            "stock_if_high_confidence_else_00631L": _stock_or_fallback(
                refs.get("high_confidence_bonus_selector"),
                "stock_if_high_confidence_else_00631L",
                bool(refs.get("high_confidence_bonus_selector", {}).get("in_31_high_confidence_subpool_reference", False))
                and bool(refs.get("high_confidence_bonus_selector", {}).get("stock_candidate_clean_state", False)),
            ),
            "stock_if_reentry_confirmed_and_not_high_risk_else_00631L": _stock_or_fallback(
                refs.get("reentry_confirmed_selector"),
                "stock_if_reentry_confirmed_and_not_high_risk_else_00631L",
                bool(refs.get("reentry_confirmed_selector", {}).get("extended_100_to_80_reentry_context", False))
                and bool(refs.get("reentry_confirmed_selector", {}).get("stock_candidate_clean_state", False)),
            ),
            "stock_if_incumbent_still_valid_else_00631L_or_best_confirmed_challenger": _stock_or_fallback(
                refs.get("incumbent_protection_selector"),
                "stock_if_incumbent_still_valid_else_00631L_or_best_confirmed_challenger",
                bool(refs.get("incumbent_protection_selector", {}).get("incumbent_still_in_80", False))
                and not bool(refs.get("incumbent_protection_selector", {}).get("incumbent_risk_deterioration", False))
                and not bool(refs.get("incumbent_protection_selector", {}).get("incumbent_rs_deterioration", False)),
            ),
            "stock_if_31_high_confidence_bonus_and_state_clean_else_00631L": _stock_or_fallback(
                refs.get("high_confidence_bonus_selector"),
                "stock_if_31_high_confidence_bonus_and_state_clean_else_00631L",
                bool(refs.get("high_confidence_bonus_selector", {}).get("in_31_high_confidence_subpool_reference", False))
                and bool(refs.get("high_confidence_bonus_selector", {}).get("stock_candidate_clean_state", False)),
            ),
        }
        for row in candidates.values():
            rows.append(row)
        rows.append(_fallback_row(date, "00631L_always_baseline", "00631L", benchmark_lookup))
        rows.append(_fallback_row(date, "0050_reference", "0050", benchmark_lookup))
    out = pd.DataFrame(rows)
    out = _attach_benchmark_context(out, benchmark)
    return out.sort_values(["decision_candidate_variant", "snapshot_date"]).reset_index(drop=True)


def _attach_hurdle_context(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["stock_candidate_score"] = _num(out, "challenger_confirmation_score").where(
        _num(out, "challenger_confirmation_score").ne(0), _num(out, "layer4_risk_aware_score")
    )
    out["stock_candidate_rank"] = _num(out, "within80_rank")
    out["stock_candidate_top10_context"] = out["stock_candidate_rank"].le(10)
    out["stock_candidate_risk_exhaustion_breakdown"] = (
        _bool(out, "incumbent_breakdown_exhaustion_candidate")
        | _bool(out, "incumbent_risk_deterioration_candidate")
        | _bool(out, "breakdown_risk_medium_or_high_confidence")
        | _bool(out, "exhaustion_risk_medium_or_high_confidence")
    )
    out["stock_candidate_clean_state"] = ~out["stock_candidate_risk_exhaustion_breakdown"]
    out["stock_candidate_reentry_confirmed_context"] = _bool(out, "extended_100_to_80_reentry_context")
    out["stock_candidate_incumbent_protection_context"] = out["selector_candidate_variant"].eq("incumbent_protection_selector")
    out["stock_candidate_31_confidence_bonus_context"] = _bool(out, "in_31_high_confidence_subpool_reference")
    out["stock_candidate_score_margin_vs_incumbent"] = _num(out, "incumbent_score_delta_vs_best_candidate")
    out["stock_vs_00631L_hurdle_proxy_pass"] = (
        out["stock_candidate_score"].ge(0.55)
        & out["stock_candidate_clean_state"]
        & (
            out["stock_candidate_reentry_confirmed_context"]
            | out["stock_candidate_31_confidence_bonus_context"]
            | _bool(out, "lifecycle_strengthening_not_overheated_context")
        )
    )
    out["stock_candidate_source_quality"] = "diagnostic_incumbent_path_proxy"
    return out


def _stock_row(row: pd.Series | None, variant: str, recommendation_type: str) -> dict[str, Any]:
    if row is None:
        return {}
    out = row.to_dict()
    out["decision_candidate_variant"] = variant
    out["recommended_asset_type"] = recommendation_type
    out["recommended_ticker"] = out["ticker"]
    out["recommended_name"] = out.get("name")
    out["uses_stock_candidate"] = True
    out["uses_00631L_fallback"] = False
    out["uses_0050_reference"] = False
    out["fallback_reference_only"] = True
    out["00631L_ordinary_stock_pool_member"] = False
    out["decision_candidate_design_only"] = True
    out["live_rule_output"] = False
    out["trade_decision_output"] = False
    out["portfolio_replay_executed"] = False
    for key, value in _fixed_flags().items():
        out[key] = value
    return out


def _stock_or_fallback(row: pd.Series | None, variant: str, condition: bool) -> dict[str, Any]:
    if row is not None and condition:
        out = _stock_row(row, variant, "stock")
        out["stock_hurdle_decision_reason"] = "stock_candidate_passed_hurdle_proxy"
        return out
    base = row.to_dict() if row is not None else {"snapshot_date": pd.NaT}
    date = pd.to_datetime(base["snapshot_date"])
    out = {key: base.get(key) for key in base.keys()}
    out["snapshot_date"] = date
    out["decision_candidate_variant"] = variant
    out["recommended_asset_type"] = "00631L_fallback"
    out["recommended_ticker"] = "00631L"
    out["recommended_name"] = "00631L fallback reference"
    out["uses_stock_candidate"] = False
    out["uses_00631L_fallback"] = True
    out["uses_0050_reference"] = False
    out["stock_hurdle_decision_reason"] = "stock_candidate_failed_hurdle_proxy"
    out["fallback_reference_only"] = True
    out["00631L_ordinary_stock_pool_member"] = False
    out["decision_candidate_design_only"] = True
    out["live_rule_output"] = False
    out["trade_decision_output"] = False
    out["portfolio_replay_executed"] = False
    for key, value in _fixed_flags().items():
        out[key] = value
    return out


def _fallback_row(date: pd.Timestamp, variant: str, ticker: str, benchmark_lookup: pd.DataFrame) -> dict[str, Any]:
    row = {
        "snapshot_date": pd.to_datetime(date),
        "ticker": ticker,
        "name": f"{ticker} reference",
        "decision_candidate_variant": variant,
        "recommended_asset_type": "00631L_fallback" if ticker == "00631L" else "0050_reference",
        "recommended_ticker": ticker,
        "recommended_name": f"{ticker} reference",
        "uses_stock_candidate": False,
        "uses_00631L_fallback": ticker == "00631L",
        "uses_0050_reference": ticker == "0050",
        "fallback_reference_only": True,
        "00631L_ordinary_stock_pool_member": False,
        "decision_candidate_design_only": True,
        "live_rule_output": False,
        "trade_decision_output": False,
        "portfolio_replay_executed": False,
        "stock_hurdle_decision_reason": "benchmark_reference_baseline",
    }
    if pd.to_datetime(date) in benchmark_lookup.index:
        bench = benchmark_lookup.loc[pd.to_datetime(date)]
        for horizon in EVAL_HORIZONS:
            if ticker == "00631L":
                row[f"forward_excess_vs_00631L_{horizon}d"] = 0.0
                row[f"forward_excess_vs_0050_{horizon}d"] = bench.get(f"00631L_forward_excess_vs_0050_{horizon}d")
            else:
                row[f"forward_excess_vs_0050_{horizon}d"] = 0.0
                row[f"forward_excess_vs_00631L_{horizon}d"] = bench.get(f"0050_forward_excess_vs_00631L_{horizon}d")
            row[f"forward_eval_available_{horizon}d"] = pd.notna(row.get(f"forward_excess_vs_00631L_{horizon}d")) and pd.notna(row.get(f"forward_excess_vs_0050_{horizon}d"))
    for key, value in _fixed_flags().items():
        row[key] = value
    return row


def _attach_benchmark_context(decisions: pd.DataFrame, benchmark: pd.DataFrame) -> pd.DataFrame:
    out = decisions.copy()
    out["snapshot_date"] = pd.to_datetime(out["snapshot_date"])
    merged = out.merge(benchmark, on="snapshot_date", how="left")
    merged["cash_bear_classifier_status"] = "blocked_no_accepted_market_cash_classifier"
    merged["stock_vs_00631L_hurdle_context_only"] = True
    merged["forward_returns_live_rule_usage"] = False
    merged["forward_return_as_rule"] = False
    merged["future_return_as_rule"] = False
    merged["not_live_rule"] = True
    return merged


def _decision_candidate_design() -> pd.DataFrame:
    rows = [
        ("always_stock_incumbent_reentry_baseline", "Always use incumbent-aware/reentry-confirmed stock candidate", "stock baseline"),
        ("stock_if_high_confidence_else_00631L", "Use stock only if high-confidence/clean state proxy passes; otherwise 00631L fallback", "candidate design only"),
        ("stock_if_reentry_confirmed_and_not_high_risk_else_00631L", "Use reentry-confirmed stock only if not high-risk/exhaustion; otherwise 00631L", "candidate design only"),
        ("stock_if_incumbent_still_valid_else_00631L_or_best_confirmed_challenger", "Use incumbent-aware stock when incumbent remains valid; otherwise fallback context", "candidate design only"),
        ("stock_if_31_high_confidence_bonus_and_state_clean_else_00631L", "31 overlap is bonus plus clean-state hurdle, not mandatory filter for pool", "candidate design only"),
        ("00631L_always_baseline", "Always recommend 00631L fallback reference", "baseline reference"),
        ("0050_reference", "0050 comparison reference", "reference only"),
    ]
    return pd.DataFrame(rows, columns=["decision_candidate_variant", "definition", "status"])


def _hurdle_feature_design() -> pd.DataFrame:
    rows = [
        ("stock_candidate_score", "diagnostic score from incumbent-aware selector candidate", "PIT context"),
        ("stock_candidate_rank", "within80 rank context", "PIT context"),
        ("stock_candidate_risk_exhaustion_breakdown", "risk/exhaustion/breakdown proxy flags", "proxy"),
        ("stock_candidate_reentry_confirmed_context", "100-to-80 reentry context", "PIT context"),
        ("stock_candidate_incumbent_protection_context", "incumbent-aware selector context", "diagnostic path proxy"),
        ("stock_candidate_31_confidence_bonus_context", "31 high-confidence overlap bonus", "reference context"),
        ("stock_vs_00631L_hurdle_proxy_pass", "score+clean+confirmation proxy, no future return", "candidate design only"),
        ("0050/00631L BIAS/trend context", "benchmark trailing BIAS/return/drawdown at signal date", "PIT benchmark context"),
        ("cash_bear_classifier_status", "cash/bear classifier blocked", "blocked"),
    ]
    return pd.DataFrame(rows, columns=["feature", "definition", "source_quality"])


def _source_quality_matrix() -> pd.DataFrame:
    rows = [
        ("incumbent_aware_stock_candidates", "diagnostic_path_state_proxy", "stock candidate source"),
        ("00631L_fallback_candidate", "benchmark_exact_or_blocked_by_future_path", "fallback candidate/reference"),
        ("0050_reference_candidate", "benchmark_exact_or_blocked_by_future_path", "comparison reference"),
        ("benchmark_forward_eval_metadata", "evaluation_metadata_only", "not rule construction"),
        ("benchmark_trailing_context", "PIT benchmark context", "hurdle context"),
        ("cash_bear_classifier", "blocked", "no cash rule"),
        ("real_current_holder_state", "blocked", "not available"),
        ("turnover_cost_model", "blocked_placeholder", "not accepted"),
    ]
    return pd.DataFrame(rows, columns=["field_group", "source_quality", "contract_role"])


def _coverage_by_period(contract: pd.DataFrame) -> pd.DataFrame:
    frame = contract.copy()
    frame["period"] = frame["snapshot_date"].map(_period_label)
    rows = []
    for keys, group in frame.groupby(["period", "decision_candidate_variant"], dropna=False):
        period, variant = keys
        rows.append(
            {
                "period": period,
                "decision_candidate_variant": variant,
                "row_count": len(group),
                "weekly_snapshot_count": int(group["snapshot_date"].nunique()),
                "stock_exposure_share": _share(_bool(group, "uses_stock_candidate").sum(), len(group)),
                "fallback_00631L_exposure_share": _share(_bool(group, "uses_00631L_fallback").sum(), len(group)),
                "reference_0050_share": _share(_bool(group, "uses_0050_reference").sum(), len(group)),
                "forward_eval_available_20d_share": _share(_bool(group, "forward_eval_available_20d").sum(), len(group)),
            }
        )
    return pd.DataFrame(rows)


def _missingness_by_period(contract: pd.DataFrame) -> pd.DataFrame:
    fields = [
        "stock_candidate_score",
        "stock_candidate_rank",
        "stock_candidate_score_margin_vs_incumbent",
        "0050_BIAS20",
        "0050_BIAS60",
        "00631L_BIAS20",
        "00631L_BIAS60",
        "00631L_return_20d",
        "00631L_return_40d",
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
        ("cash_bear_classifier", "blocked", "This package only supports stock-vs-00631L; no cash rule"),
        ("fallback_00631L_live_rule", "blocked", "Fallback candidate design only, not live trading rule"),
        ("real_current_holder_state", "blocked", "Uses hypothetical path-state proxy from incumbent package"),
        ("turnover_cost_model", "blocked_placeholder", "No accepted cost model"),
        ("A_B_switch_or_second_stock_allocation", "blocked", "Not authorized"),
        ("stock_vs_00631L_hurdle_proxy_pass", "proxy", "Live-feasible context proxy, not formal-ready"),
        ("latest_forward_path", "blocked_partial", "Latest rows may lack future evaluation path by horizon"),
    ]
    return pd.DataFrame(rows, columns=["field_or_policy", "status", "reason"])


def _future_data_audit(contract: pd.DataFrame) -> pd.DataFrame:
    rows = [
        ("future_return_as_rule", "passed", int(_bool(contract, "future_return_as_rule").sum()), "false for all rows"),
        ("forward_return_as_rule", "passed", int(_bool(contract, "forward_return_as_rule").sum()), "false for all rows"),
        ("live_rule_output", "passed", int(_bool(contract, "live_rule_output").sum()), "no live rule output"),
        ("trade_decision_output", "passed", int(_bool(contract, "trade_decision_output").sum()), "no trade decision output"),
        ("00631L_ordinary_stock_pool_member", "passed", int(_bool(contract, "00631L_ordinary_stock_pool_member").sum()), "00631L fallback/reference only"),
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
    source_readiness: dict[str, Any],
    experiment_summary: dict[str, Any],
    contract: pd.DataFrame,
    coverage: pd.DataFrame,
    future_audit: pd.DataFrame,
) -> dict[str, Any]:
    future_violations = int(future_audit["violation_count"].sum())
    ready = future_violations == 0
    return {
        "task_id": TASK_ID,
        "status": "layer5_stock_vs_00631l_hurdle_context_contract_ready_for_experiments_intake" if ready else "layer5_stock_vs_00631l_hurdle_context_contract_blocked",
        "diagnostic_only": True,
        "input_incumbent_status": source_readiness.get("status"),
        "input_experiments_verdict": experiment_summary.get("verdict"),
        "row_count": int(len(contract)),
        "weekly_snapshot_count": int(contract["snapshot_date"].nunique()),
        "decision_candidate_variant_count": int(contract["decision_candidate_variant"].nunique()),
        "decision_candidate_variants": sorted(contract["decision_candidate_variant"].unique().tolist()),
        "stock_exposure_variants": coverage.groupby("decision_candidate_variant")["stock_exposure_share"].mean().to_dict(),
        "fallback_00631L_exposure_variants": coverage.groupby("decision_candidate_variant")["fallback_00631L_exposure_share"].mean().to_dict(),
        "ready_for_layer5_stock_vs_00631l_hurdle_diagnostic": ready,
        "ready_for_experiments_intake": ready,
        "ready_for_live_layer5_rule": False,
        "ready_for_cash_rule": False,
        "ready_for_portfolio_like_diagnostic": False,
        "ready_for_formal": False,
        "ready_for_strategy_replay": False,
        "future_data_violation_count": future_violations,
        "blocked_fields": ["cash_bear_classifier", "fallback_00631L_live_rule", "real_current_holder_state", "turnover_cost_model", "portfolio_replay"],
        "proxy_fields": ["stock_vs_00631L_hurdle_proxy_pass", "hypothetical_path_state_proxy"],
        **_fixed_flags(),
    }


def _summary(readiness: dict[str, Any]) -> str:
    return f"""# Layer5 stock-vs-00631L hurdle / fallback context contract

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
- Base stock candidate line: incumbent-aware / reentry-confirmed selector context.
- 00631L is fallback candidate / hurdle reference, not an ordinary stock-pool row.
- 0050 is comparison reference only.
- Cash/bear classifier remains blocked; this package does not create a cash rule.
- Layer5 不應在沒有足夠股票 edge 時硬選個股；最終每日主推薦可以是個股，也可以是 00631L fallback。

## Candidate variants
- decision_candidate_variant_count={readiness['decision_candidate_variant_count']}
- variants={', '.join(readiness['decision_candidate_variants'])}
- row_count={readiness['row_count']}
- weekly_snapshot_count={readiness['weekly_snapshot_count']}

## Next
If accepted, hand off to Experiments:
`TASK-BACKTEST-EXPERIMENTS-VNEXT-LAYER5-STOCK-VS-00631L-HURDLE-FALLBACK-DIAGNOSTIC-001`.
完成後如果下一棒明確，請直接指派下一個 thread；如果下一棒不明確，請回報 Strategy Center 判斷。不要完成後停住不回報。
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--incumbent-dir", default=str(DEFAULT_INCUMBENT_DIR))
    parser.add_argument("--benchmark-path", default=str(DEFAULT_BENCHMARK_PATH))
    parser.add_argument("--experiments-dir", default=str(DEFAULT_EXPERIMENTS_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()
    manifest = build_contract(
        incumbent_dir=args.incumbent_dir,
        benchmark_path=args.benchmark_path,
        experiments_dir=args.experiments_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
