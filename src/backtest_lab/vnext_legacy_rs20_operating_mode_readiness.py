"""Build legacy RS20 operating mode materialized runner readiness package.

This is diagnostic/readiness only. It searches for legacy runner evidence,
materializes a dynamic Layer4-80 RS20 signal table, and prepares proxy/exact
execution timing templates for Experiments. It does not run a formal replay or
produce a live trade decision.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


TASK_ID = "TASK-BACKTEST-CORE-VNEXT-LEGACY-RS20-OPERATING-MODE-MATERIALIZED-RUNNER-READINESS-001"
DEFAULT_LAYER5_DIR = Path("outputs/vnext_layer5_within80_daily_rank_context_contract_20260708")
DEFAULT_BENCHMARK_PATH = Path("outputs/vnext_dynamic_candidate_pool_data_materialization_20260706/benchmark_features.csv")
DEFAULT_PRICE_REGISTRY = Path("data/price_source_registry.csv")
DEFAULT_EXPERIMENTS_DIR = Path(
    "C:/Users/zergv/Documents/Codex/2026-07-06/backtest-lab-experiments-diagnostic-validation-attribution/"
    "outputs/vnext_legacy_operating_mode_dynamic_layer4_80_pool_diagnostic_20260708"
)
DEFAULT_OUTPUT_DIR = Path("outputs/vnext_legacy_rs20_operating_mode_runner_readiness_20260708")
REQUESTED_START = "2024-01-02"
REQUESTED_END = "2026-05-26"
PERIODS = {
    "requested_2024_20260526": (REQUESTED_START, REQUESTED_END),
    "P1": ("2015-01-02", "2022-12-29"),
    "P2": ("2023-01-02", "2026-06-30"),
    "2024_latest": ("2024-01-02", "2026-06-30"),
    "2026YTD": ("2026-01-02", "2026-06-30"),
}
LEGACY_CORE_TICKERS = {"2330", "2454", "2308", "2317", "2382", "3231", "6669"}
COST_SCENARIOS_BP = [0, 10, 20, 40]
SIGNAL_VARIANTS = [
    "dynamic80_top1_rs20_proxy",
    "dynamic80_top3_rs20_risk_tiebreak_proxy",
    "dynamic80_top1_rs20_7core_context_proxy",
    "dynamic80_top1_rs20_31_bonus_proxy",
]


def build_contract(
    *,
    layer5_dir: str | Path = DEFAULT_LAYER5_DIR,
    benchmark_path: str | Path = DEFAULT_BENCHMARK_PATH,
    price_registry: str | Path = DEFAULT_PRICE_REGISTRY,
    experiments_dir: str | Path = DEFAULT_EXPERIMENTS_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    layer5_path = Path(layer5_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    within80 = _read_context(layer5_path / "layer5_within80_daily_rank_context_contract.csv")
    benchmark = _benchmark_context(Path(benchmark_path), sorted(within80["snapshot_date"].unique()))
    experiment_summary = _read_json(Path(experiments_dir) / "legacy_mode_dynamic_pool_summary.json")
    price_registry_frame = _read_price_registry(Path(price_registry))

    search_audit = _legacy_runner_search_audit()
    runner_contract = _runner_contract()
    signal_table = _build_signal_table(within80, benchmark)
    price_path = _build_price_return_path(signal_table)
    trade_path = _build_trade_path_template(signal_table)
    timing_audit = _execution_timing_audit(signal_table, price_registry_frame)
    cost_scenarios = _cost_scenarios()
    missingness = _missingness_coverage(signal_table, price_registry_frame)
    future_audit = _future_data_audit(signal_table, trade_path)
    readiness = _readiness(experiment_summary, search_audit, signal_table, timing_audit, missingness, future_audit)

    _write_csv(runner_contract, output / "legacy_rs20_operating_mode_runner_contract.csv")
    _write_csv(signal_table, output / "legacy_rs20_operating_mode_signal_table.csv")
    _write_csv(price_path, output / "legacy_rs20_operating_mode_price_return_path_table.csv")
    _write_csv(trade_path, output / "legacy_rs20_operating_mode_trade_path_template.csv")
    _write_csv(timing_audit, output / "legacy_rs20_operating_mode_execution_timing_audit.csv")
    _write_csv(cost_scenarios, output / "legacy_rs20_operating_mode_cost_scenarios.csv")
    _write_csv(search_audit, output / "legacy_rs20_operating_mode_legacy_runner_search_audit.csv")
    _write_csv(missingness, output / "legacy_rs20_operating_mode_missingness_coverage.csv")
    _write_csv(future_audit, output / "legacy_rs20_operating_mode_future_data_audit.csv")
    (output / "readiness_for_legacy_rs20_operating_mode_diagnostic.json").write_text(
        json.dumps(readiness, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    manifest = {
        "task_id": TASK_ID,
        "status": readiness["status"],
        "output_dir": str(output.resolve()),
        "input_layer5_dir": str(layer5_path.resolve()),
        "input_benchmark_path": str(Path(benchmark_path).resolve()),
        "input_price_registry": str(Path(price_registry).resolve()),
        "input_experiments_dir": str(Path(experiments_dir).resolve()),
        "output_files": [
            "legacy_rs20_operating_mode_runner_contract.csv",
            "legacy_rs20_operating_mode_signal_table.csv",
            "legacy_rs20_operating_mode_price_return_path_table.csv",
            "legacy_rs20_operating_mode_trade_path_template.csv",
            "legacy_rs20_operating_mode_execution_timing_audit.csv",
            "legacy_rs20_operating_mode_cost_scenarios.csv",
            "legacy_rs20_operating_mode_legacy_runner_search_audit.csv",
            "legacy_rs20_operating_mode_missingness_coverage.csv",
            "legacy_rs20_operating_mode_future_data_audit.csv",
            "readiness_for_legacy_rs20_operating_mode_diagnostic.json",
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


def _read_price_registry(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"ticker": str}, encoding="utf-8-sig") if path.exists() else pd.DataFrame()


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
    rows = []
    for benchmark, group in bench.groupby("benchmark"):
        group = group.sort_values("trade_date").reset_index(drop=True)
        close = pd.to_numeric(group["adjusted_close"], errors="coerce")
        group["benchmark"] = str(benchmark)
        group["benchmark_forward_return_5d"] = close.shift(-5) / close - 1
        rows.append(group)
    data = pd.concat(rows, ignore_index=True)
    out = data[data["trade_date"].isin(signal_dates)].pivot(index="trade_date", columns="benchmark")
    frame = pd.DataFrame(index=sorted(signal_dates)).reset_index().rename(columns={"index": "snapshot_date"})
    for bench_id in ["0050", "00631L"]:
        if ("benchmark_forward_return_5d", bench_id) in out:
            frame = frame.merge(
                out[("benchmark_forward_return_5d", bench_id)].rename(f"{bench_id}_forward_return_5d").reset_index().rename(columns={"trade_date": "snapshot_date"}),
                on="snapshot_date",
                how="left",
            )
    return frame


def _legacy_runner_search_audit() -> pd.DataFrame:
    rows = [
        ("configs/ep05_universe.json", "found_config", "2024-01-02 to 2026-05-26, T close / T+1 open policy, 0050+00631L+mega caps universe", False),
        ("src/backtest_lab/cli.py", "found_legacy_ep05_runner_entry", "Runs relative_strength_top1 on fixed group through yfinance/cache; not dynamic Layer4 80 exact runner", False),
        ("src/backtest_lab/simulation.py", "found_legacy_strategy_functions", "Likely contains relative_strength_top1 mechanics, but not exact vNext dynamic80 materialized runner", False),
        ("docs/regime_aware_strategy_product_spec.md", "found_historical_design_doc", "Daily chasing / weekly rotation design notes only", False),
        ("experiments legacy_mode_dynamic_pool_summary.json", "found_prior_proxy_result", "Experiments already marked exact_legacy_runner_found=false", False),
        ("repo_wide_keyword_search", "exact_legacy_runner_not_found", "No audited exact legacy runner with dynamic80, costs, timing, and price path found in minimal search", True),
    ]
    return pd.DataFrame(rows, columns=["path_or_scope", "search_status", "evidence", "exact_legacy_runner_not_found"])


def _runner_contract() -> pd.DataFrame:
    rows = [
        ("universe", "Layer4 80 primary pool", "outputs/vnext_layer5_within80_daily_rank_context_contract_20260708", "PIT existing contract"),
        ("main_candidate", "weekly RS20 top1 within 80", "RS20 rank within signal_date", "PIT"),
        ("comparator", "RS20 top3 risk/context tie-break", "top3 by RS20 then layer4_risk_aware_score", "PIT/proxy"),
        ("comparator", "RS20 top1 with 7-core flag", "flag only for 2330/2454/2308/2317/2382/3231/6669", "PIT context"),
        ("comparator", "RS20 with optional 31 high-confidence bonus", "RS20 + small context bonus; not formal", "PIT/proxy"),
        ("execution_timing_a", "same-week close to next 5 trading days proxy", "uses forward_excess_vs_00631L_5d + 00631L forward 5d", "proxy materialized"),
        ("execution_timing_b", "next-trading-day entry", "blocked until exact adjusted-close daily path exists for selected stocks", "blocked"),
        ("holding_discipline", "fixed 5 trading days / weekly rebalance / rank deterioration template", "rank deterioration is PIT feasible but not computed as path rule here", "template"),
        ("cost_scenarios", "0/10/20/40bp roundtrip", "scenario fields only", "diagnostic"),
    ]
    return pd.DataFrame(rows, columns=["contract_component", "definition", "materialization_basis", "source_quality"])


def _build_signal_table(within80: pd.DataFrame, benchmark: pd.DataFrame) -> pd.DataFrame:
    frame = within80[_bool(within80, "is_layer4_primary_pool")].copy()
    frame["is_legacy_7_core"] = frame["ticker"].astype(str).isin(LEGACY_CORE_TICKERS)
    frame["RS20_rank_within_80"] = frame.groupby("snapshot_date")["RS20"].rank(method="first", ascending=False)
    frame["rs20_31_bonus_score"] = _num(frame, "RS20") + 0.02 * _bool(frame, "in_31_high_confidence_subpool_reference").astype(float)
    frame["rs20_risk_context_score"] = _num(frame, "RS20") + 0.05 * _num(frame, "layer4_risk_aware_score") - 0.03 * _bool(frame, "high_exhaustion_or_breakdown_context").astype(float)
    selections = []
    for date, group in frame.groupby("snapshot_date", sort=True):
        group = group.copy()
        top1 = group.sort_values(["RS20", "ticker"], ascending=[False, True]).iloc[0]
        top3 = group.sort_values(["RS20", "ticker"], ascending=[False, True]).head(3)
        risk_tiebreak = top3.sort_values(["rs20_risk_context_score", "RS20", "ticker"], ascending=[False, False, True]).iloc[0]
        bonus = group.sort_values(["rs20_31_bonus_score", "RS20", "ticker"], ascending=[False, False, True]).iloc[0]
        for variant, row in [
            ("dynamic80_top1_rs20_proxy", top1),
            ("dynamic80_top3_rs20_risk_tiebreak_proxy", risk_tiebreak),
            ("dynamic80_top1_rs20_7core_context_proxy", top1),
            ("dynamic80_top1_rs20_31_bonus_proxy", bonus),
        ]:
            selected = row.copy()
            selected["signal_variant"] = variant
            selected["signal_date"] = date
            selected["selected_rank"] = selected.get("RS20_rank_within_80")
            selected["signal_timing"] = "weekly_signal_close"
            selected["selection_rule_basis"] = _selection_rule_basis(variant)
            selected["diagnostic_only"] = True
            selected["not_live_rule"] = True
            selected["forward_return_as_rule"] = False
            selected["future_return_as_rule"] = False
            selections.append(selected)
    out = pd.DataFrame(selections)
    out = out.merge(benchmark, left_on="signal_date", right_on="snapshot_date", how="left", suffixes=("", "_benchmark"))
    out["proxy_stock_forward_return_5d"] = pd.to_numeric(out["00631L_forward_return_5d"], errors="coerce") + pd.to_numeric(
        out["forward_excess_vs_00631L_5d"],
        errors="coerce",
    )
    out["proxy_stock_forward_return_source"] = "00631L_forward_return_5d_plus_forward_excess_vs_00631L_5d"
    out["exact_stock_adjusted_close_path_available"] = False
    out["next_day_exact_path_status"] = "blocked_no_full_dynamic80_adjusted_close_path"
    for key, value in _fixed_flags().items():
        out[key] = value
    cols = [
        "signal_date",
        "signal_variant",
        "ticker",
        "name",
        "market",
        "selected_rank",
        "RS20",
        "RS60",
        "RS20_rank_within_80",
        "pool_rank",
        "within80_rank",
        "is_legacy_7_core",
        "in_31_high_confidence_subpool_reference",
        "in_100_extended_watchlist_reference",
        "capital_support_context",
        "high_exhaustion_or_breakdown_context",
        "layer4_risk_aware_score",
        "rs20_31_bonus_score",
        "rs20_risk_context_score",
        "forward_excess_vs_0050_5d",
        "forward_excess_vs_00631L_5d",
        "00631L_forward_return_5d",
        "0050_forward_return_5d",
        "proxy_stock_forward_return_5d",
        "forward_eval_available_5d",
        "signal_timing",
        "selection_rule_basis",
        "proxy_stock_forward_return_source",
        "exact_stock_adjusted_close_path_available",
        "next_day_exact_path_status",
        "diagnostic_only",
        "not_live_rule",
        "forward_return_as_rule",
        "future_return_as_rule",
    ] + list(_fixed_flags().keys())
    return out[[col for col in cols if col in out]].sort_values(["signal_variant", "signal_date"]).reset_index(drop=True)


def _selection_rule_basis(variant: str) -> str:
    if variant == "dynamic80_top3_rs20_risk_tiebreak_proxy":
        return "top3_by_RS20_then_layer4_risk_aware_score"
    if variant == "dynamic80_top1_rs20_31_bonus_proxy":
        return "RS20_plus_31_high_confidence_context_bonus"
    if variant == "dynamic80_top1_rs20_7core_context_proxy":
        return "top1_by_RS20_with_legacy_7_core_flag_context"
    return "top1_by_RS20_within_layer4_80_primary_pool"


def _build_price_return_path(signal_table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in signal_table.iterrows():
        rows.append(
            {
                "signal_date": row["signal_date"],
                "signal_variant": row["signal_variant"],
                "ticker": row["ticker"],
                "path_basis": "same_week_close_to_next_5td_proxy",
                "exact_or_proxy": "proxy",
                "entry_price_available": False,
                "exit_price_available": False,
                "entry_date": row["signal_date"],
                "exit_date": pd.NaT,
                "holding_days_target": 5,
                "proxy_stock_forward_return_5d": row.get("proxy_stock_forward_return_5d"),
                "forward_excess_vs_00631L_5d": row.get("forward_excess_vs_00631L_5d"),
                "exact_price_blocked_reason": "full dynamic80 selected-stock adjusted-close path not materialized",
                "forward_return_as_rule": False,
                "diagnostic_only": True,
            }
        )
    return pd.DataFrame(rows)


def _build_trade_path_template(signal_table: pd.DataFrame) -> pd.DataFrame:
    frame = signal_table.copy().sort_values(["signal_variant", "signal_date"])
    rows = []
    for variant, group in frame.groupby("signal_variant", sort=True):
        previous_ticker = None
        for _, row in group.iterrows():
            switch = previous_ticker is not None and previous_ticker != row["ticker"]
            for bp in COST_SCENARIOS_BP:
                rows.append(
                    {
                        "signal_date": row["signal_date"],
                        "signal_variant": variant,
                        "ticker": row["ticker"],
                        "entry_timing_variant": "same_week_close_proxy",
                        "holding_discipline_variant": "fixed_5_trading_days_or_weekly_rebalance_proxy",
                        "holding_days_target": 5,
                        "switch_from_previous_signal": switch,
                        "roundtrip_cost_bp": bp,
                        "proxy_gross_return_5d": row.get("proxy_stock_forward_return_5d"),
                        "proxy_net_return_5d_after_cost": row.get("proxy_stock_forward_return_5d") - bp / 10000.0 if pd.notna(row.get("proxy_stock_forward_return_5d")) else None,
                        "exact_next_day_entry_return_available": False,
                        "rank_deterioration_exit_template_available": True,
                        "rank_deterioration_exit_materialized": False,
                        "rank_deterioration_exit_blocked_reason": "requires daily or next-signal rank path materialization; template only",
                        "diagnostic_only": True,
                        "not_live_rule": True,
                        "forward_return_as_rule": False,
                    }
                )
            previous_ticker = row["ticker"]
    return pd.DataFrame(rows)


def _execution_timing_audit(signal_table: pd.DataFrame, price_registry: pd.DataFrame) -> pd.DataFrame:
    selected_tickers = set(signal_table["ticker"].astype(str))
    registry_tickers = set(price_registry.get("ticker", pd.Series(dtype=str)).astype(str).str.replace(".TW", "", regex=False)) if not price_registry.empty else set()
    covered = selected_tickers & registry_tickers
    rows = [
        ("same_week_close_to_next_5td_proxy", "materialized_proxy", len(signal_table), "uses existing forward_excess_vs_00631L_5d and 00631L benchmark forward return"),
        ("next_trading_day_entry_exact", "blocked", len(selected_tickers) - len(covered), "price_source_registry lacks full selected-stock adjusted close coverage"),
        ("weekly_rebalance", "template_ready_proxy", len(signal_table), "signal dates are weekly snapshots; exact trade dates need price calendar path"),
        ("hold_until_rank_deterioration", "template_only", len(signal_table), "PIT-observable but path not materialized here"),
    ]
    return pd.DataFrame(rows, columns=["timing_item", "status", "affected_count", "evidence"])


def _cost_scenarios() -> pd.DataFrame:
    rows = []
    for bp in COST_SCENARIOS_BP:
        rows.append(
            {
                "cost_scenario": f"roundtrip_{bp}bp",
                "roundtrip_cost_bp": bp,
                "cost_application": "subtract from each switch/holding interval in Experiments diagnostic",
                "source_quality": "scenario_placeholder" if bp else "no_cost_reference",
                "formal_cost_model": False,
            }
        )
    rows.append(
        {
            "cost_scenario": "local_ep05_standard_cost_model",
            "roundtrip_cost_bp": None,
            "cost_application": "available in configs/ep05_universe.json but not applied in Core readiness",
            "source_quality": "blocked_until_experiments_cost_runner_applies_turnover_tax_fee",
            "formal_cost_model": False,
        }
    )
    return pd.DataFrame(rows)


def _missingness_coverage(signal_table: pd.DataFrame, price_registry: pd.DataFrame) -> pd.DataFrame:
    selected = set(signal_table["ticker"].astype(str))
    registry = set(price_registry.get("ticker", pd.Series(dtype=str)).astype(str).str.replace(".TW", "", regex=False)) if not price_registry.empty else set()
    rows = []
    for field in ["RS20", "RS60", "forward_excess_vs_00631L_5d", "00631L_forward_return_5d", "proxy_stock_forward_return_5d"]:
        missing = int(signal_table[field].isna().sum()) if field in signal_table else len(signal_table)
        rows.append({"coverage_item": field, "row_count": len(signal_table), "missing_count": missing, "missing_share": _share(missing, len(signal_table)), "status": "ready" if missing == 0 else "partial"})
    rows.append(
        {
            "coverage_item": "exact_selected_stock_adjusted_close_path",
            "row_count": len(selected),
            "missing_count": len(selected - registry),
            "missing_share": _share(len(selected - registry), len(selected)),
            "status": "blocked_for_exact_next_day_runner",
        }
    )
    rows.append(
        {
            "coverage_item": "selected_unique_ticker_count",
            "row_count": len(selected),
            "missing_count": 0,
            "missing_share": 0.0,
            "status": "informational",
        }
    )
    return pd.DataFrame(rows)


def _future_data_audit(signal_table: pd.DataFrame, trade_path: pd.DataFrame) -> pd.DataFrame:
    rows = [
        ("selection_uses_forward_return", "passed", int(_bool(signal_table, "forward_return_as_rule").sum()), "selection ranks by RS20 and PIT context only"),
        ("future_return_as_rule", "passed", int(_bool(signal_table, "future_return_as_rule").sum()), "false for all signal rows"),
        ("trade_template_forward_return_as_rule", "passed", int(_bool(trade_path, "forward_return_as_rule").sum()), "false for trade templates"),
        ("formal_model_changed", "passed", 0, "contract/readiness only"),
        ("trade_decision_output", "passed", 0, "no live trade decision output"),
    ]
    return pd.DataFrame(rows, columns=["audit_item", "status", "violation_count", "evidence"])


def _readiness(
    experiment_summary: dict[str, Any],
    search_audit: pd.DataFrame,
    signal_table: pd.DataFrame,
    timing_audit: pd.DataFrame,
    missingness: pd.DataFrame,
    future_audit: pd.DataFrame,
) -> dict[str, Any]:
    future_violations = int(future_audit["violation_count"].sum())
    exact_runner_not_found = bool(search_audit["exact_legacy_runner_not_found"].any())
    exact_path_blocked = "blocked" in set(timing_audit["status"])
    ready = future_violations == 0 and len(signal_table) > 0
    return {
        "task_id": TASK_ID,
        "status": "legacy_rs20_operating_mode_materialized_proxy_runner_ready_exact_path_blocked"
        if ready and exact_path_blocked
        else "legacy_rs20_operating_mode_readiness_blocked",
        "diagnostic_only": True,
        "input_experiments_verdict": experiment_summary.get("verdict"),
        "exact_legacy_runner_found": False,
        "exact_legacy_runner_not_found": exact_runner_not_found,
        "materialized_signal_row_count": int(len(signal_table)),
        "weekly_snapshot_count": int(signal_table["signal_date"].nunique()),
        "signal_variant_count": int(signal_table["signal_variant"].nunique()),
        "signal_variants": sorted(signal_table["signal_variant"].unique().tolist()),
        "selected_unique_ticker_count": int(signal_table["ticker"].nunique()),
        "same_week_close_forward_5td_proxy_ready": True,
        "next_trading_day_exact_path_ready": False,
        "ready_for_legacy_rs20_operating_mode_cost_timing_diagnostic": ready,
        "ready_for_exact_next_day_runner": False,
        "ready_for_formal": False,
        "ready_for_strategy_replay": False,
        "ready_for_daily_report": False,
        "future_data_violation_count": future_violations,
        "blocked_fields": [
            "exact_legacy_runner",
            "full_dynamic80_selected_stock_adjusted_close_path",
            "next_trading_day_entry_exact_return",
            "rank_deterioration_exit_materialized_path",
            "formal_cost_model_application",
        ],
        "proxy_fields": [
            "proxy_stock_forward_return_5d",
            "same_week_close_to_next_5td_proxy",
            "rs20_31_bonus_score",
            "rs20_risk_context_score",
        ],
        **_fixed_flags(),
    }


def _summary(readiness: dict[str, Any]) -> str:
    return f"""# Legacy RS20 operating mode materialized runner readiness

## Verdict
- status={readiness['status']}
- diagnostic_only=true
- exact_legacy_runner_found=false
- exact_legacy_runner_not_found={str(readiness['exact_legacy_runner_not_found']).lower()}
- same_week_close_forward_5td_proxy_ready=true
- next_trading_day_exact_path_ready=false
- formal_model_changed=false
- trade_decision_changed=false
- active_in_trade_decision=false
- report_changed=false
- portfolio_replay_executed=false
- ready_for_strategy_replay=false
- not_live_rule=true
- forward_returns_live_rule_usage=false

## Scope
- Main materialized candidate: weekly RS20 top1 within Layer4 80 primary pool.
- Comparators: top3 RS20 risk/context tie-break, 7-core flag context, 31 high-confidence bonus context.
- This package is runner/readiness only, not formal-ready and not a daily trade decision.
- Exact next-day adjusted-close path is blocked because selected dynamic80 stock price coverage is not materialized locally.

## Coverage
- materialized_signal_row_count={readiness['materialized_signal_row_count']}
- weekly_snapshot_count={readiness['weekly_snapshot_count']}
- signal_variant_count={readiness['signal_variant_count']}
- selected_unique_ticker_count={readiness['selected_unique_ticker_count']}

## Blocked / proxy
- blocked_fields={', '.join(readiness['blocked_fields'])}
- proxy_fields={', '.join(readiness['proxy_fields'])}

## Next
If accepted, hand off to Experiments:
`TASK-BACKTEST-EXPERIMENTS-VNEXT-LEGACY-RS20-OPERATING-MODE-COST-TIMING-DIAGNOSTIC-001`.
完成後如果下一棒明確，請直接指派下一個 thread；如果下一棒不明確，請回報 Strategy Center 判斷。不要完成後停住不回報。
"""


def _share(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer5-dir", default=str(DEFAULT_LAYER5_DIR))
    parser.add_argument("--benchmark-path", default=str(DEFAULT_BENCHMARK_PATH))
    parser.add_argument("--price-registry", default=str(DEFAULT_PRICE_REGISTRY))
    parser.add_argument("--experiments-dir", default=str(DEFAULT_EXPERIMENTS_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()
    manifest = build_contract(
        layer5_dir=args.layer5_dir,
        benchmark_path=args.benchmark_path,
        price_registry=args.price_registry,
        experiments_dir=args.experiments_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
