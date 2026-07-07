"""Rebuild Layer1 interim contract from compact Layer0 active scope.

Uses only compact Layer0 active_layer1_source_scope rows. Watchlist reference
rows remain excluded from Layer1 source scope.

This is diagnostic/source readiness only. It does not run Experiments, replay,
formal model changes, report changes, trade decisions, or t164 mass download.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from backtest_lab.vnext_layer1_reduced_universe_interim_contract import (
    PERIODS,
    _add_quality_floor_candidates,
    _blocked_proxy_fields,
    _build_materialized_contract,
    _future_data_audit,
    _missingness_by_period,
    _quality_floor_variant_design,
    _read_json,
    _read_monthly_revenue,
    _read_quarterly_fundamentals,
    _source_quality_matrix,
    _write_csv,
)


TASK_ID = "TASK-BACKTEST-CORE-VNEXT-LAYER1-COMPACT-REDUCED-UNIVERSE-INTERIM-CONTRACT-REBUILD-001"
DEFAULT_COMPACT_LAYER0_DIR = Path("outputs/vnext_layer0_compact_weekly_universe_snapshot_contract_20260707")
DEFAULT_PREVIOUS_LAYER1_DIR = Path("outputs/vnext_layer1_reduced_universe_interim_contract_20260707")
DEFAULT_OUTPUT_DIR = Path("outputs/vnext_layer1_compact_reduced_universe_interim_contract_20260707")


def build_contract(
    *,
    compact_layer0_dir: str | Path = DEFAULT_COMPACT_LAYER0_DIR,
    previous_layer1_dir: str | Path = DEFAULT_PREVIOUS_LAYER1_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    compact_layer0 = Path(compact_layer0_dir)
    previous_layer1 = Path(previous_layer1_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    compact_readiness = _read_json(compact_layer0 / "readiness_for_layer0_compact_weekly_universe_snapshot.json")
    previous_readiness = _read_json(previous_layer1 / "readiness_for_layer1_reduced_universe_interim_contract.json")
    compact_snapshot = _read_compact_active_scope(compact_layer0 / "layer0_compact_weekly_universe_snapshot.csv")
    monthly = _read_monthly_revenue()
    quarterly = _read_quarterly_fundamentals()

    contract = _build_materialized_contract(compact_snapshot, monthly, quarterly)
    contract = _add_quality_floor_candidates(contract)
    coverage = _coverage_by_period(contract)
    missingness = _missingness_by_period(contract)
    source_quality = _source_quality_matrix()
    variants = _quality_floor_variant_design()
    blocked_proxy = _blocked_proxy_fields()
    future_audit = _future_data_audit(contract)
    comparison = _compare_vs_previous(coverage, missingness, previous_readiness)
    readiness = _readiness(compact_readiness, previous_readiness, contract, coverage, future_audit)

    _write_csv(contract, output / "layer1_compact_reduced_universe_interim_contract.csv")
    _write_csv(contract.head(1000), output / "layer1_compact_reduced_universe_interim_contract_sample.csv")
    (output / ".gitignore").write_text(
        "layer1_compact_reduced_universe_interim_contract.csv\n",
        encoding="utf-8",
    )
    _write_csv(coverage, output / "layer1_compact_reduced_universe_interim_coverage_by_period.csv")
    _write_csv(missingness, output / "layer1_compact_reduced_universe_interim_missingness_by_period.csv")
    _write_csv(source_quality, output / "layer1_compact_reduced_universe_interim_source_quality_matrix.csv")
    _write_csv(variants, output / "layer1_compact_reduced_universe_quality_floor_variant_design.csv")
    _write_csv(blocked_proxy, output / "layer1_compact_reduced_universe_interim_blocked_proxy_fields.csv")
    _write_csv(future_audit, output / "layer1_compact_reduced_universe_interim_future_data_audit.csv")
    _write_csv(comparison, output / "layer1_compact_vs_top300_buffer100_cost_comparison.csv")
    (output / "readiness_for_layer1_compact_reduced_universe_interim_contract.json").write_text(
        json.dumps(readiness, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    manifest = {
        "task_id": TASK_ID,
        "status": readiness["status"],
        "output_dir": str(output.resolve()),
        "input_compact_layer0_dir": str(compact_layer0.resolve()),
        "input_compact_layer0_commit": "57c2e20",
        "input_previous_layer1_dir": str(previous_layer1.resolve()),
        "output_files": [
            "layer1_compact_reduced_universe_interim_contract.csv",
            "layer1_compact_reduced_universe_interim_contract_sample.csv",
            "layer1_compact_reduced_universe_interim_coverage_by_period.csv",
            "layer1_compact_reduced_universe_interim_missingness_by_period.csv",
            "layer1_compact_reduced_universe_interim_source_quality_matrix.csv",
            "layer1_compact_reduced_universe_quality_floor_variant_design.csv",
            "layer1_compact_reduced_universe_interim_blocked_proxy_fields.csv",
            "layer1_compact_reduced_universe_interim_future_data_audit.csv",
            "layer1_compact_vs_top300_buffer100_cost_comparison.csv",
            "readiness_for_layer1_compact_reduced_universe_interim_contract.json",
            "manifest.json",
            "final_summary_zh.md",
        ],
        "large_local_files_not_tracked": [
            "layer1_compact_reduced_universe_interim_contract.csv"
        ],
        "large_local_file_policy": "full materialized compact Layer1 contract is retained in local output path; Git tracks sample/readiness/audit files only",
        "formal_model_changed": False,
        "trade_decision_changed": False,
        "active_in_trade_decision": False,
        "report_changed": False,
        "portfolio_replay_executed": False,
        "ready_for_strategy_replay": False,
        "not_live_rule": True,
        "forward_returns_live_rule_usage": False,
        "diagnostic_only": True,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "final_summary_zh.md").write_text(_summary(readiness), encoding="utf-8")
    return manifest


def _read_compact_active_scope(path: Path) -> pd.DataFrame:
    cols = [
        "snapshot_date",
        "variant",
        "ticker",
        "name",
        "market",
        "scope_type",
        "active_for_layer1_source_scope",
        "selection_bucket",
        "traded_value_5d",
        "traded_value_20d",
        "traded_value_60d",
        "traded_value_rank_5d",
        "traded_value_rank_20d",
        "traded_value_rank_60d",
        "rank_improvement_5d_vs_60d",
        "top300_5d_count_last4w",
        "buffer_candidate_rank_251_300",
        "buffer_included_by_2in4",
        "buffer_included_by_20d60d",
        "pure_5d_burst_watchlist_only",
        "listing_status",
        "liquidity_flag",
        "is_ky_name_proxy",
        "instrument_type_source_quality",
        "market_cap_rank_source_quality",
        "event_ledger_source_quality",
    ]
    df = pd.read_csv(path, usecols=cols, dtype={"ticker": str})
    df["active_for_layer1_source_scope"] = df["active_for_layer1_source_scope"].astype(str).str.lower().eq("true")
    df = df[df["active_for_layer1_source_scope"] & df["scope_type"].eq("active_layer1_source_scope")].copy()
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
    for col in [
        "traded_value_5d",
        "traded_value_20d",
        "traded_value_60d",
        "traded_value_rank_5d",
        "traded_value_rank_20d",
        "traded_value_rank_60d",
        "rank_improvement_5d_vs_60d",
        "top300_5d_count_last4w",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.sort_values(["ticker", "snapshot_date"])


def _coverage_by_period(contract: pd.DataFrame) -> pd.DataFrame:
    rows = [_coverage_row("ALL", contract, None, None)]
    for period, (start, end) in PERIODS.items():
        rows.append(_coverage_row(period, contract, start, end))
    return pd.DataFrame(rows)


def _coverage_row(period: str, contract: pd.DataFrame, start: str | None, end: str | None) -> dict[str, Any]:
    requested_start = start or str(contract["snapshot_date"].min().date())
    requested_end = end or str(contract["snapshot_date"].max().date())
    mask = pd.Series(True, index=contract.index)
    if start:
        mask &= contract["snapshot_date"].ge(pd.Timestamp(start))
    if end:
        mask &= contract["snapshot_date"].le(pd.Timestamp(end))
    c = contract[mask]
    weekly_count = c.groupby("snapshot_date")["ticker"].nunique()
    return {
        "period": period,
        "requested_start": requested_start,
        "requested_end": requested_end,
        "actual_start": str(c["snapshot_date"].min().date()) if not c.empty else "",
        "actual_end": str(c["snapshot_date"].max().date()) if not c.empty else "",
        "weekly_snapshot_count": int(c["snapshot_date"].nunique()),
        "contract_rows": int(len(c)),
        "average_weekly_count": float(weekly_count.mean()) if not weekly_count.empty else 0.0,
        "unique_ticker_count": int(c["ticker"].nunique()),
        "monthly_revenue_available_share": float(c["monthly_revenue_available"].mean()) if not c.empty else 0.0,
        "quarterly_fundamental_available_share": float(c["quarterly_fundamental_available"].mean()) if not c.empty else 0.0,
        "liquidity_context_available_share": float(c["liquidity_context_available"].mean()) if not c.empty else 0.0,
        "listing_status_available_share": float(c["listing_status_available"].mean()) if not c.empty else 0.0,
        "diagnostic_only": True,
    }


def _compare_vs_previous(
    coverage: pd.DataFrame,
    missingness: pd.DataFrame,
    previous_readiness: dict[str, Any],
) -> pd.DataFrame:
    prev_rows = previous_readiness.get("contract_rows", 236800)
    prev_weekly = previous_readiness.get("average_weekly_ticker_count", 400.0)
    prev_unique = previous_readiness.get("unique_ticker_count_all", 1843)
    prev_monthly = previous_readiness.get("monthly_revenue_available_share", 0.9576182432432433)
    prev_quarterly = previous_readiness.get("quarterly_fundamental_available_share", 0.9387584459459459)
    all_row = coverage[coverage["period"].eq("ALL")].iloc[0]
    p2_row = coverage[coverage["period"].eq("P2")].iloc[0]
    rows = [
        ("ALL_contract_rows", float(all_row["contract_rows"]), float(prev_rows), _reduction(float(all_row["contract_rows"]), float(prev_rows))),
        ("ALL_average_weekly_count", float(all_row["average_weekly_count"]), float(prev_weekly), _reduction(float(all_row["average_weekly_count"]), float(prev_weekly))),
        ("ALL_unique_ticker_count", float(all_row["unique_ticker_count"]), float(prev_unique), _reduction(float(all_row["unique_ticker_count"]), float(prev_unique))),
        ("ALL_monthly_revenue_available_share", float(all_row["monthly_revenue_available_share"]), float(prev_monthly), float(all_row["monthly_revenue_available_share"]) - float(prev_monthly)),
        ("ALL_quarterly_fundamental_available_share", float(all_row["quarterly_fundamental_available_share"]), float(prev_quarterly), float(all_row["quarterly_fundamental_available_share"]) - float(prev_quarterly)),
        ("P2_contract_rows", float(p2_row["contract_rows"]), 72000.0, _reduction(float(p2_row["contract_rows"]), 72000.0)),
        ("P2_average_weekly_count", float(p2_row["average_weekly_count"]), 400.0, _reduction(float(p2_row["average_weekly_count"]), 400.0)),
        ("P2_unique_ticker_count", float(p2_row["unique_ticker_count"]), 1430.0, _reduction(float(p2_row["unique_ticker_count"]), 1430.0)),
    ]
    return pd.DataFrame(rows, columns=["metric", "compact_value", "top300_buffer100_value", "delta_or_reduction_vs_top300"]).assign(
        diagnostic_only=True
    )


def _reduction(new: float, old: float) -> float:
    return 1 - new / old if old else 0.0


def _readiness(
    compact_readiness: dict[str, Any],
    previous_readiness: dict[str, Any],
    contract: pd.DataFrame,
    coverage: pd.DataFrame,
    future_audit: pd.DataFrame,
) -> dict[str, Any]:
    all_row = coverage[coverage["period"].eq("ALL")].iloc[0].to_dict()
    p2_row = coverage[coverage["period"].eq("P2")].iloc[0].to_dict()
    future_count = int(future_audit["future_data_violation_count"].max())
    ready = (
        future_count == 0
        and float(all_row["monthly_revenue_available_share"]) >= 0.80
        and float(all_row["quarterly_fundamental_available_share"]) >= 0.80
    )
    return {
        "task_id": TASK_ID,
        "status": "layer1_compact_reduced_universe_interim_contract_built_ready_for_strategy_center_planning",
        "diagnostic_only": True,
        "compact_layer0_status": compact_readiness.get("status", ""),
        "primary_variant": compact_readiness.get("primary_variant", "top250_core_conditional_buffer50"),
        "watchlist_reference_excluded_from_layer1_source_scope": True,
        "contract_rows": int(len(contract)),
        "weekly_snapshot_count": int(contract["snapshot_date"].nunique()),
        "unique_ticker_count_all": int(contract["ticker"].nunique()),
        "average_weekly_count_all": float(all_row["average_weekly_count"]),
        "p2_contract_rows": int(p2_row["contract_rows"]),
        "p2_average_weekly_count": float(p2_row["average_weekly_count"]),
        "p2_unique_ticker_count": int(p2_row["unique_ticker_count"]),
        "monthly_revenue_available_share": float(all_row["monthly_revenue_available_share"]),
        "quarterly_fundamental_available_share": float(all_row["quarterly_fundamental_available_share"]),
        "liquidity_context_available_share": float(all_row["liquidity_context_available_share"]),
        "listing_status_available_share": float(all_row["listing_status_available_share"]),
        "ready_for_layer1_compact_candidate_quality_diagnostic_planning": bool(ready),
        "ready_for_experiments": False,
        "ready_for_formal": False,
        "ready_for_strategy_replay": False,
        "ready_for_t164_mass_download": False,
        "t164_stage2_scoped_to_compact_active_passers_only": True,
        "future_data_violation_count": future_count,
        "blocked_fields": ["free_float_market_cap", "exact_daily_market_cap", "full_industry_taxonomy", "full_t164_remaining_fields"],
        "proxy_fields": ["capital_stock_x_close_market_cap_proxy", "industry_proxy", "t164_capex_proxy", "receivables_basket_proxy"],
        "formal_model_changed": False,
        "trade_decision_changed": False,
        "active_in_trade_decision": False,
        "report_changed": False,
        "portfolio_replay_executed": False,
        "not_live_rule": True,
        "forward_returns_live_rule_usage": False,
    }


def _summary(readiness: dict[str, Any]) -> str:
    return f"""# Layer1 compact reduced-universe interim contract

## Verdict
- status={readiness["status"]}
- primary_variant={readiness["primary_variant"]}
- contract_rows={readiness["contract_rows"]}
- weekly_snapshot_count={readiness["weekly_snapshot_count"]}
- average_weekly_count_all={readiness["average_weekly_count_all"]}
- unique_ticker_count_all={readiness["unique_ticker_count_all"]}
- p2_contract_rows={readiness["p2_contract_rows"]}
- p2_average_weekly_count={readiness["p2_average_weekly_count"]}
- p2_unique_ticker_count={readiness["p2_unique_ticker_count"]}
- monthly_revenue_available_share={readiness["monthly_revenue_available_share"]}
- quarterly_fundamental_available_share={readiness["quarterly_fundamental_available_share"]}
- watchlist_reference_excluded_from_layer1_source_scope=true
- ready_for_layer1_compact_candidate_quality_diagnostic_planning={str(readiness["ready_for_layer1_compact_candidate_quality_diagnostic_planning"]).lower()}
- ready_for_experiments=false
- ready_for_formal=false
- ready_for_t164_mass_download=false

## Plain Summary
This contract rebuilds Layer1 interim quality-floor fields on compact Layer0 active scope only. Watchlist reference rows are excluded from the materialized Layer1 source scope. The contract preserves monthly revenue YoY/3M, quarterly profitability/margins/EPS, traded-value liquidity context, listing context, and candidate-only bottom20/bottom30 quality-floor variants. It remains diagnostic/source readiness only and is not a formal selector.

## Flags
- formal_model_changed=false
- trade_decision_changed=false
- active_in_trade_decision=false
- report_changed=false
- portfolio_replay_executed=false
- ready_for_strategy_replay=false
- not_live_rule=true
- forward_returns_live_rule_usage=false
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compact-layer0-dir", default=str(DEFAULT_COMPACT_LAYER0_DIR))
    parser.add_argument("--previous-layer1-dir", default=str(DEFAULT_PREVIOUS_LAYER1_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()
    manifest = build_contract(
        compact_layer0_dir=args.compact_layer0_dir,
        previous_layer1_dir=args.previous_layer1_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
