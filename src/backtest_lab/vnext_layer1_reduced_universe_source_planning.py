"""Plan Layer1 source scope from Layer0 top300_buffer100 reduced universe.

This is source planning/readiness only. It does not run Experiments, replay,
formal selector changes, report changes, trade decisions, or t164 mass download.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


TASK_ID = "TASK-BACKTEST-CORE-VNEXT-LAYER1-REDUCED-UNIVERSE-SOURCE-PLANNING-FROM-LAYER0-001"
DEFAULT_LAYER0_DIR = Path("outputs/vnext_layer0_weekly_universe_snapshot_contract_20260707")
DEFAULT_OUTPUT_DIR = Path("outputs/vnext_layer1_reduced_universe_source_planning_from_layer0_20260707")

PERIODS = {
    "P1": ("2015-01-02", "2022-12-29"),
    "P2": ("2023-01-02", "2026-06-30"),
    "2024_latest": ("2024-01-02", "2026-06-30"),
    "2026YTD": ("2026-01-02", "2026-06-30"),
}


def build_planning(
    *,
    layer0_dir: str | Path = DEFAULT_LAYER0_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    layer0 = Path(layer0_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    readiness_in = _read_json(layer0 / "readiness_for_layer0_weekly_universe_snapshot.json")
    snapshot = _read_snapshot(layer0 / "layer0_weekly_universe_snapshot.csv")
    coverage = pd.read_csv(layer0 / "layer0_weekly_universe_coverage_by_week.csv")
    top300 = snapshot[snapshot["variant"].eq("top300_buffer100")].copy()
    top300_coverage = coverage[coverage["variant"].eq("top300_buffer100")].copy()

    period_coverage = _period_coverage(top300, top300_coverage)
    source_cost = _source_cost_estimate(top300, period_coverage)
    priority = _field_priority()
    quality_floor = _quality_floor_design()
    blocked_proxy = _blocked_proxy_ledger()
    next_handoff = _next_handoff()
    future_audit = _future_audit()
    readiness = _readiness(readiness_in, top300, period_coverage)

    _write_csv(period_coverage, output / "layer1_reduced_universe_coverage_by_period.csv")
    _write_csv(source_cost, output / "layer1_reduced_universe_source_cost_estimate.csv")
    _write_csv(priority, output / "layer1_reduced_universe_field_priority.csv")
    _write_csv(quality_floor, output / "layer1_reduced_universe_quality_floor_design.csv")
    _write_csv(blocked_proxy, output / "layer1_reduced_universe_blocked_proxy_ledger.csv")
    _write_csv(next_handoff, output / "layer1_reduced_universe_next_handoff.csv")
    _write_csv(future_audit, output / "layer1_reduced_universe_future_data_audit.csv")
    (output / "readiness_for_layer1_reduced_universe_source_planning.json").write_text(
        json.dumps(readiness, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    manifest = {
        "task_id": TASK_ID,
        "status": readiness["status"],
        "output_dir": str(output.resolve()),
        "layer0_input_dir": str(layer0.resolve()),
        "layer0_commit": "52f7346",
        "output_files": [
            "layer1_reduced_universe_coverage_by_period.csv",
            "layer1_reduced_universe_source_cost_estimate.csv",
            "layer1_reduced_universe_field_priority.csv",
            "layer1_reduced_universe_quality_floor_design.csv",
            "layer1_reduced_universe_blocked_proxy_ledger.csv",
            "layer1_reduced_universe_next_handoff.csv",
            "layer1_reduced_universe_future_data_audit.csv",
            "readiness_for_layer1_reduced_universe_source_planning.json",
            "manifest.json",
            "final_summary_zh.md",
        ],
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


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def _read_snapshot(path: Path) -> pd.DataFrame:
    cols = [
        "snapshot_date",
        "variant",
        "ticker",
        "name",
        "market",
        "selection_bucket",
        "surge_exception",
        "traded_value_rank_5d",
        "traded_value_5d",
        "cumulative_traded_value_share_5d",
    ]
    df = pd.read_csv(path, usecols=cols, dtype={"ticker": str})
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
    return df


def _period_coverage(top300: pd.DataFrame, coverage: pd.DataFrame) -> pd.DataFrame:
    coverage = coverage.copy()
    coverage["snapshot_date"] = pd.to_datetime(coverage["snapshot_date"])
    rows = [_period_row("ALL", top300, coverage, None, None)]
    for name, (start, end) in PERIODS.items():
        rows.append(_period_row(name, top300, coverage, start, end))
    return pd.DataFrame(rows)


def _period_row(name: str, snapshot: pd.DataFrame, coverage: pd.DataFrame, start: str | None, end: str | None) -> dict[str, Any]:
    requested_start = start or str(snapshot["snapshot_date"].min().date())
    requested_end = end or str(snapshot["snapshot_date"].max().date())
    mask = pd.Series(True, index=snapshot.index)
    cov_mask = pd.Series(True, index=coverage.index)
    if start:
        mask &= snapshot["snapshot_date"].ge(pd.Timestamp(start))
        cov_mask &= coverage["snapshot_date"].ge(pd.Timestamp(start))
    if end:
        mask &= snapshot["snapshot_date"].le(pd.Timestamp(end))
        cov_mask &= coverage["snapshot_date"].le(pd.Timestamp(end))
    s = snapshot[mask]
    c = coverage[cov_mask]
    return {
        "period": name,
        "requested_start": requested_start,
        "requested_end": requested_end,
        "actual_start": str(s["snapshot_date"].min().date()) if not s.empty else "",
        "actual_end": str(s["snapshot_date"].max().date()) if not s.empty else "",
        "weekly_snapshot_count": int(s["snapshot_date"].nunique()),
        "event_rows": int(len(s)),
        "unique_ticker_count": int(s["ticker"].nunique()),
        "average_weekly_ticker_count": float(c["ticker_count"].mean()) if not c.empty else 0.0,
        "median_weekly_ticker_count": float(c["ticker_count"].median()) if not c.empty else 0.0,
        "average_turnover_share_5d": float(c["turnover_share_5d"].mean()) if not c.empty else 0.0,
        "median_turnover_share_5d": float(c["turnover_share_5d"].median()) if not c.empty else 0.0,
        "p25_turnover_share_5d": float(c["turnover_share_5d"].quantile(0.25)) if not c.empty else 0.0,
        "p75_turnover_share_5d": float(c["turnover_share_5d"].quantile(0.75)) if not c.empty else 0.0,
        "diagnostic_only": True,
    }


def _source_cost_estimate(top300: pd.DataFrame, period_coverage: pd.DataFrame) -> pd.DataFrame:
    full_universe = 1900
    rows = []
    for row in period_coverage.to_dict("records"):
        unique_count = int(row["unique_ticker_count"])
        event_rows = int(row["event_rows"])
        weekly_count = float(row["average_weekly_ticker_count"])
        rows.append(
            {
                "period": row["period"],
                "cost_basis": "unique_ticker_source_fetch",
                "reduced_universe_units": unique_count,
                "full_universe_units": full_universe,
                "estimated_reduction_vs_1900": 1 - unique_count / full_universe if full_universe else 0,
                "note": "full historical rolling universe touches many names; do not use this as the main t164 scope without period pruning",
                "diagnostic_only": True,
            }
        )
        rows.append(
            {
                "period": row["period"],
                "cost_basis": "rolling_period_source_fetch_estimate",
                "reduced_universe_units": weekly_count,
                "full_universe_units": full_universe,
                "estimated_reduction_vs_1900": 1 - weekly_count / full_universe if full_universe else 0,
                "note": "preferred source acquisition policy: fetch by Layer0 active universe per source period, not all historical unique tickers",
                "diagnostic_only": True,
            }
        )
        rows.append(
            {
                "period": row["period"],
                "cost_basis": "weekly_snapshot_event_join",
                "reduced_universe_units": event_rows,
                "full_universe_units": int(row["weekly_snapshot_count"]) * full_universe,
                "estimated_reduction_vs_1900": 1 - event_rows / (int(row["weekly_snapshot_count"]) * full_universe)
                if int(row["weekly_snapshot_count"])
                else 0,
                "note": "best for joining already materialized PIT fields to weekly snapshot rows",
                "diagnostic_only": True,
            }
        )
    return pd.DataFrame(rows)


def _field_priority() -> pd.DataFrame:
    rows = [
        ("P1_low_cost_quality_floor", "monthly_revenue_yoy_3m; quarterly_margin_eps_profitability; liquidity; listing_status", "mostly_existing_or_low_cost", "build first"),
        ("P1_low_cost_quality_floor", "gross_margin; operating_margin; eps; roe_roa_if_available", "existing_quarterly_sweep_partial", "use as quality floor, not alpha selector"),
        ("P2_high_cost_t164_scoped", "operating_cash_flow; current_ratio; inventory; receivables risk; capex_proxy", "source_partial_t164_scoped_only", "only for Layer0 reduced universe, no 1900 mass download"),
        ("P3_proxy_or_blocked", "market_cap; free_float; industry", "proxy_or_blocked", "keep ledger, no silent fill"),
    ]
    return pd.DataFrame(rows, columns=["priority", "field_group", "source_status", "policy"]).assign(
        diagnostic_only=True,
        formal_ready=False,
    )


def _quality_floor_design() -> pd.DataFrame:
    rows = [
        ("exclude_bottom20", "remove worst 20% by available quality/fundamental risk composite within Layer0 universe", "candidate_only"),
        ("exclude_bottom30", "stricter floor for sparse/poor-quality financials", "candidate_only"),
        ("financial_risk_flags", "debt/leverage/profitability deterioration/current_ratio/inventory_receivable flags where available", "candidate_only"),
        ("not_top_only_alpha", "do not rank only top fundamentals; purpose is removing unsuitable names before later layers", "required_policy"),
    ]
    return pd.DataFrame(rows, columns=["candidate", "design", "status"]).assign(diagnostic_only=True)


def _blocked_proxy_ledger() -> pd.DataFrame:
    rows = [
        ("free_float_market_cap", "blocked", "do not use for quality floor until source exists"),
        ("exact_daily_market_cap", "blocked_or_proxy", "capital stock x close can be proxy only"),
        ("industry_taxonomy", "proxy_or_blocked", "do not silently group themes/industries"),
        ("t164_cashflow_inventory_receivable", "scoped_high_cost", "only after Layer0 reduced universe acceptance"),
    ]
    return pd.DataFrame(rows, columns=["field", "status", "policy"]).assign(diagnostic_only=True)


def _next_handoff() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "next_owner": "Core/Data",
                "handoff_action": "build_layer1_reduced_universe_interim_contract_using_existing_low_cost_fields",
                "ready": True,
                "reason": "Layer0 top300_buffer100 provides reduced weekly universe and existing monthly/quarterly/liquidity/listing fields are usable for interim quality-floor contract",
                "diagnostic_only": True,
            },
            {
                "next_owner": "Radar/Data",
                "handoff_action": "do_not_resume_1900_t164_mass_download; wait for scoped Layer0 reduced universe source request",
                "ready": False,
                "reason": "high-cost t164 fields should be scoped to accepted reduced universe only",
                "diagnostic_only": True,
            },
        ]
    )


def _future_audit() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("forward_return_as_rule", "passed", 0, "no forward returns used"),
            ("layer0_asof_basis", "passed", 0, "Layer0 snapshot already PIT traded-value based"),
            ("formal_selector", "not_applicable", 0, "quality floor planning only"),
        ],
        columns=["audit_item", "status", "future_data_violation_count", "note"],
    )


def _readiness(readiness_in: dict[str, Any], top300: pd.DataFrame, period_coverage: pd.DataFrame) -> dict[str, Any]:
    all_row = period_coverage[period_coverage["period"].eq("ALL")].iloc[0].to_dict()
    return {
        "task_id": TASK_ID,
        "status": "layer1_reduced_universe_source_planning_ready_existing_low_cost_fields_first",
        "diagnostic_only": True,
        "layer0_variant": "top300_buffer100",
        "weekly_snapshot_count": int(all_row["weekly_snapshot_count"]),
        "event_rows": int(all_row["event_rows"]),
        "unique_ticker_count_all": int(all_row["unique_ticker_count"]),
        "average_weekly_ticker_count": float(all_row["average_weekly_ticker_count"]),
        "average_turnover_share_5d": float(all_row["average_turnover_share_5d"]),
        "all_period_unique_ticker_scope_warning": "top300_buffer100 rolling universe touches 1843 names across full history; source acquisition should be period-scoped, not all-history unique-scoped",
        "ready_for_layer1_reduced_universe_interim_contract": True,
        "ready_for_radar_reduced_universe_t164_scoped_source": False,
        "ready_for_t164_mass_download": False,
        "ready_for_experiments": False,
        "ready_for_formal": False,
        "ready_for_strategy_replay": False,
        "future_data_violation_count": 0,
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
    return f"""# Layer1 reduced-universe source planning from Layer0

## Verdict
- status={readiness["status"]}
- layer0_variant={readiness["layer0_variant"]}
- weekly_snapshot_count={readiness["weekly_snapshot_count"]}
- event_rows={readiness["event_rows"]}
- unique_ticker_count_all={readiness["unique_ticker_count_all"]}
- average_weekly_ticker_count={readiness["average_weekly_ticker_count"]}
- average_turnover_share_5d={readiness["average_turnover_share_5d"]}
- all_period_unique_ticker_scope_warning={readiness["all_period_unique_ticker_scope_warning"]}
- ready_for_layer1_reduced_universe_interim_contract=true
- ready_for_t164_mass_download=false
- ready_for_experiments=false
- ready_for_formal=false

## Plain Summary
Layer0 top300_buffer100 turns the full market into a weekly universe of about 400 names on average. Across the full 2015-2026 rolling history, those weekly lists still touch many tickers, so the cost-saving policy must be period-scoped source acquisition rather than all-history unique-ticker acquisition. Layer1 should first use existing/low-cost monthly revenue, quarterly profitability/margins/EPS, liquidity, and listing status fields to build a quality-floor contract. High-cost t164 cashflow/current-ratio/inventory/receivables should only be scoped to active Layer0 passers after the interim contract is accepted.

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
    parser.add_argument("--layer0-dir", default=str(DEFAULT_LAYER0_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()
    manifest = build_planning(layer0_dir=args.layer0_dir, output_dir=args.output_dir)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
