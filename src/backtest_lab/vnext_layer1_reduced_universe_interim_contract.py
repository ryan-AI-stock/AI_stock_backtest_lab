"""Build Layer1 reduced-universe interim quality-floor contract.

This materializes diagnostic-only Layer1 fields for Layer0 top300_buffer100
weekly active passers. It does not run Experiments, replay, formal selector
changes, report changes, trade decisions, or t164 mass download.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


TASK_ID = "TASK-BACKTEST-CORE-VNEXT-LAYER1-REDUCED-UNIVERSE-INTERIM-CONTRACT-BUILD-001"
DEFAULT_LAYER0_DIR = Path("outputs/vnext_layer0_weekly_universe_snapshot_contract_20260707")
DEFAULT_PLANNING_DIR = Path("outputs/vnext_layer1_reduced_universe_source_planning_from_layer0_20260707")
DEFAULT_OUTPUT_DIR = Path("outputs/vnext_layer1_reduced_universe_interim_contract_20260707")
RADAR_ROOT = Path(r"C:\Users\zergv\Documents\Codex\2026-05-23\ai-stock-rotation-radar-https-docs\outputs")
MONTHLY_REVENUE_DIR = RADAR_ROOT / "radar_dynamic_pool1_mops_monthly_revenue_full_universe_pit_20260703"
QUARTERLY_FUNDAMENTALS_DIR = RADAR_ROOT / "radar_dynamic_pool1_quarterly_fundamentals_full_sweep_20260703"

PERIODS = {
    "P1": ("2015-01-02", "2022-12-29"),
    "P2": ("2023-01-02", "2026-06-30"),
    "2024_latest": ("2024-01-02", "2026-06-30"),
    "2026YTD": ("2026-01-02", "2026-06-30"),
}


def build_contract(
    *,
    layer0_dir: str | Path = DEFAULT_LAYER0_DIR,
    planning_dir: str | Path = DEFAULT_PLANNING_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    layer0 = Path(layer0_dir)
    planning = Path(planning_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    planning_readiness = _read_json(planning / "readiness_for_layer1_reduced_universe_source_planning.json")
    layer0_snapshot = _read_layer0_snapshot(layer0 / "layer0_weekly_universe_snapshot.csv")
    monthly = _read_monthly_revenue()
    quarterly = _read_quarterly_fundamentals()

    contract = _build_materialized_contract(layer0_snapshot, monthly, quarterly)
    contract = _add_quality_floor_candidates(contract)

    coverage = _coverage_by_period(contract)
    missingness = _missingness_by_period(contract)
    source_quality = _source_quality_matrix()
    variants = _quality_floor_variant_design()
    blocked_proxy = _blocked_proxy_fields()
    future_audit = _future_data_audit(contract)
    readiness = _readiness(planning_readiness, contract, coverage, missingness, future_audit)

    _write_csv(contract, output / "layer1_reduced_universe_interim_contract.csv")
    _write_csv(contract.head(1000), output / "layer1_reduced_universe_interim_contract_sample.csv")
    (output / ".gitignore").write_text(
        "layer1_reduced_universe_interim_contract.csv\n",
        encoding="utf-8",
    )
    _write_csv(coverage, output / "layer1_reduced_universe_interim_coverage_by_period.csv")
    _write_csv(missingness, output / "layer1_reduced_universe_interim_missingness_by_period.csv")
    _write_csv(source_quality, output / "layer1_reduced_universe_interim_source_quality_matrix.csv")
    _write_csv(variants, output / "layer1_reduced_universe_quality_floor_variant_design.csv")
    _write_csv(blocked_proxy, output / "layer1_reduced_universe_interim_blocked_proxy_fields.csv")
    _write_csv(future_audit, output / "layer1_reduced_universe_interim_future_data_audit.csv")
    (output / "readiness_for_layer1_reduced_universe_interim_contract.json").write_text(
        json.dumps(readiness, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    manifest = {
        "task_id": TASK_ID,
        "status": readiness["status"],
        "output_dir": str(output.resolve()),
        "input_layer0_dir": str(layer0.resolve()),
        "input_planning_dir": str(planning.resolve()),
        "input_planning_commit": "e474050",
        "output_files": [
            "layer1_reduced_universe_interim_contract.csv",
            "layer1_reduced_universe_interim_contract_sample.csv",
            "layer1_reduced_universe_interim_coverage_by_period.csv",
            "layer1_reduced_universe_interim_missingness_by_period.csv",
            "layer1_reduced_universe_interim_source_quality_matrix.csv",
            "layer1_reduced_universe_quality_floor_variant_design.csv",
            "layer1_reduced_universe_interim_blocked_proxy_fields.csv",
            "layer1_reduced_universe_interim_future_data_audit.csv",
            "readiness_for_layer1_reduced_universe_interim_contract.json",
            "manifest.json",
            "final_summary_zh.md",
        ],
        "large_local_files_not_tracked": [
            "layer1_reduced_universe_interim_contract.csv"
        ],
        "large_local_file_policy": "full materialized contract is retained in local output path; Git tracks sample/readiness/audit files only",
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


def _read_layer0_snapshot(path: Path) -> pd.DataFrame:
    cols = [
        "snapshot_date",
        "variant",
        "ticker",
        "name",
        "market",
        "traded_value_5d",
        "traded_value_20d",
        "traded_value_60d",
        "traded_value_rank_5d",
        "traded_value_rank_20d",
        "traded_value_rank_60d",
        "rank_improvement_5d_vs_60d",
        "cumulative_traded_value_share_5d",
        "selection_bucket",
        "surge_exception",
        "listing_status",
        "liquidity_flag",
        "is_ky_name_proxy",
        "instrument_type_source_quality",
        "market_cap_rank_source_quality",
        "event_ledger_source_quality",
    ]
    df = pd.read_csv(path, usecols=cols, dtype={"ticker": str})
    df = df[df["variant"].eq("top300_buffer100")].copy()
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
    for col in [
        "traded_value_5d",
        "traded_value_20d",
        "traded_value_60d",
        "traded_value_rank_5d",
        "traded_value_rank_20d",
        "traded_value_rank_60d",
        "rank_improvement_5d_vs_60d",
        "cumulative_traded_value_share_5d",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.sort_values(["ticker", "snapshot_date"])


def _read_monthly_revenue() -> pd.DataFrame:
    files = sorted((MONTHLY_REVENUE_DIR / "accepted_monthly_revenue_rows_shards").glob("accepted_monthly_revenue_rows_*.csv"))
    frames = [
        pd.read_csv(
            f,
            dtype={"ticker": str},
            usecols=["ticker", "revenue_year_month", "revenue_value", "available_date", "formal_exact", "pit_usable"],
        )
        for f in files
    ]
    df = pd.concat(frames, ignore_index=True)
    df["available_date"] = pd.to_datetime(df["available_date"], errors="coerce")
    df["revenue_value"] = pd.to_numeric(df["revenue_value"], errors="coerce")
    df["revenue_year_month"] = pd.PeriodIndex(df["revenue_year_month"], freq="M")
    df = df.sort_values(["ticker", "revenue_year_month"])
    df["monthly_revenue_yoy"] = df.groupby("ticker")["revenue_value"].pct_change(12)
    df["revenue_3m_sum"] = df.groupby("ticker")["revenue_value"].rolling(3, min_periods=3).sum().reset_index(level=0, drop=True)
    df["monthly_revenue_3m_yoy"] = df.groupby("ticker")["revenue_3m_sum"].pct_change(12)
    df["monthly_revenue_source_quality"] = "source_candidate_conservative_available_date"
    return df[
        [
            "ticker",
            "available_date",
            "revenue_year_month",
            "revenue_value",
            "monthly_revenue_yoy",
            "monthly_revenue_3m_yoy",
            "monthly_revenue_source_quality",
        ]
    ].sort_values(["ticker", "available_date"])


def _read_quarterly_fundamentals() -> pd.DataFrame:
    files = sorted((QUARTERLY_FUNDAMENTALS_DIR / "shards").glob("accepted_quarterly_fundamentals_rows_*.csv"))
    usecols = [
        "ticker",
        "fiscal_year",
        "quarter",
        "available_date",
        "statement_profile",
        "operating_revenue",
        "gross_profit",
        "operating_income",
        "net_income",
        "eps",
        "roe",
        "gross_margin",
        "operating_margin",
        "formal_exact",
    ]
    frames = [pd.read_csv(f, dtype={"ticker": str}, usecols=usecols) for f in files]
    df = pd.concat(frames, ignore_index=True)
    df["available_date"] = pd.to_datetime(df["available_date"], errors="coerce")
    for col in ["operating_revenue", "gross_profit", "operating_income", "net_income", "eps", "roe", "gross_margin", "operating_margin"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["quarterly_fundamental_source_quality"] = "source_candidate_conservative_statutory_deadline"
    df["report_period"] = df["fiscal_year"].astype(str) + "Q" + df["quarter"].astype(str)
    return df[
        [
            "ticker",
            "available_date",
            "report_period",
            "statement_profile",
            "operating_revenue",
            "gross_profit",
            "operating_income",
            "net_income",
            "eps",
            "roe",
            "gross_margin",
            "operating_margin",
            "quarterly_fundamental_source_quality",
        ]
    ].sort_values(["ticker", "available_date"])


def _merge_asof_by_ticker(left: pd.DataFrame, right: pd.DataFrame, *, suffix: str) -> pd.DataFrame:
    pieces = []
    right_groups = {ticker: part.sort_values("available_date") for ticker, part in right.groupby("ticker", sort=False)}
    for ticker, left_part in left.groupby("ticker", sort=False):
        right_part = right_groups.get(ticker)
        base = left_part.sort_values("snapshot_date")
        if right_part is None or right_part.empty:
            pieces.append(base)
            continue
        merged = pd.merge_asof(
            base,
            right_part.drop(columns=["ticker"]),
            left_on="snapshot_date",
            right_on="available_date",
            direction="backward",
            allow_exact_matches=True,
            suffixes=("", suffix),
        )
        pieces.append(merged)
    return pd.concat(pieces, ignore_index=True)


def _build_materialized_contract(snapshot: pd.DataFrame, monthly: pd.DataFrame, quarterly: pd.DataFrame) -> pd.DataFrame:
    df = _merge_asof_by_ticker(snapshot, monthly, suffix="_monthly")
    df = df.rename(columns={"available_date": "monthly_revenue_available_date"})
    df = _merge_asof_by_ticker(df, quarterly, suffix="_quarterly")
    df = df.rename(columns={"available_date": "quarterly_fundamental_available_date"})

    df["monthly_revenue_available"] = df["monthly_revenue_available_date"].notna()
    df["quarterly_fundamental_available"] = df["quarterly_fundamental_available_date"].notna()
    df["liquidity_context_available"] = df["traded_value_5d"].notna()
    df["listing_status_available"] = df["listing_status"].notna()
    df["market_cap_proxy_available"] = False
    df["market_cap_proxy_source_quality"] = "proxy_not_materialized_in_interim_contract"
    df["t164_cashflow_current_ratio_inventory_receivable_available"] = False
    df["t164_source_stage"] = "stage2_scoped_only_not_in_interim"
    df["diagnostic_only"] = True
    df["not_live_rule"] = True
    df["forward_returns_live_rule_usage"] = False
    df["formal_model_changed"] = False
    df["trade_decision_changed"] = False
    df["active_in_trade_decision"] = False
    df["report_changed"] = False
    return df.sort_values(["snapshot_date", "traded_value_rank_5d", "ticker"])


def _add_quality_floor_candidates(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["negative_revenue_yoy_flag"] = out["monthly_revenue_yoy"].lt(0)
    out["negative_revenue_3m_yoy_flag"] = out["monthly_revenue_3m_yoy"].lt(0)
    out["negative_eps_flag"] = out["eps"].lt(0)
    out["negative_operating_income_flag"] = out["operating_income"].lt(0)
    out["low_gross_margin_flag"] = out["gross_margin"].lt(0)
    out["missing_core_fundamental_flag"] = ~(out["monthly_revenue_available"] & out["quarterly_fundamental_available"])

    risk_cols = [
        "negative_revenue_yoy_flag",
        "negative_revenue_3m_yoy_flag",
        "negative_eps_flag",
        "negative_operating_income_flag",
        "low_gross_margin_flag",
        "missing_core_fundamental_flag",
    ]
    out["layer1_financial_risk_flag_count"] = out[risk_cols].fillna(False).astype(int).sum(axis=1)
    out["layer1_quality_floor_risk_score_candidate"] = out["layer1_financial_risk_flag_count"]
    out["layer1_quality_floor_risk_pctile_by_week"] = out.groupby("snapshot_date")[
        "layer1_quality_floor_risk_score_candidate"
    ].rank(pct=True, method="average")
    out["layer1_exclude_bottom20_candidate"] = out["layer1_quality_floor_risk_pctile_by_week"].ge(0.80)
    out["layer1_exclude_bottom30_candidate"] = out["layer1_quality_floor_risk_pctile_by_week"].ge(0.70)
    out["layer1_quality_floor_variant_basis"] = "candidate_only_exclude_high_risk_bottom20_bottom30_not_top_only_selector"
    return out


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
    return {
        "period": period,
        "requested_start": requested_start,
        "requested_end": requested_end,
        "actual_start": str(c["snapshot_date"].min().date()) if not c.empty else "",
        "actual_end": str(c["snapshot_date"].max().date()) if not c.empty else "",
        "weekly_snapshot_count": int(c["snapshot_date"].nunique()),
        "contract_rows": int(len(c)),
        "unique_ticker_count": int(c["ticker"].nunique()),
        "monthly_revenue_available_share": float(c["monthly_revenue_available"].mean()) if not c.empty else 0.0,
        "quarterly_fundamental_available_share": float(c["quarterly_fundamental_available"].mean()) if not c.empty else 0.0,
        "liquidity_context_available_share": float(c["liquidity_context_available"].mean()) if not c.empty else 0.0,
        "listing_status_available_share": float(c["listing_status_available"].mean()) if not c.empty else 0.0,
        "diagnostic_only": True,
    }


def _missingness_by_period(contract: pd.DataFrame) -> pd.DataFrame:
    fields = [
        "monthly_revenue_yoy",
        "monthly_revenue_3m_yoy",
        "eps",
        "gross_margin",
        "operating_margin",
        "roe",
        "traded_value_5d",
        "listing_status",
    ]
    rows = []
    periods = {"ALL": (None, None), **PERIODS}
    for period, bounds in periods.items():
        start, end = bounds
        mask = pd.Series(True, index=contract.index)
        if start:
            mask &= contract["snapshot_date"].ge(pd.Timestamp(start))
        if end:
            mask &= contract["snapshot_date"].le(pd.Timestamp(end))
        c = contract[mask]
        for field in fields:
            rows.append(
                {
                    "period": period,
                    "field": field,
                    "row_count": int(len(c)),
                    "missing_count": int(c[field].isna().sum()) if field in c else int(len(c)),
                    "missing_share": float(c[field].isna().mean()) if field in c and len(c) else 1.0,
                    "diagnostic_only": True,
                }
            )
    return pd.DataFrame(rows)


def _source_quality_matrix() -> pd.DataFrame:
    rows = [
        ("monthly_revenue_yoy_3m", "source_candidate_ready", "conservative_available_date", "not_formal_exact"),
        ("quarterly_profitability_margins_eps", "source_candidate_ready", "conservative_statutory_deadline", "not_formal_exact"),
        ("liquidity_traded_value_context", "pit_diagnostic_ready", "Layer0 official daily traded value", "primary_context"),
        ("listing_status_basic_context", "partial", "Layer0/listing metadata partial", "not_full_cross_market_master"),
        ("capital_stock_x_close_market_cap_proxy", "proxy_not_materialized", "diagnostic proxy only", "not_hard_gate"),
        ("t164_cashflow_current_ratio_inventory_receivable", "stage2_blocked_in_interim", "scope only to Layer0 active passers later", "no_1900_mass_download"),
    ]
    return pd.DataFrame(rows, columns=["field_group", "source_quality", "timing_policy", "contract_policy"]).assign(
        diagnostic_only=True,
        accepted_for_formal=False,
    )


def _quality_floor_variant_design() -> pd.DataFrame:
    rows = [
        ("exclude_bottom20", "exclude worst 20% by interim financial risk score within each weekly Layer0 universe", "candidate_only", False),
        ("exclude_bottom30", "exclude worst 30% by interim financial risk score within each weekly Layer0 universe", "candidate_only", False),
        ("financial_risk_flags", "negative revenue growth, negative EPS, negative operating income, low gross margin, missing core fundamentals", "candidate_only", False),
        ("not_top_only", "do not select top fundamental names; use as quality floor before Layer2/3", "required_boundary", False),
    ]
    return pd.DataFrame(rows, columns=["variant", "definition", "status", "formal_ready"]).assign(diagnostic_only=True)


def _blocked_proxy_fields() -> pd.DataFrame:
    rows = [
        ("free_float_market_cap", "blocked", "not available; no silent fill"),
        ("exact_daily_market_cap", "blocked", "capital_stock_x_close remains proxy only and not materialized here"),
        ("full_industry_taxonomy", "blocked_or_proxy", "do not use as formal theme/industry floor"),
        ("full_listing_disposition_delivery_event_master", "partial", "listing/status context only; not full exclusion master"),
        ("t164_cashflow", "stage2_scoped_only", "not in interim; only for Layer0 active passers later"),
        ("current_ratio_inventory_receivable_capex_proxy", "stage2_scoped_only_proxy_or_human_review", "not in interim; no formal-ready claim"),
    ]
    return pd.DataFrame(rows, columns=["field", "status", "policy"]).assign(diagnostic_only=True)


def _future_data_audit(contract: pd.DataFrame) -> pd.DataFrame:
    monthly_bad = contract["monthly_revenue_available_date"].notna() & (
        pd.to_datetime(contract["monthly_revenue_available_date"]) > contract["snapshot_date"]
    )
    quarterly_bad = contract["quarterly_fundamental_available_date"].notna() & (
        pd.to_datetime(contract["quarterly_fundamental_available_date"]) > contract["snapshot_date"]
    )
    return pd.DataFrame(
        [
            ("monthly_revenue_available_date_lte_snapshot", "passed" if not monthly_bad.any() else "failed", int(monthly_bad.sum()), "as-of merge uses only available_date <= snapshot_date"),
            ("quarterly_fundamental_available_date_lte_snapshot", "passed" if not quarterly_bad.any() else "failed", int(quarterly_bad.sum()), "as-of merge uses only available_date <= snapshot_date"),
            ("forward_return_as_rule", "passed", 0, "no forward returns used"),
            ("formal_selector", "not_applicable", 0, "candidate quality-floor contract only"),
        ],
        columns=["audit_item", "status", "future_data_violation_count", "note"],
    )


def _readiness(
    planning_readiness: dict[str, Any],
    contract: pd.DataFrame,
    coverage: pd.DataFrame,
    missingness: pd.DataFrame,
    future_audit: pd.DataFrame,
) -> dict[str, Any]:
    all_cov = coverage[coverage["period"].eq("ALL")].iloc[0].to_dict()
    max_future_violations = int(future_audit["future_data_violation_count"].max())
    monthly_share = float(all_cov["monthly_revenue_available_share"])
    quarterly_share = float(all_cov["quarterly_fundamental_available_share"])
    ready_for_diagnostic_planning = max_future_violations == 0 and monthly_share >= 0.80 and quarterly_share >= 0.80
    return {
        "task_id": TASK_ID,
        "status": "layer1_reduced_universe_interim_contract_built_diagnostic_ready_for_strategy_center_planning",
        "diagnostic_only": True,
        "layer0_variant": "top300_buffer100",
        "planning_status": planning_readiness.get("status", ""),
        "contract_rows": int(len(contract)),
        "weekly_snapshot_count": int(contract["snapshot_date"].nunique()),
        "unique_ticker_count_all": int(contract["ticker"].nunique()),
        "monthly_revenue_available_share": monthly_share,
        "quarterly_fundamental_available_share": quarterly_share,
        "liquidity_context_available_share": float(all_cov["liquidity_context_available_share"]),
        "listing_status_available_share": float(all_cov["listing_status_available_share"]),
        "ready_for_layer1_reduced_universe_candidate_quality_diagnostic_planning": bool(ready_for_diagnostic_planning),
        "ready_for_experiments": False,
        "ready_for_formal": False,
        "ready_for_strategy_replay": False,
        "ready_for_t164_mass_download": False,
        "t164_stage2_scoped_to_layer0_active_passers_only": True,
        "future_data_violation_count": max_future_violations,
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
    return f"""# Layer1 reduced-universe interim contract

## Verdict
- status={readiness["status"]}
- layer0_variant={readiness["layer0_variant"]}
- contract_rows={readiness["contract_rows"]}
- weekly_snapshot_count={readiness["weekly_snapshot_count"]}
- unique_ticker_count_all={readiness["unique_ticker_count_all"]}
- monthly_revenue_available_share={readiness["monthly_revenue_available_share"]}
- quarterly_fundamental_available_share={readiness["quarterly_fundamental_available_share"]}
- ready_for_layer1_reduced_universe_candidate_quality_diagnostic_planning={str(readiness["ready_for_layer1_reduced_universe_candidate_quality_diagnostic_planning"]).lower()}
- ready_for_experiments=false
- ready_for_formal=false
- ready_for_t164_mass_download=false

## Plain Summary
This interim contract joins Layer0 top300_buffer100 weekly active passers to existing low-cost Layer1 sources: monthly revenue YoY/3M, quarterly profitability/margins/EPS, liquidity/traded-value context, and basic listing/investability context. Quality-floor variants are candidate-only exclude_bottom20/exclude_bottom30 risk flags, not a top-only selector and not a live rule. t164 cashflow/current-ratio/inventory/receivables remain stage2 scoped fields for Layer0 active passers only.

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
    parser.add_argument("--planning-dir", default=str(DEFAULT_PLANNING_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()
    manifest = build_contract(layer0_dir=args.layer0_dir, planning_dir=args.planning_dir, output_dir=args.output_dir)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
