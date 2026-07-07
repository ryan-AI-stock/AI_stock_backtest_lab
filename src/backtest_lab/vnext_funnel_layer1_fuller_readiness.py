"""Build fuller vNext Layer 1 PIT readiness from Radar missing-source ledger.

This extends the refreshed Layer 1 diagnostic contract with newly staged
missing-source candidates such as TPEx market-cap proxy and explicit blocked
ledgers. It remains contract/readiness only: no selector, no formal model, no
trade decision, and no replay.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from backtest_lab.vnext_funnel_layer1_refreshed_readiness import (
    CAPITAL_STOCK_DIR,
    DEFAULT_MATERIALIZATION_DIR,
    LISTING_STATUS_DIR,
    MONTHLY_REVENUE_DIR,
    QUARTERLY_FUNDAMENTALS_DIR,
    _attention_slice,
    _capital_stock_features,
    _latest_asof,
    _listing_status_features,
    _monthly_revenue_features,
    _quarterly_fundamental_features,
    _read_json,
    _read_optional_csv,
    _weekly_universe,
    _write_csv,
)


TASK_ID = "TASK-BACKTEST-CORE-VNEXT-LAYER1-FULLER-PIT-CONTRACT-FROM-RADAR-MISSING-SOURCE-READINESS-001"
DEFAULT_RADAR_MISSING_SOURCE_DIR = Path(
    r"C:\Users\zergv\Documents\Codex\2026-05-23\ai-stock-rotation-radar-https-docs\outputs"
    r"\radar_vnext_layer1_missing_fundamental_source_acquisition_readiness_20260707"
)
DEFAULT_OUTPUT_DIR = Path("outputs/vnext_layer1_fuller_pit_contract_from_radar_missing_source_20260707")
TPEX_MARKET_CAP_DIR = Path(
    r"C:\Users\zergv\Documents\Codex\2026-05-23\ai-stock-rotation-radar-https-docs\outputs"
    r"\radar_dynamic_pool1_market_cap_twse_route_tpex_full_sweep_20260703"
)
SECTOR_TAXONOMY_DIR = Path(
    r"C:\Users\zergv\Documents\Codex\2026-05-23\ai-stock-rotation-radar-https-docs\outputs"
    r"\radar_dynamic_pool1_sector_taxonomy_readiness_20260704"
)

PERIODS = [
    ("P1", "2015-01-02", "2022-12-29"),
    ("P2", "2023-01-02", "2026-06-30"),
    ("2024-latest", "2024-01-02", "2026-06-30"),
    ("2026YTD", "2026-01-02", "2026-06-30"),
]

MISSINGNESS_FIELDS = [
    "monthly_revenue_yoy",
    "monthly_revenue_mom",
    "monthly_revenue_rolling_3m_yoy",
    "quarterly_revenue_yoy",
    "gross_margin",
    "operating_margin",
    "eps_yoy",
    "tpex_market_cap_proxy",
    "capital_stock",
    "issued_shares",
    "debt_to_equity_proxy",
    "debt_to_assets_proxy",
    "solvency_source_available",
    "listing_board_proxy",
    "twse_industry_diagnostic_proxy_available",
    "average_traded_value_proxy_available",
    "turnover_proxy_available",
]


def build_fuller_layer1_readiness(
    *,
    materialization_dir: str | Path = DEFAULT_MATERIALIZATION_DIR,
    radar_missing_source_dir: str | Path = DEFAULT_RADAR_MISSING_SOURCE_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    materialization = Path(materialization_dir)
    radar_dir = Path(radar_missing_source_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    radar_readiness = _read_json(radar_dir / "readiness_for_core_layer1_fuller_contract_refresh.json")
    missing_inventory = _read_optional_csv(radar_dir / "layer1_missing_fundamental_source_inventory.csv")
    radar_blocked = _read_optional_csv(radar_dir / "layer1_blocked_proxy_fields_ledger.csv")
    timing_risk = _read_optional_csv(radar_dir / "layer1_pit_timing_risk_ledger.csv")
    sector_readiness = _read_json(SECTOR_TAXONOMY_DIR / "readiness_for_core.json")

    universe = _weekly_universe(materialization / "vnext_weekly_candidate_snapshot.csv")
    attention = _attention_slice(materialization / "attention_features.csv", universe)
    monthly = _monthly_revenue_features(MONTHLY_REVENUE_DIR)
    quarterly = _quarterly_fundamental_features(QUARTERLY_FUNDAMENTALS_DIR)
    capital = _capital_stock_features(CAPITAL_STOCK_DIR)
    listing = _listing_status_features(LISTING_STATUS_DIR)
    tpex_market_cap = _tpex_market_cap_features(TPEX_MARKET_CAP_DIR, universe)

    joined = _candidate_join_contract(universe, attention, monthly, quarterly, capital, listing, tpex_market_cap)
    pit_contract = _pit_contract(joined)
    missingness = _missingness_by_period(joined)
    source_quality = _source_quality_matrix(
        joined,
        radar_readiness=radar_readiness,
        missing_inventory=missing_inventory,
        sector_readiness=sector_readiness,
    )
    blocked = _blocked_proxy_fields(source_quality, radar_blocked)
    future_audit = _future_data_audit(joined, radar_readiness, timing_risk)
    readiness = _readiness_json(pit_contract, joined, missingness, source_quality, blocked, future_audit, radar_readiness)

    _write_csv(pit_contract, output / "funnel_layer1_fuller_fundamental_pit_contract.csv")
    _write_csv(joined, output / "funnel_layer1_fuller_candidate_join_contract.csv")
    _write_csv(source_quality, output / "funnel_layer1_fuller_source_quality_matrix.csv")
    _write_csv(missingness, output / "funnel_layer1_fuller_missingness_by_period.csv")
    _write_csv(blocked, output / "funnel_layer1_fuller_blocked_proxy_fields.csv")
    _write_csv(future_audit, output / "funnel_layer1_fuller_future_data_audit.csv")
    (output / "readiness_for_funnel_layer1_fuller_diagnostic.json").write_text(
        json.dumps(readiness, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    manifest = {
        "task_id": TASK_ID,
        "status": readiness["status"],
        "output_dir": str(output.resolve()),
        "input_materialization_dir": str(materialization.resolve()),
        "radar_missing_source_dir": str(radar_dir.resolve()),
        "radar_missing_source_commit": "26168ad",
        "output_files": [
            "funnel_layer1_fuller_fundamental_pit_contract.csv",
            "funnel_layer1_fuller_candidate_join_contract.csv",
            "funnel_layer1_fuller_source_quality_matrix.csv",
            "funnel_layer1_fuller_missingness_by_period.csv",
            "funnel_layer1_fuller_blocked_proxy_fields.csv",
            "funnel_layer1_fuller_future_data_audit.csv",
            "readiness_for_funnel_layer1_fuller_diagnostic.json",
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
    (output / "final_summary_zh.md").write_text(_summary(readiness, blocked), encoding="utf-8")
    return manifest


def _tpex_market_cap_features(source_dir: Path, universe: pd.DataFrame) -> pd.DataFrame:
    shard_dir = source_dir / "tpex_full_sweep_shards"
    paths = sorted(shard_dir.glob("accepted_tpex_market_cap_rows_*.csv"))
    if not paths:
        fallback = source_dir / "proxy_market_cap_rows.csv"
        paths = [fallback] if fallback.exists() else []
    dates = set(universe["signal_date"].dt.strftime("%Y-%m-%d"))
    tickers = set(universe["ticker"].astype(str))
    usecols = [
        "ticker",
        "date",
        "market",
        "market_cap",
        "free_float_market_cap",
        "shares_outstanding",
        "source_date",
        "available_date",
        "source_type",
        "formal_exact",
        "derivation",
    ]
    frames = []
    for path in paths:
        chunk = pd.read_csv(path, usecols=lambda col: col in usecols, dtype={"ticker": str})
        chunk = chunk[chunk["date"].astype(str).isin(dates) & chunk["ticker"].isin(tickers)]
        if not chunk.empty:
            frames.append(chunk)
    if not frames:
        return pd.DataFrame(columns=["ticker", "tpex_market_cap_date"])
    market_cap = pd.concat(frames, ignore_index=True)
    market_cap["ticker"] = market_cap["ticker"].astype(str)
    market_cap["tpex_market_cap_date"] = pd.to_datetime(market_cap["date"], errors="coerce")
    market_cap["tpex_market_cap_available_date"] = pd.to_datetime(market_cap["available_date"], errors="coerce")
    market_cap["tpex_market_cap_proxy"] = pd.to_numeric(market_cap["market_cap"], errors="coerce")
    market_cap["tpex_shares_outstanding_proxy"] = pd.to_numeric(market_cap["shares_outstanding"], errors="coerce")
    market_cap["tpex_market_cap_source_tier"] = "proxy"
    market_cap["tpex_market_cap_lag_policy"] = "TPEx daily source_date equals signal trading date; formal_exact=false"
    return market_cap.rename(
        columns={
            "market": "tpex_market_cap_market_scope",
            "source_date": "tpex_market_cap_source_date",
            "source_type": "tpex_market_cap_source_type",
            "formal_exact": "tpex_market_cap_formal_exact",
            "derivation": "tpex_market_cap_derivation",
        }
    ).reindex(
        columns=[
            "ticker",
            "tpex_market_cap_date",
            "tpex_market_cap_available_date",
            "tpex_market_cap_proxy",
            "free_float_market_cap",
            "tpex_shares_outstanding_proxy",
            "tpex_market_cap_market_scope",
            "tpex_market_cap_source_date",
            "tpex_market_cap_source_type",
            "tpex_market_cap_formal_exact",
            "tpex_market_cap_derivation",
            "tpex_market_cap_source_tier",
            "tpex_market_cap_lag_policy",
        ]
    )


def _candidate_join_contract(
    universe: pd.DataFrame,
    attention: pd.DataFrame,
    monthly: pd.DataFrame,
    quarterly: pd.DataFrame,
    capital: pd.DataFrame,
    listing: pd.DataFrame,
    tpex_market_cap: pd.DataFrame,
) -> pd.DataFrame:
    joined = universe.merge(
        attention,
        left_on=["signal_date", "ticker"],
        right_on=["trade_date", "ticker"],
        how="left",
    ).drop(columns=["trade_date"], errors="ignore")
    keys = joined[["signal_date", "ticker"]].drop_duplicates()
    joined = joined.merge(_latest_asof(keys, monthly, "monthly_available_date"), on=["signal_date", "ticker"], how="left")
    joined = joined.merge(_latest_asof(keys, quarterly, "quarterly_available_date"), on=["signal_date", "ticker"], how="left")
    joined = joined.merge(_latest_asof(keys, capital, "capital_available_date"), on=["signal_date", "ticker"], how="left")
    joined = joined.merge(_latest_asof(keys, listing, "listing_event_date"), on=["signal_date", "ticker"], how="left")
    joined = joined.merge(
        tpex_market_cap,
        left_on=["signal_date", "ticker"],
        right_on=["tpex_market_cap_date", "ticker"],
        how="left",
    ).drop(columns=["tpex_market_cap_date"], errors="ignore")

    joined["average_traded_value_proxy_available"] = joined["traded_value"].notna() if "traded_value" in joined else False
    turnover_cols = [c for c in ["turnover_5d", "turnover_20d", "turnover_60d"] if c in joined]
    joined["turnover_proxy_available"] = joined[turnover_cols].notna().any(axis=1) if turnover_cols else False
    joined["listing_board_proxy"] = joined["listing_board_proxy_from_monthly_revenue"].combine_first(
        joined["listing_board_proxy_from_quarterly"]
    ).combine_first(joined["listing_board_from_event"])
    joined["monthly_revenue_available"] = joined["monthly_revenue_value"].notna()
    joined["quarterly_fundamental_available"] = joined["quarterly_report_period"].notna()
    joined["capital_stock_proxy_available"] = joined["capital_stock"].notna()
    joined["tpex_market_cap_proxy_available"] = joined["tpex_market_cap_proxy"].notna()
    joined["listing_status_partial_available"] = joined["listing_status_partial_event_available"].fillna(False).astype(bool)
    joined["debt_to_assets_proxy"] = joined["total_liabilities"] / joined["total_assets"]
    joined["solvency_source_available"] = joined[["total_liabilities", "total_assets", "equity"]].notna().any(axis=1)
    joined["twse_industry_diagnostic_proxy_available"] = False
    joined["twse_industry_diagnostic_proxy"] = pd.NA
    joined["free_float_market_cap_available"] = False
    joined["twse_exact_daily_market_cap_available"] = False
    joined["operating_cash_flow_quality_available"] = False
    joined["free_cash_flow_quality_available"] = False
    joined["current_ratio_available"] = False
    joined["inventory_risk_available"] = False
    joined["receivable_risk_available"] = False
    joined["source_timing_policy"] = "merge_asof_backward_or_exact_signal_date_available_date_lte_signal_date"
    joined["forward_return_as_rule"] = False
    joined["not_live_rule"] = True
    joined["diagnostic_only"] = True
    return joined


def _pit_contract(joined: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "signal_date",
        "ticker",
        "name",
        "theme_id",
        "theme_name",
        "valid_universe",
        "fundamental_pass",
        "market_attention_member",
        "eligible_pool_member",
        "rank_overall",
        "risk_score",
        "risk_bucket",
        "monthly_report_period",
        "monthly_available_date",
        "monthly_revenue_value",
        "monthly_revenue_yoy",
        "monthly_revenue_mom",
        "monthly_revenue_rolling_3m_yoy",
        "quarterly_report_period",
        "quarterly_available_date",
        "operating_revenue",
        "quarterly_revenue_yoy",
        "gross_margin",
        "operating_margin",
        "eps",
        "eps_yoy",
        "net_income",
        "operating_income",
        "operating_income_yoy",
        "total_assets",
        "total_liabilities",
        "equity",
        "debt_to_equity_proxy",
        "debt_to_assets_proxy",
        "solvency_source_available",
        "tpex_market_cap_available_date",
        "tpex_market_cap_proxy",
        "tpex_shares_outstanding_proxy",
        "tpex_market_cap_source_type",
        "tpex_market_cap_source_tier",
        "tpex_market_cap_lag_policy",
        "capital_report_period",
        "capital_available_date",
        "capital_stock",
        "issued_shares",
        "capital_market_scope",
        "capital_source_tier",
        "capital_lag_policy",
        "listing_board_proxy",
        "listing_event_type",
        "listing_event_date",
        "listing_status_source_tier",
        "listing_status_lag_policy",
        "twse_industry_diagnostic_proxy",
        "twse_industry_diagnostic_proxy_available",
        "traded_value",
        "turnover_5d",
        "turnover_20d",
        "turnover_60d",
        "monthly_revenue_available",
        "quarterly_fundamental_available",
        "tpex_market_cap_proxy_available",
        "capital_stock_proxy_available",
        "listing_status_partial_available",
        "average_traded_value_proxy_available",
        "turnover_proxy_available",
        "free_float_market_cap_available",
        "twse_exact_daily_market_cap_available",
        "operating_cash_flow_quality_available",
        "free_cash_flow_quality_available",
        "current_ratio_available",
        "inventory_risk_available",
        "receivable_risk_available",
        "source_timing_policy",
        "forward_return_as_rule",
        "not_live_rule",
        "diagnostic_only",
    ]
    return joined.reindex(columns=cols)


def _missingness_by_period(joined: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for period, start, end in PERIODS:
        subset = joined[(joined["signal_date"] >= pd.Timestamp(start)) & (joined["signal_date"] <= pd.Timestamp(end))]
        for field in MISSINGNESS_FIELDS:
            if field.endswith("_available"):
                available = subset[field].astype(bool) if field in subset else pd.Series([], dtype=bool)
            else:
                available = subset[field].notna() if field in subset else pd.Series([], dtype=bool)
            rows.append(
                {
                    "period": period,
                    "requested_start": start,
                    "requested_end": end,
                    "actual_start": subset["signal_date"].min() if not subset.empty else pd.NaT,
                    "actual_end": subset["signal_date"].max() if not subset.empty else pd.NaT,
                    "field": field,
                    "rows": int(len(subset)),
                    "available_rows": int(available.sum()) if len(subset) else 0,
                    "missing_rows": int(len(subset) - available.sum()) if len(subset) else 0,
                    "available_share": float(available.mean()) if len(subset) else 0.0,
                    "diagnostic_only": True,
                }
            )
    return pd.DataFrame(rows)


def _source_quality_matrix(
    joined: pd.DataFrame,
    *,
    radar_readiness: dict[str, Any],
    missing_inventory: pd.DataFrame,
    sector_readiness: dict[str, Any],
) -> pd.DataFrame:
    def count(field: str) -> int:
        return int(joined[field].notna().sum()) if field in joined else 0

    def bool_count(field: str) -> int:
        return int(joined[field].astype(bool).sum()) if field in joined else 0

    rows = [
        ("monthly_revenue_growth", "PIT-ready", count("monthly_revenue_yoy"), "MOPS monthly revenue as-of available_date; conservative timing"),
        ("quarterly_growth_margin_profitability", "PIT-ready", count("quarterly_revenue_yoy"), "quarterly fundamentals as-of available_date; exact filing timestamp unavailable"),
        ("tpex_market_cap_proxy", "proxy", count("tpex_market_cap_proxy"), "TPEx daily market cap proxy from official close * issued shares; TWSE exact daily market cap still blocked"),
        ("twse_capital_stock_shares_proxy", "proxy", count("capital_stock"), "TWSE capital stock/shares quarterly proxy; not daily exact market cap"),
        ("debt_leverage_solvency", "blocked_or_sparse_actual_rows_zero", bool_count("solvency_source_available"), "Radar route says derivable, but accepted quarterly total_assets/total_liabilities/equity are empty locally"),
        ("listing_board_status_partial_proxy", "proxy", bool_count("listing_status_partial_available"), "partial event/status source; master_ready=false"),
        ("twse_industry_diagnostic_proxy", "proxy_not_materialized", 0, "TWSE official industry diagnostic route noted; no accepted PIT industry rows materialized in Core package"),
        ("average_traded_value", "proxy", bool_count("average_traded_value_proxy_available"), "existing attention_features signal-date traded_value"),
        ("turnover", "proxy", bool_count("turnover_proxy_available"), "existing attention_features turnover windows"),
        ("free_float_market_cap", "blocked", 0, "no local free-float shares/free-float cap route"),
        ("operating_cash_flow_quality", "blocked", 0, "cash-flow statement full sweep not materialized"),
        ("free_cash_flow_quality", "blocked", 0, "OCF/capex fields unavailable; do not proxy from profitability"),
        ("current_ratio", "blocked", 0, "current assets/current liabilities not materialized"),
        ("inventory_risk", "blocked", 0, "inventory detail not materialized"),
        ("receivable_risk", "blocked", 0, "receivable detail not materialized"),
        ("tpex_all_stock_sector_pit", "blocked", 0, "accepted TPEx sector rows=0"),
        ("twse_exact_daily_market_cap", "blocked", 0, "TWSE direct daily market cap / issued shares route remains blocked"),
        ("forward_return_as_rule", "prohibited", 0, "forward returns prohibited as Layer 1 rule inputs"),
    ]
    out = pd.DataFrame(
        [
            {
                "field_group": field,
                "source_quality": quality,
                "available_rows": available,
                "source_quality_reason": reason,
                "usable_for_layer1_fuller_diagnostic": quality in {"PIT-ready", "proxy", "proxy_not_materialized"},
                "usable_for_formal": False,
                "diagnostic_only": True,
            }
            for field, quality, available, reason in rows
        ]
    )
    out["radar_ready_for_core_layer1_fuller_contract_refresh"] = bool(
        radar_readiness.get("ready_for_core_layer1_fuller_contract_refresh", False)
    )
    out["radar_ready_for_experiments"] = bool(radar_readiness.get("ready_for_experiments", False))
    out["radar_missing_inventory_rows"] = int(len(missing_inventory))
    out["twse_official_industry_diagnostic_only"] = bool(
        sector_readiness.get("twse_official_industry_diagnostic_only", False)
    )
    return out


def _blocked_proxy_fields(source_quality: pd.DataFrame, radar_blocked: pd.DataFrame) -> pd.DataFrame:
    out = source_quality.copy()
    out["status"] = out["source_quality"].map(
        lambda quality: "prohibited"
        if quality == "prohibited"
        else "blocked"
        if "blocked" in str(quality)
        else "proxy"
        if "proxy" in str(quality)
        else "PIT-ready"
    )
    out["proxy_available"] = out["status"].eq("proxy")
    out = out.rename(columns={"field_group": "field_or_contract", "source_quality_reason": "blocked_or_proxy_reason"})[
        ["field_or_contract", "status", "source_quality", "proxy_available", "blocked_or_proxy_reason", "diagnostic_only"]
    ]
    if radar_blocked.empty:
        out["radar_ledger_status"] = pd.NA
        return out
    radar = radar_blocked.rename(columns={"field": "field_or_contract"})
    radar = radar[["field_or_contract", "status", "blocked_or_proxy_reason", "ready_for_core_contract"]]
    return out.merge(radar, on="field_or_contract", how="left", suffixes=("", "_radar"))


def _future_data_audit(joined: pd.DataFrame, radar_readiness: dict[str, Any], timing_risk: pd.DataFrame) -> pd.DataFrame:
    checks = [
        ("monthly_available_date_lte_signal_date", "monthly_available_date"),
        ("quarterly_available_date_lte_signal_date", "quarterly_available_date"),
        ("capital_available_date_lte_signal_date", "capital_available_date"),
        ("listing_event_date_lte_signal_date", "listing_event_date"),
        ("tpex_market_cap_available_date_lte_signal_date", "tpex_market_cap_available_date"),
    ]
    rows = []
    for audit_item, col in checks:
        dates = pd.to_datetime(joined.get(col), errors="coerce")
        bad_count = int((dates.notna() & (dates > joined["signal_date"])).sum())
        rows.append(
            {
                "audit_item": audit_item,
                "status": "passed" if bad_count == 0 else "failed",
                "future_data_violation_count": bad_count,
                "note": f"{col} joined by backward/as-of or exact signal-date PIT policy",
            }
        )
    rows.append(
        {
            "audit_item": "radar_missing_source_future_data_audit",
            "status": "passed" if int(radar_readiness.get("future_data_violation_count", 0)) == 0 else "failed",
            "future_data_violation_count": int(radar_readiness.get("future_data_violation_count", 0)),
            "note": f"Radar timing risk ledger rows={len(timing_risk)}",
        }
    )
    rows.append(
        {
            "audit_item": "forward_return_as_rule",
            "status": "passed",
            "future_data_violation_count": 0,
            "note": "no forward return columns are included in fuller Layer 1 contracts",
        }
    )
    return pd.DataFrame(rows)


def _readiness_json(
    pit_contract: pd.DataFrame,
    joined: pd.DataFrame,
    missingness: pd.DataFrame,
    source_quality: pd.DataFrame,
    blocked: pd.DataFrame,
    future_audit: pd.DataFrame,
    radar_readiness: dict[str, Any],
) -> dict[str, Any]:
    future_count = int(future_audit["future_data_violation_count"].sum())
    core_ready = bool(radar_readiness.get("ready_for_core_layer1_fuller_contract_refresh", False))
    added_proxy = bool(source_quality[source_quality["field_group"].eq("tpex_market_cap_proxy")]["available_rows"].sum() > 0)
    pit_ready_growth = bool(source_quality[source_quality["field_group"].eq("monthly_revenue_growth")]["available_rows"].sum() > 0)
    ready = core_ready and added_proxy and pit_ready_growth and future_count == 0
    source_upgrade = "material" if added_proxy else "minor"
    return {
        "date": "2026-07-07",
        "task_id": TASK_ID,
        "owner": "BACKTEST_LAB Core/Data",
        "status": "partial_ready_layer1_fuller_pit_contract_proxy_limited" if ready else "blocked_layer1_fuller_pit_contract",
        "ready_for_funnel_layer1_fuller_quality_floor_diagnostic": bool(ready),
        "ready_for_funnel_layer1_fuller_event_diagnostic": bool(ready),
        "layer1_fuller_exact_coverage": "partial" if ready else "blocked",
        "layer1_source_upgrade_vs_refreshed": source_upgrade,
        "quality_floor_not_top_alpha_selector": True,
        "ready_for_layer2_diagnostic": False,
        "ready_for_portfolio_like_diagnostic": False,
        "ready_for_strategy_replay": False,
        "ready_for_formal": False,
        "future_data_violation_count": future_count,
        "not_live_rule": True,
        "forward_returns_live_rule_usage": False,
        "candidate_join_rows": int(len(joined)),
        "pit_contract_rows": int(len(pit_contract)),
        "source_quality_rows": int(len(source_quality)),
        "missingness_rows": int(len(missingness)),
        "blocked_fields": blocked[blocked["status"].eq("blocked")]["field_or_contract"].tolist(),
        "proxy_fields": blocked[blocked["status"].eq("proxy")]["field_or_contract"].tolist(),
        "PIT_ready_fields": blocked[blocked["status"].eq("PIT-ready")]["field_or_contract"].tolist(),
        "formal_model_changed": False,
        "trade_decision_changed": False,
        "active_in_trade_decision": False,
        "report_changed": False,
        "portfolio_replay_executed": False,
    }


def _summary(readiness: dict[str, Any], blocked: pd.DataFrame) -> str:
    return "\n".join(
        [
            "# vNext Layer1 Fuller PIT Contract Readiness",
            "",
            f"Status: {readiness['status']}",
            "",
            "Boundary: Layer1 quality-floor / eligibility-filter contract only; no Experiments, no replay, no formal/report/trade change.",
            "",
            "Readiness:",
            f"- ready_for_funnel_layer1_fuller_quality_floor_diagnostic={str(readiness['ready_for_funnel_layer1_fuller_quality_floor_diagnostic']).lower()}",
            f"- layer1_fuller_exact_coverage={readiness['layer1_fuller_exact_coverage']}",
            f"- layer1_source_upgrade_vs_refreshed={readiness['layer1_source_upgrade_vs_refreshed']}",
            "- ready_for_layer2_diagnostic=false",
            "- ready_for_portfolio_like_diagnostic=false",
            "- ready_for_strategy_replay=false",
            "- ready_for_formal=false",
            f"- future_data_violation_count={readiness['future_data_violation_count']}",
            "- not_live_rule=true",
            "- forward_returns_live_rule_usage=false",
            "",
            "Blocked / proxy fields:",
            *[
                f"- {row.field_or_contract}: {row.status}; {row.blocked_or_proxy_reason}"
                for row in blocked.itertuples()
            ],
            "",
            "Next handoff:",
            "- vNext Research should judge whether this partial fuller package is enough to ask Experiments for Layer1 quality-floor diagnostic.",
            "- Do not start Layer2 until Layer1 fuller diagnostic receives Research / Strategy Center GO.",
            "",
            "Flags:",
            "- formal_model_changed=false",
            "- trade_decision_changed=false",
            "- active_in_trade_decision=false",
            "- report_changed=false",
            "- portfolio_replay_executed=false",
            "- ready_for_strategy_replay=false",
        ]
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--materialization-dir", type=Path, default=DEFAULT_MATERIALIZATION_DIR)
    parser.add_argument("--radar-missing-source-dir", type=Path, default=DEFAULT_RADAR_MISSING_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    manifest = build_fuller_layer1_readiness(
        materialization_dir=args.materialization_dir,
        radar_missing_source_dir=args.radar_missing_source_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
