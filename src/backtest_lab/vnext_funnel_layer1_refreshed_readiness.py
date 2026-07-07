"""Build refreshed vNext funnel Layer 1 PIT readiness from Radar sources.

This is diagnostic contract/readiness only. It joins PIT-safe source candidates
from Radar/Data into the vNext weekly signal-date universe without creating a
selector, live rule, or portfolio replay.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


TASK_ID = "TASK-BACKTEST-CORE-VNEXT-FUNNEL-LAYER1-REFRESHED-PIT-CONTRACT-READINESS-001"
MERGED_TASK_IDS = [
    "TASK-BACKTEST-CORE-VNEXT-FUNNEL-LAYER1-REFRESHED-PIT-CONTRACT-FROM-RADAR-SOURCE-INVENTORY-001",
    TASK_ID,
]

DEFAULT_MATERIALIZATION_DIR = Path("outputs/vnext_dynamic_candidate_pool_data_materialization_20260706")
DEFAULT_RADAR_INVENTORY_DIR = Path(
    r"C:\Users\zergv\Documents\Codex\2026-05-23\ai-stock-rotation-radar-https-docs\outputs"
    r"\radar_vnext_funnel_layer1_fundamental_source_inventory_readiness_20260706"
)
DEFAULT_OUTPUT_DIR = Path("outputs/vnext_funnel_layer1_refreshed_pit_contract_from_radar_20260707")

MONTHLY_REVENUE_DIR = Path(
    r"C:\Users\zergv\Documents\Codex\2026-05-23\ai-stock-rotation-radar-https-docs\outputs"
    r"\radar_dynamic_pool1_mops_monthly_revenue_full_universe_pit_20260703"
)
QUARTERLY_FUNDAMENTALS_DIR = Path(
    r"C:\Users\zergv\Documents\Codex\2026-05-23\ai-stock-rotation-radar-https-docs\outputs"
    r"\radar_dynamic_pool1_quarterly_fundamentals_full_sweep_20260703"
)
CAPITAL_STOCK_DIR = Path(
    r"C:\Users\zergv\Documents\Codex\2026-05-23\ai-stock-rotation-radar-https-docs\outputs"
    r"\radar_dynamic_pool1_twse_capital_stock_full_sweep_proxy_contract_20260703"
)
LISTING_STATUS_DIR = Path(
    r"C:\Users\zergv\Documents\Codex\2026-05-23\ai-stock-rotation-radar-https-docs\outputs"
    r"\radar_dynamic_pool1_listing_delisting_suspension_master_20260703"
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
    "eps_yoy",
    "operating_income_yoy",
    "gross_margin",
    "operating_margin",
    "roe",
    "roa",
    "capital_stock",
    "issued_shares",
    "listing_board_proxy",
    "average_traded_value_proxy_available",
    "turnover_proxy_available",
]


def build_refreshed_layer1_readiness(
    *,
    materialization_dir: str | Path = DEFAULT_MATERIALIZATION_DIR,
    radar_inventory_dir: str | Path = DEFAULT_RADAR_INVENTORY_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    materialization = Path(materialization_dir)
    radar_inventory = Path(radar_inventory_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    radar_readiness = _read_json(radar_inventory / "readiness_for_core_layer1_source_package.json")
    radar_inventory_table = _read_optional_csv(radar_inventory / "funnel_layer1_source_inventory.csv")
    radar_source_candidates = _read_optional_csv(radar_inventory / "funnel_layer1_existing_local_source_candidates.csv")

    universe = _weekly_universe(materialization / "vnext_weekly_candidate_snapshot.csv")
    attention = _attention_slice(materialization / "attention_features.csv", universe)
    monthly = _monthly_revenue_features(MONTHLY_REVENUE_DIR)
    quarterly = _quarterly_fundamental_features(QUARTERLY_FUNDAMENTALS_DIR)
    capital = _capital_stock_features(CAPITAL_STOCK_DIR)
    listing = _listing_status_features(LISTING_STATUS_DIR)

    joined = _candidate_join_contract(universe, attention, monthly, quarterly, capital, listing)
    pit_contract = _pit_contract(joined)
    missingness = _missingness_by_period(joined)
    source_quality = _source_quality_matrix(
        joined,
        radar_readiness=radar_readiness,
        radar_inventory=radar_inventory_table,
        radar_source_candidates=radar_source_candidates,
    )
    blocked = _blocked_proxy_fields(source_quality)
    future_audit = _future_data_audit(joined, radar_readiness)
    readiness = _readiness_json(pit_contract, joined, missingness, source_quality, blocked, future_audit, radar_readiness)

    _write_csv(pit_contract, output / "funnel_layer1_refreshed_pit_contract.csv")
    _write_csv(joined, output / "funnel_layer1_refreshed_candidate_join_contract.csv")
    _write_csv(missingness, output / "funnel_layer1_refreshed_missingness_by_period.csv")
    _write_csv(source_quality, output / "funnel_layer1_refreshed_source_quality_matrix.csv")
    _write_csv(blocked, output / "funnel_layer1_refreshed_blocked_proxy_fields.csv")
    _write_csv(future_audit, output / "funnel_layer1_refreshed_future_data_audit.csv")
    (output / "readiness_for_funnel_layer1_refreshed_diagnostic.json").write_text(
        json.dumps(readiness, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    manifest = {
        "task_id": TASK_ID,
        "merged_task_ids": MERGED_TASK_IDS,
        "status": readiness["status"],
        "output_dir": str(output.resolve()),
        "input_materialization_dir": str(materialization.resolve()),
        "radar_inventory_dir": str(radar_inventory.resolve()),
        "radar_source_inventory_commit": "ae52a5a",
        "output_files": [
            "funnel_layer1_refreshed_pit_contract.csv",
            "funnel_layer1_refreshed_candidate_join_contract.csv",
            "funnel_layer1_refreshed_missingness_by_period.csv",
            "funnel_layer1_refreshed_source_quality_matrix.csv",
            "funnel_layer1_refreshed_blocked_proxy_fields.csv",
            "funnel_layer1_refreshed_future_data_audit.csv",
            "readiness_for_funnel_layer1_refreshed_diagnostic.json",
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


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_optional_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _weekly_universe(path: Path) -> pd.DataFrame:
    usecols = [
        "snapshot_date",
        "ticker",
        "name",
        "theme_id",
        "theme_name",
        "valid_universe",
        "fundamental_pass",
        "market_attention_member",
        "eligible_pool_member",
        "case_trace_only",
        "diagnostic_only",
        "rank_overall",
        "turnover_state",
        "risk_score",
        "risk_bucket",
    ]
    raw = pd.read_csv(path, usecols=usecols, parse_dates=["snapshot_date"])
    raw = raw[raw["diagnostic_only"].astype(bool) & ~raw["case_trace_only"].astype(bool)].copy()
    raw["ticker"] = raw["ticker"].astype(str)
    return raw.rename(columns={"snapshot_date": "signal_date"})


def _attention_slice(path: Path, universe: pd.DataFrame) -> pd.DataFrame:
    dates = set(universe["signal_date"].dt.strftime("%Y-%m-%d"))
    tickers = set(universe["ticker"])
    wanted = [
        "trade_date",
        "ticker",
        "traded_value",
        "volume",
        "turnover_5d",
        "turnover_20d",
        "turnover_60d",
        "turnover_rank_pct_5d",
        "turnover_rank_pct_20d",
        "turnover_rank_pct_60d",
        "traded_value_rank_pct",
        "distribution_risk",
    ]
    header = pd.read_csv(path, nrows=0)
    usecols = [col for col in wanted if col in header.columns]
    parts = []
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=500_000):
        chunk["ticker"] = chunk["ticker"].astype(str)
        chunk = chunk[chunk["trade_date"].astype(str).isin(dates) & chunk["ticker"].isin(tickers)]
        if not chunk.empty:
            parts.append(chunk)
    if not parts:
        return pd.DataFrame(columns=usecols)
    out = pd.concat(parts, ignore_index=True)
    out["trade_date"] = pd.to_datetime(out["trade_date"])
    return out


def _monthly_revenue_features(source_dir: Path) -> pd.DataFrame:
    shard_dir = source_dir / "accepted_monthly_revenue_rows_shards"
    paths = sorted(shard_dir.glob("accepted_monthly_revenue_rows_*.csv"))
    if not paths:
        sample = source_dir / "accepted_monthly_revenue_rows_sample.csv"
        paths = [sample] if sample.exists() else []
    usecols = [
        "ticker",
        "market",
        "revenue_year_month",
        "revenue_value",
        "source_date",
        "release_date",
        "available_date",
        "source_type",
        "formal_exact",
        "pit_usable",
    ]
    frames = []
    for path in paths:
        frames.append(pd.read_csv(path, usecols=lambda col: col in usecols, dtype={"ticker": str}))
    if not frames:
        return pd.DataFrame(columns=["ticker", "monthly_available_date"])
    monthly = pd.concat(frames, ignore_index=True)
    monthly["ticker"] = monthly["ticker"].astype(str)
    monthly["monthly_available_date"] = pd.to_datetime(monthly["available_date"], errors="coerce")
    monthly["monthly_report_period"] = monthly["revenue_year_month"].astype(str)
    monthly["monthly_revenue_value"] = pd.to_numeric(monthly["revenue_value"], errors="coerce")
    monthly = monthly.sort_values(["ticker", "monthly_report_period", "monthly_available_date"])
    monthly["monthly_revenue_mom"] = monthly.groupby("ticker")["monthly_revenue_value"].pct_change(1)
    monthly["monthly_revenue_yoy"] = monthly.groupby("ticker")["monthly_revenue_value"].pct_change(12)
    rolling_3m = monthly.groupby("ticker")["monthly_revenue_value"].transform(lambda series: series.rolling(3, min_periods=3).sum())
    monthly["monthly_revenue_rolling_3m_yoy"] = rolling_3m / rolling_3m.groupby(monthly["ticker"]).shift(12) - 1
    monthly["monthly_revenue_source_tier"] = "higher_quality_diagnostic"
    monthly["monthly_revenue_lag_policy"] = "conservative_available_date_next_month_day10_weekday_adjusted"
    monthly["monthly_revenue_formal_exact"] = monthly.get("formal_exact", False)
    monthly["monthly_revenue_pit_usable"] = monthly.get("pit_usable", True)
    monthly = monthly.rename(
        columns={
            "market": "listing_board_proxy_from_monthly_revenue",
            "source_date": "monthly_revenue_source_date",
            "release_date": "monthly_revenue_release_date",
            "source_type": "monthly_revenue_source_type",
        }
    )
    keep = [
        "ticker",
        "monthly_available_date",
        "monthly_report_period",
        "monthly_revenue_value",
        "monthly_revenue_yoy",
        "monthly_revenue_mom",
        "monthly_revenue_rolling_3m_yoy",
        "monthly_revenue_source_date",
        "monthly_revenue_release_date",
        "monthly_revenue_source_type",
        "monthly_revenue_source_tier",
        "monthly_revenue_lag_policy",
        "monthly_revenue_formal_exact",
        "monthly_revenue_pit_usable",
        "listing_board_proxy_from_monthly_revenue",
    ]
    return monthly.reindex(columns=keep)


def _quarterly_fundamental_features(source_dir: Path) -> pd.DataFrame:
    shard_dir = source_dir / "shards"
    paths = sorted(shard_dir.glob("accepted_quarterly_fundamentals_rows_*.csv"))
    usecols = [
        "ticker",
        "market",
        "fiscal_year",
        "quarter",
        "source_date",
        "available_date",
        "source_type",
        "formal_exact",
        "statement_profile",
        "operating_revenue",
        "gross_profit",
        "operating_income",
        "net_income",
        "eps",
        "total_assets",
        "total_liabilities",
        "equity",
        "roe",
        "gross_margin",
        "operating_margin",
    ]
    frames = []
    for path in paths:
        frames.append(pd.read_csv(path, usecols=lambda col: col in usecols, dtype={"ticker": str}))
    if not frames:
        return pd.DataFrame(columns=["ticker", "quarterly_available_date"])
    quarterly = pd.concat(frames, ignore_index=True)
    quarterly["ticker"] = quarterly["ticker"].astype(str)
    quarterly["quarterly_available_date"] = pd.to_datetime(quarterly["available_date"], errors="coerce")
    quarterly["quarterly_report_period"] = quarterly["fiscal_year"].astype(str) + "Q" + quarterly["quarter"].astype(str)
    numeric_cols = [
        "operating_revenue",
        "gross_profit",
        "operating_income",
        "net_income",
        "eps",
        "total_assets",
        "total_liabilities",
        "equity",
        "roe",
        "gross_margin",
        "operating_margin",
    ]
    for col in numeric_cols:
        if col in quarterly:
            quarterly[col] = pd.to_numeric(quarterly[col], errors="coerce")
    quarterly = quarterly.sort_values(["ticker", "fiscal_year", "quarter", "quarterly_available_date"])
    quarterly["quarterly_revenue_yoy"] = quarterly.groupby("ticker")["operating_revenue"].pct_change(4)
    quarterly["eps_yoy"] = quarterly.groupby("ticker")["eps"].pct_change(4)
    quarterly["operating_income_yoy"] = quarterly.groupby("ticker")["operating_income"].pct_change(4)
    quarterly["roa"] = quarterly["net_income"] / quarterly["total_assets"]
    quarterly["debt_to_equity_proxy"] = quarterly["total_liabilities"] / quarterly["equity"]
    quarterly["quarterly_source_tier"] = "higher_quality_diagnostic"
    quarterly["quarterly_lag_policy"] = "conservative_quarter_available_date_no_exact_company_filing_timestamp"
    quarterly = quarterly.rename(
        columns={
            "market": "listing_board_proxy_from_quarterly",
            "source_date": "quarterly_source_date",
            "source_type": "quarterly_source_type",
            "formal_exact": "quarterly_formal_exact",
        }
    )
    keep = [
        "ticker",
        "quarterly_available_date",
        "quarterly_report_period",
        "quarterly_source_date",
        "quarterly_source_type",
        "quarterly_formal_exact",
        "statement_profile",
        "operating_revenue",
        "quarterly_revenue_yoy",
        "gross_profit",
        "operating_income",
        "operating_income_yoy",
        "net_income",
        "eps",
        "eps_yoy",
        "total_assets",
        "total_liabilities",
        "equity",
        "roe",
        "roa",
        "gross_margin",
        "operating_margin",
        "debt_to_equity_proxy",
        "quarterly_source_tier",
        "quarterly_lag_policy",
        "listing_board_proxy_from_quarterly",
    ]
    return quarterly.reindex(columns=keep)


def _capital_stock_features(source_dir: Path) -> pd.DataFrame:
    path = source_dir / "accepted_twse_capital_stock_rows.csv"
    if not path.exists():
        return pd.DataFrame(columns=["ticker", "capital_available_date"])
    usecols = [
        "ticker",
        "date_or_period",
        "market",
        "capital_stock",
        "issued_shares",
        "source_date",
        "available_date",
        "source_type",
        "formal_exact",
    ]
    capital = pd.read_csv(path, usecols=lambda col: col in usecols, dtype={"ticker": str})
    capital["ticker"] = capital["ticker"].astype(str)
    capital["capital_available_date"] = pd.to_datetime(capital["available_date"], errors="coerce")
    capital["capital_report_period"] = capital["date_or_period"].astype(str)
    capital["capital_stock"] = pd.to_numeric(capital["capital_stock"], errors="coerce")
    capital["issued_shares"] = pd.to_numeric(capital["issued_shares"], errors="coerce")
    capital["capital_source_tier"] = "proxy"
    capital["capital_lag_policy"] = "quarterly_available_date_proxy_no_daily_exact_issued_shares"
    capital = capital.rename(
        columns={
            "market": "capital_market_scope",
            "source_date": "capital_source_date",
            "source_type": "capital_source_type",
            "formal_exact": "capital_formal_exact",
        }
    )
    keep = [
        "ticker",
        "capital_available_date",
        "capital_report_period",
        "capital_stock",
        "issued_shares",
        "capital_market_scope",
        "capital_source_date",
        "capital_source_type",
        "capital_formal_exact",
        "capital_source_tier",
        "capital_lag_policy",
    ]
    return capital.reindex(columns=keep).sort_values(["ticker", "capital_available_date"])


def _listing_status_features(source_dir: Path) -> pd.DataFrame:
    path = source_dir / "accepted_listing_metadata_rows.csv"
    if not path.exists():
        return pd.DataFrame(columns=["ticker", "listing_event_date"])
    listing = pd.read_csv(path, dtype={"ticker": str})
    listing["ticker"] = listing["ticker"].astype(str)
    listing["listing_event_date"] = pd.to_datetime(listing.get("event_date"), errors="coerce")
    listing["listing_source_date"] = pd.to_datetime(listing.get("source_date"), errors="coerce")
    listing["listing_status_partial_event_available"] = True
    listing["listing_status_source_tier"] = "partial_event_proxy"
    listing["listing_status_lag_policy"] = "event_date_or_source_date_partial_master_not_full_listing_history"
    keep = [
        "ticker",
        "market",
        "event_type",
        "listing_event_date",
        "listing_source_date",
        "source_type",
        "formal_ready",
        "listing_status_partial_event_available",
        "listing_status_source_tier",
        "listing_status_lag_policy",
    ]
    out = listing.reindex(columns=keep)
    return out.rename(
        columns={
            "market": "listing_board_from_event",
            "event_type": "listing_event_type",
            "source_type": "listing_source_type",
            "formal_ready": "listing_formal_ready",
        }
    ).sort_values(["ticker", "listing_event_date"])


def _candidate_join_contract(
    universe: pd.DataFrame,
    attention: pd.DataFrame,
    monthly: pd.DataFrame,
    quarterly: pd.DataFrame,
    capital: pd.DataFrame,
    listing: pd.DataFrame,
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
    joined["average_traded_value_proxy_available"] = joined["traded_value"].notna() if "traded_value" in joined else False
    turnover_cols = [c for c in ["turnover_5d", "turnover_20d", "turnover_60d"] if c in joined]
    joined["turnover_proxy_available"] = joined[turnover_cols].notna().any(axis=1) if turnover_cols else False
    joined["listing_board_proxy"] = joined["listing_board_proxy_from_monthly_revenue"].combine_first(
        joined["listing_board_proxy_from_quarterly"]
    ).combine_first(joined["listing_board_from_event"])
    joined["monthly_revenue_available"] = joined["monthly_revenue_value"].notna()
    joined["quarterly_fundamental_available"] = joined["quarterly_report_period"].notna()
    joined["capital_stock_proxy_available"] = joined["capital_stock"].notna()
    joined["listing_status_partial_available"] = joined["listing_status_partial_event_available"].fillna(False).astype(bool)
    joined["market_cap_available"] = False
    joined["free_float_market_cap_available"] = False
    joined["cash_flow_quality_available"] = False
    joined["current_ratio_available"] = False
    joined["inventory_receivable_risk_available"] = False
    joined["industry_sector_available"] = False
    joined["source_timing_policy"] = "merge_asof_backward_source_available_date_lte_signal_date"
    joined["forward_return_as_rule"] = False
    joined["not_live_rule"] = True
    joined["diagnostic_only"] = True
    return joined


def _latest_asof(keys: pd.DataFrame, source: pd.DataFrame, date_col: str) -> pd.DataFrame:
    if source.empty or date_col not in source:
        out = keys.copy()
        return out
    keys = keys.sort_values(["ticker", "signal_date"]).copy()
    source = source.copy()
    source["ticker"] = source["ticker"].astype(str)
    source[date_col] = pd.to_datetime(source[date_col], errors="coerce")
    source = source.dropna(subset=[date_col]).sort_values(["ticker", date_col])
    parts = []
    for ticker, group in keys.groupby("ticker", sort=False):
        source_group = source[source["ticker"].eq(ticker)]
        if source_group.empty:
            parts.append(group.copy())
            continue
        parts.append(
            pd.merge_asof(
                group.sort_values("signal_date"),
                source_group.sort_values(date_col),
                left_on="signal_date",
                right_on=date_col,
                by="ticker",
                direction="backward",
            )
        )
    return pd.concat(parts, ignore_index=True)


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
        "monthly_revenue_source_date",
        "monthly_revenue_release_date",
        "monthly_revenue_source_type",
        "monthly_revenue_source_tier",
        "monthly_revenue_lag_policy",
        "quarterly_report_period",
        "quarterly_available_date",
        "operating_revenue",
        "quarterly_revenue_yoy",
        "gross_profit",
        "operating_income",
        "operating_income_yoy",
        "net_income",
        "eps",
        "eps_yoy",
        "roe",
        "roa",
        "gross_margin",
        "operating_margin",
        "debt_to_equity_proxy",
        "statement_profile",
        "quarterly_source_date",
        "quarterly_source_type",
        "quarterly_source_tier",
        "quarterly_lag_policy",
        "capital_report_period",
        "capital_available_date",
        "capital_stock",
        "issued_shares",
        "capital_market_scope",
        "capital_source_type",
        "capital_source_tier",
        "capital_lag_policy",
        "listing_board_proxy",
        "listing_event_type",
        "listing_event_date",
        "listing_status_source_tier",
        "listing_status_lag_policy",
        "traded_value",
        "turnover_5d",
        "turnover_20d",
        "turnover_60d",
        "traded_value_rank_pct",
        "monthly_revenue_available",
        "quarterly_fundamental_available",
        "capital_stock_proxy_available",
        "listing_status_partial_available",
        "average_traded_value_proxy_available",
        "turnover_proxy_available",
        "market_cap_available",
        "free_float_market_cap_available",
        "cash_flow_quality_available",
        "current_ratio_available",
        "inventory_receivable_risk_available",
        "industry_sector_available",
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
    radar_inventory: pd.DataFrame,
    radar_source_candidates: pd.DataFrame,
) -> pd.DataFrame:
    def count(field: str) -> int:
        return int(joined[field].notna().sum()) if field in joined else 0

    rows = [
        ("monthly_revenue_yoy_mom_rolling3m", "higher_quality_diagnostic", count("monthly_revenue_yoy"), "MOPS monthly revenue full-universe; conservative available_date; exact filing timestamp unavailable"),
        ("quarterly_revenue_eps_operating_income_growth", "higher_quality_diagnostic", count("quarterly_revenue_yoy"), "quarterly fundamentals full sweep; growth computed from past quarters only"),
        ("gross_operating_margin", "higher_quality_diagnostic", count("gross_margin"), "quarterly fundamentals as-of join"),
        ("roe_roa", "higher_quality_diagnostic_sparse", count("roe") + count("roa"), "quarterly fundamentals where profile supplies total assets/equity/ROE"),
        ("debt_to_equity_proxy", "proxy", count("debt_to_equity_proxy"), "balance-sheet proxy; not current ratio or full leverage policy"),
        ("paid_in_capital_issued_shares_proxy", "proxy", count("capital_stock"), "TWSE capital stock proxy; TPEx missing and no daily exact issued shares"),
        ("listing_status_partial_event", "partial_event_proxy", int(joined["listing_status_partial_available"].sum()), "partial accepted listing events only; not full listing master"),
        ("listing_board_proxy", "diagnostic_proxy", count("listing_board_proxy"), "derived from MOPS market/source table and partial events"),
        ("average_traded_value", "diagnostic_proxy", int(joined["average_traded_value_proxy_available"].sum()), "existing attention_features signal-date traded_value"),
        ("turnover", "diagnostic_proxy", int(joined["turnover_proxy_available"].sum()), "existing attention_features turnover windows"),
        ("industry_sector", "blocked_or_proxy", 0, "industry/sector remains blocked/proxy; TPEx all-stock historical route not accepted"),
        ("market_cap_free_float_market_cap", "blocked_or_proxy", 0, "full market cap and free-float market cap not materialized"),
        ("cash_flow_quality", "blocked", 0, "cash-flow quality contract not present"),
        ("current_ratio", "blocked", 0, "current ratio contract not present"),
        ("inventory_receivable_risk", "blocked", 0, "inventory/receivable risk contract not present"),
        ("forward_return_as_rule", "prohibited", 0, "forward returns are prohibited as Layer 1 rule inputs"),
    ]
    out = pd.DataFrame(
        [
            {
                "field_group": field,
                "source_tier": tier,
                "available_rows": available,
                "source_quality_reason": reason,
                "usable_for_layer1_refreshed_diagnostic": tier in {
                    "higher_quality_diagnostic",
                    "higher_quality_diagnostic_sparse",
                    "diagnostic_proxy",
                    "proxy",
                    "partial_event_proxy",
                },
                "usable_for_formal": False,
                "diagnostic_only": True,
            }
            for field, tier, available, reason in rows
        ]
    )
    out["radar_ready_for_core_layer1_contract_refresh"] = bool(
        radar_readiness.get("ready_for_core_layer1_contract_refresh", False)
    )
    out["radar_ready_for_experiments"] = bool(radar_readiness.get("ready_for_experiments", False))
    out["radar_inventory_rows"] = int(len(radar_inventory))
    out["radar_source_candidate_rows"] = int(len(radar_source_candidates))
    return out


def _blocked_proxy_fields(source_quality: pd.DataFrame) -> pd.DataFrame:
    out = source_quality.copy()
    out["status"] = out["source_tier"].map(
        lambda tier: "prohibited"
        if tier == "prohibited"
        else "blocked"
        if "blocked" in str(tier)
        else "proxy"
        if "proxy" in str(tier)
        else "partial"
        if "partial" in str(tier) or "sparse" in str(tier)
        else "diagnostic"
    )
    out["proxy_available"] = out["status"].isin(["proxy", "partial"])
    return out.rename(columns={"field_group": "field_or_contract", "source_quality_reason": "blocked_reason"})[
        ["field_or_contract", "status", "source_tier", "proxy_available", "blocked_reason", "diagnostic_only"]
    ]


def _future_data_audit(joined: pd.DataFrame, radar_readiness: dict[str, Any]) -> pd.DataFrame:
    checks = [
        ("monthly_available_date_lte_signal_date", "monthly_available_date"),
        ("quarterly_available_date_lte_signal_date", "quarterly_available_date"),
        ("capital_available_date_lte_signal_date", "capital_available_date"),
        ("listing_event_date_lte_signal_date", "listing_event_date"),
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
                "note": f"{col} joined by merge_asof backward",
            }
        )
    rows.append(
        {
            "audit_item": "radar_source_inventory_future_data_audit",
            "status": "passed" if int(radar_readiness.get("future_data_violation_count", 0)) == 0 else "failed",
            "future_data_violation_count": int(radar_readiness.get("future_data_violation_count", 0)),
            "note": "Radar/Data source inventory readiness audit imported",
        }
    )
    rows.append(
        {
            "audit_item": "forward_return_as_rule",
            "status": "passed",
            "future_data_violation_count": 0,
            "note": "no forward return columns are included in refreshed Layer 1 contracts",
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
    material_fields = {
        "monthly_revenue_yoy_mom_rolling3m",
        "quarterly_revenue_eps_operating_income_growth",
        "gross_operating_margin",
    }
    material_available = source_quality[
        source_quality["field_group"].isin(material_fields) & source_quality["available_rows"].gt(0)
    ]
    ready = (
        bool(radar_readiness.get("ready_for_core_layer1_contract_refresh", False))
        and not pit_contract.empty
        and len(material_available) == len(material_fields)
        and future_count == 0
    )
    blocked_any = bool(blocked["status"].isin(["blocked", "prohibited"]).any())
    layer1_coverage = "partial" if ready else "blocked"
    source_upgrade = "material" if ready else "minor" if not material_available.empty else "none"
    return {
        "date": "2026-07-07",
        "task_id": TASK_ID,
        "merged_task_ids": MERGED_TASK_IDS,
        "owner": "BACKTEST_LAB Core/Data",
        "status": "partial_ready_layer1_refreshed_pit_contract_proxy_limited" if ready else "blocked_layer1_refreshed_pit_contract",
        "ready_for_funnel_layer1_refreshed_event_diagnostic": bool(ready),
        "ready_for_funnel_layer1_candidate_pool_quality_diagnostic": bool(ready),
        "layer1_refreshed_exact_coverage": layer1_coverage,
        "layer1_exact_coverage": layer1_coverage,
        "layer1_source_upgrade_vs_previous": source_upgrade,
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
        "proxy_fields": blocked[blocked["status"].isin(["proxy", "partial"])]["field_or_contract"].tolist(),
        "diagnostic_fields": source_quality[
            source_quality["source_tier"].astype(str).str.contains("higher_quality|diagnostic", regex=True)
        ]["field_group"].tolist(),
        "known_blocker_summary": [
            "industry/sector remains blocked/proxy",
            "cash-flow quality, current ratio, inventory/receivable risk remain blocked",
            "complete market cap / free-float market cap remains blocked/proxy",
            "capital stock is TWSE proxy, not full daily exact market cap",
            "listing/status is partial event proxy, not full listing master",
        ],
        "radar_ready_for_experiments": bool(radar_readiness.get("ready_for_experiments", False)),
        "formal_model_changed": False,
        "trade_decision_changed": False,
        "active_in_trade_decision": False,
        "report_changed": False,
        "portfolio_replay_executed": False,
        "blocked_any": blocked_any,
    }


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def _summary(readiness: dict[str, Any], blocked: pd.DataFrame) -> str:
    return "\n".join(
        [
            "# vNext Funnel Layer 1 Refreshed PIT Contract Readiness",
            "",
            f"Status: {readiness['status']}",
            f"Merged task ids: {', '.join(MERGED_TASK_IDS)}",
            "",
            "Boundary: source/contract readiness only; no Experiments replay, no selector, no formal/report/trade change.",
            "",
            "Readiness:",
            f"- ready_for_funnel_layer1_refreshed_event_diagnostic={str(readiness['ready_for_funnel_layer1_refreshed_event_diagnostic']).lower()}",
            f"- ready_for_funnel_layer1_candidate_pool_quality_diagnostic={str(readiness['ready_for_funnel_layer1_candidate_pool_quality_diagnostic']).lower()}",
            f"- layer1_refreshed_exact_coverage={readiness['layer1_refreshed_exact_coverage']}",
            f"- layer1_source_upgrade_vs_previous={readiness['layer1_source_upgrade_vs_previous']}",
            "- ready_for_layer2_diagnostic=false",
            "- ready_for_portfolio_like_diagnostic=false",
            "- ready_for_strategy_replay=false",
            "- ready_for_formal=false",
            f"- future_data_violation_count={readiness['future_data_violation_count']}",
            "- not_live_rule=true",
            "- forward_returns_live_rule_usage=false",
            "",
            "Blocked / proxy fields:",
            *[f"- {row.field_or_contract}: {row.status}; {row.blocked_reason}" for row in blocked.itertuples()],
            "",
            "Next handoff:",
            "- vNext Research should judge whether this partial refreshed contract is enough for a bounded Layer 1 candidate-pool-quality diagnostic.",
            "- Strategy Center should keep Layer 2 disabled until Layer 1 diagnostic receives a Research GO.",
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
    parser.add_argument("--radar-inventory-dir", type=Path, default=DEFAULT_RADAR_INVENTORY_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    manifest = build_refreshed_layer1_readiness(
        materialization_dir=args.materialization_dir,
        radar_inventory_dir=args.radar_inventory_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
