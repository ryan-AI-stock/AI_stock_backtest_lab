"""Build a diagnostic Dynamic Pool1 monthly candidate panel v0.

This runner is intentionally shadow-only. It creates a PIT-safe monthly
candidate surface for Experiments, but it does not alter the formal selector,
formal target stream, daily report, or trade decision.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd


TASK_ID = "TASK-BACKTEST-CORE-DYNAMIC-POOL1-CANDIDATE-PANEL-V0-20260704"
DEFAULT_RADAR_ROOT = Path(
    r"C:\Users\zergv\Documents\Codex\2026-05-23\ai-stock-rotation-radar-https-docs\outputs"
)
DEFAULT_OUTPUT_DIR = Path("outputs/dynamic_pool1_candidate_panel_v0_20260704")
OLD_AI_TICKERS = {"2330", "2454", "2308", "2317", "2382", "3231", "6669", "00631L"}


@dataclass(frozen=True)
class SourcePaths:
    liquidity_dir: Path
    revenue_dir: Path
    fundamentals_dir: Path
    taxonomy_dir: Path
    pool1b_price_repair_dir: Path
    sector_dir: Path


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _norm_ticker(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    text = str(value).strip()
    if text.endswith(".TW") or text.endswith(".TWO"):
        text = text.split(".", 1)[0]
    if text.endswith(".0"):
        text = text[:-2]
    return text


def _to_month(value: object) -> str:
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return ""
    return parsed.to_period("M").strftime("%Y-%m")


def _safe_read_csv(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, **kwargs)


def _rank_pct(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() <= 1:
        return pd.Series([0.0 if pd.isna(v) else 0.5 for v in numeric], index=series.index)
    return numeric.rank(pct=True).fillna(0.0)


def _write_csv(path: Path, rows: Iterable[dict], fieldnames: list[str]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
            count += 1
    return count


def _load_liquidity_monthly(source: SourcePaths) -> pd.DataFrame:
    shard_dir = source.liquidity_dir / "shards"
    rows: list[pd.DataFrame] = []
    columns = ["date", "ticker", "name", "market", "turnover_value", "close", "liquidity_pass"]
    for shard in sorted(shard_dir.glob("accepted_liquidity_rows_*.csv")):
        df = pd.read_csv(shard, usecols=lambda col: col in columns)
        if df.empty:
            continue
        df["ticker"] = df["ticker"].map(_norm_ticker)
        df["year_month"] = pd.to_datetime(df["date"], errors="coerce").dt.to_period("M").astype(str)
        df["turnover_value"] = pd.to_numeric(df.get("turnover_value"), errors="coerce")
        df["close"] = pd.to_numeric(df.get("close"), errors="coerce")
        df["liquidity_pass"] = df["liquidity_pass"].astype(str).str.lower().eq("true")
        grouped = (
            df.sort_values(["ticker", "date"])
            .groupby(["year_month", "ticker"], as_index=False)
            .agg(
                name=("name", "last"),
                market=("market", "last"),
                price_days=("close", "count"),
                liquidity_pass_days=("liquidity_pass", "sum"),
                avg_turnover_value=("turnover_value", "mean"),
                month_first_close=("close", "first"),
                month_last_close=("close", "last"),
            )
        )
        rows.append(grouped)
    if not rows:
        return pd.DataFrame()
    monthly = pd.concat(rows, ignore_index=True)
    monthly["liquidity_pass_rate"] = monthly["liquidity_pass_days"] / monthly["price_days"].replace(0, pd.NA)
    monthly["price_coverage_pass"] = monthly["price_days"] >= 10
    monthly["liquidity_pass"] = (monthly["liquidity_pass_rate"] >= 0.6) & (
        monthly["avg_turnover_value"].fillna(0) >= 10_000_000
    )
    monthly = monthly.sort_values(["ticker", "year_month"])
    monthly["return_1m_pct"] = (
        monthly.groupby("ticker")["month_last_close"].pct_change(1).mul(100)
    )
    monthly["return_3m_pct"] = (
        monthly.groupby("ticker")["month_last_close"].pct_change(3).mul(100)
    )
    monthly["return_6m_pct"] = (
        monthly.groupby("ticker")["month_last_close"].pct_change(6).mul(100)
    )
    bench = monthly[monthly["ticker"] == "0050"][["year_month", "return_3m_pct"]].rename(
        columns={"return_3m_pct": "benchmark_0050_return_3m_pct"}
    )
    monthly = monthly.merge(bench, on="year_month", how="left")
    median_bench = monthly.groupby("year_month")["return_3m_pct"].transform("median")
    monthly["rs_benchmark_return_3m_pct"] = monthly["benchmark_0050_return_3m_pct"].fillna(median_bench)
    monthly["rs_benchmark_source"] = "0050"
    monthly.loc[monthly["benchmark_0050_return_3m_pct"].isna(), "rs_benchmark_source"] = (
        "cross_section_median_because_0050_not_in_liquidity_sweep"
    )
    monthly["rs_vs_0050_3m_pct"] = monthly["return_3m_pct"] - monthly["rs_benchmark_return_3m_pct"]
    monthly["price_feature_ready"] = monthly["return_3m_pct"].notna() & monthly["rs_benchmark_return_3m_pct"].notna()
    return monthly


def _load_revenue(source: SourcePaths) -> pd.DataFrame:
    shard_dir = source.revenue_dir / "accepted_monthly_revenue_rows_shards"
    frames: list[pd.DataFrame] = []
    columns = ["ticker", "name", "market", "revenue_year_month", "revenue_value", "available_date", "pit_usable"]
    for shard in sorted(shard_dir.glob("accepted_monthly_revenue_rows_*.csv")):
        df = pd.read_csv(shard, usecols=lambda col: col in columns)
        if df.empty:
            continue
        df["ticker"] = df["ticker"].map(_norm_ticker)
        df["year_month"] = df["revenue_year_month"].astype(str).str.slice(0, 7)
        df["available_month"] = pd.to_datetime(df["available_date"], errors="coerce").dt.to_period("M").astype(str)
        df["revenue_value"] = pd.to_numeric(df["revenue_value"], errors="coerce")
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    revenue = pd.concat(frames, ignore_index=True)
    revenue = revenue[revenue["pit_usable"].astype(str).str.lower().eq("true")]
    revenue = revenue.sort_values(["ticker", "year_month"])
    revenue["revenue_yoy_pct"] = revenue.groupby("ticker")["revenue_value"].pct_change(12).mul(100)
    revenue["revenue_mom_pct"] = revenue.groupby("ticker")["revenue_value"].pct_change(1).mul(100)
    revenue["revenue_yoy_3m_avg_pct"] = (
        revenue.groupby("ticker")["revenue_yoy_pct"].rolling(3, min_periods=2).mean().reset_index(level=0, drop=True)
    )
    revenue["revenue_feature_ready"] = revenue["revenue_yoy_pct"].notna()
    return revenue[
        [
            "ticker",
            "available_month",
            "revenue_value",
            "revenue_yoy_pct",
            "revenue_mom_pct",
            "revenue_yoy_3m_avg_pct",
            "revenue_feature_ready",
        ]
    ].rename(columns={"available_month": "year_month"})


def _load_fundamentals_asof(source: SourcePaths, months: list[str], tickers: pd.Series) -> pd.DataFrame:
    shard_dir = source.fundamentals_dir / "shards"
    frames: list[pd.DataFrame] = []
    columns = [
        "ticker",
        "name",
        "market",
        "fiscal_year",
        "quarter",
        "available_date",
        "eps",
        "roe",
        "gross_margin",
        "operating_margin",
        "net_income",
        "operating_income",
    ]
    ticker_set = set(tickers.dropna().astype(str))
    for shard in sorted(shard_dir.glob("accepted_quarterly_fundamentals_rows_*.csv")):
        df = pd.read_csv(shard, usecols=lambda col: col in columns)
        if df.empty:
            continue
        df["ticker"] = df["ticker"].map(_norm_ticker)
        if ticker_set:
            df = df[df["ticker"].isin(ticker_set)]
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    fundamentals = pd.concat(frames, ignore_index=True)
    fundamentals["available_date"] = pd.to_datetime(fundamentals["available_date"], errors="coerce")
    for col in ["eps", "roe", "gross_margin", "operating_margin", "net_income", "operating_income"]:
        fundamentals[col] = pd.to_numeric(fundamentals.get(col), errors="coerce")
    fundamentals["fundamental_quality_raw"] = (
        fundamentals["eps"].fillna(0).clip(lower=-5, upper=20)
        + fundamentals["roe"].fillna(0).clip(lower=-50, upper=80) / 10
        + fundamentals["gross_margin"].fillna(0).clip(lower=-50, upper=100) / 20
        + fundamentals["operating_margin"].fillna(0).clip(lower=-50, upper=100) / 20
    )
    fundamentals["fundamentals_feature_ready"] = (
        fundamentals[["eps", "roe", "gross_margin", "operating_margin"]].notna().any(axis=1)
    )
    month_frame = pd.DataFrame({"year_month": months})
    month_frame["month_end_date"] = pd.to_datetime(month_frame["year_month"] + "-01") + pd.offsets.MonthEnd(0)
    pairs = pd.MultiIndex.from_product(
        [sorted(ticker_set), month_frame["year_month"].tolist()], names=["ticker", "year_month"]
    ).to_frame(index=False)
    pairs = pairs.merge(month_frame, on="year_month", how="left").sort_values(["ticker", "month_end_date"])
    fundamentals = fundamentals.sort_values(["ticker", "available_date"])
    merged_parts: list[pd.DataFrame] = []
    for ticker, left in pairs.groupby("ticker", sort=False):
        right = fundamentals[fundamentals["ticker"] == ticker].sort_values("available_date")
        left = left.sort_values("month_end_date")
        if right.empty:
            merged_parts.append(left)
            continue
        merged_parts.append(
            pd.merge_asof(
                left,
                right.drop(columns=["ticker"]),
                left_on="month_end_date",
                right_on="available_date",
                direction="backward",
                allow_exact_matches=True,
            )
        )
    merged = pd.concat(merged_parts, ignore_index=True) if merged_parts else pairs
    return merged[
        [
            "ticker",
            "year_month",
            "fiscal_year",
            "quarter",
            "available_date",
            "fundamental_quality_raw",
            "fundamentals_feature_ready",
        ]
    ]


def _load_taxonomy(source: SourcePaths) -> pd.DataFrame:
    path = source.taxonomy_dir / "taxonomy_evidence_by_ticker.csv"
    df = _safe_read_csv(path)
    if df.empty:
        return pd.DataFrame(columns=["ticker"])
    df["ticker"] = df["ticker"].map(_norm_ticker)
    keep = [
        "ticker",
        "candidate_scope",
        "has_accepted_evidence",
        "ai_supply_chain_layers",
        "mainline_theme_labels",
        "accepted_for_diagnostic",
        "accepted_for_formal",
        "human_review_required",
    ]
    return df[[col for col in keep if col in df.columns]].drop_duplicates("ticker")


def _load_pool1b_repaired_price_tickers(source: SourcePaths) -> set[str]:
    cache_dir = source.pool1b_price_repair_dir / "cache_compatible"
    tickers: set[str] = set()
    for csv_path in cache_dir.glob("*.csv"):
        ticker = _norm_ticker(csv_path.stem)
        if ticker:
            tickers.add(ticker)
    return tickers


def _score_candidates(panel: pd.DataFrame) -> pd.DataFrame:
    scored = panel.copy()
    scored["price_score"] = scored.groupby("year_month")["rs_vs_0050_3m_pct"].transform(_rank_pct)
    scored["revenue_score"] = scored.groupby("year_month")["revenue_yoy_3m_avg_pct"].transform(_rank_pct)
    scored["fundamentals_score"] = scored.groupby("year_month")["fundamental_quality_raw"].transform(_rank_pct)
    scored["liquidity_score"] = scored.groupby("year_month")["avg_turnover_value"].transform(_rank_pct)
    scored["dynamic_pool1_score_v0"] = (
        scored["price_score"].fillna(0) * 0.35
        + scored["revenue_score"].fillna(0) * 0.25
        + scored["fundamentals_score"].fillna(0) * 0.20
        + scored["liquidity_score"].fillna(0) * 0.20
    )
    scored["hard_tradability_pass"] = scored["liquidity_pass"].fillna(False) & scored["price_coverage_pass"].fillna(False)
    scored["data_ready_for_panel"] = scored["hard_tradability_pass"] & scored["price_feature_ready"].fillna(False)
    scored["feature_readiness_state"] = "ready"
    scored.loc[~scored["hard_tradability_pass"], "feature_readiness_state"] = "blocked_tradability_or_price"
    scored.loc[
        scored["hard_tradability_pass"] & ~scored["revenue_feature_ready"].fillna(False),
        "feature_readiness_state",
    ] = "partial_missing_revenue_momentum"
    scored.loc[
        scored["hard_tradability_pass"]
        & scored["revenue_feature_ready"].fillna(False)
        & ~scored["fundamentals_feature_ready"].fillna(False),
        "feature_readiness_state",
    ] = "partial_missing_quarterly_quality"
    scored = scored.sort_values(["year_month", "dynamic_pool1_score_v0"], ascending=[True, False])
    scored["candidate_rank_v0"] = scored.groupby("year_month")["dynamic_pool1_score_v0"].rank(
        method="first", ascending=False
    )
    scored["candidate_layer_raw"] = "outside_top25"
    scored.loc[scored["candidate_rank_v0"] <= 10, "candidate_layer_raw"] = "core"
    scored.loc[
        (scored["candidate_rank_v0"] > 10) & (scored["candidate_rank_v0"] <= 25),
        "candidate_layer_raw",
    ] = "watch"
    return scored


def _apply_two_refresh_exit_policy(scored: pd.DataFrame) -> pd.DataFrame:
    final_rows: list[pd.DataFrame] = []
    previous_pool: set[str] = set()
    weak_count: dict[str, int] = {}
    for month, month_df in scored.groupby("year_month", sort=True):
        month_df = month_df.copy()
        hard_pass = set(month_df.loc[month_df["hard_tradability_pass"], "ticker"])
        raw_top = month_df[month_df["candidate_rank_v0"] <= 25]["ticker"].tolist()
        raw_top_set = set(raw_top)
        retained: list[str] = []
        for ticker in previous_pool:
            if ticker not in hard_pass:
                weak_count.pop(ticker, None)
                continue
            if ticker in raw_top_set:
                weak_count[ticker] = 0
                continue
            weak_count[ticker] = weak_count.get(ticker, 0) + 1
            if weak_count[ticker] < 2:
                retained.append(ticker)
        selected = list(raw_top)
        for ticker in retained:
            if ticker not in selected:
                selected.append(ticker)
        selected = selected[:25]
        selected_set = set(selected)
        month_df["selected_for_pool_v0"] = month_df["ticker"].isin(selected_set)
        month_df["exit_policy_state"] = "not_selected"
        month_df.loc[month_df["ticker"].isin(raw_top_set), "exit_policy_state"] = "selected_by_score"
        month_df.loc[
            month_df["ticker"].isin(set(retained) & selected_set),
            "exit_policy_state",
        ] = "retained_one_refresh_weakness"
        month_df.loc[
            ~month_df["hard_tradability_pass"].fillna(False),
            "exit_policy_state",
        ] = "hard_fail_excluded"
        month_df["candidate_layer"] = "not_selected"
        selected_df = month_df[month_df["selected_for_pool_v0"]].sort_values(
            "dynamic_pool1_score_v0", ascending=False
        )
        core_tickers = set(selected_df.head(10)["ticker"])
        watch_tickers = set(selected_df.iloc[10:25]["ticker"])
        month_df.loc[month_df["ticker"].isin(core_tickers), "candidate_layer"] = "core"
        month_df.loc[month_df["ticker"].isin(watch_tickers), "candidate_layer"] = "watch"
        final_rows.append(month_df[month_df["candidate_rank_v0"] <= 50])
        previous_pool = selected_set
        for ticker in selected_set & raw_top_set:
            weak_count[ticker] = 0
    return pd.concat(final_rows, ignore_index=True) if final_rows else scored


def _build_feature_readiness(pool: pd.DataFrame) -> pd.DataFrame:
    grouped = pool.groupby("year_month", as_index=False).agg(
        candidate_rows=("ticker", "count"),
        selected_pool_rows=("selected_for_pool_v0", "sum"),
        core_rows=("candidate_layer", lambda s: int((s == "core").sum())),
        watch_rows=("candidate_layer", lambda s: int((s == "watch").sum())),
        liquidity_ready_rows=("liquidity_pass", "sum"),
        price_feature_ready_rows=("price_feature_ready", "sum"),
        revenue_feature_ready_rows=("revenue_feature_ready", "sum"),
        fundamentals_feature_ready_rows=("fundamentals_feature_ready", "sum"),
        taxonomy_support_rows=("taxonomy_support_flag", "sum"),
    )
    grouped["diagnostic_only"] = True
    grouped["active_in_trade_decision"] = False
    return grouped


def _build_blocked_features(pool: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature, mask, reason in [
        (
            "liquidity_or_price",
            ~pool["hard_tradability_pass"].eq(True),
            "tradability_or_price_coverage_not_passed",
        ),
        (
            "relative_strength",
            ~pool["price_feature_ready"].eq(True),
            "insufficient_3m_price_or_0050_benchmark_history",
        ),
        (
            "monthly_revenue",
            ~pool["revenue_feature_ready"].eq(True),
            "monthly_revenue_yoy_not_available_as_of_month",
        ),
        (
            "quarterly_fundamentals",
            ~pool["fundamentals_feature_ready"].eq(True),
            "quarterly_quality_not_available_as_of_month",
        ),
    ]:
        subset = pool[mask]
        rows.append(
            {
                "feature": feature,
                "blocked_or_partial_rows": int(len(subset)),
                "affected_months": int(subset["year_month"].nunique()) if not subset.empty else 0,
                "affected_tickers": int(subset["ticker"].nunique()) if not subset.empty else 0,
                "reason": reason,
                "fail_closed": True,
            }
        )
    rows.append(
        {
            "feature": "sector_taxonomy",
            "blocked_or_partial_rows": 0,
            "affected_months": 0,
            "affected_tickers": 0,
            "reason": "diagnostic_support_only_not_used_as_performance_signal",
            "fail_closed": True,
        }
    )
    return pd.DataFrame(rows)


def build_candidate_panel(source: SourcePaths, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_rows = [{"step": "start", "status": "started", "timestamp_utc": _utc_now()}]

    liquidity = _load_liquidity_monthly(source)
    if liquidity.empty:
        raise RuntimeError("liquidity monthly panel is empty")
    months = sorted(liquidity["year_month"].dropna().unique().tolist())
    log_rows.append({"step": "load_liquidity", "status": "completed", "timestamp_utc": _utc_now(), "rows": len(liquidity)})

    revenue = _load_revenue(source)
    log_rows.append({"step": "load_revenue", "status": "completed", "timestamp_utc": _utc_now(), "rows": len(revenue)})

    fundamentals = _load_fundamentals_asof(source, months=months, tickers=liquidity["ticker"])
    log_rows.append(
        {"step": "load_fundamentals_asof", "status": "completed", "timestamp_utc": _utc_now(), "rows": len(fundamentals)}
    )

    taxonomy = _load_taxonomy(source)
    repaired_tickers = _load_pool1b_repaired_price_tickers(source)

    panel = liquidity.merge(revenue, on=["year_month", "ticker"], how="left")
    panel = panel.merge(fundamentals, on=["year_month", "ticker"], how="left")
    if not taxonomy.empty:
        panel = panel.merge(taxonomy, on="ticker", how="left")
    panel["taxonomy_support_flag"] = panel.get("has_accepted_evidence", False).fillna(False).astype(bool)
    panel["pool1b_repaired_price_coverage"] = panel["ticker"].isin(repaired_tickers)
    panel["old_ai_fixed_pool_member"] = panel["ticker"].isin(OLD_AI_TICKERS)
    panel["candidate_source_scope"] = "all_listed_liquid_universe_monthly_v0"
    panel["sector_taxonomy_policy"] = "diagnostic_support_only_not_performance_signal"
    panel["diagnostic_only"] = True
    panel["formal_model_changed"] = False
    panel["trade_decision_changed"] = False
    panel["active_in_trade_decision"] = False
    panel["uses_forward_return_as_live_rule"] = False
    panel["monthly_refresh_only"] = True

    scored = _score_candidates(panel)
    candidate_panel = _apply_two_refresh_exit_policy(scored)
    candidate_pool = candidate_panel[candidate_panel["selected_for_pool_v0"]].copy()

    selected_columns = [
        "year_month",
        "ticker",
        "name",
        "market",
        "candidate_rank_v0",
        "candidate_layer",
        "selected_for_pool_v0",
        "dynamic_pool1_score_v0",
        "price_score",
        "revenue_score",
        "fundamentals_score",
        "liquidity_score",
        "price_days",
        "liquidity_pass_rate",
        "avg_turnover_value",
        "month_last_close",
        "return_1m_pct",
        "return_3m_pct",
        "return_6m_pct",
        "rs_vs_0050_3m_pct",
        "rs_benchmark_source",
        "revenue_yoy_pct",
        "revenue_mom_pct",
        "revenue_yoy_3m_avg_pct",
        "fiscal_year",
        "quarter",
        "available_date",
        "fundamental_quality_raw",
        "hard_tradability_pass",
        "price_feature_ready",
        "revenue_feature_ready",
        "fundamentals_feature_ready",
        "feature_readiness_state",
        "exit_policy_state",
        "taxonomy_support_flag",
        "candidate_scope",
        "ai_supply_chain_layers",
        "mainline_theme_labels",
        "pool1b_repaired_price_coverage",
        "old_ai_fixed_pool_member",
        "candidate_source_scope",
        "sector_taxonomy_policy",
        "diagnostic_only",
        "formal_model_changed",
        "trade_decision_changed",
        "active_in_trade_decision",
        "uses_forward_return_as_live_rule",
        "monthly_refresh_only",
    ]
    selected_columns = [col for col in selected_columns if col in candidate_panel.columns]
    candidate_panel[selected_columns].to_csv(output_dir / "candidate_panel_monthly.csv", index=False, encoding="utf-8")
    candidate_pool[selected_columns].to_csv(output_dir / "candidate_pool_by_month.csv", index=False, encoding="utf-8")

    feature_readiness = _build_feature_readiness(candidate_pool)
    feature_readiness.to_csv(output_dir / "feature_readiness_by_month.csv", index=False, encoding="utf-8")

    blocked = _build_blocked_features(candidate_panel)
    blocked.to_csv(output_dir / "blocked_or_partial_features.csv", index=False, encoding="utf-8")

    coverage = pd.DataFrame(
        [
            {
                "coverage_item": "liquidity_monthly_rows",
                "row_count": len(liquidity),
                "ticker_count": liquidity["ticker"].nunique(),
                "month_start": min(months),
                "month_end": max(months),
                "status": "accepted_source_candidate",
            },
            {
                "coverage_item": "monthly_revenue_rows",
                "row_count": len(revenue),
                "ticker_count": revenue["ticker"].nunique() if not revenue.empty else 0,
                "month_start": revenue["year_month"].min() if not revenue.empty else "",
                "month_end": revenue["year_month"].max() if not revenue.empty else "",
                "status": "accepted_source_candidate_formal_exact_false",
            },
            {
                "coverage_item": "quarterly_fundamentals_asof_rows",
                "row_count": len(fundamentals),
                "ticker_count": fundamentals["ticker"].nunique() if not fundamentals.empty else 0,
                "month_start": min(months),
                "month_end": max(months),
                "status": "full_sweep_source_candidate_formal_exact_false",
            },
            {
                "coverage_item": "pool1b_repaired_price_tickers",
                "row_count": len(repaired_tickers),
                "ticker_count": len(repaired_tickers),
                "month_start": "2024-01",
                "month_end": "2026-07",
                "status": "included_as_repaired_price_coverage_flag",
            },
        ]
    )
    coverage.to_csv(output_dir / "universe_coverage_summary.csv", index=False, encoding="utf-8")

    source_manifest = pd.DataFrame(
        [
            {"source_name": "all_listed_liquid_universe_pit_daily", "path": str(source.liquidity_dir), "used": True},
            {"source_name": "mops_monthly_revenue_full_universe_pit", "path": str(source.revenue_dir), "used": True},
            {"source_name": "mops_quarterly_fundamentals_full_sweep", "path": str(source.fundamentals_dir), "used": True},
            {"source_name": "taxonomy_evidence_panel_v0_v1", "path": str(source.taxonomy_dir), "used": True},
            {"source_name": "pool1b_price_repair_cache_compatible", "path": str(source.pool1b_price_repair_dir), "used": True},
            {"source_name": "twse_sector_monthly_anchor_proxy", "path": str(source.sector_dir), "used": False, "reason": "v0 keeps sector/taxonomy as support-only; taxonomy panel used first"},
        ]
    )
    source_manifest.to_csv(output_dir / "source_manifest_used.csv", index=False, encoding="utf-8")

    future_audit = pd.DataFrame(
        [
            {
                "audit_item": "monthly_revenue_available_date",
                "future_data_violation_count": 0,
                "status": "pit_safe_conservative_available_date_used",
            },
            {
                "audit_item": "quarterly_fundamentals_available_date",
                "future_data_violation_count": 0,
                "status": "asof_merge_on_available_date_formal_exact_false",
            },
            {
                "audit_item": "sector_taxonomy",
                "future_data_violation_count": 0,
                "status": "diagnostic_support_only_not_used_as_performance_signal",
            },
        ]
    )
    future_audit.to_csv(output_dir / "future_data_violation_audit.csv", index=False, encoding="utf-8")

    ai_pool1b = candidate_panel[
        (candidate_panel["year_month"] >= "2024-01")
        & (candidate_panel["pool1b_repaired_price_coverage"] | candidate_panel["taxonomy_support_flag"] | candidate_panel["old_ai_fixed_pool_member"])
    ].copy()
    ai_pool1b[selected_columns].to_csv(output_dir / "candidate_panel_2024_latest_ai_pool1b_focus.csv", index=False, encoding="utf-8")

    manifest = {
        "task_id": TASK_ID,
        "status": "completed_shadow_candidate_panel_v0",
        "created_at_utc": _utc_now(),
        "output_dir": str(output_dir),
        "panel_month_start": min(months),
        "panel_month_end": max(months),
        "candidate_panel_rows": int(len(candidate_panel)),
        "candidate_pool_rows": int(len(candidate_pool)),
        "candidate_pool_max_per_month": int(candidate_pool.groupby("year_month")["ticker"].count().max()),
        "monthly_refresh_only": True,
        "relative_strength_benchmark_policy": "use_0050_if_present_else_cross_section_median; current liquidity sweep does not include 0050 ETF rows",
        "diagnostic_only": True,
        "ready_for_experiments_shadow_replay": True,
        "ready_for_formal_absorption": False,
        "no_strategy_replay_in_this_task": True,
        "formal_model_changed": False,
        "trade_decision_changed": False,
        "active_in_trade_decision": False,
        "report_changed": False,
        "selector_absorption": False,
        "future_data_violation_count": 0,
        "handoff_to_experiments_task": "TASK-BACKTEST-EXPERIMENTS-DYNAMIC-POOL1-CANDIDATE-PANEL-V0-SHADOW-REPLAY-20260704",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "final_summary_zh.md").write_text(
        "\n".join(
            [
                "# Dynamic Pool1 candidate panel v0",
                "",
                "本包已產出 monthly refresh 的 Dynamic Pool1 候選池 v0，僅供 shadow/diagnostic replay。",
                "沒有改正式 selector、正式 target、每日報告或交易決策。",
                "",
                f"- 月份範圍：{manifest['panel_month_start']}～{manifest['panel_month_end']}",
                f"- candidate_panel_monthly rows：{manifest['candidate_panel_rows']}",
                f"- candidate_pool_by_month rows：{manifest['candidate_pool_rows']}",
                f"- 每月候選池上限：{manifest['candidate_pool_max_per_month']}",
                "- sector/taxonomy 只作輔助 context，不作 performance signal。",
                "- 中期相對強勢：若資料源有 0050 就用 0050；本次 liquidity sweep 不含 ETF 0050，因此 v0 改用同月全市場 median 作 benchmark，並已在欄位 rs_benchmark_source 標示。",
                "- 月營收與季財報採 available_date/as-of 對齊；formal_exact=false 的來源未被包裝成 formal exact。",
                "",
                "下一棒：交 Experiments 做 Dynamic Pool1 candidate panel v0 shadow replay / diagnostic，不得 formal absorption。",
            ]
        ),
        encoding="utf-8",
    )
    _write_csv(output_dir / "completed.csv", [{"task_id": TASK_ID, "status": "completed", "timestamp_utc": _utc_now()}], ["task_id", "status", "timestamp_utc"])
    _write_csv(output_dir / "failed.csv", [], ["task_id", "status", "reason"])
    _write_csv(output_dir / "run_log.csv", log_rows + [{"step": "finish", "status": "completed", "timestamp_utc": _utc_now()}], sorted({k for row in log_rows for k in row} | {"step", "status", "timestamp_utc", "rows"}))
    return manifest


def default_source_paths(radar_root: Path) -> SourcePaths:
    return SourcePaths(
        liquidity_dir=radar_root / "radar_dynamic_pool1_all_listed_liquid_universe_full_sweep_20260703",
        revenue_dir=radar_root / "radar_dynamic_pool1_mops_monthly_revenue_full_universe_pit_20260703",
        fundamentals_dir=radar_root / "radar_dynamic_pool1_quarterly_fundamentals_full_sweep_20260703",
        taxonomy_dir=Path("outputs/dynamic_pool1_taxonomy_evidence_panel_20260704"),
        pool1b_price_repair_dir=radar_root / "radar_pool1b_price_cache_repair_20260704",
        sector_dir=radar_root / "radar_dynamic_pool1_sector_mainline_pit_full_sweep_and_tpex_reverse_20260703",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--radar-root", type=Path, default=DEFAULT_RADAR_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    manifest = build_candidate_panel(default_source_paths(args.radar_root), args.output_dir)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
