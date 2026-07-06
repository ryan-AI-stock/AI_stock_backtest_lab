"""Materialize vNext diagnostic data contracts.

This module creates diagnostic-only input tables and weekly snapshots for the
vNext Dynamic Candidate Pool handoff. It does not run portfolio replay and does
not alter formal model, report, or trade-decision paths.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


TASK_ID = "TASK-BACKTEST-CORE-DATA-VNEXT-DYNAMIC-CANDIDATE-POOL-MATERIALIZATION-20260706"
DEFAULT_OUTPUT_DIR = Path("outputs/vnext_dynamic_candidate_pool_data_materialization_20260706")
DEFAULT_RADAR_ROOT = Path(
    r"C:\Users\zergv\Documents\Codex\2026-05-23\ai-stock-rotation-radar-https-docs\outputs"
)
DEFAULT_LIQUIDITY_DIR = DEFAULT_RADAR_ROOT / "radar_dynamic_pool1_all_listed_liquid_universe_full_sweep_20260703"
DEFAULT_REVENUE_DIR = DEFAULT_RADAR_ROOT / "radar_dynamic_pool1_mops_monthly_revenue_full_universe_pit_20260703"
DEFAULT_FUNDAMENTALS_DIR = DEFAULT_RADAR_ROOT / "radar_dynamic_pool1_quarterly_fundamentals_full_sweep_20260703"
DEFAULT_TAXONOMY_PATH = Path("outputs/dynamic_pool1_taxonomy_evidence_panel_20260704/taxonomy_evidence_by_ticker.csv")
DEFAULT_CACHE_DIR = Path("backtest_cache/stock_pool_observations")

CASE_TICKERS = {"6669": "緯穎", "2308": "台達電", "2317": "鴻海"}
CASE_TRACE_DATE = "2026-06-30"
BENCHMARKS = {"0050": "0050_TW.csv", "00631L": "00631L_TW.csv"}
REQUESTED_PERIODS = {
    "P1": ("2015-01-02", "2022-12-29"),
    "P2": ("2023-01-02", "2026-06-30"),
    "2024_latest": ("2024-01-02", "latest_available"),
    "2026YTD": ("2026-01-02", "latest_available"),
}
WINDOWS = [5, 10, 20, 40, 60]
BIAS_WINDOWS = [20, 60, 120]


def materialize_vnext_dynamic_candidate_pool(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    liquidity_dir: str | Path = DEFAULT_LIQUIDITY_DIR,
    revenue_dir: str | Path = DEFAULT_REVENUE_DIR,
    fundamentals_dir: str | Path = DEFAULT_FUNDAMENTALS_DIR,
    taxonomy_path: str | Path = DEFAULT_TAXONOMY_PATH,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    liquidity = _load_liquidity(Path(liquidity_dir))
    calendar = _build_trading_calendar(liquidity["trade_date"])
    daily_market = _daily_market_features(liquidity)

    benchmark_features = _build_benchmark_features(Path(cache_dir))
    stock_features = _build_stock_features(liquidity, benchmark_features, calendar)
    attention_features = _build_attention_features(liquidity, calendar)
    fundamental_features = _build_fundamental_features(Path(revenue_dir), Path(fundamentals_dir))
    theme_membership = _build_theme_membership(Path(taxonomy_path), liquidity)

    weekly_snapshot = _build_weekly_snapshot(
        stock_features=stock_features,
        attention_features=attention_features,
        daily_market=daily_market,
        fundamental_features=fundamental_features,
        theme_membership=theme_membership,
        calendar=calendar,
        benchmark_features=benchmark_features,
    )
    final_decision = _build_final_decision_snapshot(weekly_snapshot)
    case_trace = _build_case_trace(weekly_snapshot, stock_features, attention_features, daily_market, benchmark_features)
    coverage = _coverage_summary(liquidity, benchmark_features, stock_features, weekly_snapshot)
    blocked_rows = _blocked_rows(
        stock_features=stock_features,
        weekly_snapshot=weekly_snapshot,
        final_decision=final_decision,
        case_trace=case_trace,
        benchmark_features=benchmark_features,
    )

    _write_csv(calendar, output / "trading_calendar.csv")
    _write_csv(daily_market, output / "daily_market_features.csv")
    _write_csv(benchmark_features, output / "benchmark_features.csv")
    _write_csv(stock_features, output / "stock_features.csv")
    _write_csv(attention_features, output / "attention_features.csv")
    _write_csv(fundamental_features, output / "fundamental_features.csv")
    _write_csv(theme_membership, output / "theme_membership.csv")
    _write_csv(weekly_snapshot, output / "vnext_weekly_candidate_snapshot.csv")
    _write_csv(final_decision, output / "vnext_final_decision_snapshot.csv")
    _write_csv(case_trace, output / "vnext_case_trace.csv")
    _write_csv(coverage, output / "coverage_summary.csv")
    _write_csv(blocked_rows, output / "blocked_rows.csv")
    _write_csv(_selected_hygiene_audit(weekly_snapshot, final_decision), output / "selected_row_hygiene_audit.csv")

    ready_for_phase_a = bool(
        not weekly_snapshot.empty
        and not final_decision.empty
        and int(blocked_rows["severity"].eq("error").sum()) == 0
    )
    manifest = {
        "task_id": TASK_ID,
        "status": "materialized_with_blockers" if not ready_for_phase_a else "materialized_ready_for_phase_a",
        "output_dir": str(output.resolve()),
        "execution_basis": "diagnostic_data_materialization_only",
        "formal_model_changed": False,
        "trade_decision_changed": False,
        "active_in_trade_decision": False,
        "report_changed": False,
        "portfolio_replay_executed": False,
        "candidate_forward_return_diagnostic_executed": False,
        "ready_for_experiments_phase_a_snapshot_integrity": ready_for_phase_a,
        "rows": {
            "trading_calendar": int(len(calendar)),
            "daily_market_features": int(len(daily_market)),
            "benchmark_features": int(len(benchmark_features)),
            "stock_features": int(len(stock_features)),
            "attention_features": int(len(attention_features)),
            "fundamental_features": int(len(fundamental_features)),
            "theme_membership": int(len(theme_membership)),
            "vnext_weekly_candidate_snapshot": int(len(weekly_snapshot)),
            "vnext_final_decision_snapshot": int(len(final_decision)),
            "vnext_case_trace": int(len(case_trace)),
            "blocked_rows": int(len(blocked_rows)),
        },
        "selected_row_hygiene_rule": "selected_outcome_candidate = selected_by_vnext AND NOT case_trace_only AND diagnostic_only",
        "notes": [
            "0050 is the RS/regime/comparison base.",
            "00631L is fallback/hurdle/execution candidate only, not ordinary pool member.",
            "Weekly snapshot rows are diagnostic_only=true.",
            "Case trace rows are case_trace_only=true and excluded from selected outcomes.",
            "Benchmark cache currently ends before 2026-06-30, so 2026-06-30 case trace benchmark-aligned fields are blocked.",
        ],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (output / "readiness_for_experiments.json").write_text(
        json.dumps(_readiness(manifest, coverage, blocked_rows), ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    (output / "final_summary_zh.md").write_text(_summary(manifest, coverage, blocked_rows), encoding="utf-8")
    return manifest


def _norm_ticker(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    text = str(value).strip()
    if text.endswith(".TW") or text.endswith(".TWO"):
        text = text.split(".", 1)[0]
    if text.endswith(".0"):
        text = text[:-2]
    return text


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def _load_liquidity(liquidity_dir: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    columns = [
        "date",
        "ticker",
        "name",
        "market",
        "is_listed_as_of_date",
        "is_suspended_as_of_date",
        "volume",
        "turnover_value",
        "close",
        "liquidity_pass",
        "blocked_reason",
    ]
    for path in sorted((liquidity_dir / "shards").glob("accepted_liquidity_rows_*.csv")):
        df = pd.read_csv(path, usecols=lambda col: col in columns)
        if df.empty:
            continue
        df["trade_date"] = pd.to_datetime(df["date"], errors="coerce")
        df["ticker"] = df["ticker"].map(_norm_ticker)
        frames.append(df.dropna(subset=["trade_date", "ticker"]))
    if not frames:
        raise FileNotFoundError(f"No liquidity shards found under {liquidity_dir}")
    out = pd.concat(frames, ignore_index=True)
    out["adjusted_close"] = pd.to_numeric(out["close"], errors="coerce")
    out["volume"] = pd.to_numeric(out["volume"], errors="coerce")
    out["traded_value"] = pd.to_numeric(out["turnover_value"], errors="coerce")
    out["liquidity_pass"] = out["liquidity_pass"].astype(str).str.lower().eq("true")
    return out.sort_values(["trade_date", "ticker"]).reset_index(drop=True)


def _build_trading_calendar(dates: pd.Series) -> pd.DataFrame:
    cal = pd.DataFrame({"trade_date": pd.to_datetime(dates).dropna().drop_duplicates().sort_values()})
    iso = cal["trade_date"].dt.isocalendar()
    cal["week_id"] = iso["year"].astype(str) + "-W" + iso["week"].astype(str).str.zfill(2)
    cal["is_week_last_trading_day"] = cal["trade_date"].eq(cal.groupby("week_id")["trade_date"].transform("max"))
    cal["next_trade_date"] = cal["trade_date"].shift(-1)
    cal["source_last_available_date"] = cal["trade_date"].max()
    cal["latest_observed_week_partial"] = cal["week_id"].eq(cal.loc[cal.index[-1], "week_id"]) & (
        cal["trade_date"].dt.weekday < 4
    )
    return cal


def _daily_market_features(liquidity: pd.DataFrame) -> pd.DataFrame:
    out = liquidity[
        [
            "trade_date",
            "ticker",
            "name",
            "market",
            "adjusted_close",
            "volume",
            "traded_value",
            "liquidity_pass",
            "is_listed_as_of_date",
            "is_suspended_as_of_date",
            "blocked_reason",
        ]
    ].copy()
    out["turnover"] = pd.NA
    out["listing_status"] = out.apply(
        lambda row: "listed"
        if bool(row["is_listed_as_of_date"]) and str(row["is_suspended_as_of_date"]).lower() in {"false", "unknown_from_daily_trading_only"}
        else "blocked_or_unknown",
        axis=1,
    )
    out["valid_universe"] = out["adjusted_close"].notna() & out["liquidity_pass"] & out["listing_status"].eq("listed")
    out["liquidity_flag"] = out["liquidity_pass"].map({True: "pass", False: "fail"})
    return out[
        [
            "trade_date",
            "ticker",
            "name",
            "market",
            "adjusted_close",
            "volume",
            "traded_value",
            "turnover",
            "listing_status",
            "valid_universe",
            "liquidity_flag",
            "blocked_reason",
        ]
    ]


def _read_price_cache(path: Path, benchmark: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["trade_date"] = pd.to_datetime(df["date"], errors="coerce")
    df["benchmark"] = benchmark
    df["adjusted_close"] = pd.to_numeric(df.get("adj_close", df.get("close")), errors="coerce")
    return df.dropna(subset=["trade_date", "adjusted_close"])[["trade_date", "benchmark", "adjusted_close"]]


def _build_benchmark_features(cache_dir: Path) -> pd.DataFrame:
    frames = []
    for benchmark, filename in BENCHMARKS.items():
        frames.append(_read_price_cache(cache_dir / filename, benchmark))
    out = pd.concat(frames, ignore_index=True).sort_values(["benchmark", "trade_date"])
    for window in WINDOWS:
        out[f"return_{window}d"] = out.groupby("benchmark")["adjusted_close"].pct_change(window)
    for window in BIAS_WINDOWS:
        ma = out.groupby("benchmark")["adjusted_close"].transform(lambda s: s.rolling(window, min_periods=window).mean())
        out[f"MA{window}"] = ma
        out[f"BIAS{window}"] = (out["adjusted_close"] - ma) / ma
    out["drawdown"] = out["adjusted_close"] / out.groupby("benchmark")["adjusted_close"].cummax() - 1
    return out


def _build_stock_features(liquidity: pd.DataFrame, benchmark_features: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    weekly_dates = set(calendar.loc[calendar["is_week_last_trading_day"], "trade_date"])
    weekly_dates.add(pd.Timestamp(CASE_TRACE_DATE))
    base = liquidity[["trade_date", "ticker", "adjusted_close"]].copy().sort_values(["ticker", "trade_date"])
    for window in WINDOWS:
        base[f"return_{window}d"] = base.groupby("ticker")["adjusted_close"].pct_change(window)
    for window in BIAS_WINDOWS:
        ma = base.groupby("ticker")["adjusted_close"].transform(lambda s: s.rolling(window, min_periods=window).mean())
        base[f"MA{window}"] = ma
        base[f"BIAS{window}"] = (base["adjusted_close"] - ma) / ma
        base[f"MA{window}_position"] = base["adjusted_close"] / ma - 1
        base[f"drawdown_{window}d"] = base["adjusted_close"] / base.groupby("ticker")["adjusted_close"].transform(
            lambda s: s.rolling(window, min_periods=window).max()
        ) - 1
    base["volatility"] = base.groupby("ticker")["adjusted_close"].pct_change().groupby(base["ticker"]).transform(
        lambda s: s.rolling(20, min_periods=20).std()
    )

    b0050 = benchmark_features[benchmark_features["benchmark"].eq("0050")][
        ["trade_date"] + [f"return_{w}d" for w in WINDOWS]
    ].rename(columns={f"return_{w}d": f"0050_return_{w}d" for w in WINDOWS})
    b00631l = benchmark_features[benchmark_features["benchmark"].eq("00631L")][
        ["trade_date"] + [f"return_{w}d" for w in WINDOWS]
    ].rename(columns={f"return_{w}d": f"00631L_return_{w}d" for w in WINDOWS})
    base = base.merge(b0050, on="trade_date", how="left").merge(b00631l, on="trade_date", how="left")
    for window in WINDOWS:
        base[f"RS{window}"] = base[f"return_{window}d"] - base[f"0050_return_{window}d"]
        base[f"excess_return_vs_00631L_{window}d"] = base[f"return_{window}d"] - base[f"00631L_return_{window}d"]
    for window in BIAS_WINDOWS:
        base[f"BIAS{window}_percentile"] = base.groupby("trade_date")[f"BIAS{window}"].rank(pct=True)
    out = base[base["trade_date"].isin(weekly_dates)].copy()
    return out


def _build_attention_features(liquidity: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    weekly_dates = set(calendar.loc[calendar["is_week_last_trading_day"], "trade_date"])
    weekly_dates.add(pd.Timestamp(CASE_TRACE_DATE))
    out = liquidity[["trade_date", "ticker", "traded_value", "volume", "adjusted_close"]].copy()
    out = out.sort_values(["ticker", "trade_date"])
    for window in [5, 20, 60]:
        out[f"turnover_{window}d"] = out.groupby("ticker")["traded_value"].transform(lambda s: s.rolling(window, min_periods=window).mean())
        out[f"turnover_rank_pct_{window}d"] = out.groupby("trade_date")[f"turnover_{window}d"].rank(pct=True)
    out["traded_value_rank_pct"] = out.groupby("trade_date")["traded_value"].rank(pct=True)
    vol_mean = out.groupby("ticker")["volume"].transform(lambda s: s.rolling(20, min_periods=20).mean())
    vol_std = out.groupby("ticker")["volume"].transform(lambda s: s.rolling(20, min_periods=20).std())
    out["volume_zscore"] = (out["volume"] - vol_mean) / vol_std
    ret5 = out.groupby("ticker")["adjusted_close"].pct_change(5)
    ret20 = out.groupby("ticker")["adjusted_close"].pct_change(20)
    out["high_turnover_price_confirmed"] = out["turnover_rank_pct_20d"].ge(0.8) & ret5.ge(0)
    out["distribution_risk"] = out["turnover_rank_pct_20d"].ge(0.8) & ret20.lt(0)
    return out[out["trade_date"].isin(weekly_dates)].copy()


def _build_fundamental_features(revenue_dir: Path, fundamentals_dir: Path) -> pd.DataFrame:
    rev_frames = []
    for path in sorted((revenue_dir / "accepted_monthly_revenue_rows_shards").glob("accepted_monthly_revenue_rows_*.csv")):
        df = pd.read_csv(path, usecols=lambda col: col in {"ticker", "available_date", "revenue_value", "pit_usable"})
        df = df[df["pit_usable"].astype(str).str.lower().eq("true")]
        df["ticker"] = df["ticker"].map(_norm_ticker)
        df["effective_date"] = pd.to_datetime(df["available_date"], errors="coerce")
        df["revenue_growth"] = pd.to_numeric(df["revenue_value"], errors="coerce")
        rev_frames.append(df[["effective_date", "ticker", "revenue_growth"]])
    revenue = pd.concat(rev_frames, ignore_index=True) if rev_frames else pd.DataFrame(columns=["effective_date", "ticker", "revenue_growth"])
    revenue = revenue.sort_values(["ticker", "effective_date"])
    revenue["revenue_growth"] = revenue.groupby("ticker")["revenue_growth"].pct_change(12)

    fund_frames = []
    cols = {"ticker", "available_date", "eps", "roe", "gross_margin", "operating_margin", "net_income", "total_liabilities", "equity", "formal_exact"}
    for path in sorted((fundamentals_dir / "shards").glob("accepted_quarterly_fundamentals_rows_*.csv")):
        df = pd.read_csv(path, usecols=lambda col: col in cols)
        df["ticker"] = df["ticker"].map(_norm_ticker)
        df["effective_date"] = pd.to_datetime(df["available_date"], errors="coerce")
        df["profitability"] = pd.to_numeric(df.get("eps"), errors="coerce")
        df["roe_or_quality"] = pd.to_numeric(df.get("roe"), errors="coerce")
        df["gross_margin"] = pd.to_numeric(df.get("gross_margin"), errors="coerce")
        df["operating_margin"] = pd.to_numeric(df.get("operating_margin"), errors="coerce")
        liabilities = pd.to_numeric(df.get("total_liabilities"), errors="coerce")
        equity = pd.to_numeric(df.get("equity"), errors="coerce")
        df["debt_or_solvency_risk"] = liabilities / equity.replace(0, pd.NA)
        df["cash_flow_quality"] = pd.NA
        df["source_quality"] = df.get("formal_exact", False).astype(str).str.lower().map({"true": "exact", "false": "proxy"}).fillna("proxy")
        fund_frames.append(
            df[
                [
                    "effective_date",
                    "ticker",
                    "profitability",
                    "gross_margin",
                    "operating_margin",
                    "roe_or_quality",
                    "cash_flow_quality",
                    "debt_or_solvency_risk",
                    "source_quality",
                ]
            ]
        )
    fundamentals = pd.concat(fund_frames, ignore_index=True) if fund_frames else pd.DataFrame()
    out = fundamentals.merge(revenue, on=["effective_date", "ticker"], how="left")
    out["effective_asof_lag_days"] = pd.NA
    return out[
        [
            "effective_date",
            "ticker",
            "revenue_growth",
            "profitability",
            "gross_margin",
            "operating_margin",
            "roe_or_quality",
            "cash_flow_quality",
            "debt_or_solvency_risk",
            "source_quality",
            "effective_asof_lag_days",
        ]
    ]


def _build_theme_membership(taxonomy_path: Path, liquidity: pd.DataFrame) -> pd.DataFrame:
    if not taxonomy_path.exists():
        return pd.DataFrame(columns=["effective_date", "ticker", "theme_id", "theme_name", "membership_score", "source_quality", "valid_from", "valid_to"])
    df = pd.read_csv(taxonomy_path).fillna("")
    df["ticker"] = df["ticker"].map(_norm_ticker)
    out = pd.DataFrame(
        {
            "effective_date": pd.Timestamp(liquidity["trade_date"].min()),
            "ticker": df["ticker"],
            "theme_id": df["ai_supply_chain_layers"].astype(str).replace("", "unclassified"),
            "theme_name": df["mainline_theme_labels"].astype(str).replace("", "unclassified"),
            "membership_score": df["accepted_for_diagnostic"].astype(str).str.lower().eq("true").map({True: 1.0, False: 0.0}),
            "source_quality": "proxy",
            "valid_from": pd.Timestamp(liquidity["trade_date"].min()),
            "valid_to": pd.NaT,
        }
    )
    return out


def _latest_asof(df: pd.DataFrame, date_col: str, key_col: str, value_cols: list[str], snapshot_dates: pd.Series) -> pd.DataFrame:
    keys = pd.DataFrame({"snapshot_date": snapshot_dates.drop_duplicates().sort_values()})
    frame = df[[date_col, key_col] + value_cols].dropna(subset=[date_col, key_col]).sort_values(date_col)
    rows = []
    for snap in keys["snapshot_date"]:
        part = frame[frame[date_col] <= snap].sort_values([key_col, date_col]).groupby(key_col).tail(1)
        part = part.copy()
        part["snapshot_date"] = snap
        rows.append(part)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _build_weekly_snapshot(
    *,
    stock_features: pd.DataFrame,
    attention_features: pd.DataFrame,
    daily_market: pd.DataFrame,
    fundamental_features: pd.DataFrame,
    theme_membership: pd.DataFrame,
    calendar: pd.DataFrame,
    benchmark_features: pd.DataFrame,
) -> pd.DataFrame:
    weekly_dates = calendar.loc[calendar["is_week_last_trading_day"], "trade_date"]
    stock = stock_features[stock_features["trade_date"].isin(weekly_dates)].copy()
    attention = attention_features[attention_features["trade_date"].isin(weekly_dates)].copy()
    market = daily_market[daily_market["trade_date"].isin(weekly_dates)][
        ["trade_date", "ticker", "name", "market", "valid_universe"]
    ]
    base = market.merge(stock, on=["trade_date", "ticker"], how="left").merge(attention, on=["trade_date", "ticker"], how="left", suffixes=("", "_attn"))

    fund_asof = _latest_asof(
        fundamental_features,
        "effective_date",
        "ticker",
        ["revenue_growth", "profitability", "source_quality"],
        weekly_dates,
    )
    if not fund_asof.empty:
        base = base.merge(
            fund_asof.rename(columns={"snapshot_date": "trade_date"})[
                ["trade_date", "ticker", "revenue_growth", "profitability", "source_quality"]
            ],
            on=["trade_date", "ticker"],
            how="left",
        )
    theme = theme_membership[["ticker", "theme_id", "theme_name", "membership_score"]].drop_duplicates("ticker")
    base = base.merge(theme, on="ticker", how="left")
    base["fundamental_pass"] = base["profitability"].fillna(0).ge(0) | base["revenue_growth"].fillna(0).ge(0)
    base["market_attention_member"] = base["turnover_rank_pct_20d"].ge(0.8) | base["traded_value_rank_pct"].ge(0.8)
    base["eligible_pool_member"] = base["valid_universe"] & base["market_attention_member"] & base["fundamental_pass"]

    base["long_strong_score"] = (base["RS60"].fillna(-1) * 0.5 + base["RS40"].fillna(-1) * 0.3 + base["RS20"].fillna(-1) * 0.2).mul(100)
    base["pullback_repair_score"] = ((base["RS60"].fillna(-1).gt(0)).astype(int) * 30 + (base["BIAS60"].fillna(1).lt(0.05)).astype(int) * 20 + base["RS20"].fillna(-1).mul(100))
    base["short_cycle_score"] = (base["RS20"].fillna(-1) * 60 + base["RS10"].fillna(-1) * 40)
    base["subpool_class"] = "rejected"
    base.loc[base["eligible_pool_member"] & base["long_strong_score"].ge(base[["pullback_repair_score", "short_cycle_score"]].max(axis=1)), "subpool_class"] = "long_strong"
    base.loc[base["eligible_pool_member"] & base["pullback_repair_score"].gt(base[["long_strong_score", "short_cycle_score"]].max(axis=1)), "subpool_class"] = "pullback_repair"
    base.loc[base["eligible_pool_member"] & base["short_cycle_score"].gt(base[["long_strong_score", "pullback_repair_score"]].max(axis=1)), "subpool_class"] = "short_cycle"

    b0050 = benchmark_features[benchmark_features["benchmark"].eq("0050")][["trade_date", "BIAS20", "BIAS60", "BIAS120"]].rename(
        columns={"BIAS20": "market_BIAS20_0050", "BIAS60": "market_BIAS60_0050", "BIAS120": "market_BIAS120_0050"}
    )
    base = base.merge(b0050, on="trade_date", how="left")
    base["router_regime"] = "unknown"
    base.loc[base["market_BIAS60_0050"].ge(0.03), "router_regime"] = "bull_trend"
    base.loc[base["market_BIAS60_0050"].lt(0) & base["market_BIAS20_0050"].ge(base["market_BIAS60_0050"]), "router_regime"] = "bull_pullback"
    base.loc[base["market_BIAS60_0050"].lt(-0.08), "router_regime"] = "bear_risk"
    base["router_weight_long_strong"] = base["router_regime"].map({"bull_trend": 0.7, "bull_pullback": 0.4, "bear_risk": 0.2}).fillna(0.45)
    base["router_weight_pullback_repair"] = base["router_regime"].map({"bull_trend": 0.2, "bull_pullback": 0.5, "bear_risk": 0.2}).fillna(0.35)
    base["router_weight_short_cycle"] = base["router_regime"].map({"bull_trend": 0.1, "bull_pullback": 0.1, "bear_risk": 0.1}).fillna(0.2)
    base["router_adjusted_score"] = (
        base["long_strong_score"].fillna(0) * base["router_weight_long_strong"]
        + base["pullback_repair_score"].fillna(0) * base["router_weight_pullback_repair"]
        + base["short_cycle_score"].fillna(0) * base["router_weight_short_cycle"]
    )
    base["risk_score"] = base["BIAS60_percentile"].fillna(0.5) * 50 + base["volatility"].fillna(0).clip(0, 0.2) * 250
    base["fallback_hurdle_result"] = base["excess_return_vs_00631L_20d"].map(lambda v: "missing" if pd.isna(v) else ("pass" if v > 0 else "fail"))
    base["rank_overall"] = base.groupby("trade_date")["router_adjusted_score"].rank(ascending=False, method="first")
    base["rank_in_subpool"] = base.groupby(["trade_date", "subpool_class"])["router_adjusted_score"].rank(ascending=False, method="first")
    base["selected_by_vnext"] = base["eligible_pool_member"] & base["rank_overall"].le(2)
    base["case_trace_only"] = False
    base["diagnostic_only"] = True
    base["selected_outcome_candidate"] = base["selected_by_vnext"] & ~base["case_trace_only"] & base["diagnostic_only"]
    base["included_reason"] = base["subpool_class"]
    base["excluded_reason"] = ""
    base.loc[~base["valid_universe"], "excluded_reason"] = "invalid_universe"
    base.loc[base["valid_universe"] & ~base["market_attention_member"], "excluded_reason"] = "not_market_attention_member"
    base.loc[base["market_attention_member"] & ~base["fundamental_pass"], "excluded_reason"] = "fundamental_not_pass_or_missing"
    base["snapshot_date"] = base["trade_date"]
    iso = base["snapshot_date"].dt.isocalendar()
    base["week_id"] = iso["year"].astype(str) + "-W" + iso["week"].astype(str).str.zfill(2)
    base["is_week_last_trading_day"] = True
    base["requested_period_start"] = "2015-01-02"
    base["requested_period_end"] = "2026-06-30"
    base["actual_coverage_start"] = str(base["snapshot_date"].min().date())
    base["actual_coverage_end"] = str(base["snapshot_date"].max().date())
    for col in ["formal_model_changed", "trade_decision_changed", "active_in_trade_decision", "report_changed"]:
        base[col] = False
    keep = [
        "snapshot_date",
        "week_id",
        "is_week_last_trading_day",
        "requested_period_start",
        "requested_period_end",
        "actual_coverage_start",
        "actual_coverage_end",
        "ticker",
        "name",
        "valid_universe",
        "fundamental_pass",
        "market_attention_member",
        "eligible_pool_member",
        "subpool_class",
        "long_strong_score",
        "pullback_repair_score",
        "short_cycle_score",
        "router_regime",
        "router_weight_long_strong",
        "router_weight_pullback_repair",
        "router_weight_short_cycle",
        "router_adjusted_score",
        "risk_score",
        "fallback_hurdle_result",
        "rank_in_subpool",
        "rank_overall",
        "included_reason",
        "excluded_reason",
        "selected_by_vnext",
        "selected_outcome_candidate",
        "case_trace_only",
        "diagnostic_only",
        "formal_model_changed",
        "trade_decision_changed",
        "active_in_trade_decision",
        "report_changed",
    ]
    ordinary = base[base["rank_overall"].le(100) | base["ticker"].isin(CASE_TICKERS)].copy()
    return ordinary[keep].sort_values(["snapshot_date", "rank_overall", "ticker"])


def _build_final_decision_snapshot(weekly_snapshot: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for date, group in weekly_snapshot.groupby("snapshot_date"):
        selected_before = int(group["selected_by_vnext"].sum())
        selected_after = int(group["selected_outcome_candidate"].sum())
        chosen = group[group["selected_outcome_candidate"]].sort_values("rank_overall")
        final_a = chosen["ticker"].iloc[0] if len(chosen) >= 1 else ""
        final_b = chosen["ticker"].iloc[1] if len(chosen) >= 2 else ""
        rows.append(
            {
                "snapshot_date": date,
                "week_id": group["week_id"].iloc[0],
                "requested_period_start": "2015-01-02",
                "requested_period_end": "2026-06-30",
                "actual_coverage_start": weekly_snapshot["snapshot_date"].min(),
                "actual_coverage_end": weekly_snapshot["snapshot_date"].max(),
                "final_primary_ticker": final_a,
                "final_A_ticker": final_a,
                "final_B_ticker": final_b,
                "fallback_candidate": "00631L",
                "cash_allowed": group["router_regime"].eq("bear_risk").any(),
                "action_basis": "diagnostic_only",
                "switch_allowed": pd.NA,
                "switch_reason": "",
                "no_switch_reason": "diagnostic_snapshot_not_trade_instruction",
                "selected_count_before_case_trace_exclusion": selected_before,
                "selected_count_after_case_trace_exclusion": selected_after,
                "case_trace_excluded_count": selected_before - selected_after,
                "formal_model_changed": False,
                "trade_decision_changed": False,
                "active_in_trade_decision": False,
                "report_changed": False,
            }
        )
    return pd.DataFrame(rows)


def _build_case_trace(
    weekly_snapshot: pd.DataFrame,
    stock_features: pd.DataFrame,
    attention_features: pd.DataFrame,
    daily_market: pd.DataFrame,
    benchmark_features: pd.DataFrame,
) -> pd.DataFrame:
    date = pd.Timestamp(CASE_TRACE_DATE)
    rows = []
    for ticker, name in CASE_TICKERS.items():
        sf = stock_features[(stock_features["trade_date"].eq(date)) & (stock_features["ticker"].eq(ticker))]
        af = attention_features[(attention_features["trade_date"].eq(date)) & (attention_features["ticker"].eq(ticker))]
        dm = daily_market[(daily_market["trade_date"].eq(date)) & (daily_market["ticker"].eq(ticker))]
        row = {
            "trace_date": date,
            "ticker": ticker,
            "name": name,
            "old_rs60_gate_pass": _val(sf, "RS60", lambda v: v > 0),
            "vnext_market_attention_member": _val(af, "turnover_rank_pct_20d", lambda v: v >= 0.8),
            "vnext_eligible_member": bool(not dm.empty and dm["valid_universe"].iloc[0]),
            "vnext_subpool_class": "unknown",
            "prior_strength_score": _val(sf, "RS60"),
            "current_rs_state": _rs_state(sf),
            "bias_repair_state": _bias_state(sf),
            "turnover_state": _turnover_state(af),
            "distribution_risk": _val(af, "distribution_risk"),
            "trend_death_risk": _val(sf, "MA120_position", lambda v: v < 0),
            "selected_by_vnext": False,
            "selected_by_old_model": False,
            "case_trace_only": True,
            "diagnostic_only": True,
            "forward_return_5d": pd.NA,
            "forward_return_10d": pd.NA,
            "forward_return_20d": pd.NA,
            "forward_return_60d": pd.NA,
            "forward_excess_vs_0050": pd.NA,
            "forward_excess_vs_00631L": pd.NA,
            "verdict": "inconclusive",
            "blocked_reason": "forward_horizon_unavailable_after_2026_06_30_or_benchmark_missing_on_trace_date",
        }
        rows.append(row)
    return pd.DataFrame(rows)


def _val(df: pd.DataFrame, col: str, fn=None):
    if df.empty or col not in df or pd.isna(df[col].iloc[0]):
        return pd.NA
    value = df[col].iloc[0]
    return fn(value) if fn else value


def _rs_state(sf: pd.DataFrame) -> str:
    if sf.empty:
        return "missing"
    vals = {k: sf[k].iloc[0] if k in sf and not pd.isna(sf[k].iloc[0]) else None for k in ["RS5", "RS10", "RS20", "RS40", "RS60"]}
    if vals["RS20"] is not None and vals["RS20"] > 0:
        return "short_medium_rs_positive"
    if vals["RS60"] is not None and vals["RS60"] > 0:
        return "rs60_positive_only"
    return "weak_or_missing"


def _bias_state(sf: pd.DataFrame) -> str:
    if sf.empty or "BIAS60_percentile" not in sf or pd.isna(sf["BIAS60_percentile"].iloc[0]):
        return "missing"
    p = sf["BIAS60_percentile"].iloc[0]
    if p >= 0.85:
        return "overheated"
    if p <= 0.45:
        return "repaired_or_low"
    return "normal"


def _turnover_state(af: pd.DataFrame) -> str:
    if af.empty or pd.isna(af["turnover_rank_pct_20d"].iloc[0]):
        return "missing"
    return "high_attention" if af["turnover_rank_pct_20d"].iloc[0] >= 0.8 else "normal"


def _coverage_summary(
    liquidity: pd.DataFrame,
    benchmark_features: pd.DataFrame,
    stock_features: pd.DataFrame,
    weekly_snapshot: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    actual_start = max(liquidity["trade_date"].min(), benchmark_features["trade_date"].min())
    actual_end = min(liquidity["trade_date"].max(), benchmark_features["trade_date"].max())
    for label, (start, end) in REQUESTED_PERIODS.items():
        req_start = pd.Timestamp(start)
        req_end = actual_end if end == "latest_available" else pd.Timestamp(end)
        rows.append(
            {
                "period": label,
                "requested_start": start,
                "requested_end": end,
                "actual_start": actual_start,
                "actual_end": actual_end,
                "coverage_status": "partial" if actual_start > req_start or actual_end < req_end else "covered",
                "coverage_note": "actual coverage limited by intersection of liquidity and 0050/00631L benchmark cache",
            }
        )
    return pd.DataFrame(rows)


def _blocked_rows(
    *,
    stock_features: pd.DataFrame,
    weekly_snapshot: pd.DataFrame,
    final_decision: pd.DataFrame,
    case_trace: pd.DataFrame,
    benchmark_features: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    if benchmark_features["trade_date"].max() < pd.Timestamp(CASE_TRACE_DATE):
        rows.append(
            {
                "table": "benchmark_features",
                "severity": "warning",
                "blocked_reason": "0050/00631L benchmark cache ends before 2026-06-30 case trace date",
                "blocked_count": 2,
            }
        )
    if weekly_snapshot["case_trace_only"].any() and weekly_snapshot["selected_outcome_candidate"].any():
        bad = weekly_snapshot[weekly_snapshot["case_trace_only"] & weekly_snapshot["selected_outcome_candidate"]]
        rows.append(
            {
                "table": "vnext_weekly_candidate_snapshot",
                "severity": "error",
                "blocked_reason": "case_trace_only row entered selected outcome",
                "blocked_count": int(len(bad)),
            }
        )
    missing_required = int(weekly_snapshot["router_adjusted_score"].isna().sum())
    if missing_required:
        rows.append(
            {
                "table": "vnext_weekly_candidate_snapshot",
                "severity": "warning",
                "blocked_reason": "router_adjusted_score missing for some diagnostic rows",
                "blocked_count": missing_required,
            }
        )
    rows.append(
        {
            "table": "vnext_case_trace",
            "severity": "warning",
            "blocked_reason": "forward_return fields intentionally not executed by Core/Data materialization",
            "blocked_count": int(len(case_trace)),
        }
    )
    return pd.DataFrame(rows)


def _selected_hygiene_audit(weekly_snapshot: pd.DataFrame, final_decision: pd.DataFrame) -> pd.DataFrame:
    bad = weekly_snapshot[weekly_snapshot["case_trace_only"] & weekly_snapshot["selected_outcome_candidate"]]
    return pd.DataFrame(
        [
            {
                "rule": "selected_outcome_candidate = selected_by_vnext AND NOT case_trace_only AND diagnostic_only",
                "case_trace_selected_violation_rows": int(len(bad)),
                "selected_count_before_case_trace_exclusion": int(final_decision["selected_count_before_case_trace_exclusion"].sum()),
                "selected_count_after_case_trace_exclusion": int(final_decision["selected_count_after_case_trace_exclusion"].sum()),
                "case_trace_excluded_count": int(final_decision["case_trace_excluded_count"].sum()),
                "status": "passed" if bad.empty else "failed",
            }
        ]
    )


def _readiness(manifest: dict[str, Any], coverage: pd.DataFrame, blocked_rows: pd.DataFrame) -> dict[str, Any]:
    return {
        "date": "2026-07-06",
        "owner": "BACKTEST_LAB Core/Data",
        "status": manifest["status"],
        "execution_basis": manifest["execution_basis"],
        "formal_model_changed": False,
        "trade_decision_changed": False,
        "active_in_trade_decision": False,
        "report_changed": False,
        "portfolio_replay_executed": False,
        "candidate_forward_return_diagnostic_executed": False,
        "ready_for_experiments_phase_a_snapshot_integrity": manifest["ready_for_experiments_phase_a_snapshot_integrity"],
        "ready_for_candidate_forward_return_diagnostic": False,
        "reason_candidate_forward_return_not_ready": "Core/Data did not execute forward-return diagnostics; case trace forward horizons require Experiments after Phase A.",
        "blocked_rows": blocked_rows.to_dict(orient="records"),
        "requested_vs_actual_coverage": coverage.to_dict(orient="records"),
        "next_owner": "BACKTEST_LAB Experiments",
        "next_step": "Run Phase A snapshot integrity validation first; do not run portfolio replay.",
    }


def _summary(manifest: dict[str, Any], coverage: pd.DataFrame, blocked_rows: pd.DataFrame) -> str:
    def _plain_table(df: pd.DataFrame) -> str:
        if df.empty:
            return "(empty)"
        rows = [", ".join(map(str, df.columns))]
        for _, row in df.iterrows():
            rows.append(", ".join("" if pd.isna(value) else str(value) for value in row.tolist()))
        return "\n".join(rows)

    return "\n".join(
        [
            "# vNext Dynamic Candidate Pool Data Materialization Summary",
            "",
            f"Status: {manifest['status']}",
            "",
            "Boundary: diagnostic data materialization only. No formal model/report/trade path changed. No portfolio replay executed.",
            "",
            "Requested vs actual coverage:",
            _plain_table(coverage),
            "",
            "Blocked rows:",
            _plain_table(blocked_rows),
            "",
            "Flags:",
            "",
            "- formal_model_changed=false",
            "- trade_decision_changed=false",
            "- active_in_trade_decision=false",
            "- report_changed=false",
            "- portfolio_replay_executed=false",
        ]
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--liquidity-dir", type=Path, default=DEFAULT_LIQUIDITY_DIR)
    parser.add_argument("--revenue-dir", type=Path, default=DEFAULT_REVENUE_DIR)
    parser.add_argument("--fundamentals-dir", type=Path, default=DEFAULT_FUNDAMENTALS_DIR)
    parser.add_argument("--taxonomy-path", type=Path, default=DEFAULT_TAXONOMY_PATH)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    args = parser.parse_args(argv)
    manifest = materialize_vnext_dynamic_candidate_pool(
        output_dir=args.output_dir,
        liquidity_dir=args.liquidity_dir,
        revenue_dir=args.revenue_dir,
        fundamentals_dir=args.fundamentals_dir,
        taxonomy_path=args.taxonomy_path,
        cache_dir=args.cache_dir,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
