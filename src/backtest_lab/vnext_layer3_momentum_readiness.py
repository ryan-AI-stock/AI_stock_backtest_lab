"""Build Layer 3 momentum sleeve redesign PIT readiness.

The contract separates gross strength features from risk/exhaustion penalty
features. It is diagnostic readiness only: no selector, no formal rule, and no
portfolio replay.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

TASK_ID = "TASK-BACKTEST-CORE-VNEXT-LAYER3-MOMENTUM-SLEEVE-GROSS-STRENGTH-RISK-PENALTY-READINESS-001"
DEFAULT_MATERIALIZATION_DIR = Path("outputs/vnext_dynamic_candidate_pool_data_materialization_20260706")
DEFAULT_OUTPUT_DIR = Path("outputs/vnext_layer3_momentum_strength_risk_readiness_20260707")

PERIODS = [
    ("P1", "2015-01-02", "2022-12-29"),
    ("P2", "2023-01-02", "2026-06-30"),
    ("2024-latest", "2024-01-02", "2026-06-30"),
    ("2026YTD", "2026-01-02", "2026-06-30"),
]


def build_layer3_momentum_readiness(
    *,
    materialization_dir: str | Path = DEFAULT_MATERIALIZATION_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    materialization = Path(materialization_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    universe = _weekly_universe(materialization / "vnext_weekly_candidate_snapshot.csv")
    stock_daily = _stock_feature_slice(materialization / "stock_features.csv", universe)
    attention = _attention_slice(materialization / "attention_features.csv", universe)
    enriched_daily = _enrich_daily_features(stock_daily)
    joined = _candidate_join(universe, enriched_daily, attention)
    rs_roles = _rs_window_roles(joined)
    strength = _strength_gross_contract(joined)
    penalty = _risk_penalty_contract(joined)
    missingness = _missingness_by_period(joined)
    blocked = _blocked_proxy_fields(joined)
    future_audit = _future_data_audit(joined)
    readiness = _readiness_json(joined, blocked, future_audit)

    _write_csv(strength, output / "layer3_momentum_strength_gross_feature_contract.csv")
    _write_csv(penalty, output / "layer3_momentum_risk_exhaustion_penalty_contract.csv")
    _write_csv(rs_roles, output / "layer3_momentum_rs_window_role_contract.csv")
    _write_csv(joined, output / "layer3_momentum_candidate_join_contract.csv")
    _write_csv(missingness, output / "layer3_momentum_missingness_by_period.csv")
    _write_csv(blocked, output / "layer3_momentum_blocked_proxy_fields.csv")
    _write_csv(future_audit, output / "layer3_momentum_future_data_audit.csv")
    (output / "readiness_for_layer3_momentum_redesign_diagnostic.json").write_text(
        json.dumps(readiness, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    manifest = {
        "task_id": TASK_ID,
        "status": readiness["status"],
        "output_dir": str(output.resolve()),
        "input_materialization_dir": str(materialization.resolve()),
        "output_files": [
            "layer3_momentum_strength_gross_feature_contract.csv",
            "layer3_momentum_risk_exhaustion_penalty_contract.csv",
            "layer3_momentum_rs_window_role_contract.csv",
            "layer3_momentum_candidate_join_contract.csv",
            "layer3_momentum_missingness_by_period.csv",
            "layer3_momentum_blocked_proxy_fields.csv",
            "layer3_momentum_future_data_audit.csv",
            "readiness_for_layer3_momentum_redesign_diagnostic.json",
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


def _weekly_universe(path: Path) -> pd.DataFrame:
    usecols = [
        "snapshot_date",
        "week_id",
        "ticker",
        "name",
        "theme_id",
        "theme_name",
        "subpool_class",
        "long_strong_score",
        "pullback_repair_score",
        "short_cycle_score",
        "rank_in_subpool",
        "rank_overall",
        "turnover_state",
        "risk_score",
        "risk_bucket",
        "final_selector_score_decomposed",
        "selected_by_vnext",
        "selected_outcome_candidate",
        "case_trace_only",
        "diagnostic_only",
    ]
    raw = pd.read_csv(path, usecols=usecols, parse_dates=["snapshot_date"], dtype={"ticker": str})
    raw = raw[raw["diagnostic_only"].astype(bool) & ~raw["case_trace_only"].astype(bool)].copy()
    return raw.rename(columns={"snapshot_date": "signal_date"})


def _stock_feature_slice(path: Path, universe: pd.DataFrame) -> pd.DataFrame:
    tickers = set(universe["ticker"].astype(str))
    usecols = [
        "trade_date",
        "ticker",
        "adjusted_close",
        "return_5d",
        "return_10d",
        "return_20d",
        "return_40d",
        "return_60d",
        "MA20",
        "BIAS20",
        "MA20_position",
        "drawdown_20d",
        "MA60",
        "BIAS60",
        "MA60_position",
        "drawdown_60d",
        "MA120",
        "BIAS120",
        "MA120_position",
        "drawdown_120d",
        "volatility",
        "RS5",
        "RS10",
        "RS20",
        "RS40",
        "RS60",
        "BIAS20_percentile",
        "BIAS60_percentile",
        "BIAS120_percentile",
    ]
    parts = []
    for chunk in pd.read_csv(path, usecols=lambda col: col in usecols, chunksize=500_000, dtype={"ticker": str}):
        chunk = chunk[chunk["ticker"].isin(tickers)]
        if not chunk.empty:
            parts.append(chunk)
    if not parts:
        return pd.DataFrame(columns=usecols)
    out = pd.concat(parts, ignore_index=True)
    out["trade_date"] = pd.to_datetime(out["trade_date"])
    numeric = [col for col in out.columns if col not in {"trade_date", "ticker"}]
    for col in numeric:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.sort_values(["ticker", "trade_date"])


def _attention_slice(path: Path, universe: pd.DataFrame) -> pd.DataFrame:
    dates = set(universe["signal_date"].dt.strftime("%Y-%m-%d"))
    tickers = set(universe["ticker"].astype(str))
    usecols = [
        "trade_date",
        "ticker",
        "traded_value",
        "turnover_20d",
        "turnover_rank_pct_20d",
        "turnover_60d",
        "turnover_rank_pct_60d",
        "traded_value_rank_pct",
        "volume_zscore",
        "high_turnover_price_confirmed",
        "distribution_risk",
    ]
    parts = []
    for chunk in pd.read_csv(path, usecols=lambda col: col in usecols, chunksize=500_000, dtype={"ticker": str}):
        chunk = chunk[chunk["trade_date"].astype(str).isin(dates) & chunk["ticker"].isin(tickers)]
        if not chunk.empty:
            parts.append(chunk)
    if not parts:
        return pd.DataFrame(columns=usecols)
    out = pd.concat(parts, ignore_index=True)
    out["trade_date"] = pd.to_datetime(out["trade_date"])
    return out


def _enrich_daily_features(stock: pd.DataFrame) -> pd.DataFrame:
    stock = stock.copy().sort_values(["ticker", "trade_date"])
    group = stock.groupby("ticker", group_keys=False)
    stock["RS30_proxy"] = stock[["RS20", "RS40"]].mean(axis=1)
    stock["RS5_minus_RS10"] = stock["RS5"] - stock["RS10"]
    stock["RS10_minus_RS20"] = stock["RS10"] - stock["RS20"]
    stock["RS20_minus_RS60"] = stock["RS20"] - stock["RS60"]
    stock["RS20_improving_proxy"] = stock["RS20"] > group["RS20"].shift(5)
    stock["RS30_improving_proxy"] = stock["RS30_proxy"] > group["RS30_proxy"].shift(5)
    stock["RS5_positive_share_20d_proxy"] = group["RS5"].transform(lambda s: s.gt(0).rolling(20, min_periods=10).mean())
    stock["RS5_positive_share_30d_proxy"] = group["RS5"].transform(lambda s: s.gt(0).rolling(30, min_periods=15).mean())
    stock["stock_1d_return"] = group["adjusted_close"].pct_change()
    stock["large_down_day_count_20d"] = group["stock_1d_return"].transform(lambda s: s.lt(-0.04).rolling(20, min_periods=10).sum())
    stock["large_down_day_count_30d"] = group["stock_1d_return"].transform(lambda s: s.lt(-0.04).rolling(30, min_periods=15).sum())
    stock["volatility_20d_proxy"] = group["stock_1d_return"].transform(lambda s: s.rolling(20, min_periods=10).std())
    stock["volatility_60d_proxy"] = group["stock_1d_return"].transform(lambda s: s.rolling(60, min_periods=30).std())
    stock["MA20_slope_20d_proxy"] = stock["MA20"] / group["MA20"].shift(20) - 1
    stock["MA60_slope_20d_proxy"] = stock["MA60"] / group["MA60"].shift(20) - 1
    stock["RS_exhaustion_proxy"] = (
        stock["RS60"].gt(stock.groupby("trade_date")["RS60"].transform(lambda s: s.quantile(0.8)))
        & stock["RS20_minus_RS60"].lt(0)
        & stock["RS10_minus_RS20"].lt(0)
    )
    stock["source_asof_date"] = stock["trade_date"]
    return stock


def _candidate_join(universe: pd.DataFrame, daily: pd.DataFrame, attention: pd.DataFrame) -> pd.DataFrame:
    joined = universe.merge(
        daily,
        left_on=["signal_date", "ticker"],
        right_on=["trade_date", "ticker"],
        how="left",
    ).drop(columns=["trade_date"], errors="ignore")
    joined = joined.merge(
        attention,
        left_on=["signal_date", "ticker"],
        right_on=["trade_date", "ticker"],
        how="left",
    ).drop(columns=["trade_date"], errors="ignore")
    joined["turnover_20d_vs_60d_change"] = joined["turnover_20d"] / joined["turnover_60d"] - 1
    joined["traded_value_rank_pct_20d_proxy"] = joined["traded_value_rank_pct"]
    joined["blowoff_turnover_proxy"] = (
        joined["turnover_rank_pct_20d"].gt(0.9)
        & joined["volume_zscore"].gt(2)
        & joined["RS5_minus_RS10"].lt(0)
    )
    joined["above_MA20"] = joined["MA20_position"].gt(0)
    joined["above_MA60"] = joined["MA60_position"].gt(0)
    joined["above_MA120"] = joined["MA120_position"].gt(0)
    joined["healthy_trend_proxy"] = (
        joined["above_MA20"]
        & joined["above_MA60"]
        & joined["MA20_slope_20d_proxy"].gt(0)
        & joined["BIAS20_percentile"].lt(0.9)
    )
    joined["extreme_overheat_flag"] = joined["BIAS20_percentile"].gt(0.95) | joined["BIAS60_percentile"].gt(0.95)
    joined["extreme_risk_bucket_flag"] = joined["risk_bucket"].astype(str).str.lower().isin(["high", "extreme"])
    joined["rs60_top20_context"] = joined.groupby("signal_date")["RS60"].rank(pct=True, ascending=False).le(0.2)
    joined["bias20_weak_bucket_context"] = joined["BIAS20_percentile"].lt(0.35)
    joined["high_risk_or_overheat_context"] = joined["extreme_risk_bucket_flag"] | joined["extreme_overheat_flag"]
    joined["diagnostic_only"] = True
    joined["not_live_rule"] = True
    joined["forward_return_as_rule"] = False
    return joined


def _rs_window_roles(joined: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "signal_date",
        "ticker",
        "RS5",
        "RS10",
        "RS20",
        "RS30_proxy",
        "RS40",
        "RS60",
        "RS5_minus_RS10",
        "RS10_minus_RS20",
        "RS20_minus_RS60",
        "RS20_improving_proxy",
        "RS30_improving_proxy",
        "RS_exhaustion_proxy",
        "rs60_top20_context",
        "diagnostic_only",
        "not_live_rule",
    ]
    out = joined.reindex(columns=cols)
    out["RS30_source_quality"] = "proxy_from_RS20_RS40"
    out["RS60_role"] = "medium_context_not_hard_gate"
    out["RS20_RS30_role"] = "primary_momentum_context"
    out["RS5_RS10_role"] = "early_acceleration_watchlist_not_standalone_action"
    return out


def _strength_gross_contract(joined: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "signal_date",
        "ticker",
        "name",
        "subpool_class",
        "RS20",
        "RS30_proxy",
        "RS40",
        "RS60",
        "RS5_minus_RS10",
        "RS10_minus_RS20",
        "RS20_minus_RS60",
        "RS5_positive_share_20d_proxy",
        "RS5_positive_share_30d_proxy",
        "turnover_rank_pct_20d",
        "turnover_rank_pct_60d",
        "traded_value_rank_pct",
        "turnover_20d_vs_60d_change",
        "MA20_slope_20d_proxy",
        "MA60_slope_20d_proxy",
        "above_MA20",
        "above_MA60",
        "above_MA120",
        "healthy_trend_proxy",
        "source_asof_date",
        "diagnostic_only",
        "not_live_rule",
    ]
    out = joined.reindex(columns=cols)
    out["strength_gross_components_available"] = out[["RS20", "RS30_proxy", "turnover_rank_pct_20d", "MA20_slope_20d_proxy"]].notna().all(axis=1)
    return out


def _risk_penalty_contract(joined: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "signal_date",
        "ticker",
        "name",
        "risk_score",
        "risk_bucket",
        "BIAS20",
        "BIAS60",
        "BIAS120",
        "BIAS20_percentile",
        "BIAS60_percentile",
        "BIAS120_percentile",
        "volatility",
        "volatility_20d_proxy",
        "volatility_60d_proxy",
        "large_down_day_count_20d",
        "large_down_day_count_30d",
        "drawdown_20d",
        "drawdown_60d",
        "drawdown_120d",
        "RS_exhaustion_proxy",
        "blowoff_turnover_proxy",
        "extreme_overheat_flag",
        "extreme_risk_bucket_flag",
        "high_risk_or_overheat_context",
        "bias20_weak_bucket_context",
        "source_asof_date",
        "diagnostic_only",
        "not_live_rule",
    ]
    out = joined.reindex(columns=cols)
    out["risk_penalty_components_available"] = out[["BIAS20_percentile", "volatility_20d_proxy", "risk_score"]].notna().all(axis=1)
    return out


def _missingness_by_period(joined: pd.DataFrame) -> pd.DataFrame:
    fields = [
        "RS5",
        "RS10",
        "RS20",
        "RS30_proxy",
        "RS40",
        "RS60",
        "RS5_positive_share_20d_proxy",
        "turnover_rank_pct_20d",
        "turnover_rank_pct_60d",
        "MA20_slope_20d_proxy",
        "BIAS20_percentile",
        "BIAS60_percentile",
        "volatility_20d_proxy",
        "large_down_day_count_20d",
        "risk_score",
        "blowoff_turnover_proxy",
    ]
    rows = []
    for period, start, end in PERIODS:
        subset = joined[(joined["signal_date"] >= pd.Timestamp(start)) & (joined["signal_date"] <= pd.Timestamp(end))]
        for field in fields:
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


def _blocked_proxy_fields(joined: pd.DataFrame) -> pd.DataFrame:
    rows = [
        ("RS30", "proxy", "RS30 exact not materialized; RS30_proxy=(RS20+RS40)/2"),
        ("persistence_daily_winrate", "proxy", "exact 1D beat-0050 daily history unavailable; RS5_positive_share_20d/30d used as proxy"),
        ("volatility_large_down_day", "PIT-ready", "computed from stock adjusted_close daily history only"),
        ("blowoff_turnover", "proxy", "turnover/value spike plus RS short-window deterioration proxy; not a formal blow-off definition"),
        ("MA20_MA60_slope", "proxy", "computed from MA level change over 20 trading days"),
        ("stock_BIAS_percentile", "PIT-ready", "materialized in stock_features"),
        ("risk_score_bucket", "PIT-ready", "existing vNext weekly snapshot diagnostic fields"),
        ("forward_return_as_rule", "prohibited", "forward returns are not used"),
    ]
    return pd.DataFrame(
        [
            {
                "field_or_contract": field,
                "status": status,
                "blocked_or_proxy_reason": reason,
                "available_rows": int(joined[field].notna().sum()) if field in joined else 0,
                "diagnostic_only": True,
            }
            for field, status, reason in rows
        ]
    )


def _future_data_audit(joined: pd.DataFrame) -> pd.DataFrame:
    source_dates = pd.to_datetime(joined["source_asof_date"], errors="coerce")
    bad = int((source_dates.notna() & (source_dates > joined["signal_date"])).sum())
    return pd.DataFrame(
        [
            {
                "audit_item": "stock_feature_source_asof_lte_signal_date",
                "status": "passed" if bad == 0 else "failed",
                "future_data_violation_count": bad,
                "note": "stock and derived rolling features are joined on exact signal_date from past/current history",
            },
            {
                "audit_item": "attention_trade_date_equals_signal_date",
                "status": "passed",
                "future_data_violation_count": 0,
                "note": "attention features joined on exact signal_date",
            },
            {
                "audit_item": "forward_return_as_rule",
                "status": "passed",
                "future_data_violation_count": 0,
                "note": "no forward return columns included",
            },
        ]
    )


def _readiness_json(joined: pd.DataFrame, blocked: pd.DataFrame, future_audit: pd.DataFrame) -> dict[str, Any]:
    future_count = int(future_audit["future_data_violation_count"].sum())
    base_fields = ["RS5", "RS10", "RS20", "RS30_proxy", "RS40", "RS60", "BIAS20_percentile", "volatility_20d_proxy"]
    ready = bool(future_count == 0 and all(joined[field].notna().any() for field in base_fields))
    return {
        "date": "2026-07-07",
        "task_id": TASK_ID,
        "owner": "BACKTEST_LAB Core/Data",
        "status": "partial_ready_layer3_momentum_redesign_proxy_limited" if ready else "blocked_layer3_momentum_redesign",
        "ready_for_layer3_momentum_redesign_event_diagnostic": ready,
        "rs30_exact_available": False,
        "rs30_proxy_available": True,
        "persistence_daily_winrate_available": False,
        "persistence_rs5_positive_share_proxy_available": True,
        "volatility_large_down_day_available": True,
        "blowoff_turnover_available": True,
        "blowoff_turnover_source_quality": "proxy",
        "ready_for_portfolio_like_diagnostic": False,
        "ready_for_strategy_replay": False,
        "ready_for_formal": False,
        "future_data_violation_count": future_count,
        "not_live_rule": True,
        "forward_returns_live_rule_usage": False,
        "candidate_join_rows": int(len(joined)),
        "blocked_fields": blocked[blocked["status"].eq("blocked")]["field_or_contract"].tolist(),
        "proxy_fields": blocked[blocked["status"].eq("proxy")]["field_or_contract"].tolist(),
        "formal_model_changed": False,
        "trade_decision_changed": False,
        "active_in_trade_decision": False,
        "report_changed": False,
        "portfolio_replay_executed": False,
    }


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def _summary(readiness: dict[str, Any], blocked: pd.DataFrame) -> str:
    return "\n".join(
        [
            "# Layer3 Momentum Strength Gross / Risk Penalty Readiness",
            "",
            f"Status: {readiness['status']}",
            "",
            "Boundary: feature/contract readiness only; no selector, no live rule, no replay, no formal/report/trade change.",
            "",
            "Readiness:",
            f"- ready_for_layer3_momentum_redesign_event_diagnostic={str(readiness['ready_for_layer3_momentum_redesign_event_diagnostic']).lower()}",
            "- rs30_exact_available=false",
            "- rs30_proxy_available=true",
            "- persistence_daily_winrate_available=false",
            "- persistence_rs5_positive_share_proxy_available=true",
            "- volatility_large_down_day_available=true",
            "- blowoff_turnover_available=true",
            "- blowoff_turnover_source_quality=proxy",
            "- ready_for_portfolio_like_diagnostic=false",
            "- ready_for_strategy_replay=false",
            "- ready_for_formal=false",
            f"- future_data_violation_count={readiness['future_data_violation_count']}",
            "",
            "Blocked / proxy fields:",
            *[f"- {row.field_or_contract}: {row.status}; {row.blocked_or_proxy_reason}" for row in blocked.itertuples()],
            "",
            "Flags:",
            "- formal_model_changed=false",
            "- trade_decision_changed=false",
            "- active_in_trade_decision=false",
            "- report_changed=false",
            "- portfolio_replay_executed=false",
            "- ready_for_strategy_replay=false",
            "- not_live_rule=true",
            "- forward_returns_live_rule_usage=false",
        ]
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--materialization-dir", type=Path, default=DEFAULT_MATERIALIZATION_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    manifest = build_layer3_momentum_readiness(
        materialization_dir=args.materialization_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
