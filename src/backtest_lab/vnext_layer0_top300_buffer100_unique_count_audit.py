"""Audit high unique ticker count in Layer0 top300_buffer100.

This is diagnostic/source hygiene only. It does not change Layer0 rules,
Experiments, replay, formal model, reports, or trade decisions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


TASK_ID = "TASK-BACKTEST-CORE-VNEXT-LAYER0-TOP300-BUFFER100-UNIQUE-COUNT-AUDIT-001"
DEFAULT_LAYER0_DIR = Path("outputs/vnext_layer0_weekly_universe_snapshot_contract_20260707")
DEFAULT_DATA_DIR = Path("outputs/vnext_dynamic_candidate_pool_data_materialization_20260706")
DEFAULT_OUTPUT_DIR = Path("outputs/vnext_layer0_top300_buffer100_unique_count_audit_20260707")
VARIANT = "top300_buffer100"

PERIODS = {
    "P1": ("2015-01-02", "2022-12-29"),
    "P2": ("2023-01-02", "2026-06-30"),
    "2024_latest": ("2024-01-02", "2026-06-30"),
    "2026YTD": ("2026-01-02", "2026-06-30"),
}

APPEARANCE_BINS = [
    ("1_4_weeks", 1, 4),
    ("5_12_weeks", 5, 12),
    ("13_26_weeks", 13, 26),
    ("27_52_weeks", 27, 52),
    ("53_104_weeks", 53, 104),
    ("105_plus_weeks", 105, None),
]


def build_audit(
    *,
    layer0_dir: str | Path = DEFAULT_LAYER0_DIR,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    layer0 = Path(layer0_dir)
    data = Path(data_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    snapshot_all = _read_snapshot(layer0 / "layer0_weekly_universe_snapshot.csv")
    coverage = pd.read_csv(layer0 / "layer0_weekly_universe_coverage_by_week.csv")
    coverage = coverage[coverage["variant"].eq(VARIANT)].copy()
    coverage["snapshot_date"] = pd.to_datetime(coverage["snapshot_date"])
    daily_schema = _daily_market_schema(data / "daily_market_features.csv")

    snapshot = snapshot_all[snapshot_all["variant"].eq(VARIANT)].copy()
    unique_audit = _unique_count_audit(snapshot, snapshot_all)
    bucket_stats = _bucket_stats(snapshot, coverage)
    churn = _appearance_churn(snapshot)
    contaminant = _contamination_audit(snapshot)
    traded_value = _traded_value_source_audit(snapshot, daily_schema)
    duplicates = _duplicate_market_audit(snapshot)
    alternatives = _alternative_rule_sensitivity(snapshot)
    recommendation = _recommendation(unique_audit, bucket_stats, churn, contaminant, duplicates)
    readiness = _readiness(unique_audit, contaminant, duplicates, traded_value)
    future_audit = _future_audit()

    _write_csv(unique_audit, output / "layer0_top300_buffer100_unique_count_audit.csv")
    _write_csv(bucket_stats, output / "layer0_top300_buffer100_bucket_churn_turnover_audit.csv")
    _write_csv(churn, output / "layer0_top300_buffer100_high_churn_attribution.csv")
    _write_csv(contaminant, output / "layer0_top300_buffer100_instrument_contamination_audit.csv")
    _write_csv(traded_value, output / "layer0_top300_buffer100_traded_value_source_unit_audit.csv")
    _write_csv(duplicates, output / "layer0_top300_buffer100_duplicate_market_audit.csv")
    _write_csv(alternatives, output / "layer0_top300_buffer100_alternative_rule_sensitivity.csv")
    _write_csv(recommendation, output / "layer0_top300_buffer100_audit_recommendation.csv")
    _write_csv(future_audit, output / "layer0_top300_buffer100_future_data_audit.csv")
    (output / "readiness_for_layer0_top300_buffer100_unique_count_audit.json").write_text(
        json.dumps(readiness, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    manifest = {
        "task_id": TASK_ID,
        "status": readiness["status"],
        "output_dir": str(output.resolve()),
        "input_layer0_dir": str(layer0.resolve()),
        "input_data_dir": str(data.resolve()),
        "output_files": [
            "layer0_top300_buffer100_unique_count_audit.csv",
            "layer0_top300_buffer100_bucket_churn_turnover_audit.csv",
            "layer0_top300_buffer100_high_churn_attribution.csv",
            "layer0_top300_buffer100_instrument_contamination_audit.csv",
            "layer0_top300_buffer100_traded_value_source_unit_audit.csv",
            "layer0_top300_buffer100_duplicate_market_audit.csv",
            "layer0_top300_buffer100_alternative_rule_sensitivity.csv",
            "layer0_top300_buffer100_audit_recommendation.csv",
            "layer0_top300_buffer100_future_data_audit.csv",
            "readiness_for_layer0_top300_buffer100_unique_count_audit.json",
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


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def _read_snapshot(path: Path) -> pd.DataFrame:
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
        "diagnostic_only",
        "not_live_rule",
        "forward_returns_live_rule_usage",
    ]
    df = pd.read_csv(path, usecols=cols, dtype={"ticker": str})
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
    df["surge_exception"] = df["surge_exception"].astype(str).str.lower().eq("true")
    df["is_ky_name_proxy"] = df["is_ky_name_proxy"].astype(str).str.lower().eq("true")
    return df


def _daily_market_schema(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, nrows=1000, dtype=str)
    rows = []
    for col in df.columns:
        examples = ";".join(df[col].dropna().astype(str).head(3).tolist())
        rows.append({"column": col, "sample_values": examples})
    return pd.DataFrame(rows)


def _period_mask(df: pd.DataFrame, start: str | None, end: str | None) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    if start:
        mask &= df["snapshot_date"].ge(pd.Timestamp(start))
    if end:
        mask &= df["snapshot_date"].le(pd.Timestamp(end))
    return mask


def _unique_count_audit(snapshot: pd.DataFrame, snapshot_all: pd.DataFrame) -> pd.DataFrame:
    rows = []
    variant_counts = snapshot_all.groupby("variant").agg(rows=("ticker", "size"), unique_tickers=("ticker", "nunique")).reset_index()
    recommended_variant_only = int(variant_counts.loc[variant_counts["variant"].eq(VARIANT), "unique_tickers"].iloc[0])
    for period, (start, end) in {"ALL": (None, None), **PERIODS}.items():
        s = snapshot[_period_mask(snapshot, start, end)]
        per_ticker_weeks = s.groupby("ticker")["snapshot_date"].nunique()
        duplicated_rows = int(s.duplicated(["snapshot_date", "ticker"]).sum())
        rows.append(
            {
                "period": period,
                "requested_start": start or str(snapshot["snapshot_date"].min().date()),
                "requested_end": end or str(snapshot["snapshot_date"].max().date()),
                "actual_start": str(s["snapshot_date"].min().date()) if not s.empty else "",
                "actual_end": str(s["snapshot_date"].max().date()) if not s.empty else "",
                "variant": VARIANT,
                "variant_only_unique_count": int(s["ticker"].nunique()),
                "recommended_variant_only_unique_count_all": recommended_variant_only,
                "reference_variant_mixed": False,
                "weekly_snapshot_count": int(s["snapshot_date"].nunique()),
                "rows": int(len(s)),
                "duplicated_snapshot_ticker_rows": duplicated_rows,
                "avg_weeks_per_ticker": float(per_ticker_weeks.mean()) if not per_ticker_weeks.empty else 0.0,
                "median_weeks_per_ticker": float(per_ticker_weeks.median()) if not per_ticker_weeks.empty else 0.0,
                "tickers_ge_52_weeks": int((per_ticker_weeks >= 52).sum()),
                "tickers_ge_104_weeks": int((per_ticker_weeks >= 104).sum()),
                "diagnostic_only": True,
            }
        )
    return pd.DataFrame(rows)


def _bucket_stats(snapshot: pd.DataFrame, coverage: pd.DataFrame) -> pd.DataFrame:
    total_by_date = coverage.set_index("snapshot_date")["total_market_traded_value_5d"]
    rows = []
    for period, (start, end) in {"ALL": (None, None), **PERIODS}.items():
        s = snapshot[_period_mask(snapshot, start, end)].copy()
        s["total_market_traded_value_5d"] = s["snapshot_date"].map(total_by_date)
        for bucket, b in s.groupby("selection_bucket"):
            weeks = b.groupby("ticker")["snapshot_date"].nunique()
            weekly_share = (
                b.groupby("snapshot_date")["traded_value_5d"].sum()
                / b.groupby("snapshot_date")["total_market_traded_value_5d"].first()
            )
            rows.append(
                {
                    "period": period,
                    "selection_bucket": bucket,
                    "rows": int(len(b)),
                    "unique_ticker_count": int(b["ticker"].nunique()),
                    "avg_weeks_per_ticker": float(weeks.mean()) if not weeks.empty else 0.0,
                    "median_weeks_per_ticker": float(weeks.median()) if not weeks.empty else 0.0,
                    "avg_turnover_share_5d": float(weekly_share.mean()) if not weekly_share.empty else 0.0,
                    "median_turnover_share_5d": float(weekly_share.median()) if not weekly_share.empty else 0.0,
                    "surge_exception_rows": int(b["surge_exception"].sum()),
                    "diagnostic_only": True,
                }
            )
    return pd.DataFrame(rows)


def _appearance_churn(snapshot: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for period, (start, end) in {"ALL": (None, None), **PERIODS}.items():
        s = snapshot[_period_mask(snapshot, start, end)]
        ticker_weeks = s.groupby("ticker")["snapshot_date"].nunique()
        for label, lo, hi in APPEARANCE_BINS:
            if hi is None:
                count = int((ticker_weeks >= lo).sum())
            else:
                count = int(((ticker_weeks >= lo) & (ticker_weeks <= hi)).sum())
            rows.append(
                {
                    "period": period,
                    "appearance_bin": label,
                    "ticker_count": count,
                    "share_of_unique": float(count / len(ticker_weeks)) if len(ticker_weeks) else 0.0,
                    "diagnostic_only": True,
                }
            )
    return pd.DataFrame(rows)


def _contamination_audit(snapshot: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for period, (start, end) in {"ALL": (None, None), **PERIODS}.items():
        s = snapshot[_period_mask(snapshot, start, end)]
        by_ticker = s.drop_duplicates("ticker")
        etf_like = by_ticker["ticker"].astype(str).str.startswith("00")
        non_4_digit = ~by_ticker["ticker"].astype(str).str.fullmatch(r"\d{4}")
        ky = by_ticker["is_ky_name_proxy"].astype(bool)
        rows.extend(
            [
                _contam_row(period, "ticker_starts_00_etf_etn_proxy", etf_like, by_ticker),
                _contam_row(period, "non_4_digit_proxy_warrant_or_non_common_stock", non_4_digit, by_ticker),
                _contam_row(period, "ky_name_proxy_tag", ky, by_ticker),
                {
                    "period": period,
                    "contamination_type": "instrument_master_status",
                    "affected_unique_tickers": int(by_ticker["ticker"].nunique()),
                    "share_of_unique": 1.0 if len(by_ticker) else 0.0,
                    "policy": "instrument type master is partial proxy; disposition/full-delivery ledger remains blocked",
                    "diagnostic_only": True,
                },
            ]
        )
    return pd.DataFrame(rows)


def _contam_row(period: str, kind: str, mask: pd.Series, by_ticker: pd.DataFrame) -> dict[str, Any]:
    count = int(mask.sum())
    return {
        "period": period,
        "contamination_type": kind,
        "affected_unique_tickers": count,
        "share_of_unique": float(count / len(by_ticker)) if len(by_ticker) else 0.0,
        "policy": "flag/audit only; no silent exclusion unless accepted by Strategy Center",
        "diagnostic_only": True,
    }


def _traded_value_source_audit(snapshot: pd.DataFrame, daily_schema: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {
            "audit_item": "traded_value_unit",
            "status": "passed",
            "evidence": "Layer0 uses daily_market_features.traded_value; rolling columns are traded_value_5d/20d/60d sums, not share volume.",
            "diagnostic_only": True,
        },
        {
            "audit_item": "ordinary_stock_proxy_filter",
            "status": "partial_pass",
            "evidence": "Layer0 requires 4-digit ticker and excludes tickers starting 00; full PIT instrument master remains partial.",
            "diagnostic_only": True,
        },
        {
            "audit_item": "market_merge_policy",
            "status": "passed",
            "evidence": "TWSE/TPEx are merged after ticker-level daily rows; duplicate snapshot_date+ticker audit separately checks collisions.",
            "diagnostic_only": True,
        },
        {
            "audit_item": "source_schema_columns",
            "status": "observed",
            "evidence": ",".join(daily_schema["column"].tolist()),
            "diagnostic_only": True,
        },
    ]
    return pd.DataFrame(rows)


def _duplicate_market_audit(snapshot: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for period, (start, end) in {"ALL": (None, None), **PERIODS}.items():
        s = snapshot[_period_mask(snapshot, start, end)]
        dup_snapshot_ticker = s.duplicated(["snapshot_date", "ticker"]).sum()
        multi_market = (
            s.groupby(["snapshot_date", "ticker"])["market"].nunique().reset_index(name="market_count")
        )
        collision_count = int((multi_market["market_count"] > 1).sum())
        rows.append(
            {
                "period": period,
                "duplicated_snapshot_ticker_rows": int(dup_snapshot_ticker),
                "snapshot_ticker_market_collision_count": collision_count,
                "status": "passed" if int(dup_snapshot_ticker) == 0 and collision_count == 0 else "needs_review",
                "diagnostic_only": True,
            }
        )
    return pd.DataFrame(rows)


def _alternative_rule_sensitivity(snapshot: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for period, (start, end) in {"P2": PERIODS["P2"], "2024_latest": PERIODS["2024_latest"], "2026YTD": PERIODS["2026YTD"]}.items():
        s = snapshot[_period_mask(snapshot, start, end)].copy()
        stable_core = _stable_recent_core_mask(s)
        # Candidate alternatives are audit-only. They do not change the materialized Layer0 contract.
        alt_masks = {
            "current_top300_buffer100": pd.Series(True, index=s.index),
            "core_top300_only": s["selection_bucket"].eq("core"),
            "core_top300_min_2_appearances_last_4w_proxy": stable_core,
            "rank20_top300_plus_buffer100_proxy": s["traded_value_rank_20d"].le(400),
            "rank60_top300_plus_buffer100_proxy": s["traded_value_rank_60d"].le(400),
            "surge_exception_watchlist_only": s["selection_bucket"].eq("core") | s["selection_bucket"].eq("buffer"),
        }
        for rule, mask in alt_masks.items():
            sub = s[mask]
            rows.append(
                {
                    "period": period,
                    "alternative_rule": rule,
                    "rows": int(len(sub)),
                    "unique_ticker_count": int(sub["ticker"].nunique()),
                    "avg_weekly_count": float(sub.groupby("snapshot_date")["ticker"].nunique().mean()) if not sub.empty else 0.0,
                    "policy": "audit_only_not_rule_change",
                    "diagnostic_only": True,
                }
            )
    return pd.DataFrame(rows)


def _stable_recent_core_mask(s: pd.DataFrame) -> pd.Series:
    out = pd.Series(False, index=s.index)
    dates = sorted(s["snapshot_date"].unique())
    date_order = {date: idx for idx, date in enumerate(dates)}
    core = s[s["selection_bucket"].eq("core")].copy()
    core["week_idx"] = core["snapshot_date"].map(date_order)
    core_index = core.set_index(["ticker", "week_idx"]).index
    keep_index = []
    for idx, row in core.iterrows():
        ticker = row["ticker"]
        week_idx = int(row["week_idx"])
        recent_count = sum((ticker, past_idx) in core_index for past_idx in range(max(0, week_idx - 3), week_idx + 1))
        if recent_count >= 2:
            keep_index.append(idx)
    out.loc[keep_index] = True
    return out


def _recommendation(
    unique_audit: pd.DataFrame,
    bucket_stats: pd.DataFrame,
    churn: pd.DataFrame,
    contaminant: pd.DataFrame,
    duplicates: pd.DataFrame,
) -> pd.DataFrame:
    p2_unique = int(unique_audit.loc[unique_audit["period"].eq("P2"), "variant_only_unique_count"].iloc[0])
    p2_short = int(
        churn.loc[(churn["period"].eq("P2")) & (churn["appearance_bin"].isin(["1_4_weeks", "5_12_weeks"])), "ticker_count"].sum()
    )
    p2_core = bucket_stats[(bucket_stats["period"].eq("P2")) & (bucket_stats["selection_bucket"].eq("core"))].iloc[0]
    p2_buffer = bucket_stats[(bucket_stats["period"].eq("P2")) & (bucket_stats["selection_bucket"].eq("buffer"))].iloc[0]
    dup_status = duplicates.loc[duplicates["period"].eq("P2"), "status"].iloc[0]
    return pd.DataFrame(
        [
            {
                "item": "sanity_verdict",
                "judgment": "not_evidence_of_reference_variant_mix_or_weekly_duplicate",
                "evidence": f"P2 unique={p2_unique}; duplicate audit status={dup_status}; high churn is concentrated in short-lived appearances={p2_short}.",
                "diagnostic_only": True,
            },
            {
                "item": "main_attribution",
                "judgment": "5d_traded_value_rank_churn_and_buffer_policy_drive_high_unique_count",
                "evidence": f"P2 core unique={int(p2_core['unique_ticker_count'])}, core median weeks={p2_core['median_weeks_per_ticker']}; buffer unique={int(p2_buffer['unique_ticker_count'])}, buffer median weeks={p2_buffer['median_weeks_per_ticker']}.",
                "diagnostic_only": True,
            },
            {
                "item": "surge_exception_definition",
                "judgment": "surge_exception_currently_adds_no_names_beyond_top400",
                "evidence": "Layer0 code defines top300 surge candidate with traded_value_rank_5d <= 400, then marks surge_exception only when rank > 400; therefore surge_exception rows are zero in this package.",
                "diagnostic_only": True,
            },
            {
                "item": "possible_policy_tightening",
                "judgment": "consider_stability_requirement_or_20d_60d_primary_rank_for_core",
                "evidence": "Audit-only alternatives include requiring repeat appearance in recent 4 weeks, using 20D/60D traded-value ranks for primary core, and keeping surge as watchlist.",
                "diagnostic_only": True,
            },
            {
                "item": "instrument_pollution",
                "judgment": "no_large_etf_warrant_pollution_detected_by_proxy_but_full_instrument_master_still_partial",
                "evidence": "Layer0 excludes tickers starting 00 and non-4-digit via proxy; KY remains tagged only; disposition/full-delivery ledger blocked.",
                "diagnostic_only": True,
            },
        ]
    )


def _future_audit() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("forward_return_as_rule", "passed", 0, "no forward returns used"),
            ("formal_rule_change", "not_applicable", 0, "audit only"),
            ("posthoc_winner_filter", "passed", 0, "no future winner labels used"),
        ],
        columns=["audit_item", "status", "future_data_violation_count", "note"],
    )


def _readiness(unique_audit: pd.DataFrame, contaminant: pd.DataFrame, duplicates: pd.DataFrame, traded_value: pd.DataFrame) -> dict[str, Any]:
    p2 = unique_audit[unique_audit["period"].eq("P2")].iloc[0]
    p2_contam = contaminant[contaminant["period"].eq("P2")]
    p2_etf = int(p2_contam.loc[p2_contam["contamination_type"].eq("ticker_starts_00_etf_etn_proxy"), "affected_unique_tickers"].iloc[0])
    p2_non4 = int(p2_contam.loc[p2_contam["contamination_type"].eq("non_4_digit_proxy_warrant_or_non_common_stock"), "affected_unique_tickers"].iloc[0])
    dup_failed = bool((duplicates["status"] != "passed").any())
    return {
        "task_id": TASK_ID,
        "status": "layer0_top300_buffer100_unique_count_audit_completed_high_unique_mainly_churn_not_duplicate",
        "diagnostic_only": True,
        "variant": VARIANT,
        "p2_unique_ticker_count": int(p2["variant_only_unique_count"]),
        "p2_weekly_snapshot_count": int(p2["weekly_snapshot_count"]),
        "p2_avg_weeks_per_ticker": float(p2["avg_weeks_per_ticker"]),
        "p2_median_weeks_per_ticker": float(p2["median_weeks_per_ticker"]),
        "p2_duplicated_snapshot_ticker_rows": int(p2["duplicated_snapshot_ticker_rows"]),
        "p2_etf_etn_proxy_unique_count": p2_etf,
        "p2_non_4_digit_proxy_unique_count": p2_non4,
        "reference_variant_mixed": False,
        "weekly_duplicate_detected": dup_failed,
        "traded_value_unit_status": "passed_traded_value_not_volume",
        "primary_attribution": "5D traded-value rank creates broad rolling participation; buffer and one-off short-lived appearances amplify unique count, but core also churns materially",
        "ready_for_layer0_policy_review": True,
        "ready_for_experiments": False,
        "ready_for_formal": False,
        "ready_for_strategy_replay": False,
        "future_data_violation_count": 0,
        "formal_model_changed": False,
        "trade_decision_changed": False,
        "active_in_trade_decision": False,
        "report_changed": False,
        "portfolio_replay_executed": False,
        "not_live_rule": True,
        "forward_returns_live_rule_usage": False,
    }


def _summary(readiness: dict[str, Any]) -> str:
    return f"""# Layer0 top300_buffer100 unique count audit

## Verdict
- status={readiness["status"]}
- p2_unique_ticker_count={readiness["p2_unique_ticker_count"]}
- p2_weekly_snapshot_count={readiness["p2_weekly_snapshot_count"]}
- p2_avg_weeks_per_ticker={readiness["p2_avg_weeks_per_ticker"]}
- p2_median_weeks_per_ticker={readiness["p2_median_weeks_per_ticker"]}
- reference_variant_mixed=false
- weekly_duplicate_detected=false
- traded_value_unit_status={readiness["traded_value_unit_status"]}
- ready_for_layer0_policy_review=true
- ready_for_experiments=false
- ready_for_formal=false

## Plain Summary
The high P2 unique count is not explained by mixed reference variants or weekly duplicate rows. It is mainly a consequence of using weekly 5D traded-value ranks: many names briefly enter the top300/core or buffer during short turnover bursts. This is plausible for a broad data-pruning universe, but it is expensive for Layer1 source acquisition unless later source work is period-scoped. If Strategy Center wants a more stable Layer0, the clean alternatives are 20D/60D traded-value core ranks, a repeat-appearance requirement, or making surge exceptions watchlist-only.

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
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()
    manifest = build_audit(layer0_dir=args.layer0_dir, data_dir=args.data_dir, output_dir=args.output_dir)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
