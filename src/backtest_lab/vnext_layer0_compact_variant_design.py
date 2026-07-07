"""Design compact Layer0 variants to reduce rolling unique ticker bloat.

This is diagnostic/readiness only. It does not change the active Layer0
contract, run Experiments, replay, formal model, report, or trade decision.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


TASK_ID = "TASK-BACKTEST-CORE-VNEXT-LAYER0-COMPACT-VARIANT-DESIGN-READINESS-001"
DEFAULT_DATA_DIR = Path("outputs/vnext_dynamic_candidate_pool_data_materialization_20260706")
DEFAULT_AUDIT_DIR = Path("outputs/vnext_layer0_top300_buffer100_unique_count_audit_20260707")
DEFAULT_OUTPUT_DIR = Path("outputs/vnext_layer0_compact_variant_design_20260707")
FULL_UNIVERSE_ESTIMATE = 1900

PERIODS = {
    "P1": ("2015-01-02", "2022-12-29"),
    "P2": ("2023-01-02", "2026-06-30"),
    "2024_latest": ("2024-01-02", "2026-06-30"),
    "2026YTD": ("2026-01-02", "2026-06-30"),
}

VARIANTS = [
    {
        "variant": "top200_core_conditional_buffer50",
        "core_rank_col": "traded_value_rank_5d",
        "core_n": 200,
        "buffer_n": 50,
        "conditional_buffer": True,
    },
    {
        "variant": "top250_core_conditional_buffer50",
        "core_rank_col": "traded_value_rank_5d",
        "core_n": 250,
        "buffer_n": 50,
        "conditional_buffer": True,
    },
    {
        "variant": "top250_core_conditional_buffer100",
        "core_rank_col": "traded_value_rank_5d",
        "core_n": 250,
        "buffer_n": 100,
        "conditional_buffer": True,
    },
    {
        "variant": "top300_core_no_buffer_reference",
        "core_rank_col": "traded_value_rank_5d",
        "core_n": 300,
        "buffer_n": 0,
        "conditional_buffer": False,
    },
    {
        "variant": "top300_buffer100_current_baseline",
        "core_rank_col": "traded_value_rank_5d",
        "core_n": 300,
        "buffer_n": 100,
        "conditional_buffer": False,
    },
]


def build_design(
    *,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    audit_dir: str | Path = DEFAULT_AUDIT_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    data = Path(data_dir)
    audit = Path(audit_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    weekly = _weekly_panel(data)
    variants = _build_variants(weekly)
    summary = _period_variant_summary(variants, weekly)
    rank_basis = _rank_basis_comparison(weekly)
    buffer_policy = _buffer_policy_design()
    recommendation = _recommendation(summary, rank_basis)
    source_cost = _source_cost(summary)
    future_audit = _future_audit()
    readiness = _readiness(summary, recommendation)

    _write_csv(summary, output / "layer0_compact_variant_period_summary.csv")
    _write_csv(rank_basis, output / "layer0_compact_variant_rank_basis_comparison.csv")
    _write_csv(buffer_policy, output / "layer0_compact_buffer_eligibility_policy.csv")
    _write_csv(recommendation, output / "layer0_compact_variant_recommendation.csv")
    _write_csv(source_cost, output / "layer0_compact_variant_layer1_cost_estimate.csv")
    _write_csv(future_audit, output / "layer0_compact_variant_future_data_audit.csv")
    _write_csv(variants.head(2000), output / "layer0_compact_variant_snapshot_sample.csv")
    (output / "readiness_for_layer0_compact_variant_design.json").write_text(
        json.dumps(readiness, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    manifest = {
        "task_id": TASK_ID,
        "status": readiness["status"],
        "output_dir": str(output.resolve()),
        "input_data_dir": str(data.resolve()),
        "input_unique_count_audit_dir": str(audit.resolve()) if audit.exists() else "",
        "output_files": [
            "layer0_compact_variant_period_summary.csv",
            "layer0_compact_variant_rank_basis_comparison.csv",
            "layer0_compact_buffer_eligibility_policy.csv",
            "layer0_compact_variant_recommendation.csv",
            "layer0_compact_variant_layer1_cost_estimate.csv",
            "layer0_compact_variant_future_data_audit.csv",
            "layer0_compact_variant_snapshot_sample.csv",
            "readiness_for_layer0_compact_variant_design.json",
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


def _weekly_panel(data: Path) -> pd.DataFrame:
    calendar = pd.read_csv(data / "trading_calendar.csv")
    calendar["trade_date"] = pd.to_datetime(calendar["trade_date"])
    week_dates = set(calendar.loc[calendar["is_week_last_trading_day"].astype(bool), "trade_date"])

    cols = ["trade_date", "ticker", "name", "market", "traded_value", "valid_universe", "liquidity_flag", "listing_status"]
    df = pd.read_csv(data / "daily_market_features.csv", usecols=cols, dtype={"ticker": str})
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df["traded_value"] = pd.to_numeric(df["traded_value"], errors="coerce").fillna(0)
    df["valid_universe_bool"] = df["valid_universe"].astype(str).str.lower().eq("true")
    df["is_common_stock_like_proxy"] = df["ticker"].astype(str).str.fullmatch(r"\d{4}") & ~df["ticker"].astype(str).str.startswith("00")
    df["is_etf_or_etn_like_proxy"] = df["ticker"].astype(str).str.startswith("00")
    df["layer0_base_eligible"] = df["valid_universe_bool"] & df["is_common_stock_like_proxy"] & ~df["is_etf_or_etn_like_proxy"]
    df = df.sort_values(["ticker", "trade_date"]).reset_index(drop=True)
    grouped = df.groupby("ticker", group_keys=False)["traded_value"]
    df["traded_value_5d"] = grouped.rolling(5, min_periods=1).sum().reset_index(level=0, drop=True)
    df["traded_value_20d"] = grouped.rolling(20, min_periods=1).sum().reset_index(level=0, drop=True)
    df["traded_value_60d"] = grouped.rolling(60, min_periods=1).sum().reset_index(level=0, drop=True)
    weekly = df[df["trade_date"].isin(week_dates) & df["layer0_base_eligible"]].copy()
    weekly["snapshot_date"] = weekly["trade_date"]
    weekly["total_market_traded_value_5d"] = weekly.groupby("snapshot_date")["traded_value_5d"].transform("sum")
    for window in ["5d", "20d", "60d"]:
        weekly[f"traded_value_rank_{window}"] = weekly.groupby("snapshot_date")[f"traded_value_{window}"].rank(
            method="first", ascending=False
        )
    weekly["rank_improvement_5d_vs_60d"] = weekly["traded_value_rank_60d"] - weekly["traded_value_rank_5d"]
    return weekly.sort_values(["snapshot_date", "traded_value_rank_5d"]).reset_index(drop=True)


def _recent_top_count_mask(weekly: pd.DataFrame, rank_col: str, max_rank: int, min_count: int = 2, lookback_weeks: int = 4) -> pd.Series:
    out = pd.Series(False, index=weekly.index)
    dates = sorted(weekly["snapshot_date"].unique())
    date_order = {date: i for i, date in enumerate(dates)}
    eligible = weekly[weekly[rank_col].le(max_rank)].copy()
    eligible["week_idx"] = eligible["snapshot_date"].map(date_order)
    index_set = set(eligible.set_index(["ticker", "week_idx"]).index)
    keep = []
    for idx, row in weekly.iterrows():
        ticker = row["ticker"]
        week_idx = date_order[row["snapshot_date"]]
        count = sum((ticker, i) in index_set for i in range(max(0, week_idx - lookback_weeks + 1), week_idx + 1))
        if count >= min_count:
            keep.append(idx)
    out.loc[keep] = True
    return out


def _build_variants(weekly: pd.DataFrame) -> pd.DataFrame:
    frames = []
    weekly = weekly.copy()
    weekly["stable_5d_top400_2in4"] = _recent_top_count_mask(weekly, "traded_value_rank_5d", 400)
    for spec in VARIANTS:
        core_rank = weekly[spec["core_rank_col"]]
        core_mask = core_rank.le(spec["core_n"])
        max_rank = spec["core_n"] + spec["buffer_n"]
        if spec["buffer_n"]:
            buffer_raw = core_rank.gt(spec["core_n"]) & core_rank.le(max_rank)
            if spec["conditional_buffer"]:
                buffer_ok = buffer_raw & (
                    weekly["stable_5d_top400_2in4"]
                    | weekly["traded_value_rank_20d"].le(max_rank)
                    | weekly["traded_value_rank_60d"].le(max_rank)
                )
            else:
                buffer_ok = buffer_raw
        else:
            buffer_ok = pd.Series(False, index=weekly.index)
        selected = weekly[core_mask | buffer_ok].copy()
        selected["variant"] = spec["variant"]
        selected["selection_bucket"] = "core"
        selected.loc[buffer_ok.loc[selected.index], "selection_bucket"] = "conditional_buffer" if spec["conditional_buffer"] else "buffer"
        selected["buffer_eligible_2in4_or_20d60d_confirmed"] = buffer_ok.loc[selected.index]
        frames.append(selected)
    out = pd.concat(frames, ignore_index=True)
    out["diagnostic_only"] = True
    out["not_live_rule"] = True
    out["forward_returns_live_rule_usage"] = False
    return out


def _period_variant_summary(variants: pd.DataFrame, weekly: pd.DataFrame) -> pd.DataFrame:
    total_by_date = weekly.groupby("snapshot_date")["traded_value_5d"].sum()
    rows = []
    for period, (start, end) in {"ALL": (None, None), **PERIODS}.items():
        period_mask = _period_mask(variants, start, end)
        for variant, v in variants[period_mask].groupby("variant"):
            per_ticker_weeks = v.groupby("ticker")["snapshot_date"].nunique()
            weekly_count = v.groupby("snapshot_date")["ticker"].nunique()
            weekly_share = v.groupby("snapshot_date")["traded_value_5d"].sum() / v.groupby("snapshot_date")["snapshot_date"].first().map(total_by_date).values
            short_count = int(((per_ticker_weeks >= 1) & (per_ticker_weeks <= 4)).sum())
            rows.append(
                {
                    "period": period,
                    "variant": variant,
                    "requested_start": start or str(variants["snapshot_date"].min().date()),
                    "requested_end": end or str(variants["snapshot_date"].max().date()),
                    "actual_start": str(v["snapshot_date"].min().date()) if not v.empty else "",
                    "actual_end": str(v["snapshot_date"].max().date()) if not v.empty else "",
                    "weekly_snapshot_count": int(v["snapshot_date"].nunique()),
                    "rows": int(len(v)),
                    "avg_weekly_count": float(weekly_count.mean()) if not weekly_count.empty else 0.0,
                    "median_weekly_count": float(weekly_count.median()) if not weekly_count.empty else 0.0,
                    "unique_ticker_count": int(v["ticker"].nunique()),
                    "median_active_weeks": float(per_ticker_weeks.median()) if not per_ticker_weeks.empty else 0.0,
                    "ge52_ticker_count": int((per_ticker_weeks >= 52).sum()),
                    "ge104_ticker_count": int((per_ticker_weeks >= 104).sum()),
                    "short_1_4w_ticker_count": short_count,
                    "short_1_4w_share": float(short_count / len(per_ticker_weeks)) if len(per_ticker_weeks) else 0.0,
                    "avg_turnover_share_5d": float(weekly_share.mean()) if not weekly_share.empty else 0.0,
                    "median_turnover_share_5d": float(weekly_share.median()) if not weekly_share.empty else 0.0,
                    "estimated_layer1_period_scope_reduction_vs_1900": 1 - (float(weekly_count.mean()) / FULL_UNIVERSE_ESTIMATE)
                    if not weekly_count.empty
                    else 0.0,
                    "diagnostic_only": True,
                }
            )
    return pd.DataFrame(rows)


def _period_mask(df: pd.DataFrame, start: str | None, end: str | None) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    if start:
        mask &= df["snapshot_date"].ge(pd.Timestamp(start))
    if end:
        mask &= df["snapshot_date"].le(pd.Timestamp(end))
    return mask


def _rank_basis_comparison(weekly: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for period, (start, end) in {"P1": PERIODS["P1"], "P2": PERIODS["P2"], "2024_latest": PERIODS["2024_latest"], "2026YTD": PERIODS["2026YTD"]}.items():
        s = weekly[_period_mask(weekly, start, end)]
        total_by_date = s.groupby("snapshot_date")["traded_value_5d"].sum()
        for rank_col in ["traded_value_rank_5d", "traded_value_rank_20d", "traded_value_rank_60d"]:
            for n in [200, 250, 300, 400]:
                sub = s[s[rank_col].le(n)].copy()
                per_ticker = sub.groupby("ticker")["snapshot_date"].nunique()
                weekly_count = sub.groupby("snapshot_date")["ticker"].nunique()
                weekly_share = sub.groupby("snapshot_date")["traded_value_5d"].sum() / sub.groupby("snapshot_date")["snapshot_date"].first().map(total_by_date).values
                rows.append(
                    {
                        "period": period,
                        "rank_basis": rank_col,
                        "top_n": n,
                        "avg_weekly_count": float(weekly_count.mean()) if not weekly_count.empty else 0.0,
                        "unique_ticker_count": int(sub["ticker"].nunique()),
                        "median_active_weeks": float(per_ticker.median()) if not per_ticker.empty else 0.0,
                        "avg_turnover_share_5d": float(weekly_share.mean()) if not weekly_share.empty else 0.0,
                        "diagnostic_only": True,
                    }
                )
    return pd.DataFrame(rows)


def _buffer_policy_design() -> pd.DataFrame:
    rows = [
        (
            "conditional_buffer_2in4",
            "buffer ticker must appear within core+buffer band at least 2 times in the latest 4 weekly snapshots before Layer1 source fetch",
            "reduces one-week burst source fetch",
        ),
        (
            "conditional_buffer_20d60d_confirmed",
            "buffer ticker can qualify if 20D or 60D traded-value rank is also inside the core+buffer band",
            "keeps persistent liquidity leaders missed by 5D noise",
        ),
        (
            "5d_surge_watchlist_only",
            "5D burst without 2in4 or 20D/60D confirmation stays watchlist, not high-cost Layer1 source scope",
            "prevents t164/current-ratio source spend on one-week turnover spikes",
        ),
    ]
    return pd.DataFrame(rows, columns=["policy", "definition", "purpose"]).assign(diagnostic_only=True, formal_ready=False)


def _recommendation(summary: pd.DataFrame, rank_basis: pd.DataFrame) -> pd.DataFrame:
    p2 = summary[summary["period"].eq("P2")].set_index("variant")
    recommended = "top250_core_conditional_buffer50"
    rec = p2.loc[recommended]
    baseline = p2.loc["top300_buffer100_current_baseline"]
    rows = [
        {
            "item": "recommended_primary_candidate",
            "recommendation": recommended,
            "reason": (
                f"P2 avg weekly count={rec['avg_weekly_count']:.1f}, unique={int(rec['unique_ticker_count'])}, "
                f"avg turnover share={rec['avg_turnover_share_5d']:.4f}; tighter than baseline unique={int(baseline['unique_ticker_count'])} "
                f"while keeping weekly list inside 250-350 target."
            ),
            "diagnostic_only": True,
        },
        {
            "item": "baseline_problem",
            "recommendation": "do_not_use_top300_buffer100_as_high_cost_layer1_source_scope_without_period_scoping",
            "reason": "Current baseline P2 avg weekly count=400 and unique=1430; rolling unique remains too broad for high-cost source acquisition.",
            "diagnostic_only": True,
        },
        {
            "item": "rank_basis_policy",
            "recommendation": "use_20d_or_60d_as_stability_reference_not_sole_rule_yet",
            "reason": "20D/60D ranks reduce unique count, but this package is cost/readiness design only; no candidate-quality validation has been run.",
            "diagnostic_only": True,
        },
    ]
    return pd.DataFrame(rows)


def _source_cost(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary[["period", "variant", "avg_weekly_count", "unique_ticker_count", "estimated_layer1_period_scope_reduction_vs_1900"]].copy()
    out["full_universe_units"] = FULL_UNIVERSE_ESTIMATE
    out["cost_policy"] = "period_scoped_layer1_source_fetch_not_all_history_unique_fetch"
    out["diagnostic_only"] = True
    return out


def _future_audit() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("forward_return_as_rule", "passed", 0, "no forward returns used"),
            ("formal_rule_change", "not_applicable", 0, "variant design only"),
            ("source_scope_change", "not_applied", 0, "active Layer0 contract unchanged"),
        ],
        columns=["audit_item", "status", "future_data_violation_count", "note"],
    )


def _readiness(summary: pd.DataFrame, recommendation: pd.DataFrame) -> dict[str, Any]:
    p2 = summary[summary["period"].eq("P2")].set_index("variant")
    rec_name = recommendation.loc[recommendation["item"].eq("recommended_primary_candidate"), "recommendation"].iloc[0]
    rec = p2.loc[rec_name]
    baseline = p2.loc["top300_buffer100_current_baseline"]
    return {
        "task_id": TASK_ID,
        "status": "layer0_compact_variant_design_ready_for_strategy_center_policy_review",
        "diagnostic_only": True,
        "recommended_primary_candidate": rec_name,
        "p2_recommended_avg_weekly_count": float(rec["avg_weekly_count"]),
        "p2_recommended_unique_ticker_count": int(rec["unique_ticker_count"]),
        "p2_recommended_avg_turnover_share_5d": float(rec["avg_turnover_share_5d"]),
        "p2_baseline_avg_weekly_count": float(baseline["avg_weekly_count"]),
        "p2_baseline_unique_ticker_count": int(baseline["unique_ticker_count"]),
        "p2_baseline_avg_turnover_share_5d": float(baseline["avg_turnover_share_5d"]),
        "ready_for_layer0_policy_review": True,
        "ready_for_layer0_contract_refresh": False,
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
    return f"""# Layer0 compact variant design

## Verdict
- status={readiness["status"]}
- recommended_primary_candidate={readiness["recommended_primary_candidate"]}
- p2_recommended_avg_weekly_count={readiness["p2_recommended_avg_weekly_count"]}
- p2_recommended_unique_ticker_count={readiness["p2_recommended_unique_ticker_count"]}
- p2_recommended_avg_turnover_share_5d={readiness["p2_recommended_avg_turnover_share_5d"]}
- p2_baseline_avg_weekly_count={readiness["p2_baseline_avg_weekly_count"]}
- p2_baseline_unique_ticker_count={readiness["p2_baseline_unique_ticker_count"]}
- p2_baseline_avg_turnover_share_5d={readiness["p2_baseline_avg_turnover_share_5d"]}
- ready_for_layer0_policy_review=true
- ready_for_experiments=false
- ready_for_formal=false

## Plain Summary
The compact candidate that best matches the user's cost-control intent is top250_core_conditional_buffer50: keep the primary weekly universe around 250-300 names, allow a small buffer only when it repeats within four weeks or is confirmed by 20D/60D traded-value rank, and keep pure 5D bursts as watchlist-only. This is a Layer0 data-pruning refinement proposal, not a formal selector.

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
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--audit-dir", default=str(DEFAULT_AUDIT_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()
    manifest = build_design(data_dir=args.data_dir, audit_dir=args.audit_dir, output_dir=args.output_dir)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
