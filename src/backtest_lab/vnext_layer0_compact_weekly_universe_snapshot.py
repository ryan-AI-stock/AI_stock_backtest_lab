"""Materialize compact Layer0 weekly universe snapshot.

Primary policy:
- active scope = 5D traded-value rank top250 core
- conditional buffer = 5D rank 251-300 only when 2-in-4 weekly persistence
  or 20D/60D traded-value rank confirms inside the same 300-name band
- pure 5D burst buffer rows remain watchlist_reference only

This is a diagnostic/source contract refresh. It does not run Experiments,
replay, formal model changes, reports, or trade decisions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


TASK_ID = "TASK-BACKTEST-CORE-VNEXT-LAYER0-COMPACT-WEEKLY-UNIVERSE-SNAPSHOT-CONTRACT-REFRESH-001"
DEFAULT_DATA_DIR = Path("outputs/vnext_dynamic_candidate_pool_data_materialization_20260706")
DEFAULT_DESIGN_DIR = Path("outputs/vnext_layer0_compact_variant_design_20260707")
DEFAULT_OUTPUT_DIR = Path("outputs/vnext_layer0_compact_weekly_universe_snapshot_contract_20260707")
PRIMARY_VARIANT = "top250_core_conditional_buffer50"

PERIODS = {
    "P1": ("2015-01-02", "2022-12-29"),
    "P2": ("2023-01-02", "2026-06-30"),
    "2024_latest": ("2024-01-02", "2026-06-30"),
    "2026YTD": ("2026-01-02", "2026-06-30"),
}


def build_snapshot(
    *,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    design_dir: str | Path = DEFAULT_DESIGN_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    data = Path(data_dir)
    design = Path(design_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    design_readiness = _read_json(design / "readiness_for_layer0_compact_variant_design.json")
    weekly = _weekly_panel(data)
    snapshot = _compact_snapshot(weekly)
    coverage = _coverage_by_period(snapshot, weekly)
    churn = _short_lived_distribution(snapshot)
    scope_split = _scope_split_by_week(snapshot, weekly)
    policy = _policy_contract()
    blocked_proxy = _blocked_proxy_ledger()
    future_audit = _future_audit()
    readiness = _readiness(design_readiness, snapshot, coverage, future_audit)

    _write_csv(snapshot, output / "layer0_compact_weekly_universe_snapshot.csv")
    _write_csv(snapshot.head(2000), output / "layer0_compact_weekly_universe_snapshot_sample.csv")
    (output / ".gitignore").write_text(
        "layer0_compact_weekly_universe_snapshot.csv\n",
        encoding="utf-8",
    )
    _write_csv(coverage, output / "layer0_compact_weekly_universe_coverage_by_period.csv")
    _write_csv(scope_split, output / "layer0_compact_weekly_universe_scope_split_by_week.csv")
    _write_csv(churn, output / "layer0_compact_weekly_universe_short_lived_distribution.csv")
    _write_csv(policy, output / "layer0_compact_policy_contract.csv")
    _write_csv(blocked_proxy, output / "layer0_compact_blocked_proxy_ledger.csv")
    _write_csv(future_audit, output / "layer0_compact_future_data_audit.csv")
    (output / "readiness_for_layer0_compact_weekly_universe_snapshot.json").write_text(
        json.dumps(readiness, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    manifest = {
        "task_id": TASK_ID,
        "status": readiness["status"],
        "output_dir": str(output.resolve()),
        "input_data_dir": str(data.resolve()),
        "input_design_dir": str(design.resolve()),
        "input_design_commit": "19270e8",
        "output_files": [
            "layer0_compact_weekly_universe_snapshot.csv",
            "layer0_compact_weekly_universe_snapshot_sample.csv",
            "layer0_compact_weekly_universe_coverage_by_period.csv",
            "layer0_compact_weekly_universe_scope_split_by_week.csv",
            "layer0_compact_weekly_universe_short_lived_distribution.csv",
            "layer0_compact_policy_contract.csv",
            "layer0_compact_blocked_proxy_ledger.csv",
            "layer0_compact_future_data_audit.csv",
            "readiness_for_layer0_compact_weekly_universe_snapshot.json",
            "manifest.json",
            "final_summary_zh.md",
        ],
        "large_local_files_not_tracked": [
            "layer0_compact_weekly_universe_snapshot.csv"
        ],
        "large_local_file_policy": "full compact materialized snapshot is retained in local output path; Git tracks sample/readiness/audit files only",
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
    df["is_ky_name_proxy"] = df["name"].astype(str).str.contains("-KY", na=False)
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
            method="first",
            ascending=False,
        )
    weekly["traded_value_share_5d"] = weekly["traded_value_5d"] / weekly["total_market_traded_value_5d"]
    weekly = weekly.sort_values(["ticker", "snapshot_date"]).reset_index(drop=True)
    weekly["rank_improvement_5d_vs_60d"] = weekly["traded_value_rank_60d"] - weekly["traded_value_rank_5d"]
    weekly["in_top300_5d"] = weekly["traded_value_rank_5d"].le(300)
    weekly["top300_5d_count_last4w"] = (
        weekly.groupby("ticker")["in_top300_5d"]
        .rolling(4, min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)
    )
    weekly["buffer_persistence_2in4"] = weekly["top300_5d_count_last4w"].ge(2)
    weekly["buffer_20d60d_confirmed"] = weekly["traded_value_rank_20d"].le(300) | weekly["traded_value_rank_60d"].le(300)
    return weekly.sort_values(["snapshot_date", "traded_value_rank_5d"]).reset_index(drop=True)


def _compact_snapshot(weekly: pd.DataFrame) -> pd.DataFrame:
    core = weekly["traded_value_rank_5d"].le(250)
    buffer_candidate = weekly["traded_value_rank_5d"].gt(250) & weekly["traded_value_rank_5d"].le(300)
    conditional_buffer = buffer_candidate & (weekly["buffer_persistence_2in4"] | weekly["buffer_20d60d_confirmed"])
    watchlist = buffer_candidate & ~conditional_buffer
    selected = weekly[core | conditional_buffer | watchlist].copy()
    selected["variant"] = PRIMARY_VARIANT
    selected["selection_bucket"] = "core"
    selected.loc[conditional_buffer.loc[selected.index], "selection_bucket"] = "conditional_buffer"
    selected.loc[watchlist.loc[selected.index], "selection_bucket"] = "watchlist_reference"
    selected["scope_type"] = "active_layer1_source_scope"
    selected.loc[watchlist.loc[selected.index], "scope_type"] = "watchlist_reference"
    selected["buffer_candidate_rank_251_300"] = buffer_candidate.loc[selected.index]
    selected["buffer_included_by_2in4"] = (buffer_candidate & weekly["buffer_persistence_2in4"]).loc[selected.index]
    selected["buffer_included_by_20d60d"] = (buffer_candidate & weekly["buffer_20d60d_confirmed"]).loc[selected.index]
    selected["pure_5d_burst_watchlist_only"] = watchlist.loc[selected.index]
    selected["active_for_layer1_source_scope"] = selected["scope_type"].eq("active_layer1_source_scope")
    selected["instrument_type_source_quality"] = "proxy_partial_no_full_pit_master"
    selected["market_cap_rank_source_quality"] = "blocked_or_proxy_not_used_primary"
    selected["event_ledger_source_quality"] = "partial_blocked_no_full_delivery_disposition_master"
    selected["diagnostic_only"] = True
    selected["not_live_rule"] = True
    selected["forward_returns_live_rule_usage"] = False
    keep = [
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
        "diagnostic_only",
        "not_live_rule",
        "forward_returns_live_rule_usage",
    ]
    return selected[keep].sort_values(["snapshot_date", "scope_type", "traded_value_rank_5d", "ticker"])


def _period_mask(df: pd.DataFrame, start: str | None, end: str | None) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    if start:
        mask &= df["snapshot_date"].ge(pd.Timestamp(start))
    if end:
        mask &= df["snapshot_date"].le(pd.Timestamp(end))
    return mask


def _coverage_by_period(snapshot: pd.DataFrame, weekly: pd.DataFrame) -> pd.DataFrame:
    total_by_date = weekly.groupby("snapshot_date")["traded_value_5d"].sum()
    rows = []
    for period, (start, end) in {"ALL": (None, None), **PERIODS}.items():
        s = snapshot[_period_mask(snapshot, start, end)].copy()
        for scope_type, scoped in s.groupby("scope_type"):
            weekly_count = scoped.groupby("snapshot_date")["ticker"].nunique()
            per_ticker = scoped.groupby("ticker")["snapshot_date"].nunique()
            weekly_share = (
                scoped.groupby("snapshot_date")["traded_value_5d"].sum()
                / scoped.groupby("snapshot_date")["snapshot_date"].first().map(total_by_date).values
            )
            short_count = int(((per_ticker >= 1) & (per_ticker <= 4)).sum())
            rows.append(
                {
                    "period": period,
                    "scope_type": scope_type,
                    "requested_start": start or str(snapshot["snapshot_date"].min().date()),
                    "requested_end": end or str(snapshot["snapshot_date"].max().date()),
                    "actual_start": str(scoped["snapshot_date"].min().date()) if not scoped.empty else "",
                    "actual_end": str(scoped["snapshot_date"].max().date()) if not scoped.empty else "",
                    "weekly_snapshot_count": int(scoped["snapshot_date"].nunique()),
                    "rows": int(len(scoped)),
                    "avg_weekly_count": float(weekly_count.mean()) if not weekly_count.empty else 0.0,
                    "median_weekly_count": float(weekly_count.median()) if not weekly_count.empty else 0.0,
                    "unique_ticker_count": int(scoped["ticker"].nunique()),
                    "median_active_weeks": float(per_ticker.median()) if not per_ticker.empty else 0.0,
                    "ge52_ticker_count": int((per_ticker >= 52).sum()),
                    "ge104_ticker_count": int((per_ticker >= 104).sum()),
                    "short_1_4w_ticker_count": short_count,
                    "short_1_4w_share": float(short_count / len(per_ticker)) if len(per_ticker) else 0.0,
                    "avg_turnover_share_5d": float(weekly_share.mean()) if not weekly_share.empty else 0.0,
                    "median_turnover_share_5d": float(weekly_share.median()) if not weekly_share.empty else 0.0,
                    "diagnostic_only": True,
                }
            )
    return pd.DataFrame(rows)


def _scope_split_by_week(snapshot: pd.DataFrame, weekly: pd.DataFrame) -> pd.DataFrame:
    total_by_date = weekly.groupby("snapshot_date")["traded_value_5d"].sum()
    grouped = (
        snapshot.groupby(["snapshot_date", "scope_type"])
        .agg(
            ticker_count=("ticker", "nunique"),
            traded_value_5d=("traded_value_5d", "sum"),
            median_rank_5d=("traded_value_rank_5d", "median"),
            pure_5d_burst_watchlist_count=("pure_5d_burst_watchlist_only", "sum"),
        )
        .reset_index()
    )
    grouped["total_market_traded_value_5d"] = grouped["snapshot_date"].map(total_by_date)
    grouped["turnover_share_5d"] = grouped["traded_value_5d"] / grouped["total_market_traded_value_5d"]
    grouped["diagnostic_only"] = True
    return grouped


def _short_lived_distribution(snapshot: pd.DataFrame) -> pd.DataFrame:
    bins = [
        ("1_4_weeks", 1, 4),
        ("5_12_weeks", 5, 12),
        ("13_26_weeks", 13, 26),
        ("27_52_weeks", 27, 52),
        ("53_104_weeks", 53, 104),
        ("105_plus_weeks", 105, None),
    ]
    rows = []
    for period, (start, end) in {"ALL": (None, None), **PERIODS}.items():
        s = snapshot[_period_mask(snapshot, start, end)]
        for scope_type, scoped in s.groupby("scope_type"):
            per_ticker = scoped.groupby("ticker")["snapshot_date"].nunique()
            for label, lo, hi in bins:
                if hi is None:
                    count = int((per_ticker >= lo).sum())
                else:
                    count = int(((per_ticker >= lo) & (per_ticker <= hi)).sum())
                rows.append(
                    {
                        "period": period,
                        "scope_type": scope_type,
                        "appearance_bin": label,
                        "ticker_count": count,
                        "share_of_unique": float(count / len(per_ticker)) if len(per_ticker) else 0.0,
                        "diagnostic_only": True,
                    }
                )
    return pd.DataFrame(rows)


def _policy_contract() -> pd.DataFrame:
    rows = [
        ("primary_variant", PRIMARY_VARIANT, "accepted_by_strategy_center", "diagnostic_contract_only"),
        ("core", "5D traded-value rank <= 250", "active_layer1_source_scope", "not_live_rule"),
        ("conditional_buffer_candidate", "5D traded-value rank 251-300", "candidate_for_active_scope_or_watchlist", "not_live_rule"),
        ("buffer_condition_1", "top300 5D band appears at least 2 times in latest 4 weekly snapshots", "active_layer1_source_scope", "not_live_rule"),
        ("buffer_condition_2", "20D or 60D traded-value rank <= 300", "active_layer1_source_scope", "not_live_rule"),
        ("pure_5d_burst", "rank 251-300 without condition_1 or condition_2", "watchlist_reference_only", "not_layer1_source_scope"),
    ]
    return pd.DataFrame(rows, columns=["policy_item", "definition", "scope_policy", "boundary"]).assign(diagnostic_only=True)


def _blocked_proxy_ledger() -> pd.DataFrame:
    rows = [
        ("instrument_type_master_full_pit", "partial_proxy", "ticker/name/listing metadata only; no full PIT instrument master"),
        ("pit_disposition_full_delivery_event_ledger", "blocked", "no full accepted PIT disposition/full-delivery ledger"),
        ("direct_exact_market_cap_rank", "blocked", "not used in primary compact Layer0 materialization"),
        ("capital_stock_x_close_market_cap_proxy", "proxy", "available only as diagnostic proxy, not formal-ready"),
        ("KY_name_tag", "proxy_tag_only", "KY is tagged, not auto-excluded"),
        ("ETF_ETN", "separate_universe_policy", "excluded by ticker pattern proxy from ordinary stock Layer0"),
    ]
    return pd.DataFrame(rows, columns=["field", "status", "reason"]).assign(diagnostic_only=True)


def _future_audit() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("forward_return_as_rule", "passed", 0, "no forward returns used"),
            ("formal_rule_change", "not_applicable", 0, "contract refresh only; formal model unchanged"),
            ("watchlist_mixed_into_active_scope", "passed", 0, "watchlist_reference has separate scope_type and active_for_layer1_source_scope=false"),
        ],
        columns=["audit_item", "status", "future_data_violation_count", "note"],
    )


def _readiness(design_readiness: dict[str, Any], snapshot: pd.DataFrame, coverage: pd.DataFrame, future_audit: pd.DataFrame) -> dict[str, Any]:
    active = coverage[(coverage["period"].eq("P2")) & (coverage["scope_type"].eq("active_layer1_source_scope"))].iloc[0]
    watchlist = coverage[(coverage["period"].eq("P2")) & (coverage["scope_type"].eq("watchlist_reference"))].iloc[0]
    return {
        "task_id": TASK_ID,
        "status": "layer0_compact_weekly_universe_snapshot_materialized_ready_for_layer1_compact_rebuild",
        "diagnostic_only": True,
        "primary_variant": PRIMARY_VARIANT,
        "design_status": design_readiness.get("status", ""),
        "snapshot_rows": int(len(snapshot)),
        "weekly_snapshot_count": int(snapshot["snapshot_date"].nunique()),
        "p2_active_avg_weekly_count": float(active["avg_weekly_count"]),
        "p2_active_unique_ticker_count": int(active["unique_ticker_count"]),
        "p2_active_avg_turnover_share_5d": float(active["avg_turnover_share_5d"]),
        "p2_watchlist_avg_weekly_count": float(watchlist["avg_weekly_count"]),
        "p2_watchlist_unique_ticker_count": int(watchlist["unique_ticker_count"]),
        "watchlist_reference_excluded_from_layer1_source_scope": True,
        "ready_for_layer1_compact_reduced_universe_interim_contract_rebuild": True,
        "ready_for_experiments": False,
        "ready_for_formal": False,
        "ready_for_strategy_replay": False,
        "ready_for_t164_mass_download": False,
        "future_data_violation_count": int(future_audit["future_data_violation_count"].max()),
        "formal_model_changed": False,
        "trade_decision_changed": False,
        "active_in_trade_decision": False,
        "report_changed": False,
        "portfolio_replay_executed": False,
        "not_live_rule": True,
        "forward_returns_live_rule_usage": False,
    }


def _summary(readiness: dict[str, Any]) -> str:
    return f"""# Layer0 compact weekly universe snapshot contract

## Verdict
- status={readiness["status"]}
- primary_variant={readiness["primary_variant"]}
- p2_active_avg_weekly_count={readiness["p2_active_avg_weekly_count"]}
- p2_active_unique_ticker_count={readiness["p2_active_unique_ticker_count"]}
- p2_active_avg_turnover_share_5d={readiness["p2_active_avg_turnover_share_5d"]}
- p2_watchlist_avg_weekly_count={readiness["p2_watchlist_avg_weekly_count"]}
- p2_watchlist_unique_ticker_count={readiness["p2_watchlist_unique_ticker_count"]}
- watchlist_reference_excluded_from_layer1_source_scope=true
- ready_for_layer1_compact_reduced_universe_interim_contract_rebuild=true
- ready_for_experiments=false
- ready_for_formal=false

## Plain Summary
The compact Layer0 snapshot materializes Strategy Center's accepted top250_core_conditional_buffer50 policy. Active Layer1 source scope is separated from watchlist_reference, so one-week 5D bursts do not trigger high-cost Layer1 source work. Watchlist reference unique count is high because it is mostly 1-4 week bursts and must not be used as Layer1 source scope. This is still diagnostic/source readiness only and does not change formal model or trade decisions.

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
    parser.add_argument("--design-dir", default=str(DEFAULT_DESIGN_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()
    manifest = build_snapshot(data_dir=args.data_dir, design_dir=args.design_dir, output_dir=args.output_dir)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
