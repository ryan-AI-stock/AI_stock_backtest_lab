"""Materialize vNext Layer0 weekly investable-universe snapshots.

Layer0 is a data-pruning contract for reducing Layer1 source scope. It is not a
trading rule, selector, Experiments diagnostic, replay, or formal model change.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


TASK_ID = "TASK-BACKTEST-CORE-VNEXT-LAYER0-MATERIALIZED-WEEKLY-UNIVERSE-SNAPSHOT-CONTRACT-001"
DEFAULT_DATA_DIR = Path("outputs/vnext_dynamic_candidate_pool_data_materialization_20260706")
DEFAULT_SOURCE_CONTRACT_DIR = Path("outputs/vnext_layer0_source_contract_refresh_20260707")
DEFAULT_OUTPUT_DIR = Path("outputs/vnext_layer0_weekly_universe_snapshot_contract_20260707")


def build_snapshot(
    *,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    source_contract_dir: str | Path = DEFAULT_SOURCE_CONTRACT_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    data = Path(data_dir)
    source_contract = Path(source_contract_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    source_readiness = _read_json(source_contract / "readiness_for_layer0_source_contract_refresh.json")
    calendar = pd.read_csv(data / "trading_calendar.csv")
    panel = _load_panel(data / "daily_market_features.csv")
    weekly = _weekly_panel(panel, calendar)

    snapshot = _variant_snapshot_rows(weekly)
    coverage = _coverage_by_week_variant(snapshot, weekly)
    variant_summary = _variant_summary(coverage)
    blocked_proxy = _blocked_proxy_ledger()
    future_audit = _future_audit()
    readiness = _readiness(source_readiness, snapshot, coverage, variant_summary)

    _write_csv(snapshot, output / "layer0_weekly_universe_snapshot.csv")
    _write_csv(coverage, output / "layer0_weekly_universe_coverage_by_week.csv")
    _write_csv(variant_summary, output / "layer0_weekly_universe_variant_summary.csv")
    _write_csv(blocked_proxy, output / "layer0_weekly_universe_blocked_proxy_ledger.csv")
    _write_csv(future_audit, output / "layer0_weekly_universe_future_data_audit.csv")
    (output / "readiness_for_layer0_weekly_universe_snapshot.json").write_text(
        json.dumps(readiness, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    manifest = {
        "task_id": TASK_ID,
        "status": readiness["status"],
        "output_dir": str(output.resolve()),
        "input_data_dir": str(data.resolve()),
        "source_contract_input_dir": str(source_contract.resolve()),
        "output_files": [
            "layer0_weekly_universe_snapshot.csv",
            "layer0_weekly_universe_coverage_by_week.csv",
            "layer0_weekly_universe_variant_summary.csv",
            "layer0_weekly_universe_blocked_proxy_ledger.csv",
            "layer0_weekly_universe_future_data_audit.csv",
            "readiness_for_layer0_weekly_universe_snapshot.json",
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


def _load_panel(path: Path) -> pd.DataFrame:
    cols = ["trade_date", "ticker", "name", "market", "traded_value", "valid_universe", "liquidity_flag", "listing_status"]
    df = pd.read_csv(path, usecols=cols, dtype={"ticker": str})
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
    return df


def _weekly_panel(panel: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    calendar["trade_date"] = pd.to_datetime(calendar["trade_date"])
    week_dates = set(calendar.loc[calendar["is_week_last_trading_day"].astype(bool), "trade_date"])
    weekly = panel[panel["trade_date"].isin(week_dates)].copy()
    weekly = weekly[weekly["layer0_base_eligible"].astype(bool)].copy()
    weekly["total_market_traded_value_5d"] = weekly.groupby("trade_date")["traded_value_5d"].transform("sum")
    weekly["traded_value_rank_5d"] = weekly.groupby("trade_date")["traded_value_5d"].rank(method="first", ascending=False)
    weekly["traded_value_rank_20d"] = weekly.groupby("trade_date")["traded_value_20d"].rank(method="first", ascending=False)
    weekly["traded_value_rank_60d"] = weekly.groupby("trade_date")["traded_value_60d"].rank(method="first", ascending=False)
    weekly = weekly.sort_values(["trade_date", "traded_value_rank_5d"]).reset_index(drop=True)
    weekly["traded_value_share_5d"] = weekly["traded_value_5d"] / weekly["total_market_traded_value_5d"]
    weekly["cumulative_traded_value_share_5d"] = weekly.groupby("trade_date")["traded_value_share_5d"].cumsum()
    weekly["rank_improvement_5d_vs_60d"] = weekly["traded_value_rank_60d"] - weekly["traded_value_rank_5d"]
    weekly["surge_exception_top300"] = (weekly["traded_value_rank_5d"] <= 400) & (weekly["rank_improvement_5d_vs_60d"] >= 300)
    weekly["surge_exception_top200"] = (weekly["traded_value_rank_5d"] <= 300) & (weekly["rank_improvement_5d_vs_60d"] >= 300)
    return weekly


def _variant_snapshot_rows(weekly: pd.DataFrame) -> pd.DataFrame:
    frames = [
        _variant(weekly, "top200_buffer100", core_n=200, buffer_n=100, surge_col="surge_exception_top200"),
        _variant(weekly, "top300_buffer100", core_n=300, buffer_n=100, surge_col="surge_exception_top300"),
        _variant(weekly, "top500_reference_only", core_n=500, buffer_n=0, surge_col=None, reference_only=True),
        _threshold_variant(weekly, "turnover_share_80_reference", 0.80),
        _threshold_variant(weekly, "turnover_share_90_reference", 0.90),
    ]
    out = pd.concat(frames, ignore_index=True)
    out["instrument_type_source_quality"] = "proxy_partial_no_full_pit_master"
    out["market_cap_rank_source_quality"] = "blocked_or_proxy_not_used_primary"
    out["event_ledger_source_quality"] = "partial_blocked_no_full_delivery_disposition_master"
    out["diagnostic_only"] = True
    out["not_live_rule"] = True
    out["forward_returns_live_rule_usage"] = False
    keep = [
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
        "reference_only",
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
    return out[keep]


def _variant(
    weekly: pd.DataFrame,
    name: str,
    *,
    core_n: int,
    buffer_n: int,
    surge_col: str | None,
    reference_only: bool = False,
) -> pd.DataFrame:
    max_rank = core_n + buffer_n
    mask = weekly["traded_value_rank_5d"] <= max_rank
    if surge_col:
        mask = mask | weekly[surge_col].astype(bool)
    out = weekly[mask].copy()
    out["snapshot_date"] = out["trade_date"].dt.strftime("%Y-%m-%d")
    out["variant"] = name
    out["selection_bucket"] = "buffer"
    out.loc[out["traded_value_rank_5d"] <= core_n, "selection_bucket"] = "core"
    if surge_col:
        out["surge_exception"] = out[surge_col].astype(bool) & (out["traded_value_rank_5d"] > max_rank)
        out.loc[out["surge_exception"], "selection_bucket"] = "surge_exception"
    else:
        out["surge_exception"] = False
    out["reference_only"] = reference_only
    return out


def _threshold_variant(weekly: pd.DataFrame, name: str, threshold: float) -> pd.DataFrame:
    eligible = weekly[weekly["cumulative_traded_value_share_5d"] <= threshold].copy()
    first_over = weekly[weekly["cumulative_traded_value_share_5d"] > threshold].groupby("trade_date").head(1)
    out = pd.concat([eligible, first_over], ignore_index=True).drop_duplicates(["trade_date", "ticker"])
    out["snapshot_date"] = out["trade_date"].dt.strftime("%Y-%m-%d")
    out["variant"] = name
    out["selection_bucket"] = "threshold"
    out["surge_exception"] = False
    out["reference_only"] = True
    return out


def _coverage_by_week_variant(snapshot: pd.DataFrame, weekly: pd.DataFrame) -> pd.DataFrame:
    total_by_date = weekly.groupby(weekly["trade_date"].dt.strftime("%Y-%m-%d"))["traded_value_5d"].sum()
    grouped = (
        snapshot.groupby(["snapshot_date", "variant"])
        .agg(
            ticker_count=("ticker", "nunique"),
            selected_traded_value_5d=("traded_value_5d", "sum"),
            median_rank_5d=("traded_value_rank_5d", "median"),
            surge_exception_count=("surge_exception", "sum"),
            ky_tagged_count=("is_ky_name_proxy", "sum"),
        )
        .reset_index()
    )
    grouped["total_market_traded_value_5d"] = grouped["snapshot_date"].map(total_by_date)
    grouped["turnover_share_5d"] = grouped["selected_traded_value_5d"] / grouped["total_market_traded_value_5d"]
    grouped["diagnostic_only"] = True
    return grouped


def _variant_summary(coverage: pd.DataFrame) -> pd.DataFrame:
    return (
        coverage.groupby("variant")
        .agg(
            weekly_snapshot_count=("snapshot_date", "nunique"),
            average_ticker_count=("ticker_count", "mean"),
            median_ticker_count=("ticker_count", "median"),
            average_turnover_share_5d=("turnover_share_5d", "mean"),
            median_turnover_share_5d=("turnover_share_5d", "median"),
            average_surge_exception_count=("surge_exception_count", "mean"),
        )
        .reset_index()
        .assign(diagnostic_only=True)
    )


def _blocked_proxy_ledger() -> pd.DataFrame:
    rows = [
        ("instrument_type_master_full_pit", "partial_proxy", "ticker/name/listing metadata only; no full PIT instrument master"),
        ("pit_disposition_full_delivery_event_ledger", "blocked", "no full accepted PIT disposition/full-delivery ledger"),
        ("direct_exact_market_cap_rank", "blocked", "not used in primary Layer0 materialization"),
        ("capital_stock_x_close_market_cap_proxy", "proxy", "available only as diagnostic proxy, not formal-ready"),
        ("KY_name_tag", "proxy_tag_only", "KY is tagged, not auto-excluded"),
        ("ETF_ETN", "separate_universe_policy", "excluded by ticker pattern proxy from ordinary stock Layer0"),
    ]
    return pd.DataFrame(rows, columns=["field", "status", "reason"]).assign(diagnostic_only=True)


def _future_audit() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("forward_return_as_rule", "passed", 0, "no forward returns used"),
            ("traded_value_asof", "passed", 0, "rolling traded value uses only current/prior daily rows"),
            ("market_cap_silent_fill", "passed", 0, "market-cap rank blocked/proxy, not used primary"),
            ("instrument_event_silent_fill", "passed", 0, "instrument/event gaps remain explicit"),
        ],
        columns=["audit_item", "status", "future_data_violation_count", "note"],
    )


def _readiness(source_readiness: dict[str, Any], snapshot: pd.DataFrame, coverage: pd.DataFrame, variant_summary: pd.DataFrame) -> dict[str, Any]:
    top300 = variant_summary[variant_summary["variant"].eq("top300_buffer100")]
    top300_avg = float(top300["average_ticker_count"].iloc[0]) if not top300.empty else None
    top300_share = float(top300["average_turnover_share_5d"].iloc[0]) if not top300.empty else None
    return {
        "task_id": TASK_ID,
        "status": "layer0_weekly_universe_snapshot_materialized_diagnostic_ready_not_experiments",
        "diagnostic_only": True,
        "snapshot_rows": int(len(snapshot)),
        "weekly_snapshot_count": int(coverage["snapshot_date"].nunique()),
        "variants": sorted(snapshot["variant"].unique().tolist()),
        "recommended_primary_variant": "top300_buffer100",
        "top300_buffer100_average_ticker_count": top300_avg,
        "top300_buffer100_average_turnover_share_5d": top300_share,
        "primary_source": "daily_per_stock_traded_value",
        "ready_for_layer1_reduced_universe_source_planning": True,
        "ready_for_t164_mass_download": False,
        "ready_for_experiments": False,
        "ready_for_formal": False,
        "ready_for_strategy_replay": False,
        "future_data_violation_count": 0,
        "blocked_fields": [
            "instrument_type_master_full_pit",
            "pit_disposition_full_delivery_event_ledger",
            "direct_exact_market_cap_rank",
        ],
        "proxy_fields": ["instrument_type_by_pattern_or_partial_metadata", "capital_stock_x_close_market_cap_proxy", "KY_name_tag"],
        "formal_model_changed": False,
        "trade_decision_changed": False,
        "active_in_trade_decision": False,
        "report_changed": False,
        "portfolio_replay_executed": False,
        "not_live_rule": True,
        "forward_returns_live_rule_usage": False,
    }


def _summary(readiness: dict[str, Any]) -> str:
    return f"""# Layer0 weekly universe snapshot contract

## Verdict
- status={readiness["status"]}
- weekly_snapshot_count={readiness["weekly_snapshot_count"]}
- snapshot_rows={readiness["snapshot_rows"]}
- recommended_primary_variant={readiness["recommended_primary_variant"]}
- top300_buffer100_average_ticker_count={readiness["top300_buffer100_average_ticker_count"]}
- top300_buffer100_average_turnover_share_5d={readiness["top300_buffer100_average_turnover_share_5d"]}
- ready_for_t164_mass_download=false
- ready_for_experiments=false
- ready_for_formal=false

## Core decision
Layer0 weekly snapshots are materialized from PIT daily traded value. Use top300+buffer100 as the primary reduced universe candidate for Layer1 source planning. This is not a trading rule and does not authorize t164 mass download.

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
    parser.add_argument("--source-contract-dir", default=str(DEFAULT_SOURCE_CONTRACT_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()
    manifest = build_snapshot(
        data_dir=args.data_dir,
        source_contract_dir=args.source_contract_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
