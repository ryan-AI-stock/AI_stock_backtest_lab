"""Build the data-readiness contract for the current Layer0 base-cycle screen.

This task is intentionally a current candidate supply diagnostic.  It does not
run a portfolio, calculate forward returns, or infer unavailable Layer0 core
membership from an older snapshot.
"""

from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "TASK-BACKTEST-CORE-VNEXT-CURRENT-LAYER0-RISK-ADJUSTED-RS20-BASE-CYCLE-STABILITY-THREE-VARIANT-SCREEN-001"
DEFAULT_SNAPSHOT = (
    REPO_ROOT
    / "outputs"
    / "vnext_layer0_compact_weekly_universe_snapshot_contract_20260707"
    / "layer0_compact_weekly_universe_snapshot.csv"
)
DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "vnext_current_layer0_rs20_base_cycle_stability_three_variant_screen_20260722"
DEFAULT_SNAPSHOT_DELTA = Path(
    "C:/Users/zergv/Documents/Codex/2026-05-23/ai-stock-rotation-radar-https-docs/outputs/"
    "radar_vnext_current_layer0_core_top250_weekly_snapshot_fill_20260722/"
    "current_layer0_core_top250_weekly_snapshot_delta.csv"
)
REQUESTED_AS_OF = "2026-07-21"
WINDOW_START = "2026-03-02"
VARIANTS = {
    "V_LOOSE": {"range_pct_min": 20.0, "low_zone_max": 40.0, "high_zone_min": 60.0},
    "V_BASE": {"range_pct_min": 25.0, "low_zone_max": 35.0, "high_zone_min": 65.0},
    "V_STRICT": {"range_pct_min": 30.0, "low_zone_max": 30.0, "high_zone_min": 70.0},
}
FLAGS = {
    "formal_model_changed": False,
    "trade_decision_changed": False,
    "active_in_trade_decision": False,
    "report_changed": False,
    "portfolio_replay_executed": False,
    "ready_for_strategy_replay": False,
    "ready_for_formal": False,
    "not_live_rule": True,
    "forward_returns_live_rule_usage": False,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit current Layer0 base-cycle screen inputs.")
    parser.add_argument("--snapshot", default=str(DEFAULT_SNAPSHOT))
    parser.add_argument("--snapshot-delta", default=str(DEFAULT_SNAPSHOT_DELTA))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--requested-as-of", default=REQUESTED_AS_OF)
    args = parser.parse_args()
    build_package(Path(args.snapshot), Path(args.output_dir), args.requested_as_of, Path(args.snapshot_delta))


def build_package(
    snapshot_path: Path,
    output_dir: Path,
    requested_as_of: str,
    snapshot_delta_path: Path | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot = pd.read_csv(snapshot_path, dtype={"ticker": str})
    snapshot["snapshot_date"] = pd.to_datetime(snapshot["snapshot_date"], errors="coerce").dt.date
    snapshot["ticker"] = snapshot["ticker"].astype(str).str.zfill(4)
    requested = date.fromisoformat(requested_as_of)
    delta_ready = snapshot_delta_path is not None and snapshot_delta_path.exists()
    if delta_ready:
        delta = pd.read_csv(snapshot_delta_path, dtype={"ticker": str})
        delta = delta.rename(columns={"top250_core_rank": "core_rank"})
        delta["snapshot_date"] = pd.to_datetime(delta["snapshot_date"], errors="coerce").dt.date
        delta["ticker"] = delta["ticker"].astype(str).str.zfill(4)
        delta["selection_bucket"] = "core"
        delta["layer0_core_top250"] = True
        snapshot = pd.concat([snapshot, delta], ignore_index=True, sort=False)
        snapshot = snapshot.drop_duplicates(["snapshot_date", "ticker"], keep="last")
    actual = max(snapshot["snapshot_date"].dropna())
    required_weeks = _weekly_requirement_dates(actual, requested)
    missing_weekly = pd.DataFrame(
        {
            "requirement_type": "layer0_core_top250_weekly_snapshot",
            "snapshot_date": [d.isoformat() for d in required_weeks],
            "required_field": "top250_core_membership_from_official_PIT_traded_value",
            "scope": "full Layer0 universe only; materialize core top250, not 251-300 buffer",
            "status": "missing_source_authority",
            "reason": "local compact Layer0 snapshot ends before requested as_of",
            "radar_authority": True,
        }
    )
    existing = snapshot.loc[
        (snapshot["snapshot_date"] >= date.fromisoformat(WINDOW_START))
        & (snapshot["snapshot_date"] <= actual)
        & snapshot["selection_bucket"].eq("core"),
        ["snapshot_date", "ticker", "name", "market"],
    ].copy()
    existing["available_locally"] = True
    coverage = _coverage(snapshot, actual, requested, required_weeks)
    current_core = snapshot.loc[
        snapshot["snapshot_date"].eq(actual) & snapshot["selection_bucket"].eq("core"),
        ["snapshot_date", "ticker", "name", "market"],
    ].copy()
    price_requirements, display_requirements, turnover_requirements = _current_source_requirements(
        current_core, actual, requested
    )
    downstream = pd.DataFrame(
        [
            {
                "dependency": "adjusted_analysis_close_exact_keys",
                "status": "exact_requirement_ledger_materialized" if delta_ready else "deferred_until_missing_weekly_core_membership_is_materialized",
                "why_not_enumerated_now": "" if delta_ready else "2026-07-16 core top250 determines the as_of candidate universe; enumerating prices from 2026-06-29 membership would silently exclude new core members.",
                "required_window": f"{WINDOW_START} through {requested_as_of}",
                "raw_as_adjusted_allowed": False,
            },
            {
                "dependency": "risk_adjusted_rs20_components_as_of",
                "status": "deferred_until_missing_weekly_core_membership_is_materialized",
                "required_components": "RS20, traded-value liquidity, low_base, BIAS60 percentile, volatility percentile, Layer1 quality risk",
                "weights_frozen": "RS20=44%;liquidity=16%;low_base=18%;1-BIAS60=11%;1-volatility=6%;1-quality=5%",
            },
        ]
    )
    policy = pd.DataFrame(
        [
            {"field": "screen_name", "value": "Layer0 core risk-adjusted diagnostic screen"},
            {"field": "as_of_requested", "value": requested_as_of},
            {"field": "as_of_actual", "value": ""},
            {"field": "current_as_of_ready", "value": False},
            {"field": "window", "value": f"{WINDOW_START} to as_of"},
            {"field": "membership", "value": "layer0_core_top250=true only"},
            {"field": "weekly_coverage", "value": ">=80%; max consecutive outside <=2"},
            {"field": "alternation", "value": "low-high-low or high-low-high; each endpoint gap >=5 trading days"},
            {"field": "anomaly_guard", "value": "single abs adjusted move contribution <=35% window range; explicit corporate-action blocker otherwise"},
            {"field": "variants", "value": json.dumps(VARIANTS, ensure_ascii=False)},
            {"field": "risk_adjusted_score", "value": "frozen 44/16/18/11/6/5 with BIAS60>=95% x0.75 and volatility>=90% x0.90"},
            {"field": "performance_authorized", "value": False},
            {"field": "future_outcome_used", "value": False},
        ]
    )
    _write_csv(existing, output_dir / "local_layer0_core_coverage_through_actual_snapshot.csv")
    _write_csv(current_core, output_dir / "current_asof_layer0_core_top250.csv")
    _write_csv(missing_weekly, output_dir / "radar_exact_weekly_layer0_core_snapshot_gap_ledger.csv")
    _write_csv(price_requirements, output_dir / "radar_exact_adjusted_analysis_close_requirement_ledger.csv")
    _write_csv(display_requirements, output_dir / "radar_exact_official_raw_display_close_requirement_ledger.csv")
    _write_csv(turnover_requirements, output_dir / "radar_exact_full_market_turnover_requirement_ledger.csv")
    _write_csv(downstream, output_dir / "downstream_adjusted_price_and_feature_dependency_ledger.csv")
    _write_csv(coverage, output_dir / "requested_vs_actual_coverage.csv")
    _write_csv(policy, output_dir / "frozen_screen_policy.csv")
    _write_csv(pd.DataFrame([{"future_data_violation_count": 0, "status": "pass", "policy": "no price, score, alternation, or ranking calculation occurred beyond local snapshot cutoff"}]), output_dir / "future_data_audit.csv")

    readiness = {
        "task": TASK_ID,
        "status": "blocked_waiting_adjusted_close_and_current_liquidity_authority" if delta_ready else "blocked_waiting_exact_weekly_layer0_core_snapshot_authority",
        "requested_as_of": requested_as_of,
        "latest_local_layer0_snapshot_date": actual.isoformat(),
        "missing_weekly_snapshot_dates": [d.isoformat() for d in required_weeks],
        "latest_layer0_core_membership_absorbed": delta_ready,
        "current_core_top250_rows": int(len(current_core)),
        "adjusted_analysis_close_requirement_rows": int(len(price_requirements)),
        "official_raw_display_close_requirement_rows": int(len(display_requirements)),
        "full_market_turnover_requirement_dates": int(len(turnover_requirements)),
        "latest_official_close_as_of_ready": False,
        "adjusted_analysis_close_exact_authority_ready": False,
        "risk_adjusted_rs20_components_as_of_ready": False,
        "ready_for_current_screen": False,
        "ready_for_experiments": False,
        "performance_authorized": False,
        "future_data_violation_count": 0,
        "blocking_summary": (
            "Do not reuse 2026-07-08 risk_adjusted_rs20 output as a current screen. "
            "Current core membership is now known, but trusted adjusted analysis closes and the current full-market turnover tail are not yet absorbed. "
            "No score, alternation path, or candidate ranking has been calculated."
            if delta_ready
            else "Do not reuse 2026-07-08 risk_adjusted_rs20 output as a current screen. Missing Layer0 core snapshots for the weekly dates listed in the exact gap ledger must be supplied first; adjusted-price and feature keys are then enumerated from the resulting current core membership."
        ),
        **FLAGS,
    }
    _write_json(output_dir / "readiness_for_current_layer0_base_cycle_stability_screen.json", readiness)
    _write_json(
        output_dir / "manifest.json",
        {
            "task": TASK_ID,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "input_snapshot": str(snapshot_path),
            "artifacts": [
                "local_layer0_core_coverage_through_actual_snapshot.csv",
                "current_asof_layer0_core_top250.csv",
                "radar_exact_weekly_layer0_core_snapshot_gap_ledger.csv",
                "radar_exact_adjusted_analysis_close_requirement_ledger.csv",
                "radar_exact_official_raw_display_close_requirement_ledger.csv",
                "radar_exact_full_market_turnover_requirement_ledger.csv",
                "downstream_adjusted_price_and_feature_dependency_ledger.csv",
                "requested_vs_actual_coverage.csv",
                "frozen_screen_policy.csv",
                "future_data_audit.csv",
                "readiness_for_current_layer0_base_cycle_stability_screen.json",
            ],
            **FLAGS,
        },
    )
    (output_dir / "final_summary_zh.md").write_text(
        "# Layer0 core risk-adjusted diagnostic screen\n\n"
        + (
            f"已吸收截至 `{actual.isoformat()}` 的 Layer0 core-top250 membership。\n\n"
            "已產生 path-independent adjusted analysis close、7/21 official raw display close、以及全市場20TD turnover尾端的 exact requirement ledgers。尚未計算價格往返、分數或候選名單，未跑績效。\n"
            if delta_ready
            else f"本機 Layer0 weekly snapshot 最後日期為 `{actual.isoformat()}`，不足以代表 requested as_of `{requested_as_of}`。\n\n已產生 Radar 的 exact weekly core-top250 snapshot gap ledger。尚未計算價格往返、分數或候選名單，未跑績效。\n"
        ),
        encoding="utf-8",
    )
    return readiness


def _weekly_requirement_dates(actual: date, requested: date) -> list[date]:
    """Layer0's existing compact contract snapshots on Thursdays; 6/29 is an extra cutoff."""
    dates: list[date] = []
    cursor = actual + timedelta(days=1)
    while cursor <= requested:
        if cursor.weekday() == 3:  # Thursday, matching the existing snapshot cadence.
            dates.append(cursor)
        cursor += timedelta(days=1)
    return dates


def _coverage(snapshot: pd.DataFrame, actual: date, requested: date, missing: list[date]) -> pd.DataFrame:
    core = snapshot.loc[
        (snapshot["snapshot_date"] >= date.fromisoformat(WINDOW_START)) & snapshot["selection_bucket"].eq("core")
    ]
    return pd.DataFrame(
        [
            {"field": "requested_as_of", "requested": requested.isoformat(), "actual": "", "ready": False},
            {"field": "latest_local_layer0_snapshot", "requested": requested.isoformat(), "actual": actual.isoformat(), "ready": actual >= requested},
            {"field": "local_weekly_core_rows_since_window_start", "requested": "complete through as_of", "actual": int(len(core)), "ready": False},
            {"field": "missing_weekly_core_snapshots", "requested": "0", "actual": ";".join(d.isoformat() for d in missing), "ready": len(missing) == 0},
        ]
    )


def _current_source_requirements(
    current_core: pd.DataFrame, actual_snapshot: date, requested_as_of: date
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create source authority only after the current core membership is exact."""
    if current_core.empty or actual_snapshot < requested_as_of - timedelta(days=10):
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    price_dates = pd.bdate_range(WINDOW_START, requested_as_of).date
    stocks = current_core[["ticker", "name", "market"]].drop_duplicates().copy()
    stock_rows = stocks.merge(pd.DataFrame({"date": price_dates}), how="cross")
    stock_rows["requirement_type"] = "trusted_adjusted_analysis_close"
    stock_rows["purpose"] = "self price range, alternation, RS20, BIAS60, volatility, low-base rebuild"
    stock_rows["raw_as_adjusted_allowed"] = False
    benchmark = pd.DataFrame(
        {
            "ticker": "0050",
            "name": "元大台灣50",
            "market": "TWSE",
            "date": price_dates,
            "requirement_type": "trusted_adjusted_analysis_close",
            "purpose": "RS20 benchmark only",
            "raw_as_adjusted_allowed": False,
        }
    )
    prices = pd.concat([stock_rows, benchmark], ignore_index=True)
    prices["date"] = pd.to_datetime(prices["date"]).dt.strftime("%Y-%m-%d")
    display = stocks.copy()
    display["date"] = requested_as_of.isoformat()
    display["requirement_type"] = "official_raw_close_display_only"
    display["purpose"] = "as_of tradability/display; never adjusted analysis"
    turnover_start = requested_as_of - timedelta(days=30)
    all_dates = pd.bdate_range(turnover_start, requested_as_of).date
    turnover = pd.DataFrame({"market_date": all_dates})
    turnover = turnover[turnover["market_date"] > actual_snapshot].copy()
    turnover["market_date"] = pd.to_datetime(turnover["market_date"]).dt.strftime("%Y-%m-%d")
    turnover["requirement_type"] = "official_full_market_turnover_value"
    turnover["purpose"] = "complete current 20TD turnover rank for frozen liquidity component"
    turnover["scope"] = "TWSE and TPEx full market; ticker tie-break only after turnover sort"
    return prices, display, turnover


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
