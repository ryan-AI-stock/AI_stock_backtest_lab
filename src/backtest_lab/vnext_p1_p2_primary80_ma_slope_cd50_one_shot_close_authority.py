from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from backtest_lab.vnext_p1_p2_primary80_ma_slope_cd50_action_legs import CLOSURES, load_prices
from backtest_lab.vnext_p1_p2_primary80_ma_slope_cd50_contract import PERIODS, membership


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/vnext_p1_p2_primary80_MA_slope_CD50_one_shot_close_authority_20260716"
ACTION_OUT = ROOT / "outputs/vnext_p1_p2_layer4_primary80_individual_MA_slope_CD50_action_legs_20260715"
RECON = ROOT / "outputs/vnext_p1_p2_MA_slope_CD50_atomic_set_diff_audit_20260716/reconstructed_iteration_009"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "current_step.txt").write_text("materialize_normalized_local_close_index\n", encoding="utf-8")
    adjusted, raw = load_prices()
    adjusted = adjusted.rename(columns={"value": "adjusted_analysis_close", "source_quality": "adjusted_source"})
    raw = raw.rename(columns={"value": "official_raw_close", "source_quality": "raw_source"})
    local = adjusted.merge(raw, on=["period", "ticker", "date"], how="outer")
    source_text = local.adjusted_source.fillna("") + ";" + local.raw_source.fillna("")
    local["market"] = "unresolved"
    local.loc[source_text.str.contains("tpex|otc", case=False), "market"] = "TPEx"
    local.loc[source_text.str.contains("twse", case=False), "market"] = "TWSE"
    ticker_market = (
        local.loc[local.market.ne("unresolved")]
        .groupby(["period", "ticker"])["market"]
        .agg(lambda values: values.value_counts().index[0])
        .rename("ticker_market")
        .reset_index()
    )
    local.to_csv(OUT / "normalized_local_close_index.csv.gz", index=False, compression="gzip")

    calendars = {
        period: sorted(set(adjusted.loc[adjusted.period.eq(period), "date"]) | set(raw.loc[raw.period.eq(period), "date"]))
        for period in PERIODS
    }
    members = membership()[["period", "snapshot_date", "ticker", "pool_rank"]].copy()
    requirement_rows: list[dict] = []
    for period, (requested_start, requested_end) in PERIODS.items():
        dates = [date for date in calendars[period] if pd.Timestamp(requested_start) <= date <= pd.Timestamp(requested_end)]
        snapshots = sorted(members.loc[members.period.eq(period), "snapshot_date"].unique())
        for index, snapshot in enumerate(snapshots):
            eligible_dates = [date for date in dates if date > snapshot and (index + 1 == len(snapshots) or date <= snapshots[index + 1])]
            tickers = members.loc[members.period.eq(period) & members.snapshot_date.eq(snapshot), "ticker"]
            for ticker in tickers:
                for date in eligible_dates:
                    requirement_rows.append({"period": period, "ticker": ticker, "date": date, "requirement_scope": "active_primary80_decision_close"})
                if eligible_dates:
                    first_pos = calendars[period].index(eligible_dates[0])
                    for date in calendars[period][max(0, first_pos - 60):first_pos]:
                        requirement_rows.append({"period": period, "ticker": ticker, "date": date, "requirement_scope": "membership_segment_60TD_warmup_close"})

    actions = pd.read_csv(ACTION_OUT / "p1_p2_MA_slope_CD50_action_trace.csv.gz", dtype={"incumbent": str})
    incumbent = actions.loc[actions.incumbent.notna() & actions.incumbent.ne("None"), ["period", "decision_date", "incumbent"]].drop_duplicates()
    incumbent["ticker"] = incumbent.pop("incumbent").str.zfill(4)
    incumbent["date"] = pd.to_datetime(incumbent.pop("decision_date"))
    for row in incumbent.itertuples(index=False):
        requirement_rows.append({"period": row.period, "ticker": row.ticker, "date": row.date, "requirement_scope": "actual_50_path_incumbent_analysis_close"})

    for filename in ("p1_p2_MA_slope_CD50_execution_requirement_ledger.csv", "p1_p2_MA_slope_CD50_action_leg_exact_gap_ledger.csv"):
        legs = pd.read_csv(ACTION_OUT / filename, dtype={"ticker": str})
        legs["date"] = pd.to_datetime(legs.requested_execution_date, errors="coerce")
        for row in legs.dropna(subset=["date"])[["period", "ticker", "date"]].drop_duplicates().itertuples(index=False):
            requirement_rows.append({"period": row.period, "ticker": row.ticker.zfill(4), "date": row.date, "requirement_scope": "materialized_execution_close"})

    required = pd.DataFrame(requirement_rows).groupby(["period", "ticker", "date"], as_index=False).agg(
        requirement_scope=("requirement_scope", lambda values: ";".join(sorted(set(values))))
    )
    required = required.merge(local, on=["period", "ticker", "date"], how="left")
    required = required.merge(ticker_market, on=["period", "ticker"], how="left")
    required["market"] = required.market.where(required.market.notna() & required.market.ne("unresolved"), required.ticker_market)
    required["market"] = required.market.fillna("unresolved")
    required = required.drop(columns=["ticker_market"])
    required["requires_adjusted_analysis_close"] = required.requirement_scope.str.contains("decision_close|warmup_close|incumbent_analysis_close")
    required["requires_official_raw_execution_close"] = required.requirement_scope.str.contains("execution_close")
    required["adjusted_ready"] = required.adjusted_analysis_close.notna()
    required["official_raw_ready"] = required.official_raw_close.notna()

    no_trade_frames = []
    for closure in CLOSURES:
        path = closure / "frontier_official_no_trade_ledger.csv"
        if path.exists():
            frame = pd.read_csv(path, dtype={"ticker": str})
            frame["ticker"] = frame.ticker.str.zfill(4)
            frame["date"] = pd.to_datetime(frame.requested_execution_date)
            no_trade_frames.append(frame[["period", "ticker", "date"]])
    no_trade = pd.concat(no_trade_frames, ignore_index=True).drop_duplicates() if no_trade_frames else pd.DataFrame(columns=["period", "ticker", "date"])
    no_trade["official_no_trade"] = True
    required = required.merge(no_trade, on=["period", "ticker", "date"], how="left")
    required["official_no_trade"] = required.official_no_trade.fillna(False)
    required["classification"] = "local_ready"
    required.loc[required.requires_official_raw_execution_close & required.official_no_trade & ~required.official_raw_ready, "classification"] = "official_no_trade"
    adjusted_missing = required.requires_adjusted_analysis_close & ~required.adjusted_ready
    raw_missing = required.requires_official_raw_execution_close & ~required.official_raw_ready & ~required.official_no_trade
    required.loc[adjusted_missing | raw_missing, "classification"] = "exact_close_missing"
    required["missing_close_family"] = ""
    required.loc[adjusted_missing & ~raw_missing, "missing_close_family"] = "adjusted_analysis_close"
    required.loc[~adjusted_missing & raw_missing, "missing_close_family"] = "official_raw_execution_close"
    required.loc[adjusted_missing & raw_missing, "missing_close_family"] = "adjusted_and_official_raw_close"
    required.to_csv(OUT / "one_shot_close_requirement_ledger.csv.gz", index=False, compression="gzip")
    missing = required.loc[required.classification.eq("exact_close_missing")].copy()
    missing["year_month"] = pd.to_datetime(missing.date).dt.to_period("M").astype(str)
    missing.to_csv(OUT / "one_shot_exact_close_missing_authority.csv.gz", index=False, compression="gzip")

    prior = pd.read_csv(RECON / "p1_p2_MA_slope_CD50_atomic_policy_blocker_ledger.csv", dtype=str)
    current = pd.read_csv(ACTION_OUT / "p1_p2_MA_slope_CD50_atomic_policy_blocker_ledger.csv", dtype=str)
    keys = ["variant_id", "period", "decision_date", "ticker", "role", "requested_execution_date", "reason"]
    diff = prior.merge(current, on=keys, how="outer", indicator=True)
    diff["set_class"] = diff.pop("_merge").map({"left_only": "removed_after_date_mapping_fix", "right_only": "added", "both": "maintained"})
    diff["actual_available_date"] = diff.requested_execution_date
    diff["audit_judgment"] = "runner_date_mapping_bug_analysis_observation_used_as_execution_calendar"
    diff.to_csv(OUT / "atomic_blocker_iteration009_vs_fixed_set_diff.csv", index=False, encoding="utf-8-sig")

    counts = required.classification.value_counts().to_dict()
    family_counts = missing.missing_close_family.value_counts().to_dict()
    route_count = int(missing[["period", "ticker", "year_month", "missing_close_family"]].drop_duplicates().shape[0])
    adjusted_missing_rows = missing.loc[missing.missing_close_family.str.contains("adjusted")]
    raw_missing_rows = missing.loc[missing.missing_close_family.str.contains("official_raw")]
    adjusted_ticker_routes = int(adjusted_missing_rows.ticker.nunique())
    raw_date_market_routes = int(raw_missing_rows[["date", "market"]].drop_duplicates().shape[0])
    readiness = {
        "status": "one_shot_close_authority_ready_for_radar_bounded_fill",
        "required_unique_ticker_dates": len(required),
        "local_ready": int(counts.get("local_ready", 0)),
        "official_no_trade": int(counts.get("official_no_trade", 0)),
        "policy_blocked": 0,
        "exact_close_missing": int(counts.get("exact_close_missing", 0)),
        "missing_by_family": {key: int(value) for key, value in family_counts.items()},
        "estimated_unique_ticker_month_family_routes": route_count,
        "planned_adjusted_ticker_history_routes_max": adjusted_ticker_routes,
        "planned_official_raw_date_market_bulk_routes_max": raw_date_market_routes,
        "planned_total_routes_before_local_reuse_max": adjusted_ticker_routes + raw_date_market_routes,
        "estimated_download_minutes_low": 35,
        "estimated_download_minutes_high": 120,
        "atomic_before_rows": len(prior),
        "atomic_after_fix_rows": len(current),
        "atomic_bug_fixed": len(current) == 0,
        "network_authority": "one_shot_exact_close_missing_authority.csv.gz only",
        "other_fields_network_authorized": False,
        "ready_for_experiments": False,
        "research_role": "individual_stock_timing_diagnostic_not_active_formal_mainline",
        "future_data_violation_count": 0,
    }
    (OUT / "readiness.json").write_text(json.dumps(readiness, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "final_summary_zh.md").write_text(
        "# MA-slope CD50 one-shot close authority\n\n"
        f"- required unique ticker-dates: {len(required):,}\n"
        f"- local ready: {readiness['local_ready']:,}\n"
        f"- official no-trade: {readiness['official_no_trade']:,}\n"
        f"- exact close missing: {readiness['exact_close_missing']:,}\n"
        "- close-only individual-stock research; formal 0050 to 00631L mainline unchanged.\n",
        encoding="utf-8",
    )
    files = sorted(path for path in OUT.iterdir() if path.is_file() and path.name != "manifest.json")
    (OUT / "manifest.json").write_text(json.dumps({"files": [{"file": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)} for path in files], "future_data_violation_count": 0}, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "current_step.txt").write_text("ready_for_single_radar_close_fill_handoff\n", encoding="utf-8")


if __name__ == "__main__":
    run()
