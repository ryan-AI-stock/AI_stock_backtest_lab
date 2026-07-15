from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/vnext_p1_p2_layer4_primary80_individual_MA_slope_CD50_contract_20260715"
POOL = ROOT / "outputs/vnext_layer4_80_primary_pool_contract_20260708/layer4_80_primary_pool_contract.csv"
P1_SOURCE = Path(r"C:\Users\zergv\Documents\Codex\2026-05-23\ai-stock-rotation-radar-https-docs\outputs\radar_vnext_p1_full_lifecycle_minimum_data_acquisition_20260710")
P3_SOURCE = Path(r"C:\Users\zergv\Documents\Codex\2026-05-23\ai-stock-rotation-radar-https-docs\outputs\radar_vnext_p3_layer5_all80_continuous_lifecycle_adjusted_hlc_bounded_delta_acquisition_20260713")
TASK = "TASK-BACKTEST-CORE-VNEXT-P1-P2-LAYER4-PRIMARY80-INDIVIDUAL-MA-SLOPE-CD50-CONTRACT-001"
PERIODS = {
    "P1": (pd.Timestamp("2015-01-02"), pd.Timestamp("2022-12-29")),
    "P2": (pd.Timestamp("2023-01-02"), pd.Timestamp("2026-06-30")),
}
SIGNALS = {
    "S01": (4, 5, 7, 7), "S02": (4, 7, 7, 10), "S03": (4, 7, 10, 10),
    "S04": (4, 7, 10, 20), "S05": (4, 10, 10, 20), "S06": (7, 7, 10, 10),
    "S07": (7, 7, 10, 20), "S08": (7, 10, 10, 20), "S09": (10, 10, 15, 20),
    "S10": (10, 20, 20, 20),
}
COOLDOWNS = (2, 3, 5, 7, 10)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parameter_matrix() -> pd.DataFrame:
    rows = []
    for signal, (entry_ma, entry_slope, exit_ma, exit_slope) in SIGNALS.items():
        for cooldown in COOLDOWNS:
            rows.append({
                "variant_id": f"{signal}_CD{cooldown}", "signal_family": signal,
                "entry_ma_window": entry_ma, "entry_slope_window": entry_slope,
                "exit_ma_window": exit_ma, "exit_slope_window": exit_slope,
                "post_buy_exit_lock_trading_days": cooldown,
                "buy_rule": f"close>MA{entry_ma} and slope{entry_slope}>0",
                "sell_rule": f"close<MA{exit_ma} and slope{exit_slope}<0",
                "post_sell_reentry_cd": 0, "slippage_bp_primary": 10,
                "slippage_bp_sensitivity": "5|20", "market_controller_used": False,
            })
    return pd.DataFrame(rows)


def membership() -> pd.DataFrame:
    use = ["snapshot_date", "ticker", "pool_rank", "is_layer4_primary_pool"]
    frame = pd.read_csv(POOL, usecols=use, dtype={"ticker": str})
    frame["snapshot_date"] = pd.to_datetime(frame.snapshot_date)
    frame["ticker"] = frame.ticker.str.zfill(4)
    frame = frame.loc[frame.is_layer4_primary_pool.astype(str).str.lower().eq("true")].copy()
    frame["period"] = ""
    for period, (start, end) in PERIODS.items():
        frame.loc[frame.snapshot_date.between(start - pd.Timedelta(days=7), end), "period"] = period
    return frame.loc[frame.period.ne("")].sort_values(["snapshot_date", "pool_rank"])


def requirement_ledger(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (period, ticker), group in frame.groupby(["period", "ticker"], sort=True):
        requested_start, requested_end = PERIODS[period]
        first, last = group.snapshot_date.min(), group.snapshot_date.max()
        rows.append({
            "period": period, "ticker": ticker, "first_snapshot_date": first,
            "last_snapshot_date": last, "snapshot_count": group.snapshot_date.nunique(),
            "adjusted_analysis_required_start": max(requested_start - pd.Timedelta(days=45), first - pd.Timedelta(days=90)),
            "adjusted_analysis_required_end": requested_end,
            "official_raw_execution_required_start": requested_start,
            "official_raw_execution_required_end": requested_end,
            "required_fields": "adjusted_close|official_raw_close",
        })
    return pd.DataFrame(rows)


def source_audit(requirements: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    p1_manifest_path = P1_SOURCE / "trusted_adjusted_analysis_manifest.csv"
    p1_manifest = pd.read_csv(p1_manifest_path, dtype={"ticker": str}) if p1_manifest_path.exists() else pd.DataFrame()
    if len(p1_manifest):
        p1_manifest["ticker"] = p1_manifest.ticker.str.zfill(4)
        p1_status = p1_manifest.drop_duplicates("ticker", keep="last").set_index("ticker")["status"].to_dict()
    else:
        p1_status = {}
    rows, gaps = [], []
    for r in requirements.itertuples(index=False):
        if r.period == "P1":
            adjusted = "ready_local_trusted_reuse" if p1_status.get(r.ticker) == "accepted" else "blocked_or_missing_trusted_adjusted"
            raw = "ready_local_official_bulk_compact" if (P1_SOURCE / "compact/official_raw_execution_ohlcv").exists() else "source_audit_required"
        else:
            adjusted = "partial_local_P3_reuse_bridge_and_tail_audit_required"
            raw = "partial_local_P3_reuse_bridge_and_tail_audit_required"
        ready = adjusted.startswith("ready") and raw.startswith("ready")
        rows.append({"period": r.period, "ticker": r.ticker, "adjusted_analysis_status": adjusted, "official_raw_execution_status": raw, "exact_path_ready": ready})
        if not ready:
            gaps.append({
                "period": r.period, "ticker": r.ticker,
                "requested_start": r.adjusted_analysis_required_start,
                "requested_end": r.adjusted_analysis_required_end,
                "missing_or_audit_families": "adjusted_analysis_close|official_raw_execution_close" if r.period == "P2" else "adjusted_analysis_close",
                "scope_policy": "bounded primary80 ticker-period plus MA20/60TD percentile warmup; reuse local exact keys first",
                "source_probe_authorized_by_core": False,
            })
    return pd.DataFrame(rows), pd.DataFrame(gaps)


def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "current_step.txt").write_text("prepare_contract_and_source_audit\n", encoding="utf-8")
    matrix = parameter_matrix()
    members = membership()
    requirements = requirement_ledger(members)
    coverage, gaps = source_audit(requirements)
    matrix.to_csv(OUT / "p1_p2_primary80_MA_slope_CD50_parameter_matrix.csv", index=False, encoding="utf-8-sig")
    members.to_csv(OUT / "p1_p2_primary80_weekly_membership_contract.csv.gz", index=False, compression="gzip")
    requirements.to_csv(OUT / "p1_p2_primary80_ticker_period_price_requirement_ledger.csv", index=False, encoding="utf-8-sig")
    coverage.to_csv(OUT / "p1_p2_primary80_local_source_reuse_coverage.csv", index=False, encoding="utf-8-sig")
    gaps.to_csv(OUT / "p1_p2_primary80_bounded_source_gap_ledger.csv", index=False, encoding="utf-8-sig")
    summary = members.groupby("period").agg(
        snapshots=("snapshot_date", "nunique"), membership_rows=("ticker", "size"),
        unique_tickers=("ticker", "nunique"), actual_snapshot_start=("snapshot_date", "min"),
        actual_snapshot_end=("snapshot_date", "max"),
    ).reset_index()
    summary["requested_start"] = summary.period.map({k: v[0] for k, v in PERIODS.items()})
    summary["requested_end"] = summary.period.map({k: v[1] for k, v in PERIODS.items()})
    summary.to_csv(OUT / "p1_p2_requested_vs_actual_membership_coverage.csv", index=False, encoding="utf-8-sig")
    policy = pd.DataFrame([
        {"rule": "membership_effective", "value": "snapshot close list becomes new-buy eligible next official trading day"},
        {"rule": "candidate_rank", "value": "price_pct60 asc; distance_above_entry_MA_pct asc; normalized_positive_slope desc; pool_rank asc; ticker asc"},
        {"rule": "incumbent", "value": "no challenger switch; pool exit does not force sell; own exit signal only"},
        {"rule": "execution", "value": "signal close PIT; next ticker trading-day official raw close; no neighbor substitution"},
        {"rule": "portfolio", "value": "P1/P2 each reset TWD 1000000; integer shares; corrected NAV; cash residual"},
        {"rule": "cost", "value": "EP05 stock fee and tax plus 10bp per side primary; 5/20bp sensitivity"},
        {"rule": "forbidden", "value": "future outcome|market controller|TAIFEX|TDCC|external market|00631L fallback|variant 51"},
    ])
    policy.to_csv(OUT / "p1_p2_primary80_MA_slope_execution_policy.csv", index=False, encoding="utf-8-sig")
    ready = gaps.empty
    readiness = {
        "task_id": TASK, "status": "ready_for_experiments" if ready else "bounded_source_gap_requires_radar",
        "fixed_variant_count": len(matrix), "membership_rows": len(members),
        "unique_tickers": int(members.ticker.nunique()), "source_gap_ticker_period_rows": len(gaps),
        "daily_features_materialized": False, "execution_requirement_materialized": False,
        "ready_for_experiments": ready, "formal_model_changed": False,
        "trade_decision_changed": False, "active_in_trade_decision": False,
        "report_changed": False, "not_live_rule": True, "future_data_violation_count": 0,
    }
    (OUT / "readiness_for_p1_p2_primary80_MA_slope_CD50.json").write_text(json.dumps(readiness, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "final_summary_zh.md").write_text(
        f"# P1/P2 primary80 MA/slope CD50 contract\n\n- fixed variants: {len(matrix)}\n- membership rows: {len(members)}\n- unique tickers: {members.ticker.nunique()}\n- bounded source gap ticker-period rows: {len(gaps)}\n- ready_for_experiments: {str(ready).lower()}\n- No market controller, future-return rule, or performance run.\n",
        encoding="utf-8",
    )
    files = sorted(p for p in OUT.iterdir() if p.is_file() and p.name != "manifest.json")
    manifest = {"task_id": TASK, "pool_sha256": _sha(POOL), "files": [{"name": p.name, "sha256": _sha(p), "bytes": p.stat().st_size} for p in files]}
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "current_step.txt").write_text("waiting_for_bounded_price_source_package\n", encoding="utf-8")


if __name__ == "__main__":
    run()
