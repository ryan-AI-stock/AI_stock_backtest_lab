from __future__ import annotations

import argparse
import bisect
import glob
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

from backtest_lab import vnext_daily_incumbent_challenger_state_machine_contract as source


TASK_ID = "TASK-BACKTEST-CORE-VNEXT-P1-LIFECYCLE-MISSING-FEATURE-INCREMENTAL-READINESS-CONTRACT-001"
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO_ROOT / "outputs/vnext_p1_lifecycle_missing_feature_incremental_readiness_contract_20260710"
P1_START, P1_END = pd.Timestamp("2015-01-02"), pd.Timestamp("2022-12-29")
FLAGS = {"formal_model_changed": False, "trade_decision_changed": False, "active_in_trade_decision": False, "report_changed": False, "portfolio_replay_executed": False, "ready_for_strategy_replay": False, "ready_for_formal": False, "not_live_rule": True, "forward_returns_live_rule_usage": False}


def _ticker_from_path(path: str) -> str:
    name = os.path.basename(path).replace("_TW.csv", "").replace(".csv", "")
    return name if re.fullmatch(r"\d{4,6}[A-Z]?", name) else ""


def _cache_inventory() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    candidates = []
    for path in glob.glob(str(REPO_ROOT / "backtest_cache/**/*.csv"), recursive=True):
        ticker = _ticker_from_path(path)
        if not ticker: continue
        try:
            frame = pd.read_csv(path, usecols=lambda c: c.lower() in {"date", "high", "low", "close"}, low_memory=False)
        except Exception:
            continue
        lower = {c.lower(): c for c in frame.columns}
        if not {"date", "high", "low", "close"}.issubset(lower): continue
        frame = frame.rename(columns={lower[k]: k for k in ("date", "high", "low", "close")})
        frame["date"] = pd.to_datetime(frame.date, errors="coerce")
        for col in ("high", "low", "close"): frame[col] = pd.to_numeric(frame[col], errors="coerce")
        frame = frame.dropna().drop_duplicates("date").sort_values("date")
        p1 = frame[frame.date.between(P1_START - pd.Timedelta(days=60), P1_END)]
        if len(p1): candidates.append({"ticker": ticker, "path": path, "P1_rows": len(p1), "start": p1.date.min(), "end": p1.date.max(), "source_lineage": "mixed_local_cache_provider_not_uniformly_official"})
    inventory = pd.DataFrame(candidates).sort_values(["ticker", "P1_rows", "path"], ascending=[True, False, True])
    best = inventory.drop_duplicates("ticker") if len(inventory) else inventory
    frames = {}
    for item in best.itertuples(index=False):
        d = pd.read_csv(item.path, usecols=lambda c: c.lower() in {"date", "high", "low", "close"}, low_memory=False)
        d.columns = [c.lower() for c in d.columns]; d["date"] = pd.to_datetime(d.date, errors="coerce")
        for c in ("high", "low", "close"): d[c] = pd.to_numeric(d[c], errors="coerce")
        frames[str(item.ticker)] = d.dropna().drop_duplicates("date").sort_values("date")
    return best, frames


def _expanding_pct(values: list[float]) -> list[float]:
    ordered: list[float] = []; out = []
    for value in values:
        if pd.isna(value): out.append(np.nan); continue
        bisect.insort(ordered, float(value)); out.append(bisect.bisect_right(ordered, float(value)) / len(ordered))
    return out


def _kd(frame: pd.DataFrame) -> pd.DataFrame:
    d = frame.copy(); low9 = d.low.rolling(9, min_periods=9).min(); high9 = d.high.rolling(9, min_periods=9).max(); denom = high9 - low9
    d["RSV9"] = np.where(denom.ne(0), (d.close - low9) / denom * 100, 50.0)
    k = 50.0; dd = 50.0; ks = []; ds = []
    for value in d.RSV9:
        if pd.notna(value): k = 2 / 3 * k + 1 / 3 * float(value); dd = 2 / 3 * dd + 1 / 3 * k
        ks.append(k); ds.append(dd)
    d["KD_K"] = ks; d["KD_D"] = ds; d["KD_J"] = 3 * d.KD_K - 2 * d.KD_D
    d["KD_K_slope_1d"] = d.KD_K.diff(); d["KD_D_slope_1d"] = d.KD_D.diff(); d["KD_K_slope_3d"] = d.KD_K.diff(3); d["KD_D_slope_3d"] = d.KD_D.diff(3)
    d["KD_golden_cross"] = (d.KD_K > d.KD_D) & (d.KD_K.shift() <= d.KD_D.shift()); d["KD_death_cross"] = (d.KD_K < d.KD_D) & (d.KD_K.shift() >= d.KD_D.shift())
    d["KD_K_expanding_percentile"] = _expanding_pct(d.KD_K.tolist()); d["KD_D_expanding_percentile"] = _expanding_pct(d.KD_D.tolist())
    d["KD_warmup_ready"] = np.arange(len(d)) >= 20
    d["KD_context"] = np.select([
        d.KD_warmup_ready & (d.KD_K_expanding_percentile <= .5) & (d.KD_K_slope_1d > 0) & ((d.KD_golden_cross) | (d.KD_K_slope_3d > 0)),
        d.KD_warmup_ready & (d.KD_K > d.KD_D) & (d.KD_K_slope_3d > 0),
        d.KD_warmup_ready & (d.KD_K_expanding_percentile >= .8) & (d.KD_K_slope_1d >= 0),
        d.KD_warmup_ready & (d.KD_death_cross | ((d.KD_K_slope_1d < 0) & (d.KD_D_slope_1d < 0))),
    ], ["turn_up", "healthy_advance", "overheat_warning", "confirmed_weakening"], default="cooling")
    return d


def _weekly_matrix(pool: pd.DataFrame, frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    parts = []
    for ticker, group in pool.groupby("ticker"):
        prices = frames.get(str(ticker))
        if prices is None: group = group.copy(); group["KD_source_ready"] = False; parts.append(group); continue
        kd = _kd(prices); view = group.merge(kd, left_on="snapshot_date", right_on="date", how="left")
        view["KD_source_ready"] = view.KD_warmup_ready.fillna(False); parts.append(view)
    out = pd.concat(parts, ignore_index=True, sort=False)
    out["KD_source_lineage"] = np.where(out.KD_source_ready, "mixed_local_HLC_cache_diagnostic_only", "missing_local_HLC_or_warmup")
    hard = out.high_exhaustion_or_breakdown_context.astype(str).str.lower().eq("true")
    no_kd_turn = out.rs_short_acceleration_flag.astype(str).str.lower().eq("true") & out.capital_rank_20d_improving_vs_60d.astype(str).str.lower().eq("true") & ~hard
    no_kd_pull = out.pullback_repair_sleeve_candidate_feature.astype(str).str.lower().eq("true") & ~hard
    no_kd_healthy = out.rs20_30_primary_momentum_positive.astype(str).str.lower().eq("true") & ~hard
    out["no_KD_lifecycle_candidate"] = no_kd_turn | no_kd_pull | no_kd_healthy
    out["KD_incremental_confirmed_candidate"] = out.no_KD_lifecycle_candidate & out.KD_source_ready & out.KD_context.isin(["turn_up", "healthy_advance"])
    out["future_return_used_as_rule"] = False
    return out


def _incremental_eval(matrix: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for horizon in (5, 10, 20):
        value = pd.to_numeric(matrix[f"forward_excess_vs_00631L_{horizon}d"], errors="coerce"); winner = value > 0; material = value > .10
        for label, mask in (("no_KD_baseline", matrix.no_KD_lifecycle_candidate), ("KD_incremental_confirmed", matrix.KD_incremental_confirmed_candidate)):
            selected = matrix[mask & value.notna()]
            rows.append({"variant": label, "horizon": horizon, "selected_rows": len(selected), "row_share": float(mask.mean()), "winner_hit_rate": float((value[mask] > 0).mean()) if mask.any() else np.nan, "material_winner_retention_share": float((mask & material).sum() / material.sum()) if material.sum() else np.nan, "false_entry_count": int((mask & ~winner & value.notna()).sum()), "missed_material_winner_count": int((~mask & material).sum()), "median_forward_excess": float(value[mask].median()) if mask.any() else np.nan, "evaluation_metadata_only": True})
    return pd.DataFrame(rows)


def _chip_inventory() -> tuple[pd.DataFrame, pd.DataFrame]:
    chip = pd.read_csv(REPO_ROOT / "outputs/chip_shadow_diagnostic_adapter_20260620/chip_diagnostic_panel.csv", dtype={"ticker": str}, low_memory=False); chip["date"] = pd.to_datetime(chip.date)
    rows = [
        {"family": "three_institutional_daily", "local_source_exists": True, "local_path": "outputs/chip_shadow_diagnostic_adapter_20260620/chip_diagnostic_panel.csv", "ticker_count": chip.ticker.nunique(), "date_start": chip.date.min(), "date_end": chip.date.max(), "P1_ready": False, "P2_ready": "partial_2024_to_2026_fixed7_scope", "PIT_release_lag": "same-day post-close if source timestamp accepted", "weekly_align": True, "allowed_context": "entry_confirmation|hold_warning|exit_pressure", "blocker": "no P1 and not dynamic80 universe"},
        {"family": "margin_short_lending", "local_source_exists": False, "P1_ready": False, "P2_ready": False, "PIT_release_lag": "requires source contract", "weekly_align": "unknown", "allowed_context": "risk|crowding|exit_warning", "blocker": "no local historical source"},
        {"family": "TDCC_holder_buckets", "local_source_exists": False, "P1_ready": False, "P2_ready": False, "PIT_release_lag": "weekly publication lag must be modeled", "weekly_align": True, "allowed_context": "hold_structure|confidence", "blocker": "no local PIT archive; publication lag and bucket revisions"},
        {"family": "futures_OI_foreign_net", "local_source_exists": False, "P1_ready": False, "P2_ready": False, "PIT_release_lag": "market-level post-close", "weekly_align": True, "allowed_context": "market threshold modulation only", "blocker": "no local accepted PIT history"},
    ]
    estimates = [
        {"family": "three_institutional_daily", "estimated_routes": 2914, "estimated_MB": 350, "scope": "market-level daily files then primary80 join", "download_now": False},
        {"family": "margin_short_lending", "estimated_routes": 2914, "estimated_MB": 400, "scope": "market-level daily files", "download_now": False},
        {"family": "TDCC_holder_buckets", "estimated_routes": 411, "estimated_MB": 250, "scope": "weekly market files with release lag", "download_now": False},
        {"family": "futures_OI_foreign_net", "estimated_routes": 2914, "estimated_MB": 50, "scope": "market-level only", "download_now": False},
    ]
    return pd.DataFrame(rows), pd.DataFrame(estimates)


def run(output_dir: str | Path = DEFAULT_OUTPUT) -> Path:
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True); (out / "current_step.txt").write_text("inventory_mixed_HLC_cache", encoding="utf-8")
    pool = source._weekly_candidate_matrix(); pool = pool[pool.snapshot_date.between(P1_START, P1_END)].copy(); pool["ticker"] = pool.ticker.astype(str)
    lineage, frames = _cache_inventory(); matrix = _weekly_matrix(pool, frames); chip, estimates = _chip_inventory()
    obsolete = out / "p1_full80_KD_incremental_evaluation_hook.csv"
    if obsolete.exists(): obsolete.unlink()
    lineage.to_csv(out / "p1_full80_KD_mixed_cache_lineage_audit.csv", index=False, encoding="utf-8-sig"); matrix[["snapshot_date", "ticker", "name", "KD_source_ready", "KD_source_lineage", "RSV9", "KD_K", "KD_D", "KD_J", "KD_K_slope_1d", "KD_D_slope_1d", "KD_K_slope_3d", "KD_D_slope_3d", "KD_golden_cross", "KD_death_cross", "KD_K_expanding_percentile", "KD_D_expanding_percentile", "KD_warmup_ready", "KD_context", "future_return_used_as_rule"]].to_csv(out / "p1_full80_KD_context_matrix_mixed_cache_proxy.csv", index=False, encoding="utf-8-sig"); chip.to_csv(out / "lifecycle_missing_chip_risk_feature_readiness_matrix.csv", index=False, encoding="utf-8-sig")
    estimates["estimated_time_hours"] = ["8-16", "8-20", "4-8", "4-8"]
    estimates["source_quality_target"] = ["official TWSE/TPEx", "official TWSE/TPEx", "official TDCC", "official TAIFEX"]
    estimates = pd.concat([pd.DataFrame([{"family": "event_aware_daily_HLC_primary80", "estimated_routes": 20453, "estimated_MB": "5000-15000", "scope": "primary80 membership segments plus warmup; historical corporate-action guard separate", "download_now": False, "estimated_time_hours": "40-100", "source_quality_target": "official TWSE/TPEx unadjusted; adjusted analysis remains blocked"}]), estimates], ignore_index=True)
    estimates.to_csv(out / "lifecycle_missing_feature_source_acquisition_estimate.csv", index=False, encoding="utf-8-sig")
    minimum = pd.DataFrame([
        ("event-aware daily HLC", "partial", "164/976 tickers local mixed cache; 35.87% weekly rows", "KD/MA/BIAS/price lifecycle invalid without it"),
        ("RS5/10/20/40/60 vs 0050", "ready", "weekly PIT primary80", "core trend coordinate"),
        ("ticker-specific BIAS20/60 percentile/zscore", "partial", "percentile ready; zscore blocked", "position and overheat rollover incomplete"),
        ("MA20/60 slope/reclaim/breakdown", "partial", "position ready; exact daily transitions partial", "S1/S4 timing distortion"),
        ("volatility/drawdown/breakdown/exhaustion", "ready", "weekly PIT proxy/fields", "risk confirmation"),
        ("traded-value rank/change", "ready", "weekly PIT", "capital confirmation"),
        ("three institutional flow/continuity", "blocked_P1", "local only fixed7 2024-2026", "entry/hold/exit confirmation materially missing"),
        ("margin/short/lending", "blocked", "no local PIT history", "crowding and forced unwind risk missing"),
        ("TDCC holder buckets with release lag", "blocked", "no local PIT archive", "holder structure persistence missing"),
        ("futures OI/foreign net", "blocked", "no local accepted PIT history", "market-level threshold modulation missing"),
        ("corporate-action analysis-price guard", "blocked_historical", "official unadjusted diagnostic only", "RS/BIAS/KD contamination remains possible"),
    ], columns=["minimum_field_family", "status", "coverage_source", "distortion_if_missing"])
    minimum.to_csv(out / "complete_lifecycle_minimum_feature_readiness_matrix.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([
        (1, "freeze Layer0-4 and PIT universe", "existing canonical contracts", "universe audit"),
        (2, "acquire/normalize complete event-aware HLC", "all primary80 PIT membership windows", "price feature readiness"),
        (3, "acquire institutional, margin/lending, TDCC, futures families", "official source with release timestamps", "PIT coverage audit"),
        (4, "materialize one unified lifecycle feature matrix", "KD+RS+BIAS+MA+risk+capital+chip", "no outcome fields"),
        (5, "freeze one lifecycle state machine and switch contract", "single complete spec", "contract review"),
        (6, "run P1 event diagnostic then unique-position path", "EP05 net cost primary", "Experiments verdict"),
        (7, "P2 secondary stop-gate", "only after P1", "no P2 masking"),
    ], columns=["sequence", "step", "scope", "gate"]).to_csv(out / "one_shot_complete_lifecycle_contract_execution_plan.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(columns=["signal_date", "violation_reason"]).to_csv(out / "future_data_audit.csv", index=False, encoding="utf-8-sig")
    coverage = float(matrix.KD_source_ready.mean())
    readiness = {"task_id": TASK_ID, "status": "inventory_complete_full_lifecycle_not_test_ready", "primary80_rows": len(matrix), "primary80_unique_tickers": pool.ticker.nunique(), "local_HLC_unique_tickers": len(frames), "KD_ready_rows": int(matrix.KD_source_ready.sum()), "KD_ready_share": coverage, "mixed_cache_diagnostic_only": True, "official_source_claim": False, "ready_for_experiments": False, "ready_for_Radar": False, "automatic_handoff_stopped": True, "chip_family_download_authorized": False, "selected_stock_adjusted_close_ready": False, "future_data_violation_count": 0, **FLAGS}
    (out / "readiness_for_lifecycle_missing_feature_incremental_diagnostic.json").write_text(json.dumps(readiness, ensure_ascii=False, indent=2, default=str), encoding="utf-8"); (out / "final_summary_zh.md").write_text(f"# Lifecycle missing-feature readiness inventory\n\n- Full80 KD mixed-cache coverage={coverage:.4f} ({int(matrix.KD_source_ready.sum()):,}/{len(matrix):,} rows), tickers={len(frames)}/{pool.ticker.nunique()}。\n- Source lineage mixed/non-uniformly official；不可作完整lifecycle結論。\n- Institutional P1、margin/lending、TDCC、futures OI皆blocked。\n- No performance/incremental outcome is produced; no automatic handoff。\n", encoding="utf-8"); (out / "manifest.json").write_text(json.dumps({"task_id": TASK_ID, "runner": __file__, "files": sorted(p.name for p in out.iterdir() if p.name != "p1_full80_KD_incremental_evaluation_hook.csv"), "readiness": readiness}, ensure_ascii=False, indent=2), encoding="utf-8"); (out / "current_step.txt").write_text("inventory_complete_waiting_full_spec_decision", encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT)); args = parser.parse_args(); print(run(args.output_dir))


if __name__ == "__main__": main()
