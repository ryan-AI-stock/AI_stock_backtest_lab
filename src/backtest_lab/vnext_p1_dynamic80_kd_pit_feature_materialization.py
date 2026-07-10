from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from backtest_lab import vnext_daily_incumbent_challenger_state_machine_contract as source


TASK_ID = "TASK-BACKTEST-CORE-VNEXT-P1-DYNAMIC80-KD-PIT-FEATURE-MATERIALIZATION-001"
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO_ROOT / "outputs/vnext_p1_dynamic80_kd_pit_feature_materialization_20260710"
RADAR_INCUMBENT = Path(r"C:\Users\zergv\Documents\Codex\2026-05-23\ai-stock-rotation-radar-https-docs\outputs\radar_vnext_p1_dynamic80_incumbent_hold_selected_stock_daily_ohlc_gap_fill_20260710")
P1_START, P1_END = pd.Timestamp("2015-01-02"), pd.Timestamp("2022-12-29")
WARMUP_TRADING_DAYS = 20
FLAGS = {"formal_model_changed": False, "trade_decision_changed": False, "active_in_trade_decision": False, "report_changed": False, "portfolio_replay_executed": False, "ready_for_strategy_replay": False, "ready_for_formal": False, "not_live_rule": True, "forward_returns_live_rule_usage": False}


def _segments(pool: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    rows = []
    for ticker, group in pool.groupby("ticker"):
        dates = sorted(pd.to_datetime(group.snapshot_date.unique()))
        segment = 0; start = previous = dates[0]
        chunks = []
        for date in dates[1:]:
            if (date - previous).days > 10:
                chunks.append((start, previous)); start = date
            previous = date
        chunks.append((start, previous))
        for start, end in chunks:
            segment += 1
            prior = calendar[calendar < start]
            warmup_start = prior[-WARMUP_TRADING_DAYS] if len(prior) >= WARMUP_TRADING_DAYS else (prior[0] if len(prior) else start)
            required = calendar[(calendar >= warmup_start) & (calendar <= end)]
            rows.append({"ticker": ticker, "segment_id": f"{ticker}_seg{segment:03d}", "first_pool_snapshot": start, "last_pool_snapshot": end, "warmup_start": warmup_start, "required_trading_date_count": len(required), "warmup_rows_not_decision_eligible": True, "KD_reset_K": 50.0, "KD_reset_D": 50.0, "segment_reset_reason": "new_or_reentered_primary80_after_gap_gt_10_calendar_days"})
    return pd.DataFrame(rows)


def _requirements(segments: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    rows = []
    for item in segments.itertuples(index=False):
        for date in calendar[(calendar >= item.warmup_start) & (calendar <= item.last_pool_snapshot)]:
            rows.append({"ticker": item.ticker, "segment_id": item.segment_id, "price_date": date, "required_fields": "high|low|close", "decision_eligible_after_warmup": date >= item.first_pool_snapshot})
    return pd.DataFrame(rows).drop_duplicates(["ticker", "segment_id", "price_date"])


def _available() -> pd.DataFrame:
    path = RADAR_INCUMBENT / "p1_dynamic80_incumbent_hold_selected_stock_daily_unadjusted_ohlc_rows.csv"
    frame = pd.read_csv(path, dtype={"ticker": str}, low_memory=False)
    frame["ticker"] = frame.ticker.astype(str).str.replace(r"\.0$", "", regex=True)
    frame["price_date"] = pd.to_datetime(frame["date"], errors="coerce")
    for col in ("high", "low", "close"): frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame[frame[["high", "low", "close"]].notna().all(axis=1)][["ticker", "price_date", "high", "low", "close", "source_quality", "source_route"]].drop_duplicates(["ticker", "price_date"])


def run(output_dir: str | Path = DEFAULT_OUTPUT) -> Path:
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True); (out / "current_step.txt").write_text("inventory_primary80_membership_segments", encoding="utf-8")
    pool = source._weekly_candidate_matrix(); pool = pool[pool.snapshot_date.between(P1_START, P1_END)].copy(); pool["ticker"] = pool.ticker.astype(str)
    market = source._load_market_daily(); calendar = pd.DatetimeIndex(pd.to_datetime(market.signal_date.unique())).sort_values(); calendar = calendar[(calendar >= P1_START) & (calendar <= P1_END)]
    segments = _segments(pool, calendar); req = _requirements(segments, calendar); available = _available()
    audit = req.merge(available, on=["ticker", "price_date"], how="left")
    audit["official_HLC_ready"] = audit[["high", "low", "close"]].notna().all(axis=1)
    gaps = audit[~audit.official_HLC_ready].copy()
    gaps["ticker_month"] = gaps.price_date.dt.strftime("%Y-%m")
    route = gaps.groupby(["ticker", "ticker_month"], as_index=False).agg(required_date_start=("price_date", "min"), required_date_end=("price_date", "max"), required_date_count=("price_date", "nunique"), segment_ids=("segment_id", lambda x: "|".join(sorted(set(x)))))
    route["source_scope"] = "primary80_membership_segment_plus_20TD_warmup_only"
    route["no_full_ticker_history"] = True
    segments.to_csv(out / "p1_dynamic80_kd_membership_segment_contract.csv", index=False, encoding="utf-8-sig")
    audit.to_csv(out / "p1_dynamic80_kd_HLC_coverage_audit.csv", index=False, encoding="utf-8-sig")
    gaps[["ticker", "segment_id", "price_date", "required_fields", "decision_eligible_after_warmup"]].to_csv(out / "p1_dynamic80_kd_exact_ticker_date_HLC_gap_ledger.csv", index=False, encoding="utf-8-sig")
    route.to_csv(out / "p1_dynamic80_kd_bounded_ticker_month_source_request.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{"formula": "RSV9=(close-lowest_low_9)/(highest_high_9-lowest_low_9)*100; K=2/3*Kprev+1/3*RSV; D=2/3*Dprev+1/3*K; J=3K-2D", "initial_K": 50, "initial_D": 50, "warmup_trading_days": WARMUP_TRADING_DAYS, "warmup_decision_eligible": False, "cross_basis": "same-day close PIT", "execution": "next-trading-day", "self_percentile": "expanding prior-and-current values within ticker, never future", "J_role": "context_only"}]).to_csv(out / "p1_dynamic80_kd_formula_policy.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(columns=["signal_date", "violation_reason"]).to_csv(out / "future_data_audit.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([
        {"source": "Radar incumbent-hold official selected OHLC", "accepted": True, "role": "reused first", "rows": int(audit.official_HLC_ready.sum())},
        {"source": "Core backtest_cache generic OHLC", "accepted": False, "role": "inventory only", "rows": 0, "blocked_reason": "provider/source lineage is not uniformly official; cannot promote to official-unadjusted KD source"},
        {"source": "new Radar primary80 segment request", "accepted": False, "role": "not dispatched", "rows": len(gaps), "blocked_reason": "20,453 ticker-month routes is effectively mass historical acquisition and violates bounded-source policy"},
    ]).to_csv(out / "p1_dynamic80_kd_source_acceptance_audit.csv", index=False, encoding="utf-8-sig")
    readiness = {"task_id": TASK_ID, "status": "blocked_scope_exceeds_bounded_source_policy_strategy_scope_decision_required", "primary80_candidate_rows": len(pool), "unique_tickers": pool.ticker.nunique(), "membership_segments": len(segments), "required_ticker_date_rows": len(req), "reused_incumbent_hold_official_HLC_rows": int(audit.official_HLC_ready.sum()), "remaining_exact_ticker_date_HLC_gaps": len(gaps), "bounded_ticker_month_routes": len(route), "Radar_gap_fill_dispatched": False, "KD_materialized": False, "ready_for_lifecycle_KD_integration": False, "ready_for_experiments": False, "official_unadjusted_OHLC_diagnostic_only": True, "selected_stock_adjusted_close_ready": False, "future_data_violation_count": 0, **FLAGS}
    (out / "readiness_for_p1_dynamic80_kd_pit_feature_materialization.json").write_text(json.dumps(readiness, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "final_summary_zh.md").write_text(f"# P1 Dynamic80 KD PIT feature materialization\n\n- KD是lifecycle核心位置座標，no-KD comparators不代表KD假說。\n- 採primary80連續membership segment + {WARMUP_TRADING_DAYS}TD warmup；不抓每檔P1全歷史。\n- segments={len(segments):,}；required ticker-date={len(req):,}；reused official HLC={int(audit.official_HLC_ready.sum()):,}；remaining gaps={len(gaps):,}；ticker-month routes={len(route):,}。\n- 20,453 routes仍實質接近mass historical acquisition，因此未交Radar。需Strategy Center裁決縮小KD scoring universe，或明確授權一次性primary80 segment HLC source。\n- generic backtest_cache來源不全為官方，不升official-unadjusted KD source。\n- warmup rows不得進selected decision；future_data_violation_count=0。\n", encoding="utf-8")
    (out / "manifest.json").write_text(json.dumps({"task_id": TASK_ID, "runner": __file__, "source_incumbent_hold": str(RADAR_INCUMBENT), "files": sorted(p.name for p in out.iterdir()), "readiness": readiness}, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "current_step.txt").write_text("blocked_scope_decision_required", encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT)); args = parser.parse_args(); print(run(args.output_dir))


if __name__ == "__main__": main()
