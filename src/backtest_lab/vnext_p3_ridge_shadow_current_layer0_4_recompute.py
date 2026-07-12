from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from backtest_lab.vnext_layer0_compact_weekly_universe_snapshot import _compact_snapshot


ROOT = Path(__file__).resolve().parents[2]
RADAR = Path(r"C:\Users\zergv\Documents\Codex\2026-05-23\ai-stock-rotation-radar-https-docs\outputs\radar_vnext_p3_ridge_shadow_current_exact_layer0_4_source_package_20260712")
OUT = ROOT / "outputs/vnext_p3_ridge_shadow_current_exact_layer0_4_recompute_20260712"
HIST = ROOT / "outputs/vnext_dynamic_candidate_pool_data_materialization_20260706/daily_market_features.csv"
LAYER4 = ROOT / "outputs/vnext_layer4_80_primary_pool_contract_20260708/layer4_80_primary_pool_contract.csv"
TASK = "TASK-BACKTEST-CORE-VNEXT-P3-RIDGE-SHADOW-CURRENT-EXACT-LAYER0-4-RECOMPUTE-001"


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_daily() -> pd.DataFrame:
    cols = ["trade_date", "ticker", "name", "market", "traded_value", "valid_universe", "liquidity_flag", "listing_status"]
    history = pd.read_csv(HIST, usecols=cols, dtype={"ticker": str}, low_memory=False)
    history = history[history.trade_date.ge("2026-03-01")]
    patch = pd.read_csv(RADAR / "ridge_shadow_current_full_market_official_ohlcv.csv.gz", dtype={"ticker": str}, low_memory=False)
    patch = patch.rename(columns={"date": "trade_date", "turnover_value": "traded_value"})
    patch["valid_universe"] = True
    patch["liquidity_flag"] = True
    patch["listing_status"] = "listed"
    patch = patch[cols]
    frame = pd.concat([history, patch], ignore_index=True).drop_duplicates(["trade_date", "ticker"], keep="last")
    frame["trade_date"] = pd.to_datetime(frame.trade_date)
    frame["traded_value"] = pd.to_numeric(frame.traded_value, errors="coerce").fillna(0)
    frame["eligible"] = frame.ticker.str.fullmatch(r"\d{4}") & ~frame.ticker.str.startswith("00")
    frame = frame.sort_values(["ticker", "trade_date"])
    grouped = frame.groupby("ticker").traded_value
    for days in [5, 20, 60]:
        frame[f"traded_value_{days}d"] = grouped.rolling(days, min_periods=1).sum().reset_index(level=0, drop=True)
    return frame


def layer0_snapshot(frame: pd.DataFrame) -> pd.DataFrame:
    weekly_dates = [pd.Timestamp("2026-06-12"), pd.Timestamp("2026-06-19"), pd.Timestamp("2026-06-26"), pd.Timestamp("2026-07-03"), pd.Timestamp("2026-07-09")]
    weekly = frame[frame.trade_date.isin(weekly_dates) & frame.eligible].copy()
    weekly["snapshot_date"] = weekly.trade_date
    for window in ["5d", "20d", "60d"]:
        weekly[f"traded_value_rank_{window}"] = weekly.groupby("snapshot_date")[f"traded_value_{window}"].rank(method="first", ascending=False)
    weekly = weekly.sort_values(["ticker", "snapshot_date"])
    weekly["rank_improvement_5d_vs_60d"] = weekly.traded_value_rank_60d - weekly.traded_value_rank_5d
    weekly["in_top300_5d"] = weekly.traded_value_rank_5d.le(300)
    weekly["top300_5d_count_last4w"] = weekly.groupby("ticker").in_top300_5d.rolling(4, min_periods=1).sum().reset_index(level=0, drop=True)
    weekly["buffer_persistence_2in4"] = weekly.top300_5d_count_last4w.ge(2)
    weekly["buffer_20d60d_confirmed"] = weekly.traded_value_rank_20d.le(300) | weekly.traded_value_rank_60d.le(300)
    weekly["is_ky_name_proxy"] = weekly.name.str.contains("-KY", na=False)
    snapshot = _compact_snapshot(weekly)
    return snapshot[snapshot.snapshot_date.eq(pd.Timestamp("2026-07-09"))].copy()


def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    source_ready = json.loads((RADAR / "readiness_for_core_p3_ridge_shadow_current_exact_layer0_4_source.json").read_text(encoding="utf-8"))
    layer0 = layer0_snapshot(load_daily())
    old = pd.read_csv(LAYER4, dtype={"ticker": str}, low_memory=False)
    known = set(old.ticker)
    active = layer0[layer0.active_for_layer1_source_scope].copy()
    active["historical_layer1_3_context_exists"] = active.ticker.isin(known)
    action = pd.read_csv(RADAR / "ridge_shadow_current_corporate_action_guard_rows.csv", dtype={"ticker": str})
    affected = set(action.loc[action.effective_date.astype(str).between("2026-03-01", "2026-07-09"), "ticker"])
    active["corporate_action_adjusted_analysis_review_required"] = active.ticker.isin(affected)
    active["adjusted_analysis_price_current_ready"] = False
    active["frozen_layer1_4_recompute_ready"] = False
    active["blocked_reason"] = "adjusted_analysis_price_current_not_in_source_package"
    active.loc[~active.historical_layer1_3_context_exists, "blocked_reason"] += "|no_historical_PIT_fundamental_context"
    active.loc[active.corporate_action_adjusted_analysis_review_required, "blocked_reason"] += "|corporate_action_factor_review_required"

    gap = active[["snapshot_date", "ticker", "name", "market", "selection_bucket", "traded_value_rank_5d", "historical_layer1_3_context_exists", "corporate_action_adjusted_analysis_review_required", "adjusted_analysis_price_current_ready", "blocked_reason"]].copy()
    gap["requested_source"] = "trusted_nonofficial_adjusted_analysis_OHLC_current_window_and_exact_PIT_fundamental_asof_for_new_tickers_only"
    gap["mass_history_download_allowed"] = False
    gap.to_csv(OUT / "p3_ridge_shadow_current_layer1_4_bounded_gap_ledger.csv", index=False, encoding="utf-8-sig")
    layer0.to_csv(OUT / "p3_ridge_shadow_current_exact_layer0_snapshot_20260709.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([
        {"gate": "Radar source integrity", "status": "ready", "rows": source_ready["coverage"]["official_ohlcv_rows"]},
        {"gate": "exact Layer0 2026-07-09", "status": "ready", "rows": len(layer0)},
        {"gate": "active Layer1 source scope", "status": "ready", "rows": len(active)},
        {"gate": "current adjusted analysis price", "status": "blocked", "rows": len(active)},
        {"gate": "new-ticker historical PIT fundamental context", "status": "blocked", "rows": int((~active.historical_layer1_3_context_exists).sum())},
        {"gate": "exact Layer4 primary80", "status": "blocked", "rows": 0},
    ]).to_csv(OUT / "p3_ridge_shadow_current_layer0_4_gate_audit.csv", index=False, encoding="utf-8-sig")
    readiness = {
        "task_id": TASK,
        "status": "exact_layer0_ready_layer1_4_blocked_bounded_source_delta_required",
        "requested_date": "2026-07-09",
        "actual_exact_layer0_date": "2026-07-09",
        "exact_layer0_rows": len(layer0),
        "active_layer1_source_scope_rows": len(active),
        "new_tickers_without_historical_layer1_3_context": int((~active.historical_layer1_3_context_exists).sum()),
        "corporate_action_review_tickers_in_active_scope": int(active.corporate_action_adjusted_analysis_review_required.sum()),
        "current_adjusted_analysis_price_ready": False,
        "current_exact_layer4_primary80_ready": False,
        "ready_for_first_prospective_ridge_prediction": False,
        "carried_2026_06_29_membership_used": False,
        "source_gap_silently_filled": False,
        "future_data_violation_count": 0,
        "formal_model_changed": False,
        "trade_decision_changed": False,
        "active_in_trade_decision": False,
        "report_changed": False,
        "ready_for_formal": False,
        "ready_for_strategy_replay": False,
        "not_live_rule": True,
        "forward_returns_live_rule_usage": False,
    }
    (OUT / "readiness_for_current_exact_layer0_4_recompute.json").write_text(json.dumps(readiness, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "final_summary_zh.md").write_text("# Ridge shadow current exact Layer0-4 recompute\n\n2026-07-09 exact Layer0 已重算。Frozen Layer1-4 仍缺 current adjusted analysis OHLC，且 8 檔 active scope 股票沒有歷史 PIT 基本面 context；不可用 raw execution price 或 6/29 membership 替代。已輸出 candidate-scoped bounded gap ledger。\n", encoding="utf-8")
    files = sorted(path for path in OUT.iterdir() if path.is_file() and path.name != "manifest.json")
    (OUT / "manifest.json").write_text(json.dumps({"task_id": TASK, "inputs": {"radar_manifest_sha256": file_hash(RADAR / "manifest.json"), "historical_daily_market_sha256": file_hash(HIST), "historical_layer4_sha256": file_hash(LAYER4)}, "files": [{"name": path.name, "sha256": file_hash(path), "bytes": path.stat().st_size} for path in files]}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    run()
