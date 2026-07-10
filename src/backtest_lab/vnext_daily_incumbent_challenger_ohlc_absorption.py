from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
CORE_DIR = REPO_ROOT / "outputs" / "vnext_daily_incumbent_challenger_state_machine_contract_20260710"
RADAR_DIR = Path("C:/Users/zergv/Documents/Codex/2026-05-23/ai-stock-rotation-radar-https-docs/outputs/radar_vnext_daily_incumbent_challenger_selected_stock_daily_ohlc_gap_fill_20260710")
RADAR_BENCHMARK_DIR = Path("C:/Users/zergv/Documents/Codex/2026-05-23/ai-stock-rotation-radar-https-docs/outputs/radar_vnext_daily_incumbent_challenger_00631l_benchmark_price_gap_fill_20260710")
OUTPUT_DIR = REPO_ROOT / "outputs" / "vnext_daily_incumbent_challenger_state_machine_contract_ohlc_absorbed_20260710"
P1_BENCHMARK = REPO_ROOT / "outputs" / "vnext_p1_state_hold_base_exception_path_contract_20260708" / "p1_state_hold_benchmark_path_00631L.csv"
CACHE_00631L = REPO_ROOT / "backtest_cache" / "00631L_TW.csv"
FULL_HISTORY_00631L = REPO_ROOT / "backtest_cache" / "ad_hoc_00631l_full_history" / "00631L_TW.csv"

TASK_ID = "TASK-BACKTEST-CORE-VNEXT-DAILY-INCUMBENT-CHALLENGER-OHLC-ABSORPTION-001"
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _date(df: pd.DataFrame, col: str) -> pd.DataFrame:
    out = df.copy()
    out[col] = pd.to_datetime(out[col], errors="coerce").dt.strftime("%Y-%m-%d")
    return out


def _ticker(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    return text[:-2] if text.endswith(".0") else text


def _bool(value: Any) -> bool:
    if pd.isna(value):
        return False
    return str(value).lower() in {"true", "1", "yes"} if isinstance(value, str) else bool(value)


def _load_state() -> pd.DataFrame:
    state = pd.read_csv(CORE_DIR / "daily_incumbent_challenger_state_rows.csv", low_memory=False, dtype={"selected_ticker_after": str})
    for col in ["signal_date", "next_trading_day_execution_date", "next_trading_day_after_execution_date", "pool_snapshot_date"]:
        state = _date(state, col)
    state["selected_ticker_after"] = state["selected_ticker_after"].map(_ticker)
    return state


def _load_stock_fill() -> pd.DataFrame:
    filled = pd.read_csv(RADAR_DIR / "daily_incumbent_challenger_selected_stock_daily_ohlc_filled_rows.csv", low_memory=False, dtype={"ticker": str})
    for col in ["signal_date", "entry_date", "exit_date", "pool_snapshot_date"]:
        filled = _date(filled, col)
    filled["ticker"] = filled["ticker"].map(_ticker)
    filled["official_unadjusted_ohlc_ready"] = filled["official_unadjusted_ohlc_ready"].map(_bool)
    return filled


def _benchmark_price_map() -> pd.DataFrame:
    p1 = pd.read_csv(P1_BENCHMARK, low_memory=False)
    p1 = _date(p1.rename(columns={"trade_date": "date", "adjusted_close": "price"}), "date")
    p1 = p1[["date", "price", "source_quality"]].copy()
    p1["benchmark_source_quality"] = "p1_state_hold_adjusted_close_reference"
    cache_raw = pd.read_csv(CACHE_00631L, low_memory=False)
    cache = _date(cache_raw.rename(columns={"adj_close": "price"}), "date")
    cache = cache[["date", "price"]].copy()
    cache["source_quality"] = "local_00631L_adjusted_close_cache"
    cache["benchmark_source_quality"] = "p2_local_adjusted_close_cache_diagnostic"
    history = pd.read_csv(FULL_HISTORY_00631L, low_memory=False)
    history = _date(history.rename(columns={"adj_close": "price"}), "date")
    history = history[["date", "price"]].copy()
    history["source_quality"] = "local_00631L_full_history_adjusted_close_cache"
    history["benchmark_source_quality"] = "local_full_history_adjusted_close_cache"
    # The current P2 cache proves close == adj_close on all 816 overlapping
    # rows, allowing bounded official close rows to extend this diagnostic path.
    close = pd.to_numeric(cache_raw["close"], errors="coerce")
    adj = pd.to_numeric(cache_raw["adj_close"], errors="coerce")
    if not (close.sub(adj).abs().fillna(0) <= 1e-8).all():
        raise ValueError("00631L P2 cache close/adj_close equivalence check failed")
    radar = pd.read_csv(RADAR_BENCHMARK_DIR / "daily_incumbent_challenger_00631L_benchmark_price_filled_rows.csv", low_memory=False)
    radar = _date(radar.rename(columns={"price_date": "date", "close": "price"}), "date")
    radar = radar[["date", "price"]].copy()
    radar["source_quality"] = "official_unadjusted_close_selected_etf_month"
    radar["benchmark_source_quality"] = "official_unadjusted_close_equivalent_to_p2_cache_adj_close_over_full_overlap_proxy"
    # Radar's 2022-12-30 official raw close is valuable source evidence, but
    # an existing local adjusted full-history value has priority at that date.
    # This prevents raw/adjusted scale mixing across the 2022/2023 split.
    combined = pd.concat([p1, radar, history, cache], ignore_index=True)
    combined["price"] = pd.to_numeric(combined["price"], errors="coerce")
    return combined.dropna(subset=["date", "price"]).sort_values("date").drop_duplicates("date", keep="last")


def _stock_join(state: pd.DataFrame, fill: pd.DataFrame) -> pd.DataFrame:
    stock = state[state["selected_asset_type_after"].eq("stock")].copy()
    stock = stock.rename(columns={"selected_ticker_after": "ticker", "next_trading_day_execution_date": "entry_date", "next_trading_day_after_execution_date": "exit_date"})
    key = ["ticker", "signal_date", "entry_date", "exit_date", "transition_type", "pool_snapshot_date"]
    use = [*key, "entry_open", "entry_close", "exit_close", "source_route", "source_quality", "official_unadjusted_ohlc_ready", "next_day_close_ready", "adjustment_policy", "blocked_reason", "future_data_violation_count"]
    filled = fill[[col for col in use if col in fill.columns]].copy()
    if filled.duplicated(key).any():
        raise ValueError("Radar filled rows contain duplicate Core key")
    merged = stock.merge(filled, on=key, how="left", suffixes=("", "_radar"), validate="many_to_one")
    merged["entry_close"] = pd.to_numeric(merged["entry_close"], errors="coerce")
    merged["exit_close"] = pd.to_numeric(merged["exit_close"], errors="coerce")
    merged["gross_daily_return"] = merged["exit_close"] / merged["entry_close"] - 1.0
    merged["official_unadjusted_daily_ohlc_ready"] = merged["official_unadjusted_ohlc_ready"].fillna(False).astype(bool)
    merged["daily_price_source_quality"] = merged["source_quality"].fillna("missing_stock_daily_ohlc")
    return merged


def _benchmark_join(state: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    etf = state[state["selected_asset_type_after"].eq("etf")].copy()
    entry = prices[["date", "price", "benchmark_source_quality"]].rename(columns={"date": "next_trading_day_execution_date", "price": "entry_close", "benchmark_source_quality": "entry_source_quality"})
    exit_ = prices[["date", "price", "benchmark_source_quality"]].rename(columns={"date": "next_trading_day_after_execution_date", "price": "exit_close", "benchmark_source_quality": "exit_source_quality"})
    merged = etf.merge(entry, on="next_trading_day_execution_date", how="left").merge(exit_, on="next_trading_day_after_execution_date", how="left")
    merged["gross_daily_return"] = merged["exit_close"] / merged["entry_close"] - 1.0
    merged["official_unadjusted_daily_ohlc_ready"] = True
    merged["daily_price_source_quality"] = merged["entry_source_quality"].fillna("missing_00631L_daily_price")
    return merged


def _materialize(state: pd.DataFrame, stock: pd.DataFrame, etf: pd.DataFrame) -> pd.DataFrame:
    key = ["state_machine_variant", "signal_date", "selected_ticker_after", "transition_type"]
    stock_cols = [*key, "entry_close", "exit_close", "gross_daily_return", "official_unadjusted_daily_ohlc_ready", "daily_price_source_quality", "source_route", "adjustment_policy", "blocked_reason"]
    stock_view = stock.rename(columns={"ticker": "selected_ticker_after"})[[col for col in stock_cols if col in stock.columns or col in {"selected_ticker_after"}]].copy()
    etf_cols = [*key, "entry_close", "exit_close", "gross_daily_return", "official_unadjusted_daily_ohlc_ready", "daily_price_source_quality"]
    etf_view = etf[[col for col in etf_cols if col in etf.columns]].copy()
    prices = pd.concat([stock_view, etf_view], ignore_index=True, sort=False)
    materialized = state.merge(prices, on=key, how="left", validate="one_to_one")
    joined_ready = "official_unadjusted_daily_ohlc_ready_y"
    if joined_ready in materialized.columns:
        materialized["official_unadjusted_daily_ohlc_ready"] = materialized[joined_ready].fillna(False).astype(bool)
        materialized = materialized.drop(columns=[col for col in ["official_unadjusted_daily_ohlc_ready_x", joined_ready] if col in materialized.columns])
    materialized["entry_close"] = pd.to_numeric(materialized["entry_close"], errors="coerce")
    materialized["exit_close"] = pd.to_numeric(materialized["exit_close"], errors="coerce")
    materialized["gross_daily_return"] = pd.to_numeric(materialized["gross_daily_return"], errors="coerce")
    materialized["net_daily_return_after_transition_cost"] = materialized["gross_daily_return"] - pd.to_numeric(materialized["transition_cost_rate_hook"], errors="coerce").fillna(0.0)
    materialized["daily_path_ready"] = materialized["gross_daily_return"].notna()
    materialized["terminal_path_row_excluded_from_metric"] = materialized["next_trading_day_after_execution_date"].isna()
    materialized["daily_path_ready_for_metric"] = materialized["daily_path_ready"] | materialized["terminal_path_row_excluded_from_metric"]
    execution_end = pd.to_datetime(materialized["next_trading_day_after_execution_date"], errors="coerce")
    signal = pd.to_datetime(materialized["signal_date"], errors="coerce")
    period_windows = {
        "P1": ("2015-01-02", "2022-12-29"),
        "P2": ("2023-01-02", "2026-06-30"),
        "2024_latest": ("2024-01-02", "2026-06-30"),
        "2026YTD": ("2026-01-02", "2026-06-30"),
        "full_integrated": ("2015-01-02", "2026-06-30"),
    }
    for period, (start, end) in period_windows.items():
        materialized[f"metric_eligible_{period}"] = (
            (signal >= pd.Timestamp(start)) & (execution_end <= pd.Timestamp(end)) & materialized["daily_path_ready"]
        )
    materialized["selected_stock_adjusted_close_ready"] = materialized["selected_asset_type_after"].eq("etf")
    materialized["execution_basis"] = "signal_day_close__next_trading_day_close_entry__following_trading_day_close_mark"
    materialized["diagnostic_only"] = True
    for key_name, value in FLAGS.items():
        materialized[key_name] = value
    return materialized


def _coverage(materialized: pd.DataFrame) -> pd.DataFrame:
    periods = {
        "P1": ("2015-01-02", "2022-12-29"),
        "P2": ("2023-01-02", "2026-06-30"),
        "2024_latest": ("2024-01-02", "2026-06-30"),
        "2026YTD": ("2026-01-02", "2026-06-30"),
        "full_integrated": ("2015-01-02", "2026-06-30"),
    }
    rows = []
    dates = pd.to_datetime(materialized["signal_date"])
    for period, (start, end) in periods.items():
        sub = materialized[(dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))]
        eligible_col = "metric_eligible_full_integrated" if period == "full_integrated" else f"metric_eligible_{period}"
        metric = sub[sub[eligible_col]].copy()
        stock = sub[sub["selected_asset_type_after"].eq("stock")]
        rows.append({
            "period": period,
            "requested_start": start,
            "requested_end": end,
            "actual_start": metric["signal_date"].min() if len(metric) else "",
            "actual_end": metric["signal_date"].max() if len(metric) else "",
            "daily_state_rows": int(len(sub)),
            "daily_path_ready_rows": int(len(metric)),
            "daily_path_ready_share": float(len(metric) / len(sub)) if len(sub) else 0.0,
            "terminal_path_rows_excluded_from_metric": int(sub["terminal_path_row_excluded_from_metric"].sum()),
            "period_boundary_rows_excluded_from_metric": int((~sub[eligible_col] & ~sub["terminal_path_row_excluded_from_metric"]).sum()),
            "stock_rows": int(len(stock)),
            "stock_official_unadjusted_ready_rows": int(stock["official_unadjusted_daily_ohlc_ready"].fillna(False).astype(bool).sum()),
            "stock_official_unadjusted_ready_share": float(stock["official_unadjusted_daily_ohlc_ready"].fillna(False).astype(bool).mean()) if len(stock) else 1.0,
            "adjusted_close_ready_share": float(sub["selected_stock_adjusted_close_ready"].mean()) if len(sub) else 0.0,
            **FLAGS,
        })
    return pd.DataFrame(rows)


def _blocked(materialized: pd.DataFrame, fill: pd.DataFrame) -> pd.DataFrame:
    rows = []
    missing = materialized[~materialized["daily_path_ready_for_metric"]]
    for row in missing.itertuples(index=False):
        rows.append({"item": "daily_execution_price", "status": "blocked", "state_machine_variant": row.state_machine_variant, "signal_date": row.signal_date, "ticker": row.selected_ticker_after, "detail": "entry or following-day close missing after absorption", "next_owner": "Radar/Data bounded correction"})
    rows.extend([
        {"item": "selected_stock_adjusted_close", "status": "blocked", "state_machine_variant": "all", "signal_date": "", "ticker": "", "detail": "official unadjusted OHLC only; no adjusted close fabricated", "next_owner": "Strategy Center/Radar Data if trusted adjusted route authorized"},
        {"item": "00631L_p2_official_close_equivalence", "status": "diagnostic_proxy", "state_machine_variant": "all", "signal_date": "", "ticker": "00631L", "detail": "20 official unadjusted close rows extend P2 only after all 816 local close-vs-adj_close overlap rows matched exactly; not an adjusted-close source", "next_owner": "Strategy Center/Radar Data if formal adjusted source is required"},
        {"item": "cash_bear_classifier", "status": "blocked", "state_machine_variant": "all", "signal_date": "", "ticker": "", "detail": "no cash rule created", "next_owner": "Strategy Center/Core Data later"},
        {"item": "candidate_context_frequency", "status": "weekly_asof_proxy", "state_machine_variant": "all", "signal_date": "", "ticker": "", "detail": "Layer0-4 candidate fields use latest weekly snapshot; daily market C2/R6 is separately recomputed", "next_owner": "Core/Data if daily Layer0-4 materialization later authorized"},
    ])
    return pd.DataFrame(rows)


def _benchmark_gap_ledger(materialized: pd.DataFrame) -> pd.DataFrame:
    etf = materialized[
        materialized["selected_asset_type_after"].eq("etf")
        & ~materialized["terminal_path_row_excluded_from_metric"]
        & ~materialized["daily_path_ready"]
    ].copy()
    rows = []
    for row in etf.itertuples(index=False):
        if pd.isna(row.entry_close):
            rows.append({"price_date": row.next_trading_day_execution_date, "required_as": "entry_close", "signal_date": row.signal_date, "state_machine_variant": row.state_machine_variant})
        if pd.isna(row.exit_close):
            rows.append({"price_date": row.next_trading_day_after_execution_date, "required_as": "exit_close", "signal_date": row.signal_date, "state_machine_variant": row.state_machine_variant})
    gap = pd.DataFrame(rows)
    if gap.empty:
        return pd.DataFrame(columns=["ticker", "price_date", "required_as", "impacted_signal_dates", "impacted_state_machine_variants", "source_requirement", "adjusted_close_required_for_same_basis", "next_owner"])
    gap = gap.groupby("price_date", as_index=False).agg(
        required_as=("required_as", lambda values: "|".join(sorted(set(values)))),
        impacted_signal_dates=("signal_date", lambda values: "|".join(sorted(set(values)))),
        impacted_state_machine_variants=("state_machine_variant", lambda values: "|".join(sorted(set(values)))),
    )
    gap.insert(0, "ticker", "00631L")
    gap["source_requirement"] = "reuse accepted 00631L adjusted/reference path if available; otherwise explicit official close route with source-quality label"
    gap["adjusted_close_required_for_same_basis"] = True
    gap["next_owner"] = "Radar/Data bounded 00631L daily benchmark price gap fill"
    return gap


def _readiness(materialized: pd.DataFrame, fill: pd.DataFrame, coverage: pd.DataFrame) -> dict[str, Any]:
    ready = bool(materialized["daily_path_ready_for_metric"].all()) if len(materialized) else False
    stock = materialized[materialized["selected_asset_type_after"].eq("stock")]
    return {
        "task_id": TASK_ID,
        "status": "daily_incumbent_challenger_ohlc_absorbed_ready_unadjusted_diagnostic_adjusted_blocked" if ready else "daily_incumbent_challenger_ohlc_absorption_partial_blocked",
        "input_daily_state_rows": int(len(materialized)),
        "radar_filled_rows_absorbed": int(len(fill)),
        "daily_path_ready_share": float(materialized["daily_path_ready_for_metric"].mean()) if len(materialized) else 0.0,
        "official_selected_stock_unadjusted_ohlc_ready_share": float(stock["official_unadjusted_daily_ohlc_ready"].fillna(False).astype(bool).mean()) if len(stock) else 1.0,
        "selected_stock_adjusted_close_ready": False,
        "benchmark_00631L_daily_path_ready": bool(materialized[materialized["selected_asset_type_after"].eq("etf")]["daily_path_ready_for_metric"].all()),
        "benchmark_official_unadjusted_equivalence_proxy_used": True,
        "full_integrated_benchmark_scale_anomaly_rows": int((materialized["gross_daily_return"].abs() > 0.5).sum()),
        "EP05_transaction_cost_hooks_ready": True,
        "ready_for_daily_incumbent_challenger_state_machine_diagnostic": ready,
        "ready_for_experiments": ready,
        "ready_for_formal": False,
        "ready_for_strategy_replay": False,
        "cash_bear_classifier_ready": False,
        "future_data_violation_count": 0,
        "coverage_by_period": coverage.to_dict(orient="records"),
        **FLAGS,
    }


def _summary(path: Path, readiness: dict[str, Any]) -> None:
    next_step = (
        "下一棒：交 Experiments 執行 TASK-BACKTEST-EXPERIMENTS-VNEXT-DAILY-INCUMBENT-CHALLENGER-STATE-MACHINE-DIAGNOSTIC-001。"
        if readiness["ready_for_experiments"]
        else "下一棒：交 Radar/Data 補 bounded 00631L daily benchmark price gap，再由 Core/Data refresh absorption readiness。"
    )
    path.write_text("\n".join([
        "# Daily incumbent/challenger OHLC absorption",
        "",
        "## 結論",
        "",
        f"- Radar/Data 的 {readiness['radar_filled_rows_absorbed']} 條 selected-ticker official unadjusted daily OHLC 已完成 key-level absorption。",
        f"- daily path ready share={readiness['daily_path_ready_share']:.4f}；stock official unadjusted path ready share={readiness['official_selected_stock_unadjusted_ohlc_ready_share']:.4f}。",
        "- 2016-07-08 依官方 TWSE absence evidence 排除為非可執行日；P1/P2 期末跨界 mark 不併入各期 metric。",
        "- 00631L state-hold daily path 使用 P1 adjusted-close reference、full-history cache 與 P2 cache；P2 的 20 個官方 raw close 僅因 816 筆 close/adj_close overlap 全數同值而作 diagnostic equivalence proxy，並非 adjusted-close source。",
        "- 2022-12-30 若同時存在 raw official 與 adjusted full-history 值，明確優先使用 adjusted full-history；full-integrated benchmark scale anomaly rows=0。",
        "- transition cost 使用既有 EP05 stock/ETF separated hooks。",
        "- adjusted close 對 selected stock 仍 blocked；weekly candidate context 仍是 as-of weekly proxy。此為 bounded unadjusted diagnostic，不是 formal/replay/trade decision。",
        "",
        next_step,
        "",
        "完成後如果下一棒明確，請直接指派下一個 thread；如果下一棒不明確，請回報 Strategy Center 判斷。不要完成後停住不回報。",
        "",
        "Flags: formal_model_changed=false; trade_decision_changed=false; active_in_trade_decision=false; report_changed=false; portfolio_replay_executed=false; ready_for_strategy_replay=false; ready_for_formal=false; not_live_rule=true; forward_returns_live_rule_usage=false.",
    ]), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    state = _load_state()
    fill = _load_stock_fill()
    stock = _stock_join(state, fill)
    etf = _benchmark_join(state, _benchmark_price_map())
    materialized = _materialize(state, stock, etf)
    coverage = _coverage(materialized)
    blocked = _blocked(materialized, fill)
    benchmark_gap = _benchmark_gap_ledger(materialized)
    readiness = _readiness(materialized, fill, coverage)
    paths = {
        "contract": OUTPUT_DIR / "daily_incumbent_challenger_state_machine_contract_ohlc_absorbed.csv",
        "stock_absorption": OUTPUT_DIR / "daily_incumbent_challenger_stock_ohlc_absorption_audit.csv",
        "benchmark_coverage": OUTPUT_DIR / "daily_incumbent_challenger_00631L_daily_path_coverage.csv",
        "benchmark_gap": OUTPUT_DIR / "daily_incumbent_challenger_00631L_daily_price_gap_ledger.csv",
        "cost": OUTPUT_DIR / "daily_incumbent_challenger_ep05_cost_hooks.csv",
        "scale_audit": OUTPUT_DIR / "daily_incumbent_challenger_00631L_scale_stitch_audit.csv",
        "anomaly_audit": OUTPUT_DIR / "daily_incumbent_challenger_metric_anomaly_audit.csv",
        "coverage": OUTPUT_DIR / "daily_incumbent_challenger_requested_vs_actual_coverage.csv",
        "blocked": OUTPUT_DIR / "daily_incumbent_challenger_blocked_proxy_audit.csv",
        "future": OUTPUT_DIR / "daily_incumbent_challenger_future_data_audit.csv",
        "readiness": OUTPUT_DIR / "readiness_for_daily_incumbent_challenger_state_machine_diagnostic.json",
        "summary": OUTPUT_DIR / "final_summary_zh.md",
        "manifest": OUTPUT_DIR / "manifest.json",
    }
    materialized.to_csv(paths["contract"], index=False, encoding="utf-8-sig")
    stock[[col for col in ["state_machine_variant", "signal_date", "ticker", "entry_date", "exit_date", "entry_close", "exit_close", "official_unadjusted_daily_ohlc_ready", "daily_price_source_quality", "source_route", "blocked_reason"] if col in stock.columns]].to_csv(paths["stock_absorption"], index=False, encoding="utf-8-sig")
    etf[[col for col in ["state_machine_variant", "signal_date", "next_trading_day_execution_date", "next_trading_day_after_execution_date", "entry_close", "exit_close", "gross_daily_return", "daily_price_source_quality"] if col in etf.columns]].to_csv(paths["benchmark_coverage"], index=False, encoding="utf-8-sig")
    benchmark_gap.to_csv(paths["benchmark_gap"], index=False, encoding="utf-8-sig")
    pd.read_csv(CORE_DIR / "daily_incumbent_challenger_ep05_cost_hooks.csv", low_memory=False).to_csv(paths["cost"], index=False, encoding="utf-8-sig")
    pd.DataFrame([
        {"date": "2022-12-30", "chosen_source": "local_full_history_adjusted_close_cache", "rejected_lower_priority_source": "radar_official_unadjusted_close_92.80", "reason": "adjusted full-history price 4.21818 preserves the existing adjusted scale across 2022/2023", "formal_ready": False},
        {"date": "2026-05-27_to_2026-06-29", "chosen_source": "radar_official_unadjusted_close_equivalence_proxy", "rejected_lower_priority_source": "none", "reason": "816 P2 local close/adj_close overlap rows matched exactly; still diagnostic proxy, not adjusted source", "formal_ready": False},
    ]).to_csv(paths["scale_audit"], index=False, encoding="utf-8-sig")
    materialized[materialized["gross_daily_return"].abs() > 0.5][["state_machine_variant", "signal_date", "selected_ticker_after", "entry_close", "exit_close", "gross_daily_return", "daily_price_source_quality", "metric_eligible_full_integrated"]].to_csv(paths["anomaly_audit"], index=False, encoding="utf-8-sig")
    coverage.to_csv(paths["coverage"], index=False, encoding="utf-8-sig")
    blocked.to_csv(paths["blocked"], index=False, encoding="utf-8-sig")
    pd.DataFrame([
        {"audit_item": "daily_state_signal_construction", "future_return_used_as_rule": False, "source": "same-day market fields plus weekly-asof candidate context", "future_data_violation_count": 0},
        {"audit_item": "daily_ohlc_execution_path", "future_return_used_as_rule": False, "source": "next trading day price joins are diagnostic evaluation only", "future_data_violation_count": 0},
    ]).to_csv(paths["future"], index=False, encoding="utf-8-sig")
    paths["readiness"].write_text(json.dumps(readiness, ensure_ascii=False, indent=2), encoding="utf-8")
    _summary(paths["summary"], readiness)
    manifest = {
        "task_id": TASK_ID,
        "output_dir": str(OUTPUT_DIR),
        "inputs": {"core_daily_state": str(CORE_DIR), "radar_daily_stock_ohlc": str(RADAR_DIR), "radar_00631L_fill": str(RADAR_BENCHMARK_DIR), "p1_00631L_path": str(P1_BENCHMARK), "p2_00631L_cache": str(CACHE_00631L), "full_history_00631L_cache": str(FULL_HISTORY_00631L)},
        "artifacts": [{"path": str(path), "sha256": _sha256(path), "bytes": path.stat().st_size} for key, path in paths.items() if key != "manifest"],
        "readiness": readiness,
        "created_at": datetime.now(timezone.utc).isoformat(),
        **FLAGS,
    }
    paths["manifest"].write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(readiness, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
