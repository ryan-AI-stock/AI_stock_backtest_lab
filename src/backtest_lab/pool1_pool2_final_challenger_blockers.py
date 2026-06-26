from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from backtest_lab.data import load_price_csv
from backtest_lab.pool1_pool2_final_challenger_robustness import CANDIDATES, MAIN_CANDIDATE, SECONDARY_CANDIDATE, _benchmark_return, _round, _round_pct, _same_date_range, _sum
from backtest_lab.pool1_pool2_veto_cap_downweight import HORIZONS, _price_on_or_before, _text


DEFAULT_SOURCE_DIR = "outputs/pool1_pool2_veto_cap_downweight_panels_20260626"
DEFAULT_ROBUSTNESS_DIR = "outputs/pool1_pool2_final_challenger_robustness_panels_20260626"
DEFAULT_TARGET_DROP_SOURCE = "outputs/three_pool_vs_pool1_comparison_panels_20260626/target_drop_from_top3_diagnostics.csv"
DEFAULT_PRICE_CACHE_DIR = "backtest_cache/stock_pool_triad_v1_corrected"
DEFAULT_OUTPUT_DIR = "outputs/pool1_pool2_final_challenger_blocker_panels_20260626"
BENCHMARKS = {"0050": "0050.TW", "00631L": "00631L.TW", "0050x2": "0050.TW"}


def run_pool1_pool2_final_challenger_blockers(
    *,
    source_dir: str | Path = DEFAULT_SOURCE_DIR,
    robustness_dir: str | Path = DEFAULT_ROBUSTNESS_DIR,
    target_drop_source_path: str | Path = DEFAULT_TARGET_DROP_SOURCE,
    price_cache_dir: str | Path = DEFAULT_PRICE_CACHE_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    run_log: list[dict[str, str]] = []

    def log(step: str, status: str, detail: str = "") -> None:
        run_log.append({"timestamp": pd.Timestamp.now(tz="Asia/Taipei").strftime("%Y-%m-%d %H:%M:%S%z"), "step": step, "status": status, "detail": detail})
        pd.DataFrame(run_log).to_csv(output / "run_log.csv", index=False, encoding="utf-8-sig")
        (output / "current_step.txt").write_text(step, encoding="utf-8")

    try:
        source = Path(source_dir)
        robustness = Path(robustness_dir)
        log("load_inputs", "started", str(source))
        daily = pd.read_csv(source / "daily_equity_by_variant.csv").fillna("")
        trades = pd.read_csv(source / "trade_ledger_by_variant.csv").fillna("")
        events = pd.read_csv(source / "pool2_disagreement_variant_events.csv").fillna("")
        trigger = pd.read_csv(robustness / "cap40_trigger_event_panel.csv").fillna("")
        target_drop_source = _read_optional_csv(Path(target_drop_source_path))
        prices = _load_prices(Path(price_cache_dir), daily, events)
        _validate_inputs(daily, trades, events)
        candidate_daily = daily[daily["variant"].isin(CANDIDATES)].copy()
        latest = str(candidate_daily["date"].max())

        log("build_reports", "started", "")
        same_next = _same_day_vs_next_day(candidate_daily, prices)
        next_day = _next_day_fill_ledger(candidate_daily, events, prices)
        next_status = _next_day_status(next_day)
        entry = _entry_without_exit(candidate_daily, events)
        target_drop = _target_drop_from_top3(candidate_daily, target_drop_source)
        stability = _target_stability_summary(entry, target_drop, candidate_daily)
        hard_gate = _hard_gate_2024_attribution(candidate_daily, prices)
        hard_choice = _hard_gate_target_choice(candidate_daily, events)
        hard_trade_diff = _hard_gate_vs_0050x2_trade_diff(candidate_daily, prices)
        hard_missed = _hard_gate_missed_upside(candidate_daily, prices)
        cap_trigger = trigger.copy()
        cap_missed = _cap40_missed_upside(trigger, prices)
        counterfactual = _cap40_counterfactual(candidate_daily)
        pool2_attr = _pool2_confirmation_attribution(events)
        cost_attr = _cost_turnover_attribution(candidate_daily)
        holding_attr = _holding_days_attribution(candidate_daily)
        readiness = _blocker_readiness(next_status, stability, hard_gate, cap_missed)

        log("write_outputs", "started", "")
        _candidate_matrix().to_csv(output / "candidate_parameter_matrix.csv", index=False, encoding="utf-8-sig")
        same_next.to_csv(output / "same_day_vs_next_day_ledger.csv", index=False, encoding="utf-8-sig")
        next_day.to_csv(output / "next_day_fill_ledger.csv", index=False, encoding="utf-8-sig")
        next_status.to_csv(output / "next_day_fill_blocked_or_completed.csv", index=False, encoding="utf-8-sig")
        entry.to_csv(output / "entry_without_exit_recomputed.csv", index=False, encoding="utf-8-sig")
        target_drop.to_csv(output / "target_drop_from_top3_recomputed.csv", index=False, encoding="utf-8-sig")
        stability.to_csv(output / "target_stability_recomputed_summary.csv", index=False, encoding="utf-8-sig")
        hard_gate.to_csv(output / "hard_gate_2024_attribution.csv", index=False, encoding="utf-8-sig")
        hard_choice.to_csv(output / "hard_gate_2024_target_choice_attribution.csv", index=False, encoding="utf-8-sig")
        hard_trade_diff.to_csv(output / "hard_gate_2024_vs_0050x2_trade_diff.csv", index=False, encoding="utf-8-sig")
        hard_missed.to_csv(output / "hard_gate_2024_missed_upside_events.csv", index=False, encoding="utf-8-sig")
        cap_trigger.to_csv(output / "cap40_trigger_event_panel.csv", index=False, encoding="utf-8-sig")
        cap_missed.to_csv(output / "cap40_missed_upside_attribution.csv", index=False, encoding="utf-8-sig")
        counterfactual.to_csv(output / "cap40_actual_vs_no_cap_counterfactual.csv", index=False, encoding="utf-8-sig")
        pool2_attr.to_csv(output / "pool2_confirmation_attribution.csv", index=False, encoding="utf-8-sig")
        cost_attr.to_csv(output / "cost_turnover_attribution.csv", index=False, encoding="utf-8-sig")
        holding_attr.to_csv(output / "holding_days_attribution.csv", index=False, encoding="utf-8-sig")
        readiness.to_csv(output / "blocker_readiness_matrix.csv", index=False, encoding="utf-8-sig")
        (output / "pool1_pool2_final_challenger_blocker_summary_zh.md").write_text(_summary_markdown(readiness, hard_gate, cap_missed), encoding="utf-8")
        manifest = {
            "schema_version": 1,
            "task_id": "TASK-BACKTEST-CORE-POOL1-POOL2-FINAL-CHALLENGER-BLOCKER-PANELS-001",
            "model": "pool1_pool2_final_challenger_blocker_panels",
            "status": "completed",
            "latest_complete_common_date": latest,
            "formal_model_changed": False,
            "trade_decision_changed": False,
            "formal_absorption_ready": False,
            "pool3_shadow_used_as_formal": False,
            "report_only_labels_used_in_performance": False,
            "rr_partial_switch_used_in_performance": False,
            "uses_forward_return_as_rule": False,
            "valuation_used": False,
            "h3_used": False,
            "benchmarks_include_0050x2": True,
            "next_day_ledger_mixed_with_same_day": False,
            "same_date_range_for_candidates": _same_date_range(candidate_daily),
            "same_cost_model_for_candidates": True,
        }
        (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        pd.DataFrame([{"status": "completed", "output_dir": str(output.resolve())}]).to_csv(output / "completed.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(columns=["step", "error"]).to_csv(output / "failed.csv", index=False, encoding="utf-8-sig")
        log("completed", "completed", str(output.resolve()))
        (output / "current_step.txt").write_text("completed\n", encoding="utf-8")
        return output
    except Exception as exc:
        pd.DataFrame([{"step": "run_pool1_pool2_final_challenger_blockers", "error": str(exc)}]).to_csv(output / "failed.csv", index=False, encoding="utf-8-sig")
        log("failed", "failed", str(exc))
        raise


def _validate_inputs(daily: pd.DataFrame, trades: pd.DataFrame, events: pd.DataFrame) -> None:
    for name, frame, required in (
        ("daily", daily, {"variant", "date", "equity", "position_ticker", "action", "target_weights"}),
        ("events", events, {"variant", "date", "pool1_vote", "pool2_vote", "target_weights", "event_reason"}),
    ):
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{name} missing columns: {missing}")


def _read_optional_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path).fillna("") if path.exists() else pd.DataFrame()


def _load_prices(price_cache_dir: Path, daily: pd.DataFrame, events: pd.DataFrame) -> dict[str, pd.Series]:
    tickers = set(BENCHMARKS.values())
    for column in ("position_ticker",):
        tickers.update(_text(value) for value in daily[column].tolist() if _text(value) and _text(value) != "cash")
    for column in ("pool1_vote", "pool2_vote"):
        tickers.update(_text(value) for value in events[column].tolist() if _text(value))
    output: dict[str, pd.Series] = {}
    for ticker in sorted(tickers):
        path = price_cache_dir / f"{ticker}.csv"
        if not path.exists():
            path = price_cache_dir / f"{ticker.replace('.', '_')}.csv"
        if path.exists():
            output[ticker] = load_price_csv(path)["close"]
    return output


def _candidate_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"candidate": MAIN_CANDIDATE, "role": "main", "formal_absorption_ready": False},
            {"candidate": SECONDARY_CANDIDATE, "role": "high_return_sensitivity", "formal_absorption_ready": False},
        ]
    )


def _same_day_vs_next_day(daily: pd.DataFrame, prices: dict[str, pd.Series]) -> pd.DataFrame:
    rows = []
    for item in daily[daily["action"].astype(str).ne("hold")].to_dict(orient="records"):
        ticker = _text(item.get("position_ticker"))
        date = _text(item.get("date"))
        next_date, next_price = _next_trade_price(prices.get(ticker), date)
        same_price = _price_on_or_before(prices.get(ticker), date) if ticker in prices else None
        rows.append({"candidate": item["variant"], "same_day_date": date, "next_day_date": next_date, "target": ticker, "same_day_price": _round(same_price), "next_day_price": _round(next_price), "same_day_equity": item.get("equity", ""), "next_day_fill_status": "completed" if next_price is not None else "blocked_missing_next_price", "next_day_ledger_mixed_with_same_day": False})
    return pd.DataFrame(rows)


def _next_day_fill_ledger(daily: pd.DataFrame, events: pd.DataFrame, prices: dict[str, pd.Series]) -> pd.DataFrame:
    event_map = {(row["variant"], str(row["date"])): row for row in events.to_dict(orient="records")}
    rows = []
    for item in daily[daily["action"].astype(str).ne("hold")].to_dict(orient="records"):
        ticker = _text(item.get("position_ticker"))
        date = _text(item.get("date"))
        next_date, next_price = _next_trade_price(prices.get(ticker), date)
        event = event_map.get((item["variant"], date), {})
        rows.append({"candidate": item["variant"], "signal_date": date, "fill_date": next_date, "target": ticker, "next_day_fill_price": _round(next_price), "target_weights": event.get("target_weights", item.get("target_weights", "")), "event_reason": event.get("event_reason", ""), "status": "completed" if next_price is not None else "blocked_missing_next_price", "next_day_ledger_mixed_with_same_day": False})
    return pd.DataFrame(rows)


def _next_day_status(ledger: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for candidate, group in ledger.groupby("candidate", dropna=False):
        rows.append({"candidate": candidate, "event_count": len(group), "completed_count": int(group["status"].eq("completed").sum()), "blocked_count": int(group["status"].astype(str).str.startswith("blocked").sum()), "next_day_ledger_mixed_with_same_day": False})
    return pd.DataFrame(rows)


def _entry_without_exit(daily: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    event_map = {(row["variant"], str(row["date"])): row for row in events.to_dict(orient="records")}
    rows = []
    for item in daily.to_dict(orient="records"):
        event = event_map.get((item["variant"], str(item["date"])), {})
        pool1 = _text(event.get("pool1_vote"))
        pool2 = _text(event.get("pool2_vote"))
        flag = bool(pool1 and pool2 and pool1 != pool2 and _text(item.get("position_ticker")) == pool1)
        rows.append({"candidate": item["variant"], "date": item["date"], "period": item.get("period", ""), "target": item.get("position_ticker", ""), "pool1_vote": pool1, "pool2_vote": pool2, "entry_without_exit_confirmation": flag, "entry_without_exit_confirmation_outcome": "needs_experiments_forward_validation" if flag else ""})
    return pd.DataFrame(rows)


def _target_drop_from_top3(daily: pd.DataFrame, source: pd.DataFrame) -> pd.DataFrame:
    base = daily[["variant", "date", "period", "position_ticker"]].rename(columns={"variant": "candidate", "position_ticker": "target"}).copy()
    columns = ["target_in_top3_today", "target_drop_from_top3_next_1d", "target_drop_from_top3_next_2d", "target_drop_from_top3_next_3d", "target_reappears_in_top3_within_5d"]
    if source.empty or "date" not in source.columns:
        for column in columns:
            base[column] = False
        base["target_drop_source_status"] = "missing_source"
        return base
    available = source[["date"] + [column for column in columns if column in source.columns]].copy()
    merged = base.merge(available, on="date", how="left")
    for column in columns:
        if column not in merged.columns:
            merged[column] = False
        merged[column] = merged[column].map(_truthy)
    merged["target_drop_source_status"] = "reused_three_pool_vs_pool1_comparison_panel"
    return merged


def _target_stability_summary(entry: pd.DataFrame, target_drop: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for candidate, group in daily.groupby("variant"):
        entry_group = entry[entry["candidate"].eq(candidate)]
        drop_group = target_drop[target_drop["candidate"].eq(candidate)]
        rows.append({"candidate": candidate, "entry_without_exit_count": int(entry_group["entry_without_exit_confirmation"].sum()), "entry_without_exit_rate": _mean_bool(entry_group["entry_without_exit_confirmation"]), "target_drop_1d_rate": _mean_bool(drop_group["target_drop_from_top3_next_1d"]), "target_drop_2d_rate": _mean_bool(drop_group["target_drop_from_top3_next_2d"]), "target_drop_3d_rate": _mean_bool(drop_group["target_drop_from_top3_next_3d"]), "target_reappears_in_top3_within_5d_rate": _mean_bool(drop_group["target_reappears_in_top3_within_5d"]), "target_changed_within_1d_rate": _changed_rate(group, 1), "target_changed_within_3d_rate": _changed_rate(group, 3), "rapid_flip_same_target_window_1_3d_rate": _rapid_flip_rate(group), "possible_execution_layer_issue_rate": _changed_rate(group, 3)})
    return pd.DataFrame(rows)


def _hard_gate_2024_attribution(daily: pd.DataFrame, prices: dict[str, pd.Series]) -> pd.DataFrame:
    frame = daily[(pd.to_datetime(daily["date"]) >= pd.Timestamp("2024-01-01")) & (pd.to_datetime(daily["date"]) <= pd.Timestamp("2024-12-31"))].copy()
    rows = []
    bench = _benchmark_return(prices["0050.TW"], "2024-01-02", "2024-12-31", 2.0)
    for candidate, group in frame.groupby("variant"):
        start = float(group["equity"].iloc[0])
        end = float(group["equity"].iloc[-1])
        ret = end / start - 1 if start else 0.0
        rows.append({"candidate": candidate, "period": "2024_hard_gate", "candidate_return": round(ret, 8), "0050x2_benchmark_return": _round(bench), "excess_vs_0050x2": _round(ret - bench if bench is not None else None), "target_choice_days": int(group["position_ticker"].astype(str).ne("cash").sum()), "00631L_days": int(group["position_ticker"].eq("00631L.TW").sum()), "cash_or_fallback_days": int(group["position_ticker"].astype(str).eq("cash").sum()), "total_cost": _sum(group, "transaction_cost"), "total_turnover": _sum(group, "turnover")})
    return pd.DataFrame(rows)


def _hard_gate_target_choice(daily: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    frame = daily[(pd.to_datetime(daily["date"]) >= pd.Timestamp("2024-01-01")) & (pd.to_datetime(daily["date"]) <= pd.Timestamp("2024-12-31"))].copy()
    rows = []
    for (candidate, target), group in frame.groupby(["variant", "position_ticker"], dropna=False):
        rows.append({"candidate": candidate, "target": target, "days": len(group), "cost": _sum(group, "transaction_cost"), "turnover": _sum(group, "turnover"), "return_proxy_sum": round(float(pd.to_numeric(group["equity"], errors="coerce").pct_change().fillna(0).sum()), 8)})
    return pd.DataFrame(rows)


def _hard_gate_vs_0050x2_trade_diff(daily: pd.DataFrame, prices: dict[str, pd.Series]) -> pd.DataFrame:
    frame = daily[(pd.to_datetime(daily["date"]) >= pd.Timestamp("2024-01-01")) & (pd.to_datetime(daily["date"]) <= pd.Timestamp("2024-12-31")) & daily["action"].astype(str).ne("hold")].copy()
    rows = []
    for item in frame.to_dict(orient="records"):
        date = item["date"]
        for horizon in HORIZONS:
            target_ret = _forward_return(prices.get(item["position_ticker"]), date, horizon)
            bench = _forward_return(prices.get("0050.TW"), date, horizon)
            rows.append({"candidate": item["variant"], "date": date, "target": item["position_ticker"], "horizon": horizon, "target_forward_return": _round(target_ret), "0050x2_forward_return": _round(None if bench is None else bench * 2), "target_excess_vs_0050x2": _round(None if target_ret is None or bench is None else target_ret - bench * 2), "uses_forward_return_as_rule": False})
    return pd.DataFrame(rows)


def _hard_gate_missed_upside(daily: pd.DataFrame, prices: dict[str, pd.Series]) -> pd.DataFrame:
    diff = _hard_gate_vs_0050x2_trade_diff(daily, prices)
    if diff.empty:
        return diff
    values = pd.to_numeric(diff["target_excess_vs_0050x2"], errors="coerce")
    return diff[values < -0.1].copy()


def _cap40_missed_upside(trigger: pd.DataFrame, prices: dict[str, pd.Series]) -> pd.DataFrame:
    rows = []
    for item in trigger.to_dict(orient="records"):
        date = _text(item.get("date"))
        row: dict[str, Any] = {"candidate": MAIN_CANDIDATE, "date": date, "target_weights": item.get("target_weights", ""), "uses_forward_return_as_rule": False}
        for horizon in HORIZONS:
            ret_00631l = _forward_return(prices.get("00631L.TW"), date, horizon)
            ret_0050 = _forward_return(prices.get("0050.TW"), date, horizon)
            row[f"00631L_forward_{horizon}d_return"] = _round(ret_00631l)
            row[f"0050x2_forward_{horizon}d_return"] = _round(None if ret_0050 is None else ret_0050 * 2)
            row[f"missed_upside_gt20pct_{horizon}d"] = bool(ret_00631l is not None and ret_00631l > 0.20)
        rows.append(row)
    return pd.DataFrame(rows)


def _cap40_counterfactual(daily: pd.DataFrame) -> pd.DataFrame:
    main = daily[daily["variant"].eq(MAIN_CANDIDATE)].copy()
    no_cap = daily[daily["variant"].eq("pool1_pool2_disagree_confirmation_1")].copy()
    if no_cap.empty:
        no_cap = daily[daily["variant"].eq(SECONDARY_CANDIDATE)].copy()
        path = "fallback_secondary_confirmation2"
    else:
        path = "no_cap_confirmation1"
    rows = []
    merged = main[["date", "equity", "drawdown"]].merge(no_cap[["date", "equity", "drawdown"]], on="date", suffixes=("_actual_cap40", "_no_cap_counterfactual"))
    for item in merged.to_dict(orient="records"):
        rows.append({"candidate": MAIN_CANDIDATE, "date": item["date"], "counterfactual_path": path, "actual_cap40_equity": item["equity_actual_cap40"], "no_cap_counterfactual_equity": item["equity_no_cap_counterfactual"], "equity_diff_actual_minus_counterfactual": round(float(item["equity_actual_cap40"]) - float(item["equity_no_cap_counterfactual"]), 2), "actual_drawdown": item["drawdown_actual_cap40"], "counterfactual_drawdown": item["drawdown_no_cap_counterfactual"], "formal_model_changed": False})
    return pd.DataFrame(rows)


def _pool2_confirmation_attribution(events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for candidate in CANDIDATES:
        group = events[events["variant"].eq(candidate)].copy()
        rows.append({"candidate": candidate, "rows": len(group), "pool2_disagreement_rows": int(group["pool2_disagreement"].map(_truthy).sum()) if "pool2_disagreement" in group.columns else "", "confirmation_not_met_rows": int(group["event_reason"].astype(str).str.contains("confirmation", regex=False).sum()), "hard_veto_rows": int(group["event_reason"].astype(str).str.contains("hard_veto", regex=False).sum())})
    return pd.DataFrame(rows)


def _cost_turnover_attribution(daily: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([{"candidate": candidate, "total_cost": _sum(group, "transaction_cost"), "total_turnover": _sum(group, "turnover"), "trade_days": int(group["action"].astype(str).ne("hold").sum())} for candidate, group in daily.groupby("variant")])


def _holding_days_attribution(daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (candidate, target), group in daily.groupby(["variant", "position_ticker"], dropna=False):
        rows.append({"candidate": candidate, "target": target, "holding_days": len(group), "share": round(len(group) / len(daily[daily["variant"].eq(candidate)]), 6)})
    return pd.DataFrame(rows)


def _blocker_readiness(next_status: pd.DataFrame, stability: pd.DataFrame, hard_gate: pd.DataFrame, cap_missed: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"blocker": "next_day_fill_ledger", "status": "completed", "ready_for_experiments": True, "detail": f"events={int(next_status['event_count'].sum()) if not next_status.empty else 0}"},
            {"blocker": "entry_target_stability", "status": "completed", "ready_for_experiments": True, "detail": f"candidates={len(stability)}"},
            {"blocker": "2024_hard_gate_vs_0050x2", "status": "completed_needs_experiments_judgment", "ready_for_experiments": True, "detail": f"rows={len(hard_gate)}"},
            {"blocker": "cap40_missed_upside", "status": "completed_needs_experiments_judgment", "ready_for_experiments": True, "detail": f"trigger_rows={len(cap_missed)}"},
        ]
    )


def _summary_markdown(readiness: pd.DataFrame, hard_gate: pd.DataFrame, cap_missed: pd.DataFrame) -> str:
    lines = ["# Pool1 + Pool2 Final Challenger Blocker Panels", "", "本輸出只補 formal absorption 前 blocker evidence，不改正式模型。", ""]
    for row in readiness.to_dict(orient="records"):
        lines.append(f"- {row['blocker']}: {row['status']}")
    lines.append("")
    for row in hard_gate.to_dict(orient="records"):
        lines.append(f"- 2024 hard gate {row['candidate']}: excess vs 0050x2 {row.get('excess_vs_0050x2')}")
    lines.append(f"- cap40 missed upside rows: {len(cap_missed)}")
    return "\n".join(lines) + "\n"


def _next_trade_price(series: pd.Series | None, date: str) -> tuple[str, float | None]:
    if series is None:
        return "", None
    future = series[series.index > pd.Timestamp(date)]
    if future.empty:
        return "", None
    return pd.Timestamp(future.index[0]).strftime("%Y-%m-%d"), float(future.iloc[0])


def _forward_return(series: pd.Series | None, date: str, horizon: int) -> float | None:
    if series is None:
        return None
    start = _price_on_or_before(series, date)
    future = series[series.index > pd.Timestamp(date)]
    if start is None or len(future) < horizon:
        return None
    return float(future.iloc[horizon - 1] / start - 1)


def _changed_rate(group: pd.DataFrame, window: int) -> float:
    values = group.sort_values("date")["position_ticker"].astype(str).tolist()
    flags = [bool(value and any(item and item != value for item in values[index + 1 : index + window + 1])) for index, value in enumerate(values)]
    return round(sum(flags) / len(flags), 6) if flags else 0.0


def _rapid_flip_rate(group: pd.DataFrame) -> float:
    values = group.sort_values("date")["position_ticker"].astype(str).tolist()
    flags = [bool(value and value in values[index + 1 : index + 4] and any(item and item != value for item in values[index + 1 : index + 4])) for index, value in enumerate(values)]
    return round(sum(flags) / len(flags), 6) if flags else 0.0


def _mean_bool(series: pd.Series) -> float:
    if len(series) == 0:
        return 0.0
    return round(float(series.map(_truthy).mean()), 6)


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Pool1+Pool2 final challenger blocker panels.")
    parser.add_argument("--source-dir", default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--robustness-dir", default=DEFAULT_ROBUSTNESS_DIR)
    parser.add_argument("--target-drop-source-path", default=DEFAULT_TARGET_DROP_SOURCE)
    parser.add_argument("--price-cache-dir", default=DEFAULT_PRICE_CACHE_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    output = run_pool1_pool2_final_challenger_blockers(source_dir=args.source_dir, robustness_dir=args.robustness_dir, target_drop_source_path=args.target_drop_source_path, price_cache_dir=args.price_cache_dir, output_dir=args.output_dir)
    print(f"OUTPUT_DIR={output.resolve()}")


if __name__ == "__main__":
    main()
