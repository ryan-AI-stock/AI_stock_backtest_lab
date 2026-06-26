from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from backtest_lab.data import load_price_csv
from backtest_lab.pool1_pool2_veto_cap_downweight import BENCHMARKS, HORIZONS, _price_on_or_before, _text


DEFAULT_SOURCE_DIR = "outputs/pool1_pool2_veto_cap_downweight_panels_20260626"
DEFAULT_OUTPUT_DIR = "outputs/pool1_pool2_final_challenger_robustness_panels_20260626"
DEFAULT_PRICE_CACHE_DIR = "backtest_cache/stock_pool_triad_v1_corrected"
MAIN_CANDIDATE = "combined_cap40_confirmation1"
SECONDARY_CANDIDATE = "pool1_pool2_disagree_confirmation_2"
CANDIDATES = (MAIN_CANDIDATE, SECONDARY_CANDIDATE)
REFERENCE_VARIANTS = ("pool1_primary_no_overlay", "pool1_pool2_veto_no_cap")
BENCHMARK_TICKERS = {"0050": "0050.TW", "00631L": "00631L.TW", "0050x2": "0050.TW"}


def run_pool1_pool2_final_challenger_robustness(
    *,
    source_dir: str | Path = DEFAULT_SOURCE_DIR,
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
        log("load_inputs", "started", str(source))
        daily = pd.read_csv(source / "daily_equity_by_variant.csv").fillna("")
        trades = pd.read_csv(source / "trade_ledger_by_variant.csv").fillna("")
        events = pd.read_csv(source / "pool2_disagreement_variant_events.csv").fillna("")
        perf_source = pd.read_csv(source / "period_performance_by_variant.csv").fillna("")
        _validate_inputs(daily, trades, events)
        daily = daily[daily["variant"].isin((*CANDIDATES, *REFERENCE_VARIANTS))].copy()
        prices = _load_prices(Path(price_cache_dir))
        latest = str(daily["date"].max())

        log("build_reports", "started", "")
        candidate_matrix = _candidate_matrix()
        period_perf = _period_performance(daily)
        benchmark = _benchmark_comparison(daily, prices)
        oos = _oos_walk_forward(daily)
        leave_one = _leave_one_period(daily)
        maturity = _pre2026_vs_2026(daily)
        by_ticker = _contribution_by_ticker(daily)
        by_period = _contribution_by_month_quarter(daily)
        exclusion = _contribution_exclusion_tests(daily)
        exposure = _00631l_exposure_period_rolling(daily)
        trigger_panel = _cap40_trigger_event_panel(events)
        trigger_attr = _cap40_trigger_attribution(trigger_panel, daily, prices)
        execution = _execution_ledger(daily, trades)
        stability = _target_stability(daily)
        cost_sensitivity = _cost_sensitivity(daily)
        scorecard = _risk_adjusted_scorecard(period_perf, exposure, execution)
        readiness = _readiness_report(period_perf, benchmark, exposure, trigger_attr)

        log("write_outputs", "started", "")
        candidate_matrix.to_csv(output / "candidate_parameter_matrix.csv", index=False, encoding="utf-8-sig")
        period_perf.to_csv(output / "period_performance_by_candidate.csv", index=False, encoding="utf-8-sig")
        benchmark.to_csv(output / "benchmark_comparison_0050_00631L_0050x2.csv", index=False, encoding="utf-8-sig")
        oos.to_csv(output / "oos_walk_forward_by_candidate.csv", index=False, encoding="utf-8-sig")
        leave_one.to_csv(output / "leave_one_period_by_candidate.csv", index=False, encoding="utf-8-sig")
        maturity.to_csv(output / "pre2026_vs_2026_maturity.csv", index=False, encoding="utf-8-sig")
        by_ticker.to_csv(output / "contribution_by_ticker_candidate.csv", index=False, encoding="utf-8-sig")
        by_period.to_csv(output / "contribution_by_month_quarter_candidate.csv", index=False, encoding="utf-8-sig")
        exclusion.to_csv(output / "contribution_exclusion_tests_candidate.csv", index=False, encoding="utf-8-sig")
        exposure.to_csv(output / "00631L_exposure_period_rolling.csv", index=False, encoding="utf-8-sig")
        trigger_panel.to_csv(output / "cap40_trigger_event_panel.csv", index=False, encoding="utf-8-sig")
        trigger_attr.to_csv(output / "cap40_trigger_attribution.csv", index=False, encoding="utf-8-sig")
        execution.to_csv(output / "execution_ledger_by_candidate.csv", index=False, encoding="utf-8-sig")
        stability.to_csv(output / "target_stability_by_candidate.csv", index=False, encoding="utf-8-sig")
        cost_sensitivity.to_csv(output / "cost_sensitivity_by_candidate.csv", index=False, encoding="utf-8-sig")
        scorecard.to_csv(output / "risk_adjusted_scorecard.csv", index=False, encoding="utf-8-sig")
        (output / "formal_challenger_readiness_report.md").write_text(readiness, encoding="utf-8")
        (output / "pool1_pool2_final_challenger_summary_zh.md").write_text(_summary_markdown(period_perf, exposure), encoding="utf-8")
        manifest = {
            "schema_version": 1,
            "task_id": "TASK-BACKTEST-CORE-POOL1-POOL2-FINAL-CHALLENGER-ROBUSTNESS-PANELS-001",
            "model": "pool1_pool2_final_challenger_robustness",
            "status": "completed",
            "source_dir": str(source),
            "latest_complete_common_date": latest,
            "main_candidate": MAIN_CANDIDATE,
            "secondary_candidate": SECONDARY_CANDIDATE,
            "formal_model_changed": False,
            "trade_decision_changed": False,
            "formal_absorption_ready": False,
            "pool3_shadow_used_as_formal": False,
            "report_only_labels_used_in_performance": False,
            "rr_partial_switch_used_in_performance": False,
            "uses_forward_return_as_rule": False,
            "valuation_used": False,
            "h3_used": False,
            "cap_performance_recomputed": True,
            "benchmarks_include_0050x2": True,
            "same_date_range_for_candidates": _same_date_range(daily[daily["variant"].isin(CANDIDATES)]),
            "same_cost_model_for_candidates": True,
            "next_day_approximation_status": "blocked_not_mixed_with_same_day",
        }
        (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        pd.DataFrame([{"status": "completed", "output_dir": str(output.resolve())}]).to_csv(output / "completed.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(columns=["step", "error"]).to_csv(output / "failed.csv", index=False, encoding="utf-8-sig")
        log("completed", "completed", str(output.resolve()))
        (output / "current_step.txt").write_text("completed\n", encoding="utf-8")
        return output
    except Exception as exc:
        pd.DataFrame([{"step": "run_pool1_pool2_final_challenger_robustness", "error": str(exc)}]).to_csv(output / "failed.csv", index=False, encoding="utf-8-sig")
        log("failed", "failed", str(exc))
        raise


def _validate_inputs(daily: pd.DataFrame, trades: pd.DataFrame, events: pd.DataFrame) -> None:
    required_daily = {"variant", "date", "equity", "position_ticker", "action", "transaction_cost", "turnover", "target_weights"}
    required_events = {"variant", "date", "pool1_vote", "pool2_vote", "target_weights", "event_reason"}
    for name, frame, required in (("daily", daily, required_daily), ("events", events, required_events)):
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{name} missing columns: {missing}")
    if trades.empty:
        return
    missing_trades = sorted({"variant", "date", "ticker", "action", "gross_amount"} - set(trades.columns))
    if missing_trades:
        raise ValueError(f"trades missing columns: {missing_trades}")


def _load_prices(price_cache_dir: Path) -> dict[str, pd.Series]:
    output: dict[str, pd.Series] = {}
    for ticker in ("0050.TW", "00631L.TW"):
        path = price_cache_dir / f"{ticker}.csv"
        if not path.exists():
            path = price_cache_dir / f"{ticker.replace('.', '_')}.csv"
        output[ticker] = load_price_csv(path)["close"]
    return output


def _candidate_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"candidate": MAIN_CANDIDATE, "role": "main", "pool2_policy": "confirmation_1", "00631L_cap": 0.40, "formal_absorption_ready": False},
            {"candidate": SECONDARY_CANDIDATE, "role": "high_return_secondary", "pool2_policy": "confirmation_2", "00631L_cap": "", "formal_absorption_ready": False},
        ]
    )


def _period_performance(daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    periods = {"2022": ("2022-01-01", "2022-12-31"), "2023": ("2023-01-01", "2023-12-31"), "2024_now": ("2024-01-01", None), "2024_hard_gate": ("2024-01-01", "2024-12-31"), "2026_ytd": ("2026-01-01", None), "full": (None, None)}
    frame = daily.copy()
    frame["date_ts"] = pd.to_datetime(frame["date"])
    for variant, group in frame.groupby("variant"):
        for label, (start, end) in periods.items():
            subset = group
            if start:
                subset = subset[subset["date_ts"] >= pd.Timestamp(start)]
            if end:
                subset = subset[subset["date_ts"] <= pd.Timestamp(end)]
            rows.append(_perf_row(variant, label, subset))
    return pd.DataFrame(rows)


def _benchmark_comparison(daily: pd.DataFrame, prices: dict[str, pd.Series]) -> pd.DataFrame:
    perf = _period_performance(daily)
    rows = []
    date_ranges = perf[["variant", "period_label", "start_date", "end_date"]].dropna()
    for item in date_ranges.to_dict(orient="records"):
        candidate_row = perf[(perf["variant"].eq(item["variant"])) & (perf["period_label"].eq(item["period_label"]))].iloc[0].to_dict()
        for label, ticker in BENCHMARK_TICKERS.items():
            multiplier = 2.0 if label == "0050x2" else 1.0
            bench_return = _benchmark_return(prices[ticker], item["start_date"], item["end_date"], multiplier)
            rows.append(
                {
                    "candidate": item["variant"],
                    "period_label": item["period_label"],
                    "benchmark": label,
                    "candidate_return_pct": candidate_row.get("return_pct", ""),
                    "benchmark_return_pct": _round_pct(bench_return),
                    "excess_return_pct": _round_pct((candidate_row.get("return_pct", 0) / 100) - bench_return if bench_return is not None else None),
                    "same_date_range": True,
                }
            )
    return pd.DataFrame(rows)


def _oos_walk_forward(daily: pd.DataFrame) -> pd.DataFrame:
    tests = {"train_2022_2024": ("2022-01-01", "2024-12-31"), "test_2025_2026": ("2025-01-01", None), "train_2022_2023": ("2022-01-01", "2023-12-31"), "test_2024_now": ("2024-01-01", None), "train_2022_2025": ("2022-01-01", "2025-12-31"), "test_2026_ytd": ("2026-01-01", None), "post_2026_exclusion": ("2022-01-01", "2025-12-31")}
    rows = []
    frame = daily[daily["variant"].isin(CANDIDATES)].copy()
    frame["date_ts"] = pd.to_datetime(frame["date"])
    for variant, group in frame.groupby("variant"):
        for label, (start, end) in tests.items():
            subset = group[group["date_ts"] >= pd.Timestamp(start)]
            if end:
                subset = subset[subset["date_ts"] <= pd.Timestamp(end)]
            rows.append(_perf_row(variant, label, subset))
    return pd.DataFrame(rows)


def _leave_one_period(daily: pd.DataFrame) -> pd.DataFrame:
    frame = daily[daily["variant"].isin(CANDIDATES)].copy()
    frame["date_ts"] = pd.to_datetime(frame["date"])
    frame["year"] = frame["date_ts"].dt.year.astype(str)
    frame["quarter"] = frame["date_ts"].dt.to_period("Q").astype(str)
    frame["month"] = frame["date_ts"].dt.to_period("M").astype(str)
    rows = []
    for variant, group in frame.groupby("variant"):
        for level in ("year", "quarter", "month"):
            values = sorted(group[level].unique()) if level == "year" else group[level].value_counts().head(16).index.tolist()
            for value in values:
                row = _perf_row(variant, f"leave_one_{level}_{value}", group[~group[level].eq(value)])
                row["excluded_level"] = level
                row["excluded_value"] = value
                rows.append(row)
    return pd.DataFrame(rows)


def _pre2026_vs_2026(daily: pd.DataFrame) -> pd.DataFrame:
    perf = _period_performance(daily[daily["variant"].isin(CANDIDATES)])
    return perf[perf["period_label"].isin(["full", "2026_ytd"])].copy()


def _contribution_by_ticker(daily: pd.DataFrame) -> pd.DataFrame:
    frame = daily[daily["variant"].isin(CANDIDATES)].copy()
    frame["daily_return"] = frame.groupby("variant")["equity"].pct_change().fillna(0)
    rows = []
    for (variant, ticker), group in frame.groupby(["variant", "position_ticker"], dropna=False):
        rows.append({"candidate": variant, "ticker": ticker, "position_days": len(group), "return_contribution_sum": round(float(group["daily_return"].sum()), 8), "mean_daily_return": round(float(group["daily_return"].mean()), 8)})
    return pd.DataFrame(rows).sort_values(["candidate", "return_contribution_sum"], ascending=[True, False])


def _contribution_by_month_quarter(daily: pd.DataFrame) -> pd.DataFrame:
    frame = daily[daily["variant"].isin(CANDIDATES)].copy()
    frame["date_ts"] = pd.to_datetime(frame["date"])
    frame["daily_return"] = frame.groupby("variant")["equity"].pct_change().fillna(0)
    rows = []
    for level, series in {"month": frame["date_ts"].dt.to_period("M").astype(str), "quarter": frame["date_ts"].dt.to_period("Q").astype(str)}.items():
        frame[level] = series
        for (variant, value), group in frame.groupby(["variant", level], dropna=False):
            rows.append({"candidate": variant, "period_level": level, "period_value": value, "row_count": len(group), "return_contribution_sum": round(float(group["daily_return"].sum()), 8)})
    return pd.DataFrame(rows).sort_values(["candidate", "period_level", "return_contribution_sum"], ascending=[True, True, False])


def _contribution_exclusion_tests(daily: pd.DataFrame) -> pd.DataFrame:
    frame = daily[daily["variant"].isin(CANDIDATES)].copy()
    frame["date_ts"] = pd.to_datetime(frame["date"])
    frame["month"] = frame["date_ts"].dt.to_period("M").astype(str)
    frame["quarter"] = frame["date_ts"].dt.to_period("Q").astype(str)
    rows = []
    for variant, group in frame.groupby("variant"):
        top_tickers = group["position_ticker"].value_counts().head(3).index.tolist()
        top_month = _top_contribution_period(group, "month")
        top_quarter = _top_contribution_period(group, "quarter")
        tests = [
            ("exclude_00631L", group[~group["position_ticker"].eq("00631L.TW")]),
            ("exclude_top_ticker", group[~group["position_ticker"].eq(top_tickers[0])] if top_tickers else group.iloc[0:0]),
            ("exclude_top_3_tickers", group[~group["position_ticker"].isin(top_tickers)]),
            (f"exclude_top_month_{top_month}", group[~group["month"].eq(top_month)]),
            (f"exclude_top_quarter_{top_quarter}", group[~group["quarter"].eq(top_quarter)]),
            ("exclude_2026Q2", group[~group["quarter"].eq("2026Q2")]),
        ]
        rows.extend(_perf_row(variant, label, subset) for label, subset in tests)
    return pd.DataFrame(rows)


def _00631l_exposure_period_rolling(daily: pd.DataFrame) -> pd.DataFrame:
    frame = daily[daily["variant"].isin(CANDIDATES)].copy()
    frame["date_ts"] = pd.to_datetime(frame["date"])
    frame["year"] = frame["date_ts"].dt.year.astype(str)
    rows = []
    for variant, group in frame.groupby("variant"):
        for label, subset in [("full", group), *[(f"year_{year}", year_group) for year, year_group in group.groupby("year")]]:
            rows.append(_exposure_row(variant, label, subset))
        ordered = group.sort_values("date_ts").copy()
        for window in (60, 120):
            exposure = ordered["position_ticker"].eq("00631L.TW").rolling(window, min_periods=1).mean()
            rows.append({"candidate": variant, "period_label": f"rolling_{window}d_max", "row_count": len(ordered), "00631L_position_day_share": round(float(exposure.max()), 6), "rolling_window": window})
    return pd.DataFrame(rows)


def _cap40_trigger_event_panel(events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    frame = events[events["variant"].eq(MAIN_CANDIDATE)].copy()
    for item in frame.to_dict(orient="records"):
        weights_text = _text(item.get("target_weights"))
        trigger = "00631L.TW" in weights_text and "0.4" in weights_text
        if trigger:
            rows.append({"candidate": MAIN_CANDIDATE, "date": item.get("date", ""), "period": item.get("period", ""), "pool1_vote": item.get("pool1_vote", ""), "pool2_vote": item.get("pool2_vote", ""), "event_reason": item.get("event_reason", ""), "target_weights": weights_text, "cap40_triggered": True, "trigger_reason": "00631L weight capped at 40%"})
    return pd.DataFrame(rows)


def _cap40_trigger_attribution(trigger: pd.DataFrame, daily: pd.DataFrame, prices: dict[str, pd.Series]) -> pd.DataFrame:
    rows = []
    if trigger.empty:
        return pd.DataFrame(columns=["candidate", "event_count"])
    main = daily[daily["variant"].eq(MAIN_CANDIDATE)][["date", "equity", "drawdown"]].copy()
    for item in trigger.to_dict(orient="records"):
        date = item["date"]
        row = {"candidate": MAIN_CANDIDATE, "date": date, "pool1_vote": item.get("pool1_vote", ""), "pool2_vote": item.get("pool2_vote", ""), "target_weights": item.get("target_weights", ""), "uses_forward_return_as_rule": False}
        match = main[main["date"].astype(str).eq(str(date))]
        row["candidate_equity_on_trigger"] = "" if match.empty else float(match["equity"].iloc[0])
        row["candidate_drawdown_on_trigger"] = "" if match.empty else float(match["drawdown"].iloc[0])
        for horizon in HORIZONS:
            ret = _forward_return(prices["00631L.TW"], date, horizon)
            row[f"00631L_forward_{horizon}d_return"] = _round(ret)
        rows.append(row)
    return pd.DataFrame(rows)


def _execution_ledger(daily: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant in CANDIDATES:
        group = daily[daily["variant"].eq(variant)].copy()
        variant_trades = trades[trades["variant"].eq(variant)].copy() if not trades.empty else pd.DataFrame()
        rows.append({"candidate": variant, "execution_mode": "same_day", "status": "completed", "trade_days": int(group["action"].astype(str).ne("hold").sum()), "trade_count": int(len(variant_trades)), "total_turnover": _sum(group, "turnover"), "total_transaction_cost": _sum(group, "transaction_cost"), "rapid_flip_rate": _rapid_flip_rate(group), "target_changed_1d_rate": _changed_rate(group, 1), "target_changed_3d_rate": _changed_rate(group, 3)})
        rows.append({"candidate": variant, "execution_mode": "next_day_approximation", "status": "blocked", "blocked_reason": "next-day approximation requires shifted fill price ledger; not mixed with same-day result"})
    return pd.DataFrame(rows)


def _target_stability(daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant in CANDIDATES:
        group = daily[daily["variant"].eq(variant)].copy()
        rows.append({"candidate": variant, "row_count": len(group), "rapid_flip_same_target_window_1_3d_rate": _rapid_flip_rate(group), "target_changed_within_1d_rate": _changed_rate(group, 1), "target_changed_within_3d_rate": _changed_rate(group, 3), "target_drop_from_top3_status": "blocked_source_not_in_candidate_panel", "entry_without_exit_status": "not_recomputed_in_final_challenger_panel"})
    return pd.DataFrame(rows)


def _cost_sensitivity(daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant in CANDIDATES:
        group = daily[daily["variant"].eq(variant)].copy()
        turnover = _sum(group, "turnover")
        base_cost = _sum(group, "transaction_cost")
        for label, bp in (("base_cost", 0.0), ("plus_10bp_turnover", 0.001), ("plus_20bp_turnover", 0.002), ("half_cost", -0.5)):
            rows.append({"candidate": variant, "cost_scenario": label, "base_transaction_cost": base_cost, "turnover": turnover, "additional_cost_proxy": round(turnover * bp if bp >= 0 else base_cost * bp, 2), "diagnostic_only": True})
    return pd.DataFrame(rows)


def _risk_adjusted_scorecard(perf: pd.DataFrame, exposure: pd.DataFrame, execution: pd.DataFrame) -> pd.DataFrame:
    full = perf[perf["period_label"].eq("full")].copy()
    rows = []
    for item in full.to_dict(orient="records"):
        variant = item["variant"]
        if variant not in CANDIDATES:
            continue
        exp = exposure[(exposure["candidate"].eq(variant)) & (exposure["period_label"].eq("full"))]
        exe = execution[(execution["candidate"].eq(variant)) & (execution["execution_mode"].eq("same_day"))]
        rows.append({"candidate": variant, "return_pct": item.get("return_pct", ""), "max_drawdown_pct": item.get("max_drawdown_pct", ""), "00631L_full_share": "" if exp.empty else exp["00631L_position_day_share"].iloc[0], "trade_days": "" if exe.empty else exe["trade_days"].iloc[0], "formal_absorption_ready": False})
    return pd.DataFrame(rows)


def _readiness_report(perf: pd.DataFrame, benchmark: pd.DataFrame, exposure: pd.DataFrame, trigger: pd.DataFrame) -> str:
    lines = ["# Pool1 + Pool2 Final Challenger Readiness", "", "本輸出是 final challenger evidence，不改正式模型，也不代表 formal absorption。", ""]
    full = perf[(perf["variant"].isin(CANDIDATES)) & (perf["period_label"].eq("full"))]
    for item in full.to_dict(orient="records"):
        lines.append(f"- {item['variant']}: full return {item.get('return_pct')}%, MDD {item.get('max_drawdown_pct')}%")
    lines.extend(["", f"- cap40 trigger events: {len(trigger)}", "- formal_absorption_ready=false", "- next_day_approximation=blocked_not_mixed_with_same_day", ""])
    return "\n".join(lines)


def _summary_markdown(perf: pd.DataFrame, exposure: pd.DataFrame) -> str:
    lines = ["# Pool1 + Pool2 Final Challenger Robustness Summary", "", "本輸出只供 Experiments 驗證 final challenger robustness。", ""]
    full = perf[(perf["variant"].isin(CANDIDATES)) & (perf["period_label"].eq("full"))]
    for item in full.to_dict(orient="records"):
        lines.append(f"- {item['variant']}: full {item.get('return_pct')}%, MDD {item.get('max_drawdown_pct')}%")
    full_exposure = exposure[exposure["period_label"].eq("full")]
    for item in full_exposure.to_dict(orient="records"):
        lines.append(f"- {item['candidate']} 00631L full share: {item.get('00631L_position_day_share')}")
    return "\n".join(lines) + "\n"


def _perf_row(variant: str, label: str, frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {"variant": variant, "period_label": label, "status": "empty"}
    equity = pd.to_numeric(frame["equity"], errors="coerce")
    start = float(equity.iloc[0])
    end = float(equity.iloc[-1])
    return {"variant": variant, "period_label": label, "status": "completed", "start_date": frame["date"].iloc[0], "end_date": frame["date"].iloc[-1], "start_equity": round(start, 2), "final_equity": round(end, 2), "return_pct": round((end / start - 1) * 100, 4) if start else 0.0, "max_drawdown_pct": round(float(pd.to_numeric(frame["drawdown"], errors="coerce").min()) * 100, 4), "trade_days": int(frame["action"].astype(str).ne("hold").sum()), "total_transaction_cost": _sum(frame, "transaction_cost"), "total_turnover": _sum(frame, "turnover")}


def _benchmark_return(series: pd.Series, start: str, end: str, multiplier: float = 1.0) -> float | None:
    start_price = _price_on_or_before(series, start)
    end_price = _price_on_or_before(series, end)
    if start_price is None or end_price is None or start_price == 0:
        return None
    return ((end_price / start_price) - 1) * multiplier


def _forward_return(series: pd.Series | None, date: str, horizon: int) -> float | None:
    if series is None:
        return None
    start_price = _price_on_or_before(series, date)
    future = series[series.index > pd.Timestamp(date)]
    if start_price is None or len(future) < horizon:
        return None
    return float(future.iloc[horizon - 1] / start_price - 1)


def _exposure_row(variant: str, label: str, subset: pd.DataFrame) -> dict[str, Any]:
    active = subset[subset["position_ticker"].astype(str).ne("cash")]
    exposure = subset[subset["position_ticker"].eq("00631L.TW")]
    returns = subset["equity"].pct_change().fillna(0)
    exposure_returns = returns.loc[exposure.index] if not exposure.empty else pd.Series(dtype=float)
    return {"candidate": variant, "period_label": label, "row_count": len(subset), "active_days": len(active), "00631L_active_days": len(exposure), "00631L_position_day_share": round(len(exposure) / len(active), 6) if len(active) else 0.0, "00631L_return_contribution_proxy": round(float(exposure_returns.sum()), 8) if len(exposure_returns) else 0.0, "00631L_min_drawdown_proxy": round(float(pd.to_numeric(exposure["drawdown"], errors="coerce").min()), 8) if not exposure.empty else ""}


def _top_contribution_period(frame: pd.DataFrame, column: str) -> str:
    temp = frame.copy()
    temp["daily_return"] = temp["equity"].pct_change().fillna(0)
    grouped = temp.groupby(column)["daily_return"].sum().sort_values(ascending=False)
    return "" if grouped.empty else str(grouped.index[0])


def _rapid_flip_rate(group: pd.DataFrame) -> float:
    values = group["position_ticker"].astype(str).tolist()
    flags = []
    for index, value in enumerate(values):
        future = values[index + 1 : index + 4]
        flags.append(bool(value and value in future and any(item and item != value for item in future)))
    return round(sum(flags) / len(flags), 6) if flags else 0.0


def _changed_rate(group: pd.DataFrame, window: int) -> float:
    values = group["position_ticker"].astype(str).tolist()
    flags = []
    for index, value in enumerate(values):
        future = values[index + 1 : index + window + 1]
        flags.append(bool(value and any(item and item != value for item in future)))
    return round(sum(flags) / len(flags), 6) if flags else 0.0


def _same_date_range(daily: pd.DataFrame) -> bool:
    ranges = daily.groupby("variant")["date"].agg(["min", "max", "count"])
    return bool(ranges["min"].nunique() == 1 and ranges["max"].nunique() == 1 and ranges["count"].nunique() == 1)


def _sum(frame: pd.DataFrame, column: str) -> float:
    return round(float(pd.to_numeric(frame.get(column, pd.Series(dtype=float)), errors="coerce").fillna(0).sum()), 2)


def _round(value: float | None) -> Any:
    return "" if value is None else round(float(value), 8)


def _round_pct(value: float | None) -> Any:
    return "" if value is None else round(float(value) * 100, 4)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Pool1+Pool2 final challenger robustness panels.")
    parser.add_argument("--source-dir", default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--price-cache-dir", default=DEFAULT_PRICE_CACHE_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    output = run_pool1_pool2_final_challenger_robustness(source_dir=args.source_dir, price_cache_dir=args.price_cache_dir, output_dir=args.output_dir)
    print(f"OUTPUT_DIR={output.resolve()}")


if __name__ == "__main__":
    main()
