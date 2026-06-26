from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from backtest_lab.three_pool_vs_pool1_comparison_panels import _load_prices, _price_on_or_before, _text


DEFAULT_CHALLENGER_DIR = "outputs/pool1_primary_risk_overlay_challenger_panels_20260626"
DEFAULT_PRICE_CACHE_DIR = "backtest_cache/stock_pool_triad_v1_corrected"
DEFAULT_OUTPUT_DIR = "outputs/pool1_pool2_veto_robustness_panels_20260626"
MAIN_VARIANT = "pool1_primary_pool2_risk_veto"
BASELINE_VARIANT = "current_formal_three_pool_baseline"
POOL1_VARIANT = "pool1_primary_no_overlay"
HORIZONS = (20, 60, 120)
BENCHMARKS = {"0050": "0050.TW", "00631L": "00631L.TW"}


def run_pool1_pool2_veto_robustness(
    *,
    challenger_dir: str | Path = DEFAULT_CHALLENGER_DIR,
    price_cache_dir: str | Path = DEFAULT_PRICE_CACHE_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    run_log: list[dict[str, str]] = []

    def log(step: str, status: str, detail: str = "") -> None:
        run_log.append(
            {
                "timestamp": pd.Timestamp.now(tz="Asia/Taipei").strftime("%Y-%m-%d %H:%M:%S%z"),
                "step": step,
                "status": status,
                "detail": detail,
            }
        )
        pd.DataFrame(run_log).to_csv(output / "run_log.csv", index=False, encoding="utf-8-sig")
        (output / "current_step.txt").write_text(step, encoding="utf-8")

    try:
        root = Path(challenger_dir)
        log("load_inputs", "started", str(root))
        daily = pd.read_csv(root / "daily_equity_by_variant.csv").fillna("")
        targets = pd.read_csv(root / "daily_target_by_variant.csv").fillna("")
        veto = pd.read_csv(root / "veto_event_panel.csv").fillna("")
        trades = pd.read_csv(root / "trade_ledger_by_variant.csv").fillna("")
        _validate_inputs(daily, targets, veto)
        prices = _load_prices(_needed_tickers(targets, veto), Path(price_cache_dir))

        log("build_reports", "started", "")
        oos = _oos_walk_forward(daily)
        leave_one = _leave_one_period(daily)
        by_ticker = _contribution_by_ticker(daily)
        by_month_quarter = _contribution_by_month_quarter(daily)
        exclusion = _contribution_exclusion_tests(daily)
        exposure = _00631l_exposure(daily, trades)
        cap_sensitivity = _00631l_cap_sensitivity(daily)
        veto_forward = _vetoed_event_forward_outcome(veto, daily, prices)
        veto_summary = _veto_reason_outcome_summary(veto_forward)
        execution = _execution_ledger_comparison(daily, trades)
        entry = _entry_without_exit_outcome(targets)
        rapid = _rapid_flip_execution_diagnostics(targets)

        log("write_outputs", "started", "")
        oos.to_csv(output / "oos_walk_forward_performance.csv", index=False, encoding="utf-8-sig")
        leave_one.to_csv(output / "leave_one_period_robustness.csv", index=False, encoding="utf-8-sig")
        by_ticker.to_csv(output / "contribution_by_ticker.csv", index=False, encoding="utf-8-sig")
        by_month_quarter.to_csv(output / "contribution_by_month_quarter.csv", index=False, encoding="utf-8-sig")
        exclusion.to_csv(output / "contribution_exclusion_tests.csv", index=False, encoding="utf-8-sig")
        exposure.to_csv(output / "00631L_exposure_breakdown.csv", index=False, encoding="utf-8-sig")
        cap_sensitivity.to_csv(output / "00631L_cap_sensitivity.csv", index=False, encoding="utf-8-sig")
        veto_forward.to_csv(output / "vetoed_event_forward_outcome.csv", index=False, encoding="utf-8-sig")
        veto_summary.to_csv(output / "veto_reason_outcome_summary.csv", index=False, encoding="utf-8-sig")
        execution.to_csv(output / "execution_ledger_comparison.csv", index=False, encoding="utf-8-sig")
        entry.to_csv(output / "entry_without_exit_outcome.csv", index=False, encoding="utf-8-sig")
        rapid.to_csv(output / "rapid_flip_execution_diagnostics.csv", index=False, encoding="utf-8-sig")
        (output / "pool1_pool2_veto_robustness_summary_zh.md").write_text(_summary_markdown(oos, exposure), encoding="utf-8")
        manifest = {
            "schema_version": 1,
            "task_id": "TASK-BACKTEST-CORE-POOL1-POOL2-VETO-ROBUSTNESS-PANELS-001",
            "model": "pool1_pool2_veto_robustness",
            "status": "completed",
            "challenger_dir": str(root),
            "price_cache_dir": str(price_cache_dir),
            "main_variant": MAIN_VARIANT,
            "formal_model_changed": False,
            "trade_decision_changed": False,
            "formal_absorption_ready": False,
            "pool3_shadow_used_as_formal": False,
            "report_only_labels_used_in_performance": False,
            "rr_partial_switch_used_in_performance": False,
            "uses_forward_return_as_rule": False,
            "valuation_used": False,
            "h3_used": False,
            "outputs": {
                "oos_walk_forward_performance": "oos_walk_forward_performance.csv",
                "leave_one_period_robustness": "leave_one_period_robustness.csv",
                "contribution_by_ticker": "contribution_by_ticker.csv",
                "contribution_by_month_quarter": "contribution_by_month_quarter.csv",
                "contribution_exclusion_tests": "contribution_exclusion_tests.csv",
                "00631L_exposure_breakdown": "00631L_exposure_breakdown.csv",
                "00631L_cap_sensitivity": "00631L_cap_sensitivity.csv",
                "vetoed_event_forward_outcome": "vetoed_event_forward_outcome.csv",
                "veto_reason_outcome_summary": "veto_reason_outcome_summary.csv",
                "execution_ledger_comparison": "execution_ledger_comparison.csv",
                "entry_without_exit_outcome": "entry_without_exit_outcome.csv",
                "rapid_flip_execution_diagnostics": "rapid_flip_execution_diagnostics.csv",
                "summary": "pool1_pool2_veto_robustness_summary_zh.md",
            },
        }
        (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        pd.DataFrame([{"status": "completed", "output_dir": str(output.resolve())}]).to_csv(output / "completed.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(columns=["step", "error"]).to_csv(output / "failed.csv", index=False, encoding="utf-8-sig")
        log("completed", "completed", str(output.resolve()))
        (output / "current_step.txt").write_text("completed\n", encoding="utf-8")
        return output
    except Exception as exc:
        pd.DataFrame([{"step": "run_pool1_pool2_veto_robustness", "error": str(exc)}]).to_csv(output / "failed.csv", index=False, encoding="utf-8-sig")
        log("failed", "failed", str(exc))
        raise


def _validate_inputs(daily: pd.DataFrame, targets: pd.DataFrame, veto: pd.DataFrame) -> None:
    for name, frame, required in (
        ("daily", daily, {"variant", "date", "equity", "position_ticker", "action", "transaction_cost", "turnover"}),
        ("targets", targets, {"variant", "date", "formal_target", "entry_signal_without_exit_confirmation"}),
        ("veto", veto, {"date", "vetoed_target", "risk_veto_reason"}),
    ):
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{name} missing columns: {missing}")


def _needed_tickers(targets: pd.DataFrame, veto: pd.DataFrame) -> list[str]:
    tickers = set()
    for column in ("formal_target", "position_ticker", "pool1_vote", "pool2_vote", "pool3_vote"):
        if column in targets.columns:
            tickers.update(_text(value) for value in targets[column].tolist() if _text(value))
    tickers.update(_text(value) for value in veto["vetoed_target"].tolist() if _text(value))
    tickers.update(BENCHMARKS.values())
    return sorted(tickers)


def _oos_walk_forward(daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    tests = {
        "train_2022_2024": ("2022-01-01", "2024-12-31"),
        "test_2025_2026": ("2025-01-01", None),
        "train_2022_2023": ("2022-01-01", "2023-12-31"),
        "test_2024_now": ("2024-01-01", None),
    }
    for variant in (MAIN_VARIANT, POOL1_VARIANT, BASELINE_VARIANT):
        frame = daily[daily["variant"].eq(variant)].copy()
        frame["date_ts"] = pd.to_datetime(frame["date"])
        for label, (start, end) in tests.items():
            subset = frame[frame["date_ts"] >= pd.Timestamp(start)]
            if end:
                subset = subset[subset["date_ts"] <= pd.Timestamp(end)]
            rows.append(_perf_row(variant, label, subset))
    return pd.DataFrame(rows)


def _leave_one_period(daily: pd.DataFrame) -> pd.DataFrame:
    frame = daily[daily["variant"].eq(MAIN_VARIANT)].copy()
    frame["date_ts"] = pd.to_datetime(frame["date"])
    frame["year"] = frame["date_ts"].dt.year.astype(str)
    frame["quarter"] = frame["date_ts"].dt.to_period("Q").astype(str)
    frame["month"] = frame["date_ts"].dt.to_period("M").astype(str)
    rows = []
    for level in ("year", "quarter", "month"):
        values = frame[level].value_counts().head(12).index.tolist() if level != "year" else sorted(frame[level].unique())
        for value in values:
            subset = frame[~frame[level].eq(value)]
            row = _perf_row(MAIN_VARIANT, f"leave_one_{level}_{value}", subset)
            row["excluded_level"] = level
            row["excluded_value"] = value
            rows.append(row)
    return pd.DataFrame(rows)


def _contribution_by_ticker(daily: pd.DataFrame) -> pd.DataFrame:
    frame = _main(daily)
    frame["daily_return"] = pd.to_numeric(frame["equity"], errors="coerce").pct_change().fillna(0)
    rows = []
    for ticker, group in frame.groupby("position_ticker", dropna=False):
        rows.append({"ticker": ticker, "position_days": len(group), "return_contribution_sum": round(float(group["daily_return"].sum()), 8), "avg_daily_return": round(float(group["daily_return"].mean()), 8)})
    return pd.DataFrame(rows).sort_values("return_contribution_sum", ascending=False)


def _contribution_by_month_quarter(daily: pd.DataFrame) -> pd.DataFrame:
    frame = _main(daily)
    frame["date_ts"] = pd.to_datetime(frame["date"])
    frame["daily_return"] = pd.to_numeric(frame["equity"], errors="coerce").pct_change().fillna(0)
    rows = []
    for level, series in {"month": frame["date_ts"].dt.to_period("M").astype(str), "quarter": frame["date_ts"].dt.to_period("Q").astype(str)}.items():
        for value, group in frame.groupby(series, dropna=False):
            rows.append({"period_level": level, "period_value": value, "row_count": len(group), "return_contribution_sum": round(float(group["daily_return"].sum()), 8)})
    return pd.DataFrame(rows).sort_values(["period_level", "return_contribution_sum"], ascending=[True, False])


def _contribution_exclusion_tests(daily: pd.DataFrame) -> pd.DataFrame:
    frame = _main(daily)
    frame["date_ts"] = pd.to_datetime(frame["date"])
    frame["month"] = frame["date_ts"].dt.to_period("M").astype(str)
    frame["quarter"] = frame["date_ts"].dt.to_period("Q").astype(str)
    top_ticker = _top_value(frame, "position_ticker")
    top_month = _top_contribution_period(frame, "month")
    top_quarter = _top_contribution_period(frame, "quarter")
    tests = [
        ("exclude_00631L", frame[~frame["position_ticker"].eq("00631L.TW")]),
        (f"exclude_top_ticker_{top_ticker}", frame[~frame["position_ticker"].eq(top_ticker)]),
        (f"exclude_top_month_{top_month}", frame[~frame["month"].eq(top_month)]),
        (f"exclude_top_quarter_{top_quarter}", frame[~frame["quarter"].eq(top_quarter)]),
    ]
    return pd.DataFrame([_perf_row(MAIN_VARIANT, label, subset) for label, subset in tests])


def _00631l_exposure(daily: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant in (MAIN_VARIANT, POOL1_VARIANT, BASELINE_VARIANT):
        frame = daily[daily["variant"].eq(variant)].copy()
        active = frame[frame["position_ticker"].astype(str).ne("cash")]
        exposure = frame[frame["position_ticker"].eq("00631L.TW")]
        variant_trades = trades[trades["variant"].eq(variant)].copy() if not trades.empty else pd.DataFrame()
        rows.append({
            "variant": variant,
            "active_days": int(len(active)),
            "00631L_active_days": int(len(exposure)),
            "00631L_position_day_share": round(len(exposure) / len(active), 6) if len(active) else 0,
            "00631L_trade_count": int(variant_trades["ticker"].astype(str).eq("00631L.TW").sum()) if not variant_trades.empty and "ticker" in variant_trades.columns else 0,
            "00631L_trade_cost": _sum_numeric(variant_trades[variant_trades.get("ticker", pd.Series(dtype=str)).astype(str).eq("00631L.TW")], "costs"),
            "drawdown_contribution_proxy": round(float(pd.to_numeric(exposure["drawdown"], errors="coerce").min()), 8) if not exposure.empty else "",
            "return_contribution_proxy": _period_return(exposure),
        })
    return pd.DataFrame(rows)


def _00631l_cap_sensitivity(daily: pd.DataFrame) -> pd.DataFrame:
    frame = _main(daily)
    share = float(frame["position_ticker"].eq("00631L.TW").mean()) if len(frame) else 0
    return pd.DataFrame([
        {"cap_rule": "cap_30_percent_diagnostic", "observed_00631L_share": round(share, 6), "cap_breached": share > 0.30, "performance_recomputed": False},
        {"cap_rule": "cap_40_percent_diagnostic", "observed_00631L_share": round(share, 6), "cap_breached": share > 0.40, "performance_recomputed": False},
        {"cap_rule": "no_cap_observed", "observed_00631L_share": round(share, 6), "cap_breached": False, "performance_recomputed": False},
    ])


def _vetoed_event_forward_outcome(veto: pd.DataFrame, daily: pd.DataFrame, prices: dict[str, pd.Series]) -> pd.DataFrame:
    held = daily[daily["variant"].eq(MAIN_VARIANT)][["date", "position_ticker"]].copy()
    rows = []
    for item in veto[veto["variant"].eq(MAIN_VARIANT)].to_dict(orient="records"):
        date = _text(item.get("date"))
        vetoed = _text(item.get("vetoed_target"))
        actual = _text(held[held["date"].astype(str).eq(date)]["position_ticker"].iloc[0]) if not held[held["date"].astype(str).eq(date)].empty else ""
        row: dict[str, Any] = {"date": date, "period": item.get("period", ""), "veto_reason": item.get("risk_veto_reason", ""), "vetoed_target": vetoed, "actual_target_or_holding": actual}
        for horizon in HORIZONS:
            vetoed_ret = _forward_return(prices.get(vetoed), date, horizon)
            actual_ret = _forward_return(prices.get(actual), date, horizon) if actual and actual != "cash" else None
            row[f"vetoed_forward_{horizon}d_return"] = _round(vetoed_ret)
            row[f"actual_holding_forward_{horizon}d_return"] = _round(actual_ret)
            row[f"vetoed_excess_vs_actual_{horizon}d"] = _round(_diff(vetoed_ret, actual_ret))
            for label, ticker in BENCHMARKS.items():
                bench = _forward_return(prices.get(ticker), date, horizon)
                row[f"vetoed_excess_vs_{label}_{horizon}d"] = _round(_diff(vetoed_ret, bench))
            bench0050 = _forward_return(prices.get("0050.TW"), date, horizon)
            row[f"vetoed_excess_vs_0050x2_{horizon}d"] = _round(_diff(vetoed_ret, None if bench0050 is None else bench0050 * 2))
        row["uses_forward_return_as_rule"] = False
        rows.append(row)
    return pd.DataFrame(rows)


def _veto_reason_outcome_summary(panel: pd.DataFrame) -> pd.DataFrame:
    if panel.empty:
        return pd.DataFrame()
    rows = []
    for reason, group in panel.groupby("veto_reason", dropna=False):
        row = {"veto_reason": reason, "event_count": len(group)}
        for horizon in HORIZONS:
            row[f"vetoed_forward_{horizon}d_mean"] = _mean(group[f"vetoed_forward_{horizon}d_return"])
            row[f"vetoed_excess_vs_actual_{horizon}d_mean"] = _mean(group[f"vetoed_excess_vs_actual_{horizon}d"])
        rows.append(row)
    return pd.DataFrame(rows)


def _execution_ledger_comparison(daily: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant, frame in daily.groupby("variant", dropna=False):
        trade_subset = trades[trades["variant"].eq(variant)] if not trades.empty and "variant" in trades.columns else pd.DataFrame()
        rows.append({
            "variant": variant,
            "same_day_full_switch": True,
            "next_day_approximation_available": False,
            "trade_days": int(frame["action"].astype(str).ne("hold").sum()),
            "rapid_flip_days": int(_rapid_flip_flags(frame["winner_ticker"].astype(str).tolist()).sum()),
            "target_changed_within_1d": int(_changed_flags(frame["winner_ticker"].astype(str).tolist(), 1).sum()),
            "target_changed_within_3d": int(_changed_flags(frame["winner_ticker"].astype(str).tolist(), 3).sum()),
            "total_transaction_cost": _sum_numeric(frame, "transaction_cost"),
            "trade_ledger_rows": int(len(trade_subset)),
            "diagnostic_only": True,
        })
    return pd.DataFrame(rows)


def _entry_without_exit_outcome(targets: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant, group in targets.groupby("variant", dropna=False):
        flags = group["entry_signal_without_exit_confirmation"].map(lambda value: str(value).lower() == "true")
        rows.append({"variant": variant, "event_count": len(group), "entry_without_exit_count": int(flags.sum()), "entry_without_exit_rate": round(float(flags.mean()), 6) if len(flags) else 0, "outcome_status": "needs_experiments_forward_validation"})
    return pd.DataFrame(rows)


def _rapid_flip_execution_diagnostics(targets: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant, group in targets.groupby("variant", dropna=False):
        values = group.sort_values("date")["formal_target"].astype(str).tolist()
        rows.append({"variant": variant, "rapid_flip_count": int(_rapid_flip_flags(values).sum()), "target_changed_within_1d_count": int(_changed_flags(values, 1).sum()), "target_changed_within_3d_count": int(_changed_flags(values, 3).sum())})
    return pd.DataFrame(rows)


def _perf_row(variant: str, label: str, frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {"variant": variant, "period_label": label, "status": "empty"}
    equity = pd.to_numeric(frame["equity"], errors="coerce")
    start = float(equity.iloc[0])
    end = float(equity.iloc[-1])
    return {"variant": variant, "period_label": label, "status": "completed", "start_date": frame["date"].iloc[0], "end_date": frame["date"].iloc[-1], "start_equity": round(start, 2), "final_equity": round(end, 2), "return_pct": round((end / start - 1) * 100, 4) if start else 0.0, "max_drawdown_pct": round(float(pd.to_numeric(frame["drawdown"], errors="coerce").min()) * 100, 4), "trade_days": int(frame["action"].astype(str).ne("hold").sum()), "total_transaction_cost": _sum_numeric(frame, "transaction_cost")}


def _main(daily: pd.DataFrame) -> pd.DataFrame:
    return daily[daily["variant"].eq(MAIN_VARIANT)].copy()


def _top_value(frame: pd.DataFrame, column: str) -> str:
    values = frame[column].astype(str)
    values = values[values.ne("") & values.ne("cash")]
    return values.value_counts().index[0] if not values.empty else ""


def _top_contribution_period(frame: pd.DataFrame, column: str) -> str:
    work = frame.copy()
    work["daily_return"] = pd.to_numeric(work["equity"], errors="coerce").pct_change().fillna(0)
    grouped = work.groupby(column)["daily_return"].sum()
    return str(grouped.sort_values(ascending=False).index[0]) if not grouped.empty else ""


def _period_return(frame: pd.DataFrame) -> Any:
    if frame.empty:
        return ""
    equity = pd.to_numeric(frame["equity"], errors="coerce")
    if len(equity) < 2 or float(equity.iloc[0]) == 0:
        return ""
    return round((float(equity.iloc[-1]) / float(equity.iloc[0]) - 1) * 100, 4)


def _forward_return(series: pd.Series | None, date: str, horizon: int) -> float | None:
    if series is None or series.empty:
        return None
    future = series.loc[series.index >= pd.Timestamp(date)]
    if len(future) <= horizon:
        return None
    start = float(future.iloc[0])
    end = float(future.iloc[horizon])
    return end / start - 1 if start else None


def _changed_flags(values: list[str], window: int) -> pd.Series:
    flags = []
    for index, value in enumerate(values):
        future = values[index + 1 : index + window + 1]
        flags.append(bool(value and any(item and item != value for item in future)))
    return pd.Series(flags)


def _rapid_flip_flags(values: list[str]) -> pd.Series:
    flags = []
    for index, value in enumerate(values):
        future = values[index + 1 : index + 4]
        flags.append(bool(value and value in future and any(item and item != value for item in future)))
    return pd.Series(flags)


def _sum_numeric(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame.columns:
        return 0.0
    return round(float(pd.to_numeric(frame[column], errors="coerce").fillna(0).sum()), 2)


def _mean(series: pd.Series) -> Any:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return round(float(values.mean()), 8) if len(values) else ""


def _diff(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return left - right


def _round(value: float | None) -> Any:
    return "" if value is None else round(float(value), 8)


def _summary_markdown(oos: pd.DataFrame, exposure: pd.DataFrame) -> str:
    lines = [
        "# Pool1 + Pool2 Veto Robustness Panels",
        "",
        "本輸出只提供主候選 robustness evidence，不改正式模型，也不把 forward return 作為規則。",
        "",
        "## OOS / Walk-forward",
        "",
    ]
    for row in oos.to_dict(orient="records"):
        lines.append(f"- {row.get('variant')} {row.get('period_label')}: return {row.get('return_pct')}%, MDD {row.get('max_drawdown_pct')}%")
    lines.extend(["", "## 00631L Exposure", ""])
    for row in exposure.to_dict(orient="records"):
        lines.append(f"- {row.get('variant')}: 00631L share {row.get('00631L_position_day_share')}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build robustness panels for Pool1 primary + Pool2 veto challenger.")
    parser.add_argument("--challenger-dir", default=DEFAULT_CHALLENGER_DIR)
    parser.add_argument("--price-cache-dir", default=DEFAULT_PRICE_CACHE_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    output = run_pool1_pool2_veto_robustness(challenger_dir=args.challenger_dir, price_cache_dir=args.price_cache_dir, output_dir=args.output_dir)
    print(f"OUTPUT_DIR={output.resolve()}")


if __name__ == "__main__":
    main()
