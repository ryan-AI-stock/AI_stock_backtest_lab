from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_STATE_PANEL = "outputs/final_decision_layer_spec_diagnostic_20260625/final_decision_state_panel.csv"
DEFAULT_PRICE_CACHE_DIR = "backtest_cache/stock_pool_triad_v1_corrected"
DEFAULT_OUTPUT_DIR = "outputs/final_decision_layer_forward_outcome_adapter_20260625"
REPORT_ONLY_BOUNDARY = "report_only_diagnostic"
HORIZONS = (20, 60, 120)
BENCHMARKS = {
    "0050": "0050.TW",
    "00631L": "00631L.TW",
}


def run_final_decision_layer_forward_outcome_adapter(
    *,
    state_panel_path: str | Path = DEFAULT_STATE_PANEL,
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
        log("load_inputs", "started", str(state_panel_path))
        state_panel = pd.read_csv(state_panel_path).fillna("")
        _validate_state_panel(state_panel)
        tickers = _needed_tickers(state_panel)
        prices = _load_prices(tickers, Path(price_cache_dir))

        log("build_forward_outcomes", "started", "")
        outcome = build_forward_outcome_panel(state_panel, prices)
        by_state = _summary_by(outcome, ["final_decision_state"])
        by_period_state = _summary_by(outcome, ["period_label", "final_decision_state"])
        by_source = _summary_by(outcome, ["decision_source"])
        by_target_type = _summary_by(outcome, ["final_target_type"])
        concentration = _concentration(outcome)
        coverage = _coverage(outcome)

        log("write_outputs", "started", "")
        outcome.to_csv(output / "final_decision_forward_outcome_panel.csv", index=False, encoding="utf-8-sig")
        by_state.to_csv(output / "forward_outcome_by_state.csv", index=False, encoding="utf-8-sig")
        by_period_state.to_csv(output / "forward_outcome_by_period_state.csv", index=False, encoding="utf-8-sig")
        by_source.to_csv(output / "forward_outcome_by_decision_source.csv", index=False, encoding="utf-8-sig")
        by_target_type.to_csv(output / "forward_outcome_by_target_type.csv", index=False, encoding="utf-8-sig")
        concentration.to_csv(output / "forward_outcome_concentration.csv", index=False, encoding="utf-8-sig")
        coverage.to_csv(output / "forward_outcome_data_coverage.csv", index=False, encoding="utf-8-sig")
        (output / "final_decision_forward_outcome_summary_zh.md").write_text(
            _summary_markdown(coverage, by_state),
            encoding="utf-8",
        )

        manifest = {
            "schema_version": 1,
            "task_id": "TASK-BACKTEST-CORE-FINAL-DECISION-LAYER-FORWARD-OUTCOME-ADAPTER-001",
            "model": "final_decision_layer_forward_outcome_adapter",
            "status": "completed",
            "state_panel_path": str(state_panel_path),
            "price_cache_dir": str(price_cache_dir),
            "formal_model_changed": False,
            "trade_decision_changed": False,
            "active_in_trade_decision": False,
            "final_decision_layer_boundary": REPORT_ONLY_BOUNDARY,
            "uses_forward_return_as_rule": False,
            "pool3_shadow_used_as_formal": False,
            "etf_counted_as_stock_vote": False,
            "rr_partial_switch_used": False,
            "valuation_used": False,
            "h3_used": False,
            "outputs": {
                "forward_outcome_panel": "final_decision_forward_outcome_panel.csv",
                "by_state": "forward_outcome_by_state.csv",
                "by_period_state": "forward_outcome_by_period_state.csv",
                "by_decision_source": "forward_outcome_by_decision_source.csv",
                "by_target_type": "forward_outcome_by_target_type.csv",
                "concentration": "forward_outcome_concentration.csv",
                "coverage": "forward_outcome_data_coverage.csv",
                "summary": "final_decision_forward_outcome_summary_zh.md",
            },
        }
        (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        pd.DataFrame([{"status": "completed", "output_dir": str(output.resolve())}]).to_csv(
            output / "completed.csv",
            index=False,
            encoding="utf-8-sig",
        )
        pd.DataFrame(columns=["step", "error"]).to_csv(output / "failed.csv", index=False, encoding="utf-8-sig")
        log("completed", "completed", str(output.resolve()))
        (output / "current_step.txt").write_text("completed\n", encoding="utf-8")
        return output
    except Exception as exc:
        pd.DataFrame([{"step": "run_final_decision_layer_forward_outcome_adapter", "error": str(exc)}]).to_csv(
            output / "failed.csv",
            index=False,
            encoding="utf-8-sig",
        )
        log("failed", "failed", str(exc))
        raise


def build_forward_outcome_panel(state_panel: pd.DataFrame, prices: dict[str, pd.Series]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for item in state_panel.to_dict(orient="records"):
        target = _text(item.get("final_target_ticker"))
        date = _text(item.get("signal_date"))
        decision_source = _decision_source(item)
        row: dict[str, Any] = {
            "decision_date": date,
            "period_label": item.get("period", ""),
            "final_decision_state": item.get("final_decision_state", ""),
            "final_target_type": item.get("final_target_type", ""),
            "final_target": target,
            "decision_source": decision_source,
            "target_priority_rank": item.get("target_priority_rank", ""),
            "market_exposure_tool": item.get("exposure_target", ""),
            "not_eligible_for_formal_selector": _truthy(item.get("not_eligible_for_formal_selector", False)),
            "eligible_for_forward_study": bool(target),
            "not_eligible_reason": "" if target else "no_final_target",
            "outcome_data_complete": True,
            "outcome_blocked_reason": "",
            "active_in_trade_decision": False,
            "uses_forward_return_as_rule": False,
        }
        target_metrics = _forward_metrics(prices.get(target), date) if target else _blank_metrics("no_final_target")
        benchmark_metrics = {label: _forward_metrics(prices.get(ticker), date) for label, ticker in BENCHMARKS.items()}
        if not target:
            row["outcome_data_complete"] = False
            row["outcome_blocked_reason"] = "no_final_target"
        for horizon in HORIZONS:
            target_return = target_metrics[horizon]["return"]
            row[f"forward_{horizon}d_return"] = _round_or_blank(target_return)
            for bench_label in ("0050", "00631L"):
                bench_return = benchmark_metrics[bench_label][horizon]["return"]
                row[f"forward_{horizon}d_excess_vs_{bench_label}"] = _round_or_blank(_subtract(target_return, bench_return))
            synthetic_0050x2 = _multiply(benchmark_metrics["0050"][horizon]["return"], 2.0)
            row[f"forward_{horizon}d_excess_vs_0050x2"] = _round_or_blank(_subtract(target_return, synthetic_0050x2))
            row[f"max_drawdown_{horizon}d"] = _round_or_blank(target_metrics[horizon]["max_drawdown"])
            row[f"max_runup_{horizon}d"] = _round_or_blank(target_metrics[horizon]["max_runup"])
            if target and target_metrics[horizon]["return"] is None:
                row["outcome_data_complete"] = False
                reason = target_metrics[horizon]["blocked_reason"] or f"missing_target_forward_{horizon}d"
                row["outcome_blocked_reason"] = _append_reason(row["outcome_blocked_reason"], reason)
        rows.append(row)
    return pd.DataFrame(rows)


def _forward_metrics(series: pd.Series | None, date: str) -> dict[int, dict[str, Any]]:
    if series is None or series.empty:
        return _blank_metrics("missing_price_series")
    date_ts = pd.Timestamp(date)
    future = series.loc[series.index >= date_ts]
    if future.empty:
        return _blank_metrics("no_price_on_or_after_decision_date")
    start = float(future.iloc[0])
    result: dict[int, dict[str, Any]] = {}
    for horizon in HORIZONS:
        if len(future) <= horizon:
            result[horizon] = {"return": None, "max_drawdown": None, "max_runup": None, "blocked_reason": f"insufficient_{horizon}d_forward_window"}
            continue
        window = future.iloc[: horizon + 1].astype(float)
        end = float(window.iloc[-1])
        if not start:
            result[horizon] = {"return": None, "max_drawdown": None, "max_runup": None, "blocked_reason": "zero_start_price"}
            continue
        relative = window / start
        result[horizon] = {
            "return": end / start - 1,
            "max_drawdown": float((relative / relative.cummax() - 1).min()),
            "max_runup": float(relative.max() - 1),
            "blocked_reason": "",
        }
    return result


def _blank_metrics(reason: str) -> dict[int, dict[str, Any]]:
    return {horizon: {"return": None, "max_drawdown": None, "max_runup": None, "blocked_reason": reason} for horizon in HORIZONS}


def _summary_by(outcome: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if outcome.empty:
        return pd.DataFrame()
    for group_key, group in outcome.groupby(keys, dropna=False):
        row: dict[str, Any] = {}
        if len(keys) == 1:
            row[keys[0]] = group_key[0] if isinstance(group_key, tuple) else group_key
        else:
            for key, value in zip(keys, group_key, strict=False):
                row[key] = value
        complete = group[group["outcome_data_complete"].astype(bool)].copy()
        row["event_count"] = int(len(group))
        row["complete_event_count"] = int(len(complete))
        row["complete_coverage_rate"] = _rate(len(complete), len(group))
        for horizon in HORIZONS:
            col = f"forward_{horizon}d_return"
            row[f"forward_{horizon}d_mean"] = _mean(complete[col])
            row[f"forward_{horizon}d_median"] = _median(complete[col])
            row[f"forward_{horizon}d_win_rate"] = _win_rate(complete[col])
            row[f"forward_{horizon}d_excess_vs_0050_mean"] = _mean(complete[f"forward_{horizon}d_excess_vs_0050"])
            row[f"forward_{horizon}d_excess_vs_00631L_mean"] = _mean(complete[f"forward_{horizon}d_excess_vs_00631L"])
            row[f"forward_{horizon}d_excess_vs_0050x2_mean"] = _mean(complete[f"forward_{horizon}d_excess_vs_0050x2"])
            row[f"max_drawdown_{horizon}d_mean"] = _mean(complete[f"max_drawdown_{horizon}d"])
            row[f"max_runup_{horizon}d_mean"] = _mean(complete[f"max_runup_{horizon}d"])
        row["active_in_trade_decision"] = False
        rows.append(row)
    return pd.DataFrame(rows)


def _concentration(outcome: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    frame = outcome.copy()
    frame["month"] = pd.to_datetime(frame["decision_date"], errors="coerce").dt.to_period("M").astype(str)
    frame["quarter"] = pd.to_datetime(frame["decision_date"], errors="coerce").dt.to_period("Q").astype(str)
    for state, group in frame.groupby("final_decision_state", dropna=False):
        total = len(group)
        for field in ("final_target", "month", "quarter"):
            values = group[field].astype(str)
            values = values[values.str.strip().ne("")]
            if values.empty:
                top_value = ""
                top_count = 0
            else:
                counts = values.value_counts()
                top_value = str(counts.index[0])
                top_count = int(counts.iloc[0])
            rows.append(
                {
                    "final_decision_state": state,
                    "concentration_field": field,
                    "top_value": top_value,
                    "top_count": top_count,
                    "event_count": total,
                    "top_share": _rate(top_count, total),
                    "active_in_trade_decision": False,
                }
            )
    return pd.DataFrame(rows)


def _coverage(outcome: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for period, group in outcome.groupby("period_label", dropna=False):
        rows.append(
            {
                "period_label": period,
                "event_count": int(len(group)),
                "eligible_for_forward_study_count": int(group["eligible_for_forward_study"].astype(bool).sum()),
                "outcome_data_complete_count": int(group["outcome_data_complete"].astype(bool).sum()),
                "outcome_data_complete_rate": _rate(group["outcome_data_complete"].astype(bool).sum(), len(group)),
                "blocked_count": int((~group["outcome_data_complete"].astype(bool)).sum()),
                "top_blocked_reason": _top_value(group.loc[~group["outcome_data_complete"].astype(bool), "outcome_blocked_reason"]),
                "active_in_trade_decision": False,
            }
        )
    return pd.DataFrame(rows)


def _validate_state_panel(panel: pd.DataFrame) -> None:
    required = {"period", "signal_date", "final_decision_state", "final_target_type", "final_target_ticker"}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError("missing final decision state panel columns: " + ",".join(sorted(missing)))


def _needed_tickers(panel: pd.DataFrame) -> list[str]:
    tickers = {"0050.TW", "00631L.TW"}
    for column in ("final_target_ticker", "exposure_target"):
        if column in panel.columns:
            tickers.update(_text(value) for value in panel[column].tolist() if _text(value))
    return sorted(tickers)


def _load_prices(tickers: list[str], cache_dir: Path) -> dict[str, pd.Series]:
    prices: dict[str, pd.Series] = {}
    for ticker in tickers:
        path = cache_dir / f"{ticker.replace('.', '_')}.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        price_col = "adj_close" if "adj_close" in frame.columns else "close" if "close" in frame.columns else ""
        if "date" not in frame.columns or not price_col:
            continue
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        series = pd.Series(pd.to_numeric(frame[price_col], errors="coerce").values, index=frame["date"])
        series = series.dropna().sort_index()
        if not series.empty:
            prices[ticker] = series
    return prices


def _decision_source(row: dict[str, Any]) -> str:
    if _text(row.get("final_decision_state")) == "strong_consensus":
        return "exact_consensus"
    if _text(row.get("final_decision_state")) == "weak_consensus":
        return "direction_consensus"
    if _truthy(row.get("decision_protocol_used", False)):
        return "protocol_resolved_divergence"
    if _text(row.get("final_decision_state")) == "defensive_market_exposure":
        return "market_exposure_layer"
    return _text(row.get("final_target_source")) or _text(row.get("decision_protocol_reason")) or "diagnostic_only"


def _append_reason(existing: str, reason: str) -> str:
    if not reason:
        return existing
    if not existing:
        return reason
    if reason in existing.split(";"):
        return existing
    return f"{existing};{reason}"


def _subtract(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return left - right


def _multiply(value: float | None, factor: float) -> float | None:
    return None if value is None else value * factor


def _mean(series: pd.Series) -> float | str:
    nums = pd.to_numeric(series, errors="coerce").dropna()
    return round(float(nums.mean()), 8) if not nums.empty else ""


def _median(series: pd.Series) -> float | str:
    nums = pd.to_numeric(series, errors="coerce").dropna()
    return round(float(nums.median()), 8) if not nums.empty else ""


def _win_rate(series: pd.Series) -> float | str:
    nums = pd.to_numeric(series, errors="coerce").dropna()
    return _rate((nums > 0).sum(), len(nums)) if not nums.empty else ""


def _top_value(series: pd.Series) -> str:
    values = series.astype(str)
    values = values[values.str.strip().ne("")]
    if values.empty:
        return ""
    return str(values.value_counts().index[0])


def _round_or_blank(value: float | None) -> float | str:
    return "" if value is None or pd.isna(value) else round(float(value), 8)


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _text(value: object) -> str:
    return "" if pd.isna(value) else str(value).strip()


def _rate(numerator: float | int, denominator: float | int) -> float:
    return round(float(numerator) / float(denominator), 6) if denominator else 0.0


def _summary_markdown(coverage: pd.DataFrame, by_state: pd.DataFrame) -> str:
    lines = [
        "# Final Decision Layer Forward Outcome Adapter",
        "",
        "本輸出只做 report-only event-study evidence，不改正式 selector、vote、target 或 trade action。",
        "",
        "## 邊界",
        "",
        "- formal_model_changed=false",
        "- trade_decision_changed=false",
        "- active_in_trade_decision=false",
        "- uses_forward_return_as_rule=false",
        "- 0050 / 00631L / 0050x2 只作 benchmark 或 market exposure outcome",
        "",
        "## Coverage",
        "",
    ]
    for row in coverage.to_dict(orient="records"):
        lines.append(
            f"- {row['period_label']}：complete={row['outcome_data_complete_count']}/{row['event_count']}，"
            f"rate={row['outcome_data_complete_rate']}"
        )
    lines.extend(["", "## State summary", ""])
    for row in by_state.to_dict(orient="records"):
        lines.append(
            f"- {row['final_decision_state']}：events={row['event_count']}，complete={row['complete_event_count']}，"
            f"20d_mean={row.get('forward_20d_mean', '')}"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build final decision layer report-only forward outcome adapter.")
    parser.add_argument("--state-panel", default=DEFAULT_STATE_PANEL)
    parser.add_argument("--price-cache-dir", default=DEFAULT_PRICE_CACHE_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    output = run_final_decision_layer_forward_outcome_adapter(
        state_panel_path=args.state_panel,
        price_cache_dir=args.price_cache_dir,
        output_dir=args.output_dir,
    )
    print(f"OUTPUT_DIR={output.resolve()}")


if __name__ == "__main__":
    main()
