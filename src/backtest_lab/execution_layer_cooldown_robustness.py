from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from backtest_lab.data import load_price_csv


DEFAULT_SOURCE_DIR = "outputs/execution_layer_next_day_ab_pool1_pool2_formal_20260626"
DEFAULT_PRICE_CACHE_DIR = "backtest_cache/stock_pool_triad_v1_corrected"
DEFAULT_OUTPUT_DIR = "outputs/execution_layer_cooldown3_robustness_20260626"
MAIN_CANDIDATE = "next_day_cooldown_after_exit_to_cash_3"
SECONDARY_CANDIDATE = "next_day_cooldown_after_exit_to_cash_2"
NEXT_DAY_BASELINE = "next_day_full_rotation"
SAME_DAY_REFERENCE = "same_day_full_rotation_reference"
REPORT_VARIANTS = (MAIN_CANDIDATE, SECONDARY_CANDIDATE, NEXT_DAY_BASELINE, SAME_DAY_REFERENCE)


def run_execution_layer_cooldown_robustness(
    *,
    source_dir: str | Path = DEFAULT_SOURCE_DIR,
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
        source = Path(source_dir)
        log("load_inputs", "started", str(source))
        daily = pd.read_csv(source / "next_day_fill_full_equity_ledger.csv").fillna("")
        trades = pd.read_csv(source / "next_day_fill_trade_ledger.csv").fillna("")
        fills = pd.read_csv(source / "fill_event_panel.csv").fillna("")
        blocked = pd.read_csv(source / "blocked_execution_events.csv").fillna("")
        alignment = pd.read_csv(source / "baseline_alignment.csv").fillna("")
        _validate_inputs(daily, trades, fills, alignment)
        daily = daily[daily["variant_id"].isin(REPORT_VARIANTS)].copy()
        trades = trades[trades["variant_id"].isin(REPORT_VARIANTS)].copy() if not trades.empty else trades
        fills = fills[fills["variant_id"].isin(REPORT_VARIANTS)].copy() if not fills.empty else fills
        blocked = blocked[blocked["variant_id"].isin(REPORT_VARIANTS)].copy() if not blocked.empty else blocked
        prices = _load_benchmark_prices(Path(price_cache_dir))
        latest = str(daily["date"].max()) if not daily.empty else ""

        log("build_reports", "started", "")
        candidate_matrix = _candidate_matrix()
        period_perf = _period_performance(daily)
        benchmark = _benchmark_comparison(period_perf, prices)
        oos = _oos_walk_forward(daily)
        leave_one = _leave_one_period(daily)
        hard_gate = _hard_gate_2024_attribution(daily, prices)
        position_concentration = _position_concentration(daily)
        trade_concentration = _trade_concentration(trades)
        cost_sensitivity = _cost_sensitivity(trades)
        blocked_attr = _blocked_event_attribution(blocked)
        readiness = _readiness_report(period_perf, hard_gate, position_concentration, alignment)

        log("write_outputs", "started", "")
        candidate_matrix.to_csv(output / "candidate_parameter_matrix.csv", index=False, encoding="utf-8-sig")
        period_perf.to_csv(output / "period_performance_by_candidate.csv", index=False, encoding="utf-8-sig")
        benchmark.to_csv(output / "benchmark_comparison_0050_00631L_0050x2.csv", index=False, encoding="utf-8-sig")
        oos.to_csv(output / "oos_walk_forward_by_candidate.csv", index=False, encoding="utf-8-sig")
        leave_one.to_csv(output / "leave_one_period_by_candidate.csv", index=False, encoding="utf-8-sig")
        hard_gate.to_csv(output / "hard_gate_2024_attribution.csv", index=False, encoding="utf-8-sig")
        position_concentration.to_csv(output / "position_concentration_by_candidate.csv", index=False, encoding="utf-8-sig")
        trade_concentration.to_csv(output / "trade_concentration_by_candidate.csv", index=False, encoding="utf-8-sig")
        cost_sensitivity.to_csv(output / "cost_sensitivity_by_candidate.csv", index=False, encoding="utf-8-sig")
        blocked_attr.to_csv(output / "blocked_event_attribution.csv", index=False, encoding="utf-8-sig")
        alignment.to_csv(output / "baseline_alignment.csv", index=False, encoding="utf-8-sig")
        readiness.to_csv(output / "cooldown_robustness_readiness_report.csv", index=False, encoding="utf-8-sig")
        (output / "execution_cooldown3_robustness_summary_zh.md").write_text(
            _summary_markdown(period_perf, hard_gate, readiness),
            encoding="utf-8",
        )
        manifest = {
            "schema_version": 1,
            "task_id": "TASK-BACKTEST-CORE-EXECUTION-LAYER-COOLDOWN3-ROBUSTNESS-001",
            "status": "completed_diagnostic_robustness",
            "formal_model_target": "combined_cap40_confirmation1_base",
            "main_candidate": MAIN_CANDIDATE,
            "secondary_candidate": SECONDARY_CANDIDATE,
            "next_day_baseline": NEXT_DAY_BASELINE,
            "same_day_reference": SAME_DAY_REFERENCE,
            "source_dir": str(source),
            "latest_complete_common_date": latest,
            "formal_model_changed": False,
            "trade_decision_changed": False,
            "active_in_trade_decision": False,
            "formal_execution_layer_activated": False,
            "formal_absorption_ready": False,
            "pool3_shadow_used": False,
            "final_decision_label_used": False,
            "rr_partial_switch_used": False,
            "uses_forward_return_as_rule": False,
            "valuation_used": False,
            "h3_used": False,
            "same_day_reference_alignment_required": True,
            "same_day_reference_max_abs_diff": _alignment_diff(alignment),
            "benchmarks_include_0050x2": True,
            "outputs": {
                "candidate_parameter_matrix": "candidate_parameter_matrix.csv",
                "period_performance": "period_performance_by_candidate.csv",
                "benchmark_comparison": "benchmark_comparison_0050_00631L_0050x2.csv",
                "oos_walk_forward": "oos_walk_forward_by_candidate.csv",
                "leave_one": "leave_one_period_by_candidate.csv",
                "hard_gate": "hard_gate_2024_attribution.csv",
                "position_concentration": "position_concentration_by_candidate.csv",
                "trade_concentration": "trade_concentration_by_candidate.csv",
                "cost_sensitivity": "cost_sensitivity_by_candidate.csv",
                "blocked_event_attribution": "blocked_event_attribution.csv",
                "readiness": "cooldown_robustness_readiness_report.csv",
                "summary": "execution_cooldown3_robustness_summary_zh.md",
            },
        }
        (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        pd.DataFrame([{"status": "completed", "output_dir": str(output.resolve())}]).to_csv(
            output / "completed.csv", index=False, encoding="utf-8-sig"
        )
        pd.DataFrame(columns=["step", "error"]).to_csv(output / "failed.csv", index=False, encoding="utf-8-sig")
        log("completed", "completed", str(output.resolve()))
        (output / "current_step.txt").write_text("completed\n", encoding="utf-8")
        return output
    except Exception as exc:
        pd.DataFrame([{"step": "run_execution_layer_cooldown_robustness", "error": str(exc)}]).to_csv(
            output / "failed.csv", index=False, encoding="utf-8-sig"
        )
        log("failed", "failed", str(exc))
        raise


def _validate_inputs(daily: pd.DataFrame, trades: pd.DataFrame, fills: pd.DataFrame, alignment: pd.DataFrame) -> None:
    missing_daily = {"variant_id", "date", "period", "portfolio_equity", "drawdown", "top_holding"} - set(daily.columns)
    if missing_daily:
        raise ValueError(f"daily ledger missing columns: {sorted(missing_daily)}")
    if not trades.empty:
        missing_trades = {"variant_id", "date", "ticker", "action", "gross_amount", "transaction_cost"} - set(trades.columns)
        if missing_trades:
            raise ValueError(f"trade ledger missing columns: {sorted(missing_trades)}")
    if not fills.empty and "variant_id" not in fills.columns:
        raise ValueError("fill event panel missing variant_id")
    if alignment.empty or "alignment_state" not in alignment.columns:
        raise ValueError("baseline alignment missing or invalid")


def _candidate_matrix() -> pd.DataFrame:
    rows = [
        (MAIN_CANDIDATE, "main_diagnostic_candidate", "cooldown_after_exit_to_cash_3", False),
        (SECONDARY_CANDIDATE, "secondary_diagnostic_candidate", "cooldown_after_exit_to_cash_2", False),
        (NEXT_DAY_BASELINE, "next_day_execution_baseline", "next_day_full_rotation", False),
        (SAME_DAY_REFERENCE, "alignment_reference_not_candidate", "same_day_reference", False),
    ]
    return pd.DataFrame(
        [
            {
                "candidate": candidate,
                "role": role,
                "execution_rule_family": family,
                "formal_absorption_ready": ready,
                "formal_model_changed": False,
                "trade_decision_changed": False,
                "active_in_trade_decision": False,
            }
            for candidate, role, family, ready in rows
        ]
    )


def _period_performance(daily: pd.DataFrame) -> pd.DataFrame:
    periods = {
        "2022": ("2022-01-01", "2022-12-31"),
        "2023": ("2023-01-01", "2023-12-31"),
        "2024_now": ("2024-01-01", None),
        "2024_hard_gate": ("2024-01-01", "2024-12-31"),
        "2026_ytd": ("2026-01-01", None),
        "full": (None, None),
    }
    frame = daily.copy()
    frame["date_ts"] = pd.to_datetime(frame["date"])
    rows: list[dict[str, Any]] = []
    for variant, group in frame.groupby("variant_id"):
        for label, (start, end) in periods.items():
            subset = group
            if start:
                subset = subset[subset["date_ts"] >= pd.Timestamp(start)]
            if end:
                subset = subset[subset["date_ts"] <= pd.Timestamp(end)]
            rows.append(_perf_row(variant, label, subset))
    return pd.DataFrame(rows)


def _oos_walk_forward(daily: pd.DataFrame) -> pd.DataFrame:
    tests = {
        "train_2022_2024": ("2022-01-01", "2024-12-31"),
        "test_2025_2026": ("2025-01-01", None),
        "train_2022_2023": ("2022-01-01", "2023-12-31"),
        "test_2024_now": ("2024-01-01", None),
        "train_2022_2025": ("2022-01-01", "2025-12-31"),
        "test_2026_ytd": ("2026-01-01", None),
        "post_2026_exclusion": ("2022-01-01", "2025-12-31"),
    }
    frame = daily[daily["variant_id"].isin((MAIN_CANDIDATE, SECONDARY_CANDIDATE, NEXT_DAY_BASELINE))].copy()
    frame["date_ts"] = pd.to_datetime(frame["date"])
    rows: list[dict[str, Any]] = []
    for variant, group in frame.groupby("variant_id"):
        for label, (start, end) in tests.items():
            subset = group[group["date_ts"] >= pd.Timestamp(start)]
            if end:
                subset = subset[subset["date_ts"] <= pd.Timestamp(end)]
            rows.append(_perf_row(variant, label, subset))
    return pd.DataFrame(rows)


def _leave_one_period(daily: pd.DataFrame) -> pd.DataFrame:
    frame = daily[daily["variant_id"].isin((MAIN_CANDIDATE, SECONDARY_CANDIDATE, NEXT_DAY_BASELINE))].copy()
    frame["date_ts"] = pd.to_datetime(frame["date"])
    frame["year"] = frame["date_ts"].dt.year.astype(str)
    frame["quarter"] = frame["date_ts"].dt.to_period("Q").astype(str)
    frame["month"] = frame["date_ts"].dt.to_period("M").astype(str)
    rows: list[dict[str, Any]] = []
    for variant, group in frame.groupby("variant_id"):
        for level in ("year", "quarter", "month"):
            values = sorted(group[level].unique()) if level == "year" else group[level].value_counts().head(18).index.tolist()
            for value in values:
                row = _perf_row(variant, f"leave_one_{level}_{value}", group[~group[level].eq(value)])
                row["excluded_level"] = level
                row["excluded_value"] = value
                rows.append(row)
    return pd.DataFrame(rows)


def _hard_gate_2024_attribution(daily: pd.DataFrame, prices: dict[str, pd.Series]) -> pd.DataFrame:
    perf = _period_performance(daily)
    hard = perf[perf["period_label"] == "2024_hard_gate"].copy()
    bench = _benchmark_return(prices.get("0050.TW"), "2024-01-02", "2024-12-31", multiplier=2.0)
    rows: list[dict[str, Any]] = []
    frame = daily.copy()
    frame["date_ts"] = pd.to_datetime(frame["date"])
    segment = frame[(frame["date_ts"] >= pd.Timestamp("2024-01-01")) & (frame["date_ts"] <= pd.Timestamp("2024-12-31"))]
    for item in hard.to_dict(orient="records"):
        variant = item["variant_id"]
        group = segment[segment["variant_id"] == variant]
        rows.append(
            {
                "variant_id": variant,
                "period_label": "2024_hard_gate",
                "candidate_return_pct": item.get("return_pct", ""),
                "candidate_mdd_pct": item.get("max_drawdown_pct", ""),
                "0050x2_return_pct": _pct(bench),
                "excess_vs_0050x2_pct": round(float(item.get("return_pct", 0.0)) - _pct(bench), 4) if bench is not None else "",
                "trade_event_days": int((pd.to_numeric(group.get("pending_order_count", pd.Series(dtype=float)), errors="coerce").fillna(0) > 0).sum()) if not group.empty else 0,
                "cash_day_share": _mean(group.get("cash_weight", pd.Series(dtype=float))) if not group.empty else "",
                "diagnostic_only": True,
                "active_in_trade_decision": False,
            }
        )
    return pd.DataFrame(rows)


def _benchmark_comparison(performance: pd.DataFrame, prices: dict[str, pd.Series]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    benchmarks = {"0050": ("0050.TW", 1.0), "00631L": ("00631L.TW", 1.0), "0050x2": ("0050.TW", 2.0)}
    for item in performance.to_dict(orient="records"):
        if str(item.get("status")) != "completed":
            continue
        for label, (ticker, multiplier) in benchmarks.items():
            bench = _benchmark_return(prices.get(ticker), str(item.get("start_date")), str(item.get("end_date")), multiplier=multiplier)
            rows.append(
                {
                    "variant_id": item["variant_id"],
                    "period_label": item["period_label"],
                    "benchmark": label,
                    "candidate_return_pct": item.get("return_pct", ""),
                    "benchmark_return_pct": _pct(bench),
                    "excess_return_pct": round(float(item.get("return_pct", 0.0)) - _pct(bench), 4) if bench is not None else "",
                    "same_date_range": True,
                    "diagnostic_only": True,
                    "active_in_trade_decision": False,
                }
            )
    return pd.DataFrame(rows)


def _position_concentration(daily: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for variant, group in daily.groupby("variant_id"):
        counts = group["top_holding"].astype(str).replace("", "cash").value_counts()
        top_ticker = str(counts.index[0]) if not counts.empty else ""
        top_share = float(counts.iloc[0] / len(group)) if len(group) and not counts.empty else 0.0
        rows.append(
            {
                "variant_id": variant,
                "top_holding": top_ticker,
                "top_holding_day_share": round(top_share, 6),
                "00631L_day_share": round(float((group["top_holding"].astype(str) == "00631L.TW").sum() / len(group)), 6) if len(group) else 0.0,
                "cash_top_holding_day_share": round(float((group["top_holding"].astype(str) == "cash").sum() / len(group)), 6) if len(group) else 0.0,
                "concentration_warning": top_share > 0.40 and top_ticker != "cash",
                "diagnostic_only": True,
                "active_in_trade_decision": False,
            }
        )
    return pd.DataFrame(rows)


def _trade_concentration(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(columns=["variant_id", "level", "top_value", "top_turnover_share", "diagnostic_only", "active_in_trade_decision"])
    frame = trades.copy()
    frame["date_ts"] = pd.to_datetime(frame["date"])
    frame["month"] = frame["date_ts"].dt.to_period("M").astype(str)
    frame["quarter"] = frame["date_ts"].dt.to_period("Q").astype(str)
    frame["turnover"] = pd.to_numeric(frame["gross_amount"], errors="coerce").fillna(0.0)
    rows: list[dict[str, Any]] = []
    for variant, group in frame.groupby("variant_id"):
        total = float(group["turnover"].sum())
        for level in ("ticker", "month", "quarter"):
            key = "ticker" if level == "ticker" else level
            by_level = group.groupby(key)["turnover"].sum().sort_values(ascending=False)
            top_value = str(by_level.index[0]) if not by_level.empty else ""
            top_share = float(by_level.iloc[0] / total) if total and not by_level.empty else 0.0
            rows.append(
                {
                    "variant_id": variant,
                    "level": level,
                    "top_value": top_value,
                    "top_turnover_share": round(top_share, 6),
                    "concentration_warning": top_share > 0.40,
                    "diagnostic_only": True,
                    "active_in_trade_decision": False,
                }
            )
    return pd.DataFrame(rows)


def _cost_sensitivity(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(columns=["variant_id", "total_turnover", "total_transaction_cost", "cost_10bp", "cost_20bp"])
    frame = trades.copy()
    frame["turnover"] = pd.to_numeric(frame["gross_amount"], errors="coerce").fillna(0.0)
    frame["cost"] = pd.to_numeric(frame["transaction_cost"], errors="coerce").fillna(0.0)
    rows: list[dict[str, Any]] = []
    for variant, group in frame.groupby("variant_id"):
        turnover = float(group["turnover"].sum())
        rows.append(
            {
                "variant_id": variant,
                "trade_rows": int(len(group)),
                "total_turnover": round(turnover, 2),
                "total_transaction_cost": round(float(group["cost"].sum()), 2),
                "cost_10bp": round(turnover * 0.001, 2),
                "cost_20bp": round(turnover * 0.002, 2),
                "diagnostic_only": True,
                "active_in_trade_decision": False,
            }
        )
    return pd.DataFrame(rows)


def _blocked_event_attribution(blocked: pd.DataFrame) -> pd.DataFrame:
    if blocked.empty:
        return pd.DataFrame(
            [{"variant_id": "", "blocked_reason": "none", "blocked_count": 0, "diagnostic_only": True, "active_in_trade_decision": False}]
        )
    rows: list[dict[str, Any]] = []
    for (variant, reason), group in blocked.groupby(["variant_id", "blocked_reason"], dropna=False):
        rows.append(
            {
                "variant_id": variant,
                "blocked_reason": reason,
                "blocked_count": int(len(group)),
                "diagnostic_only": True,
                "active_in_trade_decision": False,
            }
        )
    return pd.DataFrame(rows)


def _readiness_report(
    performance: pd.DataFrame,
    hard_gate: pd.DataFrame,
    concentration: pd.DataFrame,
    alignment: pd.DataFrame,
) -> pd.DataFrame:
    full = performance[(performance["variant_id"] == MAIN_CANDIDATE) & (performance["period_label"] == "full")]
    hard = hard_gate[hard_gate["variant_id"] == MAIN_CANDIDATE]
    conc = concentration[concentration["variant_id"] == MAIN_CANDIDATE]
    alignment_pass = bool(str(alignment.iloc[0].get("alignment_state", "")) == "passed") if not alignment.empty else False
    blockers = []
    if not alignment_pass:
        blockers.append("baseline_alignment_not_passed")
    if not hard.empty and float(hard.iloc[0].get("excess_vs_0050x2_pct", 0.0)) < 0:
        blockers.append("2024_hard_gate_underperforms_0050x2")
    if not conc.empty and bool(conc.iloc[0].get("concentration_warning", False)):
        blockers.append("position_concentration_warning")
    return pd.DataFrame(
        [
            {
                "candidate": MAIN_CANDIDATE,
                "readiness_state": "promising_diagnostic_not_formal_ready",
                "formal_absorption_ready": False,
                "blocker_count": len(blockers),
                "blockers": ";".join(blockers),
                "full_return_pct": full.iloc[0].get("return_pct", "") if not full.empty else "",
                "full_mdd_pct": full.iloc[0].get("max_drawdown_pct", "") if not full.empty else "",
                "next_step": "experiments_validate_cooldown3_robustness",
                "diagnostic_only": True,
                "active_in_trade_decision": False,
            }
        ]
    )


def _perf_row(variant: str, label: str, frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {"variant_id": variant, "period_label": label, "status": "no_rows"}
    equity = pd.to_numeric(frame["portfolio_equity"], errors="coerce")
    start = float(equity.iloc[0])
    end = float(equity.iloc[-1])
    return {
        "variant_id": variant,
        "period_label": label,
        "status": "completed",
        "start_date": str(frame["date"].iloc[0]),
        "end_date": str(frame["date"].iloc[-1]),
        "start_equity": round(start, 2),
        "final_equity": round(end, 2),
        "return_pct": round((end / start - 1) * 100, 4) if start else 0.0,
        "max_drawdown_pct": round(float(pd.to_numeric(frame["drawdown"], errors="coerce").min()) * 100, 4),
        "row_count": int(len(frame)),
        "diagnostic_only": True,
        "active_in_trade_decision": False,
    }


def _load_benchmark_prices(price_cache_dir: Path) -> dict[str, pd.Series]:
    prices: dict[str, pd.Series] = {}
    for ticker in ("0050.TW", "00631L.TW"):
        path = price_cache_dir / f"{ticker.replace('.', '_')}.csv"
        if not path.exists():
            path = price_cache_dir / f"{ticker}.csv"
        if path.exists():
            prices[ticker] = pd.to_numeric(load_price_csv(path)["adj_close"], errors="coerce").dropna()
    return prices


def _benchmark_return(series: pd.Series | None, start_date: str, end_date: str, *, multiplier: float = 1.0) -> float | None:
    if series is None or series.empty:
        return None
    start = _price_on_or_after(series, start_date)
    end = _price_on_or_before(series, end_date)
    if start is None or end is None or start <= 0:
        return None
    return (end / start - 1) * multiplier


def _price_on_or_after(series: pd.Series, date: str) -> float | None:
    valid = series.loc[series.index >= pd.Timestamp(date)]
    return float(valid.iloc[0]) if not valid.empty else None


def _price_on_or_before(series: pd.Series, date: str) -> float | None:
    valid = series.loc[series.index <= pd.Timestamp(date)]
    return float(valid.iloc[-1]) if not valid.empty else None


def _pct(value: float | None) -> float | str:
    return round(float(value) * 100, 4) if value is not None else ""


def _mean(series: pd.Series) -> float | str:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return round(float(values.mean()), 6) if not values.empty else ""


def _alignment_diff(alignment: pd.DataFrame) -> float | str:
    if alignment.empty or "max_abs_diff" not in alignment.columns:
        return ""
    value = pd.to_numeric(alignment["max_abs_diff"], errors="coerce")
    return round(float(value.iloc[0]), 6) if value.notna().any() else ""


def _summary_markdown(performance: pd.DataFrame, hard_gate: pd.DataFrame, readiness: pd.DataFrame) -> str:
    full = performance[(performance["variant_id"] == MAIN_CANDIDATE) & (performance["period_label"] == "full")]
    base = performance[(performance["variant_id"] == NEXT_DAY_BASELINE) & (performance["period_label"] == "full")]
    main = full.iloc[0].to_dict() if not full.empty else {}
    baseline = base.iloc[0].to_dict() if not base.empty else {}
    ready = readiness.iloc[0].to_dict() if not readiness.empty else {}
    return "\n".join(
        [
            "# Cooldown after exit-to-cash 3 robustness diagnostic",
            "",
            "本輸出檢查 `next_day_cooldown_after_exit_to_cash_3` 是否具備進一步 formal execution challenger 的證據。這不是正式 execution layer 上線。",
            "",
            "## 主要結果",
            f"- 主候選 full return：{main.get('return_pct', '')}%，MDD：{main.get('max_drawdown_pct', '')}%",
            f"- next-day baseline full return：{baseline.get('return_pct', '')}%，MDD：{baseline.get('max_drawdown_pct', '')}%",
            f"- readiness：{ready.get('readiness_state', '')}",
            f"- blockers：{ready.get('blockers', '')}",
            "",
            "## 邊界",
            "- formal_model_changed=false",
            "- trade_decision_changed=false",
            "- active_in_trade_decision=false",
            "- forward return 未作規則；Pool3、final decision label、RR partial switch、valuation/H3 均未使用。",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build cooldown3 execution robustness panels.")
    parser.add_argument("--source-dir", default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--price-cache-dir", default=DEFAULT_PRICE_CACHE_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    run_execution_layer_cooldown_robustness(
        source_dir=args.source_dir,
        price_cache_dir=args.price_cache_dir,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
