from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from backtest_lab.execution_layer_diagnostic import build_execution_event_study_panel, build_formal_target_change_panel
from backtest_lab.partial_execution_ledger import (
    DEFAULT_INITIAL_CASH,
    DEFAULT_PRICE_CACHE_DIR,
    ExecutionVariant,
    _baseline_alignment,
    _blocked_variants,
    _build_event_context,
    _cost_turnover_summary,
    _drawdown_summary,
    _load_prices,
    _normalize_formal_daily,
    _period_performance,
    _simulate_variant,
    _validate_formal_daily,
    _variant_parameter_matrix,
)


DEFAULT_OUTPUT_DIR = "outputs/rapid_reversal_partial_switch_narrow_20260625"


def run_rapid_reversal_partial_switch_narrow(
    *,
    formal_daily_path: str | Path,
    price_cache_dir: str | Path = DEFAULT_PRICE_CACHE_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    initial_cash: float = DEFAULT_INITIAL_CASH,
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
        log("load_inputs", "started", str(formal_daily_path))
        formal_daily = pd.read_csv(formal_daily_path).fillna("")
        _validate_formal_daily(formal_daily)
        frame = _normalize_formal_daily(formal_daily)
        prices = _load_prices(frame, Path(price_cache_dir))
        if not prices:
            raise ValueError("no prices loaded for rapid reversal narrow runner")

        log("build_event_labels", "started", "")
        target_change = build_formal_target_change_panel(formal_daily)
        event_study = build_execution_event_study_panel(formal_daily, target_change, prices)
        labels = build_rapid_reversal_event_labels(frame)
        event_context = _build_event_context(frame, event_study)
        event_context = _merge_rapid_reversal_context(event_context, labels)
        forward_eval = _forward_return_evaluation_labels(labels, event_study)

        variants = _narrow_variants()
        log("simulate_variants", "started", f"variants={len(variants)}")
        daily_frames: list[pd.DataFrame] = []
        trade_frames: list[pd.DataFrame] = []
        for variant in variants:
            daily, trades = _simulate_variant(
                frame=frame,
                prices=prices,
                event_context=event_context,
                variant=variant,
                initial_cash=initial_cash,
            )
            daily_frames.append(daily)
            trade_frames.append(trades)
        daily_ledger = pd.concat(daily_frames, ignore_index=True)
        trade_ledger = pd.concat(trade_frames, ignore_index=True)

        log("build_reports", "started", "")
        variant_df = _variant_parameter_matrix(variants)
        period_perf = _period_performance(daily_ledger, prices)
        stability = _period_stability_report(period_perf)
        cost = _cost_turnover_summary(daily_ledger, trade_ledger)
        drawdown = _drawdown_summary(daily_ledger)
        event_contribution = _event_contribution_report(daily_ledger, labels)
        concentration = _trade_concentration_report(trade_ledger)
        blocked = _blocked_or_excluded_variants()
        baseline_alignment = _baseline_alignment(frame, daily_ledger, trade_ledger)

        log("write_outputs", "started", "")
        variant_df.to_csv(output / "variant_parameter_matrix.csv", index=False, encoding="utf-8-sig")
        labels.to_csv(output / "rapid_reversal_event_labels.csv", index=False, encoding="utf-8-sig")
        daily_ledger.to_csv(output / "narrow_partial_execution_daily_ledger.csv", index=False, encoding="utf-8-sig")
        trade_ledger.to_csv(output / "narrow_partial_execution_trade_ledger.csv", index=False, encoding="utf-8-sig")
        period_perf.to_csv(output / "period_performance.csv", index=False, encoding="utf-8-sig")
        stability.to_csv(output / "period_stability_report.csv", index=False, encoding="utf-8-sig")
        cost.to_csv(output / "cost_turnover_report.csv", index=False, encoding="utf-8-sig")
        drawdown.to_csv(output / "drawdown_report.csv", index=False, encoding="utf-8-sig")
        event_contribution.to_csv(output / "event_contribution_report.csv", index=False, encoding="utf-8-sig")
        forward_eval.to_csv(output / "forward_return_evaluation_labels.csv", index=False, encoding="utf-8-sig")
        concentration.to_csv(output / "trade_concentration_report.csv", index=False, encoding="utf-8-sig")
        blocked.to_csv(output / "blocked_or_excluded_variants.csv", index=False, encoding="utf-8-sig")
        (output / "baseline_vs_narrow_challenger_summary_zh.md").write_text(
            _summary_markdown(baseline_alignment, stability, cost),
            encoding="utf-8",
        )
        manifest = {
            "schema_version": 1,
            "task_id": "TASK-BACKTEST-CORE-RAPID-REVERSAL-PARTIAL-SWITCH-NARROW-001",
            "model": "rapid_reversal_partial_switch_narrow_diagnostic_only",
            "status": "completed",
            "formal_daily_path": str(formal_daily_path),
            "price_cache_dir": str(price_cache_dir),
            "formal_model_changed": False,
            "trade_decision_changed": False,
            "active_in_trade_decision": False,
            "execution_diagnostic_active_in_trade_decision": False,
            "uses_forward_return_as_rule": False,
            "valuation_used": False,
            "h3_used": False,
            "pool3_shadow_used": False,
            "final_decision_label_used": False,
            "not_formal_execution_layer": True,
            "baseline_alignment": baseline_alignment,
            "outputs": {
                "variant_parameter_matrix": "variant_parameter_matrix.csv",
                "rapid_reversal_event_labels": "rapid_reversal_event_labels.csv",
                "daily_ledger": "narrow_partial_execution_daily_ledger.csv",
                "trade_ledger": "narrow_partial_execution_trade_ledger.csv",
                "period_performance": "period_performance.csv",
                "period_stability_report": "period_stability_report.csv",
                "cost_turnover_report": "cost_turnover_report.csv",
                "drawdown_report": "drawdown_report.csv",
                "event_contribution_report": "event_contribution_report.csv",
                "forward_return_evaluation_labels": "forward_return_evaluation_labels.csv",
                "trade_concentration_report": "trade_concentration_report.csv",
                "blocked_or_excluded_variants": "blocked_or_excluded_variants.csv",
                "summary": "baseline_vs_narrow_challenger_summary_zh.md",
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
        pd.DataFrame([{"step": "run_rapid_reversal_partial_switch_narrow", "error": str(exc)}]).to_csv(
            output / "failed.csv",
            index=False,
            encoding="utf-8-sig",
        )
        log("failed", "failed", str(exc))
        raise


def build_rapid_reversal_event_labels(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    targets = frame["formal_target"].astype(str).tolist()
    dates = frame["date"].astype(str).tolist()
    previous_target = ""
    last_change_index: int | None = None
    for index, row in frame.iterrows():
        target = str(row["formal_target"]).strip()
        target_changed = bool(target and target != previous_target)
        if not target_changed:
            previous_target = target or previous_target
            continue
        labels: dict[str, Any] = {
            "date": str(row["date"]),
            "previous_target": previous_target or "none",
            "new_target": target or "none",
            "target_changed": True,
            "rapid_reversal_any_1_3": False,
            "rapid_reversal_roundtrip_1_3": False,
            "rapid_reversal_any_1_2": False,
            "rapid_reversal_roundtrip_1_2": False,
            "prior_holding_age_le_3": False,
            "reversal_offset": "",
            "roundtrip_offset": "",
            "uses_forward_return_as_rule": False,
            "execution_diagnostic_active_in_trade_decision": False,
        }
        if last_change_index is not None and index - last_change_index <= 3:
            labels["prior_holding_age_le_3"] = True
        for offset in range(1, 4):
            future_index = index + offset
            if future_index >= len(targets):
                continue
            future_target = str(targets[future_index]).strip()
            if future_target and future_target != target:
                labels["rapid_reversal_any_1_3"] = True
                labels["reversal_offset"] = offset
                if offset <= 2:
                    labels["rapid_reversal_any_1_2"] = True
                if previous_target and future_target == previous_target:
                    labels["rapid_reversal_roundtrip_1_3"] = True
                    labels["roundtrip_offset"] = offset
                    if offset <= 2:
                        labels["rapid_reversal_roundtrip_1_2"] = True
                break
        rows.append(labels)
        previous_target = target or previous_target
        last_change_index = index
    return pd.DataFrame(rows)


def _merge_rapid_reversal_context(
    event_context: dict[str, dict[str, Any]],
    labels: pd.DataFrame,
) -> dict[str, dict[str, Any]]:
    for row in labels.to_dict(orient="records"):
        date = str(row["date"])
        event_context.setdefault(date, {})
        for column in (
            "rapid_reversal_any_1_3",
            "rapid_reversal_roundtrip_1_3",
            "rapid_reversal_any_1_2",
            "rapid_reversal_roundtrip_1_2",
            "prior_holding_age_le_3",
        ):
            event_context[date][column] = bool(row.get(column, False))
        event_context[date]["is_rapid_reversal"] = bool(row.get("rapid_reversal_any_1_3", False))
    return event_context


def _narrow_variants() -> list[ExecutionVariant]:
    return [
        ExecutionVariant("baseline_full_rotation", "baseline"),
        ExecutionVariant("partial_switch_25_global_diagnostic", "control_global_risk", partial_weight=0.25),
        ExecutionVariant("rr_partial_25_any_1_3", "rr_partial", partial_weight=0.25, subset="rapid_reversal_any_1_3"),
        ExecutionVariant("rr_partial_50_any_1_3", "rr_partial", partial_weight=0.50, subset="rapid_reversal_any_1_3"),
        ExecutionVariant("rr_partial_75_any_1_3", "rr_partial", partial_weight=0.75, subset="rapid_reversal_any_1_3"),
        ExecutionVariant("rr_partial_25_roundtrip_1_3", "rr_partial", partial_weight=0.25, subset="rapid_reversal_roundtrip_1_3"),
        ExecutionVariant("rr_partial_50_roundtrip_1_3", "rr_partial", partial_weight=0.50, subset="rapid_reversal_roundtrip_1_3"),
        ExecutionVariant("rr_partial_75_roundtrip_1_3", "rr_partial", partial_weight=0.75, subset="rapid_reversal_roundtrip_1_3"),
        ExecutionVariant("rr_partial_50_any_1_2", "rr_partial", partial_weight=0.50, subset="rapid_reversal_any_1_2"),
        ExecutionVariant("rr_partial_50_roundtrip_1_2", "rr_partial", partial_weight=0.50, subset="rapid_reversal_roundtrip_1_2"),
    ]


def _forward_return_evaluation_labels(labels: pd.DataFrame, event_study: pd.DataFrame) -> pd.DataFrame:
    if labels.empty:
        return labels.copy()
    keep = [
        "date",
        "target_change_forward_return_5d",
        "target_change_forward_return_20d",
        "target_change_forward_return_60d",
        "current_holding_forward_return_5d",
        "current_holding_forward_return_20d",
        "current_holding_forward_return_60d",
        "target_minus_current_holding_forward_return_5d",
        "target_minus_current_holding_forward_return_20d",
        "target_minus_current_holding_forward_return_60d",
    ]
    available = [column for column in keep if column in event_study.columns]
    merged = labels.merge(event_study[available], on="date", how="left")
    merged["forward_return_used_as_rule"] = False
    return merged


def _period_stability_report(period_perf: pd.DataFrame) -> pd.DataFrame:
    baseline = period_perf[period_perf["variant_id"].eq("baseline_full_rotation")][
        ["period", "total_return_pct", "max_drawdown_pct", "total_transaction_cost"]
    ].rename(
        columns={
            "total_return_pct": "baseline_return_pct",
            "max_drawdown_pct": "baseline_mdd_pct",
            "total_transaction_cost": "baseline_transaction_cost",
        }
    )
    merged = period_perf.merge(baseline, on="period", how="left")
    merged["return_delta_vs_baseline_pp"] = pd.to_numeric(merged["total_return_pct"], errors="coerce") - pd.to_numeric(
        merged["baseline_return_pct"], errors="coerce"
    )
    merged["mdd_delta_vs_baseline_pp"] = pd.to_numeric(merged["max_drawdown_pct"], errors="coerce") - pd.to_numeric(
        merged["baseline_mdd_pct"], errors="coerce"
    )
    merged["cost_delta_vs_baseline"] = pd.to_numeric(merged["total_transaction_cost"], errors="coerce") - pd.to_numeric(
        merged["baseline_transaction_cost"], errors="coerce"
    )
    merged["period_support"] = (merged["return_delta_vs_baseline_pp"] >= 0) & (merged["mdd_delta_vs_baseline_pp"] >= -2)
    return merged[
        [
            "variant_id",
            "period",
            "return_delta_vs_baseline_pp",
            "mdd_delta_vs_baseline_pp",
            "cost_delta_vs_baseline",
            "period_support",
            "execution_diagnostic_active_in_trade_decision",
        ]
    ]


def _event_contribution_report(daily: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    label_dates = {
        name: set(labels.loc[labels[name].astype(bool), "date"].astype(str).tolist())
        for name in (
            "rapid_reversal_any_1_3",
            "rapid_reversal_roundtrip_1_3",
            "rapid_reversal_any_1_2",
            "rapid_reversal_roundtrip_1_2",
        )
        if name in labels.columns
    }
    rows: list[dict[str, Any]] = []
    for variant_id, group in daily.groupby("variant_id", dropna=False):
        for label, dates in label_dates.items():
            subset = group[group["date"].astype(str).isin(dates)]
            rows.append(
                {
                    "variant_id": variant_id,
                    "event_label": label,
                    "event_days": int(len(subset)),
                    "trade_days": int((pd.to_numeric(subset.get("turnover", pd.Series(dtype=float)), errors="coerce") > 0).sum()),
                    "turnover": round(float(pd.to_numeric(subset.get("turnover", pd.Series(dtype=float)), errors="coerce").sum()), 2),
                    "transaction_cost": round(float(pd.to_numeric(subset.get("transaction_cost", pd.Series(dtype=float)), errors="coerce").sum()), 2),
                    "execution_diagnostic_active_in_trade_decision": False,
                }
            )
    return pd.DataFrame(rows)


def _trade_concentration_report(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    frame = trades.copy()
    frame["date_ts"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["month"] = frame["date_ts"].dt.to_period("M").astype(str)
    frame["quarter"] = frame["date_ts"].dt.to_period("Q").astype(str)
    rows: list[dict[str, Any]] = []
    for variant_id, group in frame.groupby("variant_id", dropna=False):
        total = float(pd.to_numeric(group["gross_amount"], errors="coerce").sum())
        for dimension, column in (("ticker", "ticker"), ("month", "month"), ("quarter", "quarter")):
            if total <= 0:
                share = 0.0
                key = ""
            else:
                agg = group.groupby(column)["gross_amount"].sum().sort_values(ascending=False)
                key = str(agg.index[0]) if not agg.empty else ""
                share = float(agg.iloc[0] / total) if not agg.empty else 0.0
            rows.append(
                {
                    "variant_id": variant_id,
                    "dimension": dimension,
                    "top_key": key,
                    "top_share": round(share, 8),
                    "concentration_pass": bool(share <= 0.40),
                    "execution_diagnostic_active_in_trade_decision": False,
                }
            )
    return pd.DataFrame(rows)


def _blocked_or_excluded_variants() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "variant_id": "sell_first_then_buy_global",
                "status": "excluded",
                "reason": "cash wait underperformed new target in previous diagnostics",
                "execution_diagnostic_active_in_trade_decision": False,
            },
            {
                "variant_id": "pause_on_conflict",
                "status": "excluded",
                "reason": "final decision diagnostic and Pool3 selector veto fields are not in formal daily stream",
                "execution_diagnostic_active_in_trade_decision": False,
            },
        ]
    )


def _summary_markdown(baseline_alignment: dict[str, Any], stability: pd.DataFrame, cost: pd.DataFrame) -> str:
    candidate = stability[
        (stability["variant_id"].astype(str).str.startswith("rr_partial_"))
        & (stability["period"].astype(str).eq("full_2022_2026"))
    ].copy()
    candidate = candidate.sort_values("return_delta_vs_baseline_pp", ascending=False).head(3)
    lines = [
        "# Rapid Reversal Partial Switch Narrow Challenger",
        "",
        "本輸出是 rapid reversal partial switch 的 diagnostic narrow challenger，不是正式 execution / exit layer。",
        "",
        "## 邊界",
        "",
        "- formal_model_changed=false",
        "- trade_decision_changed=false",
        "- active_in_trade_decision=false",
        "- execution_diagnostic_active_in_trade_decision=false",
        "- uses_forward_return_as_rule=false",
        "",
        "## Baseline 對齊",
        "",
        f"- final equity diff：{baseline_alignment.get('final_equity_diff')}",
        f"- MDD diff：{baseline_alignment.get('mdd_diff')}",
        f"- trade days：{baseline_alignment.get('simulated_trade_days')}",
        "",
        "## Full period 報酬差較高的 RR variants",
        "",
    ]
    for row in candidate.to_dict(orient="records"):
        lines.append(
            f"- {row.get('variant_id')}：return delta {row.get('return_delta_vs_baseline_pp')}pp，MDD delta {row.get('mdd_delta_vs_baseline_pp')}pp"
        )
    lines.extend(["", "## 下一步", "", "交由 Experiments 驗證跨期穩定性、集中度與是否只靠單一事件貢獻。"])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build rapid reversal partial switch narrow diagnostic outputs.")
    parser.add_argument("--formal-daily", required=True)
    parser.add_argument("--price-cache-dir", default=DEFAULT_PRICE_CACHE_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--initial-cash", type=float, default=DEFAULT_INITIAL_CASH)
    args = parser.parse_args()
    output = run_rapid_reversal_partial_switch_narrow(
        formal_daily_path=args.formal_daily,
        price_cache_dir=args.price_cache_dir,
        output_dir=args.output_dir,
        initial_cash=args.initial_cash,
    )
    print(f"OUTPUT_DIR={output.resolve()}")


if __name__ == "__main__":
    main()
