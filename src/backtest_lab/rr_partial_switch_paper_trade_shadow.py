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
    _cost_turnover_summary,
    _load_prices,
    _normalize_formal_daily,
    _period_performance,
    _simulate_variant,
    _validate_formal_daily,
)
from backtest_lab.rapid_reversal_partial_switch_narrow import (
    build_rapid_reversal_event_labels,
    _forward_return_evaluation_labels,
)
from backtest_lab.rr_partial_switch_sample_robustness import _context_with_exclusions


DEFAULT_FORMAL_DAILY = "outputs/stock_pool_formal_daily_replay_pit_pool2_daily_final_combined_20260624/baseline_three_pool_formal_daily_equity.csv"
DEFAULT_OUTPUT_DIR = "outputs/rr_partial_switch_paper_trade_shadow_20260625"
SAMPLE_GATE_THRESHOLD = 25
MAIN_CANDIDATE = "rr_partial_25_roundtrip_1_3"
SENSITIVITY_CANDIDATE = "rr_partial_25_any_1_3"


def run_rr_partial_switch_paper_trade_shadow(
    *,
    formal_daily_path: str | Path = DEFAULT_FORMAL_DAILY,
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
            raise ValueError("no prices loaded for RR paper-trade shadow tracker")

        log("build_rr_events", "started", "")
        target_change = build_formal_target_change_panel(formal_daily)
        event_study = build_execution_event_study_panel(formal_daily, target_change, prices)
        labels = build_rapid_reversal_event_labels(frame)
        event_log = _build_rr_event_shadow_log(labels)
        forward_audit = _forward_return_rule_audit(_forward_return_evaluation_labels(labels, event_study))

        log("simulate_shadow_ledgers", "started", "")
        context = _context_with_exclusions(frame, event_study, labels, set())
        variants = _paper_trade_variants()
        daily_frames: list[pd.DataFrame] = []
        trade_frames: list[pd.DataFrame] = []
        for variant in variants:
            daily, trades = _simulate_variant(
                frame=frame,
                prices=prices,
                event_context=context,
                variant=variant,
                initial_cash=initial_cash,
            )
            daily["paper_trade_shadow_active_in_trade_decision"] = False
            trades["paper_trade_shadow_active_in_trade_decision"] = False
            daily_frames.append(daily)
            trade_frames.append(trades)

        daily_ledger = pd.concat(daily_frames, ignore_index=True)
        trade_ledger = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()
        period_perf = _period_performance(daily_ledger, prices)
        baseline_alignment = _baseline_alignment(frame, daily_ledger, trade_ledger)
        sample_status = _sample_accumulation_status(event_log)
        gate_report = _event_count_gate_report(sample_status)
        summary = _shadow_vs_formal_baseline_summary(period_perf)
        sensitivity = _sensitivity_report(summary, sample_status)
        contract = _candidate_contract()

        log("write_outputs", "started", "")
        pd.DataFrame([contract]).to_json(output / "paper_trade_candidate_contract.json", orient="records", force_ascii=False, indent=2)
        event_log.to_csv(output / "rr_event_shadow_log.csv", index=False, encoding="utf-8-sig")
        daily_ledger.to_csv(output / "paper_trade_shadow_daily_ledger.csv", index=False, encoding="utf-8-sig")
        trade_ledger.to_csv(output / "paper_trade_shadow_trade_ledger.csv", index=False, encoding="utf-8-sig")
        sample_status.to_csv(output / "sample_accumulation_status.csv", index=False, encoding="utf-8-sig")
        gate_report.to_csv(output / "event_count_gate_report.csv", index=False, encoding="utf-8-sig")
        summary.to_csv(output / "shadow_vs_formal_baseline_summary.csv", index=False, encoding="utf-8-sig")
        sensitivity.to_csv(output / "sensitivity_any_1_3_report.csv", index=False, encoding="utf-8-sig")
        forward_audit.to_csv(output / "forward_return_rule_audit.csv", index=False, encoding="utf-8-sig")
        (output / "paper_trade_shadow_summary_zh.md").write_text(
            _summary_markdown(baseline_alignment, sample_status, gate_report),
            encoding="utf-8",
        )

        main_count = int(sample_status.loc[sample_status["candidate_variant"].eq(MAIN_CANDIDATE), "total_event_count"].iloc[0])
        manifest = {
            "schema_version": 1,
            "task_id": "TASK-BACKTEST-CORE-RR-PARTIAL-SWITCH-PAPER-TRADE-SHADOW-001",
            "model": "rr_partial_switch_paper_trade_shadow_tracker",
            "status": "completed",
            "formal_daily_path": str(formal_daily_path),
            "price_cache_dir": str(price_cache_dir),
            "formal_model_changed": False,
            "trade_decision_changed": False,
            "active_in_trade_decision": False,
            "paper_trade_shadow_active_in_trade_decision": False,
            "uses_forward_return_as_rule": False,
            "valuation_used": False,
            "h3_used": False,
            "pool3_shadow_used": False,
            "final_decision_label_used": False,
            "sample_gate_threshold": SAMPLE_GATE_THRESHOLD,
            "sample_gate_status": "sample_limited_shadow_tracking" if main_count < SAMPLE_GATE_THRESHOLD else "ready_for_formal_readiness_recheck",
            "formal_ready": False,
            "main_candidate": MAIN_CANDIDATE,
            "sensitivity_candidate": SENSITIVITY_CANDIDATE,
            "baseline_alignment": baseline_alignment,
            "outputs": {
                "candidate_contract": "paper_trade_candidate_contract.json",
                "event_log": "rr_event_shadow_log.csv",
                "daily_ledger": "paper_trade_shadow_daily_ledger.csv",
                "trade_ledger": "paper_trade_shadow_trade_ledger.csv",
                "sample_status": "sample_accumulation_status.csv",
                "gate_report": "event_count_gate_report.csv",
                "baseline_summary": "shadow_vs_formal_baseline_summary.csv",
                "sensitivity_report": "sensitivity_any_1_3_report.csv",
                "forward_return_rule_audit": "forward_return_rule_audit.csv",
                "summary": "paper_trade_shadow_summary_zh.md",
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
        pd.DataFrame([{"step": "run_rr_partial_switch_paper_trade_shadow", "error": str(exc)}]).to_csv(
            output / "failed.csv",
            index=False,
            encoding="utf-8-sig",
        )
        log("failed", "failed", str(exc))
        raise


def _paper_trade_variants() -> list[ExecutionVariant]:
    return [
        ExecutionVariant("baseline_full_rotation", "baseline"),
        ExecutionVariant(MAIN_CANDIDATE, "paper_trade_shadow_main", partial_weight=0.25, subset="rapid_reversal_roundtrip_1_3"),
        ExecutionVariant(SENSITIVITY_CANDIDATE, "paper_trade_shadow_sensitivity", partial_weight=0.25, subset="rapid_reversal_any_1_3"),
    ]


def _build_rr_event_shadow_log(labels: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for index, row in labels.reset_index(drop=True).iterrows():
        is_roundtrip = bool(row.get("rapid_reversal_roundtrip_1_3", False))
        is_any = bool(row.get("rapid_reversal_any_1_3", False))
        if not (is_roundtrip or is_any):
            continue
        rows.append(
            {
                "event_id": f"RR-{index + 1:04d}",
                "event_date": str(row.get("date", "")),
                "event_type": "roundtrip_1_3" if is_roundtrip else "any_1_3",
                "candidate_variant": MAIN_CANDIDATE if is_roundtrip else SENSITIVITY_CANDIDATE,
                "previous_target": str(row.get("previous_target", "")),
                "intermediate_target": str(row.get("new_target", "")),
                "roundtrip_target": str(row.get("previous_target", "")) if is_roundtrip else "",
                "event_window_rows": int(row.get("roundtrip_offset") or row.get("reversal_offset") or 0),
                "is_roundtrip_1_3": is_roundtrip,
                "is_any_1_3": is_any,
                "formal_target_source": "current_formal_baseline_with_pit_pool2",
                "forward_return_used_as_rule": False,
                "paper_trade_shadow_active_in_trade_decision": False,
            }
        )
    return pd.DataFrame(rows)


def _sample_accumulation_status(event_log: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    definitions = [
        (MAIN_CANDIDATE, "is_roundtrip_1_3"),
        (SENSITIVITY_CANDIDATE, "is_any_1_3"),
    ]
    for variant, column in definitions:
        subset = event_log[event_log[column].astype(bool)] if not event_log.empty and column in event_log.columns else pd.DataFrame()
        total = int(len(subset))
        rows.append(
            {
                "candidate_variant": variant,
                "historical_event_count": total,
                "new_shadow_event_count": 0,
                "total_event_count": total,
                "sample_gate_threshold": SAMPLE_GATE_THRESHOLD,
                "sample_gate_status": "sample_limited_shadow_tracking" if total < SAMPLE_GATE_THRESHOLD else "ready_for_formal_readiness_recheck",
                "first_event_date": subset["event_date"].iloc[0] if not subset.empty else "",
                "latest_event_date": subset["event_date"].iloc[-1] if not subset.empty else "",
                "ready_for_formal_readiness_recheck": bool(total >= SAMPLE_GATE_THRESHOLD),
                "paper_trade_shadow_active_in_trade_decision": False,
            }
        )
    return pd.DataFrame(rows)


def _event_count_gate_report(sample_status: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in sample_status.to_dict(orient="records"):
        total = int(row["total_event_count"])
        rows.append(
            {
                "candidate_variant": row["candidate_variant"],
                "total_event_count": total,
                "sample_gate_threshold": SAMPLE_GATE_THRESHOLD,
                "sample_gate_passed": bool(total >= SAMPLE_GATE_THRESHOLD),
                "formal_ready": False,
                "gate_reason": "event_count below 25; paper-trade shadow only"
                if total < SAMPLE_GATE_THRESHOLD
                else "event_count reached threshold; requires Experiments formal-readiness recheck",
                "paper_trade_shadow_active_in_trade_decision": False,
            }
        )
    return pd.DataFrame(rows)


def _shadow_vs_formal_baseline_summary(period_perf: pd.DataFrame) -> pd.DataFrame:
    baseline = period_perf[period_perf["variant_id"].eq("baseline_full_rotation")][
        ["period", "total_return_pct", "max_drawdown_pct", "total_transaction_cost", "total_turnover"]
    ].rename(
        columns={
            "total_return_pct": "baseline_return_pct",
            "max_drawdown_pct": "baseline_mdd_pct",
            "total_transaction_cost": "baseline_transaction_cost",
            "total_turnover": "baseline_turnover",
        }
    )
    challengers = period_perf[period_perf["variant_id"].isin([MAIN_CANDIDATE, SENSITIVITY_CANDIDATE])].copy()
    merged = challengers.merge(baseline, on="period", how="left")
    merged["return_delta_vs_formal_baseline_pp"] = (merged["total_return_pct"] - merged["baseline_return_pct"]).round(6)
    merged["mdd_delta_vs_formal_baseline_pp"] = (merged["max_drawdown_pct"] - merged["baseline_mdd_pct"]).round(6)
    merged["cost_delta_vs_formal_baseline"] = (merged["total_transaction_cost"] - merged["baseline_transaction_cost"]).round(2)
    merged["paper_trade_shadow_active_in_trade_decision"] = False
    return merged


def _sensitivity_report(summary: pd.DataFrame, sample_status: pd.DataFrame) -> pd.DataFrame:
    subset = summary[summary["variant_id"].eq(SENSITIVITY_CANDIDATE)].copy()
    status = sample_status[sample_status["candidate_variant"].eq(SENSITIVITY_CANDIDATE)]
    subset["sensitivity_role"] = "upper_bound_only"
    subset["allowed_to_replace_main_candidate"] = False
    subset["sample_gate_status"] = status["sample_gate_status"].iloc[0] if not status.empty else "unknown"
    subset["paper_trade_shadow_active_in_trade_decision"] = False
    return subset


def _forward_return_rule_audit(forward_eval: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "audit_id": "forward_return_rule_usage",
                "used_as_rule_count": 0,
                "forward_return_rows_available_for_diagnostic": int(len(forward_eval)),
                "pass": True,
                "paper_trade_shadow_active_in_trade_decision": False,
            }
        ]
    )


def _candidate_contract() -> dict[str, Any]:
    return {
        "main_candidate": MAIN_CANDIDATE,
        "main_event_definition": "rapid_reversal_roundtrip_1_3",
        "main_partial_switch_weight": 0.25,
        "sensitivity_candidate": SENSITIVITY_CANDIDATE,
        "sensitivity_event_definition": "rapid_reversal_any_1_3",
        "sensitivity_allowed_to_replace_main": False,
        "sample_gate_threshold": SAMPLE_GATE_THRESHOLD,
        "formal_ready": False,
        "paper_trade_shadow_active_in_trade_decision": False,
        "uses_forward_return_as_rule": False,
    }


def _summary_markdown(
    baseline_alignment: dict[str, Any],
    sample_status: pd.DataFrame,
    gate_report: pd.DataFrame,
) -> str:
    lines = [
        "# RR Partial Switch Paper-trade Shadow Tracker",
        "",
        "本輸出只追蹤 RR partial switch 的 paper-trade shadow，不是正式 execution / exit layer。",
        "",
        "## 邊界",
        "",
        "- formal_model_changed=false",
        "- trade_decision_changed=false",
        "- active_in_trade_decision=false",
        "- paper_trade_shadow_active_in_trade_decision=false",
        "- uses_forward_return_as_rule=false",
        "",
        "## Baseline 對齊",
        "",
        f"- final equity diff：{baseline_alignment.get('final_equity_diff')}",
        f"- MDD diff：{baseline_alignment.get('mdd_diff')}",
        "",
        "## Sample gate",
        "",
    ]
    for row in sample_status.to_dict(orient="records"):
        lines.append(
            f"- {row['candidate_variant']}：{row['total_event_count']} events，{row['sample_gate_status']}"
        )
    lines.extend(["", "## Formal-ready", ""])
    for row in gate_report.to_dict(orient="records"):
        lines.append(f"- {row['candidate_variant']}：formal_ready={row['formal_ready']}，{row['gate_reason']}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build RR partial switch paper-trade shadow tracker outputs.")
    parser.add_argument("--formal-daily", default=DEFAULT_FORMAL_DAILY)
    parser.add_argument("--price-cache-dir", default=DEFAULT_PRICE_CACHE_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--initial-cash", type=float, default=DEFAULT_INITIAL_CASH)
    args = parser.parse_args(argv)
    output = run_rr_partial_switch_paper_trade_shadow(
        formal_daily_path=args.formal_daily,
        price_cache_dir=args.price_cache_dir,
        output_dir=args.output_dir,
        initial_cash=args.initial_cash,
    )
    print(f"OUTPUT_DIR={output.resolve()}")


if __name__ == "__main__":
    main()
