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
    _drawdown_summary,
    _load_prices,
    _normalize_formal_daily,
    _period_performance,
    _simulate_variant,
    _validate_formal_daily,
)
from backtest_lab.rapid_reversal_partial_switch_narrow import (
    build_rapid_reversal_event_labels,
    _forward_return_evaluation_labels,
    _merge_rapid_reversal_context,
    _period_stability_report,
    _trade_concentration_report,
)


DEFAULT_OUTPUT_DIR = "outputs/rr_partial_switch_sample_robustness_20260625"
MAIN_CANDIDATE = "rr_partial_25_roundtrip_1_3"
SENSITIVITY_CANDIDATE = "rr_partial_25_any_1_3"


def run_rr_partial_switch_sample_robustness(
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
            raise ValueError("no prices loaded for RR sample robustness runner")

        log("build_event_labels", "started", "")
        target_change = build_formal_target_change_panel(formal_daily)
        event_study = build_execution_event_study_panel(formal_daily, target_change, prices)
        labels = build_rapid_reversal_event_labels(frame)
        forward_eval = _forward_return_evaluation_labels(labels, event_study)
        sample_audit = _event_sample_audit(labels)
        sample_limited = bool(sample_audit["event_count"].max() < 25) if not sample_audit.empty else True

        log("simulate_base_candidates", "started", "")
        daily, trades, period_perf, stability = _simulate_candidate_set(
            frame=frame,
            event_study=event_study,
            labels=labels,
            prices=prices,
            initial_cash=initial_cash,
        )
        baseline_alignment = _baseline_alignment(frame, daily, trades)
        candidate_matrix = _candidate_matrix(sample_audit)

        log("leave_one_reports", "started", "")
        baseline_full = _baseline_full_metrics(daily)
        leave_event = _leave_one_event_report(frame, event_study, labels, prices, initial_cash, baseline_full)
        leave_month = _leave_group_report(frame, event_study, labels, prices, initial_cash, baseline_full, group="month")
        leave_quarter = _leave_group_report(frame, event_study, labels, prices, initial_cash, baseline_full, group="quarter")
        leave_year = _leave_group_report(frame, event_study, labels, prices, initial_cash, baseline_full, group="year")
        post_2026 = _post_2026_exclusion_report(frame, event_study, labels, prices, initial_cash, baseline_full)
        train_test = _train_test_report(daily)
        pre_2026 = _pre_2026_sanity_report(daily)
        largest = _largest_contribution_exclusion_report(leave_event, leave_month, leave_quarter)
        oos_gate = _oos_gate_report(stability, leave_event, leave_month, leave_quarter, leave_year, post_2026, train_test)
        mdd_report = _mdd_non_degradation_report(stability)
        concentration = _trade_concentration_report(trades)
        forward_audit = _forward_return_rule_audit(forward_eval)
        demoted = _blocked_or_demoted_candidates(sample_limited, oos_gate)
        cost = _cost_turnover_summary(daily, trades)
        drawdown = _drawdown_summary(daily)

        log("write_outputs", "started", "")
        candidate_matrix.to_csv(output / "candidate_matrix.csv", index=False, encoding="utf-8-sig")
        sample_audit.to_csv(output / "event_sample_audit.csv", index=False, encoding="utf-8-sig")
        leave_event.to_csv(output / "leave_one_event_report.csv", index=False, encoding="utf-8-sig")
        leave_month.to_csv(output / "leave_one_month_report.csv", index=False, encoding="utf-8-sig")
        leave_quarter.to_csv(output / "leave_one_quarter_report.csv", index=False, encoding="utf-8-sig")
        leave_year.to_csv(output / "leave_one_year_report.csv", index=False, encoding="utf-8-sig")
        post_2026.to_csv(output / "post_2026_exclusion_report.csv", index=False, encoding="utf-8-sig")
        train_test.to_csv(output / "train_2022_2024_test_2025_2026_report.csv", index=False, encoding="utf-8-sig")
        pre_2026.to_csv(output / "pre_2026_sanity_report.csv", index=False, encoding="utf-8-sig")
        largest.to_csv(output / "largest_contribution_exclusion_report.csv", index=False, encoding="utf-8-sig")
        oos_gate.to_csv(output / "oos_gate_report.csv", index=False, encoding="utf-8-sig")
        mdd_report.to_csv(output / "mdd_non_degradation_report.csv", index=False, encoding="utf-8-sig")
        concentration.to_csv(output / "trade_concentration_report.csv", index=False, encoding="utf-8-sig")
        forward_audit.to_csv(output / "forward_return_rule_audit.csv", index=False, encoding="utf-8-sig")
        demoted.to_csv(output / "blocked_or_demoted_candidates.csv", index=False, encoding="utf-8-sig")
        (output / "baseline_vs_rr_sample_robustness_summary_zh.md").write_text(
            _summary_markdown(baseline_alignment, sample_audit, oos_gate, demoted),
            encoding="utf-8",
        )
        manifest = {
            "schema_version": 1,
            "task_id": "TASK-BACKTEST-CORE-RR-PARTIAL-SWITCH-SAMPLE-ROBUSTNESS-001",
            "model": "rr_partial_switch_sample_robustness_diagnostic_only",
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
            "sample_limited": sample_limited,
            "baseline_alignment": baseline_alignment,
            "outputs": {
                "candidate_matrix": "candidate_matrix.csv",
                "event_sample_audit": "event_sample_audit.csv",
                "leave_one_event_report": "leave_one_event_report.csv",
                "leave_one_month_report": "leave_one_month_report.csv",
                "leave_one_quarter_report": "leave_one_quarter_report.csv",
                "leave_one_year_report": "leave_one_year_report.csv",
                "post_2026_exclusion_report": "post_2026_exclusion_report.csv",
                "train_test_report": "train_2022_2024_test_2025_2026_report.csv",
                "pre_2026_sanity_report": "pre_2026_sanity_report.csv",
                "largest_contribution_exclusion_report": "largest_contribution_exclusion_report.csv",
                "oos_gate_report": "oos_gate_report.csv",
                "mdd_non_degradation_report": "mdd_non_degradation_report.csv",
                "trade_concentration_report": "trade_concentration_report.csv",
                "forward_return_rule_audit": "forward_return_rule_audit.csv",
                "summary": "baseline_vs_rr_sample_robustness_summary_zh.md",
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
        pd.DataFrame([{"step": "run_rr_partial_switch_sample_robustness", "error": str(exc)}]).to_csv(
            output / "failed.csv",
            index=False,
            encoding="utf-8-sig",
        )
        log("failed", "failed", str(exc))
        raise


def _simulate_candidate_set(
    *,
    frame: pd.DataFrame,
    event_study: pd.DataFrame,
    labels: pd.DataFrame,
    prices: dict[str, pd.Series],
    initial_cash: float,
    excluded_dates: set[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    context = _context_with_exclusions(frame, event_study, labels, excluded_dates or set())
    variants = _candidate_variants()
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
        daily_frames.append(daily)
        trade_frames.append(trades)
    daily_ledger = pd.concat(daily_frames, ignore_index=True)
    trade_ledger = pd.concat(trade_frames, ignore_index=True)
    period_perf = _period_performance(daily_ledger, prices)
    stability = _period_stability_report(period_perf)
    return daily_ledger, trade_ledger, period_perf, stability


def _context_with_exclusions(
    frame: pd.DataFrame,
    event_study: pd.DataFrame,
    labels: pd.DataFrame,
    excluded_dates: set[str],
) -> dict[str, dict[str, Any]]:
    from backtest_lab.partial_execution_ledger import _build_event_context
    from backtest_lab.rapid_reversal_partial_switch_narrow import _merge_rapid_reversal_context

    adjusted = labels.copy()
    if excluded_dates:
        mask = adjusted["date"].astype(str).isin(excluded_dates)
        for column in ("rapid_reversal_any_1_3", "rapid_reversal_roundtrip_1_3"):
            if column in adjusted.columns:
                adjusted.loc[mask, column] = False
    context = _build_event_context(frame, event_study)
    return _merge_rapid_reversal_context(context, adjusted)


def _candidate_variants() -> list[ExecutionVariant]:
    return [
        ExecutionVariant("baseline_full_rotation", "baseline"),
        ExecutionVariant("partial_switch_25_global_diagnostic", "control_global_risk", partial_weight=0.25),
        ExecutionVariant("rr_partial_25_roundtrip_1_3", "main_candidate", partial_weight=0.25, subset="rapid_reversal_roundtrip_1_3"),
        ExecutionVariant("rr_partial_25_any_1_3", "sensitivity_candidate", partial_weight=0.25, subset="rapid_reversal_any_1_3"),
    ]


def _candidate_variant(candidate: str) -> ExecutionVariant:
    if candidate == MAIN_CANDIDATE:
        return ExecutionVariant(MAIN_CANDIDATE, "main_candidate", partial_weight=0.25, subset="rapid_reversal_roundtrip_1_3")
    if candidate == SENSITIVITY_CANDIDATE:
        return ExecutionVariant(SENSITIVITY_CANDIDATE, "sensitivity_candidate", partial_weight=0.25, subset="rapid_reversal_any_1_3")
    raise ValueError(f"unsupported sample robustness candidate: {candidate}")


def _simulate_candidate_only(
    *,
    frame: pd.DataFrame,
    event_study: pd.DataFrame,
    labels: pd.DataFrame,
    prices: dict[str, pd.Series],
    initial_cash: float,
    excluded_dates: set[str],
    candidate: str,
) -> pd.DataFrame:
    context = _context_with_exclusions(frame, event_study, labels, excluded_dates)
    daily, _ = _simulate_variant(
        frame=frame,
        prices=prices,
        event_context=context,
        variant=_candidate_variant(candidate),
        initial_cash=initial_cash,
    )
    return daily


def _candidate_matrix(sample_audit: pd.DataFrame) -> pd.DataFrame:
    counts = {row["event_label"]: row["event_count"] for row in sample_audit.to_dict(orient="records")}
    rows = [
        ("baseline_full_rotation", "baseline", "", ""),
        ("partial_switch_25_global_diagnostic", "control_global_risk", "global", "risk-control comparison only"),
        (MAIN_CANDIDATE, "main_candidate", "rapid_reversal_roundtrip_1_3", "sample-limited main candidate"),
        (SENSITIVITY_CANDIDATE, "sensitivity_candidate", "rapid_reversal_any_1_3", "upper-bound sensitivity"),
    ]
    return pd.DataFrame(
        [
            {
                "variant_id": variant,
                "role": role,
                "event_label": label,
                "event_count": counts.get(label, ""),
                "sample_limited": bool(label and counts.get(label, 0) < 25),
                "note": note,
                "execution_diagnostic_active_in_trade_decision": False,
            }
            for variant, role, label, note in rows
        ]
    )


def _event_sample_audit(labels: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in ("rapid_reversal_any_1_3", "rapid_reversal_roundtrip_1_3"):
        subset = labels[labels[column].astype(bool)] if column in labels.columns else pd.DataFrame()
        rows.append(
            {
                "event_label": column,
                "event_count": int(len(subset)),
                "sample_limited": bool(len(subset) < 25),
                "first_event": subset["date"].iloc[0] if not subset.empty else "",
                "last_event": subset["date"].iloc[-1] if not subset.empty else "",
                "unique_months": int(pd.to_datetime(subset.get("date", pd.Series(dtype=str)), errors="coerce").dt.to_period("M").nunique()) if not subset.empty else 0,
                "unique_years": int(pd.to_datetime(subset.get("date", pd.Series(dtype=str)), errors="coerce").dt.year.nunique()) if not subset.empty else 0,
                "execution_diagnostic_active_in_trade_decision": False,
            }
        )
    return pd.DataFrame(rows)


def _leave_one_event_report(
    frame: pd.DataFrame,
    event_study: pd.DataFrame,
    labels: pd.DataFrame,
    prices: dict[str, pd.Series],
    initial_cash: float,
    baseline_full: dict[str, float],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for candidate, label in ((MAIN_CANDIDATE, "rapid_reversal_roundtrip_1_3"), (SENSITIVITY_CANDIDATE, "rapid_reversal_any_1_3")):
        events = labels[labels[label].astype(bool)]["date"].astype(str).tolist()
        for date in events:
            daily = _simulate_candidate_only(
                frame=frame,
                event_study=event_study,
                labels=labels,
                prices=prices,
                initial_cash=initial_cash,
                excluded_dates={date},
                candidate=candidate,
            )
            rows.append(_extract_full_result_from_daily(daily, candidate, "event", date, len(events) - 1, baseline_full))
    return pd.DataFrame(rows)


def _leave_group_report(
    frame: pd.DataFrame,
    event_study: pd.DataFrame,
    labels: pd.DataFrame,
    prices: dict[str, pd.Series],
    initial_cash: float,
    baseline_full: dict[str, float],
    *,
    group: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    date_series = pd.to_datetime(labels["date"], errors="coerce")
    labels = labels.copy()
    labels["month"] = date_series.dt.to_period("M").astype(str)
    labels["quarter"] = date_series.dt.to_period("Q").astype(str)
    labels["year"] = date_series.dt.year.astype("Int64").astype(str)
    for candidate, label in ((MAIN_CANDIDATE, "rapid_reversal_roundtrip_1_3"), (SENSITIVITY_CANDIDATE, "rapid_reversal_any_1_3")):
        event_rows = labels[labels[label].astype(bool)].copy()
        for key, group_rows in event_rows.groupby(group, dropna=True):
            excluded = set(group_rows["date"].astype(str).tolist())
            daily = _simulate_candidate_only(
                frame=frame,
                event_study=event_study,
                labels=labels,
                prices=prices,
                initial_cash=initial_cash,
                excluded_dates=excluded,
                candidate=candidate,
            )
            rows.append(_extract_full_result_from_daily(daily, candidate, group, str(key), len(event_rows) - len(excluded), baseline_full))
    return pd.DataFrame(rows)


def _post_2026_exclusion_report(
    frame: pd.DataFrame,
    event_study: pd.DataFrame,
    labels: pd.DataFrame,
    prices: dict[str, pd.Series],
    initial_cash: float,
    baseline_full: dict[str, float],
) -> pd.DataFrame:
    excluded = set(labels[pd.to_datetime(labels["date"], errors="coerce").dt.year.ge(2026)]["date"].astype(str).tolist())
    main = _simulate_candidate_only(
        frame=frame,
        event_study=event_study,
        labels=labels,
        prices=prices,
        initial_cash=initial_cash,
        excluded_dates=excluded,
        candidate=MAIN_CANDIDATE,
    )
    sensitivity = _simulate_candidate_only(
        frame=frame,
        event_study=event_study,
        labels=labels,
        prices=prices,
        initial_cash=initial_cash,
        excluded_dates=excluded,
        candidate=SENSITIVITY_CANDIDATE,
    )
    return pd.DataFrame(
        [
            _extract_full_result_from_daily(main, MAIN_CANDIDATE, "post_2026", "exclude_2026_events", None, baseline_full),
            _extract_full_result_from_daily(sensitivity, SENSITIVITY_CANDIDATE, "post_2026", "exclude_2026_events", None, baseline_full),
        ]
    )


def _train_test_report(daily: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    windows = {
        "train_2022_2024": ("2022-01-01", "2024-12-31"),
        "test_2025_2026": ("2025-01-01", None),
    }
    for variant_id, group in daily.groupby("variant_id", dropna=False):
        frame = group.copy()
        frame["date_ts"] = pd.to_datetime(frame["date"], errors="coerce")
        for window, (start, end) in windows.items():
            subset = frame[frame["date_ts"] >= pd.Timestamp(start)]
            if end:
                subset = subset[subset["date_ts"] <= pd.Timestamp(end)]
            if subset.empty:
                continue
            rows.append(_performance_row(variant_id, window, subset))
    report = pd.DataFrame(rows)
    return _add_baseline_deltas(report)


def _pre_2026_sanity_report(daily: pd.DataFrame) -> pd.DataFrame:
    frame = daily.copy()
    frame["date_ts"] = pd.to_datetime(frame["date"], errors="coerce")
    rows = [_performance_row(variant, "pre_2026", group[group["date_ts"] < pd.Timestamp("2026-01-01")]) for variant, group in frame.groupby("variant_id", dropna=False)]
    return _add_baseline_deltas(pd.DataFrame([row for row in rows if row]))


def _largest_contribution_exclusion_report(leave_event: pd.DataFrame, leave_month: pd.DataFrame, leave_quarter: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, frame in (("event", leave_event), ("month", leave_month), ("quarter", leave_quarter)):
        if frame.empty:
            continue
        for candidate, group in frame.groupby("candidate_variant", dropna=False):
            sorted_group = group.sort_values("return_delta_vs_baseline_pp", ascending=True)
            row = sorted_group.iloc[0].to_dict()
            rows.append(
                {
                    "candidate_variant": candidate,
                    "exclusion_level": name,
                    "largest_contribution_key": row.get("excluded_key", ""),
                    "return_delta_after_exclusion": row.get("return_delta_vs_baseline_pp", ""),
                    "mdd_delta_after_exclusion": row.get("mdd_delta_vs_baseline_pp", ""),
                    "execution_diagnostic_active_in_trade_decision": False,
                }
            )
    return pd.DataFrame(rows)


def _oos_gate_report(
    stability: pd.DataFrame,
    leave_event: pd.DataFrame,
    leave_month: pd.DataFrame,
    leave_quarter: pd.DataFrame,
    leave_year: pd.DataFrame,
    post_2026: pd.DataFrame,
    train_test: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for candidate in (MAIN_CANDIDATE, SENSITIVITY_CANDIDATE):
        full = _delta_for(stability, candidate, "full_2022_2026")
        post = post_2026[post_2026["candidate_variant"].eq(candidate)]
        test = train_test[(train_test["variant_id"].eq(candidate)) & (train_test["period"].eq("test_2025_2026"))]
        leave_event_min = _min_delta(leave_event, candidate)
        leave_month_min = _min_delta(leave_month, candidate)
        leave_quarter_min = _min_delta(leave_quarter, candidate)
        leave_year_min = _min_delta(leave_year, candidate)
        rows.append(
            {
                "candidate_variant": candidate,
                "full_return_delta_pp": full.get("return_delta_vs_baseline_pp", ""),
                "full_mdd_delta_pp": full.get("mdd_delta_vs_baseline_pp", ""),
                "leave_one_event_min_return_delta_pp": leave_event_min,
                "leave_one_month_min_return_delta_pp": leave_month_min,
                "leave_one_quarter_min_return_delta_pp": leave_quarter_min,
                "leave_one_year_min_return_delta_pp": leave_year_min,
                "post_2026_exclusion_return_delta_pp": post["return_delta_vs_baseline_pp"].iloc[0] if not post.empty else "",
                "test_2025_2026_return_delta_pp": test["return_delta_vs_baseline_pp"].iloc[0] if not test.empty else "",
                "gate_status": "diagnostic_pending_experiments",
                "execution_diagnostic_active_in_trade_decision": False,
            }
        )
    return pd.DataFrame(rows)


def _mdd_non_degradation_report(stability: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for candidate in (MAIN_CANDIDATE, SENSITIVITY_CANDIDATE):
        for period in ("full_2022_2026", "2024_now", "2024_hard_gate"):
            row = _delta_for(stability, candidate, period)
            limit = -1.0 if period == "full_2022_2026" else -2.0
            value = row.get("mdd_delta_vs_baseline_pp", "")
            rows.append(
                {
                    "candidate_variant": candidate,
                    "period": period,
                    "mdd_delta_vs_baseline_pp": value,
                    "allowed_floor_pp": limit,
                    "mdd_non_degradation_pass": bool(value != "" and float(value) >= limit),
                    "execution_diagnostic_active_in_trade_decision": False,
                }
            )
    return pd.DataFrame(rows)


def _forward_return_rule_audit(forward_eval: pd.DataFrame) -> pd.DataFrame:
    used = int(forward_eval.get("forward_return_used_as_rule", pd.Series(dtype=bool)).astype(bool).sum()) if not forward_eval.empty else 0
    return pd.DataFrame(
        [
            {
                "audit_id": "forward_return_rule_usage",
                "used_as_rule_count": used,
                "pass": used == 0,
                "execution_diagnostic_active_in_trade_decision": False,
            }
        ]
    )


def _blocked_or_demoted_candidates(sample_limited: bool, oos_gate: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "candidate_variant": MAIN_CANDIDATE,
                "status": "sample_limited_diagnostic" if sample_limited else "diagnostic_needs_experiments",
                "reason": "event_count below 25; cannot be formal-ready before robustness validation",
                "execution_diagnostic_active_in_trade_decision": False,
            },
            {
                "candidate_variant": SENSITIVITY_CANDIDATE,
                "status": "sample_limited_sensitivity" if sample_limited else "diagnostic_needs_experiments",
                "reason": "upper-bound sensitivity candidate; not main formal candidate",
                "execution_diagnostic_active_in_trade_decision": False,
            },
        ]
    )


def _extract_full_result(stability: pd.DataFrame, candidate: str, level: str, key: str, remaining_events: int | None) -> dict[str, Any]:
    row = _delta_for(stability, candidate, "full_2022_2026")
    return {
        "candidate_variant": candidate,
        "exclusion_level": level,
        "excluded_key": key,
        "remaining_events": remaining_events if remaining_events is not None else "",
        "return_delta_vs_baseline_pp": row.get("return_delta_vs_baseline_pp", ""),
        "mdd_delta_vs_baseline_pp": row.get("mdd_delta_vs_baseline_pp", ""),
        "cost_delta_vs_baseline": row.get("cost_delta_vs_baseline", ""),
        "execution_diagnostic_active_in_trade_decision": False,
    }


def _extract_full_result_from_daily(
    daily: pd.DataFrame,
    candidate: str,
    level: str,
    key: str,
    remaining_events: int | None,
    baseline_full: dict[str, float],
) -> dict[str, Any]:
    metrics = _full_metrics(daily)
    return {
        "candidate_variant": candidate,
        "exclusion_level": level,
        "excluded_key": key,
        "remaining_events": remaining_events if remaining_events is not None else "",
        "return_delta_vs_baseline_pp": round(metrics["return_pct"] - baseline_full["return_pct"], 6),
        "mdd_delta_vs_baseline_pp": round(metrics["mdd_pct"] - baseline_full["mdd_pct"], 6),
        "cost_delta_vs_baseline": round(metrics["transaction_cost"] - baseline_full["transaction_cost"], 2),
        "execution_diagnostic_active_in_trade_decision": False,
    }


def _baseline_full_metrics(daily: pd.DataFrame) -> dict[str, float]:
    subset = daily[daily["variant_id"].eq("baseline_full_rotation")].copy()
    if subset.empty:
        return {"return_pct": 0.0, "mdd_pct": 0.0, "transaction_cost": 0.0}
    return _full_metrics(subset)


def _full_metrics(daily: pd.DataFrame) -> dict[str, float]:
    if daily.empty:
        return {"return_pct": 0.0, "mdd_pct": 0.0, "transaction_cost": 0.0}
    start = float(daily["portfolio_equity_after"].iloc[0])
    final = float(daily["portfolio_equity_after"].iloc[-1])
    running_max = daily["portfolio_equity_after"].cummax()
    mdd = float((daily["portfolio_equity_after"] / running_max - 1).min()) * 100
    cost = float(pd.to_numeric(daily["transaction_cost"], errors="coerce").sum())
    return {
        "return_pct": (final / start - 1) * 100 if start else 0.0,
        "mdd_pct": mdd,
        "transaction_cost": cost,
    }


def _delta_for(stability: pd.DataFrame, candidate: str, period: str) -> dict[str, Any]:
    subset = stability[(stability["variant_id"].eq(candidate)) & (stability["period"].eq(period))]
    if subset.empty:
        return {}
    return subset.iloc[0].to_dict()


def _min_delta(frame: pd.DataFrame, candidate: str) -> float | str:
    subset = frame[frame["candidate_variant"].eq(candidate)] if not frame.empty else pd.DataFrame()
    if subset.empty:
        return ""
    return round(float(pd.to_numeric(subset["return_delta_vs_baseline_pp"], errors="coerce").min()), 6)


def _performance_row(variant_id: str, period: str, subset: pd.DataFrame) -> dict[str, Any] | None:
    if subset.empty:
        return None
    start = float(subset["portfolio_equity_after"].iloc[0])
    final = float(subset["portfolio_equity_after"].iloc[-1])
    running_max = subset["portfolio_equity_after"].cummax()
    mdd = float((subset["portfolio_equity_after"] / running_max - 1).min())
    return {
        "variant_id": variant_id,
        "period": period,
        "start_date": subset["date"].iloc[0],
        "end_date": subset["date"].iloc[-1],
        "total_return_pct": round((final / start - 1) * 100, 4),
        "max_drawdown_pct": round(mdd * 100, 4),
        "transaction_cost": round(float(pd.to_numeric(subset["transaction_cost"], errors="coerce").sum()), 2),
        "execution_diagnostic_active_in_trade_decision": False,
    }


def _add_baseline_deltas(report: pd.DataFrame) -> pd.DataFrame:
    if report.empty:
        return report
    baseline = report[report["variant_id"].eq("baseline_full_rotation")][["period", "total_return_pct", "max_drawdown_pct"]].rename(
        columns={"total_return_pct": "baseline_return_pct", "max_drawdown_pct": "baseline_mdd_pct"}
    )
    merged = report.merge(baseline, on="period", how="left")
    merged["return_delta_vs_baseline_pp"] = pd.to_numeric(merged["total_return_pct"], errors="coerce") - pd.to_numeric(
        merged["baseline_return_pct"], errors="coerce"
    )
    merged["mdd_delta_vs_baseline_pp"] = pd.to_numeric(merged["max_drawdown_pct"], errors="coerce") - pd.to_numeric(
        merged["baseline_mdd_pct"], errors="coerce"
    )
    return merged


def _summary_markdown(
    baseline_alignment: dict[str, Any],
    sample_audit: pd.DataFrame,
    oos_gate: pd.DataFrame,
    demoted: pd.DataFrame,
) -> str:
    lines = [
        "# RR Partial Switch Sample Robustness Diagnostic",
        "",
        "本輸出只檢查 RR partial switch 是否有小樣本、2026、單月或單事件集中風險，不是正式 execution / exit layer。",
        "",
        "## 邊界",
        "",
        "- formal_model_changed=false",
        "- trade_decision_changed=false",
        "- active_in_trade_decision=false",
        "- uses_forward_return_as_rule=false",
        "- sample_limited=true 時不得標 formal-ready",
        "",
        "## Baseline 對齊",
        "",
        f"- final equity diff：{baseline_alignment.get('final_equity_diff')}",
        f"- MDD diff：{baseline_alignment.get('mdd_diff')}",
        "",
        "## Event sample",
        "",
    ]
    for row in sample_audit.to_dict(orient="records"):
        lines.append(f"- {row.get('event_label')}：{row.get('event_count')} events，sample_limited={row.get('sample_limited')}")
    lines.extend(["", "## OOS Gate 初步輸出", ""])
    for row in oos_gate.to_dict(orient="records"):
        lines.append(
            f"- {row.get('candidate_variant')}：leave-one-event min {row.get('leave_one_event_min_return_delta_pp')}pp，post-2026 {row.get('post_2026_exclusion_return_delta_pp')}pp"
        )
    lines.extend(["", "## 狀態", ""])
    for row in demoted.to_dict(orient="records"):
        lines.append(f"- {row.get('candidate_variant')}：{row.get('status')}，{row.get('reason')}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build RR partial switch sample robustness diagnostic outputs.")
    parser.add_argument("--formal-daily", required=True)
    parser.add_argument("--price-cache-dir", default=DEFAULT_PRICE_CACHE_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--initial-cash", type=float, default=DEFAULT_INITIAL_CASH)
    args = parser.parse_args()
    output = run_rr_partial_switch_sample_robustness(
        formal_daily_path=args.formal_daily,
        price_cache_dir=args.price_cache_dir,
        output_dir=args.output_dir,
        initial_cash=args.initial_cash,
    )
    print(f"OUTPUT_DIR={output.resolve()}")


if __name__ == "__main__":
    main()
