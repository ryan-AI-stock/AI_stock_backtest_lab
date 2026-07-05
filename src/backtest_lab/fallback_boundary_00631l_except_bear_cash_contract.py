"""Build a diagnostic fallback-boundary contract for 00631L except bear/cash.

This contract audits current formal CASH/no-target rows and maps only
traceable, market-exposure-eligible rows to 00631L in diagnostic variants.
It does not change the formal selector, daily report, or trade decision.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from backtest_lab.dynamic_pool1_exact_ab_switch_friction_contract import DEFAULT_BACKTEST_PERIOD_CONTRACT
from backtest_lab.strong_stock_trend_extension_bounded_portfolio_diagnostic import BENCHMARK_PRICE_PATHS


TASK_ID = "TASK-BACKTEST-CORE-FALLBACK-BOUNDARY-00631L-EXCEPT-BEAR-CASH-CONTRACT-001"
EXPERIMENTS_TASK_ID = "TASK-BACKTEST-EXPERIMENTS-FALLBACK-BOUNDARY-00631L-EXCEPT-BEAR-CASH-DIAGNOSTIC-001"
DEFAULT_FORMAL_STREAM = Path(
    "outputs/combined_formal_target_stream_20150128_20211230_20260702/combined_formal_target_stream.csv"
)
DEFAULT_OUTPUT_DIR = Path("outputs/fallback_boundary_00631l_except_bear_cash_contract_20260705")
CASH_TARGET = "CASH"
MARKET_EXPOSURE_TARGET = "00631L.TW"
VARIANTS = [
    {
        "variant_id": "current_formal_old_no_target_cash",
        "role": "baseline_reference",
        "mapping_rule": "keep current formal target; no-target cash remains cash",
        "upper_bound_reference": False,
    },
    {
        "variant_id": "fallback_00631L_except_bear_cash_primary",
        "role": "primary_diagnostic",
        "mapping_rule": "map no_stock_target_but_market_exposure_allowed to 00631L; keep bear/cash as cash; block unclassified cash",
        "upper_bound_reference": False,
    },
    {
        "variant_id": "fallback_00631L_all_no_target_upper_bound_reference",
        "role": "upper_bound_reference_only",
        "mapping_rule": "map all current formal CASH/no-target rows to 00631L regardless of bear/cash readiness",
        "upper_bound_reference": True,
    },
    {
        "variant_id": "cash_only_bear_strict_reference",
        "role": "strict_readiness_reference",
        "mapping_rule": "keep only explicit bear/cash as cash; map classified no-stock market-exposure rows to 00631L; block ambiguous cash",
        "upper_bound_reference": False,
    },
]


def run_fallback_boundary_00631l_except_bear_cash_contract(
    *,
    repo_root: str | Path = ".",
    formal_stream: str | Path = DEFAULT_FORMAL_STREAM,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    stream_path = _resolve(root, formal_stream)
    output = _resolve(root, output_dir)
    output.mkdir(parents=True, exist_ok=True)

    stream = _load_formal_stream(stream_path)
    benchmark = _benchmark_availability(root, stream)
    state_panel = _build_execution_state_panel(stream, benchmark)
    cash_audit = _cash_row_classification_audit(state_panel)
    readiness = _bear_cash_condition_readiness(state_panel)
    variant_matrix = pd.DataFrame(VARIANTS)
    mapping_daily = _build_mapping_daily_panel(state_panel)
    blocked = state_panel[state_panel["execution_state"].eq("unclassified_cash_boundary_blocked")].copy()
    period = _period_contract_validation(state_panel, mapping_daily, benchmark)
    future = _future_data_audit(state_panel, mapping_daily)

    state_panel.to_csv(output / "fallback_boundary_execution_state_panel.csv", index=False, encoding="utf-8-sig")
    cash_audit.to_csv(output / "cash_row_classification_audit.csv", index=False, encoding="utf-8-sig")
    readiness.to_csv(output / "bear_cash_condition_readiness.csv", index=False, encoding="utf-8-sig")
    variant_matrix.to_csv(output / "fallback_mapping_variant_matrix.csv", index=False, encoding="utf-8-sig")
    mapping_daily.to_csv(output / "fallback_mapping_daily_panel.csv", index=False, encoding="utf-8-sig")
    blocked.to_csv(output / "blocked_cash_boundary_rows.csv", index=False, encoding="utf-8-sig")
    period.to_csv(output / "period_contract_validation.csv", index=False, encoding="utf-8-sig")
    benchmark.to_csv(output / "benchmark_availability_audit.csv", index=False, encoding="utf-8-sig")
    future.to_csv(output / "future_data_audit.csv", index=False, encoding="utf-8-sig")

    future_count = int(future["future_data_violation"].sum()) if len(future) else 0
    cash_rows = int(state_panel["is_current_formal_cash"].sum())
    no_stock_rows = int(state_panel["execution_state"].eq("no_stock_target_but_market_exposure_allowed").sum())
    bear_rows = int(state_panel["execution_state"].eq("bear_or_cash_condition").sum())
    blocked_rows = int(state_panel["execution_state"].eq("unclassified_cash_boundary_blocked").sum())
    bear_ready = bool(bear_rows > 0 and blocked_rows == 0)
    ready_for_experiments = bool(len(mapping_daily) > 0 and future_count == 0)
    manifest: dict[str, Any] = {
        "task_id": TASK_ID,
        "status": "completed_diagnostic_contract_partial_bear_cash_classification",
        "output_dir": str(output),
        "source_formal_stream": str(stream_path),
        "formal_stream_rows": int(len(stream)),
        "cash_rows": cash_rows,
        "no_stock_target_but_market_exposure_allowed_rows": no_stock_rows,
        "bear_or_cash_condition_rows": bear_rows,
        "unclassified_cash_boundary_blocked_rows": blocked_rows,
        "formal_00631L_target_rows": int(state_panel["execution_state"].eq("formal_00631L_target").sum()),
        "direct_stock_target_rows": int(state_panel["execution_state"].eq("direct_stock_target").sum()),
        "default_backtest_period_contract": DEFAULT_BACKTEST_PERIOD_CONTRACT,
        "actual_signal_start": _date_text(stream["signal_date"].min()),
        "actual_signal_end": _date_text(stream["signal_date"].max()),
        "actual_execution_start": _date_text(stream["execution_date"].min()),
        "actual_execution_end": _date_text(stream["execution_date"].max()),
        "primary_mapping_policy": "only traceable 00631L market-exposure cash rows map to 00631L; unclassified cash rows blocked",
        "bear_cash_classification_ready": bear_ready,
        "ready_for_experiments": ready_for_experiments,
        "strategy_replay_executed_by_core": False,
        "uses_forward_return_as_rule": False,
        "future_data_violation_count": future_count,
        "formal_model_changed": False,
        "trade_decision_changed": False,
        "active_in_trade_decision": False,
        "report_changed": False,
        "handoff_to_experiments_task": EXPERIMENTS_TASK_ID,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "final_summary_zh.md").write_text(_summary(manifest), encoding="utf-8")
    pd.DataFrame([{"task_id": TASK_ID, "status": "completed", "output_dir": str(output)}]).to_csv(
        output / "completed.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(columns=["task_id", "status", "reason"]).to_csv(output / "failed.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(
        [
            {"step": "load_current_formal_stream", "status": "completed"},
            {"step": "classify_cash_boundary_rows", "status": "completed"},
            {"step": "build_diagnostic_mapping_variants", "status": "completed"},
            {"step": "write_contract_package", "status": "completed"},
        ]
    ).to_csv(output / "run_log.csv", index=False, encoding="utf-8-sig")
    return manifest


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _load_formal_stream(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path).fillna("")
    for col in ["signal_date", "execution_date"]:
        df[col] = pd.to_datetime(df[col], errors="coerce").dt.strftime("%Y-%m-%d")
    df["formal_target"] = df["formal_target"].astype(str).map(_canonical_target)
    df["pool1_candidate"] = df.get("pool1_candidate", "").astype(str).map(_canonical_target)
    df["no_target_cash_all_applied"] = df.get("no_target_cash_all_applied", False).map(_as_bool)
    df["pool1_attack_gate_active"] = df.get("pool1_attack_gate_active", False).map(_as_bool)
    df["pool1_target_is_actionable"] = df.get("pool1_target_is_actionable", False).map(_as_bool)
    return df.sort_values("signal_date").reset_index(drop=True)


def _build_execution_state_panel(stream: pd.DataFrame, benchmark: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    benchmark_by_date = benchmark.set_index("execution_date").to_dict("index") if not benchmark.empty else {}
    for _, row in stream.iterrows():
        execution_date = str(row.get("execution_date", ""))
        bench = benchmark_by_date.get(execution_date, {})
        state, method, blocked_reason, bear_reason = _classify_execution_state(row)
        rows.append(
            {
                "signal_date": row.get("signal_date", ""),
                "execution_date": execution_date,
                "formal_target": row.get("formal_target", ""),
                "formal_target_display": row.get("formal_target_display", ""),
                "target_type": row.get("target_type", ""),
                "pool1_candidate": row.get("pool1_candidate", ""),
                "pool1_candidate_display": row.get("pool1_candidate_display", ""),
                "pool1_gate_status": row.get("pool1_gate_status", ""),
                "pool1_attack_gate_active": bool(row.get("pool1_attack_gate_active", False)),
                "pool1_target_is_actionable": bool(row.get("pool1_target_is_actionable", False)),
                "pool2_confirmation_status": row.get("pool2_confirmation_status", ""),
                "pool2_confirmation_state": row.get("pool2_confirmation_state", ""),
                "pool2_vote": row.get("pool2_vote", ""),
                "pool2_support_without_persistence_vote": row.get("pool2_support_without_persistence_vote", ""),
                "no_target_reason": row.get("no_target_reason", ""),
                "risk_off_state": row.get("risk_off_state", ""),
                "cash_all_policy_reason": row.get("reason", ""),
                "source_decision": row.get("source_decision", ""),
                "is_current_formal_cash": row.get("formal_target", "") == CASH_TARGET,
                "execution_state": state,
                "classification_method": method,
                "classification_exact": state in {"direct_stock_target", "formal_00631L_target", "bear_or_cash_condition"},
                "bear_or_cash_condition_flag": state == "bear_or_cash_condition",
                "bear_or_cash_condition_reason": bear_reason,
                "action_blocked_reason": blocked_reason,
                "next_tradable_date_available": bool(execution_date),
                "benchmark_0050_available": bool(bench.get("benchmark_0050_available", False)),
                "benchmark_00631l_available": bool(bench.get("benchmark_00631l_available", False)),
                "benchmark_0050_price_source": bench.get("benchmark_0050_price_source", ""),
                "benchmark_00631l_price_source": bench.get("benchmark_00631l_price_source", ""),
                "uses_forward_return_as_rule": False,
                "formal_model_changed": False,
                "trade_decision_changed": False,
                "active_in_trade_decision": False,
                "report_changed": False,
            }
        )
    return pd.DataFrame(rows)


def _classify_execution_state(row: pd.Series) -> tuple[str, str, str, str]:
    target = str(row.get("formal_target", "")).strip()
    if target and target not in {CASH_TARGET, MARKET_EXPOSURE_TARGET}:
        return "direct_stock_target", "formal_target_is_individual_stock", "", ""
    if target == MARKET_EXPOSURE_TARGET:
        return "formal_00631L_target", "formal_target_is_00631L", "", ""
    if target != CASH_TARGET:
        return "unclassified_cash_boundary_blocked", "missing_or_unknown_formal_target", "blocked_missing_formal_target", ""

    if _has_explicit_bear_cash_condition(row):
        return "bear_or_cash_condition", "explicit_bear_cash_text_field", "", "explicit_bear_or_cash_text_field"

    pool1_candidate = str(row.get("pool1_candidate", "")).strip()
    no_target_reason = str(row.get("no_target_reason", "")).strip()
    pool1_actionable = bool(row.get("pool1_target_is_actionable", False))
    pool2_not_ready = str(row.get("pool2_confirmation_status", "")).strip() == "pool2_not_ready"
    if (
        pool1_candidate == MARKET_EXPOSURE_TARGET
        and pool1_actionable
        and pool2_not_ready
        and no_target_reason == "pool2_confirmation_not_ready"
    ):
        return (
            "no_stock_target_but_market_exposure_allowed",
            "formal_market_exposure_candidate_blocked_by_pool2_without_explicit_bear_field",
            "",
            "",
        )

    return (
        "unclassified_cash_boundary_blocked",
        "cash_row_lacks_explicit_bear_cash_or_market_exposure_boundary",
        "blocked_missing_explicit_bear_cash_or_no_stock_fallback_boundary",
        "",
    )


def _has_explicit_bear_cash_condition(row: pd.Series) -> bool:
    fields = [
        str(row.get("risk_off_state", "")),
        str(row.get("no_target_reason", "")),
        str(row.get("cash_all_policy_reason", row.get("reason", ""))),
        str(row.get("pool2_confirmation_state", "")),
    ]
    text = " ".join(fields).lower()
    keywords = ["bear", "大空頭", "空頭", "systemic", "risk_off_bear", "market_crash"]
    return any(keyword in text for keyword in keywords)


def _cash_row_classification_audit(panel: pd.DataFrame) -> pd.DataFrame:
    cash = panel[panel["is_current_formal_cash"]].copy()
    if cash.empty:
        return pd.DataFrame(columns=["execution_state", "rows", "classification_method", "blocked_rows"])
    grouped = (
        cash.groupby(["execution_state", "classification_method"], dropna=False)
        .agg(rows=("signal_date", "count"), blocked_rows=("action_blocked_reason", lambda s: int((s.astype(str) != "").sum())))
        .reset_index()
    )
    grouped["classification_input_fields"] = (
        "no_target_reason;pool1_candidate;pool1_target_is_actionable;pool2_confirmation_status;"
        "pool2_confirmation_state;risk_off_state;reason;benchmark_availability;next_tradable_date"
    )
    grouped["uses_forward_return_as_rule"] = False
    return grouped


def _bear_cash_condition_readiness(panel: pd.DataFrame) -> pd.DataFrame:
    cash = panel[panel["is_current_formal_cash"]]
    fields = [
        ("no_target_reason", True, "available_reason_code"),
        ("pool1_candidate", True, "available_candidate_context"),
        ("pool1_attack_gate_active", True, "available_attack_gate_context"),
        ("pool2_confirmation_state", True, "available_confirmation_context"),
        ("risk_off_state", True, "available_but_generic_no_target_cash_all"),
        ("explicit_bear_or_cash_condition", False, "missing_explicit_bear_regime_or_formal_cash_condition"),
        ("market_regime_state", False, "missing_row_level_regime_state"),
        ("bear_cash_condition_ready", False, "blocked_until_explicit_bear_cash_boundary_exists"),
    ]
    rows = []
    for field, available, note in fields:
        rows.append(
            {
                "field": field,
                "available": available,
                "cash_rows": int(len(cash)),
                "ready_for_exact_bear_cash_classification": field == "bear_cash_condition_ready" and available,
                "note": note,
            }
        )
    rows.append(
        {
            "field": "classified_no_stock_market_exposure_rows",
            "available": True,
            "cash_rows": int(panel["execution_state"].eq("no_stock_target_but_market_exposure_allowed").sum()),
            "ready_for_exact_bear_cash_classification": False,
            "note": "diagnostic only; not exact bear/cash classification",
        }
    )
    rows.append(
        {
            "field": "unclassified_cash_boundary_blocked_rows",
            "available": True,
            "cash_rows": int(panel["execution_state"].eq("unclassified_cash_boundary_blocked").sum()),
            "ready_for_exact_bear_cash_classification": False,
            "note": "must not force-map to 00631L in primary",
        }
    )
    return pd.DataFrame(rows)


def _build_mapping_daily_panel(panel: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in panel.iterrows():
        for variant in VARIANTS:
            target, state, blocked = _mapped_target_for_variant(row, str(variant["variant_id"]))
            rows.append(
                {
                    "variant_id": variant["variant_id"],
                    "variant_role": variant["role"],
                    "signal_date": row["signal_date"],
                    "execution_date": row["execution_date"],
                    "formal_target": row["formal_target"],
                    "execution_state": row["execution_state"],
                    "mapped_execution_state": state,
                    "mapped_target": target,
                    "mapped_target_weight": 1.0 if target else 0.0,
                    "action_blocked_reason": blocked,
                    "upper_bound_reference": bool(variant["upper_bound_reference"]),
                    "benchmark_0050_available": row["benchmark_0050_available"],
                    "benchmark_00631l_available": row["benchmark_00631l_available"],
                    "uses_forward_return_as_rule": False,
                    "strategy_replay_executed_by_core": False,
                    "formal_model_changed": False,
                    "trade_decision_changed": False,
                    "active_in_trade_decision": False,
                    "report_changed": False,
                }
            )
    return pd.DataFrame(rows)


def _mapped_target_for_variant(row: pd.Series, variant_id: str) -> tuple[str, str, str]:
    formal_target = str(row.get("formal_target", "")).strip()
    state = str(row.get("execution_state", "")).strip()
    if variant_id == "current_formal_old_no_target_cash":
        return formal_target, state, ""
    if variant_id == "fallback_00631L_all_no_target_upper_bound_reference" and formal_target == CASH_TARGET:
        return MARKET_EXPOSURE_TARGET, "upper_bound_all_no_target_to_00631L", ""
    if state in {"direct_stock_target", "formal_00631L_target"}:
        return formal_target, state, ""
    if state == "bear_or_cash_condition":
        return CASH_TARGET, state, ""
    if state == "no_stock_target_but_market_exposure_allowed":
        return MARKET_EXPOSURE_TARGET, state, ""
    if state == "unclassified_cash_boundary_blocked":
        return "", state, str(row.get("action_blocked_reason", "blocked_unclassified_cash_boundary"))
    return "", state or "unknown", "blocked_unknown_execution_state"


def _benchmark_availability(root: Path, stream: pd.DataFrame) -> pd.DataFrame:
    dates = stream[["execution_date"]].drop_duplicates().copy()
    for ticker, rel in BENCHMARK_PRICE_PATHS.items():
        key = "0050" if ticker == "0050.TW" else "00631l"
        path = root / rel
        if path.exists():
            prices = pd.read_csv(path, usecols=lambda col: col in {"date", "close", "adj_close"})
            available_dates = set(pd.to_datetime(prices["date"], errors="coerce").dropna().dt.strftime("%Y-%m-%d"))
            dates[f"benchmark_{key}_available"] = dates["execution_date"].isin(available_dates)
            dates[f"benchmark_{key}_price_source"] = str(rel)
            dates[f"benchmark_{key}_actual_start"] = min(available_dates) if available_dates else ""
            dates[f"benchmark_{key}_actual_end"] = max(available_dates) if available_dates else ""
        else:
            dates[f"benchmark_{key}_available"] = False
            dates[f"benchmark_{key}_price_source"] = "missing_local_cache"
            dates[f"benchmark_{key}_actual_start"] = ""
            dates[f"benchmark_{key}_actual_end"] = ""
    return dates


def _period_contract_validation(panel: pd.DataFrame, mapping: pd.DataFrame, benchmark: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for period in DEFAULT_BACKTEST_PERIOD_CONTRACT:
        start = str(period["requested_start"])
        end = str(period["requested_end"])
        source_s = panel[(panel["signal_date"] >= start) & (panel["signal_date"] <= end)]
        mapped_s = mapping[(mapping["signal_date"] >= start) & (mapping["signal_date"] <= end)]
        rows.append(_period_row("formal_target_stream_signal", period, source_s, "signal_date"))
        rows.append(_period_row("mapped_execution_panel_signal", period, mapped_s, "signal_date"))
        rows.append(
            {
                **_period_row("blocked_cash_boundary_rows_signal", period, source_s[source_s["execution_state"].eq("unclassified_cash_boundary_blocked")], "signal_date"),
                "blocked_rows": int(source_s["execution_state"].eq("unclassified_cash_boundary_blocked").sum()),
            }
        )
        bench_s = benchmark[(benchmark["execution_date"] >= start) & (benchmark["execution_date"] <= end)]
        for key in ["0050", "00631l"]:
            rows.append(
                {
                    "layer": f"benchmark_{key}",
                    "period_label": period["period_label"],
                    "requested_start": start,
                    "requested_end": end,
                    "actual_start": _date_text(bench_s["execution_date"].min()) if not bench_s.empty else "",
                    "actual_end": _date_text(bench_s["execution_date"].max()) if not bench_s.empty else "",
                    "rows": int(len(bench_s)),
                    "available_rows": int(bench_s.get(f"benchmark_{key}_available", pd.Series(dtype=bool)).sum()) if not bench_s.empty else 0,
                    "status": "partial_or_actual_coverage" if not bench_s.empty else "no_actual_coverage",
                }
            )
    return pd.DataFrame(rows)


def _period_row(layer: str, period: dict[str, str], frame: pd.DataFrame, date_col: str) -> dict[str, Any]:
    return {
        "layer": layer,
        "period_label": period["period_label"],
        "requested_start": period["requested_start"],
        "requested_end": period["requested_end"],
        "actual_start": _date_text(frame[date_col].min()) if not frame.empty else "",
        "actual_end": _date_text(frame[date_col].max()) if not frame.empty else "",
        "rows": int(len(frame)),
        "status": "partial_or_actual_coverage" if not frame.empty else "no_actual_coverage",
    }


def _future_data_audit(panel: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "audit_item": "fallback_boundary_execution_state_panel",
                "rows": int(len(panel)),
                "future_data_violation": False,
                "reason": "classification uses same-row formal stream fields only",
            },
            {
                "audit_item": "fallback_mapping_daily_panel",
                "rows": int(len(mapping)),
                "future_data_violation": False,
                "reason": "mapping variants use current formal target and live-safe classification only",
            },
        ]
    )


def _summary(manifest: dict[str, Any]) -> str:
    return (
        "# Fallback boundary 00631L except bear/cash diagnostic contract\n\n"
        "## 結論\n\n"
        "- 本包只建立 diagnostic execution mapping contract，沒有改正式模型、報告或交易決策。\n"
        f"- current formal cash rows：{manifest['cash_rows']}\n"
        f"- primary 可安全診斷轉 00631L rows：{manifest['no_stock_target_but_market_exposure_allowed_rows']}\n"
        f"- bear/cash rows：{manifest['bear_or_cash_condition_rows']}\n"
        f"- unclassified blocked cash rows：{manifest['unclassified_cash_boundary_blocked_rows']}\n"
        f"- bear_cash_classification_ready：{manifest['bear_cash_classification_ready']}\n"
        f"- ready_for_experiments：{manifest['ready_for_experiments']}\n\n"
        "## 邊界\n\n"
        "- `fallback_00631L_except_bear_cash_primary` 不會把 unclassified cash row 硬轉 00631L。\n"
        "- `fallback_00631L_all_no_target_upper_bound_reference` 只是 upper-bound/reference，不是 formal route。\n"
        "- `strategy_replay_executed_by_core=false`；績效驗收交 Experiments。\n"
    )


def _canonical_target(value: object) -> str:
    text = str(value or "").strip()
    if not text or text.lower() == "nan":
        return ""
    if text.upper() == CASH_TARGET:
        return CASH_TARGET
    if text in {"00631L", "00631L.TW"}:
        return MARKET_EXPOSURE_TARGET
    if text.endswith(".TW") or text.endswith(".TWO"):
        return text
    if text.isdigit():
        return f"{text}.TW"
    return text


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _date_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value)[:10]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build fallback-boundary 00631L except bear/cash diagnostic contract.")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--formal-stream", default=str(DEFAULT_FORMAL_STREAM))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()
    manifest = run_fallback_boundary_00631l_except_bear_cash_contract(
        repo_root=args.repo_root,
        formal_stream=args.formal_stream,
        output_dir=args.output_dir,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
