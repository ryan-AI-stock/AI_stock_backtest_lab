from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


HORIZONS = (20, 60, 120)
TARGET_BLOCKERS = {"exact_consensus_missing", "formal_target_selector_preferred_other_pool"}


def run_pool3_selector_opportunity_diagnostic(
    *,
    event_panel_path: str | Path,
    output_dir: str | Path,
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

    log("load_event_panel", "started")
    panel = pd.read_csv(event_panel_path).fillna("")
    _validate_event_panel(panel)

    log("build_outputs", "started")
    events = _selector_event_panel(panel)
    events.to_csv(output / "pool3_selector_opportunity_event_panel.csv", index=False, encoding="utf-8-sig")

    exact_missing = events[events["pool3_blocker_category"] == "exact_consensus_missing"].copy()
    exact_missing.to_csv(output / "exact_consensus_missing_events.csv", index=False, encoding="utf-8-sig")

    selector_preferred = events[events["pool3_blocker_category"] == "formal_target_selector_preferred_other_pool"].copy()
    selector_preferred.to_csv(output / "formal_selector_preferred_other_pool_events.csv", index=False, encoding="utf-8-sig")

    comparisons = _forward_comparison_rows(events)
    comparisons.to_csv(output / "pool3_vs_final_target_forward_returns.csv", index=False, encoding="utf-8-sig")

    opportunities = events[events["pool3_opportunity_state"] == "opportunity_warning"].copy()
    opportunities.to_csv(output / "pool3_opportunity_warning_candidates.csv", index=False, encoding="utf-8-sig")

    vetoes = events[events["pool3_opportunity_state"] == "veto_warning"].copy()
    vetoes.to_csv(output / "pool3_veto_warning_candidates.csv", index=False, encoding="utf-8-sig")

    summary = _summary_by_period(events)
    summary.to_csv(output / "selector_opportunity_summary_by_period.csv", index=False, encoding="utf-8-sig")

    gate_report = _gate_report(events)
    gate_report.to_csv(output / "selector_opportunity_gate_report.csv", index=False, encoding="utf-8-sig")

    (output / "selector_opportunity_final_summary_zh.md").write_text(
        _markdown_summary(events, summary),
        encoding="utf-8",
    )
    metadata = {
        "schema_version": 1,
        "task_id": "TASK-BACKTEST-CORE-POOL3-SELECTOR-OPPORTUNITY-DIAGNOSTIC-001",
        "status": "completed",
        "model": "pool3_selector_opportunity_diagnostic",
        "formal_model_changed": False,
        "trade_decision_changed": False,
        "active_in_trade_decision": False,
        "event_panel_path": str(event_panel_path),
        "outputs": {
            "event_panel": "pool3_selector_opportunity_event_panel.csv",
            "exact_consensus_missing": "exact_consensus_missing_events.csv",
            "selector_preferred_other_pool": "formal_selector_preferred_other_pool_events.csv",
            "forward_returns": "pool3_vs_final_target_forward_returns.csv",
            "opportunity_candidates": "pool3_opportunity_warning_candidates.csv",
            "veto_candidates": "pool3_veto_warning_candidates.csv",
            "summary_by_period": "selector_opportunity_summary_by_period.csv",
            "gate_report": "selector_opportunity_gate_report.csv",
            "summary": "selector_opportunity_final_summary_zh.md",
        },
    }
    (output / "manifest.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame([{"status": "completed", "output_dir": str(output.resolve())}]).to_csv(
        output / "completed.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(columns=["step", "error"]).to_csv(output / "failed.csv", index=False, encoding="utf-8-sig")
    log("completed", "completed", str(output.resolve()))
    (output / "current_step.txt").write_text("completed\n", encoding="utf-8")
    return output


def _validate_event_panel(panel: pd.DataFrame) -> None:
    required = {
        "period",
        "signal_date",
        "pool3_ticker",
        "pool3_has_full_stock_vote",
        "pool3_blocker_category",
        "formal_final_target",
    }
    for horizon in HORIZONS:
        required.update(
            {
                f"pool3_ticker_forward_{horizon}d_return",
                f"formal_final_target_forward_{horizon}d_return",
                f"0050_TW_forward_{horizon}d_return",
                f"00631L_TW_forward_{horizon}d_return",
            }
        )
    missing = required - set(panel.columns)
    if missing:
        raise ValueError("missing event panel columns: " + ",".join(sorted(missing)))


def _selector_event_panel(panel: pd.DataFrame) -> pd.DataFrame:
    subset = panel[
        panel["pool3_has_full_stock_vote"].map(_truthy)
        & panel["pool3_blocker_category"].astype(str).isin(TARGET_BLOCKERS)
    ].copy()
    if subset.empty:
        return subset
    for horizon in HORIZONS:
        subset[f"pool3_minus_formal_{horizon}d"] = _diff(
            subset[f"pool3_ticker_forward_{horizon}d_return"],
            subset[f"formal_final_target_forward_{horizon}d_return"],
        )
        subset[f"pool3_minus_0050_{horizon}d"] = _diff(
            subset[f"pool3_ticker_forward_{horizon}d_return"],
            subset[f"0050_TW_forward_{horizon}d_return"],
        )
        subset[f"pool3_minus_00631L_{horizon}d"] = _diff(
            subset[f"pool3_ticker_forward_{horizon}d_return"],
            subset[f"00631L_TW_forward_{horizon}d_return"],
        )
    subset["pool3_opportunity_score"] = subset.apply(_opportunity_score, axis=1)
    subset["pool3_opportunity_state"] = subset["pool3_opportunity_score"].map(_opportunity_state)
    return subset


def _diff(left: pd.Series, right: pd.Series) -> pd.Series:
    left_num = pd.to_numeric(left, errors="coerce")
    right_num = pd.to_numeric(right, errors="coerce")
    return (left_num - right_num).round(8)


def _opportunity_score(row: pd.Series) -> int:
    score = 0
    for horizon in HORIZONS:
        for benchmark in ("formal", "0050", "00631L"):
            value = row.get(f"pool3_minus_{benchmark}_{horizon}d")
            if pd.notna(value):
                if float(value) > 0:
                    score += 1
                elif float(value) < 0:
                    score -= 1
    return score


def _opportunity_state(score: int) -> str:
    if score >= 5:
        return "opportunity_warning"
    if score <= -5:
        return "veto_warning"
    return "mixed_or_insufficient"


def _forward_comparison_rows(events: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in events.iterrows():
        for horizon in HORIZONS:
            rows.append(
                {
                    "period": row.get("period", ""),
                    "signal_date": row.get("signal_date", ""),
                    "pool3_ticker": row.get("pool3_ticker", ""),
                    "formal_final_target": row.get("formal_final_target", ""),
                    "pool3_blocker_category": row.get("pool3_blocker_category", ""),
                    "horizon_days": horizon,
                    "pool3_return": row.get(f"pool3_ticker_forward_{horizon}d_return", ""),
                    "formal_final_target_return": row.get(f"formal_final_target_forward_{horizon}d_return", ""),
                    "0050_return": row.get(f"0050_TW_forward_{horizon}d_return", ""),
                    "00631L_return": row.get(f"00631L_TW_forward_{horizon}d_return", ""),
                    "pool3_minus_formal": row.get(f"pool3_minus_formal_{horizon}d", ""),
                    "pool3_minus_0050": row.get(f"pool3_minus_0050_{horizon}d", ""),
                    "pool3_minus_00631L": row.get(f"pool3_minus_00631L_{horizon}d", ""),
                    "pool3_opportunity_state": row.get("pool3_opportunity_state", ""),
                }
            )
    return pd.DataFrame(rows)


def _summary_by_period(events: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if events.empty:
        return pd.DataFrame(columns=["period", "events"])
    for period, frame in events.groupby("period", dropna=False):
        row: dict[str, Any] = {
            "period": period,
            "events": int(len(frame)),
            "exact_consensus_missing_events": int((frame["pool3_blocker_category"] == "exact_consensus_missing").sum()),
            "selector_preferred_other_pool_events": int(
                (frame["pool3_blocker_category"] == "formal_target_selector_preferred_other_pool").sum()
            ),
            "opportunity_warning_events": int((frame["pool3_opportunity_state"] == "opportunity_warning").sum()),
            "veto_warning_events": int((frame["pool3_opportunity_state"] == "veto_warning").sum()),
        }
        for horizon in HORIZONS:
            for benchmark in ("formal", "0050", "00631L"):
                column = f"pool3_minus_{benchmark}_{horizon}d"
                row[f"avg_{column}"] = round(float(pd.to_numeric(frame[column], errors="coerce").mean()), 8)
                row[f"win_rate_{column}"] = _rate((pd.to_numeric(frame[column], errors="coerce") > 0).sum(), frame[column].notna().sum())
        rows.append(row)
    return pd.DataFrame(rows)


def _gate_report(events: pd.DataFrame) -> pd.DataFrame:
    total = len(events)
    opportunity = int((events["pool3_opportunity_state"] == "opportunity_warning").sum()) if total else 0
    veto = int((events["pool3_opportunity_state"] == "veto_warning").sum()) if total else 0
    return pd.DataFrame(
        [
            {
                "gate": "opportunity_warning_candidate",
                "description": "Pool3 forward return broadly beats formal target, 0050, and 00631L across horizons.",
                "events": opportunity,
                "share": _rate(opportunity, total),
                "active_in_trade_decision": False,
            },
            {
                "gate": "veto_warning_candidate",
                "description": "Pool3 forward return broadly lags formal target, 0050, and 00631L across horizons.",
                "events": veto,
                "share": _rate(veto, total),
                "active_in_trade_decision": False,
            },
        ]
    )


def _markdown_summary(events: pd.DataFrame, summary: pd.DataFrame) -> str:
    lines = [
        "# Pool3 selector/opportunity diagnostic",
        "",
        "- 狀態：report-only diagnostic；正式模型未變更。",
        f"- Pool3 ignored event rows：{len(events)}",
        "",
        "## 期間摘要",
        "",
    ]
    if summary.empty:
        lines.append("- 無可評估事件。")
    else:
        for _, row in summary.iterrows():
            lines.append(
                f"- {row['period']}: events={int(row['events'])}, "
                f"opportunity={int(row['opportunity_warning_events'])}, "
                f"veto={int(row['veto_warning_events'])}"
            )
    lines.extend(
        [
            "",
            "## 使用邊界",
            "",
            "本輸出只判斷 Pool3 被忽略時是否可能有機會成本，不改正式三池表決，也不是正式交易訊號。",
        ]
    )
    return "\n".join(lines)


def _rate(numerator: float | int, denominator: float | int) -> float:
    return round(float(numerator) / float(denominator), 6) if denominator else 0.0


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Pool3 selector/opportunity report-only diagnostics.")
    parser.add_argument("--event-panel", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = run_pool3_selector_opportunity_diagnostic(
        event_panel_path=args.event_panel,
        output_dir=args.output_dir,
    )
    print(f"OUTPUT_DIR={output.resolve()}")


if __name__ == "__main__":
    main()
