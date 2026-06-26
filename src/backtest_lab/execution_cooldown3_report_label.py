from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


DEFAULT_SOURCE_DIR = "outputs/execution_layer_cooldown3_robustness_20260626"
DEFAULT_OUTPUT_DIR = "outputs/execution_cooldown3_report_label_20260626"
COOLDOWN3_CANDIDATE = "next_day_cooldown_after_exit_to_cash_3"
LABEL_WORDING_ZH = "Cooldown3 在 next-day execution diagnostic 中相對 baseline 有小幅改善，但仍未通過正式 execution layer 驗收。特別是在 2024 hard gate 權值槓桿行情中，歷史回測仍明顯落後 0050正二。此標籤只作執行層診斷與機會成本說明，不改變正式 target，也不是交易指令。"
FORBIDDEN_WORDS = (
    "正式啟用 cooldown3",
    "應改用 cooldown3",
    "應改買 0050正二",
    "execution layer 已完成",
    "正式換倉規則",
    "買進建議",
    "賣出建議",
    "明牌",
)


def run_execution_cooldown3_report_label(
    *,
    source_dir: str | Path = DEFAULT_SOURCE_DIR,
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
        manifest_source = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
        hard_gate = pd.read_csv(source / "hard_gate_2024_attribution.csv").fillna("")
        performance = pd.read_csv(source / "period_performance_by_candidate.csv").fillna("")
        readiness = pd.read_csv(source / "cooldown_robustness_readiness_report.csv").fillna("")

        log("build_label", "started", COOLDOWN3_CANDIDATE)
        label_panel = _build_label_panel(hard_gate, performance, readiness)
        summary = _summary(label_panel)
        boundary = _boundary_markdown()
        _assert_wording_safe(boundary + "\n" + LABEL_WORDING_ZH)

        log("write_outputs", "started", "")
        label_panel.to_csv(output / "execution_cooldown3_report_label_panel.csv", index=False, encoding="utf-8-sig")
        summary.to_csv(output / "execution_cooldown3_label_summary.csv", index=False, encoding="utf-8-sig")
        (output / "execution_cooldown3_wording_boundary_zh.md").write_text(boundary, encoding="utf-8")
        (output / "execution_cooldown3_report_label_summary_zh.md").write_text(_summary_markdown(summary), encoding="utf-8")

        manifest = {
            "schema_version": 1,
            "task_id": "TASK-BACKTEST-CORE-EXECUTION-COOLDOWN3-REPORT-LABEL-001",
            "model": "execution_cooldown3_report_only_label",
            "status": "completed",
            "source_dir": str(source),
            "source_main_candidate": manifest_source.get("main_candidate", COOLDOWN3_CANDIDATE),
            "formal_model_changed": False,
            "trade_decision_changed": False,
            "active_in_trade_decision": False,
            "formal_execution_layer_activated": False,
            "execution_label_active_in_trade_decision": False,
            "pool3_shadow_used": False,
            "final_decision_label_used": False,
            "rr_partial_switch_used": False,
            "uses_forward_return_as_rule": False,
            "valuation_used": False,
            "h3_used": False,
            "label_only_does_not_modify_equity_or_trade_ledger": True,
            "formal_selector_readable": False,
            "forbidden_word_positive_hits": _forbidden_hits(boundary + "\n" + LABEL_WORDING_ZH),
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
        pd.DataFrame([{"step": "run_execution_cooldown3_report_label", "error": str(exc)}]).to_csv(
            output / "failed.csv", index=False, encoding="utf-8-sig"
        )
        log("failed", "failed", str(exc))
        raise


def _build_label_panel(hard_gate: pd.DataFrame, performance: pd.DataFrame, readiness: pd.DataFrame) -> pd.DataFrame:
    hard = hard_gate[hard_gate["variant_id"].astype(str) == COOLDOWN3_CANDIDATE].copy()
    full = performance[
        (performance["variant_id"].astype(str) == COOLDOWN3_CANDIDATE)
        & (performance["period_label"].astype(str) == "full")
    ].copy()
    ready = readiness[readiness["candidate"].astype(str) == COOLDOWN3_CANDIDATE].copy() if "candidate" in readiness.columns else pd.DataFrame()
    row = {
        "execution_diagnostic_label": "cooldown3_report_only_opportunity_cost_caveat",
        "execution_diagnostic_candidate": COOLDOWN3_CANDIDATE,
        "execution_diagnostic_status": "report_only_diagnostic_continue",
        "execution_diagnostic_active_in_trade_decision": False,
        "execution_diagnostic_boundary": "report_only",
        "execution_diagnostic_caveat": "2024_hard_gate_underperforms_0050x2",
        "execution_opportunity_cost_benchmark": "0050x2",
        "execution_opportunity_cost_period": "2024_hard_gate",
        "execution_opportunity_cost_wording_zh": LABEL_WORDING_ZH,
        "formal_selector_readable": False,
        "pit_safe_trigger_basis": "report_label_from_execution_robustness_validation_not_trade_rule",
        "full_return_pct": _value(full, "return_pct"),
        "full_mdd_pct": _value(full, "max_drawdown_pct"),
        "hard_gate_return_pct": _value(hard, "candidate_return_pct"),
        "hard_gate_mdd_pct": _value(hard, "candidate_mdd_pct"),
        "hard_gate_excess_vs_0050x2_pct": _value(hard, "excess_vs_0050x2_pct"),
        "readiness_state": _value(ready, "readiness_state"),
        "blockers": _value(ready, "blockers"),
    }
    return pd.DataFrame([row], columns=_label_columns())


def _label_columns() -> list[str]:
    return [
        "execution_diagnostic_label",
        "execution_diagnostic_candidate",
        "execution_diagnostic_status",
        "execution_diagnostic_active_in_trade_decision",
        "execution_diagnostic_boundary",
        "execution_diagnostic_caveat",
        "execution_opportunity_cost_benchmark",
        "execution_opportunity_cost_period",
        "execution_opportunity_cost_wording_zh",
        "formal_selector_readable",
        "pit_safe_trigger_basis",
        "full_return_pct",
        "full_mdd_pct",
        "hard_gate_return_pct",
        "hard_gate_mdd_pct",
        "hard_gate_excess_vs_0050x2_pct",
        "readiness_state",
        "blockers",
    ]


def _summary(panel: pd.DataFrame) -> pd.DataFrame:
    active = bool(panel["execution_diagnostic_active_in_trade_decision"].map(_truthy).any()) if not panel.empty else False
    formal_readable = bool(panel["formal_selector_readable"].map(_truthy).any()) if not panel.empty else False
    return pd.DataFrame(
        [
            {
                "label": "cooldown3_report_only_opportunity_cost_caveat",
                "row_count": int(len(panel)),
                "candidate": COOLDOWN3_CANDIDATE,
                "active_in_trade_decision": active,
                "formal_selector_readable": formal_readable,
                "boundary": "report_only",
                "wording_safe": not _forbidden_hits(LABEL_WORDING_ZH),
            }
        ]
    )


def _boundary_markdown() -> str:
    return "\n".join(
        [
            "# Cooldown3 execution diagnostic label 文字邊界",
            "",
            LABEL_WORDING_ZH,
            "",
            "邊界：",
            "- 這是 report-only execution diagnostic label。",
            "- execution_diagnostic_active_in_trade_decision=false。",
            "- 不改 formal target、formal selector、trade action、equity 或 trade ledger。",
            "- 不代表正式 execution layer 啟用，也不是交易指令。",
            "",
        ]
    )


def _summary_markdown(summary: pd.DataFrame) -> str:
    row = summary.iloc[0].to_dict()
    return "\n".join(
        [
            "# Execution Cooldown3 Report Label Summary",
            "",
            f"- label: {row.get('label')}",
            f"- candidate: {row.get('candidate')}",
            f"- row_count: {row.get('row_count')}",
            f"- boundary: {row.get('boundary')}",
            "- formal_model_changed=false",
            "- trade_decision_changed=false",
            "- execution_label_active_in_trade_decision=false",
            "",
        ]
    )


def _value(frame: pd.DataFrame, column: str) -> object:
    if frame.empty or column not in frame.columns:
        return ""
    return frame.iloc[0].get(column, "")


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _forbidden_hits(text: str) -> list[str]:
    return [word for word in FORBIDDEN_WORDS if word in text]


def _assert_wording_safe(text: str) -> None:
    hits = _forbidden_hits(text)
    if hits:
        raise ValueError(f"Forbidden wording hits: {hits}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build cooldown3 execution report-only label.")
    parser.add_argument("--source-dir", default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    output = run_execution_cooldown3_report_label(source_dir=args.source_dir, output_dir=args.output_dir)
    print(f"OUTPUT_DIR={output.resolve()}")


if __name__ == "__main__":
    main()
