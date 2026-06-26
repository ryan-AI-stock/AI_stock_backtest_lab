from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_SOURCE_DIR = "outputs/pool1_pool2_market_exposure_override_panels_20260626"
DEFAULT_OUTPUT_DIR = "outputs/pool1_pool2_0050x2_opportunity_label_20260626"
LABEL_WORDING_ZH = "這段屬於 0050正二機會成本警示：歷史回測顯示，在特定權值槓桿行情中，主候選可能落後 0050正二。此標籤只作風險與機會成本說明，不改變正式模型 target，也不是交易指令。"
FORBIDDEN_WORDS = ("應該改買", "0050正二更好", "模型建議切換", "正式 override", "買進建議", "賣出建議", "明牌")


def run_pool1_pool2_0050x2_opportunity_label(
    *,
    source_dir: str | Path = DEFAULT_SOURCE_DIR,
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
        label_source = pd.read_csv(source / "0050x2_opportunity_cost_label_panel.csv").fillna("")
        base_daily = pd.read_csv(source / "daily_equity_by_variant.csv").fillna("")
        base_trades = pd.read_csv(source / "trade_ledger_by_variant.csv").fillna("")
        manifest_source = json.loads((source / "manifest.json").read_text(encoding="utf-8"))

        log("build_label", "started", "")
        label_panel = _build_label_panel(label_source)
        summary = _summary(label_panel)
        boundary = _boundary_markdown()
        _assert_wording_safe(boundary + "\n" + LABEL_WORDING_ZH)
        equity_hash = _stable_shape_hash(base_daily)
        trade_hash = _stable_shape_hash(base_trades)

        log("write_outputs", "started", "")
        label_panel.to_csv(output / "0050x2_opportunity_label_panel.csv", index=False, encoding="utf-8-sig")
        summary.to_csv(output / "0050x2_opportunity_label_summary.csv", index=False, encoding="utf-8-sig")
        (output / "report_wording_boundary_zh.md").write_text(boundary, encoding="utf-8")
        (output / "pool1_pool2_0050x2_opportunity_label_summary_zh.md").write_text(_summary_markdown(summary), encoding="utf-8")
        manifest = {
            "schema_version": 1,
            "task_id": "TASK-BACKTEST-CORE-POOL1-POOL2-0050X2-OPPORTUNITY-LABEL-001",
            "model": "pool1_pool2_0050x2_opportunity_cost_report_label",
            "status": "completed",
            "source_dir": str(source),
            "source_formal_model_changed": manifest_source.get("formal_model_changed", False),
            "formal_model_changed": False,
            "trade_decision_changed": False,
            "formal_absorption_ready": False,
            "opportunity_cost_label_active_in_trade_decision": False,
            "market_exposure_override_absorbed": False,
            "pool3_shadow_used_as_formal": False,
            "final_decision_label_used_as_formal": False,
            "rr_partial_switch_used_in_performance": False,
            "valuation_used": False,
            "h3_used": False,
            "uses_forward_return_as_rule": False,
            "label_only_does_not_modify_equity_or_trade_ledger": True,
            "source_daily_equity_shape_hash": equity_hash,
            "source_trade_ledger_shape_hash": trade_hash,
            "forbidden_word_positive_hits": _forbidden_hits(boundary + "\n" + LABEL_WORDING_ZH),
        }
        (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        pd.DataFrame([{"status": "completed", "output_dir": str(output.resolve())}]).to_csv(output / "completed.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(columns=["step", "error"]).to_csv(output / "failed.csv", index=False, encoding="utf-8-sig")
        log("completed", "completed", str(output.resolve()))
        (output / "current_step.txt").write_text("completed\n", encoding="utf-8")
        return output
    except Exception as exc:
        pd.DataFrame([{"step": "run_pool1_pool2_0050x2_opportunity_label", "error": str(exc)}]).to_csv(output / "failed.csv", index=False, encoding="utf-8-sig")
        log("failed", "failed", str(exc))
        raise


def _build_label_panel(source: pd.DataFrame) -> pd.DataFrame:
    frame = source.copy()
    if frame.empty:
        return pd.DataFrame(columns=_label_columns())
    frame["benchmark_opportunity_cost_label"] = "0050x2_opportunity_cost_warning"
    frame["benchmark_opportunity_cost_benchmark"] = "0050x2"
    frame["benchmark_opportunity_cost_period"] = frame.get("period", "")
    frame["benchmark_opportunity_cost_reason"] = "specific_large_cap_leverage_regime_caveat"
    frame["benchmark_opportunity_cost_active_in_trade_decision"] = False
    frame["benchmark_opportunity_cost_boundary"] = "report_only"
    frame["benchmark_opportunity_cost_wording_zh"] = LABEL_WORDING_ZH
    frame["formal_selector_readable"] = False
    frame["pit_safe_trigger_basis"] = "report_label_from_historical_validation_not_trade_rule"
    return frame[_label_columns()]


def _label_columns() -> list[str]:
    return [
        "date",
        "period",
        "pool1_vote",
        "pool2_vote",
        "target_weights",
        "benchmark_opportunity_cost_label",
        "benchmark_opportunity_cost_benchmark",
        "benchmark_opportunity_cost_period",
        "benchmark_opportunity_cost_reason",
        "benchmark_opportunity_cost_active_in_trade_decision",
        "benchmark_opportunity_cost_boundary",
        "benchmark_opportunity_cost_wording_zh",
        "formal_selector_readable",
        "pit_safe_trigger_basis",
    ]


def _summary(panel: pd.DataFrame) -> pd.DataFrame:
    if panel.empty:
        return pd.DataFrame([{"label": "0050x2_opportunity_cost_warning", "row_count": 0, "active_in_trade_decision": False, "boundary": "report_only"}])
    return pd.DataFrame(
        [
            {
                "label": "0050x2_opportunity_cost_warning",
                "row_count": len(panel),
                "period_count": panel["benchmark_opportunity_cost_period"].nunique(),
                "active_in_trade_decision": bool(panel["benchmark_opportunity_cost_active_in_trade_decision"].map(_truthy).any()),
                "formal_selector_readable": bool(panel["formal_selector_readable"].map(_truthy).any()),
                "boundary": "report_only",
                "wording_safe": not _forbidden_hits(LABEL_WORDING_ZH),
            }
        ]
    )


def _boundary_markdown() -> str:
    return "\n".join(
        [
            "# 0050正二機會成本警示文字邊界",
            "",
            LABEL_WORDING_ZH,
            "",
            "邊界：",
            "- 這是 report-only label。",
            "- active_in_trade_decision=false。",
            "- 不改 formal target、formal selector、trade ledger 或 equity。",
            "- 不代表正式覆蓋規則，也不是交易指令。",
            "",
        ]
    )


def _summary_markdown(summary: pd.DataFrame) -> str:
    row = summary.iloc[0].to_dict()
    return "\n".join(
        [
            "# Pool1+Pool2 0050正二 Opportunity-Cost Label Summary",
            "",
            f"- label: {row.get('label')}",
            f"- row_count: {row.get('row_count')}",
            f"- boundary: {row.get('boundary')}",
            "- formal_model_changed=false",
            "- trade_decision_changed=false",
            "- opportunity_cost_label_active_in_trade_decision=false",
            "",
        ]
    )


def _stable_shape_hash(frame: pd.DataFrame) -> str:
    return f"rows={len(frame)};cols={len(frame.columns)};columns={','.join(map(str, frame.columns[:10]))}"


def _forbidden_hits(text: str) -> list[str]:
    return [word for word in FORBIDDEN_WORDS if word in text]


def _assert_wording_safe(text: str) -> None:
    hits = _forbidden_hits(text)
    if hits:
        raise ValueError(f"Forbidden wording hits: {hits}")


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build 0050x2 opportunity-cost report-only label.")
    parser.add_argument("--source-dir", default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    output = run_pool1_pool2_0050x2_opportunity_label(source_dir=args.source_dir, output_dir=args.output_dir)
    print(f"OUTPUT_DIR={output.resolve()}")


if __name__ == "__main__":
    main()
