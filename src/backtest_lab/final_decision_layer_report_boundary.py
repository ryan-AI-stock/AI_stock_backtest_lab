from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_STATE_PANEL = "outputs/final_decision_layer_spec_diagnostic_20260625/final_decision_state_panel.csv"
DEFAULT_FORWARD_BY_STATE = "outputs/final_decision_layer_forward_outcome_adapter_20260625/forward_outcome_by_state.csv"
DEFAULT_OUTPUT_DIR = "outputs/final_decision_layer_report_boundary_20260625"
REPORT_ONLY_BOUNDARY = "report_only_diagnostic"


def run_final_decision_layer_report_boundary(
    *,
    state_panel_path: str | Path = DEFAULT_STATE_PANEL,
    forward_by_state_path: str | Path = DEFAULT_FORWARD_BY_STATE,
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
        forward_by_state = _load_forward_by_state(forward_by_state_path)

        log("build_report_boundary", "started", "")
        panel = build_report_boundary_panel(state_panel, forward_by_state)
        summary = build_report_boundary_state_summary(panel)

        log("write_outputs", "started", "")
        panel.to_csv(output / "final_decision_report_boundary_panel.csv", index=False, encoding="utf-8-sig")
        summary.to_csv(output / "report_boundary_state_summary.csv", index=False, encoding="utf-8-sig")
        (output / "final_decision_report_boundary_summary_zh.md").write_text(
            _summary_markdown(summary),
            encoding="utf-8",
        )
        manifest = {
            "schema_version": 1,
            "task_id": "TASK-BACKTEST-CORE-FINAL-DECISION-LAYER-REPORT-BOUNDARY-001",
            "model": "final_decision_layer_report_boundary",
            "status": "completed",
            "state_panel_path": str(state_panel_path),
            "forward_by_state_path": str(forward_by_state_path),
            "formal_model_changed": False,
            "trade_decision_changed": False,
            "active_in_trade_decision": False,
            "final_decision_layer_boundary": REPORT_ONLY_BOUNDARY,
            "report_boundary_active_in_trade_decision": False,
            "pool3_shadow_used_as_formal": False,
            "etf_counted_as_stock_vote": False,
            "rr_partial_switch_used": False,
            "valuation_used": False,
            "h3_used": False,
            "outputs": {
                "report_boundary_panel": "final_decision_report_boundary_panel.csv",
                "state_summary": "report_boundary_state_summary.csv",
                "summary": "final_decision_report_boundary_summary_zh.md",
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
        pd.DataFrame([{"step": "run_final_decision_layer_report_boundary", "error": str(exc)}]).to_csv(
            output / "failed.csv",
            index=False,
            encoding="utf-8-sig",
        )
        log("failed", "failed", str(exc))
        raise


def build_report_boundary_panel(state_panel: pd.DataFrame, forward_by_state: pd.DataFrame) -> pd.DataFrame:
    evidence = _evidence_by_state(forward_by_state)
    rows: list[dict[str, Any]] = []
    for item in state_panel.to_dict(orient="records"):
        state = _text(item.get("final_decision_state"))
        policy = _state_policy(state)
        evidence_row = evidence.get(state, {})
        final_target_type = _text(item.get("final_target_type"))
        rows.append(
            {
                "signal_date": item.get("signal_date", item.get("date", "")),
                "period_label": item.get("period", ""),
                "final_decision_state": state,
                "final_target_type": final_target_type,
                "final_target_ticker": item.get("final_target_ticker", ""),
                "final_target_source": item.get("final_target_source", ""),
                "final_decision_report_confidence": policy["confidence"],
                "final_decision_report_boundary": policy["boundary"],
                "final_decision_user_reading_state": policy["user_reading_state"],
                "strong_consensus_confidence_note": policy["strong_note"],
                "divergence_fail_closed_warning": policy["divergence_warning"],
                "market_exposure_explanation_note": policy["market_note"],
                "final_decision_label_active_in_trade_decision": False,
                "report_boundary_active_in_trade_decision": False,
                "not_eligible_for_formal_selector": _truthy(item.get("not_eligible_for_formal_selector", False))
                or policy["not_formal"],
                "formal_model_changed": False,
                "trade_decision_changed": False,
                "active_in_trade_decision": False,
                "etf_counted_as_stock_vote": False,
                "pool3_shadow_used_as_formal": False,
                "event_study_complete_coverage_rate": evidence_row.get("complete_coverage_rate", ""),
                "event_study_forward_20d_mean": evidence_row.get("forward_20d_mean", ""),
                "event_study_forward_60d_mean": evidence_row.get("forward_60d_mean", ""),
                "event_study_forward_120d_mean": evidence_row.get("forward_120d_mean", ""),
                "event_study_note": policy["evidence_note"],
            }
        )
    return pd.DataFrame(rows)


def build_report_boundary_state_summary(panel: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in panel.groupby(
        ["final_decision_state", "final_decision_report_confidence", "final_decision_user_reading_state"],
        dropna=False,
    ):
        state, confidence, reading = keys
        rows.append(
            {
                "final_decision_state": state,
                "final_decision_report_confidence": confidence,
                "final_decision_user_reading_state": reading,
                "row_count": int(len(group)),
                "active_in_trade_decision_count": int(group["final_decision_label_active_in_trade_decision"].map(_truthy).sum()),
                "not_eligible_for_formal_selector_count": int(group["not_eligible_for_formal_selector"].map(_truthy).sum()),
                "sample_note": _text(group.iloc[0].get("event_study_note", "")),
            }
        )
    return pd.DataFrame(rows).sort_values(["final_decision_state"]).reset_index(drop=True)


def _load_forward_by_state(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    if not source.exists():
        return pd.DataFrame(columns=["final_decision_state"])
    return pd.read_csv(source).fillna("")


def _validate_state_panel(panel: pd.DataFrame) -> None:
    required = {"final_decision_state"}
    missing = sorted(required - set(panel.columns))
    if missing:
        raise ValueError(f"state panel missing required columns: {missing}")


def _evidence_by_state(forward_by_state: pd.DataFrame) -> dict[str, dict[str, Any]]:
    if forward_by_state.empty or "final_decision_state" not in forward_by_state.columns:
        return {}
    return {
        _text(row.get("final_decision_state")): {
            "complete_coverage_rate": row.get("complete_coverage_rate", ""),
            "forward_20d_mean": row.get("forward_20d_mean", ""),
            "forward_60d_mean": row.get("forward_60d_mean", ""),
            "forward_120d_mean": row.get("forward_120d_mean", ""),
        }
        for row in forward_by_state.to_dict(orient="records")
    }


def _state_policy(state: str) -> dict[str, Any]:
    if state == "strong_consensus":
        return {
            "confidence": "strong_consensus_supported",
            "boundary": "report_only_confidence_label",
            "user_reading_state": "strong_consensus_supported",
            "strong_note": "強共識狀態；歷史 forward outcome 較佳，但不代表保證績效。",
            "divergence_warning": "",
            "market_note": "",
            "evidence_note": "event-study 支持作為報告信心標籤；仍不得當成績效承諾。",
            "not_formal": False,
        }
    if state == "weak_consensus":
        return {
            "confidence": "weak_or_mixed",
            "boundary": "report_only_observation",
            "user_reading_state": "weak_or_mixed",
            "strong_note": "",
            "divergence_warning": "",
            "market_note": "",
            "evidence_note": "弱共識僅供觀察，不提高正式性。",
            "not_formal": True,
        }
    if state == "actionable_divergence":
        return {
            "confidence": "divergence_fail_closed",
            "boundary": "report_only_watch_only_fail_closed",
            "user_reading_state": "divergence_watch_only",
            "strong_note": "",
            "divergence_warning": "三池分歧且歷史 outcome 不支持追擊；僅列觀察與風險提醒，不進正式決策。",
            "market_note": "",
            "evidence_note": "event-study 支持 fail-closed，不支援升級為正式 selector。",
            "not_formal": True,
        }
    if state == "diagnostic_divergence":
        return {
            "confidence": "diagnostic_only",
            "boundary": "report_only_diagnostic",
            "user_reading_state": "diagnostic_only",
            "strong_note": "",
            "divergence_warning": "分歧診斷僅用於解釋模型狀態，不進正式決策。",
            "market_note": "",
            "evidence_note": "診斷層，不提高正式性。",
            "not_formal": True,
        }
    if state == "defensive_market_exposure":
        return {
            "confidence": "market_exposure_explanation",
            "boundary": "report_only_market_exposure_layer",
            "user_reading_state": "market_exposure_explanation",
            "strong_note": "",
            "divergence_warning": "",
            "market_note": "市場曝險說明層；ETF 或槓桿 ETF 不計入股票 exact consensus。",
            "evidence_note": "曝險工具只能作說明與 benchmark context，不是股票票。",
            "not_formal": True,
        }
    if state == "data_insufficient":
        return {
            "confidence": "data_blocked",
            "boundary": "report_only_data_blocked",
            "user_reading_state": "data_blocked",
            "strong_note": "",
            "divergence_warning": "",
            "market_note": "",
            "evidence_note": "資料不足，fail-closed。",
            "not_formal": True,
        }
    if state == "forced_stop":
        return {
            "confidence": "forced_stop",
            "boundary": "report_only_forced_stop",
            "user_reading_state": "forced_stop",
            "strong_note": "",
            "divergence_warning": "",
            "market_note": "",
            "evidence_note": "強制停手狀態優先於其他報告標籤。",
            "not_formal": True,
        }
    return {
        "confidence": "diagnostic_only",
        "boundary": "report_only_unknown_state",
        "user_reading_state": "diagnostic_only",
        "strong_note": "",
        "divergence_warning": "",
        "market_note": "",
        "evidence_note": "未知狀態，僅保留診斷。",
        "not_formal": True,
    }


def _summary_markdown(summary: pd.DataFrame) -> str:
    total = int(summary["row_count"].sum()) if not summary.empty else 0
    active = int(summary["active_in_trade_decision_count"].sum()) if not summary.empty else 0
    lines = [
        "# Final Decision Layer Report Boundary",
        "",
        "本輸出只整理 final decision layer 的報告文字邊界，不改正式模型、正式 selector、正式 vote、正式 target 或交易行為。",
        "",
        f"- 總列數：{total}",
        f"- active_in_trade_decision rows：{active}",
        "- strong_consensus：可作報告信心標籤，但不代表保證績效。",
        "- actionable_divergence：維持 fail-closed / watch-only，不進正式 selector。",
        "- defensive_market_exposure：只是 market exposure explanation layer，不是股票 exact consensus。",
        "",
        "## State Summary",
        "",
    ]
    for row in summary.to_dict(orient="records"):
        lines.append(
            f"- {row['final_decision_state']}: {row['final_decision_report_confidence']} "
            f"({int(row['row_count'])} rows)"
        )
    return "\n".join(lines) + "\n"


def _text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    if text.lower() == "nan":
        return ""
    return text.strip()


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return _text(value).lower() in {"1", "true", "yes", "y"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build final decision layer report-only wording boundary.")
    parser.add_argument("--state-panel", default=DEFAULT_STATE_PANEL)
    parser.add_argument("--forward-by-state", default=DEFAULT_FORWARD_BY_STATE)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    output = run_final_decision_layer_report_boundary(
        state_panel_path=args.state_panel,
        forward_by_state_path=args.forward_by_state,
        output_dir=args.output_dir,
    )
    print(f"OUTPUT_DIR={output.resolve()}")


if __name__ == "__main__":
    main()
