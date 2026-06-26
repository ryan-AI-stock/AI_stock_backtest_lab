from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from backtest_lab.execution_layer_next_day_ab_pool1_pool2_formal import (
    FORMAL_MODEL_TARGET,
    _normalize_stream,
    _parse_weights,
    _validate_stream,
)


DEFAULT_REVIEW_DIR = "outputs/execution_layer_review_pool1_pool2_formal_20260626"
DEFAULT_OUTPUT_DIR = "outputs/execution_reversal_warning_label_20260626"
WARNING_WORDING_ZH = (
    "本日正式 target 屬於近期快速反轉觀察情境。這代表模型 target 在 1-3 個交易列內出現來回切換，"
    "適合人工確認換倉穩定性、滑價與既有部位狀態；此標籤只作 report-only 風險提示，"
    "不改變正式 target，也不是交易指令。"
)
NO_WARNING_WORDING_ZH = (
    "本日正式 target 未命中 1-3 個交易列快速反轉警示。此狀態只代表換倉穩定性觀察未觸發，"
    "不代表保證績效，也不代表正式 execution layer 已成立。"
)
FORBIDDEN_WORDS = (
    "應該買",
    "應該賣",
    "正式啟用",
    "正式換倉規則",
    "交易指令",
    "明牌",
    "保證獲利",
    "保證績效會更好",
)


def run_execution_reversal_warning_label(
    *,
    review_dir: str | Path = DEFAULT_REVIEW_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    max_window_rows: int = 3,
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
        review = Path(review_dir)
        log("load_formal_target_stream", "started", str(review))
        stream = pd.read_csv(review / "formal_target_stream_adapter.csv").fillna("")
        _validate_stream(stream)
        frame = _normalize_stream(stream)

        log("build_warning_panel", "started", "")
        event_panel = _build_reversal_event_panel(frame, max_window_rows=max_window_rows)
        warning_panel = _build_current_warning_panel(frame, event_panel, max_window_rows=max_window_rows)
        summary = _summary(warning_panel, event_panel)
        wording = _wording_markdown(warning_panel)
        _assert_wording_safe(wording)

        log("write_outputs", "started", "")
        warning_panel.to_csv(output / "execution_reversal_warning_label_panel.csv", index=False, encoding="utf-8-sig")
        event_panel.to_csv(output / "execution_reversal_event_history.csv", index=False, encoding="utf-8-sig")
        summary.to_csv(output / "execution_reversal_warning_summary.csv", index=False, encoding="utf-8-sig")
        (output / "execution_reversal_warning_wording_zh.md").write_text(wording, encoding="utf-8")
        (output / "execution_reversal_warning_label_summary_zh.md").write_text(_summary_markdown(summary), encoding="utf-8")

        manifest = {
            "schema_version": 1,
            "task_id": "TASK-BACKTEST-CORE-EXECUTION-REVERSAL-WARNING-LABEL-001",
            "model": "execution_reversal_warning_report_only_label",
            "status": "completed",
            "formal_model_target": FORMAL_MODEL_TARGET,
            "formal_model_route": "pool1_primary_pool2_confirmation_cap40",
            "source_review_dir": str(review),
            "formal_model_changed": False,
            "trade_decision_changed": False,
            "active_in_trade_decision": False,
            "formal_execution_layer_activated": False,
            "execution_reversal_warning_active_in_trade_decision": False,
            "execution_reversal_warning_boundary": "report_only",
            "formal_selector_readable": False,
            "label_only_does_not_modify_equity_or_trade_ledger": True,
            "uses_forward_return_as_rule": False,
            "pool3_shadow_used": False,
            "final_decision_label_used": False,
            "rr_partial_switch_used": False,
            "valuation_used": False,
            "h3_used": False,
            "max_window_rows": max_window_rows,
            "warning_triggered": bool(warning_panel["execution_reversal_warning_triggered"].astype(bool).any()) if not warning_panel.empty else False,
            "forbidden_word_positive_hits": _forbidden_hits(wording),
            "outputs": {
                "warning_panel": "execution_reversal_warning_label_panel.csv",
                "event_history": "execution_reversal_event_history.csv",
                "summary": "execution_reversal_warning_summary.csv",
                "wording": "execution_reversal_warning_wording_zh.md",
            },
        }
        (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        pd.DataFrame([{"status": "completed", "output_dir": str(output.resolve())}]).to_csv(output / "completed.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(columns=["step", "error"]).to_csv(output / "failed.csv", index=False, encoding="utf-8-sig")
        log("completed", "completed", str(output.resolve()))
        (output / "current_step.txt").write_text("completed\n", encoding="utf-8")
        return output
    except Exception as exc:
        pd.DataFrame([{"step": "run_execution_reversal_warning_label", "error": str(exc)}]).to_csv(output / "failed.csv", index=False, encoding="utf-8-sig")
        log("failed", "failed", str(exc))
        raise


def _build_reversal_event_panel(frame: pd.DataFrame, *, max_window_rows: int) -> pd.DataFrame:
    keys = [_target_key(_parse_weights(value)) or "cash" for value in frame["target_weights"].tolist()]
    rows: list[dict[str, Any]] = []
    for index in range(1, len(keys)):
        from_key = keys[index - 1]
        temporary_key = keys[index]
        if temporary_key == from_key:
            continue
        for offset in range(1, max_window_rows + 1):
            check = index + offset
            if check >= len(keys):
                continue
            if keys[check] == from_key:
                rows.append(
                    {
                        "switch_date": str(frame.iloc[index]["date"]),
                        "reversal_date": str(frame.iloc[check]["date"]),
                        "from_target": from_key,
                        "temporary_target": temporary_key,
                        "reversal_target": keys[check],
                        "window_rows": offset,
                        "target_to_target_reversal": bool(from_key != "cash" and temporary_key != "cash"),
                        "cash_involved": bool("cash" in {from_key, temporary_key}),
                        "execution_reversal_warning_active_in_trade_decision": False,
                    }
                )
                break
    return pd.DataFrame(rows, columns=_event_columns())


def _build_current_warning_panel(frame: pd.DataFrame, event_panel: pd.DataFrame, *, max_window_rows: int) -> pd.DataFrame:
    latest = frame.iloc[-1].to_dict() if not frame.empty else {}
    latest_date = str(latest.get("date", ""))
    latest_key = _target_key(_parse_weights(latest.get("target_weights"))) or "cash"
    prior_key = "cash"
    if len(frame) >= 2:
        prior_key = _target_key(_parse_weights(frame.iloc[-2].get("target_weights"))) or "cash"
    matched = event_panel[event_panel["reversal_date"].astype(str).eq(latest_date)].copy() if not event_panel.empty else pd.DataFrame()
    triggered = not matched.empty
    event = matched.iloc[0].to_dict() if triggered else {}
    state = "reversal_warning" if triggered else "none"
    manual_context = bool(triggered and event.get("target_to_target_reversal", False))
    if triggered and event.get("cash_involved", False):
        manual_context = True
    wording = WARNING_WORDING_ZH if triggered else NO_WARNING_WORDING_ZH
    return pd.DataFrame(
        [
            {
                "signal_date": latest_date,
                "formal_model_target": FORMAL_MODEL_TARGET,
                "formal_model_route": "pool1_primary_pool2_confirmation_cap40",
                "current_formal_target": latest_key,
                "previous_formal_target": prior_key,
                "execution_reversal_warning_label": state,
                "execution_reversal_warning_triggered": triggered,
                "execution_reversal_warning_window_rows": event.get("window_rows", ""),
                "reversal_from_target": event.get("from_target", ""),
                "reversal_temporary_target": event.get("temporary_target", ""),
                "target_to_target_reversal": event.get("target_to_target_reversal", False) if triggered else False,
                "cash_involved": event.get("cash_involved", False) if triggered else False,
                "manual_confirmation_context": manual_context,
                "not_suitable_for_mechanical_full_switch_note": wording if triggered else "",
                "execution_reversal_warning_wording_zh": wording,
                "execution_reversal_warning_boundary": "report_only",
                "execution_reversal_warning_active_in_trade_decision": False,
                "formal_selector_readable": False,
                "label_only_does_not_modify_equity_or_trade_ledger": True,
                "uses_forward_return_as_rule": False,
                "max_window_rows": max_window_rows,
            }
        ]
    )


def _summary(warning_panel: pd.DataFrame, event_panel: pd.DataFrame) -> pd.DataFrame:
    row = warning_panel.iloc[0].to_dict() if not warning_panel.empty else {}
    return pd.DataFrame(
        [
            {
                "signal_date": row.get("signal_date", ""),
                "warning_triggered": bool(row.get("execution_reversal_warning_triggered", False)),
                "current_formal_target": row.get("current_formal_target", ""),
                "previous_formal_target": row.get("previous_formal_target", ""),
                "historical_reversal_event_count": int(len(event_panel)),
                "historical_target_to_target_reversal_count": int(event_panel["target_to_target_reversal"].astype(bool).sum()) if not event_panel.empty else 0,
                "historical_cash_involved_reversal_count": int(event_panel["cash_involved"].astype(bool).sum()) if not event_panel.empty else 0,
                "active_in_trade_decision": False,
                "boundary": "report_only",
            }
        ]
    )


def _wording_markdown(warning_panel: pd.DataFrame) -> str:
    row = warning_panel.iloc[0].to_dict() if not warning_panel.empty else {}
    triggered = bool(row.get("execution_reversal_warning_triggered", False))
    wording = row.get("execution_reversal_warning_wording_zh") or NO_WARNING_WORDING_ZH
    lines = [
        "# Execution reversal warning label 文字邊界",
        "",
        str(wording),
        "",
        "邊界：",
        "- 這是 report-only execution warning label。",
        "- execution_reversal_warning_active_in_trade_decision=false。",
        "- 不改 formal target、formal selector、trade action、equity 或 trade ledger。",
        "- 不代表正式 execution layer 啟用。",
        "- 用途是提醒人工檢查換倉穩定性，不替使用者決定買賣。",
        "",
        "目前狀態：",
        f"- warning_triggered={triggered}",
        f"- current_formal_target={row.get('current_formal_target', '')}",
        f"- previous_formal_target={row.get('previous_formal_target', '')}",
    ]
    return "\n".join(lines)


def _summary_markdown(summary: pd.DataFrame) -> str:
    row = summary.iloc[0].to_dict() if not summary.empty else {}
    return "\n".join(
        [
            "# Execution Reversal Warning Label Summary",
            "",
            f"- signal_date: {row.get('signal_date', '')}",
            f"- warning_triggered: {row.get('warning_triggered', False)}",
            f"- current_formal_target: {row.get('current_formal_target', '')}",
            f"- previous_formal_target: {row.get('previous_formal_target', '')}",
            f"- historical_reversal_event_count: {row.get('historical_reversal_event_count', 0)}",
            f"- boundary: {row.get('boundary', 'report_only')}",
            "- formal_model_changed=false",
            "- trade_decision_changed=false",
            "- execution_reversal_warning_active_in_trade_decision=false",
            "",
        ]
    )


def _target_key(weights: dict[str, float]) -> str:
    if not weights:
        return ""
    return "|".join(f"{ticker}:{weight:.6f}" for ticker, weight in sorted(weights.items()))


def _event_columns() -> list[str]:
    return [
        "switch_date",
        "reversal_date",
        "from_target",
        "temporary_target",
        "reversal_target",
        "window_rows",
        "target_to_target_reversal",
        "cash_involved",
        "execution_reversal_warning_active_in_trade_decision",
    ]


def _forbidden_hits(text: str) -> list[str]:
    return [word for word in FORBIDDEN_WORDS if word in text]


def _assert_wording_safe(text: str) -> None:
    hits = _forbidden_hits(text)
    if hits:
        raise ValueError(f"Forbidden wording hits: {hits}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build report-only execution reversal warning label.")
    parser.add_argument("--review-dir", default=DEFAULT_REVIEW_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-window-rows", type=int, default=3)
    args = parser.parse_args()
    output = run_execution_reversal_warning_label(
        review_dir=args.review_dir,
        output_dir=args.output_dir,
        max_window_rows=args.max_window_rows,
    )
    print(f"OUTPUT_DIR={output.resolve()}")


if __name__ == "__main__":
    main()
