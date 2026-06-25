from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_OUTPUT_DIR = "outputs/execution_layer_diagnostic_20260625"
MARKET_EXPOSURE_TICKERS = {"0050.TW", "00631L.TW"}


def run_execution_layer_diagnostic(
    *,
    formal_daily_path: str | Path,
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

    log("load_formal_daily", "started", str(formal_daily_path))
    formal_daily = pd.read_csv(formal_daily_path).fillna("")
    _validate_formal_daily(formal_daily)

    log("build_panels", "started")
    target_change = build_formal_target_change_panel(formal_daily)
    transition = build_holding_transition_diagnostic(formal_daily)
    summary = build_execution_gap_summary(formal_daily, target_change, transition)
    preplan = build_execution_variant_preplan()

    log("write_outputs", "started")
    target_change.to_csv(output / "formal_target_change_panel.csv", index=False, encoding="utf-8-sig")
    transition.to_csv(output / "holding_transition_diagnostic.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([summary]).to_csv(output / "execution_gap_summary.csv", index=False, encoding="utf-8-sig")
    preplan.to_csv(output / "execution_variant_preplan.csv", index=False, encoding="utf-8-sig")
    (output / "final_summary_zh.md").write_text(_summary_markdown(summary), encoding="utf-8")

    manifest = {
        "schema_version": 1,
        "task_id": "TASK-BACKTEST-CORE-EXECUTION-LAYER-DIAGNOSTIC-SCAFFOLD-001",
        "model": "execution_layer_diagnostic_only_scaffold",
        "status": "completed",
        "formal_daily_path": str(formal_daily_path),
        "formal_model_changed": False,
        "trade_decision_changed": False,
        "active_in_trade_decision": False,
        "execution_diagnostic_active_in_trade_decision": False,
        "boundary": "report_only_diagnostic",
        "not_formal_execution_layer": True,
        "outputs": {
            "formal_target_change_panel": "formal_target_change_panel.csv",
            "holding_transition_diagnostic": "holding_transition_diagnostic.csv",
            "execution_gap_summary": "execution_gap_summary.csv",
            "execution_variant_preplan": "execution_variant_preplan.csv",
            "summary": "final_summary_zh.md",
            "run_log": "run_log.csv",
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


def build_formal_target_change_panel(formal_daily: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    frame = _normalized_formal_daily(formal_daily)
    previous_target = ""
    previous_change_date: pd.Timestamp | None = None
    previous_change_target = ""
    change_index = 0
    targets = frame["formal_target"].tolist()
    dates = frame["date_ts"].tolist()
    for index, row in frame.iterrows():
        target = str(row["formal_target"]).strip()
        if target == previous_target:
            continue
        change_index += 1
        old_target = previous_target
        days_since_previous_change = ""
        if previous_change_date is not None:
            days_since_previous_change = int((row["date_ts"] - previous_change_date).days)
        reversal_within_3 = False
        reversal_date = ""
        for lookahead in range(index + 1, min(index + 4, len(frame))):
            future_target = str(targets[lookahead]).strip()
            if old_target and future_target == old_target:
                reversal_within_3 = True
                reversal_date = pd.Timestamp(dates[lookahead]).strftime("%Y-%m-%d")
                break
        same_ticker_recut = bool(target and target == previous_change_target and days_since_previous_change != "")
        rows.append(
            {
                "change_index": change_index,
                "date": row["date"],
                "old_target": old_target or "none",
                "new_target": target or "none",
                "old_target_role": _asset_role(old_target),
                "new_target_role": _asset_role(target),
                "target_change_type": _target_change_type(old_target, target),
                "days_since_previous_change": days_since_previous_change,
                "reversal_within_3_trading_rows": reversal_within_3,
                "reversal_to_old_target_date": reversal_date,
                "same_ticker_recut": same_ticker_recut,
                "diagnostic_only": True,
                "active_in_trade_decision": False,
            }
        )
        previous_target = target
        previous_change_date = row["date_ts"]
        previous_change_target = target
    return pd.DataFrame(rows)


def build_holding_transition_diagnostic(formal_daily: pd.DataFrame) -> pd.DataFrame:
    frame = _normalized_formal_daily(formal_daily)
    rows: list[dict[str, Any]] = []
    previous_position = "cash"
    previous_date: pd.Timestamp | None = None
    for _, row in frame.iterrows():
        target = str(row["formal_target"]).strip() or "none"
        position = str(row.get("position_ticker", "")).strip() or "cash"
        action = str(row.get("action", "")).strip() or "hold"
        old_holding = previous_position or "cash"
        new_holding = position or "cash"
        full_rotation = old_holding not in {"cash", "none", ""} and target not in {"cash", "none", ""} and old_holding != target
        same_direction = _asset_role(old_holding) == _asset_role(target) and _asset_role(target) not in {"none", "cash"}
        risk_or_exposure = _asset_role(target) in {"market_exposure", "leveraged_market_exposure", "cash", "none"}
        holding_days_before_transition = ""
        if action in {"buy", "switch"} and previous_date is not None:
            holding_days_before_transition = int((row["date_ts"] - previous_date).days)
        rows.append(
            {
                "date": row["date"],
                "hypothetical_current_holding": old_holding,
                "formal_target": target,
                "resulting_position_after_replay": new_holding,
                "trade_action": action,
                "full_rotation_flag": full_rotation,
                "same_direction_flag": same_direction,
                "risk_or_market_exposure_flag": risk_or_exposure,
                "holding_days_before_transition": holding_days_before_transition,
                "turnover": _number(row.get("turnover")),
                "transaction_cost": _number(row.get("transaction_cost")),
                "execution_diagnostic_active_in_trade_decision": False,
            }
        )
        if action in {"buy", "switch"} or old_holding != new_holding:
            previous_date = row["date_ts"]
        previous_position = new_holding
    return pd.DataFrame(rows)


def build_execution_gap_summary(
    formal_daily: pd.DataFrame,
    target_change: pd.DataFrame,
    transition: pd.DataFrame,
) -> dict[str, Any]:
    frame = _normalized_formal_daily(formal_daily)
    target_change_count = len(target_change)
    short_reversal_count = int(target_change.get("reversal_within_3_trading_rows", pd.Series(dtype=bool)).astype(bool).sum())
    rapid_flip_rate = short_reversal_count / target_change_count if target_change_count else 0.0
    trade_rows = transition[transition["trade_action"].astype(str).isin(["buy", "switch"])].copy()
    holding_days = pd.to_numeric(trade_rows["holding_days_before_transition"], errors="coerce").dropna()
    costs = pd.to_numeric(frame.get("transaction_cost", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    turnover = pd.to_numeric(frame.get("turnover", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    return {
        "status": "completed",
        "diagnostic_only": True,
        "formal_model_changed": False,
        "trade_decision_changed": False,
        "active_in_trade_decision": False,
        "execution_diagnostic_active_in_trade_decision": False,
        "start_date": frame["date"].iloc[0] if not frame.empty else "",
        "end_date": frame["date"].iloc[-1] if not frame.empty else "",
        "row_count": int(len(frame)),
        "target_change_count": int(target_change_count),
        "short_reversal_count": int(short_reversal_count),
        "rapid_flip_rate": round(rapid_flip_rate, 6),
        "trade_action_count": int(frame["action"].astype(str).isin(["buy", "switch"]).sum()),
        "full_rotation_count": int(transition["full_rotation_flag"].astype(bool).sum()) if not transition.empty else 0,
        "market_exposure_transition_count": int(transition["risk_or_market_exposure_flag"].astype(bool).sum()) if not transition.empty else 0,
        "average_holding_days_before_transition": round(float(holding_days.mean()), 4) if not holding_days.empty else "",
        "min_holding_days_before_transition": int(holding_days.min()) if not holding_days.empty else "",
        "total_transaction_cost": round(float(costs.sum()), 2),
        "total_turnover": round(float(turnover.sum()), 2),
        "cost_per_turnover": round(float(costs.sum() / turnover.sum()), 8) if float(turnover.sum()) else "",
        "cost_sensitivity_turnover_10bp": round(float(turnover.sum()) * 0.001, 2),
        "cost_sensitivity_turnover_20bp": round(float(turnover.sum()) * 0.002, 2),
    }


def build_execution_variant_preplan() -> pd.DataFrame:
    rows = [
        ("minimum_hold_N", "候選：要求 formal target 至少持續 N 個交易日才允許執行層承認切換。", "not_enabled"),
        ("cooldown_N", "候選：剛換倉後 N 個交易日內只觀察，不啟動新切換。", "not_enabled"),
        ("partial_switch", "候選：新舊標的衝突時只做部分切換，用於測試降低單次 full rotation 風險。", "not_enabled"),
        ("sell_first_then_buy", "候選：先離開舊標的，再等待確認後進入新標的。", "not_enabled"),
        ("pause_on_conflict", "候選：formal target 與診斷層高度衝突時，只標記暫停觀察。", "not_enabled"),
    ]
    return pd.DataFrame(
        [
            {
                "variant_id": variant_id,
                "description": description,
                "status": status,
                "active_in_trade_decision": False,
                "requires_experiments_validation": True,
            }
            for variant_id, description, status in rows
        ]
    )


def _validate_formal_daily(frame: pd.DataFrame) -> None:
    required = {"date", "winner_ticker", "position_ticker", "action", "turnover", "transaction_cost"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError("missing formal daily columns: " + ",".join(sorted(missing)))


def _normalized_formal_daily(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy().fillna("")
    normalized["date"] = normalized["date"].astype(str)
    normalized["date_ts"] = pd.to_datetime(normalized["date"], errors="coerce")
    normalized = normalized[normalized["date_ts"].notna()].sort_values("date_ts").reset_index(drop=True)
    normalized["formal_target"] = normalized["winner_ticker"].astype(str).str.strip()
    return normalized


def _target_change_type(old_target: str, new_target: str) -> str:
    old_role = _asset_role(old_target)
    new_role = _asset_role(new_target)
    if old_role in {"none", "cash"} and new_role not in {"none", "cash"}:
        return "entry_from_cash_or_no_target"
    if new_role in {"none", "cash"}:
        return "exit_to_cash_or_no_target"
    if old_role != new_role:
        return "role_rotation"
    return "same_role_rotation"


def _asset_role(ticker: str) -> str:
    ticker = str(ticker or "").strip()
    if not ticker or ticker == "none":
        return "none"
    if ticker == "cash":
        return "cash"
    if ticker == "00631L.TW":
        return "leveraged_market_exposure"
    if ticker == "0050.TW":
        return "market_exposure"
    if ticker in MARKET_EXPOSURE_TICKERS:
        return "market_exposure"
    return "stock_attack"


def _number(value: object) -> float:
    return float(pd.to_numeric(pd.Series([value]), errors="coerce").fillna(0.0).iloc[0])


def _summary_markdown(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Execution Layer Diagnostic-only Scaffold",
            "",
            "本輸出只用來檢查正式 baseline target stream 的執行層缺口，不是正式 execution / exit layer。",
            "",
            "## 邊界",
            "",
            "- formal_model_changed=false",
            "- trade_decision_changed=false",
            "- active_in_trade_decision=false",
            "- execution_diagnostic_active_in_trade_decision=false",
            "",
            "## 摘要",
            "",
            f"- 期間：{summary.get('start_date')} ~ {summary.get('end_date')}",
            f"- formal target 變化次數：{summary.get('target_change_count')}",
            f"- 1-3 個交易列內反轉次數：{summary.get('short_reversal_count')}",
            f"- rapid flip rate：{summary.get('rapid_flip_rate')}",
            f"- full rotation 次數：{summary.get('full_rotation_count')}",
            f"- 平均換倉前持有天數：{summary.get('average_holding_days_before_transition')}",
            f"- 交易成本敏感度 10bp：{summary.get('cost_sensitivity_turnover_10bp')}",
            "",
            "## 下一步",
            "",
            "若 Experiments 要驗證 minimum_hold、cooldown、partial_switch、sell_first_then_buy 或 pause_on_conflict，必須另跑 diagnostic-only A/B；本任務不啟用任何正式規則。",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build report-only execution/exit layer diagnostic scaffold.")
    parser.add_argument("--formal-daily", required=True)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    output = run_execution_layer_diagnostic(
        formal_daily_path=args.formal_daily,
        output_dir=args.output_dir,
    )
    print(f"OUTPUT_DIR={output.resolve()}")


if __name__ == "__main__":
    main()
