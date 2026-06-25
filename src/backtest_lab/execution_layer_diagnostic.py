from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from backtest_lab.data import load_price_csv


DEFAULT_OUTPUT_DIR = "outputs/execution_layer_diagnostic_20260625"
MARKET_EXPOSURE_TICKERS = {"0050.TW", "00631L.TW"}
FORWARD_HORIZONS = (5, 20, 60)


def run_execution_layer_diagnostic(
    *,
    formal_daily_path: str | Path,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    price_cache_dir: str | Path | None = None,
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

    log("load_price_cache", "started", str(price_cache_dir or "not_provided"))
    prices = _load_prices_for_execution_study(formal_daily, Path(price_cache_dir)) if price_cache_dir else {}

    log("build_panels", "started")
    target_change = build_formal_target_change_panel(formal_daily)
    transition = build_holding_transition_diagnostic(formal_daily)
    summary = build_execution_gap_summary(formal_daily, target_change, transition)
    preplan = build_execution_variant_preplan()
    event_study = build_execution_event_study_panel(formal_daily, target_change, prices)
    hold_cooldown = build_minimum_hold_cooldown_event_study(event_study)
    partial_daily = build_partial_switch_simulator_daily(formal_daily, partial_switch_weight=0.5)
    partial_summary = build_partial_switch_summary(partial_daily)
    sell_first_gap = build_sell_first_then_buy_gap_study(event_study)
    pause_readiness = build_pause_on_conflict_input_readiness(formal_daily, event_study)
    event_forward_coverage = _event_forward_data_coverage(event_study)
    field_contract = build_execution_layer_field_contract(
        price_cache_dir=price_cache_dir,
        prices_loaded=bool(prices),
        event_forward_coverage=event_forward_coverage,
    )

    log("write_outputs", "started")
    target_change.to_csv(output / "formal_target_change_panel.csv", index=False, encoding="utf-8-sig")
    transition.to_csv(output / "holding_transition_diagnostic.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([summary]).to_csv(output / "execution_gap_summary.csv", index=False, encoding="utf-8-sig")
    preplan.to_csv(output / "execution_variant_preplan.csv", index=False, encoding="utf-8-sig")
    event_study.to_csv(output / "execution_event_study_panel.csv", index=False, encoding="utf-8-sig")
    hold_cooldown.to_csv(output / "minimum_hold_cooldown_event_study.csv", index=False, encoding="utf-8-sig")
    partial_daily.to_csv(output / "partial_switch_simulator_daily.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([partial_summary]).to_csv(output / "partial_switch_summary.csv", index=False, encoding="utf-8-sig")
    sell_first_gap.to_csv(output / "sell_first_then_buy_gap_study.csv", index=False, encoding="utf-8-sig")
    pause_readiness.to_csv(output / "pause_on_conflict_input_readiness.csv", index=False, encoding="utf-8-sig")
    field_contract.to_csv(output / "execution_layer_field_contract.csv", index=False, encoding="utf-8-sig")
    (output / "final_summary_zh.md").write_text(_summary_markdown(summary, partial_summary, prices_loaded=bool(prices)), encoding="utf-8")

    manifest = {
        "schema_version": 2,
        "task_id": "TASK-BACKTEST-CORE-EXECUTION-LAYER-EVENT-STUDY-FIELDS-001",
        "model": "execution_layer_event_study_fields_diagnostic_only",
        "status": "completed",
        "formal_daily_path": str(formal_daily_path),
        "price_cache_dir": str(price_cache_dir or ""),
        "price_cache_loaded": bool(prices),
        "loaded_price_ticker_count": len(prices),
        "event_forward_coverage": event_forward_coverage,
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
            "execution_event_study_panel": "execution_event_study_panel.csv",
            "minimum_hold_cooldown_event_study": "minimum_hold_cooldown_event_study.csv",
            "partial_switch_simulator_daily": "partial_switch_simulator_daily.csv",
            "partial_switch_summary": "partial_switch_summary.csv",
            "sell_first_then_buy_gap_study": "sell_first_then_buy_gap_study.csv",
            "pause_on_conflict_input_readiness": "pause_on_conflict_input_readiness.csv",
            "execution_layer_field_contract": "execution_layer_field_contract.csv",
            "summary": "final_summary_zh.md",
            "run_log": "run_log.csv",
        },
        "limitations": [
            "partial_switch_simulator is a report-only cost/MDD proxy, not a formal portfolio execution ledger",
            "sell_first_then_buy gap study requires price cache for target forward returns",
            "pause_on_conflict is readiness/label only and does not pause formal trades",
        ],
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


def build_execution_event_study_panel(
    formal_daily: pd.DataFrame,
    target_change: pd.DataFrame,
    prices: dict[str, pd.Series],
) -> pd.DataFrame:
    frame = _normalized_formal_daily(formal_daily)
    if target_change.empty:
        return pd.DataFrame(columns=_event_study_columns())
    daily_by_date = {str(row["date"]): row for row in frame.to_dict(orient="records")}
    rows: list[dict[str, Any]] = []
    for item in target_change.to_dict(orient="records"):
        date = str(item.get("date", ""))
        target = _none_to_empty(item.get("new_target"))
        current = _none_to_empty(item.get("old_target"))
        daily_row = daily_by_date.get(date, {})
        row: dict[str, Any] = {
            "date": date,
            "old_target": item.get("old_target", "none"),
            "new_target": item.get("new_target", "none"),
            "target_change_type": item.get("target_change_type", ""),
            "reversal_within_3_trading_rows": bool(item.get("reversal_within_3_trading_rows", False)),
            "conflict_source_label": _conflict_source_label(item, daily_row),
            "diagnostic_only": True,
            "active_in_trade_decision": False,
        }
        for horizon in FORWARD_HORIZONS:
            target_return = _forward_return(prices.get(target), date, horizon)
            current_return = _forward_return(prices.get(current), date, horizon)
            row[f"target_change_forward_return_{horizon}d"] = _round_or_blank(target_return)
            row[f"current_holding_forward_return_{horizon}d"] = _round_or_blank(current_return)
            row[f"target_minus_current_holding_forward_return_{horizon}d"] = _round_or_blank(
                None if target_return is None or current_return is None else target_return - current_return
            )
            row[f"cash_wait_forward_return_vs_target_{horizon}d"] = _round_or_blank(
                None if target_return is None else -target_return
            )
        rows.append(row)
    return pd.DataFrame(rows, columns=_event_study_columns())


def build_minimum_hold_cooldown_event_study(event_study: pd.DataFrame) -> pd.DataFrame:
    if event_study.empty:
        return pd.DataFrame(
            columns=[
                "horizon_days",
                "event_count",
                "rows_with_forward_data",
                "short_reversal_count",
                "target_forward_return_avg",
                "current_holding_forward_return_avg",
                "target_minus_current_avg",
                "avoided_whipsaw_candidate_count",
                "missed_breakout_candidate_count",
                "diagnostic_only",
                "active_in_trade_decision",
            ]
        )
    rows: list[dict[str, Any]] = []
    reversal = event_study["reversal_within_3_trading_rows"].astype(bool)
    for horizon in FORWARD_HORIZONS:
        target_col = f"target_change_forward_return_{horizon}d"
        current_col = f"current_holding_forward_return_{horizon}d"
        spread_col = f"target_minus_current_holding_forward_return_{horizon}d"
        target = pd.to_numeric(event_study[target_col], errors="coerce")
        current = pd.to_numeric(event_study[current_col], errors="coerce")
        spread = pd.to_numeric(event_study[spread_col], errors="coerce")
        rows.append(
            {
                "horizon_days": horizon,
                "event_count": int(len(event_study)),
                "rows_with_forward_data": int(target.notna().sum()),
                "short_reversal_count": int(reversal.sum()),
                "target_forward_return_avg": _round_or_blank(float(target.mean()) if target.notna().any() else None),
                "current_holding_forward_return_avg": _round_or_blank(float(current.mean()) if current.notna().any() else None),
                "target_minus_current_avg": _round_or_blank(float(spread.mean()) if spread.notna().any() else None),
                "avoided_whipsaw_candidate_count": int(((reversal) & (spread < 0)).sum()),
                "missed_breakout_candidate_count": int(((reversal) & (spread > 0)).sum()),
                "diagnostic_only": True,
                "active_in_trade_decision": False,
            }
        )
    return pd.DataFrame(rows)


def build_partial_switch_simulator_daily(formal_daily: pd.DataFrame, *, partial_switch_weight: float = 0.5) -> pd.DataFrame:
    frame = _normalized_formal_daily(formal_daily)
    equity = pd.to_numeric(frame.get("equity", pd.Series(dtype=float)), errors="coerce").ffill().fillna(0.0)
    turnover = pd.to_numeric(frame.get("turnover", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    costs = pd.to_numeric(frame.get("transaction_cost", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    action = frame.get("action", pd.Series([""] * len(frame))).astype(str)
    switch_rows = action.isin(["buy", "switch"])
    partial_turnover = turnover.where(~switch_rows, turnover * partial_switch_weight)
    partial_cost = costs.where(~switch_rows, costs * partial_switch_weight)
    cumulative_cost_saved = (costs - partial_cost).cumsum()
    partial_equity = equity + cumulative_cost_saved
    baseline_running_max = equity.cummax().replace(0, pd.NA)
    partial_running_max = partial_equity.cummax().replace(0, pd.NA)
    rows = pd.DataFrame(
        {
            "date": frame["date"],
            "baseline_equity": equity.round(2),
            "partial_switch_equity_cost_proxy": partial_equity.round(2),
            "baseline_drawdown": (equity / baseline_running_max - 1).fillna(0.0).round(8),
            "partial_switch_drawdown_cost_proxy": (partial_equity / partial_running_max - 1).fillna(0.0).round(8),
            "baseline_turnover": turnover.round(2),
            "partial_switch_turnover": partial_turnover.round(2),
            "baseline_transaction_cost": costs.round(2),
            "partial_switch_transaction_cost": partial_cost.round(2),
            "cumulative_cost_saved": cumulative_cost_saved.round(2),
            "partial_switch_weight": partial_switch_weight,
            "simulator_scope": "cost_and_mdd_proxy_only",
            "diagnostic_only": True,
            "active_in_trade_decision": False,
        }
    )
    return rows


def build_partial_switch_summary(partial_daily: pd.DataFrame) -> dict[str, Any]:
    if partial_daily.empty:
        return {
            "status": "completed_empty",
            "diagnostic_only": True,
            "active_in_trade_decision": False,
            "simulator_scope": "cost_and_mdd_proxy_only",
        }
    return {
        "status": "completed",
        "diagnostic_only": True,
        "active_in_trade_decision": False,
        "simulator_scope": "cost_and_mdd_proxy_only",
        "partial_switch_weight": float(partial_daily["partial_switch_weight"].iloc[0]),
        "baseline_final_equity": round(float(partial_daily["baseline_equity"].iloc[-1]), 2),
        "partial_switch_final_equity_cost_proxy": round(float(partial_daily["partial_switch_equity_cost_proxy"].iloc[-1]), 2),
        "baseline_mdd": round(float(pd.to_numeric(partial_daily["baseline_drawdown"], errors="coerce").min()), 8),
        "partial_switch_mdd_cost_proxy": round(
            float(pd.to_numeric(partial_daily["partial_switch_drawdown_cost_proxy"], errors="coerce").min()), 8
        ),
        "baseline_total_turnover": round(float(pd.to_numeric(partial_daily["baseline_turnover"], errors="coerce").sum()), 2),
        "partial_switch_total_turnover": round(
            float(pd.to_numeric(partial_daily["partial_switch_turnover"], errors="coerce").sum()), 2
        ),
        "baseline_total_transaction_cost": round(
            float(pd.to_numeric(partial_daily["baseline_transaction_cost"], errors="coerce").sum()), 2
        ),
        "partial_switch_total_transaction_cost": round(
            float(pd.to_numeric(partial_daily["partial_switch_transaction_cost"], errors="coerce").sum()), 2
        ),
        "cost_saved": round(float(pd.to_numeric(partial_daily["cumulative_cost_saved"], errors="coerce").iloc[-1]), 2),
        "formal_model_changed": False,
        "trade_decision_changed": False,
    }


def build_sell_first_then_buy_gap_study(event_study: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for item in event_study.to_dict(orient="records"):
        for horizon in FORWARD_HORIZONS:
            gap = item.get(f"cash_wait_forward_return_vs_target_{horizon}d", "")
            rows.append(
                {
                    "date": item.get("date", ""),
                    "new_target": item.get("new_target", ""),
                    "horizon_days": horizon,
                    "cash_wait_forward_return_vs_target": gap,
                    "readiness_state": "ready" if str(gap) not in {"", "nan"} else "blocked_missing_forward_price",
                    "diagnostic_only": True,
                    "active_in_trade_decision": False,
                }
            )
    return pd.DataFrame(rows)


def build_pause_on_conflict_input_readiness(formal_daily: pd.DataFrame, event_study: pd.DataFrame) -> pd.DataFrame:
    frame = _normalized_formal_daily(formal_daily)
    labels = sorted({str(value) for value in event_study.get("conflict_source_label", pd.Series(dtype=str)).tolist() if str(value)})
    rows = [
        {
            "input_source": "formal_selector",
            "readiness_state": "ready",
            "available_columns": "consensus_state,winner_ticker,position_ticker,action",
            "observed_labels": ",".join(labels),
            "blocked_reason": "",
            "diagnostic_only": True,
            "active_in_trade_decision": False,
        },
        {
            "input_source": "market_exposure_conflict",
            "readiness_state": "ready",
            "available_columns": "winner_ticker",
            "observed_labels": ",".join(labels),
            "blocked_reason": "",
            "diagnostic_only": True,
            "active_in_trade_decision": False,
        },
        {
            "input_source": "risk_off",
            "readiness_state": "partial",
            "available_columns": "winner_ticker,position_ticker,consensus_state",
            "observed_labels": ",".join(labels),
            "blocked_reason": "risk_off is inferred only; no dedicated execution risk gate column in formal_daily",
            "diagnostic_only": True,
            "active_in_trade_decision": False,
        },
        {
            "input_source": "final_decision_diagnostic",
            "readiness_state": "blocked",
            "available_columns": ",".join([col for col in frame.columns if col.startswith("final_decision")]),
            "observed_labels": ",".join(labels),
            "blocked_reason": "final decision diagnostic fields are not present in this formal daily stream",
            "diagnostic_only": True,
            "active_in_trade_decision": False,
        },
        {
            "input_source": "pool3_selector_veto",
            "readiness_state": "blocked",
            "available_columns": ",".join([col for col in frame.columns if col.startswith("pool3_selector")]),
            "observed_labels": ",".join(labels),
            "blocked_reason": "Pool3 selector diagnostic fields are not present in this formal daily stream",
            "diagnostic_only": True,
            "active_in_trade_decision": False,
        },
    ]
    return pd.DataFrame(rows)


def build_execution_layer_field_contract(
    *,
    price_cache_dir: str | Path | None,
    prices_loaded: bool,
    event_forward_coverage: dict[str, Any] | None = None,
) -> pd.DataFrame:
    coverage = event_forward_coverage or {}
    forward_status = "blocked_missing_price_cache"
    if prices_loaded:
        forward_status = "ready" if float(coverage.get("target_5d_coverage_rate", 0.0)) >= 0.95 else "partial_forward_price_coverage"
    rows = [
        ("target_change_forward_return_5_20_60d", "price_cache", forward_status),
        ("current_holding_forward_return_5_20_60d", "price_cache", forward_status),
        ("cash_wait_forward_return_vs_target", "price_cache", forward_status),
        ("conflict_source_label", "formal_daily+target_change", "ready"),
        ("partial_position_equity_simulator", "formal_daily cost/turnover/equity", "ready_cost_proxy"),
        ("sell_first_then_buy_gap_study", "event_study_panel", "ready" if prices_loaded else "partial_without_forward_prices"),
        ("pause_on_conflict_input_readiness", "formal_daily columns", "partial"),
    ]
    return pd.DataFrame(
        [
            {
                "field_or_capability": name,
                "source": source,
                "readiness_state": status,
                "price_cache_dir": str(price_cache_dir or ""),
                "formal_model_changed": False,
                "trade_decision_changed": False,
                "active_in_trade_decision": False,
                "boundary": "report_only_diagnostic",
            }
            for name, source, status in rows
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


def _load_prices_for_execution_study(formal_daily: pd.DataFrame, cache_dir: Path) -> dict[str, pd.Series]:
    frame = _normalized_formal_daily(formal_daily)
    tickers = sorted(
        {
            str(value).strip()
            for column in ("winner_ticker", "position_ticker")
            if column in frame.columns
            for value in frame[column].tolist()
            if str(value).strip() and str(value).strip() not in {"cash", "none"}
        }
    )
    prices: dict[str, pd.Series] = {}
    for ticker in tickers:
        path = cache_dir / f"{ticker.replace('.', '_')}.csv"
        if not path.exists():
            continue
        frame_price = load_price_csv(path)
        close = pd.to_numeric(frame_price["adj_close"], errors="coerce").dropna()
        prices[ticker] = close
    return prices


def _forward_return(series: pd.Series | None, date: str, horizon: int) -> float | None:
    if series is None or series.empty:
        return None
    ts = pd.Timestamp(date)
    future = series.loc[series.index >= ts]
    if len(future) <= horizon:
        return None
    start = float(future.iloc[0])
    end = float(future.iloc[horizon])
    if start <= 0:
        return None
    return end / start - 1


def _round_or_blank(value: float | None) -> float | str:
    if value is None or pd.isna(value):
        return ""
    return round(float(value), 8)


def _none_to_empty(value: object) -> str:
    text = str(value or "").strip()
    if text in {"none", "cash"}:
        return ""
    return text


def _conflict_source_label(target_change_item: dict[str, Any], daily_row: dict[str, Any]) -> str:
    if bool(target_change_item.get("reversal_within_3_trading_rows", False)):
        return "rapid_reversal_conflict"
    target = _none_to_empty(target_change_item.get("new_target"))
    if _asset_role(target) in {"market_exposure", "leveraged_market_exposure"}:
        return "market_exposure_conflict"
    consensus_state = str(daily_row.get("consensus_state", "")).strip()
    if consensus_state in {"divergent", "no_vote", "insufficient_votes", "data_blocked"}:
        return "formal_selector_conflict"
    return "none"


def _event_study_columns() -> list[str]:
    columns = [
        "date",
        "old_target",
        "new_target",
        "target_change_type",
        "reversal_within_3_trading_rows",
        "conflict_source_label",
    ]
    for horizon in FORWARD_HORIZONS:
        columns.extend(
            [
                f"target_change_forward_return_{horizon}d",
                f"current_holding_forward_return_{horizon}d",
                f"target_minus_current_holding_forward_return_{horizon}d",
                f"cash_wait_forward_return_vs_target_{horizon}d",
            ]
        )
    columns.extend(["diagnostic_only", "active_in_trade_decision"])
    return columns


def _event_forward_data_coverage(event_study: pd.DataFrame) -> dict[str, Any]:
    total = int(len(event_study))
    stats: dict[str, Any] = {"event_count": total}
    for horizon in FORWARD_HORIZONS:
        target_col = f"target_change_forward_return_{horizon}d"
        current_col = f"current_holding_forward_return_{horizon}d"
        target_count = int(pd.to_numeric(event_study.get(target_col, pd.Series(dtype=float)), errors="coerce").notna().sum())
        current_count = int(pd.to_numeric(event_study.get(current_col, pd.Series(dtype=float)), errors="coerce").notna().sum())
        stats[f"target_{horizon}d_ready_rows"] = target_count
        stats[f"current_{horizon}d_ready_rows"] = current_count
        stats[f"target_{horizon}d_coverage_rate"] = round(target_count / total, 6) if total else 0.0
        stats[f"current_{horizon}d_coverage_rate"] = round(current_count / total, 6) if total else 0.0
    return stats


def _summary_markdown(
    summary: dict[str, Any],
    partial_summary: dict[str, Any] | None = None,
    *,
    prices_loaded: bool = False,
) -> str:
    partial_summary = partial_summary or {}
    return "\n".join(
        [
            "# Execution Layer Diagnostic-only Event Study Fields",
            "",
            "本輸出只用來檢查正式 baseline target stream 的執行層缺口，並補 Experiments 後續驗證所需欄位；不是正式 execution / exit layer。",
            "",
            "## 邊界",
            "",
            "- formal_model_changed=false",
            "- trade_decision_changed=false",
            "- active_in_trade_decision=false",
            "- execution_diagnostic_active_in_trade_decision=false",
            "- partial_switch_simulator 是成本與 MDD proxy，不是正式持倉帳本",
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
            f"- 價格快取載入：{str(prices_loaded).lower()}",
            f"- partial switch proxy 成本節省：{partial_summary.get('cost_saved', '')}",
            "",
            "## 下一步",
            "",
            "Experiments 可用本輸出等價檢查 minimum_hold、cooldown、partial_switch；sell_first_then_buy 與 pause_on_conflict 若欄位仍 partial/blocked，需先補資料再驗證。本任務不啟用任何正式規則。",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build report-only execution/exit layer diagnostic scaffold.")
    parser.add_argument("--formal-daily", required=True)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--price-cache-dir", default="")
    args = parser.parse_args()
    output = run_execution_layer_diagnostic(
        formal_daily_path=args.formal_daily,
        output_dir=args.output_dir,
        price_cache_dir=args.price_cache_dir or None,
    )
    print(f"OUTPUT_DIR={output.resolve()}")


if __name__ == "__main__":
    main()
