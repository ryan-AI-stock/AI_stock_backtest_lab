from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from backtest_lab.data import load_price_csv
from backtest_lab.execution_layer_diagnostic import (
    build_execution_gap_summary,
    build_formal_target_change_panel,
    build_holding_transition_diagnostic,
)


DEFAULT_CANDIDATE_DIR = "outputs/pool1_pool2_veto_cap_downweight_panels_20260626"
DEFAULT_ABSORPTION_DIR = "outputs/formal_absorb_pool1_pool2_combined_cap40_confirmation1_20260626"
DEFAULT_PRICE_CACHE_DIR = "backtest_cache/stock_pool_triad_v1_corrected"
DEFAULT_OUTPUT_DIR = "outputs/execution_layer_review_pool1_pool2_formal_20260626"
PANEL_VARIANT_ID = "combined_cap40_confirmation1"
FORMAL_MODEL_TARGET = "combined_cap40_confirmation1_base"


def run_execution_layer_review_pool1_pool2_formal(
    *,
    candidate_dir: str | Path = DEFAULT_CANDIDATE_DIR,
    absorption_dir: str | Path = DEFAULT_ABSORPTION_DIR,
    price_cache_dir: str | Path = DEFAULT_PRICE_CACHE_DIR,
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
        candidate_root = Path(candidate_dir)
        absorption_root = Path(absorption_dir)
        price_root = Path(price_cache_dir)

        log("load_inputs", "started", str(candidate_root))
        daily = pd.read_csv(candidate_root / "daily_equity_by_variant.csv").fillna("")
        events = pd.read_csv(candidate_root / "pool2_disagreement_variant_events.csv").fillna("")
        trades = pd.read_csv(candidate_root / "trade_ledger_by_variant.csv").fillna("")
        absorption_manifest = _load_json(absorption_root / "manifest.json")
        _validate_absorbed_manifest(absorption_manifest)

        log("adapt_formal_target_stream", "started", PANEL_VARIANT_ID)
        adapted = build_formal_target_stream_adapter(daily=daily, events=events)

        log("build_execution_diagnostics", "started", "")
        target_change = build_formal_target_change_panel(adapted)
        transition = build_holding_transition_diagnostic(adapted)
        gap_summary = build_execution_gap_summary(adapted, target_change, transition)
        gap_summary.update(
            {
                "formal_model_target": FORMAL_MODEL_TARGET,
                "formal_model_route": "pool1_primary_pool2_confirmation_cap40",
                "review_scope": "execution_layer_after_formal_absorption",
            }
        )

        log("build_next_day_readiness", "started", str(price_root))
        next_day = build_next_day_fill_readiness(adapted, price_root)
        next_day_preview = build_same_day_vs_next_day_preview(next_day)
        stability = build_entry_target_stability_summary(adapted, target_change, trades)
        partial_readiness = build_partial_execution_readiness(adapted, next_day)

        log("write_outputs", "started", "")
        adapted.to_csv(output / "formal_target_stream_adapter.csv", index=False, encoding="utf-8-sig")
        target_change.to_csv(output / "formal_target_change_panel.csv", index=False, encoding="utf-8-sig")
        transition.to_csv(output / "holding_transition_diagnostic.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame([gap_summary]).to_csv(output / "execution_gap_summary.csv", index=False, encoding="utf-8-sig")
        next_day.to_csv(output / "next_day_fill_readiness.csv", index=False, encoding="utf-8-sig")
        next_day_preview.to_csv(output / "same_day_vs_next_day_preview.csv", index=False, encoding="utf-8-sig")
        stability.to_csv(output / "entry_target_stability_summary.csv", index=False, encoding="utf-8-sig")
        partial_readiness.to_csv(output / "partial_execution_readiness.csv", index=False, encoding="utf-8-sig")
        (output / "execution_layer_review_summary_zh.md").write_text(
            _summary_markdown(gap_summary, stability, partial_readiness, next_day_preview),
            encoding="utf-8",
        )

        manifest = {
            "schema_version": 1,
            "task_id": "TASK-BACKTEST-CORE-EXECUTION-LAYER-REVIEW-ON-POOL1-POOL2-FORMAL-001",
            "status": "completed_diagnostic_review",
            "formal_model_target": FORMAL_MODEL_TARGET,
            "formal_model_route": "pool1_primary_pool2_confirmation_cap40",
            "source_panel_variant": PANEL_VARIANT_ID,
            "candidate_dir": str(candidate_root),
            "absorption_dir": str(absorption_root),
            "price_cache_dir": str(price_root),
            "start_date": str(adapted["date"].iloc[0]) if not adapted.empty else "",
            "latest_complete_common_date": str(adapted["date"].iloc[-1]) if not adapted.empty else "",
            "row_count": int(len(adapted)),
            "formal_model_changed": False,
            "trade_decision_changed": False,
            "active_in_trade_decision": False,
            "execution_review_active_in_trade_decision": False,
            "pool3_shadow_used": False,
            "final_decision_label_used": False,
            "rr_partial_switch_activated": False,
            "uses_forward_return_as_rule": False,
            "valuation_used": False,
            "h3_used": False,
            "next_day_ledger_mixed_with_same_day": False,
            "output_files": {
                "formal_target_stream_adapter": "formal_target_stream_adapter.csv",
                "target_change": "formal_target_change_panel.csv",
                "transition": "holding_transition_diagnostic.csv",
                "gap_summary": "execution_gap_summary.csv",
                "next_day_readiness": "next_day_fill_readiness.csv",
                "stability": "entry_target_stability_summary.csv",
                "partial_readiness": "partial_execution_readiness.csv",
                "summary": "execution_layer_review_summary_zh.md",
            },
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
        pd.DataFrame([{"step": "run_execution_layer_review_pool1_pool2_formal", "error": str(exc)}]).to_csv(
            output / "failed.csv", index=False, encoding="utf-8-sig"
        )
        log("failed", "failed", str(exc))
        raise


def build_formal_target_stream_adapter(*, daily: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    daily_variant = daily[daily["variant"].astype(str) == PANEL_VARIANT_ID].copy()
    events_variant = events[events["variant"].astype(str) == PANEL_VARIANT_ID].copy()
    if daily_variant.empty:
        raise ValueError(f"missing daily rows for variant: {PANEL_VARIANT_ID}")
    if events_variant.empty:
        raise ValueError(f"missing event rows for variant: {PANEL_VARIANT_ID}")

    merged = daily_variant.merge(
        events_variant[
            [
                "date",
                "pool1_vote",
                "pool2_vote",
                "pool2_disagreement",
                "event_reason",
                "uses_forward_return_as_rule",
            ]
        ],
        on="date",
        how="left",
    ).fillna("")
    rows: list[dict[str, Any]] = []
    previous_winner = ""
    for item in merged.to_dict(orient="records"):
        weights = _parse_target_weights(item.get("target_weights"))
        winner = _primary_target_from_weights(weights)
        source_action = str(item.get("action", "")).strip() or "hold"
        action = _execution_action(source_action, previous_winner, winner, _number(item.get("turnover")))
        row = {
            "date": str(item.get("date", "")),
            "period": str(item.get("period", "")),
            "formal_model_target": FORMAL_MODEL_TARGET,
            "source_panel_variant": PANEL_VARIANT_ID,
            "winner_ticker": winner,
            "formal_target": winner,
            "target_weights": json.dumps(weights, ensure_ascii=False, sort_keys=True),
            "target_weight_count": len(weights),
            "position_ticker": str(item.get("position_ticker", "")).strip() or "cash",
            "cash": _number(item.get("cash")),
            "equity": _number(item.get("equity")),
            "drawdown": _number(item.get("drawdown")),
            "turnover": _number(item.get("turnover")),
            "transaction_cost": _number(item.get("transaction_cost")),
            "source_action": source_action,
            "action": action,
            "pool1_vote": str(item.get("pool1_vote", "")).strip(),
            "pool2_vote": str(item.get("pool2_vote", "")).strip(),
            "pool2_disagreement": _bool(item.get("pool2_disagreement")),
            "event_reason": str(item.get("event_reason", "")).strip(),
            "uses_forward_return_as_rule": _bool(item.get("uses_forward_return_as_rule")),
            "pool3_shadow_used": False,
            "final_decision_label_used": False,
            "rr_partial_switch_activated": False,
            "execution_review_active_in_trade_decision": False,
        }
        rows.append(row)
        previous_winner = winner
    frame = pd.DataFrame(rows)
    frame["date_ts"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame[frame["date_ts"].notna()].sort_values("date_ts").drop(columns=["date_ts"]).reset_index(drop=True)
    return frame


def build_next_day_fill_readiness(formal_daily: pd.DataFrame, price_cache_dir: str | Path) -> pd.DataFrame:
    prices = _load_price_series(formal_daily, Path(price_cache_dir))
    rows: list[dict[str, Any]] = []
    for item in formal_daily.to_dict(orient="records"):
        action = str(item.get("action", "hold"))
        ticker = str(item.get("winner_ticker", "")).strip()
        if action not in {"buy", "switch", "rebalance"} and _number(item.get("turnover")) <= 0:
            continue
        status = "blocked_no_target"
        same_day_price = ""
        next_day = ""
        next_day_price = ""
        slippage = ""
        if ticker and ticker not in {"cash", "none"}:
            series = prices.get(ticker)
            if series is None or series.empty:
                status = "blocked_missing_price"
            else:
                current, next_item = _same_and_next_price(series, str(item.get("date", "")))
                if current is None:
                    status = "blocked_missing_same_day_price"
                elif next_item is None:
                    status = "blocked_missing_next_day_price"
                    same_day_price = round(float(current[1]), 6)
                else:
                    status = "completed"
                    same_day_price = round(float(current[1]), 6)
                    next_day = next_item[0].strftime("%Y-%m-%d")
                    next_day_price = round(float(next_item[1]), 6)
                    slippage = round(float(next_item[1]) / float(current[1]) - 1, 8) if float(current[1]) else ""
        rows.append(
            {
                "date": item.get("date", ""),
                "formal_target": ticker or "none",
                "trade_action": action,
                "turnover": _number(item.get("turnover")),
                "same_day_price": same_day_price,
                "next_trade_date": next_day,
                "next_day_price": next_day_price,
                "next_day_slippage_pct": slippage,
                "readiness_state": status,
                "next_day_ledger_mixed_with_same_day": False,
                "diagnostic_only": True,
                "active_in_trade_decision": False,
            }
        )
    return pd.DataFrame(rows)


def build_same_day_vs_next_day_preview(next_day: pd.DataFrame) -> pd.DataFrame:
    if next_day.empty:
        return pd.DataFrame(
            [
                {
                    "completed_events": 0,
                    "blocked_events": 0,
                    "average_next_day_slippage_pct": "",
                    "max_next_day_slippage_pct": "",
                    "min_next_day_slippage_pct": "",
                    "next_day_ledger_mixed_with_same_day": False,
                    "diagnostic_only": True,
                    "active_in_trade_decision": False,
                }
            ]
        )
    completed = next_day[next_day["readiness_state"] == "completed"].copy()
    slippage = pd.to_numeric(completed.get("next_day_slippage_pct"), errors="coerce").dropna()
    return pd.DataFrame(
        [
            {
                "completed_events": int(len(completed)),
                "blocked_events": int(len(next_day) - len(completed)),
                "average_next_day_slippage_pct": round(float(slippage.mean()), 8) if not slippage.empty else "",
                "max_next_day_slippage_pct": round(float(slippage.max()), 8) if not slippage.empty else "",
                "min_next_day_slippage_pct": round(float(slippage.min()), 8) if not slippage.empty else "",
                "next_day_ledger_mixed_with_same_day": False,
                "diagnostic_only": True,
                "active_in_trade_decision": False,
            }
        ]
    )


def build_entry_target_stability_summary(
    formal_daily: pd.DataFrame,
    target_change: pd.DataFrame,
    trade_ledger: pd.DataFrame,
) -> pd.DataFrame:
    frame = formal_daily.copy()
    target_change = target_change.copy()
    action_rows = frame[frame["action"].astype(str).isin(["buy", "switch", "rebalance"]) | (pd.to_numeric(frame["turnover"], errors="coerce").fillna(0.0) > 0)]
    pool2_disagree_entries = action_rows[action_rows["pool2_disagreement"].astype(bool)]
    confirmation_block_rows = frame[frame["event_reason"].astype(str).str.contains("confirmation_1_not_met", na=False)]
    change_days = pd.to_numeric(target_change.get("days_since_previous_change"), errors="coerce")
    target_changed_1d = int((change_days <= 1).sum()) if not change_days.empty else 0
    target_changed_3d = int((change_days <= 3).sum()) if not change_days.empty else 0
    rapid_flip = int(target_change.get("reversal_within_3_trading_rows", pd.Series(dtype=bool)).astype(bool).sum()) if not target_change.empty else 0
    trade_variant = trade_ledger[trade_ledger["variant"].astype(str) == PANEL_VARIANT_ID].copy() if not trade_ledger.empty else pd.DataFrame()
    return pd.DataFrame(
        [
            {
                "formal_model_target": FORMAL_MODEL_TARGET,
                "row_count": int(len(frame)),
                "trade_or_rebalance_rows": int(len(action_rows)),
                "trade_ledger_rows": int(len(trade_variant)),
                "pool2_disagreement_entry_count": int(len(pool2_disagree_entries)),
                "pool2_disagreement_entry_rate": round(len(pool2_disagree_entries) / len(action_rows), 6) if len(action_rows) else 0.0,
                "confirmation1_blocked_rows": int(len(confirmation_block_rows)),
                "target_change_count": int(len(target_change)),
                "target_changed_within_1d_count": target_changed_1d,
                "target_changed_within_3d_count": target_changed_3d,
                "rapid_flip_same_target_window_1_3d_count": rapid_flip,
                "target_changed_within_1d_rate": round(target_changed_1d / len(target_change), 6) if len(target_change) else 0.0,
                "target_changed_within_3d_rate": round(target_changed_3d / len(target_change), 6) if len(target_change) else 0.0,
                "rapid_flip_rate": round(rapid_flip / len(target_change), 6) if len(target_change) else 0.0,
                "possible_execution_layer_issue": bool(rapid_flip or target_changed_3d),
                "diagnostic_only": True,
                "active_in_trade_decision": False,
            }
        ]
    )


def build_partial_execution_readiness(formal_daily: pd.DataFrame, next_day: pd.DataFrame) -> pd.DataFrame:
    completed_next_day = int((next_day.get("readiness_state", pd.Series(dtype=str)).astype(str) == "completed").sum()) if not next_day.empty else 0
    blocked_next_day = int(len(next_day) - completed_next_day) if not next_day.empty else 0
    return pd.DataFrame(
        [
            {
                "candidate": "partial_switch_or_minimum_hold_execution_layer",
                "formal_target_stream": FORMAL_MODEL_TARGET,
                "stream_rows": int(len(formal_daily)),
                "next_day_completed_events": completed_next_day,
                "next_day_blocked_events": blocked_next_day,
                "ready_for_experiments_validation": completed_next_day > 0,
                "ready_for_formal_activation": False,
                "blocked_reason": "" if completed_next_day > 0 else "no_completed_next_day_events",
                "rr_partial_switch_activated": False,
                "uses_forward_return_as_rule": False,
                "diagnostic_only": True,
                "active_in_trade_decision": False,
            }
        ]
    )


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_absorbed_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("formal_model_target") != FORMAL_MODEL_TARGET:
        raise ValueError(f"formal absorption manifest target mismatch: {manifest.get('formal_model_target')}")
    if manifest.get("formal_absorption_ready") is not True:
        raise ValueError("formal absorption manifest is not ready")


def _parse_target_weights(value: object) -> dict[str, float]:
    text = str(value or "").strip()
    if not text:
        return {}
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        return {}
    return {str(k).strip(): float(v) for k, v in parsed.items() if str(k).strip() and float(v) > 0}


def _primary_target_from_weights(weights: dict[str, float]) -> str:
    if not weights:
        return ""
    return sorted(weights.items(), key=lambda item: (-float(item[1]), item[0]))[0][0]


def _execution_action(source_action: str, previous_target: str, current_target: str, turnover: float) -> str:
    if source_action in {"buy", "switch", "hold"}:
        return source_action
    if turnover <= 0:
        return "hold"
    previous = str(previous_target or "").strip()
    current = str(current_target or "").strip()
    if current and current != previous:
        return "buy" if not previous else "switch"
    if not current and previous:
        return "switch"
    return "switch"


def _load_price_series(formal_daily: pd.DataFrame, price_cache_dir: Path) -> dict[str, pd.Series]:
    tickers = sorted({str(t).strip() for t in formal_daily["winner_ticker"].tolist() if str(t).strip() and str(t).strip() not in {"cash", "none"}})
    prices: dict[str, pd.Series] = {}
    for ticker in tickers:
        path = price_cache_dir / f"{ticker.replace('.', '_')}.csv"
        if not path.exists():
            continue
        frame = load_price_csv(path)
        prices[ticker] = pd.to_numeric(frame["adj_close"], errors="coerce").dropna()
    return prices


def _same_and_next_price(series: pd.Series, date: str) -> tuple[tuple[pd.Timestamp, float] | None, tuple[pd.Timestamp, float] | None]:
    ts = pd.Timestamp(date).normalize()
    after_or_equal = series.loc[series.index >= ts]
    if after_or_equal.empty:
        return None, None
    current = (pd.Timestamp(after_or_equal.index[0]), float(after_or_equal.iloc[0]))
    after = series.loc[series.index > current[0]]
    next_item = None if after.empty else (pd.Timestamp(after.index[0]), float(after.iloc[0]))
    return current, next_item


def _number(value: object) -> float:
    return float(pd.to_numeric(pd.Series([value]), errors="coerce").fillna(0.0).iloc[0])


def _bool(value: object) -> bool:
    text = str(value).strip().lower()
    return text in {"true", "1", "yes", "y"}


def _summary_markdown(
    gap_summary: dict[str, Any],
    stability: pd.DataFrame,
    partial_readiness: pd.DataFrame,
    next_day_preview: pd.DataFrame,
) -> str:
    stable = stability.iloc[0].to_dict() if not stability.empty else {}
    readiness = partial_readiness.iloc[0].to_dict() if not partial_readiness.empty else {}
    next_day = next_day_preview.iloc[0].to_dict() if not next_day_preview.empty else {}
    return "\n".join(
        [
            "# Pool1+Pool2 正式 target stream 換倉執行層診斷",
            "",
            "本輸出基於已正式吸收的 `combined_cap40_confirmation1_base` target stream。這是 execution layer report-only diagnostic，不改正式模型、不改正式 target、不啟用分批換倉。",
            "",
            "## 主要數字",
            f"- 日期範圍：{gap_summary.get('start_date', '')} 到 {gap_summary.get('end_date', '')}",
            f"- target change count：{gap_summary.get('target_change_count', 0)}",
            f"- short reversal count：{gap_summary.get('short_reversal_count', 0)}",
            f"- rapid flip rate：{gap_summary.get('rapid_flip_rate', 0)}",
            f"- total transaction cost：{gap_summary.get('total_transaction_cost', 0)}",
            f"- Pool2 disagreement entry rate：{stable.get('pool2_disagreement_entry_rate', 0)}",
            f"- next-day completed / blocked：{next_day.get('completed_events', 0)} / {next_day.get('blocked_events', 0)}",
            "",
            "## 邊界",
            "- formal_model_changed=false",
            "- trade_decision_changed=false",
            "- active_in_trade_decision=false",
            "- RR partial switch 未啟用，forward return 未作規則。",
            "",
            "## 下一步",
            f"- Experiments 可讀取本輸出驗證換倉執行層是否需進一步做正式 A/B；目前 ready_for_formal_activation={readiness.get('ready_for_formal_activation', False)}。",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Review execution layer on absorbed Pool1+Pool2 formal target stream.")
    parser.add_argument("--candidate-dir", default=DEFAULT_CANDIDATE_DIR)
    parser.add_argument("--absorption-dir", default=DEFAULT_ABSORPTION_DIR)
    parser.add_argument("--price-cache-dir", default=DEFAULT_PRICE_CACHE_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    run_execution_layer_review_pool1_pool2_formal(
        candidate_dir=args.candidate_dir,
        absorption_dir=args.absorption_dir,
        price_cache_dir=args.price_cache_dir,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
