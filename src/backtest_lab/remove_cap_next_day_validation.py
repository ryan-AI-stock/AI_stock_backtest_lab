from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from backtest_lab.execution_layer_next_day_ab_pool1_pool2_formal import (
    INITIAL_CASH,
    VariantSpec as ExecutionVariantSpec,
    _simulate_variant as simulate_next_day_variant,
)
from backtest_lab.formal_model_contract import FORMAL_MODEL_ROUTE, FORMAL_MODEL_TARGET
from backtest_lab.pool1_pool2_veto_cap_downweight import (
    VariantSpec as DecisionVariantSpec,
    _build_target_weights,
    _load_prices,
    _needed_tickers,
    _simulate_weighted_variant,
)


DEFAULT_FORMAL_REPLAY_DIR = "outputs/stock_pool_formal_daily_replay_pit_pool2_daily_final_combined_20260624"
DEFAULT_PRICE_CACHE_DIR = "backtest_cache/stock_pool_triad_v1_corrected"
DEFAULT_OUTPUT_DIR = "outputs/remove_cap_next_day_apples_to_apples_validation_20260629"


DECISION_VARIANTS = (
    DecisionVariantSpec("cap40_confirmation1_same_day", "confirmation", cap_00631l=0.40, confirmation_days=1),
    DecisionVariantSpec("remove_cap_confirmation1_same_day", "confirmation", confirmation_days=1),
)


def run_remove_cap_next_day_validation(
    *,
    formal_replay_dir: str | Path = DEFAULT_FORMAL_REPLAY_DIR,
    price_cache_dir: str | Path = DEFAULT_PRICE_CACHE_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    initial_cash: float = INITIAL_CASH,
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
        replay = Path(formal_replay_dir)
        log("load_inputs", "started", str(replay))
        decision = pd.read_csv(replay / "formal_three_pool_decision_panel.csv").fillna("")
        _validate_decision(decision)
        prices = _load_prices(_needed_tickers(decision), Path(price_cache_dir))
        if not prices:
            raise ValueError("no prices loaded for remove-cap next-day validation")

        log("simulate_same_day", "started", "")
        same_day_daily: list[pd.DataFrame] = []
        same_day_trades: list[pd.DataFrame] = []
        same_day_events: list[pd.DataFrame] = []
        target_panels: dict[str, pd.DataFrame] = {}
        for spec in DECISION_VARIANTS:
            target_panel = _build_target_weights(decision, spec)
            target_panels[spec.variant] = target_panel
            daily, trades, events = _simulate_weighted_variant(target_panel, prices, spec, initial_cash)
            same_day_daily.append(_normalize_same_day_daily(daily, spec.variant))
            same_day_trades.append(_normalize_same_day_trades(trades, spec.variant))
            same_day_events.append(events)

        log("simulate_next_day", "started", "")
        next_day_daily: list[pd.DataFrame] = []
        next_day_trades: list[pd.DataFrame] = []
        next_day_blocked: list[pd.DataFrame] = []
        for source_variant, target_panel in target_panels.items():
            basis_variant = source_variant.replace("_same_day", "_next_day")
            frame = _target_panel_to_execution_frame(target_panel)
            daily, trades, _events, blocked = simulate_next_day_variant(
                frame,
                prices,
                ExecutionVariantSpec(basis_variant, 1, description="remove-cap apples-to-apples next-day validation"),
                initial_cash,
            )
            next_day_daily.append(_normalize_next_day_daily(daily, basis_variant))
            next_day_trades.append(_normalize_next_day_trades(trades, basis_variant))
            if not blocked.empty:
                next_day_blocked.append(blocked.assign(source_variant=source_variant))

        daily = pd.concat([*same_day_daily, *next_day_daily], ignore_index=True)
        trades = pd.concat([*same_day_trades, *next_day_trades], ignore_index=True)
        blocked = pd.concat(next_day_blocked, ignore_index=True) if next_day_blocked else pd.DataFrame()
        remove_cap_next_day = next_day_daily[1] if len(next_day_daily) > 1 else pd.DataFrame()
        remove_cap_trades = next_day_trades[1] if len(next_day_trades) > 1 else pd.DataFrame()

        log("build_reports", "started", "")
        perf = _period_performance(daily)
        summary = _summary_table(perf)
        risk = _risk_cost_turnover(daily, trades)
        worst_month = _worst_month_drawdown(daily)
        blockers = _data_blockers(decision, blocked)

        log("write_outputs", "started", "")
        summary.to_csv(output / "cap40_vs_remove_cap_same_day_next_day_summary.csv", index=False, encoding="utf-8-sig")
        remove_cap_next_day.to_csv(output / "remove_cap_next_day_equity_ledger.csv", index=False, encoding="utf-8-sig")
        remove_cap_trades.to_csv(output / "remove_cap_trade_ledger.csv", index=False, encoding="utf-8-sig")
        perf.to_csv(output / "period_performance_by_execution_basis.csv", index=False, encoding="utf-8-sig")
        risk.to_csv(output / "risk_cost_turnover_comparison.csv", index=False, encoding="utf-8-sig")
        worst_month.to_csv(output / "worst_month_drawdown_comparison.csv", index=False, encoding="utf-8-sig")
        (output / "data_blocker_report.md").write_text(_data_blocker_markdown(blockers), encoding="utf-8")
        (output / "report_wording_boundary_zh.md").write_text(_wording_boundary_markdown(), encoding="utf-8")
        (output / "final_summary_zh.md").write_text(_final_summary(summary, risk, blockers), encoding="utf-8")

        manifest = {
            "schema_version": 1,
            "task_id": "TASK-BACKTEST-CORE-REMOVE-CAP-NEXT-DAY-APPLES-TO-APPLES-VALIDATION-001",
            "status": "completed_validation_package",
            "user_requested_remove_cap_direction": True,
            "fully_validated_improvement": False,
            "formal_model_target": FORMAL_MODEL_TARGET,
            "formal_model_route": FORMAL_MODEL_ROUTE,
            "formal_target_stream_start": str(decision["date"].iloc[0]) if not decision.empty else "",
            "latest_formal_date": str(decision["date"].iloc[-1]) if not decision.empty else "",
            "same_day_result_not_used_as_next_day_proof": True,
            "formal_model_changed": True,
            "trade_decision_changed": True,
            "active_in_trade_decision": False,
            "production_grade_next_day_ledger": True,
            "simplified_experiments_ledger_used_for_formal_performance": False,
            "uses_forward_return_as_rule": False,
            "pool3_shadow_used": False,
            "rr_partial_switch_used": False,
            "valuation_used": False,
            "h3_used": False,
            "data_blocker_count": len(blockers),
            "outputs": {
                "summary": "cap40_vs_remove_cap_same_day_next_day_summary.csv",
                "remove_cap_next_day_equity_ledger": "remove_cap_next_day_equity_ledger.csv",
                "remove_cap_trade_ledger": "remove_cap_trade_ledger.csv",
                "period_performance": "period_performance_by_execution_basis.csv",
                "risk_cost_turnover": "risk_cost_turnover_comparison.csv",
                "worst_month_drawdown": "worst_month_drawdown_comparison.csv",
            },
        }
        (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        pd.DataFrame([{"status": "completed", "output_dir": str(output.resolve())}]).to_csv(output / "completed.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(columns=["step", "error"]).to_csv(output / "failed.csv", index=False, encoding="utf-8-sig")
        log("completed", "completed", str(output.resolve()))
        (output / "current_step.txt").write_text("completed\n", encoding="utf-8")
        return output
    except Exception as exc:
        pd.DataFrame([{"step": "run_remove_cap_next_day_validation", "error": str(exc)}]).to_csv(output / "failed.csv", index=False, encoding="utf-8-sig")
        log("failed", "failed", str(exc))
        raise


def _validate_decision(decision: pd.DataFrame) -> None:
    required = {"period", "date", "pool1_vote", "pool2_vote"}
    missing = sorted(required - set(decision.columns))
    if missing:
        raise ValueError(f"decision panel missing columns: {missing}")


def _target_panel_to_execution_frame(target_panel: pd.DataFrame) -> pd.DataFrame:
    frame = target_panel.copy().fillna("")
    normalized = frame["target_weights"].map(lambda value: json.dumps(json.loads(str(value or "{}")), sort_keys=True))
    changed = normalized.ne(normalized.shift(1))
    frame["action"] = changed.map(lambda value: "switch" if value else "hold")
    frame["turnover"] = frame["action"].map(lambda value: 1.0 if value == "switch" else 0.0)
    return frame


def _normalize_same_day_daily(daily: pd.DataFrame, variant: str) -> pd.DataFrame:
    out = daily.copy()
    out["execution_basis"] = "same_day"
    out["variant_id"] = variant
    out["portfolio_equity"] = pd.to_numeric(out["equity"], errors="coerce")
    out["transaction_cost"] = pd.to_numeric(out.get("transaction_cost", 0), errors="coerce").fillna(0.0)
    out["turnover"] = pd.to_numeric(out.get("turnover", 0), errors="coerce").fillna(0.0)
    out["active_in_trade_decision"] = False
    return out[["variant_id", "execution_basis", "date", "period", "target_weights", "position_ticker", "cash", "portfolio_equity", "drawdown", "turnover", "transaction_cost", "action", "active_in_trade_decision"]]


def _normalize_next_day_daily(daily: pd.DataFrame, variant: str) -> pd.DataFrame:
    out = daily.copy()
    out["execution_basis"] = "next_day"
    out["variant_id"] = variant
    out["target_weights"] = out.get("accepted_target_weights", "")
    out["position_ticker"] = out.get("top_holding", "")
    out["turnover"] = 0.0
    out["transaction_cost"] = 0.0
    out["action"] = out["pending_order_count"].map(lambda value: "pending_or_fill" if pd.to_numeric(pd.Series([value]), errors="coerce").fillna(0).iloc[0] else "hold")
    out["active_in_trade_decision"] = False
    return out[["variant_id", "execution_basis", "date", "period", "target_weights", "position_ticker", "cash", "portfolio_equity", "drawdown", "turnover", "transaction_cost", "action", "active_in_trade_decision"]]


def _normalize_same_day_trades(trades: pd.DataFrame, variant: str) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    out = trades.copy()
    out["variant_id"] = variant
    out["execution_basis"] = "same_day"
    out["transaction_cost"] = pd.to_numeric(out.get("costs", 0), errors="coerce").fillna(0.0)
    out["active_in_trade_decision"] = False
    return out


def _normalize_next_day_trades(trades: pd.DataFrame, variant: str) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    out = trades.copy()
    out["execution_basis"] = "next_day"
    out["variant_id"] = variant
    out["transaction_cost"] = pd.to_numeric(out.get("transaction_cost", 0), errors="coerce").fillna(0.0)
    out["active_in_trade_decision"] = False
    return out


def _period_performance(daily: pd.DataFrame) -> pd.DataFrame:
    periods = {
        "full": (None, None),
        "2022": ("2022-01-01", "2022-12-31"),
        "2023": ("2023-01-01", "2023-12-31"),
        "2024_now": ("2024-01-01", None),
        "2024_hard_gate": ("2024-01-01", "2024-12-31"),
    }
    frame = daily.copy()
    frame["date_ts"] = pd.to_datetime(frame["date"], errors="coerce")
    rows: list[dict[str, Any]] = []
    for (variant, basis), group in frame.groupby(["variant_id", "execution_basis"]):
        for period, (start, end) in periods.items():
            subset = group.copy()
            if start:
                subset = subset[subset["date_ts"] >= pd.Timestamp(start)]
            if end:
                subset = subset[subset["date_ts"] <= pd.Timestamp(end)]
            rows.append(_perf_row(variant, basis, period, subset))
    return pd.DataFrame(rows)


def _perf_row(variant: str, basis: str, period: str, frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {"variant_id": variant, "execution_basis": basis, "period_label": period, "status": "empty"}
    equity = pd.to_numeric(frame["portfolio_equity"], errors="coerce").dropna()
    if equity.empty:
        return {"variant_id": variant, "execution_basis": basis, "period_label": period, "status": "no_equity"}
    start = float(equity.iloc[0])
    end = float(equity.iloc[-1])
    return {
        "variant_id": variant,
        "execution_basis": basis,
        "period_label": period,
        "status": "completed",
        "start_date": str(frame["date"].iloc[0]),
        "end_date": str(frame["date"].iloc[-1]),
        "start_equity": round(start, 2),
        "final_equity": round(end, 2),
        "return_pct": round((end / start - 1) * 100, 4) if start else "",
        "max_drawdown_pct": round(float(pd.to_numeric(frame["drawdown"], errors="coerce").min()) * 100, 4),
        "trade_or_pending_days": int(frame["action"].astype(str).ne("hold").sum()),
        "diagnostic_only": True,
        "active_in_trade_decision": False,
    }


def _summary_table(perf: pd.DataFrame) -> pd.DataFrame:
    full = perf[(perf["period_label"] == "full") & (perf["status"] == "completed")].copy()
    return full.sort_values(["execution_basis", "variant_id"]).reset_index(drop=True)


def _risk_cost_turnover(daily: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    trade_costs = pd.DataFrame()
    if not trades.empty:
        trade_costs = trades.groupby(["variant_id", "execution_basis"], dropna=False)["transaction_cost"].sum().reset_index()
    rows: list[dict[str, Any]] = []
    for (variant, basis), group in daily.groupby(["variant_id", "execution_basis"]):
        cost = 0.0
        if not trade_costs.empty:
            matched = trade_costs[(trade_costs["variant_id"] == variant) & (trade_costs["execution_basis"] == basis)]
            cost = float(matched["transaction_cost"].iloc[0]) if not matched.empty else 0.0
        rows.append(
            {
                "variant_id": variant,
                "execution_basis": basis,
                "max_drawdown_pct": round(float(pd.to_numeric(group["drawdown"], errors="coerce").min()) * 100, 4),
                "trade_days_or_pending_days": int(group["action"].astype(str).ne("hold").sum()),
                "total_transaction_cost": round(cost, 2),
                "diagnostic_only": True,
                "active_in_trade_decision": False,
            }
        )
    return pd.DataFrame(rows)


def _worst_month_drawdown(daily: pd.DataFrame) -> pd.DataFrame:
    frame = daily.copy()
    frame["date_ts"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["month"] = frame["date_ts"].dt.strftime("%Y-%m")
    rows: list[dict[str, Any]] = []
    for (variant, basis, month), group in frame.groupby(["variant_id", "execution_basis", "month"]):
        equity = pd.to_numeric(group["portfolio_equity"], errors="coerce")
        peak = equity.cummax()
        month_dd = (equity / peak - 1).min()
        rows.append({"variant_id": variant, "execution_basis": basis, "month": month, "month_drawdown_pct": round(float(month_dd) * 100, 4)})
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return result.sort_values("month_drawdown_pct").groupby(["variant_id", "execution_basis"], as_index=False).first()


def _data_blockers(decision: pd.DataFrame, blocked: pd.DataFrame) -> list[dict[str, Any]]:
    blockers: list[dict[str, Any]] = []
    if decision.empty:
        blockers.append({"blocker": "formal_target_stream_empty", "severity": "blocking", "detail": "formal decision panel has no rows"})
    start = str(decision["date"].iloc[0]) if not decision.empty else ""
    latest = str(decision["date"].iloc[-1]) if not decision.empty else ""
    if start and start > "2022-01-03":
        blockers.append({"blocker": "formal_target_stream_start_after_expected", "severity": "warning", "detail": start})
    if latest and latest < "2026-06-12":
        blockers.append({"blocker": "latest_formal_date_before_expected", "severity": "warning", "detail": latest})
    if not blocked.empty:
        blockers.append({"blocker": "next_day_fill_blocked_events", "severity": "warning", "detail": str(len(blocked))})
    return blockers


def _data_blocker_markdown(blockers: list[dict[str, Any]]) -> str:
    lines = ["# Data Blocker Report", ""]
    if not blockers:
        lines.append("- No blocking data issue detected for this validation package.")
    for row in blockers:
        lines.append(f"- {row['severity']}: {row['blocker']} - {row['detail']}")
    lines.append("")
    lines.append("2014/2016-2021 do not have this formal target stream in this runner and are not backfilled as strategy performance.")
    return "\n".join(lines) + "\n"


def _wording_boundary_markdown() -> str:
    return (
        "# Remove-cap Report Wording Boundary\n\n"
        "- 使用者已指定移除 0050正二最高持倉 40% 上限。\n"
        "- 報告可描述為提高攻擊性、承擔較高波動與換手風險。\n"
        "- 不得把 same-day remove-cap 結果包裝成 next-day 可交易績效證明。\n"
        "- 不得寫成無條件更好；需同時揭露 MDD、成本、turnover 與 2024 hard gate caveat。\n"
    )


def _final_summary(summary: pd.DataFrame, risk: pd.DataFrame, blockers: list[dict[str, Any]]) -> str:
    lines = ["# Remove-cap Next-day Apples-to-apples Validation", "", "本輸出是驗收與風險標示，不改正式 selector。", ""]
    for row in summary.to_dict(orient="records"):
        lines.append(f"- {row['variant_id']} / {row['execution_basis']}: full return {row.get('return_pct')}%, MDD {row.get('max_drawdown_pct')}%")
    lines.extend(["", "## Risk / Cost", ""])
    for row in risk.to_dict(orient="records"):
        lines.append(f"- {row['variant_id']} / {row['execution_basis']}: cost {row.get('total_transaction_cost')}, trade/pending days {row.get('trade_days_or_pending_days')}")
    lines.extend(["", "## Boundary", "", "- same-day result is not used as next-day proof.", "- fully_validated_improvement=false until Experiments validates this package."])
    if blockers:
        lines.extend(["", "## Data blockers / warnings", ""])
        for row in blockers:
            lines.append(f"- {row['severity']}: {row['blocker']} - {row['detail']}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate remove-cap formal direction with same-day and next-day apples-to-apples ledgers.")
    parser.add_argument("--formal-replay-dir", default=DEFAULT_FORMAL_REPLAY_DIR)
    parser.add_argument("--price-cache-dir", default=DEFAULT_PRICE_CACHE_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    output = run_remove_cap_next_day_validation(
        formal_replay_dir=args.formal_replay_dir,
        price_cache_dir=args.price_cache_dir,
        output_dir=args.output_dir,
    )
    print(f"REMOVE_CAP_NEXT_DAY_VALIDATION_OUTPUT={output.resolve()}")


if __name__ == "__main__":
    main()
