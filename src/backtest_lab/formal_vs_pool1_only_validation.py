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
DEFAULT_OUTPUT_DIR = "outputs/formal_vs_pool1_only_apples_to_apples_20260630"
MAIN_START_DATE = "2024-01-01"


VARIANTS = (
    DecisionVariantSpec(
        "current_formal_pool1_pool2_remove_cap",
        "confirmation",
        confirmation_days=1,
    ),
    DecisionVariantSpec(
        "new_pool1_only_no_overlay",
        "warning_only",
    ),
)


def run_formal_vs_pool1_only_validation(
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
            raise ValueError("no prices loaded")

        log("simulate_same_day", "started", "")
        same_day_daily: list[pd.DataFrame] = []
        same_day_trades: list[pd.DataFrame] = []
        target_panels: dict[str, pd.DataFrame] = {}
        for spec in VARIANTS:
            target_panel = _build_target_weights(decision, spec)
            target_panels[spec.variant] = target_panel
            daily, trades, _events = _simulate_weighted_variant(target_panel, prices, spec, initial_cash)
            same_day_daily.append(_normalize_same_day_daily(daily, spec.variant))
            same_day_trades.append(_normalize_same_day_trades(trades, spec.variant))

        log("simulate_next_day", "started", "")
        next_day_daily: list[pd.DataFrame] = []
        next_day_trades: list[pd.DataFrame] = []
        blocked_frames: list[pd.DataFrame] = []
        for variant_id, target_panel in target_panels.items():
            frame = _target_panel_to_execution_frame(target_panel)
            daily, trades, _events, blocked = simulate_next_day_variant(
                frame,
                prices,
                ExecutionVariantSpec(f"{variant_id}_next_day", 1, description="formal vs pool1-only next-day validation"),
                initial_cash,
            )
            next_day_daily.append(_normalize_next_day_daily(daily, variant_id))
            next_day_trades.append(_normalize_next_day_trades(trades, variant_id))
            if not blocked.empty:
                blocked_frames.append(blocked.assign(source_variant=variant_id))

        daily_ledger = pd.concat([*same_day_daily, *next_day_daily], ignore_index=True)
        trade_ledger = pd.concat([*same_day_trades, *next_day_trades], ignore_index=True) if same_day_trades or next_day_trades else pd.DataFrame()
        blocked = pd.concat(blocked_frames, ignore_index=True) if blocked_frames else pd.DataFrame()

        log("build_reports", "started", "")
        monthly = _monthly_performance(daily_ledger)
        worst_month = _worst_month(monthly)
        costs = _trade_cost_summary(trade_ledger)
        performance = _attach_trade_costs(_period_performance(daily_ledger), costs)
        pool2_effect = _pool2_effect_decision(performance, costs)
        identities = _variant_identity()

        log("write_outputs", "started", "")
        identities.to_csv(output / "variant_identity.csv", index=False, encoding="utf-8-sig")
        daily_ledger.to_csv(output / "daily_equity_by_variant.csv", index=False, encoding="utf-8-sig")
        trade_ledger.to_csv(output / "trade_ledger_by_variant.csv", index=False, encoding="utf-8-sig")
        blocked.to_csv(output / "blocked_fill_events.csv", index=False, encoding="utf-8-sig")
        performance.to_csv(output / "performance_comparison.csv", index=False, encoding="utf-8-sig")
        monthly.to_csv(output / "monthly_performance.csv", index=False, encoding="utf-8-sig")
        worst_month.to_csv(output / "worst_month.csv", index=False, encoding="utf-8-sig")
        costs.to_csv(output / "trade_cost_summary.csv", index=False, encoding="utf-8-sig")
        pool2_effect.to_csv(output / "pool2_effect_summary.csv", index=False, encoding="utf-8-sig")
        (output / "formal_vs_pool1_only_summary_zh.md").write_text(
            _summary_markdown(performance, worst_month, costs, pool2_effect),
            encoding="utf-8",
        )
        manifest = {
            "schema_version": 1,
            "task_id": "TASK-BACKTEST-CORE-FORMAL-DAILY-VS-POOL1-ONLY-APPLES-TO-APPLES-20260630",
            "status": "completed_validation_package",
            "formal_model_target": FORMAL_MODEL_TARGET,
            "formal_model_route": FORMAL_MODEL_ROUTE,
            "pool1_only_model_id": "new_pool1_only_no_overlay",
            "pool1_only_selector": "pool1_primary_no_pool2_confirmation",
            "formal_replay_dir": str(replay),
            "price_cache_dir": str(price_cache_dir),
            "data_start": str(decision["date"].iloc[0]),
            "latest_complete_common_date": str(decision["date"].iloc[-1]),
            "main_comparison_start": MAIN_START_DATE,
            "same_date_range_for_variants": _same_date_range(daily_ledger),
            "same_cost_model_for_variants": True,
            "same_execution_basis_compared_separately": True,
            "same_day_result_not_used_as_next_day_proof": True,
            "formal_model_changed": False,
            "trade_decision_changed": False,
            "active_in_trade_decision": False,
            "pool3_shadow_used": False,
            "rr_partial_switch_used": False,
            "uses_forward_return_as_rule": False,
            "outputs": {
                "performance": "performance_comparison.csv",
                "monthly": "monthly_performance.csv",
                "worst_month": "worst_month.csv",
                "costs": "trade_cost_summary.csv",
                "pool2_effect": "pool2_effect_summary.csv",
                "summary": "formal_vs_pool1_only_summary_zh.md",
            },
        }
        (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        pd.DataFrame([{"status": "completed", "output_dir": str(output.resolve())}]).to_csv(output / "completed.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(columns=["step", "error"]).to_csv(output / "failed.csv", index=False, encoding="utf-8-sig")
        log("completed", "completed", str(output.resolve()))
        (output / "current_step.txt").write_text("completed\n", encoding="utf-8")
        return output
    except Exception as exc:
        pd.DataFrame([{"step": "run_formal_vs_pool1_only_validation", "error": str(exc)}]).to_csv(output / "failed.csv", index=False, encoding="utf-8-sig")
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


def _normalize_same_day_daily(daily: pd.DataFrame, variant_id: str) -> pd.DataFrame:
    out = daily.copy()
    out["variant_id"] = variant_id
    out["execution_basis"] = "same_day"
    out["portfolio_equity"] = pd.to_numeric(out["equity"], errors="coerce")
    out["transaction_cost"] = pd.to_numeric(out.get("transaction_cost", 0), errors="coerce").fillna(0.0)
    out["turnover"] = pd.to_numeric(out.get("turnover", 0), errors="coerce").fillna(0.0)
    out["top_holding"] = out.get("position_ticker", "")
    out["active_in_trade_decision"] = False
    return out[
        [
            "variant_id",
            "execution_basis",
            "date",
            "period",
            "target_weights",
            "top_holding",
            "cash",
            "portfolio_equity",
            "drawdown",
            "turnover",
            "transaction_cost",
            "action",
            "active_in_trade_decision",
        ]
    ]


def _normalize_next_day_daily(daily: pd.DataFrame, variant_id: str) -> pd.DataFrame:
    out = daily.copy()
    out["variant_id"] = variant_id
    out["execution_basis"] = "next_day"
    out["target_weights"] = out.get("accepted_target_weights", "")
    out["turnover"] = 0.0
    out["transaction_cost"] = 0.0
    out["action"] = out["pending_order_count"].map(lambda value: "pending_or_fill" if pd.to_numeric(pd.Series([value]), errors="coerce").fillna(0).iloc[0] else "hold")
    out["active_in_trade_decision"] = False
    return out[
        [
            "variant_id",
            "execution_basis",
            "date",
            "period",
            "target_weights",
            "top_holding",
            "cash",
            "portfolio_equity",
            "drawdown",
            "turnover",
            "transaction_cost",
            "action",
            "active_in_trade_decision",
        ]
    ]


def _normalize_same_day_trades(trades: pd.DataFrame, variant_id: str) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    out = trades.copy()
    out["variant_id"] = variant_id
    out["execution_basis"] = "same_day"
    out["transaction_cost"] = pd.to_numeric(out.get("costs", 0), errors="coerce").fillna(0.0)
    out["active_in_trade_decision"] = False
    return out


def _normalize_next_day_trades(trades: pd.DataFrame, variant_id: str) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    out = trades.copy()
    out["variant_id"] = variant_id
    out["execution_basis"] = "next_day"
    out["transaction_cost"] = pd.to_numeric(out.get("transaction_cost", 0), errors="coerce").fillna(0.0)
    out["active_in_trade_decision"] = False
    return out


def _period_performance(daily: pd.DataFrame) -> pd.DataFrame:
    periods = {
        "2024_now_main": (MAIN_START_DATE, None),
        "2024_hard_gate": ("2024-01-01", "2024-12-31"),
        "2025": ("2025-01-01", "2025-12-31"),
        "2026_ytd": ("2026-01-01", None),
        "full_available": (None, None),
    }
    frame = daily.copy()
    frame["date_ts"] = pd.to_datetime(frame["date"], errors="coerce")
    rows: list[dict[str, Any]] = []
    for (variant, basis), group in frame.groupby(["variant_id", "execution_basis"], dropna=False):
        group = group.sort_values("date_ts")
        for label, (start, end) in periods.items():
            subset = group.copy()
            if start:
                subset = subset[subset["date_ts"] >= pd.Timestamp(start)]
            if end:
                subset = subset[subset["date_ts"] <= pd.Timestamp(end)]
            rows.append(_perf_row(str(variant), str(basis), label, subset))
    return pd.DataFrame(rows)


def _perf_row(variant: str, basis: str, label: str, frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {"variant_id": variant, "execution_basis": basis, "period_label": label, "status": "empty"}
    equity = pd.to_numeric(frame["portfolio_equity"], errors="coerce").dropna()
    if equity.empty:
        return {"variant_id": variant, "execution_basis": basis, "period_label": label, "status": "no_equity"}
    start = float(equity.iloc[0])
    end = float(equity.iloc[-1])
    start_date = str(frame["date"].iloc[0])
    end_date = str(frame["date"].iloc[-1])
    years = max((pd.Timestamp(end_date) - pd.Timestamp(start_date)).days / 365.25, 1 / 365.25)
    ret = end / start - 1 if start else 0.0
    dd = _period_drawdown(equity)
    return {
        "variant_id": variant,
        "execution_basis": basis,
        "period_label": label,
        "status": "completed",
        "start_date": start_date,
        "end_date": end_date,
        "start_equity": round(start, 2),
        "final_equity": round(end, 2),
        "return_pct": round(ret * 100, 4),
        "cagr_pct": round(((end / start) ** (1 / years) - 1) * 100, 4) if start > 0 else 0.0,
        "max_drawdown_pct": round(dd * 100, 4),
        "trade_signal_days": int(frame["action"].astype(str).ne("hold").sum()),
        "total_transaction_cost": round(float(pd.to_numeric(frame["transaction_cost"], errors="coerce").fillna(0).sum()), 2),
    }


def _period_drawdown(equity: pd.Series) -> float:
    values = pd.to_numeric(equity, errors="coerce").dropna()
    if values.empty:
        return 0.0
    running_max = values.cummax()
    drawdown = values / running_max - 1
    return float(drawdown.min())


def _monthly_performance(daily: pd.DataFrame) -> pd.DataFrame:
    frame = daily.copy()
    frame["date_ts"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["month"] = frame["date_ts"].dt.to_period("M").astype(str)
    rows: list[dict[str, Any]] = []
    for (variant, basis, month), group in frame.groupby(["variant_id", "execution_basis", "month"], dropna=False):
        group = group.sort_values("date_ts")
        if group.empty:
            continue
        start = float(pd.to_numeric(group["portfolio_equity"], errors="coerce").iloc[0])
        end = float(pd.to_numeric(group["portfolio_equity"], errors="coerce").iloc[-1])
        month_ret = end / start - 1 if start else 0.0
        rows.append(
            {
                "variant_id": variant,
                "execution_basis": basis,
                "month": month,
                "start_date": str(group["date"].iloc[0]),
                "end_date": str(group["date"].iloc[-1]),
                "return_pct": round(month_ret * 100, 4),
                "win_month": bool(month_ret > 0),
                "in_2024_now_main": bool(str(group["date"].iloc[-1]) >= "2024-01-01"),
            }
        )
    return pd.DataFrame(rows)


def _worst_month(monthly: pd.DataFrame) -> pd.DataFrame:
    if monthly.empty:
        return pd.DataFrame()
    rows = []
    for (variant, basis), group in monthly.groupby(["variant_id", "execution_basis"], dropna=False):
        group = group[group["in_2024_now_main"].astype(bool)]
        if group.empty:
            continue
        group = group.sort_values("return_pct")
        worst = group.iloc[0]
        rows.append(
            {
                "variant_id": variant,
                "execution_basis": basis,
                "period_label": "2024_now_main",
                "worst_month": worst["month"],
                "worst_month_return_pct": worst["return_pct"],
                "month_win_rate": round(float(group["win_month"].mean()), 6),
                "month_count": int(len(group)),
            }
        )
    return pd.DataFrame(rows)


def _trade_cost_summary(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(columns=["variant_id", "execution_basis", "period_label", "trade_rows", "total_transaction_cost"])
    frame = trades.copy()
    date_column = "date" if "date" in frame.columns else "fill_date"
    frame["date_ts"] = pd.to_datetime(frame[date_column], errors="coerce")
    frame["period_label"] = frame["date_ts"].map(lambda value: "2024_now_main" if pd.notna(value) and value >= pd.Timestamp(MAIN_START_DATE) else "pre_2024_or_unknown")
    rows = []
    for (variant, basis, period), group in frame.groupby(["variant_id", "execution_basis", "period_label"], dropna=False):
        rows.append(
            {
                "variant_id": variant,
                "execution_basis": basis,
                "period_label": period,
                "trade_rows": int(len(group)),
                "total_transaction_cost": round(float(pd.to_numeric(group["transaction_cost"], errors="coerce").fillna(0).sum()), 2),
            }
        )
    return pd.DataFrame(rows)


def _attach_trade_costs(performance: pd.DataFrame, costs: pd.DataFrame) -> pd.DataFrame:
    if costs.empty:
        performance["trade_rows"] = 0
        return performance
    main_costs = costs[costs["period_label"].eq("2024_now_main")][
        ["variant_id", "execution_basis", "trade_rows", "total_transaction_cost"]
    ].copy()
    out = performance.drop(columns=["total_transaction_cost"], errors="ignore").merge(
        main_costs,
        on=["variant_id", "execution_basis"],
        how="left",
        suffixes=("", "_trade"),
    )
    out["trade_rows"] = pd.to_numeric(out["trade_rows"], errors="coerce").fillna(0).astype(int)
    out["total_transaction_cost"] = pd.to_numeric(out["total_transaction_cost"], errors="coerce").fillna(0.0)
    out.loc[~out["period_label"].eq("2024_now_main"), ["trade_rows", "total_transaction_cost"]] = 0
    return out


def _pool2_effect_decision(performance: pd.DataFrame, costs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for basis in ("same_day", "next_day"):
        main = performance[(performance["execution_basis"].eq(basis)) & (performance["period_label"].eq("2024_now_main"))]
        formal = main[main["variant_id"].eq("current_formal_pool1_pool2_remove_cap")]
        pool1 = main[main["variant_id"].eq("new_pool1_only_no_overlay")]
        if formal.empty or pool1.empty:
            continue
        f = formal.iloc[0]
        p = pool1.iloc[0]
        return_delta = float(f["return_pct"]) - float(p["return_pct"])
        mdd_delta = float(f["max_drawdown_pct"]) - float(p["max_drawdown_pct"])
        state = "pool2_improves_return" if return_delta > 0 else "pool2_drags_return"
        if return_delta <= 0 and mdd_delta < 0:
            state = "pool2_drags_return_and_deepens_mdd"
        elif return_delta <= 0 and mdd_delta > 0:
            state = "pool2_drags_return_but_reduces_mdd"
        rows.append(
            {
                "execution_basis": basis,
                "period_label": "2024_now_main",
                "formal_return_pct": f["return_pct"],
                "pool1_only_return_pct": p["return_pct"],
                "formal_minus_pool1_return_pp": round(return_delta, 4),
                "formal_mdd_pct": f["max_drawdown_pct"],
                "pool1_only_mdd_pct": p["max_drawdown_pct"],
                "formal_minus_pool1_mdd_pp": round(mdd_delta, 4),
                "pool2_effect_state": state,
            }
        )
    return pd.DataFrame(rows)


def _variant_identity() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "variant_id": "current_formal_pool1_pool2_remove_cap",
                "model_id": FORMAL_MODEL_TARGET,
                "selector": FORMAL_MODEL_ROUTE,
                "description": "目前每日報告正式版：Pool1 主攻，Pool2 disagreement 時需 1 日確認；不使用 00631L cap40。",
            },
            {
                "variant_id": "new_pool1_only_no_overlay",
                "model_id": "new_pool1_only_no_overlay",
                "selector": "pool1_primary_no_pool2_confirmation",
                "description": "新 Pool1-only：只使用 Pool1 目標，不使用 Pool2 確認或風控層。",
            },
        ]
    )


def _same_date_range(daily: pd.DataFrame) -> bool:
    ranges = daily.groupby(["variant_id", "execution_basis"])["date"].agg(["min", "max"])
    return len(ranges.drop_duplicates()) == 1


def _summary_markdown(performance: pd.DataFrame, worst: pd.DataFrame, costs: pd.DataFrame, effect: pd.DataFrame) -> str:
    lines = [
        "# 日報正式版 vs 新 Pool1-only 同口徑驗收",
        "",
        "本輸出只做 apples-to-apples 驗收，不改正式模型、不改交易決策。",
        "",
        f"- 目前正式 model id: `{FORMAL_MODEL_TARGET}`",
        f"- 目前正式 selector: `{FORMAL_MODEL_ROUTE}`",
        "- 新 Pool1-only id: `new_pool1_only_no_overlay`",
        "- 主比較區間：2024-01-01 起到共同 latest complete date。",
        "- same-day 與 next-day 分開列示，不混比。",
        "",
        "## Pool2 效果判讀",
    ]
    if effect.empty:
        lines.append("- 無法判讀：缺 performance rows。")
    else:
        for row in effect.to_dict(orient="records"):
            lines.append(
                "- {basis}: formal minus Pool1-only return {ret}pp; MDD delta {mdd}pp; state `{state}`.".format(
                    basis=row["execution_basis"],
                    ret=row["formal_minus_pool1_return_pp"],
                    mdd=row["formal_minus_pool1_mdd_pp"],
                    state=row["pool2_effect_state"],
                )
            )
    lines.extend(["", "## 2024-now performance", ""])
    main = performance[performance["period_label"].eq("2024_now_main")]
    for row in main.sort_values(["execution_basis", "variant_id"]).to_dict(orient="records"):
        lines.append(
                "- {variant} / {basis}: {start}~{end}, return {ret}%, CAGR {cagr}%, MDD {mdd}%, trade signal days {trades}, trade rows {trade_rows}, cost {cost}.".format(
                variant=row["variant_id"],
                basis=row["execution_basis"],
                start=row["start_date"],
                end=row["end_date"],
                ret=row["return_pct"],
                cagr=row["cagr_pct"],
                    mdd=row["max_drawdown_pct"],
                    trades=row["trade_signal_days"],
                    trade_rows=row.get("trade_rows", 0),
                    cost=row["total_transaction_cost"],
                )
            )
    lines.extend(["", "## Worst month / monthly win rate", ""])
    for row in worst.sort_values(["execution_basis", "variant_id"]).to_dict(orient="records"):
        lines.append(
            f"- {row['variant_id']} / {row['execution_basis']}: worst {row['worst_month']} {row['worst_month_return_pct']}%, monthly win rate {row['month_win_rate']}."
        )
    lines.extend(["", "## Boundary", "", "- formal_model_changed=false", "- trade_decision_changed=false", "- uses_forward_return_as_rule=false"])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate current formal daily model vs Pool1-only.")
    parser.add_argument("--formal-replay-dir", default=DEFAULT_FORMAL_REPLAY_DIR)
    parser.add_argument("--price-cache-dir", default=DEFAULT_PRICE_CACHE_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    run_formal_vs_pool1_only_validation(
        formal_replay_dir=args.formal_replay_dir,
        price_cache_dir=args.price_cache_dir,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
