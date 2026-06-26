from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from backtest_lab.costs import TaiwanCostModel
from backtest_lab.data import load_price_csv
from backtest_lab.portfolio import Portfolio, Trade


DEFAULT_FORMAL_REPLAY_DIR = "outputs/stock_pool_formal_daily_replay_pit_pool2_daily_final_combined_20260624"
DEFAULT_POOL_DIAGNOSTICS = (
    "outputs/stock_pool_consensus_health_replay_pit_pool2_daily_final_combined_20260624/"
    "stock_pool_consensus_pool_diagnostics_history.csv"
)
DEFAULT_REPORT_BOUNDARY = "outputs/final_decision_layer_report_boundary_20260625/final_decision_report_boundary_panel.csv"
DEFAULT_RR_SHADOW = "outputs/rr_partial_switch_paper_trade_shadow_20260625/sample_accumulation_status.csv"
DEFAULT_PRICE_CACHE_DIR = "backtest_cache/stock_pool_triad_v1_corrected"
DEFAULT_OUTPUT_DIR = "outputs/three_pool_vs_pool1_comparison_panels_20260626"
INITIAL_CASH = 1_000_000.0
VARIANT_POOL1 = "pool1_only_formal_replay"
VARIANT_BASELINE = "current_formal_three_pool_baseline"
VARIANT_LABELS = "three_pool_with_report_only_labels"
VARIANT_EXECUTION_SHADOW = "three_pool_with_execution_shadow_diagnostics"


def run_three_pool_vs_pool1_comparison_panels(
    *,
    formal_replay_dir: str | Path = DEFAULT_FORMAL_REPLAY_DIR,
    pool_diagnostics_path: str | Path = DEFAULT_POOL_DIAGNOSTICS,
    report_boundary_path: str | Path = DEFAULT_REPORT_BOUNDARY,
    rr_shadow_path: str | Path = DEFAULT_RR_SHADOW,
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
        formal_dir = Path(formal_replay_dir)
        log("load_inputs", "started", str(formal_dir))
        decision = pd.read_csv(formal_dir / "formal_three_pool_decision_panel.csv").fillna("")
        baseline_daily = pd.read_csv(formal_dir / "baseline_three_pool_formal_daily_equity.csv").fillna("")
        pool_signal = _read_optional_csv(pool_diagnostics_path)
        report_overlay_source = _read_optional_csv(report_boundary_path)
        rr_shadow_source = _read_optional_csv(rr_shadow_path)
        _validate_decision(decision)
        _validate_baseline_daily(baseline_daily)

        latest_complete_common_date = str(baseline_daily["date"].iloc[-1])
        start_date = str(baseline_daily["date"].iloc[0])
        tickers = _needed_tickers(decision)
        prices = _load_prices(tickers, Path(price_cache_dir))

        log("simulate_pool1_only", "started", "")
        pool1_daily, pool1_trades = _simulate_variant(
            decision,
            prices,
            winner_column="pool1_vote",
            variant=VARIANT_POOL1,
            initial_cash=initial_cash,
        )
        baseline_variant_daily = _normalize_baseline_daily(baseline_daily, VARIANT_BASELINE)
        baseline_trades = _trade_ledger_from_daily(baseline_variant_daily, VARIANT_BASELINE)

        log("build_panels", "started", "")
        daily_equity = pd.concat(
            [
                pool1_daily,
                baseline_variant_daily,
                _copy_variant_daily(baseline_variant_daily, VARIANT_LABELS),
                _copy_variant_daily(baseline_variant_daily, VARIANT_EXECUTION_SHADOW),
            ],
            ignore_index=True,
        )
        trade_ledger = pd.concat(
            [
                pool1_trades,
                baseline_trades,
                _copy_variant_trades(baseline_trades, VARIANT_LABELS),
                _copy_variant_trades(baseline_trades, VARIANT_EXECUTION_SHADOW),
            ],
            ignore_index=True,
        )
        daily_target = _build_daily_target_by_variant(decision, daily_equity)
        period_performance = _period_performance(daily_equity)
        consensus_health = _consensus_health(decision, daily_equity)
        target_stability = _target_stability(daily_target)
        rapid_flip = _rapid_flip_diagnostics(daily_target)
        target_drop = _target_drop_from_top3(decision, pool_signal)
        dominance = _pool_dominance_summary(decision)
        report_overlay = _report_only_label_overlay(report_overlay_source, baseline_variant_daily)
        execution_shadow = _execution_shadow_overlay(rr_shadow_source, baseline_variant_daily)

        log("write_outputs", "started", "")
        period_performance.to_csv(output / "period_performance_by_variant.csv", index=False, encoding="utf-8-sig")
        daily_equity.to_csv(output / "daily_equity_by_variant.csv", index=False, encoding="utf-8-sig")
        daily_target.to_csv(output / "daily_target_by_variant.csv", index=False, encoding="utf-8-sig")
        trade_ledger.to_csv(output / "trade_ledger_by_variant.csv", index=False, encoding="utf-8-sig")
        _pool_signal_panel(pool_signal, decision).to_csv(output / "pool_signal_panel.csv", index=False, encoding="utf-8-sig")
        consensus_health.to_csv(output / "consensus_health_by_variant_period.csv", index=False, encoding="utf-8-sig")
        target_stability.to_csv(output / "target_stability_panel.csv", index=False, encoding="utf-8-sig")
        rapid_flip.to_csv(output / "rapid_flip_diagnostics.csv", index=False, encoding="utf-8-sig")
        target_drop.to_csv(output / "target_drop_from_top3_diagnostics.csv", index=False, encoding="utf-8-sig")
        dominance.to_csv(output / "pool_dominance_summary.csv", index=False, encoding="utf-8-sig")
        report_overlay.to_csv(output / "report_only_label_overlay.csv", index=False, encoding="utf-8-sig")
        execution_shadow.to_csv(output / "execution_shadow_overlay.csv", index=False, encoding="utf-8-sig")
        (output / "comparison_panel_summary_zh.md").write_text(
            _summary_markdown(period_performance, latest_complete_common_date),
            encoding="utf-8",
        )
        manifest = {
            "schema_version": 1,
            "task_id": "TASK-BACKTEST-CORE-THREE-POOL-VS-POOL1-COMPARISON-PANELS-001",
            "model": "three_pool_vs_pool1_comparison_panels",
            "status": "completed",
            "formal_replay_dir": str(formal_dir),
            "pool_diagnostics_path": str(pool_diagnostics_path),
            "price_cache_dir": str(price_cache_dir),
            "initial_cash": initial_cash,
            "start_date": start_date,
            "latest_complete_common_date": latest_complete_common_date,
            "formal_model_changed": False,
            "trade_decision_changed": False,
            "pool3_shadow_used_as_formal": False,
            "report_only_labels_used_in_performance": False,
            "rr_partial_switch_used_in_performance": False,
            "valuation_used": False,
            "h3_used": False,
            "same_date_range_for_variants": _same_date_range(daily_equity),
            "same_cost_model_for_variants": True,
            "outputs": {
                "period_performance_by_variant": "period_performance_by_variant.csv",
                "daily_equity_by_variant": "daily_equity_by_variant.csv",
                "daily_target_by_variant": "daily_target_by_variant.csv",
                "trade_ledger_by_variant": "trade_ledger_by_variant.csv",
                "pool_signal_panel": "pool_signal_panel.csv",
                "consensus_health_by_variant_period": "consensus_health_by_variant_period.csv",
                "target_stability_panel": "target_stability_panel.csv",
                "rapid_flip_diagnostics": "rapid_flip_diagnostics.csv",
                "target_drop_from_top3_diagnostics": "target_drop_from_top3_diagnostics.csv",
                "pool_dominance_summary": "pool_dominance_summary.csv",
                "report_only_label_overlay": "report_only_label_overlay.csv",
                "execution_shadow_overlay": "execution_shadow_overlay.csv",
                "summary": "comparison_panel_summary_zh.md",
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
        pd.DataFrame([{"step": "run_three_pool_vs_pool1_comparison_panels", "error": str(exc)}]).to_csv(
            output / "failed.csv",
            index=False,
            encoding="utf-8-sig",
        )
        log("failed", "failed", str(exc))
        raise


def _simulate_variant(
    decision: pd.DataFrame,
    prices: dict[str, pd.Series],
    *,
    winner_column: str,
    variant: str,
    initial_cash: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    portfolio = Portfolio(initial_cash, TaiwanCostModel())
    rows: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []
    running_max = initial_cash
    previous_equity = initial_cash
    for item in decision.to_dict(orient="records"):
        date = pd.Timestamp(item["date"]).strftime("%Y-%m-%d")
        target = _text(item.get(winner_column))
        current = portfolio.current_ticker()
        transaction_cost = 0
        turnover = 0.0
        action = "hold"
        if target and target in prices:
            price = _price_on_or_before(prices[target], date)
            if price is not None and current != target:
                if current:
                    current_price = _price_on_or_before(prices.get(current, pd.Series(dtype=float)), date)
                    if current_price is not None:
                        trade = portfolio.sell_all(date, current, _asset_type(current), current_price, f"{variant}_switch")
                        if trade:
                            transaction_cost += trade.costs
                            turnover += trade.gross_amount
                            trades.append(_trade_row(trade, variant))
                trade = portfolio.buy_max(date, target, _asset_type(target), price, f"{variant}_target")
                if trade:
                    transaction_cost += trade.costs
                    turnover += trade.gross_amount
                    trades.append(_trade_row(trade, variant))
                    action = "switch" if current else "buy"
        close_prices = {
            ticker: price
            for ticker, series in prices.items()
            if (price := _price_on_or_before(series, date)) is not None
        }
        equity = portfolio.market_value(close_prices) if close_prices else previous_equity
        previous_equity = equity
        running_max = max(running_max, equity)
        position = portfolio.current_ticker() or "cash"
        rows.append(
            {
                "variant": variant,
                "date": date,
                "period": item.get("period", ""),
                "pool1_vote": item.get("pool1_vote", ""),
                "pool2_vote": item.get("pool2_vote", ""),
                "pool3_vote": item.get("pool3_vote", ""),
                "winner_ticker": target,
                "position_ticker": position,
                "cash": round(portfolio.cash, 2),
                "equity": round(equity, 2),
                "drawdown": round(equity / running_max - 1, 8) if running_max else 0.0,
                "turnover": round(turnover, 2),
                "transaction_cost": transaction_cost,
                "action": action,
                "data_status": "comparison_panel_diagnostic",
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(trades)


def _normalize_baseline_daily(frame: pd.DataFrame, variant: str) -> pd.DataFrame:
    output = frame.copy()
    output.insert(0, "variant", variant)
    output["data_status"] = "comparison_panel_formal_baseline"
    return output


def _copy_variant_daily(frame: pd.DataFrame, variant: str) -> pd.DataFrame:
    output = frame.copy()
    output["variant"] = variant
    output["data_status"] = "report_only_overlay_no_performance_change"
    return output


def _copy_variant_trades(frame: pd.DataFrame, variant: str) -> pd.DataFrame:
    output = frame.copy()
    output["variant"] = variant
    output["diagnostic_note"] = "same_trade_ledger_as_current_formal_baseline"
    return output


def _trade_ledger_from_daily(daily: pd.DataFrame, variant: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for item in daily.to_dict(orient="records"):
        if _text(item.get("action")) == "hold":
            continue
        rows.append(
            {
                "variant": variant,
                "date": item.get("date", ""),
                "ticker": item.get("winner_ticker", ""),
                "action": item.get("action", ""),
                "shares": "",
                "price": "",
                "gross_amount": item.get("turnover", 0),
                "costs": item.get("transaction_cost", 0),
                "cash_after": item.get("cash", ""),
                "reason": "reconstructed_from_daily_formal_replay",
                "diagnostic_note": "baseline daily replay has daily cost/turnover, not raw trade fills",
            }
        )
    return pd.DataFrame(rows)


def _trade_row(trade: Trade, variant: str) -> dict[str, Any]:
    row = asdict(trade)
    row["variant"] = variant
    row["diagnostic_note"] = ""
    return row


def _build_daily_target_by_variant(decision: pd.DataFrame, daily_equity: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    decision_by_date = {pd.Timestamp(row["date"]).strftime("%Y-%m-%d"): row for row in decision.to_dict(orient="records")}
    for row in daily_equity.to_dict(orient="records"):
        date = _text(row.get("date"))
        item = decision_by_date.get(date, {})
        variant = _text(row.get("variant"))
        rows.append(
            {
                "variant": variant,
                "date": date,
                "period": row.get("period", ""),
                "formal_target": row.get("winner_ticker", ""),
                "position_ticker": row.get("position_ticker", ""),
                "action": row.get("action", ""),
                "pool1_vote": item.get("pool1_vote", row.get("pool1_vote", "")),
                "pool2_vote": item.get("pool2_vote", row.get("pool2_vote", "")),
                "pool3_vote": item.get("pool3_vote", row.get("pool3_vote", "")),
                "consensus_state": item.get("consensus_state", row.get("consensus_state", "")),
                "entry_signal_without_exit_confirmation": bool(
                    variant == VARIANT_POOL1 and _text(item.get("pool1_vote")) and _text(item.get("consensus_state")) != "consensus"
                ),
                "possible_execution_layer_issue": False,
            }
        )
    panel = pd.DataFrame(rows)
    return _add_target_stability_flags(panel)


def _add_target_stability_flags(panel: pd.DataFrame) -> pd.DataFrame:
    output = panel.sort_values(["variant", "date"]).copy()
    for column in (
        "formal_target_changed_within_1d",
        "formal_target_changed_within_3d",
        "pool1_target_changed_within_1d",
        "pool1_target_changed_within_3d",
        "rapid_flip_same_target_window_1_3d",
    ):
        output[column] = False
    for variant, idx in output.groupby("variant").groups.items():
        target = output.loc[idx, "formal_target"].astype(str).tolist()
        pool1 = output.loc[idx, "pool1_vote"].astype(str).tolist()
        changed = _changed_within(target)
        pool1_changed = _changed_within(pool1)
        output.loc[idx, "formal_target_changed_within_1d"] = changed[1]
        output.loc[idx, "formal_target_changed_within_3d"] = changed[3]
        output.loc[idx, "pool1_target_changed_within_1d"] = pool1_changed[1]
        output.loc[idx, "pool1_target_changed_within_3d"] = pool1_changed[3]
        output.loc[idx, "rapid_flip_same_target_window_1_3d"] = _rapid_flip_same_target(target)
        output.loc[idx, "possible_execution_layer_issue"] = output.loc[idx, "formal_target_changed_within_3d"].map(_truthy)
    return output


def _changed_within(values: list[str]) -> dict[int, list[bool]]:
    result = {1: [], 3: []}
    for i, value in enumerate(values):
        for window in result:
            future = values[i + 1 : i + window + 1]
            result[window].append(bool(value and any(item and item != value for item in future)))
    return result


def _rapid_flip_same_target(values: list[str]) -> list[bool]:
    flags: list[bool] = []
    for i, value in enumerate(values):
        future = values[i + 1 : i + 4]
        flags.append(bool(value and value in future and any(item and item != value for item in future)))
    return flags


def _period_performance(daily: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (variant, label), group in _period_groups(daily):
        rows.append(_performance_row(variant, label, group))
    return pd.DataFrame(rows)


def _period_groups(daily: pd.DataFrame) -> list[tuple[tuple[str, str], pd.DataFrame]]:
    frame = daily.copy()
    frame["date_ts"] = pd.to_datetime(frame["date"])
    groups: list[tuple[tuple[str, str], pd.DataFrame]] = []
    periods = {
        "2022": ("2022-01-01", "2022-12-31"),
        "2023": ("2023-01-01", "2023-12-31"),
        "2024_now": ("2024-01-01", None),
        "2024_hard_gate": ("2024-01-01", "2024-12-31"),
        "full": (None, None),
    }
    for variant, by_variant in frame.groupby("variant", dropna=False):
        for label, (start, end) in periods.items():
            subset = by_variant
            if start:
                subset = subset[subset["date_ts"] >= pd.Timestamp(start)]
            if end:
                subset = subset[subset["date_ts"] <= pd.Timestamp(end)]
            groups.append(((str(variant), label), subset.copy()))
    return groups


def _performance_row(variant: str, period_label: str, group: pd.DataFrame) -> dict[str, Any]:
    if group.empty:
        return {"variant": variant, "period_label": period_label, "status": "empty"}
    equity = pd.to_numeric(group["equity"], errors="coerce")
    start = float(equity.iloc[0])
    end = float(equity.iloc[-1])
    return {
        "variant": variant,
        "period_label": period_label,
        "status": "completed",
        "start_date": group["date"].iloc[0],
        "end_date": group["date"].iloc[-1],
        "start_equity": round(start, 2),
        "final_equity": round(end, 2),
        "return_pct": round((end / start - 1) * 100, 4) if start else 0.0,
        "max_drawdown_pct": round(float(pd.to_numeric(group["drawdown"], errors="coerce").min()) * 100, 4),
        "trade_days": int(group["action"].astype(str).ne("hold").sum()),
        "total_transaction_cost": round(float(pd.to_numeric(group["transaction_cost"], errors="coerce").sum()), 2),
        "total_turnover": round(float(pd.to_numeric(group["turnover"], errors="coerce").sum()), 2),
    }


def _consensus_health(decision: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    targets = daily[["variant", "date", "winner_ticker", "action"]].copy()
    merged = decision.merge(targets, on="date", how="left", suffixes=("_decision", ""))
    for (variant, label), group in _period_groups(merged.rename(columns={"winner_ticker": "equity_target", "action": "equity_action"})):
        if group.empty:
            continue
        rows.append(
            {
                "variant": variant,
                "period_label": label,
                "row_count": int(len(group)),
                "exact_consensus_rate": _mean_bool(group["consensus_state"].astype(str).eq("consensus")),
                "divergent_rate": _mean_bool(group["consensus_state"].astype(str).eq("divergent")),
                "no_vote_rate": _mean_bool(group["consensus_state"].astype(str).isin(["no_vote", "insufficient_votes"])),
                "target_formed_rate": _mean_bool(group["equity_target"].astype(str).str.strip().ne("")),
                "trade_action_rate": _mean_bool(group["equity_action"].astype(str).ne("hold")),
            }
        )
    return pd.DataFrame(rows)


def _target_stability(daily_target: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (variant, period), group in daily_target.groupby(["variant", "period"], dropna=False):
        rows.append(
            {
                "variant": variant,
                "period_label": period,
                "row_count": int(len(group)),
                "formal_target_changed_within_1d_rate": _mean_bool(group["formal_target_changed_within_1d"]),
                "formal_target_changed_within_3d_rate": _mean_bool(group["formal_target_changed_within_3d"]),
                "pool1_target_changed_within_1d_rate": _mean_bool(group["pool1_target_changed_within_1d"]),
                "pool1_target_changed_within_3d_rate": _mean_bool(group["pool1_target_changed_within_3d"]),
                "rapid_flip_same_target_window_1_3d_rate": _mean_bool(group["rapid_flip_same_target_window_1_3d"]),
                "entry_signal_without_exit_confirmation_rate": _mean_bool(group["entry_signal_without_exit_confirmation"]),
                "possible_execution_layer_issue_rate": _mean_bool(group["possible_execution_layer_issue"]),
            }
        )
    return pd.DataFrame(rows)


def _rapid_flip_diagnostics(daily_target: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "variant",
        "date",
        "period",
        "formal_target",
        "pool1_vote",
        "formal_target_changed_within_1d",
        "formal_target_changed_within_3d",
        "pool1_target_changed_within_1d",
        "pool1_target_changed_within_3d",
        "rapid_flip_same_target_window_1_3d",
        "entry_signal_without_exit_confirmation",
        "possible_execution_layer_issue",
    ]
    return daily_target[columns].copy()


def _target_drop_from_top3(decision: pd.DataFrame, pool_signal: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    pool_map = _top3_map(pool_signal)
    ordered = decision.sort_values("date").reset_index(drop=True)
    for idx, row in ordered.iterrows():
        date = pd.Timestamp(row["date"]).strftime("%Y-%m-%d")
        target = _text(row.get("winner_ticker"))
        top3 = pool_map.get(date, set())
        future_top3 = [pool_map.get(pd.Timestamp(ordered.iloc[j]["date"]).strftime("%Y-%m-%d"), set()) for j in range(idx + 1, min(len(ordered), idx + 6))]
        rows.append(
            {
                "date": date,
                "period": row.get("period", ""),
                "formal_target": target,
                "target_in_top3_today": bool(target and target in top3),
                "target_drop_from_top3_next_1d": _drop_next(target, future_top3, 1),
                "target_drop_from_top3_next_2d": _drop_next(target, future_top3, 2),
                "target_drop_from_top3_next_3d": _drop_next(target, future_top3, 3),
                "target_reappears_in_top3_within_5d": bool(target and any(target in items for items in future_top3[:5])),
            }
        )
    return pd.DataFrame(rows)


def _top3_map(pool_signal: pd.DataFrame) -> dict[str, set[str]]:
    if pool_signal.empty or "signal_date" not in pool_signal.columns:
        return {}
    frame = pool_signal.copy()
    ticker_col = "top_ticker" if "top_ticker" in frame.columns else "vote_target" if "vote_target" in frame.columns else ""
    if not ticker_col:
        return {}
    if "rank" in frame.columns:
        frame = frame[pd.to_numeric(frame["rank"], errors="coerce").fillna(999) <= 3]
    elif "top_rank" in frame.columns:
        frame = frame[pd.to_numeric(frame["top_rank"], errors="coerce").fillna(999) <= 3]
    result: dict[str, set[str]] = {}
    for date, group in frame.groupby(frame["signal_date"].astype(str), dropna=False):
        result[str(date)] = {_text(value) for value in group[ticker_col].tolist() if _text(value)}
    return result


def _drop_next(target: str, future_top3: list[set[str]], days: int) -> bool:
    if not target or not future_top3:
        return False
    subset = future_top3[:days]
    return bool(subset and any(target not in items for items in subset))


def _pool_dominance_summary(decision: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for pool in ("pool1_vote", "pool2_vote", "pool3_vote", "winner_ticker"):
        counts = decision[pool].astype(str).str.strip()
        counts = counts[counts.ne("")]
        top_ticker = counts.value_counts().index[0] if not counts.empty else ""
        top_count = int(counts.value_counts().iloc[0]) if not counts.empty else 0
        rows.append(
            {
                "pool_or_target": pool,
                "non_empty_count": int(len(counts)),
                "unique_ticker_count": int(counts.nunique()) if not counts.empty else 0,
                "top_ticker": top_ticker,
                "top_ticker_count": top_count,
                "top_ticker_share": round(top_count / len(counts), 6) if len(counts) else 0.0,
            }
        )
    return pd.DataFrame(rows)


def _report_only_label_overlay(source: pd.DataFrame, baseline_daily: pd.DataFrame) -> pd.DataFrame:
    if source.empty:
        return pd.DataFrame([{"status": "missing_report_boundary_source", "report_only_labels_used_in_performance": False}])
    rows = []
    for state, group in source.groupby("final_decision_user_reading_state", dropna=False):
        rows.append(
            {
                "final_decision_user_reading_state": state,
                "row_count": int(len(group)),
                "active_in_trade_decision_count": int(group.get("final_decision_label_active_in_trade_decision", pd.Series(dtype=bool)).map(_truthy).sum()),
                "report_only_labels_used_in_performance": False,
                "baseline_equity_unchanged_rows": int(len(baseline_daily)),
            }
        )
    return pd.DataFrame(rows)


def _execution_shadow_overlay(source: pd.DataFrame, baseline_daily: pd.DataFrame) -> pd.DataFrame:
    if source.empty:
        return pd.DataFrame([{"status": "missing_rr_shadow_source", "rr_partial_switch_used_in_performance": False}])
    output = source.copy()
    output["rr_partial_switch_used_in_performance"] = False
    output["baseline_equity_unchanged_rows"] = len(baseline_daily)
    return output


def _pool_signal_panel(pool_signal: pd.DataFrame, decision: pd.DataFrame) -> pd.DataFrame:
    if not pool_signal.empty:
        return pool_signal
    return decision[["period", "date", "pool1_vote", "pool2_vote", "pool3_vote", "winner_ticker", "consensus_state"]].copy()


def _read_optional_csv(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    if not source.exists():
        return pd.DataFrame()
    return pd.read_csv(source).fillna("")


def _validate_decision(decision: pd.DataFrame) -> None:
    required = {"period", "date", "pool1_vote", "pool2_vote", "pool3_vote", "winner_ticker"}
    missing = sorted(required - set(decision.columns))
    if missing:
        raise ValueError(f"decision panel missing columns: {missing}")


def _validate_baseline_daily(daily: pd.DataFrame) -> None:
    required = {"date", "period", "winner_ticker", "equity", "drawdown", "transaction_cost", "turnover", "action"}
    missing = sorted(required - set(daily.columns))
    if missing:
        raise ValueError(f"baseline daily missing columns: {missing}")


def _needed_tickers(decision: pd.DataFrame) -> list[str]:
    tickers = set()
    for column in ("pool1_vote", "pool2_vote", "pool3_vote", "winner_ticker"):
        tickers.update(_text(value) for value in decision[column].tolist() if _text(value))
    return sorted(tickers)


def _load_prices(tickers: list[str], cache_dir: Path) -> dict[str, pd.Series]:
    prices: dict[str, pd.Series] = {}
    for ticker in tickers:
        path = cache_dir / f"{ticker.replace('.', '_')}.csv"
        if not path.exists():
            continue
        frame = load_price_csv(path)
        close = pd.to_numeric(frame["adj_close"], errors="coerce").dropna()
        prices[ticker] = close
    return prices


def _price_on_or_before(series: pd.Series, date: str) -> float | None:
    if series.empty:
        return None
    clipped = series.loc[series.index <= pd.Timestamp(date)]
    if clipped.empty:
        return None
    return float(clipped.iloc[-1])


def _asset_type(ticker: str) -> str:
    symbol = ticker.split(".")[0]
    return "etf" if symbol in {"0050", "00631L"} else "stock"


def _mean_bool(values: pd.Series) -> float:
    if len(values) == 0:
        return 0.0
    return round(float(values.map(_truthy).mean()), 6)


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _same_date_range(daily: pd.DataFrame) -> bool:
    ranges = daily.groupby("variant")["date"].agg(["min", "max", "count"])
    return bool(ranges["min"].nunique() == 1 and ranges["max"].nunique() == 1 and ranges["count"].nunique() == 1)


def _text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    if text.lower() == "nan":
        return ""
    return text.strip()


def _summary_markdown(period_performance: pd.DataFrame, latest_complete_common_date: str) -> str:
    full = period_performance[period_performance["period_label"].eq("full")]
    lines = [
        "# Three Pool vs Pool1 Comparison Panels",
        "",
        f"- latest_complete_common_date: {latest_complete_common_date}",
        "- 本輸出只做同口徑診斷，不改 formal selector / vote / target / trade action。",
        "- V3/V4 只疊 report-only 或 shadow 欄位，績效沿用 current formal three-pool baseline。",
        "",
        "## Full Period",
        "",
    ]
    for row in full.to_dict(orient="records"):
        lines.append(
            f"- {row.get('variant')}: return {row.get('return_pct')}%, "
            f"MDD {row.get('max_drawdown_pct')}%, trades {row.get('trade_days')}"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Pool1-only vs current three-pool comparison panels.")
    parser.add_argument("--formal-replay-dir", default=DEFAULT_FORMAL_REPLAY_DIR)
    parser.add_argument("--pool-diagnostics", default=DEFAULT_POOL_DIAGNOSTICS)
    parser.add_argument("--report-boundary", default=DEFAULT_REPORT_BOUNDARY)
    parser.add_argument("--rr-shadow", default=DEFAULT_RR_SHADOW)
    parser.add_argument("--price-cache-dir", default=DEFAULT_PRICE_CACHE_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--initial-cash", type=float, default=INITIAL_CASH)
    args = parser.parse_args()
    output = run_three_pool_vs_pool1_comparison_panels(
        formal_replay_dir=args.formal_replay_dir,
        pool_diagnostics_path=args.pool_diagnostics,
        report_boundary_path=args.report_boundary,
        rr_shadow_path=args.rr_shadow,
        price_cache_dir=args.price_cache_dir,
        output_dir=args.output_dir,
        initial_cash=args.initial_cash,
    )
    print(f"OUTPUT_DIR={output.resolve()}")


if __name__ == "__main__":
    main()
