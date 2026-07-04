from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


TASK_ID = "TASK-BACKTEST-CORE-SHORT-CYCLE-PULLBACK-ROBUSTNESS-PANELS-001"
BEST_VARIANT = "ma20_reclaim_overlay_20_when_formal_cash_or_market_exposure_hold60"
BASELINE_VARIANT = "baseline_formal_next_day"
DEFAULT_EXPERIMENTS_OUTPUT = (
    "C:/Users/zergv/Documents/Codex/2026-06-17/repo-ai-stock-backtest-lab-repo/outputs/"
    "experiments_short_cycle_pullback_portfolio_challenger_validation_20260704"
)
DEFAULT_CONTRACT_OUTPUT = "outputs/short_cycle_pullback_portfolio_challenger_spec_20260704"
DEFAULT_OUTPUT_DIR = "outputs/short_cycle_pullback_robustness_panels_20260704"


def run_short_cycle_pullback_robustness_panels(
    *,
    experiments_output: str | Path = DEFAULT_EXPERIMENTS_OUTPUT,
    contract_output: str | Path = DEFAULT_CONTRACT_OUTPUT,
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
        (output / "current_step.txt").write_text(f"{step}:{status}\n{detail}", encoding="utf-8")

    try:
        source = Path(experiments_output)
        contract = Path(contract_output)
        log("load_experiments_output", "started", str(source))
        manifest = _load_json(source / "manifest.json")
        contract_readiness = _load_json(contract / "readiness_for_experiments.json")
        daily = _read_csv_required(source / "daily_weight_ledger.csv")
        trades = _read_csv_required(source / "trade_ledger.csv")
        monthly = _read_csv_required(source / "monthly_performance.csv")
        performance = _read_csv_required(source / "portfolio_challenger_diagnostic.csv")
        overlap = _read_csv_required(source / "formal_target_overlap_audit.csv")
        event_usage = _read_csv_required(source / "event_usage_summary.csv")
        material = _read_csv_if_exists(source / "material_layer_case_slice.csv")
        case_6488 = _read_csv_if_exists(source / "6488_case_only.csv")

        log("build_panels", "started", "")
        baseline_daily = _variant_daily(daily, BASELINE_VARIANT)
        best_daily = _variant_daily(daily, BEST_VARIANT)
        baseline_perf = _variant_perf(performance, BASELINE_VARIANT)
        best_perf = _variant_perf(performance, BEST_VARIANT)
        oos = _oos_period_panel(baseline_daily, best_daily)
        contribution_month = _contribution_by_month(monthly)
        contribution_quarter = _contribution_by_quarter(contribution_month)
        contribution_ticker = _contribution_by_ticker(event_usage, trades)
        ablation = _ablation_panel(best_perf, baseline_perf, contribution_ticker, contribution_month, contribution_quarter)
        cost_turnover = _cost_turnover_panel(performance, trades)
        risk = _risk_panel(baseline_daily, best_daily, contribution_month)
        formal_conflict = _formal_overlap_conflict_audit(overlap)
        baseline_vs_best = _baseline_vs_best_periods(performance, oos)
        leave_one_event = _leave_one_event_summary(overlap)
        leave_one_month = _leave_one_summary(contribution_month, "month")
        leave_one_quarter = _leave_one_summary(contribution_quarter, "quarter")
        leave_one_ticker = _leave_one_summary(contribution_ticker, "ticker")
        cost_sensitivity = _cost_slippage_sensitivity(best_perf, baseline_perf, cost_turnover)
        pool1b_old_ai = _pool1b_vs_old_ai_contribution(contribution_ticker)
        top_events = _top_contribution_events(overlap, trades)
        readiness = _readiness(
            manifest=manifest,
            contract_readiness=contract_readiness,
            oos=oos,
            ablation=ablation,
            cost_sensitivity=cost_sensitivity,
            formal_conflict=formal_conflict,
            contribution_ticker=contribution_ticker,
            material=material,
            case_6488=case_6488,
        )
        robustness_manifest = _manifest(output, source, contract, readiness)

        log("write_outputs", "started", str(output))
        (output / "robustness_manifest.json").write_text(
            json.dumps(robustness_manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (output / "manifest.json").write_text(json.dumps(robustness_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        baseline_vs_best.to_csv(output / "baseline_vs_best_variant_periods.csv", index=False, encoding="utf-8-sig")
        oos.to_csv(output / "oos_walk_forward_summary.csv", index=False, encoding="utf-8-sig")
        oos.to_csv(output / "oos_period_panel.csv", index=False, encoding="utf-8-sig")
        leave_one_event.to_csv(output / "leave_one_event_summary.csv", index=False, encoding="utf-8-sig")
        leave_one_month.to_csv(output / "leave_one_month_summary.csv", index=False, encoding="utf-8-sig")
        leave_one_quarter.to_csv(output / "leave_one_quarter_summary.csv", index=False, encoding="utf-8-sig")
        leave_one_ticker.to_csv(output / "leave_one_ticker_summary.csv", index=False, encoding="utf-8-sig")
        top_events.to_csv(output / "top_contribution_events.csv", index=False, encoding="utf-8-sig")
        contribution_month.head(20).to_csv(output / "top_contribution_months.csv", index=False, encoding="utf-8-sig")
        contribution_quarter.head(20).to_csv(output / "top_contribution_quarters.csv", index=False, encoding="utf-8-sig")
        contribution_ticker.head(20).to_csv(output / "top_contribution_tickers.csv", index=False, encoding="utf-8-sig")
        contribution_ticker.to_csv(output / "contribution_by_ticker.csv", index=False, encoding="utf-8-sig")
        contribution_month.to_csv(output / "contribution_by_month.csv", index=False, encoding="utf-8-sig")
        contribution_quarter.to_csv(output / "contribution_by_quarter.csv", index=False, encoding="utf-8-sig")
        ablation.to_csv(output / "ablation_panel.csv", index=False, encoding="utf-8-sig")
        cost_sensitivity.to_csv(output / "cost_slippage_sensitivity.csv", index=False, encoding="utf-8-sig")
        cost_turnover.to_csv(output / "trade_turnover_cost_audit.csv", index=False, encoding="utf-8-sig")
        cost_turnover.to_csv(output / "cost_turnover_panel.csv", index=False, encoding="utf-8-sig")
        risk.to_csv(output / "risk_panel.csv", index=False, encoding="utf-8-sig")
        formal_conflict.to_csv(output / "cash_market_exposure_entry_audit.csv", index=False, encoding="utf-8-sig")
        formal_conflict.to_csv(output / "formal_target_conflict_audit.csv", index=False, encoding="utf-8-sig")
        pool1b_old_ai.to_csv(output / "pool1b_vs_old_ai_contribution.csv", index=False, encoding="utf-8-sig")
        material.to_csv(output / "material_layer_case_only_audit.csv", index=False, encoding="utf-8-sig")
        case_6488.to_csv(output / "6488_case_only_audit.csv", index=False, encoding="utf-8-sig")
        (output / "robustness_readiness_for_experiments.json").write_text(
            json.dumps(readiness, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (output / "overfit_guard.md").write_text(_overfit_guard(readiness), encoding="utf-8")
        (output / "final_summary_zh.md").write_text(_summary_zh(readiness), encoding="utf-8")
        pd.DataFrame([{"step": TASK_ID, "status": "completed_robustness_panels", "output_dir": str(output)}]).to_csv(
            output / "completed.csv",
            index=False,
            encoding="utf-8-sig",
        )
        pd.DataFrame(columns=["step", "status", "reason"]).to_csv(output / "failed.csv", index=False, encoding="utf-8-sig")
        log("completed", "completed", str(output.resolve()))
        return output
    except Exception as exc:
        pd.DataFrame([{"step": TASK_ID, "status": "failed", "reason": str(exc)}]).to_csv(
            output / "failed.csv",
            index=False,
            encoding="utf-8-sig",
        )
        log("failed", "failed", str(exc))
        raise


def _variant_daily(daily: pd.DataFrame, variant: str) -> pd.DataFrame:
    frame = daily[daily["variant_id"].astype(str).eq(variant)].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["equity"] = pd.to_numeric(frame["equity"], errors="coerce")
    frame["daily_return"] = frame["equity"].pct_change().fillna(0.0)
    return frame.sort_values("date").reset_index(drop=True)


def _variant_perf(performance: pd.DataFrame, variant: str) -> dict[str, float]:
    row = performance[performance["variant_id"].astype(str).eq(variant) & performance["period"].astype(str).eq("full")]
    if row.empty:
        row = performance[performance["variant_id"].astype(str).eq(variant)]
    if row.empty:
        return {}
    return row.iloc[0].to_dict()


def _period_metrics(frame: pd.DataFrame, period: str, start: str | None = None, end: str | None = None) -> dict[str, Any]:
    subset = frame.copy()
    if start:
        subset = subset[subset["date"] >= pd.Timestamp(start)]
    if end:
        subset = subset[subset["date"] <= pd.Timestamp(end)]
    if subset.empty:
        return {"period": period, "status": "empty"}
    start_equity = float(subset["equity"].iloc[0])
    final_equity = float(subset["equity"].iloc[-1])
    running_max = subset["equity"].cummax()
    drawdown = subset["equity"] / running_max - 1.0
    monthly = subset.set_index("date")["equity"].resample("ME").last().pct_change().dropna()
    return {
        "period": period,
        "start_date": subset["date"].iloc[0].strftime("%Y-%m-%d"),
        "end_date": subset["date"].iloc[-1].strftime("%Y-%m-%d"),
        "start_equity": round(start_equity, 2),
        "final_equity": round(final_equity, 2),
        "total_return_pct": round((final_equity / start_equity - 1) * 100, 4) if start_equity else 0.0,
        "mdd_pct": round(float(drawdown.min()) * 100, 4),
        "worst_month_pct": round(float(monthly.min()) * 100, 4) if not monthly.empty else 0.0,
        "monthly_hit_rate": round(float((monthly > 0).mean()), 4) if not monthly.empty else 0.0,
    }


def _oos_period_panel(baseline: pd.DataFrame, best: pd.DataFrame) -> pd.DataFrame:
    periods = [
        ("full", None, None),
        ("train_2022_2024", "2022-01-01", "2024-12-31"),
        ("test_2025_2026", "2025-01-01", None),
        ("pre_2026", None, "2025-12-31"),
        ("2024_now", "2024-01-01", None),
        ("2024", "2024-01-01", "2024-12-31"),
        ("2025", "2025-01-01", "2025-12-31"),
        ("2026_ytd", "2026-01-01", None),
        ("recent_90d", (best["date"].max() - pd.Timedelta(days=130)).strftime("%Y-%m-%d"), None),
    ]
    rows = []
    for period, start, end in periods:
        b = _period_metrics(baseline, period, start, end)
        v = _period_metrics(best, period, start, end)
        if b.get("status") == "empty" or v.get("status") == "empty":
            continue
        rows.append(
            {
                "period": period,
                "baseline_return_pct": b["total_return_pct"],
                "best_variant_return_pct": v["total_return_pct"],
                "delta_return_pct": round(v["total_return_pct"] - b["total_return_pct"], 4),
                "baseline_mdd_pct": b["mdd_pct"],
                "best_variant_mdd_pct": v["mdd_pct"],
                "delta_mdd_pct": round(v["mdd_pct"] - b["mdd_pct"], 4),
                "baseline_worst_month_pct": b["worst_month_pct"],
                "best_worst_month_pct": v["worst_month_pct"],
                "baseline_monthly_hit_rate": b["monthly_hit_rate"],
                "best_monthly_hit_rate": v["monthly_hit_rate"],
                "start_date": v["start_date"],
                "end_date": v["end_date"],
                "mdd_gate_pass": (v["mdd_pct"] - b["mdd_pct"]) >= -3.0,
            }
        )
    return pd.DataFrame(rows)


def _contribution_by_month(monthly: pd.DataFrame) -> pd.DataFrame:
    pivot = monthly[monthly["variant_id"].isin([BASELINE_VARIANT, BEST_VARIANT])].pivot_table(
        index="month",
        columns="variant_id",
        values="monthly_return_pct",
        aggfunc="first",
    )
    pivot = pivot.fillna(0.0).reset_index()
    pivot["delta_monthly_return_pct"] = pivot.get(BEST_VARIANT, 0.0) - pivot.get(BASELINE_VARIANT, 0.0)
    pivot["quarter"] = pivot["month"].astype(str).str.slice(0, 4) + "Q" + (
        ((pd.to_datetime(pivot["month"].astype(str) + "-01").dt.month - 1) // 3 + 1).astype(str)
    )
    return pivot.rename(columns={BASELINE_VARIANT: "baseline_monthly_return_pct", BEST_VARIANT: "best_monthly_return_pct"}).sort_values(
        "delta_monthly_return_pct", ascending=False
    )


def _contribution_by_quarter(monthly_contrib: pd.DataFrame) -> pd.DataFrame:
    if monthly_contrib.empty:
        return pd.DataFrame()
    grouped = monthly_contrib.groupby("quarter", as_index=False).agg(
        baseline_return_sum_pct=("baseline_monthly_return_pct", "sum"),
        best_return_sum_pct=("best_monthly_return_pct", "sum"),
        delta_return_sum_pct=("delta_monthly_return_pct", "sum"),
    )
    return grouped.sort_values("delta_return_sum_pct", ascending=False)


def _contribution_by_ticker(event_usage: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    usage = event_usage[event_usage["variant_id"].astype(str).eq(BEST_VARIANT)].copy()
    if usage.empty:
        return usage
    sleeve_trades = trades[
        trades["variant_id"].astype(str).eq(BEST_VARIANT)
        & trades["reason"].astype(str).str.contains("pullback|reclaim|peer", case=False, na=False)
    ].copy()
    sleeve_trades["notional"] = pd.to_numeric(sleeve_trades["notional"], errors="coerce").fillna(0.0)
    sleeve_trades["cost"] = pd.to_numeric(sleeve_trades["cost"], errors="coerce").fillna(0.0)
    trade_summary = sleeve_trades.groupby("ticker", as_index=False).agg(
        sleeve_trade_legs=("ticker", "size"),
        sleeve_notional=("notional", "sum"),
        sleeve_cost=("cost", "sum"),
    )
    merged = usage.merge(trade_summary, on="ticker", how="left").fillna(0)
    total_entries = float(pd.to_numeric(merged["entry_count"], errors="coerce").sum())
    merged["entry_share"] = pd.to_numeric(merged["entry_count"], errors="coerce") / total_entries if total_entries else 0.0
    merged["contribution_boundary"] = "entry/notional attribution; Experiments must validate PnL leave-one rerun"
    return merged.sort_values(["entry_count", "sleeve_notional"], ascending=False)


def _ablation_panel(
    best_perf: dict[str, Any],
    baseline_perf: dict[str, Any],
    by_ticker: pd.DataFrame,
    by_month: pd.DataFrame,
    by_quarter: pd.DataFrame,
) -> pd.DataFrame:
    full_delta = float(best_perf.get("delta_return_vs_baseline_pct", 0.0) or 0.0)
    rows = []
    for label, frame, key, col in [
        ("exclude_top_ticker_proxy", by_ticker, "ticker", "entry_share"),
        ("exclude_top_month_proxy", by_month, "month", "delta_monthly_return_pct"),
        ("exclude_top_quarter_proxy", by_quarter, "quarter", "delta_return_sum_pct"),
    ]:
        if frame.empty:
            continue
        top = frame.iloc[0]
        contribution = abs(float(top.get(col, 0.0) or 0.0))
        if col == "entry_share":
            estimated_remaining = full_delta * (1 - contribution)
        else:
            estimated_remaining = full_delta - contribution
        rows.append(
            {
                "ablation_id": label,
                "removed_key": top.get(key, ""),
                "removed_proxy_contribution": round(contribution, 4),
                "full_delta_return_pct": round(full_delta, 4),
                "estimated_remaining_delta_pct": round(estimated_remaining, 4),
                "requires_experiments_rerun": True,
                "diagnostic_boundary": "proxy ablation, not a recomputed portfolio ledger",
                "pass_proxy": estimated_remaining > 0,
            }
        )
    return pd.DataFrame(rows)


def _cost_turnover_panel(performance: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    rows = []
    perf = performance[performance["variant_id"].isin([BASELINE_VARIANT, BEST_VARIANT]) & performance["period"].eq("full")]
    for item in perf.to_dict(orient="records"):
        variant = item["variant_id"]
        variant_trades = trades[trades["variant_id"].astype(str).eq(variant)]
        rows.append(
            {
                "variant_id": variant,
                "trade_legs": int(item.get("trade_legs", len(variant_trades)) or 0),
                "total_cost": round(float(item.get("total_cost", 0.0) or 0.0), 2),
                "turnover": round(float(item.get("turnover", 0.0) or 0.0), 2),
                "return_pct": round(float(item.get("total_return_pct", 0.0) or 0.0), 4),
                "cost_per_trade_leg": round(float(item.get("total_cost", 0.0) or 0.0) / max(int(item.get("trade_legs", 0) or 0), 1), 2),
                "diagnostic_only": True,
            }
        )
    frame = pd.DataFrame(rows)
    if len(frame) >= 2:
        baseline_cost = float(frame[frame["variant_id"].eq(BASELINE_VARIANT)]["total_cost"].iloc[0])
        best_cost = float(frame[frame["variant_id"].eq(BEST_VARIANT)]["total_cost"].iloc[0])
        edge = float(perf[perf["variant_id"].eq(BEST_VARIANT)]["delta_return_vs_baseline_pct"].iloc[0])
        frame["incremental_cost_vs_baseline"] = frame["total_cost"] - baseline_cost
        frame["cost_increase_twd"] = best_cost - baseline_cost
        frame["full_period_edge_pct"] = edge
    return frame


def _risk_panel(baseline: pd.DataFrame, best: pd.DataFrame, monthly_contrib: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant, frame in [(BASELINE_VARIANT, baseline), (BEST_VARIANT, best)]:
        metrics = _period_metrics(frame, "full")
        rows.append(
            {
                "variant_id": variant,
                "mdd_pct": metrics["mdd_pct"],
                "worst_month_pct": metrics["worst_month_pct"],
                "monthly_hit_rate": metrics["monthly_hit_rate"],
                "worst_day_pct": round(float(frame["daily_return"].min()) * 100, 4),
                "diagnostic_only": True,
            }
        )
    return pd.DataFrame(rows)


def _formal_overlap_conflict_audit(overlap: pd.DataFrame) -> pd.DataFrame:
    frame = overlap[overlap["variant_id"].astype(str).eq(BEST_VARIANT)].copy()
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "scope",
                "entry_count",
                "same_as_formal_count",
                "invalid_scope_count",
                "conflict_status",
                "diagnostic_only",
            ]
        )
    valid_scope = frame["scope"].astype(str).eq("cash_or_market_exposure")
    grouped = frame.groupby("scope", as_index=False).agg(
        entry_count=("ticker", "size"),
        same_as_formal_count=("same_as_formal", lambda s: int(pd.Series(s).map(_truthy).sum())),
    )
    grouped["invalid_scope_count"] = 0
    grouped.loc[~grouped["scope"].astype(str).eq("cash_or_market_exposure"), "invalid_scope_count"] = grouped["entry_count"]
    grouped["conflict_status"] = grouped["invalid_scope_count"].apply(lambda value: "pass" if int(value) == 0 else "fail")
    grouped["diagnostic_only"] = True
    return grouped


def _baseline_vs_best_periods(performance: pd.DataFrame, oos: pd.DataFrame) -> pd.DataFrame:
    full = performance[performance["variant_id"].isin([BASELINE_VARIANT, BEST_VARIANT])].copy()
    return pd.concat([full, oos], ignore_index=True, sort=False)


def _leave_one_event_summary(overlap: pd.DataFrame) -> pd.DataFrame:
    frame = overlap[overlap["variant_id"].astype(str).eq(BEST_VARIANT)].copy()
    frame = frame.reset_index(drop=True)
    if frame.empty:
        return pd.DataFrame()
    frame["event_id"] = frame.index + 1
    frame["leave_one_type"] = "event"
    frame["requires_experiments_rerun"] = True
    frame["diagnostic_boundary"] = "event removal list for Experiments rerun; Core does not recompute portfolio PnL here"
    return frame[["leave_one_type", "event_id", "entry_date", "ticker", "candidate_source", "formal_target", "requires_experiments_rerun", "diagnostic_boundary"]]


def _leave_one_summary(frame: pd.DataFrame, key: str) -> pd.DataFrame:
    if frame.empty or key not in frame.columns:
        return pd.DataFrame()
    out = frame.copy()
    contribution_col = "delta_monthly_return_pct" if key == "month" else "delta_return_sum_pct" if key == "quarter" else "entry_share"
    out["leave_one_type"] = key
    out["removed_key"] = out[key]
    out["removed_proxy_contribution"] = pd.to_numeric(out.get(contribution_col, 0), errors="coerce").fillna(0.0)
    out["requires_experiments_rerun"] = True
    out["diagnostic_boundary"] = "proxy contribution panel; Experiments must validate with recomputed ledger"
    return out


def _cost_slippage_sensitivity(best_perf: dict[str, Any], baseline_perf: dict[str, Any], cost_turnover: pd.DataFrame) -> pd.DataFrame:
    best_row = cost_turnover[cost_turnover["variant_id"].eq(BEST_VARIANT)]
    baseline_row = cost_turnover[cost_turnover["variant_id"].eq(BASELINE_VARIANT)]
    if best_row.empty or baseline_row.empty:
        return pd.DataFrame()
    edge = float(best_perf.get("delta_return_vs_baseline_pct", 0.0) or 0.0)
    incremental_cost = float(best_row["total_cost"].iloc[0] - baseline_row["total_cost"].iloc[0])
    initial_capital = float(best_perf.get("start_equity", 1_000_000.0) or 1_000_000.0)
    rows = []
    for factor in [1.0, 1.5, 2.0]:
        extra_cost_pct = ((factor - 1.0) * incremental_cost / initial_capital) * 100
        remaining_edge = edge - extra_cost_pct
        rows.append(
            {
                "scenario": f"{factor}x_cost",
                "cost_multiplier": factor,
                "incremental_cost_twd": round(incremental_cost * factor, 2),
                "extra_cost_vs_current_pct": round(extra_cost_pct, 4),
                "full_period_edge_pct_before_sensitivity": round(edge, 4),
                "estimated_remaining_edge_pct": round(remaining_edge, 4),
                "sensitivity_pass": remaining_edge > 0,
                "diagnostic_boundary": "cost-only sensitivity approximation; no price slippage replay",
            }
        )
    for slippage_bp in [5, 10]:
        rows.append(
            {
                "scenario": f"next_day_slippage_{slippage_bp}bp_proxy",
                "cost_multiplier": 1.0,
                "incremental_cost_twd": round(incremental_cost, 2),
                "extra_cost_vs_current_pct": "",
                "full_period_edge_pct_before_sensitivity": round(edge, 4),
                "estimated_remaining_edge_pct": "",
                "sensitivity_pass": "",
                "diagnostic_boundary": "requires Experiments price-level rerun to apply slippage proxy",
            }
        )
    return pd.DataFrame(rows)


def _pool1b_vs_old_ai_contribution(by_ticker: pd.DataFrame) -> pd.DataFrame:
    if by_ticker.empty:
        return pd.DataFrame()
    return by_ticker.groupby("candidate_source", as_index=False).agg(
        entry_count=("entry_count", "sum"),
        sleeve_trade_legs=("sleeve_trade_legs", "sum"),
        sleeve_notional=("sleeve_notional", "sum"),
        sleeve_cost=("sleeve_cost", "sum"),
    )


def _top_contribution_events(overlap: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    frame = overlap[overlap["variant_id"].astype(str).eq(BEST_VARIANT)].copy()
    if frame.empty:
        return pd.DataFrame()
    trade_entries = trades[
        trades["variant_id"].astype(str).eq(BEST_VARIANT)
        & trades["reason"].astype(str).str.contains("pullback|reclaim|peer", case=False, na=False)
        & trades["side"].astype(str).eq("buy")
    ].copy()
    trade_entries["notional"] = pd.to_numeric(trade_entries["notional"], errors="coerce").fillna(0.0)
    joined = frame.merge(
        trade_entries[["date", "ticker", "notional", "cost", "reason"]],
        left_on=["entry_date", "ticker"],
        right_on=["date", "ticker"],
        how="left",
    ).fillna("")
    joined["notional"] = pd.to_numeric(joined["notional"], errors="coerce").fillna(0.0)
    joined["cost"] = pd.to_numeric(joined["cost"], errors="coerce").fillna(0.0)
    return joined.sort_values("notional", ascending=False).head(50)


def _readiness(
    *,
    manifest: dict[str, Any],
    contract_readiness: dict[str, Any],
    oos: pd.DataFrame,
    ablation: pd.DataFrame,
    cost_sensitivity: pd.DataFrame,
    formal_conflict: pd.DataFrame,
    contribution_ticker: pd.DataFrame,
    material: pd.DataFrame,
    case_6488: pd.DataFrame,
) -> dict[str, Any]:
    full_row = oos[oos["period"].eq("full")].iloc[0].to_dict() if not oos[oos["period"].eq("full")].empty else {}
    top_ticker_value = (
        pd.to_numeric(contribution_ticker.get("entry_share", pd.Series(dtype=float)), errors="coerce").max()
        if not contribution_ticker.empty
        else 0.0
    )
    top_ticker_share = 0.0 if pd.isna(top_ticker_value) else float(top_ticker_value)
    cost_15 = cost_sensitivity[cost_sensitivity["scenario"].eq("1.5x_cost")]
    return {
        "schema_version": 1,
        "task_id": TASK_ID,
        "status": "completed_robustness_panels_ready_for_experiments_validation",
        "source_task_id": manifest.get("task_id", ""),
        "best_variant": BEST_VARIANT,
        "baseline_variant": BASELINE_VARIANT,
        "full_delta_return_pct": round(float(full_row.get("delta_return_pct", 0.0) or 0.0), 4),
        "full_delta_mdd_pct": round(float(full_row.get("delta_mdd_pct", 0.0) or 0.0), 4),
        "top_ticker_entry_share": round(top_ticker_share, 4),
        "top_ticker_concentration_guard_pass": top_ticker_share < 0.40,
        "cost_1_5x_guard_pass": bool(cost_15["sensitivity_pass"].iloc[0]) if not cost_15.empty else False,
        "formal_conflict_guard_pass": not formal_conflict.empty
        and int(pd.to_numeric(formal_conflict["invalid_scope_count"], errors="coerce").fillna(0).sum()) == 0,
        "ablation_proxy_all_positive": bool(ablation["pass_proxy"].astype(bool).all()) if not ablation.empty else False,
        "material_layer_case_only": True,
        "material_layer_case_rows": int(len(material)),
        "case_6488_two_case_only": True,
        "case_6488_two_rows": int(len(case_6488)),
        "ready_for_experiments_robustness_validation": True,
        "ready_for_formal_absorption": False,
        "diagnostic_only": True,
        "formal_model_changed": False,
        "trade_decision_changed": False,
        "active_in_trade_decision": False,
        "uses_forward_return_as_live_rule": False,
        "future_data_violation_count": int(manifest.get("future_data_violation_count", 0) or 0),
        "contract_event_rows": int(contract_readiness.get("eligible_event_rows", 0) or 0),
    }


def _manifest(output: Path, source: Path, contract: Path, readiness: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "task_id": TASK_ID,
        "status": readiness["status"],
        "generated_at": pd.Timestamp.now(tz="Asia/Taipei").isoformat(),
        "output_dir": str(output),
        "experiments_validation_output": str(source),
        "core_contract_output": str(contract),
        **readiness,
    }


def _overfit_guard(readiness: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Overfit Guard",
            "",
            f"- Best variant: `{BEST_VARIANT}`.",
            f"- Full delta return: {readiness['full_delta_return_pct']} pp.",
            f"- Full delta MDD: {readiness['full_delta_mdd_pct']} pp.",
            f"- Top ticker entry share: {readiness['top_ticker_entry_share']:.2%}.",
            f"- Top ticker concentration guard pass: {readiness['top_ticker_concentration_guard_pass']}.",
            f"- 1.5x cost guard pass: {readiness['cost_1_5x_guard_pass']}.",
            f"- Formal conflict guard pass: {readiness['formal_conflict_guard_pass']}.",
            "",
            "These are diagnostic panels. Experiments must rerun leave-one ledgers before any later challenger decision.",
        ]
    )


def _summary_zh(readiness: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Short-cycle pullback robustness panels",
            "",
            f"- 狀態：{readiness['status']}",
            f"- 最佳 variant：{readiness['best_variant']}",
            f"- full delta return：{readiness['full_delta_return_pct']}pp",
            f"- full delta MDD：{readiness['full_delta_mdd_pct']}pp",
            f"- top ticker entry share：{readiness['top_ticker_entry_share']:.2%}",
            f"- 1.5x cost guard：{readiness['cost_1_5x_guard_pass']}",
            f"- formal conflict guard：{readiness['formal_conflict_guard_pass']}",
            "- 結論：可交 Experiments 做 robustness validation；仍是 diagnostic-only，不可 formal absorption。",
        ]
    )


def _read_csv_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path).fillna("")


def _read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path).fillna("")
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build short-cycle pullback robustness panels.")
    parser.add_argument("--experiments-output", default=DEFAULT_EXPERIMENTS_OUTPUT)
    parser.add_argument("--contract-output", default=DEFAULT_CONTRACT_OUTPUT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    run_short_cycle_pullback_robustness_panels(
        experiments_output=args.experiments_output,
        contract_output=args.contract_output,
        output_dir=args.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
