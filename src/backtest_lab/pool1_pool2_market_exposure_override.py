from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from backtest_lab.pool1_pool2_veto_cap_downweight import (
    INITIAL_CASH,
    VariantSpec,
    _forward_return,
    _load_prices,
    _price_on_or_before,
    _round,
    _simulate_weighted_variant,
    _text,
)


DEFAULT_SOURCE_DIR = "outputs/pool1_pool2_veto_cap_downweight_panels_20260626"
DEFAULT_BLOCKER_DIR = "outputs/pool1_pool2_final_challenger_blocker_panels_20260626"
DEFAULT_PRICE_CACHE_DIR = "backtest_cache/stock_pool_triad_v1_corrected"
DEFAULT_OUTPUT_DIR = "outputs/pool1_pool2_market_exposure_override_panels_20260626"
BASE_VARIANT = "combined_cap40_confirmation1"
HORIZONS = (20, 60, 120)


@dataclass(frozen=True)
class OverrideSpec:
    variant: str
    mode: str
    active_in_trade_decision: bool = False


VARIANTS = [
    OverrideSpec("combined_cap40_confirmation1_base", "base"),
    OverrideSpec("combined_cap40_confirmation1_0050x2_opportunity_cost_label_only", "label_only"),
    OverrideSpec("market_exposure_override_0050x2_when_hard_gate_opportunity_cost", "override_opportunity_cost"),
    OverrideSpec("market_exposure_override_0050x2_when_pool1_pool2_agree_market_on", "override_pool_agree_market_on"),
    OverrideSpec("market_exposure_override_0050x2_when_00631L_capped_and_0050x2_momentum_confirmed", "override_capped_momentum"),
    OverrideSpec("cap40_relax_to_00631L_when_market_on_confirmed", "relax_market_on"),
    OverrideSpec("cap40_relax_to_00631L_when_0050x2_outperforms_pool_target_diagnostic", "relax_opportunity_diagnostic"),
    OverrideSpec("override_warning_only", "warning_only"),
    OverrideSpec("override_blocked_when_drawdown_or_fake_signal", "override_guarded"),
    OverrideSpec("no_override_control", "base"),
]


def run_pool1_pool2_market_exposure_override(
    *,
    source_dir: str | Path = DEFAULT_SOURCE_DIR,
    blocker_dir: str | Path = DEFAULT_BLOCKER_DIR,
    price_cache_dir: str | Path = DEFAULT_PRICE_CACHE_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    initial_cash: float = INITIAL_CASH,
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
        blocker = Path(blocker_dir)
        log("load_inputs", "started", str(source))
        base_daily = pd.read_csv(source / "daily_equity_by_variant.csv").fillna("")
        base_events = pd.read_csv(source / "pool2_disagreement_variant_events.csv").fillna("")
        cap_trigger = pd.read_csv(blocker / "cap40_trigger_event_panel.csv").fillna("")
        base_panel = base_events[base_events["variant"].eq(BASE_VARIANT)].copy()
        prices = _load_prices(_needed_tickers(base_panel), Path(price_cache_dir))
        if "0050.TW" not in prices or "00631L.TW" not in prices:
            raise ValueError("0050.TW and 00631L.TW prices are required")
        base_drawdown = base_daily[base_daily["variant"].eq(BASE_VARIANT)].set_index("date")["drawdown"].to_dict()
        cap_dates = set(cap_trigger["date"].astype(str).tolist())

        log("simulate_variants", "started", "")
        daily_frames: list[pd.DataFrame] = []
        trade_frames: list[pd.DataFrame] = []
        event_frames: list[pd.DataFrame] = []
        for spec in VARIANTS:
            target_panel, audit = _build_variant_target_panel(base_panel, spec, prices, cap_dates, base_drawdown)
            daily, trades, events = _simulate_weighted_variant(target_panel, prices, VariantSpec(spec.variant, "warning_only"), initial_cash)
            daily_frames.append(daily)
            trade_frames.append(trades)
            event_frames.append(audit)
        daily_equity = pd.concat(daily_frames, ignore_index=True)
        trade_ledger = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()
        override_events = pd.concat(event_frames, ignore_index=True)

        log("build_reports", "started", "")
        perf = _period_performance(daily_equity)
        hard_gate = _hard_gate_2024(perf)
        label_panel = _label_panel(override_events)
        cap_relax = override_events[override_events["variant"].astype(str).str.startswith("cap40_relax")].copy()
        cap_attr = _cap_relax_attribution(cap_relax, prices)
        benchmark = _benchmark_comparison(perf, prices)
        oos = _oos(daily_equity)
        leave_one = _leave_one(daily_equity)
        guard = _guard_report(override_events, daily_equity)
        execution = _execution_stability(daily_equity)
        comparison = _override_vs_label(perf)

        log("write_outputs", "started", "")
        _variant_matrix().to_csv(output / "variant_parameter_matrix.csv", index=False, encoding="utf-8-sig")
        daily_equity.to_csv(output / "daily_equity_by_variant.csv", index=False, encoding="utf-8-sig")
        trade_ledger.to_csv(output / "trade_ledger_by_variant.csv", index=False, encoding="utf-8-sig")
        perf.to_csv(output / "period_performance_by_variant.csv", index=False, encoding="utf-8-sig")
        hard_gate.to_csv(output / "hard_gate_2024_override_attribution.csv", index=False, encoding="utf-8-sig")
        override_events.to_csv(output / "override_event_panel.csv", index=False, encoding="utf-8-sig")
        override_events.to_csv(output / "override_trigger_audit.csv", index=False, encoding="utf-8-sig")
        comparison.to_csv(output / "override_vs_label_only_comparison.csv", index=False, encoding="utf-8-sig")
        label_panel.to_csv(output / "0050x2_opportunity_cost_label_panel.csv", index=False, encoding="utf-8-sig")
        cap_relax.to_csv(output / "cap40_relaxation_event_panel.csv", index=False, encoding="utf-8-sig")
        cap_attr.to_csv(output / "cap40_relaxation_attribution.csv", index=False, encoding="utf-8-sig")
        benchmark.to_csv(output / "market_exposure_benchmark_comparison.csv", index=False, encoding="utf-8-sig")
        oos.to_csv(output / "oos_walk_forward_by_variant.csv", index=False, encoding="utf-8-sig")
        leave_one.to_csv(output / "leave_one_period_by_variant.csv", index=False, encoding="utf-8-sig")
        guard.to_csv(output / "overfit_guard_report.csv", index=False, encoding="utf-8-sig")
        execution.to_csv(output / "execution_stability_by_variant.csv", index=False, encoding="utf-8-sig")
        (output / "market_exposure_override_summary_zh.md").write_text(_summary(perf, guard), encoding="utf-8")
        manifest = {
            "schema_version": 1,
            "task_id": "TASK-BACKTEST-CORE-POOL1-POOL2-MARKET-EXPOSURE-OVERRIDE-PANELS-001",
            "model": "pool1_pool2_market_exposure_override",
            "status": "completed",
            "latest_complete_common_date": str(daily_equity["date"].max()),
            "formal_model_changed": False,
            "trade_decision_changed": False,
            "formal_absorption_ready": False,
            "pool3_shadow_used_as_formal": False,
            "final_decision_label_used_as_formal": False,
            "rr_partial_switch_used_in_performance": False,
            "valuation_used": False,
            "h3_used": False,
            "uses_forward_return_as_rule": False,
            "opportunity_cost_label_active_in_trade_decision": False,
        }
        (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        pd.DataFrame([{"status": "completed", "output_dir": str(output.resolve())}]).to_csv(output / "completed.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(columns=["step", "error"]).to_csv(output / "failed.csv", index=False, encoding="utf-8-sig")
        log("completed", "completed", str(output.resolve()))
        (output / "current_step.txt").write_text("completed\n", encoding="utf-8")
        return output
    except Exception as exc:
        pd.DataFrame([{"step": "run_pool1_pool2_market_exposure_override", "error": str(exc)}]).to_csv(output / "failed.csv", index=False, encoding="utf-8-sig")
        log("failed", "failed", str(exc))
        raise


def _needed_tickers(panel: pd.DataFrame) -> list[str]:
    tickers = {"0050.TW", "00631L.TW"}
    for column in ("pool1_vote", "pool2_vote"):
        tickers.update(_text(value) for value in panel[column].tolist() if _text(value))
    return sorted(tickers)


def _build_variant_target_panel(panel: pd.DataFrame, spec: OverrideSpec, prices: dict[str, pd.Series], cap_dates: set[str], base_drawdown: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    audit = []
    for item in panel.to_dict(orient="records"):
        date = _text(item.get("date"))
        base_weights = _parse_weights(item.get("target_weights"))
        signal = _pit_signal(date, _text(item.get("pool1_vote")), prices)
        capped = date in cap_dates
        guarded = float(base_drawdown.get(date, 0) or 0) <= -0.1 or not signal["market_on"]
        weights = dict(base_weights)
        trigger = False
        reason = "base"
        if spec.mode == "label_only":
            reason = "opportunity_cost_label_only"
        elif spec.mode == "warning_only":
            reason = "override_warning_only"
        elif spec.mode == "override_opportunity_cost" and signal["0050x2_momentum_advantage"]:
            weights = {"00631L.TW": 1.0}
            trigger = True
            reason = "pit_0050x2_momentum_advantage"
        elif spec.mode == "override_pool_agree_market_on" and _text(item.get("pool1_vote")) == _text(item.get("pool2_vote")) and signal["market_on"]:
            weights = {"00631L.TW": 1.0}
            trigger = True
            reason = "pool1_pool2_agree_market_on"
        elif spec.mode == "override_capped_momentum" and capped and signal["00631L_momentum_confirmed"]:
            weights = {"00631L.TW": 1.0}
            trigger = True
            reason = "cap_trigger_and_00631L_momentum_confirmed"
        elif spec.mode == "relax_market_on" and capped and signal["market_on"]:
            weights = {"00631L.TW": 1.0}
            trigger = True
            reason = "cap40_relax_market_on"
        elif spec.mode == "relax_opportunity_diagnostic" and capped and signal["0050x2_momentum_advantage"]:
            weights = {"00631L.TW": 1.0}
            trigger = True
            reason = "cap40_relax_opportunity_diagnostic"
        elif spec.mode == "override_guarded" and signal["0050x2_momentum_advantage"] and not guarded:
            weights = {"00631L.TW": 1.0}
            trigger = True
            reason = "guard_passed_override"
        elif spec.mode == "override_guarded" and signal["0050x2_momentum_advantage"]:
            reason = "blocked_by_drawdown_or_fake_signal_guard"
        rows.append({**item, "variant": spec.variant, "target_weights": json.dumps(weights, ensure_ascii=False), "event_reason": reason})
        audit.append({**item, "variant": spec.variant, "triggered": trigger, "trigger_reason": reason, "cap40_triggered": capped, **signal, "uses_forward_return_as_rule": False, "opportunity_cost_label_active_in_trade_decision": False})
    return pd.DataFrame(rows), pd.DataFrame(audit)


def _parse_weights(value: Any) -> dict[str, float]:
    if not _text(value):
        return {}
    try:
        parsed = json.loads(str(value))
        return {str(k): float(v) for k, v in parsed.items()}
    except Exception:
        return {}


def _pit_signal(date: str, pool_target: str, prices: dict[str, pd.Series]) -> dict[str, Any]:
    m20_0050 = _momentum(prices["0050.TW"], date, 20)
    m60_0050 = _momentum(prices["0050.TW"], date, 60)
    m20_631 = _momentum(prices["00631L.TW"], date, 20)
    m60_631 = _momentum(prices["00631L.TW"], date, 60)
    target_m60 = _momentum(prices.get(pool_target, pd.Series(dtype=float)), date, 60) if pool_target else None
    ma60 = _ma(prices["00631L.TW"], date, 60)
    px631 = _price_on_or_before(prices["00631L.TW"], date)
    return {
        "market_on": bool((m60_0050 or 0) > 0 and (px631 is not None and ma60 is not None and px631 >= ma60)),
        "0050x2_momentum_advantage": bool(m60_0050 is not None and target_m60 is not None and m60_0050 * 2 > target_m60),
        "00631L_momentum_confirmed": bool(m60_631 is not None and m20_631 is not None and m60_631 > 0 and m20_631 > -0.05),
        "0050_momentum_20d": _round(m20_0050),
        "0050_momentum_60d": _round(m60_0050),
        "00631L_momentum_20d": _round(m20_631),
        "00631L_momentum_60d": _round(m60_631),
        "pool_target_momentum_60d": _round(target_m60),
    }


def _momentum(series: pd.Series, date: str, lookback: int) -> float | None:
    clipped = series.loc[series.index <= pd.Timestamp(date)]
    if len(clipped) <= lookback:
        return None
    return float(clipped.iloc[-1] / clipped.iloc[-lookback - 1] - 1)


def _ma(series: pd.Series, date: str, window: int) -> float | None:
    clipped = series.loc[series.index <= pd.Timestamp(date)]
    if len(clipped) < window:
        return None
    return float(clipped.tail(window).mean())


def _variant_matrix() -> pd.DataFrame:
    return pd.DataFrame([spec.__dict__ for spec in VARIANTS])


def _period_performance(daily: pd.DataFrame) -> pd.DataFrame:
    periods = {"2022": ("2022-01-01", "2022-12-31"), "2023": ("2023-01-01", "2023-12-31"), "2024_now": ("2024-01-01", None), "2024_hard_gate": ("2024-01-01", "2024-12-31"), "full": (None, None)}
    rows = []
    frame = daily.copy()
    frame["date_ts"] = pd.to_datetime(frame["date"])
    for variant, group in frame.groupby("variant"):
        for label, (start, end) in periods.items():
            subset = group
            if start:
                subset = subset[subset["date_ts"] >= pd.Timestamp(start)]
            if end:
                subset = subset[subset["date_ts"] <= pd.Timestamp(end)]
            rows.append(_perf_row(variant, label, subset))
    return pd.DataFrame(rows)


def _perf_row(variant: str, label: str, frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {"variant": variant, "period_label": label, "status": "empty"}
    start = float(frame["equity"].iloc[0])
    end = float(frame["equity"].iloc[-1])
    return {"variant": variant, "period_label": label, "status": "completed", "start_date": frame["date"].iloc[0], "end_date": frame["date"].iloc[-1], "return_pct": round((end / start - 1) * 100, 4), "max_drawdown_pct": round(float(pd.to_numeric(frame["drawdown"], errors="coerce").min()) * 100, 4), "trade_days": int(frame["action"].astype(str).ne("hold").sum()), "total_cost": round(float(pd.to_numeric(frame["transaction_cost"], errors="coerce").sum()), 2)}


def _hard_gate_2024(perf: pd.DataFrame) -> pd.DataFrame:
    return perf[perf["period_label"].eq("2024_hard_gate")].copy()


def _label_panel(events: pd.DataFrame) -> pd.DataFrame:
    return events[events["variant"].eq("combined_cap40_confirmation1_0050x2_opportunity_cost_label_only")].copy()


def _cap_relax_attribution(events: pd.DataFrame, prices: dict[str, pd.Series]) -> pd.DataFrame:
    rows = []
    for item in events[events["triggered"].map(_truthy)].to_dict(orient="records"):
        row = {"variant": item["variant"], "date": item["date"], "trigger_reason": item["trigger_reason"], "uses_forward_return_as_rule": False}
        for horizon in HORIZONS:
            row[f"00631L_forward_{horizon}d"] = _round(_forward_return(prices.get("00631L.TW"), item["date"], horizon))
        rows.append(row)
    return pd.DataFrame(rows)


def _benchmark_comparison(perf: pd.DataFrame, prices: dict[str, pd.Series]) -> pd.DataFrame:
    rows = []
    for item in perf.to_dict(orient="records"):
        if item.get("status") != "completed":
            continue
        for label, ticker, mult in (("0050", "0050.TW", 1.0), ("00631L", "00631L.TW", 1.0), ("0050x2", "0050.TW", 2.0)):
            bench = _bench_return(prices[ticker], item["start_date"], item["end_date"], mult)
            rows.append({"variant": item["variant"], "period_label": item["period_label"], "benchmark": label, "variant_return_pct": item["return_pct"], "benchmark_return_pct": _pct(bench), "excess_return_pct": _pct(item["return_pct"] / 100 - bench if bench is not None else None)})
    return pd.DataFrame(rows)


def _bench_return(series: pd.Series, start: str, end: str, mult: float) -> float | None:
    start_px = _price_on_or_before(series, start)
    end_px = _price_on_or_before(series, end)
    if start_px is None or end_px is None:
        return None
    return (end_px / start_px - 1) * mult


def _oos(daily: pd.DataFrame) -> pd.DataFrame:
    tests = {"train_2022_2024": ("2022-01-01", "2024-12-31"), "test_2025_2026": ("2025-01-01", None), "test_2024_now": ("2024-01-01", None)}
    rows = []
    frame = daily.copy()
    frame["date_ts"] = pd.to_datetime(frame["date"])
    for variant, group in frame.groupby("variant"):
        for label, (start, end) in tests.items():
            subset = group[group["date_ts"] >= pd.Timestamp(start)]
            if end:
                subset = subset[subset["date_ts"] <= pd.Timestamp(end)]
            rows.append(_perf_row(variant, label, subset))
    return pd.DataFrame(rows)


def _leave_one(daily: pd.DataFrame) -> pd.DataFrame:
    frame = daily.copy()
    frame["date_ts"] = pd.to_datetime(frame["date"])
    frame["year"] = frame["date_ts"].dt.year.astype(str)
    rows = []
    for variant, group in frame.groupby("variant"):
        for year in sorted(group["year"].unique()):
            rows.append(_perf_row(variant, f"leave_one_year_{year}", group[~group["year"].eq(year)]))
    return pd.DataFrame(rows)


def _guard_report(events: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant, group in events.groupby("variant"):
        triggered = group[group["triggered"].map(_truthy)]
        rows.append({"variant": variant, "triggered_rows": len(triggered), "uses_forward_return_as_rule": bool(group["uses_forward_return_as_rule"].map(_truthy).any()), "opportunity_cost_label_active_in_trade_decision": bool(group["opportunity_cost_label_active_in_trade_decision"].map(_truthy).any()), "status": "diagnostic_only"})
    return pd.DataFrame(rows)


def _execution_stability(daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant, group in daily.groupby("variant"):
        rows.append({"variant": variant, "trade_days": int(group["action"].astype(str).ne("hold").sum()), "rapid_flip_rate": _rapid_flip_rate(group), "target_changed_3d_rate": _changed_rate(group, 3), "total_cost": round(float(pd.to_numeric(group["transaction_cost"], errors="coerce").sum()), 2)})
    return pd.DataFrame(rows)


def _override_vs_label(perf: pd.DataFrame) -> pd.DataFrame:
    full = perf[perf["period_label"].eq("full")].copy()
    base = full[full["variant"].eq("combined_cap40_confirmation1_base")]
    base_ret = None if base.empty else float(base["return_pct"].iloc[0])
    return full.assign(excess_vs_base_pct=full["return_pct"].astype(float).map(lambda value: "" if base_ret is None else round(value - base_ret, 4)))


def _rapid_flip_rate(group: pd.DataFrame) -> float:
    values = group.sort_values("date")["position_ticker"].astype(str).tolist()
    flags = [bool(value and value in values[index + 1 : index + 4] and any(item and item != value for item in values[index + 1 : index + 4])) for index, value in enumerate(values)]
    return round(sum(flags) / len(flags), 6) if flags else 0.0


def _changed_rate(group: pd.DataFrame, window: int) -> float:
    values = group.sort_values("date")["position_ticker"].astype(str).tolist()
    flags = [bool(value and any(item and item != value for item in values[index + 1 : index + window + 1])) for index, value in enumerate(values)]
    return round(sum(flags) / len(flags), 6) if flags else 0.0


def _pct(value: float | None) -> Any:
    return "" if value is None else round(value * 100, 4)


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _summary(perf: pd.DataFrame, guard: pd.DataFrame) -> str:
    lines = ["# Pool1 + Pool2 Market Exposure Override Panels", "", "本輸出只做 challenger / report-only evidence，不改正式模型。", ""]
    full = perf[perf["period_label"].eq("full")]
    for row in full.to_dict(orient="records"):
        lines.append(f"- {row['variant']}: full {row.get('return_pct')}%, MDD {row.get('max_drawdown_pct')}%")
    lines.append("")
    for row in guard.to_dict(orient="records"):
        lines.append(f"- {row['variant']}: triggers {row.get('triggered_rows')}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build market exposure override vs caveat panels.")
    parser.add_argument("--source-dir", default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--blocker-dir", default=DEFAULT_BLOCKER_DIR)
    parser.add_argument("--price-cache-dir", default=DEFAULT_PRICE_CACHE_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    output = run_pool1_pool2_market_exposure_override(source_dir=args.source_dir, blocker_dir=args.blocker_dir, price_cache_dir=args.price_cache_dir, output_dir=args.output_dir)
    print(f"OUTPUT_DIR={output.resolve()}")


if __name__ == "__main__":
    main()
