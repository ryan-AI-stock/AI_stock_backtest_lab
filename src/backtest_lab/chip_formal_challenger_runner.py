from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from backtest_lab.bear_defense_backtest import risk_on_for_rule
from backtest_lab.chip_shadow_diagnostic_adapter import DEFAULT_OUTPUT_DIR as DEFAULT_CHIP_ADAPTER_OUTPUT_DIR
from backtest_lab.frozen_strategy_engine import (
    DEFAULT_FROZEN_GROUP_ID,
    FrozenStrategyContext,
    load_frozen_strategy_context_from_cache,
    simulate_frozen_baseline,
)
from backtest_lab.regime_mode_switch import (
    MODE_DAILY,
    RegimeModeSwitchVariant,
    TargetSelectionOverlayDecision,
    _min_score_margin_for_regime,
    _passes_candidate_filter,
    frozen_cycle_proven_top1_v1_variant,
    simulate_regime_mode_switch,
)
from backtest_lab.simulation import BacktestResult
from backtest_lab.strategies import relative_strength_scores


DEFAULT_CHIP_PANEL = f"{DEFAULT_CHIP_ADAPTER_OUTPUT_DIR}/chip_diagnostic_panel.csv"
DEFAULT_OUTPUT_DIR = "outputs/chip_formal_challenger_runner_20260620"
DEFAULT_CACHE_DIR = "backtest_cache/ad_hoc_20260612_daily_targets_filled"
DEFAULT_PERIODS = {
    "2024_calibration": ("2024-01-02", "2024-12-31", "2024 calibration"),
    "2025_validation": ("2025-01-01", "2025-12-31", "2025 validation"),
    "2026_oos": ("2026-01-01", "2026-05-26", "2026 OOS"),
}
FORBIDDEN_CHIP_COLUMNS = {"day_ratio_top10", "margin_and_day_overheat_flag", "valuation_entry_block"}


@dataclass(frozen=True)
class ChipFormalVariant:
    variant_id: str
    label: str
    h1_soft_bonus: float = 0.0
    require_h1_confirmation: bool = False
    h2_score_penalty: float = 0.0
    veto_h2_pressure: bool = False
    h1_min_score: int = 1
    h2_min_score: int = 2
    use_h4_shadow_risk: bool = True


def default_chip_formal_variants() -> tuple[ChipFormalVariant, ...]:
    return (
        ChipFormalVariant("v0_noop", "V0 no-op sanity"),
        ChipFormalVariant("h1_soft_rank", "H1 soft-rank", h1_soft_bonus=0.05),
        ChipFormalVariant("h1_strict_confirmation", "H1 strict confirmation", require_h1_confirmation=True),
        ChipFormalVariant("h2_risk_veto", "H2 risk veto", h2_score_penalty=0.20, veto_h2_pressure=True),
        ChipFormalVariant(
            "h1_h2_combined",
            "H1 soft-rank + H2 veto",
            h1_soft_bonus=0.05,
            h2_score_penalty=0.20,
            veto_h2_pressure=True,
        ),
    )


def run_chip_formal_challenger_runner(
    *,
    chip_panel_path: str | Path,
    cache_dir: str | Path,
    output_dir: str | Path,
    config_path: str | Path = "configs/ep05_universe.json",
    group_id: str = DEFAULT_FROZEN_GROUP_ID,
    initial_cash: float | None = None,
    periods: dict[str, tuple[str, str, str]] | None = None,
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

    log("load_inputs", "started", "")
    chip_panel = load_chip_diagnostic_panel(chip_panel_path)
    context = load_frozen_strategy_context_from_cache(
        config_path=config_path,
        group_id=group_id,
        cache_dir=cache_dir,
    )
    selected_periods = periods or DEFAULT_PERIODS
    cash = float(initial_cash or context.config.initial_cash_twd)
    log("load_inputs", "completed", f"chip_rows={len(chip_panel)}")

    summary_rows: list[dict] = []
    diff_frames: list[pd.DataFrame] = []
    daily_frames: list[pd.DataFrame] = []
    v0_checks: list[dict] = []
    formal_variant = frozen_cycle_proven_top1_v1_variant()

    for period_id, (start_date, end_date, period_label) in selected_periods.items():
        log(f"period_{period_id}", "started", f"{start_date}~{end_date}")
        baseline = simulate_frozen_baseline(
            context=context,
            start_date=start_date,
            end_date=end_date,
            initial_cash=cash,
            name="best_v20260605",
        )
        summary_rows.append(_summary_row(period_id, period_label, "best_v20260605", "Baseline", baseline, cash))
        daily_frames.append(_daily_frame(period_id, "best_v20260605", baseline.equity_curve))

        for spec in default_chip_formal_variants():
            overlay = build_chip_target_selection_overlay(chip_panel, spec)
            result = simulate_regime_mode_switch(
                name=spec.variant_id,
                prices_by_ticker=context.prices_by_ticker,
                asset_types=context.asset_types,
                market_prices=context.prices_by_ticker["0050.TW"],
                start_date=start_date,
                end_date=end_date,
                initial_cash=cash,
                cost_model=context.config.cost_model,
                variant=formal_variant,
                dividend_series_by_ticker=context.dividends_by_ticker,
                target_selection_overlay=overlay,
            )
            summary_rows.append(_summary_row(period_id, period_label, spec.variant_id, spec.label, result, cash))
            daily_frames.append(_daily_frame(period_id, spec.variant_id, result.equity_curve))
            diff = build_decision_diff_panel(
                period_id=period_id,
                variant_id=spec.variant_id,
                variant_label=spec.label,
                baseline_curve=baseline.equity_curve,
                challenger_curve=result.equity_curve,
                chip_panel=chip_panel,
            )
            diff_frames.append(diff)
            if spec.variant_id == "v0_noop":
                v0_checks.append(_v0_alignment_check(period_id, baseline, result))
        log(f"period_{period_id}", "completed", "")

    summary = pd.DataFrame(summary_rows)
    baseline_summary = summary.loc[
        summary["variant_id"] == "best_v20260605",
        ["period_id", "total_return_pct", "max_drawdown_pct", "final_value_twd"],
    ].rename(
        columns={
            "total_return_pct": "baseline_total_return_pct",
            "max_drawdown_pct": "baseline_max_drawdown_pct",
            "final_value_twd": "baseline_final_value_twd",
        }
    )
    summary = summary.merge(baseline_summary, on="period_id", how="left")
    summary["return_diff_pct"] = (summary["total_return_pct"] - summary["baseline_total_return_pct"]).round(4)
    summary["max_drawdown_diff_pct"] = (summary["max_drawdown_pct"] - summary["baseline_max_drawdown_pct"]).round(4)
    decision_diff = pd.concat(diff_frames, ignore_index=True) if diff_frames else pd.DataFrame()
    daily = pd.concat(daily_frames, ignore_index=True) if daily_frames else pd.DataFrame()
    metadata = {
        "model": "chip_formal_challenger_runner_v1",
        "baseline": "best_v20260605 / frozen_cycle_proven_top1_v1",
        "decision_layer": "formal_challenger_replay",
        "active_in_trade_decision": False,
        "formal_model_changed": False,
        "valuation_used": False,
        "h3_used": False,
        "excluded_signals": sorted(FORBIDDEN_CHIP_COLUMNS),
        "variants": [variant.__dict__ for variant in default_chip_formal_variants()],
        "periods": selected_periods,
        "v0_noop_alignment": v0_checks,
        "v0_noop_all_aligned": all(check["aligned"] for check in v0_checks),
        "experiments_next_step": (
            "Experiments can rerun this runner, then evaluate baseline_vs_challengers and "
            "decision_diff_panel before proposing any formal promotion."
        ),
    }

    summary.to_csv(output / "baseline_vs_challengers.csv", index=False, encoding="utf-8-sig")
    decision_diff.to_csv(output / "decision_diff_panel.csv", index=False, encoding="utf-8-sig")
    daily.to_csv(output / "formal_challenger_daily.csv", index=False, encoding="utf-8-sig")
    (output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_summary(output / "final_summary_zh.md", metadata, summary, decision_diff)
    (output / "completed.txt").write_text("completed", encoding="utf-8")
    (output / "current_step.txt").write_text("completed", encoding="utf-8")
    log("completed", "completed", str(output.resolve()))
    return output


def load_chip_diagnostic_panel(path: str | Path) -> pd.DataFrame:
    panel = pd.read_csv(path)
    forbidden_present = FORBIDDEN_CHIP_COLUMNS.intersection(panel.columns)
    if forbidden_present:
        raise ValueError(f"Forbidden H3/valuation columns must not be used: {sorted(forbidden_present)}")
    required = {
        "date",
        "ticker",
        "attack_confirmation_score",
        "sell_pressure_warning_score",
        "h1_negative_or_h2_sell_pressure",
    }
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"Missing chip diagnostic columns: {sorted(missing)}")
    panel = panel.copy()
    panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()
    panel["ticker"] = panel["ticker"].astype(str)
    return panel


def build_chip_target_selection_overlay(chip_panel: pd.DataFrame, spec: ChipFormalVariant):
    lookup = _chip_lookup(chip_panel)

    def overlay(
        mode: str,
        prices_by_ticker: dict[str, pd.DataFrame],
        trade_date: pd.Timestamp,
        signal_date: pd.Timestamp,
        regime: str,
        variant: RegimeModeSwitchVariant,
        baseline_target: str | None,
        baseline_exposure: float,
    ) -> TargetSelectionOverlayDecision:
        if spec.variant_id == "v0_noop":
            return TargetSelectionOverlayDecision(
                target=baseline_target,
                reason="v0_noop_no_decision_change",
                signal_date=pd.Timestamp(signal_date).strftime("%Y-%m-%d"),
            )
        if mode != MODE_DAILY or baseline_exposure <= 0:
            return TargetSelectionOverlayDecision(
                target=baseline_target,
                reason=f"{spec.variant_id}_non_daily_or_no_exposure_no_change",
                signal_date=pd.Timestamp(signal_date).strftime("%Y-%m-%d"),
            )

        scores = _eligible_scores(prices_by_ticker, signal_date, variant)
        if not scores:
            return TargetSelectionOverlayDecision(
                target=baseline_target,
                reason=f"{spec.variant_id}_no_eligible_candidate_no_change",
                signal_date=pd.Timestamp(signal_date).strftime("%Y-%m-%d"),
            )
        fallback = _fallback_target(prices_by_ticker, signal_date, variant)
        scored: list[tuple[float, str, dict]] = []
        for ticker, score in scores.items():
            chip = lookup.get((pd.Timestamp(signal_date).normalize(), ticker), {})
            if _is_market_exposure_tool(ticker):
                adjusted = score
                reason = "market_exposure_tool_not_chip_filtered"
            else:
                h1 = _h1_score(chip)
                h2 = _h2_risk(chip, spec)
                if spec.require_h1_confirmation and h1 < spec.h1_min_score:
                    continue
                if spec.veto_h2_pressure and h2:
                    continue
                adjusted = score + (spec.h1_soft_bonus if h1 >= spec.h1_min_score else 0.0)
                adjusted -= spec.h2_score_penalty if h2 else 0.0
                reason = f"h1={h1};h2={int(h2)};base_score={score:.4f};adjusted={adjusted:.4f}"
            scored.append((adjusted, ticker, {"base_score": score, "reason": reason}))

        if not scored:
            return TargetSelectionOverlayDecision(
                target=fallback,
                reason=f"{spec.variant_id}_no_chip_confirmed_candidate_fallback={fallback or 'cash'}",
                signal_date=pd.Timestamp(signal_date).strftime("%Y-%m-%d"),
            )
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        for adjusted, ticker, detail in scored:
            if _passes_margin(ticker, detail["base_score"], prices_by_ticker, signal_date, regime, variant):
                changed = ticker != baseline_target
                return TargetSelectionOverlayDecision(
                    target=ticker,
                    reason=(
                        f"{spec.variant_id}_{'changed' if changed else 'same'};"
                        f"{detail['reason']};fallback={fallback or ''}"
                    ),
                    signal_date=pd.Timestamp(signal_date).strftime("%Y-%m-%d"),
                )
        return TargetSelectionOverlayDecision(
            target=fallback,
            reason=f"{spec.variant_id}_all_candidates_failed_margin_fallback={fallback or 'cash'}",
            signal_date=pd.Timestamp(signal_date).strftime("%Y-%m-%d"),
        )

    return overlay


def build_decision_diff_panel(
    *,
    period_id: str,
    variant_id: str,
    variant_label: str,
    baseline_curve: pd.DataFrame,
    challenger_curve: pd.DataFrame,
    chip_panel: pd.DataFrame,
) -> pd.DataFrame:
    base = _curve_with_date_column(baseline_curve).rename(
        columns={
            "current_ticker": "baseline_ticker",
            "current_exposure": "baseline_exposure",
            "total_value": "baseline_total_value_twd",
        }
    )
    challenger = _curve_with_date_column(challenger_curve).rename(
        columns={
            "current_ticker": "challenger_ticker",
            "current_exposure": "challenger_exposure",
            "total_value": "challenger_total_value_twd",
        }
    )
    merged = base[["date", "baseline_ticker", "baseline_exposure", "baseline_total_value_twd"]].merge(
        challenger[
            [
                "date",
                "challenger_ticker",
                "challenger_exposure",
                "challenger_total_value_twd",
                "target_overlay_baseline_target",
                "target_overlay_target",
                "target_overlay_reason",
                "target_overlay_signal_date",
                "target_overlay_changed",
            ]
        ],
        on="date",
        how="inner",
    )
    chip_lookup = _chip_lookup(chip_panel)
    rows: list[dict] = []
    for row in merged.itertuples(index=False):
        chip_date = _safe_date(getattr(row, "target_overlay_signal_date", ""))
        ticker = getattr(row, "target_overlay_target", "") or row.challenger_ticker
        chip = chip_lookup.get((chip_date, ticker), {}) if chip_date is not None else {}
        rows.append(
            {
                "period_id": period_id,
                "variant_id": variant_id,
                "variant_label": variant_label,
                "date": pd.Timestamp(row.date).strftime("%Y-%m-%d"),
                "baseline_ticker": row.baseline_ticker,
                "baseline_exposure": round(float(row.baseline_exposure), 4),
                "challenger_ticker": row.challenger_ticker,
                "challenger_exposure": round(float(row.challenger_exposure), 4),
                "changed_formal_candidate": bool(row.baseline_ticker != row.challenger_ticker),
                "target_selection_changed": bool(getattr(row, "target_overlay_changed", False)),
                "change_reason": getattr(row, "target_overlay_reason", ""),
                "chip_signal_date": chip_date.strftime("%Y-%m-%d") if chip_date is not None else "",
                "attack_confirmation_score": int(_h1_score(chip)),
                "sell_pressure_warning_score": int(_number(chip.get("sell_pressure_warning_score", 0))),
                "h1_negative_or_h2_sell_pressure": bool(chip.get("h1_negative_or_h2_sell_pressure", False)),
                "valuation_used": False,
                "h3_used": False,
                "baseline_total_value_twd": round(float(row.baseline_total_value_twd), 2),
                "challenger_total_value_twd": round(float(row.challenger_total_value_twd), 2),
            }
        )
    return pd.DataFrame(rows)


def _eligible_scores(
    prices_by_ticker: dict[str, pd.DataFrame],
    signal_date: pd.Timestamp,
    variant: RegimeModeSwitchVariant,
) -> dict[str, float]:
    scores = relative_strength_scores(prices_by_ticker, signal_date)
    return {
        ticker: score
        for ticker, score in scores.items()
        if (
            ticker not in variant.attack_selection_exclude_tickers
            and (
                variant.candidate_trend_filter is None
                or _passes_candidate_filter(prices_by_ticker[ticker], signal_date, variant.candidate_trend_filter)
            )
        )
    }


def _fallback_target(
    prices_by_ticker: dict[str, pd.DataFrame],
    signal_date: pd.Timestamp,
    variant: RegimeModeSwitchVariant,
) -> str | None:
    fallback = variant.relative_score_fallback_ticker or variant.fallback_ticker
    if not fallback or fallback not in prices_by_ticker:
        return None
    if variant.relative_score_fallback_defense_rule and not risk_on_for_rule(
        prices_by_ticker[fallback],
        signal_date,
        variant.relative_score_fallback_defense_rule,
    ):
        return None
    return fallback


def _passes_margin(
    ticker: str,
    score: float,
    prices_by_ticker: dict[str, pd.DataFrame],
    signal_date: pd.Timestamp,
    regime: str,
    variant: RegimeModeSwitchVariant,
) -> bool:
    min_margin = _min_score_margin_for_regime(variant, regime, prices_by_ticker, signal_date)
    fallback = variant.relative_score_fallback_ticker
    if min_margin is None or fallback is None or ticker == fallback:
        return True
    fallback_score = relative_strength_scores(prices_by_ticker, signal_date).get(fallback)
    return fallback_score is None or score - fallback_score >= min_margin


def _chip_lookup(panel: pd.DataFrame) -> dict[tuple[pd.Timestamp, str], dict]:
    lookup: dict[tuple[pd.Timestamp, str], dict] = {}
    for row in panel.to_dict(orient="records"):
        key = (pd.Timestamp(row["date"]).normalize(), str(row["ticker"]))
        lookup[key] = row
    return lookup


def _h1_score(chip: dict) -> int:
    return int(_number(chip.get("attack_confirmation_score", 0)))


def _h2_risk(chip: dict, spec: ChipFormalVariant) -> bool:
    if not chip:
        return False
    score_risk = _number(chip.get("sell_pressure_warning_score", 0)) >= spec.h2_min_score
    grouped_risk = bool(chip.get("h1_negative_or_h2_sell_pressure", False)) if spec.use_h4_shadow_risk else False
    return bool(score_risk or grouped_risk)


def _is_market_exposure_tool(ticker: str) -> bool:
    return ticker in {"0050.TW", "00631L.TW"}


def _summary_row(
    period_id: str,
    period_label: str,
    variant_id: str,
    variant_label: str,
    result: BacktestResult,
    initial_cash: float,
) -> dict:
    return {
        "period_id": period_id,
        "period_label": period_label,
        "variant_id": variant_id,
        "variant_label": variant_label,
        "final_value_twd": round(float(result.final_value), 2),
        "total_return_pct": round(float(result.total_return) * 100, 4),
        "max_drawdown_pct": round(float(result.max_drawdown) * 100, 4),
        "trade_count": len(result.trades),
        "initial_cash_twd": round(float(initial_cash), 2),
        "active_in_trade_decision": False,
        "formal_promotion_status": "challenger_only" if variant_id != "best_v20260605" else "baseline",
    }


def _daily_frame(period_id: str, variant_id: str, curve: pd.DataFrame) -> pd.DataFrame:
    frame = _curve_with_date_column(curve)
    frame["period_id"] = period_id
    frame["variant_id"] = variant_id
    return frame


def _curve_with_date_column(curve: pd.DataFrame) -> pd.DataFrame:
    frame = curve.reset_index().copy()
    if "date" not in frame.columns:
        frame = frame.rename(columns={frame.columns[0]: "date"})
    return frame


def _v0_alignment_check(period_id: str, baseline: BacktestResult, v0: BacktestResult) -> dict:
    aligned_values = baseline.equity_curve["total_value"].round(6).equals(v0.equity_curve["total_value"].round(6))
    aligned_tickers = baseline.equity_curve["current_ticker"].astype(str).equals(
        v0.equity_curve["current_ticker"].astype(str)
    )
    return {
        "period_id": period_id,
        "aligned": bool(aligned_values and aligned_tickers and len(baseline.trades) == len(v0.trades)),
        "value_aligned": bool(aligned_values),
        "ticker_aligned": bool(aligned_tickers),
        "baseline_trade_count": len(baseline.trades),
        "v0_trade_count": len(v0.trades),
    }


def _safe_date(value: object) -> pd.Timestamp | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return pd.Timestamp(text).normalize()
    except ValueError:
        return None


def _number(value: object) -> float:
    try:
        text = str(value).replace(",", "").replace("%", "").strip()
        return float(text) if text else 0.0
    except (TypeError, ValueError):
        return 0.0


def _write_summary(path: Path, metadata: dict, summary: pd.DataFrame, decision_diff: pd.DataFrame) -> None:
    changed_counts = (
        decision_diff.groupby("variant_id")["changed_formal_candidate"].sum().to_dict()
        if not decision_diff.empty
        else {}
    )
    lines = [
        "# Chip-aware Formal Challenger Runner",
        "",
        f"- active_in_trade_decision: `{metadata['active_in_trade_decision']}`",
        f"- formal_model_changed: `{metadata['formal_model_changed']}`",
        f"- valuation_used: `{metadata['valuation_used']}`",
        f"- h3_used: `{metadata['h3_used']}`",
        f"- V0 no-op all aligned: `{metadata['v0_noop_all_aligned']}`",
        "",
        "本輸出只提供 Experiments 接續驗證用，不代表正式模型升級。",
        "",
        "## Variant Summary",
        "",
        "| period | variant | return | max drawdown | return diff | changed decisions |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in summary.to_dict(orient="records"):
        lines.append(
            f"| {row['period_id']} | {row['variant_id']} | {row['total_return_pct']:.2f}% | "
            f"{row['max_drawdown_pct']:.2f}% | {row.get('return_diff_pct', 0):.2f}% | "
            f"{int(changed_counts.get(row['variant_id'], 0))} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run chip-aware formal challenger replay contract.")
    parser.add_argument("--chip-panel", default=DEFAULT_CHIP_PANEL)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--config", default="configs/ep05_universe.json")
    parser.add_argument("--group-id", default=DEFAULT_FROZEN_GROUP_ID)
    parser.add_argument("--initial-cash", type=float, default=None)
    args = parser.parse_args()
    output = run_chip_formal_challenger_runner(
        chip_panel_path=args.chip_panel,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        config_path=args.config,
        group_id=args.group_id,
        initial_cash=args.initial_cash,
    )
    print(f"OUTPUT_DIR={output.resolve()}")


if __name__ == "__main__":
    main()
