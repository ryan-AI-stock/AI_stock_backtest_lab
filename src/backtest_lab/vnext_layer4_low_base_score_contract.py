from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from backtest_lab.vnext_adhoc_20260708_signal_materialization_refresh import (
    DEFAULT_RADAR_WINDOW,
    FLAGS,
    REQUESTED_DATE,
    compute_benchmark_features,
    load_market_with_patch,
    normalize_ticker,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
RADAR_ROOT = Path("C:/Users/zergv/Documents/Codex/2026-05-23/ai-stock-rotation-radar-https-docs/outputs")
DEFAULT_BENCHMARK_GAP = RADAR_ROOT / "radar_vnext_adhoc_20260708_benchmark_etf_gap_fill_20260708"
OUTPUT_DIR = REPO_ROOT / "outputs" / "vnext_layer4_low_base_score_contract_20260709"
LAYER4_PRIMARY80 = REPO_ROOT / "outputs" / "vnext_layer4_80_primary_pool_contract_20260708" / "layer4_80_primary_pool_contract.csv"

TASK_ID = "TASK-BACKTEST-CORE-VNEXT-LAYER4-LOW-BASE-SCORE-CONTRACT-001"
LOW_BASE_FLAGS = {**FLAGS, "report_changed": False}

VARIANTS = {
    "balanced": {
        "price_position_low_base": 0.20,
        "stock_specific_bias_score": 0.16,
        "recent_runup_inverse": 0.14,
        "improving_rs_score": 0.18,
        "liquidity_improvement": 0.14,
        "quality_support": 0.13,
        "overheat_inverse": 0.05,
    },
    "momentum_low_base": {
        "price_position_low_base": 0.16,
        "stock_specific_bias_score": 0.12,
        "recent_runup_inverse": 0.10,
        "improving_rs_score": 0.30,
        "liquidity_improvement": 0.20,
        "quality_support": 0.07,
        "overheat_inverse": 0.05,
    },
    "quality_low_base": {
        "price_position_low_base": 0.18,
        "stock_specific_bias_score": 0.14,
        "recent_runup_inverse": 0.12,
        "improving_rs_score": 0.14,
        "liquidity_improvement": 0.10,
        "quality_support": 0.27,
        "overheat_inverse": 0.05,
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Layer4 low_base_score component contract.")
    parser.add_argument("--radar-window-dir", default=str(DEFAULT_RADAR_WINDOW))
    parser.add_argument("--benchmark-gap-dir", default=str(DEFAULT_BENCHMARK_GAP))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--as-of-date", default=REQUESTED_DATE)
    args = parser.parse_args()
    build_package(
        radar_window_dir=Path(args.radar_window_dir),
        benchmark_gap_dir=Path(args.benchmark_gap_dir),
        output_dir=Path(args.output_dir),
        as_of_date=args.as_of_date,
    )


def build_package(*, radar_window_dir: Path, benchmark_gap_dir: Path, output_dir: Path, as_of_date: str) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    source_rows = radar_window_dir / "vnext_adhoc_20260708_historical_window_scoped_common_stock_etf_rows.csv"
    benchmark_gap_rows = benchmark_gap_dir / "vnext_adhoc_20260708_benchmark_etf_gap_rows.csv"

    market = load_market_with_patch(source_rows)
    benchmark = compute_benchmark_features(source_rows, benchmark_gap_rows, as_of_date)
    components = build_components(market, benchmark, as_of_date)
    variants = score_variants(components)
    top10 = variants[variants["low_base_rank"] <= 10].copy()
    top10 = top10.sort_values(["score_variant", "low_base_rank"])

    component_matrix = output_dir / "layer4_low_base_score_component_matrix.csv"
    formula_path = output_dir / "layer4_low_base_score_variant_definitions.csv"
    top10_path = output_dir / "layer4_low_base_20260708_top10_sample.csv"
    overlap_path = output_dir / "existing_low_base_overlap_audit.csv"
    placement_path = output_dir / "layer4_low_base_layer_placement_decision.csv"
    blocked_path = output_dir / "layer4_low_base_blocked_proxy_audit.csv"
    readiness_path = output_dir / "readiness_for_layer4_low_base_score_diagnostic.json"
    summary_path = output_dir / "final_summary_zh.md"

    components.to_csv(component_matrix, index=False, encoding="utf-8-sig")
    variant_definitions().to_csv(formula_path, index=False, encoding="utf-8-sig")
    top10.to_csv(top10_path, index=False, encoding="utf-8-sig")
    existing_low_base_overlap_audit().to_csv(overlap_path, index=False, encoding="utf-8-sig")
    layer_placement_decision().to_csv(placement_path, index=False, encoding="utf-8-sig")
    blocked = blocked_proxy_audit()
    blocked.to_csv(blocked_path, index=False, encoding="utf-8-sig")
    future = pd.DataFrame(
        [
            {"audit_item": "future_return_used_in_low_base_score", "used": False, "future_data_violation_count": 0},
            {"audit_item": "future_winner_used_in_low_base_score", "used": False, "future_data_violation_count": 0},
        ]
    )
    future.to_csv(output_dir / "layer4_low_base_future_data_audit.csv", index=False, encoding="utf-8-sig")
    coverage = coverage_audit(components, top10)
    coverage.to_csv(output_dir / "layer4_low_base_coverage_audit.csv", index=False, encoding="utf-8-sig")

    layer4_latest = latest_layer4_date()
    layer4_ready = layer4_latest == as_of_date
    readiness = {
        "task": TASK_ID,
        "status": "low_base_score_contract_ready_reference_sample_layer4_20260708_primary80_blocked",
        "as_of_date": as_of_date,
        "component_matrix_rows": int(len(components)),
        "score_variant_count": int(len(VARIANTS)),
        "top10_reference_rows": int(len(top10)),
        "layer4_primary80_as_of_date_ready": bool(layer4_ready),
        "layer4_primary80_latest_date": layer4_latest,
        "ready_for_layer4_low_base_score_experiments_diagnostic": False,
        "ready_for_strategy_center_review": True,
        "ready_for_formal": False,
        "ready_for_strategy_replay": False,
        "future_data_violation_count": 0,
        "low_base_hard_filter_allowed": False,
        "recommended_layer": "Layer4 ranking component with Layer2/Layer3 field reuse",
        "overlap_audit_ready": True,
        "blocked_reason": "2026-07-08 exact Layer4 primary80 is not materialized; top10 sample is Layer0-active reference only.",
        **LOW_BASE_FLAGS,
    }
    write_json(readiness_path, readiness)
    write_summary(summary_path, readiness, top10)
    write_json(
        output_dir / "manifest.json",
        {
            "task": TASK_ID,
            "output_dir": str(output_dir),
            "artifacts": [
                component_matrix.name,
                formula_path.name,
                top10_path.name,
                overlap_path.name,
                placement_path.name,
                blocked_path.name,
                "layer4_low_base_coverage_audit.csv",
                "layer4_low_base_future_data_audit.csv",
                readiness_path.name,
                summary_path.name,
            ],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "formal_model_changed": False,
            "trade_decision_changed": False,
            "report_changed": False,
        },
    )
    return readiness


def build_components(market: pd.DataFrame, benchmark: pd.DataFrame, as_of_date: str) -> pd.DataFrame:
    common = market[~market["ticker"].isin(["0050", "00631L"])].copy()
    common = common.sort_values(["ticker", "date"])
    for window in [20, 60]:
        common[f"return_{window}d"] = common.groupby("ticker")["close"].pct_change(window)
        common[f"ma{window}"] = common.groupby("ticker")["close"].transform(lambda s: s.rolling(window, min_periods=window).mean())
        common[f"bias{window}"] = common["close"] / common[f"ma{window}"] - 1.0
        common[f"high_{window}d"] = common.groupby("ticker")["close"].transform(lambda s: s.rolling(window, min_periods=window).max())
        common[f"drawdown_from_{window}d_high"] = common["close"] / common[f"high_{window}d"] - 1.0
    common["ma120"] = common.groupby("ticker")["close"].transform(lambda s: s.rolling(120, min_periods=120).mean())
    common["bias120"] = common["close"] / common["ma120"] - 1.0
    common["high_120d"] = common.groupby("ticker")["close"].transform(lambda s: s.rolling(120, min_periods=120).max())
    common["drawdown_from_120d_high"] = common["close"] / common["high_120d"] - 1.0
    common["daily_return"] = common.groupby("ticker")["close"].pct_change()
    common["volatility20"] = common.groupby("ticker")["daily_return"].transform(lambda s: s.rolling(20, min_periods=20).std())
    for col in ["bias20", "bias60"]:
        mean = common.groupby("ticker")[col].transform(lambda s: s.rolling(252, min_periods=60).mean())
        std = common.groupby("ticker")[col].transform(lambda s: s.rolling(252, min_periods=60).std())
        common[f"{col}_zscore_252d"] = (common[col] - mean) / std
    for window in [20, 60]:
        common[f"turnover_{window}d"] = common.groupby("ticker")["turnover_value"].transform(lambda s: s.rolling(window, min_periods=window).mean())
    current = common[common["date"].dt.strftime("%Y-%m-%d").eq(as_of_date)].copy()
    for col in ["turnover_20d", "turnover_60d"]:
        current[col.replace("turnover_", "traded_value_rank_")] = current[col].rank(method="first", ascending=False)

    b0050 = benchmark[benchmark["ticker"].eq("0050")].iloc[0]
    current["RS20"] = current["return_20d"] - float(b0050["return_20d"])
    current["RS60"] = current["return_60d"] - float(b0050["return_60d"])
    current = current[current["turnover_20d"].notna()].copy()
    current = current.merge(latest_quality_context(), on="ticker", how="left")

    current["price_position_low_base"] = (
        ((-current["drawdown_from_120d_high"]).clip(0.03, 0.45) / 0.45) * 0.65
        + ((-current["drawdown_from_60d_high"]).clip(0.02, 0.35) / 0.35) * 0.35
    ).clip(0, 1)
    z = current["bias60_zscore_252d"].abs()
    current["stock_specific_bias_score"] = (1.0 - (z / 2.5).clip(0, 1)).fillna(0.5)
    runup = (current["return_20d"].clip(lower=0) / 0.35).clip(0, 1) * 0.45
    runup += (current["return_60d"].clip(lower=0) / 0.75).clip(0, 1) * 0.55
    current["recent_runup_penalty"] = runup.clip(0, 1)
    current["recent_runup_inverse"] = 1.0 - current["recent_runup_penalty"]
    rs = (current["RS20"].rank(pct=True) * 0.65 + current["RS60"].rank(pct=True) * 0.35).fillna(0.5)
    rs_overheat_penalty = ((current["RS60"].clip(lower=0) / 1.0).clip(0, 1) * 0.25).fillna(0)
    current["improving_rs_score"] = (rs - rs_overheat_penalty).clip(0, 1)
    rank60 = current["traded_value_rank_60d"]
    rank20 = current["traded_value_rank_20d"]
    improvement = ((rank60 - rank20) / rank60.clip(lower=1)).clip(-1, 1)
    current["liquidity_improvement"] = ((improvement + 1.0) / 2.0 * 0.45 + (1.0 - ((rank20 - 1.0) / len(current)).clip(0, 1)) * 0.55).clip(0, 1)
    q = 1.0 - pd.to_numeric(current["layer1_quality_floor_risk_pctile_by_week"], errors="coerce").fillna(0.5)
    q += current["layer1_pass_bottom30"].astype(str).str.lower().eq("true").astype(float) * 0.15
    current["quality_support"] = q.clip(0, 1)
    current["overheat_veto_flag"] = (
        (current["bias60_zscore_252d"] > 2.5)
        | (current["RS60"] > 1.0)
        | (current["return_60d"] > 0.9)
        | (current["volatility20"].rank(pct=True) > 0.95)
    )
    current["overheat_inverse"] = (~current["overheat_veto_flag"]).astype(float)
    current["sample_scope"] = "layer0_active_reference_not_exact_layer4_primary80"
    current["diagnostic_only"] = True
    current["not_live_rule"] = True
    current["forward_returns_live_rule_usage"] = False
    for key, value in LOW_BASE_FLAGS.items():
        current[key] = value
    columns = [
        "date",
        "ticker",
        "name",
        "market",
        "close",
        "return_20d",
        "return_60d",
        "RS20",
        "RS60",
        "traded_value_rank_20d",
        "traded_value_rank_60d",
        "drawdown_from_60d_high",
        "drawdown_from_120d_high",
        "bias20",
        "bias60",
        "bias20_zscore_252d",
        "bias60_zscore_252d",
        "volatility20",
        "layer1_quality_floor_risk_pctile_by_week",
        "layer1_pass_bottom30",
        "price_position_low_base",
        "stock_specific_bias_score",
        "recent_runup_penalty",
        "recent_runup_inverse",
        "improving_rs_score",
        "liquidity_improvement",
        "quality_support",
        "overheat_veto_flag",
        "overheat_inverse",
        "sample_scope",
        "diagnostic_only",
        *LOW_BASE_FLAGS.keys(),
    ]
    current["date"] = current["date"].dt.strftime("%Y-%m-%d")
    return current[[c for c in columns if c in current.columns]]


def latest_quality_context() -> pd.DataFrame:
    cols = ["snapshot_date", "ticker", "layer1_quality_floor_risk_pctile_by_week", "layer1_pass_bottom30"]
    layer4 = pd.read_csv(LAYER4_PRIMARY80, usecols=cols, dtype={"ticker": str})
    layer4["ticker"] = layer4["ticker"].map(normalize_ticker)
    latest = layer4["snapshot_date"].astype(str).max()
    latest_rows = layer4[layer4["snapshot_date"].astype(str).eq(latest)].copy()
    return latest_rows.drop_duplicates("ticker")[["ticker", "layer1_quality_floor_risk_pctile_by_week", "layer1_pass_bottom30"]]


def score_variants(components: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant, weights in VARIANTS.items():
        df = components.copy()
        df["score_variant"] = variant
        df["low_base_score"] = sum(pd.to_numeric(df[col], errors="coerce").fillna(0.5) * weight for col, weight in weights.items())
        df.loc[df["overheat_veto_flag"], "low_base_score"] *= 0.65
        df = df.sort_values(["low_base_score", "ticker"], ascending=[False, True])
        df["low_base_rank"] = range(1, len(df) + 1)
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def variant_definitions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "score_variant": variant,
                "formula": " + ".join(f"{weight}*{component}" for component, weight in weights.items()),
                "overheat_policy": "overheat_veto_flag multiplies score by 0.65; no hard deletion",
                "future_return_used": False,
                "diagnostic_only": True,
            }
            for variant, weights in VARIANTS.items()
        ]
    )


def blocked_proxy_audit() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "field": "exact_layer4_primary80_20260708",
                "status": "blocked",
                "reason": "Layer4 primary80 contract latest date is 2026-06-29",
                "policy": "2026-07-08 low_base top10 is reference sample, not selected rule",
            },
            {
                "field": "quality_support",
                "status": "proxy",
                "reason": "uses latest available Layer1 quality context by ticker from 2026-06-29 Layer4 contract",
                "policy": "allowed as diagnostic proxy; not formal",
            },
            {
                "field": "adjusted_close",
                "status": "blocked",
                "reason": "current 2026-07-08 source is official unadjusted OHLCV",
                "policy": "do not promote to formal without accepted adjusted source",
            },
            {
                "field": "stock_specific_bias_score",
                "status": "ready_proxy",
                "reason": "uses ticker-specific rolling 252D z-score from PIT-observable close history",
                "policy": "diagnostic scoring component only",
            },
        ]
    )


def existing_low_base_overlap_audit() -> pd.DataFrame:
    rows = [
        {
            "existing_area": "Layer2 / Layer4 BIAS health",
            "existing_fields_or_scores": "BIAS20, BIAS60, stock-specific percentile/zscore where available",
            "overlap_with_low_base": "high",
            "decision": "keep_existing_as_is_and_reuse",
            "low_base_gap_filled": "combine BIAS health with distance-to-high and runup penalty instead of duplicating raw BIAS thresholds",
            "hard_filter_policy": "no hard filter",
        },
        {
            "existing_area": "Layer2 risk / overheat context",
            "existing_fields_or_scores": "overheat penalty, exhaustion context, volatility proxy, risk proxy",
            "overlap_with_low_base": "high",
            "decision": "merge_into_low_base_score_as_penalty_only",
            "low_base_gap_filled": "use as score haircut/veto context; do not delete candidates unless Strategy Center later authorizes",
            "hard_filter_policy": "penalty/bonus only",
        },
        {
            "existing_area": "Layer3 broad opportunity labels",
            "existing_fields_or_scores": "pullback_repair, reacceleration, momentum_continuation, neutral_quality_liquidity",
            "overlap_with_low_base": "medium",
            "decision": "move_pullback_semantics_to_Layer3_opportunity_label",
            "low_base_gap_filled": "low_base should not replace pullback/reacceleration labels; it only supplies a position-not-overheated score",
            "hard_filter_policy": "annotation only",
        },
        {
            "existing_area": "Layer2 RS soft context",
            "existing_fields_or_scores": "RS20, RS30 proxy, RS40 proxy, RS60, RS60 high + short RS weakening exhaustion",
            "overlap_with_low_base": "medium",
            "decision": "move_momentum_context_to_Layer2_soft_score_and_reuse_in_Layer4",
            "low_base_gap_filled": "prefer improving RS without late-stage overheat, not raw strongest RS",
            "hard_filter_policy": "ranking component only",
        },
        {
            "existing_area": "Layer4 route_support / quality_risk / quality_rs",
            "existing_fields_or_scores": "route_support weighted score, quality_rs score, risk_aware score, quota pool scores",
            "overlap_with_low_base": "medium",
            "decision": "keep_as_Layer4_ranking_component",
            "low_base_gap_filled": "adds low-base tilt when route_support candidates are not overheated; does not replace route_support primary score",
            "hard_filter_policy": "secondary component / tie-break candidate",
        },
        {
            "existing_area": "Recent runup / high-distance context",
            "existing_fields_or_scores": "price_vs_ma, drawdown context, distance to recent high where materialized",
            "overlap_with_low_base": "high",
            "decision": "merge_into_low_base_score",
            "low_base_gap_filled": "explicitly quantify not-at-60D/120D-high and recent-runup penalty in one auditable component",
            "hard_filter_policy": "no hard filter",
        },
        {
            "existing_area": "Layer1 quality floor",
            "existing_fields_or_scores": "exclude_bottom30, quality_floor_risk_pctile, missingness flags",
            "overlap_with_low_base": "low",
            "decision": "keep_existing_as_is",
            "low_base_gap_filled": "quality only supports low-base candidate; low price/low base cannot override poor quality",
            "hard_filter_policy": "Layer1 remains independent eligibility base",
        },
    ]
    return pd.DataFrame(rows)


def layer_placement_decision() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "candidate_layer": "Layer2 soft scoring",
                "fit": "partial",
                "reason": "RS improvement, BIAS health, volatility and overheat are soft context fields already living near Layer2.",
                "decision": "reuse Layer2 fields; do not create duplicate Layer2 gate",
            },
            {
                "candidate_layer": "Layer3 opportunity label",
                "fit": "partial",
                "reason": "pullback_repair / reacceleration belongs to Layer3 labels, especially if the semantics is prior strength then repair.",
                "decision": "do not move full low_base_score here; keep pullback semantics in Layer3",
            },
            {
                "candidate_layer": "Layer4 ranking component",
                "fit": "primary",
                "reason": "low_base_score is best used after Layer0-Layer3 have preserved candidates, as a bonus/penalty/tie-break component inside the 80-pool ranking.",
                "decision": "recommended placement; no hard deletion",
            },
            {
                "candidate_layer": "Layer5 selected rule",
                "fit": "not_ready",
                "reason": "No diagnostic evidence yet that low_base alone should drive the daily selected asset.",
                "decision": "blocked unless Experiments later validates and Strategy Center authorizes",
            },
        ]
    )


def coverage_audit(components: pd.DataFrame, top10: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"item": "component_matrix_rows", "count": len(components), "ready": len(components) > 0},
            {"item": "top10_rows_all_variants", "count": len(top10), "ready": len(top10) == len(VARIANTS) * 10},
            {"item": "as_of_date", "count": REQUESTED_DATE, "ready": True},
            {"item": "layer4_primary80_exact_20260708", "count": latest_layer4_date(), "ready": latest_layer4_date() == REQUESTED_DATE},
        ]
    )


def latest_layer4_date() -> str:
    dates = pd.read_csv(LAYER4_PRIMARY80, usecols=["snapshot_date"])["snapshot_date"].astype(str)
    return str(dates.max())


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_summary(path: Path, readiness: dict[str, Any], top10: pd.DataFrame) -> None:
    preview = top10[top10["score_variant"].eq("balanced")].head(10)[["low_base_rank", "ticker", "name", "low_base_score"]]
    path.write_text(
        f"""# Layer4 low_base_score contract

## 結論

- 已建立 Layer4 `low_base_score` component / formula / sample contract。
- 已補 `existing_low_base_overlap_audit`，確認 low_base 不新增硬篩，只作 ranking/penalty/bonus component。
- 建議放置：Layer4 ranking component；BIAS/RS/risk 欄位重用 Layer2，pullback/reacceleration 語義保留 Layer3。
- 2026-07-08 exact Layer4 primary80 尚未 materialized，所以 top10 是 Layer0-active reference sample，不是 selected rule。
- ready_for_layer4_low_base_score_experiments_diagnostic=false，需等 exact Layer4 primary80 2026-07-08 或 historical panel 接上後再交 Experiments。

## Balanced top10 reference

{preview.to_csv(index=False)}

## Flags

- formal_model_changed=false
- trade_decision_changed=false
- active_in_trade_decision=false
- report_changed=false
- portfolio_replay_executed=false
- ready_for_strategy_replay=false
- ready_for_formal=false
- not_live_rule=true
- forward_returns_live_rule_usage=false
""",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
