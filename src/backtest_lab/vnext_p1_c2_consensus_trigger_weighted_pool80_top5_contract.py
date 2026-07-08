from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_ROOT = Path("C:/Users/zergv/Documents/Codex/2026-07-06/backtest-lab-experiments-diagnostic-validation-attribution")
LAYER4_POOL = REPO_ROOT / "outputs" / "vnext_layer4_80_primary_pool_contract_20260708" / "layer4_80_primary_pool_contract.csv"
EXACT_CONSENSUS = (
    REPO_ROOT
    / "outputs"
    / "vnext_p1_c2_exact_consensus4_top5_exception_candidate_contract_20260708"
    / "p1_c2_exact_consensus4_top5_exception_candidate_contract.csv"
)
LEGACY_TRACE = (
    EXPERIMENTS_ROOT
    / "outputs"
    / "vnext_p1_legacy_regime_unadjusted_ohlc_cost_timing_diagnostic_20260708"
    / "p1_legacy_regime_unadjusted_ohlc_trade_path_trace.csv"
)
PREV_COST_DESIGN = (
    REPO_ROOT
    / "outputs"
    / "vnext_p1_c2_top5_ohlc_absorption_and_exright_review_20260708"
    / "p1_c2_top5_exception_candidate_transition_cost_design.csv"
)
OUTPUT_DIR = REPO_ROOT / "outputs" / "vnext_p1_c2_consensus_trigger_weighted_pool80_top5_contract_20260708"

TASK_ID = "TASK-BACKTEST-CORE-VNEXT-P1-C2-CONSENSUS-TRIGGER-WEIGHTED-POOL80-TOP5-CONTRACT-001"
PRIMARY_TIMING = "next_day_close_entry_fixed_5td_exit"
SOURCE_VARIANTS = [
    "hybrid_pullback_base_mega_override",
    "conservative_hurdle_route",
    "pool_breadth_route",
    "market_bias_pool_trend_route",
    "dispersion_route",
]
VARIANTS = {
    "balanced": {
        "quality_component": 0.18,
        "rs_component": 0.22,
        "liquidity_component": 0.15,
        "bias_health_component": 0.12,
        "route_support_component": 0.18,
        "risk_inverse_component": 0.15,
    },
    "momentum": {
        "quality_component": 0.08,
        "rs_component": 0.38,
        "liquidity_component": 0.20,
        "bias_health_component": 0.12,
        "route_support_component": 0.17,
        "risk_inverse_component": 0.05,
    },
    "quality_rs": {
        "quality_component": 0.28,
        "rs_component": 0.32,
        "liquidity_component": 0.10,
        "bias_health_component": 0.10,
        "route_support_component": 0.12,
        "risk_inverse_component": 0.08,
    },
    "quality_risk": {
        "quality_component": 0.30,
        "rs_component": 0.15,
        "liquidity_component": 0.10,
        "bias_health_component": 0.15,
        "route_support_component": 0.10,
        "risk_inverse_component": 0.30,
    },
    "liquidity_rs": {
        "quality_component": 0.12,
        "rs_component": 0.25,
        "liquidity_component": 0.28,
        "bias_health_component": 0.10,
        "route_support_component": 0.15,
        "risk_inverse_component": 0.10,
    },
    "route_support": {
        "quality_component": 0.10,
        "rs_component": 0.20,
        "liquidity_component": 0.10,
        "bias_health_component": 0.10,
        "route_support_component": 0.38,
        "risk_inverse_component": 0.12,
    },
}
FLAGS = {
    "formal_model_changed": False,
    "trade_decision_changed": False,
    "active_in_trade_decision": False,
    "report_changed": False,
    "portfolio_replay_executed": False,
    "ready_for_strategy_replay": False,
    "ready_for_formal": False,
    "not_live_rule": True,
    "forward_returns_live_rule_usage": False,
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _as_bool(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    return s.astype(str).str.lower().isin(["true", "1", "yes"])


def _ticker(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text.zfill(4) if text.isdigit() and len(text) < 4 else text


def _clip01(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").clip(lower=0.0, upper=1.0)


def _rank_score(df: pd.DataFrame, col: str, higher_better: bool = True) -> pd.Series:
    vals = pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.Series(index=df.index, dtype=float)
    if vals.notna().sum() == 0:
        return pd.Series(0.5, index=df.index)
    return vals.groupby(df["snapshot_date"]).rank(pct=True, ascending=not higher_better).fillna(0.5)


def _inverse_rank_score(df: pd.DataFrame, col: str) -> pd.Series:
    vals = pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.Series(index=df.index, dtype=float)
    if vals.notna().sum() == 0:
        return pd.Series(0.5, index=df.index)
    return 1.0 - (vals.groupby(df["snapshot_date"]).rank(pct=True, ascending=True).fillna(0.5) - (1.0 / df.groupby("snapshot_date")[col].transform("count").clip(lower=1)))


def _load_triggers() -> pd.DataFrame:
    df = pd.read_csv(EXACT_CONSENSUS, low_memory=False, dtype={"ticker": str})
    df["signal_date"] = pd.to_datetime(df["signal_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    for col in ["candidate_available", "c2_market_health_gate", "raw_consensus4_exception_active", "exception_allowed_by_c2"]:
        df[col] = _as_bool(df[col])
    top1 = df[
        df["candidate_rank"].eq(1)
        & df["candidate_available"]
        & df["c2_market_health_gate"]
        & df["raw_consensus4_exception_active"]
    ].copy()
    top1["consensus_trigger"] = True
    return top1[
        [
            "signal_date",
            "consensus_trigger",
            "prior_single_exception_ticker",
            "prior_c2_allowed_exception_ticker",
            "consensus_count",
            "route_count",
            "route_source_flags",
            "variant_source_flags",
        ]
    ].drop_duplicates("signal_date")


def _route_support() -> pd.DataFrame:
    trace = pd.read_csv(LEGACY_TRACE, low_memory=False, dtype={"ticker": str})
    trace["signal_date"] = pd.to_datetime(trace["signal_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    trace["ticker"] = trace["ticker"].map(_ticker)
    trace["path_ready"] = _as_bool(trace["path_ready"])
    trace = trace[
        trace["timing_variant"].eq(PRIMARY_TIMING)
        & trace["path_bucket"].eq("ordinary_stock")
        & trace["variant"].isin(SOURCE_VARIANTS)
        & trace["path_ready"]
    ].copy()
    g = (
        trace.groupby(["signal_date", "ticker"], as_index=False)
        .agg(
            route_support_variant_count=("variant", "nunique"),
            route_support_mode_count=("route_or_mode", "nunique"),
            route_support_variant_flags=("variant", lambda x: "|".join(sorted(set(map(str, x.dropna()))))),
            route_support_mode_flags=("route_or_mode", lambda x: "|".join(sorted(set(map(str, x.dropna()))))),
        )
    )
    return g


def _path_map() -> pd.DataFrame:
    trace = pd.read_csv(LEGACY_TRACE, low_memory=False, dtype={"ticker": str})
    trace["signal_date"] = pd.to_datetime(trace["signal_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    for col in ["entry_date", "exit_date"]:
        trace[col] = pd.to_datetime(trace[col], errors="coerce").dt.strftime("%Y-%m-%d")
    trace["ticker"] = trace["ticker"].map(_ticker)
    trace["path_ready"] = _as_bool(trace["path_ready"])
    trace = trace[
        trace["timing_variant"].eq(PRIMARY_TIMING)
        & trace["path_bucket"].eq("ordinary_stock")
        & trace["path_ready"]
    ].copy()
    cols = [
        "signal_date",
        "ticker",
        "entry_date",
        "exit_date",
        "entry_open",
        "entry_close",
        "exit_close",
        "gross_return_unadjusted",
        "net_return_local_ep05_cost_unit_notional",
        "source_quality",
        "entry_source_route",
        "exit_source_route",
        "entry_adjustment_policy",
        "exit_adjustment_policy",
        "total_cost_twd",
        "cost_application_status",
    ]
    existing = [c for c in cols if c in trace.columns]
    return trace.sort_values(["signal_date", "ticker", "variant", "route_or_mode"])[existing].drop_duplicates(["signal_date", "ticker"])


def _score_components(pool: pd.DataFrame, support: pd.DataFrame, triggers: pd.DataFrame) -> pd.DataFrame:
    pool = pool.copy()
    pool["snapshot_date"] = pd.to_datetime(pool["snapshot_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    pool["ticker"] = pool["ticker"].map(_ticker)
    pool = pool[pool["is_layer4_primary_pool"].astype(str).str.lower().eq("true")].copy()
    pool = pool.merge(triggers, left_on="snapshot_date", right_on="signal_date", how="inner")
    pool = pool.merge(support, left_on=["snapshot_date", "ticker"], right_on=["signal_date", "ticker"], how="left", suffixes=("", "_support"))
    pool["route_support_variant_count"] = pd.to_numeric(pool["route_support_variant_count"], errors="coerce").fillna(0)
    pool["route_support_mode_count"] = pd.to_numeric(pool["route_support_mode_count"], errors="coerce").fillna(0)

    quality = 1.0 - _clip01(pool.get("layer1_quality_floor_risk_pctile_by_week", pd.Series(index=pool.index, dtype=float))).fillna(0.5)
    quality += pool.get("layer1_pass_bottom30", False).astype(str).str.lower().eq("true").astype(float) * 0.15
    pool["quality_component"] = quality.clip(0, 1)

    rs_scores = []
    for col in ["RS20", "RS40", "RS60", "RS30_proxy"]:
        if col in pool.columns:
            rs_scores.append(_rank_score(pool, col, higher_better=True))
    pool["rs_component"] = pd.concat(rs_scores, axis=1).mean(axis=1).fillna(0.5) if rs_scores else 0.5

    liquidity_parts = []
    for col in ["traded_value_rank_20d", "traded_value_rank_60d", "traded_value_rank_5d"]:
        if col in pool.columns:
            liquidity_parts.append(1.0 - ((pd.to_numeric(pool[col], errors="coerce") - 1.0) / 80.0).clip(0, 1))
    if "capital_reasonable_band_4w_count" in pool.columns:
        liquidity_parts.append((pd.to_numeric(pool["capital_reasonable_band_4w_count"], errors="coerce") / 4.0).clip(0, 1))
    pool["liquidity_component"] = pd.concat(liquidity_parts, axis=1).mean(axis=1).fillna(0.5) if liquidity_parts else 0.5

    bias_parts = []
    for col in ["BIAS20_percentile", "BIAS60_percentile", "BIAS120_percentile"]:
        if col in pool.columns:
            p = _clip01(pool[col]).fillna(0.5)
            bias_parts.append(1.0 - (p - 0.5).abs() * 2.0)
    pool["bias_health_component"] = pd.concat(bias_parts, axis=1).mean(axis=1).fillna(0.5) if bias_parts else 0.5

    pool["route_support_component"] = (pool["route_support_variant_count"] / len(SOURCE_VARIANTS)).clip(0, 1)

    risk_penalty = pd.Series(0.0, index=pool.index)
    for col in [
        "rs60_high_short_rs_weakening_exhaustion_context",
        "rs_exhaustion_warning_context",
        "volatility_high_context",
        "risk_overheat_penalty_context",
        "high_exhaustion_or_breakdown_context",
    ]:
        if col in pool.columns:
            risk_penalty += pool[col].astype(str).str.lower().eq("true").astype(float) * 0.15
    if "exhaustion_risk_score" in pool.columns:
        risk_penalty += pd.to_numeric(pool["exhaustion_risk_score"], errors="coerce").fillna(0) * 0.25
    if "breakdown_risk_score" in pool.columns:
        risk_penalty += pd.to_numeric(pool["breakdown_risk_score"], errors="coerce").fillna(0) * 0.25
    pool["risk_inverse_component"] = (1.0 - risk_penalty.clip(0, 1)).clip(0, 1)

    pool["pool_persistence_component_proxy"] = (
        pd.to_numeric(pool.get("capital_reasonable_band_4w_count", pd.Series(0, index=pool.index)), errors="coerce").fillna(0) / 4.0
    ).clip(0, 1)
    return pool


def _rank_variants(components: pd.DataFrame, path: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    top_rows: list[pd.DataFrame] = []
    all_rows: list[pd.DataFrame] = []
    for variant, weights in VARIANTS.items():
        df = components.copy()
        df["score_variant"] = variant
        df["weighted_score"] = sum(df[col].fillna(0.5) * weight for col, weight in weights.items())
        df = df.sort_values(["snapshot_date", "weighted_score", "ticker"], ascending=[True, False, True])
        df["pool80_quant_rank"] = df.groupby("snapshot_date").cumcount() + 1
        all_rows.append(df)
        top_rows.append(df[df["pool80_quant_rank"] <= 5].copy())
    out = pd.concat(top_rows, ignore_index=True)
    out = out.merge(path, left_on=["snapshot_date", "ticker"], right_on=["signal_date", "ticker"], how="left", suffixes=("", "_path"))
    out["signal_date"] = out["snapshot_date"]
    out["candidate_rank"] = out["pool80_quant_rank"]
    out["official_unadjusted_ohlc_path_ready"] = out[["entry_close", "exit_close"]].notna().all(axis=1)
    out["adjusted_close_ready"] = False
    out["adjustment_policy"] = "official_unadjusted_ohlc_diagnostic_only_adjusted_close_blocked"
    out["future_data_violation_count"] = 0
    out["diagnostic_only"] = True
    for key, value in FLAGS.items():
        out[key] = value
    return out, pd.concat(all_rows, ignore_index=True)


def _contract_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "signal_date",
        "score_variant",
        "candidate_rank",
        "pool80_quant_rank",
        "ticker",
        "name",
        "market",
        "weighted_score",
        "quality_component",
        "rs_component",
        "liquidity_component",
        "bias_health_component",
        "route_support_component",
        "risk_inverse_component",
        "pool_persistence_component_proxy",
        "c2_market_health_gate",
        "consensus_trigger",
        "prior_c2_allowed_exception_ticker",
        "prior_single_exception_ticker",
        "route_support_variant_count",
        "route_support_mode_count",
        "route_support_variant_flags",
        "route_support_mode_flags",
        "layer1_quality_floor_risk_pctile_by_week",
        "layer1_pass_bottom30",
        "RS20",
        "RS40",
        "RS60",
        "RS30_proxy",
        "traded_value_rank_5d",
        "traded_value_rank_20d",
        "traded_value_rank_60d",
        "BIAS20_percentile",
        "BIAS60_percentile",
        "BIAS120_percentile",
        "exhaustion_risk_score",
        "breakdown_risk_score",
        "entry_date",
        "exit_date",
        "entry_open",
        "entry_close",
        "exit_close",
        "gross_return_unadjusted",
        "net_return_local_ep05_cost_unit_notional",
        "official_unadjusted_ohlc_path_ready",
        "adjusted_close_ready",
        "source_quality",
        "entry_source_route",
        "exit_source_route",
        "entry_adjustment_policy",
        "exit_adjustment_policy",
        "total_cost_twd",
        "cost_application_status",
        "adjustment_policy",
        "future_data_violation_count",
        "diagnostic_only",
        *FLAGS.keys(),
    ]
    return df[[c for c in cols if c in df.columns]]


def _variant_definitions() -> pd.DataFrame:
    rows = []
    for variant, weights in VARIANTS.items():
        row = {
            "score_variant": variant,
            "formula": "sum(component * weight)",
            "ranking_policy": "PIT quant score within full Layer4 primary 80; C2+consensus4 only trigger eligible dates; prior top1 is comparator only",
            "component_source_policy": "Layer1 quality, RS20/40/60, traded-value ranks, stock-specific BIAS percentiles, route support, risk/exhaustion proxies",
            "future_return_used": False,
            "layer4_generic_rank_only": False,
        }
        row.update(weights)
        rows.append(row)
    return pd.DataFrame(rows)


def _coverage(contract: pd.DataFrame) -> pd.DataFrame:
    return (
        contract.groupby("score_variant", as_index=False)
        .agg(
            rows=("ticker", "size"),
            signal_dates=("signal_date", "nunique"),
            unique_tickers=("ticker", "nunique"),
            official_unadjusted_ohlc_ready_rows=("official_unadjusted_ohlc_path_ready", "sum"),
        )
        .assign(
            official_unadjusted_ohlc_ready_share=lambda x: x["official_unadjusted_ohlc_ready_rows"] / x["rows"],
            adjusted_close_ready_rows=0,
            adjusted_close_ready_share=0.0,
        )
    )


def _blocked(contract: pd.DataFrame) -> pd.DataFrame:
    rows = []
    missing = contract[~contract["official_unadjusted_ohlc_path_ready"].fillna(False).astype(bool)]
    for r in missing.itertuples(index=False):
        rows.append(
            {
                "signal_date": r.signal_date,
                "score_variant": r.score_variant,
                "candidate_rank": r.candidate_rank,
                "ticker": r.ticker,
                "blocked_item": "official_unadjusted_ohlc_path",
                "blocked_reason": "not available in local selected-trace path map",
                "next_owner": "Radar/Data if Strategy Center requires full path fill",
            }
        )
    rows.extend(
        [
            {
                "signal_date": "",
                "score_variant": "all",
                "candidate_rank": "",
                "ticker": "",
                "blocked_item": "adjusted_close",
                "blocked_reason": "exact historical ex-right date and capital-change adjustment route incomplete",
                "next_owner": "Strategy Center policy or Radar/Data adjusted source route",
            },
            {
                "signal_date": "",
                "score_variant": "all",
                "candidate_rank": "",
                "ticker": "",
                "blocked_item": "risk_bucket",
                "blocked_reason": "formal risk bucket blocked; using diagnostic proxy risk components only",
                "next_owner": "Core/Data or Radar/Data only if required",
            },
        ]
    )
    return pd.DataFrame(rows)


def _score_component_table(components: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "snapshot_date",
        "ticker",
        "name",
        "quality_component",
        "rs_component",
        "liquidity_component",
        "bias_health_component",
        "route_support_component",
        "risk_inverse_component",
        "pool_persistence_component_proxy",
        "route_support_variant_count",
        "route_support_variant_flags",
        "route_support_mode_flags",
    ]
    return components[[c for c in cols if c in components.columns]]


def _quant_component_matrix(all_scored: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "snapshot_date",
        "score_variant",
        "pool80_quant_rank",
        "ticker",
        "name",
        "market",
        "weighted_score",
        "quality_component",
        "rs_component",
        "liquidity_component",
        "bias_health_component",
        "route_support_component",
        "risk_inverse_component",
        "pool_persistence_component_proxy",
        "route_support_variant_count",
        "route_support_variant_flags",
        "route_support_mode_flags",
        "layer1_quality_floor_risk_pctile_by_week",
        "layer1_pass_bottom30",
        "RS20",
        "RS40",
        "RS60",
        "RS30_proxy",
        "traded_value_rank_5d",
        "traded_value_rank_20d",
        "traded_value_rank_60d",
        "BIAS20_percentile",
        "BIAS60_percentile",
        "BIAS120_percentile",
        "exhaustion_risk_score",
        "breakdown_risk_score",
    ]
    return all_scored[[c for c in cols if c in all_scored.columns]].rename(columns={"snapshot_date": "signal_date"})


def _prior_match_audit(contract: pd.DataFrame) -> pd.DataFrame:
    top1 = contract[contract["candidate_rank"].eq(1)].copy()
    top1["matches_prior_single_exception"] = top1["ticker"].eq(top1["prior_single_exception_ticker"].fillna(""))
    top1["matches_prior_c2_allowed_exception"] = top1["ticker"].eq(top1["prior_c2_allowed_exception_ticker"].fillna(""))
    cols = [
        "signal_date",
        "score_variant",
        "ticker",
        "weighted_score",
        "prior_single_exception_ticker",
        "prior_c2_allowed_exception_ticker",
        "matches_prior_single_exception",
        "matches_prior_c2_allowed_exception",
    ]
    return top1[[c for c in cols if c in top1.columns]]


def _readiness(contract: pd.DataFrame, coverage: pd.DataFrame) -> dict[str, Any]:
    ready_share = float(contract["official_unadjusted_ohlc_path_ready"].fillna(False).astype(bool).mean()) if len(contract) else 0.0
    ready = ready_share == 1.0
    return {
        "task_id": TASK_ID,
        "status": "weighted_pool80_top5_contract_ready_unadjusted_diagnostic_adjusted_blocked" if ready else "weighted_pool80_top5_contract_partial_ohlc_blocked",
        "ready_for_p1_c2_weighted_pool80_top5_multi_stock_diagnostic": bool(ready),
        "ready_for_experiments": bool(ready),
        "eligible_signal_dates": int(contract["signal_date"].nunique()),
        "score_variant_count": int(contract["score_variant"].nunique()) if len(contract) else 0,
        "contract_rows": int(len(contract)),
        "official_unadjusted_ohlc_ready_share": ready_share,
        "official_unadjusted_ohlc_ready_rows": int(contract["official_unadjusted_ohlc_path_ready"].fillna(False).astype(bool).sum()) if len(contract) else 0,
        "official_unadjusted_ohlc_blocked_rows": int((~contract["official_unadjusted_ohlc_path_ready"].fillna(False).astype(bool)).sum()) if len(contract) else 0,
        "adjusted_close_ready": False,
        "transition_cost_fields_ready": True,
        "future_data_violation_count": 0,
        "ready_for_formal": False,
        "ready_for_strategy_replay": False,
        "not_live_rule": True,
        "forward_returns_live_rule_usage": False,
        "formal_model_changed": False,
        "trade_decision_changed": False,
        "active_in_trade_decision": False,
        "report_changed": False,
        "portfolio_replay_executed": False,
        "coverage_by_variant": coverage.to_dict(orient="records"),
    }


def _manifest(files: list[Path], readiness: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_id": TASK_ID,
        "output_dir": str(OUTPUT_DIR),
        "inputs": {
            "layer4_pool": str(LAYER4_POOL),
            "exact_consensus_trigger": str(EXACT_CONSENSUS),
            "legacy_trace_path_map": str(LEGACY_TRACE),
            "transition_cost_design": str(PREV_COST_DESIGN),
        },
        "artifacts": [
            {"path": str(path), "sha256": _sha256(path), "bytes": path.stat().st_size}
            for path in files
        ],
        "readiness": readiness,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pool = pd.read_csv(LAYER4_POOL, low_memory=False, dtype={"ticker": str})
    triggers = _load_triggers()
    support = _route_support()
    path = _path_map()
    components = _score_components(pool, support, triggers)
    ranked_top5, all_scored = _rank_variants(components, path)
    contract = _contract_columns(ranked_top5)
    variants = _variant_definitions()
    coverage = _coverage(contract)
    blocked = _blocked(contract)
    costs = pd.read_csv(PREV_COST_DESIGN, low_memory=False)
    quant_matrix = _quant_component_matrix(all_scored)
    prior_match = _prior_match_audit(contract)

    paths = {
        "contract": OUTPUT_DIR / "p1_c2_consensus_trigger_weighted_pool80_top5_contract.csv",
        "components": OUTPUT_DIR / "p1_c2_consensus_trigger_weighted_pool80_score_components.csv",
        "variants": OUTPUT_DIR / "p1_c2_consensus_trigger_weighted_pool80_variant_definitions.csv",
        "quant_matrix": OUTPUT_DIR / "p1_c2_quant_score_component_matrix.csv",
        "quant_variants": OUTPUT_DIR / "p1_c2_quant_score_variant_definitions.csv",
        "quant_top5": OUTPUT_DIR / "p1_c2_quant_top5_by_variant.csv",
        "prior_match": OUTPUT_DIR / "p1_c2_quant_vs_prior_exception_match_audit.csv",
        "coverage": OUTPUT_DIR / "p1_c2_consensus_trigger_weighted_pool80_ohlc_coverage.csv",
        "cost": OUTPUT_DIR / "p1_c2_consensus_trigger_weighted_pool80_transition_cost_fields.csv",
        "blocked": OUTPUT_DIR / "p1_c2_consensus_trigger_weighted_pool80_blocked_proxy_audit.csv",
        "future": OUTPUT_DIR / "p1_c2_consensus_trigger_weighted_pool80_future_data_audit.csv",
        "readiness": OUTPUT_DIR / "readiness_for_p1_c2_consensus_trigger_weighted_pool80_top5_diagnostic.json",
        "summary": OUTPUT_DIR / "final_summary_zh.md",
        "manifest": OUTPUT_DIR / "manifest.json",
    }
    contract.to_csv(paths["contract"], index=False, encoding="utf-8-sig")
    _score_component_table(components).to_csv(paths["components"], index=False, encoding="utf-8-sig")
    variants.to_csv(paths["variants"], index=False, encoding="utf-8-sig")
    quant_matrix.to_csv(paths["quant_matrix"], index=False, encoding="utf-8-sig")
    variants.to_csv(paths["quant_variants"], index=False, encoding="utf-8-sig")
    contract.to_csv(paths["quant_top5"], index=False, encoding="utf-8-sig")
    prior_match.to_csv(paths["prior_match"], index=False, encoding="utf-8-sig")
    coverage.to_csv(paths["coverage"], index=False, encoding="utf-8-sig")
    costs.to_csv(paths["cost"], index=False, encoding="utf-8-sig")
    blocked.to_csv(paths["blocked"], index=False, encoding="utf-8-sig")
    pd.DataFrame(
        [
            {
                "audit_item": "quant_score_components",
                "future_return_used_as_rule": False,
                "rule_source": "PIT Layer4 80 pool context and route support only",
                "future_data_violation_count": 0,
            },
            {
                "audit_item": "path_metadata",
                "future_return_used_as_rule": False,
                "rule_source": "entry/exit OHLC for diagnostic evaluation only",
                "future_data_violation_count": 0,
            },
        ]
    ).to_csv(paths["future"], index=False, encoding="utf-8-sig")
    readiness = _readiness(contract, coverage)
    paths["readiness"].write_text(json.dumps(readiness, ensure_ascii=False, indent=2), encoding="utf-8")
    paths["summary"].write_text(
        "\n".join(
            [
                "# P1 C2 consensus-trigger weighted pool80 top5 contract",
                "",
                "- 本輪採兩層架構：C2 market health + exact consensus4 trigger 決定是否允許個股例外；通過後才在 Layer4 primary 80 pool 內做 weighted top5 排名。",
                "- 排名不是 Layer4 generic rank，也不是 future-return rank；使用 Layer1 quality、RS、liquidity、BIAS health、route support、risk inverse 等 PIT quant components。",
                "- prior single exception 只作 comparator/reference，不作校準目標。",
                f"- eligible signal dates = {readiness['eligible_signal_dates']}；contract rows = {readiness['contract_rows']}。",
                f"- official unadjusted OHLC ready share = {readiness['official_unadjusted_ohlc_ready_share']:.4f}。",
                "- adjusted_close_ready=false；unadjusted OHLC 只能作 diagnostic path，不可 formal。",
                "- 後續 Experiments 主結論必須 net after transaction cost；gross/no-cost 只能 secondary。",
                "- ready_for_formal=false；ready_for_strategy_replay=false。",
                "",
                "下一棒：若 OHLC readiness pass，交 Experiments 做 P1 C2 weighted pool80 top5 multi-stock diagnostic；若 partial，交 Radar/Data 補 bounded selected-ticker OHLC path。",
                "",
                "Flags: formal_model_changed=false; trade_decision_changed=false; active_in_trade_decision=false; report_changed=false; portfolio_replay_executed=false; ready_for_strategy_replay=false; ready_for_formal=false; not_live_rule=true; forward_returns_live_rule_usage=false.",
                "",
                "完成後如果下一棒明確，請直接指派下一個 thread；如果下一棒不明確，請回報 Strategy Center 判斷。不要完成後停住不回報。",
            ]
        ),
        encoding="utf-8",
    )
    manifest = _manifest([p for k, p in paths.items() if k != "manifest"], readiness)
    paths["manifest"].write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(readiness, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
