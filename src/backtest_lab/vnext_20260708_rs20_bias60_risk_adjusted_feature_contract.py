from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SIGNAL_DIR = REPO_ROOT / "outputs" / "vnext_adhoc_20260708_eod_signal_materialization_refresh_20260708"
DEFAULT_LOW_BASE_DIR = REPO_ROOT / "outputs" / "vnext_layer4_low_base_score_contract_20260709"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "vnext_20260708_rs20_bias60_risk_adjusted_candidate_feature_contract_20260709"
TASK_ID = "TASK-BACKTEST-CORE-VNEXT-20260708-RS20-BIAS60-RISK-ADJUSTED-CANDIDATE-FEATURE-CONTRACT-001"
AS_OF_DATE = "2026-07-08"
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Build 2026-07-08 RS20 / BIAS60 risk-adjusted feature contract.")
    parser.add_argument("--signal-dir", default=str(DEFAULT_SIGNAL_DIR))
    parser.add_argument("--low-base-dir", default=str(DEFAULT_LOW_BASE_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--as-of-date", default=AS_OF_DATE)
    args = parser.parse_args()
    build_package(
        signal_dir=Path(args.signal_dir),
        low_base_dir=Path(args.low_base_dir),
        output_dir=Path(args.output_dir),
        as_of_date=args.as_of_date,
    )


def build_package(*, signal_dir: Path, low_base_dir: Path, output_dir: Path, as_of_date: str) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    component_path = low_base_dir / "layer4_low_base_score_component_matrix.csv"
    layer0_path = signal_dir / "vnext_adhoc_20260708_layer0_compact_active_universe.csv"
    rs20_path = signal_dir / "vnext_adhoc_20260708_rs20_top3_reference.csv"

    base = pd.read_csv(component_path, dtype={"ticker": str})
    layer0 = pd.read_csv(layer0_path, dtype={"ticker": str})
    rs20_top3 = pd.read_csv(rs20_path, dtype={"ticker": str})
    contract = build_contract(base, layer0, as_of_date)
    top3 = build_top3_audit(contract, rs20_top3)
    alternatives = build_alternative_candidates(contract)
    coverage = requested_vs_actual_coverage(signal_dir, low_base_dir, contract, as_of_date)
    blocked = blocked_proxy_audit(contract, as_of_date)
    future = future_data_audit()

    contract_path = output_dir / "rs20_bias60_risk_adjusted_candidate_feature_contract.csv"
    top3_path = output_dir / "rs20_bias60_top3_audit_support.csv"
    alternatives_path = output_dir / "rs20_bias60_alternative_candidate_support.csv"
    coverage_path = output_dir / "requested_vs_actual_coverage.csv"
    blocked_path = output_dir / "blocked_proxy_audit.csv"
    future_path = output_dir / "future_data_audit.csv"
    readiness_path = output_dir / "readiness_for_rs20_bias60_risk_adjusted_candidate_diagnostic.json"
    summary_path = output_dir / "final_summary_zh.md"

    contract.to_csv(contract_path, index=False, encoding="utf-8-sig")
    top3.to_csv(top3_path, index=False, encoding="utf-8-sig")
    alternatives.to_csv(alternatives_path, index=False, encoding="utf-8-sig")
    coverage.to_csv(coverage_path, index=False, encoding="utf-8-sig")
    blocked.to_csv(blocked_path, index=False, encoding="utf-8-sig")
    future.to_csv(future_path, index=False, encoding="utf-8-sig")

    active = contract[contract["layer0_active_scope"].fillna(False)]
    readiness = {
        "task": TASK_ID,
        "status": "same_date_layer0_active_rs20_bias60_risk_feature_contract_ready_layer4_route_support_blocked",
        "as_of_date": as_of_date,
        "candidate_feature_rows": int(len(contract)),
        "layer0_active_rows": int(len(active)),
        "rs20_top3_audit_rows": int(len(top3)),
        "alternative_candidate_rows": int(len(alternatives)),
        "layer4_primary80_same_date_ready": False,
        "route_support_same_date_ready": False,
        "layer1_quality_same_date_ready": False,
        "ready_for_rs20_bias60_risk_adjusted_candidate_diagnostic": True,
        "ready_for_selected_signal": False,
        "ready_for_experiments": True,
        "ready_for_formal": False,
        "ready_for_strategy_replay": False,
        "future_data_violation_count": 0,
        "source_quality": "official_unadjusted_close_layer0_active_scope_with_same_date_risk_proxy_fields",
        "blocking_summary": "Exact 2026-07-08 Layer4 primary80 / route_support / consensus trigger remain blocked, so this package supports same-date RS20/BIAS60 risk-adjusted diagnostic only, not selected signal.",
        **FLAGS,
    }
    write_json(readiness_path, readiness)
    write_summary(summary_path, readiness, top3, alternatives)
    write_json(
        output_dir / "manifest.json",
        {
            "task": TASK_ID,
            "output_dir": str(output_dir),
            "source_signal_dir": str(signal_dir),
            "source_low_base_dir": str(low_base_dir),
            "artifacts": [
                contract_path.name,
                top3_path.name,
                alternatives_path.name,
                coverage_path.name,
                blocked_path.name,
                future_path.name,
                readiness_path.name,
                summary_path.name,
            ],
            "created_at": datetime.now(timezone.utc).isoformat(),
            **FLAGS,
        },
    )
    return readiness


def build_contract(base: pd.DataFrame, layer0: pd.DataFrame, as_of_date: str) -> pd.DataFrame:
    df = base.copy()
    df["ticker"] = df["ticker"].astype(str).str.zfill(4)
    l0_cols = [
        "ticker",
        "turnover_value",
        "turnover_5d",
        "layer0_core_top250",
        "layer0_buffer_candidate_251_300",
        "buffer_confirmation",
        "layer0_active_scope",
    ]
    l0 = layer0[[c for c in l0_cols if c in layer0.columns]].copy()
    l0["ticker"] = l0["ticker"].astype(str).str.zfill(4)
    df = df.merge(l0.drop_duplicates("ticker"), on="ticker", how="left", suffixes=("", "_layer0"))
    for col in ["layer0_core_top250", "layer0_buffer_candidate_251_300", "buffer_confirmation", "layer0_active_scope"]:
        df[col] = df[col].fillna(False).astype(bool)

    df["rs20_rank"] = pd.to_numeric(df["RS20"], errors="coerce").rank(method="first", ascending=False)
    df["rs20_rank_within_layer0_active"] = (
        df.loc[df["layer0_active_scope"], "RS20"].rank(method="first", ascending=False).reindex(df.index)
    )
    df["bias60_raw"] = pd.to_numeric(df["bias60"], errors="coerce")
    df["bias20_raw"] = pd.to_numeric(df["bias20"], errors="coerce")
    df["bias120_raw"] = pd.NA
    df["bias120_source_quality"] = "blocked_not_materialized_in_same_date_contract"
    df["bias60_stock_specific_zscore"] = pd.to_numeric(df["bias60_zscore_252d"], errors="coerce")
    df["bias20_stock_specific_zscore"] = pd.to_numeric(df["bias20_zscore_252d"], errors="coerce")
    df["bias60_stock_specific_percentile_proxy"] = df["bias60_stock_specific_zscore"].map(zscore_to_percentile)
    df["bias20_stock_specific_percentile_proxy"] = df["bias20_stock_specific_zscore"].map(zscore_to_percentile)
    df["volatility20"] = pd.to_numeric(df["volatility20"], errors="coerce")
    df["volatility_percentile_cross_section_proxy"] = df["volatility20"].rank(pct=True)
    df["layer1_quality_risk_pctile_proxy"] = pd.to_numeric(df["layer1_quality_floor_risk_pctile_by_week"], errors="coerce")
    df["layer1_pass_bottom30_proxy"] = df["layer1_pass_bottom30"].astype(str).str.lower().eq("true")
    df["route_support_score"] = pd.NA
    df["route_support_variant_count"] = pd.NA
    df["route_support_flags"] = ""
    df["route_support_source_quality"] = "blocked_20260708_route_support_not_materialized"
    df["layer4_risk_aware_score"] = pd.NA
    df["layer4_pool_rank"] = pd.NA
    df["layer4_source_quality"] = "blocked_20260708_layer4_primary80_not_materialized"
    df["overheat_bias60_flag"] = df["bias60_stock_specific_percentile_proxy"] >= 0.95
    df["overheat_volatility_flag"] = df["volatility_percentile_cross_section_proxy"] >= 0.90
    df["rs60_high_short_rs_weakening_proxy"] = (
        (pd.to_numeric(df["RS60"], errors="coerce") > 0.25)
        & (pd.to_numeric(df["RS20"], errors="coerce") < pd.to_numeric(df["RS60"], errors="coerce"))
    )
    df["recent_runup_penalty_flag"] = pd.to_numeric(df["recent_runup_penalty"], errors="coerce") >= 0.65
    if "low_base_score" not in df.columns:
        df["low_base_score"] = (
            pd.to_numeric(df["price_position_low_base"], errors="coerce").fillna(0.5) * 0.20
            + pd.to_numeric(df["stock_specific_bias_score"], errors="coerce").fillna(0.5) * 0.16
            + pd.to_numeric(df["recent_runup_inverse"], errors="coerce").fillna(0.5) * 0.14
            + pd.to_numeric(df["improving_rs_score"], errors="coerce").fillna(0.5) * 0.18
            + pd.to_numeric(df["liquidity_improvement"], errors="coerce").fillna(0.5) * 0.14
            + pd.to_numeric(df["quality_support"], errors="coerce").fillna(0.5) * 0.13
            + pd.to_numeric(df["overheat_inverse"], errors="coerce").fillna(0.5) * 0.05
        )
        df.loc[df["overheat_veto_flag"].fillna(False), "low_base_score"] *= 0.65
        df["low_base_score_source_quality"] = "rebuilt_from_low_base_component_matrix_balanced_weights"
    df["risk_adjusted_rs20_score"] = risk_adjusted_score(df)
    df["risk_adjusted_rs20_rank_active"] = (
        df.loc[df["layer0_active_scope"], "risk_adjusted_rs20_score"].rank(method="first", ascending=False).reindex(df.index)
    )
    df["source_quality"] = "same_date_layer0_candidate_unadjusted_ohlcv_with_risk_proxy_fields"
    df["diagnostic_only"] = True
    df["not_live_trade_decision"] = True
    for key, value in FLAGS.items():
        df[key] = value

    columns = [
        "date",
        "ticker",
        "name",
        "market",
        "close",
        "layer0_active_scope",
        "layer0_core_top250",
        "layer0_buffer_candidate_251_300",
        "buffer_confirmation",
        "RS20",
        "rs20_rank",
        "rs20_rank_within_layer0_active",
        "RS60",
        "bias20_raw",
        "bias60_raw",
        "bias120_raw",
        "bias20_stock_specific_percentile_proxy",
        "bias60_stock_specific_percentile_proxy",
        "bias20_stock_specific_zscore",
        "bias60_stock_specific_zscore",
        "volatility20",
        "volatility_percentile_cross_section_proxy",
        "traded_value_rank_20d",
        "traded_value_rank_60d",
        "layer1_quality_risk_pctile_proxy",
        "layer1_pass_bottom30_proxy",
        "route_support_score",
        "route_support_variant_count",
        "route_support_flags",
        "layer4_risk_aware_score",
        "layer4_pool_rank",
        "overheat_bias60_flag",
        "overheat_volatility_flag",
        "rs60_high_short_rs_weakening_proxy",
        "recent_runup_penalty",
        "recent_runup_penalty_flag",
        "low_base_score",
        "low_base_rank",
        "risk_adjusted_rs20_score",
        "risk_adjusted_rs20_rank_active",
        "source_quality",
        "route_support_source_quality",
        "layer4_source_quality",
        "bias120_source_quality",
        "diagnostic_only",
        "not_live_trade_decision",
        *FLAGS.keys(),
    ]
    df = df[df["date"].astype(str).eq(as_of_date)].copy()
    return df[[c for c in columns if c in df.columns]].sort_values(["layer0_active_scope", "risk_adjusted_rs20_score"], ascending=[False, False])


def risk_adjusted_score(df: pd.DataFrame) -> pd.Series:
    rs = pd.to_numeric(df["RS20"], errors="coerce").rank(pct=True).fillna(0.0)
    liquidity = (1.0 - ((pd.to_numeric(df["traded_value_rank_20d"], errors="coerce") - 1.0) / max(len(df), 1)).clip(0, 1)).fillna(0.0)
    bias_risk = pd.to_numeric(df["bias60_stock_specific_percentile_proxy"], errors="coerce").fillna(0.5)
    vol_risk = pd.to_numeric(df["volatility_percentile_cross_section_proxy"], errors="coerce").fillna(0.5)
    quality_risk = pd.to_numeric(df["layer1_quality_risk_pctile_proxy"], errors="coerce").fillna(0.5)
    low_base = pd.to_numeric(df.get("low_base_score"), errors="coerce").fillna(0.5)
    score = rs * 0.44 + liquidity * 0.16 + low_base * 0.18 + (1 - bias_risk) * 0.11 + (1 - vol_risk) * 0.06 + (1 - quality_risk) * 0.05
    score[df["overheat_bias60_flag"].fillna(False)] *= 0.75
    score[df["overheat_volatility_flag"].fillna(False)] *= 0.90
    return score


def build_top3_audit(contract: pd.DataFrame, top3_ref: pd.DataFrame) -> pd.DataFrame:
    ref = top3_ref[["ticker", "rs20_rank"]].copy()
    ref["ticker"] = ref["ticker"].astype(str).str.zfill(4)
    out = contract.merge(ref, on="ticker", how="inner", suffixes=("", "_reference"))
    out["raw_rs20_top3_reference_only"] = True
    out["risk_adjusted_top10_flag"] = pd.to_numeric(out["risk_adjusted_rs20_rank_active"], errors="coerce") <= 10
    out["risk_adjusted_interpretation"] = out.apply(top3_interpretation, axis=1)
    cols = [
        "date",
        "ticker",
        "name",
        "market",
        "close",
        "rs20_rank_reference",
        "RS20",
        "bias60_raw",
        "bias60_stock_specific_percentile_proxy",
        "bias60_stock_specific_zscore",
        "volatility_percentile_cross_section_proxy",
        "layer1_quality_risk_pctile_proxy",
        "risk_adjusted_rs20_score",
        "risk_adjusted_rs20_rank_active",
        "risk_adjusted_top10_flag",
        "overheat_bias60_flag",
        "overheat_volatility_flag",
        "raw_rs20_top3_reference_only",
        "risk_adjusted_interpretation",
        "source_quality",
    ]
    return out[[c for c in cols if c in out.columns]].sort_values("rs20_rank_reference")


def top3_interpretation(row: pd.Series) -> str:
    rank = pd.to_numeric(row.get("risk_adjusted_rs20_rank_active"), errors="coerce")
    flags = []
    if bool(row.get("overheat_bias60_flag")):
        flags.append("BIAS60_overheated")
    if bool(row.get("overheat_volatility_flag")):
        flags.append("volatility_high")
    if pd.notna(rank) and rank > 10:
        flags.append("not_risk_adjusted_top10")
    if not flags:
        flags.append("risk_adjusted_candidate_possible")
    return "|".join(flags)


def build_alternative_candidates(contract: pd.DataFrame) -> pd.DataFrame:
    active = contract[contract["layer0_active_scope"].fillna(False)].copy()
    active = active.sort_values(["risk_adjusted_rs20_score", "RS20", "ticker"], ascending=[False, False, True]).head(30)
    active["alternative_candidate_reason"] = "risk_adjusted_rs20_high_with_bias60_volatility_low_base_penalty"
    cols = [
        "date",
        "ticker",
        "name",
        "market",
        "close",
        "RS20",
        "rs20_rank_within_layer0_active",
        "bias60_raw",
        "bias60_stock_specific_percentile_proxy",
        "volatility_percentile_cross_section_proxy",
        "low_base_score",
        "risk_adjusted_rs20_score",
        "risk_adjusted_rs20_rank_active",
        "alternative_candidate_reason",
        "source_quality",
    ]
    return active[[c for c in cols if c in active.columns]]


def requested_vs_actual_coverage(signal_dir: Path, low_base_dir: Path, contract: pd.DataFrame, as_of_date: str) -> pd.DataFrame:
    active = contract[contract["layer0_active_scope"].fillna(False)]
    return pd.DataFrame(
        [
            {"field": "as_of_date", "requested": as_of_date, "actual": as_of_date, "ready": True},
            {"field": "layer0_active_candidate_scope", "requested": "2026-07-08", "actual": "2026-07-08", "ready": len(active) > 0, "rows": len(active), "path": str(signal_dir)},
            {"field": "rs20_bias60_feature_contract", "requested": "2026-07-08", "actual": "2026-07-08", "ready": len(contract) > 0, "rows": len(contract), "path": str(low_base_dir)},
            {"field": "layer4_primary80_same_date", "requested": "2026-07-08", "actual": "2026-06-29", "ready": False, "rows": 0},
            {"field": "route_support_same_date", "requested": "2026-07-08", "actual": "2026-06-29", "ready": False, "rows": 0},
            {"field": "selected_stock_adjusted_close", "requested": "2026-07-08", "actual": "", "ready": False, "rows": 0},
        ]
    )


def blocked_proxy_audit(contract: pd.DataFrame, as_of_date: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"field": "layer4_primary80_same_date", "field_as_of_date": "2026-06-29", "status": "blocked", "policy": "use Layer0 active/candidate feature table only; do not claim selected pool"},
            {"field": "route_support_same_date", "field_as_of_date": "2026-06-29", "status": "blocked", "policy": "route_support score/flags blank; no proxy threshold substitution"},
            {"field": "layer1_quality_same_date", "field_as_of_date": "2026-06-29", "status": "proxy", "policy": "latest ticker-level Layer1 quality context only; not formal"},
            {"field": "bias60_stock_specific_percentile", "field_as_of_date": as_of_date, "status": "proxy", "policy": "derived from rolling zscore normal-CDF proxy; label as proxy"},
            {"field": "volatility_percentile", "field_as_of_date": as_of_date, "status": "proxy", "policy": "cross-sectional percentile at same date; not stock-specific history percentile"},
            {"field": "bias120", "field_as_of_date": "", "status": "blocked", "policy": "not materialized in same-date contract"},
            {"field": "selected_stock_adjusted_close", "field_as_of_date": "", "status": "blocked", "policy": "do not fabricate adjusted close"},
        ]
    )


def future_data_audit() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"audit_item": "future_return_used", "used": False, "future_data_violation_count": 0},
            {"audit_item": "future_winner_or_hindsight_max_used", "used": False, "future_data_violation_count": 0},
            {"audit_item": "retrieval_time_used_as_market_date", "used": False, "future_data_violation_count": 0},
        ]
    )


def zscore_to_percentile(value: Any) -> float | None:
    if pd.isna(value):
        return None
    z = float(value)
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def write_summary(path: Path, readiness: dict[str, Any], top3: pd.DataFrame, alternatives: pd.DataFrame) -> None:
    top3_preview = top3[["ticker", "name", "RS20", "bias60_raw", "bias60_stock_specific_percentile_proxy", "volatility_percentile_cross_section_proxy", "risk_adjusted_rs20_rank_active"]].to_csv(index=False)
    alt_preview = alternatives[["ticker", "name", "RS20", "bias60_stock_specific_percentile_proxy", "risk_adjusted_rs20_score", "risk_adjusted_rs20_rank_active"]].head(10).to_csv(index=False)
    path.write_text(
        f"""# 2026-07-08 RS20 / BIAS60 risk-adjusted candidate feature contract

## 結論

- 已建立 same-date feature contract，as_of_date=`2026-07-08`。
- 這是 Layer0 active / candidate universe 的 risk-adjusted diagnostic support，不是 selected signal。
- exact Layer4 primary80、exact consensus trigger、route_support max1 仍 blocked 到 2026-06-29，因此本 package 不可產出主推薦。
- RS20 top3 仍是 reference；新增 BIAS60 percentile/zscore、volatility proxy、Layer1 proxy、low_base/risk-adjusted score 供 Experiments 做排序診斷。

## RS20 top3 audit support

{top3_preview}

## Alternative candidate top10 support

{alt_preview}

## Readiness

- ready_for_rs20_bias60_risk_adjusted_candidate_diagnostic={readiness['ready_for_rs20_bias60_risk_adjusted_candidate_diagnostic']}
- ready_for_selected_signal=false
- future_data_violation_count=0

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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
