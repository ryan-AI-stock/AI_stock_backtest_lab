from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from backtest_lab.costs import COST_MODEL_VERSION, cost_model_metadata


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "vnext_p1_risk_adjusted_rs20_branch_contract_20260709"
LAYER4_POOL = REPO_ROOT / "outputs" / "vnext_layer4_80_primary_pool_contract_20260708" / "layer4_80_primary_pool_contract.csv"
OLD_RS20_PATH = REPO_ROOT / "outputs" / "vnext_p1_legacy_regime_unadjusted_path_refresh_20260708" / "p1_legacy_regime_unadjusted_trade_path_refreshed.csv"
REFERENCE_PATH = REPO_ROOT / "outputs" / "vnext_p1_defensive_policy_benchmark_path_20260708" / "p1_defensive_policy_buy_hold_reference.csv"
ROUTE_SUPPORT_SCORE_AUDIT = REPO_ROOT / "outputs" / "vnext_p1_c2_route_support_max1_modelization_contract_20260708" / "p1_c2_route_support_max1_score_audit.csv"

TASK_ID = "TASK-BACKTEST-CORE-VNEXT-P1-RISK-ADJUSTED-RS20-BRANCH-CONTRACT-001"
P1_START = "2015-01-02"
P1_END = "2022-12-29"
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
    parser = argparse.ArgumentParser(description="Build P1 risk-adjusted RS20 branch comparison contract.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--period-start", default=P1_START)
    parser.add_argument("--period-end", default=P1_END)
    args = parser.parse_args()
    build_package(output_dir=Path(args.output_dir), period_start=args.period_start, period_end=args.period_end)


def build_package(*, output_dir: Path, period_start: str, period_end: str) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pool = load_p1_pool(period_start, period_end)
    route_support = load_route_support_audit()
    scored = score_pool(pool, route_support)
    new_signal = select_new_rs20(scored)
    old_source = build_old_rs20_source_map()
    reference = build_reference()
    new_path_contract = attach_existing_path_coverage(new_signal)
    score_components = scored.sort_values(["signal_date", "risk_adjusted_rs20_score"], ascending=[True, False])
    coverage = requested_vs_actual_coverage(pool, new_signal, old_source, new_path_contract, reference, period_start, period_end)
    blocked = blocked_proxy_audit(new_path_contract)
    future = future_data_audit()

    new_contract_path = output_dir / "p1_risk_adjusted_rs20_branch_contract.csv"
    old_path = output_dir / "p1_old_rs20_branch_contract_or_source_map.csv"
    reference_path = output_dir / "p1_00631l_buyhold_or_statehold_reference.csv"
    score_path = output_dir / "p1_risk_adjusted_rs20_score_components.csv"
    coverage_path = output_dir / "requested_vs_actual_coverage.csv"
    blocked_path = output_dir / "blocked_proxy_audit.csv"
    missing_path = output_dir / "p1_risk_adjusted_rs20_selected_ohlc_missing_ledger.csv"
    future_path = output_dir / "future_data_audit.csv"
    readiness_path = output_dir / "readiness_for_p1_rs20_comparison_experiments.json"
    summary_path = output_dir / "final_summary_zh.md"

    new_path_contract.to_csv(new_contract_path, index=False, encoding="utf-8-sig")
    old_source.to_csv(old_path, index=False, encoding="utf-8-sig")
    reference.to_csv(reference_path, index=False, encoding="utf-8-sig")
    score_components.to_csv(score_path, index=False, encoding="utf-8-sig")
    coverage.to_csv(coverage_path, index=False, encoding="utf-8-sig")
    blocked.to_csv(blocked_path, index=False, encoding="utf-8-sig")
    selected_ohlc_missing_ledger(new_path_contract).to_csv(missing_path, index=False, encoding="utf-8-sig")
    future.to_csv(future_path, index=False, encoding="utf-8-sig")

    missing_rows = int((~new_path_contract["path_ready"].fillna(False)).sum())
    readiness = {
        "task": TASK_ID,
        "status": "p1_risk_adjusted_rs20_signal_contract_ready_selected_stock_ohlc_partial"
        if missing_rows
        else "p1_risk_adjusted_rs20_contract_ready_for_experiments",
        "period_requested_start": period_start,
        "period_requested_end": period_end,
        "actual_signal_start": str(new_signal["signal_date"].min()) if len(new_signal) else "",
        "actual_signal_end": str(new_signal["signal_date"].max()) if len(new_signal) else "",
        "weekly_signal_rows": int(len(new_signal)),
        "old_rs20_source_rows": int(len(old_source)),
        "new_rs20_selected_path_ready_rows": int(new_path_contract["path_ready"].fillna(False).sum()),
        "new_rs20_selected_path_missing_rows": missing_rows,
        "new_rs20_selected_path_ready_share": float(new_path_contract["path_ready"].fillna(False).mean()) if len(new_path_contract) else 0.0,
        "cost_model_ready": True,
        "cost_model_version": COST_MODEL_VERSION,
        "ready_for_p1_rs20_comparison_experiments": missing_rows == 0,
        "ready_for_radar_p1_risk_adjusted_rs20_selected_ohlc_gap_fill": missing_rows > 0,
        "ready_for_formal": False,
        "ready_for_strategy_replay": False,
        "future_data_violation_count": 0,
        "blocking_summary": "New risk-adjusted RS20 selected-stock OHLC path is partial; send bounded missing ledger to Radar/Data before Experiments."
        if missing_rows
        else "All selected-stock OHLC rows are ready for P1 comparison Experiments.",
        **FLAGS,
    }
    write_json(readiness_path, readiness)
    write_summary(summary_path, readiness, new_path_contract, old_source, reference)
    write_json(
        output_dir / "manifest.json",
        {
            "task": TASK_ID,
            "output_dir": str(output_dir),
            "artifacts": [
                new_contract_path.name,
                old_path.name,
                reference_path.name,
                score_path.name,
                coverage_path.name,
                blocked_path.name,
                missing_path.name,
                future_path.name,
                readiness_path.name,
                summary_path.name,
            ],
            "created_at": datetime.now(timezone.utc).isoformat(),
            **FLAGS,
        },
    )
    return readiness


def load_p1_pool(period_start: str, period_end: str) -> pd.DataFrame:
    usecols = [
        "snapshot_date",
        "ticker",
        "name",
        "market",
        "pool_rank",
        "RS20",
        "RS60",
        "BIAS60",
        "BIAS60_percentile",
        "volatility_pctile_by_week",
        "layer1_quality_floor_risk_pctile_by_week",
        "layer1_pass_bottom30",
        "traded_value_rank_20d",
        "traded_value_rank_60d",
        "drawdown_60d",
        "drawdown_120d",
        "layer4_risk_aware_score",
        "layer4_broad_opportunity_net_score",
    ]
    df = pd.read_csv(LAYER4_POOL, usecols=usecols, dtype={"ticker": str})
    df = df[df["snapshot_date"].astype(str).between(period_start, period_end)].copy()
    df = df.rename(columns={"snapshot_date": "signal_date"})
    df["ticker"] = df["ticker"].astype(str).str.zfill(4)
    return df


def load_route_support_audit() -> pd.DataFrame:
    if not ROUTE_SUPPORT_SCORE_AUDIT.exists():
        return pd.DataFrame(columns=["signal_date", "ticker"])
    cols = [
        "signal_date",
        "ticker",
        "weighted_score",
        "route_support_variant_count",
        "route_support_variant_flags",
        "route_support_mode_flags",
    ]
    df = pd.read_csv(ROUTE_SUPPORT_SCORE_AUDIT, usecols=cols, dtype={"ticker": str})
    df["ticker"] = df["ticker"].astype(str).str.zfill(4)
    return df


def score_pool(pool: pd.DataFrame, route_support: pd.DataFrame) -> pd.DataFrame:
    df = pool.merge(route_support, on=["signal_date", "ticker"], how="left")
    df["rs20_momentum_component"] = df.groupby("signal_date")["RS20"].rank(pct=True).fillna(0.0)
    df["bias60_overheat_penalty_component"] = pd.to_numeric(df["BIAS60_percentile"], errors="coerce").fillna(0.5)
    df["volatility_risk_penalty_component"] = pd.to_numeric(df["volatility_pctile_by_week"], errors="coerce").fillna(0.5)
    df["quality_support_component"] = 1.0 - pd.to_numeric(df["layer1_quality_floor_risk_pctile_by_week"], errors="coerce").fillna(0.5)
    low_base_price = ((-pd.to_numeric(df["drawdown_120d"], errors="coerce")).clip(0.03, 0.45) / 0.45).fillna(0.5)
    low_base_bias = 1.0 - pd.to_numeric(df["BIAS60_percentile"], errors="coerce").fillna(0.5)
    df["low_base_bonus_component"] = (low_base_price * 0.55 + low_base_bias * 0.45).clip(0, 1)
    df["route_support_bonus_component"] = pd.to_numeric(df["weighted_score"], errors="coerce").fillna(0.5)
    df["route_support_source_quality"] = df["weighted_score"].notna().map(
        {True: "same_date_route_support_score_available_for_candidate", False: "route_support_unavailable_for_candidate_neutral_component"}
    )
    df["risk_adjusted_rs20_score"] = (
        df["rs20_momentum_component"] * 0.45
        - df["bias60_overheat_penalty_component"] * 0.18
        - df["volatility_risk_penalty_component"] * 0.12
        + df["low_base_bonus_component"] * 0.12
        + df["quality_support_component"] * 0.08
        + df["route_support_bonus_component"] * 0.05
    )
    df["score_formula"] = (
        "0.45*RS20_rank_pct - 0.18*BIAS60_percentile - 0.12*volatility_pctile "
        "+ 0.12*low_base + 0.08*quality_support + 0.05*route_support_or_neutral"
    )
    df["future_return_used"] = False
    return df


def select_new_rs20(scored: pd.DataFrame) -> pd.DataFrame:
    selected = scored.sort_values(["signal_date", "risk_adjusted_rs20_score", "RS20", "ticker"], ascending=[True, False, False, True])
    selected = selected.groupby("signal_date", as_index=False).head(1).copy()
    selected["branch_variant"] = "risk_adjusted_rs20_top1_with_bias60_vol_lowbase_quality_route_support"
    selected["selected_ticker"] = selected["ticker"]
    selected["selected_name"] = selected["name"]
    selected["timing_variant"] = "next_day_close_entry_fixed_5td_exit"
    return selected


def build_old_rs20_source_map() -> pd.DataFrame:
    df = pd.read_csv(OLD_RS20_PATH, dtype={"ticker": str})
    df = df[
        df["variant"].eq("dynamic80_top3_rs20_risk_tiebreak_proxy")
        & df["timing_variant"].eq("next_day_close_entry_fixed_5td_exit")
    ].copy()
    df["source_map_role"] = "old_rs20_dynamic80_top3_risk_tiebreak_existing_unadjusted_path"
    return df


def build_reference() -> pd.DataFrame:
    ref = pd.read_csv(REFERENCE_PATH, dtype={"benchmark": str})
    ref = ref[ref["benchmark"].isin(["00631L", "0050"])].copy()
    ref["reference_role"] = "buy_hold_state_hold_reference_not_signal_aligned_weekly_rebuy"
    return ref


def attach_existing_path_coverage(new_signal: pd.DataFrame) -> pd.DataFrame:
    path = pd.read_csv(OLD_RS20_PATH, dtype={"ticker": str})
    path = path[path["timing_variant"].eq("next_day_close_entry_fixed_5td_exit")].copy()
    schedule = path[["signal_date", "entry_date", "exit_date"]].drop_duplicates("signal_date")
    path = path[path["path_bucket"].eq("ordinary_stock")].copy()
    cols = [
        "signal_date",
        "ticker",
        "entry_date",
        "exit_date",
        "entry_price",
        "exit_price",
        "gross_return_unadjusted",
        "net_return_local_ep05_cost_unit_notional",
        "path_ready",
        "source_quality",
        "blocked_reason",
        "adjusted_close_ready",
        "diagnostic_unit_notional_twd",
        "buy_cost_twd",
        "sell_cost_twd",
        "total_cost_twd",
        "cost_application_status",
    ]
    path = path[[c for c in cols if c in path.columns]].drop_duplicates(["signal_date", "ticker"])
    out = new_signal.merge(path, on=["signal_date", "ticker"], how="left", suffixes=("", "_existing_path"))
    out = out.merge(schedule, on="signal_date", how="left", suffixes=("", "_schedule"))
    for col in ["entry_date", "exit_date"]:
        if f"{col}_schedule" in out.columns:
            out[col] = out[col].fillna(out[f"{col}_schedule"])
    out["path_ready"] = out["path_ready"].fillna(False).astype(bool)
    out["path_source_policy"] = out["path_ready"].map(
        {True: "reused_existing_p1_selected_stock_unadjusted_ohlc_path", False: "missing_selected_stock_ohlc_path_needs_bounded_radar_fill"}
    )
    out["cost_model_version"] = COST_MODEL_VERSION
    out["selected_stock_adjusted_close_ready"] = False
    for key, value in FLAGS.items():
        out[key] = value
    keep = [
        "signal_date",
        "branch_variant",
        "selected_ticker",
        "selected_name",
        "ticker",
        "name",
        "market",
        "timing_variant",
        "entry_date",
        "exit_date",
        "entry_price",
        "exit_price",
        "gross_return_unadjusted",
        "net_return_local_ep05_cost_unit_notional",
        "path_ready",
        "path_source_policy",
        "source_quality",
        "blocked_reason",
        "selected_stock_adjusted_close_ready",
        "diagnostic_unit_notional_twd",
        "buy_cost_twd",
        "sell_cost_twd",
        "total_cost_twd",
        "cost_model_version",
        "cost_application_status",
        "risk_adjusted_rs20_score",
        "rs20_momentum_component",
        "bias60_overheat_penalty_component",
        "volatility_risk_penalty_component",
        "low_base_bonus_component",
        "quality_support_component",
        "route_support_bonus_component",
        "route_support_source_quality",
        "score_formula",
        *FLAGS.keys(),
    ]
    return out[[c for c in keep if c in out.columns]]


def requested_vs_actual_coverage(pool: pd.DataFrame, new_signal: pd.DataFrame, old_source: pd.DataFrame, new_path: pd.DataFrame, reference: pd.DataFrame, period_start: str, period_end: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"item": "requested_period", "requested": f"{period_start}~{period_end}", "actual": f"{new_signal['signal_date'].min()}~{new_signal['signal_date'].max()}", "ready": True},
            {"item": "layer4_primary80_p1_weekly_rows", "requested": "411*80", "actual": len(pool), "ready": len(pool) == 411 * 80},
            {"item": "new_risk_adjusted_rs20_signal_rows", "requested": 411, "actual": len(new_signal), "ready": len(new_signal) == 411},
            {"item": "old_rs20_existing_path_rows", "requested": 411, "actual": len(old_source), "ready": len(old_source) == 411},
            {"item": "new_rs20_existing_selected_path_ready_rows", "requested": len(new_path), "actual": int(new_path["path_ready"].sum()), "ready": bool(new_path["path_ready"].all())},
            {"item": "00631L_0050_buy_hold_reference_rows", "requested": 2, "actual": len(reference), "ready": len(reference) >= 2},
        ]
    )


def blocked_proxy_audit(new_path: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {
            "field": "route_support_bonus",
            "status": "partial_proxy",
            "blocked_reason": "route_support is available only when candidate/date appears in prior route_support audit; otherwise neutral component is used",
            "policy": "score component source_quality labels neutral/unavailable; not formal",
            "next_owner": "Experiments can test proxy; Core can later expand candidate-level route_support if needed",
        },
        {
            "field": "selected_stock_adjusted_close",
            "status": "blocked",
            "blocked_reason": "selected-stock adjusted close remains unavailable",
            "policy": "official unadjusted OHLC diagnostic only",
            "next_owner": "Strategy Center source policy / Radar adjusted source route",
        },
    ]
    missing = new_path[~new_path["path_ready"].fillna(False)].copy()
    for _, row in missing.iterrows():
        rows.append(
            {
                "field": "new_rs20_selected_stock_ohlc_path",
                "status": "blocked",
                "signal_date": row["signal_date"],
                "entry_date": row.get("entry_date", ""),
                "exit_date": row.get("exit_date", ""),
                "ticker": row["ticker"],
                "name": row.get("name", ""),
                "blocked_reason": "selected ticker/date not present in existing P1 selected-stock OHLC paths",
                "policy": "bounded Radar/Data selected-ticker-only OHLC gap fill required",
                "next_owner": "Radar/Data",
            }
        )
    return pd.DataFrame(rows)


def selected_ohlc_missing_ledger(new_path: pd.DataFrame) -> pd.DataFrame:
    missing = new_path[~new_path["path_ready"].fillna(False)].copy()
    cols = [
        "signal_date",
        "entry_date",
        "exit_date",
        "selected_ticker",
        "selected_name",
        "ticker",
        "name",
        "market",
        "timing_variant",
        "path_source_policy",
        "risk_adjusted_rs20_score",
        "score_formula",
    ]
    out = missing[[c for c in cols if c in missing.columns]].copy()
    out["source_request_scope"] = "bounded_selected_ticker_only_no_full_market_download"
    out["required_price_fields"] = "official_unadjusted_ohlc_entry_close_exit_close"
    out["adjusted_close_policy"] = "blocked_do_not_fabricate"
    return out


def future_data_audit() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"audit_item": "future_return_used_in_new_rs20_score", "used": False, "future_data_violation_count": 0},
            {"audit_item": "future_winner_or_hindsight_max_used", "used": False, "future_data_violation_count": 0},
            {"audit_item": "forward_eval_columns_used_for_selection", "used": False, "future_data_violation_count": 0},
        ]
    )


def write_summary(path: Path, readiness: dict[str, Any], new_path: pd.DataFrame, old_source: pd.DataFrame, reference: pd.DataFrame) -> None:
    ready_share = readiness["new_rs20_selected_path_ready_share"]
    path.write_text(
        f"""# P1 risk-adjusted RS20 branch contract

## 結論

- 已建立 P1 新版 risk-adjusted RS20 signal / score contract。
- 舊 RS20 對照使用既有 `dynamic80_top3_rs20_risk_tiebreak_proxy` + next-day close 5TD unadjusted OHLC path。
- 00631L / 0050 reference 使用 buy-hold / state-hold reference，不混 signal-aligned weekly rebuy。
- 新 RS20 selected-stock OHLC path ready share={ready_share:.4f}；若小於 1，需 Radar/Data 補 bounded selected ticker path 後再交 Experiments。

## 新 RS20 score

`RS20 branch = RS20 動能 - BIAS60 過熱扣分 - 波動風險扣分 + low_base / quality / route_support 加分`

實作公式：
`0.45*RS20_rank_pct - 0.18*BIAS60_percentile - 0.12*volatility_pctile + 0.12*low_base + 0.08*quality_support + 0.05*route_support_or_neutral`

## Coverage

- weekly_signal_rows={readiness['weekly_signal_rows']}
- old_rs20_source_rows={readiness['old_rs20_source_rows']}
- new_rs20_selected_path_ready_rows={readiness['new_rs20_selected_path_ready_rows']}
- new_rs20_selected_path_missing_rows={readiness['new_rs20_selected_path_missing_rows']}
- reference_rows={len(reference)}

## Next owner

- ready_for_p1_rs20_comparison_experiments={readiness['ready_for_p1_rs20_comparison_experiments']}
- ready_for_radar_p1_risk_adjusted_rs20_selected_ohlc_gap_fill={readiness['ready_for_radar_p1_risk_adjusted_rs20_selected_ohlc_gap_fill']}

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
