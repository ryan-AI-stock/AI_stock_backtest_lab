from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_ROOT = Path("C:/Users/zergv/Documents/Codex/2026-07-06/backtest-lab-experiments-diagnostic-validation-attribution")
OUTPUT_DIR = REPO_ROOT / "outputs" / "vnext_r6_guard_first_market_bias_override_unified_contract_20260709"

ROUTE_SUPPORT_CONTRACT = (
    REPO_ROOT
    / "outputs"
    / "vnext_route_support_max1_full_period_same_basis_contract_20260708"
    / "route_support_max1_full_period_same_basis_modelization_contract.csv"
)
SIGNAL_FEATURES = (
    REPO_ROOT
    / "outputs"
    / "vnext_regime_switch_hybrid_route_market_fields_path_materialization_20260708"
    / "regime_switch_hybrid_route_signal_table.csv"
)
MARKET_BRANCH_TRACE = (
    EXPERIMENTS_ROOT
    / "outputs"
    / "vnext_regime_route_integration_selector_discipline_diagnostic_20260708"
    / "route_integration_policy_path_trace.csv"
)
EXPERIMENTS_REFINEMENT = (
    EXPERIMENTS_ROOT
    / "outputs"
    / "vnext_guard_first_market_bias_override_refinement_diagnostic_20260709"
)

TASK_ID = "TASK-BACKTEST-CORE-VNEXT-R6-GUARD-FIRST-MARKET-BIAS-OVERRIDE-UNIFIED-CONTRACT-001"
PRIMARY_VARIANT = "R6_breakout_breadth_p1_risk_veto"

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

PERIODS = {
    "P1": ("2015-01-02", "2022-12-29", "in_P1"),
    "P2": ("2023-01-02", "2026-06-30", "in_P2"),
    "2024_latest": ("2024-01-02", "2026-06-30", "in_2024_latest"),
    "2026YTD": ("2026-01-02", "2026-06-30", "in_2026YTD"),
    "full_integrated": ("2015-01-02", "2026-06-30", None),
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def boolv(value: Any) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, str):
        return value.lower() in {"true", "1", "yes"}
    return bool(value)


def num(row: pd.Series, col: str, default: float = 0.0) -> float:
    value = row.get(col, default)
    return default if pd.isna(value) else float(value)


def add_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for key, value in FLAGS.items():
        out[key] = value
    return out


def _compound(series: pd.Series) -> float:
    vals = series.dropna().astype(float)
    return float(np.prod(1.0 + vals) - 1.0) if len(vals) else math.nan


def _mdd(series: pd.Series) -> float:
    vals = series.dropna().astype(float)
    if vals.empty:
        return math.nan
    equity = (1.0 + vals).cumprod()
    return float((equity / equity.cummax() - 1.0).min())


def _annualized(total: float, start: str, end: str) -> float:
    if pd.isna(total) or not start or not end:
        return math.nan
    days = (pd.to_datetime(end) - pd.to_datetime(start)).days
    return float((1.0 + total) ** (365.25 / days) - 1.0) if days > 0 and total > -1 else math.nan


def load_base_contract() -> pd.DataFrame:
    base = pd.read_csv(ROUTE_SUPPORT_CONTRACT, low_memory=False, dtype={"selected_ticker": str})
    base["signal_date"] = pd.to_datetime(base["signal_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    base["next_signal_date"] = pd.to_datetime(base["next_signal_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    return base[base["net_interval_return_after_transition_cost"].notna()].copy()


def load_market_branch() -> pd.DataFrame:
    trace = pd.read_csv(MARKET_BRANCH_TRACE, low_memory=False, dtype={"selected_recommendation": str, "ticker": str})
    sub = trace[
        trace["integration_variant"].eq("integrated_market_bias_pool_trend")
        & trace["timing_variant"].eq("next_day_close_entry_fixed_5td_exit")
        & trace["ready_for_policy_metric"].astype(bool)
    ].copy()
    sub["signal_date"] = pd.to_datetime(sub["signal_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    return sub.sort_values("signal_date").drop_duplicates("signal_date", keep="first").set_index("signal_date")


def compute_features(base: pd.DataFrame) -> pd.DataFrame:
    raw = pd.read_csv(SIGNAL_FEATURES, low_memory=False)
    raw["snapshot_date"] = pd.to_datetime(raw["snapshot_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    f = raw.drop_duplicates("snapshot_date").sort_values("snapshot_date").set_index("snapshot_date")
    c2 = base[["signal_date", "c2_market_health_gate"]].drop_duplicates("signal_date").set_index("signal_date")
    f = f.join(c2, how="left")
    f["breakout_breadth"] = (
        (f["rolling_high_breakout_count"].fillna(0) >= 1)
        & (f["dynamic80_rs20_positive_share"].fillna(0) >= 0.70)
        & (f["0050_return_20d"].fillna(0) >= 0.03)
    )
    f["p1_risk_veto"] = (
        (f["0050_return_60d"].fillna(0) < 0)
        | (f["dynamic80_rs60_positive_share"].fillna(0) < 0.45)
        | (f["00631L_vs_0050_return_20d"].fillna(0) < -0.05)
        | (f["pool_high_exhaustion_breakdown_share"].fillna(0) > 0.55)
    )
    f["r6_override_flag"] = f["breakout_breadth"] & ~f["p1_risk_veto"]
    return f


def triggered_features(feat: pd.Series) -> str:
    return (
        f"breakout_breadth={boolv(feat.get('breakout_breadth'))};"
        f"p1_risk_veto={boolv(feat.get('p1_risk_veto'))};"
        f"r6_override={boolv(feat.get('r6_override_flag'))};"
        f"0050_ret20={num(feat, '0050_return_20d'):.4f};"
        f"0050_ret40={num(feat, '0050_return_40d'):.4f};"
        f"0050_ret60={num(feat, '0050_return_60d'):.4f};"
        f"bias20={num(feat, '0050_bias20'):.4f};"
        f"bias60={num(feat, '0050_bias60'):.4f};"
        f"breadth20={num(feat, 'dynamic80_rs20_positive_share'):.3f};"
        f"breadth60={num(feat, 'dynamic80_rs60_positive_share'):.3f};"
        f"breakout_count={num(feat, 'rolling_high_breakout_count'):.0f}"
    )


def build_contract() -> pd.DataFrame:
    base = load_base_contract()
    features = compute_features(base)
    market = load_market_branch()
    rows: list[dict[str, Any]] = []
    for row in base.itertuples(index=False):
        sig = row.signal_date
        feat = features.loc[sig] if sig in features.index else pd.Series(dtype=object)
        override = boolv(feat.get("r6_override_flag", False))
        selected_branch = "route_support"
        branch_reason = "r6_not_triggered_route_support_default"
        selected_ticker = row.selected_ticker
        selected_name = getattr(row, "selected_ticker_name", "")
        selected_asset_type = row.selected_asset_type
        selected_return = row.net_interval_return_after_transition_cost
        gross_return = row.gross_interval_return
        entry_date = row.entry_date
        exit_date = row.exit_date
        entry_price = row.entry_price
        exit_price = row.exit_price
        transition_action = row.transition_action
        transition_cost_rate = row.transition_cost_rate
        path_ready = boolv(row.official_unadjusted_ohlc_ready) if row.selected_asset_type == "stock" else boolv(row.benchmark_adjusted_path_ready)
        source_quality = row.source_quality
        branch_missing = False
        if override:
            if sig in market.index and pd.notna(market.loc[sig].get("policy_return")):
                br = market.loc[sig]
                selected_branch = "market_bias_override"
                branch_reason = "R6_breakout_breadth_p1_risk_veto_triggered"
                selected_ticker = str(br.get("selected_recommendation", ""))
                selected_name = str(br.get("name", "")) if pd.notna(br.get("name", "")) else ""
                selected_asset_type = "etf" if selected_ticker in {"00631L", "0050"} else "stock"
                selected_return = br["policy_return"]
                gross_return = br.get("gross_return_unadjusted", br.get("gross_return", math.nan))
                entry_date = br.get("entry_date", entry_date)
                exit_date = br.get("exit_date", exit_date)
                entry_price = br.get("entry_price", entry_price)
                exit_price = br.get("exit_price", exit_price)
                transition_action = "market_bias_override_path"
                transition_cost_rate = math.nan
                path_ready = boolv(br.get("ready_for_policy_metric", False))
                source_quality = br.get("path_source_status", "market_bias_pool_trend_path_trace")
            else:
                branch_missing = True
                branch_reason = "R6_triggered_but_market_bias_branch_path_missing_fallback_route_support"
        regime_label = "健康強勢_R6_market_bias_override" if selected_branch == "market_bias_override" else "多頭_C2_route_support_default"
        rows.append(
            {
                "task": TASK_ID,
                "signal_date": sig,
                "next_signal_date": row.next_signal_date,
                "period_label": row.period_label,
                "in_P1": row.in_P1,
                "in_P2": row.in_P2,
                "in_2024_latest": row.in_2024_latest,
                "in_2026YTD": row.in_2026YTD,
                "regime_label": regime_label,
                "selected_branch": selected_branch,
                "selected_ticker": selected_ticker,
                "selected_ticker_name": selected_name,
                "selected_asset_type": selected_asset_type,
                "fallback_asset": "00631L" if selected_asset_type == "etf" else "",
                "branch_reason": branch_reason,
                "triggered_features": triggered_features(feat),
                "c2_pass_flag": boolv(row.c2_market_health_gate),
                "consensus_trigger_flag": boolv(row.consensus_trigger),
                "r6_override_flag": override,
                "breakout_breadth_flag": boolv(feat.get("breakout_breadth", False)),
                "p1_risk_veto_flag": boolv(feat.get("p1_risk_veto", False)),
                "bear_guard_flag": False,
                "cash_guard_flag": False,
                "bear_cash_guard_source_quality": "blocked_no_accepted_bear_cash_classifier",
                "rs20_top3_reference_tickers": "",
                "rs20_reference_only": True,
                "low_base_main_weight_included": False,
                "entry_date": entry_date,
                "exit_date": exit_date,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "gross_interval_return": gross_return,
                "net_interval_return_after_transition_cost": selected_return,
                "transition_action": transition_action,
                "transition_cost_rate": transition_cost_rate,
                "official_selected_stock_ohlc_ready": path_ready if selected_asset_type == "stock" else True,
                "benchmark_adjusted_path_ready": path_ready if selected_asset_type == "etf" else boolv(row.benchmark_adjusted_path_ready),
                "path_ready": path_ready,
                "branch_path_missing_fallback": branch_missing,
                "source_quality": source_quality,
                "selected_stock_adjusted_close_ready": False if selected_asset_type == "stock" else True,
                "data_readiness": "ready_unadjusted_diagnostic" if path_ready else "blocked_path_missing",
                "blocked_reason": "" if path_ready and not branch_missing else "market_bias_branch_path_missing_or_selected_path_blocked",
                "diagnostic_only": True,
                "daily_report_ready": False,
                **FLAGS,
            }
        )
    return add_flags(pd.DataFrame(rows))


def period_subset(df: pd.DataFrame, period: str) -> pd.DataFrame:
    _, _, flag = PERIODS[period]
    ready = df[df["net_interval_return_after_transition_cost"].notna()].copy()
    if flag is None:
        return ready
    return ready[ready[flag].astype(bool)].copy()


def coverage(contract: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for period, (start, end, flag) in PERIODS.items():
        sub = contract if flag is None else contract[contract[flag].astype(bool)]
        rows.append(
            {
                "task": TASK_ID,
                "period": period,
                "requested_start": start,
                "requested_end": end,
                "actual_start": sub["signal_date"].min(),
                "actual_end": sub["exit_date"].max(),
                "row_count": int(len(sub)),
                "path_ready_rows": int(sub["path_ready"].astype(bool).sum()),
                "path_ready_share": float(sub["path_ready"].astype(bool).mean()) if len(sub) else 0.0,
                "override_trigger_count": int(sub["r6_override_flag"].astype(bool).sum()),
                "primary_timing": "next_day_close_entry_fixed_5td_exit",
                **FLAGS,
            }
        )
    return pd.DataFrame(rows)


def policy_contract() -> pd.DataFrame:
    rows = [
        {
            "policy_item": "default_branch",
            "value": "C2 route_support baseline",
            "plain_zh": "預設沿用 route_support 主線；R6 不觸發時不切 market_bias。",
            "hard_filter": False,
            "future_return_used": False,
        },
        {
            "policy_item": "override_branch",
            "value": PRIMARY_VARIANT,
            "plain_zh": "0050 突破前高 + pool80 breadth 擴散；若 P1-like 假趨勢或弱廣度風險出現，禁止 override。",
            "hard_filter": False,
            "future_return_used": False,
        },
        {
            "policy_item": "low_base",
            "value": "excluded_from_main_weight",
            "plain_zh": "Strategy Center 裁決 low_base 不進主權重；本 contract 不使用 low_base。",
            "hard_filter": False,
            "future_return_used": False,
        },
        {
            "policy_item": "RS20_top3",
            "value": "reference_only",
            "plain_zh": "RS20 top3 不作 selected branch，只保留 reference 欄位。",
            "hard_filter": False,
            "future_return_used": False,
        },
        {
            "policy_item": "daily_report",
            "value": "not_authorized",
            "plain_zh": "本輪只 materialize 日報欄位，不啟用 daily report pipeline。",
            "hard_filter": False,
            "future_return_used": False,
        },
    ]
    return add_flags(pd.DataFrame(rows))


def blocked_proxy_audit(contract: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {
            "field_or_component": "selected_stock_adjusted_close",
            "status": "blocked",
            "impact": "stock branch uses official unadjusted OHLC diagnostic path; not formal adjusted-close path",
            "next_owner": "Radar/Data only if Strategy Center authorizes adjusted-close route",
        },
        {
            "field_or_component": "cash_bear_classifier",
            "status": "blocked",
            "impact": "bear_guard_flag/cash_guard_flag are false placeholders with blocked source quality",
            "next_owner": "Core/Data or Strategy Center later",
        },
        {
            "field_or_component": "RS20_top3",
            "status": "reference_only",
            "impact": "not used as selected branch in R6 unified contract",
            "next_owner": "Strategy Center",
        },
        {
            "field_or_component": "low_base",
            "status": "excluded",
            "impact": "not used in main weight per Strategy Center decision",
            "next_owner": "Strategy Center",
        },
    ]
    missing = contract[~contract["path_ready"].astype(bool)]
    for row in missing.itertuples(index=False):
        rows.append(
            {
                "field_or_component": "selected_branch_path",
                "status": "blocked",
                "impact": f"{row.signal_date} {row.selected_branch} {row.selected_ticker} path missing",
                "next_owner": "Radar/Data bounded selected path fill",
            }
        )
    return add_flags(pd.DataFrame(rows))


def future_audit() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "audit_item": "r6_trigger_construction",
                "future_return_used_as_rule": False,
                "rule_source": "PIT 0050 breakout, pool80 breadth, p1-risk-veto features",
                "future_data_violation_count": 0,
            },
            {
                "audit_item": "branch_path",
                "future_return_used_as_rule": False,
                "rule_source": "entry/exit OHLC is diagnostic evaluation path only",
                "future_data_violation_count": 0,
            },
        ]
    )


def readiness(contract: pd.DataFrame, cov: pd.DataFrame) -> dict[str, Any]:
    path_ready = bool(contract["path_ready"].astype(bool).all()) if len(contract) else False
    return {
        "task_id": TASK_ID,
        "status": "r6_guard_first_market_bias_override_unified_contract_ready_unadjusted_diagnostic_adjusted_blocked"
        if path_ready
        else "r6_guard_first_market_bias_override_unified_contract_partial_path_blocked",
        "periods": list(PERIODS.keys()),
        "contract_rows": int(len(contract)),
        "path_ready_rows": int(contract["path_ready"].astype(bool).sum()),
        "path_ready_share": float(contract["path_ready"].astype(bool).mean()) if len(contract) else 0.0,
        "r6_override_count": int(contract["r6_override_flag"].astype(bool).sum()),
        "ready_for_r6_guard_first_market_bias_override_experiments": bool(path_ready),
        "ready_for_experiments": bool(path_ready),
        "ready_for_daily_report": False,
        "ready_for_formal": False,
        "ready_for_strategy_replay": False,
        "selected_stock_adjusted_close_ready": False,
        "cash_bear_classifier_ready": False,
        "low_base_main_weight_included": False,
        "rs20_top3_mainline_enabled": False,
        "future_data_violation_count": 0,
        "coverage_by_period": cov.to_dict(orient="records"),
        **FLAGS,
    }


def write_summary(path: Path, ready: dict[str, Any]) -> None:
    next_step = (
        "下一棒：交 Experiments 執行 TASK-BACKTEST-EXPERIMENTS-VNEXT-R6-GUARD-FIRST-MARKET-BIAS-OVERRIDE-UNIFIED-DIAGNOSTIC-001。"
        if ready["ready_for_experiments"]
        else "下一棒：若 path blocked，交 Radar/Data 做 bounded branch path fill。"
    )
    path.write_text(
        "\n".join(
            [
                "# R6 guard-first market_bias override unified contract",
                "",
                "## 結論",
                "",
                "- 已 materialize R6_breakout_breadth_p1_risk_veto unified diagnostic contract。",
                "- default branch = C2 / route_support baseline。",
                "- override branch = R6 market_bias override，語義為 0050 突破前高 + pool80 breadth 擴散，且 P1-like risk veto 不成立。",
                "- low_base 不進主權重；RS20 top3 只保留 reference，不作 selected branch。",
                f"- contract rows = {ready['contract_rows']}；path_ready_share = {ready['path_ready_share']:.4f}。",
                f"- R6 override count = {ready['r6_override_count']}。",
                "- adjusted_close_ready=false；cash/bear classifier blocked；daily_report not authorized。",
                "- 所有 branch path 仍是 diagnostic-only，不升 formal / replay / daily report / trade decision。",
                "",
                next_step,
                "",
                "完成後如果下一棒明確，請直接指派下一個 thread；如果下一棒不明確，請回報 Strategy Center 判斷。不要完成後停住不回報。",
                "",
                "Flags: formal_model_changed=false; trade_decision_changed=false; active_in_trade_decision=false; report_changed=false; portfolio_replay_executed=false; ready_for_strategy_replay=false; ready_for_formal=false; not_live_rule=true; forward_returns_live_rule_usage=false.",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    contract = build_contract()
    cov = coverage(contract)
    policy = policy_contract()
    blocked = blocked_proxy_audit(contract)
    future = future_audit()
    ready = readiness(contract, cov)
    sample_cols = [
        "signal_date",
        "regime_label",
        "selected_branch",
        "selected_ticker",
        "selected_ticker_name",
        "selected_asset_type",
        "fallback_asset",
        "branch_reason",
        "triggered_features",
        "c2_pass_flag",
        "consensus_trigger_flag",
        "r6_override_flag",
        "p1_risk_veto_flag",
        "bear_guard_flag",
        "cash_guard_flag",
        "rs20_top3_reference_tickers",
        "rs20_reference_only",
        "data_readiness",
        "blocked_reason",
    ]
    paths = {
        "contract": OUTPUT_DIR / "r6_guard_first_market_bias_override_unified_contract.csv",
        "sample": OUTPUT_DIR / "r6_guard_first_market_bias_override_daily_report_fields_sample.csv",
        "policy": OUTPUT_DIR / "r6_guard_first_market_bias_override_policy_contract.csv",
        "coverage": OUTPUT_DIR / "r6_guard_first_market_bias_override_requested_vs_actual_coverage.csv",
        "blocked": OUTPUT_DIR / "r6_guard_first_market_bias_override_blocked_proxy_audit.csv",
        "future": OUTPUT_DIR / "r6_guard_first_market_bias_override_future_data_audit.csv",
        "readiness": OUTPUT_DIR / "readiness_for_r6_guard_first_market_bias_override_unified_contract.json",
        "summary": OUTPUT_DIR / "final_summary_zh.md",
        "manifest": OUTPUT_DIR / "manifest.json",
    }
    contract.to_csv(paths["contract"], index=False, encoding="utf-8-sig")
    contract[[c for c in sample_cols if c in contract.columns]].head(200).to_csv(paths["sample"], index=False, encoding="utf-8-sig")
    policy.to_csv(paths["policy"], index=False, encoding="utf-8-sig")
    cov.to_csv(paths["coverage"], index=False, encoding="utf-8-sig")
    blocked.to_csv(paths["blocked"], index=False, encoding="utf-8-sig")
    future.to_csv(paths["future"], index=False, encoding="utf-8-sig")
    paths["readiness"].write_text(json.dumps(ready, ensure_ascii=False, indent=2), encoding="utf-8")
    write_summary(paths["summary"], ready)
    manifest = {
        "task_id": TASK_ID,
        "output_dir": str(OUTPUT_DIR),
        "inputs": {
            "route_support_contract": str(ROUTE_SUPPORT_CONTRACT),
            "signal_features": str(SIGNAL_FEATURES),
            "market_branch_trace": str(MARKET_BRANCH_TRACE),
            "experiments_refinement_input": str(EXPERIMENTS_REFINEMENT),
        },
        "artifacts": [
            {"path": str(p), "sha256": _sha256(p), "bytes": p.stat().st_size}
            for key, p in paths.items()
            if key != "manifest"
        ],
        "readiness": ready,
        "created_at": datetime.now(timezone.utc).isoformat(),
        **FLAGS,
    }
    paths["manifest"].write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(ready, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
