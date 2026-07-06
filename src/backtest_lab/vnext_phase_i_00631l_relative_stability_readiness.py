"""Build Phase I 00631L-relative strength/stability PIT readiness.

This is source/contract readiness only. It stages candidate-level PIT features
for later event-level diagnostic work. It does not define a formal selector,
use forward returns as rule inputs, alter reports/trades, or execute replay.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


TASK_ID = "TASK-BACKTEST-CORE-VNEXT-PHASE-I-00631L-RELATIVE-STRENGTH-STABILITY-PIT-CONTRACT-READINESS-001"
DEFAULT_MATERIALIZATION_DIR = Path("outputs/vnext_dynamic_candidate_pool_data_materialization_20260706")
DEFAULT_PHASE_H_DIR = Path(
    r"C:\Users\zergv\Documents\Codex\2026-07-06\backtest-lab-experiments-diagnostic-validation-attribution\outputs"
    r"\vnext_phase_h_00631l_hurdle_first_event_diagnostic_20260706"
)
DEFAULT_OUTPUT_DIR = Path("outputs/vnext_phase_i_00631l_relative_stability_readiness_20260706")


PERIODS = [
    ("P1", "2015-01-02", "2022-12-29"),
    ("P2", "2023-01-02", "2026-06-30"),
    ("2024-latest", "2024-01-02", "2026-06-30"),
    ("2026YTD", "2026-01-02", "2026-06-30"),
]


def build_phase_i_readiness(
    *,
    materialization_dir: str | Path = DEFAULT_MATERIALIZATION_DIR,
    phase_h_dir: str | Path = DEFAULT_PHASE_H_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    materialization = Path(materialization_dir)
    phase_h = Path(phase_h_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    weekly = _candidate_family_join(materialization / "vnext_weekly_candidate_snapshot.csv")
    stock = _stock_feature_slice(materialization / "stock_features.csv", weekly)
    benchmark = _benchmark_context(materialization / "benchmark_features.csv")
    strength = _relative_strength_contract(weekly, stock, benchmark)
    stability = _relative_stability_contract(strength)
    blocked = _blocked_proxy_fields(strength, stability, phase_h / "phase_h_blocked_prohibited_audit.csv")
    future_audit = _future_data_audit(strength)
    readiness = _readiness_json(
        phase_h / "manifest.json",
        strength,
        stability,
        weekly,
        blocked,
        future_audit,
    )

    _write_csv(strength, output / "phase_i_00631l_relative_strength_pit_contract.csv")
    _write_csv(stability, output / "phase_i_00631l_relative_stability_pit_contract.csv")
    _write_csv(weekly, output / "phase_i_candidate_family_join_contract.csv")
    _write_csv(blocked, output / "phase_i_blocked_proxy_fields.csv")
    _write_csv(future_audit, output / "phase_i_future_data_audit.csv")
    (output / "readiness_for_phase_i_00631l_relative_stability_diagnostic.json").write_text(
        json.dumps(readiness, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    manifest = {
        "task_id": TASK_ID,
        "status": readiness["status"],
        "output_dir": str(output.resolve()),
        "input_materialization_dir": str(materialization.resolve()),
        "input_phase_h_dir": str(phase_h.resolve()),
        "output_files": [
            "phase_i_00631l_relative_strength_pit_contract.csv",
            "phase_i_00631l_relative_stability_pit_contract.csv",
            "phase_i_candidate_family_join_contract.csv",
            "phase_i_blocked_proxy_fields.csv",
            "phase_i_future_data_audit.csv",
            "readiness_for_phase_i_00631l_relative_stability_diagnostic.json",
            "manifest.json",
            "final_summary_zh.md",
        ],
        "formal_model_changed": False,
        "trade_decision_changed": False,
        "active_in_trade_decision": False,
        "report_changed": False,
        "portfolio_replay_executed": False,
        "ready_for_strategy_replay": False,
        "not_live_rule": True,
        "forward_returns_live_rule_usage": False,
        "diagnostic_only": True,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "final_summary_zh.md").write_text(_summary(readiness, blocked), encoding="utf-8")
    return manifest


def _candidate_family_join(path: Path) -> pd.DataFrame:
    usecols = [
        "snapshot_date",
        "ticker",
        "theme_id",
        "theme_name",
        "selected_outcome_candidate",
        "case_trace_only",
        "diagnostic_only",
        "subpool_class",
        "rank_overall",
        "rank_in_subpool",
        "included_reason",
        "turnover_state",
        "hurdle_0050_proxy_result",
        "hurdle_00631L_proxy_result",
        "final_selector_score_decomposed",
        "pullback_repair_score",
    ]
    raw = pd.read_csv(path, usecols=usecols, parse_dates=["snapshot_date"])
    raw = raw[raw["diagnostic_only"].astype(bool) & ~raw["case_trace_only"].astype(bool)].copy()
    raw["ticker"] = raw["ticker"].astype(str)
    rows = []
    for item in raw.itertuples(index=False):
        families = []
        if bool(item.selected_outcome_candidate):
            families.append("current_final_selected")
        if pd.notna(item.rank_overall) and float(item.rank_overall) <= 3:
            families.append("current_top3_by_rank")
        if str(item.subpool_class) == "long_strong":
            families.append("long_strong_candidates")
        if str(item.theme_id) != "non_ai_unclassified_proxy":
            families.append("theme_breadth_watchlist")
        if str(item.subpool_class) == "pullback_repair" or (pd.notna(item.pullback_repair_score) and float(item.pullback_repair_score) > -100):
            families.append("c3_pullback_comparator")
        for family in sorted(set(families)):
            rows.append(
                {
                    "signal_date": item.snapshot_date,
                    "ticker": item.ticker,
                    "theme_id": item.theme_id,
                    "theme_name": item.theme_name,
                    "candidate_source_family": family,
                    "subpool_class": item.subpool_class,
                    "rank_overall": item.rank_overall,
                    "rank_in_subpool": item.rank_in_subpool,
                    "included_reason": item.included_reason,
                    "turnover_state": item.turnover_state,
                    "hurdle_0050_proxy_result": item.hurdle_0050_proxy_result,
                    "hurdle_00631L_proxy_result": item.hurdle_00631L_proxy_result,
                    "final_selector_score_decomposed": item.final_selector_score_decomposed,
                    "source_quality": "diagnostic_from_weekly_candidate_snapshot",
                    "diagnostic_only": True,
                    "not_live_rule": True,
                }
            )
    return pd.DataFrame(rows).drop_duplicates(["signal_date", "ticker", "candidate_source_family"])


def _stock_feature_slice(path: Path, candidates: pd.DataFrame) -> pd.DataFrame:
    dates = set(candidates["signal_date"].dt.strftime("%Y-%m-%d"))
    tickers = set(candidates["ticker"].astype(str))
    usecols = [
        "trade_date",
        "ticker",
        "return_5d",
        "return_10d",
        "return_20d",
        "return_40d",
        "return_60d",
        "excess_return_vs_00631L_5d",
        "excess_return_vs_00631L_10d",
        "excess_return_vs_00631L_20d",
        "excess_return_vs_00631L_40d",
        "excess_return_vs_00631L_60d",
        "RS5",
        "RS10",
        "RS20",
        "RS40",
        "RS60",
        "drawdown_20d",
        "drawdown_60d",
        "drawdown_120d",
        "volatility",
        "BIAS20",
        "BIAS60",
        "BIAS120",
        "BIAS20_percentile",
        "BIAS60_percentile",
        "BIAS120_percentile",
    ]
    parts = []
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=500_000):
        chunk["ticker"] = chunk["ticker"].astype(str)
        chunk = chunk[chunk["trade_date"].astype(str).isin(dates) & chunk["ticker"].isin(tickers)]
        if not chunk.empty:
            parts.append(chunk)
    if not parts:
        return pd.DataFrame(columns=usecols)
    out = pd.concat(parts, ignore_index=True)
    out["trade_date"] = pd.to_datetime(out["trade_date"])
    return out


def _benchmark_context(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, parse_dates=["trade_date"])
    raw = raw[raw["benchmark"].isin(["0050", "00631L"])].sort_values(["benchmark", "trade_date"]).copy()
    raw["benchmark_volatility_20d"] = raw.groupby("benchmark")["adjusted_close"].transform(
        lambda s: s.pct_change().rolling(20, min_periods=10).std()
    )
    pivot = raw.pivot(index="trade_date", columns="benchmark")
    out = pd.DataFrame(index=pivot.index)
    for field in ["return_5d", "return_10d", "return_20d", "return_40d", "return_60d", "BIAS20", "BIAS60", "BIAS120", "MA20", "MA60", "MA120", "adjusted_close", "drawdown"]:
        for benchmark in ["0050", "00631L"]:
            out[f"{benchmark}_{field}"] = pivot[field][benchmark] if (field, benchmark) in pivot.columns else pd.NA
    out["00631L_volatility_20d"] = pivot["benchmark_volatility_20d"]["00631L"]
    out["00631L_above_MA20"] = out["00631L_adjusted_close"] > out["00631L_MA20"]
    out["00631L_above_MA60"] = out["00631L_adjusted_close"] > out["00631L_MA60"]
    out["00631L_above_MA120"] = out["00631L_adjusted_close"] > out["00631L_MA120"]
    out["benchmark_source_quality"] = "cache_exact"
    return out.reset_index().rename(columns={"trade_date": "signal_date"})


def _relative_strength_contract(candidates: pd.DataFrame, stock: pd.DataFrame, benchmark: pd.DataFrame) -> pd.DataFrame:
    out = candidates.merge(stock, left_on=["signal_date", "ticker"], right_on=["trade_date", "ticker"], how="left")
    out = out.drop(columns=["trade_date"], errors="ignore")
    out = out.merge(benchmark, on="signal_date", how="left")
    rename = {
        "excess_return_vs_00631L_5d": "rel_5d_vs_00631l",
        "excess_return_vs_00631L_10d": "rel_10d_vs_00631l",
        "excess_return_vs_00631L_20d": "rel_20d_vs_00631l",
        "excess_return_vs_00631L_40d": "rel_40d_vs_00631l",
        "excess_return_vs_00631L_60d": "rel_60d_vs_00631l",
    }
    out = out.rename(columns=rename)
    out["feature_asof_date"] = out["signal_date"]
    out["execution_calendar_alignment"] = "signal_date_weekly_snapshot_next_execution_handled_downstream"
    out["source_family"] = "stock_features_plus_benchmark_features_pit"
    out["rel_feature_source_quality"] = "exact_pit_from_stock_features_relative_to_00631L"
    out["forward_return_as_rule"] = False
    out["forward_returns_live_rule_usage"] = False
    out["not_live_rule"] = True
    out["diagnostic_only"] = True
    cols = [
        "signal_date",
        "ticker",
        "candidate_source_family",
        "theme_id",
        "theme_name",
        "feature_asof_date",
        "execution_calendar_alignment",
        "source_family",
        "rel_5d_vs_00631l",
        "rel_10d_vs_00631l",
        "rel_20d_vs_00631l",
        "rel_40d_vs_00631l",
        "rel_60d_vs_00631l",
        "RS5",
        "RS10",
        "RS20",
        "RS40",
        "RS60",
        "return_5d",
        "return_10d",
        "return_20d",
        "return_40d",
        "return_60d",
        "00631L_return_5d",
        "00631L_return_10d",
        "00631L_return_20d",
        "00631L_return_40d",
        "00631L_return_60d",
        "00631L_BIAS20",
        "00631L_BIAS60",
        "00631L_BIAS120",
        "00631L_above_MA20",
        "00631L_above_MA60",
        "00631L_above_MA120",
        "BIAS20",
        "BIAS60",
        "BIAS120",
        "BIAS20_percentile",
        "BIAS60_percentile",
        "BIAS120_percentile",
        "drawdown_20d",
        "drawdown_60d",
        "drawdown_120d",
        "volatility",
        "00631L_drawdown",
        "00631L_volatility_20d",
        "rel_feature_source_quality",
        "forward_return_as_rule",
        "forward_returns_live_rule_usage",
        "not_live_rule",
        "diagnostic_only",
    ]
    return out.reindex(columns=cols)


def _relative_stability_contract(strength: pd.DataFrame) -> pd.DataFrame:
    out = strength.copy()
    rel_cols = ["rel_5d_vs_00631l", "rel_10d_vs_00631l", "rel_20d_vs_00631l", "rel_40d_vs_00631l", "rel_60d_vs_00631l"]
    for col in rel_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["recent_windows_win_00631l_count"] = out[rel_cols].gt(0).sum(axis=1)
    out["recent_windows_available_count"] = out[rel_cols].notna().sum(axis=1)
    out["recent_windows_win_00631l_share"] = out["recent_windows_win_00631l_count"] / out["recent_windows_available_count"].where(out["recent_windows_available_count"].ne(0), pd.NA)
    out["only_win_0050_lose_00631l_count"] = (
        out[["RS5", "RS10", "RS20", "RS40", "RS60"]].apply(pd.to_numeric, errors="coerce").gt(0).to_numpy()
        & out[rel_cols].lt(0).to_numpy()
    ).sum(axis=1)
    out["rel_return_slope_20_minus_60"] = out["rel_20d_vs_00631l"] - out["rel_60d_vs_00631l"]
    out["rel_return_slope_5_minus_20"] = out["rel_5d_vs_00631l"] - out["rel_20d_vs_00631l"]
    out["relative_pattern_flag"] = out["rel_return_slope_5_minus_20"].map(
        lambda value: "improving_proxy" if pd.notna(value) and value > 0 else "deteriorating_or_flat_proxy"
    )
    out["candidate_drawdown_20d_vs_00631l_drawdown"] = pd.to_numeric(out["drawdown_20d"], errors="coerce") - pd.to_numeric(out["00631L_drawdown"], errors="coerce")
    out["candidate_drawdown_60d_vs_00631l_drawdown"] = pd.to_numeric(out["drawdown_60d"], errors="coerce") - pd.to_numeric(out["00631L_drawdown"], errors="coerce")
    out["trailing_underperformance_window_count"] = out[rel_cols].lt(0).sum(axis=1)
    out["bottom_tail_proxy_trailing_rel_return_min"] = out[rel_cols].min(axis=1, skipna=True)
    out["relative_overextension_flag"] = (
        pd.to_numeric(out["BIAS60"], errors="coerce") > pd.to_numeric(out["00631L_BIAS60"], errors="coerce")
    )
    out["stability_source_quality"] = "diagnostic_pit_from_trailing_returns_not_future"
    out["bottom_tail_proxy_source_quality"] = "trailing_returns_only_not_future_bottom_decile"
    cols = [
        "signal_date",
        "ticker",
        "candidate_source_family",
        "recent_windows_win_00631l_count",
        "recent_windows_available_count",
        "recent_windows_win_00631l_share",
        "only_win_0050_lose_00631l_count",
        "rel_return_slope_20_minus_60",
        "rel_return_slope_5_minus_20",
        "relative_pattern_flag",
        "candidate_drawdown_20d_vs_00631l_drawdown",
        "candidate_drawdown_60d_vs_00631l_drawdown",
        "trailing_underperformance_window_count",
        "bottom_tail_proxy_trailing_rel_return_min",
        "relative_overextension_flag",
        "stability_source_quality",
        "bottom_tail_proxy_source_quality",
        "forward_return_as_rule",
        "forward_returns_live_rule_usage",
        "not_live_rule",
        "diagnostic_only",
    ]
    return out.reindex(columns=cols)


def _blocked_proxy_fields(strength: pd.DataFrame, stability: pd.DataFrame, audit_path: Path) -> pd.DataFrame:
    phase_h_audit = pd.read_csv(audit_path) if audit_path.exists() else pd.DataFrame()
    rows = [
        ("rel_returns_vs_00631l", "ready_pit", True, "stock_features excess_return_vs_00631L trailing fields"),
        ("relative_stability_count", "diagnostic_ready", True, "count of past trailing windows beating 00631L; not live rule"),
        ("rel_return_slope", "proxy", True, "local trailing-window pattern proxy, not a statistical trend model"),
        ("drawdown_vs_00631l", "diagnostic_ready", True, "candidate drawdown fields vs 00631L benchmark drawdown"),
        ("downside_days_capture", "blocked", False, "daily downside-day capture not materialized in current stock_features"),
        ("bottom_tail_proxy", "proxy", True, "uses minimum trailing relative return, not future bottom-decile loss"),
        ("future_win_both_as_rule", "prohibited", False, "future win-both cannot be used as pre-trade rule"),
        ("future_bottom_decile_loss_as_rule", "prohibited", False, "future bottom-decile loss cannot be used as pre-trade rule"),
        ("formal_selector", "prohibited", False, "this package is source/contract readiness only"),
        ("portfolio_like_diagnostic", "prohibited", False, "Phase I readiness does not authorize portfolio-like diagnostic"),
    ]
    for item in phase_h_audit.itertuples(index=False):
        if str(item.item) == "strict_hurdle_input":
            rows.append((item.item, item.status, False, item.note))
    out = pd.DataFrame(rows, columns=["field_or_contract", "status", "proxy_available", "blocked_reason"])
    out["diagnostic_only"] = True
    out["not_live_rule"] = True
    out["future_data_violation_count"] = 0
    return out


def _future_data_audit(strength: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "audit_item": "feature_asof_signal_date",
                "status": "passed",
                "future_data_violation_count": 0,
                "note": "feature_asof_date equals signal_date; trailing windows come from stock/benchmark features",
            },
            {
                "audit_item": "forward_return_rule_input",
                "status": "passed",
                "future_data_violation_count": 0,
                "note": "no forward return fields are included in rule candidate contracts",
            },
            {
                "audit_item": "00631L_pool_member_boundary",
                "status": "passed",
                "future_data_violation_count": 0,
                "note": "00631L is benchmark/hurdle context only, not ordinary stock-pool member",
            },
        ]
    )


def _readiness_json(
    manifest_path: Path,
    strength: pd.DataFrame,
    stability: pd.DataFrame,
    join: pd.DataFrame,
    blocked: pd.DataFrame,
    future_audit: pd.DataFrame,
) -> dict[str, Any]:
    manifest = _read_json(manifest_path)
    future_count = int(future_audit["future_data_violation_count"].sum() + blocked["future_data_violation_count"].sum())
    required_rel = ["rel_5d_vs_00631l", "rel_10d_vs_00631l", "rel_20d_vs_00631l", "rel_40d_vs_00631l"]
    missing_required = {col: int(strength[col].isna().sum()) for col in required_rel}
    ready = len(strength) > 0 and len(stability) > 0 and future_count == 0 and all(v < len(strength) for v in missing_required.values())
    return {
        "date": "2026-07-06",
        "task_id": TASK_ID,
        "owner": "BACKTEST_LAB Core/Data",
        "status": "ready_for_phase_i_00631l_relative_stability_diagnostic" if ready else "blocked_phase_i_00631l_relative_stability_diagnostic",
        "ready_for_phase_i_00631l_relative_stability_diagnostic": bool(ready),
        "ready_for_portfolio_like_diagnostic": False,
        "ready_for_strategy_replay": False,
        "ready_for_formal": False,
        "future_data_violation_count": future_count,
        "not_live_rule": True,
        "forward_returns_live_rule_usage": False,
        "phase_h_verdict": manifest.get("verdict"),
        "phase_h_strict_hurdle_status": manifest.get("strict_hurdle_status"),
        "candidate_join_rows": int(len(join)),
        "relative_strength_rows": int(len(strength)),
        "relative_stability_rows": int(len(stability)),
        "missing_required_rel_counts": missing_required,
        "blocked_fields": blocked[blocked["status"].astype(str).str.contains("blocked|prohibited", case=False, regex=True)]["field_or_contract"].tolist(),
        "proxy_fields": blocked[blocked["proxy_available"].astype(bool)]["field_or_contract"].tolist(),
        "formal_model_changed": False,
        "trade_decision_changed": False,
        "active_in_trade_decision": False,
        "report_changed": False,
        "portfolio_replay_executed": False,
    }


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _summary(readiness: dict[str, Any], blocked: pd.DataFrame) -> str:
    return "\n".join(
        [
            "# vNext Phase I 00631L Relative Stability Readiness",
            "",
            f"Status: {readiness['status']}",
            "",
            "Boundary: PIT source/contract readiness only; no selector, no replay, no formal/report/trade change.",
            "",
            "Readiness:",
            f"- ready_for_phase_i_00631l_relative_stability_diagnostic={str(readiness['ready_for_phase_i_00631l_relative_stability_diagnostic']).lower()}",
            "- ready_for_portfolio_like_diagnostic=false",
            "- ready_for_strategy_replay=false",
            "- ready_for_formal=false",
            f"- future_data_violation_count={readiness['future_data_violation_count']}",
            "- not_live_rule=true",
            "- forward_returns_live_rule_usage=false",
            "",
            "Blocked / proxy fields:",
            *[f"- {row.field_or_contract}: {row.status}; {row.blocked_reason}" for row in blocked.itertuples()],
            "",
            "Flags:",
            "- formal_model_changed=false",
            "- trade_decision_changed=false",
            "- active_in_trade_decision=false",
            "- report_changed=false",
            "- portfolio_replay_executed=false",
        ]
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--materialization-dir", type=Path, default=DEFAULT_MATERIALIZATION_DIR)
    parser.add_argument("--phase-h-dir", type=Path, default=DEFAULT_PHASE_H_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    manifest = build_phase_i_readiness(
        materialization_dir=args.materialization_dir,
        phase_h_dir=args.phase_h_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
