"""Build stricter source/contract readiness for predictive bear/cash regime.

This is diagnostic/source-contract readiness only. It stages PIT market breadth,
shock-ledger schema, threshold policy staging, and regime transition features.
It does not define a live classifier, use forward returns as rules, or run any
portfolio/strategy replay.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


TASK_ID = "TASK-BACKTEST-CORE-VNEXT-STRICTER-BEAR-CASH-SOURCE-CONTRACT-READINESS-001"
DEFAULT_MATERIALIZATION_DIR = Path("outputs/vnext_dynamic_candidate_pool_data_materialization_20260706")
DEFAULT_PREDICTIVE_DIR = Path("outputs/vnext_predictive_bear_cash_regime_readiness_20260706")
DEFAULT_EXPERIMENTS_DIR = Path(
    r"C:\Users\zergv\Documents\Codex\2026-07-06\backtest-lab-experiments-diagnostic-validation-attribution\outputs"
    r"\vnext_predictive_bear_cash_classifier_quality_state_attribution_20260706"
)
DEFAULT_OUTPUT_DIR = Path("outputs/vnext_stricter_bear_cash_source_contract_readiness_20260706")


def build_stricter_source_contract_readiness(
    *,
    materialization_dir: str | Path = DEFAULT_MATERIALIZATION_DIR,
    predictive_dir: str | Path = DEFAULT_PREDICTIVE_DIR,
    experiments_dir: str | Path = DEFAULT_EXPERIMENTS_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    materialization = Path(materialization_dir)
    predictive = Path(predictive_dir)
    experiments = Path(experiments_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    feature_contract = pd.read_csv(predictive / "predictive_bear_cash_feature_contract.csv", parse_dates=["signal_date", "execution_date"])
    signal_dates = set(feature_contract["signal_date"].dt.strftime("%Y-%m-%d"))
    breadth = _market_breadth_contract(materialization / "stock_features.csv", signal_dates)
    shock_ledger = _external_shock_ledger_contract()
    threshold_policy = _threshold_policy_staging_contract()
    transitions = _regime_transition_feature_candidates(feature_contract)
    blocked = _blocked_proxy_readiness_ledger(breadth, shock_ledger)
    readiness = _readiness_json(
        feature_contract=feature_contract,
        breadth=breadth,
        shock_ledger=shock_ledger,
        threshold_policy=threshold_policy,
        transitions=transitions,
        blocked=blocked,
        experiments_manifest=experiments / "manifest.json",
    )

    _write_csv(breadth, output / "accepted_pit_market_breadth_contract.csv")
    _write_csv(shock_ledger, output / "external_major_event_shock_ledger_contract.csv")
    _write_csv(threshold_policy, output / "accepted_threshold_policy_staging.csv")
    _write_csv(transitions, output / "regime_transition_feature_candidates.csv")
    _write_csv(blocked, output / "blocked_proxy_readiness_ledger.csv")
    (output / "readiness_for_stricter_bear_cash_classifier_diagnostic.json").write_text(
        json.dumps(readiness, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    manifest = {
        "task_id": TASK_ID,
        "status": readiness["status"],
        "output_dir": str(output.resolve()),
        "input_materialization_dir": str(materialization.resolve()),
        "input_predictive_contract_dir": str(predictive.resolve()),
        "input_experiments_dir": str(experiments.resolve()),
        "output_files": [
            "accepted_pit_market_breadth_contract.csv",
            "external_major_event_shock_ledger_contract.csv",
            "accepted_threshold_policy_staging.csv",
            "regime_transition_feature_candidates.csv",
            "blocked_proxy_readiness_ledger.csv",
            "readiness_for_stricter_bear_cash_classifier_diagnostic.json",
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
        "diagnostic_only": True,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "final_summary_zh.md").write_text(_summary(readiness, blocked), encoding="utf-8")
    return manifest


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _market_breadth_contract(stock_features_path: Path, signal_dates: set[str]) -> pd.DataFrame:
    usecols = [
        "trade_date",
        "ticker",
        "adjusted_close",
        "return_5d",
        "MA20",
        "MA60",
        "MA120",
        "MA20_position",
        "MA60_position",
        "MA120_position",
    ]
    rows: list[dict[str, Any]] = []
    for chunk in pd.read_csv(stock_features_path, usecols=usecols, chunksize=500_000):
        chunk = chunk[chunk["trade_date"].astype(str).isin(signal_dates)].copy()
        if chunk.empty:
            continue
        chunk = chunk[pd.to_numeric(chunk["adjusted_close"], errors="coerce").notna()]
        if chunk.empty:
            continue
        for col in ["return_5d", "MA20", "MA60", "MA120", "MA20_position", "MA60_position", "MA120_position"]:
            chunk[col] = pd.to_numeric(chunk[col], errors="coerce")
        chunk["above_MA20"] = chunk["MA20_position"].gt(0) | chunk["adjusted_close"].gt(chunk["MA20"])
        chunk["above_MA60"] = chunk["MA60_position"].gt(0) | chunk["adjusted_close"].gt(chunk["MA60"])
        chunk["above_MA120"] = chunk["MA120_position"].gt(0) | chunk["adjusted_close"].gt(chunk["MA120"])
        chunk["advancing_5d"] = chunk["return_5d"].gt(0)
        chunk["declining_5d"] = chunk["return_5d"].lt(0)
        grouped = chunk.groupby("trade_date", sort=False)
        for trade_date, group in grouped:
            universe_count = int(len(group))
            rows.append(
                {
                    "signal_date": trade_date,
                    "universe_definition": "stock_features rows with adjusted_close available on signal_date",
                    "universe_count": universe_count,
                    "above_MA20_count": int(group["above_MA20"].sum()),
                    "above_MA60_count": int(group["above_MA60"].sum()),
                    "above_MA120_count": int(group["above_MA120"].sum()),
                    "above_MA20_breadth": float(group["above_MA20"].mean()) if universe_count else pd.NA,
                    "above_MA60_breadth": float(group["above_MA60"].mean()) if universe_count else pd.NA,
                    "above_MA120_breadth": float(group["above_MA120"].mean()) if universe_count else pd.NA,
                    "advancing_5d_count": int(group["advancing_5d"].sum()),
                    "declining_5d_count": int(group["declining_5d"].sum()),
                    "advancing_declining_proxy": (
                        int(group["advancing_5d"].sum()) - int(group["declining_5d"].sum())
                    )
                    / universe_count
                    if universe_count
                    else pd.NA,
                    "source_quality": "diagnostic_pit_from_stock_features",
                    "future_data_violation_count": 0,
                    "diagnostic_only": True,
                    "accepted_for_formal": False,
                    "not_live_rule": True,
                }
            )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).drop_duplicates("signal_date").sort_values("signal_date")


def _external_shock_ledger_contract() -> pd.DataFrame:
    columns = [
        "event_date",
        "event_type",
        "known_asof_date",
        "severity_candidate",
        "source",
        "source_quality",
        "future_data_violation_count",
        "blocked_reason",
        "diagnostic_only",
        "accepted_for_formal",
        "not_live_rule",
    ]
    return pd.DataFrame(
        [
            {
                "event_date": "",
                "event_type": "schema_only_no_events_materialized",
                "known_asof_date": "",
                "severity_candidate": "",
                "source": "",
                "source_quality": "blocked_missing_external_event_ledger",
                "future_data_violation_count": 0,
                "blocked_reason": "external major event / shock ledger source is not materialized",
                "diagnostic_only": True,
                "accepted_for_formal": False,
                "not_live_rule": True,
            }
        ],
        columns=columns,
    )


def _threshold_policy_staging_contract() -> pd.DataFrame:
    rows = [
        ("0050_BIAS60", "negative_extreme_grid", "predeclared_grid_or_research_prior_only"),
        ("0050_drawdown_from_60d_high", "drawdown_guard_grid", "predeclared_grid_or_research_prior_only"),
        ("0050_realized_volatility_20d", "volatility_regime_percentile_grid", "rolling_past_distribution_only"),
        ("above_MA60_breadth", "breadth_collapse_grid", "accepted_pit_breadth_only"),
        ("advancing_declining_proxy", "adv_decline_proxy_grid", "accepted_pit_breadth_only"),
        ("00631L_vs_0050_return_20d", "levered_stress_grid", "same-date_pit_relative_return_only"),
        ("market_top20_traded_value_share", "turnover_concentration_grid", "diagnostic_aggregate_only"),
    ]
    return pd.DataFrame(
        [
            {
                "threshold_field": field,
                "threshold_family": family,
                "construction_basis": basis,
                "thresholds_are_candidate_falsifiable": True,
                "forward_return_derived_threshold": False,
                "per_period_evaluation_metadata_only": True,
                "live_rule": False,
                "not_live_rule": True,
                "diagnostic_only": True,
            }
            for field, family, basis in rows
        ]
    )


def _regime_transition_feature_candidates(features: pd.DataFrame) -> pd.DataFrame:
    data = features.sort_values("signal_date").copy()
    data["0050_trend_transition_MA20"] = _transition(data["0050_above_MA20"])
    data["0050_trend_transition_MA60"] = _transition(data["0050_above_MA60"])
    data["0050_trend_transition_MA120"] = _transition(data["0050_above_MA120"])
    data["volatility_20d_delta_4w"] = data["0050_realized_volatility_20d"] - data["0050_realized_volatility_20d"].shift(4)
    data["volatility_regime_transition_candidate"] = data["volatility_20d_delta_4w"].map(
        lambda value: "volatility_upshift_candidate" if pd.notna(value) and value > 0 else "none"
    )
    data["drawdown_60d_acceleration_4w"] = data["0050_drawdown_from_60d_high"] - data["0050_drawdown_from_60d_high"].shift(4)
    data["drawdown_acceleration_candidate"] = data["drawdown_60d_acceleration_4w"].map(
        lambda value: "drawdown_acceleration_candidate" if pd.notna(value) and value < -0.03 else "none"
    )
    data["00631L_stress_vs_0050_candidate"] = data["00631L_vs_0050_return_20d"].map(
        lambda value: "levered_etf_relative_stress_candidate" if pd.notna(value) and value < 0 else "none"
    )
    data["turnover_concentration_transition_candidate"] = (
        data["market_top20_traded_value_share"] - data["market_top20_traded_value_share"].shift(4)
    ).map(lambda value: "concentration_upshift_candidate" if pd.notna(value) and value > 0.05 else "none")
    cols = [
        "signal_date",
        "execution_date",
        "0050_trend_transition_MA20",
        "0050_trend_transition_MA60",
        "0050_trend_transition_MA120",
        "volatility_20d_delta_4w",
        "volatility_regime_transition_candidate",
        "drawdown_60d_acceleration_4w",
        "drawdown_acceleration_candidate",
        "00631L_vs_0050_return_20d",
        "00631L_stress_vs_0050_candidate",
        "market_top20_traded_value_share",
        "turnover_concentration_transition_candidate",
    ]
    out = data[cols].copy()
    out["source_quality"] = "diagnostic_pit_transition_from_existing_feature_contract"
    out["future_data_violation_count"] = 0
    out["diagnostic_only"] = True
    out["not_live_rule"] = True
    return out


def _transition(series: pd.Series) -> pd.Series:
    values = series.astype("boolean")
    prev = values.shift(1)
    transitions = []
    for cur, prv in zip(values, prev):
        if pd.isna(cur) or pd.isna(prv):
            transitions.append("unknown")
        elif bool(cur) and not bool(prv):
            transitions.append("cross_above")
        elif (not bool(cur)) and bool(prv):
            transitions.append("cross_below")
        else:
            transitions.append("no_transition")
    return pd.Series(transitions, index=series.index)


def _blocked_proxy_readiness_ledger(breadth: pd.DataFrame, shock_ledger: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {
            "field_or_contract": "accepted_pit_market_breadth",
            "status": "diagnostic_ready" if not breadth.empty else "blocked",
            "source_quality": "diagnostic_pit_from_stock_features" if not breadth.empty else "missing",
            "proxy_available": not breadth.empty,
            "blocked_reason": "usable for stricter diagnostic, not accepted live classifier",
        },
        {
            "field_or_contract": "external_major_event_shock_ledger",
            "status": "blocked",
            "source_quality": str(shock_ledger["source_quality"].iloc[0]),
            "proxy_available": False,
            "blocked_reason": "external event source not materialized",
        },
        {
            "field_or_contract": "accepted_threshold_policy",
            "status": "staged_candidate_only",
            "source_quality": "policy_staging",
            "proxy_available": True,
            "blocked_reason": "thresholds are falsifiable candidates; no live policy accepted",
        },
        {
            "field_or_contract": "forward_return_as_rule",
            "status": "prohibited",
            "source_quality": "prohibited",
            "proxy_available": False,
            "blocked_reason": "forward returns cannot be used in rule construction",
        },
        {
            "field_or_contract": "live_bear_cash_classifier",
            "status": "blocked",
            "source_quality": "missing",
            "proxy_available": False,
            "blocked_reason": "no live classifier is defined or approved",
        },
        {
            "field_or_contract": "portfolio_like_diagnostic",
            "status": "blocked_by_default",
            "source_quality": "policy_blocked",
            "proxy_available": False,
            "blocked_reason": "Research stopped current route; only classifier-quality diagnostic may be considered",
        },
    ]
    out = pd.DataFrame(rows)
    out["future_data_violation_count"] = 0
    out["diagnostic_only"] = True
    out["not_live_rule"] = True
    return out


def _readiness_json(
    *,
    feature_contract: pd.DataFrame,
    breadth: pd.DataFrame,
    shock_ledger: pd.DataFrame,
    threshold_policy: pd.DataFrame,
    transitions: pd.DataFrame,
    blocked: pd.DataFrame,
    experiments_manifest: Path,
) -> dict[str, Any]:
    manifest = _read_json(experiments_manifest)
    future_violations = int(blocked["future_data_violation_count"].sum())
    ready = not breadth.empty and not threshold_policy.empty and not transitions.empty and future_violations == 0
    return {
        "date": "2026-07-06",
        "task_id": TASK_ID,
        "owner": "BACKTEST_LAB Core/Data",
        "status": "ready_for_stricter_bear_cash_classifier_quality_diagnostic" if ready else "blocked_for_stricter_bear_cash_classifier_quality_diagnostic",
        "ready_for_stricter_bear_cash_classifier_quality_diagnostic": bool(ready),
        "ready_for_portfolio_like_diagnostic": False,
        "ready_for_strategy_replay": False,
        "ready_for_formal": False,
        "future_data_violation_count": future_violations,
        "not_live_rule": True,
        "diagnostic_only": True,
        "prior_experiments_verdict": manifest.get("verdict"),
        "feature_rows": int(len(feature_contract)),
        "market_breadth_rows": int(len(breadth)),
        "shock_ledger_rows": int(len(shock_ledger)),
        "threshold_policy_rows": int(len(threshold_policy)),
        "transition_feature_rows": int(len(transitions)),
        "blocked_fields": blocked[blocked["status"].astype(str).str.contains("blocked", case=False, na=False)][
            "field_or_contract"
        ].tolist(),
        "proxy_fields": blocked[blocked["proxy_available"].astype(bool)]["field_or_contract"].tolist(),
        "prohibited_fields": blocked[blocked["status"].eq("prohibited")]["field_or_contract"].tolist(),
        "formal_model_changed": False,
        "trade_decision_changed": False,
        "active_in_trade_decision": False,
        "report_changed": False,
        "portfolio_replay_executed": False,
    }


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def _summary(readiness: dict[str, Any], blocked: pd.DataFrame) -> str:
    return "\n".join(
        [
            "# vNext Stricter Bear/Cash Source Contract Readiness",
            "",
            f"Status: {readiness['status']}",
            "",
            "Boundary: source/contract readiness only; no live classifier, no replay, no formal change.",
            "",
            "Readiness:",
            f"- ready_for_stricter_bear_cash_classifier_quality_diagnostic={str(readiness['ready_for_stricter_bear_cash_classifier_quality_diagnostic']).lower()}",
            "- ready_for_portfolio_like_diagnostic=false",
            "- ready_for_strategy_replay=false",
            "- ready_for_formal=false",
            f"- future_data_violation_count={readiness['future_data_violation_count']}",
            "- not_live_rule=true",
            "",
            "Blocked / proxy ledger:",
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
    parser.add_argument("--predictive-dir", type=Path, default=DEFAULT_PREDICTIVE_DIR)
    parser.add_argument("--experiments-dir", type=Path, default=DEFAULT_EXPERIMENTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    manifest = build_stricter_source_contract_readiness(
        materialization_dir=args.materialization_dir,
        predictive_dir=args.predictive_dir,
        experiments_dir=args.experiments_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
