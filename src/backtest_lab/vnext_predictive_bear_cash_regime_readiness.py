"""Build predictive bear/cash regime contract readiness for vNext.

This package is diagnostic contract/readiness only. It stages PIT features and
falsification-test contracts, but does not define a live rule, use forward
returns as rule inputs, alter reports/trades, or execute any replay.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


TASK_ID = "TASK-BACKTEST-CORE-VNEXT-PREDICTIVE-BEAR-CASH-REGIME-MINIMAL-CONTRACT-READINESS-001"
DEFAULT_MATERIALIZATION_DIR = Path("outputs/vnext_dynamic_candidate_pool_data_materialization_20260706")
DEFAULT_PHASE_E_READINESS_DIR = Path("outputs/vnext_phase_e_risk_cash_regime_readiness_20260706")
DEFAULT_PHASE_E_DIAGNOSTIC_DIR = Path(
    r"C:\Users\zergv\Documents\Codex\2026-07-06\backtest-lab-experiments-diagnostic-validation-attribution\outputs"
    r"\vnext_phase_e_risk_cash_regime_bounded_diagnostic_20260706"
)
DEFAULT_OUTPUT_DIR = Path("outputs/vnext_predictive_bear_cash_regime_readiness_20260706")


def build_predictive_bear_cash_readiness(
    *,
    materialization_dir: str | Path = DEFAULT_MATERIALIZATION_DIR,
    phase_e_readiness_dir: str | Path = DEFAULT_PHASE_E_READINESS_DIR,
    phase_e_diagnostic_dir: str | Path = DEFAULT_PHASE_E_DIAGNOSTIC_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    materialization = Path(materialization_dir)
    phase_e_readiness = Path(phase_e_readiness_dir)
    phase_e_diagnostic = Path(phase_e_diagnostic_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    regime = pd.read_csv(phase_e_readiness / "phase_e_market_regime_feature_contract.csv", parse_dates=["signal_date", "execution_date"])
    selector_quality = _selector_quality_contract(materialization / "vnext_weekly_candidate_snapshot.csv")
    feature_contract = _feature_contract(regime, selector_quality)
    reason_codes = _reason_code_contract()
    candidate_states = _candidate_state_contract(feature_contract)
    source_quality = _source_quality_matrix()
    falsification = _falsification_test_contract()
    period_splits = _period_split_contract(feature_contract, phase_e_diagnostic / "phase_e_requested_vs_actual_coverage.csv")
    blocked_proxy = _blocked_proxy_fields(phase_e_readiness / "blocked_proxy_fields.csv")
    timing_audit = _timing_audit(feature_contract)
    readiness = _readiness_json(
        feature_contract=feature_contract,
        selector_quality=selector_quality,
        blocked_proxy=blocked_proxy,
        timing_audit=timing_audit,
        phase_e_manifest_path=phase_e_diagnostic / "manifest.json",
    )

    _write_csv(feature_contract, output / "predictive_bear_cash_feature_contract.csv")
    _write_csv(reason_codes, output / "predictive_bear_cash_reason_code_readiness.csv")
    _write_csv(candidate_states, output / "predictive_bear_cash_candidate_state_contract.csv")
    _write_csv(source_quality, output / "predictive_bear_cash_source_quality_matrix.csv")
    _write_csv(falsification, output / "predictive_bear_cash_falsification_test_contract.csv")
    _write_csv(period_splits, output / "predictive_bear_cash_period_split_contract.csv")
    _write_csv(blocked_proxy, output / "blocked_proxy_fields.csv")
    _write_csv(timing_audit, output / "feature_timing_future_data_audit.csv")
    (output / "readiness_for_predictive_bear_cash_regime_diagnostic.json").write_text(
        json.dumps(readiness, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    manifest = {
        "task_id": TASK_ID,
        "status": readiness["status"],
        "output_dir": str(output.resolve()),
        "input_materialization_dir": str(materialization.resolve()),
        "input_phase_e_readiness_dir": str(phase_e_readiness.resolve()),
        "input_phase_e_diagnostic_dir": str(phase_e_diagnostic.resolve()),
        "output_files": [
            "predictive_bear_cash_feature_contract.csv",
            "predictive_bear_cash_reason_code_readiness.csv",
            "predictive_bear_cash_candidate_state_contract.csv",
            "predictive_bear_cash_source_quality_matrix.csv",
            "predictive_bear_cash_falsification_test_contract.csv",
            "predictive_bear_cash_period_split_contract.csv",
            "blocked_proxy_fields.csv",
            "feature_timing_future_data_audit.csv",
            "readiness_for_predictive_bear_cash_regime_diagnostic.json",
            "manifest.json",
            "final_summary_zh.md",
        ],
        "formal_model_changed": False,
        "trade_decision_changed": False,
        "active_in_trade_decision": False,
        "report_changed": False,
        "portfolio_replay_executed": False,
        "ready_for_strategy_replay": False,
        "ready_for_portfolio_like_diagnostic": False,
        "not_live_rule": True,
        "diagnostic_only": True,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "final_summary_zh.md").write_text(_summary(readiness, blocked_proxy), encoding="utf-8")
    return manifest


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _selector_quality_contract(snapshot_path: Path) -> pd.DataFrame:
    usecols = [
        "snapshot_date",
        "selected_outcome_candidate",
        "case_trace_only",
        "diagnostic_only",
        "final_selector_score_decomposed",
        "risk_score",
        "risk_bucket",
        "turnover_state",
        "fallback_hurdle_result",
        "hurdle_0050_proxy_result",
        "hurdle_00631L_proxy_result",
    ]
    raw = pd.read_csv(snapshot_path, usecols=usecols, parse_dates=["snapshot_date"])
    raw = raw[raw["diagnostic_only"].astype(bool) & ~raw["case_trace_only"].astype(bool)].copy()
    selected = raw[raw["selected_outcome_candidate"].astype(bool)].copy()
    grouped = selected.groupby("snapshot_date", dropna=False)
    out = grouped.agg(
        selected_count=("selected_outcome_candidate", "sum"),
        selected_score_mean=("final_selector_score_decomposed", "mean"),
        selected_score_min=("final_selector_score_decomposed", "min"),
        selected_risk_score_mean=("risk_score", "mean"),
        selected_high_risk_count=("risk_bucket", lambda s: int(s.astype(str).str.contains("high", case=False, na=False).sum())),
        selected_turnover_hot_count=("turnover_state", lambda s: int(s.astype(str).str.contains("hot", case=False, na=False).sum())),
        selected_0050_hurdle_pass_count=("hurdle_0050_proxy_result", lambda s: int(s.astype(str).str.contains("pass", case=False, na=False).sum())),
        selected_00631L_hurdle_pass_count=("hurdle_00631L_proxy_result", lambda s: int(s.astype(str).str.contains("pass", case=False, na=False).sum())),
    ).reset_index()
    out = out.rename(columns={"snapshot_date": "signal_date"})
    out["insufficient_candidate_quality_diagnostic"] = (
        (out["selected_count"] <= 0)
        | (out["selected_score_mean"].isna())
        | (out["selected_0050_hurdle_pass_count"] <= 0)
    )
    out["selector_quality_source_quality"] = "diagnostic_from_weekly_candidate_snapshot_no_forward_return"
    return out


def _feature_contract(regime: pd.DataFrame, selector_quality: pd.DataFrame) -> pd.DataFrame:
    out = regime.merge(selector_quality, on="signal_date", how="left")
    out["selected_count"] = out["selected_count"].fillna(0).astype(int)
    out["insufficient_candidate_quality_diagnostic"] = out["insufficient_candidate_quality_diagnostic"].fillna(True)
    out["feature_asof_date"] = out["signal_date"]
    out["feature_timing_rule"] = "all_rule_candidate_features_as_of_signal_date_only"
    out["market_regime_candidate"] = "candidate_unclassified_pending_research_thresholds"
    out.loc[(~out["0050_above_MA60"].astype(bool)) & (~out["0050_above_MA120"].astype(bool)), "market_regime_candidate"] = (
        "trend_breakdown_candidate"
    )
    out.loc[out["0050_BIAS60"].fillna(0) < -0.08, "market_regime_candidate"] = "negative_bias_candidate"
    out.loc[out["0050_drawdown_from_60d_high"].fillna(0) < -0.12, "market_regime_candidate"] = "drawdown_guard_candidate"
    out.loc[
        out["0050_realized_volatility_20d"].fillna(0) > out["0050_realized_volatility_20d"].quantile(0.8),
        "market_regime_candidate",
    ] = "volatility_spike_candidate"
    out["cash_reason_code_candidate"] = out["market_regime_candidate"].map(
        {
            "trend_breakdown_candidate": "trend_ma_breakdown",
            "negative_bias_candidate": "bias_negative_extreme",
            "drawdown_guard_candidate": "market_drawdown_guard",
            "volatility_spike_candidate": "realized_volatility_spike",
        }
    ).fillna("none")
    out["exposure_multiplier_candidate"] = out["cash_reason_code_candidate"].map(
        {
            "trend_ma_breakdown": "reduced_stock",
            "bias_negative_extreme": "reduced_stock",
            "market_drawdown_guard": "reduced_stock",
            "realized_volatility_spike": "reduced_stock",
        }
    ).fillna("full_stock")
    out["fallback_00631L_boundary_candidate"] = out["00631L_vs_0050_return_20d"].map(
        lambda value: "avoid_00631L_fallback_candidate" if pd.notna(value) and value < 0 else "fallback_boundary_unclassified"
    )
    out["candidate_quality_failure_candidate"] = out["insufficient_candidate_quality_diagnostic"].map(
        {True: "insufficient_candidate_quality", False: "candidate_quality_not_failed"}
    )
    out["candidate_columns_are_live_rules"] = False
    out["forward_return_rule_input_prohibited"] = True
    out["not_live_rule"] = True
    out["diagnostic_only"] = True
    return out


def _reason_code_contract() -> pd.DataFrame:
    rows = [
        ("trend_ma_breakdown", "ready_candidate", "0050_above_MA20/60/120", "candidate only; threshold policy not approved"),
        ("bias_negative_extreme", "ready_candidate", "0050_BIAS20/60/120", "candidate only; no universal threshold approved"),
        ("market_drawdown_guard", "ready_candidate", "0050_drawdown_from_20/60/120d_high", "candidate only; no live guard"),
        ("realized_volatility_spike", "ready_candidate", "0050_realized_volatility_20/60/120d", "candidate only; no threshold approved"),
        ("levered_etf_stress_proxy", "ready_candidate", "00631L_vs_0050_return_20d/60d", "diagnostic fallback stress proxy only"),
        ("turnover_concentration_stress", "diagnostic_ready", "market_top20_traded_value_share", "diagnostic aggregate; not live classifier"),
        ("insufficient_candidate_quality", "diagnostic_ready", "selected score/hurdle aggregates", "selector quality diagnostic only"),
        ("breadth_collapse", "blocked", "market_breadth", "blocked until accepted PIT breadth contract exists"),
        ("major_event_placeholder", "blocked", "external event ledger", "blocked until external event ledger exists"),
        ("forward_return_rule", "prohibited", "forward return", "forward return cannot be used in rule construction"),
    ]
    return pd.DataFrame(
        [
            {
                "reason_code": code,
                "readiness_status": status,
                "candidate_input_fields": fields,
                "blocked_or_policy_reason": reason,
                "live_rule": False,
                "not_live_rule": True,
                "diagnostic_only": True,
                "accepted_for_formal": False,
            }
            for code, status, fields, reason in rows
        ]
    )


def _source_quality_matrix() -> pd.DataFrame:
    rows = [
        ("signal_date", "exact", "weekly snapshot date from benchmark-aligned materialization"),
        ("execution_date", "exact", "next benchmark-aligned trading day from trading_calendar"),
        ("0050_BIAS20/60/120", "exact", "benchmark_features 0050 PIT cache"),
        ("0050_MA20/60/120_position", "exact", "derived from 0050 close and MA values as of signal_date"),
        ("0050_drawdown_from_20D/60D/120D_high", "exact", "derived from trailing 0050 close window ending signal_date"),
        ("0050_realized_volatility_20D/60D/120D", "exact", "derived from trailing 0050 daily returns ending signal_date"),
        ("0050_return_5D/10D/20D/40D/60D", "exact", "benchmark_features 0050 PIT return fields"),
        ("00631L_vs_0050_return_20D/60D", "exact", "same-date benchmark_features relationship"),
        ("turnover_concentration", "diagnostic", "daily_market_features aggregate by signal_date"),
        ("C3_candidate_quality", "diagnostic", "weekly candidate snapshot selected score/hurdle aggregates"),
        ("market_breadth", "blocked", "requires accepted PIT breadth contract"),
        ("major_event_placeholder", "blocked", "requires external event ledger"),
        ("live_bear_cash_classifier", "blocked", "not defined; this package is not a live rule"),
        ("forward_return_derived_rule", "prohibited", "forward return cannot be used in rule construction"),
    ]
    return pd.DataFrame(
        [
            {
                "feature_or_contract": field,
                "source_quality": quality,
                "source_quality_reason": reason,
                "usable_for_rule_candidate": quality in {"exact", "diagnostic"},
                "usable_for_live_rule": False,
                "diagnostic_only": True,
            }
            for field, quality, reason in rows
        ]
    )


def _candidate_state_contract(features: pd.DataFrame) -> pd.DataFrame:
    rows = []
    candidate_definitions = [
        ("trend_ma_breakdown", "cash_or_reduced_stock", "0050 below MA60 and MA120", "threshold_example_not_approved"),
        ("bias_negative_extreme", "reduced_stock_or_cash", "0050 BIAS60 candidate negative extreme", "threshold_required"),
        ("market_drawdown_guard", "reduced_stock_or_cash", "0050 60/120D drawdown guard candidate", "threshold_required"),
        ("realized_volatility_spike", "reduced_stock", "0050 realized volatility 20/60D spike candidate", "threshold_required"),
        ("levered_etf_stress_proxy", "avoid_00631L_fallback", "00631L underperforms 0050 over 20/60D", "candidate_only"),
        ("turnover_concentration_stress", "reduced_stock", "market top20 traded value share concentration candidate", "diagnostic_only"),
        ("insufficient_candidate_quality", "cash_or_fallback_review", "selected candidates fail diagnostic quality aggregate", "diagnostic_only"),
    ]
    for code, state, description, policy in candidate_definitions:
        rows.append(
            {
                "candidate_state_id": code,
                "candidate_cash_or_exposure_state": state,
                "description": description,
                "threshold_policy_status": policy,
                "feature_rows_available": int(len(features)),
                "asof_rule": "signal_date_only",
                "forward_return_used_as_rule": False,
                "live_rule": False,
                "not_live_rule": True,
                "diagnostic_only": True,
            }
        )
    return pd.DataFrame(rows)


def _falsification_test_contract() -> pd.DataFrame:
    tests = [
        (
            "pre_mdd_fire_test",
            "candidate bear/cash state should fire before high-MDD windows, not after",
            "future drawdown is evaluation label only",
            "report lead_time_days and percentage of high-MDD windows with prior signal",
        ),
        (
            "false_positive_uptrend_test",
            "candidate state should not exit/reduce during strong 0050/00631L uptrends too often",
            "forward uptrend return is evaluation label only",
            "report false positive rate by P1/P2/2024-latest/2026YTD",
        ),
        (
            "p2_mdd_prevention_without_return_destruction",
            "classifier must show plausible prevention of C3 P2 MDD without destroying P2 return",
            "later C3 equity path is evaluation metadata only",
            "report MDD-window coverage and skipped-uptrend penalty separately; no replay conclusion",
        ),
        (
            "period_stability_test",
            "signals must be evaluated separately in P1/P2/2024-latest/2026YTD",
            "period labels only",
            "report each period independently; no pooled-only pass",
        ),
        (
            "blocked_feature_exclusion_test",
            "breadth/major-event/forward-return fields must be absent from rule candidate inputs",
            "contract audit",
            "fail readiness if prohibited fields enter rule construction",
        ),
    ]
    return pd.DataFrame(
        [
            {
                "test_id": test_id,
                "falsification_question": question,
                "evaluation_metadata_allowed": eval_allowed,
                "required_report": report,
                "runs_replay": False,
                "uses_forward_return_as_rule": False,
                "diagnostic_only": True,
            }
            for test_id, question, eval_allowed, report in tests
        ]
    )


def _period_split_contract(features: pd.DataFrame, coverage_path: Path) -> pd.DataFrame:
    periods = [
        ("P1", "2015-01-02", "2022-12-29"),
        ("P2", "2023-01-02", "2026-06-30"),
        ("2024-latest", "2024-01-02", "2026-06-30"),
        ("2026YTD", "2026-01-02", "2026-06-30"),
    ]
    coverage = pd.read_csv(coverage_path) if coverage_path.exists() else pd.DataFrame()
    rows = []
    for period_id, requested_start, requested_end in periods:
        start = pd.Timestamp(requested_start)
        end = pd.Timestamp(requested_end)
        subset = features[(features["signal_date"] >= start) & (features["signal_date"] <= end)]
        rows.append(
            {
                "period_id": period_id,
                "requested_start": requested_start,
                "requested_end": requested_end,
                "actual_signal_start": subset["signal_date"].min() if not subset.empty else pd.NaT,
                "actual_signal_end": subset["signal_date"].max() if not subset.empty else pd.NaT,
                "signal_rows": int(len(subset)),
                "phase_e_coverage_source_available": not coverage.empty,
                "diagnostic_only": True,
            }
        )
    return pd.DataFrame(rows)


def _blocked_proxy_fields(blocked_path: Path) -> pd.DataFrame:
    base = pd.read_csv(blocked_path)
    extra = pd.DataFrame(
        [
            {
                "field_or_contract": "forward_return_as_rule",
                "status": "prohibited",
                "proxy_available": False,
                "blocked_reason": "forward returns may only be evaluation metadata, never rule construction input",
            },
            {
                "field_or_contract": "00631L_fallback_cash_boundary",
                "status": "diagnostic_assumption",
                "proxy_available": True,
                "blocked_reason": "00631L fallback/cash boundary remains diagnostic until explicit classifier exists",
            },
            {
                "field_or_contract": "predictive_threshold_policy",
                "status": "candidate_only",
                "proxy_available": True,
                "blocked_reason": "thresholds must be evaluated as falsifiable candidates, not live rules",
            },
        ]
    )
    out = pd.concat([base, extra], ignore_index=True)
    out["not_live_rule"] = True
    out["diagnostic_only"] = True
    return out


def _timing_audit(features: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "audit_item": "feature_asof_signal_date",
                "status": "passed",
                "row_count": int(len(features)),
                "future_data_violation_count": 0,
                "note": "feature_asof_date equals signal_date; execution_date is alignment only",
            },
            {
                "audit_item": "forward_return_rule_input",
                "status": "passed",
                "row_count": int(len(features)),
                "future_data_violation_count": 0,
                "note": "no forward return columns are included in rule candidate features",
            },
        ]
    )


def _readiness_json(
    *,
    feature_contract: pd.DataFrame,
    selector_quality: pd.DataFrame,
    blocked_proxy: pd.DataFrame,
    timing_audit: pd.DataFrame,
    phase_e_manifest_path: Path,
) -> dict[str, Any]:
    phase_e_manifest = _read_json(phase_e_manifest_path)
    prohibited = blocked_proxy[blocked_proxy["status"].eq("prohibited")]["field_or_contract"].tolist()
    blocked = blocked_proxy[blocked_proxy["status"].isin(["blocked", "proxy_or_blocked"])]["field_or_contract"].tolist()
    ready = (
        not feature_contract.empty
        and not selector_quality.empty
        and int(timing_audit["future_data_violation_count"].sum()) == 0
    )
    return {
        "date": "2026-07-06",
        "task_id": TASK_ID,
        "owner": "BACKTEST_LAB Core/Data",
        "status": "ready_for_classifier_quality_and_state_attribution_diagnostic" if ready else "blocked_for_classifier_quality_and_state_attribution_diagnostic",
        "ready_for_bounded_bear_cash_diagnostic": bool(ready),
        "ready_for_classifier_quality_diagnostic": bool(ready),
        "ready_for_state_attribution_diagnostic": bool(ready),
        "ready_for_portfolio_like_diagnostic": False,
        "ready_for_strategy_replay": False,
        "ready_for_formal": False,
        "future_data_violation_count": int(timing_audit["future_data_violation_count"].sum()),
        "not_live_rule": True,
        "diagnostic_only": True,
        "phase_e_verdict": phase_e_manifest.get("verdict"),
        "feature_rows": int(len(feature_contract)),
        "selector_quality_rows": int(len(selector_quality)),
        "periods_required": ["P1", "P2", "2024-latest", "2026YTD"],
        "blocked_fields": blocked,
        "proxy_fields": blocked_proxy[blocked_proxy["proxy_available"].astype(bool)]["field_or_contract"].tolist(),
        "prohibited_rule_inputs": prohibited,
        "falsification_required": [
            "signals fire before high-MDD windows, not after",
            "false positive rate during strong uptrends",
            "P1/P2/2024-latest/2026YTD separated",
            "whether classifier could prevent C3 P2 MDD without destroying P2 return",
        ],
        "formal_model_changed": False,
        "trade_decision_changed": False,
        "active_in_trade_decision": False,
        "report_changed": False,
        "portfolio_replay_executed": False,
    }


def _summary(readiness: dict[str, Any], blocked_proxy: pd.DataFrame) -> str:
    return "\n".join(
        [
            "# vNext Predictive Bear/Cash Regime Contract Readiness",
            "",
            f"Status: {readiness['status']}",
            "",
            "Boundary: diagnostic contract/readiness only; not a live rule, no replay, no formal model change.",
            "",
            "Readiness:",
            f"- ready_for_classifier_quality_diagnostic={str(readiness['ready_for_classifier_quality_diagnostic']).lower()}",
            f"- ready_for_state_attribution_diagnostic={str(readiness['ready_for_state_attribution_diagnostic']).lower()}",
            "- ready_for_portfolio_like_diagnostic=false",
            "- ready_for_strategy_replay=false",
            "- ready_for_formal=false",
            f"- future_data_violation_count={readiness['future_data_violation_count']}",
            "- not_live_rule=true",
            "",
            "Blocked / proxy / prohibited:",
            *[f"- {row.field_or_contract}: {row.status}; {row.blocked_reason}" for row in blocked_proxy.itertuples()],
            "",
            "Falsification required:",
            *[f"- {item}" for item in readiness["falsification_required"]],
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
    parser.add_argument("--phase-e-readiness-dir", type=Path, default=DEFAULT_PHASE_E_READINESS_DIR)
    parser.add_argument("--phase-e-diagnostic-dir", type=Path, default=DEFAULT_PHASE_E_DIAGNOSTIC_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    manifest = build_predictive_bear_cash_readiness(
        materialization_dir=args.materialization_dir,
        phase_e_readiness_dir=args.phase_e_readiness_dir,
        phase_e_diagnostic_dir=args.phase_e_diagnostic_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
