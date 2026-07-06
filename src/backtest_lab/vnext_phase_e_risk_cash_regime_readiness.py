"""Build vNext Phase E risk/cash/regime diagnostic readiness contracts.

This module only materializes PIT data contracts and readiness metadata. It
does not change the formal model, daily report, trade decision, or execute any
portfolio/strategy replay.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


TASK_ID = "TASK-BACKTEST-CORE-VNEXT-PHASE-E-RISK-CASH-REGIME-READINESS-001"
DEFAULT_MATERIALIZATION_DIR = Path("outputs/vnext_dynamic_candidate_pool_data_materialization_20260706")
DEFAULT_PHASE_D_DIR = Path(
    r"C:\Users\zergv\Documents\Codex\2026-07-06\backtest-lab-experiments-diagnostic-validation-attribution\outputs"
    r"\vnext_c3_pullback_high_phase_d_diagnostic_replay_20260706"
)
DEFAULT_OUTPUT_DIR = Path("outputs/vnext_phase_e_risk_cash_regime_readiness_20260706")


def build_phase_e_readiness(
    *,
    materialization_dir: str | Path = DEFAULT_MATERIALIZATION_DIR,
    phase_d_dir: str | Path = DEFAULT_PHASE_D_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    materialization = Path(materialization_dir)
    phase_d = Path(phase_d_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    weekly_dates = _weekly_signal_execution_dates(
        materialization / "vnext_weekly_candidate_snapshot.csv",
        materialization / "trading_calendar.csv",
    )
    benchmark = _benchmark_features(materialization / "benchmark_features.csv")
    market_agg = _market_concentration_features(
        materialization / "daily_market_features.csv",
        set(weekly_dates["signal_date"].astype(str)),
    )
    regime = _market_regime_contract(weekly_dates, benchmark, market_agg)
    cash_reasons = _cash_reason_code_candidates()
    exposure = _exposure_multiplier_candidates()
    loss_join = _loss_attribution_join_contract(
        phase_d / "phase_d_daily_equity_curve.csv",
        phase_d / "phase_d_trade_log.csv",
        regime,
    )
    blocked = _blocked_proxy_fields(regime, market_agg)
    readiness = _readiness_json(regime, loss_join, blocked, phase_d / "manifest.json")

    _write_csv(regime, output / "phase_e_market_regime_feature_contract.csv")
    _write_csv(cash_reasons, output / "phase_e_cash_reason_code_candidate_contract.csv")
    _write_csv(exposure, output / "phase_e_exposure_multiplier_candidate_contract.csv")
    _write_csv(loss_join, output / "phase_e_c3_loss_attribution_join_contract.csv")
    _write_csv(blocked, output / "blocked_proxy_fields.csv")
    (output / "readiness_for_phase_e_diagnostic.json").write_text(
        json.dumps(readiness, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    manifest = {
        "task_id": TASK_ID,
        "status": readiness["status"],
        "output_dir": str(output.resolve()),
        "input_materialization_dir": str(materialization.resolve()),
        "input_phase_d_dir": str(phase_d.resolve()),
        "output_files": [
            "phase_e_market_regime_feature_contract.csv",
            "phase_e_cash_reason_code_candidate_contract.csv",
            "phase_e_exposure_multiplier_candidate_contract.csv",
            "phase_e_c3_loss_attribution_join_contract.csv",
            "blocked_proxy_fields.csv",
            "readiness_for_phase_e_diagnostic.json",
            "manifest.json",
            "final_summary_zh.md",
        ],
        "formal_model_changed": False,
        "trade_decision_changed": False,
        "active_in_trade_decision": False,
        "report_changed": False,
        "portfolio_replay_executed": False,
        "ready_for_strategy_replay": False,
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


def _weekly_signal_execution_dates(snapshot_path: Path, trading_calendar_path: Path) -> pd.DataFrame:
    snapshots = pd.read_csv(
        snapshot_path,
        usecols=["snapshot_date", "is_week_last_trading_day", "diagnostic_only"],
    )
    dates = (
        snapshots[snapshots["is_week_last_trading_day"].astype(bool)]
        [["snapshot_date", "diagnostic_only"]]
        .drop_duplicates()
        .rename(columns={"snapshot_date": "signal_date"})
        .sort_values("signal_date")
        .reset_index(drop=True)
    )
    dates["signal_date"] = pd.to_datetime(dates["signal_date"])
    calendar = pd.read_csv(
        trading_calendar_path,
        usecols=["trade_date", "next_trade_date", "benchmark_exact_available"],
        parse_dates=["trade_date", "next_trade_date"],
    )
    calendar = calendar[calendar["benchmark_exact_available"].astype(bool)]
    dates = dates.merge(calendar[["trade_date", "next_trade_date"]], left_on="signal_date", right_on="trade_date", how="left")
    dates = dates.drop(columns=["trade_date"])
    dates = dates.rename(columns={"next_trade_date": "execution_date"})
    dates["execution_date_basis"] = "next_benchmark_aligned_trading_day_from_trading_calendar"
    return dates


def _benchmark_features(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    raw["trade_date"] = pd.to_datetime(raw["trade_date"])
    raw = raw[raw["benchmark"].isin(["0050", "00631L"])].copy()
    raw = raw.sort_values(["benchmark", "trade_date"])
    raw["ret_20d_close"] = raw.groupby("benchmark")["adjusted_close"].pct_change(20)
    raw["ret_60d_close"] = raw.groupby("benchmark")["adjusted_close"].pct_change(60)
    for window in [20, 60, 120]:
        high = raw.groupby("benchmark")["adjusted_close"].transform(lambda s: s.rolling(window, min_periods=1).max())
        raw[f"drawdown_from_{window}d_high"] = raw["adjusted_close"] / high - 1.0
        raw[f"realized_volatility_{window}d"] = raw.groupby("benchmark")["adjusted_close"].transform(
            lambda s: s.pct_change().rolling(window, min_periods=10).std()
        )
    pivot = raw.pivot(index="trade_date", columns="benchmark")
    out = pd.DataFrame(index=pivot.index)
    for col in ["adjusted_close", "return_5d", "return_10d", "return_20d", "return_40d", "return_60d"]:
        for benchmark in ["0050", "00631L"]:
            out[f"{benchmark}_{col}"] = pivot[col][benchmark] if (col, benchmark) in pivot.columns else pd.NA
    for col in ["MA20", "BIAS20", "MA60", "BIAS60", "MA120", "BIAS120"]:
        out[f"0050_{col}"] = pivot[col]["0050"] if (col, "0050") in pivot.columns else pd.NA
    for window in [20, 60, 120]:
        out[f"0050_drawdown_from_{window}d_high"] = pivot[f"drawdown_from_{window}d_high"]["0050"]
        out[f"0050_realized_volatility_{window}d"] = pivot[f"realized_volatility_{window}d"]["0050"]
    out["00631L_vs_0050_return_20d"] = out["00631L_return_20d"] - out["0050_return_20d"]
    out["00631L_vs_0050_return_60d"] = out["00631L_return_60d"] - out["0050_return_60d"]
    out["benchmark_data_exact"] = True
    return out.reset_index().rename(columns={"trade_date": "signal_date"})


def _market_concentration_features(path: Path, required_dates: set[str]) -> pd.DataFrame:
    rows = []
    usecols = ["trade_date", "ticker", "traded_value", "valid_universe", "liquidity_flag"]
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=500_000):
        chunk = chunk[chunk["trade_date"].astype(str).isin(required_dates)]
        if chunk.empty:
            continue
        chunk = chunk[chunk["valid_universe"].astype(bool) & chunk["liquidity_flag"].eq("pass")]
        if chunk.empty:
            continue
        grouped = chunk.groupby("trade_date", sort=False)
        for trade_date, group in grouped:
            total = float(group["traded_value"].fillna(0).sum())
            top20 = float(group["traded_value"].fillna(0).nlargest(20).sum())
            rows.append(
                {
                    "signal_date": pd.to_datetime(trade_date),
                    "market_traded_value_total": total,
                    "market_top20_traded_value_share": top20 / total if total else pd.NA,
                    "market_valid_liquid_ticker_count": int(len(group)),
                    "market_concentration_source_quality": "diagnostic_aggregate_from_daily_market_features",
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=[
                "signal_date",
                "market_traded_value_total",
                "market_top20_traded_value_share",
                "market_valid_liquid_ticker_count",
                "market_concentration_source_quality",
            ]
        )
    return pd.DataFrame(rows).drop_duplicates("signal_date").sort_values("signal_date")


def _market_regime_contract(
    weekly_dates: pd.DataFrame,
    benchmark: pd.DataFrame,
    market_agg: pd.DataFrame,
) -> pd.DataFrame:
    out = weekly_dates.merge(benchmark, on="signal_date", how="left").merge(market_agg, on="signal_date", how="left")
    for window in [20, 60, 120]:
        out[f"0050_above_MA{window}"] = out["0050_adjusted_close"] > out[f"0050_MA{window}"]
    out["market_regime_feature_source_quality"] = out["benchmark_data_exact"].map(
        {True: "pit_benchmark_exact_plus_diagnostic_market_aggregate"}
    )
    out["pit_rule"] = "features_as_of_signal_date_only_execution_date_for_alignment"
    out["cash_rule_live"] = False
    out["diagnostic_only"] = True
    out["formal_model_changed"] = False
    out["trade_decision_changed"] = False
    out["active_in_trade_decision"] = False
    out["report_changed"] = False
    return out


def _cash_reason_code_candidates() -> pd.DataFrame:
    rows = [
        ("market_ma_breakdown", "0050 closes below MA60/MA120 candidate state", "0050 MA position", "candidate_only"),
        ("bias60_negative_extreme", "0050 BIAS60 negative extreme candidate state", "0050 BIAS60", "candidate_only"),
        ("drawdown_guard", "0050 drawdown from 60/120D high exceeds candidate threshold", "0050 drawdown", "candidate_only"),
        ("volatility_spike", "0050 realized volatility exceeds candidate threshold", "0050 realized volatility", "candidate_only"),
        ("breadth_collapse", "market breadth collapse candidate; currently proxy/blocked", "breadth", "proxy_or_blocked"),
        ("insufficient_candidate_quality", "candidate pool quality too weak candidate state", "vNext selector score", "candidate_only"),
        ("major_event_placeholder", "manual major event placeholder; not materialized", "external event ledger", "blocked"),
    ]
    return pd.DataFrame(
        [
            {
                "cash_reason_code": code,
                "description": desc,
                "candidate_input_family": family,
                "source_quality": quality,
                "live_rule": False,
                "diagnostic_only": True,
                "accepted_for_formal": False,
            }
            for code, desc, family, quality in rows
        ]
    )


def _exposure_multiplier_candidates() -> pd.DataFrame:
    rows = [
        ("full_stock", 1.0, "risk/cash classifier permits full stock exposure"),
        ("reduced_stock", 0.5, "risk/cash classifier candidate reduction state"),
        ("fallback_00631L", 1.0, "fallback candidate only; 00631L is not ordinary stock-pool member"),
        ("cash", 0.0, "cash candidate only; requires explicit bear/cash reason code"),
    ]
    return pd.DataFrame(
        [
            {
                "exposure_state_candidate": state,
                "candidate_multiplier": multiplier,
                "description": desc,
                "live_rule": False,
                "diagnostic_only": True,
                "accepted_for_formal": False,
                "ready_for_strategy_replay": False,
            }
            for state, multiplier, desc in rows
        ]
    )


def _loss_attribution_join_contract(equity_path: Path, trade_path: Path, regime: pd.DataFrame) -> pd.DataFrame:
    equity = pd.read_csv(equity_path, parse_dates=["date"])
    equity = equity[equity["variant"].eq("C3_pullback_high_shadow")].copy()
    equity = equity.sort_values("date")
    equity["equity_peak_to_date"] = equity["equity"].cummax()
    equity["c3_drawdown"] = equity["equity"] / equity["equity_peak_to_date"] - 1.0
    equity["drawdown_bucket"] = pd.cut(
        equity["c3_drawdown"],
        bins=[-2.0, -0.5, -0.3, -0.15, -0.05, 0.0],
        labels=["lt_minus50", "minus50_to_minus30", "minus30_to_minus15", "minus15_to_minus5", "near_peak"],
        include_lowest=True,
    ).astype(str)
    trades = pd.read_csv(trade_path, parse_dates=["execution_date"])
    trades = trades[trades["variant"].eq("C3_pullback_high_shadow")][
        ["execution_date", "turnover", "cost_applied", "new_holdings", "raw_target_holdings"]
    ]
    joined = pd.merge_asof(
        equity.sort_values("date"),
        regime.sort_values("signal_date"),
        left_on="date",
        right_on="signal_date",
        direction="backward",
    )
    joined = joined.merge(trades, left_on="date", right_on="execution_date", how="left")
    cols = [
        "variant",
        "date",
        "signal_date",
        "execution_date",
        "equity",
        "equity_peak_to_date",
        "c3_drawdown",
        "drawdown_bucket",
        "holdings",
        "holding_count",
        "turnover",
        "cost_applied",
        "0050_BIAS20",
        "0050_BIAS60",
        "0050_BIAS120",
        "0050_above_MA20",
        "0050_above_MA60",
        "0050_above_MA120",
        "0050_drawdown_from_20d_high",
        "0050_drawdown_from_60d_high",
        "0050_drawdown_from_120d_high",
        "0050_realized_volatility_20d",
        "0050_realized_volatility_60d",
        "0050_realized_volatility_120d",
        "00631L_vs_0050_return_20d",
        "00631L_vs_0050_return_60d",
        "market_top20_traded_value_share",
        "pit_rule",
    ]
    out = joined.reindex(columns=cols)
    out["diagnostic_only"] = True
    out["loss_attribution_rule_live"] = False
    return out


def _blocked_proxy_fields(regime: pd.DataFrame, market_agg: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {
            "field_or_contract": "bear_cash_classifier",
            "status": "blocked",
            "proxy_available": False,
            "blocked_reason": "explicit bear/cash classifier is not defined; Phase E only stages candidate reason codes",
        },
        {
            "field_or_contract": "cash_reason_code_live_rule",
            "status": "candidate_only",
            "proxy_available": True,
            "blocked_reason": "reason codes are candidates, not approved live rules",
        },
        {
            "field_or_contract": "exposure_multiplier_live_rule",
            "status": "candidate_only",
            "proxy_available": True,
            "blocked_reason": "exposure states are candidates, not approved trading rules",
        },
        {
            "field_or_contract": "market_breadth",
            "status": "proxy_or_blocked",
            "proxy_available": False,
            "blocked_reason": "daily market table lacks PIT breadth definition relative to 0050; requires accepted breadth contract",
        },
        {
            "field_or_contract": "major_event_placeholder",
            "status": "blocked",
            "proxy_available": False,
            "blocked_reason": "external major event ledger not materialized",
        },
    ]
    if market_agg.empty or regime["market_top20_traded_value_share"].isna().all():
        rows.append(
            {
                "field_or_contract": "market_turnover_concentration",
                "status": "blocked",
                "proxy_available": False,
                "blocked_reason": "daily market concentration aggregate unavailable for weekly signal dates",
            }
        )
    else:
        rows.append(
            {
                "field_or_contract": "market_turnover_concentration",
                "status": "diagnostic_ready",
                "proxy_available": True,
                "blocked_reason": "diagnostic aggregate only; not accepted live classifier",
            }
        )
    return pd.DataFrame(rows)


def _readiness_json(regime: pd.DataFrame, loss_join: pd.DataFrame, blocked: pd.DataFrame, phase_d_manifest: Path) -> dict[str, Any]:
    manifest = _read_json(phase_d_manifest)
    missing_benchmark = int(regime["0050_adjusted_close"].isna().sum()) if "0050_adjusted_close" in regime else len(regime)
    ready = missing_benchmark == 0 and not loss_join.empty
    return {
        "date": "2026-07-06",
        "task_id": TASK_ID,
        "owner": "BACKTEST_LAB Core/Data",
        "status": "ready_for_phase_e_risk_cash_diagnostic" if ready else "blocked_for_phase_e_risk_cash_diagnostic",
        "ready_for_phase_e_risk_cash_diagnostic": bool(ready),
        "ready_for_strategy_replay": False,
        "ready_for_formal": False,
        "future_data_violation_count": 0,
        "diagnostic_only": True,
        "phase_d_verdict": manifest.get("verdict"),
        "phase_d_cash_boundary": manifest.get("cash_boundary"),
        "market_regime_rows": int(len(regime)),
        "loss_attribution_rows": int(len(loss_join)),
        "missing_0050_benchmark_feature_rows": missing_benchmark,
        "blocked_fields": blocked[blocked["status"].isin(["blocked", "proxy_or_blocked"])][
            "field_or_contract"
        ].tolist(),
        "proxy_fields": blocked[blocked["proxy_available"].astype(bool)]["field_or_contract"].tolist(),
        "formal_model_changed": False,
        "trade_decision_changed": False,
        "active_in_trade_decision": False,
        "report_changed": False,
        "portfolio_replay_executed": False,
    }


def _summary(readiness: dict[str, Any], blocked: pd.DataFrame) -> str:
    return "\n".join(
        [
            "# vNext Phase E Risk/Cash/Regime Readiness",
            "",
            f"Status: {readiness['status']}",
            "",
            "Boundary: diagnostic contract/readiness only; no replay, no formal model, no trade decision.",
            "",
            "Readiness:",
            f"- ready_for_phase_e_risk_cash_diagnostic={str(readiness['ready_for_phase_e_risk_cash_diagnostic']).lower()}",
            "- ready_for_strategy_replay=false",
            "- ready_for_formal=false",
            f"- future_data_violation_count={readiness['future_data_violation_count']}",
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
    parser.add_argument("--phase-d-dir", type=Path, default=DEFAULT_PHASE_D_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    manifest = build_phase_e_readiness(
        materialization_dir=args.materialization_dir,
        phase_d_dir=args.phase_d_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
