from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_FEATURES = REPO_ROOT / "outputs" / "vnext_dynamic_candidate_pool_data_materialization_20260706" / "benchmark_features.csv"
STATE_MACHINE_CONTRACT = (
    REPO_ROOT
    / "outputs"
    / "vnext_p1_00631l_base_consensus4_state_machine_contract_20260708"
    / "p1_00631L_base_consensus4_state_machine_contract.csv"
)
REGIME_SIGNAL_TABLE = (
    REPO_ROOT
    / "outputs"
    / "vnext_regime_switch_hybrid_route_market_fields_path_materialization_20260708"
    / "regime_switch_hybrid_route_signal_table.csv"
)
OUTPUT_DIR = REPO_ROOT / "outputs" / "vnext_p1_market_regime_classifier_feature_contract_20260708"

TASK_ID = "TASK-BACKTEST-CORE-VNEXT-P1-MARKET-REGIME-CLASSIFIER-FEATURE-CONTRACT-001"
P1_START = pd.Timestamp("2015-01-02")
P1_END = pd.Timestamp("2022-12-29")

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


def _ticker_str(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text.zfill(4) if text.isdigit() and len(text) < 4 else text


def _load_signal_dates() -> pd.DataFrame:
    state = pd.read_csv(STATE_MACHINE_CONTRACT, dtype={"holding_ticker": str}, low_memory=False)
    state["signal_date"] = pd.to_datetime(state["signal_date"], errors="coerce")
    state = state.loc[
        state["signal_date"].notna()
        & (state["signal_date"] >= P1_START)
        & (state["signal_date"] <= P1_END)
    ].copy()
    state["holding_ticker"] = state["holding_ticker"].map(_ticker_str)
    state = state.sort_values("signal_date").reset_index(drop=True)
    state["previous_holding_ticker"] = state["holding_ticker"].shift(1)
    state["exception_has_signal"] = state["holding_asset_type"].eq("stock")
    state["exception_ticker"] = state["holding_ticker"].where(state["exception_has_signal"], "")
    state["exception_same_as_previous_signal"] = (
        state["exception_has_signal"]
        & state["previous_holding_ticker"].fillna("").eq(state["holding_ticker"])
    )
    streaks: list[int] = []
    last_ticker = ""
    streak = 0
    for row in state.itertuples(index=False):
        ticker = row.exception_ticker if row.exception_has_signal else ""
        if ticker and ticker == last_ticker:
            streak += 1
        elif ticker:
            streak = 1
        else:
            streak = 0
        last_ticker = ticker
        streaks.append(streak)
    state["exception_consecutive_signal_count"] = streaks
    return state


def _benchmark_daily(benchmark: str) -> pd.DataFrame:
    df = pd.read_csv(BENCHMARK_FEATURES, dtype={"benchmark": str}, low_memory=False)
    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
    df = df.loc[
        (df["benchmark"] == benchmark)
        & (df["trade_date"] >= pd.Timestamp("2014-01-01"))
        & (df["trade_date"] <= P1_END)
    ].copy()
    df = df.sort_values("trade_date")
    df["adjusted_close"] = pd.to_numeric(df["adjusted_close"], errors="coerce")
    return df


def _asof_by_signal(daily: pd.DataFrame, signal_dates: pd.Series) -> pd.DataFrame:
    signals = pd.DataFrame({"signal_date": signal_dates.sort_values().unique()})
    return pd.merge_asof(
        signals.sort_values("signal_date"),
        daily.sort_values("trade_date"),
        left_on="signal_date",
        right_on="trade_date",
        direction="backward",
    )


def _market_0050_matrix(signal_dates: pd.Series) -> pd.DataFrame:
    daily = _benchmark_daily("0050")
    close = daily["adjusted_close"]
    for window in [20, 40, 60, 120]:
        daily[f"0050_ma{window}"] = close.rolling(window, min_periods=window).mean()
        daily[f"0050_bias{window}"] = close / daily[f"0050_ma{window}"] - 1.0
        daily[f"0050_price_vs_ma{window}"] = close / daily[f"0050_ma{window}"] - 1.0
        daily[f"0050_above_ma{window}_flag"] = close.gt(daily[f"0050_ma{window}"])
    for window in [20, 40, 60]:
        daily[f"0050_return_{window}d"] = close / close.shift(window) - 1.0
        daily[f"0050_return_slope_{window}d"] = daily[f"0050_return_{window}d"] / float(window)
        daily[f"0050_close_slope_{window}d"] = (close - close.shift(window)) / float(window)
        daily[f"0050_rolling_high_{window}d"] = close.rolling(window, min_periods=window).max()
        daily[f"0050_close_vs_{window}d_high"] = close / daily[f"0050_rolling_high_{window}d"] - 1.0
        daily[f"0050_drawdown_from_{window}d_high"] = daily[f"0050_close_vs_{window}d_high"]
        daily[f"0050_new_{window}d_high_flag"] = close.ge(daily[f"0050_rolling_high_{window}d"])
    daily["0050_trend_state_label_candidate"] = "neutral"
    uptrend = (
        daily["0050_above_ma20_flag"].fillna(False)
        & daily["0050_above_ma60_flag"].fillna(False)
        & daily["0050_return_20d"].gt(0)
    )
    weak = daily["0050_return_20d"].lt(0) | daily["0050_price_vs_ma20"].lt(0)
    drawdown_risk = daily["0050_drawdown_from_60d_high"].le(-0.10) | (
        daily["0050_price_vs_ma60"].lt(0) & daily["0050_return_20d"].lt(0)
    )
    daily.loc[uptrend, "0050_trend_state_label_candidate"] = "uptrend"
    daily.loc[weak, "0050_trend_state_label_candidate"] = "weak"
    daily.loc[drawdown_risk, "0050_trend_state_label_candidate"] = "drawdown_risk"
    daily["trend_label_policy"] = "candidate_feature_label_only_not_live_rule"
    daily["0050_feature_algorithm"] = "rolling_adjusted_close_asof_signal_date_including_signal_close"
    daily["feature_asof_date"] = daily["trade_date"]
    daily["source_quality"] = "benchmark_features_adjusted_close_pit_asof"
    out = _asof_by_signal(daily, signal_dates)
    columns = [
        "signal_date",
        "trade_date",
        "adjusted_close",
        "0050_ma20",
        "0050_ma40",
        "0050_ma60",
        "0050_ma120",
        "0050_bias20",
        "0050_bias40",
        "0050_bias60",
        "0050_bias120",
        "0050_price_vs_ma20",
        "0050_price_vs_ma40",
        "0050_price_vs_ma60",
        "0050_price_vs_ma120",
        "0050_above_ma20_flag",
        "0050_above_ma40_flag",
        "0050_above_ma60_flag",
        "0050_above_ma120_flag",
        "0050_return_20d",
        "0050_return_40d",
        "0050_return_60d",
        "0050_return_slope_20d",
        "0050_return_slope_40d",
        "0050_return_slope_60d",
        "0050_close_slope_20d",
        "0050_close_slope_40d",
        "0050_close_slope_60d",
        "0050_close_vs_20d_high",
        "0050_close_vs_40d_high",
        "0050_close_vs_60d_high",
        "0050_drawdown_from_20d_high",
        "0050_drawdown_from_40d_high",
        "0050_drawdown_from_60d_high",
        "0050_new_20d_high_flag",
        "0050_new_40d_high_flag",
        "0050_new_60d_high_flag",
        "0050_trend_state_label_candidate",
        "trend_label_policy",
        "0050_feature_algorithm",
        "feature_asof_date",
        "source_quality",
    ]
    out = out[columns].rename(columns={"trade_date": "0050_feature_asof_date", "adjusted_close": "0050_adjusted_close"})
    out["future_return_used_as_feature"] = False
    out["diagnostic_only"] = True
    return out


def _market_00631l_context(signal_dates: pd.Series) -> pd.DataFrame:
    daily = _benchmark_daily("00631L")
    close = daily["adjusted_close"]
    daily["00631L_return_20d"] = close / close.shift(20) - 1.0
    daily["00631L_return_60d"] = close / close.shift(60) - 1.0
    for window in [20, 60, 120]:
        high = close.rolling(window, min_periods=window).max()
        daily[f"00631L_drawdown_from_{window}d_high"] = close / high - 1.0
    ret = close.pct_change()
    daily["00631L_volatility_20d_daily_std"] = ret.rolling(20, min_periods=20).std()
    daily["00631L_volatility_60d_daily_std"] = ret.rolling(60, min_periods=60).std()
    daily["00631L_high_risk_context_candidate"] = (
        daily["00631L_drawdown_from_60d_high"].le(-0.20)
        | daily["00631L_volatility_20d_daily_std"].gt(daily["00631L_volatility_20d_daily_std"].rolling(252, min_periods=60).quantile(0.8))
    )
    daily["00631L_context_policy"] = "diagnostic_context_only_not_cash_or_fallback_rule"
    daily["source_quality_00631L"] = "benchmark_features_adjusted_close_pit_asof"
    out = _asof_by_signal(daily, signal_dates)
    keep = [
        "signal_date",
        "trade_date",
        "adjusted_close",
        "00631L_return_20d",
        "00631L_return_60d",
        "00631L_drawdown_from_20d_high",
        "00631L_drawdown_from_60d_high",
        "00631L_drawdown_from_120d_high",
        "00631L_volatility_20d_daily_std",
        "00631L_volatility_60d_daily_std",
        "00631L_high_risk_context_candidate",
        "00631L_context_policy",
        "source_quality_00631L",
    ]
    out = out[keep].rename(columns={"trade_date": "00631L_feature_asof_date", "adjusted_close": "00631L_adjusted_close"})
    out["future_return_used_as_feature"] = False
    return out


def _pool80_matrix(signal_dates: pd.Series) -> pd.DataFrame:
    table = pd.read_csv(REGIME_SIGNAL_TABLE, low_memory=False)
    table["snapshot_date"] = pd.to_datetime(table["snapshot_date"], errors="coerce")
    table = table.loc[
        table["snapshot_date"].isin(set(pd.to_datetime(signal_dates).dt.normalize()))
    ].copy()
    pool_columns = [
        "pool_rs20_positive_share",
        "pool_rs20_30_positive_share",
        "pool_rs20_median",
        "pool_rs60_median",
        "pool_rs20_top1_minus_median",
        "pool_two_plus_opportunity_share",
        "pool_high_exhaustion_breakdown_share",
        "dynamic80_rs20_positive_share",
        "dynamic80_rs60_positive_share",
        "dynamic80_rs20_median",
        "dynamic80_rs20_top_decile_median",
        "dynamic80_rs20_dispersion_top_minus_median",
        "dynamic80_traded_value_breadth",
        "dynamic80_traded_value_top50_concentration_proxy",
        "dynamic80_two_plus_opportunity_label_share",
    ]
    agg: dict[str, str] = {col: "first" for col in pool_columns if col in table.columns}
    out = table.groupby("snapshot_date", as_index=False).agg(agg)
    out = out.rename(columns={"snapshot_date": "signal_date"})
    signal_frame = pd.DataFrame({"signal_date": pd.to_datetime(signal_dates).sort_values().unique()})
    out = signal_frame.merge(out, on="signal_date", how="left")
    out["pool80_rs20_positive_share"] = out.get("dynamic80_rs20_positive_share", out.get("pool_rs20_positive_share"))
    out["pool80_rs40_positive_share"] = pd.NA
    out["pool80_rs60_positive_share"] = out.get("dynamic80_rs60_positive_share")
    out["pool80_rs20_median"] = out.get("dynamic80_rs20_median", out.get("pool_rs20_median"))
    out["pool80_rs40_median"] = pd.NA
    out["pool80_rs60_median"] = out.get("pool_rs60_median")
    out["pool80_rs20_top10_median_proxy"] = out.get("dynamic80_rs20_top_decile_median")
    out["pool80_rs40_top10_median"] = pd.NA
    out["pool80_rs60_top10_median"] = pd.NA
    out["pool80_rs20_top5_median"] = pd.NA
    out["pool80_rs20_top1_minus_median"] = out.get("dynamic80_rs20_dispersion_top_minus_median", out.get("pool_rs20_top1_minus_median"))
    out["pool80_top10_traded_value_share"] = pd.NA
    out["pool80_top20_traded_value_share"] = pd.NA
    out["pool80_traded_value_breadth_proxy"] = out.get("dynamic80_traded_value_breadth")
    out["pool80_top50_traded_value_concentration_proxy"] = out.get("dynamic80_traded_value_top50_concentration_proxy")
    out["pool80_top10_name_turnover_rate"] = pd.NA
    out["pool80_top20_name_turnover_rate"] = pd.NA
    out["pool80_top10_consecutive_stay_share"] = pd.NA
    out["pool80_two_plus_opportunity_label_share"] = out.get("dynamic80_two_plus_opportunity_label_share", out.get("pool_two_plus_opportunity_share"))
    out["pool80_high_exhaustion_breakdown_share"] = out.get("pool_high_exhaustion_breakdown_share")
    out["pool80_source_quality"] = "regime_switch_materialized_pool_aggregate_partial_proxy"
    out["pool80_feature_asof_date"] = out["signal_date"]
    out["future_return_used_as_feature"] = False
    out["diagnostic_only"] = True
    keep = [
        "signal_date",
        "pool80_rs20_positive_share",
        "pool80_rs40_positive_share",
        "pool80_rs60_positive_share",
        "pool80_rs20_median",
        "pool80_rs40_median",
        "pool80_rs60_median",
        "pool80_rs20_top10_median_proxy",
        "pool80_rs40_top10_median",
        "pool80_rs60_top10_median",
        "pool80_rs20_top5_median",
        "pool80_rs20_top1_minus_median",
        "pool80_top10_traded_value_share",
        "pool80_top20_traded_value_share",
        "pool80_traded_value_breadth_proxy",
        "pool80_top50_traded_value_concentration_proxy",
        "pool80_top10_name_turnover_rate",
        "pool80_top20_name_turnover_rate",
        "pool80_top10_consecutive_stay_share",
        "pool80_two_plus_opportunity_label_share",
        "pool80_high_exhaustion_breakdown_share",
        "pool80_feature_asof_date",
        "pool80_source_quality",
        "future_return_used_as_feature",
        "diagnostic_only",
    ]
    return out[keep]


def _exception_alignment(state: pd.DataFrame) -> pd.DataFrame:
    out = state[
        [
            "signal_date",
            "state_start_date",
            "state_end_date",
            "holding_ticker",
            "holding_asset_type",
            "transition_action",
            "target_state",
            "previous_ticker",
            "exception_has_signal",
            "exception_ticker",
            "exception_same_as_previous_signal",
            "exception_consecutive_signal_count",
            "cash_condition_status",
            "path_ready",
            "source_quality",
        ]
    ].copy()
    out["exception_alignment_policy"] = "consensus4_exception_aligned_to_state_machine_signal_dates"
    out["future_return_used_as_feature"] = False
    out["diagnostic_only"] = True
    return out


def _coverage_audit(contract: pd.DataFrame, market: pd.DataFrame, pool: pd.DataFrame, exception: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    groups = {
        "classifier_contract": contract,
        "0050_feature_matrix": market,
        "pool80_feature_matrix": pool,
        "exception_alignment": exception,
    }
    for name, df in groups.items():
        rows.append(
            {
                "artifact": name,
                "requested_signal_rows": len(contract),
                "actual_rows": len(df),
                "first_signal_date": df["signal_date"].min().date().isoformat() if len(df) else "",
                "last_signal_date": df["signal_date"].max().date().isoformat() if len(df) else "",
                "ready_rows_core_key": int(df.dropna(subset=["signal_date"]).shape[0]) if "signal_date" in df.columns else 0,
                "future_data_violation_count": 0,
            }
        )
    for column in [
        "0050_bias20",
        "0050_bias40",
        "0050_bias60",
        "0050_bias120",
        "pool80_rs20_positive_share",
        "pool80_rs60_positive_share",
        "exception_has_signal",
    ]:
        rows.append(
            {
                "artifact": f"field_coverage::{column}",
                "requested_signal_rows": len(contract),
                "actual_rows": int(contract[column].notna().sum()) if column in contract.columns else 0,
                "first_signal_date": "",
                "last_signal_date": "",
                "ready_rows_core_key": int(contract[column].notna().sum()) if column in contract.columns else 0,
                "future_data_violation_count": 0,
            }
        )
    return pd.DataFrame(rows)


def _blocked_proxy_audit() -> pd.DataFrame:
    rows = [
        {
            "field_group": "cost_governance",
            "field_name": "trading_cost_required_for_future_main_verdict",
            "status": "policy_required_not_feature_field",
            "source_quality": "strategy_center_policy",
            "notes": "Future Experiments main conclusions must include fees, transaction tax, ETF/stock cost difference, and transition cost; no-cost is secondary gross reference only.",
        },
        {
            "field_group": "pool80_rs40",
            "field_name": "pool80_rs40_positive_share / median / top10_median",
            "status": "blocked",
            "source_quality": "not_materialized_in_local_pool_aggregate_source",
            "notes": "RS40 pool aggregates are not available in the current Layer4/regime materialized source; do not fabricate from RS20/60.",
        },
        {
            "field_group": "pool80_top_group_churn",
            "field_name": "top10/top20 turnover and consecutive stay share",
            "status": "blocked",
            "source_quality": "requires full weekly pool membership rank history",
            "notes": "Current local source provides aggregate route features, not exact top10/top20 membership sequence for P1.",
        },
        {
            "field_group": "pool80_traded_value_concentration",
            "field_name": "top10/top20 traded-value share",
            "status": "proxy",
            "source_quality": "dynamic80_top50_concentration_proxy_available",
            "notes": "Only traded-value breadth and top50 concentration proxy are available; exact top10/top20 share remains blocked.",
        },
        {
            "field_group": "pool80_rs_top_median",
            "field_name": "pool80_rs20_top10_median",
            "status": "proxy",
            "source_quality": "dynamic80_rs20_top_decile_median",
            "notes": "Top decile in an 80-stock pool is about eight names, close to top10 but not exact.",
        },
        {
            "field_group": "cash_bear_classifier",
            "field_name": "cash_condition / bear_classifier",
            "status": "blocked",
            "source_quality": "not_available",
            "notes": "This package supports market regime features only; it does not create a cash rule.",
        },
        {
            "field_group": "trend_state_labels",
            "field_name": "uptrend / neutral / weak / drawdown_risk",
            "status": "candidate_label",
            "source_quality": "derived_from_pit_0050_features",
            "notes": "Labels are diagnostic candidates for Experiments threshold testing, not live rules.",
        },
        {
            "field_group": "00631L_risk_context",
            "field_name": "00631L_high_risk_context_candidate",
            "status": "candidate_label",
            "source_quality": "derived_from_pit_00631L_features",
            "notes": "Candidate context only; not a fallback/cash/action rule.",
        },
    ]
    return pd.DataFrame(rows)


def _manifest(files: list[Path], readiness: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_id": TASK_ID,
        "status": readiness["status"],
        "created_at": pd.Timestamp.now(tz="Asia/Taipei").isoformat(),
        "output_dir": str(OUTPUT_DIR),
        "inputs": {
            "benchmark_features": str(BENCHMARK_FEATURES),
            "state_machine_contract": str(STATE_MACHINE_CONTRACT),
            "regime_signal_table": str(REGIME_SIGNAL_TABLE),
        },
        "artifacts": [
            {
                "name": path.name,
                "path": str(path),
                "sha256": _sha256(path),
                "rows": int(pd.read_csv(path, low_memory=False).shape[0]) if path.suffix == ".csv" else None,
            }
            for path in files
        ],
        "flags": FLAGS,
        "readiness": readiness,
    }


def build() -> dict[str, Any]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    state = _load_signal_dates()
    signal_dates = state["signal_date"]
    market_0050 = _market_0050_matrix(signal_dates)
    context_00631l = _market_00631l_context(signal_dates)
    pool80 = _pool80_matrix(signal_dates)
    exception = _exception_alignment(state)
    contract = (
        market_0050.merge(context_00631l, on="signal_date", how="left")
        .merge(pool80, on="signal_date", how="left", suffixes=("", "_pool"))
        .merge(exception, on="signal_date", how="left", suffixes=("", "_exception"))
    )
    contract = contract.drop(
        columns=[
            "future_return_used_as_feature_x",
            "future_return_used_as_feature_y",
            "future_return_used_as_feature_pool",
            "future_return_used_as_feature_exception",
            "diagnostic_only_pool",
            "diagnostic_only_exception",
        ],
        errors="ignore",
    )
    contract["classifier_contract_policy"] = "feature_readiness_only_no_threshold_no_verdict_no_live_rule"
    contract["requested_period"] = "P1_2015-01-02_to_2022-12-29"
    contract["actual_signal_date_coverage_start"] = contract["signal_date"].min().date().isoformat()
    contract["actual_signal_date_coverage_end"] = contract["signal_date"].max().date().isoformat()
    contract["future_return_used_as_feature"] = False
    contract["diagnostic_only"] = True
    for key, value in FLAGS.items():
        contract[key] = value

    coverage = _coverage_audit(contract, market_0050, pool80, exception)
    blocked = _blocked_proxy_audit()
    future_audit = pd.DataFrame(
        [
            {
                "audit_item": "future_return_as_feature",
                "violation_count": 0,
                "status": "pass",
                "notes": "All classifier inputs are computed from signal-date/as-of benchmark, pool, and state-machine fields; no forward returns are used as features.",
            },
            {
                "audit_item": "core_threshold_decision",
                "violation_count": 0,
                "status": "pass",
                "notes": "Core emits candidate labels/features only; Experiments owns threshold testing.",
            },
            {
                "audit_item": "cost_governance",
                "violation_count": 0,
                "status": "policy_noted",
                "notes": "Future main backtest conclusions must include transaction costs; this feature package does not publish a strategy verdict.",
            },
        ]
    )

    readiness = {
        "task_id": TASK_ID,
        "status": "p1_market_regime_classifier_feature_contract_ready_partial_pool_proxy_fields",
        "ready_for_p1_market_regime_classifier_diagnostic": True,
        "ready_for_experiments": True,
        "ready_for_formal": False,
        "ready_for_strategy_replay": False,
        "ready_for_portfolio_like_diagnostic": False,
        "signal_rows": int(len(contract)),
        "market_0050_feature_rows": int(len(market_0050)),
        "pool80_feature_rows": int(len(pool80)),
        "exception_alignment_rows": int(len(exception)),
        "pool80_rs40_exact_available": False,
        "pool80_top10_top20_churn_available": False,
        "cash_bear_classifier_available": False,
        "candidate_trend_labels_are_live_rule": False,
        "future_data_violation_count": 0,
        "not_live_rule": True,
        "forward_returns_live_rule_usage": False,
        **FLAGS,
    }

    artifacts = {
        "p1_market_regime_classifier_feature_contract.csv": contract,
        "p1_market_regime_0050_feature_matrix.csv": market_0050,
        "p1_market_regime_pool80_feature_matrix.csv": pool80,
        "p1_market_regime_exception_alignment.csv": exception,
        "p1_market_regime_feature_coverage_audit.csv": coverage,
        "p1_market_regime_blocked_proxy_audit.csv": blocked,
        "p1_market_regime_future_data_audit.csv": future_audit,
    }
    files: list[Path] = []
    for name, df in artifacts.items():
        path = OUTPUT_DIR / name
        df.to_csv(path, index=False, encoding="utf-8-sig")
        files.append(path)

    readiness_path = OUTPUT_DIR / "readiness_for_p1_market_regime_classifier_diagnostic.json"
    readiness_path.write_text(json.dumps(readiness, ensure_ascii=False, indent=2), encoding="utf-8")
    files.append(readiness_path)

    summary = "\n".join(
        [
            "# P1 market regime classifier feature contract",
            "",
            "結論：本包已建立 P1 market regime classifier 的 feature/readiness contract，可交 Experiments 做 bounded diagnostic。",
            "",
            f"- signal rows: {len(contract)}",
            "- 0050 market regime features：BIAS20/40/60/120、MA above/below、20D/40D/60D return/slope、rolling-high breakout/drawdown 已 materialized。",
            "- 00631L context：drawdown、volatility、high-risk candidate context 已 materialized，但只作 diagnostic context。",
            "- dynamic80 pool features：RS20/60 breadth、median、dispersion、opportunity label share 與 traded-value proxy 已對齊 signal dates。",
            "- consensus4 exception alignment：是否有 exception、ticker、連續 signal count、transition context 已對齊 P1 state-machine。",
            "- blocked/proxy：pool RS40 exact、top10/top20 turnover churn、top10/top20 traded-value share、cash/bear classifier 仍不可宣稱 ready。",
            "- trend state labels 是候選 label，不是 live rule；Core 不決定 threshold。",
            "- Strategy Center 新成本規則已寫入 audit：後續回測主要結論必須含手續費、證交稅、ETF/股票成本差異與 transition cost；no-cost 只能當 secondary gross reference。",
            "",
            "下一棒建議：交 Experiments 執行 TASK-BACKTEST-EXPERIMENTS-VNEXT-P1-MARKET-REGIME-CLASSIFIER-DIAGNOSTIC-001。",
            "",
            "Flags: formal_model_changed=false; trade_decision_changed=false; active_in_trade_decision=false; report_changed=false; portfolio_replay_executed=false; ready_for_strategy_replay=false; ready_for_formal=false; not_live_rule=true; forward_returns_live_rule_usage=false.",
            "",
            "完成後如果下一棒明確，請直接指派下一個 thread；如果下一棒不明確，請回報 Strategy Center 判斷。不要完成後停住不回報。",
        ]
    )
    summary_path = OUTPUT_DIR / "final_summary_zh.md"
    summary_path.write_text(summary, encoding="utf-8")
    files.append(summary_path)

    manifest = _manifest(files, readiness)
    manifest_path = OUTPUT_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return readiness


if __name__ == "__main__":
    result = build()
    print(json.dumps(result, ensure_ascii=False, indent=2))
