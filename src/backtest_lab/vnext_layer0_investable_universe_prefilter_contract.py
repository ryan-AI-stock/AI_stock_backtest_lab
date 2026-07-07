"""Build vNext Layer0 investable-universe prefilter contract/readiness.

Layer0 is a data-pruning and investability universe contract, not a trading
rule, selector, formal model, or Experiments diagnostic.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


TASK_ID = "TASK-BACKTEST-CORE-VNEXT-LAYER0-INVESTABLE-UNIVERSE-PREFILTER-CONTRACT-001"
DEFAULT_DATA_DIR = Path("outputs/vnext_dynamic_candidate_pool_data_materialization_20260706")
DEFAULT_OUTPUT_DIR = Path("outputs/vnext_layer0_investable_universe_prefilter_contract_20260707")


def build_contract(
    *,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    data = Path(data_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    daily_path = data / "daily_market_features.csv"
    recent = _recent_traded_value_panel(daily_path)
    coverage = _coverage_estimates(recent)
    source_readiness = _source_readiness()
    rule_design = _rule_design()
    exclusion_policy = _exclusion_policy()
    buffer_policy = _buffer_policy()
    leakage_audit = _future_data_audit()
    next_request = _next_request(source_readiness)
    readiness = _readiness(recent, coverage, source_readiness)

    _write_csv(recent, output / "layer0_recent_traded_value_panel_sample.csv")
    _write_csv(coverage, output / "layer0_prefilter_threshold_coverage_estimate.csv")
    _write_csv(source_readiness, output / "layer0_source_readiness_matrix.csv")
    _write_csv(rule_design, output / "layer0_prefilter_rule_design.csv")
    _write_csv(exclusion_policy, output / "layer0_instrument_exclusion_policy.csv")
    _write_csv(buffer_policy, output / "layer0_buffer_exception_policy.csv")
    _write_csv(leakage_audit, output / "layer0_future_data_audit.csv")
    _write_csv(next_request, output / "layer0_next_radar_request.csv")
    (output / "readiness_for_layer0_investable_universe_prefilter.json").write_text(
        json.dumps(readiness, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    manifest = {
        "task_id": TASK_ID,
        "status": readiness["status"],
        "output_dir": str(output.resolve()),
        "input_data_dir": str(data.resolve()),
        "output_files": [
            "layer0_recent_traded_value_panel_sample.csv",
            "layer0_prefilter_threshold_coverage_estimate.csv",
            "layer0_source_readiness_matrix.csv",
            "layer0_prefilter_rule_design.csv",
            "layer0_instrument_exclusion_policy.csv",
            "layer0_buffer_exception_policy.csv",
            "layer0_future_data_audit.csv",
            "layer0_next_radar_request.csv",
            "readiness_for_layer0_investable_universe_prefilter.json",
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
    (output / "final_summary_zh.md").write_text(_summary(readiness), encoding="utf-8")
    return manifest


def _recent_traded_value_panel(daily_path: Path) -> pd.DataFrame:
    cols = ["trade_date", "ticker", "name", "market", "traded_value", "valid_universe", "liquidity_flag", "listing_status"]
    unique_dates: set[str] = set()
    for chunk in pd.read_csv(daily_path, usecols=["trade_date"], chunksize=500_000):
        unique_dates.update(chunk["trade_date"].astype(str).unique())
    recent_dates = sorted(unique_dates)[-5:]

    parts = []
    for chunk in pd.read_csv(daily_path, usecols=cols, dtype={"ticker": str}, chunksize=500_000):
        part = chunk[chunk["trade_date"].astype(str).isin(recent_dates)].copy()
        if not part.empty:
            parts.append(part)
    if not parts:
        return pd.DataFrame(columns=cols)
    df = pd.concat(parts, ignore_index=True)
    df["traded_value"] = pd.to_numeric(df["traded_value"], errors="coerce").fillna(0)
    df["is_common_stock_like"] = df["ticker"].astype(str).str.fullmatch(r"\d{4}") & ~df["ticker"].astype(str).str.startswith("00")
    df["is_etf_or_etn_like"] = df["ticker"].astype(str).str.startswith("00")
    df["is_ky_name"] = df["name"].astype(str).str.contains("-KY", na=False)
    df["eligible_for_layer0_estimate"] = (
        df["valid_universe"].astype(str).str.lower().eq("true")
        & df["is_common_stock_like"]
        & ~df["is_etf_or_etn_like"]
    )
    grouped = (
        df.groupby(["ticker", "name", "market"], dropna=False)
        .agg(
            recent_5d_traded_value=("traded_value", "sum"),
            observed_days=("trade_date", "nunique"),
            eligible_for_layer0_estimate=("eligible_for_layer0_estimate", "max"),
            is_ky_name=("is_ky_name", "max"),
        )
        .reset_index()
    )
    total = grouped.loc[grouped["eligible_for_layer0_estimate"], "recent_5d_traded_value"].sum()
    grouped["recent_5d_traded_value_share"] = grouped["recent_5d_traded_value"] / total if total else 0
    grouped = grouped.sort_values("recent_5d_traded_value", ascending=False).reset_index(drop=True)
    grouped["traded_value_rank"] = range(1, len(grouped) + 1)
    grouped["cumulative_traded_value_share"] = grouped["recent_5d_traded_value_share"].cumsum()
    grouped["source_quality"] = "exact_recent_daily_traded_value_pit_from_local_daily_market_features"
    grouped["diagnostic_only"] = True
    return grouped


def _coverage_estimates(recent: pd.DataFrame) -> pd.DataFrame:
    eligible = recent[recent["eligible_for_layer0_estimate"].astype(bool)].copy()
    rows = []
    for n in [50, 100, 200, 300, 500]:
        top = eligible.head(n)
        rows.append(
            {
                "rule_family": "traded_value_rank",
                "threshold": f"top_{n}",
                "included_count": int(len(top)),
                "recent_5d_cumulative_traded_value_share": float(top["recent_5d_traded_value_share"].sum()) if not top.empty else 0.0,
                "future_winner_miss_risk": _risk_label(n),
                "source_quality": "exact_recent_traded_value_estimate_not_formal",
                "diagnostic_only": True,
            }
        )
    for share in [0.60, 0.70, 0.80, 0.90]:
        hit = eligible[eligible["cumulative_traded_value_share"].ge(share)]
        count = int(hit["traded_value_rank"].iloc[0]) if not hit.empty else int(len(eligible))
        rows.append(
            {
                "rule_family": "traded_value_rank",
                "threshold": f"cum_share_{int(share * 100)}pct",
                "included_count": count,
                "recent_5d_cumulative_traded_value_share": share,
                "future_winner_miss_risk": _risk_label(count),
                "source_quality": "exact_recent_traded_value_estimate_not_formal",
                "diagnostic_only": True,
            }
        )
    rows.extend(
        [
            {
                "rule_family": "market_cap_rank",
                "threshold": "cum_turnover_share_by_market_cap_rank",
                "included_count": "",
                "recent_5d_cumulative_traded_value_share": "",
                "future_winner_miss_risk": "unknown_until_full_daily_market_cap_or_accepted_proxy",
                "source_quality": "blocked_or_proxy_full_exact_daily_market_cap_unavailable",
                "diagnostic_only": True,
            },
            {
                "rule_family": "hybrid",
                "threshold": "market_cap_top_bucket_OR_traded_value_top_bucket_OR_turnover_surge_exception_plus_buffer",
                "included_count": "design_target_200_to_500",
                "recent_5d_cumulative_traded_value_share": "",
                "future_winner_miss_risk": "lower_than_market_cap_only_because_high_turnover_mid_caps_and_surge_exceptions_retained",
                "source_quality": "contract_design_ready_needs_market_cap_proxy_policy",
                "diagnostic_only": True,
            },
        ]
    )
    return pd.DataFrame(rows)


def _risk_label(count: int) -> str:
    if count <= 100:
        return "high_for_future_winner_miss_risk"
    if count <= 200:
        return "medium_high_needs_surge_buffer"
    if count <= 300:
        return "medium_with_hybrid_buffer"
    return "lower_but_data_cost_higher"


def _source_readiness() -> pd.DataFrame:
    rows = [
        ("daily_per_stock_traded_value", "ready", "daily_market_features.csv traded_value", "exact_pit_local"),
        ("total_market_traded_value", "ready_derived", "sum eligible daily_market_features traded_value by date", "derived_pit_local"),
        ("recent_5d_traded_value_rank", "ready", "aggregate last 5 trading dates", "exact_pit_local"),
        ("turnover_rank_pct_5d_20d_60d", "ready", "attention_features.csv", "exact_or_diagnostic_pit_local"),
        ("full_exact_daily_market_cap", "blocked", "not available full TWSE+TPEx exact daily market cap", "blocked"),
        ("market_cap_proxy", "partial_proxy", "tpex_market_cap_proxy/capital stock proxy exists in Layer1 contracts", "proxy_partial_not_full_universe"),
        ("instrument_type_master", "partial_proxy", "ticker pattern + listing metadata; ETF/warrant/KY policies need source hardening", "proxy"),
        ("disposition_full_delivery_ledger", "blocked", "needs accepted PIT event ledger", "blocked"),
    ]
    return pd.DataFrame(rows, columns=["source_item", "readiness", "local_source_or_gap", "source_quality"]).assign(
        diagnostic_only=True
    )


def _rule_design() -> pd.DataFrame:
    rows = [
        (
            "market_cap_rank_cumulative_turnover_share",
            "Rank ordinary common stocks by PIT market cap; add rows until cumulative traded-value share reaches 60/70/80.",
            "blocked_until_full_exact_or_accepted_proxy_market_cap",
            "Can miss smaller high-turnover momentum names if used alone.",
        ),
        (
            "traded_value_rank_cumulative_turnover_share",
            "Rank by recent PIT 5D/20D traded value; add rows until cumulative traded-value share reaches threshold.",
            "ready_with_local_daily_market_features",
            "Better cost control and liquidity focus, but can over-include short-lived heat.",
        ),
        (
            "hybrid_investable_universe",
            "Keep market-cap top bucket OR traded-value top bucket OR recent turnover-surge exception; add near-threshold buffer.",
            "recommended_design",
            "Best balance: avoids market-cap-only blind spot and reduces future-winner miss risk.",
        ),
        (
            "buffer_near_threshold",
            "Include names just below threshold for 2-4 weeks or until liquidity collapses; diagnostic only.",
            "recommended_design",
            "Prevents churn and keeps emerging mid-cap leaders from being excluded too early.",
        ),
    ]
    return pd.DataFrame(rows, columns=["rule_family", "rule_design", "readiness", "risk_note"]).assign(
        diagnostic_only=True,
        not_live_rule=True,
    )


def _exclusion_policy() -> pd.DataFrame:
    rows = [
        ("ETF_ETN", "exclude_from_common_stock_universe; keep separate benchmark/fallback universe", "ticker starts with 00 or accepted instrument master"),
        ("warrants_structured_products", "exclude", "instrument master required for robust full-universe policy"),
        ("KY", "do_not_auto_exclude; tag separately for governance/risk review", "name contains -KY proxy until exact source"),
        ("disposition_full_delivery", "exclude_or_block_until_event_ledger_confirms status", "blocked until PIT event ledger"),
        ("too_low_liquidity", "exclude if below minimum traded-value/liquidity flags over PIT window", "daily_market_features liquidity_flag + traded_value"),
        ("suspended_delisted", "exclude/block by PIT listing_status", "daily_market_features listing_status plus listing ledger"),
    ]
    return pd.DataFrame(rows, columns=["instrument_group", "policy", "source_basis"]).assign(diagnostic_only=True)


def _buffer_policy() -> pd.DataFrame:
    rows = [
        ("core_keep", "coverage threshold reached by traded_value_rank or hybrid", "target 200-300 initial design"),
        ("near_threshold_buffer", "next 50-100 names by traded value or market-cap proxy", "reduces churn/future-winner miss"),
        ("turnover_surge_exception", "include if recent 5D traded value rank improves materially vs 60D rank", "needs PIT rank deltas"),
        ("case_trace_buffer", "allow case_trace_only rows for known review cases without selected outcome inclusion", "diagnostic_only"),
    ]
    return pd.DataFrame(rows, columns=["buffer_type", "policy", "purpose"]).assign(diagnostic_only=True)


def _future_data_audit() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("forward_return_as_rule", "passed", 0, "no forward returns used in prefilter design"),
            ("asof_traded_value", "passed", 0, "use only daily traded value up to signal/asof date"),
            ("market_cap_proxy", "blocked_or_proxy", 0, "must use PIT available market cap/proxy only"),
            ("formal_selector", "not_applicable", 0, "Layer0 is data-pruning, not selector"),
        ],
        columns=["audit_item", "status", "future_data_violation_count", "note"],
    )


def _next_request(source_readiness: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "next_owner": "Radar/Data",
                "handoff_action": "provide_or_harden_layer0_full_universe_instrument_master_and_market_cap_proxy_sources",
                "ready": True,
                "reason": "Core local data is enough for traded-value based Layer0 estimate, but market-cap rank and instrument exclusion need stronger full-universe PIT/proxy sources",
                "requested_sources": "full daily traded value source confirmation; total market turnover source or derivation acceptance; instrument type master; PIT disposition/full-delivery ledger; accepted market-cap proxy policy",
                "diagnostic_only": True,
                "formal_model_changed": False,
                "trade_decision_changed": False,
                "active_in_trade_decision": False,
                "report_changed": False,
                "portfolio_replay_executed": False,
                "ready_for_strategy_replay": False,
                "not_live_rule": True,
                "forward_returns_live_rule_usage": False,
            }
        ]
    )


def _readiness(recent: pd.DataFrame, coverage: pd.DataFrame, source_readiness: pd.DataFrame) -> dict[str, Any]:
    return {
        "task_id": TASK_ID,
        "status": "layer0_investable_universe_prefilter_design_ready_traded_value_ready_market_cap_partial",
        "recommended_name": "Layer0 investable universe / data-pruning filter",
        "not_layer1_selector": True,
        "recent_panel_rows": int(len(recent)),
        "recent_panel_eligible_rows": int(recent["eligible_for_layer0_estimate"].astype(bool).sum()) if not recent.empty else 0,
        "actual_recent_coverage_basis": "latest_5_trade_dates_from_daily_market_features",
        "traded_value_prefilter_ready": True,
        "total_market_traded_value_ready": True,
        "market_cap_rank_prefilter_ready": False,
        "market_cap_rank_prefilter_status": "blocked_or_proxy_until_full_exact_or_accepted_market_cap_proxy",
        "hybrid_prefilter_contract_ready": True,
        "recommended_initial_universe_size": "200_to_500_with_buffer",
        "ready_for_layer0_event_or_coverage_diagnostic": False,
        "ready_for_radar_source_hardening": True,
        "ready_for_experiments": False,
        "ready_for_formal": False,
        "ready_for_strategy_replay": False,
        "future_data_violation_count": 0,
        "blocked_fields": [
            "full_exact_daily_market_cap",
            "free_float_market_cap",
            "instrument_type_master",
            "disposition_full_delivery_pit_ledger",
            "formal_market_cap_rank_policy",
        ],
        "proxy_fields": [
            "market_cap_proxy",
            "instrument_type_by_ticker_pattern",
            "KY_name_tag",
        ],
        "formal_model_changed": False,
        "trade_decision_changed": False,
        "active_in_trade_decision": False,
        "report_changed": False,
        "portfolio_replay_executed": False,
        "not_live_rule": True,
        "forward_returns_live_rule_usage": False,
    }


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def _summary(readiness: dict[str, Any]) -> str:
    return f"""# Layer0 investable universe prefilter contract

## Verdict
- status={readiness["status"]}
- recommended_name={readiness["recommended_name"]}
- traded_value_prefilter_ready=true
- total_market_traded_value_ready=true
- market_cap_rank_prefilter_ready=false
- hybrid_prefilter_contract_ready=true
- recommended_initial_universe_size={readiness["recommended_initial_universe_size"]}
- ready_for_experiments=false
- ready_for_formal=false

## Plain Summary
這應命名為 Layer0 investable universe / data-pruning filter，而不是 Layer1 selector。目的只是先用 PIT 可見的成交金額、流動性與標的類型，把 1900 檔縮成較可補基本面的 200-500 檔級距。

本機資料足以做 traded-value based prefilter 和 total market traded-value share estimate。Market-cap rank 版本仍需要 full daily market cap 或明確接受 proxy，否則只能標 proxy/blocked。

## Flags
- formal_model_changed=false
- trade_decision_changed=false
- active_in_trade_decision=false
- report_changed=false
- portfolio_replay_executed=false
- ready_for_strategy_replay=false
- not_live_rule=true
- forward_returns_live_rule_usage=false
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()
    manifest = build_contract(data_dir=args.data_dir, output_dir=args.output_dir)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
