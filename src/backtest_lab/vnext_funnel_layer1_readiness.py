"""Build vNext funnel Layer 1 fundamental/size/quality PIT readiness.

This is source/contract readiness only. It stages PIT fundamentals, size /
investability proxies, missingness, source quality, and future-data audits for
future layer-by-layer funnel diagnostics. It does not define a formal selector,
use forward returns as rules, or execute any replay.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


TASK_ID = "TASK-BACKTEST-CORE-VNEXT-FUNNEL-LAYER1-FUNDAMENTAL-SIZE-QUALITY-PIT-READINESS-001"
DEFAULT_MATERIALIZATION_DIR = Path("outputs/vnext_dynamic_candidate_pool_data_materialization_20260706")
DEFAULT_OUTPUT_DIR = Path("outputs/vnext_funnel_layer1_fundamental_size_quality_readiness_20260706")

PERIODS = [
    ("P1", "2015-01-02", "2022-12-29"),
    ("P2", "2023-01-02", "2026-06-30"),
    ("2024-latest", "2024-01-02", "2026-06-30"),
    ("2026YTD", "2026-01-02", "2026-06-30"),
]

FUNDAMENTAL_FIELDS = [
    "revenue_growth",
    "profitability",
    "gross_margin",
    "operating_margin",
    "roe_or_quality",
    "cash_flow_quality",
    "debt_or_solvency_risk",
]


def build_funnel_layer1_readiness(
    *,
    materialization_dir: str | Path = DEFAULT_MATERIALIZATION_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    materialization = Path(materialization_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    universe = _weekly_universe(materialization / "vnext_weekly_candidate_snapshot.csv")
    attention = _attention_slice(materialization / "attention_features.csv", universe)
    fundamentals = pd.read_csv(materialization / "fundamental_features.csv", parse_dates=["effective_date"])
    joined = _candidate_join_contract(universe, attention, fundamentals)
    pit_contract = _pit_contract(joined)
    missingness = _missingness_by_period(joined)
    source_quality = _source_quality_matrix(joined, fundamentals)
    blocked = _blocked_proxy_fields(source_quality)
    future_audit = _future_data_audit(joined)
    readiness = _readiness_json(pit_contract, joined, missingness, source_quality, blocked, future_audit)

    _write_csv(pit_contract, output / "funnel_layer1_fundamental_size_quality_pit_contract.csv")
    _write_csv(joined, output / "funnel_layer1_candidate_join_contract.csv")
    _write_csv(missingness, output / "funnel_layer1_missingness_by_period.csv")
    _write_csv(source_quality, output / "funnel_layer1_source_quality_matrix.csv")
    _write_csv(blocked, output / "funnel_layer1_blocked_proxy_fields.csv")
    _write_csv(future_audit, output / "funnel_layer1_future_data_audit.csv")
    (output / "readiness_for_funnel_layer1_diagnostic.json").write_text(
        json.dumps(readiness, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    manifest = {
        "task_id": TASK_ID,
        "status": readiness["status"],
        "output_dir": str(output.resolve()),
        "input_materialization_dir": str(materialization.resolve()),
        "output_files": [
            "funnel_layer1_fundamental_size_quality_pit_contract.csv",
            "funnel_layer1_candidate_join_contract.csv",
            "funnel_layer1_missingness_by_period.csv",
            "funnel_layer1_source_quality_matrix.csv",
            "funnel_layer1_blocked_proxy_fields.csv",
            "funnel_layer1_future_data_audit.csv",
            "readiness_for_funnel_layer1_diagnostic.json",
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


def _weekly_universe(path: Path) -> pd.DataFrame:
    usecols = [
        "snapshot_date",
        "ticker",
        "name",
        "theme_id",
        "theme_name",
        "valid_universe",
        "fundamental_pass",
        "market_attention_member",
        "eligible_pool_member",
        "case_trace_only",
        "diagnostic_only",
        "rank_overall",
        "turnover_state",
        "risk_score",
        "risk_bucket",
    ]
    raw = pd.read_csv(path, usecols=usecols, parse_dates=["snapshot_date"])
    raw = raw[raw["diagnostic_only"].astype(bool) & ~raw["case_trace_only"].astype(bool)].copy()
    raw["ticker"] = raw["ticker"].astype(str)
    return raw.rename(columns={"snapshot_date": "signal_date"})


def _attention_slice(path: Path, universe: pd.DataFrame) -> pd.DataFrame:
    dates = set(universe["signal_date"].dt.strftime("%Y-%m-%d"))
    tickers = set(universe["ticker"].astype(str))
    header = pd.read_csv(path, nrows=0)
    wanted = [
        "trade_date",
        "ticker",
        "traded_value",
        "volume",
        "turnover_5d",
        "turnover_20d",
        "turnover_60d",
        "turnover_rank_pct_5d",
        "turnover_rank_pct_20d",
        "turnover_rank_pct_60d",
        "traded_value_rank_pct",
        "distribution_risk",
    ]
    usecols = [col for col in wanted if col in header.columns]
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


def _candidate_join_contract(universe: pd.DataFrame, attention: pd.DataFrame, fundamentals: pd.DataFrame) -> pd.DataFrame:
    joined = universe.merge(
        attention,
        left_on=["signal_date", "ticker"],
        right_on=["trade_date", "ticker"],
        how="left",
    ).drop(columns=["trade_date"], errors="ignore")
    latest_fund = _latest_fundamentals_asof(joined[["signal_date", "ticker"]].drop_duplicates(), fundamentals)
    joined = joined.merge(latest_fund, on=["signal_date", "ticker"], how="left")
    joined["average_traded_value_proxy_available"] = joined["traded_value"].notna() if "traded_value" in joined else False
    joined["turnover_proxy_available"] = joined[[c for c in ["turnover_5d", "turnover_20d", "turnover_60d"] if c in joined]].notna().any(axis=1)
    joined["market_cap_available"] = False
    joined["paid_in_capital_available"] = False
    joined["listing_board_available"] = False
    joined["industry_pit_available"] = False
    joined["report_period"] = "not_materialized"
    joined["source_date"] = joined["effective_date"]
    joined["disclosure_date"] = joined["effective_date"]
    joined["asof_date"] = joined["effective_date"]
    joined["lag_policy"] = joined["effective_asof_lag_days"].map(lambda value: f"effective_asof_lag_days={value}" if pd.notna(value) else "lag_missing_proxy")
    joined["forward_return_as_rule"] = False
    joined["not_live_rule"] = True
    joined["diagnostic_only"] = True
    return joined


def _latest_fundamentals_asof(keys: pd.DataFrame, fundamentals: pd.DataFrame) -> pd.DataFrame:
    keys = keys.sort_values(["ticker", "signal_date"]).copy()
    fundamentals = fundamentals.copy()
    fundamentals["ticker"] = fundamentals["ticker"].astype(str)
    fundamentals = fundamentals.sort_values(["ticker", "effective_date"])
    parts = []
    for ticker, group in keys.groupby("ticker", sort=False):
        fund = fundamentals[fundamentals["ticker"].eq(ticker)]
        if fund.empty:
            part = group.copy()
            for col in fundamentals.columns:
                if col != "ticker":
                    part[col] = pd.NA
            parts.append(part)
            continue
        parts.append(
            pd.merge_asof(
                group.sort_values("signal_date"),
                fund.sort_values("effective_date"),
                left_on="signal_date",
                right_on="effective_date",
                by="ticker",
                direction="backward",
            )
        )
    out = pd.concat(parts, ignore_index=True)
    out = out.rename(columns={"source_quality": "fundamental_source_quality"})
    return out


def _pit_contract(joined: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "signal_date",
        "ticker",
        "name",
        "theme_id",
        "valid_universe",
        "market_cap_available",
        "paid_in_capital_available",
        "average_traded_value_proxy_available",
        "turnover_proxy_available",
        "listing_board_available",
        "industry_pit_available",
        "traded_value",
        "turnover_5d",
        "turnover_20d",
        "turnover_60d",
        "turnover_rank_pct_5d",
        "turnover_rank_pct_20d",
        "turnover_rank_pct_60d",
        "traded_value_rank_pct",
        "revenue_growth",
        "profitability",
        "gross_margin",
        "operating_margin",
        "roe_or_quality",
        "cash_flow_quality",
        "debt_or_solvency_risk",
        "report_period",
        "source_date",
        "disclosure_date",
        "asof_date",
        "effective_date",
        "lag_policy",
        "fundamental_source_quality",
        "effective_asof_lag_days",
        "forward_return_as_rule",
        "not_live_rule",
        "diagnostic_only",
    ]
    return joined.reindex(columns=cols)


def _missingness_by_period(joined: pd.DataFrame) -> pd.DataFrame:
    fields = [
        "market_cap_available",
        "paid_in_capital_available",
        "average_traded_value_proxy_available",
        "turnover_proxy_available",
        "listing_board_available",
        "industry_pit_available",
        *FUNDAMENTAL_FIELDS,
    ]
    rows = []
    for period, start, end in PERIODS:
        subset = joined[(joined["signal_date"] >= pd.Timestamp(start)) & (joined["signal_date"] <= pd.Timestamp(end))]
        for field in fields:
            if field.endswith("_available"):
                available = subset[field].astype(bool) if field in subset else pd.Series([], dtype=bool)
            else:
                available = subset[field].notna() if field in subset else pd.Series([], dtype=bool)
            rows.append(
                {
                    "period": period,
                    "requested_start": start,
                    "requested_end": end,
                    "actual_start": subset["signal_date"].min() if not subset.empty else pd.NaT,
                    "actual_end": subset["signal_date"].max() if not subset.empty else pd.NaT,
                    "field": field,
                    "rows": int(len(subset)),
                    "available_rows": int(available.sum()) if len(subset) else 0,
                    "missing_rows": int(len(subset) - available.sum()) if len(subset) else 0,
                    "available_share": float(available.mean()) if len(subset) else 0.0,
                    "diagnostic_only": True,
                }
            )
    return pd.DataFrame(rows)


def _source_quality_matrix(joined: pd.DataFrame, fundamentals: pd.DataFrame) -> pd.DataFrame:
    rows = [
        ("market_cap", "blocked", 0, "market cap is not materialized in existing PIT contracts"),
        ("paid_in_capital", "blocked", 0, "paid-in capital is not materialized in existing PIT contracts"),
        ("average_traded_value", "diagnostic_proxy", int(joined["average_traded_value_proxy_available"].sum()), "uses traded_value on signal_date / attention_features only"),
        ("turnover", "diagnostic_proxy", int(joined["turnover_proxy_available"].sum()), "uses attention_features turnover windows"),
        ("listing_board", "blocked", 0, "listing board PIT field not present in vNext materialization"),
        ("industry", "blocked", 0, "industry PIT field not present in vNext materialization"),
        ("monthly_revenue_growth", "blocked", 0, "monthly revenue PIT disclosure contract not materialized"),
        ("quarterly_revenue_growth", "blocked_or_sparse", int(fundamentals["revenue_growth"].notna().sum()) if "revenue_growth" in fundamentals else 0, "revenue_growth column exists but source rows are empty in current package"),
        ("eps_or_operating_income_growth", "blocked", 0, "EPS / operating income growth not materialized"),
        ("profitability", "diagnostic_proxy", int(joined["profitability"].notna().sum()), "as-of join from fundamental_features; source_quality proxy"),
        ("gross_margin", "diagnostic_proxy", int(joined["gross_margin"].notna().sum()), "as-of join from fundamental_features; source_quality proxy"),
        ("operating_margin", "diagnostic_proxy", int(joined["operating_margin"].notna().sum()), "as-of join from fundamental_features; source_quality proxy"),
        ("net_margin", "blocked", 0, "net margin not materialized"),
        ("roe_roa", "blocked_or_sparse", int(joined["roe_or_quality"].notna().sum()), "ROE/ROA not populated in current package"),
        ("cash_flow_quality", "blocked_or_sparse", int(joined["cash_flow_quality"].notna().sum()), "cash flow quality not populated in current package"),
        ("debt_leverage_current_ratio", "blocked_or_sparse", int(joined["debt_or_solvency_risk"].notna().sum()), "debt/leverage/current ratio not populated in current package"),
        ("inventory_receivable_risk", "blocked", 0, "inventory/receivable risk not materialized"),
        ("forward_return_as_rule", "prohibited", 0, "forward returns cannot be used for Layer 1 rule construction"),
    ]
    return pd.DataFrame(
        [
            {
                "field_group": field,
                "source_tier": tier,
                "available_rows": rows_count,
                "source_quality_reason": reason,
                "usable_for_layer1_diagnostic": tier in {"diagnostic_proxy"},
                "usable_for_formal": False,
                "diagnostic_only": True,
            }
            for field, tier, rows_count, reason in rows
        ]
    )


def _blocked_proxy_fields(source_quality: pd.DataFrame) -> pd.DataFrame:
    out = source_quality.copy()
    out["status"] = out["source_tier"].map(
        lambda tier: "prohibited" if tier == "prohibited" else "blocked" if "blocked" in tier else "proxy"
    )
    out["proxy_available"] = out["status"].eq("proxy")
    return out.rename(columns={"field_group": "field_or_contract", "source_quality_reason": "blocked_reason"})[
        ["field_or_contract", "status", "source_tier", "proxy_available", "blocked_reason", "diagnostic_only"]
    ]


def _future_data_audit(joined: pd.DataFrame) -> pd.DataFrame:
    bad_effective = int((pd.to_datetime(joined["effective_date"], errors="coerce") > joined["signal_date"]).sum())
    return pd.DataFrame(
        [
            {
                "audit_item": "fundamental_effective_date_lte_signal_date",
                "status": "passed" if bad_effective == 0 else "failed",
                "future_data_violation_count": bad_effective,
                "note": "fundamental rows are merge_asof backward by ticker",
            },
            {
                "audit_item": "attention_trade_date_equals_signal_date",
                "status": "passed",
                "future_data_violation_count": 0,
                "note": "attention features are joined on exact signal_date",
            },
            {
                "audit_item": "forward_return_as_rule",
                "status": "passed",
                "future_data_violation_count": 0,
                "note": "no forward return columns are included in Layer 1 contracts",
            },
        ]
    )


def _readiness_json(
    pit_contract: pd.DataFrame,
    joined: pd.DataFrame,
    missingness: pd.DataFrame,
    source_quality: pd.DataFrame,
    blocked: pd.DataFrame,
    future_audit: pd.DataFrame,
) -> dict[str, Any]:
    future_count = int(future_audit["future_data_violation_count"].sum())
    diagnostic_available = source_quality[source_quality["source_tier"].eq("diagnostic_proxy")]
    ready = not pit_contract.empty and not diagnostic_available.empty and future_count == 0
    exact_full = bool(source_quality["source_tier"].eq("exact").all())
    blocked_any = bool(source_quality["source_tier"].astype(str).str.contains("blocked|sparse|prohibited", regex=True).any())
    layer1_coverage = "full" if exact_full else "partial" if ready else "blocked"
    return {
        "date": "2026-07-06",
        "task_id": TASK_ID,
        "owner": "BACKTEST_LAB Core/Data",
        "status": "partial_ready_layer1_diagnostic_fundamental_size_quality_sparse_proxy_limited" if ready else "blocked_funnel_layer1_diagnostic",
        "ready_for_funnel_layer1_event_diagnostic": bool(ready),
        "layer1_exact_coverage": layer1_coverage,
        "ready_for_layer2_diagnostic": bool(ready and not blocked_any),
        "ready_for_portfolio_like_diagnostic": False,
        "ready_for_strategy_replay": False,
        "ready_for_formal": False,
        "future_data_violation_count": future_count,
        "not_live_rule": True,
        "forward_returns_live_rule_usage": False,
        "candidate_join_rows": int(len(joined)),
        "pit_contract_rows": int(len(pit_contract)),
        "diagnostic_proxy_fields": diagnostic_available["field_group"].tolist(),
        "blocked_fields": blocked[blocked["status"].eq("blocked")]["field_or_contract"].tolist(),
        "proxy_fields": blocked[blocked["status"].eq("proxy")]["field_or_contract"].tolist(),
        "formal_model_changed": False,
        "trade_decision_changed": False,
        "active_in_trade_decision": False,
        "report_changed": False,
        "portfolio_replay_executed": False,
    }


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def _summary(readiness: dict[str, Any], blocked: pd.DataFrame) -> str:
    return "\n".join(
        [
            "# vNext Funnel Layer 1 Fundamental / Size / Quality Readiness",
            "",
            f"Status: {readiness['status']}",
            "",
            "Boundary: Layer 1 source/contract readiness only; no selector, no replay, no formal/report/trade change.",
            "",
            "Readiness:",
            f"- ready_for_funnel_layer1_event_diagnostic={str(readiness['ready_for_funnel_layer1_event_diagnostic']).lower()}",
            f"- layer1_exact_coverage={readiness['layer1_exact_coverage']}",
            f"- ready_for_layer2_diagnostic={str(readiness['ready_for_layer2_diagnostic']).lower()}",
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
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    manifest = build_funnel_layer1_readiness(
        materialization_dir=args.materialization_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
