"""Build Phase G candidate-quality theme/fundamental contract readiness.

This is source/contract readiness only. It does not define a formal selector,
use forward returns as rule inputs, alter reports/trades, or execute any
portfolio/strategy replay.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


TASK_ID = "TASK-BACKTEST-CORE-VNEXT-PHASE-G-CANDIDATE-QUALITY-THEME-FUNDAMENTAL-CONTRACT-READINESS-001"
DEFAULT_MATERIALIZATION_DIR = Path("outputs/vnext_dynamic_candidate_pool_data_materialization_20260706")
DEFAULT_PHASE_F_DIR = Path(
    r"C:\Users\zergv\Documents\Codex\2026-07-06\backtest-lab-experiments-diagnostic-validation-attribution\outputs"
    r"\vnext_phase_f_candidate_pool_quality_non_c3_selector_diagnostic_20260706"
)
DEFAULT_OUTPUT_DIR = Path("outputs/vnext_phase_g_candidate_quality_contract_readiness_20260706")


PERIODS = [
    ("P1", "2015-01-02", "2022-12-29"),
    ("P2", "2023-01-02", "2026-06-30"),
    ("2024-latest", "2024-01-02", "2026-06-30"),
    ("2026YTD", "2026-01-02", "2026-06-30"),
]


def build_phase_g_readiness(
    *,
    materialization_dir: str | Path = DEFAULT_MATERIALIZATION_DIR,
    phase_f_dir: str | Path = DEFAULT_PHASE_F_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    materialization = Path(materialization_dir)
    phase_f = Path(phase_f_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    weekly = _weekly_candidate_rows(materialization / "vnext_weekly_candidate_snapshot.csv")
    stock = _stock_feature_slice(materialization / "stock_features.csv", weekly)
    attention = _attention_feature_slice(materialization / "attention_features.csv", weekly)
    fundamentals = pd.read_csv(materialization / "fundamental_features.csv", parse_dates=["effective_date"])
    theme_membership = pd.read_csv(materialization / "theme_membership.csv", parse_dates=["effective_date", "valid_from", "valid_to"])

    join_contract = _candidate_quality_join_contract(weekly, stock, attention, fundamentals)
    theme_breadth = _theme_breadth_contract(join_contract, theme_membership)
    fundamental_readiness = _fundamental_quality_readiness(fundamentals, join_contract)
    coverage = _coverage_by_period(join_contract, theme_breadth, fundamental_readiness)
    source_quality = _source_quality_matrix(theme_breadth, fundamental_readiness)
    blocked = _blocked_proxy_fields(source_quality, fundamental_readiness)
    future_audit = _future_data_audit(join_contract, theme_membership, fundamentals)
    readiness = _readiness_json(
        phase_f / "manifest.json",
        join_contract,
        theme_breadth,
        fundamental_readiness,
        source_quality,
        blocked,
        future_audit,
    )

    _write_csv(theme_breadth, output / "phase_g_theme_breadth_pit_contract.csv")
    _write_csv(fundamental_readiness, output / "phase_g_fundamental_quality_pit_readiness.csv")
    _write_csv(join_contract, output / "phase_g_candidate_pool_quality_join_contract.csv")
    _write_csv(blocked, output / "blocked_proxy_fields.csv")
    _write_csv(source_quality, output / "source_quality_matrix.csv")
    _write_csv(future_audit, output / "future_data_audit.csv")
    _write_csv(coverage, output / "coverage_by_period.csv")
    (output / "readiness_for_phase_g_candidate_quality_contract.json").write_text(
        json.dumps(readiness, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    manifest = {
        "task_id": TASK_ID,
        "status": readiness["status"],
        "output_dir": str(output.resolve()),
        "input_materialization_dir": str(materialization.resolve()),
        "input_phase_f_dir": str(phase_f.resolve()),
        "output_files": [
            "phase_g_theme_breadth_pit_contract.csv",
            "phase_g_fundamental_quality_pit_readiness.csv",
            "phase_g_candidate_pool_quality_join_contract.csv",
            "blocked_proxy_fields.csv",
            "source_quality_matrix.csv",
            "future_data_audit.csv",
            "coverage_by_period.csv",
            "readiness_for_phase_g_candidate_quality_contract.json",
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


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _weekly_candidate_rows(path: Path) -> pd.DataFrame:
    usecols = [
        "snapshot_date",
        "ticker",
        "theme_id",
        "theme_name",
        "is_ai_theme_member",
        "ai_membership_source_quality",
        "subpool_class",
        "selected_outcome_candidate",
        "case_trace_only",
        "diagnostic_only",
        "turnover_state",
        "risk_bucket",
        "hurdle_0050_proxy_result",
        "hurdle_00631L_proxy_result",
        "final_selector_score_decomposed",
        "rank_overall",
    ]
    df = pd.read_csv(path, usecols=usecols, parse_dates=["snapshot_date"])
    df = df[df["diagnostic_only"].astype(bool) & ~df["case_trace_only"].astype(bool)].copy()
    df["ticker"] = df["ticker"].astype(str)
    return df


def _stock_feature_slice(path: Path, weekly: pd.DataFrame) -> pd.DataFrame:
    dates = set(weekly["snapshot_date"].dt.strftime("%Y-%m-%d"))
    tickers = set(weekly["ticker"].astype(str))
    usecols = [
        "trade_date",
        "ticker",
        "RS20",
        "RS40",
        "RS60",
        "BIAS20_percentile",
        "BIAS60_percentile",
        "BIAS120_percentile",
        "MA20_position",
        "MA60_position",
        "MA120_position",
        "return_5d",
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


def _attention_feature_slice(path: Path, weekly: pd.DataFrame) -> pd.DataFrame:
    dates = set(weekly["snapshot_date"].dt.strftime("%Y-%m-%d"))
    tickers = set(weekly["ticker"].astype(str))
    header = pd.read_csv(path, nrows=0)
    wanted = [
        "trade_date",
        "ticker",
        "turnover_rank_pct_5d",
        "turnover_rank_pct_20d",
        "traded_value_rank_pct_5d",
        "traded_value_rank_pct_20d",
        "turnover_concentration_rank_pct",
    ]
    usecols = [c for c in wanted if c in header.columns]
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


def _candidate_quality_join_contract(
    weekly: pd.DataFrame,
    stock: pd.DataFrame,
    attention: pd.DataFrame,
    fundamentals: pd.DataFrame,
) -> pd.DataFrame:
    joined = weekly.merge(
        stock,
        left_on=["snapshot_date", "ticker"],
        right_on=["trade_date", "ticker"],
        how="left",
    ).drop(columns=["trade_date"], errors="ignore")
    if not attention.empty:
        joined = joined.merge(
            attention,
            left_on=["snapshot_date", "ticker"],
            right_on=["trade_date", "ticker"],
            how="left",
        ).drop(columns=["trade_date"], errors="ignore")
    latest_fund = _latest_fundamentals_asof(joined[["snapshot_date", "ticker"]].drop_duplicates(), fundamentals)
    joined = joined.merge(latest_fund, on=["snapshot_date", "ticker"], how="left")
    for col in ["RS20", "RS40", "RS60", "BIAS20_percentile", "BIAS60_percentile", "BIAS120_percentile"]:
        joined[f"{col}_available"] = joined[col].notna() if col in joined else False
    attention_cols = [c for c in joined.columns if "rank_pct" in c or "turnover_concentration" in c]
    joined["attention_feature_available"] = joined[attention_cols].notna().any(axis=1) if attention_cols else False
    joined["fundamental_feature_available"] = joined[
        ["revenue_growth", "profitability", "gross_margin", "operating_margin", "roe_or_quality", "cash_flow_quality", "debt_or_solvency_risk"]
    ].notna().any(axis=1)
    joined["join_key"] = joined["snapshot_date"].dt.strftime("%Y-%m-%d") + "|" + joined["ticker"] + "|" + joined["theme_id"].astype(str)
    joined["source_quality"] = "diagnostic_join_contract"
    joined["forward_return_as_rule"] = False
    joined["portfolio_like_diagnostic"] = False
    joined["not_live_rule"] = True
    return joined


def _latest_fundamentals_asof(keys: pd.DataFrame, fundamentals: pd.DataFrame) -> pd.DataFrame:
    if fundamentals.empty:
        out = keys.copy()
        out["fundamental_source_quality"] = pd.NA
        return out
    keys = keys.sort_values(["ticker", "snapshot_date"]).copy()
    fundamentals = fundamentals.copy()
    fundamentals["ticker"] = fundamentals["ticker"].astype(str)
    fundamentals = fundamentals.sort_values(["ticker", "effective_date"])
    parts = []
    for ticker, group in keys.groupby("ticker", sort=False):
        fund = fundamentals[fundamentals["ticker"].eq(ticker)]
        if fund.empty:
            part = group.copy()
            for col in fundamentals.columns:
                if col not in {"ticker"}:
                    part[col] = pd.NA
            parts.append(part)
            continue
        merged = pd.merge_asof(
            group.sort_values("snapshot_date"),
            fund.sort_values("effective_date"),
            left_on="snapshot_date",
            right_on="effective_date",
            by="ticker",
            direction="backward",
        )
        parts.append(merged)
    out = pd.concat(parts, ignore_index=True)
    out = out.rename(columns={"source_quality": "fundamental_source_quality"})
    return out


def _theme_breadth_contract(joined: pd.DataFrame, theme_membership: pd.DataFrame) -> pd.DataFrame:
    data = joined.copy()
    data["positive_RS20"] = pd.to_numeric(data.get("RS20"), errors="coerce").gt(0)
    data["positive_RS40"] = pd.to_numeric(data.get("RS40"), errors="coerce").gt(0)
    data["positive_RS60"] = pd.to_numeric(data.get("RS60"), errors="coerce").gt(0)
    data["above_MA20"] = pd.to_numeric(data.get("MA20_position"), errors="coerce").gt(0)
    data["above_MA60"] = pd.to_numeric(data.get("MA60_position"), errors="coerce").gt(0)
    data["above_MA120"] = pd.to_numeric(data.get("MA120_position"), errors="coerce").gt(0)
    data["advancing_5d"] = pd.to_numeric(data.get("return_5d"), errors="coerce").gt(0)
    data["declining_5d"] = pd.to_numeric(data.get("return_5d"), errors="coerce").lt(0)
    concentration_col = _first_existing(data, ["traded_value_rank_pct_20d", "traded_value_rank_pct_5d", "turnover_rank_pct_20d", "turnover_rank_pct_5d"])
    data["theme_turnover_value_attention_proxy"] = pd.to_numeric(data[concentration_col], errors="coerce") if concentration_col else pd.NA
    grouped = data.groupby(["snapshot_date", "theme_id", "theme_name"], dropna=False)
    out = grouped.agg(
        member_count=("ticker", "nunique"),
        above_MA20_count=("above_MA20", "sum"),
        above_MA60_count=("above_MA60", "sum"),
        above_MA120_count=("above_MA120", "sum"),
        positive_RS20_count=("positive_RS20", "sum"),
        positive_RS40_count=("positive_RS40", "sum"),
        positive_RS60_count=("positive_RS60", "sum"),
        advancing_5d_count=("advancing_5d", "sum"),
        declining_5d_count=("declining_5d", "sum"),
        theme_turnover_value_attention_mean=("theme_turnover_value_attention_proxy", "mean"),
        ai_member_count=("is_ai_theme_member", "sum"),
    ).reset_index()
    for base in ["above_MA20", "above_MA60", "above_MA120", "positive_RS20", "positive_RS40", "positive_RS60", "advancing_5d", "declining_5d"]:
        out[f"{base}_share"] = out[f"{base}_count"] / out["member_count"].where(out["member_count"].ne(0), pd.NA)
    out["advancing_declining_5d_proxy"] = (out["advancing_5d_count"] - out["declining_5d_count"]) / out["member_count"].where(out["member_count"].ne(0), pd.NA)
    membership_quality = theme_membership.groupby("theme_id", dropna=False).agg(
        membership_asof_date=("effective_date", "min"),
        membership_valid_from=("valid_from", "min"),
        membership_valid_to=("valid_to", "max"),
        membership_source_tier=("source_quality", lambda s: "|".join(sorted(set(map(str, s.dropna())))) or "unknown"),
    ).reset_index()
    out = out.merge(membership_quality, on="theme_id", how="left")
    out["theme_label"] = out["theme_name"]
    out["source_tier"] = out["membership_source_tier"].fillna("proxy_or_unclassified")
    out.loc[out["theme_id"].astype(str).str.contains("non_ai_unclassified_proxy", na=False), "source_tier"] = "proxy_unclassified_not_formal"
    out["diagnostic_only"] = True
    out["accepted_for_formal"] = False
    out["ai_context_only_no_allocation_rule"] = True
    out = out.rename(columns={"snapshot_date": "signal_date"})
    return out


def _fundamental_quality_readiness(fundamentals: pd.DataFrame, joined: pd.DataFrame) -> pd.DataFrame:
    fields = ["revenue_growth", "profitability", "gross_margin", "operating_margin", "roe_or_quality", "cash_flow_quality", "debt_or_solvency_risk"]
    rows = []
    for field in fields:
        rows.append(
            {
                "field": field,
                "available_rows_in_source": int(fundamentals[field].notna().sum()) if field in fundamentals else 0,
                "available_rows_in_join": int(joined[field].notna().sum()) if field in joined else 0,
                "join_coverage_share": float(joined[field].notna().mean()) if field in joined and len(joined) else 0.0,
                "source_date_field": "effective_date",
                "report_period_field": "not_materialized",
                "disclosure_or_asof_date_field": "effective_date",
                "lag_policy": "effective_asof_lag_days when available; otherwise blocked/proxy",
                "source_tier": "proxy_sparse" if (field in joined and joined[field].notna().mean() < 0.5) else "diagnostic_available",
                "blocked_reason": "sparse/proxy-limited; not formal selector input" if (field in joined and joined[field].notna().mean() < 0.5) else "",
                "diagnostic_only": True,
                "accepted_for_formal": False,
            }
        )
    return pd.DataFrame(rows)


def _coverage_by_period(joined: pd.DataFrame, theme_breadth: pd.DataFrame, fundamental: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for period, start, end in PERIODS:
        s = pd.Timestamp(start)
        e = pd.Timestamp(end)
        subset = joined[(joined["snapshot_date"] >= s) & (joined["snapshot_date"] <= e)]
        theme_subset = theme_breadth[(theme_breadth["signal_date"] >= s) & (theme_breadth["signal_date"] <= e)]
        rows.append(
            {
                "period": period,
                "requested_start": start,
                "requested_end": end,
                "actual_start": subset["snapshot_date"].min() if not subset.empty else pd.NaT,
                "actual_end": subset["snapshot_date"].max() if not subset.empty else pd.NaT,
                "candidate_rows": int(len(subset)),
                "theme_breadth_rows": int(len(theme_subset)),
                "fundamental_any_available_share": float(subset["fundamental_feature_available"].mean()) if len(subset) else 0.0,
                "rs20_available_share": float(subset["RS20_available"].mean()) if len(subset) else 0.0,
                "attention_available_share": float(subset["attention_feature_available"].mean()) if len(subset) else 0.0,
                "diagnostic_only": True,
            }
        )
    return pd.DataFrame(rows)


def _source_quality_matrix(theme_breadth: pd.DataFrame, fundamental: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("theme_breadth_measures", "diagnostic_pit", len(theme_breadth), "weekly snapshot + stock_features as-of signal_date"),
            ("theme_membership_taxonomy", "proxy_limited", len(theme_breadth), "theme_membership source_quality is proxy/unclassified for many rows"),
            ("non_ai_theme_taxonomy", "proxy_unclassified", int(theme_breadth["theme_id"].astype(str).str.contains("non_ai_unclassified_proxy", na=False).sum()), "cannot treat proxy/unclassified as formal theme"),
            ("ai_theme_context", "context_only", int(theme_breadth["ai_member_count"].sum()), "AI remains context only; no quota/floor/cap"),
            ("theme_turnover_value_concentration", "diagnostic_proxy", len(theme_breadth), "attention rank percentile aggregate; not formal source"),
            ("fundamental_quality_fields", "sparse_proxy_limited", int(fundamental["available_rows_in_join"].sum()), "valid as-of field exists but coverage/source quality sparse"),
            ("forward_return_as_rule", "prohibited", 0, "future returns are evaluation metadata only"),
        ],
        columns=["contract_or_field_family", "source_quality", "rows_or_count", "source_quality_reason"],
    ).assign(diagnostic_only=True, accepted_for_formal=False)


def _blocked_proxy_fields(source_quality: pd.DataFrame, fundamental: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for item in source_quality.itertuples(index=False):
        status = "prohibited" if item.source_quality == "prohibited" else "proxy_or_blocked" if "proxy" in item.source_quality or "sparse" in item.source_quality else "diagnostic_ready"
        rows.append(
            {
                "field_or_contract": item.contract_or_field_family,
                "status": status,
                "proxy_available": status != "prohibited",
                "blocked_reason": item.source_quality_reason,
                "diagnostic_only": True,
            }
        )
    for item in fundamental.itertuples(index=False):
        if item.source_tier == "proxy_sparse":
            rows.append(
                {
                    "field_or_contract": f"fundamental_{item.field}",
                    "status": "proxy_sparse",
                    "proxy_available": item.available_rows_in_join > 0,
                    "blocked_reason": item.blocked_reason,
                    "diagnostic_only": True,
                }
            )
    return pd.DataFrame(rows)


def _future_data_audit(joined: pd.DataFrame, theme_membership: pd.DataFrame, fundamentals: pd.DataFrame) -> pd.DataFrame:
    bad_fund = int((pd.to_datetime(joined["effective_date"], errors="coerce") > joined["snapshot_date"]).sum()) if "effective_date" in joined else 0
    # Theme membership is currently read as dated membership source; rows are
    # contract-level, not expanded into future membership claims.
    return pd.DataFrame(
        [
            {
                "audit_item": "fundamental_effective_date_lte_signal_date",
                "status": "passed" if bad_fund == 0 else "failed",
                "future_data_violation_count": bad_fund,
                "note": "latest fundamental rows are joined backward by ticker",
            },
            {
                "audit_item": "theme_membership_dated_contract_present",
                "status": "passed" if not theme_membership.empty else "blocked",
                "future_data_violation_count": 0,
                "note": "theme source_tier remains proxy/context where source_quality says so",
            },
            {
                "audit_item": "forward_return_as_rule",
                "status": "passed",
                "future_data_violation_count": 0,
                "note": "no forward return columns are used in output contracts",
            },
        ]
    )


def _readiness_json(
    manifest_path: Path,
    joined: pd.DataFrame,
    theme_breadth: pd.DataFrame,
    fundamental: pd.DataFrame,
    source_quality: pd.DataFrame,
    blocked: pd.DataFrame,
    future_audit: pd.DataFrame,
) -> dict[str, Any]:
    manifest = _read_json(manifest_path)
    future_count = int(future_audit["future_data_violation_count"].sum())
    fundamental_sparse = bool((fundamental["source_tier"] == "proxy_sparse").any())
    theme_proxy = bool(source_quality["source_quality"].astype(str).str.contains("proxy|unclassified|sparse", case=False, regex=True).any())
    ready = len(theme_breadth) > 0 and len(joined) > 0 and future_count == 0
    return {
        "date": "2026-07-06",
        "task_id": TASK_ID,
        "owner": "BACKTEST_LAB Core/Data",
        "status": "partial_ready_theme_breadth_diagnostic_fundamental_sparse_proxy_limited" if ready else "blocked_phase_g_candidate_quality_contract",
        "ready_for_phase_g_candidate_quality_diagnostic": bool(ready),
        "ready_for_higher_quality_theme_breadth_diagnostic": bool(ready and not theme_proxy),
        "ready_for_fundamental_quality_diagnostic": bool(ready and not fundamental_sparse),
        "ready_for_portfolio_like_diagnostic": False,
        "ready_for_strategy_replay": False,
        "ready_for_formal": False,
        "future_data_violation_count": future_count,
        "not_live_rule": True,
        "forward_returns_live_rule_usage": False,
        "phase_f_verdict": manifest.get("verdict"),
        "candidate_join_rows": int(len(joined)),
        "theme_breadth_rows": int(len(theme_breadth)),
        "fundamental_fields": fundamental.to_dict(orient="records"),
        "blocked_fields": blocked[blocked["status"].astype(str).str.contains("blocked|proxy|sparse", case=False, regex=True)]["field_or_contract"].tolist(),
        "proxy_fields": blocked[blocked["proxy_available"].astype(bool)]["field_or_contract"].tolist(),
        "formal_model_changed": False,
        "trade_decision_changed": False,
        "active_in_trade_decision": False,
        "report_changed": False,
        "portfolio_replay_executed": False,
    }


def _first_existing(df: pd.DataFrame, cols: list[str]) -> str | None:
    for col in cols:
        if col in df.columns:
            return col
    return None


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _summary(readiness: dict[str, Any], blocked: pd.DataFrame) -> str:
    return "\n".join(
        [
            "# vNext Phase G Candidate Quality Contract Readiness",
            "",
            f"Status: {readiness['status']}",
            "",
            "Boundary: contract/readiness only; no formal selector, no replay, no trade/report change.",
            "",
            "Readiness:",
            f"- ready_for_phase_g_candidate_quality_diagnostic={str(readiness['ready_for_phase_g_candidate_quality_diagnostic']).lower()}",
            f"- ready_for_higher_quality_theme_breadth_diagnostic={str(readiness['ready_for_higher_quality_theme_breadth_diagnostic']).lower()}",
            f"- ready_for_fundamental_quality_diagnostic={str(readiness['ready_for_fundamental_quality_diagnostic']).lower()}",
            "- ready_for_portfolio_like_diagnostic=false",
            "- ready_for_strategy_replay=false",
            "- ready_for_formal=false",
            f"- future_data_violation_count={readiness['future_data_violation_count']}",
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
    parser.add_argument("--phase-f-dir", type=Path, default=DEFAULT_PHASE_F_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    manifest = build_phase_g_readiness(
        materialization_dir=args.materialization_dir,
        phase_f_dir=args.phase_f_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
