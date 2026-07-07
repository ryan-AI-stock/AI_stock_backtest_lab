"""Build RS-window evaluation join for Layer2 compact diagnostics.

This package adds PIT RS window features to the compact Layer1 evaluation join.
Forward returns remain evaluation metadata only and are never live rule inputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from backtest_lab.vnext_layer1_compact_candidate_quality_evaluation_join import (
    BENCHMARKS,
    DEFAULT_DATA_DIR,
    DEFAULT_LAYER1_DIR,
    HORIZONS,
    _attach_decile_labels,
    _attach_forward_returns,
    _attach_outcome_labels,
    _blocked_rows,
    _coverage_by_period,
    _evaluation_summary,
    _future_audit,
    _read_benchmark_prices,
    _read_calendar,
    _read_json,
    _read_layer1_contract,
    _read_stock_prices,
    _write_csv,
)


TASK_ID = "TASK-BACKTEST-CORE-VNEXT-LAYER2-COMPACT-RS-WINDOW-EVALUATION-JOIN-001"
DEFAULT_OUTPUT_DIR = Path("outputs/vnext_layer2_compact_rs_window_evaluation_join_20260707")
RS_WINDOWS = [5, 10, 20, 40, 60]


def build_join(
    *,
    layer1_dir: str | Path = DEFAULT_LAYER1_DIR,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    layer1 = Path(layer1_dir)
    data = Path(data_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    layer1_readiness = _read_json(layer1 / "readiness_for_layer1_compact_reduced_universe_interim_contract.json")
    events = _read_layer1_contract(layer1 / "layer1_compact_reduced_universe_interim_contract.csv")
    calendar = _read_calendar(data / "trading_calendar.csv")
    stock_prices = _read_stock_prices(data / "daily_market_features.csv", events["ticker"].unique())
    benchmark_prices = _read_benchmark_prices(data / "benchmark_features.csv")
    rs_features = _read_rs_features(data / "stock_features.csv", events)

    joined = _attach_forward_returns(events, calendar, stock_prices, benchmark_prices)
    joined = joined.merge(rs_features, on=["snapshot_date", "ticker"], how="left")
    joined = _attach_rs_context(joined)
    joined = _attach_outcome_labels(joined)
    joined = _attach_decile_labels(joined)

    blocked_rows = _blocked_rows(joined)
    rs_missingness = _rs_missingness(joined)
    coverage = _coverage_by_period(joined)
    summary = _evaluation_summary(joined)
    future_audit = _future_audit(joined)
    feature_audit = _feature_audit(joined)
    readiness = _readiness(layer1_readiness, joined, blocked_rows, future_audit)

    _write_csv(joined, output / "layer2_compact_rs_window_evaluation_join.csv")
    _write_csv(joined.head(1000), output / "layer2_compact_rs_window_evaluation_join_sample.csv")
    (output / ".gitignore").write_text(
        "layer2_compact_rs_window_evaluation_join.csv\n",
        encoding="utf-8",
    )
    _write_csv(blocked_rows, output / "layer2_compact_rs_window_blocked_latest_rows.csv")
    _write_csv(rs_missingness, output / "layer2_compact_rs_window_missingness_by_period.csv")
    _write_csv(coverage, output / "layer2_compact_rs_window_requested_vs_actual_coverage.csv")
    _write_csv(summary, output / "layer2_compact_rs_window_evaluation_summary.csv")
    _write_csv(feature_audit, output / "layer2_compact_rs_window_feature_source_audit.csv")
    _write_csv(future_audit, output / "layer2_compact_rs_window_future_data_audit.csv")
    (output / "readiness_for_layer2_compact_rs_window_evaluation_join.json").write_text(
        json.dumps(readiness, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    manifest = {
        "task_id": TASK_ID,
        "status": readiness["status"],
        "output_dir": str(output.resolve()),
        "input_layer1_dir": str(layer1.resolve()),
        "input_data_dir": str(data.resolve()),
        "output_files": [
            "layer2_compact_rs_window_evaluation_join.csv",
            "layer2_compact_rs_window_evaluation_join_sample.csv",
            "layer2_compact_rs_window_blocked_latest_rows.csv",
            "layer2_compact_rs_window_missingness_by_period.csv",
            "layer2_compact_rs_window_requested_vs_actual_coverage.csv",
            "layer2_compact_rs_window_evaluation_summary.csv",
            "layer2_compact_rs_window_feature_source_audit.csv",
            "layer2_compact_rs_window_future_data_audit.csv",
            "readiness_for_layer2_compact_rs_window_evaluation_join.json",
            "manifest.json",
            "final_summary_zh.md",
        ],
        "large_local_files_not_tracked": ["layer2_compact_rs_window_evaluation_join.csv"],
        "large_local_file_policy": "full RS evaluation join is retained in local output path; Git tracks sample/readiness/audit files only",
        "formal_model_changed": False,
        "trade_decision_changed": False,
        "active_in_trade_decision": False,
        "report_changed": False,
        "portfolio_replay_executed": False,
        "ready_for_strategy_replay": False,
        "not_live_rule": True,
        "forward_returns_live_rule_usage": False,
        "candidate_forward_return_diagnostic_executed": False,
        "diagnostic_only": True,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "final_summary_zh.md").write_text(_summary(readiness), encoding="utf-8")
    return manifest


def _read_rs_features(path: Path, events: pd.DataFrame) -> pd.DataFrame:
    pairs = events[["snapshot_date", "ticker"]].drop_duplicates().copy()
    pairs["snapshot_date"] = pd.to_datetime(pairs["snapshot_date"])
    ticker_set = set(pairs["ticker"].astype(str))
    date_set = set(pairs["snapshot_date"].dt.strftime("%Y-%m-%d"))
    usecols = [
        "trade_date",
        "ticker",
        "RS5",
        "RS10",
        "RS20",
        "RS40",
        "RS60",
        "excess_return_vs_00631L_5d",
        "excess_return_vs_00631L_10d",
        "excess_return_vs_00631L_20d",
        "excess_return_vs_00631L_40d",
        "excess_return_vs_00631L_60d",
    ]
    chunks = []
    for chunk in pd.read_csv(path, usecols=usecols, dtype={"ticker": str}, chunksize=500_000):
        chunk = chunk[chunk["ticker"].isin(ticker_set)].copy()
        if chunk.empty:
            continue
        chunk = chunk[chunk["trade_date"].isin(date_set)].copy()
        if chunk.empty:
            continue
        chunk["snapshot_date"] = pd.to_datetime(chunk["trade_date"])
        for col in usecols:
            if col not in {"trade_date", "ticker"}:
                chunk[col] = pd.to_numeric(chunk[col], errors="coerce")
        chunks.append(chunk.drop(columns=["trade_date"]))
    if not chunks:
        out = pairs.copy()
    else:
        out = pd.concat(chunks, ignore_index=True)
        out = pairs.merge(out, on=["snapshot_date", "ticker"], how="left")

    out = out.rename(
        columns={
            "excess_return_vs_00631L_5d": "rel_return_vs_00631L_5d",
            "excess_return_vs_00631L_10d": "rel_return_vs_00631L_10d",
            "excess_return_vs_00631L_20d": "rel_return_vs_00631L_20d",
            "excess_return_vs_00631L_40d": "rel_return_vs_00631L_40d",
            "excess_return_vs_00631L_60d": "rel_return_vs_00631L_60d",
        }
    )
    out["RS30_proxy"] = out[["RS20", "RS40"]].mean(axis=1)
    out["RS30_source_quality"] = "proxy_midpoint_rs20_rs40"
    for window in RS_WINDOWS:
        out[f"RS{window}_source_quality"] = "exact_from_stock_features_pit"
    out["RS40_source_quality"] = "exact_from_stock_features_pit_optional"
    return out


def _attach_rs_context(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["rs5_minus_rs10"] = out["RS5"] - out["RS10"]
    out["rs10_minus_rs20"] = out["RS10"] - out["RS20"]
    out["rs20_minus_rs60"] = out["RS20"] - out["RS60"]
    out["rs_short_acceleration_flag"] = out["rs5_minus_rs10"].gt(0) & out["rs10_minus_rs20"].gt(0)
    out["rs_short_deterioration_flag"] = out["rs5_minus_rs10"].lt(0) & out["rs10_minus_rs20"].lt(0)
    out["rs20_primary_positive_flag"] = out["RS20"].gt(0)
    out["rs30_proxy_positive_flag"] = out["RS30_proxy"].gt(0)
    out["rs60_medium_context_positive_flag"] = out["RS60"].gt(0)

    rs60_rank = out.groupby("snapshot_date")["RS60"].rank(pct=True, method="average")
    out["rs60_pctile_by_week"] = rs60_rank
    out["rs60_top20_by_week"] = rs60_rank.ge(0.80)
    out["rs60_top10_by_week"] = rs60_rank.ge(0.90)
    out["rs60_high_short_rs_weakening_exhaustion_context"] = (
        out["rs60_top20_by_week"] & out["rs_short_deterioration_flag"]
    )
    out["rs_window_context_source_quality"] = "diagnostic_pit_from_stock_features_no_forward_return_rule"
    out["rs30_exact_available"] = False
    return out


def _rs_missingness(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    period_map = {
        "ALL": (None, None),
        "P1": ("2015-01-02", "2022-12-29"),
        "P2": ("2023-01-02", "2026-06-30"),
        "2024_latest": ("2024-01-02", "2026-06-30"),
        "2026YTD": ("2026-01-02", "2026-06-30"),
    }
    feature_cols = ["RS5", "RS10", "RS20", "RS30_proxy", "RS40", "RS60"]
    for period, (start, end) in period_map.items():
        mask = pd.Series(True, index=df.index)
        if start:
            mask &= df["snapshot_date"].ge(pd.Timestamp(start))
        if end:
            mask &= df["snapshot_date"].le(pd.Timestamp(end))
        sub = df[mask]
        for col in feature_cols:
            rows.append(
                {
                    "period": period,
                    "feature": col,
                    "rows": int(len(sub)),
                    "available_rows": int(sub[col].notna().sum()) if col in sub else 0,
                    "missing_rows": int(sub[col].isna().sum()) if col in sub else int(len(sub)),
                    "available_share": float(sub[col].notna().mean()) if len(sub) and col in sub else 0.0,
                    "source_quality": "proxy_midpoint_rs20_rs40" if col == "RS30_proxy" else "exact_from_stock_features_pit",
                }
            )
    return pd.DataFrame(rows)


def _feature_audit(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("RS5_RS10_RS20_RS40_RS60", "exact_from_stock_features_pit", int(df["RS20"].notna().sum()), "as-of snapshot_date stock_features, vs 0050"),
            ("RS30_proxy", "proxy_midpoint_rs20_rs40", int(df["RS30_proxy"].notna().sum()), "RS30 exact unavailable; midpoint of RS20 and RS40 for diagnostic only"),
            ("rs_acceleration_deterioration_flags", "diagnostic_derived_from_pit_rs", int(df["rs_short_deterioration_flag"].notna().sum()), "no forward returns used"),
            ("rs60_high_short_rs_weakening_exhaustion_context", "diagnostic_derived_from_pit_rs", int(df["rs60_high_short_rs_weakening_exhaustion_context"].notna().sum()), "context only, not live rule"),
            ("forward_returns", "evaluation_metadata_only", int(df["forward_eval_available_20d"].sum()), "evaluation labels only"),
        ],
        columns=["field_group", "source_quality", "available_rows", "note"],
    )


def _readiness(
    layer1_readiness: dict[str, Any],
    df: pd.DataFrame,
    blocked_rows: pd.DataFrame,
    future_audit: pd.DataFrame,
) -> dict[str, Any]:
    future_count = int(future_audit["future_data_violation_count"].max())
    rs20_share = float(df["RS20"].notna().mean())
    rs60_share = float(df["RS60"].notna().mean())
    eval_20_share = float(df["forward_eval_available_20d"].mean())
    return {
        "task_id": TASK_ID,
        "status": "layer2_compact_rs_window_evaluation_join_ready_for_experiments_intake",
        "diagnostic_only": True,
        "evaluation_metadata_only": True,
        "input_layer1_status": layer1_readiness.get("status", ""),
        "rows": int(len(df)),
        "weekly_snapshot_count": int(df["snapshot_date"].nunique()),
        "unique_ticker_count": int(df["ticker"].nunique()),
        "rs20_available_share": rs20_share,
        "rs60_available_share": rs60_share,
        "rs30_exact_available": False,
        "rs30_proxy_available_share": float(df["RS30_proxy"].notna().mean()),
        "forward_eval_available_share_20d": eval_20_share,
        "blocked_evaluation_rows": int(len(blocked_rows)),
        "ready_for_layer2_compact_rs_capital_interaction_diagnostic": (
            future_count == 0 and rs20_share > 0.80 and rs60_share > 0.80 and eval_20_share > 0.80
        ),
        "ready_for_experiments_intake": True,
        "ready_for_formal": False,
        "ready_for_strategy_replay": False,
        "portfolio_replay_executed": False,
        "candidate_forward_return_diagnostic_executed": False,
        "future_data_violation_count": future_count,
        "formal_model_changed": False,
        "trade_decision_changed": False,
        "active_in_trade_decision": False,
        "report_changed": False,
        "not_live_rule": True,
        "forward_returns_live_rule_usage": False,
    }


def _summary(readiness: dict[str, Any]) -> str:
    return f"""# Layer2 compact RS-window evaluation join

## Verdict
- status={readiness["status"]}
- rows={readiness["rows"]}
- weekly_snapshot_count={readiness["weekly_snapshot_count"]}
- unique_ticker_count={readiness["unique_ticker_count"]}
- rs20_available_share={readiness["rs20_available_share"]}
- rs60_available_share={readiness["rs60_available_share"]}
- rs30_exact_available=false
- rs30_proxy_available_share={readiness["rs30_proxy_available_share"]}
- forward_eval_available_share_20d={readiness["forward_eval_available_share_20d"]}
- blocked_evaluation_rows={readiness["blocked_evaluation_rows"]}
- ready_for_layer2_compact_rs_capital_interaction_diagnostic={str(readiness["ready_for_layer2_compact_rs_capital_interaction_diagnostic"]).lower()}
- ready_for_experiments_intake=true
- ready_for_formal=false
- portfolio_replay_executed=false

## Plain Summary
This package adds PIT RS5/10/20/40/60, RS30 proxy, short-window acceleration/deterioration, and RS60-high short-RS weakening context to the compact Layer1 evaluation join. RS30 is explicitly proxy. Forward returns remain evaluation_metadata_only and are not rule inputs.

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
    parser.add_argument("--layer1-dir", default=str(DEFAULT_LAYER1_DIR))
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()
    manifest = build_join(layer1_dir=args.layer1_dir, data_dir=args.data_dir, output_dir=args.output_dir)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
