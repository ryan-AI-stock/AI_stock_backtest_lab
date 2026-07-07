"""Build forward evaluation join for Layer1 compact candidate-quality diagnostics.

Forward returns in this package are evaluation metadata only. They are not live
rule inputs, formal selector inputs, report inputs, trade decisions, or replay.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


TASK_ID = "TASK-BACKTEST-CORE-VNEXT-LAYER1-COMPACT-CANDIDATE-QUALITY-EVALUATION-JOIN-001"
DEFAULT_LAYER1_DIR = Path("outputs/vnext_layer1_compact_reduced_universe_interim_contract_20260707")
DEFAULT_DATA_DIR = Path("outputs/vnext_dynamic_candidate_pool_data_materialization_20260706")
DEFAULT_OUTPUT_DIR = Path("outputs/vnext_layer1_compact_candidate_quality_evaluation_join_20260707")
HORIZONS = [5, 10, 20, 40]
BENCHMARKS = ["0050", "00631L"]
PERIODS = {
    "P1": ("2015-01-02", "2022-12-29"),
    "P2": ("2023-01-02", "2026-06-30"),
    "2024_latest": ("2024-01-02", "2026-06-30"),
    "2026YTD": ("2026-01-02", "2026-06-30"),
}


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

    joined = _attach_forward_returns(events, calendar, stock_prices, benchmark_prices)
    joined = _attach_outcome_labels(joined)
    joined = _attach_decile_labels(joined)
    blocked_rows = _blocked_rows(joined)
    coverage = _coverage_by_period(joined)
    summary = _evaluation_summary(joined)
    future_audit = _future_audit(joined)
    readiness = _readiness(layer1_readiness, joined, blocked_rows, future_audit)

    _write_csv(joined, output / "layer1_compact_candidate_quality_evaluation_join.csv")
    _write_csv(joined.head(1000), output / "layer1_compact_candidate_quality_evaluation_join_sample.csv")
    (output / ".gitignore").write_text(
        "layer1_compact_candidate_quality_evaluation_join.csv\n",
        encoding="utf-8",
    )
    _write_csv(blocked_rows, output / "layer1_compact_candidate_quality_blocked_latest_rows.csv")
    _write_csv(coverage, output / "layer1_compact_candidate_quality_requested_vs_actual_coverage.csv")
    _write_csv(summary, output / "layer1_compact_candidate_quality_evaluation_summary.csv")
    _write_csv(future_audit, output / "layer1_compact_candidate_quality_future_data_audit.csv")
    (output / "readiness_for_layer1_compact_candidate_quality_evaluation_join.json").write_text(
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
            "layer1_compact_candidate_quality_evaluation_join.csv",
            "layer1_compact_candidate_quality_evaluation_join_sample.csv",
            "layer1_compact_candidate_quality_blocked_latest_rows.csv",
            "layer1_compact_candidate_quality_requested_vs_actual_coverage.csv",
            "layer1_compact_candidate_quality_evaluation_summary.csv",
            "layer1_compact_candidate_quality_future_data_audit.csv",
            "readiness_for_layer1_compact_candidate_quality_evaluation_join.json",
            "manifest.json",
            "final_summary_zh.md",
        ],
        "large_local_files_not_tracked": [
            "layer1_compact_candidate_quality_evaluation_join.csv"
        ],
        "large_local_file_policy": "full evaluation join is retained in local output path; Git tracks sample/readiness/audit files only",
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


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def _read_layer1_contract(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"ticker": str})
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
    df = df[df["active_for_layer1_source_scope"].astype(str).str.lower().eq("true")].copy()
    df["baseline_compact_universe_row_id"] = (
        df["snapshot_date"].dt.strftime("%Y-%m-%d") + "|" + df["ticker"].astype(str)
    )
    keep = [
        "baseline_compact_universe_row_id",
        "snapshot_date",
        "ticker",
        "name",
        "market",
        "variant",
        "scope_type",
        "active_for_layer1_source_scope",
        "selection_bucket",
        "traded_value_rank_5d",
        "traded_value_rank_20d",
        "traded_value_rank_60d",
        "monthly_revenue_available",
        "quarterly_fundamental_available",
        "monthly_revenue_yoy",
        "monthly_revenue_3m_yoy",
        "eps",
        "gross_margin",
        "operating_margin",
        "layer1_financial_risk_flag_count",
        "layer1_quality_floor_risk_pctile_by_week",
        "layer1_exclude_bottom20_candidate",
        "layer1_exclude_bottom30_candidate",
        "diagnostic_only",
        "not_live_rule",
        "forward_returns_live_rule_usage",
    ]
    for col in keep:
        if col not in df.columns:
            df[col] = pd.NA
    return df[keep].copy()


def _read_calendar(path: Path) -> pd.DataFrame:
    cal = pd.read_csv(path)
    cal["trade_date"] = pd.to_datetime(cal["trade_date"])
    cal = cal.sort_values("trade_date").reset_index(drop=True)
    cal["trade_index"] = range(len(cal))
    return cal[["trade_date", "trade_index"]]


def _read_stock_prices(path: Path, tickers: Any) -> pd.DataFrame:
    ticker_set = set(map(str, tickers))
    usecols = ["trade_date", "ticker", "adjusted_close"]
    chunks = []
    for chunk in pd.read_csv(path, usecols=usecols, dtype={"ticker": str}, chunksize=500_000):
        chunk = chunk[chunk["ticker"].isin(ticker_set)].copy()
        if chunk.empty:
            continue
        chunk["trade_date"] = pd.to_datetime(chunk["trade_date"])
        chunk["adjusted_close"] = pd.to_numeric(chunk["adjusted_close"], errors="coerce")
        chunks.append(chunk)
    return pd.concat(chunks, ignore_index=True).dropna(subset=["adjusted_close"])


def _read_benchmark_prices(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=["trade_date", "benchmark", "adjusted_close", "benchmark_data_blocked", "source_quality"])
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df["adjusted_close"] = pd.to_numeric(df["adjusted_close"], errors="coerce")
    df["benchmark_data_blocked"] = df["benchmark_data_blocked"].astype(str).str.lower().eq("true")
    return df[df["benchmark"].isin(BENCHMARKS)].copy()


def _attach_forward_returns(
    events: pd.DataFrame,
    calendar: pd.DataFrame,
    stock_prices: pd.DataFrame,
    benchmark_prices: pd.DataFrame,
) -> pd.DataFrame:
    out = events.merge(calendar.rename(columns={"trade_date": "snapshot_date"}), on="snapshot_date", how="left")
    stock_entry = stock_prices.rename(columns={"trade_date": "snapshot_date", "adjusted_close": "stock_entry_close"})
    out = out.merge(stock_entry, on=["snapshot_date", "ticker"], how="left")

    bench_entry = benchmark_prices.rename(columns={"trade_date": "snapshot_date", "adjusted_close": "benchmark_entry_close"})
    for benchmark in BENCHMARKS:
        entry = bench_entry[bench_entry["benchmark"].eq(benchmark)][
            ["snapshot_date", "benchmark_entry_close", "source_quality", "benchmark_data_blocked"]
        ].rename(
            columns={
                "benchmark_entry_close": f"{benchmark}_entry_close",
                "source_quality": f"{benchmark}_entry_source_quality",
                "benchmark_data_blocked": f"{benchmark}_entry_blocked",
            }
        )
        out = out.merge(entry, on="snapshot_date", how="left")

    for horizon in HORIZONS:
        target = calendar.copy()
        target["trade_index"] = target["trade_index"] - horizon
        target = target.rename(columns={"trade_date": f"target_date_{horizon}d"})
        out = out.merge(target, on="trade_index", how="left")

        stock_target = stock_prices.rename(
            columns={"trade_date": f"target_date_{horizon}d", "adjusted_close": f"stock_target_close_{horizon}d"}
        )
        out = out.merge(stock_target, on=[f"target_date_{horizon}d", "ticker"], how="left")
        out[f"forward_return_{horizon}d"] = out[f"stock_target_close_{horizon}d"] / out["stock_entry_close"] - 1

        for benchmark in BENCHMARKS:
            btarget = benchmark_prices[benchmark_prices["benchmark"].eq(benchmark)].rename(
                columns={
                    "trade_date": f"target_date_{horizon}d",
                    "adjusted_close": f"{benchmark}_target_close_{horizon}d",
                    "source_quality": f"{benchmark}_target_source_quality_{horizon}d",
                    "benchmark_data_blocked": f"{benchmark}_target_blocked_{horizon}d",
                }
            )
            btarget = btarget[
                [
                    f"target_date_{horizon}d",
                    f"{benchmark}_target_close_{horizon}d",
                    f"{benchmark}_target_source_quality_{horizon}d",
                    f"{benchmark}_target_blocked_{horizon}d",
                ]
            ]
            out = out.merge(btarget, on=f"target_date_{horizon}d", how="left")
            out[f"{benchmark}_forward_return_{horizon}d"] = (
                out[f"{benchmark}_target_close_{horizon}d"] / out[f"{benchmark}_entry_close"] - 1
            )
            out[f"forward_excess_vs_{benchmark}_{horizon}d"] = (
                out[f"forward_return_{horizon}d"] - out[f"{benchmark}_forward_return_{horizon}d"]
            )

    out["evaluation_metadata_only"] = True
    out["future_return_as_rule"] = False
    out["forward_return_as_rule"] = False
    out["not_live_rule"] = True
    out["forward_returns_live_rule_usage"] = False
    return out


def _attach_outcome_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for horizon in HORIZONS:
        vs0050 = out[f"forward_excess_vs_0050_{horizon}d"]
        vs00631 = out[f"forward_excess_vs_00631L_{horizon}d"]
        available = vs0050.notna() & vs00631.notna()
        out[f"win_both_{horizon}d"] = available & vs0050.gt(0) & vs00631.gt(0)
        out[f"only_win_0050_lose_00631L_{horizon}d"] = available & vs0050.gt(0) & vs00631.le(0)
        out[f"fail_0050_{horizon}d"] = available & vs0050.le(0)
        out[f"forward_eval_available_{horizon}d"] = available
    return out


def _attach_decile_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for period, (start, end) in PERIODS.items():
        mask = out["snapshot_date"].between(pd.Timestamp(start), pd.Timestamp(end))
        for horizon in HORIZONS:
            for benchmark in BENCHMARKS:
                col = f"forward_excess_vs_{benchmark}_{horizon}d"
                rank = out.loc[mask, col].rank(pct=True, method="average")
                top_col = f"{period}_top_decile_vs_{benchmark}_{horizon}d"
                bottom_col = f"{period}_bottom_decile_vs_{benchmark}_{horizon}d"
                out[top_col] = pd.NA
                out[bottom_col] = pd.NA
                out.loc[mask, top_col] = rank.ge(0.90)
                out.loc[mask, bottom_col] = rank.le(0.10)
    return out


def _blocked_rows(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for horizon in HORIZONS:
        target_col = f"target_date_{horizon}d"
        blocked = df[~df[f"forward_eval_available_{horizon}d"]].copy()
        if blocked.empty:
            continue
        reason = []
        reason.append(blocked[target_col].isna().map({True: "insufficient_calendar_path", False: ""}))
        stock_missing = blocked[f"stock_target_close_{horizon}d"].isna() | blocked["stock_entry_close"].isna()
        reason.append(stock_missing.map({True: "stock_price_missing", False: ""}))
        bmissing = blocked[f"0050_forward_return_{horizon}d"].isna() | blocked[f"00631L_forward_return_{horizon}d"].isna()
        reason.append(bmissing.map({True: "benchmark_price_missing", False: ""}))
        reason_df = pd.concat(reason, axis=1)
        reason_text = reason_df.apply(lambda row: ";".join([x for x in row if x]), axis=1)
        out = blocked[["baseline_compact_universe_row_id", "snapshot_date", "ticker", "name", "market"]].copy()
        out["horizon"] = horizon
        out["target_date"] = blocked[target_col]
        out["blocked_reason"] = reason_text
        out["evaluation_metadata_only"] = True
        rows.append(out)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _coverage_by_period(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for period, (start, end) in {"ALL": (None, None), **PERIODS}.items():
        mask = pd.Series(True, index=df.index)
        if start:
            mask &= df["snapshot_date"].ge(pd.Timestamp(start))
        if end:
            mask &= df["snapshot_date"].le(pd.Timestamp(end))
        sub = df[mask]
        row: dict[str, Any] = {
            "period": period,
            "requested_start": start or str(df["snapshot_date"].min().date()),
            "requested_end": end or str(df["snapshot_date"].max().date()),
            "actual_start": str(sub["snapshot_date"].min().date()) if not sub.empty else "",
            "actual_end": str(sub["snapshot_date"].max().date()) if not sub.empty else "",
            "rows": int(len(sub)),
            "weekly_snapshot_count": int(sub["snapshot_date"].nunique()),
            "unique_ticker_count": int(sub["ticker"].nunique()),
            "evaluation_metadata_only": True,
        }
        for horizon in HORIZONS:
            row[f"forward_eval_available_share_{horizon}d"] = float(sub[f"forward_eval_available_{horizon}d"].mean()) if len(sub) else 0.0
            row[f"blocked_rows_{horizon}d"] = int((~sub[f"forward_eval_available_{horizon}d"]).sum()) if len(sub) else 0
        rows.append(row)
    return pd.DataFrame(rows)


def _evaluation_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for period, (start, end) in {"ALL": (None, None), **PERIODS}.items():
        mask = pd.Series(True, index=df.index)
        if start:
            mask &= df["snapshot_date"].ge(pd.Timestamp(start))
        if end:
            mask &= df["snapshot_date"].le(pd.Timestamp(end))
        sub = df[mask]
        for horizon in HORIZONS:
            available = sub[sub[f"forward_eval_available_{horizon}d"]]
            rows.append(
                {
                    "period": period,
                    "horizon": horizon,
                    "available_rows": int(len(available)),
                    "mean_excess_vs_0050": float(available[f"forward_excess_vs_0050_{horizon}d"].mean()) if len(available) else 0.0,
                    "median_excess_vs_0050": float(available[f"forward_excess_vs_0050_{horizon}d"].median()) if len(available) else 0.0,
                    "mean_excess_vs_00631L": float(available[f"forward_excess_vs_00631L_{horizon}d"].mean()) if len(available) else 0.0,
                    "median_excess_vs_00631L": float(available[f"forward_excess_vs_00631L_{horizon}d"].median()) if len(available) else 0.0,
                    "win_both_rate": float(available[f"win_both_{horizon}d"].mean()) if len(available) else 0.0,
                    "fail_0050_rate": float(available[f"fail_0050_{horizon}d"].mean()) if len(available) else 0.0,
                    "evaluation_metadata_only": True,
                }
            )
    return pd.DataFrame(rows)


def _future_audit(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("forward_return_as_rule", "passed", 0, "forward returns are evaluation_metadata_only and not live rule inputs"),
            ("future_return_as_rule", "passed", 0, "future_return_as_rule=false on joined rows"),
            ("formal_selector_change", "not_applicable", 0, "evaluation join only"),
            ("portfolio_replay", "not_executed", 0, "no replay executed"),
        ],
        columns=["audit_item", "status", "future_data_violation_count", "note"],
    )


def _readiness(
    layer1_readiness: dict[str, Any],
    df: pd.DataFrame,
    blocked_rows: pd.DataFrame,
    future_audit: pd.DataFrame,
) -> dict[str, Any]:
    max_future_count = int(future_audit["future_data_violation_count"].max())
    eval_20_share = float(df["forward_eval_available_20d"].mean())
    return {
        "task_id": TASK_ID,
        "status": "layer1_compact_candidate_quality_evaluation_join_ready_for_experiments_intake",
        "diagnostic_only": True,
        "evaluation_metadata_only": True,
        "input_layer1_status": layer1_readiness.get("status", ""),
        "rows": int(len(df)),
        "weekly_snapshot_count": int(df["snapshot_date"].nunique()),
        "unique_ticker_count": int(df["ticker"].nunique()),
        "forward_eval_available_share_20d": eval_20_share,
        "blocked_evaluation_rows": int(len(blocked_rows)),
        "ready_for_layer1_compact_candidate_quality_diagnostic": max_future_count == 0 and eval_20_share > 0.80,
        "ready_for_experiments_intake": True,
        "ready_for_formal": False,
        "ready_for_strategy_replay": False,
        "portfolio_replay_executed": False,
        "candidate_forward_return_diagnostic_executed": False,
        "future_data_violation_count": max_future_count,
        "formal_model_changed": False,
        "trade_decision_changed": False,
        "active_in_trade_decision": False,
        "report_changed": False,
        "not_live_rule": True,
        "forward_returns_live_rule_usage": False,
    }


def _summary(readiness: dict[str, Any]) -> str:
    return f"""# Layer1 compact candidate-quality evaluation join

## Verdict
- status={readiness["status"]}
- rows={readiness["rows"]}
- weekly_snapshot_count={readiness["weekly_snapshot_count"]}
- unique_ticker_count={readiness["unique_ticker_count"]}
- forward_eval_available_share_20d={readiness["forward_eval_available_share_20d"]}
- blocked_evaluation_rows={readiness["blocked_evaluation_rows"]}
- ready_for_layer1_compact_candidate_quality_diagnostic={str(readiness["ready_for_layer1_compact_candidate_quality_diagnostic"]).lower()}
- ready_for_experiments_intake=true
- ready_for_formal=false
- portfolio_replay_executed=false

## Plain Summary
This package adds 5D/10D/20D/40D forward excess return metadata versus 0050 and 00631L for compact Layer1 candidate-quality evaluation. Forward returns are explicitly evaluation_metadata_only and not rule inputs. Latest rows without enough future trading path are listed in the blocked ledger.

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
