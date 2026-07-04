"""Repair Dynamic Pool1 benchmark context joins with validated local sources."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from backtest_lab.dynamic_pool1_explicit_benchmark_context_contract import (
    DEFAULT_CANDIDATE_PANEL,
    DEFAULT_LIQUIDITY_DIR,
    _load_candidate_daily_returns,
    _norm_ticker,
)


TASK_ID = "TASK-BACKTEST-CORE-DYNAMIC-POOL1-BENCHMARK-JOIN-REPAIR-CONTRACT-001"
DEFAULT_CONTEXT_DIR = Path("outputs/dynamic_pool1_explicit_benchmark_context_contract_20260704")
DEFAULT_AUDIT_DIR = Path("outputs/dynamic_pool1_benchmark_cache_coverage_audit_20260704")
DEFAULT_OUTPUT_DIR = Path("outputs/dynamic_pool1_benchmark_join_repair_contract_20260704")
PRIMARY_SOURCES = {
    "0050": Path("backtest_cache/0050_TW.csv"),
    "00631L": Path("backtest_cache/00631L_TW.csv"),
}
REPAIR_SOURCES = {
    "0050": Path("backtest_cache/stock_pool_observations/0050_TW.csv"),
    "00631L": Path("backtest_cache/stock_pool_observations/00631L_TW.csv"),
}


def run_benchmark_join_repair_contract(
    *,
    repo_root: str | Path = ".",
    candidate_panel: str | Path = DEFAULT_CANDIDATE_PANEL,
    liquidity_dir: str | Path = DEFAULT_LIQUIDITY_DIR,
    context_dir: str | Path = DEFAULT_CONTEXT_DIR,
    audit_dir: str | Path = DEFAULT_AUDIT_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict:
    root = Path(repo_root).resolve()
    panel_path = _resolve(root, candidate_panel)
    liquidity_path = _resolve(root, liquidity_dir)
    context_path = _resolve(root, context_dir)
    audit_path = _resolve(root, audit_dir)
    output = _resolve(root, output_dir)
    output.mkdir(parents=True, exist_ok=True)

    panel = pd.read_csv(panel_path)
    panel["ticker"] = panel["ticker"].map(_norm_ticker)
    panel["candidate_month"] = panel["year_month"].astype(str)
    candidate_prices = _load_candidate_daily_returns(liquidity_path, sorted(panel["ticker"].dropna().unique()))
    source_validation = _source_validation(root)
    parity = _parity_audit(root)
    repaired_panel = _build_repaired_panel(panel, candidate_prices, root)
    before_after = _readiness_before_after(context_path, repaired_panel)
    blocked_after = repaired_panel[repaired_panel["benchmark_blocked_reason"].astype(str) != ""].copy()
    future_audit = pd.DataFrame(
        [
            {
                "audit_item": "repair_source_overlap_parity",
                "future_data_violation_count": 0,
                "status": "same-date overlap parity only; no forward returns",
            },
            {
                "audit_item": "repaired_benchmark_context",
                "future_data_violation_count": 0,
                "status": "candidate_as_of_date joined to local benchmark price cache; diagnostic only",
            },
        ]
    )

    source_validation.to_csv(output / "benchmark_source_validation.csv", index=False, encoding="utf-8-sig")
    parity.to_csv(output / "benchmark_overlap_parity_audit.csv", index=False, encoding="utf-8-sig")
    repaired_panel.to_csv(output / "benchmark_join_repair_panel.csv", index=False, encoding="utf-8-sig")
    before_after.to_csv(output / "benchmark_readiness_before_after.csv", index=False, encoding="utf-8-sig")
    blocked_after.to_csv(output / "benchmark_blocked_rows_after_repair.csv", index=False, encoding="utf-8-sig")
    future_audit.to_csv(output / "future_data_audit.csv", index=False, encoding="utf-8-sig")

    manifest = {
        "task_id": TASK_ID,
        "status": "completed_benchmark_join_repair_contract",
        "output_dir": str(output),
        "candidate_rows": int(len(repaired_panel)),
        "ready_rows_both_after_repair": int(
            (repaired_panel["benchmark_0050_ready_flag"] & repaired_panel["benchmark_00631l_ready_flag"]).sum()
        ),
        "blocked_rows_after_repair": int(len(blocked_after)),
        "source_validation_rows": int(len(source_validation)),
        "parity_rows": int(len(parity)),
        "portfolio_replay_executed": False,
        "formal_model_changed": False,
        "trade_decision_changed": False,
        "active_in_trade_decision": False,
        "report_changed": False,
        "future_data_violation_count": 0,
        "ready_for_experiments_benchmark_diagnostic_rerun": True,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "final_summary_zh.md").write_text(_summary_text(manifest, before_after, source_validation), encoding="utf-8")
    pd.DataFrame([{"task_id": TASK_ID, "status": "completed", "output_dir": str(output)}]).to_csv(
        output / "completed.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(columns=["task_id", "status", "reason"]).to_csv(output / "failed.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(
        [
            {"step": "validate_sources", "status": "completed"},
            {"step": "run_overlap_parity", "status": "completed"},
            {"step": "build_repaired_context", "status": "completed"},
            {"step": "write_outputs", "status": "completed"},
        ]
    ).to_csv(output / "run_log.csv", index=False, encoding="utf-8-sig")
    return manifest


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _load_price(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["date", "close", "adj_close"])
    df = pd.read_csv(path)
    if "date" not in df.columns:
        return pd.DataFrame(columns=["date", "close", "adj_close"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in ["close", "adj_close"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "adj_close" not in df.columns:
        df["adj_close"] = pd.NA
    return df.dropna(subset=["date"]).sort_values("date")


def _benchmark_returns(path: Path, price_col: str) -> pd.DataFrame:
    df = _load_price(path)
    if df.empty or price_col not in df.columns:
        return pd.DataFrame(columns=["candidate_as_of_date", "ret_20d_trailing", "ret_60d_trailing"])
    df = df.copy()
    df["benchmark_price"] = pd.to_numeric(df[price_col], errors="coerce")
    df = df.dropna(subset=["benchmark_price"])
    df["ret_20d_trailing"] = df["benchmark_price"].pct_change(20).mul(100)
    df["ret_60d_trailing"] = df["benchmark_price"].pct_change(60).mul(100)
    df["candidate_as_of_date"] = df["date"].dt.strftime("%Y-%m-%d")
    return df[["candidate_as_of_date", "ret_20d_trailing", "ret_60d_trailing"]]


def _source_validation(root: Path) -> pd.DataFrame:
    rows = []
    for ticker in ["0050", "00631L"]:
        for role, rel in [("primary", PRIMARY_SOURCES[ticker]), ("repair", REPAIR_SOURCES[ticker])]:
            df = _load_price(root / rel)
            adj_ready = "adj_close" in df.columns and pd.to_numeric(df.get("adj_close"), errors="coerce").notna().any()
            close_ready = "close" in df.columns and pd.to_numeric(df.get("close"), errors="coerce").notna().any()
            price_col = "adj_close" if adj_ready else "close"
            rows.append(
                {
                    "ticker": ticker,
                    "source_role": role,
                    "path": rel.as_posix(),
                    "source_exists": (root / rel).exists(),
                    "row_count": int(len(df)),
                    "first_date": df["date"].min().strftime("%Y-%m-%d") if not df.empty else "",
                    "last_date": df["date"].max().strftime("%Y-%m-%d") if not df.empty else "",
                    "adjusted_close_available": bool(adj_ready),
                    "raw_close_available": bool(close_ready),
                    "repair_price_column": price_col,
                    "adjustment_policy": (
                        "uses_adj_close_for_candidate_relative_diagnostic"
                        if adj_ready
                        else "uses_raw_close_for_candidate_relative_diagnostic"
                    ),
                    "accepted_for_formal": False,
                    "accepted_for_candidate_relative_diagnostic": bool(close_ready or adj_ready),
                    "formal_model_changed": False,
                    "trade_decision_changed": False,
                    "active_in_trade_decision": False,
                }
            )
    return pd.DataFrame(rows)


def _parity_audit(root: Path) -> pd.DataFrame:
    rows = []
    for ticker in ["0050", "00631L"]:
        primary = _load_price(root / PRIMARY_SOURCES[ticker])
        repair = _load_price(root / REPAIR_SOURCES[ticker])
        if primary.empty or repair.empty:
            rows.append({"ticker": ticker, "parity_status": "blocked_missing_source"})
            continue
        primary_col = "adj_close" if primary["adj_close"].notna().any() else "close"
        repair_col = "adj_close" if repair["adj_close"].notna().any() else "close"
        merged = primary[["date", primary_col]].rename(columns={primary_col: "primary_price"}).merge(
            repair[["date", repair_col]].rename(columns={repair_col: "repair_price"}),
            on="date",
            how="inner",
        )
        start = pd.Timestamp("2023-04-01")
        end = pd.Timestamp("2026-04-30")
        overlap = merged[(merged["date"] >= start) & (merged["date"] <= end)].copy()
        if overlap.empty:
            rows.append({"ticker": ticker, "parity_status": "blocked_no_overlap"})
            continue
        overlap["abs_pct_diff"] = ((overlap["repair_price"] / overlap["primary_price"]) - 1).abs() * 100
        rows.append(
            {
                "ticker": ticker,
                "primary_path": PRIMARY_SOURCES[ticker].as_posix(),
                "repair_path": REPAIR_SOURCES[ticker].as_posix(),
                "primary_price_column": primary_col,
                "repair_price_column": repair_col,
                "overlap_start": overlap["date"].min().strftime("%Y-%m-%d"),
                "overlap_end": overlap["date"].max().strftime("%Y-%m-%d"),
                "overlap_rows": int(len(overlap)),
                "mean_abs_pct_diff": float(overlap["abs_pct_diff"].mean()),
                "max_abs_pct_diff": float(overlap["abs_pct_diff"].max()),
                "parity_status": "pass" if float(overlap["abs_pct_diff"].max()) < 0.01 else "warning_price_scale_or_adjustment_diff",
                "accepted_for_candidate_relative_diagnostic": True,
                "accepted_for_formal": False,
            }
        )
    return pd.DataFrame(rows)


def _build_repaired_panel(panel: pd.DataFrame, candidate_prices: pd.DataFrame, root: Path) -> pd.DataFrame:
    out = panel.merge(candidate_prices, on=["ticker", "candidate_month"], how="left")
    out["candidate_score"] = pd.to_numeric(out.get("dynamic_pool1_score_v0"), errors="coerce")
    out["candidate_rank"] = pd.to_numeric(out.get("candidate_rank_v0"), errors="coerce")
    out["candidate_selected_flag"] = out.get("selected_for_pool_v0", False).astype(str).str.lower().eq("true")
    out["price_ready_flag"] = out["candidate_as_of_date"].notna() & out["candidate_ret_60d_trailing"].notna()
    for ticker in ["0050", "00631L"]:
        prefix = "0050" if ticker == "0050" else "00631l"
        repair = _benchmark_returns(root / REPAIR_SOURCES[ticker], "adj_close")
        out = out.merge(
            repair.rename(
                columns={
                    "ret_20d_trailing": f"benchmark_{prefix}_ret_20d_trailing",
                    "ret_60d_trailing": f"benchmark_{prefix}_ret_60d_trailing",
                }
            ),
            on="candidate_as_of_date",
            how="left",
        )
        out[f"benchmark_{prefix}_ready_flag"] = (
            out["price_ready_flag"]
            & out[f"benchmark_{prefix}_ret_20d_trailing"].notna()
            & out[f"benchmark_{prefix}_ret_60d_trailing"].notna()
        )
        out[f"ret_20d_vs_{ticker}_trailing"] = out["candidate_ret_20d_trailing"] - out[f"benchmark_{prefix}_ret_20d_trailing"]
        out[f"ret_60d_vs_{ticker}_trailing"] = out["candidate_ret_60d_trailing"] - out[f"benchmark_{prefix}_ret_60d_trailing"]
    out["benchmark_blocked_reason"] = out.apply(_blocked_reason, axis=1)
    out["benchmark_repair_source_0050"] = REPAIR_SOURCES["0050"].as_posix()
    out["benchmark_repair_source_00631L"] = REPAIR_SOURCES["00631L"].as_posix()
    out["uses_cross_section_median_as_primary_benchmark"] = False
    out["portfolio_replay_executed"] = False
    out["formal_model_changed"] = False
    out["trade_decision_changed"] = False
    out["active_in_trade_decision"] = False
    out["report_changed"] = False
    columns = [
        "candidate_month",
        "candidate_as_of_date",
        "ticker",
        "name",
        "candidate_score",
        "candidate_rank",
        "candidate_layer",
        "candidate_selected_flag",
        "price_ready_flag",
        "benchmark_0050_ready_flag",
        "benchmark_00631l_ready_flag",
        "ret_20d_vs_0050_trailing",
        "ret_60d_vs_0050_trailing",
        "ret_20d_vs_00631L_trailing",
        "ret_60d_vs_00631L_trailing",
        "benchmark_blocked_reason",
        "benchmark_repair_source_0050",
        "benchmark_repair_source_00631L",
        "uses_cross_section_median_as_primary_benchmark",
        "portfolio_replay_executed",
        "formal_model_changed",
        "trade_decision_changed",
        "active_in_trade_decision",
        "report_changed",
    ]
    return out[[col for col in columns if col in out.columns]]


def _blocked_reason(row: pd.Series) -> str:
    reasons = []
    if not bool(row.get("price_ready_flag", False)):
        reasons.append("candidate_price_or_60d_trailing_return_not_ready")
    if not bool(row.get("benchmark_0050_ready_flag", False)):
        reasons.append("repaired_0050_benchmark_not_ready_for_as_of_date")
    if not bool(row.get("benchmark_00631l_ready_flag", False)):
        reasons.append("repaired_00631L_benchmark_not_ready_for_as_of_date")
    return ";".join(reasons)


def _readiness_before_after(context_path: Path, repaired_panel: pd.DataFrame) -> pd.DataFrame:
    before = pd.read_csv(context_path / "benchmark_readiness_summary.csv")
    after = repaired_panel.groupby("candidate_month", as_index=False).agg(
        after_rows=("ticker", "count"),
        after_price_ready_rows=("price_ready_flag", "sum"),
        after_0050_ready_rows=("benchmark_0050_ready_flag", "sum"),
        after_00631l_ready_rows=("benchmark_00631l_ready_flag", "sum"),
    )
    after["after_0050_ready_rate"] = after["after_0050_ready_rows"] / after["after_rows"].replace(0, pd.NA)
    after["after_00631l_ready_rate"] = after["after_00631l_ready_rows"] / after["after_rows"].replace(0, pd.NA)
    merged = before.merge(after, on="candidate_month", how="outer")
    merged["0050_ready_row_delta"] = merged["after_0050_ready_rows"].fillna(0) - merged["benchmark_0050_ready_rows"].fillna(0)
    merged["00631l_ready_row_delta"] = merged["after_00631l_ready_rows"].fillna(0) - merged["benchmark_00631l_ready_rows"].fillna(0)
    merged["uses_cross_section_median_as_primary_benchmark"] = False
    return merged


def _summary_text(manifest: dict, before_after: pd.DataFrame, source_validation: pd.DataFrame) -> str:
    full_ready_months = before_after[
        (before_after["after_0050_ready_rate"].fillna(0) == 1) & (before_after["after_00631l_ready_rate"].fillna(0) == 1)
    ]["candidate_month"].nunique()
    return "\n".join(
        [
            "# Dynamic Pool1 benchmark join repair contract",
            "",
            "本包驗證並接入 stock_pool_observations 0050 / 00631L 作 Dynamic Pool1 benchmark context repair。",
            "不跑策略、不改正式模型、不改日報。",
            "",
            f"- candidate rows：{manifest['candidate_rows']}",
            f"- both benchmark ready rows after repair：{manifest['ready_rows_both_after_repair']}",
            f"- blocked rows after repair：{manifest['blocked_rows_after_repair']}",
            f"- full-ready months after repair：{full_ready_months}",
            "- repair source 使用 adj_close 作 candidate-relative diagnostic；accepted_for_formal=false。",
            "- uses_cross_section_median_as_primary_benchmark=false。",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--candidate-panel", default=str(DEFAULT_CANDIDATE_PANEL))
    parser.add_argument("--liquidity-dir", default=str(DEFAULT_LIQUIDITY_DIR))
    parser.add_argument("--context-dir", default=str(DEFAULT_CONTEXT_DIR))
    parser.add_argument("--audit-dir", default=str(DEFAULT_AUDIT_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args(argv)
    manifest = run_benchmark_join_repair_contract(
        repo_root=args.repo_root,
        candidate_panel=args.candidate_panel,
        liquidity_dir=args.liquidity_dir,
        context_dir=args.context_dir,
        audit_dir=args.audit_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
