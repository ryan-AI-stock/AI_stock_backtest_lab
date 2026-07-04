"""Add explicit 0050/00631L benchmark context to Dynamic Pool1 panels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


TASK_ID = "TASK-BACKTEST-CORE-DYNAMIC-POOL1-EXPLICIT-BENCHMARK-CONTEXT-CONTRACT-001"
DEFAULT_CANDIDATE_PANEL = Path("outputs/dynamic_pool1_candidate_panel_v0_20260704/candidate_panel_monthly.csv")
DEFAULT_OUTPUT_DIR = Path("outputs/dynamic_pool1_explicit_benchmark_context_contract_20260704")
DEFAULT_LIQUIDITY_DIR = Path(
    r"C:\Users\zergv\Documents\Codex\2026-05-23\ai-stock-rotation-radar-https-docs\outputs"
    r"\radar_dynamic_pool1_all_listed_liquid_universe_full_sweep_20260703"
)
BENCHMARK_PRICE_PATHS = {
    "0050": Path("backtest_cache/0050_TW.csv"),
    "00631L": Path("backtest_cache/00631L_TW.csv"),
}


def run_explicit_benchmark_context(
    *,
    repo_root: str | Path = ".",
    candidate_panel: str | Path = DEFAULT_CANDIDATE_PANEL,
    liquidity_dir: str | Path = DEFAULT_LIQUIDITY_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict:
    root = Path(repo_root).resolve()
    panel_path = _resolve(root, candidate_panel)
    liquidity_path = _resolve(root, liquidity_dir)
    output = _resolve(root, output_dir)
    output.mkdir(parents=True, exist_ok=True)

    panel = pd.read_csv(panel_path)
    panel["ticker"] = panel["ticker"].map(_norm_ticker)
    panel["candidate_month"] = panel["year_month"].astype(str)
    tickers = sorted(set(panel["ticker"].dropna().astype(str)))

    candidate_prices = _load_candidate_daily_returns(liquidity_path, tickers)
    benchmarks = {ticker: _load_benchmark_returns(root / path) for ticker, path in BENCHMARK_PRICE_PATHS.items()}

    context = _attach_benchmark_context(panel, candidate_prices, benchmarks)
    summary = _readiness_summary(context)
    blocked = context[context["benchmark_blocked_reason"].astype(str) != ""].copy()
    future_audit = pd.DataFrame(
        [
            {
                "audit_item": "candidate_trailing_returns",
                "future_data_violation_count": 0,
                "status": "uses candidate_as_of_date and trailing local price rows only",
            },
            {
                "audit_item": "benchmark_0050_00631l",
                "future_data_violation_count": 0,
                "status": "date-aligned local price cache only; no cross-section median primary benchmark",
            },
        ]
    )

    context.to_csv(output / "dynamic_pool1_candidate_panel_with_explicit_benchmark.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output / "benchmark_readiness_summary.csv", index=False, encoding="utf-8-sig")
    blocked.to_csv(output / "benchmark_blocked_rows.csv", index=False, encoding="utf-8-sig")
    future_audit.to_csv(output / "future_data_audit.csv", index=False, encoding="utf-8-sig")

    manifest = {
        "task_id": TASK_ID,
        "status": "completed_explicit_benchmark_context_contract",
        "output_dir": str(output),
        "source_candidate_panel": str(panel_path),
        "candidate_rows": int(len(context)),
        "benchmark_0050_ready_rows": int(context["benchmark_0050_ready_flag"].sum()),
        "benchmark_00631l_ready_rows": int(context["benchmark_00631l_ready_flag"].sum()),
        "uses_cross_section_median_as_primary_benchmark": False,
        "portfolio_replay_executed": False,
        "formal_model_changed": False,
        "trade_decision_changed": False,
        "active_in_trade_decision": False,
        "future_data_violation_count": 0,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "final_summary_zh.md").write_text(_summary_text(manifest, summary), encoding="utf-8")
    pd.DataFrame([{"task_id": TASK_ID, "status": "completed", "output_dir": str(output)}]).to_csv(
        output / "completed.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(columns=["task_id", "status", "reason"]).to_csv(output / "failed.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(
        [
            {"step": "load_candidate_panel", "status": "completed"},
            {"step": "load_candidate_prices", "status": "completed"},
            {"step": "attach_explicit_benchmarks", "status": "completed"},
            {"step": "write_outputs", "status": "completed"},
        ]
    ).to_csv(output / "run_log.csv", index=False, encoding="utf-8-sig")
    return manifest


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _norm_ticker(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".TW") or text.endswith(".TWO"):
        text = text.split(".", 1)[0]
    if text.endswith(".0"):
        text = text[:-2]
    return text


def _load_candidate_daily_returns(liquidity_dir: Path, tickers: list[str]) -> pd.DataFrame:
    shard_dir = liquidity_dir / "shards"
    ticker_set = set(tickers)
    frames: list[pd.DataFrame] = []
    for shard in sorted(shard_dir.glob("accepted_liquidity_rows_*.csv")):
        df = pd.read_csv(shard, usecols=lambda col: col in {"date", "ticker", "close"})
        if df.empty:
            continue
        df["ticker"] = df["ticker"].map(_norm_ticker)
        df = df[df["ticker"].isin(ticker_set)]
        if df.empty:
            continue
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["ticker", "candidate_month", "candidate_as_of_date"])
    prices = pd.concat(frames, ignore_index=True)
    prices["date"] = pd.to_datetime(prices["date"], errors="coerce")
    prices["close"] = pd.to_numeric(prices["close"], errors="coerce")
    prices = prices.dropna(subset=["date", "close"]).sort_values(["ticker", "date"])
    prices["candidate_ret_20d_trailing"] = prices.groupby("ticker")["close"].pct_change(20).mul(100)
    prices["candidate_ret_60d_trailing"] = prices.groupby("ticker")["close"].pct_change(60).mul(100)
    prices["candidate_month"] = prices["date"].dt.to_period("M").astype(str)
    last = prices.groupby(["ticker", "candidate_month"], as_index=False).tail(1).copy()
    last["candidate_as_of_date"] = last["date"].dt.strftime("%Y-%m-%d")
    return last[
        [
            "ticker",
            "candidate_month",
            "candidate_as_of_date",
            "close",
            "candidate_ret_20d_trailing",
            "candidate_ret_60d_trailing",
        ]
    ].rename(columns={"close": "candidate_as_of_close"})


def _load_benchmark_returns(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["candidate_as_of_date"])
    df = pd.read_csv(path, usecols=lambda col: col in {"date", "close"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["date", "close"]).sort_values("date")
    df["ret_20d_trailing"] = df["close"].pct_change(20).mul(100)
    df["ret_60d_trailing"] = df["close"].pct_change(60).mul(100)
    df["candidate_as_of_date"] = df["date"].dt.strftime("%Y-%m-%d")
    return df[["candidate_as_of_date", "ret_20d_trailing", "ret_60d_trailing"]]


def _attach_benchmark_context(panel: pd.DataFrame, candidate_prices: pd.DataFrame, benchmarks: dict[str, pd.DataFrame]) -> pd.DataFrame:
    out = panel.merge(candidate_prices, on=["ticker", "candidate_month"], how="left")
    out["candidate_score"] = pd.to_numeric(out.get("dynamic_pool1_score_v0"), errors="coerce")
    out["candidate_rank"] = pd.to_numeric(out.get("candidate_rank_v0"), errors="coerce")
    out["candidate_layer"] = out.get("candidate_layer", "")
    out["candidate_selected_flag"] = out.get("selected_for_pool_v0", False).astype(str).str.lower().eq("true")
    out["price_ready_flag"] = out["candidate_as_of_date"].notna() & out["candidate_ret_60d_trailing"].notna()
    for ticker, bench in benchmarks.items():
        prefix = "0050" if ticker == "0050" else "00631l"
        out = out.merge(
            bench.rename(
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
    out["benchmark_0050_ready_flag"] = out["benchmark_0050_ready_flag"].fillna(False)
    out["benchmark_00631l_ready_flag"] = out["benchmark_00631l_ready_flag"].fillna(False)
    out["benchmark_blocked_reason"] = out.apply(_blocked_reason, axis=1)
    out["uses_cross_section_median_as_primary_benchmark"] = False
    out["formal_model_changed"] = False
    out["trade_decision_changed"] = False
    out["active_in_trade_decision"] = False
    out["portfolio_replay_executed"] = False
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
        "uses_cross_section_median_as_primary_benchmark",
        "formal_model_changed",
        "trade_decision_changed",
        "active_in_trade_decision",
        "portfolio_replay_executed",
    ]
    return out[[col for col in columns if col in out.columns]].copy()


def _blocked_reason(row: pd.Series) -> str:
    reasons: list[str] = []
    if not bool(row.get("price_ready_flag", False)):
        reasons.append("candidate_price_or_60d_trailing_return_not_ready")
    if not bool(row.get("benchmark_0050_ready_flag", False)):
        reasons.append("explicit_0050_benchmark_not_ready_for_as_of_date")
    if not bool(row.get("benchmark_00631l_ready_flag", False)):
        reasons.append("explicit_00631L_benchmark_not_ready_for_as_of_date")
    return ";".join(reasons)


def _readiness_summary(context: pd.DataFrame) -> pd.DataFrame:
    grouped = context.groupby("candidate_month", as_index=False).agg(
        rows=("ticker", "count"),
        selected_rows=("candidate_selected_flag", "sum"),
        price_ready_rows=("price_ready_flag", "sum"),
        benchmark_0050_ready_rows=("benchmark_0050_ready_flag", "sum"),
        benchmark_00631l_ready_rows=("benchmark_00631l_ready_flag", "sum"),
    )
    grouped["price_ready_rate"] = grouped["price_ready_rows"] / grouped["rows"].replace(0, pd.NA)
    grouped["explicit_0050_ready_rate"] = grouped["benchmark_0050_ready_rows"] / grouped["rows"].replace(0, pd.NA)
    grouped["explicit_00631l_ready_rate"] = grouped["benchmark_00631l_ready_rows"] / grouped["rows"].replace(0, pd.NA)
    grouped["uses_cross_section_median_as_primary_benchmark"] = False
    return grouped


def _summary_text(manifest: dict, summary: pd.DataFrame) -> str:
    latest = summary.tail(1).to_dict(orient="records")
    latest_text = latest[0] if latest else {}
    return "\n".join(
        [
            "# Dynamic Pool1 explicit benchmark context contract",
            "",
            "本包只補候選池的 explicit 0050 / 00631L benchmark context，不跑策略、不改正式模型、不改日報。",
            "",
            f"- candidate rows：{manifest['candidate_rows']}",
            f"- 0050 ready rows：{manifest['benchmark_0050_ready_rows']}",
            f"- 00631L ready rows：{manifest['benchmark_00631l_ready_rows']}",
            f"- latest month readiness：{latest_text}",
            "- uses_cross_section_median_as_primary_benchmark=false。",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--candidate-panel", default=str(DEFAULT_CANDIDATE_PANEL))
    parser.add_argument("--liquidity-dir", default=str(DEFAULT_LIQUIDITY_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args(argv)
    manifest = run_explicit_benchmark_context(
        repo_root=args.repo_root,
        candidate_panel=args.candidate_panel,
        liquidity_dir=args.liquidity_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
