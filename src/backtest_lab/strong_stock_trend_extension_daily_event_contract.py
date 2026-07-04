"""Build exact daily trend-extension event contract for strong stocks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


TASK_ID = "TASK-BACKTEST-CORE-STRONG-STOCK-TREND-EXTENSION-DAILY-EVENT-CONTRACT-001"
EXPERIMENTS_TASK_ID = "TASK-BACKTEST-EXPERIMENTS-STRONG-STOCK-TREND-EXTENSION-DAILY-EVENT-CONTRACT-VALIDATION-001"
DEFAULT_CANDIDATE_CONTRACT = Path(
    "outputs/dynamic_pool1_benchmark_aware_candidate_contract_20260704/dynamic_pool1_benchmark_aware_candidate_contract.csv"
)
DEFAULT_CANDIDATE_CONTEXT = Path("outputs/dynamic_pool1_candidate_panel_v0_20260704/candidate_pool_by_month.csv")
DEFAULT_LIQUIDITY_DIR = Path(
    r"C:\Users\zergv\Documents\Codex\2026-05-23\ai-stock-rotation-radar-https-docs\outputs"
    r"\radar_dynamic_pool1_all_listed_liquid_universe_full_sweep_20260703"
)
DEFAULT_OUTPUT_DIR = Path("outputs/strong_stock_trend_extension_daily_event_contract_20260704")
BENCHMARK_PRICE_PATHS = {
    "0050": Path("backtest_cache/stock_pool_observations/0050_TW.csv"),
    "00631L": Path("backtest_cache/stock_pool_observations/00631L_TW.csv"),
}


def run_strong_stock_trend_extension_daily_event_contract(
    *,
    repo_root: str | Path = ".",
    candidate_contract: str | Path = DEFAULT_CANDIDATE_CONTRACT,
    candidate_context: str | Path = DEFAULT_CANDIDATE_CONTEXT,
    liquidity_dir: str | Path = DEFAULT_LIQUIDITY_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict:
    root = Path(repo_root).resolve()
    candidates = _load_candidates(_resolve(root, candidate_contract), _resolve(root, candidate_context))
    liquidity = _resolve(root, liquidity_dir)
    output = _resolve(root, output_dir)
    output.mkdir(parents=True, exist_ok=True)
    prices = _load_candidate_prices(liquidity, candidates)
    benchmark = _load_benchmark_panel(root)
    daily = _build_daily_feature_panel(candidates, prices, benchmark)
    contract = _build_event_contract(daily)
    readiness = _candidate_readiness(candidates, daily)
    benchmark_audit = _benchmark_readiness_audit(daily)
    future_audit = _future_data_audit(contract)
    summary = _variant_summary(contract)

    contract.to_csv(output / "trend_extension_daily_event_contract.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output / "trend_extension_variant_summary.csv", index=False, encoding="utf-8-sig")
    readiness.to_csv(output / "candidate_readiness_and_blockers.csv", index=False, encoding="utf-8-sig")
    benchmark_audit.to_csv(output / "benchmark_readiness_audit.csv", index=False, encoding="utf-8-sig")
    future_audit.to_csv(output / "future_data_audit.csv", index=False, encoding="utf-8-sig")
    manifest = {
        "task_id": TASK_ID,
        "status": "completed_trend_extension_daily_event_contract_ready",
        "output_dir": str(output),
        "candidate_contract_source": str(_resolve(root, candidate_contract)),
        "candidate_context_source": str(_resolve(root, candidate_context)),
        "candidate_rows": int(len(candidates)),
        "daily_feature_rows": int(len(daily)),
        "event_rows": int(len(contract)),
        "variant_count": int(contract["event_variant"].nunique()) if not contract.empty else 0,
        "blocked_candidate_rows": int(readiness["blocked"].sum()) if not readiness.empty else 0,
        "benchmark_blocked_rows": int((~daily["benchmark_0050_ready"] | ~daily["benchmark_00631l_ready"]).sum())
        if not daily.empty
        else 0,
        "future_data_violation_count": int(future_audit["future_data_violation"].sum()) if not future_audit.empty else 0,
        "uses_forward_return_as_rule": False,
        "uses_cross_section_median_as_primary_benchmark": False,
        "portfolio_replay_executed": False,
        "formal_model_changed": False,
        "trade_decision_changed": False,
        "active_in_trade_decision": False,
        "report_changed": False,
        "ready_for_experiments_validation": True,
        "handoff_to_experiments_task": EXPERIMENTS_TASK_ID,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "final_summary_zh.md").write_text(_summary(manifest, summary), encoding="utf-8")
    pd.DataFrame([{"task_id": TASK_ID, "status": "completed", "output_dir": str(output)}]).to_csv(
        output / "completed.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(columns=["task_id", "status", "reason"]).to_csv(output / "failed.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(
        [
            {"step": "load_monthly_candidate_contract", "status": "completed"},
            {"step": "load_daily_price_liquidity_and_benchmarks", "status": "completed"},
            {"step": "compute_exact_daily_trend_extension_features", "status": "completed"},
            {"step": "write_event_contract_package", "status": "completed"},
        ]
    ).to_csv(output / "run_log.csv", index=False, encoding="utf-8-sig")
    return manifest


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _load_candidates(contract_path: Path, context_path: Path) -> pd.DataFrame:
    if not contract_path.exists():
        raise FileNotFoundError(contract_path)
    candidates = pd.read_csv(contract_path).fillna("")
    candidates = candidates[candidates["benchmark_filter_primary_selected"].map(_as_bool)].copy()
    candidates["candidate_as_of_date"] = pd.to_datetime(candidates["candidate_as_of_date"], errors="coerce")
    candidates = candidates.dropna(subset=["candidate_as_of_date"])
    candidates["ticker"] = candidates["ticker"].astype(str).map(_base_ticker)
    candidates["candidate_rank"] = pd.to_numeric(candidates["candidate_rank"], errors="coerce")
    candidates["candidate_score"] = pd.to_numeric(candidates["candidate_score"], errors="coerce")
    if context_path.exists():
        context = pd.read_csv(
            context_path,
            usecols=lambda col: col in {"year_month", "ticker", "name", "market", "candidate_layer", "selected_for_pool_v0"},
        ).fillna("")
        context = context.rename(columns={"year_month": "candidate_month", "name": "candidate_name"})
        context["ticker"] = context["ticker"].astype(str).map(_base_ticker)
        candidates = candidates.merge(
            context[["candidate_month", "ticker", "candidate_name", "market", "selected_for_pool_v0"]],
            on=["candidate_month", "ticker"],
            how="left",
        )
    else:
        candidates["candidate_name"] = ""
        candidates["market"] = ""
        candidates["selected_for_pool_v0"] = ""
    candidates["candidate_name"] = candidates.get("candidate_name", "").fillna("").astype(str)
    candidates["market"] = candidates.get("market", "").fillna("").astype(str)
    candidates["canonical_ticker"] = candidates.apply(lambda row: _canonical_ticker(row["ticker"], row["market"]), axis=1)
    candidates = candidates.sort_values(["ticker", "candidate_as_of_date", "candidate_month"])
    candidates["next_candidate_as_of_date"] = candidates.groupby("ticker")["candidate_as_of_date"].shift(-1)
    candidates["window_end_exclusive"] = candidates["next_candidate_as_of_date"].fillna(candidates["candidate_as_of_date"] + pd.Timedelta(days=35))
    return candidates


def _load_candidate_prices(liquidity_dir: Path, candidates: pd.DataFrame) -> pd.DataFrame:
    tickers = set(candidates["ticker"].astype(str))
    frames = []
    for shard in sorted((liquidity_dir / "shards").glob("accepted_liquidity_rows_*.csv")):
        try:
            frame = pd.read_csv(
                shard,
            usecols=lambda col: col in {"date", "ticker", "name", "market", "close", "turnover_value", "turnover", "liquidity_pass"},
            ).fillna("")
        except (OSError, ValueError):
            continue
        frame["ticker"] = frame["ticker"].astype(str).map(_base_ticker)
        frame = frame[frame["ticker"].isin(tickers)].copy()
        if frame.empty:
            continue
        frame["signal_date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
        turnover_col = "turnover_value" if "turnover_value" in frame.columns else "turnover"
        frame["turnover"] = pd.to_numeric(frame.get(turnover_col, 0), errors="coerce").fillna(0.0)
        frame["liquidity_ready"] = frame.get("liquidity_pass", True).map(_as_bool)
        frame["canonical_ticker"] = frame.apply(lambda row: _canonical_ticker(row["ticker"], row.get("market", "")), axis=1)
        frame["price_name"] = frame.get("name", "").astype(str)
        frames.append(frame[["signal_date", "ticker", "canonical_ticker", "price_name", "close", "turnover", "liquidity_ready"]])
    if not frames:
        return pd.DataFrame(columns=["signal_date", "ticker", "canonical_ticker", "close", "turnover", "liquidity_ready"])
    prices = pd.concat(frames, ignore_index=True, sort=False).dropna(subset=["signal_date", "close"])
    prices = prices.sort_values(["ticker", "signal_date"]).drop_duplicates(["ticker", "signal_date"], keep="last")
    prices["ma5"] = prices.groupby("ticker")["close"].transform(lambda s: s.rolling(5, min_periods=5).mean())
    prices["ma20"] = prices.groupby("ticker")["close"].transform(lambda s: s.rolling(20, min_periods=20).mean())
    prices["ma60"] = prices.groupby("ticker")["close"].transform(lambda s: s.rolling(60, min_periods=60).mean())
    prices["ma120"] = prices.groupby("ticker")["close"].transform(lambda s: s.rolling(120, min_periods=120).mean())
    prices["ma5_slope_5d"] = prices.groupby("ticker")["ma5"].transform(lambda s: (s / s.shift(5) - 1.0) * 100.0)
    prices["ma20_slope_10d"] = prices.groupby("ticker")["ma20"].transform(lambda s: (s / s.shift(10) - 1.0) * 100.0)
    prices["ma60_slope_20d"] = prices.groupby("ticker")["ma60"].transform(lambda s: (s / s.shift(20) - 1.0) * 100.0)
    prices["high20"] = prices.groupby("ticker")["close"].transform(lambda s: s.rolling(20, min_periods=20).max())
    prices["high60"] = prices.groupby("ticker")["close"].transform(lambda s: s.rolling(60, min_periods=60).max())
    prices["high120"] = prices.groupby("ticker")["close"].transform(lambda s: s.rolling(120, min_periods=120).max())
    prices["turnover_60d_median"] = prices.groupby("ticker")["turnover"].transform(lambda s: s.rolling(60, min_periods=20).median())
    prices["ret20"] = prices.groupby("ticker")["close"].transform(lambda s: (s / s.shift(20) - 1.0) * 100.0)
    prices["ret60"] = prices.groupby("ticker")["close"].transform(lambda s: (s / s.shift(60) - 1.0) * 100.0)
    return prices


def _load_benchmark_panel(root: Path) -> pd.DataFrame:
    frames = []
    for key, rel in BENCHMARK_PRICE_PATHS.items():
        path = root / rel
        if not path.exists():
            continue
        frame = pd.read_csv(path, usecols=["date", "close"]).dropna()
        frame["signal_date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame = frame.dropna(subset=["signal_date"])
        frame[f"{key}_close"] = pd.to_numeric(frame["close"], errors="coerce")
        frame = frame.sort_values("signal_date")
        frame[f"{key}_ret20"] = (frame[f"{key}_close"] / frame[f"{key}_close"].shift(20) - 1.0) * 100.0
        frame[f"{key}_ret60"] = (frame[f"{key}_close"] / frame[f"{key}_close"].shift(60) - 1.0) * 100.0
        frames.append(frame[["signal_date", f"{key}_ret20", f"{key}_ret60"]])
    if not frames:
        return pd.DataFrame(columns=["signal_date", "0050_ret20", "0050_ret60", "00631L_ret20", "00631L_ret60"])
    out = frames[0]
    for frame in frames[1:]:
        out = out.merge(frame, on="signal_date", how="outer")
    return out


def _build_daily_feature_panel(candidates: pd.DataFrame, prices: pd.DataFrame, benchmark: pd.DataFrame) -> pd.DataFrame:
    rows = []
    candidate_cols = [
        "ticker",
        "candidate_month",
        "candidate_as_of_date",
        "window_end_exclusive",
        "canonical_ticker",
        "candidate_name",
        "candidate_rank",
        "candidate_score",
        "candidate_layer",
        "market",
    ]
    for ticker, ticker_prices in prices.groupby("ticker", sort=False):
        ticker_candidates = candidates[candidates["ticker"].eq(ticker)][candidate_cols].copy()
        if ticker_candidates.empty:
            continue
        joined = pd.merge_asof(
            ticker_prices.sort_values("signal_date"),
            ticker_candidates.sort_values("candidate_as_of_date"),
            left_on="signal_date",
            right_on="candidate_as_of_date",
            by="ticker",
            direction="backward",
            allow_exact_matches=True,
        )
        joined = joined[
            joined["candidate_as_of_date"].notna()
            & joined["window_end_exclusive"].notna()
            & joined["signal_date"].lt(joined["window_end_exclusive"])
        ].copy()
        if not joined.empty:
            rows.append(joined)
    if not rows:
        return pd.DataFrame()
    daily = pd.concat(rows, ignore_index=True, sort=False)
    daily = daily.merge(benchmark, on="signal_date", how="left")
    daily["next_tradable_date"] = daily.groupby("ticker")["signal_date"].shift(-1)
    daily["candidate_name"] = daily.get("candidate_name", "").fillna("").astype(str)
    daily["price_name"] = daily.get("price_name", "").fillna("").astype(str)
    daily["candidate_name"] = daily["candidate_name"].where(daily["candidate_name"].str.len().gt(0), daily["price_name"])
    daily["candidate_name"] = daily["candidate_name"].where(daily["candidate_name"].str.len().gt(0), daily["ticker"].astype(str))
    daily["price_ready"] = daily[["close", "ma5", "ma20", "ma60", "ma120"]].notna().all(axis=1)
    daily["benchmark_0050_ready"] = daily[["0050_ret20", "0050_ret60"]].notna().all(axis=1)
    daily["benchmark_00631l_ready"] = daily[["00631L_ret20", "00631L_ret60"]].notna().all(axis=1)
    daily["close_vs_ma5_pct"] = (daily["close"] / daily["ma5"] - 1.0) * 100.0
    daily["close_vs_ma20_pct"] = (daily["close"] / daily["ma20"] - 1.0) * 100.0
    daily["close_vs_ma60_pct"] = (daily["close"] / daily["ma60"] - 1.0) * 100.0
    daily["new_20d_high"] = daily["close"].ge(daily["high20"])
    daily["new_60d_high"] = daily["close"].ge(daily["high60"])
    daily["near_60d_high_pct"] = (daily["close"] / daily["high60"] - 1.0) * 100.0
    daily["near_120d_high_pct"] = (daily["close"] / daily["high120"] - 1.0) * 100.0
    daily["rs20_vs_0050"] = daily["ret20"] - daily["0050_ret20"]
    daily["rs60_vs_0050"] = daily["ret60"] - daily["0050_ret60"]
    daily["rs20_vs_00631L"] = daily["ret20"] - daily["00631L_ret20"]
    daily["rs60_vs_00631L"] = daily["ret60"] - daily["00631L_ret60"]
    daily["turnover_vs_60d_median"] = daily["turnover"] / daily["turnover_60d_median"]
    daily["candidate_source"] = "dynamic_pool1_benchmark_aware_candidate_contract"
    return daily


def _build_event_contract(daily: pd.DataFrame) -> pd.DataFrame:
    if daily.empty:
        return pd.DataFrame()
    base_ready = (
        daily["price_ready"]
        & daily["liquidity_ready"]
        & daily["benchmark_0050_ready"]
        & daily["benchmark_00631l_ready"]
        & daily["next_tradable_date"].notna()
    )
    variants = [
        (
            "trend_ext_ma_stack_breakout",
            base_ready
            & (daily["close"] > daily["ma5"])
            & (daily["ma5"] > daily["ma20"])
            & (daily["ma20"] > daily["ma60"])
            & (daily["new_20d_high"] | (daily["near_60d_high_pct"] >= -3.0))
            & (daily["rs20_vs_0050"] > 0)
            & (daily["rs20_vs_00631L"] > 0),
        ),
        (
            "trend_ext_slope_acceleration",
            base_ready
            & (daily["ma20_slope_10d"] > 0)
            & (daily["ma60_slope_20d"] > 0)
            & daily["close_vs_ma20_pct"].between(3.0, 15.0, inclusive="both")
            & (daily["turnover_vs_60d_median"] >= 1.2),
        ),
        (
            "trend_ext_new_high_rs_confirm",
            base_ready
            & (daily["new_60d_high"] | (daily["near_120d_high_pct"] >= -5.0))
            & (daily["rs20_vs_0050"] > 0)
            & (daily["rs60_vs_0050"] > 0)
            & ((daily["rs20_vs_00631L"] > 0) | (daily["rs60_vs_00631L"] > 0)),
        ),
    ]
    frames = []
    for variant, mask in variants:
        selected = daily[mask].copy()
        if selected.empty:
            continue
        selected["event_variant"] = variant
        selected["event_variant_role"] = "primary_exact_daily"
        selected["uses_forward_return_as_rule"] = False
        selected["formal_model_changed"] = False
        selected["trade_decision_changed"] = False
        selected["active_in_trade_decision"] = False
        selected["report_changed"] = False
        selected["portfolio_replay_executed"] = False
        frames.append(selected)
    if not frames:
        return pd.DataFrame()
    contract = pd.concat(frames, ignore_index=True, sort=False)
    contract["signal_date"] = contract["signal_date"].dt.strftime("%Y-%m-%d")
    contract["next_tradable_date"] = pd.to_datetime(contract["next_tradable_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    return contract[_event_columns()]


def _event_columns() -> list[str]:
    return [
        "signal_date",
        "next_tradable_date",
        "ticker",
        "candidate_name",
        "candidate_source",
        "candidate_layer",
        "price_ready",
        "liquidity_ready",
        "benchmark_0050_ready",
        "benchmark_00631l_ready",
        "close",
        "ma5",
        "ma20",
        "ma60",
        "ma120",
        "close_vs_ma5_pct",
        "close_vs_ma20_pct",
        "close_vs_ma60_pct",
        "ma5_slope_5d",
        "ma20_slope_10d",
        "ma60_slope_20d",
        "new_20d_high",
        "new_60d_high",
        "near_60d_high_pct",
        "near_120d_high_pct",
        "rs20_vs_0050",
        "rs20_vs_00631L",
        "rs60_vs_0050",
        "rs60_vs_00631L",
        "turnover",
        "turnover_60d_median",
        "turnover_vs_60d_median",
        "event_variant",
        "event_variant_role",
        "uses_forward_return_as_rule",
        "formal_model_changed",
        "trade_decision_changed",
        "active_in_trade_decision",
        "report_changed",
        "portfolio_replay_executed",
    ]


def _variant_summary(contract: pd.DataFrame) -> pd.DataFrame:
    if contract.empty:
        return pd.DataFrame(columns=["event_variant", "event_count", "unique_tickers", "first_signal_date", "last_signal_date"])
    return (
        contract.groupby("event_variant", as_index=False)
        .agg(
            event_count=("ticker", "count"),
            unique_tickers=("ticker", "nunique"),
            first_signal_date=("signal_date", "min"),
            last_signal_date=("signal_date", "max"),
            top_ticker_share=("ticker", lambda s: round(float(s.value_counts(normalize=True).iloc[0]), 6)),
        )
        .sort_values("event_variant")
    )


def _candidate_readiness(candidates: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    observed = daily.groupby(["candidate_month", "ticker"], as_index=False).agg(
        daily_rows=("signal_date", "count"),
        price_ready_rows=("price_ready", "sum"),
        benchmark_0050_ready_rows=("benchmark_0050_ready", "sum"),
        benchmark_00631l_ready_rows=("benchmark_00631l_ready", "sum"),
    )
    out = candidates[["candidate_month", "ticker", "candidate_as_of_date", "candidate_layer"]].copy()
    out = out.merge(observed, on=["candidate_month", "ticker"], how="left").fillna(0)
    out["blocked"] = out["daily_rows"].eq(0) | out["price_ready_rows"].eq(0)
    out["blocked_reason"] = out.apply(_readiness_reason, axis=1)
    return out


def _readiness_reason(row: pd.Series) -> str:
    reasons = []
    if row["daily_rows"] == 0:
        reasons.append("missing_daily_price_rows_after_candidate_as_of")
    if row["price_ready_rows"] == 0:
        reasons.append("missing_ma_price_readiness")
    if row["benchmark_0050_ready_rows"] == 0:
        reasons.append("missing_0050_benchmark_readiness")
    if row["benchmark_00631l_ready_rows"] == 0:
        reasons.append("missing_00631l_benchmark_readiness")
    return ";".join(reasons)


def _benchmark_readiness_audit(daily: pd.DataFrame) -> pd.DataFrame:
    if daily.empty:
        return pd.DataFrame(columns=["year_month", "daily_rows"])
    frame = daily.copy()
    frame["year_month"] = frame["signal_date"].dt.to_period("M").astype(str)
    return (
        frame.groupby("year_month", as_index=False)
        .agg(
            daily_rows=("ticker", "count"),
            benchmark_0050_ready_rows=("benchmark_0050_ready", "sum"),
            benchmark_00631l_ready_rows=("benchmark_00631l_ready", "sum"),
        )
        .sort_values("year_month")
    )


def _future_data_audit(contract: pd.DataFrame) -> pd.DataFrame:
    if contract.empty:
        return pd.DataFrame(columns=["signal_date", "ticker", "event_variant", "future_data_violation", "reason"])
    out = contract[["signal_date", "next_tradable_date", "ticker", "event_variant"]].copy()
    out["future_data_violation"] = False
    out["reason"] = ""
    return out


def _summary(manifest: dict, variant_summary: pd.DataFrame) -> str:
    return "\n".join(
        [
            "# Strong stock trend-extension daily event contract",
            "",
            "本包只建立 trend-extension / breakout exact daily event contract；不跑 portfolio，不改正式模型、日報或交易決策。",
            "",
            f"- candidate rows：{manifest['candidate_rows']}",
            f"- daily feature rows：{manifest['daily_feature_rows']}",
            f"- event rows：{manifest['event_rows']}",
            f"- blocked candidate rows：{manifest['blocked_candidate_rows']}",
            f"- future data violation count：{manifest['future_data_violation_count']}",
            f"- uses cross-section median as primary benchmark：{manifest['uses_cross_section_median_as_primary_benchmark']}",
            "",
            "## Variant summary",
            ("no rows" if variant_summary.empty else variant_summary.to_csv(index=False).strip()),
        ]
    )


def _canonical_ticker(ticker: str, market: str) -> str:
    suffix = ".TWO" if str(market) == "TPEx" else ".TW"
    return f"{_base_ticker(ticker)}{suffix}"


def _base_ticker(ticker: str) -> str:
    return str(ticker).strip().split(".")[0]


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--candidate-contract", default=str(DEFAULT_CANDIDATE_CONTRACT))
    parser.add_argument("--candidate-context", default=str(DEFAULT_CANDIDATE_CONTEXT))
    parser.add_argument("--liquidity-dir", default=str(DEFAULT_LIQUIDITY_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args(argv)
    manifest = run_strong_stock_trend_extension_daily_event_contract(
        repo_root=args.repo_root,
        candidate_contract=args.candidate_contract,
        candidate_context=args.candidate_context,
        liquidity_dir=args.liquidity_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
