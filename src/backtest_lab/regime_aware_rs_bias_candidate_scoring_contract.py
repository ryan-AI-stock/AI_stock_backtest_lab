"""Build regime-aware RS/BIAS candidate scoring contract.

This module produces diagnostic candidate-context rows only.  It does not run a
portfolio replay and does not alter any formal target, report, or trade action.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


TASK_ID = "TASK-BACKTEST-CORE-REGIME-AWARE-RS-BIAS-CANDIDATE-SCORING-CONTRACT-001"
EXPERIMENTS_TASK_ID = "TASK-BACKTEST-EXPERIMENTS-REGIME-AWARE-RS-BIAS-CANDIDATE-SCORING-DIAGNOSTIC-001"
DEFAULT_CANDIDATE_CONTRACT = Path(
    "outputs/dynamic_pool1_benchmark_aware_candidate_contract_20260704/dynamic_pool1_benchmark_aware_candidate_contract.csv"
)
DEFAULT_CANDIDATE_CONTEXT = Path("outputs/dynamic_pool1_candidate_panel_v0_20260704/candidate_pool_by_month.csv")
DEFAULT_PRICE_SOURCE_DIR = Path("backtest_cache/stock_pool_observations")
DEFAULT_OUTPUT_DIR = Path("outputs/regime_aware_rs_bias_candidate_scoring_contract_20260705")
BENCHMARK_PRICE_PATHS = {
    "0050": Path("backtest_cache/stock_pool_observations/0050_TW.csv"),
    "00631L": Path("backtest_cache/stock_pool_observations/00631L_TW.csv"),
}
BRANCH_VARIANTS = [
    "long_strong_rs40_bias_guard",
    "short_cycle_rs20_bias_repair",
    "pullback_prior_strength_bias_repair",
    "fallback_market_bias_context",
]
CASE_TICKERS = ["6669", "2308", "2317", "2454"]
CASE_TRACE_AS_OF_DATE = "2026-06-30"
CASE_NAMES = {"6669": "緯穎", "2308": "台達電", "2317": "鴻海", "2454": "聯發科"}


def run_regime_aware_rs_bias_candidate_scoring_contract(
    *,
    repo_root: str | Path = ".",
    candidate_contract: str | Path = DEFAULT_CANDIDATE_CONTRACT,
    candidate_context: str | Path = DEFAULT_CANDIDATE_CONTEXT,
    liquidity_dir: str | Path = DEFAULT_PRICE_SOURCE_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    output = _resolve(root, output_dir)
    output.mkdir(parents=True, exist_ok=True)

    candidates = _load_candidates(_resolve(root, candidate_contract), _resolve(root, candidate_context))
    prices = _load_candidate_price_features(_resolve(root, liquidity_dir), candidates)
    benchmarks = _load_benchmark_features(root)
    feature_panel = _asof_feature_join(candidates, prices, benchmarks)
    contract = _expand_branch_contract(feature_panel)
    definitions = _branch_variant_definitions()
    readiness = _feature_readiness_audit(feature_panel, contract)
    case_trace = _case_trace(contract)
    future = _future_data_audit(contract)

    contract.to_csv(output / "regime_aware_rs_bias_candidate_contract.csv", index=False, encoding="utf-8-sig")
    definitions.to_csv(output / "branch_variant_definitions.csv", index=False, encoding="utf-8-sig")
    readiness.to_csv(output / "feature_readiness_audit.csv", index=False, encoding="utf-8-sig")
    case_trace.to_csv(output / "case_trace_6669_2308_2317_2454.csv", index=False, encoding="utf-8-sig")
    future.to_csv(output / "future_data_audit.csv", index=False, encoding="utf-8-sig")

    future_count = int(future["future_data_violation"].sum()) if not future.empty else 0
    manifest: dict[str, Any] = {
        "task_id": TASK_ID,
        "status": "completed_regime_aware_rs_bias_candidate_scoring_contract",
        "output_dir": str(output),
        "candidate_contract_source": str(_resolve(root, candidate_contract)),
        "candidate_context_source": str(_resolve(root, candidate_context)),
        "price_feature_source": str(_resolve(root, liquidity_dir)),
        "candidate_rows": int(len(candidates)),
        "feature_panel_rows": int(len(feature_panel)),
        "contract_rows": int(len(contract)),
        "branch_variants": BRANCH_VARIANTS,
        "case_trace_rows": int(len(case_trace)),
        "future_data_violation_count": future_count,
        "uses_forward_return_as_rule": False,
        "portfolio_replay_executed": False,
        "formal_model_changed": False,
        "trade_decision_changed": False,
        "active_in_trade_decision": False,
        "report_changed": False,
        "ready_for_experiments": bool(future_count == 0 and not contract.empty),
        "handoff_to_experiments_task": EXPERIMENTS_TASK_ID,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "final_summary_zh.md").write_text(_summary(manifest, readiness), encoding="utf-8")
    pd.DataFrame([{"task_id": TASK_ID, "status": "completed", "output_dir": str(output)}]).to_csv(
        output / "completed.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(columns=["task_id", "status", "reason"]).to_csv(output / "failed.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(
        [
            {"step": "load_candidate_contract", "status": "completed"},
            {"step": "load_pit_price_and_benchmark_features", "status": "completed"},
            {"step": "compute_regime_aware_rs_bias_features", "status": "completed"},
            {"step": "write_contract_package", "status": "completed"},
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
    candidates["ticker"] = candidates["ticker"].astype(str).map(_base_ticker)
    candidates["as_of_date"] = pd.to_datetime(candidates["candidate_as_of_date"], errors="coerce")
    candidates = candidates.dropna(subset=["as_of_date"]).copy()
    candidates["candidate_rank"] = pd.to_numeric(candidates.get("candidate_rank"), errors="coerce")
    candidates["candidate_score"] = pd.to_numeric(candidates.get("candidate_score"), errors="coerce")
    for col in [
        "ret_20d_vs_0050_trailing",
        "ret_60d_vs_0050_trailing",
        "ret_20d_vs_00631L_trailing",
        "ret_60d_vs_00631L_trailing",
    ]:
        if col in candidates.columns:
            candidates[col] = pd.to_numeric(candidates[col], errors="coerce")
    if context_path.exists():
        context = pd.read_csv(
            context_path,
            usecols=lambda col: col
            in {
                "year_month",
                "ticker",
                "name",
                "market",
                "candidate_layer",
                "selected_for_pool_v0",
                "ai_supply_chain_layers",
                "mainline_theme_labels",
            },
        ).fillna("")
        context = context.rename(columns={"year_month": "candidate_month", "name": "candidate_name"})
        context["ticker"] = context["ticker"].astype(str).map(_base_ticker)
        candidates = candidates.merge(
            context[
                [
                    "candidate_month",
                    "ticker",
                    "candidate_name",
                    "market",
                    "selected_for_pool_v0",
                    "ai_supply_chain_layers",
                    "mainline_theme_labels",
                ]
            ],
            on=["candidate_month", "ticker"],
            how="left",
        )
    for col in ["candidate_name", "market", "selected_for_pool_v0", "ai_supply_chain_layers", "mainline_theme_labels"]:
        if col not in candidates.columns:
            candidates[col] = ""
        candidates[col] = candidates[col].fillna("").astype(str)
    if "candidate_layer" not in candidates.columns:
        candidates["candidate_layer"] = ""
    candidates["candidate_layer"] = candidates["candidate_layer"].fillna("").astype(str)
    candidates["case_trace_only"] = False
    candidates = _append_case_trace_candidates(candidates)
    return candidates.sort_values(["ticker", "as_of_date", "candidate_rank"]).reset_index(drop=True)


def _append_case_trace_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    existing = set(zip(candidates["ticker"].astype(str), candidates["as_of_date"].dt.strftime("%Y-%m-%d")))
    rows = []
    for ticker in CASE_TICKERS:
        if (ticker, CASE_TRACE_AS_OF_DATE) in existing:
            continue
        rows.append(
            {
                "candidate_month": CASE_TRACE_AS_OF_DATE[:7],
                "candidate_as_of_date": CASE_TRACE_AS_OF_DATE,
                "ticker": ticker,
                "as_of_date": pd.Timestamp(CASE_TRACE_AS_OF_DATE),
                "candidate_rank": pd.NA,
                "candidate_score": pd.NA,
                "candidate_layer": "case_trace_only",
                "candidate_name": CASE_NAMES.get(ticker, ticker),
                "market": "TWSE",
                "selected_for_pool_v0": False,
                "ai_supply_chain_layers": "",
                "mainline_theme_labels": "",
                "case_trace_only": True,
            }
        )
    if not rows:
        return candidates
    return pd.concat([candidates, pd.DataFrame(rows)], ignore_index=True, sort=False)


def _load_candidate_price_features(liquidity_dir: Path, candidates: pd.DataFrame) -> pd.DataFrame:
    tickers = set(candidates["ticker"].astype(str)) | set(CASE_TICKERS)
    frames = []
    shard_dir = liquidity_dir / "shards"
    if shard_dir.exists():
        for shard in sorted(shard_dir.glob("accepted_liquidity_rows_*.csv")):
            try:
                frame = pd.read_csv(
                    shard,
                    usecols=lambda col: col
                    in {"date", "ticker", "name", "market", "close", "turnover_value", "liquidity_pass"},
                )
            except (OSError, ValueError):
                continue
            frame["ticker"] = frame["ticker"].astype(str).map(_base_ticker)
            frame = frame[frame["ticker"].isin(tickers)].copy()
            if frame.empty:
                continue
            frame["price_date"] = pd.to_datetime(frame["date"], errors="coerce")
            frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
            frame["turnover_value"] = pd.to_numeric(frame.get("turnover_value", 0), errors="coerce")
            frame["liquidity_ready"] = frame.get("liquidity_pass", True).map(_as_bool)
            frames.append(frame[["price_date", "ticker", "name", "market", "close", "turnover_value", "liquidity_ready"]])
    else:
        for ticker in sorted(tickers):
            for suffix, market in [("_TW.csv", "TWSE"), ("_TWO.csv", "TPEx")]:
                path = liquidity_dir / f"{ticker}{suffix}"
                if not path.exists():
                    continue
                try:
                    frame = pd.read_csv(path, usecols=lambda col: col in {"date", "close", "adj_close", "volume"})
                except (OSError, ValueError):
                    continue
                price_col = "adj_close" if "adj_close" in frame.columns else "close"
                frame["price_date"] = pd.to_datetime(frame["date"], errors="coerce")
                frame["close"] = pd.to_numeric(frame[price_col], errors="coerce")
                volume = pd.to_numeric(frame.get("volume", 0), errors="coerce").fillna(0.0)
                raw_close = pd.to_numeric(frame.get("close", frame[price_col]), errors="coerce").fillna(frame["close"])
                frame["turnover_value"] = raw_close * volume
                frame["ticker"] = ticker
                frame["name"] = ""
                frame["market"] = market
                frame["liquidity_ready"] = True
                frames.append(frame[["price_date", "ticker", "name", "market", "close", "turnover_value", "liquidity_ready"]])
                break
    if not frames:
        return pd.DataFrame()
    prices = pd.concat(frames, ignore_index=True, sort=False).dropna(subset=["price_date", "close"])
    prices = prices.sort_values(["ticker", "price_date"]).drop_duplicates(["ticker", "price_date"], keep="last")
    grouped = prices.groupby("ticker", group_keys=False)
    for window in [5, 10, 20, 40, 60]:
        prices[f"ret{window}_pct"] = grouped["close"].transform(lambda s, w=window: (s / s.shift(w) - 1.0) * 100.0)
    for window in [20, 60, 120]:
        prices[f"ma{window}"] = grouped["close"].transform(lambda s, w=window: s.rolling(w, min_periods=w).mean())
        prices[f"stock_bias{window}_pct"] = (prices["close"] / prices[f"ma{window}"] - 1.0) * 100.0
    prices["high60"] = grouped["close"].transform(lambda s: s.rolling(60, min_periods=40).max())
    prices["high120"] = grouped["close"].transform(lambda s: s.rolling(120, min_periods=80).max())
    prices["drawdown_from_60d_high_pct"] = (prices["close"] / prices["high60"] - 1.0) * 100.0
    prices["near_120d_high_pct"] = (prices["close"] / prices["high120"] - 1.0) * 100.0
    return prices.reset_index(drop=True)


def _load_benchmark_features(root: Path) -> pd.DataFrame:
    frames = []
    for key, rel_path in BENCHMARK_PRICE_PATHS.items():
        path = root / rel_path
        if not path.exists():
            continue
        frame = pd.read_csv(path, usecols=lambda col: col in {"date", "close", "adj_close"}).dropna(subset=["date"])
        frame["benchmark_price_date"] = pd.to_datetime(frame["date"], errors="coerce")
        price_col = "adj_close" if "adj_close" in frame.columns else "close"
        frame[f"{key}_close"] = pd.to_numeric(frame[price_col], errors="coerce")
        frame = frame.dropna(subset=["benchmark_price_date", f"{key}_close"]).sort_values("benchmark_price_date")
        for window in [5, 10, 20, 40, 60]:
            frame[f"{key}_ret{window}_pct"] = (frame[f"{key}_close"] / frame[f"{key}_close"].shift(window) - 1.0) * 100.0
        for window in [20, 60, 120]:
            ma = frame[f"{key}_close"].rolling(window, min_periods=window).mean()
            frame[f"{key}_bias{window}_pct"] = (frame[f"{key}_close"] / ma - 1.0) * 100.0
        keep = ["benchmark_price_date"] + [col for col in frame.columns if col.startswith(f"{key}_ret") or col.startswith(f"{key}_bias")]
        frames.append(frame[keep])
    if not frames:
        return pd.DataFrame()
    out = frames[0]
    for frame in frames[1:]:
        out = out.merge(frame, on="benchmark_price_date", how="outer")
    return out.sort_values("benchmark_price_date")


def _asof_feature_join(candidates: pd.DataFrame, prices: pd.DataFrame, benchmarks: pd.DataFrame) -> pd.DataFrame:
    if prices.empty:
        base = candidates.copy()
        base["price_date"] = pd.NaT
    else:
        rows = []
        for ticker, ticker_candidates in candidates.groupby("ticker", sort=False):
            ticker_prices = prices[prices["ticker"].eq(ticker)].sort_values("price_date")
            if ticker_prices.empty:
                missing = ticker_candidates.copy()
                missing["price_date"] = pd.NaT
                rows.append(missing)
                continue
            rows.append(
                pd.merge_asof(
                    ticker_candidates.sort_values("as_of_date"),
                    ticker_prices,
                    left_on="as_of_date",
                    right_on="price_date",
                    by="ticker",
                    direction="backward",
                    allow_exact_matches=True,
                )
            )
        base = pd.concat(rows, ignore_index=True, sort=False)
    if not benchmarks.empty:
        base = pd.merge_asof(
            base.sort_values("as_of_date"),
            benchmarks.sort_values("benchmark_price_date"),
            left_on="as_of_date",
            right_on="benchmark_price_date",
            direction="backward",
            allow_exact_matches=True,
        ).sort_values(["ticker", "as_of_date", "candidate_rank"])
    else:
        base["benchmark_price_date"] = pd.NaT
    if "market" not in base.columns:
        base["market"] = ""
    if "market_x" in base.columns:
        base["market"] = base["market_x"].where(base["market_x"].fillna("").astype(str).str.len().gt(0), base["market"])
    if "market_y" in base.columns:
        base["market"] = base["market"].where(base["market"].fillna("").astype(str).str.len().gt(0), base["market_y"])
    if "name" not in base.columns:
        base["name"] = ""
    base = _add_bias_distribution_context(base, prices)
    base = _derive_contract_features(base)
    return base.reset_index(drop=True)


def _add_bias_distribution_context(base: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    out = base.copy()
    out["stock_bias20_percentile"] = pd.NA
    out["stock_bias60_percentile"] = pd.NA
    out["stock_bias60_zscore"] = pd.NA
    if prices.empty or "price_date" not in out.columns:
        return out
    history_by_ticker = {
        ticker: frame.sort_values("price_date").reset_index(drop=True)
        for ticker, frame in prices.groupby("ticker", sort=False)
    }
    for idx, row in out.iterrows():
        ticker = row.get("ticker")
        price_date = row.get("price_date")
        if pd.isna(price_date) or ticker not in history_by_ticker:
            continue
        history = history_by_ticker[ticker]
        history = history[history["price_date"].le(price_date)].tail(252)
        out.at[idx, "stock_bias20_percentile"] = _point_percentile(history["stock_bias20_pct"])
        out.at[idx, "stock_bias60_percentile"] = _point_percentile(history["stock_bias60_pct"])
        out.at[idx, "stock_bias60_zscore"] = _point_zscore(history["stock_bias60_pct"])
    return out


def _derive_contract_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for window in [5, 10, 20, 40, 60]:
        out[f"rs{window}_vs_0050_pct"] = out.get(f"ret{window}_pct") - out.get(f"0050_ret{window}_pct")
        if window in {20, 40, 60}:
            out[f"rs{window}_vs_00631L_pct"] = out.get(f"ret{window}_pct") - out.get(f"00631L_ret{window}_pct")
    out["rs20_vs_0050_pct"] = out["rs20_vs_0050_pct"].fillna(out.get("ret_20d_vs_0050_trailing"))
    out["rs60_vs_0050_pct"] = out["rs60_vs_0050_pct"].fillna(out.get("ret_60d_vs_0050_trailing"))
    out["rs20_vs_00631L_pct"] = out["rs20_vs_00631L_pct"].fillna(out.get("ret_20d_vs_00631L_trailing"))
    out["rs60_vs_00631L_pct"] = out["rs60_vs_00631L_pct"].fillna(out.get("ret_60d_vs_00631L_trailing"))
    out["market_bias20_0050"] = out.get("0050_bias20_pct")
    out["market_bias60_0050"] = out.get("0050_bias60_pct")
    out["market_bias120_0050"] = out.get("0050_bias120_pct")
    out["market_regime_0050"] = out.apply(_market_regime, axis=1)
    out["prior_strength_flag"] = (
        out["rs60_vs_0050_pct"].gt(0)
        & out["near_120d_high_pct"].ge(-10)
        & out["stock_bias60_percentile"].fillna(1).le(0.9)
    )
    out["rs5_repair_flag"] = out["rs5_vs_0050_pct"].gt(0) & out["stock_bias20_pct"].gt(out["stock_bias60_pct"])
    out["rs10_repair_flag"] = out["rs10_vs_0050_pct"].gt(0) & out["stock_bias20_pct"].gt(out["stock_bias60_pct"])
    out["rs20_repair_flag"] = out["rs20_vs_0050_pct"].gt(0) & out["stock_bias20_percentile"].fillna(1).between(0.2, 0.75)
    out["price_ready"] = out[["price_date", "close", "stock_bias20_pct", "stock_bias60_pct"]].notna().all(axis=1)
    out["benchmark_ready"] = out[["benchmark_price_date", "0050_ret20_pct", "00631L_ret20_pct"]].notna().all(axis=1)
    out["bias_feature_ready"] = out[["stock_bias20_percentile", "stock_bias60_percentile", "stock_bias60_zscore"]].notna().all(axis=1)
    out["feature_blocked_reason"] = out.apply(_feature_blocked_reason, axis=1)
    return out


def _expand_branch_contract(feature_panel: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in feature_panel.iterrows():
        for branch in BRANCH_VARIANTS:
            selected, label, score, components = _score_branch(row, branch)
            rows.append(
                {
                    "ticker": _canonical_ticker(row.get("ticker"), row.get("market")),
                    "base_ticker": row.get("ticker", ""),
                    "candidate_name": row.get("candidate_name") or row.get("name", ""),
                    "candidate_month": row.get("candidate_month", ""),
                    "as_of_date": _date_str(row.get("as_of_date")),
                    "price_as_of_date": _date_str(row.get("price_date")),
                    "benchmark_as_of_date": _date_str(row.get("benchmark_price_date")),
                    "market_regime_0050": row.get("market_regime_0050", "unknown"),
                    "candidate_rank": row.get("candidate_rank", pd.NA),
                    "candidate_score": row.get("candidate_score", pd.NA),
                    "candidate_layer": row.get("candidate_layer", ""),
                    "case_trace_only": bool(row.get("case_trace_only", False)),
                    "ai_supply_chain_layers": row.get("ai_supply_chain_layers", ""),
                    "mainline_theme_labels": row.get("mainline_theme_labels", ""),
                    "rs5_vs_0050_pct": _round(row.get("rs5_vs_0050_pct")),
                    "rs10_vs_0050_pct": _round(row.get("rs10_vs_0050_pct")),
                    "rs20_vs_0050_pct": _round(row.get("rs20_vs_0050_pct")),
                    "rs40_vs_0050_pct": _round(row.get("rs40_vs_0050_pct")),
                    "rs60_vs_0050_pct": _round(row.get("rs60_vs_0050_pct")),
                    "rs20_vs_00631L_pct": _round(row.get("rs20_vs_00631L_pct")),
                    "rs40_vs_00631L_pct": _round(row.get("rs40_vs_00631L_pct")),
                    "rs60_vs_00631L_pct": _round(row.get("rs60_vs_00631L_pct")),
                    "stock_bias20_pct": _round(row.get("stock_bias20_pct")),
                    "stock_bias60_pct": _round(row.get("stock_bias60_pct")),
                    "stock_bias120_pct": _round(row.get("stock_bias120_pct")),
                    "stock_bias20_percentile": _round(row.get("stock_bias20_percentile")),
                    "stock_bias60_percentile": _round(row.get("stock_bias60_percentile")),
                    "stock_bias60_zscore": _round(row.get("stock_bias60_zscore")),
                    "market_bias20_0050": _round(row.get("market_bias20_0050")),
                    "market_bias60_0050": _round(row.get("market_bias60_0050")),
                    "market_bias120_0050": _round(row.get("market_bias120_0050")),
                    "prior_strength_flag": bool(row.get("prior_strength_flag", False)),
                    "rs5_repair_flag": bool(row.get("rs5_repair_flag", False)),
                    "rs10_repair_flag": bool(row.get("rs10_repair_flag", False)),
                    "rs20_repair_flag": bool(row.get("rs20_repair_flag", False)),
                    "branch_variant": branch,
                    "branch_candidate_label": label,
                    "branch_candidate_selected": selected,
                    "branch_score": _round(score),
                    "branch_score_components": json.dumps(components, ensure_ascii=False, sort_keys=True),
                    "feature_blocked_reason": row.get("feature_blocked_reason", ""),
                    "uses_forward_return_as_rule": False,
                    "formal_model_changed": False,
                    "trade_decision_changed": False,
                    "active_in_trade_decision": False,
                    "report_changed": False,
                    "portfolio_replay_executed": False,
                }
            )
    return pd.DataFrame(rows)


def _score_branch(row: pd.Series, branch: str) -> tuple[bool, str, float, dict[str, Any]]:
    price_ready = bool(row.get("price_ready", False))
    benchmark_ready = bool(row.get("benchmark_ready", False))
    bias_ready = bool(row.get("bias_feature_ready", False))
    regime = row.get("market_regime_0050", "unknown")
    components = {
        "price_ready": price_ready,
        "benchmark_ready": benchmark_ready,
        "bias_ready": bias_ready,
        "market_regime_0050": regime,
    }
    if not (price_ready and benchmark_ready and bias_ready):
        return False, "data_not_ready", 0.0, components
    if branch == "long_strong_rs40_bias_guard":
        score = (
            _num(row.get("rs40_vs_0050_pct")) * 0.5
            + _num(row.get("rs60_vs_0050_pct")) * 0.25
            - max(0.0, _num(row.get("stock_bias60_zscore")) - 1.5) * 5.0
        )
        selected = regime == "long_strong" and row.get("rs40_vs_0050_pct", 0) > 0 and _num(row.get("stock_bias60_percentile")) <= 0.9
        label = "candidate_context" if selected else "long_strong_context_not_selected"
    elif branch == "short_cycle_rs20_bias_repair":
        score = _num(row.get("rs20_vs_0050_pct")) + 2.0 * int(row.get("rs5_repair_flag", False)) + 2.0 * int(row.get("rs10_repair_flag", False))
        selected = regime in {"short_cycle_rotation", "ordinary_or_no_edge"} and row.get("rs20_vs_0050_pct", 0) > 0 and (
            row.get("rs5_repair_flag", False) or row.get("rs10_repair_flag", False)
        )
        label = "candidate_context" if selected else "short_cycle_context_not_selected"
    elif branch == "pullback_prior_strength_bias_repair":
        score = (
            _num(row.get("rs20_vs_0050_pct"))
            + 5.0 * int(row.get("prior_strength_flag", False))
            + 2.0 * int(row.get("rs20_repair_flag", False))
            - abs(_num(row.get("stock_bias20_percentile")) - 0.35) * 2.0
        )
        selected = bool(row.get("prior_strength_flag", False)) and _num(row.get("stock_bias20_percentile")) <= 0.55 and (
            row.get("rs5_repair_flag", False) or row.get("rs10_repair_flag", False) or row.get("rs20_repair_flag", False)
        )
        label = "candidate_context" if selected else "pullback_context_not_selected"
    else:
        score = _num(row.get("market_bias20_0050")) + _num(row.get("market_bias60_0050")) * 0.5
        selected = False
        label = "fallback_risk_context_only"
    components.update(
        {
            "rs20_vs_0050_pct": _round(row.get("rs20_vs_0050_pct")),
            "rs40_vs_0050_pct": _round(row.get("rs40_vs_0050_pct")),
            "rs60_vs_0050_pct": _round(row.get("rs60_vs_0050_pct")),
            "stock_bias20_percentile": _round(row.get("stock_bias20_percentile")),
            "stock_bias60_percentile": _round(row.get("stock_bias60_percentile")),
            "stock_bias60_zscore": _round(row.get("stock_bias60_zscore")),
            "prior_strength_flag": bool(row.get("prior_strength_flag", False)),
            "rs5_repair_flag": bool(row.get("rs5_repair_flag", False)),
            "rs10_repair_flag": bool(row.get("rs10_repair_flag", False)),
            "rs20_repair_flag": bool(row.get("rs20_repair_flag", False)),
        }
    )
    return bool(selected), label, float(score), components


def _branch_variant_definitions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "branch_variant": "long_strong_rs40_bias_guard",
                "branch_role": "primary_long_strong_context",
                "required_regime": "long_strong",
                "live_safe_features": "RS40/RS60 vs 0050/00631L; stock BIAS60 percentile/zscore",
                "purpose": "Long-strong candidate context while avoiding extreme overheat chasing.",
                "uses_forward_return_as_rule": False,
            },
            {
                "branch_variant": "short_cycle_rs20_bias_repair",
                "branch_role": "short_cycle_context",
                "required_regime": "short_cycle_rotation_or_ordinary",
                "live_safe_features": "RS20 plus RS5/RS10 repair flags; stock BIAS20/60 normalization",
                "purpose": "Faster candidate context, not standalone action.",
                "uses_forward_return_as_rule": False,
            },
            {
                "branch_variant": "pullback_prior_strength_bias_repair",
                "branch_role": "pullback_turning_context",
                "required_regime": "any_non_blocked_context",
                "live_safe_features": "prior strength/high proximity; BIAS pullback percentile; RS5/10/20 repair",
                "purpose": "Lowpoint or turning-point candidate context.",
                "uses_forward_return_as_rule": False,
            },
            {
                "branch_variant": "fallback_market_bias_context",
                "branch_role": "fallback_risk_context_only",
                "required_regime": "market_context_only",
                "live_safe_features": "0050 BIAS20/60/120 and 00631L RS context",
                "purpose": "Fallback risk context only; does not alter fallback mapping.",
                "uses_forward_return_as_rule": False,
            },
        ]
    )


def _feature_readiness_audit(feature_panel: pd.DataFrame, contract: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature, cols in {
        "price_features": ["price_date", "close", "stock_bias20_pct", "stock_bias60_pct"],
        "benchmark_rs_features": ["benchmark_price_date", "0050_ret20_pct", "00631L_ret20_pct"],
        "bias_percentile_features": ["stock_bias20_percentile", "stock_bias60_percentile", "stock_bias60_zscore"],
        "branch_contract_rows": ["branch_candidate_label"],
    }.items():
        source = contract if feature == "branch_contract_rows" else feature_panel
        ready = source[cols].notna().all(axis=1) if all(col in source.columns for col in cols) else pd.Series([False] * len(source))
        rows.append(
            {
                "feature_group": feature,
                "rows": int(len(source)),
                "ready_rows": int(ready.sum()) if len(source) else 0,
                "ready_rate": round(float(ready.mean()), 6) if len(source) else 0.0,
                "blocked_rows": int((~ready).sum()) if len(source) else 0,
            }
        )
    if not contract.empty:
        by_branch = contract.groupby("branch_variant")["branch_candidate_selected"].agg(["count", "sum"]).reset_index()
        for _, row in by_branch.iterrows():
            rows.append(
                {
                    "feature_group": f"selected_{row['branch_variant']}",
                    "rows": int(row["count"]),
                    "ready_rows": int(row["sum"]),
                    "ready_rate": round(float(row["sum"]) / float(row["count"]), 6) if row["count"] else 0.0,
                    "blocked_rows": int(row["count"] - row["sum"]),
                }
            )
    return pd.DataFrame(rows)


def _case_trace(contract: pd.DataFrame) -> pd.DataFrame:
    if contract.empty:
        return contract
    case = contract[contract["base_ticker"].astype(str).isin(CASE_TICKERS)].copy()
    if case.empty:
        return case
    latest_dates = case.groupby("base_ticker")["as_of_date"].transform("max")
    return case[case["as_of_date"].eq(latest_dates)].sort_values(["base_ticker", "branch_variant"])


def _future_data_audit(contract: pd.DataFrame) -> pd.DataFrame:
    if contract.empty:
        return pd.DataFrame(
            [{"audit_item": "regime_aware_rs_bias_contract", "rows": 0, "future_data_violation": False, "reason": "empty_contract"}]
        )
    asof = pd.to_datetime(contract["as_of_date"], errors="coerce")
    price_date = pd.to_datetime(contract["price_as_of_date"], errors="coerce")
    benchmark_date = pd.to_datetime(contract["benchmark_as_of_date"], errors="coerce")
    return pd.DataFrame(
        [
            {
                "audit_item": "stock_price_asof_not_future",
                "rows": int(len(contract)),
                "future_data_violation": bool((price_date.dropna() > asof.loc[price_date.dropna().index]).any()),
                "reason": "price_as_of_date must be <= as_of_date",
            },
            {
                "audit_item": "benchmark_asof_not_future",
                "rows": int(len(contract)),
                "future_data_violation": bool((benchmark_date.dropna() > asof.loc[benchmark_date.dropna().index]).any()),
                "reason": "benchmark_as_of_date must be <= as_of_date",
            },
            {
                "audit_item": "forward_return_not_used_as_rule",
                "rows": int(len(contract)),
                "future_data_violation": bool(contract["uses_forward_return_as_rule"].any()),
                "reason": "contract uses trailing price, RS, BIAS, and market context only",
            },
        ]
    )


def _point_percentile(series: pd.Series) -> float | Any:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if len(clean) < 40:
        return pd.NA
    current = clean.iloc[-1]
    return float((clean <= current).mean())


def _point_zscore(series: pd.Series) -> float | Any:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if len(clean) < 40:
        return pd.NA
    std = clean.std()
    if not std:
        return pd.NA
    return float((clean.iloc[-1] - clean.mean()) / std)


def _market_regime(row: pd.Series) -> str:
    bias20 = _num(row.get("0050_bias20_pct"))
    bias60 = _num(row.get("0050_bias60_pct"))
    bias120 = _num(row.get("0050_bias120_pct"))
    if pd.isna(row.get("0050_bias60_pct")) or pd.isna(row.get("0050_bias120_pct")):
        return "unknown"
    if bias60 >= 8 and bias120 >= 10:
        return "long_strong"
    if bias60 <= -5 or bias120 <= -8:
        return "defensive_or_bear"
    if bias20 > 0 and -5 < bias60 < 8:
        return "short_cycle_rotation"
    return "ordinary_or_no_edge"


def _feature_blocked_reason(row: pd.Series) -> str:
    reasons = []
    if not bool(row.get("price_ready", False)):
        reasons.append("price_or_stock_bias_feature_not_ready")
    if not bool(row.get("benchmark_ready", False)):
        reasons.append("benchmark_rs_feature_not_ready")
    if not bool(row.get("bias_feature_ready", False)):
        reasons.append("stock_specific_bias_percentile_not_ready")
    return ";".join(reasons)


def _base_ticker(value: object) -> str:
    text = str(value).strip()
    if "." in text:
        return text.split(".", 1)[0]
    return text


def _canonical_ticker(ticker: object, market: object) -> str:
    base = _base_ticker(ticker)
    market_text = str(market or "").upper()
    if base in {"0050", "00631L"}:
        return f"{base}.TW"
    if "TPEX" in market_text or "OTC" in market_text or "TWO" in market_text:
        return f"{base}.TWO"
    return f"{base}.TW"


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _round(value: object, digits: int = 6) -> float | Any:
    if pd.isna(value):
        return pd.NA
    return round(float(value), digits)


def _num(value: object) -> float:
    if pd.isna(value):
        return 0.0
    return float(value)


def _date_str(value: object) -> str:
    if pd.isna(value):
        return ""
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _summary(manifest: dict[str, Any], readiness: pd.DataFrame) -> str:
    selected = readiness[readiness["feature_group"].str.startswith("selected_")].copy() if not readiness.empty else pd.DataFrame()
    lines = [
        "# Regime-aware RS/BIAS candidate scoring contract",
        "",
        "## 結論",
        "",
        "- 本包只建立 candidate scoring / feature readiness contract，沒有 portfolio replay，也沒有改 formal/report/trade。",
        f"- contract rows：{manifest['contract_rows']}",
        f"- candidate rows：{manifest['candidate_rows']}",
        f"- case trace rows：{manifest['case_trace_rows']}",
        f"- future_data_violation_count：{manifest['future_data_violation_count']}",
        f"- ready_for_experiments：{manifest['ready_for_experiments']}",
        "",
        "## Branch selected rows",
        "",
    ]
    if selected.empty:
        lines.append("- no selected branch rows")
    else:
        for _, row in selected.iterrows():
            lines.append(f"- {row['feature_group']}: {int(row['ready_rows'])} / {int(row['rows'])}")
    lines.extend(
        [
            "",
            "## 邊界",
            "",
            "- RS60 不再作 universal hard gate；此處只輸出依 regime / branch 的 diagnostic context。",
            "- fallback_market_bias_context 只作 fallback risk context，不改 fallback mapping。",
            "- uses_forward_return_as_rule=false；所有 rule 欄位只用 trailing RS / BIAS / price context。",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build regime-aware RS/BIAS candidate scoring contract.")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--candidate-contract", default=str(DEFAULT_CANDIDATE_CONTRACT))
    parser.add_argument("--candidate-context", default=str(DEFAULT_CANDIDATE_CONTEXT))
    parser.add_argument("--liquidity-dir", default=str(DEFAULT_PRICE_SOURCE_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()
    manifest = run_regime_aware_rs_bias_candidate_scoring_contract(
        repo_root=args.repo_root,
        candidate_contract=args.candidate_contract,
        candidate_context=args.candidate_context,
        liquidity_dir=args.liquidity_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
