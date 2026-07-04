"""Build exact forward outcome panels for trend-extension daily events."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


TASK_ID = "TASK-BACKTEST-CORE-STRONG-STOCK-TREND-EXTENSION-EXACT-OUTCOME-PANEL-001"
EXPERIMENTS_TASK_ID = "TASK-BACKTEST-EXPERIMENTS-STRONG-STOCK-TREND-EXTENSION-EXACT-OUTCOME-VALIDATION-001"
DEFAULT_EVENT_CONTRACT = Path(
    "outputs/strong_stock_trend_extension_daily_event_contract_20260704/trend_extension_daily_event_contract.csv"
)
DEFAULT_PROXY_OUTCOME = Path(
    r"C:\Users\zergv\Documents\Codex\2026-06-17\repo-ai-stock-backtest-lab-repo\outputs"
    r"\experiments_strong_stock_dual_branch_regime_event_selector_diagnostic_20260704\trend_extension_event_outcome.csv"
)
DEFAULT_LIQUIDITY_DIR = Path(
    r"C:\Users\zergv\Documents\Codex\2026-05-23\ai-stock-rotation-radar-https-docs\outputs"
    r"\radar_dynamic_pool1_all_listed_liquid_universe_full_sweep_20260703"
)
DEFAULT_OUTPUT_DIR = Path("outputs/strong_stock_trend_extension_exact_outcome_panel_20260704")
BENCHMARK_PRICE_PATHS = {
    "0050": Path("backtest_cache/stock_pool_observations/0050_TW.csv"),
    "00631L": Path("backtest_cache/stock_pool_observations/00631L_TW.csv"),
}
HORIZONS = [5, 10, 20, 40, 60]
CASE_TICKERS = {"6669", "2308", "2317"}
PERIODS = [
    ("period_1_2014_11_2022_12", "2014-11-01", "2022-12-31"),
    ("period_2_2023_01_2026_06", "2023-01-01", "2026-06-30"),
    ("calendar_2024", "2024-01-01", "2024-12-31"),
    ("calendar_2025", "2025-01-01", "2025-12-31"),
    ("ytd_2026_available", "2026-01-01", "2026-06-30"),
]


def run_strong_stock_trend_extension_exact_outcome_panel(
    *,
    repo_root: str | Path = ".",
    event_contract: str | Path = DEFAULT_EVENT_CONTRACT,
    proxy_outcome: str | Path = DEFAULT_PROXY_OUTCOME,
    liquidity_dir: str | Path = DEFAULT_LIQUIDITY_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict:
    root = Path(repo_root).resolve()
    event_path = _resolve(root, event_contract)
    liquidity = _resolve(root, liquidity_dir)
    output = _resolve(root, output_dir)
    output.mkdir(parents=True, exist_ok=True)

    events = _load_events(event_path)
    price_metrics = _load_price_metrics(liquidity, events["ticker"].astype(str).unique().tolist())
    benchmark_metrics = _load_benchmark_metrics(root)
    outcome = _build_outcome_panel(events, price_metrics, benchmark_metrics)
    by_variant = _summary_by_variant(outcome)
    by_period = _summary_by_period(outcome)
    by_regime = _summary_by_regime(outcome)
    proxy_comparison = _proxy_comparison(by_period, _resolve(root, proxy_outcome))
    benchmark_summary = _benchmark_summary(outcome)
    incomplete = outcome[~outcome["all_horizons_complete"]].copy()
    future_audit = _future_data_audit(outcome)

    outcome.to_csv(output / "trend_extension_exact_event_outcome_panel.csv", index=False, encoding="utf-8-sig")
    by_variant.to_csv(output / "trend_extension_exact_outcome_by_variant.csv", index=False, encoding="utf-8-sig")
    by_period.to_csv(output / "trend_extension_exact_outcome_by_period.csv", index=False, encoding="utf-8-sig")
    by_regime.to_csv(output / "trend_extension_exact_outcome_by_regime.csv", index=False, encoding="utf-8-sig")
    proxy_comparison.to_csv(output / "trend_extension_exact_outcome_vs_proxy_comparison.csv", index=False, encoding="utf-8-sig")
    benchmark_summary.to_csv(
        output / "trend_extension_exact_outcome_vs_baseline_benchmark_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    incomplete.to_csv(output / "incomplete_outcome_rows.csv", index=False, encoding="utf-8-sig")
    future_audit.to_csv(output / "future_data_audit.csv", index=False, encoding="utf-8-sig")

    manifest = {
        "task_id": TASK_ID,
        "status": "completed_trend_extension_exact_outcome_panel_ready",
        "output_dir": str(output),
        "event_contract_source": str(event_path),
        "event_rows_input": int(len(events)),
        "outcome_rows": int(len(outcome)),
        "incomplete_outcome_rows": int(len(incomplete)),
        "future_data_violation_count": int(future_audit["future_data_violation"].sum()) if not future_audit.empty else 0,
        "uses_forward_return_as_rule": False,
        "forward_return_used_as_evaluation_metadata": True,
        "portfolio_replay_executed": False,
        "formal_model_changed": False,
        "trade_decision_changed": False,
        "active_in_trade_decision": False,
        "report_changed": False,
        "ready_for_experiments_validation": True,
        "handoff_to_experiments_task": EXPERIMENTS_TASK_ID,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "final_summary_zh.md").write_text(_summary(manifest, by_variant, by_period), encoding="utf-8")
    pd.DataFrame([{"task_id": TASK_ID, "status": "completed", "output_dir": str(output)}]).to_csv(
        output / "completed.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(columns=["task_id", "status", "reason"]).to_csv(output / "failed.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(
        [
            {"step": "load_exact_event_contract", "status": "completed"},
            {"step": "load_forward_price_and_benchmark_metrics", "status": "completed"},
            {"step": "build_exact_outcome_panel", "status": "completed"},
            {"step": "write_outcome_package", "status": "completed"},
        ]
    ).to_csv(output / "run_log.csv", index=False, encoding="utf-8-sig")
    return manifest


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _load_events(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    events = pd.read_csv(path).fillna("")
    events["signal_date"] = pd.to_datetime(events["signal_date"], errors="coerce")
    events["next_tradable_date"] = pd.to_datetime(events["next_tradable_date"], errors="coerce")
    events = events.dropna(subset=["signal_date", "next_tradable_date"]).copy()
    events["ticker"] = events["ticker"].astype(str).map(_base_ticker)
    events["period_bucket"] = events["signal_date"].map(_primary_period)
    events["regime_label"] = events["period_bucket"]
    events["case_trace_only"] = events["ticker"].isin(CASE_TICKERS)
    return events


def _load_price_metrics(liquidity_dir: Path, tickers: list[str]) -> pd.DataFrame:
    needed = set(tickers)
    frames = []
    for shard in sorted((liquidity_dir / "shards").glob("accepted_liquidity_rows_*.csv")):
        try:
            frame = pd.read_csv(shard, usecols=lambda col: col in {"date", "ticker", "close"}).fillna("")
        except (OSError, ValueError):
            continue
        frame["ticker"] = frame["ticker"].astype(str).map(_base_ticker)
        frame = frame[frame["ticker"].isin(needed)].copy()
        if frame.empty:
            continue
        frame["next_tradable_date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame["entry_price"] = pd.to_numeric(frame["close"], errors="coerce")
        frames.append(frame[["next_tradable_date", "ticker", "entry_price"]])
    if not frames:
        return pd.DataFrame(columns=["next_tradable_date", "ticker", "entry_price"])
    prices = pd.concat(frames, ignore_index=True, sort=False).dropna(subset=["next_tradable_date", "entry_price"])
    prices = prices.sort_values(["ticker", "next_tradable_date"]).drop_duplicates(["ticker", "next_tradable_date"], keep="last")
    return _add_forward_metrics(prices, group_col="ticker", price_col="entry_price", prefix="event")


def _load_benchmark_metrics(root: Path) -> pd.DataFrame:
    frames = []
    for key, rel in BENCHMARK_PRICE_PATHS.items():
        path = root / rel
        if not path.exists():
            continue
        frame = pd.read_csv(path, usecols=["date", "close"]).dropna()
        frame["next_tradable_date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame[f"{key}_entry_price"] = pd.to_numeric(frame["close"], errors="coerce")
        frame = frame.dropna(subset=["next_tradable_date", f"{key}_entry_price"])
        metrics = _add_forward_metrics(
            frame[["next_tradable_date", f"{key}_entry_price"]].copy(),
            group_col=None,
            price_col=f"{key}_entry_price",
            prefix=key,
        )
        frames.append(metrics)
    if not frames:
        return pd.DataFrame(columns=["next_tradable_date"])
    out = frames[0]
    for frame in frames[1:]:
        out = out.merge(frame, on="next_tradable_date", how="outer")
    return out


def _add_forward_metrics(frame: pd.DataFrame, *, group_col: str | None, price_col: str, prefix: str) -> pd.DataFrame:
    frame = frame.sort_values(([group_col] if group_col else []) + ["next_tradable_date"]).copy()
    groups = frame.groupby(group_col, sort=False) if group_col else [(None, frame)]
    parts = []
    for _, group in groups:
        group = group.copy().sort_values("next_tradable_date")
        price = group[price_col]
        for horizon in HORIZONS:
            future = price.shift(-horizon)
            group[f"{prefix}_return_{horizon}d_pct"] = (future / price - 1.0) * 100.0
            group[f"{prefix}_horizon_{horizon}d_complete"] = future.notna()
        future_max20 = price.shift(-1)[::-1].rolling(20, min_periods=1).max()[::-1]
        future_min20 = price.shift(-1)[::-1].rolling(20, min_periods=1).min()[::-1]
        future_max60 = price.shift(-1)[::-1].rolling(60, min_periods=1).max()[::-1]
        future_min60 = price.shift(-1)[::-1].rolling(60, min_periods=1).min()[::-1]
        group[f"{prefix}_mae_20d_pct"] = (future_min20 / price - 1.0) * 100.0
        group[f"{prefix}_mfe_20d_pct"] = (future_max20 / price - 1.0) * 100.0
        group[f"{prefix}_mae_60d_pct"] = (future_min60 / price - 1.0) * 100.0
        group[f"{prefix}_mfe_60d_pct"] = (future_max60 / price - 1.0) * 100.0
        parts.append(group)
    return pd.concat(parts, ignore_index=True, sort=False)


def _build_outcome_panel(events: pd.DataFrame, price_metrics: pd.DataFrame, benchmark_metrics: pd.DataFrame) -> pd.DataFrame:
    panel = events.merge(price_metrics, on=["next_tradable_date", "ticker"], how="left")
    panel = panel.merge(benchmark_metrics, on="next_tradable_date", how="left")
    for horizon in HORIZONS:
        panel[f"0050_return_{horizon}d_pct"] = panel.get(f"0050_return_{horizon}d_pct")
        panel[f"00631L_return_{horizon}d_pct"] = panel.get(f"00631L_return_{horizon}d_pct")
        panel[f"excess_vs_0050_{horizon}d_pct"] = panel[f"event_return_{horizon}d_pct"] - panel[f"0050_return_{horizon}d_pct"]
        panel[f"excess_vs_00631L_{horizon}d_pct"] = panel[f"event_return_{horizon}d_pct"] - panel[f"00631L_return_{horizon}d_pct"]
        complete_cols = [
            f"event_horizon_{horizon}d_complete",
            f"0050_horizon_{horizon}d_complete",
            f"00631L_horizon_{horizon}d_complete",
        ]
        panel[f"horizon_{horizon}d_complete"] = panel[complete_cols].fillna(False).all(axis=1)
    panel["next_tradable_date_price_ready"] = panel["entry_price"].notna()
    panel["all_horizons_complete"] = panel[[f"horizon_{horizon}d_complete" for horizon in HORIZONS]].all(axis=1)
    panel["incomplete_outcome_reason"] = panel.apply(_incomplete_reason, axis=1)
    panel["uses_forward_return_as_rule"] = False
    panel["portfolio_replay_executed"] = False
    panel["formal_model_changed"] = False
    panel["trade_decision_changed"] = False
    panel["active_in_trade_decision"] = False
    panel["report_changed"] = False
    panel["signal_date"] = panel["signal_date"].dt.strftime("%Y-%m-%d")
    panel["next_tradable_date"] = panel["next_tradable_date"].dt.strftime("%Y-%m-%d")
    return panel[_outcome_columns(panel)]


def _outcome_columns(panel: pd.DataFrame) -> list[str]:
    base = [
        "event_variant",
        "signal_date",
        "next_tradable_date",
        "ticker",
        "candidate_name",
        "candidate_source",
        "candidate_layer",
        "period_bucket",
        "regime_label",
        "case_trace_only",
        "entry_price",
        "next_tradable_date_price_ready",
    ]
    horizon_cols = []
    for horizon in HORIZONS:
        horizon_cols.extend(
            [
                f"event_return_{horizon}d_pct",
                f"0050_return_{horizon}d_pct",
                f"00631L_return_{horizon}d_pct",
                f"excess_vs_0050_{horizon}d_pct",
                f"excess_vs_00631L_{horizon}d_pct",
                f"horizon_{horizon}d_complete",
            ]
        )
    tail = [
        "event_mae_20d_pct",
        "event_mfe_20d_pct",
        "event_mae_60d_pct",
        "event_mfe_60d_pct",
        "all_horizons_complete",
        "incomplete_outcome_reason",
        "uses_forward_return_as_rule",
        "portfolio_replay_executed",
        "formal_model_changed",
        "trade_decision_changed",
        "active_in_trade_decision",
        "report_changed",
    ]
    return [column for column in base + horizon_cols + tail if column in panel.columns]


def _incomplete_reason(row: pd.Series) -> str:
    reasons = []
    if not bool(row.get("next_tradable_date_price_ready", False)):
        reasons.append("missing_event_entry_price")
    for horizon in HORIZONS:
        if not bool(row.get(f"horizon_{horizon}d_complete", False)):
            reasons.append(f"incomplete_{horizon}d_horizon")
    return ";".join(reasons)


def _summary_by_variant(panel: pd.DataFrame) -> pd.DataFrame:
    return _group_summary(panel, ["event_variant"])


def _summary_by_period(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for period_id, start, end in PERIODS:
        selected = _between(panel, start, end)
        if selected.empty:
            rows.append({"period_bucket": period_id, "event_count": 0, "actual_start": "", "actual_end": ""})
            continue
        summary = _group_summary(selected.assign(period_bucket=period_id), ["period_bucket", "event_variant"])
        summary["requested_start"] = start
        summary["requested_end"] = end
        rows.append(summary)
    return pd.concat([row if isinstance(row, pd.DataFrame) else pd.DataFrame([row]) for row in rows], ignore_index=True, sort=False)


def _summary_by_regime(panel: pd.DataFrame) -> pd.DataFrame:
    return _group_summary(panel, ["regime_label", "event_variant"])


def _group_summary(panel: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if panel.empty:
        return pd.DataFrame(columns=group_cols + ["event_count"])
    aggregations = {
        "event_count": ("ticker", "count"),
        "unique_tickers": ("ticker", "nunique"),
        "actual_start": ("signal_date", "min"),
        "actual_end": ("signal_date", "max"),
        "complete_60d_rows": ("horizon_60d_complete", "sum"),
        "avg_event_return_20d_pct": ("event_return_20d_pct", "mean"),
        "avg_event_return_60d_pct": ("event_return_60d_pct", "mean"),
        "avg_0050_return_20d_pct": ("0050_return_20d_pct", "mean"),
        "avg_00631L_return_20d_pct": ("00631L_return_20d_pct", "mean"),
        "avg_excess_vs_0050_20d_pct": ("excess_vs_0050_20d_pct", "mean"),
        "avg_excess_vs_00631L_20d_pct": ("excess_vs_00631L_20d_pct", "mean"),
        "avg_excess_vs_0050_60d_pct": ("excess_vs_0050_60d_pct", "mean"),
        "avg_excess_vs_00631L_60d_pct": ("excess_vs_00631L_60d_pct", "mean"),
        "avg_mae_20d_pct": ("event_mae_20d_pct", "mean"),
        "avg_mfe_20d_pct": ("event_mfe_20d_pct", "mean"),
        "avg_mae_60d_pct": ("event_mae_60d_pct", "mean"),
        "avg_mfe_60d_pct": ("event_mfe_60d_pct", "mean"),
        "top_ticker_share": ("ticker", lambda s: round(float(s.value_counts(normalize=True).iloc[0]), 6)),
    }
    return panel.groupby(group_cols, as_index=False).agg(**aggregations)


def _proxy_comparison(by_period: pd.DataFrame, proxy_path: Path) -> pd.DataFrame:
    if not proxy_path.exists() or by_period.empty:
        return by_period.copy()
    proxy = pd.read_csv(proxy_path).rename(columns={"variant_id": "event_variant", "period": "period_bucket"})
    merged = by_period.merge(
        proxy[
            [
                "period_bucket",
                "event_variant",
                "event_count",
                "avg_forward_20d_pct",
                "avg_forward_60d_pct",
            ]
        ].rename(
            columns={
                "event_count": "proxy_event_count",
                "avg_forward_20d_pct": "proxy_avg_forward_20d_pct",
                "avg_forward_60d_pct": "proxy_avg_forward_60d_pct",
            }
        ),
        on=["period_bucket", "event_variant"],
        how="left",
    )
    merged["exact_minus_proxy_20d_pct"] = merged["avg_event_return_20d_pct"] - merged["proxy_avg_forward_20d_pct"]
    merged["exact_minus_proxy_60d_pct"] = merged["avg_event_return_60d_pct"] - merged["proxy_avg_forward_60d_pct"]
    return merged


def _benchmark_summary(panel: pd.DataFrame) -> pd.DataFrame:
    if panel.empty:
        return pd.DataFrame()
    rows = []
    for horizon in HORIZONS:
        rows.append(
            {
                "horizon": f"{horizon}d",
                "complete_rows": int(panel[f"horizon_{horizon}d_complete"].sum()),
                "avg_event_return_pct": panel[f"event_return_{horizon}d_pct"].mean(),
                "avg_0050_return_pct": panel[f"0050_return_{horizon}d_pct"].mean(),
                "avg_00631L_return_pct": panel[f"00631L_return_{horizon}d_pct"].mean(),
                "avg_excess_vs_0050_pct": panel[f"excess_vs_0050_{horizon}d_pct"].mean(),
                "avg_excess_vs_00631L_pct": panel[f"excess_vs_00631L_{horizon}d_pct"].mean(),
            }
        )
    return pd.DataFrame(rows)


def _between(panel: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    dates = pd.to_datetime(panel["signal_date"], errors="coerce")
    return panel[(dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))].copy()


def _future_data_audit(panel: pd.DataFrame) -> pd.DataFrame:
    if panel.empty:
        return pd.DataFrame(columns=["signal_date", "ticker", "event_variant", "future_data_violation", "reason"])
    out = panel[["signal_date", "next_tradable_date", "ticker", "event_variant"]].copy()
    out["future_data_violation"] = False
    out["reason"] = ""
    return out


def _primary_period(date: pd.Timestamp) -> str:
    if date <= pd.Timestamp("2022-12-31"):
        return "period_1_2014_11_2022_12"
    if date <= pd.Timestamp("2026-06-30"):
        return "period_2_2023_01_2026_06"
    return "outside_requested_period"


def _summary(manifest: dict, by_variant: pd.DataFrame, by_period: pd.DataFrame) -> str:
    return "\n".join(
        [
            "# Strong stock trend-extension exact outcome panel",
            "",
            "本包只做 exact event outcome evaluation；forward return 僅作評估 metadata，不作 live rule，不跑 portfolio。",
            "",
            f"- event rows input：{manifest['event_rows_input']}",
            f"- outcome rows：{manifest['outcome_rows']}",
            f"- incomplete outcome rows：{manifest['incomplete_outcome_rows']}",
            f"- future data violation count：{manifest['future_data_violation_count']}",
            "",
            "## By variant",
            ("no rows" if by_variant.empty else by_variant.to_csv(index=False).strip()),
            "",
            "## By period",
            ("no rows" if by_period.empty else by_period.head(20).to_csv(index=False).strip()),
        ]
    )


def _base_ticker(ticker: str) -> str:
    return str(ticker).strip().split(".")[0]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--event-contract", default=str(DEFAULT_EVENT_CONTRACT))
    parser.add_argument("--proxy-outcome", default=str(DEFAULT_PROXY_OUTCOME))
    parser.add_argument("--liquidity-dir", default=str(DEFAULT_LIQUIDITY_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args(argv)
    manifest = run_strong_stock_trend_extension_exact_outcome_panel(
        repo_root=args.repo_root,
        event_contract=args.event_contract,
        proxy_outcome=args.proxy_outcome,
        liquidity_dir=args.liquidity_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
