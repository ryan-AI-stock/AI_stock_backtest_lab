"""Repair latest RS fields for narrow-capital case diagnostics.

This is a data-readiness package only.  It computes trailing RS/MA/turnover
fields from local adjusted price caches for the latest narrow-capital case
tickers, without running portfolio replay or changing formal decisions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


TASK_ID = "TASK-BACKTEST-CORE-NARROW-CAPITAL-LATEST-RS-PANEL-REPAIR-001"
DEFAULT_OUTPUT_DIR = Path("outputs/narrow_capital_latest_rs_panel_repair_20260705")
PRICE_CACHE_DIR = Path("backtest_cache/stock_pool_observations")
PRIMARY_AS_OF_DATE = "2026-06-30"
CASE_TRACE_AS_OF_DATE = "2026-07-03"
CASE_TICKERS = ["6669.TW", "2308.TW", "2317.TW"]
BENCHMARK_TICKERS = ["0050.TW", "00631L.TW"]
MINIMUM_TICKERS = CASE_TICKERS + BENCHMARK_TICKERS


def run_narrow_capital_latest_rs_panel_repair(
    *,
    repo_root: str | Path = ".",
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    price_cache_dir: str | Path = PRICE_CACHE_DIR,
    primary_as_of_date: str = PRIMARY_AS_OF_DATE,
    case_trace_as_of_date: str = CASE_TRACE_AS_OF_DATE,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    output = _resolve(root, output_dir)
    output.mkdir(parents=True, exist_ok=True)
    price_dir = _resolve(root, price_cache_dir)

    prices, coverage = _load_all_prices(price_dir)
    primary_panel = _build_rs_panel(
        prices,
        requested_as_of_date=primary_as_of_date,
        requested_tickers=None,
        panel_scope="full_local_cache_latest_coverage",
        case_trace_only=False,
    )
    case_panel = primary_panel[primary_panel["ticker"].isin(MINIMUM_TICKERS)].copy()
    trace_panel = _build_rs_panel(
        prices,
        requested_as_of_date=case_trace_as_of_date,
        requested_tickers=MINIMUM_TICKERS,
        panel_scope="case_trace_20260703",
        case_trace_only=True,
    )
    benchmark = _benchmark_source_validation(coverage)
    future = _future_data_audit(primary_panel, case_panel, trace_panel)

    primary_panel.to_csv(output / "latest_rs_panel_repaired.csv", index=False, encoding="utf-8-sig")
    case_panel.to_csv(output / "case_ticker_rs_panel_20260630.csv", index=False, encoding="utf-8-sig")
    trace_panel.to_csv(output / "case_trace_rs_panel_20260703.csv", index=False, encoding="utf-8-sig")
    benchmark.to_csv(output / "benchmark_source_validation.csv", index=False, encoding="utf-8-sig")
    coverage.to_csv(output / "price_coverage_audit.csv", index=False, encoding="utf-8-sig")
    future.to_csv(output / "future_data_audit.csv", index=False, encoding="utf-8-sig")

    future_count = int(future["future_data_violation"].sum()) if len(future) else 0
    case_ready = int(case_panel["rs_fields_ready"].sum())
    trace_ready = int(trace_panel["as_of_price_available"].sum())
    manifest: dict[str, Any] = {
        "task_id": TASK_ID,
        "status": "completed_targeted_case_rs_repair",
        "output_dir": str(output),
        "primary_as_of_date": primary_as_of_date,
        "primary_price_as_of_policy": "latest local price date <= requested as-of date; staleness explicitly reported",
        "turnover_rank_scope": "full local stock_pool_observations cache available on or before requested as-of date; not a formal all-market rank",
        "case_trace_as_of_date": case_trace_as_of_date,
        "case_trace_policy": "case-only; blocked when local price cache does not cover requested date",
        "minimum_required_tickers": MINIMUM_TICKERS,
        "latest_rs_panel_rows": int(len(primary_panel)),
        "case_ticker_rows": int(len(case_panel)),
        "case_ticker_rs_ready_rows": case_ready,
        "case_trace_rows": int(len(trace_panel)),
        "case_trace_as_of_price_available_rows": trace_ready,
        "adjusted_close_policy": "returns, RS, MA, and drawdown use adj_close when available; turnover uses raw close * volume",
        "future_data_violation_count": future_count,
        "formal_model_changed": False,
        "trade_decision_changed": False,
        "active_in_trade_decision": False,
        "report_changed": False,
        "portfolio_replay_executed": False,
        "uses_forward_return_as_rule": False,
        "ready_for_case_membership_rerun": bool(case_ready >= len(MINIMUM_TICKERS) and future_count == 0),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "final_summary_zh.md").write_text(_summary(manifest), encoding="utf-8")
    pd.DataFrame([{"task_id": TASK_ID, "status": "completed", "output_dir": str(output)}]).to_csv(
        output / "completed.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(columns=["task_id", "status", "reason"]).to_csv(output / "failed.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(
        [
            {"step": "load_local_price_cache", "status": "completed"},
            {"step": "compute_latest_rs_fields", "status": "completed"},
            {"step": "write_case_panels", "status": "completed"},
            {"step": "write_contract_package", "status": "completed"},
        ]
    ).to_csv(output / "run_log.csv", index=False, encoding="utf-8-sig")
    return manifest


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _load_all_prices(price_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    coverage_rows: list[dict[str, Any]] = []
    for path in sorted(price_dir.glob("*.csv")):
        ticker = _ticker_from_file(path)
        frame = pd.read_csv(path, usecols=lambda col: col in {"date", "close", "adj_close", "volume"})
        if "adj_close" not in frame.columns:
            frame["adj_close"] = frame.get("close")
        if "volume" not in frame.columns:
            frame["volume"] = pd.NA
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
        frame["adj_close"] = pd.to_numeric(frame["adj_close"], errors="coerce")
        frame["volume"] = pd.to_numeric(frame["volume"], errors="coerce")
        frame = frame.dropna(subset=["date", "adj_close"]).sort_values("date").drop_duplicates("date")
        if frame.empty:
            coverage_rows.append({"ticker": ticker, "rows": 0, "source_price_coverage_start": "", "source_price_coverage_end": ""})
            continue
        frame["ticker"] = ticker
        frame["turnover_value"] = frame["close"] * frame["volume"]
        frames.append(frame[["date", "ticker", "close", "adj_close", "volume", "turnover_value"]])
        coverage_rows.append(
            {
                "ticker": ticker,
                "rows": int(len(frame)),
                "source_price_coverage_start": str(frame["date"].min()),
                "source_price_coverage_end": str(frame["date"].max()),
                "price_source_path": str(path),
                "adjusted_close_available": bool("adj_close" in pd.read_csv(path, nrows=0).columns),
                "adjusted_close_policy": "use adj_close for returns/RS/MA/drawdown; close*volume for turnover",
            }
        )
    prices = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    return prices, pd.DataFrame(coverage_rows)


def _ticker_from_file(path: Path) -> str:
    stem = path.stem
    if stem.endswith("_TW"):
        return stem[:-3] + ".TW"
    if stem.endswith("_TWO"):
        return stem[:-4] + ".TWO"
    return stem


def _build_rs_panel(
    prices: pd.DataFrame,
    *,
    requested_as_of_date: str,
    requested_tickers: list[str] | None,
    panel_scope: str,
    case_trace_only: bool,
) -> pd.DataFrame:
    tickers = sorted(requested_tickers or prices["ticker"].dropna().unique())
    benchmark_0050 = _metric_row(prices, "0050.TW", requested_as_of_date)
    benchmark_00631l = _metric_row(prices, "00631L.TW", requested_as_of_date)
    rows: list[dict[str, Any]] = []
    turnover_metrics = {ticker: _metric_row(prices, ticker, requested_as_of_date) for ticker in tickers}
    turnover_rank = _turnover_ranks(turnover_metrics)
    for ticker in tickers:
        metric = turnover_metrics[ticker]
        row = {
            "requested_as_of_date": requested_as_of_date,
            "price_as_of_date": metric.get("price_as_of_date", ""),
            "price_staleness_days": metric.get("price_staleness_days", pd.NA),
            "ticker": ticker,
            "panel_scope": panel_scope,
            "case_trace_only": case_trace_only,
            "as_of_price_available": metric.get("as_of_price_available", False),
            "rs_fields_ready": bool(metric.get("data_ready", False) and benchmark_0050.get("data_ready", False) and benchmark_00631l.get("data_ready", False)),
            "ret20_trailing_pct": metric.get("ret20_trailing_pct", pd.NA),
            "ret60_trailing_pct": metric.get("ret60_trailing_pct", pd.NA),
            "rs20_vs_0050_pct": _subtract(metric.get("ret20_trailing_pct"), benchmark_0050.get("ret20_trailing_pct")),
            "rs60_vs_0050_pct": _subtract(metric.get("ret60_trailing_pct"), benchmark_0050.get("ret60_trailing_pct")),
            "rs20_vs_00631L_pct": _subtract(metric.get("ret20_trailing_pct"), benchmark_00631l.get("ret20_trailing_pct")),
            "rs60_vs_00631L_pct": _subtract(metric.get("ret60_trailing_pct"), benchmark_00631l.get("ret60_trailing_pct")),
            "close_vs_ma20": metric.get("close_vs_ma20", pd.NA),
            "close_vs_ma60": metric.get("close_vs_ma60", pd.NA),
            "drawdown_from_20d_high_pct": metric.get("drawdown_from_20d_high_pct", pd.NA),
            "drawdown_from_60d_high_pct": metric.get("drawdown_from_60d_high_pct", pd.NA),
            "turnover_rank": turnover_rank.get(ticker, pd.NA),
            "turnover_rank_scope": (
                "full_local_price_cache_panel_at_asof"
                if requested_tickers is None
                else "requested_case_tickers_only_at_asof"
            ),
            "rolling_turnover_value": metric.get("rolling_turnover_value", pd.NA),
            "source_price_coverage_start": metric.get("source_price_coverage_start", ""),
            "source_price_coverage_end": metric.get("source_price_coverage_end", ""),
            "adjusted_close_policy": "adj_close_for_returns_rs_ma_drawdown__close_times_volume_for_turnover",
            "blocked_reason": metric.get("blocked_reason", ""),
            "formal_model_changed": False,
            "trade_decision_changed": False,
            "active_in_trade_decision": False,
            "report_changed": False,
            "portfolio_replay_executed": False,
            "uses_forward_return_as_rule": False,
        }
        if case_trace_only and not row["as_of_price_available"]:
            row["blocked_reason"] = "local_price_cache_not_available_for_requested_case_trace_date"
        rows.append(row)
    return pd.DataFrame(rows)


def _metric_row(prices: pd.DataFrame, ticker: str, requested_as_of_date: str) -> dict[str, Any]:
    frame = prices[prices["ticker"].eq(ticker)].sort_values("date").reset_index(drop=True)
    if frame.empty:
        return {"ticker": ticker, "data_ready": False, "blocked_reason": "missing_price_cache"}
    history = frame[frame["date"] <= requested_as_of_date].copy()
    if history.empty:
        return {
            "ticker": ticker,
            "data_ready": False,
            "source_price_coverage_start": str(frame["date"].min()),
            "source_price_coverage_end": str(frame["date"].max()),
            "blocked_reason": "no_price_on_or_before_requested_as_of_date",
        }
    asof = history.iloc[-1]
    price_as_of_date = str(asof["date"])
    history = history.tail(80).copy()
    data_ready = len(history) >= 61
    if not data_ready:
        return {
            "ticker": ticker,
            "data_ready": False,
            "price_as_of_date": price_as_of_date,
            "as_of_price_available": price_as_of_date == requested_as_of_date,
            "price_staleness_days": _calendar_days(price_as_of_date, requested_as_of_date),
            "source_price_coverage_start": str(frame["date"].min()),
            "source_price_coverage_end": str(frame["date"].max()),
            "blocked_reason": "insufficient_60d_history",
        }
    close = float(history.iloc[-1]["adj_close"])
    ret20 = (close / float(history.iloc[-21]["adj_close"]) - 1.0) * 100.0
    ret60 = (close / float(history.iloc[-61]["adj_close"]) - 1.0) * 100.0
    ma20 = float(history["adj_close"].tail(20).mean())
    ma60 = float(history["adj_close"].tail(60).mean())
    high20 = float(history["adj_close"].tail(20).max())
    high60 = float(history["adj_close"].tail(60).max())
    rolling_turnover = float(history["turnover_value"].tail(20).mean())
    return {
        "ticker": ticker,
        "data_ready": True,
        "price_as_of_date": price_as_of_date,
        "as_of_price_available": price_as_of_date == requested_as_of_date,
        "price_staleness_days": _calendar_days(price_as_of_date, requested_as_of_date),
        "ret20_trailing_pct": round(ret20, 6),
        "ret60_trailing_pct": round(ret60, 6),
        "close_vs_ma20": round((close / ma20 - 1.0) * 100.0, 6) if ma20 else pd.NA,
        "close_vs_ma60": round((close / ma60 - 1.0) * 100.0, 6) if ma60 else pd.NA,
        "drawdown_from_20d_high_pct": round((close / high20 - 1.0) * 100.0, 6) if high20 else pd.NA,
        "drawdown_from_60d_high_pct": round((close / high60 - 1.0) * 100.0, 6) if high60 else pd.NA,
        "rolling_turnover_value": round(rolling_turnover, 2),
        "source_price_coverage_start": str(frame["date"].min()),
        "source_price_coverage_end": str(frame["date"].max()),
        "blocked_reason": "",
    }


def _turnover_ranks(metrics: dict[str, dict[str, Any]]) -> dict[str, int]:
    values = [
        (ticker, metric.get("rolling_turnover_value"))
        for ticker, metric in metrics.items()
        if pd.notna(metric.get("rolling_turnover_value"))
    ]
    values = sorted(values, key=lambda item: float(item[1]), reverse=True)
    return {ticker: idx + 1 for idx, (ticker, _) in enumerate(values)}


def _benchmark_source_validation(coverage: pd.DataFrame) -> pd.DataFrame:
    out = coverage[coverage["ticker"].isin(BENCHMARK_TICKERS)].copy()
    out["benchmark_required"] = True
    out["benchmark_source_validated"] = out["rows"].fillna(0).astype(int) > 0
    return out


def _future_data_audit(*frames: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for idx, frame in enumerate(frames, start=1):
        violation = False
        if "price_as_of_date" in frame.columns and "requested_as_of_date" in frame.columns:
            dates = frame[["price_as_of_date", "requested_as_of_date"]].dropna()
            if not dates.empty:
                violation = bool((pd.to_datetime(dates["price_as_of_date"]) > pd.to_datetime(dates["requested_as_of_date"])).any())
        rows.append(
            {
                "audit_item": f"latest_rs_panel_{idx}",
                "rows": int(len(frame)),
                "future_data_violation": violation,
                "reason": "price_as_of_date must be <= requested_as_of_date; trailing returns only",
            }
        )
    return pd.DataFrame(rows)


def _subtract(value: object, benchmark: object) -> float | Any:
    if pd.isna(value) or pd.isna(benchmark):
        return pd.NA
    return round(float(value) - float(benchmark), 6)


def _calendar_days(start: str, end: str) -> int:
    return int((pd.Timestamp(end) - pd.Timestamp(start)).days)


def _summary(manifest: dict[str, Any]) -> str:
    return (
        "# Narrow capital latest RS panel repair\n\n"
        "## 結論\n\n"
        "- 本包只修 latest RS/readiness 欄位，沒有跑 portfolio，也沒有改 formal/report/trade。\n"
        f"- primary as-of：{manifest['primary_as_of_date']}\n"
        f"- latest RS panel rows：{manifest['latest_rs_panel_rows']}\n"
        f"- case ticker rows：{manifest['case_ticker_rows']}\n"
        f"- case ticker RS ready rows：{manifest['case_ticker_rs_ready_rows']}\n"
        f"- 2026-07-03 case trace rows：{manifest['case_trace_rows']}\n"
        f"- 2026-07-03 local price available rows：{manifest['case_trace_as_of_price_available_rows']}\n"
        f"- ready_for_case_membership_rerun：{manifest['ready_for_case_membership_rerun']}\n\n"
        "## 邊界\n\n"
        "- 2026-06-30 使用 latest local price date <= requested as-of date，並輸出 price_staleness_days。\n"
        f"- turnover rank scope：{manifest['turnover_rank_scope']}。\n"
        "- 2026-07-03 若本地價格不到該日，只能 case-only blocked trace，不硬算成 latest。\n"
        "- `uses_forward_return_as_rule=false`；所有欄位都是 trailing / same-source cache diagnostics。\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair latest narrow-capital RS panel fields.")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()
    manifest = run_narrow_capital_latest_rs_panel_repair(repo_root=args.repo_root, output_dir=args.output_dir)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
