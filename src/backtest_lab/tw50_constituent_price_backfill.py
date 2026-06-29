from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import pandas as pd

from backtest_lab.data import download_yfinance_prices, load_price_csv
from backtest_lab.tw50_backfill_audit import DEFAULT_CONSTITUENTS_PATH


TASK_ID = "TASK-BACKTEST-CORE-DATA-BACKFILL-0050-CONSTITUENT-PRICE-201411-LATEST-20260629"
DEFAULT_OUTPUT_DIR = "outputs/core_0050_constituent_price_backfill_201411_latest_20260629"
DEFAULT_CACHE_DIR = "backtest_cache/stock_pool_observations"
DEFAULT_START_DATE = "2014-11-01"
DEFAULT_END_DATE = "2026-06-29"
BENCHMARK_TICKERS = ("0050.TW", "00631L.TW")
PRICE_DOWNLOADER = Callable[[list[str], str, str, str | Path], dict[str, pd.DataFrame]]


def run_tw50_constituent_price_backfill(
    *,
    constituents_path: str | Path = DEFAULT_CONSTITUENTS_PATH,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
    downloader: PRICE_DOWNLOADER | None = None,
) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    run_log: list[dict[str, str]] = []

    def log(step: str, status: str, detail: str = "") -> None:
        run_log.append(
            {
                "timestamp": pd.Timestamp.now(tz="Asia/Taipei").strftime("%Y-%m-%d %H:%M:%S%z"),
                "step": step,
                "status": status,
                "detail": detail,
            }
        )
        pd.DataFrame(run_log).to_csv(output / "run_log.csv", index=False, encoding="utf-8-sig")
        (output / "current_step.txt").write_text(step, encoding="utf-8")

    try:
        log("load_universe", "started", str(constituents_path))
        universe, source_ledger = _load_known_universe(Path(constituents_path))
        tickers = sorted(set(universe["ticker"].astype(str)) | set(BENCHMARK_TICKERS))
        before = [_coverage_for_ticker(ticker, cache) for ticker in tickers]
        log("before_coverage", "completed", f"tickers={len(tickers)}")

        completed_rows: list[dict[str, object]] = []
        failed_rows: list[dict[str, object]] = []
        refreshed_frames: dict[str, pd.DataFrame] = {}
        fetch_prices = downloader or _download_one_ticker
        for ticker in tickers:
            step = f"refresh_{ticker}"
            log(step, "started", f"{start_date} to {end_date}")
            try:
                prices = fetch_prices([ticker], start_date, end_date, cache)
                frame = prices.get(ticker)
                if frame is None or frame.empty:
                    raise ValueError("download returned no price frame")
                refreshed_frames[ticker] = frame
                coverage = _coverage_from_frame(ticker, frame, cache)
                completed_rows.append(
                    {
                        "ticker": ticker,
                        "status": "completed",
                        "first_date": coverage["first_date"],
                        "last_date": coverage["last_date"],
                        "row_count": coverage["row_count"],
                        "cache_path": coverage["cache_path"],
                        "source": "yfinance_cache_refresh",
                    }
                )
                log(step, "completed", f"{coverage['first_date']} to {coverage['last_date']}")
            except Exception as exc:  # noqa: BLE001 - keep per-ticker failure observable and resumable.
                latest = _coverage_for_ticker(ticker, cache)
                failed_rows.append(
                    {
                        "ticker": ticker,
                        "status": "failed",
                        "latest_available_date": latest.get("last_date", ""),
                        "reason": str(exc),
                        "cache_path": latest.get("cache_path", ""),
                    }
                )
                log(step, "failed", str(exc))

        after = [_coverage_for_ticker(ticker, cache, loaded_frame=refreshed_frames.get(ticker)) for ticker in tickers]
        coverage = _price_coverage_matrix(universe, before, after, start_date, end_date)
        missing = _missing_price_periods(coverage, start_date, end_date)
        completed = pd.DataFrame(
            completed_rows,
            columns=["ticker", "status", "first_date", "last_date", "row_count", "cache_path", "source"],
        )
        failed = pd.DataFrame(
            failed_rows,
            columns=["ticker", "status", "latest_available_date", "reason", "cache_path"],
        )
        manifest = _manifest(output, tickers, coverage, completed, failed, start_date, end_date, cache, constituents_path)

        log("write_outputs", "started", "")
        (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        coverage.to_csv(output / "price_coverage_matrix.csv", index=False, encoding="utf-8-sig")
        completed.to_csv(output / "price_backfill_completed.csv", index=False, encoding="utf-8-sig")
        failed.to_csv(output / "price_backfill_failed.csv", index=False, encoding="utf-8-sig")
        missing.to_csv(output / "missing_price_periods.csv", index=False, encoding="utf-8-sig")
        source_ledger.to_csv(output / "universe_source_ledger.csv", index=False, encoding="utf-8-sig")
        (output / "final_summary_zh.md").write_text(_summary_zh(manifest, coverage, missing, failed), encoding="utf-8")
        log("completed", manifest["status"], str(output.resolve()))
        (output / "current_step.txt").write_text("completed\n", encoding="utf-8")
        return output
    except Exception as exc:
        pd.DataFrame([{"step": "run_tw50_constituent_price_backfill", "error": str(exc)}]).to_csv(
            output / "price_backfill_failed.csv", index=False, encoding="utf-8-sig"
        )
        log("failed", "failed", str(exc))
        raise


def _download_one_ticker(tickers: list[str], start_date: str, end_date: str, cache_dir: str | Path) -> dict[str, pd.DataFrame]:
    return download_yfinance_prices(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        cache_dir=cache_dir,
        allow_edge_gap=True,
    )


def _load_known_universe(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not path.exists():
        raise FileNotFoundError(f"constituents file not found: {path}")
    frame = pd.read_csv(path).fillna("")
    required = {"ticker", "name"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"constituents file missing columns: {', '.join(sorted(missing))}")
    frame["ticker"] = frame["ticker"].astype(str).str.strip()
    frame = frame[frame["ticker"].ne("")].copy()
    if "effective_date" not in frame.columns:
        frame["effective_date"] = ""
    if "source" not in frame.columns:
        frame["source"] = "unknown"
    if "source_updated_at" not in frame.columns:
        frame["source_updated_at"] = ""
    frame["universe_scope"] = "current_or_known_0050_constituent_provisional"
    source_rows = []
    for source, group in frame.groupby("source", dropna=False):
        source_rows.append(
            {
                "source_path": str(path),
                "source": str(source),
                "source_type": _source_type(str(source)),
                "effective_dates": ";".join(sorted(set(group["effective_date"].astype(str)))),
                "ticker_count": int(group["ticker"].nunique()),
                "formal_exact_pit": False,
                "provisional_universe": True,
                "user_manual_download_required": False,
                "notes": "Core/Data owns acquisition. This current/known universe is for price cache backfill only, not historical PIT replay.",
            }
        )
    source_rows.append(
        {
            "source_path": "built_in_benchmark_tickers",
            "source": "0050_and_00631L_benchmarks",
            "source_type": "benchmark",
            "effective_dates": "",
            "ticker_count": len(BENCHMARK_TICKERS),
            "formal_exact_pit": False,
            "provisional_universe": False,
            "user_manual_download_required": False,
            "notes": "Benchmarks are included for forward/relative performance and do not imply PIT constituent membership.",
        }
    )
    return frame, pd.DataFrame(source_rows)


def _source_type(source: str) -> str:
    text = source.lower()
    if "exact" in text:
        return "exact_candidate"
    if "manual" in text:
        return "manual_ledger_candidate"
    if "snapshot" in text or "seed" in text or "proxy" in text:
        return "current_proxy_snapshot"
    return "unknown"


def _coverage_for_ticker(ticker: str, cache_dir: Path, *, loaded_frame: pd.DataFrame | None = None) -> dict[str, object]:
    path = _cache_path(cache_dir, ticker)
    frame = loaded_frame
    if frame is None and path.exists():
        try:
            frame = load_price_csv(path)
        except Exception:
            frame = None
    if frame is None or frame.empty:
        return {
            "ticker": ticker,
            "first_date": "",
            "last_date": "",
            "row_count": 0,
            "adjusted_close_available": False,
            "cache_path": str(path),
            "cache_exists": path.exists(),
        }
    return _coverage_from_frame(ticker, frame, cache_dir)


def _coverage_from_frame(ticker: str, frame: pd.DataFrame, cache_dir: Path) -> dict[str, object]:
    return {
        "ticker": ticker,
        "first_date": frame.index.min().strftime("%Y-%m-%d"),
        "last_date": frame.index.max().strftime("%Y-%m-%d"),
        "row_count": int(len(frame)),
        "adjusted_close_available": "adj_close" in frame.columns and pd.to_numeric(frame["adj_close"], errors="coerce").notna().any(),
        "cache_path": str(_cache_path(cache_dir, ticker)),
        "cache_exists": _cache_path(cache_dir, ticker).exists(),
    }


def _cache_path(cache_dir: Path, ticker: str) -> Path:
    return cache_dir / f"{ticker.replace('.', '_')}.csv"


def _price_coverage_matrix(
    universe: pd.DataFrame,
    before: list[dict[str, object]],
    after: list[dict[str, object]],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    names = {str(row["ticker"]): str(row.get("name") or "") for row in universe.to_dict(orient="records")}
    sources = {str(row["ticker"]): str(row.get("source") or "") for row in universe.to_dict(orient="records")}
    before_by_ticker = {str(row["ticker"]): row for row in before}
    rows = []
    for row in after:
        ticker = str(row["ticker"])
        before_row = before_by_ticker.get(ticker, {})
        first = str(row.get("first_date", ""))
        last = str(row.get("last_date", ""))
        first_ok = bool(first and pd.Timestamp(first) <= pd.Timestamp(start_date) + pd.Timedelta(days=10))
        last_ok = bool(last and pd.Timestamp(last) >= pd.Timestamp(end_date) - pd.Timedelta(days=10))
        adjusted = _as_bool(row.get("adjusted_close_available"))
        rows.append(
            {
                "ticker": ticker,
                "name": names.get(ticker, _benchmark_name(ticker)),
                "universe_source": sources.get(ticker, "benchmark"),
                "source": "yfinance_cache_refresh",
                "before_first_date": before_row.get("first_date", ""),
                "before_last_date": before_row.get("last_date", ""),
                "first_date": first,
                "last_date": last,
                "row_count": row.get("row_count", 0),
                "missing_periods": _missing_period_text(first, last, start_date, end_date),
                "adjusted_close_available": adjusted,
                "ready_for_backtest_price_only": bool(first_ok and last_ok and adjusted),
                "strategy_ready": False,
                "strategy_ready_blocker": "Historical PIT constituents and target/signal stream are not complete.",
                "provisional_universe": ticker not in BENCHMARK_TICKERS,
                "cache_path": row.get("cache_path", ""),
            }
        )
    return pd.DataFrame(rows).sort_values("ticker").reset_index(drop=True)


def _benchmark_name(ticker: str) -> str:
    return {"0050.TW": "元大台灣50", "00631L.TW": "元大台灣50正2"}.get(ticker, "")


def _missing_period_text(first: str, last: str, start_date: str, end_date: str) -> str:
    periods: list[str] = []
    if not first:
        return f"{start_date}~{end_date}: no local price data"
    if pd.Timestamp(first) > pd.Timestamp(start_date) + pd.Timedelta(days=10):
        periods.append(f"{start_date}~{first}: no price before first available date")
    if not last or pd.Timestamp(last) < pd.Timestamp(end_date) - pd.Timedelta(days=10):
        periods.append(f"{last or 'unknown'}~{end_date}: no latest price coverage")
    return "; ".join(periods)


def _missing_price_periods(coverage: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    rows = []
    for row in coverage.to_dict(orient="records"):
        missing = str(row.get("missing_periods") or "")
        if not missing:
            continue
        rows.append(
            {
                "ticker": row["ticker"],
                "name": row.get("name", ""),
                "missing_periods": missing,
                "first_date": row.get("first_date", ""),
                "last_date": row.get("last_date", ""),
                "requested_start_date": start_date,
                "requested_end_date": end_date,
                "reason": "price source starts after requested start, latest source is missing, or ticker has no local source",
                "requires_user_manual_download": False,
                "next_action_owner": "Core/Data",
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "ticker",
            "name",
            "missing_periods",
            "first_date",
            "last_date",
            "requested_start_date",
            "requested_end_date",
            "reason",
            "requires_user_manual_download",
            "next_action_owner",
        ],
    )


def _manifest(
    output: Path,
    tickers: list[str],
    coverage: pd.DataFrame,
    completed: pd.DataFrame,
    failed: pd.DataFrame,
    start_date: str,
    end_date: str,
    cache_dir: Path,
    constituents_path: str | Path,
) -> dict[str, object]:
    ready_count = int(coverage["ready_for_backtest_price_only"].map(_as_bool).sum()) if not coverage.empty else 0
    incomplete_count = int(len(coverage) - ready_count)
    return {
        "schema_version": 1,
        "task_id": TASK_ID,
        "status": "completed" if failed.empty else "partial_completed_with_failures",
        "formal_model_changed": False,
        "trade_decision_changed": False,
        "start_date": start_date,
        "end_date": end_date,
        "cache_dir": str(cache_dir),
        "constituents_path": str(constituents_path),
        "known_universe_ticker_count": len(tickers),
        "price_backfill_completed_count": int(len(completed)),
        "price_backfill_failed_count": int(len(failed)),
        "price_only_ready_count": ready_count,
        "price_only_incomplete_count": incomplete_count,
        "historical_pit_ready": False,
        "strategy_ready": False,
        "provisional_universe": True,
        "current_snapshot_used_as_historical_pit": False,
        "user_manual_download_required": False,
        "data_acquisition_owner": "Core/Data/Radar",
        "output_dir": str(output.resolve()),
    }


def _summary_zh(manifest: dict[str, object], coverage: pd.DataFrame, missing: pd.DataFrame, failed: pd.DataFrame) -> str:
    lines = [
        "# 0050 成分股 known/provisional universe price backfill",
        "",
        "## 結論",
        "",
        f"- 狀態：{manifest['status']}。",
        f"- 本次處理 known/provisional universe 共 {manifest['known_universe_ticker_count']} 檔，包含 0050 與 00631L。",
        f"- price-only ready：{manifest['price_only_ready_count']} 檔；仍有價格期間缺口：{manifest['price_only_incomplete_count']} 檔。",
        f"- failed tickers：{manifest['price_backfill_failed_count']} 檔。",
        "- 本任務沒有改 formal selector、formal target 或 trade decision。",
        "- current 0050 成分股只作 provisional universe 的價格補齊，不代表 2014-2023 歷史 PIT 成分已完成。",
        "- 若後續需要 historical holdings/PIT PDF，應由 Core/Data/Radar 自動定位、下載或記錄 HTTP/URL 失敗原因；不是要求使用者人工找。",
        "",
        "## 主要缺口",
        "",
    ]
    if missing.empty:
        lines.append("- 指定期間內沒有價格期間缺口。")
    else:
        lines.append(f"- 有 {len(missing)} 檔存在 requested start/end coverage 缺口，詳見 `missing_price_periods.csv`。")
    if not failed.empty:
        lines += ["", "## Failed tickers", ""]
        for row in failed.to_dict(orient="records"):
            lines.append(f"- {row['ticker']}：{row['reason']}")
    return "\n".join(lines) + "\n"


def _as_bool(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill current/known 0050 constituent price cache from 2014-11 to latest.")
    parser.add_argument("--constituents-path", default=DEFAULT_CONSTITUENTS_PATH)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", default=DEFAULT_END_DATE)
    args = parser.parse_args()
    output = run_tw50_constituent_price_backfill(
        constituents_path=args.constituents_path,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    print(f"TW50_CONSTITUENT_PRICE_BACKFILL_OUTPUT={output.resolve()}")


if __name__ == "__main__":
    main()
