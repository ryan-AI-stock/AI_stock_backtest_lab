from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import pandas as pd

from backtest_lab.data import load_price_csv
from backtest_lab.stock_pool_observation import _load_observation_price_frames


TASK_ID = "TASK-BACKTEST-CORE-DATA-REFRESH-AI-MEGACAP-PRICE-CACHE-FOR-MA-SIMILARITY-20260629"
DEFAULT_OUTPUT_DIR = "outputs/core_ai_megacap_price_refresh_for_ma_similarity_20260629"
DEFAULT_CACHE_DIR = "backtest_cache/stock_pool_observations"
DEFAULT_START_DATE = "2022-01-01"
DEFAULT_END_DATE = "2026-06-29"
DEFAULT_TICKERS = (
    "2330.TW",
    "2454.TW",
    "2308.TW",
    "2317.TW",
    "2382.TW",
    "3231.TW",
    "6669.TW",
    "0050.TW",
    "00631L.TW",
)
PRICE_LOADER = Callable[[list[str], str, str, str | Path], tuple[dict[str, pd.DataFrame], list[str]]]


def run_ai_megacap_price_refresh(
    *,
    tickers: tuple[str, ...] = DEFAULT_TICKERS,
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    loader: PRICE_LOADER | None = None,
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

    refresh_loader = loader or _load_observation_price_frames
    before = [_coverage_for_ticker(ticker, cache) for ticker in tickers]
    log("load_before_coverage", "completed", f"tickers={len(tickers)}")

    refreshed_rows: list[dict[str, object]] = []
    failed_rows: list[dict[str, object]] = []
    frames: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        step = f"refresh_{ticker}"
        log(step, "started", f"{start_date} to {end_date}")
        try:
            loaded, missing = refresh_loader(
                tickers=[ticker],
                start_date=start_date,
                end_date=end_date,
                cache_dir=cache,
            )
            if ticker in loaded and not loaded[ticker].empty:
                frames[ticker] = loaded[ticker]
                row = _coverage_from_frame(ticker, loaded[ticker])
                row.update(
                    {
                        "status": "refreshed",
                        "requested_end_date": end_date,
                        "cache_path": str(_cache_path(cache, ticker)),
                    }
                )
                refreshed_rows.append(row)
                log(step, "completed", f"latest={row['last_date']}")
            else:
                reason = "loader returned no frame"
                if ticker in missing:
                    reason = "ticker marked missing by loader"
                failed_rows.append(
                    {
                        "ticker": ticker,
                        "status": "failed",
                        "requested_end_date": end_date,
                        "latest_available_date": "",
                        "reason": reason,
                    }
                )
                log(step, "failed", reason)
        except Exception as exc:  # noqa: BLE001 - write per-ticker failure and continue.
            failed_rows.append(
                {
                    "ticker": ticker,
                    "status": "failed",
                    "requested_end_date": end_date,
                    "latest_available_date": _coverage_for_ticker(ticker, cache).get("last_date", ""),
                    "reason": str(exc),
                }
            )
            log(step, "failed", str(exc))

    after = [_coverage_for_ticker(ticker, cache, loaded_frame=frames.get(ticker)) for ticker in tickers]
    coverage = _before_after_coverage(before, after, end_date)
    refreshed = pd.DataFrame(refreshed_rows, columns=["ticker", "first_date", "last_date", "row_count", "status", "requested_end_date", "cache_path"])
    failed = pd.DataFrame(failed_rows, columns=["ticker", "status", "requested_end_date", "latest_available_date", "reason"])
    manifest = _manifest(output, tickers, start_date, end_date, coverage, refreshed, failed, cache)

    coverage.to_csv(output / "price_refresh_before_after_coverage.csv", index=False, encoding="utf-8-sig")
    refreshed.to_csv(output / "refreshed_tickers.csv", index=False, encoding="utf-8-sig")
    failed.to_csv(output / "failed_tickers.csv", index=False, encoding="utf-8-sig")
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "final_summary_zh.md").write_text(_summary_zh(manifest, coverage, failed), encoding="utf-8")
    log("completed", manifest["status"], str(output.resolve()))
    (output / "current_step.txt").write_text("completed\n", encoding="utf-8")
    return output


def _cache_path(cache_dir: Path, ticker: str) -> Path:
    return cache_dir / f"{ticker.replace('.', '_')}.csv"


def _coverage_for_ticker(ticker: str, cache_dir: Path, *, loaded_frame: pd.DataFrame | None = None) -> dict[str, object]:
    frame = loaded_frame
    path = _cache_path(cache_dir, ticker)
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
            "cache_path": str(path),
            "cache_exists": path.exists(),
        }
    return {
        "ticker": ticker,
        "first_date": frame.index.min().strftime("%Y-%m-%d"),
        "last_date": frame.index.max().strftime("%Y-%m-%d"),
        "row_count": int(len(frame)),
        "cache_path": str(path),
        "cache_exists": path.exists(),
    }


def _coverage_from_frame(ticker: str, frame: pd.DataFrame) -> dict[str, object]:
    return {
        "ticker": ticker,
        "first_date": frame.index.min().strftime("%Y-%m-%d") if not frame.empty else "",
        "last_date": frame.index.max().strftime("%Y-%m-%d") if not frame.empty else "",
        "row_count": int(len(frame)),
    }


def _before_after_coverage(before: list[dict[str, object]], after: list[dict[str, object]], end_date: str) -> pd.DataFrame:
    before_by_ticker = {str(row["ticker"]): row for row in before}
    rows: list[dict[str, object]] = []
    requested = pd.Timestamp(end_date)
    for row in after:
        ticker = str(row["ticker"])
        previous = before_by_ticker.get(ticker, {})
        after_last = str(row.get("last_date", ""))
        before_last = str(previous.get("last_date", ""))
        latest_is_newer = bool(after_last and (not before_last or pd.Timestamp(after_last) > pd.Timestamp(before_last)))
        gap_days = (requested - pd.Timestamp(after_last)).days if after_last else None
        rows.append(
            {
                "ticker": ticker,
                "before_first_date": previous.get("first_date", ""),
                "before_last_date": before_last,
                "before_row_count": previous.get("row_count", 0),
                "after_first_date": row.get("first_date", ""),
                "after_last_date": after_last,
                "after_row_count": row.get("row_count", 0),
                "requested_end_date": end_date,
                "latest_is_newer_than_before": latest_is_newer,
                "latest_available_date": after_last,
                "calendar_gap_to_requested_end": "" if gap_days is None else gap_days,
                "cache_path": row.get("cache_path", ""),
                "cache_exists": row.get("cache_exists", False),
            }
        )
    return pd.DataFrame(rows)


def _manifest(
    output: Path,
    tickers: tuple[str, ...],
    start_date: str,
    end_date: str,
    coverage: pd.DataFrame,
    refreshed: pd.DataFrame,
    failed: pd.DataFrame,
    cache_dir: Path,
) -> dict[str, object]:
    after_dates = [pd.Timestamp(value) for value in coverage["after_last_date"].astype(str) if value]
    latest_complete = max(after_dates).strftime("%Y-%m-%d") if after_dates else ""
    stale_count = int((coverage["after_last_date"].astype(str) <= "2026-06-12").sum()) if not coverage.empty else 0
    return {
        "schema_version": 1,
        "task_id": TASK_ID,
        "status": "completed" if failed.empty else "partial_completed_with_failures",
        "formal_model_changed": False,
        "trade_decision_changed": False,
        "ticker_count": len(tickers),
        "requested_tickers": list(tickers),
        "start_date": start_date,
        "requested_end_date": end_date,
        "cache_dir": str(cache_dir),
        "refreshed_count": int(len(refreshed)),
        "failed_count": int(len(failed)),
        "latest_available_date_max": latest_complete,
        "stale_at_or_before_2026_06_12_count": stale_count,
        "no_forward_fill_used": True,
        "full_market_download": False,
        "output_dir": str(output.resolve()),
    }


def _summary_zh(manifest: dict[str, object], coverage: pd.DataFrame, failed: pd.DataFrame) -> str:
    lines = [
        "# AI megacap price refresh for MA similarity",
        "",
        "## 結論",
        "",
        f"- 狀態：{manifest['status']}。",
        f"- 指定 tickers：{manifest['ticker_count']} 檔；成功讀取/刷新：{manifest['refreshed_count']}；失敗：{manifest['failed_count']}。",
        f"- 目前可見最大 latest_available_date：{manifest['latest_available_date_max']}。",
        "- 本任務沒有改正式模型、formal target 或 trade decision。",
        "- 本任務只處理指定 9 檔，沒有全市場下載，沒有 forward-fill 缺價。",
        "",
        "## Coverage after refresh",
        "",
    ]
    for row in coverage.to_dict(orient="records"):
        lines.append(f"- {row['ticker']}：{row['before_last_date']} -> {row['after_last_date']}")
    if not failed.empty:
        lines += ["", "## Failed tickers", ""]
        for row in failed.to_dict(orient="records"):
            lines.append(f"- {row['ticker']}：{row['reason']}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh selected AI megacap and benchmark price caches for MA similarity rerun.")
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", default=DEFAULT_END_DATE)
    parser.add_argument("--ticker", action="append", default=[])
    args = parser.parse_args()
    tickers = tuple(args.ticker) if args.ticker else DEFAULT_TICKERS
    output = run_ai_megacap_price_refresh(
        tickers=tickers,
        start_date=args.start_date,
        end_date=args.end_date,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
    )
    print(f"AI_MEGACAP_PRICE_REFRESH_OUTPUT={output.resolve()}")


if __name__ == "__main__":
    main()
