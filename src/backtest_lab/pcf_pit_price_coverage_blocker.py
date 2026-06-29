from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import pandas as pd

from backtest_lab.data import download_yfinance_prices, load_price_csv
from backtest_lab.pcf_pit_candidate_adapter import DEFAULT_MONTHLY_ANCHOR_PATH, load_0050_pcf_monthly_anchor
from backtest_lab.tw50_constituent_price_backfill import DEFAULT_CACHE_DIR


TASK_ID = "TASK-BACKTEST-CORE-0050-PIT-PRICE-COVERAGE-BLOCKER-201411-202312-20260629"
DEFAULT_OUTPUT_DIR = "outputs/core_0050_pit_price_coverage_blocker_201411_202312_20260629"
DEFAULT_REPLAY_END_DATE = "2023-12-31"
PRICE_DOWNLOADER = Callable[[list[str], str, str, str | Path], dict[str, pd.DataFrame]]


def run_0050_pit_price_coverage_blocker(
    *,
    monthly_anchor_path: str | Path = DEFAULT_MONTHLY_ANCHOR_PATH,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    replay_end_date: str = DEFAULT_REPLAY_END_DATE,
    refresh_missing: bool = False,
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

    try:
        log("load_monthly_anchor", "started", str(monthly_anchor_path))
        anchor = load_0050_pcf_monthly_anchor(monthly_anchor_path)
        requirements = _ticker_requirements(anchor, replay_end_date)
        before = _coverage_status(requirements, cache)
        gap_before = before[before["coverage_status"].ne("price_only_ready")].copy()

        completed_rows: list[dict[str, object]] = []
        failed_rows: list[dict[str, object]] = []
        if refresh_missing and not gap_before.empty:
            fetch = downloader or _download_one_ticker
            for row in gap_before.to_dict(orient="records"):
                ticker = str(row["ticker"])
                start = str(row["required_start_date"])
                end = str(row["required_end_date"])
                step = f"refresh_{ticker}"
                log(step, "started", f"{start} to {end}")
                try:
                    downloaded = fetch([ticker], start, end, cache)
                    frame = downloaded.get(ticker)
                    if frame is None or frame.empty:
                        raise ValueError("download returned no price frame")
                    cov = _coverage_for_ticker(ticker, cache, loaded_frame=frame)
                    completed_rows.append(
                        {
                            "ticker": ticker,
                            "status": "completed",
                            "first_date": cov["first_date"],
                            "last_date": cov["last_date"],
                            "row_count": cov["row_count"],
                            "cache_path": cov["cache_path"],
                            "source": "yfinance_cache_refresh",
                        }
                    )
                    log(step, "completed", f"{cov['first_date']} to {cov['last_date']}")
                except Exception as exc:  # noqa: BLE001 - per-ticker failure must be observable.
                    cov = _coverage_for_ticker(ticker, cache)
                    failed_rows.append(
                        {
                            "ticker": ticker,
                            "status": "failed",
                            "latest_available_date": cov.get("last_date", ""),
                            "reason": str(exc),
                            "cache_path": cov.get("cache_path", ""),
                        }
                    )
                    log(step, "failed", str(exc))

        after = _coverage_status(requirements, cache)
        missing = after[after["coverage_status"].eq("missing_coverage_row")].copy()
        not_ready = after[after["coverage_status"].eq("not_ready")].copy()
        ready = after[after["coverage_status"].eq("price_only_ready")].copy()
        blockers = _remaining_blockers(after)

        after.to_csv(output / "pit_universe_price_coverage_status.csv", index=False, encoding="utf-8-sig")
        missing.to_csv(output / "missing_price_coverage_tickers.csv", index=False, encoding="utf-8-sig")
        not_ready.to_csv(output / "not_ready_price_coverage_tickers.csv", index=False, encoding="utf-8-sig")
        blockers.to_csv(output / "remaining_blockers_after_price.csv", index=False, encoding="utf-8-sig")
        completed_columns = ["ticker", "status", "first_date", "last_date", "row_count", "cache_path", "source"]
        failed_columns = ["ticker", "status", "latest_available_date", "reason", "cache_path"]
        pd.DataFrame(completed_rows, columns=completed_columns).to_csv(
            output / "price_backfill_completed.csv",
            index=False,
            encoding="utf-8-sig",
        )
        pd.DataFrame(failed_rows, columns=failed_columns).to_csv(
            output / "price_backfill_failed.csv",
            index=False,
            encoding="utf-8-sig",
        )
        (output / "price_data_fill_plan.md").write_text(_price_data_fill_plan(after, refresh_missing), encoding="utf-8")
        (output / "price_data_fill_task_for_data_thread.md").write_text(_data_thread_task(after), encoding="utf-8")

        manifest = {
            "task_id": TASK_ID,
            "generated_at": pd.Timestamp.now(tz="Asia/Taipei").isoformat(),
            "monthly_anchor_path": str(monthly_anchor_path),
            "cache_dir": str(cache),
            "refresh_missing_attempted": bool(refresh_missing),
            "pit_universe_tickers": int(after["ticker"].nunique()),
            "price_ready_tickers": int(len(ready)),
            "missing_coverage_tickers": int(len(missing)),
            "not_ready_tickers": int(len(not_ready)),
            "price_backfill_completed_count": int(len(completed_rows)),
            "price_backfill_failed_count": int(len(failed_rows)),
            "price_blocker_cleared": bool(missing.empty and not_ready.empty),
            "formal_model_changed": False,
            "trade_decision_changed": False,
            "formal_exact": False,
            "strategy_ready": False,
        }
        (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        (output / "final_summary_zh.md").write_text(_summary_zh(manifest, after), encoding="utf-8")
        pd.DataFrame([{"step": "run_0050_pit_price_coverage_blocker", "status": "completed"}]).to_csv(
            output / "completed.csv", index=False, encoding="utf-8-sig"
        )
        pd.DataFrame(columns=["step", "status", "reason"]).to_csv(output / "failed.csv", index=False, encoding="utf-8-sig")
        log("completed", "completed", str(output.resolve()))
        return output
    except Exception as exc:
        pd.DataFrame([{"step": "run_0050_pit_price_coverage_blocker", "status": "failed", "reason": str(exc)}]).to_csv(
            output / "failed.csv", index=False, encoding="utf-8-sig"
        )
        log("failed", "failed", str(exc))
        raise


def _ticker_requirements(anchor: pd.DataFrame, replay_end_date: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for ticker, group in anchor.groupby("ticker"):
        ordered = group.sort_values(["effective_date", "effective_month"])
        first = ordered.iloc[0]
        last = ordered.iloc[-1]
        rows.append(
            {
                "ticker": _tw(str(ticker)),
                "name": str(first["name"]),
                "first_anchor_month": str(first["effective_month"]),
                "first_anchor_date": pd.Timestamp(first["effective_date"]).strftime("%Y-%m-%d"),
                "last_anchor_month": str(last["effective_month"]),
                "last_anchor_date": pd.Timestamp(last["effective_date"]).strftime("%Y-%m-%d"),
                "anchor_month_count": int(group["effective_month"].nunique()),
                "required_start_date": pd.Timestamp(first["effective_date"]).strftime("%Y-%m-%d"),
                "required_end_date": min(pd.Timestamp(last["effective_date"]), pd.Timestamp(replay_end_date)).strftime("%Y-%m-%d"),
            }
        )
    return pd.DataFrame(rows).sort_values("ticker").reset_index(drop=True)


def _coverage_status(requirements: pd.DataFrame, cache_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for req in requirements.to_dict(orient="records"):
        ticker = str(req["ticker"])
        cov = _coverage_for_ticker(ticker, cache_dir)
        first = str(cov.get("first_date", ""))
        last = str(cov.get("last_date", ""))
        adjusted = _as_bool(cov.get("adjusted_close_available"))
        required_start = pd.Timestamp(str(req["required_start_date"]))
        required_end = pd.Timestamp(str(req["required_end_date"]))
        first_ok = bool(first and pd.Timestamp(first) <= required_start + pd.Timedelta(days=10))
        last_ok = bool(last and pd.Timestamp(last) >= required_end - pd.Timedelta(days=10))
        if not first:
            status = "missing_coverage_row"
        elif first_ok and last_ok and adjusted:
            status = "price_only_ready"
        else:
            status = "not_ready"
        rows.append(
            {
                **req,
                "coverage_status": status,
                "first_date": first,
                "last_date": last,
                "row_count": cov.get("row_count", 0),
                "adjusted_close_available": str(adjusted).lower(),
                "ready_for_backtest_price_only": str(status == "price_only_ready").lower(),
                "cache_path": cov.get("cache_path", ""),
                "missing_periods": _missing_period_text(first, last, str(req["required_start_date"]), str(req["required_end_date"])),
            }
        )
    return pd.DataFrame(rows)


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
        }
    return {
        "ticker": ticker,
        "first_date": frame.index.min().strftime("%Y-%m-%d"),
        "last_date": frame.index.max().strftime("%Y-%m-%d"),
        "row_count": int(len(frame)),
        "adjusted_close_available": "adj_close" in frame.columns and pd.to_numeric(frame["adj_close"], errors="coerce").notna().any(),
        "cache_path": str(path),
    }


def _download_one_ticker(tickers: list[str], start_date: str, end_date: str, cache_dir: str | Path) -> dict[str, pd.DataFrame]:
    return download_yfinance_prices(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        cache_dir=cache_dir,
        allow_edge_gap=True,
    )


def _cache_path(cache_dir: Path, ticker: str) -> Path:
    return cache_dir / f"{ticker.replace('.', '_')}.csv"


def _tw(ticker: str) -> str:
    return ticker if ticker.endswith(".TW") else f"{ticker}.TW"


def _missing_period_text(first: str, last: str, start_date: str, end_date: str) -> str:
    periods: list[str] = []
    if not first:
        return f"{start_date}~{end_date}: no local price data"
    if pd.Timestamp(first) > pd.Timestamp(start_date) + pd.Timedelta(days=10):
        periods.append(f"{start_date}~{first}: no price before first available date")
    if not last or pd.Timestamp(last) < pd.Timestamp(end_date) - pd.Timedelta(days=10):
        periods.append(f"{last or 'unknown'}~{end_date}: no latest price coverage")
    return "; ".join(periods)


def _remaining_blockers(status: pd.DataFrame) -> pd.DataFrame:
    missing = int((status["coverage_status"] == "missing_coverage_row").sum())
    not_ready = int((status["coverage_status"] == "not_ready").sum())
    return pd.DataFrame(
        [
            {
                "blocker": "pit_universe_price_coverage",
                "status": "cleared" if missing == 0 and not_ready == 0 else "partial",
                "blocks_2014_2023_backtest": missing > 0 or not_ready > 0,
                "detail": f"missing={missing}; not_ready={not_ready}; ready={int((status['coverage_status'] == 'price_only_ready').sum())}",
                "next_owner": "Core/Data",
            },
            {
                "blocker": "formal_target_signal_stream_2014_2021",
                "status": "missing",
                "blocks_2014_2023_backtest": True,
                "detail": "Pool1 ranking, Pool2 confirmation, score margin, and formal target stream remain missing.",
                "next_owner": "Core/Research/Experiments",
            },
            {
                "blocker": "execution_ledger_2014_2021",
                "status": "missing",
                "blocks_2014_2023_backtest": True,
                "detail": "Execution ledger can be rebuilt only after target stream exists.",
                "next_owner": "Core/Experiments",
            },
        ]
    )


def _price_data_fill_plan(status: pd.DataFrame, refresh_missing: bool) -> str:
    missing = status[status["coverage_status"] == "missing_coverage_row"]
    not_ready = status[status["coverage_status"] == "not_ready"]
    return f"""# 0050 PIT universe price data fill plan

## Current status

- PIT universe tickers: {status['ticker'].nunique()}
- price_only_ready: {(status['coverage_status'] == 'price_only_ready').sum()}
- missing coverage rows: {len(missing)}
- not ready rows: {len(not_ready)}
- refresh attempted in this run: {str(refresh_missing).lower()}

## Rule

Coverage is judged from each ticker's first monthly-anchor appearance, not blindly from 2014-11 for every ticker. This avoids falsely blocking newer constituents such as 6669.TW when they already have prices before their first anchor month.

## Next action

If this run did not clear all gaps, Core/Data should refresh only the remaining missing/not-ready tickers from their `required_start_date` through 2023-12-31, record per-ticker failures, and avoid forward-filling or synthetic prices.
"""


def _data_thread_task(status: pd.DataFrame) -> str:
    gaps = status[status["coverage_status"].ne("price_only_ready")].copy()
    tickers = ", ".join(gaps["ticker"].tolist())
    return f"""# Task for Core/Data: fill 0050 PIT universe price coverage

Target period: each ticker's first 0050 PCF monthly anchor date through 2023-12-31.

Tickers needing work ({len(gaps)}):
{tickers}

Requirements:
- Use real adjusted price source compatible with BACKTEST_LAB cache schema.
- Do not forward-fill missing history.
- Do not use synthetic prices.
- Preserve source, first_date, last_date, adjusted_close_available, failed reason.
- Output updated coverage matrix and failed ledger.

Boundaries:
- formal_model_changed=false
- trade_decision_changed=false
- formal_exact=false for PCF PIT candidate layer
"""


def _summary_zh(manifest: dict[str, object], status: pd.DataFrame) -> str:
    return f"""# 0050 PIT universe price coverage blocker

## 結論

- PIT universe tickers：{manifest['pit_universe_tickers']}
- price-ready：{manifest['price_ready_tickers']}
- missing coverage：{manifest['missing_coverage_tickers']}
- not ready：{manifest['not_ready_tickers']}
- refresh attempted：{manifest['refresh_missing_attempted']}
- price blocker cleared：{manifest['price_blocker_cleared']}

這一棒只處理價格覆蓋 blocker，不改正式 selector、不改交易決策、不把 PCF candidate 升成 formal exact。

## 重要口徑

價格需求起點是每檔股票第一次出現在 0050 monthly anchor 的日期，而不是一律要求所有股票從 2014-11 起有資料。

## 剩餘工作

若 price blocker 尚未清完，請依 `missing_price_coverage_tickers.csv`、`not_ready_price_coverage_tickers.csv` 與 `price_data_fill_task_for_data_thread.md` 接續處理。
"""


def _as_bool(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build or refresh 0050 PIT universe price coverage blocker package.")
    parser.add_argument("--monthly-anchor-path", default=DEFAULT_MONTHLY_ANCHOR_PATH)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--replay-end-date", default=DEFAULT_REPLAY_END_DATE)
    parser.add_argument("--refresh-missing", action="store_true")
    args = parser.parse_args()
    output = run_0050_pit_price_coverage_blocker(
        monthly_anchor_path=args.monthly_anchor_path,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        replay_end_date=args.replay_end_date,
        refresh_missing=args.refresh_missing,
    )
    print(f"0050_PIT_PRICE_COVERAGE_BLOCKER_OUTPUT={output.resolve()}")


if __name__ == "__main__":
    main()
