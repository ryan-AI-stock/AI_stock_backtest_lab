from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

from backtest_lab.data import download_yfinance_prices


DEFAULT_COVERAGE_CSV = "outputs/sector_dynamic_pool/radar_cache_coverage_audit/latest/cache_coverage.csv"
DEFAULT_SELECTED_CACHE_DIR = "backtest_cache/radar_core_pool_v1_selected_themes"
DEFAULT_OUTPUT_DIR = "outputs/sector_dynamic_pool/radar_cache_fill/latest"


def main() -> None:
    parser = argparse.ArgumentParser(description="Fill radar selected price cache from existing caches and yfinance.")
    parser.add_argument("--coverage-csv", default=DEFAULT_COVERAGE_CSV)
    parser.add_argument("--selected-cache-dir", default=DEFAULT_SELECTED_CACHE_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--start-date", default="2020-01-01")
    parser.add_argument("--end-date", default="2023-12-29")
    parser.add_argument("--copy-only", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="Maximum missing tickers to download. 0 means all.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_cache = Path(args.selected_cache_dir)
    selected_cache.mkdir(parents=True, exist_ok=True)

    rows = _read_coverage(Path(args.coverage_csv))
    run_rows = _read_run_log(output_dir / "run_log.csv")
    done = {row["ticker"] for row in run_rows if row["status"] in {"copied", "downloaded"}}

    _write_step(output_dir, "copy_other_cache")
    for row in rows:
        if row["ticker"] in done or row["status"] != "other_cache":
            continue
        source = _first_other_cache_path(row)
        if not source:
            run_rows.append(_log_row(row, "failed", "other_cache_path_missing"))
            _write_run_log(output_dir / "run_log.csv", run_rows)
            continue
        destination = selected_cache / _cache_filename(row["ticker"])
        shutil.copy2(source, destination)
        run_rows.append(_log_row(row, "copied", str(source)))
        done.add(row["ticker"])
        _write_run_log(output_dir / "run_log.csv", run_rows)

    if args.copy_only:
        _write_step(output_dir, "completed_copy_only")
        return

    _write_step(output_dir, "download_missing")
    attempted_count = 0
    for row in rows:
        if row["ticker"] in done or row["status"] != "missing":
            continue
        if args.limit and attempted_count >= args.limit:
            break
        attempted_count += 1
        try:
            download_yfinance_prices(
                [row["ticker"]],
                start_date=args.start_date,
                end_date=args.end_date,
                cache_dir=selected_cache,
            )
            run_rows.append(_log_row(row, "downloaded", ""))
            done.add(row["ticker"])
        except Exception as exc:  # noqa: BLE001 - cache filler records individual failures and continues.
            run_rows.append(_log_row(row, "failed", str(exc)))
        _write_run_log(output_dir / "run_log.csv", run_rows)

    _write_step(output_dir, "completed")


def _read_coverage(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _read_run_log(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _write_run_log(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = ["ticker", "symbol", "name", "theme", "status", "message"]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_step(output_dir: Path, step: str) -> None:
    (output_dir / "current_step.txt").write_text(step + "\n", encoding="utf-8")


def _first_other_cache_path(row: dict[str, str]) -> Path | None:
    raw_paths = [item for item in row.get("other_cache_paths", "").split(";") if item]
    if not raw_paths:
        return None
    return Path(raw_paths[0])


def _cache_filename(ticker: str) -> str:
    return ticker.replace(".", "_") + ".csv"


def _log_row(row: dict[str, str], status: str, message: str) -> dict[str, str]:
    return {
        "ticker": row["ticker"],
        "symbol": row["symbol"],
        "name": row["name"],
        "theme": row["theme"],
        "status": status,
        "message": message,
    }


if __name__ == "__main__":
    main()
