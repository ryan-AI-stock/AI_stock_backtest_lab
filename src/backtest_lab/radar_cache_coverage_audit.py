from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

from backtest_lab.radar_core_pool_v1 import load_radar_core_members


DEFAULT_RADAR_ROOT = r"C:\Users\zergv\Documents\Codex\2026-05-23\ai-stock-rotation-radar-https-docs"
DEFAULT_BACKTEST_CACHE_ROOT = "backtest_cache"
DEFAULT_SELECTED_CACHE_DIR = "backtest_cache/radar_core_pool_v1_selected_themes"
DEFAULT_OUTPUT_DIR = "outputs/sector_dynamic_pool/radar_cache_coverage_audit/latest"


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit local price cache coverage for radar core members.")
    parser.add_argument("--radar-root", default=DEFAULT_RADAR_ROOT)
    parser.add_argument("--backtest-cache-root", default=DEFAULT_BACKTEST_CACHE_ROOT)
    parser.add_argument("--selected-cache-dir", default=DEFAULT_SELECTED_CACHE_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    members = load_radar_core_members(Path(args.radar_root) / "data" / "theme_map.csv")
    selected_cache = Path(args.selected_cache_dir)
    cache_root = Path(args.backtest_cache_root)
    all_cache_files = _index_cache_files(cache_root)

    rows: list[dict[str, str]] = []
    for member in members:
        filename = _cache_filename(member.ticker)
        selected_path = selected_cache / filename
        anywhere_paths = sorted(all_cache_files.get(filename, []))
        status = "selected_cache" if selected_path.exists() else "other_cache" if anywhere_paths else "missing"
        rows.append(
            {
                "symbol": member.symbol,
                "ticker": member.ticker,
                "name": member.name,
                "theme": member.theme,
                "status": status,
                "selected_cache_path": str(selected_path) if selected_path.exists() else "",
                "other_cache_paths": ";".join(str(path) for path in anywhere_paths if path != selected_path),
            }
        )

    _write_csv(output_dir / "cache_coverage.csv", rows)
    _write_report(output_dir / "cache_coverage_report.md", rows, selected_cache, cache_root)
    (output_dir / "current_step.txt").write_text("completed\n", encoding="utf-8")


def _cache_filename(ticker: str) -> str:
    return ticker.replace(".", "_") + ".csv"


def _index_cache_files(cache_root: Path) -> dict[str, list[Path]]:
    files: dict[str, list[Path]] = defaultdict(list)
    if not cache_root.exists():
        return files
    for path in cache_root.rglob("*.csv"):
        files[path.name].append(path)
    return files


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = ["symbol", "ticker", "name", "theme", "status", "selected_cache_path", "other_cache_paths"]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_report(path: Path, rows: list[dict[str, str]], selected_cache: Path, cache_root: Path) -> None:
    status_counts = Counter(row["status"] for row in rows)
    theme_missing = Counter(row["theme"] for row in rows if row["status"] == "missing")
    theme_other = Counter(row["theme"] for row in rows if row["status"] == "other_cache")
    examples_missing = [row for row in rows if row["status"] == "missing"][:20]
    examples_other = [row for row in rows if row["status"] == "other_cache"][:20]

    lines = [
        "# Radar core pool price cache coverage audit",
        "",
        f"- selected_cache_dir: `{selected_cache}`",
        f"- backtest_cache_root: `{cache_root}`",
        f"- total_primary_members: {len(rows)}",
        f"- selected_cache: {status_counts.get('selected_cache', 0)}",
        f"- other_cache_only: {status_counts.get('other_cache', 0)}",
        f"- missing_everywhere: {status_counts.get('missing', 0)}",
        "",
        "## Missing by theme",
        "",
        *_counter_lines(theme_missing),
        "",
        "## Other-cache-only by theme",
        "",
        *_counter_lines(theme_other),
        "",
        "## Missing examples",
        "",
        *_example_lines(examples_missing),
        "",
        "## Other-cache-only examples",
        "",
        *_example_lines(examples_other),
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _counter_lines(counter: Counter[str]) -> list[str]:
    if not counter:
        return ["- none"]
    return [f"- {theme}: {count}" for theme, count in counter.most_common()]


def _example_lines(rows: list[dict[str, str]]) -> list[str]:
    if not rows:
        return ["- none"]
    return [f"- {row['symbol']} {row['name']} ({row['theme']})" for row in rows]


if __name__ == "__main__":
    main()
