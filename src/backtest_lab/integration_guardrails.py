from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BenchmarkGuard:
    name: str
    path: Path
    variant_id: str
    final_value: float
    total_return_pct: float
    max_drawdown_pct: float
    trades: int
    final_value_tolerance: float = 1.0
    pct_tolerance: float = 0.01


DEFAULT_GUARDS = (
    BenchmarkGuard(
        name="large_cap_best_v20260605_2022_2023",
        path=Path("outputs/sector_dynamic_pool/benchmark_v20260605_2022_2023/latest/summary.csv"),
        variant_id="best_v20260605",
        final_value=3_540_377.00,
        total_return_pct=254.04,
        max_drawdown_pct=-21.09,
        trades=22,
    ),
    BenchmarkGuard(
        name="radar_core_mid_small_calibrated_v1_2022_2023",
        path=Path("outputs/sector_dynamic_pool/radar_core_pool_refactor_check_v1/latest/radar_core_pool_v1_summary.csv"),
        variant_id="radar_core_v1_score_risk_stock00_turnover60m_overheat62",
        final_value=5_252_467.85,
        total_return_pct=425.25,
        max_drawdown_pct=-28.51,
        trades=96,
    ),
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Check core benchmark results after integration/refactor work.")
    parser.add_argument("--root", default=".", help="Repo root that contains outputs/.")
    args = parser.parse_args()

    root = Path(args.root)
    failures: list[str] = []
    for guard in DEFAULT_GUARDS:
        failures.extend(check_guard(root, guard))

    if failures:
        for failure in failures:
            print(f"FAIL {failure}")
        raise SystemExit(1)

    for guard in DEFAULT_GUARDS:
        print(f"OK {guard.name}: {guard.variant_id}")


def check_guard(root: Path, guard: BenchmarkGuard) -> list[str]:
    path = root / guard.path
    if not path.exists():
        return [f"{guard.name}: missing summary {path}"]
    rows = _read_csv(path)
    row = _find_row(rows, guard.variant_id)
    if row is None:
        return [f"{guard.name}: missing variant_id {guard.variant_id} in {path}"]

    failures: list[str] = []
    failures.extend(_check_float(guard.name, row, ("final_value", "final_value_twd"), guard.final_value, guard.final_value_tolerance))
    failures.extend(_check_float(guard.name, row, ("total_return_pct",), guard.total_return_pct, guard.pct_tolerance))
    failures.extend(_check_float(guard.name, row, ("max_drawdown_pct",), guard.max_drawdown_pct, guard.pct_tolerance))
    failures.extend(_check_int(guard.name, row, ("trades", "trade_count"), guard.trades))
    return failures


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _find_row(rows: list[dict[str, str]], variant_id: str) -> dict[str, str] | None:
    for row in rows:
        if row.get("variant_id") == variant_id or row.get("candidate_id") == variant_id or row.get("model_id") == variant_id:
            return row
    return None


def _check_float(
    guard_name: str,
    row: dict[str, str],
    columns: tuple[str, ...],
    expected: float,
    tolerance: float,
) -> list[str]:
    value_text = _first_present(row, columns)
    if value_text is None:
        return [f"{guard_name}: missing columns {columns}"]
    actual = float(value_text)
    if abs(actual - expected) > tolerance:
        return [f"{guard_name}: {columns[0]} expected {expected}, got {actual}"]
    return []


def _check_int(guard_name: str, row: dict[str, str], columns: tuple[str, ...], expected: int) -> list[str]:
    value_text = _first_present(row, columns)
    if value_text is None:
        return [f"{guard_name}: missing columns {columns}"]
    actual = int(float(value_text))
    if actual != expected:
        return [f"{guard_name}: {columns[0]} expected {expected}, got {actual}"]
    return []


def _first_present(row: dict[str, str], columns: tuple[str, ...]) -> str | None:
    for column in columns:
        value = row.get(column)
        if value not in (None, ""):
            return value
    return None


if __name__ == "__main__":
    main()
