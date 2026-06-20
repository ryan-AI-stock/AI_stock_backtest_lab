from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

import test_paths  # noqa: F401

from backtest_lab.tw50_constituent_coverage import run_tw50_constituent_coverage


class Tw50ConstituentCoverageTest(unittest.TestCase):
    def test_coverage_ready_when_every_period_has_enough_active_constituents(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "tw50.csv"
            _write_constituents(source, "2022-01-01", count=46)

            metadata = run_tw50_constituent_coverage(
                constituent_path=source,
                output_dir=root / "out",
                periods={"sample": ("2022-01-03", "2022-01-05")},
                minimum_active_count=45,
            )

            self.assertEqual(metadata["readiness_status"], "ready")
            summary = pd.read_csv(root / "out" / "tw50_constituent_coverage_summary.csv")
            self.assertEqual(int(summary.iloc[0]["ready_dates"]), 3)
            self.assertEqual(float(summary.iloc[0]["coverage_ratio"]), 1.0)

    def test_coverage_blocks_when_snapshot_starts_after_period(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "tw50.csv"
            _write_constituents(source, "2025-06-23", count=50)

            metadata = run_tw50_constituent_coverage(
                constituent_path=source,
                output_dir=root / "out",
                periods={"sample": ("2022-01-03", "2022-01-05")},
                minimum_active_count=45,
            )

            self.assertEqual(metadata["readiness_status"], "blocked_no_historical_coverage")
            gaps = pd.read_csv(root / "out" / "tw50_constituent_gap_dates.csv")
            self.assertEqual(len(gaps), 3)
            self.assertTrue(gaps["gap_reason"].str.contains("No TW50 constituents active").all())


def _write_constituents(path: Path, effective_date: str, *, count: int) -> None:
    rows = [
        {
            "effective_date": effective_date,
            "ticker": f"{1000 + index}.TW",
            "name": f"Stock{index}",
            "source": "test",
            "source_updated_at": "2026-06-20",
        }
        for index in range(count)
    ]
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    unittest.main()
