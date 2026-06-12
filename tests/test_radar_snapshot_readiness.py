from __future__ import annotations

import csv
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from backtest_lab.radar_snapshot_readiness import REQUIRED_SNAPSHOT_COLUMNS, evaluate_radar_snapshot_readiness


class RadarSnapshotReadinessTest(unittest.TestCase):
    def test_ready_when_snapshots_have_required_columns_and_pass_coverage(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for day in range(1, 21):
                _write_snapshot(root / f"radar_snapshot_202605{day:02d}.csv", fundamental_pass=day <= 5)

            readiness = evaluate_radar_snapshot_readiness(root)

            self.assertTrue(readiness.ready)
            self.assertEqual(readiness.snapshot_count, 20)
            self.assertEqual(readiness.dates_with_fundamental_pass, 5)

    def test_not_ready_when_only_latest_day_has_fundamental_pass(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for day in range(1, 21):
                _write_snapshot(root / f"radar_snapshot_202605{day:02d}.csv", fundamental_pass=day == 20)

            readiness = evaluate_radar_snapshot_readiness(root)

            self.assertFalse(readiness.ready)
            self.assertEqual(readiness.dates_with_fundamental_pass, 1)
            self.assertTrue(any("fundamental_pass" in warning for warning in readiness.warnings))

    def test_not_ready_when_required_columns_are_missing(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            path = root / "radar_snapshot_20260501.csv"
            with path.open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["date", "symbol"])
                writer.writeheader()
                writer.writerow({"date": "2026-05-01", "symbol": "2330"})

            readiness = evaluate_radar_snapshot_readiness(root, min_snapshots=1)

            self.assertFalse(readiness.ready)
            self.assertIn("fundamental_pass", readiness.missing_columns)

    def test_not_ready_when_fundamental_source_date_is_in_the_future(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for day in range(1, 21):
                _write_snapshot(
                    root / f"radar_snapshot_202605{day:02d}.csv",
                    fundamental_pass=True,
                    source_date="2026-05-31",
                )

            readiness = evaluate_radar_snapshot_readiness(root)

            self.assertFalse(readiness.ready)
            self.assertTrue(any("later than snapshot date" in warning for warning in readiness.warnings))


def _write_snapshot(path: Path, *, fundamental_pass: bool, source_date: str = "2026-05-01") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(REQUIRED_SNAPSHOT_COLUMNS))
        writer.writeheader()
        for symbol in ("2330", "2454", "2382", "3037", "2408"):
            row = {column: "0" for column in sorted(REQUIRED_SNAPSHOT_COLUMNS)}
            row.update(
                {
                    "date": "2026-05-01",
                    "theme": "AI伺服器/ODM",
                    "symbol": symbol,
                    "name": symbol,
                    "bucket": "theme_leader" if fundamental_pass else "excluded_missing_fundamental",
                    "fundamental_pass": str(fundamental_pass).lower(),
                    "fundamental_data_status": "ok" if fundamental_pass else "missing_fundamental_data",
                    "fundamental_source_date": source_date if fundamental_pass else "",
                }
            )
            writer.writerow(row)


if __name__ == "__main__":
    unittest.main()
