from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from backtest_lab.radar_core_pool_runner import _completed_summary_rows


class RadarCorePoolRunnerTest(unittest.TestCase):
    def test_completed_summary_rows_merges_latest_completed_variants(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "completed_variants.csv"
            with path.open("w", newline="", encoding="utf-8-sig") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "timestamp",
                        "variant_id",
                        "label",
                        "final_value",
                        "total_return_pct",
                        "max_drawdown_pct",
                        "trades",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "timestamp": "2026-06-16 20:00:00",
                        "variant_id": "completed_a",
                        "label": "old row",
                        "final_value": "100.0",
                        "total_return_pct": "0.0",
                        "max_drawdown_pct": "-1.0",
                        "trades": "1",
                    }
                )
                writer.writerow(
                    {
                        "timestamp": "2026-06-16 20:01:00",
                        "variant_id": "completed_a",
                        "label": "latest row",
                        "final_value": "125.432",
                        "total_return_pct": "25.432",
                        "max_drawdown_pct": "-3.21",
                        "trades": "2",
                    }
                )
                writer.writerow(
                    {
                        "timestamp": "2026-06-16 20:02:00",
                        "variant_id": "current_run",
                        "label": "current row",
                        "final_value": "130.0",
                        "total_return_pct": "30.0",
                        "max_drawdown_pct": "-5.0",
                        "trades": "3",
                    }
                )

            rows = _completed_summary_rows(path, exclude_variant_ids={"current_run"})

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["variant_id"], "completed_a")
        self.assertEqual(rows[0]["label"], "latest row")
        self.assertEqual(rows[0]["final_value"], 125.43)
        self.assertEqual(rows[0]["total_return_pct"], 25.43)
        self.assertEqual(rows[0]["max_drawdown_pct"], -3.21)
        self.assertEqual(rows[0]["trades"], 2)


if __name__ == "__main__":
    unittest.main()
