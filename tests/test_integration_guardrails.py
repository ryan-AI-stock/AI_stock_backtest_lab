from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from backtest_lab.integration_guardrails import BenchmarkGuard, check_guard


class IntegrationGuardrailsTest(unittest.TestCase):
    def test_check_guard_passes_matching_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            summary = root / "summary.csv"
            _write_summary(summary, final_value=100.0, total_return_pct=10.0, max_drawdown_pct=-5.0, trades=2)

            failures = check_guard(
                root,
                BenchmarkGuard(
                    name="sample",
                    path=Path("summary.csv"),
                    variant_id="winner",
                    final_value=100.0,
                    total_return_pct=10.0,
                    max_drawdown_pct=-5.0,
                    trades=2,
                ),
            )

            self.assertEqual(failures, [])

    def test_check_guard_fails_when_value_drifts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            summary = root / "summary.csv"
            _write_summary(summary, final_value=99.0, total_return_pct=10.0, max_drawdown_pct=-5.0, trades=2)

            failures = check_guard(
                root,
                BenchmarkGuard(
                    name="sample",
                    path=Path("summary.csv"),
                    variant_id="winner",
                    final_value=100.0,
                    total_return_pct=10.0,
                    max_drawdown_pct=-5.0,
                    trades=2,
                    final_value_tolerance=0.1,
                ),
            )

            self.assertIn("sample: final_value expected 100.0, got 99.0", failures)


def _write_summary(
    path: Path,
    *,
    final_value: float,
    total_return_pct: float,
    max_drawdown_pct: float,
    trades: int,
) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["variant_id", "final_value", "total_return_pct", "max_drawdown_pct", "trades"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "variant_id": "winner",
                "final_value": final_value,
                "total_return_pct": total_return_pct,
                "max_drawdown_pct": max_drawdown_pct,
                "trades": trades,
            }
        )


if __name__ == "__main__":
    unittest.main()
