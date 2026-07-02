from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

import test_paths  # noqa: F401

from backtest_lab.pool2_date_batched_persistence_reconstruction import (
    run_pool2_date_batched_persistence_reconstruction,
)


class Pool2DateBatchedPersistenceReconstructionTest(unittest.TestCase):
    def test_runner_writes_progress_and_eligible_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache = root / "cache"
            pool1_status = root / "pool1"
            output = root / "out"
            cache.mkdir()
            pool1_status.mkdir()

            dates = pd.bdate_range(end="2020-08-31", periods=170)
            _price_frame(dates, start=100.0, daily_step=0.02).to_csv(cache / "0050_TW.csv", index=False)
            _price_frame(dates, start=100.0, daily_step=0.35).to_csv(cache / "2330_TW.csv", index=False)
            anchor = root / "anchor.csv"
            pd.DataFrame(
                [
                    _anchor_row("2020-08", "2020-08-31", "0050", "0050"),
                    _anchor_row("2020-08", "2020-08-31", "2330", "台積電"),
                ]
            ).to_csv(anchor, index=False)
            (pool1_status / "manifest.json").write_text(
                json.dumps({"pool1_blocked_rows": 0}, ensure_ascii=False),
                encoding="utf-8",
            )

            result = run_pool2_date_batched_persistence_reconstruction(
                output_dir=output,
                start_date="2020-08-31",
                end_date="2020-08-31",
                batch_size=1,
                price_cache_dir=cache,
                monthly_anchor_path=anchor,
                price_source_registry=root / "missing_registry.csv",
                pool1_status_output=pool1_status,
            )

            self.assertEqual(result, output)
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["executed_batches"], 1)
            self.assertGreater(manifest["persistence_passed_rows"], 0)
            self.assertGreater(manifest["eligible_for_pool_selection_rows"], 0)
            self.assertFalse(manifest["formal_model_changed"])
            self.assertFalse(manifest["trade_decision_changed"])
            self.assertTrue((output / "run_log.csv").exists())
            self.assertTrue((output / "current_step.txt").exists())
            eligible = pd.read_csv(output / "pool2_reconstructed_eligible_rows.csv")
            self.assertIn("2330.TW", set(eligible["candidate_ticker"]))


def _price_frame(dates: pd.DatetimeIndex, *, start: float, daily_step: float) -> pd.DataFrame:
    values = [start + idx * daily_step for idx in range(len(dates))]
    return pd.DataFrame(
        {
            "date": [date.strftime("%Y-%m-%d") for date in dates],
            "open": values,
            "high": [value * 1.01 for value in values],
            "low": [value * 0.99 for value in values],
            "close": values,
            "adj_close": values,
            "volume": [20_000_000] * len(dates),
        }
    )


def _anchor_row(month: str, date: str, ticker: str, name: str) -> dict[str, str]:
    return {
        "effective_month": month,
        "effective_date": date,
        "ticker": ticker,
        "name": name,
        "source_url": "test",
        "raw_source_id": "test",
        "source_type": "source_backed_manual_candidate",
        "formal_exact": "false",
    }


if __name__ == "__main__":
    unittest.main()
