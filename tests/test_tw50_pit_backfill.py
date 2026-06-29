import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from backtest_lab.tw50_pit_backfill import run_tw50_pit_backfill


class Tw50PitBackfillTest(unittest.TestCase):
    def test_builds_resumable_phase2_runner_outputs_without_download(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "data"
            cache = root / "cache"
            output = root / "output"
            data.mkdir()
            cache.mkdir()
            pd.DataFrame(
                [
                    {
                        "effective_date": "2025-06-23",
                        "ticker": "2330.TW",
                        "name": "台積電",
                        "source": "seed_snapshot",
                        "source_updated_at": "2026-06-13",
                    }
                ]
            ).to_csv(data / "tw50_constituents.csv", index=False)
            _price_csv(cache / "0050_TW.csv", "2014-10-31", periods=2600)
            _price_csv(cache / "00631L_TW.csv", "2016-01-04", periods=2000)
            _price_csv(cache / "2330_TW.csv", "2014-10-31", periods=2600)

            run_tw50_pit_backfill(
                constituents_path=data / "tw50_constituents.csv",
                price_roots=(cache,),
                output_dir=output,
            )

            expected_files = {
                "manifest.json",
                "current_step.txt",
                "backfill_job_plan.csv",
                "pit_source_acquisition_plan.csv",
                "provisional_price_backfill_jobs.csv",
                "readiness_ledger.csv",
                "completed.csv",
                "failed.csv",
                "run_log.csv",
                "final_summary_zh.md",
                "manual_pit_constituents_template.csv",
                "normalized_pit_constituents_sample.csv",
                "updated_price_coverage_panel.csv",
            }
            self.assertTrue(expected_files.issubset({path.name for path in output.iterdir()}))
            self.assertTrue((output / "raw_source_archive").is_dir())
            self.assertTrue((output / "normalized").is_dir())
            self.assertTrue((output / "price_backfill").is_dir())
            self.assertTrue((output / "checkpoints").is_dir())

            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertFalse(manifest["formal_model_changed"])
            self.assertFalse(manifest["trade_decision_changed"])
            self.assertFalse(manifest["large_download_started"])
            self.assertTrue(manifest["can_resume"])
            self.assertEqual(manifest["etf_00631l_201411_trading_status"], "confirmed_traded_2014_11")
            self.assertEqual(manifest["exact_historical_total_ticker_count"], "blocked_until_PIT_archive_acquired")
            self.assertIn("formal_target_stream_rebuild", manifest["missing_layers_for_full_previous_best_replay"])

            pit_sources = pd.read_csv(output / "pit_source_acquisition_plan.csv")
            self.assertIn("exact_candidate", set(pit_sources["source_type"]))
            self.assertIn("proxy_current_snapshot", set(pit_sources["source_type"]))

            jobs = pd.read_csv(output / "provisional_price_backfill_jobs.csv")
            self.assertEqual(
                jobs.loc[jobs["ticker"] == "00631L.TW", "job_status"].iloc[0],
                "needs_price_backfill_confirmed_traded",
            )
            self.assertIn(
                "price/source/cache backfill gap",
                jobs.loc[jobs["ticker"] == "00631L.TW", "missing_reason"].iloc[0],
            )
            self.assertEqual(
                jobs.loc[jobs["ticker"] == "0050.TW", "job_status"].iloc[0],
                "price_ready_skip_download",
            )

            readiness = pd.read_csv(output / "readiness_ledger.csv")
            self.assertEqual(
                readiness.loc[readiness["layer"] == "pit_constituents_exact", "status"].iloc[0],
                "blocked_missing_exact_pit",
            )
            self.assertEqual(
                readiness.loc[readiness["layer"] == "formal_target_stream_rebuild", "status"].iloc[0],
                "blocked_missing_2014_2021_target_evidence",
            )
            self.assertEqual(
                readiness.loc[readiness["layer"] == "00631l_price_source_backfill", "status"].iloc[0],
                "confirmed_traded_2014_11_needs_price_source_backfill",
            )

            sample = pd.read_csv(output / "normalized_pit_constituents_sample.csv")
            self.assertFalse(bool(sample["formal_ready"].iloc[0]))
            self.assertEqual(sample["source_type"].iloc[0], "proxy_current_snapshot")


def _price_csv(path: Path, start: str, periods: int) -> None:
    dates = pd.bdate_range(start, periods=periods)
    pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "adj_close": 100.0,
            "volume": 1000,
            "dividend": 0.0,
            "stock_split": 0.0,
        }
    ).to_csv(path, index=False)


if __name__ == "__main__":
    unittest.main()
