import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from backtest_lab.price_and_target_gap_backfill import run_price_and_target_gap_backfill


class PriceAndTargetGapBackfillTest(unittest.TestCase):
    def test_builds_phase3_price_and_target_gap_ledgers(self) -> None:
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

            run_price_and_target_gap_backfill(
                constituents_path=data / "tw50_constituents.csv",
                price_roots=(cache,),
                output_dir=output,
            )

            expected_files = {
                "manifest.json",
                "current_step.txt",
                "00631l_price_backfill_jobs.csv",
                "provisional_universe_price_backfill_jobs.csv",
                "price_source_priority.csv",
                "target_stream_reconstruction_gaps.csv",
                "readiness_ledger.csv",
                "completed.csv",
                "failed.csv",
                "run_log.csv",
                "final_summary_zh.md",
            }
            self.assertTrue(expected_files.issubset({path.name for path in output.iterdir()}))

            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertFalse(manifest["formal_model_changed"])
            self.assertFalse(manifest["trade_decision_changed"])
            self.assertFalse(manifest["large_download_started"])
            self.assertFalse(manifest["network_fetch_started"])
            self.assertFalse(manifest["synthetic_00631l_used"])
            self.assertFalse(manifest["provisional_universe_is_complete_history"])
            self.assertEqual(manifest["00631l_monthly_twse_jobs"], 14)

            jobs_00631l = pd.read_csv(output / "00631l_price_backfill_jobs.csv")
            self.assertEqual(len(jobs_00631l), 14)
            self.assertEqual(set(jobs_00631l["source"]), {"TWSE_STOCK_DAY"})
            self.assertEqual(set(jobs_00631l["synthetic_allowed"]), {False})
            self.assertIn("1031101", set(jobs_00631l["twse_query_date"].astype(str)))

            provisional = pd.read_csv(output / "provisional_universe_price_backfill_jobs.csv")
            self.assertEqual(
                provisional.loc[provisional["ticker"] == "00631L.TW", "job_status"].iloc[0],
                "ready_for_00631l_real_price_backfill",
            )
            self.assertTrue(bool(provisional.loc[provisional["ticker"] == "00631L.TW", "not_complete_historical_universe"].iloc[0]))

            gaps = pd.read_csv(output / "target_stream_reconstruction_gaps.csv")
            self.assertIn("pool1_candidate_ranking_scores", set(gaps["layer"]))
            self.assertIn("pool2_confirmation_state", set(gaps["layer"]))
            self.assertIn("formal_target_stream", set(gaps["layer"]))

            readiness = pd.read_csv(output / "readiness_ledger.csv")
            self.assertEqual(
                readiness.loc[readiness["area"] == "00631l_real_price_backfill", "status"].iloc[0],
                "jobs_ready_not_downloaded",
            )


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
