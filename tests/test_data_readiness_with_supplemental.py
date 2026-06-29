import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from backtest_lab.data_readiness_with_supplemental import run_data_readiness_with_supplemental


class DataReadinessWithSupplementalTest(unittest.TestCase):
    def test_combines_00631l_base_cache_with_supplemental_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "data"
            cache = root / "cache"
            normalized = data / "normalized_prices"
            output = root / "output"
            data.mkdir()
            cache.mkdir()
            normalized.mkdir()
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
            _price_csv(cache / "00631L_TW.csv", "2016-01-04", periods=2100)
            _price_csv(cache / "2330_TW.csv", "2014-10-31", periods=2600)
            supplemental = normalized / "00631L_twse_stock_day_201411_201512.csv"
            _price_csv(supplemental, "2014-11-03", periods=288)
            pd.DataFrame(
                [
                    {
                        "ticker": "00631L.TW",
                        "source_id": "00631l_twse_stock_day_201411_201512",
                        "source_path": supplemental.as_posix(),
                        "source_type": "twse_stock_day_backfill",
                        "first_date": "2014-11-03",
                        "last_date": "2015-12-31",
                        "price_source_ready": True,
                        "strategy_ready": False,
                        "synthetic_used": False,
                        "provenance": "test",
                        "notes": "test",
                    }
                ]
            ).to_csv(data / "price_source_registry.csv", index=False)

            run_data_readiness_with_supplemental(
                constituents_path=data / "tw50_constituents.csv",
                price_roots=(cache,),
                registry_path=data / "price_source_registry.csv",
                output_dir=output,
            )

            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertFalse(manifest["formal_model_changed"])
            self.assertFalse(manifest["trade_decision_changed"])
            self.assertEqual(manifest["00631l_base_first_date"], "2016-01-04")
            self.assertEqual(manifest["00631l_supplemental_first_date"], "2014-11-03")
            self.assertEqual(manifest["00631l_combined_first_date"], "2014-11-03")
            self.assertEqual(manifest["00631l_source_type"], "twse_stock_day_backfill")
            self.assertFalse(manifest["00631l_synthetic_used"])
            self.assertFalse(manifest["strategy_ready"])

            coverage = pd.read_csv(output / "price_coverage_with_supplemental.csv")
            row = coverage[coverage["ticker"] == "00631L.TW"].iloc[0]
            self.assertEqual(row["base_first_date"], "2016-01-04")
            self.assertEqual(row["supplemental_first_date"], "2014-11-03")
            self.assertEqual(row["combined_first_date"], "2014-11-03")
            self.assertEqual(row["supplemental_source_type"], "twse_stock_day_backfill")
            self.assertFalse(bool(row["strategy_ready"]))

            usage = pd.read_csv(output / "supplemental_source_usage.csv")
            self.assertEqual(usage["source_type"].iloc[0], "twse_stock_day_backfill")
            self.assertTrue(bool(usage["used_in_combined_coverage"].iloc[0]))

            blockers = pd.read_csv(output / "remaining_blockers.csv")
            self.assertIn("tw50_exact_pit_archive", set(blockers["blocker"]))
            self.assertIn("formal_target_signal_stream_2014_2021", set(blockers["blocker"]))


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
