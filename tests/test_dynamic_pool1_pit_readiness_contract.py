import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

import test_paths  # noqa: F401

from backtest_lab.dynamic_pool1_pit_readiness_contract import (
    run_dynamic_pool1_pit_readiness_contract,
)


class DynamicPool1PitReadinessContractTest(unittest.TestCase):
    def test_builds_readiness_contract_without_accepting_current_snapshots_as_formal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache = root / "cache"
            radar = root / "radar"
            output = root / "out"
            cache.mkdir()
            radar.mkdir()

            pd.DataFrame(
                {
                    "date": ["2015-01-05", "2015-01-06"],
                    "open": [100, 101],
                    "high": [101, 102],
                    "low": [99, 100],
                    "close": [100, 101],
                    "adj_close": [100, 101],
                    "volume": [1000, 1200],
                }
            ).to_csv(cache / "2330_TW.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "ticker": "00631L.TW",
                        "source_id": "00631l_twse_stock_day",
                        "source_path": "data/normalized_prices/00631L.csv",
                        "source_type": "twse_stock_day_backfill",
                        "first_date": "2014-11-03",
                        "last_date": "2015-12-31",
                        "price_source_ready": True,
                        "strategy_ready": False,
                        "synthetic_used": False,
                        "provenance": "test",
                        "notes": "price-only",
                    }
                ]
            ).to_csv(root / "price_source_registry.csv", index=False)
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
            ).to_csv(root / "tw50_constituents.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "effective_date": "2026-06-01",
                        "ticker": "2330.TW",
                        "symbol": "2330",
                        "name": "台積電",
                        "theme_role": "AI半導體",
                        "review_status": "active",
                    }
                ]
            ).to_csv(root / "ai_theme_candidates.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "symbol": "2330",
                        "sector": "semiconductor",
                        "source_date": "2026-07-01",
                    }
                ]
            ).to_csv(radar / "sector_map.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "symbol": "2330",
                        "market_cap_twd": 1000000000000,
                        "source_date": "2026-07-01",
                    }
                ]
            ).to_csv(radar / "stock_metrics.refreshed.csv", index=False)

            result = run_dynamic_pool1_pit_readiness_contract(
                output_dir=output,
                price_cache_dir=cache,
                price_source_registry=root / "price_source_registry.csv",
                tw50_constituents_path=root / "tw50_constituents.csv",
                ai_theme_candidates_path=root / "ai_theme_candidates.csv",
                radar_data_dir=radar,
            )
            self.assertEqual(result, output)

            required = [
                "all_listed_liquid_universe_pit_daily.csv",
                "monthly_revenue_pit.csv",
                "quarterly_fundamentals_pit.csv",
                "market_cap_pit.csv",
                "sector_membership_pit.csv",
                "sector_breadth_pit_daily.csv",
                "candidate_data_readiness_by_date.csv",
                "future_data_violation_audit.csv",
                "source_manifest.json",
                "readiness.json",
            ]
            for name in required:
                self.assertTrue((output / name).exists(), name)

            readiness = json.loads((output / "readiness.json").read_text(encoding="utf-8"))
            self.assertFalse(readiness["formal_model_changed"])
            self.assertFalse(readiness["trade_decision_changed"])
            self.assertFalse(readiness["active_in_trade_decision"])
            self.assertFalse(readiness["dynamic_pool1_shadow_challenger_ready"])
            self.assertEqual(readiness["future_data_violation_count"], 0)
            self.assertEqual(readiness["table_status"]["monthly_revenue_pit"]["status"], "blocked")
            self.assertEqual(readiness["table_status"]["sector_membership_pit"]["status"], "blocked")

            sector = pd.read_csv(output / "sector_membership_pit.csv")
            self.assertTrue(sector["diagnostic_only"].astype(bool).all())
            self.assertFalse(sector["accepted_for_formal"].astype(bool).any())

            audit = pd.read_csv(output / "future_data_violation_audit.csv")
            self.assertFalse(audit["future_data_violation"].astype(bool).any())
            self.assertFalse(audit["current_snapshot_used_as_historical"].astype(bool).any())

            by_date = pd.read_csv(output / "candidate_data_readiness_by_date.csv")
            self.assertEqual(set(by_date["year_bucket"]), {"2015-2021", "2022-2023", "2024-latest"})
            self.assertIn("source_date", pd.read_csv(output / "all_listed_liquid_universe_pit_daily.csv").columns)
            self.assertIn("release_date", pd.read_csv(output / "monthly_revenue_pit.csv").columns)
            self.assertIn("effective_date", pd.read_csv(output / "quarterly_fundamentals_pit.csv").columns)


if __name__ == "__main__":
    unittest.main()
