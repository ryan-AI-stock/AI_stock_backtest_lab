import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from backtest_lab.integrate_00631l_backfill_source import run_00631l_backfill_cache_integration
from backtest_lab.supplemental_price_sources import load_supplemental_price_source


class BackfillCacheIntegrationTest(unittest.TestCase):
    def test_integrates_00631l_backfill_as_supplemental_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            phase4 = root / "phase4.csv"
            normalized_dir = root / "data" / "normalized_prices"
            registry = root / "data" / "price_source_registry.csv"
            output = root / "output"
            _phase4_csv(phase4)

            run_00631l_backfill_cache_integration(
                source_csv=phase4,
                normalized_price_dir=normalized_dir,
                registry_path=registry,
                output_dir=output,
            )

            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertFalse(manifest["formal_model_changed"])
            self.assertFalse(manifest["trade_decision_changed"])
            self.assertFalse(manifest["cache_overwritten"])
            self.assertTrue(manifest["price_source_ready"])
            self.assertFalse(manifest["strategy_ready"])
            self.assertFalse(manifest["synthetic_used"])
            self.assertEqual(manifest["source_type"], "twse_stock_day_backfill")

            integrated = normalized_dir / "00631L_twse_stock_day_201411_201512.csv"
            self.assertTrue(integrated.exists())
            self.assertTrue(registry.exists())

            loaded, provenance = load_supplemental_price_source("00631L.TW", registry_path=registry)
            self.assertIn(pd.Timestamp("2014-11-03"), loaded.index)
            self.assertEqual(provenance["source_type"], "twse_stock_day_backfill")
            self.assertEqual(str(provenance["strategy_ready"]).lower(), "false")
            self.assertEqual(str(provenance["synthetic_used"]).lower(), "false")

            coverage = pd.read_csv(output / "00631l_price_coverage_after_integration.csv")
            self.assertTrue(bool(coverage["price_source_ready"].iloc[0]))
            self.assertFalse(bool(coverage["strategy_ready"].iloc[0]))


def _phase4_csv(path: Path) -> None:
    pd.DataFrame(
        {
            "date": ["2014-11-03", "2015-12-31"],
            "ticker": ["00631L.TW", "00631L.TW"],
            "open": [20.18, 10.0],
            "high": [20.55, 10.2],
            "low": [20.18, 9.8],
            "close": [20.36, 10.1],
            "adj_close": [20.36, 10.1],
            "volume": [9493012, 1000],
            "source": ["TWSE_STOCK_DAY", "TWSE_STOCK_DAY"],
            "source_month": ["2014-11", "2015-12"],
            "source_type": ["official_real_price", "official_real_price"],
            "adjustment_policy": [
                "twse_raw_close_as_adj_close_pending_distribution_review",
                "twse_raw_close_as_adj_close_pending_distribution_review",
            ],
        }
    ).to_csv(path, index=False)


if __name__ == "__main__":
    unittest.main()
