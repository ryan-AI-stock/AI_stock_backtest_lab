import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from backtest_lab.formal_vs_pool1_only_validation import run_formal_vs_pool1_only_validation


class FormalVsPool1OnlyValidationTest(unittest.TestCase):
    def test_builds_apples_to_apples_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            replay = root / "replay"
            cache = root / "cache"
            output = root / "out"
            replay.mkdir()
            cache.mkdir()

            dates = pd.bdate_range("2024-01-02", periods=8)
            pd.DataFrame(
                [
                    {
                        "period": "2024_now",
                        "date": date.strftime("%Y-%m-%d"),
                        "pool1_vote": "00631L.TW" if index >= 2 else "2330.TW",
                        "pool2_vote": "2330.TW",
                    }
                    for index, date in enumerate(dates)
                ]
            ).to_csv(replay / "formal_three_pool_decision_panel.csv", index=False)
            _price_csv(cache / "00631L_TW.csv", dates, [100, 105, 110, 120, 125, 130, 135, 140])
            _price_csv(cache / "2330_TW.csv", dates, [100, 101, 102, 103, 104, 105, 106, 107])
            _price_csv(cache / "0050_TW.csv", dates, [100, 101, 102, 103, 104, 105, 106, 107])

            run_formal_vs_pool1_only_validation(
                formal_replay_dir=replay,
                price_cache_dir=cache,
                output_dir=output,
            )

            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["pool1_only_model_id"], "new_pool1_only_no_overlay")
            self.assertFalse(manifest["formal_model_changed"])
            self.assertFalse(manifest["trade_decision_changed"])
            self.assertTrue(manifest["same_execution_basis_compared_separately"])

            performance = pd.read_csv(output / "performance_comparison.csv")
            self.assertIn("current_formal_pool1_pool2_remove_cap", set(performance["variant_id"]))
            self.assertIn("new_pool1_only_no_overlay", set(performance["variant_id"]))
            effect = pd.read_csv(output / "pool2_effect_summary.csv")
            self.assertIn("pool2_effect_state", effect.columns)
            self.assertTrue((output / "formal_vs_pool1_only_summary_zh.md").exists())


def _price_csv(path: Path, dates: pd.DatetimeIndex, prices: list[float]) -> None:
    pd.DataFrame(
        {
            "date": [date.strftime("%Y-%m-%d") for date in dates],
            "open": prices,
            "close": prices,
            "adj_close": prices,
        }
    ).to_csv(path, index=False)


if __name__ == "__main__":
    unittest.main()
