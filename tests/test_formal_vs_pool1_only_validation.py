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

    def test_no_formal_target_panel_holds_previous_formal_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            replay = root / "replay"
            cache = root / "cache"
            output = root / "out"
            replay.mkdir()
            cache.mkdir()

            dates = pd.to_datetime(["2026-06-05", "2026-06-08", "2026-06-09", "2026-06-10"])
            pd.DataFrame(
                [
                    {"period": "2026", "date": "2026-06-05", "pool1_vote": "2454.TW", "pool2_vote": "2454.TW"},
                    {"period": "2026", "date": "2026-06-08", "pool1_vote": "", "pool2_vote": "2454.TW"},
                    {"period": "2026", "date": "2026-06-09", "pool1_vote": "00631L.TW", "pool2_vote": "2327.TW"},
                    {"period": "2026", "date": "2026-06-10", "pool1_vote": "00631L.TW", "pool2_vote": "2327.TW"},
                ]
            ).to_csv(replay / "formal_three_pool_decision_panel.csv", index=False)
            _price_csv(cache / "2454_TW.csv", dates, [100, 95, 90, 88])
            _price_csv(cache / "00631L_TW.csv", dates, [50, 52, 54, 56])
            _price_csv(cache / "2327_TW.csv", dates, [400, 405, 410, 415])
            _price_csv(cache / "0050_TW.csv", dates, [100, 101, 102, 103])

            run_formal_vs_pool1_only_validation(
                formal_replay_dir=replay,
                price_cache_dir=cache,
                output_dir=output,
            )

            daily = pd.read_csv(output / "daily_equity_by_variant.csv")
            formal_next_day = daily[
                daily["variant_id"].eq("current_formal_pool1_pool2_remove_cap")
                & daily["execution_basis"].eq("next_day")
            ]
            no_target_row = formal_next_day[formal_next_day["date"].astype(str).eq("2026-06-09")].iloc[0]
            self.assertEqual(no_target_row["top_holding"], "2454.TW")
            self.assertEqual(json.loads(no_target_row["target_weights"]), {"2454.TW": 1.0})

            trades = pd.read_csv(output / "trade_ledger_by_variant.csv")
            false_sell = trades[
                trades["variant_id"].eq("current_formal_pool1_pool2_remove_cap")
                & trades["execution_basis"].eq("next_day")
                & trades["signal_date"].astype(str).eq("2026-06-08")
                & trades["action"].astype(str).eq("sell")
            ]
            self.assertTrue(false_sell.empty)


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
