import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from backtest_lab.remove_cap_next_day_validation import run_remove_cap_next_day_validation


class RemoveCapNextDayValidationTest(unittest.TestCase):
    def test_builds_apples_to_apples_validation_package(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            replay = root / "replay"
            cache = root / "cache"
            output = root / "out"
            replay.mkdir()
            cache.mkdir()

            dates = pd.bdate_range("2022-01-03", periods=8)
            pd.DataFrame(
                [
                    {
                        "period": "2022",
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

            run_remove_cap_next_day_validation(
                formal_replay_dir=replay,
                price_cache_dir=cache,
                output_dir=output,
            )

            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertTrue(manifest["user_requested_remove_cap_direction"])
            self.assertFalse(manifest["fully_validated_improvement"])
            self.assertTrue(manifest["same_day_result_not_used_as_next_day_proof"])
            self.assertTrue(manifest["production_grade_next_day_ledger"])
            self.assertFalse(manifest["simplified_experiments_ledger_used_for_formal_performance"])
            self.assertFalse(manifest["active_in_trade_decision"])
            self.assertFalse(manifest["uses_forward_return_as_rule"])
            self.assertEqual(manifest["formal_target_stream_start"], "2022-01-03")

            summary = pd.read_csv(output / "cap40_vs_remove_cap_same_day_next_day_summary.csv")
            self.assertIn("remove_cap_confirmation1_next_day", set(summary["variant_id"]))
            self.assertIn("cap40_confirmation1_next_day", set(summary["variant_id"]))
            ledger = pd.read_csv(output / "remove_cap_next_day_equity_ledger.csv")
            self.assertFalse(ledger["active_in_trade_decision"].map(lambda value: str(value).lower() == "true").any())
            self.assertTrue((output / "report_wording_boundary_zh.md").exists())


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
