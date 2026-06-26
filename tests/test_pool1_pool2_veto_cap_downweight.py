import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from backtest_lab.pool1_pool2_veto_cap_downweight import run_pool1_pool2_veto_cap_downweight


class Pool1Pool2VetoCapDownweightTest(unittest.TestCase):
    def test_outputs_recompute_cap_and_keep_warning_only_performance_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            formal_dir = root / "formal"
            price_dir = root / "prices"
            output_dir = root / "out"
            formal_dir.mkdir()
            price_dir.mkdir()

            dates = pd.date_range("2024-01-02", periods=8, freq="B")
            decision = pd.DataFrame(
                {
                    "date": dates.strftime("%Y-%m-%d"),
                    "period": ["2024_now"] * len(dates),
                    "pool1_vote": [
                        "00631L.TW",
                        "00631L.TW",
                        "2330.TW",
                        "2330.TW",
                        "00631L.TW",
                        "00631L.TW",
                        "2330.TW",
                        "2330.TW",
                    ],
                    "pool2_vote": [
                        "0050.TW",
                        "0050.TW",
                        "2330.TW",
                        "0050.TW",
                        "00631L.TW",
                        "0050.TW",
                        "0050.TW",
                        "2330.TW",
                    ],
                }
            )
            decision.to_csv(formal_dir / "formal_three_pool_decision_panel.csv", index=False)

            for ticker, prices in {
                "00631L.TW": [10, 11, 12, 11, 12, 13, 12, 14],
                "0050.TW": [20, 20, 21, 21, 22, 22, 23, 23],
                "2330.TW": [100, 102, 101, 103, 104, 105, 106, 107],
            }.items():
                pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "close": prices}).to_csv(
                    price_dir / f"{ticker}.csv",
                    index=False,
                )

            run_pool1_pool2_veto_cap_downweight(
                formal_replay_dir=formal_dir,
                price_cache_dir=price_dir,
                output_dir=output_dir,
                initial_cash=1_000_000,
            )

            manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertFalse(manifest["formal_model_changed"])
            self.assertFalse(manifest["trade_decision_changed"])
            self.assertFalse(manifest["formal_absorption_ready"])
            self.assertTrue(manifest["cap_performance_recomputed"])
            self.assertEqual(manifest["latest_complete_common_date"], "2024-01-11")

            variant_matrix = pd.read_csv(output_dir / "variant_parameter_matrix.csv")
            self.assertIn("pool1_pool2_veto_00631L_cap_40", set(variant_matrix["variant"]))
            self.assertIn("pool1_pool2_disagree_warning_only", set(variant_matrix["variant"]))
            self.assertIn("combined_cap40_downweight", set(variant_matrix["variant"]))

            cap_true_equity = pd.read_csv(output_dir / "00631L_cap_true_equity.csv")
            self.assertIn("pool1_pool2_veto_00631L_cap_40", set(cap_true_equity["variant"]))
            self.assertTrue(cap_true_equity["performance_recomputed"].all())

            daily = pd.read_csv(output_dir / "daily_equity_by_variant.csv")
            no_overlay = daily[daily["variant"].eq("pool1_primary_no_overlay")]["equity"].reset_index(drop=True)
            warning = daily[daily["variant"].eq("pool1_pool2_disagree_warning_only")]["equity"].reset_index(drop=True)
            pd.testing.assert_series_equal(no_overlay, warning, check_names=False)

            forward = pd.read_csv(output_dir / "vetoed_event_forward_outcome_by_variant.csv")
            self.assertFalse(forward["uses_forward_return_as_rule"].any())

            exposure = pd.read_csv(output_dir / "00631L_exposure_by_variant.csv")
            self.assertIn("00631L_position_day_share", exposure.columns)


if __name__ == "__main__":
    unittest.main()
