import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from backtest_lab.final_decision_layer_review_after_absorption import run_final_decision_layer_review_after_absorption


class FinalDecisionLayerReviewAfterAbsorptionTest(unittest.TestCase):
    def test_review_retires_three_pool_layer_and_allows_execution_next(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "absorb"
            output = root / "out"
            source.mkdir()
            (source / "manifest.json").write_text(
                json.dumps(
                    {
                        "formal_model_target": "combined_cap40_confirmation1_base",
                        "formal_model_route": "pool1_primary_pool2_confirmation_cap40",
                        "formal_absorption_ready": True,
                        "three_pool_formal_route_abandoned": True,
                        "pool3_shadow_used_as_formal": False,
                    }
                ),
                encoding="utf-8",
            )
            pd.DataFrame(
                [
                    {
                        "dimension": "formal_route",
                        "before": "current_formal_three_pool_baseline",
                        "after": "combined_cap40_confirmation1_base",
                        "changed": True,
                        "evidence": "three_pool_formal_route_abandoned=true",
                    }
                ]
            ).to_csv(source / "formal_selector_before_after.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "blocker_id": "2024_hard_gate_0050x2_caveat",
                        "blocks_formal_absorption": False,
                    }
                ]
            ).to_csv(source / "blocker_matrix.csv", index=False)

            run_final_decision_layer_review_after_absorption(absorption_dir=source, output_dir=output)

            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertFalse(manifest["formal_model_changed"])
            self.assertFalse(manifest["trade_decision_changed"])
            self.assertFalse(manifest["active_in_trade_decision"])
            self.assertTrue(manifest["three_pool_decision_layer_retired_from_formal_route"])
            self.assertFalse(manifest["needs_legacy_three_pool_final_decision_layer"])
            self.assertFalse(manifest["needs_new_formal_decision_layer_before_execution_review"])
            self.assertTrue(manifest["execution_layer_can_continue"])
            self.assertTrue(manifest["pool3_backlog_removed_from_mainline"])

            review = pd.read_csv(output / "final_decision_layer_review.csv")
            self.assertIn("legacy_three_pool_final_decision_layer", set(review["review_item"]))
            self.assertIn("retire_from_formal_route", set(review["decision"]))

            states = pd.read_csv(output / "two_pool_decision_state_contract.csv")
            self.assertIn("pool2_confirmation_pending", set(states["state"]))
            self.assertIn("cap40_applied", set(states["state"]))


if __name__ == "__main__":
    unittest.main()
