import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

import test_paths  # noqa: F401

from backtest_lab.no_target_risk_off_absorption_review import (
    run_no_target_risk_off_absorption_review,
)


class NoTargetRiskOffAbsorptionReviewTest(unittest.TestCase):
    def test_builds_review_without_activating_formal_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            output = root / "output"
            source.mkdir()

            (source / "manifest.json").write_text(
                json.dumps(
                    {
                        "formal_model_target": "pool1_pool2_confirmation1_base",
                        "formal_model_route": "pool1_primary_pool2_confirmation",
                        "bug_cash_mapping_used_as_baseline": False,
                    }
                ),
                encoding="utf-8",
            )
            pd.DataFrame(
                [
                    _perf("baseline_hold_through", "full_available", 1849.72, -41.89),
                    _perf("no_target_cash_all", "full_available", 2377.67, -30.78),
                    _perf("no_target_cash_max_3", "full_available", 1716.57, -47.35),
                    _perf("no_target_reduce_exposure_50", "full_available", 2191.74, -31.61),
                    _perf("baseline_hold_through", "2024_hard_gate", 41.05, -29.62),
                    _perf("no_target_cash_all", "2024_hard_gate", 42.17, -29.98),
                    _perf("baseline_hold_through", "2026_ytd", 205.13, -18.68),
                    _perf("no_target_cash_all", "2026_ytd", 214.63, -18.68),
                ]
            ).to_csv(source / "performance_by_variant.csv", index=False)
            pd.DataFrame(
                [
                    {"variant_id": "baseline_hold_through", "no_formal_target_policy": "hold_previous", "is_formal_baseline": True},
                    {"variant_id": "no_target_cash_all", "no_formal_target_policy": "exit_to_cash", "is_formal_baseline": False},
                    {"variant_id": "no_target_cash_max_3", "no_formal_target_policy": "cash_max_3", "is_formal_baseline": False},
                    {"variant_id": "no_target_reduce_exposure_50", "no_formal_target_policy": "reduce_exposure_50", "is_formal_baseline": False},
                ]
            ).to_csv(source / "variant_contract.csv", index=False)
            pd.DataFrame(
                [
                    {"contract_stage": "before_current_formal_baseline", "variant_id": "baseline_hold_through", "no_formal_target_policy": "hold_previous"},
                    {"contract_stage": "after_explicit_formal_challenger_candidate", "variant_id": "no_target_cash_all", "no_formal_target_policy": "exit_to_cash"},
                ]
            ).to_csv(source / "formal_challenger_before_after_contract.csv", index=False)
            pd.DataFrame([{"variant_id": "no_target_cash_all", "no_target_event_days": 10}]).to_csv(source / "no_target_event_attribution.csv", index=False)
            pd.DataFrame([{"variant_id": "no_target_cash_all", "trade_rows": 50, "total_transaction_cost": 1000}]).to_csv(source / "trade_cost_summary.csv", index=False)

            run_no_target_risk_off_absorption_review(source_dir=source, output_dir=output)

            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertTrue(manifest["formal_absorption_review_ready"])
            self.assertTrue(manifest["formal_absorption_candidate_ready"])
            self.assertFalse(manifest["formal_absorption_activated"])
            self.assertTrue(manifest["requires_user_formal_decision_before_activation"])
            self.assertFalse(manifest["formal_model_changed"])
            self.assertFalse(manifest["trade_decision_changed"])

            review = pd.read_csv(output / "formal_absorption_review_summary.csv")
            main = review[review["variant_id"].eq("no_target_cash_all")].iloc[0]
            self.assertEqual(main["candidate_review_decision"], "ready_for_formal_absorption_decision")
            self.assertGreater(float(main["full_return_delta_vs_baseline_pp"]), 0)

            blockers = pd.read_csv(output / "formal_absorption_blocker_matrix.csv")
            self.assertIn("user_formal_activation_decision_missing", set(blockers["blocker"]))
            technical_blockers = blockers[
                blockers["blocks_formal_absorption"].astype(bool)
                & ~blockers["blocker"].eq("user_formal_activation_decision_missing")
            ]
            self.assertTrue(technical_blockers.empty)


def _perf(variant: str, period: str, ret: float, mdd: float) -> dict:
    return {
        "variant_id": variant,
        "execution_basis": "next_day",
        "period_label": period,
        "start_date": "2022-01-03",
        "end_date": "2026-06-12",
        "return_pct": ret,
        "max_drawdown_pct": mdd,
        "trade_rows": 10,
        "total_transaction_cost": 100,
    }


if __name__ == "__main__":
    unittest.main()
