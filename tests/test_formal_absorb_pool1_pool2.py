import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from backtest_lab.formal_absorb_pool1_pool2 import run_formal_absorb_pool1_pool2
from backtest_lab.formal_model_contract import FORMAL_MODEL_TARGET, get_formal_model_contract


class FormalAbsorbPool1Pool2Test(unittest.TestCase):
    def test_absorption_package_marks_formal_switch_and_preserves_boundaries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate = root / "candidate"
            three_pool = root / "three_pool"
            label = root / "label"
            output = root / "out"
            candidate.mkdir()
            three_pool.mkdir()
            label.mkdir()

            period_rows = [
                {
                    "variant": "combined_cap40_confirmation1",
                    "period_label": "full",
                    "status": "completed",
                    "start_date": "2022-01-03",
                    "end_date": "2026-06-12",
                    "start_equity": 1_000_000,
                    "final_equity": 31_434_534.58,
                    "return_pct": 3043.4535,
                    "max_drawdown_pct": -19.6723,
                    "trade_days": 53,
                    "total_transaction_cost": 1_111_648,
                },
                {
                    "variant": "combined_cap40_confirmation1",
                    "period_label": "2024_hard_gate",
                    "status": "completed",
                    "start_date": "2024-01-02",
                    "end_date": "2024-12-31",
                    "start_equity": 3_713_317.28,
                    "final_equity": 5_523_080.80,
                    "return_pct": 48.7371,
                    "max_drawdown_pct": -17.2396,
                    "trade_days": 16,
                    "total_transaction_cost": 209_533,
                },
            ]
            pd.DataFrame(period_rows).to_csv(candidate / "period_performance_by_variant.csv", index=False)
            dates = pd.date_range("2026-06-01", periods=35, freq="B")
            daily = pd.DataFrame(
                {
                    "variant": ["combined_cap40_confirmation1"] * len(dates),
                    "date": dates.strftime("%Y-%m-%d"),
                    "period": ["2024_now"] * len(dates),
                    "target_weights": ['{"00631L.TW": 0.4}'] * len(dates),
                    "position_ticker": ["00631L.TW"] * len(dates),
                    "cash": [600_000] * len(dates),
                    "equity": [1_000_000 + i * 1000 for i in range(len(dates))],
                    "drawdown": [0.0] * len(dates),
                    "turnover": [0.0] * len(dates),
                    "transaction_cost": [0] * len(dates),
                    "action": ["hold"] * len(dates),
                }
            )
            daily.to_csv(candidate / "daily_equity_by_variant.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "variant": "combined_cap40_confirmation1",
                        "date": dates[0].strftime("%Y-%m-%d"),
                        "ticker": "00631L.TW",
                        "action": "buy",
                        "gross_amount": 400_000,
                        "costs": 570,
                    }
                ]
            ).to_csv(candidate / "trade_ledger_by_variant.csv", index=False)
            pd.DataFrame(
                {
                    "variant": ["combined_cap40_confirmation1"] * len(dates),
                    "date": dates.strftime("%Y-%m-%d"),
                    "period": ["2024_now"] * len(dates),
                    "pool1_vote": ["00631L.TW"] * len(dates),
                    "pool2_vote": ["00631L.TW"] * len(dates),
                    "pool2_disagreement": [False] * len(dates),
                    "event_reason": ["pool1_primary"] * len(dates),
                    "target_weights": ['{"00631L.TW": 0.4}'] * len(dates),
                }
            ).to_csv(candidate / "pool2_disagreement_variant_events.csv", index=False)
            pd.DataFrame([{"variant": "combined_cap40_confirmation1", "pool2_policy": "confirmation", "cap_00631l": 0.4, "confirmation_days": 1}]).to_csv(
                candidate / "variant_parameter_matrix.csv",
                index=False,
            )
            pd.DataFrame(
                [
                    {
                        "status": "completed",
                        "start_date": "2022-01-03",
                        "end_date": "2026-06-12",
                        "total_return_pct": 193.1021,
                        "max_drawdown_pct": -62.024,
                    }
                ]
            ).to_csv(three_pool / "formal_three_pool_summary.csv", index=False)
            (label / "manifest.json").write_text(
                json.dumps(
                    {
                        "opportunity_cost_label_active_in_trade_decision": False,
                        "market_exposure_override_absorbed": False,
                        "forbidden_word_positive_hits": [],
                    }
                ),
                encoding="utf-8",
            )

            run_formal_absorb_pool1_pool2(
                candidate_dir=candidate,
                three_pool_dir=three_pool,
                opportunity_label_dir=label,
                output_dir=output,
            )

            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["formal_model_target"], FORMAL_MODEL_TARGET)
            self.assertTrue(manifest["formal_model_changed"])
            self.assertTrue(manifest["trade_decision_changed"])
            self.assertTrue(manifest["formal_absorption_ready"])
            self.assertTrue(manifest["three_pool_formal_route_abandoned"])
            self.assertFalse(manifest["pool3_shadow_used_as_formal"])
            self.assertFalse(manifest["0050x2_opportunity_label_active_in_trade_decision"])
            self.assertFalse(manifest["uses_forward_return_as_rule"])

            before_after = pd.read_csv(output / "formal_selector_before_after.csv")
            self.assertIn("selector_logic", set(before_after["dimension"]))

            blockers = pd.read_csv(output / "blocker_matrix.csv")
            self.assertIn("2024_hard_gate_0050x2_caveat", set(blockers["blocker_id"]))
            self.assertFalse(blockers["blocks_formal_absorption"].map(lambda value: str(value).lower() == "true").any())

            sample = pd.read_csv(output / "formal_trade_decision_sample.csv")
            self.assertFalse(sample["pool3_shadow_used_as_formal"].map(lambda value: str(value).lower() == "true").any())
            self.assertFalse(sample["uses_forward_return_as_rule"].map(lambda value: str(value).lower() == "true").any())

    def test_contract_names_pool3_as_diagnostic_only(self) -> None:
        contract = get_formal_model_contract()
        self.assertEqual(contract["formal_model_target"], FORMAL_MODEL_TARGET)
        self.assertTrue(contract["three_pool_formal_route_abandoned"])
        self.assertFalse(contract["pool3_shadow_used_as_formal"])
        self.assertFalse(contract["market_exposure_override_absorbed"])


if __name__ == "__main__":
    unittest.main()
