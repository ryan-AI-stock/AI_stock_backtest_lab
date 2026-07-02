import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

import test_paths  # noqa: F401

from backtest_lab.no_target_risk_off_formal_activation import run_no_target_risk_off_formal_activation


class NoTargetRiskOffFormalActivationTest(unittest.TestCase):
    def test_builds_formal_activation_package(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            review = root / "review"
            output = root / "activation"
            review.mkdir()

            (review / "manifest.json").write_text(
                json.dumps(
                    {
                        "bug_cash_mapping_used_as_baseline": False,
                        "main_candidate": "no_target_cash_all",
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            pd.DataFrame(
                [
                    {
                        "variant_id": "baseline_hold_through",
                        "full_return_pct": 1849.72,
                        "full_mdd_pct": -41.89,
                    },
                    {
                        "variant_id": "no_target_cash_all",
                        "full_return_pct": 2377.67,
                        "full_mdd_pct": -30.78,
                        "full_return_delta_vs_baseline_pp": 527.95,
                        "full_mdd_delta_vs_baseline_pp": 11.11,
                    },
                ]
            ).to_csv(review / "formal_absorption_review_summary.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "blocker": "user_formal_activation_decision_missing",
                        "blocks_formal_absorption": True,
                    }
                ]
            ).to_csv(review / "formal_absorption_blocker_matrix.csv", index=False)

            run_no_target_risk_off_formal_activation(review_dir=review, output_dir=output)

            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertTrue(manifest["formal_model_changed"])
            self.assertTrue(manifest["trade_decision_changed"])
            self.assertTrue(manifest["active_in_trade_decision"])
            self.assertTrue(manifest["no_target_risk_off_active"])
            self.assertEqual(manifest["no_target_risk_off_policy"], "cash_all")
            self.assertFalse(manifest["bug_cash_mapping_used_as_baseline"])

            wording = pd.read_csv(output / "report_wording_examples.csv")
            no_target = wording[wording["scenario"].eq("no_formal_target")].iloc[0].to_dict()
            self.assertEqual(no_target["status_zh"], "風險控管空手")
            self.assertIn("模型未找到合格攻擊標的", no_target["reason_zh"])

            contract_text = (output / "formal_activation_contract_zh.md").read_text(encoding="utf-8")
            self.assertIn("有正式目標時", contract_text)
            self.assertIn("no-target risk-off 正式規則", contract_text)
            self.assertNotIn("舊隱含空手映射合理化", no_target["reason_zh"])


if __name__ == "__main__":
    unittest.main()
