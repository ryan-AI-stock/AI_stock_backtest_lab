import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

import test_paths  # noqa: F401

from backtest_lab.execution_cooldown3_report_label import (
    COOLDOWN3_CANDIDATE,
    FORBIDDEN_WORDS,
    run_execution_cooldown3_report_label,
)


class ExecutionCooldown3ReportLabelTest(unittest.TestCase):
    def test_report_only_label_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            output = root / "out"
            source.mkdir()
            (source / "manifest.json").write_text(
                json.dumps({"main_candidate": COOLDOWN3_CANDIDATE, "formal_model_changed": False}),
                encoding="utf-8",
            )
            pd.DataFrame(
                [
                    {
                        "variant_id": COOLDOWN3_CANDIDATE,
                        "period_label": "full",
                        "return_pct": 2222.2624,
                        "max_drawdown_pct": -27.9816,
                    }
                ]
            ).to_csv(source / "period_performance_by_candidate.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "variant_id": COOLDOWN3_CANDIDATE,
                        "candidate_return_pct": 41.2115,
                        "candidate_mdd_pct": -22.3504,
                        "excess_vs_0050x2_pct": -57.2886,
                    }
                ]
            ).to_csv(source / "hard_gate_2024_attribution.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "candidate": COOLDOWN3_CANDIDATE,
                        "readiness_state": "promising_diagnostic_not_formal_ready",
                        "blockers": "2024_hard_gate_underperforms_0050x2",
                    }
                ]
            ).to_csv(source / "cooldown_robustness_readiness_report.csv", index=False)

            run_execution_cooldown3_report_label(source_dir=source, output_dir=output)

            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertFalse(manifest["formal_model_changed"])
            self.assertFalse(manifest["trade_decision_changed"])
            self.assertFalse(manifest["active_in_trade_decision"])
            self.assertFalse(manifest["formal_execution_layer_activated"])
            self.assertFalse(manifest["execution_label_active_in_trade_decision"])
            self.assertFalse(manifest["formal_selector_readable"])
            self.assertTrue(manifest["label_only_does_not_modify_equity_or_trade_ledger"])
            self.assertEqual(manifest["forbidden_word_positive_hits"], [])

            panel = pd.read_csv(output / "execution_cooldown3_report_label_panel.csv")
            self.assertEqual(panel.loc[0, "execution_diagnostic_candidate"], COOLDOWN3_CANDIDATE)
            self.assertEqual(panel.loc[0, "execution_diagnostic_boundary"], "report_only")
            self.assertFalse(panel["execution_diagnostic_active_in_trade_decision"].map(lambda value: str(value).lower() == "true").any())
            self.assertFalse(panel["formal_selector_readable"].map(lambda value: str(value).lower() == "true").any())

            wording = (output / "execution_cooldown3_wording_boundary_zh.md").read_text(encoding="utf-8")
            for word in FORBIDDEN_WORDS:
                self.assertNotIn(word, wording)


if __name__ == "__main__":
    unittest.main()
