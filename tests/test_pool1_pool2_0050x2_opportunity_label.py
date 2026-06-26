import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from backtest_lab.pool1_pool2_0050x2_opportunity_label import FORBIDDEN_WORDS, run_pool1_pool2_0050x2_opportunity_label


class Pool1Pool20050x2OpportunityLabelTest(unittest.TestCase):
    def test_report_only_label_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            output = root / "out"
            source.mkdir()
            pd.DataFrame(
                {
                    "date": ["2024-01-02", "2024-01-03"],
                    "period": ["2024", "2024"],
                    "pool1_vote": ["00631L.TW", "2330.TW"],
                    "pool2_vote": ["00631L.TW", "0050.TW"],
                    "target_weights": ['{"00631L.TW": 0.4}', '{"2330.TW": 1.0}'],
                }
            ).to_csv(source / "0050x2_opportunity_cost_label_panel.csv", index=False)
            pd.DataFrame({"variant": ["base"], "date": ["2024-01-02"], "equity": [1_000_000]}).to_csv(source / "daily_equity_by_variant.csv", index=False)
            pd.DataFrame({"variant": ["base"], "date": ["2024-01-02"], "ticker": ["00631L.TW"]}).to_csv(source / "trade_ledger_by_variant.csv", index=False)
            (source / "manifest.json").write_text(json.dumps({"formal_model_changed": False}), encoding="utf-8")

            run_pool1_pool2_0050x2_opportunity_label(source_dir=source, output_dir=output)

            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertFalse(manifest["formal_model_changed"])
            self.assertFalse(manifest["trade_decision_changed"])
            self.assertFalse(manifest["formal_absorption_ready"])
            self.assertFalse(manifest["opportunity_cost_label_active_in_trade_decision"])
            self.assertFalse(manifest["market_exposure_override_absorbed"])
            self.assertEqual(manifest["forbidden_word_positive_hits"], [])

            panel = pd.read_csv(output / "0050x2_opportunity_label_panel.csv")
            self.assertFalse(panel["benchmark_opportunity_cost_active_in_trade_decision"].any())
            self.assertFalse(panel["formal_selector_readable"].any())
            self.assertEqual(set(panel["benchmark_opportunity_cost_boundary"]), {"report_only"})

            wording = (output / "report_wording_boundary_zh.md").read_text(encoding="utf-8")
            for word in FORBIDDEN_WORDS:
                self.assertNotIn(word, wording)


if __name__ == "__main__":
    unittest.main()
