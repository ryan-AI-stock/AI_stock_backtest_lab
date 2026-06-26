import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

import test_paths  # noqa: F401

from backtest_lab.execution_reversal_warning_label import FORBIDDEN_WORDS, run_execution_reversal_warning_label


class ExecutionReversalWarningLabelTest(unittest.TestCase):
    def test_report_only_reversal_warning_label(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            review = root / "review"
            output = root / "output"
            review.mkdir()
            dates = pd.date_range("2024-01-02", periods=5, freq="B")
            pd.DataFrame(
                [
                    _row(dates[0], '{"00631L.TW": 0.4}'),
                    _row(dates[1], '{"2454.TW": 1.0}'),
                    _row(dates[2], '{"00631L.TW": 0.4}'),
                    _row(dates[3], '{"00631L.TW": 0.4}'),
                    _row(dates[4], '{"00631L.TW": 0.4}'),
                ]
            ).to_csv(review / "formal_target_stream_adapter.csv", index=False)

            result = run_execution_reversal_warning_label(review_dir=review, output_dir=output, max_window_rows=3)

            manifest = json.loads((result / "manifest.json").read_text(encoding="utf-8"))
            self.assertFalse(manifest["formal_model_changed"])
            self.assertFalse(manifest["trade_decision_changed"])
            self.assertFalse(manifest["active_in_trade_decision"])
            self.assertFalse(manifest["formal_execution_layer_activated"])
            self.assertFalse(manifest["execution_reversal_warning_active_in_trade_decision"])
            self.assertFalse(manifest["formal_selector_readable"])
            self.assertTrue(manifest["label_only_does_not_modify_equity_or_trade_ledger"])
            self.assertFalse(manifest["uses_forward_return_as_rule"])
            self.assertEqual(manifest["forbidden_word_positive_hits"], [])

            panel = pd.read_csv(result / "execution_reversal_warning_label_panel.csv")
            self.assertFalse(panel["execution_reversal_warning_active_in_trade_decision"].astype(bool).any())
            self.assertFalse(panel["formal_selector_readable"].astype(bool).any())
            self.assertEqual(panel.loc[0, "execution_reversal_warning_boundary"], "report_only")

            history = pd.read_csv(result / "execution_reversal_event_history.csv")
            self.assertGreaterEqual(len(history), 1)
            self.assertTrue(history["target_to_target_reversal"].astype(bool).any())
            self.assertFalse(history["execution_reversal_warning_active_in_trade_decision"].astype(bool).any())

            wording = (result / "execution_reversal_warning_wording_zh.md").read_text(encoding="utf-8")
            for word in FORBIDDEN_WORDS:
                self.assertNotIn(word, wording)


def _row(date: pd.Timestamp, target_weights: str) -> dict:
    parsed = json.loads(target_weights)
    target = next(iter(parsed)) if parsed else ""
    return {
        "date": date.strftime("%Y-%m-%d"),
        "period": "2024_hard_gate",
        "formal_model_target": "combined_cap40_confirmation1_base",
        "source_panel_variant": "combined_cap40_confirmation1",
        "winner_ticker": target,
        "formal_target": target,
        "target_weights": target_weights,
        "position_ticker": target or "cash",
        "equity": 1_000_000.0,
        "drawdown": 0.0,
        "turnover": 0.0,
        "transaction_cost": 0.0,
        "action": "hold",
    }


if __name__ == "__main__":
    unittest.main()
