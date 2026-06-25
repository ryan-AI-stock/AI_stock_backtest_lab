from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

import test_paths  # noqa: F401

from backtest_lab.execution_layer_diagnostic import run_execution_layer_diagnostic


class ExecutionLayerDiagnosticTest(unittest.TestCase):
    def test_runner_builds_report_only_execution_gap_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            formal_daily = root / "formal_daily.csv"
            output = root / "out"
            pd.DataFrame(
                [
                    _daily_row("2024-01-02", "00631L.TW", "00631L.TW", "buy", 100_000, 100),
                    _daily_row("2024-01-03", "2454.TW", "2454.TW", "switch", 200_000, 220),
                    _daily_row("2024-01-04", "00631L.TW", "00631L.TW", "switch", 210_000, 230),
                    _daily_row("2024-01-05", "00631L.TW", "00631L.TW", "hold", 0, 0),
                ]
            ).to_csv(formal_daily, index=False)

            result = run_execution_layer_diagnostic(formal_daily_path=formal_daily, output_dir=output)

            manifest = json.loads((result / "manifest.json").read_text(encoding="utf-8"))
            self.assertFalse(manifest["formal_model_changed"])
            self.assertFalse(manifest["trade_decision_changed"])
            self.assertFalse(manifest["active_in_trade_decision"])
            self.assertFalse(manifest["execution_diagnostic_active_in_trade_decision"])
            self.assertEqual(manifest["boundary"], "report_only_diagnostic")

            changes = pd.read_csv(result / "formal_target_change_panel.csv")
            self.assertIn("reversal_within_3_trading_rows", changes.columns)
            self.assertTrue(changes["reversal_within_3_trading_rows"].astype(bool).any())

            transitions = pd.read_csv(result / "holding_transition_diagnostic.csv")
            self.assertIn("full_rotation_flag", transitions.columns)
            self.assertTrue(transitions["full_rotation_flag"].astype(bool).any())
            self.assertFalse(transitions["execution_diagnostic_active_in_trade_decision"].astype(bool).any())

            summary = pd.read_csv(result / "execution_gap_summary.csv").iloc[0]
            self.assertEqual(int(summary["target_change_count"]), 3)
            self.assertGreater(float(summary["rapid_flip_rate"]), 0)
            self.assertFalse(bool(summary["active_in_trade_decision"]))

            preplan = pd.read_csv(result / "execution_variant_preplan.csv")
            self.assertIn("minimum_hold_N", preplan["variant_id"].tolist())
            self.assertFalse(preplan["active_in_trade_decision"].astype(bool).any())
            self.assertTrue((result / "final_summary_zh.md").exists())


def _daily_row(date: str, winner: str, position: str, action: str, turnover: float, cost: float) -> dict:
    return {
        "date": date,
        "period": "fixture",
        "pool1_vote": winner,
        "pool2_vote": winner,
        "pool3_vote": "",
        "consensus_state": "consensus",
        "winner_ticker": winner,
        "position_ticker": position,
        "cash": 0,
        "equity": 1_000_000,
        "drawdown": 0,
        "turnover": turnover,
        "transaction_cost": cost,
        "action": action,
        "data_status": "formal_daily_replay",
    }


if __name__ == "__main__":
    unittest.main()
