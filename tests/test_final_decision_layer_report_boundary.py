from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

import test_paths  # noqa: F401

from backtest_lab.final_decision_layer_report_boundary import run_final_decision_layer_report_boundary


class FinalDecisionLayerReportBoundaryTest(unittest.TestCase):
    def test_report_boundary_keeps_divergence_and_exposure_report_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state_panel = root / "state.csv"
            forward_by_state = root / "forward.csv"
            output = root / "out"

            pd.DataFrame(
                [
                    _state_row("2024-01-02", "strong_consensus", "2330.TW", "stock_attack", False),
                    _state_row("2024-01-03", "actionable_divergence", "2454.TW", "stock_attack", True),
                    _state_row("2024-01-04", "defensive_market_exposure", "00631L.TW", "market_exposure_tool", True),
                    _state_row("2024-01-05", "data_insufficient", "", "none", True),
                ]
            ).to_csv(state_panel, index=False)
            pd.DataFrame(
                [
                    _forward_row("strong_consensus", 0.8257, 0.0738, 0.1779, 0.4479),
                    _forward_row("actionable_divergence", 1.0, -0.0865, -0.1613, -0.2778),
                    _forward_row("defensive_market_exposure", 0.2618, -0.0130, 0.0424, 0.10),
                ]
            ).to_csv(forward_by_state, index=False)

            result = run_final_decision_layer_report_boundary(
                state_panel_path=state_panel,
                forward_by_state_path=forward_by_state,
                output_dir=output,
            )

            manifest = json.loads((result / "manifest.json").read_text(encoding="utf-8"))
            self.assertFalse(manifest["formal_model_changed"])
            self.assertFalse(manifest["trade_decision_changed"])
            self.assertFalse(manifest["active_in_trade_decision"])
            self.assertFalse(manifest["report_boundary_active_in_trade_decision"])
            self.assertEqual(manifest["final_decision_layer_boundary"], "report_only_diagnostic")

            panel = pd.read_csv(result / "final_decision_report_boundary_panel.csv")
            self.assertFalse(panel["final_decision_label_active_in_trade_decision"].map(bool).any())
            self.assertFalse(panel["active_in_trade_decision"].map(bool).any())

            strong = panel[panel["final_decision_state"].eq("strong_consensus")].iloc[0]
            self.assertEqual(strong["final_decision_report_confidence"], "strong_consensus_supported")
            self.assertIn("不代表保證績效", strong["strong_consensus_confidence_note"])

            divergence = panel[panel["final_decision_state"].eq("actionable_divergence")].iloc[0]
            self.assertEqual(divergence["final_decision_report_confidence"], "divergence_fail_closed")
            self.assertEqual(divergence["final_decision_user_reading_state"], "divergence_watch_only")
            self.assertTrue(bool(divergence["not_eligible_for_formal_selector"]))
            joined_divergence_text = " ".join(str(divergence[col]) for col in panel.columns)
            for forbidden in ["可買進", "正式候選", "買進建議", "賣出建議", "明牌"]:
                self.assertNotIn(forbidden, joined_divergence_text)

            exposure = panel[panel["final_decision_state"].eq("defensive_market_exposure")].iloc[0]
            self.assertEqual(exposure["final_decision_report_confidence"], "market_exposure_explanation")
            self.assertIn("不計入股票 exact consensus", exposure["market_exposure_explanation_note"])
            self.assertFalse(bool(exposure["etf_counted_as_stock_vote"]))

            summary = pd.read_csv(result / "report_boundary_state_summary.csv")
            self.assertEqual(int(summary["active_in_trade_decision_count"].sum()), 0)


def _state_row(date: str, state: str, target: str, target_type: str, not_eligible: bool) -> dict[str, object]:
    return {
        "period": "2024_now",
        "signal_date": date,
        "final_decision_state": state,
        "final_target_type": target_type,
        "final_target_ticker": target,
        "final_target_source": "exact_consensus" if target else "none",
        "not_eligible_for_formal_selector": not_eligible,
    }


def _forward_row(state: str, coverage: float, d20: float, d60: float, d120: float) -> dict[str, object]:
    return {
        "final_decision_state": state,
        "complete_coverage_rate": coverage,
        "forward_20d_mean": d20,
        "forward_60d_mean": d60,
        "forward_120d_mean": d120,
    }


if __name__ == "__main__":
    unittest.main()
