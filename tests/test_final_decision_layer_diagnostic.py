from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

import test_paths  # noqa: F401

from backtest_lab.final_decision_layer_diagnostic import run_final_decision_layer_diagnostic


class FinalDecisionLayerDiagnosticTest(unittest.TestCase):
    def test_exact_consensus_passthrough_and_pool3_shadow_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            event = root / "event.csv"
            selector = root / "selector.csv"
            pd.DataFrame(
                [
                    _event_row(
                        "2024-01-02",
                        p1="2330.TW",
                        p2="2330.TW",
                        p1_dir="stock_attack",
                        p2_dir="stock_attack",
                        final_target="2330.TW",
                    )
                ]
            ).to_csv(event, index=False)
            pd.DataFrame(
                [
                    {
                        "signal_date": "2024-01-02",
                        "pool3_selector_diagnostic_state": "veto_explanation",
                        "pool3_selector_diagnostic_boundary": "report_only",
                    }
                ]
            ).to_csv(selector, index=False)

            output = run_final_decision_layer_diagnostic(
                event_panel_path=event,
                pool3_selector_panel_path=selector,
                output_dir=root / "out",
            )

            panel = pd.read_csv(output / "final_decision_layer_diagnostic_panel.csv")
            row = panel.iloc[0]
            self.assertEqual(row["final_decision_layer_state"], "consensus_passthrough")
            self.assertEqual(row["final_decision_diagnostic_state"], "consensus_passthrough")
            self.assertEqual(row["exact_ticker_consensus_state"], "exact_consensus")
            self.assertEqual(row["direction_consensus_state"], "direction_consensus")
            self.assertEqual(row["actionable_decision_consensus"], "actionable_target_formed")
            self.assertEqual(row["exact_ticker_consensus_rate"], 1.0)
            self.assertTrue(bool(row["pool3_shadow_not_formal_flag"]))
            self.assertFalse(bool(row["pool3_shadow_used_in_trade_decision"]))
            self.assertEqual(row["pool3_shadow_boundary"], "report_only")
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertFalse(manifest["formal_model_changed"])
            self.assertFalse(manifest["trade_decision_changed"])
            self.assertFalse(manifest["active_in_trade_decision"])
            self.assertFalse(manifest["pool3_shadow_used_in_trade_decision"])
            self.assertEqual(manifest["final_decision_layer_boundary"], "report_only_diagnostic")

    def test_direction_consensus_without_exact_is_fake_health_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            event = root / "event.csv"
            pd.DataFrame(
                [
                    _event_row(
                        "2024-01-02",
                        p1="2330.TW",
                        p2="2454.TW",
                        p1_dir="stock_attack",
                        p2_dir="stock_attack",
                        final_target="",
                    )
                ]
            ).to_csv(event, index=False)

            output = run_final_decision_layer_diagnostic(
                event_panel_path=event,
                pool3_selector_panel_path=None,
                output_dir=root / "out",
            )

            panel = pd.read_csv(output / "final_decision_layer_diagnostic_panel.csv")
            row = panel.iloc[0]
            self.assertEqual(row["final_decision_layer_state"], "protocol_overuse_warning")
            self.assertEqual(row["direction_consensus_rate"], 1.0)
            self.assertEqual(row["exact_ticker_consensus_rate"], 0.0)
            self.assertEqual(row["exact_ticker_consensus_state"], "no_exact_consensus")
            self.assertEqual(row["direction_consensus_state"], "direction_consensus")
            self.assertEqual(row["actionable_decision_consensus"], "not_actionable")
            self.assertTrue(bool(row["fake_direction_consensus_flag"]))
            self.assertTrue(bool(row["decision_protocol_overuse_flag"]))

    def test_pool3_shadow_does_not_override_pool1_pool2_formal_conflict(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            event = root / "event.csv"
            selector = root / "selector.csv"
            rows = [
                _event_row(
                    "2024-01-02",
                    p1="2330.TW",
                    p2="2454.TW",
                    p1_dir="stock_attack",
                    p2_dir="market_exposure",
                    final_target="",
                )
            ]
            rows.extend(
                _event_row(
                    f"2024-01-{day:02d}",
                    p1="2330.TW",
                    p2="2330.TW",
                    p1_dir="stock_attack",
                    p2_dir="stock_attack",
                    final_target="2330.TW",
                )
                for day in range(3, 9)
            )
            pd.DataFrame(rows).to_csv(event, index=False)
            pd.DataFrame(
                [
                    {
                        "signal_date": "2024-01-02",
                        "pool3_selector_diagnostic_state": "veto_explanation",
                        "pool3_selector_diagnostic_boundary": "report_only",
                    }
                ]
            ).to_csv(selector, index=False)

            output = run_final_decision_layer_diagnostic(
                event_panel_path=event,
                pool3_selector_panel_path=selector,
                output_dir=root / "out",
            )

            panel = pd.read_csv(output / "final_decision_layer_diagnostic_panel.csv")
            row = panel[panel["signal_date"] == "2024-01-02"].iloc[0]
            self.assertEqual(row["final_decision_layer_state"], "pool1_pool2_formal_conflict")
            self.assertEqual(row["pool3_shadow_diagnostic_state"], "veto_explanation")
            self.assertTrue(bool(row["pool3_shadow_not_formal_flag"]))


def _event_row(
    date: str,
    *,
    p1: str,
    p2: str,
    p1_dir: str,
    p2_dir: str,
    final_target: str,
) -> dict[str, object]:
    return {
        "period": "2024_now",
        "signal_date": date,
        "pool1_ticker": p1,
        "pool1_vote_state": "eligible_vote" if p1 else "no_selection",
        "pool1_direction_state": p1_dir,
        "pool2_ticker": p2,
        "pool2_vote_state": "eligible_vote" if p2 else "no_selection",
        "pool2_direction_state": p2_dir,
        "pool3_ticker": "2882.TW",
        "pool3_vote_state": "eligible_vote",
        "pool3_direction_state": "stock_attack",
        "formal_final_target": final_target,
        "final_target_source": "exact_ticker_consensus" if final_target else "none",
        "trade_action": "hold",
    }


if __name__ == "__main__":
    unittest.main()
