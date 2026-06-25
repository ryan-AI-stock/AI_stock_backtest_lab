from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

import test_paths  # noqa: F401

from backtest_lab.final_decision_layer_spec_diagnostic import run_final_decision_layer_spec_diagnostic


class FinalDecisionLayerSpecDiagnosticTest(unittest.TestCase):
    def test_etf_and_pool3_shadow_are_excluded_from_formal_stock_vote(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            event = root / "event.csv"
            pd.DataFrame(
                [
                    _row(
                        "2024-01-02",
                        pool1="00631L.TW",
                        pool1_layer="market_exposure_tool",
                        pool1_dir="market_exposure",
                        pool2="00631L.TW",
                        pool2_layer="market_exposure_tool",
                        pool2_dir="market_exposure",
                        pool3="00631L.TW",
                        pool3_layer="market_exposure_tool",
                        pool3_dir="market_exposure",
                        final_target="00631L.TW",
                    )
                ]
            ).to_csv(event, index=False)

            output = run_final_decision_layer_spec_diagnostic(event_panel_path=event, output_dir=root / "out")

            pool = pd.read_csv(output / "pool_signal_normalized_panel.csv")
            self.assertFalse(pool["eligible_stock_vote"].astype(bool).any())
            self.assertTrue(pool["eligible_market_exposure"].astype(bool).any())
            pool3 = pool[pool["pool_id"].eq("pool3")].iloc[0]
            self.assertTrue(bool(pool3["pool3_shadow_not_formal_flag"]))

            state = pd.read_csv(output / "final_decision_state_panel.csv").iloc[0]
            self.assertEqual(state["final_decision_state"], "defensive_market_exposure")
            self.assertEqual(state["final_target_type"], "market_exposure_tool")
            self.assertFalse(bool(state["etf_counted_as_stock_vote"]))
            self.assertFalse(bool(state["pool3_shadow_used_as_formal"]))
            self.assertEqual(int(state["eligible_formal_stock_vote_count"]), 0)

            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertFalse(manifest["formal_model_changed"])
            self.assertFalse(manifest["trade_decision_changed"])
            self.assertFalse(manifest["active_in_trade_decision"])
            self.assertEqual(manifest["final_decision_layer_boundary"], "report_only_diagnostic")
            self.assertFalse(manifest["pool3_shadow_used_as_formal"])
            self.assertFalse(manifest["etf_counted_as_stock_vote"])

    def test_forced_stop_and_data_insufficient_take_priority(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            event = root / "event.csv"
            pd.DataFrame(
                [
                    _row(
                        "2024-01-02",
                        pool1="2330.TW",
                        pool2="2330.TW",
                        final_target="2330.TW",
                        trade_blocked_reason="risk_gate_forced_no_trade",
                    ),
                    _row(
                        "2024-01-03",
                        pool1="",
                        pool1_state="no_selection",
                        pool2="",
                        pool2_state="no_selection",
                        final_target="",
                    ),
                ]
            ).to_csv(event, index=False)

            output = run_final_decision_layer_spec_diagnostic(event_panel_path=event, output_dir=root / "out")
            panel = pd.read_csv(output / "final_decision_state_panel.csv")
            self.assertEqual(panel.iloc[0]["final_decision_state"], "forced_stop")
            self.assertTrue(bool(panel.iloc[0]["not_eligible_for_formal_selector"]))
            self.assertEqual(panel.iloc[1]["final_decision_state"], "data_insufficient")
            self.assertTrue(bool(panel.iloc[1]["data_insufficient_flag"]))

    def test_exact_direction_and_fake_health_are_separate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            event = root / "event.csv"
            pd.DataFrame(
                [
                    _row(
                        "2024-01-02",
                        pool1="2330.TW",
                        pool2="2454.TW",
                        pool1_dir="stock_attack",
                        pool2_dir="stock_attack",
                        final_target="",
                    ),
                    _row(
                        "2024-01-03",
                        pool1="2330.TW",
                        pool2="2330.TW",
                        pool1_dir="stock_attack",
                        pool2_dir="stock_attack",
                        final_target="2330.TW",
                    ),
                ]
            ).to_csv(event, index=False)

            output = run_final_decision_layer_spec_diagnostic(event_panel_path=event, output_dir=root / "out")
            panel = pd.read_csv(output / "final_decision_state_panel.csv")
            fake = panel[panel["signal_date"].eq("2024-01-02")].iloc[0]
            self.assertEqual(fake["exact_ticker_consensus_state"], "no_exact_consensus")
            self.assertEqual(fake["direction_consensus_state"], "direction_consensus")
            self.assertTrue(bool(fake["fake_direction_consensus_flag"]))
            self.assertTrue(bool(fake["not_eligible_for_formal_selector"]))
            strong = panel[panel["signal_date"].eq("2024-01-03")].iloc[0]
            self.assertEqual(strong["final_decision_state"], "strong_consensus")
            self.assertEqual(strong["exact_ticker_consensus_state"], "exact_consensus")

            health = pd.read_csv(output / "consensus_health_by_period.csv").iloc[0]
            self.assertEqual(float(health["exact_ticker_consensus_rate"]), 0.5)
            self.assertEqual(float(health["direction_consensus_rate"]), 1.0)
            self.assertTrue((output / "target_priority_panel.csv").exists())
            self.assertTrue((output / "market_exposure_tool_panel.csv").exists())


def _row(
    date: str,
    *,
    period: str = "2024_now",
    pool1: str = "2330.TW",
    pool1_state: str = "eligible_vote",
    pool1_layer: str = "formal_candidate",
    pool1_dir: str = "stock_attack",
    pool2: str = "2330.TW",
    pool2_state: str = "eligible_vote",
    pool2_layer: str = "formal_candidate",
    pool2_dir: str = "stock_attack",
    pool3: str = "2882.TW",
    pool3_state: str = "eligible_vote",
    pool3_layer: str = "formal_candidate",
    pool3_dir: str = "stock_attack",
    final_target: str = "2330.TW",
    trade_blocked_reason: str = "",
) -> dict[str, object]:
    return {
        "period": period,
        "signal_date": date,
        "pool1_ticker": pool1,
        "pool1_selection_layer": pool1_layer,
        "pool1_vote_state": pool1_state,
        "pool1_direction_state": pool1_dir,
        "pool1_blocked_reason": "",
        "pool2_ticker": pool2,
        "pool2_selection_layer": pool2_layer,
        "pool2_vote_state": pool2_state,
        "pool2_direction_state": pool2_dir,
        "pool2_blocked_reason": "",
        "pool3_ticker": pool3,
        "pool3_selection_layer": pool3_layer,
        "pool3_vote_state": pool3_state,
        "pool3_direction_state": pool3_dir,
        "pool3_blocked_reason": "",
        "formal_final_target": final_target,
        "final_target_source": "exact_ticker_consensus" if final_target else "none",
        "trade_action": "hold",
        "trade_blocked_reason": trade_blocked_reason,
    }


if __name__ == "__main__":
    unittest.main()
