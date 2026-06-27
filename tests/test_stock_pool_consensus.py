from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import test_paths  # noqa: F401

from backtest_lab.decision_layers import CANDIDATE_SOURCE
from backtest_lab.stock_pool_consensus import build_consensus, write_consensus_outputs


class StockPoolConsensusTest(unittest.TestCase):
    def test_build_consensus_selects_two_to_one_winner(self) -> None:
        consensus = build_consensus(
            {
                "signal_date": "2026-06-12",
                "generated": [
                    _generated("AI池", "2330.TW", "台積電(2330)"),
                    _generated("0050池", "2330.TW", "台積電(2330)"),
                    _generated("核心池", "00631L.TW", "0050正二(00631L)"),
                ],
                "skipped": [],
            }
        )

        self.assertEqual(consensus["result_state"], "consensus")
        self.assertEqual(consensus["winner_ticker"], "2330.TW")
        self.assertIn("2/3", consensus["reason"])
        self.assertEqual(consensus["consensus_type"], "consensus_observation")
        self.assertEqual(consensus["decision_layer"], CANDIDATE_SOURCE)
        self.assertFalse(consensus["active_in_trade_decision"])
        self.assertIsNone(consensus["formal_trade_target"])
        self.assertFalse(consensus["voters"][0]["active_in_trade_decision"])
        health = consensus["health_diagnostic"]
        self.assertEqual(health["decision_state"], "weak_consensus")
        self.assertEqual(health["consensus_strength"], "weak")
        self.assertEqual(health["exact_ticker_consensus_rate"], 0.6667)
        self.assertTrue(health["exact_ticker_consensus"])
        self.assertEqual(health["exact_ticker_consensus_group"], "2330.TW")
        self.assertEqual(health["decision_source"], "exact_2_of_3_ticker")
        self.assertEqual(health["consensus_health_bucket"], "acceptable")
        self.assertEqual(health["actionable_decision_rate"], 1.0)
        self.assertFalse(health["active_in_trade_decision"])
        self.assertFalse(health["formal_model_changed"])
        self.assertFalse(health["trade_decision_changed"])

    def test_build_consensus_marks_three_way_divergence(self) -> None:
        consensus = build_consensus(
            {
                "signal_date": "2026-06-12",
                "generated": [
                    _generated("AI池", "2454.TW", "聯發科(2454)"),
                    _generated("0050池", "2330.TW", "台積電(2330)"),
                    _generated("核心池", "00631L.TW", "0050正二(00631L)"),
                ],
                "skipped": [],
            }
        )

        self.assertEqual(consensus["result_state"], "divergent")
        self.assertIsNone(consensus["winner_ticker"])
        self.assertEqual(consensus["health_diagnostic"]["decision_state"], "divergent_observe")
        self.assertEqual(consensus["health_diagnostic"]["divergent_rate"], 1.0)
        self.assertEqual(consensus["health_diagnostic"]["actionable_decision_rate"], 0.0)

    def test_build_consensus_ignores_observation_only_pool_vote(self) -> None:
        consensus = build_consensus(
            {
                "signal_date": "2026-06-18",
                "generated": [
                    _generated(
                        "AI池",
                        "2454.TW",
                        "聯發科(2454)",
                        eligible=False,
                        selection_layer="observation_only",
                        selection_reason="個股攻擊閘門未開啟",
                    ),
                    _generated(
                        "0050池",
                        "00631L.TW",
                        "0050正二(00631L)",
                        eligible=True,
                        selection_layer="market_exposure_tool",
                    ),
                    _generated(
                        "核心池",
                        "00631L.TW",
                        "0050正二(00631L)",
                        eligible=True,
                        selection_layer="market_exposure_tool",
                    ),
                ],
                "skipped": [],
            }
        )

        self.assertEqual(consensus["result_state"], "consensus")
        self.assertEqual(consensus["winner_ticker"], "00631L.TW")
        self.assertEqual(len(consensus["voters"]), 2)
        self.assertEqual(consensus["votes"][0]["vote_count"], 2)
        self.assertEqual(consensus["skipped_vote_pools"][0]["top_ticker"], "2454.TW")
        self.assertFalse(consensus["skipped_vote_pools"][0]["eligible_for_pool_selection"])
        self.assertEqual(consensus["health_diagnostic"]["decision_state"], "defensive_or_market_exposure")
        self.assertEqual(consensus["health_diagnostic"]["exact_ticker_consensus_rate"], 0.6667)

    def test_build_consensus_handles_no_selection_votes(self) -> None:
        consensus = build_consensus(
            {
                "signal_date": "2026-06-18",
                "generated": [
                    _generated(
                        "AI池",
                        "2454.TW",
                        "聯發科(2454)",
                        eligible=False,
                        selection_layer="observation_only",
                    ),
                    _generated(
                        "0050池",
                        "2327.TW",
                        "國巨(2327)",
                        eligible=False,
                        selection_layer="observation_only",
                    ),
                ],
                "skipped": [],
            }
        )

        self.assertEqual(consensus["result_state"], "no_vote")
        self.assertIsNone(consensus["winner_ticker"])
        self.assertEqual(consensus["health_diagnostic"]["decision_state"], "data_insufficient")
        self.assertEqual(consensus["health_diagnostic"]["no_vote_or_data_insufficient_rate"], 1.0)

    def test_build_consensus_marks_three_to_zero_as_strong_consensus(self) -> None:
        consensus = build_consensus(
            {
                "signal_date": "2026-06-12",
                "generated": [
                    _generated("AI池", "2330.TW", "台積電(2330)"),
                    _generated("0050池", "2330.TW", "台積電(2330)"),
                    _generated("核心池", "2330.TW", "台積電(2330)"),
                ],
                "skipped": [],
            }
        )

        self.assertEqual(consensus["result_state"], "consensus")
        self.assertEqual(consensus["winner_ticker"], "2330.TW")
        self.assertEqual(consensus["health_diagnostic"]["decision_state"], "strong_consensus")
        self.assertEqual(consensus["health_diagnostic"]["exact_ticker_consensus_rate"], 1.0)
        self.assertEqual(consensus["health_diagnostic"]["direction_consensus_rate"], 1.0)
        self.assertEqual(consensus["health_diagnostic"]["decision_source"], "exact_3_of_3_ticker")
        self.assertEqual(consensus["health_diagnostic"]["consensus_health_bucket"], "healthy")

    def test_write_consensus_outputs_writes_health_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = {
                "signal_date": "2026-06-12",
                "generated": [
                    _generated("AI池", "2330.TW", "台積電(2330)"),
                    _generated("0050池", "2330.TW", "台積電(2330)"),
                    _generated("核心池", "00631L.TW", "0050正二(00631L)"),
                ],
                "skipped": [],
            }

            consensus = write_consensus_outputs(root, manifest)

            self.assertEqual(consensus["health_diagnostic"]["decision_state"], "weak_consensus")
            self.assertTrue((root / "stock_pool_consensus_health.csv").exists())
            report = (root / "stock_pool_consensus_report.md").read_text(encoding="utf-8")
            self.assertIn("共識健康診斷", report)
            self.assertIn("decision_state：weak_consensus", report)
            self.assertIn("decision_source：exact_2_of_3_ticker", report)

    def test_consensus_report_hides_legacy_pool3_visible_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = {
                "signal_date": "2026-06-25",
                "generated": [
                    _generated("AI主線池", "2454.TW", "聯發科(2454)"),
                    _generated("大型廣度池", "2303.TW", "聯電(2303)"),
                    _generated("large_core_bluechip_v0", "00631L.TW", "0050正二(00631L)"),
                ],
                "skipped": [],
            }

            consensus = write_consensus_outputs(root, manifest)
            report = (root / "stock_pool_consensus_report.md").read_text(encoding="utf-8")

            self.assertEqual(len(consensus["pool_diagnostics"]), 3)
            self.assertIn("AI主線池", report)
            self.assertIn("大型廣度池", report)
            self.assertNotIn("large_core_bluechip_v0", report)
            self.assertNotIn("風格補強", report)

    def test_build_consensus_marks_direction_consensus_as_protocol_candidate_only(self) -> None:
        consensus = build_consensus(
            {
                "signal_date": "2026-06-12",
                "generated": [
                    _generated("AI池", "2454.TW", "聯發科(2454)"),
                    _generated("0050池", "2330.TW", "台積電(2330)"),
                    _generated("核心池", "2308.TW", "台達電(2308)"),
                ],
                "skipped": [],
            }
        )

        self.assertEqual(consensus["result_state"], "divergent")
        self.assertIsNone(consensus["winner_ticker"])
        health = consensus["health_diagnostic"]
        self.assertTrue(health["direction_consensus"])
        self.assertEqual(health["direction_consensus_group"], "stock_attack")
        self.assertEqual(health["decision_source"], "protocol_resolved_divergence")
        self.assertFalse(health["decision_protocol_used"])
        self.assertEqual(health["protocol_usage_category"], "candidate_not_applied")
        self.assertEqual(health["actionable_decision_state"], "protocol_candidate_diagnostic")
        self.assertEqual(health["actionable_decision_rate"], 0.0)

    def test_build_consensus_marks_data_blocked_and_fake_consensus_flags(self) -> None:
        consensus = build_consensus(
            {
                "signal_date": "2026-06-18",
                "generated": [
                    _generated(
                        "AI池",
                        "2454.TW",
                        "聯發科(2454)",
                        eligible=False,
                        selection_layer="observation_only",
                        selection_reason="個股攻擊閘門未開啟",
                    ),
                    _generated("0050池", "00631L.TW", "0050正二(00631L)", selection_layer="market_exposure_tool"),
                    _generated("核心池", "00631L.TW", "0050正二(00631L)", selection_layer="market_exposure_tool"),
                ],
                "skipped": [],
            }
        )

        health = consensus["health_diagnostic"]
        self.assertEqual(consensus["winner_ticker"], "00631L.TW")
        self.assertEqual(health["decision_source"], "exact_2_of_3_ticker")
        self.assertEqual(health["consensus_health_bucket"], "warning")
        self.assertIn("consensus_with_ineligible_pool", health["fake_consensus_flags"])
        self.assertIn("observation_only_excluded", health["fake_consensus_flags"])
        self.assertEqual(consensus["pool_diagnostics"][0]["eligible_vote"], True)
        self.assertEqual(consensus["pool_diagnostics"][-1]["data_readiness_state"], "blocked")
        self.assertEqual(consensus["pool_diagnostics"][-1]["blocked_reason"], "個股攻擊閘門未開啟")


def _generated(
    pool_name: str,
    ticker: str,
    display: str,
    *,
    eligible: bool = True,
    selection_layer: str = "formal_candidate",
    selection_reason: str = "",
) -> dict:
    return {
        "pool_id": pool_name,
        "pool_name": pool_name,
        "vote_group": "three_perspective_v1",
        "top_ticker": ticker,
        "top_display": display,
        "eligible_for_pool_selection": eligible,
        "selection_layer": selection_layer,
        "selection_reason": selection_reason,
        "action_state": "有合格模型目標",
    }


if __name__ == "__main__":
    unittest.main()
