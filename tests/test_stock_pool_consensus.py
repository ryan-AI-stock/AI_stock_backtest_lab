from __future__ import annotations

import unittest

import test_paths  # noqa: F401

from backtest_lab.decision_layers import CANDIDATE_SOURCE
from backtest_lab.stock_pool_consensus import build_consensus


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
