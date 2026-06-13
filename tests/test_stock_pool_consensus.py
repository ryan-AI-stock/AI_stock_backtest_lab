from __future__ import annotations

import unittest

import test_paths  # noqa: F401

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


def _generated(pool_name: str, ticker: str, display: str) -> dict:
    return {
        "pool_id": pool_name,
        "pool_name": pool_name,
        "vote_group": "three_perspective_v1",
        "top_ticker": ticker,
        "top_display": display,
        "action_state": "有合格模型目標",
    }


if __name__ == "__main__":
    unittest.main()

