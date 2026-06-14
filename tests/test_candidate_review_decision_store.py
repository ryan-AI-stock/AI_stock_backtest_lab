from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import test_paths  # noqa: F401

from backtest_lab.candidate_review_decision_store import CandidateReviewDecisionStore


class CandidateReviewDecisionStoreTest(unittest.TestCase):
    def test_record_upserts_latest_decision_for_pool_and_ticker(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = CandidateReviewDecisionStore(Path(tmp) / "decisions.json")

            first = store.record(
                {
                    "pool_id": "ai_theme_large_cap_v20260613",
                    "pool_name": "AI主線攻擊池",
                    "ticker": "2330.TW",
                    "display": "台積電(2330)",
                    "decision": "keep_watch",
                    "signal_date": "2026-06-12",
                    "note": "先觀察",
                }
            )
            second = store.record(
                {
                    "pool_id": "ai_theme_large_cap_v20260613",
                    "ticker": "2330.TW",
                    "display": "台積電(2330)",
                    "decision": "approve_add",
                    "signal_date": "2026-06-13",
                }
            )

            state = store.state()
            self.assertEqual(first["decision_label"], "保留觀察")
            self.assertEqual(second["decision_label"], "列入候選")
            self.assertEqual(len(state["decisions"]), 1)
            self.assertEqual(state["latest_by_key"]["ai_theme_large_cap_v20260613|2330.TW"]["decision"], "approve_add")

    def test_rejects_unknown_decision(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = CandidateReviewDecisionStore(Path(tmp) / "decisions.json")

            with self.assertRaises(ValueError):
                store.record(
                    {
                        "pool_id": "pool",
                        "ticker": "2330.TW",
                        "decision": "buy_now",
                        "signal_date": "2026-06-12",
                    }
                )


if __name__ == "__main__":
    unittest.main()
