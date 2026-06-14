from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import test_paths  # noqa: F401

from backtest_lab.candidate_review_decision_draft import build_candidate_review_decision_draft


class CandidateReviewDecisionDraftTest(unittest.TestCase):
    def test_builds_update_and_append_draft_without_writing_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "candidates.csv"
            original = (
                "effective_date,ticker,symbol,name,review_status,is_current_member,review_reason\n"
                "2026-06-01,2330.TW,2330,台積電,watch,false,觀察中\n"
            )
            source.write_text(original, encoding="utf-8")
            pools = [
                {
                    "pool_id": "pool1",
                    "name": "測試池",
                    "candidate_review_config": {
                        "source_mode": "ai_theme_candidate_csv",
                        "path": str(source),
                    },
                }
            ]
            decisions = [
                {
                    "pool_id": "pool1",
                    "ticker": "2330.TW",
                    "display": "台積電(2330)",
                    "decision": "approve_add",
                    "decision_label": "列入候選",
                    "signal_date": "2026-06-12",
                    "note": "升級為 active",
                },
                {
                    "pool_id": "pool1",
                    "ticker": "2454.TW",
                    "display": "聯發科(2454)",
                    "decision": "keep_watch",
                    "decision_label": "保留觀察",
                    "signal_date": "2026-06-12",
                },
            ]

            draft = build_candidate_review_decision_draft(pools=pools, decisions=decisions)

            self.assertEqual(draft["change_count"], 2)
            self.assertEqual(draft["changes"][0]["action"], "update_row")
            self.assertEqual(draft["changes"][0]["draft_status"], "active")
            self.assertEqual(draft["changes"][0]["row_preview"]["review_reason"], "升級為 active")
            self.assertEqual(draft["changes"][1]["action"], "append_row")
            self.assertEqual(draft["changes"][1]["draft_status"], "watch")
            self.assertEqual(source.read_text(encoding="utf-8"), original)

    def test_skips_non_csv_source_modes(self) -> None:
        pools = [
            {
                "pool_id": "tw50",
                "candidate_review_config": {"source_mode": "point_in_time_constituents"},
            }
        ]
        decisions = [{"pool_id": "tw50", "ticker": "2330.TW", "decision": "keep_current", "signal_date": "2026-06-12"}]

        draft = build_candidate_review_decision_draft(pools=pools, decisions=decisions)

        self.assertEqual(draft["change_count"], 0)
        self.assertEqual(draft["skipped"][0]["reason"], "unsupported_source_mode")


if __name__ == "__main__":
    unittest.main()
