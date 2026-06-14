from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

import test_paths  # noqa: F401

from backtest_lab.stock_pool_candidate_review import build_candidate_review, write_candidate_reviews
from backtest_lab.stock_pool_store import StockPoolStore, symbol_entry


class StockPoolCandidateReviewTest(unittest.TestCase):
    def test_ai_theme_pool_uses_monthly_manual_evidence_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pool = next(
                item
                for item in StockPoolStore(Path(tmp) / "stock_pools.json").list_pools()
                if item["pool_id"] == "ai_theme_large_cap_v20260613"
            )

        review = build_candidate_review(pool, signal_date="2026-06-12")

        self.assertEqual(review["frequency"], "monthly")
        self.assertEqual(review["source_mode"], "manual_evidence_gate")
        self.assertEqual(review["source_status"], "manual_review_required")
        self.assertEqual(review["decision"], "keep_current_until_monthly_evidence_review")
        self.assertIn("AI主線受惠程度", review["required_evidence"])
        self.assertEqual(review["candidate_count"], 9)

    def test_tw50_pool_marks_point_in_time_source_ready(self) -> None:
        pool = {
            "pool_id": "tw50_dynamic_constituents_v0",
            "name": "大型市場廣度池 v0",
            "candidate_review_frequency": "monthly",
            "dynamic_constituents": {"source": "tw50_history_csv", "path": "data/tw50_constituents.csv"},
            "resolved_symbols": [symbol_entry("2330.TW", source="tw50_constituents")],
        }

        review = build_candidate_review(pool, signal_date="2026-06-12")

        self.assertEqual(review["source_mode"], "point_in_time_constituents")
        self.assertEqual(review["source_status"], "source_ready")
        self.assertEqual(review["decision"], "monthly_auto_review_available")

    def test_write_candidate_reviews_creates_json_and_csv(self) -> None:
        manifest = {
            "generated": [
                {
                    "candidate_review": {
                        "pool_id": "ai_theme_large_cap_v20260613",
                        "pool_name": "AI主線攻擊池 v20260613",
                        "review_date": "2026-06-12",
                        "frequency": "monthly",
                        "source_mode": "manual_evidence_gate",
                        "source_status": "manual_review_required",
                        "decision": "keep_current_until_monthly_evidence_review",
                        "candidate_count": 9,
                        "required_evidence": ["AI主線受惠程度"],
                        "policy": "月頻檢查候選名單",
                    }
                }
            ],
            "skipped": [],
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_candidate_reviews(root, manifest)

            self.assertTrue((root / "stock_pool_candidate_reviews.json").exists())
            rows = pd.read_csv(root / "stock_pool_candidate_reviews.csv")
            self.assertEqual(rows.loc[0, "frequency"], "monthly")
            self.assertEqual(rows.loc[0, "source_status"], "manual_review_required")


if __name__ == "__main__":
    unittest.main()
