from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

import test_paths  # noqa: F401

from backtest_lab.stock_pool_candidate_review import (
    build_candidate_review,
    load_ai_theme_candidate_source,
    load_core_defensive_candidate_source,
    write_candidate_reviews,
)
from backtest_lab.stock_pool_store import StockPoolStore, symbol_entry


class StockPoolCandidateReviewTest(unittest.TestCase):
    def test_ai_theme_pool_uses_monthly_candidate_csv_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pool = next(
                item
                for item in StockPoolStore(Path(tmp) / "stock_pools.json").list_pools()
                if item["pool_id"] == "ai_theme_large_cap_v20260613"
            )

        review = build_candidate_review(pool, signal_date="2026-06-12")

        self.assertEqual(review["frequency"], "monthly")
        self.assertEqual(review["source_mode"], "ai_theme_candidate_csv")
        self.assertEqual(review["source_status"], "source_ready")
        self.assertEqual(review["decision"], "monthly_auto_review_available")
        self.assertIn("AI主線受惠程度", review["required_evidence"])
        self.assertEqual(review["candidate_count"], 9)
        self.assertEqual(review["source_active_count"], 7)
        self.assertGreaterEqual(review["source_watch_count"], 1)

    def test_ai_theme_candidate_source_ignores_future_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ai_theme_candidates.csv"
            path.write_text(
                "\n".join(
                    [
                        "effective_date,ticker,symbol,name,theme_role,review_status,is_current_member,ai_exposure_score,liquidity_score,fundamental_score,theme_strength_score,review_reason",
                        "2026-06-01,2330.TW,2330,台積電,AI半導體核心製造,active,true,95,98,92,95,ready",
                        "2026-07-01,9999.TW,9999,未來股,測試,watch,false,99,99,99,99,future",
                    ]
                ),
                encoding="utf-8",
            )

            candidates = load_ai_theme_candidate_source(path, signal_date="2026-06-12")

        self.assertEqual([item["ticker"] for item in candidates], ["2330.TW"])

    def test_core_defensive_pool_uses_monthly_candidate_csv_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pool = next(
                item
                for item in StockPoolStore(Path(tmp) / "stock_pools.json").list_pools()
                if item["pool_id"] == "large_core_bluechip_v0"
            )

        review = build_candidate_review(pool, signal_date="2026-06-12")

        self.assertEqual(review["frequency"], "monthly")
        self.assertEqual(review["source_mode"], "core_defensive_candidate_csv")
        self.assertEqual(review["source_status"], "source_ready")
        self.assertEqual(review["decision"], "monthly_auto_review_available")
        self.assertIn("跨產業代表性", review["required_evidence"])
        self.assertEqual(review["candidate_count"], 17)
        self.assertEqual(review["source_active_count"], 17)
        self.assertGreaterEqual(review["source_watch_count"], 1)

    def test_core_defensive_candidate_source_ignores_future_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "core_defensive_candidates.csv"
            path.write_text(
                "\n".join(
                    [
                        "effective_date,ticker,symbol,name,style_role,review_status,is_current_member,defensive_score,stability_score,cross_sector_score,fundamental_score,review_reason",
                        "2026-06-01,2412.TW,2412,中華電,電信防守核心,active,true,92,94,82,84,ready",
                        "2026-07-01,9999.TW,9999,未來防守股,測試,watch,false,99,99,99,99,future",
                    ]
                ),
                encoding="utf-8",
            )

            candidates = load_core_defensive_candidate_source(path, signal_date="2026-06-12")

        self.assertEqual([item["ticker"] for item in candidates], ["2412.TW"])

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
                        "source_mode": "ai_theme_candidate_csv",
                        "source_status": "source_ready",
                        "decision": "monthly_auto_review_available",
                        "candidate_count": 9,
                        "source_candidate_count": 11,
                        "source_active_count": 7,
                        "source_watch_count": 4,
                        "source_path": "data/ai_theme_candidates.csv",
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
            self.assertEqual(rows.loc[0, "source_status"], "source_ready")
            self.assertEqual(rows.loc[0, "source_active_count"], 7)


if __name__ == "__main__":
    unittest.main()
