from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import test_paths  # noqa: F401

from backtest_lab.stock_pool_store import StockPoolStore, normalize_ticker, parse_symbol_lines


class StockPoolStoreTest(unittest.TestCase):
    def test_default_pools_include_core_experiment_legacy_sections(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pools = StockPoolStore(Path(tmp) / "stock_pools.json").list_pools()

        pool_ids = {pool["pool_id"] for pool in pools}
        self.assertIn("ai_theme_large_cap_v20260613", pool_ids)
        self.assertIn("tw50_dynamic_constituents_v0", pool_ids)
        self.assertIn("large_core_bluechip_v0", pool_ids)
        self.assertIn("large_cap_best_v20260605", pool_ids)
        self.assertIn("radar_mid_small_calibrated_v1", pool_ids)
        self.assertIn("model_scorecard_ep10", pool_ids)
        core_ids = {pool["pool_id"] for pool in pools if pool["ui_section"] == "official_core"}
        self.assertEqual(
            core_ids,
            {"ai_theme_large_cap_v20260613", "tw50_dynamic_constituents_v0", "large_core_bluechip_v0"},
        )
        large = next(pool for pool in pools if pool["pool_id"] == "large_cap_best_v20260605")
        core = next(pool for pool in pools if pool["pool_id"] == "large_core_bluechip_v0")
        scorecard = next(pool for pool in pools if pool["pool_id"] == "model_scorecard_ep10")
        official = {pool["pool_id"]: pool for pool in pools if pool["ui_section"] == "official_core"}
        self.assertEqual(len(large["resolved_symbols"]), 9)
        self.assertEqual(official["ai_theme_large_cap_v20260613"]["role_name"], "主線攻擊專家")
        self.assertEqual(official["tw50_dynamic_constituents_v0"]["role_name"], "市場廣度專家")
        self.assertEqual(core["role_name"], "核心防守與風格轉移專家")
        self.assertEqual(
            {pool["candidate_review_frequency"] for pool in official.values()},
            {"monthly"},
        )
        self.assertEqual(official["ai_theme_large_cap_v20260613"]["candidate_review_config"]["source_mode"], "ai_theme_candidate_csv")
        self.assertEqual(official["ai_theme_large_cap_v20260613"]["candidate_review_config"]["path"], "data/ai_theme_candidates.csv")
        self.assertEqual(official["tw50_dynamic_constituents_v0"]["candidate_review_config"]["source_mode"], "point_in_time_constituents")
        self.assertEqual(official["large_core_bluechip_v0"]["candidate_review_config"]["source_mode"], "manual_evidence_gate")
        self.assertEqual(core["strategy_preset"], "core_defensive_style_v1")
        self.assertFalse(large["operational_observation"])
        self.assertFalse(scorecard["operational_observation"])
        self.assertEqual(large["dispatch"]["workflow_file"], "stock_pool_observation.yml")
        self.assertEqual(scorecard["dispatch"]["workflow_file"], "model_scorecard_report.yml")

    def test_scorecard_pool_dynamic_third_symbol_follows_latest_signal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pools = StockPoolStore(Path(tmp) / "stock_pools.json").list_pools(
                latest_signal={"target_ticker": "2330.TW"}
            )

        scorecard = next(pool for pool in pools if pool["pool_id"] == "model_scorecard_ep10")
        self.assertEqual([item["ticker"] for item in scorecard["resolved_symbols"]], ["0050.TW", "00631L.TW", "2330.TW"])
        self.assertEqual(scorecard["resolved_symbols"][2]["display"], "台積電(2330)")
        self.assertEqual(scorecard["dynamic_binding"]["source_pool_id"], "ai_theme_large_cap_v20260613")

    def test_custom_pool_upsert_and_delete(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = StockPoolStore(Path(tmp) / "stock_pools.json")
            store.upsert_pool({"name": "我的觀察池", "symbols_text": "2330\n2454.TW\n2330"})
            pool = next(pool for pool in store.list_pools() if pool["name"] == "我的觀察池")
            self.assertEqual([item["ticker"] for item in pool["symbols"]], ["2330.TW", "2454.TW"])

            store.delete_pool(pool["pool_id"])
            self.assertNotIn(pool["pool_id"], {item["pool_id"] for item in store.list_pools()})

    def test_rejects_unknown_strategy_preset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = StockPoolStore(Path(tmp) / "stock_pools.json")
            with self.assertRaisesRegex(ValueError, "Unsupported strategy_preset"):
                store.upsert_pool({"name": "錯誤池", "symbols_text": "2330", "strategy_preset": "bad_preset"})

    def test_official_core_pool_is_editable_but_not_deletable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = StockPoolStore(Path(tmp) / "stock_pools.json")
            store.upsert_pool(
                {
                    "pool_id": "large_core_bluechip_v0",
                    "name": "大型核心權值股池 v0",
                    "strategy_preset": "core_defensive_style_v1",
                    "symbols_text": "0050\n00631L\n2330",
                    "description": "updated",
                }
            )
            pool = next(pool for pool in store.list_pools() if pool["pool_id"] == "large_core_bluechip_v0")
            self.assertTrue(pool["locked"])
            self.assertEqual([item["ticker"] for item in pool["symbols"]], ["0050.TW", "00631L.TW", "2330.TW"])
            with self.assertRaisesRegex(ValueError, "不可刪除"):
                store.delete_pool("large_core_bluechip_v0")

    def test_existing_store_merges_new_default_core_pools(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "stock_pools.json"
            path.write_text(
                '{"schema_version": 1, "pools": [{"pool_id": "model_scorecard_ep10", "name": "old", "kind": "task", "locked": false, "strategy_preset": "delayed_public_scorecard_v1", "operational_observation": false, "symbols": []}, {"pool_id": "large_cap_best_v20260605", "name": "old legacy", "kind": "built_in", "locked": true, "strategy_preset": "best_v20260605", "operational_observation": true, "symbols": []}]}',
                encoding="utf-8",
            )
            pools = StockPoolStore(path).list_pools()

        pool_ids = {pool["pool_id"] for pool in pools}
        self.assertIn("ai_theme_large_cap_v20260613", pool_ids)
        self.assertIn("tw50_dynamic_constituents_v0", pool_ids)
        self.assertIn("large_core_bluechip_v0", pool_ids)
        legacy = next(pool for pool in pools if pool["pool_id"] == "large_cap_best_v20260605")
        self.assertFalse(legacy["operational_observation"])

    def test_normalize_manual_ticker_input(self) -> None:
        self.assertEqual(normalize_ticker("台積電(2330)"), "2330.TW")
        self.assertEqual(normalize_ticker("00631L"), "00631L.TW")
        self.assertEqual(parse_symbol_lines("2330\n\n2454")[0]["display"], "台積電(2330)")


if __name__ == "__main__":
    unittest.main()
