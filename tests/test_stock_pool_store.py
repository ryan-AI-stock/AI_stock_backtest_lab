from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import test_paths  # noqa: F401

from backtest_lab.stock_pool_store import StockPoolStore, normalize_ticker, parse_symbol_lines


class StockPoolStoreTest(unittest.TestCase):
    def test_default_pools_include_large_cap_radar_and_scorecard(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pools = StockPoolStore(Path(tmp) / "stock_pools.json").list_pools()

        pool_ids = {pool["pool_id"] for pool in pools}
        self.assertIn("large_cap_best_v20260605", pool_ids)
        self.assertIn("radar_mid_small_calibrated_v1", pool_ids)
        self.assertIn("model_scorecard_ep10", pool_ids)
        large = next(pool for pool in pools if pool["pool_id"] == "large_cap_best_v20260605")
        scorecard = next(pool for pool in pools if pool["pool_id"] == "model_scorecard_ep10")
        self.assertEqual(len(large["resolved_symbols"]), 9)
        self.assertTrue(large["operational_observation"])
        self.assertFalse(scorecard["operational_observation"])
        self.assertEqual(large["dispatch"]["workflow_file"], "frozen_strategy_daily_report.yml")
        self.assertEqual(scorecard["dispatch"]["workflow_file"], "model_scorecard_report.yml")

    def test_scorecard_pool_dynamic_third_symbol_follows_latest_signal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pools = StockPoolStore(Path(tmp) / "stock_pools.json").list_pools(
                latest_signal={"target_ticker": "2330.TW"}
            )

        scorecard = next(pool for pool in pools if pool["pool_id"] == "model_scorecard_ep10")
        self.assertEqual([item["ticker"] for item in scorecard["resolved_symbols"]], ["0050.TW", "00631L.TW", "2330.TW"])
        self.assertEqual(scorecard["resolved_symbols"][2]["display"], "台積電(2330)")

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

    def test_normalize_manual_ticker_input(self) -> None:
        self.assertEqual(normalize_ticker("台積電(2330)"), "2330.TW")
        self.assertEqual(normalize_ticker("00631L"), "00631L.TW")
        self.assertEqual(parse_symbol_lines("2330\n\n2454")[0]["display"], "台積電(2330)")


if __name__ == "__main__":
    unittest.main()
