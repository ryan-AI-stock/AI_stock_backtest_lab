from __future__ import annotations

import http.client
import json
import tempfile
import threading
import time
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path

import test_paths  # noqa: F401

from backtest_lab.costs import TaiwanCostModel
from backtest_lab.candidate_review_decision_store import CandidateReviewDecisionStore
from backtest_lab.portfolio_app import PortfolioStore, create_handler, load_latest_observation_state
from backtest_lab.stock_pool_store import StockPoolStore


class _Completed:
    returncode = 0
    stdout = "https://github.com/ryan-AI-stock/AI_stock_backtest_lab/actions/runs/1\n"
    stderr = ""


class PortfolioAppHttpTest(unittest.TestCase):
    def test_state_and_portfolio_update_over_http(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            signal_dir = tmp_path / "signals" / "20260604"
            signal_dir.mkdir(parents=True)
            (signal_dir / "frozen_strategy_signal.json").write_text(
                json.dumps(
                    {
                        "status": "ready",
                        "signal": {
                            "signal_date": "2026-06-04",
                            "target_ticker": "2454.TW",
                            "target_exposure": 1.0,
                            "close_prices": {"2454.TW": 1000.0},
                        },
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            server = ThreadingHTTPServer(
                ("127.0.0.1", 0),
                create_handler(
                    store=PortfolioStore(tmp_path / "store.json"),
                    pool_store=StockPoolStore(tmp_path / "stock_pools.json"),
                    candidate_decision_store=CandidateReviewDecisionStore(tmp_path / "candidate_review_decisions.json"),
                    candidate_review_backup_root=tmp_path / "candidate_review_backups",
                    signal_root=str(tmp_path / "signals"),
                    observation_root=str(tmp_path / "observations"),
                    asset_types={"2454.TW": "stock"},
                    cost_model=TaiwanCostModel(),
                    command_runner=lambda *args, **kwargs: _Completed(),
                ),
            )
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                port = server.server_address[1]
                response = _request(port, "GET", "/api/state")
                self.assertEqual(response["signal"]["target_ticker"], "2454.TW")

                updated = _request(
                    port,
                    "POST",
                    "/api/portfolio",
                    {
                        "cash_twd": 100000,
                        "positions": [{"ticker": "2454.TW", "shares": 10, "avg_cost": 900}],
                    },
                )
                self.assertEqual(updated["portfolio"]["positions"][0]["shares"], 10)

                synced = _request(port, "POST", "/api/sync-secret-and-run", {})
                self.assertTrue(synced["sync_result"]["ok"])
                self.assertTrue(synced["action_result"]["ok"])
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

    def test_stock_pool_api_lists_defaults_and_accepts_custom_pool(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            signal_dir = tmp_path / "signals" / "20260604"
            signal_dir.mkdir(parents=True)
            (signal_dir / "frozen_strategy_signal.json").write_text(
                json.dumps({"status": "ready", "signal": {"signal_date": "2026-06-04", "target_ticker": "2330.TW"}}),
                encoding="utf-8",
            )
            server = ThreadingHTTPServer(
                ("127.0.0.1", 0),
                create_handler(
                    store=PortfolioStore(tmp_path / "store.json"),
                    pool_store=StockPoolStore(tmp_path / "stock_pools.json"),
                    signal_root=str(tmp_path / "signals"),
                    observation_root=str(tmp_path / "observations"),
                    asset_types={"2454.TW": "stock"},
                    cost_model=TaiwanCostModel(),
                    command_runner=lambda *args, **kwargs: _Completed(),
                ),
            )
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                port = server.server_address[1]
                state = _request(port, "GET", "/api/pools")
                self.assertEqual(state["latest_signal"]["target_ticker"], "2330.TW")
                self.assertEqual(
                    {pool["pool_id"] for pool in state["pool_sections"]["official_core"]},
                    {"ai_theme_large_cap_v20260613", "tw50_dynamic_constituents_v0", "large_core_bluechip_v0"},
                )
                scorecard = next(pool for pool in state["pools"] if pool["pool_id"] == "model_scorecard_ep10")
                self.assertEqual(scorecard["resolved_symbols"][2]["ticker"], "2330.TW")

                updated = _request(
                    port,
                    "POST",
                    "/api/pools",
                    {"name": "自訂觀察池", "symbols_text": "2330\n2454", "strategy_preset": "universal_pool_custom"},
                )
                self.assertIn("自訂觀察池", {pool["name"] for pool in updated["pools"]})
                self.assertIn("自訂觀察池", {pool["name"] for pool in updated["pool_sections"]["experiment"]})

                synced = _request(port, "POST", "/api/sync-pools-secret-and-run", {"signal_date": "2026-06-04"})
                self.assertTrue(synced["sync_result"]["ok"])
                self.assertEqual(synced["sync_result"]["secret_name"], "STOCK_POOLS_JSON")
                self.assertEqual(synced["action_result"]["workflow_file"], "stock_pool_observation.yml")
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

    def test_candidate_review_api_lists_official_monthly_sources(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            signal_dir = tmp_path / "signals" / "20260612"
            signal_dir.mkdir(parents=True)
            (signal_dir / "frozen_strategy_signal.json").write_text(
                json.dumps({"status": "ready", "signal": {"signal_date": "2026-06-12", "target_ticker": "00631L.TW"}}),
                encoding="utf-8",
            )
            server = ThreadingHTTPServer(
                ("127.0.0.1", 0),
                create_handler(
                    store=PortfolioStore(tmp_path / "store.json"),
                    pool_store=StockPoolStore(tmp_path / "stock_pools.json"),
                    signal_root=str(tmp_path / "signals"),
                    observation_root=str(tmp_path / "observations"),
                    asset_types={"2454.TW": "stock"},
                    cost_model=TaiwanCostModel(),
                    command_runner=lambda *args, **kwargs: _Completed(),
                ),
            )
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                response = _request(server.server_address[1], "GET", "/api/candidate-reviews")
                self.assertEqual(response["status"], "ready")
                self.assertEqual(response["signal_date"], "2026-06-12")
                self.assertEqual(
                    {review["pool_id"] for review in response["reviews"]},
                    {"ai_theme_large_cap_v20260613", "tw50_dynamic_constituents_v0", "large_core_bluechip_v0"},
                )
                core_review = next(
                    review for review in response["reviews"] if review["pool_id"] == "large_core_bluechip_v0"
                )
                self.assertEqual(core_review["source_mode"], "core_defensive_candidate_csv")
                self.assertEqual(core_review["source_status"], "source_ready")
                self.assertEqual(core_review["source_active_count"], 16)
                self.assertEqual(core_review["source_watch_count"], 6)

                recorded = _request(
                    server.server_address[1],
                    "POST",
                    "/api/candidate-review-decisions",
                    {
                        "pool_id": "large_core_bluechip_v0",
                        "pool_name": "核心風格補強池 v1",
                        "ticker": "2412.TW",
                        "display": "中華電(2412)",
                        "decision": "keep_current",
                        "signal_date": "2026-06-12",
                        "note": "保留作為防守代表",
                    },
                )
                key = "large_core_bluechip_v0|2412.TW"
                self.assertEqual(recorded["recorded"]["decision_label"], "維持現有")
                self.assertEqual(recorded["latest_by_key"][key]["note"], "保留作為防守代表")

                decisions = _request(server.server_address[1], "GET", "/api/candidate-review-decisions")
                self.assertEqual(decisions["latest_by_key"][key]["decision"], "keep_current")

                draft = _request(server.server_address[1], "GET", "/api/candidate-review-decision-draft")
                self.assertEqual(draft["change_count"], 1)
                self.assertEqual(draft["changes"][0]["ticker"], "2412.TW")
                self.assertEqual(draft["changes"][0]["draft_status"], "active")

                applied = _request(server.server_address[1], "POST", "/api/candidate-review-decision-draft/apply", {})
                self.assertEqual(applied["status"], "applied")
                self.assertEqual(applied["applied_change_count"], 1)
                self.assertTrue(Path(applied["applied"][0]["backup_path"]).exists())
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

    def test_observation_api_reads_latest_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            root = tmp_path / "observations"
            old_dir = root / "20260526"
            new_dir = root / "20260605"
            old_dir.mkdir(parents=True)
            new_dir.mkdir(parents=True)
            (old_dir / "stock_pool_observation_manifest.json").write_text(
                json.dumps({"signal_date": "2026-05-26", "generated": [], "skipped": []}),
                encoding="utf-8",
            )
            (new_dir / "stock_pool_observation_manifest.json").write_text(
                json.dumps(
                    {
                        "signal_date": "2026-06-05",
                        "generated": [
                            {
                                "pool_id": "large_cap_best_v20260605",
                                "pool_name": "AI中大型權值股池最佳版 v20260605",
                                "top_display": "聯發科(2454)",
                                "action_state": "watch_candidate",
                            }
                        ],
                        "skipped": [],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            state = load_latest_observation_state(root)

            self.assertEqual(state["status"], "ready")
            self.assertEqual(state["manifest"]["signal_date"], "2026-06-05")

            server = ThreadingHTTPServer(
                ("127.0.0.1", 0),
                create_handler(
                    store=PortfolioStore(tmp_path / "store.json"),
                    pool_store=StockPoolStore(tmp_path / "stock_pools.json"),
                    signal_root=str(tmp_path / "signals"),
                    observation_root=str(root),
                    asset_types={"2454.TW": "stock"},
                    cost_model=TaiwanCostModel(),
                    command_runner=lambda *args, **kwargs: _Completed(),
                ),
            )
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                response = _request(server.server_address[1], "GET", "/api/observations")
                self.assertEqual(response["manifest"]["generated"][0]["top_display"], "聯發科(2454)")
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)


def _request(port: int, method: str, path: str, payload: dict | None = None) -> dict:
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = {"Content-Type": "application/json"} if body else {}
    last_error: Exception | None = None
    for attempt in range(3):
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        try:
            connection.request(method, path, body=body, headers=headers)
            response = connection.getresponse()
            data = json.loads(response.read().decode("utf-8"))
            if response.status >= 400:
                raise AssertionError(data)
            return data
        except (ConnectionError, TimeoutError, OSError) as error:
            last_error = error
            if attempt == 2:
                raise
            time.sleep(0.1)
        finally:
            connection.close()
    raise AssertionError(f"request failed without response: {last_error}")


if __name__ == "__main__":
    unittest.main()
