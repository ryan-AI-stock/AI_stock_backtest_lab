from __future__ import annotations

import http.client
import json
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path

import test_paths  # noqa: F401

from backtest_lab.costs import TaiwanCostModel
from backtest_lab.portfolio_app import PortfolioStore, create_handler


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
                    signal_root=str(tmp_path / "signals"),
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


def _request(port: int, method: str, path: str, payload: dict | None = None) -> dict:
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = {"Content-Type": "application/json"} if body else {}
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    try:
        connection.request(method, path, body=body, headers=headers)
        response = connection.getresponse()
        data = json.loads(response.read().decode("utf-8"))
        if response.status >= 400:
            raise AssertionError(data)
        return data
    finally:
        connection.close()


if __name__ == "__main__":
    unittest.main()
