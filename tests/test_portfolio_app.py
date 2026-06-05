from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import test_paths  # noqa: F401

from backtest_lab.costs import TaiwanCostModel
from backtest_lab.portfolio_app import (
    PORTFOLIO_SECRET_NAME,
    PortfolioStore,
    _max_affordable_shares,
    build_dashboard,
    sync_portfolio_secret,
    trigger_report_workflow,
)


class _Completed:
    returncode = 0
    stdout = "https://github.com/ryan-AI-stock/AI_stock_backtest_lab/actions/runs/1\n"
    stderr = ""


class PortfolioAppTest(unittest.TestCase):
    def setUp(self) -> None:
        self.cost_model = TaiwanCostModel()
        self.asset_types = {"0050.TW": "etf", "2454.TW": "stock"}

    def test_manual_trade_updates_cash_cost_and_position(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = PortfolioStore(Path(tmp) / "store.json")
            store.replace_portfolio(user_id="default", cash_twd=100_000, positions=[])

            user = store.record_trade(
                user_id="default",
                trade={"date": "2026-06-04", "ticker": "2454.TW", "side": "buy", "shares": 10, "price": 1000},
                asset_types=self.asset_types,
                cost_model=self.cost_model,
            )

            self.assertEqual(user["positions"]["2454.TW"]["shares"], 10)
            self.assertLess(user["cash_twd"], 90_000)
            self.assertEqual(len(user["trades"]), 1)

    def test_dashboard_suggests_odd_lot_shares_from_target_exposure(self) -> None:
        user = {
            "user_id": "default",
            "display_name": "主要使用者",
            "cash_twd": 100_000,
            "positions": {},
            "trades": [],
        }
        signal = {
            "signal_date": "2026-06-04",
            "target_ticker": "2454.TW",
            "target_exposure": 1.0,
            "close_prices": {"2454.TW": 1000.0},
        }

        dashboard = build_dashboard(user, signal, self.asset_types, self.cost_model)

        self.assertEqual(dashboard["recommendations"][0]["action"], "buy")
        self.assertGreater(dashboard["recommendations"][0]["shares"], 0)
        self.assertLessEqual(dashboard["recommendations"][0]["shares"], 100)

    def test_portfolio_summary_keeps_reference_price_precision(self) -> None:
        user = {
            "user_id": "default",
            "display_name": "主要使用者",
            "cash_twd": 5000,
            "positions": {"2454.TW": {"shares": 10, "avg_cost": 100.1234}},
            "trades": [],
        }
        signal = {
            "signal_date": "2026-06-04",
            "target_ticker": "2454.TW",
            "target_exposure": 1.0,
            "close_prices": {"2454.TW": 100.125},
        }

        dashboard = build_dashboard(user, signal, self.asset_types, self.cost_model)
        position = dashboard["portfolio"]["positions"][0]

        self.assertEqual(position["reference_price"], 100.125)
        self.assertEqual(position["market_value_twd"], 1001.25)
        self.assertEqual(position["unrealized_pnl_twd"], 0.02)

    def test_affordable_shares_include_broker_fee(self) -> None:
        shares = _max_affordable_shares(10_000, 1000, self.cost_model)

        self.assertEqual(shares, 9)

    def test_sync_portfolio_secret_uses_stdin_without_printing_json(self) -> None:
        calls = []

        def runner(args, **kwargs):
            calls.append((args, kwargs))
            return _Completed()

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "portfolio_store.json"
            path.write_text('{"positions":{"2454.TW":{"shares":291}}}', encoding="utf-8")

            result = sync_portfolio_secret(store_path=path, repo="owner/repo", runner=runner)

        args, kwargs = calls[0]
        self.assertEqual(result["secret_name"], PORTFOLIO_SECRET_NAME)
        self.assertEqual(args[:4], ["gh", "secret", "set", PORTFOLIO_SECRET_NAME])
        self.assertNotIn("--body", args)
        self.assertNotIn("2454.TW", " ".join(args))
        self.assertIn("2454.TW", kwargs["input"])
        self.assertTrue(kwargs["capture_output"])

    def test_trigger_report_workflow_uses_signal_date_input(self) -> None:
        calls = []

        def runner(args, **kwargs):
            calls.append(args)
            return _Completed()

        result = trigger_report_workflow(signal_date="2026-06-05", repo="owner/repo", runner=runner)

        self.assertEqual(result["signal_date"], "2026-06-05")
        self.assertIn("signal_date=2026-06-05", calls[0])


if __name__ == "__main__":
    unittest.main()
