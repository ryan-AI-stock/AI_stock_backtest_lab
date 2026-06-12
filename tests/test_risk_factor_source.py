from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

import test_paths  # noqa: F401

from backtest_lab.risk_factor_source import load_first_available_risk_factors


class RiskFactorSourceTest(unittest.TestCase):
    def test_loads_latest_not_after_signal_date_and_combines_sources(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pd.DataFrame(
                [
                    {
                        "date": "2026-06-11",
                        "symbol": "2454",
                        "foreign_consecutive_sell_days": 2,
                        "trust_consecutive_sell_days": 0,
                        "foreign_net_buy_shares": 1000,
                    },
                    {
                        "date": "2026-06-12",
                        "symbol": "2454",
                        "foreign_consecutive_sell_days": 3,
                        "trust_consecutive_sell_days": 0,
                        "foreign_net_buy_shares": -1000,
                    },
                    {
                        "date": "2026-06-13",
                        "symbol": "2454",
                        "foreign_consecutive_sell_days": 9,
                        "trust_consecutive_sell_days": 9,
                    },
                ]
            ).to_csv(root / "institutional_flows.latest.csv", index=False, encoding="utf-8-sig")
            pd.DataFrame(
                [
                    {
                        "date": "2026-06-12",
                        "ticker": "2454.TW",
                        "margin_balance_5d_change_pct": 18.0,
                        "margin_overheat_flag": "true",
                    }
                ]
            ).to_csv(root / "margin_short.latest.csv", index=False, encoding="utf-8-sig")

            signals, sources = load_first_available_risk_factors(signal_date="2026-06-12", radar_data_dir=root)

        self.assertIn("institutional", sources)
        self.assertIn("margin_short", sources)
        signal = signals["2454.TW"]
        self.assertGreater(signal.institutional_risk, 0)
        self.assertGreater(signal.margin_risk, 0)
        self.assertGreater(signal.total_risk_score, signal.institutional_risk)
        self.assertIn("外資連賣3日", signal.reason_text)
        self.assertIn("融資短線升溫", signal.reason_text)
        self.assertEqual(signal.source_dates, ("2026-06-12",))

    def test_can_read_stock_metrics_for_day_trading_and_sentiment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pd.DataFrame(
                [
                    {
                        "report_date": "2026-06-12",
                        "symbol": "2408",
                        "day_trading_volume_ratio": 42.0,
                        "sentiment_score": 0.8,
                        "social_heat_score": 90.0,
                    }
                ]
            ).to_csv(root / "stock_metrics.refreshed.csv", index=False, encoding="utf-8-sig")

            signals, sources = load_first_available_risk_factors(signal_date="2026-06-12", radar_data_dir=root)

        self.assertIn("day_trading", sources)
        self.assertIn("sentiment", sources)
        signal = signals["2408.TW"]
        self.assertGreater(signal.day_trading_risk, 0)
        self.assertGreater(signal.sentiment_risk, 0)
        self.assertEqual(signal.sentiment_score, 0.8)


if __name__ == "__main__":
    unittest.main()
