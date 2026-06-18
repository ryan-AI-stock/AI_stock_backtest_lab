from __future__ import annotations

import unittest

import pandas as pd

import test_paths  # noqa: F401

from backtest_lab.formal_overlay_challenger import (
    build_chip_flow_exposure_overlay,
    selected_formal_challenger_rules,
)


class FormalOverlayChallengerTest(unittest.TestCase):
    def test_selected_rules_are_only_dd10_promotion_candidates(self) -> None:
        names = {rule.name for rule in selected_formal_challenger_rules()}

        self.assertEqual(
            names,
            {
                "chip_two_signal_price_dd10_cash",
                "chip_two_signal_price_dd10_reduce25",
                "chip_two_signal_price_dd10_reduce50",
                "chip_two_signal_price_dd10_reduce75",
            },
        )

    def test_chip_flow_overlay_builder_caps_exposure_when_two_signals_and_price_confirm(self) -> None:
        rule = next(rule for rule in selected_formal_challenger_rules() if rule.name.endswith("reduce50"))
        dates = pd.bdate_range("2022-01-03", periods=12)
        prices = pd.DataFrame(
            {
                "open": [120.0] * 9 + [112.0, 105.0, 100.0],
                "close": [120.0] * 9 + [112.0, 105.0, 100.0],
                "adj_close": [120.0] * 9 + [112.0, 105.0, 100.0],
            },
            index=dates,
        )
        signal_date = dates[-2]
        trade_date = dates[-1]
        institutional = pd.DataFrame(
            [
                {
                    "date": signal_date,
                    "ticker": "2454.TW",
                    "foreign_consecutive_sell_days": 3,
                    "trust_consecutive_sell_days": 0,
                    "foreign_net_buy_shares": -1000,
                    "investment_trust_net_buy_shares": 0,
                    "dealer_net_buy_shares": 0,
                }
            ]
        )
        margin = pd.DataFrame(
            [
                {
                    "date": signal_date,
                    "ticker": "2454.TW",
                    "margin_overheat_flag": True,
                    "short_lending_pressure_flag": False,
                }
            ]
        )
        day_trading = pd.DataFrame(
            [
                {
                    "date": signal_date,
                    "ticker": "2454.TW",
                    "day_trading_overheat_flag": False,
                    "day_trading_volume_ratio": 0.0,
                }
            ]
        )

        overlay = build_chip_flow_exposure_overlay(
            institutional_frame=institutional,
            margin_frame=margin,
            day_trading_frame=day_trading,
            prices_by_ticker={"2454.TW": prices},
            rule=rule,
        )
        decision = overlay("2454.TW", trade_date, signal_date, 1.0)

        self.assertTrue(decision.risk_flag)
        self.assertEqual(decision.adjusted_exposure, 0.5)
        self.assertIn("price_dd10_over8", decision.reason)


if __name__ == "__main__":
    unittest.main()
