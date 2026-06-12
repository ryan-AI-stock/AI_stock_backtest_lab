from __future__ import annotations

import unittest
from types import SimpleNamespace

import pandas as pd

from backtest_lab.institutional_flow_overlay_shadow import (
    ChipFlowOverlayRule,
    InstitutionalOverlayRule,
    apply_institutional_overlay,
    chip_flow_risk_flag,
    institutional_risk_flag,
    price_confirmation_flag,
    previous_flow_date,
)
from backtest_lab.portfolio import Trade
from backtest_lab.simulation import BacktestResult


class InstitutionalFlowOverlayShadowTest(unittest.TestCase):
    def test_institutional_risk_flag_uses_prior_sell_streaks(self) -> None:
        rule = InstitutionalOverlayRule(
            name="test",
            foreign_sell_days=3,
            trust_sell_days=2,
            exposure_cap=0.5,
        )
        row = SimpleNamespace(
            foreign_consecutive_sell_days=3,
            trust_consecutive_sell_days=0,
            foreign_net_buy_shares=-1000,
            investment_trust_net_buy_shares=0,
            dealer_net_buy_shares=100,
        )

        flagged, reason = institutional_risk_flag(row, ticker="2454.TW", rule=rule)

        self.assertTrue(flagged)
        self.assertIn("foreign_sell_3d", reason)
        self.assertIn("total_net_sell_shares", reason)

    def test_previous_flow_date_never_uses_same_day_data(self) -> None:
        flow_dates = [pd.Timestamp("2023-01-02"), pd.Timestamp("2023-01-03")]

        self.assertEqual(previous_flow_date(flow_dates, pd.Timestamp("2023-01-03")), pd.Timestamp("2023-01-02"))

    def test_apply_overlay_caps_baseline_return_on_risk_day(self) -> None:
        baseline_curve = pd.DataFrame(
            {
                "total_value": [110.0, 121.0],
                "current_ticker": ["2454.TW", "2454.TW"],
                "current_exposure": [1.0, 1.0],
                "regime": ["strong_bull", "strong_bull"],
                "mode": ["daily_strength", "daily_strength"],
            },
            index=pd.to_datetime(["2023-01-03", "2023-01-04"]),
        )
        baseline = BacktestResult(
            name="baseline",
            final_value=121.0,
            total_return=0.21,
            max_drawdown=0.0,
            trades=[Trade("2023-01-03", "2454.TW", "buy", 1, 100.0, 100.0, 0, 0.0, "test")],
            equity_curve=baseline_curve,
        )
        flow_frame = pd.DataFrame(
            {
                "date": pd.to_datetime(["2023-01-02", "2023-01-03"]),
                "ticker": ["2454.TW", "2454.TW"],
                "foreign_net_buy_shares": [-1000, 0],
                "investment_trust_net_buy_shares": [0, 0],
                "dealer_net_buy_shares": [0, 0],
                "foreign_consecutive_sell_days": [3, 0],
                "trust_consecutive_sell_days": [0, 0],
            }
        )
        rule = InstitutionalOverlayRule(
            name="test",
            foreign_sell_days=3,
            trust_sell_days=2,
            exposure_cap=0.5,
        )

        overlay = apply_institutional_overlay(
            baseline=baseline,
            flow_frame=flow_frame,
            rule=rule,
            initial_cash=100.0,
        )

        self.assertEqual(round(float(overlay["total_value"].iloc[0]), 4), 105.0)
        self.assertTrue(bool(overlay["risk_flag"].iloc[0]))

    def test_apply_overlay_matches_baseline_when_no_risk_flag(self) -> None:
        baseline_curve = pd.DataFrame(
            {
                "total_value": [100.0, 95.0, 95.0],
                "current_ticker": ["2454.TW", "cash", "cash"],
                "current_exposure": [1.0, 0.0, 0.0],
                "regime": ["strong_bull", "strong_bull", "strong_bull"],
                "mode": ["daily_strength", "cash", "cash"],
            },
            index=pd.to_datetime(["2023-01-03", "2023-01-04", "2023-01-05"]),
        )
        baseline = BacktestResult(
            name="baseline",
            final_value=95.0,
            total_return=-0.05,
            max_drawdown=-0.05,
            trades=[],
            equity_curve=baseline_curve,
        )
        flow_frame = pd.DataFrame(
            {
                "date": pd.to_datetime(["2023-01-02", "2023-01-03", "2023-01-04"]),
                "ticker": ["2454.TW", "2454.TW", "2454.TW"],
                "foreign_net_buy_shares": [1000, 1000, 1000],
                "investment_trust_net_buy_shares": [0, 0, 0],
                "dealer_net_buy_shares": [0, 0, 0],
                "foreign_consecutive_sell_days": [0, 0, 0],
                "trust_consecutive_sell_days": [0, 0, 0],
            }
        )
        rule = InstitutionalOverlayRule(
            name="test",
            foreign_sell_days=3,
            trust_sell_days=2,
            exposure_cap=0.5,
        )

        overlay = apply_institutional_overlay(
            baseline=baseline,
            flow_frame=flow_frame,
            rule=rule,
            initial_cash=100.0,
        )

        self.assertEqual(list(overlay["total_value"].round(4)), [100.0, 95.0, 95.0])

    def test_chip_flow_risk_flag_can_require_two_signals(self) -> None:
        rule = ChipFlowOverlayRule(
            name="test",
            exposure_cap=0.5,
            institutional_foreign_sell_days=3,
            institutional_trust_sell_days=2,
            use_margin_overheat=True,
            use_day_trading_overheat=True,
            require_two_signals=True,
        )
        institutional_row = SimpleNamespace(
            foreign_consecutive_sell_days=3,
            trust_consecutive_sell_days=0,
            foreign_net_buy_shares=-1000,
            investment_trust_net_buy_shares=0,
            dealer_net_buy_shares=0,
        )
        margin_row = SimpleNamespace(margin_overheat_flag="false", short_lending_pressure_flag="false")
        day_row = SimpleNamespace(day_trading_volume_ratio=12.0, day_trading_overheat_flag="true")

        flagged, reason = chip_flow_risk_flag(
            institutional_row,
            margin_row,
            day_row,
            ticker="2454.TW",
            rule=rule,
        )

        self.assertTrue(flagged)
        self.assertIn("institutional:", reason)
        self.assertIn("day_trading_overheat", reason)

    def test_chip_flow_risk_flag_ignores_etf_when_stock_only(self) -> None:
        rule = ChipFlowOverlayRule(
            name="test",
            exposure_cap=0.5,
            use_day_trading_overheat=True,
            stock_only=True,
        )
        day_row = SimpleNamespace(day_trading_volume_ratio=80.0, day_trading_overheat_flag="true")

        flagged, reason = chip_flow_risk_flag(None, None, day_row, ticker="00631L.TW", rule=rule)

        self.assertFalse(flagged)
        self.assertEqual(reason, "")

    def test_price_confirmation_requires_weak_price(self) -> None:
        prices = pd.DataFrame(
            {"close": [100, 102, 104, 106, 108, 110, 112, 114, 116, 118]},
            index=pd.date_range("2023-01-01", periods=10),
        )

        confirmed, reason = price_confirmation_flag(
            prices,
            pd.Timestamp("2023-01-10"),
            "below_ma10",
        )

        self.assertFalse(confirmed)
        self.assertEqual(reason, "")

    def test_price_confirmation_flags_break_below_ma10(self) -> None:
        prices = pd.DataFrame(
            {"close": [100, 102, 104, 106, 108, 110, 112, 114, 116, 90]},
            index=pd.date_range("2023-01-01", periods=10),
        )

        confirmed, reason = price_confirmation_flag(
            prices,
            pd.Timestamp("2023-01-10"),
            "below_ma10",
        )

        self.assertTrue(confirmed)
        self.assertEqual(reason, "price_below_ma10")
