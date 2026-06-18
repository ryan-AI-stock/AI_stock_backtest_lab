from __future__ import annotations

import unittest

import pandas as pd

import test_paths  # noqa: F401

from backtest_lab.costs import TaiwanCostModel
from backtest_lab.regime_mode_switch import (
    ExposureOverlayDecision,
    MODE_DAILY,
    RegimeModeSwitchVariant,
    simulate_regime_mode_switch,
)


class RegimeModeSwitchOverlayTest(unittest.TestCase):
    def test_exposure_overlay_is_optional_and_preserves_baseline_columns(self) -> None:
        prices = _synthetic_prices()
        variant = _daily_variant()

        baseline = simulate_regime_mode_switch(
            name="baseline",
            prices_by_ticker=prices,
            asset_types={ticker: "stock" for ticker in prices},
            market_prices=prices["0050.TW"],
            start_date="2022-01-03",
            end_date="2022-01-07",
            initial_cash=1_000_000,
            cost_model=TaiwanCostModel(),
            variant=variant,
        )

        self.assertNotIn("overlay_risk_flag", baseline.equity_curve.columns)
        self.assertGreater(float(baseline.equity_curve["current_exposure"].max()), 0.9)

    def test_exposure_overlay_caps_position_through_formal_rebalance(self) -> None:
        prices = _synthetic_prices()
        variant = _daily_variant()

        def overlay(
            ticker: str | None,
            trade_date: pd.Timestamp,
            signal_date: pd.Timestamp,
            proposed_exposure: float,
        ) -> ExposureOverlayDecision:
            if ticker == "2454.TW":
                return ExposureOverlayDecision(
                    adjusted_exposure=0.50,
                    risk_flag=True,
                    reason="unit_test_cap",
                    signal_date=signal_date.strftime("%Y-%m-%d"),
                )
            return ExposureOverlayDecision(adjusted_exposure=proposed_exposure)

        result = simulate_regime_mode_switch(
            name="overlay",
            prices_by_ticker=prices,
            asset_types={ticker: "stock" for ticker in prices},
            market_prices=prices["0050.TW"],
            start_date="2022-01-03",
            end_date="2022-01-07",
            initial_cash=1_000_000,
            cost_model=TaiwanCostModel(),
            variant=variant,
            exposure_overlay=overlay,
        )

        self.assertIn("overlay_risk_flag", result.equity_curve.columns)
        self.assertTrue(result.equity_curve["overlay_risk_flag"].any())
        self.assertLess(float(result.equity_curve["current_exposure"].max()), 0.56)
        self.assertTrue(any("overlay_unit_test_cap" in trade.reason for trade in result.trades))


def _daily_variant() -> RegimeModeSwitchVariant:
    return RegimeModeSwitchVariant(
        name="unit_daily",
        regime_modes={
            "strong_bull": MODE_DAILY,
            "recovery_bull": MODE_DAILY,
            "range_bound": MODE_DAILY,
            "correction_bear": MODE_DAILY,
            "systemic_bear": MODE_DAILY,
        },
        regime_exposures={
            "strong_bull": 1.0,
            "recovery_bull": 1.0,
            "range_bound": 1.0,
            "correction_bear": 1.0,
            "systemic_bear": 1.0,
        },
    )


def _synthetic_prices() -> dict[str, pd.DataFrame]:
    dates = pd.bdate_range("2021-01-01", periods=270)
    return {
        "2454.TW": _price_frame(dates, 100.0, 0.90),
        "2330.TW": _price_frame(dates, 100.0, 0.25),
        "0050.TW": _price_frame(dates, 100.0, 0.15),
    }


def _price_frame(dates: pd.DatetimeIndex, start: float, step: float) -> pd.DataFrame:
    values = [start + (index * step) for index in range(len(dates))]
    return pd.DataFrame(
        {
            "open": values,
            "close": values,
            "adj_close": values,
        },
        index=dates,
    )


if __name__ == "__main__":
    unittest.main()
