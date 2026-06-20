from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

import test_paths  # noqa: F401

from backtest_lab.chip_formal_challenger_runner import (
    ChipFormalVariant,
    build_chip_target_selection_overlay,
    build_decision_diff_panel,
    load_chip_diagnostic_panel,
)
from backtest_lab.costs import TaiwanCostModel
from backtest_lab.regime_mode_switch import (
    MODE_DAILY,
    RegimeModeSwitchVariant,
    TargetSelectionOverlayDecision,
    simulate_regime_mode_switch,
)


class ChipFormalChallengerRunnerTest(unittest.TestCase):
    def test_engine_target_selection_noop_aligns_with_baseline(self) -> None:
        prices = _price_frames()
        variant = _simple_variant()
        baseline = simulate_regime_mode_switch(
            name="baseline",
            prices_by_ticker=prices,
            asset_types={ticker: "stock" for ticker in prices},
            market_prices=prices["0050.TW"],
            start_date="2024-04-01",
            end_date="2024-04-12",
            initial_cash=1_000_000,
            cost_model=TaiwanCostModel(),
            variant=variant,
        )

        def noop(*args):
            baseline_target = args[6]
            signal_date = args[3]
            return TargetSelectionOverlayDecision(
                target=baseline_target,
                reason="noop",
                signal_date=pd.Timestamp(signal_date).strftime("%Y-%m-%d"),
            )

        challenger = simulate_regime_mode_switch(
            name="noop",
            prices_by_ticker=prices,
            asset_types={ticker: "stock" for ticker in prices},
            market_prices=prices["0050.TW"],
            start_date="2024-04-01",
            end_date="2024-04-12",
            initial_cash=1_000_000,
            cost_model=TaiwanCostModel(),
            variant=variant,
            target_selection_overlay=noop,
        )

        self.assertAlmostEqual(baseline.final_value, challenger.final_value)
        self.assertTrue(
            baseline.equity_curve["current_ticker"].astype(str).equals(
                challenger.equity_curve["current_ticker"].astype(str)
            )
        )
        self.assertIn("target_overlay_changed", challenger.equity_curve.columns)

    def test_loader_rejects_h3_or_valuation_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "chip.csv"
            frame = _chip_panel()
            frame["day_ratio_top10"] = True
            frame.to_csv(path, index=False)

            with self.assertRaisesRegex(ValueError, "Forbidden H3/valuation"):
                load_chip_diagnostic_panel(path)

    def test_h1_strict_confirmation_replaces_unconfirmed_target_with_confirmed_candidate(self) -> None:
        prices = _price_frames()
        signal_date = prices["0050.TW"].index[-2]
        overlay = build_chip_target_selection_overlay(
            _chip_panel(signal_date=signal_date, h1_2454=0, h1_2330=2),
            ChipFormalVariant(
                "h1_strict_confirmation",
                "H1 strict confirmation",
                require_h1_confirmation=True,
            ),
        )

        decision = overlay(
            MODE_DAILY,
            prices,
            prices["0050.TW"].index[-1],
            signal_date,
            "strong_bull",
            _simple_variant(),
            "2454.TW",
            1.0,
        )

        self.assertEqual(decision.target, "2330.TW")
        self.assertIn("h1_strict_confirmation_changed", decision.reason)

    def test_h2_veto_replaces_sell_pressure_target_with_clean_candidate(self) -> None:
        prices = _price_frames()
        signal_date = prices["0050.TW"].index[-2]
        overlay = build_chip_target_selection_overlay(
            _chip_panel(signal_date=signal_date, h1_2454=1, h2_2454=3, h1_2330=1, h2_2330=0),
            ChipFormalVariant("h2_risk_veto", "H2 risk veto", h2_score_penalty=0.20, veto_h2_pressure=True),
        )

        decision = overlay(
            MODE_DAILY,
            prices,
            prices["0050.TW"].index[-1],
            signal_date,
            "strong_bull",
            _simple_variant(),
            "2454.TW",
            1.0,
        )

        self.assertEqual(decision.target, "2330.TW")
        self.assertIn("h2_risk_veto_changed", decision.reason)

    def test_decision_diff_keeps_guardrail_flags_and_chip_scores(self) -> None:
        dates = pd.bdate_range("2024-01-01", periods=3)
        baseline = pd.DataFrame(
            {
                "total_value": [100.0, 101.0, 102.0],
                "current_ticker": ["2454.TW", "2454.TW", "2454.TW"],
                "current_exposure": [1.0, 1.0, 1.0],
            },
            index=dates,
        )
        challenger = pd.DataFrame(
            {
                "total_value": [100.0, 100.5, 101.0],
                "current_ticker": ["2454.TW", "2330.TW", "2330.TW"],
                "current_exposure": [1.0, 1.0, 1.0],
                "target_overlay_baseline_target": ["2454.TW", "2454.TW", "2454.TW"],
                "target_overlay_target": ["2454.TW", "2330.TW", "2330.TW"],
                "target_overlay_reason": ["same", "changed", "changed"],
                "target_overlay_signal_date": [d.strftime("%Y-%m-%d") for d in dates],
                "target_overlay_changed": [False, True, True],
            },
            index=dates,
        )
        panel = _chip_panel(signal_date=dates[1], h1_2330=2, h2_2330=0)

        diff = build_decision_diff_panel(
            period_id="unit",
            variant_id="h1_strict_confirmation",
            variant_label="H1 strict confirmation",
            baseline_curve=baseline,
            challenger_curve=challenger,
            chip_panel=panel,
        )

        changed = diff.loc[diff["date"] == dates[1].strftime("%Y-%m-%d")].iloc[0]
        self.assertTrue(bool(changed["changed_formal_candidate"]))
        self.assertEqual(int(changed["attack_confirmation_score"]), 2)
        self.assertFalse(bool(changed["valuation_used"]))
        self.assertFalse(bool(changed["h3_used"]))


def _simple_variant() -> RegimeModeSwitchVariant:
    modes = {
        "strong_bull": MODE_DAILY,
        "recovery_bull": MODE_DAILY,
        "range_bound": MODE_DAILY,
        "correction_bear": MODE_DAILY,
        "systemic_bear": MODE_DAILY,
    }
    return RegimeModeSwitchVariant(
        name="unit_daily",
        regime_modes=modes,
        regime_exposures={key: 1.0 for key in modes},
        relative_score_fallback_ticker="0050.TW",
        min_score_over_fallback=0.0,
    )


def _price_frames() -> dict[str, pd.DataFrame]:
    dates = pd.bdate_range("2023-01-02", periods=380)
    return {
        "0050.TW": _price_frame(dates, 100, 112),
        "2454.TW": _price_frame(dates, 100, 180),
        "2330.TW": _price_frame(dates, 100, 170),
    }


def _price_frame(dates: pd.DatetimeIndex, start: float, end: float) -> pd.DataFrame:
    values = [start + ((end - start) * index / (len(dates) - 1)) for index in range(len(dates))]
    return pd.DataFrame(
        {
            "open": values,
            "high": [value * 1.01 for value in values],
            "low": [value * 0.99 for value in values],
            "close": values,
            "adj_close": values,
            "volume": [1000] * len(values),
            "dividends": [0.0] * len(values),
        },
        index=dates,
    )


def _chip_panel(
    *,
    signal_date: pd.Timestamp | None = None,
    h1_2454: int = 1,
    h2_2454: int = 0,
    h1_2330: int = 1,
    h2_2330: int = 0,
) -> pd.DataFrame:
    date = pd.Timestamp(signal_date or "2024-04-18").normalize()
    return pd.DataFrame(
        [
            {
                "date": date,
                "ticker": "2454.TW",
                "attack_confirmation_score": h1_2454,
                "sell_pressure_warning_score": h2_2454,
                "h1_negative_or_h2_sell_pressure": h2_2454 > 0,
            },
            {
                "date": date,
                "ticker": "2330.TW",
                "attack_confirmation_score": h1_2330,
                "sell_pressure_warning_score": h2_2330,
                "h1_negative_or_h2_sell_pressure": h2_2330 > 0,
            },
        ]
    )


if __name__ == "__main__":
    unittest.main()
