from __future__ import annotations

import unittest

import pandas as pd

import test_paths  # noqa: F401

from backtest_lab.future_path_scenario_simulator import (
    ScenarioInputs,
    _build_normalized_paths,
    _find_analog_cases,
    _summarize_paths,
)


class FuturePathScenarioSimulatorTest(unittest.TestCase):
    def test_find_analog_cases_prefers_strict_matches(self) -> None:
        dates = pd.bdate_range("2024-01-02", periods=180)
        equity = pd.DataFrame(
            {
                "total_value": [1_000_000 + index * 1000 for index in range(len(dates))],
                "current_ticker": ["00631L.TW"] * len(dates),
                "regime": ["strong_bull"] * len(dates),
                "mode": ["0050_defense"] * len(dates),
                "attack_gate_active": [False] * len(dates),
                "attack_gate_ever_activated": [True] * len(dates),
            },
            index=dates,
        )
        prices = {
            "00631L.TW": _price_frame(dates, 100, 0.1),
            "2454.TW": _price_frame(dates, 100, 0.3),
            "0050.TW": _price_frame(dates, 100, 0.05),
        }
        current = ScenarioInputs(
            signal_date=dates[-1].strftime("%Y-%m-%d"),
            scenario_start="2026-06-15",
            scenario_end="2026-12-31",
            history_start="2024-01-02",
            replay_start="2020-01-02",
            initial_capital=1_328_709,
            horizon_days=20,
            target_ticker="00631L.TW",
            target_label="0050正二",
            current_regime="strong_bull",
            current_mode="0050_defense",
            current_attack_gate_active=False,
            current_top_stock="2454.TW",
            current_top_stock_label="聯發科",
            current_top_stock_score=0.9,
            current_target_score=0.3,
        )

        analogs = _find_analog_cases(
            equity=equity,
            prices=prices,
            labels={"00631L.TW": "0050正二", "2454.TW": "聯發科", "0050.TW": "0050"},
            current=current,
            history_start="2024-01-02",
            horizon_days=20,
            min_analogs=8,
        )

        self.assertFalse(analogs.empty)
        self.assertTrue((analogs["match_tier"] == "strict").all())

    def test_normalized_paths_and_summary_use_initial_capital(self) -> None:
        dates = pd.bdate_range("2024-01-02", periods=5)
        equity = pd.DataFrame(
            {
                "total_value": [100, 110, 105, 120, 130],
                "current_ticker": ["00631L.TW"] * 5,
                "regime": ["strong_bull"] * 5,
            },
            index=dates,
        )
        analogs = pd.DataFrame(
            [
                {
                    "analog_start": dates[0].strftime("%Y-%m-%d"),
                    "match_tier": "strict",
                    "similarity_score": 1.0,
                }
            ]
        )
        paths = _build_normalized_paths(equity=equity, analogs=analogs, initial_capital=1000, horizon_days=4)
        current = ScenarioInputs(
            signal_date="2026-06-12",
            scenario_start="2026-06-15",
            scenario_end="2026-12-31",
            history_start="2024-01-02",
            replay_start="2020-01-02",
            initial_capital=1000,
            horizon_days=4,
            target_ticker="00631L.TW",
            target_label="0050正二",
            current_regime="strong_bull",
            current_mode="0050_defense",
            current_attack_gate_active=False,
            current_top_stock="2454.TW",
            current_top_stock_label="聯發科",
            current_top_stock_score=0.9,
            current_target_score=0.3,
        )

        summary = _summarize_paths(paths, current)

        self.assertEqual(float(paths.iloc[0]["projected_value"]), 1000)
        self.assertEqual(float(paths.iloc[-1]["projected_value"]), 1300)
        self.assertIn("median_p50", set(summary["scenario"]))


def _price_frame(dates: pd.DatetimeIndex, start: float, step: float) -> pd.DataFrame:
    closes = [start + index * step for index in range(len(dates))]
    return pd.DataFrame(
        {
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "adj_close": closes,
            "volume": [1_000_000] * len(dates),
        },
        index=dates,
    )


if __name__ == "__main__":
    unittest.main()
