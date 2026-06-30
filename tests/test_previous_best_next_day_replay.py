from __future__ import annotations

import json
import unittest

import pandas as pd

import test_paths  # noqa: F401

from backtest_lab.previous_best_next_day_replay import (
    _build_daily_target_stream,
    _execution_frame_from_target_stream,
    _intended_policy_exposure,
)


class PreviousBestNextDayReplayTest(unittest.TestCase):
    def test_policy_exposure_snaps_raw_drift_without_daily_rebalance(self) -> None:
        self.assertEqual(_intended_policy_exposure("2454.TW", 0.974), 1.0)
        self.assertEqual(_intended_policy_exposure("00631L.TW", 0.249), 0.25)
        self.assertEqual(_intended_policy_exposure("cash", 0.0), 0.0)

    def test_target_stream_uses_policy_exposure_not_raw_position_drift(self) -> None:
        dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"])
        prices = {
            ticker: pd.DataFrame(
                {
                    "open": [100, 101, 102, 103],
                    "close": [100, 101, 102, 103],
                    "adj_close": [100, 101, 102, 103],
                },
                index=dates,
            )
            for ticker in ["0050.TW", "2454.TW", "6669.TW"]
        }
        equity = pd.DataFrame(
            [
                {
                    "total_value": 1_000_000,
                    "current_ticker": "2454.TW",
                    "current_exposure": 0.974,
                    "regime": "strong_bull",
                    "mode": "daily_strength",
                    "risk_off_active": False,
                    "attack_gate_active": True,
                    "attack_gate_ever_activated": True,
                },
                {
                    "total_value": 1_020_000,
                    "current_ticker": "2454.TW",
                    "current_exposure": 0.973,
                    "regime": "strong_bull",
                    "mode": "daily_strength",
                    "risk_off_active": False,
                    "attack_gate_active": True,
                    "attack_gate_ever_activated": True,
                },
                {
                    "total_value": 1_030_000,
                    "current_ticker": "6669.TW",
                    "current_exposure": 0.998,
                    "regime": "strong_bull",
                    "mode": "daily_strength",
                    "risk_off_active": False,
                    "attack_gate_active": True,
                    "attack_gate_ever_activated": True,
                },
            ],
            index=pd.to_datetime(["2024-01-03", "2024-01-04", "2024-01-05"]),
        )

        stream = _build_daily_target_stream(
            equity,
            prices,
            signal_start=pd.Timestamp("2024-01-02"),
            signal_end=pd.Timestamp("2024-01-04"),
        )

        self.assertEqual(stream["signal_date"].tolist(), ["2024-01-02", "2024-01-03", "2024-01-04"])
        self.assertEqual(stream["action"].tolist(), ["buy", "hold", "switch"])
        self.assertEqual(json.loads(stream.iloc[0]["target_weights"]), {"2454.TW": 1.0})
        self.assertEqual(json.loads(stream.iloc[1]["target_weights"]), {"2454.TW": 1.0})
        self.assertEqual(json.loads(stream.iloc[2]["target_weights"]), {"6669.TW": 1.0})
        self.assertAlmostEqual(float(stream.iloc[1]["raw_current_exposure_after_strategy_execution"]), 0.973)

    def test_execution_frame_appends_terminal_fill_row_without_new_signal(self) -> None:
        stream = pd.DataFrame(
            [
                {
                    "signal_date": "2024-01-02",
                    "date": "2024-01-02",
                    "target_weights": '{"2454.TW": 1.0}',
                    "action": "buy",
                    "turnover": 1.0,
                    "period": "2024",
                }
            ]
        )
        frame = _execution_frame_from_target_stream(stream, pd.Timestamp("2024-01-03"))
        self.assertEqual(frame["date"].tolist(), ["2024-01-02", "2024-01-03"])
        self.assertEqual(frame["action"].tolist(), ["buy", "hold"])
        self.assertEqual(frame["is_terminal_execution_row"].tolist(), [False, True])


if __name__ == "__main__":
    unittest.main()
