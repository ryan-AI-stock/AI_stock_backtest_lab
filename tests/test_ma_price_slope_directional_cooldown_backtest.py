from __future__ import annotations

import unittest

from backtest_lab.ma_price_slope_directional_cooldown_backtest import (
    COOLDOWN_SCENARIOS,
    SIGNAL_PAIRS,
    CooldownScenario,
    experiment_matrix,
    opposite_action_allowed,
)


class MaPriceSlopeDirectionalCooldownBacktestTests(unittest.TestCase):
    def test_matrix_has_two_signal_pairs_and_ten_cooldown_scenarios(self) -> None:
        matrix = experiment_matrix()
        self.assertEqual(len(SIGNAL_PAIRS), 2)
        self.assertEqual(len(COOLDOWN_SCENARIOS), 10)
        self.assertEqual(len(matrix), 20)
        self.assertEqual(matrix["strategy"].nunique(), 20)
        self.assertEqual(int(matrix["is_current_reference"].sum()), 1)

    def test_exit_can_be_unlocked_while_reentry_remains_locked(self) -> None:
        scenario = CooldownScenario("TEST", 0, 7)
        self.assertTrue(
            opposite_action_allowed(
                11, holding_stock=True, last_buy_index=10, last_sell_index=None, scenario=scenario
            )
        )
        self.assertFalse(
            opposite_action_allowed(
                16, holding_stock=False, last_buy_index=None, last_sell_index=10, scenario=scenario
            )
        )
        self.assertTrue(
            opposite_action_allowed(
                18, holding_stock=False, last_buy_index=None, last_sell_index=10, scenario=scenario
            )
        )

    def test_reentry_can_be_unlocked_while_exit_remains_locked(self) -> None:
        scenario = CooldownScenario("TEST", 7, 0)
        self.assertFalse(
            opposite_action_allowed(
                16, holding_stock=True, last_buy_index=10, last_sell_index=None, scenario=scenario
            )
        )
        self.assertTrue(
            opposite_action_allowed(
                11, holding_stock=False, last_buy_index=None, last_sell_index=10, scenario=scenario
            )
        )

    def test_zero_zero_allows_next_trading_day_opposite_action(self) -> None:
        scenario = CooldownScenario("TEST", 0, 0)
        self.assertTrue(
            opposite_action_allowed(
                11, holding_stock=True, last_buy_index=10, last_sell_index=None, scenario=scenario
            )
        )
        self.assertTrue(
            opposite_action_allowed(
                11, holding_stock=False, last_buy_index=None, last_sell_index=10, scenario=scenario
            )
        )


if __name__ == "__main__":
    unittest.main()
