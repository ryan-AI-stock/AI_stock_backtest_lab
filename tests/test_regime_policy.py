import unittest

import test_paths  # noqa: F401

from backtest_lab.regime_policy import policy_for


class RegimePolicyTest(unittest.TestCase):
    def test_daily_strength_stops_new_entries_in_systemic_bear(self) -> None:
        policy = policy_for("daily_strength", "systemic_bear")

        self.assertFalse(policy.allow_new_entry)
        self.assertEqual(policy.max_equity_exposure, 0.0)
        self.assertEqual(policy.product_mode, "系統性空頭防守")

    def test_weekly_rotation_reduces_exposure_in_correction_bear(self) -> None:
        policy = policy_for("weekly_rotation", "correction_bear")

        self.assertTrue(policy.allow_new_entry)
        self.assertEqual(policy.max_equity_exposure, 0.3)
        self.assertEqual(policy.min_cash_ratio, 0.7)


if __name__ == "__main__":
    unittest.main()
