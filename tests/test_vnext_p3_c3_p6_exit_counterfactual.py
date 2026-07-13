import json
import unittest

import pandas as pd

from backtest_lab import vnext_p3_c3_p6_exit_counterfactual as subject


class P6ExitCounterfactualTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        subject.run()
        cls.path = pd.read_csv(subject.OUT / "p3_C3_P6_exit_to_cash_corrected_NAV_daily_wealth_ledger.csv")
        cls.ready = json.loads((subject.OUT / "readiness_for_P6_exit_counterfactual.json").read_text(encoding="utf-8"))

    def test_exact_event_and_no_replacement(self):
        event = self.path.loc[(self.path.date.eq("2024-08-06")) & self.path.slippage_bp_per_side.eq(10)].iloc[0]
        self.assertEqual(event.transition_type, "stock_to_no_position")
        self.assertTrue(pd.isna(event.counterfactual_target))

    def test_nav_and_governance(self):
        self.assertTrue(self.ready["exact_rechain_ready"])
        self.assertFalse(self.ready["P3_2_outcome_read_authorized"])
        self.assertEqual(self.ready["future_data_violation_count"], 0)


if __name__ == "__main__":
    unittest.main()
