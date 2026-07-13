import json
import unittest

import pandas as pd

from backtest_lab import vnext_p3_c3_top1_incumbent_path_contract as subject


class IncumbentPathContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        subject.run()
        cls.ready = json.loads((subject.OUT / "readiness_for_incumbent_path_corrected_NAV.json").read_text(encoding="utf-8"))
        cls.actions = pd.read_csv(subject.OUT / "p3_C3_top1_incumbent_daily_action_ledger.csv")

    def test_scope_and_governance(self):
        self.assertEqual(self.ready["decision_dates"], 482)
        self.assertFalse(self.ready["P3_2_outcome_read_authorized"])
        self.assertEqual(self.ready["future_data_violation_count"], 0)

    def test_one_action_per_day(self):
        self.assertEqual(len(self.actions), 482)
        self.assertFalse(self.actions.decision_date.duplicated().any())

    def test_radar_fill_closes_incumbent_pit_gap(self):
        if subject.RADAR_COMPACT.exists():
            self.assertEqual(self.ready["incumbent_PIT_missing_rows"], 0)
            self.assertTrue(self.ready["exact_path_ready"])


if __name__ == "__main__":
    unittest.main()
