import json
import unittest

import pandas as pd

from backtest_lab import vnext_p3_c3_leave_one_episode_out as subject


class LeaveOneEpisodeOutTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        subject.run()
        cls.ready = json.loads((subject.OUT / "readiness_for_leave_one_episode_out_exact_rechain.json").read_text(encoding="utf-8"))
        cls.summary = pd.read_csv(subject.OUT / "p3_C3_leave_one_episode_out_baseline_reconciliation.csv")

    def test_all_scenarios_and_paths(self):
        self.assertEqual(self.ready["scenario_count"], 4)
        self.assertEqual(len(self.summary), 12)
        self.assertTrue(self.ready["exact_rechain_ready"])

    def test_governance(self):
        self.assertFalse(self.ready["P3_2_outcome_read_authorized"])
        self.assertFalse(self.ready["new_strategy_variant"])
        self.assertEqual(self.ready["future_data_violation_count"], 0)


if __name__ == "__main__":
    unittest.main()
