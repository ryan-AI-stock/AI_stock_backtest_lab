import json
import unittest

import pandas as pd

from backtest_lab import vnext_p3_all80_candidate_position_dual_state_contract as subject


class CandidatePositionDualStateTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        subject.run()
        cls.ready = json.loads((subject.OUT / "readiness_for_candidate_position_dual_state.json").read_text(encoding="utf-8"))
        cls.panel = pd.read_csv(subject.OUT / "p3_all80_candidate_C0_C3_daily_panel.csv.gz")

    def test_scope_and_separation(self):
        self.assertEqual(self.ready["P3_1_dates"], 482)
        self.assertEqual(self.ready["candidate_rows"], 38560)
        self.assertTrue(self.ready["candidate_and_position_states_separated"])
        self.assertFalse(self.ready["performance_authorized"])

    def test_candidate_states_only(self):
        self.assertTrue(set(self.panel.candidate_state).issubset({"C0","C1","C2","C3","BLOCKED"}))


if __name__ == "__main__":
    unittest.main()
