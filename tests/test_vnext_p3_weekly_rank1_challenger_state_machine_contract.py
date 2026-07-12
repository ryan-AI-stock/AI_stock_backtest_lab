import unittest

import pandas as pd

from backtest_lab.vnext_p3_weekly_rank1_challenger_state_machine_contract import OUT, run


class Rank1ChallengerContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        run()

    def test_no_performance_or_action_is_materialized(self):
        readiness = pd.read_json(OUT / "readiness_for_p3_rank1_challenger_state_machine.json", typ="series")
        self.assertFalse(bool(readiness["ready_for_experiments"]))
        self.assertFalse(bool(readiness["state_machine_actions_materialized"]))

    def test_decision_table_has_no_core_default(self):
        decisions = pd.read_csv(OUT / "p3_rank1_challenger_minimum_strategy_center_decision_table.csv")
        self.assertEqual(len(decisions), 5)
        self.assertFalse(decisions.core_default_applied.astype(bool).any())


if __name__ == "__main__":
    unittest.main()
