import json
import unittest
import pandas as pd
from backtest_lab import vnext_p3_all80_self_range_matched_exit_contract as subject

class MatchedExitContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ready=json.loads((subject.OUT/"readiness_for_KrangeGT30_matched_exit.json").read_text(encoding="utf-8"))
        cls.ledger=pd.read_csv(subject.OUT/"p3_all80_KrangeGT30_matched_entry_exit_ledger.csv.gz",dtype={"ticker":str})
        cls.audit=pd.read_csv(subject.OUT/"p3_all80_KrangeGT30_matched_exit_future_PIT_audit.csv")
    def test_authority_and_scope(self):
        self.assertEqual(len(self.ledger),self.ready["authority_entry_rows"])
        self.assertEqual(set(self.ledger.K_range_threshold),{30})
        self.assertFalse(self.ready["P3_2_outcome_read_authorized"])
        self.assertFalse(self.ready["market_used"])
        self.assertFalse(self.ready["ranking_used"])
    def test_pit_and_costs(self):
        self.assertEqual(int(self.audit.violations.sum()),0)
        self.assertTrue({"EP05_round_trip_cost_rate_5bp","EP05_round_trip_cost_rate_10bp","EP05_round_trip_cost_rate_20bp"}.issubset(self.ledger.columns))

if __name__=="__main__": unittest.main()
