import json,unittest
import pandas as pd
from backtest_lab import vnext_p3_individual_stock_sequential_confirmation_stage_a as subject
class IndividualSequentialTest(unittest.TestCase):
 @classmethod
 def setUpClass(c):c.r=json.loads((subject.OUT/'readiness_for_individual_stock_sequential_stage_A.json').read_text());c.s=pd.read_csv(subject.OUT/'p3_individual_stock_sequential_fold_supply_gate.csv');c.a=pd.read_csv(subject.OUT/'p3_individual_stock_sequential_future_PIT_audit.csv')
 def test_scope(c):c.assertEqual(c.r['fixed_platforms'],12);c.assertTrue(c.r['sequential_not_additive']);c.assertFalse(c.r['market_controller_used']);c.assertFalse(c.r['portfolio_performance_authorized']);c.assertFalse(c.r['P3_2_outcome_read'])
 def test_folds(c):c.assertEqual(c.s.platform.nunique(),12);c.assertEqual(c.s.fold_id.nunique(),3);c.assertEqual(int(c.a.violations.sum()),0)
if __name__=='__main__':unittest.main()
