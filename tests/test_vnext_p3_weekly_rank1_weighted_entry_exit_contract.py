import json
import unittest

import pandas as pd

from backtest_lab.vnext_p3_weekly_rank1_weighted_entry_exit_contract import OUT, run


class WeeklyRank1WeightedEntryExitContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        run()
        cls.frame = pd.read_csv(OUT / "p3_rank1_daily_stock_market_feature_contract.csv.gz", dtype={"ticker": str})

    def test_one_canonical_rank1_per_day(self):
        self.assertEqual(len(self.frame), 715)
        self.assertEqual(self.frame.decision_date.nunique(), 715)
        self.assertTrue(self.frame.pool_rank.eq(1).all())

    def test_exit_is_not_entry_sign_flip(self):
        self.assertFalse(self.frame.exit_short_rs_deterioration.equals(100 - self.frame.entry_momentum))
        self.assertTrue(self.frame.overheat_warning_only_no_exit.notna().all())

    def test_p3_2_untouched_and_no_nav(self):
        folds = pd.read_csv(OUT / "p3_rank1_P3_1_expanding_fold_calendar.csv")
        self.assertFalse(folds.P3_2_used_for_selection.astype(bool).any())
        ready = json.loads((OUT / "readiness_for_p3_rank1_weighted_entry_exit.json").read_text(encoding="utf-8"))
        self.assertFalse(ready["portfolio_NAV_materialized"])
        self.assertTrue(ready["ready_for_stage_A_candidate_quality"])

    def test_tdcc_not_zero_filled_or_used(self):
        p31 = self.frame[self.frame.P3_segment.str.startswith("P3-1")]
        self.assertTrue(p31.tdcc_score.isna().all())
        self.assertFalse(self.frame.TDCC_main_score_used.astype(bool).any())


if __name__ == "__main__":
    unittest.main()
