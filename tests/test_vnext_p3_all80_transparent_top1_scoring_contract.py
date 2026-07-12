import json
import unittest

import pandas as pd

from backtest_lab.vnext_p3_all80_transparent_top1_scoring_contract import OUT, run


class All80Top1ContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        run()

    def test_exact_all80_keys(self):
        coverage = pd.read_csv(OUT / "p3_all80_daily_key_coverage.csv")
        self.assertTrue(coverage.exact_80_key_ready.all())
        self.assertEqual(len(coverage), 715)

    def test_p3_2_is_untouched(self):
        folds = pd.read_csv(OUT / "p3_all80_P3_1_expanding_fold_calendar.csv")
        self.assertFalse(folds.P3_2_used_for_selection.astype(bool).any())
        self.assertEqual(len(folds), 3)

    def test_no_top1_or_portfolio_yet(self):
        readiness = json.loads((OUT / "readiness_for_p3_all80_transparent_top1_scoring.json").read_text(encoding="utf-8"))
        self.assertFalse(readiness["Top1_predictions_materialized"])
        self.assertFalse(readiness["portfolio_NAV_executed"])


if __name__ == "__main__":
    unittest.main()
