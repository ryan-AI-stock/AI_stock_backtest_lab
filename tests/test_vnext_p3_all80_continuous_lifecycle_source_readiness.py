import json
import unittest

import pandas as pd

from backtest_lab.vnext_p3_all80_continuous_lifecycle_source_readiness import OUT, run


class All80ContinuousLifecycleSourceReadinessTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        run()

    def test_exact_p31_scope(self):
        summary = pd.read_csv(OUT / "p3_all80_continuous_source_readiness_summary.csv").iloc[0]
        self.assertEqual(summary.P3_1_dates, 482)
        self.assertEqual(summary.candidate_rows, 38560)
        self.assertEqual(summary.unique_tickers, 610)

    def test_gaps_block_state_supply(self):
        ready = json.loads((OUT / "readiness_for_p3_all80_continuous_lifecycle_state_supply.json").read_text(encoding="utf-8"))
        self.assertGreater(ready["adjusted_HLC_gap_rows"], 0)
        self.assertFalse(ready["state_supply_materialized"])
        self.assertFalse(ready["ready_for_experiments"])
        self.assertTrue(ready["represents_intended_all80_layer5_state_supply"])

    def test_no_future_or_raw_adjusted_substitution(self):
        audit = pd.read_csv(OUT / "p3_all80_continuous_future_PIT_audit.csv")
        self.assertEqual(audit.violations.sum(), 0)


if __name__ == "__main__":
    unittest.main()
