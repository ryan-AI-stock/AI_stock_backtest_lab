import unittest

import numpy as np
import pandas as pd

from backtest_lab.vnext_p1_p2_primary80_ma_slope_cd50_action_legs import feature_panel


class MASlopeActionLegTest(unittest.TestCase):
    def test_slope_definition_matches_close_difference(self):
        frame = pd.DataFrame({"period": "P1", "ticker": "2330", "date": pd.date_range("2020-01-01", periods=65), "value": np.arange(1.0, 66.0), "source_quality": "test"})
        result = feature_panel(frame)
        self.assertEqual(result.iloc[-1].slope5, 4.0)
        self.assertEqual(result.iloc[-1].slope20, 19.0)

    def test_history_ready_needs_sixty_rows(self):
        frame = pd.DataFrame({"period": "P1", "ticker": "2330", "date": pd.date_range("2020-01-01", periods=60), "value": np.arange(1.0, 61.0), "source_quality": "test"})
        result = feature_panel(frame)
        self.assertFalse(result.iloc[-2].history_ready)
        self.assertTrue(result.iloc[-1].history_ready)


if __name__ == "__main__":
    unittest.main()
