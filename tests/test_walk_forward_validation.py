from __future__ import annotations

import unittest

import pandas as pd

from backtest_lab.walk_forward_validation import walk_forward_folds


class WalkForwardValidationTest(unittest.TestCase):
    def test_folds_use_two_year_train_then_future_test(self) -> None:
        folds = walk_forward_folds()

        self.assertEqual(len(folds), 5)
        for fold in folds:
            self.assertLess(pd.Timestamp(fold.train_end), pd.Timestamp(fold.test_start))
            self.assertLessEqual(pd.Timestamp(fold.test_start), pd.Timestamp(fold.test_end))


if __name__ == "__main__":
    unittest.main()
