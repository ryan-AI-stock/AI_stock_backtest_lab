from __future__ import annotations

import unittest

import pandas as pd

from backtest_lab.ma_price_slope_cd7_hypothesis_backtest import (
    BASELINE_STRATEGY,
    ENTRY_RULES,
    EXIT_RULES,
    add_features,
    add_rule_signals,
    hypothesis_matrix,
)


class MaPriceSlopeCd7HypothesisBacktestTests(unittest.TestCase):
    def test_matrix_contains_one_baseline_and_35_new_hypotheses(self) -> None:
        matrix = hypothesis_matrix()
        self.assertEqual(len(ENTRY_RULES), 6)
        self.assertEqual(len(EXIT_RULES), 6)
        self.assertEqual(len(matrix), 36)
        self.assertEqual(matrix["strategy"].nunique(), 36)
        self.assertEqual(int(matrix["is_existing_baseline"].sum()), 1)
        self.assertEqual(int((~matrix["is_existing_baseline"]).sum()), 35)
        self.assertIn(BASELINE_STRATEGY, set(matrix["strategy"]))

    def test_baseline_signal_matches_original_definition(self) -> None:
        dates = pd.date_range("2020-01-01", periods=40, freq="B")
        prices = [100 + index for index in range(40)]
        frame = pd.DataFrame({"date": dates, "0050_adj_close": prices, "00631L_adj_close": prices})
        featured = add_features(frame)
        signaled = add_rule_signals(featured, ENTRY_RULES[0], EXIT_RULES[0])
        expected_buy = (featured["0050_adj_close"] > featured["ma4"]) & (featured["slope7"] > 0)
        expected_sell = (featured["0050_adj_close"] < featured["ma10"]) & (featured["slope20"] < 0)
        pd.testing.assert_series_equal(signaled["buy_signal"], expected_buy.fillna(False), check_names=False)
        pd.testing.assert_series_equal(signaled["sell_signal"], expected_sell.fillna(False), check_names=False)

    def test_stricter_entry_variants_are_subsets_of_baseline(self) -> None:
        dates = pd.date_range("2020-01-01", periods=80, freq="B")
        prices = [100 + ((index % 20) - 10) * 0.7 + index * 0.05 for index in range(80)]
        frame = pd.DataFrame({"date": dates, "0050_adj_close": prices, "00631L_adj_close": prices})
        featured = add_features(frame)
        baseline = add_rule_signals(featured, ENTRY_RULES[0], EXIT_RULES[0])["buy_signal"]
        for rule in ENTRY_RULES[1:]:
            candidate = add_rule_signals(featured, rule, EXIT_RULES[0])["buy_signal"]
            self.assertFalse(bool((candidate & ~baseline).any()), rule.rule_id)


if __name__ == "__main__":
    unittest.main()
