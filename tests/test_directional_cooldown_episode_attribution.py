from __future__ import annotations

import unittest

import pandas as pd

from backtest_lab.directional_cooldown_episode_attribution import (
    CHALLENGER_SCENARIO,
    COST_BASIS,
    PERIOD,
    REFERENCE_SCENARIO,
    SIGNAL_PAIR,
    closed_episodes,
    pair_episode_attribution,
    reconciliation,
)


def trade(scenario: str, side: str, index: int, cash_before: float, cash_after: float) -> dict[str, object]:
    return {
        "period": PERIOD,
        "signal_pair": SIGNAL_PAIR,
        "cooldown_scenario": scenario,
        "cost_basis": COST_BASIS,
        "signal_date": f"2025-01-{index:02d}",
        "execution_date": f"2025-01-{index + 1:02d}",
        "execution_index": index,
        "side": side,
        "cash_before": cash_before,
        "cash_after": cash_after,
    }


class DirectionalCooldownEpisodeAttributionTests(unittest.TestCase):
    def test_closed_episodes_builds_exact_wealth_factors(self) -> None:
        trades = pd.DataFrame(
            [
                trade(REFERENCE_SCENARIO, "buy", 1, 100.0, 0.0),
                trade(REFERENCE_SCENARIO, "sell", 3, 0.0, 110.0),
                trade(REFERENCE_SCENARIO, "buy", 5, 110.0, 0.0),
                trade(REFERENCE_SCENARIO, "sell", 8, 0.0, 99.0),
            ]
        )
        episodes = closed_episodes(trades, REFERENCE_SCENARIO)
        self.assertEqual(episodes["wealth_factor"].round(6).tolist(), [1.1, 0.9])
        self.assertEqual(episodes["holding_td"].tolist(), [2, 3])

    def test_reconciliation_matches_compounded_final_nav(self) -> None:
        rows = [
            trade(REFERENCE_SCENARIO, "buy", 1, 100.0, 0.0),
            trade(REFERENCE_SCENARIO, "sell", 3, 0.0, 110.0),
            trade(REFERENCE_SCENARIO, "buy", 5, 110.0, 0.0),
            trade(REFERENCE_SCENARIO, "sell", 8, 0.0, 99.0),
            trade(CHALLENGER_SCENARIO, "buy", 1, 100.0, 0.0),
            trade(CHALLENGER_SCENARIO, "sell", 3, 0.0, 105.0),
            trade(CHALLENGER_SCENARIO, "buy", 6, 105.0, 0.0),
            trade(CHALLENGER_SCENARIO, "sell", 8, 0.0, 105.0),
        ]
        trades = pd.DataFrame(rows)
        ref = closed_episodes(trades, REFERENCE_SCENARIO)
        alt = closed_episodes(trades, CHALLENGER_SCENARIO)
        paired = pair_episode_attribution(ref, alt, reference_final_equity=990_000.0)
        checks = reconciliation(
            ref,
            alt,
            paired,
            reference_final_equity=990_000.0,
            challenger_final_equity=1_050_000.0,
        )
        self.assertTrue(checks["pass"].all())
        self.assertEqual(paired["buy_delay_td"].tolist(), [0, 1])

    def test_open_episode_is_rejected(self) -> None:
        trades = pd.DataFrame([trade(REFERENCE_SCENARIO, "buy", 1, 100.0, 0.0)])
        with self.assertRaises(ValueError):
            closed_episodes(trades, REFERENCE_SCENARIO)


if __name__ == "__main__":
    unittest.main()
