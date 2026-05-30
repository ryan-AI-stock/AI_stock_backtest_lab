import unittest
from pathlib import Path

import test_paths  # noqa: F401

from backtest_lab.config import load_config


class ConfigTest(unittest.TestCase):
    def test_loads_ep05_universe_and_separates_entry_rules(self) -> None:
        config = load_config(Path("configs") / "ep05_universe.json")

        self.assertEqual(config.start_date, "2024-01-02")
        self.assertEqual(config.warmup_start_date, "2023-01-01")
        self.assertEqual(config.execution.benchmark_entry_rule, "buy_and_hold_own_benchmark")
        self.assertEqual(
            config.execution.initial_entry_rule,
            "use_strategy_signal_from_previous_available_close",
        )
        self.assertEqual(config.groups[0].benchmark, "0050.TW")
        self.assertEqual(config.groups[1].benchmark, "00631L.TW")
        self.assertIn("dual_momentum_vol_control", config.strategies)


if __name__ == "__main__":
    unittest.main()
