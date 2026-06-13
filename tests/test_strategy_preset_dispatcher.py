from __future__ import annotations

import unittest

import test_paths  # noqa: F401

from backtest_lab.strategy_preset_dispatcher import dispatch_pool, resolve_strategy_preset


class StrategyPresetDispatcherTest(unittest.TestCase):
    def test_dispatches_known_preset_to_workflow_and_report_line(self) -> None:
        routed = dispatch_pool(
            {
                "pool_id": "large_cap_best_v20260605",
                "name": "AI中大型權值股池最佳版 v20260605",
                "strategy_preset": "best_v20260605",
            }
        )

        self.assertEqual(routed["workflow_file"], "stock_pool_observation.yml")
        self.assertEqual(routed["report_line"], "stock_pool_observation")
        self.assertTrue(routed["operational_observation"])

    def test_scorecard_preset_is_not_operational_observation(self) -> None:
        spec = resolve_strategy_preset("delayed_public_scorecard_v1")

        self.assertEqual(spec.workflow_file, "model_scorecard_report.yml")
        self.assertFalse(spec.operational_observation)
        self.assertTrue(spec.public_scorecard)

    def test_core_defensive_preset_is_operational_observation(self) -> None:
        spec = resolve_strategy_preset("core_defensive_style_v1")

        self.assertEqual(spec.label, "核心防守風格池 v1")
        self.assertEqual(spec.workflow_file, "stock_pool_observation.yml")
        self.assertTrue(spec.operational_observation)

    def test_rejects_unknown_preset(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported strategy_preset"):
            resolve_strategy_preset("unknown_strategy")


if __name__ == "__main__":
    unittest.main()
