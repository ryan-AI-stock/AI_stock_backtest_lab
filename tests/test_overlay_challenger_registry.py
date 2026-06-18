from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import test_paths  # noqa: F401

from backtest_lab.decision_layers import SHADOW_OVERLAY
from backtest_lab.institutional_flow_overlay_shadow import default_chip_flow_rules, default_overlay_rules
from backtest_lab.overlay_challenger_registry import (
    build_overlay_challenger_specs,
    overlay_challenger_manifest,
    write_overlay_challenger_manifest,
)


class OverlayChallengerRegistryTest(unittest.TestCase):
    def test_registry_marks_all_challengers_as_shadow_not_formal(self) -> None:
        specs = build_overlay_challenger_specs()

        self.assertEqual(len(specs), len(default_overlay_rules()) + len(default_chip_flow_rules()))
        self.assertTrue(all(spec.decision_layer == SHADOW_OVERLAY for spec in specs))
        self.assertTrue(all(not spec.active_in_trade_decision for spec in specs))
        self.assertTrue(all("formal" in spec.promotion_gate.lower() for spec in specs))

    def test_manifest_includes_crowding_and_chip_flow_specs(self) -> None:
        manifest = overlay_challenger_manifest()
        ids = {item["challenger_id"] for item in manifest["challengers"]}

        self.assertEqual(manifest["decision_layer"], SHADOW_OVERLAY)
        self.assertFalse(manifest["active_in_trade_decision"])
        self.assertEqual(manifest["formal_promotion_status"], "not_promoted")
        self.assertIn("institutional_overlay_foreign3_or_trust2_reduce50", ids)
        self.assertIn("chip_flow_overlay_chip_two_signal_price_dd10_reduce50", ids)

        chip = next(item for item in manifest["challengers"] if item["challenger_id"] == "chip_flow_overlay_chip_two_signal_price_dd10_reduce50")
        self.assertEqual(chip["rule_family"], "chip_flow_crowding_overheat")
        self.assertIn("margin_short_daily", chip["data_sources_required"])
        self.assertIn("day_trading_daily", chip["data_sources_required"])
        self.assertFalse(chip["active_in_trade_decision"])

    def test_writes_manifest_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "overlay_challenger_manifest.json"
            manifest = write_overlay_challenger_manifest(path)

            self.assertTrue(path.exists())
            self.assertEqual(manifest["candidate_count"], len(manifest["challengers"]))
            self.assertIn("not_promoted", path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
