from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import test_paths  # noqa: F401

from backtest_lab.decision_layers import (
    CANDIDATE_SOURCE,
    DIAGNOSTIC,
    FORMAL_TRADE_SIGNAL,
    SHADOW_OVERLAY,
    decision_layer_metadata,
    default_stock_pool_model_layer_audit,
    validate_decision_layer,
    write_model_layer_audit,
)


class DecisionLayersTest(unittest.TestCase):
    def test_metadata_serializes_trade_decision_status(self) -> None:
        metadata = decision_layer_metadata(
            decision_layer=FORMAL_TRADE_SIGNAL,
            active_in_trade_decision=True,
            source_module="frozen_strategy_monitor",
            signal_date="2026-06-12",
        )

        self.assertEqual(metadata["decision_layer"], FORMAL_TRADE_SIGNAL)
        self.assertTrue(metadata["active_in_trade_decision"])
        self.assertEqual(metadata["source_module"], "frozen_strategy_monitor")

    def test_invalid_decision_layer_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown decision layer"):
            validate_decision_layer("trade_signal_but_unclear")

    def test_model_layer_audit_marks_risk_and_valuation_as_diagnostic(self) -> None:
        audit = default_stock_pool_model_layer_audit(
            signal_date="2026-06-12",
            generated_pools=[
                {
                    "pool_id": "large_cap_best_v20260605",
                    "decision_layer": FORMAL_TRADE_SIGNAL,
                    "active_in_trade_decision": True,
                }
            ],
            risk_factor_sources={"margin_short": "margin_short.latest.csv"},
            valuation_source="valuation.latest.csv",
        )
        items = {item["layer_name"]: item for item in audit["items"]}

        self.assertTrue(items["market_regime_and_risk_budget"]["used_by_formal_trade"])
        self.assertEqual(items["stock_pool_candidate_source"]["decision_layer"], CANDIDATE_SOURCE)
        self.assertEqual(items["chip_margin_overheat"]["decision_layer"], DIAGNOSTIC)
        self.assertFalse(items["chip_margin_overheat"]["used_by_formal_trade"])
        self.assertEqual(items["official_margin_short_ingestion"]["decision_layer"], DIAGNOSTIC)
        self.assertFalse(items["official_margin_short_ingestion"]["used_by_formal_trade"])
        self.assertEqual(items["official_margin_short_ingestion"]["source_module"], "margin_short_ingestion_spec")
        self.assertEqual(items["crowding_chip_flow_challengers"]["decision_layer"], SHADOW_OVERLAY)
        self.assertFalse(items["crowding_chip_flow_challengers"]["used_by_formal_trade"])
        self.assertEqual(items["crowding_chip_flow_challengers"]["source_module"], "overlay_challenger_registry")
        self.assertEqual(items["valuation_sanity"]["decision_layer"], DIAGNOSTIC)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model_layer_audit.json"
            write_model_layer_audit(path, audit)
            self.assertIn("chip_margin_overheat", path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
