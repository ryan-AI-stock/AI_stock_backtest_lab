import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

import test_paths  # noqa: F401

from backtest_lab.dynamic_pool1_taxonomy_evidence_panel import run_dynamic_pool1_taxonomy_evidence_panel


class DynamicPool1TaxonomyEvidencePanelTest(unittest.TestCase):
    def test_builds_diagnostic_taxonomy_panel_without_formalizing_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            radar = root / "radar"
            output = root / "out"
            radar.mkdir()
            _write_radar_fixture(radar)

            run_dynamic_pool1_taxonomy_evidence_panel(radar_output=radar, output_dir=output)

            readiness = json.loads((output / "taxonomy_evidence_readiness.json").read_text(encoding="utf-8"))
            self.assertEqual(readiness["status"], "diagnostic_evidence_surface_ready")
            self.assertEqual(readiness["accepted_evidence_rows"], 2)
            self.assertEqual(readiness["accepted_unique_tickers"], 1)
            self.assertEqual(readiness["blocked_or_needs_review_tickers"], 1)
            self.assertFalse(readiness["ready_for_strategy_replay"])
            self.assertFalse(readiness["formal_taxonomy"])
            self.assertFalse(readiness["accepted_for_formal"])
            self.assertTrue(readiness["human_review_required"])

            panel = pd.read_csv(output / "taxonomy_evidence_panel.csv")
            self.assertEqual(len(panel), 2)
            self.assertTrue(panel["accepted_for_diagnostic"].astype(bool).all())
            self.assertFalse(panel["accepted_for_formal"].astype(bool).any())
            self.assertTrue(panel["human_review_required"].astype(bool).all())
            self.assertFalse(panel["active_in_trade_decision"].astype(bool).any())

            by_ticker = pd.read_csv(output / "taxonomy_evidence_by_ticker.csv")
            accepted = by_ticker[by_ticker["ticker"].eq("2330.TW")].iloc[0]
            blocked = by_ticker[by_ticker["ticker"].eq("6488.TWO")].iloc[0]
            self.assertEqual(int(accepted["accepted_evidence_count"]), 2)
            self.assertFalse(bool(blocked["has_accepted_evidence"]))
            self.assertIn("document", blocked["next_programmatic_source"])

            blockers = pd.read_csv(output / "taxonomy_blocker_panel.csv")
            self.assertEqual(len(blockers), 1)
            self.assertFalse(blockers["accepted_for_formal"].astype(bool).any())


def _write_radar_fixture(root: Path) -> None:
    payload = {
        "task_id": "TASK-RADAR-DATA-DYNAMIC-POOL1-MOPS-MAINLINE-EVIDENCE-LEDGER-20260704",
        "status": "partial_ready_diagnostic_evidence_v0",
        "accepted_theme_taxonomy_rows": 2,
        "accepted_unique_tickers": 1,
        "blocked_or_needs_review_tickers": 1,
        "ready_for_taxonomy_evidence_panel": True,
        "ready_for_strategy_replay": False,
        "accepted_for_formal": False,
        "diagnostic_only": True,
        "future_data_violation_count": 0,
        "formal_model_changed": False,
        "trade_decision_changed": False,
        "active_in_trade_decision": False,
        "source_boundary": "dated official diagnostic evidence only",
    }
    (root / "readiness_for_core.json").write_text(json.dumps(payload), encoding="utf-8")
    (root / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")
    pd.DataFrame(
        [
            _accepted("2330.TW", "台積電", "server", "AI server"),
            _accepted("2330.TW", "台積電", "semiconductor", "semiconductor"),
        ]
    ).to_csv(root / "accepted_evidence_rows.csv", index=False)
    pd.DataFrame(
        [
            {
                "ticker": "6488.TWO",
                "company_name": "環球晶",
                "market": "TPEx",
                "candidate_scope": "pool1b_materials",
                "status": "needs_review_or_no_keyword_match_in_v0_sources",
                "blocked_reason": "No accepted date-aware keyword evidence found.",
                "next_programmatic_source": "bounded document locator",
                "accepted_for_diagnostic": False,
                "accepted_for_formal": False,
            }
        ]
    ).to_csv(root / "blocked_or_needs_review.csv", index=False)
    pd.DataFrame(
        [
            {
                "category": "accepted_diagnostic_rows",
                "row_count": 2,
                "accepted_for_diagnostic": True,
                "accepted_for_formal": False,
                "human_review_required": True,
                "decision": "partial_ready",
            }
        ]
    ).to_csv(root / "taxonomy_acceptance_audit.csv", index=False)
    pd.DataFrame(
        [
            {
                "data_area": "taxonomy",
                "future_data_violation": False,
                "future_data_violation_count": 0,
                "audit_reason": "fixture",
            }
        ]
    ).to_csv(root / "future_data_violation_audit.csv", index=False)


def _accepted(ticker: str, name: str, layer: str, theme: str) -> dict[str, object]:
    return {
        "ticker": ticker,
        "company_name": name,
        "market": "TWSE",
        "candidate_scope": "old_ai",
        "ai_supply_chain_layer": layer,
        "mainline_theme_label": theme,
        "source_doc_type": "TWSE OpenAPI ESG disclosure aggregation",
        "source_doc_date": "2026-07-04",
        "source_date": "2026-07-04",
        "effective_date": "2025-12-31",
        "as_of_date": "2025-12-31",
        "confidence_level": "medium",
        "accepted_for_diagnostic": True,
        "accepted_for_formal": False,
        "formal_exact": False,
        "human_review_required": True,
    }


if __name__ == "__main__":
    unittest.main()
