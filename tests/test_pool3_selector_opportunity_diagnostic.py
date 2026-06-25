from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

import test_paths  # noqa: F401

from backtest_lab.pool3_selector_opportunity_diagnostic import run_pool3_selector_opportunity_diagnostic


class Pool3SelectorOpportunityDiagnosticTest(unittest.TestCase):
    def test_marks_pool3_ignored_opportunity_and_veto(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            panel = root / "event.csv"
            pd.DataFrame(
                [
                    _row("2024-01-02", "exact_consensus_missing", pool3=0.20, formal=0.01, etf50=0.02, lev=0.03),
                    _row(
                        "2024-01-03",
                        "formal_target_selector_preferred_other_pool",
                        pool3=-0.10,
                        formal=0.10,
                        etf50=0.08,
                        lev=0.12,
                    ),
                ]
            ).to_csv(panel, index=False)

            output = run_pool3_selector_opportunity_diagnostic(event_panel_path=panel, output_dir=root / "out")

            events = pd.read_csv(output / "pool3_selector_opportunity_event_panel.csv")
            self.assertEqual(set(events["pool3_opportunity_state"]), {"opportunity_warning", "veto_warning"})
            self.assertEqual(set(events["pool3_selector_diagnostic_boundary"]), {"report_only"})
            self.assertEqual(set(events["pool3_selector_diagnostic_active_in_trade_decision"]), {False})
            self.assertIn("concentration_blocked", set(events["pool3_selector_diagnostic_state"]))
            self.assertIn("veto_explanation", set(events["pool3_selector_diagnostic_state"]))
            self.assertEqual(len(pd.read_csv(output / "pool3_opportunity_warning_candidates.csv")), 1)
            self.assertEqual(len(pd.read_csv(output / "pool3_veto_warning_candidates.csv")), 1)
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertFalse(manifest["formal_model_changed"])
            self.assertFalse(manifest["trade_decision_changed"])
            self.assertFalse(manifest["active_in_trade_decision"])
            self.assertFalse(manifest["pool3_selector_diagnostic_active_in_trade_decision"])

    def test_mixed_or_insufficient_and_none_states_are_report_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            panel = root / "event.csv"
            pd.DataFrame(
                [
                    _row("2024-01-02", "exact_consensus_missing", pool3=0.04, formal=0.01, etf50=0.02, lev=0.08),
                    _row("2024-01-03", "pool3_part_of_exact_consensus", pool3=0.20, formal=0.01, etf50=0.02, lev=0.03),
                ]
            ).to_csv(panel, index=False)

            output = run_pool3_selector_opportunity_diagnostic(event_panel_path=panel, output_dir=root / "out")

            events = pd.read_csv(output / "pool3_selector_opportunity_event_panel.csv")
            self.assertEqual(len(events), 1)
            self.assertEqual(events.iloc[0]["pool3_selector_diagnostic_state"], "mixed_or_insufficient")
            self.assertTrue(_blank(events.iloc[0]["pool3_selector_veto_explanation"]))
            self.assertTrue(_blank(events.iloc[0]["pool3_opportunity_watch_only_reason"]))


def _row(date: str, blocker: str, *, pool3: float, formal: float, etf50: float, lev: float) -> dict[str, object]:
    row: dict[str, object] = {
        "period": "2024_now",
        "signal_date": date,
        "pool3_ticker": "2882.TW",
        "pool3_has_full_stock_vote": True,
        "pool3_blocker_category": blocker,
        "formal_final_target": "2454.TW",
    }
    for horizon in (20, 60, 120):
        row[f"pool3_ticker_forward_{horizon}d_return"] = pool3
        row[f"formal_final_target_forward_{horizon}d_return"] = formal
        row[f"0050_TW_forward_{horizon}d_return"] = etf50
        row[f"00631L_TW_forward_{horizon}d_return"] = lev
    return row


def _blank(value: object) -> bool:
    return pd.isna(value) or str(value).strip() == ""


if __name__ == "__main__":
    unittest.main()
