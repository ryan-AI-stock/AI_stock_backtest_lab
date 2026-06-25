from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

import test_paths  # noqa: F401

from backtest_lab.final_decision_layer_forward_outcome_adapter import run_final_decision_layer_forward_outcome_adapter


class FinalDecisionLayerForwardOutcomeAdapterTest(unittest.TestCase):
    def test_forward_outcome_adapter_keeps_report_only_boundaries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state_panel = root / "state.csv"
            cache = root / "prices"
            output = root / "out"
            cache.mkdir()
            pd.DataFrame(
                [
                    _state_row("2024-01-02", "strong_consensus", "2330.TW", "stock_attack", not_eligible=False),
                    _state_row("2024-01-03", "diagnostic_divergence", "", "none", not_eligible=True),
                    _state_row("2024-05-10", "defensive_market_exposure", "00631L.TW", "market_exposure_tool", not_eligible=True),
                ]
            ).to_csv(state_panel, index=False)
            _write_price(cache, "2330.TW", 100, 150)
            _write_price(cache, "0050.TW", 100, 150, step=0.2)
            _write_price(cache, "00631L.TW", 50, 150, step=0.3)

            result = run_final_decision_layer_forward_outcome_adapter(
                state_panel_path=state_panel,
                price_cache_dir=cache,
                output_dir=output,
            )

            manifest = json.loads((result / "manifest.json").read_text(encoding="utf-8"))
            self.assertFalse(manifest["formal_model_changed"])
            self.assertFalse(manifest["trade_decision_changed"])
            self.assertFalse(manifest["active_in_trade_decision"])
            self.assertFalse(manifest["uses_forward_return_as_rule"])
            self.assertEqual(manifest["final_decision_layer_boundary"], "report_only_diagnostic")

            panel = pd.read_csv(result / "final_decision_forward_outcome_panel.csv")
            complete = panel[panel["decision_date"].eq("2024-01-02")].iloc[0]
            self.assertTrue(bool(complete["outcome_data_complete"]))
            self.assertNotEqual(str(complete["forward_20d_return"]), "")
            self.assertIn("forward_20d_excess_vs_0050", panel.columns)
            self.assertIn("forward_20d_excess_vs_00631L", panel.columns)
            self.assertIn("forward_20d_excess_vs_0050x2", panel.columns)
            self.assertFalse(bool(complete["uses_forward_return_as_rule"]))

            no_target = panel[panel["decision_date"].eq("2024-01-03")].iloc[0]
            self.assertFalse(bool(no_target["outcome_data_complete"]))
            self.assertEqual(no_target["outcome_blocked_reason"], "no_final_target")
            self.assertTrue(bool(no_target["not_eligible_for_formal_selector"]))

            immature = panel[panel["decision_date"].eq("2024-05-10")].iloc[0]
            self.assertFalse(bool(immature["outcome_data_complete"]))
            self.assertIn("insufficient_120d_forward_window", str(immature["outcome_blocked_reason"]))
            self.assertEqual(immature["final_target_type"], "market_exposure_tool")

            by_state = pd.read_csv(result / "forward_outcome_by_state.csv")
            strong = by_state[by_state["final_decision_state"].eq("strong_consensus")].iloc[0]
            self.assertEqual(int(strong["event_count"]), 1)
            self.assertEqual(int(strong["complete_event_count"]), 1)
            defensive = by_state[by_state["final_decision_state"].eq("defensive_market_exposure")].iloc[0]
            self.assertEqual(int(defensive["complete_event_count"]), 0)

            coverage = pd.read_csv(result / "forward_outcome_data_coverage.csv")
            self.assertEqual(int(coverage.iloc[0]["event_count"]), 3)
            self.assertEqual(int(coverage.iloc[0]["outcome_data_complete_count"]), 1)


def _state_row(date: str, state: str, target: str, target_type: str, *, not_eligible: bool) -> dict[str, object]:
    return {
        "period": "2024_now",
        "signal_date": date,
        "final_decision_state": state,
        "final_target_type": target_type,
        "final_target_ticker": target,
        "decision_protocol_used": state in {"diagnostic_divergence", "actionable_divergence"},
        "decision_protocol_reason": "tail_divergence_diagnostic" if state == "diagnostic_divergence" else "",
        "final_target_source": "exact_consensus" if target else "none",
        "exposure_target": target if target_type == "market_exposure_tool" else "",
        "target_priority_rank": 1 if target else "",
        "not_eligible_for_formal_selector": not_eligible,
    }


def _write_price(cache: Path, ticker: str, start: float, days: int, *, step: float = 1.0) -> None:
    rows = []
    for index in range(days):
        close = start + index * step
        rows.append(
            {
                "date": (pd.Timestamp("2024-01-02") + pd.Timedelta(days=index)).strftime("%Y-%m-%d"),
                "open": close,
                "close": close,
                "adj_close": close,
            }
        )
    pd.DataFrame(rows).to_csv(cache / f"{ticker.replace('.', '_')}.csv", index=False)


if __name__ == "__main__":
    unittest.main()
