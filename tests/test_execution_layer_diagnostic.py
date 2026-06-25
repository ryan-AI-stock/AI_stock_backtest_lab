from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

import test_paths  # noqa: F401

from backtest_lab.execution_layer_diagnostic import run_execution_layer_diagnostic


class ExecutionLayerDiagnosticTest(unittest.TestCase):
    def test_runner_builds_report_only_execution_gap_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            formal_daily = root / "formal_daily.csv"
            price_cache = root / "prices"
            price_cache.mkdir()
            output = root / "out"
            pd.DataFrame(
                [
                    _daily_row("2024-01-02", "00631L.TW", "00631L.TW", "buy", 100_000, 100),
                    _daily_row("2024-01-03", "2454.TW", "2454.TW", "switch", 200_000, 220),
                    _daily_row("2024-01-04", "00631L.TW", "00631L.TW", "switch", 210_000, 230),
                    _daily_row("2024-01-05", "00631L.TW", "00631L.TW", "hold", 0, 0),
                ]
            ).to_csv(formal_daily, index=False)
            _write_price(price_cache, "00631L.TW", [10, 11, 9, 12, 13, 14, 15, 16])
            _write_price(price_cache, "2454.TW", [100, 101, 105, 103, 107, 110, 112, 115])

            result = run_execution_layer_diagnostic(
                formal_daily_path=formal_daily,
                output_dir=output,
                price_cache_dir=price_cache,
            )

            manifest = json.loads((result / "manifest.json").read_text(encoding="utf-8"))
            self.assertFalse(manifest["formal_model_changed"])
            self.assertFalse(manifest["trade_decision_changed"])
            self.assertFalse(manifest["active_in_trade_decision"])
            self.assertFalse(manifest["execution_diagnostic_active_in_trade_decision"])
            self.assertEqual(manifest["boundary"], "report_only_diagnostic")
            self.assertTrue(manifest["price_cache_loaded"])

            changes = pd.read_csv(result / "formal_target_change_panel.csv")
            self.assertIn("reversal_within_3_trading_rows", changes.columns)
            self.assertTrue(changes["reversal_within_3_trading_rows"].astype(bool).any())

            transitions = pd.read_csv(result / "holding_transition_diagnostic.csv")
            self.assertIn("full_rotation_flag", transitions.columns)
            self.assertTrue(transitions["full_rotation_flag"].astype(bool).any())
            self.assertFalse(transitions["execution_diagnostic_active_in_trade_decision"].astype(bool).any())

            summary = pd.read_csv(result / "execution_gap_summary.csv").iloc[0]
            self.assertEqual(int(summary["target_change_count"]), 3)
            self.assertGreater(float(summary["rapid_flip_rate"]), 0)
            self.assertFalse(bool(summary["active_in_trade_decision"]))

            preplan = pd.read_csv(result / "execution_variant_preplan.csv")
            self.assertIn("minimum_hold_N", preplan["variant_id"].tolist())
            self.assertFalse(preplan["active_in_trade_decision"].astype(bool).any())

            event_study = pd.read_csv(result / "execution_event_study_panel.csv")
            self.assertIn("target_change_forward_return_5d", event_study.columns)
            self.assertIn("current_holding_forward_return_5d", event_study.columns)
            self.assertIn("cash_wait_forward_return_vs_target_5d", event_study.columns)
            self.assertIn("conflict_source_label", event_study.columns)
            self.assertTrue(pd.to_numeric(event_study["target_change_forward_return_5d"], errors="coerce").notna().any())
            self.assertFalse(event_study["active_in_trade_decision"].astype(bool).any())

            hold_cooldown = pd.read_csv(result / "minimum_hold_cooldown_event_study.csv")
            self.assertEqual(hold_cooldown["horizon_days"].tolist(), [5, 20, 60])

            partial_daily = pd.read_csv(result / "partial_switch_simulator_daily.csv")
            self.assertIn("partial_switch_equity_cost_proxy", partial_daily.columns)
            self.assertFalse(partial_daily["active_in_trade_decision"].astype(bool).any())

            partial_summary = pd.read_csv(result / "partial_switch_summary.csv").iloc[0]
            self.assertEqual(partial_summary["simulator_scope"], "cost_and_mdd_proxy_only")
            self.assertFalse(bool(partial_summary["formal_model_changed"]))

            sell_first = pd.read_csv(result / "sell_first_then_buy_gap_study.csv")
            self.assertIn("cash_wait_forward_return_vs_target", sell_first.columns)

            pause_readiness = pd.read_csv(result / "pause_on_conflict_input_readiness.csv")
            self.assertIn("final_decision_diagnostic", pause_readiness["input_source"].tolist())
            final_decision = pause_readiness[pause_readiness["input_source"] == "final_decision_diagnostic"].iloc[0]
            self.assertEqual(final_decision["readiness_state"], "blocked")

            contract = pd.read_csv(result / "execution_layer_field_contract.csv")
            self.assertIn("target_change_forward_return_5_20_60d", contract["field_or_capability"].tolist())
            self.assertFalse(contract["active_in_trade_decision"].astype(bool).any())
            self.assertTrue((result / "final_summary_zh.md").exists())


def _daily_row(date: str, winner: str, position: str, action: str, turnover: float, cost: float) -> dict:
    return {
        "date": date,
        "period": "fixture",
        "pool1_vote": winner,
        "pool2_vote": winner,
        "pool3_vote": "",
        "consensus_state": "consensus",
        "winner_ticker": winner,
        "position_ticker": position,
        "cash": 0,
        "equity": 1_000_000,
        "drawdown": 0,
        "turnover": turnover,
        "transaction_cost": cost,
        "action": action,
        "data_status": "formal_daily_replay",
    }


def _write_price(cache: Path, ticker: str, closes: list[float]) -> None:
    rows = []
    for index, close in enumerate(closes):
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
