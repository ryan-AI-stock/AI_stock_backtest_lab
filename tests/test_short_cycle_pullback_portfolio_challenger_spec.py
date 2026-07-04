import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

import test_paths  # noqa: F401

from backtest_lab.short_cycle_pullback_portfolio_challenger_spec import (
    run_short_cycle_pullback_portfolio_challenger_spec,
)


class ShortCyclePullbackPortfolioChallengerSpecTest(unittest.TestCase):
    def test_builds_diagnostic_contract_without_formalizing_event_signal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            event_dir = root / "events"
            event_dir.mkdir()
            formal_stream = root / "formal_stream.csv"
            output = root / "out"
            _write_event_panel_fixture(event_dir)
            _write_formal_stream_fixture(formal_stream)

            run_short_cycle_pullback_portfolio_challenger_spec(
                event_panel_dir=event_dir,
                formal_target_stream=formal_stream,
                output_dir=output,
            )

            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "completed_portfolio_challenger_contract_ready")
            self.assertEqual(manifest["eligible_event_rows"], 2)
            self.assertEqual(manifest["eligible_pool1b_rows"], 1)
            self.assertEqual(manifest["case_6488_two_rows"], 1)
            self.assertFalse(manifest["formal_model_changed"])
            self.assertFalse(manifest["trade_decision_changed"])
            self.assertFalse(manifest["active_in_trade_decision"])
            self.assertFalse(manifest["uses_forward_return_as_live_rule"])

            variants = pd.read_csv(output / "execution_rule_variants.csv")
            self.assertIn("ma20_reclaim_overlay_20_when_formal_cash_or_market_exposure", set(variants["variant_id"]))
            self.assertIn("ma20_reclaim_overlay_10_when_formal_cash_or_market_exposure", set(variants["variant_id"]))
            self.assertFalse(variants["same_day_allowed"].astype(bool).any())
            self.assertFalse(variants["active_in_trade_decision"].astype(bool).any())
            self.assertLessEqual(float(variants["sleeve_weight"].max()), 0.20)

            cost = pd.read_csv(output / "cost_model_contract.csv")
            self.assertIn("sell_stock", set(cost["scope"]))
            stock_sell = cost[cost["scope"].eq("sell_stock")].iloc[0]
            self.assertAlmostEqual(float(stock_sell["securities_transaction_tax_rate"]), 0.003)

            readiness = json.loads((output / "readiness_for_experiments.json").read_text(encoding="utf-8"))
            self.assertTrue(readiness["ready_for_experiments_portfolio_challenger_validation"])
            self.assertFalse(readiness["ready_for_strategy_replay"])
            self.assertTrue(readiness["diagnostic_only"])
            self.assertTrue(readiness["material_layer_case_only"])
            self.assertTrue(readiness["case_6488_two_case_only"])


def _write_event_panel_fixture(root: Path) -> None:
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "status": "completed_production_grade_diagnostic_event_panel",
                "future_data_violation_count": 0,
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            _event("2026-01-02", "2330.TW", "old_ai", "strong_stock_ma20_pullback_reclaim"),
            _event("2026-01-03", "6488.TWO", "pool1b", "pullback_candidate_wait_for_peer_breadth"),
            _event("2026-01-04", "2454.TW", "old_ai", "pullback_watch_then_confirm_2day"),
        ]
    ).to_csv(root / "short_cycle_pullback_reversal_event_panel.csv", index=False)


def _event(signal_date: str, ticker: str, source: str, variant: str) -> dict[str, object]:
    return {
        "signal_date": signal_date,
        "next_tradable_date": "2026-01-05",
        "ticker": ticker,
        "candidate_name": ticker,
        "variant_id": variant,
        "candidate_source": source,
        "price_data_ready": True,
        "diagnostic_only": True,
        "is_trade_rule": False,
        "uses_forward_return_as_live_rule": False,
        "rs_vs_0050_60d_pct": 10.0,
        "rs_vs_0050_20d_pct": 5.0,
        "peer_recovery_count": 3,
        "drawdown_from_60d_high_pct": -8.0,
        "supply_chain_layer": "Semiconductor materials" if ticker == "6488.TWO" else "",
    }


def _write_formal_stream_fixture(path: Path) -> None:
    pd.DataFrame(
        [
            {
                "signal_date": "2026-01-02",
                "execution_date": "2026-01-05",
                "formal_target": "CASH",
                "target_weights": "{}",
                "risk_off_state": "no_target_cash_all",
                "execution_action_basis": "next_day",
                "next_day_tradable_flag": True,
            }
        ]
    ).to_csv(path, index=False)


if __name__ == "__main__":
    unittest.main()
