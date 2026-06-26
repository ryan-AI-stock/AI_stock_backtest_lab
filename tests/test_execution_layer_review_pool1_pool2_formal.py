import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

import test_paths  # noqa: F401

from backtest_lab.execution_layer_review_pool1_pool2_formal import (
    FORMAL_MODEL_TARGET,
    run_execution_layer_review_pool1_pool2_formal,
)


class ExecutionLayerReviewPool1Pool2FormalTest(unittest.TestCase):
    def test_runner_adapts_absorbed_target_stream_without_formal_activation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate = root / "candidate"
            absorption = root / "absorption"
            prices = root / "prices"
            output = root / "out"
            candidate.mkdir()
            absorption.mkdir()
            prices.mkdir()

            dates = pd.date_range("2024-01-02", periods=5, freq="B")
            daily_rows = [
                _daily_row(dates[0], "{}", "cash", "hold", 0, 0),
                _daily_row(dates[1], '{"00631L.TW": 0.4}', "00631L.TW", "rebalance", 400_000, 570),
                _daily_row(dates[2], '{"2454.TW": 1.0}', "2454.TW", "switch", 900_000, 1200),
                _daily_row(dates[3], '{"00631L.TW": 0.4}', "00631L.TW", "switch", 850_000, 1100),
                _daily_row(dates[4], '{"00631L.TW": 0.4}', "00631L.TW", "hold", 0, 0),
            ]
            pd.DataFrame(daily_rows).to_csv(candidate / "daily_equity_by_variant.csv", index=False)
            pd.DataFrame(
                [
                    _event_row(dates[0], "00631L.TW", "2454.TW", True, "pool2_disagrees_confirmation_1_not_met", "{}"),
                    _event_row(dates[1], "00631L.TW", "00631L.TW", False, "pool1_primary", '{"00631L.TW": 0.4}'),
                    _event_row(dates[2], "2454.TW", "2454.TW", False, "pool1_primary", '{"2454.TW": 1.0}'),
                    _event_row(dates[3], "00631L.TW", "00631L.TW", False, "pool1_primary", '{"00631L.TW": 0.4}'),
                    _event_row(dates[4], "00631L.TW", "00631L.TW", False, "pool1_primary", '{"00631L.TW": 0.4}'),
                ]
            ).to_csv(candidate / "pool2_disagreement_variant_events.csv", index=False)
            pd.DataFrame(
                [
                    {"variant": "combined_cap40_confirmation1", "date": dates[1].strftime("%Y-%m-%d"), "ticker": "00631L.TW", "action": "buy"},
                    {"variant": "combined_cap40_confirmation1", "date": dates[2].strftime("%Y-%m-%d"), "ticker": "2454.TW", "action": "buy"},
                ]
            ).to_csv(candidate / "trade_ledger_by_variant.csv", index=False)
            (absorption / "manifest.json").write_text(
                json.dumps({"formal_model_target": FORMAL_MODEL_TARGET, "formal_absorption_ready": True}),
                encoding="utf-8",
            )
            _write_price(prices, "00631L.TW", [10, 11, 12, 13, 14, 15, 16])
            _write_price(prices, "2454.TW", [100, 99, 101, 102, 103, 104, 105])

            result = run_execution_layer_review_pool1_pool2_formal(
                candidate_dir=candidate,
                absorption_dir=absorption,
                price_cache_dir=prices,
                output_dir=output,
            )

            manifest = json.loads((result / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["formal_model_target"], FORMAL_MODEL_TARGET)
            self.assertFalse(manifest["formal_model_changed"])
            self.assertFalse(manifest["trade_decision_changed"])
            self.assertFalse(manifest["active_in_trade_decision"])
            self.assertFalse(manifest["execution_review_active_in_trade_decision"])
            self.assertFalse(manifest["pool3_shadow_used"])
            self.assertFalse(manifest["final_decision_label_used"])
            self.assertFalse(manifest["rr_partial_switch_activated"])
            self.assertFalse(manifest["uses_forward_return_as_rule"])
            self.assertFalse(manifest["next_day_ledger_mixed_with_same_day"])

            adapted = pd.read_csv(result / "formal_target_stream_adapter.csv")
            self.assertIn("winner_ticker", adapted.columns)
            self.assertEqual(adapted.loc[1, "winner_ticker"], "00631L.TW")
            self.assertFalse(adapted["pool3_shadow_used"].astype(bool).any())
            self.assertFalse(adapted["final_decision_label_used"].astype(bool).any())

            changes = pd.read_csv(result / "formal_target_change_panel.csv")
            self.assertGreaterEqual(len(changes), 3)
            self.assertTrue(changes["reversal_within_3_trading_rows"].astype(bool).any())

            next_day = pd.read_csv(result / "next_day_fill_readiness.csv")
            self.assertIn("next_day_slippage_pct", next_day.columns)
            self.assertTrue((next_day["readiness_state"] == "completed").any())
            self.assertFalse(next_day["next_day_ledger_mixed_with_same_day"].astype(bool).any())

            stability = pd.read_csv(result / "entry_target_stability_summary.csv").iloc[0]
            self.assertGreaterEqual(int(stability["confirmation1_blocked_rows"]), 1)
            self.assertFalse(bool(stability["active_in_trade_decision"]))

            readiness = pd.read_csv(result / "partial_execution_readiness.csv").iloc[0]
            self.assertFalse(bool(readiness["ready_for_formal_activation"]))
            self.assertFalse(bool(readiness["active_in_trade_decision"]))


def _daily_row(date: pd.Timestamp, weights: str, position: str, action: str, turnover: float, cost: float) -> dict:
    return {
        "variant": "combined_cap40_confirmation1",
        "date": date.strftime("%Y-%m-%d"),
        "period": "fixture",
        "target_weights": weights,
        "position_ticker": position,
        "cash": 1_000_000,
        "equity": 1_000_000,
        "drawdown": 0,
        "turnover": turnover,
        "transaction_cost": cost,
        "action": action,
        "data_status": "fixture",
    }


def _event_row(date: pd.Timestamp, pool1: str, pool2: str, disagreement: bool, reason: str, weights: str) -> dict:
    return {
        "variant": "combined_cap40_confirmation1",
        "date": date.strftime("%Y-%m-%d"),
        "period": "fixture",
        "pool1_vote": pool1,
        "pool2_vote": pool2,
        "pool2_disagreement": disagreement,
        "event_reason": reason,
        "target_weights": weights,
        "uses_forward_return_as_rule": False,
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
