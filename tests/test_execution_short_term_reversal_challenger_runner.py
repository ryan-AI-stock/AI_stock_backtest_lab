import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

import test_paths  # noqa: F401

from backtest_lab.execution_short_term_reversal_challenger_runner import run_execution_short_term_reversal_challenger_runner


class ExecutionShortTermReversalChallengerRunnerTest(unittest.TestCase):
    def test_runner_builds_short_term_reversal_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            review = root / "review"
            prices = root / "prices"
            output = root / "output"
            review.mkdir()
            prices.mkdir()

            dates = pd.date_range("2024-01-02", periods=12, freq="B")
            stream = pd.DataFrame(
                [
                    _stream_row(dates[0], "{}"),
                    _stream_row(dates[1], '{"00631L.TW": 0.4}', action="buy", turnover=400_000),
                    _stream_row(dates[2], '{"00631L.TW": 0.4}'),
                    _stream_row(dates[3], '{"2454.TW": 1.0}', action="switch", turnover=900_000),
                    _stream_row(dates[4], '{"00631L.TW": 0.4}', action="switch", turnover=900_000),
                    _stream_row(dates[5], '{"2454.TW": 1.0}', action="switch", turnover=900_000),
                    _stream_row(dates[6], '{"2454.TW": 1.0}'),
                    _stream_row(dates[7], '{"2454.TW": 1.0}'),
                    _stream_row(dates[8], "{}" , action="sell", turnover=900_000),
                    _stream_row(dates[9], '{"2454.TW": 1.0}', action="buy", turnover=900_000),
                    _stream_row(dates[10], '{"2454.TW": 1.0}'),
                    _stream_row(dates[11], '{"2454.TW": 1.0}'),
                ]
            )
            stream.to_csv(review / "formal_target_stream_adapter.csv", index=False)
            _write_price(prices, "00631L.TW", [10, 11, 12, 13, 12, 14, 15, 16, 17, 18, 19, 20, 21, 22])
            _write_price(prices, "2454.TW", [100, 99, 101, 103, 104, 102, 105, 106, 108, 109, 111, 110, 112, 113])

            result = run_execution_short_term_reversal_challenger_runner(
                review_dir=review,
                price_cache_dir=prices,
                output_dir=output,
            )

            manifest = json.loads((result / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["execution_layer_status"], "challenger")
            self.assertTrue(manifest["production_grade_next_day_ledger"])
            self.assertFalse(manifest["formal_execution_layer_activated"])
            self.assertFalse(manifest["formal_model_changed"])
            self.assertFalse(manifest["trade_decision_changed"])
            self.assertFalse(manifest["uses_forward_return_as_rule"])
            self.assertFalse(manifest["same_day_and_next_day_mixed"])

            matrix = pd.read_csv(result / "variant_parameter_matrix.csv")
            self.assertEqual(
                set(matrix["variant_id"]),
                {
                    "selector_full_switch_same_day_reference",
                    "selector_full_switch_next_day_baseline",
                    "confirm_2_before_switch_next_day",
                    "confirm_3_before_switch_next_day",
                    "partial_50_until_confirm_2_next_day",
                    "partial_75_until_confirm_2_next_day",
                },
            )
            self.assertFalse(matrix["active_in_trade_decision"].astype(bool).any())

            reversal = pd.read_csv(result / "short_term_reversal_event_panel.csv")
            self.assertGreaterEqual(len(reversal), 1)
            self.assertFalse(reversal["active_in_trade_decision"].astype(bool).any())

            policy = pd.read_csv(result / "reversal_policy_daily_panel.csv")
            self.assertIn("wait_for_confirmation", set(policy["policy_state"]))
            self.assertIn("partial_until_confirmed", set(policy["policy_state"]))

            cash = pd.read_csv(result / "cash_ledger_by_variant.csv")
            self.assertTrue(cash["cash_non_negative"].astype(bool).all())

            exposure = pd.read_csv(result / "exposure_integrity_checks.csv")
            self.assertTrue(exposure["integrity_pass"].astype(bool).all())

            for name in (
                "daily_equity_by_variant.csv",
                "portfolio_weight_ledger_by_variant.csv",
                "trade_ledger_by_variant.csv",
                "same_day_vs_next_day_alignment.csv",
                "execution_short_term_reversal_challenger_summary_zh.md",
                "completed.csv",
                "failed.csv",
            ):
                self.assertTrue((result / name).exists(), name)


def _stream_row(date: pd.Timestamp, target_weights: str, *, action: str = "hold", turnover: float = 0.0) -> dict:
    target = ""
    parsed = json.loads(target_weights)
    if parsed:
        target = next(iter(parsed))
    return {
        "date": date.strftime("%Y-%m-%d"),
        "period": "2024_hard_gate",
        "formal_model_target": "combined_cap40_confirmation1_base",
        "source_panel_variant": "combined_cap40_confirmation1",
        "winner_ticker": target,
        "formal_target": target,
        "target_weights": target_weights,
        "position_ticker": target or "cash",
        "equity": 1_000_000.0,
        "drawdown": 0.0,
        "turnover": turnover,
        "transaction_cost": 0.0,
        "action": action,
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
