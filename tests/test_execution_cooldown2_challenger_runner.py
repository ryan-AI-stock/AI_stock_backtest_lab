import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

import test_paths  # noqa: F401

from backtest_lab.execution_cooldown2_challenger_runner import run_execution_cooldown2_challenger_runner


class ExecutionCooldown2ChallengerRunnerTest(unittest.TestCase):
    def test_runner_builds_production_grade_cooldown2_outputs(self) -> None:
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
                    _stream_row(dates[6], "{}" , action="sell", turnover=900_000),
                    _stream_row(dates[7], '{"2454.TW": 1.0}', action="buy", turnover=900_000),
                    _stream_row(dates[8], '{"2454.TW": 1.0}'),
                    _stream_row(dates[9], '{"00631L.TW": 0.4}', action="switch", turnover=900_000),
                    _stream_row(dates[10], '{"00631L.TW": 0.4}'),
                    _stream_row(dates[11], '{"00631L.TW": 0.4}'),
                ]
            )
            stream.to_csv(review / "formal_target_stream_adapter.csv", index=False)
            _write_price(prices, "00631L.TW", [10, 11, 12, 13, 12, 14, 15, 16, 17, 18, 19, 20, 21, 22])
            _write_price(prices, "2454.TW", [100, 99, 101, 103, 104, 102, 105, 106, 108, 109, 111, 110, 112, 113])

            result = run_execution_cooldown2_challenger_runner(
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
            self.assertFalse(manifest["simplified_experiments_ledger_used_for_formal_performance"])

            matrix = pd.read_csv(result / "variant_parameter_matrix.csv")
            self.assertEqual(
                set(matrix["variant_id"]),
                {
                    "selector_full_switch_same_day_reference",
                    "selector_full_switch_next_day_baseline",
                    "cooldown_after_switch_2",
                    "cooldown_after_switch_3",
                },
            )
            self.assertFalse(matrix["active_in_trade_decision"].astype(bool).any())

            daily = pd.read_csv(result / "daily_equity_by_variant.csv")
            self.assertIn("portfolio_equity", daily.columns)
            self.assertFalse(daily["execution_diagnostic_active_in_trade_decision"].astype(bool).any())

            weights = pd.read_csv(result / "portfolio_weight_ledger_by_variant.csv")
            self.assertFalse(weights["active_in_trade_decision"].astype(bool).any())

            cash = pd.read_csv(result / "cash_ledger_by_variant.csv")
            self.assertTrue(cash["cash_non_negative"].astype(bool).all())

            cooldown_events = pd.read_csv(result / "cooldown_event_panel.csv")
            self.assertIn("blocked_switch", set(cooldown_events["state"]))

            exposure = pd.read_csv(result / "exposure_integrity_checks.csv")
            self.assertTrue(exposure["integrity_pass"].astype(bool).all())

            caveat = pd.read_csv(result / "hard_gate_2024_benchmark_caveat.csv")
            self.assertTrue((caveat["caveat_state"] == "retained_not_resolved").all())

            for name in (
                "daily_equity_by_variant.csv",
                "trade_ledger_by_variant.csv",
                "blocked_fill_events.csv",
                "same_day_vs_next_day_alignment.csv",
                "execution_cooldown2_challenger_summary_zh.md",
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
