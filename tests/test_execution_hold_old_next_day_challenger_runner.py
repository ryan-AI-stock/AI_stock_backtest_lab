import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

import test_paths  # noqa: F401

from backtest_lab.execution_hold_old_next_day_challenger_runner import run_execution_hold_old_next_day_challenger_runner


class ExecutionHoldOldNextDayChallengerRunnerTest(unittest.TestCase):
    def test_runner_builds_hold_old_next_day_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            review = root / "review"
            prices = root / "prices"
            output = root / "output"
            review.mkdir()
            prices.mkdir()

            dates = pd.date_range("2024-01-02", periods=28, freq="B")
            stream = pd.DataFrame(
                [
                    _stream_row(dates[0], "{}"),
                    _stream_row(dates[1], '{"00631L.TW": 0.4}', action="buy", turnover=400_000),
                    _stream_row(dates[2], '{"00631L.TW": 0.4}'),
                    _stream_row(dates[3], '{"00631L.TW": 0.4}'),
                    _stream_row(dates[4], '{"00631L.TW": 0.4}'),
                    _stream_row(dates[5], '{"00631L.TW": 0.4}'),
                    _stream_row(dates[6], '{"00631L.TW": 0.4}'),
                    _stream_row(dates[7], '{"00631L.TW": 0.4}'),
                    _stream_row(dates[8], '{"00631L.TW": 0.4}'),
                    _stream_row(dates[9], '{"00631L.TW": 0.4}'),
                    _stream_row(dates[10], '{"00631L.TW": 0.4}'),
                    _stream_row(dates[11], '{"00631L.TW": 0.4}'),
                    _stream_row(dates[12], '{"00631L.TW": 0.4}'),
                    _stream_row(dates[13], '{"00631L.TW": 0.4}'),
                    _stream_row(dates[14], '{"00631L.TW": 0.4}'),
                    _stream_row(dates[15], '{"00631L.TW": 0.4}'),
                    _stream_row(dates[16], '{"00631L.TW": 0.4}'),
                    _stream_row(dates[17], '{"00631L.TW": 0.4}'),
                    _stream_row(dates[18], '{"00631L.TW": 0.4}'),
                    _stream_row(dates[19], '{"00631L.TW": 0.4}'),
                    _stream_row(dates[20], '{"2454.TW": 1.0}', action="switch", turnover=900_000),
                    _stream_row(dates[21], '{"2454.TW": 1.0}'),
                    _stream_row(dates[22], '{"00631L.TW": 0.4}', action="switch", turnover=900_000),
                    _stream_row(dates[23], '{"00631L.TW": 0.4}'),
                    _stream_row(dates[24], "{}" , action="sell", turnover=900_000),
                    _stream_row(dates[25], '{"2454.TW": 1.0}', action="buy", turnover=900_000),
                    _stream_row(dates[26], '{"2454.TW": 1.0}'),
                    _stream_row(dates[27], '{"2454.TW": 1.0}'),
                ]
            )
            stream.to_csv(review / "formal_target_stream_adapter.csv", index=False)
            _write_price(prices, "00631L.TW", [10 + index * 0.5 for index in range(35)])
            _write_price(prices, "2454.TW", [100 + index for index in range(35)])

            result = run_execution_hold_old_next_day_challenger_runner(
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
                    "hold_old_if_still_valid_ma20_next_day",
                },
            )
            self.assertFalse(matrix["active_in_trade_decision"].astype(bool).any())

            events = pd.read_csv(result / "hold_old_event_panel.csv")
            self.assertIn("hold_old_if_still_valid_ma20_next_day", set(events["variant_id"]))

            reason = pd.read_csv(result / "old_holding_still_valid_reason_trace.csv")
            self.assertIn("hold_old_still_valid", set(reason["policy_state"]))

            cash = pd.read_csv(result / "cash_ledger_by_variant.csv")
            self.assertTrue(cash["cash_non_negative"].astype(bool).all())

            exposure = pd.read_csv(result / "exposure_integrity_checks.csv")
            self.assertTrue(exposure["integrity_pass"].astype(bool).all())

            for name in (
                "daily_equity_by_variant.csv",
                "portfolio_weight_ledger_by_variant.csv",
                "trade_ledger_by_variant.csv",
                "same_day_vs_next_day_alignment.csv",
                "contribution_concentration_by_variant.csv",
                "execution_hold_old_next_day_challenger_summary_zh.md",
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
