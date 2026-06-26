import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

import test_paths  # noqa: F401

from backtest_lab.execution_position_transition_runner import run_execution_position_transition_runner


class ExecutionPositionTransitionRunnerTest(unittest.TestCase):
    def test_runner_builds_position_sizing_transition_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prices = root / "prices"
            output = root / "output"
            prices.mkdir()
            dates = pd.date_range("2024-01-02", periods=30, freq="B")
            stream = pd.DataFrame(
                [
                    _row(dates[0], "{}"),
                    _row(dates[1], '{"00631L.TW": 0.4}'),
                    _row(dates[2], '{"00631L.TW": 0.4}'),
                    _row(dates[3], '{"2454.TW": 1.0}'),
                    _row(dates[4], '{"2454.TW": 1.0}'),
                    _row(dates[5], '{"2454.TW": 1.0}'),
                    _row(dates[6], '{"00631L.TW": 0.4}'),
                    _row(dates[7], '{"00631L.TW": 0.4}'),
                    _row(dates[8], "{}"),
                    _row(dates[9], '{"2454.TW": 1.0}'),
                ]
            )
            stream.to_csv(root / "stream.csv", index=False)
            _write_price(prices, "00631L.TW", [10 + i * 0.3 for i in range(40)])
            _write_price(prices, "2454.TW", [100 + i * 0.8 for i in range(40)])

            result = run_execution_position_transition_runner(
                formal_target_stream=root / "stream.csv",
                price_cache_dir=prices,
                output_dir=output,
            )

            manifest = json.loads((result / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["execution_layer_status"], "diagnostic_challenger")
            self.assertEqual(manifest["replay_timing_basis"], "same_day_formal_replay_口徑")
            self.assertFalse(manifest["formal_execution_layer_activated"])
            self.assertFalse(manifest["formal_model_changed"])
            self.assertFalse(manifest["trade_decision_changed"])
            self.assertFalse(manifest["uses_forward_return_as_rule"])

            matrix = pd.read_csv(result / "variant_parameter_matrix.csv")
            self.assertIn("full_switch_100_same_day_baseline", matrix["variant_id"].tolist())
            self.assertIn("partial_switch_10_on_change", matrix["variant_id"].tolist())
            self.assertIn("partial_switch_30_on_change", matrix["variant_id"].tolist())
            self.assertIn("partial_switch_50_on_change", matrix["variant_id"].tolist())
            self.assertIn("partial_switch_100_on_change", matrix["variant_id"].tolist())
            self.assertEqual(
                10,
                int(matrix["variant_id"].astype(str).str.startswith("partial_switch_").sum()),
            )
            self.assertIn("staged_switch_50_then_100_confirm_2", matrix["variant_id"].tolist())
            self.assertIn("hold_old_if_still_valid_ma20", matrix["variant_id"].tolist())

            daily = pd.read_csv(result / "position_transition_daily_ledger.csv")
            self.assertIn("portfolio_equity", daily.columns)
            self.assertFalse(daily["execution_diagnostic_active_in_trade_decision"].astype(bool).any())

            events = pd.read_csv(result / "position_transition_event_panel.csv")
            self.assertIn("partial_switch_30", set(events["execution_action_type"]))
            self.assertIn("staged_first_50", set(events["execution_action_type"]))
            self.assertIn("forced_exit_to_cash", set(events["execution_action_type"]))

            exposure = pd.read_csv(result / "exposure_integrity_checks.csv")
            self.assertTrue(exposure["integrity_pass"].astype(bool).all())

            performance = pd.read_csv(result / "period_performance_by_variant.csv")
            self.assertIn("full", set(performance["period_label"]))
            sizing = pd.read_csv(result / "sizing_curve_report.csv")
            partial_curve = sizing[sizing["sizing_family"].eq("partial_switch_curve")]
            self.assertEqual(set(range(10, 101, 10)), set(partial_curve["switch_pct"].astype(int)))
            self.assertTrue((result / "execution_position_transition_summary_zh.md").exists())


def _row(date: pd.Timestamp, weights: str) -> dict:
    parsed = json.loads(weights)
    target = next(iter(parsed)) if parsed else ""
    return {
        "date": date.strftime("%Y-%m-%d"),
        "period": "2024_hard_gate",
        "target_weights": weights,
        "formal_target": target,
        "winner_ticker": target,
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
