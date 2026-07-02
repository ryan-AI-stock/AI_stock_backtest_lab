import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

import test_paths  # noqa: F401

from backtest_lab.execution_layer_next_day_ab_pool1_pool2_formal import (
    FORMAL_MODEL_TARGET,
    VariantSpec,
    run_execution_layer_next_day_ab_pool1_pool2_formal,
    _simulate_variant,
)


class ExecutionLayerNextDayABPool1Pool2FormalTest(unittest.TestCase):
    def test_runner_builds_next_day_full_equity_and_keeps_diagnostic_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            review = root / "review"
            prices = root / "prices"
            output = root / "output"
            review.mkdir()
            prices.mkdir()

            dates = pd.date_range("2024-01-02", periods=10, freq="B")
            stream = pd.DataFrame(
                [
                    _stream_row(dates[0], "{}"),
                    _stream_row(dates[1], '{"00631L.TW": 0.4}'),
                    _stream_row(dates[2], '{"00631L.TW": 0.4}'),
                    _stream_row(dates[3], '{"2454.TW": 1.0}'),
                    _stream_row(dates[4], '{"2454.TW": 1.0}'),
                    _stream_row(dates[5], '{"2454.TW": 1.0}'),
                    _stream_row(dates[6], "{}"),
                    _stream_row(dates[7], '{"00631L.TW": 0.4}'),
                    _stream_row(dates[8], '{"00631L.TW": 0.4}'),
                    _stream_row(dates[9], '{"00631L.TW": 0.4}'),
                ]
            )
            stream.to_csv(review / "formal_target_stream_adapter.csv", index=False)
            _write_price(prices, "00631L.TW", [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21])
            _write_price(prices, "2454.TW", [100, 101, 99, 102, 103, 104, 105, 106, 107, 108, 109, 110])

            result = run_execution_layer_next_day_ab_pool1_pool2_formal(
                review_dir=review,
                price_cache_dir=prices,
                output_dir=output,
            )

            manifest = json.loads((result / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["formal_model_target"], FORMAL_MODEL_TARGET)
            self.assertFalse(manifest["formal_model_changed"])
            self.assertFalse(manifest["trade_decision_changed"])
            self.assertFalse(manifest["active_in_trade_decision"])
            self.assertFalse(manifest["execution_diagnostic_active_in_trade_decision"])
            self.assertFalse(manifest["formal_execution_layer_activated"])
            self.assertFalse(manifest["same_day_and_next_day_mixed"])
            self.assertFalse(manifest["uses_forward_return_as_rule"])

            matrix = pd.read_csv(result / "variant_parameter_matrix.csv")
            self.assertIn("next_day_minimum_hold_3", matrix["variant_id"].tolist())
            self.assertFalse(matrix["active_in_trade_decision"].astype(bool).any())

            daily = pd.read_csv(result / "next_day_fill_full_equity_ledger.csv")
            self.assertIn("portfolio_equity", daily.columns)
            self.assertIn("weight_sum", daily.columns)
            self.assertFalse(daily["execution_diagnostic_active_in_trade_decision"].astype(bool).any())
            weights = pd.to_numeric(daily["weight_sum"], errors="coerce").dropna()
            self.assertTrue(((weights >= 0.99) & (weights <= 1.01)).all())

            trades = pd.read_csv(result / "next_day_fill_trade_ledger.csv")
            self.assertIn("signal_date", trades.columns)
            for column in (
                "buy_fee",
                "sell_fee",
                "securities_transaction_tax",
                "total_transaction_cost",
                "cost_model_version",
            ):
                self.assertIn(column, trades.columns)
            self.assertTrue((pd.to_numeric(trades["total_transaction_cost"], errors="coerce") == pd.to_numeric(trades["transaction_cost"], errors="coerce")).all())
            self.assertTrue((pd.to_numeric(trades["securities_transaction_tax"], errors="coerce").fillna(0) >= 0).all())
            self.assertFalse(trades["execution_diagnostic_active_in_trade_decision"].astype(bool).any())

            events = pd.read_csv(result / "fill_event_panel.csv")
            next_day_events = events[events["variant_id"] == "next_day_full_rotation"]
            self.assertTrue((pd.to_datetime(next_day_events["fill_date"]) > pd.to_datetime(next_day_events["signal_date"])).any())

            min_hold_events = events[events["variant_id"] == "next_day_minimum_hold_3"]
            self.assertTrue((pd.to_numeric(min_hold_events["minimum_hold_rows"], errors="coerce") == 3).any())

            summary = pd.read_csv(result / "minimum_hold_cooldown_ab_summary.csv")
            self.assertIn("next_day_full_rotation", summary["variant_id"].tolist())
            self.assertFalse(summary["active_in_trade_decision"].astype(bool).any())

            readiness = pd.read_csv(result / "execution_ab_readiness_report.csv").iloc[0]
            self.assertFalse(bool(readiness["formal_activation_ready"]))
            self.assertFalse(bool(readiness["active_in_trade_decision"]))

    def test_no_formal_target_holds_previous_target_instead_of_selling_to_cash(self) -> None:
        dates = pd.to_datetime(["2026-06-05", "2026-06-08", "2026-06-09", "2026-06-10"])
        frame = pd.DataFrame(
            [
                _stream_row(dates[0], '{"2454.TW": 1.0}'),
                _stream_row(dates[1], "{}"),
                _stream_row(dates[2], '{"00631L.TW": 1.0}'),
                _stream_row(dates[3], '{"00631L.TW": 1.0}'),
            ]
        )
        frame.loc[0, "action"] = "switch"
        frame.loc[0, "turnover"] = 1
        frame.loc[1, "action"] = "switch"
        frame.loc[1, "turnover"] = 1
        frame.loc[2, "action"] = "switch"
        frame.loc[2, "turnover"] = 1
        prices = {
            "2454.TW": pd.Series([100.0, 95.0, 90.0, 88.0], index=dates),
            "00631L.TW": pd.Series([50.0, 52.0, 54.0, 56.0], index=dates),
        }

        daily, trades, _events, blocked = _simulate_variant(
            frame,
            prices,
            VariantSpec("regression_next_day", 1),
            1_000_000.0,
        )

        no_target_trades = trades[trades["signal_date"].astype(str).eq("2026-06-08")] if not trades.empty else pd.DataFrame()
        self.assertTrue(no_target_trades.empty)
        self.assertIn("no_formal_target_hold_previous", set(blocked["blocked_reason"]))
        row = daily[daily["date"].astype(str).eq("2026-06-09")].iloc[0]
        self.assertEqual(row["top_holding"], "2454.TW")
        self.assertEqual(json.loads(row["accepted_target_weights"]), {"2454.TW": 1.0})

    def test_explicit_no_formal_target_exit_to_cash_challenger_can_sell(self) -> None:
        dates = pd.to_datetime(["2026-06-05", "2026-06-08", "2026-06-09"])
        frame = pd.DataFrame(
            [
                _stream_row(dates[0], '{"2454.TW": 1.0}'),
                _stream_row(dates[1], "{}"),
                _stream_row(dates[2], "{}"),
            ]
        )
        frame.loc[0, "action"] = "switch"
        frame.loc[0, "turnover"] = 1
        frame.loc[1, "action"] = "switch"
        frame.loc[1, "turnover"] = 1
        prices = {"2454.TW": pd.Series([100.0, 95.0, 90.0], index=dates)}

        daily, trades, _events, blocked = _simulate_variant(
            frame,
            prices,
            VariantSpec("explicit_cash_challenger", 1, no_formal_target_policy="exit_to_cash"),
            1_000_000.0,
        )

        no_target_sell = trades[
            trades["signal_date"].astype(str).eq("2026-06-08")
            & trades["action"].astype(str).eq("sell")
        ]
        self.assertFalse(no_target_sell.empty)
        self.assertTrue(blocked.empty)
        row = daily[daily["date"].astype(str).eq("2026-06-09")].iloc[0]
        self.assertEqual(row["top_holding"], "cash")


def _stream_row(date: pd.Timestamp, target_weights: str) -> dict:
    return {
        "date": date.strftime("%Y-%m-%d"),
        "period": "fixture",
        "formal_model_target": FORMAL_MODEL_TARGET,
        "source_panel_variant": "combined_cap40_confirmation1",
        "winner_ticker": "",
        "formal_target": "",
        "target_weights": target_weights,
        "position_ticker": "cash",
        "turnover": 0,
        "transaction_cost": 0,
        "action": "hold",
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
