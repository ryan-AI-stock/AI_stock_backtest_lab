from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import test_paths  # noqa: F401

from backtest_lab.stock_pool_historical_replay import (
    _forward_metrics,
    run_stock_pool_historical_replay,
)
from backtest_lab.stock_pool_store import symbol_entry


class StockPoolHistoricalReplayTest(unittest.TestCase):
    def test_forward_metrics_calculates_return_drawdown_and_runup(self) -> None:
        dates = pd.bdate_range("2025-01-02", periods=30)
        frame = _price_frame(dates, [100, 105, 102, 110, 108] + [112 + i for i in range(25)])

        metrics = _forward_metrics(frame, signal_date="2025-01-02", horizon=4)

        self.assertEqual(metrics["forward_status"], "ready")
        self.assertAlmostEqual(metrics["forward_return"], 0.08)
        self.assertAlmostEqual(metrics["max_drawdown"], -0.02857143)
        self.assertAlmostEqual(metrics["max_runup"], 0.10)
        self.assertEqual(metrics["end_date"], dates[4].strftime("%Y-%m-%d"))

    def test_replay_runner_writes_contract_outputs_without_changing_formal_model(self) -> None:
        dates = pd.bdate_range("2025-01-02", periods=180)
        prices = {
            "00631L.TW": _trend_frame(dates, start=100, step=0.8),
        }
        pool = {
            "pool_id": "custom_etf_pool",
            "name": "自訂ETF觀察池",
            "strategy_preset": "universal_pool_custom",
            "operational_observation": True,
            "resolved_symbols": [
                symbol_entry("00631L.TW", source="manual"),
            ],
        }

        class StoreStub:
            def __init__(self, path: str | Path) -> None:
                self.path = path

            def list_pools(self) -> list[dict]:
                return [pool]

        def load_prices(**kwargs):
            return {ticker: prices[ticker] for ticker in kwargs["tickers"] if ticker in prices}, []

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "replay"
            with (
                patch("backtest_lab.stock_pool_historical_replay.StockPoolStore", StoreStub),
                patch("backtest_lab.stock_pool_historical_replay._load_observation_price_frames", side_effect=load_prices),
                patch("backtest_lab.stock_pool_historical_replay.load_first_available_risk_factors", return_value=({}, {})),
            ):
                result = run_stock_pool_historical_replay(
                    output_dir=output_dir,
                    periods={"test": ("2025-06-02", "2025-06-02")},
                    warmup_start="2025-01-02",
                    cache_only=False,
                    max_dates=1,
                )

            self.assertEqual(result.replay_rows, 1)
            self.assertGreaterEqual(result.candidate_rows, 1)
            self.assertTrue((output_dir / "stock_pool_replay_panel.csv").exists())
            self.assertTrue((output_dir / "stock_pool_replay_top_candidates.csv").exists())
            self.assertTrue((output_dir / "stock_pool_replay_forward_returns.csv").exists())
            metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["status"], "completed")
            self.assertIn("selection_layer", metadata["contract_fields"]["replay_panel"])
            panel = pd.read_csv(output_dir / "stock_pool_replay_panel.csv")
            self.assertEqual(panel.loc[0, "pool_id"], "custom_etf_pool")
            self.assertEqual(panel.loc[0, "status"], "generated")
            self.assertIn(panel.loc[0, "selection_layer"], {"market_exposure_tool", "no_selection"})


def _trend_frame(dates: pd.DatetimeIndex, *, start: float, step: float) -> pd.DataFrame:
    values = [start + index * step for index in range(len(dates))]
    return _price_frame(dates, values)


def _price_frame(dates: pd.DatetimeIndex, values: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": values,
            "high": [value * 1.01 for value in values],
            "low": [value * 0.99 for value in values],
            "close": values,
            "adj_close": values,
            "volume": [20_000_000] * len(values),
        },
        index=dates,
    )


if __name__ == "__main__":
    unittest.main()
