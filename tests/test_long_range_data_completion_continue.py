import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

import test_paths  # noqa: F401

from backtest_lab.current_formal_pool1_pool2_signal_panels import POOL1_TICKERS, TW50_BENCHMARK
from backtest_lab.long_range_data_completion_continue import run_long_range_data_completion_continue


class LongRangeDataCompletionContinueTest(unittest.TestCase):
    def _price_frame(self, start: str = "2020-01-01", periods: int = 90, offset: float = 0.0) -> pd.DataFrame:
        dates = pd.bdate_range(start=start, periods=periods)
        close = [100.0 + offset + index for index in range(periods)]
        return pd.DataFrame(
            {
                "date": [date.strftime("%Y-%m-%d") for date in dates],
                "open": close,
                "high": close,
                "low": close,
                "close": close,
                "adj_close": close,
                "volume": [1000000] * periods,
                "dividend": [0.0] * periods,
                "stock_split": [0.0] * periods,
            }
        )

    def test_bounded_runner_keeps_partial_rows_and_blocks_combined_stream(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache = root / "cache"
            dynamic = root / "dynamic"
            previous = root / "previous"
            pool2 = root / "pool2"
            output = root / "out"
            for folder in (cache, dynamic, previous, pool2):
                folder.mkdir()

            for index, ticker in enumerate(sorted(set(POOL1_TICKERS) | {TW50_BENCHMARK})):
                self._price_frame(offset=float(index)).to_csv(cache / f"{ticker.replace('.', '_')}.csv", index=False)

            pd.DataFrame(
                [
                    {"signal_date": "2020-01-02", "available_universe_count": 7, "candidate_tickers": "00631L.TW|2330.TW"},
                    {
                        "signal_date": "2020-01-03",
                        "available_universe_count": len(POOL1_TICKERS),
                        "candidate_tickers": "|".join(POOL1_TICKERS),
                    },
                ]
            ).to_csv(dynamic / "dynamic_universe_state_replay_coverage.csv", index=False)

            pd.DataFrame(
                [
                    {
                        "signal_date": "2020-01-03",
                        "pool1_target": "00631L.TW",
                        "pool1_target_weights": '{"00631L.TW": 1.0}',
                        "source_formal_ready": True,
                    }
                ]
            ).to_csv(previous / "pool1_full_state_replayed_signals.csv", index=False)

            pd.DataFrame(
                [
                    {
                        "date": "2020-01-02",
                        "candidate_ticker": "2330.TW",
                        "candidate_name": "TSMC",
                        "score": 1.0,
                        "rank": 1,
                        "raw_rank": 1,
                        "eligible_for_pool_selection": False,
                        "confirmation_state": "not_sufficiently_confirmed",
                        "market_exposure_support": "not_sufficiently_confirmed",
                    }
                ]
            ).to_csv(pool2 / "pool2_daily_confirmation_panel.csv", index=False)

            result = run_long_range_data_completion_continue(
                dynamic_universe_dir=dynamic,
                pool1_previous_dir=previous,
                pool2_panel_dir=pool2,
                price_cache_dir=cache,
                price_source_registry=root / "missing_registry.csv",
                output_dir=output,
                start_date="2020-01-02",
                end_date="2020-01-03",
            )

            self.assertEqual(result, output)
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["pool1_rows"], 1)
            self.assertEqual(manifest["pool1_blocked_rows"], 1)
            self.assertFalse(manifest["combined_formal_ready"])
            self.assertFalse(manifest["formal_model_changed"])
            self.assertFalse(manifest["trade_decision_changed"])
            self.assertFalse(manifest["no_target_cash_all_applied_to_2014_2021"])

            attempts = pd.read_csv(output / "pool1_segment_replay_attempts.csv")
            self.assertIn("blocked_requires_dynamic_state_adapter", set(attempts["status"].astype(str)))

            pool2_blockers = pd.read_csv(output / "blocker_by_pool2_field.csv")
            self.assertEqual(pool2_blockers.iloc[0]["blocker"], "no_eligible_pool2_rows_201411_202112")

            combined = pd.read_csv(output / "combined_formal_target_stream_201411_202112.csv")
            self.assertTrue(combined.empty)


if __name__ == "__main__":
    unittest.main()
