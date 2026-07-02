import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

import test_paths  # noqa: F401

from backtest_lab.current_formal_pool1_pool2_signal_panels import POOL1_TICKERS, TW50_BENCHMARK
from backtest_lab.pool1_full_state_replay_201411_dynamic_universe import (
    run_pool1_full_state_replay_201411_dynamic_universe,
)


class Pool1FullStateReplay201411DynamicUniverseTest(unittest.TestCase):
    def _price_frame(self, start: str = "2021-01-01", periods: int = 390, offset: float = 0.0) -> pd.DataFrame:
        dates = pd.bdate_range(start=start, periods=periods)
        close = [100.0 + offset + index * (1.0 + offset / 100.0) for index in range(periods)]
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

    def test_replays_static_segment_and_blocks_pre_static_dynamic_segment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache = root / "cache"
            dynamic = root / "dynamic"
            score = root / "score"
            equivalence = root / "equivalence"
            output = root / "out"
            for folder in (cache, dynamic, score, equivalence):
                folder.mkdir()

            tickers = sorted(set(POOL1_TICKERS) | {TW50_BENCHMARK})
            for index, ticker in enumerate(tickers):
                self._price_frame(offset=float(index)).to_csv(cache / f"{ticker.replace('.', '_')}.csv", index=False)

            pd.DataFrame(
                [
                    {
                        "signal_date": "2022-03-31",
                        "available_universe_count": 7,
                        "candidate_tickers": ",".join(tickers[:7]),
                    },
                    {
                        "signal_date": "2022-04-01",
                        "available_universe_count": len(POOL1_TICKERS),
                        "candidate_tickers": ",".join(POOL1_TICKERS),
                    },
                    {
                        "signal_date": "2022-04-04",
                        "available_universe_count": len(POOL1_TICKERS),
                        "candidate_tickers": ",".join(POOL1_TICKERS),
                    },
                ]
            ).to_csv(dynamic / "dynamic_universe_state_replay_coverage.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "signal_date": "2022-03-31",
                        "top_ticker": "2330.TW",
                        "fallback_0050_score_ready": True,
                        "raw_dynamic_attack_gate_pass": False,
                    }
                ]
            ).to_csv(score / "dynamic_score_margin_panel.csv", index=False)
            (equivalence / "manifest.json").write_text(
                json.dumps({"equivalence_pass": True}, ensure_ascii=False),
                encoding="utf-8",
            )

            result = run_pool1_full_state_replay_201411_dynamic_universe(
                dynamic_universe_dir=dynamic,
                score_margin_dir=score,
                equivalence_dir=equivalence,
                price_cache_dir=cache,
                price_source_registry=root / "missing_registry.csv",
                output_dir=output,
                start_date="2022-03-31",
                end_date="2022-04-04",
            )

            self.assertEqual(result, output)
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertTrue(manifest["pool1_full_state_replay_formal_ready_static_segment"])
            self.assertFalse(manifest["pool1_full_state_replay_formal_ready_full_period"])
            self.assertFalse(manifest["formal_model_changed"])
            self.assertFalse(manifest["trade_decision_changed"])
            self.assertFalse(manifest["no_target_cash_all_applied_to_2014_2021"])

            blocked = pd.read_csv(output / "blocked_signal_rows.csv")
            self.assertEqual(len(blocked), 1)
            self.assertFalse(bool(blocked.iloc[0]["source_formal_ready"]))

            replayed = pd.read_csv(output / "pool1_full_state_replayed_signals.csv")
            self.assertGreaterEqual(len(replayed), 1)
            self.assertTrue(replayed["source_formal_ready"].astype(bool).all())


if __name__ == "__main__":
    unittest.main()
