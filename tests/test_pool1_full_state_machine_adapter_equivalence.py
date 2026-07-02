import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

import test_paths  # noqa: F401

from backtest_lab.current_formal_pool1_pool2_signal_panels import POOL1_TICKERS, TW50_BENCHMARK
from backtest_lab.pool1_full_state_machine_adapter_equivalence import (
    run_pool1_full_state_machine_adapter_equivalence,
)


class Pool1FullStateMachineAdapterEquivalenceTest(unittest.TestCase):
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

    def test_full_state_adapter_matches_reference_without_changing_formal_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache = root / "cache"
            formal = root / "formal"
            previous = root / "previous"
            output = root / "out"
            cache.mkdir()
            formal.mkdir()
            previous.mkdir()

            tickers = sorted(set(POOL1_TICKERS) | {TW50_BENCHMARK})
            for index, ticker in enumerate(tickers):
                self._price_frame(offset=float(index)).to_csv(cache / f"{ticker.replace('.', '_')}.csv", index=False)

            pd.DataFrame(
                [
                    {
                        "signal_date": "2022-04-01",
                        "formal_target": "00631L.TW",
                        "risk_off_state": "formal_target_active",
                        "pool1_top_candidate": "00631L.TW",
                    },
                    {
                        "signal_date": "2022-04-04",
                        "formal_target": "00631L.TW",
                        "risk_off_state": "formal_target_active",
                        "pool1_top_candidate": "00631L.TW",
                    },
                ]
            ).to_csv(formal / "formal_long_range_target_stream.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "equivalence_pass": False,
                        "mismatch_rows": 2,
                        "gate_state_match_rate": 0.0,
                        "target_match_rate": 0.0,
                    }
                ]
            ).to_csv(previous / "equivalence_summary.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "signal_date": "2022-04-01",
                        "adapter_top_ticker": "6669.TW",
                        "formal_pool1_target": "00631L.TW",
                    }
                ]
            ).to_csv(previous / "equivalence_mismatch_samples.csv", index=False)

            result = run_pool1_full_state_machine_adapter_equivalence(
                formal_long_range_dir=formal,
                previous_equivalence_dir=previous,
                price_cache_dir=cache,
                price_source_registry=root / "missing_registry.csv",
                output_dir=output,
                start_date="2022-04-01",
                end_date="2022-04-04",
            )

            self.assertEqual(result, output)
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertTrue(manifest["equivalence_pass"])
            self.assertFalse(manifest["formal_model_changed"])
            self.assertFalse(manifest["trade_decision_changed"])
            self.assertFalse(manifest["raw_diagnostic_pass_used_as_formal_target"])
            self.assertEqual(manifest["next_required_task"], "pool1_full_state_replay_201411_202112_dynamic_universe")

            summary = pd.read_csv(output / "equivalence_summary.csv").iloc[0]
            self.assertTrue(bool(summary["equivalence_pass"]))
            self.assertEqual(int(summary["mismatch_rows"]), 0)

            equivalence = pd.read_csv(output / "full_state_equivalence_2022plus.csv")
            self.assertTrue(equivalence["row_match_for_formal_readiness"].astype(bool).all())
            self.assertIn("raw_dynamic_attack_gate_pass", equivalence.columns)

            contract = pd.read_csv(output / "full_state_machine_adapter_contract.csv")
            self.assertIn("dynamic_universe_2014_2021", set(contract["component"]))


if __name__ == "__main__":
    unittest.main()
