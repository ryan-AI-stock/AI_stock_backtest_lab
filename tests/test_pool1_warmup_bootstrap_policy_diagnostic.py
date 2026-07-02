from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

import test_paths  # noqa: F401

from backtest_lab.current_formal_pool1_pool2_signal_panels import POOL1_TICKERS
from backtest_lab.pool1_warmup_bootstrap_policy_diagnostic import (
    run_pool1_warmup_bootstrap_policy_diagnostic,
)


class Pool1WarmupBootstrapPolicyDiagnosticTest(unittest.TestCase):
    def test_diagnostic_recommends_warmup_only_not_bootstrap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            long_range = root / "long"
            lifecycle = root / "lifecycle"
            pool2 = root / "pool2"
            cache = root / "cache"
            output = root / "out"
            for folder in (long_range, lifecycle, pool2, cache):
                folder.mkdir()

            pd.DataFrame(
                [
                    {
                        "signal_date": "2014-11-03",
                        "available_universe_count": 0,
                        "candidate_tickers": "",
                        "blocker": "pool1_dynamic_universe_warmup_or_lifecycle_not_ready",
                    },
                    {
                        "signal_date": "2015-01-27",
                        "available_universe_count": 1,
                        "candidate_tickers": "00631L.TW",
                        "blocker": "pool1_dynamic_universe_warmup_or_lifecycle_not_ready",
                    },
                ]
            ).to_csv(long_range / "remaining_blocked_rows.csv", index=False)
            pd.DataFrame(
                [
                    {"signal_date": "2015-01-28", "pool1_target": "00631L.TW"},
                    {"signal_date": "2015-01-29", "pool1_target": "2330.TW"},
                ]
            ).to_csv(long_range / "pool1_full_state_replay_201411_202112.csv", index=False)
            pd.DataFrame(
                [
                    {"signal_date": "2015-01-28", "pool2_vote": "2330.TW"},
                    {"signal_date": "2015-01-29", "pool2_vote": "2454.TW"},
                ]
            ).to_csv(pool2 / "pool2_daily_vote_status.csv", index=False)
            pd.DataFrame([{"date": "2015-01-28", "candidate_ticker": "2330.TW"}]).to_csv(
                pool2 / "pool2_reconstructed_eligible_rows.csv",
                index=False,
            )
            pd.DataFrame([_lifecycle_row(ticker) for ticker in POOL1_TICKERS]).to_csv(
                lifecycle / "pool1_ticker_lifecycle_contract.csv",
                index=False,
            )
            for ticker in POOL1_TICKERS:
                _price_frame("2014-11-03", 61).to_csv(cache / f"{ticker.replace('.', '_')}.csv", index=False)

            result = run_pool1_warmup_bootstrap_policy_diagnostic(
                output_dir=output,
                long_range_dir=long_range,
                lifecycle_dir=lifecycle,
                pool2_dir=pool2,
                price_cache_dir=cache,
                price_source_registry=root / "missing_registry.csv",
            )

            self.assertEqual(result, output)
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["blocked_rows"], 2)
            self.assertEqual(manifest["recommended_policy"], "A_start_formal_replay_2015_01_28")
            self.assertFalse(manifest["bootstrap_formal_ready"])
            self.assertFalse(manifest["formal_model_changed"])
            options = pd.read_csv(output / "warmup_policy_options.csv")
            recommended = options[options["recommended"].astype(bool)].iloc[0]
            self.assertEqual(recommended["option_id"], "A_start_formal_replay_2015_01_28")
            breakdown = pd.read_csv(output / "pool1_warmup_blocker_breakdown.csv")
            self.assertIn("candidate_universe_insufficient", breakdown.columns)


def _lifecycle_row(ticker: str) -> dict[str, str]:
    return {
        "ticker": ticker,
        "name": ticker,
        "pool1_role": "market_exposure_tool" if ticker == "00631L.TW" else "ai_main_attack_candidate",
        "first_tradable_date": "2014-11-03",
        "last_tradable_date": "2026-06-29",
        "data_start": "2014-11-03",
        "data_end": "2026-06-29",
        "first_pool1_scoring_date": "2015-01-28",
    }


def _price_frame(start: str, periods: int) -> pd.DataFrame:
    dates = pd.bdate_range(start=start, periods=periods)
    close = [100.0 + index for index in range(periods)]
    return pd.DataFrame(
        {
            "date": [date.strftime("%Y-%m-%d") for date in dates],
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "adj_close": close,
            "volume": [1_000_000] * periods,
        }
    )


if __name__ == "__main__":
    unittest.main()
