from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

import test_paths  # noqa: F401

from backtest_lab.combined_formal_target_stream_2015_2021 import (
    run_combined_formal_target_stream_2015_2021,
)


class CombinedFormalTargetStream20152021Test(unittest.TestCase):
    def test_builds_active_and_cash_rows_without_warmup_leakage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pool1 = root / "pool1"
            pool2 = root / "pool2"
            warmup = root / "warmup"
            cache = root / "cache"
            output = root / "out"
            for path in (pool1, pool2, warmup, cache):
                path.mkdir()

            _pool1_rows().to_csv(pool1 / "pool1_full_state_replay_201411_202112.csv", index=False)
            _pool2_rows().to_csv(pool2 / "pool2_daily_vote_status.csv", index=False)
            _warmup_rows().to_csv(warmup / "pool1_warmup_blocker_breakdown.csv", index=False)
            _price_rows().to_csv(cache / "00631L_TW.csv", index=False)

            result = run_combined_formal_target_stream_2015_2021(
                pool1_output=pool1,
                pool2_output=pool2,
                warmup_output=warmup,
                output_dir=output,
                price_cache_dir=cache,
                price_source_registry=root / "missing_registry.csv",
                start_date="2020-01-02",
                end_date="2020-01-03",
            )

            self.assertEqual(result, output)
            stream = pd.read_csv(output / "combined_formal_target_stream.csv")
            self.assertEqual(len(stream), 2)
            self.assertEqual(stream.loc[0, "formal_target"], "00631L.TW")
            self.assertEqual(stream.loc[0, "target_type"], "market_exposure_tool")
            self.assertEqual(stream.loc[1, "formal_target"], "CASH")
            self.assertEqual(stream.loc[1, "risk_off_state"], "no_target_cash_all")
            self.assertEqual(stream.loc[1, "no_target_reason"], "pool2_confirmation_not_ready")
            self.assertFalse(stream["warmup_only"].any())

            warmup_exclusion = pd.read_csv(output / "warmup_exclusion.csv")
            self.assertEqual(warmup_exclusion.loc[0, "readiness_state"], "warmup_only_non_tradable")

            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["combined_stream_rows"], 2)
            self.assertEqual(manifest["warmup_excluded_rows"], 1)
            self.assertFalse(manifest["formal_model_changed"])
            self.assertFalse(manifest["trade_decision_changed"])
            self.assertTrue(manifest["stream_only_not_performance_replay"])
            self.assertTrue(manifest["ready_for_experiments_next_day_replay"])

    def test_future_anchor_blocks_pool2_confirmation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pool1 = root / "pool1"
            pool2 = root / "pool2"
            warmup = root / "warmup"
            cache = root / "cache"
            output = root / "out"
            for path in (pool1, pool2, warmup, cache):
                path.mkdir()

            _pool1_rows().head(1).to_csv(pool1 / "pool1_full_state_replay_201411_202112.csv", index=False)
            future_pool2 = _pool2_rows().head(1).copy()
            future_pool2["anchor_after_query_date"] = True
            future_pool2["pit_safe_for_query_date"] = False
            future_pool2.to_csv(pool2 / "pool2_daily_vote_status.csv", index=False)
            _warmup_rows().to_csv(warmup / "pool1_warmup_blocker_breakdown.csv", index=False)
            _price_rows().head(1).to_csv(cache / "00631L_TW.csv", index=False)

            run_combined_formal_target_stream_2015_2021(
                pool1_output=pool1,
                pool2_output=pool2,
                warmup_output=warmup,
                output_dir=output,
                price_cache_dir=cache,
                price_source_registry=root / "missing_registry.csv",
                start_date="2020-01-02",
                end_date="2020-01-02",
            )

            stream = pd.read_csv(output / "combined_formal_target_stream.csv")
            self.assertEqual(stream.loc[0, "formal_target"], "CASH")
            self.assertEqual(stream.loc[0, "pool2_confirmation_status"], "pool2_not_ready")


def _pool1_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _pool1_row("2020-01-02"),
            _pool1_row("2020-01-03"),
        ]
    )


def _pool1_row(date: str) -> dict[str, object]:
    return {
        "signal_date": date,
        "pool1_target": "00631L.TW",
        "pool1_target_display": "0050正二",
        "pool1_target_weights": '{"00631L.TW": 1.0}',
        "attack_gate_active": False,
        "attack_gate_ever_activated": False,
        "risk_off_active": False,
        "target_is_actionable": True,
        "model_target_status": "has_formal_pool1_target",
        "mode": "0050_defense",
        "regime": "strong_bull",
        "current_exposure": 1.0,
        "available_universe_count": 7,
        "candidate_tickers": "00631L.TW|2330.TW",
        "segment_id": "test",
        "segment_source": "test",
        "candidate_universe_fallback_separated": True,
        "source_formal_ready": True,
        "no_target_cash_all_applied": False,
    }


def _pool2_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "signal_date": "2020-01-02",
                "pool2_vote": "2330.TW",
                "pool2_support_without_persistence_vote": "2330.TW",
                "pool2_confirmation_ready": True,
                "pool2_blocker": "",
                "anchor_after_query_date": False,
                "pit_safe_for_query_date": True,
            },
            {
                "signal_date": "2020-01-03",
                "pool2_vote": "",
                "pool2_support_without_persistence_vote": "",
                "pool2_confirmation_ready": False,
                "pool2_blocker": "no_pool2_persistent_eligible_candidate",
                "anchor_after_query_date": False,
                "pit_safe_for_query_date": True,
            },
        ]
    )


def _warmup_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "signal_date": "2014-11-03",
                "available_universe_count": 0,
                "required_dynamic_universe_count": 7,
                "ready_tickers_by_history": "",
                "blocker_category": "no_pool1_candidate_has_60d_warmup",
                "blocker_reason_zh": "所有 Pool1 標的都還沒累積滿正式 60 日相對強度所需歷史。",
            }
        ]
    )


def _price_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": ["2020-01-02", "2020-01-03"],
            "open": [100.0, 101.0],
            "high": [101.0, 102.0],
            "low": [99.0, 100.0],
            "close": [100.0, 101.0],
            "adj_close": [100.0, 101.0],
            "volume": [1_000_000, 1_000_000],
        }
    )


if __name__ == "__main__":
    unittest.main()
