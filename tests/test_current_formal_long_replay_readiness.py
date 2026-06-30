from __future__ import annotations

import unittest

import test_paths  # noqa: F401

from backtest_lab.current_formal_long_replay_readiness import (
    _data_blockers,
    _missing_inputs,
    _target_stream_stub,
)


class CurrentFormalLongReplayReadinessTest(unittest.TestCase):
    def test_blockers_are_precise_signal_stream_blockers(self) -> None:
        blockers = _data_blockers()
        blocking_ids = set(blockers[blockers["blocks_long_replay"].eq(True)]["blocker_id"])

        self.assertIn("missing_pool1_daily_candidate_ranking_panel", blocking_ids)
        self.assertIn("missing_pool2_daily_confirmation_panel", blocking_ids)
        self.assertIn("missing_formal_target_stream", blocking_ids)
        self.assertNotIn("four_tickers_unadjusted_only", blocking_ids)

    def test_missing_inputs_include_minimum_fix_not_generic_data_shortage(self) -> None:
        missing = _missing_inputs()
        self.assertIn("minimum_fix", missing.columns)
        self.assertTrue(
            missing["minimum_fix"].astype(str).str.contains("Pool1 ranking|Pool2|target_weights|next-day", regex=True).any()
        )
        self.assertFalse(missing["missing_detail"].astype(str).str.fullmatch("資料不足").any())

    def test_target_stream_stub_does_not_fake_historical_targets(self) -> None:
        stub = _target_stream_stub()
        self.assertEqual(stub.iloc[0]["status"], "blocked")
        self.assertFalse(bool(stub.iloc[0]["can_emit_target_stream"]))


if __name__ == "__main__":
    unittest.main()
