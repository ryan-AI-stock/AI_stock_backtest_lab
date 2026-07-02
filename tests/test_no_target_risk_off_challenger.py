import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

import test_paths  # noqa: F401

from backtest_lab.no_target_risk_off_challenger import run_no_target_risk_off_challenger


class NoTargetRiskOffChallengerTest(unittest.TestCase):
    def test_runner_keeps_formal_baseline_hold_through_and_explicit_cash_challenger(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            replay = root / "replay"
            cache = root / "cache"
            output = root / "out"
            replay.mkdir()
            cache.mkdir()

            dates = pd.to_datetime(["2026-06-05", "2026-06-08", "2026-06-09", "2026-06-10", "2026-06-11"])
            pd.DataFrame(
                [
                    {"period": "2026", "date": "2026-06-05", "pool1_vote": "2454.TW", "pool2_vote": "2454.TW"},
                    {"period": "2026", "date": "2026-06-08", "pool1_vote": "", "pool2_vote": "2454.TW"},
                    {"period": "2026", "date": "2026-06-09", "pool1_vote": "", "pool2_vote": "2454.TW"},
                    {"period": "2026", "date": "2026-06-10", "pool1_vote": "00631L.TW", "pool2_vote": "2327.TW"},
                    {"period": "2026", "date": "2026-06-11", "pool1_vote": "00631L.TW", "pool2_vote": "2327.TW"},
                ]
            ).to_csv(replay / "formal_three_pool_decision_panel.csv", index=False)
            _price_csv(cache / "2454_TW.csv", dates, [100, 96, 92, 90, 91])
            _price_csv(cache / "00631L_TW.csv", dates, [50, 51, 52, 53, 54])
            _price_csv(cache / "2327_TW.csv", dates, [400, 401, 402, 403, 404])
            _price_csv(cache / "0050_TW.csv", dates, [100, 101, 102, 103, 104])

            run_no_target_risk_off_challenger(
                formal_replay_dir=replay,
                price_cache_dir=cache,
                output_dir=output,
            )

            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["formal_default_no_formal_target_policy"], "hold_previous")
            self.assertFalse(manifest["bug_cash_mapping_used_as_baseline"])
            self.assertFalse(manifest["formal_model_changed"])
            self.assertFalse(manifest["trade_decision_changed"])

            contract = pd.read_csv(output / "variant_contract.csv")
            baseline = contract[contract["variant_id"].eq("baseline_hold_through")].iloc[0]
            cash_all = contract[contract["variant_id"].eq("no_target_cash_all")].iloc[0]
            self.assertTrue(bool(baseline["is_formal_baseline"]))
            self.assertEqual(cash_all["no_formal_target_policy"], "exit_to_cash")

            daily = pd.read_csv(output / "daily_equity_by_variant.csv")
            baseline_day = daily[
                daily["variant_id"].eq("baseline_hold_through")
                & daily["date"].astype(str).eq("2026-06-09")
            ].iloc[0]
            cash_day = daily[
                daily["variant_id"].eq("no_target_cash_all")
                & daily["date"].astype(str).eq("2026-06-09")
            ].iloc[0]
            self.assertEqual(baseline_day["top_holding"], "2454.TW")
            self.assertEqual(cash_day["top_holding"], "cash")


def _price_csv(path: Path, dates: pd.DatetimeIndex, prices: list[float]) -> None:
    pd.DataFrame(
        {
            "date": [date.strftime("%Y-%m-%d") for date in dates],
            "open": prices,
            "close": prices,
            "adj_close": prices,
        }
    ).to_csv(path, index=False)


if __name__ == "__main__":
    unittest.main()
