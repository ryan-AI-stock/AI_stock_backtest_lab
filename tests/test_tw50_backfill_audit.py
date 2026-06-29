import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from backtest_lab.tw50_backfill_audit import run_tw50_backfill_audit


class Tw50BackfillAuditTest(unittest.TestCase):
    def test_builds_audit_outputs_without_marking_proxy_as_exact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "data"
            cache = root / "cache"
            replay = root / "replay"
            output = root / "output"
            data.mkdir()
            cache.mkdir()
            replay.mkdir()
            pd.DataFrame(
                [
                    {
                        "effective_date": "2025-06-23",
                        "ticker": "2330.TW",
                        "name": "台積電",
                        "source": "seed_snapshot",
                        "source_updated_at": "2026-06-13",
                    }
                ]
            ).to_csv(data / "tw50_constituents.csv", index=False)
            _price_csv(cache / "0050_TW.csv", "2014-11-03", periods=2600)
            _price_csv(cache / "00631L_TW.csv", "2016-01-04", periods=2000)
            _price_csv(cache / "2330_TW.csv", "2014-11-03", periods=2600)
            (replay / "metadata.json").write_text(json.dumps({"initial_cash": 1_000_000}), encoding="utf-8")

            run_tw50_backfill_audit(
                constituents_path=data / "tw50_constituents.csv",
                price_roots=(cache,),
                formal_replay_metadata=replay / "metadata.json",
                output_dir=output,
            )

            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertFalse(manifest["formal_model_changed"])
            self.assertFalse(manifest["trade_decision_changed"])
            self.assertFalse(manifest["large_download_started"])
            self.assertFalse(manifest["pit_exact_coverage_201411_202312_ready"])
            self.assertEqual(manifest["previous_best_initial_capital"], 1_000_000)
            self.assertEqual(manifest["exact_historical_total_ticker_count"], "blocked_until_PIT_archive_acquired")

            pit = pd.read_csv(output / "tw50_pit_constituents_coverage.csv")
            self.assertEqual(pit["source_type"].iloc[0], "proxy_current_snapshot")
            readiness = pd.read_csv(output / "data_readiness_ledger.csv")
            self.assertIn("tw50_pit_constituents", set(readiness["layer"]))
            plan = pd.read_csv(output / "missing_data_backfill_plan.csv")
            self.assertIn("archive_tw50_pit_constituent_sources", set(plan["step"]))


def _price_csv(path: Path, start: str, periods: int) -> None:
    dates = pd.bdate_range(start, periods=periods)
    pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "adj_close": 100.0,
            "volume": 1000,
            "dividend": 0.0,
            "stock_split": 0.0,
        }
    ).to_csv(path, index=False)


if __name__ == "__main__":
    unittest.main()
