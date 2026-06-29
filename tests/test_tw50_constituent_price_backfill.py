from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from backtest_lab.tw50_constituent_price_backfill import run_tw50_constituent_price_backfill


class Tw50ConstituentPriceBackfillTests(unittest.TestCase):
    def test_runner_writes_price_only_readiness_without_pit_claims(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            constituents = root / "tw50_constituents.csv"
            constituents.write_text(
                "effective_date,ticker,name,source,source_updated_at\n"
                "2025-06-23,2330.TW,台積電,seed_snapshot,2026-06-13\n",
                encoding="utf-8",
            )
            cache = root / "cache"
            cache.mkdir()
            output = root / "out"

            def downloader(tickers: list[str], start_date: str, end_date: str, cache_dir: str | Path):
                ticker = tickers[0]
                frame = pd.DataFrame(
                    {
                        "open": [100.0, 110.0],
                        "high": [101.0, 111.0],
                        "low": [99.0, 109.0],
                        "close": [100.0, 110.0],
                        "adj_close": [100.0, 110.0],
                        "volume": [1000, 1100],
                    },
                    index=pd.to_datetime(["2014-11-03", "2026-06-29"]),
                )
                _write_price(Path(cache_dir) / f"{ticker.replace('.', '_')}.csv", frame)
                return {ticker: frame}

            run_tw50_constituent_price_backfill(
                constituents_path=constituents,
                cache_dir=cache,
                output_dir=output,
                start_date="2014-11-01",
                end_date="2026-06-29",
                downloader=downloader,
            )

            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertFalse(manifest["formal_model_changed"])
            self.assertFalse(manifest["trade_decision_changed"])
            self.assertFalse(manifest["historical_pit_ready"])
            self.assertFalse(manifest["strategy_ready"])
            self.assertFalse(manifest["current_snapshot_used_as_historical_pit"])
            self.assertFalse(manifest["user_manual_download_required"])
            self.assertEqual(manifest["data_acquisition_owner"], "Core/Data/Radar")

            coverage = pd.read_csv(output / "price_coverage_matrix.csv")
            self.assertIn("2330.TW", set(coverage["ticker"]))
            self.assertIn("0050.TW", set(coverage["ticker"]))
            self.assertIn("00631L.TW", set(coverage["ticker"]))
            self.assertTrue(coverage["ready_for_backtest_price_only"].all())
            self.assertFalse(coverage["strategy_ready"].any())

            source = pd.read_csv(output / "universe_source_ledger.csv")
            self.assertFalse(source["formal_exact_pit"].any())
            self.assertFalse(source["user_manual_download_required"].any())


def _write_price(path: Path, frame: pd.DataFrame) -> None:
    payload = frame.copy()
    payload.insert(0, "date", payload.index.strftime("%Y-%m-%d"))
    payload["dividend"] = 0
    payload["stock_split"] = 0
    payload.to_csv(path, index=False)


if __name__ == "__main__":
    unittest.main()
