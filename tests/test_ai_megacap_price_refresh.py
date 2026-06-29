from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from backtest_lab.ai_megacap_price_refresh import run_ai_megacap_price_refresh


class AiMegacapPriceRefreshTests(unittest.TestCase):
    def test_refresh_writes_before_after_and_failed_ledgers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache = root / "cache"
            cache.mkdir()
            output = root / "out"
            _write_price(cache / "2330_TW.csv", ["2026-06-12"], [100.0])

            def loader(
                *,
                tickers: list[str],
                start_date: str,
                end_date: str,
                cache_dir: str | Path,
            ):
                self.assertEqual(start_date, "2026-01-01")
                self.assertEqual(end_date, "2026-06-29")
                ticker = tickers[0]
                if ticker == "2454.TW":
                    return {}, ["2454.TW"]
                frame = pd.DataFrame(
                    {
                        "open": [100.0, 110.0],
                        "high": [101.0, 111.0],
                        "low": [99.0, 109.0],
                        "close": [100.0, 110.0],
                        "adj_close": [100.0, 110.0],
                        "volume": [1000, 1100],
                    },
                    index=pd.to_datetime(["2026-06-12", "2026-06-26"]),
                )
                _write_price(Path(cache_dir) / f"{ticker.replace('.', '_')}.csv", ["2026-06-12", "2026-06-26"], [100.0, 110.0])
                return {ticker: frame}, []

            run_ai_megacap_price_refresh(
                tickers=("2330.TW", "2454.TW"),
                start_date="2026-01-01",
                end_date="2026-06-29",
                cache_dir=cache,
                output_dir=output,
                loader=loader,
            )

            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertFalse(manifest["formal_model_changed"])
            self.assertFalse(manifest["trade_decision_changed"])
            self.assertEqual(manifest["ticker_count"], 2)
            self.assertEqual(manifest["refreshed_count"], 1)
            self.assertEqual(manifest["failed_count"], 1)
            self.assertFalse(manifest["full_market_download"])
            self.assertTrue(manifest["no_forward_fill_used"])

            coverage = pd.read_csv(output / "price_refresh_before_after_coverage.csv")
            row_2330 = coverage[coverage["ticker"] == "2330.TW"].iloc[0]
            self.assertEqual(row_2330["before_last_date"], "2026-06-12")
            self.assertEqual(row_2330["after_last_date"], "2026-06-26")

            failed = pd.read_csv(output / "failed_tickers.csv")
            self.assertEqual(failed.iloc[0]["ticker"], "2454.TW")
            self.assertIn("missing", failed.iloc[0]["reason"])


def _write_price(path: Path, dates: list[str], closes: list[float]) -> None:
    rows = []
    for date, close in zip(dates, closes):
        rows.append(
            {
                "date": date,
                "open": close,
                "high": close,
                "low": close,
                "close": close,
                "adj_close": close,
                "volume": 1000,
                "dividend": 0,
                "stock_split": 0,
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


if __name__ == "__main__":
    unittest.main()
