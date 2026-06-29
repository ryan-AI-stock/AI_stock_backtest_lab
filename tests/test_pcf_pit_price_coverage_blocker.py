import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from backtest_lab.pcf_pit_price_coverage_blocker import run_0050_pit_price_coverage_blocker


class PcfPitPriceCoverageBlockerTest(unittest.TestCase):
    def test_uses_first_anchor_date_for_price_requirement(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            anchor = root / "anchor.csv"
            cache = root / "cache"
            out = root / "out"
            cache.mkdir()
            _anchor_csv(anchor)
            _price_csv(cache / "6669_TW.csv", "2017-11-13", "2023-12-29")

            run_0050_pit_price_coverage_blocker(
                monthly_anchor_path=anchor,
                cache_dir=cache,
                output_dir=out,
                replay_end_date="2023-12-31",
                refresh_missing=False,
            )

            manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
            self.assertFalse(manifest["formal_model_changed"])
            self.assertFalse(manifest["trade_decision_changed"])
            self.assertEqual(manifest["pit_universe_tickers"], 2)
            self.assertEqual(manifest["price_ready_tickers"], 1)
            self.assertEqual(manifest["missing_coverage_tickers"], 1)

            status = pd.read_csv(out / "pit_universe_price_coverage_status.csv")
            row_6669 = status[status["ticker"] == "6669.TW"].iloc[0]
            self.assertEqual(row_6669["coverage_status"], "price_only_ready")
            self.assertEqual(row_6669["required_start_date"], "2020-06-30")

            missing = pd.read_csv(out / "missing_price_coverage_tickers.csv")
            self.assertEqual(missing["ticker"].tolist(), ["1101.TW"])

    def test_refresh_missing_records_completed_download(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            anchor = root / "anchor.csv"
            cache = root / "cache"
            out = root / "out"
            cache.mkdir()
            _anchor_csv(anchor, tickers=(("1101", "台泥", "2014-11", "2014-11-28"),))

            def fake_downloader(tickers, start_date, end_date, cache_dir):
                frame = _price_frame(start_date, end_date)
                frame.reset_index(names="date").to_csv(Path(cache_dir) / "1101_TW.csv", index=False)
                return {"1101.TW": frame}

            run_0050_pit_price_coverage_blocker(
                monthly_anchor_path=anchor,
                cache_dir=cache,
                output_dir=out,
                replay_end_date="2023-12-31",
                refresh_missing=True,
                downloader=fake_downloader,
            )

            manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["price_ready_tickers"], 1)
            self.assertEqual(manifest["missing_coverage_tickers"], 0)
            self.assertTrue(manifest["price_blocker_cleared"])

            completed = pd.read_csv(out / "price_backfill_completed.csv")
            self.assertEqual(completed["ticker"].tolist(), ["1101.TW"])


def _anchor_csv(
    path: Path,
    tickers=(("1101", "台泥", "2014-11", "2014-11-28"), ("6669", "緯穎", "2020-06", "2020-06-30")),
) -> None:
    rows = []
    for ticker, name, month, date in tickers:
        rows.append(
            {
                "effective_month": month,
                "effective_date": date,
                "holdings_date": date,
                "source_date": date,
                "ticker": ticker,
                "name": name,
                "source_url": "https://example.test",
                "raw_source_id": f"raw#{ticker}",
                "source_type": "source_backed_manual_candidate",
                "formal_exact": "false",
                "proxy_row_used": "false",
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def _price_csv(path: Path, start: str, end: str) -> None:
    _price_frame(start, end).reset_index(names="date").to_csv(path, index=False)


def _price_frame(start: str, end: str) -> pd.DataFrame:
    dates = pd.bdate_range(start, end)
    return pd.DataFrame(
        {
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "adj_close": 100.0,
            "volume": 1000,
            "dividend": 0.0,
            "stock_split": 0.0,
        },
        index=dates,
    )


if __name__ == "__main__":
    unittest.main()
