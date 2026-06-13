from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

import test_paths  # noqa: F401

from backtest_lab.tw50_constituents import load_tw50_constituents_for_date
from backtest_lab.tw50_constituents_update import normalize_constituent_frame, update_tw50_constituents


class Tw50ConstituentsUpdateTest(unittest.TestCase):
    def test_normalizes_source_with_chinese_columns(self) -> None:
        frame = pd.DataFrame(
            [
                {"資料日期": "2026-06-12", "股票代號": "2330", "股票名稱": "台積電"},
                {"資料日期": "2026-06-12", "股票代號": "2454", "股票名稱": "聯發科"},
            ]
        )

        normalized = normalize_constituent_frame(
            frame,
            as_of_date="2026-06-12",
            source="unit_test",
            min_count=2,
            max_count=2,
        )

        self.assertEqual([item for item in normalized["ticker"]], ["2330.TW", "2454.TW"])
        self.assertEqual([item for item in normalized["name"]], ["台積電", "聯發科"])
        self.assertEqual(set(normalized["effective_date"]), {"2026-06-12"})

    def test_updates_from_seed_and_can_be_loaded_by_point_in_time_reader(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            seed = root / "seed.csv"
            seed.write_text(
                "\n".join(
                    [
                        "effective_date,ticker,name,source,source_updated_at",
                        "2026-06-01,2330.TW,台積電,seed,2026-06-12",
                        "2026-06-01,2454.TW,聯發科,seed,2026-06-12",
                    ]
                ),
                encoding="utf-8",
            )
            output = root / "tw50_constituents.csv"
            status = root / "status.json"

            result = update_tw50_constituents(
                output_path=output,
                as_of_date="2026-06-12",
                seed_path=seed,
                allow_seed_fallback=True,
                min_count=2,
                max_count=2,
            )
            status.write_text(json.dumps(result.to_dict(), ensure_ascii=False), encoding="utf-8")
            entries = load_tw50_constituents_for_date(output, "2026-06-12")

        self.assertTrue(result.used_fallback)
        self.assertEqual(result.row_count, 2)
        self.assertEqual([entry["ticker"] for entry in entries], ["2330.TW", "2454.TW"])

    def test_primary_source_appends_new_effective_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "tw50_constituents.csv"
            pd.DataFrame(
                [{"effective_date": "2026-06-01", "ticker": "2330.TW", "name": "台積電"}]
            ).to_csv(output, index=False)
            source = root / "source.csv"
            pd.DataFrame(
                [
                    {"ticker": "2330", "name": "台積電"},
                    {"ticker": "2454", "name": "聯發科"},
                ]
            ).to_csv(source, index=False)

            result = update_tw50_constituents(
                output_path=output,
                as_of_date="2026-06-12",
                source_csv=source,
                allow_seed_fallback=False,
                min_count=2,
                max_count=2,
            )
            frame = pd.read_csv(output)

        self.assertFalse(result.used_fallback)
        self.assertEqual(set(frame["effective_date"].astype(str)), {"2026-06-01", "2026-06-12"})
        self.assertEqual(result.total_row_count, 3)


if __name__ == "__main__":
    unittest.main()
