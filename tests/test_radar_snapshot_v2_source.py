from __future__ import annotations

import csv
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from backtest_lab.radar_snapshot_readiness import REQUIRED_SNAPSHOT_COLUMNS
from backtest_lab.radar_snapshot_v2_source import load_radar_snapshot_history, select_radar_snapshot_candidates


class RadarSnapshotV2SourceTest(unittest.TestCase):
    def test_selects_latest_snapshot_not_after_signal_date(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write_snapshot(root / "radar_snapshot_20260501.csv", date="2026-05-01", symbol="1111", score="50")
            _write_snapshot(root / "radar_snapshot_20260503.csv", date="2026-05-03", symbol="2222", score="90")

            history = load_radar_snapshot_history(root)
            candidates = select_radar_snapshot_candidates(history, "2026-05-02")

            self.assertEqual(str(candidates.snapshot_date.date()), "2026-05-01")
            self.assertEqual(candidates.rows.iloc[0]["symbol"], "1111")

    def test_filters_fundamental_fail_and_overheated_rows(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            path = root / "radar_snapshot_20260501.csv"
            _write_snapshot(path, date="2026-05-01", symbol="1111", score="80")
            _append_row(path, date="2026-05-01", symbol="2222", score="99", fundamental_pass="false")
            _append_row(path, date="2026-05-01", symbol="3333", score="98", overheated_flag="true")

            history = load_radar_snapshot_history(root)
            candidates = select_radar_snapshot_candidates(history, "2026-05-01")

            self.assertEqual(candidates.rows["symbol"].tolist(), ["1111"])

    def test_rejects_future_fundamental_source_date(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write_snapshot(
                root / "radar_snapshot_20260501.csv",
                date="2026-05-01",
                symbol="1111",
                source_date="2026-05-02",
            )

            history = load_radar_snapshot_history(root)

            with self.assertRaisesRegex(ValueError, "future fundamental"):
                select_radar_snapshot_candidates(history, "2026-05-01")


def _write_snapshot(
    path: Path,
    *,
    date: str,
    symbol: str,
    score: str = "80",
    source_date: str | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(REQUIRED_SNAPSHOT_COLUMNS))
        writer.writeheader()
        writer.writerow(_row(date=date, symbol=symbol, score=score, source_date=source_date or date))


def _append_row(
    path: Path,
    *,
    date: str,
    symbol: str,
    score: str,
    fundamental_pass: str = "true",
    overheated_flag: str = "false",
) -> None:
    with path.open("a", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(REQUIRED_SNAPSHOT_COLUMNS))
        writer.writerow(
            _row(
                date=date,
                symbol=symbol,
                score=score,
                fundamental_pass=fundamental_pass,
                overheated_flag=overheated_flag,
            )
        )


def _row(
    *,
    date: str,
    symbol: str,
    score: str,
    source_date: str | None = None,
    fundamental_pass: str = "true",
    overheated_flag: str = "false",
) -> dict[str, str]:
    row = {column: "" for column in sorted(REQUIRED_SNAPSHOT_COLUMNS)}
    row.update(
        {
            "date": date,
            "theme": "記憶體",
            "symbol": symbol,
            "name": symbol,
            "theme_rank": "1",
            "theme_score": score,
            "capital_share": "0.1",
            "turnover_value": "1000000",
            "stock_score": score,
            "bucket": "theme_leader",
            "fundamental_pass": fundamental_pass,
            "fundamental_score": "80",
            "fundamental_data_status": "ok" if fundamental_pass == "true" else "low_quality",
            "fundamental_source_date": source_date or date,
            "risk_heat": "0.2",
            "liquidity": "ok",
            "stock_turnover_rank_in_theme": "1",
            "stock_turnover_share_in_theme": "0.5",
            "theme_leader_flag": "true",
            "theme_second_line_flag": "false",
            "theme_laggard_rebound_flag": "false",
            "overheated_flag": overheated_flag,
        }
    )
    return row


if __name__ == "__main__":
    unittest.main()
