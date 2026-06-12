from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

import test_paths  # noqa: F401

from backtest_lab.formal_radar_candidates import (
    BUCKET_ACTIONABLE,
    BUCKET_WATCH,
    formal_radar_candidates_to_symbols,
    load_formal_radar_candidates,
)


class FormalRadarCandidatesTest(unittest.TestCase):
    def test_uses_actionable_bucket_before_watch_bucket(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_stock_metrics(
                root / "stock_metrics.refreshed.csv",
                [
                    _row("1111", "分類一", pullback=85, chip=80, technical=100, liquidity=100, risk_heat=45),
                    _row("2222", "分類二", pullback=70, chip=56, technical=78, liquidity=100, risk_heat=53),
                ],
            )

            candidates = load_formal_radar_candidates(root)

        self.assertEqual([item.symbol for item in candidates], ["1111"])
        self.assertEqual(candidates[0].bucket, BUCKET_ACTIONABLE)

    def test_falls_back_to_top_three_watch_bucket_when_no_actionable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_stock_metrics(
                root / "stock_metrics.refreshed.csv",
                [
                    _row("2368", "金像電", pullback=70, chip=56, technical=78, liquidity=100, risk_heat=53),
                    _row("2408", "南亞科", pullback=69, chip=56, technical=78, liquidity=100, risk_heat=53),
                    _row("3037", "欣興", pullback=68, chip=56, technical=78, liquidity=100, risk_heat=53),
                    _row("9999", "排除", pullback=20, chip=20, technical=20, liquidity=20, risk_heat=90),
                ],
            )

            candidates = load_formal_radar_candidates(root)

        self.assertEqual([item.symbol for item in candidates], ["2368", "2408", "3037"])
        self.assertTrue(all(item.bucket == BUCKET_WATCH for item in candidates))
        self.assertEqual([item.rank for item in candidates], [1, 2, 3])

    def test_converts_candidates_to_stock_pool_symbols(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_stock_metrics(
                root / "stock_metrics.refreshed.csv",
                [_row("2368", "金像電", pullback=70, chip=56, technical=78, liquidity=100, risk_heat=53)],
            )
            candidates = load_formal_radar_candidates(root)

        symbols = formal_radar_candidates_to_symbols(candidates)

        self.assertEqual(symbols[0]["ticker"], "2368.TW")
        self.assertEqual(symbols[0]["display"], "金像電(2368)")
        self.assertEqual(symbols[0]["source"], "formal_radar_bucket")
        self.assertEqual(symbols[0]["formal_bucket"], BUCKET_WATCH)


def _row(
    symbol: str,
    name: str,
    *,
    pullback: float,
    chip: float,
    technical: float,
    liquidity: float,
    risk_heat: float,
) -> dict[str, object]:
    return {
        "symbol": symbol,
        "name": name,
        "sector": "記憶體",
        "close": 100,
        "pullback_quality": pullback,
        "chip_cleanliness": chip,
        "foreign_5d": 0,
        "trust_5d": 0,
        "margin_change_5d": 0,
        "pe": 0,
        "sector_pe_low": 0,
        "sector_pe_avg": 0,
        "sector_pe_high": 0,
        "fair_value_low": 0,
        "fair_value_avg": 0,
        "fair_value_high": 0,
        "revenue_yoy": 0,
        "revenue_mom": 0,
        "technical_setup": technical,
        "liquidity": liquidity,
        "risk_heat": risk_heat,
        "thesis": "",
        "risk_reason": "",
    }


def _write_stock_metrics(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    unittest.main()
