from __future__ import annotations

import unittest

import pandas as pd

from backtest_lab.universal_pool_strategy import (
    POOL_LARGE_LIQUID,
    POOL_MID_SMALL_LIQUID,
    POOL_THIN_OR_MIXED,
    classify_candidate_profile,
    default_parameters_for_profile,
    infer_pool_profile,
    score_universal_candidate,
    score_universal_candidates,
    universal_stock_score,
)


class UniversalPoolStrategyTest(unittest.TestCase):
    def test_large_liquid_pool_does_not_need_turnover_gate_by_default(self) -> None:
        dates = pd.bdate_range("2024-01-02", periods=30)
        prices = {
            f"{ticker}.TW": _price_frame(dates, close=100 + index, volume=20_000_000)
            for index, ticker in enumerate(("2330", "2454", "2308", "2317", "2382"))
        }

        profile = infer_pool_profile(prices, dates[-1])
        params = default_parameters_for_profile(profile)

        self.assertEqual(profile.pool_type, POOL_LARGE_LIQUID)
        self.assertEqual(params.min_avg_turnover_twd, 0.0)
        self.assertEqual(params.score_mode, "relative_strength")

    def test_mid_small_pool_defaults_to_liquidity_and_overheat_controls(self) -> None:
        dates = pd.bdate_range("2024-01-02", periods=30)
        prices = {
            f"{ticker}.TW": _price_frame(dates, close=80 + index, volume=800_000)
            for index, ticker in enumerate(("2408", "2344", "2337", "3006", "3260", "8299", "8271", "8088", "6531", "6770"))
        }

        profile = infer_pool_profile(prices, dates[-1], theme_by_ticker={ticker: "記憶體" for ticker in prices})
        params = default_parameters_for_profile(profile)

        self.assertEqual(profile.pool_type, POOL_MID_SMALL_LIQUID)
        self.assertEqual(params.min_avg_turnover_twd, 60_000_000)
        self.assertAlmostEqual(params.overheated_20d_return, 0.62)
        self.assertEqual(params.score_mode, "risk_adjusted")

    def test_risk_adjusted_score_penalizes_volatility(self) -> None:
        low_vol = universal_stock_score(
            ret20=0.10,
            ret60=0.30,
            ret120=0.50,
            vol20=0.20,
            mode="risk_adjusted",
        )
        high_vol = universal_stock_score(
            ret20=0.10,
            ret60=0.30,
            ret120=0.50,
            vol20=0.80,
            mode="risk_adjusted",
        )

        self.assertGreater(low_vol, high_vol)

    def test_universal_candidate_records_rejection_reason(self) -> None:
        dates = pd.bdate_range("2024-01-02", periods=150)
        prices = _price_frame(dates, close=100, volume=10_000)
        profile = infer_pool_profile({"A.TW": prices}, dates[-1])
        params = default_parameters_for_profile(profile)

        score = score_universal_candidate(
            ticker="A.TW",
            prices=prices,
            signal_date=dates[-1],
            params=params,
        )

        self.assertFalse(score.passed)
        self.assertEqual(score.reason, "流動性不足")

    def test_mixed_pool_scores_each_stock_with_its_own_profile(self) -> None:
        dates = pd.bdate_range("2024-01-02", periods=150)
        prices = {
            "2330.TW": _price_frame(dates, close=100, volume=20_000_000),
            "2408.TW": _price_frame(dates, close=80, volume=800_000),
            "9999.TW": _price_frame(dates, close=50, volume=10_000),
        }
        pool_params = default_parameters_for_profile(infer_pool_profile(prices, dates[-1]))

        scores = score_universal_candidates(prices, dates[-1], params=pool_params)

        self.assertEqual(classify_candidate_profile(prices["2330.TW"], dates[-1]), POOL_LARGE_LIQUID)
        self.assertEqual(scores["2330.TW"].profile_type, POOL_LARGE_LIQUID)
        self.assertEqual(scores["2330.TW"].applied_score_mode, "relative_strength")
        self.assertEqual(classify_candidate_profile(prices["2408.TW"], dates[-1]), POOL_MID_SMALL_LIQUID)
        self.assertEqual(scores["2408.TW"].profile_type, POOL_MID_SMALL_LIQUID)
        self.assertEqual(scores["2408.TW"].applied_score_mode, "risk_adjusted")
        self.assertEqual(classify_candidate_profile(prices["9999.TW"], dates[-1]), POOL_THIN_OR_MIXED)
        self.assertEqual(scores["9999.TW"].profile_type, POOL_THIN_OR_MIXED)
        self.assertEqual(scores["9999.TW"].reason, "流動性不足")


def _price_frame(dates: pd.DatetimeIndex, *, close: float, volume: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": [close] * len(dates),
            "high": [close] * len(dates),
            "low": [close] * len(dates),
            "close": [close] * len(dates),
            "adj_close": [close] * len(dates),
            "volume": [volume] * len(dates),
        },
        index=dates,
    )


if __name__ == "__main__":
    unittest.main()
