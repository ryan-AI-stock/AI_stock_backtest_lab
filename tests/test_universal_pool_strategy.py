from __future__ import annotations

import unittest

import pandas as pd

from backtest_lab.risk_factor_source import RiskFactorSignal
from backtest_lab.universal_pool_strategy import (
    POOL_HIGH_LIQUIDITY,
    POOL_LOW_LIQUIDITY_OR_MIXED,
    POOL_STANDARD_LIQUIDITY,
    SIZE_LARGE_CAP,
    SIZE_MICRO_CAP,
    SIZE_MID_CAP,
    SIZE_SMALL_CAP,
    SIZE_UNKNOWN,
    UniversalPoolParameters,
    classify_candidate_size_profile,
    classify_candidate_liquidity_profile,
    default_parameters_for_profile,
    infer_pool_profile,
    score_universal_candidate,
    score_universal_candidates,
    universal_stock_score,
)


class UniversalPoolStrategyTest(unittest.TestCase):
    def test_high_liquidity_pool_does_not_need_turnover_gate_by_default(self) -> None:
        dates = pd.bdate_range("2024-01-02", periods=30)
        prices = {
            f"{ticker}.TW": _price_frame(dates, close=100 + index, volume=20_000_000)
            for index, ticker in enumerate(("2330", "2454", "2308", "2317", "2382"))
        }

        profile = infer_pool_profile(prices, dates[-1])
        params = default_parameters_for_profile(profile)

        self.assertEqual(profile.pool_type, POOL_HIGH_LIQUIDITY)
        self.assertEqual(profile.classification_basis, "liquidity")
        self.assertEqual(params.min_avg_turnover_twd, 0.0)
        self.assertEqual(params.score_mode, "relative_strength")

    def test_standard_liquidity_pool_defaults_to_liquidity_and_overheat_controls(self) -> None:
        dates = pd.bdate_range("2024-01-02", periods=30)
        prices = {
            f"{ticker}.TW": _price_frame(dates, close=80 + index, volume=800_000)
            for index, ticker in enumerate(("2408", "2344", "2337", "3006", "3260", "8299", "8271", "8088", "6531", "6770"))
        }

        profile = infer_pool_profile(prices, dates[-1], theme_by_ticker={ticker: "記憶體" for ticker in prices})
        params = default_parameters_for_profile(profile)

        self.assertEqual(profile.pool_type, POOL_STANDARD_LIQUIDITY)
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

    def test_risk_signal_is_diagnostic_until_weight_is_enabled(self) -> None:
        dates = pd.bdate_range("2024-01-02", periods=150)
        prices = _price_frame(dates, close=100, volume=20_000_000)
        risk_signal = RiskFactorSignal(
            ticker="2454.TW",
            total_risk_score=50.0,
            margin_risk=50.0,
            score_adjustment=-0.05,
            reasons=("融資短線升溫",),
            source_dates=("2024-07-29",),
            source_kinds=("margin_short",),
        )
        base_params = UniversalPoolParameters(
            min_avg_turnover_twd=0.0,
            min_stock_score=-1.0,
            overheated_20d_return=0.90,
            score_mode="relative_strength",
            risk_signal_weight=0.0,
        )
        weighted_params = UniversalPoolParameters(
            min_avg_turnover_twd=0.0,
            min_stock_score=-1.0,
            overheated_20d_return=0.90,
            score_mode="relative_strength",
            risk_signal_weight=1.0,
        )

        diagnostic = score_universal_candidate(
            ticker="2454.TW",
            prices=prices,
            signal_date=dates[-1],
            params=base_params,
            risk_signal=risk_signal,
        )
        weighted = score_universal_candidate(
            ticker="2454.TW",
            prices=prices,
            signal_date=dates[-1],
            params=weighted_params,
            risk_signal=risk_signal,
        )

        self.assertEqual(diagnostic.score, 0.0)
        self.assertEqual(diagnostic.flow_risk_score, 50.0)
        self.assertEqual(diagnostic.flow_risk_reasons, "融資短線升溫")
        self.assertLess(weighted.score, diagnostic.score)

    def test_mixed_pool_scores_each_stock_with_its_own_liquidity_profile(self) -> None:
        dates = pd.bdate_range("2024-01-02", periods=150)
        prices = {
            "2330.TW": _price_frame(dates, close=100, volume=20_000_000),
            "2408.TW": _price_frame(dates, close=80, volume=800_000),
            "9999.TW": _price_frame(dates, close=50, volume=10_000),
        }
        pool_params = default_parameters_for_profile(infer_pool_profile(prices, dates[-1]))

        scores = score_universal_candidates(prices, dates[-1], params=pool_params)

        self.assertEqual(classify_candidate_liquidity_profile(prices["2330.TW"], dates[-1]), POOL_HIGH_LIQUIDITY)
        self.assertEqual(scores["2330.TW"].liquidity_profile, POOL_HIGH_LIQUIDITY)
        self.assertEqual(scores["2330.TW"].size_profile, SIZE_UNKNOWN)
        self.assertEqual(scores["2330.TW"].applied_score_mode, "relative_strength")
        self.assertEqual(classify_candidate_liquidity_profile(prices["2408.TW"], dates[-1]), POOL_STANDARD_LIQUIDITY)
        self.assertEqual(scores["2408.TW"].liquidity_profile, POOL_STANDARD_LIQUIDITY)
        self.assertEqual(scores["2408.TW"].applied_score_mode, "risk_adjusted")
        self.assertEqual(classify_candidate_liquidity_profile(prices["9999.TW"], dates[-1]), POOL_LOW_LIQUIDITY_OR_MIXED)
        self.assertEqual(scores["9999.TW"].liquidity_profile, POOL_LOW_LIQUIDITY_OR_MIXED)
        self.assertEqual(scores["9999.TW"].reason, "流動性不足")

    def test_size_profile_uses_market_cap_when_available(self) -> None:
        dates = pd.bdate_range("2024-01-02", periods=150)
        prices = {
            "2330.TW": _price_frame(dates, close=100, volume=20_000_000),
            "2454.TW": _price_frame(dates, close=80, volume=5_000_000),
            "2408.TW": _price_frame(dates, close=60, volume=3_000_000),
            "9999.TW": _price_frame(dates, close=50, volume=3_000_000),
        }
        pool_params = default_parameters_for_profile(infer_pool_profile(prices, dates[-1]))

        scores = score_universal_candidates(
            prices,
            dates[-1],
            params=pool_params,
            market_cap_by_ticker={
                "2330.TW": 2_000_000_000_000,
                "2454.TW": 200_000_000_000,
                "2408.TW": 20_000_000_000,
                "9999.TW": 2_000_000_000,
            },
        )

        self.assertEqual(scores["2330.TW"].size_profile, SIZE_LARGE_CAP)
        self.assertEqual(scores["2454.TW"].size_profile, SIZE_MID_CAP)
        self.assertEqual(scores["2408.TW"].size_profile, SIZE_SMALL_CAP)
        self.assertEqual(scores["9999.TW"].size_profile, SIZE_MICRO_CAP)
        self.assertEqual(scores["2330.TW"].size_basis, "market_cap_twd")

    def test_size_profile_reads_latest_market_cap_column_without_future_data(self) -> None:
        dates = pd.bdate_range("2024-01-02", periods=150)
        prices = _price_frame(dates, close=100, volume=20_000_000)
        prices["market_cap_twd"] = 40_000_000_000
        prices.loc[dates[-1], "market_cap_twd"] = 80_000_000_000
        prices.loc[pd.Timestamp("2025-01-02"), "market_cap_twd"] = 1_000_000_000_000

        size_profile, market_cap, size_basis = classify_candidate_size_profile(prices, dates[-1])

        self.assertEqual(size_profile, SIZE_MID_CAP)
        self.assertEqual(market_cap, 80_000_000_000)
        self.assertEqual(size_basis, "market_cap_twd")


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
