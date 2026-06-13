from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from backtest_lab.config import BacktestConfig, GroupConfig, load_config
from backtest_lab.data import load_price_csv, split_adjusted_dividends
from backtest_lab.regime_mode_switch import frozen_cycle_proven_top1_v1_variant, simulate_regime_mode_switch
from backtest_lab.simulation import BacktestResult


DEFAULT_FROZEN_GROUP_ID = "group_c_0050_00631l_plus_mega_caps"


@dataclass(frozen=True)
class FrozenStrategyContext:
    config: BacktestConfig
    group: GroupConfig
    labels: dict[str, str]
    asset_types: dict[str, str]
    prices_by_ticker: dict[str, pd.DataFrame]
    dividends_by_ticker: dict[str, pd.Series]


def load_frozen_strategy_context_from_cache(
    *,
    config_path: str | Path = "configs/ep05_universe.json",
    group_id: str = DEFAULT_FROZEN_GROUP_ID,
    cache_dir: str | Path,
) -> FrozenStrategyContext:
    config = load_config(config_path)
    group = config.group_by_id(group_id)
    labels = {asset.ticker: asset.label for asset in group.assets}
    prices = _load_cached_prices(cache_dir, labels)
    return build_frozen_strategy_context(
        config=config,
        group_id=group_id,
        prices_by_ticker=prices,
    )


def build_frozen_strategy_context(
    *,
    config: BacktestConfig,
    group_id: str = DEFAULT_FROZEN_GROUP_ID,
    prices_by_ticker: dict[str, pd.DataFrame],
) -> FrozenStrategyContext:
    group = config.group_by_id(group_id)
    labels = {asset.ticker: asset.label for asset in group.assets}
    asset_types = {asset.ticker: asset.asset_type for asset in group.assets}
    missing = sorted(set(labels) - set(prices_by_ticker))
    if missing:
        raise ValueError(f"Missing frozen strategy prices: {', '.join(missing)}")
    prices = {ticker: prices_by_ticker[ticker] for ticker in labels}
    dividends = build_dividend_series_by_ticker(
        prices_by_ticker=prices,
        manual_splits=config.manual_splits,
    )
    return FrozenStrategyContext(
        config=config,
        group=group,
        labels=labels,
        asset_types=asset_types,
        prices_by_ticker=prices,
        dividends_by_ticker=dividends,
    )


def build_dividend_series_by_ticker(
    *,
    prices_by_ticker: dict[str, pd.DataFrame],
    manual_splits: dict[str, tuple[dict[str, float | str], ...]] | None,
) -> dict[str, pd.Series]:
    splits = manual_splits or {}
    return {
        ticker: split_adjusted_dividends(frame, splits.get(ticker, ()))
        for ticker, frame in prices_by_ticker.items()
    }


def simulate_frozen_baseline(
    *,
    context: FrozenStrategyContext,
    start_date: str,
    end_date: str,
    initial_cash: float,
    name: str = "frozen_cycle_proven_top1_v1",
) -> BacktestResult:
    return simulate_regime_mode_switch(
        name=name,
        prices_by_ticker=context.prices_by_ticker,
        asset_types=context.asset_types,
        market_prices=context.prices_by_ticker["0050.TW"],
        start_date=start_date,
        end_date=end_date,
        initial_cash=initial_cash,
        cost_model=context.config.cost_model,
        variant=frozen_cycle_proven_top1_v1_variant(),
        dividend_series_by_ticker=context.dividends_by_ticker,
    )


def _load_cached_prices(cache_dir: str | Path, labels: dict[str, str]) -> dict[str, pd.DataFrame]:
    root = Path(cache_dir)
    prices: dict[str, pd.DataFrame] = {}
    missing: list[str] = []
    for ticker in labels:
        path = root / f"{ticker.replace('.', '_')}.csv"
        if not path.exists():
            missing.append(ticker)
            continue
        prices[ticker] = load_price_csv(path)
    if missing:
        raise ValueError(f"Missing cached price files: {', '.join(missing)}")
    return prices
