from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from backtest_lab.costs import TaiwanCostModel


@dataclass(frozen=True)
class AssetConfig:
    ticker: str
    label: str
    asset_type: str


@dataclass(frozen=True)
class GroupConfig:
    group_id: str
    benchmark: str
    benchmark_label: str
    assets: tuple[AssetConfig, ...]

    def asset_type(self, ticker: str) -> str:
        for asset in self.assets:
            if asset.ticker == ticker:
                return asset.asset_type
        raise KeyError(f"Unknown asset in group {self.group_id}: {ticker}")


@dataclass(frozen=True)
class ExecutionConfig:
    signal_timing: str
    trade_timing: str
    benchmark_entry_rule: str
    initial_entry_rule: str
    max_trades_per_day: int
    allow_no_trade: bool


@dataclass(frozen=True)
class BacktestConfig:
    project: str
    episode: str
    warmup_start_date: str
    start_date: str
    end_date: str
    initial_cash_twd: int
    execution: ExecutionConfig
    cost_model: TaiwanCostModel
    groups: tuple[GroupConfig, ...]
    strategies: tuple[str, ...]


def load_config(path: str | Path) -> BacktestConfig:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return parse_config(raw)


def parse_config(raw: dict[str, Any]) -> BacktestConfig:
    costs = raw["costs"]
    execution = raw["execution"]
    groups = tuple(
        GroupConfig(
            group_id=group["id"],
            benchmark=group["benchmark"],
            benchmark_label=group["benchmark_label"],
            assets=tuple(
                AssetConfig(
                    ticker=asset["ticker"],
                    label=asset["label"],
                    asset_type=asset["type"],
                )
                for asset in group["assets"]
            ),
        )
        for group in raw["groups"]
    )
    return BacktestConfig(
        project=raw["project"],
        episode=raw["episode"],
        warmup_start_date=raw["warmup_start_date"],
        start_date=raw["start_date"],
        end_date=raw["end_date"],
        initial_cash_twd=raw["initial_cash_twd"],
        execution=ExecutionConfig(
            signal_timing=execution["signal_timing"],
            trade_timing=execution["trade_timing"],
            benchmark_entry_rule=execution["benchmark_entry"]["selection_rule"],
            initial_entry_rule=execution["initial_entry"]["selection_rule"],
            max_trades_per_day=execution["max_trades_per_day"],
            allow_no_trade=execution["allow_no_trade"],
        ),
        cost_model=TaiwanCostModel(
            broker_fee_rate=costs["broker_fee_rate"],
            broker_fee_discount=costs["broker_fee_discount"],
            minimum_fee_twd=costs["minimum_fee_twd"],
            stock_sell_tax_rate=costs["stock_sell_tax_rate"],
            etf_sell_tax_rate=costs["etf_sell_tax_rate"],
        ),
        groups=groups,
        strategies=tuple(raw["strategies"]),
    )

