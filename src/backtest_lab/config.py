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
    comparison_benchmarks: tuple[str, ...]
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
    active_group_id: str
    warmup_start_date: str
    start_date: str
    end_date: str
    initial_cash_twd: int
    execution: ExecutionConfig
    cost_model: TaiwanCostModel
    manual_splits: dict[str, tuple[dict[str, float | str], ...]]
    reference_values: dict[str, dict[str, float | str]]
    groups: tuple[GroupConfig, ...]
    strategies: tuple[str, ...]

    def group_by_id(self, group_id: str) -> GroupConfig:
        for group in self.groups:
            if group.group_id == group_id:
                return group
        raise KeyError(f"Unknown group: {group_id}")

    @property
    def active_group(self) -> GroupConfig:
        return self.group_by_id(self.active_group_id)


def load_config(path: str | Path) -> BacktestConfig:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return parse_config(raw)


def parse_config(raw: dict[str, Any]) -> BacktestConfig:
    costs = raw["costs"]
    execution = raw["execution"]
    corporate_actions = raw.get("corporate_actions", {})
    groups = tuple(
        GroupConfig(
            group_id=group["id"],
            benchmark=group["benchmark"],
            benchmark_label=group["benchmark_label"],
            comparison_benchmarks=tuple(group.get("comparison_benchmarks", [group["benchmark"]])),
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
        active_group_id=raw.get("active_group_id", groups[0].group_id),
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
        manual_splits={
            ticker: tuple(events)
            for ticker, events in corporate_actions.get("manual_splits", {}).items()
        },
        reference_values=corporate_actions.get("reference_values", {}),
        groups=groups,
        strategies=tuple(raw["strategies"]),
    )
