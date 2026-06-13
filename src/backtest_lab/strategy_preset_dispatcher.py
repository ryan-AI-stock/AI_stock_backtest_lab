from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class StrategyPresetSpec:
    preset: str
    label: str
    engine_module: str
    workflow_file: str
    report_line: str
    operational_observation: bool
    public_scorecard: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


PRESET_SPECS: dict[str, StrategyPresetSpec] = {
    "ai_theme_large_cap_v20260613": StrategyPresetSpec(
        preset="ai_theme_large_cap_v20260613",
        label="AI中大型權值股池最佳版 v20260613",
        engine_module="backtest_lab.stock_pool_observation",
        workflow_file="stock_pool_observation.yml",
        report_line="stock_pool_observation",
        operational_observation=True,
    ),
    "best_v20260605": StrategyPresetSpec(
        preset="best_v20260605",
        label="AI中大型權值股最佳版 v20260605",
        engine_module="backtest_lab.stock_pool_observation",
        workflow_file="stock_pool_observation.yml",
        report_line="stock_pool_observation",
        operational_observation=True,
    ),
    "radar_core_mid_small_calibrated_v1": StrategyPresetSpec(
        preset="radar_core_mid_small_calibrated_v1",
        label="雷達中小型校準版",
        engine_module="backtest_lab.stock_pool_observation",
        workflow_file="stock_pool_observation.yml",
        report_line="stock_pool_observation",
        operational_observation=True,
    ),
    "universal_pool_custom": StrategyPresetSpec(
        preset="universal_pool_custom",
        label="通用股票池預設參數版",
        engine_module="backtest_lab.stock_pool_observation",
        workflow_file="stock_pool_observation.yml",
        report_line="stock_pool_observation",
        operational_observation=True,
    ),
    "core_defensive_style_v1": StrategyPresetSpec(
        preset="core_defensive_style_v1",
        label="核心防守風格池 v1",
        engine_module="backtest_lab.stock_pool_observation",
        workflow_file="stock_pool_observation.yml",
        report_line="stock_pool_observation",
        operational_observation=True,
    ),
    "delayed_public_scorecard_v1": StrategyPresetSpec(
        preset="delayed_public_scorecard_v1",
        label="模型延遲公開成績單",
        engine_module="backtest_lab.model_scorecard_report",
        workflow_file="model_scorecard_report.yml",
        report_line="delayed_public_scorecard",
        operational_observation=False,
        public_scorecard=True,
    ),
}


def resolve_strategy_preset(preset: str | None) -> StrategyPresetSpec:
    key = str(preset or "universal_pool_custom").strip() or "universal_pool_custom"
    if key not in PRESET_SPECS:
        raise ValueError(f"Unsupported strategy_preset: {key}")
    return PRESET_SPECS[key]


def dispatch_pool(pool: dict[str, Any]) -> dict[str, Any]:
    spec = resolve_strategy_preset(pool.get("strategy_preset"))
    return {
        "pool_id": pool.get("pool_id", ""),
        "pool_name": pool.get("name", ""),
        "strategy_preset": spec.preset,
        "strategy_label": spec.label,
        "engine_module": spec.engine_module,
        "workflow_file": spec.workflow_file,
        "report_line": spec.report_line,
        "operational_observation": bool(pool.get("operational_observation", spec.operational_observation)),
        "public_scorecard": spec.public_scorecard,
    }


def dispatch_pools(pools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [dispatch_pool(pool) for pool in pools]
