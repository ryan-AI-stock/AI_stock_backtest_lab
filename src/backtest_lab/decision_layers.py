from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


FORMAL_TRADE_SIGNAL = "formal_trade_signal"
CANDIDATE_SOURCE = "candidate_source"
SHADOW_OVERLAY = "shadow_overlay"
DIAGNOSTIC = "diagnostic"
REPORT_WORDING = "report_wording"
DATA_READINESS = "data_readiness"

VALID_DECISION_LAYERS = {
    FORMAL_TRADE_SIGNAL,
    CANDIDATE_SOURCE,
    SHADOW_OVERLAY,
    DIAGNOSTIC,
    REPORT_WORDING,
    DATA_READINESS,
}


@dataclass(frozen=True)
class DecisionLayerMetadata:
    decision_layer: str
    active_in_trade_decision: bool
    source_module: str
    signal_date: str = ""
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        validate_decision_layer(self.decision_layer)
        return asdict(self)


@dataclass(frozen=True)
class ModelLayerAuditItem:
    layer_name: str
    decision_layer: str
    order: int
    gate_type: str
    weight: float | None
    enabled: bool
    input_columns: list[str] = field(default_factory=list)
    data_source: str = ""
    used_by_formal_trade: bool = False
    report_only: bool = False
    known_limitations: str = ""
    source_module: str = ""

    def to_dict(self) -> dict[str, Any]:
        validate_decision_layer(self.decision_layer)
        return asdict(self)


def validate_decision_layer(decision_layer: str) -> None:
    if decision_layer not in VALID_DECISION_LAYERS:
        valid = ", ".join(sorted(VALID_DECISION_LAYERS))
        raise ValueError(f"Unknown decision layer: {decision_layer}; valid layers: {valid}")


def decision_layer_metadata(
    *,
    decision_layer: str,
    active_in_trade_decision: bool,
    source_module: str,
    signal_date: str = "",
    notes: str = "",
) -> dict[str, Any]:
    return DecisionLayerMetadata(
        decision_layer=decision_layer,
        active_in_trade_decision=active_in_trade_decision,
        source_module=source_module,
        signal_date=signal_date,
        notes=notes,
    ).to_dict()


def default_stock_pool_model_layer_audit(
    *,
    signal_date: str,
    generated_pools: list[dict[str, Any]],
    risk_factor_sources: dict[str, str] | None = None,
    valuation_source: str = "",
) -> dict[str, Any]:
    risk_factor_sources = risk_factor_sources or {}
    items = [
        ModelLayerAuditItem(
            layer_name="data_readiness",
            decision_layer=DATA_READINESS,
            order=1,
            gate_type="availability_check",
            weight=None,
            enabled=True,
            input_columns=["price", "resolved_symbols", "signal_date"],
            data_source="price_cache / pool_store / dynamic_constituents",
            used_by_formal_trade=True,
            known_limitations="資料缺漏時應跳過或標記，不應用舊資料硬補成正式訊號。",
            source_module="stock_pool_observation",
        ),
        ModelLayerAuditItem(
            layer_name="market_regime_and_risk_budget",
            decision_layer=FORMAL_TRADE_SIGNAL,
            order=2,
            gate_type="regime_gate",
            weight=None,
            enabled=any(_pool_is_formal(item) for item in generated_pools),
            input_columns=["close", "momentum", "attack_gate", "risk_off"],
            data_source="frozen_strategy_monitor / regime_mode_switch",
            used_by_formal_trade=True,
            known_limitations="00631L 是槓桿曝險/等待工具，不得描述成一般低風險防守資產。",
            source_module="frozen_strategy_monitor",
        ),
        ModelLayerAuditItem(
            layer_name="stock_pool_candidate_source",
            decision_layer=CANDIDATE_SOURCE,
            order=3,
            gate_type="pool_membership",
            weight=None,
            enabled=True,
            input_columns=["resolved_symbols", "candidate_review"],
            data_source="stock_pool_store / formal_radar_candidates / tw50_constituents",
            used_by_formal_trade=False,
            report_only=False,
            known_limitations="候選來源只定義可觀察名單，不等於正式交易目標。",
            source_module="stock_pool_observation",
        ),
        ModelLayerAuditItem(
            layer_name="individual_strength_ranking",
            decision_layer=CANDIDATE_SOURCE,
            order=4,
            gate_type="score_rank",
            weight=1.0,
            enabled=True,
            input_columns=["ret20", "ret60", "ret120", "vol20", "drawdown20"],
            data_source="universal_pool_strategy / regime ranking",
            used_by_formal_trade=any(_pool_is_formal(item) for item in generated_pools),
            known_limitations="強弱排名本身是觀察清單；正式目標仍要看對應策略閘門。",
            source_module="universal_pool_strategy",
        ),
        ModelLayerAuditItem(
            layer_name="chip_margin_overheat",
            decision_layer=DIAGNOSTIC,
            order=5,
            gate_type="diagnostic_overlay",
            weight=0.0,
            enabled=bool(risk_factor_sources),
            input_columns=["institutional_flow", "margin_short", "borrow_lending", "day_trading", "sentiment"],
            data_source=", ".join(sorted(risk_factor_sources)) if risk_factor_sources else "",
            used_by_formal_trade=False,
            report_only=False,
            known_limitations="目前先標為 diagnostic；不得未經回測直接升格正式交易閘門。",
            source_module="risk_factor_source",
        ),
        ModelLayerAuditItem(
            layer_name="official_margin_short_ingestion",
            decision_layer=DIAGNOSTIC,
            order=6,
            gate_type="schema_readiness",
            weight=0.0,
            enabled=bool(risk_factor_sources.get("margin_short")),
            input_columns=[
                "date",
                "ticker",
                "margin_balance",
                "short_balance",
                "margin_balance_5d_change_pct",
                "short_balance_5d_change_pct",
            ],
            data_source=risk_factor_sources.get("margin_short", ""),
            used_by_formal_trade=False,
            report_only=False,
            known_limitations="TWSE/TPEx 逐檔融資融券資料先接入 diagnostic schema；未經 challenger 驗證不得升正式交易閘門。",
            source_module="margin_short_ingestion_spec",
        ),
        ModelLayerAuditItem(
            layer_name="crowding_chip_flow_challengers",
            decision_layer=SHADOW_OVERLAY,
            order=7,
            gate_type="shadow_challenger_validation",
            weight=None,
            enabled=bool(risk_factor_sources),
            input_columns=[
                "institutional_flows_daily",
                "margin_short_daily",
                "day_trading_daily",
                "price_confirmation",
            ],
            data_source=", ".join(sorted(risk_factor_sources)) if risk_factor_sources else "",
            used_by_formal_trade=False,
            report_only=False,
            known_limitations="籌碼/過熱 challenger 只作 shadow 驗證；未經 walk-forward/out-of-sample 證明，不得升正式交易邏輯。",
            source_module="overlay_challenger_registry",
        ),
        ModelLayerAuditItem(
            layer_name="valuation_sanity",
            decision_layer=DIAGNOSTIC,
            order=8,
            gate_type="diagnostic_overlay",
            weight=0.0,
            enabled=bool(valuation_source),
            input_columns=["fair_price", "buy_price", "safety_margin_pct"],
            data_source=valuation_source,
            used_by_formal_trade=False,
            report_only=False,
            known_limitations="估值資料先做安全邊際註記；不同策略型態需另設權重與驗證。",
            source_module="valuation_source",
        ),
        ModelLayerAuditItem(
            layer_name="three_pool_consensus",
            decision_layer=CANDIDATE_SOURCE,
            order=9,
            gate_type="consensus_observation",
            weight=None,
            enabled=True,
            input_columns=["top_ticker", "vote_group"],
            data_source="stock_pool_consensus",
            used_by_formal_trade=False,
            report_only=False,
            known_limitations="2/3 表決是跨池觀察結論，不得自動視為正式交易目標。",
            source_module="stock_pool_consensus",
        ),
        ModelLayerAuditItem(
            layer_name="report_wording",
            decision_layer=REPORT_WORDING,
            order=10,
            gate_type="wording_boundary",
            weight=None,
            enabled=True,
            input_columns=["action_state", "source_metadata", "consensus"],
            data_source="report templates",
            used_by_formal_trade=False,
            report_only=True,
            known_limitations="報告應維持 AI 輔助觀察語氣，不得喊買賣或保證績效。",
            source_module="stock_pool_observation",
        ),
    ]
    return {
        "schema_version": 1,
        "signal_date": signal_date,
        "items": [item.to_dict() for item in items],
    }


def write_model_layer_audit(path: str | Path, audit: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")


def _pool_is_formal(item: dict[str, Any]) -> bool:
    return bool(item.get("active_in_trade_decision")) or item.get("decision_layer") == FORMAL_TRADE_SIGNAL
