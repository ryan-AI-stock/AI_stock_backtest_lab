from __future__ import annotations

from copy import deepcopy
from typing import Any


FORMAL_MODEL_TARGET = "combined_cap40_confirmation1_base"
FORMAL_MODEL_ROUTE = "pool1_primary_pool2_confirmation_cap40"
FORMAL_MODEL_EFFECTIVE_DATE = "2026-06-26"


FORMAL_MODEL_CONTRACT: dict[str, Any] = {
    "formal_model_target": FORMAL_MODEL_TARGET,
    "formal_model_route": FORMAL_MODEL_ROUTE,
    "formal_model_effective_date": FORMAL_MODEL_EFFECTIVE_DATE,
    "three_pool_formal_route_abandoned": True,
    "pool1_role": "primary_attack_selector",
    "pool2_role": "confirmation_and_risk_layer",
    "pool2_policy": "confirmation_1_signal_day_when_pool2_disagrees_with_pool1",
    "pool3_role": "shadow_or_diagnostic_only",
    "pool3_shadow_used_as_formal": False,
    "leveraged_etf_cap": {"ticker": "00631L.TW", "max_weight": 0.40, "residual": "cash"},
    "market_exposure_override_absorbed": False,
    "0050x2_opportunity_label_active_in_trade_decision": False,
    "rr_partial_switch_used_in_performance": False,
    "valuation_used": False,
    "h3_used": False,
    "uses_forward_return_as_rule": False,
}


def get_formal_model_contract() -> dict[str, Any]:
    return deepcopy(FORMAL_MODEL_CONTRACT)


def formal_model_report_description() -> str:
    return (
        "正式 baseline 已切換為 Pool1 主攻 selector + PIT-ready Pool2 確認/風控層；"
        "使用 combined_cap40_confirmation1_base，00631L 目標權重上限 40%。"
        "三池表決不再作為正式 performance selector。"
    )
