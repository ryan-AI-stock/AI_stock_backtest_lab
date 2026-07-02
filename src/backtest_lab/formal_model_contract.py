from __future__ import annotations

from copy import deepcopy
from typing import Any


FORMAL_MODEL_TARGET = "pool1_pool2_confirmation1_base"
FORMAL_MODEL_ROUTE = "pool1_primary_pool2_confirmation"
FORMAL_MODEL_EFFECTIVE_DATE = "2026-07-02"


FORMAL_MODEL_CONTRACT: dict[str, Any] = {
    "formal_model_target": FORMAL_MODEL_TARGET,
    "formal_model_route": FORMAL_MODEL_ROUTE,
    "formal_model_effective_date": FORMAL_MODEL_EFFECTIVE_DATE,
    "three_pool_formal_route_abandoned": True,
    "pool1_role": "primary_attack_selector",
    "pool2_role": "confirmation_and_risk_layer",
    "pool2_policy": "confirmation_1_signal_day_when_pool2_disagrees_with_pool1",
    "formal_execution_risk_control": "no_target_cash_all",
    "no_target_risk_off_active": True,
    "no_target_risk_off_policy": "cash_all",
    "no_target_execution_policy": "exit_to_cash",
    "bug_cash_mapping_used_as_baseline": False,
    "pool3_role": "shadow_or_diagnostic_only",
    "pool3_shadow_used_as_formal": False,
    "leveraged_etf_max_weight_limit": None,
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
        "目前正式版以主攻池提出觀察標的，確認池負責做風險確認；"
        "若模型未找到合格攻擊標的，正式啟用風險控管空手，直到下一個正式目標出現。"
    )
