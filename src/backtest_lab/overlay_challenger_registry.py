from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from backtest_lab.decision_layers import SHADOW_OVERLAY


@dataclass(frozen=True)
class OverlayChallengerSpec:
    challenger_id: str
    label: str
    decision_layer: str = SHADOW_OVERLAY
    active_in_trade_decision: bool = False
    source_module: str = "institutional_flow_overlay_shadow"
    data_sources_required: tuple[str, ...] = ()
    rule_family: str = ""
    rule_name: str = ""
    exposure_cap: float | None = None
    trigger_summary: str = ""
    promotion_gate: str = (
        "Only eligible for formal-engine proposal after out-of-sample or walk-forward validation "
        "improves risk-adjusted performance and does not degrade frozen-baseline objectives."
    )
    known_limitations: str = (
        "Shadow approximation scales realized baseline returns; it is not the formal execution engine "
        "and must not be interpreted as an investment instruction."
    )
    parameters: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_overlay_challenger_specs() -> list[OverlayChallengerSpec]:
    from backtest_lab.institutional_flow_overlay_shadow import default_chip_flow_rules, default_overlay_rules

    specs: list[OverlayChallengerSpec] = []
    for rule in default_overlay_rules():
        specs.append(_institutional_spec(rule))
    for rule in default_chip_flow_rules():
        specs.append(_chip_flow_spec(rule))
    return specs


def overlay_challenger_manifest() -> dict[str, Any]:
    specs = build_overlay_challenger_specs()
    return {
        "schema_version": 1,
        "decision_layer": SHADOW_OVERLAY,
        "active_in_trade_decision": False,
        "candidate_count": len(specs),
        "formal_promotion_status": "not_promoted",
        "promotion_boundary": (
            "These challengers are shadow overlays for validation. They do not change frozen_cycle_proven_top1_v1 "
            "or any formal report target unless Core later promotes a versioned challenger with tests and backtest evidence."
        ),
        "challengers": [spec.to_dict() for spec in specs],
    }


def write_overlay_challenger_manifest(path: str | Path) -> dict[str, Any]:
    manifest = overlay_challenger_manifest()
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def _institutional_spec(rule: Any) -> OverlayChallengerSpec:
    parameters = {
        "foreign_sell_days": rule.foreign_sell_days,
        "trust_sell_days": rule.trust_sell_days,
        "require_negative_total_shares": rule.require_negative_total_shares,
        "stock_only": rule.stock_only,
    }
    trigger = f"foreign_sell_days>={rule.foreign_sell_days} or trust_sell_days>={rule.trust_sell_days}"
    if rule.require_negative_total_shares:
        trigger += " with negative total institutional shares"
    return OverlayChallengerSpec(
        challenger_id=f"institutional_overlay_{rule.name}",
        label=f"法人籌碼 shadow：{rule.name}",
        data_sources_required=("institutional_flows_daily",),
        rule_family="institutional_flow",
        rule_name=rule.name,
        exposure_cap=rule.exposure_cap,
        trigger_summary=trigger,
        parameters=parameters,
    )


def _chip_flow_spec(rule: Any) -> OverlayChallengerSpec:
    required = ["institutional_flows_daily"]
    triggers: list[str] = []
    if rule.institutional_foreign_sell_days is not None and rule.institutional_trust_sell_days is not None:
        triggers.append(
            f"institutional foreign>={rule.institutional_foreign_sell_days}d or trust>={rule.institutional_trust_sell_days}d"
        )
    if rule.use_margin_overheat or rule.use_short_pressure:
        required.append("margin_short_daily")
    if rule.use_margin_overheat:
        triggers.append("margin_overheat")
    if rule.use_short_pressure:
        triggers.append("short_lending_pressure")
    if rule.use_day_trading_overheat or rule.min_day_trading_ratio is not None:
        required.append("day_trading_daily")
    if rule.use_day_trading_overheat:
        triggers.append("day_trading_overheat")
    if rule.min_day_trading_ratio is not None:
        triggers.append(f"day_trading_ratio>={rule.min_day_trading_ratio}")
    if rule.require_two_signals:
        triggers.append("requires_two_signals")
    if rule.price_confirmation:
        triggers.append(f"price_confirmation={rule.price_confirmation}")
    return OverlayChallengerSpec(
        challenger_id=f"chip_flow_overlay_{rule.name}",
        label=f"籌碼/過熱 shadow：{rule.name}",
        data_sources_required=tuple(dict.fromkeys(required)),
        rule_family="chip_flow_crowding_overheat",
        rule_name=rule.name,
        exposure_cap=rule.exposure_cap,
        trigger_summary=" + ".join(triggers) if triggers else "no_trigger",
        parameters={
            "institutional_foreign_sell_days": rule.institutional_foreign_sell_days,
            "institutional_trust_sell_days": rule.institutional_trust_sell_days,
            "use_margin_overheat": rule.use_margin_overheat,
            "use_short_pressure": rule.use_short_pressure,
            "use_day_trading_overheat": rule.use_day_trading_overheat,
            "min_day_trading_ratio": rule.min_day_trading_ratio,
            "require_two_signals": rule.require_two_signals,
            "price_confirmation": rule.price_confirmation,
            "stock_only": rule.stock_only,
        },
    )
