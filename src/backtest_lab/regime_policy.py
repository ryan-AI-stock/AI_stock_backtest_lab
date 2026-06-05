from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class StrategyPolicy:
    strategy_id: str
    regime: str
    allow_new_entry: bool
    allow_rebalance: bool
    max_equity_exposure: float
    min_cash_ratio: float
    rebalance_frequency: str
    switch_score_margin: float | None
    min_candidate_score: float | None
    product_mode: str
    risk_message: str

    def to_dict(self) -> dict:
        return asdict(self)


_POLICY_TABLE: dict[str, dict[str, dict]] = {
    "daily_strength": {
        "strong_bull": {
            "allow_new_entry": True,
            "allow_rebalance": True,
            "max_equity_exposure": 1.0,
            "rebalance_frequency": "daily",
            "switch_score_margin": 0.0,
            "min_candidate_score": 0.0,
            "product_mode": "積極追強",
            "risk_message": "市場處於強多頭，允許每日追強勢，但仍需注意交易成本與追高風險。",
        },
        "recovery_bull": {
            "allow_new_entry": True,
            "allow_rebalance": True,
            "max_equity_exposure": 0.7,
            "rebalance_frequency": "daily",
            "switch_score_margin": 0.02,
            "min_candidate_score": 0.0,
            "product_mode": "修復追強",
            "risk_message": "市場處於修復多頭，只追明顯強勢標的，不採滿倉模式。",
        },
        "range_bound": {
            "allow_new_entry": True,
            "allow_rebalance": True,
            "max_equity_exposure": 0.4,
            "rebalance_frequency": "cooldown_3d",
            "switch_score_margin": 0.04,
            "min_candidate_score": 0.0,
            "product_mode": "震盪觀察",
            "risk_message": "市場處於震盪盤，追強容易被短線雜訊干擾，需提高換股門檻。",
        },
        "correction_bear": {
            "allow_new_entry": True,
            "allow_rebalance": True,
            "max_equity_exposure": 0.2,
            "rebalance_frequency": "weekly",
            "switch_score_margin": 0.06,
            "min_candidate_score": 0.02,
            "product_mode": "空頭觀察倉",
            "risk_message": "市場處於修正空頭，只允許小部位觀察，不啟用積極追強。",
        },
        "systemic_bear": {
            "allow_new_entry": False,
            "allow_rebalance": False,
            "max_equity_exposure": 0.0,
            "rebalance_frequency": "watch_only",
            "switch_score_margin": None,
            "min_candidate_score": None,
            "product_mode": "系統性空頭防守",
            "risk_message": "市場處於系統性空頭，停止追強，只輸出觀察名單。",
        },
    },
    "weekly_rotation": {
        "strong_bull": {
            "allow_new_entry": True,
            "allow_rebalance": True,
            "max_equity_exposure": 1.0,
            "rebalance_frequency": "weekly",
            "switch_score_margin": 0.0,
            "min_candidate_score": 0.0,
            "product_mode": "週頻正常輪動",
            "risk_message": "市場處於強多頭，週頻紀律輪動正常運作。",
        },
        "recovery_bull": {
            "allow_new_entry": True,
            "allow_rebalance": True,
            "max_equity_exposure": 0.8,
            "rebalance_frequency": "weekly",
            "switch_score_margin": 0.02,
            "min_candidate_score": 0.0,
            "product_mode": "週頻修復輪動",
            "risk_message": "市場處於修復多頭，允許週頻輪動，但不採滿倉進攻。",
        },
        "range_bound": {
            "allow_new_entry": True,
            "allow_rebalance": True,
            "max_equity_exposure": 0.5,
            "rebalance_frequency": "biweekly",
            "switch_score_margin": 0.04,
            "min_candidate_score": 0.0,
            "product_mode": "週頻震盪過濾",
            "risk_message": "市場處於震盪盤，週頻輪動需提高換股門檻，避免來回交易。",
        },
        "correction_bear": {
            "allow_new_entry": True,
            "allow_rebalance": True,
            "max_equity_exposure": 0.3,
            "rebalance_frequency": "weekly",
            "switch_score_margin": 0.06,
            "min_candidate_score": 0.03,
            "product_mode": "週頻防守觀察",
            "risk_message": "市場處於修正空頭，只允許最強且趨勢明確標的小部位觀察。",
        },
        "systemic_bear": {
            "allow_new_entry": False,
            "allow_rebalance": False,
            "max_equity_exposure": 0.0,
            "rebalance_frequency": "watch_only",
            "switch_score_margin": None,
            "min_candidate_score": None,
            "product_mode": "週頻系統性防守",
            "risk_message": "市場處於系統性空頭，週頻策略只輸出觀察名單，不建立新倉。",
        },
    },
}


def policy_for(strategy_id: str, regime: str) -> StrategyPolicy:
    try:
        raw = _POLICY_TABLE[strategy_id][regime]
    except KeyError as exc:
        raise ValueError(f"Unsupported policy: strategy_id={strategy_id}, regime={regime}") from exc
    max_exposure = float(raw["max_equity_exposure"])
    return StrategyPolicy(
        strategy_id=strategy_id,
        regime=regime,
        allow_new_entry=bool(raw["allow_new_entry"]),
        allow_rebalance=bool(raw["allow_rebalance"]),
        max_equity_exposure=max_exposure,
        min_cash_ratio=round(1.0 - max_exposure, 4),
        rebalance_frequency=str(raw["rebalance_frequency"]),
        switch_score_margin=raw["switch_score_margin"],
        min_candidate_score=raw["min_candidate_score"],
        product_mode=str(raw["product_mode"]),
        risk_message=str(raw["risk_message"]),
    )
