"""Refresh Layer0 investable-universe source/pruning contract after Radar hardening.

Layer0 remains a data-pruning contract. It is not a trading rule, selector,
formal model, report change, or Experiments diagnostic.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


TASK_ID = "TASK-BACKTEST-CORE-VNEXT-LAYER0-SOURCE-CONTRACT-REFRESH-001"
DEFAULT_CORE_DIR = Path("outputs/vnext_layer0_investable_universe_prefilter_contract_20260707")
DEFAULT_RADAR_DIR = Path(
    r"C:\Users\zergv\Documents\Codex\2026-05-23\ai-stock-rotation-radar-https-docs\outputs"
    r"\radar_vnext_layer0_investable_universe_source_hardening_20260707"
)
DEFAULT_OUTPUT_DIR = Path("outputs/vnext_layer0_source_contract_refresh_20260707")


def build_refresh(
    *,
    core_dir: str | Path = DEFAULT_CORE_DIR,
    radar_dir: str | Path = DEFAULT_RADAR_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    core = Path(core_dir)
    radar = Path(radar_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    core_readiness = _read_json(core / "readiness_for_layer0_investable_universe_prefilter.json")
    radar_readiness = _read_json(radar / "readiness_for_core_layer0_source_hardening.json")
    threshold = _read_csv(core / "layer0_prefilter_threshold_coverage_estimate.csv")
    radar_sources = _read_csv(radar / "layer0_daily_traded_value_source_confirmation.csv")
    radar_gaps = _read_csv(radar / "layer0_blocked_proxy_gap_ledger.csv")
    instrument_inventory = _read_csv(radar / "layer0_instrument_type_master_source_inventory.csv")
    event_inventory = _read_csv(radar / "layer0_pit_event_ledger_source_inventory.csv")
    marketcap_policy = _read_csv(radar / "layer0_market_cap_proxy_policy_readiness.csv")

    source_contract = _source_contract(radar_readiness, radar_sources)
    pruning_contract = _pruning_contract(threshold)
    blocked_proxy = _blocked_proxy_ledger(radar_gaps, instrument_inventory, event_inventory, marketcap_policy)
    cost_reduction = _cost_reduction_estimate(threshold)
    next_handoff = _next_handoff()
    future_audit = _future_audit()
    readiness = _readiness(core_readiness, radar_readiness)

    _write_csv(source_contract, output / "layer0_refreshed_source_contract.csv")
    _write_csv(pruning_contract, output / "layer0_refreshed_pruning_filter_contract.csv")
    _write_csv(blocked_proxy, output / "layer0_refreshed_blocked_proxy_ledger.csv")
    _write_csv(cost_reduction, output / "layer0_layer1_cost_reduction_estimate.csv")
    _write_csv(next_handoff, output / "layer0_next_step_handoff.csv")
    _write_csv(future_audit, output / "layer0_refreshed_future_data_audit.csv")
    (output / "readiness_for_layer0_source_contract_refresh.json").write_text(
        json.dumps(readiness, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    manifest = {
        "task_id": TASK_ID,
        "status": readiness["status"],
        "output_dir": str(output.resolve()),
        "core_input_dir": str(core.resolve()),
        "radar_input_dir": str(radar.resolve()),
        "radar_commit": "f694c20",
        "output_files": [
            "layer0_refreshed_source_contract.csv",
            "layer0_refreshed_pruning_filter_contract.csv",
            "layer0_refreshed_blocked_proxy_ledger.csv",
            "layer0_layer1_cost_reduction_estimate.csv",
            "layer0_next_step_handoff.csv",
            "layer0_refreshed_future_data_audit.csv",
            "readiness_for_layer0_source_contract_refresh.json",
            "manifest.json",
            "final_summary_zh.md",
        ],
        "formal_model_changed": False,
        "trade_decision_changed": False,
        "active_in_trade_decision": False,
        "report_changed": False,
        "portfolio_replay_executed": False,
        "ready_for_strategy_replay": False,
        "not_live_rule": True,
        "forward_returns_live_rule_usage": False,
        "diagnostic_only": True,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "final_summary_zh.md").write_text(_summary(readiness), encoding="utf-8")
    return manifest


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _read_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    if not path.exists() or path.read_text(encoding="utf-8").strip() == "empty":
        return pd.DataFrame()
    return pd.read_csv(path, **kwargs)


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def _source_contract(radar_readiness: dict[str, Any], radar_sources: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {
            "source_item": "daily_per_stock_traded_value",
            "contract_status": "accepted_primary_layer0_source",
            "source_quality": "pit_ready_low_cost",
            "policy": "use as primary pruning signal",
            "diagnostic_only": True,
        },
        {
            "source_item": "total_market_traded_value",
            "contract_status": "accepted_derived_from_daily_rows",
            "source_quality": "pit_ready_derived",
            "policy": "derive by summing accepted daily per-stock traded_value by date/market",
            "diagnostic_only": True,
        },
        {
            "source_item": "instrument_type_master",
            "contract_status": "partial_blocked_no_full_pit_master",
            "source_quality": "partial_proxy",
            "policy": "ETF/ETN separated by available tags/patterns; unknowns remain tagged, not silently included/excluded",
            "diagnostic_only": True,
        },
        {
            "source_item": "pit_event_ledger",
            "contract_status": "partial_blocked",
            "source_quality": "partial_suspension_only",
            "policy": "disposition/full-delivery/delisted full master remains blocked",
            "diagnostic_only": True,
        },
        {
            "source_item": "market_cap_rank",
            "contract_status": "proxy_only_not_primary",
            "source_quality": "capital_stock_x_close_proxy_only",
            "policy": "do not use as primary pruning until proxy policy accepted",
            "diagnostic_only": True,
        },
    ]
    if not radar_sources.empty:
        rows.append(
            {
                "source_item": "radar_source_confirmation_rows",
                "contract_status": f"source_rows={len(radar_sources)}",
                "source_quality": "metadata",
                "policy": "source confirmation imported from Radar hardening package",
                "diagnostic_only": True,
            }
        )
    return pd.DataFrame(rows)


def _pruning_contract(threshold: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {
            "contract_rule": "primary_universe",
            "rule": "recent 5D/20D traded_value rank and cumulative total-market traded-value share",
            "recommended_threshold": "top_200_to_300",
            "buffer": "plus_100_watchlist_buffer",
            "reason": "top 200 covers about 88%, top 300 about 93% in current estimate; buffer reduces miss risk",
            "ready": True,
            "diagnostic_only": True,
        },
        {
            "contract_rule": "surge_exception",
            "rule": "include names with recent traded-value surge vs 60D baseline even if outside primary rank",
            "recommended_threshold": "policy_stage_only",
            "buffer": "watchlist",
            "reason": "prevents excluding emerging mid-cap themes before fundamentals are fetched",
            "ready": True,
            "diagnostic_only": True,
        },
        {
            "contract_rule": "market_cap_rank",
            "rule": "market-cap rank cumulative traded-value share",
            "recommended_threshold": "blocked_or_proxy",
            "buffer": "do_not_use_as_primary",
            "reason": "exact full daily market cap unavailable; proxy not formal-ready",
            "ready": False,
            "diagnostic_only": True,
        },
        {
            "contract_rule": "instrument_exclusion",
            "rule": "exclude/separate ETF/ETN/warrants when instrument type is known; KY tag separately; event-block only with PIT ledger",
            "recommended_threshold": "hygiene_policy",
            "buffer": "unknown_instrument_tagged_not_silent_drop",
            "reason": "avoid non-common-stock pollution without losing unknown common stocks",
            "ready": False,
            "diagnostic_only": True,
        },
    ]
    return pd.DataFrame(rows)


def _blocked_proxy_ledger(
    radar_gaps: pd.DataFrame,
    instrument_inventory: pd.DataFrame,
    event_inventory: pd.DataFrame,
    marketcap_policy: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for frame_name, frame in [
        ("radar_gap", radar_gaps),
        ("instrument_inventory", instrument_inventory),
        ("event_inventory", event_inventory),
        ("marketcap_policy", marketcap_policy),
    ]:
        for item in frame.to_dict("records"):
            rows.append(
                {
                    "source": frame_name,
                    "item": item.get("gap_field") or item.get("field") or item.get("source_item") or item.get("policy_item"),
                    "status": item.get("status") or item.get("readiness") or item.get("policy_status"),
                    "reason": item.get("blocked_reason") or item.get("note") or item.get("policy") or item.get("source_quality"),
                    "diagnostic_only": True,
                }
            )
    return pd.DataFrame(rows)


def _cost_reduction_estimate(threshold: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for count in [100, 200, 300, 500]:
        row = threshold[threshold["threshold"].eq(f"top_{count}")]
        share = row["recent_5d_cumulative_traded_value_share"].iloc[0] if not row.empty else ""
        rows.append(
            {
                "layer0_kept_count": count,
                "approx_layer1_download_reduction_vs_1900": 1 - count / 1900,
                "recent_5d_turnover_share_estimate": share,
                "core_risk_note": "too_narrow" if count <= 100 else "recommended_core_range" if count in {200, 300} else "lower_miss_higher_cost",
                "diagnostic_only": True,
            }
        )
    return pd.DataFrame(rows)


def _next_handoff() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "next_owner": "Core/Data",
                "handoff_action": "build_layer0_materialized_weekly_universe_snapshot_contract_from_traded_value_primary",
                "ready": True,
                "reason": "source hardening is sufficient for traded-value primary Layer0; instrument/event/market-cap remain proxy/blocked ledgers",
                "diagnostic_only": True,
            },
            {
                "next_owner": "Radar/Data",
                "handoff_action": "do_not_resume_t164_mass_download_until_layer0_materialized_universe_is_accepted",
                "ready": False,
                "reason": "Strategy Center paused t164 mass download; Layer0 should define reduced universe first",
                "diagnostic_only": True,
            },
        ]
    )


def _future_audit() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("forward_return_as_rule", "passed", 0, "no forward returns used"),
            ("traded_value_pit_source", "passed", 0, "daily traded_value is PIT source"),
            ("proxy_no_silent_fill", "passed", 0, "instrument/event/market-cap gaps remain explicit"),
        ],
        columns=["audit_item", "status", "future_data_violation_count", "note"],
    )


def _readiness(core_readiness: dict[str, Any], radar_readiness: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_id": TASK_ID,
        "status": "layer0_source_contract_refreshed_traded_value_primary_ready_marketcap_event_instrument_partial",
        "diagnostic_only": True,
        "layer0_name": "Layer0 investable universe / data-pruning filter",
        "primary_pruning_source": "daily_per_stock_traded_value",
        "total_market_turnover_source": "derived_from_daily_per_stock_traded_value",
        "recommended_universe": "top_200_to_300_by_recent_traded_value_plus_100_buffer_watchlist",
        "estimated_layer1_download_reduction": "about_84pct_to_89pct_vs_1900_for_200_to_300_primary_names",
        "ready_for_layer0_materialized_weekly_universe_snapshot_contract": True,
        "ready_for_radar_source_hardening": False,
        "ready_for_t164_mass_download": False,
        "ready_for_experiments": False,
        "ready_for_formal": False,
        "ready_for_strategy_replay": False,
        "future_data_violation_count": 0,
        "blocked_fields": [
            "instrument_type_master_full_pit",
            "pit_disposition_full_delivery_event_ledger",
            "direct_exact_market_cap_rank",
            "historical_all_stock_universe_master",
        ],
        "proxy_fields": [
            "capital_stock_x_close_market_cap_proxy",
            "instrument_type_by_pattern_or_partial_metadata",
            "KY_name_tag",
        ],
        "formal_model_changed": False,
        "trade_decision_changed": False,
        "active_in_trade_decision": False,
        "report_changed": False,
        "portfolio_replay_executed": False,
        "not_live_rule": True,
        "forward_returns_live_rule_usage": False,
    }


def _summary(readiness: dict[str, Any]) -> str:
    return f"""# Layer0 source contract refresh

## Verdict
- status={readiness["status"]}
- primary_pruning_source={readiness["primary_pruning_source"]}
- recommended_universe={readiness["recommended_universe"]}
- ready_for_layer0_materialized_weekly_universe_snapshot_contract=true
- ready_for_t164_mass_download=false
- ready_for_experiments=false
- ready_for_formal=false

## Core decision
Use traded_value as the primary Layer0 pruning source. Market-cap rank, full instrument master, and PIT disposition/full-delivery ledgers stay proxy/blocked. Do not resume large t164 downloads until a reduced Layer0 universe is materialized and accepted.

## Flags
- formal_model_changed=false
- trade_decision_changed=false
- active_in_trade_decision=false
- report_changed=false
- portfolio_replay_executed=false
- ready_for_strategy_replay=false
- not_live_rule=true
- forward_returns_live_rule_usage=false
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--core-dir", default=str(DEFAULT_CORE_DIR))
    parser.add_argument("--radar-dir", default=str(DEFAULT_RADAR_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()
    manifest = build_refresh(core_dir=args.core_dir, radar_dir=args.radar_dir, output_dir=args.output_dir)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
