"""Build vNext theme taxonomy policy-staging readiness artifacts.

This is diagnostic policy staging only. It does not alter formal model,
reports, trade decisions, or execute portfolio/strategy replay.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


TASK_ID = "TASK-BACKTEST-CORE-VNEXT-THEME-TAXONOMY-POLICY-STAGING-READINESS-001"
DEFAULT_RADAR_PACKAGE = Path(
    r"C:\Users\zergv\Documents\Codex\2026-05-23\ai-stock-rotation-radar-https-docs\outputs"
    r"\radar_vnext_theme_taxonomy_higher_quality_source_acquisition_20260706"
)
DEFAULT_OUTPUT_DIR = Path("outputs/vnext_theme_taxonomy_policy_staging_readiness_20260706")


def build_policy_staging_readiness(
    *,
    radar_package_dir: str | Path = DEFAULT_RADAR_PACKAGE,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    radar_dir = Path(radar_package_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    readiness = _read_json(radar_dir / "readiness_for_core_research.json")
    manifest_in = _read_json(radar_dir / "manifest.json")
    human_review = pd.read_csv(radar_dir / "human_review_ai_membership_contract.csv")
    source_quality = pd.read_csv(radar_dir / "source_quality_upgrade_ledger.csv")
    tpex_routes = pd.read_csv(radar_dir / "tpex_sector_route_or_alternative_evidence.csv")
    new_high = pd.read_csv(radar_dir / "rolling_new_high_materialization_contract.csv")

    ai_membership = _human_review_required_ai_membership(human_review)
    twse_proxy = _twse_proxy_comparator_table(source_quality, tpex_routes)
    source_tier = _source_tier_matrix(source_quality, human_review, tpex_routes, new_high)
    scenarios = _scenario_readiness(readiness)
    blocked = _blocked_proxy_ledger(source_quality, tpex_routes, new_high)
    policy_readiness = _policy_readiness_json(readiness, scenarios, source_tier, manifest_in)

    _write_csv(ai_membership, output / "human_review_required_ai_membership_table.csv")
    _write_csv(twse_proxy, output / "twse_official_industry_proxy_comparator_table.csv")
    _write_csv(source_tier, output / "source_tier_matrix.csv")
    _write_csv(scenarios, output / "scenario_readiness.csv")
    _write_csv(blocked, output / "blocked_proxy_fields_ledger.csv")
    (output / "policy_staging_readiness.json").write_text(
        json.dumps(policy_readiness, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    manifest = {
        "task_id": TASK_ID,
        "status": policy_readiness["status"],
        "output_dir": str(output.resolve()),
        "input_radar_package": str(radar_dir.resolve()),
        "output_files": [
            "policy_staging_readiness.json",
            "human_review_required_ai_membership_table.csv",
            "twse_official_industry_proxy_comparator_table.csv",
            "source_tier_matrix.csv",
            "scenario_readiness.csv",
            "blocked_proxy_fields_ledger.csv",
            "manifest.json",
            "final_summary_zh.md",
        ],
        "formal_model_changed": False,
        "trade_decision_changed": False,
        "active_in_trade_decision": False,
        "report_changed": False,
        "portfolio_replay_executed": False,
        "ready_for_strategy_replay": False,
        "diagnostic_only": True,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "final_summary_zh.md").write_text(_summary(policy_readiness, scenarios), encoding="utf-8")
    return manifest


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def _human_review_required_ai_membership(human_review: pd.DataFrame) -> pd.DataFrame:
    out = human_review.copy()
    out["diagnostic_only"] = True
    out["policy_accepted"] = False
    out["ready_for_higher_quality_ai_allocation_rerun"] = False
    out["ready_for_formal"] = False
    out["ready_for_strategy_replay"] = False
    out["blocked_reason"] = "human review contract exists, but review decision is not executed or approved"
    return out


def _twse_proxy_comparator_table(source_quality: pd.DataFrame, tpex_routes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    candidates = source_quality[source_quality["field"].eq("non_ai_market_theme_taxonomy")]
    for item in candidates.itertuples(index=False):
        rows.append(
            {
                "comparator_policy": "twse_official_industry_proxy",
                "source_quality": item.source_quality,
                "rows_or_routes": item.rows_or_routes,
                "accepted_for_diagnostic_before_policy": bool(item.accepted_for_diagnostic),
                "accepted_for_formal": bool(item.accepted_for_formal),
                "policy_accepted": False,
                "ready_for_higher_quality_ai_allocation_rerun": False,
                "ready_for_strategy_replay": False,
                "blocked_reason": item.blocked_reason,
                "diagnostic_only": True,
            }
        )
    for item in tpex_routes.itertuples(index=False):
        rows.append(
            {
                "comparator_policy": item.route_id,
                "source_quality": "blocked",
                "rows_or_routes": None,
                "accepted_for_diagnostic_before_policy": bool(item.accepted_for_diagnostic),
                "accepted_for_formal": bool(item.accepted_for_formal),
                "policy_accepted": False,
                "ready_for_higher_quality_ai_allocation_rerun": False,
                "ready_for_strategy_replay": False,
                "blocked_reason": item.blocked_reason_or_notes,
                "diagnostic_only": True,
            }
        )
    return pd.DataFrame(rows)


def _source_tier_matrix(
    source_quality: pd.DataFrame,
    human_review: pd.DataFrame,
    tpex_routes: pd.DataFrame,
    new_high: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    tier_order = {
        "exact": 1,
        "human_reviewed_pending_policy": 2,
        "higher-quality diagnostic": 3,
        "diagnostic": 4,
        "proxy": 5,
        "blocked": 6,
    }
    for item in source_quality.itertuples(index=False):
        tier = "human_reviewed_pending_policy" if item.field == "dated_ai_membership" else str(item.source_quality)
        rows.append(
            {
                "field": item.field,
                "source_tier": tier,
                "tier_rank": tier_order.get(tier, 99),
                "rows_or_routes": item.rows_or_routes,
                "accepted_for_diagnostic_before_policy": bool(item.accepted_for_diagnostic),
                "accepted_for_formal": bool(item.accepted_for_formal),
                "ready_for_higher_quality_ai_allocation_rerun_before_policy": bool(
                    item.ready_for_higher_quality_ai_allocation_rerun
                ),
                "policy_required": item.field in {"dated_ai_membership", "non_ai_market_theme_taxonomy"},
                "blocked_reason": item.blocked_reason,
                "diagnostic_only": True,
            }
        )
    rows.append(
        {
            "field": "human_review_ai_membership_contract",
            "source_tier": "human_reviewed_pending_policy",
            "tier_rank": tier_order["human_reviewed_pending_policy"],
            "rows_or_routes": int(len(human_review)),
            "accepted_for_diagnostic_before_policy": True,
            "accepted_for_formal": False,
            "ready_for_higher_quality_ai_allocation_rerun_before_policy": False,
            "policy_required": True,
            "blocked_reason": "review decisions are allowed by contract but not executed/approved",
            "diagnostic_only": True,
        }
    )
    rows.append(
        {
            "field": "tpex_route_or_alternative_evidence",
            "source_tier": "blocked",
            "tier_rank": tier_order["blocked"],
            "rows_or_routes": int(len(tpex_routes)),
            "accepted_for_diagnostic_before_policy": False,
            "accepted_for_formal": False,
            "ready_for_higher_quality_ai_allocation_rerun_before_policy": False,
            "policy_required": True,
            "blocked_reason": "TPEx exact all-stock dated membership remains unavailable",
            "diagnostic_only": True,
        }
    )
    rows.append(
        {
            "field": "rolling_new_high_materialization_contract",
            "source_tier": "blocked",
            "tier_rank": tier_order["blocked"],
            "rows_or_routes": int(len(new_high)),
            "accepted_for_diagnostic_before_policy": True,
            "accepted_for_formal": False,
            "ready_for_higher_quality_ai_allocation_rerun_before_policy": False,
            "policy_required": True,
            "blocked_reason": "accepted membership universe and rolling window policy are missing",
            "diagnostic_only": True,
        }
    )
    return pd.DataFrame(rows).sort_values(["tier_rank", "field"])


def _scenario_readiness(readiness: dict[str, Any]) -> pd.DataFrame:
    scenarios = [
        (False, False, "neither_accepted"),
        (True, False, "human_review_ai_membership_accepted_only"),
        (False, True, "twse_industry_proxy_accepted_only"),
        (True, True, "both_accepted"),
    ]
    rows = []
    for human_ok, twse_ok, scenario_id in scenarios:
        higher_quality_ready = bool(human_ok and twse_ok)
        rows.append(
            {
                "scenario_id": scenario_id,
                "human_review_ai_membership_accepted": human_ok,
                "twse_industry_proxy_accepted": twse_ok,
                "ready_for_proxy_limited_ai_allocation_rerun": bool(
                    readiness.get("ready_for_proxy_limited_ai_allocation_rerun", False)
                ),
                "ready_for_higher_quality_ai_allocation_rerun": higher_quality_ready,
                "ready_for_strategy_replay": False,
                "ready_for_formal": False,
                "future_data_violation_count": int(readiness.get("future_data_violation_count", 0) or 0),
                "remaining_blocker": _scenario_blocker(human_ok, twse_ok),
                "diagnostic_only": True,
            }
        )
    return pd.DataFrame(rows)


def _scenario_blocker(human_ok: bool, twse_ok: bool) -> str:
    blockers = []
    if not human_ok:
        blockers.append("human-reviewed AI membership policy not accepted")
    if not twse_ok:
        blockers.append("TWSE official industry proxy comparator policy not accepted")
    if human_ok and twse_ok:
        blockers.append("still not formal/replay; exact TPEx and formal taxonomy remain unavailable")
    return "; ".join(blockers)


def _blocked_proxy_ledger(
    source_quality: pd.DataFrame,
    tpex_routes: pd.DataFrame,
    new_high: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for item in source_quality.itertuples(index=False):
        rows.append(
            {
                "field": item.field,
                "status": item.source_quality,
                "proxy_or_contract_available": bool(item.accepted_for_diagnostic),
                "blocked_reason": item.blocked_reason,
                "accepted_for_formal": bool(item.accepted_for_formal),
                "ready_for_strategy_replay": False,
            }
        )
    for item in tpex_routes.itertuples(index=False):
        rows.append(
            {
                "field": item.route_id,
                "status": "blocked",
                "proxy_or_contract_available": False,
                "blocked_reason": item.blocked_reason_or_notes,
                "accepted_for_formal": bool(item.accepted_for_formal),
                "ready_for_strategy_replay": False,
            }
        )
    for item in new_high.itertuples(index=False):
        rows.append(
            {
                "field": item.component,
                "status": item.source_quality,
                "proxy_or_contract_available": bool(item.current_proxy),
                "blocked_reason": item.blocked_reason,
                "accepted_for_formal": False,
                "ready_for_strategy_replay": False,
            }
        )
    return pd.DataFrame(rows)


def _policy_readiness_json(
    radar_readiness: dict[str, Any],
    scenarios: pd.DataFrame,
    source_tier: pd.DataFrame,
    manifest_in: dict[str, Any],
) -> dict[str, Any]:
    both = scenarios[scenarios["scenario_id"].eq("both_accepted")].iloc[0].to_dict()
    return {
        "date": "2026-07-06",
        "task_id": TASK_ID,
        "owner": "BACKTEST_LAB Core/Data",
        "status": "policy_staging_ready_waiting_strategy_center_user_decision",
        "diagnostic_only": True,
        "input_radar_task_id": radar_readiness.get("task_id") or manifest_in.get("task_id"),
        "ready_for_proxy_limited_ai_allocation_rerun": bool(
            radar_readiness.get("ready_for_proxy_limited_ai_allocation_rerun", False)
        ),
        "ready_for_higher_quality_ai_allocation_rerun": False,
        "ready_for_higher_quality_ai_allocation_rerun_if_both_policy_accepted": bool(
            both["ready_for_higher_quality_ai_allocation_rerun"]
        ),
        "ready_for_strategy_replay": False,
        "ready_for_formal": False,
        "future_data_violation_count": int(radar_readiness.get("future_data_violation_count", 0) or 0),
        "exact_ai_membership_rows": int(radar_readiness.get("exact_ai_membership_rows", 0) or 0),
        "human_review_contract_rows": int(radar_readiness.get("human_review_contract_rows", 0) or 0),
        "higher_quality_dated_ai_membership_rows_from_core": int(
            radar_readiness.get("higher_quality_dated_ai_membership_rows_from_core", 0) or 0
        ),
        "source_tier_counts": source_tier["source_tier"].value_counts(dropna=False).to_dict(),
        "policy_decisions_required": [
            "whether to accept human-reviewed AI membership contract for diagnostic high-quality rerun",
            "whether to accept TWSE official industry proxy as diagnostic non-AI comparator",
            "whether to accept rolling new-high window/universe policy before exact materialization",
        ],
        "blocking_summary": radar_readiness.get("blocking_summary", []),
        "formal_model_changed": False,
        "trade_decision_changed": False,
        "active_in_trade_decision": False,
        "report_changed": False,
        "portfolio_replay_executed": False,
    }


def _summary(readiness: dict[str, Any], scenarios: pd.DataFrame) -> str:
    return "\n".join(
        [
            "# vNext Theme Taxonomy Policy-Staging Readiness",
            "",
            f"Status: {readiness['status']}",
            "",
            "Conclusion: policy staging is ready, but no policy has been accepted automatically.",
            "",
            "Readiness:",
            f"- ready_for_proxy_limited_ai_allocation_rerun={str(readiness['ready_for_proxy_limited_ai_allocation_rerun']).lower()}",
            "- ready_for_higher_quality_ai_allocation_rerun=false",
            "- ready_for_strategy_replay=false",
            "- ready_for_formal=false",
            f"- future_data_violation_count={readiness['future_data_violation_count']}",
            "",
            "Scenario readiness:",
            *[
                "- {scenario_id}: higher_quality={ready_for_higher_quality_ai_allocation_rerun}, blocker={remaining_blocker}".format(
                    **row._asdict()
                )
                for row in scenarios.itertuples(index=False)
            ],
            "",
            "Flags:",
            "- formal_model_changed=false",
            "- trade_decision_changed=false",
            "- active_in_trade_decision=false",
            "- report_changed=false",
            "- portfolio_replay_executed=false",
        ]
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--radar-package-dir", type=Path, default=DEFAULT_RADAR_PACKAGE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    manifest = build_policy_staging_readiness(
        radar_package_dir=args.radar_package_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
