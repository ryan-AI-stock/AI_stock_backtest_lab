"""Build vNext theme taxonomy readiness package.

This is diagnostic/data-readiness only. It does not alter formal model,
reports, trade decisions, or execute any portfolio replay.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


TASK_ID = "TASK-BACKTEST-CORE-VNEXT-THEME-TAXONOMY-SOURCE-INGEST-READINESS-001"
DEFAULT_OUTPUT_DIR = Path("outputs/vnext_theme_taxonomy_source_ingest_readiness_20260706")
DEFAULT_MATERIALIZATION_DIR = Path("outputs/vnext_dynamic_candidate_pool_data_materialization_20260706")
DEFAULT_TAXONOMY_EVIDENCE = Path("outputs/dynamic_pool1_taxonomy_evidence_panel_20260704/taxonomy_evidence_by_ticker.csv")
DEFAULT_TAXONOMY_READINESS = Path("outputs/dynamic_pool1_taxonomy_evidence_panel_20260704/taxonomy_evidence_readiness.json")
DEFAULT_SECTOR_READINESS = Path(
    r"C:\Users\zergv\Documents\Codex\2026-05-23\ai-stock-rotation-radar-https-docs\outputs"
    r"\radar_dynamic_pool1_sector_mainline_pit_full_sweep_and_tpex_reverse_20260703\readiness_for_core.json"
)
DEFAULT_SECTOR_TAXONOMY_READINESS = Path(
    r"C:\Users\zergv\Documents\Codex\2026-05-23\ai-stock-rotation-radar-https-docs\outputs"
    r"\radar_dynamic_pool1_sector_taxonomy_readiness_20260704\readiness_for_core.json"
)
DEFAULT_RADAR_SOURCE_PACKAGE = Path(
    r"C:\Users\zergv\Documents\Codex\2026-05-23\ai-stock-rotation-radar-https-docs\outputs"
    r"\radar_vnext_theme_taxonomy_source_package_20260706"
)


def build_vnext_theme_taxonomy_readiness(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    materialization_dir: str | Path = DEFAULT_MATERIALIZATION_DIR,
    taxonomy_evidence_path: str | Path = DEFAULT_TAXONOMY_EVIDENCE,
    taxonomy_readiness_path: str | Path = DEFAULT_TAXONOMY_READINESS,
    sector_readiness_path: str | Path = DEFAULT_SECTOR_READINESS,
    sector_taxonomy_readiness_path: str | Path = DEFAULT_SECTOR_TAXONOMY_READINESS,
    radar_source_package_dir: str | Path = DEFAULT_RADAR_SOURCE_PACKAGE,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    materialization = Path(materialization_dir)

    membership = pd.read_csv(materialization / "vnext_ai_theme_membership_contract.csv")
    strength = pd.read_csv(materialization / "vnext_ai_theme_strength_snapshot.csv")
    allocation = pd.read_csv(materialization / "vnext_ai_allocation_variant_support.csv")
    taxonomy = pd.read_csv(taxonomy_evidence_path)
    taxonomy_readiness = _read_json(Path(taxonomy_readiness_path))
    sector_readiness = _read_json(Path(sector_readiness_path))
    sector_taxonomy_readiness = _read_json(Path(sector_taxonomy_readiness_path))
    radar_package = _load_radar_source_package(Path(radar_source_package_dir))

    ai_ledger = _ai_membership_source_quality_ledger(taxonomy, membership)
    non_ai_readiness = _non_ai_theme_taxonomy_readiness(sector_readiness, sector_taxonomy_readiness, membership)
    component_readiness = _theme_strength_component_readiness(strength, membership)
    blocked_fields = _blocked_fields(taxonomy_readiness, sector_readiness, sector_taxonomy_readiness)
    if radar_package:
        ai_ledger = _radar_ai_membership_source_quality_ledger(radar_package["ai_membership_upgrade"])
        non_ai_readiness = _radar_non_ai_theme_taxonomy_readiness(
            radar_package["non_ai_taxonomy"],
            fallback=non_ai_readiness,
        )
        component_readiness = _radar_theme_strength_component_readiness(radar_package["component_readiness"])
        blocked_fields = _radar_blocked_fields(radar_package["blocked_fields"])
    rerun_readiness = _rerun_readiness(
        membership=membership,
        ai_ledger=ai_ledger,
        non_ai_readiness=non_ai_readiness,
        component_readiness=component_readiness,
        blocked_fields=blocked_fields,
        allocation=allocation,
        radar_package=radar_package,
    )

    _write_csv(ai_ledger, output / "theme_membership_source_quality_ledger.csv")
    _write_csv(non_ai_readiness, output / "non_ai_theme_taxonomy_readiness.csv")
    _write_csv(component_readiness, output / "theme_strength_component_readiness.csv")
    _write_csv(blocked_fields, output / "blocked_fields_and_proxy_fields.csv")
    _write_csv(_ai_membership_contract_readiness(membership), output / "dated_ai_membership_readiness.csv")
    if radar_package:
        _write_csv(ai_ledger, output / "dated_ai_membership_source_ingest_contract.csv")
        _write_csv(non_ai_readiness, output / "non_ai_theme_comparator_contract.csv")
        _write_csv(component_readiness, output / "theme_strength_score_component_readiness_contract.csv")
        _write_csv(radar_package["ai_subtheme_evidence"], output / "ai_subtheme_classification_evidence_ledger.csv")
        _write_csv(radar_package["ai_subtheme_evidence"], output / "ai_subtheme_evidence_contract.csv")
        _write_csv(radar_package["source_attempt_evidence"], output / "source_attempt_evidence.csv")
        (output / "radar_source_package_readiness.json").write_text(
            json.dumps(radar_package["readiness"], ensure_ascii=False, indent=2), encoding="utf-8"
        )
    (output / "readiness_for_ai_allocation_rerun.json").write_text(
        json.dumps(rerun_readiness, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    manifest = {
        "task_id": TASK_ID,
        "status": rerun_readiness["status"],
        "output_dir": str(output.resolve()),
        "output_files": [
            "theme_membership_source_quality_ledger.csv",
            "non_ai_theme_taxonomy_readiness.csv",
            "theme_strength_component_readiness.csv",
            "blocked_fields_and_proxy_fields.csv",
            "dated_ai_membership_readiness.csv",
            *(
                [
                    "dated_ai_membership_source_ingest_contract.csv",
                    "ai_subtheme_evidence_contract.csv",
                    "non_ai_theme_comparator_contract.csv",
                    "theme_strength_score_component_readiness_contract.csv",
                    "ai_subtheme_classification_evidence_ledger.csv",
                    "source_attempt_evidence.csv",
                    "radar_source_package_readiness.json",
                ]
                if radar_package
                else []
            ),
            "readiness_for_ai_allocation_rerun.json",
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
    (output / "final_summary_zh.md").write_text(_summary(rerun_readiness, blocked_fields), encoding="utf-8")
    return manifest


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _load_radar_source_package(path: Path) -> dict[str, Any]:
    readiness_path = path / "readiness_for_core_research.json"
    if not readiness_path.exists():
        return {}
    return {
        "source_dir": str(path),
        "readiness": _read_json(readiness_path),
        "ai_membership_upgrade": pd.read_csv(path / "ai_membership_source_package_upgrade.csv"),
        "ai_subtheme_evidence": pd.read_csv(path / "ai_subtheme_classification_evidence_ledger.csv"),
        "non_ai_taxonomy": pd.read_csv(path / "non_ai_theme_taxonomy_candidate_ledger.csv"),
        "component_readiness": pd.read_csv(path / "theme_strength_score_component_readiness_ledger.csv"),
        "blocked_fields": pd.read_csv(path / "blocked_fields_and_proxy_fields.csv"),
        "source_attempt_evidence": pd.read_csv(path / "source_attempt_evidence.csv"),
    }


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def _norm_ticker(value: object) -> str:
    text = "" if value is None else str(value).strip()
    if text.endswith(".TW") or text.endswith(".TWO"):
        text = text.split(".", 1)[0]
    if text.endswith(".0"):
        text = text[:-2]
    return text


def _ai_membership_source_quality_ledger(taxonomy: pd.DataFrame, membership: pd.DataFrame) -> pd.DataFrame:
    taxonomy = taxonomy.copy()
    taxonomy["ticker_base"] = taxonomy["ticker"].map(_norm_ticker)
    ai_members = membership[membership["is_ai_theme_member"].astype(bool)].copy()
    ai_members["ticker"] = ai_members["ticker"].map(_norm_ticker)
    unique_ai = ai_members[["ticker", "theme_id", "theme_name", "ai_subtheme"]].drop_duplicates()
    ledger = unique_ai.merge(
        taxonomy,
        left_on="ticker",
        right_on="ticker_base",
        how="left",
        suffixes=("", "_evidence"),
    )
    ledger["ai_membership_source_quality"] = "proxy"
    ledger["exact_membership_available"] = False
    ledger["higher_quality_dated_membership_available"] = ledger["has_accepted_evidence"].fillna(False).astype(bool)
    ledger["source_quality_reason"] = (
        "diagnostic taxonomy evidence exists but accepted_for_formal=false and human_review_required=true"
    )
    ledger["diagnostic_only"] = True
    ledger["formal_model_changed"] = False
    ledger["trade_decision_changed"] = False
    ledger["active_in_trade_decision"] = False
    ledger["report_changed"] = False
    cols = [
        "ticker",
        "theme_id",
        "theme_name",
        "ai_subtheme",
        "ai_membership_source_quality",
        "exact_membership_available",
        "higher_quality_dated_membership_available",
        "source_package_task_ids",
        "evidence_versions",
        "confidence_levels",
        "accepted_for_diagnostic",
        "accepted_for_formal",
        "human_review_required",
        "source_quality_reason",
        "diagnostic_only",
        "formal_model_changed",
        "trade_decision_changed",
        "active_in_trade_decision",
        "report_changed",
    ]
    return ledger.reindex(columns=cols).sort_values(["ticker", "theme_id"])


def _radar_ai_membership_source_quality_ledger(radar_membership: pd.DataFrame) -> pd.DataFrame:
    ledger = radar_membership.copy()
    ledger["ticker"] = ledger["ticker"].map(_norm_ticker)
    ledger["ai_membership_source_quality"] = ledger["source_quality"]
    ledger["source_package_task_ids"] = ledger["source_package_task_id"]
    ledger["evidence_versions"] = ledger["source_doc_type"]
    ledger["confidence_levels"] = ledger["ai_membership_confidence"] = ledger.get(
        "ai_membership_confidence", "source_package_diagnostic"
    )
    ledger["source_quality_reason"] = ledger["blocked_reason"]
    ledger["diagnostic_only"] = True
    ledger["formal_model_changed"] = False
    ledger["trade_decision_changed"] = False
    ledger["active_in_trade_decision"] = False
    ledger["report_changed"] = False
    cols = [
        "ticker",
        "theme_id",
        "theme_name",
        "ai_subtheme",
        "ai_membership_source_quality",
        "exact_membership_available",
        "higher_quality_dated_membership_available",
        "source",
        "source_doc_type",
        "source_date",
        "effective_date",
        "source_package_task_ids",
        "evidence_versions",
        "confidence_levels",
        "accepted_for_diagnostic",
        "accepted_for_formal",
        "human_review_required",
        "source_quality_reason",
        "diagnostic_only",
        "formal_model_changed",
        "trade_decision_changed",
        "active_in_trade_decision",
        "report_changed",
    ]
    return ledger.reindex(columns=cols).sort_values(["ticker", "theme_id"])


def _ai_membership_contract_readiness(membership: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        membership.groupby(["ticker", "is_ai_theme_member", "ai_membership_source_quality", "ai_subtheme"], dropna=False)
        .agg(
            first_snapshot_date=("snapshot_date", "min"),
            last_snapshot_date=("snapshot_date", "max"),
            row_count=("snapshot_date", "count"),
        )
        .reset_index()
    )
    grouped["diagnostic_only"] = True
    grouped["exact_membership_available"] = False
    grouped["ready_for_formal"] = False
    grouped["blocked_reason"] = grouped["is_ai_theme_member"].map(
        {True: "AI membership is proxy-heavy and not human-reviewed for formal use", False: "non-AI membership is unclassified proxy"}
    )
    return grouped


def _non_ai_theme_taxonomy_readiness(
    sector_readiness: dict[str, Any],
    sector_taxonomy_readiness: dict[str, Any],
    membership: pd.DataFrame,
) -> pd.DataFrame:
    non_ai_rows = int((~membership["is_ai_theme_member"].astype(bool)).sum())
    return pd.DataFrame(
        [
            {
                "taxonomy_layer": "twse_official_sector_monthly_anchor",
                "status": "partial_diagnostic_ready",
                "source_quality": "proxy",
                "rows_or_members": sector_readiness.get("twse_sector_membership_rows"),
                "coverage": "TWSE only, 2015-2026 monthly anchor",
                "blocked_reason": "monthly anchor is not full daily exact membership",
                "ready_for_ai_allocation_rerun": True,
                "ready_for_formal": False,
            },
            {
                "taxonomy_layer": "tpex_sector_membership",
                "status": "blocked",
                "source_quality": "missing",
                "rows_or_members": sector_taxonomy_readiness.get("accepted_tpex_sector_rows", 0),
                "coverage": "TPEx all-stock historical membership unavailable",
                "blocked_reason": "TPEx all-stock historical sector route remains locked",
                "ready_for_ai_allocation_rerun": False,
                "ready_for_formal": False,
            },
            {
                "taxonomy_layer": "non_ai_theme_taxonomy",
                "status": "proxy_only",
                "source_quality": "unknown",
                "rows_or_members": non_ai_rows,
                "coverage": "current vNext package uses non_ai_unclassified_proxy",
                "blocked_reason": "official industry to market-mainline/theme mapping not materialized",
                "ready_for_ai_allocation_rerun": False,
                "ready_for_formal": False,
            },
        ]
    )


def _radar_non_ai_theme_taxonomy_readiness(
    radar_non_ai: pd.DataFrame,
    *,
    fallback: pd.DataFrame,
) -> pd.DataFrame:
    if radar_non_ai.empty:
        return fallback
    rows = []
    for item in radar_non_ai.itertuples(index=False):
        rows.append(
            {
                "taxonomy_layer": item.theme_id,
                "status": "partial_diagnostic_candidate" if bool(item.accepted_for_diagnostic) else "blocked",
                "source_quality": item.source_quality,
                "rows_or_members": None,
                "coverage": item.coverage,
                "blocked_reason": item.blocked_reason,
                "ready_for_ai_allocation_rerun": bool(item.accepted_for_diagnostic),
                "ready_for_formal": bool(item.accepted_for_formal),
                "human_review_required": bool(item.human_review_required),
            }
        )
    return pd.DataFrame(rows)


def _theme_strength_component_readiness(strength: pd.DataFrame, membership: pd.DataFrame) -> pd.DataFrame:
    components = [
        ("ai_theme_excess_vs_0050", "ready", "trailing non-forward RS/excess proxy from weekly snapshot"),
        ("ai_theme_excess_vs_00631L", "ready", "trailing non-forward excess proxy from weekly snapshot"),
        ("ai_breadth_vs_0050", "ready", "trailing non-forward breadth proxy"),
        ("ai_breadth_vs_00631L", "ready", "trailing non-forward breadth proxy"),
        ("ai_turnover_concentration", "ready", "from turnover_state diagnostic fields"),
        ("ai_new_high_count", "proxy", "uses drawdown_60d >= -2%, not exact rolling new-high count"),
        ("ai_long_strong_count", "ready", "from subpool_class"),
        ("ai_pullback_repair_count", "ready", "from subpool_class"),
        ("ai_drawdown_resilience", "ready", "from drawdown_60d"),
        ("best_non_ai_theme_id", "proxy", "uses non_ai_unclassified_proxy, not real non-AI theme taxonomy"),
        ("best_non_ai_theme_score", "proxy", "computed from non_ai_unclassified_proxy aggregate"),
        ("ai_vs_best_non_ai_theme_spread", "proxy", "depends on proxy non-AI comparator"),
        ("ai_theme_state", "proxy_limited", "state computed from proxy-heavy membership and comparator"),
    ]
    rows = []
    for field, status, reason in components:
        rows.append(
            {
                "component": field,
                "status": status,
                "missing_count": int(strength[field].isna().sum()) if field in strength else None,
                "source_quality": "proxy" if "proxy" in status or "proxy" in reason else "diagnostic",
                "reason": reason,
                "diagnostic_only": True,
            }
        )
    return pd.DataFrame(rows)


def _radar_theme_strength_component_readiness(radar_components: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for item in radar_components.itertuples(index=False):
        rows.append(
            {
                "component": item.component,
                "status": item.readiness_status,
                "missing_count": None,
                "source_quality": item.source_quality,
                "reason": item.blocked_reason,
                "exact_available": bool(item.exact_available),
                "proxy_available": bool(item.proxy_available),
                "accepted_for_diagnostic": bool(item.accepted_for_diagnostic),
                "accepted_for_formal": bool(item.accepted_for_formal),
                "human_review_required": bool(item.human_review_required),
                "diagnostic_only": True,
            }
        )
    return pd.DataFrame(rows)


def _blocked_fields(
    taxonomy_readiness: dict[str, Any],
    sector_readiness: dict[str, Any],
    sector_taxonomy_readiness: dict[str, Any],
) -> pd.DataFrame:
    rows = [
        {
            "field_or_contract": "exact_dated_ai_membership",
            "status": "blocked",
            "proxy_available": True,
            "blocked_reason": "AI taxonomy evidence has accepted diagnostic rows but accepted_for_formal=false and human_review_required=true",
            "next_programmatic_source": "human-reviewed MOPS/filing evidence ledger with effective dates and source quality",
        },
        {
            "field_or_contract": "non_ai_theme_taxonomy",
            "status": "blocked",
            "proxy_available": True,
            "blocked_reason": "current package uses non_ai_unclassified_proxy; official industry-to-theme mapping is not materialized",
            "next_programmatic_source": "build dated non-AI theme taxonomy from TWSE sector monthly anchors plus accepted theme mapping",
        },
        {
            "field_or_contract": "tpex_sector_membership",
            "status": "blocked",
            "proxy_available": False,
            "blocked_reason": "TPEx all-stock historical sector membership route remains locked",
            "next_programmatic_source": "reverse TPEx IC/statistics endpoints with date/as-of parameter",
        },
        {
            "field_or_contract": "ai_new_high_count",
            "status": "proxy",
            "proxy_available": True,
            "blocked_reason": "current count uses drawdown_60d >= -2%, not exact rolling new-high count",
            "next_programmatic_source": "compute exact rolling 60/120d highs from stock_features for all AI members",
        },
    ]
    for blocker in taxonomy_readiness.get("remaining_blockers", []):
        rows.append(
            {
                "field_or_contract": "taxonomy_evidence_panel",
                "status": "blocked",
                "proxy_available": True,
                "blocked_reason": blocker,
                "next_programmatic_source": "taxonomy evidence human review / formal taxonomy policy",
            }
        )
    for blocker in sector_readiness.get("remaining_blockers", []):
        rows.append(
            {
                "field_or_contract": "sector_mainline_pit",
                "status": "blocked",
                "proxy_available": True,
                "blocked_reason": blocker,
                "next_programmatic_source": "Radar/Data sector source expansion",
            }
        )
    for blocker in sector_taxonomy_readiness.get("remaining_blockers", []):
        rows.append(
            {
                "field_or_contract": "sector_taxonomy_readiness",
                "status": "blocked",
                "proxy_available": False,
                "blocked_reason": blocker,
                "next_programmatic_source": "Radar/Data taxonomy route unlock",
            }
        )
    return pd.DataFrame(rows)


def _radar_blocked_fields(radar_blocked: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for item in radar_blocked.itertuples(index=False):
        rows.append(
            {
                "field_or_contract": item.field,
                "status": "blocked" if not bool(item.accepted_for_formal) else "ready",
                "proxy_available": bool(str(item.proxy_field_or_source).strip()),
                "blocked_reason": item.blocked_reason,
                "next_programmatic_source": item.proxy_field_or_source,
                "future_data_violation_count": int(item.future_data_violation_count),
                "accepted_for_formal": bool(item.accepted_for_formal),
                "ready_for_strategy_replay": bool(item.ready_for_strategy_replay),
            }
        )
    return pd.DataFrame(rows)


def _rerun_readiness(
    *,
    membership: pd.DataFrame,
    ai_ledger: pd.DataFrame,
    non_ai_readiness: pd.DataFrame,
    component_readiness: pd.DataFrame,
    blocked_fields: pd.DataFrame,
    allocation: pd.DataFrame,
    radar_package: dict[str, Any] | None = None,
) -> dict[str, Any]:
    proxy_heavy = bool((membership["ai_membership_source_quality"].eq("proxy").sum() > 0))
    non_ai_blocked = bool(non_ai_readiness["status"].isin(["blocked", "proxy_only"]).any())
    exact_ai_rows = int(ai_ledger["exact_membership_available"].fillna(False).sum()) if not ai_ledger.empty else 0
    return {
        "date": "2026-07-06",
        "task_id": TASK_ID,
        "owner": "BACKTEST_LAB Core/Data",
        "status": "blocked_for_high_quality_rerun_proxy_package_available",
        "ready_for_proxy_limited_ai_allocation_rerun": True,
        "ready_for_higher_quality_ai_allocation_rerun": False,
        "ready_for_strategy_replay": False,
        "ready_for_formal": False,
        "future_data_violation_count": int(
            (radar_package or {}).get("readiness", {}).get("future_data_violation_count", 0) or 0
        ),
        "diagnostic_only": True,
        "formal_model_changed": False,
        "trade_decision_changed": False,
        "active_in_trade_decision": False,
        "report_changed": False,
        "portfolio_replay_executed": False,
        "membership_rows": int(len(membership)),
        "ai_member_rows": int(membership["is_ai_theme_member"].astype(bool).sum()),
        "exact_ai_membership_rows": exact_ai_rows,
        "higher_quality_dated_ai_membership_rows": int(
            ai_ledger["higher_quality_dated_membership_available"].fillna(False).sum()
        )
        if "higher_quality_dated_membership_available" in ai_ledger
        else 0,
        "ai_membership_source_quality_counts": membership["ai_membership_source_quality"].value_counts(dropna=False).to_dict(),
        "source_package_ingested": bool(radar_package),
        "source_package_status": (radar_package or {}).get("readiness", {}).get("status"),
        "source_package_future_data_violation_count": (radar_package or {}).get("readiness", {}).get(
            "future_data_violation_count"
        ),
        "allocation_variant_rows": int(len(allocation)),
        "component_readiness_counts": component_readiness["status"].value_counts(dropna=False).to_dict(),
        "blocked_field_count": int(len(blocked_fields)),
        "blocking_summary": [
            "AI membership remains proxy-heavy and not formal-reviewed.",
            "Non-AI taxonomy remains mostly non_ai_unclassified_proxy.",
            "TPEx historical sector membership is still blocked.",
            "TWSE sector source is monthly-anchor diagnostic, not daily exact.",
        ],
        "next_owner": "Radar/Data or Core/Data source acquisition",
        "next_step": "Improve exact/higher-quality dated AI membership and non-AI theme taxonomy before rerunning AI allocation diagnostic for a stronger conclusion.",
    }


def _summary(readiness: dict[str, Any], blocked_fields: pd.DataFrame) -> str:
    return "\n".join(
        [
            "# vNext Theme Taxonomy Readiness",
            "",
            f"Status: {readiness['status']}",
            "",
            "Conclusion: proxy-limited package remains available, but high-quality AI allocation rerun is blocked.",
            "",
            "Flags:",
            "- formal_model_changed=false",
            "- trade_decision_changed=false",
            "- active_in_trade_decision=false",
            "- report_changed=false",
            "- portfolio_replay_executed=false",
            "",
            "Top blockers:",
            *[f"- {row.blocked_reason}" for row in blocked_fields.head(8).itertuples()],
        ]
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    manifest = build_vnext_theme_taxonomy_readiness(output_dir=args.output_dir)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
