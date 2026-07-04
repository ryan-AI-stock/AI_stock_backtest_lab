from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


TASK_ID = "TASK-BACKTEST-CORE-DYNAMIC-POOL1-TAXONOMY-EVIDENCE-PANEL-001"
DEFAULT_RADAR_OUTPUT = (
    "C:/Users/zergv/Documents/Codex/2026-05-23/ai-stock-rotation-radar-https-docs/outputs/"
    "radar_dynamic_pool1_mops_mainline_evidence_ledger_20260704"
)
DEFAULT_OUTPUT_DIR = "outputs/dynamic_pool1_taxonomy_evidence_panel_20260704"


def run_dynamic_pool1_taxonomy_evidence_panel(
    *,
    radar_output: str | Path = DEFAULT_RADAR_OUTPUT,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    run_log: list[dict[str, str]] = []

    def log(step: str, status: str, detail: str = "") -> None:
        run_log.append(
            {
                "timestamp": pd.Timestamp.now(tz="Asia/Taipei").strftime("%Y-%m-%d %H:%M:%S%z"),
                "step": step,
                "status": status,
                "detail": detail,
            }
        )
        pd.DataFrame(run_log).to_csv(output / "run_log.csv", index=False, encoding="utf-8-sig")
        (output / "current_step.txt").write_text(f"{step}:{status}\n{detail}", encoding="utf-8")

    try:
        radar_root = Path(radar_output)
        log("load_radar_evidence", "started", str(radar_root))
        readiness = _load_json(radar_root / "readiness_for_core.json")
        manifest = _load_json(radar_root / "manifest.json")
        accepted = _read_csv_required(radar_root / "accepted_evidence_rows.csv")
        blocked = _read_csv_required(radar_root / "blocked_or_needs_review.csv")
        future_audit = _read_csv_if_exists(radar_root / "future_data_violation_audit.csv")
        source_audit = _read_csv_if_exists(radar_root / "taxonomy_acceptance_audit.csv")

        log("build_panels", "started", "")
        evidence_panel = _build_evidence_panel(accepted)
        blocker_panel = _build_blocker_panel(blocked)
        by_ticker = _build_by_ticker(evidence_panel, blocker_panel)
        layer_summary = _build_layer_summary(evidence_panel, blocker_panel)
        source_quality = _build_source_quality_audit(source_audit, evidence_panel, blocker_panel)
        future_output = _build_future_audit(future_audit, readiness)
        readiness_output = _build_readiness(readiness, manifest, evidence_panel, blocker_panel)
        manifest_output = _build_manifest(output, radar_root, readiness_output)

        log("write_outputs", "started", str(output))
        evidence_panel.to_csv(output / "taxonomy_evidence_panel.csv", index=False, encoding="utf-8-sig")
        by_ticker.to_csv(output / "taxonomy_evidence_by_ticker.csv", index=False, encoding="utf-8-sig")
        blocker_panel.to_csv(output / "taxonomy_blocker_panel.csv", index=False, encoding="utf-8-sig")
        blocker_panel.to_csv(output / "blocked_or_needs_review_tickers.csv", index=False, encoding="utf-8-sig")
        layer_summary.to_csv(output / "taxonomy_layer_coverage_summary.csv", index=False, encoding="utf-8-sig")
        source_quality.to_csv(output / "source_quality_audit.csv", index=False, encoding="utf-8-sig")
        future_output.to_csv(output / "future_data_violation_audit.csv", index=False, encoding="utf-8-sig")
        (output / "taxonomy_evidence_readiness.json").write_text(
            json.dumps(readiness_output, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (output / "manifest.json").write_text(
            json.dumps(manifest_output, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (output / "final_summary_zh.md").write_text(_summary_zh(readiness_output), encoding="utf-8")
        pd.DataFrame(
            [
                {
                    "step": TASK_ID,
                    "status": "completed_diagnostic_taxonomy_evidence_panel",
                    "output_dir": str(output),
                }
            ]
        ).to_csv(output / "completed.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(columns=["step", "status", "reason"]).to_csv(output / "failed.csv", index=False, encoding="utf-8-sig")
        log("completed", "completed", str(output.resolve()))
        return output
    except Exception as exc:
        pd.DataFrame([{"step": TASK_ID, "status": "failed", "reason": str(exc)}]).to_csv(
            output / "failed.csv", index=False, encoding="utf-8-sig"
        )
        log("failed", "failed", str(exc))
        raise


def _build_evidence_panel(accepted: pd.DataFrame) -> pd.DataFrame:
    frame = accepted.copy().fillna("")
    for column in _evidence_columns():
        if column not in frame.columns:
            frame[column] = ""
    frame["candidate_scope"] = frame.get("candidate_scope", "")
    frame["has_accepted_evidence"] = True
    frame["accepted_evidence_count"] = frame.groupby("ticker")["ticker"].transform("count").astype(int)
    frame["blocked_reason"] = ""
    frame["next_programmatic_source"] = ""
    frame["diagnostic_only"] = True
    frame["ready_for_strategy_replay"] = False
    frame["formal_taxonomy"] = False
    frame["formal_model_changed"] = False
    frame["trade_decision_changed"] = False
    frame["active_in_trade_decision"] = False
    frame["accepted_for_diagnostic"] = frame["accepted_for_diagnostic"].map(_bool_like)
    frame["accepted_for_formal"] = False
    frame["human_review_required"] = True
    frame["formal_exact"] = frame["formal_exact"].map(_bool_like)
    return frame[_evidence_panel_columns()].sort_values(["ticker", "source_date", "source_doc_type"]).reset_index(drop=True)


def _build_blocker_panel(blocked: pd.DataFrame) -> pd.DataFrame:
    frame = blocked.copy().fillna("")
    for column in _blocker_columns():
        if column not in frame.columns:
            frame[column] = ""
    frame["has_accepted_evidence"] = False
    frame["accepted_evidence_count"] = 0
    frame["accepted_for_diagnostic"] = False
    frame["accepted_for_formal"] = False
    frame["human_review_required"] = True
    frame["diagnostic_only"] = True
    frame["ready_for_strategy_replay"] = False
    frame["formal_taxonomy"] = False
    frame["formal_model_changed"] = False
    frame["trade_decision_changed"] = False
    frame["active_in_trade_decision"] = False
    return frame[_blocker_panel_columns()].sort_values(["ticker"]).reset_index(drop=True)


def _build_by_ticker(evidence: pd.DataFrame, blockers: pd.DataFrame) -> pd.DataFrame:
    accepted_rows: list[dict[str, Any]] = []
    if not evidence.empty:
        for ticker, group in evidence.groupby("ticker", dropna=False):
            first = group.iloc[0]
            accepted_rows.append(
                {
                    "ticker": ticker,
                    "company_name": first.get("company_name", ""),
                    "market": first.get("market", ""),
                    "candidate_scope": first.get("candidate_scope", ""),
                    "taxonomy_status": "accepted_diagnostic_evidence_needs_human_review",
                    "has_accepted_evidence": True,
                    "accepted_evidence_count": int(len(group)),
                    "ai_supply_chain_layers": "|".join(sorted(set(group["ai_supply_chain_layer"].astype(str)) - {""})),
                    "mainline_theme_labels": "|".join(sorted(set(group["mainline_theme_label"].astype(str)) - {""})),
                    "confidence_levels": "|".join(sorted(set(group["confidence_level"].astype(str)) - {""})),
                    "accepted_for_diagnostic": True,
                    "accepted_for_formal": False,
                    "human_review_required": True,
                    "blocked_reason": "",
                    "next_programmatic_source": "",
                    "diagnostic_only": True,
                    "ready_for_strategy_replay": False,
                    "formal_taxonomy": False,
                    "formal_model_changed": False,
                    "trade_decision_changed": False,
                    "active_in_trade_decision": False,
                }
            )
    blocked_rows = []
    accepted_tickers = {row["ticker"] for row in accepted_rows}
    for item in blockers.to_dict(orient="records"):
        if item.get("ticker", "") in accepted_tickers:
            continue
        blocked_rows.append(
            {
                "ticker": item.get("ticker", ""),
                "company_name": item.get("company_name", ""),
                "market": item.get("market", ""),
                "candidate_scope": item.get("candidate_scope", ""),
                "taxonomy_status": item.get("status", "needs_review_or_blocked"),
                "has_accepted_evidence": False,
                "accepted_evidence_count": 0,
                "ai_supply_chain_layers": "",
                "mainline_theme_labels": "",
                "confidence_levels": "",
                "accepted_for_diagnostic": False,
                "accepted_for_formal": False,
                "human_review_required": True,
                "blocked_reason": item.get("blocked_reason", ""),
                "next_programmatic_source": item.get("next_programmatic_source", ""),
                "diagnostic_only": True,
                "ready_for_strategy_replay": False,
                "formal_taxonomy": False,
                "formal_model_changed": False,
                "trade_decision_changed": False,
                "active_in_trade_decision": False,
            }
        )
    return pd.DataFrame(accepted_rows + blocked_rows).sort_values(["has_accepted_evidence", "ticker"], ascending=[False, True])


def _build_layer_summary(evidence: pd.DataFrame, blockers: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if not evidence.empty:
        grouped = evidence.groupby(["ai_supply_chain_layer", "mainline_theme_label", "confidence_level"], dropna=False)
        for keys, group in grouped:
            layer, theme, confidence = keys
            rows.append(
                {
                    "ai_supply_chain_layer": layer,
                    "mainline_theme_label": theme,
                    "confidence_level": confidence,
                    "accepted_evidence_rows": int(len(group)),
                    "accepted_unique_tickers": int(group["ticker"].nunique()),
                    "blocked_or_needs_review_tickers": 0,
                    "accepted_for_diagnostic": True,
                    "accepted_for_formal": False,
                    "human_review_required": True,
                    "diagnostic_only": True,
                    "ready_for_strategy_replay": False,
                }
            )
    if not blockers.empty:
        rows.append(
            {
                "ai_supply_chain_layer": "blocked_or_needs_review",
                "mainline_theme_label": "blocked_or_needs_review",
                "confidence_level": "not_available",
                "accepted_evidence_rows": 0,
                "accepted_unique_tickers": 0,
                "blocked_or_needs_review_tickers": int(blockers["ticker"].nunique()),
                "accepted_for_diagnostic": False,
                "accepted_for_formal": False,
                "human_review_required": True,
                "diagnostic_only": True,
                "ready_for_strategy_replay": False,
            }
        )
    return pd.DataFrame(rows)


def _build_source_quality_audit(source_audit: pd.DataFrame, evidence: pd.DataFrame, blockers: pd.DataFrame) -> pd.DataFrame:
    if not source_audit.empty:
        frame = source_audit.copy()
    else:
        frame = pd.DataFrame()
    extra = pd.DataFrame(
        [
            {
                "category": "core_taxonomy_evidence_panel",
                "row_count": int(len(evidence)),
                "accepted_for_diagnostic": True,
                "accepted_for_formal": False,
                "human_review_required": True,
                "decision": "diagnostic_evidence_surface_only_not_strategy_replay",
            },
            {
                "category": "core_taxonomy_blocker_panel",
                "row_count": int(len(blockers)),
                "accepted_for_diagnostic": False,
                "accepted_for_formal": False,
                "human_review_required": True,
                "decision": "bounded_document_extraction_required",
            },
        ]
    )
    return pd.concat([frame, extra], ignore_index=True, sort=False)


def _build_future_audit(future_audit: pd.DataFrame, readiness: dict[str, Any]) -> pd.DataFrame:
    if not future_audit.empty:
        return future_audit
    return pd.DataFrame(
        [
            {
                "data_area": "dynamic_pool1_taxonomy_evidence_panel",
                "future_data_violation": False,
                "future_data_violation_count": int(readiness.get("future_data_violation_count", 0) or 0),
                "audit_reason": "Core only reshapes Radar/Data dated evidence ledger; no current/static/generated taxonomy map is merged.",
            }
        ]
    )


def _build_readiness(
    readiness: dict[str, Any],
    manifest: dict[str, Any],
    evidence: pd.DataFrame,
    blockers: pd.DataFrame,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "task_id": TASK_ID,
        "status": "diagnostic_evidence_surface_ready",
        "source_task_id": readiness.get("task_id", manifest.get("task_id", "")),
        "accepted_evidence_rows": int(len(evidence)),
        "accepted_unique_tickers": int(evidence["ticker"].nunique()) if not evidence.empty else 0,
        "blocked_or_needs_review_tickers": int(blockers["ticker"].nunique()) if not blockers.empty else 0,
        "diagnostic_only": True,
        "ready_for_strategy_replay": False,
        "strategy_replay": False,
        "formal_taxonomy": False,
        "selector_changed": False,
        "accepted_for_formal": False,
        "human_review_required": True,
        "formal_model_changed": False,
        "trade_decision_changed": False,
        "active_in_trade_decision": False,
        "future_data_violation_count": int(readiness.get("future_data_violation_count", 0) or 0),
        "source_boundary": readiness.get("source_boundary", ""),
        "remaining_blockers": readiness.get(
            "remaining_blockers",
            [
                "bounded MOPS document extraction v1 still required for blocked tickers",
                "formal taxonomy policy not approved",
            ],
        ),
    }


def _build_manifest(output: Path, radar_root: Path, readiness: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "task_id": TASK_ID,
        "status": readiness["status"],
        "generated_at": pd.Timestamp.now(tz="Asia/Taipei").isoformat(),
        "output_dir": str(output),
        "radar_output": str(radar_root),
        "accepted_evidence_rows": readiness["accepted_evidence_rows"],
        "accepted_unique_tickers": readiness["accepted_unique_tickers"],
        "blocked_or_needs_review_tickers": readiness["blocked_or_needs_review_tickers"],
        "diagnostic_only": True,
        "ready_for_strategy_replay": False,
        "strategy_replay": False,
        "formal_taxonomy": False,
        "selector_changed": False,
        "accepted_for_formal": False,
        "human_review_required": True,
        "formal_model_changed": False,
        "trade_decision_changed": False,
        "active_in_trade_decision": False,
        "future_data_violation_count": readiness["future_data_violation_count"],
        "outputs": {
            "taxonomy_evidence_panel": "taxonomy_evidence_panel.csv",
            "taxonomy_evidence_by_ticker": "taxonomy_evidence_by_ticker.csv",
            "taxonomy_blocker_panel": "taxonomy_blocker_panel.csv",
            "blocked_or_needs_review_tickers": "blocked_or_needs_review_tickers.csv",
            "taxonomy_layer_coverage_summary": "taxonomy_layer_coverage_summary.csv",
            "taxonomy_evidence_readiness": "taxonomy_evidence_readiness.json",
            "source_quality_audit": "source_quality_audit.csv",
            "future_data_violation_audit": "future_data_violation_audit.csv",
            "final_summary_zh": "final_summary_zh.md",
        },
    }


def _summary_zh(readiness: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Dynamic Pool1 taxonomy evidence panel",
            "",
            f"- 狀態：{readiness['status']}",
            f"- accepted evidence rows：{readiness['accepted_evidence_rows']}",
            f"- accepted unique tickers：{readiness['accepted_unique_tickers']}",
            f"- blocked / needs review tickers：{readiness['blocked_or_needs_review_tickers']}",
            "- 用途：Research diagnostic evidence surface。",
            "- 不可用於 strategy replay，不是 formal taxonomy，不改 selector / target / trade action。",
            "- 所有 accepted evidence 仍需 human review，accepted_for_formal=false。",
        ]
    )


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _read_csv_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path).fillna("")


def _read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path).fillna("")
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _bool_like(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _evidence_columns() -> list[str]:
    return [
        "ticker",
        "company_name",
        "market",
        "candidate_scope",
        "ai_supply_chain_layer",
        "mainline_theme_label",
        "source_doc_type",
        "source_doc_date",
        "source_date",
        "effective_date",
        "as_of_date",
        "confidence_level",
        "accepted_for_diagnostic",
        "accepted_for_formal",
        "formal_exact",
        "human_review_required",
        "blocked_reason",
        "next_programmatic_source",
    ]


def _blocker_columns() -> list[str]:
    return [
        "ticker",
        "company_name",
        "market",
        "candidate_scope",
        "status",
        "blocked_reason",
        "next_programmatic_source",
        "accepted_for_diagnostic",
        "accepted_for_formal",
    ]


def _evidence_panel_columns() -> list[str]:
    return [
        "ticker",
        "company_name",
        "market",
        "candidate_scope",
        "has_accepted_evidence",
        "accepted_evidence_count",
        "ai_supply_chain_layer",
        "mainline_theme_label",
        "source_doc_type",
        "source_doc_date",
        "source_date",
        "effective_date",
        "as_of_date",
        "confidence_level",
        "accepted_for_diagnostic",
        "accepted_for_formal",
        "formal_exact",
        "human_review_required",
        "blocked_reason",
        "next_programmatic_source",
        "diagnostic_only",
        "ready_for_strategy_replay",
        "formal_taxonomy",
        "formal_model_changed",
        "trade_decision_changed",
        "active_in_trade_decision",
    ]


def _blocker_panel_columns() -> list[str]:
    return [
        "ticker",
        "company_name",
        "market",
        "candidate_scope",
        "status",
        "has_accepted_evidence",
        "accepted_evidence_count",
        "accepted_for_diagnostic",
        "accepted_for_formal",
        "human_review_required",
        "blocked_reason",
        "next_programmatic_source",
        "diagnostic_only",
        "ready_for_strategy_replay",
        "formal_taxonomy",
        "formal_model_changed",
        "trade_decision_changed",
        "active_in_trade_decision",
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build Dynamic Pool1 diagnostic taxonomy evidence panel.")
    parser.add_argument("--radar-output", default=DEFAULT_RADAR_OUTPUT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    run_dynamic_pool1_taxonomy_evidence_panel(radar_output=args.radar_output, output_dir=args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
