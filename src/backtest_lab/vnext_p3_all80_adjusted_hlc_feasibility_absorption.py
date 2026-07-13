from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RADAR = Path(r"C:\Users\zergv\Documents\Codex\2026-05-23\ai-stock-rotation-radar-https-docs\outputs\radar_vnext_p3_layer5_all80_continuous_lifecycle_adjusted_hlc_delta_feasibility_20260713")
OUT = ROOT / "outputs/vnext_p3_layer5_all80_continuous_adjusted_hlc_delta_feasibility_absorption_20260713"
TASK = "TASK-BACKTEST-CORE-VNEXT-P3-LAYER5-ALL80-CONTINUOUS-LIFECYCLE-ADJUSTED-HLC-DELTA-FEASIBILITY-ABSORPTION-001"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    source_readiness = json.loads((RADAR / "readiness_for_core_all80_adjusted_hlc_delta_feasibility.json").read_text(encoding="utf-8"))
    summary = pd.read_csv(RADAR / "all80_adjusted_hlc_gap_classification_summary.csv")
    routes = pd.read_csv(RADAR / "all80_adjusted_hlc_unique_ticker_month_routes.csv")

    coverage = source_readiness["coverage"]
    accounted = (
        coverage["locally_reconstructable_rows"]
        + coverage["official_zero_or_not_applicable_proven_rows"]
        + coverage["remaining_rows_requiring_source_or_policy"]
    )
    if accounted != coverage["input_gap_rows"]:
        raise ValueError(f"classification does not reconcile: {accounted} != {coverage['input_gap_rows']}")
    if not source_readiness["bounded_stop_gate_pass"]:
        raise ValueError("bounded source gate did not pass")
    if source_readiness["future_data_violation_count"] != 0:
        raise ValueError("future-data violation present")

    summary.to_csv(OUT / "absorbed_gap_classification_summary.csv", index=False, encoding="utf-8-sig")
    planning = {
        "task_id": TASK,
        "status": "bounded_delta_acquisition_authorized_with_structural_blockers_retained",
        "source_commit": "dfd11e2",
        "input_gap_rows": coverage["input_gap_rows"],
        "local_reuse_rows": coverage["locally_reconstructable_rows"],
        "official_zero_or_not_applicable_rows": coverage["official_zero_or_not_applicable_proven_rows"],
        "bounded_delta_rows": coverage["remaining_rows_requiring_source_or_policy"],
        "bounded_routes": coverage["unique_ticker_month_routes"],
        "raw_routes": coverage["raw_month_routes"],
        "factor_ticker_routes": coverage["trusted_factor_ticker_routes"],
        "route_ledger_rows": len(routes),
        "structural_factor_blocked_rows": int(summary.loc[summary.classification.eq("symbol_or_source_structural_factor_blocked"), "gap_rows"].sum()),
        "download_scope_expansion_prohibited": True,
        "ready_for_bounded_delta_acquisition": True,
        "ready_for_state_supply_rerun": False,
        "ready_for_experiments": False,
        "performance_authorized": False,
        "P3_2_outcome_read_authorized": False,
        "Top3_authorized": False,
        "future_data_violation_count": 0,
        "formal_model_changed": False,
        "trade_decision_changed": False,
        "active_in_trade_decision": False,
        "report_changed": False,
        "not_live_rule": True,
        "forward_returns_live_rule_usage": False,
    }
    (OUT / "bounded_delta_acquisition_planning.json").write_text(json.dumps(planning, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "final_summary_zh.md").write_text(
        "# P3 all80 continuous lifecycle adjusted-HLC feasibility absorption\n\n"
        "149,369列原始缺口已完整對帳：136,491列可本機重用、1,200列為官方zero/not-applicable、"
        "11,678列需bounded來源或政策處理。2,460 routes低於既定stop gate；僅授權該delta scope，"
        "不擴full-market。structural factor blockers保留，完成來源包前不得重跑state supply。\n",
        encoding="utf-8",
    )
    files = sorted(path for path in OUT.iterdir() if path.is_file() and path.name != "manifest.json")
    (OUT / "manifest.json").write_text(json.dumps({"task_id": TASK, "files": [{"name": p.name, "sha256": _sha(p), "bytes": p.stat().st_size} for p in files]}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    run()
