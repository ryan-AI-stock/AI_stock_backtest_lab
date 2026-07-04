"""Build benchmark-aware Dynamic Pool1 candidate contract.

This is a candidate-pool contract only. It does not run portfolio replay and
does not alter the formal selector, target, report, or trade action.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


TASK_ID = "TASK-BACKTEST-CORE-DYNAMIC-POOL1-BENCHMARK-AWARE-CANDIDATE-CONTRACT-001"
DEFAULT_REPAIR_PANEL = Path("outputs/dynamic_pool1_benchmark_join_repair_contract_20260704/benchmark_join_repair_panel.csv")
DEFAULT_OUTPUT_DIR = Path("outputs/dynamic_pool1_benchmark_aware_candidate_contract_20260704")


def run_benchmark_aware_candidate_contract(
    *,
    repo_root: str | Path = ".",
    repair_panel: str | Path = DEFAULT_REPAIR_PANEL,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict:
    root = Path(repo_root).resolve()
    panel_path = _resolve(root, repair_panel)
    output = _resolve(root, output_dir)
    output.mkdir(parents=True, exist_ok=True)

    panel = pd.read_csv(panel_path)
    contract = _build_contract(panel)
    month_summary = _summary_by(contract, ["candidate_month"])
    regime_summary = _summary_by(contract, ["regime_label"])
    blocked = contract[contract["benchmark_blocked_reason"].fillna("").astype(str) != ""].copy()
    future_audit = pd.DataFrame(
        [
            {
                "audit_item": "benchmark_filter_contract",
                "future_data_violation_count": 0,
                "status": "uses trailing benchmark-relative fields only; no forward return as contract rule",
            },
            {
                "audit_item": "cross_section_median",
                "future_data_violation_count": 0,
                "status": "uses_cross_section_median_as_primary_benchmark=false",
            },
        ]
    )

    contract.to_csv(output / "dynamic_pool1_benchmark_aware_candidate_contract.csv", index=False, encoding="utf-8-sig")
    month_summary.to_csv(output / "benchmark_filter_summary_by_month.csv", index=False, encoding="utf-8-sig")
    regime_summary.to_csv(output / "benchmark_filter_summary_by_regime.csv", index=False, encoding="utf-8-sig")
    blocked.to_csv(output / "blocked_rows.csv", index=False, encoding="utf-8-sig")
    future_audit.to_csv(output / "future_data_audit.csv", index=False, encoding="utf-8-sig")

    manifest = {
        "task_id": TASK_ID,
        "status": "completed_benchmark_aware_candidate_contract",
        "output_dir": str(output),
        "source_repair_panel": str(panel_path),
        "candidate_rows": int(len(contract)),
        "benchmark_ready_rows": int((contract["benchmark_0050_ready_flag"] & contract["benchmark_00631l_ready_flag"]).sum()),
        "primary_selected_rows": int(contract["benchmark_filter_primary_selected"].sum()),
        "sensitivity_selected_rows": int(contract["benchmark_filter_sensitivity_selected"].sum()),
        "blocked_rows": int(len(blocked)),
        "primary_filter": "rs60_positive_vs_both",
        "sensitivity_filters": "rs20_and_rs60_positive_vs_both;top10_and_rs60_positive_vs_both",
        "uses_cross_section_median_as_primary_benchmark": False,
        "forward_return_used_as_contract_rule": False,
        "portfolio_replay_executed": False,
        "formal_model_changed": False,
        "trade_decision_changed": False,
        "active_in_trade_decision": False,
        "report_changed": False,
        "future_data_violation_count": 0,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "final_summary_zh.md").write_text(_summary_text(manifest), encoding="utf-8")
    pd.DataFrame([{"task_id": TASK_ID, "status": "completed", "output_dir": str(output)}]).to_csv(
        output / "completed.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(columns=["task_id", "status", "reason"]).to_csv(output / "failed.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(
        [
            {"step": "load_repaired_panel", "status": "completed"},
            {"step": "build_benchmark_filters", "status": "completed"},
            {"step": "write_outputs", "status": "completed"},
        ]
    ).to_csv(output / "run_log.csv", index=False, encoding="utf-8-sig")
    return manifest


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _build_contract(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    out["benchmark_blocked_reason"] = out.get("benchmark_blocked_reason", "").fillna("")
    for col in [
        "benchmark_0050_ready_flag",
        "benchmark_00631l_ready_flag",
        "price_ready_flag",
        "uses_cross_section_median_as_primary_benchmark",
    ]:
        out[col] = out[col].astype(str).str.lower().eq("true")
    out["candidate_rank"] = pd.to_numeric(out["candidate_rank"], errors="coerce")
    for col in [
        "ret_60d_vs_0050_trailing",
        "ret_60d_vs_00631L_trailing",
        "ret_20d_vs_0050_trailing",
        "ret_20d_vs_00631L_trailing",
    ]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    ready = out["price_ready_flag"] & out["benchmark_0050_ready_flag"] & out["benchmark_00631l_ready_flag"]
    out["rs60_positive_vs_both"] = ready & (out["ret_60d_vs_0050_trailing"] > 0) & (out["ret_60d_vs_00631L_trailing"] > 0)
    out["rs20_and_rs60_positive_vs_both"] = (
        out["rs60_positive_vs_both"]
        & (out["ret_20d_vs_0050_trailing"] > 0)
        & (out["ret_20d_vs_00631L_trailing"] > 0)
    )
    out["top10_and_rs60_positive_vs_both"] = out["rs60_positive_vs_both"] & (out["candidate_rank"] <= 10)
    out["benchmark_filter_primary_selected"] = out["rs60_positive_vs_both"]
    out["benchmark_filter_sensitivity_selected"] = out["rs20_and_rs60_positive_vs_both"] | out["top10_and_rs60_positive_vs_both"]
    out["regime_label"] = out["candidate_month"].map(_regime_label)
    out["forward_return_used_as_contract_rule"] = False
    out["portfolio_replay_executed"] = False
    out["formal_model_changed"] = False
    out["trade_decision_changed"] = False
    out["active_in_trade_decision"] = False
    out["report_changed"] = False
    fields = [
        "candidate_month",
        "candidate_as_of_date",
        "ticker",
        "candidate_rank",
        "candidate_score",
        "candidate_layer",
        "price_ready_flag",
        "benchmark_0050_ready_flag",
        "benchmark_00631l_ready_flag",
        "ret_60d_vs_0050_trailing",
        "ret_60d_vs_00631L_trailing",
        "ret_20d_vs_0050_trailing",
        "ret_20d_vs_00631L_trailing",
        "rs60_positive_vs_both",
        "rs20_and_rs60_positive_vs_both",
        "top10_and_rs60_positive_vs_both",
        "benchmark_filter_primary_selected",
        "benchmark_filter_sensitivity_selected",
        "benchmark_blocked_reason",
        "uses_cross_section_median_as_primary_benchmark",
        "regime_label",
        "forward_return_used_as_contract_rule",
        "portfolio_replay_executed",
        "formal_model_changed",
        "trade_decision_changed",
        "active_in_trade_decision",
        "report_changed",
    ]
    return out[fields]


def _regime_label(month: object) -> str:
    text = str(month)
    if text >= "2026-01":
        return "2026YTD"
    if text >= "2024-01":
        return "2024_latest"
    if text >= "2022-01":
        return "2022_2023"
    if text >= "2015-01":
        return "2015_2021"
    return "out_of_scope"


def _summary_by(contract: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    grouped = contract.groupby(keys, as_index=False).agg(
        rows=("ticker", "count"),
        benchmark_ready_rows=("benchmark_0050_ready_flag", lambda s: int((s & contract.loc[s.index, "benchmark_00631l_ready_flag"]).sum())),
        primary_selected_rows=("benchmark_filter_primary_selected", "sum"),
        rs20_and_rs60_rows=("rs20_and_rs60_positive_vs_both", "sum"),
        top10_and_rs60_rows=("top10_and_rs60_positive_vs_both", "sum"),
        blocked_rows=("benchmark_blocked_reason", lambda s: int((s.fillna("").astype(str) != "").sum())),
        unique_tickers=("ticker", "nunique"),
    )
    grouped["primary_selected_rate"] = grouped["primary_selected_rows"] / grouped["rows"].replace(0, pd.NA)
    grouped["benchmark_ready_rate"] = grouped["benchmark_ready_rows"] / grouped["rows"].replace(0, pd.NA)
    grouped["uses_cross_section_median_as_primary_benchmark"] = False
    return grouped


def _summary_text(manifest: dict) -> str:
    return "\n".join(
        [
            "# Dynamic Pool1 benchmark-aware candidate contract",
            "",
            "本包建立 benchmark-aware candidate contract，不跑 portfolio、不改 formal、不改 report。",
            "",
            f"- candidate rows：{manifest['candidate_rows']}",
            f"- benchmark ready rows：{manifest['benchmark_ready_rows']}",
            f"- primary filter：{manifest['primary_filter']}",
            f"- primary selected rows：{manifest['primary_selected_rows']}",
            f"- sensitivity selected rows：{manifest['sensitivity_selected_rows']}",
            f"- blocked rows：{manifest['blocked_rows']}",
            "- uses_cross_section_median_as_primary_benchmark=false。",
            "- forward_return_used_as_contract_rule=false。",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--repair-panel", default=str(DEFAULT_REPAIR_PANEL))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args(argv)
    manifest = run_benchmark_aware_candidate_contract(
        repo_root=args.repo_root,
        repair_panel=args.repair_panel,
        output_dir=args.output_dir,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
