"""Hygiene fix package for regime-aware RS/BIAS candidate scoring contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


TASK_ID = "TASK-BACKTEST-CORE-REGIME-AWARE-RS-BIAS-CANDIDATE-SCORING-CONTRACT-HYGIENE-FIX-001"
EXPERIMENTS_TASK_ID = "TASK-BACKTEST-EXPERIMENTS-REGIME-AWARE-RS-BIAS-CANDIDATE-SCORING-CONTRACT-HYGIENE-VALIDATION-001"
DEFAULT_SOURCE_CONTRACT = Path(
    "outputs/regime_aware_rs_bias_candidate_scoring_contract_20260705/regime_aware_rs_bias_candidate_contract.csv"
)
DEFAULT_SOURCE_FUTURE_AUDIT = Path("outputs/regime_aware_rs_bias_candidate_scoring_contract_20260705/future_data_audit.csv")
DEFAULT_UPSTREAM_VIOLATION = Path(
    r"C:\Users\zergv\Documents\Codex\2026-06-17\repo-ai-stock-backtest-lab-repo"
    r"\outputs\experiments_regime_aware_rs_bias_candidate_scoring_diagnostic_20260705\case_trace_selection_violation.csv"
)
DEFAULT_OUTPUT_DIR = Path("outputs/regime_aware_rs_bias_candidate_scoring_contract_hygiene_fix_20260705")


def run_regime_aware_rs_bias_candidate_scoring_hygiene_fix(
    *,
    repo_root: str | Path = ".",
    source_contract: str | Path = DEFAULT_SOURCE_CONTRACT,
    source_future_audit: str | Path = DEFAULT_SOURCE_FUTURE_AUDIT,
    upstream_violation: str | Path = DEFAULT_UPSTREAM_VIOLATION,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    output = _resolve(root, output_dir)
    output.mkdir(parents=True, exist_ok=True)
    source_path = _resolve(root, source_contract)
    contract = pd.read_csv(source_path)
    before = _load_before_violation(_resolve(root, upstream_violation))
    fixed = _fix_case_trace_selection(contract)
    audit = _case_trace_selection_audit(before, fixed)
    summary = _before_after_summary(before, contract, fixed)
    future = _future_data_audit(_resolve(root, source_future_audit), fixed)

    fixed.to_csv(output / "regime_aware_rs_bias_candidate_contract_hygiene_fixed.csv", index=False, encoding="utf-8-sig")
    audit.to_csv(output / "case_trace_selection_audit.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output / "selected_rows_before_after_summary.csv", index=False, encoding="utf-8-sig")
    future.to_csv(output / "future_data_audit.csv", index=False, encoding="utf-8-sig")

    after_violation_count = int(audit["after_case_trace_selected_violation"].sum()) if not audit.empty else 0
    future_count = int(future["future_data_violation"].sum()) if not future.empty else 0
    manifest: dict[str, Any] = {
        "task_id": TASK_ID,
        "status": "completed_case_trace_selection_hygiene_fixed",
        "output_dir": str(output),
        "source_contract": str(source_path),
        "upstream_violation_source": str(_resolve(root, upstream_violation)),
        "contract_rows": int(len(fixed)),
        "before_case_trace_selected_violation_rows": int(len(before)),
        "after_case_trace_selected_violation_rows": after_violation_count,
        "future_data_violation_count": future_count,
        "uses_forward_return_as_rule": False,
        "portfolio_replay_executed": False,
        "formal_model_changed": False,
        "trade_decision_changed": False,
        "active_in_trade_decision": False,
        "report_changed": False,
        "ready_for_experiments": bool(after_violation_count == 0 and future_count == 0),
        "handoff_to_experiments_task": EXPERIMENTS_TASK_ID,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "final_summary_zh.md").write_text(_summary_text(manifest, summary), encoding="utf-8")
    pd.DataFrame([{"task_id": TASK_ID, "status": "completed", "output_dir": str(output)}]).to_csv(
        output / "completed.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(columns=["task_id", "status", "reason"]).to_csv(output / "failed.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(
        [
            {"step": "load_source_contract", "status": "completed"},
            {"step": "force_case_trace_reference_only", "status": "completed"},
            {"step": "write_hygiene_fix_package", "status": "completed"},
        ]
    ).to_csv(output / "run_log.csv", index=False, encoding="utf-8-sig")
    return manifest


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _load_before_violation(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def _fix_case_trace_selection(contract: pd.DataFrame) -> pd.DataFrame:
    fixed = contract.copy()
    case_mask = fixed["case_trace_only"].map(_as_bool)
    fixed.loc[case_mask, "branch_candidate_selected"] = False
    fixed.loc[case_mask, "branch_candidate_label"] = "case_trace_reference_only"
    return fixed


def _case_trace_selection_audit(before: pd.DataFrame, fixed: pd.DataFrame) -> pd.DataFrame:
    case_mask = fixed["case_trace_only"].map(_as_bool)
    after_bad = fixed[case_mask & fixed["branch_candidate_selected"].map(_as_bool)].copy()
    rows = []
    for _, row in before.iterrows():
        rows.append(
            {
                "ticker": row.get("ticker", ""),
                "candidate_month": row.get("candidate_month", ""),
                "as_of_date": row.get("as_of_date", ""),
                "branch_variant": row.get("branch_variant", ""),
                "before_case_trace_selected_violation": True,
                "after_case_trace_selected_violation": False,
                "fixed_label": "case_trace_reference_only",
            }
        )
    for _, row in after_bad.iterrows():
        rows.append(
            {
                "ticker": row.get("ticker", ""),
                "candidate_month": row.get("candidate_month", ""),
                "as_of_date": row.get("as_of_date", ""),
                "branch_variant": row.get("branch_variant", ""),
                "before_case_trace_selected_violation": False,
                "after_case_trace_selected_violation": True,
                "fixed_label": row.get("branch_candidate_label", ""),
            }
        )
    if not rows:
        rows.append(
            {
                "ticker": "",
                "candidate_month": "",
                "as_of_date": "",
                "branch_variant": "",
                "before_case_trace_selected_violation": False,
                "after_case_trace_selected_violation": False,
                "fixed_label": "no_case_trace_selection_violation_after_fix",
            }
        )
    return pd.DataFrame(rows)


def _before_after_summary(before: pd.DataFrame, original: pd.DataFrame, fixed: pd.DataFrame) -> pd.DataFrame:
    original_case = original["case_trace_only"].map(_as_bool)
    fixed_case = fixed["case_trace_only"].map(_as_bool)
    return pd.DataFrame(
        [
            {
                "metric": "case_trace_selected_violation_rows",
                "before_value": int(len(before)),
                "after_value": int((fixed_case & fixed["branch_candidate_selected"].map(_as_bool)).sum()),
            },
            {
                "metric": "selected_rows_all",
                "before_value": int(original["branch_candidate_selected"].map(_as_bool).sum()),
                "after_value": int(fixed["branch_candidate_selected"].map(_as_bool).sum()),
            },
            {
                "metric": "case_trace_rows",
                "before_value": int(original_case.sum()),
                "after_value": int(fixed_case.sum()),
            },
        ]
    )


def _future_data_audit(path: Path, fixed: pd.DataFrame) -> pd.DataFrame:
    if path.exists():
        audit = pd.read_csv(path)
    else:
        audit = pd.DataFrame(columns=["audit_item", "rows", "future_data_violation", "reason"])
    extra = pd.DataFrame(
        [
            {
                "audit_item": "case_trace_selection_hygiene",
                "rows": int(len(fixed)),
                "future_data_violation": bool(
                    (fixed["case_trace_only"].map(_as_bool) & fixed["branch_candidate_selected"].map(_as_bool)).any()
                ),
                "reason": "case_trace_only rows must not be selected",
            }
        ]
    )
    return pd.concat([audit, extra], ignore_index=True, sort=False)


def _summary_text(manifest: dict[str, Any], summary: pd.DataFrame) -> str:
    lines = [
        "# Regime-aware RS/BIAS candidate scoring contract hygiene fix",
        "",
        "## 結論",
        "",
        "- 本包只修 case-trace selection hygiene，沒有 portfolio replay，也沒有改 formal/report/trade。",
        f"- before case_trace selected violation rows：{manifest['before_case_trace_selected_violation_rows']}",
        f"- after case_trace selected violation rows：{manifest['after_case_trace_selected_violation_rows']}",
        f"- future_data_violation_count：{manifest['future_data_violation_count']}",
        f"- ready_for_experiments：{manifest['ready_for_experiments']}",
        "",
        "## Summary",
        "",
    ]
    for _, row in summary.iterrows():
        lines.append(f"- {row['metric']}: {row['before_value']} -> {row['after_value']}")
    lines.extend(
        [
            "",
            "## 邊界",
            "",
            "- `case_trace_only=true` rows remain explanatory/reference only.",
            "- `branch_candidate_selected=false` for all case trace rows.",
            "- formal_model_changed=false；trade_decision_changed=false；active_in_trade_decision=false；report_changed=false。",
        ]
    )
    return "\n".join(lines) + "\n"


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Fix case-trace selection hygiene for regime-aware RS/BIAS contract.")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--source-contract", default=str(DEFAULT_SOURCE_CONTRACT))
    parser.add_argument("--source-future-audit", default=str(DEFAULT_SOURCE_FUTURE_AUDIT))
    parser.add_argument("--upstream-violation", default=str(DEFAULT_UPSTREAM_VIOLATION))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()
    manifest = run_regime_aware_rs_bias_candidate_scoring_hygiene_fix(
        repo_root=args.repo_root,
        source_contract=args.source_contract,
        source_future_audit=args.source_future_audit,
        upstream_violation=args.upstream_violation,
        output_dir=args.output_dir,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
