from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_SOURCE_DIR = "outputs/no_target_risk_off_formal_challenger_smoke_20260702"
DEFAULT_OUTPUT_DIR = "outputs/no_target_risk_off_formal_absorption_review_20260702"
TASK_ID = "TASK-BACKTEST-CORE-NO-TARGET-RISK-OFF-FORMAL-ABSORPTION-REVIEW-20260702"
MAIN_CANDIDATE = "no_target_cash_all"
BASELINE = "baseline_hold_through"


def run_no_target_risk_off_absorption_review(
    *,
    source_dir: str | Path = DEFAULT_SOURCE_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> Path:
    source = Path(source_dir)
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
        (output / "current_step.txt").write_text(step, encoding="utf-8")

    try:
        log("load_core_smoke", "started", str(source))
        manifest = _load_json(source / "manifest.json")
        performance = pd.read_csv(source / "performance_by_variant.csv")
        variant_contract = pd.read_csv(source / "variant_contract.csv")
        before_after = pd.read_csv(source / "formal_challenger_before_after_contract.csv")
        event_attribution = pd.read_csv(source / "no_target_event_attribution.csv")
        trade_cost = pd.read_csv(source / "trade_cost_summary.csv")

        log("build_review", "started", "")
        review = _absorption_review_summary(performance, variant_contract, manifest)
        blockers = _blocker_matrix(review, variant_contract, manifest)
        contract = _formal_absorption_contract(review, blockers, manifest)
        report_diff = _report_contract_diff(before_after, variant_contract)
        handoff = _experiments_handoff(review, blockers)

        log("write_outputs", "started", "")
        review.to_csv(output / "formal_absorption_review_summary.csv", index=False, encoding="utf-8-sig")
        blockers.to_csv(output / "formal_absorption_blocker_matrix.csv", index=False, encoding="utf-8-sig")
        before_after.to_csv(output / "formal_challenger_before_after_contract.csv", index=False, encoding="utf-8-sig")
        report_diff.to_csv(output / "report_contract_before_after.csv", index=False, encoding="utf-8-sig")
        event_attribution.to_csv(output / "no_target_event_attribution.csv", index=False, encoding="utf-8-sig")
        trade_cost.to_csv(output / "trade_cost_summary.csv", index=False, encoding="utf-8-sig")
        (output / "formal_absorption_contract_zh.md").write_text(contract, encoding="utf-8")
        (output / "next_experiments_absorption_validation_task.md").write_text(handoff, encoding="utf-8")
        (output / "final_summary_zh.md").write_text(_final_summary(review, blockers), encoding="utf-8")

        output_manifest = {
            "schema_version": 1,
            "task_id": TASK_ID,
            "status": "completed_absorption_review_ready_for_user_decision",
            "source_smoke_dir": str(source),
            "experiments_smoke_passed": True,
            "main_candidate": MAIN_CANDIDATE,
            "baseline_variant": BASELINE,
            "formal_absorption_review_ready": True,
            "formal_absorption_candidate_ready": _ready(review, blockers),
            "formal_absorption_activated": False,
            "requires_user_formal_decision_before_activation": True,
            "formal_model_changed": False,
            "trade_decision_changed": False,
            "active_in_trade_decision": False,
            "bug_cash_mapping_used_as_baseline": False,
            "explicit_no_target_risk_off_challenger": True,
            "uses_forward_return_as_rule": False,
            "outputs": {
                "review": "formal_absorption_review_summary.csv",
                "blockers": "formal_absorption_blocker_matrix.csv",
                "contract": "formal_absorption_contract_zh.md",
                "event_attribution": "no_target_event_attribution.csv",
                "handoff": "next_experiments_absorption_validation_task.md",
                "summary": "final_summary_zh.md",
            },
        }
        (output / "manifest.json").write_text(json.dumps(output_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        pd.DataFrame([{"status": "completed", "output_dir": str(output.resolve())}]).to_csv(output / "completed.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(columns=["step", "error"]).to_csv(output / "failed.csv", index=False, encoding="utf-8-sig")
        log("completed", "completed", str(output.resolve()))
        (output / "current_step.txt").write_text("completed\n", encoding="utf-8")
        return output
    except Exception as exc:
        pd.DataFrame([{"step": "run_no_target_risk_off_absorption_review", "error": str(exc)}]).to_csv(output / "failed.csv", index=False, encoding="utf-8-sig")
        log("failed", "failed", str(exc))
        raise


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _absorption_review_summary(performance: pd.DataFrame, contract: pd.DataFrame, manifest: dict[str, Any]) -> pd.DataFrame:
    full = performance[performance["period_label"].eq("full_available")].copy()
    hard_gate = performance[performance["period_label"].eq("2024_hard_gate")].copy()
    ytd = performance[performance["period_label"].eq("2026_ytd")].copy()
    rows = []
    for variant_id in (BASELINE, MAIN_CANDIDATE, "no_target_cash_max_3", "no_target_reduce_exposure_50"):
        full_row = _row_for(full, variant_id)
        hard_row = _row_for(hard_gate, variant_id)
        ytd_row = _row_for(ytd, variant_id)
        contract_row = _row_for(contract, variant_id)
        rows.append(
            {
                "variant_id": variant_id,
                "role": "baseline" if variant_id == BASELINE else ("main_absorption_candidate" if variant_id == MAIN_CANDIDATE else "sensitivity"),
                "no_formal_target_policy": contract_row.get("no_formal_target_policy", ""),
                "full_return_pct": full_row.get("return_pct", ""),
                "full_mdd_pct": full_row.get("max_drawdown_pct", ""),
                "full_trade_rows": full_row.get("trade_rows", ""),
                "full_total_transaction_cost": full_row.get("total_transaction_cost", ""),
                "hard_gate_2024_return_pct": hard_row.get("return_pct", ""),
                "hard_gate_2024_mdd_pct": hard_row.get("max_drawdown_pct", ""),
                "ytd_2026_return_pct": ytd_row.get("return_pct", ""),
                "ytd_2026_mdd_pct": ytd_row.get("max_drawdown_pct", ""),
                "formal_model_changed": False,
                "trade_decision_changed": False,
                "active_in_trade_decision": False,
                "bug_cash_mapping_used_as_baseline": manifest.get("bug_cash_mapping_used_as_baseline", False),
            }
        )
    baseline = _row_by_variant(rows, BASELINE)
    main = _row_by_variant(rows, MAIN_CANDIDATE)
    if baseline and main:
        main["full_return_delta_vs_baseline_pp"] = _num(main["full_return_pct"]) - _num(baseline["full_return_pct"])
        main["full_mdd_delta_vs_baseline_pp"] = _num(main["full_mdd_pct"]) - _num(baseline["full_mdd_pct"])
        main["hard_gate_return_delta_vs_baseline_pp"] = _num(main["hard_gate_2024_return_pct"]) - _num(baseline["hard_gate_2024_return_pct"])
        main["candidate_review_decision"] = "ready_for_formal_absorption_decision"
    return pd.DataFrame(rows)


def _blocker_matrix(review: pd.DataFrame, contract: pd.DataFrame, manifest: dict[str, Any]) -> pd.DataFrame:
    main = review[review["variant_id"].eq(MAIN_CANDIDATE)].iloc[0].to_dict()
    blockers = [
        {
            "blocker": "user_formal_activation_decision_missing",
            "severity": "requires_decision",
            "blocks_formal_absorption": True,
            "owner": "user_strategy_center",
            "minimum_fix": "使用者明確批准後，Core 才能把 no_target_cash_all 吸收為正式模型行為。",
        },
        {
            "blocker": "legacy_bug_cash_mapping_boundary",
            "severity": "controlled",
            "blocks_formal_absorption": False,
            "owner": "core",
            "minimum_fix": "保留 wording/manifest：bug_cash_mapping_used_as_baseline=false。",
        },
        {
            "blocker": "sensitivity_not_mainline",
            "severity": "controlled",
            "blocks_formal_absorption": False,
            "owner": "core_experiments",
            "minimum_fix": "no_target_cash_max_3 與 reduce_exposure_50 只保留 sensitivity/shadow，不進主線。",
        },
    ]
    if manifest.get("bug_cash_mapping_used_as_baseline") is not False:
        blockers.append(
            {
                "blocker": "bug_cash_mapping_used_as_baseline",
                "severity": "critical",
                "blocks_formal_absorption": True,
                "owner": "core",
                "minimum_fix": "不得把舊隱含 bug 當正式 baseline。",
            }
        )
    if _num(main.get("full_return_delta_vs_baseline_pp")) <= 0:
        blockers.append(
            {
                "blocker": "main_candidate_full_return_not_above_baseline",
                "severity": "critical",
                "blocks_formal_absorption": True,
                "owner": "experiments",
                "minimum_fix": "重新驗證 full period apples-to-apples。",
            }
        )
    return pd.DataFrame(blockers)


def _formal_absorption_contract(review: pd.DataFrame, blockers: pd.DataFrame, manifest: dict[str, Any]) -> str:
    main = review[review["variant_id"].eq(MAIN_CANDIDATE)].iloc[0].to_dict()
    blocking = int(blockers["blocks_formal_absorption"].map(_truthy).sum()) if not blockers.empty else 0
    return "\n".join(
        [
            "# No-target Risk-off Formal Absorption Review Contract",
            "",
            "本文件是 Core formal absorption review，不是正式吸收 commit。",
            "",
            "## Candidate",
            "",
            f"- 主候選：`{MAIN_CANDIDATE}`",
            "- 語意：no formal target 出現時，顯式進入 risk-off / cash，直到下一個正式目標出現。",
            "- baseline：`baseline_hold_through`，也就是 e7487dc 後的正式修正版。",
            "- 舊 bug cash mapping 不得當正式 baseline。",
            "",
            "## Evidence",
            "",
            f"- full return delta vs baseline：{main.get('full_return_delta_vs_baseline_pp', '')}pp",
            f"- full MDD delta vs baseline：{main.get('full_mdd_delta_vs_baseline_pp', '')}pp",
            f"- 2024 hard gate return delta vs baseline：{main.get('hard_gate_return_delta_vs_baseline_pp', '')}pp",
            "",
            "## Activation Boundary",
            "",
            "- formal_model_changed=false",
            "- trade_decision_changed=false",
            "- active_in_trade_decision=false",
            "- formal_absorption_activated=false",
            "- requires_user_formal_decision_before_activation=true",
            f"- blocking_items_before_activation={blocking}",
            "",
            "## Report Wording Boundary",
            "",
            "- 可以寫：顯式 no-target risk-off challenger。",
            "- 不可寫：舊 bug 被正式化。",
            "- 不可寫：目前正式日報已自動 no-target 空手。",
            "",
        ]
    )


def _report_contract_diff(before_after: pd.DataFrame, contract: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, item in before_after.iterrows():
        rows.append(
            {
                "contract_stage": item.get("contract_stage", ""),
                "variant_id": item.get("variant_id", ""),
                "no_formal_target_policy": item.get("no_formal_target_policy", ""),
                "visible_report_wording": (
                    "正式 baseline：無新正式目標時沿用前一正式持倉。"
                    if item.get("variant_id") == BASELINE
                    else "正式吸收前只能寫成顯式 risk-off 候選，不可寫成已正式啟用。"
                ),
                "formal_report_changed": False,
            }
        )
    return pd.DataFrame(rows)


def _experiments_handoff(review: pd.DataFrame, blockers: pd.DataFrame) -> str:
    return "\n".join(
        [
            "【Core 交 Experiments｜No-target risk-off formal absorption validation】",
            "",
            "請驗收 Core formal absorption review package：",
            f"- main candidate: `{MAIN_CANDIDATE}`",
            f"- baseline: `{BASELINE}`",
            "- status: ready_for_user_formal_decision，不是已正式吸收。",
            "",
            "請確認：",
            "1. no_target_cash_all 是否可進 formal absorption。",
            "2. 使用者批准前，formal_model_changed/trade_decision_changed 必須保持 false。",
            "3. wording 不得把 legacy bug cash mapping 寫成正式規則。",
            "4. sensitivity 不得升主線。",
            "",
        ]
    )


def _final_summary(review: pd.DataFrame, blockers: pd.DataFrame) -> str:
    main = review[review["variant_id"].eq(MAIN_CANDIDATE)].iloc[0].to_dict()
    blocking = int(blockers["blocks_formal_absorption"].map(_truthy).sum()) if not blockers.empty else 0
    return "\n".join(
        [
            "# No-target risk-off formal absorption review summary",
            "",
            f"- 主候選：`{MAIN_CANDIDATE}`。",
            "- 結論：Core review package 已完成，可交使用者/策略中心做正式吸收決策。",
            "- 目前未啟用正式模型，日報正式 trade action 不變。",
            f"- full return delta vs baseline：{main.get('full_return_delta_vs_baseline_pp', '')}pp。",
            f"- full MDD delta vs baseline：{main.get('full_mdd_delta_vs_baseline_pp', '')}pp。",
            f"- activation 前 blocker count：{blocking}；主要 blocker 是使用者尚未正式批准吸收。",
            "- bug_cash_mapping_used_as_baseline=false。",
            "",
        ]
    )


def _ready(review: pd.DataFrame, blockers: pd.DataFrame) -> bool:
    if review.empty or blockers.empty:
        return False
    technical_blockers = blockers[
        blockers["blocks_formal_absorption"].map(_truthy)
        & ~blockers["blocker"].eq("user_formal_activation_decision_missing")
    ]
    return technical_blockers.empty


def _row_for(frame: pd.DataFrame, variant_id: str) -> dict[str, Any]:
    if frame.empty or "variant_id" not in frame.columns:
        return {}
    subset = frame[frame["variant_id"].eq(variant_id)]
    return subset.iloc[0].to_dict() if not subset.empty else {}


def _row_by_variant(rows: list[dict[str, Any]], variant_id: str) -> dict[str, Any] | None:
    return next((row for row in rows if row.get("variant_id") == variant_id), None)


def _num(value: object) -> float:
    return float(pd.to_numeric(pd.Series([value]), errors="coerce").fillna(0.0).iloc[0])


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build no-target risk-off formal absorption review package.")
    parser.add_argument("--source-dir", default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    run_no_target_risk_off_absorption_review(source_dir=args.source_dir, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
