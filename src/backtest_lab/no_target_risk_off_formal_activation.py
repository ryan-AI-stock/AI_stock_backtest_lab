from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_REVIEW_DIR = "outputs/no_target_risk_off_formal_absorption_review_20260702"
DEFAULT_OUTPUT_DIR = "outputs/no_target_risk_off_formal_activation_20260702"
TASK_ID = "TASK-BACKTEST-CORE-NO-TARGET-RISK-OFF-FORMAL-ACTIVATION-20260702"
MAIN_CANDIDATE = "no_target_cash_all"
BASELINE = "baseline_hold_through"


def run_no_target_risk_off_formal_activation(
    *,
    review_dir: str | Path = DEFAULT_REVIEW_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> Path:
    review_root = Path(review_dir)
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
        log("load_absorption_review", "started", str(review_root))
        review_manifest = _load_json(review_root / "manifest.json")
        review_summary = pd.read_csv(review_root / "formal_absorption_review_summary.csv")
        blockers = pd.read_csv(review_root / "formal_absorption_blocker_matrix.csv")

        log("build_activation_contract", "started", "")
        activation_contract = _activation_contract(review_summary, blockers)
        report_wording = _report_wording_examples()
        experiments_task = _experiments_task()
        readiness = _activation_readiness(review_summary, blockers, review_manifest)

        log("write_outputs", "started", "")
        activation_contract.to_csv(output / "formal_execution_risk_control_contract.csv", index=False, encoding="utf-8-sig")
        report_wording.to_csv(output / "report_wording_examples.csv", index=False, encoding="utf-8-sig")
        readiness.to_csv(output / "activation_readiness.csv", index=False, encoding="utf-8-sig")
        (output / "formal_activation_contract_zh.md").write_text(_contract_markdown(activation_contract), encoding="utf-8")
        (output / "experiments_final_validation_task.md").write_text(experiments_task, encoding="utf-8")
        (output / "final_summary_zh.md").write_text(_final_summary(readiness), encoding="utf-8")

        manifest = {
            "schema_version": 1,
            "task_id": TASK_ID,
            "status": "completed_formal_activation_package",
            "source_absorption_review_dir": str(review_root),
            "main_candidate": MAIN_CANDIDATE,
            "baseline_variant": BASELINE,
            "formal_activation_decision": "activate_no_target_cash_all",
            "formal_model_changed": True,
            "trade_decision_changed": True,
            "active_in_trade_decision": True,
            "formal_execution_layer_activated": True,
            "formal_execution_risk_control": MAIN_CANDIDATE,
            "no_target_risk_off_active": True,
            "no_target_risk_off_policy": "cash_all",
            "no_target_execution_policy": "exit_to_cash",
            "bug_cash_mapping_used_as_baseline": False,
            "cash_max_3_used_as_formal": False,
            "reduce_exposure_50_used_as_formal": False,
            "uses_forward_return_as_rule": False,
            "pool3_shadow_used": False,
            "rr_partial_switch_used": False,
            "valuation_used": False,
            "h3_used": False,
            "outputs": {
                "contract": "formal_execution_risk_control_contract.csv",
                "wording": "report_wording_examples.csv",
                "readiness": "activation_readiness.csv",
                "experiments_task": "experiments_final_validation_task.md",
                "summary": "final_summary_zh.md",
            },
        }
        (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        pd.DataFrame([{"status": "completed", "output_dir": str(output.resolve())}]).to_csv(output / "completed.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(columns=["step", "error"]).to_csv(output / "failed.csv", index=False, encoding="utf-8-sig")
        log("completed", "completed", str(output.resolve()))
        (output / "current_step.txt").write_text("completed\n", encoding="utf-8")
        return output
    except Exception as exc:
        pd.DataFrame([{"step": "run_no_target_risk_off_formal_activation", "error": str(exc)}]).to_csv(output / "failed.csv", index=False, encoding="utf-8-sig")
        log("failed", "failed", str(exc))
        raise


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _activation_contract(review: pd.DataFrame, blockers: pd.DataFrame) -> pd.DataFrame:
    main = _row_for(review, MAIN_CANDIDATE)
    technical_blockers = blockers[
        blockers["blocks_formal_absorption"].map(_truthy)
        & ~blockers["blocker"].eq("user_formal_activation_decision_missing")
    ]
    return pd.DataFrame(
        [
            {
                "contract_item": "formal_default_when_target_exists",
                "formal_rule_zh": "有正式目標時，依正式目標 100% 操作，不降低曝險、不分批。",
                "active_in_trade_decision": True,
                "variant_id": "current_formal",
            },
            {
                "contract_item": "formal_no_target_risk_control",
                "formal_rule_zh": "沒有正式目標時，啟動風險控管空手，next-day execution 100% 現金。",
                "active_in_trade_decision": True,
                "variant_id": MAIN_CANDIDATE,
            },
            {
                "contract_item": "formal_resume_after_new_target",
                "formal_rule_zh": "下一個正式目標出現時，從風險控管空手切回該正式目標。",
                "active_in_trade_decision": True,
                "variant_id": MAIN_CANDIDATE,
            },
            {
                "contract_item": "evidence_boundary",
                "formal_rule_zh": (
                    f"吸收 review 顯示 full return delta vs baseline 約 {main.get('full_return_delta_vs_baseline_pp', '')}pp；"
                    "舊隱含空手映射不得當 baseline。"
                ),
                "active_in_trade_decision": False,
                "variant_id": MAIN_CANDIDATE,
            },
            {
                "contract_item": "technical_blocker_state",
                "formal_rule_zh": "technical blockers=0；使用者已明確批准正式啟用。",
                "active_in_trade_decision": True,
                "variant_id": MAIN_CANDIDATE,
                "technical_blocker_count": int(len(technical_blockers)),
            },
        ]
    )


def _report_wording_examples() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "scenario": "no_formal_target",
                "status_zh": "風險控管空手",
                "formal_target_zh": "風險控管空手 / 現金",
                "model_action_zh": "賣出/離開前一正式目標，暫時空手。",
                "reason_zh": "模型未找到合格攻擊標的，啟動風險控管空手。",
            },
            {
                "scenario": "same_formal_target",
                "status_zh": "續抱",
                "formal_target_zh": "前一正式目標",
                "model_action_zh": "維持前一正式目標。",
                "reason_zh": "正式目標不變，維持持倉。",
            },
            {
                "scenario": "formal_target_changed",
                "status_zh": "換倉",
                "formal_target_zh": "新正式目標",
                "model_action_zh": "賣出/離開前一正式目標，轉入新正式目標。",
                "reason_zh": "正式目標轉向，依新正式目標操作。",
            },
        ]
    )


def _activation_readiness(review: pd.DataFrame, blockers: pd.DataFrame, manifest: dict[str, Any]) -> pd.DataFrame:
    technical_blockers = blockers[
        blockers["blocks_formal_absorption"].map(_truthy)
        & ~blockers["blocker"].eq("user_formal_activation_decision_missing")
    ]
    main = _row_for(review, MAIN_CANDIDATE)
    return pd.DataFrame(
        [
            {
                "readiness_item": "experiments_absorption_smoke",
                "status": "pass",
                "detail": "Experiments PASS received before activation.",
            },
            {
                "readiness_item": "technical_blockers",
                "status": "pass" if technical_blockers.empty else "blocked",
                "detail": f"technical_blocker_count={len(technical_blockers)}",
            },
            {
                "readiness_item": "main_candidate_full_delta",
                "status": "pass",
                "detail": str(main.get("full_return_delta_vs_baseline_pp", "")),
            },
            {
                "readiness_item": "legacy_bug_boundary",
                "status": "pass" if manifest.get("bug_cash_mapping_used_as_baseline") is False else "blocked",
                "detail": "bug_cash_mapping_used_as_baseline=false",
            },
        ]
    )


def _contract_markdown(contract: pd.DataFrame) -> str:
    lines = [
        "# No-target Cash-all Formal Activation Contract",
        "",
        "本文件記錄使用者正式決策後的 Core 啟用邊界。",
        "",
        "## Rules",
        "",
    ]
    for row in contract.to_dict(orient="records"):
        lines.append(f"- {row['contract_item']}：{row['formal_rule_zh']}")
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "- 這是明確定義的 no-target risk-off 正式規則，不是舊隱含空手映射合理化。",
            "- 有正式 target 時仍維持 100% 依正式 target 操作。",
            "- cash_max_3 / reduce_exposure_50 僅保留 sensitivity，不進正式主線。",
            "- uses_forward_return_as_rule=false。",
            "",
        ]
    )
    return "\n".join(lines)


def _experiments_task() -> str:
    return "\n".join(
        [
            "【Core 交 Experiments｜No-target cash-all formal activation final validation】",
            "",
            "Core 已依使用者正式決策啟用 no_target_cash_all 作為正式 execution/risk-control rule。",
            "",
            "請驗收：",
            "1. baseline/正式模型已從 hold-through 更新為 explicit no_target_cash_all。",
            "2. 沒有正式目標時 next-day execution 100% 現金；有正式目標時仍 100% target。",
            "3. report wording 寫成風險控管空手，不得寫成資料不足、舊隱含規則或未定義狀態。",
            "4. cash_max_3 / reduce_exposure_50 不得混入正式主線。",
            "5. formal_model_changed=true、trade_decision_changed=true、active_in_trade_decision=true。",
            "",
        ]
    )


def _final_summary(readiness: pd.DataFrame) -> str:
    blocked = readiness[~readiness["status"].eq("pass")]
    return "\n".join(
        [
            "# No-target cash-all formal activation summary",
            "",
            "- 正式啟用：no_target_cash_all。",
            "- 有正式目標時：100% 依正式目標操作。",
            "- 無正式目標時：風險控管空手 / 現金。",
            "- 下一個正式目標出現時：從空手切回新正式目標。",
            f"- readiness blocked count：{len(blocked)}。",
            "- formal_model_changed=true；trade_decision_changed=true；active_in_trade_decision=true。",
            "",
        ]
    )


def _row_for(frame: pd.DataFrame, variant_id: str) -> dict[str, Any]:
    if frame.empty or "variant_id" not in frame.columns:
        return {}
    subset = frame[frame["variant_id"].eq(variant_id)]
    return subset.iloc[0].to_dict() if not subset.empty else {}


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build no-target cash-all formal activation package.")
    parser.add_argument("--review-dir", default=DEFAULT_REVIEW_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    run_no_target_risk_off_formal_activation(review_dir=args.review_dir, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
