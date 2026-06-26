from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


TASK_ID = "TASK-BACKTEST-CORE-FINAL-DECISION-LAYER-REVIEW-AFTER-POOL1-POOL2-ABSORPTION-001"
DEFAULT_ABSORPTION_DIR = "outputs/formal_absorb_pool1_pool2_combined_cap40_confirmation1_20260626"
DEFAULT_OUTPUT_DIR = "outputs/final_decision_layer_review_after_pool1_pool2_absorption_20260626"


def run_final_decision_layer_review_after_absorption(
    *,
    absorption_dir: str | Path = DEFAULT_ABSORPTION_DIR,
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
        (output / "current_step.txt").write_text(step, encoding="utf-8")

    try:
        source = Path(absorption_dir)
        log("load_absorption_package", "started", str(source))
        manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
        before_after = pd.read_csv(source / "formal_selector_before_after.csv").fillna("")
        blockers = pd.read_csv(source / "blocker_matrix.csv").fillna("")
        _validate_absorption_manifest(manifest)

        log("build_review", "started", "")
        review = _review_rows(manifest)
        state_contract = _state_contract()
        retired = _retired_three_pool_layer(before_after)
        next_steps = _next_step_recommendation(blockers)

        review.to_csv(output / "final_decision_layer_review.csv", index=False, encoding="utf-8-sig")
        state_contract.to_csv(output / "two_pool_decision_state_contract.csv", index=False, encoding="utf-8-sig")
        retired.to_csv(output / "retired_three_pool_decision_layer.csv", index=False, encoding="utf-8-sig")
        next_steps.to_csv(output / "next_step_recommendation.csv", index=False, encoding="utf-8-sig")
        (output / "final_decision_layer_review_summary_zh.md").write_text(
            _summary_markdown(manifest, review, next_steps),
            encoding="utf-8",
        )

        output_manifest = {
            "schema_version": 1,
            "task_id": TASK_ID,
            "status": "completed_review",
            "source_absorption_dir": str(source),
            "formal_model_target": manifest.get("formal_model_target"),
            "formal_model_route": manifest.get("formal_model_route"),
            "formal_model_changed": False,
            "trade_decision_changed": False,
            "active_in_trade_decision": False,
            "review_only": True,
            "three_pool_decision_layer_retired_from_formal_route": True,
            "needs_legacy_three_pool_final_decision_layer": False,
            "needs_new_formal_decision_layer_before_execution_review": False,
            "recommended_replacement": "two_pool_formal_decision_audit_and_report_boundary",
            "execution_layer_can_continue": True,
            "pool3_blocks_mainline": False,
            "pool3_backlog_removed_from_mainline": True,
            "market_exposure_override_absorbed": False,
            "0050x2_opportunity_label_active_in_trade_decision": False,
            "uses_forward_return_as_rule": False,
        }
        (output / "manifest.json").write_text(json.dumps(output_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        pd.DataFrame([{"status": "completed_review", "output_dir": str(output.resolve())}]).to_csv(
            output / "completed.csv",
            index=False,
            encoding="utf-8-sig",
        )
        pd.DataFrame(columns=["step", "error"]).to_csv(output / "failed.csv", index=False, encoding="utf-8-sig")
        log("completed", "completed", str(output.resolve()))
        (output / "current_step.txt").write_text("completed\n", encoding="utf-8")
        return output
    except Exception as exc:
        pd.DataFrame([{"step": "run_final_decision_layer_review_after_absorption", "error": str(exc)}]).to_csv(
            output / "failed.csv",
            index=False,
            encoding="utf-8-sig",
        )
        log("failed", "failed", str(exc))
        raise


def _validate_absorption_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("formal_model_target") != "combined_cap40_confirmation1_base":
        raise ValueError("absorption manifest is not the expected Pool1+Pool2 formal target")
    if not _truthy(manifest.get("formal_absorption_ready")):
        raise ValueError("absorption manifest is not formal ready")
    if not _truthy(manifest.get("three_pool_formal_route_abandoned")):
        raise ValueError("three-pool formal route was not abandoned")
    if _truthy(manifest.get("pool3_shadow_used_as_formal")):
        raise ValueError("Pool3 shadow is unexpectedly formal")


def _review_rows(manifest: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "review_item": "legacy_three_pool_final_decision_layer",
                "decision": "retire_from_formal_route",
                "reason": "Formal selector is no longer 2/3 three-pool consensus; it is Pool1 primary plus Pool2 confirmation/risk.",
                "active_in_trade_decision": False,
            },
            {
                "review_item": "new_final_decision_layer",
                "decision": "not_needed_before_execution_review",
                "reason": "Pool1+Pool2 contract already defines formal target formation; adding another decision layer now would duplicate selector logic.",
                "active_in_trade_decision": False,
            },
            {
                "review_item": "replacement_diagnostic",
                "decision": "keep_two_pool_decision_audit",
                "reason": "Reports should explain Pool1 target, Pool2 confirmation/disagreement, cap40, cash/no-target and opportunity-cost caveat.",
                "active_in_trade_decision": False,
            },
            {
                "review_item": "pool3",
                "decision": "remove_from_mainline_blockers",
                "reason": "Pool3 remains a future candidate source/research topic, but no longer blocks final decision or execution layer work.",
                "active_in_trade_decision": False,
            },
            {
                "review_item": "execution_layer",
                "decision": "can_continue_next",
                "reason": "A stable formal target stream now exists; execution review can proceed against combined_cap40_confirmation1_base.",
                "active_in_trade_decision": False,
            },
        ]
    )


def _state_contract() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "state": "formal_target_active",
                "meaning": "Pool1 target is active and Pool2 state does not block it, or confirmation1 is satisfied.",
                "formal_selector_effect": "target_or_weighted_target_active",
            },
            {
                "state": "pool2_confirmation_pending",
                "meaning": "Pool2 disagrees with Pool1 and one prior same Pool1 signal day has not been satisfied.",
                "formal_selector_effect": "no_new_target_or_cash_per_contract",
            },
            {
                "state": "pool2_risk_blocked",
                "meaning": "Pool2 risk/confirmation layer blocks the Pool1 target under the formal contract.",
                "formal_selector_effect": "no_new_target_or_cash_per_contract",
            },
            {
                "state": "cap40_applied",
                "meaning": "00631L target weight is capped at 40%; residual remains cash.",
                "formal_selector_effect": "weighted_target_active",
            },
            {
                "state": "opportunity_cost_label_only",
                "meaning": "0050x2 opportunity-cost caveat is shown for reading context only.",
                "formal_selector_effect": "no_effect",
            },
            {
                "state": "diagnostic_only_context",
                "meaning": "Three-pool, Pool3, final decision legacy labels, chip shadow or other diagnostics.",
                "formal_selector_effect": "no_effect",
            },
        ]
    )


def _retired_three_pool_layer(before_after: pd.DataFrame) -> pd.DataFrame:
    route = before_after[before_after["dimension"].astype(str).eq("formal_route")].copy()
    if route.empty:
        return pd.DataFrame(
            [
                {
                    "legacy_layer": "three_pool_2_of_3_consensus",
                    "formal_status": "retired",
                    "replacement": "combined_cap40_confirmation1_base",
                    "evidence": "formal absorption manifest",
                }
            ]
        )
    row = route.iloc[0].to_dict()
    return pd.DataFrame(
        [
            {
                "legacy_layer": row.get("before", "current_formal_three_pool_baseline"),
                "formal_status": "retired",
                "replacement": row.get("after", "combined_cap40_confirmation1_base"),
                "evidence": row.get("evidence", "three_pool_formal_route_abandoned=true"),
            }
        ]
    )


def _next_step_recommendation(blockers: pd.DataFrame) -> pd.DataFrame:
    caveat_count = int(len(blockers))
    blocking = int(blockers.get("blocks_formal_absorption", pd.Series(dtype=bool)).map(_truthy).sum()) if not blockers.empty else 0
    return pd.DataFrame(
        [
            {
                "priority": 1,
                "next_step": "execution_layer_review",
                "recommendation": "Continue with execution/switch layer review against the new Pool1+Pool2 formal target stream.",
                "blocked": False,
                "reason": "Formal target stream is stable; legacy final decision layer is not needed as a prerequisite.",
            },
            {
                "priority": 2,
                "next_step": "overall_model_diagnostic_and_formal_backtest",
                "recommendation": "Run after execution review is complete.",
                "blocked": True,
                "reason": "Execution layer still needs review before final whole-model diagnostic/backtest.",
            },
            {
                "priority": 3,
                "next_step": "pool3_future_research",
                "recommendation": "Remove from mainline queue; revisit only when a new source/scoring hypothesis exists.",
                "blocked": False,
                "reason": "Pool3 should not block Pool1+Pool2 formal model stabilization.",
            },
            {
                "priority": 4,
                "next_step": "residual_caveats",
                "recommendation": f"Track {caveat_count} residual caveats; blocking caveats={blocking}.",
                "blocked": bool(blocking),
                "reason": "Current caveats are report-only or future validation items.",
            },
        ]
    )


def _summary_markdown(manifest: dict[str, Any], review: pd.DataFrame, next_steps: pd.DataFrame) -> str:
    lines = [
        "# Final Decision Layer Review After Pool1+Pool2 Absorption",
        "",
        "## 結論",
        "",
        "- 三池版 final decision layer 不再需要作為正式決策層，應從 formal route 退役。",
        "- 目前不需要新增另一個 formal final decision layer；Pool1+Pool2 contract 已經定義 target formation。",
        "- 需要保留的是二池 formal decision audit / report boundary，用來解釋 Pool1、Pool2、cap40、cash/no-target 與 0050正二機會成本 caveat。",
        "- 下一步可以進 execution / switch layer review。",
        "- Pool3 從主線 blocker 移除；未來有新候選來源或 scoring 假設時再回來處理。",
        "",
        "## Formal Target",
        "",
        f"- formal_model_target: `{manifest.get('formal_model_target')}`",
        f"- formal_model_route: `{manifest.get('formal_model_route')}`",
        "",
        "## Review Items",
        "",
    ]
    for row in review.to_dict(orient="records"):
        lines.append(f"- {row['review_item']}: {row['decision']}。{row['reason']}")
    lines.extend(["", "## Next Steps", ""])
    for row in next_steps.sort_values("priority").to_dict(orient="records"):
        lines.append(f"- P{row['priority']} {row['next_step']}: {row['recommendation']}")
    lines.extend(["", "本輸出為 Core review/report-only，不改正式模型與交易決策。", ""])
    return "\n".join(lines)


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Review final decision layer after Pool1+Pool2 formal absorption.")
    parser.add_argument("--absorption-dir", default=DEFAULT_ABSORPTION_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    output = run_final_decision_layer_review_after_absorption(absorption_dir=args.absorption_dir, output_dir=args.output_dir)
    print(f"OUTPUT_DIR={output.resolve()}")


if __name__ == "__main__":
    main()
