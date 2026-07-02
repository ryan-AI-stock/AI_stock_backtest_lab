from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from backtest_lab.formal_model_contract import FORMAL_MODEL_ROUTE, FORMAL_MODEL_TARGET
from backtest_lab.regime_mode_switch import frozen_cycle_proven_top1_v1_variant


TASK_ID = "TASK-BACKTEST-CORE-DATE-AWARE-DYNAMIC-UNIVERSE-STATE-REPLAY-201411-20260702"
DEFAULT_PANEL_DIR = "outputs/current_formal_pool1_pool2_signal_panels_201411_202112_20260630"
DEFAULT_LIFECYCLE_DIR = "outputs/pool1_ticker_lifecycle_contract_201411_202112_20260702"
DEFAULT_OUTPUT_DIR = "outputs/date_aware_dynamic_universe_state_replay_201411_202112_20260702"


def run_date_aware_dynamic_universe_state_replay(
    *,
    panel_dir: str | Path = DEFAULT_PANEL_DIR,
    lifecycle_dir: str | Path = DEFAULT_LIFECYCLE_DIR,
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
        panel_root = Path(panel_dir)
        lifecycle_root = Path(lifecycle_dir)
        log("load_inputs", "started", f"{panel_root}; {lifecycle_root}")
        panel_manifest = _load_json(panel_root / "manifest.json")
        readiness = pd.read_csv(panel_root / "formal_policy_input_readiness.csv").fillna("")
        ranking = pd.read_csv(panel_root / "pool1_daily_candidate_ranking_panel.csv").fillna("")
        lifecycle = pd.read_csv(lifecycle_root / "pool1_ticker_lifecycle_contract.csv").fillna("")
        availability = pd.read_csv(lifecycle_root / "pool1_date_aware_candidate_availability_daily.csv").fillna("")

        log("build_dynamic_universe_contract", "started", "")
        contract = _dynamic_universe_contract()
        transition_rules = _state_transition_rules()
        coverage = _dynamic_universe_coverage(readiness, ranking, availability)
        blocked = _blocked_signal_rows(coverage)
        blockers = _blocker_by_field()
        source_decision = _source_decisions()

        log("write_outputs", "started", "")
        contract.to_csv(output / "dynamic_universe_state_replay_contract.csv", index=False, encoding="utf-8-sig")
        transition_rules.to_csv(output / "dynamic_universe_state_transition_rules.csv", index=False, encoding="utf-8-sig")
        coverage.to_csv(output / "dynamic_universe_state_replay_coverage.csv", index=False, encoding="utf-8-sig")
        blocked.to_csv(output / "blocked_signal_rows.csv", index=False, encoding="utf-8-sig")
        blockers.to_csv(output / "blocker_by_field.csv", index=False, encoding="utf-8-sig")
        source_decision.to_csv(output / "proxy_or_formal_source_decision.csv", index=False, encoding="utf-8-sig")
        (output / "next_step_handoff.md").write_text(_next_step_handoff(), encoding="utf-8")
        (output / "final_summary_zh.md").write_text(
            _final_summary(panel_manifest, coverage, blockers),
            encoding="utf-8",
        )

        output_manifest = {
            "schema_version": 1,
            "task_id": TASK_ID,
            "status": "completed_blocked_dynamic_universe_contract_package",
            "formal_model_target": FORMAL_MODEL_TARGET,
            "formal_model_route": FORMAL_MODEL_ROUTE,
            "date_start": str(panel_manifest.get("date_start") or _first_date(readiness)),
            "date_end": str(panel_manifest.get("date_end") or _last_date(readiness)),
            "dynamic_universe_contract_defined": True,
            "dynamic_universe_daily_coverage_rows": int(len(coverage)),
            "formal_ready_pool1_rows": 0,
            "blocked_signal_rows": int(len(blocked)),
            "pool1_lifecycle_contract_ready": True,
            "daily_dynamic_candidate_universe_ready": True,
            "dynamic_universe_state_replay_formal_ready": False,
            "missing_score_margin_panel": True,
            "missing_dynamic_state_machine_equivalence_test": True,
            "no_target_cash_all_applied": False,
            "proxy_used_as_formal": False,
            "uses_forward_return_as_rule": False,
            "formal_model_changed": False,
            "trade_decision_changed": False,
            "active_in_trade_decision": False,
            "next_required_task": "pool1_dynamic_universe_score_margin_and_state_machine_adapter",
            "outputs": {
                "contract": "dynamic_universe_state_replay_contract.csv",
                "transition_rules": "dynamic_universe_state_transition_rules.csv",
                "coverage": "dynamic_universe_state_replay_coverage.csv",
                "blocked_signal_rows": "blocked_signal_rows.csv",
                "blockers": "blocker_by_field.csv",
                "source_decision": "proxy_or_formal_source_decision.csv",
                "handoff": "next_step_handoff.md",
                "summary": "final_summary_zh.md",
            },
        }
        (output / "manifest.json").write_text(json.dumps(output_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        pd.DataFrame([{"status": "completed", "output_dir": str(output.resolve())}]).to_csv(
            output / "completed.csv",
            index=False,
            encoding="utf-8-sig",
        )
        pd.DataFrame(columns=["step", "error"]).to_csv(output / "failed.csv", index=False, encoding="utf-8-sig")
        log("completed", "completed", str(output.resolve()))
        (output / "current_step.txt").write_text("completed\n", encoding="utf-8")
        return output
    except Exception as exc:
        pd.DataFrame([{"step": "run_date_aware_dynamic_universe_state_replay", "error": str(exc)}]).to_csv(
            output / "failed.csv",
            index=False,
            encoding="utf-8-sig",
        )
        log("failed", "failed", str(exc))
        raise


def _dynamic_universe_contract() -> pd.DataFrame:
    variant = frozen_cycle_proven_top1_v1_variant()
    rows = [
        _contract_row(
            "daily_candidate_universe",
            "pool1_ticker_lifecycle_contract",
            "每日只納入 lifecycle-ready 且滿 60 日 warmup 的 ticker",
            "available",
            True,
            "",
        ),
        _contract_row(
            "daily_rank_score",
            "pool1_daily_candidate_ranking_panel.score",
            "可取得 dynamic universe 候選的 20/60 relative strength score",
            "available_for_candidates",
            True,
            "",
        ),
        _contract_row(
            "fallback_0050_score",
            "not stored in existing Pool1 ranking panel",
            f"attack gate margin needs top_score - 0050_score >= {variant.attack_gate_margin_over_fallback}",
            "missing",
            False,
            "missing_daily_fallback_score_margin_panel",
        ),
        _contract_row(
            "attack_gate_persistence",
            "current _attack_gate_passes uses prior dynamic top days",
            f"lookback={variant.attack_gate_persistence_lookback_days}; min_top_days={variant.attack_gate_min_top_days}",
            "missing_adapter",
            False,
            "missing_dynamic_universe_persistence_adapter",
        ),
        _contract_row(
            "attack_gate_reentry",
            "current variant reentry margin/acceleration",
            f"reentry_margin={variant.attack_gate_reentry_margin_over_fallback}; reentry_ratio={variant.attack_gate_reentry_min_short_to_medium_momentum_ratio}",
            "missing_adapter",
            False,
            "missing_reentry_state_transition_contract",
        ),
        _contract_row(
            "risk_off_active",
            "current market risk filter in regime_mode_switch",
            f"market_risk_off_filter={variant.market_risk_off_filter}",
            "missing_adapter",
            False,
            "missing_dynamic_universe_risk_off_state_replay",
        ),
        _contract_row(
            "equivalence_test",
            "2022+ static engine vs dynamic adapter",
            "dynamic adapter must match current formal output when universe is fully available",
            "missing",
            False,
            "missing_2022_plus_equivalence_regression",
        ),
    ]
    return pd.DataFrame(rows)


def _contract_row(
    field_name: str,
    source: str,
    rule: str,
    availability: str,
    formal_ready: bool,
    blocker: str,
) -> dict[str, Any]:
    return {
        "field_name": field_name,
        "source": source,
        "rule": rule,
        "2014_2021_availability": availability,
        "formal_ready": formal_ready,
        "blocker": blocker,
    }


def _state_transition_rules() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "state_or_event": "ticker_enters_candidate_universe",
                "rule": "ticker becomes candidate only from first_pool1_scoring_date onward",
                "formal_ready": True,
                "notes": "6669 starts no earlier than 2018-02-06 in this contract.",
            },
            {
                "state_or_event": "ticker_before_first_scoring_date",
                "rule": "ticker is excluded; no backfill and no placeholder score",
                "formal_ready": True,
                "notes": "Prevents not-yet-listed or warmup-insufficient ticker leakage.",
            },
            {
                "state_or_event": "current_target_removed_from_universe",
                "rule": "must be explicitly specified before formal replay",
                "formal_ready": False,
                "notes": "Existing target continuity cannot be inferred from ranking panel alone.",
            },
            {
                "state_or_event": "attack_gate_active",
                "rule": "carry forward prior state; update only from dynamic _attack_gate_passes result",
                "formal_ready": False,
                "notes": "Requires dynamic-universe score margin and persistence adapter.",
            },
            {
                "state_or_event": "attack_gate_ever_activated",
                "rule": "persist true after first activation unless current variant reentry reset condition triggers",
                "formal_ready": False,
                "notes": "Requires full state replay, not a daily proxy.",
            },
            {
                "state_or_event": "risk_off_active",
                "rule": "update via current market_risk_off_filter and gate interaction",
                "formal_ready": False,
                "notes": "Requires a dynamic adapter that mirrors regime_mode_switch state updates.",
            },
            {
                "state_or_event": "no_target_cash_all",
                "rule": "not applied in this step",
                "formal_ready": True,
                "notes": "Can only be applied after a full formal target stream is validated.",
            },
        ]
    )


def _dynamic_universe_coverage(readiness: pd.DataFrame, ranking: pd.DataFrame, availability: pd.DataFrame) -> pd.DataFrame:
    available_by_date = _available_by_date(availability)
    top_by_date = _top_by_date(ranking)
    rows: list[dict[str, Any]] = []
    for item in readiness.to_dict(orient="records"):
        date = str(item.get("date") or "")
        universe = available_by_date.get(date, [])
        top = top_by_date.get(date, {})
        rows.append(
            {
                "signal_date": date,
                "available_universe_count": len(universe),
                "candidate_tickers": "|".join(universe),
                "pool1_top_candidate": str(top.get("candidate_ticker") or ""),
                "pool1_top_candidate_score": _clean_text(top.get("score")),
                "daily_dynamic_candidate_universe_ready": True,
                "fallback_0050_score_ready": False,
                "score_margin_over_fallback_ready": False,
                "dynamic_persistence_state_ready": False,
                "dynamic_risk_off_state_ready": False,
                "state_equivalence_test_ready": False,
                "source_formal_ready": False,
                "replay_status": "blocked_missing_score_margin_and_state_machine_adapter",
                "reason": "每日 dynamic universe 已可列出，但缺 0050 fallback score/margin panel、dynamic _attack_gate_passes adapter 與 2022+ equivalence test。",
            }
        )
    return pd.DataFrame(rows)


def _blocked_signal_rows(coverage: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for item in coverage.to_dict(orient="records"):
        rows.append(
            {
                "signal_date": item["signal_date"],
                "available_universe_count": item["available_universe_count"],
                "candidate_tickers": item["candidate_tickers"],
                "pool1_target": "",
                "attack_gate_active": "",
                "target_is_actionable": "",
                "model_target_status": "",
                "reason": item["reason"],
                "source_formal_ready": False,
                "no_target_cash_all_applied": False,
            }
        )
    return pd.DataFrame(rows)


def _blocker_by_field() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "field_name": "fallback_0050_score",
                "blocker": "missing_daily_fallback_score_margin_panel",
                "severity": "blocking_attack_gate_passes",
                "required_code_contract": "Build a daily dynamic-universe score panel that includes 0050 fallback score and top_score minus fallback_score.",
                "formal_ready": False,
            },
            {
                "field_name": "dynamic_attack_gate_passes",
                "blocker": "missing_dynamic_universe_attack_gate_adapter",
                "severity": "blocking_attack_gate_active",
                "required_code_contract": "Refactor _attack_gate_passes logic to accept date-aware eligible score history without requiring static common-date prices_by_ticker.",
                "formal_ready": False,
            },
            {
                "field_name": "attack_gate_persistence_history",
                "blocker": "missing_dynamic_universe_persistence_history",
                "severity": "blocking_attack_gate_active",
                "required_code_contract": "Persist dynamic top ticker history for lookback=10/min_top_days=10 using only lifecycle-ready candidates.",
                "formal_ready": False,
            },
            {
                "field_name": "risk_off_active",
                "blocker": "missing_dynamic_risk_off_state_replay",
                "severity": "blocking_model_target_status",
                "required_code_contract": "Mirror current regime_mode_switch risk-off update rules after attack gate state is available.",
                "formal_ready": False,
            },
            {
                "field_name": "equivalence_test",
                "blocker": "missing_2022_plus_equivalence_regression",
                "severity": "blocking_formal_readiness",
                "required_code_contract": "Run dynamic adapter over 2022+ and prove it matches existing formal target stream before using it for 2014-2021.",
                "formal_ready": False,
            },
        ]
    )


def _source_decisions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "source_layer": "lifecycle_daily_availability",
                "source_path": DEFAULT_LIFECYCLE_DIR + "/pool1_date_aware_candidate_availability_daily.csv",
                "status": "accepted",
                "decision": "可作 dynamic candidate universe provider。",
                "formal_or_proxy": "formal_input",
            },
            {
                "source_layer": "pool1_ranking_panel",
                "source_path": DEFAULT_PANEL_DIR + "/pool1_daily_candidate_ranking_panel.csv",
                "status": "accepted_partial",
                "decision": "可提供候選分數與 top candidate，但沒有 0050 fallback score/margin。",
                "formal_or_proxy": "partial_input",
            },
            {
                "source_layer": "ranking_first_as_formal_target",
                "source_path": "",
                "status": "rejected",
                "decision": "不得把 ranking first 包裝成 formal target 或 attack gate active。",
                "formal_or_proxy": "proxy_rejected",
            },
            {
                "source_layer": "no_target_cash_all",
                "source_path": "formal execution risk-control rule",
                "status": "not_applied",
                "decision": "本步尚未產出完整 formal target stream，不得套用 no-target cash-all。",
                "formal_or_proxy": "formal_rule_waiting_for_target_stream",
            },
        ]
    )


def _next_step_handoff() -> str:
    return "\n".join(
        [
            "# Date-aware dynamic universe state replay handoff",
            "",
            "## 判定",
            "已定義 date-aware dynamic universe adapter contract，並產出每日 candidate universe coverage；但 2014/11～2021 Pool1 attack gate state 尚未 formal-ready。",
            "",
            "## 為什麼不能直接 formal-ready",
            "現有 Pool1 ranking panel 沒有保存 0050 fallback score 與 top_score - fallback_score margin；現行 `_attack_gate_passes` 又依賴 static `prices_by_ticker` 與 prior top history。若沒有 dynamic-universe score margin panel 與 2022+ equivalence test，就不能宣稱重放結果等於現行正式模型。",
            "",
            "## 下一個最小任務",
            "`TASK-BACKTEST-CORE-POOL1-DYNAMIC-UNIVERSE-SCORE-MARGIN-AND-STATE-ADAPTER-201411-20260702`：",
            "1. 產每日 dynamic score panel，包含候選 score、0050 fallback score、score margin。",
            "2. 實作 dynamic `_attack_gate_passes` adapter，使用 lifecycle-ready universe 與 persistence 10/10。",
            "3. 在 2022+ 做 equivalence regression，證明 adapter 與現行 formal stream 對齊。",
            "",
            "## 禁止",
            "- 不得把 ranking first 當 formal target。",
            "- 不得套 no-target cash-all。",
            "- 不得交 Experiments 跑績效。",
        ]
    ) + "\n"


def _final_summary(panel_manifest: dict[str, Any], coverage: pd.DataFrame, blockers: pd.DataFrame) -> str:
    counts = coverage["available_universe_count"].astype(int)
    return "\n".join(
        [
            "# Date-aware dynamic universe state replay",
            "",
            "## 判定",
            "本棒已建立 date-aware dynamic universe adapter contract，且能逐日列出 lifecycle-ready candidate universe；但 Pool1 attack gate state 仍不能 formal-ready 重放。",
            "",
            "## 本批輸出",
            f"- 來源區間：{panel_manifest.get('date_start')}～{panel_manifest.get('date_end')}",
            f"- coverage rows：{len(coverage)}",
            f"- dynamic universe count range：{int(counts.min()) if not counts.empty else 0}～{int(counts.max()) if not counts.empty else 0}",
            f"- blocked fields：{len(blockers)}",
            "",
            "## 主要 blocker",
            "缺每日 0050 fallback score / score margin panel、dynamic `_attack_gate_passes` adapter、以及 2022+ equivalence regression。這些完成前，不得產 formal-ready Pool1 rows。",
            "",
            "- formal_model_changed=false",
            "- trade_decision_changed=false",
            "- no_target_cash_all_applied=false",
        ]
    ) + "\n"


def _available_by_date(availability: pd.DataFrame) -> dict[str, list[str]]:
    frame = availability[availability["candidate_available_for_pool1_ranking"].map(_bool_like)].copy()
    if frame.empty:
        return {}
    result: dict[str, list[str]] = {}
    for date, group in frame.groupby("date"):
        result[str(date)] = sorted(group["ticker"].astype(str).tolist())
    return result


def _top_by_date(ranking: pd.DataFrame) -> dict[str, dict[str, Any]]:
    if ranking.empty:
        return {}
    frame = ranking.copy()
    frame["rank_num"] = pd.to_numeric(frame.get("rank", ""), errors="coerce")
    top = frame.sort_values(["date", "rank_num", "candidate_ticker"]).groupby("date", as_index=False).first()
    return {str(item.get("date") or ""): item for item in top.to_dict(orient="records")}


def _clean_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value)


def _bool_like(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _first_date(frame: pd.DataFrame) -> str:
    return "" if frame.empty else str(frame["date"].iloc[0])


def _last_date(frame: pd.DataFrame) -> str:
    return "" if frame.empty else str(frame["date"].iloc[-1])


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build date-aware dynamic universe state replay contract package.")
    parser.add_argument("--panel-dir", default=DEFAULT_PANEL_DIR)
    parser.add_argument("--lifecycle-dir", default=DEFAULT_LIFECYCLE_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    output = run_date_aware_dynamic_universe_state_replay(
        panel_dir=args.panel_dir,
        lifecycle_dir=args.lifecycle_dir,
        output_dir=args.output_dir,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
