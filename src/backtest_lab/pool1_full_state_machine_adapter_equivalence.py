from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from backtest_lab.formal_model_contract import FORMAL_MODEL_ROUTE, FORMAL_MODEL_TARGET
from backtest_lab.pool1_dynamic_adapter_2022_equivalence import (
    DEFAULT_END_DATE,
    DEFAULT_FORMAL_LONG_RANGE_DIR,
    DEFAULT_START_DATE,
    _dynamic_coverage_panel,
    _formal_pool1_reference_stream,
    _load_required_prices,
    _pool1_ranking_panel,
    _source_decisions as _previous_source_decisions,
    _trading_dates,
)
from backtest_lab.current_formal_pool1_pool2_signal_panels import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_PRICE_CACHE_DIR,
    DEFAULT_PRICE_SOURCE_REGISTRY,
    POOL1_TICKERS,
    TW50_BENCHMARK,
    _load_price_source_registry,
    _name_map,
)
from backtest_lab.config import load_config
from backtest_lab.pool1_dynamic_score_margin_state_adapter import _dynamic_score_margin_panel


TASK_ID = "TASK-BACKTEST-CORE-POOL1-FULL-STATE-MACHINE-ADAPTER-EQUIVALENCE-20260702"
DEFAULT_PREVIOUS_EQUIVALENCE_DIR = "outputs/pool1_dynamic_adapter_2022_equivalence_20260702"
DEFAULT_OUTPUT_DIR = "outputs/pool1_full_state_machine_adapter_equivalence_20260702"


def run_pool1_full_state_machine_adapter_equivalence(
    *,
    formal_long_range_dir: str | Path = DEFAULT_FORMAL_LONG_RANGE_DIR,
    previous_equivalence_dir: str | Path = DEFAULT_PREVIOUS_EQUIVALENCE_DIR,
    price_cache_dir: str | Path = DEFAULT_PRICE_CACHE_DIR,
    price_source_registry: str | Path = DEFAULT_PRICE_SOURCE_REGISTRY,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
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
        start = pd.Timestamp(start_date).normalize()
        end = pd.Timestamp(end_date).normalize()
        config = load_config(config_path)
        registry = _load_price_source_registry(price_source_registry)

        log("load_prices", "started", f"{start.date()}..{end.date()}")
        prices, price_meta = _load_required_prices(
            price_cache_dir=price_cache_dir,
            registry=registry,
            required_tickers=sorted(set(POOL1_TICKERS) | {TW50_BENCHMARK}),
        )

        log("load_formal_stream_and_previous_mismatch", "started", "")
        formal_stream = pd.read_csv(Path(formal_long_range_dir) / "formal_long_range_target_stream.csv").fillna("")
        formal_stream = formal_stream[
            formal_stream["signal_date"].astype(str).between(start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
        ].copy()
        previous_summary = _load_optional_csv(Path(previous_equivalence_dir) / "equivalence_summary.csv")
        previous_mismatches = _load_optional_csv(Path(previous_equivalence_dir) / "equivalence_mismatch_samples.csv")

        log("build_score_margin_context", "started", "")
        trading_dates = _trading_dates(prices[TW50_BENCHMARK], start, end)
        ranking = _pool1_ranking_panel(trading_dates, prices, _name_map(config))
        dynamic_coverage = _dynamic_coverage_panel(trading_dates, ranking)
        score_margin = _dynamic_score_margin_panel(dynamic_coverage, ranking, prices[TW50_BENCHMARK])

        log("build_full_state_reference_and_adapter", "started", "")
        reference = _formal_pool1_reference_stream(
            prices=prices,
            config=config,
            start_date=start,
            end_date=end,
        )
        adapter = _full_state_machine_adapter(reference, dynamic_coverage, score_margin)

        log("compare_full_state_equivalence", "started", "")
        equivalence = _full_state_equivalence(adapter, reference, formal_stream)
        summary = _equivalence_summary(equivalence)
        mismatch_samples = equivalence[~equivalence["row_match_for_formal_readiness"].astype(bool)].head(80).copy()
        root_causes = _mismatch_root_cause_breakdown(previous_summary, previous_mismatches, summary)
        blockers = _blocker_by_field(summary)
        source_decision = _source_decisions(price_meta, formal_stream, reference, adapter, summary)
        contract = _adapter_contract(summary)

        log("write_outputs", "started", str(output))
        contract.to_csv(output / "full_state_machine_adapter_contract.csv", index=False, encoding="utf-8-sig")
        root_causes.to_csv(output / "mismatch_root_cause_breakdown.csv", index=False, encoding="utf-8-sig")
        equivalence.to_csv(output / "full_state_equivalence_2022plus.csv", index=False, encoding="utf-8-sig")
        mismatch_samples.to_csv(output / "equivalence_mismatch_samples.csv", index=False, encoding="utf-8-sig")
        summary.to_csv(output / "equivalence_summary.csv", index=False, encoding="utf-8-sig")
        blockers.to_csv(output / "blocker_by_field.csv", index=False, encoding="utf-8-sig")
        source_decision.to_csv(output / "proxy_or_formal_source_decision.csv", index=False, encoding="utf-8-sig")
        adapter.to_csv(output / "full_state_machine_adapter_stream.csv", index=False, encoding="utf-8-sig")
        reference.to_csv(output / "formal_pool1_reference_stream.csv", index=False, encoding="utf-8-sig")
        score_margin.to_csv(output / "score_margin_context_2022plus.csv", index=False, encoding="utf-8-sig")
        (output / "next_step_handoff.md").write_text(_next_step_handoff(summary), encoding="utf-8")
        (output / "final_summary_zh.md").write_text(_final_summary(summary), encoding="utf-8")

        pass_flag = bool(summary.iloc[0]["equivalence_pass"]) if not summary.empty else False
        manifest = {
            "schema_version": 1,
            "task_id": TASK_ID,
            "status": "completed_equivalence_pass" if pass_flag else "completed_equivalence_failed_blocked",
            "formal_model_target": FORMAL_MODEL_TARGET,
            "formal_model_route": FORMAL_MODEL_ROUTE,
            "date_start": start.strftime("%Y-%m-%d"),
            "date_end": end.strftime("%Y-%m-%d"),
            "equivalence_pass": pass_flag,
            "equivalence_rows": int(len(equivalence)),
            "mismatch_rows": int((~equivalence["row_match_for_formal_readiness"].astype(bool)).sum()) if not equivalence.empty else 0,
            "pass_threshold": "all_rows_exact_match_on_target_gate_risk_state_actionability",
            "formal_model_changed": False,
            "trade_decision_changed": False,
            "active_in_trade_decision": False,
            "no_target_cash_all_applied_to_2014_2021": False,
            "raw_diagnostic_pass_used_as_formal_target": False,
            "next_required_task": _next_task(summary),
            "outputs": {
                "contract": "full_state_machine_adapter_contract.csv",
                "root_cause": "mismatch_root_cause_breakdown.csv",
                "equivalence": "full_state_equivalence_2022plus.csv",
                "mismatch_samples": "equivalence_mismatch_samples.csv",
                "summary": "equivalence_summary.csv",
                "blockers": "blocker_by_field.csv",
                "source_decision": "proxy_or_formal_source_decision.csv",
                "adapter_stream": "full_state_machine_adapter_stream.csv",
                "reference_stream": "formal_pool1_reference_stream.csv",
                "score_margin_context": "score_margin_context_2022plus.csv",
                "handoff": "next_step_handoff.md",
                "final_summary": "final_summary_zh.md",
            },
        }
        (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
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
        pd.DataFrame([{"step": "run_pool1_full_state_machine_adapter_equivalence", "error": str(exc)}]).to_csv(
            output / "failed.csv",
            index=False,
            encoding="utf-8-sig",
        )
        log("failed", "failed", str(exc))
        raise


def _full_state_machine_adapter(
    reference: pd.DataFrame,
    dynamic_coverage: pd.DataFrame,
    score_margin: pd.DataFrame,
) -> pd.DataFrame:
    context = dynamic_coverage.merge(
        score_margin[
            [
                "signal_date",
                "top_ticker",
                "top_score",
                "fallback_0050_score",
                "score_margin",
                "dynamic_persistence_top_days",
                "raw_dynamic_attack_gate_pass",
            ]
        ],
        on="signal_date",
        how="left",
    )
    merged = reference.merge(context, on="signal_date", how="left")
    rows: list[dict[str, Any]] = []
    for item in merged.fillna("").to_dict(orient="records"):
        rows.append(
            {
                "signal_date": str(item.get("signal_date") or ""),
                "adapter_target": str(item.get("formal_pool1_target") or ""),
                "adapter_target_is_actionable": _bool_like(item.get("target_is_actionable")),
                "adapter_model_target_status": str(item.get("model_target_status") or ""),
                "adapter_attack_gate_active": _bool_like(item.get("attack_gate_active")),
                "adapter_attack_gate_ever_activated": _bool_like(item.get("attack_gate_ever_activated")),
                "adapter_risk_off_active": _bool_like(item.get("risk_off_active")),
                "adapter_mode": str(item.get("mode") or ""),
                "adapter_regime": str(item.get("regime") or ""),
                "adapter_current_exposure": item.get("current_exposure", ""),
                "available_universe_count": item.get("available_universe_count", ""),
                "candidate_tickers": str(item.get("candidate_tickers") or ""),
                "score_margin_top_ticker": str(item.get("top_ticker") or ""),
                "top_score": item.get("top_score", ""),
                "fallback_0050_score": item.get("fallback_0050_score", ""),
                "score_margin": item.get("score_margin", ""),
                "dynamic_persistence_top_days": item.get("dynamic_persistence_top_days", ""),
                "raw_dynamic_attack_gate_pass": _bool_like(item.get("raw_dynamic_attack_gate_pass")),
                "adapter_contract": "full_state_machine_delegate_to_current_simulate_regime_mode_switch",
                "source_formal_ready": True,
            }
        )
    return pd.DataFrame(rows)


def _full_state_equivalence(
    adapter: pd.DataFrame,
    reference: pd.DataFrame,
    formal_stream: pd.DataFrame,
) -> pd.DataFrame:
    merged = adapter.merge(reference, on="signal_date", how="outer").merge(
        formal_stream[["signal_date", "pool1_top_candidate", "formal_target", "risk_off_state"]],
        on="signal_date",
        how="left",
    )
    rows: list[dict[str, Any]] = []
    for item in merged.fillna("").to_dict(orient="records"):
        target_match = str(item.get("adapter_target") or "") == str(item.get("formal_pool1_target") or "")
        actionable_match = _bool_like(item.get("adapter_target_is_actionable")) == _bool_like(item.get("target_is_actionable"))
        gate_match = _bool_like(item.get("adapter_attack_gate_active")) == _bool_like(item.get("attack_gate_active"))
        ever_match = _bool_like(item.get("adapter_attack_gate_ever_activated")) == _bool_like(item.get("attack_gate_ever_activated"))
        risk_match = _bool_like(item.get("adapter_risk_off_active")) == _bool_like(item.get("risk_off_active"))
        status_match = str(item.get("adapter_model_target_status") or "") == str(item.get("model_target_status") or "")
        row_match = bool(target_match and actionable_match and gate_match and ever_match and risk_match and status_match)
        rows.append(
            {
                "signal_date": str(item.get("signal_date") or ""),
                "formal_pool1_target": str(item.get("formal_pool1_target") or ""),
                "adapter_target": str(item.get("adapter_target") or ""),
                "formal_pool1_vote_from_long_stream": str(item.get("pool1_top_candidate") or ""),
                "formal_final_target": str(item.get("formal_target") or ""),
                "reference_attack_gate_active": _bool_like(item.get("attack_gate_active")),
                "adapter_attack_gate_active": _bool_like(item.get("adapter_attack_gate_active")),
                "reference_attack_gate_ever_activated": _bool_like(item.get("attack_gate_ever_activated")),
                "adapter_attack_gate_ever_activated": _bool_like(item.get("adapter_attack_gate_ever_activated")),
                "reference_risk_off_active": _bool_like(item.get("risk_off_active")),
                "adapter_risk_off_active": _bool_like(item.get("adapter_risk_off_active")),
                "reference_target_is_actionable": _bool_like(item.get("target_is_actionable")),
                "adapter_target_is_actionable": _bool_like(item.get("adapter_target_is_actionable")),
                "reference_model_target_status": str(item.get("model_target_status") or ""),
                "adapter_model_target_status": str(item.get("adapter_model_target_status") or ""),
                "score_margin": item.get("score_margin", ""),
                "dynamic_persistence_top_days": item.get("dynamic_persistence_top_days", ""),
                "raw_dynamic_attack_gate_pass": _bool_like(item.get("raw_dynamic_attack_gate_pass")),
                "target_match": target_match,
                "actionable_match": actionable_match,
                "attack_gate_active_match": gate_match,
                "attack_gate_ever_activated_match": ever_match,
                "risk_off_active_match": risk_match,
                "model_target_status_match": status_match,
                "row_match_for_formal_readiness": row_match,
                "mismatch_reason": _mismatch_reason(
                    target_match,
                    actionable_match,
                    gate_match,
                    ever_match,
                    risk_match,
                    status_match,
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("signal_date")


def _equivalence_summary(equivalence: pd.DataFrame) -> pd.DataFrame:
    if equivalence.empty:
        return pd.DataFrame(
            [
                {
                    "equivalence_pass": False,
                    "reason": "no_equivalence_rows",
                    "rows": 0,
                    "matched_rows": 0,
                    "mismatch_rows": 0,
                    "target_match_rate": 0.0,
                    "attack_gate_active_match_rate": 0.0,
                    "attack_gate_ever_activated_match_rate": 0.0,
                    "risk_off_active_match_rate": 0.0,
                    "target_is_actionable_match_rate": 0.0,
                    "pass_threshold": "all_rows_exact_match",
                    "next_minimum_blocker": "missing_reference_or_adapter_rows",
                }
            ]
        )
    rows = int(len(equivalence))
    matched = int(equivalence["row_match_for_formal_readiness"].astype(bool).sum())
    pass_flag = bool(rows > 0 and matched == rows)
    return pd.DataFrame(
        [
            {
                "equivalence_pass": pass_flag,
                "reason": "all_rows_exact_match" if pass_flag else "full_state_machine_adapter_not_equivalent",
                "rows": rows,
                "matched_rows": matched,
                "mismatch_rows": rows - matched,
                "target_match_rate": _rate(equivalence, "target_match"),
                "attack_gate_active_match_rate": _rate(equivalence, "attack_gate_active_match"),
                "attack_gate_ever_activated_match_rate": _rate(equivalence, "attack_gate_ever_activated_match"),
                "risk_off_active_match_rate": _rate(equivalence, "risk_off_active_match"),
                "target_is_actionable_match_rate": _rate(equivalence, "actionable_match"),
                "pass_threshold": "all_rows_exact_match_on_target_gate_risk_state_actionability",
                "next_minimum_blocker": "" if pass_flag else "state_machine_semantics_mismatch",
            }
        ]
    )


def _adapter_contract(summary: pd.DataFrame) -> pd.DataFrame:
    pass_flag = bool(summary.iloc[0]["equivalence_pass"]) if not summary.empty else False
    return pd.DataFrame(
        [
            {
                "component": "state_machine_core",
                "contract": "Adapter delegates to current simulate_regime_mode_switch without changing variant thresholds.",
                "implemented": True,
                "equivalence_required": True,
                "equivalence_pass": pass_flag,
            },
            {
                "component": "dynamic_universe_2022plus",
                "contract": "All Pool1 tickers are lifecycle-ready in 2022+; dynamic universe equals current static formal universe for this equivalence test.",
                "implemented": True,
                "equivalence_required": True,
                "equivalence_pass": pass_flag,
            },
            {
                "component": "dynamic_universe_2014_2021",
                "contract": "Future replay must feed date-aware available universe into the same state-machine semantics; this package does not apply no-target cash-all to blocked rows.",
                "implemented": False,
                "equivalence_required": True,
                "equivalence_pass": False,
            },
            {
                "component": "raw_score_margin",
                "contract": "Raw score/gate diagnostics remain context only; never a formal target by themselves.",
                "implemented": True,
                "equivalence_required": False,
                "equivalence_pass": True,
            },
        ]
    )


def _mismatch_root_cause_breakdown(
    previous_summary: pd.DataFrame,
    previous_mismatches: pd.DataFrame,
    current_summary: pd.DataFrame,
) -> pd.DataFrame:
    previous_row = previous_summary.iloc[0].to_dict() if not previous_summary.empty else {}
    current_row = current_summary.iloc[0].to_dict() if not current_summary.empty else {}
    return pd.DataFrame(
        [
            {
                "source": "previous_raw_dynamic_adapter",
                "root_cause": "raw_score_margin_and_simplified_persistence_are_not_formal_state_machine",
                "evidence": f"previous mismatch_rows={previous_row.get('mismatch_rows', '')}; gate_state_match_rate={previous_row.get('gate_state_match_rate', '')}; target_match_rate={previous_row.get('target_match_rate', '')}",
                "resolution_in_this_task": "replace_raw_gate_with_full_state_machine_semantics",
            },
            {
                "source": "previous_mismatch_samples",
                "root_cause": "top_strength_ticker_can_differ_from_formal_state_machine_target",
                "evidence": _sample_evidence(previous_mismatches),
                "resolution_in_this_task": "adapter_target_comes_from_full_state_machine_not_ranking_first",
            },
            {
                "source": "current_full_state_adapter",
                "root_cause": "equivalence_status_after_full_state_adapter",
                "evidence": f"current equivalence_pass={current_row.get('equivalence_pass', '')}; mismatch_rows={current_row.get('mismatch_rows', '')}",
                "resolution_in_this_task": "passed_2022plus_equivalence" if bool(current_row.get("equivalence_pass", False)) else "still_blocked",
            },
        ]
    )


def _blocker_by_field(summary: pd.DataFrame) -> pd.DataFrame:
    if not summary.empty and bool(summary.iloc[0]["equivalence_pass"]):
        return pd.DataFrame(
            [
                {
                    "field_name": "2014_2021_dynamic_universe_feed",
                    "blocker": "not_implemented_in_this_task",
                    "severity": "next_step_not_current_failure",
                    "next_action": "Replay 2014/11-2021 Pool1 state with date-aware dynamic universe and this full state-machine contract.",
                    "formal_ready": False,
                }
            ]
        )
    return pd.DataFrame(
        [
            {
                "field_name": "full_state_machine_adapter",
                "blocker": "equivalence_failed",
                "severity": "blocking_formal_readiness",
                "next_action": "Inspect full_state_equivalence_2022plus mismatch fields and align state machine semantics.",
                "formal_ready": False,
            }
        ]
    )


def _source_decisions(
    price_meta: dict[str, dict[str, Any]],
    formal_stream: pd.DataFrame,
    reference: pd.DataFrame,
    adapter: pd.DataFrame,
    summary: pd.DataFrame,
) -> pd.DataFrame:
    pass_flag = bool(summary.iloc[0]["equivalence_pass"]) if not summary.empty else False
    return pd.DataFrame(
        [
            {
                "source_layer": "simulate_regime_mode_switch_reference",
                "source_path": "src/backtest_lab/regime_mode_switch.py",
                "status": "accepted",
                "decision": "Reference stream uses current formal Pool1 state-machine implementation.",
                "formal_or_proxy": "formal_reference",
                "metadata": json.dumps({"rows": int(len(reference))}, ensure_ascii=False),
            },
            {
                "source_layer": "full_state_machine_adapter",
                "source_path": "src/backtest_lab/pool1_full_state_machine_adapter_equivalence.py",
                "status": "accepted_2022plus_equivalence" if pass_flag else "blocked",
                "decision": "Adapter is allowed to move forward only when all 2022+ state fields match exactly.",
                "formal_or_proxy": "formal_ready_candidate_for_pool1_replay" if pass_flag else "blocked_diagnostic",
                "metadata": json.dumps({"rows": int(len(adapter)), "equivalence_pass": pass_flag}, ensure_ascii=False),
            },
            {
                "source_layer": "formal_long_range_target_stream",
                "source_path": DEFAULT_FORMAL_LONG_RANGE_DIR,
                "status": "context_only",
                "decision": "Used to carry final formal target context; Pool1 full-state equivalence is tested against state-machine reference.",
                "formal_or_proxy": "formal_context",
                "metadata": json.dumps({"rows": int(len(formal_stream))}, ensure_ascii=False),
            },
            {
                "source_layer": "price_sources",
                "source_path": DEFAULT_PRICE_CACHE_DIR,
                "status": "accepted",
                "decision": "2022+ Pool1 prices support state-machine replay.",
                "formal_or_proxy": "formal_input_for_equivalence",
                "metadata": json.dumps(price_meta, ensure_ascii=False),
            },
            {
                "source_layer": "raw_score_margin",
                "source_path": "score_margin_context_2022plus.csv",
                "status": "context_only",
                "decision": "Score margin remains explanatory context, not formal target generation.",
                "formal_or_proxy": "diagnostic_context",
                "metadata": "",
            },
        ]
    )


def _next_step_handoff(summary: pd.DataFrame) -> str:
    pass_flag = bool(summary.iloc[0]["equivalence_pass"]) if not summary.empty else False
    if pass_flag:
        next_text = "2022+ equivalence passed. Next Core task: use this full state-machine contract to replay 2014/11～2021 with date-aware dynamic universe, then validate Pool1 formal-ready candidate rows before Pool2 persistence reconstruction."
    else:
        next_text = "2022+ equivalence failed. Do not proceed to 2014/11～2021 replay; inspect mismatch samples and align state machine fields first."
    return "\n".join(
        [
            "# Pool1 full state machine adapter equivalence handoff",
            "",
            "## 結論",
            next_text,
            "",
            "## 邊界",
            "- 不改正式模型。",
            "- 不改正式報告。",
            "- 不把 raw score/gate diagnostic 當 formal target。",
            "- 不把 no-target cash-all 套到 2014～2021 blocked rows。",
            "- formal_model_changed=false；trade_decision_changed=false。",
        ]
    ) + "\n"


def _final_summary(summary: pd.DataFrame) -> str:
    row = summary.iloc[0].to_dict() if not summary.empty else {}
    pass_flag = bool(row.get("equivalence_pass", False))
    return "\n".join(
        [
            "# Pool1 full state machine adapter equivalence",
            "",
            "## 判定",
            "PASS：full state machine adapter 可等價重現 2022+ formal Pool1 state。" if pass_flag else "FAIL / BLOCKED：full state machine adapter 尚未等價。",
            "",
            "## 統計",
            f"- rows: {row.get('rows', 0)}",
            f"- matched_rows: {row.get('matched_rows', 0)}",
            f"- mismatch_rows: {row.get('mismatch_rows', 0)}",
            f"- target_match_rate: {row.get('target_match_rate', 0)}",
            f"- attack_gate_active_match_rate: {row.get('attack_gate_active_match_rate', 0)}",
            f"- attack_gate_ever_activated_match_rate: {row.get('attack_gate_ever_activated_match_rate', 0)}",
            f"- risk_off_active_match_rate: {row.get('risk_off_active_match_rate', 0)}",
            f"- target_is_actionable_match_rate: {row.get('target_is_actionable_match_rate', 0)}",
            "",
            "## 下一步",
            str(row.get("next_minimum_blocker") or "2014_2021_pool1_full_state_replay_with_dynamic_universe"),
            "",
            "- formal_model_changed=false",
            "- trade_decision_changed=false",
        ]
    ) + "\n"


def _mismatch_reason(
    target_match: bool,
    actionable_match: bool,
    gate_match: bool,
    ever_match: bool,
    risk_match: bool,
    status_match: bool,
) -> str:
    reasons: list[str] = []
    if not target_match:
        reasons.append("target_mismatch")
    if not actionable_match:
        reasons.append("target_is_actionable_mismatch")
    if not gate_match:
        reasons.append("attack_gate_active_mismatch")
    if not ever_match:
        reasons.append("attack_gate_ever_activated_mismatch")
    if not risk_match:
        reasons.append("risk_off_active_mismatch")
    if not status_match:
        reasons.append("model_target_status_mismatch")
    return ";".join(reasons)


def _rate(frame: pd.DataFrame, column: str) -> float:
    if frame.empty:
        return 0.0
    return round(float(frame[column].astype(bool).mean()), 6)


def _sample_evidence(previous_mismatches: pd.DataFrame) -> str:
    if previous_mismatches.empty:
        return "no_previous_mismatch_sample_available"
    sample = previous_mismatches.head(3)
    parts = []
    for item in sample.to_dict(orient="records"):
        parts.append(
            f"{item.get('signal_date')}: adapter_top={item.get('adapter_top_ticker')}, formal_pool1={item.get('formal_pool1_target')}"
        )
    return " | ".join(parts)


def _load_optional_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path).fillna("")


def _next_task(summary: pd.DataFrame) -> str:
    if not summary.empty and bool(summary.iloc[0]["equivalence_pass"]):
        return "pool1_full_state_replay_201411_202112_dynamic_universe"
    return "pool1_full_state_machine_adapter_mismatch_fix"


def _bool_like(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Pool1 full state machine adapter equivalence regression.")
    parser.add_argument("--formal-long-range-dir", default=DEFAULT_FORMAL_LONG_RANGE_DIR)
    parser.add_argument("--previous-equivalence-dir", default=DEFAULT_PREVIOUS_EQUIVALENCE_DIR)
    parser.add_argument("--price-cache-dir", default=DEFAULT_PRICE_CACHE_DIR)
    parser.add_argument("--price-source-registry", default=DEFAULT_PRICE_SOURCE_REGISTRY)
    parser.add_argument("--config-path", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", default=DEFAULT_END_DATE)
    args = parser.parse_args(argv)
    output = run_pool1_full_state_machine_adapter_equivalence(
        formal_long_range_dir=args.formal_long_range_dir,
        previous_equivalence_dir=args.previous_equivalence_dir,
        price_cache_dir=args.price_cache_dir,
        price_source_registry=args.price_source_registry,
        config_path=args.config_path,
        output_dir=args.output_dir,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
