from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from backtest_lab.formal_model_contract import FORMAL_MODEL_ROUTE, FORMAL_MODEL_TARGET
from backtest_lab.regime_mode_switch import frozen_cycle_proven_top1_v1_variant


TASK_ID = "TASK-BACKTEST-CORE-POOL1-ATTACK-GATE-STATE-REPLAY-201411-20260702"
DEFAULT_PANEL_DIR = "outputs/current_formal_pool1_pool2_signal_panels_201411_202112_20260630"
DEFAULT_LIFECYCLE_DIR = "outputs/pool1_ticker_lifecycle_contract_201411_202112_20260702"
DEFAULT_OUTPUT_DIR = "outputs/pool1_attack_gate_state_replay_201411_202112_20260702"


def run_pool1_attack_gate_state_replay(
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

        log("build_replay_contract", "started", "")
        contract = _state_replay_contract()
        coverage = _state_replay_coverage(readiness, ranking, lifecycle, availability)
        blocked = _blocked_signal_rows(coverage)
        blockers = _blocker_by_field(lifecycle)
        source_decisions = _source_decisions()

        log("write_outputs", "started", "")
        contract.to_csv(output / "pool1_attack_gate_state_replay_contract.csv", index=False, encoding="utf-8-sig")
        coverage.to_csv(output / "pool1_attack_gate_state_replay_coverage.csv", index=False, encoding="utf-8-sig")
        blocked.to_csv(output / "blocked_signal_rows.csv", index=False, encoding="utf-8-sig")
        blockers.to_csv(output / "blocker_by_field.csv", index=False, encoding="utf-8-sig")
        source_decisions.to_csv(output / "proxy_or_formal_source_decision.csv", index=False, encoding="utf-8-sig")
        (output / "next_step_handoff.md").write_text(_next_step_handoff(), encoding="utf-8")
        (output / "final_summary_zh.md").write_text(_final_summary(panel_manifest, coverage, blockers), encoding="utf-8")

        static_ready_date = _static_all_ticker_scoring_ready_date(lifecycle)
        output_manifest = {
            "schema_version": 1,
            "task_id": TASK_ID,
            "status": "completed_blocked_state_replay_contract",
            "formal_model_target": FORMAL_MODEL_TARGET,
            "formal_model_route": FORMAL_MODEL_ROUTE,
            "date_start": str(panel_manifest.get("date_start") or _first_date(readiness)),
            "date_end": str(panel_manifest.get("date_end") or _last_date(readiness)),
            "pool1_attack_gate_state_formal_ready": False,
            "formal_vote_ready_days": 0,
            "blocked_signal_rows": int(len(blocked)),
            "static_all_ticker_scoring_ready_date": static_ready_date,
            "static_universe_simulator_supports_2014_start": False,
            "dynamic_universe_state_replay_required": True,
            "pool1_lifecycle_contract_ready": True,
            "pool1_ranking_panel_available": True,
            "no_target_cash_all_applied": False,
            "proxy_used_as_formal": False,
            "uses_forward_return_as_rule": False,
            "formal_model_changed": False,
            "trade_decision_changed": False,
            "active_in_trade_decision": False,
            "remaining_blocker_after_state_replay": "dynamic_universe_state_replay_adapter_or_midstream_initialization_contract",
            "outputs": {
                "contract": "pool1_attack_gate_state_replay_contract.csv",
                "coverage": "pool1_attack_gate_state_replay_coverage.csv",
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
        pd.DataFrame([{"step": "run_pool1_attack_gate_state_replay", "error": str(exc)}]).to_csv(
            output / "failed.csv",
            index=False,
            encoding="utf-8-sig",
        )
        log("failed", "failed", str(exc))
        raise


def _state_replay_contract() -> pd.DataFrame:
    variant = frozen_cycle_proven_top1_v1_variant()
    rows = [
        _contract_row(
            "prices_by_ticker",
            "simulate_regime_mode_switch input",
            "all Pool1 fixed universe price frames plus 0050 benchmark",
            "partial",
            False,
            "current simulator expects static common-date universe; 6669 starts later than 2014",
        ),
        _contract_row(
            "asset_types",
            "ep05 universe config / report pool",
            "ETF vs stock classification required for costs and target type",
            "available",
            True,
            "",
        ),
        _contract_row(
            "market_prices",
            "0050.TW price frame",
            "market regime classification and fallback defense rules",
            "available",
            True,
            "",
        ),
        _contract_row(
            "relative_strength_scores",
            "Pool1 ranking panel / strategies.relative_strength_scores",
            "20/60 relative strength score for date-aware available candidates",
            "available",
            True,
            "",
        ),
        _contract_row(
            "attack_gate_active",
            "simulate_regime_mode_switch equity_curve.attack_gate_active",
            f"margin={variant.attack_gate_margin_over_fallback}; fallback={variant.attack_gate_fallback_ticker}; confirmation={variant.attack_gate_activation_confirmation_days}",
            "missing_formal_state",
            False,
            "requires dynamic-universe state replay from 2014 start",
        ),
        _contract_row(
            "attack_gate_ever_activated",
            "simulate_regime_mode_switch equity_curve.attack_gate_ever_activated",
            f"initialize_history_days={variant.attack_gate_initialize_history_days}; initialize_from_history={variant.attack_gate_initialize_active_from_history}",
            "missing_formal_state",
            False,
            "requires continuous attack gate history before and after delayed ticker availability",
        ),
        _contract_row(
            "risk_off_active",
            "simulate_regime_mode_switch equity_curve.risk_off_active",
            f"market_risk_off_filter={variant.market_risk_off_filter}",
            "missing_formal_state",
            False,
            "depends on gate interaction and regime state in state machine",
        ),
        _contract_row(
            "target_is_actionable",
            "FrozenStrategySignal.target_is_actionable",
            "target_ticker != cash and target_exposure > 0",
            "missing_formal_state",
            False,
            "requires target output from formal state machine",
        ),
        _contract_row(
            "model_target_status",
            "FrozenStrategySignal.model_target_status",
            "formal target vs no target/risk state display contract",
            "missing_formal_state",
            False,
            "requires target output from formal state machine",
        ),
    ]
    return pd.DataFrame(rows)


def _contract_row(
    field_name: str,
    source: str,
    condition: str,
    availability: str,
    formal_ready: bool,
    blocker: str,
) -> dict[str, Any]:
    return {
        "field_name": field_name,
        "formal_2022_source": source,
        "formal_2022_condition": condition,
        "2014_2021_availability": availability,
        "formal_ready": formal_ready,
        "blocker": blocker,
    }


def _state_replay_coverage(
    readiness: pd.DataFrame,
    ranking: pd.DataFrame,
    lifecycle: pd.DataFrame,
    availability: pd.DataFrame,
) -> pd.DataFrame:
    ranking_counts = ranking.groupby("date").size().to_dict() if not ranking.empty else {}
    top_rank = _top_rank_by_date(ranking)
    available_counts = (
        availability[availability["candidate_available_for_pool1_ranking"].map(_bool_like)]
        .groupby("date")
        .size()
        .to_dict()
        if not availability.empty
        else {}
    )
    static_ready_date = _static_all_ticker_scoring_ready_date(lifecycle)
    rows: list[dict[str, Any]] = []
    for item in readiness.to_dict(orient="records"):
        date = str(item.get("date") or "")
        top = top_rank.get(date, {})
        before_static_ready = bool(static_ready_date and date < static_ready_date)
        rows.append(
            {
                "date": date,
                "available_candidate_count": int(available_counts.get(date, 0)),
                "ranking_candidate_count": int(ranking_counts.get(date, 0)),
                "pool1_top_candidate": str(top.get("candidate_ticker") or ""),
                "pool1_top_candidate_score": _clean_text(top.get("score")),
                "date_aware_lifecycle_ready": True,
                "static_all_ticker_scoring_ready": bool(static_ready_date and date >= static_ready_date),
                "current_state_machine_supports_dynamic_universe": False,
                "attack_gate_active_ready": False,
                "attack_gate_ever_activated_ready": False,
                "risk_off_active_ready": False,
                "target_is_actionable_ready": False,
                "model_target_status_ready": False,
                "pool1_formal_vote_ready": False,
                "replay_status": "blocked_before_static_universe_ready"
                if before_static_ready
                else "blocked_dynamic_universe_state_continuity_not_defined",
                "blocked_reason": "現行 simulator 使用 static common-date universe；2014 起跑時 6669 尚無價格/未滿 warmup，不能用排名第一替代 state machine。"
                if before_static_ready
                else "雖然所有 ticker 已可 scoring，但缺從 2014 起的 dynamic-universe attack gate state continuity contract，不能 midstream 宣稱 formal-ready。",
            }
        )
    return pd.DataFrame(rows)


def _blocked_signal_rows(coverage: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for item in coverage.to_dict(orient="records"):
        rows.append(
            {
                "signal_date": item["date"],
                "pool1_target": "",
                "pool1_target_weights": "{}",
                "attack_gate_active": "",
                "attack_gate_ever_activated": "",
                "risk_off_active": "",
                "target_is_actionable": "",
                "model_target_status": "",
                "reason": item["blocked_reason"],
                "source_formal_ready": False,
                "no_target_cash_all_applied": False,
            }
        )
    return pd.DataFrame(rows)


def _blocker_by_field(lifecycle: pd.DataFrame) -> pd.DataFrame:
    static_ready_date = _static_all_ticker_scoring_ready_date(lifecycle)
    return pd.DataFrame(
        [
            {
                "field_name": "prices_by_ticker",
                "blocker": "static_common_date_universe_blocks_2014_start",
                "severity": "blocking_full_2014_2021_state_replay",
                "current_status": f"all fixed Pool1 tickers scoring-ready from {static_ready_date}; earlier dates require dynamic universe",
                "next_action": "Build dynamic-universe replay adapter or define an accepted midstream initialization contract.",
                "formal_ready": False,
            },
            {
                "field_name": "attack_gate_active",
                "blocker": "missing_dynamic_universe_attack_gate_state_replay",
                "severity": "blocking_pool1_formal_vote",
                "current_status": "ranking and lifecycle are available, but formal gate state is not replayed",
                "next_action": "Replay current variant state with date-aware eligible universe, preserving fallback and persistence rules.",
                "formal_ready": False,
            },
            {
                "field_name": "attack_gate_ever_activated",
                "blocker": "missing_state_continuity_from_2014_start",
                "severity": "blocking_pool1_formal_vote",
                "current_status": "cannot infer from daily rank alone",
                "next_action": "Persist gate history across dynamic ticker availability changes.",
                "formal_ready": False,
            },
            {
                "field_name": "risk_off_active",
                "blocker": "missing_regime_mode_switch_state_replay",
                "severity": "blocking_pool1_formal_vote",
                "current_status": "depends on market risk filter and gate interaction",
                "next_action": "Replay state machine after dynamic universe contract is implemented.",
                "formal_ready": False,
            },
            {
                "field_name": "target_is_actionable",
                "blocker": "missing_formal_target_output",
                "severity": "blocking_pool1_formal_vote",
                "current_status": "depends on state machine target/exposure output",
                "next_action": "Derive only after attack gate state replay produces target_ticker and target_exposure.",
                "formal_ready": False,
            },
            {
                "field_name": "model_target_status",
                "blocker": "missing_formal_target_output",
                "severity": "blocking_pool1_formal_vote",
                "current_status": "depends on state machine target/exposure output",
                "next_action": "Derive only after attack gate state replay produces target_ticker and target_exposure.",
                "formal_ready": False,
            },
        ]
    )


def _source_decisions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "source_layer": "pool1_ranking_panel",
                "source_path": DEFAULT_PANEL_DIR + "/pool1_daily_candidate_ranking_panel.csv",
                "status": "accepted_as_input",
                "formal_or_proxy": "formal_input_not_formal_output",
                "decision": "可作候選排序輸入，不可替代 attack gate state。",
            },
            {
                "source_layer": "ticker_lifecycle_contract",
                "source_path": DEFAULT_LIFECYCLE_DIR,
                "status": "accepted_as_input",
                "formal_or_proxy": "formal_availability_input",
                "decision": "可決定 ticker 當日是否可進 Pool1 ranking。",
            },
            {
                "source_layer": "simulate_regime_mode_switch",
                "source_path": "src/backtest_lab/regime_mode_switch.py",
                "status": "blocked_for_2014_dynamic_universe",
                "formal_or_proxy": "formal_engine_needs_adapter",
                "decision": "現行函式使用 static prices_by_ticker common-date universe；不能直接從 2014 重放含 6669 晚上市的 Pool1。",
            },
            {
                "source_layer": "ranking_first_proxy",
                "source_path": "",
                "status": "rejected",
                "formal_or_proxy": "proxy_rejected",
                "decision": "排名第一不得包裝成 formal target 或 attack gate active。",
            },
        ]
    )


def _next_step_handoff() -> str:
    return "\n".join(
        [
            "# Pool1 attack gate state replay handoff",
            "",
            "## 判定",
            "2014/11～2021 不能直接用現行 `simulate_regime_mode_switch` formal-ready 重放 Pool1 attack gate state。",
            "",
            "## 原因",
            "現行 simulator 以 static `prices_by_ticker` common trade dates 起跑；Pool1 固定 universe 中 6669 到 2017-11-13 才有價格，2018-02-06 才滿 60 日 scoring warmup。若直接放入全部 ticker，2014～2017 會被 common-date / lifecycle 卡住；若排除 6669，又不是現行固定 Pool1 universe。",
            "",
            "## 下一步",
            "Core 需新增一個 date-aware dynamic-universe state replay adapter，仍沿用現行 variant 參數、fallback、persistence、risk filter，但每日只把 lifecycle contract 判定可 scoring 的 ticker 送入 attack candidate scoring。完成後才能產 Pool1 formal vote candidate rows。",
            "",
            "## 禁止",
            "- 不得用 ranking first 取代 formal target。",
            "- 不得在 2014～2021 blocked rows 套 no-target cash-all。",
            "- 不得把 2018 後 static-universe midstream replay 直接當完整 2014 formal stream。",
        ]
    ) + "\n"


def _final_summary(panel_manifest: dict[str, Any], coverage: pd.DataFrame, blockers: pd.DataFrame) -> str:
    static_ready = coverage.loc[coverage["static_all_ticker_scoring_ready"].map(_bool_like), "date"]
    static_ready_date = str(static_ready.iloc[0]) if not static_ready.empty else ""
    return "\n".join(
        [
            "# Pool1 attack gate state replay",
            "",
            "## 判定",
            "Pool1 attack gate state 尚不能 formal-ready 重放 2014/11～2021。",
            "Pool1 ranking 與 ticker lifecycle 已 ready，但現行 `simulate_regime_mode_switch` 是 static universe 狀態機，不能直接處理 6669 晚上市/晚滿 warmup 的 dynamic candidate universe。",
            "",
            "## 本批輸出",
            f"- 來源區間：{panel_manifest.get('date_start')}～{panel_manifest.get('date_end')}",
            f"- blocked signal rows：{len(coverage)}",
            f"- static all-ticker scoring ready date：{static_ready_date}",
            f"- field blockers：{len(blockers)}",
            "",
            "## 結論",
            "下一棒應補 `date_aware_dynamic_universe_state_replay_adapter`，不是交 Experiments 跑績效。",
            "",
            "- formal_model_changed=false",
            "- trade_decision_changed=false",
            "- no_target_cash_all_applied=false",
        ]
    ) + "\n"


def _top_rank_by_date(ranking: pd.DataFrame) -> dict[str, dict[str, Any]]:
    if ranking.empty:
        return {}
    frame = ranking.copy()
    frame["rank_num"] = pd.to_numeric(frame.get("rank", ""), errors="coerce")
    top = frame.sort_values(["date", "rank_num", "candidate_ticker"]).groupby("date", as_index=False).first()
    return {str(item.get("date") or ""): item for item in top.to_dict(orient="records")}


def _static_all_ticker_scoring_ready_date(lifecycle: pd.DataFrame) -> str:
    dates = pd.to_datetime(lifecycle["first_pool1_scoring_date"], errors="coerce").dropna()
    if dates.empty:
        return ""
    return pd.Timestamp(dates.max()).strftime("%Y-%m-%d")


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
    parser = argparse.ArgumentParser(description="Build Pool1 attack gate state replay contract package.")
    parser.add_argument("--panel-dir", default=DEFAULT_PANEL_DIR)
    parser.add_argument("--lifecycle-dir", default=DEFAULT_LIFECYCLE_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    output = run_pool1_attack_gate_state_replay(
        panel_dir=args.panel_dir,
        lifecycle_dir=args.lifecycle_dir,
        output_dir=args.output_dir,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
