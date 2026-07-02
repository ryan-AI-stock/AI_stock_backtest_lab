from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from backtest_lab.config import load_config
from backtest_lab.current_formal_pool1_pool2_signal_panels import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_PRICE_CACHE_DIR,
    DEFAULT_PRICE_SOURCE_REGISTRY,
    POOL1_TICKERS,
    TW50_BENCHMARK,
    _load_price_source_registry,
)
from backtest_lab.formal_model_contract import FORMAL_MODEL_ROUTE, FORMAL_MODEL_TARGET
from backtest_lab.pool1_dynamic_adapter_2022_equivalence import _load_required_prices
from backtest_lab.pool1_full_state_machine_adapter_equivalence import _full_state_machine_adapter
from backtest_lab.pool1_dynamic_score_margin_state_adapter import _dynamic_score_margin_panel
from backtest_lab.pool1_dynamic_adapter_2022_equivalence import _dynamic_coverage_panel, _pool1_ranking_panel
from backtest_lab.pool1_dynamic_adapter_2022_equivalence import _formal_pool1_reference_stream
from backtest_lab.current_formal_pool1_pool2_signal_panels import _name_map


TASK_ID = "TASK-BACKTEST-CORE-POOL1-FULL-STATE-REPLAY-201411-202112-DYNAMIC-UNIVERSE-20260702"
DEFAULT_DYNAMIC_UNIVERSE_DIR = "outputs/date_aware_dynamic_universe_state_replay_201411_202112_20260702"
DEFAULT_SCORE_MARGIN_DIR = "outputs/pool1_dynamic_score_margin_state_adapter_201411_202112_20260702"
DEFAULT_EQUIVALENCE_DIR = "outputs/pool1_full_state_machine_adapter_equivalence_20260702"
DEFAULT_OUTPUT_DIR = "outputs/pool1_full_state_replay_201411_202112_dynamic_universe_20260702"
DEFAULT_START_DATE = "2014-11-03"
DEFAULT_END_DATE = "2021-12-31"


def run_pool1_full_state_replay_201411_dynamic_universe(
    *,
    dynamic_universe_dir: str | Path = DEFAULT_DYNAMIC_UNIVERSE_DIR,
    score_margin_dir: str | Path = DEFAULT_SCORE_MARGIN_DIR,
    equivalence_dir: str | Path = DEFAULT_EQUIVALENCE_DIR,
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

        log("load_contract_inputs", "started", "")
        dynamic_coverage = pd.read_csv(Path(dynamic_universe_dir) / "dynamic_universe_state_replay_coverage.csv").fillna("")
        dynamic_coverage = dynamic_coverage[
            dynamic_coverage["signal_date"].astype(str).between(start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
        ].copy()
        prior_score_margin = pd.read_csv(Path(score_margin_dir) / "dynamic_score_margin_panel.csv").fillna("")
        equivalence_manifest = _load_json(Path(equivalence_dir) / "manifest.json")

        if not bool(equivalence_manifest.get("equivalence_pass")):
            raise ValueError("Full state machine equivalence must pass before this replay can run.")

        log("find_static_ready_segment", "started", "")
        static_ready_start = _first_static_all_ticker_date(dynamic_coverage)
        blocked = _blocked_dynamic_rows(dynamic_coverage, static_ready_start)

        log("load_prices", "started", "")
        prices, price_meta = _load_required_prices(
            price_cache_dir=price_cache_dir,
            registry=registry,
            required_tickers=sorted(set(POOL1_TICKERS) | {TW50_BENCHMARK}),
        )

        log("replay_static_ready_segment", "started", static_ready_start)
        replayed, score_context = _replay_static_ready_segment(
            prices=prices,
            config=config,
            static_ready_start=static_ready_start,
            end=end,
        )
        replayed = _attach_dynamic_context(replayed, dynamic_coverage, score_context)

        log("build_outputs", "started", "")
        coverage = _coverage_summary(dynamic_coverage, blocked, replayed, static_ready_start)
        contract = _replay_contract(static_ready_start)
        blockers = _blocker_by_field(static_ready_start, blocked, replayed)
        source_decision = _source_decisions(price_meta, equivalence_manifest, static_ready_start, blocked, replayed)

        contract.to_csv(output / "pool1_full_state_replay_contract.csv", index=False, encoding="utf-8-sig")
        coverage.to_csv(output / "pool1_full_state_replay_coverage.csv", index=False, encoding="utf-8-sig")
        replayed.to_csv(output / "pool1_full_state_replayed_signals.csv", index=False, encoding="utf-8-sig")
        blocked.to_csv(output / "blocked_signal_rows.csv", index=False, encoding="utf-8-sig")
        blockers.to_csv(output / "blocker_by_field.csv", index=False, encoding="utf-8-sig")
        source_decision.to_csv(output / "proxy_or_formal_source_decision.csv", index=False, encoding="utf-8-sig")
        prior_score_margin.to_csv(output / "dynamic_score_margin_context_201411_202112.csv", index=False, encoding="utf-8-sig")
        (output / "next_step_handoff.md").write_text(_next_step_handoff(static_ready_start), encoding="utf-8")
        (output / "final_summary_zh.md").write_text(_final_summary(coverage, blockers), encoding="utf-8")

        manifest = {
            "schema_version": 1,
            "task_id": TASK_ID,
            "status": "completed_partial_pool1_full_state_replay",
            "formal_model_target": FORMAL_MODEL_TARGET,
            "formal_model_route": FORMAL_MODEL_ROUTE,
            "date_start": start.strftime("%Y-%m-%d"),
            "date_end": end.strftime("%Y-%m-%d"),
            "full_state_equivalence_2022plus_pass": True,
            "static_all_ticker_replay_start": static_ready_start,
            "formal_ready_pool1_rows": int(len(replayed)),
            "blocked_signal_rows": int(len(blocked)),
            "pool1_full_state_replay_formal_ready_full_period": False,
            "pool1_full_state_replay_formal_ready_static_segment": bool(not replayed.empty),
            "dynamic_universe_pre_static_segment_blocked": bool(not blocked.empty),
            "formal_model_changed": False,
            "trade_decision_changed": False,
            "active_in_trade_decision": False,
            "no_target_cash_all_applied_to_2014_2021": False,
            "raw_diagnostic_pass_used_as_formal_target": False,
            "next_required_task": "pool2_persistence_full_reconstruction" if not replayed.empty else "dynamic_universe_state_injection_api",
            "outputs": {
                "contract": "pool1_full_state_replay_contract.csv",
                "coverage": "pool1_full_state_replay_coverage.csv",
                "replayed_signals": "pool1_full_state_replayed_signals.csv",
                "blocked_rows": "blocked_signal_rows.csv",
                "blockers": "blocker_by_field.csv",
                "source_decision": "proxy_or_formal_source_decision.csv",
                "score_context": "dynamic_score_margin_context_201411_202112.csv",
                "handoff": "next_step_handoff.md",
                "summary": "final_summary_zh.md",
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
        pd.DataFrame([{"step": "run_pool1_full_state_replay_201411_dynamic_universe", "error": str(exc)}]).to_csv(
            output / "failed.csv",
            index=False,
            encoding="utf-8-sig",
        )
        log("failed", "failed", str(exc))
        raise


def _first_static_all_ticker_date(dynamic_coverage: pd.DataFrame) -> str:
    frame = dynamic_coverage.copy()
    frame["available_universe_count_num"] = pd.to_numeric(frame["available_universe_count"], errors="coerce")
    eligible = frame[frame["available_universe_count_num"].eq(len(POOL1_TICKERS))]
    if eligible.empty:
        return ""
    return str(eligible.sort_values("signal_date").iloc[0]["signal_date"])


def _blocked_dynamic_rows(dynamic_coverage: pd.DataFrame, static_ready_start: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for item in dynamic_coverage.to_dict(orient="records"):
        signal_date = str(item.get("signal_date") or "")
        if static_ready_start and signal_date >= static_ready_start:
            continue
        rows.append(
            {
                "signal_date": signal_date,
                "available_universe_count": item.get("available_universe_count", ""),
                "candidate_tickers": str(item.get("candidate_tickers") or ""),
                "pool1_target": "",
                "pool1_target_weights": "{}",
                "attack_gate_active": "",
                "attack_gate_ever_activated": "",
                "risk_off_active": "",
                "target_is_actionable": "",
                "model_target_status": "",
                "reason": "Blocked: current simulate_regime_mode_switch requires a static common-date universe and has no API to inject/carry dynamic-universe state before all Pool1 tickers are scoring-ready.",
                "source_formal_ready": False,
                "no_target_cash_all_applied": False,
            }
        )
    return pd.DataFrame(rows)


def _replay_static_ready_segment(
    *,
    prices: dict[str, pd.DataFrame],
    config: Any,
    static_ready_start: str,
    end: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not static_ready_start:
        return pd.DataFrame(), pd.DataFrame()
    start = pd.Timestamp(static_ready_start).normalize()
    reference = _formal_pool1_reference_stream(
        prices=prices,
        config=config,
        start_date=start,
        end_date=end,
    )
    trading_dates = [pd.Timestamp(date).normalize() for date in reference["signal_date"].astype(str)]
    ranking = _pool1_ranking_panel(trading_dates, prices, _name_map(config))
    dynamic_coverage = _dynamic_coverage_panel(trading_dates, ranking)
    score_context = _dynamic_score_margin_panel(dynamic_coverage, ranking, prices[TW50_BENCHMARK])
    rows: list[dict[str, Any]] = []
    for item in reference.to_dict(orient="records"):
        target = str(item.get("formal_pool1_target") or "")
        actionable = _bool_like(item.get("target_is_actionable"))
        rows.append(
            {
                "signal_date": str(item.get("signal_date") or ""),
                "pool1_target": target,
                "pool1_target_weights": json.dumps({target: 1.0} if target and actionable else {}, ensure_ascii=False),
                "attack_gate_active": _bool_like(item.get("attack_gate_active")),
                "attack_gate_ever_activated": _bool_like(item.get("attack_gate_ever_activated")),
                "risk_off_active": _bool_like(item.get("risk_off_active")),
                "target_is_actionable": actionable,
                "model_target_status": str(item.get("model_target_status") or ""),
                "mode": str(item.get("mode") or ""),
                "regime": str(item.get("regime") or ""),
                "reason": "Full state-machine replay on static all-ticker Pool1 universe after all tickers are scoring-ready.",
                "source_formal_ready": True,
                "no_target_cash_all_applied": False,
            }
        )
    return pd.DataFrame(rows), score_context


def _attach_dynamic_context(
    replayed: pd.DataFrame,
    dynamic_coverage: pd.DataFrame,
    score_context: pd.DataFrame,
) -> pd.DataFrame:
    if replayed.empty:
        return replayed
    context = dynamic_coverage.merge(score_context, on="signal_date", how="left")
    keep_cols = [
        "signal_date",
        "available_universe_count",
        "candidate_tickers",
        "top_ticker",
        "top_score",
        "fallback_0050_score",
        "score_margin",
        "dynamic_persistence_top_days",
        "raw_dynamic_attack_gate_pass",
    ]
    existing = [col for col in keep_cols if col in context.columns]
    return replayed.merge(context[existing], on="signal_date", how="left")


def _coverage_summary(
    dynamic_coverage: pd.DataFrame,
    blocked: pd.DataFrame,
    replayed: pd.DataFrame,
    static_ready_start: str,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "period": "2014-11-03_to_before_static_all_ticker_ready",
                "start_date": str(dynamic_coverage["signal_date"].iloc[0]) if not dynamic_coverage.empty else "",
                "end_date": _previous_business_day(static_ready_start) if static_ready_start else "",
                "rows": int(len(blocked)),
                "pool1_full_state_replay_ready": False,
                "coverage_state": "blocked_dynamic_universe_state_carryover_missing",
            },
            {
                "period": "static_all_ticker_ready_to_2021-12-31",
                "start_date": static_ready_start,
                "end_date": str(replayed["signal_date"].iloc[-1]) if not replayed.empty else "",
                "rows": int(len(replayed)),
                "pool1_full_state_replay_ready": bool(not replayed.empty),
                "coverage_state": "pool1_full_state_replayed_static_universe_segment",
            },
            {
                "period": "full_2014_2021",
                "start_date": str(dynamic_coverage["signal_date"].iloc[0]) if not dynamic_coverage.empty else "",
                "end_date": str(dynamic_coverage["signal_date"].iloc[-1]) if not dynamic_coverage.empty else "",
                "rows": int(len(dynamic_coverage)),
                "pool1_full_state_replay_ready": False,
                "coverage_state": "partial_only_until_dynamic_state_injection_exists",
            },
        ]
    )


def _replay_contract(static_ready_start: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "component": "2022plus_full_state_equivalence",
                "contract": "Full state-machine adapter passed all-row exact equivalence in 2022+.",
                "implemented": True,
                "formal_ready": True,
            },
            {
                "component": "static_all_ticker_segment",
                "contract": f"From {static_ready_start}, all 8 Pool1 tickers are scoring-ready; static universe replay can use current state machine.",
                "implemented": bool(static_ready_start),
                "formal_ready": bool(static_ready_start),
            },
            {
                "component": "pre_static_dynamic_universe_segment",
                "contract": "Before all 8 tickers are scoring-ready, replay needs state carryover across changing universe sets.",
                "implemented": False,
                "formal_ready": False,
            },
            {
                "component": "no_target_cash_all",
                "contract": "Not applied in this Pool1 signal replay package; execution risk-control waits for full formal target stream.",
                "implemented": False,
                "formal_ready": False,
            },
        ]
    )


def _blocker_by_field(static_ready_start: str, blocked: pd.DataFrame, replayed: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if not blocked.empty:
        rows.append(
            {
                "field_name": "dynamic_universe_state_carryover",
                "blocker": "simulate_regime_mode_switch_has_static_common_date_universe_contract",
                "severity": "blocks_pre_static_segment_formal_ready",
                "affected_period": f"{blocked['signal_date'].iloc[0]}..{blocked['signal_date'].iloc[-1]}",
                "next_action": "Add a state-machine replay adapter that can accept date-aware candidate universe and carry attack_gate/risk_off/account state across universe changes.",
                "formal_ready": False,
            }
        )
    if replayed.empty:
        rows.append(
            {
                "field_name": "static_all_ticker_segment",
                "blocker": "no_static_all_ticker_ready_segment_found",
                "severity": "blocks_all_2014_2021",
                "affected_period": "",
                "next_action": "Inspect lifecycle and price coverage.",
                "formal_ready": False,
            }
        )
    else:
        rows.append(
            {
                "field_name": "pool2_persistence_full_reconstruction",
                "blocker": "not_in_this_task",
                "severity": "next_blocker_after_pool1_static_segment",
                "affected_period": f"{static_ready_start}..{replayed['signal_date'].iloc[-1]}",
                "next_action": "Reconstruct Pool2 persistence before building combined formal target stream.",
                "formal_ready": False,
            }
        )
    return pd.DataFrame(rows)


def _source_decisions(
    price_meta: dict[str, dict[str, Any]],
    equivalence_manifest: dict[str, Any],
    static_ready_start: str,
    blocked: pd.DataFrame,
    replayed: pd.DataFrame,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "source_layer": "full_state_machine_equivalence_2022plus",
                "source_path": DEFAULT_EQUIVALENCE_DIR,
                "status": "accepted",
                "decision": "2022+ all-row equivalence pass allows use of same full state-machine semantics where the universe is static/all-ticker ready.",
                "formal_or_proxy": "formal_contract",
                "metadata": json.dumps(equivalence_manifest, ensure_ascii=False),
            },
            {
                "source_layer": "static_all_ticker_segment",
                "source_path": "pool1_full_state_replayed_signals.csv",
                "status": "created" if not replayed.empty else "missing",
                "decision": f"Pool1 full-state replay starts at {static_ready_start}; this is a formal-ready Pool1 candidate segment, not a combined formal target stream.",
                "formal_or_proxy": "pool1_formal_ready_candidate",
                "metadata": json.dumps({"rows": int(len(replayed))}, ensure_ascii=False),
            },
            {
                "source_layer": "pre_static_dynamic_universe_segment",
                "source_path": "blocked_signal_rows.csv",
                "status": "blocked" if not blocked.empty else "not_applicable",
                "decision": "Not formal-ready until dynamic universe state carryover adapter exists.",
                "formal_or_proxy": "blocked",
                "metadata": json.dumps({"rows": int(len(blocked))}, ensure_ascii=False),
            },
            {
                "source_layer": "price_sources",
                "source_path": DEFAULT_PRICE_CACHE_DIR,
                "status": "accepted",
                "decision": "Price sources are sufficient for static all-ticker segment replay.",
                "formal_or_proxy": "formal_input_for_pool1_segment",
                "metadata": json.dumps(price_meta, ensure_ascii=False),
            },
        ]
    )


def _next_step_handoff(static_ready_start: str) -> str:
    return "\n".join(
        [
            "# Pool1 full-state replay 2014～2021 handoff",
            "",
            "## 已完成",
            f"- 2022+ full state-machine equivalence 已通過後，回套 2014～2021 的 static all-ticker segment：{static_ready_start}～2021-12-31。",
            "- 輸出 `pool1_full_state_replayed_signals.csv`，可作 Pool1 formal-ready candidate segment。",
            "",
            "## 仍 blocked",
            "- 2014-11-03～static all-ticker ready 前一日仍缺 dynamic universe state carryover API。",
            "- 這不是 price/PIT 缺口，而是 `simulate_regime_mode_switch` 目前的 static common-date universe 合約限制。",
            "",
            "## 下一步",
            "1. 若要完整 2014/11 起 Pool1：補 dynamic universe state injection/carryover adapter。",
            "2. 若先推後半段 formal target stream：接 Pool2 persistence full reconstruction。",
            "",
            "## 邊界",
            "- 不套 no-target cash-all 到 2014～2021。",
            "- 不產 combined formal target stream。",
            "- formal_model_changed=false；trade_decision_changed=false。",
        ]
    ) + "\n"


def _final_summary(coverage: pd.DataFrame, blockers: pd.DataFrame) -> str:
    coverage_lines = [
        f"- {row['period']}: rows={row['rows']}, ready={row['pool1_full_state_replay_ready']}, state={row['coverage_state']}"
        for row in coverage.to_dict(orient="records")
    ]
    blocker_lines = [
        f"- {row['field_name']}: {row['blocker']} ({row['affected_period']})"
        for row in blockers.to_dict(orient="records")
    ]
    return "\n".join(
        [
            "# Pool1 full-state replay 2014～2021 dynamic universe",
            "",
            "## 判定",
            "部分完成：2018-02-06～2021-12-31 可用完整 state-machine contract 產 Pool1 formal-ready candidate rows；2014-11-03～2018-02-05 仍 blocked。",
            "",
            "## Coverage",
            *coverage_lines,
            "",
            "## Blockers",
            *blocker_lines,
            "",
            "- formal_model_changed=false",
            "- trade_decision_changed=false",
            "- no_target_cash_all_applied=false",
        ]
    ) + "\n"


def _previous_business_day(value: str) -> str:
    if not value:
        return ""
    return (pd.Timestamp(value) - pd.offsets.BDay(1)).strftime("%Y-%m-%d")


def _bool_like(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Replay 2014-2021 Pool1 full state where dynamic universe contract permits.")
    parser.add_argument("--dynamic-universe-dir", default=DEFAULT_DYNAMIC_UNIVERSE_DIR)
    parser.add_argument("--score-margin-dir", default=DEFAULT_SCORE_MARGIN_DIR)
    parser.add_argument("--equivalence-dir", default=DEFAULT_EQUIVALENCE_DIR)
    parser.add_argument("--price-cache-dir", default=DEFAULT_PRICE_CACHE_DIR)
    parser.add_argument("--price-source-registry", default=DEFAULT_PRICE_SOURCE_REGISTRY)
    parser.add_argument("--config-path", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", default=DEFAULT_END_DATE)
    args = parser.parse_args(argv)
    output = run_pool1_full_state_replay_201411_dynamic_universe(
        dynamic_universe_dir=args.dynamic_universe_dir,
        score_margin_dir=args.score_margin_dir,
        equivalence_dir=args.equivalence_dir,
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
