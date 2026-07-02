from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from backtest_lab.current_formal_pool1_pool2_signal_panels import (
    DEFAULT_PRICE_CACHE_DIR,
    DEFAULT_PRICE_SOURCE_REGISTRY,
    TW50_BENCHMARK,
    _load_price_source,
    _load_price_source_registry,
)
from backtest_lab.formal_model_contract import FORMAL_MODEL_ROUTE, FORMAL_MODEL_TARGET
from backtest_lab.regime_mode_switch import frozen_cycle_proven_top1_v1_variant
from backtest_lab.strategies import relative_strength_scores


TASK_ID = "TASK-BACKTEST-CORE-POOL1-DYNAMIC-SCORE-MARGIN-STATE-MACHINE-ADAPTER-201411-20260702"
DEFAULT_PANEL_DIR = "outputs/current_formal_pool1_pool2_signal_panels_201411_202112_20260630"
DEFAULT_DYNAMIC_UNIVERSE_DIR = "outputs/date_aware_dynamic_universe_state_replay_201411_202112_20260702"
DEFAULT_OUTPUT_DIR = "outputs/pool1_dynamic_score_margin_state_adapter_201411_202112_20260702"


def run_pool1_dynamic_score_margin_state_adapter(
    *,
    panel_dir: str | Path = DEFAULT_PANEL_DIR,
    dynamic_universe_dir: str | Path = DEFAULT_DYNAMIC_UNIVERSE_DIR,
    price_cache_dir: str | Path = DEFAULT_PRICE_CACHE_DIR,
    price_source_registry: str | Path = DEFAULT_PRICE_SOURCE_REGISTRY,
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
        universe_root = Path(dynamic_universe_dir)
        log("load_inputs", "started", f"{panel_root}; {universe_root}")
        panel_manifest = _load_json(panel_root / "manifest.json")
        ranking = pd.read_csv(panel_root / "pool1_daily_candidate_ranking_panel.csv").fillna("")
        dynamic_coverage = pd.read_csv(universe_root / "dynamic_universe_state_replay_coverage.csv").fillna("")

        log("load_fallback_prices", "started", TW50_BENCHMARK)
        registry = _load_price_source_registry(price_source_registry)
        fallback_prices, fallback_meta = _load_price_source(
            TW50_BENCHMARK,
            price_cache_dir=price_cache_dir,
            registry=registry,
        )
        if fallback_prices is None:
            raise FileNotFoundError(f"Missing fallback price source for {TW50_BENCHMARK}")

        log("build_score_margin_panel", "started", "")
        score_margin = _dynamic_score_margin_panel(dynamic_coverage, ranking, fallback_prices)
        attack_contract = _attack_gate_adapter_contract()
        persistence_contract = _persistence_contract()
        equivalence = _equivalence_regression_2022plus()
        blocked = _blocked_signal_rows(score_margin)
        blockers = _blocker_by_field()
        source_decision = _source_decisions(fallback_meta)

        log("write_outputs", "started", "")
        score_margin.to_csv(output / "dynamic_score_margin_panel.csv", index=False, encoding="utf-8-sig")
        attack_contract.to_csv(output / "dynamic_attack_gate_adapter_contract.csv", index=False, encoding="utf-8-sig")
        persistence_contract.to_csv(output / "dynamic_persistence_state_contract.csv", index=False, encoding="utf-8-sig")
        equivalence.to_csv(output / "equivalence_regression_2022plus.csv", index=False, encoding="utf-8-sig")
        blocked.to_csv(output / "blocked_signal_rows.csv", index=False, encoding="utf-8-sig")
        blockers.to_csv(output / "blocker_by_field.csv", index=False, encoding="utf-8-sig")
        source_decision.to_csv(output / "proxy_or_formal_source_decision.csv", index=False, encoding="utf-8-sig")
        (output / "next_step_handoff.md").write_text(_next_step_handoff(), encoding="utf-8")
        (output / "final_summary_zh.md").write_text(_final_summary(panel_manifest, score_margin, blockers), encoding="utf-8")

        fallback_ready_days = int(score_margin["fallback_0050_score_ready"].sum()) if not score_margin.empty else 0
        output_manifest = {
            "schema_version": 1,
            "task_id": TASK_ID,
            "status": "completed_blocked_score_margin_adapter_package",
            "formal_model_target": FORMAL_MODEL_TARGET,
            "formal_model_route": FORMAL_MODEL_ROUTE,
            "date_start": str(panel_manifest.get("date_start") or _first_signal_date(score_margin)),
            "date_end": str(panel_manifest.get("date_end") or _last_signal_date(score_margin)),
            "dynamic_score_margin_panel_ready": True,
            "score_margin_rows": int(len(score_margin)),
            "fallback_0050_score_ready_days": fallback_ready_days,
            "dynamic_attack_gate_adapter_contract_defined": True,
            "dynamic_attack_gate_formal_ready": False,
            "equivalence_regression_2022plus_status": "blocked_missing_2022_dynamic_equivalence_runner",
            "formal_ready_pool1_rows": 0,
            "blocked_signal_rows": int(len(blocked)),
            "no_target_cash_all_applied": False,
            "proxy_used_as_formal": False,
            "uses_forward_return_as_rule": False,
            "formal_model_changed": False,
            "trade_decision_changed": False,
            "active_in_trade_decision": False,
            "next_required_task": "pool1_dynamic_adapter_2022_equivalence_runner",
            "outputs": {
                "score_margin_panel": "dynamic_score_margin_panel.csv",
                "attack_gate_contract": "dynamic_attack_gate_adapter_contract.csv",
                "persistence_contract": "dynamic_persistence_state_contract.csv",
                "equivalence_regression": "equivalence_regression_2022plus.csv",
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
        pd.DataFrame([{"step": "run_pool1_dynamic_score_margin_state_adapter", "error": str(exc)}]).to_csv(
            output / "failed.csv",
            index=False,
            encoding="utf-8-sig",
        )
        log("failed", "failed", str(exc))
        raise


def _dynamic_score_margin_panel(
    dynamic_coverage: pd.DataFrame,
    ranking: pd.DataFrame,
    fallback_prices: pd.DataFrame,
) -> pd.DataFrame:
    top_by_date = _top_by_date(ranking)
    fallback_scores = _fallback_scores(dynamic_coverage["signal_date"].astype(str).tolist(), fallback_prices)
    rows: list[dict[str, Any]] = []
    for item in dynamic_coverage.to_dict(orient="records"):
        date = str(item.get("signal_date") or "")
        top = top_by_date.get(date, {})
        top_score = _to_float(top.get("score"))
        fallback_score = fallback_scores.get(date)
        score_margin = None if top_score is None or fallback_score is None else top_score - fallback_score
        top_ticker = str(top.get("candidate_ticker") or "")
        raw_margin_pass = bool(score_margin is not None and score_margin >= frozen_cycle_proven_top1_v1_variant().attack_gate_margin_over_fallback)
        rows.append(
            {
                "signal_date": date,
                "available_universe_count": int(item.get("available_universe_count") or 0),
                "candidate_tickers": str(item.get("candidate_tickers") or ""),
                "top_ticker": top_ticker,
                "top_score": "" if top_score is None else round(top_score, 8),
                "fallback_0050_score": "" if fallback_score is None else round(fallback_score, 8),
                "fallback_0050_score_ready": fallback_score is not None,
                "score_margin": "" if score_margin is None else round(score_margin, 8),
                "top_rank": _clean_text(top.get("rank")),
                "candidate_reason": "dynamic universe score margin available" if score_margin is not None else "missing_top_or_fallback_score",
                "raw_margin_pass": raw_margin_pass,
                "dynamic_persistence_ready": False,
                "dynamic_attack_gate_formal_ready": False,
                "source_formal_ready": False,
            }
        )
    panel = pd.DataFrame(rows)
    return _apply_dynamic_persistence_diagnostic(panel)


def _apply_dynamic_persistence_diagnostic(panel: pd.DataFrame) -> pd.DataFrame:
    variant = frozen_cycle_proven_top1_v1_variant()
    lookback = int(variant.attack_gate_persistence_lookback_days or 0)
    min_days = int(variant.attack_gate_min_top_days or 0)
    if lookback <= 0 or min_days <= 0 or panel.empty:
        panel["dynamic_persistence_top_days"] = 0
        panel["raw_dynamic_attack_gate_pass"] = panel["raw_margin_pass"]
        return panel
    top_history = panel["top_ticker"].astype(str).tolist()
    margins = panel["raw_margin_pass"].astype(bool).tolist()
    top_days: list[int] = []
    raw_passes: list[bool] = []
    for index, ticker in enumerate(top_history):
        prior = top_history[max(0, index - lookback) : index]
        same_top_days = sum(1 for prior_ticker in prior if ticker and prior_ticker == ticker)
        top_days.append(same_top_days)
        raw_passes.append(bool(margins[index] and len(prior) >= lookback and same_top_days >= min_days))
    panel["dynamic_persistence_top_days"] = top_days
    panel["raw_dynamic_attack_gate_pass"] = raw_passes
    return panel


def _fallback_scores(dates: list[str], fallback_prices: pd.DataFrame) -> dict[str, float | None]:
    scores: dict[str, float | None] = {}
    price_map = {TW50_BENCHMARK: fallback_prices}
    for date in dates:
        try:
            result = relative_strength_scores(price_map, pd.Timestamp(date), windows=(20, 60))
        except Exception:
            result = {}
        score = result.get(TW50_BENCHMARK)
        scores[date] = None if score is None else float(score)
    return scores


def _attack_gate_adapter_contract() -> pd.DataFrame:
    variant = frozen_cycle_proven_top1_v1_variant()
    return pd.DataFrame(
        [
            {
                "component": "margin_gate",
                "rule": f"top_score - fallback_0050_score >= {variant.attack_gate_margin_over_fallback}",
                "implemented_in_this_package": True,
                "formal_ready": False,
                "blocker": "needs_2022_equivalence_regression",
            },
            {
                "component": "persistence_gate",
                "rule": f"same top ticker at least {variant.attack_gate_min_top_days} days in prior {variant.attack_gate_persistence_lookback_days} days",
                "implemented_in_this_package": "diagnostic_only",
                "formal_ready": False,
                "blocker": "must_match_current_regime_mode_switch_prior_dates_semantics",
            },
            {
                "component": "reentry_gate",
                "rule": f"reentry_margin={variant.attack_gate_reentry_margin_over_fallback}; reentry_ratio={variant.attack_gate_reentry_min_short_to_medium_momentum_ratio}",
                "implemented_in_this_package": False,
                "formal_ready": False,
                "blocker": "requires_stateful_attack_gate_ever_activated_replay",
            },
            {
                "component": "risk_off_state",
                "rule": f"market_risk_off_filter={variant.market_risk_off_filter}",
                "implemented_in_this_package": False,
                "formal_ready": False,
                "blocker": "requires_full_regime_mode_switch_state_equivalence",
            },
        ]
    )


def _persistence_contract() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "state": "dynamic_top_history",
                "contract": "Use only dates where ticker is lifecycle-ready; do not backfill not-yet-listed ticker history.",
                "implemented": True,
                "formal_ready": False,
                "blocker": "needs equivalence test against current static engine semantics",
            },
            {
                "state": "new_ticker_enters_universe",
                "contract": "Ticker can become top candidate only from first_pool1_scoring_date; persistence count starts at zero.",
                "implemented": True,
                "formal_ready": False,
                "blocker": "needs formal acceptance of dynamic-universe state continuity",
            },
            {
                "state": "attack_gate_ever_activated",
                "contract": "Carry forward after first formal activation, with current reentry reset rules.",
                "implemented": False,
                "formal_ready": False,
                "blocker": "requires full state machine adapter",
            },
        ]
    )


def _equivalence_regression_2022plus() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "check_name": "2022_plus_dynamic_adapter_vs_current_formal_stream",
                "status": "blocked_not_run",
                "date_start": "2022-01-03",
                "date_end": "2026-06-12",
                "reason": "This package builds 2014-2021 score margin panel only. A 2022+ dynamic score margin panel and formal attack gate state reference are required before equivalence can be tested.",
                "formal_ready": False,
            }
        ]
    )


def _blocked_signal_rows(score_margin: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for item in score_margin.to_dict(orient="records"):
        rows.append(
            {
                "signal_date": item["signal_date"],
                "available_universe_count": item["available_universe_count"],
                "candidate_tickers": item["candidate_tickers"],
                "pool1_target": "",
                "pool1_target_weights": "{}",
                "top_ticker": item["top_ticker"],
                "top_score": item["top_score"],
                "fallback_0050_score": item["fallback_0050_score"],
                "score_margin": item["score_margin"],
                "raw_dynamic_attack_gate_pass": item["raw_dynamic_attack_gate_pass"],
                "reason": "Score margin diagnostic is available, but formal state replay is blocked until 2022+ equivalence regression passes.",
                "source_formal_ready": False,
                "no_target_cash_all_applied": False,
            }
        )
    return pd.DataFrame(rows)


def _blocker_by_field() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "field_name": "equivalence_regression_2022plus",
                "blocker": "missing_2022_dynamic_score_margin_and_formal_state_reference",
                "severity": "blocking_formal_readiness",
                "next_action": "Generate the same dynamic score margin panel for 2022+ and compare adapter output with current formal target stream and attack gate metadata.",
                "formal_ready": False,
            },
            {
                "field_name": "attack_gate_ever_activated",
                "blocker": "missing_full_state_machine_adapter",
                "severity": "blocking_formal_target_status",
                "next_action": "Implement stateful replay including reentry reset and risk-off interaction after equivalence scaffold exists.",
                "formal_ready": False,
            },
            {
                "field_name": "risk_off_active",
                "blocker": "missing_full_regime_mode_switch_state_equivalence",
                "severity": "blocking_model_target_status",
                "next_action": "Mirror current market risk filter and interaction with attack gate in adapter.",
                "formal_ready": False,
            },
        ]
    )


def _source_decisions(fallback_meta: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "source_layer": "0050_fallback_score",
                "source_path": "backtest_cache/stock_pool_observations or configured fallback cache",
                "status": "accepted",
                "decision": "0050 fallback score can be computed daily where 60d warmup exists.",
                "formal_or_proxy": "formal_input",
                "metadata": json.dumps(fallback_meta, ensure_ascii=False),
            },
            {
                "source_layer": "dynamic_score_margin_panel",
                "source_path": "dynamic_score_margin_panel.csv",
                "status": "created",
                "decision": "Can support adapter diagnostics, but not formal output until equivalence passes.",
                "formal_or_proxy": "diagnostic_input_pending_equivalence",
                "metadata": "",
            },
            {
                "source_layer": "ranking_first_as_formal_target",
                "source_path": "",
                "status": "rejected",
                "decision": "Ranking first and raw margin pass cannot be packaged as formal target.",
                "formal_or_proxy": "proxy_rejected",
                "metadata": "",
            },
            {
                "source_layer": "no_target_cash_all",
                "source_path": "formal execution risk-control rule",
                "status": "not_applied",
                "decision": "No formal target stream exists for 2014-2021 in this package.",
                "formal_or_proxy": "formal_rule_waiting_for_target_stream",
                "metadata": "",
            },
        ]
    )


def _next_step_handoff() -> str:
    return "\n".join(
        [
            "# Pool1 dynamic score margin and state adapter handoff",
            "",
            "## 已完成",
            "- 2014/11～2021 dynamic score margin panel 已產出。",
            "- 0050 fallback score 可逐日計算；warmup 不足日會明確空白。",
            "- 已產 raw margin pass / diagnostic persistence，不包裝成 formal target。",
            "",
            "## 尚未 formal-ready",
            "缺 2022+ equivalence regression。下一棒必須用相同 adapter 產 2022+ dynamic score margin panel，並與現行 formal target stream / attack gate metadata 對齊。沒通過前，不得交 Experiments 跑績效。",
            "",
            "## 下一個任務",
            "`TASK-BACKTEST-CORE-POOL1-DYNAMIC-ADAPTER-2022-EQUIVALENCE-RUNNER-20260702`",
            "",
            "## 邊界",
            "- 不套 no-target cash-all。",
            "- 不用 ranking first 當 formal target。",
            "- formal_model_changed=false；trade_decision_changed=false。",
        ]
    ) + "\n"


def _final_summary(panel_manifest: dict[str, Any], score_margin: pd.DataFrame, blockers: pd.DataFrame) -> str:
    fallback_ready = int(score_margin["fallback_0050_score_ready"].sum()) if not score_margin.empty else 0
    raw_passes = int(score_margin["raw_dynamic_attack_gate_pass"].sum()) if not score_margin.empty else 0
    return "\n".join(
        [
            "# Pool1 dynamic score margin and state adapter",
            "",
            "## 判定",
            "2014/11～2021 的每日 dynamic score margin panel 已可產出，0050 fallback score 可計算；但 attack gate state 仍不能 formal-ready，因為尚未完成 2022+ equivalence regression 與完整 state machine adapter。",
            "",
            "## 本批輸出",
            f"- 來源區間：{panel_manifest.get('date_start')}～{panel_manifest.get('date_end')}",
            f"- score margin rows：{len(score_margin)}",
            f"- fallback score ready days：{fallback_ready}",
            f"- raw dynamic attack gate pass diagnostic days：{raw_passes}",
            f"- blockers：{len(blockers)}",
            "",
            "## 結論",
            "目前只能視為 adapter diagnostic / blocker package，不得產 formal-ready Pool1 rows，也不得交 Experiments 跑績效。",
            "",
            "- formal_model_changed=false",
            "- trade_decision_changed=false",
            "- no_target_cash_all_applied=false",
        ]
    ) + "\n"


def _top_by_date(ranking: pd.DataFrame) -> dict[str, dict[str, Any]]:
    if ranking.empty:
        return {}
    frame = ranking.copy()
    frame["rank_num"] = pd.to_numeric(frame.get("rank", ""), errors="coerce")
    top = frame.sort_values(["date", "rank_num", "candidate_ticker"]).groupby("date", as_index=False).first()
    return {str(item.get("date") or ""): item for item in top.to_dict(orient="records")}


def _to_float(value: Any) -> float | None:
    try:
        if value is None or str(value) == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _clean_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value)


def _first_signal_date(frame: pd.DataFrame) -> str:
    return "" if frame.empty else str(frame["signal_date"].iloc[0])


def _last_signal_date(frame: pd.DataFrame) -> str:
    return "" if frame.empty else str(frame["signal_date"].iloc[-1])


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build Pool1 dynamic score margin and state adapter package.")
    parser.add_argument("--panel-dir", default=DEFAULT_PANEL_DIR)
    parser.add_argument("--dynamic-universe-dir", default=DEFAULT_DYNAMIC_UNIVERSE_DIR)
    parser.add_argument("--price-cache-dir", default=DEFAULT_PRICE_CACHE_DIR)
    parser.add_argument("--price-source-registry", default=DEFAULT_PRICE_SOURCE_REGISTRY)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    output = run_pool1_dynamic_score_margin_state_adapter(
        panel_dir=args.panel_dir,
        dynamic_universe_dir=args.dynamic_universe_dir,
        price_cache_dir=args.price_cache_dir,
        price_source_registry=args.price_source_registry,
        output_dir=args.output_dir,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
