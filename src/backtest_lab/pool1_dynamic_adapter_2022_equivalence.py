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
    _load_price_source,
    _load_price_source_registry,
    _name_map,
)
from backtest_lab.data import split_adjusted_dividends
from backtest_lab.formal_model_contract import FORMAL_MODEL_ROUTE, FORMAL_MODEL_TARGET
from backtest_lab.pool1_dynamic_score_margin_state_adapter import _dynamic_score_margin_panel
from backtest_lab.regime_mode_switch import frozen_cycle_proven_top1_v1_variant, simulate_regime_mode_switch
from backtest_lab.stock_pool_observation import FROZEN_BEST_GROUP_ID
from backtest_lab.strategies import relative_strength_scores


TASK_ID = "TASK-BACKTEST-CORE-POOL1-DYNAMIC-ADAPTER-2022-EQUIVALENCE-RUNNER-20260702"
DEFAULT_FORMAL_LONG_RANGE_DIR = "outputs/formal_long_range_signal_reconstruction_201411_latest_20260702"
DEFAULT_OUTPUT_DIR = "outputs/pool1_dynamic_adapter_2022_equivalence_20260702"
DEFAULT_START_DATE = "2022-01-03"
DEFAULT_END_DATE = "2026-06-12"


def run_pool1_dynamic_adapter_2022_equivalence(
    *,
    formal_long_range_dir: str | Path = DEFAULT_FORMAL_LONG_RANGE_DIR,
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
        config = load_config(config_path)
        registry = _load_price_source_registry(price_source_registry)
        start = pd.Timestamp(start_date).normalize()
        end = pd.Timestamp(end_date).normalize()

        log("load_prices", "started", f"{start.date()}..{end.date()}")
        prices, price_meta = _load_required_prices(
            price_cache_dir=price_cache_dir,
            registry=registry,
            required_tickers=sorted(set(POOL1_TICKERS) | {TW50_BENCHMARK}),
        )

        log("load_formal_long_range_target_stream", "started", str(formal_long_range_dir))
        formal_stream = pd.read_csv(Path(formal_long_range_dir) / "formal_long_range_target_stream.csv").fillna("")
        formal_stream = formal_stream[
            formal_stream["signal_date"].astype(str).between(start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
        ].copy()

        log("build_dynamic_adapter_panel", "started", "")
        trading_dates = _trading_dates(prices[TW50_BENCHMARK], start, end)
        ranking = _pool1_ranking_panel(trading_dates, prices, _name_map(config))
        dynamic_coverage = _dynamic_coverage_panel(trading_dates, ranking)
        adapter_panel = _dynamic_score_margin_panel(dynamic_coverage, ranking, prices[TW50_BENCHMARK])

        log("build_formal_pool1_reference", "started", "")
        reference = _formal_pool1_reference_stream(
            prices=prices,
            config=config,
            start_date=start,
            end_date=end,
        )

        log("compare_equivalence", "started", "")
        regression = _equivalence_regression(adapter_panel, reference, formal_stream)
        mismatch_samples = regression[~regression["row_match_for_formal_readiness"].astype(bool)].head(80).copy()
        summary = _equivalence_summary(regression)
        blockers = _blocker_by_field(summary)
        source_decision = _source_decisions(price_meta, formal_stream, reference, summary)

        log("write_outputs", "started", str(output))
        adapter_panel.to_csv(output / "dynamic_adapter_2022_score_margin_panel.csv", index=False, encoding="utf-8-sig")
        reference.to_csv(output / "formal_pool1_reference_stream.csv", index=False, encoding="utf-8-sig")
        regression.to_csv(output / "equivalence_regression_2022plus.csv", index=False, encoding="utf-8-sig")
        mismatch_samples.to_csv(output / "equivalence_mismatch_samples.csv", index=False, encoding="utf-8-sig")
        summary.to_csv(output / "equivalence_summary.csv", index=False, encoding="utf-8-sig")
        blockers.to_csv(output / "blocker_by_field.csv", index=False, encoding="utf-8-sig")
        source_decision.to_csv(output / "proxy_or_formal_source_decision.csv", index=False, encoding="utf-8-sig")
        (output / "next_step_handoff.md").write_text(_next_step_handoff(summary), encoding="utf-8")
        (output / "final_summary_zh.md").write_text(_final_summary(summary), encoding="utf-8")

        equivalence_pass = bool(summary.iloc[0]["equivalence_pass"]) if not summary.empty else False
        manifest = {
            "schema_version": 1,
            "task_id": TASK_ID,
            "status": "completed_equivalence_failed_blocked" if not equivalence_pass else "completed_equivalence_pass",
            "formal_model_target": FORMAL_MODEL_TARGET,
            "formal_model_route": FORMAL_MODEL_ROUTE,
            "date_start": start.strftime("%Y-%m-%d"),
            "date_end": end.strftime("%Y-%m-%d"),
            "dynamic_adapter_rows": int(len(adapter_panel)),
            "formal_reference_rows": int(len(reference)),
            "equivalence_rows": int(len(regression)),
            "equivalence_pass": equivalence_pass,
            "mismatch_rows": int((~regression["row_match_for_formal_readiness"].astype(bool)).sum()) if not regression.empty else 0,
            "formal_model_changed": False,
            "trade_decision_changed": False,
            "active_in_trade_decision": False,
            "no_target_cash_all_applied_to_2014_2021": False,
            "raw_diagnostic_pass_used_as_formal_target": False,
            "next_required_task": _next_task(summary),
            "outputs": {
                "dynamic_adapter_panel": "dynamic_adapter_2022_score_margin_panel.csv",
                "formal_pool1_reference": "formal_pool1_reference_stream.csv",
                "equivalence_regression": "equivalence_regression_2022plus.csv",
                "mismatch_samples": "equivalence_mismatch_samples.csv",
                "equivalence_summary": "equivalence_summary.csv",
                "blockers": "blocker_by_field.csv",
                "source_decision": "proxy_or_formal_source_decision.csv",
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
        pd.DataFrame([{"step": "run_pool1_dynamic_adapter_2022_equivalence", "error": str(exc)}]).to_csv(
            output / "failed.csv",
            index=False,
            encoding="utf-8-sig",
        )
        log("failed", "failed", str(exc))
        raise


def _load_required_prices(
    *,
    price_cache_dir: str | Path,
    registry: pd.DataFrame,
    required_tickers: list[str],
) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, Any]]]:
    prices: dict[str, pd.DataFrame] = {}
    meta: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    for ticker in required_tickers:
        frame, source_meta = _load_price_source(ticker, price_cache_dir=price_cache_dir, registry=registry)
        if frame is None:
            missing.append(ticker)
            continue
        prices[ticker] = frame
        meta[ticker] = source_meta
    if missing:
        raise FileNotFoundError(f"Missing required Pool1 prices: {', '.join(missing)}")
    return prices, meta


def _trading_dates(benchmark: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    dates = benchmark.index[(benchmark.index >= start) & (benchmark.index <= end)]
    return [pd.Timestamp(date).normalize() for date in dates]


def _pool1_ranking_panel(
    trading_dates: list[pd.Timestamp],
    prices: dict[str, pd.DataFrame],
    names: dict[str, str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    price_subset = {ticker: prices[ticker] for ticker in POOL1_TICKERS if ticker in prices}
    for signal_date in trading_dates:
        available = {
            ticker: frame
            for ticker, frame in price_subset.items()
            if frame.index.min() <= signal_date and frame.loc[frame.index <= signal_date, "adj_close"].dropna().shape[0] > 60
        }
        scores = relative_strength_scores(available, signal_date, windows=(20, 60))
        ranked = sorted(scores.items(), key=lambda item: (item[1], item[0]), reverse=True)
        for rank, (ticker, score) in enumerate(ranked, start=1):
            rows.append(
                {
                    "date": signal_date.strftime("%Y-%m-%d"),
                    "candidate_ticker": ticker,
                    "candidate_name": names.get(ticker, ticker.replace(".TW", "")),
                    "score": round(float(score), 8),
                    "raw_rank": rank,
                    "rank": rank,
                    "passed": "true",
                    "source": "2022_dynamic_adapter_equivalence_rebuild",
                }
            )
    return pd.DataFrame(rows)


def _dynamic_coverage_panel(trading_dates: list[pd.Timestamp], ranking: pd.DataFrame) -> pd.DataFrame:
    grouped = {
        str(date): group.sort_values("rank")["candidate_ticker"].astype(str).tolist()
        for date, group in ranking.groupby("date")
    }
    rows = []
    for signal_date in trading_dates:
        key = signal_date.strftime("%Y-%m-%d")
        tickers = grouped.get(key, [])
        rows.append(
            {
                "signal_date": key,
                "available_universe_count": len(tickers),
                "candidate_tickers": ",".join(tickers),
            }
        )
    return pd.DataFrame(rows)


def _formal_pool1_reference_stream(
    *,
    prices: dict[str, pd.DataFrame],
    config: Any,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    group = config.group_by_id(FROZEN_BEST_GROUP_ID)
    labels = {asset.ticker: asset.label for asset in group.assets}
    asset_types = {asset.ticker: asset.asset_type for asset in group.assets}
    required = {asset.ticker for asset in group.assets}
    variant = frozen_cycle_proven_top1_v1_variant()
    dividends = {
        ticker: split_adjusted_dividends(prices[ticker], config.manual_splits.get(ticker, ()))
        for ticker in required
        if ticker in prices
    }
    result = simulate_regime_mode_switch(
        name="pool1_dynamic_adapter_2022_reference",
        prices_by_ticker={ticker: prices[ticker] for ticker in sorted(required)},
        asset_types=asset_types,
        market_prices=prices[TW50_BENCHMARK],
        start_date=start_date.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d"),
        initial_cash=config.initial_cash_twd,
        cost_model=config.cost_model,
        variant=variant,
        dividend_series_by_ticker=dividends,
    )
    rows: list[dict[str, Any]] = []
    equity = result.equity_curve.reset_index().rename(columns={"date": "signal_date"})
    for item in equity.to_dict(orient="records"):
        signal_date = str(pd.Timestamp(item["signal_date"]).strftime("%Y-%m-%d"))
        target = str(item.get("current_ticker") or "")
        exposure = _to_float(item.get("current_exposure")) or 0.0
        actionable = bool(target and target.lower() != "cash" and exposure > 0)
        rows.append(
            {
                "signal_date": signal_date,
                "formal_pool1_target": target if actionable else "",
                "formal_pool1_target_display": labels.get(target, target) if actionable else "",
                "target_is_actionable": actionable,
                "model_target_status": "有合格模型目標" if actionable else "沒有合格持股目標",
                "attack_gate_active": bool(item.get("attack_gate_active")),
                "attack_gate_ever_activated": bool(item.get("attack_gate_ever_activated")),
                "risk_off_active": bool(item.get("risk_off_active")),
                "mode": str(item.get("mode") or ""),
                "regime": str(item.get("regime") or ""),
                "current_exposure": round(exposure, 8),
                "source": "simulate_regime_mode_switch_current_pool1_static_universe",
            }
        )
    return pd.DataFrame(rows)


def _equivalence_regression(
    adapter_panel: pd.DataFrame,
    reference: pd.DataFrame,
    formal_stream: pd.DataFrame,
) -> pd.DataFrame:
    merged = adapter_panel.merge(reference, on="signal_date", how="outer").merge(
        formal_stream[["signal_date", "pool1_top_candidate", "formal_target", "risk_off_state"]],
        on="signal_date",
        how="left",
    )
    rows: list[dict[str, Any]] = []
    for item in merged.fillna("").to_dict(orient="records"):
        adapter_top = str(item.get("top_ticker") or "")
        formal_pool1_target = str(item.get("formal_pool1_target") or "")
        pool1_vote = str(item.get("pool1_top_candidate") or "")
        raw_gate = _bool_like(item.get("raw_dynamic_attack_gate_pass"))
        attack_gate_active = _bool_like(item.get("attack_gate_active"))
        target_is_actionable = _bool_like(item.get("target_is_actionable"))
        top_matches_vote = bool(adapter_top and pool1_vote and adapter_top == pool1_vote)
        raw_gate_matches_state = raw_gate == attack_gate_active
        target_matches = bool(formal_pool1_target and adapter_top == formal_pool1_target and target_is_actionable)
        row_match = bool(top_matches_vote and raw_gate_matches_state and target_matches)
        rows.append(
            {
                "signal_date": str(item.get("signal_date") or ""),
                "formal_pool1_target": formal_pool1_target,
                "formal_pool1_vote_from_long_stream": pool1_vote,
                "formal_final_target": str(item.get("formal_target") or ""),
                "adapter_top_ticker": adapter_top,
                "adapter_raw_gate_pass": raw_gate,
                "reference_attack_gate_active": attack_gate_active,
                "reference_target_is_actionable": target_is_actionable,
                "reference_model_target_status": str(item.get("model_target_status") or ""),
                "score_margin": item.get("score_margin", ""),
                "top_matches_pool1_vote": top_matches_vote,
                "raw_gate_matches_reference_attack_gate": raw_gate_matches_state,
                "adapter_top_matches_formal_pool1_target": target_matches,
                "row_match_for_formal_readiness": row_match,
                "mismatch_reason": _mismatch_reason(top_matches_vote, raw_gate_matches_state, target_matches),
            }
        )
    return pd.DataFrame(rows).sort_values("signal_date")


def _equivalence_summary(regression: pd.DataFrame) -> pd.DataFrame:
    if regression.empty:
        return pd.DataFrame(
            [
                {
                    "equivalence_pass": False,
                    "reason": "no_regression_rows",
                    "rows": 0,
                    "matched_rows": 0,
                    "mismatch_rows": 0,
                    "top_vote_match_rate": 0.0,
                    "gate_state_match_rate": 0.0,
                    "target_match_rate": 0.0,
                    "next_minimum_blocker": "missing_reference_rows",
                }
            ]
        )
    rows = int(len(regression))
    matched = int(regression["row_match_for_formal_readiness"].astype(bool).sum())
    top_rate = float(regression["top_matches_pool1_vote"].astype(bool).mean())
    gate_rate = float(regression["raw_gate_matches_reference_attack_gate"].astype(bool).mean())
    target_rate = float(regression["adapter_top_matches_formal_pool1_target"].astype(bool).mean())
    pass_flag = matched == rows and rows > 0
    return pd.DataFrame(
        [
            {
                "equivalence_pass": pass_flag,
                "reason": "all_rows_match" if pass_flag else "dynamic_adapter_not_equivalent_to_current_pool1_state_machine",
                "rows": rows,
                "matched_rows": matched,
                "mismatch_rows": rows - matched,
                "top_vote_match_rate": round(top_rate, 6),
                "gate_state_match_rate": round(gate_rate, 6),
                "target_match_rate": round(target_rate, 6),
                "next_minimum_blocker": "" if pass_flag else "full_state_machine_adapter_or_reference_semantics_alignment",
            }
        ]
    )


def _blocker_by_field(summary: pd.DataFrame) -> pd.DataFrame:
    if not summary.empty and bool(summary.iloc[0]["equivalence_pass"]):
        return pd.DataFrame(columns=["field_name", "blocker", "severity", "next_action", "formal_ready"])
    return pd.DataFrame(
        [
            {
                "field_name": "raw_dynamic_attack_gate_pass",
                "blocker": "does_not_equivalently_reproduce_reference_attack_gate_active",
                "severity": "blocking_formal_readiness",
                "next_action": "Implement full stateful adapter for current simulate_regime_mode_switch semantics, including attack_gate_ever_activated and risk_off interaction.",
                "formal_ready": False,
            },
            {
                "field_name": "adapter_top_ticker_to_formal_pool1_target",
                "blocker": "ranking_top_and_raw_gate_are_not_formal_target_contract",
                "severity": "blocking_formal_target_stream",
                "next_action": "Do not use raw diagnostic pass as formal target. Align against full state machine output first.",
                "formal_ready": False,
            },
            {
                "field_name": "2022_plus_equivalence_regression",
                "blocker": "equivalence_failed",
                "severity": "blocks_2014_2021_formal_ready_promotion",
                "next_action": "Keep 2014-2021 dynamic adapter output as blocked diagnostic until equivalence passes.",
                "formal_ready": False,
            },
        ]
    )


def _source_decisions(
    price_meta: dict[str, dict[str, Any]],
    formal_stream: pd.DataFrame,
    reference: pd.DataFrame,
    summary: pd.DataFrame,
) -> pd.DataFrame:
    pass_flag = bool(summary.iloc[0]["equivalence_pass"]) if not summary.empty else False
    return pd.DataFrame(
        [
            {
                "source_layer": "price_sources",
                "source_path": DEFAULT_PRICE_CACHE_DIR,
                "status": "accepted",
                "decision": "2022+ Pool1 prices can build dynamic score margin diagnostics.",
                "formal_or_proxy": "formal_input_for_2022_equivalence",
                "metadata": json.dumps(price_meta, ensure_ascii=False),
            },
            {
                "source_layer": "formal_long_range_target_stream",
                "source_path": DEFAULT_FORMAL_LONG_RANGE_DIR,
                "status": "accepted_limited",
                "decision": "Contains pool1 vote/final target, but not enough by itself for attack-gate metadata.",
                "formal_or_proxy": "formal_target_stream_limited_metadata",
                "metadata": json.dumps({"rows": int(len(formal_stream))}, ensure_ascii=False),
            },
            {
                "source_layer": "simulate_regime_mode_switch_reference",
                "source_path": "src/backtest_lab/regime_mode_switch.py",
                "status": "accepted_for_reference_replay",
                "decision": "Used to rebuild current Pool1 static-universe state metadata for 2022+ equivalence.",
                "formal_or_proxy": "formal_reference_replay",
                "metadata": json.dumps({"rows": int(len(reference))}, ensure_ascii=False),
            },
            {
                "source_layer": "dynamic_adapter_2014_2021_promotion",
                "source_path": "outputs/pool1_dynamic_score_margin_state_adapter_201411_202112_20260702",
                "status": "blocked" if not pass_flag else "candidate_ready_for_next_validation",
                "decision": "Do not promote 2014-2021 dynamic adapter to formal-ready unless 2022+ equivalence passes.",
                "formal_or_proxy": "blocked_diagnostic" if not pass_flag else "candidate_pending_experiments_validation",
                "metadata": json.dumps({"equivalence_pass": pass_flag}, ensure_ascii=False),
            },
        ]
    )


def _mismatch_reason(top_match: bool, gate_match: bool, target_match: bool) -> str:
    reasons: list[str] = []
    if not top_match:
        reasons.append("adapter_top_differs_from_formal_pool1_vote")
    if not gate_match:
        reasons.append("raw_gate_pass_differs_from_reference_attack_gate_active")
    if not target_match:
        reasons.append("adapter_top_not_equivalent_to_formal_pool1_target")
    return ";".join(reasons)


def _next_step_handoff(summary: pd.DataFrame) -> str:
    pass_flag = bool(summary.iloc[0]["equivalence_pass"]) if not summary.empty else False
    if pass_flag:
        next_line = "Equivalence passed. Next step: hand 2014/11～2021 Pool1 dynamic adapter output to Experiments for formal-readiness validation or continue Pool2 persistence full reconstruction."
    else:
        next_line = "Equivalence failed. Next step: implement a full stateful adapter that mirrors `simulate_regime_mode_switch`, especially attack_gate_ever_activated / risk_off / target selection semantics. Do not run 2014～2021 formal performance."
    return "\n".join(
        [
            "# Pool1 dynamic adapter 2022+ equivalence handoff",
            "",
            "## 結論",
            next_line,
            "",
            "## 邊界",
            "- formal_model_changed=false",
            "- trade_decision_changed=false",
            "- raw diagnostic pass is not a formal target",
            "- no-target cash-all is not applied to blocked 2014～2021 rows",
        ]
    ) + "\n"


def _final_summary(summary: pd.DataFrame) -> str:
    row = summary.iloc[0].to_dict() if not summary.empty else {}
    pass_flag = bool(row.get("equivalence_pass", False))
    return "\n".join(
        [
            "# Pool1 dynamic adapter 2022+ equivalence",
            "",
            "## 判定",
            "PASS：adapter 可等價重現 2022+ formal Pool1 判斷。" if pass_flag else "FAIL / BLOCKED：adapter 尚不能等價重現 2022+ formal Pool1 state machine。",
            "",
            "## 統計",
            f"- rows: {row.get('rows', 0)}",
            f"- matched_rows: {row.get('matched_rows', 0)}",
            f"- mismatch_rows: {row.get('mismatch_rows', 0)}",
            f"- top_vote_match_rate: {row.get('top_vote_match_rate', 0)}",
            f"- gate_state_match_rate: {row.get('gate_state_match_rate', 0)}",
            f"- target_match_rate: {row.get('target_match_rate', 0)}",
            "",
            "## 下一步",
            str(row.get("next_minimum_blocker") or "handoff_to_experiments_or_pool2_persistence"),
            "",
            "- formal_model_changed=false",
            "- trade_decision_changed=false",
        ]
    ) + "\n"


def _next_task(summary: pd.DataFrame) -> str:
    if not summary.empty and bool(summary.iloc[0]["equivalence_pass"]):
        return "pool2_persistence_full_reconstruction_or_experiments_pool1_adapter_validation"
    return "pool1_full_state_machine_adapter_equivalence"


def _to_float(value: Any) -> float | None:
    try:
        if value is None or str(value) == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _bool_like(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Pool1 dynamic adapter 2022+ equivalence regression.")
    parser.add_argument("--formal-long-range-dir", default=DEFAULT_FORMAL_LONG_RANGE_DIR)
    parser.add_argument("--price-cache-dir", default=DEFAULT_PRICE_CACHE_DIR)
    parser.add_argument("--price-source-registry", default=DEFAULT_PRICE_SOURCE_REGISTRY)
    parser.add_argument("--config-path", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", default=DEFAULT_END_DATE)
    args = parser.parse_args(argv)
    output = run_pool1_dynamic_adapter_2022_equivalence(
        formal_long_range_dir=args.formal_long_range_dir,
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
