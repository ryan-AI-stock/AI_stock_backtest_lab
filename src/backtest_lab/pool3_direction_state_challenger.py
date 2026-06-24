from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


POOL3_ID = "large_core_bluechip_v0"
GATE_RULE_ID = "pool3_direction_state_layer_v1"
ETF_TICKERS = {"0050.TW", "00631L.TW"}
LEVERAGED_ETF_TICKERS = {"00631L.TW"}
VARIANTS = (
    "pool3_direction_state_base_v1",
    "pool3_direction_state_style_rotation_v1",
    "pool3_direction_state_attack_candidate_v1",
    "pool3_direction_state_core_alignment_soft_v1",
    "pool3_direction_state_core_alignment_strict_ablation",
    "pool3_direction_state_support_cap_v1",
)


def run_pool3_direction_state_challenger(
    *,
    replay_panel_path: str | Path,
    top_candidates_path: str | Path,
    output_dir: str | Path,
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

    log("load_inputs", "started", "")
    replay = pd.read_csv(replay_panel_path).fillna("")
    top_candidates = pd.read_csv(top_candidates_path).fillna("")
    _validate_inputs(replay, top_candidates)

    log("build_variant_panels", "started", "")
    diff_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    variant_outputs: dict[str, str] = {}
    for variant in VARIANTS:
        panel, diffs = _build_variant_panel(replay, top_candidates, variant=variant)
        panel_path = output / f"{variant}_replay_panel.csv"
        panel.to_csv(panel_path, index=False, encoding="utf-8-sig")
        variant_outputs[variant] = str(panel_path)
        diff_rows.extend(diffs)
        summary_rows.append(_variant_summary(panel, diffs, variant=variant))

    diff_path = output / "pool3_direction_state_decision_diff.csv"
    summary_path = output / "pool3_direction_state_variant_summary.csv"
    pd.DataFrame(diff_rows).to_csv(diff_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False, encoding="utf-8-sig")
    metadata = {
        "schema_version": 1,
        "task_id": "TASK-BACKTEST-CORE-POOL3-DIRECTION-STATE-LAYER-001",
        "status": "completed",
        "model": "pool3_direction_state_challenger",
        "formal_model_changed": False,
        "trade_decision_changed": False,
        "active_in_trade_decision": False,
        "gate_rule_id": GATE_RULE_ID,
        "replay_panel_path": str(replay_panel_path),
        "top_candidates_path": str(top_candidates_path),
        "variants": list(VARIANTS),
        "outputs": {
            "variant_replay_panels": variant_outputs,
            "decision_diff": str(diff_path),
            "variant_summary": str(summary_path),
            "run_log": str(output / "run_log.csv"),
        },
        "direction_state_contract": {
            "attack_confirmed": "full stock vote may be emitted",
            "attack_candidate": "direction support only unless variant promotes it",
            "style_rotation": "direction support or observation, depending on core state",
            "risk_off": "abstain",
            "no_edge": "observation only",
            "data_blocked": "fail-closed abstain",
        },
        "hard_boundaries": [
            "pool3_formal_vote_target_stock_only",
            "etf_not_eligible_for_pool3_stock_vote",
            "leveraged_etf_not_eligible_for_exact_ticker_consensus",
            "direction_support_only_not_exact_consensus",
            "pool3_radar_not_used",
            "valuation_not_used",
            "h3_day_trading_margin_overheat_not_used",
        ],
    }
    (output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame([{"status": "completed", "output_dir": str(output.resolve())}]).to_csv(
        output / "completed.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(columns=["step", "error"]).to_csv(output / "failed.csv", index=False, encoding="utf-8-sig")
    log("completed", "completed", str(output.resolve()))
    (output / "current_step.txt").write_text("completed\n", encoding="utf-8")
    return output


def _validate_inputs(replay: pd.DataFrame, top_candidates: pd.DataFrame) -> None:
    required_replay = {"period", "requested_signal_date", "pool_id", "top_ticker", "selection_layer", "eligible_for_pool_selection"}
    required_candidates = {"period", "requested_signal_date", "pool_id", "ticker", "selection_layer", "eligible_for_pool_selection"}
    missing_replay = required_replay - set(replay.columns)
    missing_candidates = required_candidates - set(top_candidates.columns)
    if missing_replay:
        raise ValueError("missing replay panel columns: " + ",".join(sorted(missing_replay)))
    if missing_candidates:
        raise ValueError("missing top candidate columns: " + ",".join(sorted(missing_candidates)))


def _build_variant_panel(
    replay: pd.DataFrame,
    top_candidates: pd.DataFrame,
    *,
    variant: str,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    panel = replay.copy().astype(object)
    _ensure_contract_columns(panel)
    panel = panel.astype(object)
    diffs: list[dict[str, Any]] = []
    support_counter = 0
    for index, row in panel[panel["pool_id"].astype(str) == POOL3_ID].iterrows():
        original = row.to_dict()
        adjusted = _apply_variant(panel, top_candidates, row, variant=variant, support_counter=support_counter)
        if adjusted.get("pool3_vote_state") == "direction_support_only":
            support_counter += 1
        for key, value in adjusted.items():
            panel.at[index, key] = value
        if _changed(original, adjusted):
            diffs.append(
                {
                    "variant": variant,
                    "period": row.get("period", ""),
                    "requested_signal_date": row.get("requested_signal_date", row.get("signal_date", "")),
                    "original_ticker": original.get("top_ticker", ""),
                    "challenger_ticker": adjusted.get("top_ticker", ""),
                    "original_selection_layer": original.get("selection_layer", ""),
                    "challenger_selection_layer": adjusted.get("selection_layer", ""),
                    "pool3_market_suitability_state": adjusted.get("pool3_market_suitability_state", ""),
                    "pool3_direction_state": adjusted.get("pool3_direction_state", ""),
                    "pool3_vote_state": adjusted.get("pool3_vote_state", ""),
                    "blocked_reason": adjusted.get("blocked_reason", ""),
                }
            )
    return panel, diffs


def _ensure_contract_columns(panel: pd.DataFrame) -> None:
    for column in (
        "pool3_market_suitability_state",
        "pool3_direction_state",
        "pool3_direction_state_reason",
        "pool3_vote_state",
        "pool3_candidate_group_strength",
        "pool3_style_breadth_score",
        "pool3_style_rotation_score",
        "pool3_candidate_relative_strength_vs_0050",
        "pool3_candidate_relative_strength_vs_style_benchmark",
        "pool3_candidate_trend_quality",
        "pool3_candidate_persistence_days",
        "pool3_top_cluster_gap",
        "pool3_candidate_score_gap",
        "core_pool_attack_count",
        "core_pool_risk_off_count",
        "pool3_direction_support_allowed",
        "fake_consensus_risk_flag",
        "asset_class",
        "is_etf",
        "is_leveraged_etf",
        "eligible_for_pool3_stock_vote",
        "eligible_for_exact_ticker_consensus",
        "blocked_reason",
    ):
        if column not in panel.columns:
            panel[column] = ""


def _apply_variant(
    panel: pd.DataFrame,
    top_candidates: pd.DataFrame,
    row: pd.Series,
    *,
    variant: str,
    support_counter: int,
) -> dict[str, Any]:
    adjusted = row.to_dict()
    date = str(row.get("requested_signal_date") or row.get("signal_date") or "")
    period = str(row.get("period") or "")
    peer = _peer_state(panel, period=period, date=date)
    candidates = _pool3_candidates(top_candidates, period=period, date=date)
    metrics = _candidate_metrics(candidates)
    candidate = _first_stock_candidate(candidates)
    original_ticker = str(row.get("top_ticker") or "").strip()
    market_state = _market_suitability_state(peer=peer, candidate=candidate)
    direction_state = _direction_state(candidate=candidate, metrics=metrics, market_state=market_state, variant=variant)
    support_allowed = _support_allowed(variant=variant, support_counter=support_counter)
    base = _base_contract(original_ticker, peer=peer, metrics=metrics, market_state=market_state, direction_state=direction_state)

    if candidate is None:
        return _as_abstain(adjusted, base, original_ticker, reason="Pool3 direction-state：無可用純個股候選。")
    ticker = str(candidate.get("ticker") or "").strip()
    if direction_state == "attack_confirmed":
        return _as_full_stock_vote(adjusted, base, candidate, reason="attack_confirmed：候選強度與市場適合度通過。")
    if direction_state in {"attack_candidate", "style_rotation"}:
        if support_allowed:
            return _as_direction_support(
                adjusted,
                base,
                candidate,
                reason=f"{direction_state}：方向可支持，但不投 exact ticker。",
            )
        return _as_observation(adjusted, base, ticker, reason=f"{direction_state}：support cap 或 variant 限制，僅觀察。")
    if direction_state == "risk_off":
        return _as_abstain(adjusted, base, ticker, reason="risk_off：核心池未開啟攻擊方向。")
    if direction_state == "data_blocked":
        return _as_abstain(adjusted, base, ticker, reason="data_blocked：資料不足，fail-closed abstain。")
    return _as_observation(adjusted, base, ticker, reason="no_edge：有資料但沒有明確優勢。")


def _pool3_candidates(top_candidates: pd.DataFrame, *, period: str, date: str) -> pd.DataFrame:
    subset = top_candidates[
        (top_candidates["period"].astype(str) == period)
        & (top_candidates["requested_signal_date"].astype(str) == date)
        & (top_candidates["pool_id"].astype(str) == POOL3_ID)
    ].copy()
    if "rank" in subset.columns:
        subset["_rank_number"] = pd.to_numeric(subset["rank"], errors="coerce").fillna(999)
        subset = subset.sort_values("_rank_number")
    return subset


def _first_stock_candidate(candidates: pd.DataFrame) -> dict[str, Any] | None:
    for _, row in candidates.iterrows():
        candidate = row.to_dict()
        ticker = str(candidate.get("ticker") or "").strip()
        if not ticker or _is_etf(ticker, candidate.get("asset_type", "")):
            continue
        if not _truthy(candidate.get("eligible_for_pool_selection")):
            continue
        if str(candidate.get("selection_layer") or "") != "formal_candidate":
            continue
        return candidate
    return None


def _candidate_metrics(candidates: pd.DataFrame) -> dict[str, float]:
    stock = []
    for _, row in candidates.iterrows():
        item = row.to_dict()
        ticker = str(item.get("ticker") or "").strip()
        if ticker and not _is_etf(ticker, item.get("asset_type", "")):
            stock.append(item)
    score_values = [_score(item) for item in stock]
    top = score_values[0] if score_values else 0.0
    second = score_values[1] if len(score_values) > 1 else 0.0
    return {
        "group_strength": round(top, 6),
        "breadth_score": round(min(1.0, len([score for score in score_values if score >= 0.5]) / 3.0), 6),
        "rotation_score": round(max(0.0, top - second), 6),
        "top_cluster_gap": round(top - second, 6),
        "candidate_score_gap": round(top - second, 6),
    }


def _market_suitability_state(*, peer: dict[str, str], candidate: dict[str, Any] | None) -> str:
    if candidate is None:
        return "data_blocked"
    attack_count = _core_attack_count(peer)
    if attack_count <= 0:
        return "risk_off"
    return "market_on" if attack_count >= 1 else "neutral"


def _direction_state(
    *,
    candidate: dict[str, Any] | None,
    metrics: dict[str, float],
    market_state: str,
    variant: str,
) -> str:
    if candidate is None:
        return "data_blocked"
    if market_state == "risk_off":
        return "risk_off"
    rank = _rank(candidate)
    score = _score(candidate)
    if variant == "pool3_direction_state_attack_candidate_v1" and rank <= 5 and score >= 0.25:
        return "attack_candidate"
    if rank <= 3 and score >= 0.5:
        return "attack_confirmed"
    if variant == "pool3_direction_state_style_rotation_v1" and metrics["breadth_score"] >= 0.34:
        return "style_rotation"
    if variant == "pool3_direction_state_core_alignment_soft_v1" and score >= 0.25:
        return "attack_candidate"
    if variant == "pool3_direction_state_core_alignment_strict_ablation" and rank <= 2 and score >= 0.6:
        return "attack_confirmed"
    if score >= 0.25:
        return "attack_candidate"
    return "no_edge"


def _support_allowed(*, variant: str, support_counter: int) -> bool:
    if variant == "pool3_direction_state_strict_core_alignment_ablation":
        return False
    if variant == "pool3_direction_state_support_cap_v1":
        return support_counter % 3 == 0
    return True


def _as_full_stock_vote(row: dict[str, Any], base: dict[str, Any], candidate: dict[str, Any], *, reason: str) -> dict[str, Any]:
    ticker = str(candidate.get("ticker") or "").strip()
    display = str(candidate.get("display") or candidate.get("top_display") or ticker).strip()
    row.update(base)
    row.update(
        {
            "top_ticker": ticker,
            "top_display": display,
            "top_asset_type": "stock",
            "asset_class": "stock",
            "is_etf": "false",
            "is_leveraged_etf": "false",
            "eligible_for_market_exposure": "false",
            "selection_layer": "formal_candidate",
            "eligible_for_pool_selection": "true",
            "eligible_for_pool3_stock_vote": "true",
            "eligible_for_exact_ticker_consensus": "true",
            "pool3_vote_state": "full_stock_vote",
            "attack_gate_open": "true",
            "gate_rule_id": GATE_RULE_ID,
            "gate_reason": reason,
            "selection_reason": reason,
            "pool3_direction_state_reason": reason,
            "blocked_reason": "",
        }
    )
    return row


def _as_direction_support(row: dict[str, Any], base: dict[str, Any], candidate: dict[str, Any], *, reason: str) -> dict[str, Any]:
    ticker = str(candidate.get("ticker") or "").strip()
    display = str(candidate.get("display") or candidate.get("top_display") or ticker).strip()
    row.update(base)
    row.update(
        {
            "top_ticker": ticker,
            "top_display": display,
            "top_asset_type": "stock",
            "asset_class": "stock",
            "is_etf": "false",
            "is_leveraged_etf": "false",
            "eligible_for_market_exposure": "false",
            "selection_layer": "direction_support_only",
            "eligible_for_pool_selection": "false",
            "eligible_for_pool3_stock_vote": "false",
            "eligible_for_exact_ticker_consensus": "false",
            "pool3_vote_state": "direction_support_only",
            "attack_gate_open": "false",
            "gate_rule_id": GATE_RULE_ID,
            "gate_reason": reason,
            "selection_reason": reason,
            "pool3_direction_state_reason": reason,
            "blocked_reason": reason,
        }
    )
    return row


def _as_observation(row: dict[str, Any], base: dict[str, Any], ticker: str, *, reason: str) -> dict[str, Any]:
    row.update(base)
    row.update(
        {
            "top_ticker": ticker,
            "selection_layer": "observation_only",
            "eligible_for_pool_selection": "false",
            "eligible_for_pool3_stock_vote": "false",
            "eligible_for_exact_ticker_consensus": "false",
            "pool3_vote_state": "observation_only",
            "attack_gate_open": "false",
            "gate_rule_id": GATE_RULE_ID,
            "gate_reason": reason,
            "selection_reason": reason,
            "pool3_direction_state_reason": reason,
            "blocked_reason": reason,
        }
    )
    return row


def _as_abstain(row: dict[str, Any], base: dict[str, Any], ticker: str, *, reason: str) -> dict[str, Any]:
    row.update(base)
    row.update(
        {
            "top_ticker": ticker,
            "selection_layer": "abstain",
            "eligible_for_pool_selection": "false",
            "eligible_for_pool3_stock_vote": "false",
            "eligible_for_exact_ticker_consensus": "false",
            "pool3_vote_state": "abstain",
            "attack_gate_open": "false",
            "gate_rule_id": GATE_RULE_ID,
            "gate_reason": reason,
            "selection_reason": reason,
            "pool3_direction_state_reason": reason,
            "blocked_reason": reason,
        }
    )
    return row


def _base_contract(ticker: str, *, peer: dict[str, str], metrics: dict[str, float], market_state: str, direction_state: str) -> dict[str, Any]:
    is_etf = _is_etf(ticker)
    attack_count = _core_attack_count(peer)
    risk_off_count = 2 - attack_count
    return {
        "asset_class": "leveraged_etf" if ticker in LEVERAGED_ETF_TICKERS else ("etf" if is_etf else "stock"),
        "is_etf": _bool_text(is_etf),
        "is_leveraged_etf": _bool_text(ticker in LEVERAGED_ETF_TICKERS),
        "pool3_market_suitability_state": market_state,
        "pool3_direction_state": direction_state,
        "pool3_candidate_group_strength": metrics["group_strength"],
        "pool3_style_breadth_score": metrics["breadth_score"],
        "pool3_style_rotation_score": metrics["rotation_score"],
        "pool3_candidate_relative_strength_vs_0050": metrics["group_strength"],
        "pool3_candidate_relative_strength_vs_style_benchmark": metrics["group_strength"],
        "pool3_candidate_trend_quality": "pass" if metrics["group_strength"] >= 0.5 else "watch",
        "pool3_candidate_persistence_days": "",
        "pool3_top_cluster_gap": metrics["top_cluster_gap"],
        "pool3_candidate_score_gap": metrics["candidate_score_gap"],
        "core_pool_attack_count": attack_count,
        "core_pool_risk_off_count": risk_off_count,
        "pool3_direction_support_allowed": _bool_text(direction_state in {"attack_candidate", "style_rotation"}),
        "fake_consensus_risk_flag": "watch_direction_support_rate",
    }


def _peer_state(panel: pd.DataFrame, *, period: str, date: str) -> dict[str, str]:
    subset = panel[
        (panel["period"].astype(str) == period)
        & (panel["requested_signal_date"].astype(str) == date)
        & panel["eligible_for_pool_selection"].map(_truthy)
    ]
    return {
        "pool1": _vote_for_fragment(subset, "ai_theme_large_cap"),
        "pool2": _vote_for_fragment(subset, "tw50_dynamic_constituents"),
    }


def _vote_for_fragment(subset: pd.DataFrame, fragment: str) -> str:
    rows = subset[subset["pool_id"].astype(str).str.contains(fragment, na=False)]
    if rows.empty:
        return ""
    return str(rows.iloc[0].get("top_ticker") or "").strip()


def _core_attack_count(peer: dict[str, str]) -> int:
    return sum(1 for ticker in peer.values() if str(ticker or "").strip())


def _score(candidate: dict[str, Any]) -> float:
    for key in ("rank_score", "score"):
        value = pd.to_numeric(pd.Series([candidate.get(key, "")]), errors="coerce").iloc[0]
        if pd.notna(value):
            return float(value)
    rank = _rank(candidate)
    return max(0.0, 1.0 - rank / 10.0)


def _rank(candidate: dict[str, Any]) -> float:
    value = pd.to_numeric(pd.Series([candidate.get("rank", "")]), errors="coerce").iloc[0]
    return float(value) if pd.notna(value) else 999.0


def _is_etf(ticker: str, asset_type: object = "") -> bool:
    return ticker in ETF_TICKERS or str(asset_type).strip().lower() == "etf"


def _changed(original: dict[str, Any], adjusted: dict[str, Any]) -> bool:
    keys = ("top_ticker", "selection_layer", "eligible_for_pool_selection", "pool3_vote_state", "pool3_direction_state")
    return any(str(original.get(key, "")) != str(adjusted.get(key, "")) for key in keys)


def _variant_summary(panel: pd.DataFrame, diff_rows: list[dict[str, Any]], *, variant: str) -> dict[str, Any]:
    pool3 = panel[panel["pool_id"].astype(str) == POOL3_ID]
    non_empty_tickers = pool3["top_ticker"].astype(str).str.strip()
    non_empty_tickers = non_empty_tickers[non_empty_tickers.ne("")]
    counts = non_empty_tickers.value_counts()
    top_ticker = str(counts.index[0]) if len(counts) else ""
    top_share = round(float(counts.iloc[0] / len(pool3)), 6) if len(counts) else 0.0
    direction_support_rows = int((pool3["pool3_vote_state"].astype(str) == "direction_support_only").sum())
    full_vote_rows = int((pool3["pool3_vote_state"].astype(str) == "full_stock_vote").sum())
    return {
        "variant": variant,
        "pool3_rows": int(len(pool3)),
        "full_stock_vote_rows": full_vote_rows,
        "direction_support_only_rows": direction_support_rows,
        "observation_only_rows": int((pool3["pool3_vote_state"].astype(str) == "observation_only").sum()),
        "abstain_rows": int((pool3["pool3_vote_state"].astype(str) == "abstain").sum()),
        "attack_confirmed_rows": int((pool3["pool3_direction_state"].astype(str) == "attack_confirmed").sum()),
        "attack_candidate_rows": int((pool3["pool3_direction_state"].astype(str) == "attack_candidate").sum()),
        "style_rotation_rows": int((pool3["pool3_direction_state"].astype(str) == "style_rotation").sum()),
        "risk_off_rows": int((pool3["pool3_direction_state"].astype(str) == "risk_off").sum()),
        "no_edge_rows": int((pool3["pool3_direction_state"].astype(str) == "no_edge").sum()),
        "data_blocked_rows": int((pool3["pool3_direction_state"].astype(str) == "data_blocked").sum()),
        "pool3_etf_stock_vote_rows": int((pool3["is_etf"].map(_truthy) & pool3["eligible_for_pool3_stock_vote"].map(_truthy)).sum()),
        "pool3_etf_exact_consensus_rows": int((pool3["is_etf"].map(_truthy) & pool3["eligible_for_exact_ticker_consensus"].map(_truthy)).sum()),
        "direction_support_only_rate": round(float(direction_support_rows / len(pool3)), 6) if len(pool3) else 0.0,
        "direction_support_over_full_vote": direction_support_rows > full_vote_rows,
        "top_ticker": top_ticker,
        "top_ticker_day_share": top_share,
        "top_ticker_day_share_over_40pct": top_share > 0.4,
        "changed_rows": len(diff_rows),
        "active_in_trade_decision": False,
        "formal_model_changed": False,
    }


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _bool_text(value: bool) -> str:
    return "true" if value else "false"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Pool3 direction-state challenger replay panels.")
    parser.add_argument("--replay-panel", required=True)
    parser.add_argument("--top-candidates", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = run_pool3_direction_state_challenger(
        replay_panel_path=args.replay_panel,
        top_candidates_path=args.top_candidates,
        output_dir=args.output_dir,
    )
    print(f"OUTPUT_DIR={output.resolve()}")


if __name__ == "__main__":
    main()
