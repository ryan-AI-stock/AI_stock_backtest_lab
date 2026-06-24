from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


POOL3_ID = "large_core_bluechip_v0"
GATE_RULE_ID = "pool3_vote_state_layer_v1"
ETF_TICKERS = {"0050.TW", "00631L.TW"}
LEVERAGED_ETF_TICKERS = {"00631L.TW"}
AI_MAINLINE_TICKERS = {
    "2330.TW",
    "2454.TW",
    "2308.TW",
    "2317.TW",
    "2382.TW",
    "3231.TW",
    "6669.TW",
}
VARIANTS = (
    "pool3_vote_state_style_base",
    "pool3_vote_state_early_cycle_activation",
    "pool3_vote_state_soft_low_correlation",
    "pool3_vote_state_trend_quality_plus_persistence",
    "pool3_vote_state_direction_support_only_ablation",
    "pool3_vote_state_strict_core_alignment_ablation",
)


def run_pool3_vote_state_challenger(
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

    diff_path = output / "pool3_vote_state_decision_diff.csv"
    summary_path = output / "pool3_vote_state_variant_summary.csv"
    pd.DataFrame(diff_rows).to_csv(diff_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False, encoding="utf-8-sig")
    metadata = {
        "schema_version": 1,
        "task_id": "TASK-BACKTEST-CORE-POOL3-VOTE-STATE-LAYER-001",
        "status": "completed",
        "model": "pool3_vote_state_challenger",
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
        "vote_state_contract": {
            "full_stock_vote": "eligible for formal Pool3 ticker vote and exact/direction consensus",
            "direction_support_only": "supports direction only; not eligible for exact ticker consensus",
            "observation_only": "candidate observed but blocked from vote and direction support",
            "abstain": "Pool3 abstains because data, risk, ETF-only, or candidate source is unavailable",
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
    required_replay = {
        "period",
        "requested_signal_date",
        "pool_id",
        "top_ticker",
        "selection_layer",
        "eligible_for_pool_selection",
    }
    required_candidates = {
        "period",
        "requested_signal_date",
        "pool_id",
        "ticker",
        "selection_layer",
        "eligible_for_pool_selection",
    }
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
    diffs: list[dict[str, Any]] = []
    for index, row in panel[panel["pool_id"].astype(str) == POOL3_ID].iterrows():
        original = row.to_dict()
        adjusted = _apply_variant(panel, top_candidates, row, variant=variant)
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
                    "original_eligible": original.get("eligible_for_pool_selection", ""),
                    "challenger_eligible": adjusted.get("eligible_for_pool_selection", ""),
                    "pool3_vote_state": adjusted.get("pool3_vote_state", ""),
                    "direction_support_state": adjusted.get("direction_support_state", ""),
                    "blocked_reason": adjusted.get("blocked_reason", ""),
                }
            )
    return panel, diffs


def _ensure_contract_columns(panel: pd.DataFrame) -> None:
    for column in (
        "asset_class",
        "is_etf",
        "is_leveraged_etf",
        "style_bucket",
        "pool3_vote_state",
        "pool3_vote_state_reason",
        "direction_support_state",
        "eligible_for_pool3_stock_vote",
        "eligible_for_market_exposure",
        "eligible_for_exact_ticker_consensus",
        "correlation_to_pool1_signal",
        "correlation_to_pool2_signal",
        "blocked_reason",
    ):
        if column not in panel.columns:
            panel[column] = ""


def _apply_variant(panel: pd.DataFrame, top_candidates: pd.DataFrame, row: pd.Series, *, variant: str) -> dict[str, Any]:
    adjusted = row.to_dict()
    date = str(row.get("requested_signal_date") or row.get("signal_date") or "")
    period = str(row.get("period") or "")
    peer = _peer_state(panel, period=period, date=date)
    candidates = _pool3_candidates(top_candidates, period=period, date=date)
    selected = _select_candidate(candidates, peer=peer, variant=variant)
    support_candidate = selected or _first_stock_candidate(candidates)
    original_ticker = str(row.get("top_ticker") or "").strip()

    if variant == "pool3_vote_state_direction_support_only_ablation":
        if support_candidate is None:
            return _as_abstain(adjusted, original_ticker, reason="無純個股候選可支持方向。", peer=peer)
        return _as_direction_support(adjusted, support_candidate, reason="direction_support_only ablation：只支持方向，不投 ticker。", peer=peer)

    if selected is None:
        if support_candidate is not None and _peer_market_on(peer):
            return _as_direction_support(
                adjusted,
                support_candidate,
                reason="有純個股候選且市場方向開啟，但未達 full_stock_vote 條件；僅方向支持。",
                peer=peer,
            )
        return _as_abstain(adjusted, original_ticker, reason="Pool3 vote-state abstain：無可用純個股候選或 peer risk-off。", peer=peer)

    if variant == "pool3_vote_state_strict_core_alignment_ablation" and not _same_as_peer(str(selected.get("ticker") or ""), peer):
        return _as_direction_support(
            adjusted,
            selected,
            reason="strict alignment ablation：未與池1/池2同標的；降為方向支持。",
            peer=peer,
        )
    return _as_full_stock_vote(adjusted, selected, variant=variant, peer=peer)


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


def _select_candidate(candidates: pd.DataFrame, *, peer: dict[str, str], variant: str) -> dict[str, Any] | None:
    stock_candidates = [_ for _ in (_candidate_dict(row) for _, row in candidates.iterrows()) if _stock_candidate_passes(_)]
    if not stock_candidates:
        return None
    if variant == "pool3_vote_state_early_cycle_activation":
        for candidate in stock_candidates:
            if _peer_market_on(peer) and _rank(candidate) <= 5:
                return candidate
        return None
    if variant == "pool3_vote_state_soft_low_correlation":
        different = [candidate for candidate in stock_candidates if not _same_as_peer(str(candidate.get("ticker") or ""), peer)]
        return (different or stock_candidates)[0]
    if variant == "pool3_vote_state_trend_quality_plus_persistence":
        for candidate in stock_candidates:
            if _trend_quality_pass(candidate) and _rank(candidate) <= 3:
                return candidate
        return None
    if variant == "pool3_vote_state_strict_core_alignment_ablation":
        for candidate in stock_candidates:
            if _same_as_peer(str(candidate.get("ticker") or ""), peer):
                return candidate
        return stock_candidates[0]
    return stock_candidates[0] if _peer_market_on(peer) else None


def _first_stock_candidate(candidates: pd.DataFrame) -> dict[str, Any] | None:
    for _, row in candidates.iterrows():
        candidate = _candidate_dict(row)
        if _stock_candidate_passes(candidate):
            return candidate
    return None


def _candidate_dict(row: pd.Series) -> dict[str, Any]:
    return row.to_dict()


def _stock_candidate_passes(candidate: dict[str, Any]) -> bool:
    ticker = str(candidate.get("ticker") or "").strip()
    if not ticker or _is_etf(ticker, candidate.get("asset_type", "")):
        return False
    if not _truthy(candidate.get("eligible_for_pool_selection")):
        return False
    return str(candidate.get("selection_layer") or "") == "formal_candidate"


def _as_full_stock_vote(row: dict[str, Any], candidate: dict[str, Any], *, variant: str, peer: dict[str, str]) -> dict[str, Any]:
    ticker = str(candidate.get("ticker") or "").strip()
    display = str(candidate.get("display") or candidate.get("top_display") or ticker).strip()
    row.update(_base_contract(ticker, peer=peer))
    row.update(
        {
            "top_ticker": ticker,
            "top_display": display,
            "top_asset_type": "stock",
            "selection_layer": "formal_candidate",
            "eligible_for_pool_selection": "true",
            "eligible_for_pool3_stock_vote": "true",
            "eligible_for_exact_ticker_consensus": "true",
            "pool3_vote_state": "full_stock_vote",
            "direction_support_state": "full_vote_supports_direction",
            "attack_gate_open": "true",
            "gate_rule_id": GATE_RULE_ID,
            "gate_reason": f"Pool3 vote-state {variant}：full_stock_vote；純個股且 peer market-on。",
            "selection_reason": f"Pool3 vote-state {variant}：full_stock_vote；純個股且 peer market-on。",
            "pool3_vote_state_reason": "正式個股票，可計入 exact / direction consensus。",
            "blocked_reason": "",
        }
    )
    return row


def _as_direction_support(row: dict[str, Any], candidate: dict[str, Any], *, reason: str, peer: dict[str, str]) -> dict[str, Any]:
    ticker = str(candidate.get("ticker") or "").strip()
    display = str(candidate.get("display") or candidate.get("top_display") or ticker).strip()
    row.update(_base_contract(ticker, peer=peer))
    row.update(
        {
            "top_ticker": ticker,
            "top_display": display,
            "top_asset_type": "stock",
            "selection_layer": "direction_support_only",
            "eligible_for_pool_selection": "false",
            "eligible_for_pool3_stock_vote": "false",
            "eligible_for_exact_ticker_consensus": "false",
            "pool3_vote_state": "direction_support_only",
            "direction_support_state": "supports_stock_attack_direction",
            "attack_gate_open": "false",
            "gate_rule_id": GATE_RULE_ID,
            "gate_reason": reason,
            "selection_reason": reason,
            "pool3_vote_state_reason": "只支持攻擊方向，不投 ticker，不計入 exact consensus。",
            "blocked_reason": reason,
        }
    )
    return row


def _as_abstain(row: dict[str, Any], ticker: str, *, reason: str, peer: dict[str, str]) -> dict[str, Any]:
    row.update(_base_contract(ticker, peer=peer))
    row.update(
        {
            "selection_layer": "abstain",
            "eligible_for_pool_selection": "false",
            "eligible_for_pool3_stock_vote": "false",
            "eligible_for_exact_ticker_consensus": "false",
            "pool3_vote_state": "abstain",
            "direction_support_state": "none",
            "attack_gate_open": "false",
            "gate_rule_id": GATE_RULE_ID,
            "gate_reason": reason,
            "selection_reason": reason,
            "pool3_vote_state_reason": "Pool3 abstain；不投 ticker、不支持方向。",
            "blocked_reason": reason,
        }
    )
    return row


def _base_contract(ticker: str, *, peer: dict[str, str]) -> dict[str, Any]:
    is_etf = _is_etf(ticker)
    return {
        "asset_class": "leveraged_etf" if ticker in LEVERAGED_ETF_TICKERS else ("etf" if is_etf else "stock"),
        "is_etf": _bool_text(is_etf),
        "is_leveraged_etf": _bool_text(ticker in LEVERAGED_ETF_TICKERS),
        "style_bucket": _style_bucket(ticker),
        "eligible_for_market_exposure": _bool_text(is_etf),
        "correlation_to_pool1_signal": _correlation_text(ticker, peer.get("pool1", "")),
        "correlation_to_pool2_signal": _correlation_text(ticker, peer.get("pool2", "")),
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


def _peer_market_on(peer: dict[str, str]) -> bool:
    return bool(peer.get("pool1") or peer.get("pool2"))


def _same_as_peer(ticker: str, peer: dict[str, str]) -> bool:
    return ticker in {value for value in peer.values() if value}


def _trend_quality_pass(candidate: dict[str, Any]) -> bool:
    text = " ".join(str(candidate.get(key, "")) for key in ("gate_reason", "selection_reason", "reason"))
    if any(marker in text for marker in ("回撤風險", "過熱", "blocked", "不入選")):
        return False
    return True


def _rank(candidate: dict[str, Any]) -> float:
    value = pd.to_numeric(pd.Series([candidate.get("rank", "")]), errors="coerce").iloc[0]
    return float(value) if pd.notna(value) else 999.0


def _is_etf(ticker: str, asset_type: object = "") -> bool:
    return ticker in ETF_TICKERS or str(asset_type).strip().lower() == "etf"


def _style_bucket(ticker: str) -> str:
    symbol = ticker.split(".")[0]
    if ticker in ETF_TICKERS:
        return "market_exposure"
    if ticker in AI_MAINLINE_TICKERS:
        return "ai_mainline_overlap"
    if symbol.startswith("28"):
        return "financial"
    if symbol.startswith(("12", "13", "20", "22", "26", "29")):
        return "non_tech_largecap"
    if symbol.startswith(("23", "24", "30", "32", "34", "36", "49", "52", "62", "64", "66")):
        return "technology_largecap"
    return "style_complement_stock"


def _correlation_text(ticker: str, peer_ticker: str) -> str:
    if not ticker or not peer_ticker:
        return "unknown"
    return "same_ticker" if ticker == peer_ticker else "different_ticker"


def _changed(original: dict[str, Any], adjusted: dict[str, Any]) -> bool:
    keys = ("top_ticker", "selection_layer", "eligible_for_pool_selection", "pool3_vote_state")
    return any(str(original.get(key, "")) != str(adjusted.get(key, "")) for key in keys)


def _variant_summary(panel: pd.DataFrame, diff_rows: list[dict[str, Any]], *, variant: str) -> dict[str, Any]:
    pool3 = panel[panel["pool_id"].astype(str) == POOL3_ID]
    non_empty_tickers = pool3["top_ticker"].astype(str).str.strip()
    non_empty_tickers = non_empty_tickers[non_empty_tickers.ne("")]
    counts = non_empty_tickers.value_counts()
    top_ticker = str(counts.index[0]) if len(counts) else ""
    top_share = round(float(counts.iloc[0] / len(pool3)), 6) if len(counts) else 0.0
    return {
        "variant": variant,
        "pool3_rows": int(len(pool3)),
        "full_stock_vote_rows": int((pool3["pool3_vote_state"].astype(str) == "full_stock_vote").sum()),
        "direction_support_only_rows": int((pool3["pool3_vote_state"].astype(str) == "direction_support_only").sum()),
        "observation_only_rows": int((pool3["pool3_vote_state"].astype(str) == "observation_only").sum()),
        "abstain_rows": int((pool3["pool3_vote_state"].astype(str) == "abstain").sum()),
        "pool3_eligible_rows": int(pool3["eligible_for_pool_selection"].map(_truthy).sum()),
        "pool3_stock_vote_rows": int(pool3["eligible_for_pool3_stock_vote"].map(_truthy).sum()),
        "pool3_exact_consensus_eligible_rows": int(pool3["eligible_for_exact_ticker_consensus"].map(_truthy).sum()),
        "pool3_etf_stock_vote_rows": int((pool3["is_etf"].map(_truthy) & pool3["eligible_for_pool3_stock_vote"].map(_truthy)).sum()),
        "pool3_etf_exact_consensus_rows": int((pool3["is_etf"].map(_truthy) & pool3["eligible_for_exact_ticker_consensus"].map(_truthy)).sum()),
        "direction_support_only_rate": round(float((pool3["pool3_vote_state"].astype(str) == "direction_support_only").sum() / len(pool3)), 6)
        if len(pool3)
        else 0.0,
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
    parser = argparse.ArgumentParser(description="Build Pool3 vote-state challenger replay panels.")
    parser.add_argument("--replay-panel", required=True)
    parser.add_argument("--top-candidates", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = run_pool3_vote_state_challenger(
        replay_panel_path=args.replay_panel,
        top_candidates_path=args.top_candidates,
        output_dir=args.output_dir,
    )
    print(f"OUTPUT_DIR={output.resolve()}")


if __name__ == "__main__":
    main()
