from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


POOL3_ID = "large_core_bluechip_v0"
GATE_RULE_ID = "pool3_pure_stock_low_correlation_style_complement_v1"
AI_MAINLINE_TICKERS = {
    "2330.TW",
    "2454.TW",
    "2308.TW",
    "2317.TW",
    "2382.TW",
    "3231.TW",
    "6669.TW",
}
LEVERAGED_ETF_TICKERS = {"00631L.TW"}
ETF_TICKERS = {"0050.TW", "00631L.TW"}
VARIANTS = (
    "pool3_stock_only_style_base",
    "pool3_stock_only_low_correlation",
    "pool3_stock_only_non_ai_largecap",
    "pool3_stock_only_trend_quality",
    "pool3_stock_only_plus_market_exposure_diagnostic",
    "pool3_opportunity_cost_diagnostic_only",
)


def run_pool3_pure_stock_low_correlation_challenger(
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

    diff_path = output / "pool3_pure_stock_low_correlation_decision_diff.csv"
    summary_path = output / "pool3_pure_stock_low_correlation_variant_summary.csv"
    pd.DataFrame(diff_rows).to_csv(diff_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False, encoding="utf-8-sig")
    metadata = {
        "schema_version": 1,
        "task_id": "TASK-BACKTEST-CORE-POOL3-PURE-STOCK-LOW-CORRELATION-CHALLENGER-001",
        "status": "completed",
        "model": "pool3_pure_stock_low_correlation_challenger",
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
        "hard_boundaries": [
            "pool3_formal_vote_target_stock_only",
            "etf_not_eligible_for_pool3_stock_vote",
            "leveraged_etf_not_eligible_for_exact_ticker_consensus",
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
                    "blocked_reason": adjusted.get("blocked_reason", ""),
                    "correlation_to_pool1_signal": adjusted.get("correlation_to_pool1_signal", ""),
                    "correlation_to_pool2_signal": adjusted.get("correlation_to_pool2_signal", ""),
                }
            )
    return panel, diffs


def _ensure_contract_columns(panel: pd.DataFrame) -> None:
    for column in (
        "asset_class",
        "is_etf",
        "is_leveraged_etf",
        "style_bucket",
        "eligible_for_pool3_stock_vote",
        "eligible_for_market_exposure",
        "eligible_for_exact_ticker_consensus",
        "correlation_to_pool1_signal",
        "correlation_to_pool2_signal",
        "concentration_warning",
        "blocked_reason",
        "pool3_pure_stock_gate_pass",
    ):
        if column not in panel.columns:
            panel[column] = ""


def _apply_variant(panel: pd.DataFrame, top_candidates: pd.DataFrame, row: pd.Series, *, variant: str) -> dict[str, Any]:
    adjusted = row.to_dict()
    date = str(row.get("requested_signal_date") or row.get("signal_date") or "")
    period = str(row.get("period") or "")
    peer_votes = _peer_votes(panel, period=period, date=date)
    candidates = _pool3_candidates(top_candidates, period=period, date=date)
    selected = _select_stock_candidate(candidates, peer_votes=peer_votes, variant=variant)
    original_ticker = str(row.get("top_ticker") or "").strip()
    original_is_etf = _is_etf(original_ticker, row.get("top_asset_type", ""))

    if variant == "pool3_opportunity_cost_diagnostic_only":
        return _as_no_vote(
            adjusted,
            ticker=original_ticker,
            reason="Pool3 opportunity-cost diagnostic only；不投正式股票票。",
            peer_votes=peer_votes,
        )

    if selected is None:
        if original_is_etf and variant == "pool3_stock_only_plus_market_exposure_diagnostic":
            return _as_market_exposure_diagnostic(adjusted, original_ticker, peer_votes=peer_votes)
        return _as_no_vote(
            adjusted,
            ticker=original_ticker,
            reason="Pool3 pure-stock gate blocked：無符合條件的純個股候選。",
            peer_votes=peer_votes,
        )

    selected_ticker = str(selected.get("ticker") or "").strip()
    if _is_etf(selected_ticker, selected.get("asset_type", "")):
        return _as_no_vote(
            adjusted,
            ticker=selected_ticker,
            reason="Pool3 pure-stock gate blocked：ETF/槓桿ETF 不可作為 Pool3 正式股票票。",
            peer_votes=peer_votes,
        )
    if variant == "pool3_stock_only_low_correlation" and _same_as_peer(selected_ticker, peer_votes):
        return _as_no_vote(
            adjusted,
            ticker=selected_ticker,
            reason="Pool3 low-correlation gate blocked：候選與池1/池2同標的，無法提供第三專家差異化。",
            peer_votes=peer_votes,
        )

    return _as_stock_vote(adjusted, selected, peer_votes=peer_votes, variant=variant)


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


def _select_stock_candidate(candidates: pd.DataFrame, *, peer_votes: dict[str, str], variant: str) -> dict[str, Any] | None:
    for _, candidate in candidates.iterrows():
        ticker = str(candidate.get("ticker") or "").strip()
        if not ticker or _is_etf(ticker, candidate.get("asset_type", "")):
            continue
        if not _truthy(candidate.get("eligible_for_pool_selection")):
            continue
        if str(candidate.get("selection_layer") or "") != "formal_candidate":
            continue
        if variant == "pool3_stock_only_non_ai_largecap" and ticker in AI_MAINLINE_TICKERS:
            continue
        if variant in {"pool3_stock_only_low_correlation", "pool3_stock_only_plus_market_exposure_diagnostic"}:
            if _same_as_peer(ticker, peer_votes):
                continue
        if variant == "pool3_stock_only_trend_quality" and not _trend_quality_pass(candidate):
            continue
        return candidate.to_dict()
    return None


def _as_stock_vote(row: dict[str, Any], candidate: dict[str, Any], *, peer_votes: dict[str, str], variant: str) -> dict[str, Any]:
    ticker = str(candidate.get("ticker") or "").strip()
    display = str(candidate.get("display") or candidate.get("top_display") or ticker).strip()
    row.update(
        {
            "top_ticker": ticker,
            "top_display": display,
            "top_asset_type": "stock",
            "asset_class": "stock",
            "is_etf": "false",
            "is_leveraged_etf": "false",
            "style_bucket": _style_bucket(ticker),
            "selection_layer": "formal_candidate",
            "eligible_for_pool_selection": "true",
            "eligible_for_pool3_stock_vote": "true",
            "eligible_for_market_exposure": "false",
            "eligible_for_exact_ticker_consensus": "true",
            "attack_gate_open": "true",
            "gate_rule_id": GATE_RULE_ID,
            "gate_reason": f"Pool3 pure-stock {variant}：純個股候選通過；ETF/正二不作正式股票票。",
            "selection_reason": f"Pool3 pure-stock {variant}：純個股候選通過；ETF/正二不作正式股票票。",
            "pool3_pure_stock_gate_pass": "true",
            "correlation_to_pool1_signal": _correlation_text(ticker, peer_votes.get("pool1", "")),
            "correlation_to_pool2_signal": _correlation_text(ticker, peer_votes.get("pool2", "")),
            "concentration_warning": "",
            "blocked_reason": "",
        }
    )
    return row


def _as_market_exposure_diagnostic(row: dict[str, Any], ticker: str, *, peer_votes: dict[str, str]) -> dict[str, Any]:
    row.update(
        {
            "top_ticker": ticker,
            "top_asset_type": "etf",
            "asset_class": "leveraged_etf" if ticker in LEVERAGED_ETF_TICKERS else "etf",
            "is_etf": "true",
            "is_leveraged_etf": _bool_text(ticker in LEVERAGED_ETF_TICKERS),
            "style_bucket": "market_exposure",
            "selection_layer": "market_exposure_tool",
            "eligible_for_pool_selection": "false",
            "eligible_for_pool3_stock_vote": "false",
            "eligible_for_market_exposure": "true",
            "eligible_for_exact_ticker_consensus": "false",
            "attack_gate_open": "false",
            "gate_rule_id": GATE_RULE_ID,
            "gate_reason": "Pool3 pure-stock：ETF/正二僅保留為市場曝險診斷，不投 Pool3 正式股票票。",
            "selection_reason": "Pool3 pure-stock：ETF/正二僅保留為市場曝險診斷，不投 Pool3 正式股票票。",
            "pool3_pure_stock_gate_pass": "false",
            "correlation_to_pool1_signal": _correlation_text(ticker, peer_votes.get("pool1", "")),
            "correlation_to_pool2_signal": _correlation_text(ticker, peer_votes.get("pool2", "")),
            "concentration_warning": "market_exposure_diagnostic_only",
            "blocked_reason": "ETF/leveraged ETF excluded from Pool3 stock vote",
        }
    )
    return row


def _as_no_vote(row: dict[str, Any], *, ticker: str, reason: str, peer_votes: dict[str, str]) -> dict[str, Any]:
    row.update(
        {
            "asset_class": "leveraged_etf" if ticker in LEVERAGED_ETF_TICKERS else ("etf" if ticker in ETF_TICKERS else "stock"),
            "is_etf": _bool_text(ticker in ETF_TICKERS),
            "is_leveraged_etf": _bool_text(ticker in LEVERAGED_ETF_TICKERS),
            "style_bucket": _style_bucket(ticker),
            "selection_layer": "observation_only",
            "eligible_for_pool_selection": "false",
            "eligible_for_pool3_stock_vote": "false",
            "eligible_for_market_exposure": _bool_text(ticker in ETF_TICKERS),
            "eligible_for_exact_ticker_consensus": "false",
            "attack_gate_open": "false",
            "gate_rule_id": GATE_RULE_ID,
            "gate_reason": reason,
            "selection_reason": reason,
            "pool3_pure_stock_gate_pass": "false",
            "correlation_to_pool1_signal": _correlation_text(ticker, peer_votes.get("pool1", "")),
            "correlation_to_pool2_signal": _correlation_text(ticker, peer_votes.get("pool2", "")),
            "concentration_warning": "not_applicable_no_vote",
            "blocked_reason": reason,
        }
    )
    return row


def _peer_votes(panel: pd.DataFrame, *, period: str, date: str) -> dict[str, str]:
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


def _trend_quality_pass(candidate: dict[str, Any]) -> bool:
    rank = pd.to_numeric(pd.Series([candidate.get("rank", "")]), errors="coerce").iloc[0]
    if pd.notna(rank) and float(rank) > 3:
        return False
    text = " ".join(str(candidate.get(key, "")) for key in ("gate_reason", "selection_reason", "reason"))
    if any(marker in text for marker in ("回撤風險", "過熱", "blocked", "不入選")):
        return False
    return True


def _is_etf(ticker: str, asset_type: object = "") -> bool:
    return ticker in ETF_TICKERS or str(asset_type).strip().lower() == "etf"


def _same_as_peer(ticker: str, peer_votes: dict[str, str]) -> bool:
    return ticker in {value for value in peer_votes.values() if value}


def _style_bucket(ticker: str) -> str:
    symbol = ticker.split(".")[0]
    if ticker in ETF_TICKERS:
        return "market_exposure"
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
    keys = ("top_ticker", "selection_layer", "eligible_for_pool_selection", "eligible_for_exact_ticker_consensus")
    return any(str(original.get(key, "")) != str(adjusted.get(key, "")) for key in keys)


def _variant_summary(panel: pd.DataFrame, diff_rows: list[dict[str, Any]], *, variant: str) -> dict[str, Any]:
    pool3 = panel[panel["pool_id"].astype(str) == POOL3_ID]
    non_empty_tickers = pool3["top_ticker"].astype(str).str.strip()
    non_empty_tickers = non_empty_tickers[non_empty_tickers.ne("")]
    if pool3.empty or non_empty_tickers.empty:
        top_share = 0.0
        top_ticker = ""
    else:
        counts = non_empty_tickers.value_counts()
        top_ticker = str(counts.index[0]) if len(counts) else ""
        top_share = round(float(counts.iloc[0] / len(pool3)), 6) if len(counts) else 0.0
    return {
        "variant": variant,
        "pool3_rows": int(len(pool3)),
        "pool3_eligible_rows": int(pool3["eligible_for_pool_selection"].map(_truthy).sum()),
        "pool3_stock_vote_rows": int(pool3["eligible_for_pool3_stock_vote"].map(_truthy).sum()),
        "pool3_market_exposure_diagnostic_rows": int(pool3["eligible_for_market_exposure"].map(_truthy).sum()),
        "pool3_exact_consensus_eligible_rows": int(pool3["eligible_for_exact_ticker_consensus"].map(_truthy).sum()),
        "pool3_etf_rows": int(pool3["is_etf"].map(_truthy).sum()),
        "pool3_leveraged_etf_rows": int(pool3["is_leveraged_etf"].map(_truthy).sum()),
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
    parser = argparse.ArgumentParser(description="Build Pool3 pure-stock low-correlation challenger replay panels.")
    parser.add_argument("--replay-panel", required=True)
    parser.add_argument("--top-candidates", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = run_pool3_pure_stock_low_correlation_challenger(
        replay_panel_path=args.replay_panel,
        top_candidates_path=args.top_candidates,
        output_dir=args.output_dir,
    )
    print(f"OUTPUT_DIR={output.resolve()}")


if __name__ == "__main__":
    main()
