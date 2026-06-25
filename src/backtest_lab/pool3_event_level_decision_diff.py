from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


POOL1_FRAGMENT = "ai_theme_large_cap"
POOL2_FRAGMENT = "tw50_dynamic_constituents"
POOL3_FRAGMENT = "large_core_bluechip"
ETF_TICKERS = {"0050.TW", "00631L.TW"}
FORWARD_HORIZONS = (20, 60, 120)


def run_pool3_event_level_decision_diff(
    *,
    replay_panel_path: str | Path,
    formal_decision_panel_path: str | Path | None,
    price_cache_dir: str | Path,
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

    log("load_inputs", "started")
    replay = pd.read_csv(replay_panel_path).fillna("")
    _validate_replay(replay)
    formal = _load_formal_panel(formal_decision_panel_path)

    log("build_event_panel", "started")
    event_panel = _build_event_panel(replay, formal)
    prices = _load_prices(_needed_tickers(event_panel), Path(price_cache_dir))
    event_panel = _add_forward_returns(event_panel, prices)
    event_panel.to_csv(output / "event_decision_diff_panel.csv", index=False, encoding="utf-8-sig")

    log("build_summaries", "started")
    _pool3_vote_blocker_summary(event_panel).to_csv(output / "pool3_vote_blocker_summary.csv", index=False, encoding="utf-8-sig")
    _hard_gate_summary(event_panel).to_csv(output / "hard_gate_2023_2024_rootcause.csv", index=False, encoding="utf-8-sig")
    _pool1_pool2_state_summary(event_panel).to_csv(output / "pool1_pool2_state_when_pool3_votes.csv", index=False, encoding="utf-8-sig")
    _exact_vs_direction_blockers(event_panel).to_csv(output / "exact_vs_direction_consensus_blockers.csv", index=False, encoding="utf-8-sig")
    _formal_target_selector_trace(event_panel).to_csv(output / "formal_target_selector_trace.csv", index=False, encoding="utf-8-sig")
    _missed_opportunity_forward_returns(event_panel).to_csv(output / "missed_opportunity_forward_returns.csv", index=False, encoding="utf-8-sig")
    (output / "event_diff_final_summary_zh.md").write_text(_markdown_summary(event_panel), encoding="utf-8")

    metadata = {
        "schema_version": 1,
        "task_id": "TASK-BACKTEST-CORE-POOL3-EVENT-LEVEL-DECISION-DIFF-001",
        "status": "completed",
        "model": "pool3_event_level_decision_diff",
        "formal_model_changed": False,
        "trade_decision_changed": False,
        "active_in_trade_decision": False,
        "replay_panel_path": str(replay_panel_path),
        "formal_decision_panel_path": str(formal_decision_panel_path or ""),
        "price_cache_dir": str(price_cache_dir),
        "outputs": {
            "event_panel": "event_decision_diff_panel.csv",
            "blocker_summary": "pool3_vote_blocker_summary.csv",
            "hard_gate_rootcause": "hard_gate_2023_2024_rootcause.csv",
            "pool1_pool2_state": "pool1_pool2_state_when_pool3_votes.csv",
            "exact_vs_direction": "exact_vs_direction_consensus_blockers.csv",
            "formal_target_trace": "formal_target_selector_trace.csv",
            "missed_forward_returns": "missed_opportunity_forward_returns.csv",
            "summary": "event_diff_final_summary_zh.md",
        },
    }
    (output / "manifest.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame([{"status": "completed", "output_dir": str(output.resolve())}]).to_csv(
        output / "completed.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(columns=["step", "error"]).to_csv(output / "failed.csv", index=False, encoding="utf-8-sig")
    log("completed", "completed", str(output.resolve()))
    (output / "current_step.txt").write_text("completed\n", encoding="utf-8")
    return output


def _validate_replay(replay: pd.DataFrame) -> None:
    required = {"period", "requested_signal_date", "pool_id", "top_ticker", "selection_layer", "eligible_for_pool_selection"}
    missing = required - set(replay.columns)
    if missing:
        raise ValueError("missing replay columns: " + ",".join(sorted(missing)))


def _load_formal_panel(path: str | Path | None) -> pd.DataFrame:
    if not path:
        return pd.DataFrame()
    formal_path = Path(path)
    if not formal_path.exists():
        raise FileNotFoundError(f"formal decision panel not found: {formal_path}")
    return pd.read_csv(formal_path).fillna("")


def _build_event_panel(replay: pd.DataFrame, formal: pd.DataFrame) -> pd.DataFrame:
    formal_by_date = {}
    if not formal.empty and "date" in formal.columns:
        formal_by_date = {
            str(row["date"]): row.to_dict()
            for _, row in formal.iterrows()
        }
    rows: list[dict[str, Any]] = []
    frame = replay.copy()
    frame["signal_date"] = frame["signal_date"].astype(str)
    frame = frame[frame["signal_date"].str.strip().ne("")]
    frame = frame[pd.to_datetime(frame["signal_date"], errors="coerce").notna()]
    for (period, signal_date), group in frame.groupby(["period", "signal_date"], dropna=False):
        pools = {
            "pool1": _pool_state(group, POOL1_FRAGMENT),
            "pool2": _pool_state(group, POOL2_FRAGMENT),
            "pool3": _pool_state(group, POOL3_FRAGMENT),
        }
        votes = [state["ticker"] for state in pools.values() if state["eligible"] and state["ticker"]]
        exact_state, exact_winner, exact_count = _exact_consensus(votes)
        directions = [state["direction"] for state in pools.values() if state["direction_eligible"] and state["direction"]]
        direction_state, direction_group, direction_count = _direction_consensus(directions)
        formal_row = formal_by_date.get(str(signal_date), {})
        final_target = str(formal_row.get("winner_ticker") or exact_winner or "").strip()
        consensus_state = str(formal_row.get("consensus_state") or exact_state)
        action = str(formal_row.get("action") or "")
        trade_executed = bool(action and action not in {"hold", "no_action"})
        blocker = _pool3_blocker_category(pools, exact_state, exact_winner, direction_state, final_target, trade_executed)
        rows.append(
            {
                "period": period,
                "signal_date": str(signal_date),
                "pool1_ticker": pools["pool1"]["ticker"],
                "pool1_selection_layer": pools["pool1"]["selection_layer"],
                "pool1_vote_state": pools["pool1"]["vote_state"],
                "pool1_direction_state": pools["pool1"]["direction"],
                "pool1_blocked_reason": pools["pool1"]["blocked_reason"],
                "pool2_ticker": pools["pool2"]["ticker"],
                "pool2_selection_layer": pools["pool2"]["selection_layer"],
                "pool2_vote_state": pools["pool2"]["vote_state"],
                "pool2_direction_state": pools["pool2"]["direction"],
                "pool2_blocked_reason": pools["pool2"]["blocked_reason"],
                "pool3_ticker": pools["pool3"]["ticker"],
                "pool3_selection_layer": pools["pool3"]["selection_layer"],
                "pool3_vote_state": pools["pool3"]["vote_state"],
                "pool3_direction_state": pools["pool3"]["direction"],
                "pool3_blocked_reason": pools["pool3"]["blocked_reason"],
                "pool3_has_full_stock_vote": pools["pool3"]["eligible"] and pools["pool3"]["asset_type"] == "stock",
                "pool3_matches_pool1_ticker": pools["pool3"]["ticker"] and pools["pool3"]["ticker"] == pools["pool1"]["ticker"],
                "pool3_matches_pool2_ticker": pools["pool3"]["ticker"] and pools["pool3"]["ticker"] == pools["pool2"]["ticker"],
                "pool3_matches_pool1_direction": pools["pool3"]["direction"] and pools["pool3"]["direction"] == pools["pool1"]["direction"],
                "pool3_matches_pool2_direction": pools["pool3"]["direction"] and pools["pool3"]["direction"] == pools["pool2"]["direction"],
                "exact_ticker_consensus": exact_state,
                "exact_ticker_consensus_group": exact_winner,
                "exact_ticker_consensus_count": exact_count,
                "direction_consensus": direction_state,
                "direction_consensus_group": direction_group,
                "direction_consensus_count": direction_count,
                "raw_consensus_state": consensus_state,
                "actionable_decision_state": "actionable" if final_target else "no_final_target",
                "formal_final_target": final_target,
                "final_target_source": _final_target_source(final_target, pools, exact_winner),
                "trade_action": action or ("hold" if final_target else "no_trade"),
                "trade_executed": trade_executed,
                "trade_blocked_reason": "" if trade_executed else _trade_blocked_reason(final_target, exact_state),
                "pool3_ignored_reason": blocker,
                "pool3_blocker_category": blocker,
            }
        )
    return pd.DataFrame(rows).sort_values("signal_date").reset_index(drop=True)


def _pool_state(group: pd.DataFrame, fragment: str) -> dict[str, Any]:
    rows = group[group["pool_id"].astype(str).str.contains(fragment, na=False)].copy()
    if rows.empty:
        return _empty_pool_state("pool_missing")
    eligible_rows = rows[rows["eligible_for_pool_selection"].map(_truthy)]
    row = eligible_rows.iloc[0] if not eligible_rows.empty else rows.iloc[0]
    ticker = str(row.get("top_ticker") or "").strip()
    selection_layer = str(row.get("selection_layer") or "").strip()
    asset_type = _asset_type(ticker, row.get("top_asset_type", ""))
    eligible = bool(_truthy(row.get("eligible_for_pool_selection")) and ticker)
    direction = _direction(asset_type=asset_type, selection_layer=selection_layer, eligible=eligible, ticker=ticker)
    return {
        "ticker": ticker,
        "selection_layer": selection_layer,
        "vote_state": "eligible_vote" if eligible else selection_layer or "no_vote",
        "asset_type": asset_type,
        "eligible": eligible,
        "direction": direction,
        "direction_eligible": bool(direction),
        "blocked_reason": "" if eligible else str(row.get("blocked_reason") or row.get("selection_reason") or row.get("reason") or ""),
    }


def _empty_pool_state(reason: str) -> dict[str, Any]:
    return {
        "ticker": "",
        "selection_layer": "no_selection",
        "vote_state": "no_vote",
        "asset_type": "",
        "eligible": False,
        "direction": "",
        "direction_eligible": False,
        "blocked_reason": reason,
    }


def _exact_consensus(votes: list[str]) -> tuple[str, str, int]:
    if not votes:
        return "no_vote", "", 0
    if len(votes) < 2:
        return "insufficient_votes", "", 1
    ticker, count = Counter(votes).most_common(1)[0]
    if count >= 2:
        return "consensus", ticker, int(count)
    return "divergent", "", int(count)


def _direction_consensus(directions: list[str]) -> tuple[str, str, int]:
    if not directions:
        return "no_direction", "", 0
    if len(directions) < 2:
        return "insufficient_direction", "", 1
    direction, count = Counter(directions).most_common(1)[0]
    if count >= 2:
        return "direction_consensus", direction, int(count)
    return "direction_divergent", "", int(count)


def _pool3_blocker_category(
    pools: dict[str, dict[str, Any]],
    exact_state: str,
    exact_winner: str,
    direction_state: str,
    final_target: str,
    trade_executed: bool,
) -> str:
    pool3 = pools["pool3"]
    if not pool3["eligible"]:
        return "data_or_candidate_blocked"
    if exact_winner and exact_winner == pool3["ticker"]:
        return "pool3_part_of_exact_consensus"
    if final_target and final_target != pool3["ticker"]:
        return "formal_target_selector_preferred_other_pool"
    if exact_state != "consensus":
        if not pools["pool1"]["eligible"]:
            return "pool1_no_vote_or_risk_off"
        if not pools["pool2"]["eligible"]:
            return "pool2_no_vote_or_risk_off"
        return "exact_consensus_missing"
    if direction_state not in {"direction_consensus"}:
        return "direction_consensus_missing"
    if final_target and not trade_executed:
        return "trade_execution_not_triggered"
    return "unknown_needs_trace"


def _final_target_source(final_target: str, pools: dict[str, dict[str, Any]], exact_winner: str) -> str:
    if not final_target:
        return "none"
    if exact_winner and final_target == exact_winner:
        return "exact_ticker_consensus"
    for name, state in pools.items():
        if final_target == state["ticker"]:
            return name
    if final_target in ETF_TICKERS:
        return "market_exposure_override"
    return "unknown"


def _trade_blocked_reason(final_target: str, exact_state: str) -> str:
    if not final_target:
        return "no_formal_final_target"
    if exact_state != "consensus":
        return "no_exact_consensus"
    return "hold_existing_or_missing_price"


def _add_forward_returns(event_panel: pd.DataFrame, prices: dict[str, pd.Series]) -> pd.DataFrame:
    panel = event_panel.copy()
    for label in ("pool3_ticker", "formal_final_target"):
        for horizon in FORWARD_HORIZONS:
            panel[f"{label}_forward_{horizon}d_return"] = [
                _forward_return(prices, str(row.get(label) or ""), str(row.get("signal_date") or ""), horizon)
                for row in panel.to_dict(orient="records")
            ]
    for ticker in ("0050.TW", "00631L.TW"):
        label = ticker.replace(".", "_")
        for horizon in FORWARD_HORIZONS:
            panel[f"{label}_forward_{horizon}d_return"] = [
                _forward_return(prices, ticker, str(row.get("signal_date") or ""), horizon)
                for row in panel.to_dict(orient="records")
            ]
    return panel


def _needed_tickers(event_panel: pd.DataFrame) -> list[str]:
    tickers = {"0050.TW", "00631L.TW"}
    for column in ("pool3_ticker", "formal_final_target"):
        if column in event_panel.columns:
            tickers.update(str(value).strip() for value in event_panel[column].dropna().tolist() if str(value).strip())
    return sorted(tickers)


def _load_prices(tickers: list[str], cache_dir: Path) -> dict[str, pd.Series]:
    prices: dict[str, pd.Series] = {}
    for ticker in tickers:
        path = cache_dir / f"{ticker.replace('.', '_')}.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        if "date" not in frame.columns or "adj_close" not in frame.columns:
            continue
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        series = pd.Series(pd.to_numeric(frame["adj_close"], errors="coerce").values, index=frame["date"])
        series = series.dropna().sort_index()
        if not series.empty:
            prices[ticker] = series
    return prices


def _forward_return(prices: dict[str, pd.Series], ticker: str, signal_date: str, horizon: int) -> float | None:
    series = prices.get(ticker)
    if series is None or series.empty or not ticker:
        return None
    date = pd.Timestamp(signal_date)
    future = series.loc[series.index >= date]
    if len(future) <= horizon:
        return None
    start = float(future.iloc[0])
    end = float(future.iloc[horizon])
    if not start:
        return None
    return round(end / start - 1, 8)


def _pool3_vote_blocker_summary(panel: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    subset = panel[panel["pool3_has_full_stock_vote"].map(_truthy)]
    for period, frame in subset.groupby("period", dropna=False):
        counts = frame["pool3_blocker_category"].value_counts()
        for category, count in counts.items():
            rows.append(
                {
                    "period": period,
                    "pool3_blocker_category": category,
                    "rows": int(count),
                    "share": round(float(count / len(frame)), 6) if len(frame) else 0.0,
                }
            )
    return pd.DataFrame(rows)


def _hard_gate_summary(panel: pd.DataFrame) -> pd.DataFrame:
    hard = panel[panel["period"].astype(str).isin({"2023", "2024_now"})].copy()
    hard = hard[hard["pool3_has_full_stock_vote"].map(_truthy)]
    rows: list[dict[str, Any]] = []
    for period, frame in hard.groupby("period", dropna=False):
        rows.append(
            {
                "period": period,
                "pool3_full_stock_vote_days": int(len(frame)),
                "exact_consensus_days": int((frame["exact_ticker_consensus"] == "consensus").sum()),
                "direction_consensus_days": int((frame["direction_consensus"] == "direction_consensus").sum()),
                "formal_final_target_days": int(frame["formal_final_target"].astype(str).str.strip().ne("").sum()),
                "trade_executed_days": int(frame["trade_executed"].map(_truthy).sum()),
                "top_blocker": str(frame["pool3_blocker_category"].value_counts().index[0]) if not frame.empty else "",
            }
        )
    return pd.DataFrame(rows)


def _pool1_pool2_state_summary(panel: pd.DataFrame) -> pd.DataFrame:
    subset = panel[panel["pool3_has_full_stock_vote"].map(_truthy)]
    return (
        subset.groupby(["period", "pool1_vote_state", "pool2_vote_state"], dropna=False)
        .size()
        .reset_index(name="rows")
        .sort_values(["period", "rows"], ascending=[True, False])
    )


def _exact_vs_direction_blockers(panel: pd.DataFrame) -> pd.DataFrame:
    subset = panel[panel["pool3_has_full_stock_vote"].map(_truthy)]
    return (
        subset.groupby(["period", "exact_ticker_consensus", "direction_consensus"], dropna=False)
        .size()
        .reset_index(name="rows")
        .sort_values(["period", "rows"], ascending=[True, False])
    )


def _formal_target_selector_trace(panel: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "period",
        "signal_date",
        "pool1_ticker",
        "pool2_ticker",
        "pool3_ticker",
        "exact_ticker_consensus",
        "exact_ticker_consensus_group",
        "direction_consensus",
        "direction_consensus_group",
        "formal_final_target",
        "final_target_source",
        "trade_action",
        "trade_executed",
        "pool3_blocker_category",
    ]
    return panel[columns].copy()


def _missed_opportunity_forward_returns(panel: pd.DataFrame) -> pd.DataFrame:
    subset = panel[
        panel["pool3_has_full_stock_vote"].map(_truthy)
        & (panel["formal_final_target"].astype(str) != panel["pool3_ticker"].astype(str))
    ].copy()
    keep = [
        "period",
        "signal_date",
        "pool3_ticker",
        "formal_final_target",
        "pool3_blocker_category",
    ]
    for horizon in FORWARD_HORIZONS:
        keep.extend(
            [
                f"pool3_ticker_forward_{horizon}d_return",
                f"formal_final_target_forward_{horizon}d_return",
                f"0050_TW_forward_{horizon}d_return",
                f"00631L_TW_forward_{horizon}d_return",
            ]
        )
    return subset[keep].copy()


def _markdown_summary(panel: pd.DataFrame) -> str:
    pool3 = panel[panel["pool3_has_full_stock_vote"].map(_truthy)]
    lines = [
        "# Pool3 event-level decision diff",
        "",
        "- 狀態：diagnostic only；正式模型未變更。",
        f"- 全期間日期數：{len(panel)}",
        f"- Pool3 full_stock_vote 日期數：{len(pool3)}",
        "",
        "## Pool3 有票時的主要 blocker",
        "",
    ]
    if pool3.empty:
        lines.append("- 無 Pool3 full_stock_vote rows。")
    else:
        counts = pool3["pool3_blocker_category"].value_counts()
        for category, count in counts.items():
            lines.append(f"- {category}: {int(count)} ({count / len(pool3):.2%})")
    lines.extend(
        [
            "",
            "## 使用邊界",
            "",
            "本輸出只用於 Research / Experiments 判讀三池決策鏈，不是正式交易決策，也不改三池表決規則。",
        ]
    )
    return "\n".join(lines)


def _direction(*, asset_type: str, selection_layer: str, eligible: bool, ticker: str) -> str:
    if not eligible or not ticker:
        return ""
    if asset_type == "stock":
        return "stock_attack"
    if asset_type in {"etf", "leveraged_etf"}:
        return "market_exposure"
    if selection_layer == "direction_support_only":
        return "stock_attack"
    return "observation"


def _asset_type(ticker: str, asset_type: object = "") -> str:
    if ticker in ETF_TICKERS:
        return "leveraged_etf" if ticker == "00631L.TW" else "etf"
    text = str(asset_type).strip().lower()
    if text == "etf":
        return "etf"
    return "stock" if ticker else ""


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Pool3 event-level decision chain diagnostics.")
    parser.add_argument("--replay-panel", required=True)
    parser.add_argument("--formal-decision-panel", default="")
    parser.add_argument("--price-cache-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = run_pool3_event_level_decision_diff(
        replay_panel_path=args.replay_panel,
        formal_decision_panel_path=args.formal_decision_panel or None,
        price_cache_dir=args.price_cache_dir,
        output_dir=args.output_dir,
    )
    print(f"OUTPUT_DIR={output.resolve()}")


if __name__ == "__main__":
    main()
