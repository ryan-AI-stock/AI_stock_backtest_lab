from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


POOL1_ID = "ai_theme_large_cap_v20260613"
POOL2_ID = "tw50_dynamic_constituents_v0"
POOL3_ID = "large_core_bluechip_v0"
HORIZONS = (20, 60, 120)


def run_stock_pool_role_diagnostic(
    *,
    replay_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    current_step = root / "current_step.txt"
    run_log = root / "run_log.csv"
    _write_csv(run_log, [{"event": "started", "detail": str(replay_dir)}])
    current_step.write_text("loading replay inputs\n", encoding="utf-8")

    replay_path = Path(replay_dir) / "stock_pool_replay_panel.csv"
    forward_path = Path(replay_dir) / "stock_pool_replay_forward_returns.csv"
    replay = pd.read_csv(replay_path).fillna("")
    forward = pd.read_csv(forward_path).fillna("")

    current_step.write_text("building vote diagnostics\n", encoding="utf-8")
    vote_rows = _vote_diagnostic_rows(replay, forward)
    vote_frame = pd.DataFrame(vote_rows)
    _write_csv(root / "pool3_vote_diagnostics.csv", vote_rows)

    availability_rows = _availability_rows(replay)
    _write_csv(root / "pool_availability_summary.csv", availability_rows)

    summary_rows = _summary_rows(vote_frame)
    _write_csv(root / "pool3_role_summary.csv", summary_rows)

    minority_rows = _pool3_minority_rows(vote_frame)
    _write_csv(root / "pool3_minority_forward_summary.csv", minority_rows)

    policy_rows = _policy_comparison_rows(vote_frame)
    _write_csv(root / "two_pool_vs_three_pool_policy_summary.csv", policy_rows)

    current_step.write_text("writing markdown\n", encoding="utf-8")
    report = _markdown_report(availability_rows, summary_rows, minority_rows, policy_rows)
    (root / "pool3_role_diagnostic.md").write_text(report, encoding="utf-8")
    metadata = {
        "schema_version": 1,
        "status": "completed",
        "purpose": "diagnose_pool3_role_in_three_pool_consensus",
        "replay_dir": str(replay_path.parent),
        "inputs": {
            "replay_panel": str(replay_path),
            "forward_returns": str(forward_path),
        },
        "outputs": {
            "vote_diagnostics": str(root / "pool3_vote_diagnostics.csv"),
            "pool_availability_summary": str(root / "pool_availability_summary.csv"),
            "role_summary": str(root / "pool3_role_summary.csv"),
            "minority_forward_summary": str(root / "pool3_minority_forward_summary.csv"),
            "policy_summary": str(root / "two_pool_vs_three_pool_policy_summary.csv"),
            "markdown": str(root / "pool3_role_diagnostic.md"),
            "run_log": str(run_log),
        },
        "rows": {
            "vote_diagnostics": len(vote_rows),
            "pool_availability_summary": len(availability_rows),
            "role_summary": len(summary_rows),
            "minority_forward_summary": len(minority_rows),
            "policy_summary": len(policy_rows),
        },
        "pool_ids": {
            "pool1": POOL1_ID,
            "pool2": POOL2_ID,
            "pool3": POOL3_ID,
        },
    }
    (root / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    (root / "completed.txt").write_text("completed\n", encoding="utf-8")
    current_step.write_text("completed\n", encoding="utf-8")
    return metadata


def _vote_diagnostic_rows(replay: pd.DataFrame, forward: pd.DataFrame) -> list[dict[str, Any]]:
    generated = replay[replay["status"] == "generated"].copy()
    rows: list[dict[str, Any]] = []
    for (period, signal_date), group in generated.groupby(["period", "signal_date"], dropna=False):
        by_pool = {str(row["pool_id"]): row for _, row in group.iterrows()}
        pool1 = by_pool.get(POOL1_ID)
        pool2 = by_pool.get(POOL2_ID)
        pool3 = by_pool.get(POOL3_ID)
        if pool1 is None or pool2 is None or pool3 is None:
            continue
        tickers = {
            "pool1": _eligible_ticker(pool1),
            "pool2": _eligible_ticker(pool2),
            "pool3": _eligible_ticker(pool3),
        }
        two_pool_winner = tickers["pool1"] if tickers["pool1"] == tickers["pool2"] else ""
        three_pool_winner = _consensus_winner(tickers.values())
        pool3_matches_pool1 = tickers["pool3"] == tickers["pool1"]
        pool3_matches_pool2 = tickers["pool3"] == tickers["pool2"]
        pool3_solo = not pool3_matches_pool1 and not pool3_matches_pool2
        pool3_breaks_tie = bool(three_pool_winner) and not two_pool_winner and tickers["pool3"] == three_pool_winner
        pool3_warns_two_pool = bool(two_pool_winner) and tickers["pool3"] != two_pool_winner
        for horizon in HORIZONS:
            pool1_return = _forward_return(forward, period, signal_date, POOL1_ID, tickers["pool1"], horizon)
            pool2_return = _forward_return(forward, period, signal_date, POOL2_ID, tickers["pool2"], horizon)
            pool3_return = _forward_return(forward, period, signal_date, POOL3_ID, tickers["pool3"], horizon)
            two_pool_return = (
                _first_ready_return(
                    forward,
                    period,
                    signal_date,
                    [POOL1_ID, POOL2_ID],
                    two_pool_winner,
                    horizon,
                )
                if two_pool_winner
                else None
            )
            three_pool_return = (
                _first_ready_return(
                    forward,
                    period,
                    signal_date,
                    [POOL1_ID, POOL2_ID, POOL3_ID],
                    three_pool_winner,
                    horizon,
                )
                if three_pool_winner
                else None
            )
            rows.append(
                {
                    "period": period,
                    "signal_date": signal_date,
                    "horizon": horizon,
                    "pool1_ticker": tickers["pool1"],
                    "pool2_ticker": tickers["pool2"],
                    "pool3_ticker": tickers["pool3"],
                    "pool1_selection_layer": pool1.get("selection_layer", ""),
                    "pool2_selection_layer": pool2.get("selection_layer", ""),
                    "pool3_selection_layer": pool3.get("selection_layer", ""),
                    "pool1_eligible": _is_eligible(pool1),
                    "pool2_eligible": _is_eligible(pool2),
                    "pool3_eligible": _is_eligible(pool3),
                    "pool3_top_asset_type": pool3.get("top_asset_type", ""),
                    "pool3_matches_pool1": pool3_matches_pool1,
                    "pool3_matches_pool2": pool3_matches_pool2,
                    "pool3_solo": pool3_solo,
                    "pool3_breaks_tie": pool3_breaks_tie,
                    "pool3_warns_two_pool_consensus": pool3_warns_two_pool,
                    "two_pool_winner": two_pool_winner,
                    "three_pool_winner": three_pool_winner,
                    "pool1_forward_return": pool1_return,
                    "pool2_forward_return": pool2_return,
                    "pool3_forward_return": pool3_return,
                    "two_pool_forward_return": two_pool_return,
                    "three_pool_forward_return": three_pool_return,
                    "pool3_minus_pool1": _diff(pool3_return, pool1_return),
                    "pool3_minus_pool2": _diff(pool3_return, pool2_return),
                    "pool3_minus_two_pool": _diff(pool3_return, two_pool_return),
                    "three_pool_minus_two_pool": _diff(three_pool_return, two_pool_return),
                }
            )
    return rows


def _eligible_ticker(row: pd.Series) -> str:
    if not _is_eligible(row):
        return ""
    return str(row.get("top_ticker", "") or "")


def _is_eligible(row: pd.Series) -> bool:
    return str(row.get("eligible_for_pool_selection", "")).lower() in {"true", "1"} and str(row.get("top_ticker", "") or "") != ""


def _availability_rows(replay: pd.DataFrame) -> list[dict[str, Any]]:
    generated = replay[replay["status"] == "generated"].copy()
    if generated.empty:
        return []
    rows: list[dict[str, Any]] = []
    for pool_id, frame in generated.groupby("pool_id"):
        eligible = frame["eligible_for_pool_selection"].astype(str).str.lower().isin({"true", "1"}) & (frame["top_ticker"].astype(str) != "")
        rows.append(
            {
                "pool_id": pool_id,
                "generated_rows": len(frame),
                "eligible_rows": int(eligible.sum()),
                "eligible_rate": round(float(eligible.mean()), 6),
                "no_selection_rows": int((frame["selection_layer"].astype(str) == "no_selection").sum()),
                "market_exposure_tool_rows": int((frame["selection_layer"].astype(str) == "market_exposure_tool").sum()),
                "formal_candidate_rows": int((frame["selection_layer"].astype(str) == "formal_candidate").sum()),
            }
        )
    return rows


def _summary_rows(vote_frame: pd.DataFrame) -> list[dict[str, Any]]:
    if vote_frame.empty:
        return []
    rows: list[dict[str, Any]] = []
    for horizon, frame in vote_frame.groupby("horizon"):
        date_count = len(frame)
        three_pool_avg = _mean_number(frame["three_pool_forward_return"])
        two_pool_avg = _mean_number(frame["two_pool_forward_return"])
        rows.append(
            {
                "horizon": horizon,
                "sample_rows": date_count,
                "pool3_pool1_match_rate": _mean_bool(frame["pool3_matches_pool1"]),
                "pool3_pool2_match_rate": _mean_bool(frame["pool3_matches_pool2"]),
                "pool3_solo_rate": _mean_bool(frame["pool3_solo"]),
                "pool3_breaks_tie_rate": _mean_bool(frame["pool3_breaks_tie"]),
                "pool3_warns_two_pool_rate": _mean_bool(frame["pool3_warns_two_pool_consensus"]),
                "pool3_avg_forward_return": _mean_number(frame["pool3_forward_return"]),
                "pool1_avg_forward_return": _mean_number(frame["pool1_forward_return"]),
                "pool2_avg_forward_return": _mean_number(frame["pool2_forward_return"]),
                "three_pool_avg_forward_return_when_consensus": three_pool_avg,
                "two_pool_avg_forward_return_when_consensus": two_pool_avg,
                "three_pool_minus_two_pool_paired_avg": _mean_number(frame["three_pool_minus_two_pool"]),
                "three_pool_minus_two_pool_unpaired_avg_delta": _diff(three_pool_avg, two_pool_avg),
            }
        )
    return rows


def _pool3_minority_rows(vote_frame: pd.DataFrame) -> list[dict[str, Any]]:
    if vote_frame.empty:
        return []
    rows: list[dict[str, Any]] = []
    for horizon, frame in vote_frame[vote_frame["pool3_warns_two_pool_consensus"]].groupby("horizon"):
        rows.append(
            {
                "horizon": horizon,
                "sample_rows": len(frame),
                "pool3_avg_forward_return": _mean_number(frame["pool3_forward_return"]),
                "two_pool_winner_avg_forward_return": _mean_number(frame["two_pool_forward_return"]),
                "pool3_minus_two_pool_avg": _mean_number(frame["pool3_minus_two_pool"]),
                "pool3_beats_two_pool_rate": _mean_bool(frame["pool3_minus_two_pool"].map(lambda value: _number_or_none(value) is not None and float(value) > 0)),
            }
        )
    return rows


def _policy_comparison_rows(vote_frame: pd.DataFrame) -> list[dict[str, Any]]:
    if vote_frame.empty:
        return []
    rows: list[dict[str, Any]] = []
    for horizon, frame in vote_frame.groupby("horizon"):
        three_pool_avg = _mean_number(frame["three_pool_forward_return"])
        two_pool_avg = _mean_number(frame["two_pool_forward_return"])
        rows.append(
            {
                "horizon": horizon,
                "three_pool_consensus_count": int(frame["three_pool_forward_return"].notna().sum()),
                "two_pool_consensus_count": int(frame["two_pool_forward_return"].notna().sum()),
                "three_pool_avg_forward_return": three_pool_avg,
                "two_pool_avg_forward_return": two_pool_avg,
                "three_pool_minus_two_pool_paired_avg": _mean_number(frame["three_pool_minus_two_pool"]),
                "three_pool_minus_two_pool_unpaired_avg_delta": _diff(three_pool_avg, two_pool_avg),
                "three_pool_outperforms_two_pool_rate": _mean_bool(frame["three_pool_minus_two_pool"].map(lambda value: _number_or_none(value) is not None and float(value) > 0)),
            }
        )
    return rows


def _consensus_winner(tickers: Any) -> str:
    counts: dict[str, int] = {}
    for ticker in tickers:
        text = str(ticker or "")
        if not text:
            continue
        counts[text] = counts.get(text, 0) + 1
    for ticker, count in counts.items():
        if count >= 2:
            return ticker
    return ""


def _forward_return(
    forward: pd.DataFrame,
    period: str,
    signal_date: str,
    pool_id: str,
    ticker: str,
    horizon: int,
) -> float | None:
    rows = forward[
        (forward["period"].astype(str) == str(period))
        & (forward["signal_date"].astype(str) == str(signal_date))
        & (forward["pool_id"].astype(str) == str(pool_id))
        & (forward["ticker"].astype(str) == str(ticker))
        & (forward["horizon"].astype(str) == str(horizon))
        & (forward["forward_status"].astype(str) == "ready")
    ]
    if rows.empty:
        return None
    return _number_or_none(rows.iloc[0].get("forward_return"))


def _first_ready_return(
    forward: pd.DataFrame,
    period: str,
    signal_date: str,
    pool_ids: list[str],
    ticker: str,
    horizon: int,
) -> float | None:
    for pool_id in pool_ids:
        value = _forward_return(forward, period, signal_date, pool_id, ticker, horizon)
        if value is not None:
            return value
    return None


def _diff(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return round(left - right, 8)


def _number_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        if pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean_number(series: pd.Series) -> float | None:
    values = [_number_or_none(value) for value in series]
    clean = [value for value in values if value is not None]
    if not clean:
        return None
    return round(float(sum(clean) / len(clean)), 8)


def _mean_bool(series: pd.Series) -> float:
    if len(series) == 0:
        return 0.0
    return round(float(series.astype(bool).mean()), 6)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def _markdown_report(
    availability_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    minority_rows: list[dict[str, Any]],
    policy_rows: list[dict[str, Any]],
) -> str:
    lines = [
        "# 池3角色診斷",
        "",
        "定位：這是三池表決中的池3作用檢查，用來判斷池3是有效第三視角、warning/veto 層，還是噪音來源。",
        "",
        "## 三池可用率與從缺率",
        "",
        _markdown_table(availability_rows),
        "",
        "## 池3一致率與平均後續表現",
        "",
        _markdown_table(summary_rows),
        "",
        "## 池3少數票時的後續表現",
        "",
        _markdown_table(minority_rows),
        "",
        "## 三池 vs 只看池1+池2",
        "",
        _markdown_table(policy_rows),
        "",
        "使用邊界：本診斷只做歷史 replay 與 forward-return 檢查，不改正式模型，不構成投資建議。",
    ]
    return "\n".join(lines)


def _markdown_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "_無可用樣本。_"
    columns = list(rows[0].keys())
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_format_cell(row.get(column)) for column in columns) + " |")
    return "\n".join(lines)


def _format_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose pool3 role in stock-pool consensus replay outputs.")
    parser.add_argument("--replay-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    metadata = run_stock_pool_role_diagnostic(replay_dir=args.replay_dir, output_dir=args.output_dir)
    print(f"POOL3_ROLE_DIAGNOSTIC={Path(metadata['outputs']['markdown']).resolve()}")


if __name__ == "__main__":
    main()
