from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from backtest_lab.current_formal_pool1_pool2_signal_panels import (
    DEFAULT_PRICE_CACHE_DIR,
    DEFAULT_PRICE_SOURCE_REGISTRY,
    _load_price_source,
    _load_price_source_registry,
)


TASK_ID = "TASK-BACKTEST-CORE-2015-2021-CANDIDATE-RANKING-SCORE-CONTRACT-001"
DEFAULT_LIFECYCLE_DIR = "outputs/pool1_ticker_lifecycle_contract_201411_202112_20260702"
DEFAULT_OUTPUT_DIR = "outputs/candidate_ranking_score_contract_2015_2021_20260703"
DEFAULT_START_DATE = "2015-01-01"
DEFAULT_END_DATE = "2021-12-30"

WINDOWS = (10, 20, 30, 40, 60)
VARIANTS = {
    "10_30": (10, 30),
    "10_40": (10, 40),
    "20_60": (20, 60),
}


def run_candidate_ranking_score_contract_2015_2021(
    *,
    lifecycle_dir: str | Path = DEFAULT_LIFECYCLE_DIR,
    price_cache_dir: str | Path = DEFAULT_PRICE_CACHE_DIR,
    price_source_registry: str | Path = DEFAULT_PRICE_SOURCE_REGISTRY,
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
        (output / "current_step.txt").write_text(f"{step}:{status}\n{detail}", encoding="utf-8")

    try:
        log("load_inputs", "started", str(lifecycle_dir))
        lifecycle_root = Path(lifecycle_dir)
        lifecycle_daily = pd.read_csv(lifecycle_root / "pool1_date_aware_candidate_availability_daily.csv").fillna("")
        lifecycle_contract = pd.read_csv(lifecycle_root / "pool1_ticker_lifecycle_contract.csv").fillna("")

        start = pd.Timestamp(start_date).normalize()
        end = pd.Timestamp(end_date).normalize()
        lifecycle_daily["date"] = pd.to_datetime(lifecycle_daily["date"]).dt.normalize()
        lifecycle_daily = lifecycle_daily[(lifecycle_daily["date"] >= start) & (lifecycle_daily["date"] <= end)].copy()
        if lifecycle_daily.empty:
            raise ValueError(f"No lifecycle rows in requested range {start_date}..{end_date}")

        tickers = sorted(lifecycle_daily["ticker"].astype(str).unique())
        registry = _load_price_source_registry(price_source_registry)
        prices_by_ticker: dict[str, pd.DataFrame] = {}
        price_meta: dict[str, dict[str, Any]] = {}
        log("load_price_sources", "started", f"tickers={len(tickers)}")
        for ticker in tickers:
            frame, meta = _load_price_source(ticker, price_cache_dir=price_cache_dir, registry=registry)
            if frame is not None:
                prices_by_ticker[ticker] = frame
            price_meta[ticker] = meta

        log("build_panel", "started", "")
        panel = _build_candidate_panel(lifecycle_daily, lifecycle_contract, prices_by_ticker, price_meta)
        panel = _attach_rankings(panel)

        readiness = _data_readiness_by_date(panel)
        blocked = panel[~panel["data_ready"].astype(bool)].copy()
        summary = _candidate_ranking_summary(panel, readiness)
        turnover = _top_name_turnover_summary(panel, readiness)
        margins = _score_margin_distribution(panel, readiness)

        log("write_outputs", "started", str(output))
        panel.to_csv(output / "candidate_ranking_panel_2015_2021.csv", index=False, encoding="utf-8-sig")
        summary.to_csv(output / "candidate_ranking_summary.csv", index=False, encoding="utf-8-sig")
        readiness.to_csv(output / "data_readiness_by_date.csv", index=False, encoding="utf-8-sig")
        blocked.to_csv(output / "blocked_rows.csv", index=False, encoding="utf-8-sig")
        turnover.to_csv(output / "top_name_turnover_summary.csv", index=False, encoding="utf-8-sig")
        margins.to_csv(output / "score_margin_distribution.csv", index=False, encoding="utf-8-sig")
        (output / "next_experiments_task.md").write_text(_next_experiments_task(output), encoding="utf-8")
        (output / "final_summary_zh.md").write_text(_final_summary(summary, readiness, blocked), encoding="utf-8")

        manifest = _manifest(
            output=output,
            panel=panel,
            readiness=readiness,
            blocked=blocked,
            start_date=start_date,
            end_date=end_date,
            lifecycle_dir=lifecycle_dir,
            price_cache_dir=price_cache_dir,
            price_source_registry=price_source_registry,
        )
        (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        pd.DataFrame([{"step": TASK_ID, "status": "completed_diagnostic_only"}]).to_csv(
            output / "completed.csv",
            index=False,
            encoding="utf-8-sig",
        )
        pd.DataFrame(columns=["step", "status", "reason"]).to_csv(output / "failed.csv", index=False, encoding="utf-8-sig")
        log("completed", "completed", str(output.resolve()))
        (output / "current_step.txt").write_text("completed\n", encoding="utf-8")
        return output
    except Exception as exc:
        pd.DataFrame([{"step": TASK_ID, "status": "failed", "reason": str(exc)}]).to_csv(
            output / "failed.csv",
            index=False,
            encoding="utf-8-sig",
        )
        log("failed", "failed", str(exc))
        raise


def _build_candidate_panel(
    lifecycle_daily: pd.DataFrame,
    lifecycle_contract: pd.DataFrame,
    prices_by_ticker: dict[str, pd.DataFrame],
    price_meta: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    names = {
        str(row.get("ticker", "")): str(row.get("name", ""))
        for row in lifecycle_contract.to_dict(orient="records")
    }
    rows: list[dict[str, Any]] = []
    for item in lifecycle_daily.sort_values(["date", "ticker"]).to_dict(orient="records"):
        date = pd.Timestamp(item["date"]).normalize()
        ticker = str(item.get("ticker") or "")
        price_frame = prices_by_ticker.get(ticker)
        price_available = _price_available(price_frame, date)
        lifecycle_available = _bool(item.get("has_valid_price_on_date")) and price_available
        formal_warmup_available = _bool(item.get("candidate_available_for_pool1_ranking"))
        blocked_reasons: list[str] = []
        if not lifecycle_available:
            excluded = str(item.get("excluded_reason") or "").strip()
            blocked_reasons.append(excluded or "price_or_lifecycle_not_available")
        returns = _window_returns(price_frame, date) if lifecycle_available else {window: None for window in WINDOWS}
        for window in WINDOWS:
            if lifecycle_available and returns.get(window) is None:
                blocked_reasons.append(f"insufficient_{window}d_history")
        variant_scores = {
            variant: _variant_score(returns.get(short), returns.get(long))
            for variant, (short, long) in VARIANTS.items()
        }
        for variant, score in variant_scores.items():
            if lifecycle_available and score is None:
                blocked_reasons.append(f"{variant}_score_not_ready")
        meta = price_meta.get(ticker, {})
        rows.append(
            {
                "date": date.strftime("%Y-%m-%d"),
                "candidate_ticker": ticker,
                "candidate_name": names.get(ticker, ticker.replace(".TW", "")),
                "candidate_source": "pool1_date_aware_lifecycle_universe",
                "is_candidate_available": lifecycle_available,
                "pool1_formal_60d_warmup_available": formal_warmup_available,
                "price_available": price_available,
                "rs_10d": _round_or_blank(returns.get(10)),
                "rs_20d": _round_or_blank(returns.get(20)),
                "rs_30d": _round_or_blank(returns.get(30)),
                "rs_40d": _round_or_blank(returns.get(40)),
                "rs_60d": _round_or_blank(returns.get(60)),
                "rank_score_10_30": _round_or_blank(variant_scores["10_30"]),
                "rank_score_10_40": _round_or_blank(variant_scores["10_40"]),
                "rank_score_20_60": _round_or_blank(variant_scores["20_60"]),
                "rank_10_30": "",
                "rank_10_40": "",
                "rank_20_60": "",
                "top1_10_30": "",
                "top1_10_40": "",
                "top1_20_60": "",
                "top3_10_30": "",
                "top3_10_40": "",
                "top3_20_60": "",
                "score_margin_top1_top2_10_30": "",
                "score_margin_top1_top2_10_40": "",
                "score_margin_top1_top2_20_60": "",
                "candidate_count": 0,
                "data_ready": lifecycle_available and all(score is not None for score in variant_scores.values()),
                "blocked_reason": ";".join(dict.fromkeys(reason for reason in blocked_reasons if reason)),
                "adjusted_close_available": bool(meta.get("adjusted_close_available", True)),
                "synthetic_used": _bool(item.get("synthetic_used")),
                "contract_boundary": "diagnostic_only",
                "active_in_trade_decision": False,
            }
        )
    return pd.DataFrame(rows)


def _attach_rankings(panel: pd.DataFrame) -> pd.DataFrame:
    result = panel.copy()
    for date, group in result.groupby("date", sort=True):
        date_mask = result["date"].eq(date)
        candidate_count = int(group["is_candidate_available"].astype(bool).sum())
        result.loc[date_mask, "candidate_count"] = candidate_count
        for variant in VARIANTS:
            score_column = f"rank_score_{variant}"
            rank_column = f"rank_{variant}"
            top1_column = f"top1_{variant}"
            top3_column = f"top3_{variant}"
            margin_column = f"score_margin_top1_top2_{variant}"
            rankable = group[group[score_column].astype(str).ne("")].copy()
            if rankable.empty:
                continue
            rankable["_score"] = rankable[score_column].astype(float)
            rankable = rankable.sort_values(["_score", "candidate_ticker"], ascending=[False, True]).reset_index()
            top1 = str(rankable.iloc[0]["candidate_ticker"])
            top3 = "|".join(rankable.head(3)["candidate_ticker"].astype(str).tolist())
            margin = ""
            if len(rankable) >= 2:
                margin = str(round(float(rankable.iloc[0]["_score"]) - float(rankable.iloc[1]["_score"]), 8))
            result.loc[date_mask, top1_column] = top1
            result.loc[date_mask, top3_column] = top3
            result.loc[date_mask, margin_column] = margin
            for rank, row in enumerate(rankable.to_dict(orient="records"), start=1):
                result.loc[int(row["index"]), rank_column] = str(rank)
    return result


def _data_readiness_by_date(panel: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for date, group in panel.groupby("date", sort=True):
        row: dict[str, Any] = {
            "date": date,
            "candidate_count": int(group["is_candidate_available"].astype(bool).sum()),
            "price_available_count": int(group["price_available"].astype(bool).sum()),
        }
        blockers: list[str] = []
        for variant in VARIANTS:
            rankable = int(group[f"rank_score_{variant}"].astype(str).ne("").sum())
            row[f"rankable_{variant}_count"] = rankable
            row[f"ready_{variant}"] = rankable >= 2
            if rankable < 2:
                blockers.append(f"{variant}_rankable_count_lt_2")
        row["data_ready_all_variants"] = all(bool(row[f"ready_{variant}"]) for variant in VARIANTS)
        row["blocked_reason"] = ";".join(blockers)
        rows.append(row)
    return pd.DataFrame(rows)


def _candidate_ranking_summary(panel: pd.DataFrame, readiness: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = [
        {"metric": "task_id", "value": TASK_ID},
        {"metric": "status", "value": "completed_diagnostic_only"},
        {"metric": "date_start", "value": str(panel["date"].min())},
        {"metric": "date_end", "value": str(panel["date"].max())},
        {"metric": "panel_rows", "value": int(len(panel))},
        {"metric": "unique_candidates", "value": int(panel["candidate_ticker"].nunique())},
        {"metric": "candidate_available_rows", "value": int(panel["is_candidate_available"].astype(bool).sum())},
        {"metric": "data_ready_rows_all_variants", "value": int(panel["data_ready"].astype(bool).sum())},
        {"metric": "data_ready_dates_all_variants", "value": int(readiness["data_ready_all_variants"].astype(bool).sum())},
        {"metric": "diagnostic_only", "value": True},
        {"metric": "formal_model_changed", "value": False},
        {"metric": "trade_decision_changed", "value": False},
        {"metric": "active_in_trade_decision", "value": False},
        {"metric": "uses_forward_return", "value": False},
    ]
    for variant in VARIANTS:
        rows.append(
            {
                "metric": f"ready_dates_{variant}",
                "value": int(readiness[f"ready_{variant}"].astype(bool).sum()),
            }
        )
    return pd.DataFrame(rows)


def _top_name_turnover_summary(panel: pd.DataFrame, readiness: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    names = panel.drop_duplicates("candidate_ticker").set_index("candidate_ticker")["candidate_name"].to_dict()
    for variant in VARIANTS:
        ready_dates = readiness.loc[readiness[f"ready_{variant}"].astype(bool), "date"].astype(str).tolist()
        tops = (
            panel[panel["date"].isin(ready_dates)]
            .drop_duplicates("date")
            .sort_values("date")[["date", f"top1_{variant}", f"top3_{variant}"]]
        )
        if tops.empty:
            rows.append(_empty_turnover_row(variant))
            continue
        top1_values = tops[f"top1_{variant}"].astype(str).tolist()
        top3_values = tops[f"top3_{variant}"].astype(str).tolist()
        top1_changes = sum(1 for previous, current in zip(top1_values, top1_values[1:]) if previous != current)
        top3_changes = sum(1 for previous, current in zip(top3_values, top3_values[1:]) if previous != current)
        denominator = max(len(tops) - 1, 1)
        rows.append(
            {
                "variant": variant,
                "ready_dates": int(len(tops)),
                "first_ready_date": str(tops.iloc[0]["date"]),
                "last_ready_date": str(tops.iloc[-1]["date"]),
                "top1_change_count": int(top1_changes),
                "top1_change_rate": round(top1_changes / denominator, 8),
                "top3_change_count": int(top3_changes),
                "top3_change_rate": round(top3_changes / denominator, 8),
                "most_recent_top1": top1_values[-1],
                "most_recent_top1_name": names.get(top1_values[-1], top1_values[-1]),
            }
        )
    return pd.DataFrame(rows)


def _score_margin_distribution(panel: pd.DataFrame, readiness: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for variant in VARIANTS:
        dates = readiness.loc[readiness[f"ready_{variant}"].astype(bool), "date"].astype(str).tolist()
        margins = (
            panel[panel["date"].isin(dates)]
            .drop_duplicates("date")[f"score_margin_top1_top2_{variant}"]
            .replace("", pd.NA)
            .dropna()
            .astype(float)
        )
        if margins.empty:
            rows.append({"variant": variant, "count": 0})
            continue
        rows.append(
            {
                "variant": variant,
                "count": int(len(margins)),
                "mean": round(float(margins.mean()), 8),
                "median": round(float(margins.median()), 8),
                "min": round(float(margins.min()), 8),
                "p10": round(float(margins.quantile(0.10)), 8),
                "p25": round(float(margins.quantile(0.25)), 8),
                "p75": round(float(margins.quantile(0.75)), 8),
                "p90": round(float(margins.quantile(0.90)), 8),
                "max": round(float(margins.max()), 8),
            }
        )
    return pd.DataFrame(rows)


def _window_returns(frame: pd.DataFrame | None, signal_date: pd.Timestamp) -> dict[int, float | None]:
    if frame is None or frame.empty or "adj_close" not in frame.columns:
        return {window: None for window in WINDOWS}
    history = frame.loc[frame.index <= signal_date, "adj_close"].dropna()
    returns: dict[int, float | None] = {}
    for window in WINDOWS:
        if len(history) <= window:
            returns[window] = None
        else:
            returns[window] = float(history.iloc[-1] / history.iloc[-window] - 1)
    return returns


def _variant_score(short_return: float | None, long_return: float | None) -> float | None:
    if short_return is None or long_return is None:
        return None
    return (0.4 * float(short_return)) + (0.6 * float(long_return))


def _price_available(frame: pd.DataFrame | None, signal_date: pd.Timestamp) -> bool:
    if frame is None or frame.empty or "adj_close" not in frame.columns:
        return False
    rows = frame.loc[frame.index <= signal_date, "adj_close"].dropna()
    return not rows.empty and rows.index.max().normalize() == signal_date


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _round_or_blank(value: float | None) -> str | float:
    if value is None:
        return ""
    return round(float(value), 8)


def _empty_turnover_row(variant: str) -> dict[str, Any]:
    return {
        "variant": variant,
        "ready_dates": 0,
        "first_ready_date": "",
        "last_ready_date": "",
        "top1_change_count": 0,
        "top1_change_rate": "",
        "top3_change_count": 0,
        "top3_change_rate": "",
        "most_recent_top1": "",
        "most_recent_top1_name": "",
    }


def _manifest(
    *,
    output: Path,
    panel: pd.DataFrame,
    readiness: pd.DataFrame,
    blocked: pd.DataFrame,
    start_date: str,
    end_date: str,
    lifecycle_dir: str | Path,
    price_cache_dir: str | Path,
    price_source_registry: str | Path,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "task_id": TASK_ID,
        "status": "completed_diagnostic_only",
        "panel_boundary": "diagnostic_only",
        "candidate_pool_formal_ready": False,
        "date_start": start_date,
        "date_end": end_date,
        "panel_rows": int(len(panel)),
        "blocked_rows": int(len(blocked)),
        "unique_candidates": int(panel["candidate_ticker"].nunique()),
        "data_ready_dates_all_variants": int(readiness["data_ready_all_variants"].astype(bool).sum()),
        "rank_variants": list(VARIANTS),
        "score_formula": {
            "rank_score_10_30": "0.4 * rs_10d + 0.6 * rs_30d",
            "rank_score_10_40": "0.4 * rs_10d + 0.6 * rs_40d",
            "rank_score_20_60": "0.4 * rs_20d + 0.6 * rs_60d",
        },
        "uses_forward_return": False,
        "uses_sector_static_map": False,
        "proxy_used_as_formal": False,
        "formal_model_changed": False,
        "trade_decision_changed": False,
        "active_in_trade_decision": False,
        "lifecycle_input_dir": str(lifecycle_dir),
        "price_cache_dir": str(price_cache_dir),
        "price_source_registry": str(price_source_registry),
        "output_dir": str(output.resolve()),
        "next_experiments_task": "TASK-BACKTEST-EXPERIMENTS-SHORT-WINDOW-REGIME-CHALLENGER-REPLAY-001",
        "outputs": {
            "candidate_ranking_panel": "candidate_ranking_panel_2015_2021.csv",
            "candidate_ranking_summary": "candidate_ranking_summary.csv",
            "data_readiness_by_date": "data_readiness_by_date.csv",
            "blocked_rows": "blocked_rows.csv",
            "top_name_turnover_summary": "top_name_turnover_summary.csv",
            "score_margin_distribution": "score_margin_distribution.csv",
            "final_summary": "final_summary_zh.md",
        },
    }


def _next_experiments_task(output: Path) -> str:
    return "\n".join(
        [
            "# Experiments handoff",
            "",
            "Task: TASK-BACKTEST-EXPERIMENTS-SHORT-WINDOW-REGIME-CHALLENGER-REPLAY-001",
            "",
            f"Core input: `{output.resolve()}`",
            "",
            "Use `candidate_ranking_panel_2015_2021.csv` to test 10/30, 10/40, and 20/60 top1/top3 turnover.",
            "The panel is diagnostic-only and must not be treated as a formal selector or trade action.",
        ]
    )


def _final_summary(summary: pd.DataFrame, readiness: pd.DataFrame, blocked: pd.DataFrame) -> str:
    metrics = dict(zip(summary["metric"], summary["value"]))
    ready_dates = metrics.get("data_ready_dates_all_variants", 0)
    total_dates = len(readiness)
    return "\n".join(
        [
            "# 2015-2021 candidate ranking / score contract",
            "",
            f"- 狀態：completed_diagnostic_only。",
            f"- 期間：{metrics.get('date_start')}～{metrics.get('date_end')}。",
            f"- panel rows：{metrics.get('panel_rows')}；unique candidates：{metrics.get('unique_candidates')}。",
            f"- 三組分數都可計算的日期：{ready_dates}/{total_dates}。",
            f"- blocked rows：{len(blocked)}，皆保留 fail-closed reason，沒有補假分數。",
            "- 邊界：formal_model_changed=false、trade_decision_changed=false、active_in_trade_decision=false。",
            "- 下一棒：交 Experiments 重算 10/30、10/40、20/60 selector 與 top1/top3 turnover。",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build 2015-2021 diagnostic candidate ranking score contract.")
    parser.add_argument("--lifecycle-dir", default=DEFAULT_LIFECYCLE_DIR)
    parser.add_argument("--price-cache-dir", default=DEFAULT_PRICE_CACHE_DIR)
    parser.add_argument("--price-source-registry", default=DEFAULT_PRICE_SOURCE_REGISTRY)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", default=DEFAULT_END_DATE)
    args = parser.parse_args()
    output = run_candidate_ranking_score_contract_2015_2021(
        lifecycle_dir=args.lifecycle_dir,
        price_cache_dir=args.price_cache_dir,
        price_source_registry=args.price_source_registry,
        output_dir=args.output_dir,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    print(output.resolve())


if __name__ == "__main__":
    main()
