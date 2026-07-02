from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from backtest_lab.config import load_config
from backtest_lab.data import load_price_csv, normalize_price_frame
from backtest_lab.pcf_pit_candidate_adapter import (
    DEFAULT_MONTHLY_ANCHOR_PATH,
    load_0050_pcf_monthly_anchor,
)
from backtest_lab.stock_pool_store import KNOWN_SYMBOLS
from backtest_lab.strategies import relative_strength_scores
from backtest_lab.universal_pool_strategy import (
    UniversalCandidateScore,
    universal_stock_score,
    window_return,
)


TASK_ID = "TASK-BACKTEST-CORE-CURRENT-FORMAL-POOL1-POOL2-SIGNAL-PANELS-201411-202112-001"
DEFAULT_OUTPUT_DIR = "outputs/current_formal_pool1_pool2_signal_panels_201411_202112_20260630"
DEFAULT_PRICE_CACHE_DIR = "backtest_cache/stock_pool_observations"
DEFAULT_PRICE_SOURCE_REGISTRY = "data/price_source_registry.csv"
DEFAULT_CONFIG_PATH = "configs/ep05_universe.json"
DEFAULT_START_DATE = "2014-11-03"
DEFAULT_END_DATE = "2021-12-31"
POOL1_TICKERS = (
    "00631L.TW",
    "2330.TW",
    "2454.TW",
    "2308.TW",
    "2317.TW",
    "2382.TW",
    "3231.TW",
    "6669.TW",
)
FORMAL_CANDIDATE_EXCLUDED_TICKERS = {"0050.TW"}
TW50_BENCHMARK = "0050.TW"
TW50_RET60_MARGIN = 0.08
TW50_RET20_MIN = 0.03
TW50_RET60_MIN = 0.12
TW50_PERSISTENCE_LOOKBACK = 10
TW50_PERSISTENCE_MIN_DAYS = 5


def run_current_formal_pool1_pool2_signal_panels(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
    price_cache_dir: str | Path = DEFAULT_PRICE_CACHE_DIR,
    monthly_anchor_path: str | Path = DEFAULT_MONTHLY_ANCHOR_PATH,
    price_source_registry: str | Path = DEFAULT_PRICE_SOURCE_REGISTRY,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
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
        start = pd.Timestamp(start_date).normalize()
        end = pd.Timestamp(end_date).normalize()
        log("load_inputs", "started", f"{start.date()}..{end.date()}")
        config = load_config(config_path)
        names = _name_map(config)
        anchor = load_0050_pcf_monthly_anchor(monthly_anchor_path)
        registry = _load_price_source_registry(price_source_registry)
        trading_dates = _trading_dates(price_cache_dir, start, end)

        all_tickers = sorted(set(POOL1_TICKERS) | _anchor_tickers(anchor) | {TW50_BENCHMARK})
        prices_by_ticker: dict[str, pd.DataFrame] = {}
        price_source_meta: dict[str, dict[str, Any]] = {}
        for ticker in all_tickers:
            frame, meta = _load_price_source(ticker, price_cache_dir=price_cache_dir, registry=registry)
            if frame is not None:
                prices_by_ticker[ticker] = frame
                price_source_meta[ticker] = meta

        log("build_pool1_panel", "started", f"dates={len(trading_dates)}")
        pool1_panel, pool1_daily = _build_pool1_panel(
            trading_dates=trading_dates,
            prices_by_ticker=prices_by_ticker,
            names=names,
            price_source_meta=price_source_meta,
        )
        log("build_pool2_panel", "started", f"dates={len(trading_dates)}")
        pool2_panel, pool2_daily = _build_pool2_panel(
            trading_dates=trading_dates,
            anchor=anchor,
            prices_by_ticker=prices_by_ticker,
            names=names,
            price_source_meta=price_source_meta,
        )
        readiness = _formal_policy_input_readiness(trading_dates, pool1_daily, pool2_daily)
        blockers = _data_blockers(pool1_panel, pool2_panel, readiness)

        log("write_outputs", "started", str(output))
        pool1_panel.to_csv(output / "pool1_daily_candidate_ranking_panel.csv", index=False, encoding="utf-8-sig")
        pool2_panel.to_csv(output / "pool2_daily_confirmation_panel.csv", index=False, encoding="utf-8-sig")
        readiness.to_csv(output / "formal_policy_input_readiness.csv", index=False, encoding="utf-8-sig")
        blockers.to_csv(output / "data_blockers.csv", index=False, encoding="utf-8-sig")

        manifest = _manifest(
            pool1_panel=pool1_panel,
            pool2_panel=pool2_panel,
            readiness=readiness,
            blockers=blockers,
            start_date=start,
            end_date=end,
            trading_dates=trading_dates,
            monthly_anchor_path=monthly_anchor_path,
        )
        (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        (output / "final_summary_zh.md").write_text(_summary_zh(manifest, blockers), encoding="utf-8")
        pd.DataFrame([{"step": TASK_ID, "status": "completed_partial_signal_panels"}]).to_csv(
            output / "completed.csv", index=False, encoding="utf-8-sig"
        )
        pd.DataFrame(columns=["step", "status", "reason"]).to_csv(output / "failed.csv", index=False, encoding="utf-8-sig")
        log("completed", "completed", str(output.resolve()))
        return output
    except Exception as exc:
        pd.DataFrame([{"step": TASK_ID, "status": "failed", "reason": str(exc)}]).to_csv(
            output / "failed.csv", index=False, encoding="utf-8-sig"
        )
        log("failed", "failed", str(exc))
        raise


def _load_price_source_registry(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    if not source.exists():
        return pd.DataFrame()
    return pd.read_csv(source).fillna("")


def _trading_dates(price_cache_dir: str | Path, start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    benchmark, _ = _load_price_source(
        TW50_BENCHMARK,
        price_cache_dir=price_cache_dir,
        registry=pd.DataFrame(),
    )
    if benchmark is None:
        raise FileNotFoundError(f"Missing benchmark price cache for {TW50_BENCHMARK}")
    dates = benchmark.index[(benchmark.index >= start) & (benchmark.index <= end)]
    return [pd.Timestamp(date).normalize() for date in dates]


def _load_price_source(
    ticker: str,
    *,
    price_cache_dir: str | Path,
    registry: pd.DataFrame,
) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    frames: list[pd.DataFrame] = []
    meta = {
        "base_cache_used": False,
        "supplemental_used": False,
        "adjusted_close_available": True,
        "source_types": [],
    }
    cache_path = Path(price_cache_dir) / f"{ticker.replace('.', '_')}.csv"
    if cache_path.exists():
        frame = load_price_csv(cache_path)
        frame["price_source_type"] = "base_cache"
        frame["price_source_adjusted_close_available"] = True
        frame["_source_priority"] = 0
        frames.append(frame)
        meta["base_cache_used"] = True
        meta["source_types"].append("base_cache")
    if not registry.empty and "ticker" in registry.columns:
        for _, row in registry[registry["ticker"].astype(str).eq(ticker)].iterrows():
            path = Path(str(row.get("source_path", "")))
            if not path.exists():
                continue
            frame = pd.read_csv(path, parse_dates=["date"], index_col="date")
            normalized = frame.copy()
            normalized.columns = [str(column).lower().replace(" ", "_") for column in normalized.columns]
            if "adj_close" not in normalized.columns or normalized["adj_close"].isna().all():
                normalized["adj_close"] = normalized["close"]
                adjusted = False
            else:
                adjusted = True
            normalized = normalize_price_frame(normalized)
            normalized["price_source_type"] = str(row.get("source_type", "supplemental_price_source"))
            normalized["price_source_adjusted_close_available"] = adjusted
            normalized["_source_priority"] = 1
            frames.append(normalized)
            meta["supplemental_used"] = True
            meta["adjusted_close_available"] = bool(meta["adjusted_close_available"] and adjusted)
            meta["source_types"].append(str(row.get("source_type", "supplemental_price_source")))
    if not frames:
        return None, meta
    combined = pd.concat(frames)
    combined["_date_index"] = combined.index
    combined = combined.sort_values(["_date_index", "_source_priority"]).set_index("_date_index")
    combined.index.name = "date"
    combined = combined[~combined.index.duplicated(keep="last")]
    combined = combined.drop(columns=["_source_priority"], errors="ignore")
    return combined, meta


def _name_map(config: dict[str, Any]) -> dict[str, str]:
    names: dict[str, str] = {}
    for group in getattr(config, "groups", ()):
        for asset in getattr(group, "assets", ()):
            ticker = str(getattr(asset, "ticker", "")).strip()
            label = str(getattr(asset, "label", "")).strip()
            if ticker and label:
                names[ticker] = label
    for ticker, row in KNOWN_SYMBOLS.items():
        names.setdefault(ticker, str(row.get("name") or ticker.replace(".TW", "")))
    return names


def _display_name(ticker: str, names: dict[str, str], fallback: str = "") -> str:
    if fallback:
        return fallback
    return names.get(ticker, KNOWN_SYMBOLS.get(ticker, {}).get("name", ticker.replace(".TW", "")))


def _anchor_tickers(anchor: pd.DataFrame) -> set[str]:
    return {f"{str(ticker).strip().zfill(4)}.TW" for ticker in anchor["ticker"].astype(str)}


def _build_pool1_panel(
    *,
    trading_dates: list[pd.Timestamp],
    prices_by_ticker: dict[str, pd.DataFrame],
    names: dict[str, str],
    price_source_meta: dict[str, dict[str, Any]],
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    daily: dict[str, dict[str, Any]] = {}
    price_subset = {ticker: prices_by_ticker[ticker] for ticker in POOL1_TICKERS if ticker in prices_by_ticker}
    for signal_date in trading_dates:
        available = {
            ticker: frame
            for ticker, frame in price_subset.items()
            if frame.index.min() <= signal_date and frame.loc[frame.index <= signal_date, "adj_close"].dropna().shape[0] > 60
        }
        scores = relative_strength_scores(available, signal_date, windows=(20, 60))
        ranked = sorted(scores.items(), key=lambda item: (item[1], item[0]), reverse=True)
        eligible_rank = 0
        top_candidate = ""
        for raw_rank, (ticker, score) in enumerate(ranked, start=1):
            excluded = ticker in FORMAL_CANDIDATE_EXCLUDED_TICKERS
            if not excluded:
                eligible_rank += 1
                if not top_candidate:
                    top_candidate = ticker
            meta = price_source_meta.get(ticker, {})
            rows.append(
                {
                    "date": signal_date.strftime("%Y-%m-%d"),
                    "pool_id": "ai_theme_large_cap_v20260613",
                    "pool_name": "AI主線池",
                    "candidate_ticker": ticker,
                    "candidate_name": _display_name(ticker, names),
                    "score": round(float(score), 8),
                    "raw_rank": raw_rank,
                    "rank": eligible_rank if not excluded else "",
                    "passed": str(not excluded).lower(),
                    "attack_gate_status": "ranking_reconstructed_attack_gate_not_reconstructed",
                    "reason": "Pool1 2014-2021 可重建相對強度排名；正式攻擊 gate / target stream 還缺 date-aware 合約。",
                    "formal_vote_ready": "false",
                    "price_only_used": str(not bool(meta.get("adjusted_close_available", True))).lower(),
                    "adjusted_close_available": str(bool(meta.get("adjusted_close_available", True))).lower(),
                }
            )
        daily[signal_date.strftime("%Y-%m-%d")] = {
            "pool1_top_candidate": top_candidate,
            "pool1_candidate_count": len(ranked),
            "pool1_formal_vote_ready": False,
            "pool1_blocker": "missing_date_aware_pool1_attack_gate_and_formal_target_contract",
        }
    return pd.DataFrame(rows), daily


def _build_pool2_panel(
    *,
    trading_dates: list[pd.Timestamp],
    anchor: pd.DataFrame,
    prices_by_ticker: dict[str, pd.DataFrame],
    names: dict[str, str],
    price_source_meta: dict[str, dict[str, Any]],
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    daily: dict[str, dict[str, Any]] = {}
    ret60_by_ticker = _ret60_series_by_ticker(prices_by_ticker)
    features_by_ticker = _pool2_feature_frames(prices_by_ticker)
    for signal_date in trading_dates:
        constituents, anchor_meta = _resolve_anchor_from_frame(anchor, signal_date)
        anchor_names = {
            f"{str(row.ticker).zfill(4)}.TW": str(row.name).strip()
            for row in constituents[["ticker", "name"]].itertuples(index=False)
        }
        candidate_tickers = list(anchor_names)
        prices = {ticker: prices_by_ticker[ticker] for ticker in candidate_tickers if ticker in prices_by_ticker}
        if TW50_BENCHMARK in prices_by_ticker:
            prices[TW50_BENCHMARK] = prices_by_ticker[TW50_BENCHMARK]
        if not prices:
            daily[signal_date.strftime("%Y-%m-%d")] = {
                "pool2_vote": "",
                "pool2_vote_ready": False,
                "pool2_blocker": "no_price_frames_for_pit_constituents",
            }
            continue
        candidates = _score_pool2_candidates_direct(prices.keys(), signal_date, features_by_ticker)
        gate_by_ticker = _tw50_gate_details_by_ticker(candidates, prices, signal_date, ret60_by_ticker)
        ranked = sorted(
            candidates,
            key=lambda item: (
                bool(gate_by_ticker.get(item.ticker, {}).get("candidate_support_without_persistence", False)),
                item.score,
                item.ticker,
            ),
            reverse=True,
        )
        eligible_rank = 0
        vote = ""
        for raw_rank, candidate in enumerate(ranked, start=1):
            excluded = candidate.ticker in FORMAL_CANDIDATE_EXCLUDED_TICKERS
            gate = gate_by_ticker.get(candidate.ticker, _empty_tw50_gate(candidate))
            support_candidate = bool(gate.get("candidate_support_without_persistence", False)) and not excluded
            eligible = bool(gate.get("eligible_for_pool_selection", False)) and not excluded
            if support_candidate:
                eligible_rank += 1
                if not vote:
                    vote = candidate.ticker
            meta = price_source_meta.get(candidate.ticker, {})
            rows.append(
                {
                    "date": signal_date.strftime("%Y-%m-%d"),
                    "pool_id": "tw50_dynamic_constituents_v0_pcf_monthly_anchor_candidate",
                    "pool_name": "大型廣度池",
                    "candidate_ticker": candidate.ticker,
                    "candidate_name": _display_name(
                        candidate.ticker,
                        names,
                        fallback=anchor_names.get(candidate.ticker, ""),
                    ),
                    "score": round(float(candidate.score), 8),
                    "raw_rank": raw_rank,
                    "rank": eligible_rank if support_candidate else "",
                    "passed": str(bool(candidate.passed)).lower(),
                    "confirmation_state": _pool2_confirmation_state(vote, support_candidate),
                    "market_exposure_support": "supports_market_exposure" if vote else "not_sufficiently_confirmed",
                    "eligible_for_pool_selection": str(eligible).lower(),
                    "candidate_support_without_persistence": str(support_candidate).lower(),
                    "attack_gate_open": str(bool(gate.get("attack_gate_open", False))).lower(),
                    "reason": gate.get("gate_reason", candidate.reason),
                    "base_pool_passed": str(bool(gate.get("base_pool_passed", candidate.passed))).lower(),
                    "benchmark_margin_passed": str(bool(gate.get("benchmark_margin_passed", False))).lower(),
                    "momentum_quality_passed": str(bool(gate.get("momentum_quality_passed", False))).lower(),
                    "persistence_passed": str(bool(gate.get("persistence_passed", False))).lower(),
                    "anchor_effective_month": anchor_meta["effective_month"],
                    "anchor_effective_date": anchor_meta["anchor_effective_date"],
                    "anchor_after_query_date": str(anchor_meta["anchor_after_query_date"]).lower(),
                    "pit_safe_for_query_date": str(anchor_meta["pit_safe_for_query_date"]).lower(),
                    "formal_exact": "false",
                    "source_type": "source_backed_manual_candidate",
                    "price_only_used": str(not bool(meta.get("adjusted_close_available", True))).lower(),
                    "adjusted_close_available": str(bool(meta.get("adjusted_close_available", True))).lower(),
                }
            )
        daily[signal_date.strftime("%Y-%m-%d")] = {
            "pool2_vote": vote,
            "pool2_vote_ready": False,
            "pool2_blocker": "missing_pool2_persistence_full_reconstruction",
            "anchor_after_query_date": anchor_meta["anchor_after_query_date"],
            "pit_safe_for_query_date": anchor_meta["pit_safe_for_query_date"],
        }
    return pd.DataFrame(rows), daily


def _score_pool2_candidates_direct(
    tickers: Any,
    signal_date: pd.Timestamp,
    features_by_ticker: dict[str, pd.DataFrame],
) -> list[UniversalCandidateScore]:
    candidates: list[UniversalCandidateScore] = []
    for ticker in tickers:
        if ticker == TW50_BENCHMARK:
            continue
        features = features_by_ticker.get(ticker)
        row = _last_feature_row_on_or_before(features, signal_date)
        if row is None or int(row.get("history_len", 0)) < 126:
            candidates.append(_pool2_reject(ticker, "warmup不足"))
            continue
        close = float(row["close"])
        ma20 = float(row["ma20"])
        ma60 = float(row["ma60"])
        avg_turnover = float(row["avg_turnover20"])
        ret20 = float(row["ret20"])
        ret60 = float(row["ret60"])
        ret120 = float(row["ret120"])
        drawdown20 = float(row["drawdown20"])
        vol20 = float(row["vol20"])
        if pd.isna([close, ma20, ma60, avg_turnover, ret20, ret60, ret120, drawdown20, vol20]).any():
            candidates.append(_pool2_reject(ticker, "特徵資料不足"))
            continue
        score = universal_stock_score(
            ret20=ret20,
            ret60=ret60,
            ret120=ret120,
            vol20=vol20,
            mode="risk_adjusted",
        )
        reason = ""
        passed = True
        if close < ma20:
            passed = False
            reason = "跌破20日均線"
        elif close < ma60:
            passed = False
            reason = "跌破60日均線"
        elif avg_turnover < 60_000_000:
            passed = False
            reason = "流動性不足"
        elif ret20 > 0.62:
            passed = False
            reason = "20日漲幅過熱"
        elif drawdown20 < -0.25:
            passed = False
            reason = "20日回撤過深"
        elif score < 0.0:
            passed = False
            reason = "分數未達門檻"
        candidates.append(
            UniversalCandidateScore(
                ticker=ticker,
                score=score,
                ret20=ret20,
                ret60=ret60,
                ret120=ret120,
                vol20=vol20,
                avg_turnover_twd=avg_turnover,
                drawdown20=drawdown20,
                passed=passed,
                reason=reason,
                applied_score_mode="risk_adjusted",
            )
        )
    return candidates


def _pool2_feature_frames(prices_by_ticker: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    features: dict[str, pd.DataFrame] = {}
    for ticker, prices in prices_by_ticker.items():
        if prices.empty or "adj_close" not in prices.columns:
            continue
        frame = prices.sort_index().copy()
        adj = frame["adj_close"].astype(float)
        close = frame["close"].astype(float) if "close" in frame.columns else adj
        volume = frame["volume"].fillna(0).astype(float) if "volume" in frame.columns else pd.Series(0.0, index=frame.index)
        result = pd.DataFrame(index=frame.index)
        result["history_len"] = adj.notna().cumsum()
        result["close"] = adj
        result["ma20"] = adj.rolling(20).mean()
        result["ma60"] = adj.rolling(60).mean()
        result["avg_turnover20"] = (close * volume).rolling(20).mean()
        result["ret20"] = adj / adj.shift(20) - 1
        result["ret60"] = adj / adj.shift(60) - 1
        result["ret120"] = adj / adj.shift(120) - 1
        result["drawdown20"] = adj / adj.rolling(20).max() - 1
        result["vol20"] = adj.pct_change().rolling(20).std() * (252**0.5)
        features[ticker] = result
    return features


def _last_feature_row_on_or_before(frame: pd.DataFrame | None, signal_date: pd.Timestamp) -> pd.Series | None:
    if frame is None or frame.empty:
        return None
    position = frame.index.searchsorted(signal_date, side="right") - 1
    if position < 0:
        return None
    return frame.iloc[int(position)]


def _pool2_reject(ticker: str, reason: str) -> UniversalCandidateScore:
    return UniversalCandidateScore(
        ticker=ticker,
        score=0.0,
        ret20=0.0,
        ret60=0.0,
        ret120=0.0,
        vol20=0.0,
        avg_turnover_twd=0.0,
        drawdown20=0.0,
        passed=False,
        reason=reason,
        applied_score_mode="risk_adjusted",
    )


def _resolve_anchor_from_frame(anchor: pd.DataFrame, signal_date: pd.Timestamp) -> tuple[pd.DataFrame, dict[str, Any]]:
    target = signal_date.normalize()
    eligible = anchor[anchor["effective_date"] <= target]
    if eligible.empty:
        month = target.strftime("%Y-%m")
        selected = anchor[anchor["effective_month"].astype(str).eq(month)].copy()
    else:
        month = str(eligible.sort_values("effective_date").iloc[-1]["effective_month"])
        selected = anchor[anchor["effective_month"].astype(str).eq(month)].copy()
    if selected.empty:
        raise ValueError(f"No monthly anchor for {target.date()}")
    anchor_date = pd.Timestamp(selected["effective_date"].iloc[0]).normalize()
    return selected, {
        "effective_month": str(selected["effective_month"].iloc[0]),
        "anchor_effective_date": anchor_date.strftime("%Y-%m-%d"),
        "anchor_after_query_date": bool(anchor_date > target),
        "pit_safe_for_query_date": bool(anchor_date <= target),
    }


def _tw50_gate_details_by_ticker(
    candidates: list[UniversalCandidateScore],
    prices_by_ticker: dict[str, pd.DataFrame],
    signal_date: pd.Timestamp,
    ret60_by_ticker: dict[str, pd.Series],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    benchmark_ret60 = _window_return_on_or_before(prices_by_ticker.get(TW50_BENCHMARK), signal_date, 60)
    for candidate in candidates:
        if benchmark_ret60 is None:
            result[candidate.ticker] = _tw50_gate_result(
                candidate,
                eligible=False,
                benchmark_margin_passed=False,
                momentum_quality_passed=False,
                persistence_passed=False,
                ret60_margin=None,
                persistence_days=0,
                persistence_total=0,
                reason_prefix="缺少 0050 benchmark 價格，不能確認相對超額",
            )
            continue
        ret60_margin = candidate.ret60 - benchmark_ret60
        benchmark_margin_passed = ret60_margin >= TW50_RET60_MARGIN
        momentum_quality_passed = candidate.ret20 >= TW50_RET20_MIN and candidate.ret60 >= TW50_RET60_MIN and candidate.ret60 > 0
        persistence_days, persistence_total = _tw50_persistence_days(candidate.ticker, ret60_by_ticker, signal_date)
        persistence_passed = persistence_total >= TW50_PERSISTENCE_LOOKBACK and persistence_days >= TW50_PERSISTENCE_MIN_DAYS
        support_without_persistence = bool(candidate.passed and benchmark_margin_passed and momentum_quality_passed)
        eligible = bool(support_without_persistence and persistence_passed)
        result[candidate.ticker] = _tw50_gate_result(
            candidate,
            eligible=eligible,
            support_without_persistence=support_without_persistence,
            benchmark_margin_passed=benchmark_margin_passed,
            momentum_quality_passed=momentum_quality_passed,
            persistence_passed=persistence_passed,
            ret60_margin=ret60_margin,
            persistence_days=persistence_days,
            persistence_total=persistence_total,
            reason_prefix="",
        )
    return result


def _tw50_gate_result(
    candidate: UniversalCandidateScore,
    *,
    eligible: bool,
    support_without_persistence: bool = False,
    benchmark_margin_passed: bool,
    momentum_quality_passed: bool,
    persistence_passed: bool,
    ret60_margin: float | None,
    persistence_days: int,
    persistence_total: int,
    reason_prefix: str,
) -> dict[str, Any]:
    if reason_prefix:
        reason = f"大型廣度池 v1 未通過：{reason_prefix}。"
    else:
        reason = (
            "大型廣度池 v1："
            f"base={'Y' if candidate.passed else 'N'}；"
            f"60日相對0050超額={ret60_margin:.1%}({'Y' if benchmark_margin_passed else 'N'})；"
            f"20/60動能品質={'Y' if momentum_quality_passed else 'N'}；"
            f"持續性={persistence_days}/{persistence_total}日({'Y' if persistence_passed else 'N'})"
        )
    return {
        "base_pool_passed": bool(candidate.passed),
        "benchmark_margin_passed": benchmark_margin_passed,
        "momentum_quality_passed": momentum_quality_passed,
        "persistence_passed": persistence_passed,
        "attack_gate_open": eligible,
        "eligible_for_pool_selection": eligible,
        "candidate_support_without_persistence": support_without_persistence,
        "gate_reason": reason,
    }


def _empty_tw50_gate(candidate: UniversalCandidateScore) -> dict[str, Any]:
    return {
        "base_pool_passed": bool(candidate.passed),
        "benchmark_margin_passed": False,
        "momentum_quality_passed": False,
        "persistence_passed": False,
        "attack_gate_open": False,
        "eligible_for_pool_selection": False,
        "candidate_support_without_persistence": False,
        "gate_reason": candidate.reason,
    }


def _window_return_on_or_before(frame: pd.DataFrame | None, signal_date: pd.Timestamp, window: int) -> float | None:
    if frame is None or frame.empty:
        return None
    history = frame.loc[frame.index <= signal_date].dropna(subset=["adj_close"])
    if len(history) <= window:
        return None
    return window_return(history["adj_close"], window)


def _ret60_series_by_ticker(prices_by_ticker: dict[str, pd.DataFrame]) -> dict[str, pd.Series]:
    series: dict[str, pd.Series] = {}
    for ticker, frame in prices_by_ticker.items():
        if frame.empty or "adj_close" not in frame.columns:
            continue
        values = frame["adj_close"].dropna()
        series[ticker] = values / values.shift(60) - 1
    return series


def _tw50_persistence_days(ticker: str, ret60_by_ticker: dict[str, pd.Series], signal_date: pd.Timestamp) -> tuple[int, int]:
    benchmark = ret60_by_ticker.get(TW50_BENCHMARK)
    candidate = ret60_by_ticker.get(ticker)
    if benchmark is None or candidate is None:
        return 0, 0
    dates = list(benchmark.index[benchmark.index <= signal_date])[-TW50_PERSISTENCE_LOOKBACK:]
    passed = 0
    evaluated = 0
    for current_date in dates:
        benchmark_ret60 = _series_value_on_or_before(benchmark, current_date)
        if benchmark_ret60 is None or pd.isna(benchmark_ret60):
            continue
        rows = []
        for item, series in ret60_by_ticker.items():
            if item == TW50_BENCHMARK:
                continue
            ret60 = _series_value_on_or_before(series, current_date)
            if ret60 is not None and not pd.isna(ret60):
                rows.append((item, ret60))
        if not rows:
            continue
        evaluated += 1
        rows.sort(key=lambda row: row[1], reverse=True)
        top_cutoff = max(5, int(len(rows) * 0.2 + 0.9999))
        top = {item for item, _ in rows[:top_cutoff]}
        ticker_ret60 = dict(rows).get(ticker)
        if ticker in top and ticker_ret60 is not None and ticker_ret60 - benchmark_ret60 >= TW50_RET60_MARGIN:
            passed += 1
    return passed, evaluated


def _series_value_on_or_before(series: pd.Series, signal_date: pd.Timestamp) -> float | None:
    history = series.loc[series.index <= signal_date].dropna()
    if history.empty:
        return None
    return float(history.iloc[-1])


def _pool2_confirmation_state(vote: str, eligible: bool) -> str:
    if vote:
        return "confirmation_candidate_available"
    if eligible:
        return "confirmation_candidate_pending"
    return "not_sufficiently_confirmed"


def _formal_policy_input_readiness(
    trading_dates: list[pd.Timestamp],
    pool1_daily: dict[str, dict[str, Any]],
    pool2_daily: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for signal_date in trading_dates:
        key = signal_date.strftime("%Y-%m-%d")
        p1 = pool1_daily.get(key, {})
        p2 = pool2_daily.get(key, {})
        pool1_ready = bool(p1.get("pool1_formal_vote_ready", False))
        pool2_ready = bool(p2.get("pool2_vote_ready", False))
        ready = bool(pool1_ready and pool2_ready)
        blockers = []
        if not pool1_ready:
            blockers.append(str(p1.get("pool1_blocker") or "pool1_formal_vote_missing"))
        if not pool2_ready:
            blockers.append(str(p2.get("pool2_blocker") or "pool2_confirmation_missing"))
        rows.append(
            {
                "date": key,
                "pool1_top_candidate": p1.get("pool1_top_candidate", ""),
                "pool1_formal_vote_ready": str(pool1_ready).lower(),
                "pool2_vote": p2.get("pool2_vote", ""),
                "pool2_confirmation_ready": str(pool2_ready).lower(),
                "anchor_after_query_date": str(bool(p2.get("anchor_after_query_date", False))).lower(),
                "sufficient_for_pool1_primary_pool2_confirmation": str(ready).lower(),
                "readiness_state": "ready_for_formal_target_stream" if ready else "blocked_for_formal_target_stream",
                "blocker_reason": "; ".join(blockers),
            }
        )
    return pd.DataFrame(rows)


def _data_blockers(pool1_panel: pd.DataFrame, pool2_panel: pd.DataFrame, readiness: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {
            "blocker": "pool1_date_aware_formal_attack_gate_contract",
            "status": "missing",
            "blocks_formal_target_stream": True,
            "detail": "Pool1 2014-2021 ranking can be reconstructed, but the current formal AI-theme attack gate / formal target stream contract is not implemented for date-aware historical universe handling.",
            "next_owner": "Core",
        },
        {
            "blocker": "pool1_not_yet_listed_ticker_lifecycle_contract",
            "status": "missing",
            "blocks_formal_target_stream": True,
            "detail": "Pool1 fixed universe includes tickers that were not listed for the full 2014-2021 period, especially 6669. Formal replay needs an explicit lifecycle/exclusion rule before target stream generation.",
            "next_owner": "Core/Research",
        },
        {
            "blocker": "pool2_persistence_full_reconstruction",
            "status": "missing",
            "blocks_formal_target_stream": True,
            "detail": "Pool2 daily panel includes base score, 0050 relative margin, and momentum quality, but the full 10-day persistence gate is not executed in this long-run panel yet.",
            "next_owner": "Core",
        },
        {
            "blocker": "pool2_pit_candidate_not_formal_exact",
            "status": "caveat",
            "blocks_formal_target_stream": False,
            "detail": "Pool2 uses 0050 PCF/Daily monthly anchors as source-backed manual candidate data; formal_exact=false must remain disclosed.",
            "next_owner": "Core/Research",
        },
        {
            "blocker": "unadjusted_only_price_sources",
            "status": "caveat",
            "blocks_formal_target_stream": False,
            "detail": "Four PIT universe tickers have price-only unadjusted TWSE STOCK_DAY supplemental sources. Candidate panels label price_only_used when applicable; total-return replay remains blocked until adjusted series policy is accepted.",
            "next_owner": "Core/Data",
        },
        {
            "blocker": "formal_policy_input_readiness",
            "status": "blocked" if (readiness["sufficient_for_pool1_primary_pool2_confirmation"].astype(str) != "true").any() else "ready",
            "blocks_formal_target_stream": (readiness["sufficient_for_pool1_primary_pool2_confirmation"].astype(str) != "true").any(),
            "detail": "Formal policy stream cannot be generated until Pool1 formal vote contract is ready for every signal date.",
            "next_owner": "Core",
        },
    ]
    return pd.DataFrame(rows)


def _manifest(
    *,
    pool1_panel: pd.DataFrame,
    pool2_panel: pd.DataFrame,
    readiness: pd.DataFrame,
    blockers: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    trading_dates: list[pd.Timestamp],
    monthly_anchor_path: str | Path,
) -> dict[str, Any]:
    readiness_ready = readiness["sufficient_for_pool1_primary_pool2_confirmation"].astype(str).eq("true")
    return {
        "task_id": TASK_ID,
        "generated_at": pd.Timestamp.now(tz="Asia/Taipei").isoformat(),
        "date_start": start_date.strftime("%Y-%m-%d"),
        "date_end": end_date.strftime("%Y-%m-%d"),
        "trading_date_count": len(trading_dates),
        "pool1_panel_rows": int(len(pool1_panel)),
        "pool2_panel_rows": int(len(pool2_panel)),
        "pool1_daily_candidate_ranking_panel_generated": not pool1_panel.empty,
        "pool2_daily_confirmation_panel_generated": not pool2_panel.empty,
        "formal_policy_ready_days": int(readiness_ready.sum()),
        "formal_policy_blocked_days": int((~readiness_ready).sum()),
        "formal_target_stream_ready": bool(readiness_ready.all()) if not readiness.empty else False,
        "monthly_anchor_path": str(monthly_anchor_path),
        "pool2_pit_source_type": "source_backed_manual_candidate",
        "pool2_formal_exact": False,
        "price_only_signal_panel": True,
        "uses_forward_return_as_rule": False,
        "formal_model_changed": False,
        "trade_decision_changed": False,
        "active_in_trade_decision": False,
        "blocking_count": int(blockers["blocks_formal_target_stream"].astype(bool).sum()),
    }


def _summary_zh(manifest: dict[str, Any], blockers: pd.DataFrame) -> str:
    blocking = blockers[blockers["blocks_formal_target_stream"].astype(bool)]
    blocker_lines = "\n".join(f"- {row.blocker}: {row.status} - {row.detail}" for row in blocking.itertuples())
    return f"""# Current formal Pool1/Pool2 signal panels 2014-2021

本棒已產出 2014-11-03～2021-12-31 的 Pool1 / Pool2 signal panels，但尚不能直接產生正式 target stream。

## 已完成

- Pool1 daily candidate ranking panel rows: {manifest['pool1_panel_rows']}
- Pool2 daily confirmation panel rows: {manifest['pool2_panel_rows']}
- trading dates: {manifest['trading_date_count']}
- formal_model_changed=false
- trade_decision_changed=false

## 可用性判定

- Pool1：可重建日排名，但正式 AI 主線池 attack gate / formal target vote 的 date-aware 合約仍缺。
- Pool2：已用 0050 PCF/Daily monthly anchor candidate 產生每日確認池候選與確認 gate；formal_exact=false caveat 保留。
- Formal policy input：目前 blocked，不能把這份 panel 直接包成 `pool1_primary_pool2_confirmation` target stream。

## 剩餘 formal target stream blockers

{blocker_lines}

## 下一步

Core 下一棒應補 `Pool1 date-aware formal attack gate / target vote contract`，特別是 2014-2021 固定 Pool1 標的上市前處理與正式 gate 對齊。補完後才能用本次 Pool2 panel 產 formal target stream，再進 next-day replay。
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Build current formal Pool1/Pool2 2014-2021 daily signal panels.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", default=DEFAULT_END_DATE)
    parser.add_argument("--price-cache-dir", default=DEFAULT_PRICE_CACHE_DIR)
    parser.add_argument("--monthly-anchor-path", default=DEFAULT_MONTHLY_ANCHOR_PATH)
    parser.add_argument("--price-source-registry", default=DEFAULT_PRICE_SOURCE_REGISTRY)
    parser.add_argument("--config-path", default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args()
    output = run_current_formal_pool1_pool2_signal_panels(
        output_dir=args.output_dir,
        start_date=args.start_date,
        end_date=args.end_date,
        price_cache_dir=args.price_cache_dir,
        monthly_anchor_path=args.monthly_anchor_path,
        price_source_registry=args.price_source_registry,
        config_path=args.config_path,
    )
    print(f"CURRENT_FORMAL_POOL1_POOL2_SIGNAL_PANELS_OUTPUT={output.resolve()}")


if __name__ == "__main__":
    main()
