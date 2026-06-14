from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from backtest_lab.stock_pool_store import normalize_ticker


def build_candidate_review(
    pool: dict[str, Any],
    *,
    signal_date: str,
    resolved_symbols: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    frequency = str(pool.get("candidate_review_frequency") or "").strip() or "unspecified"
    config = pool.get("candidate_review_config") or {}
    source_mode = str(config.get("source_mode") or _default_source_mode(pool)).strip()
    symbols = resolved_symbols if resolved_symbols is not None else pool.get("resolved_symbols") or pool.get("symbols") or []
    source_payload = _source_payload(pool, source_mode=source_mode, signal_date=signal_date)
    source_status = source_payload["source_status"]
    decision = _review_decision(frequency=frequency, source_status=source_status)
    review = {
        "pool_id": pool.get("pool_id", ""),
        "pool_name": pool.get("name", ""),
        "review_date": signal_date,
        "frequency": frequency,
        "source_mode": source_mode,
        "source_status": source_status,
        "decision": decision,
        "candidate_count": len(symbols),
        "source_candidate_count": source_payload.get("source_candidate_count", 0),
        "source_active_count": source_payload.get("source_active_count", 0),
        "source_watch_count": source_payload.get("source_watch_count", 0),
        "source_path": source_payload.get("source_path", ""),
        "required_evidence": list(config.get("required_evidence") or _default_required_evidence(pool)),
        "current_candidates": [
            {
                "ticker": symbol.get("ticker", ""),
                "display": symbol.get("display") or symbol.get("ticker") or "",
                "source": symbol.get("source", ""),
            }
            for symbol in symbols
        ],
        "policy": pool.get("candidate_update_policy", ""),
    }
    if source_payload.get("source_candidates"):
        review["source_candidates"] = source_payload["source_candidates"]
    return review


def write_candidate_reviews(root: Path, manifest: dict[str, Any]) -> None:
    reviews = [
        item["candidate_review"]
        for item in [*manifest.get("generated", []), *manifest.get("skipped", [])]
        if item.get("candidate_review")
    ]
    if not reviews:
        return
    (root / "stock_pool_candidate_reviews.json").write_text(
        json.dumps(reviews, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    rows = []
    for review in reviews:
        rows.append(
            {
                "pool_id": review.get("pool_id", ""),
                "pool_name": review.get("pool_name", ""),
                "review_date": review.get("review_date", ""),
                "frequency": review.get("frequency", ""),
                "source_mode": review.get("source_mode", ""),
                "source_status": review.get("source_status", ""),
                "decision": review.get("decision", ""),
                "candidate_count": review.get("candidate_count", 0),
                "source_candidate_count": review.get("source_candidate_count", 0),
                "source_active_count": review.get("source_active_count", 0),
                "source_watch_count": review.get("source_watch_count", 0),
                "source_path": review.get("source_path", ""),
                "required_evidence": "；".join(review.get("required_evidence") or []),
                "policy": review.get("policy", ""),
            }
        )
    pd.DataFrame(rows).to_csv(root / "stock_pool_candidate_reviews.csv", index=False, encoding="utf-8-sig")


def _default_source_mode(pool: dict[str, Any]) -> str:
    if pool.get("dynamic_constituents", {}).get("source") == "tw50_history_csv":
        return "point_in_time_constituents"
    if pool.get("strategy_preset") == "radar_core_mid_small_calibrated_v1":
        return "formal_radar_candidates"
    return "manual_evidence_gate"


def _source_status(pool: dict[str, Any], *, source_mode: str) -> str:
    if source_mode == "ai_theme_candidate_csv":
        config = pool.get("candidate_review_config") or {}
        source_path = str(config.get("path") or "").strip()
        return "source_ready" if source_path and Path(source_path).exists() else "source_missing"
    if source_mode == "point_in_time_constituents":
        return "source_ready" if pool.get("dynamic_constituents", {}).get("path") else "source_missing"
    if source_mode == "formal_radar_candidates":
        return "source_ready"
    if source_mode == "manual_evidence_gate":
        return "manual_review_required"
    return "source_unspecified"


def _source_payload(pool: dict[str, Any], *, source_mode: str, signal_date: str) -> dict[str, Any]:
    config = pool.get("candidate_review_config") or {}
    if source_mode != "ai_theme_candidate_csv":
        return {"source_status": _source_status(pool, source_mode=source_mode)}
    path = Path(str(config.get("path") or ""))
    if not path.exists():
        return {"source_status": "source_missing", "source_path": str(path)}
    candidates = load_ai_theme_candidate_source(path, signal_date=signal_date)
    return {
        "source_status": "source_ready",
        "source_path": str(path),
        "source_candidate_count": len(candidates),
        "source_active_count": sum(1 for item in candidates if item.get("review_status") == "active"),
        "source_watch_count": sum(1 for item in candidates if item.get("review_status") == "watch"),
        "source_candidates": candidates,
    }


def load_ai_theme_candidate_source(path: str | Path, *, signal_date: str) -> list[dict[str, Any]]:
    frame = pd.read_csv(path, dtype=str).fillna("")
    if frame.empty:
        return []
    if "effective_date" in frame.columns:
        signal_ts = pd.Timestamp(signal_date)
        frame["_effective_ts"] = pd.to_datetime(frame["effective_date"], errors="coerce")
        frame = frame[(frame["_effective_ts"].isna()) | (frame["_effective_ts"] <= signal_ts)]
    rows = []
    for _, row in frame.iterrows():
        ticker = normalize_ticker(str(row.get("ticker") or row.get("symbol") or ""))
        rows.append(
            {
                "ticker": ticker,
                "display": _display_from_row(row, ticker),
                "theme_role": str(row.get("theme_role") or "").strip(),
                "review_status": str(row.get("review_status") or "").strip().lower() or "watch",
                "is_current_member": _truthy(row.get("is_current_member")),
                "ai_exposure_score": _number(row.get("ai_exposure_score")),
                "liquidity_score": _number(row.get("liquidity_score")),
                "fundamental_score": _number(row.get("fundamental_score")),
                "theme_strength_score": _number(row.get("theme_strength_score")),
                "review_reason": str(row.get("review_reason") or "").strip(),
            }
        )
    rows.sort(
        key=lambda item: (
            item["review_status"] == "active",
            item["ai_exposure_score"],
            item["theme_strength_score"],
            item["liquidity_score"],
            item["ticker"],
        ),
        reverse=True,
    )
    return rows


def _review_decision(*, frequency: str, source_status: str) -> str:
    if frequency.lower() != "monthly":
        return "document_policy_only"
    if source_status == "source_ready":
        return "monthly_auto_review_available"
    if source_status == "manual_review_required":
        return "keep_current_until_monthly_evidence_review"
    return "keep_current_until_source_ready"


def _display_from_row(row: pd.Series, ticker: str) -> str:
    symbol = str(row.get("symbol") or ticker.split(".")[0]).strip()
    name = str(row.get("name") or symbol).strip()
    return f"{name}({symbol})"


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "是", "active"}


def _number(value: object) -> float:
    try:
        return float(str(value).replace(",", "").strip() or 0)
    except ValueError:
        return 0.0


def _default_required_evidence(pool: dict[str, Any]) -> tuple[str, ...]:
    pool_id = str(pool.get("pool_id") or "")
    if pool_id == "ai_theme_large_cap_v20260613":
        return (
            "AI主線受惠程度",
            "中大型市值或足夠成交金額",
            "基本面品質未明顯惡化",
            "題材強度仍高於一般大型權值股",
            "不因單日漲跌自動替換",
        )
    if pool_id == "large_core_bluechip_v0":
        return (
            "跨產業代表性",
            "低波動或回撤控制特性",
            "基本面品質未明顯惡化",
            "能代表防守或風格轉移",
            "不追逐每日題材熱度",
        )
    if pool_id == "tw50_dynamic_constituents_v0":
        return ("台灣50/0050成分股有效日期來源",)
    return ()
