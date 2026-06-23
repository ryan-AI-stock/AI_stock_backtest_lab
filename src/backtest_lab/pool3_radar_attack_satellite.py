from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from backtest_lab.decision_layers import SHADOW_OVERLAY


SATELLITE_ID = "pool3_radar_attack_satellite"
SATELLITE_LABEL = "Pool3 Radar Top10 攻擊衛星觀察"
SATELLITE_WORDING = "Pool3：00631L 主體，Radar Top10 攻擊衛星觀察"
OVERLAY_CANDIDATE_ID = "ma200_radar20_00631l80_else_top10"
OVERLAY_2024_RISK_NOTE = "2024 overlay 只小幅贏 0050，但最大回撤較深，不能直接升正式。"


@dataclass(frozen=True)
class RadarSatelliteCandidate:
    rank: int
    ticker: str
    symbol: str
    name: str
    theme: str
    score: float
    weight: float
    bucket: str = ""
    thesis: str = ""
    risk_reason: str = ""
    overlaps_pool1: bool = False
    overlaps_pool2: bool = False
    readiness: str = "partial"
    active_in_trade_decision: bool = False
    decision_layer: str = SHADOW_OVERLAY

    @property
    def display(self) -> str:
        return f"{self.name}({self.symbol})" if self.name else self.symbol

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["display"] = self.display
        return payload


def build_pool3_radar_attack_satellite(
    *,
    radar_data_dir: str | Path | None,
    signal_date: str,
    pool1_tickers: set[str] | None = None,
    pool2_tickers: set[str] | None = None,
    top_theme_count: int = 3,
    top_member_count: int = 10,
) -> dict[str, Any]:
    """Build a report-only Pool3 radar attack satellite block.

    This block deliberately stays outside formal three-pool voting. It is a
    shadow/report overlay used to observe whether Radar top themes contain
    attack satellite candidates.
    """

    pool1_tickers = pool1_tickers or set()
    pool2_tickers = pool2_tickers or set()
    data_dir = Path(radar_data_dir) if radar_data_dir else None
    base = {
        "satellite_id": SATELLITE_ID,
        "label": SATELLITE_LABEL,
        "wording": SATELLITE_WORDING,
        "signal_date": signal_date,
        "decision_layer": SHADOW_OVERLAY,
        "active_in_trade_decision": False,
        "valuation_used": False,
        "h3_used": False,
        "formal_vote_ready": False,
        "included_in_three_pool_vote": False,
        "pool3_formal_vote_target_unchanged": True,
        "overlay_candidate_id": OVERLAY_CANDIDATE_ID,
        "risk_notes": [
            OVERLAY_2024_RISK_NOTE,
            "Radar Top10 是籃子觀察，不是單一股票票。",
            "formal_top3 / date-aware theme membership readiness 若為 blocked 或 partial，不可 fail-open 成正式訊號。",
        ],
        "readiness": {
            "status": "blocked",
            "reason": "radar_data_dir_missing" if data_dir is None else "",
        },
        "top_themes": [],
        "candidates": [],
    }
    if data_dir is None or not data_dir.exists():
        return base

    top_themes, theme_source = _load_top_themes(data_dir, top_theme_count)
    readiness = _load_readiness(data_dir)
    base["readiness"] = {
        **readiness,
        "theme_source": str(theme_source) if theme_source else "",
        "status": _readiness_status(readiness, bool(top_themes)),
    }
    base["top_themes"] = top_themes
    if not top_themes:
        base["readiness"]["reason"] = "top_themes_unavailable"
        return base

    candidate_rows, candidate_source = _load_candidate_rows(data_dir, signal_date=signal_date)
    base["candidate_source"] = str(candidate_source) if candidate_source else ""
    if candidate_rows.empty:
        base["readiness"]["reason"] = "radar_candidates_unavailable"
        return base

    selected = candidate_rows[candidate_rows["theme"].isin([item["theme"] for item in top_themes])].copy()
    if selected.empty:
        base["readiness"]["reason"] = "no_candidates_in_top3_themes"
        return base

    selected["_rank_score"] = pd.to_numeric(selected["score"], errors="coerce").fillna(0.0)
    selected["_symbol"] = selected["symbol"].astype(str).str.strip()
    selected = selected.sort_values(["_rank_score", "_symbol"], ascending=[False, True]).head(top_member_count)
    total_score = float(selected["_rank_score"].clip(lower=0).sum())
    candidates: list[RadarSatelliteCandidate] = []
    for rank, (_, row) in enumerate(selected.iterrows(), start=1):
        symbol = str(row.get("symbol") or "").strip()
        ticker = _to_ticker(symbol)
        score = float(row.get("_rank_score") or 0.0)
        weight = (score / total_score) if total_score > 0 else (1.0 / len(selected))
        candidates.append(
            RadarSatelliteCandidate(
                rank=rank,
                ticker=ticker,
                symbol=symbol,
                name=str(row.get("name") or symbol).strip(),
                theme=str(row.get("theme") or "").strip(),
                score=round(score, 6),
                weight=round(weight, 6),
                bucket=str(row.get("bucket") or row.get("bucket_key") or "").strip(),
                thesis=str(row.get("thesis") or "").strip(),
                risk_reason=str(row.get("risk_reason") or "").strip(),
                overlaps_pool1=ticker in pool1_tickers,
                overlaps_pool2=ticker in pool2_tickers,
                readiness=str(base["readiness"].get("status") or "partial"),
            )
        )
    base["candidates"] = [item.to_dict() for item in candidates]
    base["overlap_summary"] = {
        "pool1_overlap_count": sum(1 for item in candidates if item.overlaps_pool1),
        "pool2_overlap_count": sum(1 for item in candidates if item.overlaps_pool2),
    }
    return base


def write_pool3_radar_attack_satellite_outputs(output_dir: Path, satellite: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "pool3_radar_attack_satellite.json").write_text(
        json.dumps(satellite, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame(satellite.get("candidates") or []).to_csv(
        output_dir / "pool3_radar_attack_satellite.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (output_dir / "pool3_radar_attack_satellite.md").write_text(
        markdown_pool3_radar_attack_satellite(satellite),
        encoding="utf-8",
    )


def markdown_pool3_radar_attack_satellite(satellite: dict[str, Any]) -> str:
    lines = [
        "# Pool3 Radar Top10 攻擊衛星觀察",
        "",
        f"- 語意：{satellite.get('wording', SATELLITE_WORDING)}",
        "- 決策層：shadow/report-only，不納入三池正式投票。",
        f"- active_in_trade_decision：{str(satellite.get('active_in_trade_decision')).lower()}",
        f"- valuation_used：{str(satellite.get('valuation_used')).lower()}",
        f"- h3_used：{str(satellite.get('h3_used')).lower()}",
        f"- readiness：{(satellite.get('readiness') or {}).get('status', 'unknown')}",
        "",
        "## 風險與邊界",
    ]
    for note in satellite.get("risk_notes") or []:
        lines.append(f"- {note}")
    lines.extend(["", "## Top10", "", "| Rank | 標的 | Theme | Score | Weight | Pool1 overlap | Pool2 overlap |", "| --- | --- | --- | ---: | ---: | --- | --- |"])
    for row in satellite.get("candidates") or []:
        lines.append(
            "| {rank} | {display} | {theme} | {score:.2f} | {weight:.1%} | {p1} | {p2} |".format(
                rank=row.get("rank", ""),
                display=row.get("display") or row.get("ticker") or "",
                theme=row.get("theme", ""),
                score=float(row.get("score") or 0.0),
                weight=float(row.get("weight") or 0.0),
                p1="Y" if row.get("overlaps_pool1") else "N",
                p2="Y" if row.get("overlaps_pool2") else "N",
            )
        )
    return "\n".join(lines)


def _load_top_themes(data_dir: Path, top_theme_count: int) -> tuple[list[dict[str, Any]], Path | None]:
    for filename in ("sector_metrics.refreshed.csv", "sector_metrics.csv"):
        path = data_dir / filename
        if not path.exists():
            continue
        frame = pd.read_csv(path).fillna("")
        if "name" not in frame.columns:
            continue
        for col in ("capital_inflow_rank", "capital_share", "turnover_share_change", "momentum_20d", "risk_heat"):
            if col in frame.columns:
                frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0.0)
        sort_columns = [col for col in ("capital_inflow_rank", "capital_share", "turnover_share_change") if col in frame.columns]
        if sort_columns:
            frame = frame.sort_values(sort_columns, ascending=[False] * len(sort_columns))
        themes = []
        for rank, (_, row) in enumerate(frame.head(top_theme_count).iterrows(), start=1):
            themes.append(
                {
                    "rank": rank,
                    "theme": str(row.get("name") or "").strip(),
                    "theme_description": str(row.get("theme") or "").strip(),
                    "capital_inflow_rank": _number(row.get("capital_inflow_rank")),
                    "capital_share": _number(row.get("capital_share")),
                    "turnover_share_change": _number(row.get("turnover_share_change")),
                    "momentum_20d": _number(row.get("momentum_20d")),
                    "risk_heat": _number(row.get("risk_heat")),
                }
            )
        return themes, path
    return [], None


def _load_candidate_rows(data_dir: Path, *, signal_date: str) -> tuple[pd.DataFrame, Path | None]:
    candidate_path = data_dir / "formal_radar_candidates.latest.csv"
    if candidate_path.exists():
        frame = pd.read_csv(candidate_path, dtype={"symbol": str}).fillna("")
        report_dates = {str(value).strip() for value in frame.get("report_date", []) if str(value).strip()}
        if signal_date in report_dates:
            frame = frame[frame["report_date"].astype(str).str.strip() == signal_date]
        return _normalize_candidate_frame(frame), candidate_path

    stock_metrics_path = data_dir / "stock_metrics.refreshed.csv"
    if stock_metrics_path.exists():
        frame = pd.read_csv(stock_metrics_path, dtype={"symbol": str}).fillna("")
        return _normalize_candidate_frame(frame), stock_metrics_path
    return pd.DataFrame(), None


def _normalize_candidate_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["symbol", "name", "theme", "score", "bucket", "thesis", "risk_reason"])
    normalized = frame.copy()
    if "theme" not in normalized.columns and "sector" in normalized.columns:
        normalized["theme"] = normalized["sector"]
    if "score" not in normalized.columns:
        score_parts = [
            pd.to_numeric(normalized.get(col, 0), errors="coerce").fillna(0.0)
            for col in ("pullback_quality", "chip_cleanliness", "technical_setup", "liquidity")
            if col in normalized.columns
        ]
        normalized["score"] = sum(score_parts) / len(score_parts) if score_parts else 0.0
    if "bucket" not in normalized.columns and "bucket_key" in normalized.columns:
        normalized["bucket"] = normalized["bucket_key"]
    for col in ("symbol", "name", "theme", "score", "bucket", "thesis", "risk_reason"):
        if col not in normalized.columns:
            normalized[col] = ""
    return normalized[["symbol", "name", "theme", "score", "bucket", "thesis", "risk_reason"]].copy()


def _load_readiness(data_dir: Path) -> dict[str, Any]:
    readiness: dict[str, Any] = {
        "formal_top3_ready": None,
        "theme_membership_v2_ready": None,
        "cache_only": False,
        "blocking_issues": [],
        "warnings": [],
    }
    top3_path = data_dir / "history_replay" / "formal_top3_capital_flow_2022_2023" / "top3_capital_flow_readiness.json"
    if top3_path.exists():
        payload = _read_json(top3_path)
        readiness["formal_top3_ready"] = bool(payload.get("ready"))
        readiness["formal_top3_source_mode"] = payload.get("source_mode", "")
        readiness["formal_top3_blocking_issues"] = payload.get("blocking_issues", [])
        readiness["blocking_issues"].extend(payload.get("blocking_issues", []))
        readiness["warnings"].extend(payload.get("warnings", []))
    theme_path = data_dir / "formal_sources" / "date_aware_theme_membership_v2_readiness.json"
    if theme_path.exists():
        payload = _read_json(theme_path)
        readiness["theme_membership_v2_ready"] = bool(payload.get("ready"))
        readiness["theme_membership_v2_formal_top3_status"] = payload.get("formal_top3_status", "")
        readiness["theme_membership_v2_accepted_rows"] = payload.get("accepted_evidence_row_count", 0)
        readiness["theme_membership_v2_usable_rows"] = payload.get("usable_for_formal_replay_count", 0)
        readiness["blocking_issues"].extend(payload.get("blocking_issues", []))
        readiness["warnings"].extend(payload.get("warnings", []))
    if readiness["formal_top3_ready"] is False or readiness["theme_membership_v2_ready"] is False:
        readiness["cache_only"] = True
    return readiness


def _readiness_status(readiness: dict[str, Any], has_top_themes: bool) -> str:
    if not has_top_themes:
        return "blocked"
    if readiness.get("formal_top3_ready") is False or readiness.get("theme_membership_v2_ready") is False:
        return "partial"
    if readiness.get("formal_top3_ready") is True and readiness.get("theme_membership_v2_ready") is True:
        return "ready"
    return "partial"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _to_ticker(symbol: str) -> str:
    text = str(symbol).strip()
    if "." in text:
        return text
    return f"{text}.TW"


def _number(value: object) -> float:
    try:
        return float(str(value).replace(",", "").strip() or 0)
    except ValueError:
        return 0.0
