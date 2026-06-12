from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


BUCKET_ACTIONABLE = "actionable"
BUCKET_WATCH = "watch"
BUCKET_EXCLUDED = "excluded"


@dataclass(frozen=True)
class FormalRadarCandidate:
    symbol: str
    name: str
    sector: str
    score: float
    bucket: str
    rank: int

    @property
    def ticker(self) -> str:
        return f"{self.symbol}.TW"

    @property
    def display(self) -> str:
        return f"{self.name}({self.symbol})"


def load_formal_radar_candidates(
    data_dir: str | Path,
    *,
    stock_metrics_filename: str = "stock_metrics.refreshed.csv",
) -> list[FormalRadarCandidate]:
    path = Path(data_dir) / stock_metrics_filename
    if not path.exists():
        raise FileNotFoundError(f"Formal radar stock metrics not found: {path}")
    frame = pd.read_csv(path, dtype={"symbol": str}).fillna("")
    scored = sorted(
        (_score_row(row) for _, row in frame.iterrows()),
        key=lambda candidate: (candidate.score, candidate.symbol),
        reverse=True,
    )
    actionable = [candidate for candidate in scored if candidate.bucket == BUCKET_ACTIONABLE]
    watch = [candidate for candidate in scored if candidate.bucket == BUCKET_WATCH]
    source = actionable if actionable else watch
    limit = 6 if actionable else 3
    return [
        FormalRadarCandidate(
            symbol=candidate.symbol,
            name=candidate.name,
            sector=candidate.sector,
            score=candidate.score,
            bucket=candidate.bucket,
            rank=index + 1,
        )
        for index, candidate in enumerate(source[:limit])
    ]


def formal_radar_candidates_to_symbols(candidates: list[FormalRadarCandidate]) -> list[dict[str, Any]]:
    return [
        {
            "ticker": candidate.ticker,
            "symbol": candidate.symbol,
            "name": candidate.name,
            "display": candidate.display,
            "asset_type": "stock",
            "source": "formal_radar_bucket",
            "theme": candidate.sector,
            "formal_bucket": candidate.bucket,
            "formal_score": candidate.score,
            "formal_rank": candidate.rank,
        }
        for candidate in candidates
    ]


@dataclass(frozen=True)
class _ScoredRow:
    symbol: str
    name: str
    sector: str
    score: float
    bucket: str


def _score_row(row: pd.Series) -> _ScoredRow:
    symbol = str(row.get("symbol") or "").strip()
    name = str(row.get("name") or symbol).strip() or symbol
    sector = str(row.get("sector") or "").strip()
    pullback = _clamp(_number(row.get("pullback_quality")))
    chip = _clamp(_number(row.get("chip_cleanliness")))
    valuation = _valuation_position_score(row)
    fundamental = _clamp(_fundamental_score(_number(row.get("revenue_yoy")), _number(row.get("revenue_mom"))))
    technical = _clamp(_number(row.get("technical_setup")))
    liquidity = _clamp(_number(row.get("liquidity")))
    risk_heat = _number(row.get("risk_heat"))
    margin_change = _number(row.get("margin_change_5d"))
    pe = _number(row.get("pe"))
    sector_pe_high = _number(row.get("sector_pe_high"))
    risk_penalty = 0.0
    if risk_heat > 65:
        risk_penalty += (risk_heat - 65) * 0.35
    if margin_change > 12:
        risk_penalty += min(10, (margin_change - 12) * 0.8)
    if sector_pe_high > 0 and pe > sector_pe_high * 0.95:
        risk_penalty += 7
    total = round(
        _clamp(
            (pullback * 0.20)
            + (chip * 0.20)
            + (valuation * 0.20)
            + (fundamental * 0.15)
            + (technical * 0.15)
            + (liquidity * 0.10)
            - risk_penalty
        ),
        1,
    )
    return _ScoredRow(
        symbol=symbol,
        name=name,
        sector=sector,
        score=total,
        bucket=_classify_stock(total, pullback, chip, valuation, liquidity, risk_heat),
    )


def _classify_stock(
    total: float,
    pullback_quality: float,
    chip_cleanliness: float,
    valuation_position: float,
    liquidity: float,
    risk_heat: float,
) -> str:
    if risk_heat >= 82 or liquidity < 45:
        return BUCKET_EXCLUDED
    if valuation_position < 25 and pullback_quality < 72:
        return BUCKET_EXCLUDED
    if total >= 72 and pullback_quality >= 70 and chip_cleanliness >= 68:
        return BUCKET_ACTIONABLE
    if total >= 58:
        return BUCKET_WATCH
    return BUCKET_EXCLUDED


def _valuation_position_score(row: pd.Series) -> float:
    pe = _number(row.get("pe"))
    low = _number(row.get("sector_pe_low"))
    avg = _number(row.get("sector_pe_avg"))
    high = _number(row.get("sector_pe_high"))
    if pe <= 0 or high <= low:
        return 45.0
    span = high - low
    percentile = (pe - low) / span * 100
    relative_to_avg = pe / avg if avg else 1.0
    low_range_bonus = 100 - percentile
    avg_discount_bonus = _clamp((1.25 - relative_to_avg) * 100)
    return _clamp(low_range_bonus * 0.65 + avg_discount_bonus * 0.35)


def _fundamental_score(revenue_yoy: float, revenue_mom: float) -> float:
    yoy_score = _clamp(50 + revenue_yoy * 1.2)
    mom_score = _clamp(50 + revenue_mom * 2.0)
    return yoy_score * 0.7 + mom_score * 0.3


def _number(value: object) -> float:
    try:
        return float(str(value).replace(",", "").strip() or 0)
    except ValueError:
        return 0.0


def _clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))
