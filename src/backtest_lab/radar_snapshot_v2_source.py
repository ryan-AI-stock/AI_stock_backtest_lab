from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from backtest_lab.radar_snapshot_readiness import REQUIRED_SNAPSHOT_COLUMNS


DEFAULT_ALLOWED_BUCKETS = frozenset(
    {
        "theme_leader",
        "theme_second_line",
        "theme_laggard_rebound",
    }
)


@dataclass(frozen=True)
class RadarSnapshotCandidateSet:
    signal_date: pd.Timestamp
    snapshot_date: pd.Timestamp
    rows: pd.DataFrame


def load_radar_snapshot_history(snapshot_dir: str | Path) -> pd.DataFrame:
    files = sorted(Path(snapshot_dir).glob("radar_snapshot_*.csv"))
    if not files:
        raise FileNotFoundError(f"No radar_snapshot_*.csv files found in {snapshot_dir}")

    frames: list[pd.DataFrame] = []
    for path in files:
        frame = pd.read_csv(path, dtype=str).fillna("")
        missing = REQUIRED_SNAPSHOT_COLUMNS - set(frame.columns)
        if missing:
            raise ValueError(f"{path.name} missing columns: {', '.join(sorted(missing))}")
        frame["_source_file"] = path.name
        frames.append(frame)

    history = pd.concat(frames, ignore_index=True)
    history["date"] = pd.to_datetime(history["date"], errors="coerce").dt.normalize()
    history["fundamental_source_date"] = pd.to_datetime(
        history["fundamental_source_date"].replace("", pd.NA),
        errors="coerce",
    ).dt.normalize()
    for column in (
        "theme_rank",
        "theme_score",
        "stock_score",
        "fundamental_score",
        "risk_heat",
        "stock_turnover_rank_in_theme",
        "stock_turnover_share_in_theme",
        "turnover_value",
        "capital_share",
    ):
        history[column] = pd.to_numeric(history[column], errors="coerce")
    for column in (
        "fundamental_pass",
        "theme_leader_flag",
        "theme_second_line_flag",
        "theme_laggard_rebound_flag",
        "overheated_flag",
    ):
        history[column] = history[column].map(_truthy)
    return history


def select_radar_snapshot_candidates(
    history: pd.DataFrame,
    signal_date: str | pd.Timestamp,
    *,
    allowed_buckets: set[str] | frozenset[str] = DEFAULT_ALLOWED_BUCKETS,
    top_n: int | None = None,
) -> RadarSnapshotCandidateSet:
    signal_ts = pd.Timestamp(signal_date).normalize()
    valid_snapshot_dates = history.loc[history["date"] <= signal_ts, "date"].dropna().sort_values()
    if valid_snapshot_dates.empty:
        raise ValueError(f"No radar snapshot available on or before {signal_ts.date()}.")

    snapshot_ts = pd.Timestamp(valid_snapshot_dates.iloc[-1]).normalize()
    rows = history.loc[history["date"] == snapshot_ts].copy()
    future_source = rows["fundamental_source_date"].notna() & (rows["fundamental_source_date"] > snapshot_ts)
    if bool(future_source.any()):
        count = int(future_source.sum())
        raise ValueError(f"{count} rows use future fundamental data in snapshot {snapshot_ts.date()}.")

    candidates = rows.loc[
        rows["fundamental_pass"]
        & ~rows["overheated_flag"]
        & rows["bucket"].isin(set(allowed_buckets))
    ].copy()
    candidates = candidates.sort_values(
        by=[
            "theme_rank",
            "theme_score",
            "stock_score",
            "fundamental_score",
            "stock_turnover_share_in_theme",
        ],
        ascending=[True, False, False, False, False],
        na_position="last",
    ).reset_index(drop=True)
    candidates["candidate_rank"] = candidates.index + 1
    if top_n is not None:
        candidates = candidates.head(top_n).copy()
    return RadarSnapshotCandidateSet(signal_date=signal_ts, snapshot_date=snapshot_ts, rows=candidates)


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}
