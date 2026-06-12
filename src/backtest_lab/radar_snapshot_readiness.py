from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd


REQUIRED_SNAPSHOT_COLUMNS = {
    "date",
    "theme",
    "symbol",
    "name",
    "theme_rank",
    "theme_score",
    "capital_share",
    "turnover_value",
    "stock_score",
    "bucket",
    "fundamental_pass",
    "fundamental_score",
    "fundamental_data_status",
    "fundamental_source_date",
    "risk_heat",
    "liquidity",
    "stock_turnover_rank_in_theme",
    "stock_turnover_share_in_theme",
    "theme_leader_flag",
    "theme_second_line_flag",
    "theme_laggard_rebound_flag",
    "overheated_flag",
}


@dataclass(frozen=True)
class RadarSnapshotReadiness:
    ready: bool
    snapshot_count: int
    total_rows: int
    dates_with_fundamental_pass: int
    total_fundamental_pass_rows: int
    average_fundamental_pass_ratio: float
    missing_columns: list[str]
    warnings: list[str]


def evaluate_radar_snapshot_readiness(
    snapshot_dir: str | Path,
    *,
    min_snapshots: int = 20,
    min_dates_with_fundamental_pass: int = 5,
    min_average_fundamental_pass_ratio: float = 0.03,
) -> RadarSnapshotReadiness:
    files = sorted(Path(snapshot_dir).glob("radar_snapshot_*.csv"))
    warnings: list[str] = []
    if not files:
        return RadarSnapshotReadiness(
            ready=False,
            snapshot_count=0,
            total_rows=0,
            dates_with_fundamental_pass=0,
            total_fundamental_pass_rows=0,
            average_fundamental_pass_ratio=0.0,
            missing_columns=sorted(REQUIRED_SNAPSHOT_COLUMNS),
            warnings=["No radar_snapshot_*.csv files found."],
        )

    frames = []
    missing_columns: set[str] = set()
    for path in files:
        frame = pd.read_csv(path, dtype=str).fillna("")
        missing_columns.update(REQUIRED_SNAPSHOT_COLUMNS - set(frame.columns))
        frame["_source_file"] = path.name
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)

    if missing_columns:
        warnings.append(f"Missing required columns: {', '.join(sorted(missing_columns))}")
    if len(files) < min_snapshots:
        warnings.append(f"Only {len(files)} snapshots found; expected at least {min_snapshots}.")

    if "fundamental_pass" in combined.columns:
        pass_flags = combined["fundamental_pass"].astype(str).str.lower().isin({"true", "1", "yes"})
    else:
        pass_flags = pd.Series([False] * len(combined))
    if {"date", "fundamental_source_date"}.issubset(combined.columns):
        invalid_source_dates = _future_source_date_rows(combined)
        if invalid_source_dates:
            warnings.append(f"{invalid_source_dates} rows have fundamental_source_date later than snapshot date.")
    elif "fundamental_source_date" not in combined.columns:
        warnings.append("Missing fundamental_source_date column; cannot verify no future fundamental leakage.")
    rows_by_file = combined.groupby("_source_file").size()
    pass_by_file = combined.loc[pass_flags].groupby("_source_file").size()
    pass_ratio_by_file = (pass_by_file / rows_by_file).fillna(0.0)
    dates_with_pass = int((pass_by_file > 0).sum())
    total_pass = int(pass_flags.sum())
    average_pass_ratio = float(pass_ratio_by_file.mean()) if not pass_ratio_by_file.empty else 0.0

    if dates_with_pass < min_dates_with_fundamental_pass:
        warnings.append(
            f"Only {dates_with_pass} snapshots have any fundamental_pass rows; "
            f"expected at least {min_dates_with_fundamental_pass}."
        )
    if average_pass_ratio < min_average_fundamental_pass_ratio:
        warnings.append(
            f"Average fundamental pass ratio is {average_pass_ratio:.2%}; "
            f"expected at least {min_average_fundamental_pass_ratio:.2%}."
        )

    ready = not missing_columns and len(files) >= min_snapshots and dates_with_pass >= min_dates_with_fundamental_pass
    ready = ready and average_pass_ratio >= min_average_fundamental_pass_ratio
    if any("fundamental_source_date later" in warning for warning in warnings):
        ready = False
    return RadarSnapshotReadiness(
        ready=ready,
        snapshot_count=len(files),
        total_rows=int(len(combined)),
        dates_with_fundamental_pass=dates_with_pass,
        total_fundamental_pass_rows=total_pass,
        average_fundamental_pass_ratio=average_pass_ratio,
        missing_columns=sorted(missing_columns),
        warnings=warnings,
    )


def _future_source_date_rows(frame: pd.DataFrame) -> int:
    snapshot_dates = pd.to_datetime(frame["date"], errors="coerce")
    source_dates = pd.to_datetime(frame["fundamental_source_date"].replace("", pd.NA), errors="coerce")
    valid = source_dates.notna() & snapshot_dates.notna()
    return int((source_dates[valid] > snapshot_dates[valid]).sum())


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate AI_stock_rotation_radar snapshots before backtesting.")
    parser.add_argument("--snapshot-dir", required=True)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    readiness = evaluate_radar_snapshot_readiness(args.snapshot_dir)
    payload = asdict(readiness)
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
