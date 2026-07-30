"""Event-driven rechain preparation for the three exact whole-share paths.

The runner deliberately separates validation from performance reporting.  It
accepts only Radar's rebuilt current-path event package, never legacy holding
hits or adjusted-factor inferred events.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path


REQUIRED_EVENT_COLUMNS = {
    "strategy_id", "ticker", "share_class", "event_entitlement_date",
    "event_type", "event_status", "source_url", "source_hash",
}
REQUIRED_POSITION_COLUMNS = {
    "strategy_id", "date", "ticker", "share_class", "position_units", "cash",
}
STRATEGIES = {"v4d_best", "fixed7_S10_CD10", "00631L_S04_CD7"}


def read_csv(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def validate(position_path: Path, event_path: Path | None) -> dict:
    positions = read_csv(position_path)
    if not positions or not REQUIRED_POSITION_COLUMNS.issubset(positions[0]):
        raise ValueError("exact daily position authority is missing required columns")
    unknown = {row["strategy_id"] for row in positions} - STRATEGIES
    if unknown:
        raise ValueError(f"out-of-scope strategy ids: {sorted(unknown)}")
    result = {
        "position_rows": len(positions),
        "strategy_ids": sorted({row["strategy_id"] for row in positions}),
        "event_package_present": event_path is not None,
        "event_schema_valid": False,
        "ready_to_rechain": False,
    }
    if event_path is None:
        return result
    events = read_csv(event_path)
    if not events or not REQUIRED_EVENT_COLUMNS.issubset(events[0]):
        raise ValueError("current-path event package is missing required columns")
    if {row["strategy_id"] for row in events} - STRATEGIES:
        raise ValueError("event package contains out-of-scope strategy")
    result.update(event_rows=len(events), event_schema_valid=True, ready_to_rechain=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--position-ledger", required=True, type=Path)
    parser.add_argument("--event-package", type=Path)
    parser.add_argument("--blocker-ledger", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = validate(args.position_ledger, args.event_package)
    blocker_rows = read_csv(args.blocker_ledger) if args.blocker_ledger else []
    if blocker_rows:
        result["remaining_true_holder_blocked_rows"] = len(blocker_rows)
        result["ready_to_rechain"] = False
    result.update(
        task_id="TASK-BACKTEST-CORE-VNEXT-THREE-STRATEGY-EVENT-DRIVEN-RECHAIN-001",
        performance_executed=False,
        sheet_written=False,
        adjusted_factor_event_inference_used=False,
        legacy_holding_hit_used=False,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "rechain_readiness.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output / "current_step.txt").write_text("ready_for_current_path_event_package\n" if not result["ready_to_rechain"] else "event_schema_valid_rechain_execution_authorized\n", encoding="utf-8")


if __name__ == "__main__":
    main()
